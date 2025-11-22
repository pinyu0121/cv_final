# cat_seg/modeling/clip_refine.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def _connected_components(mask_np):
    """
    mask_np: 2D numpy array, dtype=bool
    return:
        labeled: same shape, each connected component has an int label (1..num)
        num: number of components
    """
    labeled, num = ndimage.label(mask_np)
    return labeled, num


@torch.no_grad()
def clip_refine_postprocess(
    logits: torch.Tensor,      # [B, C, H, W]
    pixel_feats: torch.Tensor, # [B, D, H, W]
    text_feats: torch.Tensor,  # [C, D]
    score_thresh: float = 0.3,
    low_factor: float = 0.1,
    base: float = 1.0,
    gain: float = 0.0,
) -> torch.Tensor:
    """
    Training-free region-level re-ranking.

    Args:
        logits:      [B, C, H, W] raw logits from H-CLIP head
        pixel_feats: [B, D, H, W] pixel-level feature (e.g. CLIP ViT dense feat)
        text_feats:  [C, D] class text embeddings (same D as pixel_feats)
        score_thresh: threshold τ. Regions with score < τ will be strongly suppressed.
        low_factor:   γ. Scalar factor for low-score regions (e.g. 0.1).
        base, gain:   scaling for high-score regions: scale = base + gain * score.
                      If you just want to keep logits unchanged, use base=1.0, gain=0.0.

    Returns:
        refined_logits: [B, C, H, W]
    """
    device = logits.device
    B, C, H, W = logits.shape
    _, D, Hf, Wf = pixel_feats.shape

    # 確保 pixel_feats 跟 logits 同一解析度
    if (Hf, Wf) != (H, W):
        pixel_feats = F.interpolate(pixel_feats, size=(H, W), mode="bilinear", align_corners=False)

    # L2-normalize text & pixel feats for cosine
    text_feats = F.normalize(text_feats.to(device), dim=-1)  # [C, D]
    pixel_feats = F.normalize(pixel_feats, dim=1)            # [B, D, H, W]

    refined_logits = logits.clone()

    # 逐 batch, 逐 class 處理
    for b in range(B):
        # 取出該張圖的 pixel feature: [D, H, W]
        feat_b = pixel_feats[b]  # [D, H, W]

        for c in range(C):
            # 1) 取出該 class 的 logits map
            logit_map = logits[b, c]  # [H, W]

            # 2) threshold → binary mask (候選區域)
            # 這裡 threshold 可以根據你實驗再調，先用 0 當 logit 門檻
            mask = (logit_map > 0)

            if mask.sum() == 0:
                continue  # 沒有區域就略過

            # 3) connected components on CPU with scipy
            mask_np = mask.detach().cpu().numpy().astype(np.bool_)
            labeled, num = _connected_components(mask_np)
            if num == 0:
                continue

            # 4) 對每一個 component 計算 region embedding & score
            for region_id in range(1, num + 1):
                region_mask = (labeled == region_id)  # numpy bool [H, W]
                idx = torch.from_numpy(region_mask).to(device)  # [H, W] bool

                area = idx.sum().item()
                if area == 0:
                    continue

                # region pooling: 平均 pixel_feats 上這個區域
                # feat_b: [D, H, W] → masked: [D, area]
                region_feat = feat_b[:, idx].mean(dim=-1)  # [D]
                region_feat = F.normalize(region_feat, dim=0)

                # text_feats[c]: [D]
                txt = text_feats[c]  # [D]

                # cosine similarity
                sim = torch.dot(region_feat, txt)  # scalar in [-1, 1]
                score = (sim + 1.0) / 2.0         # 映射到 [0, 1]

                # 5) 根據 score reweight logits
                if score < score_thresh:
                    # 低分區域：整塊壓低
                    refined_logits[b, c][idx] *= low_factor
                else:
                    # 高分區域：線性縮放
                    scale = base + gain * score
                    refined_logits[b, c][idx] *= scale

    return refined_logits
