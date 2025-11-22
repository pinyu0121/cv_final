#!/bin/bash
# 只做 ADE20K-150 評估和視覺化（不訓練）
# 需要預訓練模型

set -e

cd /tmp2/b12902107/cv_final

# 設定環境變數
export DETECTRON2_DATASETS=/tmp2/b12902107/datasets
export CUDA_VISIBLE_DEVICES=1

echo "========================================"
echo "ADE20K-150 評估和視覺化（僅評估模式）"
echo "========================================"

# 檢查預訓練模型
if [ ! -f "pretrained_models/model_final.pth" ]; then
    echo "❌ 錯誤：找不到預訓練模型"
    echo ""
    echo "請將預訓練模型放在以下位置之一："
    echo "  1. pretrained_models/model_final.pth"
    echo "  2. 或指定模型路徑：bash $0 /path/to/model.pth"
    echo ""
    echo "或者下載 COCO 數據集進行完整訓練"
    exit 1
fi

MODEL_PATH=${1:-"pretrained_models/model_final.pth"}
OUTPUT_DIR="output/ade150_experiment"

echo "模型路徑: $MODEL_PATH"
echo "輸出目錄: $OUTPUT_DIR"
echo "使用 GPU: 1"
echo "========================================"

# 評估
echo ""
echo "Step 1/2: 在 ADE20K-150 上評估..."
python train_net.py --config configs/vitb_384_oft.yaml \
    --num-gpus 1 \
    --dist-url "auto" \
    --eval-only \
    OUTPUT_DIR $OUTPUT_DIR \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
    DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS $MODEL_PATH 

# 視覺化
# echo ""
# echo "Step 2/2: 生成視覺化..."
# mkdir -p $OUTPUT_DIR/visualizations

# python demo/visualization_ade150.py \
#     --config-file configs/vitb_384_oft.yaml \
#     --input $DETECTRON2_DATASETS/ADEChallengeData2016/images/validation \
#     --output $OUTPUT_DIR/visualizations \
#     --opts \
#         MODEL.WEIGHTS $MODEL_PATH \
#         MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
#         TEST.SLIDING_WINDOW "True" \
#         MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#         DATASETS.TEST "('ade20k_150_test_sem_seg',)"

echo ""
echo "========================================"
echo "✓ 完成！結果在："
echo "  - 評估: $OUTPUT_DIR/log.txt"
echo "  - 視覺化: $OUTPUT_DIR/visualizations/"
echo "========================================"
