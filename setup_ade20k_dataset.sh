#!/bin/bash
# 自動下載和準備 ADE20K-150 數據集

set -e  # 遇到錯誤就停止

echo "========================================"
echo "ADE20K-150 數據集下載和準備"
echo "========================================"

# 設定路徑
DATASET_ROOT="/tmp2/b12902107/datasets"
ADE20K_DIR="$DATASET_ROOT/ADEChallengeData2016"
HCLIP_DIR="/tmp2/b12902107/cv_final"

# 創建數據集目錄
mkdir -p $DATASET_ROOT
cd $DATASET_ROOT

# 檢查是否已經下載
if [ -d "$ADE20K_DIR" ] && [ -f "$ADE20K_DIR/images/validation/ADE_val_00000001.jpg" ]; then
    echo "✓ ADE20K 數據集已存在，跳過下載"
else
    echo "Step 1/4: 下載 ADE20K 數據集 (約 923MB)..."
    if [ ! -f "ADEChallengeData2016.zip" ]; then
        wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    else
        echo "  壓縮檔已存在，跳過下載"
    fi
    
    echo ""
    echo "Step 2/4: 解壓數據集..."
    unzip -q ADEChallengeData2016.zip
    echo "✓ 解壓完成"
fi

# 檢查數據集結構
echo ""
echo "Step 3/4: 檢查數據集結構..."
if [ ! -d "$ADE20K_DIR/images/validation" ]; then
    echo "✗ 錯誤：找不到 images/validation 目錄"
    exit 1
fi

if [ ! -d "$ADE20K_DIR/annotations/validation" ]; then
    echo "✗ 錯誤：找不到 annotations/validation 目錄"
    exit 1
fi

# 統計檔案數量
IMG_COUNT=$(ls -1 $ADE20K_DIR/images/validation/*.jpg 2>/dev/null | wc -l)
ANN_COUNT=$(ls -1 $ADE20K_DIR/annotations/validation/*.png 2>/dev/null | wc -l)

echo "  找到 $IMG_COUNT 張驗證圖片"
echo "  找到 $ANN_COUNT 個標註檔案"

if [ "$IMG_COUNT" -lt 2000 ] || [ "$ANN_COUNT" -lt 2000 ]; then
    echo "✗ 警告：檔案數量似乎不完整"
fi

# 準備 Detectron2 格式的標註
echo ""
echo "Step 4/4: 準備 Detectron2 格式的標註..."

# 設定環境變數
export DETECTRON2_DATASETS=$DATASET_ROOT

# 執行準備腳本
cd $HCLIP_DIR/datasets
python prepare_ade20k_150.py

# 檢查轉換結果
if [ -d "$ADE20K_DIR/annotations_detectron2/validation" ]; then
    CONV_COUNT=$(ls -1 $ADE20K_DIR/annotations_detectron2/validation/*.png 2>/dev/null | wc -l)
    echo "  轉換了 $CONV_COUNT 個標註檔案"
    if [ "$CONV_COUNT" -eq "$ANN_COUNT" ]; then
        echo "✓ 標註轉換完成"
    else
        echo "✗ 警告：轉換的檔案數量不匹配"
    fi
else
    echo "✗ 錯誤：標註轉換失敗"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ 數據集準備完成！"
echo "========================================"
echo ""
echo "數據集位置：$ADE20K_DIR"
echo "  - 圖片：$ADE20K_DIR/images/validation/ ($IMG_COUNT 張)"
echo "  - 標註：$ADE20K_DIR/annotations_detectron2/validation/ ($CONV_COUNT 個)"
echo ""
echo "環境變數設定："
echo "  export DETECTRON2_DATASETS=$DATASET_ROOT"
echo ""
echo "現在可以執行訓練了："
echo "  cd $HCLIP_DIR"
echo "  export DETECTRON2_DATASETS=$DATASET_ROOT"
echo "  bash run_global_ade150.sh"
echo "========================================"
