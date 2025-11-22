# CVPDL FINAL PROJECT


### Installation and Data Preparation

This project uses micromamba as its conda environment provider. 
Please run the following command after you created a conda environment.
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Run the following command to get detectron2_full
```
git submodule add https://github.com/facebookresearch/detectron2.git detectron2_full
```

Expected Architecture: Please put ```detectron2_full``` outside of ```cv_final```.
```
.
├── cv_final(the whole thing you cloned)
└── detectron2_full
```

### Dataset setup
Check the file, as there might be several data paths that you should modify. 
```
bash setup_ade20k_dataset.sh
```
###  Evaluation & Inference
Please add the ```model_final.pth``` to a folder ```pretrained_models``` under ```cv_final``` to run the following command nice and clean. If you want to modify the path, please ensure```eval_ade150_only.sh``` is also modified.

This runs the baseline.
```bash
bash eval_ade150_only.sh
```

### Current Status
✅ Baseline Model Training & Evaluation -> Reached mIoU 29.92%, training batch = 1 & inference batch = 4 

✅ Added ```clip_refine.py``` as the modification of inference part -> It doesn't interfere with the baseline evaluation, so you can ignore it by now.

🔴 Fix visulization part in ```eval_only_ade150.sh```

🔴 Fix the code so that it could run inference with refined logits.


