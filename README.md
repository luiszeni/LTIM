# Source Code for LTIM 
## Weakly Supervised Object Detection with Long-Term Instance Mining
**Abstract:** Weakly supervised object detection (WSOD) addresses the challenge of using only image-level labels to train models to recognize and locate different objects in images. Multiple Instance Learning (MIL) is a common strategy in WSOD, but MIL-based approaches struggle to detect several instances of the same class in an image. This paper introduces a novel instance discovery module designed to identify additional instances for supervision in a baseline MIL-based approach. Our module builds a memory bank at training time containing several instances of the desired categories, and employs a MultiLayer Perceptron (MLP) to project the features related to these instances onto a space where same-class instances are close while different-class instances are distant based on a discriminative loss function. We also propose a proposal overlap suppression strategy to select the best instances when the module detects multiple proposals belonging to the same object. Our method achieves the best mAP metrics among the state-of-the-art for VOC2007 and VOC2012 datasets, and the second-best AP50 results for the COCO dataset.

Note: the paper is under review on IEEE Transactions on Image Processing. Once it is published we will update it's link here.

## Environment setup

* [Python 3.7](https://pytorch.org)
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)
```bash
git clone https://github.com/luiszeni/LTIM
cd LTIM

conda create --name LTIM python=3.8.16
conda activate LTIM

pip install ninja yacs cython matplotlib tqdm opencv-python  pycocotools
pip install torch==2.0 torchvision==0.15.1 

pip install git+https://github.com/facebookresearch/fvcore.git

python setup.py build develop
```

## Datasets Configuration
* [PASCAL VOC (2007, 2012)](http://host.robots.ox.ac.uk/pascal/VOC/)
* [MS-COCO (2014, 2017)](https://cocodataset.org/#download)  
```bash
mkdir -p datasets/{coco/voc}
    datasets/
    ├── voc/
    │   ├── VOC2007
    │   │   ├── Annotations/
    │   │   ├── JPEGImages/
    │   │   ├── ...
    │   ├── VOC2012/
    │   │   ├── ...
    ├── coco/
    │   ├── annotations/
    │   ├── train2014/
    │   ├── val2014/
    │   ├── train2017/
    │   ├── ...
    ├── ...
```

## Proposals Configuration
Download .pkl file from [Dropbox](https://www.dropbox.com/sh/sprm4dxg7l22jrg/AAD0kBctuRnCg_rlZHzEBemQa?dl=0)
```bash
mkdir proposal
    proposal/
    ├── SS/
    │   ├── voc
    │   │   ├── SS-voc07_trainval.pkl/
    │   │   ├── SS-voc07_test.pkl/
    │   │   ├── ...
    ├── MCG/
    │   ├── voc
    │   │   ├── ...
    │   ├── coco
    │   │   ├── MCG-coco_2014_train_boxes.pkl/
    │   │   ├── ...
    ├── ...
```

## Training a LTIM model
Template:
```bash
python tools/train_net.py --config-file configs/{config_file}.yaml
                          OUTPUT_DIR {output_dir}
                          nms {nms threshold}
                          lmda {lambda value}
                          iou {iou threshold}
                          temp {temperature}
```
Example:
```bash
python3 tools/train_net.py  --config-file configs/voc/voc07_contra_db_b8_lr0.01_mcg.yaml 
                            OUTPUT_DIR ltim_voc_07  
                            nms 0.1  
                            lmda 0.03  
                            iou 0.5 
                            temp 0.2
```

## Testing a LTIM model
Template:
```bash
python tools/test_net.py --config-file configs/{config_file}.yaml
                          TEST.IMS_PER_BATCH 8 
                          OUTPUT_DIR {output_dir} 
                          MODEL.WEIGHT {model_weight}.pth
```
Example:
```bash
python tools/test_net.py --config-file configs/voc07_contra_db_b8_lr0.01_mcg.yaml
                          TEST.IMS_PER_BATCH 8 
                          OUTPUT_DIR ltim_voc_07 
                          MODEL.WEIGHT ltim_voc_07/model_final.pth
```


## On Giant's Shoulders
The primary codebase of our project was adapted from the repositories <a href="https://github.com/NVlabs/wetectron">wetectron</a> and <a href="https://github.com/jinhseo/OD-WSCL">OD-WSCL</a>. 
```BibTex
@inproceedings{ren_wetectron_2020,
  title = {Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection},
  author = {Zhongzheng Ren and Zhiding Yu and Xiaodong Yang and Ming-Yu Liu and Yong Jae Lee and Alexander G. Schwing and Jan Kautz},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}

@inproceedings{seo_od-wscl_2022,
  title={Object discovery via contrastive learning for weakly supervised object detection},
  author={Seo, Jinhwan and Bae, Wonho and Sutherland, Danica J and Noh, Junhyug and Kim, Daijin},
  booktitle={European Conference on Computer Vision},
  pages={312--329},
  year={2022},
  organization={Springer}
}
```
We would like to express our gratitude to the authors for generously providing access to their implementation.

## Cite our Work
If our work contribute to your research, we kindly ask that you cite it:
```BibTex
@inproceedings{zeni_ltim_2025,
      Our paper is in the publication review process. Once published, we will update here. 
}
```