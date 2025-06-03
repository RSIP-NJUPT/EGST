# EGSTNet: Edge-Guided Structural Transformer Network for Semantic Segmentation Based on SAR Time Series (under review)

Kang Ni*, Chunyang Yuan*, Qichao Liu, Zhizhong Zheng, Peng Wang

*: Kang Ni and Chunyang Yuan contributed equally to this work
## Abstract
Real-time and accurate interpretation of time-series SAR imagery is vital for land-cover monitoring. Nevertheless, the presence of speckle noise makes it extremely challenging to distill the essential information from redundant SAR time series. 
Current methods not only struggle to suppress temporal-feature redundancy but also tend to overlook the structural cues along land-cover boundaries that are crucial for boosting classification performance, leading to slower inference and sub-optimal accuracy.
To address these challenges, we propose Edge-Guided Structural Transformer Network (EGSTNet), a lightweight and efficient semantic segmentation network for time-series SAR imagery. EGSTNet comprises three key modules: edge-guided convolution (EGConv), axial separable large kernel attention (ASLKA), and implicit semantic graph Transformer (ISGT). EGConv and ASLKA, based on a lightweight design, utilize a re-parameterizable multi-branch convolutional module and a large-kernel attention mechanism to capture multi-order gradient features and model global spatiotemporal characteristics along both spatial and temporal dimensions in SAR time series. Notably, an implicit semantic graph style is incorporated into ISGT to facilitate efficient joint modeling of spatiotemporal and structural information of land covers. Experimental results on three benchmark time-series SAR semantic segmentation datasets demonstrate that EGSTNet outperforms other related methods in classification accuracy while maintaining a smaller model parameter and faster inference speed, reaching up to 105 FPS. Code is available at [https://github.com/RSIP-NJUPT/EGST](https://github.com/RSIP-NJUPT/EGST).


## News
- 2025/06/03. We completed training and inference testing on Ubuntu 18.04.6 LTS.

## Usage
1. Install mmseg

```shell
# 1. Create a conda environment
conda create --name slgt2 python=3.9 -y
conda activate slgt2

# 2. Install PyTorch
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit==11.6 -c pytorch -c conda-forge
# CUDA 11.6, 如果 conda 出现问题，使用 pip 即可
# pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

# 3. Install OpenMMLab codebases and other dependencies
# openmmlab codebases
pip install -U openmim
mim install mmcv-full==1.7.2
pip install mmsegmentation==0.30.0 # 0.30.0

# 4. other dependencies
pip install -r requirements.txt
```

2. Prepare the datasets
* Download [Slovenia (MTS12) dataset](http://gpcv.whu.edu.cn/data/dataset12/dataset12.html), thanks [Linying Zhao & Shunping Ji, CNN, RNN, or ViT? An evaluation of different deep learning architectures for spatio-temporal representation of Sentinel time series, JSTARS, 2022](https://ieeexplore.ieee.org/document/9940533). 
* Download [Brandenburg Sentinel-1 time-series dataset](https://github.com/hanzhu97702/ISPRS_STMA), thanks [Spatio-temporal multi-level attention crop mapping method using time-series SAR imagery](https://www.sciencedirect.com/science/article/pii/S0924271623003210).
* For Slovenia dataset, we need to create a data folder, and put the Slovenia dataset in the data folder. The file structure of Slovenia dataset is as followed: 
```shell
  ├── configs
  ├── data                                                
  │   ├── slovenia                                      
  │   │   ├── label                                      
  │   │   │   ├── test                                     
  │   │   │   ├── train                                   
  │   │   │   ├── val                                     
  │   │   ├── s1 (only use s1)                                
  │   │   │   ├── test (297 files)                                     
  │   │   │   ├── train (509 files)                                   
  │   │   │   │   ├── eopatch_id_0_col_0_row_19.mat                 
  │   │   │   │   ├── ......                 
  │   │   │   │   ├── eopatch_id_939_col_49_row_26.mat                 
  │   │   │   ├── val (130 files)                                     
  │   │   ├── s2 (not use)                        
  │   │   │   ├── test                                     
  │   │   │   ├── train                                   
  │   │   │   ├── val                                     
  ```



3. Training
  

- single gpu training

```shell
python tools/train.py ${CONFIG_FILE}  
# for example:
python tools/train.py configs/egst/egst_224x224_40k_slovenia.py
```

- multiple gpus training (assume there are four GPUs)


```shell
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_train.sh ${CONFIG_FILE} 4  
# for example:
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/egst/egst_224x224_40k_slovenia.py 4
```

4. Evaluation

<!-- - pretrained model and training logs

You can download by [Baidu Netdisk](https://pan.baidu.com/s/1kmdtT97en4wfaSRQLYYNlw) (access code: 1234) or [Google Drive](https://drive.google.com/drive/folders/1lqT1fFq_8w6FZH4e-BvXIY_EqFvd7iWI?usp=drive_link). -->

- single gpu testing

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
# for example:
python tools/test.py configs/egst/egst_224x224_40k_slovenia.py work_dirs/egst_224x224_40k_slovenia/train_20241025_022920/iter_40000.pth
```

- multiple gpus testing (assume there are four GPUs)

```shell
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} 4
# for example:
CUDA_VISIABLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/egst/egst_224x224_40k_slovenia.py work_dirs/egst_224x224_40k_slovenia/train_20241025_022920/iter_40000.pth 4
```

If this codebase is helpful for you, please consider give me a star ⭐ 😊.

## Citation
  If you find EGSTNet is useful in your research, please consider citing:
  ```shell

  ```


## Our previous work on time-series SAR image classification
- [SL-GT2Net GitHub link.](https://github.com/RSIP-NJUPT/SL-GT2Net)
- [SL-GT2Net paper link.](https://ieeexplore.ieee.org/document/11015963)
  ```shell
  @ARTICLE{TAES2025SL-GT2Net,
    author={Ni, Kang and Yuan, Chunyang and Zheng, Zhizhong and Wang, Peng},
    journal={IEEE Transactions on Aerospace and Electronic Systems}, 
    title={SAR Image Time Series for Land Cover Mapping via Sparse Local-Global Temporal Transformer Network}, 
    year={2025},
    volume={},
    number={},
    pages={1-17},
    keywords={Land surface;Transformers;Spatiotemporal phenomena;Time series analysis;Crops;Synthetic aperture radar;Radar polarimetry;Telecommunications;Remote sensing;Feature extraction;Deep learning;SAR land cover mapping;Spatial-temporal remote sensing images;Spatial-temporal self-attention;Vision transformer},
    doi={10.1109/TAES.2025.3574022}
    }
  ```

## Acknowledgement
Thanks [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) contribution to the community!
