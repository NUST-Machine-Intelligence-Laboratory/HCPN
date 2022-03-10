## HCPN
Code for paper: Hierarchical Co-attention Propagation Network for Zero-Shot Video Object Segmentation

## Requirements
- Python (3.6.12)

- PyTorch (version:1.7.0) 

- Requirements in the requirements.txt files.


## Training
### Download Datasets
1. Download the DAVIS-2017 dataset from [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip).
2. Download the YouTube-VOS dataset from [YouTube-VOS](https://youtube-vos.org/dataset/).
3. Download the YouTube-hed and DAVIS-hed datasets from [DuBox](https://dubox.com/s/1tA7075RgnKtLRQdpIMFCsg) code: 1gih.
4. Download the YouTube-ctr and DAVIS-ctr datasets from [GoogleDriver](https://drive.google.com/drive/folders/1GspmMlDQA4pyheiBU62RYuKzho2x_YS7?usp=sharing).
5. The optical flow files are obtained by [RAFT](https://github.com/princeton-vl/RAFT), we provide demo code that can be run directly on path ```flow```.
We also provide optical flow of YouTube-VOS (18G) in [DuBox](https://dubox.com/s/1TDIU_cY218Ygc3q86JM-fQ) code: w9yn, 
   optical flow of DAVIS can be found in Section Testing.
### Dataset Format
Please ensure the datasets are organized as following format. 
```
YouTube-VOS
|----train
      |----Annotations
      |----Annotations_ctr
      |----JPEGImages
      |----YouTube-flow
      |----YouTube-hed
      |----meta.json
|----valid
      |----Annotations
      |----JPEGImages
      |----meta.json
```

```
DAVIS
      |----Annotations
      |----Annotations_ctr
      |----ImageSets
      |----JPEGImages
      |----davis-flow
      |----davis-hed
```
### Run train.py
Change your dataset paths, then run ```python train.py``` for training model.

We also provide multi-GPU parallel code based on [apex](https://github.com/NVIDIA/apex).
Run ```CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 train_apex.py``` for distributed training in Pytorch.
### Note: 
Please change the path in two codes (```libs/utils/config_davis.py```and ```libs/utils/config_youtubevos.py```) to your own dataset path.
## Testing
If you want to test the model results directly, you can follow the settings below.
1. Download the pretrained model from [GoogleDrive](https://drive.google.com/drive/folders/1LYyAZtDHv8nTKVB6xY05TUSJ_7QnhnmJ?usp=sharing) and put it into the "model/HCPN" files. 

2. Download the optical flow of DAVIS from [GoogleDrive](https://drive.google.com/file/d/1ADBNzRyZwJUJVO77Iutu_H6tIN2n5SS0/view?usp=sharing).

The code directory structure is as follows.
```
HCPN
  |----libs
  |----model
  |----apply_densecrf_davis.py
  |----args.py
  |----train.py
  |----test.py
```
3. Change your path in ```test.py```, then run ```python test.py```.


4. Evaluation code from [DAVIS_Evaluation](https://github.com/davisvideochallenge/davis-matlab/tree/davis-2016), the python version is available at[PyDavis16EvalToolbox](https://github.com/lartpang/PyDavis16EvalToolbox).
## Results
If you are not able to run our code but interested in our results, 
the segmentation results can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1EIzgDZaylhZ9rNkf4bShOY_slSqinD_W/view?usp=sharing).

1. **DAVIS-16**:

In the inference stage, we ran using the 512x512 size of DAVIS (480p).

**Mean J&F** |  **J score** | **F score** | 
---------|  :---------: | :---------: 
 **85.6** | **85.8** | **85.4** |


 2. **Youtube-Objects**:
 
**Airplane** | **Bird** | **Boat** |  **Car** | **Cat** | **Cow** |  **Dog** | **Horse** | **Motorbike** |**Train** |**Mean** |
---------|  :---------: | :---------: |:---------: | :---------: |:---------: | :---------: |:---------: | :---------: | :---------: | :---------: 
 **84.5** | **79.6** | **67.3** |**87.8** | **74.1** | **71.2** |**76.5** | **66.2** | **65.8** |**59.7** | **73.3** |


 3. **FBMS**:

**Mean J** |
---------|
 **78.3** |
 
 4. **DAVIS-17**:
 
**Mean J&F** |  **J score** | **F score** | 
---------|  :---------: | :---------: 
 **70.7** | **68.7** | **72.7** |

DAVIS-2017
## Video for Demo
[Demo_DAVIS2016](https://drive.google.com/file/d/1j5mpv8R5c1CUtqX5-DIi1tuLLpS6SD_A/view?usp=sharing)

[Demo_YouTube-Objects](https://drive.google.com/file/d/1VtUZPkvip0Gqnlt_DvjgeijGF-SQatGL/view?usp=sharing)

[Demo_FBMS](https://drive.google.com/file/d/1-pFVc1wrB41QnefzUGZdHFfRDm0e3svB/view?usp=sharing)

[Demo_DAVIS17](https://drive.google.com/file/d/1qbrJtavp2xUI8fm53urL6KLWvzni17CK/view?usp=sharing)

## Acknowledge

1. Motion-Attentive Transition for Zero-Shot Video Object Segmentation, AAAI 2020 (https://github.com/tfzhou/MATNet)
2. Video Object Segmentation Using Space-Time Memory Networks, ICCV 2019 (https://github.com/seoungwugoh/STM)
3. See More, Know More: Unsupervised Video Object Segmentation With Co-Attention Siamese Networks, CVPR 2019 (https://github.com/carrierlxk/COSNet)
