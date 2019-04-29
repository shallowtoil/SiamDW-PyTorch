# SiamDW-Pytorch

This is a Pytorch implementation of SiamDW with train codes, which is mainly based on [deeper_wider_siamese_trackers](https://github.com/cvpr2019/deeper_wider_siamese_trackers) and [Siamese-RPN](https://github.com/HelloRicky123/Siamese-RPN). I re-format my code with reference to the author's official code [SiamDW](https://github.com/researchmm/SiamDW). 

For more details about the CIR tracker please refer to the paper: [Deeper and Wider Siamese Networks for Real-Time Visual Tracking](https://arxiv.org/abs/1901.01660?context=cs) by Zhipeng Zhang and Houwen Peng.

**NOTE**: The author proposed CIR/CIR-D unit into both `SiamFC` and `SiamRPN`, repectively denoted as `SiamFC+` and `SiamRPN+`. Currently this repo only contained that of `SiamFC+` and `SiamRPN+` with backbone `ResNet22` and others will be listed into future work.

The repo is still under development.

### Requirements
- python == 3.6
- pytorch == 0.3.1
- numpy == 1.12.1
- opencv == 3.1.0

### Training
- data preparation

  1. Follow the instructions in [Siamese-RPN](https://github.com/HelloRicky123/Siamese-RPN) to curate the `VID` and `YTB` dataset. If you also want to use `GOT10K`, follow the same instructions as above or download the curated data by [author](https://github.com/researchmm/SiamDW).
  
  2. Create the soft links `data_curated` and `data_curated.lmdb` to folder `dataset`.

- download pretrained model

  1. Download pretrained model from [OneDrive](https://mailccsf-my.sharepoint.com/:u:/g/personal/zhipeng_mail_ccsf_edu/EXLC8YnM9B9Kq5KcqfjbFg4B-OIwp6ZflvW_p0s0K3R1_Q?e=XNqj3n), [GoogleDrive](https://drive.google.com/open?id=1RIMB9542xXp60bZwndTvmIt2jogxAIX3) or [BaiduDrive](https://pan.baidu.com/s/1TmIW8AsLEr9Mk3qSsT1pIg). Extracted code for BaiduDrive is `7rfu`.
  
  2. Put them to `models/pretrain` directory. 

- chosse the training dataset by set the parameters in `lib/utils/config.py`.

  For example, if you would like to use both `VID` and `YTB` dataset to train `SiamRPN+`, then just simply set both `VID_used` and `YTB_used` into `True`.
  
- choose the model to be trained by modifying `train.sh`, e.g,  to train `SiamFC+` using command 
    ```
    CUDA_VISIBLE_DEVICES=0 python bin/train_siamfc.py --arch SiamFC_Res22 --resume ./models/SiamFC_Res22_mine.pth
    ```
	or to train `SiamRPN+` by
    ```
    CUDA_VISIBLE_DEVICES=0 python bin/train_siamrpn.py --arch SiamRPN_Res22 --resume ./models/SiamRPN_Res22_mine.pth
    ```


### Tracking
- data preparation

  1. Create the soft link `OTB2015` to folder `dataset`

- start tracking by modifying `test.sh` as above

### Benchmark result
- OTB2015

`CIRfc_baseline` tracking with original model `pretrianed/CIResNet22.pth` from the author 

`CIRfc_pretrained` tracking with model `models/CIResNet22_pretrained.pth` trained from scratch


Models  | Success | Percision
:-------------: | :-------------: | :-------------:
SiamFC_Res22  | 0.639 | 0.839
SiamRPN_Res22  | 0.662 | 0.872
SiamFC_Res22(mine)  | 0.632 | 0.831
SiamRPN_Res22(mine)  |  |

<center class="half">
   <img src="https://i.postimg.cc/sxZCTVZN/success-plots.png" width = "400"/> <img src="https://i.postimg.cc/Y9PwN4jF/precision-plots.png" width = "400"/>
</center>

### Future work
- [ ] Further performance improvement of `SiamFC+` and `SiamRPN+`. Welcome to any advice and suggestions. My email address is jensen.zhoujh@qq.com.

### Reference
[1] Zhipeng Zhang, Houwen Peng. Deeper and Wider Siamese Networks for Real-Time Visual Tracking. Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019.
