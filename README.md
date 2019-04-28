# CIR

This is a Pytorch implementation of CIR with train codes, which is mainly based on [deeper_wider_siamese_trackers](https://github.com/cvpr2019/deeper_wider_siamese_trackers) and [SiamFC-PyTorch](https://github.com/StrangerZhang/SiamFC-PyTorch). I re-format my code with reference to the author's official code [SiamDW](https://github.com/researchmm/SiamDW)

For more details about the CIR tracker please refer to the paper: [Deeper and Wider Siamese Networks for Real-Time Visual Tracking](https://arxiv.org/abs/1901.01660?context=cs) by Zhipeng Zhang and Houwen Peng.

**NOTE**: The author proposed CIR unit into both `SiamFC` and `SiamRPN`, repectively denoted as `SiamFC+` and `SiamRPN+`. Currently this repo only contained that of `SiamFC+` with backbone `ResNet22` and others will be listed into future work.

### Requirements
- python 3.6
- pytorch == 0.3.1
- numpy == 1.12.1
- opencv == 3.1.0

### Training
- data preparation

  1. Follow the instructions in [SiamFC-PyTorch](https://github.com/StrangerZhang/SiamFC-PyTorch) to curate the VID dataset
  
  2. Create the soft links `data_curated` and `data_curated.lmdb` to folder `dataset`.
  
- start training by
```
$ ./train.sh
```

### Tracking
- data preparation

  1. Create the soft link `OTB2015` to folder `dataset`

- start tracking by
```
$ ./track.sh
```

### Benchmark result
- OTB2015

`CIRfc_baseline` tracking with original model `pretrianed/CIResNet22.pth` from the author 

`CIRfc_pretrained` tracking with model `models/CIResNet22_pretrained.pth` trained from scratch


SiamFC+  | Success | Percision
:-------------: | :-------------: | :-------------:
paper  | 0.618 | 0.820
**here**  | **0.601** | **0.796**

<center class="half">
   <img src="https://i.postimg.cc/sxZCTVZN/success-plots.png" width = "400"/><img src="https://i.postimg.cc/Y9PwN4jF/precision-plots.png" width = "400"/>
</center>

### Future work
- [ ] Further performance improvement of `SiamFC+`. Welcome to any advice and suggestions. My email address is jensen.zhoujh@qq.com.
- [ ] Support for `SiamRPN+`

### Reference
[1] Zhipeng Zhang, Houwen Peng. Deeper and Wider Siamese Networks for Real-Time Visual Tracking. Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019.
