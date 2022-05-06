<img src="./data/images/logo.png" width="150" >

# FaceX-Zoo
FaceX-Zoo is a PyTorch toolbox for face recognition. It provides a training module with various supervisory heads and backbones towards state-of-the-art face recognition, as well as a standardized evaluation module which enables to evaluate the models in most of the popular benchmarks just by editing a simple configuration. Also, a simple yet fully functional face SDK is provided for the validation and primary application of the trained models. Rather than including as many as possible of the prior techniques, we enable FaceX-Zoo to easilyupgrade and extend along with the development of face related domains. Please refer to the [technical report](https://arxiv.org/pdf/2101.04407.pdf) for more detailed information about this project.
  
About the name:
* "Face" - this repo is mainly for face recognition.
* "X" - we also aim to provide something beyond face recognition, e.g. face parsing, face lightning.
* "Zoo" - there include a lot of algorithms and models in this repo.
![image](data/images/arch.jpg)

# What's New
- [Oct. 2021] [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) is supported now! We obtain a quite promising result of 98.17 in MegaFace rank1 protocol with Swin-S. For more results, pretrained models and logs please refer to [3.1 Experiments of SOTA backbones](training_mode/README.md). For training face recognition model with Swin Transformer, please refer to [swin_training](training_mode/swin_training). Note that the input size is 224\*224 but not 112\*112 and we use AdamW+CosineLRScheduler to optimize it instead of SGD+MultiStepLR.
- [Sep. 2021] We provide a [Dockfile](docker/Dockerfile) to buid the docker image of this project.
- [Aug. 2021] [RepVGG](https://arxiv.org/pdf/2101.03697.pdf) has been added to the backbones for face recognition, the performance of RepVGG_A0, B0, B1 can be found in [3.1 Experiments of SOTA backbones](training_mode).
- [Jul. 2021] A method for facial expression recognition named [DMUE](https://openaccess.thecvf.com/content/CVPR2021/papers/She_Dive_Into_Ambiguity_Latent_Distribution_Mining_and_Pairwise_Uncertainty_Estimation_CVPR_2021_paper.pdf) has been accepted by CVPR2021, and all codes have been released [here](addition_module/DMUE).
- [Jun. 2021] We evaluate some knowledge distillation methods on face recognition task, results and codes can be found in [face_lightning](addition_module/face_lightning/KDF) module.
- [May. 2021] Tools to convert a trained model to onnx format and the provided sdk format can be found in [model_convertor](addition_module/model_convertor).
- [Apr. 2021] IJB-C 1:1 protocol has been added to the [evaluation module](test_protocol/test_ijbc.sh).
- [Mar. 2021] [ResNeSt](https://hangzhang.org/files/resnest.pdf) and [ReXNet](https://arxiv.org/pdf/2007.00992.pdf) have been added to the backbones, [MagFace](https://arxiv.org/pdf/2103.06627.pdf) has been added to the heads. 
- [Feb. 2021] Distributed training and mixed precision training by [apex](https://github.com/NVIDIA/apex) is supported. Please check [distributed_training](training_mode/distributed_training) and [train_amp.py](training_mode/conventional_training/train_amp.py)
- [Jan. 2021] We commit the initial version of FaceX-Zoo.

# Requirements
* python >= 3.7.1
* pytorch >= 1.1.0
* torchvision >= 0.3.0 

# Model Training  
See [README.md](training_mode/README.md) in [training_mode](training_mode), currently support conventional training and [semi-siamese training](https://arxiv.org/abs/2007.08398).
# Model Evaluation  
See [README.md](test_protocol/README.md) in [test_protocol](test_protocol), currently support [LFW](https://people.cs.umass.edu/~elm/papers/lfw.pdf), [CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf), [CALFW](https://arxiv.org/abs/1708.08197), [RFW](https://arxiv.org/abs/1812.00194), [AgeDB30](https://core.ac.uk/download/pdf/83949017.pdf), [IJB-C](http://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf), [MegaFace](https://arxiv.org/abs/1512.00596) and MegaFace-mask.
# Face SDK
See [README.md](face_sdk/README.md) in [face_sdk](face_sdk), currently support face detection, face alignment and face recognition.
# Face Mask Adding
See [README.md](addition_module/face_mask_adding/FMA-3D/README.md) in [FMA-3D](addition_module/face_mask_adding/FMA-3D).

# License
FaceX-Zoo is released under the [Apache License, Version 2.0](LICENSE).

# Acknowledgements
This repo is mainly inspired by [InsightFace](https://github.com/deepinsight/insightface), [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch), [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/README.md). We thank the authors a lot for their valuable efforts.

# Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```BibTeX
@inproceedings{wang2021facex,
  author = {Jun Wang, Yinglu Liu, Yibo Hu, Hailin Shi and Tao Mei},
  title = {FaceX-Zoo: A PyTorh Toolbox for Face Recognition},
  journal = {Proceedings of the 29th ACM international conference on Multimedia},
  year = {2021}
}
```
If you have any questions, please contact with Jun Wang (wangjun492@jd.com), Yinglu Liu (liuyinglu1@jd.com), [Yibo Hu](https://aberhu.github.io/) (huyibo6@jd.com) or [Hailin Shi](https://sites.google.com/view/hailin-shi) (shihailin@jd.com).



优势和特点
支持戴口罩人脸识别模型开发
由于新冠肺炎的影响，人们慢慢习惯于戴口罩出行，对戴着口罩的人脸进行识别给人脸识别算法带来了新的挑战。为了开发一个戴口罩人脸识别模型，需要三方面的前置条件：

（1）戴口罩人脸识别模型训练数据；

（2）戴口罩人脸识别模型测试benchmark；

（3）戴口罩人脸识别模型训练算法；

本项目中提供了所有上述前置条件。

问题描述： 如下图所示，左边为不戴口罩的底库人脸，右边为戴口罩的人脸，戴口罩人脸识别目的在于比对一张戴口罩人脸和不戴口罩底库人脸是否为同一个人。

![image](https://user-images.githubusercontent.com/20282909/167055389-0dc63585-95c6-44ae-a613-d4aca2f58267.png)


戴口罩人脸识别模型训练数据 由问题描述可知戴口罩人脸识别模型的训练需要同时具有一个人戴口罩的人脸图片和不戴口罩的人脸图片。这样的数据采集成本是非常巨大的，甚至是不现实的。该项目设计了基于3D的虚拟口罩添加方式FMA-3D，该方法可鲁棒的在已有不戴口罩人脸数据上添加口罩，从而得到戴口罩人脸训练数据。相关算法原理如下图所示。

 ![image](https://user-images.githubusercontent.com/20282909/167055400-c535f1fa-a48e-4086-a257-836b3c2d62a3.png)


相比目前已有的众多基于2D的虚拟口罩添加方式，基于3D的添加方式能对大姿态等极端条件表现得更加鲁棒；相比一些基于GAN的虚拟口罩添加方法，该方法能几乎不损失原人脸的真实性。一些添加的样本如下图所示，第一行为原图，第二行为添加虚拟口罩后的图。 

![image](https://user-images.githubusercontent.com/20282909/167055415-56231655-4bab-4681-9ee5-507ac5d3360a.png)

戴口罩人脸识别测试benchmark 该项目基于MegaFace设计了大规模戴口罩人脸识别模型测试协议MegaFace-Mask。即将MegaFace测试中的Probe进行虚拟口罩添加并和百万人脸底库进行比对。值得一提的是，所有实现都基于Python实现，通过修改简单的配置即可进行测试。


戴口罩人脸识别模型训练算法 该项目对几种戴口罩人脸识别模型基准算法进行了比对，包括：（1）直接在普通数据上训练得到的模型model1；（2）在上半部分（眼部及以上）人脸数据上训练得到的模型model2；（3）在普通数据+添加虚拟口罩数据上训练得到的model3；（4）融合model2和model3得到的模型model4。四个模型精度对比如下图所示。 

![image](https://user-images.githubusercontent.com/20282909/167055431-7457308c-7a6a-48f4-bb57-1117421da348.png)

浅层人脸识别解决方案
浅层数据即每个类（id）只包含极少数图片的数据，最典型的情况为每个类只包含两张图片，如下图所示。该类型数据在实际应用场景中极为常见，对该类数据进行传统的分类训练精度往往很差。

  ![image](https://user-images.githubusercontent.com/20282909/167055449-d2e6292d-042b-4162-ba3e-fb0a257e1c8d.png)


传统对该类型数据的训练方法往往是基于hard example mining的contrasive loss和triplet loss等，但是这些方法往往比较trick，难以调整。近年来self-supervised learning在图片分类领域非常火热，该项目将其思想应用到浅层人脸识别中，取得了很不错的效果，相关分析在ECCV 2020论文 Semi-Siamese Training for Shallow Face Learning 中。算法框架如下图所示。 

![image](https://user-images.githubusercontent.com/20282909/167055461-1bb6813d-041d-4ba8-9f87-78e3bd9dd6f5.png)


该项目将semi-siamese training作为与传统训练模式（conventional training）对应的一种新的训练模式。并提供了端到端的训练方案，只需在项目中进行简单配置即可对浅层数据进行训练。在浅层数据上，semi-siamese training与conventional training方式精度对比如下表所示。 

![image](https://user-images.githubusercontent.com/20282909/167055472-70137d9d-985e-44e0-a48d-d5961908b8a2.png)


统一的测试协议
人脸识别中有许多测试benchmark，但其中大多数都是基于传统LFW和MegaFace的测试协议。不同benchmark的差异主要表现在测试数据和选取的测试pair上。该项目将这些benchmark进行了整合，只需要通过简单的修改即可在不同的benchmark上进行测试。此外项目还release了LFW和MegaFace的检测框和106点人脸关键点，这些检测框对原始官方release的检测框做了修正（只修正检测框不准的问题，对于一些官方给错检测框id的情况，为了保持测试的一致性，并未做修正）。测试模块的设计如下图所示。目前包含的测试benchmark包括：LFW, CPLFW, CALFW, RFW, AgeDB30, MegaFace 和MegaFace-mask，值得一提的是项目中将原始.bin的MegaFace用Python实现了一遍，测试精度与原始.bin版几乎完全一致。 
![image](https://user-images.githubusercontent.com/20282909/167055485-356fef9b-cd1e-4b36-b240-87b2c60897f0.png)



模块化设计并包含大多主流算法
该项目大多数模块都采用面向对象的方式进行设计，包括了state-of-the-art 的backbone和head。目前项目中包含的backbone和head如下所示。 


![image](https://user-images.githubusercontent.com/20282909/167055490-edbf2a63-5772-4362-bdfc-b417f16602e1.png)

总结
该项目首先包括了自己差异化的一些东西，比如上述的戴口罩人脸识别、浅层人脸识别以及统一测试协议等解决方案。也尽可能的包含了目前state-of-the-art的算法，甚至还包含一个用来直观验证算法效果的python版SDK。可以说是一个既有明确优势特点也功能较为齐全的项目。作者也将从更多的方面去进一步完善项目，主要完善的方向为（1）更多的additional modules，比如人脸模型小型化，人脸解析等；（2）更多的state-of-the-art算法；（3）更高效的训练，包括混合精度训练、分布式训练等。
