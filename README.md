# Machine Learning papers


##### Table of Contents
- [Neural Networks](#neural-networks)
  - [Models](#models)  
  - [Optimization and Regularisation](#optimization-and-regularisation)  
  - [Visualization](#visualization-and-analysis)  
  - [Data Augmentation](#data-augmentation)  
- [Domain Adaptation and Transfer Learning](#domain-adaptation-and-transfer-learning) 
  - [Surveys](#surveys)
  - [Discrepancy-based Approaches](#discrepancy-based-approaches)
  - [Adversarial-based Approaches](#adversarial-based-approaches)
    - [Generative Models](#generative-models) 
    - [Non-generative Models](#non-generative-models) 
  - [Reconstruction-based Approaches](#reconstruction-based-approaches)
  - [Synthetic Data](#synthetic-data)
  - [Style Transfer](#style-transfer) 
  - [Texture Synthesis](#texture-synthesis) 
- [Anomaly Detection](#anomaly-detection)   
- [Unsupervised Learning](#unsupervised-learning)
- [Reinforcement Learning](#reinforcement-learning)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)
- [Applications](#applications)
  - [Applications: Medical Imaging](#applications-medical-imaging)
  - [Applications: X-ray Imaging](#applications-x-ray-imaging)
  - [Applications: Video and Motion](#applications-video-and-motion)
  - [Application: Denoising and Superresolution](#application-denoising-and-superresolution)
  - [Applications: Inpainting](#applications-inpainting)
  - [Applications: Photography](#applications-photography)
  - [Applications: Misc](#applications-misc)
- [Software](#software)  


Notations

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) Read and worked

![#ff0000](https://placehold.it/15/ff0000/000000?text=+) TODO


# Neural Networks

## Models

[Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[**[AlexNet]** ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229.pdf)

[Learning Hierarchical Features for Scene Labeling](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

[**[VGG]** Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

[**[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

<img src="http://3.bp.blogspot.com/-aM69pqJLP9k/VT2927f8WmI/AAAAAAAAAv8/7S49kEq5Ss0/s1600/%E6%93%B7%E5%8F%96.PNG" width="400">

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[**[GoogleNet]** Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[**[ResNet]** Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[**[FCN]** Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

<img src="http://deeplearning.net/tutorial/_images/cat_segmentation.png" width="400">


![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[**[U-net]**: Convolutional networks for biomedical image segmentation (2015)](https://arxiv.org/abs/1505.04597)

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 
[**[TernausNet]**: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation](https://arxiv.org/abs/1801.05746)

[**[V-Net]**: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)

[**[MobileNets]**: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Pyramid Scene Parsing Network (2017)](https://arxiv.org/abs/1612.01105v2) [[github]](https://github.com/hszhao/PSPNet)

<img src="https://hszhao.github.io/projects/pspnet/figures/pspnet.png" width="400">

[Feature Pyramid Networks for Object Detection (2017)](https://arxiv.org/abs/1612.03144)

<img src="https://1.bp.blogspot.com/-Q0-o_ej8BDU/WTYnS568nPI/AAAAAAAAADQ/TTBczrPIQi8IvXZrjy3suRDBlo_p1pONQCLcB/s640/r1.png" width="400">

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[**[DeepLab]**: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)

[**[DeepLabv3]**: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

<img src="https://2.bp.blogspot.com/-gxnbZ9w2Dro/WqMOQTJ_zzI/AAAAAAAACeA/dyLgkY5TnFEf2j6jyXDXIDWj_wrbHhteQCLcBGAs/s640/image2.png" width="400">

[**[Xception]**: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
[Implementation](https://colab.research.google.com/drive/1BT_t64JCzr8ge51orG8uLBLIL7w1Hos4)


[Rethinking Atrous Convolution for Semantic Image Segmentation (2017)](https://arxiv.org/pdf/1706.05587.pdf)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[**[EncNet]** Context Encoding for Semantic Segmentation (2018)](https://arxiv.org/abs/1803.08904v1) [[github]](https://github.com/zhanghang1989/PyTorch-Encoding)
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZS4gdvv26N8N7dpr92pPoHmVP3RQ8ztddravjJlwHr1Sw5fCT" width="400">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation (2019)](https://arxiv.org/abs/1903.02120v3) [[github]](https://github.com/LinZhuoChen/DUpsampling)
<img src="https://tonghe90.github.io/papers/cvpr2019_tz.png" width="400">

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[**[FastFCN]**: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation (2019)](https://arxiv.org/abs/1903.11816v1) [[github]](https://github.com/wuhuikai/FastFCN)

<img src="http://wuhuikai.me/FastFCNProject/images/Framework.png" width="350">


[Revisiting Unreasonable Effectiveness of Data in Deep Learning Era](https://arxiv.org/abs/1707.02968)

[An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)

[Image Captioning with Semantic Attention](https://arxiv.org/abs/1603.03925)
<img src="https://github.com/axruff/ML_papers/raw/master/images/ImageCaptioningSemanticAttention.png" width="400">

[Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)

<img src="http://vladlen.info/wp-content/uploads/2011/12/densecrf1.png" width="300">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[CubeNet: Equivariance to 3D Rotation and Translation (2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Daniel_Worrall_CubeNet_Equivariance_to_ECCV_2018_paper.pdf)[[github]](https://github.com/deworrall92/cubenet), [*[video]*](https://www.youtube.com/watch?v=TlzRyHbWeP0&feature=youtu.be)

<img src="https://i.pinimg.com/564x/8c/c8/44/8cc844bb8784d93790f9d2d2552297bf.jpg" width="350">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Deep Rotation Equivariant Network (2018)](https://arxiv.org/abs/1705.08623)[[github]](https://github.com/ZJULearning/DREN/raw/master/img/rotate_equivariant.png)

<img src="https://github.com/ZJULearning/DREN/raw/master/img/rotate_equivariant.png" width="350">


[**[PacNet]**: Pixel-Adaptive Convolutional Neural Networks(2019)](https://arxiv.org/abs/1904.05373)

<img src="https://suhangpro.github.io/pac/fig/pac.png" width="350">

## Optimization and Regularisation

Random search for hyper-parameter optimisation
http://www.jmlr.org/papers/v13/bergstra12a.html

Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
https://arxiv.org/pdf/1502.03167.pdf

Adam: A Method for Stochastic Optimization
https://arxiv.org/abs/1412.6980

Dropout: A Simple Way to Prevent Neural Networks from Overfitting
http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf


![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks
https://arxiv.org/abs/1406.6909

Multi-Scale Context Aggregation by Dilated Convolutions
https://arxiv.org/abs/1511.07122

<img src="https://user-images.githubusercontent.com/22321977/48708394-7121c980-ec3d-11e8-98ab-2c116df0aaae.png" width="300">


DARTS: Differentiable Architecture Search
https://arxiv.org/abs/1806.09055

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) **Bag of Tricks** for Image Classification with Convolutional Neural Networks
https://arxiv.org/abs/1812.01187v1

[**Tune**: A Research Platform for Distributed Model Selection and Training (2018)](https://arxiv.org/abs/1807.05118) [[github]](https://github.com/ray-project/ray/tree/master/python/ray/tune)

[Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation (2017)](https://arxiv.org/abs/1602.05179)

[Error Forward-Propagation: Reusing Feedforward Connections to Propagate Errors in Deep Learning (2018)](https://arxiv.org/abs/1808.03357)

[Training Neural Networks with Local Error Signals (2019)](https://arxiv.org/abs/1901.06656) [[github]](https://github.com/anokland/local-loss)



## Visualization and Analysis

Visualizing the Loss Landscape of Neural Nets
https://arxiv.org/abs/1712.09913

Visualizing and Understanding Recurrent Networks
https://arxiv.org/abs/1506.02078

GAN Dissection: Visualizing and Understanding Generative Adversarial Networks
https://arxiv.org/abs/1811.10597v1
Interactive tool:
https://gandissect.csail.mit.edu/

[**[Netron ]** Visualizer for deep learning and machine learning models](https://github.com/lutzroeder/Netron)

<img src="https://raw.githubusercontent.com/lutzroeder/netron/master/media/screenshot.png" width="400">

[Discovering Causal Signals in Images (2016)](https://arxiv.org/abs/1605.08179)

<img src="https://2.bp.blogspot.com/-ZS7WHgo3f9U/XD26idxNEEI/AAAAAAAABl8/DipJ1Fm3ZK0C3tXhu03psC4nByTlID-sQCLcBGAs/s1600/Screen%2BShot%2B2019-01-15%2Bat%2B19.48.13.png" width="400">



## Data Augmentation

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks
https://arxiv.org/abs/1406.6909

<img src="https://lmb.informatik.uni-freiburg.de/Publications/2016/DFB16/augmentation.png" width="300">

Albumentations: fast and flexible image augmentations
https://arxiv.org/abs/1809.06839

Code
https://github.com/albu/albumentations

[**[UDA]** Unsupervised Data Augmentation for Consistency Training (2019)](Unsupervised Data Augmentation for Consistency Training)[[github]](https://github.com/google-research/uda)

[Random Erasing Data Augmentation (2017)](https://arxiv.org/abs/1708.04896v2)[[github]](https://github.com/zhunzhong07/Random-Erasing)

<img src="https://github.com/zhunzhong07/Random-Erasing/raw/master/all_examples-page-001.jpg" width="350">


[Population Based Training of Neural Networks (2017)](https://arxiv.org/abs/1711.09846)


[Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules (2019)](https://arxiv.org/abs/1905.05393) [[github]](https://github.com/arcelien/pba)

[AutoAugment: Learning Augmentation Policies from Data (2018)](https://arxiv.org/abs/1805.09501)

[Synthetic Data Augmentation using GAN for Improved Liver Lesion Classification (2018)](https://arxiv.org/abs/1801.02385)

[GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks (2018)](https://arxiv.org/abs/1810.10863)



# Domain Adaptation and Transfer Learning

## Surveys 

[A Survey on Transfer Learning (2010)](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)

[Transfer learning for visual categorization: A survey (2015)](https://ieeexplore.ieee.org/document/6847217)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Domain Adaptation for Visual Applications: A Comprehensive Survey (2017)](https://arxiv.org/abs/1702.05374)

[Visual domain adaptation: A survey of recent advances (2015)](https://ieeexplore.ieee.org/document/7078994)

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[Deep Visual Domain Adaptation: A Survey (2018)](https://arxiv.org/abs/1802.03601)

[A survey on heterogeneous transfer learning (2017)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0089-0)

[Transfer learning for cross-dataset recognition: A survey (2017)](https://arxiv.org/abs/1705.04396)

[Visual Domain Adaptation Challenge (2018)] (http://ai.bu.edu/visda-2017/)


## Discrepancy-based Approaches

*Description: fine-tuning the deep network with labeled or unlabeled target data to diminish the domain shift*

- **Class Criterion**: uses the class label information as a guide for transferring knowledge between different domains. When the labeled samples from the target domain are available in supervised DA, **soft label** and metric learning are always effective [118], [86], [53], [45], [79]. When such samples are unavailable, some other techniques can be adopted to substitute for class labeled data, such as **pseudo labels** [75], [139], [130],[98] and **attribute representation** [29], [118]. Usually a small
number of labeled samples from the target dataset is assumed to be available. 

- **Statistic Criterion**: aligns the statistical distribution shift between the source and target domains using some mechanisms. 

- **Architecture Criterion**: aims at improving the ability of learning more transferable features by adjusting the architectures of deep networks.

- **Geometric Criterion**: bridges the source and target domains according to their geometrical properties.

#### Class Criterion

Using **soft labels** rather than hard labels can preserve the relationships between classes across domains.

Humans can identify unseen classes given only a high-level description. For instance, when provided the description ”tall brown
animals with long necks”, we are able to recognize giraffes. To imitate the ability of humans, [64] introduced high-level **semantic attributes** per class.

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 
[Fine-grained recognition in the wild: A multi-task domain adaptation approach (2017)](https://arxiv.org/abs/1709.02476) [soft labels, semantic attributes]

[Deep transfer metric learning (2015)](https://ieeexplore.ieee.org/document/7298629)


Occasionally, when fine-tuning the network in unsupervised DA, a label of target data, which is called a pseudo label, can preliminarily be obtained based on the maximum posterior probability

[Mind the class weight bias: Weighted maximum mean discrepancy for unsupervised domain adaptation (2017)](https://arxiv.org/abs/1705.00609)

In [[98]](https://arxiv.org/abs/1702.08400), two different networks assign **pseudo-labels** to unlabeled samples, another network is trained by the samples to obtain target discriminative representations.

[Asymmetric tri-training for unsupervised domain adaptation (2017)](https://arxiv.org/abs/1702.08400)

[**[DTN]** Deep transfer network: Unsupervised domain adaptation (2015)](https://arxiv.org/abs/1503.00591)

#### Statistic Criterion

Although some discrepancy-based approaches search for pseudo labels, attribute labels or other substitutes to labeled
target data, more work focuses on learning **domain-invariant representations** via minimizing the domain distribution discrepancy in unsupervised DA.

**Maximum mean discrepancy** (MMD) is an effective metric for comparing the distributions between two datasets by a kernel two-sample test [3].

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 
[**[DDC]** Deep domain confusion: Maximizing for domain invariance (2014)](https://arxiv.org/abs/1412.3474)

<img src="https://www.groundai.com/media/arxiv_projects/85020/x1.png.750x0_q75_crop.png" width="300">

Rather than using a single layer and linear MMD, Long et al. [[73]](https://arxiv.org/abs/1502.02791) proposed the deep adaptation network (DAN) that matches the shift
in marginal distributions across domains by adding multiple adaptation layers and exploring multiple kernels, assuming that the conditional distributions remain unchanged.

[**[DAN]** Learning transferable features with deep adaptation networks (2015)](https://arxiv.org/abs/1502.02791)

[**[JAN]** Deep transfer learning with joint adaptation networks (2016)](https://arxiv.org/abs/1605.06636)

[**[RTN]** Unsupervised domain adaptation with residual transfer networks (2016)](https://arxiv.org/abs/1602.04433)

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0925231218306684-gr6.jpg" width="400">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Associative Domain Adaptation (2017)](https://arxiv.org/abs/1708.00938)

<img src="https://vision.in.tum.de/_media/spezial/bib/haeusser_iccv_17.png" width="400">

[Return of frustratingly easy domain adaptation (2015)](https://arxiv.org/abs/1511.05547)


#### Architectural Criterion

[Deeper, broader and artier domain generalization (2017)](https://arxiv.org/abs/1710.03077)

<img src="http://www.eecs.qmul.ac.uk/~dl307/img/project_img1.png" width="300">

#### Geometric Criterion

[**[Dlid]**: Deep learning for domain adaptation by interpolating between domains (2013)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.664.4509) [geometric criterion]

## Adversarial-based Approaches

*Description: using domain discriminators to encourage domain confusion through an adversarial objective*

### Generative Models

[Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[**[DANN]** Domain-Adversarial Training of Neural Networks (2015)](https://arxiv.org/abs/1505.07818) [github](https://github.com/fungtion/DANN)

<img src="https://camo.githubusercontent.com/5201a6af692fe44c22cc2dfda8e9db02fb0e0ffc/68747470733a2f2f73312e617831782e636f6d2f323031382f30312f31322f70384b5479442e6d642e6a7067" width="350">


[Improved techniques for training GANs (2016)](https://arxiv.org/abs/1606.03498) [github](https://github.com/openai/improved-gan)


[Domain Separation Networks (2016)](https://arxiv.org/abs/1608.06019)

<img src="https://i.pinimg.com/564x/de/50/fa/de50fac81074e16ca78114f78a379246.jpg" width="350">


![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 
[**[PixelDA]** Unsupervised pixel-level domain adaptation with generative adversarial networks (2016)](https://arxiv.org/abs/1612.05424) 

***[Needs a lot of target images to successfully learn the generator]***

<img src="https://i.pinimg.com/564x/f8/52/1e/f8521e45415762465e5e01452a963a31.jpg" width="400">

[**[ADDA]** Adversarial discriminative domain adaptation (2017)](https://arxiv.org/abs/1702.05464)

<img src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/ADDA_1.jpg" width="400">

This weight-sharing constraint allows **CoGAN** to achieve a domain-invariant feature space
without correspondence supervision. A trained CoGAN can adapt the input noise vector to paired images that are from the
two distributions and share the labels. Therefore, the shared labels of synthetic target samples can be used to train the target
model.

[**[CoGAN]** Coupled generative adversarial networks (2016)](https://arxiv.org/abs/1606.07536)

[Pixel-level domain transfer (2016)](https://arxiv.org/pdf/1603.07442.pdf) [[github]](https://github.com/fxia22/PixelDTGAN)

<img src="https://pbs.twimg.com/media/CgKhQ2hWEAAE231.jpg:large" width="400">

Shrivastava et al. [104]() developed a method for **simulated+unsupervised (S+U)** learning
that uses a combined objective of minimizing an adversarial
loss and a self-regularization loss, where the goal is to improve
the realism of synthetic images using unlabeled real data

[Learning from Simulated and Unsupervised Images through Adversarial Training (2016)](https://arxiv.org/abs/1612.07828)

<img src="https://github.com/axruff/ML_papers/raw/master/images/123.png" width="300">

[Improved Adversarial Systems for 3D Object Generation and Reconstruction (2017)](https://arxiv.org/abs/1707.09557)


[Toward Multimodal Image-to-Image Translation (2018)](https://arxiv.org/abs/1711.11586)

<img src="https://junyanz.github.io/BicycleGAN/index_files/teaser.jpg" width="400"> 

[From Source to Target and Back: Symmetric Bi-Directional Adaptive GAN (2018)](http://openaccess.thecvf.com/content_cvpr_2018/html/Russo_From_Source_to_CVPR_2018_paper.html)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[SRDA: Generating Instance Segmentation Annotation Via Scanning, Reasoning And Domain Adaptation (2018)](https://arxiv.org/abs/1801.08839) ***[Geometry-guided GAN]***

[How good is my GAN? (2018)](https://arxiv.org/abs/1807.09499)

<img src="http://thoth.inrialpes.fr/research/ganeval/images/fig1.png" width="350">


### Non-generative Models

## Reconstruction-based Approaches

*Description: using the data reconstruction as an auxiliary task to ensure feature invariance*


## Others

GAN Zoo
[[link]](https://github.com/hindupuravinash/the-gan-zoo)

<img src="https://github.com/hindupuravinash/the-gan-zoo/raw/master/The_GAN_Zoo.jpg" width="250">


Self-ensembling for visual domain adaptation
[[link]](https://arxiv.org/abs/1706.05208)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Context Encoders: Feature Learning by Inpainting (2016)](https://arxiv.org/abs/1604.07379) [[github]](https://github.com/pathak22/context-encoder)

<img src="https://i.pinimg.com/564x/57/be/d5/57bed585ea990b858314f919db5fc522.jpg" width="400">

[Compositional GAN: Learning Conditional Image Composition (2018)](https://arxiv.org/abs/1807.07560)

<img src="http://pbs.twimg.com/media/Di7lWdWXoAAhd6B.jpg" width="400">

GAN Dissection: Visualizing and Understanding Generative Adversarial Networks
[[link]](https://arxiv.org/abs/1811.10597v)
Interactive tool:
https://gandissect.csail.mit.edu/

Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning
[[link]](https://arxiv.org/abs/1704.03976)

<img src="https://www.researchgate.net/profile/Takeru_Miyato/publication/316098571/figure/fig2/AS:667791753498635@1536225369918/Demonstration-of-how-our-VAT-works-on-semi-supervised-learning-We-generated-8-labeled.png" width="300">

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.jpg" width="300">

[Pros and Cons of GAN Evaluation Measures (2018)](https://arxiv.org/abs/1802.03446)

# Synthetic Data

[Playing for Data: Ground Truth from Computer Games ()](https://arxiv.org/abs/1608.02192)

<img src="https://github.com/axruff/ML_papers/raw/master/images/PlayingforData.png" width="300">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[On Pre-Trained Image Features and Synthetic Images for Deep Learning (2017)](https://arxiv.org/abs/1710.10710)

<img src="https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-3-030-11009-3_42/MediaObjects/478770_1_En_42_Fig3_HTML.png" width="350">

[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World (2017)](https://arxiv.org/abs/1703.06907)

<img src="https://i.pinimg.com/564x/99/e9/65/99e9657382ef1e94e2acb958c7c1cf6c.jpg" width="350">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization (2017)](https://arxiv.org/abs/1804.06516)

<img src="https://research.nvidia.com/sites/default/files/publications/cvpr-fig1_down4.png" width="350">


[Learning to Segment via Cut-and-Paste (2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tal_Remez_Learning_to_Segment_ECCV_2018_paper.pdf)

<img src="https://cdn-images-1.medium.com/max/1600/0*b4CBigBlK_LyGU16" width="350">

# Style Transfer

[Image Style Transfer Using Convolutional Neural Networks (2016)](https://ieeexplore.ieee.org/document/7780634)

<imf src="https://i-h1.pinimg.com/564x/8a/e4/97/8ae497d18a7c409c2da67833d5586461.jpg" width="250">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Perceptual losses for real-time style transfer and super-resolution (2016)](https://arxiv.org/abs/1603.08155?context=cs) [[github]](https://github.com/jcjohnson/fast-neural-style)

<img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/RTNS.png" width="350">


[**[CycleGAN]** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (2017)](https://arxiv.org/abs/1703.10593)

<img src="https://junyanz.github.io/CycleGAN/images/teaser.jpg" width="400">

[**[pix2pix]** Image-to-Image Translation with Conditional Adversarial Networks (2016)](https://arxiv.org/abs/1611.07004)[[github]](https://phillipi.github.io/pix2pix/)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[**[DRIT]** Diverse Image-to-Image Translation via Disentangled Representations (2018)](https://arxiv.org/abs/1808.00948)[[github]](https://github.com/HsinYingLee/DRIT)

<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-01246-5_3/MediaObjects/474172_1_En_3_Fig3_HTML.gif" width="350">

[A Style-Based Generator Architecture for Generative Adversarial Networks (2018)](https://arxiv.org/abs/1812.04948)[[github]](https://github.com/NVlabs/stylegan)

<img src="https://github.com/NVlabs/stylegan/raw/master/stylegan-teaser.png" width="350">

[Deep Painterly Harmonization (2018)](https://arxiv.org/abs/1804.03189v3) [[github]](https://github.com/luanfujun/deep-painterly-harmonization)

<img src="https://i.pinimg.com/564x/f6/fa/74/f6fa740395c1b99ecee2b71f46b16751.jpg" width="350">

# Texture Synthesis

[Texture Synthesis Using Convolutional Neural Networks (2015)](https://arxiv.org/abs/1505.07376) [[github]](https://mc.ai/tensorflow-implementation-of-paper-texture-synthesis-using-convolutional-neural-networks/)

<img src="https://dmitryulyanov.github.io/assets/online-neural-doodle/textures.png" width="300">

DeepTextures
http://bethgelab.org/deeptextures/

Textures database
https://www.textures.com/index.php

# Anomaly Detection

Anomaly Detection: A Survey
https://www.vs.inf.ethz.ch/edu/HS2011/CPS/papers/chandola09_anomaly-detection-survey.pdf

[Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (2017)](https://arxiv.org/abs/1703.05921v1)


# Unsupervised Learning

[Greedy InfoMax for Biologically Plausible Self-Supervised Representation Learning (2019)](https://arxiv.org/abs/1905.11786)

[Unsupervised Learning via Meta-Learning (2019)](https://arxiv.org/abs/1810.02334)

[Representation Learning with Contrastive Predictive Coding (2019)](https://arxiv.org/abs/1807.03748)


# Reinforcement Learning

An Introduction to Deep Reinforcement Learning
https://arxiv.org/abs/1811.12560

Deep Reinforcement Learning
https://arxiv.org/abs/1810.06339

Playing Atari with Deep Reinforcement Learning
https://arxiv.org/pdf/1312.5602.pdf

Key Papers in Deep Reinforcment Learning
https://spinningup.openai.com/en/latest/spinningup/keypapers.html

Recurrent Models of Visual Attention
https://arxiv.org/abs/1406.6247

<img src="https://raw.githubusercontent.com/torch/torch.github.io/master/blog/_posts/images/rva-diagram.png" width="400">

# Datasets

[**[ADE20K Dataset]**: Semantic Segmentation [website]](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

Scene Parsing through ADE20K Dataset
http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf

<img src="http://groups.csail.mit.edu/vision/datasets/ADE20K/assets/images/examples.png" width="400">

[**[OPENSURFACES]**: A Richly Annotated Catalog of Surface Appearance](http://opensurfaces.cs.cornell.edu/)

https://www.cs.cornell.edu/~paulu/opensurfaces.pdf

<img src="http://opensurfaces.cs.cornell.edu/static/img/teaser4-web.jpg" width="400">

[**[ShapeNet]** - a richly-annotated, large-scale dataset of 3D shapes [website] ](https://www.shapenet.org/) 

[ShapeNet: An Information-Rich 3D Model Repository](https://arxiv.org/abs/1512.03012)

<img src="https://www.shapenet.org/resources/images/logo.png" width="500">

[Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild [website]](http://cvgl.stanford.edu/projects/pascal3d.html)
[[paper]](https://ieeexplore.ieee.org/document/6836101)

[**[ObjectNet3D]**: A Large Scale Database for 3D Object Recognition](http://cvgl.stanford.edu/projects/objectnet3d/)

<img src="http://cvgl.stanford.edu/projects/objectnet3d/ObjectNet3D.png" width="300">

[**[ModelNet]**: a comprehensive clean collection of 3D CAD models for objects [website]](http://modelnet.cs.princeton.edu/)

<img src="http://3dvision.princeton.edu/projects/2014/ModelNet/thumbnail.jpg" width="300">

[**[3D ShapeNets]**: A Deep Representation for Volumetric Shapes (2015)](https://ieeexplore.ieee.org/document/7298801)

[**[BLEND SWAP]**: IS A COMMUNITY OF PASSIONATE BLENDER ARTISTS WHO SHARE THEIR WORK UNDER CREATIVE COMMONS LICENSES](https://www.blendswap.com/)

# Benchmarks


[MLPerf: A broad ML benchmark suite for measuring performance of ML software frameworks, ML hardware accelerators, and ML cloud platforms.](https://mlperf.org/results/)

[DAWNBench: is a benchmark suite for end-to-end deep learning training and inference.](https://dawn.cs.stanford.edu/benchmark/)

[DAWNBench: An End-to-End Deep Learning Benchmark and Competition (paper) (2017)](https://dawn.cs.stanford.edu/benchmark/papers/nips17-dawnbench.pdf)




# Applications

Material Recognition in the Wild with the Materials in Context Database
http://opensurfaces.cs.cornell.edu/publications/minc/

tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
https://ge.in.tum.de/publications/tempogan/

<img src="https://ge.in.tum.de/wp-content/uploads/2018/02/teaser-1080x368.jpg" width="400">

[BubGAN: Bubble Generative Adversarial Networks for Synthesizing Realistic Bubbly Flow Images](https://arxiv.org/abs/1809.02266)
<img src="https://i.pinimg.com/564x/61/64/cb/6164cbe1104d7c38f06307c74da5c14e.jpg" width="400">

[Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping (2018)](https://ieeexplore.ieee.org/abstract/document/8460875)

<img src="https://www.alexirpan.com/public/sim2realgrasping/image-comparison.png" width="350">

# Applications: Medical Imaging

Generative adversarial networks for specular highlight removal in endoscopic images
https://doi.org/10.1117/12.2293755
<img src="https://github.com/axruff/ML_papers/raw/master/images/1902.png" width="600">

[Deep learning with domain adaptation for accelerated projection‐reconstruction MR (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27106)

[Synthetic Data Augmentation using GAN for Improved Liver Lesion Classification (2018)](https://arxiv.org/abs/1801.02385)

[GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks (2018)](https://arxiv.org/abs/1810.10863)

# Applications: X-ray Imaging
Low-dose X-ray tomography through a deep convolutional neural network
https://www.nature.com/articles/s41598-018-19426-7

*In synchrotron-based XRT, CNN-based processing improves the SNR in the data by an order of magnitude, which enables low-dose fast acquisition of radiation-sensitive samples*

# Applications: Video and Motion

Flow-Guided Feature Aggregation for Video Object Detection
https://arxiv.org/abs/1703.10025

Deep Feature Flow for Video Recognition
https://arxiv.org/abs/1611.07715

[Video-to-Video Synthesis (2018)](https://arxiv.org/abs/1808.06601) [[github]](https://github.com/NVIDIA/vid2vid)

# Application: Denoising and Superresolution

[Residual Dense Network for Image Restoration (2018)](https://arxiv.org/abs/1812.10477v1) [[github]](https://github.com/yulunzhang/RDN)

<img src="https://i.pinimg.com/564x/12/7e/b4/127eb4dfbf482db1ba436ea960821fae.jpg" width="350" >

# Applications: Inpainting

[Image Inpainting for Irregular Holes Using Partial Convolutions (2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf)[[github official]](https://github.com/NVIDIA/partialconv), [[github]](https://github.com/MathiasGruber/PConv-Keras)

<img src="https://i.pinimg.com/564x/63/fa/a3/63faa338eba25225c7e84f1d3bad74d3.jpg" width="350">

[Globally and Locally Consistent Image Completion (2017)](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)[[github]](https://github.com/satoshiiizuka/siggraph2017_inpainting)

<img src="http://iizuka.cs.tsukuba.ac.jp/projects/completion/images/teaser/flickr_4_o.png" width=350>

[Generative Image Inpainting with Contextual Attention(2017)](https://arxiv.org/abs/1801.07892)[[github]](https://github.com/JiahuiYu/generative_inpainting)

<img src="https://user-images.githubusercontent.com/22609465/35364552-6e9dfab0-0135-11e8-8bc1-5f370a9f4b0a.png" width="350">

[Free-Form Image Inpainting with Gated Convolution (2018)](https://arxiv.org/abs/1806.03589)

<img src="https://user-images.githubusercontent.com/22609465/41198673-1aac4f2e-6c38-11e8-9f75-6bac82b94265.jpg" width="350">

# Applications: Photography

[Photo-realistic single image super-resolution using a generative adversarial network (2016)](https://arxiv.org/abs/1609.04802)[[github]](https://github.com/tensorlayer/srgan)

<img src="https://vitalab.github.io/deep-learning/images/srgan-super-resolution/figure2.png" width="350">

[A Closed-form Solution to Photorealistic Image Stylization (2018)](https://arxiv.org/abs/1802.06474)[[github]](https://github.com/NVIDIA/FastPhotoStyle)

<img src="http://i.gzn.jp/img/2018/02/21/nvidia-fastphotostyle/00.jpg" width="350">

# Applications: Misc

[**[pix2code]**: Generating Code from a Graphical User Interface Screenshot](https://arxiv.org/abs/1705.07962v2) [[github]](https://github.com/tonybeltramelli/pix2code)

<img src="https://i.pinimg.com/564x/be/e6/30/bee6302aec1c80d81ba0d206b47222b9.jpg" width="350">

# Software

[**Caffe**: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093)

[**Tune**: A Research Platform for Distributed Model Selection and Training (2018)](https://arxiv.org/abs/1807.05118) [[github]](https://github.com/ray-project/ray/tree/master/python/ray/tune)

[**Glow**: Compiler for Neural Network hardware accelerators](https://github.com/pytorch/glow)

<img src="https://github.com/pytorch/glow/raw/master/docs/3LevelIR.png" width="400">

