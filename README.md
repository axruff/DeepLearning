# Machine Learning papers and resources


##### Table of Contents
- [Neural Networks](#neural-networks)
  - [Overview](#overview)
  - [Opinions](#opinions)
  - [Models](#models)
    - [Multi-level](#multi-level)
    - [Context and Attention](#context-and-attention)
    - [Composition](#composition)
  - [Optimization and Regularisation](#optimization-and-regularisation)
  - [Speed: Pruning, Compression](#pruning-and-compression)
  - [Visualization](#visualization-and-analysis)  
  - [Data Augmentation](#data-augmentation)
  - [Segmentation](#segmentation) 
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

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Overview
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2016 - An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)

[2017 - Revisiting Unreasonable Effectiveness of Data in Deep Learning Era](https://arxiv.org/abs/1707.02968)



<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Opinions
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->


[2016 - Building Machines That Learn and Think Like People](https://www.semanticscholar.org/paper/Building-Machines-That-Learn-and-Think-Like-People-Lake-Ullman/5721a0c623aeb12a65b4d6f5a5c83a5f82988d7c)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2016 - A Berkeley View of Systems Challenges for AI](https://arxiv.org/abs/1712.05855)

[2018 - Deep Learning: A Critical Appraisal](https://arxiv.org/abs/1801.00631)

[2018 - Human-level intelligence or animal-like abilities?](https://dl.acm.org/citation.cfm?id=3271625)

[2019 - Deep Nets: What have they ever done for Vision?](https://arxiv.org/abs/1805.04025)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)



<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Models
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)


[**[AlexNet]** ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229.pdf)

[Learning Hierarchical Features for Scene Labeling](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)


[**[VGG]** Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[**[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

<img src="http://3.bp.blogspot.com/-aM69pqJLP9k/VT2927f8WmI/AAAAAAAAAv8/7S49kEq5Ss0/s1600/%E6%93%B7%E5%8F%96.PNG" width="400">


[**[GoogleNet]** Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)


[**[ResNet]** Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2016 - **[WRN]**: Wide Residual Networks](https://arxiv.org/abs/1605.07146) [[github]](https://github.com/szagoruyko/wide-residual-networks)

[2015 - **[FCN]** Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

<img src="http://deeplearning.net/tutorial/_images/cat_segmentation.png" width="400">


[2015 - **[U-net]**: Convolutional networks for biomedical image segmentation](https://arxiv.org/abs/1505.04597)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2016 - **[Xception]**: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
[Implementation](https://colab.research.google.com/drive/1BT_t64JCzr8ge51orG8uLBLIL7w1Hos4)

[2016 - **[V-Net]**: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)

[2017 - **[MobileNets]**: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)

<img src="http://vladlen.info/wp-content/uploads/2011/12/densecrf1.png" width="300">


[2018 - **[TernausNet]**: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation](https://arxiv.org/abs/1801.05746)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 


[2018 - CubeNet: Equivariance to 3D Rotation and Translation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Daniel_Worrall_CubeNet_Equivariance_to_ECCV_2018_paper.pdf)[[github]](https://github.com/deworrall92/cubenet), [*[video]*](https://www.youtube.com/watch?v=TlzRyHbWeP0&feature=youtu.be)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://i.pinimg.com/564x/8c/c8/44/8cc844bb8784d93790f9d2d2552297bf.jpg" width="350">


[2018 - Deep Rotation Equivariant Network](https://arxiv.org/abs/1705.08623)[[github]](https://github.com/ZJULearning/DREN/raw/master/img/rotate_equivariant.png)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://github.com/ZJULearning/DREN/raw/master/img/rotate_equivariant.png" width="350">


[2019 - **[PacNet]**: Pixel-Adaptive Convolutional Neural Networks](https://arxiv.org/abs/1904.05373)

<img src="https://suhangpro.github.io/pac/fig/pac.png" width="350">

[2019 - Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation](https://arxiv.org/abs/1903.02120v3) [[github]](https://github.com/LinZhuoChen/DUpsampling)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://tonghe90.github.io/papers/cvpr2019_tz.png" width="400">

[2019 - Panoptic Feature Pyramid Networks](http://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.html)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2019 - **[DeeperLab]**: Single-Shot Image Parser](https://arxiv.org/abs/1902.05093)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="http://deeperlab.mit.edu/deeperlab_illustration.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
### Multi-level
<!--- ------------------------------------------------------------------------------- -->

[2014 - **[SPP-Net]** Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)

<img src="http://kaiminghe.com/eccv14sppnet/img/sppnet.jpg" width="350">


[2016 - **[ParseNet]**: Looking Wider to See Better](https://arxiv.org/abs/1506.04579)

<img src="https://miro.medium.com/max/700/1*dRhGetHArI_bs6IdiIFhkA.png" width="350">


[2016 - **[PSPNet]**: Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105v2) [[github]](https://github.com/hszhao/PSPNet)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://hszhao.github.io/projects/pspnet/figures/pspnet.png" width="400">


[2016 - **[DeepLab]**: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)


[2015 - Zoom Better to See Clearer: Human and Object Parsing with Hierarchical Auto-Zoom Net](https://arxiv.org/abs/1511.06881)

<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-319-46454-1_39/MediaObjects/419978_1_En_39_Fig1_HTML.gif" width="350">

[2016 - Attention to Scale: Scale-aware Semantic Image Segmentation](https://arxiv.org/abs/1511.03339)

<img src="http://liangchiehchen.com/fig/attention.jpg" width="350">

[2017 - Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)


[2017 - Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

<img src="https://1.bp.blogspot.com/-Q0-o_ej8BDU/WTYnS568nPI/AAAAAAAAADQ/TTBczrPIQi8IvXZrjy3suRDBlo_p1pONQCLcB/s640/r1.png" width="400">


[2018 - **[DeepLabv3]**: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

<img src="https://2.bp.blogspot.com/-gxnbZ9w2Dro/WqMOQTJ_zzI/AAAAAAAACeA/dyLgkY5TnFEf2j6jyXDXIDWj_wrbHhteQCLcBGAs/s640/image2.png" width="400">


[2019 - **[FastFCN]**: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation](https://arxiv.org/abs/1903.11816v1) [[github]](https://github.com/wuhuikai/FastFCN)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="http://wuhuikai.me/FastFCNProject/images/Framework.png" width="350">

[2019 - Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)

<img src="https://i.pinimg.com/564x/72/a2/5c/72a25c7d87e1c4dfef45bec81adee2e7.jpg" width="250">


<!--- ------------------------------------------------------------------------------- -->
### Context and Attention
<!--- ------------------------------------------------------------------------------- -->

[Image Captioning with Semantic Attention](https://arxiv.org/abs/1603.03925)

<img src="http://cdn-ak.f.st-hatena.com/images/fotolife/P/PDFangeltop1/20160406/20160406161035.png" width="350">

[**[EncNet]** Context Encoding for Semantic Segmentation (2018)](https://arxiv.org/abs/1803.08904v1) [[github]](https://github.com/zhanghang1989/PyTorch-Encoding)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZS4gdvv26N8N7dpr92pPoHmVP3RQ8ztddravjJlwHr1Sw5fCT" width="400">



<!--- ------------------------------------------------------------------------------- -->
### Composition
<!--- ------------------------------------------------------------------------------- -->

[2005 - Image Parsing: Unifying Segmentation, Detection, and Recognition](https://link.springer.com/article/10.1007/s11263-005-6642-x)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)


[2013 - Complexity of Representation and Inference in Compositional Models with Part Sharing](https://arxiv.org/abs/1301.3560)

[2019 - Local Relation Networks for Image Recognition ](https://arxiv.org/pdf/1904.11491.pdf)

<img src="https://www.groundai.com/media/arxiv_projects/536654/x1.png" width="350">

[2017 - Teaching Compositionality to CNNs](https://www.semanticscholar.org/paper/Teaching-Compositionality-to-CNNs-Stone-Wang/3726b82007512a15a530fd1adad57af58a9abb62)

<img src="https://www.vicarious.com/wp-content/uploads/2017/10/compositionality3.png" width="350">



<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Optimization and Regularisation
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->


[Random search for hyper-parameter optimisation](http://www.jmlr.org/papers/v13/bergstra12a.html)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

[**[Adam]**: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

[**[Dropout]**: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)


[Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/abs/1406.6909)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

Multi-Scale Context Aggregation by Dilated Convolutions
https://arxiv.org/abs/1511.07122

<img src="https://user-images.githubusercontent.com/22321977/48708394-7121c980-ec3d-11e8-98ab-2c116df0aaae.png" width="300">


DARTS: Differentiable Architecture Search
https://arxiv.org/abs/1806.09055
 
[**Bag of Tricks** for Image Classification with Convolutional Neural Networks](htps://arxiv.org/abs/1812.01187v1)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2018 - **Tune**: A Research Platform for Distributed Model Selection and Training](https://arxiv.org/abs/1807.05118) [[github]](https://github.com/ray-project/ray/tree/master/python/ray/tune)

[2017 - Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179)

[2018 - Error Forward-Propagation: Reusing Feedforward Connections to Propagate Errors in Deep Learning](https://arxiv.org/abs/1808.03357)


[2019 - Training Neural Networks with Local Error Signals](https://arxiv.org/abs/1901.06656) [[github]](https://github.com/anokland/local-loss)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2019 - Switchable Normalization for Learning-to-Normalize Deep Representation](https://arxiv.org/abs/1907.10473)

<img src="http://luoping.me/post/family-normalization/SN.png" width="350">


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Pruning and Compression
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2015 - Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

<img src="https://xmfbit.github.io/img/paper-pruning-network-demo.png" width="350">

[2015 - Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img scr="https://anandj.in/wp-content/uploads/dc.png" width="350">


[2019 - The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://miro.medium.com/max/2916/1*IraKnowykSyMZtrW1dJOVA.png" width="350">

> Based on these results, we articulate the lottery ticket hypothesis: dense, randomly-initialized, feed-forward
networks contain subnetworks (winning tickets) that—when trained in isolation—
reach test accuracy comparable to the original network in a similar number of
iterations.

> The winning tickets we find have won the **initialization** lottery: their
connections have initial weights that make training particularly effective.


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Visualization and Analysis
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->


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

[2016 - Discovering Causal Signals in Images](https://arxiv.org/abs/1605.08179)

<img src="https://2.bp.blogspot.com/-ZS7WHgo3f9U/XD26idxNEEI/AAAAAAAABl8/DipJ1Fm3ZK0C3tXhu03psC4nByTlID-sQCLcBGAs/s1600/Screen%2BShot%2B2019-01-15%2Bat%2B19.48.13.png" width="400">


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Data Augmentation
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2014 - Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/abs/1406.6909)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 

<img src="https://lmb.informatik.uni-freiburg.de/Publications/2016/DFB16/augmentation.png" width="300">
 
[2018 - Albumentations: fast and flexible image augmentations](https://arxiv.org/abs/1809.06839) [[github]](https://github.com/albu/albumentations)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2019 - **UDA**: Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)[[github]](https://github.com/google-research/uda)

[2017 - Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2)[[github]](https://github.com/zhunzhong07/Random-Erasing)

<img src="https://github.com/zhunzhong07/Random-Erasing/raw/master/all_examples-page-001.jpg" width="350">

[2018 - Data Augmentation by Pairing Samples for Images Classification](https://arxiv.org/abs/1801.02929)

[2017 - Smart Augmentation - Learning an Optimal Data Augmentation Strategy](https://arxiv.org/abs/1703.08383)

[2017 - Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)


[2019 - Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules](https://arxiv.org/abs/1905.05393) [[github]](https://github.com/arcelien/pba)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2018 - **[AutoAugment]**: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)

[2018 - Synthetic Data Augmentation using GAN for Improved Liver Lesion Classification](https://arxiv.org/abs/1801.02385)

[2018 - GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks](https://arxiv.org/abs/1810.10863)



<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Segmentation
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2019 - Panoptic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://miro.medium.com/max/1400/1*OelVuv2thUGAj_400WfseQ.png" width="350">

[2019 - The Best of Both Modes: Separately Leveraging RGB and Depth for Unseen Object Instance Segmentation](https://arxiv.org/abs/1907.13236)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://i.pinimg.com/564x/31/a7/a1/31a7a1a70bd76e035d92f811cb4701d0.jpg" width="350">

> Recognizing unseen objects is a challenging perception task
since the robot needs to learn the concept of “objects” and generalize it to unseen objects

> An ideal method would combine the generalization capability of training on synthetic depth
and the ability to produce sharp masks by training on RGB.

> Training DSN with depth images allows for better generalization to the real world data

> We posit that mask refinement is an easier problem than directly using RGB as input to produce instance masks.

> For the semantic segmentation loss, we use a weighted cross entropy as this
has been shown to work well in detecting object boundaries in imbalanced images [29].

> In order to train the RRN, we need examples of perturbed masks along with ground truth masks.
Since such perturbations do not exist, this problem can be seen as a data augmentation task where we
augment the ground truth mask into something that resembles an initial mask

> In order to seek a fair comparison, all models trained in this section are trained for 100k iterations
of SGD using a fixed learning rate of 1e-2 and batch size of 8. 

[2019 - ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors](https://arxiv.org/abs/1904.03239)

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_225114/project_350444/images/figures/shapemask_fig1_v3.jpg" width="300">

[2019 - Learning to Segment via Cut-and-Paste](https://arxiv.org/abs/1803.06414)

<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-01234-2_3/MediaObjects/474212_1_En_3_Fig3_HTML.gif" width="350">

[2019 - YOLACT Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)[[github]](https://github.com/dbolya/yolact)

<img src="https://i.pinimg.com/564x/52/0c/3e/520c3ee5e0695482c12a73e096dd4b9f.jpg" width="350">

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Anomaly Detection
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2009 - Anomaly Detection: A Survey](https://www.vs.inf.ethz.ch/edu/HS2011/CPS/papers/chandola09_anomaly-detection-survey.pdf)

[2017 - Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921v1)

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Visual Questions and Object Retrieval
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2015 - Natural Language Object Retrieval](https://arxiv.org/abs/1511.04164)

[2019 - CLEVR-Ref+: Diagnosing Visual Reasoning with Referring Expressions](https://arxiv.org/abs/1901.00850)


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Unsupervised Learning
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2019 - Greedy InfoMax for Biologically Plausible Self-Supervised Representation Learning](https://arxiv.org/abs/1905.11786)

[2019 - Unsupervised Learning via Meta-Learning](https://arxiv.org/abs/1810.02334)

[2019 - Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Reinforcement Learning
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

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


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Datasets
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->


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

[**[DTD]**: Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

<img src="https://i.pinimg.com/564x/16/3e/e0/163ee076d96ef19bf5f7b241d42b60f9.jpg" width="350">


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Benchmarks
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->


[MLPerf: A broad ML benchmark suite for measuring performance of ML software frameworks, ML hardware accelerators, and ML cloud platforms.](https://mlperf.org/results/)

[DAWNBench: is a benchmark suite for end-to-end deep learning training and inference.](https://dawn.cs.stanford.edu/benchmark/)

[DAWNBench: An End-to-End Deep Learning Benchmark and Competition (paper) (2017)](https://dawn.cs.stanford.edu/benchmark/papers/nips17-dawnbench.pdf)




<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Applications
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

Material Recognition in the Wild with the Materials in Context Database
http://opensurfaces.cs.cornell.edu/publications/minc/

tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow
https://ge.in.tum.de/publications/tempogan/

<img src="https://ge.in.tum.de/wp-content/uploads/2018/02/teaser-1080x368.jpg" width="400">

[BubGAN: Bubble Generative Adversarial Networks for Synthesizing Realistic Bubbly Flow Images](https://arxiv.org/abs/1809.02266)
<img src="https://i.pinimg.com/564x/61/64/cb/6164cbe1104d7c38f06307c74da5c14e.jpg" width="400">

[Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping (2018)](https://ieeexplore.ieee.org/abstract/document/8460875)

<img src="https://www.alexirpan.com/public/sim2realgrasping/image-comparison.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
# Applications: Medical Imaging
<!--- ------------------------------------------------------------------------------- -->

Generative adversarial networks for specular highlight removal in endoscopic images
https://doi.org/10.1117/12.2293755
<img src="https://github.com/axruff/ML_papers/raw/master/images/1902.png" width="600">

[Deep learning with domain adaptation for accelerated projection‐reconstruction MR (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27106)

[Synthetic Data Augmentation using GAN for Improved Liver Lesion Classification (2018)](https://arxiv.org/abs/1801.02385)

[GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks (2018)](https://arxiv.org/abs/1810.10863)

[Abdominal multi-organ segmentation with organ-attention networks and statistical fusion (2018)](https://arxiv.org/abs/1804.08414)

[Prior-aware Neural Network for Partially-Supervised Multi-Organ Segmentation (2019)](https://arxiv.org/abs/1904.06346)

<img src="https://www.groundai.com/media/arxiv_projects/530543/x2.png" width="350">

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1361841518302524-gr1.jpg" width="350">

[Breast Tumor Segmentation and Shape Classification in Mammograms using Generative Adversarial and Convolutional Neural Network (2018)](https://arxiv.org/abs/1809.01687)

[2019 - H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes](https://arxiv.org/abs/1709.07330v3)

<img src="https://miro.medium.com/max/2000/1*upabGHvSJDva8wVct21hNg.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
# Applications: X-ray Imaging
<!--- ------------------------------------------------------------------------------- -->

Low-dose X-ray tomography through a deep convolutional neural network
https://www.nature.com/articles/s41598-018-19426-7

*In synchrotron-based XRT, CNN-based processing improves the SNR in the data by an order of magnitude, which enables low-dose fast acquisition of radiation-sensitive samples*

<!--- ------------------------------------------------------------------------------- -->
# Applications: Video and Motion
<!--- ------------------------------------------------------------------------------- -->

Flow-Guided Feature Aggregation for Video Object Detection
https://arxiv.org/abs/1703.10025

Deep Feature Flow for Video Recognition
https://arxiv.org/abs/1611.07715

[Video-to-Video Synthesis (2018)](https://arxiv.org/abs/1808.06601) [[github]](https://github.com/NVIDIA/vid2vid)

<!--- ------------------------------------------------------------------------------- -->
# Application: Denoising and Superresolution
<!--- ------------------------------------------------------------------------------- -->

[Residual Dense Network for Image Restoration (2018)](https://arxiv.org/abs/1812.10477v1) [[github]](https://github.com/yulunzhang/RDN)

<img src="https://i.pinimg.com/564x/12/7e/b4/127eb4dfbf482db1ba436ea960821fae.jpg" width="350" >

<!--- ------------------------------------------------------------------------------- -->
# Applications: Inpainting
<!--- ------------------------------------------------------------------------------- -->

[Image Inpainting for Irregular Holes Using Partial Convolutions (2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf)[[github official]](https://github.com/NVIDIA/partialconv), [[github]](https://github.com/MathiasGruber/PConv-Keras)

<img src="https://i.pinimg.com/564x/63/fa/a3/63faa338eba25225c7e84f1d3bad74d3.jpg" width="350">

[Globally and Locally Consistent Image Completion (2017)](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)[[github]](https://github.com/satoshiiizuka/siggraph2017_inpainting)

<img src="http://iizuka.cs.tsukuba.ac.jp/projects/completion/images/teaser/flickr_4_o.png" width=350>

[Generative Image Inpainting with Contextual Attention(2017)](https://arxiv.org/abs/1801.07892)[[github]](https://github.com/JiahuiYu/generative_inpainting)

<img src="https://user-images.githubusercontent.com/22609465/35364552-6e9dfab0-0135-11e8-8bc1-5f370a9f4b0a.png" width="350">

[Free-Form Image Inpainting with Gated Convolution (2018)](https://arxiv.org/abs/1806.03589)

<img src="https://user-images.githubusercontent.com/22609465/41198673-1aac4f2e-6c38-11e8-9f75-6bac82b94265.jpg" width="350">

<!--- ------------------------------------------------------------------------------- -->
# Applications: Photography
<!--- ------------------------------------------------------------------------------- -->

[Photo-realistic single image super-resolution using a generative adversarial network (2016)](https://arxiv.org/abs/1609.04802)[[github]](https://github.com/tensorlayer/srgan)

<img src="https://vitalab.github.io/deep-learning/images/srgan-super-resolution/figure2.png" width="350">

[A Closed-form Solution to Photorealistic Image Stylization (2018)](https://arxiv.org/abs/1802.06474)[[github]](https://github.com/NVIDIA/FastPhotoStyle)

<img src="http://i.gzn.jp/img/2018/02/21/nvidia-fastphotostyle/00.jpg" width="350">

<!--- ------------------------------------------------------------------------------- -->
# Applications: Misc
<!--- ------------------------------------------------------------------------------- -->

[**[pix2code]**: Generating Code from a Graphical User Interface Screenshot](https://arxiv.org/abs/1705.07962v2) [[github]](https://github.com/tonybeltramelli/pix2code)

<img src="https://i.pinimg.com/564x/be/e6/30/bee6302aec1c80d81ba0d206b47222b9.jpg" width="350">

[Fast Interactive Object Annotation with Curve-GCN (2019)](https://arxiv.org/abs/1903.06874v1)

<img src="https://raw.githubusercontent.com/fidler-lab/curve-gcn/master/docs/model.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
# Software
<!--- ------------------------------------------------------------------------------- -->

[**Caffe**: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093)

[**Tune**: A Research Platform for Distributed Model Selection and Training (2018)](https://arxiv.org/abs/1807.05118) [[github]](https://github.com/ray-project/ray/tree/master/python/ray/tune)

[**Glow**: Compiler for Neural Network hardware accelerators](https://github.com/pytorch/glow)

<img src="https://github.com/pytorch/glow/raw/master/docs/3LevelIR.png" width="400">

