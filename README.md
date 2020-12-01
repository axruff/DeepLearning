# Deep Learning papers and resources


##### Table of Contents
- [Neural Networks](#neural-networks)
  - [Models](#models)
    - [Multi-level](#multi-level)
    - [Context and Attention](#context-and-attention)
    - [Composition](#composition)
    - [Capsule Networks](#capsule-networks)
  - [Optimization](#optimization)
    - [Optimization and Regularisation](#optimization-and-regularisation)
    - [Pruning, Compression](#pruning-and-compression)
  - [Analysis and Interpretability](#analysis-and-interpretability) 
- [Tasks](#tasks)
  - [Segmentation](#segmentation)
- [Methods](#neural-networks)
  - [TL - Transfer Learning](#transfer-learning)
  - [GM - Generative Modelling](#generative-modelling)
  - [WS - Weakly Supervised Learning](#weakly-supervised)
  - [SSL - Semi-supervised Learning](#semi-supervised)
  - [USL - Un- and Self-supervised Learning](#unsupervised-learning)
  - [CL - Collaborative Learning](#mutual-learning)
  - [MTL - Multi-task Learning](#multitask-learning)
  - [AD - Anomaly Detection](#anomaly-detection)
  - [RL - Reinforcement Learning](#reinforcement-learning)
    - [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)
- [Applications](#applications)
  - [Applications: Medical Imaging](#applications-medical-imaging)
  - [Applications: X-ray Imaging](#applications-x-ray-imaging)
  - [Applications: Image Registration](#applications-image-registration)
  - [Applications: Video and Motion](#applications-video-and-motion)
  - [Applications: Denoising and Superresolution](#application-denoising-and-superresolution)
  - [Applications: Inpainting](#applications-inpainting)
  - [Applications: Photography](#applications-photography)
  - [Applications: Misc](#applications-misc)
- [Software](#software)
- [Overview](#overview)
- [Opinions](#opinions)
- [Future of AI](#future-of-ai) 



Notations

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) Checked

![#ff0000](https://placehold.it/15/ff0000/000000?text=+) To Check


# Neural Networks


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Models
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[1998 - **[LeNet]**: Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)


[2012 - **[AlexNet]** ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2013 - Learning Hierarchical Features for Scene Labeling](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)

[2013 - **[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

<sup>Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012---achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also compare R-CNN to OverFeat, a recently proposed sliding-window detector based on a similar CNN architecture. We find that R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset.</sup>

<img src="http://3.bp.blogspot.com/-aM69pqJLP9k/VT2927f8WmI/AAAAAAAAAv8/7S49kEq5Ss0/s1600/%E6%93%B7%E5%8F%96.PNG" width="400">

[2014 - **[OverFeat]**: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229.pdf)

[2014 - **[Seq2Seq]**: Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)


[2014 - **[VGG]** Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<sub>In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.</sub>

<img src="https://cdn-images-1.medium.com/max/1000/1*HzxRI1qHXjiVXla-_NiMBA.png" width="350">


[2014 - **[GoogleNet]** Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<sub>We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.</sub>

<img src="https://miro.medium.com/max/5176/1*ZFPOSAted10TPd3hBQU8iQ.png" width="350">

[2015 - **[ResNet]** Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<sub>Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.</sub>

<img src="https://miro.medium.com/max/3048/1*6hF97Upuqg_LdsqWY6n_wg.png" width="350">

[2015 - Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)

<sub>Convolutional Neural Networks define an exceptionally powerful class of models, but are still limited by the lack of ability to be spatially invariant to the input data in a computationally and parameter efficient manner. In this work we introduce a new learnable module, the Spatial Transformer, which <b>explicitly allows the spatial manipulation</b> of data within the network. <b>This differentiable module</b> can be inserted into existing convolutional architectures, giving neural networks the ability to actively spatially transform feature maps, conditional on the feature map itself, without any extra training supervision or modification to the optimisation process. We show that the use of spatial transformers results in models which learn <b>invariance to translation, scale, rotation and more generic warping</b>, resulting in state-of-the-art performance on several benchmarks, and for a number of classes of transformations.</sub>

<img src="https://miro.medium.com/max/1104/0*n3FxIWWb46ARPww-" width="350">

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


[2019 - **[EfficientNet]**: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://miro.medium.com/max/4044/1*xQCVt1tFWe7XNWVEmC6hGQ.png" width="350">

[2020 - Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis](https://arxiv.org/abs/2002.08725)

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_14/project_408932/images/x1.png" width="350">

[2020 - **[NeRF]**: Representing Scenes as Neural Radiance Fields for View Synthesis ](https://arxiv.org/abs/2003.08934)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://uploads-ssl.webflow.com/51e0d73d83d06baa7a00000f/5e700ef6067b43821ed52768_pipeline_website-01-p-800.png" width="350">



<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
### Multi-level
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
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

[2019 - **[LEDNet]**: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423v1)

<img src="http://www.programmersought.com/images/387/eb5e83159442106d19fbd79698e299eb.png" width="300">

[2019 - Feature Pyramid Encoding Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1909.08599v1)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_290654%2Fproject_390693%2Fimages%2FFPENet.png" width="350">

[2019 - Efficient Segmentation: Learning Downsampling Near Semantic Boundaries](https://arxiv.org/abs/1907.07156)

<img src="https://images.deepai.org/converted-papers/1907.07156/x5.png" width="250">

[2019 - PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)

<img src="https://media.arxiv-vanity.com/render-output/1976701/x3.png" width="300">

[2019 - Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://raw.githubusercontent.com/facebookresearch/FixRes/master/image/image2.png" width="350">

> This paper first shows that existing augmentations induce a significant discrepancy between the typical size of the objects seen by the classifier at train and test time. 

> We experimentally validate that, for a target test resolu- tion, using a lower train resolution offers better classification at test time.


<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
### Context and Attention
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->

[2016 - Image Captioning with Semantic Attention](https://arxiv.org/abs/1603.03925)

<img src="http://cdn-ak.f.st-hatena.com/images/fotolife/P/PDFangeltop1/20160406/20160406161035.png" width="350">

[2018 - **[EncNet]** Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904v1) [[github]](https://github.com/zhanghang1989/PyTorch-Encoding)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZS4gdvv26N8N7dpr92pPoHmVP3RQ8ztddravjJlwHr1Sw5fCT" width="400">

[2018 - Tell Me Where to Look: Guided Attention Inference Network](https://arxiv.org/abs/1802.10171)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_55108%2Fproject_88090%2Fimages%2Fx1.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
### Composition
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->


[2005 - Image Parsing: Unifying Segmentation, Detection, and Recognition](https://link.springer.com/article/10.1007/s11263-005-6642-x)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)


[2013 - Complexity of Representation and Inference in Compositional Models with Part Sharing](https://arxiv.org/abs/1301.3560)


[2017 - Interpretable Convolutional Neural Networks](https://arxiv.org/abs/1710.00935)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://miro.medium.com/max/2712/0*DGs0o1DFHCaCMZvY" width="350">

[2019 - Local Relation Networks for Image Recognition ](https://arxiv.org/pdf/1904.11491.pdf)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_10859%2Fproject_356834%2Fimages%2Fx1.png" width="350">

[2017 - Teaching Compositionality to CNNs](https://www.semanticscholar.org/paper/Teaching-Compositionality-to-CNNs-Stone-Wang/3726b82007512a15a530fd1adad57af58a9abb62)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://www.vicarious.com/wp-content/uploads/2017/10/compositionality3.png" width="350">


[2020 - Concept Bottleneck Models](https://arxiv.org/abs/2007.04612)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://images.deepai.org/converted-papers/2007.04612/figures/teaser.png" width="300">

<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
### Capsule Networks
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->


[2017 - Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://cdn-images-1.medium.com/fit/t/1600/480/0*9fvb_xaSSqW7XVb_.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Optimization and Regularisation
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
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

[2017 - The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning)

> (i) Adaptive methods find solutions that generalize worse than those found by non-adaptive methods.

> (ii) Even when the adaptive methods achieve
the same training loss or lower than non-adaptive methods, the development or test performance
is worse.

> (iii) Adaptive methods often display faster initial progress on the training set, but their
performance quickly plateaus on the development set. 

> (iv) Though conventional wisdom suggests
that Adam does not require tuning, we find that tuning the initial learning rate and decay scheme for
Adam yields significant improvements over its default settings in all cases.

DARTS: Differentiable Architecture Search
https://arxiv.org/abs/1806.09055
 
[**Bag of Tricks** for Image Classification with Convolutional Neural Networks](htps://arxiv.org/abs/1812.01187v1)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2018 - **Tune**: A Research Platform for Distributed Model Selection and Training](https://arxiv.org/abs/1807.05118) [[github]](https://github.com/ray-project/ray/tree/master/python/ray/tune)

[2017 - Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179)

[2017 - Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2018 - Error Forward-Propagation: Reusing Feedforward Connections to Propagate Errors in Deep Learning](https://arxiv.org/abs/1808.03357)

[2018 - An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162v1)

<img src="https://i.pinimg.com/564x/36/bb/e4/36bbe4d951a1c100714ea7baa43e0e44.jpg" width="350">

[2018 - A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2019 - Training Neural Networks with Local Error Signals](https://arxiv.org/abs/1901.06656) [[github]](https://github.com/anokland/local-loss)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2019 - Switchable Normalization for Learning-to-Normalize Deep Representation](https://arxiv.org/abs/1907.10473)

<img src="http://luoping.me/post/family-normalization/SN.png" width="350">

[2019 - Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)

[2019 - Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

<img src="https://www.pyimagesearch.com/wp-content/uploads/2019/07/keras_clr_triangular2.png" width="350">

[2020 - Fantastic Generalization Measures and Where to Find Them](https://arxiv.org/abs/1912.02178)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

> The most direct and principled approach for studying
generalization in deep learning is to prove a **generalization bound** which is typically an upper
bound on the test error based on some quantity that can be calculated on the training set.

> **Kendall’s Rank-Correlation Coefficient**: Given a set of models
resulted by training with hyperparameters in the set Θ, their associated generalization gap {g(θ)| θ ∈
Θ}, and their respective values of the measure {µ(θ)| θ ∈ Θ}, our goal is to analyze how consistent
a measure (e.g. L2 norm of network weights) is with the empirically observed generalization. 
If complexity and generalization are independent, the coefficient becomes zero

> **VC-dimension** as well as the number of parameters are **negatively correlated** with
generalization gap which confirms the widely known empirical observation that overparametrization
improves generalization in deep learning.

> These results confirm the general understanding that larger margin, **lower cross-entropy** and higher entropy would
lead to **better generalization**

> we observed that the **initial phase** (to reach cross-entropy value of 0.1) of the optimization is **negatively
correlated** with the ??speed of optimization?? (error?) for both τ and Ψ. This would suggest that the **difficulty
of optimization** during the initial phase of the optimization **benefits the final generalization**.

> Towards the end of the training, the variance of the gradients also
captures a particular type of “flatness” of the local minima. This measure is surprisingly predictive
of the generalization both in terms of τ and Ψ, and more importantly, is positively correlated across
every type of hyperparameter. 

> There are **mixed** results about how the **optimization speed** is relevant to generalization. On one hand
we know that adding Batch Normalization or using shortcuts in residual architectures help both
optimization and generalization.On the other hand, there are empirical results showing that adaptive
optimization methods that are faster, usually generalize worse (Wilson et al., 2017b).

> Based on empirical observations made by the community as a whole, the canonical ordering we give
to each of the hyper-parameter categories are as follows:
> 1. Batchsize: smaller batchsize leads to smaller generalization gap
> 2. Depth: deeper network leads to smaller generalization gap
> 3. Width: wider network leads to smaller generalization gap
> 4. Dropout: The higher the dropout (≤ 0.5) the smaller the generalization gap
> 5. Weight decay: The higher the weight decay (smaller than the maximum for each optimizer)
the smaller the generalization gap
> 6. Learning rate: The higher the learning rate (smaller than the maximum for each optimizer)
the smaller the generalization gap
> 7. Optimizer: Generalization gap of Momentum SGD < Generalization gap of Adam < Generalization gap of RMSProp

[2020 - Descending through a Crowded Valley -- Benchmarking Deep Learning Optimizers](https://arxiv.org/abs/2007.01547)

<img src="https://user-images.githubusercontent.com/544269/95753705-f18fbe80-0cdc-11eb-9499-6bf22fa456e0.png" width="250">

[2020 - Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth](https://arxiv.org/abs/2010.15327)
<img src="https://i.pinimg.com/564x/15/8b/af/158baf37ea0b6f05cc0b0d1fd2f364d2.jpg" width="250">

<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Pruning and Compression
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->


[2013 - Do Deep Nets Really Need to be Deep?](https://arxiv.org/abs/1312.6184)

[2015 - Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

<img src="https://xmfbit.github.io/img/paper-pruning-network-demo.png" width="350">

[2015 - Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)

<img src="https://anandj.in/wp-content/uploads/dc.png" width="350">

[2015 - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2017 - Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519) - [[github]](https://github.com/liuzhuang13/slimming)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://user-images.githubusercontent.com/8370623/29604272-d56a73f4-879b-11e7-80ea-0702de6bd584.jpg" width="350">

[2018 - Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTRq9LlknFNmCyXoKoEVqfMX3JgP66T5Ezpbh4FF9xUVLBU0jO6" width="350">


> For all state-of-the-art structured pruning algorithms we examined, fine-tuning a pruned model only gives
comparable or worse performance than training that model with randomly initialized weights. For pruning algorithms which assume a predefined target network architecture, one can get rid of the full pipeline and directly train the target network from scratch.

> Our observations are consistent for multiple network architectures, datasets, and tasks, which imply that: 

> 1) training a large, over-parameterized model is often not necessary to obtain an efficient final model

> 2) learned “important” weights of the large model are typically not useful for the small pruned
model

> 3) the pruned architecture itself, rather than a set of inherited “important”
weights, is more crucial to the efficiency in the final model, which suggests that in
some cases pruning can be useful as an architecture search paradigm.

[2018 - Slimmable Neural Networks](https://arxiv.org/abs/1812.08928)

<img src="https://user-images.githubusercontent.com/22609465/50390872-1b3fb600-0702-11e9-8034-d0f41825d775.png" width="350">


[2019 - Universally Slimmable Networks and Improved Training Techniques](https://arxiv.org/abs/1903.05134)

<img src="https://user-images.githubusercontent.com/22609465/54562571-45b5ae00-4995-11e9-8984-49e32d07e325.png" width="300">



[2019 - The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://miro.medium.com/max/2916/1*IraKnowykSyMZtrW1dJOVA.png" width="350">

> Based on these results, we articulate the lottery ticket hypothesis: dense, randomly-initialized, feed-forward
networks contain subnetworks (winning tickets) that—when trained in isolation—
reach test accuracy comparable to the original network in a similar number of
iterations.

> The winning tickets we find have won the **initialization** lottery: their
connections have initial weights that make training particularly effective.

[2019 - AutoSlim: Towards One-Shot Architecture Search for Channel Numbers](https://arxiv.org/abs/1903.11728)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_372245%2Fimages%2Fx1.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Analysis and Interpretability
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->
<!--- ------------------------------------------------------------------------------- -->


[2015 - Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078)

[2016 - Discovering Causal Signals in Images](https://arxiv.org/abs/1605.08179)

<img src="https://2.bp.blogspot.com/-ZS7WHgo3f9U/XD26idxNEEI/AAAAAAAABl8/DipJ1Fm3ZK0C3tXhu03psC4nByTlID-sQCLcBGAs/s1600/Screen%2BShot%2B2019-01-15%2Bat%2B19.48.13.png" width="400">

[2016 - **[Grad-CAM]**: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) [[github]](https://github.com/jacobgil/pytorch-grad-cam)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSR95EORUuYqxk3MtWiiQoDmHnizHVPxr1JnGVbfWJrHesJjZln&s" width="350">

[2017 - Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)

<img src="https://github.com/tomgoldstein/loss-landscape/raw/master/doc/images/resnet56_noshort_small.jpg" width="250">

[2019 - **[SURVEY]** Visual Analytics in Deep Learning: An Interrogative Survey for the Next Frontiers](https://ieeexplore.ieee.org/document/8371286)


[2018 - GAN Dissection: Visualizing and Understanding Generative Adversarial Networks](https://arxiv.org/abs/1811.10597v1)

<img src="https://i.pinimg.com/originals/5a/df/e9/5adfe97e85a9023d7f11499ab57e7daf.png" width="350">

[2018 Interactive tool](https://gandissect.csail.mit.edu/)

[**[Netron ]** Visualizer for deep learning and machine learning models](https://github.com/lutzroeder/Netron)

<img src="https://raw.githubusercontent.com/lutzroeder/netron/master/media/screenshot.png" width="400">

[2019 - **[Distill]**: Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/)

[2019 - On the Units of GANs](https://arxiv.org/abs/1901.09887)

<img src="https://neurohive.io/wp-content/uploads/2018/12/unit-distr-770x382.jpg" width="350">

[2020 - Actionable Attribution Maps for Scientific Machine Learning](https://arxiv.org/abs/2006.16533)

<img src="https://i.pinimg.com/564x/45/b0/51/45b05100bff866b98ff050433d4e64dd.jpg" width="350">

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
## Anomaly Detection
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2009 - Anomaly Detection: A Survey](https://www.vs.inf.ethz.ch/edu/HS2011/CPS/papers/chandola09_anomaly-detection-survey.pdf)

[2017 - Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921v1)


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Transfer Learning
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

- [Transfer Learning](https://github.com/axruff/TransferLearning)
- [Domain Adaptation](https://github.com/axruff/TransferLearning)
- [Domain Randomization](https://github.com/axruff/TransferLearning#domain-randomization)
- [Style Transfer](https://github.com/axruff/TransferLearning#style-transfer)

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Generative Modelling
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

- [Generative Models](https://github.com/axruff/TransferLearning#generative-models)

 <!--- ===================================================================================
 <!---   ____                 _                                       _              _ 
 <!---  / ___|  ___ _ __ ___ (_)      ___ _   _ _ __   ___ _ ____   _(_)___  ___  __| |
 <!---  \___ \ / _ \ '_ ` _ \| |_____/ __| | | | '_ \ / _ \ '__\ \ / / / __|/ _ \/ _` |
 <!---   ___) |  __/ | | | | | |_____\__ \ |_| | |_) |  __/ |   \ V /| \__ \  __/ (_| |
 <!---  |____/ \___|_| |_| |_|_|     |___/\__,_| .__/ \___|_|    \_/ |_|___/\___|\__,_|
 <!---                                         |_|                                     
<!--- ===================================================================================


<!--- ------------------------------------------------------------------------------- -->
## Weakly Supervised
<!--- ------------------------------------------------------------------------------- -->


[2015 - Constrained Convolutional Neural Networks for Weakly Supervised Segmentation](https://arxiv.org/abs/1506.03648)

<img src="https://people.eecs.berkeley.edu/~pathak/images/iccv15.png" width="300">

[2018 - Deep Learning with Mixed Supervision for Brain Tumor Segmentation](https://arxiv.org/abs/1812.04571)
<img src="https://www.spiedigitallibrary.org/ContentImages/Journals/JMIOBU/6/3/034002/WebImages/JMI_6_3_034002_f001.png" widtg="350">

[2019 - Localization with Limited Annotation for Chest X-rays](https://arxiv.org/abs/1909.08842v1)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOFaxbxbwuKln6SgbFVWyVP2A7tj-CTQe05isVKH3gb1IGqg84ig&s" width="350">

[2019 - Doubly Weak Supervision of Deep Learning Models for Head CT](https://jdunnmon.github.io/miccai_crc.pdf)

<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-32248-9_90/MediaObjects/490277_1_En_90_Fig2_HTML.png" width="350">

[2019 - Training Complex Models with Multi-Task Weak Supervision](https://www.ncbi.nlm.nih.gov/pubmed/31565535)

<sub>As machine learning models continue to increase in complexity, collecting large hand-labeled training sets has become one of the biggest roadblocks in practice. Instead, weaker forms of supervision that provide noisier but cheaper labels are often used. However, these weak supervision sources have diverse and unknown accuracies, may output correlated labels, and may label different tasks or apply at different levels of granularity. We propose a framework for integrating and modeling such weak supervision sources by viewing them as labeling different related sub-tasks of a problem, which we refer to as the multi-task weak supervision setting. We show that by solving a matrix completion-style problem, we can recover the accuracies of these multi-task sources given their dependency structure, but without any labeled data, leading to higher-quality supervision for training an end model. Theoretically, we show that the generalization error of models trained with this approach improves with the number of unlabeled data points, and characterize the scaling with respect to the task and dependency structures. On three fine-grained classification problems, we show that our approach leads to average gains of 20.2 points in accuracy over a traditional supervised approach, 6.8 points over a majority vote baseline, and 4.1 points over a previously proposed weak supervision method that models tasks separately.</sub>

<img src="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6765366/bin/nihms-1037643-f0001.jpg" width="350">

[2020 - Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods](https://arxiv.org/abs/2002.11955)


<!--- ------------------------------------------------------------------------------- -->
## Semi Supervised
<!--- ------------------------------------------------------------------------------- -->


[2014 - Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/abs/1406.6909)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 

<img src="https://www.inference.vc/content/images/2017/05/Screen-Shot-2017-05-11-at-9.31.37-AM.png" width="300">
 
[2017 - Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2)[[github]](https://github.com/zhunzhong07/Random-Erasing)

<img src="https://github.com/zhunzhong07/Random-Erasing/raw/master/all_examples-page-001.jpg" width="350">

[2017 - Smart Augmentation - Learning an Optimal Data Augmentation Strategy](https://arxiv.org/abs/1703.08383)

[2017 - Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)


[2018 - **[Survey]**: Not-so-supervised: a survey of semi-supervised, multi-instance, and transfer learning in medical image analysis](https://arxiv.org/abs/1804.06353)

[2018 - Albumentations: fast and flexible image augmentations](https://arxiv.org/abs/1809.06839) - [[github]](https://github.com/albu/albumentations)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)


[2018 - Data Augmentation by Pairing Samples for Images Classification](https://arxiv.org/abs/1801.02929)

[2018 - **[AutoAugment]**: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)

[2018 - Synthetic Data Augmentation using GAN for Improved Liver Lesion Classification](https://arxiv.org/abs/1801.02385)

[2018 - GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks](https://arxiv.org/abs/1810.10863)

[2019 - **[UDA]**: Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848) - [[github]](https://github.com/google-research/uda)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<sub>Common among recent approaches is the use of <b>consistency training</b> on a large amount of unlabeled data to constrain model predictions to be invariant to input noise. In this work, we present a new perspective on how to effectively noise unlabeled examples and argue that the quality of noising, specifically those produced by <b>advanced data augmentation methods</b>, plays a <b>crucial role</b> in semi-supervised learning. Our method also combines well with <b>transfer learning</b>, e.g., when finetuning from BERT, and yields improvements in high-data regime, such as ImageNet, whether when there is only 10% labeled data or when a full labeled set with 1.3M extra unlabeled examples is used.</sub>

<img src="https://camo.githubusercontent.com/0896cb65f9a87983bee3f2f71f3c064c33216413/68747470733a2f2f692e696d6775722e636f6d2f4c38476b3634622e706e67" width="350">

[2019 - **[MixMatch]**: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<sub>Semi-supervised learning has proven to be a powerful paradigm for leveraging unlabeled data to mitigate the reliance on large labeled datasets. In this work, we unify the current dominant approaches for semi-supervised learning to produce a new algorithm, MixMatch, that works by <b>guessing low-entropy labels</b> for data-augmented unlabeled examples and <b>mixing labeled and unlabeled</b> data using MixUp. We show that MixMatch obtains state-of-the-art results by a large margin across many datasets and labeled data amounts.</sub>

<img src="https://miro.medium.com/max/1402/1*i4OfXztihCXgrxR52ZlowQ.png" width="350">


[2019 - **[RealMix]**: Towards Realistic Semi-Supervised Deep Learning Algorithms](https://arxiv.org/abs/1912.08766v1)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_14/project_402411/images/RealMix.png" width="350">


[2019 - Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules](https://arxiv.org/abs/1905.05393) [[github]](https://github.com/arcelien/pba)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2019 - **[AugMix]**: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781v1) [[github]](https://github.com/google-research/augmix)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://pythonawesome.com/content/images/2019/12/AugMix.jpg" width="350">

[2019 - Self-training with **[Noisy Student]** improves ImageNet classification](https://arxiv.org/abs/1911.04252)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<sub>We present a simple <b>self-training</b> method that achieves 88.4% top-1 accuracy on ImageNet, which is 2.0% better than the state-of-the-art model that requires 3.5B weakly labeled Instagram images. On <b>robustness test sets</b>, it improves ImageNet-A top-1 accuracy from 61.0% to 83.7%.
To achieve this result, we first train an EfficientNet model on labeled ImageNet images and <b>use it as a teacher to generate pseudo labels</b> on 300M unlabeled images. We then train a larger EfficientNet as <b>a student model on the combination of labeled and pseudo labeled images</b>. We <b>iterate this process</b> by putting back the student as the teacher. During the generation of the pseudo labels, the teacher is not noised so that the pseudo labels are as accurate as possible. However, during the learning of the student, we <b>inject noise such as dropout, stochastic depth and data augmentation</b> via RandAugment to the student so that the student generalizes better than the teacher.</sub>

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_23782%2Fproject_397607%2Fimages%2Fx1.png" width="250">


[2020 - Rain rendering for evaluating and improving robustness to bad weather](https://arxiv.org/abs/2009.03683)

<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs11263-020-01366-3/MediaObjects/11263_2020_1366_Fig13_HTML.png" width="350">


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Unsupervised Learning
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2019 - Greedy InfoMax for Biologically Plausible Self-Supervised Representation Learning](https://arxiv.org/abs/1905.11786)

[2019 - Unsupervised Learning via Meta-Learning](https://arxiv.org/abs/1810.02334)

[2019 - Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)

[2019 - **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

<sub>We present Momentum Contrast (MoCo) for unsupervised <b>visual representation learning</b>. From a perspective on contrastive learning as dictionary look-up, we build a <b>dynamic dictionary with a queue</b> and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. MoCo provides competitive results under the common linear protocol on ImageNet classification. More importantly, the representations learned by MoCo transfer well to downstream tasks. MoCo can outperform its supervised pre-training counterpart in 7 detection/segmentation tasks on PASCAL VOC, COCO, and other datasets, sometimes surpassing it by large margins. This suggests that <b>the gap between unsupervised and supervised</b> representation learning has been largely closed in many vision tasks.</sub>

<img src="https://pythonawesome.com/content/images/2020/03/MoCo.png" width="350">

[2019 - Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey](https://arxiv.org/abs/1902.06162)

[2020 - A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://miro.medium.com/max/8300/1*1uaA1tE5PDnVpSljxSTEoQ.png" width="250">

[2020 - **::SURVEY::** Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey](https://arxiv.org/abs/1902.06162)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2020 - **[NeurIPS 2020 Workshop]**: Self-Supervised Learning - Theory and Practice](https://sslneuips20.github.io/pages/Accepted%20Paper.html)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<!--- ------------------------------------------------------------------------------- -->
## Mutual Learning
<!--- ------------------------------------------------------------------------------- -->

[2017 - Deep Mutual Learning](https://arxiv.org/abs/1706.00384)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_1989/project_107452/images/x1.png" width="350">


[2019 - Feature Fusion for Online Mutual Knowledge Distillation ](https://arxiv.org/abs/1904.09058)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_228887%2Fproject_355567%2Fimages%2Foverallprocess.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
## Multitask Learning
<!--- ------------------------------------------------------------------------------- -->
[2017 - An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098)

<sub>Multi-task learning (MTL) has led to successes in many applications of machine learning, from natural language processing and speech recognition to computer vision and drug discovery. This article aims to give a general overview of MTL, particularly in deep neural networks. It introduces the two most common methods for MTL in Deep Learning, gives an overview of the literature, and discusses recent advances. In particular, it seeks to help ML practitioners apply MTL by shedding light on how MTL works and providing guidelines for choosing appropriate auxiliary tasks.</sub>

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Reinforcement Learning
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
### Inverse Reinforcement Learning
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2019 - On the Feasibility of Learning, Rather than Assuming, Human Biases for Reward Inference](https://arxiv.org/abs/1906.09624)

<img src="https://i.pinimg.com/564x/cf/0a/08/cf0a0859ca749d389c5ccc24d20fd1a3.jpg" widtg="350">

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

<img src="https://www.shapenet.org/resources/images/logo.png" width="350">

[ShapeNet: An Information-Rich 3D Model Repository](https://arxiv.org/abs/1512.03012)


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

[**[MegaDepth]**: Learning Single-View Depth Prediction from Internet Photos](https://research.cs.cornell.edu/megadepth/)

<img src="https://research.cs.cornell.edu/megadepth/demo2.png" width="350">

[Microsoft **[COCO]**: Common Objects in Context](https://arxiv.org/abs/1405.0312)[[website]](https://cocodataset.org/#home)

<img src="https://cdn.slidesharecdn.com/ss_thumbnails/cocodataset-190410053316-thumbnail-4.jpg?cb=1554874430" width="350">


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



<!--- █████╗ ██████╗ ██████╗ ██╗     ██╗ ██████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
<!---██╔══██╗██╔══██╗██╔══██╗██║     ██║██╔════╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
<!---███████║██████╔╝██████╔╝██║     ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
<!---██╔══██║██╔═══╝ ██╔═══╝ ██║     ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
<!---██║  ██║██║     ██║     ███████╗██║╚██████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
<!---╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
                                                                                          
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

[2020 - **[TorchIO]**: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning](https://arxiv.org/abs/2003.04696) [[github]](https://github.com/fepegar/torchio)

[2020 - Reconstructing lost BOLD signal in individual participants using deep machine learning](https://www.nature.com/articles/s41467-020-18823-9#disqus_thread)

<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41467-020-18823-9/MediaObjects/41467_2020_18823_Fig1_HTML.png?as=webp" width="350">


<!--- ------------------------------------------------------------------------------- -->
# Applications: X-ray Imaging
<!--- ------------------------------------------------------------------------------- -->

Low-dose X-ray tomography through a deep convolutional neural network
https://www.nature.com/articles/s41598-018-19426-7

*In synchrotron-based XRT, CNN-based processing improves the SNR in the data by an order of magnitude, which enables low-dose fast acquisition of radiation-sensitive samples*

[2020 - Deep Learning Techniques for Inverse Problems in Imaging](https://ieeexplore.ieee.org/abstract/document/9084378)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

# Applications: Image Registration

[2014 - Do Convnets Learn Correspondence?](https://arxiv.org/abs/1411.1091)

<img src="https://i.pinimg.com/564x/73/f2/12/73f212fc05c87151112a8381f1904cc0.jpg" width="350">

[2016 - Universal Correspondence Network](https://arxiv.org/abs/1606.03558)

<sub>We present a deep learning framework for accurate visual correspondences and demonstrate its effectiveness for both geometric and semantic matching, spanning across rigid motions to intra-class shape or appearance variations. In contrast to previous CNN-based approaches that optimize a surrogate patch similarity objective, we use deep metric learning to <b>directly learn a feature space</b> that preserves either geometric or semantic similarity. Our fully convolutional architecture, along with a <b>novel correspondence contrastive loss</b> allows faster training by effective reuse of computations, accurate gradient computation through the use of thousands of examples per image pair and faster testing with O(n) feed forward passes for n keypoints, instead of O(n2) for typical patch similarity methods. We propose a <b>convolutional spatial transformer</b> to mimic patch normalization in traditional features like SIFT, which is shown to dramatically boost accuracy for semantic correspondences across intra-class shape variations. Extensive experiments on KITTI, PASCAL, and CUB-2011 datasets demonstrate the significant advantages of our features over prior works that use either hand-constructed or learned features.</sub>

<img src="https://cvgl.stanford.edu/projects/ucn/imgs/overview-nn_sm.png" width="380">

[2016 - Learning Dense Correspondence via 3D-guided Cycle Consistency](https://arxiv.org/abs/1604.05383)

<sub>Discriminative deep learning approaches have shown impressive results for problems where human-labeled ground truth is plentiful, but what about tasks where labels are difficult or impossible to obtain? This paper tackles one such problem: establishing dense visual correspondence across different object instances. For this task, although we do not know what the ground-truth is, we know it should be consistent across instances of that category. <b>We exploit this consistency as a supervisory signal</b> to train a convolutional neural network to predict cross-instance correspondences between pairs of images depicting objects of the same category. For each pair of training images we find an appropriate 3D CAD model and render two synthetic views to link in with the pair, establishing a correspondence flow 4-cycle. We use ground-truth synthetic-to-synthetic correspondences, provided by the rendering engine, to train a ConvNet to predict synthetic-to-real, real-to-real and real-to-synthetic correspondences that are cycle-consistent with the ground-truth. At test time, no CAD models are required. We demonstrate that our end-to-end trained ConvNet supervised by cycle-consistency outperforms state-of-the-art pairwise matching methods in correspondence-related tasks.</sub>

<img src="https://people.eecs.berkeley.edu/~tinghuiz/projects/learnCycle/images/teaser.png" width="350">

[2017 - Convolutional neural network architecture for geometric matching](https://arxiv.org/abs/1703.05593)[[github]](https://github.com/ignacio-rocco/cnngeometric_pytorch)

<img src="https://www.di.ens.fr/willow/research/cnngeometric/images/diagram.png" width="350">

[2018 - **[DGC-Net]**: Dense Geometric Correspondence Network](https://arxiv.org/abs/1810.08393) [[github]](https://github.com/AaltoVision/DGC-Net)

<img src="https://i.pinimg.com/564x/53/76/df/5376df64a3ef357a7306a2d8f96ac407.jpg" width="350">

[2018 - An Unsupervised Learning Model for Deformable Medical Image Registration](https://arxiv.org/abs/1802.02604)

<img src="https://vitalab.github.io/article/images/unsupervised-registration/figure2.png" width="350">

[2018 - VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231) [[github]](https://github.com/voxelmorph/voxelmorph)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_388296%2Fimages%2Fx2.png" width="300">

[2019 - A Deep Learning Framework for Unsupervised Affine and Deformable Image Registration](https://arxiv.org/abs/1809.06130)

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1361841518300495-gr12.jpg" width="250">

[2020 - RANSAC-Flow: generic two-stage image alignment](https://arxiv.org/abs/2004.01526)

<img src="http://imagine.enpc.fr/~shenx/RANSAC-Flow/img/overview.jpg" width="350">



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

[2017 - "Zero-Shot" Super-Resolution using Deep Internal Learning](https://arxiv.org/abs/1712.06087)

<img src="https://i.pinimg.com/564x/5c/80/fc/5c80fcbf98bad9c0c9aa8abb0f142724.jpg" width="350">

[2018 - Residual Dense Network for Image Restoration](https://arxiv.org/abs/1812.10477v1) [[github]](https://github.com/yulunzhang/RDN)

<img src="https://i.pinimg.com/564x/12/7e/b4/127eb4dfbf482db1ba436ea960821fae.jpg" width="350" >

[2018 - Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

<img src="https://miro.medium.com/max/1200/1*pOUFUNgBQwSh3EUbJI2m3Q.png" width="350">


<!--- ------------------------------------------------------------------------------- -->
# Applications: Inpainting
<!--- ------------------------------------------------------------------------------- -->

[2018 - Image Inpainting for Irregular Holes Using Partial Convolutions](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf)[[github official]](https://github.com/NVIDIA/partialconv), [[github]](https://github.com/MathiasGruber/PConv-Keras)

<img src="https://i.pinimg.com/564x/63/fa/a3/63faa338eba25225c7e84f1d3bad74d3.jpg" width="350">

[2017 - Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)[[github]](https://github.com/satoshiiizuka/siggraph2017_inpainting)

<img src="http://iizuka.cs.tsukuba.ac.jp/projects/completion/images/teaser/flickr_4_o.png" width=350>

[2017 - Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892)[[github]](https://github.com/JiahuiYu/generative_inpainting)

<img src="https://user-images.githubusercontent.com/22609465/35364552-6e9dfab0-0135-11e8-8bc1-5f370a9f4b0a.png" width="350">

[2018 - Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)

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

[2017 - Learning Fashion Compatibility with Bidirectional LSTMs](https://arxiv.org/abs/1707.05691)[[github]](https://github.com/xthan/polyvore)

<img src="https://i.pinimg.com/564x/4b/af/fc/4baffc51cc87b1354ed9e88cc8bd534e.jpg" width="350">

[2020 - A Systematic Literature Review on the Use of Deep Learning in Software Engineering Research](https://arxiv.org/abs/2009.06520)
<img src="https://pbs.twimg.com/media/Ei2XG-EX0AA2xzD?format=jpg&name=large" width="350">

[2020 - Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

<!--- ------------------------------------------------------------------------------- -->
# Software
<!--- ------------------------------------------------------------------------------- -->

[**Caffe**: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093)

[**Tune**: A Research Platform for Distributed Model Selection and Training (2018)](https://arxiv.org/abs/1807.05118) [[github]](https://github.com/ray-project/ray/tree/master/python/ray/tune)

[**Glow**: Compiler for Neural Network hardware accelerators](https://github.com/pytorch/glow)

<img src="https://github.com/pytorch/glow/raw/master/docs/3LevelIR.png" width="400">

[**Lucid**: A collection of infrastructure and tools for research in neural network interpretability](https://github.com/tensorflow/lucid)

[**PySyft**: A generic framework for privacy preserving deep learning](https://arxiv.org/abs/1811.04017) [[github]](https://github.com/OpenMined/PySyft)

[**Crypten**: A framework for Privacy Preserving Machine Learning](https://crypten.ai/) [[github]](https://github.com/facebookresearch/crypten)

[**[Snorkel]**: Programmatically Building and Managing Training Data](https://www.snorkel.org/)

[**[Netron ]** Visualizer for deep learning and machine learning models](https://github.com/lutzroeder/Netron)

<img src="https://raw.githubusercontent.com/lutzroeder/netron/master/media/screenshot.png" width="400">

[**[Interactive Tools]** for ML, DL and Math](https://github.com/Machine-Learning-Tokyo/Interactive_Tools)

[**[Efemarai]**](https://efemarai.com/)

[**[mlflow]** - An open source platform for the machine learning lifecycle](https://mlflow.org/)

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Overview
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2016 - An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)

[2017 - Revisiting Unreasonable Effectiveness of Data in Deep Learning Era](https://arxiv.org/abs/1707.02968)

[2019 - High-performance medicine: the convergence of human and artificial intelligence](https://www.nature.com/articles/s41591-018-0300-7)

[2020 - Maithra Raghu, Eric Schmidt. A Survey of Deep Learning for Scientific Discovery](https://arxiv.org/abs/2003.11755v1)



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

<sub>This is an opinion paper about the strengths and weaknesses of Deep Nets for vision. They are at the center of recent progress on artificial intelligence and are of growing importance in cognitive science and neuroscience. They have enormous successes but also clear limitations. There is also only partial understanding of their inner workings. It seems unlikely that Deep Nets in their current form will be the best long-term solution either for building general purpose intelligent machines or for understanding the mind/brain, but it is likely that many aspects of them will remain. At present Deep Nets do very well on specific types of visual tasks and on specific benchmarked datasets. But Deep Nets are much less general purpose, flexible, and adaptive than the human visual system. Moreover, methods like Deep Nets may run into fundamental difficulties when faced with the enormous complexity of natural images which can lead to a combinatorial explosion. To illustrate our main points, while keeping the references small, this paper is slightly biased towards work from our group.</sub>

[2020 - State of AI Report 2020](https://www.stateof.ai/)

<!--- ------------------------------------------------------------------------------- -->
# Future of AI
<!--- ------------------------------------------------------------------------------- -->

[2018 - When Will AI Exceed Human Performance? Evidence from AI Experts](https://arxiv.org/abs/1705.08807)

<img src="https://i.pinimg.com/564x/66/2a/af/662aaf5ca744d6bd7aad44d4a70523a6.jpg" width="200">

[2018 - The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://docs.google.com/document/d/e/2PACX-1vQzbSybtXtYzORLqGhdRYXUqiFsaEOvftMSnhVgJ-jRh6plwkzzJXoQ-sKtej3HW_0pzWTFY7-1eoGf/pub)

<img src="https://www.cser.ac.uk/media/uploads/files/front_cover_malicious_use_square.png" width="200">

[2018 - Deciphering China’s AI Dream: The context, components, capabilities, and consequences of China’s strategy to lead the world in AI](https://www.fhi.ox.ac.uk/deciphering-chinas-ai-dream/)



