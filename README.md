# Machine Learning papers and resources


##### Table of Contents
- [Neural Networks](#neural-networks)
  - [Models](#models)
    - [Multi-level](#multi-level)
    - [Context and Attention](#context-and-attention)
    - [Composition](#composition)
    - [Mutual Learning](#mutual-learning)
  - [Optimization](#optimization)
    - [Optimization and Regularisation](#optimization-and-regularisation)
    - [Pruning, Compression](#pruning-and-compression)
  - [Visualization](#visualization-and-analysis)  
  - [Data Augmentation](#data-augmentation)
- [Segmentation](#segmentation) 
- [Semi and Weak Supervision](#semi-and-weak-supervision)
- [Unsupervised / Self-supervised Learning](#unsupervised-learning)
- [Anomaly Detection](#anomaly-detection)
- [Reinforcement Learning](#reinforcement-learning)
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


[2019 - **[EfficientNet]**: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://miro.medium.com/max/4044/1*xQCVt1tFWe7XNWVEmC6hGQ.png" width="350">

[2020 - Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis](https://arxiv.org/abs/2002.08725)

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_14/project_408932/images/x1.png" width="350">


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
### Context and Attention
<!--- ------------------------------------------------------------------------------- -->

[2016 - Image Captioning with Semantic Attention](https://arxiv.org/abs/1603.03925)

<img src="http://cdn-ak.f.st-hatena.com/images/fotolife/P/PDFangeltop1/20160406/20160406161035.png" width="350">

[2018 - **[EncNet]** Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904v1) [[github]](https://github.com/zhanghang1989/PyTorch-Encoding)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZS4gdvv26N8N7dpr92pPoHmVP3RQ8ztddravjJlwHr1Sw5fCT" width="400">

[2018 - Tell Me Where to Look: Guided Attention Inference Network](https://arxiv.org/abs/1802.10171)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_55108%2Fproject_88090%2Fimages%2Fx1.png" width="350">

<!--- ------------------------------------------------------------------------------- -->
### Composition
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


<!--- ------------------------------------------------------------------------------- -->
### Mutual Learning
<!--- ------------------------------------------------------------------------------- -->

[2017 - Deep Mutual Learning](https://arxiv.org/abs/1706.00384)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_1989/project_107452/images/x1.png" width="350">


[2019 - Feature Fusion for Online Mutual Knowledge Distillation ](https://arxiv.org/abs/1904.09058)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_228887%2Fproject_355567%2Fimages%2Foverallprocess.png" width="350">

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

[2019 - Training Neural Networks with Local Error Signals](https://arxiv.org/abs/1901.06656) [[github]](https://github.com/anokland/local-loss)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2019 - Switchable Normalization for Learning-to-Normalize Deep Representation](https://arxiv.org/abs/1907.10473)

<img src="http://luoping.me/post/family-normalization/SN.png" width="350">

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

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
## Pruning and Compression
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->


[2013 - Do Deep Nets Really Need to be Deep?](https://arxiv.org/abs/1312.6184)

[2015 - Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

<img src="https://xmfbit.github.io/img/paper-pruning-network-demo.png" width="350">

[2015 - Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) ![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

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

[2016 - **[Grad-CAM]**: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) [[github]](https://github.com/jacobgil/pytorch-grad-cam)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSR95EORUuYqxk3MtWiiQoDmHnizHVPxr1JnGVbfWJrHesJjZln&s" width="350">

[Distill: Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/)

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
 
[2017 - Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2)[[github]](https://github.com/zhunzhong07/Random-Erasing)

<img src="https://github.com/zhunzhong07/Random-Erasing/raw/master/all_examples-page-001.jpg" width="350">

[2017 - Smart Augmentation - Learning an Optimal Data Augmentation Strategy](https://arxiv.org/abs/1703.08383)

[2017 - Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

[2018 - Albumentations: fast and flexible image augmentations](https://arxiv.org/abs/1809.06839) [[github]](https://github.com/albu/albumentations)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)


[2018 - Data Augmentation by Pairing Samples for Images Classification](https://arxiv.org/abs/1801.02929)

[2018 - **[AutoAugment]**: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)

[2018 - Synthetic Data Augmentation using GAN for Improved Liver Lesion Classification](https://arxiv.org/abs/1801.02385)

[2018 - GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks](https://arxiv.org/abs/1810.10863)

[2019 - **UDA**: Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)[[github]](https://github.com/google-research/uda)

[2019 - Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules](https://arxiv.org/abs/1905.05393) [[github]](https://github.com/arcelien/pba)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

[2019 - **AugMix**: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781v1) [[github]](https://github.com/google-research/augmix)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://pythonawesome.com/content/images/2019/12/AugMix.jpg" width="350">

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
# Semi and Weak Supervision
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2019 - Localization with Limited Annotation for Chest X-rays](https://arxiv.org/abs/1909.08842v1)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOFaxbxbwuKln6SgbFVWyVP2A7tj-CTQe05isVKH3gb1IGqg84ig&s" width="350">

[2019 - **[RealMix]**: Towards Realistic Semi-Supervised Deep Learning Algorithms](https://arxiv.org/abs/1912.08766v1)
![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)

<img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_14/project_402411/images/RealMix.png" width="350">


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

[2020 - A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)



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
# Inverse Reinforcement Learning
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

[2020 - **[TorchIO]**: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning](https://arxiv.org/abs/2003.04696) [[github]](https://github.com/fepegar/torchio)


<!--- ------------------------------------------------------------------------------- -->
# Applications: X-ray Imaging
<!--- ------------------------------------------------------------------------------- -->

Low-dose X-ray tomography through a deep convolutional neural network
https://www.nature.com/articles/s41598-018-19426-7

*In synchrotron-based XRT, CNN-based processing improves the SNR in the data by an order of magnitude, which enables low-dose fast acquisition of radiation-sensitive samples*


# Applications: Image Registration

[2018 - An Unsupervised Learning Model for Deformable Medical Image Registration](https://arxiv.org/abs/1802.02604)

<img src="https://vitalab.github.io/article/images/unsupervised-registration/figure2.png" width="350">

[2018 - VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231) [[github]](https://github.com/voxelmorph/voxelmorph)

<img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_388296%2Fimages%2Fx2.png" width="300">

[2019 - A Deep Learning Framework for Unsupervised Affine and Deformable Image Registration](https://arxiv.org/abs/1809.06130)

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1361841518300495-gr12.jpg" width="250">

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

[2017 - Learning Fashion Compatibility with Bidirectional LSTMs](https://arxiv.org/abs/1707.05691)[[github]](https://github.com/xthan/polyvore)

<img src="https://i.pinimg.com/564x/4b/af/fc/4baffc51cc87b1354ed9e88cc8bd534e.jpg" width="350">

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
# Future of AI
<!--- ------------------------------------------------------------------------------- -->

[2018 - When Will AI Exceed Human Performance? Evidence from AI Experts](https://arxiv.org/abs/1705.08807)

<img src="https://i.pinimg.com/564x/66/2a/af/662aaf5ca744d6bd7aad44d4a70523a6.jpg" width="200">

[2018 - The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://docs.google.com/document/d/e/2PACX-1vQzbSybtXtYzORLqGhdRYXUqiFsaEOvftMSnhVgJ-jRh6plwkzzJXoQ-sKtej3HW_0pzWTFY7-1eoGf/pub)

<img src="https://www.cser.ac.uk/media/uploads/files/front_cover_malicious_use_square.png" width="200">

[2018 - Deciphering China’s AI Dream: The context, components, capabilities, and consequences of China’s strategy to lead the world in AI](https://www.fhi.ox.ac.uk/deciphering-chinas-ai-dream/)



