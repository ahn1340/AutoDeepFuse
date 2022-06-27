# AutoDeepFuse
Master Project: Neural Architecture Search for Multi-Modal Stream Fusion in Deep Video Action Recognition

Summary: This project aims to discover network architectures that optimally perform fusion of multiple modality
features (RGB, optical flow, pose estimate) that are extracted from spatio-temporal ResNet models pre-trained on the
[Kinetics700](https://arxiv.org/abs/1907.06987) via differentiable architecture search. Architecture search is conducted
under various settings, i.e., different search space, varying optimization strategy, etc., in order to address the problem
of optimization gap and dominating skip-connection, which are some of the well-known drawbacks of DARTS.

Full description of the work can be found in my [thesis][thesis].

[thesis]: https://ahn1340.github.io/pdfs/Master_Thesis_Ahn.pdf

##### Modalities
- [x] Raw RGB frames
- [x] Pose Estimate in the form of object segmentation
- [x] OpticalFlow estimate

##### Operations
- [x]  3x3x3 convolution
- [x]  1x1x1 convolution
- [x]  Depthwise separable convolution
- [x]  (2+1)D convolution
- [x]  Skip connection
- [x]  Zero

##### Datasets
- [x] HMDB
- [x] UCF101

### Packages
- pytorch=1.1.+
- torchvision=0.3.+
- tensorboard
- future
- yaml, pyyaml
- torchsummary

# Related Projects
[DARTS](https://arxiv.org/abs/1806.09055): Differentiable Architecture Search \
[PC-DARTS](https://arxiv.org/abs/1907.05737): DARTS with Partial Channel sampling \
[Two-stream Network for Video Understanding](https://papers.nips.cc/paper/2014/file/00ec53c4682d36f5c4359f4ae7bd7ba1-Paper.pdf): Video action recognition using RGB and Flow networks \
[EvaNet](https://arxiv.org/abs/1811.10636): NAS for video understanding using evolutionary strategy \
[Video-DARTS](https://arxiv.org/abs/1907.04632): Video architecture search using DARTS \ 
[MARS](https://github.com/craston/MARS): Motion-Augmented RGB Stream for Action Recognition \
[3D ResNets for Action Recognition](https://github.com/kenshohara/3D-ResNets-PyTorch) \
