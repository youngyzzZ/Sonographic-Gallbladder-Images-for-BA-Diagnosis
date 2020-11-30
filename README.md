# Sonographic_Gallbladder_Diagnose_System

![https://opensource.org/licenses/MIT](https://img.shields.io/badge/license-MIT-green.svg)
![https://www.python.org/](https://img.shields.io/badge/language-python-yellow.svg)
![https://pytorch.org/get-started/locally/](https://img.shields.io/badge/backbone-PyTorch-important.svg)
![](https://img.shields.io/badge/version-1.0.0-blue.svg)

# Content
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Demo](#demo)
- [Result of CAM](#result-of-cam)
- [Result](#result)

# Overview
It is still difficult to make accurate diagnosis of biliary atresia (BA) by sonographic gallbladder images particularly in rural area without relevant expertise. To help diagnose BA based on sonographic gallbladder images, an ensembled deep learning model was developed. The model yields a patient-level sensitivity 93.1% and specificity 93.9% \[with areas under the receiver operating characteristic curve of 0.956 (95% confidence interval: 0.928-0.977)] on the multi-center external validation dataset, superior to that of human experts. With the help of the model, the performance of human experts with various levels would be improved further. Moreover, the diagnosis based on smartphone photos of sonographic gallbladder images through a smartphone app and based on video sequences by the model still yielded expert-level performances. In this work, the ensembled deep learning model provides a solution to help radiologists improve BA diagnosis in various clinical application scenarios, particularly in rural and undeveloped regions with limited expertise. 

# System Requirements
## Hardware requirements
The source code require at lest 6GB GPU memory to support it to work.

## Software requirements
### OS Requirements
This package is supported for *Windoes* and *Linux*. The package has been tested on the following systems:
+ Windows: Microsoft Windows 10 Pro 10.0.1.18363
+ Linux: Ubuntu 18.04

### Python Dependencies
The following python pakage are required :

```
pytorch
torchvision
numpy
sklearn
tensorboardX
PIL
tqdm
SimpleITK
pandas
pretrainedmodels
efficientnet_pytorch
matplotlib
```

# Demo
# Result of CAM
# Result


