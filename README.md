# DDcGAN in TensorFlow 2

This repository contains an implementation of the Dual-Discriminator Conditional Generative Adversarial Network (DDcGAN) using TensorFlow 2. DDcGAN is a powerful framework for multi-modal image fusion, particularly suited for infrared-visible and MRI-PET image pairs.

This implementation is based on the original [DDcGAN paper](https://ieeexplore.ieee.org/abstract/document/9031751).

The following datasets are used in this project:
- [Training_Dataset.h5](https://drive.google.com/file/d/1o-dhSphyyiqSHu9veiKWvxViZ_FSeZWJ/view?usp=share_link) - infrared and visible images from TNO Human Factors Dataset.

## Features
- Implemented in TensorFlow 2
- Supports multi-resolution image fusion
- Includes data preprocessing and augmentation
- Provides evaluation metrics for generated images

## Installation

```
git clone https://github.com/yourusername/ddcgan-tensorflow2.git
cd ddcgan-tensorflow2
pip install -r requirements.txt
```
