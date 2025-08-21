# Inverse design of terahertz amplitude modulator using tandem deep neural networks

## Introduction

This repository contains the code used in the following publication:  
<https://pubs.acs.org/doi/xx.xxxx/acssensors.xxxxxxx>

Here, we use **tandem deep neural networks** for inverse design for THz metasurfaces composed of split-ring resonators (SRR). SRR-based metasurfaces are widely used because their electromagnetic response can be precisely controlled by adjusting their geometric parameters.

In this study, the SRR geometry is defined using the following eight parameters:

- **gap number (G<sub>n</sub>)**
- **periodicity (P)**
- **structure width (S<sub>w</sub>)**
- **line width (L<sub>w</sub>)**
- **gap distance (G<sub>d</sub>)**
- **position of gap in the first gap (G<sub>p_1st</sub>)**
- **position of gap in the second gap (G<sub>p_2nd</sub>)**
- **gap rotation (G<sub>p_r</sub>)**

![ex_Geometry](.\img\Geometry.png)

These parameters directly determine the modulation characteristics in the THz band, and the goal of this work is to perform inverse design to obtain desired spectra by finding appropriate SRR parameters.

---

## Requirements

To run the provided scripts, the following software is required. GPU is recommended for training to reduce the lengthy training times.

- Python 3.7
- PyTorch 1.12.1
- CUDA 11.2
- numpy
- matplotlib
- pandas
- os

---

## Steps

### 1) Train the Autoencoder (Autoencoder_Train.py)

The autoencoder performs nonlinear dimensionality reduction while preserving key spectral features. By projecting input spectra onto a learned manifold, it ensures compatibility with the inverse design model and training data distribution. This projection reduces inconsistencies and guides the network toward physically realistic solutions, improving robustness and accuracy. 


---

### 2) Train the Tandem network (Tandem_Train.py)

The tandem network is trained on paired structural parameters and simulated spectra. The ForwardNet is first trained to predict spectra from parameters and then fixed, and the InverseNet is trained in tandem with the fixed pre-trained forward model to predict parameters from a target spectrum while ensuring the reconstructed spectrum matches the target spectrum.

The network consists of two parts:

- **ForwardNet**: predicts the spectrum from structural parameters, trained with an mean squared error (MSE) loss function with a spectral weighting mask to suppress Fabry-Perot interference.
- **InverseNet**: predicts structural parameters from a target spectrum, optimized in a tandem architecture by reconstructing the spectrum through the fixed ForwardNet. 

The optimal model performance was achieved at 500 epochs, producing the files:
- `Forwardmodel.pth`
- `Inversemodel.pth`


---

### 3) Predict parameters from spectrum using Tandem network (Tandem_Predict.py)

The pretrained autoencoder, forward model, and inverse model are loaded to perform inverse design. The generated target spectrum with Lorentzian dip is processed through the autoencoder to ensure compatibility with the inverse model, and then passed though the inverse model to predict the corresponding structural parameters.

---

## Citation

If you find this code or data helpful, please cite our work using the following:

> Jeong, et al. *Inverse design of terahertz amplitude modulator using tandem deep neural networks.* journal, 2025.

---
