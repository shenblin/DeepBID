<p align="center">
  <img src="assets/title.jpg">
</p>

This project hosts the scripts for training and testing CL-SSL, as presented in our paper: Surmounting photon limits and motion artifacts for in vivo imaging of rapid biodynamics via ultrafast conjugate-line self-supervised learning (CL-SSL)


## Introduction

The presence of obstinate noise and artifacts in optical microscopy can exert a substantial influence on signals, particularly when observing moving organisms or rapid dynamic biological processes. While deep learning can post-process data, it proves to be ineffective in mitigating these disturbances in rapid biodynamics, primarily due to the challenges associated with reference signal acquisition. We introduce an advanced self-supervised technique for real-time denoising and deblurring by exploiting correlations along conjugated collinear scan paths, circumventing pristine signal acquisition. The study showcases this deep learning approach on vibrating calcium recording and pulsating cardiac imaging, demonstrating effective noise and motion artifact removal. It also enables visualization of mass transport in rapid vascular flow and restoration of volumetric astrocytic structures. By enhancing correlation, signal-to-noise ratio and structural similarity, the method facilitates swift examination of morphological and functional aspects of rapid biological dynamics. This highlights the potential of the self-supervised deep learning to strengthen live microscopy under challenging imaging conditions, advancing biological discovery.

## Self-supervised enhanced in vivo imaging 
3D self-supervised learning realized high-speed, high-SNR in vivo two-photon microscopy.

<p align="center">
  <img src="assets/diagram.gif">
</p>

Learn and restrore the spatiotemporal relationship of time-lapse images for motion-affected biodynamics.

<p align="center">
  <img src="assets/comparison.gif">
</p>


## Network
📕 Dependencies and Installation

Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

    ```bash
    git clone https://github.com/shenblin/Enhanced3D.git
    ```

2. Install dependent packages

    ```bash
    pip install -r requirements.txt
     ```
     ```bash
    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
      ```
     ```bash
    python setup.py develop
    ```
   
📕 Dataset Preparation

Please refer to [DataPreparation](datasets/Data_Download.md). It mainly includes Synthetic and Experimental data for training and testing.


⚡ Train and Test

- **Training and testing commands**: For single gpu, use the following command as example:
1. **Training**

   Bidirectional collinear scan (y = 2x in pixel)

    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_UNet3D_self_lines_train_bi_scan.yml
    ```
    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/Synthetic_Ca_UNet3D_self_lines_train.yml
    ```
    
   Normal scan, CL-SSL
    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_UNet3D_self_lines_train.yml
    ```
    CF-SSL
    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_UNet_self_frames_train.yml
     ```

2. **Testing**
     
    CL-SSL
    ```bash
    python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_UNet3D_self_lines_test.yml
    ```
    CF-SSL
    ```bash
    python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_UNet_self_frames_test.yml
     ```
 
📢 Results

For more results and further analyses, please refer to our paper.


📜 Acknowledgement

Thanks [paper](https://arxiv.org/pdf/2007.15651) authers for the wonderful open source project!


🌏 Citations

If you find this work useful in your research, please consider citing the paper:

B. Shen, et al.

📧 Contact

If you have any questions, please email `shenblin@foxmail.com`.
