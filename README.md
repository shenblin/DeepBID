# Surmounting photon limits and motion artifacts for biological dynamics imaging via dual-perspective self-supervised learning (DeepBID)

<p align="center">
  <img src="assets/title.png">
</p>

### [Paper]()
This project hosts the scripts for training and testing DeepBID, a self-supervised paradigm for biodynamics imaging denoising and deblurring under challenging in vivo conditions, as presented in our paper: Surmounting photon limits and motion artifacts for biological dynamics imaging via dual-perspective self-supervised learning.

## Contents

- [Introduction](#Introduction)
- [Network](#Network)
- [Dataset](#Dataset-download)
- [Train and Test](#Train-and-Test)
- [Results](#Results)
- [License](#License-and-Acknowledgement)
- [Citation](#Citation)

## Introduction

Visualizing rapid biological dynamics like neuronal signaling and microvascular flow is crucial yet challenging due to photon noise and motion artifacts. Here we present a deep learning framework for enhancing the spatiotemporal relations of optical microscopy data. Our approach leverages correlations of mirrored perspectives from conjugated scan paths, training a model to suppress noise and motion blur by restoring degraded spatial features. Quantitative validation on vibrational calcium imaging validates significant gains in spatiotemporal correlation, signal-to-noise ratio, feature accuracy, and motion tolerance compared to raw data. We further apply the framework to diverse in vivo contexts from mouse cerebral hemodynamics to zebrafish cardiac dynamics. This approach enables the clear visualization of the rapid nutrient flow in microcirculation and the systolic and diastolic processes of heartbeat. Unlike techniques relying on temporal correlations, learning inherent spatial priors avoids motion-induced artifacts. This self-supervised strategy flexibly enhances live microscopy under photon-limited and motion-prone regimes.

## Self-supervised enhanced in vivo imaging 
3D self-supervised learning realized high-speed, high-SNR in vivo brain imaging.

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
   
## Dataset download

Please refer to [DataPreparation](datasets/Data_Download.md). It mainly includes Synthetic and Experimental data for training and testing.

Images were produced in 8-bit TIFF files with customized **macro** in Fiji to reduce storage requirements, speed up data read, write and transfer, and accelerate network train and test.
Run the [macro processing file](Macro_process_stack_folder_(8-bit).ijm) in Fiji will ask you to select a folder, then automatically batch load all the images in the selected folder into 8-bit. **Note that this will overwrite the source images!!** Pleaase back up the original images.


## Train and Test

⚡  **Training and testing commands**: We provide an operable [SH file](run_self_supervised.sh) for Ubuntu that contains the following commands. Alternatively, these commands can be run on Windows and Ubuntu bashs.
For single gpu, use the following commands as example:
1. **Training**


   Bidirectional collinear scan (y = 2x in pixel) for MP-SSL

    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_self_lines_train_bi_scan.yml
    ```
    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/Synthetic_Ca_self_lines_train.yml
    ```
    
   Normal scan for MP-SSL
    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_self_lines_train.yml
    ```
    TP -SSL
    ```bash
    python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_self_frames_train.yml
     ```

2. **Testing**
     
    MP-SSL
    ```bash
    python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_self_lines_test.yml
    ```
    ```bash
    python basicsr/test.py -opt options/in_vivo_brain/Synthetic_Ca_test.yml
    ```
    TP-SSL
    ```bash
    python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_self_frames_test.yml
     ```

3. **Tutorial explaining the parameters in yml files**
 <p align="center">
  <img src="assets/Tutorial.jpg">
</p>

## Results

📢 For more results and further analyses, please refer to our paper.


## License and Acknowledgement

📜 This project is released under the [Apache 2.0 license](license/LICENSE.txt).<br>
More details about **license** and **acknowledgement** are in [LICENSE](license/README.md).

 ## Citation

🌏 If you find this work useful in your research, please consider citing the paper:

B. Shen, et al.

📧 Contact

If you have any questions, please email `shenblin@foxmail.com`.
