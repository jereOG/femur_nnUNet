# 3D CT Proximal Femora Segmentation Using Optimized nnU-Net

## Project Overview

This repository supports a project about implementing, optimizing, and evaluating the nnU-Net model by Isensee et al. (2021) [nnU-Net paper](https://doi.org/10.1038/s41592-020-01008-z) for segmenting femur CT images. The primary objective is to improve the segmentation process through manual correction of ground truth data, parameter tuning, and analysis of the performance of the best 2D and 3D models.

This use the deep learning architecture from [https://github.com/MIC-DKFZ/nnUNet] with changed plans files and Jupyter notebooks specifically for the training, prediction and evaluation of femur CT images. Due to data access restrictions, the datasets used in this project cannot be provided publicly.

It is important to note that the core architecture and system of nnU-Net are not my original work. This repository is intended only for the reproducibility of a project report focused on the segmentation of proximal femur CT images, providing the possibility of easily reproducing the results or applying the pretrained model to similar bone CT images.

## Setup and Installation

The installation process is similar to the original nnU-Net, with the addition of a few additional Python packages required for the `predicting.ipynb` notebook.

1. Ensure you are using Python 3.11.9.

2. Install PyTorch version 2.4.0 with CUDA 11.8 by running the following command:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ``` 

3. Clone the nnU-Net repository and install it:
   ```bash
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install -e .
   ```

4. Move the files from this repository into the cloned `nnUNet` repository, ensuring they are at the same directory level as the `nnUNet` directory.

5. Install the required Python packages listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

The plan files in the nnU-Net will be automatically updated when you run the `training.ipynb` notebook.

## Usage

Follow the steps documented in the notebooks in this order: preprocessing, training, predicting. For more detailed information on the nnU-Net structure and functionality, please refer to the official nnU-Net documentation at [https://github.com/MIC-DKFZ/nnUNet].

The best model found in the project, trained on both the original and corrected images, is labeled as `3d_fullres` and can be used directly for inference in the `predicting.ipynb` notebook.

## References

1. Isensee, Fabian, et al. "nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation." *Nature Methods*, 18(2):203-211, 2021. DOI: [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)

