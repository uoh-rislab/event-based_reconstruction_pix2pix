
# Pix2Pix Image Translation using PyTorch

This repository implements a **Pix2Pix** model for image-to-image translation using PyTorch. The architecture includes a U-Net-based generator and a PatchGAN-based discriminator. The code supports loading datasets from directories, training the model, and saving intermediate results and model checkpoints.

---

## 📚 Project Overview

The project implements the following main components:

- **Dataset Loader:** Loads paired images (`X1` and `X2`) from specified folders and processes them for training.
- **Generator (U-Net):** Converts input images (`X1`) to target images (`X2`).
- **Discriminator (PatchGAN):** Classifies whether an image pair is real or fake.
- **Training Loop:** Trains the Pix2Pix model with BCELoss and L1 Loss for adversarial and content similarity.

---

## 🛠️ Requirements

To set up the environment, use the following command:

```bash
# Create a virtual environment
conda create -n pix2pix_env python=3.8
conda activate pix2pix_env

# Install required packages
pip install torch torchvision numpy matplotlib tqdm opencv-python
```

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/username/pix2pix_pytorch.git
cd pix2pix_pytorch
```

### 2. Prepare Dataset

Download the “maps” dataset in the pix2pix_unet/data/ folder.

```bash
cd input/
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
tar -xzvf maps.tar.gz
```

Then, run the following script that will pre-process the data for training

```bash
cd pix2pix_unet/pre-process/
python split_maps_dataset.py
```

The dataset should be organized as follows:

```
pix2pix_unet/
├── data/
│   ├── X1/         # Input images folder
│   │   └── train/  # Training images for X1
│   └── X2/         # Target images folder
│       └── train/  # Training images for X2
└── output/         # Folder for saving results
```

### 3. Train the Model

```bash
python train_pytorch.py
```

---

## 📊 Training Details

During training:
- Model checkpoints and visual samples are saved in the `output/` directory.
- Training progress is displayed using `tqdm` with information on discriminator and generator losses.

### Model Output

- Trained model weights (`.pth`) are saved periodically.
- Visual comparisons of real, fake, and target images are saved as `.png`.

---

## 📂 Output Structure

The output directory contains:

```
output/
├── plot_XXXXXX.png  # Visualization of generated images
└── model_XXXXXX.pth  # Model checkpoint
```

---

## 📝 Custom Configuration

You can modify the following parameters in the script:
- `root_dir_X1` and `root_dir_X2` in the `__main__` section to point to your dataset folders.
- `n_epochs` in the `train()` function to change the number of epochs.

---

## 🧠 Model Details

### Generator (U-Net)

The generator consists of:
- 7 encoder layers with downsampling
- A bottleneck layer
- 7 decoder layers with upsampling and skip connections

### Discriminator (PatchGAN)

The discriminator is a PatchGAN that classifies image patches as real or fake, with:
- 5 convolutional layers
- Sigmoid activation for binary classification

---

## 📧 Contact

For any issues, questions, or contributions, please contact:

- **Author:** Ignacio
- **Email:** ignacio@example.com
- **GitHub:** [GitHub Profile](https://github.com/ibugueno)

--- 

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
