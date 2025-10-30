import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import cv2
import torch
from torch.utils.data import Dataset

import os
import cv2
import torch
from torch.utils.data import Dataset
from datetime import datetime
import time
from tqdm import tqdm

import yaml

import argparse

import math
from typing import Dict, List


class Pix2PixDataset(Dataset):
    def __init__(self, root_dir_X1, root_dir_X2, image_size=(256, 256)):
        """Inicializar el dataset desde carpetas"""
        self.root_dir_X1 = root_dir_X1
        self.root_dir_X2 = root_dir_X2

        # Obtener lista de nombres de archivos de ambas carpetas (ordenada para consistencia)
        self.image_filenames = sorted(os.listdir(root_dir_X1))
        
        # Dimensión final a la que redimensionaremos las imágenes
        self.image_size = image_size

    def __len__(self):
        """Número total de muestras"""
        return len(self.image_filenames)


    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_A_path = os.path.join(self.root_dir_X1, img_name)
        img_B_path = os.path.join(self.root_dir_X2, img_name)

        img_A = cv2.imread(img_A_path)
        img_B = cv2.imread(img_B_path)
        if img_A is None or img_B is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_A_path} o {img_B_path}")

        img_A = cv2.resize(img_A, self.image_size)
        img_B = cv2.resize(img_B, self.image_size)

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        img_A = torch.tensor(img_A, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        img_B = torch.tensor(img_B, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0

        return img_A, img_B, img_name


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchDiscriminator, self).__init__()

        # C64: Conv -> LeakyReLU
        self.conv1 = nn.Conv2d(input_channels * 2, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        # C128: Conv -> BatchNorm -> LeakyReLU
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        # C256: Conv -> BatchNorm -> LeakyReLU
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        # C512: Conv -> BatchNorm -> LeakyReLU
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        # Segunda capa final: Conv -> BatchNorm -> LeakyReLU
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)

        # Capa de salida: Conv -> Sigmoid
        self.conv6 = nn.Conv2d(512, 1, kernel_size=4, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_A, img_B):
        # Concatenar imágenes input y target en la dimensión del canal
        x = torch.cat([img_A, img_B], dim=1)

        # Aplicar las capas secuenciales
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.lrelu5(self.bn5(self.conv5(x)))

        # Capa de salida
        x = self.sigmoid(self.conv6(x))

        return x

# Define Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(EncoderBlock, self).__init__()
        # Inicialización similar a Keras
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.lrelu(x)
        return x

# Define Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(DecoderBlock, self).__init__()
        # Transposed Convolution (Upsampling)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        # Concatenar con skip connection
        x = torch.cat([x, skip_connection], dim=1)
        x = self.relu(x)
        return x

# Define Generator with Encoder-Decoder (UNet style)
class Pix2PixGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Pix2PixGenerator, self).__init__()

        # Encoder (Downsampling)
        self.e1 = EncoderBlock(input_channels, 64, batchnorm=False)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 512)
        self.e6 = EncoderBlock(512, 512)
        self.e7 = EncoderBlock(512, 512)

        # Bottleneck (sin BatchNorm + ReLU)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (Upsampling)
        self.d1 = DecoderBlock(512, 512)
        self.d2 = DecoderBlock(1024, 512)
        self.d3 = DecoderBlock(1024, 512)
        self.d4 = DecoderBlock(1024, 512, dropout=False)
        self.d5 = DecoderBlock(1024, 256, dropout=False)
        self.d6 = DecoderBlock(512, 128, dropout=False)
        self.d7 = DecoderBlock(256, 64, dropout=False)

        # Output layer: ConvTranspose + Tanh
        self.out_conv = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        # Encoder Forward Pass
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decoder Forward Pass + Skip Connections
        d1 = self.d1(b, e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)

        # Output Layer
        out_image = self.out_activation(self.out_conv(d7))
        return out_image

class Pix2PixGAN(nn.Module):
    def __init__(self, generator, discriminator, lr=0.0002, beta1=0.5):
        super(Pix2PixGAN, self).__init__()
        
        self.generator = generator
        self.discriminator = discriminator

        # Inicializar pesos usando la función interna
        self._initialize_weights()

        # Congelar el discriminador durante el entrenamiento del generador
        for param in self.discriminator.parameters():
            param.requires_grad = False

        # Configurar optimizadores para generador y discriminador
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    def forward(self, img_A):
        """Genera imágenes fake y evalúa su calidad."""
        # Generar imagen fake desde img_A
        fake_B = self.generator(img_A)
        # Evaluar si fake_B es real o falsa
        pred_fake = self.discriminator(img_A, fake_B)
        return pred_fake, fake_B

    def _initialize_weights(self):
        """Inicialización de pesos con distribución normal N(0, 0.02)"""
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)

        # Inicializar pesos del generador y del discriminador
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def get_optimizers(self):
        """Devuelve los optimizadores para usar durante el entrenamiento."""
        return self.optimizer_G, self.optimizer_D

def load_real_samples(filename):
    """Carga datos desde un archivo .npz y escala los valores de [0, 255] a [-1, 1]."""
    # Cargar datos comprimidos
    data = load(filename)
    # Desempaquetar arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # Escalar valores de píxeles de [0,255] a [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return X1, X2

def generate_real_samples(dataset, n_samples, patch_shape):
    """Selecciona muestras reales aleatorias del dataset."""
    # Desempaquetar dataset
    trainA, trainB = dataset
    # Elegir índices aleatorios
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    # Obtener imágenes seleccionadas
    X1, X2 = trainA[ix], trainB[ix]
    # Crear etiquetas para imágenes reales (valor 1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X1, X2, y

def generate_fake_samples(generator, samples, patch_shape):
    """Genera imágenes falsas usando el generador y asigna etiquetas de clase 0."""
    # Generar imágenes falsas desde el generador
    X_fake = generator(samples)
    # Crear etiquetas para imágenes falsas (valor 0)
    y_fake = np.zeros((len(X_fake), patch_shape, patch_shape, 1))
    return X_fake, y_fake


@torch.no_grad()
def summarize_performance(
    step: int,
    generator: nn.Module,
    dataset,                 # <- ahora genérico: puede ser train o val
    output_dir: str,
    subjects: List[str],
    device,
    split: str,              # <- "train" o "val"
    save_model: bool = True
):
    """
    Genera DOS PDFs por sujeto y por split:
      - Horizontal (1x3) -> *_H.pdf
      - Vertical   (3x1) -> *_V.pdf
    Orden: Input (invertido) / Generated / Target.
    Se guardan en: .../epoch_XXXX/<split>/
    """
    generator.eval()

    # Carpeta por época y por split
    epoch_dir = os.path.join(output_dir, f'epoch_{step+1:04d}', split)
    os.makedirs(epoch_dir, exist_ok=True)

    def _to_np_batches(one_A: torch.Tensor, one_B: torch.Tensor):
        fake_B = generator(one_A).detach()
        X_realA = to_numpy_img_batch(one_A)
        X_realB = to_numpy_img_batch(one_B)
        X_fakeB = to_numpy_img_batch(fake_B)
        X_realA_inv = 1.0 - X_realA   # solo visual
        return X_realA_inv[0], X_fakeB[0], X_realB[0]

    def _save_horizontal(inp, gen, tgt, out_path: str):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(inp); axs[0].axis('off'); axs[0].set_title('Input')
        axs[1].imshow(gen); axs[1].axis('off'); axs[1].set_title('Generated')
        axs[2].imshow(tgt); axs[2].axis('off'); axs[2].set_title('Target')
        plt.tight_layout(pad=0.5, w_pad=0.3)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {out_path}")

    def _save_vertical(inp, gen, tgt, out_path: str):
        fig, axs = plt.subplots(3, 1, figsize=(3, 9))
        axs[0].imshow(inp); axs[0].axis('off'); axs[0].set_title('Input')
        axs[1].imshow(gen); axs[1].axis('off'); axs[1].set_title('Generated')
        axs[2].imshow(tgt); axs[2].axis('off'); axs[2].set_title('Target')
        plt.tight_layout(pad=0.5, h_pad=0.3)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {out_path}")

    # --- helper robusto para encontrar índice por sujeto ---
    def _find_idx_for_subject(name_list: List[str], subj: str):
        # intenta varias formas: "S52", "S052", "_52_", "-52", "52_"
        candidates = {subj, f"S{subj}", f"S{subj.zfill(3)}"}
        # también prueba si el número aislado aparece con delimitadores comunes
        extra = [f"_{subj}_", f"-{subj}-", f"_{subj}.", f"-{subj}.", f"{subj}_", f"{subj}-"]
        candidates.update(extra)
        for i, fname in enumerate(name_list):
            if any(c in fname for c in candidates):
                return i
        # última chance: contiene el número “tal cual”
        for i, fname in enumerate(name_list):
            if subj in fname:
                return i
        return None

    any_plotted = False
    for subj in subjects:
        idx_match = _find_idx_for_subject(dataset.image_filenames, subj)
        if idx_match is None:
            print(f"[WARN] No matching samples for subject '{subj}' in split '{split}'.")
            continue

        a, b, fname = dataset[idx_match]
        one_A = a.unsqueeze(0).to(device)
        one_B = b.unsqueeze(0).to(device)

        inp, gen, tgt = _to_np_batches(one_A, one_B)

        base = os.path.join(epoch_dir, f'plot_{step+1:06d}_{split}_{subj}')
        _save_horizontal(inp, gen, tgt, f"{base}_H.pdf")
        _save_vertical(inp, gen, tgt, f"{base}_V.pdf")
        any_plotted = True

    # Guarda el modelo una sola vez por época si hubo al menos un plot (solo en split val)
    if save_model and any_plotted and split == "val":
        model_filename = os.path.join(output_dir, f'epoch_{step+1:04d}', f'model_{step+1:06d}.pth')
        torch.save(generator.state_dict(), model_filename)
        print(f"Model saved: {model_filename}")

    generator.train()




def save_losses_txt(losses, output_dir):
    """Guarda las pérdidas en un archivo TXT."""
    filename = os.path.join(output_dir, "losses.txt")
    with open(filename, mode='w') as file:
        for epoch, d_loss, g_loss, l1_loss in losses:
            file.write(
                f"Epoch {int(epoch)}: D_loss = {d_loss:.4f}, G_loss = {g_loss:.4f}, L1_loss = {l1_loss:.4f}\n"
            )
    print(f"Losses saved inn: {filename}")

def plot_losses(losses, output_dir):
    """Genera un gráfico de las pérdidas durante el entrenamiento."""
    losses = np.array(losses)
    epochs = losses[:, 0]
    d_loss = losses[:, 1]
    g_loss = losses[:, 2]
    l1_loss = losses[:, 3]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, d_loss, label='D_loss', color='red')
    plt.plot(epochs, g_loss, label='G_loss', color='blue')
    plt.plot(epochs, l1_loss, label='L1_loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()
    print(f"Loss graph saved in: {os.path.join(output_dir, 'loss_plot.png')}")

def save_epoch_metrics_txt(output_dir, epoch, train_m, val_m):
    fp = os.path.join(output_dir, "metrics_summary.txt")
    with open(fp, "a") as f:
        f.write(
            f"Epoch {epoch}: "
            f"TRAIN[MSE={train_m['MSE']:.6f}, RMSE={train_m['RMSE']:.6f}, MAE={train_m['MAE']:.6f}, PSNR={train_m['PSNR']:.6f}, SSIM={train_m['SSIM']:.6f}, NCC={train_m['NCC']:.6f}] | "
            f"VAL[MSE={val_m['MSE']:.6f}, RMSE={val_m['RMSE']:.6f}, MAE={val_m['MAE']:.6f}, PSNR={val_m['PSNR']:.6f}, SSIM={val_m['SSIM']:.6f}, NCC={val_m['NCC']:.6f}]\n"
        )



# Cargar configuración desde config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def to_numpy_img_batch(t: torch.Tensor) -> np.ndarray:
    """
    Convierte un tensor en [-1,1] con shape (N,3,H,W) a numpy [0,1] (N,H,W,3).
    """
    x = (t.detach().cpu().permute(0, 2, 3, 1).numpy() + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)

def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse_np(a, b)))

def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    mse = mse_np(a, b)
    if mse == 0:
        return float("inf")
    PIX_MAX = 1.0  # porque trabajamos en [0,1]
    return 20.0 * math.log10(PIX_MAX) - 10.0 * math.log10(mse)

def ncc_np(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (H,W,3) en [0,1]
    a_f = a.reshape(-1, 3).astype(np.float64)
    b_f = b.reshape(-1, 3).astype(np.float64)
    a_f -= a_f.mean(axis=0, keepdims=True)
    b_f -= b_f.mean(axis=0, keepdims=True)
    denom = (np.linalg.norm(a_f, axis=0) * np.linalg.norm(b_f, axis=0) + 1e-12)
    corr = (a_f * b_f).sum(axis=0) / denom
    return float(np.nanmean(corr))

def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    SSIM simple por canal + promedio, usando filtro Gauss (aprox).
    a,b: (H,W,3) en [0,1]
    """
    def _ssim_single(c1: np.ndarray, c2: np.ndarray) -> float:
        # constantes (L=1)
        K1, K2, L = 0.01, 0.03, 1.0
        C1, C2 = (K1*L)**2, (K2*L)**2
        # blur gaussiano
        mu1 = cv2.GaussianBlur(c1, (11,11), 1.5)
        mu2 = cv2.GaussianBlur(c2, (11,11), 1.5)
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2

        sigma1_sq = cv2.GaussianBlur(c1*c1, (11,11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(c2*c2, (11,11), 1.5) - mu2_sq
        sigma12   = cv2.GaussianBlur(c1*c2, (11,11), 1.5) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)* (sigma1_sq + sigma2_sq + C2) + 1e-12)
        return float(ssim_map.mean())

    # por canal, luego promedio
    ssim_vals = []
    for ch in range(3):
        ssim_vals.append(_ssim_single(a[..., ch], b[..., ch]))
    return float(np.mean(ssim_vals))

def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true, y_pred: (N,H,W,3) en [0,1]
    Devuelve promedios por batch.
    """
    mses, rmses, maes, psnrs, ssims, nccs = [], [], [], [], [], []
    for i in range(y_true.shape[0]):
        gt = y_true[i]
        pr = y_pred[i]
        mses.append(mse_np(gt, pr))
        rmses.append(rmse_np(gt, pr))
        maes.append(mae_np(gt, pr))
        psnrs.append(psnr_np(gt, pr))
        ssims.append(ssim_np(gt, pr))
        nccs.append(ncc_np(gt, pr))
    return {
        "MSE":  float(np.mean(mses)),
        "RMSE": float(np.mean(rmses)),
        "MAE":  float(np.mean(maes)),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "NCC":  float(np.mean(nccs)),
    }

@torch.no_grad()
def evaluate(generator: nn.Module, dataloader: DataLoader, device) -> Dict[str, float]:
    generator.eval()
    agg = {"MSE":0,"RMSE":0,"MAE":0,"PSNR":0,"SSIM":0,"NCC":0}
    n_batches = 0
    for batch in dataloader:
        if len(batch) == 3:
            real_A, real_B, _ = batch
        else:
            real_A, real_B = batch
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        fake_B = generator(real_A)

        y_true = to_numpy_img_batch(real_B)
        y_pred = to_numpy_img_batch(fake_B)

        m = compute_metrics_np(y_true, y_pred)
        for k in agg: agg[k] += m[k]
        n_batches += 1

    for k in agg: agg[k] = float(agg[k] / max(n_batches,1))
    generator.train()
    return agg


def train(d_model, g_model, gan_model,
          train_loader, val_loader,
          output_dir, device,
          n_epochs=100, save_every_n=5):

    sample_input = torch.randn(1, 3, 256, 256, device=device)
    n_patch = d_model(sample_input, sample_input).shape[2]

    all_losses = []
    metrics_csv = os.path.join(output_dir, "metrics_per_epoch.csv")
    with open(metrics_csv, "w") as f:
        f.write("epoch,split,MSE,RMSE,MAE,PSNR,SSIM,NCC\n")

    for epoch in range(n_epochs):
        start_time = time.time()
        d_model.train(); g_model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader),
                  desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch") as t:
            for i, batch in t:
                if len(batch) == 3:
                    real_A, real_B, _ = batch
                else:
                    real_A, real_B = batch

                real_A, real_B = real_A.to(device), real_B.to(device)

                # --- Discriminator ---
                for p in d_model.parameters(): p.requires_grad = True
                gan_model.optimizer_D.zero_grad()

                fake_B = g_model(real_A).to(device)
                y_real = torch.ones((real_A.size(0), 1, n_patch, n_patch), device=device)
                y_fake = torch.zeros((real_A.size(0), 1, n_patch, n_patch), device=device)

                pred_real = d_model(real_A, real_B)
                loss_D_real = nn.BCELoss()(pred_real, y_real)

                pred_fake = d_model(real_A, fake_B.detach())
                loss_D_fake = nn.BCELoss()(pred_fake, y_fake)

                loss_D = 0.5*(loss_D_real + loss_D_fake)
                loss_D.backward()
                gan_model.optimizer_D.step()

                # --- Generator ---
                for p in d_model.parameters(): p.requires_grad = False
                gan_model.optimizer_G.zero_grad()

                fake_B = g_model(real_A)
                pred_fake = d_model(real_A, fake_B)

                loss_GAN = nn.BCELoss()(pred_fake, y_real)
                loss_L1  = nn.L1Loss()(fake_B, real_B) * 100
                loss_G   = loss_GAN + loss_L1
                loss_G.backward()
                gan_model.optimizer_G.step()

                t.set_postfix({
                    "D_loss": f"{loss_D.item():.4f}",
                    "G_loss": f"{loss_G.item():.4f}",
                    "L1":     f"{loss_L1.item():.4f}",
                })

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{n_epochs} completed in {epoch_time:.2f}s.\n")

        all_losses.append((epoch + 1, loss_D.item(), loss_G.item(), loss_L1.item()))

        # --- MÉTRICAS TRAIN/VAL ---
        train_metrics = evaluate(g_model, train_loader, device)
        val_metrics   = evaluate(g_model, val_loader, device)

        # Log a CSV
        with open(metrics_csv, "a") as f:
            f.write(f"{epoch+1},train,{train_metrics['MSE']:.6f},{train_metrics['RMSE']:.6f},{train_metrics['MAE']:.6f},{train_metrics['PSNR']:.6f},{train_metrics['SSIM']:.6f},{train_metrics['NCC']:.6f}\n")
            f.write(f"{epoch+1},val,{val_metrics['MSE']:.6f},{val_metrics['RMSE']:.6f},{val_metrics['MAE']:.6f},{val_metrics['PSNR']:.6f},{val_metrics['SSIM']:.6f},{val_metrics['NCC']:.6f}\n")

        # TXT resumen por época
        save_epoch_metrics_txt(output_dir, epoch+1, train_metrics, val_metrics)

        print(f"[Epoch {epoch+1}] Train metrics: {train_metrics}")
        print(f"[Epoch {epoch+1}] Val   metrics: {val_metrics}")

        if (epoch + 1) % save_every_n == 0:
            # Sujetos VAL (como ya usabas)
            val_subjects = ["S130", "S132", "S111", "S124", "S125"]
            summarize_performance(
                step=epoch,
                generator=g_model,
                dataset=val_loader.dataset,
                output_dir=output_dir,
                subjects=val_subjects,
                device=device,
                split="val",
                save_model=True      # guarda el .pth una vez por época
            )

            # Sujetos TRAIN 
            train_subjects = ["S052", "S055", "S074", "S106", "S113", "S121"]
            summarize_performance(
                step=epoch,
                generator=g_model,
                dataset=train_loader.dataset,
                output_dir=output_dir,
                subjects=train_subjects,
                device=device,
                split="train",
                save_model=False     # no necesitamos guardar otra vez
            )

            save_losses_txt(all_losses, output_dir)
            plot_losses(all_losses, output_dir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pix2pix with faces+events (CK+ as target).")
    parser.add_argument(
        "--device",
        choices=["asus", "dgx-1"],
        default="asus",
        help="Selecciona el dispositivo para cargar la config YAML (asus -> config_asus.yaml, dgx-1 -> config_dgx-1.yaml)."
    )
    parser.add_argument(
        "--x1",
        choices=["X1_t0_t", "X1_t0_t0", "X1_t_t", "X1_t_t0"],
        default="X1_t0_t",
        help="Variante de eventos a usar como X1."
    )
    parser.add_argument(
        "--dataset_name",
        default="faces_events_ck",
        help="Nombre del dataset de entrada (raíz dentro de ../input/). Por defecto faces_events_ck."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="IDs de GPU a usar. Ej.: '3' para una GPU o '0,1,2' para múltiples (DataParallel)."
    )
    args = parser.parse_args()

    # --- Selección de GPU(s) por ID (antes de crear modelos/tensors) ---
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Cargar hiperparámetros desde el YAML correspondiente
    config_path = f"yaml/config_{args.device}.yaml"
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    BATCH_SIZE = config.get("BATCH_SIZE", 16)
    EPOCHS = config.get("EPOCHS", 100)
    SAVE_EVERY_N_EPOCHS = config.get("SAVE_EVERY_N_EPOCHS", 5)
    NUM_WORKERS = config.get("NUM_WORKERS", 4)

    # Nombre del dataset y variante X1
    dataset_name = args.dataset_name
    x1_name = args.x1  # p.ej., "X1_t0_t"

    # Carpeta de resultados con timestamp + dataset + variante
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../output/{timestamp}_pix2pix_unet_{dataset_name}_{x1_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be stored in: {output_dir}")

    # Rutas de train/val (nueva estructura)
    root_dir_X1_train = f"../input/{dataset_name}/{x1_name}/train/"
    root_dir_X2_train = f"../input/{dataset_name}/X2/train/"
    root_dir_X1_val   = f"../input/{dataset_name}/{x1_name}/val/"
    root_dir_X2_val   = f"../input/{dataset_name}/X2/val/"

    # Datasets
    dataset_train = Pix2PixDataset(root_dir_X1_train, root_dir_X2_train)
    dataset_val   = Pix2PixDataset(root_dir_X1_val,   root_dir_X2_val)
    print(f"Train samples: {len(dataset_train)} | Val samples: {len(dataset_val)}")

    # DataLoaders
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    dataloader_val   = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Dispositivo (CUDA/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch sees {torch.cuda.device_count()} CUDA device(s). Using device: {device}")

    # Modelos
    generator = Pix2PixGenerator(input_channels=3, output_channels=3)
    discriminator = PatchDiscriminator(input_channels=3)

    # Múltiples GPUs visibles -> DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs (visible via CUDA_VISIBLE_DEVICES).")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # GAN
    gan_model = Pix2PixGAN(generator, discriminator).to(device)

    train(discriminator, generator, gan_model,
        dataloader_train, dataloader_val,
        output_dir, device,
        n_epochs=EPOCHS, save_every_n=SAVE_EVERY_N_EPOCHS)