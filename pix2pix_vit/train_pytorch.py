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
import timm

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
        """Obtener una muestra individual"""
        img_name = self.image_filenames[idx]

        # Obtener las rutas de las imágenes de entrada (X1) y objetivo (X2)
        img_A_path = os.path.join(self.root_dir_X1, img_name)
        img_B_path = os.path.join(self.root_dir_X2, img_name)

        # Cargar imágenes usando OpenCV (cv2) en formato BGR
        img_A = cv2.imread(img_A_path)  # Cargar imagen A
        img_B = cv2.imread(img_B_path)  # Cargar imagen B

        # Verificar que las imágenes no sean None
        if img_A is None or img_B is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_A_path} o {img_B_path}")

        # Redimensionar imágenes al tamaño especificado (por defecto 256x256)
        img_A = cv2.resize(img_A, self.image_size)
        img_B = cv2.resize(img_B, self.image_size)

        # Convertir de BGR a RGB (para que coincida con el formato estándar)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # Convertir imágenes de numpy a tensor de PyTorch y escalar valores a [-1, 1]
        img_A = torch.tensor(img_A, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        img_B = torch.tensor(img_B, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0

        return img_A, img_B


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

# Define Encoder con ViT
class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768):
        super(ViTEncoder, self).__init__()

        # Cargar modelo ViT base preentrenado para 224x224
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)

        # Cambiar img_size y número de patches para 256x256
        self.vit.patch_embed.img_size = (img_size, img_size)  # Forzar a 256x256
        self.vit.patch_embed.grid_size = (img_size // patch_size, img_size // patch_size)
        self.vit.patch_embed.num_patches = (img_size // patch_size) ** 2

        # Interpolar pos_embed para 256 patches
        num_patches_vit = (224 // patch_size) ** 2  # 14x14 = 196
        num_patches_new = (img_size // patch_size) ** 2  # 16x16 = 256

        # Si el tamaño no coincide, redimensionar pos_embed
        if num_patches_new != num_patches_vit:
            print(f"Interpolating pos_embed from {num_patches_vit} to {num_patches_new} patches for {img_size}x{img_size}")
            
            # Obtener pos_embed excepto el token de clase
            pos_embed_interp = F.interpolate(
                self.vit.pos_embed[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2),
                size=(img_size // patch_size, img_size // patch_size),
                mode='bicubic',
                align_corners=False
            ).permute(0, 2, 3, 1).reshape(1, num_patches_new, -1)

            # Concatenar token de clase al nuevo pos_embed interpolado
            self.vit.pos_embed = nn.Parameter(
                torch.cat([self.vit.pos_embed[:, :1], pos_embed_interp], dim=1)
            )

    def forward(self, x):
        # NO redimensionar x, directamente pasarla al modelo ViT
        x = self.vit.forward_features(x)
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

        # Validación para ajustar tamaño antes de concatenar
        if x.size(2) != skip_connection.size(2) or x.size(3) != skip_connection.size(3):
            skip_connection = F.interpolate(skip_connection, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Concatenar después de ajustar tamaño
        x = torch.cat([x, skip_connection], dim=1)
        x = self.relu(x)
        return x

# Define Generator with Encoder-Decoder (UNet style)
class Pix2PixGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Pix2PixGenerator, self).__init__()

        # Encoder ViT modificado para 256x256
        self.vit_encoder = ViTEncoder(img_size=256, patch_size=16, embed_dim=768)

        # Proyectar ViT a 512 canales para el bottleneck
        self.vit_to_512 = nn.Conv2d(768, 512, kernel_size=1)

        # Proyecciones para skip connections
        self.skip256_proj = nn.Conv2d(512, 256, kernel_size=1)
        self.skip128_proj = nn.Conv2d(512, 128, kernel_size=1)

        # Proyección de la imagen original para skip final
        self.input_proj = nn.Conv2d(3, 128, kernel_size=1)

        # Bloques del decoder (ajustado para llegar exactamente a 256x256)
        self.d1 = DecoderBlock(512, 512)   # 16 -> 32
        self.d2 = DecoderBlock(1024, 512)  # 32 -> 64
        self.d3 = DecoderBlock(640, 256)   # 64 -> 128
        self.d4 = DecoderBlock(512, 256, dropout=False)  # 128 -> 256 (corregido a 512)

        # Proyección de la imagen original para skip final a 256x256
        self.input_proj = nn.Conv2d(3, 128, kernel_size=1)

        # Capa final ajustada para recibir 128+128 canales a 256x256
        self.out_conv = nn.ConvTranspose2d(256 + 128 + 256, 3, kernel_size=3, stride=1, padding=1)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        batch = x.size(0)
        embed_dim = 768
        h, w = 16, 16

        feat = self.vit_encoder(x)  # [batch, num_patches+1, embed_dim]

        # Usa otro nombre aquí, como vit_feat:
        vit_feat = feat[:, 1:, :].permute(0, 2, 1)
        vit_feat = vit_feat.reshape(batch, embed_dim, h, w)

        # Continúa correctamente usando vit_feat en adelante:
        feat = self.vit_to_512(vit_feat)

        skip_256 = self.skip256_proj(F.interpolate(feat, size=(64, 64), mode='bilinear', align_corners=False))
        skip_128 = self.skip128_proj(F.interpolate(feat, size=(32, 32), mode='bilinear', align_corners=False))

        d1_out = self.d1(feat, feat)         # 32x32
        d2_out = self.d2(d1_out, skip_128)   # 64x64
        d3_out = self.d3(d2_out, skip_256)   # 128x128
        d4_out = self.d4(d3_out, skip_256)   # 256x256 (final size)

        # Ahora sí usas el x original (entrada RGB, 3 canales):
        orig_feat = self.input_proj(
            F.interpolate(x, size=(d4_out.size(2), d4_out.size(3)), mode='bilinear', align_corners=False)
        )

        final_in = torch.cat([d4_out, orig_feat], dim=1)

        out_image = self.out_activation(self.out_conv(final_in))
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
        self.optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))



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
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
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

def summarize_performance(step, generator, dataloader, output_dir, n_samples=3):
    """Genera ejemplos y guarda resultados visuales del modelo."""
    # Obtener ejemplos reales del DataLoader
    data_iter = iter(dataloader)
    X_realA, X_realB = next(data_iter)
    X_realA, X_realB = X_realA.cuda(), X_realB.cuda()

    # Generar imágenes falsas
    X_fakeB = generator(X_realA).detach()

    # Desnormalizar imágenes para visualización
    X_realA = (X_realA.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2.0
    X_realB = (X_realB.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2.0
    X_fakeB = (X_fakeB.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2.0

    # Asegurar que las dimensiones sean (N, H, W, C) incluso si el batch_size = 1
    if X_realA.ndim == 3:
        X_realA = np.expand_dims(X_realA, axis=0)
        X_realB = np.expand_dims(X_realB, axis=0)
        X_fakeB = np.expand_dims(X_fakeB, axis=0)

    # Ajustar el número de muestras si el batch_size es 1
    n_samples = min(n_samples, X_realA.shape[0])  # Evita errores por batch_size pequeño

    # Crear carpeta para la época actual en formato epoch_XXXX
    epoch_dir = os.path.join(output_dir, f'epoch_{step+1:04d}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Crear figura horizontal (n_samples filas, 3 columnas)
    fig, axs = plt.subplots(n_samples, 3, figsize=(3 * 3, 3 * n_samples))

    if n_samples == 1:
        axs[0].imshow(X_realA[0])
        axs[0].axis('off')
        axs[0].set_title('Input')

        axs[1].imshow(X_fakeB[0])
        axs[1].axis('off')
        axs[1].set_title('Generated')

        axs[2].imshow(X_realB[0])
        axs[2].axis('off')
        axs[2].set_title('Target')
    else:
        for i in range(n_samples):
            axs[i, 0].imshow(X_realA[i])
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Input')

            axs[i, 1].imshow(X_fakeB[i])
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Generated')

            axs[i, 2].imshow(X_realB[i])
            axs[i, 2].axis('off')
            axs[i, 2].set_title('Target')

    # Ajustar bordes para eliminar espacio en blanco
    plt.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    # Guardar imágenes en carpeta de la época actual
    filename = os.path.join(epoch_dir, f'plot_{step+1:06d}.pdf')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {filename}")

    # Guardar modelo del generador en la carpeta de la época
    model_filename = os.path.join(epoch_dir, f'model_{step+1:06d}.pth')
    torch.save(generator.state_dict(), model_filename)
    print(f"Model saved: {model_filename}")


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


# Cargar configuración desde config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train(d_model, g_model, gan_model, dataloader, n_epochs=100, save_every_n=5):
    """Entrena la GAN con muestras reales y falsas generadas usando DataLoader."""
    # Determinar tamaño de salida del discriminador
    sample_input = torch.randn(1, 3, 256, 256).to(device)
    n_patch = d_model(sample_input, sample_input).to(device).shape[2]

    # Lista para almacenar pérdidas por época
    all_losses = []

    # Ciclo de entrenamiento
    for epoch in range(n_epochs):
        start_time = time.time()

        # Usar tqdm para la barra de progreso
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch") as t:
            for i, (real_A, real_B) in t:
                iter_start_time = time.time()

                # Mover datos a GPU si está disponible
                real_A, real_B = real_A.to(device), real_B.to(device)

                #### ---------------------------
                #### (1) Actualizar el Discriminador
                #### ---------------------------
                d_model.train()
                for param in d_model.parameters():
                    param.requires_grad = True

                gan_model.optimizer_D.zero_grad()

                # Generar imágenes falsas
                fake_B = g_model(real_A)  # No es necesario moverlo a device después

                # Crear etiquetas para reales y falsas
                y_real = torch.ones((real_A.size(0), 1, n_patch, n_patch)).to(device) * 0.9
                y_fake = torch.zeros((real_A.size(0), 1, n_patch, n_patch)).to(device) + 0.1


                # Pérdida del discriminador con imágenes reales
                pred_real = d_model(real_A, real_B)
                loss_D_real = nn.BCELoss()(pred_real, y_real)

                # Pérdida del discriminador con imágenes falsas
                pred_fake = d_model(real_A, fake_B.detach())  # CORRECTO
                loss_D_fake = nn.BCELoss()(pred_fake, y_fake)

                # Pérdida total del discriminador
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                gan_model.optimizer_D.step()

                #### ---------------------------
                #### (2) Actualizar el Generador
                #### ---------------------------
                g_model.train()
                gan_model.optimizer_G.zero_grad()

                # Generar imágenes falsas para actualizar el generador
                fake_B = g_model(real_A)
                pred_fake = d_model(real_A, fake_B)

                # Pérdida adversaria para el generador
                loss_GAN = nn.BCELoss()(pred_fake, y_real)

                # Pérdida L1 para mantener similitud con la imagen real
                loss_L1 = nn.L1Loss()(fake_B, real_B) * 5

                # Pérdida total del generador
                loss_G = loss_GAN + 50 * loss_L1
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(gan_model.generator.parameters(), max_norm=1.0)

                gan_model.optimizer_G.step()

                #### ---------------------------
                #### (3) Actualizar la barra de progreso
                #### ---------------------------
                iter_time = time.time() - iter_start_time
                t.set_postfix({
                    "D_loss": f"{loss_D.item():.4f}",
                    "G_loss": f"{loss_G.item():.4f}",
                    "L1": f"{loss_L1.item():.4f}",
                    "it_time": f"{iter_time:.2f}s"
                })

        # Tiempo total por época
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{n_epochs} completed in {epoch_time:.2f} seconds.\n")

        # Almacenar pérdidas para visualización
        all_losses.append((epoch + 1, loss_D.item(), loss_G.item(), loss_L1.item()))

        #### ---------------------------
        #### (4) Guardar resultados y pérdidas cada N épocas
        #### ---------------------------
        if (epoch + 1) % save_every_n == 0:
            summarize_performance(epoch, g_model, dataloader, output_dir)
            save_losses_txt(all_losses, output_dir)
            plot_losses(all_losses, output_dir)

    # Guardar pérdidas finales y graficarlas al finalizar el entrenamiento
    save_losses_txt(all_losses, output_dir)
    plot_losses(all_losses, output_dir)


if __name__ == "__main__":

    # Cargar hiperparámetros desde config.yaml
    config = load_config("yaml/config_dgx-1.yaml")

    BATCH_SIZE = config["BATCH_SIZE"]
    EPOCHS = config["EPOCHS"]
    SAVE_EVERY_N_EPOCHS = config["SAVE_EVERY_N_EPOCHS"]

    # Crear carpeta dinámica de resultados basada en timestamp y dataset
    dataset_name = "maps_processed"  # Cambia el nombre si es necesario
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../output/{timestamp}_pix2pix_vit_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results to be stored in: {output_dir}")

    # Rutas de las carpetas de imágenes
    root_dir_X1 = f"../input/{dataset_name}/X1/train/"  # Carpeta de imágenes de entrada
    root_dir_X2 = f"../input/{dataset_name}/X2/train/"  # Carpeta de imágenes objetivo

    # Crear dataset personalizado desde carpetas
    dataset = Pix2PixDataset(root_dir_X1, root_dir_X2)
    print(f"Data loaded from folders: {len(dataset)} samples.")

    # Crear DataLoader para cargar datos por lotes
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Aquí debes añadir el código para usar múltiples GPUs
    # Usar múltiples GPUs si están disponibles
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializar modelo en múltiples GPUs si es posible
    generator = Pix2PixGenerator(input_channels=3, output_channels=3)
    discriminator = PatchDiscriminator(input_channels=3)

    #'''
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    #'''

    # Mover modelos a GPU o CPU según disponibilidad
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Crear GAN combinada
    gan_model = Pix2PixGAN(generator, discriminator).to(device)

    # Entrenar modelo usando DataLoader
    train(discriminator, generator, gan_model, dataloader, n_epochs=EPOCHS, save_every_n=SAVE_EVERY_N_EPOCHS)

