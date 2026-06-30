from __future__ import annotations

import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, normalize: bool):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: bool):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(
        self, inputs: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((self.block(inputs), skip), dim=1)


class Pix2PixGenerator(nn.Module):
    """Eight-level U-Net generator for 512x512 grayscale translation."""

    def __init__(self) -> None:
        super().__init__()
        self.e1 = EncoderBlock(1, 64, normalize=False)
        self.e2 = EncoderBlock(64, 128, normalize=True)
        self.e3 = EncoderBlock(128, 256, normalize=True)
        self.e4 = EncoderBlock(256, 512, normalize=True)
        self.e5 = EncoderBlock(512, 512, normalize=True)
        self.e6 = EncoderBlock(512, 512, normalize=True)
        self.e7 = EncoderBlock(512, 512, normalize=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.d1 = DecoderBlock(512, 512, dropout=True)
        self.d2 = DecoderBlock(1024, 512, dropout=True)
        self.d3 = DecoderBlock(1024, 512, dropout=True)
        self.d4 = DecoderBlock(1024, 512, dropout=False)
        self.d5 = DecoderBlock(1024, 256, dropout=False)
        self.d6 = DecoderBlock(512, 128, dropout=False)
        self.d7 = DecoderBlock(256, 64, dropout=False)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(inputs)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        hidden = self.bottleneck(e7)
        hidden = self.d1(hidden, e7)
        hidden = self.d2(hidden, e6)
        hidden = self.d3(hidden, e5)
        hidden = self.d4(hidden, e4)
        hidden = self.d5(hidden, e3)
        hidden = self.d6(hidden, e2)
        hidden = self.d7(hidden, e1)
        return self.output(hidden)


class PatchDiscriminator(nn.Module):
    """70x70-style PatchGAN discriminator without a final sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.model(torch.cat((inputs, targets), dim=1))


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)

