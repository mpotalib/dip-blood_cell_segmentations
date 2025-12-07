from typing import List, Optional

import torch
import torch.nn as nn
import torchvision


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        features: Optional[List[int]] = None,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        feats = features or [64, 128, 256, 512]
        self.inc = DoubleConv(in_channels, feats[0])
        self.down1 = Down(feats[0], feats[1])
        self.down2 = Down(feats[1], feats[2])
        self.down3 = Down(feats[2], feats[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(feats[3], feats[3] * factor)
        self.up1 = Up(feats[3] * factor + feats[3], feats[2], bilinear)
        self.up2 = Up(feats[2] + feats[2], feats[1], bilinear)
        self.up3 = Up(feats[1] + feats[1], feats[0], bilinear)
        self.up4 = Up(feats[0] + feats[0], feats[0], bilinear)
        self.outc = OutConv(feats[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model(
    name: str = "unet",
    in_channels: int = 3,
    num_classes: int = 3,
    pretrained_backbone: bool = False,
):
    name = name.lower()
    if name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes)
    if name == "deeplab":
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT" if pretrained_backbone else None)
        in_features = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_features, num_classes, kernel_size=1)
        if model.aux_classifier:
            aux_in = model.aux_classifier[-1].in_channels
            model.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)
        return model
    raise ValueError(f"Unknown model name: {name}")
