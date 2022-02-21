import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels, out_channels=num_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: int,
        res_channels: int,
        num_downscales: int,
        num_blocks: int,
    ):
        super().__init__()

        blocks = []

        # Downscaling blocks
        for i in range(num_downscales):
            blocks += [
                nn.Conv2d(
                    in_channels=num_channels // 2 if i > 0 else in_channels,
                    out_channels=num_channels // 2
                    if i < num_downscales - 1
                    else num_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ]

        # Residual blocks
        for i in range(num_blocks):
            blocks += [
                ResidualBlock(num_channels=num_channels, mid_channels=res_channels),
                nn.ReLU(),
            ]

        # Output head block
        blocks += [
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        ]

        self.module = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: int,
        res_channels: int,
        num_upscales: int,
        num_blocks: int,
    ):
        super().__init__()

        blocks = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
            )
        ]

        # Residual blocks
        for i in range(num_blocks):
            blocks += [
                ResidualBlock(num_channels=num_channels, mid_channels=res_channels),
                nn.ReLU(),
            ]

        # Upscaling blocks
        for i in range(num_upscales):
            blocks += [
                nn.ConvTranspose2d(
                    in_channels=num_channels // 2 if i > 0 else num_channels,
                    out_channels=num_channels // 2
                    if i < num_upscales - 1
                    else out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ]
            if i < num_upscales - 1:
                blocks += [nn.ReLU()]

        self.module = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
