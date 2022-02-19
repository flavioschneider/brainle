from typing import List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import DropPath, LayerNorm, trunc_normal_


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depths: List[int] = [3, 3],
        dims: List[int] = [96, 192],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.num_stages = len(dims)

        # Initial stem downsamples by a factor of 4
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                )
            ]
        )

        # All subsequent downsamples are by a factor of 2
        for i in range(self.num_stages - 1):
            self.downsample_layers += [
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            ]

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(self.num_stages):
            self.stages += [
                nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dim=dims[i],
                            drop_path=0,
                            layer_scale_init_value=layer_scale_init_value,
                        )
                        for j in range(depths[i])
                    ]
                )
            ]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


class ConvNeXtDecoder(nn.Module):
    def __init__(
        self,
        in_channels=192,
        depths: List[int] = [3, 3, 3],
        dims: List[int] = [96, 24, 3],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.num_stages = len(dims)

        self.upsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels=dims[0], kernel_size=2, stride=2
                    ),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                )
            ]
        )
        for i in range(self.num_stages - 1):
            self.upsample_layers += [
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(
                        dims[i], out_channels=dims[i + 1], kernel_size=2, stride=2
                    ),
                )
            ]

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages += [
                nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dim=dims[i],
                            drop_path=0,
                            layer_scale_init_value=layer_scale_init_value,
                        )
                        for j in range(depths[i])
                    ]
                )
            ]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.num_stages):
            x = self.upsample_layers[i](x)
            x = self.stages[i](x)
        return x
