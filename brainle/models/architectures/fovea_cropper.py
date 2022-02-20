from typing import List

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from einops import rearrange, reduce


class FoveaCropper(nn.Module):
    def __init__(self, out_size: int, sizes: List[int]):
        super().__init__()
        self.out_size = out_size
        self.sizes = sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Crops images of different sizes from the center of x and resizes them to the same out_size, rearranging them as new channels.
        """
        crops = []
        for s in self.sizes:
            crops += [
                F.resize(
                    self.crop(x, s),
                    size=self.out_size,
                    interpolation=F.InterpolationMode.BILINEAR,
                )
            ]

        return rearrange(crops, "n b c h w -> b (c n) h w")

    def crop(self, x: torch.Tensor, size: int) -> torch.Tensor:
        b, c, h, w = x.shape
        return F.crop(
            x, top=h // 2 - size // 2, left=w // 2 - size // 2, height=size, width=size
        )

    def get_overlap(self, y: torch.Tensor, size: int) -> torch.Tensor:
        """
        Build an image that is the overlapped version of all crops, for visualization purposes.
        y: the tensor returned by forward of shape [b, c*n, h, w]
        size: the image output size
        output: a tensor of shape [3, h, w]
        """
        b = y.shape[0]
        crops = rearrange(y, "b (c n) h w -> n b c h w", c=3)
        output = torch.zeros([b, 3, size, size]).to(y)
        for crop, s in reversed(list(zip(crops, self.sizes))):
            crop_upscaled = F.resize(
                crop, size=s, interpolation=F.InterpolationMode.NEAREST
            )
            corner = size // 2 - s // 2
            output[:, :, corner : corner + s, corner : corner + s] = crop_upscaled
        return output
