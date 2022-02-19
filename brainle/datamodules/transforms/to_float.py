from typing import Any

import torch
import torchvision.transforms.functional as F


class ToFloat:
    """
    Converts image to float32
    """

    def __call__(self, input: Any) -> Any:
        return F.convert_image_dtype(input, torch.float)
