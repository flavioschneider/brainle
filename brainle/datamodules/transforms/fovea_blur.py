from typing import Tuple

import numpy as np
import cv2
import math
from PIL import Image


class FoveaBlur:
    def __init__(
        self,
        # size: int = 64,
        # position: Tuple[int, int] = [0,0],
        # The amount of blur as we get further away from the center of the fovea
        # blur_fn: Callable = lambda x: math.exp(x)
    ):
        # self.size = size
        # self.position = position
        # self.blur_fn = blur_fn
        pass

    def blur(self, image, radius, sigma=1):
        image = np.asarray(image)
        H, W, _ = image.shape
        mask = np.zeros(image.shape, np.uint8)
        mask = cv2.GaussianBlur(
            cv2.circle(
                mask,
                center=(H // 2, W // 2),
                radius=radius,
                color=(255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            ),
            ksize=(51, 51),
            sigmaX=5,
        )

        image2 = cv2.GaussianBlur(image, ksize=(51, 51), sigmaX=sigma)

        alpha = mask / 255.0
        return cv2.convertScaleAbs(image * alpha + image2 * (1 - alpha))

    def __call__(self, image):
        image_blurred = self.blur(image, 64, 10)
        image_blurred = self.blur(image_blurred, 32, 5)
        image_blurred = self.blur(image_blurred, 16, 2)
        image_blurred = self.blur(image_blurred, 8, 2)
        return Image.fromarray(image_blurred)
