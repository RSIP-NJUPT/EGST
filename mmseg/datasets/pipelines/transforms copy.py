import numpy as np
from numpy import random

import mmcv
from mmcv.utils import deprecated_api_warning
from ..builder import PIPELINES

@PIPELINES.register_module()
class AddSpeckleNoise(object):
    """Add Speckle Noise to the image.

    Added key is "img_noise_cfg".

    Args:
        L (int): The shape parameter for the gamma distribution.
        scale (float): The scaling factor for the noise, default is 1.0.
    """

    def __init__(self, L=1.0, scale=1.0):
        self.L = L  # Shape parameter for gamma distribution
        self.scale = scale

    def __call__(self, results):
        """Call function to add speckle noise.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Noisy results, 'img_noise_cfg' key is added into result dict.
        """
        img = results["img"].astype(np.float32)  # Ensure the image is float32 for processing

        # Generate gamma noise with specified shape parameter L
        noise = np.random.gamma(self.L, 1.0 / self.L, img.shape)

        min_val = np.min(img)
        max_val = np.max(img)

        # Normalize the image to [0, 255]
        data = ((img - min_val) / (max_val - min_val) * 255).astype(np.float32)

        # Apply speckle noise (multiplicative noise)
        noisy_image = data * noise * self.scale
        
        # Clip the noisy image to ensure values stay within [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # Reverse the normalization to the original image range
        results["img"] = (noisy_image / 255) * (max_val - min_val) + min_val

        # Add noise configuration to results
        results["img_noise_cfg"] = dict(L=self.L, scale=self.scale)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(L={self.L}, scale={self.scale})"
        return repr_str
