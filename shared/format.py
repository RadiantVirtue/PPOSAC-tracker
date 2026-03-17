"""Observation preprocessing utilities.

Crafter returns Box(0,255,(64,64,3),uint8) observations directly.
SB3 handles normalisation internally. This module provides a helper
used by the analysis pipeline to convert uint8 numpy obs to float32 tensors.
"""

import numpy
import torch


def preprocess_images(images, device=None):
    """Convert a list/array of uint8 images to a float32 tensor normalised to [0, 1].

    Args:
        images: list of (H, W, C) uint8 arrays, or a single (N, H, W, C) array.
        device: optional torch device.

    Returns:
        Float32 tensor of shape (N, H, W, C) with values in [0, 1].
    """
    arr = numpy.array(images, dtype=numpy.float32) / 255.0
    return torch.tensor(arr, device=device, dtype=torch.float32)
