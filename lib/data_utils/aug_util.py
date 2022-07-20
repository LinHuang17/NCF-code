#!/usr/bin/env python3

"""Image augmentation operations."""


import random
from typing import Iterable, Tuple, NamedTuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL.Image import Image as ImageType


class AugmentOp(NamedTuple):
    """Parameters of an augmentation operation.

    name: Name of the augmentation operation.
    prob: Probability with which the operation is applied.
    param_range: A range from which to sample the value of the key parameter
        (each augmentation operation is assumed to have one key parameter).
    """

    name: str
    prob: float
    param_range: Tuple[float, float]


def _augment_pil_filter(
    im: ImageType,
    fn: ImageFilter.MultibandFilter,
    prob: float,
    param_range: Tuple[float, float],
) -> ImageType:
    """Generic function for augmentations based on PIL's filter function.

    Args:
        im: An input image.
        fn: A filtering function to apply to the image.
        prob: Probability with which the function is applied.
        param_range: A range from which the value of the key parameter is sampled.
    Returns:
        A potentially augmented image.
    """

    if random.random() <= prob:
        im = im.filter(fn(random.randint(*map(int, param_range))))  # pyre-ignore
    return im


def _augment_pil_enhance(
    im: ImageType,
    fn: ImageEnhance._Enhance,
    prob: float,
    param_range: Tuple[float, float],
) -> ImageType:
    """Generic function for augmentations based on PIL's enhance function.

    Args:
        im: An input image.
        fn: A filtering function to apply to the image.
        prob: Probability with which the function is applied.
        param_range: A range from which the value of the key parameter is sampled.
    Returns:
        A potentially augmented image.
    """

    if random.random() <= prob:
        im = fn(im).enhance(factor=random.uniform(*param_range))  # pyre-ignore
    return im


def blur(im, prob=0.5, param_range=(1, 3)):
    return _augment_pil_filter(im, ImageFilter.GaussianBlur, prob, param_range)


def sharpness(im, prob=0.5, param_range=(0.0, 50.0)):
    return _augment_pil_enhance(im, ImageEnhance.Sharpness, prob, param_range)


def contrast(im, prob=0.5, param_range=(0.2, 50.0)):
    return _augment_pil_enhance(im, ImageEnhance.Contrast, prob, param_range)


def brightness(im, prob=0.5, param_range=(0.1, 6.0)):
    return _augment_pil_enhance(im, ImageEnhance.Brightness, prob, param_range)


def color(im, prob=0.5, param_range=(0.0, 20.0)):
    return _augment_pil_enhance(im, ImageEnhance.Color, prob, param_range)


# def augment_image(im: np.ndarray, augment_ops: Iterable[AugmentOp]) -> np.ndarray:
#     """Applies a list of augmentations to an image.

#     Args:
#         im: An input image.
#         augment_ops: A list of augmentations to apply.
#     Returns:
#         A potentially augmented image.
#     """

#     im_pil = Image.fromarray(im)
#     for op in augment_ops:
#         im_pil = globals()[op.name](im_pil, op.prob, op.param_range)
#     return np.array(im_pil)

def augment_image(im: ImageType, augment_ops: Iterable[AugmentOp]) -> ImageType:
    """Applies a list of augmentations to an image.

    Args:
        im: An input image.
        augment_ops: A list of augmentations to apply.
    Returns:
        A potentially augmented image.
    """

    # im_pil = Image.fromarray(im)
    im_pil = im
    for op in augment_ops:
        im_pil = globals()[op.name](im_pil, op.prob, op.param_range)
    return im_pil
