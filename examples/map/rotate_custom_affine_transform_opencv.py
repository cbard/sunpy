#!/usr/bin/env python
# coding: utf-8

# # Using a Custom Affine Transform in the Sunpy Map rotate() function
# #### (with OpenCV)
# ----------
# How to construct a custom affine transform for `map.rotate`.
# Requires OpenCV (cv2) Python library.

import sunpy.data.sample
import sunpy.map
import numbers
import numpy as np
import cv2
from sunpy.image.transform import _calculate_shift

#############################
# Rotating a map in sunpy (via `map.rotate()`) has a choice between three libraries:
# `scipy`, `skimage`, and `cv2` (OpenCV).
# However, the `method=` argument in `sunpy.map.rotate` can accept a custom function designed
# to use an external library. Here, we illustrate this process by defining a function
# with the OpenCV library; this will be identical to the built-in `cv2` option.

# First, our custom method must have a similar function call to
# `sunpy.image.transform.affine_transform` and return the same output,
# namely the rotated, scaled, and transformed image.
# The required arguments are:

# Parameters
# ----------
# image: `numpy.ndarray`
#     2D image to be rotated
# rmatrix : `numpy.ndarray` that is 2x2
#     Linear transformation rotation matrix.
# order : `int` 0-5
#     Interpolation order to be used
# scale : `float`
#     A scale factor for the image
# missing : `float`
#    The value to replace any missing data after the transformation.
# image_center : tuple, optional
#     The point in the image to rotate around (axis of rotation).
#     Defaults to the center of the array.
# recenter : `bool` or array-like, optional
#     Move the axis of rotation to the center of the array or recenter coords.
#     Defaults to `True` i.e., recenter to the center of the array.


# NOTE: the required libraries have already been imported globally
# (see above: cv2, np, numbers)
def cv_rotate(image, rmatrix, order, scale, missing, image_center, recenter):
    """
    Uses `cv2.warpAffine` to do the affine transform on input `image` in same manner
    as sunpy's default `skimage.transform.warp`.
    """

    # Flags for converting input order from `integer` to the appropriate interpolation flag
    # As of Sept. 2020, OpenCV warpAffine does not support order 2,4,5
    _CV_ORDER_FLAGS = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_CUBIC,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_CUBIC,
        5: cv2.INTER_CUBIC
    }

    # convert order to appropriate cv2 flag
    order = _CV_ORDER_FLAGS[max(0, min(order, 5))]

    # needed to convert `missing` from potentially a np.dtype
    # to the native `int` type required for cv2.warpAffine
    try:
        missing = missing.tolist()
    except AttributeError:
        pass

    # OpenCV applies the shift+rotation operations in a different order(?); we need to calculate
    # translation using `rmatrix/scale`, but scale+rotation with `rmatrix*scale`
    # in order to match what skimage/scipy do

    shift = _calculate_shift(image, rmatrix/scale, image_center, recenter)

    rmatrix = rmatrix*scale

    trans = np.eye(3, 3)
    rot_scale = np.eye(3, 3)

    # openCV defines the translation matrix as [right, down]
    # but `_calculate_shift` returns [left,up], so we have to adjust
    trans[:2, 2] = [-shift[0], -shift[1]]

    # CV rotation is defined clockwise, so we transpose rmatrix
    rot_scale[:2, :2] = rmatrix.T
    rmatrix = (rot_scale @ trans)[:2]

    # cast input image to float, if needed
    # code adapted from sunpy.transform source code
    if issubclass(image.dtype.type, numbers.Integral):
        adjusted_image = image.astype(np.float64)
    else:
        adjusted_image = image.copy()

    h, w = adjusted_image.shape

    # equivalent to skimage.transform.warp(adjusted_image, tform, order=order,
    #                                     mode='constant', cval=adjusted_missing)
    return cv2.warpAffine(adjusted_image, rmatrix, (w, h), flags=order,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=missing)


# Now we test our implementation against the built-in openCV method
aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

map_r_cv = aia_map.rotate(order=3, recenter=True, method='cv2')
map_r_cv2 = aia_map.rotate(order=3, recenter=True, method=cv_rotate)

assert np.allclose(map_r_cv.data, map_r_cv2.data)
