"""
Functions for geometrical image transformation and warping.
"""
import numbers
import warnings

import numpy as np
import scipy.ndimage.interpolation

from sunpy.util.exceptions import SunpyUserWarning

try:
    import cv2
    HAS_CV = True
except ImportError:
    HAS_CV = False

__all__ = ['affine_transform']

# Flags for converting input order to appropriate cv2 interpolation flag
_CV_ORDER_FLAGS = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_CUBIC
}

def affine_transform(image, rmatrix, order=3, scale=1.0, image_center=None,
                     recenter=False, missing=0.0, use_scipy=False, use_cv2=False):
    """
    Rotates, shifts and scales an image.

    Will use `skimage.transform.warp` unless scikit-image can't be imported
    then it will use`scipy.ndimage.affine_transform`.
    If `use_cv2` flag is enabled, will use `cv2.warpAffine`; otherwise will
    default to above behavior.

    Parameters
    ----------
    image : `numpy.ndarray`
        2D image to be rotated.
    rmatrix : `numpy.ndarray` that is 2x2
        Linear transformation rotation matrix.
    order : `int` 0-5, optional
        Interpolation order to be used, defaults to 3. When using scikit-image this parameter
        is passed into `skimage.transform.warp` (e.g., 3 corresponds to bi-cubic interpolation).
        When using scipy it is passed into
        `scipy.ndimage.affine_transform` where it controls the order of the spline.
    scale : `float`
        A scale factor for the image with the default being no scaling.
    image_center : tuple, optional
        The point in the image to rotate around (axis of rotation).
        Defaults to the center of the array.
    recenter : `bool` or array-like, optional
        Move the axis of rotation to the center of the array or recenter coords.
        Defaults to `True` i.e., recenter to the center of the array.
    missing : `float`, optional
        The value to replace any missing data after the transformation.
    use_scipy : `bool`, optional
        Force use of `scipy.ndimage.affine_transform`.
        Will set all "NaNs" in image to zero before doing the transform.
        Defaults to `False`, unless scikit-image can't be imported.
    use_cv2 : `bool`, optional
        Force use of `cv2.warpAffine`
        Will override `use_scipy`; if cv2 can't be imported, then function 
        will default to behavior defined by `use_scipy`

    Returns
    -------
    `numpy.ndarray`:
        New rotated, scaled and translated image.

    Notes
    -----
    This algorithm uses an affine transformation as opposed to a polynomial
    geometrical transformation, which by default is `skimage.transform.warp`.
    One can specify using `scipy.ndimage.affine_transform` as
    an alternative affine transformation. The two transformations use different
    algorithms and thus do not give identical output.
    One can also specify using `cv2.warpAffine` as an alternative affine
    transformation. This also uses a different algorithm, and does not give
    identical output.

    When using `skimage.transform.warp` with order >= 4 or using
    `scipy.ndimage.affine_transform` at all, "NaN" values will be replaced with
    zero prior to rotation. No attempt is made to retain the "NaN" values except when
    the entire image consists of "NaN" values only and `skimage.transform.warp` is used,
    where it is passed down as it is.

    When using for `cv2.warpAffine`, only order = 0, 1, 3 are accepted.
    order = 2 or order > 3 will default to order = 3.
    order < 0 will default to 0.

    Input arrays with integer data are cast to float 64 and can be re-cast using
    `numpy.ndarray.astype` if desired.

    In the case of `skimage.transform.warp`, the image is normalized to [0, 1]
    before passing it to the function. It is later rescaled back to the original range.

    Although this function is analogous to the IDL's ``rot`` function, it does not
    use the same algorithm as the IDL ``rot`` function.
    IDL's ``rot`` calls the `POLY_2D <https://www.harrisgeospatial.com/docs/poly_2d.html>`__
    method to calculate the inverse mapping of original to target pixel
    coordinates. This is a polynomial geometrical transformation.
    Then optionally it uses a bicubic convolution interpolation
    algorithm to map the original to target pixel values.
    """
    rmatrix = rmatrix / scale
    array_center = (np.array(image.shape)[::-1] - 1) / 2.0

    # Make sure the image center is an array and is where it's supposed to be
    if image_center is not None:
        image_center = np.asanyarray(image_center)
    else:
        image_center = array_center

    # Determine center of rotation based on use (or not) of the recenter keyword
    if recenter:
        rot_center = array_center
    else:
        rot_center = image_center

    displacement = np.dot(rmatrix, rot_center)
    shift = image_center - displacement

    if use_cv2:
        try:
            from cv2 import warpAffine
            order = _CV_ORDER_FLAGS[max(0,min(order,4))]
            use_scipy = False
        except ImportError:
            warnings.warn("cv2.warpAffine could not be imported.",
                          "Image rotation will default to use_scipy={}".format(use_scipy), ImportWarning)
            use_cv2 = False

    if use_cv2:
        # needed to convert missing from np.dtype to native type
        # required for warpAffine input
        try:
            missing = missing.tolist()
        except AttributeError:
            pass

        # blurb for getting appropriate cv transform matrix
        rmatrix = rmatrix*scale*scale
        trans = np.eye(3,3)
        rot_scale = np.eye(3,3)
        trans[:2,2] = [-shift[0],-shift[1]]
        rot_scale[:2,:2] = rmatrix.T
        rmatrix = (rot_scale @ trans)[:2]

        if issubclass(image.dtype.type, numbers.Integral):
            warnings.warn("Integer input data has been cast to float64, "
                          "which is required for the opencv2-image transform.",
                          SunpyUserWarning)
            adjusted_image = image.astype(np.float64)
        else:
            adjusted_image = image.copy()

        h,w = adjusted_image.shape
        rotated_image = cv2.warpAffine(adjusted_image, rmatrix, (w,h), flags=order,borderMode=cv2.BORDER_CONSTANT, borderValue=missing)
        
    elif not use_scipy:
        try:
            import skimage.transform
        except ImportError:
            warnings.warn("scikit-image could not be imported. Image rotation will use scipy",
                          ImportWarning)
            use_scipy = True
    if use_scipy:
        if np.any(np.isnan(image)):
            warnings.warn("Setting NaNs to 0 for SciPy rotation.", SunpyUserWarning)
        # Transform the image using the scipy affine transform
        rotated_image = scipy.ndimage.interpolation.affine_transform(
            np.nan_to_num(image).T, rmatrix, offset=shift, order=order,
            mode='constant', cval=missing).T
    elif not use_cv2:
        # Make the rotation matrix 3x3 to include translation of the image
        skmatrix = np.zeros((3, 3))
        skmatrix[:2, :2] = rmatrix
        skmatrix[2, 2] = 1.0
        skmatrix[:2, 2] = shift
        tform = skimage.transform.AffineTransform(skmatrix)

        if issubclass(image.dtype.type, numbers.Integral):
            warnings.warn("Integer input data has been cast to float64, "
                          "which is required for the scikit-image transform.",
                          SunpyUserWarning)
            adjusted_image = image.astype(np.float64)
        else:
            adjusted_image = image.copy()
        if np.any(np.isnan(adjusted_image)) and order >= 4:
            warnings.warn("Setting NaNs to 0 for higher-order scikit-image rotation.",
                          SunpyUserWarning)
            adjusted_image = np.nan_to_num(adjusted_image)

        # Scale image to range [0, 1] if it is valid (not made up entirely of NaNs)
        is_nan_image = np.all(np.isnan(adjusted_image))
        if is_nan_image:
            adjusted_missing = missing
        else:
            im_min = np.nanmin(adjusted_image)
            adjusted_image -= im_min
            im_max = np.nanmax(adjusted_image)
            if im_max > 0:
                adjusted_image /= im_max
                adjusted_missing = (missing - im_min) / im_max
            else:
                adjusted_missing = missing - im_min

        rotated_image = skimage.transform.warp(adjusted_image, tform, order=order,
                                               mode='constant', cval=adjusted_missing, preserve_range=True)

        # Convert the image back to its original range if it is valid
        if not is_nan_image:
            if im_max > 0:
                rotated_image *= im_max
            rotated_image += im_min

    return rotated_image
