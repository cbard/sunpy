"""
Map is a generic Map class from which all other Map classes inherit from.
"""
from __future__ import absolute_import

#pylint: disable=E1101,E1121,W0404,W0613
__authors__ = ["Russell Hewett, Stuart Mumford, Keith Hughitt, Steven Christe"]
__email__ = "stuart@mumford.me.uk"

import warnings
import inspect
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm

import astropy.wcs
from .nddata_compat import NDDataCompat as NDData
from astropy.coordinates import Longitude, Latitude

from sunpy.image.transform import affine_transform

import sunpy.io as io
import sunpy.wcs as wcs
from sunpy.visualization import toggle_pylab, wcsaxes_compat
from sunpy.sun import constants
from sunpy.sun import sun
from sunpy.time import parse_time, is_time
from sunpy.image.rescale import reshape_image_to_4d_superpixel
from sunpy.image.rescale import resample as sunpy_image_resample

import astropy.units as u

from collections import namedtuple
Pair = namedtuple('Pair', 'x y')

__all__ = ['GenericMap']

from sunpy import config
TIME_FORMAT = config.get("general", "time_format")

"""
Questions
---------
* Should we use Helioviewer or VSO's data model? (e.g. map.meas, map.wavelength
or something else?)
* Should 'center' be renamed to 'offset' and crpix1 & 2 be used for 'center'?
"""

class GenericMap(NDData):
    """
    A Generic spatially-aware 2D data array

    Parameters
    ----------
    data : numpy.ndarray, list
        A 2d list or ndarray containing the map data
    header : dict
        A dictionary of the original image header tags

    Attributes
    ----------
    cmap : matplotlib.colors.Colormap
        A color map used for plotting with matplotlib.
    mpl_color_normalizer : matplotlib.colors.Normalize
        A matplotlib normalizer used to scale the image plot.

    Examples
    --------
    >>> import sunpy.map
    >>> aia = sunpy.map.Map(sunpy.AIA_171_IMAGE)
    >>> aia.T
    AIAMap([[ 0.3125,  1.    , -1.1875, ..., -0.625 ,  0.5625,  0.5   ],
    [-0.0625,  0.1875,  0.375 , ...,  0.0625,  0.0625, -0.125 ],
    [-0.125 , -0.8125, -0.5   , ..., -0.3125,  0.5625,  0.4375],
    ...,
    [ 0.625 ,  0.625 , -0.125 , ...,  0.125 , -0.0625,  0.6875],
    [-0.625 , -0.625 , -0.625 , ...,  0.125 , -0.0625,  0.6875],
    [ 0.    ,  0.    , -1.1875, ...,  0.125 ,  0.    ,  0.6875]])
    >>> aia.units['x']
    'arcsec'
    >>> aia.peek()

    References
    ----------
    | http://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    | http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    | http://docs.scipy.org/doc/numpy/reference/ufuncs.html
    | http://www.scipy.org/Subclasses

    Notes
    -----

    A number of the properties of this class are returned as two-value named
    tuples that can either be indexed by position ([0] or [1]) or be accessed by
    name (.x or .y).  The names "x" and "y" here refer to the first and second
    axes of the map, and may not necessarily correspond to any similarly named
    axes in the coordinate system.

    This class makes some assumptions about the WCS information contained in
    the meta data. The first and most extensive assumption is that it is
    FITS-like WCS information as defined in the FITS WCS papers.

    Within this scope it also makes some other assumptions.

    * In the case of APIS convention headers where the CROTAi/j arguments are
      provided it assumes that these can be converted to the standard PCi_j
      notation using equations 32 in Thompson (2006).

    * If a CDi_j matrix is provided it is assumed that it can be converted to a
      PCi_j matrix and CDELT keywords as descirbed in Greisen & Calabretta (2002).

    * The 'standard' FITS keywords that are used by this class are the PCi_j
      matrix and CDELT, along with the other keywords specified in the WCS papers.
      All subclasses of this class must convert their header information to
      this formalism. The CROTA to PCi_j conversion is done in this class.

    .. warning::
        This class currently assumes that a header with the CDi_j matrix
        information also includes the CDELT keywords, without these keywords
        this class will not process the WCS information. This will be fixed.
        Also the rotation_matrix does not work if the CDELT1 and CDELT2
        keywords are exactly equal.
    """

    def __init__(self, data, header, **kwargs):

        super(GenericMap, self).__init__(data, meta=header, **kwargs)

        # Correct possibly missing meta keywords
        self._fix_date()
        self._fix_naxis()

        # Setup some attributes
        self._name = self.observatory + " " + str(self.measurement)
        self._nickname = self.detector

        # Validate header
        # TODO: This should be a function of the header, not of the map
        self._validate()

        # Visualization attributes
        self.plot_settings = {'cmap': cm.gray,
                              'norm': self._get_mpl_normalizer(),
                              'title': "{name} {date:{tmf}}".format(name=self.name,
                                                                    date=parse_time(self.date),
                                                                    tmf=TIME_FORMAT)
                              }

    def __getitem__(self, key):
        """ This should allow indexing by physical coordinate """
        raise NotImplementedError(
    "The ability to index Map by physical coordinate is not yet implemented.")

    def __repr__(self):
        if not self.observatory:
            return self.data.__repr__()
        return (
"""SunPy {dtype!s}
---------
Observatory:\t {obs}
Instrument:\t {inst}
Detector:\t {det}
Measurement:\t {meas:0.0f}
Obs Date:\t {date}
dt:\t\t {dt:f}
Dimension:\t {dim}
scale:\t\t [{dx}, {dy}]

""".format(dtype=self.__class__.__name__,
           obs=self.observatory, inst=self.instrument, det=self.detector,
           meas=self.measurement, date=self.date, dt=self.exposure_time,
           dim=u.Quantity(self.dimensions),
           dx=self.scale.x, dy=self.scale.y)
+ self.data.__repr__())

    @property
    def wcs(self):
        w2 = astropy.wcs.WCS(naxis=2)
        w2.wcs.crpix = u.Quantity(self.reference_pixel)
        # Make these a quantity array to prevent the numpy setting element of
        # array with sequence error.
        w2.wcs.cdelt = u.Quantity(self.scale)
        w2.wcs.crval = u.Quantity(self.reference_coordinate)
        w2.wcs.ctype = self.coordinate_system
        w2.wcs.pc = self.rotation_matrix
        w2.wcs.cunit = self.units

        return w2

    #Some numpy extraction
    @property
    def dimensions(self):
        """
        The dimensions of the array (x axis first, y axis second).
        """
        return Pair(*u.Quantity(np.flipud(self.data.shape), 'pixel'))

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return u.Quantity(self.data.size, 'pixel')

    @property
    def ndim(self):
        return self.data.ndim

    def std(self, *args, **kwargs):
        return self.data.std(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.data.mean(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self.data.min(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self.data.max(*args, **kwargs)

# #### Keyword attribute and other attribute definitions #### #

    @property
    def name(self):
        """Human-readable description of map-type"""
        return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def nickname(self):
        """An abbreviated human-readable description of the map-type; part of the Helioviewer data model"""
        return self._nickname

    @nickname.setter
    def nickname(self, n):
        self._nickname = n

    @property
    def date(self):
        """Image observation time"""
        time = parse_time(self.meta.get('date-obs', 'now'))
        if time is None:
            warnings.warn_explicit("Missing metadata for observation time. Using current time.",
                                       Warning, __file__, inspect.currentframe().f_back.f_lineno)
        return parse_time(time)

#    @date.setter
#    def date(self, new_date):
#        self.meta['date-obs'] = new_date
#        #propagate change to malformed FITS keywords
#        if is_time(self.meta.get('date_obs', None)):
#            self.meta['date_obs'] = new_date

    @property
    def detector(self):
        """Detector name"""
        return self.meta.get('detector', "")

    @property
    def dsun(self):
        """
        The observer distance from the Sun.
        """
        dsun = self.meta.get('dsun_obs', None)

        if dsun is None:
            warnings.warn_explicit("Missing metadata for Sun-spacecraft separation: assuming Sun-Earth distance",
                                   Warning, __file__, inspect.currentframe().f_back.f_lineno)
            dsun = sun.sunearth_distance(self.date).to(u.m)

        return u.Quantity(dsun, 'm')

    @property
    def exposure_time(self):
        """Exposure time of the image in seconds."""
        return self.meta.get('exptime', 0.0) * u.s

    @property
    def instrument(self):
        """Instrument name"""
        return self.meta.get('instrume', "")

    @property
    def measurement(self):
        """Measurement name, defaults to the wavelength of image"""
        return u.Quantity(self.meta.get('wavelnth', 0), self.meta.get('waveunit', ""))

    @property
    def wavelength(self):
        """wavelength of the observation"""
        return u.Quantity(self.meta.get('wavelnth', 0), self.meta.get('waveunit', ""))

    @property
    def observatory(self):
        """Observatory or Telescope name"""
        return self.meta.get('obsrvtry', self.meta.get('telescop', ""))

    @property
    def xrange(self):
        """Return the X range of the image in arcsec from edge to edge."""
        xmin = self.center.x - self.dimensions[0] / 2. * self.scale.x
        xmax = self.center.x + self.dimensions[0] / 2. * self.scale.x
        return u.Quantity([xmin, xmax])

    @property
    def yrange(self):
        """Return the Y range of the image in arcsec from edge to edge."""
        ymin = self.center.y - self.dimensions[1] / 2. * self.scale.y
        ymax = self.center.y + self.dimensions[1] / 2. * self.scale.y
        return u.Quantity([ymin, ymax])

    @property
    def center(self):
        """Returns the offset between the center of the Sun and the center of
        the map."""
        return Pair(wcs.get_center(self.dimensions[0], self.scale.x,
                                   self.reference_pixel.x,
                                   self.reference_coordinate.x),
                    wcs.get_center(self.dimensions[1], self.scale.y,
                                   self.reference_pixel.y,
                                   self.reference_coordinate.y))

    @property
    def rsun_meters(self):
        """Radius of the sun in meters"""
        return u.Quantity(self.meta.get('rsun_ref', constants.radius), 'meter')

    @property
    def rsun_obs(self):
        """Radius of the sun in arcseconds"""
        rsun_arcseconds = self.meta.get('rsun_obs',
                                        self.meta.get('solar_r',
                                                      self.meta.get('radius', None)))

        if rsun_arcseconds is None:
            warnings.warn_explicit("Missing metadata for solar radius: assuming photospheric limb as seen from Earth",
                                   Warning, __file__, inspect.currentframe().f_back.f_lineno)
            rsun_arcseconds = sun.solar_semidiameter_angular_size(self.date).to('arcsec').value

        return u.Quantity(rsun_arcseconds, 'arcsec')

    @property
    def coordinate_system(self):
        """Coordinate system used for x and y axes (ctype1/2)"""
        return Pair(self.meta.get('ctype1', 'HPLN-TAN'),
                    self.meta.get('ctype2', 'HPLT-TAN'))

    @property
    def carrington_longitude(self):
        """Carrington longitude (crln_obs)"""
        carrington_longitude = self.meta.get('crln_obs', None)

        if carrington_longitude is None:
            warnings.warn_explicit("Missing metadata for Carrington longitude: assuming Earth-based observer",
                                   Warning, __file__, inspect.currentframe().f_back.f_lineno)
            carrington_longitude = (sun.heliographic_solar_center(self.date))[0]

        return u.Quantity(carrington_longitude, 'deg')

    @property
    def heliographic_latitude(self):
        """Heliographic latitude in degrees"""
        heliographic_latitude = self.meta.get('hglt_obs',
                                              self.meta.get('crlt_obs',
                                                            self.meta.get('solar_b0', None)))

        if heliographic_latitude is None:
            warnings.warn_explicit("Missing metadata for heliographic latitude: assuming Earth-based observer",
                                   Warning, __file__, inspect.currentframe().f_back.f_lineno)
            heliographic_latitude = (sun.heliographic_solar_center(self.date))[1]

        return u.Quantity(heliographic_latitude, 'deg')

    @property
    def heliographic_longitude(self):
        """Heliographic longitude in degrees"""
        return u.Quantity(self.meta.get('hgln_obs', 0.), 'deg')

    @property
    def reference_coordinate(self):
        """Reference point WCS axes in data units (crval1/2)"""
        return Pair(self.meta.get('crval1', 0.) * self.units.x,
                    self.meta.get('crval2', 0.) * self.units.y)

    @property
    def reference_pixel(self):
        """Reference point axes in pixels (crpix1/2)"""
        return Pair(self.meta.get('crpix1', (self.meta.get('naxis1') + 1) / 2.) * u.pixel,
                    self.meta.get('crpix2', (self.meta.get('naxis2') + 1) / 2.) * u.pixel)

    @property
    def scale(self):
        """Image scale along the x and y axes in units/pixel (cdelt1/2)"""
        #TODO: Fix this if only CDi_j matrix is provided
        return Pair(self.meta.get('cdelt1', 1.) * self.units.x / u.pixel,
                    self.meta.get('cdelt2', 1.) * self.units.y / u.pixel)

    @property
    def units(self):
        """Image coordinate units along the x and y axes (cunit1/2)."""
        return Pair(u.Unit(self.meta.get('cunit1', 'arcsec')),
                    u.Unit(self.meta.get('cunit2', 'arcsec')))

    @property
    def rotation_matrix(self):
        """Matrix describing the rotation required to align solar North with
        the top of the image."""
        if 'PC1_1' in self.meta:
            return np.matrix([[self.meta['PC1_1'], self.meta['PC1_2']],
                              [self.meta['PC2_1'], self.meta['PC2_2']]])

        elif 'CD1_1' in self.meta:
            cd = np.matrix([[self.meta['CD1_1'], self.meta['CD1_2']],
                            [self.meta['CD2_1'], self.meta['CD2_2']]])

            cdelt = u.Quantity(self.scale).value

            return cd / cdelt
        else:
            return self._rotation_matrix_from_crota()

    def _rotation_matrix_from_crota(self):
        """
        This method converts the deprecated CROTA FITS kwargs to the new
        PC rotation matrix.

        This method can be overriden if an instruments header does not use this
        conversion.
        """
        lam = self.scale.y / self.scale.x
        p = np.deg2rad(self.meta.get('CROTA2', 0))

        return np.matrix([[np.cos(p), -1 * lam * np.sin(p)],
                          [1/lam * np.sin(p), np.cos(p)]])

# #### Miscellaneous #### #

    def _fix_date(self):
        # Check commonly used but non-standard FITS keyword for observation time
        # and correct the keyword if we can.  Keep updating old one for
        # backwards compatibility.
        if is_time(self.meta.get('date_obs', None)):
            self.meta['date-obs'] = self.meta['date_obs']

    def _fix_naxis(self):
        # If naxis is not specified, get it from the array shape
        if 'naxis1' not in self.meta:
            self.meta['naxis1'] = self.data.shape[1]
        if 'naxis2' not in self.meta:
            self.meta['naxis2'] = self.data.shape[0]
        if 'naxis' not in self.meta:
            self.meta['naxis'] = self.ndim

    def _fix_bitpix(self):
        # Bit-depth
        #
        #   8    Character or unsigned binary integer
        #  16    16-bit twos-complement binary integer
        #  32    32-bit twos-complement binary integer
        # -32    IEEE single precision floating point
        # -64    IEEE double precision floating point
        #
        if 'bitpix' not in self.meta:
            float_fac = -1 if self.dtype.kind == "f" else 1
            self.meta['bitpix'] = float_fac * 8 * self.dtype.itemsize

    def _get_cmap_name(self):
        """Build the default color map name."""
        cmap_string = self.observatory + self.meta['detector'] + str(int(self.wavelength.to('angstrom').value))
        return cmap_string.lower()

    def _validate(self):
        """Validates the meta-information associated with a Map.

        This function includes very basic validation checks which apply to
        all of the kinds of files that SunPy can read. Datasource-specific
        validation should be handled in the relevant file in the
        sunpy.map.sources package."""
#        if (self.dsun <= 0 or self.dsun >= 40 * constants.au):
#            raise InvalidHeaderInformation("Invalid value for DSUN")
        pass

# #### Data conversion routines #### #

    @u.quantity_input(x=u.deg, y=u.deg)
    def data_to_pixel(self, x, y, origin=0):
        """
        Convert a data (world) coordinate to a pixel coordinate by using
        `~astropy.wcs.WCS.wcs_world2pix`.

        Parameters
        ----------

        x : `~astropy.units.Quantity`
            Data coordinate of the CTYPE1 axis. (Normally solar-x).

        y : `~astropy.units.Quantity`
            Data coordinate of the CTYPE2 axis. (Normally solar-y).

        origin : int
            Origin of the top-left corner. i.e. count from 0 or 1.
            Normally, origin should be 0 when passing numpy indicies, or 1 if
            passing values from FITS header or map attributes.
            See `~astropy.wcs.WCS.wcs_world2pix` for more information.

        Returns
        -------

        x : float
            Pixel coordinate on the CTYPE1 axis.

        y : float
            Pixel coordinate on the CTYPE2 axis.
        """
        x, y = self.wcs.wcs_world2pix(x.to(u.deg).value, y.to(u.deg).value, origin)

        return x * u.pixel, y * u.pixel

    @u.quantity_input(x=u.pixel, y=u.pixel)
    def pixel_to_data(self, x, y, origin=0):
        """
        Convert a pixel coordinate to a data (world) coordinate by using
        `~astropy.wcs.WCS.wcs_pix2world`.

        Parameters
        ----------

        x : float
            Pixel coordinate of the CTYPE1 axis. (Normally solar-x).

        y : float
            Pixel coordinate of the CTYPE2 axis. (Normally solar-y).

        origin : int
            Origin of the top-left corner. i.e. count from 0 or 1.
            Normally, origin should be 0 when passing numpy indicies, or 1 if
            passing values from FITS header or map attributes.
            See `~astropy.wcs.WCS.wcs_pix2world` for more information.

        Returns
        -------

        x : `~astropy.units.Quantity`
            Coordinate of the CTYPE1 axis. (Normally solar-x).

        y : `~astropy.units.Quantity`
            Coordinate of the CTYPE2 axis. (Normally solar-y).
        """
        x, y = self.wcs.wcs_pix2world(x, y, origin)

        # WCS always outputs degrees.
        x *= u.deg
        y *= u.deg

        x = Longitude(x, wrap_angle=180*u.deg)
        y = Latitude(y)

        return x.to(self.units.x), y.to(self.units.y)


# #### I/O routines #### #

    def save(self, filepath, filetype='auto', **kwargs):
        """Saves the SunPy Map object to a file.

        Currently SunPy can only save files in the FITS format. In the future
        support will be added for saving to other formats.

        Parameters
        ----------
        filepath : string
            Location to save file to.

        filetype : string
            'auto' or any supported file extension
        """
        io.write_file(filepath, self.data, self.meta, filetype=filetype,
                      **kwargs)

# #### Image processing routines #### #

    @u.quantity_input(dimensions=u.pixel)
    def resample(self, dimensions, method='linear'):
        """Returns a new Map that has been resampled up or down

        Arbitrary resampling of the Map to new dimension sizes.

        Uses the same parameters and creates the same co-ordinate lookup points
        as IDL''s congrid routine, which apparently originally came from a
        VAX/VMS routine of the same name.

        Parameters
        ----------
        dimensions : `~astropy.units.Quantity`
            Pixel dimensions that new Map should have.
            Note: the first argument corresponds to the 'x' axis and the second
            argument corresponds to the 'y' axis.
        method : {'neighbor' | 'nearest' | 'linear' | 'spline'}
            Method to use for resampling interpolation.
                * neighbor - Closest value from original data
                * nearest and linear - Uses n x 1-D interpolations using
                  scipy.interpolate.interp1d
                * spline - Uses ndimage.map_coordinates

        Returns
        -------
        out : Map
            A new Map which has been resampled to the desired dimensions.

        References
        ----------
        | http://www.scipy.org/Cookbook/Rebinning (Original source, 2011/11/19)
        """

        # Note: because the underlying ndarray is transposed in sense when
        #   compared to the Map, the ndarray is transposed, resampled, then
        #   transposed back
        # Note: "center" defaults to True in this function because data
        #   coordinates in a Map are at pixel centers

        # Make a copy of the original data and perform resample
        new_data = sunpy_image_resample(self.data.copy().T, dimensions,
                                    method, center=True)
        new_data = new_data.T

        scale_factor_x = float(self.dimensions[0] / dimensions[0])
        scale_factor_y = float(self.dimensions[1] / dimensions[1])

        new_map = deepcopy(self)
        # Update image scale and number of pixels
        new_meta = self.meta.copy()

        # Update metadata
        new_meta['cdelt1'] *= scale_factor_x
        new_meta['cdelt2'] *= scale_factor_y
        if 'CD1_1' in new_meta:
            new_meta['CD1_1'] *= scale_factor_x
            new_meta['CD2_1'] *= scale_factor_x
            new_meta['CD1_2'] *= scale_factor_y
            new_meta['CD2_2'] *= scale_factor_y
        new_meta['crpix1'] = (dimensions[0].value + 1) / 2.
        new_meta['crpix2'] = (dimensions[1].value + 1) / 2.
        new_meta['crval1'] = self.center.x.value
        new_meta['crval2'] = self.center.y.value

        # Create new map instance
        new_map.data = new_data
        new_map.meta = new_meta
        return new_map

    def rotate(self, angle=None, rmatrix=None, order=4, scale=1.0,
               recenter=False, missing=0.0, use_scipy=False):
        """
        Returns a new rotated and rescaled map.  Specify either a rotation
        angle or a rotation matrix, but not both.  If neither an angle or a
        rotation matrix are specified, the map will be rotated by the rotation
        angle in the metadata.

        The map will be rotated around the reference coordinate defined in the
        meta data.

        Also updates the rotation_matrix attribute and any appropriate header
        data so that they correctly describe the new map.

        Parameters
        ----------
        angle : `~astropy.units.Quantity`
            The angle (degrees) to rotate counterclockwise.
        rmatrix : 2x2
            Linear transformation rotation matrix.
        order : int 0-5
            Interpolation order to be used. When using scikit-image this parameter
            is passed into :func:`skimage.transform.warp` (e.g., 4 corresponds to
            bi-quartic interpolation).
            When using scipy it is passed into
            :func:`scipy.ndimage.interpolation.affine_transform` where it controls
            the order of the spline.
            Faster performance may be obtained at the cost of accuracy by using lower values.
            Default: 4
        scale : float
            A scale factor for the image, default is no scaling
        recenter : bool
            If True, position the axis of rotation at the center of the new map
            Default: False
        missing : float
            The numerical value to fill any missing points after rotation.
            Default: 0.0
        use_scipy : bool
            If True, forces the rotation to use
            :func:`scipy.ndimage.interpolation.affine_transform`, otherwise it
            uses the :func:`skimage.transform.warp`.
            Default: False, unless scikit-image can't be imported

        Returns
        -------
        out : Map
            A new Map instance containing the rotated and rescaled data of the
            original map.

        See Also
        --------
        sunpy.image.transform.affine_transform : The routine this method calls for the rotation.

        Notes
        -----
        This function will remove old CROTA keywords from the header.
        This function will also convert a CDi_j matrix to a PCi_j matrix.

        See :func:`sunpy.image.transform.affine_transform` for details on the
        transformations, situations when the underlying data is modified prior to rotation,
        and differences from IDL's rot().
        """
        if angle is not None and rmatrix is not None:
            raise ValueError("You cannot specify both an angle and a matrix")
        elif angle is None and rmatrix is None:
            rmatrix = self.rotation_matrix

        # This is out of the quantity_input decorator. To allow the angle=None
        # case. See https://github.com/astropy/astropy/issues/3734
        if angle:
            try:
                equivalent = angle.unit.is_equivalent(u.deg)

                if not equivalent:
                    raise u.UnitsError("Argument '{0}' to function '{1}'"
                                       " must be in units convertable to"
                                       " '{2}'.".format('angle', 'rotate',
                                                      u.deg.to_string()))

            # Either there is no .unit or no .is_equivalent
            except AttributeError:
                if hasattr(angle, "unit"):
                    error_msg = "a 'unit' attribute without an 'is_equivalent' method"
                else:
                    error_msg = "no 'unit' attribute"
                raise TypeError("Argument '{0}' to function '{1}' has {2}. "
                      "You may want to pass in an astropy Quantity instead."
                         .format('angle', 'rotate', error_msg))

        # Interpolation parameter sanity
        if order not in range(6):
            raise ValueError("Order must be between 0 and 5")

        # The FITS-WCS transform is by definition defined around the
        # reference coordinate in the header.
        rotation_center = u.Quantity([self.reference_coordinate.x,
                                      self.reference_coordinate.y])

        # Copy Map
        new_map = deepcopy(self)

        if angle is not None:
            # Calulate the parameters for the affine_transform
            c = np.cos(np.deg2rad(angle))
            s = np.sin(np.deg2rad(angle))
            rmatrix = np.matrix([[c, -s], [s, c]])

        # Calculate the shape in pixels to contain all of the image data
        extent = np.max(np.abs(np.vstack((new_map.data.shape * rmatrix,
                                          new_map.data.shape * rmatrix.T))), axis=0)
        # Calculate the needed padding or unpadding
        diff = np.asarray(np.ceil((extent - new_map.data.shape) / 2)).ravel()
        # Pad the image array
        pad_x = np.max((diff[1], 0))
        pad_y = np.max((diff[0], 0))
        new_map.data = np.pad(new_map.data,
                              ((pad_y, pad_y), (pad_x, pad_x)),
                              mode='constant',
                              constant_values=(missing, missing))
        new_map.meta['crpix1'] += pad_x
        new_map.meta['crpix2'] += pad_y

        # All of the following pixel calculations use a pixel origin of 0

        pixel_array_center = (np.flipud(new_map.data.shape) - 1) / 2.0

        # Convert the axis of rotation from data coordinates to pixel coordinates
        pixel_rotation_center = u.Quantity(new_map.data_to_pixel(*rotation_center,
                                                                 origin=0)).value
        if recenter:
            pixel_center = pixel_rotation_center
        else:
            pixel_center = pixel_array_center

        # Apply the rotation to the image data
        new_map.data = affine_transform(new_map.data.T,
                                        np.asarray(rmatrix),
                                        order=order, scale=scale,
                                        image_center=np.flipud(pixel_center),
                                        recenter=recenter, missing=missing,
                                        use_scipy=use_scipy).T

        if recenter:
            new_reference_pixel = pixel_array_center
        else:
            # Calculate new pixel coordinates for the rotation center
            new_reference_pixel = pixel_center + np.dot(rmatrix,
                                                        pixel_rotation_center - pixel_center)
            new_reference_pixel = np.array(new_reference_pixel).ravel()

        # Define the new reference_pixel
        new_map.meta['crval1'] = rotation_center[0].value
        new_map.meta['crval2'] = rotation_center[1].value
        new_map.meta['crpix1'] = new_reference_pixel[0] + 1 # FITS pixel origin is 1
        new_map.meta['crpix2'] = new_reference_pixel[1] + 1 # FITS pixel origin is 1

        # Unpad the array if necessary
        unpad_x = -np.min((diff[1], 0))
        if unpad_x > 0:
            new_map.data = new_map.data[:, unpad_x:-unpad_x]
            new_map.meta['crpix1'] -= unpad_x
        unpad_y = -np.min((diff[0], 0))
        if unpad_y > 0:
            new_map.data = new_map.data[unpad_y:-unpad_y, :]
            new_map.meta['crpix2'] -= unpad_y

        # Calculate the new rotation matrix to store in the header by
        # "subtracting" the rotation matrix used in the rotate from the old one
        # That being calculate the dot product of the old header data with the
        # inverse of the rotation matrix.
        pc_C = np.dot(new_map.rotation_matrix, rmatrix.I)
        new_map.meta['PC1_1'] = pc_C[0,0]
        new_map.meta['PC1_2'] = pc_C[0,1]
        new_map.meta['PC2_1'] = pc_C[1,0]
        new_map.meta['PC2_2'] = pc_C[1,1]

        # Update pixel size if image has been scaled.
        if scale != 1.0:
            new_map.meta['cdelt1'] = (new_map.scale.x / scale).value
            new_map.meta['cdelt2'] = (new_map.scale.y / scale).value

        # Remove old CROTA kwargs because we have saved a new PCi_j matrix.
        new_map.meta.pop('CROTA1', None)
        new_map.meta.pop('CROTA2', None)
        # Remove CDi_j header
        new_map.meta.pop('CD1_1', None)
        new_map.meta.pop('CD1_2', None)
        new_map.meta.pop('CD2_1', None)
        new_map.meta.pop('CD2_2', None)

        return new_map


    def submap(self, range_a, range_b):
        """
        Returns a submap of the map with the specified range.


        Parameters
        ----------
        range_a : `astropy.units.Quantity`
            The range of the Map to select across either the x axis.
            Can be either in data units (normally arcseconds) or pixel units.
        range_b : `astropy.units.Quantity`
            The range of the Map to select across either the y axis.
            Can be either in data units (normally arcseconds) or pixel units.

        Returns
        -------
        out : Map
            A new map instance is returned representing to specified sub-region

        Examples
        --------
        >>> aia.submap([-5,5]*u.arcsec, [-5,5]*u.arcsec)
        AIAMap([[ 341.3125,  266.5   ,  329.375 ,  330.5625,  298.875 ],
        [ 347.1875,  273.4375,  247.4375,  303.5   ,  305.3125],
        [ 322.8125,  302.3125,  298.125 ,  299.    ,  261.5   ],
        [ 334.875 ,  289.75  ,  269.25  ,  256.375 ,  242.3125],
        [ 273.125 ,  241.75  ,  248.8125,  263.0625,  249.0625]])

        >>> aia.submap([0,5]*u.pixel, [0,5]*u.pixel)
        AIAMap([[ 0.3125, -0.0625, -0.125 ,  0.    , -0.375 ],
        [ 1.    ,  0.1875, -0.8125,  0.125 ,  0.3125],
        [-1.1875,  0.375 , -0.5   ,  0.25  , -0.4375],
        [-0.6875, -0.3125,  0.8125,  0.0625,  0.1875],
        [-0.875 ,  0.25  ,  0.1875,  0.    , -0.6875]])
        """

        # Do manual Quantity input validation to allow for two unit options
        if ((isinstance(range_a, u.Quantity) and isinstance(range_b, u.Quantity)) or
            (hasattr(range_a, 'unit') and hasattr(range_b, 'unit'))):

            if (range_a.unit.is_equivalent(self.units.x) and
                range_b.unit.is_equivalent(self.units.y)):
                units = 'data'
            elif range_a.unit.is_equivalent(u.pixel) and range_b.unit.is_equivalent(u.pixel):
                units = 'pixels'
            else:
                raise u.UnitsError("range_a and range_b but be "
                                   "in units convertable to {} or {}".format(self.units['x'],
                                                                             u.pixel))
        else:
            raise TypeError("Arguments range_a and range_b to function submap "
                            "have an invalid unit attribute "
                            "You may want to pass in an astropy Quantity instead.")

        if units is "data":
            # Check edges (e.g. [:512,..] or [:,...])
            if range_a[0] is None:
                range_a[0] = self.xrange[0]
            if range_a[1] is None:
                range_a[1] = self.xrange[1]
            if range_b[0] is None:
                range_b[0] = self.yrange[0]
            if range_b[1] is None:
                range_b[1] = self.yrange[1]

            x1, y1 = np.ceil(u.Quantity(self.data_to_pixel(range_a[0], range_b[0]))).value
            x2, y2 = np.floor(u.Quantity(self.data_to_pixel(range_a[1], range_b[1])) + 1*u.pix).value

            x_pixels = [x1, x2]
            y_pixels = [y1, y2]

        elif units is "pixels":
            # Check edges
            if range_a[0] is None:
                range_a[0] = 0
            if range_a[1] is None:
                range_a[1] = self.data.shape[1]
            if range_b[0] is None:
                range_b[0] = 0
            if range_b[1] is None:
                range_b[1] = self.data.shape[0]

            x_pixels = range_a.value
            y_pixels = range_b.value
        else:
            raise ValueError(
                "Invalid unit. Must be one of 'data' or 'pixels'")

        x_pixels = np.array(x_pixels)
        y_pixels = np.array(y_pixels)
        # Clip pixel values to max of array, prevents negative
        # indexing
        x_pixels[np.less(x_pixels, 0)] = 0
        x_pixels[np.greater(x_pixels, self.data.shape[1])] = self.data.shape[1]

        y_pixels[np.less(y_pixels, 0)] = 0
        y_pixels[np.greater(y_pixels, self.data.shape[0])] = self.data.shape[0]

        # Get ndarray representation of submap
        xslice = slice(x_pixels[0], x_pixels[1])
        yslice = slice(y_pixels[0], y_pixels[1])
        new_data = self.data[yslice, xslice].copy()

        # Make a copy of the header with updated centering information
        new_map = deepcopy(self)
        new_map.meta['crpix1'] = self.reference_pixel.x.value - x_pixels[0]
        new_map.meta['crpix2'] = self.reference_pixel.y.value - y_pixels[0]
        new_map.meta['naxis1'] = new_data.shape[1]
        new_map.meta['naxis2'] = new_data.shape[0]

        # Create new map instance
        new_map.data = new_data
        return new_map

    @u.quantity_input(dimensions=u.pixel)
    def superpixel(self, dimensions, method='sum'):
        """Returns a new map consisting of superpixels formed from the
        original data.  Useful for increasing signal to noise ratio in images.

        Parameters
        ----------
        dimensions : tuple
            One superpixel in the new map is equal to (dimension[0],
            dimension[1]) pixels of the original map
            Note: the first argument corresponds to the 'x' axis and the second
            argument corresponds to the 'y' axis.
        method : {'sum' | 'average'}
            What each superpixel represents compared to the original data
                * sum - add up the original data
                * average - average the sum over the number of original pixels

        Returns
        -------
        out : Map
            A new Map which has superpixels of the required size.

        References
        ----------
        | http://mail.scipy.org/pipermail/numpy-discussion/2010-July/051760.html
        """

        # Note: because the underlying ndarray is transposed in sense when
        #   compared to the Map, the ndarray is transposed, resampled, then
        #   transposed back
        # Note: "center" defaults to True in this function because data
        #   coordinates in a Map are at pixel centers

        # Make a copy of the original data and perform reshaping
        reshaped = reshape_image_to_4d_superpixel(self.data.copy(),
                                                  [dimensions.value[1], dimensions.value[0]])
        if method == 'sum':
            new_data = reshaped.sum(axis=3).sum(axis=1)
        elif method == 'average':
            new_data = ((reshaped.sum(axis=3).sum(axis=1)) /
                    np.float32(dimensions[0] * dimensions[1]))

        # Update image scale and number of pixels
        new_map = deepcopy(self)
        new_meta = new_map.meta

        new_nx = (self.dimensions[0] / dimensions[0]).value
        new_ny = (self.dimensions[1] / dimensions[1]).value

        # Update metadata
        new_meta['cdelt1'] = (dimensions[0] * self.scale.x).value
        new_meta['cdelt2'] = (dimensions[1] * self.scale.y).value
        if 'CD1_1' in new_meta:
            new_meta['CD1_1'] *= dimensions[0].value
            new_meta['CD2_1'] *= dimensions[0].value
            new_meta['CD1_2'] *= dimensions[1].value
            new_meta['CD2_2'] *= dimensions[1].value
        new_meta['crpix1'] = (new_nx + 1) / 2.
        new_meta['crpix2'] = (new_ny + 1) / 2.
        new_meta['crval1'] = self.center.x.value
        new_meta['crval2'] = self.center.y.value

        # Create new map instance
        new_map.data = new_data
        return new_map

# #### Visualization #### #

    @u.quantity_input(grid_spacing=u.deg)
    def draw_grid(self, axes=None, grid_spacing=15*u.deg, **kwargs):
        """Draws a grid over the surface of the Sun

        Parameters
        ----------
        axes: matplotlib.axes object or None
        Axes to plot limb on or None to use current axes.

        grid_spacing: float
            Spacing (in degrees) for longitude and latitude grid.

        Returns
        -------
        matplotlib.axes object

        Notes
        -----
        keyword arguments are passed onto matplotlib.pyplot.plot
        """

        if not axes:
            axes = wcsaxes_compat.gca_wcs(self.wcs)

        # Do not automatically rescale axes when plotting the overlay
        axes.set_autoscale_on(False)

        transform = wcsaxes_compat.get_world_transform(axes)

        XX, YY = np.meshgrid(np.arange(self.data.shape[0]),
                             np.arange(self.data.shape[1]))
        x, y = self.pixel_to_data(XX*u.pix, YY*u.pix)
        dsun = self.dsun

        b0 = self.heliographic_latitude.to(u.deg).value
        l0 = self.heliographic_longitude.to(u.deg).value
        units = self.units

        #Prep the plot kwargs
        plot_kw = {'color':'white',
                   'linestyle':'dotted',
                   'zorder':100,
                   'transform':transform}
        plot_kw.update(kwargs)

        hg_longitude_deg = np.linspace(-180, 180, num=361) + l0
        hg_latitude_deg = np.arange(-90, 90, grid_spacing.to(u.deg).value)

        # draw the latitude lines
        for lat in hg_latitude_deg:
            x, y = wcs.convert_hg_hpc(hg_longitude_deg, lat * np.ones(361),
                                      b0_deg=b0, l0_deg=l0, dsun_meters=dsun,
                                      angle_units=units.x, occultation=True)
            valid = np.logical_and(np.isfinite(x), np.isfinite(y))
            x = x[valid]
            y = y[valid]
            if wcsaxes_compat.is_wcsaxes(axes):
                x = (x*u.arcsec).to(u.deg).value
                y = (y*u.arcsec).to(u.deg).value
            axes.plot(x, y, **plot_kw)

        hg_longitude_deg = np.arange(-180, 180, grid_spacing.to(u.deg).value) + l0
        hg_latitude_deg = np.linspace(-90, 90, num=181)

        # draw the longitude lines
        for lon in hg_longitude_deg:
            x, y = wcs.convert_hg_hpc(lon * np.ones(181), hg_latitude_deg,
                                      b0_deg=b0, l0_deg=l0, dsun_meters=dsun,
                                      angle_units=units[0], occultation=True)
            valid = np.logical_and(np.isfinite(x), np.isfinite(y))
            x = x[valid]
            y = y[valid]
            if wcsaxes_compat.is_wcsaxes(axes):
                x = (x*u.arcsec).to(u.deg).value
                y = (y*u.arcsec).to(u.deg).value
            axes.plot(x, y, **plot_kw)

        # Turn autoscaling back on.
        axes.set_autoscale_on(True)
        return axes

    def draw_limb(self, axes=None, **kwargs):
        """Draws a circle representing the solar limb

            Parameters
            ----------
            axes: matplotlib.axes object or None
                Axes to plot limb on or None to use current axes.

            Returns
            -------
            matplotlib.axes object

            Notes
            -----
            keyword arguments are passed onto the Circle Patch, see:
            http://matplotlib.org/api/artist_api.html#matplotlib.patches.Patch
            http://matplotlib.org/api/artist_api.html#matplotlib.patches.Circle
        """

        if not axes:
            axes = wcsaxes_compat.gca_wcs(self.wcs)

        transform = wcsaxes_compat.get_world_transform(axes)
        if wcsaxes_compat.is_wcsaxes(axes):
            radius = self.rsun_obs.to(u.deg).value
        else:
            radius = self.rsun_obs.value
        c_kw = {'radius':radius,
                'fill':False,
                'color':'white',
                'zorder':100,
                'transform': transform
                }
        c_kw.update(kwargs)

        circ = patches.Circle([0, 0], **c_kw)
        axes.add_artist(circ)

        return axes

    @toggle_pylab
    def peek(self, draw_limb=False, draw_grid=False,
                   colorbar=True, basic_plot=False, **matplot_args):
        """Displays the map in a new figure

        Parameters
        ----------
        draw_limb : bool
            Whether the solar limb should be plotted.

        draw_grid : bool or `~astropy.units.Quantity`
            Whether solar meridians and parallels are plotted.
            If `~astropy.units.Quantity` then sets degree difference between
            parallels and meridians.
        gamma : float
            Gamma value to use for the color map
        colorbar : bool
            Whether to display a colorbar next to the plot
        basic_plot : bool
            If true, the data is plotted by itself at it's natural scale; no
            title, labels, or axes are shown.
        **matplot_args : dict
            Matplotlib Any additional imshow arguments that should be used
            when plotting the image.
        """

        # Create a figure and add title and axes
        figure = plt.figure(frameon=not basic_plot)

        # Basic plot
        if basic_plot:
            axes = plt.Axes(figure, [0., 0., 1., 1.])
            axes.set_axis_off()
            figure.add_axes(axes)
            matplot_args.update({'annotate':False})

        # Normal plot
        else:
            axes = wcsaxes_compat.gca_wcs(self.wcs)

        im = self.plot(axes=axes, **matplot_args)

        if colorbar and not basic_plot:
            figure.colorbar(im)

        if draw_limb:
            self.draw_limb(axes=axes)

        if isinstance(draw_grid, bool):
            if draw_grid:
                self.draw_grid(axes=axes)
        elif isinstance(draw_grid, (int, long, float)):
            self.draw_grid(axes=axes, grid_spacing=draw_grid)
        else:
            raise TypeError("draw_grid should be bool, int, long or float")

        figure.show()

    @toggle_pylab
    def plot(self, annotate=True, axes=None, **imshow_args):
        """ Plots the map object using matplotlib, in a method equivalent
        to plt.imshow() using nearest neighbour interpolation.

        Parameters
        ----------
        gamma : float
            Gamma value to use for the color map

        annotate : bool
            If true, the data is plotted at it's natural scale; with
            title and axis labels.

        axes: matplotlib.axes object or None
            If provided the image will be plotted on the given axes. Else the
            current matplotlib axes will be used.

        **imshow_args : dict
            Any additional imshow arguments that should be used
            when plotting the image.

        Examples
        --------
        #Simple Plot with color bar
        plt.figure()
        aiamap.plot()
        plt.colorbar()

        #Add a limb line and grid
        aia.plot()
        aia.draw_limb()
        aia.draw_grid()
        """

        #Get current axes
        if not axes:
            axes = wcsaxes_compat.gca_wcs(self.wcs)

        # Check that the image is properly oriented
        if (not wcsaxes_compat.is_wcsaxes(axes) and
            not np.array_equal(self.rotation_matrix, np.matrix(np.identity(2)))):
            warnings.warn("This map is not properly oriented. Plot axes may be incorrect",
                          Warning)

        # Normal plot
        if annotate:
            axes.set_title(self.plot_settings.get('title'))

            # x-axis label
            if self.coordinate_system.x == 'HG':
                xlabel = 'Longitude [{lon}]'.format(lon=self.units.x)
            else:
                xlabel = 'X-position [{xpos}]'.format(xpos=self.units.x)

            # y-axis label
            if self.coordinate_system.y == 'HG':
                ylabel = 'Latitude [{lat}]'.format(lat=self.units.y)
            else:
                ylabel = 'Y-position [{ypos}]'.format(ypos=self.units.y)

            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)


        cmap = deepcopy(self.plot_settings['cmap'])

        kwargs = self._mpl_imshow_kwargs(axes, cmap)
        kwargs.update(imshow_args)

        ret = axes.imshow(self.data, **kwargs)

        if wcsaxes_compat.is_wcsaxes(axes):
            wcsaxes_compat.default_wcs_grid(axes)

        #Set current image (makes colorbar work)
        plt.sci(ret)
        return ret

    def _mpl_imshow_kwargs(self, axes, cmap):
        """
        Return the keyword arguments for imshow to display this map
        """
        if wcsaxes_compat.is_wcsaxes(axes):
            kwargs = {'cmap': self.plot_settings['cmap'],
                      'origin': 'lower',
                      'norm': self.mpl_color_normalizer,
                      'interpolation': 'nearest'}
        else:
            # make imshow kwargs a dict
            kwargs = {'origin': 'lower',
                      'cmap': self.plot_settings['cmap'],
                      'norm': self.mpl_color_normalizer,
                      'extent': list(self.xrange.value) + list(self.yrange.value),
                      'interpolation': 'nearest'}

        return kwargs


    def _get_mpl_normalizer(self):
        """
        Returns a default mpl.colors.Normalize instance for plot scaling.

        Not yet implemented.
        """
        return None


class InvalidHeaderInformation(ValueError):
    """Exception to raise when an invalid header tag value is encountered for a
    FITS/JPEG 2000 file."""
    pass
