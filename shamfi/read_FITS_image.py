from __future__ import print_function,division
from numpy import *
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D
from sys import exit

##Convert degress to radians
D2R = pi/180.
##Convert radians to degrees
R2D = 180./pi

##convert between FWHM and std dev for the gaussian function
FWHM_factor = 2. * sqrt(2.*log(2.))

class FITSInformation():
    """
    This class reads in data and metadata from a FITS image file, and stores
    it for use in generating relevant coord systems and data when fitting
    shapelets. Expects an CLEANed image, testing against WSClean outputs.
    See attributes for futher functionality, and variables for what can be
    accessed.

    :param str fitsfile: name of a FITS image file to gather relevant data and metadata from
    :ivar str fitsfile: the given fitsfile parameter
    :ivar header: an astropy header instance of the FITS file
    :ivar array data: a 2D numpy array of the image data
    :ivar bool read_data: True if successful in reading data

    :ivar float ra_reso: RA resolution (deg)
    :ivar float dec_reso: Dec resolution (deg)
    :ivar float pix_area_rad: pixel area (ster radian)

    :ivar int len1: number of pixels in NAXIS1
    :ivar int len2: number of pixels in NAXIS2

    :ivar wcs: an astropy WCS instance based on self.header
    :ivar array flat_data: self.data.flatten()

    :ivar float bmaj: restoring beam major axis - if keyword BMAJ not in header, set to None
    :ivar float bmin: restoring beam minor axis - if keyword BMIN not in header, set to None

    :ivar float solid_beam: solid beam angle: (pi*self.bmaj*self.bmin) / (4*log(2))

    :ivar float convert2pixel: conversion factor need to convert Jy/beam to Jy/pixel

    :ivar bool got_convert2pixel: True if all metadata to create self.convert2pixel was available


    """
    def __init__(self,fitsfile):
        try:
            with fits.open(fitsfile) as hdu:
                self.header = hdu[0].header
                self.data = hdu[0].data
                self.read_data = True

                self._get_basic_info()
                self._get_convert2pixel()
                self._get_frequency()

                self.fitsfile = fitsfile

        except:
            print('Failed to open and read this FITS file: %s' %fitsfile)
            self.read_data = False

    def _get_basic_info(self):
        '''Use the FITS header and data to find some key information'''

        self.data_dims = len(self.data.shape)

        if self.data_dims == 2:
            self.data = self.data
        elif self.data_dims == 3:
            self.data = self.data[0,:,:]
        elif self.data_dims == 4:
            self.data = self.data[0,0,:,:]

        try:
            self.ra_reso = float(self.header['CDELT1'])
            self.dec_reso = float(self.header['CDELT2'])
        except:
            self.ra_reso = float(self.header['CD1_1'])
            self.dec_reso = float(self.header['CD2_2'])

        self.pix_area_rad = abs(self.ra_reso*self.dec_reso*D2R**2)

        self.len1 = int(self.header['NAXIS1'])
        self.len2 = int(self.header['NAXIS2'])

        self.naxis1 = int(self.header['NAXIS1'])
        self.naxis2 = int(self.header['NAXIS2'])

        self.wcs = WCS(self.header)
        self.flat_data = self.data.flatten()


    def _get_convert2pixel(self):
        '''Use a FITS header and gets required info to calculate a conversion
        from Jy/beam to Jy/pixel'''

        try:
            # TODO Currently this will only work if BMAJ,BMIN,ra_reso,dec_reso are
            #all in the same units
            self.bmaj = float(self.header['BMAJ'])
            self.bmin = float(self.header['BMIN'])

            self.solid_beam = (pi*self.bmaj*self.bmin) / (4*log(2))

            self.solid_pixel = abs(self.ra_reso*self.dec_reso)
            self.convert2pixel = self.solid_pixel/self.solid_beam

            self.got_convert2pixel = True

        except:
            self.bmaj = None
            self.bmin = None
            self.solid_beam = None
            self.solid_pixel = None
            self.convert2pixel = None
            self.got_convert2pixel = False
    #
    def _get_frequency(self):
        '''Attempts to get frequency of the image from the FITS header'''

        self.freq = None
        self.found_freq = False
        try:
            self.freq = float(self.header['FREQ'])
            self.found_freq = True
        except:
            ctypes = self.header['CTYPE*']
            for ctype in ctypes:
                if self.header[ctype] == 'FREQ':
                    self.freq = float(self.header['CRVAL%d' %(int(ctype[-1]))])
                    self.found_freq = True


    def get_radec_edgepad(self,edge_pad=False):
        """
        Use FITS information to form values of RA, DEC for all pixels in this image
        If specified, edge pad the data with zeros, and add extra RA and DEC
        range accordingly.

        :param int edge_pad: If True, edge pad the data with the specified number of pixels
        :ivar array ras: RA values of all pixels in image, flattened into a 1D array (radians)
        :ivar array decs: DEC values of all pixels in image, flattened into a 1D array (radians)
        """
        ##TODO - this is fine for small images, but for large images projection
        ## I think we should do this using WCS

        if edge_pad:

            self.len1 += edge_pad*2
            self.len2 += edge_pad*2

            ras = (arange(self.len1) - (int(self.header['CRPIX1'])+edge_pad))*self.ra_reso
            decs = (arange(self.len2) - (int(self.header['CRPIX2'])+edge_pad))*self.dec_reso
            pad_image = zeros((self.len1,self.len2))
            pad_image[edge_pad:self.data.shape[0]+edge_pad,edge_pad:self.data.shape[1]+edge_pad] = self.data
            self.data = pad_image

        else:
            ras = (arange(self.len1) - int(self.header['CRPIX1']))*self.ra_reso
            decs = (arange(self.len2) - int(self.header['CRPIX2']))*self.dec_reso

        ##Get the ra,dec range for all pixels
        ras_mesh,decs_mesh = meshgrid(ras,decs)
        ras,decs = ras_mesh.flatten(),decs_mesh.flatten()

        self.ras = ras * D2R
        self.decs = decs * D2R
        self.edge_pad = edge_pad
        self.flat_data = self.data.flatten()


    def covert_to_jansky_per_pix(self):
        '''Coverts data in Jy/beam to Jy/pixel as stored in self.data and self.flat_data'''
        self.data *= self.convert2pixel
        self.flat_data *= self.convert2pixel


    def create_restoring_kernel(self):
        """
        Use the FITS header metadata to create 2D restoring Gaussian beam kernel
        with the height and width of the kernel set to 8 times larger than BMAJ

        :returns: 2D array of the restoring beam at the same resolution of the image. Also stored as self.rest_gauss_kern
        :rtype: array
        """

        x_stddev = self.bmaj / (FWHM_factor*abs(self.ra_reso))
        y_stddev = self.bmin / (FWHM_factor*self.dec_reso)

        try:
            bpa = self.header['BPA'] * D2R
        except:
            print('Could not find beam PA from FITS file. Setting to zero for the restoring kernel')
            bpa = 0.0

        rest_gauss_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev, y_stddev=y_stddev,theta=pi/2 + bpa)

        ##Sample restore beam to 8 times larger than bmaj in pixels
        rest_samp_max = int(ceil(self.bmaj / abs(self.ra_reso)))

        xrange = arange(-rest_samp_max,rest_samp_max + 1)
        yrange = arange(-rest_samp_max,rest_samp_max + 1)

        x_mesh, y_mesh = meshgrid(xrange,yrange)
        rest_gauss_kern = rest_gauss_func(x_mesh,y_mesh)
        rest_gauss_kern /= rest_gauss_kern.sum()

        self.rest_gauss_kern = rest_gauss_kern

        return rest_gauss_kern
