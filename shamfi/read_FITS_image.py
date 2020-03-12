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




# import matplotlib
# ##useful when using a super cluster to specify Agg
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from scipy.special import factorial,eval_hermite

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.axes import Axes
# from numpy import abs as np_abs

# from sys import exit
# import scipy.optimize as opt
# from copy import deepcopy
# from scipy.signal import fftconvolve
# import os
# from astropy.modeling.models import Gaussian2D
# from progressbar import progressbar
# from subprocess import check_output
# import pkg_resources
# from shamfi.git_helper import get_gitdict, write_git_header

# ##Max x value of stored basis functions
# xmax = 250
# ##Number of samples in stored basis functions
# n_x = 20001
# ##More basis function values
# x_cent = int(floor(n_x / 2))
# xrange = linspace(-xmax,xmax,n_x)
# xres = xrange[1] - xrange[0]
#
# ##convert between FWHM and std dev for the gaussian function
# factor = 2. * sqrt(2.*log(2.))
#
# ##converts between FWHM and std dev for the RTS
# rts_factor = sqrt(pi**2 / (2.*log(2.)))
#
# ##Use package manager to get hold of the basis functions
# basis_path = pkg_resources.resource_filename("shamfi", "image_shapelet_basis.npz")
#
# ##Load the basis functions
# image_shapelet_basis = load(basis_path)
# basis_matrix = image_shapelet_basis['basis_matrix']
# gauss_array = image_shapelet_basis['gauss_array']



class FITSInformation():
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
        '''Use FITS information to form a values for all of RA, DEC values for this image
        If specified, edge pad the data with zeros, and add extra RA and DEC
        range accordingly. Returns RA/DEC in radians'''

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
        '''Apply the conversion from Jy/beam to Jy/pixel on the data'''
        self.data *= self.convert2pixel
        self.flat_data *= self.convert2pixel


    def create_restoring_kernel(self):
        '''Use the FITS header to create a restoring Gaussian beam kernel'''
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
