from __future__ import print_function,division
from numpy import *
from numpy import abs as np_abs
import scipy.optimize as opt
from copy import deepcopy
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
# from astropy.io import fits
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.axes import Axes

# from astropy.wcs import WCS
# from sys import exit

# from scipy.signal import fftconvolve
# import os
# from astropy.modeling.models import Gaussian2D
# from progressbar import progressbar
# from subprocess import check_output
# import pkg_resources
# from shamfi.git_helper import get_gitdict, write_git_header
#

#
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

# (2*pi) / (sqrt(pi**2 / (2*log(2))))
#
# ##Use package manager to get hold of the basis functions
# basis_path = pkg_resources.resource_filename("shamfi", "image_shapelet_basis.npz")
#
# ##Load the basis functions
# image_shapelet_basis = load(basis_path)
# basis_matrix = image_shapelet_basis['basis_matrix']
# gauss_array = image_shapelet_basis['gauss_array']

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
    """
    Creates a model for a 2D Gaussian, by taking two 2D coordinate arrays in x,y.
    Returns a flattened array of the model to make fitting more straight forward

    Parameters
    ----------
    xy : list containing two 2D numpy arrays
        A list containing the x and y coordinates to calculate the gaussian at.
        x and y are separated into individual 2D arrays. xy = [x(2D), y(2D)]
    amplitude : float
        Amplitude to scale the Gaussian by
    xo : float
        Value of the central x pixel
    yo : float
        Value of the central y pixel
    sigma_x : float
        Sigma value for the x dimension
    sigma_y : float
        Sigma value for the y dimension
    theta : float
        Rotation angle (radians)

    Returns
    -------
    gaussian.flatten() : numpy array (floats)
        A 2D gaussian model, flattened into a 1D numpy array
    """

    x,y = xy

    xo = float(xo)
    yo = float(yo)
    a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
    b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
    c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
    gaussian = amplitude*exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return gaussian.flatten()


class ShapeletCoords():
    def __init__(self,fits_data=False):
        if fits_data:
            pass
        else:
            print('To initialise a ShapeletCoords Class you need to supply a Class as the fits_data argument')

        self.fits_data = fits_data
        self.edge_pad = self.fits_data.edge_pad

    def find_good_pixels(self,fit_box=False,exclude_box=False,ignore_negative=False):
        '''Uses the specified arguments to come up with an array of pixel indexes
        to fit'''
        ##If a box is specified, limit pixels to within that box
        if fit_box:
            pixel_inds_to_use = []
            low_x,high_x,low_y,high_y = array(fit_box.split(','),dtype=int)
            for y in range(low_y,high_y+1):
                for x in range(low_x,high_x+1):
                    pixel_inds_to_use.append(y*self.fits_data.len1 + x)

            self.pixel_inds_to_use = array(pixel_inds_to_use)
            print('Will fit box defined by low_x,high_x,low_y,high_y: ',low_x,high_x,low_y,high_y)

        else:
            ##If nothing declared, just use all the pixels
            if not exclude_box:
                self.pixel_inds_to_use = arange(len(self.fits_data.flat_data))

            ##Otherwise, use the defined boxes in --exclude_box to flag pixels to
            ##avoid
            else:
                try:
                    avoid_inds = []
                    for box in exclude_box:
                        low_x,high_x,low_y,high_y = array(box.split(','),dtype=int)
                        for y in range(low_y,high_y+1):
                            for x in range(low_x,high_x+1):
                                avoid_inds.append(y*self.fits_data.len1 + x)

                    pixel_inds_to_use = arange(len(flat_data))
                    self.pixel_inds_to_use = setxor1d(pixel_inds_to_use,avoid_inds)
                    print('Will fit avoiding boxes defined in --exclude_box')
                except:
                    self.pixel_inds_to_use = arange(len(flat_data))
                    print('Failed to convert --exclude_box into something \
                          sensible. Will fit using all pixels in image')

        ##If we are to ignore negative pixels, find all negative pixels in the
        ##good list of pixels. Keep as a separate pixel mask as when generating
        ##basis functions we convolve with the restoring beam, and if random
        ##pixels are missing this convolution become innaccurate
        if ignore_negative:
            print('Ignoring negative pixels in fit')
            fluxes = self.fits_data.flat_data[self.pixel_inds_to_use]
            self.negative_pix_mask = where(fluxes >= 0.0)[0]
        else:
            print('Will include negative pixels in fit')
            self.negative_pix_mask = arange(len(self.pixel_inds_to_use))

        self._find_image_centre_celestial()


    def _find_image_centre_celestial(self):
        '''Find the flux-weighted central position of an image'''
        power = 4
        ra_cent = sum(self.fits_data.flat_data[self.pixel_inds_to_use]**power*self.fits_data.ras[self.pixel_inds_to_use])
        ra_cent /= sum(self.fits_data.flat_data[self.pixel_inds_to_use]**power)

        dec_cent = sum(self.fits_data.flat_data[self.pixel_inds_to_use]**power*self.fits_data.decs[self.pixel_inds_to_use])
        dec_cent /= sum(self.fits_data.flat_data[self.pixel_inds_to_use]**power)

        resolution = abs(self.fits_data.ras[1] - self.fits_data.ras[0])
        ##Find the difference between the gridded ra coords and the desired ra_cent
        ra_offs = np_abs(self.fits_data.ras - ra_cent)
        ##Find out where in the gridded ra coords the current ra_cent lives;
        ##This is a boolean array of length len(ra_offs)
        ra_true = ra_offs < resolution/2.0
        ##Find the index so we can access the correct entry in the container
        ra_ind = where(ra_true == True)[0]

        ##Use the numpy abs because it's faster (np_abs)
        dec_offs = np_abs(self.fits_data.decs - dec_cent)
        dec_true = dec_offs < resolution/2
        dec_ind = where(dec_true == True)[0]

        ##If ra_ind,dec_ind coord sits directly between two grid points,
        ##just choose the first one
        if len(ra_ind) == 0:
            ra_true = ra_offs <= resolution/2
            ra_ind = where(ra_true == True)[0]
        if len(dec_ind) == 0:
            dec_true = dec_offs <= resolution/2
            dec_ind = where(dec_true == True)[0]
        ra_ind,dec_ind = ra_ind[0],dec_ind[0]

        ##Central dec index has multiple rows as it is from flattended coords,
        ##remove that here
        dec_ind = floor(dec_ind / self.fits_data.len1)
        print('Centre of flux pixel in image found as x,y',ra_ind,dec_ind)

        ra_mesh = deepcopy(self.fits_data.ras)
        ra_mesh.shape = self.fits_data.data.shape

        dec_mesh = deepcopy(self.fits_data.decs)
        dec_mesh.shape = self.fits_data.data.shape

        ra_range = ra_mesh[0,:]
        dec_range = dec_mesh[:,0]

        self.ra_cent_ind = ra_ind
        self.dec_cent_ind = dec_ind
        self.ra_mesh = ra_mesh
        self.dec_mesh = dec_mesh
        self.ra_range = ra_range
        self.dec_range = dec_range

    def fit_gauss_and_centre_coords(self,b1_max=False,b2_max=False):
        '''Try and fit a Gaussian to the image, using the flux weighted central pixel
        location, and maximum b1 and b2 values, as an initial parameter estimate'''
        ##Fit a gaussian to the data to find pa
        ##guess is: amp, xo, yo, sigma_x, sigma_y, pa
        initial_guess = (self.fits_data.data.max(),self.ra_range[int(self.ra_cent_ind)],self.dec_range[int(self.dec_cent_ind)],
                        (b1_max / 60.0)*D2R,(b2_max / 60.0)*D2R,0)

        popt, pcov = opt.curve_fit(twoD_Gaussian, (self.ra_mesh, self.dec_mesh), self.fits_data.flat_data, p0=initial_guess)
        #
        ##Check pa is between 0 <= pa < 2pi
        pa = popt[5]
        if pa < 0:
            pa += 2*pi
        ##Necessary to move from my gaussian which has theta = 0 at x = 0 and
        ##actual PA which is east from north
        pa += pi / 2.0
        if pa > 2*pi:
            pa -= 2*pi

        self.pa = pa
        self.popt = popt

        x0 = popt[1]
        y0 = popt[2]
        #
        # ##Set central ra, dec pixel to zero in prep for scaling to x,y coords
        self.set_central_pixel_to_zero(x0,y0)

    def set_central_pixel_to_zero(self,x0,y0):
        '''Using the central position found when fitting a gaussian (popt) takes
        the ra,dec coord system and sets x0,y0=0,0'''

        ra_offs = np_abs(self.ra_range - x0)
        dec_offs = np_abs(self.dec_range - y0)

        ra_ind = where(ra_offs < abs(self.ra_range[1] - self.ra_range[0])/2.0)[0][0]
        dec_ind = where(dec_offs < abs(self.dec_range[1] - self.dec_range[0])/2.0)[0][0]

        ra_cent_off = self.ra_range[ra_ind]
        dec_cent_off = self.dec_range[dec_ind]

        if self.fits_data.data_dims == 2:
            self.ra_cent,self.dec_cent = self.fits_data.wcs.wcs_pix2world(ra_ind-self.edge_pad,dec_ind-self.edge_pad,0)
        elif self.fits_data.data_dims == 3:
            self.ra_cent,self.dec_cent = self.fits_data.wcs.wcs_pix2world(ra_ind-self.edge_pad,dec_ind-self.edge_pad,0,0)
        elif self.fits_data.data_dims == 4:
            self.ra_cent,self.dec_cent,_,_ = self.fits_data.wcs.wcs_pix2world(ra_ind-self.edge_pad,dec_ind-self.edge_pad,0,0,0)

        self.ras = self.fits_data.ras - ra_cent_off
        self.decs = self.fits_data.decs - dec_cent_off



    def radec2xy(self,b1,b2,crop=False):
        '''Transforms the RA/DEC coords system into the shapelet x/y system
        for given b1,b2 parameters. Also applies the pixel_inds_to_use cut
        to only return the pixels to be used in the shapelet fit'''

        ##If we want to ignore bad pixels, do the crop here
        if crop:
            ##RA increases in opposite direction to x
            x = -self.ras[self.pixel_inds_to_use]
            y = self.decs[self.pixel_inds_to_use]

        else:
            ##RA increases in opposite direction to x
            x = -self.ras
            y = self.decs

        ##Rotation is east from north, (positive RA is negative x)
        angle = -self.pa

        yrot = x*cos(angle) + -y*sin(angle)
        xrot = x*sin(angle) + y*cos(angle)

        ##Apply conversion into stdev from FWHM and beta params
        xrot *= FWHM_factor / b1
        yrot *= FWHM_factor / b2

        return xrot,yrot
