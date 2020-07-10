from __future__ import print_function,division
from numpy import *
from shamfi.shamfi_plotting import add_colourbar

from copy import deepcopy
import os
from astropy.modeling.models import Gaussian2D
from scipy.signal import fftconvolve
from shamfi.git_helper import get_gitdict, write_git_header


##convert between FWHM and std dev for the gaussian function
FWHM_factor = 2. * sqrt(2.*log(2.))

##converts between FWHM and std dev for the RTS
rts_factor = sqrt(pi**2 / (2.*log(2.)))



def subtract_gauss(ind,x,y,major,minor,pa,flux,ax1,ax2,ax3,fig,fits_data):
    """
    Takes a 2D CLEAN restored image array (data) and subtracts a Gaussian
    using the specified parameters. The subtracted Gaussian is convolved with
    the restoring beam kernel to ensure the correct Gaussian properties. Plots
    the subtracted gaussians with postage stamps before and after subtraction

    Parameters
    ----------
    ind : int
        index of the Gaussian being fit, used to correctly title the outputs when plotting
    x : int
        The central x coord of the Gaussian to subtract
    y : int
        The central y coord of the Gaussian to subtract
    major : float
        The major axis of the Gaussian to subtract (arcmin)
    minor : float
        The minor axis of the Gaussian to subtract (arcmin)
    pa : float
        The pa of the Gaussian to subtract (deg)
    flux : float
        The integrated flux density of the Gaussian to subract (Jy)
    ax1 : matplotlib.pyplot.figure.add_subplot instance
        The axis to plot the data before subtraction on
    ax2 : matplotlib.pyplot.figure.add_subplot instance
        The axis to plot the Gaussian to be subtracted on
    ax3 : matplotlib.pyplot.figure.add_subplot instance
        The axis to plot the data after subtraction on
    fig : matplotlib.pyplot.figure
        The figure to plot on
    fits_data : shamfi.read_FITS_image.FITSInformation instance
        A :class:`FITSInformation` class containing the image data

    Return
    ------
    data : 2D numpy array
        2D numpy image array with the Gaussian subtracted
    ra : float
        The RA of the the subtracted Gaussian (deg)
    dec : float
        The Dec of the the subtracted Gaussian (deg)

    """
    ##Setup the gaussian
    major *= (1/60.0)
    minor *= (1/60.0)
    pa *= (pi/180.0)

    # ra_reso = abs(float(header['CDELT1']))
    # dec_reso = float(header['CDELT2'])

    data = fits_data.data

    x_stddev = major / (FWHM_factor*fits_data.ra_reso)
    y_stddev = minor / (FWHM_factor*fits_data.dec_reso)

    gauss_func = Gaussian2D(amplitude=1.0, x_mean=x, y_mean=y, x_stddev=x_stddev, y_stddev=y_stddev,theta=pi/2.0 + pa)

    xrange = arange(fits_data.header['NAXIS1'])
    yrange = arange(fits_data.header['NAXIS2'])

    x_mesh, y_mesh = meshgrid(xrange,yrange)
    gauss_subtrac = gauss_func(x_mesh,y_mesh)

    ##Set up the restoring beam and convolve gaussian to subtract to mimic fitting convolved with a restoring beam
    # rest_bmaj = float(header['BMAJ'])
    # rest_bmin = float(header['BMIN'])
    # rest_pa = header['BPA'] * (pi/180.0)
    # rest_gauss_kern = create_restoring_kernel(rest_bmaj,rest_bmin,rest_pa,ra_reso,dec_reso)
    rest_gauss_kern = fits_data.create_restoring_kernel()

    ##Convolve with restoring beam
    gauss_subtrac = fftconvolve(gauss_subtrac, rest_gauss_kern, 'same')

    ##Get the convertsion from Jy/beam to Jy/pixel
    convert2pixel = fits_data.convert2pixel
    ##Scale the gaussian to subtract to match the desired integrated flux
    gauss_subtrac *= flux / (gauss_subtrac.sum()*convert2pixel)

    ##Define a plotting area about middle of gaussian
    half_width = 20
    low_y = int(round(y - half_width))
    high_y = int(round(y + half_width))
    low_x = int(round(x - half_width))
    high_x = int(round(x + half_width))

    data_plot = data[low_y:high_y,low_x:high_x]
    ##Set the same vmin and vmax for data and gauss for easy comparison
    vmin = data_plot.min()
    vmax = data_plot.max()

    ##Plot the data and the gaussian
    im1 = ax1.imshow(data_plot,origin='lower',vmin=vmin,vmax=vmax)
    im2 = ax2.imshow(gauss_subtrac[low_y:high_y,low_x:high_x],origin='lower',vmin=vmin,vmax=vmax)

    ##subtract the gaussian
    data -= gauss_subtrac
    ##Plot the data after subtraction
    im3 = ax3.imshow(data[low_y:high_y,low_x:high_x],origin='lower')

    ##Set some colourbars and get rid of tick labels
    for ax,im in zip([ax1,ax2,ax3],[im1,im2,im3]):
        add_colourbar(ax=ax,fig=fig,im=im)
        ax.set_xticks([])
        ax.set_yticks([])

    ##Add titles if at top of plot
    if ind == 0:
        ax1.set_title('Data before subtract')
        ax2.set_title('Convolved gauss to subtract')
        ax3.set_title('Data after subtract')

    ##Convert the pixel location into an ra / dec
    if fits_data.data_dims == 4:
        ra,dec,_,_ = fits_data.wcs.all_pix2world(x,y,0,0,0)
    elif fits_data.data_dims == 3:
        ra,dec,_,_ = fits_data.wcs.all_pix2world(x,y,0,0)
    elif fits_data.data_dims == 2:
        ra,dec,_,_ = fits_data.wcs.all_pix2world(x,y,0)

    return data,ra,dec
