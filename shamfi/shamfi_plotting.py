from __future__ import print_function,division
from numpy import *
import matplotlib
##useful when using a super cluster to specify Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from copy import deepcopy

from shamfi.shapelet_coords import twoD_Gaussian
from shamfi import shapelets



# from scipy.special import factorial,eval_hermite
# from astropy.io import fits

#
# from numpy import abs as np_abs
# from astropy.wcs import WCS
# from sys import exit
# import scipy.optimize as opt
#
# from scipy.signal import fftconvolve
# import os
# from astropy.modeling.models import Gaussian2D
# from progressbar import progressbar
# from subprocess import check_output
# import pkg_resources
# from shamfi.git_helper import get_gitdict, write_git_header
#
##Convert degress to radians
D2R = pi/180.
##Convert radians to degrees
R2D = 180./pi
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
#
# ##Use package manager to get hold of the basis functions
# basis_path = pkg_resources.resource_filename("shamfi", "image_shapelet_basis.npz")
#
# ##Load the basis functions
# image_shapelet_basis = load(basis_path)
# basis_matrix = image_shapelet_basis['basis_matrix']
# gauss_array = image_shapelet_basis['gauss_array']



def add_colourbar(fig=None,ax=None,im=None,label=False,top=False):
    """
    Adds a colourbar (colorbar, fine) in a nice way to a subplot

    Parameters
    ----------
    fig : matplotlib.pyplot.figure instance
        The figure that the plot lives on
    ax : figure.add_subplot instance
        The axis to append a colorbar to
    im : ax.imshow output
        The output of imshow to base the colourbar on
    label : string
        Optional - add a label to the colorbar
    top : Bool
        Optional - put the colorbar above the axis instead of to the right
    """

    divider = make_axes_locatable(ax)
    if top == True:
        cax = divider.append_axes("top", size="5%", pad=0.05,axes_class=Axes)
        cbar = fig.colorbar(im, cax = cax,orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
    else:
        cax = divider.append_axes("right", size="5%", pad=0.05,axes_class=Axes)
        cbar = fig.colorbar(im, cax = cax)
    if label:
        cbar.set_label(label)

def plot_grid_search(shapelet_fitter,save_tag):
    '''Plot a matrix of residuals found when fitting for b1,b2'''
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(shapelet_fitter.residuals_array)

    ax.set_xticks(arange(len(shapelet_fitter.b1_grid)))
    ax.set_yticks(arange(len(shapelet_fitter.b2_grid)))

    labelsy = ['%.2f' %b for b in (shapelet_fitter.b1_grid/D2R)*60.0]
    labelsx = ['%.2f' %b for b in (shapelet_fitter.b2_grid/D2R)*60.0]

    ax.set_yticklabels(labelsy)
    ax.set_xticklabels(labelsx)

    ax.set_xlabel('b2 (arcmins)')
    ax.set_ylabel('b1 (arcmins)')

    add_colourbar(fig=fig,im=im,ax=ax)

    ax.contour(shapelet_fitter.residuals_array,colors='w',alpha=0.4)

    fig.savefig('grid-fit_matrix_%s_nmax%03d_p%03d.png' %(save_tag,shapelet_fitter.nmax,shapelet_fitter.model_percentage), bbox_inches='tight')
    plt.close()


def plot_gaussian_fit(shpcoord,save_tag):
    '''Plot a contour of an intial gaussian fit over an image, using the results
    of a ShapeletCoords Class'''
    mask = twoD_Gaussian((shpcoord.ra_mesh, shpcoord.dec_mesh), *shpcoord.popt)
    mask.shape = shpcoord.ra_mesh.shape

    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)

    im1 = ax1.imshow(shpcoord.fits_data.data,origin='lower')
    ax1.contour(mask,colors='r',alpha=0.3)

    add_colourbar(ax=ax1,im=im1,fig=fig)

    fig.savefig('pa_fit_%s.png' %save_tag ,bbox_inches='tight')

def do_subplot(fig,ax,data,label,vmin,vmax):
    if vmin:
        ax.imshow(data,origin='lower',vmin=vmin,vmax=vmax)
    else:
        im = ax.imshow(data,origin='lower')
    add_colourbar(fig=fig,im=im,ax=ax)
    ax.set_title(label)

def make_masked_image(flat_data,shapelet_fitter):
    '''Create a 2D array for plotting purposes, where all the pixels that were
    originally masked in the fit are set to NaN so they don't show during
    imshow'''
    ##Array of just nans of the correct dimension
    masked_data = ones(shapelet_fitter.fits_data.data.shape)*nan

    ##Need to flatten to apply the masks correctly
    masked_data = masked_data.flatten()

    ##Make a mask of both the selected pixels, and the negative pixels if they were masked
    inds_to_use = shapelet_fitter.shpcoord.pixel_inds_to_use[shapelet_fitter.shpcoord.negative_pix_mask]

    ##Set the correct indexes to the data to be plotted
    masked_data[inds_to_use] = flat_data

    ##Reshape and send on it's way
    masked_data.shape = shapelet_fitter.fits_data.data.shape

    return masked_data

def plot_full_shamfi_fit(shapelet_fitter, save_tag, plot_edge_pad=False):
    fig = plt.figure(figsize=(10,8))

    ##If plotting edge pad, need an extra axis
    if plot_edge_pad:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
    else:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
    #
    # if args.plot_lims:
    #     vmin,vmax = map(float,args.plot_lims.split(','))
    #
    # else:
    vmin,vmax = False, False

    masked_data = make_masked_image(shapelet_fitter.data_to_fit, shapelet_fitter)
    masked_fit = make_masked_image(shapelet_fitter.fit_data, shapelet_fitter)

    do_subplot(fig,ax1,masked_data,'Data',vmin,vmax)
    do_subplot(fig,ax2,masked_fit,'Fit (convolved with\nrestoring beam)',vmin,vmax)
    do_subplot(fig,ax3,masked_data - masked_fit,'Data - Fit',vmin,vmax)
    #
    if plot_edge_pad:

        print('Generating model for edge padded image')

        if shapelet_fitter.fits_data.edge_pad:
            b1 = shapelet_fitter.best_b1
            b2 = shapelet_fitter.best_b2
            n1s = shapelet_fitter.fit_n1s
            n2s = shapelet_fitter.fit_n2s
            nmax = shapelet_fitter.nmax
            convolve_kern = shapelet_fitter.convolve_kern
            shape = shapelet_fitter.fits_data.data.shape

            xrot,yrot = shapelet_fitter.shpcoord.radec2xy(b1, b2, crop=False)

            _, _, A_shape_basis_edge = shapelets.gen_A_shape_matrix(n1s=n1s,n2s=n2s,xrot=xrot,yrot=yrot,
                                                          nmax=nmax,b1=b1,b2=b2,
                                                          convolve_kern=convolve_kern,shape=shape)
        # else:
        #     edge_pad = int(fit_data.shape[0] / 5)
        #     data,flat_data,ras,decs,convert2pixel,ra_reso,dec_reso,freq,len1,len2,wcs,dims,rest_bmaj,rest_bmin,rest_pa = get_fits_info(args.fits_file,edge_pad=edge_pad,freq=args.freq)
        #     ra_ind,dec_ind,ra_mesh,dec_mesh,ra_range,dec_range = find_image_centre_celestial(ras=ras,decs=decs,flat_data=flat_data,pixel_inds_to_use=pixel_inds_to_use,data=data)
        #     ra_cent, dec_cent, ras, decs = set_central_pixel_to_zero(popt,ras,decs,ra_range,dec_range,args,edge_pad,dims,wcs)
        #
        #     xrot,yrot = radec2xy(ras,decs,pa,b1,b2)

        fitted_coeffs = deepcopy(shapelet_fitter.fitted_coeffs)
        fitted_coeffs.shape = (len(fitted_coeffs),1)
        fit_data_edge = matmul(A_shape_basis_edge,fitted_coeffs)

        fit_data_edge.shape = shape

        do_subplot(fig,ax4,fit_data_edge,'Edge padded fit \n(convolved with restoring beam)',vmin,vmax)

    fig.tight_layout()
    fig.savefig('shamfi_%s_nmax%03d_p%03d.png' %(save_tag,shapelet_fitter.nmax,shapelet_fitter.model_percentage), bbox_inches='tight')
    plt.close()

def plot_compressed_fits(args, compressed_images, flat_data, data_shape, pixel_inds_to_use,
                  save_tag, compress_values, nmax):
    '''Takes a list of compressed_images containg 2D arrays, and plots the
    compression results after a SHAMFI fit'''

    fig = plt.figure(figsize=(6,2*len(compressed_images)))
    if args.plot_lims:
        vmin,vmax = map(float,args.plot_lims.split(','))
    else:
        vmin,vmax = False, False

    bad_inds = setdiff1d(arange(len(flat_data)),pixel_inds_to_use)
    flat_data[bad_inds] = nan
    flat_data.shape = data_shape

    for plot_ind in arange(len(compressed_images)):

        compressed_image = compressed_images[plot_ind]
        compressed_image.shape = data_shape

        ax_img = fig.add_subplot(len(compressed_images),2,plot_ind*2+1)
        ax_diff = fig.add_subplot(len(compressed_images),2,plot_ind*2+2)

        do_subplot(fig,ax_img,compressed_image,'Fit %03.1f%%' %(compress_values[plot_ind]),vmin,vmax)
        do_subplot(fig,ax_diff,flat_data - compressed_image,'Fit %03.1f%%' %(compress_values[plot_ind]),vmin,vmax)

    fig.tight_layout()
    fig.savefig('shamfi_%s_nmax%d_compressed_fits.png' %(save_tag,nmax), bbox_inches='tight')
    plt.close()

    return flat_data
