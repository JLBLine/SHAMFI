from __future__ import print_function,division
from numpy import *
import matplotlib
##useful when using a super cluster to specify Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from copy import deepcopy

from shamfi.shapelet_coords import twoD_Gaussian, ShapeletCoords
from shamfi import shapelets
from shamfi.read_FITS_image import FITSInformation

##Convert degress to radians
D2R = pi/180.
##Convert radians to degrees
R2D = 180./pi


#
def add_colourbar(fig=None,ax=None,im=None,label=False,top=False):
    """
    Adds a colourbar (colorbar, fine) in a nice way to a subplot

    Parameters
    ----------
    fig : matplotlib.pyplot.figure instance
        The figure that the plot lives on
    ax : matplotlib.pyplot.figure.add_subplot instance
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
    """
    Plot a matrix of the image based residuals as a function of b1, b2

    Parameters
    ----------
    shapelet_fitter : shamfi.shapelets.FitShapelets instance
        The :class:`FitShapelets` used to run the shapelet fitting
    save_tag : string
        A tag to add into the file name to save the plot to
    """
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
    """Plot a contour of an intial gaussian fit over an image, using the results
    of a :class:`ShapeletCoords`

    Parameters
    ----------
    shpcoord : shamfi.shapelet_coords.ShapeletCoords instance
        The :class:`FitShapelets` used to run the shapelet fitting
    save_tag : string
        A tag to add into the file name to save the plot to
    """
    mask = twoD_Gaussian((shpcoord.ra_mesh, shpcoord.dec_mesh), *shpcoord.popt)
    mask.shape = shpcoord.ra_mesh.shape

    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)

    im1 = ax1.imshow(shpcoord.fits_data.data,origin='lower')
    ax1.contour(mask,colors='r',alpha=0.3)

    add_colourbar(ax=ax1,im=im1,fig=fig)

    fig.savefig('pa_fit_%s.png' %save_tag ,bbox_inches='tight')

def do_subplot(fig,ax,data,label,vmin,vmax):
    """
    Plots a 2D numpy array (data) with a colorbar. Optionally can set vlim and
    vmin to control the colour scale

    Parameters
    ----------
    fig : matplotlib.pyplot.figure instance
        The figure that the plot lives on
    ax : matplotlib.pyplot.figure.add_subplot instance
        The axis to plot on
    data : 2D numpy array
        The data to plot
    label : string
        The title for the subplot
    vmin : float
        Optional - lower value for the colour scale, passed to imshow
    vmax : float
        Optional - upper value for the colour scale, passed to imshow
    """
    if vmin:
        ax.imshow(data,origin='lower',vmin=vmin,vmax=vmax)
    else:
        im = ax.imshow(data,origin='lower')
    add_colourbar(fig=fig,im=im,ax=ax)
    ax.set_title(label)

def make_masked_image(flat_data,shapelet_fitter):
    """
    Create a 2D array for plotting purposes, where all the pixels that were
    originally masked in the fit are set to NaN so they don't show during
    imshow

    Parameters
    ----------
    flat_data : array
        numpy array of data to mask
    shapelet_fitter : shamfi.shapelets.FitShapelets instance
        The :class:`FitShapelets` used to run the shapelet fitting
    Returns
    -------
    masked_data: 2D array
        A 2D array in the original shape of the image to be fitted, with the
        cuts that were applied during fitted set to NaNs for plotting

    """
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
    """
    Take a :class:`FitShapelets` class that has been run, and plot the results.
    Plots the data with top left, fit top right, and residuals bottom left.
    Optionally, plot an edge padded version of the final fit bottom right - useful
    to check that unconstrained areas outside of the fitting region haven't ended up
    with crazy modelled flux.

    Parameters
    ----------
    shapelet_fitter : shamfi.shapelets.FitShapelets instance
        The :class:`FitShapelets` used to run the shapelet fitting
    save_tag : string
        A tag to add into the file name to save the plot to
    plot_edge_pad : bool
        If True, plot a version of the model using edge padded coords, to check
        for run away modelling outside the fitting area

    """

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

    vmin,vmax = False, False

    masked_data = make_masked_image(shapelet_fitter.data_to_fit, shapelet_fitter)
    masked_fit = make_masked_image(shapelet_fitter.fit_data, shapelet_fitter)

    do_subplot(fig,ax1,masked_data,'Data',vmin,vmax)
    do_subplot(fig,ax2,masked_fit,'Fit (convolved with\nrestoring beam)',vmin,vmax)
    do_subplot(fig,ax3,masked_data - masked_fit,'Data - Fit',vmin,vmax)
    #
    if plot_edge_pad:

        print('Generating model for edge padded image')

        ## Gather fitting results
        b1 = shapelet_fitter.best_b1
        b2 = shapelet_fitter.best_b2
        n1s = shapelet_fitter.fit_n1s
        n2s = shapelet_fitter.fit_n2s
        nmax = shapelet_fitter.nmax
        convolve_kern = shapelet_fitter.convolve_kern
        b1_max = max(shapelet_fitter.b1_grid)
        b2_max = max(shapelet_fitter.b2_grid)

        ##Edge pad by one fifth the total width
        edge_pad = int(shapelet_fitter.fits_data.data.shape[0] / 5)

        ##Set up a new edge padded coord system
        new_fits_data = FITSInformation(shapelet_fitter.fits_data.fitsfile)
        new_fits_data.get_radec_edgepad(edge_pad=edge_pad)
        shpcoord_pad = ShapeletCoords(new_fits_data)
        shpcoord_pad.find_good_pixels()
        shpcoord_pad.fit_gauss_and_centre_coords(b1_max=False,b2_max=False)
        xrot,yrot = shpcoord_pad.radec2xy(b1, b2, crop=False)

        ##Generate a new A matrix with the edge padded coords
        shape = new_fits_data.data.shape
        _, _, A_shape_basis_edge = shapelets.gen_A_shape_matrix(n1s=n1s,n2s=n2s,xrot=xrot,yrot=yrot,
                                                      nmax=nmax,b1=b1,b2=b2,
                                                      convolve_kern=convolve_kern,shape=shape)

        ##Generate an image and plot
        fitted_coeffs = deepcopy(shapelet_fitter.fitted_coeffs)
        fitted_coeffs.shape = (len(fitted_coeffs),1)
        fit_data_edge = matmul(A_shape_basis_edge,fitted_coeffs)

        fit_data_edge.shape = shape

        do_subplot(fig,ax4,fit_data_edge,'Edge padded fit \n(convolved with restoring beam)',vmin,vmax)

    fig.tight_layout()
    fig.savefig('shamfi_%s_nmax%03d_p%03d.png' %(save_tag,shapelet_fitter.nmax,shapelet_fitter.model_percentage), bbox_inches='tight')
    plt.close()
