#!/usr/bin/env python
from __future__ import print_function,division
from numpy import *
import matplotlib.pyplot as plt
from scipy.special import factorial,eval_hermite
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
import math as m
from math import atan2
from numpy import abs as np_abs
from astropy.wcs import WCS
from shapelets import *

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="A script to fit a shapelet model consistent with the RTS")

    parser.add_argument('--just_plot', default=False, action='store_true',
                        help='Default behaviour is to fit a shapelet model to --fits_file. If just plotting pass this to switch off fitting')

    parser.add_argument('--no_srclist', default=False, action='store_true',
                        help='Default behaviour is to create an RTS style srclist - add this to switch off')

    parser.add_argument('--fits_file', default=False,
                        help='Name of fits file to fit data from - also required for plotting shapelets to get coords system')

    parser.add_argument('--b1', default=False,type=float,
                        help="The beta scale along the major axis (arcmins) - currently this is hardcoded to DEC (no rotation of basis functions is applied)" )

    parser.add_argument('--b2', default=False,type=float,
                        help="The beta scale along the minor (arcmins) - currently this is hardcoded to RA (no rotation of basis functions is applied)" )

    parser.add_argument('--nmax', default=31,type=int,
                        help='Maximum value of n1 to include in the basis functions - current maximum possible in the RTS is 31\n (The bigger the n1, the higher the resolution of the fitted model)')

    parser.add_argument('--save_tag', default='model',
                        help='A tag to name the outputs with - defaults to "model"')

    parser.add_argument('--plot_lims', default=False,
                    help='Flux limits for the plot - enter as vmin,vmax. Default is min(image),max(image)')

    parser.add_argument('--freq', default=False,type=float,
                    help='Frequency (MHz) to put in the srclist')

    args = parser.parse_args()

    save_tag = args.save_tag
    nmax = args.nmax


    edge_pad = 100
    data,flat_data,ras,decs,ra_cent,dec_cent,convert2pixel = get_fits_info(args.fits_file,edge_pad=100)


    xlow = 140 + edge_pad
    xhigh = 860 + edge_pad
    ylow = 300 + edge_pad
    yhigh = 800 + edge_pad

    new_data = zeros(data.shape)
    new_data[ylow:yhigh,xlow:xhigh] = data[ylow:yhigh,xlow:xhigh]
    data = new_data

    flux_cut = 0.1
    data[data < flux_cut] = 0
    flat_data = data.flatten()
    flux_cut = 0


    b1 = (args.b1 / 60.0)*D2R
    b2 = (args.b2 / 60.0)*D2R

    pa = -77*D2R
    # pa = 0.0
    xrot,yrot = radec2xy(ras,decs,pa,b1,b2)


    n1s, n2s, A_shape_basis = gen_A_shape_matrix(xrot=xrot,yrot=yrot,nmax=nmax,b1=b1,b2=b2)
    fitted_coeffs = linear_solve(flat_data=flat_data,A_shape_basis=A_shape_basis)

    # minco(flat_data,b1,b2,n1s,n2s,xrot,yrot,fitted_coeffs)

    fit_data = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis)

    if args.no_srclist:
        pass
    else:
        save_srclist(save_tag=save_tag, nmax=nmax, n1s=n1s, n2s=n2s, fitted_coeffs=fitted_coeffs, b1=b1, b2=b2,
            fitted_model=fit_data, ra_cent=ra_cent, dec_cent=dec_cent, freq=args.freq, pa=pa, convert2pixel=convert2pixel)

    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(221)


    if args.plot_lims:
        vmin,vmax = map(float,args.plot_lims.split(','))
        im1 = ax1.imshow(data,origin='lower',vmin=vmin,vmax=vmax)
    else:
        im1 = ax1.imshow(data,origin='lower') #,extent=extent,aspect=aspect) #'ras_mesh.shape[0]'
    add_colourbar(fig=fig,im=im1,ax=ax1)
    ax1.set_title('Data')

    ax2 = fig.add_subplot(222)
    fit_data.shape = data.shape
    # fit_data.shape = dataset.shape
    im2 = ax2.imshow(fit_data,origin='lower')
    add_colourbar(fig=fig,im=im2,ax=ax2)
    ax2.set_title('Fit')

    ax3 = fig.add_subplot(212)
    if args.plot_lims:
        vmin,vmax = map(float,args.plot_lims.split(','))
        im3 = ax3.imshow(data - fit_data,origin='lower',vmin=vmin,vmax=vmax)
    else:
        im3 = ax3.imshow(data - fit_data,origin='lower')
    add_colourbar(fig=fig,im=im3,ax=ax3)
    ax3.set_title('Residuals')

    fig.tight_layout()

    fig.savefig('shapelets_%s_nmax%d_fit.png' %(save_tag,nmax), bbox_inches='tight')
