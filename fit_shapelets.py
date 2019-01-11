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

    parser.add_argument('--already_jy_per_pixel', default=False, action='store_true',
                        help='Add to NOT convert pixels from Jy/beam into Jy/pixel')

    parser.add_argument('--edge_pad', default=100,type=int,
                        help="By default, add empty pixels outside image to stop fitting artefacts outside the desired image - defaults to 100 pixels. Set to desired amount using --edge_pad=number. To swich off, set --edge_pad=0" )

    parser.add_argument('--num_coeffs', default=100,type=int,
                        help="Refit using the most significant fitted basis functions, up to num_coeffs. Defaults to 100, change using --num_coeffs=integer" )

    args = parser.parse_args()

    save_tag = args.save_tag
    nmax = args.nmax


    edge_pad = args.edge_pad
    data,flat_data,ras,decs,ra_cent,dec_cent,convert2pixel,ra_reso,dec_reso = get_fits_info(args.fits_file,edge_pad=edge_pad,freq=args.freq)

    if not args.already_jy_per_pixel:
        flat_data *= convert2pixel
        data *= convert2pixel

    print('Sum of flux in data is %.2f' %(sum(flat_data)))

    ##TODO? Add in flux cut option for fitting
    ##TODO don't fit negative flux is good I think?
    # flux_cut = 0.0
    # data[data < flux_cut] = 0

    ra_cent_off,dec_cent_off,ra_ind,dec_ind = find_image_centre_celestial(ras=ras,decs=decs,flat_data=flat_data)

    dec_ind = floor(dec_ind / data.shape[0])

    hdu = fits.open(args.fits_file)
    wcs = WCS(hdu[0].header)
    try:
        ra_cent,dec_cent,meh1,meh2 = wcs.wcs_pix2world(ra_ind-edge_pad,dec_ind-edge_pad,0,0,0)
    except:
        ra_cent,dec_cent = wcs.wcs_pix2world(ra_ind-edge_pad,dec_ind-edge_pad,0)
    ras -= ra_cent_off
    decs -= dec_cent_off

    ##Scale beta params in radian
    b1 = (args.b1 / 60.0)*D2R
    b2 = (args.b2 / 60.0)*D2R

    #TODO - setup PA fitting
    pa = 0.0
    xrot,yrot = radec2xy(ras,decs,pa,b1,b2)

    ##Does a fit using all possible basis functions up to nmax
    n1s, n2s, A_shape_basis = gen_A_shape_matrix(xrot=xrot,yrot=yrot,nmax=nmax,b1=b1,b2=b2)
    fitted_coeffs = linear_solve(flat_data=flat_data,A_shape_basis=A_shape_basis)

    ##Creates a model of the fully fitted coeffs and a matching srclist
    fit_data_full = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis)
    if args.no_srclist:
        pass
    else:
        pix_area = ra_reso*dec_reso
        save_srclist(save_tag=save_tag+'_ncoeffs_full', nmax=nmax, n1s=n1s, n2s=n2s, fitted_coeffs=fitted_coeffs, b1=b1, b2=b2,
            fitted_model=fit_data_full, ra_cent=ra_cent, dec_cent=dec_cent, freq=args.freq, pa=pa,
            pix_area=pix_area)


    ##Sort the basis functions by highest ranking, and return to top num_coeffs
    n1s_compressed,n2s_compressed,fitted_coeffs_compressed,order = compress_coeffs(n1s,n2s,fitted_coeffs,args.num_coeffs,xrot,yrot,b1,b2)

    A_shape_basis_compressed = gen_reduced_A_shape_matrix(n1s=n1s_compressed,n2s=n2s_compressed,xrot=xrot,yrot=yrot,b1=b1,b2=b2)
    fitted_coeffs_compressed = linear_solve(flat_data=flat_data,A_shape_basis=A_shape_basis_compressed)
    fit_data_compressed = fitted_model(coeffs=fitted_coeffs_compressed,A_shape_basis=A_shape_basis_compressed)

    if args.no_srclist:
        pass
    else:
        pix_area = ra_reso*dec_reso
        save_srclist(save_tag=save_tag+'_ncoeffs%03d' %len(n1s_compressed), nmax=nmax, n1s=n1s_compressed, n2s=n2s_compressed, fitted_coeffs=fitted_coeffs_compressed, b1=b1, b2=b2,
            fitted_model=fit_data_compressed, ra_cent=ra_cent, dec_cent=dec_cent, freq=args.freq, pa=pa,
            pix_area=pix_area)

    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    if args.plot_lims:
        vmin,vmax = map(float,args.plot_lims.split(','))

    else:
        vmin,vmax = False, False

    def do_plot(ax,data,label):
        if vmin:
            ax1.imshow(data,origin='lower',vmin=vmin,vmax=vmax)
        else:
            im = ax.imshow(data,origin='lower') #,extent=extent,aspect=aspect) #'ras_mesh.shape[0]'
        add_colourbar(fig=fig,im=im,ax=ax)
        ax.set_title(label)

    do_plot(ax1,data,'Data')

    fit_data_full.shape = data.shape
    fit_data_compressed.shape = data.shape

    do_plot(ax2,fit_data_full,'Full fit (%d coeffs)' %len(n1s))
    do_plot(ax3,fit_data_compressed,'Compressed fit (%d coeffs)' %len(n1s_compressed))
    do_plot(ax4,data - fit_data_compressed,'Residuals')

    fig.tight_layout()

    fig.savefig('shapelets_%s_nmax%d_fit.png' %(save_tag,nmax), bbox_inches='tight')
