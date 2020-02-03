#!/usr/bin/env python
from __future__ import print_function,division
from numpy import *
from astropy.wcs import WCS
import shamfi
from shamfi.shamfi_lib import *
from shamfi.git_helper import print_version_info
from copy import deepcopy
from subprocess import check_output
import argparse
import os
from shamfi import __cite__
from sys import exit

parser = argparse.ArgumentParser(description="A script to fit a shapelet model consistent with the RTS or WODEN")

parser.add_argument('--rts_srclist', default=False, action='store_true',
                    help='Output an RTS style srclist')

parser.add_argument('--woden_srclist', default=False, action='store_true',
                    help='Output a WODEN style srclist')

parser.add_argument('--fits_file', default=False,
                    help='Name of fits file to fit data from')

parser.add_argument('--b1s', default=False,
                    help="The beta scale range along the major axis (arcmins). Enter as a lower and upper bound separated by a comma, eg 2,10" )

parser.add_argument('--b2s', default=False,
                    help="The beta scale along the minor (arcmins). Enter as a lower and upper bound separated by a comma, eg 2,10" )

parser.add_argument('--nmax', default=31,type=int,
                    help='Maximum value of n1 to include in the basis functions - current maximum possible in the RTS is 31\n (The bigger the n1, the higher the resolution of the fitted model)')

parser.add_argument('--save_tag', default='model',
                    help='A tag to name the outputs with - defaults to "model"')

parser.add_argument('--plot_lims', default=False,
                help='Flux limits for the plot - enter as vmin,vmax. Default is min(image),max(image)')

parser.add_argument('--freq', default='from_FITS',
                help='Frequency (Hz) of the image - defaults to looking for keyword FREQ and associated value')

parser.add_argument('--already_jy_per_pixel', default=False, action='store_true',
                    help='Add to NOT convert pixels from Jy/beam into Jy/pixel')

parser.add_argument('--edge_pad', default=0,type=int,
                    help="Add empty pixels outside image to stop fitting artefacts outside the desired image - defaults to 0 pixels. Set to desired amount using --edge_pad=number." )

parser.add_argument('--num_beta_values', default=5,type=int,
                    help="Num of beta params fit over ranges --b1s and --b2s")

parser.add_argument('--exclude_box', action='append',
    help='Any number of areas to exclude from the fit. Specify by user pixel numbers. Add each box as as: \
    x_low,x_high,y_low,y_high. For example, to exclude to areas, one between \
    0 to 10 in the x range, 10 to 20 in the y range, and the other between 100 to 400 in the x range, \
    300 to 455 in the y range, enter this on the command line: \
    fit_shapelets.py --exclude_box 0,10,10,20 --exclude_box 100,400,300,455')

parser.add_argument('--diff_box',default=False,
    help='Define a box in which to calculate residuals to the fit. Add as x_low,x_high,y_low,y_high')

parser.add_argument('--fit_box',default=False,
    help='Only fit shapelets to data within the designated box. Add as x_low,x_high,y_low,y_high')

parser.add_argument('--plot_resid_grid',default=False, action='store_true',
    help='Add to plot the residuals matrix found for all values of b1 and b2')

parser.add_argument('--plot_initial_gaussian_fit',default=False, action='store_true',
    help='Add to plot the intial gaussian fit used to fine b1 and b2')

parser.add_argument('--plot_edge_pad',default=False, action='store_true',
    help='When plotting fit results, also plot the fit with an edge pad to check the fit outside the image')

parser.add_argument('--compress',default=False,
    help='Add a list of comma separated percentage compression values to apply to the data, e.g. --compress=90,80,70')

parser.add_argument('--just_plot', default=False, action='store_true',
                    help='Default behaviour is to fit a shapelet model to --fits_file. If just plotting pass this to switch off fitting')

parser.add_argument('--plot_reso', default=0.05,
                    help='Resolution (deg) of output plot when using --just_plot. Default = 0.05')

parser.add_argument('--plot_size', default=0.75,
                    help='Size (deg) of output plot when using --just_plot. Default = 0.75')

##Version and citation informations
parser.add_argument('--version', default=False, action='store_true',
                    help='Prints the version info and exits')
parser.add_argument('--cite', default=False, action='store_true',
                    help='Prints a bibtex entry for citing this work (well it will once the paper is published)')

args = parser.parse_args()

if args.version:
    print_version_info(os.path.realpath(__file__))
    exit()

if args.cite:
    print(__cite__)
    exit()

##Get some argument values
save_tag = args.save_tag
nmax = args.nmax
edge_pad = args.edge_pad
b1_min,b1_max = map(float,args.b1s.split(','))
b2_min,b2_max = map(float,args.b2s.split(','))
num_beta_points = args.num_beta_values

if args.compress:
    try:
        compress_values = array(args.compress.split(','),dtype=float)
    except:
        print('Cannot convert --compress into something sensible. Please check you have comma separated value e.g. --compress=90,80,70')
        print('Continuing without running compression')

##Grab a bunch of data and details from the FITS to fit
data,flat_data,ras,decs,convert2pixel,ra_reso,dec_reso,freq,len1,len2,wcs,dims,rest_bmaj,rest_bmin,rest_pa = get_fits_info(args.fits_file,edge_pad=edge_pad,freq=args.freq)
pix_area_rad = ra_reso*dec_reso*D2R**2

##Unless specified, convert from Jy / beam to Jy / pixel
if not args.already_jy_per_pixel:
    flat_data *= convert2pixel
    data *= convert2pixel

##Find the indexes of pixels to be fitted
pixel_inds_to_use = find_good_pixels(args,edge_pad,flat_data,len1+2*edge_pad)
print('Sum of flux in data is %.2f' %(sum(flat_data[pixel_inds_to_use])))

##Find the flux weighted central pixel of the data to be fitted
ra_ind,dec_ind,ra_mesh,dec_mesh,ra_range,dec_range = find_image_centre_celestial(ras=ras,decs=decs,flat_data=flat_data,pixel_inds_to_use=pixel_inds_to_use,data=data)

##Fit a gaussian to the data to find pa
##guess is: amp, xo, yo, sigma_x, sigma_y, pa
initial_guess = (data.max(),ra_range[int(ra_ind)],dec_range[int(dec_ind)],(b1_max / 60.0)*D2R,(b2_max / 60.0)*D2R,0)
popt, pcov = opt.curve_fit(twoD_Gaussian, (ra_mesh, dec_mesh), flat_data, p0=initial_guess)

##Check pa is between 0 <= pa < 2pi
pa = popt[5]
if pa < 0:
    pa += 2*pi
##Necessary to move from my gaussian which has theta = 0 at x = 0 and
##actual PA which is east from north
pa += pi / 2.0
if pa > 2*pi:
    pa -= 2*pi

##Set central ra, dec pixel to zero in prep for scaling to x,y coords
ra_cent, dec_cent, ras, decs = set_central_pixel_to_zero(popt,ras,decs,ra_range,dec_range,args,edge_pad,dims,wcs)
##If requested, plot the initial gaussian fit
if args.plot_initial_gaussian_fit: plot_gaussian_fit(ra_mesh, dec_mesh, popt, data, save_tag)

##Set up the grids over which to fit b1 and b2
b1_grid = linspace((b1_min/60.0)*D2R,(b1_max/60.0)*D2R,num_beta_points)
b2_grid = linspace((b2_min/60.0)*D2R,(b2_max/60.0)*D2R,num_beta_points)

print('b1_range is',(b1_grid / D2R)*60.0)
print('b2_range is',(b2_grid / D2R)*60.0)

##Figure out a restoring beam kernel to convolve the basis functions with
rest_gauss_kern = create_restoring_kernel(rest_bmaj,rest_bmin,rest_pa,ra_reso,dec_reso)

##Number of coeffs up to order nmax
total_coeffs = 0.5*(nmax+1)*(nmax+2)
b1,b2,best_b1_ind,best_b2_ind,n1s,n2s,fitted_coeffs,fit_data_full,xrot,yrot,matrix_plot = do_grid_search_fit(total_coeffs,
                                                                                          flat_data,b1_grid,b2_grid,
                                                                                          pa,nmax,rest_gauss_kern,data,
                                                                                          pixel_inds_to_use,args,
                                                                                          num_beta_points,ras,decs)

##Plot the grid residual search if you want
if args.plot_resid_grid:
    plot_grid_search(matrix_plot,num_beta_points,b1_grid,b2_grid,save_tag)
    savez_compressed('%s_grid.npz' %save_tag,matrix_plot=matrix_plot,b1_grid=b1_grid,b2_grid=b2_grid)

##Write srclists if needed
if args.rts_srclist:
    save_srclist(save_tag=save_tag+'_nmax%03d_p100' %nmax, nmax=nmax, n1s=n1s, n2s=n2s, fitted_coeffs=fitted_coeffs, b1=b1, b2=b2,
        fitted_model=fit_data_full, ra_cent=ra_cent, dec_cent=dec_cent, freq=freq, pa=pa,
        pix_area=pix_area_rad)

if args.woden_srclist:
    save_srclist(save_tag=save_tag+'_nmax%03d_p100' %nmax, nmax=nmax, n1s=n1s, n2s=n2s, fitted_coeffs=fitted_coeffs, b1=b1, b2=b2,
        fitted_model=fit_data_full, ra_cent=ra_cent, dec_cent=dec_cent, freq=freq, pa=pa,
        pix_area=pix_area_rad,rts_srclist=False)

if args.compress:

    compressed_images = []

    _, _, A_shape_basis = gen_A_shape_matrix(xrot=xrot,yrot=yrot,nmax=nmax,b1=b1,b2=b2,convolve_kern=rest_gauss_kern,shape=data.shape)
    ##Sort the basis functions by highest absolute flux contribution to the model
    basis_sums, sums_order = order_basis_by_flux(fitted_coeffs,A_shape_basis)

    for compress_value in compress_values:
        print('--------------------------------------')
        print('Running compression at %.1f%%' %compress_value)

        ##Compress the basis functions by only including the basis functions that contribute up to some percentage of
        ##the overall flux of the model
        n1s_compressed,n2s_compressed,fitted_coeffs_compressed,order = compress_by_flux_percentage(basis_sums, sums_order, compress_value, n1s, n2s, fitted_coeffs)

        ##Do grid based fitting again in case a better b1,b2 combo works with the compressed coeffs
        b1_compressed,b2_compressed,n1s_compressed,n2s_compressed, \
        fitted_coeffs_compressed,fit_data_compressed,xrot_compressed, \
        yrot_compressed,matrix_plot_compressed = do_grid_search_fit(len(n1s_compressed),flat_data,b1_grid,b2_grid,pa,nmax,
                                                                       rest_gauss_kern,data,pixel_inds_to_use,args,
                                                                       num_beta_points,ras,decs,full_fit=False,
                                                                       n1s_compressed=n1s_compressed,n2s_compressed=n2s_compressed)

        ##Collect fitted images to plot later on
        compressed_images.append(fit_data_compressed)
        ##Write out srclists if asked for
        if args.rts_srclist:
            save_srclist(save_tag=save_tag+'_nmax%03d_p%03d' %(nmax,compress_value), nmax=nmax,
                n1s=n1s_compressed, n2s=n2s_compressed, fitted_coeffs=fitted_coeffs_compressed, b1=b1, b2=b2,
                fitted_model=fit_data_compressed, ra_cent=ra_cent, dec_cent=dec_cent, freq=freq, pa=pa,
                pix_area=pix_area_rad)

        if args.woden_srclist:
            save_srclist(save_tag=save_tag+'_nmax%03d_p%03d' %(nmax,compress_value), nmax=nmax,
                n1s=n1s_compressed, n2s=n2s_compressed, fitted_coeffs=fitted_coeffs_compressed, b1=b1, b2=b2,
                fitted_model=fit_data_compressed, ra_cent=ra_cent, dec_cent=dec_cent, freq=freq, pa=pa,
                pix_area=pix_area_rad, rts_srclist=False)

    ##Plot the results
    plot_compressed_fits(args, compressed_images, flat_data, data.shape, pixel_inds_to_use,
                      save_tag, compress_values, nmax)

##Doing the plots and stuff changes the shape of some arrays, so run all finals
##FITS and plot generation after compression
save_output_FITS(args.fits_file,fit_data_full,data.shape,save_tag,nmax,edge_pad,len1,len2,convert2pixel)
masked_data_plot = plot_full_fit(args, fit_data_full, flat_data.flatten(), data.shape, pixel_inds_to_use, save_tag, nmax, popt, pa, b1, b2, rest_gauss_kern, fitted_coeffs)
