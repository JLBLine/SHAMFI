#!/usr/bin/env python
from __future__ import print_function,division


def get_parser():

    import argparse

    parser = argparse.ArgumentParser(description="A script to fit a shapelet model consistent with the RTS or WODEN")

    parser.add_argument('--rts_srclist', default=False, action='store_true',
        help='Just save an RTS style srclist - default is to saving both RTS and WODEN srclists')

    parser.add_argument('--woden_srclist', default=False, action='store_true',
        help='Just save a WODEN style srclist - default is to saving both RTS and WODEN srclists')

    parser.add_argument('--no_srclist', default=False, action='store_true',
        help='Do not save any srclists')

    parser.add_argument('--fits_file', default=False,
        help='Name of fits file to fit data from')

    parser.add_argument('--b1s', default=False,
        help="The beta scale range along the major axis (arcmins). Enter as a lower and upper bound separated by a comma, eg 2,10" )

    parser.add_argument('--b2s', default=False,
        help="The beta scale along the minor (arcmins). Enter as a lower and upper bound separated by a comma, eg 2,10" )

    parser.add_argument('--nmax', default=10,type=int,
        help='Maximum value of n1 to include in the basis functions - current maximum possible in the RTS is 100 \
                         (The bigger the n1, the higher the resolution of the fitted model)')

    parser.add_argument('--save_tag', default='model',
        help='A tag to name the outputs with - defaults to "model"')

    parser.add_argument('--plot_lims', default=False,
        help='Flux limits for the plot - enter as vmin,vmax. Default is min(image),max(image)')

    parser.add_argument('--freq', default='from_FITS',
        help='Frequency (Hz) of the image - defaults to looking for keyword FREQ and associated value')

    parser.add_argument('--already_jy_per_pixel', default=False, action='store_true',
        help='Add to NOT convert pixels from Jy/beam into Jy/pixel')

    parser.add_argument('--edge_pad', default=0,type=int,
        help="Add empty pixels outside image to stop fitting artefacts outside the desired image - defaults to 0 pixels. \
        Set to desired amount using --edge_pad=number." )

    parser.add_argument('--num_beta_values', default=5,type=int,
        help="Num of beta params fit over ranges --b1s and --b2s")

    parser.add_argument('--exclude_box', action='append',
        help='Any number of areas to exclude from the fit. Specify by user pixel numbers. Add each box as as: \
        x_low,x_high,y_low,y_high. For example, to exclude to areas, one between \
        0 to 10 in the x range, 10 to 20 in the y range, and the other between 100 to 400 in the x range, \
        300 to 455 in the y range, enter this on the command line: \
        fit_shapelets.py --exclude_box 0,10,10,20 --exclude_box 100,400,300,455')

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

    parser.add_argument('--ignore_negative',default=False, action='store_true',
        help='Add to ignore all negative pixels in image during the fit')

    parser.add_argument('--max_baseline',default=3e3,
        help='If no restoring beam information available, use this maximum baseline length in \
              conjunction with the frequency to calculate a resolution to set BMAJ and BMIN')

    ##TODO - make the script able to just plot a given shapelet model
    # parser.add_argument('--just_plot', default=False, action='store_true',
    #     help='Default behaviour is to fit a shapelet model to --fits_file. If just plotting pass this to switch off fitting')
    #
    # parser.add_argument('--plot_reso', default=0.05,
    #     help='Resolution (deg) of output plot when using --just_plot. Default = 0.05')
    #
    # parser.add_argument('--plot_size', default=0.75,
    #     help='Size (deg) of output plot when using --just_plot. Default = 0.75')

    ##Version and citation informations
    parser.add_argument('--version', default=False, action='store_true',
        help='Prints the version info and exits')
    parser.add_argument('--cite', default=False, action='store_true',
        help='Prints a bibtex entry for citing this work (well it will once the paper is published)')


    return parser

def apply_srclist_option(args,shapelet_fitter,save_tag):
    '''Uses the user supplied arguments to write out the correct number of
    srclists'''
    if args.no_srclist:
        pass
    else:
        if args.rts_srclist and args.woden_srclist:
            shapelet_fitter.save_srclist(save_tag)
        elif args.rts_srclist:
            shapelet_fitter.save_srclist(save_tag,woden_srclist=False)
        elif args.woden_srclist:
            shapelet_fitter.save_srclist(save_tag,rts_srclist=False)
        else:
            shapelet_fitter.save_srclist(save_tag)


if __name__ == '__main__':
    from sys import path
    path.append('/home/jline/software/SHAMFI/')
    from shamfi import read_FITS_image, shapelet_coords, shamfi_plotting, shapelets
    from shamfi import __cite__
    from numpy import *
    from astropy.wcs import WCS
    # import shamfi
    # from shamfi.shamfi_lib import *
    from shamfi.git_helper import print_version_info
    from copy import deepcopy
    from subprocess import check_output
    import os
    from sys import exit

    ##Convert degress to radians
    D2R = pi/180.
    ##Convert radians to degrees
    R2D = 180./pi

    ##Grab the parser and parse some args
    parser = get_parser()
    args = parser.parse_args()

    ##print out the version if requested
    if args.version:
        print_version_info(os.path.realpath(__file__))
        exit()
    ## print out the citation information if requested
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

    ##Check compression args
    if args.compress:
        try:
            compress_values = array(args.compress.split(','),dtype=float)
        except:
            print('Cannot convert --compress into something sensible. Please check you have comma separated value e.g. --compress=90,80,70')
            print('Continuing without running compression')

    ##Read in the FITS data and header
    fits_data = read_FITS_image.FITSInformation(args.fits_file)

    ##Check everything that was read in
    if fits_data.read_data:
        pass
    else:
        msg = 'Unable to read basic data from FITS this fits file: \n'
        msg += '%s. Exiting now.' %args.fits_file
        exit(msg)

    ##Check the frequency has been set - request it to be specified if not
    if args.freq == 'from_FITS':
        if fits_data.found_freq:
            pass
        else:
            msg = 'Could not find necessary frequency information in the FITS file. \n'
            msg += 'Please specify using the --freq argument. Exiting now.'
            exit(msg)
    else:
        try:
            fits_data.freq = float(args.freq)
        except:
            msg = 'Could not change --freq=%s into a sensible frequency. \n' %args.freq
            msg += 'Please check and try again - exiting now'
            exit(msg)

    ##If reading major,minor information from the FITS file has failed,
    ##estimate using the maxiumum baseline and frequency
    if fits_data.got_convert2pixel:
        pass
    else:
        VELC = 299792458.0
        baseline = float(args.baseline)
        reso = ((VELC/(fits_data.freq*1e+6))/baseline)*R2D
        fits_data.bmaj = reso*0.5
        fits_data.bmin = reso*0.5
        rest_pa = 0.0
        print('Assuming BMAJ,BMIN = %.3f,%.3f' %(bmaj,bmin))

        fits_data.solid_beam = (pi*rest_bmaj*rest_bmin) / (4*log(2))
        fits_data.solid_pixel = abs(ra_reso)*dec_reso

        fits_data.convert2pixel = solid_pixel/solid_beam
    #
    # ##Unless specified, convert from Jy / beam to Jy / pixel
    if not args.already_jy_per_pixel:
        fits_data.covert_to_jansky_per_pix()

    ##If requested, edge pad the data, otherwise just create 1D RA/DEC arrays
    fits_data.get_radec_edgepad(edge_pad=args.edge_pad)

    ##Set up a shapelet coordinate class based on the FITS information
    shpcoord = shapelet_coords.ShapeletCoords(fits_data)

    ##Find the indexes of pixels to be fitted, based on user supplied args
    ##This function will also find the flux-weighted centre of the image,
    ##based on the pixel cuts
    shpcoord.find_good_pixels(fit_box=args.fit_box,exclude_box=args.exclude_box,ignore_negative=args.ignore_negative)
    # pixel_inds_to_use = find_good_pixels(args,edge_pad,flat_data,len1+2*edge_pad,ignore_negative=args.ignore_negative)
    print('Sum of flux in data is %.2f Jy' %(sum(fits_data.flat_data[shpcoord.pixel_inds_to_use])))

    ##Fit a gaussian around the central point, using the beta params as an
    ##initial guess. Also centres the zeroes the coord system about the fits
    shpcoord.fit_gauss_and_centre_coords(b1_max=b1_max,b2_max=b2_max)

    ##If requested, plot the initial gaussian fit
    if args.plot_initial_gaussian_fit: shamfi_plotting.plot_gaussian_fit(shpcoord,save_tag)

    # ##Set up the grids over which to fit b1 and b2
    b1_grid = linspace((b1_min/60.0)*D2R,(b1_max/60.0)*D2R,num_beta_points)
    b2_grid = linspace((b2_min/60.0)*D2R,(b2_max/60.0)*D2R,num_beta_points)

    print('b1_range is',(b1_grid / D2R)*60.0)
    print('b2_range is',(b2_grid / D2R)*60.0)
    #
    ##Calculate a restoring beam kernel to convolve the basis functions with
    fits_data.create_restoring_kernel()

    ##setup a shapelet fitting object
    shapelet_fitter = shapelets.FitShapelets(fits_data=fits_data,shpcoord=shpcoord)

    ##TODO make switching off the output FITS file a parser arguement
    shapelet_fitter.do_grid_search_fit(b1_grid, b2_grid, nmax, save_tag=save_tag)

    ##TODO make switching this plotting off a parser arguement
    shamfi_plotting.plot_full_shamfi_fit(shapelet_fitter, save_tag, plot_edge_pad=args.plot_edge_pad)

    ##Plot the grid residual search if you want
    if args.plot_resid_grid:
        shamfi_plotting.plot_grid_search(shapelet_fitter, save_tag)
        # savez_compressed('%s_grid.npz' %save_tag,matrix_plot=shapelet_fitter.residuals_array,
        #     b1_grid=shapelet_fitter.b1_grid,b2_grid=shapelet_fitter.b2_grid)

    apply_srclist_option(args,shapelet_fitter,save_tag)

    ##Do some compression if you fancy it
    if args.compress:
        ##order the basis functions by their absolute flux contribution to the model
        shapelet_fitter.find_flux_order_of_basis_functions()

        ##For each level of compression, do a grid search
        for compress_value in compress_values:
            ##TODO make switching off the output FITS file a parser arguement
            shapelet_fitter.do_grid_search_fit_compressed(compress_value,save_tag=save_tag)

            ##TODO make switching this plotting off a parser arguement
            shamfi_plotting.plot_full_shamfi_fit(shapelet_fitter, save_tag, plot_edge_pad=args.plot_edge_pad)

            ##Plot the grid residual search if you want
            if args.plot_resid_grid:
                shamfi_plotting.plot_grid_search(shapelet_fitter, save_tag)
                # savez_compressed('%s_grid.npz' %save_tag,matrix_plot=shapelet_fitter.residuals_array,
                #     b1_grid=shapelet_fitter.b1_grid,b2_grid=shapelet_fitter.b2_grid)

            apply_srclist_option(args,shapelet_fitter,save_tag)
