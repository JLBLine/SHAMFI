#!/usr/bin/env python
from __future__ import print_function,division
from numpy import *
import matplotlib
matplotlib.use('Agg')
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
from copy import deepcopy
from astropy.modeling.models import Gaussian2D

factor = 2. * sqrt(2.*log(2.))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="A script to fit a shapelet model consistent with the RTS")

    parser.add_argument('--just_plot', default=False, action='store_true',
                        help='Default behaviour is to fit a shapelet model to --fits_file. If just plotting pass this to switch off fitting')

    parser.add_argument('--no_srclist', default=False, action='store_true',
                        help='Default behaviour is to create an RTS style srclist - add this to switch off')

    parser.add_argument('--fits_file', default=False,
                        help='Name of fits file to fit data from - also required for plotting shapelets to get coords system')

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

    parser.add_argument('--freq', default=False,type=float,
                    help='Frequency (MHz) to put in the srclist')

    parser.add_argument('--already_jy_per_pixel', default=False, action='store_true',
                        help='Add to NOT convert pixels from Jy/beam into Jy/pixel')

    parser.add_argument('--edge_pad', default=100,type=int,
                        help="By default, add empty pixels outside image to stop fitting artefacts outside the desired image - defaults to 100 pixels. Set to desired amount using --edge_pad=number. To swich off, set --edge_pad=0" )

    parser.add_argument('--num_coeffs', default=100,type=int,
                        help="Refit using the most significant fitted basis functions, up to num_coeffs. Defaults to 100, change using --num_coeffs=integer" )

    parser.add_argument('--num_beta_values', default=5,type=int,
                        help="Num of beta params to to test fit over")

    parser.add_argument('--exclude_box', action='append',
        help='Any number of areas to exclude from the fit. Specify by user pixel numbers. Add each box as as: \
        x_low,x_high,y_low,y_high. For example, to exclude to areas, one between \
        0 to 10 in the x range, 10 to 20 in the y range, and the other between 100 to 400 in the x range, \
        300 to 455 in the y range, enter this on the command line: \
        fit_shapelets.py --exclude_box 0,10,10,20 --exclude_box 100,400,300,455')

    args = parser.parse_args()

    save_tag = args.save_tag
    nmax = args.nmax
    edge_pad = args.edge_pad
    hdu,data,flat_data,ras,decs,ra_cent,dec_cent,convert2pixel,ra_reso,dec_reso = get_fits_info(args.fits_file,edge_pad=edge_pad,freq=args.freq)

    if not args.already_jy_per_pixel:
        flat_data *= convert2pixel
        data *= convert2pixel

    y_len,x_len = data.shape

    ##Need to mask out bad pixels here
    try:
        avoid_inds = []
        for box in args.exclude_box:
            low_x,high_x,low_y,high_y = array(map(int,box.split(','))) + edge_pad
            for y in range(low_y,high_y+1):
                for x in range(low_x,high_x+1):
                    avoid_inds.append(y*x_len + x)

        good_inds = arange(len(flat_data))
        good_inds = setxor1d(good_inds,avoid_inds)
    except:
        good_inds = arange(len(flat_data))

    print('Sum of flux in data is %.2f' %(sum(flat_data[good_inds])))

    ##TODO? Add in flux cut option for fitting
    ##TODO don't fit negative flux is good I think?
    # flux_cut = 0.0
    # data[data < flux_cut] = 0

    ra_cent_off,dec_cent_off,ra_ind,dec_ind = find_image_centre_celestial(ras=ras[good_inds],decs=decs[good_inds],flat_data=flat_data[good_inds])
    dec_ind = floor(dec_ind / data.shape[0])

    ra_mesh = deepcopy(ras)
    ra_mesh.shape = data.shape

    dec_mesh = deepcopy(decs)
    dec_mesh.shape = data.shape

    ra_range = ra_mesh[0,:]
    dec_range = dec_mesh[:,0]

    b1_min,b1_max = map(float,args.b1s.split(','))
    b2_min,b2_max = map(float,args.b2s.split(','))

    # b1_guess = (b1_max - b1_min) / 2.0
    # b2_guess = (b2_max - b2_min) / 2.0

    b1_guess = 6.0
    b2_guess = 6.0

    ##guess is: amp, xo, yo, sigma_x, sigma_y, pa
    initial_guess = (data.max(),ra_range[int(ra_ind)],dec_range[int(dec_ind)],(b1_guess / 60.0)*D2R,(b2_guess / 60.0)*D2R,0)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (ra_mesh, dec_mesh), flat_data, p0=initial_guess)

    x0 = popt[1]
    y0 = popt[2]
    b1 = popt[3]
    b2 = popt[4]
    pa = popt[5]

    ra_offs = np_abs(ra_range - x0)
    dec_offs = np_abs(dec_range - y0)

    ra_ind = where(ra_offs < abs(ra_range[1] - ra_range[0])/2.0)[0][0]
    dec_ind = where(dec_offs < abs(dec_range[1] - dec_range[0])/2.0)[0][0]

    ra_cent_off = ra_range[ra_ind]
    dec_cent_off = dec_range[dec_ind]

    # print('Amp',popt[0])
    # print('xo',popt[1]/D2R)
    # print('yo',popt[2]/D2R)
    # print('major',(popt[3]/D2R)*60.)
    # print('minor',(popt[4]/D2R)*60.)
    # print('pa',popt[5]/D2R)
    #
    hdu = fits.open(args.fits_file)
    wcs = WCS(hdu[0].header)
    try:
        ra_cent,dec_cent,meh1,meh2 = wcs.wcs_pix2world(ra_ind-edge_pad,dec_ind-edge_pad,0,0,0)
    except:
        ra_cent,dec_cent = wcs.wcs_pix2world(ra_ind-edge_pad,dec_ind-edge_pad,0)

    # print(ra_cent,dec_cent)
    #
    ras -= ra_cent_off
    decs -= dec_cent_off

    mask = twoD_Gaussian((ra_mesh, dec_mesh), *popt)
    mask.shape = ra_mesh.shape


    fig = plt.figure(figsize=(7,7))

    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(111)

    ra_mesh /= D2R
    dec_mesh /= D2R

    # im1 = ax1.imshow(ra_mesh)
    # add_colourbar(ax=ax1,im=im1,fig=fig)
    # im2 = ax2.imshow(dec_mesh)
    # add_colourbar(ax=ax2,im=im2,fig=fig)

    im3 = ax3.imshow(data,origin='lower')
    ax3.contour(mask,colors='r',alpha=0.3)

    add_colourbar(ax=ax3,im=im3,fig=fig)

    fig.savefig('pa_fit_%s.png' %save_tag ,bbox_inches='tight')

    if pa < 0:
        pa += 2*pi

    pa += pi / 2.0

    if pa > 2*pi:
        pa -= 2*pi

    num_points = args.num_beta_values

    b1_grid = linspace((b1_min/60.0)*D2R,(b1_max/60.0)*D2R,num_points)
    b2_grid = linspace((b2_min/60.0)*D2R,(b2_max/60.0)*D2R,num_points)

    print('b1_range is',(b1_grid / D2R)*60.0)
    print('b2_range is',(b2_grid / D2R)*60.0)

    ##Figure out a restoring beam kernel to convolve the basis functions with

    header = hdu[0].header
    rest_bmaj = float(header['BMAJ'])
    rest_bmin = float(header['BMIN'])
    rest_pa = header['BPA'] * (pi/180.0)

    x_stddev = rest_bmaj / (factor*abs(float(header['CDELT1'])))
    y_stddev = rest_bmin / (factor*float(header['CDELT2']))

    rest_gauss_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev, y_stddev=y_stddev,theta=pi/2 + rest_pa)

    xrange = arange(-25,26)
    yrange = arange(-25,26)

    x_mesh, y_mesh = meshgrid(xrange,yrange)
    rest_gauss_kern = rest_gauss_func(x_mesh,y_mesh)
    rest_gauss_kern /= rest_gauss_kern.sum()

    # @profile
    def do_fitting(b1,b2):
        xrot,yrot = radec2xy(ras,decs,pa,b1,b2)

        ##Does a fit using all possible basis functions up to nmax
        n1s, n2s, A_shape_basis = gen_A_shape_matrix(xrot=xrot,yrot=yrot,nmax=nmax,b1=b1,b2=b2,convolve_kern=rest_gauss_kern,shape=data.shape)

        print(A_shape_basis.shape,A_shape_basis[good_inds,:].shape)


        fitted_coeffs = linear_solve(flat_data=flat_data[good_inds],A_shape_basis=A_shape_basis[good_inds,:])

        ##Creates a model of the fully fitted coeffs and a matching srclist
        fit_data_full = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis)

        resid = find_resids(data=flat_data,fit_data=fit_data_full)
        print(resid)

        return resid, fit_data_full, fitted_coeffs, n1s, n2s, xrot, yrot


    fit_datas = []
    resids = []
    matrix_plot = zeros((num_points,num_points))

    total_coeffs = 0.5*(nmax+1)*(nmax+2)
    fitted_coeffs_matrix = zeros((num_points,num_points,int(total_coeffs)))
    fitted_datas_matrix = zeros((num_points,num_points,len(flat_data)))
    xrot_matrix = zeros((num_points,num_points,len(flat_data)))
    yrot_matrix = zeros((num_points,num_points,len(flat_data)))

    for b1_ind in arange(num_points):
        for b2_ind in arange(num_points):
            # try:
            print('Doing fit',b1_ind*num_points+b2_ind+1,'of',num_points**2)
            resid, fit_data, fitted_coeffs, n1s, n2s, xrot, yrot = do_fitting(b1_grid[b1_ind],b2_grid[b2_ind])

            matrix_plot[b1_ind,b2_ind] = resid
            fitted_coeffs_matrix[b1_ind,b2_ind] = fitted_coeffs.flatten()
            fitted_datas_matrix[b1_ind,b2_ind] = fit_data.flatten()
            xrot_matrix[b1_ind,b2_ind] = xrot
            yrot_matrix[b1_ind,b2_ind] = yrot
            # except:
            #     matrix_plot[b1_ind,b2_ind] = nan
            #     fitted_coeffs_matrix[b1_ind,b2_ind] = zeros(int(total_coeffs))
            #     fitted_datas_matrix[b1_ind,b2_ind] = zeros(len(flat_data))
            #     xrot_matrix[b1_ind,b2_ind] = zeros((num_points,num_points))
            #     yrot_matrix[b1_ind,b2_ind] = zeros((num_points,num_points))


    print(matrix_plot)
    best_b1_ind,best_b2_ind = where(matrix_plot == nanmin(matrix_plot))

    savez_compressed('%s_grid.npz' %save_tag,matrix_plot=matrix_plot,b1_grid=b1_grid,b2_grid=b2_grid)

    b1 = b1_grid[best_b1_ind[0]]
    b2 = b2_grid[best_b2_ind[0]]
    fitted_coeffs = fitted_coeffs_matrix[best_b1_ind[0],best_b2_ind[0]]
    fit_data_full = fitted_datas_matrix[best_b1_ind[0],best_b2_ind[0]]
    xrot = xrot_matrix[best_b1_ind[0],best_b2_ind[0]]
    yrot = yrot_matrix[best_b1_ind[0],best_b2_ind[0]]

    best_b1 = (b1/D2R)*60.0
    best_b2 = (b2/D2R)*60.0
    print('Best b1',best_b1,'Best b2',best_b2)

    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(1,1,1)
    # im = ax.imshow(matrix_plot)
    #
    # ax.set_xticks(arange(num_points))
    # ax.set_yticks(arange(num_points))
    #
    # labelsy = ['%.2f' %b for b in (b1_grid/D2R)*60.0]
    # labelsx = ['%.2f' %b for b in (b2_grid/D2R)*60.0]
    #
    # ax.set_yticklabels(labelsy)
    # ax.set_xticklabels(labelsx)
    #
    # ax.set_xlabel('b2 (arcmins)')
    # ax.set_ylabel('b1 (arcmins)')
    #
    # add_colourbar(fig=fig,im=im,ax=ax)
    #
    # ax.contour(matrix_plot,colors='w',alpha=0.4)
    #
    # fig.savefig('test_grid-fit_matrix_%s.png' %save_tag, bbox_inches='tight')
    # plt.close()
    #

    if args.no_srclist:
        pass
    else:
        pix_area = ra_reso*dec_reso
        save_srclist(save_tag=save_tag+'_ncoeffs_full', nmax=nmax, n1s=n1s, n2s=n2s, fitted_coeffs=fitted_coeffs, b1=b1, b2=b2,
            fitted_model=fit_data_full, ra_cent=ra_cent, dec_cent=dec_cent, freq=args.freq, pa=pa,
            pix_area=pix_area)

    ##Sort the basis functions by highest ranking, and return to top num_coeffs
    n1s_compressed,n2s_compressed,fitted_coeffs_compressed,order = compress_coeffs(n1s,n2s,fitted_coeffs,args.num_coeffs,xrot,yrot,b1,b2,convolve_kern=rest_gauss_kern,shape=data.shape)

    A_shape_basis_compressed = gen_reduced_A_shape_matrix(n1s=n1s_compressed,n2s=n2s_compressed,xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=rest_gauss_kern,shape=data.shape)
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

    A_shape_basis_no_conv = gen_reduced_A_shape_matrix(n1s=n1s,n2s=n2s,xrot=xrot,yrot=yrot,b1=b1,b2=b2)
    fitted_coeffs.shape = (len(fitted_coeffs),1)
    fit_data_no_conv = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis_no_conv)


    def do_plot(ax,data,label):
        if vmin:
            ax1.imshow(data,origin='lower',vmin=vmin,vmax=vmax)
        else:
            im = ax.imshow(data,origin='lower') #,extent=extent,aspect=aspect) #'ras_mesh.shape[0]'
        add_colourbar(fig=fig,im=im,ax=ax)
        ax.set_title(label)

    bad_inds = setdiff1d(arange(len(flat_data)),good_inds)
    flat_data[bad_inds] = nan

    flat_data.shape = data.shape

    do_plot(ax1,flat_data,'Data')

    fit_data_full.shape = data.shape
    fit_data_compressed.shape = data.shape
    fit_data_no_conv.shape = data.shape
    do_plot(ax2,fit_data_full,'Full convolved fit (%d coeffs)' %len(n1s))
    do_plot(ax3,fit_data_no_conv,'Unconvolved fit (%d coeffs)' %len(n1s))
    do_plot(ax4,data - fit_data_full,'Residuals of convolved fit')

    fig.tight_layout()

    fig.savefig('shapelets_%s_nmax%d_fit.png' %(save_tag,nmax), bbox_inches='tight')


    len1 = hdu[0].header['NAXIS1']
    len2 = hdu[0].header['NAXIS2']

    if len(hdu[0].data.shape) == 2:
        hdu[0].data = fit_data_full[edge_pad:edge_pad+len1,edge_pad:edge_pad+len2]  / convert2pixel
    elif len(hdu[0].data.shape) == 3:
        hdu[0].data[0,:,:] = fit_data_full[edge_pad:edge_pad+len1,edge_pad:edge_pad+len2]  / convert2pixel
    elif len(hdu[0].data.shape) == 4:
        hdu[0].data[0,0,:,:] = fit_data_full[edge_pad:edge_pad+len1,edge_pad:edge_pad+len2]  / convert2pixel

    hdu.writeto('shapelets_%s_nmax%d_fit.fits' %(save_tag,nmax),overwrite=True)
