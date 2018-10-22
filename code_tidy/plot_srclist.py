#!/usr/bin/env python
from __future__ import print_function,division
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import factorial,eval_hermite
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
import math as m
from scipy import ndimage
from fit_shapelets import *

D2R = pi/180.

def add_colourbar(fig=None,ax=None,im=None,label=False,top=False):
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


def gen_shape_basis(n1=None,n2=None,xrot=None,yrot=None,b1=None,b2=None):
    '''Generates the shapelet basis function for given n1,n2,b1,b2 parameters,
    at the given coords xrot,yrot
    b1,b2,xrot,yrot should be in radians'''

    ##TODO at some point fold in a PA rotation

    gauss = exp(-0.5*(array(xrot)**2+array(yrot)**2))
    # norm = 1.0 / sqrt(pow(2,n1+n2)*pi*b1*b2*factorial(n1)*factorial(n2))
    # norm = 1.0 / sqrt(pow(2,n1+n2)*pow(pi,2)*factorial(n1)*factorial(n2))
    norm = 1.0 / sqrt(pow(2,n1+n2)*pi*b1*b2*factorial(n1)*factorial(n2))

    h1 = eval_hermite(n1,xrot)
    h2 = eval_hermite(n2,yrot)

    return gauss*norm*h1*h2



def gen_A_shape_matrix(n1s=None, n2s=None, xrot=None,yrot=None,b1=None,b2=None):

    A_shape_basis = zeros((len(xrot),len(n1s)))
    for index,n1 in enumerate(n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2)

    return A_shape_basis

def fitted_model(coeffs=None,A_shape_basis=None):
    '''Generates the fitted shapelet model for the given coeffs
    and A_shape_basis'''
    return matrix(A_shape_basis)*matrix(coeffs)

def get_lm(ra,ra0,dec,dec0):
    '''Calculate l,m for a given phase centre ra0,dec0 and sky point ra,dec
    Enter angles in radians'''

    ##RTS way of doing it
    cdec0 = cos(dec0)
    sdec0 = sin(dec0)
    cdec = cos(dec)
    sdec = sin(dec)
    cdra = cos(ra-ra0)
    sdra = sin(ra-ra0)
    l = cdec*sdra
    m = sdec*cdec0 - cdec*sdec0*cdra
    n = sdec*sdec0 + cdec*cdec0*cdra

    return l,m,n


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="A script to fit a shapelet model consistent with the RTS")

    parser.add_argument('--srclist',
                        help='srclist containing coeffs to plot')

    parser.add_argument('--plot_lims', default=False,
                        help='Flux limits for the plot - enter as vmin,vmax. Default is min(image),max(image)')

    args = parser.parse_args()

    lines = open(args.srclist,'r').read().split('\n')

    n1s = []
    n2s = []
    coeffs = []

    for line in lines:
        if 'SHAPELET' in line:
            meh,pa,b1,b2 = line.split()
            b1 = (float(b1) / 60.0)*D2R
            b2 = (float(b2) / 60.0)*D2R
            pa = float(pa)*D2R
        elif 'COEFF' in line:
            meh,n1,n2,coeff = line.split()
            n1s.append(int(float(n1)))
            n2s.append(int(float(n2)))
            coeffs.append(float(coeff))
        elif 'SOURCE' in line and 'ENDSOURCE' not in line:
            men,meh1,ra_cent,dec_cent = line.split()
            ra_cent = (float(ra_cent)*15.0)*D2R
            dec_cent = float(dec_cent)*D2R

    n1s = array(n1s)
    n2s = array(n2s)
    coeffs = array(coeffs)

    # compress_coeffs(n1s,n2s,coeffs)

    ext = 1.5
    ras = linspace(ext,-ext,400)
    decs = linspace(-ext,ext,400)

    ras_mesh,decs_mesh = meshgrid(ras,decs)
    ras,decs = ras_mesh.flatten()*D2R,decs_mesh.flatten()*D2R

    # data,flat_data,ras,decs,ra_cent,dec_cent,convert2pixel = get_fits_info('/home/jline/Documents/Shapelets/Python_code/Fits_files/FnxA.fits')
    xrot,yrot = radec2xy(ras,decs,pa,b1,b2)

    A_shape_basis = gen_A_shape_matrix(n1s=n1s,n2s=n2s,xrot=xrot,yrot=yrot,b1=b1,b2=b2)

    coeffs.shape = (len(coeffs),1)
    fit_data = fitted_model(coeffs=coeffs,A_shape_basis=A_shape_basis)


    print(sum(fit_data))



    fig = plt.figure(figsize=(7,7))

    ax1 = fig.add_subplot(111)
    try:
        fit_data.shape = data.shape
    except:
        fit_data.shape = ras_mesh.shape
    factor = 0.0001
    fit_data[where(abs(fit_data) < fit_data.max()*factor)] = NaN
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='grey')
    if args.plot_lims:
        vmin,vmax = map(float,args.plot_lims.split(','))
        im1 = ax1.imshow(fit_data,vmin=vmin,vmax=vmax,origin='lower')
    else:
        im1 = ax1.imshow(fit_data,origin='lower')

    # ax1.contour(dataset0,colors='w',alpha=0.3)
    add_colourbar(fig=fig,im=im1,ax=ax1)
    ax1.set_title('Model in %s' %args.srclist)

    ax1.axvline(floor(fit_data.shape[1]/2),color='w',alpha=0.3,lw=0.5)
    ax1.axhline(floor(fit_data.shape[0]/2),color='w',alpha=0.3,lw=0.5)


    fig.savefig('%s_large.png' %(args.srclist.split('/')[-1][:-4]), bbox_inches='tight')

    # hdu[0].data[0,0,:,:] = fit_data
    # hdu.writeto('%s.fits' %(args.srclist[:-4]),overwrite=True)
