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
from sys import exit
import scipy.optimize as opt
from copy import deepcopy
from scipy.signal import convolve2d
from scipy import ndimage
from scipy.signal import fftconvolve
import os

D2R = pi/180.

xmax = 250
n_x = 20001
x_cent = int(floor(n_x / 2))
xrange = linspace(-xmax,xmax,n_x)
xres = xrange[1] - xrange[0]


fileloc = os.path.realpath(__file__)

if fileloc[-1] == 'c':
    fileloc = fileloc.replace('shapelets.pyc', '')
else:
    fileloc = fileloc.replace('shapelets.py', '')



image_shapelet_basis = load('%simage_shapelet_basis.npz' %fileloc)
basis_matrix = image_shapelet_basis['basis_matrix']
gauss_array = image_shapelet_basis['gauss_array']

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    '''Model for a 2D gaussian I got from the internet - flattens it to make
    fitting more straight forward'''
    xo = float(xo)
    yo = float(yo)
    a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
    b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
    c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.flatten()

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

def find_image_centre(ls=None,ms=None,flat_data=None):
    power = 4
    l_cent = sum(flat_data**power*ls) / sum(flat_data**power)
    m_cent = sum(flat_data**power*ms) / sum(flat_data**power)

    resolution = abs(ls[1] - ls[0])
    ##Find the difference between the gridded u coords and the desired u
    l_offs = np_abs(ls - l_cent)
    ##Find out where in the gridded u coords the current u lives;
    ##This is a boolean array of length len(l_offs)
    l_true = l_offs < resolution/2.0
    ##Find the index so we can access the correct entry in the container
    l_ind = where(l_true == True)[0]

    ##Use the numpy abs because it's faster (np_abs)
    m_offs = np_abs(ms - m_cent)
    m_true = m_offs < resolution/2
    m_ind = where(m_true == True)[0]

    ##If the u or v coord sits directly between two grid points,
    ##just choose the first one ##TODO choose smaller offset?
    if len(l_ind) == 0:
        l_true = l_offs <= resolution/2
        l_ind = where(l_true == True)[0]
        #print('here')
        #print(ls.min())
    if len(m_ind) == 0:
        m_true = m_offs <= resolution/2
        m_ind = where(m_true == True)[0]
    l_ind,m_ind = l_ind[0],m_ind[0]

    return l_cent,m_cent,l_ind,m_ind

def find_image_centre_celestial(ras=None,decs=None,flat_data=None,good_inds=None):
    power = 4
    ra_cent = sum(flat_data[good_inds]**power*ras[good_inds]) / sum(flat_data[good_inds]**power)
    dec_cent = sum(flat_data[good_inds]**power*decs[good_inds]) / sum(flat_data[good_inds]**power)

    resolution = abs(ras[1] - ras[0])
    ##Find the difference between the gridded u coords and the desired u
    ra_offs = np_abs(ras - ra_cent)
    ##Find out where in the gridded u coords the current u lives;
    ##This is a boolean array of length len(ra_offs)
    ra_true = ra_offs < resolution/2.0
    ##Find the index so we can access the correct entry in the container
    ra_ind = where(ra_true == True)[0]

    ##Use the numpy abs because it's faster (np_abs)
    dec_offs = np_abs(decs - dec_cent)
    dec_true = dec_offs < resolution/2
    dec_ind = where(dec_true == True)[0]

    ##If the u or v coord sits directly between two grid points,
    ##just choose the first one ##TODO choose smaller offset?
    if len(ra_ind) == 0:
        ra_true = ra_offs <= resolution/2
        ra_ind = where(ra_true == True)[0]
        #print('here')
        #print(ras.min())
    if len(dec_ind) == 0:
        dec_true = dec_offs <= resolution/2
        dec_ind = where(dec_true == True)[0]
    ra_ind,dec_ind = ra_ind[0],dec_ind[0]

    return ra_cent,dec_cent,ra_ind,dec_ind

def gen_shape_basis_direct(n1=None,n2=None,xrot=None,yrot=None,b1=None,b2=None,convolve_kern=False,shape=False):
    '''Generates the shapelet basis function for given n1,n2,b1,b2 parameters,
    at the given coords xrot,yrot
    b1,b2 should be in radians'''

    this_xrot = deepcopy(xrot)
    this_yrot = deepcopy(yrot)

    gauss = exp(-0.5*(array(this_xrot)**2+array(this_yrot)**2))

    n1 = int(n1)
    n2 = int(n2)

    norm = 1.0 / (sqrt(2**n1*factorial(n1))*sqrt(2**n2*factorial(n2)))
    norm *= 1.0 / (b1 * b2)

    #This extra sqrt(pi) / 2 makes it all consistent with the RTS
    norm *= sqrt(pi) / 2

    if type(convolve_kern) == ndarray:
        this_xrot.shape = shape
        this_yrot.shape = shape
        gauss.shape = shape

        h1 = eval_hermite(n1,this_xrot)
        h2 = eval_hermite(n2,this_yrot)
        basis = gauss*norm*h1*h2

        basis = fftconvolve(basis, convolve_kern, 'same')
        basis = basis.flatten()

    else:

        h1 = eval_hermite(n1,this_xrot)
        h2 = eval_hermite(n2,this_yrot)
        basis = gauss*norm*h1*h2

        # basis = gauss*h1*h2*sqrt(pi) / ((sqrt(2**n1*factorial(n1))*sqrt(2**n2*factorial(n2)))*b1*b2*2.0)

    return basis

def interp_basis(xrot=None,yrot=None,n1=None,n2=None):
    '''Uses basis lookup tables to generate 2D shapelet basis function for given
    xrot, yrot coords and n1,n2 orders'''
    xpos = xrot / xres + x_cent
    xindex = floor(xpos)
    xlow = basis_matrix[n1,xindex.astype(int)]
    xhigh = basis_matrix[n1,xindex.astype(int)+1]
    x_val = xlow + (xhigh-xlow)*(xpos-xindex)

    ypos = yrot / xres + x_cent
    yindex = floor(ypos)
    ylow = basis_matrix[n2,yindex.astype(int)]
    yhigh = basis_matrix[n2,yindex.astype(int)+1]
    y_val = ylow + (yhigh-ylow)*(ypos-yindex)

    gxpos = xrot / xres + x_cent
    gxindex = floor(gxpos)
    gxlow = gauss_array[gxindex.astype(int)]
    gxhigh = gauss_array[gxindex.astype(int)+1]
    gx_val = gxlow + (gxhigh-gxlow)*(gxpos-gxindex)

    gypos = yrot / xres + x_cent
    gyindex = floor(gypos)
    gylow = gauss_array[gyindex.astype(int)]
    gyhigh = gauss_array[gyindex.astype(int)+1]
    gy_val = gylow + (gyhigh-gylow)*(gypos-gyindex)

    return x_val*y_val*gx_val*gy_val

# @profile
def gen_shape_basis(n1=None,n2=None,xrot=None,yrot=None,b1=None,b2=None,convolve_kern=False,shape=False):
    '''Generates the shapelet basis function for given n1,n2,b1,b2 parameters,
    using lookup tables, at the given coords xrot,yrot
    b1,b2 should be in radians'''

    ##Do a copy incase we are reshaping to do the convolution
    this_xrot = deepcopy(xrot)
    this_yrot = deepcopy(yrot)

    ##Ensure n1,n2 are ints
    n1 = int(n1)
    n2 = int(n2)

    norm = 1.0 / (b1 * b2)
    #This extra sqrt(pi) / 2 makes it all consistent with the RTS
    norm *= sqrt(pi) / 2

    if type(convolve_kern) == ndarray:
        this_xrot.shape = shape
        this_yrot.shape = shape
        basis = interp_basis(xrot=this_xrot,yrot=this_yrot,n1=n1,n2=n2)
        basis = fftconvolve(basis, convolve_kern, 'same')
        basis = basis.flatten()

    else:
        basis = interp_basis(xrot=this_xrot,yrot=this_yrot,n1=n1,n2=n2)

    return basis*norm

def gen_reduced_A_shape_matrix(n1s=None, n2s=None, xrot=None,yrot=None,b1=None,b2=None,convolve_kern=False,shape=False):

    A_shape_basis = zeros((len(xrot),len(n1s)))
    for index,n1 in enumerate(n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=convolve_kern,shape=shape)

    return A_shape_basis

# @profile
def gen_A_shape_matrix(xrot=None,yrot=None,nmax=None,b1=None,b2=None,convolve_kern=False,shape=False):
    n1s = []
    n2s = []

    with errstate(divide='raise',invalid='raise'):
        for n1 in range(nmax+1):
            for n2 in range(nmax-n1+1):
                ##If the norm factor is tiny going to have problems -
                ##skip if we get issues
                # norm = sqrt(pow(2,n1+n2)*pi*b1*b2*factorial(n1)*factorial(n2))
                # norm = sqrt(pow(2,n1+n2)*pow(pi,2)*factorial(n1)*factorial(n2))

                try:
                    norm = 1.0 / (sqrt(2**n1*factorial(n1))*sqrt(2**n2*factorial(n2)))
                    norm *= 1.0 / (b1 * b2)
                    n1s.append(n1)
                    n2s.append(n2)
                #
                except FloatingPointError:
                    print("Skipped n1=%d, n2=%d, b1=%.7f b2=%.7f problem with normalisation factor is too small" %(n1,n2,b1,b2))

                # print("These nums n1=%d, n2=%d, b1=%.7f b2=%.7f" %(n1,n2,b1,b2))
                # print(sqrt(2**n2*factorial(n2)))
                # print(sqrt(2**n1*factorial(n1)))
                # norm = 1.0 / (sqrt(2**n1*factorial(n1))*sqrt(2**n2*factorial(n2)))

    n1s = array(n1s)
    n2s = array(n2s)
    # print('Number of coefficients to fit is', len(n1s))

    A_shape_basis = zeros((len(xrot),len(n1s)))
    for index,n1 in enumerate(n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=convolve_kern,shape=shape)

    return n1s, n2s, A_shape_basis


def linear_solve(flat_data=None,A_shape_basis=None):
    '''Fit the image_data using the given A_shape_basis matrix
    Essentially solving for x in the equation Ax = b where:
    A = A_shape_basis
    b = the image data
    x = coefficients for the basis functions in A

    returns: the fitted coefficients in an array'''

    # flat_data = image_data.flatten()
    flat_data.shape = (len(flat_data),1)

    shape_coeffs,resid,rank,s = linalg.lstsq(A_shape_basis,flat_data,rcond=None)
    return shape_coeffs

def fitted_model(coeffs=None,A_shape_basis=None):
    '''Generates the fitted shapelet model for the given coeffs
    and A_shape_basis'''
    return matrix(A_shape_basis)*matrix(coeffs)

def save_srclist(save_tag=None, nmax=None, n1s=None, n2s=None, fitted_coeffs=None,
    b1=None, b2=None, fitted_model=None, ra_cent=None, dec_cent=None, freq=None,
    pa=0.0, pix_area=None):
    '''Take the fitted parameters and creates and RTS style srclist with them'''

    all_flux = sum(fitted_model)
    print('TOTAL FLUX in convolved model is %.2f' %all_flux)

    ##This scaling removes pixel effects, and sets the model to sum to one -
    ##this way when the RTS creates the model and multiplies by the reported
    ##flux density we get the correct answer
    scale = 1 / (pix_area*all_flux)

    ##Scale to arcmin or deg for the RTS
    major, minor = (b1 / D2R)*60, (b2 / D2R)*60
    pa /= D2R

    outfile = open('srclist_%s.txt' %(save_tag),'w+')
    outfile.write('SOURCE %s %.6f %.6f\n' %(save_tag[:15],ra_cent/15.0,dec_cent))
    outfile.write("FREQ %.2fe+6 %.5f 0 0 0\n" %(freq,all_flux))
    outfile.write("SHAPELET %.8f %.8f %.8f\n" %(pa,major,minor))

    for index,coeff in enumerate(fitted_coeffs):
        outfile.write("COEFF %.1f %.1f %.12f\n" %(n1s[index],n2s[index],coeff * scale))

    outfile.write('ENDSOURCE')
    outfile.close()

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

def radec2xy(ras,decs,pa,b1,b2):

    x = -ras
    y = decs

    angle = -pa

    yrot = x*cos(angle) + -y*sin(angle)
    xrot = x*sin(angle) + y*cos(angle)

    factor = (2*pi) / (sqrt(pi**2 / (2*log(2))))

    xrot *= factor / b1
    yrot *= factor / b2

    return xrot,yrot

def get_fits_info(fitsfile,edge_pad=False,freq=None):

    hdu = fits.open(fitsfile)
    header = hdu[0].header

    if len(hdu[0].data.shape) == 2:
        data = hdu[0].data
    elif len(hdu[0].data.shape) == 3:
        data = hdu[0].data[0,:,:]
    elif len(hdu[0].data.shape) == 4:
        data = hdu[0].data[0,0,:,:]

    zero_index = 0

    if edge_pad:

        ras = (arange(zero_index,(int(header['NAXIS1'])+edge_pad*2+zero_index)) - (int(header['CRPIX1'])+edge_pad))*float(header['CDELT1'])
        decs = (arange(zero_index,(int(header['NAXIS2'])+edge_pad*2+zero_index)) - (int(header['CRPIX2'])+edge_pad))*float(header['CDELT2'])
        pad_image = zeros((int(header['NAXIS1'])+edge_pad*2,int(header['NAXIS2'])+edge_pad*2))
        pad_image[edge_pad:data.shape[0]+edge_pad,edge_pad:data.shape[1]+edge_pad] = data
        data = pad_image

    else:
        ras = (arange(zero_index,int(header['NAXIS1'])+zero_index) - int(header['CRPIX1']))*float(header['CDELT1'])
        decs = (arange(zero_index,int(header['NAXIS2'])+zero_index) - int(header['CRPIX2']))*float(header['CDELT2'])


    ras_mesh,decs_mesh = meshgrid(ras,decs)

    # if edge_pad:
    #
    #     ras = (arange(zero_index,(int(header['NAXIS1'])+edge_pad*2+zero_index)) - (int(header['CRPIX1'])+edge_pad))*float(header['CDELT1'])
    #     decs = (arange(zero_index,(int(header['NAXIS2'])+edge_pad*2+zero_index)) - (int(header['CRPIX2'])+edge_pad))*float(header['CDELT2'])
    #     pad_image = zeros((int(header['NAXIS1'])+edge_pad*2,int(header['NAXIS2'])+edge_pad*2))
    #     pad_image[edge_pad:data.shape[0]+edge_pad,edge_pad:data.shape[1]+edge_pad] = data
    #     data = pad_image
    #
    # else:
    #
    #
    #     wcs = WCS(header)
    #
    #     xrange = arange(header['NAXIS1'])
    #     yrange = arange(header['NAXIS2'])
    #
    #     x_mesh,y_mesh = meshgrid(xrange,yrange)
    #
    #     if len(hdu[0].data.shape) == 2:
    #         ras_mesh,decs_mesh = wcs.all_pix2world(x_mesh,y_mesh,0)
    #     elif len(hdu[0].data.shape) == 3:
    #         ras_mesh,decs_mesh,meh1 = wcs.all_pix2world(x_mesh,y_mesh,0,0)
    #     elif len(hdu[0].data.shape) == 4:
    #         ras_mesh,decs_mesh,meh1,meh2 = wcs.all_pix2world(x_mesh,y_mesh,0,0,0)
    #
    #
    #
    #     ras_mesh -= header['CRVAL1']
    #     decs_mesh -= header['CRVAL2']

    ras,decs = ras_mesh.flatten(),decs_mesh.flatten()
    flat_data = data.flatten()

    ras *= D2R
    decs *= D2R

    ra_cent = float(header['CRVAL1'])
    dec_cent = float(header['CRVAL2'])

    try:
        bmaj = float(header['BMAJ'])
        bmin = float(header['BMIN'])
    except:
        print('No BMAJ,BMIN data in header')
        print('Assuming max baseline is 3.0e3')
        print('Assuming BMAJ,BMIN are equal')
        VELC = 299792458.0
        R2D = 180.0 / pi
        baseline = 3e+3
        reso = ((VELC/(freq*1e+6))/baseline)*R2D*0.5
        bmaj = reso
        bmin = reso
        print('Assuming BMAJ,BMIN = %.3f,%.3f' %(bmaj,bmin))

    solid_beam = (pi*bmaj*bmin) / (4*log(2))
    solid_pixel = abs(float(header['CDELT1'])*float(header['CDELT2']))

    convert2pixel = solid_pixel/solid_beam

    return hdu,data,flat_data,ras,decs,ra_cent,dec_cent,convert2pixel,abs(float(header['CDELT1'])*D2R),float(header['CDELT2'])*D2R

def compress_coeffs(n1s,n2s,coeffs,num_coeffs,xrot,yrot,b1,b2,convolve_kern=False,shape=False):
    sums = []
    for index,n1 in enumerate(n1s):
        val = sum(abs(coeffs[index]*gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=convolve_kern,shape=shape)))
        sums.append(val)

    sums = array(sums)
    order = argsort(sums)[::-1]
    order_high = order[:num_coeffs]
    order_low = order[num_coeffs:]

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.plot(sums[order],label='All')
    ax.plot(sums[order_high],'o',linestyle='none',label='Selected')
    ax.set_xlabel('rank')
    ax.set_ylabel('value')
    fig.savefig('rank_order.png')
    plt.close()

    return n1s[order_high],n2s[order_high],coeffs[order_high],order_high


def find_resids(data=None,fit_data=None):
    '''Just finds the sum of squares of the residuals'''

    this_fit = asarray(fit_data).flatten()
    diffs = (data - this_fit)**2
    return diffs.sum()
