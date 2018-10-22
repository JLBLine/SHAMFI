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
from sys import exit

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
    norm = 1.0 / sqrt(pow(2,n1+n2)*pi*b1*b2*factorial(n1)*factorial(n2))
    # norm = 1.0 / sqrt(pow(2,n1+n2)*pow(pi,2)*factorial(n1)*factorial(n2))

    h1 = eval_hermite(n1,xrot)
    h2 = eval_hermite(n2,yrot)

    return gauss*norm*h1*h2

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

def find_image_centre_celestial(ras=None,decs=None,flat_data=None):
    power = 4
    ra_cent = sum(flat_data**power*ras) / sum(flat_data**power)
    dec_cent = sum(flat_data**power*decs) / sum(flat_data**power)

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

def gen_reduced_A_shape_matrix(n1s=None, n2s=None, xrot=None,yrot=None,b1=None,b2=None):

    A_shape_basis = zeros((len(xrot),len(n1s)))
    for index,n1 in enumerate(n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2)

    return A_shape_basis

def gen_A_shape_matrix(xrot=None,yrot=None,nmax=None,b1=None,b2=None):
    n1s = []
    n2s = []
    for n1 in arange(nmax+1):
        for n2 in range(nmax-n1+1):
            ##If the norm factor is tiny going to have problems -
            ##sky if we get issues
            # norm = sqrt(pow(2,n1+n2)*pi*b1*b2*factorial(n1)*factorial(n2))
            norm = sqrt(pow(2,n1+n2)*pow(pi,2)*factorial(n1)*factorial(n2))

            if norm == 0.0 or isnan(norm) == True:
                print("Skipped n1=%d, n2=%d, normalisation factor is too small" %(n1,n2))
            else:
                n1s.append(n1)
                n2s.append(n2)


    n1s = array(n1s)
    n2s = array(n2s)
    print('Number of coefficients to fit is', len(n1s))

    A_shape_basis = zeros((len(xrot),len(n1s)))
    for index,n1 in enumerate(n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2)

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

def jenny_style_save(save_tag, nmax, n1s, n2s, fitted_coeffs, b1, b2, ra_cent, dec_cent):
    save_info = empty((len(n1s),3))
    save_info[:,0] = n1s
    save_info[:,1] = n2s
    save_info[:,2] = fitted_coeffs[:,0]

    num_coeffs = int(0.5*(nmax+1)*(nmax+2))

    savetxt('%s_%dcoeffs.txt' %(save_tag,num_coeffs),save_info)

    PA = 0.0

    outfile = open('%s_paras.txt' %(save_tag),'w+')
    outfile.write('%.6f\n' %ra_cent)
    outfile.write('%.6f\n' %dec_cent)
    outfile.write('%.6f\n' %((b1 / D2R)*60))
    outfile.write('%.6f\n' %((b2 / D2R)*60))
    outfile.write('%.6f\n' %PA)

    outfile.close()

def save_srclist(save_tag=None, nmax=None, n1s=None, n2s=None, fitted_coeffs=None,
    b1=None, b2=None, fitted_model=None, ra_cent=None, dec_cent=None, freq=None,
    pa=0.0, convert2pixel=None):
    '''Take the fitted parameters and creates and RTS style srclist with them'''


    flux = sum(fitted_model)*convert2pixel
    major, minor = (b1 / D2R)*60, (b2 / D2R)*60
    pa /= D2R

    outfile = open('srclist_%s.txt' %(save_tag),'w+')
    outfile.write('SOURCE %s %.6f %.6f\n' %(save_tag,ra_cent/15.0,dec_cent))
    outfile.write("FREQ %.2fe+6 %.5f 0 0 0\n" %(freq,flux))

    # outfile.write('SOURCE %s 3.3609780000 -37.1509690000\n' %(save_tag))
    # outfile.write("FREQ 1.7000e+08 235.71000 0 0 0\n")

    outfile.write("SHAPELET %.8f %.8f %.8f\n" %(pa,major,minor))

    for index,coeff in enumerate(fitted_coeffs):
    #     if coeff < 0:
    #         pass
    #     else:
        outfile.write("COEFF %.1f %.1f %.8f\n" %(n1s[index],n2s[index],coeff))

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

# def import_fits(filename):
#     # open file
#     obj_info = zeros((2,4))
#     obj_data = fits.open(filename)
#     fits_info = obj_data[0].header
#     alldata = obj_data[0].data
#
#     # relevant coordinates from header
#     ra = radians(fits_info['CRVAL1'])
#     dec = radians(fits_info['CRVAL2'])
#     ra_res = radians(fits_info['CDELT1'])
#     dec_res = radians(fits_info['CDELT2'])
#     ra_pxl = fits_info['CRPIX1']
#     dec_pxl = fits_info['CRPIX2']
#
#     ra_side = fits_info['NAXIS1']
#     dec_side = fits_info['NAXIS2']
#
#     data = alldata[0,0,:,:]
#
#     # packaging row 0 = ra stuff, row 1 = dec stuff
#     obj_info[0,:] = [ra, ra_res, ra_pxl, ra_side]
#     obj_info[1,:] = [dec, dec_res, dec_pxl, dec_side]
#
#     obj_data.close()
#     return (obj_info, data)
#
# #######################################################################
# ## polish_data: responsible for resizing the data, transforming it into
# # columns and creating meaningful axii.
# #=====================================================================#
#
# def polish_data(sinfo, in_data, sizing, ang_res):
#
#     fudge = 100.
#     # assumes square dataset
#     side = max(sinfo[0,3], sinfo[1,3])-1
#     obj_size_pxls = int(round(sizing/ang_res)*fudge)
#
#     if obj_size_pxls > side:
#         nside = side
#     else:
#         nside = obj_size_pxls
#
#     midpix_ra = sinfo[0,2]
#     midpix_dec = sinfo[1,2]
#
#     # ensuring we stay within the bounds of the array
#     if (sinfo[0,2]+nside/2) > side:
#         nside = 2*(side-midpix_ra)
#     if (sinfo[0,2]-nside/2) < 0:
#         nside = 2*midpix_ra
#
#     if (sinfo[1,2]+nside/2) > side:
#         nside = 2*(side-midpix_dec)
#     if (sinfo[1,2]-nside/2) < 0:
#         nside = 2*midpix_dec
#
#      # Initialise arrays
#     nside = int(nside)
#     midpix = int(nside/2)
#     num_pxl = int(nside*nside)
#
#     coords = zeros((num_pxl,2))
#     rowaxis = zeros((nside,1))
#     ra = zeros((nside,1))
#     dec = zeros((nside,1))
#     sky_coords = zeros((nside,2))
#     out_data = zeros((nside, nside))
#     col_data = zeros((num_pxl,1))
#
#     for i in range(0,nside):
#         rowaxis[i] = (i-midpix+1)*ang_res
#
#     off1 = int(sinfo[1,2]-midpix)
#     off0 = int(sinfo[0,2]-midpix)
#
#     k=-1
#     for i in range(0,nside):
#         for j in range(0,nside):
#               k+=1
#               coords[k,0]=rowaxis[i]
#               coords[k,1]=rowaxis[j]
#               out_data[i,j] = in_data[i+off1, j+off0]
#               col_data[k,0] = out_data[i,j]
#
#     dec = degrees(rowaxis) + sinfo[1,0]*ones((nside,1))
#     ra = -1*degrees(rowaxis) - sinfo[0,0]*ones((nside,1))
#     sky_coords = concatenate((ra, dec), axis=1)
#
#     return coords, sky_coords, out_data, col_data
#
# def cov_fit(coords, data):
#     S = sum(sum(data))
#     nside = sqrt(coords.shape[0])
#     nside = int(nside)
#     npix = nside*nside
#     offsets = (0.0,0.0)
#     addmat = zeros((npix,2))
#
#     x = 0
#     y = 0
#     Sxx = 0
#     Sxy = 0
#     Syy = 0
#
#     k=-1
#     for i in range(0,nside):
#         for j in range(0,nside):
#             k += 1
#             x += data[i,j]*coords[k,0]
#             y += data[i,j]*coords[k,1]
#
#     x0 = (x/S)
#     y0 = (y/S)
#
#     offsets = (x0, y0)
#
#     coords[:,0]-= x0
#     coords[:,1]-= y0
#
#     k=-1
#     for i in range(0,nside):
#         for j in range(0,nside):
#             k=k+1
#             Sxx = Sxx + data[i,j]*coords[k,0]*coords[k,0]
#             Sxy = Sxy + data[i,j]*coords[k,0]*coords[k,1]
#             Syy = Syy + data[i,j]*coords[k,1]*coords[k,1]
#
#     a11 = Sxx/S
#     a12 = Sxy/S
#     a22 = Syy/S
#
#     C = array([[a11, a12], [a12, a22]])
#
#     (eigenval, eigenvect) = linalg.eig(C)
#
#     minor = sqrt(eigenval[0])
#     major = sqrt(eigenval[1])
#
#     PA = atan2(eigenvect[1,1],eigenvect[0,1])
#
#     return major, minor, PA, offsets

def minco(flat_data,b1,b2,n1s,n2s,xrot,yrot,coeffs):

    mse = 10e-6
    factor = 10
    resid_nmse = 100
    target=mse/factor
    i=0
    flat_data.shape = len(flat_data)
    model = zeros(len(flat_data))
    # print(xrot.shape,yrot.shape)

    chi_sq = 0

    while (resid_nmse > target and i < len(n1s)):
        model += coeffs[i]*gen_shape_basis(n1=n1s[i],n2=n2s[i],xrot=xrot,yrot=yrot,b1=b1,b2=b2)

        # A_matrix = gen_reduced_A_shape_matrix(n1s=[n1s[i]],n2s=[n2s[i]],xrot=xrot,yrot=yrot,b1=b1,b2=b2)
        # coeff = array(coeffs[i])
        # # coeff.shape = (1,1)
        # model += fitted_model(coeffs=coeff,A_shape_basis=A_matrix)


        # print('Why dis crash')
        # print(model.shape,flat_data.shape)

        # model.shape = (512,512)
        # plt.imshow(model)
        # plt.show()
        # plt.close()

        model.shape = len(flat_data)

        # z = sum(flat_data - model)
        # mse += z*z
        #
        # resid_nmse = mse / (sum(flat_data)**2)

        chi_sq = sum(flat_data - model)**2

        print(n1s[i],n2s[i],chi_sq)
        i+=1

    tcoeffs = i+1
    print(tcoeffs)
    return tcoeffs


def radec2xy(ras,decs,pa,b1,b2):
    x = -ras
    y = decs

    angle = -pa

    yrot = x*cos(angle) + -y*sin(angle)
    xrot = x*sin(angle) + y*cos(angle)

    xrot /= b1
    yrot /= b2

    return xrot,yrot

def get_fits_info(fitsfile,edge_pad=False):

    hdu = fits.open(fitsfile)
    header = hdu[0].header

    if len(hdu[0].data.shape) == 2:
        data = hdu[0].data
    elif len(hdu[0].data.shape) == 3:
        data = hdu[0].data[0,:,:]
    elif len(hdu[0].data.shape) == 4:
        data = hdu[0].data[0,0,:,:]

    zero_index = 1

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
        exit('No BMAJ,BMIN data in header - exiting')

    solid_beam = (pi*bmaj*bmin) / (4*log(2))
    solid_pixel = abs(float(header['CDELT1'])*float(header['CDELT2']))

    convert2pixel = solid_pixel/solid_beam

    return data,flat_data,ras,decs,ra_cent,dec_cent,convert2pixel

def compress_coeffs(n1s,n2s,coeffs,num_coeffs,xrot,yrot,b1,b2):
    sums = []
    for index,n1 in enumerate(n1s):
        val = coeffs[index]*sum(abs(gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2)))
        sums.append(sum)

    sums = array(sums)


    order = argsort(sums)[::-1][:num_coeffs]
    print(order)
    return n1s[order],n2s[order],coeffs[order]
