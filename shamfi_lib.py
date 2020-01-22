from __future__ import print_function,division
from numpy import *
import matplotlib
##useful when using a super cluster to specify Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import factorial,eval_hermite
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from numpy import abs as np_abs
from astropy.wcs import WCS
from sys import exit
import scipy.optimize as opt
from copy import deepcopy
from scipy.signal import fftconvolve
import os
from astropy.modeling.models import Gaussian2D
from progressbar import progressbar
from subprocess import check_output

##Convert degress to radians
D2R = pi/180.
##Convert radians to degrees
R2D = 180./pi

##Max x value of stored basis functions
xmax = 250
##Number of samples in stored basis functions
n_x = 20001
##More basis function values
x_cent = int(floor(n_x / 2))
xrange = linspace(-xmax,xmax,n_x)
xres = xrange[1] - xrange[0]

##convert between FWHM and std dev
factor = 2. * sqrt(2.*log(2.))

##Find where this file is so we can find the basis functions
fileloc = os.path.realpath(__file__)
if fileloc[-1] == 'c':
    fileloc = fileloc.replace('shamfi_lib.pyc', '')
else:
    fileloc = fileloc.replace('shamfi_lib.py', '')

##Import the basis functions
image_shapelet_basis = load('%simage_shapelet_basis.npz' %fileloc)
basis_matrix = image_shapelet_basis['basis_matrix']
gauss_array = image_shapelet_basis['gauss_array']


def get_gitlabel():
    '''Find out what git commit we are working with'''
    ##Find out where the git repo is, cd in and grab the git label
    ##TODO do this in a better way
    fileloc = os.path.realpath(__file__)
    cwd = os.getcwd()
    os.chdir(('/').join(fileloc.split('/')[:-1]))
    gitlabel = check_output(["git", "describe", "--always"],universal_newlines=True).strip()
    ##Get back to where we were before
    os.chdir(cwd)

    return gitlabel

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
    '''Model for a 2D gaussian  - flattens it to make
    fitting more straight forward'''

    x,y = xy

    xo = float(xo)
    yo = float(yo)
    a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
    b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
    c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.flatten()

def add_colourbar(fig=None,ax=None,im=None,label=False,top=False):
    '''Adds a colourbar in a nice way to a sublot'''
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

def find_image_centre_celestial(ras=None,decs=None,flat_data=None,pixel_inds_to_use=None,data=None):
    '''Find the flux-weighted central position of an image'''
    power = 4
    ra_cent = sum(flat_data[pixel_inds_to_use]**power*ras[pixel_inds_to_use]) / sum(flat_data[pixel_inds_to_use]**power)
    dec_cent = sum(flat_data[pixel_inds_to_use]**power*decs[pixel_inds_to_use]) / sum(flat_data[pixel_inds_to_use]**power)

    resolution = abs(ras[1] - ras[0])
    ##Find the difference between the gridded ra coords and the desired ra_cent
    ra_offs = np_abs(ras - ra_cent)
    ##Find out where in the gridded ra coords the current ra_cent lives;
    ##This is a boolean array of length len(ra_offs)
    ra_true = ra_offs < resolution/2.0
    ##Find the index so we can access the correct entry in the container
    ra_ind = where(ra_true == True)[0]

    ##Use the numpy abs because it's faster (np_abs)
    dec_offs = np_abs(decs - dec_cent)
    dec_true = dec_offs < resolution/2
    dec_ind = where(dec_true == True)[0]

    ##If ra_ind,dec_ind coord sits directly between two grid points,
    ##just choose the first one
    if len(ra_ind) == 0:
        ra_true = ra_offs <= resolution/2
        ra_ind = where(ra_true == True)[0]
    if len(dec_ind) == 0:
        dec_true = dec_offs <= resolution/2
        dec_ind = where(dec_true == True)[0]
    ra_ind,dec_ind = ra_ind[0],dec_ind[0]

    dec_ind = floor(dec_ind / data.shape[0])
    print('Centre of flux pixel found as x,y',ra_ind,dec_ind)

    ra_mesh = deepcopy(ras)
    ra_mesh.shape = data.shape

    dec_mesh = deepcopy(decs)
    dec_mesh.shape = data.shape

    ra_range = ra_mesh[0,:]
    dec_range = dec_mesh[:,0]

    return ra_ind,dec_ind,ra_mesh,dec_mesh,ra_range,dec_range

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
    '''Setup the A matrix used in the linear least squares fit of the basis functions for just n1s,n2s'''
    A_shape_basis = zeros((len(xrot),len(n1s)))
    for index,n1 in enumerate(n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=convolve_kern,shape=shape)

    return A_shape_basis

def gen_A_shape_matrix(xrot=None,yrot=None,nmax=None,b1=None,b2=None,convolve_kern=False,shape=False):
    '''Setup the A matrix used in the linear least squares fit of the basis functions. Works out all
    valid n1,n2 combinations up to nmax'''
    n1s = []
    n2s = []

    with errstate(divide='raise',invalid='raise'):
        for n1 in range(nmax+1):
            for n2 in range(nmax-n1+1):
                try:
                    norm = 1.0 / (sqrt(2**n1*factorial(n1))*sqrt(2**n2*factorial(n2)))
                    norm *= 1.0 / (b1 * b2)
                    n1s.append(n1)
                    n2s.append(n2)
                #
                except FloatingPointError:
                    print("Skipped n1=%d, n2=%d, b1=%.7f b2=%.7f problem with normalisation factor is too small" %(n1,n2,b1,b2))

    n1s = array(n1s)
    n2s = array(n2s)

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

    ##lstsq is real picky about the data shape
    flat_data.shape = (len(flat_data),1)

    shape_coeffs,resid,rank,s = linalg.lstsq(A_shape_basis,flat_data,rcond=None)
    return shape_coeffs

def fitted_model(coeffs=None,A_shape_basis=None):
    '''Generates the fitted shapelet model for the given coeffs
    and A_shape_basis'''
    return matrix(A_shape_basis)*matrix(coeffs)

def save_srclist(save_tag=None, nmax=None, n1s=None, n2s=None, fitted_coeffs=None,
    b1=None, b2=None, fitted_model=None, ra_cent=None, dec_cent=None, freq=None,
    pa=0.0, pix_area=None, gitlabel=None, rts_srclist=True):
    '''Take the fitted parameters and creates an RTS/WODEN style srclist with them'''

    all_flux = sum(fitted_model)
    print('TOTAL FLUX in convolved model is %.2f' %all_flux)

    ##This scaling removes pixel effects, and sets the model to sum to one -
    ##this way when the RTS creates the model and multiplies by the reported
    ##flux density we get the correct answer
    scale = 1 / (pix_area*all_flux)

    ##Scale to arcmin or deg
    major, minor = (b1 / D2R)*60, (b2 / D2R)*60
    pa /= D2R

    if rts_srclist:
        outfile = open('srclist-rts_%s.txt' %(save_tag),'w+')
    else:
        outfile = open('srclist-woden_%s.txt' %(save_tag),'w+')


    outfile.write('##Created with SHAMFI git label %s\n' %gitlabel)

    if rts_srclist:
        outfile.write('SOURCE %s %.6f %.6f\n' %(save_tag[:16],ra_cent/15.0,dec_cent))
        outfile.write("FREQ %.5e %.5f 0 0 0\n" %(freq,all_flux))
        outfile.write("SHAPELET2 %.8f %.8f %.8f\n" %(pa,major,minor))

        for index,coeff in enumerate(fitted_coeffs):
            outfile.write("COEFF %.1f %.1f %.12f\n" %(n1s[index],n2s[index],coeff * scale))

        outfile.write('ENDSOURCE\n')

    else:
        outfile.write('SOURCE %s P 0 G 0 S 1 %d\n' %(save_tag,len(fitted_coeffs)))
        outfile.write('COMPONENT SHAPELET %.6f %.6f\n' %(ra_cent/15.0,dec_cent))
        outfile.write("FREQ %.5e %.5f 0 0 0\n" %(freq,all_flux))
        outfile.write("SPARAMS %.8f %.8f %.8f\n" %(pa,major,minor))

        for index,coeff in enumerate(fitted_coeffs):
            outfile.write("SCOEFF %.1f %.1f %.12f\n" %(n1s[index],n2s[index],coeff * scale))
        outfile.write('ENDCOMPONENT\n')
        outfile.write('ENDSOURCE\n')

    outfile.close()

def radec2xy(ras,decs,pa,b1,b2):
    '''Transforms the RA/DEC coords system into the shapelet x/y system'''

    ##RA increases in opposite direction to x
    x = -ras
    y = decs

    ##Rotation is east from north, (positive RA is negative x)
    angle = -pa

    yrot = x*cos(angle) + -y*sin(angle)
    xrot = x*sin(angle) + y*cos(angle)

    factor = (2*pi) / (sqrt(pi**2 / (2*log(2))))

    ##Apply conversion into stdev from FWHM and beta params
    xrot *= factor / b1
    yrot *= factor / b2

    return xrot,yrot

def get_conert2pixel(header):
    '''Takes a FITS header and gets required info to calculate a conversion
    from Jy/beam to Jy/pixel'''

    bmaj = float(header['BMAJ'])
    bmin = float(header['BMIN'])
    solid_beam = (pi*bmaj*bmin) / (4*log(2))
    solid_pixel = abs(float(header['CDELT1'])*float(header['CDELT2']))
    convert2pixel = solid_pixel/solid_beam

    return convert2pixel

def get_frequency(freq,header):
    '''Either attempts to find the keyword FREQ from a FITS header and the
    associated value, or it just formates the freq arg'''
    ##If freq provided by user, use that value
    ##If not, try and find it from the header of the FITS
    if freq == 'from_FITS':
        ctypes = header['CTYPE*']
        for ctype in ctypes:
            if header[ctype] == 'FREQ':
                freq = float(header['CRVAL%d' %(int(ctype[-1]))])
    else:
        freq = float(freq)

    return freq

def get_fits_info(fitsfile,edge_pad=False,freq=None):
    '''Open up the fitsfile and grab a bunch o information'''

    R2D = 180.0 / pi
    with fits.open(fitsfile) as hdu:
        header = hdu[0].header

        if len(hdu[0].data.shape) == 2:
            data = hdu[0].data
        elif len(hdu[0].data.shape) == 3:
            data = hdu[0].data[0,:,:]
        elif len(hdu[0].data.shape) == 4:
            data = hdu[0].data[0,0,:,:]

        zero_index = 0

        ##If we need to add zero-padding around the image, do this here
        if edge_pad:

            ras = (arange(zero_index,(int(header['NAXIS1'])+edge_pad*2+zero_index)) - (int(header['CRPIX1'])+edge_pad))*float(header['CDELT1'])
            decs = (arange(zero_index,(int(header['NAXIS2'])+edge_pad*2+zero_index)) - (int(header['CRPIX2'])+edge_pad))*float(header['CDELT2'])
            pad_image = zeros((int(header['NAXIS1'])+edge_pad*2,int(header['NAXIS2'])+edge_pad*2))
            pad_image[edge_pad:data.shape[0]+edge_pad,edge_pad:data.shape[1]+edge_pad] = data
            data = pad_image

        else:
            ras = (arange(zero_index,int(header['NAXIS1'])+zero_index) - int(header['CRPIX1']))*float(header['CDELT1'])
            decs = (arange(zero_index,int(header['NAXIS2'])+zero_index) - int(header['CRPIX2']))*float(header['CDELT2'])

        ##Get the ra,dec range for all pixels
        ras_mesh,decs_mesh = meshgrid(ras,decs)

        ras,decs = ras_mesh.flatten(),decs_mesh.flatten()
        flat_data = data.flatten()

        ras *= D2R
        decs *= D2R

        ##Gran the frequency from the header if possible
        freq = get_frequency(freq,header)

        ##Check for restoring beam info so we can convolve our basis functions
        print('Freqeuncy is', freq)
        try:
            rest_bmaj = float(header['BMAJ'])
            rest_bmin = float(header['BMIN'])
            rest_pa = header['BPA'] * D2R
        except:
            ##If fail, assume phase 1 MWA
            print('No BMAJ,BMIN data in header')
            print('Assuming max baseline is 3.0e3')
            print('Assuming BMAJ,BMIN are equal')
            VELC = 299792458.0
            baseline = 3e+3
            reso = ((VELC/(freq*1e+6))/baseline)*R2D*0.5
            rest_bmaj = reso
            rest_bmin = reso
            rest_pa = 0.0
            print('Assuming BMAJ,BMIN = %.3f,%.3f' %(bmaj,bmin))

        solid_beam = (pi*rest_bmaj*rest_bmin) / (4*log(2))
        solid_pixel = abs(float(header['CDELT1'])*float(header['CDELT2']))

        convert2pixel = solid_pixel/solid_beam

        ra_reso = abs(float(header['CDELT1']))
        dec_reso = float(header['CDELT2'])

        len1 = hdu[0].header['NAXIS1']
        len2 = hdu[0].header['NAXIS2']

        wcs = WCS(hdu[0].header)
        dims = len(hdu[0].data.shape)

    return data,flat_data,ras,decs,convert2pixel,ra_reso,dec_reso,freq,len1,len2,wcs,dims,rest_bmaj,rest_bmin,rest_pa

def compress_coeffs(n1s,n2s,coeffs,num_coeffs,xrot,yrot,b1,b2,convolve_kern=False,shape=False):
    '''Go through all basis functions defined by n1,n2,b1,b2, generate them, multiply by
    the fitted coeffs, and sum the absolute values to rank them by flux'''
    sums = []
    ##For each n1,n2, generate basis function, take absolute, sum, and store in sums
    for index,n1 in enumerate(n1s):
        val = sum(abs(coeffs[index]*gen_shape_basis(n1=n1,n2=n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=convolve_kern,shape=shape)))
        sums.append(val)

    ##Sort them, find all coeffs up to num_coeffs
    sums = array(sums)
    order = argsort(sums)[::-1]
    order_high = order[:num_coeffs]
    order_low = order[num_coeffs:]

    return n1s[order_high],n2s[order_high],coeffs[order_high],order_high


def find_resids(data=None,fit_data=None):
    '''Just finds the sum of squares of the residuals
    Stops problematic memory errors with matrix algebra class'''

    this_fit = asarray(fit_data).flatten()
    this_data = asarray(data).flatten()
    # print('WTF',data.shape,fit_data.shape)

    diffs = (this_data - this_fit)**2
    return diffs.sum()

def plot_grid_search(matrix_plot,num_beta_points,b1_grid,b2_grid,save_tag):
    '''Plot a matrix of residuals found when fitting for b1,b2'''
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(matrix_plot)

    ax.set_xticks(arange(num_beta_points))
    ax.set_yticks(arange(num_beta_points))

    labelsy = ['%.2f' %b for b in (b1_grid/D2R)*60.0]
    labelsx = ['%.2f' %b for b in (b2_grid/D2R)*60.0]

    ax.set_yticklabels(labelsy)
    ax.set_xticklabels(labelsx)

    ax.set_xlabel('b2 (arcmins)')
    ax.set_ylabel('b1 (arcmins)')

    add_colourbar(fig=fig,im=im,ax=ax)

    ax.contour(matrix_plot,colors='w',alpha=0.4)

    fig.savefig('grid-fit_matrix_%s.png' %save_tag, bbox_inches='tight')
    plt.close()

def plot_gaussian_fit(ra_mesh, dec_mesh, popt, data, save_tag):
    '''Plot a contour of a gaussian fit over an image'''
    mask = twoD_Gaussian((ra_mesh, dec_mesh), *popt)
    mask.shape = ra_mesh.shape

    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)

    im1 = ax1.imshow(data,origin='lower')
    ax1.contour(mask,colors='r',alpha=0.3)

    add_colourbar(ax=ax1,im=im1,fig=fig)

    fig.savefig('pa_fit_%s.png' %save_tag ,bbox_inches='tight')

def find_good_pixels(args,edge_pad,flat_data,x_len):
    '''Uses the specified arguments to come up with an array of pixel indexes
    to fit'''
    ##If a box is specified, limit pixels to within that box
    if args.fit_box:
        pixel_inds_to_use = []
        low_x,high_x,low_y,high_y = array(map(int,args.fit_box.split(','))) + edge_pad
        for y in range(low_y,high_y+1):
            for x in range(low_x,high_x+1):
                pixel_inds_to_use.append(y*x_len + x)

        pixel_inds_to_use = array(pixel_inds_to_use)
        print('Will fit box defined by low_x,high_x,low_y,high_y: ',low_x,high_x,low_y,high_y)

    else:
        ##If nothing declared, just use all the pixels
        if not args.exclude_box:
            pixel_inds_to_use = arange(len(flat_data))
            print('Will fit using all pixels in image')

        ##Otherwise, use the defined boxes in --exclude_box to flag pixels to
        ##avoid
        else:
            try:
                avoid_inds = []
                for box in args.exclude_box:
                    low_x,high_x,low_y,high_y = array(box.split(','),dtype=int) + edge_pad
                    for y in range(low_y,high_y+1):
                        for x in range(low_x,high_x+1):
                            avoid_inds.append(y*x_len + x)

                pixel_inds_to_use = arange(len(flat_data))
                pixel_inds_to_use = setxor1d(pixel_inds_to_use,avoid_inds)
                print('Will fit avoiding boxes defined in --exclude_box')
            except:
                pixel_inds_to_use = arange(len(flat_data))
                print('Failed to convert --exclude_box into something \
                      sensible. Will fit using all pixels in image')

    return pixel_inds_to_use

def set_central_pixel_to_zero(popt,ras,decs,ra_range,dec_range,args,edge_pad,dims,wcs):
    '''Using the central position found when fitting a gaussian (popt) takes
    the ra,dec coord system and sets x0,y0=0,0'''
    x0 = popt[1]
    y0 = popt[2]

    ra_offs = np_abs(ra_range - x0)
    dec_offs = np_abs(dec_range - y0)

    ra_ind = where(ra_offs < abs(ra_range[1] - ra_range[0])/2.0)[0][0]
    dec_ind = where(dec_offs < abs(dec_range[1] - dec_range[0])/2.0)[0][0]

    ra_cent_off = ra_range[ra_ind]
    dec_cent_off = dec_range[dec_ind]

    if dims == 2:
        ra_cent,dec_cent = wcs.wcs_pix2world(ra_ind-edge_pad,dec_ind-edge_pad,0)
    elif dims == 3:
        ra_cent,dec_cent = wcs.wcs_pix2world(ra_ind-edge_pad,dec_ind-edge_pad,0,0)
    elif dims == 4:
        ra_cent,dec_cent,_,_ = wcs.wcs_pix2world(ra_ind-edge_pad,dec_ind-edge_pad,0,0,0)

    ras -= ra_cent_off
    decs -= dec_cent_off

    return ra_cent, dec_cent, ras, decs

def save_output_FITS(fitsfile,save_data,data_shape,save_tag,nmax,edge_pad,
                     len1,len2,convert2pixel,gitlabel):
    '''Saves the model into a FITS file, taking into account any edge padding
    that happened, and converting back into Jy/beam'''

    save_data.shape = data_shape

    with fits.open(fitsfile) as hdu:
        if len(hdu[0].data.shape) == 2:
            hdu[0].data = save_data[edge_pad:edge_pad+len1,edge_pad:edge_pad+len2]  / convert2pixel
        elif len(hdu[0].data.shape) == 3:
            hdu[0].data[0,:,:] = save_data[edge_pad:edge_pad+len1,edge_pad:edge_pad+len2]  / convert2pixel
        elif len(hdu[0].data.shape) == 4:
            hdu[0].data[0,0,:,:] = save_data[edge_pad:edge_pad+len1,edge_pad:edge_pad+len2]  / convert2pixel

        hdu[0].header['SHAMFIv'] = gitlabel

        hdu.writeto('shamfi_%s_nmax%d_fit.fits' %(save_tag,nmax),overwrite=True)



def create_restoring_kernel(rest_bmaj,rest_bmin,rest_pa,ra_reso,dec_reso):
    factor = 2. * sqrt(2.*log(2.))
    x_stddev = rest_bmaj / (factor*ra_reso)
    y_stddev = rest_bmin / (factor*dec_reso)

    rest_gauss_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev, y_stddev=y_stddev,theta=pi/2 + rest_pa)

    xrange = arange(-25,26)
    yrange = arange(-25,26)

    x_mesh, y_mesh = meshgrid(xrange,yrange)
    rest_gauss_kern = rest_gauss_func(x_mesh,y_mesh)
    rest_gauss_kern /= rest_gauss_kern.sum()

    return rest_gauss_kern

def do_subplot(fig,ax,data,label,vmin,vmax):
    if vmin:
        ax.imshow(data,origin='lower',vmin=vmin,vmax=vmax)
    else:
        im = ax.imshow(data,origin='lower')
    add_colourbar(fig=fig,im=im,ax=ax)
    ax.set_title(label)

def plot_full_fit(args, fit_data, flat_data, data_shape, pixel_inds_to_use,
                  save_tag, nmax, popt, pa, b1, b2, rest_gauss_kern, fitted_coeffs):

    # A_shape_basis_no_conv = gen_reduced_A_shape_matrix(n1s=n1s,n2s=n2s,xrot=xrot,yrot=yrot,b1=b1,b2=b2)
    # fitted_coeffs.shape = (len(fitted_coeffs),1)
    # fit_data_no_conv = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis_no_conv)#
    fig = plt.figure(figsize=(10,8))
    if args.plot_edge_pad:

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
    else:
    # fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
    #
    if args.plot_lims:
        vmin,vmax = map(float,args.plot_lims.split(','))

    else:
        vmin,vmax = False, False


    bad_inds = setdiff1d(arange(len(flat_data)),pixel_inds_to_use)
    flat_data[bad_inds] = nan
    #
    flat_data.shape = data_shape

    do_subplot(fig,ax1,flat_data,'Data',vmin,vmax)
    do_subplot(fig,ax2,fit_data,'Fit (convolved with\nrestoring beam)',vmin,vmax)
    do_subplot(fig,ax3,flat_data - fit_data,'Data - Fit',vmin,vmax)
    #
    if args.plot_edge_pad:
        print('--------------------------------------')
        print('Generating model for edge padded image')

        edge_pad = int(fit_data.shape[0] / 5)
        data,flat_data,ras,decs,convert2pixel,ra_reso,dec_reso,freq,len1,len2,wcs,dims,rest_bmaj,rest_bmin,rest_pa = get_fits_info(args.fits_file,edge_pad=edge_pad,freq=args.freq)
        ra_ind,dec_ind,ra_mesh,dec_mesh,ra_range,dec_range = find_image_centre_celestial(ras=ras,decs=decs,flat_data=flat_data,pixel_inds_to_use=pixel_inds_to_use,data=data)
        ra_cent, dec_cent, ras, decs = set_central_pixel_to_zero(popt,ras,decs,ra_range,dec_range,args,edge_pad,dims,wcs)

        xrot,yrot = radec2xy(ras,decs,pa,b1,b2)

        n1s, n2s, A_shape_basis_edge = gen_A_shape_matrix(xrot=xrot,yrot=yrot,nmax=nmax,b1=b1,b2=b2,convolve_kern=rest_gauss_kern,shape=data.shape)
        ##Creates a model of the fully fitted coeffs and a matching srclist

        fitted_coeffs.shape = (len(fitted_coeffs),1)
        fit_data_edge = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis_edge)

        fit_data_edge.shape = data.shape

        do_subplot(fig,ax4,fit_data_edge,'Edge padded fit \n(convolved with restoring beam)',vmin,vmax)

    #
    fig.tight_layout()
    #
    fig.savefig('shamfi_%s_nmax%d_p100_fit.png' %(save_tag,nmax), bbox_inches='tight')
    plt.close()

    return flat_data

def do_fitting(ras,decs,b1,b2,pa,nmax,rest_gauss_kern,data,flat_data,pixel_inds_to_use,args):
    '''Takes the data and fits shapelet basis functions to them. Uses the given
    b1, b2, pa to scale the coords, nmax to create the number of basis functions,
    and the rest_gauss_kern to apply to restoring beam to the basis functions'''

    xrot,yrot = radec2xy(ras,decs,pa,b1,b2)

    n1s, n2s, A_shape_basis = gen_A_shape_matrix(xrot=xrot,yrot=yrot,nmax=nmax,b1=b1,b2=b2,convolve_kern=rest_gauss_kern,shape=data.shape)

    fitted_coeffs = linear_solve(flat_data=flat_data[pixel_inds_to_use],A_shape_basis=A_shape_basis[pixel_inds_to_use,:])
    ##Creates a model of the fully
    fit_data_full = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis)


    if args.diff_box:
        diff_inds = []
        low_x,high_x,low_y,high_y = array(map(int,args.diff_box.split(','))) + edge_pad
        for y in range(low_y,high_y+1):
            for x in range(low_x,high_x+1):
                diff_inds.append(y*x_len + x)
        diff_inds = array(diff_inds)
    else:
        diff_inds = arange(len(flat_data))

    resid = find_resids(data=flat_data[diff_inds],fit_data=fit_data_full[diff_inds])
    # print(resid)

    return resid, fit_data_full, fitted_coeffs, n1s, n2s, xrot, yrot

def do_fitting_compressed(ras,decs,b1,b2,pa,n1s_compressed,n2s_compressed,rest_gauss_kern,data,flat_data,pixel_inds_to_use,args):
    '''Takes the data and fits shapelet basis functions to them. Uses the given
    b1, b2, pa to scale the coords, n1s_compressed, n2s_compressed to create the number of basis functions,
    and the rest_gauss_kern to apply to restoring beam to the basis functions'''

    xrot,yrot = radec2xy(ras,decs,pa,b1,b2)

    A_shape_basis_compressed = gen_reduced_A_shape_matrix(n1s=n1s_compressed,n2s=n2s_compressed,xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=rest_gauss_kern,shape=data.shape)
    fitted_coeffs_compressed = linear_solve(flat_data=flat_data,A_shape_basis=A_shape_basis_compressed)
    fit_data_compressed = fitted_model(coeffs=fitted_coeffs_compressed,A_shape_basis=A_shape_basis_compressed)

    if args.diff_box:
        diff_inds = []
        low_x,high_x,low_y,high_y = array(map(int,args.diff_box.split(','))) + edge_pad
        for y in range(low_y,high_y+1):
            for x in range(low_x,high_x+1):
                diff_inds.append(y*x_len + x)
        diff_inds = array(diff_inds)
    else:
        diff_inds = arange(len(flat_data))

    # fit_data_compressed.shape = len(fit_data_compressed)

    resid = find_resids(data=flat_data[diff_inds],fit_data=fit_data_compressed[diff_inds])
    # print(resid)

    return resid, fit_data_compressed, fitted_coeffs_compressed, xrot, yrot

def do_grid_search_fit(total_coeffs,flat_data,b1_grid,b2_grid,pa,nmax,
                       rest_gauss_kern,data,pixel_inds_to_use,args,
                       num_beta_points,ras,decs,full_fit=True,
                       n1s_compressed=False,n2s_compressed=False,
                       save_grid_npz=False):
    ##Setup up empty lists / arrays to store fitted data results
    fit_datas = []
    resids = []
    matrix_plot = zeros((num_beta_points,num_beta_points))


    fitted_coeffs_matrix = zeros((num_beta_points,num_beta_points,int(total_coeffs)))
    fitted_datas_matrix = zeros((num_beta_points,num_beta_points,len(flat_data)))
    xrot_matrix = zeros((num_beta_points,num_beta_points,len(flat_data)))
    yrot_matrix = zeros((num_beta_points,num_beta_points,len(flat_data)))

    ##Set up ranges of b1,b2 to fit
    b_inds = []
    for b1_ind in arange(num_beta_points):
        for b2_ind in arange(num_beta_points): b_inds.append([b1_ind,b2_ind])

    ##Run full nmax fits over range of b1,b2 using progressbar to, well, show progress
    for b1_ind,b2_ind in progressbar(b_inds,prefix='Fitting shapelets: '):
        if full_fit:
            resid, fit_data, fitted_coeffs, n1s, n2s, xrot, yrot = do_fitting(ras,decs,b1_grid[b1_ind],b2_grid[b2_ind],pa,nmax,rest_gauss_kern,data,flat_data,pixel_inds_to_use,args)
        else:
            resid, fit_data, fitted_coeffs, xrot, yrot = do_fitting_compressed(ras,decs,b1_grid[b1_ind],b2_grid[b2_ind],
                                                         pa,n1s_compressed,n2s_compressed,rest_gauss_kern,
                                                         data,flat_data,pixel_inds_to_use,args)
            n1s = n1s_compressed
            n2s = n2s_compressed
        ##Stick outputs in output arrays
        matrix_plot[b1_ind,b2_ind] = resid
        fitted_coeffs_matrix[b1_ind,b2_ind] = fitted_coeffs.flatten()
        fitted_datas_matrix[b1_ind,b2_ind] = fit_data.flatten()
        xrot_matrix[b1_ind,b2_ind] = xrot
        yrot_matrix[b1_ind,b2_ind] = yrot

    ##Find the minimum point
    best_b1_ind,best_b2_ind = where(matrix_plot == nanmin(matrix_plot))
    if save_grid_npz: savez_compressed('%s_grid.npz' %save_tag,matrix_plot=matrix_plot,b1_grid=b1_grid,b2_grid=b2_grid)

    ##Use minimum indexes to find best fit results
    best_b1_ind,best_b2_ind = best_b1_ind[0],best_b2_ind[0]
    b1 = b1_grid[best_b1_ind]
    b2 = b2_grid[best_b2_ind]
    fitted_coeffs = fitted_coeffs_matrix[best_b1_ind,best_b2_ind]
    fit_data = fitted_datas_matrix[best_b1_ind,best_b2_ind]
    xrot = xrot_matrix[best_b1_ind,best_b2_ind]
    yrot = yrot_matrix[best_b1_ind,best_b2_ind]

    ##Tell the user what we found in arcmin
    best_b1 = (b1/D2R)*60.0
    best_b2 = (b2/D2R)*60.0
    print('Best b1 %.2e arcmin b2 %.2e arcmin' %(best_b1,best_b2))

    return b1,b2,n1s,n2s,fitted_coeffs,fit_data,xrot,yrot,matrix_plot


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

def check_rts_or_woden_get_lines(filename):
    '''Split filename into lines (ignoring # commented lines), and check the
    first line to establish if this is an RTS or WODEN style srclist'''
    with open(filename,'r') as infile:
        lines = [line for line in infile.read().split('\n') if '#' not in line and line != '']
    if len(lines[0].split()) == 9:
        type = 'woden'
    elif len(lines[0].split()) == 4:
        type = 'rts'
    else:
        exit("The first line of %s doesn't look like either an RTS or WODEN \
              srclist. Please check your inputs" %filename)
    return type,lines


class RTS_source():
    '''Class to contain an RTS source information'''
    def __init__(self):
        self.name = ''
        self.ras = []
        self.decs = []
        self.flux_lines = []
        self.component_infos = []

class Component_Info():
    '''Class to contain an RTS component information'''
    def __init__(self):
        self.comp_type = None
        self.pa = None
        self.major = None
        self.minor = None
        self.shapelet_coeffs = []

def get_RTS_sources(srclist, all_RTS_sources):
    '''Takes a path to an RTS srclist, breaks it up into SOURCES, populates
    this information into RTS_source classes and appends them to all_RTS_sources
    list'''

    with open(srclist,'r') as srcfile:
        source_src_info = srcfile.read().split('ENDSOURCE')
        del source_src_info[-1]

        for source_info in source_src_info:
            source = RTS_source()

            primary_info = source_info.split('COMPONENT')[0].split('\n')
            primary_info = [info for info in primary_info if info!='' and '#' not in info]
            _,prim_name,prim_ra,prim_dec = primary_info[0].split()

            ##Put in to the source class
            source.name = prim_name
            source.ras.append(prim_ra)
            source.decs.append(prim_dec)
            ##Find the fluxes and append to the source class
            prim_flux_lines = []
            for line in primary_info:
                if 'FREQ' in line:
                    prim_flux_lines.append(line)
            source.flux_lines.append(prim_flux_lines)

            ##Split all info into lines and get rid of blank entries
            lines = source_info.split('\n')
            lines = [line for line in lines if line!='' and '#' not in line]
            ##If there are components to the source, see where the components start and end
            comp_starts = [i for i in range(len(lines)) if 'COMPONENT' in lines[i] and 'END' not in lines[i]]
            comp_ends = [i for i in range(len(lines)) if lines[i]=='ENDCOMPONENT']

            primary_comp = Component_Info()
            primary_comp.comp_type = 'POINT'
            ##Check to see if the primary source is a gaussian or shapelet
            for line in primary_info:
                ###Check here to see if primary source is a gaussian:
                if 'GAUSSIAN' in line:
                    primary_comp.comp_type = 'GAUSSIAN'
                    _,pa,major,minor = line.split()
                    ##convert all to radians
                    primary_comp.pa = pa
                    primary_comp.major = major
                    primary_comp.minor = minor
                ##As shapelet line comes after the
                elif 'SHAPELET' in line:
                    primary_comp.comp_type = 'SHAPELET'
                    _,pa,major,minor = line.split()
                    ##convert all to radians
                    primary_comp.pa = pa
                    primary_comp.major = major
                    primary_comp.minor = minor
                    ##If the primary source is a shapelet, search for shapelet coeffs in primary data,
                    ##gather in a list and append to the source class
                    for line in primary_info:
                        if 'COEFF' in line:
                            primary_comp.shapelet_coeffs.append(line)
                else:
                    pass

            source.component_infos.append(primary_comp)
            ##For each component, go through and find ra,dec,freqs and fluxs
            ##Also check here if the component is a gaussian or shapelet
            for start,end in zip(comp_starts,comp_ends):
                flux_lines = []
                coeffs = []
                comp = Component_Info()
                comp.comp_type = 'POINT'
                for line in lines[start:end]:
                    if 'COMPONENT' in line: #and 'END' not in line:
                        _,ra,dec = line.split()
                        source.ras.append(ra)
                        source.decs.append(dec)
                    if 'FREQ' in line:
                        flux_lines.append(line)
                    if 'GAUSSIAN' in line:
                        comp.comp_type = 'GAUSSIAN'
                        _,pa,major,minor = line.split()
                        comp.pa = pa
                        comp.major = major
                        comp.minor = minor
                    if 'SHAPELET' in line:
                        comp.comp_type = 'SHAPELET'
                        _,pa,major,minor = line.split()
                        comp.pa = pa
                        comp.major = major
                        comp.minor = minor

                    if 'COEFF' in line:
                        comp.shapelet_coeffs.append(line)

                source.flux_lines.append(flux_lines)
                source.component_infos.append(comp)

            all_RTS_sources.append(source)

    return all_RTS_sources

def write_woden_component_as_RTS(lines, outfile, name = False):
    '''Take in a number of text lines in the WODEN format and write them out
    in the RTS format'''
    for line in lines:
        if 'COMPONENT' in line and 'END' not in line:
            _,comp_type,ra,dec = line.split()
            if name:
                outfile.write('SOURCE %s %s %s\n' %(name,ra,dec))
            else:
                outfile.write('COMPONENT %s %s\n' %(ra,dec))
        if 'FREQ' in line:
            outfile.write(line + '\n')
        if 'GPARAMS' in line:
            _,pa,major,minor = line.split()
            ##The RTS uses the std instead of the FWHM for major/minor
            outfile.write('GAUSSIAN %s %s %s\n' %(pa,major*factor,minor*factor))
        if 'SPARAMS' in line:
            _,pa,major,minor = line.split()
            outfile.write('SHAPELET2 %s %s %s\n' %(pa,major,minor))
        if 'SCOEFF' in line:
            outfile.write(line[1:]+'\n')

    if name:
        pass
    else:
        outfile.write('ENDCOMPONENT\n')


def write_woden_from_RTS_sources(RTS_sources,outname,gitlabel):
    '''Takes a list of RTS_source classes and uses the to write a WODEN
    style srclist called outname'''

    all_comp_types = []
    all_shape_coeffs = 0
    for source in RTS_sources:
        for comp in source.component_infos:
            all_comp_types.append(comp.comp_type)
            all_shape_coeffs += len(comp.shapelet_coeffs)

    all_comp_types = array(all_comp_types)

    num_point = len(where(all_comp_types == 'POINT')[0])
    num_gauss = len(where(all_comp_types == 'GAUSSIAN')[0])
    num_shape = len(where(all_comp_types == 'SHAPELET')[0])

    with open(outname,'w+') as outfile:
        outfile.write('##Written with SHAMFI commit %s\n' %gitlabel)
        outfile.write('SOURCE %s P %d G %d S %d %d\n' %(RTS_sources[0].name,
                                num_point,num_gauss,num_shape,all_shape_coeffs))

        for source in RTS_sources:
            for comp_ind,comp_info in enumerate(source.component_infos):
                outfile.write('COMPONENT %s %s %s\n' %(comp_info.comp_type,source.ras[comp_ind],source.decs[comp_ind]))

                for line in source.flux_lines[comp_ind]:
                    outfile.write(line+'\n')

                if comp_info.comp_type == 'GAUSSIAN':
                    ##RTS gaussians are std dev, WODEN are FWHM
                    outfile.write('GPARAMS %s %s %s\n' %(comp_info.pa,comp_info.major/factor,comp_info.minor/factor))

                elif comp_info.comp_type == 'SHAPELET':
                    outfile.write('SPARAMS %s %s %s\n' %(comp_info.pa,comp_info.major,comp_info.minor))

                    for line in comp_info.shapelet_coeffs:
                        outfile.write('S'+line+'\n')
                outfile.write('ENDCOMPONENT\n')

        outfile.write('ENDSOURCE')

def write_singleRTS_from_RTS_sources(RTS_sources,outname,gitlabel):
    '''Takes a list RTS_sources containg RTS_source classes, and writes
    them out into a single SOURCE RTS srclist of name outname'''
    with open(outname,'w+') as outfile:
        outfile.write('##Written with SHAMFI commit %s\n' %gitlabel)

        for source_ind,source in enumerate(RTS_sources):
            for comp_ind,comp_info in enumerate(source.component_infos):
                if source_ind == 0 and comp_ind == 0:
                    outfile.write('SOURCE combined_source %s %s\n' %(source.ras[comp_ind],source.decs[comp_ind]))
                else:
                    outfile.write('COMPONENT %s %s\n' %(source.ras[comp_ind],source.decs[comp_ind]))

                for line in source.flux_lines[comp_ind]:
                    outfile.write(line+'\n')

                if comp_info.comp_type == 'GAUSSIAN':
                    outfile.write('GAUSSIAN %s %s %s\n' %(comp_info.pa,comp_info.major,comp_info.minor))

                elif comp_info.comp_type == 'SHAPELET':
                    outfile.write('SHAPELET2 %s %s %s\n' %(comp_info.pa,comp_info.major,comp_info.minor))

                    for line in comp_info.shapelet_coeffs:
                        outfile.write(line+'\n')

                if source_ind == 0 and comp_ind == 0:
                    pass
                else:
                    outfile.write('ENDCOMPONENT\n')

        outfile.write('ENDSOURCE')

def subtract_gauss(data,ind,x,y,major,minor,pa,flux,header,wcs,data_dims,
                   ax1,ax2,ax3,fig):
    '''Takes a 2D CLEAN restored image array (data) and subtracts
    a gaussian using the specified parameters. Plots the subtracted
    gaussians with postage stamps before and after subtraction'''
    ##Setup the gaussian
    major *= (1/60.0)
    minor *= (1/60.0)
    pa *= (pi/180.0)

    ra_reso = abs(float(header['CDELT1']))
    dec_reso = float(header['CDELT2'])

    x_stddev = major / (factor*ra_reso)
    y_stddev = minor / (factor*dec_reso)

    gauss_func = Gaussian2D(amplitude=1.0, x_mean=x, y_mean=y, x_stddev=x_stddev, y_stddev=y_stddev,theta=pi/2.0 + pa)

    xrange = arange(header['NAXIS1'])
    yrange = arange(header['NAXIS2'])

    x_mesh, y_mesh = meshgrid(xrange,yrange)
    gauss_subtrac = gauss_func(x_mesh,y_mesh)

    ##Set up the restoring beam and convolve gaussian to subtract to mimic fitting convolved with a restoring beam
    rest_bmaj = float(header['BMAJ'])
    rest_bmin = float(header['BMIN'])
    rest_pa = header['BPA'] * (pi/180.0)
    rest_gauss_kern = create_restoring_kernel(rest_bmaj,rest_bmin,rest_pa,ra_reso,dec_reso)

    ##Convolve with restoring beam
    gauss_subtrac = fftconvolve(gauss_subtrac, rest_gauss_kern, 'same')

    ##Get the convertsion from Jy/beam to Jy/pixel
    convert2pixel = get_conert2pixel(header)
    ##Scale the gaussian to subtract to match the desired integrated flux
    gauss_subtrac *= flux / (gauss_subtrac.sum()*convert2pixel)

    ##Define a plotting area about middle of gaussian
    half_width = 20
    low_y = int(round(y - half_width))
    high_y = int(round(y + half_width))
    low_x = int(round(x - half_width))
    high_x = int(round(x + half_width))

    ##Plot the data and the gaussian
    im1 = ax1.imshow(data[low_y:high_y,low_x:high_x],origin='lower')
    im2 = ax2.imshow(gauss_subtrac[low_y:high_y,low_x:high_x],origin='lower')

    ##subtract the gaussian
    data -= gauss_subtrac
    ##Plot the data after subtraction
    im3 = ax3.imshow(data[low_y:high_y,low_x:high_x],origin='lower')

    ##Set some colourbars and get rid of tick labels
    for ax,im in zip([ax1,ax2,ax3],[im1,im2,im3]):
        add_colourbar(ax=ax,fig=fig,im=im)
        ax.set_xticks([])
        ax.set_yticks([])

    ##Add titles if at top of plot
    if ind == 0:
        ax1.set_title('Data before subtract')
        ax2.set_title('Convolved gauss to subtract')
        ax3.set_title('Data after subtract')

    ##Convert the pixel location into an ra / dec
    if data_dims == 4:
        ra,dec,_,_ = wcs.all_pix2world(x,y,0,0,0)
    elif data_dims == 3:
        ra,dec,_,_ = wcs.all_pix2world(x,y,0,0)
    elif data_dims == 2:
        ra,dec,_,_ = wcs.all_pix2world(x,y,0)

    return data,ra,dec

def check_file_exists(filename,argname):
    '''Checks if a file exists, and throws an error and exits if not'''
    if os.path.isfile(filename):
        return filename
    else:
        exit('The input "%s=%s" does not exist\nExiting now' %(argname,filename))

def check_if_txt_extension(name):
    '''Checks if 'name' ends in '.txt', appends if not '''
    if name[-4:] == '.txt':
        outname = name
    else:
        outname = name + '.txt'

    return outname

def check_if_fits_extension(name):
    '''Checks if 'name' ends in '.fits', appends if not '''
    if name[-5:] == '.fits':
        outname = name
    else:
        outname = name + '.fits'

    return outname
