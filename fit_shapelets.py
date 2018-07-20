#!/usr/bin/env python
from __future__ import print_function,division
from numpy import *
import matplotlib.pyplot as plt
from scipy.special import factorial,eval_hermite
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes

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
    norm = sqrt(pow(2,n1+n2)*pi*b1*b2*factorial(n1)*factorial(n2))

    h1 = eval_hermite(n1,xrot)
    h2 = eval_hermite(n2,yrot)
        
    return gauss/norm*h1*h2


def gen_A_shape_matrix(xrot=None,yrot=None,nmax=None,b1=None,b2=None):
    n1s = []
    n2s = []
    for n1 in arange(nmax+1):
        for n2 in range(nmax-n1+1):
            ##If the norm factor is tiny going to have problems - 
            ##sky if we get issues
            norm = sqrt(pow(2,n1+n2)*pi*b1*b2*factorial(n1)*factorial(n2))
            
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
        

def linear_solve(image_data=None,A_shape_basis=None):
    '''Fit the image_data using the given A_shape_basis matrix
    Essentially solving for x in the equation Ax = b where:
    A = A_shape_basis
    b = the image data
    x = coefficients for the basis functions in A
    
    returns: the fitted coefficients in an array'''
    
    flat_data = image_data.flatten()
    flat_data.shape = (len(flat_data),1)
    shape_coeffs,resid,rank,s = linalg.lstsq(A_shape_basis,flat_data)
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
    
def save_srclist(save_tag=None, nmax=None, n1s=None, n2s=None, fitted_coeffs=None, b1=None, b2=None, fitted_model=None, ra_cent=None, dec_cent=None, freq=None):
    '''Take the fitted parameters and creates and RTS style srclist with them'''
    
    
    flux = sum(fitted_model)
    PA = 0.0
    major, minor = (b1 / D2R)*60, (b2 / D2R)*60
    
    
    outfile = open('srclist_%s.txt' %(save_tag),'w+')
    outfile.write('SOURCE %s %.6f %.6f\n' %(save_tag,ra_cent/15.0,dec_cent))
    outfile.write("FREQ %.2fe+6 %.5f 0 0 0\n" %(freq,flux))
    outfile.write("SHAPELET %.8f %.8f %.8f\n" %(PA,major,minor))
    
    for index,coeff in enumerate(fitted_coeffs):
        outfile.write("COEFF %.1f %.1f %.8f\n" %(n1s[index],n2s[index],coeff))
    
    outfile.write('ENDSOURCE')
    outfile.close()
    

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

    hdu = fits.open(args.fits_file)
    header = hdu[0].header

    ras = (arange(int(header['NAXIS1'])) - int(header['CRPIX1']))*float(header['CDELT1'])
    decs = (arange(int(header['NAXIS2'])) - int(header['CRPIX2']))*float(header['CDELT2'])
    
    
    if len(hdu[0].data.shape) == 2:
        data = hdu[0].data
    elif len(hdu[0].data.shape) == 3:
        data = hdu[0].data[0,:,:]
    elif len(hdu[0].data.shape) == 4:
        data = hdu[0].data[0,0,:,:]

    ras_mesh,decs_mesh = meshgrid(ras,decs)
    ras,decs = ras_mesh.flatten(),decs_mesh.flatten()
    
    ras *= D2R
    decs *= D2R
    
    b1 = (args.b1 / 60.0)*D2R
    b2 = (args.b2 / 60.0)*D2R
    
    ##This combination of b1,b2,ra,dec and rotation
    ##makes this consistent with the RTS
    
    x = ras / b2
    y = decs / b1
    
    ##TODO - use this code to introduce PA rotations
    angle = pi/2
    xrot = x*cos(angle) + y*sin(angle)
    yrot = -x*sin(angle) + y*cos(angle)
    
    n1s, n2s, A_shape_basis = gen_A_shape_matrix(xrot=xrot,yrot=yrot,nmax=nmax,b1=b1,b2=b2)

    fitted_coeffs = linear_solve(image_data=data,A_shape_basis=A_shape_basis)
    
    fit_data = fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis)
    
    #print(fit_data.max(),fit_data.min(),sum(fit_data))
    #print(data.max(),data.min(),sum(data))
    
    #jenny_style_save(save_tag, nmax, n1s, n2s, fitted_coeffs, b1, b2, float(header['CRVAL1']), float(header['CRVAL2']))
    
    if args.no_srclist:
        pass
    else:
        save_srclist(save_tag=save_tag, nmax=nmax, n1s=n1s, n2s=n2s, fitted_coeffs=fitted_coeffs, b1=b1, b2=b2, 
            fitted_model=fit_data, ra_cent= float(header['CRVAL1']), dec_cent=float(header['CRVAL2']), freq=args.freq)
    
    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(221)
    data.shape = ras_mesh.shape
    
    if args.plot_lims:
        vmin,vmax = map(float,args.plot_lims.split(','))
        im1 = ax1.imshow(data,origin='lower',vmin=vmin,vmax=vmax)
    else:
        im1 = ax1.imshow(data,origin='lower')
    add_colourbar(fig=fig,im=im1,ax=ax1)
    ax1.set_title('Data')

    ax2 = fig.add_subplot(222)
    fit_data.shape = ras_mesh.shape
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

    for ax in [ax1,ax2,ax3]:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.tight_layout()

    fig.savefig('shapelets_%s_nmax%d_fit.png' %(save_tag,nmax), bbox_inches='tight')