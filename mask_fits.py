#!/usr/bin/env python
from astropy.io import fits
from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as opt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
import argparse

parser = argparse.ArgumentParser(description="A script to split large radio galaxies into separate portions to be fitted")

parser.add_argument('--fits_file', default=False,
    help='Name of fits file to split')

parser.add_argument('--box', action='append',
    help='Any number of areas to fit a gaussian to - specify by user pixel numbers. Add each box as as: x_low,x_high,y_low,y_high. For example, to fit two gaussians, one between \
    0 to 10 in the x range, 10 to 20 in the y range, and the other between 100 to 400 in the x range, 300 to 455 in the y range, enter this on the command line: \
    mask_fits.py --box 0,10,10,20 --box 100,400,300,455')

parser.add_argument('--output_tag', default='mask',
    help='Tag name to add to outputs')

args = parser.parse_args()

# print(args.box)

def add_colourbar(fig=None,ax=None,im=None,label=False,top=False):
    '''Adds a colourbar in a nice tidy way'''
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

hdu = fits.open(args.fits_file)

data_shape = len(hdu[0].data.shape)

if data_shape == 2:
    data = hdu[0].data
elif data_shape == 3:
    data = hdu[0].data[0,:,:]
elif data_shape == 4:
    data = hdu[0].data[0,0,:,:]

hdu.close()

xlen = int(hdu[0].header['NAXIS1'])
ylen = int(hdu[0].header['NAXIS2'])

x_mesh,y_mesh = meshgrid(arange(xlen),arange(ylen))

def make_gauss_mask(low_x,high_x,low_y,high_y):

    new_data = zeros(data.shape)

    edge = 50

    low_y -= edge
    high_y += edge
    low_x -= edge
    high_x += edge

    new_data[low_y:high_y,low_x:high_x] = data[low_y:high_y,low_x:high_x]

    xcent = (high_x + low_x) / 2
    ycent = (high_y + low_y) / 2

    initial_guess = (3,xcent,ycent,20,20,0)

    popt, pcov = opt.curve_fit(twoD_Gaussian, (x_mesh, y_mesh), new_data.flatten(), p0=initial_guess)

    popt[0] = 1

    mask = twoD_Gaussian((x_mesh, y_mesh), *popt)
    mask.shape = x_mesh.shape

    # new_data[new_data <= 0] = 0

    return mask

##FnxA.fits
# mask1,data1 = make_gauss_mask(low_x=500,high_x=900,low_y=400,high_y=800)
# mask2,data2 = make_gauss_mask(low_x=200,high_x=550,low_y=300,high_y=650)

masks = []

for box in args.box:
    low_x,high_x,low_y,high_y = map(int,box.split(','))

    mask = make_gauss_mask(low_x=low_x,high_x=high_x,low_y=low_y,high_y=high_y)
    masks.append(mask)

total_mask = zeros(masks[0].shape)
for mask in masks: total_mask += mask
fracs = [mask/total_mask for mask in masks]

num_plots = float(len(masks) + 1)
num_cols = int(ceil(sqrt(num_plots)))
num_rows = int(ceil(num_plots / num_cols))

print(num_cols,num_rows)

fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(num_cols,num_rows,1)
im1 = ax1.imshow(data,origin='lower')

for mask in masks:
    ax1.contour(mask,colors='r',alpha=0.3)

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Fitted masks')

for ind,mask in enumerate(masks):

    hdu = fits.open(args.fits_file)
    data_shape = len(hdu[0].data.shape)

    if data_shape == 2:
        data = hdu[0].data
        hdu[0].data = fracs[ind]*data
    elif data_shape == 3:
        data = hdu[0].data[0,:,:]
        hdu[0].data[0,:,:] = fracs[ind]*data
    elif data_shape == 4:
        data = hdu[0].data[0,0,:,:]
        hdu[0].data[0,0,:,:] = fracs[ind]*data

    hdu.writeto('%s_split%02d.fits' %(args.output_tag,ind+1),overwrite=True)
    hdu.close()

    ax = fig.add_subplot(num_cols,num_rows,ind+2)
    im = ax.imshow(fracs[ind]*data,origin='lower')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Split Source %02d' %(ind+1))




# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
#
#
#
#
# ax1.contour(mask2,colors='r',alpha=0.3)
#
# # im3 = ax3.imshow(frac1,origin='lower')
# # im4 = ax4.imshow(frac2,origin='lower')
#
# im3 = ax3.imshow(frac1*data,origin='lower')
# im4 = ax4.imshow(frac2*data,origin='lower')
#
# add_colourbar(fig=fig,im=im1,ax=ax1)
# add_colourbar(fig=fig,im=im3,ax=ax3)
# add_colourbar(fig=fig,im=im4,ax=ax4)
#
plt.tight_layout()
fig.savefig('%s_masked.png' %args.output_tag,bbox_inches='tight')

# hdu[0].data[0,0,:,:] = frac2*data
# hdu.writeto('old_lobe2.fits',overwrite=True)

#




# low_x = 500
# high_x = 568
# low_y = 400
# high_y = 573
#
# new_data[low_y:high_y,low_x:high_x] = 0.0
#
# new_data[new_data < 0.001] = 0.0
#
# plt.imshow(new_data)
# plt.show()
