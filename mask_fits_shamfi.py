#!/usr/bin/env python
from astropy.io import fits
from numpy import *
from shamfi_lib import add_colourbar,twoD_Gaussian
import matplotlib.pyplot as plt
import scipy.optimize as opt
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

    return mask

##Get the x and y size of the image
with fits.open(args.fits_file) as hdu:

    data_shape = len(hdu[0].data.shape)

    if data_shape == 2:
        data = hdu[0].data
    elif data_shape == 3:
        data = hdu[0].data[0,:,:]
    elif data_shape == 4:
        data = hdu[0].data[0,0,:,:]

    xlen = int(hdu[0].header['NAXIS1'])
    ylen = int(hdu[0].header['NAXIS2'])

x_mesh,y_mesh = meshgrid(arange(xlen),arange(ylen))



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


fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(num_cols,num_rows,1)
im1 = ax1.imshow(data,origin='lower')

for mask in masks:
    ax1.contour(mask,colors='r',alpha=0.3)

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Fitted masks')

for ind,mask in enumerate(masks):

    with fits.open(args.fits_file) as hdu:
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

    ax = fig.add_subplot(num_cols,num_rows,ind+2)
    im = ax.imshow(fracs[ind]*data,origin='lower')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Split Source %02d' %(ind+1))

plt.tight_layout()
fig.savefig('%s_masked.png' %args.output_tag,bbox_inches='tight')
