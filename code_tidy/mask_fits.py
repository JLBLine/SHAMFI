from astropy.io import fits
from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as opt
from my_plotting_lib import *


def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
    b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
    c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.flatten()


# hdu = fits.open('FnxA.fits')
hdu = fits.open('cropped_ben_FornaxA.fits')


if len(hdu[0].data.shape) == 2:
    data = hdu[0].data
elif len(hdu[0].data.shape) == 3:
    data = hdu[0].data[0,:,:]
elif len(hdu[0].data.shape) == 4:
    data = hdu[0].data[0,0,:,:]


##Try and get rid of central point SOURCE



mean,std = 0.2156,0.0233

low_x = 122
high_x = 130
low_y = 127
high_y = 135

noise_hole = random.normal(mean,std,(high_y - low_y,high_x-low_x))

data[low_y:high_y,low_x:high_x] = noise_hole


mean,std = 0.03504,0.01729

low_x = 229
high_x = 236
low_y = 189
high_y = 197

noise_hole = random.normal(mean,std,(high_y - low_y,high_x-low_x))

data[low_y:high_y,low_x:high_x] = noise_hole





xlen = int(hdu[0].header['NAXIS1'])
ylen = int(hdu[0].header['NAXIS2'])

x_mesh,y_mesh = meshgrid(arange(xlen),arange(ylen))

# mask = x_mask,y_mask


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

    # if popt[0] < 1e-3:
    #     popt[0] = 0.1

    popt[0] = 1

    mask = twoD_Gaussian((x_mesh, y_mesh), *popt)
    mask.shape = x_mesh.shape

    new_data[new_data <= 0] = 0

    return mask,new_data


##FnxA.fits
# mask1,data1 = make_gauss_mask(low_x=500,high_x=900,low_y=400,high_y=800)
# mask2,data2 = make_gauss_mask(low_x=200,high_x=550,low_y=300,high_y=650)

mask1,data1 = make_gauss_mask(low_x=126,high_x=231,low_y=90,high_y=212)
mask2,data2 = make_gauss_mask(low_x=15,high_x=115,low_y=55,high_y=165)

total_mask = mask1+mask2
frac1 = mask1 / total_mask
frac2 = mask2 / total_mask

fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

im1 = ax1.imshow(data,origin='lower')

ax1.contour(mask1,colors='r',alpha=0.3)
ax1.contour(mask2,colors='r',alpha=0.3)

# im3 = ax3.imshow(frac1,origin='lower')
# im4 = ax4.imshow(frac2,origin='lower')

im3 = ax3.imshow(frac1*data,origin='lower')
im4 = ax4.imshow(frac2*data,origin='lower')

add_colourbar(fig=fig,im=im1,ax=ax1)
add_colourbar(fig=fig,im=im3,ax=ax3)
add_colourbar(fig=fig,im=im4,ax=ax4)

plt.tight_layout()
fig.savefig('mask_fits.png',bbox_inches='tight')

# hdu[0].data[0,0,:,:] = frac2*data
# hdu.writeto('ben_lobe2.fits',overwrite=True)

hdu[0].data[0,0,:,:] = frac1*data
hdu.writeto('ben_lobe1.fits',overwrite=True)




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
