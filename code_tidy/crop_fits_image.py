from astropy.io import fits
import matplotlib.pyplot as plt

HDU = fits.open('2014B_2018A_joint_idg_stokesI_deeper_self_cal_02_spec_pol_1-MFS-image-pb.fits')

data = HDU[0].data
header = HDU[0].header

cent_x = int(header['CRPIX1'])
cent_y = int(header['CRPIX2'])

new_size = 256
half = int(new_size / 2)

print cent_y-half-1,cent_y+half,cent_x-half-1,cent_x+half

crop_data = data[:,:,cent_y-half-1:cent_y+half,cent_x-half-1:cent_x+half]

hdu = fits.PrimaryHDU(crop_data)
hdulist = fits.HDUList([hdu])
hdulist[0].header = HDU[0].header
new_header = hdulist[0].header

new_header['NAXIS1'] = new_size
new_header['NAXIS2'] = new_size
new_header['CRPIX1'] = half + 1
new_header['CRPIX2'] = half + 1

hdu.writeto('cropped_ben_FornaxA.fits',overwrite=True)
