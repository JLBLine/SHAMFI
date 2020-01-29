#!/usr/bin/env python
from numpy import *
from shamfi.shamfi_lib import *
from shamfi.git_helper import *
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import argparse

##convert between FWHM and std dev
factor = 2. * sqrt(2.*log(2.))

parser = argparse.ArgumentParser(description="Use a table of gaussian properties to subtract \
                                 gaussian sources from a specfied FITS image. Writes the \
                                 gaussian subtracted image into a new FITS file. Also writes the \
                                 subtracted sources into either a WODEN or RTS srclist")

parser.add_argument('--fits_file',default=False,
    help='FITS file to subtract gaussians from')
parser.add_argument('--gauss_table',default=False,
    help='Text file containing any number of gaussian source parameters, each line entered as: \
          x_cent(pixels) y_cent(pixels) major(FWHM, arcmins) minor(FWHM, arcmins) pa(deg) int_flux(Jy)')
parser.add_argument('--output_srclist_name',default='srclist_gaussian',
    help='Name of output srclist containing the subtracted gaussians - will add rts/woden \
          depending on type of srclist to be output')
parser.add_argument('--srclist_type',default='both',
    help='Type of srclist to output - options are: woden, rts, both, none. Defaults to both')
parser.add_argument('--outname',default='use_fits',
    help='Name for gaussian subtracted output FITS file. Defaults to using the input FITS name')
parser.add_argument('--freq', default='from_FITS',
                help='Frequency (Hz) of the image - defaults to looking for keyword FREQ and associated value')

args = parser.parse_args()

##Check inputs are real
fits_file = check_file_exists(args.fits_file,'--fits_file')
gauss_table = check_file_exists(args.gauss_table,'--gauss_table')

##Grab input gaussian params
xs, ys, majors, minors, pas, int_fluxs = loadtxt(gauss_table,comments="#",unpack=True)

##Check naming of args or setup defaults
if args.outname == 'use_fits':
    outname = 'subtracted_%s' %(fits_file.split('/')[-1])
else:
    outname = check_if_fits_extension(args.outname)

if args.output_srclist_name == 'srclist_gaussian':
    srclist_prepend = 'srclist_gaussian'
else:
    if args.output_srclist_name[:-4] == '.txt':
        srclist_prepend = args.output_srclist_name[:-4]
    else:
        srclist_prepend = args.output_srclist_name

##Open FITS file and have at thee
with fits.open(fits_file) as hdu:
    header = hdu[0].header
    data = hdu[0].data

    ##Figure out how many axis we have (WSClean is usually 4)
    data_dims = len(data.shape)

    ##Reduce data to 2D image if need be
    if data_dims == 4:
        data = data[0,0,:,:]
    elif data_dims == 3:
        data = data[0,:,:]

    ##Get the world coord system
    wcs = WCS(header)

    ##Get the frequency of the image
    freq = get_frequency(args.freq,header)

    ##Have to check if there is only one gaussian to subtract, because loadtxt
    ##will have either output a float or an array depending on how many gaussians
    ##there are
    try:
        num_gauss = len(xs)
    except TypeError:
        num_gauss = 1
        xs = array([xs])
        ys = array([ys])
        majors = array([majors])
        minors = array([minors])
        pas = array([pas])
        int_fluxs = array([int_fluxs])


    print('Found %d gaussians, subtracting now' %num_gauss)

    ##Plot the outputs
    fig = plt.figure(figsize=(10,10))

    ##Store results in this class to write them out to srclists
    source = RTS_source()
    source.name = 'gauss_subtrac'

    for ind in arange(num_gauss):
        ax1 = fig.add_subplot(num_gauss,3,3*ind+1)
        ax2 = fig.add_subplot(num_gauss,3,3*ind+2)
        ax3 = fig.add_subplot(num_gauss,3,3*ind+3)

        data,ra,dec = subtract_gauss(data,ind=ind,x=xs[ind],y=ys[ind],major=majors[ind],
                                 minor=minors[ind],pa=pas[ind],flux=int_fluxs[ind],
                                 header=header,wcs=wcs,data_dims=data_dims,
                                 ax1=ax1,ax2=ax2,ax3=ax3,fig=fig)

        ##Get the info into the source
        comp = Component_Info()
        comp.comp_type = 'GAUSSIAN'
        comp.pa = pas[ind]
        ##RTS major and minors are std dev, not FWHM
        comp.major = majors[ind]*factor
        comp.minor = minors[ind]*factor
        ##RTS is in hours, not deg
        source.ras.append(ra / 15.0)
        source.decs.append(dec)
        source.flux_lines.append(['FREQ %.5e %.10f 0 0 0' %(freq,int_fluxs[ind])])
        source.component_infos.append(comp)

    ##Make nice looking and save the plot
    fig.tight_layout()
    fig.savefig(outname[:-5]+'.png',bbox_inches='tight')

    ##substitute the subtracted data in to the hdu and save
    if data_dims == 4:
        hdu[0].data[0,0,:,:] = data
    elif data_dims == 3:
        hdu[0].data[0,:,:] = data
    elif data_dims == 2:
        hdu[0].data[:,:] = data

    git_dict = get_gitdict()

    hdu[0].header['SHAMFIv'] = git_dict['describe']
    hdu[0].header['SHAMFId'] = git_dict['date']
    hdu[0].header['SHAMFIb'] = git_dict['branch']

    ##Write out subtracted source lists if necessary
    if args.srclist_type == 'none':
        pass
    elif args.srclist_type == 'rts':
        write_singleRTS_from_RTS_sources([source],'%s-rts.txt' %srclist_prepend)
    elif args.srclist_type == 'woden':
        write_woden_from_RTS_sources([source],'%s-woden.txt' %srclist_prepend)
    else:
        write_singleRTS_from_RTS_sources([source],'%s-rts.txt' %srclist_prepend)
        write_woden_from_RTS_sources([source],'%s-woden.txt' %srclist_prepend)
