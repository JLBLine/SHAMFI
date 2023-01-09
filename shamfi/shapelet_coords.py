from __future__ import print_function,division
from numpy import *
from numpy import abs as np_abs
import scipy.optimize as opt
from copy import deepcopy
import matplotlib.pyplot as plt


##Convert degress to radians
D2R = pi/180.
##Convert radians to degrees
R2D = 180./pi

##convert between FWHM and std dev for the gaussian function
FWHM_factor = 2. * sqrt(2.*log(2.))

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
    """
    Creates a model for a 2D Gaussian, by taking two 2D coordinate arrays in x,y.
    Returns a flattened array of the model to make fitting more straight forward

    :param list xy: list containing two 2D numpy arrays
        A list containing the x and y coordinates to calculate the gaussian at.
        x and y are separated into individual 2D arrays. xy = [x(2D), y(2D)]
    :param float amplitude: float
        Amplitude to scale the Gaussian by
    :param float xo: float
        Value of the central x pixel
    :param float yo: float
        Value of the central y pixel
    :param float sigma_x: float
        Sigma value for the x dimension
    :param float sigma_y: float
        Sigma value for the y dimension
    :param float theta: float
        Rotation angle (radians)
    :return gaussian.flatten():
        A 2D gaussian model, flattened into a 1D numpy array
    :rtype: array
    """

    x,y = xy

    xo = float(xo)
    yo = float(yo)
    a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
    b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
    c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
    gaussian = amplitude*exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return gaussian.flatten()


class ShapeletCoords():
    """
    Takes a shamfi.read_FITS_image class, and uses it to generate a cartesian
    coord system to fit shapelets in. Include attributes to flux centre the
    coord system, and to mask the data as required. See the attribute docs
    below for details.

    :param fits_data: a shamfi.read_FITS_image.FITSInformation class that has already been initialised
    :ivar fits_data: The supplied shamfi.read_FITS_image.FITSInformation class parameter
    """

    def __init__(self,fits_data=False):
        if fits_data:
            pass
        else:
            print('To initialise a ShapeletCoords Class you need to supply a shamfi.read_FITS_image class as the fits_data argument')

        self.fits_data = fits_data
        self.edge_pad = self.fits_data.edge_pad

    def find_good_pixels(self,fit_box=False,exclude_box=False,ignore_negative=False):
        """
        Generates a mask to ignore pixels when fitting the shapelets based on
        the given arguments

        :param str fit_box: only use the pixels as bounded by fit_box = "low_x,high_x,low_y,high_y" (pixel coords)
        :param list exclude_box: exclude pixels bounded by the boxes defined in exclude_box, e.g. exclude_box=["low_x,high_x,low_y,high_y", "low_x,high_x,low_y,high_y"]
        :param bool ignore_negative: mask all pixels with negative values in self.fits_data.flat_data

        :ivar array pixel_inds_to_use: array of pixel indexes to use
        :ivar array negative_pix_mask: index of positive data pixels
        """
        ##If a box is specified, limit pixels to within that box
        if fit_box:
            pixel_inds_to_use = []
            low_x,high_x,low_y,high_y = array(fit_box.split(','),dtype=int)



            ##If edge padding is greater than zero, but the user wants to use
            ##only a certain box, we need to move the zero padding to surrond
            ##that patch of the image, so set the appropriate parts of the
            ##image to zero here
            if self.edge_pad > 0:
                high_x += self.edge_pad*2
                high_y += self.edge_pad*2

                self.fits_data.data[low_y:low_y+self.edge_pad,:] = 0.0
                self.fits_data.data[high_y-self.edge_pad:high_y,:] = 0.0
                self.fits_data.data[:,low_x:low_x+self.edge_pad] = 0.0
                self.fits_data.data[:,high_x-self.edge_pad:high_x] = 0.0

                self.fits_data.flat_data = self.fits_data.data.flatten()

            for y in range(low_y,high_y+1):
                for x in range(low_x,high_x+1):
                    pixel_inds_to_use.append(y*self.fits_data.len1 + x)

            self.pixel_inds_to_use = array(pixel_inds_to_use)
            print('Will fit box defined by low_x,high_x,low_y,high_y: ',low_x,high_x,low_y,high_y)
            # print("JUST DONE CALC", len(pixel_inds_to_use))

        else:
            ##If nothing declared, just use all the pixels
            if not exclude_box:
                self.pixel_inds_to_use = arange(len(self.fits_data.flat_data))

            ##Otherwise, use the defined boxes in --exclude_box to flag pixels to
            ##avoid
            else:
                try:
                    avoid_inds = []
                    for box in exclude_box:
                        low_x,high_x,low_y,high_y = array(box.split(','),dtype=int)
                        for y in range(low_y,high_y+1):
                            for x in range(low_x,high_x+1):
                                avoid_inds.append(y*self.fits_data.len1 + x)

                    pixel_inds_to_use = arange(len(self.fits_data.flat_data))
                    self.pixel_inds_to_use = setxor1d(pixel_inds_to_use,avoid_inds)
                    print('Will fit avoiding boxes defined in --exclude_box')
                except:
                    self.pixel_inds_to_use = arange(len(self.fits_data.flat_data))
                    print('Failed to convert --exclude_box into something \
                          sensible. Will fit using all pixels in image')

        ##If we are to ignore negative pixels, find all negative pixels in the
        ##good list of pixels. Keep as a separate pixel mask as when generating
        ##basis functions we convolve with the restoring beam, and if random
        ##pixels are missing this convolution become innaccurate
        if ignore_negative:
            print('Ignoring negative pixels in fit')
            fluxes = self.fits_data.flat_data[self.pixel_inds_to_use]
            self.negative_pix_mask = where(fluxes >= 0.0)[0]
        else:
            print('Will include negative pixels in fit')
            self.negative_pix_mask = arange(len(self.pixel_inds_to_use))

        self._find_image_centre_celestial()


    def _find_image_centre_celestial(self):
        '''Find the flux-weighted central position of an image'''
        power = 4

        ras_to_use = deepcopy(self.fits_data.ras[self.pixel_inds_to_use])

        ##Make it wrap throught -180.0 to 180 if we go over the 0/360 wrap

        if ras_to_use.min() < 1.0*D2R and ras_to_use.min() > 359.0*D2R:
            ras_to_use[ras_to_use > pi] -= 2*pi
            ras_to_use[ras_to_use < -pi] += 2*pi


        ra_cent = sum(np_abs(self.fits_data.flat_data[self.pixel_inds_to_use])**power*ras_to_use)
        ra_cent /= sum(np_abs(self.fits_data.flat_data[self.pixel_inds_to_use])**power)

        dec_cent = sum(np_abs(self.fits_data.flat_data[self.pixel_inds_to_use])**power*self.fits_data.decs[self.pixel_inds_to_use])
        dec_cent /= sum(np_abs(self.fits_data.flat_data[self.pixel_inds_to_use])**power)

        # resolution = abs(self.fits_data.ras[1] - self.fits_data.ras[0])
        # ##Find the difference between the gridded ra coords and the desired ra_cent
        # ra_offs = np_abs(self.fits_data.ras - ra_cent)
        # ##Find out where in the gridded ra coords the current ra_cent lives;
        # ##This is a boolean array of length len(ra_offs)
        # ra_true = ra_offs < resolution/2.0
        # ##Find the index so we can access the correct entry in the container
        # ra_ind = where(ra_true == True)[0]
        #
        # ##Use the numpy abs because it's faster (np_abs)
        # dec_offs = np_abs(self.fits_data.decs - dec_cent)
        # dec_true = dec_offs < resolution/2
        # dec_ind = where(dec_true == True)[0]
        #
        # ##If ra_ind,dec_ind coord sits directly between two grid points,
        # ##just choose the first one
        # if len(ra_ind) == 0:
        #     ra_true = ra_offs <= resolution/2
        #     ra_ind = where(ra_true == True)[0]
        # if len(dec_ind) == 0:
        #     dec_true = dec_offs <= resolution/2
        #     dec_ind = where(dec_true == True)[0]
        # ra_ind,dec_ind = ra_ind[0],dec_ind[0]
        #
        # ##Central dec index has multiple rows as it is from flattended coords,
        # ##remove that here
        # dec_ind = floor(dec_ind / self.fits_data.len1)
        # print('Centre of flux pixel in image found as x,y',ra_ind,dec_ind)

        ra_mesh = deepcopy(self.fits_data.ras)
        ra_mesh.shape = self.fits_data.data.shape

        dec_mesh = deepcopy(self.fits_data.decs)
        dec_mesh.shape = self.fits_data.data.shape

        ra_range = ra_mesh[0,:]
        dec_range = dec_mesh[:,0]

        self.ra_cent = ra_cent
        self.dec_cent = dec_cent

        self.ra_mesh = ra_mesh
        self.dec_mesh = dec_mesh

    def fit_gauss_and_centre_coords(self,b1_max=False,b2_max=False):
        """
        Try and fit a Gaussian to the image, using the flux weighted central pixel
        location, and maximum b1 and b2 values, as an initial parameter estimate

        :param float b1_max: maximum beta scaling for major axis direction
        :param float b2_max: maximum beta scaling for minor axis direction
        :ivar float ra_cent: RA value of the flux weighted central pixel
        :ivar float dec_cent: Dec value of the flux weighted central pixel
        :ivar array ras: all RA values, zeroed on the fluxed weighted pixel (deg)
        :ivar array decs: all Dec values, zeroed on the fluxed weighted pixel (deg)
        """
        ##Fit a gaussian to the data to find pa
        ##guess is: amp, xo, yo, sigma_x, sigma_y, pa
        initial_guess = (self.fits_data.data.max(),self.ra_cent,self.dec_cent,
                        (b1_max / 60.0)*D2R,(b2_max / 60.0)*D2R,0)

        popt, pcov = opt.curve_fit(twoD_Gaussian, (self.ra_mesh, self.dec_mesh), self.fits_data.flat_data, p0=initial_guess)

        # print(popt[5], popt[5] % (2*pi))

        #
        ##Check pa is between 0 <= pa < 2pi
        pa = popt[5]
        if pa < 0:
            pa += 2*pi
        ##Necessary to move from my gaussian which has theta = 0 at x = 0 and
        ##actual PA which is east from north
        pa += pi / 2.0
        if pa > 2*pi:
            pa -= 2*pi

        self.pa = pa
        self.popt = popt

        x0 = popt[1]
        y0 = popt[2]

        #
        # ##Set central ra, dec pixel to zero in prep for scaling to x,y coords
        self._set_central_pixel_to_zero(x0, y0)

    def set_centre_coords_to_fitting_region_cent(self, pa=0.0):
        """
        Run after self.find_good_pixels. This function just sets the centre
        of the basis functions to sit at the centre of fitting region. Also
        sets the pa = 0

        """

        print(f"Setting pa to {pa/D2R} deg")
        self.pa = pa

        # fit_region_ras = deepcopy(self.fits_data.ras[self.pixel_inds_to_use])
        # fit_region_decs = self.fits_data.decs[self.pixel_inds_to_use]
        #
        # ##If we cross the 360.0 / 0.0 RA deg boundary, set > 180.0 to negative
        # if fit_region_ras.min() < 90.0*D2R and fit_region_ras.max() > 270.0*D2R:
        #
        #     fit_region_ras[fit_region_ras > pi] -= 2*pi
        #
        # ra_cent = sum(fit_region_ras) / len(fit_region_ras)
        # dec_cent = sum(fit_region_decs) / len(fit_region_decs)
        #
        # if ra_cent < 0: ra_cent += 2*pi

        xmesh, ymesh = meshgrid(arange(self.fits_data.len1),
                                arange(self.fits_data.len2))

        xmesh = xmesh.flatten()[self.pixel_inds_to_use]
        ymesh = ymesh.flatten()[self.pixel_inds_to_use]

        # print(xmesh)

        cent_pix_x = mean(xmesh) + 0.5
        cent_pix_y = mean(ymesh) + 0.5

        print(cent_pix_x, cent_pix_y)

        ra_cent, dec_cent = self.fits_data.wcs.all_pix2world(cent_pix_x, cent_pix_y, 0)

        ra_cent *= D2R
        dec_cent *= D2R



        # print('THIS', ra_cent, dec_cent)

        # dec_cent_ind = int(floor(dec_cent_ind / self.fits_data.len1))
        #
        #
        # fit_region_pix_xs = self.fits_data.ras[self.pixel_inds_to_use]
        # fit_region_pix_ys = self.fits_data.decs[self.pixel_inds_to_use]

        print('Centre pixel forced to be ra,dec',ra_cent/D2R,dec_cent/D2R)

        self._set_central_pixel_to_zero(ra_cent, dec_cent)


        # print(ra_cent, dec_cent)
        #
        # print(f"Setting the central ra,dec to {ra_cent}, {dec_cent} deg")
        #
        # print(f"mean coords {mean(self.fits_data.ras)} {mean(self.fits_data.decs)}")
        #
        # resolution = abs(self.fits_data.ras[1] - self.fits_data.ras[0])
        # ##Find the difference between the gridded ra coords and the desired ra_cent
        # ra_offs = np_abs(self.fits_data.ras - ra_cent)
        # ##Find out where in the gridded ra coords the current ra_cent lives;
        # ##This is a boolean array of length len(ra_offs)
        # ra_true = ra_offs < resolution/2.0
        # ##Find the index so we can access the correct entry in the container
        # ra_ind = where(ra_true == True)[0]
        #
        # # print(resolution)
        #
        # ##Use the numpy abs because it's faster (np_abs)
        # dec_offs = np_abs(self.fits_data.decs - dec_cent)
        # dec_true = dec_offs < resolution/2.0
        # dec_ind = where(dec_true == True)[0]
        #
        # ##If ra_ind,dec_ind coord sits directly between two grid points,
        # ##just choose the first one
        # if len(ra_ind) == 0:
        #     ra_true = ra_offs < resolution
        #     ra_ind = where(ra_true == True)[0]
        # if len(dec_ind) == 0:
        #     dec_true = dec_offs < resolution
        #     dec_ind = where(dec_true == True)[0]
        #
        # # print(where(dec_true == True))
        #
        # ra_ind,dec_ind = ra_ind[0],dec_ind[0]
        #
        # ##Central dec index has multiple rows as it is from flattended coords,
        # ##remove that here
        # dec_ind = int(floor(dec_ind / self.fits_data.len1))
        # print('Centre pixel forced to be x,y',ra_ind,dec_ind)





        # self.ra_cent_ind = ra_ind
        # self.dec_cent_ind = dec_ind

        ##Set central ra, dec pixel to zero in prep for scaling to x,y coords
        # self._set_central_pixel_to_zero(ra_cent, dec_cent)

    def _set_central_pixel_to_zero(self, ra_cent, dec_cent):
        """
        Takes the self.ras, self.decs coord system and sets ra_cent, dec_cent = 0,0
        """

        print(f"Setting the central ra,dec to {ra_cent/D2R}, {dec_cent/D2R} deg")

        self.ra_cent = ra_cent
        self.dec_cent = dec_cent

        self.ras = self.fits_data.ras - ra_cent
        self.decs = self.fits_data.decs - dec_cent

        ##Make things bound by -180 to 180 deg
        self.ras[self.ras < -pi] += 2*pi
        self.ras[self.ras > pi] -= 2*pi

        plot_ras = deepcopy(self.ras)
        plot_decs = deepcopy(self.decs)

        plot_ras.shape = (self.fits_data.len1, self.fits_data.len2)
        plot_decs.shape = (self.fits_data.len1, self.fits_data.len2)

        # fig, axs = plt.subplots(1,2)
        #
        # im0 = axs[0].imshow(plot_ras*R2D, origin='lower')
        # im1 = axs[1].imshow(plot_decs*R2D, origin='lower')
        #
        # from shamfi.shamfi_plotting import add_colourbar
        #
        # add_colourbar(im=im0, ax=axs[0], fig=fig)
        # add_colourbar(im=im1, ax=axs[1], fig=fig)
        #
        # plt.tight_layout()
        # fig.savefig("check_coords.png",bbox_inches='tight')
        # plt.close()


    def _set_given_radec_to_zero_pixel(self, ra_cent, dec_cent):
        """Given an ra,dec (in radians), set this ra/dec to zero by subtracting
        from self.ras, self.decs"""


        self._set_central_pixel_to_zero(ra_cent, dec_cent)



    def _get_lm(self, ra, ra0, dec, dec0):
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
        # n = sdec*sdec0 + cdec*cdec0*cdra

        return l,m


    def radec2xy(self,b1,b2,crop=False):
        """
        Transforms the RA/DEC coords system into the shapelet x/y system
        for given b1,b2 parameters, and the self.pa rotation angle.
        Also optinally applies the self.pixel_inds_to_use cut to only return
        the pixels to be used in the shapelet fit

        :param float b1: beta parameter to scale the x-axis (radians)
        :param float b2: beta parameter to scale the y-axis (radians)
        :param bool crop: If True, apply self.pixel_inds_to_use to the transformation
        :return xrot,yrot: The x,y coords in 1D numpy arrays
        :rtype: arrays

        """

        ##If we want to ignore bad pixels, do the crop here
        if crop:
            ##RA increases in opposite direction to x
            x = -self.ras[self.pixel_inds_to_use]
            y = self.decs[self.pixel_inds_to_use]

            # ra_max = x.max()
            # ra_min = x.min()
            #
            # dec_max = y.max()
            # dec_min = y.min()
            #
            # ##maximum coord of the shapelet basis functions
            # XMAX = 250
            # XSPAN = 2*XMAX
            #
            # xspan = (XSPAN*b1)/FWHM_factor
            # yspan = (XSPAN*b2)/FWHM_factor
            #
            # print(f" + x span of basis functions is {xspan:.1f} arcmins ({xspan/60.0} deg)")
            # print(f" + y span of basis functions is {xspan:.1f} arcmins ({xspan/60.0} deg)")
            #
            # ra_span = (ra_max - ra_min) * R2D
            # dec_span = (dec_max - dec_min) * R2D
            #
            # print(f" + RA span of fitting region is {ra_span*60.0:.1f} arcmins ({ra_span} deg)")
            # print(f" + Dec span of fitting region is {dec_span*60.0:.1f} arcmins ({dec_span} deg)")

        else:
            ##RA increases in opposite direction to x
            x = -self.ras
            y = self.decs


        # x, y = self._get_lm(x, 0.0, y, 0.0)

        # print(len(x), len(self.pixel_inds_to_use))


        x, y = self._get_lm(x + self.ra_cent, self.ra_cent,
                            y + self.dec_cent, self.dec_cent)


        print(f"Rotating by position angle of {-self.pa/D2R}")
        ##Rotation is east from north, (positive RA is negative x)
        angle = -self.pa

        yrot = x*cos(angle) + -y*sin(angle)
        xrot = x*sin(angle) + y*cos(angle)

        ##Apply conversion into stdev from FWHM and beta params
        xrot *= FWHM_factor / b1
        yrot *= FWHM_factor / b2

        return xrot,yrot
