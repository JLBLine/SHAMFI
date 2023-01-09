from __future__ import print_function,division
from numpy import *
from astropy.convolution import convolve, convolve_fft
from progressbar import progressbar
import pkg_resources
from copy import deepcopy
from astropy.io import fits
import warnings
from astropy.utils.exceptions import AstropyWarning
from shamfi.git_helper import get_gitdict, write_git_header
# from shamfi import shamfi_plotting
from scipy.signal import fftconvolve
import scipy
# from numba import jit

try:
    from scipy.special import factorial,eval_hermite
except:
    print('Could not import scipy.special.factorial - you will not be able to use the function gen_shape_basis_direct')

##Convert degress to radians
D2R = pi/180.
##Convert radians to degrees
R2D = 180./pi

##Max x value of stored basis functions
XMAX = 250
##Number of samples in stored basis functions
NX = 20001
##More basis function values
XCENT = int(floor(NX / 2))
XRANGE = linspace(-XMAX,XMAX,NX)
XRES = XRANGE[1] - XRANGE[0]

##convert between FWHM and std dev for the gaussian function
FWHM_factor = 2. * sqrt(2.*log(2.))

##converts between FWHM and std dev for the RTS
rts_factor = sqrt(pi**2 / (2.*log(2.)))

##Use package manager to get hold of the basis functions
basis_path = pkg_resources.resource_filename("shamfi", "image_shapelet_basis.npz")

##Load the basis functions
image_shapelet_basis = load(basis_path)
basis_matrix = image_shapelet_basis['basis_matrix']
gauss_array = image_shapelet_basis['gauss_array']

def gen_shape_basis_direct(n1=None,n2=None,xrot=None,yrot=None,b1=None,b2=None,convolve_kern=False,shape=False):
    """
    Directly generates the shapelet basis function for given n1,n2,b1,b2 parameters,
    at the given coords xrot,yrot. b1,b2 should be in radians. Uses scipy to generate
    factorials and hermite polynomials

    Parameters
    ----------
    n1 : int
        The first order of the basis function to generate
    n2 : int
        The second order of the basis function to generate
    xrot : numpy array
        1D array of the x-coords to generate the basis functions at
    yrot : numpy array
        1D array of the y-coords to generate the basis functions at
    b1 : float
        The major axis beta scaling parameter (radians)
    b2 : float
        The minor axis beta scaling parameter (radians)
    convolve_kern : 2D numpy array
        A kernel to convolve the basis functions with - if modelling a CLEANed image this should be the restoring beam
    shape : tuple
        The 2D shape of the image being modelled, needed if convolving with a kernel

    Returns
    -------
    basis : numpy array
        A 1D array of the values of the basis function
    """

    this_xrot = deepcopy(xrot)
    this_yrot = deepcopy(yrot)

    gauss = exp(-0.5*(array(this_xrot)**2+array(this_yrot)**2))

    n1 = int(n1)
    n2 = int(n2)

    # print("THIS THING",2**n1*factorial(n1))*sqrt(2**n2*factorial(n2))

    norm = 1.0 / (sqrt(2**n1*factorial(n1))*sqrt(2**n2*factorial(n2)))
    # print("THE NORM INSIDE", norm)
    norm *= 1.0 / (b1 * b2)
    norm *= sqrt(pi) / 2

    extra_norm = (2*sqrt(b1)*sqrt(b2)) / pi
    # norm *= extra_norm


    # norm_lower1 = sqrt(2**n1*sqrt(pi)*factorial(n1))*sqrt(b1)
    # norm_lower2 = sqrt(2**n2*sqrt(pi)*factorial(n2))*sqrt(b2)
    #
    # norm = 1 / (norm_lower1*norm_lower2)

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
        # print("THE HERM", h1*h2)

    return basis

# @profile
# @jit
def interp_basis(xrot=None,yrot=None,n1=None,n2=None):
    """
    Uses basis lookup tables to generate 2D shapelet basis function for given
    xrot, yrot coords and n1,n2 orders. Does NOT include the b1,b2 normalisation,
    this is applied by the function gen_shape_basis

    Parameters
    ----------
    n1 : int
        The first order of the basis function to generate
    n2 : int
        The second order of the basis function to generate
    xrot : numpy array
        1D array of the x-coords to generate the basis functions at
    yrot : numpy array
        1D array of the y-coords to generate the basis functions at

    Returns
    -------
    basis : numpy array
        A 1D array of the values of the basis function
    """

    xpos = xrot / XRES + XCENT
    xindex = floor(xpos)
    xlow = basis_matrix[n1,xindex.astype(int)]
    xhigh = basis_matrix[n1,xindex.astype(int)+1]
    x_val = xlow + (xhigh-xlow)*(xpos-xindex)

    ypos = yrot / XRES + XCENT
    yindex = floor(ypos)
    ylow = basis_matrix[n2,yindex.astype(int)]
    yhigh = basis_matrix[n2,yindex.astype(int)+1]
    y_val = ylow + (yhigh-ylow)*(ypos-yindex)

    gxpos = xrot / XRES + XCENT
    gxindex = floor(gxpos)
    gxlow = gauss_array[gxindex.astype(int)]
    gxhigh = gauss_array[gxindex.astype(int)+1]
    gx_val = gxlow + (gxhigh-gxlow)*(gxpos-gxindex)

    gypos = yrot / XRES + XCENT
    gyindex = floor(gypos)
    gylow = gauss_array[gyindex.astype(int)]
    gyhigh = gauss_array[gyindex.astype(int)+1]
    gy_val = gylow + (gyhigh-gylow)*(gypos-gyindex)

    return x_val*y_val*gx_val*gy_val

# @profile
def gen_shape_basis(n1=None,n2=None,xrot=None,yrot=None,b1=None,b2=None,
                    convolve_kern=False, shape=False, pixel_mask=False):
    """
    Generates the shapelet basis function for given n1,n2,b1,b2 parameters, using
    lookup tables and interpolation, at the given coords xrot,yrot.
    b1,b2 should be in radians.

    Parameters
    ----------
    n1 : int
        The first order of the basis function to generate
    n2 : int
        The second order of the basis function to generate
    xrot : numpy array
        1D array of the x-coords to generate the basis functions at
    yrot : numpy array
        1D array of the y-coords to generate the basis functions at
    b1 : float
        The major axis beta scaling parameter (radians)
    b2 : float
        The minor axis beta scaling parameter (radians)
    convolve_kern : 2D numpy array
        A kernel to convolve the basis functions with - if modelling a CLEANed image this should be the restoring beam
    shape : tuple
        The 2D shape of the image being modelled, needed if convolving with a kernel

    Returns
    -------
    basis : numpy array
        A 1D array of the values of the basis function
    """

    ##Ensure n1,n2 are ints
    n1 = int(n1)
    n2 = int(n2)

    ##Calculate the normalisation
    norm = 1.0 / (b1 * b2)
    norm *= sqrt(pi) / 2

    ##If pixel_mask is an array, use it to select the correct pixels
    if type(pixel_mask) == ndarray:
        pass
    ##If not, generate a mask that just calls all the pixels
    else:
        pixel_mask = arange(len(xrot))

    # if type(convolve_kern) == ndarray:
    #     xrot.shape = shape
    #     yrot.shape = shape
    #     basis = interp_basis(xrot=xrot,yrot=yrot,n1=n1,n2=n2)
    #     # basis = convolve(basis, convolve_kern) #, 'same')
    #
    #     basis = fftconvolve(basis, convolve_kern, 'same')
    #
    #     basis = basis.flatten()
    #
    #     xrot = xrot.flatten()
    #     yrot = yrot.flatten()
    #
    # else:
    #     basis = interp_basis(xrot=xrot,yrot=yrot,n1=n1,n2=n2)

    if type(convolve_kern) == ndarray:

        ##Doing the basis function interpolation can be expensive, so
        ##only do it for the pixels were are going to fit for

        basis = interp_basis(xrot=xrot[pixel_mask],
                             yrot=yrot[pixel_mask],
                             n1=n1,n2=n2)

        ## We want to do a 2D convolution however, and everything here
        ## is currently 1D. The pixels we want to fit aren't
        ## necessarily in a nice grid though, so make a 2D array with
        ## NaNs where ever a pixel has been masked, and use astropy
        ## convolution which can deal with NaN pixels
        basis_for_convolution = ones(len(xrot))*nan
        ##Stick in the basis function date for pixels we have calculated
        ##above
        basis_for_convolution[pixel_mask] = basis

        ##Make 2D
        basis_for_convolution.shape = shape

        ##Astropy spits out warnings about convolving with many nans
        ##when we're fitting only portions of the box, so silence the
        ##warnings here
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            ##Convolve and flatten
            basis_for_convolution = convolve(basis_for_convolution, convolve_kern)
            # basis_for_convolution = convolve_fft(basis_for_convolution,convolve_kern)
        basis_for_convolution = basis_for_convolution.flatten()
        basis = basis_for_convolution[pixel_mask]

    else:
        basis = interp_basis(xrot=xrot[pixel_mask],yrot=yrot[pixel_mask],
                             n1=n1,n2=n2)

    # the_thing = basis*norm

    # if the_thing.min() < 1e-6 or the_thing.max() > 1e+6:
    #     print('Things are massive mate', n1, n2)

    # print('Min and Max of basis', n1, n2, the_thing.min(), the_thing.max())

    return basis*norm

# @profile
def gen_A_shape_matrix(n1s=None, n2s=None, xrot=None, yrot=None,
                       b1=None, b2=None, convolve_kern=False, shape=False,
                       pixel_mask=False):
    """
    Generates the 'A' matrix in Ax = b when fitting shapelets, where:

     - b = 1D matrix of data points to mode
     - x = 1D matrix containing coefficients for basis functions
     - A = 2D matrix containg shapelet basis function values

    Generates basis function values using a lookup table method

    Parameters
    ----------
    n1s : array
        The first orders of the basis functions to generate
    n2s : array
        The second orders of the basis function to generate
    xrot : numpy array
        1D array of the x-coords to generate the basis functions at
    yrot : numpy array
        1D array of the y-coords to generate the basis functions at
    b1 : float
        The major axis beta scaling parameter (radians)
    b2 : float
        The minor axis beta scaling parameter (radians)
    convolve_kern : 2D numpy array
        A kernel to convolve the basis functions with - if modelling a CLEANed image this should be the restoring beam
    shape : tuple
        The 2D shape of the image being modelled, needed if convolving with a kernel
    pixel_mask : array
        Array of pixel indexes to use in the fitting

    Returns
    -------
    n1s : array
        The first orders of the basis functions to generate
    n2s : array
        The second orders of the basis function to generate
    A_shape_basis : 2D array
        The generated 2D 'A' matrix

    """

    ##If pixel_mask is an array, use it to select the correct pixels
    if type(pixel_mask) == ndarray:
        pass
    ##If not, generate a mask that just calls all the pixels
    else:
        pixel_mask = arange(len(xrot))

    ##A_matrix only needs to be as big as the pixels we are actually fitting
    A_shape_basis = zeros((len(pixel_mask),len(n1s)))
    for index,n1 in enumerate(n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1, n2=n2s[index],
                                                xrot=xrot, yrot=yrot,
                                                b1=b1, b2=b2,
                                                convolve_kern=convolve_kern,
                                                shape=shape,
                                                pixel_mask=pixel_mask)
        ##TODO can check quality of generated basis functions here and remove if necessary

    return n1s, n2s, A_shape_basis

def gen_A_shape_matrix_direct(n1s=None, n2s=None, xrot=None, yrot=None, b1=None, b2=None, convolve_kern=False, shape=False):
    """
    Generates the 'A' matrix in Ax = b when fitting shapelets, where:

     - b = 1D matrix of data points to mode
     - x = 1D matrix containing coefficients for basis functions
     - A = 2D matrix containg shapelet basis function values

    Generates the basis functions directly using scipy

    Parameters
    ----------
    n1s : array
        The first orders of the basis functions to generate
    n2s : array
        The second orders of the basis function to generate
    xrot : numpy array
        1D array of the x-coords to generate the basis functions at
    yrot : numpy array
        1D array of the y-coords to generate the basis functions at
    b1 : float
        The major axis beta scaling parameter (radians)
    b2 : float
        The minor axis beta scaling parameter (radians)
    convolve_kern : 2D numpy array
        A kernel to convolve the basis functions with - if modelling a CLEANed image this should be the restoring beam
    shape : tuple
        The 2D shape of the image being modelled, needed if convolving with a kernel

    Returns
    -------
    checked_n1s : array
        The first orders of the basis functions to generate - checks for any
        errors when generating the basis functions and omits those n1,n2 values
    checked_n2s : array
        The second orders of the basis function to generate - checks for any
        errors when generating the basis functions and omits those n1,n2 values
    A_shape_basis : 2D array
        The generated 2D 'A' matrix

    """
    A_shape_basis = zeros((len(xrot),len(n1s)))

    checked_n1s = []
    checked_n2s = []

    ##If the combination of n1,n2 is very high, sometimes the norm calculation
    ##throws an error - in this case, skip that basis function

    with errstate(divide='raise',invalid='raise'):
        for n1, n2 in zip(n1s,n2s):
            try:
                norm = 1.0 / (sqrt(2**n1*factorial(n1))*sqrt(2**n2*factorial(n2)))
                norm *= 1.0 / (b1 * b2)
                checked_n1s.append(n1)
                checked_n2s.append(n2)
            #
            except FloatingPointError:
                print("Skipped n1=%d, n2=%d, b1=%.7f b2=%.7f problem with normalisation factor is too small" %(n1,n2,b1,b2))

    for index,n1 in enumerate(checked_n1s):
        A_shape_basis[:,index] = gen_shape_basis(n1=n1,n2=checked_n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2,convolve_kern=convolve_kern,shape=shape)

    return checked_n1s, checked_n2s, A_shape_basis

class FitShapelets():
    """
    This class takes in FITS data and fitting parameters, and sets up and
    performs shapelet model fitting and generation. The main work-horse of the
    SHAMFI package

    Parameters
    ----------
    fits_data : shamfi.read_FITS_image.FITSInformation instance
        A :class:`FITSInformation` class containing the data to be fit
    shpcoord : shamfi.shapelet_coords.ShapeletCoords instance
        A :class:`ShapeletCoords` class containing the shapelet coordinate system

    """
    def __init__(self,fits_data=False,shpcoord=False):
        if fits_data:
            pass
        else:
            exit('Must provide the FitShapelets class with a shamfi.read_FITS_image.FITSInformation class via the fits_data argument.Exiting now')

        if shpcoord:
            pass
        else:
            exit('Must provide the FitShapelets class with a shamfi.shapelet_coords.ShapeletCoords class via the shpcoord argument.Exiting now')

        self.fits_data = fits_data
        self.shpcoord = shpcoord
        self.data_to_fit = fits_data.flat_data[shpcoord.pixel_inds_to_use]

        ##If a negative mask was generated, apply it. fits_data.negative_pix_mask
        ##is set to use all pixels if negative pixels are to be fit.
        self.data_to_fit = self.data_to_fit[shpcoord.negative_pix_mask]

    # @profile
    def do_grid_search_fit(self, b1_grid, b2_grid, nmax,
                           pa=False, convolve_kern=False,
                           save_FITS=True, save_tag='shapelet'):
        """
        Do a grid search over all b1, b2 values specified in b1_grid, b2_grid,
        fitting all basis functions up to nmax, donig a least squares
        minimisation for each combination of b1, b2 to generate models

        Parameters
        ----------
        b1_grid : array
            A range of major axis beta scaling parameters to fit over
        b2_grid :
            A range of minor axis beta scaling parameters to fit over
        nmax : int
            The maximum order of basis function to generate up to
        pa : float
            If provided, use this postion angle to rotate the basis functions, instead of that found when fitting a Gaussian using shamfi.shapelet_coords.ShapeletCoords.fit_gauss_and_centre_coords
        convolve_kern : 2D numpy array
            If provided, use this convolution kernel instead of the restoring beam of the CLEANed image
        save_FITS: bool
            Save the fitted shapelet model image to a FITS file
        save_tag : string
            A tag to add into the file name to save the plot to

        """

        ##If a different pa is specified use that, otherwise use the pa during
        ##the intial gaussian fit by shpcoord
        if pa:
            pass
        else:
            pa = self.shpcoord.pa

        self.pa = pa

        ##If a specfic convolution kernel is given use that, otherwise
        ##use the restoring beam kernel in fits_data
        if convolve_kern:
            self.convolve_kern = convolve_kern
        else:
            self.convolve_kern = self.fits_data.rest_gauss_kern

        ##Setup up all the possible n1,n2 combinations and append to lists
        self.nmax = nmax

        self.n1s = []
        self.n2s = []

        for n1 in range(self.nmax+1):
            for n2 in range(self.nmax-n1+1):
                self.n1s.append(n1)
                self.n2s.append(n2)

        ##Setup up beta parameters
        self.b1_grid = b1_grid
        self.b2_grid = b2_grid

        self.num_beta_points = len(b1_grid)

        ##Set up ranges of b1,b2 to fit
        self.b_inds = []
        for b1_ind in arange(self.num_beta_points):
            for b2_ind in arange(self.num_beta_points): self.b_inds.append([b1_ind,b2_ind])


        ##Add some containers for the fitting results
        self.residuals_array = zeros((self.num_beta_points,self.num_beta_points))
        self.fitted_data_array = zeros((self.num_beta_points,self.num_beta_points,len(self.data_to_fit)))

        ##The number of coeffs might vary due to quality cuts applied to basis
        ##functions, so store these in a dictionary, as well as reduced n1s,n2s
        ##if we had to skip a basis function
        self.fitted_coeffs_dict = {}
        self.checked_n1s_dict = {}
        self.checked_n2s_dict = {}

        ##Run full nmax fits over range of b1,b2 using progressbar to, well, show progress
        for b1_ind,b2_ind in progressbar(self.b_inds,prefix='Fitting shapelets: '):
            self._do_fitting(b1_ind, b2_ind, self.pa)

        ##When running with self.do_grid_search_fit fitting is automatically done at
        ##100 percent of basis functions. Percentage is written in names of
        ##outputs however so set it here
        self.model_percentage = 100

        self._get_best_params()

        if save_FITS:
            full_model = self._gen_full_model()
            self._save_output_FITS(full_model, save_tag)


    def _do_fitting(self, b1_ind, b2_ind, pa):
        """Takes the data and fits shapelet basis functions to them. Uses the given
        b1, b2 to scale the x,y coords, nmax to create the number of basis functions,
        and the rest_gauss_kern to apply to restoring beam to the basis functions
        """

        ##Select the correct beta params
        b1, b2 = self.b1_grid[b1_ind], self.b2_grid[b2_ind]

        ##Transform ra,dec to shapelet basis function coords by scaling by b1,b2
        xrot,yrot = self.shpcoord.radec2xy(b1, b2, crop=False)

        ##CHANGE CHANGE CHANGE
        ##Generate the basis functions inside the design matrix used to setup
        ##linear equations
        # checked_n1s, checked_n2s, A_shape_basis = self._gen_A_shape_matrix(xrot=xrot, yrot=yrot,
        #                           b1=b1, b2=b2)

        # checked_n1s, checked_n2s, A_shape_basis = gen_A_shape_matrix_direct(self.n1s, self.n2s,
        #                                             xrot, yrot, b1, b2,
        #                                             self.convolve_kern,
        #                                             self.fits_data.data.shape)


        checked_n1s, checked_n2s, A_shape_basis = gen_A_shape_matrix(n1s=self.n1s,
                               n2s=self.n2s, xrot=xrot, yrot=yrot,
                               b1=b1, b2=b2, convolve_kern=self.convolve_kern,
                               shape=self.fits_data.data.shape,
                               pixel_mask=self.shpcoord.pixel_inds_to_use)

        ##Cut out any negative pixels from the fit if required
        A_shape_basis_fit = A_shape_basis[self.shpcoord.negative_pix_mask,:]

        fitted_coeffs = self._linear_solve(flat_data=self.data_to_fit,A_shape_basis=A_shape_basis_fit)
        fitted_data = self._fitted_model(coeffs=fitted_coeffs,A_shape_basis=A_shape_basis_fit)

        resid = self._find_resids(data=self.data_to_fit,fit_data=fitted_data)

        self.residuals_array[b1_ind,b2_ind] = resid
        self.fitted_data_array[b1_ind,b2_ind] = fitted_data.flatten()

        ##The number of coeffs might vary due to quality cuts applied to basis
        ##functions, so store these in a dictionary, as well as reduced n1s,n2s
        ##if we had to skip a basis function
        self.fitted_coeffs_dict['%03d%03d' %(b1_ind,b2_ind)] = fitted_coeffs.flatten()
        self.checked_n1s_dict['%03d%03d' %(b1_ind,b2_ind)] = checked_n1s
        self.checked_n2s_dict['%03d%03d' %(b1_ind,b2_ind)] = checked_n2s


    # def _gen_A_shape_matrix(self,xrot=None,yrot=None,b1=None,b2=None):
    #     '''Setup the A matrix used in the linear least squares fit of the basis functions. Works out all
    #     valid n1,n2 combinations up to nmax'''
    #
    #     ##Make a design matrix array. We will crop out unnecessary pixels when generating
    #     ##basis functions so only make it the size of shpcoord.pixel_inds_to_use
    #     A_shape_basis = zeros((len(self.shpcoord.pixel_inds_to_use),len(self.n1s)))
    #     for index,n1 in enumerate(self.n1s):
    #         A_shape_basis[:,index] = self._gen_shape_basis(n1=n1,n2=self.n2s[index],xrot=xrot,yrot=yrot,b1=b1,b2=b2)
    #
    #         ##TODO can check quality of generated basis functions here and remove if necessary
    #
    #     return self.n1s, self.n2s, A_shape_basis

    # def _gen_shape_basis(self,n1=None,n2=None,xrot=None,yrot=None,b1=None,b2=None):
    #     '''Generates the shapelet basis function for given n1,n2,b1,b2 parameters,
    #     using lookup tables, at the given coords xrot,yrot
    #     b1,b2 should be in radians'''
    #
    #     convolve_kern=self.convolve_kern
    #     shape=self.fits_data.data.shape
    #
    #     ##Ensure n1,n2 are ints
    #     n1 = int(n1)
    #     n2 = int(n2)
    #
    #     basis = gen_shape_basis(n1=n1, n2=n2, xrot=xrot, yrot=yrot, b1=b1, b2=b2,
    #                     convolve_kern=convolve_kern, shape=shape,
    #                     pixel_mask=self.shpcoord.pixel_inds_to_use)
    #
    #
    #     return basis

    # @profile
    def _gen_full_model(self):
        '''Use the best fit coeffs to generate a model image that covers the
        full image area'''

        ##Transform ra,dec to shapelet basis function coords by scaling by b1,b2
        xrot,yrot = self.shpcoord.radec2xy(self.best_b1, self.best_b2, crop=False)

        ##Generate the basis functions inside the design matrix used to setup
        ##linear equations
        _, _, A_shape_basis = gen_A_shape_matrix(n1s=self.fit_n1s, n2s=self.fit_n2s,
                                  xrot=xrot, yrot=yrot, b1=self.best_b1, b2=self.best_b2,
                                  convolve_kern=self.convolve_kern, shape=self.fits_data.data.shape)

        fitted_data = self._fitted_model(coeffs=self.fitted_coeffs,A_shape_basis=A_shape_basis)

        fitted_data.shape = self.fits_data.data.shape

        return fitted_data

    def _linear_solve(self,flat_data=None,A_shape_basis=None):
        '''Fit the image_data using the given A_shape_basis matrix
        Essentially solving for x in the equation Ax = b where:
        A = A_shape_basis
        b = the image data
        x = coefficients for the basis functions in A

        returns: the fitted coefficients in an array'''

        current_shape = flat_data.shape
        ##lstsq is real picky about the data shape cos y'know matrix equations
        flat_data.shape = (len(flat_data),1)

        ##Do the fitting
        shape_coeffs,resid,rank,s = linalg.lstsq(A_shape_basis,flat_data,rcond=None)

        ##Do the fitting
        # shape_coeffs,resid,rank,s = scipy.linalg.lstsq(A_shape_basis,flat_data)
        # lapack_driver='gelsy'

        ##Revert back to original shape
        flat_data.shape = current_shape

        return shape_coeffs

    def _get_best_params(self):
        """
        Find the b1, b2 parameter for the model that left the smallest residuals
        """
        ##Find the minimum point
        best_b1_ind,best_b2_ind = where(self.residuals_array == nanmin(self.residuals_array))

        ##Use minimum indexes to find best fit results
        best_b1_ind,best_b2_ind = best_b1_ind[0],best_b2_ind[0]

        self.best_b1 = self.b1_grid[best_b1_ind]
        self.best_b2 = self.b2_grid[best_b2_ind]
        self.fit_data = self.fitted_data_array[best_b1_ind,best_b2_ind]

        self.fitted_coeffs = self.fitted_coeffs_dict['%03d%03d' %(best_b1_ind,best_b2_ind)]
        self.fit_n1s = self.checked_n1s_dict['%03d%03d' %(best_b1_ind,best_b2_ind)]
        self.fit_n2s = self.checked_n2s_dict['%03d%03d' %(best_b1_ind,best_b2_ind)]

        if self.model_percentage == 100:
            ##These will be used later if compression is to be applied
            self.full_fitted_coeffs = self.fitted_coeffs_dict['%03d%03d' %(best_b1_ind,best_b2_ind)]
            self.full_fit_n1s = array(self.checked_n1s_dict['%03d%03d' %(best_b1_ind,best_b2_ind)])
            self.full_fit_n2s = array(self.checked_n2s_dict['%03d%03d' %(best_b1_ind,best_b2_ind)])
            self.n1s = self.full_fit_n1s
            self.n2s = self.full_fit_n2s

        ##Tell the user what we found in arcmin
        best_b1 = (self.best_b1 / D2R)*60.0
        best_b2 = (self.best_b2 / D2R)*60.0
        print('Best b1 %.2e arcmin b2 %.2e arcmin' %(best_b1,best_b2))

    def _fitted_model(self,coeffs=None,A_shape_basis=None):
        '''Generates the fitted shapelet model for the given coeffs
        and A_shape_basis'''

        return matmul(A_shape_basis,coeffs)

    def _find_resids(self, data=None,fit_data=None):
        '''Just finds the sum of squares of the residuals
        Stops problematic memory errors with matrix algebra class'''

        this_fit = asarray(fit_data).flatten()
        this_data = asarray(data).flatten()
        # print('WTF',data.shape,fit_data.shape)

        diffs = (this_data - this_fit)**2
        return diffs.sum()


    def _save_output_FITS(self, save_data, save_tag):
        '''Saves the model into a FITS file, taking into account any edge padding
        that happened, and converting back into Jy/beam'''

        save_data.shape = self.fits_data.data.shape

        naxis1 = self.fits_data.naxis1
        naxis2 = self.fits_data.naxis2
        edge_pad = self.fits_data.edge_pad
        convert2pixel = self.fits_data.convert2pixel

        with fits.open(self.fits_data.fitsfile) as hdu:
            if len(hdu[0].data.shape) == 2:
                hdu[0].data = save_data[edge_pad:edge_pad+naxis1,edge_pad:edge_pad+naxis2]  / convert2pixel
            elif len(hdu[0].data.shape) == 3:
                hdu[0].data[0,:,:] = save_data[edge_pad:edge_pad+naxis1,edge_pad:edge_pad+naxis2]  / convert2pixel
            elif len(hdu[0].data.shape) == 4:
                hdu[0].data[0,0,:,:] = save_data[edge_pad:edge_pad+naxis1,edge_pad:edge_pad+naxis2]  / convert2pixel

            git_dict = get_gitdict()

            hdu[0].header['SHAMFIv'] = git_dict['describe']
            hdu[0].header['SHAMFId'] = git_dict['date']
            hdu[0].header['SHAMFIb'] = git_dict['branch']

            hdu.writeto('shamfi_%s_nmax%03d_p%03d.fits' %(save_tag,self.nmax,int(self.model_percentage)), overwrite=True)

    def save_srclist(self, save_tag='shapelet', rts_srclist=True, woden_srclist=True, norm_convo=False):
        """
        Uses the best fitted parameters and creates an RTS/WODEN style
        srclist with them, saved as text files

        Parameters
        ----------
        save_tag : string
            A tag to add into the file name to save the plot to
        rts_srclist : bool
            If True, save a sky model compatible with the RTS (Mitchell et al, 2008)
        woden_srclist : bool
            If True, save a sky model compatible with WODEN (Line et al, 2020)
        norm_convo : bool
            If True, apply a further normalisation so that this model can be
            normed during a uv-space based convolution of two shapelet models

        """

        all_flux = sum(self.fit_data)
        print('Total flux in convolved model is %.2f' %all_flux)

        ##This scaling removes pixel effects, and sets the model to sum to one -
        ##this way when the RTS creates the model and multiplies by the reported
        ##flux density we get the correct answer
        scale = 1 / (self.fits_data.pix_area_rad*all_flux)

        if norm_convo:
            print(f"Applying extra normalisation for convolution {self.fit_data.max()} {self.fits_data.solid_beam} {self.fits_data.pix_area_rad}")
            # scale *= self.fits_data.solid_beam / self.fit_data.max()
            scale *= self.fits_data.pix_area_rad

            self.fitted_coeffs *= all_flux
            all_flux = 1.0

        ##Scale to arcmin or deg
        major, minor = (self.best_b1 / D2R)*60, (self.best_b2 / D2R)*60
        pa = self.shpcoord.pa / D2R

        if rts_srclist:
            with open('srclist-rts_%s_nmax%03d_p%03d.txt' %(save_tag,self.nmax,int(self.model_percentage)),'w+') as outfile:
                write_git_header(outfile)
                outfile.write('SOURCE %s %.10f %.10f\n' %(save_tag[:16],self.shpcoord.ra_cent*(R2D/15.0),self.shpcoord.dec_cent*R2D))
                outfile.write("FREQ %.5e %.5f 0 0 0\n" %(self.fits_data.freq,all_flux))
                outfile.write("SHAPELET2 %.8f %.8f %.8f\n" %(pa,major,minor))

                for index,coeff in enumerate(self.fitted_coeffs):
                    outfile.write("COEFF %.1f %.1f %.2f\n" %(self.fit_n1s[index],self.fit_n2s[index],coeff * scale))

                outfile.write('ENDSOURCE\n')

        if woden_srclist:
            with open('srclist-woden_%s_nmax%03d_p%03d.txt' %(save_tag,self.nmax,int(self.model_percentage)),'w+') as outfile:
                write_git_header(outfile)
                outfile.write('SOURCE %s P 0 G 0 S 1 %d\n' %(save_tag,len(self.fitted_coeffs)))
                outfile.write('COMPONENT SHAPELET %.10f %.10f\n' %(self.shpcoord.ra_cent*(R2D/15.0),self.shpcoord.dec_cent*R2D))
                outfile.write("FREQ %.5e %.5f 0 0 0\n" %(self.fits_data.freq,all_flux))
                outfile.write("SPARAMS %.8f %.8f %.8f\n" %(pa,major,minor))

                for index,coeff in enumerate(self.fitted_coeffs):
                    outfile.write("SCOEFF %.1f %.1f %.20f\n" %(self.fit_n1s[index],self.fit_n2s[index],coeff * scale))
                outfile.write('ENDCOMPONENT\n')
                outfile.write('ENDSOURCE\n')

    def find_flux_order_of_basis_functions(self):
        """
        After fitting models, this function orders the fitted basis functions
        by absolution value in image space, to find those that contribute the
        most flux. To do this an 'A' matrix has to be generated to create an
        image

        :ivar array basis_sums: an array containing the sum of the flux of each basis function
        :ivar array sums_order: an array of the argsorted index of the sums of the flux of each basis function
        """
        xrot,yrot = self.shpcoord.radec2xy(self.best_b1, self.best_b2, crop=False)
        # _, _, A_shape_basis = self._gen_A_shape_matrix(xrot=xrot,yrot=yrot,b1=self.best_b1,b2=self.best_b2)
        _, _, A_shape_basis = gen_A_shape_matrix(n1s=self.n1s,
                           n2s=self.n2s, xrot=xrot, yrot=yrot,
                           b1=self.best_b1, b2=self.best_b2, convolve_kern=self.convolve_kern,
                           shape=self.fits_data.data.shape)
                           # pixel_mask=self.shpcoord.pixel_inds_to_use)

        ##Sort the basis functions by highest absolute flux contribution to the model
        self._order_basis_by_flux(A_shape_basis)


    def _order_basis_by_flux(self,A_matrix):
        '''Go through all basis functions values in the A_matrix, multiply by the
        fitted coefficent, sum the asbolute value over all pixels to work out the
        contribution of each basis function to the model'''
        basis_sums = []
        ##For each coefficient, multiply the coeff and basis function for all pixels,
        ##and sum the absolute vales
        for index,coeff in enumerate(self.full_fitted_coeffs):
            val = sum(abs(coeff*A_matrix[:,index]))
            basis_sums.append(val)

        ##Sort them by largest contribution first
        self.basis_sums = array(basis_sums)
        self.sums_order = argsort(basis_sums)[::-1]


    def _compress_by_flux_percentage(self):
        '''Take the abs value of contribution of each basis function to the model
        in basis_sums, and resets self.n1s, self.n2s to only include the basis
        functions that contribute up to percetage defined by self.model_percentage'''

        ##Set a flux limit by taking  a percentage of the full model flux
        flux_lim = (self.model_percentage / 100.0)*sum(self.basis_sums)
        model_flux = 0
        comp_ind = 0
        ##Put absolute values in order, with largest first
        mags = self.basis_sums[self.sums_order]

        ##If no compression, use all values
        ##Shouldn't really get here but play it safe in case
        if float(self.model_percentage) == 100.0:
            comp_ind = len(mags)

        else:
            while model_flux < flux_lim:
                model_flux += mags[comp_ind]

                comp_ind += 1

        order_high = self.sums_order[:comp_ind]
        order_low = self.sums_order[comp_ind:]

        self.n1s = self.full_fit_n1s[order_high]
        self.n2s = self.full_fit_n2s[order_high]

    def do_grid_search_fit_compressed(self, compress_value, save_FITS=True, save_tag='shapelet'):
        """
        Do a grid search over all b1, b2 values for a given compression value.
        Can only be run after fitting the full model, via `self.do_grid_search_fit`
        so all b1,b2,nmax options have already been set and stored internally
        to the class.

        Parameters
        ----------
        compress_value: float
            A value to compress (truncate) the fit results to (percentage, e.g. 80 for 80%)
        save_FITS: bool
            Save the fitted shapelet model image to a FITS file
        save_tag : string
            A tag to add into the file name to save the plot to

        """

        ##As we are compressing based on results from self.do_grid_search_fit,
        ##use the same pa,b1_grid,b2_grid, convolve_kernel. No need to reset
        ##here

        ##Compress the model by the given percentage
        self.model_percentage = compress_value
        self._compress_by_flux_percentage()


        ##Reset some containers for the fitting results
        self.residuals_array = zeros((self.num_beta_points,self.num_beta_points))
        self.fitted_data_array = zeros((self.num_beta_points,self.num_beta_points,len(self.data_to_fit)))

        ##The number of coeffs might vary due to quality cuts applied to basis
        ##functions, so store these in a dictionary, as well as reduced n1s,n2s
        ##if we had to skip a basis function
        self.fitted_coeffs_dict = {}
        self.checked_n1s_dict = {}
        self.checked_n2s_dict = {}

        ##Run full nmax fits over range of b1,b2 using progressbar to, well, show progress
        for b1_ind,b2_ind in progressbar(self.b_inds,prefix='Fitting shapelets: '):
            self._do_fitting(b1_ind, b2_ind, self.pa)

        self._get_best_params()

        if save_FITS:
            full_model = self._gen_full_model()
            self._save_output_FITS(full_model, save_tag)
