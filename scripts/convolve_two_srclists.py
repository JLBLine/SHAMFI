import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from copy import deepcopy


##Convert degress to radians
D2R = np.pi/180.
##Convert radians to degrees
R2D = 180./np.pi
##convert between FWHM and std dev for the gaussian function
FWHM_factor = 2. * np.sqrt(2.*np.log(2.))
# print(FWHM_factor, 2.35482004503)

SOLAR2SIDEREAL = 1.00274
DS2R = 7.2722052166430399038487115353692196393452995355905e-5

def get_L_value(big_L_array, l, m, n, nmax):
    """
    For the given l,m,n values, either calculate L or try and look up the
    values from big_L_dict
    """

    if (l + m + n) == 0:
        this_L = 1.0

    elif (l + m + n) == 1 or (l + m + n) < 0:
        this_L = 0.0

    else:
        index = get_lmn_index(l, m, n, nmax)
        this_L = big_L_array[index]

    return this_L

def get_the_lmn_minus(big_L_array, l, m, n, nmax):
    """
    Either calculate or lookup what previous vales of L are

    Here l_minus = L_(l-1, m, n)
    Here m_minus = L_(l, m-1, n)
    Here n_minus = L_(l, m, n-1)

    If l + m + n = 0, L is one
    If l + m + n = odd, L is zero

    If l == 0 we can ignore l_minus (it'll be multipled by zero)
    If m == 0 we can ignore m_minus (it'll be multipled by zero)
    If n == 0 we can ignore n_minus (it'll be multipled by zero)

    Just return a zero in these last cases
    """

    if l == 0:
        l_minus = 0.0
    else:
        l_minus = get_L_value(big_L_array, l-1, m, n, nmax)

    if m == 0:
        m_minus = 0.0
    else:
        m_minus = get_L_value(big_L_array, l, m-1, n, nmax)

    if n == 0:
        n_minus = 0.0
    else:
        n_minus = get_L_value(big_L_array, l, m, n-1, nmax)

    return l_minus, m_minus, n_minus

def calc_l_plus_one(big_L_array, l_plus_one, m, n, nmax, a, b, c):
    """
    Right the big_L is L_(l,m,n)

    Using previous values of L in big_L_array, calculate L_(l+1, m, n)

    Here l_minus = L_(l-1, m, n)
    Here m_minus = L_(l, m-1, n)
    Here n_minus = L_(l, m, n-1)

    """

    l = l_plus_one - 1

    l_minus, m_minus, n_minus = get_the_lmn_minus(big_L_array, l, m, n, nmax)

    l_plus_one = 2*l*(a**2 - 1)*l_minus + 2*m*a*b*m_minus + 2*n*a*c*n_minus

    index = get_lmn_index(l + 1, m, n, nmax)
    big_L_array[index] = l_plus_one

    return big_L_array

def calc_m_plus_one(big_L_array, l, m_plus_one, n, nmax, a, b, c):
    """
    Right the big_L is L_(l,m,n)

    Using previous values of L in big_L_array, calculate L_(l, m + 1, n)

    Here l_minus = L_(l-1, m, n)
    Here m_minus = L_(l, m-1, n)
    Here n_minus = L_(l, m, n-1)

    """

    m = m_plus_one - 1

    l_minus, m_minus, n_minus = get_the_lmn_minus(big_L_array, l, m, n, nmax)

    m_plus_one = 2*m*(b**2 - 1)*m_minus + 2*l*a*b*l_minus + 2*n*b*c*n_minus

    index = get_lmn_index(l, m + 1, n, nmax)
    big_L_array[index] = m_plus_one

    return big_L_array

def calc_n_plus_one(big_L_array, l, m, n_plus_one, nmax, a, b, c):
    """
    Right the big_L is L_(l,m,n)

    Using previous values of L in big_L_array, calculate L_(l, m, n + 1)

    Here l_minus = L_(l-1, m, n)
    Here m_minus = L_(l, m-1, n)
    Here n_minus = L_(l, m, n-1)

    """

    n = n_plus_one - 1

    l_minus, m_minus, n_minus = get_the_lmn_minus(big_L_array, l, m, n, nmax)

    n_plus_one = 2*n*(c**2 - 1)*n_minus + 2*m*c*b*m_minus + 2*l*a*c*l_minus

    index = get_lmn_index(l, m, n + 1, nmax)
    big_L_array[index] = n_plus_one

    return big_L_array

def get_n1n2_index(n1, n2, nmax):
    """For a complete sequence of all n1, n2 pairs with some nmax, find the
    index of the pair in a 1D array"""

    return int(n2 + n1*(nmax -n1/2 + 1.5))

def get_num_shape_basis(nmax):
    """For a given nmax, return the number of possible 2D shapelet basis functions"""

    return int(((nmax + 2)*(nmax + 1))/2)


def calc_n1n2_pairs(nmax):
    """Calculates all the n1,n2 pairs used up to maximum basis order `nmax`"""

    num_basis = get_num_shape_basis(nmax)

    n1s = np.empty(num_basis)
    n2s = np.empty(num_basis)

    count = 0

    for n1 in range(nmax + 1):
        for n2 in range(nmax - n1 + 1):

            index = get_n1n2_index(n1, n2, nmax)

            n1s[int(index)] = n1
            n2s[int(index)] = n2

            count += 1

    return n1s, n2s


def get_lmn_index(l, m, n, nmax):
    """For maximum order nmax, find the index of C_lmn or B_lmn
    for a given l,m,n"""

    index = l*(nmax+1)*(nmax+1) + m*(nmax+1) + n

    return index

def calc_Blmn_array(nmax, a, b, c):

    Blmn_array = np.empty((nmax+1)**3)
    big_L_array = np.empty((nmax+1)**3)

    nu = np.sqrt(1 / (a**(-2) + b**(-2) + c**(-2)))

    a_scale = np.sqrt(2)*(nu / a)
    b_scale = np.sqrt(2)*(nu / b)
    c_scale = np.sqrt(2)*(nu / c)

    for l in range(nmax + 1):
        for m in range(nmax + 1):
            for n in range(nmax + 1):
                index = get_lmn_index(l, m, n, nmax)

                if l == 0 and m == 0 and n == 0:
                    big_L_array[index] = 1.0
                else:
                    if l > m and l > n:
                        big_L_array = calc_l_plus_one(big_L_array, l, m, n, nmax,
                                                     a_scale, b_scale, c_scale)
                        # print("calc_l_plus_one", index, big_L_array[index])
                    elif m > l and m > n:
                        big_L_array = calc_m_plus_one(big_L_array, l, m, n, nmax,
                                                     a_scale, b_scale, c_scale)
                        # print("calc_m_plus_one", index, big_L_array[index])
                    elif n > l and n > m:
                        big_L_array = calc_n_plus_one(big_L_array, l, m, n, nmax,
                                                     a_scale, b_scale, c_scale)
                        # print("calc_n_plus_one", index, big_L_array[index])
                    elif l > m:
                        big_L_array = calc_l_plus_one(big_L_array, l, m, n, nmax,
                                                     a_scale, b_scale, c_scale)
                        # print("calc_l_plus_one", index, big_L_array[index])
                    elif l > n:
                        big_L_array = calc_l_plus_one(big_L_array, l, m, n, nmax,
                                                     a_scale, b_scale, c_scale)
                        # print("calc_l_plus_one", index, big_L_array[index])
                    else:
                        big_L_array = calc_m_plus_one(big_L_array, l, m, n, nmax,
                                                     a_scale, b_scale, c_scale)
                        # print("calc_m_plus_one", index, big_L_array[index])



                norm = nu / np.sqrt(2**(l+m+n-1)*np.sqrt(np.pi)*factorial(l)*factorial(m)*factorial(n)*a*b*c)
                Blmn_array[index] = norm*big_L_array[index]


    return Blmn_array


# @profile
def calc_Clmn(l1, m1, n1, l2, m2, n2, nmax, B1lmn_array, B2lmn_array):
    """Using a precalculated Blmn_arrays, calculate the 2D convolutional tensor
    maybe"""

    Bl1m1n1 = B1lmn_array[get_lmn_index(l1, m1, n1, nmax)]
    Bl2m2n2 = B2lmn_array[get_lmn_index(l2, m2, n2, nmax)]

    norm = (-1)**(l1+m1+n1)*(-1)**(l2+m2+n2)

    Clmn = norm*Bl1m1n1*Bl2m2n2

    # if np.abs(Clmn) > 1e-4:
    #     print(f"{l1,m1,n1} {l2,m2,n2} {Bl1m1n1:.5e} {Bl2m2n2:.5e} {norm:.3e}")

    return Clmn


def calc_Clmn_vector(l1s, m1s, n1s, l2s, m2s, n2s, nmax, B1lmn_array, B2lmn_array):
    """Using a precalculated Blmn_arrays, calculate the 2D convolutional tensor
    maybe"""

    indexes1 = get_lmn_index(l1s, m1s, n1s, nmax)
    indexes2 = get_lmn_index(l2s, m2s, n2s, nmax)

    # indexes1 = np.array([int(get_lmn_index(l, m, n, nmax)) for l,m,n in zip(l1s, m1s, n1s)])
    # indexes2 = np.array([int(get_lmn_index(l, m, n, nmax)) for l,m,n in zip(l2s, m2s, n2s)])

    Bl1m1n1 = B1lmn_array[indexes1.astype(int)]
    Bl2m2n2 = B2lmn_array[indexes2.astype(int)]

    norm = (-1)**(l1s+m1s+n1s)*(-1)**(l2s+m2s+n2s)

    # print(norm)

    Clmns = norm*Bl1m1n1*Bl2m2n2

    # if np.abs(Clmn) > 1e-4:
    #     print(f"{l1,m1,n1} {l2,m2,n2} {Bl1m1n1:.5e} {Bl2m2n2:.5e} {norm:.3e}")

    # print(Clmns)

    return Clmns

# @profile
def convolve_two_shapelet_model_coeffs(coeffs1, coeffs2, nmax,
                                       alpha_1, alpha_2,
                                       beta_1, beta_2,
                                       gamma_1, gamma_2):

    # print("Making B1lmn_array==================================================")
    B1lmn_array = calc_Blmn_array(nmax, 1/gamma_1, 1/alpha_1, 1/beta_1)
    # print("Making B2lmn_array==================================================")
    B2lmn_array = calc_Blmn_array(nmax, 1/gamma_2, 1/alpha_2, 1/beta_2)

    l1s, l2s = calc_n1n2_pairs(nmax)
    m1s, m2s = calc_n1n2_pairs(nmax)
    n1s, n2s = calc_n1n2_pairs(nmax)

    combined_coeffs = np.empty(get_num_shape_basis(nmax)) #, dtype=complex)

    basis_indexes = [get_n1n2_index(n1, n2, nmax) for n1, n2 in zip(n1s, n2s)]
    for l1, l2 in zip(l1s, l2s):
        print(f"Doing {l1} {l2} basis")
        # sum_coeff = 0.0
        # for m_index, m1, m2 in zip(basis_indexes, m1s, m2s):
        #     for n_index, n1, n2 in zip(basis_indexes, n1s, n2s):
        #
        #         # coeff1 = coeffs1[get_n1n2_index(m1, m2, nmax)]
        #         # coeff2 = coeffs2[get_n1n2_index(n1, n2, nmax)]
        #
        #         coeff1 = coeffs1[m_index]
        #         coeff2 = coeffs2[n_index]
        #
        #         # if coeff1 != 0 and coeff2 != 0 and (l1 + m1 + n1) % 2 == 0 and (l2 + m2 + n2) % 2 == 0:
        #
        #             # print(coeff1, coeff2, l1, m1, n1, l2, m2, n2)
        #
        #         Clmn = calc_Clmn(int(l1), int(m1), int(n1), int(l2), int(m2), int(n2),
        #                          nmax, B1lmn_array, B2lmn_array)
        #
        #         # print(Clmn)
        #         # print(coeff1)
        #         # print(coeff2)
        #
        #         print(Clmn*coeff1*coeff2)
        #
        #         sum_coeff += Clmn*coeff1*coeff2

        coeff1_indexes = np.repeat(basis_indexes, len(basis_indexes))
        coeff2_indexes = np.tile(basis_indexes, len(basis_indexes))

        nonzero = np.where((coeffs1[coeff1_indexes] != 0) & (coeffs2[coeff2_indexes] != 0))

        # nonzero = np.arange(len(coeff1_indexes))

        all_m1s = np.repeat(m1s, len(basis_indexes))[nonzero]
        all_m2s = np.repeat(m2s, len(basis_indexes))[nonzero]

        all_n1s = np.tile(n1s, len(basis_indexes))[nonzero]
        all_n2s = np.tile(n2s, len(basis_indexes))[nonzero]

        all_l1s = np.full(len(basis_indexes)**2, l1)[nonzero]
        all_l2s = np.full(len(basis_indexes)**2, l2)[nonzero]

        all_coeff1s =  coeffs1[coeff1_indexes][nonzero]
        all_coeff2s =  coeffs2[coeff2_indexes][nonzero]

        # even = np.where((all_l1s+all_m1s+all_n1s % 2 == 0) & (all_l2s+all_m2s+all_n2s % 2 == 0))
        #
        # all_m1s = all_m1s[even]
        # all_m2s = all_m2s[even]
        # all_n1s = all_n1s[even]
        # all_n2s = all_n2s[even]
        # all_l1s = all_l1s[even]
        # all_l2s = all_l2s[even]
        # all_coeff1s = all_coeff1s[even]
        # all_coeff2s = all_coeff2s[even]


        # for m_index, m1, m2 in zip(basis_indexes, m1s, m2s):
        #     for n_index, n1, n2 in zip(basis_indexes, n1s, n2s):


        Clmns = calc_Clmn_vector(all_l1s, all_m1s, all_n1s, all_l2s, all_m2s, all_n2s,
                                 nmax, B1lmn_array, B2lmn_array)


        # print(Clmns)
        # print(all_coeff1s)
        # print(all_coeff2s)

        # print(Clmns*all_coeff1s*all_coeff2s)

        sum_coeff = np.sum(Clmns*all_coeff1s*all_coeff2s)


        combined_coeffs[get_n1n2_index(l1, l2, nmax)] = sum_coeff

    return n1s, n2s, combined_coeffs


def read_woden_srclist(srclist):
    """Reads in a shapelet models from a WODEN type srclist"""

    with open(srclist,'r') as srcfile:
        source_src_info = srcfile.read().split('ENDSOURCE')
        del source_src_info[-1]

        lines = source_src_info[0].split('\n')

    n1s = []
    n2s = []
    coeffs = []

    for line in lines:
        if 'COMPONENT' in line and 'END' not in line:
            _, _, ra, dec = line.split()

            ra_cent = float(ra)*15.0*D2R
            dec_cent = float(dec)*D2R


        elif 'SPARAMS' in line:
            _, pa, bmaj, bmin = line.split()

            pa = float(pa)*D2R
            bmaj = float(bmaj) * (D2R / 60.0)
            bmin = float(bmin) * (D2R / 60.0)

        elif 'SCOEFF' in line:
            _, n1, n2, coeff = line.split()

            n1s.append(float(n1))
            n2s.append(float(n2))

            # conv = (2*np.sqrt(bmaj)*np.sqrt(bmin)) / np.pi
            # conv = 1 / conv
            conv = 1

            coeffs.append(float(coeff)*conv)

        elif 'FREQ' in line:
            _, freq, flux, _, _, _ = line.split()

        elif 'LINEAR' in line:
            _, freq, flux, _, _, _, SI = line.split()

    return ra_cent, dec_cent, pa, bmaj, bmin, np.array(n1s), np.array(n2s), np.array(coeffs), float(freq), float(flux)


def fill_basis_up_to_nmax(n1s, n2s, coeffs, nmax):
    """To do the convolution, easiest to loop over all n1,n2 pairs up to
    some nmax. Remake n1, n2, and coeffs to hold zeros if the pair is
    missing, and make sure indexing works as expected"""

    num_coeffs = get_num_shape_basis(nmax)

    new_coeffs = np.zeros(num_coeffs) #, dtype=complex)

    for old_ind, n1, n2 in zip(np.arange(len(n1s)), n1s, n2s):
        new_ind = get_n1n2_index(n1, n2, nmax)
        new_coeffs[new_ind] = coeffs[old_ind]

    return new_coeffs

def write_srclist(n1s, n2s, coeffs, ra, dec, flux, freq, filename,
                  pa, b1, b2):

    with open(filename, 'w') as outfile:

        outfile.write(f'SOURCE convolved_shape P 0 G 0 S 1 {len(n1s)}\n')

        outfile.write(f'COMPONENT SHAPELET {ra/(15.0*D2R):.10f} {dec/D2R:.10f}\n')
        outfile.write(f'FREQ {freq:.6e} 1.0 0 0 0\n')
        outfile.write(f'SPARAMS 0 {b1 * (60./D2R):.5f} {b2 * (60./D2R):.5f}\n')

        for n1, n2, coeff in zip(n1s, n2s, coeffs):

            if np.real(coeff) == 0:
                pass
            else:
                outfile.write(f'SCOEFF {int(n1):d} {int(n2):d} {flux*np.real(coeff):.10f}\n')

        outfile.write('ENDCOMPONENT\n')
        outfile.write('ENDSOURCE')

# @profile
def main(args):

    f_ra_cent, f_dec_cent, f_pa, f_b1, f_b2, m1s, m2s, f_coeffs, f_freq, f_flux = read_woden_srclist(args.srclist1)
    g_ra_cent, g_dec_cent, g_pa, g_b1, g_b2, l1s, l2s, g_coeffs, g_freq, g_flux = read_woden_srclist(args.srclist2)

    if f_ra_cent != g_ra_cent or f_dec_cent != g_dec_cent:
        exit("RA,Dec coords between two srclist much match"
             f"{f_ra_cent/D2R,f_dec_cent/D2R} != {g_ra_cent/D2R,g_dec_cent/D2R}")

    lmax = int(np.max(l1s+l2s))
    mmax = int(np.max(m1s+m2s))

    nmax = lmax + mmax

    f_coeffs = fill_basis_up_to_nmax(m1s, m2s, f_coeffs, nmax)
    g_coeffs = fill_basis_up_to_nmax(l1s, l2s, g_coeffs, nmax)

    uv_conv_beta_1 = np.sqrt((1/f_b1)**2 + (1/g_b1)**2)
    uv_conv_beta_2 = np.sqrt((1/f_b2)**2 + (1/g_b2)**2)

    n1s, n2s, uv_convolve_coeffs = convolve_two_shapelet_model_coeffs(f_coeffs, g_coeffs, nmax,
                                           1 / f_b1, 1 / f_b2,
                                           1 / g_b1, 1 / g_b2,
                                           uv_conv_beta_1, uv_conv_beta_2)


    ## The stored SHAMFI coeffs have a different norm to those in the
    ##literature, so that the GPU code that generates the uv-shapelets from
    ##a lookup table has less operations. This swaps between the literature
    ##norm which is used in the convolution method, to the norm used by
    ##SHAMFI

    def conversion_fac(b1, b2):
        """Converts from the literature shapelet normalisaiton to the SHAMFI
        loopup table normalisations"""

        return (2*np.sqrt(b1)*np.sqrt(b2)) / np.pi


    conv_beta_1 = 1 / uv_conv_beta_1
    conv_beta_2 = 1 / uv_conv_beta_2

    coeff_norm = conversion_fac(conv_beta_1, conv_beta_2) / (conversion_fac(f_b1, f_b2)*conversion_fac(g_b1, g_b2))

    uv_convolve_coeffs *= coeff_norm

    flux = f_flux*g_flux

    if args.output_name[-4:] == '.txt':
        filename = args.output_name
    else:
        filename = args.output_name + ".txt"

    write_srclist(n1s, n2s, uv_convolve_coeffs, f_ra_cent, f_dec_cent, flux, f_freq,
                  filename, 0.0, 1.0 / uv_conv_beta_1, 1.0 / uv_conv_beta_2)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="I dunno")

    parser.add_argument('--srclist1', default=False,
        help='Path to first srclist to convolve')

    parser.add_argument('--srclist2', default=False,
        help='Path to second srclist to convolve')

    parser.add_argument('--output_name', default=False,
        help='Name for the output srclist')

    args = parser.parse_args()

    main(args)
