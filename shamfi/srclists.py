from __future__ import print_function,division
# from numpy import *
from numpy import abs as np_abs
from astropy.wcs import WCS
from sys import exit
import scipy.optimize as opt
from copy import deepcopy
from scipy.signal import fftconvolve
import os
from subprocess import check_output
import pkg_resources
from shamfi.git_helper import get_gitdict, write_git_header
import numpy as np

##convert between FWHM and std dev for the gaussian function
factor = 2. * np.sqrt(2.*np.log(2.))

##converts between FWHM and std dev for the RTS
rts_factor = np.sqrt(np.pi**2 / (2.*np.log(2.)))

##Use package manager to get hold of the basis functions
basis_path = pkg_resources.resource_filename("shamfi", "image_shapelet_basis.npz")

D2R = np.pi / 180.0

def check_rts_or_woden_get_lines(filename):
    """
    Opens the file at path `filename`, and splits into lines by return carriage.
    Ignores all lines commented with #. Uses the first line of the file to
    determine if this is an RTS or WODEN style srclist

    Parameters
    ----------
    filename: string
        Path to a text file
    Return
    ------
    type: string
        The type of srclist, either "woden" or "rts"
    lines: list
        The lines of the file as string, split by return carriage, in a list

    """
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
    """
    Class to contain an RTS source information

    :ivar str name: Source name
    :ivar list ras: List of all RA for all components in this source
    :ivar list decs: List of all Dec for all components in this source
    :ivar list flux_lines: List to contain list of flux lines for all components in this source
    :ivar list component_infos: List to contain :class:`Component_Info` classes for all components in this source

    """
    def __init__(self):
        self.name = ''
        self.ras = []
        self.decs = []
        self.flux_lines = []
        self.component_infos = []

class Component_Info():
    """
    Class to contain an RTS source information

    :ivar str comp_type: The component type: either POINT, GAUSSIAN, or SHAPELET
    :ivar float pa: Position Angle of the component
    :ivar float major: Major angle of the component
    :ivar float minor: Minor angle of the component
    :ivar list shapelet_coeffs: List to contain lists of shapelet coeffs if the source is a SHAPELET
    """
    def __init__(self):
        self.source_name = None
        self.comp_type = None
        self.pa = None
        self.major = None
        self.minor = None
        self.shapelet_coeffs = []
        self.n1s = []
        self.n2s = []
        # self.shapelet_coeff_values = []
        self.ra = None
        self.dec = None
        self.flux = None
        self.freq = None
        
        self.fluxes = []
        self.freqs = []
        
        self.flux_type = None
        
        self.curve_q = None
        
        self.SI = -0.8

    def calc_flux(self, freq):
        """Return the flux of the component at the given frequency
        by scaling via the spectral index"""

        flux = self.flux * (freq/self.freq)**self.SI
        return flux

def get_RTS_sources(srclist, all_RTS_sources):
    """
    Takes a path to an RTS srclist, breaks it up into SOURCES, populates
    this information into RTS_source classes and appends them to all_RTS_sources
    list

    Parameters
    ----------
    srclist : string
        Path to a texfile of an RTS srclist
    all_RTS_sources : list
        All sources found in `srclist` are used to populate an :class:`RTS_source`
        class, and appended to `all_RTS_sources`

    Return
    ------
    all_RTS_sources: list of :class:`RTS_source`
        The original `all_RTS_sources` list, with any new :class:`RTS_source` s appended

    """

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
            ##If there are components to the source, see np.where the components start and end
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
    """
    Take in a number of text lines in the WODEN format and writes them out
    to `outfile` in the RTS format

    Parameters
    ----------
    lines : list
        A list of strings, each a line from a WODEN style srclist
    outfile : open(filename) instance
        An opened textfile to write outputs to
    name : string
        If a name is supplied, this is the first component of a SOURCE, which
        requires extra formatting in an RTS srclist

    """


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
            outfile.write('GAUSSIAN %s %.10f %.10f\n' %(pa,float(major)*rts_factor,float(minor)*rts_factor))
        if 'SPARAMS' in line:
            _,pa,major,minor = line.split()
            outfile.write('SHAPELET2 %s %s %s\n' %(pa,major,minor))
        if 'SCOEFF' in line:
            outfile.write(line[1:]+'\n')

    if name:
        pass
    else:
        outfile.write('ENDCOMPONENT\n')

def write_woden_from_RTS_sources(RTS_sources,outname):
    """
    Takes a list of :class:`RTS_source` classes and uses the to write a WODEN
    style srclist called `outname`

    Parameters
    ----------
    RTS_sources : list
        A list of :class:`RTS_source` s to write out to a WODEN style srclist
    outname : string
        Path to save the output WODEN srclist text file to

    """

    all_comp_types = []
    all_shape_coeffs = 0
    for source in RTS_sources:
        for comp in source.component_infos:
            all_comp_types.append(comp.comp_type)
            all_shape_coeffs += len(comp.shapelet_coeffs)

    all_comp_types = np.array(all_comp_types)

    num_point = len(np.where(all_comp_types == 'POINT')[0])
    num_gauss = len(np.where(all_comp_types == 'GAUSSIAN')[0])
    num_shape = len(np.where(all_comp_types == 'SHAPELET')[0])

    with open(outname,'w+') as outfile:
        write_git_header(outfile)
        outfile.write('SOURCE %s P %d G %d S %d %d\n' %(RTS_sources[0].name,
                                num_point,num_gauss,num_shape,all_shape_coeffs))

        for source in RTS_sources:
            for comp_ind,comp_info in enumerate(source.component_infos):
                outfile.write('COMPONENT %s %s %s\n' %(comp_info.comp_type,source.ras[comp_ind],source.decs[comp_ind]))

                for line in source.flux_lines[comp_ind]:
                    outfile.write(line+'\n')

                if comp_info.comp_type == 'GAUSSIAN':
                    ##RTS gaussians are std dev, WODEN are FWHM
                    outfile.write('GPARAMS %s %.10f %.10f\n' %(comp_info.pa,float(comp_info.major)/rts_factor,float(comp_info.minor)/rts_factor))

                elif comp_info.comp_type == 'SHAPELET':
                    outfile.write('SPARAMS %s %s %s\n' %(comp_info.pa,comp_info.major,comp_info.minor))

                    for line in comp_info.shapelet_coeffs:
                        outfile.write('S'+line+'\n')
                outfile.write('ENDCOMPONENT\n')

        outfile.write('ENDSOURCE')

def write_singleRTS_from_RTS_sources(RTS_sources,outname,name='combined_name'):
    '''Takes a list RTS_sources containg RTS_source classes, and writes
    them out into a single SOURCE RTS srclist of name outname'''

    """
    Takes a list of :class:`RTS_source` classes and uses the to write a RTS
    style srclist called `outname`, combining all sources in `RTS_sources` into
    a single :class:`RTS_source`

    Parameters
    ----------
    RTS_sources : list
        A list of :class:`RTS_source` s to write out to a WODEN style srclist
    outname : string
        Path to save the output WODEN srclist text file to
    name : string
        Name for the single output RTS source

    """

    with open(outname,'w+') as outfile:
        write_git_header(outfile)

        for source_ind,source in enumerate(RTS_sources):
            for comp_ind,comp_info in enumerate(source.component_infos):
                if source_ind == 0 and comp_ind == 0:
                    outfile.write('SOURCE %s %s %s\n' %(name,source.ras[comp_ind],source.decs[comp_ind]))
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

def read_woden_srclist(srclist):
    """Reads in information from WODEN-type text srclist

    Parameters
    ----------
    srclist : string
        Path to a text file of a WODEN srclist

    Return
    ------
    components: np.array of :class:`Component_Info`
        An np.array containing `Component_Info` classes of all components in the
        WODEN-style srclist


    """

    with open(srclist,'r') as srcfile:
        source_chunks = srcfile.read().split('ENDSOURCE')
        del source_chunks[-1]

        # lines = source_src_info[0].split('\n')

    components = []


    for source_chunk in source_chunks:

        comp_chunks = source_chunk.split('ENDCOMPONENT')

        for comp_chunk in comp_chunks:

            lines = comp_chunk.split('\n')

            if lines == ['', '']:
                pass
            else:
                component = Component_Info()

                for line in lines:
                    if 'COMPONENT' in line and 'END' not in line:
                        _, comp_type, ra, dec = line.split()

                        component.dec = float(dec)*D2R
                        component.ra = float(ra)*15.0*D2R
                        component.comp_type = comp_type


                    elif 'SPARAMS' in line or 'GPARAMS' in line:
                        _, pa, major, minor = line.split()

                        component.pa = float(pa)*D2R
                        component.major = float(major) * (D2R / 60.0)
                        component.minor = float(minor) * (D2R / 60.0)

                    elif 'SCOEFF' in line:
                        _, n1, n2, coeff = line.split()

                        component.n1s.append(float(n1))
                        component.n2s.append(float(n2))
                        component.shapelet_coeffs.append(float(coeff))

                    elif 'FREQ' in line:
                        _, freq, flux, _, _, _ = line.split()
                        component.freq = float(freq)
                        component.flux = float(flux)
                    elif 'LINEAR' in line:
                        _, freq, flux, _, _, _, SI = line.split()
                        component.freq = float(freq)
                        component.flux = float(flux)
                        component.SI = float(SI)


                component.n1s = np.array(component.n1s)
                component.n2s = np.array(component.n2s)
                component.shapelet_coeffs = np.array(component.shapelet_coeffs)

                components.append(component)

    return np.array(components)


def read_hyperdrive_srclist(srclist):
    """Reads in sources from a hyperdrive srclist and returns """

    with open(srclist) as file:

        # current_source = 0

        components = []
        sources = []

        component = False
        source_name = False
        current_source = 0

        source_indexes = []
        
        freq_count = 0
        freq_indent = 0

        for line in file:
            if line != '---\n' and '#' not in line and line != ''  and line != ' ' and line != '\n':

                if line[0] != ' ':
                    # print(current_source)
                    source_name = line[:-2]
                    current_source += 1

                elif 'ra:' in line:

                    ##ra should be the first thing in a component, so we need
                    ##to append all the previously found values and reset the
                    ##counters

                    ##If a previous component exists, append in to the list
                    ##of all components, and then make a new one
                    if component:
                        ##Make some things into np.arrays so we can maths them
                        component.n1s = np.array(component.n1s)
                        component.n2s = np.array(component.n2s)
                        component.shapelet_coeffs = np.array(component.shapelet_coeffs)
                        components.append(component)

                    component = Component_Info()
                    freq_count = 0
                    
                    component.source_name = source_name
                    component.ra = float(line.split()[-1])*D2R

                    # print(component.ra)
                    source_indexes.append(current_source)

                elif 'dec:' in line:
                    component.dec = float(line.split()[-1])*D2R

                elif 'comp_type: point' in line:
                    component.comp_type = 'POINT'
                elif 'gaussian:' in line:
                    component.comp_type = 'GAUSSIAN'
                elif 'shapelet:' in line:
                    component.comp_type = 'SHAPELET'
                    
                elif "maj:" in line:
                    component.major = float(line.split()[-1])*(D2R / 3600.0)
                elif "min:" in line:
                    component.minor = float(line.split()[-1])*(D2R / 3600.0)
                elif "pa:" in line:
                    component.pa = float(line.split()[-1])*D2R

                elif 'n1:' in line:
                    component.n1s.append(float(line.split()[-1]))
                elif 'n2:' in line:
                    component.n2s.append(float(line.split()[-1]))
                elif 'value:' in line:
                    component.shapelet_coeffs.append(float(line.split()[-1]))
                    
                elif 'power_law:' in line:
                    component.flux_type = 'POWER'
                elif 'curved_power_law:' in line:
                    component.flux_type = 'CURVE'
                    ##TODO read in curved stuff properly
                    
                elif 'si:' in line:
                    component.SI = float(line.split()[-1])
                    
                elif 'list:' in line:
                    component.flux_type = 'LIST'

                elif 'freq:' in line:
                    freq_count += 1
                    component.freqs.append(float(line.split()[-1]))
                    
                    ##Stick in an empty np.array for Stokes I,Q,U,V
                    component.fluxes.append(np.array([0.0, 0.0, 0.0, 0.0]))
                    
                    ##See what indent this freq entry starts at - used to
                    ##line up following freq entries, as `q` can either mean
                    ##stokes Q or q curvature param
                    freq_indent = line.index('f')
                    
                    
                elif ' i:' in line:
                    component.fluxes[freq_count - 1][0] = float(line.split()[-1])
                    
                ##Gotta be fancy here to work out if this is a Stokes Q or a 
                ##curved power law 'q' param
                elif ' q:' in line:
                    q = float(line.split()[-1])
                    if line.index('q') == freq_indent:
                        component.fluxes[freq_count - 1][1] = q
                    else:
                        if component.flux_type == 'CURVE':
                            component.curve_q = q
                            
                elif ' u:' in line:
                    component.fluxes[freq_count - 1][2] = float(line.split()[-1])
                    
                elif ' v:' in line:
                    component.fluxes[freq_count - 1][3] = float(line.split()[-1])
                    
    return np.array(components)

def extrapolate_component_flux(component : Component_Info, extrap_freqs):
    """Extrapolate the fluxe to a given frequency. Currently just does
    Stokes I, can be expanded to do Q,U,V as well"""
    
    extrap_stokesI = False
    
    if type(extrap_freqs) == float:
        extrap_freqs = np.array([extrap_freqs])
        
    if component.flux_type == 'POWER':

        flux_ratio = (extrap_freqs / component.freqs[0])**component.SI

        extrap_stokesI = component.fluxes[0][0]*flux_ratio
        extrap_stokesQ = component.fluxes[0][1]*flux_ratio
        extrap_stokesU = component.fluxes[0][2]*flux_ratio
        extrap_stokesV = component.fluxes[0][3]*flux_ratio


    elif component.flux_type == 'CURVE':
        q = component.curve_q
        
        si_ratio = (extrap_freqs / component.freqs[0])**component.SI

        exp_ratio = np.exp(q*np.log(extrap_freqs)**2) / np.exp(q*np.log(component.freqs[0])**2)

        extrap_stokesI = component.fluxes[0][0]*exp_ratio
        extrap_stokesQ = component.fluxes[0][1]*exp_ratio
        extrap_stokesU = component.fluxes[0][2]*exp_ratio
        extrap_stokesV = component.fluxes[0][3]*exp_ratio



    elif component.flux_type == 'LIST':
        
        extrap_stokesI = np.empty(len(extrap_freqs))
        all_fluxes = np.array(component.fluxes)
        ref_freqs = component.freqs
        
        ##just use stokes I
        ref_fluxes = all_fluxes[:, 0]
        
        for find, extrap_freq in enumerate(extrap_freqs):
            ##If there is only one freq entry, use power law model
            if len(ref_freqs) == 1:
                
                flux_ratio = (extrap_freq / ref_freqs[0])**component.SI
                extrap_stokesI[find] = ref_fluxes[0]*flux_ratio
            else:

                # ##Happen to be extrapolating to a reference frequency
                if extrap_freq in ref_freqs:
                    extrap_flux = ref_fluxes[np.where(ref_freqs == extrap_freq)][0]
                else:

                    freq_diffs = ref_freqs - extrap_freq

                    low_ind_1 = -1.0

                    low_val_1 = 1e16
                    # low_val_2 = 1e16

                    for ind in np.arange(len(ref_freqs)):
                        abs_diff = abs(freq_diffs[ind])

                        if abs_diff < low_val_1:
                            low_val_1 = abs_diff
                            low_ind_1 = ind

                    ##Closest frequency is the lowest
                    if low_ind_1 == 0:
                        low_ind_2 = 1
                    ##Closest frequency is the highest
                    elif low_ind_1 == len(ref_freqs) - 1:
                        low_ind_2 = low_ind_1 - 1
                    ##otherwise, choose either above or below
                    else:
                        ##closest freq is higher than desired
                        if ref_freqs[low_ind_1] > extrap_freq:
                            low_ind_2 = low_ind_1 - 1
                        else:
                            low_ind_2 = low_ind_1 + 1

                    if ref_fluxes[low_ind_1] <= 0 or ref_fluxes[low_ind_2] <= 0:

                        gradient = (ref_fluxes[low_ind_2] - ref_fluxes[low_ind_1]) / (ref_freqs[low_ind_2] - ref_freqs[low_ind_1])
                        extrap_flux =  ref_fluxes[low_ind_1] + gradient*(extrap_freq - ref_freqs[low_ind_1])

                    else:

                        flux1 = np.log10(ref_fluxes[low_ind_1])
                        flux2 = np.log10(ref_fluxes[low_ind_2])
                        freq1 = np.log10(ref_freqs[low_ind_1])
                        freq2 = np.log10(ref_freqs[low_ind_2])
                        extrap_freq = np.log10(extrap_freq)

                        gradient = (flux2 - flux1) / (freq2 - freq1)
                        extrap_flux =  flux1 + gradient*(extrap_freq - freq1)

                        extrap_flux = 10**extrap_flux
                
                    extrap_stokesI[find] = extrap_flux

    if len(extrap_stokesI) == 1:
        extrap_stokesI = extrap_stokesI[0]

    return extrap_stokesI