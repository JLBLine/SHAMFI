from __future__ import print_function,division
from numpy import *
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

##convert between FWHM and std dev for the gaussian function
factor = 2. * sqrt(2.*log(2.))

##converts between FWHM and std dev for the RTS
rts_factor = sqrt(pi**2 / (2.*log(2.)))

##Use package manager to get hold of the basis functions
basis_path = pkg_resources.resource_filename("shamfi", "image_shapelet_basis.npz")

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
        self.comp_type = None
        self.pa = None
        self.major = None
        self.minor = None
        self.shapelet_coeffs = []

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
            ##If there are components to the source, see where the components start and end
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

    all_comp_types = array(all_comp_types)

    num_point = len(where(all_comp_types == 'POINT')[0])
    num_gauss = len(where(all_comp_types == 'GAUSSIAN')[0])
    num_shape = len(where(all_comp_types == 'SHAPELET')[0])

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
