from __future__ import print_function,division
import os

def check_file_exists(filename, argname):
    """
    Checks if a file exists, and throws an error and exits if that file does
    not exist

    Parameters
    ----------
    filename : string
        Path to a file
    argname : string
        The argument that was passed to a script, for error checking purposes

    Returns
    -------
    filename : string
        Path to a file

    """
    if os.path.isfile(filename):
        return filename
    else:
        exit('The input "%s=%s" does not exist\nExiting now' %(argname,filename))

def check_if_txt_extension(name):
    """
    Checks if string `name` ends in ".txt", appends if not

    Parameters
    ----------
    name : string
        String to check

    Returns
    -------
    outname : string
        A string that definitely ends in ".txt"

    """
    if name[-4:] == '.txt':
        outname = name
    else:
        outname = name + '.txt'

    return outname

def check_if_fits_extension(name):
    """
    Checks if `name` ends in ".fits", appends if not

    Parameters
    ----------
    name : string
        String to check

    Returns
    -------
    outname : string
        A string that definitely ends in ".fits"

    """
    if name[-5:] == '.fits':
        outname = name
    else:
        outname = name + '.fits'

    return outname
