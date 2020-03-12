from subprocess import check_output
import os
import json
import pkg_resources
from shamfi import __version__,__path__

def get_commandline_output(command_list):
    """
    Takes a command line entry separated into list entries, and returns the
    output from the command line as a string

    Parameters
    ----------
    command_list : list of strings
        list of strings that when combined form a coherent command to input into
        the command line

    Returns
    -------
    output : string
        the output result of running the command

    """
    output = check_output(command_list,universal_newlines=True).strip()
    return output

def make_gitdict():
    """
    Makes a dictionary containing key git information about the repo by running
    specific commands on the command line

    Returns
    -------
    git_dict : dictionary
        A dictionary containing git information with keywords: describe, date,
        branch

    """

    git_dict = {
        'describe': get_commandline_output(["git", "describe", "--always"]),
        'date': get_commandline_output(["git", "log", "-1", "--format=%cd"]),
        'branch': get_commandline_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }

    return git_dict

def get_gitdict():
    """
    Get the git dictionary that was created by setup.py and/or during pip install
    and was stored in a json file

    Returns
    -------
    git_dict : dictionary
        A dictionary containing git information with keywords: describe, date,
        branch

    """
    json_loc = os.path.join(__path__[0],"shamfi_gitinfo.json")

    with open(json_loc,'r') as json_file:
        git_dict = json.load(json_file)

    return git_dict

def write_git_header(outfile):
    """
    Takes a textfile and writes a git summary at the top, commented with #

    Parameters
    ----------
    outfile : text file instance
        Text file that gets useful version information appended to it

    """
    git_dict = get_gitdict()

    outfile.write('## Written with SHAMFI (Copyright (c) J. L. B. Line)\n')
    outfile.write('## Git describe: %s\n' %(git_dict['describe']))
    outfile.write('## Git date: %s\n' %(git_dict['date']))
    outfile.write('## Git branch: %s\n' %(git_dict['branch']))

def print_version_info(script_loc):
    """
    Takes the location of the script calling this function, and prints
    out useful git information

    Parameters
    ----------
    script_loc : string
        Absolute path of the file calling this function

    """
    ''''''
    git_dict = get_gitdict()

    print('This is SHAMFI version %s (Copyright (c) J. L. B. Line)' %__version__)
    print('The script you are using lives here:\n%s' %script_loc)
    print('More specifically, here is some git information:')
    print(' - Git describe: %s' %(git_dict['describe']))
    print(' - Git date: %s' %(git_dict['date']))
    print(' - Git branch: %s' %(git_dict['branch']))
