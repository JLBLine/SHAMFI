#! /usr/bin/env python
'''
Setup for SHAMFI
'''
import os
import sys
from setuptools import setup, find_packages
import setuptools.command.build_py
from shamfi.git_helper import make_gitdict
from sys import path
import json
import distutils.cmd


class GitInfo(distutils.cmd.Command):
  '''A custom command to create a json file containing SHAMFI git information.'''

  description = 'Create the file "shamfi/shamfi_gitinfo.json" containing git information '
  user_options = []

  def initialize_options(self):
    '''Set default values for options (this has to be included for
    distutils.cmd to work)'''
    # Each user option must be listed here with their default value.
    self.git_info = True

  def finalize_options(self):
    '''Post-process options (this has to be included for
    distutils.cmd to work)'''
    if self.git_info:
        print('Creating file shamfi/shamfi_gitinfo.jsons')
      # assert os.path.exists(self.pylint_rcfile), (
      #     'Pylint config file %s does not exist.' % self.pylint_rcfile)

  def run(self):
    '''Write the shamfi git json file.'''
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'shamfi', 'shamfi_gitinfo.json'), 'w') as outfile:
        json.dump(make_gitdict(), outfile)


class BuildPyCommand(setuptools.command.build_py.build_py):
  '''Custom build command to run the gitinfo command during build'''

  def run(self):
    self.run_command('gitinfo')
    setuptools.command.build_py.build_py.run(self)


def read(fname):
    '''Reads a file so we can read the REAMME.'''
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

##Module requirements
reqs = ['matplotlib>=2.2.4',
        'scipy>=1.2.2',
        'astropy>=2.0.15',
        'numpy>=1.16.5',
        'progressbar2>=3.38.0']

setup(
    name="shamfi",
    version='0.1b',
    author="Jack Line",
    author_email="jack.l.b.line@gmail.com",
    description="The SHAMFI shapelet fitted code and associated scripts",
    url="https://github.com/JLBLine/SHAMFI.git",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=['shamfi'],
    install_requires=reqs,
    scripts=['scripts/fit_shamfi.py', 'scripts/mask_fits_shamfi.py',
             'scripts/subtract_gauss_from_image_shamfi.py',
             'scripts/combine_srclists_shamfi.py',
             'scripts/convert_srclists_shamfi.py'],
    python_requires='>=2.7',
    cmdclass={'gitinfo': GitInfo,
              'build_py': BuildPyCommand,
              },
    package_data={"shamfi": ["image_shapelet_basis.npz", "shamfi_gitinfo.json"]},
)
