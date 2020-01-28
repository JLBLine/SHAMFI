#! /usr/bin/env python
"""
Setup for AegeanTools
"""
import os
import sys
# from setuptools import
from distutils.command.build_py import build_py
# from distutils.core import setup
from setuptools import setup, find_packages
from shamfi.git_helper import make_gitdict
from sys import path
import json

def read(fname):
    """Read a file"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

reqs = ['matplotlib>=2.2.4',
        'scipy>=1.2.2',
        'astropy>=2.0.15',
        'numpy>=1.16.5',
        'progressbar2>=3.38.0']

class my_build_py(build_py):
    def run(self):
        # honour the --dry-run flag
        if not self.dry_run:
            ##Write a dictionary containing git info to the library
            ##Copy it to the build directory later
            with open(os.path.join('shamfi', 'shamfi_gitinfo.json'), 'w') as outfile:
                json.dump(make_gitdict(), outfile)

        # distutils uses old-style classes, so no super()
        build_py.run(self)


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
    cmdclass={'build_py': my_build_py},
    package_data={"shamfi": ["image_shapelet_basis.npz", "shamfi_gitinfo.json"]},
)
