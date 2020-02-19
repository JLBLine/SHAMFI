.. SHAMFI documentation master file, created by
   sphinx-quickstart on Mon Feb 17 13:02:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SHAMFI Documentation
================================================================

**SHA**\pelet **M**\odelling **F**\or **I**\nterferometers

SHAMFI is designed to fit images out of radio-frequency interferometers using shapelet basis functions (`Refregier 2013`_). It's currently limited to fitting a FITS file, such as those output by WSClean_. It can output models that work with the calibration and imaging software the RTS (`Mitchell et al 2008`_) and the simulator WODEN_, so that the image based fits can be used to generate visibilities.

.. _Refregier 2013: https://doi.org/10.1046/j.1365-8711.2003.05901.x
.. _WSClean: https://sourceforge.net/p/wsclean/wiki/Installation/
.. _Mitchell et al 2008: https://ieeexplore.ieee.org/document/4703504?arnumber=4703504 "IEEExplorer"
.. _WODEN: https://github.com/JLBLine/WODEN


The repo includes:

+---------------------------------------+------------------------------------------------------------------------------+
| Script                                | Overall Function                                                             |
+=======================================+==============================================================================+
|``fit_shamfi.py``                      | Takes a FITS file, and fits a shapelet model                                 |
+---------------------------------------+------------------------------------------------------------------------------+
|``mask_fits_shamfi.py``                | Splits a FITS image into multiple images using gaussian masks                |
+---------------------------------------+------------------------------------------------------------------------------+
|``subtract_gauss_from_image_shamfi.py``| Subtracts specified gaussians from an image to make fitting shapelets easier |
+---------------------------------------+------------------------------------------------------------------------------+
|``combine_srclists_shamfi.py``         | Combine multiple RTS/WODEN source catalogues into one                        |
+---------------------------------------+------------------------------------------------------------------------------+
|``convert_srclists_shamfi.py``         | Convert between RTS and WODEN source catalogue formats                       |
+---------------------------------------+------------------------------------------------------------------------------+

Installation
------------------------------------------
Grab the source code from this git repo:

``git clone https://github.com/JLBLine/SHAMFI``

Then navigate into that directory and run

``pip install .``

or alternatively

``python setup.py install``

Usage
------------------------------------------
Check out the tutorial page for an example of how to run the scripts

.. toctree::
   :maxdepth: 1

   tutorial

Scripts
------------------------------------------
.. toctree::
   :maxdepth: 1

   script_documentation

Modules
------------------------------------------

.. toctree::
   :maxdepth: 1

   module_documentation
