# **SHAMFI** (**SHA**pelet **M**odelling **F**or **I**nterferometers)
SHAMFI is designed to fit images out of radio-frequency interferometers using shapelet basis functions ([Refregier 2013](https://doi.org/10.1046/j.1365-8711.2003.05901.x)). It's currently limited to fitting a FITS file, such as those output by [WSClean](https://sourceforge.net/p/wsclean/wiki/Installation/). It can output models that work with the calibration and imaging software the RTS ([Mitchell et al. 2008](https://ieeexplore.ieee.org/document/4703504?arnumber=4703504 "IEEExplorer")) and the simulator [WODEN](https://github.com/JLBLine/WODEN), so that the image based fits can be used to generate visibilities.

This repo includes:

Script  | Overall Function
--|--
`fit_shamfi.py` | Takes a FITS file and fits a shapelet model
`mask_fits_shamfi.py` | Splits a FITS image into multiple images using gaussian masks
`subtract_gauss_from_image_shamfi.py` | Subtracts specified gaussians from an image to make fitting shapelets easier
`combine_srclists_shamfi.py` | Combine multiple RTS/WODEN source catalogues into one
`convert_srclists_shamfi.py` | Convert between RTS and WODEN source catalogue formats

# Installation
Grab the source code from this git repo:
```sh
git clone https://github.com/JLBLine/SHAMFI
```
Then navigate into that directory and run
```
pip install .
```
or alternatively
```
python setup.py install
```
