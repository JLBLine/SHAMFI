# ./fit_shapelets.py --fits_file=../srclist_basic.fits \
#     --b1=3.0 --b2=3.0 --nmax=0 --save_tag=srclist_basic \
#     --freq=180 #--plot_lims=-0.005,0.05
# python plot_srclist.py --srclist=srclist_basic_fitted.txt --plot_lims=0,1

./fit_shapelets.py --fits_file=/home/jline/Documents/Shapelets/Python_code/Fits_files/FnxA.fits \
    --b1=5.2 --b2=3.0 --nmax=21 --save_tag=MWA_ForA \
    --freq=180
python plot_srclist.py --srclist=srclist_MWA_ForA.txt --plot_lims=0,1
