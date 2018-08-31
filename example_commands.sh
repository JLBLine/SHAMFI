#./fit_shapelets.py --fits_file=NGC_1316_I_20cm_fev1989_i.fits \
#    --b1=4.0 --b2=5.0 --nmax=3 --save_tag=VLA_ForA --plot_lims=-0.02,0.05 \
#    --freq=1500


#./fit_shapelets.py --fits_file=NGC_1316_I_20cm_fev1989_i.fits \
#    --b1=4.0 --b2=5.0 --nmax=70 --save_tag=VLA_ForA --plot_lims=-0.02,0.05 \
#    --freq=1500 --no_srclist



#time ./fit_shapelets.py --fits_file=/usr/local/MWA_Tools/Models/PicA.fits \
#    --b1=3.0 --b2=5.0 --nmax=70 --save_tag=PicA \
#    --freq=150


time ./fit_shapelets.py --fits_file=cropped_ben_FornaxA.fits \
    --b1=8.0 --b2=8.0 --nmax=31 --save_tag=ForA_Ben \
    --freq=184.955
