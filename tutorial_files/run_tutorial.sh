subtract_gauss_from_image_shamfi.py \
    --fits_file=cropped_FornaxA_real_phase1+2.fits \
    --gauss_table=gaussians_to_subtract.txt \
    --outname=gauss-subtracted_phase1+2_real_data.fits

mask_fits_shamfi.py \
    --fits_file=gauss-subtracted_phase1+2_real_data.fits \
    --output_tag=real_ForA_phase1+2 \
    --box=6,120,50,170 --box=117,246,75,218

fit_shamfi.py \
    --save_tag=real_ForA_phase1+2_lobe1 \
    --fits_file=real_ForA_phase1+2_split01.fits \
    --b1s=3.0,4.0 --b2s=3.0,4.0 --nmax=86 \
    --num_beta_values=5 \
    --edge_pad=25 \
    --fit_box=0,200,50,240 \
    --woden_srclist --plot_resid_grid --plot_edge_pad \
    --compress=90.0,80.0,70.0
#
fit_shamfi.py \
    --save_tag=real_ForA_phase1+2_lobe2 \
    --fits_file=real_ForA_phase1+2_split02.fits \
    --b1s=3.0,4.0 --b2s=3.0,4.0 --nmax=86 \
    --num_beta_values=5 \
    --fit_box=100,300,80,270 \
    --edge_pad=25 \
    --woden_srclist --plot_resid_grid --plot_edge_pad \
    --compress=90.0,80.0,70.0


for percent in '100' '090' '080' '070'
do
  combine_srclists_shamfi.py \
    --srclist=srclist-woden_real_ForA_phase1+2_lobe1_nmax086_p$percent.txt \
    --srclist=srclist-woden_real_ForA_phase1+2_lobe2_nmax086_p$percent.txt \
    --srclist=srclist_gaussian-woden.txt \
    --outname=srclist-woden_real_ForA_phase1+2_nmax086_p$percent.txt

  convert_srclists_shamfi.py \
    --srclist=srclist-woden_real_ForA_phase1+2_nmax086_p$percent.txt \
    --outname=srclist-rts_real_ForA_phase1+2_nmax086_p$percent.txt
done
