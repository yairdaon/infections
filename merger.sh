#!/bin/bash

## Clean everything if you really want to. Used for testing
#rm -rvf pix
#rm -rvf tables
#rm data/*.pickle

rm -rvf *pix*
rm -rvf *tables*
python3 risks.py ## -skip skips figures, -d runs in debug mode


## Trim everybody
find . -name '*.jpg' -exec mogrify -trim \> {} \;

## Debugging
# convert pix/cb_horizontal.jpg -resize 760x88 pix/cb_horizontal_africa.jpg
# convert pix/cb_horizontal.jpg -resize 640x88 pix/cb_horizontal_india.jpg

## Real Deal
convert pix/cb_horizontal_africa.jpg -resize 2815x337 pix/cb_horizontal_africa.jpg
convert pix/cb_horizontal_india.jpg -resize 1769x337 pix/cb_horizontal_india.jpg
convert pix/cb_horizontal_brazil.jpg -resize 1824x337 pix/cb_horizontal_brazil.jpg
convert pix/cb_horizontal_russia.jpg -resize 2146x337 pix/cb_horizontal_russia.jpg


## Stack (and trim, even thogh it's not necessary) using imagemagick

## Main figures
for region in global india africa usa russia brazil europe china
do
    convert pix/$region/geodesics.jpg pix/$region/risks_rep_wuhan3_kappa1_main.jpg pix/cb_horizontal_$region.jpg -append fig_$region.jpg
done

# convert pix/africa/geodesics.jpg pix/africa/risks_rep_wuhan3_kappa1_main.jpg pix/cb_africa_horizontal.jpg -append fig_africa.jpg
# convert pix/india/geodesics.jpg  pix/india/risks_rep_wuhan3_kappa1_main.jpg  pix/cb_india_horizontal.jpg  -append fig_india.jpg


## Supplementary Figures
convert pix/cb_horizontal_global.jpg -resize 3952x337 pix/cb_horizontal_global_months.jpg
convert pix/global/risks_monthly_wuhan3_kappa1.jpg pix/cb_horizontal_global_months.jpg -append fig_S1_monthly.jpg
convert pix/global/risks_rep_wuhan3_kappa1.jpg pix/global/risks_rep_wuhan3_kappa3.jpg pix/global/risks_rep_wuhan3_kappa6.jpg pix/cb_horizontal_global.jpg -append fig_S2_different_kappas.jpg
convert pix/global/risks_rep_wuhan2_kappa1.jpg pix/global/risks_rep_wuhan3_kappa1.jpg pix/global/risks_rep_wuhan4_kappa1.jpg pix/cb_horizontal_global.jpg -append fig_S3_different_R0.jpg
convert pix/global/risks_rep_wuhan3_kappa1.jpg pix/global/risks_rep_wuhan3_kappasuperspreaders.jpg pix/cb_horizontal_global.jpg -append fig_S4_superspreaders.jpg
convert pix/airports_p_outbreak_from_one_wuhan3_kappa1.jpg pix/airports_p_outbreak_from_one_wuhan3_kappa1_cb.jpg -append fig_S5_p_outbreak_from_one.jpg


## Move images back where they belong
mv fig*.jpg pix/
mv tables/global_risks.csv tables/TableS1.csv
rm -rvf **/*~
rm pix/*cb*.jpg*
