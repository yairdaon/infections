#!/bin/bash
rm -rvf pix
python3 risks.py

convert pix/global/geodesics.jpg pix/global/risks_annual_wuhan3_n1_kappa1.jpg pix/cb_horizontal.jpg -trim -append pix/global.jpg
convert pix/africa/geodesics.jpg pix/africa/risks_annual_wuhan3_n1_kappa1.jpg pix/cb_horizontal.jpg -append pix/africa.jpg
convert pix/india/geodesics.jpg pix/india/risks_annual_wuhan3_n1_kappa1.jpg pix/cb_horizontal.jpg -append pix/india.jpg

# convert pix/global/geodesics.jpg pix/global/risks_annual_wuhan3_n1_kappa1.jpg -append pix/global.jpg
# convert pix/africa/geodesics.jpg pix/africa/risks_annual_wuhan3_n1_kappa1.jpg -append pix/africa.jpg
# convert pix/india/geodesics.jpg pix/india/risks_annual_wuhan3_n1_kappa1.jpg -append pix/india.jpg
