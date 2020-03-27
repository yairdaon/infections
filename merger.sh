#!/bin/bash
rm -rvf pix
python3 risks.py -d

convert -append pix/global/geodesics.jpg pix/global/risks_annual_wuhan3_n1_kappa1.jpg pix/global.jpg
convert -append pix/africa/geodesics.jpg pix/africa/risks_annual_wuhan3_n1_kappa1.jpg pix/africa.jpg
convert -append pix/india/geodesics.jpg pix/india/risks_annual_wuhan3_n1_kappa1.jpg pix/india.jpg
