#!/bin/bash
rm -rvf pix/*
python3 risks.py

## Trim everybody
find . -name '*.jpg' -exec mogrify -trim \> {} \;

## Debugging
#convert pix/cb_horizontal.jpg -resize 760x88 pix/cb_horizontal_africa.jpg
#convert pix/cb_horizontal.jpg -resize 640x88 pix/cb_horizontal_india.jpg

## Real Deal
convert pix/cb_horizontal.jpg -resize 2815x337 pix/cb_horizontal_africa.jpg
convert pix/cb_horizontal.jpg -resize 2520x337 pix/cb_horizontal_india.jpg

## Stack (and trim, even thogh it's not necessary) using imagemagick
convert pix/global/geodesics.jpg pix/global/risks_annual_wuhan3_n1_kappa1.jpg pix/cb_horizontal.jpg        -append fig_global.jpg
convert pix/africa/geodesics.jpg pix/africa/risks_annual_wuhan3_n1_kappa1.jpg pix/cb_horizontal_africa.jpg -append fig_africa.jpg
convert pix/india/geodesics.jpg  pix/india/risks_annual_wuhan3_n1_kappa1.jpg  pix/cb_horizontal_india.jpg  -append fig_india.jpg
convert pix/R0_wuhan3.jpg pix/R0_wuhan3_cb.jpg -append fig_R0.jpg

## Move images back where they belong
mv fig*.jpg pix/

