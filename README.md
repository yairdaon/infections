Suggesting policy changes for mitigating COVID-19 spread after initial
decline. You will need:

Python packages - you'll need python3 with standard packets (numpy, scipy,
matplotlib, pandas etc.) and cartopy, which is somewhat difficult to
install.

Data - see loaders.py for where to get it. For gridded population of
the world I used 2.5 minute resolution. Better resolution will come is
several files which this code cannot handle. Place everything in data
directory.

Image processing - You can (but don't have to) install imagemagick,
which is an open-source tool. If you run Ubuntu (or any Unix OS, I
think, including MacOS) you should be able to install it easily.

Run the script - the bash script merger.sh just does everything,
provided you have all the data in place. In Ubuntu, do

chmod +x merger.sh
./merger.sh

This should work in any Debian (probably), any Linux (possibly), MacOS
(potentially) and Windows (actually, it won't).