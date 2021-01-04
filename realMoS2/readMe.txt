The tif file contains actual images taken with the HAADF-STEM at ibs.
Note that the black-and-white contrast of the images are not the same as the raw simulated
computem images. You can either adjust the contrast of the simulated images to fit the real images
(recommended), or if you think that the simulated images have better contrast (by better, whether
the images look 'nice' for a publication), you can adjust the contrast of the real images with
3rd party software, such as ImageJ(free) or use python libraries.



The dm4 file contains the raw files (.dm4) from HAADF-STEM.
Since we only need the images and do not need the metadata, these files are of little use
for this project, but it's good to keep it at hand just in case.


(For future reference:
If you happen to work on a different project for which metadata is needed, or if you are only given .dm4 files without extracted images/videos, there are ways to read in .dm4 files in python, which requires installing some aditional packages such as HyperSpy.) 
