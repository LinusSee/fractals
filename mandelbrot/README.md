# The Mandelbot Set
The goal of this part is twofold.
<br>
First I want to write a version of the Mandelbrot set in C++. It should be zoomable and have a high rate of 
detail. It does not matter if the computations of quick zooming make it slow, in fact I am hoping for it.
<br>
The second part is copying the code and transforming it to a version that can be run using CUDA. It is 
supposed  to utilize CUDA to speed up the calculations so as not to cause any delays when 
zooming/recalculating.


## Resources
This project uses the image file format called PPM.
See https://en.wikipedia.org/wiki/Netpbm_format
