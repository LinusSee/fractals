# The Mandelbot Set
The goal of this part is threefold.
<br>
1. First I want to write a version of the Mandelbrot set in C++. It should be possible to display it and it should be colored.
2. The second part is copying the code and transforming it to a version that can be run using CUDA. It is 
supposed  to utilize CUDA to speed up the calculations so as not to cause any waiting times.
3. After that I reached the important goals, but want to make it a bit more interactive by allowing the user to zoom in. This is also a great test for how performant my CUDA code is, since delays become quite noticable when zooming.


## Resources
In this project I use SFML and Nvidia Cuda.
I found the SFML setup (on Windows) rather troublesome, this [tutorial](https://www.youtube.com/watch?v=fcZFaiGFIMA) really helped me out.
Also pay attention to the compiler you are using matching the one specified by SFML.
