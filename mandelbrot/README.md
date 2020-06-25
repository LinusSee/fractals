# The Mandelbot Set
The goal of this part is threefold.
<br>
1. First I want to write a version of the Mandelbrot set in C++. It should be possible to display it and it should be colored.
2. The second part is copying the code and transforming it to a version that can be run using CUDA. It is 
supposed  to utilize CUDA to speed up the calculations so as not to cause any waiting times.
3. After that I reached the important goals, but want to make it a bit more interactive by allowing the user to zoom in. This is also a great test for how performant my CUDA code is, since delays become quite noticable when zooming.

## Progress
### Goal #1
The Mandelbrot set is implemented and can be run using SFML and the Codeblocks IDE (or with a little more effort manually).
The first result was this gray image.
![alt mandelbrot_in_gray](./assets/images/sfml_mandelbrot_gray.JPG)
<br>
<br>
<br>
Then I (primitively) interpolated the color depending on how quickly a point escaped from the set and got this.
<br>
<br>
![alt mandelbrot_in_color](./assets/images/sfml_mandelbrot_color.JPG)

### Goal #2
Since CUDA is (on Windows) only compatible with VisualC++, which I conveniently ignored, I had some trouble matching my TDM or MinGW compiled SFML code with CUDA code.
<br>
Since I am using VS 19 I had to build SFML locally and then included it in a VS project to run a SFML code sample. Then I updated CUDA to 10.1 update 2, so I have a VS CUDA template and now I am finally able to create CUDA project in VS code. Now all I need to do is combine SFML and CUDA in a project to get going with goal #2.

## Resources
In this project I use SFML and Nvidia Cuda.
I found the SFML setup (on Windows) rather troublesome, this [tutorial](https://www.youtube.com/watch?v=fcZFaiGFIMA) really helped me out.
Also pay attention to the compiler you are using matching the one specified by SFML.
