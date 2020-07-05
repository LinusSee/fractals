#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <cuComplex.h>
#include <thrust/complex.h>

#include <stdio.h>
#include <iostream>

#include <SFML/Graphics.hpp>

using namespace std;



__device__
int mandelbrot_point(thrust::complex<double> c, int max_iterations)
{
    thrust::complex<double> z = thrust::complex<double>(0, 0);
    for (int i = 0; i < max_iterations; i++)
    {
        z = z * z + c;
        
        if (thrust::abs(z) > 4)
        {
            return i;
        }
    }
    return max_iterations;
}


__global__
void mandelbrot_set(int* m_set, double start_x, double end_x, double start_y, double end_y, int num_points, int max_iterations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int length = num_points * num_points;

    double spacing_x = abs(end_x - start_x) / ((double)num_points - 1.0);
    double spacing_y = abs(end_y - start_y) / ((double)num_points - 1.0);

    for (int i = index; i < length; i += stride)
    {
        int pos_x = i % num_points;
        int pos_y = i / num_points;
        double x = start_x + pos_x * spacing_x;
        double y = start_y + pos_y * spacing_y;

        thrust::complex<double> c = thrust::complex<double>(x, y);

        int iterations = max_iterations;
        thrust::complex<double> z = thrust::complex<double>(0, 0);
        for (int i = 0; i < max_iterations; i++)
        {
            z = z * z + c;
            if (thrust::abs(z) > 4)
            {
                iterations = i;
                break;
            }
        }
        m_set[i] = iterations;
    }
}

int* point_color(int iterations, int max_iterations)
{
    float percentage = (1.0f * iterations) / max_iterations;
    int* colors = new int[4];
    colors[3] = 255;
    if (percentage <= 0.33f)
    {
        percentage = percentage / 0.33f;
        colors[0] = std::ceil(255.f * percentage);
        colors[1] = 0;
        colors[1] = 0;
    }
    else if (percentage <= 0.66f)
    {
        percentage = (0.66f - percentage) / 0.33f;
        colors[0] = std::ceil(255.f * percentage);
        colors[1] = std::ceil(255.f * (1.f - percentage));
        colors[2] = 0;
    }
    else
    {
        percentage = (1.0f - percentage) / 0.34f;
        colors[0] = 0;
        colors[1] = std::ceil(255.f * percentage);
        colors[2] = std::ceil(255.f * (1.f - percentage));
    }

    return colors;
}







sf::Uint8* set_to_image(int* m_set, int num_points, int max_iterations)
{
    sf::Uint8* image = new sf::Uint8[(long long)num_points * num_points * 4];
    for (int x = 0; x < num_points * num_points; x++)
    {
        int index = x * 4;
        int* color = point_color(m_set[x], max_iterations);
        image[index] = color[0];
        image[index + 1] = color[1];
        image[index + 2] = color[2];
        image[index + 3] = color[3];
    }
    return image;
}

void refresh_mandelbrot(int* m_set, int num_points, int max_iterations, int x, int y, int zoom, int numBlocks, int blockSize)
{
    double startX = -2.25;
    double endX = 0.75;
    double startY = -1.5;
    double endY = 1.5;
    double factor = 1;
    if (zoom > 0)
    {
        factor = 1.0 / zoom;
    }
    else if (zoom < 0) {
        factor = std::abs(zoom);
    }
    startX = startX * factor;
    endX = endX * factor;
    startY = startY * factor;
    endY = endY * factor;

    mandelbrot_set<<<numBlocks, blockSize>>>(m_set, startX, endX, startY, endY, num_points, max_iterations);
    cudaDeviceSynchronize();
}

int main()
{
    constexpr int num_points = 1000;
    constexpr int max_iterations = 120;
    size_t size_old = num_points * num_points;
    size_t size = num_points * num_points * sizeof(int);


    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    int* m_set;
    cudaMallocManaged(&m_set, size);
    mandelbrot_set<<<numBlocks, blockSize>>>(m_set, -2.25, 0.75, -1.5, 1.5, num_points, max_iterations);
    cudaDeviceSynchronize();

    constexpr int width = num_points;
    constexpr int height = num_points;
    sf::RenderWindow window(sf::VideoMode(width, height), "It works!");
    
    sf::Uint8* image = set_to_image(m_set, num_points, 120);

    sf::Texture texture;
    if (!texture.create(width, height))
    {
        return -1;
    }
    texture.update(image);
    sf::Sprite sprite;
    sprite.setTexture(texture);
    int zoom = 0;
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
                {
                    int x = event.mouseWheelScroll.x;
                    int y = event.mouseWheelScroll.y;
                    zoom += event.mouseWheelScroll.delta;
                    refresh_mandelbrot(m_set, num_points, max_iterations, x, y, zoom, numBlocks, blockSize);
                    image = set_to_image(m_set, num_points, 120);
                    texture.update(image);
                }
            }
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }

    cudaFree(m_set);
    return 0;
}
/*
                   std::cout << "Vertical wheel" << std::endl;
                    std::cout << "Wheel movement: " << event.mouseWheelScroll.delta << std::endl;
                    std::cout << "mouse x: " << event.mouseWheelScroll.x << std::endl;
                    std::cout << "mouse y: " << event.mouseWheelScroll.y << std::endl;
                    */