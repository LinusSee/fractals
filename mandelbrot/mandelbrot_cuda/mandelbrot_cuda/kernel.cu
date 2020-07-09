#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <cuComplex.h>
#include <thrust/complex.h>

#include <stdio.h>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <math.h>
#include <math_functions.h>
#include <device_functions.h>

//using namespace std;


struct set_area {
    double startX;
    double endX;
    double startY;
    double endY;
};

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
void mandelbrot_set(int* m_set, set_area area, int num_points, int max_iterations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int length = num_points * num_points;

    double spacing_x = abs(area.endX - area.startX) / ((double)num_points - 1.0);
    double spacing_y = abs(area.endY - area.startY) / ((double)num_points - 1.0);

    for (int i = index; i < length; i += stride)
    {
        int pos_x = i % num_points;
        int pos_y = i / num_points;
        double x = area.startX + pos_x * spacing_x;
        double y = area.startY + pos_y * spacing_y;

        thrust::complex<double> c = thrust::complex<double>(x, y);

        int iterations = mandelbrot_point(c, max_iterations);

        m_set[i] = iterations;
    }
}

/*int* point_color(int iterations, int max_iterations)
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
}*/



int* point_color(int iterations, int min, int max)
{
    double percentage = (1.0 * (iterations - min)) / (max - min);
    int* colors = new int[4];
    colors[3] = 255;
    if (percentage <= 0.33)
    {
        percentage = percentage / 0.33;
        colors[0] = std::ceil(255.0 * percentage);
        colors[1] = 0;
        colors[1] = 0;
    }
    else if (percentage <= 0.66)
    {
        percentage = (0.66 - percentage) / 0.33;
        colors[0] = std::ceil(255.0 * percentage);
        colors[1] = std::ceil(255.0 * (1.0 - percentage));
        colors[2] = 0;
    }
    else
    {
        percentage = (1.0 - percentage) / 0.34;
        colors[0] = 0;
        colors[1] = std::ceil(255.0 * percentage);
        colors[2] = std::ceil(255.0 * (1.0 - percentage));
    }

    return colors;
}






/*__global__
void set_to_image(int* m_set, sf::Uint8* image, int num_points, int max_iterations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int length = num_points * num_points;

    for (int i = index; i < length; i += stride)
    {
        int min_iter = max_iterations;
        int max_iter = 0;
        min_iter = min(min_iter, m_set[i]);
        max_iter = max(max_iter, m_set[i]);


        long long col_index = (long long) i * 4;
        int* color = point_color(m_set[i], min_iter, max_iter);
        image[index] = color[0];
        image[index + 1] = color[1];
        image[index + 2] = color[2];
        image[index + 3] = color[3];
    } 
}*/


sf::Uint8* set_to_image(int* m_set, int num_points, int max_iterations)
{
    int min = max_iterations;
    int max = 0;
    for (int x = 0; x < num_points * num_points; x++)
    {
        min = std::min(min, m_set[x]);
        max = std::max(max, m_set[x]);
    }

    //cout << "Min: " << min << endl;
    //cout << "Max: " << max << endl;

    sf::Uint8* image = new sf::Uint8[(long long)num_points * num_points * 4];
    for (int x = 0; x < num_points * num_points; x++)
    {
        int index = x * 4;
        //int* color = point_color(m_set[x], max_iterations);
        int* color = point_color(m_set[x], min, max);
        image[index] = color[0];
        image[index + 1] = color[1];
        image[index + 2] = color[2];
        image[index + 3] = color[3];
    }
    return image;
}


set_area refresh_mandelbrot(int* m_set, int num_points, int max_iterations, set_area previousArea, int x, int y, int zoomDelta, int numBlocks, int blockSize)
{
    double factor = 1;
    double rangeX = previousArea.endX - previousArea.startX;
    double rangeY = previousArea.endY - previousArea.startY;
    double factorX = (double) x / num_points - 0.5;
    double factorY = (double) y / num_points - 0.5;
    //cout << "FactorX " << factorX << endl;
    //cout << "FactorY " << factorY << endl;

    previousArea.startX += factorX * rangeX;
    previousArea.endX += factorX * rangeX;
    previousArea.startY += factorY * rangeY;
    previousArea.endY += factorY * rangeY;

    double centerX = previousArea.startX + rangeX / 2.0;
    double centerY = previousArea.startY + rangeY / 2.0;
    //cout << "CenterX " << centerX << endl;
    //cout << "CenterY " << centerY << endl;

    if (zoomDelta > 0)
    {
        factor = 1.0 / (zoomDelta + 1);
    }
    else if (zoomDelta < 0) {
        factor = std::abs(zoomDelta - 1);
    }
    previousArea.startX = centerX - (rangeX / 2.0) * factor;
    previousArea.endX = centerX + (rangeX / 2.0) * factor;
    previousArea.startY = centerY - (rangeY / 2.0) * factor;
    previousArea.endY = centerY + (rangeY / 2.0) * factor;
    //cout << "In method: " << previousArea.startX << endl;
    //cout << previousArea.endX << endl;
    //cout << previousArea.startY << endl;
    //cout << previousArea.endY << endl;

    mandelbrot_set<<<numBlocks, blockSize>>>(m_set, previousArea, num_points, max_iterations);
    cudaDeviceSynchronize();

    return previousArea;
}

int main()
{
    constexpr int num_points = 1000;
    constexpr int base_iterations = 50;
    size_t size_old = num_points * num_points;
    size_t size = num_points * num_points * sizeof(int);

    set_area area;
    area.startX = -2.25;
    area.endX = 0.75;
    area.startY = -1.5;
    area.endY = 1.5;

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    int* m_set;
    cudaMallocManaged(&m_set, size);
    mandelbrot_set << <numBlocks, blockSize >> > (m_set, area, num_points, base_iterations);
    cudaDeviceSynchronize();

    constexpr int width = num_points;
    constexpr int height = num_points;
    sf::RenderWindow window(sf::VideoMode(width, height), "It works!");
    
    //sf::Uint8* image = new sf::Uint8[(long long)num_points * num_points * 4];
    //cudaMallocManaged(&m_set, (long long)num_points * num_points * 4 * sizeof(sf::Uint8));
    sf::Uint8* image = set_to_image(m_set, num_points, 120);
    //cudaDeviceSynchronize();

    /*for (int y = 500; y < 510; y++)
    {
        for (int x = 500; x < 510; x++)
        {
            cout << image[y * num_points * 4 + x * 4 + 2] << " - ";
            if (m_set[y * num_points + x] > 1)
            {
                cout << "ERROR! " << m_set[y * num_points + x];
            }
        }
        cout << endl;
    }*/

    sf::Texture texture;
    if (!texture.create(width, height))
    {
        return -1;
    }
    texture.update(image);
    /*for (int x = 0; x < 50; x++)
    {
        for (int y = 0; y < 50; y++)
        {
            cout << m_set[y * num_points + x] << " - ";
            if (m_set[y * num_points + x] > 1)
            {
                cout << "ERROR! " << m_set[y * num_points + x];
            }
        }
        cout << endl;
    }*/

    sf::Sprite sprite;
    sprite.setTexture(texture);
    int total_zoom = 0;
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
                    int zoom = event.mouseWheelScroll.delta;
                    total_zoom += zoom;
                    int max_iterations = std::max(base_iterations, base_iterations * total_zoom * 4 / 5);
                    std::cout << "Iter: " << max_iterations << std::endl;
                    area = refresh_mandelbrot(m_set, num_points, max_iterations, area, x, y, zoom, numBlocks, blockSize);
                    image = set_to_image(m_set, num_points, max_iterations);
                    //cudaDeviceSynchronize();
                    texture.update(image);
                    //cout << area.startX << endl;
                    //std::cout << "mouse x: " << event.mouseWheelScroll.x << std::endl;
                    //std::cout << "mouse y: " << event.mouseWheelScroll.y << std::endl;
                }
            }
        }
        //241 - 306
        //20 - 57
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