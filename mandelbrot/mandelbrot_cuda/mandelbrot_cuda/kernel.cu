#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <complex>
#include <tuple>

#include <SFML/Graphics.hpp>

using namespace std;


int mandelbrot_point(complex<double> c, int max_iterations)
{
    std::complex<double> z(0, 0);
    for (int i = 0; i < max_iterations; i++)
    {
        z = z * z + c;
        if (abs(z) > 4)
        {
            return i;
        }
    }
    return max_iterations;
}

//__global__
void mandelbrot_set(int* m_set, double start_x, double end_x, double start_y, double end_y, int num_points, int max_iterations)
{
    double spacing_x = abs(end_x - start_x) / ((double) num_points - 1.0);
    double spacing_y = abs(end_y - start_y) / ((double) num_points - 1.0);

    cout << "SpacingX: " << spacing_x << endl;
    cout << "SpacingY: " << spacing_y << endl;

    double current_x = start_x;
    double current_y = start_y;

    //int* m_set = new int[num_points * num_points];
    for (int i = 0; i < num_points; i++)
    {
        for (int j = 0; j < num_points; j++)
        {
            int index = i * num_points + j;

            complex<double> c(current_x, current_y);
            int iterations = mandelbrot_point(c, max_iterations);

            m_set[index] = iterations;
            current_x = current_x + spacing_x;
        }
        current_x = start_x;
        current_y = current_y + spacing_y;
    }

    //return m_set;
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


int main()
{
    constexpr int num_points = 1000;
    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    int* m_set = new int[num_points * num_points];
    mandelbrot_set(m_set, -2.25, 0.75, -1.5, 1.5, num_points, 120);


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

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }
    return 0;
}
