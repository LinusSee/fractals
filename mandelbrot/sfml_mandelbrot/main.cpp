#include <iostream>

#include <SFML/Graphics.hpp>

int main()
{
    constexpr float width = 500;
    constexpr float height = 400;
    sf::RenderWindow window(sf::VideoMode(width, height), "It works!");

    sf::Image image;
    image.create(20, 20, sf::Color::Red);
    sf::Texture texture;
    if(!texture.create(200, 200))
    {
        return -1;
    }
    texture.update(image);
    sf::Sprite sprite;
    sprite.setTexture(texture);

    while(window.isOpen())
    {
        sf::Event event;
        while(window.pollEvent(event))
        {
            if(event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }
    return 0;
}
