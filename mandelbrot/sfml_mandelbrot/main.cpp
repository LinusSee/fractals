#include <iostream>
#include <vector>

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <SFML/Graphics.hpp>

constexpr float width     = 1280;
constexpr float height    = 720;

const std::vector<GLfloat> vertexPositions
{
    //Back
    1.1,  -1.1,  -1.1,
    -1.1,  -1.1,  -1.1,
    -1.1,   1.1,  -1.1,
    1.1,   1.1,  -1.1,

    //Right-Side
    1.1, -1.1,     1.1,
    1.1, -1.1,    -1.1,
    1.1,  1.1,    -1.1,
    1.1,  1.1,     1.1,

    //Front
    -1.1,  -1.1,  1.1,
    1.1,  -1.1,  1.1,
    1.1,   1.1,  1.1,
    -1.1,   1.1,  1.1,

    //Left
    -1.1, -1.1,   -1.1,
    -1.1, -1.1,    1.1,
    -1.1,  1.1,    1.1,
    -1.1,  1.1,   -1.1,

    //Top
    -1.1,  1.1,    1.1,
    1.1,  1.1,    1.1,
    1.1,  1.1,   -1.1,
    -1.1,  1.1,   -1.1,

    //Bottom
    -1.1,  -1.1,  -1.1,
    1.1,  -1.1,  -1.1,
    1.1,  -1.1,   1.1,
    -1.1,  -1.1,   1.1
};

const std::vector<GLuint> cubeIndices
{
    0, 1, 2,
    2, 3, 0,

    4, 5, 6,
    6, 7, 4,

    8, 9, 10,
    10, 11, 8,

    12, 13, 14,
    14, 15, 12,

    16, 17, 18,
    18, 19, 16,

    20, 21, 22,
    22, 23, 20
};

const GLchar* vertexShaderSource =
"                                                   \
#version 330                                        \n\
                                                    \
layout (location = 0) in vec3 inVertexPosition;     \
                                                    \
out float z;                                        \
                                                    \
uniform mat4 modelMatrix;                           \
uniform mat4 projViewMatrix;                        \
                                                    \
void main()                                         \
{                                                   \
    gl_Position = projViewMatrix * modelMatrix *    \
                  vec4 (inVertexPosition.xyz, 1.0); \
    z = sin(exp(gl_Position.z));                              \
}                                                   \
";

const GLchar* fragmentShaderSource =
"                                                   \
#version 330                                        \n\
in float z;                                         \
out vec4 colour;                                    \
                                                    \
void main()                                         \
{                                                   \
    colour = vec4(z, 1, 0, 1);                      \
}                                                   \
";

int main()
{
    sf::ContextSettings contextSettings;
    contextSettings.depthBits = 24;
    contextSettings.stencilBits = 8;
    contextSettings.antialiasingLevel = 0;
    contextSettings.majorVersion = 3;
    contextSettings.minorVersion = 3;

    sf::Window window(sf::VideoMode(width, height), "It Works!", sf::Style::Close, contextSettings);
    window.setFramerateLimit(30);

    glewInit();

    glEnable(GL_DEPTH_TEST);

    glViewport(0, 0, width, height);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    GLuint ebo;

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 vertexPositions.size() * sizeof(vertexPositions.at(0)),
                 vertexPositions.data(),
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          0,
                          (GLvoid*) 0);


    glEnableVertexAttribArray(0);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 cubeIndices.size() * sizeof(cubeIndices.at(0)),
                 cubeIndices.data(),
                 GL_STATIC_DRAW);

    auto shaderID = glCreateProgram();

    auto vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderID, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShaderID);

    {
        GLint isSuccess;
        GLchar infoLog[512];

        glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &isSuccess);

        if(!isSuccess)
        {
            glGetShaderInfoLog(vertexShaderID, 512, nullptr, infoLog);
            throw std::runtime_error ("Error compiling shader: " + std::string(infoLog));
        }
    }

    auto fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderID, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShaderID);

    {
        GLint isSuccess;
        GLchar infoLog[512];

        glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &isSuccess);

        if(!isSuccess)
        {
            glGetShaderInfoLog(fragmentShaderID, 512, nullptr, infoLog);
            throw std::runtime_error ("Error compiling shader: " + std::string(infoLog));
        }
    }

    glAttachShader(shaderID, vertexShaderID);
    glAttachShader(shaderID, fragmentShaderID);
    glLinkProgram   (shaderID);
    glValidateProgram(shaderID);

    {
        GLint success;
        GLchar infoLog[512];

        glGetShaderiv(shaderID, GL_LINK_STATUS, &success);

        if (!success)
        {
            glGetShaderInfoLog(shaderID, 512, NULL, infoLog);     // Generate error as infoLog
            std::cout << "Error " << infoLog << std::endl;  // Display
        }
    }

    glBindVertexArray(vao);

    glUseProgram (shaderID);
    GLint modelLocation = glGetUniformLocation(shaderID, "modelMatrix");
    GLint pvLocation    = glGetUniformLocation(shaderID, "projViewMatrix");
    glUseProgram (0);

    glm::mat4 viewMatrix{1.0f}, projMatrix{1.0f}, pv{1.0f}, modelMatrix{1.0f};

    viewMatrix = glm::translate(viewMatrix, {0, 0, -5});
    projMatrix = glm::perspective(glm::radians(85.0f), width / height, 0.01f, 1000.0f);

    pv = projMatrix * viewMatrix;

    glUniformMatrix4fv(pvLocation, 1, GL_FALSE, glm::value_ptr(pv));

    sf::Clock c;

    while (window.isOpen())
    {
        glClearColor(0.1, 0.5, 1.0, 1.0);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        glm::mat4 rot{1.0f};
        rot = glm::rotate(rot, (float)sin(c.getElapsedTime().asSeconds()/ 100.0f), {1, 0, 0});

        glm::mat4 trans{1.0f};
        trans = glm::translate(trans, {std::sin(c.getElapsedTime().asSeconds())/10, 0, 0});

        modelMatrix = rot * trans * modelMatrix;


        glUseProgram(shaderID);
        glUniformMatrix4fv(pvLocation, 1, GL_FALSE, glm::value_ptr(pv));
        glUniformMatrix4fv(modelLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));

        glDrawElements(GL_TRIANGLES, cubeIndices.size(), GL_UNSIGNED_INT, nullptr);

        window.display();
    }

    return 0;
}
