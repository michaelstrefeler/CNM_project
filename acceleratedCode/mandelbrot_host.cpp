// Compiler :
// nvcc -c mandelbrot_device.cu -o mandelbrot_device.o
// g++ -std=c++11 mandelbrot_host.cpp mandelbrot_device.o -o mandelbrot_acceleration -lsfml-graphics -lsfml-window -lsfml-system -fopenmp -lcudart

#include <SFML/Graphics.hpp>
#include <complex>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

// Dimensions de la fenêtre (et de l'image)
static const unsigned WIDTH  = 1000;
static const unsigned HEIGHT = 800;

// Nombre maximum d'itérations pour la fractale
static const unsigned MAX_ITER = 3000;

// Limites initiales du plan complexe
double minX = -4; //-2.0;
double maxX = 2; //1.0;
double minY = -2.4;//-1.2;
double maxY = 2.4; //1.2;

extern "C" void launchMandelbrotCUDA(unsigned char* hostPixels, unsigned width, unsigned height,
                                     double minX, double maxX, double minY, double maxY, unsigned maxIter);




// ---------------------------------------------------------------------------
// Fonction CPU avec OpenMP pour calculer la fractale de Mandelbrot
// ---------------------------------------------------------------------------
std::vector<sf::Uint8> computeMandelbrotOpenMP(unsigned width, unsigned height,
                                               double minX, double maxX,
                                               double minY, double maxY,
                                               unsigned maxIter) {
    std::vector<sf::Uint8> pixels(width * height * 4); // RGBA

    #pragma omp parallel for collapse(2)
    for (unsigned py = 0; py < height; ++py) {
        for (unsigned px = 0; px < width; ++px) {
            double x0 = minX + (px / (double)width) * (maxX - minX);
            double y0 = minY + (py / (double)height) * (maxY - minY);

            std::complex<double> c(x0, y0);
            std::complex<double> z(0, 0);

            unsigned iteration = 0;
            while (std::abs(z) < 2.0 && iteration < maxIter) {
                z = z * z + c;
                iteration++;
            }

            double t = (double)iteration / (double)maxIter;
            sf::Uint8 r = static_cast<sf::Uint8>(9 * (1 - t) * t * t * t * 255);
            sf::Uint8 g = static_cast<sf::Uint8>(15 * (1 - t) * (1 - t) * t * t * 255);
            sf::Uint8 b = static_cast<sf::Uint8>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

            unsigned index = 4 * (py * width + px);
            pixels[index + 0] = r;
            pixels[index + 1] = g;
            pixels[index + 2] = b;
            pixels[index + 3] = 255;
        }
    }

    return pixels;
}

int main()
{
    // Création de la fenêtre SFML
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Mandelbrot Zoom");
    window.setFramerateLimit(30);

    std::vector<sf::Uint8> pixelData(WIDTH * HEIGHT * 4);
    bool useGPU = false; // Toggle between CPU (false) and GPU (true)

    // Mesurer le temps de calcul de la fractale
    auto start = std::chrono::high_resolution_clock::now();

    if (useGPU) {
        
        launchMandelbrotCUDA(pixelData.data(), WIDTH, HEIGHT, minX, maxX, minY, maxY, MAX_ITER);

    } else {
        pixelData = computeMandelbrotOpenMP(WIDTH, HEIGHT, minX, maxX, minY, maxY, MAX_ITER);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Calculer la durée et l'afficher
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time to compute Mandelbrot: " << elapsed.count() << " seconds." << std::endl;

    // Création d'une texture et d'un sprite SFML pour l'affichage
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    texture.update(pixelData.data());

    sf::Sprite sprite(texture);

    // Rectangle visuel pour le zoom (qu'on dessine pendant la sélection)
    sf::RectangleShape selectionRect;
    selectionRect.setFillColor(sf::Color(0, 0, 0, 0));   // Transparent
    selectionRect.setOutlineColor(sf::Color::White);
    selectionRect.setOutlineThickness(1.f);

    bool isSelecting = false;
    sf::Vector2i startSelect; // Position de départ (clic)
    sf::Vector2i endSelect;   // Position courante (drag)

    // Boucle principale
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Fermeture de la fenêtre
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // Début de la sélection (clic gauche)
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    isSelecting = true;
                    startSelect = sf::Mouse::getPosition(window);
                    endSelect   = startSelect;
                }
            }

            // Fin de la sélection (relâchement du clic)
            if (event.type == sf::Event::MouseButtonReleased) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    isSelecting = false;
                    endSelect = sf::Mouse::getPosition(window);

                    // Calcul du nouveau rectangle de zoom, en coordonnées écran
                    int x1 = std::min(startSelect.x, endSelect.x);
                    int x2 = std::max(startSelect.x, endSelect.x);
                    int y1 = std::min(startSelect.y, endSelect.y);
                    int y2 = std::max(startSelect.y, endSelect.y);

                    // Éviter un zoom si le rectangle est trop petit
                    if (std::abs(x2 - x1) > 5 && std::abs(y2 - y1) > 5) {
                        // 1) Calcul brut de w et h
                        double w = x2 - x1;
                        double h = y2 - y1;

                        // 2) Ratio de la fenêtre
                        double ratioWindow = (double)WIDTH / (double)HEIGHT;
                        double ratioSelection = w / h;

                        // 3) Ajustement si nécessaire
                        if (ratioSelection > ratioWindow) {
                            // Trop large => on ajuste la hauteur h
                            h = w / ratioWindow;
                            // Si on veut conserver le coin en haut à gauche :
                            y2 = y1 + (int)h;
                        }
                        else if (ratioSelection < ratioWindow) {
                            // Trop haut => on ajuste la largeur w
                            w = h * ratioWindow;
                            x2 = x1 + (int)w;
                        }

                        // Conversion du rectangle écran => rectangle complexe
                        double newMinX = minX + (double)x1 / WIDTH  * (maxX - minX);
                        double newMaxX = minX + (double)x2 / WIDTH  * (maxX - minX);
                        double newMinY = minY + (double)y1 / HEIGHT * (maxY - minY);
                        double newMaxY = minY + (double)y2 / HEIGHT * (maxY - minY);

                        // Mise à jour des bornes
                        minX = newMinX;
                        maxX = newMaxX;
                        minY = newMinY;
                        maxY = newMaxY;

                        start = std::chrono::high_resolution_clock::now();

                        // Recalcule la fractale avec le nouveau cadre
                        if (useGPU) {
                            
                            launchMandelbrotCUDA(pixelData.data(), WIDTH, HEIGHT, minX, maxX, minY, maxY, MAX_ITER);

                        } else {
                            pixelData = computeMandelbrotOpenMP(WIDTH, HEIGHT, minX, maxX, minY, maxY, MAX_ITER);
                        }
                        
                        end = std::chrono::high_resolution_clock::now();

                        // Calculer la durée et l'afficher
                        elapsed = end - start;
                        std::cout << "Time to compute Mandelbrot zoom : " << elapsed.count() << " seconds." << std::endl;
                        
                        texture.update(pixelData.data());

                    }
                }
            }

            // Mouvement de la souris (pour dessiner le rectangle de sélection)
            if (event.type == sf::Event::MouseMoved) {
                if (isSelecting) {
                    endSelect = sf::Mouse::getPosition(window);
                }
            }
        }

        // Mise à jour du rectangle de sélection en temps réel
        if (isSelecting) {
            int x1 = std::min(startSelect.x, endSelect.x);
            int y1 = std::min(startSelect.y, endSelect.y);
            int x2 = std::max(startSelect.x, endSelect.x);
            int y2 = std::max(startSelect.y, endSelect.y);

            selectionRect.setPosition(static_cast<float>(x1), static_cast<float>(y1));
            selectionRect.setSize(sf::Vector2f((float)(x2 - x1), (float)(y2 - y1)));
        } else {
            // Si on ne sélectionne pas, on ne dessine pas le rectangle
            selectionRect.setSize(sf::Vector2f(0.f, 0.f));
        }

        // Affichage
        window.clear();
        window.draw(sprite);

        // Dessin du rectangle si en cours de sélection
        if (isSelecting) {
            window.draw(selectionRect);
        }

        window.display();
    }

    return 0;
}
