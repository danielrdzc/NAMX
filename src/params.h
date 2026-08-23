#pragma once
//
// Registro de parametros.
//
// Cada efecto declara sus perillas: nombre, rango, y como leerlas y escribirlas.
// La interfaz web se genera SOLA a partir de este registro, asi que al agregar
// un efecto nuevo sus controles aparecen sin tocar nada de la UI. La pantalla
// OLED va a leer de aqui mismo cuando llegue: dos interfaces, un solo modelo.
//
// Los setters escriben atomics (directamente o via Smoothed), asi que llamarlos
// desde el hilo del servidor es seguro sin candados.
//
#include <functional>
#include <string>
#include <vector>

namespace fx {

struct Param {
    std::string id;        // "delay.time"
    std::string label;     // "Time"
    std::string unit;      // "ms", "dB", ""
    float min = 0.0f, max = 1.0f, step = 0.01f;
    std::function<void(float)>  set;
    std::function<float()>      get;
};

} // namespace fx
