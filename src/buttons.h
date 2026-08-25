#pragma once
//
// Botones fisicos por GPIO (libgpiod v2).
//
// POR QUE SONDEO Y NO EVENTOS. libgpiod sabe entregar eventos de flanco, pero un
// contacto mecanico REBOTA: un solo apreton genera una rafaga de flancos en
// pocos milisegundos. Con eventos habria que filtrar la rafaga igual, y encima
// lidiar con la cola de eventos del kernel. Leer el estado cada 5 ms y exigir que
// se mantenga estable es mas simple, mas predecible, y el costo de CPU es
// literalmente inmedible al lado de la inferencia.
//
// Este hilo NO es de tiempo real y no toca el hilo de audio: solo llama a los
// handlers, que a su vez mueven atomics o cargan presets.
//
#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#if defined(NAMX_HAS_GPIOD)
  #include <gpiod.h>
#endif

namespace fx {

class Buttons {
public:
    struct Def {
        unsigned int offset;              // numero de GPIO (no el pin fisico)
        std::function<void()> onPress;
    };

    ~Buttons() { stop(); }

#if !defined(NAMX_HAS_GPIOD)
    bool start(const std::string&, std::vector<Def>) { return false; }
    void stop() {}
    static bool available() { return false; }
#else
    static bool available() { return true; }

    bool start(const std::string& chipPath, std::vector<Def> defs) {
        if (defs.empty()) return false;
        _defs = std::move(defs);

        _chip = gpiod_chip_open(chipPath.c_str());
        if (!_chip) return false;

        gpiod_line_settings* settings = gpiod_line_settings_new();
        gpiod_line_settings_set_direction(settings, GPIOD_LINE_DIRECTION_INPUT);
        // El pull-up interno es lo que hace que baste un boton contra tierra:
        // sin el, la linea queda flotando y lee basura.
        gpiod_line_settings_set_bias(settings, GPIOD_LINE_BIAS_PULL_UP);

        std::vector<unsigned int> offsets;
        for (const auto& d : _defs) offsets.push_back(d.offset);

        gpiod_line_config* lcfg = gpiod_line_config_new();
        gpiod_line_config_add_line_settings(lcfg, offsets.data(), offsets.size(), settings);

        gpiod_request_config* rcfg = gpiod_request_config_new();
        gpiod_request_config_set_consumer(rcfg, "namx");

        _req = gpiod_chip_request_lines(_chip, rcfg, lcfg);

        gpiod_request_config_free(rcfg);
        gpiod_line_config_free(lcfg);
        gpiod_line_settings_free(settings);

        if (!_req) { gpiod_chip_close(_chip); _chip = nullptr; return false; }

        _stable.assign(_defs.size(), 3);      // 3 = suelto (pull-up: alto)
        _pressed.assign(_defs.size(), false);
        _run.store(true);
        _th = std::thread([this] { loop(); });
        return true;
    }

    void stop() {
        if (!_run.exchange(false)) return;
        if (_th.joinable()) _th.join();
        if (_req)  { gpiod_line_request_release(_req); _req = nullptr; }
        if (_chip) { gpiod_chip_close(_chip); _chip = nullptr; }
    }

private:
    void loop() {
        // Un apreton solo cuenta cuando la linea lleva 3 lecturas seguidas
        // (15 ms) en el mismo estado. Los rebotes duran mucho menos que eso.
        constexpr int kStableReads = 3;

        while (_run.load()) {
            for (size_t i = 0; i < _defs.size(); ++i) {
                const gpiod_line_value v =
                    gpiod_line_request_get_value(_req, _defs[i].offset);
                // ACTIVE = alto = suelto. INACTIVE = a tierra = presionado.
                const bool down = (v == GPIOD_LINE_VALUE_INACTIVE);

                if (down == _pressed[i]) {
                    _stable[i] = kStableReads;       // sin cambio, reinicia el contador
                } else if (--_stable[i] <= 0) {
                    _pressed[i] = down;
                    _stable[i]  = kStableReads;
                    if (down && _defs[i].onPress) _defs[i].onPress();   // solo al bajar
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    gpiod_chip*         _chip = nullptr;
    gpiod_line_request* _req  = nullptr;
    std::vector<Def>    _defs;
    std::vector<int>    _stable;
    std::vector<bool>   _pressed;
#endif

    std::atomic<bool> _run{false};
    std::thread _th;
};

} // namespace fx
