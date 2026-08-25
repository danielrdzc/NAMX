#pragma once
//
// El modelo NAM envuelto como un eslabon mas de la cadena.
//
// Cambio de modelo EN CALIENTE. El problema: cargar un .nam reserva memoria,
// lee un archivo y hace prewarm -- nada de eso puede pasar en el hilo de audio.
// La solucion es la misma que ya usamos para el orden de la cadena, y la misma
// que usa ContainerModel dentro del core de NAM:
//
//   - Hay dos ranuras. El hilo de control carga el modelo nuevo en la que NO
//     se esta usando, la deja lista (Reset + prewarm), y recien entonces
//     publica el puntero con un atomico.
//   - El hilo de audio lee ese atomico UNA vez por bloque. Nunca ve un modelo
//     a medio construir: o esta el viejo completo, o el nuevo completo.
//   - El viejo no se libera de inmediato: se espera a que el audio salga del
//     bloque en curso.
//
#include "effects.h"
#include "NAM/get_dsp.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace fx {

class NamModel : public Effect {
public:
    const char* name() const override { return "nam"; }

    void prepare(double sampleRate, int maxBlock) override {
        _sr       = sampleRate;
        _maxBlock = maxBlock;
        _scratch.assign(static_cast<size_t>(maxBlock), 0.0f);
        if (nam::DSP* m = _active.load(std::memory_order_acquire))
            m->Reset(sampleRate, maxBlock);
    }

    void process(float* buf, int n) override {
        nam::DSP* m = _active.load(std::memory_order_acquire);
        if (!m || n > _maxBlock) return;        // sin modelo: pasa de largo
        float* in[]  = { buf };
        float* out[] = { _scratch.data() };
        m->process(in, out, n);
        std::memcpy(buf, _scratch.data(), static_cast<size_t>(n) * sizeof(float));
    }

    // --- Solo desde el hilo de control ------------------------------------
    // Devuelve un mensaje de error vacio si todo salio bien.
    std::string loadModel(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(_loadMutex);   // serializa cargas, no toca audio

        if (!std::filesystem::exists(path)) return "no existe: " + path.string();

        std::unique_ptr<nam::DSP> fresh;
        try {
            fresh = nam::get_dsp(path);
        } catch (std::exception& e) {
            return std::string("no se pudo cargar: ") + e.what();
        }
        if (!fresh) return "modelo nulo";
        if (fresh->NumInputChannels() != 1 || fresh->NumOutputChannels() != 1)
            return "solo se soportan modelos mono 1-in/1-out";

        // Dejarlo COMPLETAMENTE listo antes de publicarlo.
        if (_sr > 0.0) fresh->Reset(_sr, _maxBlock);

        nam::DSP* act = _active.load(std::memory_order_acquire);
        // Escribir siempre en la ranura que no esta sonando. Asignar aqui libera
        // el modelo retirado de la vez anterior, que ya nadie usa.
        std::unique_ptr<nam::DSP>& target = (act == _slotA.get()) ? _slotB : _slotA;
        target = std::move(fresh);

        _active.store(target.get(), std::memory_order_release);

        // Dejar que el hilo de audio salga del bloque en curso antes de que la
        // proxima carga libere el modelo viejo.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        _path = path.string();
        return "";
    }

    std::string path() const { return _path; }

    nam::DSP* dsp() { return _active.load(std::memory_order_acquire); }

    double loudness() {
        nam::DSP* m = dsp();
        return (m && m->HasLoudness()) ? m->GetLoudness() : 0.0;
    }

private:
    std::unique_ptr<nam::DSP> _slotA, _slotB;
    std::atomic<nam::DSP*> _active{nullptr};
    std::mutex _loadMutex;

    std::vector<float> _scratch;
    double _sr = 0.0;
    int    _maxBlock = 0;
    std::string _path;
};

} // namespace fx
