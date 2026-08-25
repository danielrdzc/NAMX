#pragma once
//
// Afinador.
//
// El analisis NO corre en el hilo de audio. El callback solo copia muestras a
// un buffer circular (un memcpy, nada mas), y la deteccion de tono se hace en
// el hilo de control cuando alguien pregunta. Detectar tono cuesta mas de un
// millon de operaciones: meterlo en el callback seria regalar margen de CPU
// para algo que se consulta unas pocas veces por segundo.
//
// METODO. Autocorrelacion normalizada (NSDF, la de McLeod). La autocorrelacion
// simple tiene un problema clasico: se confunde de octava, porque una senal
// periodica tambien se parece a si misma al doble del periodo. Normalizar por
// la energia de las dos ventanas comparadas y quedarse con el PRIMER pico que
// supera un umbral -- no con el mas alto -- resuelve casi todos esos errores.
//
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace fx {

struct TunerReading {
    bool  valid = false;
    float freq  = 0.0f;      // Hz
    float cents = 0.0f;      // desviacion respecto a la nota mas cercana
    float clarity = 0.0f;    // 0..1, que tan periodica es la senal
    std::string note;        // "E2", "A4", ...
};

class Tuner {
public:
    static constexpr int kBuf    = 16384;   // ~341 ms de historia
    static constexpr int kWindow = 4096;    // ventana de analisis

    void prepare(double sampleRate) {
        _sr = sampleRate;
        _ring.assign(kBuf, 0.0f);
        _write.store(0);
    }

    // --- hilo de audio: solo copiar --------------------------------------
    inline void push(const float* buf, int n) {
        if (_ring.empty()) return;
        int w = _write.load(std::memory_order_relaxed);
        for (int i = 0; i < n; ++i) {
            _ring[static_cast<size_t>(w)] = buf[i];
            if (++w >= kBuf) w = 0;
        }
        _write.store(w, std::memory_order_release);
    }

    // --- hilo de control: aqui si se puede pensar -------------------------
    TunerReading detect() {
        TunerReading r;
        if (_ring.empty()) return r;

        // Leer por detras del puntero de escritura para minimizar el solape.
        const int w = _write.load(std::memory_order_acquire);
        std::vector<float> x(static_cast<size_t>(kWindow));
        for (int i = 0; i < kWindow; ++i) {
            int idx = w - kWindow + i;
            while (idx < 0) idx += kBuf;
            x[static_cast<size_t>(i)] = _ring[static_cast<size_t>(idx % kBuf)];
        }

        // Sin senal suficiente no tiene sentido analizar.
        double rms = 0.0;
        for (float v : x) rms += static_cast<double>(v) * v;
        rms = std::sqrt(rms / kWindow);
        if (rms < 0.0015) return r;                  // ~ -56 dBFS

        // Rango util de una guitarra: de un Si grave (drop) a un armonico alto.
        const int minLag = static_cast<int>(_sr / 1300.0);
        const int maxLag = std::min(static_cast<int>(_sr / 55.0), kWindow / 2);
        if (maxLag <= minLag) return r;

        const int half = kWindow / 2;
        std::vector<float> nsdf(static_cast<size_t>(maxLag + 1), 0.0f);
        for (int lag = minLag; lag <= maxLag; ++lag) {
            double ac = 0.0, e1 = 0.0, e2 = 0.0;
            for (int i = 0; i < half; ++i) {
                const double a = x[static_cast<size_t>(i)];
                const double b = x[static_cast<size_t>(i + lag)];
                ac += a * b;  e1 += a * a;  e2 += b * b;
            }
            const double den = e1 + e2;
            nsdf[static_cast<size_t>(lag)] =
                (den > 1e-12) ? static_cast<float>(2.0 * ac / den) : 0.0f;
        }

        // Maximo global, para fijar el umbral relativo.
        float peak = 0.0f;
        for (int lag = minLag; lag <= maxLag; ++lag) peak = std::max(peak, nsdf[static_cast<size_t>(lag)]);
        if (peak < 0.5f) return r;                   // poco periodico: ruido o acorde
        const float thresh = peak * 0.9f;

        // El PRIMER pico que pasa el umbral, no el mas alto: asi no se va de octava.
        int best = -1;
        for (int lag = minLag + 1; lag < maxLag; ++lag) {
            const float v = nsdf[static_cast<size_t>(lag)];
            if (v > thresh && v >= nsdf[static_cast<size_t>(lag - 1)]
                           && v >= nsdf[static_cast<size_t>(lag + 1)]) { best = lag; break; }
        }
        if (best < 0) return r;

        // Interpolacion parabolica: el pico real casi nunca cae justo en una
        // muestra, y sin esto la lectura salta de a saltos de varios cents.
        const float y0 = nsdf[static_cast<size_t>(best - 1)];
        const float y1 = nsdf[static_cast<size_t>(best)];
        const float y2 = nsdf[static_cast<size_t>(best + 1)];
        const float den = 2.0f * (2.0f * y1 - y0 - y2);
        const float shift = (std::fabs(den) > 1e-9f) ? (y2 - y0) / den : 0.0f;

        const double lag = best + static_cast<double>(shift);
        if (lag <= 0.0) return r;

        r.freq    = static_cast<float>(_sr / lag);
        r.clarity = y1;
        r.valid   = true;

        // Frecuencia -> nota. 69 es A4 en MIDI, 440 Hz.
        const double midi = 69.0 + 12.0 * std::log2(r.freq / 440.0);
        const int    near = static_cast<int>(std::lround(midi));
        r.cents = static_cast<float>((midi - near) * 100.0);

        static const char* kNames[12] =
            {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
        const int pc  = ((near % 12) + 12) % 12;
        const int oct = near / 12 - 1;
        r.note = std::string(kNames[pc]) + std::to_string(oct);
        return r;
    }

private:
    std::vector<float> _ring;
    std::atomic<int>   _write{0};
    double _sr = 48000.0;
};

} // namespace fx
