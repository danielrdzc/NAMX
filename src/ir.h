#pragma once
//
// Cargador de IR (respuesta al impulso de gabinete), con convolucion
// particionada por FFT.
//
// POR QUE PARTICIONADA. Convolucionar directamente una IR de 4096 muestras
// cuesta 4096 multiplicaciones POR MUESTRA -- imposible en tiempo real. Hacer
// una sola FFT de toda la IR es eficiente pero obliga a esperar a tener 4096
// muestras de entrada antes de producir la primera salida: 85 ms de latencia,
// inaceptable en un pedal.
//
// La solucion: partir la IR en trozos del tamano del bloque de audio. Cada
// bloque que llega se convoluciona con todos los trozos y los resultados se
// suman desplazados en el tiempo. Sale la misma respuesta completa, con la
// eficiencia de la FFT y CERO latencia anadida.
//
// El "frequency delay line" (_fdl) guarda los espectros de los ultimos K
// bloques de entrada, para poder multiplicar cada uno por su trozo de IR.
//
#include "effects.h"
#include "fft.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fx {

// ---------------------------------------------------------------------------
// Lector de WAV: PCM 16/24/32 bits y float 32. Devuelve el canal 0 en float.
// ---------------------------------------------------------------------------
inline bool loadWavMono(const std::filesystem::path& path,
                        std::vector<float>& out, int& sampleRate, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "no se pudo abrir"; return false; }
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 44 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4)) {
        err = "no es un WAV"; return false;
    }

    auto rd32 = [&](size_t p) { uint32_t v; std::memcpy(&v, d.data() + p, 4); return v; };
    auto rd16 = [&](size_t p) { uint16_t v; std::memcpy(&v, d.data() + p, 2); return v; };

    size_t pos = 12;
    int channels = 0, bits = 0, format = 0;
    size_t dataPos = 0, dataLen = 0;
    while (pos + 8 <= d.size()) {
        const uint32_t sz = rd32(pos + 4);
        if (!std::memcmp(d.data() + pos, "fmt ", 4)) {
            format     = rd16(pos + 8);
            channels   = rd16(pos + 10);
            sampleRate = static_cast<int>(rd32(pos + 12));
            bits       = rd16(pos + 22);
        } else if (!std::memcmp(d.data() + pos, "data", 4)) {
            dataPos = pos + 8;
            dataLen = sz;
        }
        pos += 8 + sz + (sz & 1);
    }
    if (!dataPos || channels < 1) { err = "WAV sin datos utiles"; return false; }
    if (dataPos + dataLen > d.size()) dataLen = d.size() - dataPos;

    const int bytes = bits / 8;
    if (bytes < 2 || bytes > 4) { err = "solo 16, 24 o 32 bits"; return false; }
    const size_t frames = dataLen / static_cast<size_t>(bytes * channels);
    out.resize(frames);

    for (size_t i = 0; i < frames; ++i) {
        const size_t p = dataPos + i * static_cast<size_t>(bytes * channels);   // canal 0
        if (format == 3 && bytes == 4) {                       // float32
            float v; std::memcpy(&v, d.data() + p, 4); out[i] = v;
        } else if (bytes == 2) {
            int16_t v; std::memcpy(&v, d.data() + p, 2); out[i] = v / 32768.0f;
        } else if (bytes == 3) {
            const int32_t v = (static_cast<int32_t>(static_cast<uint8_t>(d[p])) << 8)
                            | (static_cast<int32_t>(static_cast<uint8_t>(d[p + 1])) << 16)
                            | (static_cast<int32_t>(static_cast<int8_t>(d[p + 2])) << 24);
            out[i] = static_cast<float>(v >> 8) / 8388608.0f;
        } else {
            int32_t v; std::memcpy(&v, d.data() + p, 4); out[i] = v / 2147483648.0f;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Convolucion particionada uniforme (overlap-save).
// ---------------------------------------------------------------------------
class IRLoader : public Effect {
public:
    // 4096 muestras son 85 ms: de sobra para un gabinete de guitarra, y acota
    // el costo de CPU a algo predecible.
    static constexpr int kMaxIrLen = 4096;

    IRLoader() { setEnabled(false); _gain.set(1.0f); }   // sin IR cargada, no hace nada

    const char* name() const override { return "ir"; }

    void prepare(double sampleRate, int maxBlock) override {
        _sr = sampleRate;
        _P  = nextPow2(maxBlock);
        _N  = _P * 2;
        _fft.prepare(_N);
        _gain.prepare(sampleRate, 30.0);

        _xre.assign(static_cast<size_t>(_N), 0.0f);
        _xim.assign(static_cast<size_t>(_N), 0.0f);
        _yre.assign(static_cast<size_t>(_N), 0.0f);
        _yim.assign(static_cast<size_t>(_N), 0.0f);
        _hist.assign(static_cast<size_t>(_N), 0.0f);
        _fifoIn.assign(static_cast<size_t>(_P), 0.0f);
        _fifoOut.assign(static_cast<size_t>(_P), 0.0f);

        rebuild();
    }

    void reset() override {
        std::fill(_hist.begin(), _hist.end(), 0.0f);
        for (auto& v : _fdlRe) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : _fdlIm) std::fill(v.begin(), v.end(), 0.0f);
        _fdlPos = 0; _fifoCount = 0;
        std::fill(_fifoOut.begin(), _fifoOut.end(), 0.0f);
    }

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"ir.level", "Level", "dB", -24.0f, 12.0f, 0.1f,
            [this](float v) { _gain.set(dbToGain(v)); _db = v; },
            [this]() { return _db; }});
    }

    void process(float* buf, int n) override {
        if (_K == 0) return;                     // sin IR cargada

        if (n % _P == 0) {                       // camino normal: bloques alineados
            for (int off = 0; off < n; off += _P) convolve(buf + off, buf + off);
        } else {
            // Bloque de tamano raro: se acumula. Cuesta hasta P muestras de
            // latencia, pero con JACK el tamano es constante y no ocurre.
            for (int i = 0; i < n; ++i) {
                _fifoIn[static_cast<size_t>(_fifoCount)] = buf[i];
                buf[i] = _fifoOut[static_cast<size_t>(_fifoCount)];
                if (++_fifoCount == _P) {
                    convolve(_fifoIn.data(), _fifoOut.data());
                    _fifoCount = 0;
                }
            }
        }
        for (int i = 0; i < n; ++i) buf[i] *= _gain.next();
    }

    // --- hilo de control ---------------------------------------------------
    std::string loadIR(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (path.empty()) {                       // descargar
            setEnabled(false);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            _pending.clear(); _ir.clear(); _K = 0; _path.clear();
            return "";
        }
        std::vector<float> data;
        int sr = 0;
        std::string err;
        if (!loadWavMono(path, data, sr, err)) return err;
        if (data.empty()) return "IR vacia";

        // Reajuste de tasa. Muchas IR vienen a 44.1 kHz; interpolar linealmente
        // una IR es aceptable -- es una respuesta corta y suave.
        if (sr > 0 && sr != static_cast<int>(_sr)) {
            const double ratio = _sr / static_cast<double>(sr);
            std::vector<float> rs(static_cast<size_t>(data.size() * ratio));
            for (size_t i = 0; i < rs.size(); ++i) {
                const double src = i / ratio;
                const size_t i0 = static_cast<size_t>(src);
                const double fr = src - static_cast<double>(i0);
                const float a = data[std::min(i0, data.size() - 1)];
                const float b = data[std::min(i0 + 1, data.size() - 1)];
                rs[i] = static_cast<float>(a + fr * (b - a));
            }
            data.swap(rs);
        }
        if (static_cast<int>(data.size()) > kMaxIrLen) data.resize(kMaxIrLen);

        // Normalizar por energia: cambiar de IR no deberia cambiar el volumen.
        double sum = 0.0;
        for (float v : data) sum += static_cast<double>(v) * v;
        if (sum > 1e-12) {
            const float k = static_cast<float>(1.0 / std::sqrt(sum));
            for (float& v : data) v *= k;
        }

        setEnabled(false);                       // silenciar durante el cambio
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        _ir = std::move(data);
        rebuild();
        _path = path.string();
        setEnabled(true);
        return "";
    }

    std::string path() const { return _path; }

private:
    static int nextPow2(int v) { int p = 1; while (p < v) p <<= 1; return p; }

    // Precalcula el espectro de cada trozo de la IR.
    void rebuild() {
        _K = 0;
        _fdlRe.clear(); _fdlIm.clear(); _hre.clear(); _him.clear();
        if (_ir.empty() || _N == 0) return;

        _K = (static_cast<int>(_ir.size()) + _P - 1) / _P;
        _hre.assign(static_cast<size_t>(_K), std::vector<float>(static_cast<size_t>(_N), 0.0f));
        _him.assign(static_cast<size_t>(_K), std::vector<float>(static_cast<size_t>(_N), 0.0f));
        for (int k = 0; k < _K; ++k) {
            for (int i = 0; i < _P; ++i) {
                const size_t src = static_cast<size_t>(k * _P + i);
                _hre[static_cast<size_t>(k)][static_cast<size_t>(i)] =
                    (src < _ir.size()) ? _ir[src] : 0.0f;
            }
            _fft.run(_hre[static_cast<size_t>(k)].data(), _him[static_cast<size_t>(k)].data(), false);
        }
        _fdlRe.assign(static_cast<size_t>(_K), std::vector<float>(static_cast<size_t>(_N), 0.0f));
        _fdlIm.assign(static_cast<size_t>(_K), std::vector<float>(static_cast<size_t>(_N), 0.0f));
        _fdlPos = 0;
        std::fill(_hist.begin(), _hist.end(), 0.0f);
    }

    // Un bloque de exactamente _P muestras, overlap-save.
    void convolve(const float* in, float* out) {
        // La ventana son las P muestras anteriores + las P nuevas.
        std::memmove(_hist.data(), _hist.data() + _P, static_cast<size_t>(_P) * sizeof(float));
        std::memcpy(_hist.data() + _P, in, static_cast<size_t>(_P) * sizeof(float));

        std::memcpy(_xre.data(), _hist.data(), static_cast<size_t>(_N) * sizeof(float));
        std::fill(_xim.begin(), _xim.end(), 0.0f);
        _fft.run(_xre.data(), _xim.data(), false);

        std::memcpy(_fdlRe[static_cast<size_t>(_fdlPos)].data(), _xre.data(),
                    static_cast<size_t>(_N) * sizeof(float));
        std::memcpy(_fdlIm[static_cast<size_t>(_fdlPos)].data(), _xim.data(),
                    static_cast<size_t>(_N) * sizeof(float));

        std::fill(_yre.begin(), _yre.end(), 0.0f);
        std::fill(_yim.begin(), _yim.end(), 0.0f);
        for (int k = 0; k < _K; ++k) {
            const int slot = (_fdlPos - k + _K * 2) % _K;
            const float* ar = _fdlRe[static_cast<size_t>(slot)].data();
            const float* ai = _fdlIm[static_cast<size_t>(slot)].data();
            const float* br = _hre[static_cast<size_t>(k)].data();
            const float* bi = _him[static_cast<size_t>(k)].data();
            for (int i = 0; i < _N; ++i) {
                _yre[static_cast<size_t>(i)] += ar[i] * br[i] - ai[i] * bi[i];
                _yim[static_cast<size_t>(i)] += ar[i] * bi[i] + ai[i] * br[i];
            }
        }
        _fft.run(_yre.data(), _yim.data(), true);

        // Overlap-save: la mitad valida es la segunda.
        std::memcpy(out, _yre.data() + _P, static_cast<size_t>(_P) * sizeof(float));

        _fdlPos = (_fdlPos + 1) % _K;
    }

    FFT _fft;
    std::mutex _mutex;
    std::vector<float> _ir, _pending;
    std::vector<std::vector<float>> _hre, _him, _fdlRe, _fdlIm;
    std::vector<float> _xre, _xim, _yre, _yim, _hist, _fifoIn, _fifoOut;
    Smoothed _gain;
    double _sr = 48000.0;
    int _P = 0, _N = 0, _K = 0, _fdlPos = 0, _fifoCount = 0;
    float _db = 0.0f;
    std::string _path;
};

} // namespace fx
