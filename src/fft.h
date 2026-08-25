#pragma once
//
// FFT radix-2 iterativa, in-place. Lo minimo para hacer convolucion.
//
// No uso una libreria externa a proposito: esto son 60 lineas, no tiene
// dependencias, y compila igual en el Pi que en Windows. FFTW seria mas rapida,
// pero a los tamanos que usamos (128-512 puntos) la diferencia es irrelevante
// comparada con la inferencia de NAM.
//
#include <cmath>
#include <cstddef>
#include <vector>

namespace fx {

class FFT {
public:
    // n debe ser potencia de 2.
    void prepare(int n) {
        _n = n;
        _cosT.resize(static_cast<size_t>(n / 2));
        _sinT.resize(static_cast<size_t>(n / 2));
        for (int i = 0; i < n / 2; ++i) {
            const double a = -2.0 * 3.14159265358979323846 * i / n;
            _cosT[static_cast<size_t>(i)] = static_cast<float>(std::cos(a));
            _sinT[static_cast<size_t>(i)] = static_cast<float>(std::sin(a));
        }
        _rev.resize(static_cast<size_t>(n));
        int bits = 0;
        while ((1 << bits) < n) ++bits;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
            _rev[static_cast<size_t>(i)] = r;
        }
    }

    int size() const { return _n; }

    // Transformada en el sitio. inverse=true aplica el signo opuesto y NO escala.
    void run(float* re, float* im, bool inverse) const {
        for (int i = 0; i < _n; ++i) {
            const int j = _rev[static_cast<size_t>(i)];
            if (j > i) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= _n; len <<= 1) {
            const int half = len >> 1;
            const int step = _n / len;
            for (int i = 0; i < _n; i += len) {
                for (int k = 0; k < half; ++k) {
                    const int t = k * step;
                    const float wr = _cosT[static_cast<size_t>(t)];
                    const float wi = inverse ? -_sinT[static_cast<size_t>(t)]
                                             :  _sinT[static_cast<size_t>(t)];
                    const int a = i + k, b = i + k + half;
                    const float xr = re[b] * wr - im[b] * wi;
                    const float xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;  im[b] = im[a] - xi;
                    re[a] += xr;         im[a] += xi;
                }
            }
        }
        if (inverse) {
            const float s = 1.0f / static_cast<float>(_n);
            for (int i = 0; i < _n; ++i) { re[i] *= s; im[i] *= s; }
        }
    }

private:
    int _n = 0;
    std::vector<float> _cosT, _sinT;
    std::vector<int>   _rev;
};

} // namespace fx
