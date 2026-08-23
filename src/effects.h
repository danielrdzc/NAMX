#pragma once
//
// Bloques de efectos para el pedal NAM.
//
// Reglas del hilo de audio, sin excepciones:
//   - process() NO reserva memoria, NO toma mutexes, NO imprime, NO abre archivos.
//   - Todo lo que necesite memoria se reserva en prepare().
//   - Los parametros se escriben desde otro hilo con atomics y se suavizan aqui,
//     para que mover una perilla no truene.
//
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__aarch64__)
  #include <cstdint>
#elif defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP) || defined(__x86_64__)
  #include <xmmintrin.h>
  #include <pmmintrin.h>
#endif

namespace fx {

// MSVC no define M_PI salvo que se pida; mejor no depender de eso.
constexpr float kPi = 3.14159265358979323846f;

// ---------------------------------------------------------------------------
// Denormales.
//
// Delay y reverb tienen lazos de realimentacion. Cuando la señal decae hacia
// cero aparecen numeros denormales, que en varios procesadores son
// dramaticamente lentos -- xruns misteriosos justo cuando DEJAS de tocar.
// Esto le dice a la FPU que trate esos numeros como cero.
// Hay que llamarlo desde el hilo de audio; es por hilo, no global.
// ---------------------------------------------------------------------------
inline void enableFlushToZero() {
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24);                       // bit FZ
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__SSE__) || defined(_M_X64) || defined(__x86_64__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

inline float dbToGain(float db) { return std::pow(10.0f, db * 0.05f); }

// ---------------------------------------------------------------------------
// Parametro suavizado: se escribe desde cualquier hilo, se lee en el de audio.
// ---------------------------------------------------------------------------
class Smoothed {
public:
    void prepare(double sampleRate, double ms = 20.0) {
        _coef = static_cast<float>(std::exp(-1.0 / (sampleRate * ms * 0.001)));
        _cur  = _target.load(std::memory_order_relaxed);
    }
    void set(float v)      { _target.store(v, std::memory_order_relaxed); }
    void snap(float v)     { _target.store(v, std::memory_order_relaxed); _cur = v; }
    float target() const   { return _target.load(std::memory_order_relaxed); }
    inline float next() {
        const float t = _target.load(std::memory_order_relaxed);
        _cur = t + _coef * (_cur - t);
        return _cur;
    }
private:
    std::atomic<float> _target{0.0f};
    float _cur  = 0.0f;
    float _coef = 0.0f;
};

// ---------------------------------------------------------------------------
// Interfaz comun. Mono, in-place.
// ---------------------------------------------------------------------------
class Effect {
public:
    virtual ~Effect() = default;
    virtual const char* name() const = 0;
    virtual void prepare(double sampleRate, int maxBlock) = 0;
    virtual void reset() {}
    virtual void process(float* buf, int n) = 0;

    void setEnabled(bool e) { _enabled.store(e, std::memory_order_relaxed); }
    bool enabled() const    { return _enabled.load(std::memory_order_relaxed); }
private:
    std::atomic<bool> _enabled{true};
};

// ---------------------------------------------------------------------------
// Cadena de efectos, en orden.
// ---------------------------------------------------------------------------
class Chain {
public:
    void add(std::unique_ptr<Effect> e) { _fx.push_back(std::move(e)); }
    void prepare(double sampleRate, int maxBlock) {
        for (auto& e : _fx) e->prepare(sampleRate, maxBlock);
    }
    void reset() { for (auto& e : _fx) e->reset(); }
    inline void process(float* buf, int n) {
        for (auto& e : _fx) if (e->enabled()) e->process(buf, n);
    }
    size_t size() const { return _fx.size(); }
    Effect* at(size_t i) { return _fx[i].get(); }
private:
    std::vector<std::unique_ptr<Effect>> _fx;
};

// ---------------------------------------------------------------------------
// Noise gate.
//
// Seguidor de envolvente con ataque instantaneo y caida exponencial, histeresis
// para que no parpadee en el umbral, y tiempo de retencion para que las notas
// que se apagan no queden cortadas de golpe.
// ---------------------------------------------------------------------------
class NoiseGate : public Effect {
public:
    const char* name() const override { return "gate"; }

    void setThresholdDb(float db) { _thresholdDb.store(db, std::memory_order_relaxed); }
    void setHoldMs(float ms)      { _holdMs = ms; }
    void setReleaseMs(float ms)   { _releaseMs = ms; }

    void prepare(double sampleRate, int) override {
        _sr = sampleRate;
        _envCoef  = static_cast<float>(std::exp(-1.0 / (sampleRate * 0.030)));   // 30 ms
        _attCoef  = static_cast<float>(std::exp(-1.0 / (sampleRate * 0.002)));   // 2 ms
        _relCoef  = static_cast<float>(std::exp(-1.0 / (sampleRate * _releaseMs * 0.001)));
        _holdLen  = static_cast<int>(sampleRate * _holdMs * 0.001);
        reset();
    }

    void reset() override { _env = 0.0f; _gain = 0.0f; _hold = 0; }

    void process(float* buf, int n) override {
        const float openThr  = dbToGain(_thresholdDb.load(std::memory_order_relaxed));
        const float closeThr = openThr * 0.5f;      // 6 dB de histeresis
        for (int i = 0; i < n; ++i) {
            const float a = std::fabs(buf[i]);
            _env = (a > _env) ? a : _env * _envCoef;     // ataque instantaneo

            if (_env > openThr)        { _open = true;  _hold = _holdLen; }
            else if (_env < closeThr)  { if (_hold > 0) --_hold; else _open = false; }

            const float target = _open ? 1.0f : 0.0f;
            const float coef   = (target > _gain) ? _attCoef : _relCoef;
            _gain = target + coef * (_gain - target);
            buf[i] *= _gain;
        }
    }

private:
    std::atomic<float> _thresholdDb{-45.0f};
    float _holdMs = 40.0f, _releaseMs = 120.0f;
    double _sr = 48000.0;
    float _envCoef = 0, _attCoef = 0, _relCoef = 0;
    float _env = 0, _gain = 0;
    int   _hold = 0, _holdLen = 0;
    bool  _open = false;
};

// ---------------------------------------------------------------------------
// Overdrive estilo Tube Screamer.
//
// Lo que le da el caracter a un TS no es el recorte en si, es QUE recorta: la
// etapa de ganancia realza medios y agudos y deja los graves casi en unidad,
// asi que lo que se satura es la parte media del espectro. Por eso empuja al
// ampli sin embarrarlo. Aqui: pasa-altos -> se suma amplificado a la señal
// original -> recorte suave -> control de tono -> nivel.
//
// Nota: el tanh a 48 kHz sin sobremuestreo produce aliasing con drive alto.
// Se nota como aspereza en notas agudas. Si molesta, se agrega 2x oversampling.
// ---------------------------------------------------------------------------
class Overdrive : public Effect {
public:
    const char* name() const override { return "overdrive"; }

    void setDrive(float v) { _drive.set(clamp01(v)); }   // 0..1
    void setTone(float v)  { _tone.set(clamp01(v)); }    // 0..1
    void setLevel(float v) { _level.set(clamp01(v)); }   // 0..1

    void prepare(double sampleRate, int) override {
        _sr = sampleRate;
        // pasa-altos de un polo a ~720 Hz: la frecuencia de esquina del TS
        _hpCoef = static_cast<float>(std::exp(-2.0 * static_cast<double>(kPi) * 720.0 / sampleRate));
        _drive.prepare(sampleRate);
        _tone.prepare(sampleRate);
        _level.prepare(sampleRate);
        reset();
    }

    void reset() override { _hpX = _hpY = _lp = 0.0f; }

    void process(float* buf, int n) override {
        for (int i = 0; i < n; ++i) {
            const float x = buf[i];

            // pasa-altos de un polo
            _hpY = _hpCoef * (_hpY + x - _hpX);
            _hpX = x;

            // la ganancia de la etapa: 1x en graves, hasta ~30x en medios
            const float drive = _drive.next();
            const float boost = 1.0f + drive * 29.0f;
            float y = std::tanh(x + boost * _hpY);

            // tono: pasa-bajos de un polo entre 1.2 kHz y 8 kHz
            const float tone   = _tone.next();
            const float fc     = 1200.0f + tone * 6800.0f;
            const float lpCoef = 1.0f - std::exp(-2.0f * kPi * fc / static_cast<float>(_sr));
            _lp += lpCoef * (y - _lp);

            // compensacion: a mas drive, mas fuerte sale, asi que se baja
            const float makeup = 1.0f / (1.0f + drive * 2.0f);
            buf[i] = _lp * makeup * (0.2f + _level.next() * 1.8f);
        }
    }

private:
    static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
    Smoothed _drive, _tone, _level;
    double _sr = 48000.0;
    float _hpCoef = 0, _hpX = 0, _hpY = 0, _lp = 0;
};


// ---------------------------------------------------------------------------
// Phaser.
//
// Como funciona: una cadena de filtros pasa-todo de primer orden. Un pasa-todo
// no cambia la amplitud de nada -- solo corre la fase, y cuanto mas alta la
// frecuencia, mas la corre. Al sumar esa señal desfasada con la original, en las
// frecuencias donde quedaron en oposicion se cancelan: aparecen muescas. Un LFO
// barre la frecuencia de los pasa-todo, las muescas se mueven, y eso es el
// sonido. Cada PAR de etapas produce una muesca: 4 etapas = 2 muescas (el
// MXR Phase 90), 8 etapas = 4 (mas denso, mas Univibe).
//
// La realimentacion reinyecta la salida a la entrada y afila las muescas,
// haciendo el efecto mas resonante y vocal.
// ---------------------------------------------------------------------------
class Phaser : public Effect {
public:
    static constexpr int kMaxStages = 8;

    const char* name() const override { return "phaser"; }

    void setRateHz(float hz)   { _rate.store(clampf(hz, 0.02f, 12.0f), std::memory_order_relaxed); }
    void setDepth(float v)     { _depth.set(clampf(v, 0.0f, 1.0f)); }
    void setFeedback(float v)  { _fb.set(clampf(v, 0.0f, 0.85f)); }   // >0.9 se vuelve inestable
    void setMix(float v)       { _mix.set(clampf(v, 0.0f, 1.0f)); }
    void setStages(int s)      { _stages = (s < 2) ? 2 : (s > kMaxStages ? kMaxStages : (s & ~1)); }

    void prepare(double sampleRate, int) override {
        _sr = sampleRate;
        _depth.prepare(sampleRate, 30.0);
        _fb.prepare(sampleRate, 30.0);
        _mix.prepare(sampleRate, 30.0);
        reset();
    }

    void reset() override {
        for (int i = 0; i < kMaxStages; ++i) { _x1[i] = 0.0f; _y1[i] = 0.0f; }
        _phase = 0.0f; _last = 0.0f; _a = 0.0f; _counter = 0;
    }

    void process(float* buf, int n) override {
        const float rate = _rate.load(std::memory_order_relaxed);
        const float phaseInc = rate / static_cast<float>(_sr);

        for (int i = 0; i < n; ++i) {
            // El coeficiente del pasa-todo solo se recalcula cada 32 muestras.
            // El LFO va a pocos Hz, asi que 1.5 kHz de actualizacion sobra, y nos
            // ahorra un tan() por muestra.
            if (_counter <= 0) {
                _counter = 32;
                const float lfo = std::sin(2.0f * kPi * _phase);          // -1..1
                const float d   = _depth.next();
                // barrido logaritmico, que es como se percibe la frecuencia
                const float fmin = 200.0f;
                const float fmax = 200.0f + 1800.0f * d;
                const float t    = 0.5f * (lfo + 1.0f);
                const float fc   = fmin * std::pow(fmax / fmin, t);
                const float tanv = std::tan(kPi * fc / static_cast<float>(_sr));
                _a = (tanv - 1.0f) / (tanv + 1.0f);
            }
            --_counter;
            _phase += phaseInc;
            if (_phase >= 1.0f) _phase -= 1.0f;

            const float dry = buf[i];
            float x = dry + _fb.next() * _last;

            // cadena de pasa-todo: H(z) = (a + z^-1) / (1 + a z^-1)
            for (int s = 0; s < _stages; ++s) {
                const float y = _a * x + _x1[s] - _a * _y1[s];
                _x1[s] = x;
                _y1[s] = y;
                x = y;
            }
            _last = x;

            const float mix = _mix.next();
            buf[i] = dry * (1.0f - mix) + x * mix;
        }
    }

private:
    static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
    std::atomic<float> _rate{0.5f};
    Smoothed _depth, _fb, _mix;
    double _sr = 48000.0;
    int   _stages = 4;
    float _x1[kMaxStages]{}, _y1[kMaxStages]{};
    float _phase = 0.0f, _last = 0.0f, _a = 0.0f;
    int   _counter = 0;
};

// ---------------------------------------------------------------------------
// Ganancia simple. Sirve para normalizar la salida del modelo con GetLoudness()
// para que cambiar de capture no te cambie el volumen.
// ---------------------------------------------------------------------------
class Gain : public Effect {
public:
    const char* name() const override { return "gain"; }
    void setGainDb(float db) { _gain.set(dbToGain(db)); }
    void prepare(double sampleRate, int) override { _gain.prepare(sampleRate, 30.0); }
    void process(float* buf, int n) override {
        for (int i = 0; i < n; ++i) buf[i] *= _gain.next();
    }
private:
    Smoothed _gain;
};

} // namespace fx
