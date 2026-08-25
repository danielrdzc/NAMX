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
#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include "params.h"

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

    // Perillas que este efecto expone. Se llama una vez, al armar la cadena.
    virtual void collectParams(std::vector<Param>&) {}
private:
    std::atomic<bool> _enabled{true};
};

// ---------------------------------------------------------------------------
// Cadena de efectos, en orden.
// ---------------------------------------------------------------------------
// El orden se puede cambiar en vivo. Los efectos NUNCA se mueven de _fx: lo que
// cambia es una lista de indices. Hay dos listas y un atomico que dice cual esta
// viva; el hilo de UI escribe SIEMPRE la que no se esta usando y luego voltea el
// atomico. El hilo de audio lee el atomico una vez por bloque. Sin candados, sin
// reservar memoria, y sin que el audio vea nunca una lista a medio escribir.
class Chain {
public:
    void add(std::unique_ptr<Effect> e) {
        _fx.push_back(std::move(e));
        _orderA.push_back(static_cast<int>(_fx.size()) - 1);
        _orderB = _orderA;
    }

    void prepare(double sampleRate, int maxBlock) {
        for (auto& e : _fx) e->prepare(sampleRate, maxBlock);
    }
    void reset() { for (auto& e : _fx) e->reset(); }

    inline void process(float* buf, int n) {
        const std::vector<int>& ord = _live.load(std::memory_order_acquire) ? _orderB : _orderA;
        for (int idx : ord) {
            Effect* e = _fx[static_cast<size_t>(idx)].get();
            if (e->enabled()) e->process(buf, n);
        }
    }

    size_t size() const { return _fx.size(); }

    // En orden de senal, no en orden de construccion.
    Effect* at(size_t pos) {
        const std::vector<int>& ord = _live.load(std::memory_order_acquire) ? _orderB : _orderA;
        return _fx[static_cast<size_t>(ord[pos])].get();
    }

    std::vector<std::string> order() {
        const std::vector<int>& ord = _live.load(std::memory_order_acquire) ? _orderB : _orderA;
        std::vector<std::string> names;
        names.reserve(ord.size());
        for (int i : ord) names.push_back(_fx[static_cast<size_t>(i)]->name());
        return names;
    }

    // Devuelve false si la lista no es una permutacion exacta de los efectos.
    // Se llama desde el hilo de control, nunca desde el de audio.
    bool setOrder(const std::vector<std::string>& names) {
        if (names.size() != _fx.size()) return false;
        std::vector<int> built;
        built.reserve(names.size());
        for (const auto& nm : names) {
            int found = -1;
            for (size_t i = 0; i < _fx.size(); ++i)
                if (nm == _fx[i]->name()) { found = static_cast<int>(i); break; }
            if (found < 0) return false;
            for (int b : built) if (b == found) return false;   // repetido
            built.push_back(found);
        }

        const bool liveIsB = _live.load(std::memory_order_acquire);
        (liveIsB ? _orderA : _orderB) = built;                  // escribir la inactiva
        _live.store(!liveIsB, std::memory_order_release);       // y voltear

        // Antes de volver a tocar la otra lista hay que dejar que el hilo de
        // audio salga del bloque en curso. Un bloque dura microsegundos; esperar
        // aqui es gratis para un arrastre del raton y elimina la carrera.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return true;
    }

    std::vector<Param> collectParams() {
        std::vector<Param> out;
        const std::vector<int>& ord = _live.load(std::memory_order_acquire) ? _orderB : _orderA;
        for (int idx : ord) {
            Effect* raw = _fx[static_cast<size_t>(idx)].get();
            out.push_back(Param{
                std::string(raw->name()) + ".enabled", "On", "", 0.0f, 1.0f, 1.0f,
                [raw](float v) { raw->setEnabled(v >= 0.5f); },
                [raw]() { return raw->enabled() ? 1.0f : 0.0f; }});
            raw->collectParams(out);
        }
        return out;
    }

private:
    std::vector<std::unique_ptr<Effect>> _fx;
    std::vector<int> _orderA, _orderB;
    std::atomic<bool> _live{false};        // false -> A, true -> B
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

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"gate.threshold", "Threshold", "dB", -80.0f, 0.0f, 0.5f,
            [this](float v) { setThresholdDb(v); },
            [this]() { return _thresholdDb.load(std::memory_order_relaxed); }});
    }

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

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"overdrive.drive", "Drive", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setDrive(v); }, [this]() { return _drive.target(); }});
        out.push_back(Param{"overdrive.tone", "Tone", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setTone(v); }, [this]() { return _tone.target(); }});
        out.push_back(Param{"overdrive.level", "Level", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setLevel(v); }, [this]() { return _level.target(); }});
    }

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

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"phaser.rate", "Rate", "Hz", 0.02f, 12.0f, 0.01f,
            [this](float v) { setRateHz(v); },
            [this]() { return _rate.load(std::memory_order_relaxed); }});
        out.push_back(Param{"phaser.depth", "Depth", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setDepth(v); }, [this]() { return _depth.target(); }});
        out.push_back(Param{"phaser.feedback", "Feedback", "", 0.0f, 0.85f, 0.01f,
            [this](float v) { setFeedback(v); }, [this]() { return _fb.target(); }});
        out.push_back(Param{"phaser.mix", "Mix", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setMix(v); }, [this]() { return _mix.target(); }});
        out.push_back(Param{"phaser.stages", "Stages", "", 2.0f, 8.0f, 2.0f,
            [this](float v) { setStages(static_cast<int>(v)); },
            [this]() { return static_cast<float>(_stages); }});
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
// Delay.
//
// Buffer circular con lectura FRACCIONARIA e interpolacion lineal. Eso permite
// dos cosas: tiempos de retardo que no caen en muestras exactas, y que al mover
// el tiempo la lectura se deslice en vez de saltar -- suena como una cinta
// acelerando o frenando, que es justo lo que hace un delay analogico.
//
// El filtro pasa-bajos va DENTRO del lazo de realimentacion, asi que cada
// repeticion sale mas oscura que la anterior. Sin eso las repeticiones suenan
// digitales y frias; con eso se van apagando como en una cinta.
// ---------------------------------------------------------------------------
class Delay : public Effect {
public:
    const char* name() const override { return "delay"; }

    void setTimeMs(float ms)   { _timeMs = clampf(ms, 1.0f, 1990.0f); }
    void setFeedback(float v)  { _fb.set(clampf(v, 0.0f, 0.95f)); }   // 1.0 se realimenta sin fin
    void setMix(float v)       { _mix.set(clampf(v, 0.0f, 1.0f)); }
    void setTone(float v)      { _tone = clampf(v, 0.0f, 1.0f); }     // 0 = repeticiones oscuras

    void prepare(double sampleRate, int) override {
        _sr   = sampleRate;
        _size = static_cast<int>(sampleRate * 2.0) + 4;               // 2 segundos
        _buf.assign(static_cast<size_t>(_size), 0.0f);
        _fb.prepare(sampleRate, 40.0);
        _mix.prepare(sampleRate, 40.0);
        // el tiempo se suaviza LENTO a proposito: es lo que produce el glissando
        _time.prepare(sampleRate, 250.0);
        _time.snap(static_cast<float>(_timeMs * 0.001 * sampleRate));
        const float fc = 800.0f + _tone * 7200.0f;
        _lpCoef = 1.0f - std::exp(-2.0f * kPi * fc / static_cast<float>(sampleRate));
        reset();
    }

    void reset() override {
        std::fill(_buf.begin(), _buf.end(), 0.0f);
        _write = 0; _lp = 0.0f;
    }

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"delay.time", "Time", "ms", 1.0f, 1990.0f, 1.0f,
            [this](float v) { setTimeMs(v); }, [this]() { return _timeMs; }});
        out.push_back(Param{"delay.feedback", "Feedback", "", 0.0f, 0.95f, 0.01f,
            [this](float v) { setFeedback(v); }, [this]() { return _fb.target(); }});
        out.push_back(Param{"delay.mix", "Mix", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setMix(v); }, [this]() { return _mix.target(); }});
        out.push_back(Param{"delay.tone", "Tone", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setTone(v); _updateTone(); }, [this]() { return _tone; }});
    }

    void process(float* buf, int n) override {
        _time.set(static_cast<float>(_timeMs * 0.001 * _sr));
        for (int i = 0; i < n; ++i) {
            const float dry   = buf[i];
            const float delay = _time.next();

            float readPos = static_cast<float>(_write) - delay;
            while (readPos < 0.0f) readPos += static_cast<float>(_size);

            const int   i0   = static_cast<int>(readPos);
            const float frac = readPos - static_cast<float>(i0);
            const int   i1   = (i0 + 1) % _size;
            const float wet  = _buf[i0] + frac * (_buf[i1] - _buf[i0]);

            // pasa-bajos dentro del lazo: cada repeticion mas oscura
            _lp += _lpCoef * (wet - _lp);

            _buf[_write] = dry + _lp * _fb.next();
            if (++_write >= _size) _write = 0;

            const float mix = _mix.next();
            buf[i] = dry + wet * mix;
        }
    }

private:
    static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
    void _updateTone() {
        const float fc = 800.0f + _tone * 7200.0f;
        _lpCoef = 1.0f - std::exp(-2.0f * kPi * fc / static_cast<float>(_sr));
    }
    std::vector<float> _buf;
    Smoothed _fb, _mix, _time;
    double _sr = 48000.0;
    float _timeMs = 400.0f, _tone = 0.5f, _lpCoef = 0.0f, _lp = 0.0f;
    int _size = 0, _write = 0;
};

// ---------------------------------------------------------------------------
// Reverb (topologia Freeverb, de Jezar).
//
// La idea: ocho filtros peine en PARALELO generan la cola de ecos densos --
// cada uno con un retardo distinto y primo entre si, para que sus repeticiones
// no coincidan y no se oiga un patron. Despues, cuatro pasa-todo EN SERIE
// difuminan el resultado: no cambian el espectro, solo desparraman los ecos en
// el tiempo hasta que dejan de oirse como ecos y empiezan a oirse como espacio.
//
// El amortiguamiento (damp) es un pasa-bajos dentro de cada peine: los agudos
// se apagan antes que los graves, igual que en un cuarto real donde las
// superficies absorben mas las frecuencias altas.
// ---------------------------------------------------------------------------
class Reverb : public Effect {
public:
    const char* name() const override { return "reverb"; }

    void setRoomSize(float v) { _rawRoom = clampf(v, 0.0f, 1.0f); _room.set(_rawRoom * 0.28f + 0.7f); }
    void setDamp(float v)     { _rawDamp = clampf(v, 0.0f, 1.0f); _damp.set(_rawDamp * 0.4f); }
    void setMix(float v)      { _mix.set(clampf(v, 0.0f, 1.0f)); }

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"reverb.mix", "Mix", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setMix(v); }, [this]() { return _mix.target(); }});
        out.push_back(Param{"reverb.room", "Room", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setRoomSize(v); }, [this]() { return _rawRoom; }});
        out.push_back(Param{"reverb.damp", "Damp", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setDamp(v); }, [this]() { return _rawDamp; }});
    }

    void prepare(double sampleRate, int) override {
        // Las longitudes originales de Freeverb son para 44.1 kHz; se escalan.
        const double k = sampleRate / 44100.0;
        static const int combTune[kCombs] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
        static const int apTune[kAllpass] = {556, 441, 341, 225};

        for (int i = 0; i < kCombs; ++i) {
            _combLen[i] = static_cast<int>(combTune[i] * k);
            _comb[i].assign(static_cast<size_t>(_combLen[i]), 0.0f);
        }
        for (int i = 0; i < kAllpass; ++i) {
            _apLen[i] = static_cast<int>(apTune[i] * k);
            _ap[i].assign(static_cast<size_t>(_apLen[i]), 0.0f);
        }
        _room.prepare(sampleRate, 50.0);
        _damp.prepare(sampleRate, 50.0);
        _mix.prepare(sampleRate, 50.0);
        reset();
    }

    void reset() override {
        for (int i = 0; i < kCombs; ++i) {
            std::fill(_comb[i].begin(), _comb[i].end(), 0.0f);
            _combIdx[i] = 0; _store[i] = 0.0f;
        }
        for (int i = 0; i < kAllpass; ++i) {
            std::fill(_ap[i].begin(), _ap[i].end(), 0.0f);
            _apIdx[i] = 0;
        }
    }

    void process(float* buf, int n) override {
        for (int i = 0; i < n; ++i) {
            const float dry  = buf[i];
            const float in   = dry * kFixedGain;
            const float room = _room.next();
            const float damp = _damp.next();

            // peines en paralelo
            float acc = 0.0f;
            for (int c = 0; c < kCombs; ++c) {
                const float out = _comb[c][_combIdx[c]];
                _store[c] = out * (1.0f - damp) + _store[c] * damp;   // amortiguamiento
                _comb[c][_combIdx[c]] = in + _store[c] * room;
                if (++_combIdx[c] >= _combLen[c]) _combIdx[c] = 0;
                acc += out;
            }

            // pasa-todo en serie: difusion
            float y = acc;
            for (int a = 0; a < kAllpass; ++a) {
                const float bufout = _ap[a][_apIdx[a]];
                const float out    = bufout - y;
                _ap[a][_apIdx[a]]  = y + bufout * 0.5f;
                if (++_apIdx[a] >= _apLen[a]) _apIdx[a] = 0;
                y = out;
            }

            const float mix = _mix.next();
            buf[i] = dry * (1.0f - mix * 0.5f) + y * mix;
        }
    }

private:
    static constexpr int   kCombs      = 8;
    static constexpr int   kAllpass    = 4;
    static constexpr float kFixedGain  = 0.015f;
    static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

    std::vector<float> _comb[kCombs], _ap[kAllpass];
    int   _combLen[kCombs]{}, _combIdx[kCombs]{}, _apLen[kAllpass]{}, _apIdx[kAllpass]{};
    float _store[kCombs]{};
    Smoothed _room, _damp, _mix;
    float _rawRoom = 0.6f, _rawDamp = 0.5f;
};


// ---------------------------------------------------------------------------
// Fuzz.
//
// Un fuzz no es un overdrive con mas ganancia. Un overdrive redondea las puntas
// de la onda; un fuzz las corta de tajo y convierte la senal en algo cercano a
// una onda cuadrada. Dos cosas definen su caracter:
//
// 1. BIAS. En un Fuzz Face el bias es el punto de trabajo del transistor. Si se
//    desplaza, el recorte deja de ser simetrico: la mitad positiva y la negativa
//    se cortan a alturas distintas. Eso genera armonicos PARES, que es lo que da
//    ese sonido vocal y nasal. Llevado al extremo, el transistor se queda sin
//    margen y la senal se corta a pedazos -- el "starved fuzz" que escupe y se
//    apaga entre notas. Esa aspereza es la gracia, no un defecto.
//
// 2. BLOQUEO DE DC. Recortar asimetricamente mete un offset de continua en la
//    senal. Si no se quita, todo lo que venga despues trabaja descentrado: el
//    ampli se satura antes de un lado, el gate lee mal, y el altavoz recibe DC.
//    Por eso el bloqueador de DC despues del recorte no es opcional.
//
// El pasa-altos de entrada emula el capacitor de acoplamiento: sin el, los
// graves saturan primero y el fuzz suena a manta mojada.
// ---------------------------------------------------------------------------
class Fuzz : public Effect {
public:
    const char* name() const override { return "fuzz"; }

    void setDrive(float v) { _drive.set(clamp01(v)); }
    void setBias(float v)  { _bias.set(clampf(v, -1.0f, 1.0f)); }   // 0 = simetrico
    void setTone(float v)  { _tone.set(clamp01(v)); }
    void setLevel(float v) { _level.set(clamp01(v)); }

    void prepare(double sampleRate, int) override {
        _sr = sampleRate;
        _hpCoef = static_cast<float>(std::exp(-2.0 * static_cast<double>(kPi) * 100.0 / sampleRate));
        _dcCoef = static_cast<float>(std::exp(-2.0 * static_cast<double>(kPi) * 12.0 / sampleRate));
        _drive.prepare(sampleRate);
        _bias.prepare(sampleRate);
        _tone.prepare(sampleRate);
        _level.prepare(sampleRate);
        reset();
    }

    void reset() override { _hpX = _hpY = _dcX = _dcY = _lp = 0.0f; }

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"fuzz.drive", "Drive", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setDrive(v); }, [this]() { return _drive.target(); }});
        out.push_back(Param{"fuzz.bias", "Bias", "", -1.0f, 1.0f, 0.01f,
            [this](float v) { setBias(v); }, [this]() { return _bias.target(); }});
        out.push_back(Param{"fuzz.tone", "Tone", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setTone(v); }, [this]() { return _tone.target(); }});
        out.push_back(Param{"fuzz.level", "Level", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setLevel(v); }, [this]() { return _level.target(); }});
    }

    void process(float* buf, int n) override {
        for (int i = 0; i < n; ++i) {
            const float x = buf[i];

            // capacitor de acoplamiento: fuera los graves antes de recortar
            _hpY = _hpCoef * (_hpY + x - _hpX);
            _hpX = x;

            const float drive = _drive.next();
            const float gain  = 2.0f + drive * drive * 200.0f;   // hasta ~200x
            float v = _hpY * gain + _bias.next();

            // recorte cubico: suave hasta +-1, y de ahi plano
            v = clipCubic(v);

            // bloqueo de DC -- obligatorio despues de un recorte asimetrico
            _dcY = v - _dcX + _dcCoef * _dcY;
            _dcX = v;
            float y = _dcY;

            // tono: pasa-bajos entre 700 Hz y 6 kHz
            const float fc     = 700.0f + _tone.next() * 5300.0f;
            const float lpCoef = 1.0f - std::exp(-2.0f * kPi * fc / static_cast<float>(_sr));
            _lp += lpCoef * (y - _lp);

            buf[i] = _lp * _level.next() * 0.6f;
        }
    }

private:
    static inline float clipCubic(float v) {
        if (v >  1.0f) return  1.0f;
        if (v < -1.0f) return -1.0f;
        return 1.5f * v - 0.5f * v * v * v;
    }
    static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
    static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

    Smoothed _drive, _bias, _tone, _level;
    double _sr = 48000.0;
    float _hpCoef = 0, _hpX = 0, _hpY = 0;
    float _dcCoef = 0, _dcX = 0, _dcY = 0;
    float _lp = 0;
};

// ---------------------------------------------------------------------------
// Tremolo.
//
// Modulacion de amplitud pura: un LFO subiendo y bajando el volumen. Lo unico
// interesante es la FORMA de onda. El tremolo optico de un Fender es casi
// senoidal, porque una bombilla no puede encenderse ni apagarse de golpe, y por
// eso late suave. Los tremolos de "bias" y los digitales pueden ser cuadrados,
// y eso pica en vez de latir.
//
// El parametro shape recorre de senoidal a cuadrada saturando la senoidal cada
// vez mas. Es el mismo truco que usa un recortador, aplicado al LFO.
// ---------------------------------------------------------------------------
class Tremolo : public Effect {
public:
    const char* name() const override { return "tremolo"; }

    void setRateHz(float hz) { _rate.store(clampf(hz, 0.1f, 20.0f), std::memory_order_relaxed); }
    void setDepth(float v)   { _depth.set(clamp01(v)); }
    void setShape(float v)   { _shape.set(clamp01(v)); }   // 0 = senoidal, 1 = cuadrada

    void prepare(double sampleRate, int) override {
        _sr = sampleRate;
        _depth.prepare(sampleRate, 30.0);
        _shape.prepare(sampleRate, 30.0);
        reset();
    }

    void reset() override { _phase = 0.0f; }

    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"tremolo.rate", "Rate", "Hz", 0.1f, 20.0f, 0.05f,
            [this](float v) { setRateHz(v); },
            [this]() { return _rate.load(std::memory_order_relaxed); }});
        out.push_back(Param{"tremolo.depth", "Depth", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setDepth(v); }, [this]() { return _depth.target(); }});
        out.push_back(Param{"tremolo.shape", "Shape", "", 0.0f, 1.0f, 0.01f,
            [this](float v) { setShape(v); }, [this]() { return _shape.target(); }});
    }

    void process(float* buf, int n) override {
        const float inc = _rate.load(std::memory_order_relaxed) / static_cast<float>(_sr);
        for (int i = 0; i < n; ++i) {
            float lfo = std::sin(2.0f * kPi * _phase);

            const float shape = _shape.next();
            if (shape > 0.001f) {
                const float k = 1.0f + shape * 24.0f;
                lfo = std::tanh(lfo * k) / std::tanh(k);
            }

            _phase += inc;
            if (_phase >= 1.0f) _phase -= 1.0f;

            // lfo=+1 -> volumen completo; lfo=-1 -> atenuado segun depth
            const float g = 1.0f - _depth.next() * 0.5f * (1.0f - lfo);
            buf[i] *= g;
        }
    }

private:
    static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
    static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
    std::atomic<float> _rate{4.0f};
    Smoothed _depth, _shape;
    double _sr = 48000.0;
    float _phase = 0.0f;
};

// ---------------------------------------------------------------------------
// Ganancia simple. Sirve para normalizar la salida del modelo con GetLoudness()
// para que cambiar de capture no te cambie el volumen.
// ---------------------------------------------------------------------------
class Gain : public Effect {
public:
    const char* name() const override { return "gain"; }
    void setGainDb(float db) { _db = db; _gain.set(dbToGain(db)); }
    void collectParams(std::vector<Param>& out) override {
        out.push_back(Param{"gain.level", "Level", "dB", -24.0f, 24.0f, 0.1f,
            [this](float v) { setGainDb(v); }, [this]() { return _db; }});
    }
    void prepare(double sampleRate, int) override { _gain.prepare(sampleRate, 30.0); }
    void process(float* buf, int n) override {
        for (int i = 0; i < n; ++i) buf[i] *= _gain.next();
    }
private:
    Smoothed _gain;
    float _db = 0.0f;
};

} // namespace fx
