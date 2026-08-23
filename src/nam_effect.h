#pragma once
//
// El modelo NAM envuelto como un eslabon mas de la cadena, para que no sea un
// caso especial dentro del callback.
//
#include "effects.h"
#include "NAM/get_dsp.h"
#include <stdexcept>

namespace fx {

class NamModel : public Effect {
public:
    explicit NamModel(std::unique_ptr<nam::DSP> dsp) : _dsp(std::move(dsp)) {
        if (!_dsp) throw std::runtime_error("NamModel: modelo nulo");
    }

    const char* name() const override { return "nam"; }
    nam::DSP* dsp() { return _dsp.get(); }

    void prepare(double sampleRate, int maxBlock) override {
        _scratch.assign(static_cast<size_t>(maxBlock), 0.0f);
        // Reset() fija el sample rate, dimensiona los buffers internos y prewarmea.
        _dsp->Reset(sampleRate, maxBlock);
        _maxBlock = maxBlock;
    }

    void process(float* buf, int n) override {
        if (n > _maxBlock) return;          // nunca reservar aqui
        float* in[]  = { buf };
        float* out[] = { _scratch.data() };
        _dsp->process(in, out, n);
        std::memcpy(buf, _scratch.data(), static_cast<size_t>(n) * sizeof(float));
    }

private:
    std::unique_ptr<nam::DSP> _dsp;
    std::vector<float> _scratch;
    int _maxBlock = 0;
};

} // namespace fx
