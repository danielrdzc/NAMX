#include "RtAudio.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"
#include "effects.h"
#include "nam_effect.h"
#include "ir.h"
#include "tuner.h"
#include "presets.h"
#include "webui.h"
#include <csignal>
#if !defined(_WIN32)
  #include <unistd.h>
#endif
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <fstream>
#include <cctype>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include "json.hpp"

constexpr unsigned int SAMPLE_RATE   = 48000;
constexpr unsigned int BUFFER_FRAMES = 64;

// Open the device with its NATIVE channel count. The Scarlett (and most
// interfaces) are stereo in / stereo out; asking RtAudio for 1 channel makes it
// extract and re-insert a channel behind our back, and on ALSA that conversion
// layer is where the periodic artifacts came from. We take stereo, pull channel
// 0 out ourselves, and write the result to both outputs.
constexpr unsigned int CHANNELS = 2;

// Biggest block we are willing to handle without reallocating in the callback.
constexpr unsigned int MAX_FRAMES = 8192;

// Hard clamp on the output so samples over +/-1.0 can't wrap around in the
// driver's float->int conversion (wrap-around is what "bit crushed" sounds like).
constexpr bool CLAMP_OUTPUT = true;

// --bypass: skip the model entirely (straight memcpy in->out). If the sound is
// still fuzzy with this on, the problem is in the audio path, not in NAM.
bool g_bypass = false;

// --buffer N: override the block size at runtime, to test whether the artifacts
// depend on how many frames the model is asked to process per call.
unsigned int g_buffer_frames = BUFFER_FRAMES;

// --device N : force a device id (see --list). -1 = auto-detect.
// --list     : print the devices and exit.
int  g_device_id = -1;
bool g_list_only = false;

// --api alsa|jack|pulse : which backend RtAudio should use (Linux).
std::string g_api_name = "";

// --slim V : for SlimmableContainer models, pick the submodel. 0.0 = smallest
// and cheapest, 1.0 = full size. Negative means "leave the model's default".
double g_slim = -1.0;

// --selftest : no audio hardware at all. Runs the same signal through the model
// at several block sizes and compares the outputs. A model MUST produce the same
// result regardless of how the samples are chopped up; if it doesn't, that is
// the bug, and this proves it without any driver in the picture.
bool g_selftest = false;

// --periods N : how many periods of cushion the device buffer gets.
// The callback's own duration delays the write relative to the read, so with
// only a couple of periods the playback side starves periodically even when
// there is plenty of CPU headroom. 0 = leave RtAudio's default.
unsigned int g_periods = 0;

// --record FILE.wav : capture exactly what the callback receives and exactly
// what it writes, into a stereo float32 WAV (L = input, R = output). The
// callback only memcpys into a preallocated buffer; the file is written at exit.
std::string g_record_path;
std::vector<float>        g_rec;          // interleaved L=in, R=out
std::atomic<size_t>       g_rec_pos{0};   // in frames
size_t                    g_rec_capacity = 0;

// Efectos. Apagados por omision: se encienden pasando su flag.
bool  g_gate_on   = false;   float g_gate_db    = -45.0f;
bool  g_od_on     = false;   float g_od_drive   = 0.5f;
float g_od_tone   = 0.5f;    float g_od_level   = 0.5f;
bool  g_normalize = false;   float g_target_db  = -18.0f;
bool  g_ph_on     = false;   float g_ph_rate    = 0.5f;
float g_ph_depth  = 0.7f;    float g_ph_fb      = 0.3f;
float g_ph_mix    = 0.5f;    int   g_ph_stages  = 4;
bool  g_dly_on    = false;   float g_dly_ms     = 400.0f;
float g_dly_fb    = 0.35f;   float g_dly_mix    = 0.3f;   float g_dly_tone = 0.5f;
bool  g_rev_on    = false;   float g_rev_mix    = 0.25f;
float g_rev_room  = 0.6f;    float g_rev_damp   = 0.5f;
bool  g_web_on    = true;    int   g_web_port   = 8080;
bool  g_fz_on     = false;   float g_fz_drive   = 0.6f;
float g_fz_bias   = 0.0f;    float g_fz_tone    = 0.5f;  float g_fz_level = 0.5f;
bool  g_tr_on     = false;   float g_tr_rate    = 4.0f;
float g_tr_depth  = 0.6f;    float g_tr_shape   = 0.0f;
bool  g_cp_on     = false;   float g_cp_thr     = -18.0f;  float g_cp_ratio = 4.0f;
bool  g_eq_on     = false;

// --preset NOMBRE: carga un preset al arrancar. Para el servicio de systemd es
// lo unico que hace falta: el preset trae el modelo adentro.
std::string g_preset;
std::string g_ir_path;              // --ir RUTA.wav

// El afinador vive fuera de la cadena: no modifica el audio, solo lo observa.
fx::Tuner g_tuner;

// Cuando corre como servicio no hay stdin ni terminal. En ese caso no se puede
// esperar un ENTER: hay que esperar a que systemd mande SIGTERM.
std::atomic<bool> g_stop{false};
extern "C" void onSignal(int) { g_stop.store(true); }


// How long model->process() takes, against the deadline the block gives us.
// This is the number that actually decides whether the Pi can run a model:
// RtAudio's xrun flag on ALSA stays 0 even while the card is starving.
std::atomic<long long>    g_proc_ns_total{0};
std::atomic<long long>    g_proc_count{0};
std::atomic<long long>    g_proc_ns_max{0};
std::atomic<long long>    g_deadline_misses{0};
double g_deadline_ns = 0.0;

std::atomic<int>       g_xrun_count{0};
std::atomic<long long> g_clip_in{0};
std::atomic<long long> g_clip_out{0};
std::atomic<float>     g_peak_in{0.0f};
std::atomic<float>     g_peak_out{0.0f};
// Copias aparte para la web. Si compartieran las de arriba, el medidor de la
// terminal y el del navegador se robarian las lecturas (ambos hacen exchange).
std::atomic<float>     g_web_in{0.0f};
std::atomic<float>     g_web_out{0.0f};
std::atomic<bool>      g_running{true};

// Largest nFrames the callback has actually been handed. NAM sizes its internal
// buffers from the value passed to Reset(); if the driver ever delivers a bigger
// block than that, the core overruns them silently in a Release build (the check
// inside WaveNet::process is an assert, which NDEBUG compiles out).
std::atomic<unsigned int> g_max_nframes{0};
unsigned int g_reset_frames = 0;

inline void atomicMax(std::atomic<float>& target, float value) {
    float prev = target.load(std::memory_order_relaxed);
    while (prev < value &&
           !target.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {
    }
}

struct CallbackData {
    fx::Chain* chain;
    // Scratch buffers, allocated before the stream starts. Never allocate here.
    std::vector<float> mono_in;
    std::vector<float> mono_out;
};

// Counts blocks we had to drop because they were bigger than MAX_FRAMES.
std::atomic<int> g_oversize_blocks{0};

int audioCallback(void* outputBuffer, void* inputBuffer,
                   unsigned int nFrames, double,
                   RtAudioStreamStatus status, void* userData) {
    if (status) {
        g_xrun_count++;
    }

    unsigned int prevMax = g_max_nframes.load(std::memory_order_relaxed);
    while (prevMax < nFrames &&
           !g_max_nframes.compare_exchange_weak(prevMax, nFrames, std::memory_order_relaxed)) {
    }

    float* in  = static_cast<float*>(inputBuffer);
    float* out = static_cast<float*>(outputBuffer);

    if (!out) return 0;
    if (!in || nFrames > MAX_FRAMES) {
        if (nFrames > MAX_FRAMES) g_oversize_blocks.fetch_add(1, std::memory_order_relaxed);
        std::memset(out, 0, nFrames * CHANNELS * sizeof(float));
        return 0;
    }

    CallbackData* data = static_cast<CallbackData*>(userData);
    float* mi = data->mono_in.data();
    float* mo = data->mono_out.data();

    // ---- de-interleave channel 0, and measure it before the model sees it ---
    float     peakIn = 0.0f;
    long long clipIn = 0;
    for (unsigned int i = 0; i < nFrames; ++i) {
        const float x = in[i * CHANNELS];        // channel 0 = input 1 = guitar
        mi[i] = x;
        const float a = std::fabs(x);
        if (a > peakIn) peakIn = a;
        if (a >= 0.999f) ++clipIn;               // already clipped by the interface
    }

    // El afinador observa la senal seca: afinar con distorsion encima es
    // adivinar. Aqui solo se copia a un buffer circular; el analisis pasa en
    // otro hilo.
    g_tuner.push(mi, static_cast<int>(nFrames));

    // Denormales a cero. Es por hilo, y este es el hilo de audio.
    static thread_local bool ftzDone = false;
    if (!ftzDone) { fx::enableFlushToZero(); ftzDone = true; }

    const auto t0 = std::chrono::steady_clock::now();
    std::memcpy(mo, mi, nFrames * sizeof(float));
    if (!g_bypass) {
        data->chain->process(mo, static_cast<int>(nFrames));
    }
    const long long elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();

    g_proc_ns_total.fetch_add(elapsed_ns, std::memory_order_relaxed);
    g_proc_count.fetch_add(1, std::memory_order_relaxed);
    long long prevMaxNs = g_proc_ns_max.load(std::memory_order_relaxed);
    while (prevMaxNs < elapsed_ns &&
           !g_proc_ns_max.compare_exchange_weak(prevMaxNs, elapsed_ns,
                                                std::memory_order_relaxed)) {
    }
    if (g_deadline_ns > 0.0 && static_cast<double>(elapsed_ns) > g_deadline_ns) {
        g_deadline_misses.fetch_add(1, std::memory_order_relaxed);
    }

    // ---- measure, clamp, and write the same signal to both outputs ----------
    float     peakOut = 0.0f;
    long long clipOut = 0;
    for (unsigned int i = 0; i < nFrames; ++i) {
        float y = mo[i];
        const float a = std::fabs(y);
        if (a > peakOut) peakOut = a;
        if (a > 1.0f) {
            ++clipOut;
            if (CLAMP_OUTPUT) y = (y > 0.0f) ? 1.0f : -1.0f;
        }
        out[i * CHANNELS]     = y;
        out[i * CHANNELS + 1] = y;
    }

    // Capture in/out for offline analysis. Preallocated; no I/O, no locks.
    if (g_rec_capacity > 0) {
        const size_t pos = g_rec_pos.load(std::memory_order_relaxed);
        if (pos + nFrames <= g_rec_capacity) {
            float* dst = g_rec.data() + pos * 2;
            for (unsigned int i = 0; i < nFrames; ++i) {
                dst[i * 2]     = mi[i];
                dst[i * 2 + 1] = mo[i];
            }
            g_rec_pos.store(pos + nFrames, std::memory_order_relaxed);
        }
    }

    atomicMax(g_peak_in, peakIn);
    atomicMax(g_peak_out, peakOut);
    atomicMax(g_web_in, peakIn);
    atomicMax(g_web_out, peakOut);
    if (clipIn)  g_clip_in.fetch_add(clipIn, std::memory_order_relaxed);
    if (clipOut) g_clip_out.fetch_add(clipOut, std::memory_order_relaxed);

    return 0;
}

// Case-insensitive substring match, for finding the interface by name on any OS.
static bool containsNoCase(const std::string& haystack, const std::string& needle) {
    if (needle.size() > haystack.size()) return false;
    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        size_t j = 0;
        for (; j < needle.size(); ++j) {
            if (std::tolower(static_cast<unsigned char>(haystack[i + j]))
                != std::tolower(static_cast<unsigned char>(needle[j]))) break;
        }
        if (j == needle.size()) return true;
    }
    return false;
}

static std::string dbfs(float peak) {
    if (peak <= 0.0f) return "  -inf";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%6.1f", 20.0 * std::log10(static_cast<double>(peak)));
    return std::string(buf);
}



// Minimal 32-bit-float stereo WAV writer.
static bool writeWavF32(const std::string& path, const float* data,
                        size_t frames, unsigned int channels, unsigned int rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t dataBytes = static_cast<uint32_t>(frames * channels * sizeof(float));
    const uint32_t byteRate  = rate * channels * sizeof(float);
    const uint16_t blockAlign = static_cast<uint16_t>(channels * sizeof(float));
    auto u32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&f](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("RIFF", 4);  u32(36 + dataBytes);  f.write("WAVE", 4);
    f.write("fmt ", 4);  u32(16);  u16(3);              // 3 = IEEE float
    u16(static_cast<uint16_t>(channels));  u32(rate);  u32(byteRate);
    u16(blockAlign);  u16(32);
    f.write("data", 4);  u32(dataBytes);
    f.write(reinterpret_cast<const char*>(data), dataBytes);
    return f.good();
}

// ---------------------------------------------------------------------------
// Offline block-size consistency check.
// ---------------------------------------------------------------------------
static int runSelfTest(const std::filesystem::path& modelPath, double slim) {
    // Length must divide evenly by every block size, or the bigger blocks leave
    // an unprocessed tail and the comparison measures the tail, not the model.
    // 98304 = 512 * 192, and divides by 256/128/64/32 too.
    constexpr int kNumFrames = 98304;
    const std::vector<int> blockSizes = {512, 256, 128, 64, 32};

    // A steady tone: any periodic artifact stands out against it, and it makes
    // the comparison between block sizes meaningful.
    std::vector<float> signal(kNumFrames);
    for (int i = 0; i < kNumFrames; ++i) {
        const double t = static_cast<double>(i) / 48000.0;
        signal[i] = 0.25f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 220.0 * t));
    }

    std::cout << "Offline self-test: 220 Hz tone, "
              << (static_cast<double>(kNumFrames) / 48000.0) << " s, "
              << kNumFrames << " samples.\n";
    std::cout << "Processing the SAME signal at different block sizes.\n\n";

    std::vector<std::vector<float>> results;
    std::vector<int> processedLen;
    for (int bs : blockSizes) {
        std::unique_ptr<nam::DSP> m;
        try {
            m = nam::get_dsp(modelPath);
        } catch (std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << "\n";
            return 1;
        }
        m->Reset(48000.0, bs);
        if (slim >= 0.0) {
            if (auto* sl = dynamic_cast<nam::SlimmableModel*>(m.get())) sl->SetSlimmableSize(slim);
        }

        std::vector<float> out(kNumFrames, 0.0f);
        std::vector<float> inBlock(bs), outBlock(bs);
        int processed = 0;
        for (int pos = 0; pos + bs <= kNumFrames; pos += bs) {
            std::memcpy(inBlock.data(), signal.data() + pos, bs * sizeof(float));
            float* ins[]  = { inBlock.data() };
            float* outs[] = { outBlock.data() };
            m->process(ins, outs, bs);
            std::memcpy(out.data() + pos, outBlock.data(), bs * sizeof(float));
            processed = pos + bs;
        }
        results.push_back(std::move(out));
        processedLen.push_back(processed);
        std::cout << "  block " << std::setw(4) << bs << " done ("
                  << processed << " of " << kNumFrames << " samples processed)\n";
    }

    // Compare everything against the largest block size. Skip the first 16k
    // samples so start-up transients don't pollute the comparison.
    const int skip = 16384;
    int common = kNumFrames;
    for (int p : processedLen) common = std::min(common, p);
    const std::vector<float>& ref = results[0];
    std::cout << "\n  comparing samples " << skip << " .. " << common << "\n";
    std::cout << "\n  vs block " << blockSizes[0] << ":\n";
    std::cout << std::scientific << std::setprecision(3);
    bool mismatch = false;
    for (size_t k = 1; k < results.size(); ++k) {
        double maxDiff = 0.0, sumSq = 0.0;
        long long n = 0;
        for (int i = skip; i < common; ++i) {
            const double d = std::fabs(static_cast<double>(results[k][i]) - ref[i]);
            if (d > maxDiff) maxDiff = d;
            sumSq += d * d;
            ++n;
        }
        const double rms = (n > 0) ? std::sqrt(sumSq / n) : 0.0;
        std::cout << "  block " << std::setw(4) << blockSizes[k]
                  << " : max diff " << maxDiff << "   rms diff " << rms;
        if (maxDiff > 1e-4) { std::cout << "   <-- DIFFERENT"; mismatch = true; }
        std::cout << "\n";
    }

    std::cout << std::defaultfloat << "\n";
    if (mismatch) {
        std::cout << "The model's output DEPENDS on the block size. That is a bug in the\n"
                     "inference, not in the audio path -- no driver was involved here.\n";
    } else {
        std::cout << "The model is block-size independent. The artifact lives in the\n"
                     "audio path, not in the inference.\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    // Args are parsed before anything else so --list can bail out early.
    std::filesystem::path modelPath = "test_model.nam";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bypass" || arg == "-b") {
            g_bypass = true;
        } else if (arg == "--list") {
            g_list_only = true;
        } else if ((arg == "--buffer" || arg == "-n") && i + 1 < argc) {
            g_buffer_frames = static_cast<unsigned int>(std::atoi(argv[++i]));
            if (g_buffer_frames == 0) g_buffer_frames = BUFFER_FRAMES;
        } else if ((arg == "--device" || arg == "-d") && i + 1 < argc) {
            g_device_id = std::atoi(argv[++i]);
        } else if (arg == "--api" && i + 1 < argc) {
            g_api_name = argv[++i];
        } else if (arg == "--slim" && i + 1 < argc) {
            g_slim = std::atof(argv[++i]);
        } else if (arg == "--selftest") {
            g_selftest = true;
        } else if ((arg == "--periods" || arg == "-p") && i + 1 < argc) {
            g_periods = static_cast<unsigned int>(std::atoi(argv[++i]));
        } else if (arg == "--record" && i + 1 < argc) {
            g_record_path = argv[++i];
        } else if (arg == "--gate" && i + 1 < argc) {
            g_gate_on = true;  g_gate_db = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--od" && i + 1 < argc) {
            g_od_on = true;    g_od_drive = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--od-tone" && i + 1 < argc) {
            g_od_tone = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--od-level" && i + 1 < argc) {
            g_od_level = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--normalize") {
            g_normalize = true;
        } else if (arg == "--phaser" && i + 1 < argc) {
            g_ph_on = true;    g_ph_rate = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--phaser-depth" && i + 1 < argc) {
            g_ph_depth = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--phaser-fb" && i + 1 < argc) {
            g_ph_fb = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--phaser-mix" && i + 1 < argc) {
            g_ph_mix = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--phaser-stages" && i + 1 < argc) {
            g_ph_stages = std::atoi(argv[++i]);
        } else if (arg == "--delay" && i + 1 < argc) {
            g_dly_on = true;   g_dly_ms = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--delay-fb" && i + 1 < argc) {
            g_dly_fb = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--delay-mix" && i + 1 < argc) {
            g_dly_mix = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--delay-tone" && i + 1 < argc) {
            g_dly_tone = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--reverb" && i + 1 < argc) {
            g_rev_on = true;   g_rev_mix = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--reverb-room" && i + 1 < argc) {
            g_rev_room = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--reverb-damp" && i + 1 < argc) {
            g_rev_damp = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--port" && i + 1 < argc) {
            g_web_port = std::atoi(argv[++i]);
        } else if (arg == "--no-web") {
            g_web_on = false;
        } else if (arg == "--preset" && i + 1 < argc) {
            g_preset = argv[++i];
        } else if (arg == "--ir" && i + 1 < argc) {
            g_ir_path = argv[++i];
        } else if (arg == "--comp" && i + 1 < argc) {
            g_cp_on = true;    g_cp_thr = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--comp-ratio" && i + 1 < argc) {
            g_cp_ratio = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--eq") {
            g_eq_on = true;
        } else if (arg == "--fuzz" && i + 1 < argc) {
            g_fz_on = true;    g_fz_drive = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--fuzz-bias" && i + 1 < argc) {
            g_fz_bias = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--fuzz-tone" && i + 1 < argc) {
            g_fz_tone = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--fuzz-level" && i + 1 < argc) {
            g_fz_level = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--tremolo" && i + 1 < argc) {
            g_tr_on = true;    g_tr_rate = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--tremolo-depth" && i + 1 < argc) {
            g_tr_depth = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--tremolo-shape" && i + 1 < argc) {
            g_tr_shape = static_cast<float>(std::atof(argv[++i]));
        } else {
            modelPath = arg;
        }
    }

    if (g_selftest) {
        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "Model file not found: " << modelPath << "\n";
            return 1;
        }
        return runSelfTest(modelPath, g_slim);
    }

#ifdef _WIN32
    RtAudio* audioPtr = new RtAudio(RtAudio::WINDOWS_ASIO);
    if (audioPtr->getDeviceCount() < 1) {
        std::cerr << "No ASIO devices found, falling back to default API.\n";
        delete audioPtr;
        audioPtr = new RtAudio();
    }
    RtAudio& audio = *audioPtr;
#else
    // ALSA talks to the hardware directly. JACK is worth trying when ALSA's
    // duplex handling misbehaves -- it was built to keep capture and playback
    // locked together.
    RtAudio::Api api = RtAudio::LINUX_ALSA;
    if      (g_api_name == "jack")  api = RtAudio::UNIX_JACK;
    else if (g_api_name == "pulse") api = RtAudio::LINUX_PULSE;
    else if (g_api_name == "alsa")  api = RtAudio::LINUX_ALSA;
    else if (!g_api_name.empty()) {
        std::cerr << "Unknown --api '" << g_api_name << "'. Use alsa, jack or pulse.\n";
        return 1;
    }

    RtAudio* audioPtr = new RtAudio(api);
    if (audioPtr->getDeviceCount() < 1) {
        // Never fall back silently when the API was asked for explicitly: you end
        // up measuring the backend you were trying to avoid and not noticing.
        if (!g_api_name.empty()) {
            std::cerr << "\nThe '" << g_api_name << "' API has no devices.\n";
            if (g_api_name == "jack")
                std::cerr << "The JACK server is not running. Start it first:\n"
                             "  jackd -R -P 90 -d alsa -d hw:2 -r 48000 -p 64 -n 3\n"
                             "and check it with jack_lsp before running this again.\n";
            return 1;
        }
        std::cerr << "No devices on the default API, trying the fallback.\n";
        delete audioPtr;
        audioPtr = new RtAudio();
    }
    RtAudio& audio = *audioPtr;
#endif

    if (audio.getDeviceCount() < 1) {
        std::cerr << "No audio devices found.\n";
        return 1;
    }

    std::cout << "Using audio API: " << RtAudio::getApiDisplayName(audio.getCurrentApi()) << "\n";

    std::cout << "Available audio devices:\n";
    std::vector<unsigned int> ids = audio.getDeviceIds();
    unsigned int interfaceId = 0;
    bool foundInterface = false;
    for (unsigned int id : ids) {
        RtAudio::DeviceInfo info = audio.getDeviceInfo(id);
        if (info.name.empty()) continue;
        std::cout << "  [id=" << id << "] " << info.name
                  << " (in:" << info.inputChannels
                  << " out:" << info.outputChannels << ")"
                  << (info.isDefaultInput ? "  <- default input" : "")
                  << (info.isDefaultOutput ? "  <- default output" : "")
                  << "\n";

        // Full duplex only; on Linux the same card shows up under several names.
        if (info.inputChannels == 0 || info.outputChannels == 0) continue;

        const bool isOurs = containsNoCase(info.name, "Focusrite")
                         || containsNoCase(info.name, "Scarlett");
        // Prefer the ASIO entry on Windows over anything matched earlier.
        const bool isAsio = containsNoCase(info.name, "Focusrite USB ASIO");
        if (isOurs && (!foundInterface || isAsio)) {
            interfaceId = id;
            foundInterface = true;
        }
    }

    if (g_device_id >= 0) {
        interfaceId = static_cast<unsigned int>(g_device_id);
        foundInterface = true;
        std::cout << "Using device id " << interfaceId << " (forced with --device)\n";
    } else if (foundInterface) {
        std::cout << "Auto-selected device id " << interfaceId << "\n";
    } else {
        std::cerr << "No Focusrite/Scarlett found. Falling back to the system default.\n"
                     "Run with --list to see the ids, then pass --device N.\n";
    }

    if (g_list_only) {
        return 0;
    }

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "Model file not found: " << modelPath << "\n";
        std::cerr << "Pass a path as the first argument, or put a file named "
                     "'test_model.nam' in the folder you run this exe from.\n";
        return 1;
    }

    std::cout << "Model loaded: " << std::filesystem::absolute(modelPath) << "\n";

    // Print what the capture calls itself, so there's never any doubt about
    // which file is actually running.
    try {
        std::ifstream f(modelPath);
        nlohmann::json j;
        f >> j;
        if (j.contains("metadata")) {
            const auto& m = j["metadata"];
            auto field = [&m](const char* key) -> std::string {
                return (m.contains(key) && m[key].is_string()) ? m[key].get<std::string>() : std::string("?");
            };
            std::cout << "Capture name: " << field("name") << "\n";
            const std::string gear = field("gear_type");
            std::cout << "Gear type:    " << gear;
            if (gear == "amp" || gear == "preamp")
                std::cout << "   (no cab baked in; needs an IR after it)";
            else if (gear == "amp_cab")
                std::cout << "   (cab included; do NOT add an IR)";
            std::cout << "\n";
        }
    } catch (...) {
        // Metadata is a nicety; never let it stop the program.
    }
    if (!g_record_path.empty()) {
        g_rec_capacity = 60 * SAMPLE_RATE;              // 60 seconds is plenty
        g_rec.assign(g_rec_capacity * 2, 0.0f);
        std::cout << "Recording to " << g_record_path << " (up to 60 s, L=input R=output)\n";
    }

    // ---- cadena de efectos -------------------------------------------------
    // Orden de rig real: gate -> overdrive -> AMPLI -> nivel.
    // El overdrive va ANTES del modelo, igual que un pedal delante de un ampli.
    // La cadena se arma SIEMPRE completa; los flags solo fijan valores iniciales
    // y que efectos arrancan encendidos. Asi la interfaz web puede prender y
    // apagar bloques en vivo sin reconstruir nada en el hilo de audio.
    fx::Chain chain;

    auto gate = std::make_unique<fx::NoiseGate>();
    gate->setThresholdDb(g_gate_db);
    gate->setEnabled(g_gate_on);
    chain.add(std::move(gate));

    auto fz = std::make_unique<fx::Fuzz>();
    fz->setDrive(g_fz_drive);
    fz->setBias(g_fz_bias);
    fz->setTone(g_fz_tone);
    fz->setLevel(g_fz_level);
    fz->setEnabled(g_fz_on);
    chain.add(std::move(fz));

    auto od = std::make_unique<fx::Overdrive>();
    od->setDrive(g_od_drive);
    od->setTone(g_od_tone);
    od->setLevel(g_od_level);
    od->setEnabled(g_od_on);
    chain.add(std::move(od));

    auto comp = std::make_unique<fx::Compressor>();
    comp->setEnabled(g_cp_on);
    comp->setThresholdDb(g_cp_thr);
    comp->setRatio(g_cp_ratio);
    chain.add(std::move(comp));

    auto ph = std::make_unique<fx::Phaser>();
    ph->setRateHz(g_ph_rate);
    ph->setDepth(g_ph_depth);
    ph->setFeedback(g_ph_fb);
    ph->setMix(g_ph_mix);
    ph->setStages(g_ph_stages);
    ph->setEnabled(g_ph_on);
    chain.add(std::move(ph));

    auto namFx = std::make_unique<fx::NamModel>();
    fx::NamModel* namPtr = namFx.get();
    chain.add(std::move(namFx));

    // El gabinete va inmediatamente despues del ampli: es el orden fisico de un
    // rig real, y el unico que suena bien.
    auto irFx = std::make_unique<fx::IRLoader>();
    fx::IRLoader* irPtr = irFx.get();
    chain.add(std::move(irFx));

    // El EQ va despues del gabinete: es el equivalente a ecualizar el microfono,
    // que es donde se hace en un estudio.
    auto eq = std::make_unique<fx::EQ>();
    eq->setEnabled(g_eq_on);
    chain.add(std::move(eq));

    {
        auto gain = std::make_unique<fx::Gain>();
        gain->setGainDb(0.0f);
        gain->setEnabled(g_normalize);
        chain.add(std::move(gain));
    }

    // El tremolo va DESPUES del ampli: en un Fender el circuito esta entre el
    // previo y la etapa de potencia, o sea que modula la senal ya distorsionada.
    // Antes del ampli el resultado es otro -- la compresion del ampli aplana la
    // modulacion y el latido casi desaparece.
    auto tr = std::make_unique<fx::Tremolo>();
    tr->setRateHz(g_tr_rate);
    tr->setDepth(g_tr_depth);
    tr->setShape(g_tr_shape);
    tr->setEnabled(g_tr_on);
    chain.add(std::move(tr));

    // Delay y reverb van DESPUES del ampli, como en un loop de efectos.
    // Antes del ampli el delay se distorsiona y se vuelve papilla.
    auto dly = std::make_unique<fx::Delay>();
    dly->setTimeMs(g_dly_ms);
    dly->setFeedback(g_dly_fb);
    dly->setMix(g_dly_mix);
    dly->setTone(g_dly_tone);
    dly->setEnabled(g_dly_on);
    chain.add(std::move(dly));

    auto rev = std::make_unique<fx::Reverb>();
    rev->setMix(g_rev_mix);
    rev->setRoomSize(g_rev_room);
    rev->setDamp(g_rev_damp);
    rev->setEnabled(g_rev_on);
    chain.add(std::move(rev));

    // Los presets viven junto al proyecto, y los modelos en models/.
    const std::string modelsDir = "models";
    const std::string irDir     = "irs";
    fx::Presets presets(&chain, namPtr, irPtr, "presets");

    CallbackData callbackData;
    callbackData.chain = &chain;
    callbackData.mono_in.assign(MAX_FRAMES, 0.0f);
    callbackData.mono_out.assign(MAX_FRAMES, 0.0f);

    RtAudio::StreamParameters inParams, outParams;
    inParams.deviceId   = foundInterface ? interfaceId : audio.getDefaultInputDevice();
    inParams.nChannels  = CHANNELS;
    outParams.deviceId  = foundInterface ? interfaceId : audio.getDefaultOutputDevice();
    outParams.nChannels = CHANNELS;

    unsigned int bufferFrames = g_buffer_frames;

    RtAudio::StreamOptions options;
    options.flags = RTAUDIO_SCHEDULE_REALTIME;
    options.priority = 90;
    if (g_periods > 0) options.numberOfBuffers = g_periods;

    try {
        audio.openStream(&outParams, &inParams, RTAUDIO_FLOAT32,
                         SAMPLE_RATE, &bufferFrames, &audioCallback,
                         &callbackData, &options);

        // Reset() sets the sample rate + sizes the internal buffers and prewarms.
        // It must happen after openStream (bufferFrames may come back different
        // from what we asked for) and before startStream.
        g_reset_frames = bufferFrames;
        g_deadline_ns = 1e9 * static_cast<double>(bufferFrames) / SAMPLE_RATE;

        // prepare() reserva toda la memoria de la cadena y hace el Reset del
        // modelo. Despues de esto, nada en el hilo de audio reserva nada.
        chain.prepare(static_cast<double>(SAMPLE_RATE), static_cast<int>(bufferFrames));
        g_tuner.prepare(static_cast<double>(SAMPLE_RATE));

        // El modelo se carga DESPUES de prepare(): asi NamModel ya conoce el
        // sample rate y el tamano de bloque, y puede dejarlo listo antes de
        // publicarlo. Es el mismo camino que usa el cambio en caliente.
        // Un preset trae su propio modelo, asi que se intenta primero. El
        // argumento posicional queda como respaldo.
        bool ready = false;
        if (!g_preset.empty()) {
            const std::string perr = presets.load(g_preset);
            if (perr.empty() && namPtr->dsp()) {
                std::cout << "Preset cargado: " << g_preset << "\n";
                ready = true;
            } else {
                std::cerr << "No se pudo cargar el preset '" << g_preset
                          << "': " << perr << "\n";
            }
        }
        if (!ready) {
            const std::string err = namPtr->loadModel(modelPath);
            if (!err.empty()) {
                std::cerr << "Error cargando el modelo: " << err << "\n";
                return 1;
            }
        }
        if (!g_ir_path.empty()) {
            const std::string ierr = irPtr->loadIR(g_ir_path);
            if (!ierr.empty()) std::cerr << "IR: " << ierr << "\n";
        }

        std::cout << "Model loaded: " << namPtr->path() << "\n";
        std::cout << "Model loudness: " << namPtr->loudness() << " dB\n";
        if (!irPtr->path().empty()) std::cout << "IR loaded:    " << irPtr->path() << "\n";

        if (auto* slim = dynamic_cast<nam::SlimmableModel*>(namPtr->dsp())) {
            const auto breakpoints = slim->GetSlimmableSizeBreakpoints();
            std::cout << "Slimmable model. Breakpoints:";
            if (breakpoints.empty()) std::cout << " (none)";
            for (double b : breakpoints) std::cout << " " << b;
            std::cout << "   -- select with --slim 0.0 .. 1.0\n";
            if (g_slim >= 0.0) {
                slim->SetSlimmableSize(g_slim);
                std::cout << "Slimmable size set to " << g_slim << "\n";
            }
        } else if (g_slim >= 0.0) {
            std::cout << "This model is not slimmable; --slim ignored.\n";
        }
        chain.reset();

        audio.startStream();
    } catch (std::exception& e) {
        std::cerr << "Error opening stream: " << e.what() << "\n";
        return 1;
    }

    double latencyMs = 2.0 * bufferFrames / SAMPLE_RATE * 1000.0;
    std::cout << "\nStream running.\n";
    std::cout << "Requested buffer: " << g_buffer_frames << " frames\n";
    std::cout << "Actual buffer:    " << bufferFrames << " frames";
    if (bufferFrames != g_buffer_frames) {
        std::cout << "   <-- the ASIO driver overrode the request;\n"
                     "                            change it in the Focusrite control panel";
    }
    std::cout << "\n";
    std::cout << "Approx algorithmic latency (round trip): "
              << latencyMs << " ms\n";
    std::cout << "(actual latency also includes driver + hardware,"
                 " typically +1-3ms)\n";
    std::cout << "Periods:      "
              << (options.numberOfBuffers ? std::to_string(options.numberOfBuffers)
                                          : std::string("driver default"))
              << "   (cushion = periods x buffer frames)\n";
    std::cout << "Chain:        ";
    if (g_bypass) std::cout << "(bypass)";
    else for (size_t i = 0; i < chain.size(); ++i) {
        std::cout << (i ? " -> " : "") << chain.at(i)->name();
        if (!chain.at(i)->enabled()) std::cout << "(off)";
    }
    std::cout << "\n";
    std::cout << "Channels:     " << CHANNELS << " in / " << CHANNELS
              << " out (device native; guitar read from channel 1)\n";
    std::cout << "Output clamp: " << (CLAMP_OUTPUT ? "ON" : "OFF") << "\n";
    std::cout << "Model:        " << (g_bypass ? "BYPASSED (straight passthrough)" : "active") << "\n";
    // Interfaz web. Solo escribe atomics de parametros; nunca toca el audio.
    fx::WebUI web;
    if (g_web_on) {
        auto metersFn = []() {
            const long long n   = g_proc_count.load();
            const long long tot = g_proc_ns_total.load();
            const double load = (n > 0 && g_deadline_ns > 0.0)
                              ? 100.0 * (static_cast<double>(tot) / n) / g_deadline_ns : 0.0;
            char b[256];
            std::snprintf(b, sizeof(b),
                "{\"in\":%.1f,\"out\":%.1f,\"load\":%.1f,\"late\":%lld,\"xruns\":%d}",
                20.0 * std::log10(std::max(1e-6f, g_web_in.exchange(0.0f))),
                20.0 * std::log10(std::max(1e-6f, g_web_out.exchange(0.0f))),
                load, g_deadline_misses.load(), g_xrun_count.load());
            return std::string(b);
        };

        if (web.start(g_web_port, &chain, &presets, namPtr, irPtr, &g_tuner,
                      modelsDir, irDir, metersFn))
            std::cout << "Web UI:       http://nampedal.local:" << g_web_port
                      << "   (" << chain.collectParams().size()
                      << " parametros, cadena reordenable, presets)\n";
        else
            std::cerr << "Web UI:       no se pudo abrir el puerto " << g_web_port << "\n";
    }

    // Con terminal: medidores en vivo y ENTER para salir.
    // Como servicio: sin medidores (llenarian el journal) y se espera SIGTERM.
#if defined(_WIN32)
    const bool interactive = true;
#else
    const bool interactive = ::isatty(fileno(stdin)) != 0;
#endif

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    std::thread meter;
    if (interactive) {
        std::cout << "\nLive meters (peak over each 500ms window):\n";
        std::cout << "Press ENTER to stop...\n\n";
        meter = std::thread([]() {
            while (g_running.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                const float pin  = g_peak_in.exchange(0.0f, std::memory_order_relaxed);
                const float pout = g_peak_out.exchange(0.0f, std::memory_order_relaxed);
                const long long n   = g_proc_count.load();
                const long long tot = g_proc_ns_total.load();
                const double avgPct = (n > 0 && g_deadline_ns > 0.0)
                                    ? 100.0 * (static_cast<double>(tot) / n) / g_deadline_ns : 0.0;
                const double maxPct = (g_deadline_ns > 0.0)
                                    ? 100.0 * g_proc_ns_max.load() / g_deadline_ns : 0.0;
                std::cout << "\r  in " << dbfs(pin) << " dBFS   out " << dbfs(pout)
                          << " dBFS   load avg/max: " << std::fixed << std::setprecision(0)
                          << avgPct << "%/" << maxPct << "%"
                          << "   late: " << g_deadline_misses.load()
                          << "   clip: " << g_clip_in.load() << "/" << g_clip_out.load()
                          << "   xruns: " << g_xrun_count.load()
                          << "        " << std::flush;
            }
        });
        std::cin.get();
    } else {
        std::cout << "\nCorriendo como servicio. Esperando SIGTERM." << std::endl;
        while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    g_running.store(false, std::memory_order_relaxed);
    if (meter.joinable()) meter.join();

    std::cout << "\n\n--- session summary ---\n";
    std::cout << "Xruns:            " << g_xrun_count.load() << "\n";
    const long long n = g_proc_count.load();
    if (n > 0) {
        const double avgUs = (static_cast<double>(g_proc_ns_total.load()) / n) / 1000.0;
        const double maxUs = g_proc_ns_max.load() / 1000.0;
        const double budgetUs = g_deadline_ns / 1000.0;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Block deadline:   " << budgetUs << " us\n";
        std::cout << "Chain time:       avg " << avgUs << " us ("
                  << (100.0 * avgUs / budgetUs) << "% of deadline), max "
                  << maxUs << " us (" << (100.0 * maxUs / budgetUs) << "%)\n";
        std::cout << "Blocks over deadline: " << g_deadline_misses.load()
                  << " of " << n << "\n";
        if (g_deadline_misses.load() > 0) {
            std::cout << "-> The model does not fit in this block size on this machine.\n"
                         "   Raise the buffer, or use --slim to pick a cheaper submodel.\n";
        }
    }
    std::cout << "Oversize blocks dropped: " << g_oversize_blocks.load() << "\n";
    std::cout << "Largest block seen: " << g_max_nframes.load()
              << "  (model was Reset for " << g_reset_frames << ")\n";
    if (g_max_nframes.load() > g_reset_frames) {
        std::cout << "-> The driver handed us a BIGGER block than the model was sized for.\n"
                     "   NAM overran its internal buffers. This is the bug.\n";
    }
    std::cout << "Input samples at/over full scale:  " << g_clip_in.load() << "\n";
    std::cout << "Output samples over full scale:    " << g_clip_out.load() << "\n";
    if (g_clip_in.load() > 0) {
        std::cout << "-> The INPUT is clipping at the interface. Turn the Scarlett's\n"
                     "   gain down; the model is being fed a squared-off signal.\n";
    }
    if (g_clip_out.load() > 0) {
        std::cout << "-> The MODEL OUTPUT exceeds full scale. Without the clamp this\n"
                     "   wraps around in the driver and sounds bit-crushed.\n";
    }
    if (g_xrun_count.load() > 0) {
        std::cout << "-> Xruns occurred: raise the buffer size or check your OS config.\n";
    }

    web.stop();
    audio.stopStream();
    audio.closeStream();

    if (!g_record_path.empty()) {
        const size_t frames = g_rec_pos.load();
        if (writeWavF32(g_record_path, g_rec.data(), frames, 2, SAMPLE_RATE)) {
            std::cout << "\nWrote " << frames << " frames ("
                      << (static_cast<double>(frames) / SAMPLE_RATE)
                      << " s) to " << g_record_path << "\n";
        } else {
            std::cerr << "\nFailed to write " << g_record_path << "\n";
        }
    }
    return 0;
}
