#include "RtAudio.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"
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
    nam::DSP* model;
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

    const auto t0 = std::chrono::steady_clock::now();
    if (g_bypass) {
        std::memcpy(mo, mi, nFrames * sizeof(float));
    } else {
        float* in_channels[]  = { mi };
        float* out_channels[] = { mo };
        data->model->process(in_channels, out_channels, static_cast<int>(nFrames));
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

    std::unique_ptr<nam::DSP> model;
    try {
        model = nam::get_dsp(modelPath);
    } catch (std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    if (model->NumInputChannels() != 1 || model->NumOutputChannels() != 1) {
        std::cerr << "This prototype only supports mono 1-in/1-out models.\n";
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
    if (model->HasLoudness()) {
        std::cout << "Model loudness: " << model->GetLoudness() << " dB\n";
    } else {
        std::cout << "Model loudness: unknown\n";
    }

    const double modelRate = model->GetExpectedSampleRate();
    if (modelRate > 0.0 && modelRate != static_cast<double>(SAMPLE_RATE)) {
        std::cerr << "WARNING: model expects " << modelRate << " Hz but the stream runs at "
                  << SAMPLE_RATE << " Hz. No resampling is done, so it will sound off "
                     "(brighter/darker and wrong in time).\n";
    }

    if (!g_record_path.empty()) {
        g_rec_capacity = 60 * SAMPLE_RATE;              // 60 seconds is plenty
        g_rec.assign(g_rec_capacity * 2, 0.0f);
        std::cout << "Recording to " << g_record_path << " (up to 60 s, L=input R=output)\n";
    }

    CallbackData callbackData;
    callbackData.model = model.get();
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
        model->Reset(static_cast<double>(SAMPLE_RATE), static_cast<int>(bufferFrames));

        // Slimmable models carry several submodels at different cost/quality.
        // Reset() first so the submodel gets the right sample rate and block size.
        if (auto* slim = dynamic_cast<nam::SlimmableModel*>(model.get())) {
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
    std::cout << "Channels:     " << CHANNELS << " in / " << CHANNELS
              << " out (device native; guitar read from channel 1)\n";
    std::cout << "Output clamp: " << (CLAMP_OUTPUT ? "ON" : "OFF") << "\n";
    std::cout << "Model:        " << (g_bypass ? "BYPASSED (straight passthrough)" : "active") << "\n";
    std::cout << "\nLive meters (peak over each 500ms window):\n";
    std::cout << "Press ENTER to stop...\n\n";

    // Meter thread. Reads the peaks the callback records and resets them, so
    // each line is the peak of the last window rather than of the whole run.
    std::thread meter([]() {
        while (g_running.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const float pin  = g_peak_in.exchange(0.0f, std::memory_order_relaxed);
            const float pout = g_peak_out.exchange(0.0f, std::memory_order_relaxed);
            const long long n   = g_proc_count.load();
            const long long tot  = g_proc_ns_total.load();
            const double avgPct  = (n > 0 && g_deadline_ns > 0.0)
                                 ? 100.0 * (static_cast<double>(tot) / n) / g_deadline_ns : 0.0;
            const double maxPct  = (g_deadline_ns > 0.0)
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
    g_running.store(false, std::memory_order_relaxed);
    meter.join();

    std::cout << "\n\n--- session summary ---\n";
    std::cout << "Xruns:            " << g_xrun_count.load() << "\n";
    const long long n = g_proc_count.load();
    if (n > 0) {
        const double avgUs = (static_cast<double>(g_proc_ns_total.load()) / n) / 1000.0;
        const double maxUs = g_proc_ns_max.load() / 1000.0;
        const double budgetUs = g_deadline_ns / 1000.0;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Block deadline:   " << budgetUs << " us\n";
        std::cout << "Process time:     avg " << avgUs << " us ("
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
