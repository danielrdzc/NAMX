#include "RtAudio.h"
#include "NAM/get_dsp.h"
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
#include <filesystem>
#include "json.hpp"

constexpr unsigned int SAMPLE_RATE   = 48000;
constexpr unsigned int BUFFER_FRAMES = 64;
constexpr unsigned int CHANNELS      = 1;

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
};

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

    if (!in) {
        std::memset(out, 0, nFrames * CHANNELS * sizeof(float));
        return 0;
    }

    CallbackData* data = static_cast<CallbackData*>(userData);

    // ---- measure the input BEFORE the model sees it -------------------------
    float     peakIn = 0.0f;
    long long clipIn = 0;
    for (unsigned int i = 0; i < nFrames * CHANNELS; ++i) {
        const float a = std::fabs(in[i]);
        if (a > peakIn) peakIn = a;
        if (a >= 0.999f) ++clipIn;   // already clipped by the interface
    }

    if (g_bypass) {
        std::memcpy(out, in, nFrames * CHANNELS * sizeof(float));
    } else {
        float* in_channels[]  = { in };
        float* out_channels[] = { out };
        data->model->process(in_channels, out_channels, static_cast<int>(nFrames));
    }

    // ---- measure (and optionally clamp) what the model produced -------------
    float     peakOut = 0.0f;
    long long clipOut = 0;
    for (unsigned int i = 0; i < nFrames * CHANNELS; ++i) {
        const float y = out[i];
        const float a = std::fabs(y);
        if (a > peakOut) peakOut = a;
        if (a > 1.0f) {
            ++clipOut;
            if (CLAMP_OUTPUT) out[i] = (y > 0.0f) ? 1.0f : -1.0f;
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
        } else {
            modelPath = arg;
        }
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
    // ALSA gives direct hardware access; JACK/Pulse would add their own buffering.
    RtAudio* audioPtr = new RtAudio(RtAudio::LINUX_ALSA);
    if (audioPtr->getDeviceCount() < 1) {
        std::cerr << "No ALSA devices found, falling back to default API.\n";
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
            std::cout << "Gear type:    " << field("gear_type")
                      << "   (\"amp\" = no cab baked in; you need an IR after it)\n";
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

    CallbackData callbackData{ model.get() };

    RtAudio::StreamParameters inParams, outParams;
    inParams.deviceId   = foundInterface ? interfaceId : audio.getDefaultInputDevice();
    inParams.nChannels  = CHANNELS;
    outParams.deviceId  = foundInterface ? interfaceId : audio.getDefaultOutputDevice();
    outParams.nChannels = CHANNELS;

    unsigned int bufferFrames = g_buffer_frames;

    RtAudio::StreamOptions options;
    options.flags = RTAUDIO_SCHEDULE_REALTIME;
    options.priority = 90;

    try {
        audio.openStream(&outParams, &inParams, RTAUDIO_FLOAT32,
                         SAMPLE_RATE, &bufferFrames, &audioCallback,
                         &callbackData, &options);

        // Reset() sets the sample rate + sizes the internal buffers and prewarms.
        // It must happen after openStream (bufferFrames may come back different
        // from what we asked for) and before startStream.
        g_reset_frames = bufferFrames;
        model->Reset(static_cast<double>(SAMPLE_RATE), static_cast<int>(bufferFrames));

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
            std::cout << "\r  in " << dbfs(pin) << " dBFS   out " << dbfs(pout)
                      << " dBFS   clipped in/out: " << g_clip_in.load() << "/"
                      << g_clip_out.load() << "   xruns: " << g_xrun_count.load()
                      << "        " << std::flush;
        }
    });

    std::cin.get();
    g_running.store(false, std::memory_order_relaxed);
    meter.join();

    std::cout << "\n\n--- session summary ---\n";
    std::cout << "Xruns:            " << g_xrun_count.load() << "\n";
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
    return 0;
}
