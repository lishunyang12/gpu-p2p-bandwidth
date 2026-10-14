// CUDA peer-copy bandwidth. No models, Python, NCCL, or FlashInfer required.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CUDA(call) check((call), #call)

struct Options {
    int gpus = 8, warmup = 3, iterations = 20, trials = 3;
    int source = -1, destination = -1;
    size_t bytes = 256ULL * 1024 * 1024;
    std::string csv = "bandwidth.csv";
};

void usage() {
    std::cout << "Usage: p2p_bandwidth [options]\n"
              << "  --gpus N          First N visible GPUs (default: 8)\n"
              << "  --pair SRC DST    Only this directed pair (visible ordinals)\n"
              << "  --size-mib N      Buffer size per copy (default: 256 MiB)\n"
              << "  --warmup N        Untimed copies per pair (default: 3)\n"
              << "  --iterations N    Copies per timed trial (default: 20)\n"
              << "  --trials N        Timed trials per pair (default: 3)\n"
              << "  --csv PATH        CSV output (default: bandwidth.csv)\n"
              << "  --help            Help without initializing CUDA\n\n"
              << "Rows=sources, columns=destinations; decimal GB/s.\n"
              << "Pairs run sequentially, not concurrently.\n"
              << "CUDA chooses the PCIe/NVLink route; PCIe is not forced.\n";
}

int integer(const std::string& text, const std::string& option, bool zero = false) {
    size_t used = 0;
    long long value;
    try { value = std::stoll(text, &used); }
    catch (...) { throw std::runtime_error("Invalid integer for " + option); }
    if (used != text.size() || value < (zero ? 0 : 1) || value > std::numeric_limits<int>::max())
        throw std::runtime_error("Invalid integer for " + option);
    return static_cast<int>(value);
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") { usage(); std::exit(0); }
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Missing value for " + key);
            return argv[i];
        };
        if (key == "--gpus") options.gpus = integer(next(), key);
        else if (key == "--size-mib") options.bytes = size_t(integer(next(), key)) * 1024 * 1024;
        else if (key == "--warmup") options.warmup = integer(next(), key, true);
        else if (key == "--iterations") options.iterations = integer(next(), key);
        else if (key == "--trials") options.trials = integer(next(), key);
        else if (key == "--csv") options.csv = next();
        else if (key == "--pair") {
            options.source = integer(next(), key, true);
            options.destination = integer(next(), key, true);
        } else throw std::runtime_error("Unknown option: " + key);
    }
    if (options.gpus < 2) throw std::runtime_error("At least two GPUs are required");
    if (options.source >= 0 && (options.source >= options.gpus ||
        options.destination >= options.gpus || options.source == options.destination))
        throw std::runtime_error("--pair needs two different ordinals within --gpus");
    return options;
}

struct Buffer {
    int device;
    void* pointer = nullptr;
    Buffer(int id, size_t bytes) : device(id) {
        CUDA(cudaSetDevice(device)); CUDA(cudaMalloc(&pointer, bytes));
    }
    ~Buffer() { cudaSetDevice(device); cudaFree(pointer); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// Only the active pair is mapped. Do not substitute host staging on failure.
struct PeerAccess {
    int device, peer;
    bool owned = false;
    PeerAccess(int from, int to) : device(from), peer(to) {
        CUDA(cudaSetDevice(device));
        auto status = cudaDeviceEnablePeerAccess(peer, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else { check(status, "cudaDeviceEnablePeerAccess"); owned = true; }
    }
    ~PeerAccess() {
        if (owned) { cudaSetDevice(device); cudaDeviceDisablePeerAccess(peer); }
    }
    PeerAccess(const PeerAccess&) = delete;
    PeerAccess& operator=(const PeerAccess&) = delete;
};

struct Timer {
    int device;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    explicit Timer(int id) : device(id) {
        CUDA(cudaSetDevice(device));
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    }
    ~Timer() {
        cudaSetDevice(device);
        cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

std::vector<double> measure(int source, int destination, const Options& options) {
    PeerAccess forward(source, destination), backward(destination, source);
    Buffer input(source, options.bytes), output(destination, options.bytes);
    // Nonuniform data catches partial or misplaced copies; verification is untimed.
    std::vector<uint32_t> expected(options.bytes / sizeof(uint32_t)), observed(expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = (uint32_t(i) * 2654435761u) ^ (uint32_t(source + 1) * 2246822519u);
    CUDA(cudaSetDevice(source));
    CUDA(cudaMemcpy(input.pointer, expected.data(), options.bytes, cudaMemcpyHostToDevice));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaSetDevice(destination));
    CUDA(cudaMemset(output.pointer, 0, options.bytes));
    CUDA(cudaDeviceSynchronize());
    Timer timer(source);
    auto copy = [&]() {
        CUDA(cudaMemcpyPeerAsync(output.pointer, destination, input.pointer, source,
                                 options.bytes, timer.stream));
    };
    auto verify = [&]() {
        CUDA(cudaStreamSynchronize(timer.stream));
        CUDA(cudaSetDevice(destination));
        CUDA(cudaMemcpy(observed.data(), output.pointer, options.bytes, cudaMemcpyDeviceToHost));
        CUDA(cudaSetDevice(source));
        if (observed != expected)
            throw std::runtime_error("P2P data verification failed for " +
                                     std::to_string(source) + " -> " + std::to_string(destination));
    };
    copy(); verify();
    for (int i = 0; i < options.warmup; ++i) copy();
    CUDA(cudaStreamSynchronize(timer.stream));
    std::vector<double> samples;
    for (int trial = 0; trial < options.trials; ++trial) {
        CUDA(cudaEventRecord(timer.start, timer.stream));
        for (int i = 0; i < options.iterations; ++i) copy();
        CUDA(cudaEventRecord(timer.stop, timer.stream));
        CUDA(cudaEventSynchronize(timer.stop));
        float elapsed_ms = 0;
        CUDA(cudaEventElapsedTime(&elapsed_ms, timer.start, timer.stop));
        if (!(elapsed_ms > 0) || !std::isfinite(elapsed_ms))
            throw std::runtime_error("Invalid CUDA event duration");
        samples.push_back(double(options.bytes) * options.iterations / (elapsed_ms * 1e6));
    }
    verify();
    return samples;
}

int run(const Options& options) {
    int count = 0;
    CUDA(cudaGetDeviceCount(&count));
    if (count < options.gpus)
        throw std::runtime_error("Need " + std::to_string(options.gpus) +
                                 " visible GPUs, found " + std::to_string(count));
    std::cout << "Using " << options.gpus << " GPUs; " << options.bytes / (1024 * 1024)
              << " MiB/copy; warmup=" << options.warmup << "; iterations="
              << options.iterations << "; trials=" << options.trials << '\n';
    for (int device = 0; device < options.gpus; ++device) {
        cudaDeviceProp properties{};
        char bus[32]{};
        CUDA(cudaGetDeviceProperties(&properties, device));
        CUDA(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));
        std::cout << "GPU " << device << ": " << properties.name << " | PCI " << bus << '\n';
    }
    std::ofstream csv(options.csv);
    if (!csv) throw std::runtime_error("Cannot create CSV: " + options.csv);
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "source,destination,status,bytes_per_copy,iterations,trials,median_GBps,trial_GBps\n";
    std::vector<std::vector<double>> matrix(options.gpus, std::vector<double>(options.gpus, -1));
    int tested = 0, unsupported = 0;
    for (int source = 0; source < options.gpus; ++source) {
        for (int destination = 0; destination < options.gpus; ++destination) {
            if (source == destination || (options.source >= 0 &&
                (source != options.source || destination != options.destination))) continue;
            int forward = 0, backward = 0;
            CUDA(cudaDeviceCanAccessPeer(&forward, source, destination));
            CUDA(cudaDeviceCanAccessPeer(&backward, destination, source));
            if (!forward || !backward) {
                ++unsupported;
                csv << source << ',' << destination << ",unsupported," << options.bytes
                    << ',' << options.iterations << ',' << options.trials << ",,\n";
                std::cout << source << " -> " << destination << ": P2P unsupported\n";
            } else {
                auto samples = measure(source, destination, options);
                double bandwidth = median(samples);
                matrix[source][destination] = bandwidth;
                ++tested;
                csv << source << ',' << destination << ",verified," << options.bytes << ','
                    << options.iterations << ',' << options.trials << ',' << std::fixed
                    << std::setprecision(6) << bandwidth << ',';
                for (size_t i = 0; i < samples.size(); ++i) {
                    if (i) csv << ';';
                    csv << samples[i];
                }
                csv << '\n';
                std::cout << source << " -> " << destination << ": " << std::fixed
                          << std::setprecision(2) << bandwidth << " GB/s (verified)\n";
            }
            csv.flush(); std::cout.flush();
        }
    }
    std::cout << "\nUnidirectional P2P GB/s (median); rows=source, columns=destination\n"
              << std::setw(7) << "src/dst";
    for (int i = 0; i < options.gpus; ++i) std::cout << std::setw(10) << i;
    std::cout << '\n';
    for (int source = 0; source < options.gpus; ++source) {
        std::cout << std::setw(7) << source;
        for (int destination = 0; destination < options.gpus; ++destination) {
            if (source == destination) std::cout << std::setw(10) << "--";
            else if (matrix[source][destination] < 0) std::cout << std::setw(10) << "N/A";
            else std::cout << std::setw(10) << matrix[source][destination];
        }
        std::cout << '\n';
    }
    std::cout << "\nVerified pairs: " << tested << "; unsupported: " << unsupported
              << "\nCSV: " << options.csv << '\n';
    return unsupported ? 2 : 0;
}

int main(int argc, char** argv) {
    try { return run(parse(argc, argv)); }
    catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
