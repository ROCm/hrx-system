// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, hip_wait_mode, "auto",
          "HIP synchronization policy: auto, spin, yield, or blocking.");

namespace {

constexpr uint8_t kPayload = 0xA5;
constexpr uint8_t kPadding = 0x3C;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

void Check(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "HIP operation failed: %d\n",
                 static_cast<int>(result));
    std::abort();
  }
}

// All rows share one frontend, device, and stream. Allocations and registration
// belong to each row and stay outside the timed copy/completion boundary.
class TransferRuntime {
 public:
  TransferRuntime() {
    if (!library_.Open()) {
      Require(false, library_.error().c_str());
    }
    Check(Resolve<decltype(&hipInit)>("hipInit")(0));
    const std::string wait_mode = FLAG_hip_wait_mode;
    unsigned int flags = hipDeviceScheduleAuto;
    if (wait_mode == "spin") {
      flags = hipDeviceScheduleSpin;
    } else if (wait_mode == "yield") {
      flags = hipDeviceScheduleYield;
    } else if (wait_mode == "blocking") {
      flags = hipDeviceScheduleBlockingSync;
    } else {
      Require(wait_mode == "auto", "invalid HIP synchronization policy");
    }
    Check(Resolve<decltype(&hipSetDeviceFlags)>("hipSetDeviceFlags")(flags));
    int device = 0;
    Check(Resolve<decltype(&hipGetDevice)>("hipGetDevice")(&device));
    hipDeviceProp_t properties = {};
    Check(Resolve<decltype(&hipGetDevicePropertiesR0600)>(
        "hipGetDevicePropertiesR0600")(&properties, device));
    char pci_address[32] = {};
    Check(Resolve<decltype(&hipDeviceGetPCIBusId)>("hipDeviceGetPCIBusId")(
        pci_address, sizeof(pci_address), device));
    const char* driver = std::getenv("HRX_GPU_DRIVER");
    label_ = std::string(driver && driver[0] ? driver : "configured-default") +
             " " + properties.gcnArchName + " " + pci_address +
             " wait=" + wait_mode;
    Check(Resolve<decltype(&hipStreamCreateWithFlags)>(
        "hipStreamCreateWithFlags")(&stream_, hipStreamNonBlocking));
    synchronize_ =
        Resolve<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
  }

  ~TransferRuntime() {
    Wait();
    Check(Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy")(stream_));
    Check(Resolve<decltype(&hipHALDeinit)>("hipHALDeinit")());
    if (!library_.Close()) {
      Require(false, library_.error().c_str());
    }
  }

  template <typename T>
  T Resolve(const char* name) {
    T function = library_.Resolve<T>(name);
    Require(function != nullptr, library_.error().c_str());
    return function;
  }

  void Wait() { Check(synchronize_(stream_)); }
  hipStream_t stream() const { return stream_; }
  const std::string& label() const { return label_; }

 private:
  // Exact frontend DSO, closed after synchronous runtime teardown.
  hrx::hip::testing::HipDso library_;
  // One nonblocking stream reused throughout this process.
  hipStream_t stream_ = nullptr;
  // Cached public completion function included in the measured interval.
  decltype(&hipStreamSynchronize) synchronize_ = nullptr;
  // Backend, actual GPU identity, and public wait policy.
  std::string label_;
};

TransferRuntime& Runtime() {
  static TransferRuntime runtime;
  return runtime;
}

enum DownloadFlags : uint32_t {
  kDefault = 0,
  kRegistered = 1u << 0,
  kPitched = 1u << 1,
  kDeviceInterval = 1u << 2,
};

void Download(benchmark::State& state, uint32_t flags,
              size_t row_width = 4096) {
  auto& runtime = Runtime();
  const bool registered = (flags & kRegistered) != 0;
  const bool pitched = (flags & kPitched) != 0;
  const size_t length = state.range(0);
  const size_t width = pitched ? std::min(row_width, length) : length;
  const size_t height = length / width;
  const size_t source_pitch = width + (pitched ? 64 : 0);
  const size_t target_pitch = width + (pitched ? 128 : 0);
  const size_t source_length = source_pitch * height;
  const size_t target_length = target_pitch * height;
  const size_t allocation_length = (target_length + 4095) & ~size_t{4095};
  auto* target =
      static_cast<uint8_t*>(std::aligned_alloc(4096, allocation_length));
  Require(target != nullptr, "host allocation failed");
  std::memset(target, kPadding, allocation_length);
  if (registered) {
    Check(runtime.Resolve<decltype(&hipHostRegister)>("hipHostRegister")(
        target, allocation_length, hipHostRegisterDefault));
  }
  hipDeviceptr_t source = nullptr;
  Check(runtime.Resolve<decltype(&hipMalloc)>("hipMalloc")(&source,
                                                           source_length));
  {
    std::vector<uint8_t> seed(source_length, kPayload);
    for (size_t row = 0; row < height; ++row) {
      const uint32_t begin = static_cast<uint32_t>(row);
      const uint32_t end = ~begin;
      std::memcpy(seed.data() + row * source_pitch, &begin, sizeof(begin));
      std::memcpy(seed.data() + row * source_pitch + width - sizeof(end), &end,
                  sizeof(end));
    }
    Check(runtime.Resolve<decltype(&hipMemcpy)>("hipMemcpy")(
        source, seed.data(), seed.size(), hipMemcpyHostToDevice));
  }
  auto copy = runtime.Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
  auto copy_2d =
      runtime.Resolve<decltype(&hipMemcpy2DAsync)>("hipMemcpy2DAsync");
  const auto transfer = [&] {
    if (pitched) {
      Check(copy_2d(target, target_pitch, source, source_pitch, width, height,
                    hipMemcpyDeviceToHost, runtime.stream()));
    } else {
      Check(copy(target, source, length, hipMemcpyDeviceToHost,
                 runtime.stream()));
    }
  };
  const auto verify = [&] {
    for (size_t row = 0; row < height; ++row) {
      const uint8_t* bytes = target + row * target_pitch;
      uint32_t begin = 0, end = 0;
      std::memcpy(&begin, bytes, sizeof(begin));
      std::memcpy(&end, bytes + width - sizeof(end), sizeof(end));
      Require(begin == row && end == ~static_cast<uint32_t>(row),
              "download row identity mismatch");
      for (size_t column = sizeof(begin); column < width - sizeof(end);
           ++column) {
        Require(bytes[column] == kPayload, "download payload mismatch");
      }
      for (size_t column = width; column < target_pitch; ++column) {
        Require(bytes[column] == kPadding, "download overwrote row padding");
      }
    }
    for (size_t i = target_length; i < allocation_length; ++i) {
      Require(target[i] == kPadding, "download overwrote allocation padding");
    }
  };
  transfer();
  runtime.Wait();
  verify();
  std::memset(target, kPadding, allocation_length);
  if (flags & kDeviceInterval) {
    auto create = runtime.Resolve<decltype(&hipEventCreate)>("hipEventCreate");
    auto destroy =
        runtime.Resolve<decltype(&hipEventDestroy)>("hipEventDestroy");
    auto record = runtime.Resolve<decltype(&hipEventRecord)>("hipEventRecord");
    auto elapsed =
        runtime.Resolve<decltype(&hipEventElapsedTime)>("hipEventElapsedTime");
    hipEvent_t begin = nullptr, end = nullptr;
    Check(create(&begin));
    Check(create(&end));
    const auto enqueue = [&] {
      Check(record(begin, runtime.stream()));
      transfer();
      Check(record(end, runtime.stream()));
      runtime.Wait();
    };
    enqueue();
    double host_nanoseconds = 0;
    for (auto _ : state) {
      const auto start = std::chrono::steady_clock::now();
      enqueue();
      const auto finish = std::chrono::steady_clock::now();
      float milliseconds = 0;
      Check(elapsed(&milliseconds, begin, end));
      Require(milliseconds > 0, "device event interval did not advance");
      state.SetIterationTime(static_cast<double>(milliseconds) * 1e-3);
      host_nanoseconds +=
          std::chrono::duration<double, std::nano>(finish - start).count();
    }
    Check(destroy(end));
    Check(destroy(begin));
    state.counters["host_ns"] = benchmark::Counter(
        host_nanoseconds, benchmark::Counter::kAvgIterations);
  } else {
    for (auto _ : state) {
      transfer();
      runtime.Wait();
    }
  }
  verify();
  Check(runtime.Resolve<decltype(&hipFree)>("hipFree")(source));
  if (registered) {
    Check(runtime.Resolve<decltype(&hipHostUnregister)>("hipHostUnregister")(
        target));
  }
  std::free(target);
  state.SetLabel(runtime.label());
  state.SetBytesProcessed(state.iterations() * length);
  state.counters["width"] = width;
  state.counters["height"] = height;
  state.counters["source_pitch"] = source_pitch;
  state.counters["target_pitch"] = target_pitch;
}

#define DOWNLOAD_CASE(name, flags)         \
  BENCHMARK_CAPTURE(Download, name, flags) \
      ->Name("HIP/Download/" #name)        \
      ->Arg(4 * 1024)                      \
      ->Arg(1024 * 1024)                   \
      ->Arg(64 * 1024 * 1024)              \
      ->UseRealTime()
DOWNLOAD_CASE(PageableContiguous, kDefault);
DOWNLOAD_CASE(RegisteredContiguous, kRegistered);
DOWNLOAD_CASE(PageablePitched, kPitched);
DOWNLOAD_CASE(RegisteredPitched, kRegistered | kPitched);
#undef DOWNLOAD_CASE

// Sparse event pairs expose device-timeline gaps independently of host return
// and waiting. Intervals include any host gaps between device commands, and
// host_ns includes event enqueues and completion but not the elapsed-time
// query. Constant payload with varying row width separates byte and copy-count
// costs.
#define DOWNLOAD_INTERVAL_CASE(name, flags, width)                             \
  BENCHMARK_CAPTURE(Download, name##DeviceInterval, (flags) | kDeviceInterval, \
                    width)                                                     \
      ->Name("HIP/DownloadDeviceInterval/" #name)                              \
      ->Arg(64 * 1024 * 1024)                                                  \
      ->UseManualTime()
DOWNLOAD_INTERVAL_CASE(RegisteredContiguous, kRegistered, 0);
DOWNLOAD_INTERVAL_CASE(RegisteredPitched4K, kRegistered | kPitched, 4096);
DOWNLOAD_INTERVAL_CASE(RegisteredPitched64K, kRegistered | kPitched, 65536);
DOWNLOAD_INTERVAL_CASE(RegisteredPitched1M, kRegistered | kPitched,
                       1024 * 1024);
DOWNLOAD_INTERVAL_CASE(RegisteredPitched4M, kRegistered | kPitched,
                       4 * 1024 * 1024);
DOWNLOAD_INTERVAL_CASE(PageableContiguous, kDefault, 0);
DOWNLOAD_INTERVAL_CASE(PageablePitched4K, kPitched, 4096);
DOWNLOAD_INTERVAL_CASE(PageablePitched64K, kPitched, 65536);
DOWNLOAD_INTERVAL_CASE(PageablePitched1M, kPitched, 1024 * 1024);
DOWNLOAD_INTERVAL_CASE(PageablePitched4M, kPitched, 4 * 1024 * 1024);
#undef DOWNLOAD_INTERVAL_CASE

}  // namespace
