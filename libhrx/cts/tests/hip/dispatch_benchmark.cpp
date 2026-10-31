// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "benchmark/benchmark.h"
#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/base/tooling/flags.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

IREE_FLAG(string, hip_wait_mode, "auto",
          "HIP synchronization policy: auto, spin, yield, or blocking.");

namespace {

constexpr uint32_t kExpectedOutput = 0xC0FFEE;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

// One runtime and module span every benchmark row. Only graphs are rebuilt
// between rows; native drivers and devices are never cycled in this process.
class HipRuntime {
 public:
  HipRuntime() {
    if (!library_.Open()) {
      Require(false, library_.error().c_str());
    }
    Check(Resolve<decltype(&hipInit)>("hipInit")(0));
    unsigned int device_flags = hipDeviceScheduleAuto;
    const std::string wait_mode = FLAG_hip_wait_mode;
    if (wait_mode == "spin") {
      device_flags = hipDeviceScheduleSpin;
    } else if (wait_mode == "yield") {
      device_flags = hipDeviceScheduleYield;
    } else if (wait_mode == "blocking") {
      device_flags = hipDeviceScheduleBlockingSync;
    } else {
      Require(wait_mode == "auto", "invalid HIP synchronization policy");
    }
    Check(Resolve<decltype(&hipSetDeviceFlags)>("hipSetDeviceFlags")(
        device_flags));
    launch_ =
        Resolve<decltype(&hipModuleLaunchKernel)>("hipModuleLaunchKernel");
    synchronize_ =
        Resolve<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
    graph_launch_ = Resolve<decltype(&hipGraphLaunch)>("hipGraphLaunch");

    Check(Resolve<decltype(&hipGetDevice)>("hipGetDevice")(&device_));
    hipDeviceProp_t properties = {};
    Check(Resolve<decltype(&hipGetDevicePropertiesR0600)>(
        "hipGetDevicePropertiesR0600")(&properties, device_));
    char pci_address[32] = {};
    Check(Resolve<decltype(&hipDeviceGetPCIBusId)>("hipDeviceGetPCIBusId")(
        pci_address, sizeof(pci_address), device_));
    const char* driver = std::getenv("HRX_GPU_DRIVER");
    label_ = std::string(driver && driver[0] ? driver : "configured-default") +
             " " + properties.gcnArchName + " " + pci_address +
             " dispatch=queue wait=" + wait_mode;

    const auto image =
        hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
    Require(image.file != nullptr,
            "no configured HSACO for the selected device");
    Check(Resolve<decltype(&hipModuleLoadData)>("hipModuleLoadData")(
        &module_, image.file->data));
    Check(Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction")(
        &function_, module_, "hrx_store_output"));
    Check(
        Resolve<decltype(&hipMalloc)>("hipMalloc")(&output_, sizeof(uint32_t)));
    Check(Resolve<decltype(&hipStreamCreateWithFlags)>(
        "hipStreamCreateWithFlags")(&stream_, hipStreamNonBlocking));
  }

  ~HipRuntime() {
    Wait();
    Check(Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy")(stream_));
    Check(Resolve<decltype(&hipFree)>("hipFree")(output_));
    Check(Resolve<decltype(&hipModuleUnload)>("hipModuleUnload")(module_));
    Check(Resolve<decltype(&hipHALDeinit)>("hipHALDeinit")());
    if (!library_.Close()) {
      Require(false, library_.error().c_str());
    }
  }

  HipRuntime(const HipRuntime&) = delete;
  HipRuntime& operator=(const HipRuntime&) = delete;

  template <typename T>
  T Resolve(const char* name) {
    const auto function = library_.Resolve<T>(name);
    Require(function != nullptr, library_.error().c_str());
    return function;
  }

  static void Check(hipError_t result) {
    if (result != hipSuccess) {
      std::fprintf(stderr, "HIP operation failed: %d\n",
                   static_cast<int>(result));
      std::abort();
    }
  }

  void Launch() {
    uint32_t value = kExpectedOutput;
    void* arguments[] = {&output_, &value};
    Check(launch_(function_, 1, 1, 1, 1, 1, 1, 0, stream_, arguments, nullptr));
  }

  void LaunchGraph(hipGraphExec_t executable) {
    Check(graph_launch_(executable, stream_));
  }

  void Wait() { Check(synchronize_(stream_)); }

  void ResetOutput() {
    Check(Resolve<decltype(&hipMemset)>("hipMemset")(output_, 0,
                                                     sizeof(uint32_t)));
    Wait();
  }

  void VerifyOutput() {
    uint32_t actual = 0;
    Check(Resolve<decltype(&hipMemcpy)>("hipMemcpy")(
        &actual, output_, sizeof(actual), hipMemcpyDeviceToHost));
    Require(actual == kExpectedOutput, "kernel output mismatch");
  }

  hipStream_t stream() const { return stream_; }
  int device() const { return device_; }
  const std::string& label() const { return label_; }

 private:
  // Exact built frontend DSO, closed only after synchronous runtime teardown.
  hrx::hip::testing::HipDso library_;
  // Visible ordinal selected by the frontend, shared by application threads.
  int device_ = 0;
  // Reused code object, retained through every graph and launch.
  hipModule_t module_ = nullptr;
  // Finite single-workitem kernel writing one exact output word.
  hipFunction_t function_ = nullptr;
  // Four-byte device allocation live through final completion.
  hipDeviceptr_t output_ = nullptr;
  // Nonblocking HIP stream shared by recording and streaming rows.
  hipStream_t stream_ = nullptr;
  // Cached public launch entry point used in the timed region.
  decltype(&hipModuleLaunchKernel) launch_ = nullptr;
  // Cached public completion entry point included only in end-to-end rows.
  decltype(&hipStreamSynchronize) synchronize_ = nullptr;
  // Cached public graph entry point used in the timed region.
  decltype(&hipGraphLaunch) graph_launch_ = nullptr;
  // Backend, actual architecture/PCI identity, and adapter submission policy.
  std::string label_;
};

HipRuntime& Runtime() {
  static HipRuntime runtime;
  return runtime;
}

enum class DownloadMode {
  kIdle,
  kPageable,
  kRegistered,
};

enum class DownloadPriority {
  kNormal,
  kLow,
};

// Owns one application's continuous download stream. The worker is necessary
// because a pageable HIP copy may block its caller; it shares the existing
// primary context and never creates another native driver or device.
class BackgroundDownload {
 public:
  BackgroundDownload(HipRuntime& runtime, size_t length, DownloadMode mode,
                     int priority)
      : runtime_(runtime),
        length_(length),
        allocation_length_(((length + 4095) & ~size_t{4095}) + 4096),
        mode_(mode) {
    copy_ = runtime_.Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
    synchronize_ = runtime_.Resolve<decltype(&hipStreamSynchronize)>(
        "hipStreamSynchronize");
    set_device_ = runtime_.Resolve<decltype(&hipSetDevice)>("hipSetDevice");
    target_ =
        static_cast<uint8_t*>(std::aligned_alloc(4096, allocation_length_));
    Require(target_ != nullptr, "background download allocation failed");
    std::memset(target_, kGuard, allocation_length_);
    if (mode_ == DownloadMode::kRegistered) {
      HipRuntime::Check(
          runtime_.Resolve<decltype(&hipHostRegister)>("hipHostRegister")(
              target_, allocation_length_, hipHostRegisterDefault));
    }
    HipRuntime::Check(
        runtime_.Resolve<decltype(&hipMalloc)>("hipMalloc")(&source_, length_));
    HipRuntime::Check(runtime_.Resolve<decltype(&hipStreamCreateWithPriority)>(
        "hipStreamCreateWithPriority")(&stream_, hipStreamNonBlocking,
                                       priority));
    HipRuntime::Check(runtime_.Resolve<decltype(&hipMemsetAsync)>(
        "hipMemsetAsync")(source_, kPayload, length_, stream_));
    HipRuntime::Check(synchronize_(stream_));
    worker_ = std::thread([this] {
      HipRuntime::Check(set_device_(runtime_.device()));
      do {
        HipRuntime::Check(
            copy_(target_, source_, length_, hipMemcpyDeviceToHost, stream_));
        HipRuntime::Check(synchronize_(stream_));
        if (completed_.fetch_add(1, std::memory_order_release) == 0) {
          completed_.notify_one();
        }
      } while (!stop_.load(std::memory_order_relaxed));
    });
    // The timed foreground begins only after this stream has completed real
    // work. Readiness does not depend on a scheduler delay or wall-clock timer.
    completed_.wait(0, std::memory_order_acquire);
  }

  ~BackgroundDownload() {
    Stop();
    Require(std::all_of(target_, target_ + length_,
                        [](uint8_t value) { return value == kPayload; }),
            "background download payload mismatch");
    Require(std::all_of(target_ + length_, target_ + allocation_length_,
                        [](uint8_t value) { return value == kGuard; }),
            "background download overwrote its guard");
    HipRuntime::Check(runtime_.Resolve<decltype(&hipStreamDestroy)>(
        "hipStreamDestroy")(stream_));
    HipRuntime::Check(runtime_.Resolve<decltype(&hipFree)>("hipFree")(source_));
    if (mode_ == DownloadMode::kRegistered) {
      HipRuntime::Check(runtime_.Resolve<decltype(&hipHostUnregister)>(
          "hipHostUnregister")(target_));
    }
    std::free(target_);
  }

  BackgroundDownload(const BackgroundDownload&) = delete;
  BackgroundDownload& operator=(const BackgroundDownload&) = delete;

  void Stop() {
    stop_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  uint64_t completed() const {
    return completed_.load(std::memory_order_acquire);
  }

 private:
  static constexpr uint8_t kPayload = 0xA5;
  static constexpr uint8_t kGuard = 0x3C;

  // Borrowed process runtime, alive until after the application thread joins.
  HipRuntime& runtime_;
  // Useful bytes copied in each background transaction.
  size_t length_;
  // Page-aligned destination extent, including an untouched guard page.
  size_t allocation_length_;
  // Host destination registration policy, fixed throughout the row.
  DownloadMode mode_;
  // Device source retained until every background operation completes.
  hipDeviceptr_t source_ = nullptr;
  // CPU-owned destination retained through completion and unregistration.
  uint8_t* target_ = nullptr;
  // Independent nonblocking stream, joined before destruction.
  hipStream_t stream_ = nullptr;
  // Cached copy entry point called by the background application thread.
  decltype(&hipMemcpyAsync) copy_ = nullptr;
  // Cached completion entry point for the background stream.
  decltype(&hipStreamSynchronize) synchronize_ = nullptr;
  // Cached entry point selecting the existing primary context on the worker.
  decltype(&hipSetDevice) set_device_ = nullptr;
  // Complete downloads; the first increment releases constructor readiness.
  std::atomic<uint64_t> completed_{0};
  // Stop request observed only between joined downloads.
  std::atomic<bool> stop_{false};
  // Application copy thread, joined before releasing any captured resource.
  std::thread worker_;
};

// Foreground latency includes one dispatch and its usable host completion.
// Process CPU includes both callers and the runtime's service/retirement work.
void DispatchDuringDownload(
    benchmark::State& state, DownloadMode mode,
    DownloadPriority priority = DownloadPriority::kNormal) {
  auto& runtime = Runtime();
  int copy_priority = 0;
  if (priority == DownloadPriority::kLow) {
    int greatest_priority = 0;
    HipRuntime::Check(
        runtime.Resolve<decltype(&hipDeviceGetStreamPriorityRange)>(
            "hipDeviceGetStreamPriorityRange")(&copy_priority,
                                               &greatest_priority));
    if (copy_priority <= 0) {
      state.SkipWithError("HIP device has no stream priority below normal");
      return;
    }
  }
  runtime.Launch();
  runtime.Wait();
  runtime.VerifyOutput();
  runtime.ResetOutput();
  const size_t length = state.range(0);
  std::unique_ptr<BackgroundDownload> background;
  if (mode != DownloadMode::kIdle) {
    background = std::make_unique<BackgroundDownload>(runtime, length, mode,
                                                      copy_priority);
  }
  const uint64_t first = background ? background->completed() : 0;
  const auto start = std::chrono::steady_clock::now();
  for (auto _ : state) {
    runtime.Launch();
    runtime.Wait();
  }
  const auto finish = std::chrono::steady_clock::now();
  const uint64_t count = background ? background->completed() - first : 0;
  if (background) {
    background->Stop();
  }
  runtime.VerifyOutput();
  state.SetLabel(runtime.label());
  state.SetItemsProcessed(state.iterations());
  state.counters["downloads"] = static_cast<double>(count);
  state.counters["download_priority"] = copy_priority;
  state.counters["download_bytes_per_second"] =
      static_cast<double>(count) * length /
      std::chrono::duration<double>(finish - start).count();
}

// Owns the public capture and executable until every measured replay completes.
class HipGraph {
 public:
  HipGraph(HipRuntime& runtime, int64_t dispatch_count) : runtime_(runtime) {
    HipRuntime::Check(runtime_.Resolve<decltype(&hipStreamBeginCapture)>(
        "hipStreamBeginCapture")(runtime_.stream(),
                                 hipStreamCaptureModeThreadLocal));
    for (int64_t i = 0; i < dispatch_count; ++i) {
      runtime_.Launch();
    }
    HipRuntime::Check(runtime_.Resolve<decltype(&hipStreamEndCapture)>(
        "hipStreamEndCapture")(runtime_.stream(), &graph_));
    HipRuntime::Check(runtime_.Resolve<decltype(&hipGraphInstantiate)>(
        "hipGraphInstantiate")(&executable_, graph_, nullptr, nullptr, 0));
  }

  ~HipGraph() {
    HipRuntime::Check(runtime_.Resolve<decltype(&hipGraphExecDestroy)>(
        "hipGraphExecDestroy")(executable_));
    HipRuntime::Check(runtime_.Resolve<decltype(&hipGraphDestroy)>(
        "hipGraphDestroy")(graph_));
  }

  HipGraph(const HipGraph&) = delete;
  HipGraph& operator=(const HipGraph&) = delete;

  void Launch() { runtime_.LaunchGraph(executable_); }

 private:
  // Borrowed runtime outliving all captures, executables, and queued work.
  HipRuntime& runtime_;
  // Captured serial dispatch program, retained until after its executable.
  hipGraph_t graph_ = nullptr;
  // Reused executable; every iteration waits before the next replay.
  hipGraphExec_t executable_ = nullptr;
};

template <bool kGraph, bool kEndToEnd>
void Dispatch(benchmark::State& state) {
  auto& runtime = Runtime();
  const int64_t dispatch_count = state.range(0);
  std::unique_ptr<HipGraph> graph;
  if constexpr (kGraph) {
    graph = std::make_unique<HipGraph>(runtime, dispatch_count);
    graph->Launch();
  } else {
    for (int64_t i = 0; i < dispatch_count; ++i) {
      runtime.Launch();
    }
  }
  runtime.Wait();
  runtime.VerifyOutput();
  runtime.ResetOutput();

  for (auto _ : state) {
    if constexpr (kGraph) {
      graph->Launch();
    } else {
      for (int64_t i = 0; i < dispatch_count; ++i) {
        runtime.Launch();
      }
    }
    if constexpr (!kEndToEnd) {
      state.PauseTiming();
    }
    runtime.Wait();
    if constexpr (!kEndToEnd) {
      state.ResumeTiming();
    }
  }
  runtime.VerifyOutput();
  state.SetLabel(runtime.label());
  state.SetItemsProcessed(state.iterations() * dispatch_count);
}

// Device event intervals include preparation and execution between two queue
// timestamps. The zero-dispatch row measures the event pair alone. Host time
// includes both event enqueues and completion, excluding the elapsed-time read.
void GraphEventInterval(benchmark::State& state) {
  auto& runtime = Runtime();
  const int64_t dispatch_count = state.range(0);
  auto create = runtime.Resolve<decltype(&hipEventCreate)>("hipEventCreate");
  auto destroy = runtime.Resolve<decltype(&hipEventDestroy)>("hipEventDestroy");
  auto record = runtime.Resolve<decltype(&hipEventRecord)>("hipEventRecord");
  auto elapsed =
      runtime.Resolve<decltype(&hipEventElapsedTime)>("hipEventElapsedTime");
  hipEvent_t begin = nullptr, end = nullptr;
  HipRuntime::Check(create(&begin));
  HipRuntime::Check(create(&end));
  std::unique_ptr<HipGraph> graph;
  if (dispatch_count) {
    graph = std::make_unique<HipGraph>(runtime, dispatch_count);
  }
  const auto enqueue = [&]() {
    HipRuntime::Check(record(begin, runtime.stream()));
    if (graph) {
      graph->Launch();
    }
    HipRuntime::Check(record(end, runtime.stream()));
    runtime.Wait();
  };
  enqueue();
  if (graph) {
    runtime.VerifyOutput();
  }
  runtime.ResetOutput();

  double host_nanoseconds = 0;
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    enqueue();
    const auto finish = std::chrono::steady_clock::now();
    float milliseconds = 0;
    HipRuntime::Check(elapsed(&milliseconds, begin, end));
    Require(milliseconds > 0, "device event interval did not advance");
    state.SetIterationTime(static_cast<double>(milliseconds) * 1e-3);
    host_nanoseconds +=
        std::chrono::duration<double, std::nano>(finish - start).count();
  }
  if (graph) {
    runtime.VerifyOutput();
  }
  HipRuntime::Check(destroy(end));
  HipRuntime::Check(destroy(begin));
  state.SetLabel(runtime.label());
  state.counters["host_ns"] =
      benchmark::Counter(host_nanoseconds, benchmark::Counter::kAvgIterations);
  state.SetItemsProcessed(state.iterations() * dispatch_count);
}

// Enqueue rows measure the public API return with completion outside timing.
// Direct dispatches and graph launches both publish to the physical queue.
BENCHMARK_TEMPLATE(Dispatch, false, false)
    ->Name("HIP/Dispatch/Enqueue")
    ->Arg(1)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->UseRealTime();
BENCHMARK_TEMPLATE(Dispatch, false, true)
    ->Name("HIP/Dispatch/EndToEnd")
    ->Arg(1)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->UseRealTime();
BENCHMARK_TEMPLATE(Dispatch, true, false)
    ->Name("HIP/Graph/Enqueue")
    ->Arg(1)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->UseRealTime();
BENCHMARK_TEMPLATE(Dispatch, true, true)
    ->Name("HIP/Graph/EndToEnd")
    ->Arg(1)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->UseRealTime();

BENCHMARK(GraphEventInterval)
    ->Name("HIP/Graph/EventInterval")
    ->Arg(0)
    ->Arg(1)
    ->Arg(32)
    ->Arg(256)
    ->Arg(4096)
    ->UseManualTime();

BENCHMARK_CAPTURE(DispatchDuringDownload, Idle, DownloadMode::kIdle)
    ->Name("HIP/ConcurrentDownload/Idle")
    ->Arg(64 * 1024 * 1024)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(DispatchDuringDownload, Pageable, DownloadMode::kPageable)
    ->Name("HIP/ConcurrentDownload/Pageable")
    ->Arg(64 * 1024 * 1024)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(DispatchDuringDownload, Registered, DownloadMode::kRegistered)
    ->Name("HIP/ConcurrentDownload/Registered")
    ->Arg(64 * 1024 * 1024)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(DispatchDuringDownload, PageableLowPriority,
                  DownloadMode::kPageable, DownloadPriority::kLow)
    ->Name("HIP/ConcurrentDownload/PageableLowPriority")
    ->Arg(64 * 1024 * 1024)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(DispatchDuringDownload, RegisteredLowPriority,
                  DownloadMode::kRegistered, DownloadPriority::kLow)
    ->Name("HIP/ConcurrentDownload/RegisteredLowPriority")
    ->Arg(64 * 1024 * 1024)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

}  // namespace
