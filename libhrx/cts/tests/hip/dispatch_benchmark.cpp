// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

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
  const std::string& label() const { return label_; }

 private:
  // Exact built frontend DSO, closed only after synchronous runtime teardown.
  hrx::hip::testing::HipDso library_;
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

// Enqueue rows measure the public API return. With dispatch=recording, the
// adapter may defer physical submission to the untimed completion wait. The
// dispatch=queue policy and graph launches include physical queue submission.
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

}  // namespace
