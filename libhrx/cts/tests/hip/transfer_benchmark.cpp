// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <sys/mman.h>

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
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

IREE_FLAG(string, hip_wait_mode, "auto",
          "HIP synchronization policy: auto, spin, yield, or blocking.");
IREE_FLAG(bool, hip_native_runtime, false,
          "Benchmark native HIP selected by HRX_TEST_LIBAMDHIP64, without "
          "requiring HRX's private hipHALDeinit entry point.");

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
    int runtime_version = 0;
    Check(Resolve<decltype(&hipRuntimeGetVersion)>("hipRuntimeGetVersion")(
        &runtime_version));
    std::fprintf(stderr, "HIP runtime: %s, version: %d\n",
                 FLAG_hip_native_runtime ? "native" : "HRX", runtime_version);
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
    architecture_ = properties.gcnArchName;
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
    if (module_) {
      Check(Resolve<decltype(&hipModuleUnload)>("hipModuleUnload")(module_));
    }
    if (!FLAG_hip_native_runtime) {
      Check(Resolve<decltype(&hipHALDeinit)>("hipHALDeinit")());
    }
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
  hipFunction_t read_function() {
    if (!module_) {
      const auto image = hrx_cts::FindAmdgpuExecutableTestImage(architecture_);
      Require(image.file != nullptr,
              "no configured HSACO for the selected device");
      Check(Resolve<decltype(&hipModuleLoadData)>("hipModuleLoadData")(
          &module_, image.file->data));
      Check(Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction")(
          &read_function_, module_, "hrx_read_repeated"));
    }
    return read_function_;
  }
  hipStream_t stream() const { return stream_; }
  const std::string& label() const { return label_; }

 private:
  // Exact frontend DSO, closed after synchronous runtime teardown.
  hrx::hip::testing::HipDso library_;
  // One nonblocking stream reused throughout this process.
  hipStream_t stream_ = nullptr;
  // Cached public completion function included in the measured interval.
  decltype(&hipStreamSynchronize) synchronize_ = nullptr;
  // Device architecture used to select the build's matching code object.
  std::string architecture_;
  // Read benchmark code loaded once, only when a read row runs.
  hipModule_t module_ = nullptr;
  // Finite grid-stride reduction with one verified result per work-item.
  hipFunction_t read_function_ = nullptr;
  // Backend, actual GPU identity, and public wait policy.
  std::string label_;
};

TransferRuntime& Runtime() {
  static TransferRuntime runtime;
  return runtime;
}

enum TransferFlags : uint32_t {
  kDefault = 0,
  kRegistered = 1u << 0,
  kPitched = 1u << 1,
  kDeviceInterval = 1u << 2,
  kSynchronous = 1u << 3,
  kFreshHostMapping = 1u << 4,
  kHostInterval = 1u << 5,
};

enum class CopyStorage {
  kHost,
  kPageableUpload,
  kRegisteredUpload,
  kPageableDownload,
  kRegisteredDownload,
  kDevice,
};

// Each timed iteration submits a bounded chain to one stream. Submission rows
// pause timing only for the final wait; completion rows include that same wait.
// Storage, payload initialization, and full output verification are untimed.
template <bool complete>
void ContiguousCopy(benchmark::State& state, CopyStorage storage) {
  auto& runtime = Runtime();
  const auto copy =
      runtime.Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
  const auto blocking_copy = runtime.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
  const auto allocate = runtime.Resolve<decltype(&hipMalloc)>("hipMalloc");
  const auto release = runtime.Resolve<decltype(&hipFree)>("hipFree");
  const size_t copy_length = state.range(2);
  const size_t length = std::max(size_t{512}, copy_length + 32);
  constexpr size_t kSourceOffset = 7;
  constexpr size_t kTargetOffset = 11;
  std::vector<uint8_t> source(length);
  std::vector<uint8_t> target(length, kPadding);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i * 13 + i / 32);
  }
  const bool device_source = storage == CopyStorage::kDevice ||
                             storage == CopyStorage::kPageableDownload ||
                             storage == CopyStorage::kRegisteredDownload;
  const bool device_target = storage == CopyStorage::kDevice ||
                             storage == CopyStorage::kPageableUpload ||
                             storage == CopyStorage::kRegisteredUpload;
  void* source_base = source.data();
  void* target_base = target.data();
  void* registered = nullptr;
  if (storage == CopyStorage::kRegisteredUpload ||
      storage == CopyStorage::kRegisteredDownload) {
    Check(runtime.Resolve<decltype(&hipHostAlloc)>("hipHostMalloc")(
        &registered, length, hipHostMallocDefault));
    if (storage == CopyStorage::kRegisteredUpload) {
      source_base = registered;
      std::memcpy(source_base, source.data(), length);
    } else {
      target_base = registered;
      std::memcpy(target_base, target.data(), length);
    }
  }
  if (device_source) {
    Check(allocate(&source_base, length));
    Check(blocking_copy(source_base, source.data(), length,
                        hipMemcpyHostToDevice));
  }
  if (device_target) {
    Check(allocate(&target_base, length));
    Check(blocking_copy(target_base, target.data(), length,
                        hipMemcpyHostToDevice));
  }
  const hipMemcpyKind explicit_kind =
      device_source
          ? (device_target ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost)
          : (device_target ? hipMemcpyHostToDevice : hipMemcpyHostToHost);
  const hipMemcpyKind kind = state.range(1) ? hipMemcpyDefault : explicit_kind;
  const auto enqueue = [&] {
    for (int64_t i = 0; i < state.range(0); ++i) {
      Check(copy(static_cast<uint8_t*>(target_base) + kTargetOffset,
                 static_cast<uint8_t*>(source_base) + kSourceOffset,
                 copy_length, kind, runtime.stream()));
    }
  };
  enqueue();
  runtime.Wait();
  for (auto _ : state) {
    enqueue();
    if constexpr (!complete) {
      state.PauseTiming();
    }
    runtime.Wait();
    if constexpr (!complete) {
      state.ResumeTiming();
    }
  }
  if (device_target) {
    Check(blocking_copy(target.data(), target_base, length,
                        hipMemcpyDeviceToHost));
  } else if (registered) {
    std::memcpy(target.data(), target_base, length);
  }
  for (size_t i = 0; i < target.size(); ++i) {
    const uint8_t expected =
        i >= kTargetOffset && i < kTargetOffset + copy_length
            ? source[i - kTargetOffset + kSourceOffset]
            : kPadding;
    Require(target[i] == expected,
            "contiguous copy changed payload or padding");
  }
  if (device_source) {
    Check(release(source_base));
  }
  if (device_target) {
    Check(release(target_base));
  }
  if (registered) {
    Check(runtime.Resolve<decltype(&hipFreeHost)>("hipHostFree")(registered));
  }
  state.SetLabel(runtime.label());
  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * copy_length);
}

#define CONTIGUOUS_CASE(storage, complete, name)                      \
  BENCHMARK_CAPTURE(ContiguousCopy<complete>, storage##name,          \
                    CopyStorage::k##storage)                          \
      ->Name("HIP/Contiguous/" #storage "/" #name)                    \
      ->ArgNames({"copies", "default_kind", "bytes"})                 \
      ->ArgsProduct(                                                  \
          {{1, 32}, {0, 1}, {4, 16, 64, 256, 4096, 262144, 1048576}}) \
      ->UseRealTime()
#define CONTIGUOUS_STORAGE(storage)        \
  CONTIGUOUS_CASE(storage, false, Submit); \
  CONTIGUOUS_CASE(storage, true, Complete)
CONTIGUOUS_STORAGE(Host);
CONTIGUOUS_STORAGE(PageableUpload);
CONTIGUOUS_STORAGE(RegisteredUpload);
CONTIGUOUS_STORAGE(PageableDownload);
CONTIGUOUS_STORAGE(RegisteredDownload);
CONTIGUOUS_STORAGE(Device);
#undef CONTIGUOUS_STORAGE
#undef CONTIGUOUS_CASE

// One public operation retains the full rectangle or volume. Captured variants
// measure replay only; graph construction, allocation, and byte verification
// remain outside both the submit-only and completed-transfer intervals.
template <bool complete, bool fill, bool graph>
void GeometryTransfer(benchmark::State& state) {
  auto& runtime = Runtime();
  const auto copy = runtime.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
  const auto copy_2d =
      runtime.Resolve<decltype(&hipMemcpy2DAsync)>("hipMemcpy2DAsync");
  const auto copy_3d =
      runtime.Resolve<decltype(&hipMemcpy3DAsync)>("hipMemcpy3DAsync");
  const auto fill_2d =
      runtime.Resolve<decltype(&hipMemset2DAsync)>("hipMemset2DAsync");
  const auto fill_3d =
      runtime.Resolve<decltype(&hipMemset3DAsync)>("hipMemset3DAsync");
  const auto allocate = runtime.Resolve<decltype(&hipMalloc)>("hipMalloc");
  const auto release = runtime.Resolve<decltype(&hipFree)>("hipFree");
  const size_t width = state.range(0);
  const size_t height = state.range(1);
  const size_t depth = state.range(2);
  const int64_t operation_count = state.range(3);
  const size_t source_pitch = width + 64;
  const size_t target_pitch = width + 128;
  const size_t rows_per_slice = height + 1;
  const size_t source_slice = source_pitch * rows_per_slice;
  const size_t target_slice = target_pitch * rows_per_slice;
  constexpr size_t kSourceOffset = 7;
  constexpr size_t kTargetOffset = 11;
  std::vector<uint8_t> source(kSourceOffset + source_slice * depth + 16);
  std::vector<uint8_t> target(kTargetOffset + target_slice * depth + 16,
                              kPadding);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i * 13 + i / 32);
  }
  void* source_device = nullptr;
  void* target_device = nullptr;
  Check(allocate(&source_device, source.size()));
  Check(allocate(&target_device, target.size()));
  Check(
      copy(source_device, source.data(), source.size(), hipMemcpyHostToDevice));
  Check(
      copy(target_device, target.data(), target.size(), hipMemcpyHostToDevice));
  const hipPitchedPtr source_pointer = {
      static_cast<uint8_t*>(source_device) + kSourceOffset, source_pitch, width,
      rows_per_slice};
  const hipPitchedPtr target_pointer = {
      static_cast<uint8_t*>(target_device) + kTargetOffset, target_pitch, width,
      rows_per_slice};
  const hipMemcpy3DParms parameters = {
      .srcPtr = source_pointer,
      .dstPtr = target_pointer,
      .extent = {width, height, depth},
      .kind = hipMemcpyDeviceToDevice,
  };
  const auto issue = [&] {
    for (int64_t i = 0; i < operation_count; ++i) {
      if constexpr (fill) {
        if (depth == 1) {
          Check(fill_2d(target_pointer.ptr, target_pitch, kPayload, width,
                        height, runtime.stream()));
        } else {
          Check(fill_3d(target_pointer, kPayload, {width, height, depth},
                        runtime.stream()));
        }
      } else if (depth == 1) {
        Check(copy_2d(target_pointer.ptr, target_pitch, source_pointer.ptr,
                      source_pitch, width, height, hipMemcpyDeviceToDevice,
                      runtime.stream()));
      } else {
        Check(copy_3d(&parameters, runtime.stream()));
      }
    }
  };
  hipGraph_t graph_handle = nullptr;
  hipGraphExec_t executable = nullptr;
  decltype(&hipGraphLaunch) launch = nullptr;
  if constexpr (graph) {
    launch = runtime.Resolve<decltype(&hipGraphLaunch)>("hipGraphLaunch");
    Check(runtime.Resolve<decltype(&hipStreamBeginCapture)>(
        "hipStreamBeginCapture")(runtime.stream(), hipStreamCaptureModeGlobal));
    issue();
    Check(runtime.Resolve<decltype(&hipStreamEndCapture)>(
        "hipStreamEndCapture")(runtime.stream(), &graph_handle));
    Check(runtime.Resolve<decltype(&hipGraphInstantiateWithFlags)>(
        "hipGraphInstantiateWithFlags")(&executable, graph_handle, 0));
  }
  const auto enqueue = [&] {
    if constexpr (graph) {
      Check(launch(executable, runtime.stream()));
    } else {
      issue();
    }
  };
  enqueue();
  runtime.Wait();
  for (auto _ : state) {
    enqueue();
    if constexpr (!complete) {
      state.PauseTiming();
    }
    runtime.Wait();
    if constexpr (!complete) {
      state.ResumeTiming();
    }
  }
  Check(
      copy(target.data(), target_device, target.size(), hipMemcpyDeviceToHost));
  std::vector<uint8_t> expected(target.size(), kPadding);
  for (size_t slice = 0; slice < depth; ++slice) {
    for (size_t row = 0; row < height; ++row) {
      for (size_t column = 0; column < width; ++column) {
        expected[kTargetOffset + slice * target_slice + row * target_pitch +
                 column] = fill ? kPayload
                                : source[kSourceOffset + slice * source_slice +
                                         row * source_pitch + column];
      }
    }
  }
  Require(target == expected,
          "rectangular transfer changed payload or padding");
  if constexpr (graph) {
    Check(runtime.Resolve<decltype(&hipGraphExecDestroy)>(
        "hipGraphExecDestroy")(executable));
    Check(runtime.Resolve<decltype(&hipGraphDestroy)>("hipGraphDestroy")(
        graph_handle));
  }
  Check(release(target_device));
  Check(release(source_device));
  state.SetLabel(runtime.label());
  state.SetItemsProcessed(state.iterations() * operation_count);
  state.SetBytesProcessed(state.iterations() * operation_count * width *
                          height * depth);
}

#define GEOMETRY_CASE(name, complete, fill, graph, timing)      \
  BENCHMARK_TEMPLATE(GeometryTransfer, complete, fill, graph)   \
      ->Name("HIP/Geometry/" #name "/" #timing)                 \
      ->ArgNames({"row_bytes", "rows", "slices", "operations"}) \
      ->Args({64, 1, 1, 1})                                     \
      ->Args({64, 32, 1, 1})                                    \
      ->Args({4096, 128, 1, 1})                                 \
      ->Args({257, 17, 3, 1})                                   \
      ->Args({64, 1, 1, 32})                                    \
      ->Args({64, 32, 1, 32})                                   \
      ->UseRealTime()
#define GEOMETRY_VARIANT(name, fill, graph)        \
  GEOMETRY_CASE(name, false, fill, graph, Submit); \
  GEOMETRY_CASE(name, true, fill, graph, Complete)
GEOMETRY_VARIANT(Copy, false, false);
GEOMETRY_VARIANT(Fill, true, false);
GEOMETRY_VARIANT(GraphCopy, false, true);
GEOMETRY_VARIANT(GraphFill, true, true);
#undef GEOMETRY_VARIANT
#undef GEOMETRY_CASE

// Enclose one asynchronous transfer in device events. The interval includes
// any gap between commands; host_ns also includes event submission and the
// final stream wait, but excludes querying the completed event timestamps.
template <typename Enqueue>
void MeasureDeviceInterval(benchmark::State& state, Enqueue enqueue) {
  auto& runtime = Runtime();
  auto create = runtime.Resolve<decltype(&hipEventCreate)>("hipEventCreate");
  auto destroy = runtime.Resolve<decltype(&hipEventDestroy)>("hipEventDestroy");
  auto record = runtime.Resolve<decltype(&hipEventRecord)>("hipEventRecord");
  auto elapsed =
      runtime.Resolve<decltype(&hipEventElapsedTime)>("hipEventElapsedTime");
  hipEvent_t begin = nullptr, end = nullptr;
  Check(create(&begin));
  Check(create(&end));
  const auto transfer = [&] {
    Check(record(begin, runtime.stream()));
    enqueue();
    Check(record(end, runtime.stream()));
    runtime.Wait();
  };
  transfer();
  double host_nanoseconds = 0;
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    transfer();
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
  state.counters["host_ns"] =
      benchmark::Counter(host_nanoseconds, benchmark::Counter::kAvgIterations);
}

// Split ordinary copy completion into submission and waiting, without adding
// device events or changing the queue dependency sequence. Separate rows keep
// the three host clock reads out of the ordinary completed-transfer benchmark.
template <typename Enqueue>
void MeasureHostInterval(benchmark::State& state, Enqueue enqueue) {
  auto& runtime = Runtime();
  double submit_nanoseconds = 0;
  double wait_nanoseconds = 0;
  for (auto _ : state) {
    const auto begin = std::chrono::steady_clock::now();
    enqueue();
    const auto submitted = std::chrono::steady_clock::now();
    runtime.Wait();
    const auto completed = std::chrono::steady_clock::now();
    submit_nanoseconds +=
        std::chrono::duration<double, std::nano>(submitted - begin).count();
    wait_nanoseconds +=
        std::chrono::duration<double, std::nano>(completed - submitted).count();
  }
  state.counters["submit_ns"] = benchmark::Counter(
      submit_nanoseconds, benchmark::Counter::kAvgIterations);
  state.counters["wait_ns"] =
      benchmark::Counter(wait_nanoseconds, benchmark::Counter::kAvgIterations);
}

// Cover publication-window and bulk-queue boundaries on both sides. The row
// sizes divide each extent so pitched cases move the same useful byte count.
void HostTransferSizes(benchmark::internal::Benchmark* benchmark) {
  for (int64_t length :
       {4096, 16384, 65536, 131072, 258048, 262144, 266240, 524288, 1044480,
        1048576, 1052672, 2097152, 4194304, 8388608, 16777216, 67108864}) {
    benchmark->Arg(length);
  }
}

void Download(benchmark::State& state, uint32_t flags, size_t row_width = 4096,
              size_t host_count = 1) {
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
  const bool fresh_host_mapping = (flags & kFreshHostMapping) != 0;
  const auto allocate_host = [&]() -> uint8_t* {
    if (fresh_host_mapping) {
      void* mapping = mmap(nullptr, allocation_length, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      Require(mapping != MAP_FAILED, "host mapping failed");
      return static_cast<uint8_t*>(mapping);
    }
    return static_cast<uint8_t*>(
        std::aligned_alloc(4096, allocation_length * host_count));
  };
  auto* allocation = allocate_host();
  Require(allocation != nullptr, "host allocation failed");
  std::memset(allocation, kPadding, allocation_length * host_count);
  uint8_t* target = allocation;
  if (registered) {
    Check(runtime.Resolve<decltype(&hipHostRegister)>("hipHostRegister")(
        allocation, allocation_length * host_count, hipHostRegisterDefault));
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
  for (size_t i = 0; i < host_count; ++i) {
    target = allocation + i * allocation_length;
    transfer();
    runtime.Wait();
    verify();
    std::memset(target, kPadding, allocation_length);
  }
  target = allocation;
  if (fresh_host_mapping) {
    // Releasing the CPU mapping also removes runtime-retained GPU access.
    // Allocation, first touch, verification, and CPU unmap are outside copy
    // timing. Report unmap separately because a runtime may defer work to it.
    Require(munmap(allocation, allocation_length) == 0, "host unmap failed");
    double unmap_nanoseconds = 0;
    for (auto _ : state) {
      state.PauseTiming();
      target = allocate_host();
      std::memset(target, kPadding, allocation_length);
      state.ResumeTiming();
      transfer();
      runtime.Wait();
      state.PauseTiming();
      verify();
      const auto begin = std::chrono::steady_clock::now();
      Require(munmap(target, allocation_length) == 0, "host unmap failed");
      const auto end = std::chrono::steady_clock::now();
      unmap_nanoseconds +=
          std::chrono::duration<double, std::nano>(end - begin).count();
      state.ResumeTiming();
    }
    allocation = nullptr;
    state.counters["host_unmap_ns"] = benchmark::Counter(
        unmap_nanoseconds, benchmark::Counter::kAvgIterations);
  } else if (flags & kDeviceInterval) {
    MeasureDeviceInterval(state, transfer);
  } else if (flags & kHostInterval) {
    MeasureHostInterval(state, transfer);
  } else {
    size_t host_index = 0;
    for (auto _ : state) {
      target = allocation + host_index * allocation_length;
      transfer();
      runtime.Wait();
      host_index = (host_index + 1) % host_count;
    }
  }
  for (size_t i = 0;
       !fresh_host_mapping &&
       i < std::min(host_count, static_cast<size_t>(state.iterations()));
       ++i) {
    target = allocation + i * allocation_length;
    verify();
  }
  Check(runtime.Resolve<decltype(&hipFree)>("hipFree")(source));
  if (registered) {
    Check(runtime.Resolve<decltype(&hipHostUnregister)>("hipHostUnregister")(
        allocation));
  }
  std::free(allocation);
  state.SetLabel(runtime.label());
  state.SetBytesProcessed(state.iterations() * length);
  state.counters["width"] = width;
  state.counters["height"] = height;
  state.counters["source_pitch"] = source_pitch;
  state.counters["target_pitch"] = target_pitch;
  state.counters["host_buffers"] = host_count;
}

#define DOWNLOAD_CASE(name, flags)         \
  BENCHMARK_CAPTURE(Download, name, flags) \
      ->Name("HIP/Download/" #name)        \
      ->Apply(HostTransferSizes)           \
      ->MeasureProcessCPUTime()            \
      ->UseRealTime()
DOWNLOAD_CASE(PageableContiguous, kDefault);
DOWNLOAD_CASE(RegisteredContiguous, kRegistered);
DOWNLOAD_CASE(PageablePitched, kPitched);
DOWNLOAD_CASE(RegisteredPitched, kRegistered | kPitched);
#undef DOWNLOAD_CASE

BENCHMARK_CAPTURE(Download, PageableRotating, kDefault, 4096, 4)
    ->Name("HIP/Download/PageableRotating")
    ->Apply(HostTransferSizes)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(Download, PageableRotatingPitched, kPitched, 4096, 4)
    ->Name("HIP/Download/PageableRotatingPitched")
    ->Apply(HostTransferSizes)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_CAPTURE(Download, PageableFresh, kFreshHostMapping)
    ->Name("HIP/Download/PageableFresh")
    ->Apply(HostTransferSizes)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(Download, PageableFreshPitched, kFreshHostMapping | kPitched)
    ->Name("HIP/Download/PageableFreshPitched")
    ->Apply(HostTransferSizes)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// Sparse event pairs expose device-timeline gaps independently of host return
// and waiting. Intervals include any host gaps between device commands, and
// host_ns includes event enqueues and completion but not the elapsed-time
// query. Constant payload with varying row width separates byte and copy-count
// costs.
#define DOWNLOAD_INTERVAL_CASE(name, flags, width)                             \
  BENCHMARK_CAPTURE(Download, name##DeviceInterval, (flags) | kDeviceInterval, \
                    width)                                                     \
      ->Name("HIP/DownloadDeviceInterval/" #name)                              \
      ->RangeMultiplier(4)                                                     \
      ->Range(4096, 64 * 1024 * 1024)                                          \
      ->MeasureProcessCPUTime()                                                \
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

#define DOWNLOAD_HOST_INTERVAL_CASE(name, flags)                           \
  BENCHMARK_CAPTURE(Download, name##HostInterval, (flags) | kHostInterval) \
      ->Name("HIP/DownloadHostInterval/" #name)                            \
      ->Apply(HostTransferSizes)                                           \
      ->MeasureProcessCPUTime()                                            \
      ->UseRealTime()
DOWNLOAD_HOST_INTERVAL_CASE(RegisteredContiguous, kRegistered);
DOWNLOAD_HOST_INTERVAL_CASE(RegisteredPitched, kRegistered | kPitched);
#undef DOWNLOAD_HOST_INTERVAL_CASE

// A blocking HIP completion owns the borrowed upload bytes until return. Use
// persistent host ranges so steady-state measurements include no application
// allocation, registration, or validation work. Rotating ranges have distinct
// contents, allowing readback to identify the last submitted source.
void Upload(benchmark::State& state, uint32_t flags, size_t host_count = 1,
            size_t row_width = 4096) {
  auto& runtime = Runtime();
  const bool registered = (flags & kRegistered) != 0;
  const bool pitched = (flags & kPitched) != 0;
  const size_t length = state.range(0);
  const size_t width = pitched ? std::min(row_width, length) : length;
  const size_t height = length / width;
  const size_t source_pitch = width + (pitched ? 64 : 0);
  const size_t target_pitch = width + (pitched ? 128 : 0);
  const size_t target_length = target_pitch * height;
  const size_t allocation_length =
      ((target_length + 4095) & ~size_t{4095}) + 4096;
  auto* allocation = static_cast<uint8_t*>(
      std::aligned_alloc(4096, allocation_length * host_count));
  Require(allocation != nullptr, "upload host allocation failed");
  for (size_t i = 0; i < host_count; ++i) {
    for (size_t row = 0; row < height; ++row) {
      for (size_t column = 0; column < width; ++column) {
        allocation[i * allocation_length + row * source_pitch + column] =
            (i + row * width + column) % 251;
      }
    }
  }
  uint8_t* source = allocation;
  if (registered) {
    Check(runtime.Resolve<decltype(&hipHostRegister)>("hipHostRegister")(
        allocation, allocation_length * host_count, hipHostRegisterDefault));
  }
  hipDeviceptr_t target = nullptr;
  Check(runtime.Resolve<decltype(&hipMalloc)>("hipMalloc")(&target,
                                                           allocation_length));
  Check(runtime.Resolve<decltype(&hipMemsetAsync)>("hipMemsetAsync")(
      target, kPadding, allocation_length, runtime.stream()));
  runtime.Wait();
  auto copy = runtime.Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
  auto copy_2d =
      runtime.Resolve<decltype(&hipMemcpy2DAsync)>("hipMemcpy2DAsync");
  const auto copy_synchronous =
      runtime.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
  const auto enqueue = [&] {
    if ((flags & kSynchronous) != 0) {
      Check(copy_synchronous(target, source, length, hipMemcpyHostToDevice));
    } else if (pitched) {
      Check(copy_2d(target, target_pitch, source, source_pitch, width, height,
                    hipMemcpyHostToDevice, runtime.stream()));
    } else {
      Check(copy(target, source, length, hipMemcpyHostToDevice,
                 runtime.stream()));
    }
  };
  const auto transfer = [&] {
    enqueue();
    if (!(flags & kSynchronous)) {
      runtime.Wait();
    }
  };
  transfer();
  if (flags & kDeviceInterval) {
    MeasureDeviceInterval(state, enqueue);
  } else if (flags & kHostInterval) {
    MeasureHostInterval(state, enqueue);
  } else {
    size_t host_index = 0;
    for (auto _ : state) {
      source = allocation + host_index * allocation_length;
      transfer();
      host_index = (host_index + 1) % host_count;
    }
  }

  // Reuse the input allocation for readback after all measured uploads finish.
  // Both the varying payload and a full guard page come from device memory.
  source = allocation;
  std::memset(source, 0, allocation_length);
  Check(copy(source, target, allocation_length, hipMemcpyDeviceToHost,
             runtime.stream()));
  runtime.Wait();
  const size_t last_host_index = (state.iterations() - 1) % host_count;
  for (size_t row = 0; row < height; ++row) {
    for (size_t column = 0; column < width; ++column) {
      Require(source[row * target_pitch + column] ==
                  (last_host_index + row * width + column) % 251,
              "upload payload mismatch");
    }
    for (size_t column = width; column < target_pitch; ++column) {
      Require(source[row * target_pitch + column] == kPadding,
              "upload overwrote row padding");
    }
  }
  for (size_t i = target_length; i < allocation_length; ++i) {
    Require(source[i] == kPadding, "upload overwrote its guard");
  }
  Check(runtime.Resolve<decltype(&hipFree)>("hipFree")(target));
  if (registered) {
    Check(runtime.Resolve<decltype(&hipHostUnregister)>("hipHostUnregister")(
        source));
  }
  std::free(source);
  state.SetLabel(runtime.label());
  state.SetBytesProcessed(state.iterations() * length);
  state.counters["width"] = width;
  state.counters["height"] = height;
  state.counters["source_pitch"] = source_pitch;
  state.counters["target_pitch"] = target_pitch;
  state.counters["host_buffers"] = host_count;
}

#define UPLOAD_CASE(name, flags)         \
  BENCHMARK_CAPTURE(Upload, name, flags) \
      ->Name("HIP/Upload/" #name)        \
      ->Apply(HostTransferSizes)         \
      ->Arg(1024 * 1024 + 123)           \
      ->MeasureProcessCPUTime()          \
      ->UseRealTime()
UPLOAD_CASE(Pageable, kDefault);
UPLOAD_CASE(Registered, kRegistered);
#undef UPLOAD_CASE

BENCHMARK_CAPTURE(Upload, PageableRotating, kDefault, 4)
    ->Name("HIP/Upload/PageableRotating")
    ->Apply(HostTransferSizes)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(Upload, PageableRotatingPitched, kPitched, 4)
    ->Name("HIP/Upload/PageableRotatingPitched")
    ->Apply(HostTransferSizes)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_CAPTURE(Upload, SynchronousPageable, kSynchronous)
    ->Name("HIP/Upload/SynchronousPageable")
    ->Arg(4 * 1024)
    ->Arg(256 * 1024)
    ->RangeMultiplier(2)
    ->Range(1024 * 1024, 64 * 1024 * 1024)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

#define UPLOAD_PITCHED_CASE(name, flags)              \
  BENCHMARK_CAPTURE(Upload, name, (flags) | kPitched) \
      ->Name("HIP/Upload/" #name)                     \
      ->Apply(HostTransferSizes)                      \
      ->MeasureProcessCPUTime()                       \
      ->UseRealTime()
UPLOAD_PITCHED_CASE(PageablePitched, kDefault);
UPLOAD_PITCHED_CASE(RegisteredPitched, kRegistered);
#undef UPLOAD_PITCHED_CASE

#define UPLOAD_INTERVAL_CASE(name, flags, width)                             \
  BENCHMARK_CAPTURE(Upload, name##DeviceInterval, (flags) | kDeviceInterval, \
                    1, width)                                                \
      ->Name("HIP/UploadDeviceInterval/" #name)                              \
      ->RangeMultiplier(4)                                                   \
      ->Range(4096, 64 * 1024 * 1024)                                        \
      ->MeasureProcessCPUTime()                                              \
      ->UseManualTime()
UPLOAD_INTERVAL_CASE(RegisteredContiguous, kRegistered, 0);
UPLOAD_INTERVAL_CASE(RegisteredPitched4K, kRegistered | kPitched, 4096);
UPLOAD_INTERVAL_CASE(RegisteredPitched64K, kRegistered | kPitched, 65536);
UPLOAD_INTERVAL_CASE(RegisteredPitched1M, kRegistered | kPitched, 1024 * 1024);
UPLOAD_INTERVAL_CASE(RegisteredPitched4M, kRegistered | kPitched,
                     4 * 1024 * 1024);
#undef UPLOAD_INTERVAL_CASE

#define UPLOAD_HOST_INTERVAL_CASE(name, flags)                           \
  BENCHMARK_CAPTURE(Upload, name##HostInterval, (flags) | kHostInterval) \
      ->Name("HIP/UploadHostInterval/" #name)                            \
      ->Apply(HostTransferSizes)                                         \
      ->MeasureProcessCPUTime()                                          \
      ->UseRealTime()
UPLOAD_HOST_INTERVAL_CASE(RegisteredContiguous, kRegistered);
UPLOAD_HOST_INTERVAL_CASE(RegisteredPitched, kRegistered | kPitched);
#undef UPLOAD_HOST_INTERVAL_CASE

// Reusing one source can keep its CPU cache lines resident for every GPU read.
// Rotating explicitly registered sources also covers uploads from a working
// set larger than the cache, without including registration in copy timing.
#define REGISTERED_ROTATING_UPLOAD_CASE(name, flags, count)               \
  BENCHMARK_CAPTURE(Upload, name##Rotating##count, kRegistered | (flags), \
                    count)                                                \
      ->Name("HIP/Upload/Registered" #name "Rotating" #count)             \
      ->Arg(4096)                                                         \
      ->Arg(1024 * 1024)                                                  \
      ->Arg(4 * 1024 * 1024)                                              \
      ->MeasureProcessCPUTime()                                           \
      ->UseRealTime()
REGISTERED_ROTATING_UPLOAD_CASE(Contiguous, kDefault, 4);
REGISTERED_ROTATING_UPLOAD_CASE(Contiguous, kDefault, 16);
REGISTERED_ROTATING_UPLOAD_CASE(Pitched, kPitched, 4);
REGISTERED_ROTATING_UPLOAD_CASE(Pitched, kPitched, 16);
#undef REGISTERED_ROTATING_UPLOAD_CASE

// Separate the host cost of preparing/uploading the source from the GPU's
// subsequent read cost. Repeated reads expose cache-resident throughput;
// bytes_processed includes every pass and is not physical DRAM bandwidth.
// Upload/fill occurs once per row, before the first read and all timed reads.
void ReadAfterUpload(benchmark::State& state, bool device_fill) {
  auto& runtime = Runtime();
  const auto function = runtime.read_function();
  const auto launch = runtime.Resolve<decltype(&hipModuleLaunchKernel)>(
      "hipModuleLaunchKernel");
  const auto copy = runtime.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
  const auto allocate = runtime.Resolve<decltype(&hipMalloc)>("hipMalloc");
  const auto release = runtime.Resolve<decltype(&hipFree)>("hipFree");
  const auto record =
      runtime.Resolve<decltype(&hipEventRecord)>("hipEventRecord");
  const auto synchronize =
      runtime.Resolve<decltype(&hipEventSynchronize)>("hipEventSynchronize");
  const auto elapsed =
      runtime.Resolve<decltype(&hipEventElapsedTime)>("hipEventElapsedTime");
  const auto create =
      runtime.Resolve<decltype(&hipEventCreate)>("hipEventCreate");
  const auto destroy =
      runtime.Resolve<decltype(&hipEventDestroy)>("hipEventDestroy");
  const size_t length = state.range(0);
  uint64_t element_count = length / 16;
  uint32_t passes = length <= 32 * 1024 * 1024
                        ? std::max(size_t{4}, (256 * 1024 * 1024) / length)
                        : 2;
  constexpr uint32_t kGroupCount = 640;
  constexpr uint32_t kGroupSize = 256;
  constexpr uint32_t kWorkitemCount = kGroupCount * kGroupSize;
  hipDeviceptr_t input = nullptr;
  hipDeviceptr_t output = nullptr;
  Check(allocate(&input, length));
  Check(allocate(&output, kWorkitemCount * sizeof(uint32_t)));
  hipEvent_t begin = nullptr;
  hipEvent_t end = nullptr;
  Check(create(&begin));
  Check(create(&end));
  const auto prepare_start = std::chrono::steady_clock::now();
  std::vector<uint8_t> source(device_fill ? 0 : length, 1);
  const auto upload_start = std::chrono::steady_clock::now();
  if (device_fill) {
    Check(runtime.Resolve<decltype(&hipMemsetAsync)>("hipMemsetAsync")(
        input, 1, length, runtime.stream()));
    runtime.Wait();
  } else {
    Check(copy(input, source.data(), length, hipMemcpyHostToDevice));
  }
  const auto upload_end = std::chrono::steady_clock::now();
  void* arguments[] = {&input, &element_count, &passes, &output};
  const auto read = [&] {
    Check(record(begin, runtime.stream()));
    Check(launch(function, kGroupCount, 1, 1, kGroupSize, 1, 1, 0,
                 runtime.stream(), arguments, nullptr));
    Check(record(end, runtime.stream()));
    Check(synchronize(end));
    float milliseconds = 0;
    Check(elapsed(&milliseconds, begin, end));
    Require(milliseconds > 0, "device read timestamps did not advance");
    return static_cast<double>(milliseconds) * 1e-3;
  };
  const double first_read_seconds = read();
  for (auto _ : state) {
    state.SetIterationTime(read());
  }

  // Every lane writes a result, including lanes with no input elements. With
  // 0x01 source bytes, each uint4 sums to 0x04040404 and XOR repetition has a
  // closed-form expected value independent of the GPU implementation.
  std::vector<uint32_t> actual(kWorkitemCount);
  Check(copy(actual.data(), output, actual.size() * sizeof(uint32_t),
             hipMemcpyDeviceToHost));
  uint32_t repeated_value = 0;
  for (uint32_t pass = 0; pass < passes; ++pass) {
    repeated_value ^= 0x04040404u + pass;
  }
  for (uint32_t i = 0; i < kWorkitemCount; ++i) {
    const uint64_t count =
        i < element_count ? (element_count - 1 - i) / kWorkitemCount + 1 : 0;
    Require(actual[i] == ((count & 1) ? repeated_value : 0),
            "GPU read checksum mismatch");
  }
  // Check the entire uploaded payload too: reductions alone can alias errors.
  if (!device_fill) {
    Check(copy(source.data(), input, length, hipMemcpyDeviceToHost));
    Require(std::all_of(source.begin(), source.end(),
                        [](uint8_t value) { return value == 1; }),
            "read source payload mismatch");
  }
  Check(destroy(end));
  Check(destroy(begin));
  Check(release(output));
  Check(release(input));
  state.SetLabel(runtime.label());
  state.SetBytesProcessed(state.iterations() * length * passes);
  state.counters["passes"] = passes;
  state.counters["source_prepare_ns"] =
      std::chrono::duration<double, std::nano>(upload_start - prepare_start)
          .count();
  state.counters["initialize_ns"] =
      std::chrono::duration<double, std::nano>(upload_end - upload_start)
          .count();
  state.counters["first_read_ns"] = first_read_seconds * 1e9;
}

#define READ_CASE(name, fill)                    \
  BENCHMARK_CAPTURE(ReadAfterUpload, name, fill) \
      ->Name("HIP/ReadAfterUpload/" #name)       \
      ->Arg(256 * 1024)                          \
      ->RangeMultiplier(2)                       \
      ->Range(1024 * 1024, 64 * 1024 * 1024)     \
      ->Arg(256 * 1024 * 1024)                   \
      ->UseManualTime()
READ_CASE(SynchronousPageable, false);
READ_CASE(DeviceFill, true);
#undef READ_CASE

}  // namespace
