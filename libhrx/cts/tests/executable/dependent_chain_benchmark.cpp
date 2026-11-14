// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "binding/hip/api.h"
#include "hrx_loader.hpp"
#include "hrx_runtime.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"

IREE_FLAG(string, chain_runtime, "hrx", "Runtime: hrx or hip.");
IREE_FLAG(string, chain_device, "amdgpu://0", "Exact HRX device URI.");
IREE_FLAG(string, chain_hip_library, "", "Exact HIP implementation DSO.");
IREE_FLAG(int32_t, chain_hip_device, 0, "HIP device ordinal.");
IREE_FLAG(bool, chain_active_wait, true,
          "Poll the public stream query API before waiting for completion.");

namespace {

enum class Submission { kFlushEach, kFlushEnd, kGraph };

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

void Check(hrx_status_t status) {
  if (!hrx_status_is_ok(status)) {
    char* message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    std::fprintf(stderr, "%s\n", message);
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    std::abort();
  }
}

void Check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP operation failed: %d\n", int(status));
    std::abort();
  }
}

iree::hal::cts::ExecutableTarget FindImage(const char* architecture) {
  iree_hal_amd_gpu_target_identity_t identity;
  iree_status_t status = iree_hal_amd_gpu_target_identity_parse_artifact_key(
      iree_make_cstring_view(architecture), &identity);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    std::abort();
  }
  const auto images =
      iree::hal::cts::CtsRegistry::ListExecutableTargets("hrx_dependent_chain");
  const auto* image =
      iree::hal::amd::gpu::cts::SelectExecutableTarget(images, identity);
  Require(image != nullptr, "no configured Loom image for selected GPU");
  return *image;
}

// The immutable source and two alternating destinations stay on the device.
// Every invocation repeats the same recurrence; verification transfers only
// the final output, outside timing. Integer arithmetic has exact wraparound.
struct Workload {
  explicit Workload(uint32_t element_count, uint32_t dispatch_count)
      : count(element_count),
        depth(dispatch_count),
        input(count),
        expected(count) {
    Require(count >= 256 && count <= 2097152 && !(count & (count - 1)),
            "element count must be a power of two between 256 and 2097152");
    for (uint32_t i = 0; i < count; ++i) {
      input[i] = i * 747796405u + 2891336453u;
    }
    expected = input;
    std::vector<uint32_t> next(count);
    for (uint32_t step = 1; step <= depth; ++step) {
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t adjacent = expected[(i + 257) & (count - 1)];
        next[i] = (expected[i] ^ ((adjacent << 1) | (adjacent >> 31))) + step;
      }
      expected.swap(next);
    }
  }

  size_t bytes() const { return size_t(count) * sizeof(uint32_t); }
  uint32_t source(uint32_t step) const {
    return step ? 1 + ((step - 1) & 1) : 0;
  }
  uint32_t target(uint32_t step) const { return 1 + (step & 1); }

  // Elements in each equally sized buffer.
  uint32_t count;
  // Ordered dispatches in one completed chain.
  uint32_t depth;
  // Immutable source uploaded before timing.
  std::vector<uint32_t> input;
  // Independent CPU recurrence for the complete final output.
  std::vector<uint32_t> expected;
};

class HrxRuntime {
 public:
  HrxRuntime() {
    hrx_runtime_t runtime = nullptr;
    Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime));
    const hrx_device_create_params_t params = {
        .uri = FLAG_chain_device,
        .completion_wait_flags =
            FLAG_chain_active_wait ? HRX_WAIT_FLAG_ACTIVE : HRX_WAIT_FLAG_NONE,
    };
    const auto status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    Check(status);
    device_ = hrx_device_group_get_device(group_, 0);
    for (uint32_t i = 0; i < hrx_device_queue_family_count(device_); ++i) {
      hrx_queue_family_info_t family;
      Check(hrx_device_query_queue_family(device_, i, &family));
      const uint32_t roles = HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER;
      if (family.provisioned_queue_count && (family.roles & roles) == roles) {
        queue_ = hrx_device_get_queue(device_, i, 0);
        break;
      }
    }
    Require(queue_ != nullptr, "no combined dispatch/transfer queue");
    Check(hrx_stream_create(queue_, 0, &stream_));
    char architecture[64] = {};
    Check(hrx_device_get_property(device_, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                  architecture, sizeof(architecture)));
    const auto image = FindImage(architecture);
    const auto data = image.data_fn(IREE_SV("dependent_chain.bin"));
    Check(hrx_executable_load_data(device_, data.data, data.data_length,
                                   image.family, image.target_key, &module_));
    Check(hrx_executable_lookup_export_by_name(module_, "dependent_chain",
                                               &ordinal_));
    hrx_executable_export_info_t info;
    Check(hrx_executable_export_info(module_, ordinal_, &info));
    Require(info.binding_count == 2 && info.constant_byte_length == 8,
            "unexpected dependent-chain kernel ABI");
    label_ = std::string("HRX ") + architecture + " " + FLAG_chain_device;
  }

  ~HrxRuntime() {
    hrx_stream_release(stream_);
    hrx_executable_release(module_);
    hrx_device_group_release(group_);
  }

  void Prepare(const Workload& workload, Submission submission) {
    const hrx_buffer_params_t params = {
        .type = HRX_MEMORY_TYPE_DEVICE_LOCAL,
        .access = HRX_MEMORY_ACCESS_ALL,
        .usage = HRX_BUFFER_USAGE_DEFAULT,
    };
    for (auto& buffer : buffers_) {
      Check(hrx_allocator_allocate_buffer(hrx_device_allocator(device_), params,
                                          workload.bytes(), &buffer));
    }
    Check(hrx_synchronous_h2d(queue_, workload.input.data(), buffers_[0], 0,
                              workload.bytes()));
    if (submission == Submission::kGraph) {
      Check(hrx_graph_create(device_, 0, &graph_));
      hrx_graph_node_t previous = nullptr;
      for (uint32_t step = 0; step < workload.depth; ++step) {
        const auto bindings = Bindings(workload, step);
        const std::array<uint32_t, 2> constants = {workload.count, step + 1};
        const hrx_graph_kernel_node_attrs_t attrs = {
            module_,           ordinal_,
            Config(workload),  constants.data(),
            sizeof(constants), bindings.data(),
            bindings.size(),   HRX_DISPATCH_FLAG_NONE,
        };
        hrx_graph_node_t node = nullptr;
        Check(hrx_graph_add_kernel_node(graph_, previous ? &previous : nullptr,
                                        previous ? 1 : 0, &attrs, &node));
        previous = node;
      }
      Check(hrx_graph_instantiate(graph_, queue_, 0, &graph_exec_));
    }
  }

  void Launch(const Workload& workload, Submission submission) {
    if (submission == Submission::kGraph) {
      Check(hrx_graph_exec_launch(graph_exec_, stream_));
      return;
    }
    const auto config = Config(workload);
    for (uint32_t step = 0; step < workload.depth; ++step) {
      const auto bindings = Bindings(workload, step);
      const std::array<uint32_t, 2> constants = {workload.count, step + 1};
      Check(hrx_stream_dispatch(stream_, module_, ordinal_, &config,
                                constants.data(), sizeof(constants),
                                bindings.data(), bindings.size(),
                                HRX_DISPATCH_FLAG_NONE));
      if (submission == Submission::kFlushEach) {
        Check(hrx_stream_flush(stream_));
      }
    }
    Check(hrx_stream_flush(stream_));
  }

  void Wait() {
    if (FLAG_chain_active_wait) {
      bool complete = false;
      do {
        Check(hrx_stream_query(stream_, &complete));
      } while (!complete);
    }
    Check(hrx_stream_wait(stream_));
  }
  void Verify(const Workload& workload) {
    std::vector<uint32_t> actual(workload.count);
    Check(hrx_synchronous_d2h(queue_,
                              buffers_[workload.target(workload.depth - 1)], 0,
                              actual.data(), workload.bytes()));
    Require(actual == workload.expected, "HRX dependent-chain output mismatch");
  }
  void Reset() {
    hrx_graph_exec_release(graph_exec_);
    hrx_graph_release(graph_);
    graph_exec_ = nullptr;
    graph_ = nullptr;
    for (auto& buffer : buffers_) {
      hrx_buffer_release(buffer);
      buffer = nullptr;
    }
  }
  const std::string& label() const { return label_; }

 private:
  static hrx_dispatch_config_t Config(const Workload& workload) {
    return {{workload.count / 256, 1, 1}, {256, 1, 1}, 0};
  }
  std::array<hrx_buffer_ref_t, 2> Bindings(const Workload& workload,
                                           uint32_t step) const {
    return {{{buffers_[workload.source(step)], 0, workload.bytes()},
             {buffers_[workload.target(step)], 0, workload.bytes()}}};
  }
  // Explicit runtime owner retained for the process-wide benchmark fixture.
  hrx_device_group_t group_ = nullptr;
  // Borrowed group device used to allocate the resident working set.
  hrx_device_t device_ = nullptr;
  // Borrowed provisioned queue used for both setup transfers and the chain.
  hrx_queue_t queue_ = nullptr;
  // Ordered stream reused across rows without cycling native queues.
  hrx_stream_t stream_ = nullptr;
  // Loom code object shared with the HIP comparison.
  hrx_executable_t module_ = nullptr;
  // Reflected dependent-chain export index.
  uint32_t ordinal_ = 0;
  // Immutable input followed by two alternating device-local outputs.
  std::array<hrx_buffer_t, 3> buffers_ = {};
  // Row-local immutable dependency graph.
  hrx_graph_t graph_ = nullptr;
  // Row-local executable graph retained through final completion.
  hrx_graph_exec_t graph_exec_ = nullptr;
  // Reported architecture and selected device URI.
  std::string label_;
};

class HipRuntime {
 public:
  HipRuntime() {
    Require(FLAG_chain_hip_library[0], "--chain_hip_library is required");
    library_.load(FLAG_chain_hip_library);
    Check(Resolve<decltype(&hipInit)>("hipInit")(0));
    Check(Resolve<decltype(&hipSetDevice)>("hipSetDevice")(
        FLAG_chain_hip_device));
    Check(Resolve<decltype(&hipSetDeviceFlags)>("hipSetDeviceFlags")(
        FLAG_chain_active_wait ? hipDeviceScheduleSpin
                               : hipDeviceScheduleBlockingSync));
    hipDeviceProp_t properties = {};
    Check(Resolve<decltype(&hipGetDevicePropertiesR0600)>(
        "hipGetDevicePropertiesR0600")(&properties, FLAG_chain_hip_device));
    char pci_address[32] = {};
    Check(Resolve<decltype(&hipDeviceGetPCIBusId)>("hipDeviceGetPCIBusId")(
        pci_address, sizeof(pci_address), FLAG_chain_hip_device));
    const auto image = FindImage(properties.gcnArchName);
    const auto data = image.data_fn(IREE_SV("dependent_chain.bin"));
    Check(Resolve<decltype(&hipModuleLoadData)>("hipModuleLoadData")(
        &module_, data.data));
    Check(Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction")(
        &function_, module_, "dependent_chain"));
    Check(Resolve<decltype(&hipStreamCreateWithFlags)>(
        "hipStreamCreateWithFlags")(&stream_, hipStreamNonBlocking));
    dispatch_ =
        Resolve<decltype(&hipModuleLaunchKernel)>("hipModuleLaunchKernel");
    wait_ = Resolve<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
    query_ = Resolve<decltype(&hipStreamQuery)>("hipStreamQuery");
    replay_ = Resolve<decltype(&hipGraphLaunch)>("hipGraphLaunch");
    label_ = std::string("HIP ") + properties.gcnArchName + " " + pci_address;
  }

  ~HipRuntime() {
    Check(Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy")(stream_));
    Check(Resolve<decltype(&hipModuleUnload)>("hipModuleUnload")(module_));
  }

  void Prepare(const Workload& workload, Submission submission) {
    for (auto& buffer : buffers_) {
      Check(Resolve<decltype(&hipMalloc)>("hipMalloc")(&buffer,
                                                       workload.bytes()));
    }
    Check(Resolve<decltype(&hipMemcpy)>("hipMemcpy")(
        buffers_[0], workload.input.data(), workload.bytes(),
        hipMemcpyHostToDevice));
    if (submission == Submission::kGraph) {
      Check(Resolve<decltype(&hipStreamBeginCapture)>("hipStreamBeginCapture")(
          stream_, hipStreamCaptureModeGlobal));
      Dispatch(workload);
      Check(Resolve<decltype(&hipStreamEndCapture)>("hipStreamEndCapture")(
          stream_, &graph_));
      Check(Resolve<decltype(&hipGraphInstantiate)>("hipGraphInstantiate")(
          &graph_exec_, graph_, nullptr, nullptr, 0));
    }
  }

  void Launch(const Workload& workload, Submission submission) {
    if (submission == Submission::kGraph) {
      Check(replay_(graph_exec_, stream_));
    } else {
      // HIP eagerly submits ordinary launches; both HRX flush schedules compare
      // with this same ordered stream program, without an added host wait.
      Dispatch(workload);
    }
  }
  void Wait() {
    if (FLAG_chain_active_wait) {
      hipError_t status;
      do {
        status = query_(stream_);
      } while (status == hipErrorNotReady);
      Check(status);
    }
    Check(wait_(stream_));
  }
  void Verify(const Workload& workload) {
    std::vector<uint32_t> actual(workload.count);
    Check(Resolve<decltype(&hipMemcpy)>("hipMemcpy")(
        actual.data(), buffers_[workload.target(workload.depth - 1)],
        workload.bytes(), hipMemcpyDeviceToHost));
    Require(actual == workload.expected, "HIP dependent-chain output mismatch");
  }
  void Reset() {
    if (graph_exec_) {
      Check(Resolve<decltype(&hipGraphExecDestroy)>("hipGraphExecDestroy")(
          graph_exec_));
      Check(Resolve<decltype(&hipGraphDestroy)>("hipGraphDestroy")(graph_));
      graph_exec_ = nullptr;
      graph_ = nullptr;
    }
    for (auto& buffer : buffers_) {
      Check(Resolve<decltype(&hipFree)>("hipFree")(buffer));
      buffer = nullptr;
    }
  }
  const std::string& label() const { return label_; }

 private:
  template <typename T>
  T Resolve(const char* name) {
    return reinterpret_cast<T>(library_.loadSymbol(name));
  }
  void Dispatch(const Workload& workload) {
    for (uint32_t step = 0; step < workload.depth; ++step) {
      void* source = buffers_[workload.source(step)];
      void* target = buffers_[workload.target(step)];
      uint32_t count = workload.count;
      uint32_t increment = step + 1;
      void* arguments[] = {&source, &target, &count, &increment};
      Check(dispatch_(function_, count / 256, 1, 1, 256, 1, 1, 0, stream_,
                      arguments, nullptr));
    }
  }
  // Exact selected implementation; no ambient HIP symbol resolution.
  HrxDynamicLibrary library_;
  // Same Loom code object as the HRX measurement.
  hipModule_t module_ = nullptr;
  // Reflected dependent-chain export.
  hipFunction_t function_ = nullptr;
  // Ordered nonblocking stream with active completion observation.
  hipStream_t stream_ = nullptr;
  // Immutable input followed by two alternating device-local outputs.
  std::array<void*, 3> buffers_ = {};
  // Row-local captured dependency chain.
  hipGraph_t graph_ = nullptr;
  // Row-local replay executable retained through final completion.
  hipGraphExec_t graph_exec_ = nullptr;
  // Resolved dispatch entry point, outside the timed lookup path.
  decltype(&hipModuleLaunchKernel) dispatch_ = nullptr;
  // Resolved completion entry point.
  decltype(&hipStreamSynchronize) wait_ = nullptr;
  // Resolved public completion query used by the active-wait rows.
  decltype(&hipStreamQuery) query_ = nullptr;
  // Resolved graph replay entry point.
  decltype(&hipGraphLaunch) replay_ = nullptr;
  // Reported architecture and physical PCI identity.
  std::string label_;
};

template <typename Runtime>
Runtime& GetRuntime() {
  static Runtime runtime;
  return runtime;
}

template <typename Runtime, Submission Mode, bool IncludeCompletion>
void Run(benchmark::State& state) {
  auto& runtime = GetRuntime<Runtime>();
  const Workload workload(state.range(0), state.range(1));
  runtime.Prepare(workload, Mode);
  for (int i = 0; i < 8; ++i) {
    runtime.Launch(workload, Mode);
    runtime.Wait();
  }
  runtime.Verify(workload);
  state.SetLabel(
      runtime.label() +
      (FLAG_chain_active_wait ? " host_wait=query" : " host_wait=blocking") +
      " memory=device-local");
  for (auto _ : state) {
    runtime.Launch(workload, Mode);
    if constexpr (!IncludeCompletion) {
      state.PauseTiming();
    }
    runtime.Wait();
    if constexpr (!IncludeCompletion) {
      state.ResumeTiming();
    }
  }
  runtime.Verify(workload);
  runtime.Reset();
  state.counters["dispatches"] = workload.depth;
  state.counters["resident_bytes"] = 3 * workload.bytes();
  // Logical shader accesses, not a claim about measured memory-bus traffic.
  state.counters["logical_bytes"] = 3 * workload.bytes() * workload.depth;
  state.SetItemsProcessed(state.iterations() * workload.depth);
}

template <Submission Mode, bool IncludeCompletion>
void DependentChain(benchmark::State& state) {
  const std::string runtime = FLAG_chain_runtime;
  if (runtime == "hrx") {
    Run<HrxRuntime, Mode, IncludeCompletion>(state);
  } else {
    Require(runtime == "hip", "unknown --chain_runtime");
    Run<HipRuntime, Mode, IncludeCompletion>(state);
  }
}

void Cases(benchmark::Benchmark* benchmark) {
  for (int64_t count : {4096, 131072, 262144, 1048576, 2097152}) {
    for (int64_t depth : {8, 1, 32, 33, 64}) {
      benchmark->Args({count, depth});
    }
  }
  benchmark->ArgNames({"elements", "dispatches"});
  benchmark->UseRealTime()->Unit(benchmark::kMicrosecond);
}

BENCHMARK_TEMPLATE(DependentChain, Submission::kFlushEach, true)
    ->Name("DependentChain/StreamFlushEach")
    ->Apply(Cases);
BENCHMARK_TEMPLATE(DependentChain, Submission::kFlushEnd, true)
    ->Name("DependentChain/StreamFlushEnd")
    ->Apply(Cases);
BENCHMARK_TEMPLATE(DependentChain, Submission::kGraph, true)
    ->Name("DependentChain/GraphReplay")
    ->Apply(Cases);

// Submission rows exclude the final completion wait. Each iteration retires
// before the next begins, so memory reuse and queue occupancy match the
// corresponding completed-chain row. Fixed iteration counts bound real work.
BENCHMARK_TEMPLATE(DependentChain, Submission::kFlushEach, false)
    ->Name("DependentChainSubmit/StreamFlushEach")
    ->Apply(Cases);
BENCHMARK_TEMPLATE(DependentChain, Submission::kFlushEnd, false)
    ->Name("DependentChainSubmit/StreamFlushEnd")
    ->Apply(Cases);
BENCHMARK_TEMPLATE(DependentChain, Submission::kGraph, false)
    ->Name("DependentChainSubmit/GraphReplay")
    ->Apply(Cases);

}  // namespace
