// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "binding/hip/api.h"
#include "hrx_loader.hpp"
#include "iree/base/tooling/flags.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

IREE_FLAG(string, graph_runtime, "hrx",
          "Graph implementation: hrx, hip (HRX frontend), or rocm.");
IREE_FLAG(string, graph_hip_library, "", "Exact selected libamdhip64 path.");
IREE_FLAG(int32_t, graph_host_interval, 0,
          "Insert a host checkpoint after each group of this many dispatches; "
          "zero records one uninterrupted device sequence.");

namespace {

constexpr uint32_t kOutputBase = 0xC0FFEE;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

void Check(hrx_status_t status) {
  if (hrx_status_is_ok(status)) {
    return;
  }
  char* message = nullptr;
  size_t message_length = 0;
  HrxLoader::instance().status_to_string(status, &message, &message_length);
  std::fprintf(stderr, "HRX operation failed: %s\n", message);
  HrxLoader::instance().status_free_message(message);
  HrxLoader::instance().status_ignore(status);
  std::abort();
}

void Check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP operation failed: %d\n",
                 static_cast<int>(status));
    std::abort();
  }
}

// Each implementation keeps one driver, device, stream, and module for the
// process. Per-row graphs differ only in node count and output memory class.
class HrxRuntime {
 public:
  using GraphType = hrx_graph_t;

  HrxRuntime() : api_(HrxLoader::instance()) {
    hrx_runtime_t runtime = nullptr;
    Check(api_.runtime_create(nullptr, api_.host_allocator_system(), &runtime));
    const char* driver = std::getenv("HRX_GPU_DRIVER");
    const std::string uri =
        std::string(driver && driver[0] ? driver : "amdgpu") + "://0";
    const hrx_device_create_params_t params = {
        .uri = uri.c_str(),
        .completion_wait_flags = HRX_WAIT_FLAG_YIELD,
    };
    hrx_status_t status =
        api_.device_group_create(runtime, 1, &params, &group_);
    api_.runtime_release(runtime);
    Check(status);
    device_ = api_.device_group_get_device(group_, 0);
    for (uint32_t i = 0; i < api_.device_queue_family_count(device_); ++i) {
      hrx_queue_family_info_t family;
      Check(api_.device_query_queue_family(device_, i, &family));
      if (!transfer_queue_ && (family.roles & HRX_QUEUE_ROLE_TRANSFER) &&
          family.provisioned_queue_count) {
        transfer_queue_ = api_.device_get_queue(device_, i, 0);
      }
      if (!dispatch_queue_ && (family.roles & HRX_QUEUE_ROLE_DISPATCH) &&
          (family.roles & HRX_QUEUE_ROLE_TRANSFER) &&
          family.provisioned_queue_count) {
        dispatch_queue_ = api_.device_get_queue(device_, i, 0);
      }
    }
    Require(transfer_queue_ != nullptr, "no provisioned transfer queue");
    Require(dispatch_queue_ != nullptr, "no provisioned compute queue");
    Check(api_.stream_create(dispatch_queue_, 0, &stream_));
    char architecture[64] = {};
    Check(api_.device_get_property(device_, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                   architecture, sizeof(architecture)));
    label_ = std::string("HRX ") + architecture + " wait=blocking";
    const auto image = hrx_cts::FindAmdgpuExecutableTestImage(architecture);
    Require(image.file != nullptr, "no configured HSACO for selected GPU");
    Check(api_.executable_load_data(device_, image.file->data, image.file->size,
                                    "amdgpu", image.target_key.c_str(),
                                    &module_));
    Check(api_.executable_lookup_export_by_name(module_, "hrx_store_output",
                                                &ordinal_));
    Check(api_.executable_export_info(module_, ordinal_, &info_));
  }

  ~HrxRuntime() {
    Wait();
    api_.stream_release(stream_);
    api_.executable_release(module_);
    api_.device_group_release(group_);
  }

  void Prepare(size_t count, bool host) {
    host_call_count_ = 0;
    launch_count_ = 0;
    values_.assign(count, 0);
    hrx_buffer_params_t params = {
        .type = host ? HRX_MEMORY_TYPE_HOST_LOCAL |
                           HRX_MEMORY_TYPE_HOST_COHERENT |
                           HRX_MEMORY_TYPE_DEVICE_VISIBLE
                     : HRX_MEMORY_TYPE_DEVICE_LOCAL,
        .access = HRX_MEMORY_ACCESS_ALL,
        .usage = HRX_BUFFER_USAGE_DEFAULT |
                 (host ? HRX_BUFFER_USAGE_MAPPING_PERSISTENT : 0u),
    };
    Check(api_.allocator_allocate_buffer(api_.device_allocator(device_), params,
                                         count * sizeof(uint32_t), &buffer_));
    if (host) {
      void* mapped = nullptr;
      Check(api_.buffer_map(buffer_, HRX_MAP_READ | HRX_MAP_WRITE, 0,
                            count * sizeof(uint32_t), &mapped));
      mapped_ = static_cast<uint32_t*>(mapped);
      std::fill_n(mapped_, count, 0);
    } else {
      Check(api_.synchronous_h2d(transfer_queue_, values_.data(), buffer_, 0,
                                 count * sizeof(uint32_t)));
    }
    value_bias_ = 0;
    BuildGraph(count, 0, &graph_);
    Check(api_.graph_instantiate(graph_, dispatch_queue_, 0, &executable_));
  }

  void PrepareUpdate() { BuildGraph(values_.size(), 1, &replacement_); }

  void Update() {
    const uint32_t next_bias = value_bias_ ^ 1;
    Check(
        api_.graph_exec_update(executable_, next_bias ? replacement_ : graph_));
    value_bias_ = next_bias;
  }

  void Reinstantiate() { InstallTemplate(graph_); }

  GraphType ConstructTemplate() {
    hrx_graph_t graph = nullptr;
    BuildGraph(values_.size(), 0, &graph);
    return graph;
  }

  void ReleaseTemplate(GraphType graph) { api_.graph_release(graph); }

  void InstallTemplate(GraphType graph) {
    hrx_graph_exec_t executable = nullptr;
    Check(api_.graph_instantiate(graph, dispatch_queue_, 0, &executable));
    api_.graph_exec_release(executable_);
    executable_ = executable;
  }

  void Launch() {
    Check(api_.graph_exec_launch(executable_, stream_));
    ++launch_count_;
  }
  void Wait() { Check(api_.stream_synchronize(stream_)); }

  void Verify() {
    const size_t checkpoints = FLAG_graph_host_interval > 0
                                   ? values_.size() / FLAG_graph_host_interval
                                   : 0;
    Require(host_call_count_ == launch_count_ * checkpoints,
            "HRX graph host checkpoint count mismatch");
    if (mapped_) {
      std::copy_n(mapped_, values_.size(), values_.begin());
    } else {
      Check(api_.synchronous_d2h(transfer_queue_, buffer_, 0, values_.data(),
                                 values_.size() * sizeof(uint32_t)));
    }
    for (size_t i = 0; i < values_.size(); ++i) {
      Require(values_[i] == kOutputBase + i + value_bias_,
              "HRX graph output mismatch");
    }
  }

  void Reset() {
    Wait();
    api_.graph_exec_release(executable_);
    api_.graph_release(graph_);
    api_.graph_release(replacement_);
    if (mapped_) {
      Check(api_.buffer_unmap(buffer_));
    }
    api_.buffer_release(buffer_);
    executable_ = nullptr;
    graph_ = nullptr;
    replacement_ = nullptr;
    buffer_ = nullptr;
    mapped_ = nullptr;
  }

  const std::string& label() const { return label_; }

 private:
  void BuildGraph(size_t count, uint32_t value_bias, hrx_graph_t* out_graph) {
    Check(api_.graph_create(device_, 0, out_graph));
    hrx_graph_node_t previous = nullptr;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t value = kOutputBase + i + value_bias;
      const hrx_buffer_ref_t binding = {buffer_, i * sizeof(uint32_t),
                                        sizeof(uint32_t)};
      const hrx_graph_kernel_node_attrs_t attrs = {
          module_,
          ordinal_,
          {{1, 1, 1},
           {info_.workgroup_size[0], info_.workgroup_size[1],
            info_.workgroup_size[2]},
           0},
          &value,
          sizeof(value),
          &binding,
          1,
          HRX_DISPATCH_FLAG_NONE,
      };
      hrx_graph_node_t node = nullptr;
      Check(api_.graph_add_kernel_node(*out_graph,
                                       previous ? &previous : nullptr,
                                       previous ? 1 : 0, &attrs, &node));
      previous = node;
      if (FLAG_graph_host_interval > 0 &&
          (i + 1) % FLAG_graph_host_interval == 0) {
        const hrx_graph_host_call_node_attrs_t host_attrs = {
            .fn = +[](void* user_data) -> hrx_status_t {
              ++*static_cast<uint64_t*>(user_data);
              return hrx_ok_status();
            },
            .user_data = &host_call_count_,
        };
        Check(api_.graph_add_host_call_node(*out_graph, &node, 1, &host_attrs,
                                            &previous));
      }
    }
  }

  // Public entry points from the exact HRX DSO being measured.
  HrxLoader& api_;
  // Explicit runtime and native-device owner shared across all rows.
  hrx_device_group_t group_ = nullptr;
  // Group-owned device borrowed through benchmark teardown.
  hrx_device_t device_ = nullptr;
  // Borrowed provisioned queue for setup uploads and verification readbacks.
  hrx_queue_t transfer_queue_ = nullptr;
  // Borrowed compute queue selecting graph recording and execution.
  hrx_queue_t dispatch_queue_ = nullptr;
  // Ordered timeline shared by all rows.
  hrx_stream_t stream_ = nullptr;
  // Persistent code object containing the finite store kernel.
  hrx_executable_t module_ = nullptr;
  // Reflected store export index.
  uint32_t ordinal_ = 0;
  // Reflected dimensions used by every dispatch.
  hrx_executable_export_info_t info_ = {};
  // Row-local graph retained through its last completion.
  hrx_graph_t graph_ = nullptr;
  // Second template with changed constants for alternating updates.
  hrx_graph_t replacement_ = nullptr;
  // Value adjustment written by the currently installed template.
  uint32_t value_bias_ = 0;
  // Instantiated graph reused throughout one row.
  hrx_graph_exec_t executable_ = nullptr;
  // One output word per node in the selected memory class.
  hrx_buffer_t buffer_ = nullptr;
  // Persistent coherent host mapping, or null for device-local output.
  uint32_t* mapped_ = nullptr;
  // CPU comparison storage used only outside timed regions.
  std::vector<uint32_t> values_;
  // Architecture and wait policy included in every result.
  std::string label_;
  // Ordered graph callbacks observed only after stream completion.
  uint64_t host_call_count_ = 0;
  // Accepted invocations used to check every host checkpoint executed.
  uint64_t launch_count_ = 0;
};

class HipRuntime {
 public:
  using GraphType = hipGraph_t;

  HipRuntime() {
    Require(FLAG_graph_hip_library[0] != 0, "--graph_hip_library is required");
    library_.load(FLAG_graph_hip_library);
    Check(Resolve<decltype(&hipInit)>("hipInit")(0));
    Check(Resolve<decltype(&hipSetDeviceFlags)>("hipSetDeviceFlags")(
        hipDeviceScheduleBlockingSync));
    hipDeviceProp_t properties = {};
    Check(Resolve<decltype(&hipGetDevicePropertiesR0600)>(
        "hipGetDevicePropertiesR0600")(&properties, 0));
    char pci_address[32] = {};
    Check(Resolve<decltype(&hipDeviceGetPCIBusId)>("hipDeviceGetPCIBusId")(
        pci_address, sizeof(pci_address), 0));
    label_ =
        std::string(std::string(FLAG_graph_runtime) == "hip" ? "HRX HIP "
                                                             : "ROCm HIP ") +
        properties.gcnArchName + " " + pci_address + " wait=blocking";
    const auto image =
        hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
    Require(image.file != nullptr, "no configured HSACO for selected GPU");
    Check(Resolve<decltype(&hipModuleLoadData)>("hipModuleLoadData")(
        &module_, image.file->data));
    Check(Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction")(
        &function_, module_, "hrx_store_output"));
    Check(Resolve<decltype(&hipStreamCreateWithFlags)>(
        "hipStreamCreateWithFlags")(&stream_, hipStreamNonBlocking));
    launch_ = Resolve<decltype(&hipGraphLaunch)>("hipGraphLaunch");
    wait_ = Resolve<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
    update_ = Resolve<decltype(&hipGraphExecUpdate)>("hipGraphExecUpdate");
    instantiate_ =
        Resolve<decltype(&hipGraphInstantiate)>("hipGraphInstantiate");
    destroy_ = Resolve<decltype(&hipGraphExecDestroy)>("hipGraphExecDestroy");
  }

  ~HipRuntime() {
    Wait();
    Check(Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy")(stream_));
    Check(Resolve<decltype(&hipModuleUnload)>("hipModuleUnload")(module_));
    if (std::string(FLAG_graph_runtime) == "hip") {
      Check(Resolve<decltype(&hipHALDeinit)>("hipHALDeinit")());
    }
  }

  template <typename T>
  T Resolve(const char* name) {
    return reinterpret_cast<T>(library_.loadSymbol(name));
  }

  void Prepare(size_t count, bool host) {
    host_call_count_ = 0;
    launch_count_ = 0;
    values_.assign(count, 0);
    host_ = host;
    if (host) {
      // hipHostAlloc accepts only CUDA-compatible flags. The HIP-native entry
      // point accepts the explicit coherent memory requirement.
      Check(Resolve<decltype(&hipHostAlloc)>("hipHostMalloc")(
          &output_, count * sizeof(uint32_t),
          hipHostMallocMapped | hipHostMallocCoherent));
      Check(Resolve<decltype(&hipHostGetDevicePointer)>(
          "hipHostGetDevicePointer")(&device_output_, output_, 0));
      std::fill_n(static_cast<uint32_t*>(output_), count, 0);
    } else {
      Check(Resolve<decltype(&hipMalloc)>("hipMalloc")(
          &output_, count * sizeof(uint32_t)));
      device_output_ = output_;
      Check(Resolve<decltype(&hipMemcpy)>("hipMemcpy")(output_, values_.data(),
                                                       count * sizeof(uint32_t),
                                                       hipMemcpyHostToDevice));
    }
    value_bias_ = 0;
    BuildGraph(count, 0, &graph_);
    Check(instantiate_(&executable_, graph_, nullptr, nullptr, 0));
  }

  void PrepareUpdate() { BuildGraph(values_.size(), 1, &replacement_); }

  void Update() {
    const uint32_t next_bias = value_bias_ ^ 1;
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result;
    Check(update_(executable_, next_bias ? replacement_ : graph_, &error_node,
                  &result));
    Require(result == hipGraphExecUpdateSuccess, "graph replacement rejected");
    value_bias_ = next_bias;
  }

  void Reinstantiate() { InstallTemplate(graph_); }

  GraphType ConstructTemplate() {
    hipGraph_t graph = nullptr;
    BuildGraph(values_.size(), 0, &graph);
    return graph;
  }

  void ReleaseTemplate(GraphType graph) {
    Check(Resolve<decltype(&hipGraphDestroy)>("hipGraphDestroy")(graph));
  }

  void InstallTemplate(GraphType graph) {
    hipGraphExec_t executable = nullptr;
    Check(instantiate_(&executable, graph, nullptr, nullptr, 0));
    Check(destroy_(executable_));
    executable_ = executable;
  }

  void Launch() {
    Check(launch_(executable_, stream_));
    ++launch_count_;
  }
  void Wait() { Check(wait_(stream_)); }

  void Verify() {
    const size_t checkpoints = FLAG_graph_host_interval > 0
                                   ? values_.size() / FLAG_graph_host_interval
                                   : 0;
    Require(host_call_count_ == launch_count_ * checkpoints,
            "HIP graph host checkpoint count mismatch");
    if (host_) {
      std::copy_n(static_cast<uint32_t*>(output_), values_.size(),
                  values_.begin());
    } else {
      Check(Resolve<decltype(&hipMemcpy)>("hipMemcpy")(
          values_.data(), output_, values_.size() * sizeof(uint32_t),
          hipMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < values_.size(); ++i) {
      Require(values_[i] == kOutputBase + i + value_bias_,
              "HIP graph output mismatch");
    }
  }

  void Reset() {
    Wait();
    Check(destroy_(executable_));
    Check(Resolve<decltype(&hipGraphDestroy)>("hipGraphDestroy")(graph_));
    if (replacement_) {
      Check(
          Resolve<decltype(&hipGraphDestroy)>("hipGraphDestroy")(replacement_));
      replacement_ = nullptr;
    }
    if (host_) {
      Check(Resolve<decltype(&hipFreeHost)>("hipFreeHost")(output_));
    } else {
      Check(Resolve<decltype(&hipFree)>("hipFree")(output_));
    }
  }

  const std::string& label() const { return label_; }

 private:
  void BuildGraph(size_t count, uint32_t value_bias, hipGraph_t* out_graph) {
    Check(Resolve<decltype(&hipStreamBeginCapture)>("hipStreamBeginCapture")(
        stream_, hipStreamCaptureModeGlobal));
    const auto dispatch =
        Resolve<decltype(&hipModuleLaunchKernel)>("hipModuleLaunchKernel");
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t value = kOutputBase + i + value_bias;
      void* output = static_cast<uint32_t*>(device_output_) + i;
      void* arguments[] = {&output, &value};
      Check(dispatch(function_, 1, 1, 1, 1, 1, 1, 0, stream_, arguments,
                     nullptr));
      if (FLAG_graph_host_interval > 0 &&
          (i + 1) % FLAG_graph_host_interval == 0) {
        Check(Resolve<decltype(&hipLaunchHostFunc)>("hipLaunchHostFunc")(
            stream_,
            +[](void* user_data) { ++*static_cast<uint64_t*>(user_data); },
            &host_call_count_));
      }
    }
    Check(Resolve<decltype(&hipStreamEndCapture)>("hipStreamEndCapture")(
        stream_, out_graph));
  }

  // Exact selected HIP implementation; symbols resolve only through this DSO.
  HrxDynamicLibrary library_;
  // Persistent code object and reflected finite store function.
  hipModule_t module_ = nullptr;
  // Store kernel using the same embedded bytes as HRX.
  hipFunction_t function_ = nullptr;
  // Ordered stream using native blocking waits.
  hipStream_t stream_ = nullptr;
  // Row-local captured graph.
  hipGraph_t graph_ = nullptr;
  // Second captured template with different scalar values.
  hipGraph_t replacement_ = nullptr;
  // Value adjustment written by the currently installed template.
  uint32_t value_bias_ = 0;
  // Replayed graph executable.
  hipGraphExec_t executable_ = nullptr;
  // Output allocation, coherent host memory or device-local storage.
  void* output_ = nullptr;
  // GPU-accessible address of output_.
  void* device_output_ = nullptr;
  // True when CPU verification may read output_ directly.
  bool host_ = false;
  // CPU comparison storage used only outside timed regions.
  std::vector<uint32_t> values_;
  // Cached public graph-launch entry point.
  decltype(&hipGraphLaunch) launch_ = nullptr;
  // Cached public stream-completion entry point.
  decltype(&hipStreamSynchronize) wait_ = nullptr;
  // Cached graph replacement entry point, with symbol lookup outside timing.
  decltype(&hipGraphExecUpdate) update_ = nullptr;
  // Cached constructor measured by the instantiation rows.
  decltype(&hipGraphInstantiate) instantiate_ = nullptr;
  // Cached destructor included when replacing an instantiated executable.
  decltype(&hipGraphExecDestroy) destroy_ = nullptr;
  // Architecture, PCI identity, and wait policy included in every result.
  std::string label_;
  // Ordered graph callbacks observed only after stream completion.
  uint64_t host_call_count_ = 0;
  // Accepted invocations used to check every host checkpoint executed.
  uint64_t launch_count_ = 0;
};

template <typename Runtime>
Runtime& GetRuntime() {
  static Runtime runtime;
  return runtime;
}

template <typename Runtime, bool IncludeCompletion>
void RunGraph(benchmark::State& state) {
  auto& runtime = GetRuntime<Runtime>();
  const size_t node_count = state.range(0);
  const size_t launch_count = state.range(1);
  const bool host = state.range(2) != 0;
  runtime.Prepare(node_count, host);
  for (int i = 0; i < 16; ++i) {
    runtime.Launch();
    runtime.Wait();
  }
  runtime.Verify();
  state.SetLabel(runtime.label() +
                 (host ? " output=host-coherent" : " output=device-local"));
  for (auto _ : state) {
    for (size_t i = 0; i < launch_count; ++i) {
      runtime.Launch();
    }
    if constexpr (!IncludeCompletion) {
      state.PauseTiming();
    }
    runtime.Wait();
    if constexpr (!IncludeCompletion) {
      state.ResumeTiming();
    }
  }
  runtime.Verify();
  runtime.Reset();
  state.counters["launches"] =
      benchmark::Counter(static_cast<double>(launch_count),
                         benchmark::Counter::kIsIterationInvariant);
  state.counters["nodes"] = static_cast<double>(node_count);
  state.counters["host_interval"] = FLAG_graph_host_interval;
  state.SetItemsProcessed(state.iterations() * launch_count);
}

template <bool IncludeCompletion>
void Graph(benchmark::State& state) {
  const std::string runtime = FLAG_graph_runtime;
  if (runtime == "hrx") {
    RunGraph<HrxRuntime, IncludeCompletion>(state);
  } else {
    Require(runtime == "hip" || runtime == "rocm", "unknown --graph_runtime");
    RunGraph<HipRuntime, IncludeCompletion>(state);
  }
}

// Measures preparation separately from enqueue and completion. Graph templates
// already exist; instantiation includes release of the previous executable.
template <typename Runtime, bool Update>
void RunGraphEdit(benchmark::State& state) {
  auto& runtime = GetRuntime<Runtime>();
  const size_t node_count = state.range(0);
  const bool host = state.range(1) != 0;
  runtime.Prepare(node_count, host);
  if constexpr (Update) {
    runtime.PrepareUpdate();
  }
  state.SetLabel(runtime.label() +
                 (host ? " output=host-coherent" : " output=device-local"));
  for (auto _ : state) {
    if constexpr (Update) {
      runtime.Update();
    } else {
      runtime.Reinstantiate();
    }
    state.PauseTiming();
    runtime.Launch();
    runtime.Wait();
    runtime.Verify();
    state.ResumeTiming();
  }
  runtime.Reset();
  state.counters["nodes"] = static_cast<double>(node_count);
  state.counters["host_interval"] = FLAG_graph_host_interval;
  state.SetItemsProcessed(state.iterations());
}

template <bool Update>
void GraphEdit(benchmark::State& state) {
  const std::string runtime = FLAG_graph_runtime;
  if (runtime == "hrx") {
    RunGraphEdit<HrxRuntime, Update>(state);
  } else {
    Require(runtime == "hip" || runtime == "rocm", "unknown --graph_runtime");
    RunGraphEdit<HipRuntime, Update>(state);
  }
}

// Builds a bounded live set of real dispatch templates. Native HRX constructs
// explicit nodes and HIP captures the matching dispatch sequence. Compilation,
// completion, and releasing the live set stay outside construction timing.
template <typename Runtime>
void RunGraphConstruct(benchmark::State& state) {
  auto& runtime = GetRuntime<Runtime>();
  const size_t node_count = state.range(0);
  const size_t retained_count = state.range(1);
  runtime.Prepare(node_count, /*host=*/true);
  std::vector<typename Runtime::GraphType> graphs(retained_count);
  for (auto& graph : graphs) {
    graph = runtime.ConstructTemplate();
  }
  runtime.InstallTemplate(graphs.back());
  runtime.Launch();
  runtime.Wait();
  runtime.Verify();
  for (auto graph : graphs) {
    runtime.ReleaseTemplate(graph);
  }
  state.SetLabel(runtime.label() + " output=host-coherent");
  for (auto _ : state) {
    for (auto& graph : graphs) {
      graph = runtime.ConstructTemplate();
    }
    benchmark::DoNotOptimize(graphs.data());
    state.PauseTiming();
    for (auto graph : graphs) {
      runtime.ReleaseTemplate(graph);
    }
    state.ResumeTiming();
  }
  runtime.Reset();
  state.counters["nodes"] = static_cast<double>(node_count);
  state.counters["retained_graphs"] = static_cast<double>(retained_count);
  state.counters["host_interval"] = FLAG_graph_host_interval;
  state.SetItemsProcessed(state.iterations() * node_count * retained_count);
}

void GraphConstruct(benchmark::State& state) {
  const std::string runtime = FLAG_graph_runtime;
  if (runtime == "hrx") {
    RunGraphConstruct<HrxRuntime>(state);
  } else {
    Require(runtime == "hip" || runtime == "rocm", "unknown --graph_runtime");
    RunGraphConstruct<HipRuntime>(state);
  }
}

// Keeps a bounded window of launches, each consuming a different recording.
// UpdateSubmit excludes the final join; UpdateComplete includes it. Neither
// waits on the host between replacement and the next enqueue within a batch.
template <typename Runtime, bool IncludeCompletion>
void RunGraphUpdate(benchmark::State& state) {
  auto& runtime = GetRuntime<Runtime>();
  const size_t node_count = state.range(0);
  const size_t launch_count = state.range(1);
  const bool host = state.range(2) != 0;
  runtime.Prepare(node_count, host);
  runtime.PrepareUpdate();
  state.SetLabel(runtime.label() +
                 (host ? " output=host-coherent" : " output=device-local"));
  for (auto _ : state) {
    for (size_t i = 0; i < launch_count; ++i) {
      runtime.Update();
      runtime.Launch();
    }
    if constexpr (!IncludeCompletion) {
      state.PauseTiming();
    }
    runtime.Wait();
    if constexpr (!IncludeCompletion) {
      state.ResumeTiming();
    }
  }
  runtime.Verify();
  runtime.Reset();
  state.counters["launches"] =
      benchmark::Counter(static_cast<double>(launch_count),
                         benchmark::Counter::kIsIterationInvariant);
  state.counters["nodes"] = static_cast<double>(node_count);
  state.counters["host_interval"] = FLAG_graph_host_interval;
  state.SetItemsProcessed(state.iterations() * launch_count);
}

template <bool IncludeCompletion>
void GraphUpdate(benchmark::State& state) {
  const std::string runtime = FLAG_graph_runtime;
  if (runtime == "hrx") {
    RunGraphUpdate<HrxRuntime, IncludeCompletion>(state);
  } else {
    Require(runtime == "hip" || runtime == "rocm", "unknown --graph_runtime");
    RunGraphUpdate<HipRuntime, IncludeCompletion>(state);
  }
}

void GraphEditArguments(benchmark::Benchmark* benchmark) {
  for (int host : {0, 1}) {
    for (int count : {1, 32, 256}) {
      benchmark->Args({count, host});
    }
  }
}

void GraphUpdateArguments(benchmark::Benchmark* benchmark) {
  for (int host : {0, 1}) {
    for (int count : {1, 32, 256}) {
      for (int batch : {1, 8}) {
        benchmark->Args({count, batch, host});
      }
    }
  }
}

void GraphArguments(benchmark::Benchmark* benchmark) {
  for (int host : {0, 1}) {
    for (int count : {1, 2, 4, 8, 16, 32, 256}) {
      for (int batch : {1, 32}) {
        benchmark->Args({count, batch, host});
      }
    }
  }
}

void GraphConstructArguments(benchmark::Benchmark* benchmark) {
  for (int count : {16, 256, 4096}) {
    for (int retained : {1, 16, 64}) {
      benchmark->Args({count, retained});
    }
  }
}

BENCHMARK(GraphConstruct)
    ->Name("Graph/Construct")
    ->Apply(GraphConstructArguments)
    ->ArgNames({"nodes", "retained"})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(Graph, false)
    ->Name("Graph/Submit")
    ->Apply(GraphArguments)
    ->ArgNames({"nodes", "batch", "host"})
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(Graph, true)
    ->Name("Graph/Complete")
    ->Apply(GraphArguments)
    ->ArgNames({"nodes", "batch", "host"})
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);

BENCHMARK_TEMPLATE(GraphEdit, false)
    ->Name("Graph/Instantiate")
    ->Apply(GraphEditArguments)
    ->ArgNames({"nodes", "host"})
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(GraphEdit, true)
    ->Name("Graph/Update")
    ->Apply(GraphEditArguments)
    ->ArgNames({"nodes", "host"})
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(GraphUpdate, false)
    ->Name("Graph/UpdateSubmit")
    ->Apply(GraphUpdateArguments)
    ->ArgNames({"nodes", "batch", "host"})
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(GraphUpdate, true)
    ->Name("Graph/UpdateComplete")
    ->Apply(GraphUpdateArguments)
    ->ArgNames({"nodes", "batch", "host"})
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);

}  // namespace
