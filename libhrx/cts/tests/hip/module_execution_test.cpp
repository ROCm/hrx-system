// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"
#include "libhrx/cts/core/amdgpu_hip_cooperative_test_data.hpp"
#include "libhrx/cts/core/amdgpu_hip_printf_test_data.hpp"

namespace {

const char* CandidateLibPath() {
  if (const char* environment_path = std::getenv("HRX_TEST_LIBAMDHIP64");
      environment_path && *environment_path != '\0') {
    return environment_path;
  }
  return "libamdhip64.so";
}

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipSetDeviceFn = hipError_t (*)(int device);
using HipGetDeviceCountFn = hipError_t (*)(int* device_count);
using HipGetDevicePropertiesR0600Fn =
    hipError_t (*)(hipDeviceProp_t* properties, int device);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t* module,
                                           const void* image);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t module);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipModule_t module,
                                              const char* name);
using HipFuncGetAttributeFn = hipError_t (*)(int* value,
                                             hipFuncAttribute_t attribute,
                                             hipFunction_t function);
using HipModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments, void** extra);
using HipModuleLaunchCooperativeKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments);
using HipLaunchKernelExCFn = hipError_t (*)(const hipLaunchConfig_t* config,
                                            const void* function,
                                            void** arguments);
using HipDrvLaunchKernelExFn = hipError_t (*)(const HIP_LAUNCH_CONFIG* config,
                                              hipFunction_t function,
                                              void** arguments, void** extra);
using HipModuleOccupancyMaxActiveBlocksPerMultiprocessorFn =
    hipError_t (*)(int* block_count, hipFunction_t function, int block_size,
                   size_t dynamic_shared_memory_bytes);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipMallocFn = hipError_t (*)(hipDeviceptr_t* pointer, size_t size);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t pointer);
using HipMemcpyFn = hipError_t (*)(void* target, const void* source,
                                   size_t size, hipMemcpyKind kind);
using HipMemcpyAsyncFn = hipError_t (*)(void* target, const void* source,
                                        size_t size, hipMemcpyKind kind,
                                        hipStream_t stream);
using HipMemsetAsyncFn = hipError_t (*)(void* target, int value, size_t size,
                                        hipStream_t stream);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamQueryFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipLaunchHostFuncFn = hipError_t (*)(hipStream_t stream, hipHostFn_t fn,
                                           void* user_data);
using HipHostMallocFn = hipError_t (*)(void** pointer, size_t size,
                                       unsigned int flags);
using HipHostFreeFn = hipError_t (*)(void* pointer);
using HipHostGetDevicePointerFn = hipError_t (*)(hipDeviceptr_t* device_pointer,
                                                 void* host_pointer,
                                                 unsigned int flags);
using HipDeviceGetDevResourceFn = hipError_t (*)(hipDevice_t device,
                                                 hipDevResource* resource,
                                                 hipDevResourceType type);
using HipDeviceGetExecutionCtxFn = hipError_t (*)(hipExecutionCtx_t* context,
                                                  int device);
using HipDevResourceGenerateDescFn =
    hipError_t (*)(hipDevResourceDesc_t* descriptor, hipDevResource* resources,
                   unsigned int resource_count);
using HipGreenCtxCreateFn = hipError_t (*)(hipExecutionCtx_t* context,
                                           hipDevResourceDesc_t descriptor,
                                           int device, unsigned int flags);
using HipExecutionCtxDestroyFn = hipError_t (*)(hipExecutionCtx_t context);
using HipExecutionCtxStreamCreateFn = hipError_t (*)(hipStream_t* stream,
                                                     hipExecutionCtx_t context,
                                                     unsigned int flags,
                                                     int priority);
using HipExtLaunchMultiKernelMultiDeviceFn = hipError_t (*)(
    hipLaunchParams* launch_params, int device_count, unsigned int flags);
using HipRegisterFatBinaryFn = void** (*)(const void* image);
using HipUnregisterFatBinaryFn = void (*)(void** registration);
// The compiler ABI names these pointer-only metadata slots `uint3`. The local
// binding's `dim3` has the same three-unsigned-component representation and the
// CTS always passes null for both slots.
using HipCompilerIndex = dim3;
using HipRegisterFunctionFn = void (*)(
    void** registration, const void* host_function, char* device_function,
    const char* device_name, unsigned int thread_limit,
    HipCompilerIndex* thread_index, HipCompilerIndex* block_index,
    dim3* block_dimensions, dim3* grid_dimensions, int* shared_memory_size);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipGraphGetNodesFn = hipError_t (*)(hipGraph_t graph,
                                          hipGraphNode_t* nodes,
                                          size_t* node_count);
using HipGraphNodeGetTypeFn = hipError_t (*)(hipGraphNode_t node,
                                             hipGraphNodeType* type);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddKernelNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, const void* params);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* graph_executable,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t graph_executable,
                                        hipStream_t stream);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t graph_executable);
using HipGraphExecKernelNodeSetParamsFn =
    hipError_t (*)(hipGraphExec_t graph_executable, hipGraphNode_t node,
                   const hipKernelNodeParams* params);

struct ExecutionContextDeleter {
  // Runtime entry point used to destroy a live execution context.
  HipExecutionCtxDestroyFn destroy = nullptr;

  void operator()(ihipExecutionCtx_t* context) const {
    if (context) {
      const hipError_t result = destroy(context);
      EXPECT_EQ(hipSuccess, result);
    }
  }
};

using ScopedExecutionContext =
    std::unique_ptr<ihipExecutionCtx_t, ExecutionContextDeleter>;

struct StreamDeleter {
  // Runtime entry point used to destroy a live stream.
  HipStreamDestroyFn destroy = nullptr;

  void operator()(hipStream_st* stream) const {
    if (stream) {
      const hipError_t result = destroy(stream);
      EXPECT_EQ(hipSuccess, result);
    }
  }
};

using ScopedStream = std::unique_ptr<hipStream_st, StreamDeleter>;

struct ScopedRegistration {
  ~ScopedRegistration() {
    if (value) {
      unregister(value);
    }
  }

  // Compiler-style fat-binary registration released at scope exit.
  void** value = nullptr;
  // Registration release entry point from the binding under test.
  HipUnregisterFatBinaryFn unregister = nullptr;
};

struct ScopedDeviceSelection {
  ~ScopedDeviceSelection() {
    if (set_device) {
      const hipError_t result = set_device(original_device);
      EXPECT_EQ(hipSuccess, result);
    }
  }

  // Device that was selected before the test began switching devices.
  int original_device = 0;
  // Device selection entry point from the binding under test.
  HipSetDeviceFn set_device = nullptr;
};

// Host callback that holds a stream at a test-controlled point. An armed gate
// keeps its storage alive until the callback has returned on every exit path.
class HostGate {
 public:
  ~HostGate() {
    Open();
    std::unique_lock<std::mutex> lock(mutex_);
    if (is_armed_) {
      condition_.wait(lock, [this] { return is_finished_; });
    }
  }

  static void Callback(void* user_data) {
    static_cast<HostGate*>(user_data)->Wait();
  }

  void Arm() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_armed_ = true;
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return is_entered_; });
  }

  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      is_open_ = true;
    }
    condition_.notify_all();
  }

  void WaitUntilFinished() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return is_finished_; });
  }

 private:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    is_entered_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this] { return is_open_; });
    is_finished_ = true;
    condition_.notify_all();
  }

  // Serializes gate state accessed by the test and callback threads.
  std::mutex mutex_;
  // Notifies entry, release, and completion state transitions.
  std::condition_variable condition_;
  // True after the callback has been successfully enqueued.
  bool is_armed_ = false;
  // True once the stream callback has begun waiting.
  bool is_entered_ = false;
  // True once the callback is allowed to return.
  bool is_open_ = false;
  // True once the callback no longer accesses this gate.
  bool is_finished_ = false;
};

// Coherent host allocation used to control and observe a running test kernel.
struct MultiDeviceKernelGate {
  void Reset() {
    __atomic_store_n(&entered, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&released, 0, __ATOMIC_RELEASE);
  }

  void Release() { __atomic_store_n(&released, 1, __ATOMIC_RELEASE); }

  // Set by the kernel once it has begun executing.
  uint32_t entered;
  // Set by the host to allow the kernel to finish.
  uint32_t released;
};
static_assert(offsetof(MultiDeviceKernelGate, released) == sizeof(uint32_t));
static_assert(sizeof(MultiDeviceKernelGate) == 2 * sizeof(uint32_t));

struct MultiDeviceLaunchState {
  ~MultiDeviceLaunchState() {
    if (!stream && !output && !host_gate) {
      return;
    }
    const hipError_t select_result = set_device(device);
    EXPECT_EQ(hipSuccess, select_result);
    if (select_result != hipSuccess) {
      return;
    }
    if (host_gate) {
      host_gate->Release();
    }
    if (stream) {
      const hipError_t destroy_result = stream_destroy(stream);
      EXPECT_EQ(hipSuccess, destroy_result);
      stream = nullptr;
    }
    if (output) {
      const hipError_t free_result = free_memory(output);
      EXPECT_EQ(hipSuccess, free_result);
      output = nullptr;
    }
    if (host_gate) {
      host_gate->~MultiDeviceKernelGate();
      const hipError_t free_result = free_host_memory(host_gate);
      EXPECT_EQ(hipSuccess, free_result);
      host_gate = nullptr;
    }
  }

  // Device owning the stream and allocation.
  int device = 0;
  // Explicit stream receiving the launch.
  hipStream_t stream = nullptr;
  // Device allocation written by the kernel.
  hipDeviceptr_t output = nullptr;
  // Coherent host allocation controlling the running kernel.
  MultiDeviceKernelGate* host_gate = nullptr;
  // Device mapping of |host_gate|.
  hipDeviceptr_t device_gate = nullptr;
  // Device gate pointer passed by address through the argument array.
  hipDeviceptr_t device_gate_argument = nullptr;
  // Device pointer value passed by address through the argument array.
  hipDeviceptr_t output_argument = nullptr;
  // Distinct value expected from this device's launch.
  uint32_t expected_value = 0;
  // Array of pointers to the kernel argument values.
  void* arguments[3] = {};
  // Device selection entry point used for cleanup.
  HipSetDeviceFn set_device = nullptr;
  // Stream destruction entry point used for cleanup.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Device allocation release entry point used for cleanup.
  HipFreeFn free_memory = nullptr;
  // Host allocation release entry point used for cleanup.
  HipHostFreeFn free_host_memory = nullptr;
};

void MultiDeviceGatedStoreHostStub() {}

void CooperativeGridSyncHostStub() {}

struct PointerArguments {
  // Device input values read by the kernel.
  uint32_t* input;
  // Device output values written by the kernel.
  uint32_t* output;
};

hipError_t AddKernelGraphNode(HipGraphAddKernelNodeFn add_kernel_node,
                              hipGraph_t graph, hipFunction_t function,
                              hipDeviceptr_t input, hipDeviceptr_t output,
                              hipGraphNode_t* out_node) {
  PointerArguments pointer_arguments = {
      static_cast<uint32_t*>(input),
      static_cast<uint32_t*>(output),
  };
  uint32_t scale = 5;
  uint32_t offset = 2;
  void* arguments[] = {&pointer_arguments, &scale, &offset};
  const hipKernelNodeParams params = {
      /*.blockDim=*/{1, 1, 1},
      /*.extra=*/nullptr,
      /*.func=*/function,
      /*.gridDim=*/{1, 1, 1},
      /*.kernelParams=*/arguments,
      /*.sharedMemBytes=*/0,
  };
  const hipError_t result =
      add_kernel_node(out_node, graph, /*dependencies=*/nullptr,
                      /*dependency_count=*/0, &params);

  // Poison every caller-owned argument container before returning. The graph
  // must already own the copied native argument image.
  pointer_arguments = {};
  scale = UINT32_MAX;
  offset = UINT32_MAX;
  std::fill(std::begin(arguments), std::end(arguments), nullptr);
  return result;
}

hipError_t AddPrintfGraphNode(HipGraphAddKernelNodeFn add_kernel_node,
                              hipGraph_t graph, hipFunction_t function,
                              hipDeviceptr_t output, hipGraphNode_t* out_node) {
  uint32_t value = 42;
  hipDeviceptr_t result = output;
  void* arguments[] = {&value, &result};
  const hipKernelNodeParams params = {
      /*.blockDim=*/{1, 1, 1},
      /*.extra=*/nullptr,
      /*.func=*/function,
      /*.gridDim=*/{1, 1, 1},
      /*.kernelParams=*/arguments,
      /*.sharedMemBytes=*/0,
  };
  const hipError_t add_result =
      add_kernel_node(out_node, graph, /*dependencies=*/nullptr,
                      /*dependency_count=*/0, &params);

  // The graph must own the copied native argument image.
  value = UINT32_MAX;
  result = nullptr;
  std::fill(std::begin(arguments), std::end(arguments), nullptr);
  return add_result;
}

TEST(HipModuleExecutionTest, OwnsLoadAndGraphInputsAcrossReloads) {
  void* library = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto get_device =
      ResolveHipSymbol<HipGetDeviceFn>(library, "hipGetDevice");
  const auto get_device_properties =
      ResolveHipSymbol<HipGetDevicePropertiesR0600Fn>(
          library, "hipGetDevicePropertiesR0600");
  const auto module_load_data =
      ResolveHipSymbol<HipModuleLoadDataFn>(library, "hipModuleLoadData");
  const auto module_unload =
      ResolveHipSymbol<HipModuleUnloadFn>(library, "hipModuleUnload");
  const auto module_get_function =
      ResolveHipSymbol<HipModuleGetFunctionFn>(library, "hipModuleGetFunction");
  const auto function_get_attribute =
      ResolveHipSymbol<HipFuncGetAttributeFn>(library, "hipFuncGetAttribute");
  const auto module_launch_kernel = ResolveHipSymbol<HipModuleLaunchKernelFn>(
      library, "hipModuleLaunchKernel");
  const auto device_synchronize =
      ResolveHipSymbol<HipDeviceSynchronizeFn>(library, "hipDeviceSynchronize");
  const auto hip_malloc = ResolveHipSymbol<HipMallocFn>(library, "hipMalloc");
  const auto hip_free = ResolveHipSymbol<HipFreeFn>(library, "hipFree");
  const auto hip_memcpy = ResolveHipSymbol<HipMemcpyFn>(library, "hipMemcpy");
  const auto graph_create =
      ResolveHipSymbol<HipGraphCreateFn>(library, "hipGraphCreate");
  const auto graph_destroy =
      ResolveHipSymbol<HipGraphDestroyFn>(library, "hipGraphDestroy");
  const auto graph_add_kernel_node = ResolveHipSymbol<HipGraphAddKernelNodeFn>(
      library, "hipGraphAddKernelNode");
  const auto graph_instantiate =
      ResolveHipSymbol<HipGraphInstantiateFn>(library, "hipGraphInstantiate");
  const auto graph_launch =
      ResolveHipSymbol<HipGraphLaunchFn>(library, "hipGraphLaunch");
  const auto graph_exec_destroy =
      ResolveHipSymbol<HipGraphExecDestroyFn>(library, "hipGraphExecDestroy");

  ASSERT_NE(nullptr, init);
  ASSERT_NE(nullptr, get_device);
  ASSERT_NE(nullptr, get_device_properties);
  ASSERT_NE(nullptr, module_load_data);
  ASSERT_NE(nullptr, module_unload);
  ASSERT_NE(nullptr, module_get_function);
  ASSERT_NE(nullptr, function_get_attribute);
  ASSERT_NE(nullptr, module_launch_kernel);
  ASSERT_NE(nullptr, device_synchronize);
  ASSERT_NE(nullptr, hip_malloc);
  ASSERT_NE(nullptr, hip_free);
  ASSERT_NE(nullptr, hip_memcpy);
  ASSERT_NE(nullptr, graph_create);
  ASSERT_NE(nullptr, graph_destroy);
  ASSERT_NE(nullptr, graph_add_kernel_node);
  ASSERT_NE(nullptr, graph_instantiate);
  ASSERT_NE(nullptr, graph_launch);
  ASSERT_NE(nullptr, graph_exec_destroy);

  const hipError_t init_result = init(/*flags=*/0);
  if (init_result != hipSuccess) {
    GTEST_SKIP() << "hipInit failed: " << init_result;
  }
  int device = 0;
  ASSERT_EQ(hipSuccess, get_device(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, get_device_properties(&properties, device));

  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << properties.gcnArchName;

  auto load_transient_image = [&](hipModule_t* out_module) {
    std::vector<uint8_t> image(test_image.file->data,
                               test_image.file->data + test_image.file->size);
    const hipError_t result = module_load_data(out_module, image.data());
    std::fill(image.begin(), image.end(), uint8_t{0xA5});
    std::vector<uint8_t>().swap(image);
    return result;
  };

  hipDeviceptr_t input = nullptr;
  hipDeviceptr_t output = nullptr;
  ASSERT_EQ(hipSuccess, hip_malloc(&input, 4 * sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess, hip_malloc(&output, 4 * sizeof(uint32_t)));
  const std::array<uint32_t, 4> input_values = {1, 2, 3, 4};
  ASSERT_EQ(hipSuccess,
            hip_memcpy(input, input_values.data(), sizeof(input_values),
                       hipMemcpyHostToDevice));

  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, load_transient_image(&module));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess, module_get_function(&function, module,
                                            "hrx_transform_nested_pointers"));

  int maximum_threads_per_block = 0;
  int fixed_shared_memory_size = 0;
  int local_memory_size = 0;
  int register_count = 0;
  int maximum_dynamic_shared_memory_size = 0;
  ASSERT_EQ(hipSuccess, function_get_attribute(
                            &maximum_threads_per_block,
                            hipFuncAttributeMaxThreadsPerBlock, function));
  ASSERT_EQ(hipSuccess,
            function_get_attribute(&fixed_shared_memory_size,
                                   hipFuncAttributeSharedSizeBytes, function));
  ASSERT_EQ(hipSuccess,
            function_get_attribute(&local_memory_size,
                                   hipFuncAttributeLocalSizeBytes, function));
  ASSERT_EQ(hipSuccess,
            function_get_attribute(&register_count, hipFuncAttributeNumRegs,
                                   function));
  ASSERT_EQ(hipSuccess,
            function_get_attribute(&maximum_dynamic_shared_memory_size,
                                   hipFuncAttributeMaxDynamicSharedSizeBytes,
                                   function));
  EXPECT_GT(maximum_threads_per_block, 0);
  EXPECT_LE(maximum_threads_per_block, properties.maxThreadsPerBlock);
  ASSERT_GE(fixed_shared_memory_size, 0);
  EXPECT_GE(local_memory_size, 0);
  EXPECT_GT(register_count, 0);
  ASSERT_GE(maximum_dynamic_shared_memory_size, 0);
  EXPECT_EQ(properties.sharedMemPerBlock,
            static_cast<size_t>(fixed_shared_memory_size) +
                maximum_dynamic_shared_memory_size);

  PointerArguments pointer_arguments = {
      static_cast<uint32_t*>(input),
      static_cast<uint32_t*>(output),
  };
  uint32_t scale = 4;
  uint32_t offset = 1;
  void* arguments[] = {&pointer_arguments, &scale, &offset};
  ASSERT_EQ(hipSuccess, module_launch_kernel(function, 1, 1, 1, 1, 1, 1,
                                             /*shared_memory_bytes=*/0,
                                             /*stream=*/nullptr, arguments,
                                             /*extra=*/nullptr));
  ASSERT_EQ(hipSuccess, device_synchronize());
  std::array<uint32_t, 4> actual = {};
  ASSERT_EQ(hipSuccess, hip_memcpy(actual.data(), output, sizeof(actual),
                                   hipMemcpyDeviceToHost));
  EXPECT_EQ((std::array<uint32_t, 4>{5, 9, 13, 17}), actual);

  ASSERT_EQ(hipSuccess, module_unload(module));
  module = nullptr;
  function = nullptr;

  // Reload the same code object after both the first module and its caller-
  // owned image have been destroyed.
  ASSERT_EQ(hipSuccess, load_transient_image(&module));
  ASSERT_EQ(hipSuccess, module_get_function(&function, module,
                                            "hrx_transform_nested_pointers"));

  const std::array<uint32_t, 4> zero_values = {};
  ASSERT_EQ(hipSuccess, hip_memcpy(output, zero_values.data(),
                                   sizeof(zero_values), hipMemcpyHostToDevice));
  struct NativeArguments {
    // Device pointers packed as one by-value argument.
    PointerArguments pointers;
    // Multiplication applied to each input value.
    uint32_t scale;
    // Offset added to each scaled input value.
    uint32_t offset;
    // Caller-provided trailing ABI padding.
    uint8_t trailing_padding[16];
  } native_arguments = {
      /*.pointers=*/pointer_arguments,
      /*.scale=*/3,
      /*.offset=*/10,
      /*.trailing_padding=*/{},
  };
  static_assert(offsetof(NativeArguments, scale) == 2 * sizeof(void*));
  size_t native_arguments_size = sizeof(native_arguments);
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &native_arguments,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &native_arguments_size,
      HIP_LAUNCH_PARAM_END,
  };
  ASSERT_EQ(hipSuccess, module_launch_kernel(function, 1, 1, 1, 1, 1, 1,
                                             /*shared_memory_bytes=*/0,
                                             /*stream=*/nullptr,
                                             /*arguments=*/nullptr, extra));
  ASSERT_EQ(hipSuccess, device_synchronize());
  actual = {};
  ASSERT_EQ(hipSuccess, hip_memcpy(actual.data(), output, sizeof(actual),
                                   hipMemcpyDeviceToHost));
  EXPECT_EQ((std::array<uint32_t, 4>{13, 16, 19, 22}), actual);

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, graph_create(&graph, /*flags=*/0));
  hipGraphNode_t graph_node = nullptr;
  ASSERT_EQ(hipSuccess,
            AddKernelGraphNode(graph_add_kernel_node, graph, function, input,
                               output, &graph_node));
  hipGraphExec_t graph_executable = nullptr;
  ASSERT_EQ(hipSuccess,
            graph_instantiate(&graph_executable, graph,
                              /*error_node=*/nullptr, /*log_buffer=*/nullptr,
                              /*buffer_size=*/0));

  const std::array<uint32_t, 4> graph_expected = {7, 12, 17, 22};
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(hipSuccess,
              hip_memcpy(output, zero_values.data(), sizeof(zero_values),
                         hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, graph_launch(graph_executable, /*stream=*/nullptr));
    ASSERT_EQ(hipSuccess, device_synchronize());
    actual = {};
    ASSERT_EQ(hipSuccess, hip_memcpy(actual.data(), output, sizeof(actual),
                                     hipMemcpyDeviceToHost));
    EXPECT_EQ(graph_expected, actual);
  }

  EXPECT_EQ(hipSuccess, graph_exec_destroy(graph_executable));
  EXPECT_EQ(hipSuccess, graph_destroy(graph));
  EXPECT_EQ(hipSuccess, module_unload(module));
  EXPECT_EQ(hipSuccess, hip_free(output));
  EXPECT_EQ(hipSuccess, hip_free(input));
}

TEST(HipModuleExecutionTest,
     ExtMultiKernelLaunchExecutesEntriesAndHonorsSynchronizationFlags) {
  void* library = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto get_device =
      ResolveHipSymbol<HipGetDeviceFn>(library, "hipGetDevice");
  const auto set_device =
      ResolveHipSymbol<HipSetDeviceFn>(library, "hipSetDevice");
  const auto get_device_count =
      ResolveHipSymbol<HipGetDeviceCountFn>(library, "hipGetDeviceCount");
  const auto get_device_properties =
      ResolveHipSymbol<HipGetDevicePropertiesR0600Fn>(
          library, "hipGetDevicePropertiesR0600");
  const auto hip_malloc = ResolveHipSymbol<HipMallocFn>(library, "hipMalloc");
  const auto hip_free = ResolveHipSymbol<HipFreeFn>(library, "hipFree");
  const auto hip_memcpy = ResolveHipSymbol<HipMemcpyFn>(library, "hipMemcpy");
  const auto stream_create =
      ResolveHipSymbol<HipStreamCreateFn>(library, "hipStreamCreate");
  const auto stream_destroy =
      ResolveHipSymbol<HipStreamDestroyFn>(library, "hipStreamDestroy");
  const auto stream_query =
      ResolveHipSymbol<HipStreamQueryFn>(library, "hipStreamQuery");
  const auto stream_synchronize =
      ResolveHipSymbol<HipStreamSynchronizeFn>(library, "hipStreamSynchronize");
  const auto launch_host_function =
      ResolveHipSymbol<HipLaunchHostFuncFn>(library, "hipLaunchHostFunc");
  const auto host_malloc =
      ResolveHipSymbol<HipHostMallocFn>(library, "hipHostMalloc");
  const auto host_free =
      ResolveHipSymbol<HipHostFreeFn>(library, "hipHostFree");
  const auto host_get_device_pointer =
      ResolveHipSymbol<HipHostGetDevicePointerFn>(library,
                                                  "hipHostGetDevicePointer");
  const auto launch_multi_device =
      ResolveHipSymbol<HipExtLaunchMultiKernelMultiDeviceFn>(
          library, "hipExtLaunchMultiKernelMultiDevice");
  const auto register_fat_binary = ResolveHipSymbol<HipRegisterFatBinaryFn>(
      library, "__hipRegisterFatBinary");
  const auto unregister_fat_binary = ResolveHipSymbol<HipUnregisterFatBinaryFn>(
      library, "__hipUnregisterFatBinary");
  const auto register_function =
      ResolveHipSymbol<HipRegisterFunctionFn>(library, "__hipRegisterFunction");

  ASSERT_NE(nullptr, init);
  ASSERT_NE(nullptr, get_device);
  ASSERT_NE(nullptr, set_device);
  ASSERT_NE(nullptr, get_device_count);
  ASSERT_NE(nullptr, get_device_properties);
  ASSERT_NE(nullptr, hip_malloc);
  ASSERT_NE(nullptr, hip_free);
  ASSERT_NE(nullptr, hip_memcpy);
  ASSERT_NE(nullptr, stream_create);
  ASSERT_NE(nullptr, stream_destroy);
  ASSERT_NE(nullptr, stream_query);
  ASSERT_NE(nullptr, stream_synchronize);
  ASSERT_NE(nullptr, launch_host_function);
  ASSERT_NE(nullptr, host_malloc);
  ASSERT_NE(nullptr, host_free);
  ASSERT_NE(nullptr, host_get_device_pointer);
  ASSERT_NE(nullptr, launch_multi_device);
  ASSERT_NE(nullptr, register_fat_binary);
  ASSERT_NE(nullptr, unregister_fat_binary);
  ASSERT_NE(nullptr, register_function);

  const hipError_t init_result = init(/*flags=*/0);
  if (init_result != hipSuccess) {
    GTEST_SKIP() << "hipInit failed: " << init_result;
  }
  int original_device = 0;
  ASSERT_EQ(hipSuccess, get_device(&original_device));
  ScopedDeviceSelection restore_device = {
      /*.original_device=*/original_device,
      /*.set_device=*/set_device,
  };
  int available_device_count = 0;
  ASSERT_EQ(hipSuccess, get_device_count(&available_device_count));
  ASSERT_GT(available_device_count, 0);

  hipDeviceProp_t reference_properties = {};
  ASSERT_EQ(hipSuccess,
            get_device_properties(&reference_properties, original_device));
  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(reference_properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << reference_properties.gcnArchName;

  ScopedRegistration registration = {
      /*.value=*/register_fat_binary(test_image.file->data),
      /*.unregister=*/unregister_fat_binary,
  };
  ASSERT_NE(nullptr, registration.value);
  char device_function_name[] = "hrx_gated_store_output";
  register_function(
      registration.value,
      reinterpret_cast<const void*>(&MultiDeviceGatedStoreHostStub),
      device_function_name, "hrx_gated_store_output",
      /*thread_limit=*/0,
      /*thread_index=*/nullptr, /*block_index=*/nullptr,
      /*block_dimensions=*/nullptr, /*grid_dimensions=*/nullptr,
      /*shared_memory_size=*/nullptr);

  // Two entries cover the cross-stream transaction when the host has matching
  // devices without scaling a routine CTS case with the machine size. A
  // single-device host still exercises registration, argument packing, enqueue,
  // synchronization, and execution through the extension entry point.
  std::array<MultiDeviceLaunchState, 2> states = {};
  std::array<hipLaunchParams, 2> launches = {};
  size_t launch_count = 0;
  const std::string reference_architecture = reference_properties.gcnArchName;
  for (int device = 0;
       device < available_device_count && launch_count < states.size();
       ++device) {
    hipDeviceProp_t properties = {};
    ASSERT_EQ(hipSuccess, get_device_properties(&properties, device));
    if (reference_architecture != properties.gcnArchName) {
      continue;
    }

    MultiDeviceLaunchState& state = states[launch_count];
    state.device = device;
    state.expected_value = UINT32_C(0xC001) + (uint32_t)launch_count;
    state.set_device = set_device;
    state.stream_destroy = stream_destroy;
    state.free_memory = hip_free;
    state.free_host_memory = host_free;
    ASSERT_EQ(hipSuccess, set_device(device));
    ASSERT_EQ(hipSuccess, stream_create(&state.stream));
    ASSERT_EQ(hipSuccess, hip_malloc(&state.output, sizeof(uint32_t)));
    void* host_gate = nullptr;
    ASSERT_EQ(hipSuccess,
              host_malloc(&host_gate, sizeof(MultiDeviceKernelGate),
                          hipHostMallocMapped | hipHostMallocCoherent));
    state.host_gate = new (host_gate) MultiDeviceKernelGate{};
    state.host_gate->Reset();
    ASSERT_EQ(hipSuccess,
              host_get_device_pointer(&state.device_gate, state.host_gate,
                                      /*flags=*/0));
    state.device_gate_argument = state.device_gate;
    state.output_argument = state.output;
    state.arguments[0] = &state.device_gate_argument;
    state.arguments[1] = &state.output_argument;
    state.arguments[2] = &state.expected_value;
    launches[launch_count] = {
        /*.func=*/reinterpret_cast<void*>(&MultiDeviceGatedStoreHostStub),
        /*.gridDim=*/{1, 1, 1},
        /*.blockDim=*/{1, 1, 1},
        /*.args=*/state.arguments,
        /*.sharedMem=*/0,
        /*.stream=*/state.stream,
    };
    ++launch_count;
  }
  ASSERT_GT(launch_count, 0u);

  struct SynchronizationCase {
    // Human-readable flag combination used in assertion diagnostics.
    const char* name;
    // Synchronization flags passed to the extension entry point.
    unsigned int flags;
  };
  const std::array<SynchronizationCase, 4> synchronization_cases = {{
      {/*.name=*/"pre-and-post", /*.flags=*/0},
      {/*.name=*/"post-only",
       /*.flags=*/hipCooperativeLaunchMultiDeviceNoPreSync},
      {/*.name=*/"pre-only",
       /*.flags=*/hipCooperativeLaunchMultiDeviceNoPostSync},
      {/*.name=*/"neither",
       /*.flags=*/hipCooperativeLaunchMultiDeviceNoPreSync |
           hipCooperativeLaunchMultiDeviceNoPostSync},
  }};

  for (size_t case_ordinal = 0; case_ordinal < synchronization_cases.size();
       ++case_ordinal) {
    const SynchronizationCase& synchronization_case =
        synchronization_cases[case_ordinal];
    const bool has_pre_sync = (synchronization_case.flags &
                               hipCooperativeLaunchMultiDeviceNoPreSync) == 0;
    const bool has_post_sync = (synchronization_case.flags &
                                hipCooperativeLaunchMultiDeviceNoPostSync) == 0;

    for (size_t i = 0; i < launch_count; ++i) {
      MultiDeviceLaunchState& state = states[i];
      state.expected_value = UINT32_C(0xC001) + (uint32_t)i +
                             (uint32_t)case_ordinal * UINT32_C(0x100);
      state.host_gate->Reset();
    }
    HostGate prefix_gate;
    ASSERT_EQ(hipSuccess, set_device(states[0].device));
    const hipError_t enqueue_result = launch_host_function(
        states[0].stream, HostGate::Callback, &prefix_gate);
    if (enqueue_result == hipSuccess) {
      prefix_gate.Arm();
    }
    ASSERT_EQ(hipSuccess, enqueue_result);
    prefix_gate.WaitUntilEntered();

    std::atomic<bool> call_started{false};
    std::atomic<bool> call_returned{false};
    std::atomic<hipError_t> call_result{hipErrorUnknown};
    std::thread launch_thread([&] {
      call_started.store(true, std::memory_order_release);
      call_result.store(launch_multi_device(launches.data(), (int)launch_count,
                                            synchronization_case.flags),
                        std::memory_order_release);
      call_returned.store(true, std::memory_order_release);
    });
    while (!call_started.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    if (has_pre_sync) {
      // The pre barrier must keep the call from returning past a blocked
      // prefix in any participating stream.
      EXPECT_FALSE(call_returned.load(std::memory_order_acquire))
          << synchronization_case.name;
    } else if (!has_post_sync) {
      // With both barriers disabled the call must return while a participating
      // stream is still parked before its launch. Post-only cannot expose this
      // return point because its post barrier intentionally dominates it.
      while (!call_returned.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      EXPECT_EQ(hipErrorNotReady, stream_query(states[0].stream))
          << synchronization_case.name;
    } else {
      // The post barrier still covers the blocked prefix when the pre barrier
      // is disabled.
      EXPECT_FALSE(call_returned.load(std::memory_order_acquire))
          << synchronization_case.name;
    }

    prefix_gate.Open();
    prefix_gate.WaitUntilFinished();

    bool all_kernels_entered = false;
    while (!all_kernels_entered) {
      if (call_returned.load(std::memory_order_acquire) &&
          call_result.load(std::memory_order_acquire) != hipSuccess) {
        break;
      }
      all_kernels_entered = true;
      for (size_t i = 0; i < launch_count; ++i) {
        all_kernels_entered &= __atomic_load_n(&states[i].host_gate->entered,
                                               __ATOMIC_ACQUIRE) != 0;
      }
      if (!all_kernels_entered) {
        std::this_thread::yield();
      }
    }

    if (all_kernels_entered) {
      if (has_post_sync) {
        EXPECT_FALSE(call_returned.load(std::memory_order_acquire))
            << synchronization_case.name;
      } else {
        while (!call_returned.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        for (size_t i = 0; i < launch_count; ++i) {
          EXPECT_EQ(hipErrorNotReady, stream_query(states[i].stream))
              << synchronization_case.name << " stream " << i;
        }
      }
    }

    for (size_t i = 0; i < launch_count; ++i) {
      states[i].host_gate->Release();
    }
    while (!call_returned.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    launch_thread.join();
    EXPECT_EQ(hipSuccess, call_result.load(std::memory_order_acquire))
        << synchronization_case.name;

    for (size_t i = 0; i < launch_count; ++i) {
      MultiDeviceLaunchState& state = states[i];
      ASSERT_EQ(hipSuccess, set_device(state.device));
      if (has_post_sync) {
        EXPECT_EQ(hipSuccess, stream_query(state.stream))
            << synchronization_case.name << " stream " << i;
      }
      ASSERT_EQ(hipSuccess, stream_synchronize(state.stream));
      EXPECT_EQ(hipSuccess, stream_query(state.stream));
      uint32_t actual_value = 0;
      ASSERT_EQ(hipSuccess,
                hip_memcpy(&actual_value, state.output, sizeof(actual_value),
                           hipMemcpyDeviceToHost));
      EXPECT_EQ(state.expected_value, actual_value)
          << synchronization_case.name << " stream " << i;
    }
  }
}

TEST(HipModuleExecutionTest, BlockingPrintfDirectAndGraphReplay) {
  if (hrx_cts_amdgpu_hip_printf_test_kernels_size() == 0) {
    GTEST_SKIP() << "ROCm device libraries are unavailable";
  }

  void* library = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto get_device =
      ResolveHipSymbol<HipGetDeviceFn>(library, "hipGetDevice");
  const auto get_device_properties =
      ResolveHipSymbol<HipGetDevicePropertiesR0600Fn>(
          library, "hipGetDevicePropertiesR0600");
  const auto module_load_data =
      ResolveHipSymbol<HipModuleLoadDataFn>(library, "hipModuleLoadData");
  const auto module_unload =
      ResolveHipSymbol<HipModuleUnloadFn>(library, "hipModuleUnload");
  const auto module_get_function =
      ResolveHipSymbol<HipModuleGetFunctionFn>(library, "hipModuleGetFunction");
  const auto module_launch_kernel = ResolveHipSymbol<HipModuleLaunchKernelFn>(
      library, "hipModuleLaunchKernel");
  const auto device_synchronize =
      ResolveHipSymbol<HipDeviceSynchronizeFn>(library, "hipDeviceSynchronize");
  const auto hip_malloc = ResolveHipSymbol<HipMallocFn>(library, "hipMalloc");
  const auto hip_free = ResolveHipSymbol<HipFreeFn>(library, "hipFree");
  const auto hip_memcpy = ResolveHipSymbol<HipMemcpyFn>(library, "hipMemcpy");
  const auto graph_create =
      ResolveHipSymbol<HipGraphCreateFn>(library, "hipGraphCreate");
  const auto graph_destroy =
      ResolveHipSymbol<HipGraphDestroyFn>(library, "hipGraphDestroy");
  const auto graph_add_kernel_node = ResolveHipSymbol<HipGraphAddKernelNodeFn>(
      library, "hipGraphAddKernelNode");
  const auto graph_instantiate =
      ResolveHipSymbol<HipGraphInstantiateFn>(library, "hipGraphInstantiate");
  const auto graph_launch =
      ResolveHipSymbol<HipGraphLaunchFn>(library, "hipGraphLaunch");
  const auto graph_exec_destroy =
      ResolveHipSymbol<HipGraphExecDestroyFn>(library, "hipGraphExecDestroy");

  ASSERT_NE(nullptr, init);
  ASSERT_NE(nullptr, get_device);
  ASSERT_NE(nullptr, get_device_properties);
  ASSERT_NE(nullptr, module_load_data);
  ASSERT_NE(nullptr, module_unload);
  ASSERT_NE(nullptr, module_get_function);
  ASSERT_NE(nullptr, module_launch_kernel);
  ASSERT_NE(nullptr, device_synchronize);
  ASSERT_NE(nullptr, hip_malloc);
  ASSERT_NE(nullptr, hip_free);
  ASSERT_NE(nullptr, hip_memcpy);
  ASSERT_NE(nullptr, graph_create);
  ASSERT_NE(nullptr, graph_destroy);
  ASSERT_NE(nullptr, graph_add_kernel_node);
  ASSERT_NE(nullptr, graph_instantiate);
  ASSERT_NE(nullptr, graph_launch);
  ASSERT_NE(nullptr, graph_exec_destroy);

  const hipError_t init_result = init(/*flags=*/0);
  if (init_result != hipSuccess) {
    GTEST_SKIP() << "hipInit failed: " << init_result;
  }
  int device = 0;
  ASSERT_EQ(hipSuccess, get_device(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, get_device_properties(&properties, device));

  const hrx_cts::AmdgpuHipPrintfTestImage test_image =
      hrx_cts::FindAmdgpuHipPrintfTestImage(properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HIP printf HSACO for " << properties.gcnArchName;

  std::vector<uint8_t> image(test_image.file->data,
                             test_image.file->data + test_image.file->size);
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, module_load_data(&module, image.data()));
  std::fill(image.begin(), image.end(), uint8_t{0xA5});
  std::vector<uint8_t>().swap(image);

  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            module_get_function(&function, module, "hrx_blocking_printf"));

  hipDeviceptr_t output = nullptr;
  ASSERT_EQ(hipSuccess, hip_malloc(&output, 2 * sizeof(int)));
  const std::array<int, 2> initial_result = {-1, -1};
  ASSERT_EQ(hipSuccess,
            hip_memcpy(output, initial_result.data(), sizeof(initial_result),
                       hipMemcpyHostToDevice));

  uint32_t value = 42;
  hipDeviceptr_t result = output;
  void* arguments[] = {&value, &result};
  ::testing::internal::CaptureStdout();
  const hipError_t launch_result =
      module_launch_kernel(function, 1, 1, 1, 1, 1, 1,
                           /*shared_memory_bytes=*/0, /*stream=*/nullptr,
                           arguments, /*extra=*/nullptr);
  const hipError_t synchronize_result =
      launch_result == hipSuccess ? device_synchronize() : launch_result;
  std::fflush(stdout);
  const std::string direct_output = ::testing::internal::GetCapturedStdout();
  ASSERT_EQ(hipSuccess, launch_result);
  ASSERT_EQ(hipSuccess, synchronize_result);
  EXPECT_EQ("hrx cts printf value=42\n", direct_output);

  constexpr int kExpectedPrintfLength = sizeof("hrx cts printf value=42\n") - 1;
  constexpr int kPostPrintfMarker = 0xC0FFEE;
  std::array<int, 2> actual_result = {};
  ASSERT_EQ(hipSuccess,
            hip_memcpy(actual_result.data(), output, sizeof(actual_result),
                       hipMemcpyDeviceToHost));
  EXPECT_EQ(kExpectedPrintfLength, actual_result[0]);
  EXPECT_EQ(kPostPrintfMarker, actual_result[1]);

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, graph_create(&graph, /*flags=*/0));
  hipGraphNode_t graph_node = nullptr;
  ASSERT_EQ(hipSuccess, AddPrintfGraphNode(graph_add_kernel_node, graph,
                                           function, output, &graph_node));
  hipGraphExec_t graph_executable = nullptr;
  ASSERT_EQ(hipSuccess,
            graph_instantiate(&graph_executable, graph,
                              /*error_node=*/nullptr, /*log_buffer=*/nullptr,
                              /*buffer_size=*/0));

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(hipSuccess,
              hip_memcpy(output, initial_result.data(), sizeof(initial_result),
                         hipMemcpyHostToDevice));
    ::testing::internal::CaptureStdout();
    const hipError_t graph_launch_result =
        graph_launch(graph_executable, /*stream=*/nullptr);
    const hipError_t graph_synchronize_result =
        graph_launch_result == hipSuccess ? device_synchronize()
                                          : graph_launch_result;
    std::fflush(stdout);
    const std::string graph_output = ::testing::internal::GetCapturedStdout();
    ASSERT_EQ(hipSuccess, graph_launch_result);
    ASSERT_EQ(hipSuccess, graph_synchronize_result);
    EXPECT_EQ("hrx cts printf value=42\n", graph_output);

    actual_result = {};
    ASSERT_EQ(hipSuccess,
              hip_memcpy(actual_result.data(), output, sizeof(actual_result),
                         hipMemcpyDeviceToHost));
    EXPECT_EQ(kExpectedPrintfLength, actual_result[0]);
    EXPECT_EQ(kPostPrintfMarker, actual_result[1]);
  }

  EXPECT_EQ(hipSuccess, graph_exec_destroy(graph_executable));
  EXPECT_EQ(hipSuccess, graph_destroy(graph));

  hipFunction_t many_workitems_function = nullptr;
  ASSERT_EQ(hipSuccess,
            module_get_function(&many_workitems_function, module,
                                "hrx_blocking_printf_many_workitems"));
  EXPECT_EQ(hipSuccess, hip_free(output));
  // One workgroup exercises concurrent fragmented calls from multiple
  // resident waves while keeping routine CTS execution bounded.
  constexpr uint32_t kBlockSize = 256;
  constexpr uint32_t kWorkitemCount = kBlockSize;
  static_assert(kWorkitemCount % kBlockSize == 0);
  ASSERT_EQ(hipSuccess, hip_malloc(&output, kWorkitemCount * sizeof(int)));
  uint32_t workgroup_size = kBlockSize;
  uint32_t workitem_count = kWorkitemCount;
  result = output;
  void* many_workitems_arguments[] = {&workgroup_size, &workitem_count,
                                      &result};
  ::testing::internal::CaptureStdout();
  const hipError_t many_workitems_launch_result = module_launch_kernel(
      many_workitems_function, kWorkitemCount / kBlockSize, 1, 1, kBlockSize, 1,
      1, /*shared_memory_bytes=*/0, /*stream=*/nullptr,
      many_workitems_arguments, /*extra=*/nullptr);
  const hipError_t many_workitems_synchronize_result =
      many_workitems_launch_result == hipSuccess ? device_synchronize()
                                                 : many_workitems_launch_result;
  std::fflush(stdout);
  const std::string many_workitems_output =
      ::testing::internal::GetCapturedStdout();
  ASSERT_EQ(hipSuccess, many_workitems_launch_result);
  ASSERT_EQ(hipSuccess, many_workitems_synchronize_result);
  EXPECT_EQ(kWorkitemCount, many_workitems_output.size());
  EXPECT_TRUE(std::all_of(many_workitems_output.begin(),
                          many_workitems_output.end(),
                          [](char value) { return value == '7'; }));

  std::vector<int> many_workitems_results(kWorkitemCount);
  ASSERT_EQ(hipSuccess, hip_memcpy(many_workitems_results.data(), output,
                                   many_workitems_results.size() * sizeof(int),
                                   hipMemcpyDeviceToHost));
  EXPECT_TRUE(std::all_of(many_workitems_results.begin(),
                          many_workitems_results.end(),
                          [](int value) { return value == 1; }));

  EXPECT_EQ(hipSuccess, hip_free(output));
  EXPECT_EQ(hipSuccess, module_unload(module));
}

TEST(HipModuleExecutionTest, CooperativeLaunchPreservesStreamAndGraphOrdering) {
  if (hrx_cts_amdgpu_hip_cooperative_test_kernels_size() == 0) {
    GTEST_SKIP() << "ROCm device libraries are unavailable";
  }

  void* library = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto get_device =
      ResolveHipSymbol<HipGetDeviceFn>(library, "hipGetDevice");
  const auto get_device_properties =
      ResolveHipSymbol<HipGetDevicePropertiesR0600Fn>(
          library, "hipGetDevicePropertiesR0600");
  const auto module_load_data =
      ResolveHipSymbol<HipModuleLoadDataFn>(library, "hipModuleLoadData");
  const auto module_unload =
      ResolveHipSymbol<HipModuleUnloadFn>(library, "hipModuleUnload");
  const auto module_get_function =
      ResolveHipSymbol<HipModuleGetFunctionFn>(library, "hipModuleGetFunction");
  const auto module_launch_cooperative_kernel =
      ResolveHipSymbol<HipModuleLaunchCooperativeKernelFn>(
          library, "hipModuleLaunchCooperativeKernel");
  const auto launch_kernel_ex =
      ResolveHipSymbol<HipLaunchKernelExCFn>(library, "hipLaunchKernelExC");
  const auto driver_launch_kernel_ex =
      ResolveHipSymbol<HipDrvLaunchKernelExFn>(library, "hipDrvLaunchKernelEx");
  const auto module_occupancy =
      ResolveHipSymbol<HipModuleOccupancyMaxActiveBlocksPerMultiprocessorFn>(
          library, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
  const auto hip_malloc = ResolveHipSymbol<HipMallocFn>(library, "hipMalloc");
  const auto hip_free = ResolveHipSymbol<HipFreeFn>(library, "hipFree");
  const auto hip_memcpy_async =
      ResolveHipSymbol<HipMemcpyAsyncFn>(library, "hipMemcpyAsync");
  const auto hip_memset_async =
      ResolveHipSymbol<HipMemsetAsyncFn>(library, "hipMemsetAsync");
  const auto stream_create =
      ResolveHipSymbol<HipStreamCreateFn>(library, "hipStreamCreate");
  const auto stream_destroy =
      ResolveHipSymbol<HipStreamDestroyFn>(library, "hipStreamDestroy");
  const auto stream_synchronize =
      ResolveHipSymbol<HipStreamSynchronizeFn>(library, "hipStreamSynchronize");
  const auto device_get_resource = ResolveHipSymbol<HipDeviceGetDevResourceFn>(
      library, "hipDeviceGetDevResource");
  const auto device_get_execution_context =
      ResolveHipSymbol<HipDeviceGetExecutionCtxFn>(library,
                                                   "hipDeviceGetExecutionCtx");
  const auto generate_resource_descriptor =
      ResolveHipSymbol<HipDevResourceGenerateDescFn>(
          library, "hipDevResourceGenerateDesc");
  const auto green_context_create =
      ResolveHipSymbol<HipGreenCtxCreateFn>(library, "hipGreenCtxCreate");
  const auto execution_context_destroy =
      ResolveHipSymbol<HipExecutionCtxDestroyFn>(library,
                                                 "hipExecutionCtxDestroy");
  const auto execution_context_stream_create =
      ResolveHipSymbol<HipExecutionCtxStreamCreateFn>(
          library, "hipExecutionCtxStreamCreate");
  const auto stream_begin_capture = ResolveHipSymbol<HipStreamBeginCaptureFn>(
      library, "hipStreamBeginCapture");
  const auto stream_end_capture =
      ResolveHipSymbol<HipStreamEndCaptureFn>(library, "hipStreamEndCapture");
  const auto graph_get_nodes =
      ResolveHipSymbol<HipGraphGetNodesFn>(library, "hipGraphGetNodes");
  const auto graph_node_get_type =
      ResolveHipSymbol<HipGraphNodeGetTypeFn>(library, "hipGraphNodeGetType");
  const auto graph_destroy =
      ResolveHipSymbol<HipGraphDestroyFn>(library, "hipGraphDestroy");
  const auto graph_instantiate =
      ResolveHipSymbol<HipGraphInstantiateFn>(library, "hipGraphInstantiate");
  const auto graph_launch =
      ResolveHipSymbol<HipGraphLaunchFn>(library, "hipGraphLaunch");
  const auto graph_exec_destroy =
      ResolveHipSymbol<HipGraphExecDestroyFn>(library, "hipGraphExecDestroy");
  const auto graph_exec_kernel_node_set_params =
      ResolveHipSymbol<HipGraphExecKernelNodeSetParamsFn>(
          library, "hipGraphExecKernelNodeSetParams");
  const auto register_fat_binary = ResolveHipSymbol<HipRegisterFatBinaryFn>(
      library, "__hipRegisterFatBinary");
  const auto unregister_fat_binary = ResolveHipSymbol<HipUnregisterFatBinaryFn>(
      library, "__hipUnregisterFatBinary");
  const auto register_function =
      ResolveHipSymbol<HipRegisterFunctionFn>(library, "__hipRegisterFunction");

  ASSERT_NE(nullptr, init);
  ASSERT_NE(nullptr, get_device);
  ASSERT_NE(nullptr, get_device_properties);
  ASSERT_NE(nullptr, module_load_data);
  ASSERT_NE(nullptr, module_unload);
  ASSERT_NE(nullptr, module_get_function);
  ASSERT_NE(nullptr, module_launch_cooperative_kernel);
  ASSERT_NE(nullptr, launch_kernel_ex);
  ASSERT_NE(nullptr, driver_launch_kernel_ex);
  ASSERT_NE(nullptr, module_occupancy);
  ASSERT_NE(nullptr, hip_malloc);
  ASSERT_NE(nullptr, hip_free);
  ASSERT_NE(nullptr, hip_memcpy_async);
  ASSERT_NE(nullptr, hip_memset_async);
  ASSERT_NE(nullptr, stream_create);
  ASSERT_NE(nullptr, stream_destroy);
  ASSERT_NE(nullptr, stream_synchronize);
  ASSERT_NE(nullptr, device_get_resource);
  ASSERT_NE(nullptr, device_get_execution_context);
  ASSERT_NE(nullptr, generate_resource_descriptor);
  ASSERT_NE(nullptr, green_context_create);
  ASSERT_NE(nullptr, execution_context_destroy);
  ASSERT_NE(nullptr, execution_context_stream_create);
  ASSERT_NE(nullptr, stream_begin_capture);
  ASSERT_NE(nullptr, stream_end_capture);
  ASSERT_NE(nullptr, graph_get_nodes);
  ASSERT_NE(nullptr, graph_node_get_type);
  ASSERT_NE(nullptr, graph_destroy);
  ASSERT_NE(nullptr, graph_instantiate);
  ASSERT_NE(nullptr, graph_launch);
  ASSERT_NE(nullptr, graph_exec_destroy);
  ASSERT_NE(nullptr, graph_exec_kernel_node_set_params);
  ASSERT_NE(nullptr, register_fat_binary);
  ASSERT_NE(nullptr, unregister_fat_binary);
  ASSERT_NE(nullptr, register_function);

  const hipError_t init_result = init(/*flags=*/0);
  if (init_result != hipSuccess) {
    GTEST_SKIP() << "hipInit failed: " << init_result;
  }
  int device = 0;
  ASSERT_EQ(hipSuccess, get_device(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, get_device_properties(&properties, device));
  if (!properties.cooperativeLaunch) {
    GTEST_SKIP() << "device does not support cooperative launch";
  }

  const hrx_cts::AmdgpuHipCooperativeTestImage test_image =
      hrx_cts::FindAmdgpuHipCooperativeTestImage(properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded cooperative HSACO for " << properties.gcnArchName;

  ScopedRegistration registration = {
      /*.value=*/register_fat_binary(test_image.file->data),
      /*.unregister=*/unregister_fat_binary,
  };
  ASSERT_NE(nullptr, registration.value);
  char device_function_name[] = "hrx_cooperative_grid_sync";
  register_function(registration.value,
                    reinterpret_cast<const void*>(&CooperativeGridSyncHostStub),
                    device_function_name, "hrx_cooperative_grid_sync",
                    /*thread_limit=*/0, /*thread_index=*/nullptr,
                    /*block_index=*/nullptr,
                    /*block_dimensions=*/nullptr, /*grid_dimensions=*/nullptr,
                    /*shared_memory_size=*/nullptr);

  std::vector<uint8_t> image(test_image.file->data,
                             test_image.file->data + test_image.file->size);
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, module_load_data(&module, image.data()));
  std::fill(image.begin(), image.end(), uint8_t{0xA5});
  std::vector<uint8_t>().swap(image);
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess, module_get_function(&function, module,
                                            "hrx_cooperative_grid_sync"));

  constexpr uint32_t kWorkgroupCount = 4;
  constexpr uint32_t kWorkgroupSize = 64;
  hipDeviceptr_t scratch = nullptr;
  hipDeviceptr_t output = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_malloc(&scratch, kWorkgroupCount * sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess,
            hip_malloc(&output, kWorkgroupCount * sizeof(uint32_t)));
  hipStream_t direct_stream = nullptr;
  ASSERT_EQ(hipSuccess, stream_create(&direct_stream));
  ScopedStream direct_stream_guard(direct_stream,
                                   StreamDeleter{stream_destroy});
  hipStream_t graph_stream = nullptr;
  ASSERT_EQ(hipSuccess, stream_create(&graph_stream));
  ScopedStream graph_stream_guard(graph_stream, StreamDeleter{stream_destroy});

  hipExecutionCtx_t primary_context = nullptr;
  ASSERT_EQ(hipSuccess, device_get_execution_context(&primary_context, device));
  hipStream_t primary_context_stream = nullptr;
  ASSERT_EQ(hipSuccess, execution_context_stream_create(
                            &primary_context_stream, primary_context,
                            hipStreamDefault, /*priority=*/0));
  ScopedStream primary_context_stream_guard(primary_context_stream,
                                            StreamDeleter{stream_destroy});

  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess,
            device_get_resource(device, &full_resource, hipDevResourceTypeSm));
  hipDevResourceDesc_t full_resource_descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            generate_resource_descriptor(&full_resource_descriptor,
                                         &full_resource, /*resource_count=*/1));
  hipExecutionCtx_t full_resource_context = nullptr;
  ASSERT_EQ(hipSuccess, green_context_create(&full_resource_context,
                                             full_resource_descriptor, device,
                                             /*flags=*/0));
  ScopedExecutionContext full_resource_context_guard(
      full_resource_context,
      ExecutionContextDeleter{execution_context_destroy});
  hipStream_t full_resource_stream = nullptr;
  ASSERT_EQ(hipSuccess, execution_context_stream_create(
                            &full_resource_stream, full_resource_context,
                            hipStreamDefault, /*priority=*/0));
  ScopedStream full_resource_stream_guard(full_resource_stream,
                                          StreamDeleter{stream_destroy});

  auto launch = [&](hipStream_t stream, uint32_t incarnation) {
    hipDeviceptr_t scratch_argument = scratch;
    hipDeviceptr_t output_argument = output;
    uint32_t workgroup_count = kWorkgroupCount;
    void* arguments[] = {&scratch_argument, &output_argument, &incarnation,
                         &workgroup_count};
    return module_launch_cooperative_kernel(
        function, kWorkgroupCount, 1, 1, kWorkgroupSize, 1, 1,
        /*shared_memory_bytes=*/0, stream, arguments);
  };
  hipLaunchAttribute cooperative_attribute = {};
  cooperative_attribute.id = hipLaunchAttributeCooperative;
  cooperative_attribute.val.cooperative = 1;
  auto launch_runtime_extended = [&](hipStream_t stream, uint32_t incarnation) {
    hipDeviceptr_t scratch_argument = scratch;
    hipDeviceptr_t output_argument = output;
    uint32_t workgroup_count = kWorkgroupCount;
    void* arguments[] = {&scratch_argument, &output_argument, &incarnation,
                         &workgroup_count};
    hipLaunchConfig_t config = {
        /*.gridDim=*/{kWorkgroupCount, 1, 1},
        /*.blockDim=*/{kWorkgroupSize, 1, 1},
        /*.dynamicSmemBytes=*/0,
        /*.stream=*/stream,
        /*.attrs=*/&cooperative_attribute,
        /*.numAttrs=*/1,
    };
    return launch_kernel_ex(
        &config, reinterpret_cast<const void*>(&CooperativeGridSyncHostStub),
        arguments);
  };
  auto launch_driver_extended = [&](hipStream_t stream, uint32_t incarnation) {
    hipDeviceptr_t scratch_argument = scratch;
    hipDeviceptr_t output_argument = output;
    uint32_t workgroup_count = kWorkgroupCount;
    void* arguments[] = {&scratch_argument, &output_argument, &incarnation,
                         &workgroup_count};
    HIP_LAUNCH_CONFIG config = {
        /*.gridDimX=*/kWorkgroupCount,
        /*.gridDimY=*/1,
        /*.gridDimZ=*/1,
        /*.blockDimX=*/kWorkgroupSize,
        /*.blockDimY=*/1,
        /*.blockDimZ=*/1,
        /*.sharedMemBytes=*/0,
        /*.hStream=*/stream,
        /*.attrs=*/&cooperative_attribute,
        /*.numAttrs=*/1,
    };
    return driver_launch_kernel_ex(&config, function, arguments,
                                   /*extra=*/nullptr);
  };
  auto launch_driver_extended_prepacked = [&](hipStream_t stream,
                                              uint32_t incarnation) {
    struct NativeArguments {
      uint32_t* scratch;
      uint32_t* output;
      uint32_t incarnation;
      uint32_t workgroup_count;
    } arguments = {
        /*.scratch=*/static_cast<uint32_t*>(scratch),
        /*.output=*/static_cast<uint32_t*>(output),
        /*.incarnation=*/incarnation,
        /*.workgroup_count=*/kWorkgroupCount,
    };
    static_assert(offsetof(NativeArguments, output) == sizeof(void*));
    static_assert(offsetof(NativeArguments, incarnation) == 2 * sizeof(void*));
    static_assert(sizeof(NativeArguments) ==
                  2 * sizeof(void*) + 2 * sizeof(uint32_t));
    size_t arguments_size = sizeof(arguments);
    void* extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        &arguments,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &arguments_size,
        HIP_LAUNCH_PARAM_END,
    };
    HIP_LAUNCH_CONFIG config = {
        /*.gridDimX=*/kWorkgroupCount,
        /*.gridDimY=*/1,
        /*.gridDimZ=*/1,
        /*.blockDimX=*/kWorkgroupSize,
        /*.blockDimY=*/1,
        /*.blockDimZ=*/1,
        /*.sharedMemBytes=*/0,
        /*.hStream=*/stream,
        /*.attrs=*/&cooperative_attribute,
        /*.numAttrs=*/1,
    };
    return driver_launch_kernel_ex(&config, function,
                                   /*arguments=*/nullptr, extra);
  };
  auto expected_sum = [](uint32_t incarnation) {
    return kWorkgroupCount * incarnation +
           (kWorkgroupCount * (kWorkgroupCount - 1)) / 2;
  };

  std::array<uint32_t, kWorkgroupCount> actual = {};
  const auto verify_extended_launch = [&](uint32_t incarnation,
                                          const auto& launch_extended) {
    ASSERT_EQ(hipSuccess,
              hip_memset_async(output, 0, kWorkgroupCount * sizeof(uint32_t),
                               direct_stream));
    ASSERT_EQ(hipSuccess, launch_extended(direct_stream, incarnation));
    actual.fill(UINT32_MAX);
    ASSERT_EQ(hipSuccess,
              hip_memcpy_async(actual.data(), output, sizeof(actual),
                               hipMemcpyDeviceToHost, direct_stream));
    ASSERT_EQ(hipSuccess, stream_synchronize(direct_stream));
    std::array<uint32_t, kWorkgroupCount> expected = {};
    expected.fill(expected_sum(incarnation));
    EXPECT_EQ(expected, actual);
  };
  verify_extended_launch(/*incarnation=*/25, launch_runtime_extended);
  verify_extended_launch(/*incarnation=*/50, launch_driver_extended);
  verify_extended_launch(/*incarnation=*/75, launch_driver_extended_prepacked);

  constexpr uint32_t kDirectIncarnation = 100;
  std::array<uint32_t, kWorkgroupCount> direct_expected = {};
  direct_expected.fill(expected_sum(kDirectIncarnation));
  // These three operations execute on ordinary, cooperative, and ordinary
  // queue realizations. Their shared stream timeline must order both queue
  // transitions without relying on either hardware queue being FIFO.
  ASSERT_EQ(hipSuccess,
            hip_memset_async(output, 0, kWorkgroupCount * sizeof(uint32_t),
                             direct_stream));
  ASSERT_EQ(hipSuccess, launch(direct_stream, kDirectIncarnation));
  actual = {};
  ASSERT_EQ(hipSuccess, hip_memcpy_async(actual.data(), output, sizeof(actual),
                                         hipMemcpyDeviceToHost, direct_stream));
  ASSERT_EQ(hipSuccess, stream_synchronize(direct_stream));
  EXPECT_EQ(direct_expected, actual);

  constexpr uint32_t kPrimaryContextIncarnation = 125;
  std::array<uint32_t, kWorkgroupCount> primary_context_expected = {};
  primary_context_expected.fill(expected_sum(kPrimaryContextIncarnation));
  ASSERT_EQ(hipSuccess,
            hip_memset_async(output, 0, kWorkgroupCount * sizeof(uint32_t),
                             primary_context_stream));
  ASSERT_EQ(hipSuccess,
            launch(primary_context_stream, kPrimaryContextIncarnation));
  actual.fill(UINT32_MAX);
  ASSERT_EQ(hipSuccess,
            hip_memcpy_async(actual.data(), output, sizeof(actual),
                             hipMemcpyDeviceToHost, primary_context_stream));
  ASSERT_EQ(hipSuccess, stream_synchronize(primary_context_stream));
  EXPECT_EQ(primary_context_expected, actual);

  constexpr uint32_t kFullResourceIncarnation = 150;
  std::array<uint32_t, kWorkgroupCount> full_resource_expected = {};
  full_resource_expected.fill(expected_sum(kFullResourceIncarnation));
  ASSERT_EQ(hipSuccess,
            hip_memset_async(output, 0, kWorkgroupCount * sizeof(uint32_t),
                             full_resource_stream));
  ASSERT_EQ(hipSuccess, launch(full_resource_stream, kFullResourceIncarnation));
  actual.fill(UINT32_MAX);
  ASSERT_EQ(hipSuccess,
            hip_memcpy_async(actual.data(), output, sizeof(actual),
                             hipMemcpyDeviceToHost, full_resource_stream));
  ASSERT_EQ(hipSuccess, stream_synchronize(full_resource_stream));
  EXPECT_EQ(full_resource_expected, actual);

  constexpr uint32_t kGraphIncarnation = 200;
  std::array<uint32_t, kWorkgroupCount> graph_expected = {};
  graph_expected.fill(expected_sum(kGraphIncarnation));
  // Capture creates a recordable memset partition followed by an extended
  // cooperative dispatch partition. Replaying twice proves each launch gets
  // fresh grid-synchronization state and rejoins the launching stream tail.
  ASSERT_EQ(hipSuccess,
            stream_begin_capture(graph_stream, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess,
            hip_memset_async(output, 0, kWorkgroupCount * sizeof(uint32_t),
                             graph_stream));
  ASSERT_EQ(hipSuccess,
            launch_runtime_extended(graph_stream, kGraphIncarnation));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, stream_end_capture(graph_stream, &graph));
  ASSERT_NE(nullptr, graph);
  hipGraphExec_t graph_executable = nullptr;
  ASSERT_EQ(hipSuccess,
            graph_instantiate(&graph_executable, graph,
                              /*error_node=*/nullptr, /*log_buffer=*/nullptr,
                              /*buffer_size=*/0));

  for (int replay = 0; replay < 2; ++replay) {
    actual.fill(UINT32_MAX);
    ASSERT_EQ(hipSuccess, graph_launch(graph_executable, graph_stream));
    ASSERT_EQ(hipSuccess,
              hip_memcpy_async(actual.data(), output, sizeof(actual),
                               hipMemcpyDeviceToHost, graph_stream));
    ASSERT_EQ(hipSuccess, stream_synchronize(graph_stream));
    EXPECT_EQ(graph_expected, actual);
  }

  size_t graph_node_count = 0;
  ASSERT_EQ(hipSuccess,
            graph_get_nodes(graph, /*nodes=*/nullptr, &graph_node_count));
  ASSERT_EQ(2u, graph_node_count);
  std::vector<hipGraphNode_t> graph_nodes(graph_node_count);
  ASSERT_EQ(hipSuccess,
            graph_get_nodes(graph, graph_nodes.data(), &graph_node_count));
  hipGraphNode_t kernel_node = nullptr;
  for (hipGraphNode_t node : graph_nodes) {
    hipGraphNodeType type = hipGraphNodeTypeCount;
    ASSERT_EQ(hipSuccess, graph_node_get_type(node, &type));
    if (type == hipGraphNodeTypeKernel) {
      ASSERT_EQ(nullptr, kernel_node);
      kernel_node = node;
    }
  }
  ASSERT_NE(nullptr, kernel_node);

  int active_blocks_per_multiprocessor = 0;
  ASSERT_EQ(hipSuccess, module_occupancy(&active_blocks_per_multiprocessor,
                                         function, kWorkgroupSize,
                                         /*dynamic_shared_memory_bytes=*/0));
  ASSERT_GT(active_blocks_per_multiprocessor, 0);
  ASSERT_GT(properties.multiProcessorCount, 0);
  const uint64_t maximum_resident_grid =
      static_cast<uint64_t>(active_blocks_per_multiprocessor) *
      static_cast<uint64_t>(properties.multiProcessorCount);
  ASSERT_LT(maximum_resident_grid,
            static_cast<uint64_t>(properties.maxGridSize[0]));
  ASSERT_LT(maximum_resident_grid, static_cast<uint64_t>(UINT32_MAX));
  const uint32_t oversized_grid =
      static_cast<uint32_t>(maximum_resident_grid + 1);

  hipDeviceptr_t oversized_scratch = nullptr;
  hipDeviceptr_t oversized_output = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_malloc(&oversized_scratch, oversized_grid * sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess,
            hip_malloc(&oversized_output, oversized_grid * sizeof(uint32_t)));

  hipDeviceptr_t updated_scratch_argument = oversized_scratch;
  hipDeviceptr_t updated_output_argument = oversized_output;
  uint32_t updated_incarnation = 250;
  uint32_t updated_workgroup_count = oversized_grid;
  void* updated_arguments[] = {&updated_scratch_argument,
                               &updated_output_argument, &updated_incarnation,
                               &updated_workgroup_count};
  const hipKernelNodeParams oversized_params = {
      /*.blockDim=*/{kWorkgroupSize, 1, 1},
      /*.extra=*/nullptr,
      /*.func=*/reinterpret_cast<void*>(&CooperativeGridSyncHostStub),
      /*.gridDim=*/{oversized_grid, 1, 1},
      /*.kernelParams=*/updated_arguments,
      /*.sharedMemBytes=*/0,
  };
  ASSERT_EQ(hipSuccess, graph_exec_kernel_node_set_params(
                            graph_executable, kernel_node, &oversized_params));

  // The captured memset is an observable prefix ahead of the cooperative
  // node. Rejection must happen before either node is submitted.
  ASSERT_EQ(hipSuccess,
            hip_memset_async(output, 0xA5, kWorkgroupCount * sizeof(uint32_t),
                             graph_stream));
  ASSERT_EQ(hipSuccess, stream_synchronize(graph_stream));
  EXPECT_EQ(hipErrorCooperativeLaunchTooLarge,
            graph_launch(graph_executable, graph_stream));
  actual.fill(0);
  ASSERT_EQ(hipSuccess, hip_memcpy_async(actual.data(), output, sizeof(actual),
                                         hipMemcpyDeviceToHost, graph_stream));
  ASSERT_EQ(hipSuccess, stream_synchronize(graph_stream));
  std::array<uint32_t, kWorkgroupCount> untouched_prefix = {};
  untouched_prefix.fill(UINT32_C(0xA5A5A5A5));
  EXPECT_EQ(untouched_prefix, actual);
  EXPECT_EQ(hipSuccess, hip_free(oversized_output));
  EXPECT_EQ(hipSuccess, hip_free(oversized_scratch));

  EXPECT_EQ(hipSuccess, graph_exec_destroy(graph_executable));
  EXPECT_EQ(hipSuccess, graph_destroy(graph));
  EXPECT_EQ(hipSuccess, hip_free(output));
  EXPECT_EQ(hipSuccess, hip_free(scratch));
  EXPECT_EQ(hipSuccess, module_unload(module));
}

}  // namespace
