// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

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
using HipGetDevicePropertiesFn = hipError_t (*)(hipDeviceProp_t* properties,
                                                int device);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t* module,
                                           const void* image);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t module);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipModule_t module,
                                              const char* name);
using HipModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments, void** extra);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipMallocFn = hipError_t (*)(hipDeviceptr_t* pointer, size_t size);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t pointer);
using HipMemcpyFn = hipError_t (*)(void* target, const void* source,
                                   size_t size, hipMemcpyKind kind);
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

TEST(HipModuleExecutionTest, OwnsLoadAndGraphInputsAcrossReloads) {
  void* library = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto get_device =
      ResolveHipSymbol<HipGetDeviceFn>(library, "hipGetDevice");
  const auto get_device_properties = ResolveHipSymbol<HipGetDevicePropertiesFn>(
      library, "hipGetDeviceProperties");
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

}  // namespace
