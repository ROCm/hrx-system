// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>

#include "binding/hip/api.h"
#include "common/internal.h"
#include "iree/testing/gtest.h"

namespace {

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return "libamdhip64.so";
#endif
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipLaunchKernelFn = hipError_t (*)(const void* function, dim3 grid_dim,
                                         dim3 block_dim, void** arguments,
                                         size_t shared_memory_bytes,
                                         hipStream_t stream);
using HipExtLaunchKernelFn = hipError_t (*)(const void* function, dim3 grid_dim,
                                            dim3 block_dim, void** arguments,
                                            size_t shared_memory_bytes,
                                            hipStream_t stream,
                                            hipEvent_t start_event,
                                            hipEvent_t stop_event, int flags);
using HipModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments, void** extra);
using HipFuncGetAttributeFn = hipError_t (*)(int* value,
                                             hipFuncAttribute_t attribute,
                                             hipFunction_t function);
using HipFuncSetAttributeFn = hipError_t (*)(hipFunction_t function,
                                             hipFuncAttribute_t attribute,
                                             int value);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddKernelNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, const void* params);
using HipGraphKernelNodeGetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                     void* params);
using HipGraphKernelNodeSetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                     const void* params);

// Owns an RTLD_LOCAL HIP runtime instance and the entry points exercised by
// this test. All calls use the loaded library instead of a link-time runtime.
struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;
  // Creates the stream used by immediate launch entry points.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys the stream used by immediate launch entry points.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Launches a registered runtime kernel.
  HipLaunchKernelFn launch_kernel = nullptr;
  // Launches a registered runtime kernel with extended launch arguments.
  HipExtLaunchKernelFn ext_launch_kernel = nullptr;
  // Launches a module kernel with prepacked or pointer-array arguments.
  HipModuleLaunchKernelFn module_launch_kernel = nullptr;
  // Queries a cached function compatibility attribute.
  HipFuncGetAttributeFn function_get_attribute = nullptr;
  // Updates a mutable function compatibility attribute.
  HipFuncSetAttributeFn function_set_attribute = nullptr;
  // Creates a graph template.
  HipGraphCreateFn graph_create = nullptr;
  // Destroys a graph template.
  HipGraphDestroyFn graph_destroy = nullptr;
  // Adds a kernel node to a graph template.
  HipGraphAddKernelNodeFn graph_add_kernel_node = nullptr;
  // Reads the parameters retained by a graph kernel node.
  HipGraphKernelNodeGetParamsFn graph_kernel_node_get_params = nullptr;
  // Replaces the parameters retained by a graph kernel node.
  HipGraphKernelNodeSetParamsFn graph_kernel_node_set_params = nullptr;
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class HipLaunchValidationApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      api_.library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
      if (!api_.library) {
        GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                     << dlerror();
      }

      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.stream_create =
          ResolveHipSymbol<HipStreamCreateFn>(api_.library, "hipStreamCreate");
      api_.stream_destroy = ResolveHipSymbol<HipStreamDestroyFn>(
          api_.library, "hipStreamDestroy");
      api_.launch_kernel =
          ResolveHipSymbol<HipLaunchKernelFn>(api_.library, "hipLaunchKernel");
      api_.ext_launch_kernel = ResolveHipSymbol<HipExtLaunchKernelFn>(
          api_.library, "hipExtLaunchKernel");
      api_.module_launch_kernel = ResolveHipSymbol<HipModuleLaunchKernelFn>(
          api_.library, "hipModuleLaunchKernel");
      api_.function_get_attribute = ResolveHipSymbol<HipFuncGetAttributeFn>(
          api_.library, "hipFuncGetAttribute");
      api_.function_set_attribute = ResolveHipSymbol<HipFuncSetAttributeFn>(
          api_.library, "hipFuncSetAttribute");
      api_.graph_create =
          ResolveHipSymbol<HipGraphCreateFn>(api_.library, "hipGraphCreate");
      api_.graph_destroy =
          ResolveHipSymbol<HipGraphDestroyFn>(api_.library, "hipGraphDestroy");
      api_.graph_add_kernel_node = ResolveHipSymbol<HipGraphAddKernelNodeFn>(
          api_.library, "hipGraphAddKernelNode");
      api_.graph_kernel_node_get_params =
          ResolveHipSymbol<HipGraphKernelNodeGetParamsFn>(
              api_.library, "hipGraphKernelNodeGetParams");
      api_.graph_kernel_node_set_params =
          ResolveHipSymbol<HipGraphKernelNodeSetParamsFn>(
              api_.library, "hipGraphKernelNodeSetParams");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.launch_kernel);
    ASSERT_NE(nullptr, api_.ext_launch_kernel);
    ASSERT_NE(nullptr, api_.module_launch_kernel);
    ASSERT_NE(nullptr, api_.function_get_attribute);
    ASSERT_NE(nullptr, api_.function_set_attribute);
    ASSERT_NE(nullptr, api_.graph_create);
    ASSERT_NE(nullptr, api_.graph_destroy);
    ASSERT_NE(nullptr, api_.graph_add_kernel_node);
    ASSERT_NE(nullptr, api_.graph_kernel_node_get_params);
    ASSERT_NE(nullptr, api_.graph_kernel_node_set_params);

    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
      stream_ = nullptr;
    }
    // Keep the process-global runtime instance loaded across test cases. The
    // driver services it owns outlive an individual stream and are not
    // reinitializable after the final dlclose within the same process.
  }

  // Runtime entry points loaded once from the HIP shared object under test.
  static HipRuntimeApi api_;
  // Stream supplied to immediate launch entry points.
  hipStream_t stream_ = nullptr;
};

HipRuntimeApi HipLaunchValidationApiTest::api_;

TEST_F(HipLaunchValidationApiTest,
       FunctionDynamicSharedMemoryAttributeHonorsGenericCeiling) {
  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.function_attributes.provided_flags =
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY;
  symbol.function_attributes.maximum_configurable_dynamic_shared_memory_size =
      4096;
  iree_atomic_store(
      &symbol.function_attributes.configured_dynamic_shared_memory_size, 2048,
      iree_memory_order_relaxed);
  hipFunction_t function =
      reinterpret_cast<hipFunction_t>(iree_hal_streaming_symbol_tag(&symbol));

  int value = 0;
  EXPECT_EQ(hipSuccess,
            api_.function_get_attribute(
                &value, hipFuncAttributeMaxDynamicSharedSizeBytes, function));
  EXPECT_EQ(2048, value);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.function_set_attribute(
                function, hipFuncAttributeMaxDynamicSharedSizeBytes, -1));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.function_set_attribute(
                function, hipFuncAttributeMaxDynamicSharedSizeBytes, 4097));
  EXPECT_EQ(hipSuccess,
            api_.function_set_attribute(
                function, hipFuncAttributeMaxDynamicSharedSizeBytes, 4096));
  EXPECT_EQ(hipSuccess,
            api_.function_get_attribute(
                &value, hipFuncAttributeMaxDynamicSharedSizeBytes, function));
  EXPECT_EQ(4096, value);
}

TEST_F(HipLaunchValidationApiTest,
       LaunchEntryPointsRejectInvalidConfiguration) {
  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  const void* function = iree_hal_streaming_symbol_tag(&symbol);
  const dim3 invalid_grid = {0, 1, 1};
  const dim3 valid_dimension = {1, 1, 1};

  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.launch_kernel(function, invalid_grid, valid_dimension,
                               /*arguments=*/nullptr,
                               /*shared_memory_bytes=*/0, stream_));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.ext_launch_kernel(function, invalid_grid, valid_dimension,
                                   /*arguments=*/nullptr,
                                   /*shared_memory_bytes=*/0, stream_, nullptr,
                                   nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.module_launch_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/0, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0, stream_,
                /*arguments=*/nullptr, /*extra=*/nullptr));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipKernelNodeParams valid_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/nullptr,
      /*.func=*/const_cast<void*>(function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&node, graph, /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, &valid_params));

  hipKernelNodeParams invalid_params = valid_params;
  invalid_params.gridDim = invalid_grid;
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.graph_kernel_node_set_params(node, &invalid_params));

  hipKernelNodeParams retained_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_kernel_node_get_params(node, &retained_params));
  EXPECT_EQ(valid_params.func, retained_params.func);
  EXPECT_EQ(valid_params.gridDim.x, retained_params.gridDim.x);

  hipGraphNode_t rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(
      hipErrorInvalidConfiguration,
      api_.graph_add_kernel_node(&rejected_node, graph,
                                 /*dependencies=*/nullptr,
                                 /*dependency_count=*/0, &invalid_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipLaunchValidationApiTest,
       LaunchEntryPointsRejectOutOfRangeSharedMemory) {
  if (sizeof(size_t) <= sizeof(uint32_t)) {
    GTEST_SKIP() << "size_t cannot represent a value above uint32_t";
  }

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  const void* function = iree_hal_streaming_symbol_tag(&symbol);
  const dim3 valid_dimension = {1, 1, 1};
  const size_t largest_dispatch_shared_memory = UINT32_MAX;
  const size_t oversized_shared_memory = (size_t)UINT32_MAX + 1;

  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.launch_kernel(function, valid_dimension, valid_dimension,
                               /*arguments=*/nullptr,
                               largest_dispatch_shared_memory, stream_));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.ext_launch_kernel(function, valid_dimension, valid_dimension,
                                   /*arguments=*/nullptr,
                                   largest_dispatch_shared_memory, stream_,
                                   nullptr, nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.launch_kernel(function, valid_dimension, valid_dimension,
                               /*arguments=*/nullptr, oversized_shared_memory,
                               stream_));
  EXPECT_EQ(
      hipErrorInvalidConfiguration,
      api_.ext_launch_kernel(function, valid_dimension, valid_dimension,
                             /*arguments=*/nullptr, oversized_shared_memory,
                             stream_, nullptr, nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.module_launch_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, UINT32_MAX, stream_, /*arguments=*/nullptr,
                /*extra=*/nullptr));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipKernelNodeParams valid_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/nullptr,
      /*.func=*/const_cast<void*>(function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&node, graph, /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, &valid_params));

  hipKernelNodeParams rejected_params = valid_params;
  rejected_params.sharedMemBytes = largest_dispatch_shared_memory;
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.graph_kernel_node_set_params(node, &rejected_params));
  hipKernelNodeParams retained_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_kernel_node_get_params(node, &retained_params));
  EXPECT_EQ(0u, retained_params.sharedMemBytes);

  hipGraphNode_t rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(
      hipErrorInvalidConfiguration,
      api_.graph_add_kernel_node(&rejected_node, graph,
                                 /*dependencies=*/nullptr,
                                 /*dependency_count=*/0, &rejected_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);

  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipLaunchValidationApiTest,
       PrepackedGraphArgumentsRejectShortSpansWithoutMutatingTheNode) {
  iree_hal_streaming_symbol_t empty_symbol = {};
  empty_symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  iree_hal_streaming_symbol_t prepacked_symbol = {};
  prepacked_symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  prepacked_symbol.parameters.constant_bytes = 16;
  prepacked_symbol.parameters.direct_arg_bytes = 16;
  const void* empty_function = iree_hal_streaming_symbol_tag(&empty_symbol);
  const void* prepacked_function =
      iree_hal_streaming_symbol_tag(&prepacked_symbol);
  const dim3 valid_dimension = {1, 1, 1};

  uint8_t argument_storage[16] = {};
  size_t short_argument_size = 0;
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      argument_storage,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &short_argument_size,
      HIP_LAUNCH_PARAM_END,
  };
  hipKernelNodeParams empty_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/nullptr,
      /*.func=*/const_cast<void*>(empty_function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };
  hipKernelNodeParams short_prepacked_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/extra,
      /*.func=*/const_cast<void*>(prepacked_function),
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };

  EXPECT_EQ(
      hipErrorInvalidValue,
      api_.module_launch_kernel(
          (hipFunction_t)prepacked_function, /*grid_dim_x=*/1,
          /*grid_dim_y=*/1, /*grid_dim_z=*/1, /*block_dim_x=*/1,
          /*block_dim_y=*/1, /*block_dim_z=*/1,
          /*shared_memory_bytes=*/0, stream_, /*arguments=*/nullptr, extra));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&node, graph, /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, &empty_params));

  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_kernel_node_set_params(node, &short_prepacked_params));
  hipKernelNodeParams retained_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_kernel_node_get_params(node, &retained_params));
  EXPECT_EQ(empty_params.func, retained_params.func);
  EXPECT_EQ(empty_params.extra, retained_params.extra);

  hipGraphNode_t rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_add_kernel_node(&rejected_node, graph,
                                       /*dependencies=*/nullptr,
                                       /*dependency_count=*/0,
                                       &short_prepacked_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

}  // namespace
