// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipGetDevicePropertiesR0600Fn =
    hipError_t (*)(hipDeviceProp_t* properties, int device);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t* module,
                                           const void* image);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t module);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipModule_t module,
                                              const char* name);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipStreamGetIdFn = hipError_t (*)(hipStream_t stream,
                                        unsigned long long* stream_id);
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
using HipModuleLaunchCooperativeKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments);
using HipExtLaunchMultiKernelMultiDeviceFn = hipError_t (*)(
    hipLaunchParams* launch_parameters, int device_count, unsigned int flags);
using HipLaunchKernelExCFn = hipError_t (*)(const hipLaunchConfig_t* config,
                                            const void* function,
                                            void** arguments);
using HipDrvLaunchKernelExFn = hipError_t (*)(const HIP_LAUNCH_CONFIG* config,
                                              hipFunction_t function,
                                              void** arguments, void** extra);
using HipFuncGetAttributeFn = hipError_t (*)(int* value,
                                             hipFuncAttribute_t attribute,
                                             hipFunction_t function);
using HipFuncSetAttributeFn = hipError_t (*)(hipFunction_t function,
                                             hipFuncAttribute_t attribute,
                                             int value);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddEmptyNodeFn =
    hipError_t (*)(hipGraphNode_t* node, hipGraph_t graph,
                   const hipGraphNode_t* dependencies, size_t dependency_count);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* executable,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t executable,
                                        hipStream_t stream);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t executable);
using HipGraphAddKernelNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, const void* params);
using HipGraphKernelNodeGetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                     void* params);
using HipGraphKernelNodeSetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                     const void* params);
using HipGraphKernelNodeSetAttributeFn =
    hipError_t (*)(hipGraphNode_t node, hipKernelNodeAttrID attribute,
                   const hipKernelNodeAttrValue* value);
using HipGraphKernelNodeGetAttributeFn =
    hipError_t (*)(hipGraphNode_t node, hipKernelNodeAttrID attribute,
                   hipKernelNodeAttrValue* value);

// Owns an RTLD_LOCAL HIP runtime instance and the entry points exercised by
// this test. All calls use the loaded library instead of a link-time runtime.
struct HipRuntimeApi {
  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;
  // Deinitializes the HIP runtime instance before unloading its DSO.
  HipHalDeinitFn hal_deinit = nullptr;
  // Queries the device selected by the current thread.
  HipGetDeviceFn get_device = nullptr;
  // Queries the selected device architecture used to choose an HSACO image.
  HipGetDevicePropertiesR0600Fn get_device_properties = nullptr;
  // Loads an executable image into the current context.
  HipModuleLoadDataFn module_load_data = nullptr;
  // Unloads an executable image from the current context.
  HipModuleUnloadFn module_unload = nullptr;
  // Resolves a production function handle from a loaded module.
  HipModuleGetFunctionFn module_get_function = nullptr;
  // Creates the stream used by immediate launch entry points.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys the stream used by immediate launch entry points.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Waits for all work previously enqueued on a stream.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Queries a stream without dereferencing a stale public handle.
  HipStreamGetIdFn stream_get_id = nullptr;
  // Launches a registered runtime kernel.
  HipLaunchKernelFn launch_kernel = nullptr;
  // Launches a registered runtime kernel with extended launch arguments.
  HipExtLaunchKernelFn ext_launch_kernel = nullptr;
  // Launches a module kernel with prepacked or pointer-array arguments.
  HipModuleLaunchKernelFn module_launch_kernel = nullptr;
  // Launches a module kernel cooperatively.
  HipModuleLaunchCooperativeKernelFn module_launch_cooperative_kernel = nullptr;
  // Launches one matching kernel on each explicit device stream.
  HipExtLaunchMultiKernelMultiDeviceFn launch_multi_device = nullptr;
  // Launches a runtime kernel from an extensible configuration.
  HipLaunchKernelExCFn launch_kernel_ex = nullptr;
  // Launches a module kernel from an extensible configuration.
  HipDrvLaunchKernelExFn driver_launch_kernel_ex = nullptr;
  // Queries a cached function compatibility attribute.
  HipFuncGetAttributeFn function_get_attribute = nullptr;
  // Updates a mutable function compatibility attribute.
  HipFuncSetAttributeFn function_set_attribute = nullptr;
  // Creates a graph template.
  HipGraphCreateFn graph_create = nullptr;
  // Destroys a graph template.
  HipGraphDestroyFn graph_destroy = nullptr;
  // Adds a dependency-only node to a graph template.
  HipGraphAddEmptyNodeFn graph_add_empty_node = nullptr;
  // Instantiates a graph template.
  HipGraphInstantiateFn graph_instantiate = nullptr;
  // Launches a graph executable.
  HipGraphLaunchFn graph_launch = nullptr;
  // Destroys a graph executable.
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;
  // Adds a kernel node to a graph template.
  HipGraphAddKernelNodeFn graph_add_kernel_node = nullptr;
  // Reads the parameters retained by a graph kernel node.
  HipGraphKernelNodeGetParamsFn graph_kernel_node_get_params = nullptr;
  // Replaces the parameters retained by a graph kernel node.
  HipGraphKernelNodeSetParamsFn graph_kernel_node_set_params = nullptr;
  // Updates one attribute retained by a graph kernel node.
  HipGraphKernelNodeSetAttributeFn graph_kernel_node_set_attribute = nullptr;
  // Reads one attribute retained by a graph kernel node.
  HipGraphKernelNodeGetAttributeFn graph_kernel_node_get_attribute = nullptr;
};

class HipLaunchValidationApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!dso_.is_open()) {
      ASSERT_TRUE(dso_.Open()) << dso_.error();
      api_.init = dso_.Resolve<HipInitFn>("hipInit");
      api_.hal_deinit = dso_.Resolve<HipHalDeinitFn>("hipHALDeinit");
      api_.get_device = dso_.Resolve<HipGetDeviceFn>("hipGetDevice");
      api_.get_device_properties = dso_.Resolve<HipGetDevicePropertiesR0600Fn>(
          "hipGetDevicePropertiesR0600");
      api_.module_load_data =
          dso_.Resolve<HipModuleLoadDataFn>("hipModuleLoadData");
      api_.module_unload = dso_.Resolve<HipModuleUnloadFn>("hipModuleUnload");
      api_.module_get_function =
          dso_.Resolve<HipModuleGetFunctionFn>("hipModuleGetFunction");
      api_.stream_create = dso_.Resolve<HipStreamCreateFn>("hipStreamCreate");
      api_.stream_destroy =
          dso_.Resolve<HipStreamDestroyFn>("hipStreamDestroy");
      api_.stream_synchronize =
          dso_.Resolve<HipStreamSynchronizeFn>("hipStreamSynchronize");
      api_.stream_get_id = dso_.Resolve<HipStreamGetIdFn>("hipStreamGetId");
      api_.launch_kernel = dso_.Resolve<HipLaunchKernelFn>("hipLaunchKernel");
      api_.ext_launch_kernel =
          dso_.Resolve<HipExtLaunchKernelFn>("hipExtLaunchKernel");
      api_.module_launch_kernel =
          dso_.Resolve<HipModuleLaunchKernelFn>("hipModuleLaunchKernel");
      api_.module_launch_cooperative_kernel =
          dso_.Resolve<HipModuleLaunchCooperativeKernelFn>(
              "hipModuleLaunchCooperativeKernel");
      api_.launch_multi_device =
          dso_.Resolve<HipExtLaunchMultiKernelMultiDeviceFn>(
              "hipExtLaunchMultiKernelMultiDevice");
      api_.launch_kernel_ex =
          dso_.Resolve<HipLaunchKernelExCFn>("hipLaunchKernelExC");
      api_.driver_launch_kernel_ex =
          dso_.Resolve<HipDrvLaunchKernelExFn>("hipDrvLaunchKernelEx");
      api_.function_get_attribute =
          dso_.Resolve<HipFuncGetAttributeFn>("hipFuncGetAttribute");
      api_.function_set_attribute =
          dso_.Resolve<HipFuncSetAttributeFn>("hipFuncSetAttribute");
      api_.graph_create = dso_.Resolve<HipGraphCreateFn>("hipGraphCreate");
      api_.graph_destroy = dso_.Resolve<HipGraphDestroyFn>("hipGraphDestroy");
      api_.graph_add_empty_node =
          dso_.Resolve<HipGraphAddEmptyNodeFn>("hipGraphAddEmptyNode");
      api_.graph_instantiate =
          dso_.Resolve<HipGraphInstantiateFn>("hipGraphInstantiate");
      api_.graph_launch = dso_.Resolve<HipGraphLaunchFn>("hipGraphLaunch");
      api_.graph_exec_destroy =
          dso_.Resolve<HipGraphExecDestroyFn>("hipGraphExecDestroy");
      api_.graph_add_kernel_node =
          dso_.Resolve<HipGraphAddKernelNodeFn>("hipGraphAddKernelNode");
      api_.graph_kernel_node_get_params =
          dso_.Resolve<HipGraphKernelNodeGetParamsFn>(
              "hipGraphKernelNodeGetParams");
      api_.graph_kernel_node_set_params =
          dso_.Resolve<HipGraphKernelNodeSetParamsFn>(
              "hipGraphKernelNodeSetParams");
      api_.graph_kernel_node_set_attribute =
          dso_.Resolve<HipGraphKernelNodeSetAttributeFn>(
              "hipGraphKernelNodeSetAttribute");
      api_.graph_kernel_node_get_attribute =
          dso_.Resolve<HipGraphKernelNodeGetAttributeFn>(
              "hipGraphKernelNodeGetAttribute");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.hal_deinit);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.get_device_properties);
    ASSERT_NE(nullptr, api_.module_load_data);
    ASSERT_NE(nullptr, api_.module_unload);
    ASSERT_NE(nullptr, api_.module_get_function);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.stream_synchronize);
    ASSERT_NE(nullptr, api_.stream_get_id);
    ASSERT_NE(nullptr, api_.launch_kernel);
    ASSERT_NE(nullptr, api_.ext_launch_kernel);
    ASSERT_NE(nullptr, api_.module_launch_kernel);
    ASSERT_NE(nullptr, api_.module_launch_cooperative_kernel);
    ASSERT_NE(nullptr, api_.launch_multi_device);
    ASSERT_NE(nullptr, api_.launch_kernel_ex);
    ASSERT_NE(nullptr, api_.driver_launch_kernel_ex);
    ASSERT_NE(nullptr, api_.function_get_attribute);
    ASSERT_NE(nullptr, api_.function_set_attribute);
    ASSERT_NE(nullptr, api_.graph_create);
    ASSERT_NE(nullptr, api_.graph_destroy);
    ASSERT_NE(nullptr, api_.graph_add_empty_node);
    ASSERT_NE(nullptr, api_.graph_instantiate);
    ASSERT_NE(nullptr, api_.graph_launch);
    ASSERT_NE(nullptr, api_.graph_exec_destroy);
    ASSERT_NE(nullptr, api_.graph_add_kernel_node);
    ASSERT_NE(nullptr, api_.graph_kernel_node_get_params);
    ASSERT_NE(nullptr, api_.graph_kernel_node_set_params);
    ASSERT_NE(nullptr, api_.graph_kernel_node_set_attribute);
    ASSERT_NE(nullptr, api_.graph_kernel_node_get_attribute);

    const hipError_t init_result = api_.init(/*flags=*/0);
    ASSERT_EQ(hipSuccess, init_result);
    if (!module_) {
      int device = 0;
      ASSERT_EQ(hipSuccess, api_.get_device(&device));
      hipDeviceProp_t properties = {};
      ASSERT_EQ(hipSuccess, api_.get_device_properties(&properties, device));
      supports_cooperative_launch_ = properties.cooperativeLaunch != 0;
      const hrx_cts::AmdgpuExecutableTestImage test_image =
          hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
      ASSERT_NE(nullptr, test_image.file)
          << "no embedded executable HSACO for " << properties.gcnArchName;
      ASSERT_EQ(hipSuccess,
                api_.module_load_data(&module_, test_image.file->data));
      ASSERT_EQ(hipSuccess, api_.module_get_function(&empty_function_, module_,
                                                     "hrx_noop"));
      ASSERT_EQ(hipSuccess,
                api_.module_get_function(&prepacked_function_, module_,
                                         "hrx_transform_nested_pointers"));
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

  static void TearDownTestSuite() {
    if (!dso_.is_open()) {
      return;
    }
    ASSERT_NE(nullptr, api_.module_unload);
    if (module_) {
      EXPECT_EQ(hipSuccess, api_.module_unload(module_));
      module_ = nullptr;
      empty_function_ = nullptr;
      prepacked_function_ = nullptr;
    }
    ASSERT_NE(nullptr, api_.hal_deinit);
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    api_ = {};
    EXPECT_TRUE(dso_.Close()) << dso_.error();
  }

  // Runtime entry points loaded once from the HIP shared object under test.
  static HipRuntimeApi api_;
  // Exact DSO owner shared by every test in this fixture.
  static hrx::hip::testing::HipDso dso_;
  // Module loaded through the public API and retained across test cases.
  static hipModule_t module_;
  // No-argument kernel used when argument packing is not under test.
  static hipFunction_t empty_function_;
  // Kernel with a nonempty native argument image used for span validation.
  static hipFunction_t prepacked_function_;
  // Whether the selected device supports cooperative dispatch.
  static bool supports_cooperative_launch_;
  // Stream supplied to immediate launch entry points.
  hipStream_t stream_ = nullptr;
};

HipRuntimeApi HipLaunchValidationApiTest::api_;
hrx::hip::testing::HipDso HipLaunchValidationApiTest::dso_;
hipModule_t HipLaunchValidationApiTest::module_;
hipFunction_t HipLaunchValidationApiTest::empty_function_;
hipFunction_t HipLaunchValidationApiTest::prepacked_function_;
bool HipLaunchValidationApiTest::supports_cooperative_launch_ = false;

TEST_F(HipLaunchValidationApiTest,
       FunctionDynamicSharedMemoryAttributeRejectsInvalidValues) {
  int original_value = 0;
  ASSERT_EQ(hipSuccess,
            api_.function_get_attribute(
                &original_value, hipFuncAttributeMaxDynamicSharedSizeBytes,
                empty_function_));
  EXPECT_GE(original_value, 0);
  EXPECT_EQ(
      hipErrorInvalidValue,
      api_.function_set_attribute(
          empty_function_, hipFuncAttributeMaxDynamicSharedSizeBytes, -1));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.function_set_attribute(
                empty_function_, hipFuncAttributeMaxDynamicSharedSizeBytes,
                std::numeric_limits<int>::max()));
  EXPECT_EQ(hipSuccess,
            api_.function_set_attribute(
                empty_function_, hipFuncAttributeMaxDynamicSharedSizeBytes, 0));
  int value = -1;
  EXPECT_EQ(hipSuccess, api_.function_get_attribute(
                            &value, hipFuncAttributeMaxDynamicSharedSizeBytes,
                            empty_function_));
  EXPECT_EQ(0, value);
  EXPECT_EQ(hipSuccess,
            api_.function_set_attribute(
                empty_function_, hipFuncAttributeMaxDynamicSharedSizeBytes,
                original_value));
}

TEST_F(HipLaunchValidationApiTest,
       LaunchEntryPointsRejectInvalidConfiguration) {
  const void* function = empty_function_;
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

  hipLaunchConfig_t runtime_config = {};
  runtime_config.gridDim = invalid_grid;
  runtime_config.blockDim = valid_dimension;
  runtime_config.stream = stream_;
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api_.launch_kernel_ex(&runtime_config, function,
                                  /*arguments=*/nullptr));

  HIP_LAUNCH_CONFIG driver_config = {};
  driver_config.gridDimY = 1;
  driver_config.gridDimZ = 1;
  driver_config.blockDimX = 1;
  driver_config.blockDimY = 1;
  driver_config.blockDimZ = 1;
  driver_config.hStream = stream_;
  EXPECT_EQ(
      hipErrorInvalidConfiguration,
      api_.driver_launch_kernel_ex(&driver_config, (hipFunction_t)function,
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
       ExtendedLaunchEntryPointsRejectMalformedAttributes) {
  const dim3 one = {1, 1, 1};
  hipLaunchAttribute unsupported_attribute = {};
  unsupported_attribute.id = hipLaunchAttributeClusterDimension;

  hipLaunchConfig_t runtime_config = {};
  runtime_config.gridDim = one;
  runtime_config.blockDim = one;
  runtime_config.stream = stream_;
  runtime_config.attrs = &unsupported_attribute;
  runtime_config.numAttrs = 1;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.launch_kernel_ex(&runtime_config, empty_function_,
                                  /*arguments=*/nullptr));

  HIP_LAUNCH_CONFIG driver_config = {};
  driver_config.gridDimX = 1;
  driver_config.gridDimY = 1;
  driver_config.gridDimZ = 1;
  driver_config.blockDimX = 1;
  driver_config.blockDimY = 1;
  driver_config.blockDimZ = 1;
  driver_config.hStream = stream_;
  driver_config.attrs = &unsupported_attribute;
  driver_config.numAttrs = 1;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.driver_launch_kernel_ex(&driver_config, empty_function_,
                                         /*arguments=*/nullptr,
                                         /*extra=*/nullptr));

  runtime_config.attrs = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.launch_kernel_ex(&runtime_config, empty_function_,
                                  /*arguments=*/nullptr));
  driver_config.attrs = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.driver_launch_kernel_ex(&driver_config, empty_function_,
                                         /*arguments=*/nullptr,
                                         /*extra=*/nullptr));

  hipLaunchAttribute prefetch_attribute = {};
  prefetch_attribute.id = hipLaunchAttributeExtDynDataPrefetch;
  runtime_config.attrs = &prefetch_attribute;
  driver_config.attrs = &prefetch_attribute;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.launch_kernel_ex(&runtime_config, empty_function_,
                                  /*arguments=*/nullptr));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.driver_launch_kernel_ex(&driver_config, empty_function_,
                                         /*arguments=*/nullptr,
                                         /*extra=*/nullptr));

  prefetch_attribute.val.dynDataPrefetch =
      reinterpret_cast<const hipExtDynDataPrefetchConfig*>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            api_.launch_kernel_ex(&runtime_config, empty_function_,
                                  /*arguments=*/nullptr));
  EXPECT_EQ(hipErrorNotSupported,
            api_.driver_launch_kernel_ex(&driver_config, empty_function_,
                                         /*arguments=*/nullptr,
                                         /*extra=*/nullptr));
}

TEST_F(HipLaunchValidationApiTest,
       ExtendedLaunchEntryPointsDispatchNoArgumentKernel) {
  const dim3 one = {1, 1, 1};
  hipLaunchConfig_t runtime_config = {};
  runtime_config.gridDim = one;
  runtime_config.blockDim = one;
  runtime_config.stream = stream_;

  HIP_LAUNCH_CONFIG driver_config = {};
  driver_config.gridDimX = 1;
  driver_config.gridDimY = 1;
  driver_config.gridDimZ = 1;
  driver_config.blockDimX = 1;
  driver_config.blockDimY = 1;
  driver_config.blockDimZ = 1;
  driver_config.hStream = stream_;

  ASSERT_EQ(hipSuccess, api_.launch_kernel_ex(&runtime_config, empty_function_,
                                              /*arguments=*/nullptr));
  ASSERT_EQ(hipSuccess,
            api_.driver_launch_kernel_ex(&driver_config, empty_function_,
                                         /*arguments=*/nullptr,
                                         /*extra=*/nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  if (supports_cooperative_launch_) {
    hipLaunchAttribute cooperative_attribute = {};
    cooperative_attribute.id = hipLaunchAttributeCooperative;
    cooperative_attribute.val.cooperative = 1;
    runtime_config.attrs = &cooperative_attribute;
    runtime_config.numAttrs = 1;
    driver_config.attrs = &cooperative_attribute;
    driver_config.numAttrs = 1;

    ASSERT_EQ(hipSuccess,
              api_.launch_kernel_ex(&runtime_config, empty_function_,
                                    /*arguments=*/nullptr));
    ASSERT_EQ(hipSuccess,
              api_.driver_launch_kernel_ex(&driver_config, empty_function_,
                                           /*arguments=*/nullptr,
                                           /*extra=*/nullptr));

    uint8_t empty_argument_buffer = 0;
    size_t empty_argument_buffer_size = 0;
    void* extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        &empty_argument_buffer,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &empty_argument_buffer_size,
        HIP_LAUNCH_PARAM_END,
    };
    ASSERT_EQ(hipSuccess,
              api_.driver_launch_kernel_ex(&driver_config, empty_function_,
                                           /*arguments=*/nullptr, extra));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  }
}

TEST_F(HipLaunchValidationApiTest, ZeroAccessPolicyWindowRemainsSupported) {
  const dim3 one = {1, 1, 1};
  hipKernelNodeParams params = {
      /*.blockDim=*/one,
      /*.extra=*/nullptr,
      /*.func=*/empty_function_,
      /*.gridDim=*/one,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_kernel_node(&node, graph, /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, &params));

  hipKernelNodeAttrValue access_policy = {};
  access_policy.accessPolicyWindow.hitProp = hipAccessPropertyNormal;
  access_policy.accessPolicyWindow.missProp = hipAccessPropertyNormal;
  EXPECT_EQ(hipSuccess, api_.graph_kernel_node_set_attribute(
                            node, hipKernelNodeAttributeAccessPolicyWindow,
                            &access_policy));

  access_policy.accessPolicyWindow.num_bytes = 1;
  EXPECT_EQ(
      hipErrorInvalidValue,
      api_.graph_kernel_node_set_attribute(
          node, hipKernelNodeAttributeAccessPolicyWindow, &access_policy));

  hipKernelNodeAttrValue retained_access_policy = {};
  ASSERT_EQ(hipSuccess, api_.graph_kernel_node_get_attribute(
                            node, hipKernelNodeAttributeAccessPolicyWindow,
                            &retained_access_policy));
  EXPECT_EQ(0u, retained_access_policy.accessPolicyWindow.num_bytes);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipLaunchValidationApiTest, LaunchEntryPointsRejectDestroyedStreams) {
  const void* function = empty_function_;
  const dim3 valid_dimension = {1, 1, 1};

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t empty_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(&empty_node, graph,
                                                  /*dependencies=*/nullptr,
                                                  /*dependency_count=*/0));
  hipGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&graph_exec, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*buffer_size=*/0));

  hipStream_t stale_stream = stream_;
  ASSERT_EQ(hipSuccess, api_.stream_destroy(stream_));
  stream_ = nullptr;

  EXPECT_EQ(hipErrorInvalidValue,
            api_.launch_kernel(function, valid_dimension, valid_dimension,
                               /*arguments=*/nullptr,
                               /*shared_memory_bytes=*/0, stale_stream));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.ext_launch_kernel(function, valid_dimension, valid_dimension,
                                   /*arguments=*/nullptr,
                                   /*shared_memory_bytes=*/0, stale_stream,
                                   nullptr, nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorContextIsDestroyed,
            api_.module_launch_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0, stale_stream,
                /*arguments=*/nullptr, /*extra=*/nullptr));
  EXPECT_EQ(hipErrorContextIsDestroyed,
            api_.module_launch_cooperative_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0, stale_stream,
                /*arguments=*/nullptr));

  hipLaunchParams multi_device_launch = {
      /*.func=*/reinterpret_cast<void*>(uintptr_t{1}),
      /*.gridDim=*/valid_dimension,
      /*.blockDim=*/valid_dimension,
      /*.args=*/nullptr,
      /*.sharedMem=*/0,
      /*.stream=*/stale_stream,
  };
  EXPECT_EQ(hipErrorInvalidValue,
            api_.launch_multi_device(&multi_device_launch,
                                     /*device_count=*/1, /*flags=*/0));

  hipLaunchConfig_t runtime_config = {};
  runtime_config.gridDim = valid_dimension;
  runtime_config.blockDim = valid_dimension;
  runtime_config.stream = stale_stream;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.launch_kernel_ex(&runtime_config, function,
                                  /*arguments=*/nullptr));

  HIP_LAUNCH_CONFIG driver_config = {};
  driver_config.gridDimX = 1;
  driver_config.gridDimY = 1;
  driver_config.gridDimZ = 1;
  driver_config.blockDimX = 1;
  driver_config.blockDimY = 1;
  driver_config.blockDimZ = 1;
  driver_config.hStream = stale_stream;
  EXPECT_EQ(
      hipErrorContextIsDestroyed,
      api_.driver_launch_kernel_ex(&driver_config, (hipFunction_t)function,
                                   /*arguments=*/nullptr,
                                   /*extra=*/nullptr));

  hipLaunchAttribute cooperative_attribute = {};
  cooperative_attribute.id = hipLaunchAttributeCooperative;
  cooperative_attribute.val.cooperative = 1;
  runtime_config.attrs = &cooperative_attribute;
  runtime_config.numAttrs = 1;
  driver_config.attrs = &cooperative_attribute;
  driver_config.numAttrs = 1;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.launch_kernel_ex(&runtime_config, function,
                                  /*arguments=*/nullptr));
  EXPECT_EQ(
      hipErrorContextIsDestroyed,
      api_.driver_launch_kernel_ex(&driver_config, (hipFunction_t)function,
                                   /*arguments=*/nullptr,
                                   /*extra=*/nullptr));
  EXPECT_EQ(hipErrorContextIsDestroyed,
            api_.graph_launch(graph_exec, stale_stream));

  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(graph_exec));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipLaunchValidationApiTest,
       ConcurrentQueriesObserveStreamDestructionWithoutDereferencingIt) {
  hipStream_t stream = stream_;
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> observed_zero_id{false};
  std::atomic<hipError_t> final_result{hipSuccess};
  std::thread query_thread([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!stop.load(std::memory_order_acquire)) {
      unsigned long long stream_id = 0;
      const hipError_t result = api_.stream_get_id(stream, &stream_id);
      if (result != hipSuccess) {
        final_result.store(result, std::memory_order_release);
        return;
      }
      if (stream_id == 0) {
        observed_zero_id.store(true, std::memory_order_release);
      }
    }
  });

  start.store(true, std::memory_order_release);
  const hipError_t destroy_result = api_.stream_destroy(stream_);
  if (destroy_result == hipSuccess) {
    stream_ = nullptr;
  } else {
    stop.store(true, std::memory_order_release);
  }
  query_thread.join();
  EXPECT_EQ(hipSuccess, destroy_result);
  EXPECT_FALSE(observed_zero_id.load(std::memory_order_acquire));
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            final_result.load(std::memory_order_acquire));
}

TEST_F(HipLaunchValidationApiTest,
       LaunchEntryPointsRejectOutOfRangeSharedMemory) {
  if (sizeof(size_t) <= sizeof(uint32_t)) {
    GTEST_SKIP() << "size_t cannot represent a value above uint32_t";
  }

  const void* function = empty_function_;
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
      /*.func=*/empty_function_,
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };
  hipKernelNodeParams short_prepacked_params = {
      /*.blockDim=*/valid_dimension,
      /*.extra=*/extra,
      /*.func=*/prepacked_function_,
      /*.gridDim=*/valid_dimension,
      /*.kernelParams=*/nullptr,
      /*.sharedMemBytes=*/0,
  };

  EXPECT_EQ(hipErrorInvalidValue,
            api_.module_launch_kernel(prepacked_function_, /*grid_dim_x=*/1,
                                      /*grid_dim_y=*/1,
                                      /*grid_dim_z=*/1, /*block_dim_x=*/1,
                                      /*block_dim_y=*/1, /*block_dim_z=*/1,
                                      /*shared_memory_bytes=*/0, stream_,
                                      /*arguments=*/nullptr, extra));

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
