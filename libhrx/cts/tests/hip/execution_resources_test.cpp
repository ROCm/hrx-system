// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

class HipExecutionResourcesTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(library_.Open()) << library_.error();
    const auto initialize = library_.Resolve<decltype(&hipInit)>("hipInit");
    ASSERT_NE(initialize, nullptr);
    ASSERT_EQ(initialize(0), hipSuccess);
  }

  static void TearDownTestSuite() {
    if (!library_.is_open()) {
      return;
    }
    const auto deinitialize =
        library_.Resolve<decltype(&hipHALDeinit)>("hipHALDeinit");
    ASSERT_NE(deinitialize, nullptr);
    ASSERT_EQ(deinitialize(), hipSuccess);
    EXPECT_TRUE(library_.Close()) << library_.error();
  }

  void TearDown() override {
    if (stream_) {
      const auto synchronize =
          library_.Resolve<decltype(&hipStreamSynchronize)>(
              "hipStreamSynchronize");
      ASSERT_NE(synchronize, nullptr);
      ASSERT_EQ(synchronize(stream_), hipSuccess);
    }
    if (graph_executable_) {
      const auto destroy = library_.Resolve<decltype(&hipGraphExecDestroy)>(
          "hipGraphExecDestroy");
      ASSERT_NE(destroy, nullptr);
      EXPECT_EQ(destroy(graph_executable_), hipSuccess);
    }
    if (graph_) {
      const auto destroy =
          library_.Resolve<decltype(&hipGraphDestroy)>("hipGraphDestroy");
      ASSERT_NE(destroy, nullptr);
      EXPECT_EQ(destroy(graph_), hipSuccess);
    }
    if (output_) {
      const auto release = library_.Resolve<decltype(&hipFree)>("hipFree");
      ASSERT_NE(release, nullptr);
      EXPECT_EQ(release(output_), hipSuccess);
    }
    if (module_) {
      const auto unload =
          library_.Resolve<decltype(&hipModuleUnload)>("hipModuleUnload");
      ASSERT_NE(unload, nullptr);
      EXPECT_EQ(unload(module_), hipSuccess);
    }
    if (stream_) {
      const auto destroy =
          library_.Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy");
      ASSERT_NE(destroy, nullptr);
      EXPECT_EQ(destroy(stream_), hipSuccess);
    }
    if (context_) {
      const auto destroy = library_.Resolve<decltype(&hipExecutionCtxDestroy)>(
          "hipExecutionCtxDestroy");
      ASSERT_NE(destroy, nullptr);
      EXPECT_EQ(destroy(context_), hipSuccess);
    }
  }

  // Exact frontend DSO retained through joined runtime shutdown.
  static inline hrx::hip::testing::HipDso library_;
  // Selected-resource owner, outliving its stream.
  hipExecutionCtx_t context_ = nullptr;
  // Sole restricted stream used for direct and captured execution.
  hipStream_t stream_ = nullptr;
  // Compiler-produced module retained through all dispatches.
  hipModule_t module_ = nullptr;
  // Bounded output storage shared by successive executions.
  void* output_ = nullptr;
  // Captured stream program borrowing the loaded function.
  hipGraph_t graph_ = nullptr;
  // Executable graph released after its final stream completion.
  hipGraphExec_t graph_executable_ = nullptr;
};

TEST_F(HipExecutionResourcesTest,
       PartitionedStreamRunsDirectAndCapturedKernels) {
  const auto get_device =
      library_.Resolve<decltype(&hipGetDevice)>("hipGetDevice");
  const auto get_properties =
      library_.Resolve<decltype(&hipGetDeviceProperties)>(
          "hipGetDevicePropertiesR0600");
  const auto get_resource =
      library_.Resolve<decltype(&hipDeviceGetDevResource)>(
          "hipDeviceGetDevResource");
  const auto split = library_.Resolve<decltype(&hipDevSmResourceSplitByCount)>(
      "hipDevSmResourceSplitByCount");
  const auto describe = library_.Resolve<decltype(&hipDevResourceGenerateDesc)>(
      "hipDevResourceGenerateDesc");
  const auto create =
      library_.Resolve<decltype(&hipGreenCtxCreate)>("hipGreenCtxCreate");
  const auto create_stream =
      library_.Resolve<decltype(&hipExecutionCtxStreamCreate)>(
          "hipExecutionCtxStreamCreate");
  const auto stream_resource =
      library_.Resolve<decltype(&hipStreamGetDevResource)>(
          "hipStreamGetDevResource");
  const auto load =
      library_.Resolve<decltype(&hipModuleLoadData)>("hipModuleLoadData");
  const auto lookup =
      library_.Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction");
  const auto allocate = library_.Resolve<decltype(&hipMalloc)>("hipMalloc");
  const auto fill =
      library_.Resolve<decltype(&hipMemsetAsync)>("hipMemsetAsync");
  const auto copy =
      library_.Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
  const auto launch = library_.Resolve<decltype(&hipModuleLaunchKernel)>(
      "hipModuleLaunchKernel");
  const auto synchronize =
      library_.Resolve<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
  const auto begin_capture = library_.Resolve<decltype(&hipStreamBeginCapture)>(
      "hipStreamBeginCapture");
  const auto end_capture =
      library_.Resolve<decltype(&hipStreamEndCapture)>("hipStreamEndCapture");
  const auto instantiate =
      library_.Resolve<decltype(&hipGraphInstantiate)>("hipGraphInstantiate");
  const auto launch_graph =
      library_.Resolve<decltype(&hipGraphLaunch)>("hipGraphLaunch");
  ASSERT_NE(get_device, nullptr);
  ASSERT_NE(get_properties, nullptr);
  ASSERT_NE(get_resource, nullptr);
  ASSERT_NE(split, nullptr);
  ASSERT_NE(describe, nullptr);
  ASSERT_NE(create, nullptr);
  ASSERT_NE(create_stream, nullptr);
  ASSERT_NE(stream_resource, nullptr);
  ASSERT_NE(load, nullptr);
  ASSERT_NE(lookup, nullptr);
  ASSERT_NE(allocate, nullptr);
  ASSERT_NE(fill, nullptr);
  ASSERT_NE(copy, nullptr);
  ASSERT_NE(launch, nullptr);
  ASSERT_NE(synchronize, nullptr);
  ASSERT_NE(begin_capture, nullptr);
  ASSERT_NE(end_capture, nullptr);
  ASSERT_NE(instantiate, nullptr);
  ASSERT_NE(launch_graph, nullptr);

  int device = 0;
  ASSERT_EQ(get_device(&device), hipSuccess);
  hipDevResource full = {};
  ASSERT_EQ(get_resource(device, &full, hipDevResourceTypeSm), hipSuccess);
  ASSERT_GT(full.sm.minSmPartitionSize, 0u);
  if (full.sm.minSmPartitionSize == full.sm.smCount) {
    GTEST_SKIP() << "device exposes an indivisible compute resource";
  }
  hipDevResource partition = {}, remainder = {};
  unsigned int count = 1;
  ASSERT_EQ(split(&partition, &count, &full, &remainder, 0,
                  full.sm.minSmPartitionSize),
            hipSuccess);
  ASSERT_EQ(count, 1u);
  ASSERT_LT(partition.sm.smCount, full.sm.smCount);
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(describe(&descriptor, &partition, 1), hipSuccess);
  ASSERT_EQ(create(&context_, descriptor, device, 0), hipSuccess);
  ASSERT_EQ(create_stream(&stream_, context_, hipStreamNonBlocking, 0),
            hipSuccess);
  hipDevResource achieved = {};
  ASSERT_EQ(stream_resource(stream_, &achieved, hipDevResourceTypeSm),
            hipSuccess);
  EXPECT_EQ(achieved.sm.smCount, partition.sm.smCount);
  RecordProperty("selected_compute_units", achieved.sm.smCount);
  RecordProperty("device_compute_units", full.sm.smCount);

  hipDeviceProp_t properties = {};
  ASSERT_EQ(get_properties(&properties, device), hipSuccess);
  const auto image =
      hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
  ASSERT_NE(image.file, nullptr) << properties.gcnArchName;
  ASSERT_EQ(load(&module_, image.file->data), hipSuccess);
  hipFunction_t function = nullptr;
  ASSERT_EQ(lookup(&function, module_, "hrx_store_exact_workitem_indices"),
            hipSuccess);
  constexpr uint32_t kWorkitems = 128;
  constexpr uint32_t kDispatches = 8;
  std::array<uint32_t, kWorkitems * kDispatches> actual = {};
  ASSERT_EQ(allocate(&output_, sizeof(actual)), hipSuccess);
  ASSERT_EQ(fill(output_, 0, sizeof(actual), stream_), hipSuccess);
  uint32_t local_size = 64;
  void* arguments[] = {&local_size, &output_};
  ASSERT_EQ(launch(function, 2, 1, 1, local_size, 1, 1, 0, stream_, arguments,
                   nullptr),
            hipSuccess);
  ASSERT_EQ(copy(actual.data(), output_, sizeof(actual), hipMemcpyDeviceToHost,
                 stream_),
            hipSuccess);
  ASSERT_EQ(synchronize(stream_), hipSuccess);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], i < kWorkitems ? i + 1 : 0);
  }

  ASSERT_EQ(fill(output_, 0, sizeof(actual), stream_), hipSuccess);
  ASSERT_EQ(begin_capture(stream_, hipStreamCaptureModeGlobal), hipSuccess);
  for (uint32_t i = 0; i < kDispatches; ++i) {
    void* slice = static_cast<uint32_t*>(output_) + i * kWorkitems;
    void* captured_arguments[] = {&local_size, &slice};
    ASSERT_EQ(launch(function, 2, 1, 1, local_size, 1, 1, 0, stream_,
                     captured_arguments, nullptr),
              hipSuccess);
  }
  ASSERT_EQ(end_capture(stream_, &graph_), hipSuccess);
  ASSERT_EQ(instantiate(&graph_executable_, graph_, nullptr, nullptr, 0),
            hipSuccess);
  ASSERT_EQ(launch_graph(graph_executable_, stream_), hipSuccess);
  ASSERT_EQ(copy(actual.data(), output_, sizeof(actual), hipMemcpyDeviceToHost,
                 stream_),
            hipSuccess);
  ASSERT_EQ(synchronize(stream_), hipSuccess);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], i % kWorkitems + 1);
  }
}

}  // namespace
