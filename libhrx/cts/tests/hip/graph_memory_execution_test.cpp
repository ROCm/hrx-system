// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

class HipGraphMemoryExecutionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(library_.Open()) << library_.error();
    const auto initialize = library_.Resolve<decltype(&hipInit)>("hipInit");
    ASSERT_NE(nullptr, initialize);
    ASSERT_EQ(hipSuccess, initialize(0));
  }

  static void TearDownTestSuite() {
    if (library_.is_open()) {
      const auto deinitialize =
          library_.Resolve<hipError_t (*)()>("hipHALDeinit");
      ASSERT_NE(nullptr, deinitialize);
      EXPECT_EQ(hipSuccess, deinitialize());
      EXPECT_TRUE(library_.Close()) << library_.error();
    }
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, library_.Resolve<decltype(&hipStreamSynchronize)>(
                                "hipStreamSynchronize")(stream_));
    }
    if (executable_) {
      EXPECT_EQ(hipSuccess, library_.Resolve<decltype(&hipGraphExecDestroy)>(
                                "hipGraphExecDestroy")(executable_));
    }
    if (graph_) {
      EXPECT_EQ(hipSuccess, library_.Resolve<decltype(&hipGraphDestroy)>(
                                "hipGraphDestroy")(graph_));
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, library_.Resolve<decltype(&hipModuleUnload)>(
                                "hipModuleUnload")(module_));
    }
    if (stream_) {
      EXPECT_EQ(hipSuccess, library_.Resolve<decltype(&hipStreamDestroy)>(
                                "hipStreamDestroy")(stream_));
    }
  }

  // One public HIP runtime shared through synchronous suite teardown.
  static inline hrx::hip::testing::HipDso library_;
  // Stream joined before any graph, module, or host storage is released.
  hipStream_t stream_ = nullptr;
  // Template retaining the stable virtual address.
  hipGraph_t graph_ = nullptr;
  // Compiled replay, including copied opaque kernel arguments.
  hipGraphExec_t executable_ = nullptr;
  // Native executable providing the numeric kernel.
  hipModule_t module_ = nullptr;
  // Host input remains alive through teardown on assertion failures.
  std::array<uint32_t, 4> input_ = {1, 2, 3, 4};
  // Host output remains alive through teardown on assertion failures.
  std::array<uint32_t, 4> output_ = {};
};

TEST_F(HipGraphMemoryExecutionTest, OpaqueKernelArgumentsSurviveTrimAndReplay) {
#define RESOLVE(name)                                           \
  const auto name = library_.Resolve<decltype(&::name)>(#name); \
  ASSERT_NE(nullptr, name)
  RESOLVE(hipGetDevice);
  RESOLVE(hipGetDevicePropertiesR0600);
  RESOLVE(hipModuleLoadData);
  RESOLVE(hipModuleGetFunction);
  RESOLVE(hipStreamCreateWithFlags);
  RESOLVE(hipStreamSynchronize);
  RESOLVE(hipGraphCreate);
  RESOLVE(hipGraphAddMemAllocNode);
  RESOLVE(hipGraphAddMemcpyNode1D);
  RESOLVE(hipGraphAddKernelNode);
  RESOLVE(hipGraphAddMemFreeNode);
  RESOLVE(hipGraphInstantiateWithFlags);
  RESOLVE(hipGraphLaunch);
  RESOLVE(hipDeviceGraphMemTrim);
#undef RESOLVE

  int device = -1;
  ASSERT_EQ(hipSuccess, hipGetDevice(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, hipGetDevicePropertiesR0600(&properties, device));
  const auto image =
      hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
  ASSERT_NE(nullptr, image.file)
      << "no embedded HSACO for " << properties.gcnArchName;
  ASSERT_EQ(hipSuccess, hipModuleLoadData(&module_, image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess, hipModuleGetFunction(&function, module_,
                                             "hrx_transform_nested_pointers"));
  ASSERT_EQ(hipSuccess,
            hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess, hipGraphCreate(&graph_, 0));

  hipMemAllocNodeParams allocation = {};
  allocation.poolProps.allocType = hipMemAllocationTypePinned;
  allocation.poolProps.location.type = hipMemLocationTypeDevice;
  allocation.poolProps.location.id = device;
  allocation.bytesize = sizeof(input_) + sizeof(output_);
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, hipGraphAddMemAllocNode(&allocation_node, graph_,
                                                nullptr, 0, &allocation));
  auto* input = static_cast<uint32_t*>(allocation.dptr);
  auto* output = input + input_.size();
  hipGraphNode_t upload = nullptr;
  ASSERT_EQ(hipSuccess,
            hipGraphAddMemcpyNode1D(&upload, graph_, &allocation_node, 1, input,
                                    input_.data(), sizeof(input_),
                                    hipMemcpyHostToDevice));
  struct {
    // The first pointer is nested in a by-value kernel argument.
    uint32_t* input;
    // The second is an interior pointer into the same graph allocation.
    uint32_t* output;
  } pointers = {input, output};
  uint32_t scale = 5;
  uint32_t offset = 2;
  void* arguments[] = {&pointers, &scale, &offset};
  const hipKernelNodeParams parameters = {
      .blockDim = {1, 1, 1},
      .func = function,
      .gridDim = {1, 1, 1},
      .kernelParams = arguments,
  };
  hipGraphNode_t kernel = nullptr;
  ASSERT_EQ(hipSuccess,
            hipGraphAddKernelNode(&kernel, graph_, &upload, 1, &parameters));
  hipGraphNode_t download = nullptr;
  ASSERT_EQ(hipSuccess, hipGraphAddMemcpyNode1D(
                            &download, graph_, &kernel, 1, output_.data(),
                            output, sizeof(output_), hipMemcpyDeviceToHost));
  hipGraphNode_t free_node = nullptr;
  ASSERT_EQ(hipSuccess, hipGraphAddMemFreeNode(&free_node, graph_, &download, 1,
                                               allocation.dptr));
  ASSERT_EQ(hipSuccess, hipGraphInstantiateWithFlags(&executable_, graph_, 0));
  for (uint32_t replay = 0; replay < 3; ++replay) {
    ASSERT_EQ(hipSuccess, hipGraphLaunch(executable_, stream_));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream_));
    for (size_t i = 0; i < input_.size(); ++i) {
      EXPECT_EQ(input_[i] * scale + offset, output_[i]);
      input_[i] += 7;
    }
    // Return the actual backing while the recorded kernel keeps the same
    // opaque pointers. The next launch must remap before executing it.
    ASSERT_EQ(hipSuccess, hipDeviceGraphMemTrim(device));
  }
}

}  // namespace
