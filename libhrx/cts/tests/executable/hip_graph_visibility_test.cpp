// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HipGraphVisibilityTest
    : public testing::TestWithParam<
          std::tuple<uint32_t, uint32_t, uint32_t, bool>> {
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
    for (void* buffer : buffers_) {
      if (buffer) {
        EXPECT_EQ(hipSuccess,
                  library_.Resolve<decltype(&hipFree)>("hipFree")(buffer));
      }
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

  // Exact build artifact, retained through synchronous suite teardown.
  static inline hrx::hip::testing::HipDso library_;
  // Launch stream joined before assertion unwinding releases any resources.
  hipStream_t stream_ = nullptr;
  // Immutable template containing independent chains and their final join.
  hipGraph_t graph_ = nullptr;
  // Compiled replay retaining copied kernel parameters.
  hipGraphExec_t executable_ = nullptr;
  // Loom-authored kernel image retained through every graph execution.
  hipModule_t module_ = nullptr;
  // Immutable input, two alternating intermediates, and final output.
  std::array<void*, 4> buffers_ = {};
};

TEST_P(HipGraphVisibilityTest,
       JoinedChainsPublishEveryPayloadBeforeConsumption) {
#define RESOLVE(name)                                           \
  const auto name = library_.Resolve<decltype(&::name)>(#name); \
  ASSERT_NE(nullptr, name)
  RESOLVE(hipGetDevice);
  RESOLVE(hipGetDevicePropertiesR0600);
  RESOLVE(hipModuleLoadData);
  RESOLVE(hipModuleGetFunction);
  RESOLVE(hipStreamCreateWithFlags);
  RESOLVE(hipStreamSynchronize);
  RESOLVE(hipMalloc);
  RESOLVE(hipMemcpy);
  RESOLVE(hipGraphCreate);
  RESOLVE(hipGraphAddKernelNode);
  RESOLVE(hipGraphAddMemcpyNode1D);
  RESOLVE(hipGraphAddEmptyNode);
  RESOLVE(hipGraphInstantiateWithFlags);
  RESOLVE(hipGraphLaunch);
#undef RESOLVE

  const auto [width, depth, leaf_count, transfer_edges] = GetParam();
  const uint32_t count = width * leaf_count;
  const size_t byte_length = count * sizeof(uint32_t);
  int device = -1;
  ASSERT_EQ(hipSuccess, hipGetDevice(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, hipGetDevicePropertiesR0600(&properties, device));
  iree_hal_amd_gpu_target_identity_t identity;
  IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
      iree_make_cstring_view(properties.gcnArchName), &identity));
  const auto images =
      iree::hal::cts::CtsRegistry::ListExecutableTargets("hrx_dependent_chain");
  const auto* image =
      iree::hal::amd::gpu::cts::SelectExecutableTarget(images, identity);
  ASSERT_NE(nullptr, image);
  const auto data = image->data_fn(IREE_SV("dependent_chain.bin"));
  ASSERT_EQ(hipSuccess, hipModuleLoadData(&module_, data.data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            hipModuleGetFunction(&function, module_, "dependent_chain"));
  ASSERT_EQ(hipSuccess,
            hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  for (auto& buffer : buffers_) {
    ASSERT_EQ(hipSuccess, hipMalloc(&buffer, byte_length));
  }
  ASSERT_EQ(hipSuccess, hipGraphCreate(&graph_, 0));

  const auto add_kernel = [&](const hipGraphNode_t* dependencies,
                              size_t dependency_count, void* source,
                              void* target, uint32_t length, uint32_t step,
                              hipGraphNode_t* out_node) {
    void* arguments[] = {&source, &target, &length, &step};
    const hipKernelNodeParams parameters = {
        .blockDim = {256, 1, 1},
        .func = function,
        .gridDim = {length / 256, 1, 1},
        .kernelParams = arguments,
    };
    return hipGraphAddKernelNode(out_node, graph_, dependencies,
                                 dependency_count, &parameters);
  };
  // The large variant uses native DMA and the small variant may use a shader
  // copy. Both are transfer operations at the graph/HAL boundary.
  hipGraphNode_t input_copy = nullptr;
  if (transfer_edges) {
    ASSERT_EQ(hipSuccess,
              hipGraphAddMemcpyNode1D(&input_copy, graph_, nullptr, 0,
                                      buffers_[2], buffers_[0], byte_length,
                                      hipMemcpyDeviceToDevice));
  }
  std::vector<hipGraphNode_t> tails(width, input_copy);
  for (uint32_t phase = 0; phase < depth; ++phase) {
    const uint32_t source =
        phase ? 1 + ((phase - 1) & 1) : (transfer_edges ? 2 : 0);
    const uint32_t target = 1 + (phase & 1);
    for (uint32_t lane = 0; lane < width; ++lane) {
      hipGraphNode_t node = nullptr;
      ASSERT_EQ(
          hipSuccess,
          add_kernel(
              &tails[lane], tails[lane] ? 1 : 0,
              static_cast<uint32_t*>(buffers_[source]) + lane * leaf_count,
              static_cast<uint32_t*>(buffers_[target]) + lane * leaf_count,
              leaf_count, phase + 1, &node));
      tails[lane] = node;
    }
  }
  hipGraphNode_t join = nullptr;
  ASSERT_EQ(hipSuccess,
            hipGraphAddEmptyNode(&join, graph_, tails.data(), tails.size()));
  hipGraphNode_t final = nullptr;
  ASSERT_EQ(hipSuccess, add_kernel(&join, 1, buffers_[1 + ((depth - 1) & 1)],
                                   buffers_[3], count, 9001, &final));
  if (transfer_edges) {
    hipGraphNode_t output_copy = nullptr;
    ASSERT_EQ(hipSuccess,
              hipGraphAddMemcpyNode1D(&output_copy, graph_, &final, 1,
                                      buffers_[0], buffers_[3], byte_length,
                                      hipMemcpyDeviceToDevice));
  }
  ASSERT_EQ(hipSuccess, hipGraphInstantiateWithFlags(&executable_, graph_, 0));

  // Every replay changes every input. The final consumer reads neighbors
  // across workgroup and leaf boundaries, exposing stale block payloads.
  std::vector<uint32_t> expected(count), next(count), actual(count);
  for (uint32_t replay = 0; replay < 3; ++replay) {
    for (uint32_t i = 0; i < count; ++i) {
      expected[i] = i * 747796405u + 2891336453u + replay * 131u;
    }
    ASSERT_EQ(hipSuccess, hipMemcpy(buffers_[0], expected.data(), byte_length,
                                    hipMemcpyHostToDevice));
    for (uint32_t phase = 0; phase <= depth; ++phase) {
      const uint32_t length = phase == depth ? count : leaf_count;
      const uint32_t step = phase == depth ? 9001 : phase + 1;
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t adjacent =
            expected[(i / length) * length + ((i + 257) & (length - 1))];
        next[i] = (expected[i] ^ ((adjacent << 1) | (adjacent >> 31))) + step;
      }
      expected.swap(next);
    }
    ASSERT_EQ(hipSuccess, hipGraphLaunch(executable_, stream_));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream_));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(actual.data(), buffers_[transfer_edges ? 0 : 3],
                        byte_length, hipMemcpyDeviceToHost));
    ASSERT_EQ(actual, expected) << "replay " << replay;
  }
}

INSTANTIATE_TEST_SUITE_P(IndependentChains, HipGraphVisibilityTest,
                         testing::Combine(testing::Values(4u, 8u),
                                          testing::Values(9u, 32u),
                                          testing::Values(4096u, 262144u),
                                          testing::Bool()));

}  // namespace
