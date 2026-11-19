// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>

#include "binding/hip/hip_graph_memory_test.h"
#include "iree/testing/gtest.h"

namespace {

using hrx::hip::testing::HipGraphMemoryApiTest;

TEST_F(HipGraphMemoryApiTest, FillGeometrySurvivesCloneAndParameterUpdates) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  hipMemsetParams params = {
      .dst = static_cast<uint8_t*>(device_) + 4,
      .elementSize = 1,
      .height = 3,
      .pitch = 16,
      .value = 0x7953A216u,
      .width = 2,
  };
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_fill(&node, graph_, nullptr, 0, &params));
  const auto check = [&]() {
    source_.fill(0xCD);
    ASSERT_EQ(hipSuccess, api_.copy(device_, source_.data(), source_.size(),
                                    hipMemcpyHostToDevice));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess,
              api_.copy(destination_.data(), device_, destination_.size(),
                        hipMemcpyDeviceToHost));
    auto expected = source_;
    const size_t width = params.width * params.elementSize;
    const size_t pitch = params.pitch ? params.pitch : width;
    const auto* pattern = reinterpret_cast<const uint8_t*>(&params.value);
    for (size_t y = 0; y < params.height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        expected[4 + y * pitch + x] = pattern[x % params.elementSize];
      }
    }
    EXPECT_EQ(destination_, expected);
  };
  for (unsigned int pattern_size : {1u, 2u, 4u}) {
    for (size_t pitch : {size_t{0}, size_t{16}}) {
      SCOPED_TRACE(pattern_size);
      SCOPED_TRACE(pitch);
      params.elementSize = pattern_size;
      params.pitch = pitch;
      params.value = 0x7953A216u;
      ASSERT_EQ(hipSuccess, api_.set_fill(node, &params));
      ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));
      hipGraphNode_t clone_node = nullptr;
      ASSERT_EQ(hipSuccess, api_.find_clone_node(&clone_node, node, clone_));
      hipMemsetParams observed = {};
      ASSERT_EQ(hipSuccess, api_.get_fill(clone_node, &observed));
      EXPECT_EQ(observed.dst, params.dst);
      EXPECT_EQ(observed.elementSize, params.elementSize);
      EXPECT_EQ(observed.height, params.height);
      EXPECT_EQ(observed.pitch, params.pitch);
      EXPECT_EQ(observed.width, params.width);
      EXPECT_EQ(observed.value, params.value);
      ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
      check();
      ASSERT_FALSE(HasFatalFailure());
      params.value = 0x12CA9386u;
      ASSERT_EQ(hipSuccess,
                api_.set_executable_fill(executable_, clone_node, &params));
      check();
      ASSERT_FALSE(HasFatalFailure());
      params.value = 0x8537AB24u;
      ASSERT_EQ(hipSuccess, api_.set_fill(clone_node, &params));
      hipGraphNode_t error_node = nullptr;
      hipGraphExecUpdateResult result = hipGraphExecUpdateError;
      ASSERT_EQ(hipSuccess, api_.update_executable(executable_, clone_,
                                                   &error_node, &result));
      EXPECT_EQ(result, hipGraphExecUpdateSuccess);
      EXPECT_EQ(error_node, nullptr);
      check();
      ASSERT_FALSE(HasFatalFailure());
      ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
      executable_ = nullptr;
      ASSERT_EQ(hipSuccess, api_.graph_destroy(clone_));
      clone_ = nullptr;
    }
  }
}

TEST_F(HipGraphMemoryApiTest,
       FillRejectsMisalignedRowsAndRetainsValidParameters) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  hipMemsetParams params = {
      .dst = device_,
      .elementSize = 4,
      .height = 3,
      .pitch = 16,
      .value = 0x3514AB29u,
      .width = 2,
  };
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_fill(&node, graph_, nullptr, 0, &params));
  for (bool misalign_origin : {false, true}) {
    auto invalid = params;
    if (misalign_origin) {
      invalid.dst = static_cast<uint8_t*>(device_) + 1;
    } else {
      // The whole span is pattern-aligned, but its middle row is not.
      invalid.pitch = 10;
    }
    hipGraphNode_t rejected = nullptr;
    EXPECT_EQ(hipErrorInvalidValue,
              api_.add_fill(&rejected, graph_, nullptr, 0, &invalid));
    EXPECT_EQ(hipErrorInvalidValue, api_.set_fill(node, &invalid));
  }
  hipMemsetParams observed = {};
  ASSERT_EQ(hipSuccess, api_.get_fill(node, &observed));
  EXPECT_EQ(observed.dst, params.dst);
  EXPECT_EQ(observed.pitch, params.pitch);
  size_t count = 0;
  ASSERT_EQ(hipSuccess, api_.get_nodes(graph_, nullptr, &count));
  EXPECT_EQ(count, 1u);
}

TEST_F(HipGraphMemoryApiTest, CapturedPitchedFillLeavesPaddingUntouched) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  ASSERT_EQ(hipSuccess,
            api_.begin_capture_to_graph(stream_, graph_, nullptr, nullptr, 0,
                                        hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipSuccess, api_.fill_2d_async(static_cast<uint8_t*>(device_) + 3,
                                           13, 0xA5, 7, 3, stream_));
  hipGraph_t captured = nullptr;
  ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &captured));
  ASSERT_EQ(captured, graph_);
  size_t count = 0;
  ASSERT_EQ(hipSuccess, api_.get_nodes(graph_, nullptr, &count));
  ASSERT_EQ(count, 1u);
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  for (int iteration = 0; iteration < 2; ++iteration) {
    source_.fill(0x35);
    ASSERT_EQ(hipSuccess, api_.copy(device_, source_.data(), source_.size(),
                                    hipMemcpyHostToDevice));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess,
              api_.copy(destination_.data(), device_, destination_.size(),
                        hipMemcpyDeviceToHost));
    auto expected = source_;
    for (size_t y = 0; y < 3; ++y) {
      for (size_t x = 0; x < 7; ++x) {
        expected[3 + y * 13 + x] = 0xA5;
      }
    }
    EXPECT_EQ(destination_, expected);
  }
}

}  // namespace
