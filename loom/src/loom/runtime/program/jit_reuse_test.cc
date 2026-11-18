// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/status_matchers.h"
#include "loom/runtime/program/jit.h"
#include "loom/runtime/testing/jit.h"

namespace {

class JitClassReuseTest : public loom_runtime::testing::JitTest {
 protected:
  void Check(iree_string_view_t first, iree_string_view_t second,
             std::array<int32_t, 2> expected, size_t native_count) {
    IREE_ASSERT_OK(Compile(nullptr, 0, first));
    EXPECT_EQ(compilation_.kernels, 1u);
    IREE_ASSERT_OK(Compile(nullptr, 1, second));
    EXPECT_EQ(compilation_.commands, 2u);
    EXPECT_EQ(compilation_.kernels, native_count);
    EXPECT_EQ(compilation_.reused_kernels, 2u - native_count);
    const auto storage = loom_runtime_jit_cache_storage(jit_);
    EXPECT_EQ(storage.count, native_count);
    EXPECT_EQ(storage.executable_count, native_count);

    // Recorded commands keep native ownership after their stages retire.
    loom_runtime_jit_trim(jit_);
    EXPECT_EQ(loom_runtime_jit_cache_storage(jit_).count, native_count);
    loom_runtime_jit_release(jit_);
    jit_ = nullptr;

    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        loom_runtime_pool_storage(storage_owner_),
        iree_hal_make_pool_reservation_request(sizeof(input_), 0),
        iree_infinite_timeout(), &buffer_));
    const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
    for (size_t i = 0; i < commands_.size(); ++i) {
      SCOPED_TRACE(i);
      uint64_t completion = 0;
      const iree_hal_transfer_operation_t upload = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {input_.data(), buffer_, 0, sizeof(input_)},
      };
      IREE_ASSERT_OK(
          loom_runtime_execution_transfer(execution_, 1, &upload, &completion));
      IREE_ASSERT_OK(loom_runtime_execution_execute(
          execution_, commands_[i], {1, &binding}, &completion));
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {buffer_, 0, output_.data(), sizeof(output_)},
      };
      IREE_ASSERT_OK(loom_runtime_execution_feedback(execution_, 1, &download,
                                                     &completion));
      IREE_ASSERT_OK(
          loom_runtime_execution_feedback_wait(execution_, completion));
      EXPECT_EQ(output_[0], expected[i]);
      for (size_t j = 1; j < output_.size(); ++j) {
        EXPECT_EQ(output_[j], input_[j]) << j;
      }
    }
  }
};

TEST_P(JitClassReuseTest, DifferentFactsShareSelectedBody) {
  Check(IREE_SV("selected_four"), IREE_SV("selected_eight"), {2, 2}, 1);
}

TEST_P(JitClassReuseTest, IdenticalFactsStillReuse) {
  Check(IREE_SV("selected_four"), IREE_SV("selected_four_again"), {2, 2}, 1);
}

TEST_P(JitClassReuseTest, DifferentProvidersStayDistinct) {
  Check(IREE_SV("selected_four"), IREE_SV("selected_one"), {2, 1}, 2);
}

TEST_P(JitClassReuseTest, LiveValuesStayDistinct) {
  Check(IREE_SV("live_four"), IREE_SV("live_eight"), {6, 10}, 2);
}

TEST_P(JitClassReuseTest, InvalidSelectedBodyPublishesNoNativeRequest) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile(nullptr, 0, IREE_SV("invalid_selected")));
  EXPECT_EQ(commands_[0], nullptr);
  EXPECT_EQ(compilation_.kernels, 0u);
  EXPECT_EQ(loom_runtime_jit_cache_storage(jit_).count, 0u);
  EXPECT_GT(compilation_.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  IREE_ASSERT_OK(Compile(nullptr, 0, IREE_SV("selected_four")));
  EXPECT_EQ(compilation_.kernels, 1u);
}

TEST_P(JitClassReuseTest, RejectedParentDrainsAcceptedNatives) {
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Compile(nullptr, 0, IREE_SV("partially_invalid_selected")));
  EXPECT_EQ(commands_[0], nullptr);
  EXPECT_EQ(compilation_.kernels, 1u);
  EXPECT_EQ(loom_runtime_jit_cache_storage(jit_).count, 1u);
  IREE_ASSERT_OK(Compile(nullptr, 0, IREE_SV("selected_four")));
  EXPECT_EQ(compilation_.kernels, 1u);
  EXPECT_EQ(compilation_.reused_kernels, 1u);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitClassReuseTest,
                         ::testing::Values(1, 4));

}  // namespace
