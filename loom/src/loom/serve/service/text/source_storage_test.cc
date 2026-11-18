// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/execution/workspace.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class SourceStorageTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void Prepare(bool reject_after_copy) {
    const auto prepare =
        iree::testing::MakeTempFilePath("text_storage_prepare", ".loom");
    const auto control =
        iree::testing::MakeTempFilePath("text_storage_control", ".loom");
    ASSERT_NO_FATAL_FAILURE(
        EditNativeProvider(directory_ + "/prepare.loom", prepare,
                           "config.def @fixture.card = false : i1", R"(
func.decl import("prepare", "storage") @declare_storage(%bytes: i64, %alignment: i64) -> (i32)
func.def @declare_output_storage() {
  %bytes = scalar.constant 64 : i64
  %alignment = scalar.constant 64 : i64
  %ordinal = func.call @declare_storage(%bytes, %alignment) : (i64, i64) -> (i32)
  func.return
}
config.def @fixture.card = false : i1
)"));
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        prepare, prepare, "%c0_source = scalar.constant 0 : i32",
        "func.call @declare_output_storage() : ()\n"
        "  %c0_source = scalar.constant 0 : i32"));
    const std::string declaration =
        "func.decl import(\"input\") @require(%condition: i32, "
        "%diagnostic: buffer)";
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/control.loom", control, declaration, declaration + R"(
func.decl import("storage", "buffer") @stored_buffer(%ordinal: i32) -> (hal.buffer)
func.decl import("storage", "copy") @stored_copy(%source: hal.buffer, %source_offset: i64, %target: hal.buffer, %target_offset: i64, %length: i64) -> (i64)
)"));
    const std::string feedback =
        "%feedback = func.call @feedback(%c0, %outputs, %c0_i64, %length) : "
        "(i32, hal.buffer, i64, i64) -> (i64)";
    std::string replacement = R"(
  %retained = func.call @stored_buffer(%c0) : (i32) -> (hal.buffer)
  %copy = func.call @stored_copy(%outputs, %c0_i64, %retained, %c0_i64, %length) : (hal.buffer, i64, hal.buffer, i64, i64) -> (i64)
)";
    if (reject_after_copy) {
      replacement += R"(
  %error = buffer.constant utf8("rejected after retained output copy") : buffer
  func.call @require(%c0, %error) : (i32, buffer)
)";
    }
    replacement +=
        "%feedback = func.call @feedback(%c0, %retained, %c0_i64, %length) : "
        "(i32, hal.buffer, i64, i64) -> (i64)";
    ASSERT_NO_FATAL_FAILURE(
        EditNativeProvider(control, control, feedback, replacement));
    auto options = ModelOptions();
    options.sources.prepare = iree_make_cstring_view(prepare.c_str());
    options.sources.control = iree_make_cstring_view(control.c_str());
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model_));
  }
};

TEST_F(SourceStorageTest, OutputUsesStableRootsAcrossRequestsAndRowGrowth) {
  ASSERT_NO_FATAL_FAILURE(Prepare(false));
  ASSERT_EQ(model_->state.declared_storage.count, 1u);
  auto* root = model_->state.declared_storage.buffers[0];
  ASSERT_NE(root, nullptr);
  ASSERT_NE(model_->control.storage_module, nullptr);
  EXPECT_EQ(iree_hal_buffer_byte_length(root), 64u);
  auto* row = loom_serve_text_model_row(model_, 0);
  const int32_t token = Oracle::Id('a');
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  Oracle first;
  first.Consume("a");
  EXPECT_EQ(loom_serve_text_row_token(row), first.Predict());

  loom_serve_text_resize_t* resize = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_prepare_resize(model_, 8, &resize));
  ASSERT_NE(resize, nullptr);
  loom_runtime_execution_frontier_t frontier;
  auto status = loom_serve_text_model_submit_resize(model_, resize, &frontier);
  loom_serve_text_resize_free(resize);
  IREE_ASSERT_OK(status);
  for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
    IREE_ASSERT_OK(loom_runtime_workspace_wait(
        loom_runtime_pool_workspace(storage_owner_), frontier.semaphores[i],
        frontier.values[i]));
  }
  loom_serve_text_model_complete_resize(model_);
  EXPECT_EQ(model_->state.declared_storage.buffers[0], root);
  EXPECT_EQ(loom_serve_text_model_row(model_, 0), row);

  auto* next = loom_serve_text_model_row(model_, 7);
  const int32_t prompt[] = {Oracle::Id('b'), Oracle::Id('x')};
  IREE_ASSERT_OK(loom_serve_text_row_prefill(next, 2, prompt));
  Oracle second;
  second.Consume("bx");
  EXPECT_EQ(loom_serve_text_row_token(next), second.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_decode(row));
  first.Consume(first.Predict());
  EXPECT_EQ(loom_serve_text_row_token(row), first.Predict());
}

TEST_F(SourceStorageTest, SourceFailurePreservesEscapedRootsUntilRetirement) {
  ASSERT_NO_FATAL_FAILURE(Prepare(true));
  ASSERT_EQ(model_->state.declared_storage.count, 1u);
  auto* root = model_->state.declared_storage.buffers[0];
  // An ordinary native borrower can outlive the failed source invocation.
  // Keeping it explicit makes the retirement assertion independent of timing.
  iree_hal_buffer_retain(root);
  const int32_t token = Oracle::Id('a');
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_row_prefill(
                            loom_serve_text_model_row(model_, 0), 1, &token));
  loom_serve_text_model_retire(model_, nullptr, nullptr);
  EXPECT_FALSE(loom_serve_text_model_is_retired(model_));
  iree_hal_buffer_release(root);
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
}

}  // namespace
