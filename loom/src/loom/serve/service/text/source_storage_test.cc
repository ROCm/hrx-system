// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <filesystem>
#include <vector>

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
    const char* source_path = std::getenv("LOOM_SERVE_TEST_STORAGE_SOURCE");
    ASSERT_NE(source_path, nullptr);
    const auto prepare = std::filesystem::absolute(source_path).string();
    const auto feedback_source =
        std::filesystem::path(prepare).parent_path() / "feedback.loom";
    const auto fixture_prepare =
        iree::testing::MakeTempFilePath("text_storage_prepare", ".loom");
    const auto control =
        iree::testing::MakeTempFilePath("text_storage_control", ".loom");
    const auto feedback =
        iree::testing::MakeTempFilePath("text_storage_feedback", ".loom");
    // The checked-in providers compose the ordinary fixture's preparation and
    // replace its feedback dependency. Only symbol wiring and a configuration
    // value change here; the programs themselves remain authored source.
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(directory_ + "/prepare.loom",
                                               fixture_prepare, "@prepare(",
                                               "@fixture_prepare("));
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/control.loom", control,
        "func.decl import(\"runner\") @feedback", "func.decl @feedback"));
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        feedback_source.string(), feedback, "@source_storage.reject = false",
        reject_after_copy ? "@source_storage.reject = true"
                          : "@source_storage.reject = false"));
    auto options = ModelOptions();
    std::vector<iree_string_view_t> providers(
        options.sources.values, options.sources.values + options.sources.count);
    providers.push_back(iree_make_cstring_view(fixture_prepare.c_str()));
    providers.push_back(iree_make_cstring_view(feedback.c_str()));
    options.sources.prepare = iree_make_cstring_view(prepare.c_str());
    options.sources.control = iree_make_cstring_view(control.c_str());
    options.sources.count = providers.size();
    options.sources.values = providers.data();
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
