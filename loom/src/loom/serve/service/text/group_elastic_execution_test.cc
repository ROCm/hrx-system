// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <future>

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {
using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class GroupElasticTextServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.group_mode = LOOM_SERVE_GROUP_MODE_TWO_BANK;
    return options;
  }
  loom_serve_pool_backing_t PoolBacking() const override {
    return LOOM_SERVE_POOL_BACKING_ELASTIC;
  }
};

TEST_F(GroupElasticTextServiceTest,
       BothBanksAreChargedAndRunWithinOnePhysicalBudget) {
  auto* memory = loom_serve_pool_memory(storage_owner_);
  ASSERT_NE(memory, nullptr);
  EXPECT_GT(loom_serve_memory_pool_live_bytes(memory), 0u);
  EXPECT_LE(loom_serve_memory_pool_live_bytes(memory),
            loom_serve_memory_pool_limit(memory));
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_serve_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));
  loom_serve_text_result_t result;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &result));
  Oracle oracle;
  for (int32_t token : tokens) {
    oracle.Consume(token);
  }
  EXPECT_EQ(result.tokens[0], oracle.Predict());
  EXPECT_LE(loom_serve_memory_pool_live_bytes(memory),
            loom_serve_memory_pool_limit(memory));
}

TEST_F(GroupElasticTextServiceTest,
       VariantBankWorkspacesBeyondBudgetRejectBeforeSwap) {
  auto* memory = loom_serve_pool_memory(storage_owner_);
  ASSERT_NE(memory, nullptr);
  const auto compile = [&](uint64_t bytes, loom_serve_text_variant_t** out) {
    const auto path =
        iree::testing::MakeTempFilePath("group_elastic_variant", ".loom");
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/advance.loom", path,
        "  %bytes = index.scale %spans, %word : index, offset -> offset\n"
        "  %temporary = buffer.alloca<global> align(16) %bytes : buffer\n"
        "  kernel.launch @copy_group_hidden",
        "  %bytes = index.constant " + std::to_string(bytes) +
            " : offset\n"
            "  %temporary = buffer.alloca<global> align(16) %bytes : buffer\n"
            "  kernel.launch @copy_group_hidden"));
    const auto provider = iree_make_cstring_view(path.c_str());
    const loom_serve_sources_t sources = {.count = 1, .values = &provider};
    struct Compilation {
      iree_status_t status = iree_ok_status();
      std::promise<loom_serve_text_variant_t*> ready;
    } completion;
    IREE_ASSERT_OK(loom_serve_text_model_compile_variant_async(
        model_, &sources, nullptr,
        [](void* user_data, iree_status_t status, void* variant,
           const loom_serve_compilation_report_t* report) {
          auto* completion = static_cast<Compilation*>(user_data);
          completion->status = status;
          completion->ready.set_value(
              static_cast<loom_serve_text_variant_t*>(variant));
        },
        &completion));
    *out = completion.ready.get_future().get();
    IREE_ASSERT_OK(completion.status);
  };
  using Variant = std::unique_ptr<loom_serve_text_variant_t,
                                  decltype(&loom_serve_text_variant_free)>;
  loom_serve_text_variant_t* raw = nullptr;
  ASSERT_NO_FATAL_FAILURE(compile(48 * 1024 * 1024, &raw));
  Variant oversized(raw, loom_serve_text_variant_free);
  ASSERT_NO_FATAL_FAILURE(compile(65536, &raw));
  Variant modest(raw, loom_serve_text_variant_free);

  // Two 48 MiB bank workspaces exceed the 64 MiB serving ceiling. The single
  // bank allocation transaction is never attempted and nothing is swapped.
  auto* plan = model_->execution_plan;
  iree_hal_buffer_t* workspace =
      model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE];
  const uint64_t live = loom_serve_memory_pool_live_bytes(memory);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_text_model_select_variant(model_, oversized.get()));
  EXPECT_EQ(model_->execution_plan, plan);
  EXPECT_EQ(model_->prefill_group.stages, nullptr);
  EXPECT_EQ(model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE], workspace);
  EXPECT_EQ(loom_serve_memory_pool_live_bytes(memory), live);

  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, modest.get()));
  EXPECT_GE(iree_hal_buffer_byte_length(
                model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE]),
            65536u);
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_serve_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));
  loom_serve_text_result_t result;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &result));
  Oracle oracle;
  for (int32_t token : tokens) {
    oracle.Consume(token);
  }
  EXPECT_EQ(result.tokens[0], oracle.Predict());
  EXPECT_LE(loom_serve_memory_pool_live_bytes(memory),
            loom_serve_memory_pool_limit(memory));
}

}  // namespace
