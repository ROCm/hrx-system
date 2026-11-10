// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

}  // namespace
