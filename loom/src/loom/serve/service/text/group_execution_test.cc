// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {
using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class GroupTextServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.group_mode = LOOM_SERVE_GROUP_MODE_TWO_BANK;
    return options;
  }
};

TEST_F(GroupTextServiceTest, BanksOwnPacketsAndPrivateRootsShareSequenceState) {
  EXPECT_GT(model_->stage_workspace_bytes, 0u);
  EXPECT_GE(iree_hal_buffer_byte_length(
                model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE]),
            model_->stage_workspace_bytes);
  EXPECT_GE(iree_hal_buffer_byte_length(
                model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE]),
            model_->stage_workspace_bytes);
  for (unsigned i = 0; i < LOOM_SERVE_TEXT_BANK_ROOT_COUNT; ++i) {
    EXPECT_NE(model_->banks[0].roots[i], nullptr);
    EXPECT_NE(model_->banks[1].roots[i], nullptr);
    EXPECT_NE(model_->banks[0].roots[i], model_->banks[1].roots[i]);
  }
  for (unsigned i = 0; i < LOOM_SERVE_TEXT_BANK_HOST_COUNT; ++i) {
    EXPECT_NE(model_->banks[0].host[i], model_->banks[1].host[i]);
  }
  EXPECT_EQ(model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_RESIDUAL],
            model_->state.residual);
  EXPECT_EQ(model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_METADATA],
            model_->state.epoch.buffers[1]);
}

TEST_F(GroupTextServiceTest, UnequalPairPublishesOnceThenDecodesGoldenTokens) {
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_serve_execution_frontier_t frontier;
  auto* row = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  EXPECT_EQ(loom_serve_text_row_position(row), 0u);
  EXPECT_EQ(row->state->source_position, 0u);
  loom_serve_execution_frontier_t rejected;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &rejected));
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));
  loom_serve_text_result_t result;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &result));
  EXPECT_EQ(result.known_count, 5u);
  EXPECT_EQ(result.consumed_count, 5u);
  EXPECT_EQ(result.output_count, 1u);
  EXPECT_EQ(loom_serve_text_row_position(row), 5u);
  EXPECT_EQ(row->state->source_position, 10u);
  Oracle oracle;
  for (auto token : tokens) {
    oracle.Consume(token);
  }
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
    oracle.Consume(oracle.Predict());
    IREE_ASSERT_OK(loom_serve_text_row_decode(row));
  }
}

TEST_F(GroupTextServiceTest, RejectsInvalidPairsBeforeEnqueue) {
  const int32_t tokens[] = {0, 1, 2};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 1, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 1,
                              .token_ids = tokens + 1,
                              .input_offset = 2};
  const auto before = loom_serve_execution_frontier(model_->execution);
  loom_serve_execution_frontier_t frontier;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  b.input_offset = 1;
  b.flags = LOOM_SERVICE_TEXT_SPAN_FLAG_PROPOSE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  b.flags = 0;
  b.row_index = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  b.row_index = 0;
  a.flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  EXPECT_EQ(loom_serve_execution_frontier(model_->execution).values[0],
            before.values[0]);
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            0u);
}

TEST_F(GroupTextServiceTest, SequentialAndSingleModelsHaveIndependentSettings) {
  auto options = ModelOptions();
  options.group_mode = LOOM_SERVE_GROUP_MODE_SINGLE;
  loom_serve_text_model_t* single = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &single));
  EXPECT_FALSE(loom_serve_text_model_two_bank(single));
  EXPECT_TRUE(loom_serve_text_model_two_bank(model_));
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  auto* reference = loom_serve_text_model_row(single, 0);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(reference, 5, tokens));
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
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, nullptr));
  EXPECT_EQ(loom_serve_text_row_token(reference),
            loom_serve_text_row_token(loom_serve_text_model_row(model_, 0)));
  uint32_t recurrence[2] = {};
  for (unsigned i = 0; i < 2; ++i) {
    auto* current = i ? single : model_;
    iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {
            .source_buffer = current->rows[0]->state->buffers[TEXT_RECURRENT],
            .target = &recurrence[i],
            .length = 4}};
    uint64_t value = 0;
    IREE_ASSERT_OK(loom_serve_execution_transfer(current->execution, 1,
                                                 &download, &value));
    IREE_ASSERT_OK(loom_serve_execution_wait(current->execution, value));
  }
  EXPECT_EQ(recurrence[0], recurrence[1]);
  IREE_ASSERT_OK(loom_serve_text_model_free(single));
  options.group_mode = LOOM_SERVE_GROUP_MODE_TWO_BANK;
  options.plan_provider = IREE_SV("sequential_v1");
  loom_serve_text_model_t* sequential = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &sequential));
  IREE_ASSERT_OK(loom_serve_text_model_submit_epoch_pair(sequential, 0, &a, 0,
                                                         &b, &frontier));
  IREE_ASSERT_OK(
      loom_serve_execution_wait(sequential->execution, frontier.values[0]));
  loom_serve_text_result_t result;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(sequential, &result));
  EXPECT_EQ(result.tokens[0],
            loom_serve_text_row_token(loom_serve_text_model_row(model_, 0)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_text_model_complete_epoch(sequential, &result));
  IREE_ASSERT_OK(loom_serve_text_model_free(sequential));
}

TEST_F(GroupTextServiceTest,
       GroupedCheckpointForkMatchesSingleStateAndKeepsPrefix) {
  const int32_t prefix[] = {0, 1, 2, 0};
  const int32_t extension[] = {1, 2, 0, 1, 0};
  auto* original = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(original, 4, prefix));
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(original, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  for (unsigned index : {1u, 2u}) {
    bool restored = false;
    IREE_ASSERT_OK(loom_serve_text_row_try_restore(
        loom_serve_text_model_row(model_, index), checkpoint, &restored));
    ASSERT_TRUE(restored);
  }
  loom_serve_text_span_t a = {
      .row_index = 1, .token_count = 3, .token_ids = extension};
  loom_serve_text_span_t b = {.row_index = 1,
                              .token_count = 2,
                              .token_ids = extension + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_serve_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 1)),
            4u);
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, nullptr));
  auto* single = loom_serve_text_model_row(model_, 2);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(single, 5, extension));
  Oracle oracle;
  for (int32_t token : prefix) {
    oracle.Consume(token);
  }
  EXPECT_EQ(loom_serve_text_row_position(original), 4u);
  EXPECT_EQ(loom_serve_text_row_token(original), oracle.Predict());
  for (int32_t token : extension) {
    oracle.Consume(token);
  }
  EXPECT_EQ(loom_serve_text_row_token(single), oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_token(loom_serve_text_model_row(model_, 1)),
            oracle.Predict());
  loom_serve_text_checkpoint_release(checkpoint);
}

TEST_F(GroupTextServiceTest,
       ResidentGrowthPreservesBanksAndUsesNewSharedMappings) {
  loom_serve_text_resize_t* resize = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_prepare_resize(model_, 6, &resize));
  ASSERT_NE(resize, nullptr);
  loom_serve_execution_frontier_t frontier;
  auto status = loom_serve_text_model_submit_resize(model_, resize, &frontier);
  loom_serve_text_resize_free(resize);
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(model_->execution,
                                                    frontier.values[1]));
  loom_serve_text_model_complete_resize(model_);
  const int32_t tokens[] = {0, 1, 2};
  loom_serve_text_span_t a = {
      .row_index = 5, .token_count = 2, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 5,
                              .token_count = 1,
                              .token_ids = tokens + 2,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 2};
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
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 5)),
            3u);
}

TEST_F(GroupTextServiceTest,
       RetirementKeepsAcceptedGroupPayloadsUntilQueuesReleaseThem) {
  struct Gate {
    iree_hal_semaphore_t* semaphore = nullptr;
    ~Gate() {
      if (semaphore) {
        iree_status_ignore(iree_hal_semaphore_signal(semaphore, 1, nullptr));
        iree_hal_semaphore_release(semaphore);
      }
    }
  } gate;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate.semaphore));
  uint64_t one = 1;
  IREE_ASSERT_OK(loom_serve_execution_barrier(model_->execution,
                                              {1, &gate.semaphore, &one}));
  const int32_t tokens[] = {0, 1};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 1, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 1,
                              .token_ids = tokens + 1,
                              .input_offset = 1};
  loom_serve_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            0u);
  loom_serve_text_model_retire(model_, nullptr, nullptr);
  EXPECT_FALSE(loom_serve_text_model_is_retired(model_));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphore, 1, nullptr));
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
}

TEST_F(GroupTextServiceTest,
       RejectedGroupNodePoisonsModelAndRetiresAcceptedPrefix) {
  struct Gate {
    iree_hal_semaphore_t* semaphore = nullptr;
    ~Gate() {
      if (semaphore) {
        iree_status_ignore(iree_hal_semaphore_signal(semaphore, 1, nullptr));
        iree_hal_semaphore_release(semaphore);
      }
    }
  } gate;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate.semaphore));
  uint64_t one = 1;
  IREE_ASSERT_OK(loom_serve_execution_barrier(model_->execution,
                                              {1, &gate.semaphore, &one}));
  // B's token root keeps its extent but cannot be a transfer target. The HAL
  // rejects that upload after A's two packets and B's metadata were accepted.
  iree_hal_buffer_t** tokens =
      &model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_TOKENS];
  const iree_device_size_t length = iree_hal_buffer_byte_length(*tokens);
  auto* storage = new uint8_t[length]();
  iree_hal_buffer_t* rejected = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE, length,
      iree_make_byte_span(storage, length),
      {[](void* user_data, iree_hal_buffer_t*) {
         delete[] static_cast<uint8_t*>(user_data);
       },
       storage},
      iree_allocator_system(), &rejected));
  iree_hal_buffer_release(*tokens);
  *tokens = rejected;
  // Test-held views are dropped before model free joins state retirement.
  using Borrow =
      std::unique_ptr<iree_hal_buffer_t, void (*)(iree_hal_buffer_t*)>;
  auto borrow = [](iree_hal_buffer_t* buffer) {
    iree_hal_buffer_retain(buffer);
    return Borrow(buffer, iree_hal_buffer_release);
  };
  Borrow rejected_borrow = borrow(rejected);
  Borrow metadata =
      borrow(model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_METADATA]);
  const iree_byte_span_t outputs =
      model_->banks[1].bytes[LOOM_SERVE_TEXT_BANK_HOST_OUTPUTS];
  memset(outputs.data, 0xA5, outputs.data_length);

  const auto before = loom_serve_execution_frontier(model_->execution);
  const int32_t ids[] = {0, 1, 2, 0};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 2, .token_ids = ids};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = ids + 2,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 2};
  loom_serve_execution_frontier_t frontier;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  // State growth precedes the group; A's packets and B's metadata follow.
  EXPECT_EQ(frontier.semaphores[0], before.semaphores[0]);
  EXPECT_GE(frontier.values[0], before.values[0] + 3);
  EXPECT_EQ(frontier.values[1], before.values[1]);
  const std::vector<uint8_t> packet(
      model_->banks[1].bytes[LOOM_SERVE_TEXT_BANK_HOST_METADATA].data,
      model_->banks[1].bytes[LOOM_SERVE_TEXT_BANK_HOST_METADATA].data +
          model_->banks[1]
              .bytes[LOOM_SERVE_TEXT_BANK_HOST_METADATA]
              .data_length);
  auto* row = loom_serve_text_model_row(model_, 0);
  EXPECT_EQ(loom_serve_text_row_position(row), 0u);
  EXPECT_EQ(row->state->source_position, 0u);

  // Neither the group nor the ordinary execution path replays numerical work.
  loom_serve_execution_frontier_t retry;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &retry));
  const int32_t token = 1;
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {
          .source = &token,
          .target_buffer = model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_TOKENS],
          .length = sizeof(token)}};
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_execution_transfer(model_->execution, 1, &upload, &value));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_execution_barrier(
                            model_->execution, {1, &gate.semaphore, &one}));
  const auto poisoned = loom_serve_execution_frontier(model_->execution);
  EXPECT_EQ(poisoned.values[0], frontier.values[0]);
  EXPECT_EQ(poisoned.values[1], frontier.values[1]);

  // The accepted packets keep the model alive until their real retirement.
  loom_serve_text_model_retire(model_, nullptr, nullptr);
  EXPECT_FALSE(loom_serve_text_model_is_retired(model_));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphore, 1, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(model_->execution));
  // The frontier was the actual accepted value, not a promised later signal:
  // the last accepted node landed and nothing after the rejection ran.
  uint64_t signaled = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(frontier.semaphores[0], &signaled));
  EXPECT_EQ(signaled, frontier.values[0]);
  std::vector<uint8_t> uploaded(packet.size());
  const iree_hal_transfer_operation_t readback = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {.source_buffer = metadata.get(),
                   .target = uploaded.data(),
                   .length = uploaded.size()}};
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &done));
  iree_status_t status = iree_hal_queue_transfer(
      loom_serve_device_transfer_queue(device_),
      iree_hal_semaphore_list_empty(), {1, &done, &one}, 1, &readback, nullptr);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(done, 1, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(done);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(uploaded, packet);
  EXPECT_TRUE(std::all_of(storage, storage + length,
                          [](uint8_t byte) { return byte == 0; }));
  rejected_borrow.reset();
  metadata.reset();
  EXPECT_TRUE(std::all_of(outputs.data, outputs.data + outputs.data_length,
                          [](uint8_t byte) { return byte == 0xA5; }));
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
}

TEST_F(GroupTextServiceTest, ServiceGroupsLongInputAndHandlesOddTail) {
  Start(3);
  const std::string prompt = "abxababxababxa";
  const auto response = Read(Send(Request(Message("user", prompt), 5)));
  EXPECT_EQ(Content(response), Oracle::Complete(prompt, 5));
}
}  // namespace
