// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>
#include <future>

#include "loom/serve/runtime/execution/execution_testing.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/prefill_group.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {
using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

using Variant = std::unique_ptr<loom_serve_text_variant_t,
                                decltype(&loom_serve_text_variant_free)>;

class GroupTextServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.group_mode = LOOM_SERVE_GROUP_MODE_TWO_BANK;
    return options;
  }

  // Writes a provider whose group residual leaf reflects |bytes| of private
  // workspace. |reselect| also changes every selection from mod 3 to mod 2.
  void WorkspaceProvider(uint64_t bytes, std::string* out_path,
                         bool reselect = false) {
    *out_path = iree::testing::MakeTempFilePath("group_variant", ".loom");
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/advance.loom", *out_path,
        "  %bytes = index.scale %spans, %word : index, offset -> offset\n"
        "  %temporary = buffer.alloca<global> align(16) %bytes : buffer\n"
        "  kernel.launch @copy_group_hidden",
        "  %bytes = index.constant " + std::to_string(bytes) +
            " : offset\n"
            "  %temporary = buffer.alloca<global> align(16) %bytes : buffer\n"
            "  kernel.launch @copy_group_hidden"));
    if (reselect) {
      ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
          *out_path, *out_path, "scalar.remui %total, %c3_i32",
          "scalar.remui %total, %c2_i32"));
    }
  }

  static void CompileVariant(loom_serve_text_model_t* model,
                             const std::string& path, Variant* out_variant) {
    const auto provider = iree_make_cstring_view(path.c_str());
    const loom_serve_sources_t sources = {.count = 1, .values = &provider};
    struct Compilation {
      iree_status_t status = iree_ok_status();
      std::promise<loom_serve_text_variant_t*> ready;
    } completion;
    IREE_ASSERT_OK(loom_serve_text_model_compile_variant_async(
        model, &sources, nullptr,
        [](void* user_data, iree_status_t status, void* variant,
           const loom_serve_compilation_report_t* report) {
          auto* completion = static_cast<Compilation*>(user_data);
          completion->status = status;
          completion->ready.set_value(
              static_cast<loom_serve_text_variant_t*>(variant));
        },
        &completion));
    out_variant->reset(completion.ready.get_future().get());
    IREE_ASSERT_OK(completion.status);
  }

  // Observable result of one unequal pair on a fresh row: B's selected token
  // and the row's recurrent state bytes.
  struct PairResult {
    int32_t token = -1;
    uint32_t recurrence = 0;
    bool operator==(const PairResult& other) const {
      return token == other.token && recurrence == other.recurrence;
    }
  };

  static void RunPair(loom_serve_text_model_t* model, iree_host_size_t row,
                      PairResult* out_result) {
    const int32_t tokens[] = {0, 1, 2, 0, 1};
    loom_serve_text_span_t a = {
        .row_index = row, .token_count = 3, .token_ids = tokens};
    loom_serve_text_span_t b = {.row_index = row,
                                .token_count = 2,
                                .token_ids = tokens + 3,
                                .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                                .input_offset = 3};
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(loom_serve_text_model_submit_epoch_pair(model, 2, &a, 1, &b,
                                                           &frontier));
    IREE_ASSERT_OK(
        loom_serve_execution_wait(model->execution, frontier.values[0]));
    loom_serve_text_result_t result;
    IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model, &result));
    ASSERT_EQ(result.output_count, 1u);
    out_result->token = result.tokens[0];
    iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {
            .source_buffer = model->rows[row]->state->buffers[TEXT_RECURRENT],
            .target = &out_result->recurrence,
            .length = sizeof(out_result->recurrence)}};
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(model->execution, 1, &download, &value));
    IREE_ASSERT_OK(loom_serve_execution_wait(model->execution, value));
  }

  static PairResult Expected() {
    Oracle oracle;
    for (int32_t token : {0, 1, 2, 0, 1}) {
      oracle.Consume(token);
    }
    return {.token = oracle.Predict()};
  }

  static iree_device_size_t WorkspaceLength(loom_serve_text_model_t* model,
                                            unsigned bank) {
    return iree_hal_buffer_byte_length(
        model->banks[bank].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE]);
  }

  static iree_status_t FailPrepare(void*, const loom_serve_plan_description_t*,
                                   iree_allocator_t,
                                   loom_serve_execution_plan_t** out_plan) {
    *out_plan = nullptr;
    return iree_make_status(IREE_STATUS_ABORTED, "injected provider failure");
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
  auto* plan = model_->execution_plan;
  iree_vm_buffer_t* control_state = model_->source.control_state;
  // A plan export failure leaves the prepared growth and every model binding
  // untouched.
  const auto provider = model_->plan_provider;
  model_->plan_provider.prepare = FailPrepare;
  loom_serve_text_resize_t* resize = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_serve_text_model_prepare_resize(model_, 6, &resize));
  model_->plan_provider = provider;
  EXPECT_EQ(resize, nullptr);
  EXPECT_EQ(model_->execution_plan, plan);
  EXPECT_EQ(model_->source.control_state, control_state);
  EXPECT_EQ(loom_serve_text_model_row_count(model_), 4u);

  IREE_ASSERT_OK(loom_serve_text_model_prepare_resize(model_, 6, &resize));
  ASSERT_NE(resize, nullptr);
  EXPECT_EQ(model_->execution_plan, plan);
  loom_serve_execution_frontier_t frontier;
  auto status = loom_serve_text_model_submit_resize(model_, resize, &frontier);
  loom_serve_text_resize_free(resize);
  IREE_ASSERT_OK(status);
  // The plan is re-exported from the resized control state and published
  // with it; bank geometry is unchanged by resident growth.
  EXPECT_NE(model_->execution_plan, plan);
  EXPECT_NE(model_->execution_plan, nullptr);
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
  // B's token root is replaced by an observable buffer of the same extent. Its
  // upload passes append validation; the queue then refuses it, as a driver
  // would after HAL validation, once A's two packets and B's metadata were
  // accepted.
  iree_hal_buffer_t** tokens =
      &model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_TOKENS];
  const iree_device_size_t length = iree_hal_buffer_byte_length(*tokens);
  auto* storage = new uint8_t[length]();
  iree_hal_buffer_t* rejected = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER, length,
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
  IREE_ASSERT_OK(
      loom_serve_execution_reject_group_node_for_testing(model_->execution, 3));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
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

TEST_F(GroupTextServiceTest, QueueFaultIsACleanGroupRejection) {
  // A storage-only bank root is refused at append, before any group node is
  // accepted: the execution is not poisoned and the model can pair again.
  iree_hal_buffer_t** tokens =
      &model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_TOKENS];
  const iree_device_size_t length = iree_hal_buffer_byte_length(*tokens);
  std::vector<uint8_t> storage(length);
  iree_hal_buffer_t* storage_only = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE, length,
      iree_make_byte_span(storage.data(), length),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &storage_only));
  iree_hal_buffer_t* original = *tokens;
  *tokens = storage_only;
  const int32_t ids[] = {0, 1, 2, 0};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 2, .token_ids = ids};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = ids + 2,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 2};
  loom_serve_execution_frontier_t frontier;
  auto status =
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier);
  *tokens = original;
  iree_hal_buffer_release(storage_only);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, status);
  EXPECT_EQ(model_->prefill_group.pending, nullptr);
  EXPECT_FALSE(model_->prefill_group.active);
  auto* row = loom_serve_text_model_row(model_, 0);
  EXPECT_EQ(loom_serve_text_row_position(row), 0u);
  EXPECT_EQ(frontier.values[0],
            loom_serve_execution_frontier(model_->execution).values[0]);
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));

  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 0, &a, 0, &b, &frontier));
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));
  loom_serve_text_result_t result;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &result));
  Oracle oracle;
  for (int32_t token : ids) {
    oracle.Consume(token);
  }
  EXPECT_EQ(result.tokens[0], oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_position(row), 4u);
  EXPECT_TRUE(std::all_of(storage.begin(), storage.end(),
                          [](uint8_t byte) { return byte == 0; }));
}

TEST_F(GroupTextServiceTest, VariantSelectionRepreparesGroupLikeFreshModel) {
  std::string path;
  ASSERT_NO_FATAL_FAILURE(WorkspaceProvider(65536, &path, /*reselect=*/true));
  Variant variant(nullptr, loom_serve_text_variant_free);
  ASSERT_NO_FATAL_FAILURE(CompileVariant(model_, path, &variant));
  auto* plan = model_->execution_plan;
  iree_hal_buffer_t* workspaces[] = {
      model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE],
      model_->banks[1].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE]};
  const iree_device_size_t original_length = WorkspaceLength(model_, 0);
  ASSERT_LT(original_length, 65536u);

  // In-flight group work refuses selection without changing anything.
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
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_text_model_select_variant(model_, variant.get()));
  EXPECT_EQ(model_->execution_plan, plan);
  EXPECT_EQ(model_->prefill_group.stages, nullptr);
  EXPECT_EQ(model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE],
            workspaces[0]);
  IREE_ASSERT_OK(
      loom_serve_execution_wait(model_->execution, frontier.values[0]));
  loom_serve_text_result_t drained;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &drained));
  EXPECT_EQ(drained.tokens[0], Expected().token);

  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, variant.get()));
  EXPECT_NE(model_->execution_plan, plan);
  EXPECT_NE(model_->prefill_group.stages, nullptr);
  for (unsigned i = 0; i < 2; ++i) {
    EXPECT_NE(model_->banks[i].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE],
              workspaces[i]);
    EXPECT_GE(WorkspaceLength(model_, i), 65536u);
  }
  // The grouped leaves now select modulo 2: B's token differs from the
  // original program's while the shared recurrence is unchanged.
  PairResult switched;
  ASSERT_NO_FATAL_FAILURE(RunPair(model_, 1, &switched));
  EXPECT_NE(switched.token, Expected().token);
  EXPECT_EQ(switched.token, 0);

  // The same commands prepared cold produce the same plan geometry, bank
  // workspaces and bytes.
  auto options = ModelOptions();
  const iree_string_view_t providers[] = {iree_make_cstring_view(path.c_str()),
                                          IREE_SV("storage.loom")};
  options.sources.count = IREE_ARRAYSIZE(providers);
  options.sources.values = providers;
  loom_serve_text_model_t* fresh_model = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &fresh_model));
  const auto release_model = [](loom_serve_text_model_t* model) {
    IREE_EXPECT_OK(loom_serve_text_model_free(model));
  };
  std::unique_ptr<loom_serve_text_model_t, decltype(release_model)> fresh(
      fresh_model, release_model);
  PairResult cold;
  ASSERT_NO_FATAL_FAILURE(RunPair(fresh_model, 1, &cold));
  EXPECT_EQ(cold, switched);
  // Ordinary decode after the grouped prefill uses the same selected code.
  auto* row = loom_serve_text_model_row(model_, 1);
  auto* cold_row = loom_serve_text_model_row(fresh_model, 1);
  for (unsigned i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(loom_serve_text_row_decode(row));
    IREE_ASSERT_OK(loom_serve_text_row_decode(cold_row));
    EXPECT_EQ(loom_serve_text_row_token(row),
              loom_serve_text_row_token(cold_row));
  }
  for (unsigned i = 0; i < 2; ++i) {
    EXPECT_EQ(WorkspaceLength(fresh_model, i), WorkspaceLength(model_, i));
  }
  EXPECT_EQ(
      loom_serve_execution_plan_node_capacity(fresh_model->execution_plan),
      loom_serve_execution_plan_node_capacity(model_->execution_plan));

  // Restoring the original stages restores the original bank workspaces.
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, nullptr));
  EXPECT_EQ(model_->prefill_group.stages, nullptr);
  EXPECT_EQ(WorkspaceLength(model_, 0), original_length);
  EXPECT_EQ(WorkspaceLength(model_, 1), original_length);
  PairResult restored;
  ASSERT_NO_FATAL_FAILURE(RunPair(model_, 2, &restored));
  EXPECT_EQ(restored.token, Expected().token);
  EXPECT_EQ(restored.recurrence, switched.recurrence);
}

TEST_F(GroupTextServiceTest, VariantFailureBeforeSwapKeepsCurrentGroup) {
  std::string path;
  ASSERT_NO_FATAL_FAILURE(WorkspaceProvider(65536, &path, /*reselect=*/true));
  Variant variant(nullptr, loom_serve_text_variant_free);
  ASSERT_NO_FATAL_FAILURE(CompileVariant(model_, path, &variant));
  auto* plan = model_->execution_plan;
  iree_hal_buffer_t* workspace =
      model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE];
  const auto expect_unchanged = [&] {
    EXPECT_EQ(model_->execution_plan, plan);
    EXPECT_EQ(model_->prefill_group.stages, nullptr);
    EXPECT_EQ(model_->banks[0].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE],
              workspace);
  };

  // Provider re-preparation fails before any allocation.
  const auto provider = model_->plan_provider;
  model_->plan_provider.prepare = FailPrepare;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_serve_text_model_select_variant(model_, variant.get()));
  model_->plan_provider = provider;
  expect_unchanged();
  PairResult result;
  ASSERT_NO_FATAL_FAILURE(RunPair(model_, 0, &result));
  EXPECT_EQ(result.token, Expected().token);

  // A candidate whose replacement workspaces were already allocated is
  // discarded without touching the model.
  loom_serve_text_group_candidate_t candidate;
  IREE_ASSERT_OK(loom_serve_text_prefill_group_prepare_variant(
      model_, variant.get(), &candidate));
  EXPECT_NE(candidate.plan, nullptr);
  EXPECT_NE(candidate.workspaces[0], nullptr);
  EXPECT_NE(candidate.workspaces[1], nullptr);
  loom_serve_text_prefill_group_discard(&candidate);
  expect_unchanged();
  PairResult after;
  ASSERT_NO_FATAL_FAILURE(RunPair(model_, 1, &after));
  EXPECT_EQ(after, result);

  // A candidate of another model is refused before any preparation.
  auto options = ModelOptions();
  loom_serve_text_model_t* other = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &other));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_text_model_select_variant(other, variant.get()));
  EXPECT_EQ(other->prefill_group.stages, nullptr);
  IREE_ASSERT_OK(loom_serve_text_model_free(other));
}

TEST_F(GroupTextServiceTest, SingleModelVariantSelectionIsUnchanged) {
  auto options = ModelOptions();
  options.group_mode = LOOM_SERVE_GROUP_MODE_SINGLE;
  loom_serve_text_model_t* single_model = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &single_model));
  const auto release_model = [](loom_serve_text_model_t* model) {
    IREE_EXPECT_OK(loom_serve_text_model_free(model));
  };
  std::unique_ptr<loom_serve_text_model_t, decltype(release_model)> single(
      single_model, release_model);
  std::string path;
  ASSERT_NO_FATAL_FAILURE(WorkspaceProvider(65536, &path));
  Variant variant(nullptr, loom_serve_text_variant_free);
  ASSERT_NO_FATAL_FAILURE(CompileVariant(single_model, path, &variant));
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  auto* row = loom_serve_text_model_row(single_model, 0);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 3, tokens));
  IREE_ASSERT_OK(
      loom_serve_text_model_select_variant(single_model, variant.get()));
  EXPECT_EQ(single_model->execution_plan, nullptr);
  EXPECT_EQ(single_model->prefill_group.stages, nullptr);
  EXPECT_EQ(single_model->banks[0].roots[LOOM_SERVE_TEXT_BANK_WORKSPACE],
            nullptr);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 2, tokens + 3));
  EXPECT_EQ(loom_serve_text_row_token(row), Expected().token);
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(single_model, nullptr));
  Oracle oracle;
  for (int32_t token : tokens) {
    oracle.Consume(token);
  }
  oracle.Consume(oracle.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_decode(row));
  EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
}

class GroupReconfigurationTest : public GroupTextServiceTest {
 protected:
  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = GroupTextServiceTest::ServiceOptions();
    options.variant_capacity = 4;
    return options;
  }

  static uint64_t Field(const std::string& response, const std::string& key) {
    const auto at = response.find("\"" + key + "\":");
    return at == std::string::npos
               ? 0
               : std::stoull(response.substr(at + key.size() + 3));
  }
};

TEST_F(GroupReconfigurationTest, ServiceSelectsGroupVariantAtDrainedBoundary) {
  Start(3);
  const uint64_t original =
      Field(Read(Send("", "X-Loom-Model: fixture-text\r\n",
                      "/v1/models/variants", "GET")),
            "selected");
  ASSERT_NE(original, 0u);
  std::string path;
  ASSERT_NO_FATAL_FAILURE(WorkspaceProvider(65536, &path));
  const auto staged = Read(
      Send("{\"model\":\"fixture-text\",\"base\":" + std::to_string(original) +
               ",\"sources\":[\"" + std::filesystem::absolute(path).string() +
               "\"],\"wait\":true}",
           "", "/v1/models/variants"));
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  EXPECT_EQ(Field(staged, "selected"), original);
  const uint64_t candidate = Field(staged, "id");
  ASSERT_NE(candidate, original);
  // A grouped request is in flight while selection is requested; selection
  // waits for a drained boundary and the request completes exactly.
  const std::string prompt = "abxababxababxa";
  const int pending = Send(Request(Message("user", prompt), 8));
  const auto selected = Read(Send(
      "{\"model\":\"fixture-text\",\"variant\":" + std::to_string(candidate) +
          ",\"expected\":" + std::to_string(original) + "}",
      "", "/v1/models/variants/select"));
  ASSERT_EQ(selected.find("HTTP/1.1 200"), 0u) << selected;
  EXPECT_EQ(Field(selected, "selected"), candidate);
  EXPECT_EQ(Content(Read(pending)), Oracle::Complete(prompt, 8));
  EXPECT_EQ(Content(Read(Send(Request(Message("user", prompt), 5)))),
            Oracle::Complete(prompt, 5));
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  EXPECT_NE(model_->prefill_group.stages, nullptr);
  EXPECT_GE(WorkspaceLength(model_, 1), 65536u);
}

TEST_F(GroupTextServiceTest, ServiceGroupsLongInputAndHandlesOddTail) {
  Start(3);
  const std::string prompt = "abxababxababxa";
  const auto response = Read(Send(Request(Message("user", prompt), 5)));
  EXPECT_EQ(Content(response), Oracle::Complete(prompt, 5));
}
}  // namespace
