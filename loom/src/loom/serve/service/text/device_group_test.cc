// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "loom/runtime/execution/execution_testing.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace loom_serve::testing::text {

class GroupTextServiceTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
  iree_host_size_t DeviceCount() const override { return 2; }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_text_model_free(original_));
    IREE_EXPECT_OK(loom_serve_text_model_free(cancelled_));
    SourceTextServiceTest::TearDown();
  }

  // Drives the actual owner state machine, including draining every accepted
  // callback after cancellation. No scheduler worker blocks on this join.
  iree_status_t Prepare(size_t index, bool cancel,
                        loom_runtime_compilation_report_t* report,
                        loom_serve_text_model_t** out_model) {
    const auto options = ModelOptions();
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    loom_serve_text_load_t* load = nullptr;
    auto status = loom_serve_text_load_allocate(
        loom_runtime_device_group_at(group_, index),
        loom_runtime_device_group_pool_at(group_, index), &options,
        [](void* payload) {
          iree_notification_post(static_cast<iree_notification_t*>(payload),
                                 IREE_ALL_WAITERS);
        },
        &notification, host_allocator_, &load);
    if (iree_status_is_ok(status)) {
      bool complete = false;
      loom_serve_text_load_flags_t flags = 0;
      while (!complete) {
        const auto ticket = iree_notification_prepare_wait(&notification);
        status = loom_serve_text_load_advance(load, flags, &complete);
        if (complete) {
          iree_notification_cancel_wait(&notification);
        } else {
          IREE_CHECK_OK(status);
          iree_notification_commit_wait(&notification, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
        if (cancel) {
          flags = LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL;
        }
      }
      *report = *loom_serve_text_load_report(load);
      *out_model = loom_serve_text_load_take_model(load);
      loom_serve_text_load_free(load);
    }
    iree_notification_deinitialize(&notification);
    return status;
  }

  // Original compiling model, distinct from the HTTP service's device owner.
  loom_serve_text_model_t* original_ = nullptr;
  // Partial ownership returned by the cancelled asynchronous preparation.
  loom_serve_text_model_t* cancelled_ = nullptr;
};

TEST_F(GroupTextServiceTest,
       SharedNativeArtifactsSurviveOriginalAndCancelledPeerModels) {
  ASSERT_EQ(loom_runtime_device_group_count(group_), 2u);
  EXPECT_NE(
      loom_runtime_device_handle(loom_runtime_device_group_at(group_, 0)),
      loom_runtime_device_handle(loom_runtime_device_group_at(group_, 1)));
  loom_runtime_compilation_report_t first = {}, second = {}, cancelled = {};
  IREE_ASSERT_OK(Prepare(0, false, &first, &original_));
  ASSERT_GT(first.kernels, 0u);
  IREE_ASSERT_OK(Prepare(1, false, &second, &model_));
  EXPECT_EQ(second.kernels, 0u);
  EXPECT_GT(second.reused_kernels, 0u);
  EXPECT_EQ(second.entries, first.entries);
  IREE_ASSERT_OK(loom_serve_text_model_free(std::exchange(original_, nullptr)));

  // The second device's independently loaded code serves the exact source
  // oracle after the original compiler and parameter owners have retired.
  Start();
  const std::string prompt = "abxa";
  const int client = Send(Request(Message("user", prompt), 4));
  const auto before = loom_runtime_pool_statistics(storage_owner_);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_CANCELLED,
                        Prepare(0, true, &cancelled, &cancelled_));
  ASSERT_NE(cancelled_, nullptr);
  EXPECT_TRUE(loom_serve_text_model_is_retired(cancelled_));
  IREE_ASSERT_OK(
      loom_serve_text_model_free_retired(std::exchange(cancelled_, nullptr)));
  EXPECT_EQ(Content(Read(client)), Oracle::Complete(prompt, 4));
  const auto after = loom_runtime_pool_statistics(storage_owner_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "bax"), 4)))),
            Oracle::Complete("bax", 4));
}

TEST_F(GroupTextServiceTest,
       PlacedStagesRunOnBothFamiliesWithJoinedPublication) {
  const loom_runtime_pool_options_t pool_options = {};
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group_, {IREE_SV("placed-text"), 3}, &pool_options, &storage_owner_));
  const loom_model_device_placement_t placement[] = {
      {IREE_SV("advance"), IREE_SV("device1")},
      {IREE_SV("group_residual"), IREE_SV("device1")},
      {IREE_SV("*"), IREE_SV("device0")}};
  auto options = ModelOptions();
  options.group_mode = LOOM_RUNTIME_GROUP_MODE_TWO_BANK;
  options.device_placement = {IREE_ARRAYSIZE(placement), placement};
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &model_));
  EXPECT_EQ(model_->placement.count, 2u);
  const auto* foreign_family =
      iree_hal_queue_family(loom_runtime_device_dispatch_queue(
          loom_runtime_device_group_at(group_, 1)));
  size_t foreign_stages = 0;
  for (size_t i = 0; i < model_->stage_count; ++i) {
    if (!model_->stages[i].execution) {
      continue;
    }
    ++foreign_stages;
    EXPECT_EQ(loom_runtime_jit_stage_queue_family(model_->stages[i].compiled),
              foreign_family);
    EXPECT_EQ(iree_hal_command_buffer_queue_family(model_->stages[i].command),
              foreign_family);
  }
  EXPECT_EQ(foreign_stages, 2u);
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_runtime_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            0u);
  IREE_ASSERT_OK(
      loom_runtime_execution_wait(model_->execution, frontier.values[0]));
  loom_serve_text_result_t result;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &result));
  Oracle oracle;
  for (int32_t token : tokens) {
    oracle.Consume(token);
  }
  EXPECT_EQ(result.known_count, 5u);
  EXPECT_EQ(result.tokens[0], oracle.Predict());
  IREE_ASSERT_OK(
      loom_serve_text_row_decode(loom_serve_text_model_row(model_, 0)));
  oracle.Consume(oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_token(loom_serve_text_model_row(model_, 0)),
            oracle.Predict());
  Start();
  const std::string prompt = "abxaabxbabxa";
  EXPECT_EQ(Content(Read(Send(Request(Message("user", prompt), 4)))),
            Oracle::Complete(prompt, 4));
}

TEST_F(GroupTextServiceTest, PlacementRejectsUnknownRootsAndUnsharedPool) {
  const loom_model_device_placement_t invalid[] = {
      {IREE_SV("not_a_stage"), IREE_SV("device0")},
      {IREE_SV("advance"), IREE_SV("device1")},
      {IREE_SV("advance"), IREE_SV("missing_device")}};
  for (const auto& placement : invalid) {
    auto options = ModelOptions();
    options.device_placement = {1, &placement};
    const auto status = loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model_);
    EXPECT_FALSE(iree_status_is_ok(status));
    iree_status_ignore(status);
    EXPECT_EQ(model_, nullptr);
  }
}

TEST_F(GroupTextServiceTest,
       ForeignStageRejectionRetainsAcceptedGroupForRetirement) {
  const loom_runtime_pool_options_t pool_options = {};
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group_, {IREE_SV("placed-failure"), 3}, &pool_options, &storage_owner_));
  const loom_model_device_placement_t placement = {IREE_SV("group_residual"),
                                                   IREE_SV("device1")};
  auto options = ModelOptions();
  options.group_mode = LOOM_RUNTIME_GROUP_MODE_TWO_BANK;
  options.device_placement = {1, &placement};
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &model_));
  loom_runtime_execution_t* peer = nullptr;
  for (size_t i = 0; i < model_->stage_count; ++i) {
    if (model_->stages[i].execution) {
      peer = model_->stages[i].execution;
    }
  }
  ASSERT_NE(peer, nullptr);
  // Four uploads and two prefixes precede the first foreign residual.
  IREE_ASSERT_OK(loom_runtime_execution_reject_group_node_for_testing(peer, 6));
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_runtime_execution_frontier_t frontier;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  EXPECT_NE(model_->prefill_group.pending, nullptr);
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            0u);
  IREE_ASSERT_OK(loom_serve_text_model_free(std::exchange(model_, nullptr)));
}

TEST_F(GroupTextServiceTest, ModelCardPlacementFeedsTheNormalLoader) {
  const loom_runtime_pool_options_t pool_options = {};
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group_, {IREE_SV("card-placement"), 3}, &pool_options, &storage_owner_));
  std::ifstream input(directory_ + "/native.card.json");
  ASSERT_TRUE(input.good());
  std::string json((std::istreambuf_iterator<char>(input)), {});
  const auto version = json.find("\"schema_version\": 1");
  ASSERT_NE(version, std::string::npos);
  json.replace(version, strlen("\"schema_version\": 1"),
               "\"schema_version\": 2");
  const auto program = json.find("\"inputs\":");
  ASSERT_NE(program, std::string::npos);
  json.insert(program,
              R"("device_placement":{"advance":"device1","*":"device0"},)");
  const auto weights = json.find("\"fixture.card\": true");
  ASSERT_NE(weights, std::string::npos);
  json.replace(weights, strlen("\"fixture.card\": true"),
               "\"fixture.card\": false");
  loom_model_card_t* raw_card = nullptr;
  const loom_model_selector_t selector = {.preset = IREE_SV("check")};
  IREE_ASSERT_OK(
      loom_model_card_parse(iree_make_string_view(json.data(), json.size()),
                            &selector, host_allocator_, &raw_card));
  std::unique_ptr<loom_model_card_t, decltype(&loom_model_card_release)> card(
      raw_card, loom_model_card_release);
  auto options = ModelOptions();
  options.sources.card = card.get();
  options.sources.count = 0;
  options.sources.values = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &model_));
  EXPECT_EQ(model_->placement.count, 2u);
  loom_runtime_execution_t* foreign = nullptr;
  for (size_t i = 0; i < model_->stage_count; ++i) {
    if (model_->stages[i].execution) {
      foreign = model_->stages[i].execution;
    }
  }
  ASSERT_NE(foreign, nullptr);
  const auto before = loom_runtime_execution_frontier(foreign);
  const int32_t tokens[] = {0, 1, 2};
  IREE_ASSERT_OK(loom_serve_text_row_prefill(
      loom_serve_text_model_row(model_, 0), 3, tokens));
  EXPECT_GT(loom_runtime_execution_frontier(foreign).values[0],
            before.values[0]);
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            3u);
}

class FfnGroupTextServiceTest : public GroupTextServiceTest {
 protected:
  void PrepareFfn(bool draft = false, bool resident = false) {
    prepare_ = iree::testing::MakeTempFilePath("ffn_prepare", ".loom");
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/prepare.loom", prepare_,
        "config.def @fixture.ffn = false", "config.def @fixture.ffn = true"));
    if (draft) {
      ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
          prepare_, prepare_, "config.def @fixture.draft = false",
          "config.def @fixture.draft = true"));
    }
    if (resident) {
      ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
          prepare_, prepare_, "config.def @fixture.resident = false",
          "config.def @fixture.resident = true"));
    }
    const loom_runtime_pool_options_t pool_options = {
        .uncached = resident, .uncached_device_mask = resident ? 1u : 0u};
    const auto name = iree_make_cstring_view(
        ::testing::UnitTest::GetInstance()->current_test_info()->name());
    IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
        group_, {name, 3}, &pool_options, &storage_owner_));
    auto options = FfnOptions(draft);
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model_));
  }

  loom_serve_text_options_t FfnOptions(bool draft) const {
    auto options = ModelOptions();
    options.sources.prepare = iree_make_cstring_view(prepare_.c_str());
    options.group_mode = LOOM_RUNTIME_GROUP_MODE_TWO_BANK;
    options.device_placement = {IREE_ARRAYSIZE(placement_), placement_};
    options.draft_depth = draft ? 3 : 0;
    return options;
  }

  static std::vector<int32_t> DraftCache(loom_serve_text_model_t* model) {
    auto* buffer = model->state.draft.cache;
    std::vector<int32_t> words(iree_hal_buffer_byte_length(buffer) / 4);
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = buffer,
                     .target = words.data(),
                     .length = words.size() * 4}};
    uint64_t value = 0;
    IREE_EXPECT_OK(loom_runtime_execution_transfer(model->execution, 1,
                                                   &download, &value));
    IREE_EXPECT_OK(loom_runtime_execution_wait(model->execution, value));
    return words;
  }

  const loom_model_device_placement_t placement_[2] = {
      {IREE_SV("group_ffn_secondary"), IREE_SV("device1")},
      {IREE_SV("group_ffn_down_secondary"), IREE_SV("device1")}};
  std::string prepare_;
};

TEST_F(FfnGroupTextServiceTest,
       FfnForkJoinUsesPrivateLaneWorkspacesAndMergedOutput) {
  ASSERT_NO_FATAL_FAILURE(PrepareFfn());
  EXPECT_EQ(model_->placement.count, 2u);
  EXPECT_GE(model_->prefill_group.workspace.allocation_size, 512u);
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_runtime_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            0u);
  IREE_ASSERT_OK(
      loom_runtime_execution_wait(model_->execution, frontier.values[0]));
  loom_serve_text_result_t result;
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &result));
  Oracle oracle;
  for (int32_t token : tokens) {
    oracle.Consume(token);
  }
  EXPECT_EQ(result.known_count, 5u);
  EXPECT_EQ(result.tokens[0], oracle.Predict());
  IREE_ASSERT_OK(
      loom_serve_text_row_decode(loom_serve_text_model_row(model_, 0)));
  oracle.Consume(oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_token(loom_serve_text_model_row(model_, 0)),
            oracle.Predict());
  Start();
  const std::string prompt = "abxaabxbabxa";
  EXPECT_EQ(Content(Read(Send(Request(Message("user", prompt), 4)))),
            Oracle::Complete(prompt, 4));
}

TEST_F(FfnGroupTextServiceTest,
       ResidentTextDescriptorConsumesAcksAndReusesBanks) {
  ASSERT_NO_FATAL_FAILURE(PrepareFfn(false, true));
  ASSERT_EQ(model_->prefill_group.transport_count, 2u);
  Oracle oracle;
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  for (int invocation = 0; invocation < 2; ++invocation) {
    const loom_serve_text_span_t a = {
        .row_index = 0, .token_count = 3, .token_ids = tokens};
    const loom_serve_text_span_t b = {
        .row_index = 0,
        .token_count = 2,
        .token_ids = tokens + 3,
        .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
        .input_offset = 3};
    loom_runtime_execution_frontier_t frontier;
    IREE_ASSERT_OK(loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b,
                                                           &frontier));
    IREE_ASSERT_OK(
        loom_runtime_execution_wait(model_->execution, frontier.values[0]));
    for (size_t bank = 0; bank < 2; ++bank) {
      iree_hal_buffer_mapping_t mapping = {};
      IREE_ASSERT_OK(iree_hal_buffer_map_range(
          model_->banks[bank].roots[0], IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_READ, 0, 512, 2624, &mapping));
      EXPECT_EQ(iree_unaligned_load_le_u32(mapping.contents.data), 1u);
      EXPECT_EQ(iree_unaligned_load_le_u32(mapping.contents.data + 64), 1u);
      EXPECT_EQ(iree_unaligned_load_le_u32(mapping.contents.data + 1408 + 64),
                1u);
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    }
    loom_serve_text_result_t result;
    IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, &result));
    for (int32_t token : tokens) {
      oracle.Consume(token);
    }
    EXPECT_EQ(result.known_count, 5u);
    EXPECT_EQ(result.tokens[0], oracle.Predict());
    EXPECT_EQ(model_->prefill_group.exchange, nullptr);
  }
}

TEST_F(FfnGroupTextServiceTest,
       ResidentTextRejectsCachedConsumerEvenWhenProducerHasLiveAccess) {
  ASSERT_NO_FATAL_FAILURE(PrepareFfn(false, true));
  const loom_runtime_pool_options_t options = {.uncached = true,
                                               .uncached_device_mask = 2};
  loom_runtime_pool_t* pool = nullptr;
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group_, {IREE_SV("cached_consumer"), 3}, &options, &pool));
  auto model_options = FfnOptions(false);
  loom_serve_text_model_t* rejected = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_allocate(device_, pool, &model_options,
                                     host_allocator_, &rejected));
  EXPECT_EQ(rejected, nullptr);
}

TEST_F(FfnGroupTextServiceTest,
       RejectedResidentConsumerAbortsWithoutPublication) {
  ASSERT_NO_FATAL_FAILURE(PrepareFfn(false, true));
  const int32_t tokens[] = {0, 1, 2, 0, 1};
  const loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  const loom_serve_text_span_t b = {.row_index = 0,
                                    .token_count = 2,
                                    .token_ids = tokens + 3,
                                    .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                                    .input_offset = 3};
  IREE_ASSERT_OK(loom_runtime_execution_reject_group_node_for_testing(
      model_->execution, 9));
  loom_runtime_execution_frontier_t frontier;
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            0u);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      model_->banks[0].roots[0], IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ, 0, 512, 192, &mapping));
  EXPECT_EQ(iree_unaligned_load_le_u32(mapping.contents.data + 64), 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(mapping.contents.data + 128),
            UINT32_MAX);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_serve_text_model_free(std::exchange(model_, nullptr)));
}

TEST_F(FfnGroupTextServiceTest, RejectedSecondaryRetiresWithoutPublication) {
  ASSERT_NO_FATAL_FAILURE(PrepareFfn());
  loom_runtime_execution_t* peer = nullptr;
  for (size_t i = 0; i < model_->stage_count; ++i) {
    if (model_->stages[i].execution) {
      peer = model_->stages[i].execution;
    }
  }
  ASSERT_NE(peer, nullptr);
  // Four uploads, bank A prefix and its primary branch precede this node.
  IREE_ASSERT_OK(loom_runtime_execution_reject_group_node_for_testing(peer, 6));
  const int32_t tokens[] = {0, 1, 2, 2, 0};
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .input_offset = 3};
  loom_runtime_execution_frontier_t frontier;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  EXPECT_NE(model_->prefill_group.pending, nullptr);
  EXPECT_EQ(loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)),
            0u);
  IREE_ASSERT_OK(loom_serve_text_model_free(std::exchange(model_, nullptr)));
}

TEST_F(FfnGroupTextServiceTest, DraftCacheMatchesSingleAfterBothMerges) {
  ASSERT_NO_FATAL_FAILURE(PrepareFfn(true));
  auto options = FfnOptions(true);
  options.group_mode = LOOM_RUNTIME_GROUP_MODE_SINGLE;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &original_));
  const int32_t tokens[] = {0, 1, 2, 2, 0};
  auto* baseline = loom_serve_text_model_row(original_, 0);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(baseline, 3, tokens));
  IREE_ASSERT_OK(loom_serve_text_row_prefill(baseline, 2, tokens + 3));
  loom_serve_text_span_t a = {
      .row_index = 0, .token_count = 3, .token_ids = tokens};
  loom_serve_text_span_t b = {.row_index = 0,
                              .token_count = 2,
                              .token_ids = tokens + 3,
                              .flags = LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT,
                              .input_offset = 3};
  loom_runtime_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_epoch_pair(model_, 2, &a, 1, &b, &frontier));
  IREE_ASSERT_OK(
      loom_runtime_execution_wait(model_->execution, frontier.values[0]));
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, nullptr));
  EXPECT_EQ(DraftCache(model_), DraftCache(original_));
  auto* grouped = loom_serve_text_model_row(model_, 0);
  EXPECT_EQ(loom_serve_text_row_token(grouped),
            loom_serve_text_row_token(baseline));
  IREE_ASSERT_OK(loom_serve_text_row_decode(grouped));
  IREE_ASSERT_OK(loom_serve_text_row_decode(baseline));
  EXPECT_EQ(DraftCache(model_), DraftCache(original_));
  EXPECT_EQ(loom_serve_text_row_token(grouped),
            loom_serve_text_row_token(baseline));
}

TEST_F(GroupTextServiceTest,
       ResidentExchangeRequiresEveryExactGenerationAndResets) {
  alignas(64) uint32_t words[304] = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      sizeof(words), iree_make_byte_span(words, sizeof(words)), {},
      host_allocator_, &buffer));
  for (int generation = 0; generation < 2; ++generation) {
    loom_serve_text_group_exchange_t* exchange = nullptr;
    IREE_ASSERT_OK(
        loom_serve_text_group_exchange_create(2, host_allocator_, &exchange));
    iree_hal_semaphore_t* ready = nullptr;
    IREE_ASSERT_OK(loom_serve_text_group_exchange_add(
        exchange, loom_runtime_device_handle(device_), buffer, 0, 192, &ready));
    iree_hal_semaphore_t* alias = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_text_group_exchange_add(
                              exchange, loom_runtime_device_handle(device_),
                              buffer, 0, 192, &alias));
    EXPECT_EQ(alias, nullptr);
    EXPECT_EQ(words[0], 1u);
    EXPECT_EQ(words[16], 0u);
    EXPECT_EQ(words[32], 0u);
    for (size_t i = 0; i < 16; ++i) {
      EXPECT_EQ(words[48 + i * 16], 0u);
    }
    IREE_ASSERT_OK(
        loom_serve_text_group_exchange_start(exchange, INT64_C(5000000000)));
    EXPECT_EQ(words[32], 1u);
    auto* response = reinterpret_cast<iree_atomic_int32_t*>(words + 48);
    for (size_t i = 0; i < 15; ++i) {
      iree_atomic_store(&response[i * 16], 1, iree_memory_order_release);
    }
    iree_atomic_store(&response[15 * 16], 2, iree_memory_order_release);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DEADLINE_EXCEEDED,
        iree_hal_semaphore_wait(ready, 1, iree_make_timeout_ms(2), 0));
    iree_atomic_store(&response[15 * 16], 1, iree_memory_order_release);
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(ready, 1, iree_make_timeout_ms(5000), 0));
    EXPECT_EQ(words[16], 0u);
    loom_serve_text_group_exchange_abort(exchange);
    EXPECT_EQ(words[16], 1u);
    EXPECT_EQ(words[32], UINT32_MAX);
    loom_serve_text_group_exchange_free(exchange);
  }
  iree_hal_buffer_release(buffer);
}

TEST_F(GroupTextServiceTest,
       ContinuousExchangeDefersGenerationAndAbortsTheWholeLifetime) {
  alignas(64) uint32_t words[304] = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      sizeof(words), iree_make_byte_span(words, sizeof(words)), {},
      host_allocator_, &buffer));
  for (int invocation = 0; invocation < 2; ++invocation) {
    loom_serve_text_group_exchange_t* exchange = nullptr;
    IREE_ASSERT_OK(
        loom_serve_text_group_exchange_create(1, host_allocator_, &exchange));
    IREE_ASSERT_OK(loom_serve_text_group_exchange_add_resident(
        exchange, loom_runtime_device_handle(device_), buffer, 0, 192, 256));
    EXPECT_EQ(words[0], 0u);
    EXPECT_EQ(words[16], 0u);
    EXPECT_EQ(words[32], 0u);
    IREE_ASSERT_OK(
        loom_serve_text_group_exchange_start(exchange, INT64_C(5000000000)));
    EXPECT_EQ(words[32], 1u);
    EXPECT_EQ(words[0], 0u);
    if (invocation == 1) {
      iree_atomic_store(reinterpret_cast<iree_atomic_int32_t*>(words + 16), 256,
                        iree_memory_order_release);
      IREE_ASSERT_OK(loom_serve_text_group_exchange_check(exchange));
      loom_serve_text_group_exchange_free(exchange);
      EXPECT_EQ(words[0], 0u);
      EXPECT_EQ(words[16], 256u);
      EXPECT_EQ(words[32], 1u);
      continue;
    }
    loom_serve_text_group_exchange_abort(exchange);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                          loom_serve_text_group_exchange_check(exchange));
    EXPECT_EQ(words[0], uint32_t(INT32_MAX));
    EXPECT_EQ(words[16], 256u);
    EXPECT_EQ(words[32], UINT32_MAX);
    loom_serve_text_group_exchange_free(exchange);
  }
  iree_hal_buffer_release(buffer);
}

TEST_F(GroupTextServiceTest,
       ResidentExchangeTimeoutAbortsAndUnblocksReadiness) {
  alignas(64) uint32_t words[304] = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      sizeof(words), iree_make_byte_span(words, sizeof(words)), {},
      host_allocator_, &buffer));
  loom_serve_text_group_exchange_t* exchange = nullptr;
  IREE_ASSERT_OK(
      loom_serve_text_group_exchange_create(1, host_allocator_, &exchange));
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(loom_serve_text_group_exchange_add(
      exchange, loom_runtime_device_handle(device_), buffer, 0, 192, &ready));
  IREE_ASSERT_OK(
      loom_serve_text_group_exchange_start(exchange, INT64_C(1000000)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_wait(ready, 1, iree_make_timeout_ms(5000), 0));
  loom_serve_text_group_exchange_abort(exchange);
  EXPECT_EQ(words[0], 1u);
  EXPECT_EQ(words[16], 1u);
  EXPECT_EQ(words[32], UINT32_MAX);
  loom_serve_text_group_exchange_free(exchange);
  iree_hal_buffer_release(buffer);
}

}  // namespace loom_serve::testing::text
