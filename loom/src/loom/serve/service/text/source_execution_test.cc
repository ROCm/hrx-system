// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_F(SourceTextServiceTest, PreparationActivatesWeightsOnlyWithDemand) {
  for (const auto flags : {0u, uint32_t{LOOM_SERVE_TEXT_LOAD_FLAG_DEMAND}}) {
    SCOPED_TRACE(flags);
    auto* pool = loom_serve_device_memory_pool(device_);
    if (pool) {
      IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    }
    const auto before = loom_serve_device_memory_statistics(device_);
    const auto options = ModelOptions();
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    loom_serve_text_load_t* load = nullptr;
    IREE_ASSERT_OK(loom_serve_text_load_allocate(
        device_, &options,
        [](void* payload) {
          iree_notification_post(static_cast<iree_notification_t*>(payload),
                                 IREE_ALL_WAITERS);
        },
        &notification, host_allocator_, &load));
    bool complete = false;
    while (!complete) {
      const auto ticket = iree_notification_prepare_wait(&notification);
      IREE_ASSERT_OK(loom_serve_text_load_advance(load, flags, &complete));
      if (complete) {
        iree_notification_cancel_wait(&notification);
      } else {
        iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    auto* model = loom_serve_text_load_take_model(load);
    loom_serve_text_load_free(load);
    const auto parameters =
        loom_serve_residency_snapshot(loom_serve_text_model_residency(model));
    // Warmup owns an independent invocation. Preparation does not import its
    // weight-ready dependency into the mutable-state execution timeline.
    EXPECT_FALSE(parameters.active);
    EXPECT_EQ(parameters.pin_count, 0u);
    EXPECT_EQ(parameters.activity.activations, flags ? 1u : 0u);
    auto* row = loom_serve_text_model_row(model, 0);
    const int32_t token = Oracle::Id('a');
    IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
    Oracle reference;
    reference.Consume("a");
    EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
    EXPECT_EQ(loom_serve_text_model_weight_activity(model).activations, 1u);
    loom_serve_text_warmup_poll(loom_serve_text_model_warmup(model));
    IREE_ASSERT_OK(
        loom_serve_text_warmup_status(loom_serve_text_model_warmup(model)));
    IREE_ASSERT_OK(loom_serve_text_model_free(model));
    iree_notification_deinitialize(&notification);
    // Model retirement releases logical ownership. Native disposal may finish
    // later and remains charged to the shared pool until its outer join.
    if (pool) {
      IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    }
    const auto after = loom_serve_device_memory_statistics(device_);
    EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
    EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  }
}

TEST_F(SourceTextServiceTest, FailedLinkRetiresAcceptedInitialization) {
  const auto package = iree::testing::MakeTempFilePath("failed_text_link");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file : {"prepare.loom", "advance.loom", "model.json",
                           "weights.loom", "tokenizer.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    std::ifstream input(directory_ + "/control.loom");
    std::string source((std::istreambuf_iterator<char>(input)), {});
    const auto offset = source.find("import(\"runner\")");
    ASSERT_NE(offset, std::string::npos);
    source.replace(offset, std::string("import(\"runner\")").size(),
                   "import(\"missing\")");
    std::ofstream output(package + "/control.loom");
    output << source;
    ASSERT_TRUE(output.good());
  }
  auto* pool = loom_serve_device_memory_pool(device_);
  if (pool) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  const auto before = loom_serve_device_memory_statistics(device_);
  auto options = ModelOptions();
  options.sources.base_directory = iree_make_cstring_view(package.c_str());
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  const auto wake = [](void* payload) {
    iree_notification_post(static_cast<iree_notification_t*>(payload),
                           IREE_ALL_WAITERS);
  };
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_allocate(
      device_, &options, wake, &notification, host_allocator_, &load));
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_text_load_advance(
        load, LOOM_SERVE_TEXT_LOAD_FLAG_DEMAND, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      IREE_ASSERT_OK(status);
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  EXPECT_NE(
      iree::Status::ToString(status).find("requires missing module 'missing'"),
      std::string::npos);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  auto* partial = loom_serve_text_load_take_model(load);
  EXPECT_EQ(loom_serve_text_model_weight_activity(partial).activations, 1u);
  loom_serve_text_load_free(load);
  loom_serve_text_model_retire(partial, wake, &notification);
  EXPECT_TRUE(iree_notification_await(
      &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      partial, iree_infinite_timeout()));
  IREE_ASSERT_OK(loom_serve_text_model_free_retired(partial));
  iree_notification_deinitialize(&notification);
  if (pool) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  auto* row = loom_serve_text_model_row(model_, 0);
  const int32_t token = Oracle::Id('a');
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  Oracle reference;
  reference.Consume("a");
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
}

TEST_F(SourceTextServiceTest, MissingCheckpointFailsBeforeParameterPlacement) {
  auto* pool = loom_serve_device_memory_pool(device_);
  if (pool) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  const auto before = loom_serve_device_memory_statistics(device_);
  auto options = ModelOptions();
  const auto missing = iree::testing::MakeTempFilePath("missing_checkpoint");
  ASSERT_FALSE(std::filesystem::exists(missing));
  options.weights_path = iree_make_cstring_view(missing.c_str());
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_allocate(
      device_, &options,
      [](void* payload) {
        iree_notification_post(static_cast<iree_notification_t*>(payload),
                               IREE_ALL_WAITERS);
      },
      &notification, host_allocator_, &load));
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_text_load_advance(
        load, LOOM_SERVE_TEXT_LOAD_FLAG_DEMAND, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      IREE_ASSERT_OK(status);
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  auto* partial = loom_serve_text_load_take_model(load);
  const auto parameters = loom_serve_text_model_weight_statistics(partial);
  EXPECT_EQ(parameters.reserved_bytes, 0u);
  EXPECT_EQ(parameters.committed_bytes, 0u);
  EXPECT_EQ(loom_serve_text_model_weight_activity(partial).activations, 0u);
  loom_serve_text_load_free(load);
  EXPECT_TRUE(loom_serve_text_model_is_retired(partial));
  IREE_ASSERT_OK(loom_serve_text_model_free_retired(partial));
  iree_notification_deinitialize(&notification);
  if (pool) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  // The failed preparation did not retire shared queues or existing models.
  auto* row = loom_serve_text_model_row(model_, 0);
  const int32_t token = Oracle::Id('a');
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  Oracle reference;
  reference.Consume("a");
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
}

class PooledTextServiceTest
    : public SourceTextServiceTest,
      public ::testing::WithParamInterface<iree_host_size_t> {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.pool_capacity = GetParam();
    return options;
  }
};

INSTANTIATE_TEST_SUITE_P(PoolCapacity, PooledTextServiceTest,
                         ::testing::Values(iree_host_size_t{256},
                                           iree_host_size_t{4194308},
                                           iree_host_size_t{32 * 1024 * 1024}));

TEST_P(PooledTextServiceTest, ChunkingRowsAndForksMatchDenseOracle) {
  const std::string prompt = "abxabbaaxbabx";
  std::vector<int32_t> tokens;
  for (char character : prompt) {
    tokens.push_back(Oracle::Id(character));
  }
  Oracle reference;
  reference.Consume(prompt);
  auto* full = loom_serve_text_model_row(model_, 0);
  auto* chunks = loom_serve_text_model_row(model_, 1);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(full, tokens.size(), tokens.data()));
  for (size_t offset = 0; offset < tokens.size(); offset += 3) {
    IREE_ASSERT_OK(loom_serve_text_row_prefill(
        chunks, std::min<size_t>(3, tokens.size() - offset),
        tokens.data() + offset));
  }
  EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(chunks), reference.Predict());
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(full, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  auto* fork = loom_serve_text_model_row(model_, 2);
  bool restored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  const auto before = loom_serve_text_model_pool_usage(model_);
  const int32_t a = 0;
  const int32_t b = 1;
  const loom_serve_text_span_t spans[] = {
      {0, 1, &a, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {2, 1, &b, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  IREE_ASSERT_OK(
      loom_serve_text_model_epoch(model_, 1, IREE_ARRAYSIZE(spans), spans));
  Oracle branch = reference;
  reference.Consume(a);
  branch.Consume(b);
  EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(fork), branch.Predict());
  // Both writers detach the pinned partial page; the immutable endpoint stays.
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available + 8,
            before.available);
  IREE_ASSERT_OK(
      loom_serve_text_row_try_restore(chunks, checkpoint, &restored));
  ASSERT_TRUE(restored);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(chunks, 1, &b));
  EXPECT_EQ(loom_serve_text_row_token(chunks), branch.Predict());
  loom_serve_text_checkpoint_release(checkpoint);
  for (size_t i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        loom_serve_text_row_reset(loom_serve_text_model_row(model_, i)));
  }
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, GetParam());
}

TEST_P(PooledTextServiceTest, KnownInputCheckpointsPreserveNoPrediction) {
  const std::string prompt = "abxabbaaxba";
  std::vector<int32_t> tokens;
  for (char character : prompt) {
    tokens.push_back(Oracle::Id(character));
  }
  auto* row = loom_serve_text_model_row(model_, 0);
  const loom_serve_text_span_t prefix = {0, tokens.size(), tokens.data(), 0};
  IREE_ASSERT_OK(loom_serve_text_model_epoch(model_, 1, 1, &prefix));
  EXPECT_EQ(loom_serve_text_row_position(row), 11u);
  loom_serve_execution_frontier_t frontier;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_row_submit_decode(row, &frontier));

  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(row, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);

  // The next ID belongs to the caller, not a local prediction. Consuming it
  // advances retained state without making the row eligible for decoding.
  const int32_t b = Oracle::Id('b');
  const loom_serve_text_span_t append = {0, 1, &b, 0};
  IREE_ASSERT_OK(loom_serve_text_model_epoch(model_, 0, 1, &append));
  EXPECT_EQ(loom_serve_text_row_position(row), 12u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_row_submit_decode(row, &frontier));

  auto* fork = loom_serve_text_model_row(model_, 1);
  bool restored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  EXPECT_EQ(loom_serve_text_row_position(fork), 11u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_row_submit_decode(fork, &frontier));

  // An explicit later selection observes the independently evolved states.
  const int32_t x = Oracle::Id('x');
  const loom_serve_text_span_t branches[] = {
      {0, 1, &x, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {1, 1, &b, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  IREE_ASSERT_OK(loom_serve_text_model_epoch(
      model_, 1, IREE_ARRAYSIZE(branches), branches));
  Oracle reference;
  reference.Consume(prompt + "bx");
  Oracle branch;
  branch.Consume(prompt + "b");
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(fork), branch.Predict());
  EXPECT_EQ(loom_serve_text_row_position(row), 13u);
  EXPECT_EQ(loom_serve_text_row_position(fork), 12u);

  // Restoring over a selected row must clear its old prediction as well.
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(row, checkpoint, &restored));
  ASSERT_TRUE(restored);
  EXPECT_EQ(loom_serve_text_row_position(row), 11u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_row_submit_decode(row, &frontier));
  loom_serve_text_checkpoint_release(checkpoint);
  IREE_ASSERT_OK(loom_serve_text_row_reset(row));
  IREE_ASSERT_OK(loom_serve_text_row_reset(fork));
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, GetParam());
}

TEST_F(SourceTextServiceTest, RejectsPoolOutsideSourceIntegerRepresentation) {
  if (sizeof(iree_host_size_t) < sizeof(uint64_t)) {
    GTEST_SKIP() << "host token counts already fit signed i64";
  }
  auto options = ModelOptions();
  options.pool_capacity =
      static_cast<iree_host_size_t>(uint64_t{INT64_MAX} + 1);
  loom_serve_text_model_t* rejected = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &rejected));
  EXPECT_EQ(rejected, nullptr);
}

TEST_F(SourceTextServiceTest, WideSourceEpochsRetainExactHttpHistory) {
  for (size_t capacity : {513u, 1024u, 4097u}) {
    SCOPED_TRACE(capacity);
    IREE_ASSERT_OK(loom_serve_text_model_free(model_));
    model_ = nullptr;
    const loom_serve_packing_shape_t shapes[] = {{32, 4}, {capacity, 4}};
    auto options = ModelOptions();
    options.prefill_capacity = capacity;
    options.context_capacity = capacity + 64;
    options.pool_capacity = 4 * ((options.context_capacity + 3) / 4 * 4);
    options.epoch_count = IREE_ARRAYSIZE(shapes);
    options.epoch_shapes = shapes;
    IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                  host_allocator_, &model_));
    EXPECT_EQ(loom_serve_text_model_prefill_capacity(model_), capacity);
    std::vector<int32_t> tokens(capacity - 9);
    Oracle reference;
    for (size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = i % 3;
      reference.Consume(tokens[i]);
    }
    auto* full = loom_serve_text_model_row(model_, 0);
    auto* chunks = loom_serve_text_model_row(model_, 1);
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(full, tokens.size(), tokens.data()));
    for (size_t offset = 0; offset < tokens.size(); offset += 31) {
      IREE_ASSERT_OK(loom_serve_text_row_prefill(
          chunks, std::min<size_t>(31, tokens.size() - offset),
          tokens.data() + offset));
    }
    EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
    EXPECT_EQ(loom_serve_text_row_token(chunks), reference.Predict());

    // The decode row and a fresh long prefill fill the same wide epoch.
    const int32_t pending = loom_serve_text_row_token(full);
    tokens.resize(capacity - 1, 1);
    const loom_serve_text_span_t spans[] = {
        {0, 1, &pending, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
        {2, tokens.size(), tokens.data(), LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
    };
    IREE_ASSERT_OK(
        loom_serve_text_model_epoch(model_, 1, IREE_ARRAYSIZE(spans), spans));
    reference.Consume(pending);
    EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
    Oracle fresh;
    for (int32_t token : tokens) {
      fresh.Consume(token);
    }
    EXPECT_EQ(loom_serve_text_row_token(loom_serve_text_model_row(model_, 2)),
              fresh.Predict());
    for (size_t row = 0; row < 4; ++row) {
      IREE_ASSERT_OK(
          loom_serve_text_row_reset(loom_serve_text_model_row(model_, row)));
    }
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available,
              options.pool_capacity);
  }

  // Same server binary and source package, now with a 4097-token host shape.
  Start(4097);
  std::string prompt;
  for (size_t i = 0; i < 4088; ++i) {
    prompt.push_back("abx"[i % 3]);
  }
  const int retained =
      Send(Request(Message("user", prompt), 2), "X-Loom-Session: wide\r\n");
  const int other = Send(Request(Message("user", std::string(769, 'x'))));
  const std::string first = Content(Read(retained));
  ASSERT_EQ(first, Oracle::Complete(prompt, 2));
  EXPECT_EQ(Content(Read(other)), Oracle::Complete(std::string(769, 'x'), 16));
  const std::string request =
      Request(Message("user", prompt) + "," + Message("assistant", first) +
              "," + Message("user", "bax"));
  const std::string resumed = Read(Send(request, "X-Loom-Session: wide\r\n"));
  EXPECT_EQ(Content(resumed), Oracle::Complete(prompt + first + "#bax", 16));
  EXPECT_NE(resumed.find("\"cached_tokens\":4089"), std::string::npos)
      << resumed;
  EXPECT_EQ(Content(Read(Send(request))), Content(resumed));
}

TEST_F(SourceTextServiceTest,
       SourceRejectsUnsupportedConfigurationAndUsesWeights) {
  auto options = ModelOptions();
  options.draft_depth = 3;
  loom_serve_text_model_t* unsupported = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  options.draft_depth = 0;
  options.options = IREE_SV("{\"unknown\":true}");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  options.options = IREE_SV("{}");
  options.pool_capacity = 0;
  options.checkpoint_capacity = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(WriteWeights(7));
  options = ModelOptions();
  IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                host_allocator_, &model_));
  Start();
  const std::string prompt = "axb";
  const std::string actual =
      Content(Read(Send(Request(Message("user", prompt)))));
  EXPECT_EQ(actual, Oracle::Complete(prompt, 16, 7));
  EXPECT_NE(actual, Oracle::Complete(prompt, 16, 2));
}

TEST_F(SourceTextServiceTest, CheckpointMapExtentFailureRetiresTheLoad) {
  struct Allocations {
    // Oversized byte extent rejected without allocating its payload.
    std::atomic<uint64_t> rejected_bytes = 0;
    // Successful allocations still owned by the native model load.
    std::atomic<int64_t> live_count = 0;
  } allocations;
  const iree_allocator_t allocator = {
      &allocations,
      [](void* user_data, iree_allocator_command_t command, const void* params,
         void** inout_pointer) -> iree_status_t {
        auto* allocations = static_cast<Allocations*>(user_data);
        if (command != IREE_ALLOCATOR_COMMAND_FREE) {
          const auto length =
              static_cast<const iree_allocator_alloc_params_t*>(params)
                  ->byte_length;
          if (length > 64 * 1024 * 1024) {
            allocations->rejected_bytes.store(length);
            return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
          }
        }
        const bool had_pointer = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                                  command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                                 *inout_pointer != nullptr;
        const auto delegate = iree_allocator_system();
        iree_status_t status =
            delegate.ctl(delegate.self, command, params, inout_pointer);
        if (iree_status_is_ok(status)) {
          if (command == IREE_ALLOCATOR_COMMAND_FREE) {
            allocations->live_count.fetch_sub(had_pointer ? 1 : 0);
          } else if (command != IREE_ALLOCATOR_COMMAND_REALLOC ||
                     !had_pointer) {
            allocations->live_count.fetch_add(*inout_pointer ? 1 : 0);
          }
        }
        return status;
      },
  };
  auto options = ModelOptions();
  options.context_capacity = 262144;
  options.pool_capacity = 262144;
  // Four-token pages require 65536 map entries per checkpoint. Cover byte
  // extent overflow, element-count wrap to zero, and a wrapped nonzero count.
  for (const iree_host_size_t count : {16384u, 65536u, 65537u}) {
    SCOPED_TRACE(count);
    options.checkpoint_capacity = count;
    allocations.rejected_bytes.store(0);
    loom_serve_text_model_t* rejected = nullptr;
    const auto expected_status = sizeof(iree_host_size_t) == 4
                                     ? IREE_STATUS_OUT_OF_RANGE
                                     : IREE_STATUS_RESOURCE_EXHAUSTED;
    IREE_EXPECT_STATUS_IS(expected_status,
                          loom_serve_text_model_allocate(device_, &options,
                                                         allocator, &rejected));
    EXPECT_EQ(rejected, nullptr);
    IREE_EXPECT_OK(loom_serve_text_model_free(rejected));
    EXPECT_EQ(allocations.rejected_bytes.load(),
              sizeof(iree_host_size_t) == 4
                  ? 0
                  : uint64_t{count} * 65536 * sizeof(uint32_t));
    EXPECT_EQ(allocations.live_count.load(), 0);
  }

  // Failure leaves the native device and allocator available for another
  // load. The same long-context source works with a representable map count.
  options.checkpoint_capacity = 2;
  loom_serve_text_model_t* recovered = nullptr;
  IREE_ASSERT_OK(
      loom_serve_text_model_allocate(device_, &options, allocator, &recovered));
  const int32_t tokens[] = {0, 1, 2, 0};
  auto* row = loom_serve_text_model_row(recovered, 0);
  IREE_EXPECT_OK(
      loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
  Oracle reference;
  reference.Consume("abxa");
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
  IREE_EXPECT_OK(loom_serve_text_model_free(recovered));
  EXPECT_EQ(allocations.live_count.load(), 0);
}

TEST_F(SourceTextServiceTest, RejectsImagesWithoutProjection) {
  Start();
  const std::string message =
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b");
  const std::string response = Read(Send(Request(message, 2)));
  EXPECT_NE(response.find("501 Not Implemented"), std::string::npos)
      << response;
  EXPECT_EQ(response.find("chat.completion"), std::string::npos) << response;
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "ab"), 2)))),
            Oracle::Complete("ab", 2));
}

TEST_F(SourceTextServiceTest, InputResourceLimitDoesNotRetireTheModel) {
  Start();
  std::string message = R"({"role":"user","content":[)";
  for (size_t i = 0; i <= LOOM_SERVE_TEXT_CHAT_MAX_CONTENT_PART_COUNT; ++i) {
    if (i) {
      message += ",";
    }
    message += R"({"type":"text","text":"a"})";
  }
  message += "]}";
  const std::string response = Read(Send(Request(message, 2)));
  EXPECT_NE(response.find("503 Service Unavailable"), std::string::npos)
      << response;
  EXPECT_NE(response.find("ordered content parts"), std::string::npos)
      << response;
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "ab"), 2)))),
            Oracle::Complete("ab", 2));
}

}  // namespace
