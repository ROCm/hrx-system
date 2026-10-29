// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>

#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/testing/service_application.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_F(SourceTextServiceTest, PreparationActivatesWeightsOnlyWithDemand) {
  for (const auto flags : {0u, uint32_t{LOOM_SERVE_TEXT_LOAD_FLAG_DEMAND}}) {
    SCOPED_TRACE(flags);
    auto* pool = loom_serve_pool_memory(storage_owner_);
    if (pool) {
      IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    }
    const auto before = loom_serve_pool_statistics(storage_owner_);
    const auto options = ModelOptions();
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    loom_serve_text_load_t* load = nullptr;
    IREE_ASSERT_OK(loom_serve_text_load_allocate(
        device_, storage_owner_, &options,
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
    const auto after = loom_serve_pool_statistics(storage_owner_);
    EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
    EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  }
}

TEST_F(SourceTextServiceTest, IndependentParameterPreparationJoinsAllDomains) {
  for (const bool fail_policy : {false, true}) {
    SCOPED_TRACE(fail_policy);
    const auto package =
        iree::testing::MakeTempFilePath("text_parameter_domains");
    ASSERT_TRUE(std::filesystem::create_directory(package));
    for (const char* file : {"advance.loom", "storage.loom", "control.loom",
                             "model.json", "tokenizer.json"}) {
      ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                             package + "/" + file));
    }
    // The media stage has an independent preparation policy and parameter
    // domain while the ordinary text stage retains the fixture's oracle.
    EditNativeProvider(directory_ + "/prepare.loom", package + "/prepare.loom",
                       "func.call @parameter(%media_stage, %c0_i32, "
                       "%c0_source, %checkpoint, %policy)",
                       "%media_policy = buffer.constant "
                       "utf8(\"media-weights.loom\") : buffer\n"
                       "  func.call @parameter(%media_stage, %c0_i32, "
                       "%c0_source, %checkpoint, %media_policy)");
    EditNativeProvider(package + "/prepare.loom", package + "/prepare.loom",
                       "func.return %effective, %c2_i32, %state",
                       "func.return %effective, %c1_i32, %state");
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/weights.loom",
                                           package + "/media-weights.loom"));
    if (fail_policy) {
      std::ofstream output(package + "/weights.loom");
      output << "invalid preparation policy\n";
      ASSERT_TRUE(output.good());
    } else {
      ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/weights.loom",
                                             package + "/weights.loom"));
    }
    auto* pool = loom_serve_pool_memory(storage_owner_);
    if (pool) {
      IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    }
    const auto before = loom_serve_pool_statistics(storage_owner_);
    auto options = ModelOptions();
    options.sources.base_directory = iree_make_cstring_view(package.c_str());
    loom_serve_text_model_t* model = nullptr;
    auto status = loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model);
    if (fail_policy) {
      IREE_EXPECT_NOT_OK(status);
      EXPECT_EQ(model, nullptr);
    } else {
      IREE_ASSERT_OK(status);
      const int32_t token = Oracle::Id('a');
      auto* row = loom_serve_text_model_row(model, 0);
      IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
      Oracle reference;
      reference.Consume("a");
      EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
      EXPECT_EQ(loom_serve_text_model_weight_activity(model).activations, 2u);
    }
    IREE_ASSERT_OK(loom_serve_text_model_free(model));
    if (pool) {
      IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    }
    const auto after = loom_serve_pool_statistics(storage_owner_);
    EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
    EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  }
}

TEST_F(SourceTextServiceTest, FailedLinkRetiresAcceptedInitialization) {
  const auto package = iree::testing::MakeTempFilePath("failed_text_link");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file : {"prepare.loom", "advance.loom", "storage.loom",
                           "model.json", "weights.loom", "tokenizer.json"}) {
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
  auto* pool = loom_serve_pool_memory(storage_owner_);
  if (pool) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  const auto before = loom_serve_pool_statistics(storage_owner_);
  auto options = ModelOptions();
  options.sources.base_directory = iree_make_cstring_view(package.c_str());
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  const auto wake = [](void* payload) {
    iree_notification_post(static_cast<iree_notification_t*>(payload),
                           IREE_ALL_WAITERS);
  };
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_allocate(device_, storage_owner_,
                                               &options, wake, &notification,
                                               host_allocator_, &load));
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
  const auto after = loom_serve_pool_statistics(storage_owner_);
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
  auto* pool = loom_serve_pool_memory(storage_owner_);
  if (pool) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  const auto before = loom_serve_pool_statistics(storage_owner_);
  auto options = ModelOptions();
  const auto missing = iree::testing::MakeTempFilePath("missing_checkpoint");
  ASSERT_FALSE(std::filesystem::exists(missing));
  options.weights_path = iree_make_cstring_view(missing.c_str());
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_allocate(
      device_, storage_owner_, &options,
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
  const auto after = loom_serve_pool_statistics(storage_owner_);
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

class WideCohortTextServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.context_capacity = 256;
    options.pool_capacity = 4096;
    options.row_count = 257;
    options.epoch_count = IREE_ARRAYSIZE(wide_shapes_);
    options.epoch_shapes = wide_shapes_;
    return options;
  }

  const loom_serve_packing_shape_t wide_shapes_[2] = {{1, 1}, {256, 129}};
};

class GrowingTextServiceTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const auto application_options =
        loom_serve_test_application_options_default();
    const loom_serve_pool_options_t pool_options = {
        .elastic =
            application_options.backing == LOOM_SERVE_POOL_BACKING_ELASTIC,
        .weight_slab_size = application_options.slab_size,
        .state_slab_size = application_options.slab_size,
    };
    const std::string name =
        "growth" + std::to_string(loom_serve_device_group_pool_count(group_));
    IREE_ASSERT_OK(loom_serve_device_group_create_pool(
        group_, {iree_make_cstring_view(name.c_str()), 1}, &pool_options,
        &pool_));
    const auto options = ModelOptions();
    IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, pool_, &options,
                                                  host_allocator_, &model_));
    EXPECT_EQ(loom_serve_text_model_pool(model_), pool_);
    EXPECT_EQ(
        loom_serve_residency_cache(loom_serve_text_model_residency(model_)),
        loom_serve_pool_residency_cache(pool_));
  }

  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.context_capacity = 256;
    options.pool_capacity = 4096;
    options.epoch_count = IREE_ARRAYSIZE(growth_shapes_);
    options.epoch_shapes = growth_shapes_;
    return options;
  }

  // Explicit storage owner independent of the execution device's default pool.
  loom_serve_pool_t* pool_ = nullptr;
  // Large input staging makes 129 residents cross physical slab boundaries,
  // while the actual numerical workload remains a few tokens per resident.
  const loom_serve_packing_shape_t growth_shapes_[1] = {{4096, 129}};
};

TEST_F(GrowingTextServiceTest, GrowthKeepsLiveHistoriesHandlesAndCheckpoints) {
  const auto default_before = loom_serve_pool_statistics(storage_owner_);
  std::vector<Oracle> references(129);
  auto* original = loom_serve_text_model_row(model_, 0);
  const int32_t prompt[] = {0, 1, 2};
  IREE_ASSERT_OK(loom_serve_text_row_prefill(original, 3, prompt));
  references[0].Consume("abx");
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(original, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  const auto activity = loom_serve_text_model_weight_activity(model_);
  const auto initial = loom_serve_text_model_memory_statistics(model_);
  for (size_t capacity : {17u, 129u}) {
    const auto previous_count = loom_serve_text_model_row_count(model_);
    const auto previous = loom_serve_text_model_memory_statistics(model_);
    loom_serve_text_resize_t* resize = nullptr;
    IREE_ASSERT_OK(
        loom_serve_text_model_prepare_resize(model_, capacity, &resize));
    ASSERT_NE(resize, nullptr);
    // The worker prepares host state only. Pool accounting and visible model
    // ownership remain unchanged until the serialized owner submits growth.
    EXPECT_EQ(loom_serve_text_model_row_count(model_), previous_count);
    EXPECT_EQ(loom_serve_text_model_row(model_, 0), original);
    const auto prepared = loom_serve_text_model_memory_statistics(model_);
    EXPECT_EQ(prepared.reserved_bytes, previous.reserved_bytes);
    EXPECT_EQ(prepared.committed_bytes, previous.committed_bytes);
    EXPECT_EQ(prepared.pending_bytes, previous.pending_bytes);
    EXPECT_EQ(prepared.releasing_bytes, previous.releasing_bytes);
    EXPECT_EQ(prepared.released_bytes, previous.released_bytes);
    loom_serve_execution_frontier_t frontier;
    auto status =
        loom_serve_text_model_submit_resize(model_, resize, &frontier);
    loom_serve_text_resize_free(resize);
    IREE_ASSERT_OK(status);
    for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
      IREE_ASSERT_OK(loom_serve_workspace_wait(loom_serve_pool_workspace(pool_),
                                               frontier.semaphores[i],
                                               frontier.values[i]));
    }
    loom_serve_text_model_complete_resize(model_);
    EXPECT_EQ(loom_serve_text_model_row_count(model_), capacity);
    EXPECT_EQ(loom_serve_text_model_row(model_, 0), original);
    EXPECT_EQ(loom_serve_text_model_memory_statistics(model_).committed_bytes,
              initial.committed_bytes);
    std::vector<int32_t> tokens(capacity);
    std::vector<loom_serve_text_span_t> spans(capacity);
    for (size_t i = 0; i < capacity; ++i) {
      tokens[i] = i % 3;
      references[i].Consume(tokens[i]);
      spans[i] = {i, 1, &tokens[i], LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
    }
    IREE_ASSERT_OK(
        loom_serve_text_model_epoch(model_, 0, capacity, spans.data()));
    for (size_t i = 0; i < capacity; ++i) {
      EXPECT_EQ(loom_serve_text_row_token(loom_serve_text_model_row(model_, i)),
                references[i].Predict())
          << i;
    }
  }
  auto* fork = loom_serve_text_model_row(model_, 128);
  IREE_ASSERT_OK(loom_serve_text_row_reset(fork));
  bool restored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  Oracle fork_reference;
  fork_reference.Consume("abx");
  EXPECT_EQ(loom_serve_text_row_token(fork), fork_reference.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_decode(fork));
  fork_reference.Consume(fork_reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(fork), fork_reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(original), references[0].Predict());
  loom_serve_text_checkpoint_release(checkpoint);
  const auto after = loom_serve_text_model_weight_activity(model_);
  EXPECT_EQ(after.activations, activity.activations);
  EXPECT_EQ(after.read_bytes, activity.read_bytes);
  const auto default_after = loom_serve_pool_statistics(storage_owner_);
  EXPECT_EQ(default_after.storage.bytes_reserved,
            default_before.storage.bytes_reserved);
  EXPECT_EQ(default_after.retained.reserved_bytes,
            default_before.retained.reserved_bytes);
}

TEST_F(WideCohortTextServiceTest, WideEpochsPreserveSparseResidentsAndForks) {
  constexpr size_t kCount = 129;
  std::vector<Oracle> references(kCount);
  std::vector<int32_t> tokens(2 * kCount);
  std::vector<loom_serve_text_span_t> spans(kCount);
  for (size_t epoch = 0; epoch < 3; ++epoch) {
    for (size_t i = 0; i < kCount; ++i) {
      const size_t row = 256 - 2 * i;
      const size_t count = epoch ? 1 : 1 + i % 2;
      tokens[2 * i] = epoch ? references[i].Predict() : i % 3;
      tokens[2 * i + 1] = (i + 1) % 3;
      for (size_t j = 0; j < count; ++j) {
        references[i].Consume(tokens[2 * i + j]);
      }
      spans[i] = {row, count, &tokens[2 * i], LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
    }
    const loom_serve_text_span_t duplicate[] = {spans[0], spans[0]};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_text_model_epoch(model_, 1, 2, duplicate));
    IREE_ASSERT_OK(
        loom_serve_text_model_epoch(model_, 1, kCount, spans.data()));
    for (size_t i = 0; i < kCount; ++i) {
      EXPECT_EQ(loom_serve_text_row_token(
                    loom_serve_text_model_row(model_, spans[i].row_index)),
                references[i].Predict());
    }
  }

  // Retained identity 256 forks into 255, neither a bit in a machine word nor
  // an ordinal in the previous cohort. Each writer must detach shared state.
  auto* original = loom_serve_text_model_row(model_, 256);
  auto* fork = loom_serve_text_model_row(model_, 255);
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(original, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  bool restored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  const int32_t a = 0, b = 1;
  const loom_serve_text_span_t branches[] = {
      {255, 1, &a, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {256, 1, &b, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}};
  IREE_ASSERT_OK(loom_serve_text_model_epoch(model_, 1, 2, branches));
  Oracle fork_reference = references[0];
  fork_reference.Consume(a);
  references[0].Consume(b);
  EXPECT_EQ(loom_serve_text_row_token(fork), fork_reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(original), references[0].Predict());
  loom_serve_text_checkpoint_release(checkpoint);
  for (size_t i = 0; i < 257; ++i) {
    IREE_ASSERT_OK(
        loom_serve_text_row_reset(loom_serve_text_model_row(model_, i)));
  }
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 4096u);
}

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
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_allocate(device_, storage_owner_, &options,
                                     host_allocator_, &rejected));
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
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model_));
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
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_allocate(device_, storage_owner_, &options,
                                     host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  options.draft_depth = 0;
  options.options = IREE_SV("{\"unknown\":true}");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_allocate(device_, storage_owner_, &options,
                                     host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  options.options = IREE_SV("{}");
  options.pool_capacity = 0;
  options.checkpoint_capacity = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_allocate(device_, storage_owner_, &options,
                                     host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(WriteWeights(7));
  options = ModelOptions();
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &model_));
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
    IREE_EXPECT_STATUS_IS(expected_status, loom_serve_text_model_allocate(
                                               device_, storage_owner_,
                                               &options, allocator, &rejected));
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
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, allocator, &recovered));
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
  EXPECT_NE(response.find("400 Bad Request"), std::string::npos) << response;
  EXPECT_EQ(response.find("chat.completion"), std::string::npos) << response;
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "ab"), 2)))),
            Oracle::Complete("ab", 2));
}

TEST_F(SourceTextServiceTest, InputContentLimitDoesNotRetireTheModel) {
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
  EXPECT_NE(response.find("400 Bad Request"), std::string::npos) << response;
  EXPECT_NE(response.find("ordered content parts"), std::string::npos)
      << response;
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "ab"), 2)))),
            Oracle::Complete("ab", 2));
}

}  // namespace
