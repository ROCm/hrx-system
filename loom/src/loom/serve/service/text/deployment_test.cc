// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <condition_variable>
#include <mutex>

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/service_application.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class TextDeploymentTest : public SourceTextServiceTest {
 protected:
  static void SetUpTestSuite() {
    auto options = loom_serve_test_application_options_default();
    options.device_count = 3;
    if (options.backing == LOOM_SERVE_POOL_BACKING_ELASTIC) {
      options.memory_limit = 64 * 1024 * 1024;
    }
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &options, iree_allocator_system(), &application));
    auto* group = loom_serve_test_application_device_group(application);
    // One application owns every test's devices. Explicit peer-access pools
    // also exercise storage ownership independent of device identity.
    loom_serve_pool_options_t pool_options = {
        .elastic = true,
        .weight_slab_size = 4096,
        .state_slab_size = 4096,
    };
    const iree_string_view_t names[] = {IREE_SV("shared0"), IREE_SV("shared1"),
                                        IREE_SV("shared2"), IREE_SV("fixed0"),
                                        IREE_SV("fixed1"),  IREE_SV("fixed2")};
    for (size_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
      pool_options.elastic = i < 3;
      pool_options.memory_limit = pool_options.elastic ? 256 * 1024 : 0;
      loom_serve_pool_t* pool = nullptr;
      IREE_ASSERT_OK(loom_serve_device_group_create_pool(
          group, {.name = names[i], .device_mask = 7}, &pool_options, &pool));
    }
  }

  bool NeedsStandaloneModel() const override { return false; }
  iree_host_size_t DeviceCount() const override { return 3; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    for (size_t i = 0; i < devices_.size(); ++i) {
      // Source ordinals deliberately differ from application order.
      devices_[i] = loom_serve_device_group_at(group_, (i + 2) % 3);
      pools_[i] = loom_serve_device_group_pool_at(group_, i + 3);
    }
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
    }
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(held_));
    loom_serve_text_variant_free(candidate_);
    SourceTextServiceTest::TearDown();
    iree_hal_semaphore_release(gate_);
    for (auto* device : devices_) {
      loom_serve_device_trim_programs(device);
    }
    for (size_t i = 3; i < loom_serve_device_group_pool_count(group_); ++i) {
      auto* pool = loom_serve_device_group_pool_at(group_, i);
      IREE_EXPECT_OK(loom_serve_test_workspace_await_idle(
          loom_serve_pool_workspace(pool)));
      IREE_EXPECT_OK(loom_serve_test_pool_trim_and_drain(pool, 0));
      const auto memory = loom_serve_pool_statistics(pool);
      EXPECT_EQ(memory.retained.committed_bytes, 0u);
      EXPECT_EQ(memory.retained.pending_bytes, 0u);
      EXPECT_EQ(memory.retained.releasing_bytes, 0u);
      EXPECT_EQ(memory.workspace.bytes_committed, 0u);
      EXPECT_FALSE(loom_serve_pool_is_in_use(pool));
    }
  }

  iree_status_t Load(const loom_serve_text_options_t& options) {
    return loom_serve_text_model_allocate(devices_.size(), devices_.data(),
                                          pools_.data(), &options,
                                          host_allocator_, &model_);
  }

  void CompilePrivateContributions(size_t bytes = 16 * 1024) {
    const auto path =
        iree::testing::MakeTempFilePath("private_contributions", ".loom");
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/advance.loom", path,
        "config.def @fixture.private_bytes = 0 : offset",
        "config.def @fixture.private_bytes = " + std::to_string(bytes) +
            " : offset"));
    ASSERT_NO_FATAL_FAILURE(
        EditNativeProvider(path, path, "scalar.muli %value, %coefficient32",
                           "scalar.addi %value, %coefficient32"));
    const auto provider = iree_make_cstring_view(path.c_str());
    const loom_serve_sources_t sources = {.count = 1, .values = &provider};
    struct Compilation {
      // Joins the terminal callback's final access with the outer owner.
      std::mutex mutex;
      // Compiler workers publish without waiting for any other task.
      std::condition_variable condition;
      // Owned outcome transferred after terminal publication.
      iree_status_t status = iree_ok_status();
      // Owned successful or failed candidate.
      loom_serve_text_variant_t* variant = nullptr;
      // Protected by mutex, including the callback's final notification.
      bool complete = false;
    } compilation;
    IREE_ASSERT_OK(loom_serve_text_model_compile_variant_async(
        model_, &sources, nullptr,
        [](void* payload, iree_status_t status, void* candidate,
           const loom_serve_compilation_report_t* report) {
          auto* compilation = static_cast<Compilation*>(payload);
          std::lock_guard<std::mutex> lock(compilation->mutex);
          compilation->status = status;
          compilation->variant =
              static_cast<loom_serve_text_variant_t*>(candidate);
          compilation->complete = true;
          compilation->condition.notify_one();
        },
        &compilation));
    std::unique_lock<std::mutex> lock(compilation.mutex);
    compilation.condition.wait(lock, [&] { return compilation.complete; });
    candidate_ = compilation.variant;
    IREE_ASSERT_OK(compilation.status);
  }

  void CheckToken(size_t row_index, uint32_t bias) {
    const int32_t token = 1;
    auto* row = loom_serve_text_model_row(model_, row_index);
    IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
    Oracle oracle(bias);
    oracle.Consume(token);
    EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
  }

  void StartFailedWarmup() {
    IREE_ASSERT_OK(Load(ModelOptions()));
    // Preparation indexes the real file without reading its payload. A later
    // truncation fails deferred loading through the model's actual groups.
    std::filesystem::resize_file(weights_, 0);
    IREE_ASSERT_OK(loom_serve_text_warmup_allocate(
        model_->residency.count, devices_.data(), pools_.data(),
        model_->residency.groups, nullptr, nullptr, host_allocator_,
        &model_->warmup));
    loom_serve_text_warmup_poll(model_->warmup);
  }

  // Ordered execution and storage spans remain alive through model retirement.
  std::array<loom_serve_device_t*, 3> devices_ = {};
  // Explicit pool membership permits aliases without merging executions.
  std::array<loom_serve_pool_t*, 3> pools_ = {};
  // Candidate remains owned even when replacement validation rejects it.
  loom_serve_text_variant_t* candidate_ = nullptr;
  // Native predecessor is always released before fixture teardown joins work.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Actual competing backing, not a fabricated accounting counter.
  loom_serve_virtual_buffer_t* held_ = nullptr;
  // Statistics outlive the competing reservation and its queued disposal.
  loom_serve_memory_statistics_t held_usage_ = {};
};

TEST_F(TextDeploymentTest, CompleteOwnerSetProducesRetainedHttpOutput) {
  IREE_ASSERT_OK(Load(ModelOptions()));
  iree_host_size_t count = 0;
  auto* const* residencies = loom_serve_text_model_residencies(model_, &count);
  ASSERT_EQ(count, devices_.size());
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(loom_serve_residency_cache(residencies[i]),
              loom_serve_pool_residency_cache(pools_[i]));
  }
  IREE_ASSERT_OK(loom_serve_text_model_activate(model_));
  EXPECT_EQ(loom_serve_text_model_weight_activity(model_).activations, 3u);
  Start();
  const std::array<std::string, 3> prompts = {"abxa", "xab", "bbax"};
  std::array<int, 3> clients;
  for (size_t i = 0; i < prompts.size(); ++i) {
    clients[i] = Send(Request(Message("user", prompts[i]), 4),
                      "X-Loom-Session: owner" + std::to_string(i) + "\r\n");
  }
  for (size_t i = 0; i < prompts.size(); ++i) {
    const auto first = Content(Read(clients[i]));
    EXPECT_EQ(first, Oracle::Complete(prompts[i], 4, 12));
    const auto next = Read(Send(
        Request(Message("user", prompts[i]) + "," +
                    Message("assistant", first) + "," + Message("user", "bax"),
                4),
        "X-Loom-Session: owner" + std::to_string(i) + "\r\n"));
    EXPECT_EQ(Content(next),
              Oracle::Complete(prompts[i] + first + "#bax", 4, 12));
    EXPECT_NE(next.find("\"cached_tokens\":" +
                        std::to_string(prompts[i].size() + first.size() - 1)),
              std::string::npos)
        << next;
  }
}

TEST_F(TextDeploymentTest, FailedWarmupStillReleasesTheRetiredModel) {
  ASSERT_NO_FATAL_FAILURE(StartFailedWarmup());
  loom_serve_text_warmup_await(model_->warmup);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_text_warmup_status(model_->warmup));

  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_model_retire(
      model_,
      [](void* payload) {
        iree_notification_post(static_cast<iree_notification_t*>(payload),
                               IREE_ALL_WAITERS);
      },
      &notification);
  std::array<loom_serve_workspace_t*, 3> workspaces;
  for (size_t i = 0; i < pools_.size(); ++i) {
    workspaces[i] = loom_serve_pool_workspace(pools_[i]);
  }
  IREE_EXPECT_OK(loom_serve_workspace_await(
      workspaces.size(), workspaces.data(), &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      model_));
  EXPECT_TRUE(loom_serve_text_model_is_retired(model_));
  IREE_EXPECT_OK(
      loom_serve_text_model_free_retired(std::exchange(model_, nullptr)));
  iree_notification_deinitialize(&notification);
}

TEST_F(TextDeploymentTest, ColdShutdownReportsTheWarmupFailure) {
  ASSERT_NO_FATAL_FAILURE(StartFailedWarmup());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_text_model_free(std::exchange(model_, nullptr)));
}

TEST_F(TextDeploymentTest, AliasedPoolsSumConcurrentExecutionPeaks) {
  for (const std::array<size_t, 3> owners :
       {std::array<size_t, 3>{0, 1, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}}) {
    SCOPED_TRACE(::testing::Message() << owners[0] << owners[1] << owners[2]);
    for (size_t i = 0; i < owners.size(); ++i) {
      pools_[i] = loom_serve_device_group_pool_at(group_, owners[i] + 3);
    }
    IREE_ASSERT_OK(Load(ModelOptions()));
    const auto maximum = *std::max_element(owners.begin(), owners.end());
    EXPECT_EQ(model_->residency.count, maximum + 1);
    for (size_t i = 0; i < owners.size(); ++i) {
      EXPECT_EQ(model_->residency.pool_indices[i], owners[i]);
    }
    ASSERT_NO_FATAL_FAILURE(CheckToken(0, 12));
    // Each auxiliary execution fits alone. Their simultaneous sum cannot fit
    // a shared 256 KiB pool, even though both commands use the same parameter.
    ASSERT_NO_FATAL_FAILURE(CompilePrivateContributions(144 * 1024));
    if (owners[1] == owners[2]) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          loom_serve_text_model_select_variant(model_, candidate_));
      ASSERT_NO_FATAL_FAILURE(CheckToken(1, 12));
    } else {
      IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, candidate_));
      ASSERT_NO_FATAL_FAILURE(CheckToken(1, 11));
    }
    IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, nullptr));
    ASSERT_NO_FATAL_FAILURE(CheckToken(2, 12));
    loom_serve_text_variant_free(candidate_);
    candidate_ = nullptr;
    IREE_ASSERT_OK(loom_serve_text_model_free(model_));
    model_ = nullptr;
    for (size_t i = 0; i < owners.size(); ++i) {
      IREE_ASSERT_OK(loom_serve_test_pool_trim_and_drain(
          loom_serve_device_group_pool_at(group_, i + 3), 0));
    }
  }
}

TEST_F(TextDeploymentTest, LateOwnerPeakRejectionPreservesSelectedProgram) {
  IREE_ASSERT_OK(Load(ModelOptions()));
  ASSERT_NO_FATAL_FAILURE(CompilePrivateContributions());
  auto* row = loom_serve_text_model_row(model_, 0);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 16, &admitted));
  ASSERT_TRUE(admitted);
  ASSERT_NO_FATAL_FAILURE(CheckToken(0, 12));
  auto* pool = loom_serve_pool_memory(pools_.back());
  const uint64_t bytes = loom_serve_memory_pool_limit(pool) -
                         loom_serve_memory_pool_live_bytes(pool) - 4096;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_state_geometry(pools_.back()), bytes, 256,
      &held_usage_, &held_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(held_, 0, bytes));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_text_model_select_variant(model_, candidate_));
  Oracle original(12);
  original.Consume(1);
  original.Consume(original.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_decode(row));
  EXPECT_EQ(loom_serve_text_row_token(row), original.Predict());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(held_));
  held_ = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, candidate_));
  ASSERT_NO_FATAL_FAILURE(CheckToken(1, 11));
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, nullptr));
  ASSERT_NO_FATAL_FAILURE(CheckToken(2, 12));
}

TEST_F(TextDeploymentTest, LatePinnedOwnerPreventsPartialDeactivation) {
  IREE_ASSERT_OK(Load(ModelOptions()));
  IREE_ASSERT_OK(loom_serve_text_model_activate(model_));
  loom_serve_residency_set_pinned(model_->residency.values[2], true);
  std::array<loom_serve_residency_snapshot_t, 3> before;
  for (size_t i = 0; i < before.size(); ++i) {
    before[i] = loom_serve_residency_snapshot(model_->residency.values[i]);
  }
  bool complete = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_model_deactivate(model_, &complete));
  EXPECT_FALSE(complete);
  for (size_t i = 0; i < before.size(); ++i) {
    const auto after =
        loom_serve_residency_snapshot(model_->residency.values[i]);
    EXPECT_EQ(after.active, before[i].active);
    EXPECT_EQ(after.activity.deactivations, before[i].activity.deactivations);
    EXPECT_EQ(after.memory.releasing_bytes, 0u);
  }
  loom_serve_residency_set_pinned(model_->residency.values[2], false);
  while (!complete) {
    IREE_ASSERT_OK(loom_serve_text_model_deactivate(model_, &complete));
    for (auto* pool : pools_) {
      IREE_ASSERT_OK(
          loom_serve_memory_pool_await(loom_serve_pool_memory(pool)));
    }
  }
  EXPECT_FALSE(loom_serve_text_model_parameter_snapshot(model_).active);
  ASSERT_NO_FATAL_FAILURE(CheckToken(0, 12));
}

TEST_F(TextDeploymentTest, FixedAndMixedBackingPreserveCompleteOwnership) {
  for (bool mixed : {false, true}) {
    SCOPED_TRACE(mixed);
    for (size_t i = 0; i < pools_.size(); ++i) {
      pools_[i] =
          loom_serve_device_group_pool_at(group_, i + (mixed && i < 2 ? 3 : 6));
    }
    IREE_ASSERT_OK(Load(ModelOptions()));
    IREE_ASSERT_OK(loom_serve_text_model_activate(model_));
    bool complete = false;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                          loom_serve_text_model_deactivate(model_, &complete));
    EXPECT_FALSE(complete);
    // A fixed last owner must not leave the earlier elastic weights unloaded.
    EXPECT_EQ(loom_serve_text_model_weight_activity(model_).deactivations, 0u);
    ASSERT_NO_FATAL_FAILURE(CheckToken(0, 12));
    IREE_ASSERT_OK(loom_serve_text_model_free(model_));
    model_ = nullptr;
  }
}

TEST_F(TextDeploymentTest, FailedEpochRetiresAcceptedWorkOnEveryOwner) {
  const auto path = iree::testing::MakeTempFilePath("failed_epoch", ".loom");
  ASSERT_NO_FATAL_FAILURE(
      EditNativeProvider(directory_ + "/control.loom", path,
                         "config.def @fixture.fail_epoch = false",
                         "config.def @fixture.fail_epoch = true"));
  auto options = ModelOptions();
  options.sources.control = iree_make_cstring_view(path.c_str());
  IREE_ASSERT_OK(Load(options));
  ASSERT_NO_FATAL_FAILURE(CompilePrivateContributions());
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, candidate_));
  IREE_ASSERT_OK(loom_serve_text_model_activate(model_));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(loom_serve_device_handle(devices_.back()),
                                IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  uint64_t value = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(model_->executions[2], {1, &gate_, &value}));
  const int32_t token = 1;
  const loom_serve_text_span_t span = {0, 1, &token,
                                       LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
  loom_serve_execution_frontier_t frontier;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_model_submit_epoch(model_, 0, 1, &span, nullptr, nullptr,
                                         nullptr, &frontier));
  for (size_t i = 0; i < devices_.size(); ++i) {
    EXPECT_GT(
        loom_serve_residency_snapshot(model_->residency.values[i]).pin_count,
        0u);
    EXPECT_GT(loom_serve_execution_frontier(model_->executions[i]).values[0],
              0u);
  }
  loom_serve_text_variant_free(candidate_);
  candidate_ = nullptr;
  loom_serve_text_model_retire(model_, nullptr, nullptr);
  EXPECT_FALSE(loom_serve_text_model_is_retired(model_));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  iree_hal_semaphore_release(gate_);
  gate_ = nullptr;
  IREE_ASSERT_OK(
      loom_serve_execution_drain_all(model_->device_count, model_->executions));
  EXPECT_TRUE(loom_serve_text_model_is_retired(model_));
}

TEST_F(TextDeploymentTest, ColdHistoryOperationsProgressPeerWorkspaces) {
  IREE_ASSERT_OK(Load(ModelOptions()));
  ASSERT_NO_FATAL_FAILURE(CompilePrivateContributions());
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, candidate_));
  auto* history = loom_serve_text_model_row(model_, 0);
  auto* running = loom_serve_text_model_row(model_, 1);
  ASSERT_NO_FATAL_FAILURE(CheckToken(0, 11));
  ASSERT_NO_FATAL_FAILURE(CheckToken(1, 11));
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(history, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  const std::unique_ptr<loom_serve_text_checkpoint_t,
                        decltype(&loom_serve_text_checkpoint_release)>
      retained(checkpoint, loom_serve_text_checkpoint_release);
  Oracle oracle(11);
  oracle.Consume(1);
  for (int operation = 0; operation < 4; ++operation) {
    SCOPED_TRACE(operation);
    for (size_t i = 1; i < pools_.size(); ++i) {
      IREE_ASSERT_OK(loom_serve_test_pool_trim_and_drain(pools_[i], 0));
      EXPECT_EQ(loom_serve_pool_statistics(pools_[i]).workspace.bytes_committed,
                0u);
    }
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(loom_serve_text_row_submit_decode(running, &frontier));
    bool restored = false;
    switch (operation) {
      case 0:
        IREE_ASSERT_OK(loom_serve_text_checkpoint_suspend(checkpoint));
        EXPECT_GT(loom_serve_text_checkpoint_suspended_bytes(checkpoint), 0u);
        break;
      case 1:
        IREE_ASSERT_OK(
            loom_serve_text_row_try_restore(history, checkpoint, &restored));
        ASSERT_TRUE(restored);
        break;
      case 2:
        IREE_ASSERT_OK(loom_serve_text_row_suspend(history));
        EXPECT_GT(loom_serve_text_row_suspended_bytes(history), 0u);
        break;
      case 3:
        IREE_ASSERT_OK(loom_serve_text_row_try_resume(history, &restored));
        ASSERT_TRUE(restored);
        break;
    }
    IREE_ASSERT_OK(loom_serve_text_row_complete_step(running));
    oracle.Consume(oracle.Predict());
    EXPECT_EQ(loom_serve_text_row_token(running), oracle.Predict());
  }
  Oracle restored(11);
  restored.Consume(1);
  restored.Consume(restored.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_decode(history));
  EXPECT_EQ(loom_serve_text_row_token(history), restored.Predict());
}

}  // namespace
