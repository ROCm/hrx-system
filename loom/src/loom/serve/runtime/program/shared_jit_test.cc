// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <utility>

#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/jit.h"
#include "loomc/iree.h"
#include "loomc/pass_trace.h"

namespace {

// Owns asynchronous layout/completion observations through task retirement.
class Compilation {
 public:
  ~Compilation() {
    if (started_) {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [&] { return complete_; });
    }
    iree_status_free(status_);
    loom_serve_jit_stage_free(stage_);
  }

  void Start(loom_serve_jit_t* jit, loomc_task_sink_t sink,
             iree_string_view_t root = IREE_SV("advance"),
             bool direct = false) {
    started_ = true;
    auto status = direct
                      ? loom_serve_jit_compile_dispatch_async(
                            jit, sink, root, &config_, Complete, this)
                      : loom_serve_jit_compile_async(
                            jit, sink, root, &config_,
                            [](void* payload, const loom_cmd_program_t*) {
                              auto* self = static_cast<Compilation*>(payload);
                              std::lock_guard<std::mutex> lock(self->mutex_);
                              self->layout_ = true;
                              self->condition_.notify_all();
                            },
                            Complete, this);
    if (!iree_status_is_ok(status)) {
      Complete(this, status, nullptr);
    }
  }

  void AwaitLayout() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return layout_ || complete_; });
    EXPECT_TRUE(layout_);
    EXPECT_FALSE(complete_);
  }

  iree_status_t Await() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return complete_; });
    return iree_status_clone(status_);
  }

  loom_serve_jit_stage_t* stage() const { return stage_; }

 private:
  static void Complete(void* payload, iree_status_t status,
                       loom_serve_jit_stage_t* stage) {
    auto* self = static_cast<Compilation*>(payload);
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->status_ = status;
    self->stage_ = stage;
    self->complete_ = true;
    self->condition_.notify_all();
  }

  // Request configuration borrowed until completion.
  const loomc_config_options_t config_ = {
      nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  // Serializes the observing thread with compiler callbacks.
  std::mutex mutex_;
  // Explicit layout and terminal completion events.
  std::condition_variable condition_;
  // Whether Start accepted responsibility for completion.
  bool started_ = false;
  // Command construction reached native publication.
  bool layout_ = false;
  // Final callback has relinquished this record.
  bool complete_ = false;
  // Owned terminal outcome, retained through repeated observations.
  iree_status_t status_ = iree_ok_status();
  // Owned partial or complete stage.
  loom_serve_jit_stage_t* stage_ = nullptr;
};

// The parent runs on the actual executor, while its accepted native tasks wait
// outside workers. Opening the gate also admits their later load continuations.
class NativeGate {
 public:
  explicit NativeGate(loomc_task_sink_t target) : target_(target) {}
  ~NativeGate() { Open(); }
  loomc_task_sink_t sink() {
    return {[](void* payload, loomc_task_t* task) {
              auto* self = static_cast<NativeGate*>(payload);
              std::unique_lock<std::mutex> lock(self->mutex_);
              if (!self->parent_ || self->open_) {
                self->parent_ = true;
                lock.unlock();
                return loomc_task_sink_submit(self->target_, task);
              }
              task->next = self->pending_;
              self->pending_ = task;
              return loomc_ok_status();
            },
            this};
  }
  void Open() {
    loomc_task_t* pending = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
      pending = std::exchange(pending_, nullptr);
    }
    while (pending) {
      auto* task = pending;
      pending = task->next;
      task->next = nullptr;
      IREE_CHECK_OK(
          iree_status_from_loomc(loomc_task_sink_submit(target_, task)));
    }
  }

 private:
  // Shared compiler queue preserving real worker ordinals.
  loomc_task_sink_t target_;
  // Serializes publication with the outer-thread release.
  std::mutex mutex_;
  // First task is the independent parent, not a native compilation.
  bool parent_ = false;
  // All future submissions pass to the real queue after release.
  bool open_ = false;
  // Accepted children retained without occupying executor workers.
  loomc_task_t* pending_ = nullptr;
};

class SharedJitTest : public loom_serve::testing::JitTest {
 protected:
  iree_host_size_t DeviceCount() const override { return 2; }

  void SetUp() override {
    JitTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    peer_device_ = loom_serve_device_group_at(group_, 1);
    peer_pool_ = loom_serve_device_group_pool_at(group_, 1);
    EXPECT_NE(loom_serve_device_handle(peer_device_), device_);
    EXPECT_EQ(loom_serve_device_host(peer_device_),
              loom_serve_device_host(device_owner_));
    IREE_ASSERT_OK(CreatePeer());
    IREE_ASSERT_OK(loom_serve_device_create_execution(peer_device_, peer_pool_,
                                                      &peer_execution_));
  }

  void TearDown() override {
    if (peer_execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(peer_execution_));
    }
    iree_hal_buffer_release(peer_buffer_);
    loom_serve_jit_release(peer_);
    loom_serve_execution_release(peer_execution_);
    if (peer_device_) {
      loom_serve_device_trim_programs(peer_device_);
      IREE_EXPECT_OK(loom_serve_test_pool_trim_and_drain(peer_pool_, 0));
    }
    JitTest::TearDown();
  }

  iree_status_t CreatePeer(
      const loomc_sanitizer_options_t* sanitizer = nullptr) {
    const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                            IREE_SV("stage.loom")};
    const loom_serve_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = IREE_ARRAYSIZE(providers),
        .values = providers,
    };
    return loom_serve_jit_create(loom_serve_device_handle(peer_device_),
                                 loom_serve_device_dispatch_queue(peer_device_),
                                 loom_serve_device_host(peer_device_), &sources,
                                 sanitizer, &peer_report_, host_allocator_,
                                 &peer_);
  }

  void Execute(loom_serve_jit_stage_t* stage, bool peer) {
    Execute(
        stage, peer,
        loom_serve_jit_stage_dispatch(stage)
            ? std::array<int32_t, 8>{105, 100, 100, 100, 100, 100, 100, 100}
            : std::array<int32_t, 8>{105, 106, 107, 108, 109, 110, 111, 112});
  }

  void Execute(loom_serve_jit_stage_t* stage, bool peer,
               const std::array<int32_t, 8>& expected) {
    auto* owner = peer ? peer_device_ : device_owner_;
    auto* pool = peer ? peer_pool_ : storage_owner_;
    auto* execution = peer ? peer_execution_ : execution_;
    auto** buffer = peer ? &peer_buffer_ : &buffer_;
    if (!*buffer) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          loom_serve_pool_storage(pool),
          iree_hal_make_pool_reservation_request(sizeof(input_), 0),
          iree_infinite_timeout(), buffer));
    }
    iree_hal_command_buffer_t* command = nullptr;
    auto* direct = loom_serve_jit_stage_dispatch(stage);
    if (!direct) {
      IREE_ASSERT_OK(loom_serve_jit_stage_record(
          stage, iree_hal_queue_family(loom_serve_device_dispatch_queue(owner)),
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, nullptr, &command));
    }
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {input_.data(), *buffer, 0, sizeof(input_)},
    };
    uint64_t completion = 0;
    auto status =
        loom_serve_execution_transfer(execution, 1, &upload, &completion);
    const iree_hal_buffer_binding_t binding = {*buffer, 0, sizeof(input_)};
    if (iree_status_is_ok(status)) {
      if (direct) {
        iree_hal_buffer_ref_t scratch;
        status = loom_serve_dispatch_issue(direct, execution, {1, &binding},
                                           &scratch, &completion);
      } else {
        status = loom_serve_execution_execute(execution, command, {1, &binding},
                                              &completion);
      }
    }
    iree_hal_command_buffer_release(command);
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {*buffer, 0, output_.data(), sizeof(output_)},
    };
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_execution_feedback(execution, 1, &download, &completion);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback_wait(execution, completion);
    }
    IREE_ASSERT_OK(status);
    EXPECT_EQ(output_, expected);
  }

  void CheckLiveStageArtifacts(bool direct) {
    const auto root = direct ? IREE_SV("increment_0") : IREE_SV("advance");
    const size_t count = direct ? 1 : 8;
    {
      Compilation first;
      first.Start(jit_, loom_serve_jit_task_sink(jit_), root, direct);
      IREE_ASSERT_OK(first.Await());
      EXPECT_EQ(compilation_.kernels, count);
      // Preparation releases its compiler catalog while the model's immutable
      // stage remains live. A later device still reuses that exact compilation.
      loom_serve_jit_release(std::exchange(jit_, nullptr));
      Compilation second;
      second.Start(peer_, loom_serve_jit_task_sink(peer_), root, direct);
      IREE_ASSERT_OK(second.Await());
      EXPECT_EQ(peer_report_.kernels, 0u);
      EXPECT_EQ(peer_report_.reused_kernels, count);
      loom_serve_jit_release(std::exchange(peer_, nullptr));
      Execute(first.stage(), false);
      Execute(second.stage(), true);
    }
    // Both stages and catalogs have retired. No host-owned history keeps their
    // input keys alive, so a new consumer compiles again.
    IREE_ASSERT_OK(CreatePeer());
    peer_report_ = {};
    Compilation fresh;
    fresh.Start(peer_, loom_serve_jit_task_sink(peer_), root, direct);
    IREE_ASSERT_OK(fresh.Await());
    EXPECT_EQ(peer_report_.kernels, count);
    EXPECT_EQ(peer_report_.reused_kernels, 0u);
    Execute(fresh.stage(), true);
  }

  // Independent device owner sharing only the host and immutable artifacts.
  loom_serve_device_t* peer_device_ = nullptr;
  // Group-owned backing belonging to the second execution domain.
  loom_serve_pool_t* peer_pool_ = nullptr;
  // Independently loaded native catalog and copied compilation evidence.
  loom_serve_jit_t* peer_ = nullptr;
  // Terminal report survives both catalog lifetimes.
  loom_serve_compilation_report_t peer_report_ = {};
  // Independent submission/feedback timelines.
  loom_serve_execution_t* peer_execution_ = nullptr;
  // Peer device state, never reused across device domains.
  iree_hal_buffer_t* peer_buffer_ = nullptr;
};

TEST_P(SharedJitTest, PendingArtifactsLoadAndExecuteOnIndependentDevices) {
  Compilation first, second, remapped;
  NativeGate gate(loom_serve_jit_task_sink(jit_));
  first.Start(jit_, gate.sink());
  first.AwaitLayout();
  second.Start(peer_, loom_serve_jit_task_sink(peer_));
  second.AwaitLayout();
  // The same pending producers serve a different parent's sparse, reordered
  // binding map after that consumer has relinquished its serialized inputs.
  remapped.Start(peer_, loom_serve_jit_task_sink(peer_),
                 IREE_SV("selected_pair"));
  remapped.AwaitLayout();
  // Relinquish the originating catalog while its producer and the other
  // device's subscription are both pending. Accepted jobs alone keep it alive.
  loom_serve_jit_release(std::exchange(jit_, nullptr));
  gate.Open();
  IREE_ASSERT_OK(first.Await());
  IREE_ASSERT_OK(second.Await());
  IREE_ASSERT_OK(remapped.Await());
  EXPECT_EQ(compilation_.kernels, 8u);
  EXPECT_EQ(peer_report_.kernels, 0u);
  EXPECT_EQ(peer_report_.reused_kernels, 10u);
  EXPECT_EQ(loom_serve_jit_cache_storage(peer_).executable_count, 8u);
  // Stages retain device-local loaded code and its exact immutable input after
  // both compilation catalogs have retired.
  loom_serve_jit_release(std::exchange(peer_, nullptr));
  Execute(first.stage(), false);
  Execute(second.stage(), true);
  Execute(remapped.stage(), true, {100, 100, 107, 100, 100, 100, 100, 124});
  // A new catalog reuses the live stages without retaining their compilers.
  IREE_ASSERT_OK(CreatePeer());
  peer_report_ = {};
  Compilation fresh;
  fresh.Start(peer_, loom_serve_jit_task_sink(peer_));
  IREE_ASSERT_OK(fresh.Await());
  EXPECT_EQ(peer_report_.kernels, 0u);
  EXPECT_EQ(peer_report_.reused_kernels, 8u);
  Execute(fresh.stage(), true);
}

TEST_P(SharedJitTest, CachedArtifactsDoNotBypassPipelineOptionValidation) {
  Compilation first;
  first.Start(jit_, loom_serve_jit_task_sink(jit_));
  IREE_ASSERT_OK(first.Await());
  EXPECT_EQ(compilation_.kernels, 8u);
  loom_serve_jit_release(std::exchange(peer_, nullptr));
  const loomc_pass_trace_options_t trace = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .structure_size = sizeof(trace),
      .flags = LOOMC_PASS_TRACE_FLAG_BEFORE_ALL,
      .sink = {[](void*, loomc_string_view_t) { return loomc_ok_status(); },
               nullptr},
  };
  const loomc_sanitizer_options_t sanitizer = {
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(sanitizer),
      .next = &trace,
  };
  IREE_ASSERT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, CreatePeer(&sanitizer));
  EXPECT_EQ(peer_, nullptr);
  IREE_ASSERT_OK(CreatePeer());
  Compilation second;
  second.Start(peer_, loom_serve_jit_task_sink(peer_));
  IREE_ASSERT_OK(second.Await());
  EXPECT_EQ(peer_report_.kernels, 0u);
  EXPECT_EQ(peer_report_.reused_kernels, 8u);
  Execute(second.stage(), true);
}

TEST_P(SharedJitTest, LiveStagesRetainArtifactsAfterCatalogRetirement) {
  CheckLiveStageArtifacts(false);
}

TEST_P(SharedJitTest, LiveDirectStagesRetainArtifactsAfterCatalogRetirement) {
  CheckLiveStageArtifacts(true);
}

TEST_P(SharedJitTest, LoadSubmissionFailureDoesNotPoisonSharedCompilation) {
  Compilation first, second;
  NativeGate gate(loom_serve_jit_task_sink(jit_));
  first.Start(jit_, gate.sink());
  first.AwaitLayout();
  struct LoadFailure {
    // Real queue for the peer's parent compilation.
    loomc_task_sink_t target;
    // After the parent, exact shared hits submit only independent loads.
    bool parent = false;
  } failure{loom_serve_jit_task_sink(peer_)};
  const loomc_task_sink_t sink = {
      [](void* payload, loomc_task_t* task) {
        auto* failure = static_cast<LoadFailure*>(payload);
        if (!failure->parent) {
          failure->parent = true;
          return loomc_task_sink_submit(failure->target, task);
        }
        return loomc_status_from_iree(
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED));
      },
      &failure};
  second.Start(peer_, sink);
  second.AwaitLayout();
  gate.Open();
  IREE_ASSERT_OK(first.Await());
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, second.Await());
  EXPECT_EQ(compilation_.kernels, 8u);
  EXPECT_EQ(peer_report_.kernels, 0u);
  EXPECT_EQ(loom_serve_jit_cache_storage(peer_).count, 0u);
  Execute(first.stage(), false);
  Compilation retry;
  retry.Start(peer_, loom_serve_jit_task_sink(peer_));
  IREE_ASSERT_OK(retry.Await());
  EXPECT_EQ(peer_report_.kernels, 0u);
  EXPECT_EQ(peer_report_.reused_kernels, 16u);
  Execute(retry.stage(), true);
}

TEST_P(SharedJitTest, SharedFailureReplaysDiagnosticsAndRemainsRetryable) {
  Compilation first, second;
  NativeGate gate(loom_serve_jit_task_sink(jit_));
  first.Start(jit_, gate.sink(), IREE_SV("invalid_native"));
  first.AwaitLayout();
  second.Start(peer_, loom_serve_jit_task_sink(peer_),
               IREE_SV("invalid_native"));
  second.AwaitLayout();
  gate.Open();
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, first.Await());
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, second.Await());
  EXPECT_EQ(compilation_.kernels, 9u);
  EXPECT_EQ(peer_report_.kernels, 0u);
  EXPECT_EQ(peer_report_.reused_kernels, 9u);
  ASSERT_GT(compilation_.diagnostic_count, 0u);
  ASSERT_EQ(peer_report_.diagnostic_count, compilation_.diagnostic_count);
  for (size_t i = 0; i < compilation_.diagnostic_count; ++i) {
    const auto& expected = compilation_.diagnostics[i];
    const auto& actual = peer_report_.diagnostics[i];
    EXPECT_EQ(actual.severity, expected.severity);
    EXPECT_EQ(actual.line, expected.line);
    EXPECT_EQ(actual.column, expected.column);
    EXPECT_EQ(std::string(actual.message, sizeof(actual.message)),
              std::string(expected.message, sizeof(expected.message)));
    EXPECT_EQ(std::string(actual.source, sizeof(actual.source)),
              std::string(expected.source, sizeof(expected.source)));
  }
  Compilation retry;
  retry.Start(peer_, loom_serve_jit_task_sink(peer_),
              IREE_SV("invalid_native"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, retry.Await());
  EXPECT_EQ(peer_report_.kernels, 1u);
  EXPECT_EQ(peer_report_.reused_kernels, 17u);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, SharedJitTest, ::testing::Values(1, 4));

}  // namespace
