// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/jit.h"

#include <atomic>
#include <filesystem>
#include <thread>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/jit.h"
#if defined(__linux__)
#include "loom/serve/testing/stalled_stderr.h"
#endif
#include "loomc/iree.h"

IREE_FLAG(string, invalid_sources, "", "Malformed source catalog.");

namespace {

using loom_serve::testing::JitTest;
using loom_serve::testing::loom_serve_test_compilation_t;

// A real compiler allocation boundary, not a mocked compiler or task sink.
// Holding the first allocations proves distinct parents have entered compiler
// work concurrently before either can publish a native request.
class JitConcurrentTest : public JitTest {
 protected:
  void SetUp() override {
    JitTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    loom_serve_jit_release(jit_);
    jit_ = nullptr;
    const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                            IREE_SV("stage.loom")};
    const loom_serve_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = IREE_ARRAYSIZE(providers),
        .values = providers,
    };
    IREE_ASSERT_OK(loom_serve_jit_create(
        device_, dispatch_, loom_serve_device_host(device_owner_), &sources,
        nullptr, &compilation_, {this, Allocate}, &jit_));
  }

  void HoldParents() {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = GetParam() > 1;
  }

  void JoinParentEntries() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (holding_) {
      condition_.wait(lock, [&] { return entered_ >= 2; });
      holding_ = false;
      condition_.notify_all();
    }
  }

  void CheckConcurrent(bool direct_dispatch);

 private:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* params, void** inout_ptr) {
    auto* test = static_cast<JitConcurrentTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        std::this_thread::get_id() != test->owner_) {
      std::unique_lock<std::mutex> lock(test->mutex_);
      if (test->holding_) {
        ++test->entered_;
        test->condition_.notify_all();
        test->condition_.wait(lock, [&] { return !test->holding_; });
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, params, inout_ptr);
  }

  // Outer test thread is not a compiler participant in the gate.
  std::thread::id owner_ = std::this_thread::get_id();
  // Joins the gate state and final release notification.
  std::mutex mutex_;
  // Compiler allocations wait only while the outer test holds their dependency.
  std::condition_variable condition_;
  // Number of blocked worker entries; a worker cannot enter twice while held.
  size_t entered_ = 0;
  // Armed only on multiworker runs; one-worker coverage exercises join
  // progress.
  bool holding_ = false;
};

void JitConcurrentTest::CheckConcurrent(bool direct_dispatch) {
  const loomc_config_options_t config = {
      nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  struct Submissions {
    // Independent scheduling domain over the catalog's same worker pool.
    loomc_task_sink_t target;
    // Includes parents and their newly published native requests.
    std::atomic<size_t> count = 0;
  } submissions{
      loom_serve_host_preparation_sink(loom_serve_device_host(device_owner_))};
  const loomc_task_sink_t sink = {
      .submit =
          [](void* payload, loomc_task_t* task) {
            auto* submissions = static_cast<Submissions*>(payload);
            ++submissions->count;
            return loomc_task_sink_submit(submissions->target, task);
          },
      .user_data = &submissions,
  };
  struct loom_serve_test_jit_result_t {
    // Joins callback publication before the outer test consumes its output.
    std::mutex mutex;
    // Outer-thread completion notification.
    std::condition_variable condition;
    // Final callback has finished using this record.
    bool complete = false;
    // Owned callback status, consumed after the join.
    iree_status_t status = iree_ok_status();
    // Owned output independent of the source catalog.
    loom_serve_jit_stage_t* stage = nullptr;
  } results[3];
  const iree_string_view_t roots[] = {
      direct_dispatch ? IREE_SV("increment_0") : IREE_SV("advance"),
      direct_dispatch ? IREE_SV("increment_0") : IREE_SV("selected_pair"),
      direct_dispatch ? IREE_SV("absent") : IREE_SV("invalid_argument"),
  };
  HoldParents();
  auto status = iree_ok_status();
  for (size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    const auto completion = [](void* payload, iree_status_t outcome,
                               loom_serve_jit_stage_t* stage) {
      auto* result = static_cast<loom_serve_test_jit_result_t*>(payload);
      std::lock_guard<std::mutex> lock(result->mutex);
      result->status = outcome;
      result->stage = stage;
      result->complete = true;
      result->condition.notify_one();
    };
    auto submitted =
        direct_dispatch
            ? loom_serve_jit_compile_dispatch_async(
                  jit_, sink, roots[i], &config, completion, &results[i])
            : loom_serve_jit_compile_async(jit_, sink, roots[i], &config,
                                           nullptr, completion, &results[i]);
    if (!iree_status_is_ok(submitted)) {
      results[i].complete = true;
    }
    status = iree_status_join(status, submitted);
  }
  JoinParentEntries();
  // Accepted jobs retain the catalog. Its final release cannot discard pending
  // native results, even when one parent fails after publishing a request.
  loom_serve_jit_release(jit_);
  jit_ = nullptr;
  for (size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    auto& result = results[i];
    std::unique_lock<std::mutex> lock(result.mutex);
    result.condition.wait(lock, [&] { return result.complete; });
    if (i == 2) {
      IREE_EXPECT_STATUS_IS(direct_dispatch ? IREE_STATUS_NOT_FOUND
                                            : IREE_STATUS_INVALID_ARGUMENT,
                            result.status);
      EXPECT_EQ(result.stage, nullptr);
    } else {
      status = iree_status_join(status, result.status);
      if (result.stage && direct_dispatch) {
        dispatches_[i] = loom_serve_jit_stage_dispatch(result.stage);
        loom_serve_dispatch_retain(dispatches_[i]);
      } else if (result.stage) {
        status = iree_status_join(
            status, loom_serve_jit_stage_record(
                        result.stage, iree_hal_queue_family(dispatch_),
                        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, nullptr,
                        &commands_[i]));
      }
    }
    loom_serve_jit_stage_free(result.stage);
  }
  IREE_ASSERT_OK(status);
  EXPECT_EQ(compilation_.commands, direct_dispatch ? 0u : 3u);
  EXPECT_EQ(compilation_.kernels, direct_dispatch ? 1u : 8u);
  // Direct roots compile in the submitted task; command parents publish
  // independent native tasks, which must inherit the caller's sink too.
  EXPECT_EQ(
      submissions.count,
      IREE_ARRAYSIZE(results) + (direct_dispatch ? 0 : compilation_.kernels));
  EXPECT_EQ(compilation_.reused_kernels, direct_dispatch ? 1u : 3u);
  EXPECT_EQ(compilation_.entries, direct_dispatch ? 2u : 10u);
  if (direct_dispatch) {
    EXPECT_EQ(dispatches_[0], dispatches_[1]);
  }
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), buffer_, 0, sizeof(input_)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  for (size_t i = 0; i < commands_.size(); ++i) {
    if (direct_dispatch) {
      iree_hal_buffer_ref_t scratch;
      IREE_ASSERT_OK(loom_serve_dispatch_issue(
          dispatches_[i], execution_, {1, &binding}, &scratch, &completion));
    } else {
      IREE_ASSERT_OK(loom_serve_execution_execute(execution_, commands_[i],
                                                  {1, &binding}, &completion));
    }
  }
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], direct_dispatch
                              ? (i == 0 ? 110 : 100)
                              : 105 + i + (i == 7 ? 24 : (i == 2 ? 7 : 0)));
  }
}

TEST_P(JitConcurrentTest, SharedParentsJoinNativesAndRetainExecutableOutput) {
  CheckConcurrent(false);
}

TEST_P(JitConcurrentTest, DirectStagesJoinNativesAndRetainExecutableOutput) {
  CheckConcurrent(true);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitConcurrentTest,
                         ::testing::Values(1, 4));

TEST_P(JitTest, LogicalScalarCommandArgumentsReachNativeKernels) {
  IREE_ASSERT_OK(Compile(nullptr, 0, IREE_SV("boolean_arguments")));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), buffer_, 0, sizeof(input_)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  IREE_ASSERT_OK(loom_serve_execution_execute(execution_, commands_[0],
                                              {1, &binding}, &completion));
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  EXPECT_EQ(output_[0], 17);
  EXPECT_EQ(output_[1], -23);
  for (size_t i = 2; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], input_[i]);
  }
}

class JitCacheTest : public JitTest {
 protected:
  void ReleaseProgram() override {
    for (auto* stage : stages_) {
      loom_serve_jit_stage_free(stage);
    }
  }

  void CheckPruning(bool direct_dispatch) {
    const loomc_config_options_t config = {
        nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
    auto compile = [&](iree_string_view_t root, size_t index) {
      loom_serve_test_compilation_t job;
      job.jit = jit_;
      job.direct_dispatch = direct_dispatch;
      job.root = root;
      job.config = &config;
      job.Start(
          loom_serve_host_executor(loom_serve_device_host(device_owner_)));
      return job.Await(&stages_[index]);
    };
    const auto first_root =
        direct_dispatch ? IREE_SV("increment_0") : IREE_SV("advance");
    const auto second_root =
        direct_dispatch ? IREE_SV("increment_0") : IREE_SV("selected_pair");
    IREE_ASSERT_OK(compile(first_root, 0));
    const auto initial = loom_serve_jit_cache_storage(jit_);
    EXPECT_EQ(initial.count, direct_dispatch ? 1u : 8u);
    EXPECT_GT(initial.host_bytes, 0u);
    // These clean, small programs retain source keys and native ownership,
    // not a full empty diagnostic report for every cached kernel.
    EXPECT_LT(initial.host_bytes - initial.code_bytes,
              initial.count * sizeof(loom_serve_compilation_report_t));
    if (direct_dispatch) {
      // One small linked kernel, including its key and ownership records,
      // fits in 3 KiB without doubling the key for a trailing root name.
      EXPECT_LT(initial.host_bytes - initial.code_bytes, 3u * 1024u);
    }
    EXPECT_GT(initial.code_bytes, 0u);
    EXPECT_GT(loom_serve_jit_stage_storage_size(stages_[0]), 0u);
    loom_serve_jit_trim(jit_);
    const auto retained = loom_serve_jit_cache_storage(jit_);
    EXPECT_EQ(retained.count, initial.count);
    EXPECT_EQ(retained.host_bytes, initial.host_bytes);
    EXPECT_EQ(retained.code_bytes, initial.code_bytes);

    IREE_ASSERT_OK(compile(second_root, 1));
    EXPECT_EQ(compilation_.kernels, initial.count);
    EXPECT_EQ(compilation_.reused_kernels, direct_dispatch ? 1u : 2u);
    for (size_t i = 0; i < stages_.size(); ++i) {
      if (direct_dispatch) {
        dispatches_[i] = loom_serve_jit_stage_dispatch(stages_[i]);
        loom_serve_dispatch_retain(dispatches_[i]);
      } else {
        IREE_ASSERT_OK(loom_serve_jit_stage_record(
            stages_[i], iree_hal_queue_family(dispatch_),
            IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, nullptr,
            &commands_[i]));
      }
    }
    loom_serve_jit_stage_free(stages_[0]);
    stages_[0] = nullptr;
    loom_serve_jit_trim(jit_);
    EXPECT_EQ(loom_serve_jit_cache_storage(jit_).count, initial.count);
    loom_serve_jit_stage_free(stages_[1]);
    stages_[1] = nullptr;
    loom_serve_jit_trim(jit_);
    EXPECT_EQ(loom_serve_jit_cache_storage(jit_).count, initial.count);

    // Weight preparation retains recordings, not compiler stages. Another
    // checkpoint with the same preparation code must reuse those live natives.
    IREE_ASSERT_OK(compile(first_root, 0));
    EXPECT_EQ(compilation_.kernels, initial.count);
    EXPECT_EQ(compilation_.reused_kernels,
              initial.count + (direct_dispatch ? 1u : 2u));
    loom_serve_jit_stage_free(stages_[0]);
    stages_[0] = nullptr;

    // The independently retained consumers still execute after every compiled
    // stage has retired.
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(device_owner_),
        iree_hal_make_pool_reservation_request(sizeof(input_), 0),
        iree_infinite_timeout(), &buffer_));
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {input_.data(), buffer_, 0, sizeof(input_)},
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution_, 1, &upload, &completion));
    const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
    for (size_t i = 0; i < stages_.size(); ++i) {
      if (direct_dispatch) {
        iree_hal_buffer_ref_t scratch;
        IREE_ASSERT_OK(loom_serve_dispatch_issue(
            dispatches_[i], execution_, {1, &binding}, &scratch, &completion));
      } else {
        IREE_ASSERT_OK(loom_serve_execution_execute(
            execution_, commands_[i], {1, &binding}, &completion));
      }
    }
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {buffer_, 0, output_.data(), sizeof(output_)},
    };
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution_, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
    for (size_t i = 0; i < output_.size(); ++i) {
      EXPECT_EQ(output_[i], direct_dispatch
                                ? (i == 0 ? 110 : 100)
                                : 105 + i + (i == 7 ? 24 : (i == 2 ? 7 : 0)));
    }

    for (size_t i = 0; i < stages_.size(); ++i) {
      iree_hal_command_buffer_release(commands_[i]);
      commands_[i] = nullptr;
      loom_serve_dispatch_release(dispatches_[i]);
      dispatches_[i] = nullptr;
      loom_serve_jit_trim(jit_);
      const auto remaining = loom_serve_jit_cache_storage(jit_);
      if (i == 0) {
        EXPECT_EQ(remaining.count, direct_dispatch ? 1u : 2u);
        if (!direct_dispatch) {
          EXPECT_LT(remaining.host_bytes, initial.host_bytes);
          EXPECT_LT(remaining.code_bytes, initial.code_bytes);
        }
      } else {
        EXPECT_EQ(remaining.count, 0u);
        EXPECT_EQ(remaining.code_bytes, 0u);
        EXPECT_LT(remaining.host_bytes, initial.host_bytes);
      }
    }

    // Eviction preserves the immutable source index, so a later miss can
    // reconstruct the same native inputs without reopening provider files.
    IREE_ASSERT_OK(compile(first_root, 0));
    EXPECT_EQ(compilation_.kernels, initial.count * 2);
    EXPECT_EQ(loom_serve_jit_cache_storage(jit_).count, initial.count);
  }

  void CheckFailedDiagnostics(bool direct_dispatch) {
    const loomc_config_options_t config = {
        nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
    for (size_t attempt = 0; attempt < 2; ++attempt) {
      SCOPED_TRACE(attempt);
      compilation_ = {};
      loom_serve_test_compilation_t job;
      job.jit = jit_;
      job.direct_dispatch = direct_dispatch;
      job.root = direct_dispatch ? IREE_SV("requires_host")
                                 : IREE_SV("invalid_native");
      job.config = &config;
      job.Start(
          loom_serve_host_executor(loom_serve_device_host(device_owner_)));
      IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                            job.Await(&stages_[0]));
      if (direct_dispatch) {
        EXPECT_EQ(stages_[0], nullptr);
        EXPECT_EQ(job.layout, nullptr);
      } else {
        ASSERT_NE(stages_[0], nullptr);
        ASSERT_NE(job.layout, nullptr);
        EXPECT_EQ(job.layout, loom_serve_jit_stage_program(stages_[0]));
        EXPECT_EQ(job.layout->requirements.rebindable_binding_count, 1u);
      }
      EXPECT_GT(compilation_.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
      ASSERT_GT(compilation_.diagnostic_count, 0u);
      EXPECT_GT(compilation_.diagnostics[0].message_length, 0u);
      // Failed native evidence survives task retirement, but failed inputs
      // are not cached. Successful siblings remain available to the retry.
      EXPECT_EQ(compilation_.kernels, direct_dispatch || attempt ? 1u : 9u);
      EXPECT_EQ(compilation_.reused_kernels,
                !direct_dispatch && attempt ? 8u : 0u);
      EXPECT_EQ(loom_serve_jit_cache_storage(jit_).count,
                direct_dispatch ? 0u : 8u);
      loom_serve_jit_stage_free(stages_[0]);
      stages_[0] = nullptr;
    }
    const auto diagnostic = compilation_.diagnostics[0];
    loom_serve_jit_release(jit_);
    jit_ = nullptr;
    EXPECT_EQ(compilation_.diagnostics[0].message_length,
              diagnostic.message_length);
    EXPECT_EQ(std::string(compilation_.diagnostics[0].message,
                          sizeof(diagnostic.message)),
              std::string(diagnostic.message, sizeof(diagnostic.message)));
  }

 private:
  // Compiled-stage borrows are separate from recorded command consumers.
  std::array<loom_serve_jit_stage_t*, 2> stages_ = {};
};

TEST_P(JitCacheTest, CommandStagesPreserveSharedNativeReuseDuringEviction) {
  CheckPruning(false);
}

TEST_P(JitCacheTest, DirectStagesPreserveSharedNativeReuseDuringEviction) {
  CheckPruning(true);
}

TEST_P(JitCacheTest, FailedCommandNativesRetainDiagnosticsWithoutCaching) {
  CheckFailedDiagnostics(false);
}

TEST_P(JitCacheTest, FailedDirectNativesRetainDiagnosticsWithoutCaching) {
  CheckFailedDiagnostics(true);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitCacheTest, ::testing::Values(1, 4));

TEST_P(JitTest, SourceFailureOutlivesUnconstructedCatalogAndVm) {
#if defined(__linux__)
  loom_serve::testing::StalledStderr blocked;
  ASSERT_NO_FATAL_FAILURE(blocked.Begin());
#endif
  const std::string directory =
      std::filesystem::path(FLAG_invalid_sources).parent_path().string();
  loom_serve_jit_t* invalid = nullptr;
  loom_serve_compilation_report_t report = {};
  const iree_string_view_t providers[] = {IREE_SV("invalid.loom")};
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(directory.c_str()),
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
  };
  auto status = loom_serve_jit_create(
      device_, dispatch_, loom_serve_device_host(device_owner_), &sources,
      nullptr, &report, host_allocator_, &invalid);
  EXPECT_NE(iree_status_code(status), IREE_STATUS_OK);
  EXPECT_EQ(invalid, nullptr);
  EXPECT_GT(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  ASSERT_GT(report.diagnostic_count, 0u);
  EXPECT_GT(report.diagnostics[0].line, 0u);
  EXPECT_GT(report.diagnostics[0].source_length, 0u);
  const auto message = iree_status_message(status);
  EXPECT_NE(std::string(message.data, message.size).find("invalid.loom"),
            std::string::npos);
  iree_status_free(status);
  loom_serve_jit_release(invalid);

  const iree_string_view_t root = IREE_SVL("broken");
  // Both compiler entry paths publish owned diagnostics without writing to a
  // potentially blocked stderr sink.
  iree_const_byte_span_t image = iree_const_byte_span_empty();
  status = loom_serve_jit_compile_vm(
      loom_serve_host_target_environment(loom_serve_device_host(device_owner_)),
      &sources, IREE_SV("invalid.loom"), 1, &root, &report, host_allocator_,
      &image);
  IREE_EXPECT_NOT_OK(status);
  EXPECT_EQ(image.data, nullptr);
  iree_allocator_free(host_allocator_, const_cast<uint8_t*>(image.data));
  EXPECT_GE(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 2u);
  EXPECT_EQ(report.commands, 0u);
}

TEST_P(JitTest, CompletedCatalogsRetireWithoutClosingSharedCompilerQueue) {
  // The first catalog is warm; the independent peer must compile its own copy.
  IREE_ASSERT_OK(Compile(nullptr, 0));
  iree_hal_command_buffer_release(commands_[0]);
  commands_[0] = nullptr;
  loom_serve_compilation_report_t peer_report = {};
  loom_serve_jit_t* peer = nullptr;
  auto* host = loom_serve_device_host(device_owner_);
  const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                          IREE_SV("stage.loom")};
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(directory_.c_str()),
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
  };
  IREE_ASSERT_OK(loom_serve_jit_create(device_, dispatch_, host, &sources,
                                       nullptr, &peer_report, host_allocator_,
                                       &peer));
  const loomc_config_options_t config = {
      nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  loom_serve_test_compilation_t jobs[2];
  jobs[0].jit = jit_;
  jobs[0].root = IREE_SV("selected_pair");
  jobs[1].jit = peer;
  jobs[1].root = IREE_SV("advance");
  // Both catalogs are now owned through their worker completion callbacks.
  jit_ = nullptr;
  for (auto& job : jobs) {
    job.retire_catalog = true;
    job.config = &config;
    job.Start(loom_serve_host_executor(host));
  }
  auto status = iree_ok_status();
  for (size_t i = 0; i < IREE_ARRAYSIZE(jobs); ++i) {
    loom_serve_jit_stage_t* stage = nullptr;
    auto outcome = jobs[i].Await(&stage);
    EXPECT_EQ(jobs[i].jit, nullptr);
    if (iree_status_is_ok(outcome)) {
      outcome =
          loom_serve_jit_stage_record(stage, iree_hal_queue_family(dispatch_),
                                      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                      nullptr, nullptr, &commands_[i]);
    }
    status = iree_status_join(status, outcome);
    loom_serve_jit_stage_free(stage);
  }
  IREE_ASSERT_OK(status);
  EXPECT_EQ(compilation_.entries, 10u);
  EXPECT_EQ(compilation_.kernels, 8u);
  EXPECT_EQ(compilation_.reused_kernels, 2u);
  EXPECT_EQ(peer_report.entries, 8u);
  EXPECT_EQ(peer_report.kernels, 8u);
  EXPECT_EQ(peer_report.reused_kernels, 0u);
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), buffer_, 0, sizeof(input_)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  for (auto* command : commands_) {
    IREE_ASSERT_OK(loom_serve_execution_execute(execution_, command,
                                                {1, &binding}, &completion));
  }
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], 105 + i + (i == 7 ? 24 : (i == 2 ? 7 : 0)));
  }
}

TEST_P(JitTest, InvalidConfigurationLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Compile("99", 0));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_P(JitTest, FailedParentRetiresCatalogAfterAcceptedNativeTasks) {
  const loomc_config_options_t config = {
      nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  loom_serve_test_compilation_t job;
  job.jit = jit_;
  job.root = IREE_SV("invalid_argument");
  job.config = &config;
  job.retire_catalog = true;
  jit_ = nullptr;
  job.Start(loom_serve_host_executor(loom_serve_device_host(device_owner_)));
  loom_serve_jit_stage_t* stage = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, job.Await(&stage));
  EXPECT_EQ(stage, nullptr);
  EXPECT_EQ(job.jit, nullptr);
  EXPECT_EQ(compilation_.commands, 1u);
  EXPECT_EQ(compilation_.kernels, 1u);
  EXPECT_EQ(compilation_.entries, 0u);
  EXPECT_GT(compilation_.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  loom_serve_jit_stage_free(stage);
}

TEST_P(JitTest, InvalidRootLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("absent")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("missing")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        Compile("3", 0, IREE_SV("ambiguous")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("3", 0, IREE_SV("not_command")));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_P(JitTest, CommandWithoutNativeRequests) {
  IREE_ASSERT_OK(Compile("3", 0, IREE_SV("idle")));
}

class JitReadinessTest : public JitTest {};

TEST_P(JitReadinessTest, IndependentCompletionDoesNotWaitForEarlierProduct) {
  struct DeferredNatives {
    // Shared executor preserving the catalog's worker ordinals.
    loomc_task_sink_t target;
    // The outer caller submits one parent before it can publish natives.
    bool parent_submitted = false;
    // Native tasks appended by that parent before its layout callback.
    loomc_task_t* head = nullptr;
    // Final queued task; ownership transfers after layout publication.
    loomc_task_t* tail = nullptr;
  } deferred{.target = loom_serve_jit_task_sink(jit_)};
  const loomc_task_sink_t sink = {
      .submit =
          [](void* payload, loomc_task_t* task) {
            auto* deferred = static_cast<DeferredNatives*>(payload);
            if (!deferred->parent_submitted) {
              deferred->parent_submitted = true;
              return loomc_task_sink_submit(deferred->target, task);
            }
            if (deferred->tail) {
              deferred->tail->next = task;
            } else {
              deferred->head = task;
            }
            deferred->tail = task;
            return loomc_ok_status();
          },
      .user_data = &deferred,
  };
  struct Completion {
    // Serializes the outer observer with both producer callbacks.
    std::mutex mutex;
    // Explicit dependency events; no timing-based readiness assumptions.
    std::condition_variable condition;
    // Borrowed immutable bytes owned by the eventual stage.
    const loom_cmd_program_t* program = nullptr;
    // Terminal publication relinquished this record.
    bool complete = false;
    // Owned terminal status, consumed by the outer observer.
    iree_status_t status = iree_ok_status();
    // Owned partial or final result, independent of catalog lifetime.
    loom_serve_jit_stage_t* stage = nullptr;
  } first, second;
  const auto completed = [](void* payload, iree_status_t status,
                            loom_serve_jit_stage_t* stage) {
    auto* result = static_cast<Completion*>(payload);
    std::lock_guard<std::mutex> lock(result->mutex);
    result->status = status;
    result->stage = stage;
    result->complete = true;
    result->condition.notify_one();
  };
  const loomc_config_options_t config = {
      nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  // Publish the real command layout while its native tasks remain queued.
  // Neither the compiler nor its layout callback occupies a waiting worker.
  IREE_ASSERT_OK(loom_serve_jit_compile_async(
      jit_, sink, IREE_SV("advance"), &config,
      [](void* payload, const loom_cmd_program_t* program) {
        auto* result = static_cast<Completion*>(payload);
        std::lock_guard<std::mutex> lock(result->mutex);
        result->program = program;
        result->condition.notify_one();
      },
      completed, &first));
  {
    std::unique_lock<std::mutex> lock(first.mutex);
    first.condition.wait(lock, [&] { return first.program != nullptr; });
  }
  auto status = loom_serve_jit_compile_async(
      jit_, loom_serve_jit_task_sink(jit_), IREE_SV("idle"), &config, nullptr,
      completed, &second);
  if (iree_status_is_ok(status)) {
    std::unique_lock<std::mutex> lock(second.mutex);
    second.condition.wait(lock, [&] { return second.complete; });
    status = second.status;
  }
  // Snapshot while the first compilation is still outstanding. Reentrant
  // publication and report access must not require a whole-catalog join.
  loom_serve_compilation_report_t snapshot = {};
  loom_serve_jit_report_snapshot(jit_, &snapshot);
  EXPECT_EQ(snapshot.commands, 1u);
  {
    std::lock_guard<std::mutex> lock(first.mutex);
    EXPECT_FALSE(first.complete);
  }
  EXPECT_NE(deferred.head, nullptr);
  while (deferred.head) {
    auto* task = deferred.head;
    deferred.head = task->next;
    task->next = nullptr;
    IREE_ASSERT_OK(
        iree_status_from_loomc(loomc_task_sink_submit(deferred.target, task)));
  }
  {
    std::unique_lock<std::mutex> lock(first.mutex);
    first.condition.wait(lock, [&] { return first.complete; });
  }
  EXPECT_EQ(first.program, loom_serve_jit_stage_program(first.stage));
  loom_serve_jit_stage_free(first.stage);
  loom_serve_jit_stage_free(second.stage);
  IREE_ASSERT_OK(iree_status_join(status, first.status));
  EXPECT_EQ(compilation_.commands, 2u);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitReadinessTest, ::testing::Values(1));

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitTest, ::testing::Values(1, 4));

}  // namespace
