// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/stages.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/jit.h"
#include "loom/serve/testing/native_allocator.h"
#include "loomc/iree.h"

namespace {

// Controls only the scheduling dependency. Every accepted task still records
// real commands against independently compiled native programs.
class RecordingSink {
 public:
  ~RecordingSink() {
    while (!tasks.empty()) {
      RetireOne();
    }
  }

  loomc_task_sink_t sink() {
    return {[](void* user_data, loomc_task_t* task) {
              auto* self = static_cast<RecordingSink*>(user_data);
              if (self->submissions++ == self->accept_limit) {
                return loomc_status_from_iree(
                    iree_make_status(IREE_STATUS_UNAVAILABLE, "sink stopped"));
              }
              if (self->inline_execution) {
                loomc_task_execute(task, 0);
              } else {
                self->tasks.push_back(task);
              }
              return loomc_ok_status();
            },
            this};
  }

  void RetireOne() {
    auto* task = tasks.back();
    tasks.pop_back();
    loomc_task_execute(task, 0);
  }

  // Accepted tasks, drained in reverse submission order.
  std::vector<loomc_task_t*> tasks;
  // Number of offers seen, including the first rejected submission.
  size_t submissions = 0;
  // Number of offers to accept before closing the sink.
  size_t accept_limit = SIZE_MAX;
  // Completes accepted work before submit returns instead of queuing it.
  bool inline_execution = false;
};

class TextStagesTest : public loom_serve::testing::JitTest {
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
        nullptr, &compilation_, native_allocator_.allocator(), &jit_));
    for (size_t i = 0; i < stages_.size(); ++i) {
      const std::string delta = std::to_string(i + 1);
      const loomc_config_binding_t binding = {
          loomc_make_cstring_view("increment.delta"),
          loomc_make_cstring_view(delta.c_str())};
      const loomc_config_options_t config = {
          &binding, 1, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
      loom_serve::testing::loom_serve_test_compilation_t compilation;
      compilation.jit = jit_;
      compilation.root = IREE_SV("advance");
      compilation.config = &config;
      compilation.Start(
          loom_serve_host_executor(loom_serve_device_host(device_owner_)));
      IREE_ASSERT_OK(compilation.Await(&stages_[i].compiled));
      stages_[i].program = *loom_serve_jit_stage_program(stages_[i].compiled);
    }
    // Prepared products, not their source catalog, own the executable inputs.
    loom_serve_jit_release(jit_);
    jit_ = nullptr;
  }

  void ReleaseProgram() override {
    for (auto& stage : stages_) {
      iree_hal_command_buffer_release(stage.command);
      loom_serve_jit_stage_free(stage.compiled);
    }
    IREE_EXPECT_OK(status_);
  }

  iree_status_t StartRecording(loomc_task_sink_t sink,
                               iree_host_size_t concurrency,
                               iree_allocator_t allocator) {
    complete_ = false;
    return loom_serve_text_stages_record_async(
        sink, concurrency, iree_hal_queue_family(dispatch_),
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, stages_.size(), stages_.data(),
        [](void* user_data, iree_status_t status) {
          auto* self = static_cast<TextStagesTest*>(user_data);
          std::lock_guard<std::mutex> lock(self->mutex_);
          self->status_ = status;
          ++self->record_completions_;
          self->complete_ = true;
          self->condition_.notify_one();
        },
        this, allocator);
  }

  iree_status_t AwaitRecording() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return complete_; });
    auto status = status_;
    status_ = iree_ok_status();
    return status;
  }

  void CheckRecorded(bool recorded) {
    for (const auto& stage : stages_) {
      EXPECT_EQ(stage.command != nullptr, recorded);
    }
  }

  void Run() {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        loom_serve_pool_storage(storage_owner_),
        iree_hal_make_pool_reservation_request(sizeof(input_), 0),
        iree_infinite_timeout(), &buffer_));
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {input_.data(), buffer_, 0, sizeof(input_)},
    };
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {buffer_, 0, output_.data(), sizeof(output_)},
    };
    const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
    for (size_t i = 0; i < stages_.size(); ++i) {
      ASSERT_NE(stages_[i].command, nullptr);
      uint64_t completion = 0;
      IREE_ASSERT_OK(
          loom_serve_execution_transfer(execution_, 1, &upload, &completion));
      IREE_ASSERT_OK(loom_serve_execution_execute(
          execution_, stages_[i].command, {1, &binding}, &completion));
      IREE_ASSERT_OK(
          loom_serve_execution_feedback(execution_, 1, &download, &completion));
      IREE_ASSERT_OK(
          loom_serve_execution_feedback_wait(execution_, completion));
      for (size_t element = 0; element < output_.size(); ++element) {
        EXPECT_EQ(output_[element], 100 + i + 1 + element);
      }
    }
  }

  // Owns compiled inputs and disjoint recorded outputs until execution drains.
  std::array<loom_serve_text_stage_t, 3> stages_ = {};
  // Allocation dependency for compiler products and recording scratch.
  loom_serve::testing::NativeAllocator native_allocator_;
  // Joins asynchronous completion before the outer test consumes results.
  std::mutex mutex_;
  // Completion notification without wall-clock assumptions.
  std::condition_variable condition_;
  // Final callback has stopped borrowing the job and stage array.
  bool complete_ = false;
  // Owned completion outcome, consumed once by AwaitRecording.
  iree_status_t status_ = iree_ok_status();
  // Number of recording callbacks; completion transfers status exactly once.
  size_t record_completions_ = 0;
};

TEST_P(TextStagesTest, SharedWorkersRecordIndependentPrograms) {
  auto* host = loom_serve_device_host(device_owner_);
  IREE_ASSERT_OK(StartRecording(
      loom_serve_host_preparation_sink(host),
      loomc_task_pool_worker_count(loom_serve_host_task_pool(host)),
      iree_allocator_system()));
  IREE_ASSERT_OK(AwaitRecording());
  EXPECT_EQ(record_completions_, 1u);
  Run();
}

TEST_P(TextStagesTest, RecordingJoinsEveryAcceptedTaskBeforePublication) {
  RecordingSink sink;
  IREE_ASSERT_OK(StartRecording(sink.sink(), 2, iree_allocator_system()));
  ASSERT_EQ(sink.tasks.size(), 2u);
  EXPECT_EQ(record_completions_, 0u);
  sink.RetireOne();
  // One task can record every stage, but publication still waits for the other
  // accepted task to relinquish its borrowed job and model state.
  EXPECT_EQ(record_completions_, 0u);
  sink.RetireOne();
  EXPECT_EQ(record_completions_, 1u);
  IREE_ASSERT_OK(AwaitRecording());
  Run();
}

TEST_P(TextStagesTest, InlineRecordingKeepsSubmissionOwnershipUntilJoined) {
  RecordingSink sink;
  sink.inline_execution = true;
  IREE_ASSERT_OK(StartRecording(sink.sink(), 4, iree_allocator_system()));
  EXPECT_EQ(sink.submissions, 3u);
  EXPECT_EQ(record_completions_, 1u);
  IREE_ASSERT_OK(AwaitRecording());
  Run();
}

TEST_P(TextStagesTest, RecordingRejectionWaitsForAcceptedTasks) {
  for (size_t accepted : {0u, 2u}) {
    RecordingSink sink;
    sink.accept_limit = accepted;
    const size_t prior = record_completions_;
    IREE_ASSERT_OK(StartRecording(sink.sink(), 4, iree_allocator_system()));
    EXPECT_EQ(sink.submissions, accepted + 1);
    ASSERT_EQ(sink.tasks.size(), accepted);
    while (!sink.tasks.empty()) {
      EXPECT_EQ(record_completions_, prior);
      sink.RetireOne();
    }
    EXPECT_EQ(record_completions_, prior + 1);
    IREE_ASSERT_STATUS_IS(IREE_STATUS_UNAVAILABLE, AwaitRecording());
    CheckRecorded(false);
  }
}

TEST_P(TextStagesTest, RecordingRejectionPreservesCompletedCommands) {
  RecordingSink sink;
  sink.inline_execution = true;
  sink.accept_limit = 1;
  IREE_ASSERT_OK(StartRecording(sink.sink(), 4, iree_allocator_system()));
  EXPECT_EQ(sink.submissions, 2u);
  EXPECT_EQ(record_completions_, 1u);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_UNAVAILABLE, AwaitRecording());
  CheckRecorded(true);
}

TEST_P(TextStagesTest, RecordingFailureJoinsOtherAcceptedTasks) {
  RecordingSink sink;
  IREE_ASSERT_OK(StartRecording(sink.sink(), 4, iree_allocator_system()));
  ASSERT_EQ(sink.tasks.size(), 3u);
  native_allocator_.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC);
  sink.RetireOne();
  EXPECT_EQ(record_completions_, 0u);
  sink.RetireOne();
  EXPECT_EQ(record_completions_, 0u);
  sink.RetireOne();
  EXPECT_EQ(record_completions_, 1u);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, AwaitRecording());
  CheckRecorded(false);
}

TEST_P(TextStagesTest, RecordingAllocationRejectionOwesNoCallback) {
  RecordingSink sink;
  IREE_EXPECT_NOT_OK(StartRecording(sink.sink(), 4, iree_allocator_null()));
  EXPECT_EQ(sink.submissions, 0u);
  EXPECT_EQ(record_completions_, 0u);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, TextStagesTest,
                         ::testing::Values(1, 4));

}  // namespace
