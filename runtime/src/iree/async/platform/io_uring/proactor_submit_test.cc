// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/eventfd.h>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/io_uring/api.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

extern "C" iree_async_operation_t*
iree_async_proactor_io_uring_build_software_submission_list(
    iree_async_operation_list_t operations);

namespace {

struct CompletionState {
  int call_count = 0;
  std::vector<iree_status_code_t> status_codes;

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    auto* state = static_cast<CompletionState*>(user_data);
    (void)operation;
    (void)flags;
    ++state->call_count;
    state->status_codes.push_back(iree_status_code(status));
    iree_status_free(status);
  }
};

struct ReuseCompletionState {
  iree_host_size_t call_count = 0;

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    (void)flags;
    auto* state = static_cast<ReuseCompletionState*>(user_data);
    ++state->call_count;
    iree_status_free(status);
    iree_async_operation_initialize(operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                                    IREE_ASYNC_OPERATION_FLAG_NONE, Callback,
                                    state);
  }
};

TEST(IoUringSubmissionPlanTest, SurvivesPublishedOperationReuse) {
  ReuseCompletionState completion;
  iree_async_operation_t storage[7] = {};
  const iree_async_operation_type_t types[] = {
      IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV,
      IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV,
      IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV,
      IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV,
      IREE_ASYNC_OPERATION_TYPE_NOP,
  };
  iree_async_operation_t* operations[IREE_ARRAYSIZE(storage)] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(storage); ++i) {
    iree_async_operation_flags_t flags = (i == 0 || i == 2 || i == 4)
                                             ? IREE_ASYNC_OPERATION_FLAG_LINKED
                                             : IREE_ASYNC_OPERATION_FLAG_NONE;
    iree_async_operation_initialize(&storage[i], types[i], flags,
                                    ReuseCompletionState::Callback,
                                    &completion);
    operations[i] = &storage[i];
  }

  iree_async_operation_t* software_operation =
      iree_async_proactor_io_uring_build_software_submission_list(
          iree_async_operation_list_make(operations,
                                         IREE_ARRAYSIZE(operations)));
  EXPECT_EQ(software_operation, &storage[2]);

  // Kernel submission and a concurrent software completion may reuse earlier
  // operations after the plan is built. Traversal must depend only on the
  // captured intrusive links.
  storage[0].completion_fn(storage[0].user_data, &storage[0], iree_ok_status(),
                           IREE_ASYNC_COMPLETION_FLAG_NONE);
  iree_async_operation_t* next_software_operation = software_operation->next;
  software_operation->next = nullptr;
  software_operation->completion_fn(software_operation->user_data,
                                    software_operation, iree_ok_status(),
                                    IREE_ASYNC_COMPLETION_FLAG_NONE);
  EXPECT_EQ(next_software_operation, &storage[6]);

  software_operation = next_software_operation;
  next_software_operation = software_operation->next;
  software_operation->next = nullptr;
  software_operation->completion_fn(software_operation->user_data,
                                    software_operation, iree_ok_status(),
                                    IREE_ASYNC_COMPLETION_FLAG_NONE);
  EXPECT_EQ(next_software_operation, nullptr);
  EXPECT_EQ(completion.call_count, 3u);
}

class IoUringSubmitTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kSubmissionQueueEntries = 8;

  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    options.max_concurrent_operations = kSubmissionQueueEntries;
    options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
    options.allowed_capabilities &=
        ~IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING;
    iree_status_t status = iree_async_proactor_create_io_uring(
        options, iree_allocator_system(), &proactor_);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "io_uring is unavailable";
    }
    IREE_ASSERT_OK(status);
  }

  void TearDown() override {
    if (thread_) {
      iree_async_proactor_thread_request_stop(thread_);
      IREE_EXPECT_OK(
          iree_async_proactor_thread_join(thread_, IREE_DURATION_INFINITE));
      IREE_EXPECT_OK(iree_async_proactor_thread_consume_status(thread_));
      iree_async_proactor_thread_release(thread_);
    }
    DrainFillers();
    iree_async_proactor_release(target_proactor_);
    iree_async_proactor_release(proactor_);
  }

  void CreateTarget() {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    options.allowed_capabilities &=
        ~IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING;
    IREE_ASSERT_OK(iree_async_proactor_create_io_uring(
        options, iree_allocator_system(), &target_proactor_));
  }

  template <typename Predicate>
  void PollUntil(iree_async_proactor_t* proactor, Predicate condition) {
    while (!condition()) {
      iree_host_size_t completed_count = 0;
      IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                              &completed_count));
    }
  }

  void PollImmediate(iree_async_proactor_t* proactor) {
    iree_host_size_t completed_count = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor, iree_immediate_timeout(), &completed_count);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_free(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  void FillSubmissionQueue() {
    filler_timers_.resize(kSubmissionQueueEntries);
    filler_operations_.resize(kSubmissionQueueEntries);
    filler_completion_.call_count = 0;
    filler_completion_.status_codes.clear();
    for (uint32_t i = 0; i < kSubmissionQueueEntries; ++i) {
      iree_async_timer_operation_t& timer = filler_timers_[i];
      timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
      timer.base.completion_fn = CompletionState::Callback;
      timer.base.user_data = &filler_completion_;
      timer.deadline_ns = iree_time_now();
      filler_operations_[i] = &timer.base;
    }
    IREE_ASSERT_OK(iree_async_proactor_submit(
        proactor_, iree_async_operation_list_make(filler_operations_.data(),
                                                  filler_operations_.size())));
    filler_count_ = kSubmissionQueueEntries;
  }

  void DrainFillers() {
    if (!proactor_ || filler_count_ == 0) {
      return;
    }
    PollUntil(proactor_, [&] {
      return filler_completion_.call_count == static_cast<int>(filler_count_);
    });
    filler_count_ = 0;
    filler_operations_.clear();
    filler_timers_.clear();
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_proactor_t* target_proactor_ = nullptr;
  // Optional infinite-wait runner for cross-thread submission coverage.
  iree_async_proactor_thread_t* thread_ = nullptr;
  std::vector<iree_async_timer_operation_t> filler_timers_;
  std::vector<iree_async_operation_t*> filler_operations_;
  CompletionState filler_completion_;
  uint32_t filler_count_ = 0;
};

TEST_F(IoUringSubmitTest, CrossThreadNopDoesNotLoseIdleWake) {
  IREE_ASSERT_OK(iree_async_proactor_thread_create(
      proactor_, iree_async_proactor_thread_options_default(),
      iree_allocator_system(), &thread_));
  struct State {
    // Protects publication of completed callbacks to the submitting task.
    std::mutex mutex;
    // Joins actual completion without periodically waking the proactor.
    std::condition_variable condition;
    // Number of operations returned to the submitting task.
    int completed_count = 0;
  } state;
  iree_async_nop_operation_t operation = {};
  iree_async_operation_initialize(
      &operation.base, IREE_ASYNC_OPERATION_TYPE_NOP, 0,
      [](void* user_data, iree_async_operation_t*, iree_status_t status,
         iree_async_completion_flags_t flags) {
        IREE_EXPECT_OK(status);
        EXPECT_EQ(flags, IREE_ASYNC_COMPLETION_FLAG_NONE);
        auto* state = static_cast<State*>(user_data);
        std::lock_guard<std::mutex> lock(state->mutex);
        ++state->completed_count;
        state->condition.notify_one();
      },
      &state);

  // Resubmit as soon as ownership returns, racing the poller's final software
  // drain and transition to idle. No native operation can incidentally wake a
  // lost NOP, and the normal runner never polls on a timer.
  for (int i = 0; i < 10000; ++i) {
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &operation.base));
    std::unique_lock<std::mutex> lock(state.mutex);
    state.condition.wait(lock, [&] { return state.completed_count == i + 1; });
  }
}

TEST_F(IoUringSubmitTest, FullSqRejectsSequenceWithoutStartingIt) {
  FillSubmissionQueue();

  CompletionState sequence_completion;
  iree_async_sequence_operation_t sequence = {};
  sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
  sequence.base.completion_fn = CompletionState::Callback;
  sequence.base.user_data = &sequence_completion;

  CompletionState timer_completion;
  iree_async_timer_operation_t timer = {};
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer.base.completion_fn = CompletionState::Callback;
  timer.base.user_data = &timer_completion;
  timer.deadline_ns = iree_time_now();

  iree_async_operation_t* operations[] = {&sequence.base, &timer.base};
  iree_async_operation_list_t operation_list =
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_async_proactor_submit(proactor_, operation_list));
  EXPECT_EQ(sequence_completion.call_count, 0);
  EXPECT_EQ(timer_completion.call_count, 0);

  DrainFillers();
  EXPECT_EQ(sequence_completion.call_count, 0);
  EXPECT_EQ(timer_completion.call_count, 0);

  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, operation_list));
  PollUntil(proactor_, [&] {
    return sequence_completion.call_count == 1 &&
           timer_completion.call_count == 1;
  });
  EXPECT_EQ(sequence_completion.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
  EXPECT_EQ(timer_completion.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
}

TEST_F(IoUringSubmitTest, FileSequenceProgressesFullSubmissionQueue) {
  iree::testing::TempFilePath path("iree_async_sequence_admission");
  const uint8_t expected[] = {0x17, 0x83, 0x49, 0xBA, 0x25, 0xFC, 0x61, 0xD2};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(expected, sizeof(expected)),
      iree_allocator_system()));

  struct State {
    // Runs on the poll owner after open and before the dependent read.
    std::function<void()> before_read;
    // Terminal status of the accepted sequence.
    CompletionState completion;
  } state;
  state.before_read = [&] {
    // Hold the poll owner in this callback while an external producer fills
    // the SQ. The sequence's next step must progress that prepared work,
    // without dispatching its callbacks recursively, to admit the read.
    std::thread producer([&] { FillSubmissionQueue(); });
    producer.join();
    EXPECT_EQ(filler_completion_.call_count, 0);
  };

  iree_async_file_open_operation_t open = {};
  open.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open.path = path.path().c_str();
  open.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  uint8_t actual[sizeof(expected)] = {};
  iree_async_file_read_operation_t read = {};
  read.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read.buffer = iree_async_span_from_ptr(actual, sizeof(actual));
  iree_async_operation_t* steps[] = {&open.base, &read.base};
  iree_async_sequence_operation_t sequence = {};
  iree_async_operation_initialize(
      &sequence.base, IREE_ASYNC_OPERATION_TYPE_SEQUENCE,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      [](void* user_data, iree_async_operation_t* operation,
         iree_status_t status, iree_async_completion_flags_t flags) {
        auto* state = static_cast<State*>(user_data);
        CompletionState::Callback(&state->completion, operation, status, flags);
      },
      &state);
  sequence.steps = steps;
  sequence.step_count = IREE_ARRAYSIZE(steps);
  sequence.step_fn = [](void* user_data, iree_async_operation_t* completed_step,
                        iree_async_operation_t* next_step) {
    if (next_step) {
      auto* state = static_cast<State*>(user_data);
      auto* open =
          reinterpret_cast<iree_async_file_open_operation_t*>(completed_step);
      auto* read =
          reinterpret_cast<iree_async_file_read_operation_t*>(next_step);
      read->file = open->opened_file;
      state->before_read();
    }
    return iree_ok_status();
  };
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));
  PollUntil(proactor_, [&] { return state.completion.call_count == 1; });
  EXPECT_EQ(state.completion.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
  EXPECT_EQ(sequence.current_step, IREE_ARRAYSIZE(steps));
  EXPECT_EQ(read.bytes_read, sizeof(expected));
  EXPECT_EQ(std::vector<uint8_t>(actual, actual + sizeof(actual)),
            std::vector<uint8_t>(expected, expected + sizeof(expected)));
  DrainFillers();
  iree_async_file_release(open.opened_file);
  EXPECT_TRUE(path.Remove());
}

TEST_F(IoUringSubmitTest, OwnerCallbackAdmitsOnlyBatchesThatFitRing) {
  for (uint32_t request_count :
       {1u, kSubmissionQueueEntries, kSubmissionQueueEntries + 1}) {
    SCOPED_TRACE(request_count);
    CompletionState completion;
    std::vector<iree_async_timer_operation_t> timers(request_count);
    std::vector<iree_async_operation_t*> operations(request_count);
    for (uint32_t i = 0; i < request_count; ++i) {
      iree_async_operation_initialize(&timers[i].base,
                                      IREE_ASYNC_OPERATION_TYPE_TIMER,
                                      IREE_ASYNC_OPERATION_FLAG_NONE,
                                      CompletionState::Callback, &completion);
      timers[i].deadline_ns = iree_time_now();
      operations[i] = &timers[i].base;
    }
    bool attempted = false;
    iree_status_code_t submit_code = IREE_STATUS_UNKNOWN;
    std::function<void()> submit = [&] {
      std::thread producer([&] { FillSubmissionQueue(); });
      producer.join();
      iree_status_t status = iree_async_proactor_submit(
          proactor_,
          iree_async_operation_list_make(operations.data(), operations.size()));
      submit_code = iree_status_code(status);
      iree_status_free(status);
      EXPECT_EQ(completion.call_count, 0);
      EXPECT_EQ(filler_completion_.call_count, 0);
      attempted = true;
    };
    iree_async_nop_operation_t callback = {};
    iree_async_operation_initialize(
        &callback.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        [](void* user_data, iree_async_operation_t*, iree_status_t status,
           iree_async_completion_flags_t) {
          IREE_EXPECT_OK(status);
          (*static_cast<std::function<void()>*>(user_data))();
        },
        &submit);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &callback.base));
    PollUntil(proactor_, [&] { return attempted; });
    EXPECT_EQ(submit_code, request_count <= kSubmissionQueueEntries
                               ? IREE_STATUS_OK
                               : IREE_STATUS_RESOURCE_EXHAUSTED);
    if (submit_code == IREE_STATUS_OK) {
      PollUntil(proactor_, [&] {
        return completion.call_count == static_cast<int>(request_count);
      });
      EXPECT_EQ(completion.status_codes,
                std::vector<iree_status_code_t>(request_count, IREE_STATUS_OK));
    }
    DrainFillers();
  }
}

TEST_F(IoUringSubmitTest, FailedPredecessorDoesNotPublishNotification) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  const uint32_t initial_epoch =
      iree_async_notification_query_epoch(notification);
  CompletionState completion;
  iree_async_file_open_operation_t open = {};
  iree_async_operation_initialize(
      &open.base, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN,
      IREE_ASYNC_OPERATION_FLAG_LINKED, CompletionState::Callback, &completion);
  // The descriptor path names a file, not a directory. Admission succeeds but
  // the native open fails, exercising continuation cancellation from a CQE.
  open.path = "/dev/null/notification";
  open.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  iree_async_notification_signal_operation_t signal = {};
  iree_async_operation_initialize(
      &signal.base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,
      IREE_ASYNC_OPERATION_FLAG_NONE, CompletionState::Callback, &completion);
  signal.notification = notification;
  signal.wake_count = 1;
  iree_async_operation_t* operations[] = {&open.base, &signal.base};
  IREE_ASSERT_OK(iree_async_proactor_submit(
      proactor_,
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations))));
  EXPECT_EQ(iree_async_notification_query_epoch(notification), initial_epoch);
  PollUntil(proactor_, [&] { return completion.call_count == 2; });
  EXPECT_EQ(completion.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_FAILED_PRECONDITION,
                                             IREE_STATUS_CANCELLED}));
  EXPECT_EQ(iree_async_notification_query_epoch(notification), initial_epoch);
  EXPECT_EQ(open.opened_file, nullptr);
  iree_async_notification_release(notification);
}

TEST_F(IoUringSubmitTest, FullSqRejectsBatchWithoutPublishingNotification) {
  FillSubmissionQueue();
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  const uint32_t initial_epoch =
      iree_async_notification_query_epoch(notification);
  CompletionState completion;
  iree_async_notification_signal_operation_t signal = {};
  iree_async_operation_initialize(
      &signal.base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,
      IREE_ASYNC_OPERATION_FLAG_NONE, CompletionState::Callback, &completion);
  signal.notification = notification;
  signal.wake_count = 1;
  iree_async_timer_operation_t timer = {};
  iree_async_operation_initialize(&timer.base, IREE_ASYNC_OPERATION_TYPE_TIMER,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  CompletionState::Callback, &completion);
  timer.deadline_ns = iree_time_now();
  iree_async_operation_t* operations[] = {&signal.base, &timer.base};
  auto batch =
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_async_proactor_submit(proactor_, batch));
  EXPECT_EQ(iree_async_notification_query_epoch(notification), initial_epoch);
  EXPECT_EQ(completion.call_count, 0);
  DrainFillers();
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, batch));
  PollUntil(proactor_, [&] { return completion.call_count == 2; });
  EXPECT_EQ(completion.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK, IREE_STATUS_OK}));
  EXPECT_EQ(iree_async_notification_query_epoch(notification),
            initial_epoch + 1);
  iree_async_notification_release(notification);
}

TEST_F(IoUringSubmitTest, MalformedTailDoesNotConsumeCloseOwnership) {
  int file_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(file_fd, 0);
  iree_async_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_async_file_import(proactor_,
                                        iree_async_primitive_from_fd(file_fd),
                                        IREE_ASYNC_FILE_FLAG_NONE, &file));

  CompletionState close_completion;
  iree_async_file_close_operation_t close_operation = {};
  close_operation.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE;
  close_operation.base.completion_fn = CompletionState::Callback;
  close_operation.base.user_data = &close_completion;
  close_operation.file = file;

  CompletionState malformed_completion;
  iree_async_socket_recv_operation_t malformed_operation = {};
  malformed_operation.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
  malformed_operation.base.completion_fn = CompletionState::Callback;
  malformed_operation.base.user_data = &malformed_completion;
  malformed_operation.buffers.count = IREE_ASYNC_SOCKET_RECV_MAX_BUFFERS + 1;

  iree_async_operation_t* operations[] = {&close_operation.base,
                                          &malformed_operation.base};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_submit(proactor_,
                                 iree_async_operation_list_make(
                                     operations, IREE_ARRAYSIZE(operations))));
  EXPECT_EQ(close_completion.call_count, 0);
  EXPECT_EQ(malformed_completion.call_count, 0);

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &close_operation.base));
  PollUntil(proactor_, [&] { return close_completion.call_count == 1; });
  EXPECT_EQ(close_completion.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
}

TEST_F(IoUringSubmitTest, MalformedTailDoesNotPublishFallbackMessage) {
  CreateTarget();

  std::vector<uint64_t> received_messages;
  iree_async_proactor_set_message_callback(
      target_proactor_,
      iree_async_proactor_message_callback_t{
          +[](iree_async_proactor_t* proactor, uint64_t message_data,
              void* user_data) {
            (void)proactor;
            static_cast<std::vector<uint64_t>*>(user_data)->push_back(
                message_data);
          },
          &received_messages});

  CompletionState message_completion;
  iree_async_message_operation_t message = {};
  message.base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
  message.base.completion_fn = CompletionState::Callback;
  message.base.user_data = &message_completion;
  message.target = target_proactor_;
  message.message_data = 0xA11CE;

  CompletionState malformed_completion;
  iree_async_socket_recv_operation_t malformed_operation = {};
  malformed_operation.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
  malformed_operation.base.completion_fn = CompletionState::Callback;
  malformed_operation.base.user_data = &malformed_completion;
  malformed_operation.buffers.count = IREE_ASYNC_SOCKET_RECV_MAX_BUFFERS + 1;

  iree_async_operation_t* operations[] = {&message.base,
                                          &malformed_operation.base};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_submit(proactor_,
                                 iree_async_operation_list_make(
                                     operations, IREE_ARRAYSIZE(operations))));
  PollImmediate(target_proactor_);
  EXPECT_TRUE(received_messages.empty());
  EXPECT_EQ(message_completion.call_count, 0);
  EXPECT_EQ(malformed_completion.call_count, 0);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &message.base));
  PollUntil(proactor_, [&] { return message_completion.call_count == 1; });
  PollUntil(target_proactor_, [&] { return received_messages.size() == 1; });
  EXPECT_EQ(received_messages, (std::vector<uint64_t>{0xA11CE}));
  EXPECT_EQ(message_completion.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
}

}  // namespace
