// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/eventfd.h>
#include <unistd.h>

#include <cstring>
#include <vector>

#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/io_uring/api.h"
#include "iree/async/relay.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ProactorLifetimeTest : public ::testing::Test {
 protected:
  struct CountingAllocatorState {
    // Allocator receiving forwarded commands.
    iree_allocator_t delegate = iree_allocator_system();

    // Most recent allocation returned by the delegate.
    void* last_allocation = nullptr;

    // Allocation whose release the current test observes.
    void* watched_allocation = nullptr;

    // Set when the watched allocation is passed to FREE.
    bool watched_allocation_freed = false;
  };

  struct NopCompletionState {
    // Number of NOP callbacks dispatched by the poll loop.
    iree_host_size_t count = 0;
  };

  static iree_status_t CountingAllocatorCtl(void* self,
                                            iree_allocator_command_t command,
                                            const void* params,
                                            void** inout_ptr) {
    auto* state = static_cast<CountingAllocatorState*>(self);
    void* old_ptr = *inout_ptr;
    if (command == IREE_ALLOCATOR_COMMAND_FREE &&
        old_ptr == state->watched_allocation) {
      state->watched_allocation_freed = true;
    }
    iree_status_t status =
        state->delegate.ctl(state->delegate.self, command, params, inout_ptr);
    if (iree_status_is_ok(status) &&
        (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
         command == IREE_ALLOCATOR_COMMAND_CALLOC ||
         (command == IREE_ALLOCATOR_COMMAND_REALLOC && !old_ptr))) {
      state->last_allocation = *inout_ptr;
    }
    return status;
  }

  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    options.max_concurrent_operations = 8;
    iree_allocator_t allocator = {
        &allocator_state_,
        CountingAllocatorCtl,
    };
    iree_status_t create_status =
        iree_async_proactor_create_io_uring(options, allocator, &proactor_);
    if (iree_status_is_unavailable(create_status)) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, create_status);
      GTEST_SKIP() << "io_uring is unavailable";
    }
    IREE_ASSERT_OK(create_status);

    // Bind the poll owner and arm the persistent wake operation before tests
    // fill the submission queue.
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DEADLINE_EXCEEDED,
        iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr));
  }

  void TearDown() override {
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

  void FillSubmissionQueue() {
    // Submit without polling until get_sqe() reports the queue full. These
    // SQEs remain unpublished, guaranteeing cancellation cannot allocate an
    // SQE on its first attempt.
    nop_operations_.resize(1024);
    for (auto& nop : nop_operations_) {
      memset(&nop, 0, sizeof(nop));
      nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
      nop.base.completion_fn =
          +[](void* user_data, iree_async_operation_t* operation,
              iree_status_t status, iree_async_completion_flags_t flags) {
            auto* state = static_cast<NopCompletionState*>(user_data);
            (void)operation;
            (void)flags;
            IREE_EXPECT_OK(status);
            ++state->count;
          };
      nop.base.user_data = &nop_state_;
      iree_status_t submit_status =
          iree_async_proactor_submit_one(proactor_, &nop.base);
      if (iree_status_is_resource_exhausted(submit_status)) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, submit_status);
        break;
      }
      IREE_ASSERT_OK(submit_status);
      ++submitted_count_;
    }
    ASSERT_GT(submitted_count_, 0u);
    ASSERT_LT(submitted_count_, nop_operations_.size());
  }

  void PollOnce() {
    IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                            /*out_completed_count=*/nullptr));
  }

  iree_async_proactor_t* proactor_ = nullptr;
  CountingAllocatorState allocator_state_;
  std::vector<iree_async_nop_operation_t> nop_operations_;
  NopCompletionState nop_state_;
  iree_host_size_t submitted_count_ = 0;
};

TEST_F(ProactorLifetimeTest,
       EventSourceUnregistrationRetriesAfterSubmissionQueuePressure) {
  int event_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(event_fd, 0);

  struct EventState {
    // Number of event source callbacks dispatched by the poll loop.
    iree_host_size_t count = 0;
  } event_state;
  iree_async_event_source_callback_t event_callback = {
      +[](void* user_data, iree_async_event_source_t* source,
          iree_async_poll_events_t events) {
        auto* state = static_cast<EventState*>(user_data);
        (void)source;
        (void)events;
        ++state->count;
      },
      &event_state,
  };
  iree_async_event_source_t* event_source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_, iree_async_primitive_from_fd(event_fd), event_callback,
      &event_source));
  ASSERT_EQ(static_cast<void*>(event_source), allocator_state_.last_allocation);
  allocator_state_.watched_allocation = event_source;

  FillSubmissionQueue();
  iree_async_proactor_unregister_event_source(proactor_, event_source);

  while (!allocator_state_.watched_allocation_freed ||
         nop_state_.count < submitted_count_) {
    PollOnce();
  }

  EXPECT_EQ(event_state.count, 0u);
  EXPECT_EQ(nop_state_.count, submitted_count_);
  close(event_fd);
}

TEST_F(ProactorLifetimeTest,
       RelayUnregistrationRetriesAfterSubmissionQueuePressure) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));
  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));

  FillSubmissionQueue();

  struct UnregistrationState {
    // Set after the relay has no remaining backend references.
    bool completed = false;
  } unregistration_state;
  iree_async_relay_unregistered_callback_t unregistered_callback = {
      +[](void* user_data) {
        static_cast<UnregistrationState*>(user_data)->completed = true;
      },
      &unregistration_state,
  };
  iree_async_proactor_unregister_relay(proactor_, relay, unregistered_callback);
  EXPECT_FALSE(unregistration_state.completed);

  while (!unregistration_state.completed ||
         nop_state_.count < submitted_count_) {
    PollOnce();
  }

  EXPECT_EQ(nop_state_.count, submitted_count_);
  iree_async_notification_release(source_notification);
  iree_async_notification_release(sink_notification);
}

TEST_F(ProactorLifetimeTest,
       PersistentRelayFaultCancelsSourceBeforeUnregistration) {
  int source_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(source_fd, 0);
  int sink_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(sink_fd, 0);

  struct FaultState {
    // Set when the relay reports its terminal sink failure.
    bool faulted = false;
    // Status code reported for the failed sink write.
    iree_status_code_t status_code = IREE_STATUS_OK;
  } fault_state;
  iree_async_relay_error_callback_t error_callback = {
      +[](void* user_data, iree_async_relay_t* relay, iree_status_t status) {
        auto* state = static_cast<FaultState*>(user_data);
        (void)relay;
        state->status_code = iree_status_code(status);
        state->faulted = true;
        iree_status_free(status);
      },
      &fault_state,
  };

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, error_callback, &relay));

  // Closing the sink makes the next source transfer fault. The source remains
  // a live multishot poll until the backend receives its terminal cancel CQE.
  close(sink_fd);
  uint64_t signal_value = 1;
  ASSERT_EQ(write(source_fd, &signal_value, sizeof(signal_value)),
            sizeof(signal_value));
  while (!fault_state.faulted) PollOnce();
  EXPECT_NE(fault_state.status_code, IREE_STATUS_OK);

  struct UnregistrationState {
    // Set after the relay has no remaining backend references.
    bool completed = false;
  } unregistration_state;
  iree_async_relay_unregistered_callback_t unregistered_callback = {
      +[](void* user_data) {
        static_cast<UnregistrationState*>(user_data)->completed = true;
      },
      &unregistration_state,
  };
  iree_async_proactor_unregister_relay(proactor_, relay, unregistered_callback);
  while (!unregistration_state.completed) PollOnce();

  close(source_fd);
}

}  // namespace
