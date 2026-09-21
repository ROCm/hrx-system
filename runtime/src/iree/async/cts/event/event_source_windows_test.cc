// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
#include <windows.h>
// clang-format on

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/event.h"
#include "iree/async/proactor.h"

namespace iree::async::cts {
namespace {

struct EventSourceCallbackState {
  // Number of callbacks dispatched.
  int call_count = 0;

  // Polling thread that dispatched the most recent callback.
  DWORD thread_id = 0;

  // Events delivered to the most recent callback.
  iree_async_poll_events_t events = IREE_ASYNC_POLL_EVENT_NONE;
};

static void RecordEventSourceCallback(void* user_data,
                                      iree_async_event_source_t* source,
                                      iree_async_poll_events_t events) {
  (void)source;
  auto* state = static_cast<EventSourceCallbackState*>(user_data);
  ++state->call_count;
  state->thread_id = GetCurrentThreadId();
  state->events = events;
}

class EventSourceWindowsTest : public CtsTestBase<> {
 protected:
  HANDLE CreateTestEvent() {
    HANDLE event_handle = CreateEventW(NULL, /*bManualReset=*/FALSE,
                                       /*bInitialState=*/FALSE, NULL);
    EXPECT_NE(event_handle, static_cast<HANDLE>(NULL));
    return event_handle;
  }

  iree_async_event_source_callback_t MakeCallback(
      EventSourceCallbackState* state) {
    return {RecordEventSourceCallback, state};
  }
};

TEST_P(EventSourceWindowsTest, RejectsInvalidArguments) {
  EventSourceCallbackState state;
  iree_async_event_source_t* source = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_proactor_register_event_source(
                            proactor_, iree_async_primitive_none(),
                            MakeCallback(&state), &source));
  EXPECT_EQ(source, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_register_event_source(
          proactor_, iree_async_primitive_from_win32_handle(0),
          MakeCallback(&state), &source));
  EXPECT_EQ(source, nullptr);

  HANDLE event_handle = CreateTestEvent();
  ASSERT_NE(event_handle, static_cast<HANDLE>(NULL));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_register_event_source(
          proactor_,
          iree_async_primitive_from_win32_handle((uintptr_t)event_handle),
          iree_async_event_source_callback_null(), &source));
  EXPECT_EQ(source, nullptr);
  EXPECT_TRUE(CloseHandle(event_handle));
}

TEST_P(EventSourceWindowsTest, UnregisterPreservesCallerOwnedHandle) {
  HANDLE event_handle = CreateTestEvent();
  ASSERT_NE(event_handle, static_cast<HANDLE>(NULL));
  EventSourceCallbackState state;

  iree_async_event_source_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_,
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle),
      MakeCallback(&state), &source));
  ASSERT_NE(source, nullptr);

  WaitForEventSourceUnregistration(source);
  EXPECT_TRUE(SetEvent(event_handle));
  EXPECT_EQ(WaitForSingleObject(event_handle, 0), WAIT_OBJECT_0);
  EXPECT_TRUE(CloseHandle(event_handle));
}

TEST_P(EventSourceWindowsTest, CallbackRunsOnPollingThread) {
  HANDLE event_handle = CreateTestEvent();
  ASSERT_NE(event_handle, static_cast<HANDLE>(NULL));
  EventSourceCallbackState state;

  iree_async_event_source_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_,
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle),
      MakeCallback(&state), &source));

  DWORD polling_thread_id = GetCurrentThreadId();
  ASSERT_TRUE(SetEvent(event_handle));
  PollUntilCondition([&] { return state.call_count == 1; },
                     "event source callback");
  EXPECT_EQ(state.thread_id, polling_thread_id);
  EXPECT_TRUE(iree_all_bits_set(state.events, IREE_ASYNC_POLL_EVENT_IN));

  WaitForEventSourceUnregistration(source);
  EXPECT_TRUE(CloseHandle(event_handle));
}

TEST_P(EventSourceWindowsTest, RearmsAfterEachCallback) {
  HANDLE event_handle = CreateTestEvent();
  ASSERT_NE(event_handle, static_cast<HANDLE>(NULL));
  EventSourceCallbackState state;

  iree_async_event_source_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_,
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle),
      MakeCallback(&state), &source));

  constexpr int kSignalCount = 3;
  for (int i = 0; i < kSignalCount; ++i) {
    ASSERT_TRUE(SetEvent(event_handle));
    PollUntilCondition([&] { return state.call_count == i + 1; },
                       "event source callback");
  }
  EXPECT_EQ(state.call_count, kSignalCount);

  WaitForEventSourceUnregistration(source);
  EXPECT_TRUE(CloseHandle(event_handle));
}

TEST_P(EventSourceWindowsTest, ConsumptionPreservesSignalFollowingNativeWait) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));
  struct CallbackState {
    // Caller event whose first signal is consumed by the native wait.
    iree_async_event_t* event;
    // Number of callbacks received over the one registration.
    int count = 0;
  } state = {event};
  iree_async_event_source_callback_t callback = {
      +[](void* user_data, iree_async_event_source_t*,
          iree_async_poll_events_t events) {
        auto* state = static_cast<CallbackState*>(user_data);
        EXPECT_TRUE(iree_all_bits_set(events, IREE_ASYNC_POLL_EVENT_IN));
        if (++state->count == 1) {
          // The native wait has consumed the first signal. This second signal
          // must survive the portable consumption of the delivered hint.
          iree_async_event_set(state->event);
        }
        IREE_EXPECT_OK(iree_async_event_consume(state->event));
      },
      &state,
  };
  iree_async_event_source_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_, event->native.wait_primitive, callback, &source));
  iree_async_event_set(event);
  PollUntilCondition([&] { return state.count >= 2; });
  EXPECT_EQ(state.count, 2);
  WaitForEventSourceUnregistration(source);
  iree_async_event_release(event);
}

TEST_P(EventSourceWindowsTest, SignalBeforeUnregisterRemovesCompletion) {
  EventSourceCallbackState state;
  constexpr int kIterationCount = 64;
  for (int i = 0; i < kIterationCount; ++i) {
    HANDLE event_handle = CreateTestEvent();
    ASSERT_NE(event_handle, static_cast<HANDLE>(NULL));
    iree_async_event_source_t* source = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_register_event_source(
        proactor_,
        iree_async_primitive_from_win32_handle((uintptr_t)event_handle),
        MakeCallback(&state), &source));
    ASSERT_TRUE(SetEvent(event_handle));
    WaitForEventSourceUnregistration(source);
    EXPECT_TRUE(CloseHandle(event_handle));
  }

  iree_host_size_t completed_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(),
                               &completed_count));
  EXPECT_EQ(completed_count, 0u);
  EXPECT_EQ(state.call_count, 0);
}

TEST_P(EventSourceWindowsTest, DispatchesIndependentSources) {
  constexpr int kSourceCount = 3;
  HANDLE event_handles[kSourceCount] = {};
  iree_async_event_source_t* sources[kSourceCount] = {};
  EventSourceCallbackState states[kSourceCount];

  for (int i = 0; i < kSourceCount; ++i) {
    event_handles[i] = CreateTestEvent();
    ASSERT_NE(event_handles[i], static_cast<HANDLE>(NULL));
    IREE_ASSERT_OK(iree_async_proactor_register_event_source(
        proactor_,
        iree_async_primitive_from_win32_handle((uintptr_t)event_handles[i]),
        MakeCallback(&states[i]), &sources[i]));
  }

  ASSERT_TRUE(SetEvent(event_handles[1]));
  PollUntilCondition([&] { return states[1].call_count == 1; },
                     "selected event source callback");
  EXPECT_EQ(states[0].call_count, 0);
  EXPECT_EQ(states[2].call_count, 0);

  for (int i = 0; i < kSourceCount; ++i) {
    WaitForEventSourceUnregistration(sources[i]);
    EXPECT_TRUE(CloseHandle(event_handles[i]));
  }
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(EventSourceWindowsTest,
                                  {"wait_completion_packet"}, {});

}  // namespace
}  // namespace iree::async::cts
