// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for Win32 HANDLE event source operations.

#include <atomic>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/proactor.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace iree::async::cts {

class EventSourceWin32Test : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
#if !defined(IREE_PLATFORM_WINDOWS)
    GTEST_SKIP() << "EventSourceWin32Test requires Win32 HANDLEs";
#endif  // !IREE_PLATFORM_WINDOWS
  }

#if defined(IREE_PLATFORM_WINDOWS)
  HANDLE CreateAutoResetEvent() {
    HANDLE handle = CreateEventW(/*lpEventAttributes=*/NULL,
                                 /*bManualReset=*/FALSE,
                                 /*bInitialState=*/FALSE, /*lpName=*/NULL);
    EXPECT_NE(handle, nullptr) << "CreateEventW failed";
    return handle;
  }
#endif  // IREE_PLATFORM_WINDOWS
};

#if defined(IREE_PLATFORM_WINDOWS)

TEST_P(EventSourceWin32Test, RegisterUnregister) {
  HANDLE event_handle = CreateAutoResetEvent();
  if (!event_handle) return;

  std::atomic<int> callback_count{0};
  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* count = static_cast<std::atomic<int>*>(user_data);
        count->fetch_add(1, std::memory_order_relaxed);
      },
      &callback_count};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, primitive,
                                                           callback, &source));
  ASSERT_NE(source, nullptr);

  iree_async_proactor_unregister_event_source(proactor_, source);
  CloseHandle(event_handle);
}

TEST_P(EventSourceWin32Test, CallbackFires) {
  HANDLE event_handle = CreateAutoResetEvent();
  if (!event_handle) return;

  struct CallbackState {
    std::atomic<int> call_count{0};
    std::atomic<iree_async_poll_events_t> last_events{0};
  };
  CallbackState state;

  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* state = static_cast<CallbackState*>(user_data);
        state->last_events.store(events, std::memory_order_relaxed);
        state->call_count.fetch_add(1, std::memory_order_relaxed);
      },
      &state};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, primitive,
                                                           callback, &source));

  ASSERT_TRUE(SetEvent(event_handle));
  PollUntilCondition(
      [&] { return state.call_count.load(std::memory_order_relaxed) >= 1; },
      "Win32 event source callback");

  EXPECT_TRUE(state.last_events.load(std::memory_order_relaxed) &
              IREE_ASYNC_POLL_EVENT_IN);

  iree_async_proactor_unregister_event_source(proactor_, source);
  CloseHandle(event_handle);
}

TEST_P(EventSourceWin32Test, MultipleSignals) {
  HANDLE event_handle = CreateAutoResetEvent();
  if (!event_handle) return;

  std::atomic<int> call_count{0};
  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* count = static_cast<std::atomic<int>*>(user_data);
        count->fetch_add(1, std::memory_order_relaxed);
      },
      &call_count};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, primitive,
                                                           callback, &source));

  constexpr int kSignalCount = 3;
  for (int i = 0; i < kSignalCount; ++i) {
    ASSERT_TRUE(SetEvent(event_handle));
    PollUntilCondition(
        [&] { return call_count.load(std::memory_order_relaxed) > i; },
        "Win32 event source signal");
  }

  EXPECT_GE(call_count.load(std::memory_order_relaxed), kSignalCount);

  iree_async_proactor_unregister_event_source(proactor_, source);
  CloseHandle(event_handle);
}

TEST_P(EventSourceWin32Test, UnregisterStopsCallbacks) {
  HANDLE event_handle = CreateAutoResetEvent();
  if (!event_handle) return;

  std::atomic<int> call_count{0};
  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* count = static_cast<std::atomic<int>*>(user_data);
        count->fetch_add(1, std::memory_order_relaxed);
      },
      &call_count};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, primitive,
                                                           callback, &source));
  iree_async_proactor_unregister_event_source(proactor_, source);

  ASSERT_TRUE(SetEvent(event_handle));
  PollOnce();
  EXPECT_EQ(call_count.load(std::memory_order_relaxed), 0);

  CloseHandle(event_handle);
}

TEST_P(EventSourceWin32Test, UnregisterAfterSignalStopsCallbacks) {
  HANDLE event_handle = CreateAutoResetEvent();
  if (!event_handle) return;

  std::atomic<int> call_count{0};
  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* count = static_cast<std::atomic<int>*>(user_data);
        count->fetch_add(1, std::memory_order_relaxed);
      },
      &call_count};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, primitive,
                                                           callback, &source));

  ASSERT_TRUE(SetEvent(event_handle));
  iree_async_proactor_unregister_event_source(proactor_, source);

  PollOnce();
  EXPECT_EQ(call_count.load(std::memory_order_relaxed), 0);

  CloseHandle(event_handle);
}

TEST_P(EventSourceWin32Test, MultipleEventSources) {
  constexpr int kSourceCount = 3;

  HANDLE event_handles[kSourceCount] = {nullptr, nullptr, nullptr};
  iree_async_event_source_t* sources[kSourceCount] = {nullptr, nullptr,
                                                      nullptr};
  std::atomic<int> callback_counts[kSourceCount];
  for (int i = 0; i < kSourceCount; ++i) {
    event_handles[i] = CreateAutoResetEvent();
    ASSERT_NE(event_handles[i], nullptr);
    callback_counts[i].store(0, std::memory_order_relaxed);
  }

  struct CallbackData {
    std::atomic<int>* count;
  };
  CallbackData callback_data[kSourceCount];
  for (int i = 0; i < kSourceCount; ++i) {
    callback_data[i].count = &callback_counts[i];
    iree_async_event_source_callback_t callback = {
        [](void* user_data, iree_async_event_source_t* source,
           iree_async_poll_events_t events) {
          auto* data = static_cast<CallbackData*>(user_data);
          data->count->fetch_add(1, std::memory_order_relaxed);
        },
        &callback_data[i]};
    iree_async_primitive_t primitive =
        iree_async_primitive_from_win32_handle((uintptr_t)event_handles[i]);
    IREE_ASSERT_OK(iree_async_proactor_register_event_source(
        proactor_, primitive, callback, &sources[i]));
  }

  ASSERT_TRUE(SetEvent(event_handles[1]));
  PollUntilCondition(
      [&] { return callback_counts[1].load(std::memory_order_relaxed) >= 1; },
      "middle Win32 event source callback");
  EXPECT_EQ(callback_counts[0].load(std::memory_order_relaxed), 0);
  EXPECT_GE(callback_counts[1].load(std::memory_order_relaxed), 1);
  EXPECT_EQ(callback_counts[2].load(std::memory_order_relaxed), 0);

  for (int i = 0; i < kSourceCount; ++i) {
    ASSERT_TRUE(SetEvent(event_handles[i]));
  }
  PollUntilCondition(
      [&] {
        for (int i = 0; i < kSourceCount; ++i) {
          if (callback_counts[i].load(std::memory_order_relaxed) < 1) {
            return false;
          }
        }
        return true;
      },
      "all Win32 event source callbacks");

  for (int i = 0; i < kSourceCount; ++i) {
    iree_async_proactor_unregister_event_source(proactor_, sources[i]);
    CloseHandle(event_handles[i]);
  }
}

#endif  // IREE_PLATFORM_WINDOWS

CTS_REGISTER_TEST_SUITE(EventSourceWin32Test);

}  // namespace iree::async::cts
