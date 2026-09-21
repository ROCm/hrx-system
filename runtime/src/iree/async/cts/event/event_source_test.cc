// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/event.h"
#include "iree/async/proactor.h"

namespace iree::async::cts {
namespace {

struct EventSourceOwner {
  // Independently owned resources borrowed by the registered source.
  iree_async_event_native_t native = {};
  // Number of readiness callbacks consumed.
  int ready_count = 0;
  // Number of terminal callbacks returning native ownership.
  int unregistered_count = 0;

  EventSourceOwner() {
    IREE_CHECK_OK(iree_async_event_native_initialize(&native));
  }
  ~EventSourceOwner() { iree_async_event_native_deinitialize(&native); }

  static void Ready(void* context, iree_async_event_source_t*,
                    iree_async_poll_events_t events) {
    auto* owner = static_cast<EventSourceOwner*>(context);
    EXPECT_TRUE(iree_any_bit_set(events, IREE_ASYNC_POLL_EVENT_IN));
    IREE_EXPECT_OK(iree_async_event_native_consume(&owner->native));
    ++owner->ready_count;
  }

  static void Unregistered(void* context) {
    auto* owner = static_cast<EventSourceOwner*>(context);
    iree_async_event_native_deinitialize(&owner->native);
    ++owner->unregistered_count;
  }
};

class EventSourceTest : public CtsTestBase<> {};

TEST_P(EventSourceTest, NullSourceCompletesInline) {
  int count = 0;
  iree_async_proactor_unregister_event_source(
      proactor_, nullptr,
      {+[](void* context) { ++*static_cast<int*>(context); }, &count});
  EXPECT_EQ(count, 1);
}

TEST_P(EventSourceTest, CompletionReturnsBorrowedNativeOwnership) {
  EventSourceOwner owner;
  iree_async_event_source_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_, owner.native.wait_primitive, {EventSourceOwner::Ready, &owner},
      &source));
  for (int i = 0; i < 3; ++i) {
    iree_async_event_native_set(&owner.native);
    PollUntilCondition([&] { return owner.ready_count > i; });
  }
  iree_async_proactor_unregister_event_source(
      proactor_, source, {EventSourceOwner::Unregistered, &owner});
  PollUntilCondition([&] { return owner.unregistered_count != 0; });
  EXPECT_EQ(owner.unregistered_count, 1);
  EXPECT_EQ(owner.native.wait_primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
}

TEST_P(EventSourceTest, DestructionCompletesAdmittedUnregistration) {
  EventSourceOwner owner;
  iree_async_event_source_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_, owner.native.wait_primitive, {EventSourceOwner::Ready, &owner},
      &source));
  // Establish native monitoring without requiring a producer to signal.
  iree_async_proactor_wake(proactor_);
  PollOneProgressEvent();
  iree_async_proactor_unregister_event_source(
      proactor_, source, {EventSourceOwner::Unregistered, &owner});
  iree_async_proactor_release(proactor_);
  proactor_ = nullptr;
  EXPECT_EQ(owner.unregistered_count, 1);
  EXPECT_EQ(owner.ready_count, 0);
}

#if defined(IREE_PLATFORM_WINDOWS)
CTS_REGISTER_TEST_SUITE_WITH_TAGS(EventSourceTest, {"wait_completion_packet"},
                                  {});
#else
CTS_REGISTER_TEST_SUITE(EventSourceTest);
#endif

class EventConsumptionTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (IsSkipped() || HasFatalFailure()) {
      return;
    }
    IREE_ASSERT_OK(iree_async_event_create(proactor_, &event_));
  }

  void TearDown() override {
    if (source_) {
      WaitForEventSourceUnregistration(source_);
    }
    iree_async_event_release(event_);
    CtsTestBase<>::TearDown();
  }

  void Register() {
    iree_async_event_source_callback_t callback = {
        +[](void* user_data, iree_async_event_source_t* source,
            iree_async_poll_events_t events) {
          auto* self = static_cast<EventConsumptionTest*>(user_data);
          EXPECT_EQ(source, self->source_);
          EXPECT_TRUE(iree_all_bits_set(events, IREE_ASYNC_POLL_EVENT_IN));
          IREE_EXPECT_OK(iree_async_event_consume(self->event_));
          ++self->callback_count_;
          if (self->signal_after_consumption_) {
            self->signal_after_consumption_ = false;
            iree_async_event_set(self->event_);
          }
        },
        this,
    };
    iree::Status status(iree_async_proactor_register_event_source(
        proactor_, event_->native.wait_primitive, callback, &source_));
    if (status.code() == iree::StatusCode::kUnavailable) {
      GTEST_SKIP() << "Persistent event sources unavailable: "
                   << status.ToString();
    }
    IREE_ASSERT_OK(status);
  }

  void Unregister() {
    WaitForEventSourceUnregistration(source_);
    source_ = nullptr;
  }

  void ExpectNoCallback() {
    const int previous_count = callback_count_;
    iree_host_size_t completed_count = 0;
    iree::Status status(iree_async_proactor_poll(
        proactor_, iree_immediate_timeout(), &completed_count));
    EXPECT_TRUE(status.ok() ||
                status.code() == iree::StatusCode::kDeadlineExceeded)
        << status.ToString();
    EXPECT_EQ(callback_count_, previous_count);
  }

  // Caller-owned event remains live across source unregistration/recreation.
  iree_async_event_t* event_ = nullptr;
  // One persistent registration; no per-signal operation object.
  iree_async_event_source_t* source_ = nullptr;
  // Number of delivered hints, not a producer progress counter.
  int callback_count_ = 0;
  // Emits one fresh signal from the callback after readiness consumption.
  bool signal_after_consumption_ = false;
};

TEST_P(EventConsumptionTest, CoalescesSignalsBeforeRegistration) {
  for (int i = 0; i < 3; ++i) {
    iree_async_event_set(event_);
  }
  ASSERT_NO_FATAL_FAILURE(Register());
  if (IsSkipped()) {
    return;
  }
  PollUntilCondition([&] { return callback_count_ != 0; });
  EXPECT_EQ(callback_count_, 1);
  ExpectNoCallback();
}

TEST_P(EventConsumptionTest, ConsumesSuccessiveSignalsWithOneRegistration) {
  ASSERT_NO_FATAL_FAILURE(Register());
  if (IsSkipped()) {
    return;
  }
  for (int i = 0; i < 3; ++i) {
    iree_async_event_set(event_);
    PollUntilCondition([&] { return callback_count_ >= i + 1; });
    EXPECT_EQ(callback_count_, i + 1);
    ExpectNoCallback();
  }
}

TEST_P(EventConsumptionTest, SignalAfterConsumptionSurvivesCallbackRearm) {
  ASSERT_NO_FATAL_FAILURE(Register());
  if (IsSkipped()) {
    return;
  }
  signal_after_consumption_ = true;
  iree_async_event_set(event_);
  PollUntilCondition([&] { return callback_count_ >= 2; });
  EXPECT_EQ(callback_count_, 2);
  ExpectNoCallback();
}

TEST_P(EventConsumptionTest, UnregistrationPreservesEventForFreshRegistration) {
  ASSERT_NO_FATAL_FAILURE(Register());
  if (IsSkipped()) {
    return;
  }
  iree_async_event_set(event_);
  PollUntilCondition([&] { return callback_count_ != 0; });
  ASSERT_EQ(callback_count_, 1);

  Unregister();
  iree_async_event_set(event_);
  ExpectNoCallback();
  ASSERT_NO_FATAL_FAILURE(Register());
  PollUntilCondition([&] { return callback_count_ >= 2; });
  EXPECT_EQ(callback_count_, 2);
  ExpectNoCallback();
}

#if defined(IREE_PLATFORM_WINDOWS)
CTS_REGISTER_TEST_SUITE_WITH_TAGS(EventConsumptionTest,
                                  {"wait_completion_packet"}, {});
#else
CTS_REGISTER_TEST_SUITE(EventConsumptionTest);
#endif

}  // namespace
}  // namespace iree::async::cts
