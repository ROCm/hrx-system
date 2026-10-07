// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

#include "iree/async/api.h"
#include "iree/async/platform/io_uring/api.h"
#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/async/platform/io_uring/relay.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum class Observer { kEventSource, kPrimitiveRelay, kNotificationRelay };
enum class ReceiptOrder { kPollFirst, kCancellationFirst };
enum class Admission { kLogical, kStaged, kSubmitted, kNotified };

enum ReceiptFlagBits {
  kPollReceived = 1u << 0,
  kCancellationReceived = 1u << 1,
};
using ReceiptFlags = uint32_t;

// Native operations run normally. Only delivery of their actual receipts is
// held until both arrive, then dispatched in the selected order. All borrowed
// handles and callback contexts remain alive through the test.
struct RetirementWitness {
  // Observer whose native receipts are selected from the real completion ring.
  Observer observer;
  // Order in which the two native receipts reach the production dispatcher.
  ReceiptOrder order;
  // Logical-only observers have no native completion obligation.
  bool native_expected;
  // Actual terminal poll CQE, retained until its cancellation receipt arrives.
  iree_io_uring_cqe_t poll = {};
  // Actual cancellation CQE, retained until its terminal poll receipt arrives.
  iree_io_uring_cqe_t cancellation = {};
  // Native receipts collected and retained for ordered delivery.
  ReceiptFlags received = 0;
  // Number of collected receipts released to the production dispatcher.
  int dispatched = 0;
  // Number of terminal ownership callbacks delivered by the real observer.
  int unregistered = 0;
  // Number of ring deinitializations after terminal ownership returns.
  int ring_closes = 0;

  static void Unregistered(void* user_data) {
    auto* witness = static_cast<RetirementWitness*>(user_data);
    EXPECT_EQ(witness->dispatched, witness->native_expected ? 2 : 0);
    EXPECT_EQ(witness->ring_closes, 0);
    ++witness->unregistered;
  }

  static void EventSourceUnregistered(void* user_data, iree_status_t status) {
    IREE_EXPECT_OK(status);
    Unregistered(user_data);
  }
};

// Tests and native dispatch run on one thread; other rings are not intercepted.
thread_local RetirementWitness* retirement_witness = nullptr;

}  // namespace

extern "C" iree_host_size_t __real_iree_async_proactor_io_uring_process_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_status_t* inout_status);

extern "C" iree_host_size_t __wrap_iree_async_proactor_io_uring_process_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_status_t* inout_status) {
  auto* witness = retirement_witness;
  if (!witness) {
    return __real_iree_async_proactor_io_uring_process_cqe(proactor, cqe,
                                                           inout_status);
  }
  bool is_poll = false;
  bool is_cancellation = false;
  if (iree_io_uring_is_internal_cqe(cqe)) {
    auto tag = iree_io_uring_internal_tag(cqe->user_data);
    switch (witness->observer) {
      case Observer::kEventSource:
        is_poll = tag == IREE_IO_URING_TAG_EVENT_SOURCE;
        is_cancellation = tag == IREE_IO_URING_TAG_EVENT_SOURCE_CANCEL;
        break;
      case Observer::kPrimitiveRelay:
        is_poll = tag == IREE_IO_URING_TAG_RELAY;
        is_cancellation = tag == IREE_IO_URING_TAG_RELAY_CANCEL;
        break;
      case Observer::kNotificationRelay:
        is_cancellation = tag == IREE_IO_URING_TAG_CANCEL &&
                          iree_io_uring_internal_payload(cqe->user_data) != 0;
        break;
    }
  } else {
    is_poll = witness->observer == Observer::kNotificationRelay;
  }
  if (!is_poll && !is_cancellation) {
    return __real_iree_async_proactor_io_uring_process_cqe(proactor, cqe,
                                                           inout_status);
  }
  // A readiness receipt racing shutdown does not retire a multishot poll.
  // The production dispatcher suppresses its callback after unregistration.
  if (is_poll && (cqe->flags & IREE_IORING_CQE_F_MORE)) {
    return __real_iree_async_proactor_io_uring_process_cqe(proactor, cqe,
                                                           inout_status);
  }
  EXPECT_TRUE(witness->native_expected);
  EXPECT_EQ(witness->unregistered, 0);
  EXPECT_EQ(witness->ring_closes, 0);
  if (is_poll) {
    EXPECT_FALSE(iree_any_bit_set(witness->received, kPollReceived));
    EXPECT_EQ(cqe->flags & IREE_IORING_CQE_F_MORE, 0u);
    witness->poll = *cqe;
    witness->received |= kPollReceived;
  } else {
    EXPECT_FALSE(iree_any_bit_set(witness->received, kCancellationReceived));
    witness->cancellation = *cqe;
    witness->received |= kCancellationReceived;
  }
  if (witness->received != (kPollReceived | kCancellationReceived)) {
    return 0;
  }
  const auto* first = witness->order == ReceiptOrder::kPollFirst
                          ? &witness->poll
                          : &witness->cancellation;
  const auto* second = witness->order == ReceiptOrder::kPollFirst
                           ? &witness->cancellation
                           : &witness->poll;
  ++witness->dispatched;
  iree_host_size_t completed = __real_iree_async_proactor_io_uring_process_cqe(
      proactor, first, inout_status);
  EXPECT_EQ(witness->unregistered, 0);
  ++witness->dispatched;
  completed += __real_iree_async_proactor_io_uring_process_cqe(proactor, second,
                                                               inout_status);
  return completed;
}

TEST(IoUringEventSourceTest, CancellationFailureRetainsSourceAndProactor) {
  iree_async_proactor_io_uring_t proactor = {};
  iree_async_event_source_t source = {};
  source.flags = IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_POLL_IN_FLIGHT |
                 IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_IN_FLIGHT;
  proactor.event_sources = &source;
  struct TerminalResult {
    int callback_count = 0;
    iree_status_code_t code = IREE_STATUS_OK;
  } result;
  source.unregistered_callback = {
      +[](void* user_data, iree_status_t status) {
        auto* result = static_cast<TerminalResult*>(user_data);
        result->code = iree_status_code(status);
        iree_status_free(status);
        ++result->callback_count;
      },
      &result,
  };
  iree_io_uring_cqe_t cqe = {};
  cqe.user_data = iree_io_uring_internal_encode(
      IREE_IO_URING_TAG_EVENT_SOURCE_CANCEL, (uintptr_t)&source);
  cqe.res = -EINVAL;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_io_uring_event_source_complete_cancel(&proactor, &cqe));

  EXPECT_EQ(result.callback_count, 1);
  EXPECT_EQ(result.code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(proactor.event_sources, &source);
  EXPECT_TRUE(iree_any_bit_set(source.flags,
                               IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_RETAINED));
  EXPECT_FALSE(iree_any_bit_set(
      source.flags, IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_IN_FLIGHT));

  // Final release cannot wait for a poll whose cancellation already failed.
  // Destruction diagnoses the retained source and leaves the graph intact.
  iree_async_proactor_io_uring_destroy(&proactor.base);
  EXPECT_EQ(proactor.event_sources, &source);
}

extern "C" void __real_iree_io_uring_ring_deinitialize(
    iree_io_uring_ring_t* ring);
extern "C" void __wrap_iree_io_uring_ring_deinitialize(
    iree_io_uring_ring_t* ring) {
  if (auto* witness = retirement_witness) {
    EXPECT_EQ(witness->dispatched, witness->native_expected ? 2 : 0);
    EXPECT_EQ(witness->unregistered, 1);
    ++witness->ring_closes;
  }
  __real_iree_io_uring_ring_deinitialize(ring);
}

namespace {

class ProactorShutdownTest
    : public ::testing::TestWithParam<
          std::tuple<Observer, ReceiptOrder, Admission>> {
 protected:
  void SetUp() override {
    iree_status_t status = iree_async_proactor_create_io_uring(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "io_uring is unavailable";
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_async_event_native_initialize(&source_));
    IREE_ASSERT_OK(iree_async_event_native_initialize(&sink_));
  }

  void TearDown() override {
    retirement_witness = nullptr;
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
    iree_async_event_native_deinitialize(&sink_);
    iree_async_event_native_deinitialize(&source_);
  }

  void Register(Observer observer) {
    if (observer == Observer::kEventSource) {
      IREE_ASSERT_OK(iree_async_proactor_register_event_source(
          proactor_, source_.wait_primitive,
          {+[](void*, iree_async_event_source_t*, iree_async_poll_events_t) {
             ADD_FAILURE()
                 << "Readiness must not dispatch after unregistration";
           },
           nullptr},
          &event_source_));
    } else {
      auto source =
          iree_async_relay_source_from_primitive(source_.wait_primitive);
      if (observer == Observer::kNotificationRelay) {
        IREE_ASSERT_OK(iree_async_notification_create(
            proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
        source = iree_async_relay_source_from_notification(notification_);
      }
      IREE_ASSERT_OK(iree_async_proactor_register_relay(
          proactor_, source,
          iree_async_relay_sink_signal_primitive(sink_.signal_primitive, 1),
          IREE_ASYNC_RELAY_FLAG_PERSISTENT,
          iree_async_relay_error_callback_none(), &relay_));
    }
  }

  // Proactor and all native resources stay live while receipts are withheld.
  iree_async_proactor_t* proactor_ = nullptr;
  // Borrowed native source, never signaled in the retirement cases.
  iree_async_event_native_t source_ = {};
  // Borrowed relay sink, retained independently of the relay.
  iree_async_event_native_t sink_ = {};
  // Managed source whose final reference is released before ring destruction.
  iree_async_notification_t* notification_ = nullptr;
  // Registered event source, if selected by the test parameters.
  iree_async_event_source_t* event_source_ = nullptr;
  // Registered primitive or notification relay, if selected.
  iree_async_relay_t* relay_ = nullptr;
};

TEST_P(ProactorShutdownTest, JoinsNativeReceiptsBeforeOwnershipReturn) {
  auto [observer, order, admission] = GetParam();
  RetirementWitness witness{observer, order, admission != Admission::kLogical};
  ASSERT_NO_FATAL_FAILURE(Register(observer));
  auto* backend = iree_async_proactor_io_uring_cast(proactor_);
  if (admission == Admission::kStaged) {
    // Stage the real monitor without publishing the SQ tail to the kernel.
    EXPECT_FALSE(iree_async_io_uring_event_source_submit_pending(backend));
    EXPECT_FALSE(iree_async_io_uring_retry_pending_relays(backend));
    iree_async_io_uring_notification_drain_pending(backend);
    EXPECT_EQ(*backend->ring.sq_head, *backend->ring.sq_tail);
    EXPECT_NE(backend->ring.sq_local_tail, *backend->ring.sq_tail);
  } else if (admission == Admission::kSubmitted ||
             admission == Admission::kNotified) {
    iree_status_t status =
        iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_free(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  if (admission == Admission::kNotified) {
    // Queue native readiness after arming but before the next GETEVENTS runs
    // deferred task work. Cancellation must terminate the persistent poll even
    // while that task work owns it; readiness is not terminal completion.
    if (notification_) {
      iree_async_notification_signal(notification_, 1);
    } else {
      iree_async_event_native_set(&source_);
    }
  }

  retirement_witness = &witness;
  if (event_source_) {
    iree_async_proactor_unregister_event_source(
        proactor_, event_source_,
        {RetirementWitness::EventSourceUnregistered, &witness});
    event_source_ = nullptr;
  } else {
    iree_async_proactor_unregister_relay(
        proactor_, relay_, {RetirementWitness::Unregistered, &witness});
    relay_ = nullptr;
  }
  iree_async_notification_release(notification_);
  notification_ = nullptr;
  iree_async_proactor_release(proactor_);
  proactor_ = nullptr;
  retirement_witness = nullptr;
  EXPECT_EQ(witness.unregistered, 1);
  EXPECT_EQ(witness.ring_closes, 1);
  EXPECT_EQ(witness.dispatched, witness.native_expected ? 2 : 0);
}

INSTANTIATE_TEST_SUITE_P(
    NativeOwnership, ProactorShutdownTest,
    ::testing::Combine(
        ::testing::Values(Observer::kEventSource, Observer::kPrimitiveRelay,
                          Observer::kNotificationRelay),
        ::testing::Values(ReceiptOrder::kPollFirst,
                          ReceiptOrder::kCancellationFirst),
        ::testing::Values(Admission::kLogical, Admission::kStaged,
                          Admission::kSubmitted, Admission::kNotified)));

}  // namespace
