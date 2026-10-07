// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/iocp/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr LONG kNativeCancellationFailure =
    static_cast<LONG>(0xC000000DL);
static constexpr LONG kNativeCompletionPublished =
    static_cast<LONG>(0x00000103L);

static LONG NTAPI ReturnCancellationFailure(HANDLE handle,
                                            BOOLEAN remove_packet) {
  (void)handle;
  (void)remove_packet;
  return kNativeCancellationFailure;
}

static LONG NTAPI ReturnWithdrawal(HANDLE handle, BOOLEAN remove_packet) {
  (void)handle;
  (void)remove_packet;
  return 0;
}

static LONG NTAPI ReturnPublishedCompletion(HANDLE handle,
                                            BOOLEAN remove_packet) {
  (void)handle;
  (void)remove_packet;
  return kNativeCompletionPublished;
}

struct CompletionState {
  // Number of terminal operation callbacks observed.
  int count = 0;

  // Status code delivered by the last terminal callback.
  iree_status_code_t code = IREE_STATUS_UNKNOWN;

  static void Record(void* user_data, iree_async_operation_t*,
                     iree_status_t status,
                     iree_async_completion_flags_t flags) {
    EXPECT_EQ(flags, IREE_ASYNC_COMPLETION_FLAG_NONE);
    auto* self = static_cast<CompletionState*>(user_data);
    ++self->count;
    self->code = iree_status_code(status);
    iree_status_free(status);
  }
};

struct OwnedCompletionState {
  // Poll owner for joining a target that retires before its request.
  iree_async_proactor_t* proactor = nullptr;

  // Caller-owned cancellation storage retained through its receipt.
  iree_async_cancel_request_t request = {};

  // Number of cancellation-key retirement receipts observed.
  int receipt_count = 0;

  // Independent terminal result of the cancellation target.
  CompletionState target;

  static void Receipt(void* user_data) {
    ++static_cast<OwnedCompletionState*>(user_data)->receipt_count;
  }

  static void Target(void* user_data, iree_async_operation_t* operation,
                     iree_status_t status,
                     iree_async_completion_flags_t flags) {
    auto* self = static_cast<OwnedCompletionState*>(user_data);
    CompletionState::Record(&self->target, operation, status, flags);
    if (self->request.phase != IREE_ASYNC_CANCEL_REQUEST_PHASE_IDLE) {
      iree_async_proactor_cancel_request_target_retired(self->proactor,
                                                        &self->request);
    }
  }
};

class IocpCancelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_async_proactor_create_iocp(iree_async_proactor_options_default(),
                                        iree_allocator_system(), &proactor_));
  }

  void TearDown() override { iree_async_proactor_release(proactor_); }

  iree_async_proactor_iocp_t* iocp() {
    return iree_async_proactor_iocp_cast(proactor_);
  }

  // Proactor under test.
  iree_async_proactor_t* proactor_ = nullptr;
};

TEST(IocpWaitCancellationTest, NativeFailureLeavesReachabilityUnresolved) {
  iree_async_proactor_iocp_t proactor = {};
  proactor.nt_wait_api.available = true;
  proactor.nt_wait_api.NtCancelWaitCompletionPacket = ReturnCancellationFailure;
  iree_async_iocp_carrier_t carrier = {};
  carrier.data.event_wait.wait_handle = reinterpret_cast<HANDLE>(1);
  iree_async_iocp_wait_cancel_result_t result =
      IREE_ASYNC_IOCP_WAIT_CANCEL_WITHDRAWN;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_async_proactor_iocp_cancel_wait(&proactor, &carrier, &result));

  EXPECT_EQ(result, IREE_ASYNC_IOCP_WAIT_CANCEL_UNRESOLVED);
  EXPECT_EQ(carrier.data.event_wait.wait_handle, reinterpret_cast<HANDLE>(1));
}

TEST(IocpWaitCancellationTest, NativeWithdrawalClosesRegistration) {
  HANDLE wait_handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  ASSERT_NE(wait_handle, nullptr);
  iree_async_proactor_iocp_t proactor = {};
  proactor.nt_wait_api.available = true;
  proactor.nt_wait_api.NtCancelWaitCompletionPacket = ReturnWithdrawal;
  iree_async_iocp_carrier_t carrier = {};
  carrier.data.event_wait.wait_handle = wait_handle;
  iree_async_iocp_wait_cancel_result_t result =
      IREE_ASYNC_IOCP_WAIT_CANCEL_UNRESOLVED;

  IREE_ASSERT_OK(
      iree_async_proactor_iocp_cancel_wait(&proactor, &carrier, &result));

  EXPECT_EQ(result, IREE_ASYNC_IOCP_WAIT_CANCEL_WITHDRAWN);
  EXPECT_EQ(carrier.data.event_wait.wait_handle, nullptr);
}

TEST(IocpWaitCancellationTest, CloseFailurePreservesPublishedResult) {
  iree_async_proactor_iocp_t proactor = {};
  proactor.nt_wait_api.available = true;
  proactor.nt_wait_api.NtCancelWaitCompletionPacket = ReturnPublishedCompletion;
  iree_async_iocp_carrier_t carrier = {};
  carrier.data.event_wait.wait_handle = INVALID_HANDLE_VALUE;
  iree_async_iocp_wait_cancel_result_t result =
      IREE_ASYNC_IOCP_WAIT_CANCEL_UNRESOLVED;

  iree_status_t status =
      iree_async_proactor_iocp_cancel_wait(&proactor, &carrier, &result);

  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_free(status);
  EXPECT_EQ(result, IREE_ASYNC_IOCP_WAIT_CANCEL_PUBLISHED);
  EXPECT_EQ(carrier.data.event_wait.wait_handle, INVALID_HANDLE_VALUE);
}

TEST(IocpWaitCancellationTest, LegacyUnregisterFailureIsUnresolved) {
  iree_async_proactor_iocp_t proactor = {};
  iree_async_iocp_carrier_t carrier = {};
  carrier.data.event_wait.wait_handle = INVALID_HANDLE_VALUE;
  iree_async_iocp_wait_cancel_result_t result =
      IREE_ASYNC_IOCP_WAIT_CANCEL_WITHDRAWN;

  iree_status_t status =
      iree_async_proactor_iocp_cancel_wait(&proactor, &carrier, &result);

  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_free(status);
  EXPECT_EQ(result, IREE_ASYNC_IOCP_WAIT_CANCEL_UNRESOLVED);
  EXPECT_EQ(carrier.data.event_wait.wait_handle, INVALID_HANDLE_VALUE);
}

TEST(IocpWaitCancellationTest, DestructionRetainsUnresolvedCarrier) {
  iree_async_proactor_iocp_t proactor = {};
  proactor.nt_wait_api.available = true;
  proactor.nt_wait_api.NtCancelWaitCompletionPacket = ReturnCancellationFailure;
  iree_async_operation_t operation = {};
  iree_async_iocp_carrier_t carrier = {};
  carrier.type = IREE_ASYNC_IOCP_CARRIER_EVENT_WAIT;
  carrier.operation = &operation;
  carrier.data.event_wait.wait_handle = reinterpret_cast<HANDLE>(1);
  operation.next = reinterpret_cast<iree_async_operation_t*>(&carrier);
  proactor.active_carriers = &carrier;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_async_proactor_iocp_deinitialize_waits(&proactor));

  EXPECT_EQ(proactor.active_carriers, &carrier);
  EXPECT_EQ(operation.next,
            reinterpret_cast<iree_async_operation_t*>(&carrier));
}

TEST_F(IocpCancelTest, OrdinaryCancellationFailureRetainsOperation) {
  if (!iocp()->nt_wait_api.available) {
    GTEST_SKIP() << "wait completion packets unavailable";
  }
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));
  CompletionState completion;
  iree_async_event_wait_operation_t operation = {};
  iree_async_operation_initialize(
      &operation.base, IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, CompletionState::Record, &completion);
  operation.event = event;
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &operation.base));
  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr));

  auto native_cancel = iocp()->nt_wait_api.NtCancelWaitCompletionPacket;
  iocp()->nt_wait_api.NtCancelWaitCompletionPacket = ReturnCancellationFailure;
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &operation.base));
  iree_status_t poll_status =
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr);
  iocp()->nt_wait_api.NtCancelWaitCompletionPacket = native_cancel;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, poll_status);
  EXPECT_EQ(completion.count, 0);
  ASSERT_NE(iocp()->active_carriers, nullptr);
  EXPECT_EQ(operation.base.next,
            reinterpret_cast<iree_async_operation_t*>(iocp()->active_carriers));
  EXPECT_EQ(iree_atomic_load(&iocp()->pending_event_wait_cancellation_count,
                             iree_memory_order_acquire),
            1);

  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr));
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(completion.code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(iocp()->active_carriers, nullptr);
  iree_async_event_release(event);
}

TEST_F(IocpCancelTest, OwnedCancellationFailureRetainsRequestAndTarget) {
  if (!iocp()->nt_wait_api.available) {
    GTEST_SKIP() << "wait completion packets unavailable";
  }
  iree_async_event_native_t event = {};
  IREE_ASSERT_OK(iree_async_event_native_initialize(&event));
  OwnedCompletionState completion;
  completion.proactor = proactor_;
  iree_async_cancel_request_initialize(
      {OwnedCompletionState::Receipt, &completion}, &completion.request);
  iree_async_handle_poll_operation_t operation = {};
  iree_async_operation_initialize(&operation.base,
                                  IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  OwnedCompletionState::Target, &completion);
  operation.primitive = event.wait_primitive;
  operation.events = IREE_ASYNC_POLL_EVENT_IN;
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &operation.base));
  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr));

  auto native_cancel = iocp()->nt_wait_api.NtCancelWaitCompletionPacket;
  iocp()->nt_wait_api.NtCancelWaitCompletionPacket = ReturnCancellationFailure;
  IREE_ASSERT_OK(iree_async_proactor_request_cancel(proactor_, &operation.base,
                                                    &completion.request));
  iree_status_t poll_status =
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr);
  iocp()->nt_wait_api.NtCancelWaitCompletionPacket = native_cancel;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, poll_status);
  EXPECT_EQ(completion.receipt_count, 0);
  EXPECT_EQ(completion.target.count, 0);
  EXPECT_EQ(completion.request.phase, IREE_ASYNC_CANCEL_REQUEST_PHASE_QUEUED);
  ASSERT_NE(iocp()->active_carriers, nullptr);
  EXPECT_EQ(operation.base.next,
            reinterpret_cast<iree_async_operation_t*>(iocp()->active_carriers));

  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr));
  EXPECT_EQ(completion.receipt_count, 1);
  EXPECT_EQ(completion.target.count, 1);
  EXPECT_EQ(completion.target.code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(completion.request.phase, IREE_ASYNC_CANCEL_REQUEST_PHASE_IDLE);
  EXPECT_EQ(iocp()->active_carriers, nullptr);
  iree_async_event_native_deinitialize(&event);
}

}  // namespace
