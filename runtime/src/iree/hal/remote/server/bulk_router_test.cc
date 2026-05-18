// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_router.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct fake_router_context_t {
  // Number of retain callbacks observed.
  int retain_count;

  // Number of release callbacks observed.
  int release_count;

  // Last START transfer ID.
  uint64_t start_transfer_id;

  // Last START transfer total size.
  uint64_t start_total_size;

  // Last START frame flags.
  iree_net_bulk_frame_flags_t start_flags;

  // Last DATA transfer ID.
  uint64_t data_transfer_id;

  // Last DATA chunk offset.
  uint64_t data_chunk_offset;

  // Last DATA sequence number.
  uint32_t data_sequence;

  // Last DATA frame flags.
  iree_net_bulk_frame_flags_t data_flags;

  // Last DATA payload length.
  iree_host_size_t data_length;

  // Last COMPLETE transfer ID.
  uint64_t complete_transfer_id;

  // Last ABORT transfer ID.
  uint64_t abort_transfer_id;

  // Number of credit callbacks observed.
  int credit_count;

  // Number of send-complete callbacks observed.
  int send_complete_count;

  // Last send-complete operation user data.
  uint64_t send_complete_operation_user_data;

  // Last send-complete status code.
  iree_status_code_t send_complete_status_code;

  // Number of transport-error callbacks observed.
  int transport_error_count;

  // Last transport-error status code.
  iree_status_code_t transport_error_status_code;
} fake_router_context_t;

static void FakeRetain(void* user_data) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  ++context->retain_count;
}

static void FakeRelease(void* user_data) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  ++context->release_count;
}

static iree_status_t FakeStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_size,
                               iree_net_bulk_frame_flags_t flags) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  context->start_transfer_id = transfer_id;
  context->start_total_size = total_size;
  context->start_flags = flags;
  return iree_ok_status();
}

static iree_status_t FakeData(void* user_data, uint64_t transfer_id,
                              uint64_t chunk_offset, uint32_t sequence,
                              iree_net_bulk_frame_flags_t flags,
                              iree_const_byte_span_t chunk_data,
                              iree_async_buffer_lease_t* lease) {
  (void)lease;
  auto* context = static_cast<fake_router_context_t*>(user_data);
  context->data_transfer_id = transfer_id;
  context->data_chunk_offset = chunk_offset;
  context->data_sequence = sequence;
  context->data_flags = flags;
  context->data_length = chunk_data.data_length;
  return iree_ok_status();
}

static iree_status_t FakeComplete(void* user_data, uint64_t transfer_id) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  context->complete_transfer_id = transfer_id;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "complete failure");
}

static iree_status_t FakeAbort(void* user_data, uint64_t transfer_id) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  context->abort_transfer_id = transfer_id;
  return iree_ok_status();
}

static void FakeTransportError(void* user_data, iree_status_t status) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  ++context->transport_error_count;
  context->transport_error_status_code = iree_status_code(status);
  iree_status_free(status);
}

static void FakeSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  ++context->send_complete_count;
  context->send_complete_operation_user_data = operation_user_data;
  context->send_complete_status_code = iree_status_code(status);
  iree_status_free(status);
}

static void FakeCredit(void* user_data) {
  auto* context = static_cast<fake_router_context_t*>(user_data);
  ++context->credit_count;
}

static iree_hal_remote_server_bulk_router_operations_t FakeOperations() {
  iree_hal_remote_server_bulk_router_operations_t operations = {};
  operations.retain = FakeRetain;
  operations.release = FakeRelease;
  operations.start = FakeStart;
  operations.data = FakeData;
  operations.complete = FakeComplete;
  operations.abort = FakeAbort;
  operations.transport_error = FakeTransportError;
  operations.send_complete = FakeSendComplete;
  operations.credit = FakeCredit;
  return operations;
}

TEST(BulkRouterTest, RoutesReceiveFrames) {
  fake_router_context_t context = {};
  iree_hal_remote_server_bulk_router_t router;
  iree_hal_remote_server_bulk_router_initialize(FakeOperations(), &context,
                                                &router);
  iree_net_bulk_channel_callbacks_t callbacks =
      iree_hal_remote_server_bulk_router_callbacks(&router);

  IREE_ASSERT_OK(callbacks.on_start(callbacks.user_data, /*transfer_id=*/12,
                                    /*total_size=*/4096,
                                    IREE_NET_BULK_FRAME_FLAG_NONE));
  uint8_t data[5] = {0};
  IREE_ASSERT_OK(callbacks.on_data(
      callbacks.user_data, /*transfer_id=*/12, /*chunk_offset=*/128,
      /*sequence=*/7, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
      iree_make_const_byte_span(data, sizeof(data)), /*lease=*/nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      callbacks.on_complete(callbacks.user_data, /*transfer_id=*/12));
  uint8_t abort_data[3] = {0};
  IREE_ASSERT_OK(callbacks.on_abort(
      callbacks.user_data, /*transfer_id=*/13,
      iree_make_const_byte_span(abort_data, sizeof(abort_data)),
      /*lease=*/nullptr));

  EXPECT_EQ(context.start_transfer_id, 12u);
  EXPECT_EQ(context.start_total_size, 4096u);
  EXPECT_EQ(context.start_flags, IREE_NET_BULK_FRAME_FLAG_NONE);
  EXPECT_EQ(context.data_transfer_id, 12u);
  EXPECT_EQ(context.data_chunk_offset, 128u);
  EXPECT_EQ(context.data_sequence, 7u);
  EXPECT_EQ(context.data_flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(context.data_length, sizeof(data));
  EXPECT_EQ(context.complete_transfer_id, 12u);
  EXPECT_EQ(context.abort_transfer_id, 13u);

  iree_hal_remote_server_bulk_router_deinitialize(&router);
}

TEST(BulkRouterTest, NormalizesCreditCallback) {
  fake_router_context_t context = {};
  iree_hal_remote_server_bulk_router_t router;
  iree_hal_remote_server_bulk_router_initialize(FakeOperations(), &context,
                                                &router);
  iree_net_bulk_channel_callbacks_t callbacks =
      iree_hal_remote_server_bulk_router_callbacks(&router);

  callbacks.on_credit(callbacks.user_data, /*credit_delta=*/4,
                      /*available_credit_count=*/32);

  EXPECT_EQ(context.credit_count, 1);
  iree_hal_remote_server_bulk_router_deinitialize(&router);
}

TEST(BulkRouterTest, SendCompleteFailureRoutesTransportError) {
  fake_router_context_t context = {};
  iree_hal_remote_server_bulk_router_t router;
  iree_hal_remote_server_bulk_router_initialize(FakeOperations(), &context,
                                                &router);
  iree_net_bulk_channel_callbacks_t callbacks =
      iree_hal_remote_server_bulk_router_callbacks(&router);

  callbacks.on_send_complete(
      callbacks.user_data, /*operation_user_data=*/42,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "send failed"));

  EXPECT_EQ(context.retain_count, 1);
  EXPECT_EQ(context.release_count, 1);
  EXPECT_EQ(context.send_complete_count, 1);
  EXPECT_EQ(context.send_complete_operation_user_data, 42u);
  EXPECT_EQ(context.send_complete_status_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(context.transport_error_count, 1);
  EXPECT_EQ(context.transport_error_status_code, IREE_STATUS_UNAVAILABLE);
  iree_hal_remote_server_bulk_router_deinitialize(&router);
}

TEST(BulkRouterTest, CreditSendCompletionDoesNotRouteTransportError) {
  fake_router_context_t context = {};
  iree_hal_remote_server_bulk_router_t router;
  iree_hal_remote_server_bulk_router_initialize(FakeOperations(), &context,
                                                &router);
  iree_net_bulk_channel_callbacks_t callbacks =
      iree_hal_remote_server_bulk_router_callbacks(&router);

  callbacks.on_send_complete(
      callbacks.user_data, /*operation_user_data=*/0,
      iree_make_status(IREE_STATUS_ABORTED, "credit refresh failed"));

  EXPECT_EQ(context.retain_count, 1);
  EXPECT_EQ(context.release_count, 1);
  EXPECT_EQ(context.send_complete_count, 1);
  EXPECT_EQ(context.send_complete_operation_user_data, 0u);
  EXPECT_EQ(context.send_complete_status_code, IREE_STATUS_ABORTED);
  EXPECT_EQ(context.transport_error_count, 0);
  iree_hal_remote_server_bulk_router_deinitialize(&router);
}

}  // namespace
