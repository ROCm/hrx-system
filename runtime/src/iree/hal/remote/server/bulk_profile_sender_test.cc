// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_profile_sender.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/server/bulk.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/bulk_test_util.h"
#include "iree/hal/remote/server/profile_relay.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/channel/util/sequence_window.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::remote::server::testing::MockCarrier;
using iree::hal::remote::server::testing::MockEndpoint;
using iree::hal::remote::server::testing::ParseBulkHeader;
using iree::hal::remote::server::testing::TestBufferPool;

typedef struct test_profile_sink_t {
  // Resource header for iree_hal_profile_sink_t lifetime management.
  iree_hal_resource_t resource;

  // Host allocator used to release this sink.
  iree_allocator_t host_allocator;

  // Counter incremented when the sink is destroyed.
  int* destroy_count;
} test_profile_sink_t;

static test_profile_sink_t* test_profile_sink_cast(
    iree_hal_profile_sink_t* base_sink) {
  return reinterpret_cast<test_profile_sink_t*>(base_sink);
}

static void test_profile_sink_destroy(iree_hal_profile_sink_t* base_sink) {
  test_profile_sink_t* sink = test_profile_sink_cast(base_sink);
  ++*sink->destroy_count;
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t test_profile_sink_begin_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)base_sink;
  (void)metadata;
  return iree_ok_status();
}

static iree_status_t test_profile_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  (void)base_sink;
  (void)metadata;
  (void)iovec_count;
  (void)iovecs;
  return iree_ok_status();
}

static iree_status_t test_profile_sink_end_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)base_sink;
  (void)metadata;
  (void)session_status_code;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t test_profile_sink_vtable = {
    /*.destroy=*/test_profile_sink_destroy,
    /*.begin_session=*/test_profile_sink_begin_session,
    /*.write=*/test_profile_sink_write,
    /*.end_session=*/test_profile_sink_end_session,
};

static iree_status_t test_profile_sink_create(
    iree_allocator_t host_allocator, int* destroy_count,
    iree_hal_profile_sink_t** out_sink) {
  *out_sink = NULL;
  test_profile_sink_t* sink = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink);
  if (iree_status_is_ok(status)) {
    memset(sink, 0, sizeof(*sink));
    iree_hal_resource_initialize(&test_profile_sink_vtable, &sink->resource);
    sink->host_allocator = host_allocator;
    sink->destroy_count = destroy_count;
    *out_sink = reinterpret_cast<iree_hal_profile_sink_t*>(sink);
  }
  return status;
}

static iree_status_t AllocateProfilePayload(uint64_t sequence,
                                            iree_host_size_t body_length,
                                            iree_allocator_t host_allocator,
                                            iree_byte_span_t* out_payload) {
  *out_payload = iree_byte_span_empty();
  const iree_host_size_t total_length =
      sizeof(iree_hal_remote_profile_transfer_header_t) + body_length;
  void* storage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_length, &storage);
  if (iree_status_is_ok(status)) {
    memset(storage, 0, total_length);
    iree_hal_remote_profile_transfer_header_t header = {};
    header.sequence = sequence;
    header.session_id = 1;
    header.payload_length = body_length;
    header.physical_device_ordinal = UINT32_MAX;
    header.queue_ordinal = UINT32_MAX;
    header.callback_type = IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK;
    memcpy(storage, &header, sizeof(header));
    uint8_t* body = static_cast<uint8_t*>(storage) + sizeof(header);
    for (iree_host_size_t i = 0; i < body_length; ++i) {
      body[i] = static_cast<uint8_t>(sequence + i);
    }
    *out_payload = iree_make_byte_span(storage, total_length);
  }
  return status;
}

struct ProfileBulkCallbacks {
  // Session slot receiving bulk-channel callbacks.
  iree_hal_remote_server_session_t* session_slot = nullptr;

  // Send completion operation user data values observed by the bulk channel.
  std::vector<uint64_t> send_completions;

  // Send completion status codes observed by the bulk channel.
  std::vector<iree_status_code_t> send_completion_errors;

  // Transport error status codes observed by the bulk channel.
  std::vector<iree_status_code_t> transport_errors;

  // Credit replenishment deltas observed by the bulk channel.
  std::vector<uint32_t> credit_deltas;

  // Available credit counts after each replenishment.
  std::vector<uint32_t> available_credit_counts;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_size,
                               iree_net_bulk_frame_flags_t flags) {
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    return iree_hal_remote_server_bulk_on_start(callbacks->session_slot,
                                                transfer_id, total_size, flags);
  }

  static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                              uint64_t chunk_offset, uint32_t sequence,
                              iree_net_bulk_frame_flags_t flags,
                              iree_const_byte_span_t chunk_data,
                              iree_async_buffer_lease_t* lease) {
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    return iree_hal_remote_server_bulk_on_data(
        callbacks->session_slot, transfer_id, chunk_offset, sequence, flags,
        chunk_data, lease);
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    return iree_hal_remote_server_bulk_on_complete(callbacks->session_slot,
                                                   transfer_id);
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t abort_data,
                               iree_async_buffer_lease_t* lease) {
    (void)abort_data;
    (void)lease;
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    return iree_hal_remote_server_bulk_on_abort(callbacks->session_slot,
                                                transfer_id);
  }

  static void OnTransportError(void* user_data, iree_status_t status) {
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    callbacks->transport_errors.push_back(iree_status_code(status));
    iree_status_free(status);
  }

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    callbacks->send_completions.push_back(operation_user_data);
    callbacks->send_completion_errors.push_back(iree_status_code(status));
    iree_hal_remote_server_bulk_on_send_complete(callbacks->session_slot,
                                                 operation_user_data, status);
  }

  static void OnCredit(void* user_data, uint32_t credit_delta,
                       uint32_t available_credit_count) {
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    callbacks->credit_deltas.push_back(credit_delta);
    callbacks->available_credit_counts.push_back(available_credit_count);
    iree_hal_remote_server_bulk_on_credit(callbacks->session_slot);
  }

  iree_net_bulk_channel_callbacks_t MakeCallbacks() {
    iree_net_bulk_channel_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_start = OnStart;
    callbacks.on_data = OnData;
    callbacks.on_complete = OnComplete;
    callbacks.on_abort = OnAbort;
    callbacks.on_transport_error = OnTransportError;
    callbacks.on_send_complete = OnSendComplete;
    callbacks.on_credit = OnCredit;
    callbacks.user_data = this;
    return callbacks;
  }
};

class BulkProfileSenderTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kSessionId = 1;
  static constexpr iree_host_size_t kProfileBodyLength = 4;
  static constexpr iree_host_size_t kProfilePayloadLength =
      sizeof(iree_hal_remote_profile_transfer_header_t) + kProfileBodyLength;

  void SetUp() override {
    memset(&server_, 0, sizeof(server_));
    memset(&session_, 0, sizeof(session_));

    iree_atomic_ref_count_init(&server_.ref_count);
    server_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&server_.session_mutex);

    session_.server = &server_;
    session_.session_id = kSessionId;
    session_.session = reinterpret_cast<iree_net_session_t*>(this);
    iree_hal_remote_server_bulk_session_options_t bulk_options =
        iree_hal_remote_server_bulk_session_options_default();
    bulk_options.active_transfer_capacity = 1;
    IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_allocate(
        &session_, &bulk_options, iree_allocator_system(),
        &session_.bulk_session));

    carrier_ = MockCarrier::Create();
    endpoint_.carrier = carrier_.get();
    channel_callbacks_.session_slot = &session_;
    TestBufferPool buffer_pool;
    IREE_ASSERT_OK(
        buffer_pool.Initialize(/*buffer_count=*/16, /*buffer_size=*/1024));
    IREE_ASSERT_OK(iree_net_bulk_channel_create(
        endpoint_.as_endpoint(), nullptr, buffer_pool.release(),
        channel_callbacks_.MakeCallbacks(), iree_allocator_system(),
        &bulk_channel_));
    IREE_ASSERT_OK(iree_net_bulk_channel_activate(bulk_channel_));
    IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_attach_channel(
        &session_, bulk_channel_));

    IREE_ASSERT_OK(test_profile_sink_create(
        iree_allocator_system(), &sink_destroy_count_, &profile_sink_));
    IREE_ASSERT_OK(iree_hal_remote_server_profile_relay_prepare_begin(
        &session_, profile_sink_));
  }

  void TearDown() override {
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfers =
        nullptr;
    iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
    pending_transfers =
        iree_hal_remote_server_profile_take_pending_transfers_locked(&session_);
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(&session_));
    iree_hal_remote_server_profile_pending_transfer_free_list(
        &server_, pending_transfers);
    iree_hal_remote_server_profile_relay_deinitialize(&session_);
    iree_hal_remote_server_bulk_session_free(session_.bulk_session);
    session_.bulk_session = nullptr;
    iree_net_bulk_channel_release(bulk_channel_);
    bulk_channel_ = nullptr;

    iree_hal_profile_sink_release(profile_sink_);
    profile_sink_ = nullptr;
    iree_slim_mutex_deinitialize(&server_.session_mutex);
  }

  void SubmitProfilePayload(uint64_t sequence) {
    iree_byte_span_t payload = iree_byte_span_empty();
    IREE_ASSERT_OK(AllocateProfilePayload(sequence, kProfileBodyLength,
                                          server_.host_allocator, &payload));
    IREE_ASSERT_OK(iree_hal_remote_server_profile_submit_transfer(
        &session_, kSessionId, profile_sink_, payload));
  }

  uint64_t ExpectStartFrame(iree_host_size_t send_index) {
    iree_net_bulk_frame_header_t header =
        ParseBulkHeader(carrier_->sends[send_index].data);
    EXPECT_EQ(header.type, IREE_NET_BULK_FRAME_TYPE_START);
    EXPECT_EQ(header.flags, IREE_NET_BULK_FRAME_FLAG_NONE);
    EXPECT_EQ(header.total_size, kProfilePayloadLength);
    EXPECT_EQ(header.chunk_length, 0u);
    return header.transfer_id;
  }

  void ExpectDataFrame(iree_host_size_t send_index, uint64_t transfer_id,
                       uint64_t profile_sequence) {
    const std::vector<uint8_t>& send = carrier_->sends[send_index].data;
    iree_net_bulk_frame_header_t header = ParseBulkHeader(send);
    EXPECT_EQ(header.type, IREE_NET_BULK_FRAME_TYPE_DATA);
    EXPECT_EQ(header.flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
    EXPECT_EQ(header.transfer_id, transfer_id);
    EXPECT_EQ(header.chunk_offset, 0u);
    EXPECT_EQ(header.sequence, 0u);
    EXPECT_EQ(header.chunk_length, kProfilePayloadLength);
    ASSERT_EQ(send.size(),
              IREE_NET_BULK_FRAME_HEADER_SIZE + kProfilePayloadLength);

    iree_hal_remote_profile_transfer_header_t profile_header;
    memcpy(&profile_header, send.data() + IREE_NET_BULK_FRAME_HEADER_SIZE,
           sizeof(profile_header));
    EXPECT_EQ(profile_header.sequence, profile_sequence);
    EXPECT_EQ(profile_header.payload_length, kProfileBodyLength);
    EXPECT_EQ(profile_header.callback_type,
              IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK);
  }

  void ExpectCompleteFrame(iree_host_size_t send_index, uint64_t transfer_id) {
    iree_net_bulk_frame_header_t header =
        ParseBulkHeader(carrier_->sends[send_index].data);
    EXPECT_EQ(header.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
    EXPECT_EQ(header.flags, IREE_NET_BULK_FRAME_FLAG_NONE);
    EXPECT_EQ(header.transfer_id, transfer_id);
    EXPECT_EQ(header.chunk_length, 0u);
  }

  void CompleteSend(iree_host_size_t send_index, iree_status_t status) {
    carrier_->CompleteSend(send_index, status);
  }

  bool HasPendingTransfers() const {
    return iree_hal_remote_server_profile_has_pending_transfers_locked(
        &session_);
  }

  iree_net_bulk_transfer_t* LookupTransfer(uint64_t transfer_id) {
    return iree_hal_remote_bulk_transfer_scheduler_lookup(
        iree_hal_remote_server_bulk_session_scheduler(&session_), transfer_id);
  }

  iree_host_size_t ActiveTransferCount() {
    return iree_hal_remote_bulk_transfer_scheduler_count(
        iree_hal_remote_server_bulk_session_scheduler(&session_));
  }

  iree_hal_remote_server_profile_relay_t* profile_relay() {
    return iree_hal_remote_server_bulk_session_profile_relay(&session_);
  }

  // Stack server shell retained by transfer state.
  iree_hal_remote_server_t server_ = {};

  // Stack session shell containing the scheduler under test.
  iree_hal_remote_server_session_t session_ = {};

  // Capturing carrier backing |bulk_channel_|.
  std::unique_ptr<MockCarrier> carrier_;

  // Capturing endpoint backing |bulk_channel_|.
  MockEndpoint endpoint_;

  // Bulk channel callback capture and server routing state.
  ProfileBulkCallbacks channel_callbacks_;

  // Bulk channel used by sender operations.
  iree_net_bulk_channel_t* bulk_channel_ = nullptr;

  // Active profile sink passed to the server profile relay.
  iree_hal_profile_sink_t* profile_sink_ = nullptr;

  // Number of destroyed profile sinks.
  int sink_destroy_count_ = 0;
};

TEST_F(BulkProfileSenderTest,
       QueuedCallbacksDrainAfterActiveTransferCompletes) {
  SubmitProfilePayload(/*sequence=*/1);
  ASSERT_EQ(carrier_->sends.size(), 1u);
  uint64_t first_transfer_id = ExpectStartFrame(/*send_index=*/0);

  SubmitProfilePayload(/*sequence=*/2);
  EXPECT_TRUE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 1u);
  EXPECT_EQ(carrier_->sends.size(), 1u);

  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  EXPECT_EQ(carrier_->sends.size(), 1u);

  CompleteSend(/*send_index=*/0, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 2u);
  ExpectDataFrame(/*send_index=*/1, first_transfer_id,
                  /*profile_sequence=*/1);

  CompleteSend(/*send_index=*/1, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 3u);
  ExpectCompleteFrame(/*send_index=*/2, first_transfer_id);

  CompleteSend(/*send_index=*/2, iree_ok_status());
  EXPECT_TRUE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 1u);

  IREE_EXPECT_OK(
      iree_hal_remote_server_bulk_on_complete(&session_, first_transfer_id));
  ASSERT_EQ(carrier_->sends.size(), 4u);
  uint64_t second_transfer_id = ExpectStartFrame(/*send_index=*/3);
  EXPECT_NE(second_transfer_id, first_transfer_id);
  EXPECT_FALSE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 1u);
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            1u);

  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  CompleteSend(/*send_index=*/3, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 5u);
  ExpectDataFrame(/*send_index=*/4, second_transfer_id,
                  /*profile_sequence=*/2);
  CompleteSend(/*send_index=*/4, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 6u);
  ExpectCompleteFrame(/*send_index=*/5, second_transfer_id);
  CompleteSend(/*send_index=*/5, iree_ok_status());

  IREE_EXPECT_OK(
      iree_hal_remote_server_bulk_on_complete(&session_, second_transfer_id));
  EXPECT_FALSE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 0u);
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            2u);
}

TEST_F(BulkProfileSenderTest, SendFailureDrainsQueuedCallback) {
  SubmitProfilePayload(/*sequence=*/1);
  ASSERT_EQ(carrier_->sends.size(), 1u);
  uint64_t first_transfer_id = ExpectStartFrame(/*send_index=*/0);

  SubmitProfilePayload(/*sequence=*/2);
  EXPECT_TRUE(HasPendingTransfers());

  CompleteSend(/*send_index=*/0,
               iree_status_from_code(IREE_STATUS_UNAVAILABLE));
  ASSERT_EQ(channel_callbacks_.send_completion_errors.size(), 1u);
  EXPECT_EQ(channel_callbacks_.send_completion_errors[0],
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(profile_relay()->transfer_failure_sequence, 1u);
  EXPECT_EQ(profile_relay()->transfer_failure_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(LookupTransfer(first_transfer_id), nullptr);
  EXPECT_FALSE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 1u);

  ASSERT_EQ(carrier_->sends.size(), 2u);
  uint64_t second_transfer_id = ExpectStartFrame(/*send_index=*/1);
  EXPECT_NE(second_transfer_id, first_transfer_id);

  CompleteSend(/*send_index=*/1, iree_ok_status());
  EXPECT_EQ(carrier_->sends.size(), 2u);
}

TEST_F(BulkProfileSenderTest, PeerAbortWaitsForPendingDataSendCompletion) {
  SubmitProfilePayload(/*sequence=*/1);
  ASSERT_EQ(carrier_->sends.size(), 1u);
  uint64_t first_transfer_id = ExpectStartFrame(/*send_index=*/0);

  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  CompleteSend(/*send_index=*/0, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 2u);
  ExpectDataFrame(/*send_index=*/1, first_transfer_id,
                  /*profile_sequence=*/1);

  SubmitProfilePayload(/*sequence=*/2);
  EXPECT_TRUE(HasPendingTransfers());

  IREE_EXPECT_OK(
      iree_hal_remote_server_bulk_on_abort(&session_, first_transfer_id));
  EXPECT_EQ(profile_relay()->transfer_failure_sequence, 1u);
  EXPECT_EQ(profile_relay()->transfer_failure_code, IREE_STATUS_ABORTED);
  EXPECT_NE(LookupTransfer(first_transfer_id), nullptr);
  EXPECT_TRUE(HasPendingTransfers());
  EXPECT_EQ(carrier_->sends.size(), 2u);

  CompleteSend(/*send_index=*/1, iree_ok_status());
  EXPECT_EQ(LookupTransfer(first_transfer_id), nullptr);
  EXPECT_FALSE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 1u);
  ASSERT_EQ(carrier_->sends.size(), 3u);
  uint64_t second_transfer_id = ExpectStartFrame(/*send_index=*/2);
  EXPECT_NE(second_transfer_id, first_transfer_id);

  CompleteSend(/*send_index=*/2, iree_ok_status());
  EXPECT_EQ(carrier_->sends.size(), 3u);
}

TEST_F(BulkProfileSenderTest, QueuedCallbackDrainsAfterActiveSinkDetach) {
  SubmitProfilePayload(/*sequence=*/1);
  ASSERT_EQ(carrier_->sends.size(), 1u);
  uint64_t first_transfer_id = ExpectStartFrame(/*send_index=*/0);

  SubmitProfilePayload(/*sequence=*/2);
  EXPECT_TRUE(HasPendingTransfers());

  iree_hal_profile_sink_t* detached_sink =
      iree_hal_remote_server_profile_relay_detach_active_sink(&session_,
                                                              profile_sink_);
  ASSERT_EQ(detached_sink, profile_sink_);
  iree_hal_profile_sink_release(detached_sink);
  EXPECT_EQ(profile_relay()->active_sink, nullptr);

  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  CompleteSend(/*send_index=*/0, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 2u);
  ExpectDataFrame(/*send_index=*/1, first_transfer_id,
                  /*profile_sequence=*/1);
  CompleteSend(/*send_index=*/1, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 3u);
  ExpectCompleteFrame(/*send_index=*/2, first_transfer_id);
  CompleteSend(/*send_index=*/2, iree_ok_status());

  IREE_EXPECT_OK(
      iree_hal_remote_server_bulk_on_complete(&session_, first_transfer_id));
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            1u);

  ASSERT_EQ(carrier_->sends.size(), 4u);
  uint64_t second_transfer_id = ExpectStartFrame(/*send_index=*/3);
  EXPECT_FALSE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 1u);

  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  CompleteSend(/*send_index=*/3, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 5u);
  ExpectDataFrame(/*send_index=*/4, second_transfer_id,
                  /*profile_sequence=*/2);
  CompleteSend(/*send_index=*/4, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 6u);
  ExpectCompleteFrame(/*send_index=*/5, second_transfer_id);
  CompleteSend(/*send_index=*/5, iree_ok_status());

  IREE_EXPECT_OK(
      iree_hal_remote_server_bulk_on_complete(&session_, second_transfer_id));
  EXPECT_EQ(ActiveTransferCount(), 0u);
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            2u);
}

TEST_F(BulkProfileSenderTest, NoBacklogCompletionDoesNotStartMoreWork) {
  SubmitProfilePayload(/*sequence=*/1);
  ASSERT_EQ(carrier_->sends.size(), 1u);
  uint64_t transfer_id = ExpectStartFrame(/*send_index=*/0);

  IREE_ASSERT_OK(endpoint_.InjectCredit(/*credit_delta=*/1));
  CompleteSend(/*send_index=*/0, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 2u);
  ExpectDataFrame(/*send_index=*/1, transfer_id,
                  /*profile_sequence=*/1);
  CompleteSend(/*send_index=*/1, iree_ok_status());
  ASSERT_EQ(carrier_->sends.size(), 3u);
  ExpectCompleteFrame(/*send_index=*/2, transfer_id);
  CompleteSend(/*send_index=*/2, iree_ok_status());
  EXPECT_FALSE(HasPendingTransfers());

  IREE_EXPECT_OK(
      iree_hal_remote_server_bulk_on_complete(&session_, transfer_id));
  EXPECT_FALSE(HasPendingTransfers());
  EXPECT_EQ(ActiveTransferCount(), 0u);
  EXPECT_EQ(carrier_->sends.size(), 3u);
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            1u);
}

}  // namespace
