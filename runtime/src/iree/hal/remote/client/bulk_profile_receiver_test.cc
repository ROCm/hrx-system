// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_profile_receiver.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/hal/remote/client/bulk.h"
#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/channel/bulk/transfer_table.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct RecordedProfileCallback {
  // Profile callback type delivered to the recording sink.
  iree_hal_remote_profile_callback_type_t callback_type = 0;

  // Event ID copied from the callback metadata.
  uint64_t event_id = 0;

  // Flattened payload bytes delivered with the callback.
  std::vector<uint8_t> payload;

  // Address of the first payload byte delivered to the sink, or 0 if empty.
  uintptr_t payload_address = 0;
};

typedef struct recording_profile_sink_t {
  // HAL resource header for iree_hal_profile_sink_t lifetime management.
  iree_hal_resource_t resource;

  // Callback records appended by sink vtable methods.
  std::vector<RecordedProfileCallback>* records;

  // Status returned by write callbacks.
  iree_status_code_t write_status_code;
} recording_profile_sink_t;

static recording_profile_sink_t* recording_profile_sink_cast(
    iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<recording_profile_sink_t*>(sink);
}

static void recording_profile_sink_destroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static void recording_profile_sink_append_record(
    recording_profile_sink_t* sink,
    iree_hal_remote_profile_callback_type_t callback_type,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t payload_count, const iree_const_byte_span_t* payloads) {
  RecordedProfileCallback record;
  record.callback_type = callback_type;
  record.event_id = metadata->event_id;
  if (payload_count > 0) {
    record.payload_address = reinterpret_cast<uintptr_t>(payloads[0].data);
  }
  for (iree_host_size_t i = 0; i < payload_count; ++i) {
    const auto* payload_data = static_cast<const uint8_t*>(payloads[i].data);
    record.payload.insert(record.payload.end(), payload_data,
                          payload_data + payloads[i].data_length);
  }
  sink->records->push_back(std::move(record));
}

static iree_status_t recording_profile_sink_begin_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  recording_profile_sink_append_record(
      recording_profile_sink_cast(sink),
      IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_BEGIN_SESSION, metadata,
      /*payload_count=*/0, /*payloads=*/NULL);
  return iree_ok_status();
}

static iree_status_t recording_profile_sink_write(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t payload_count, const iree_const_byte_span_t* payloads) {
  recording_profile_sink_t* recording_sink = recording_profile_sink_cast(sink);
  if (recording_sink->write_status_code != IREE_STATUS_OK) {
    return iree_status_from_code(recording_sink->write_status_code);
  }
  recording_profile_sink_append_record(
      recording_sink, IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK,
      metadata, payload_count, payloads);
  return iree_ok_status();
}

static iree_status_t recording_profile_sink_end_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)session_status_code;
  recording_profile_sink_append_record(
      recording_profile_sink_cast(sink),
      IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_END_SESSION, metadata,
      /*payload_count=*/0, /*payloads=*/NULL);
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kRecordingProfileSinkVTable = {
    /*.destroy=*/recording_profile_sink_destroy,
    /*.begin_session=*/recording_profile_sink_begin_session,
    /*.write=*/recording_profile_sink_write,
    /*.end_session=*/recording_profile_sink_end_session,
};

static void recording_profile_sink_initialize(
    std::vector<RecordedProfileCallback>* records,
    recording_profile_sink_t* out_sink) {
  memset(out_sink, 0, sizeof(*out_sink));
  iree_hal_resource_initialize(&kRecordingProfileSinkVTable,
                               &out_sink->resource);
  out_sink->records = records;
  out_sink->write_status_code = IREE_STATUS_OK;
}

static iree_hal_profile_sink_t* recording_profile_sink_as_base(
    recording_profile_sink_t* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

struct CapturedBulkSend {
  // Concatenated scatter-gather bytes submitted through the endpoint.
  std::vector<uint8_t> data;

  // Total byte length reported to frame-sender completion.
  iree_host_size_t total_length = 0;

  // Frame-sender context pointer forwarded as endpoint user data.
  uint64_t endpoint_user_data = 0;

  // True after the test completes the send.
  bool completed = false;
};

struct MockEndpoint {
  // Endpoint callbacks installed by the bulk channel.
  iree_net_message_endpoint_callbacks_t callbacks = {};

  // Sends captured at the endpoint boundary.
  std::vector<CapturedBulkSend> sends;

  // Status returned by the next endpoint send.
  iree_status_code_t next_send_error = IREE_STATUS_OK;

  // True after endpoint activation.
  bool activated = false;

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    static_cast<MockEndpoint*>(self)->callbacks = callbacks;
  }

  static iree_status_t Activate(void* self) {
    static_cast<MockEndpoint*>(self)->activated = true;
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data) {
    static_cast<MockEndpoint*>(self)->activated = false;
    if (callback) callback(user_data);
    return iree_ok_status();
  }

  static iree_status_t Send(
      void* self, const iree_net_message_endpoint_send_params_t* params) {
    MockEndpoint* endpoint = static_cast<MockEndpoint*>(self);
    if (endpoint->next_send_error != IREE_STATUS_OK) {
      iree_status_code_t send_error = endpoint->next_send_error;
      endpoint->next_send_error = IREE_STATUS_OK;
      return iree_status_from_code(send_error);
    }

    CapturedBulkSend captured;
    captured.endpoint_user_data = params->user_data;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      if (!iree_host_size_checked_add(captured.total_length,
                                      params->data.values[i].length,
                                      &captured.total_length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "captured send size overflow");
      }
    }
    captured.data.reserve(captured.total_length);
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      iree_async_span_t span = params->data.values[i];
      const uint8_t* span_data = iree_async_span_ptr(span);
      captured.data.insert(captured.data.end(), span_data,
                           span_data + span.length);
    }
    endpoint->sends.push_back(std::move(captured));
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    (void)self;
    return {1024 * 1024, 64};
  }

  static iree_status_t BeginSend(void* self, iree_host_size_t size,
                                 void** out_ptr,
                                 iree_net_carrier_send_handle_t* out_handle) {
    (void)self;
    (void)size;
    (void)out_ptr;
    (void)out_handle;
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t CommitSend(void* self,
                                  iree_net_carrier_send_handle_t handle) {
    (void)self;
    (void)handle;
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static void AbortSend(void* self, iree_net_carrier_send_handle_t handle) {
    (void)self;
    (void)handle;
  }

  static const iree_net_message_endpoint_vtable_t* VTable() {
    static const iree_net_message_endpoint_vtable_t vtable = {
        MockEndpoint::SetCallbacks,    MockEndpoint::Activate,
        MockEndpoint::Deactivate,      MockEndpoint::Send,
        MockEndpoint::QuerySendBudget, MockEndpoint::BeginSend,
        MockEndpoint::CommitSend,      MockEndpoint::AbortSend,
    };
    return &vtable;
  }

  iree_net_message_endpoint_t as_endpoint() { return {this, VTable()}; }

  void CompleteSend(iree_host_size_t send_index, iree_status_t status) {
    CapturedBulkSend& captured = sends[send_index];
    captured.completed = true;
    iree_net_frame_sender_dispatch_carrier_completion(
        /*callback_user_data=*/NULL, captured.endpoint_user_data, status,
        captured.total_length, /*recv_lease=*/NULL);
  }

  void CompleteAllSends() {
    for (iree_host_size_t i = 0; i < sends.size(); ++i) {
      if (!sends[i].completed) CompleteSend(i, iree_ok_status());
    }
  }
};

struct BulkChannelCallbacks {
  // Send-completion operation user data values observed by the channel.
  std::vector<uint64_t> send_completions;

  // Send-completion status codes observed by the channel.
  std::vector<iree_status_code_t> send_completion_status_codes;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_size,
                               iree_net_bulk_frame_flags_t flags) {
    (void)user_data;
    (void)transfer_id;
    (void)total_size;
    (void)flags;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected START");
  }

  static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                              uint64_t chunk_offset, uint32_t sequence,
                              iree_net_bulk_frame_flags_t flags,
                              iree_const_byte_span_t chunk_data,
                              iree_async_buffer_lease_t* lease) {
    (void)user_data;
    (void)transfer_id;
    (void)chunk_offset;
    (void)sequence;
    (void)flags;
    (void)chunk_data;
    (void)lease;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected DATA");
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    (void)user_data;
    (void)transfer_id;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected COMPLETE");
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t abort_data,
                               iree_async_buffer_lease_t* lease) {
    (void)user_data;
    (void)transfer_id;
    (void)abort_data;
    (void)lease;
    return iree_make_status(IREE_STATUS_INTERNAL, "unexpected ABORT");
  }

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    BulkChannelCallbacks* callbacks =
        static_cast<BulkChannelCallbacks*>(user_data);
    callbacks->send_completions.push_back(operation_user_data);
    callbacks->send_completion_status_codes.push_back(iree_status_code(status));
    iree_status_free(status);
  }

  iree_net_bulk_channel_callbacks_t MakeCallbacks() {
    iree_net_bulk_channel_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_start = OnStart;
    callbacks.on_data = OnData;
    callbacks.on_complete = OnComplete;
    callbacks.on_abort = OnAbort;
    callbacks.on_send_complete = OnSendComplete;
    callbacks.user_data = this;
    return callbacks;
  }
};

static iree_net_bulk_frame_header_t ParseBulkHeader(
    const CapturedBulkSend& send) {
  iree_net_bulk_frame_header_t header;
  memset(&header, 0, sizeof(header));
  memcpy(&header, send.data.data(), sizeof(header));
  return header;
}

static iree_status_t BuildProfilePayload(
    uint64_t sequence, iree_hal_remote_profile_callback_type_t callback_type,
    iree_const_byte_span_t callback_payload,
    std::vector<uint8_t>* out_payload) {
  out_payload->clear();

  iree_host_size_t content_type_offset = 0;
  iree_host_size_t name_offset = 0;
  iree_host_size_t payload_offset = 0;
  iree_host_size_t required_length = 0;
  const iree_string_view_t content_type = IREE_SV("x");
  const iree_string_view_t name = IREE_SV("y");
  iree_status_t status = iree_hal_remote_profile_transfer_layout(
      (uint16_t)content_type.size, (uint16_t)name.size,
      callback_payload.data_length, &required_length, &content_type_offset,
      &name_offset, &payload_offset);
  if (iree_status_is_ok(status)) {
    out_payload->resize(required_length);
    memset(out_payload->data(), 0, out_payload->size());

    iree_hal_remote_profile_transfer_header_t header;
    memset(&header, 0, sizeof(header));
    header.sequence = sequence;
    header.session_id = 1;
    header.stream_id = 2;
    header.event_id = sequence;
    header.payload_length = callback_payload.data_length;
    header.physical_device_ordinal = UINT32_MAX;
    header.queue_ordinal = UINT32_MAX;
    header.content_type_length = (uint16_t)content_type.size;
    header.name_length = (uint16_t)name.size;
    header.callback_type = callback_type;
    memcpy(out_payload->data(), &header, sizeof(header));
    memcpy(out_payload->data() + content_type_offset, content_type.data,
           content_type.size);
    memcpy(out_payload->data() + name_offset, name.data, name.size);
    memcpy(out_payload->data() + payload_offset, callback_payload.data,
           callback_payload.data_length);
  }
  return status;
}

class ClientBulkProfileReceiverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&device_, 0, sizeof(device_));
    device_.host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_session_initialize(
        device_.host_allocator, &device_.bulk_session));
    IREE_ASSERT_OK(
        iree_hal_remote_client_device_initialize_bulk_transfers(&device_));

    iree_async_buffer_pool_t* header_pool = NULL;
    IREE_ASSERT_OK(iree_hal_remote_create_queue_header_pool(
        /*buffer_count=*/16, /*buffer_size=*/128, iree_allocator_system(),
        &header_pool));
    IREE_ASSERT_OK(iree_net_bulk_channel_create(
        endpoint_.as_endpoint(), /*options=*/NULL, header_pool,
        bulk_callbacks_.MakeCallbacks(), iree_allocator_system(),
        &bulk_channel_));
    IREE_ASSERT_OK(iree_net_bulk_channel_activate(bulk_channel_));

    recording_profile_sink_initialize(&profile_records_, &profile_sink_);
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_begin_profile_session(
        &device_, recording_profile_sink_as_base(&profile_sink_)));
  }

  void TearDown() override {
    endpoint_.CompleteAllSends();
    if (bulk_channel_) {
      iree_net_bulk_channel_detach(bulk_channel_);
      iree_net_bulk_channel_release(bulk_channel_);
      bulk_channel_ = NULL;
    }
    if (iree_hal_remote_client_bulk_has_profile_session(&device_)) {
      iree_hal_remote_client_bulk_end_profile_session(&device_);
    }
    iree_hal_remote_client_device_deinitialize_bulk_transfers(&device_);
    iree_hal_remote_client_bulk_session_deinitialize(&device_.bulk_session);
  }

  void BeginTransfer(uint64_t transfer_id,
                     const std::vector<uint8_t>& payload) {
    iree_slim_mutex_lock(&device_.bulk_session.transfer_mutex);
    iree_status_t status =
        iree_hal_remote_client_bulk_profile_receiver_begin_locked(
            &device_, transfer_id, payload.size());
    iree_slim_mutex_unlock(&device_.bulk_session.transfer_mutex);
    IREE_ASSERT_OK(status);
  }

  void ReceivePayload(uint64_t transfer_id,
                      const std::vector<uint8_t>& payload) {
    bool handled = false;
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_profile_receiver_on_data(
        &device_, bulk_channel_, transfer_id, /*chunk_offset=*/0,
        IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
        iree_make_const_byte_span(payload.data(), payload.size()), &handled));
    ASSERT_TRUE(handled);
  }

  void CompleteTransfer(uint64_t transfer_id) {
    bool handled = false;
    IREE_ASSERT_OK(iree_hal_remote_client_bulk_profile_receiver_on_complete(
        &device_, bulk_channel_, transfer_id, &handled));
    ASSERT_TRUE(handled);
  }

  std::vector<uint8_t> BuildWritePayload(uint64_t sequence,
                                         std::vector<uint8_t> bytes) {
    std::vector<uint8_t> payload;
    IREE_EXPECT_OK(BuildProfilePayload(
        sequence, IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK,
        iree_make_const_byte_span(bytes.data(), bytes.size()), &payload));
    return payload;
  }

  iree_hal_remote_client_device_t device_;
  MockEndpoint endpoint_;
  BulkChannelCallbacks bulk_callbacks_;
  iree_net_bulk_channel_t* bulk_channel_ = NULL;
  std::vector<RecordedProfileCallback> profile_records_;
  recording_profile_sink_t profile_sink_;
};

TEST_F(ClientBulkProfileReceiverTest,
       CompletedProfilesDispatchInSequenceOrder) {
  constexpr uint64_t kTransfer2 = 200;
  constexpr uint64_t kTransfer1 = 100;
  std::vector<uint8_t> payload2 = BuildWritePayload(/*sequence=*/2, {2});
  std::vector<uint8_t> payload1 = BuildWritePayload(/*sequence=*/1, {1});

  BeginTransfer(kTransfer2, payload2);
  ReceivePayload(kTransfer2, payload2);
  CompleteTransfer(kTransfer2);
  EXPECT_TRUE(profile_records_.empty());

  BeginTransfer(kTransfer1, payload1);
  ReceivePayload(kTransfer1, payload1);
  CompleteTransfer(kTransfer1);

  ASSERT_EQ(profile_records_.size(), 2u);
  EXPECT_EQ(profile_records_[0].event_id, 1u);
  EXPECT_EQ(profile_records_[0].payload_address %
                IREE_HAL_REMOTE_PROFILE_PAYLOAD_ALIGNMENT,
            0u);
  ASSERT_EQ(profile_records_[0].payload.size(), 1u);
  EXPECT_EQ(profile_records_[0].payload[0], 1u);
  EXPECT_EQ(profile_records_[1].event_id, 2u);
  ASSERT_EQ(profile_records_[1].payload.size(), 1u);
  EXPECT_EQ(profile_records_[1].payload[0], 2u);

  ASSERT_EQ(endpoint_.sends.size(), 4u);
  iree_net_bulk_frame_header_t header0 = ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(header0));
  EXPECT_EQ(header0.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  iree_net_bulk_frame_header_t header1 = ParseBulkHeader(endpoint_.sends[1]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(header1));
  EXPECT_EQ(header1.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  iree_net_bulk_frame_header_t header2 = ParseBulkHeader(endpoint_.sends[2]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(header2));
  EXPECT_EQ(header2.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
  EXPECT_EQ(header2.transfer_id, kTransfer1);
  iree_net_bulk_frame_header_t header3 = ParseBulkHeader(endpoint_.sends[3]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(header3));
  EXPECT_EQ(header3.type, IREE_NET_BULK_FRAME_TYPE_COMPLETE);
  EXPECT_EQ(header3.transfer_id, kTransfer2);

  endpoint_.CompleteAllSends();
  ASSERT_EQ(bulk_callbacks_.send_completion_status_codes.size(), 4u);
  for (iree_status_code_t status_code :
       bulk_callbacks_.send_completion_status_codes) {
    EXPECT_EQ(status_code, IREE_STATUS_OK);
  }
}

TEST_F(ClientBulkProfileReceiverTest, RejectsOutOfRangeData) {
  constexpr uint64_t kTransferId = 100;
  std::vector<uint8_t> payload = BuildWritePayload(/*sequence=*/1, {1, 2});
  BeginTransfer(kTransferId, payload);

  bool handled = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_remote_client_bulk_profile_receiver_on_data(
          &device_, bulk_channel_, kTransferId,
          /*chunk_offset=*/payload.size() - 1,
          IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK,
          iree_make_const_byte_span(payload.data(), 2), &handled));
  EXPECT_TRUE(handled);
  EXPECT_TRUE(endpoint_.sends.empty());
}

TEST_F(ClientBulkProfileReceiverTest, SinkFailureSendsAbort) {
  constexpr uint64_t kTransferId = 100;
  profile_sink_.write_status_code = IREE_STATUS_DATA_LOSS;
  std::vector<uint8_t> payload = BuildWritePayload(/*sequence=*/1, {1});

  BeginTransfer(kTransferId, payload);
  ReceivePayload(kTransferId, payload);
  CompleteTransfer(kTransferId);

  EXPECT_TRUE(profile_records_.empty());
  ASSERT_EQ(endpoint_.sends.size(), 2u);
  iree_net_bulk_frame_header_t credit_header =
      ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(credit_header));
  EXPECT_EQ(credit_header.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
  iree_net_bulk_frame_header_t abort_header =
      ParseBulkHeader(endpoint_.sends[1]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(abort_header));
  EXPECT_EQ(abort_header.type, IREE_NET_BULK_FRAME_TYPE_ABORT);
  EXPECT_EQ(abort_header.transfer_id, kTransferId);
}

TEST_F(ClientBulkProfileReceiverTest, CompleteReturnsAckSendFailure) {
  constexpr uint64_t kTransferId = 100;
  std::vector<uint8_t> payload = BuildWritePayload(/*sequence=*/1, {1});

  BeginTransfer(kTransferId, payload);
  ReceivePayload(kTransferId, payload);
  endpoint_.next_send_error = IREE_STATUS_UNAVAILABLE;

  bool handled = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_remote_client_bulk_profile_receiver_on_complete(
          &device_, bulk_channel_, kTransferId, &handled));

  EXPECT_TRUE(handled);
  ASSERT_EQ(profile_records_.size(), 1u);
  EXPECT_EQ(profile_records_[0].event_id, 1u);
  ASSERT_EQ(endpoint_.sends.size(), 1u);
  iree_net_bulk_frame_header_t credit_header =
      ParseBulkHeader(endpoint_.sends[0]);
  IREE_EXPECT_OK(iree_net_bulk_frame_header_validate(credit_header));
  EXPECT_EQ(credit_header.type, IREE_NET_BULK_FRAME_TYPE_CREDIT);
}

TEST_F(ClientBulkProfileReceiverTest, EndSessionDropsPendingProfileTransfers) {
  constexpr uint64_t kTransferId = 200;
  std::vector<uint8_t> payload = BuildWritePayload(/*sequence=*/2, {2});

  BeginTransfer(kTransferId, payload);
  ReceivePayload(kTransferId, payload);
  CompleteTransfer(kTransferId);
  EXPECT_TRUE(profile_records_.empty());
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            1u);

  iree_hal_remote_client_bulk_end_profile_session(&device_);

  EXPECT_FALSE(iree_hal_remote_client_bulk_has_profile_session(&device_));
  EXPECT_EQ(iree_net_bulk_transfer_table_count(device_.bulk_session.transfers),
            0u);
}

}  // namespace
