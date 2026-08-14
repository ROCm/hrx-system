// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/session.h"

#include <cstring>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/server/atomic.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/timestamp.h"
#include "iree/net/channel/queue/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TestBufferPool {
 public:
  TestBufferPool() = default;

  iree_status_t Initialize(iree_host_size_t buffer_count,
                           iree_host_size_t buffer_size) {
    iree_host_size_t total_size = 0;
    if (!iree_host_size_checked_mul(buffer_count, buffer_size, &total_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "test buffer pool size overflow");
    }
    void* memory = malloc(total_size);
    if (!memory) return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    memset(memory, 0, total_size);

    region_ =
        static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
    if (!region_) {
      free(memory);
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    memset(region_, 0, sizeof(*region_));
    iree_atomic_ref_count_init(&region_->ref_count);
    region_->destroy_fn = DestroyRegion;
    region_->base_ptr = memory;
    region_->length = total_size;
    region_->buffer_size = buffer_size;
    region_->buffer_count = static_cast<uint32_t>(buffer_count);

    iree_status_t status =
        iree_async_buffer_pool_create(region_, iree_allocator_system(), &pool_);
    iree_async_region_release(region_);
    region_ = nullptr;
    return status;
  }

  ~TestBufferPool() { iree_async_buffer_pool_release(pool_); }

  iree_async_buffer_pool_t* release() {
    iree_async_buffer_pool_t* pool = pool_;
    pool_ = nullptr;
    return pool;
  }

 private:
  static void DestroyRegion(iree_async_region_t* region) {
    free(region->base_ptr);
    free(region);
  }

  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* pool_ = nullptr;
};

struct CapturedSend {
  std::vector<uint8_t> data;
  uint64_t user_data = 0;
  bool completed = false;
};

struct CapturingEndpoint {
  iree_net_message_endpoint_callbacks_t callbacks = {};
  bool activated = false;
  bool complete_sends_immediately = true;
  iree_status_code_t next_send_error = IREE_STATUS_OK;
  std::vector<CapturedSend> sends;

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    static_cast<CapturingEndpoint*>(self)->callbacks = callbacks;
  }

  static iree_status_t Activate(void* self) {
    static_cast<CapturingEndpoint*>(self)->activated = true;
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data) {
    static_cast<CapturingEndpoint*>(self)->activated = false;
    if (callback) callback(user_data);
    return iree_ok_status();
  }

  static iree_status_t Send(
      void* self, const iree_net_message_endpoint_send_params_t* params) {
    auto* endpoint = static_cast<CapturingEndpoint*>(self);
    if (endpoint->next_send_error != IREE_STATUS_OK) {
      iree_status_code_t send_error = endpoint->next_send_error;
      endpoint->next_send_error = IREE_STATUS_OK;
      return iree_status_from_code(send_error);
    }
    CapturedSend captured;
    captured.user_data = params->user_data;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      iree_async_span_t span = params->data.values[i];
      const uint8_t* data =
          static_cast<const uint8_t*>(iree_async_span_ptr(span));
      captured.data.insert(captured.data.end(), data, data + span.length);
    }
    endpoint->sends.push_back(std::move(captured));
    if (endpoint->complete_sends_immediately) {
      endpoint->CompleteSend(endpoint->sends.size() - 1, iree_ok_status());
    }
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    return {1024 * 1024, 64};
  }

  static iree_status_t BeginSend(void* self, iree_host_size_t size,
                                 void** out_ptr,
                                 iree_net_carrier_send_handle_t* out_handle) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "direct send unused");
  }

  static iree_status_t CommitSend(void* self,
                                  iree_net_carrier_send_handle_t handle) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "direct send unused");
  }

  static void AbortSend(void* self, iree_net_carrier_send_handle_t handle) {}

  iree_net_message_endpoint_t as_endpoint() { return {this, &vtable}; }

  void InjectSendReady() {
    if (callbacks.on_send_ready) {
      callbacks.on_send_ready(callbacks.user_data);
    }
  }

  void CompleteSend(iree_host_size_t send_index, iree_status_t status) {
    IREE_ASSERT_LT(send_index, sends.size());
    CapturedSend& send = sends[send_index];
    IREE_ASSERT_FALSE(send.completed);
    send.completed = true;
    iree_net_frame_sender_dispatch_carrier_completion(
        /*callback_user_data=*/NULL, IREE_NET_CARRIER_COMPLETION_SEND,
        send.user_data, status, /*bytes_transferred=*/0,
        /*recv_lease=*/NULL);
  }

  static const iree_net_message_endpoint_vtable_t vtable;
};

const iree_net_message_endpoint_vtable_t CapturingEndpoint::vtable = {
    CapturingEndpoint::SetCallbacks,    CapturingEndpoint::Activate,
    CapturingEndpoint::Deactivate,      CapturingEndpoint::Send,
    CapturingEndpoint::QuerySendBudget, CapturingEndpoint::BeginSend,
    CapturingEndpoint::CommitSend,      CapturingEndpoint::AbortSend,
};

static iree_status_t UnusedOnCommand(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease) {
  return iree_make_status(IREE_STATUS_INTERNAL, "unexpected callback");
}

static void CountLeaseRelease(void* user_data, uint32_t buffer_index) {
  auto* release_count = static_cast<int*>(user_data);
  ++*release_count;
}

struct TestResource {
  iree_hal_resource_t resource;
  int* destroy_count = nullptr;
};

static void DestroyTestResource(iree_hal_resource_t* base_resource) {
  auto* resource = reinterpret_cast<TestResource*>(base_resource);
  ++*resource->destroy_count;
}

static const iree_hal_resource_vtable_t kTestResourceVTable = {
    DestroyTestResource,
};

static void InitializeTestResource(TestResource* resource, int* destroy_count) {
  iree_hal_resource_initialize(&kTestResourceVTable, &resource->resource);
  resource->destroy_count = destroy_count;
}

class RemoteServerSessionHarness {
 public:
  iree_status_t Initialize() {
    topology_.axes = &queue_axis;
    topology_.current_epochs = &queue_epoch;
    topology_.axis_count = 1;
    topology_.machine_index = 1;
    topology_.session_epoch = 1;

    iree_atomic_ref_count_init(&server.ref_count);
    server.host_allocator = iree_allocator_system();
    server.local_topology = topology_;
    iree_slim_mutex_initialize(&server.session_mutex);
    server_mutex_initialized_ = true;

    iree_net_queue_channel_callbacks_t callbacks = {};
    callbacks.on_command = UnusedOnCommand;
    callbacks.on_send_complete = iree_hal_remote_server_on_queue_send_complete;
    callbacks.on_send_ready = iree_hal_remote_server_on_queue_send_ready;
    callbacks.user_data = &session;
    iree_status_t status =
        header_pool_.Initialize(/*buffer_count=*/4, /*buffer_size=*/1024);
    if (iree_status_is_ok(status)) {
      status = iree_net_queue_channel_create(
          endpoint.as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
          header_pool_.release(), callbacks, iree_allocator_system(),
          &queue_channel);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_queue_channel_activate(queue_channel);
    }

    session.server = &server;
    session.session = reinterpret_cast<iree_net_session_t*>(1);
    session.session_id = 1;
    session.queue_channel = queue_channel;
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_resource_table_initialize(
          /*capacity=*/16, iree_allocator_system(), &session.resource_table);
      resource_table_initialized_ = iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_sequence_window_initialize(
          /*initial_observed_sequence=*/0, /*initial_capacity=*/16,
          iree_allocator_system(), &session.observed_submission_window);
      observed_window_initialized_ = iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_sequence_window_initialize(
          /*initial_observed_sequence=*/0, /*initial_capacity=*/16,
          iree_allocator_system(), &session.completed_signal_window);
      completed_window_initialized_ = iree_status_is_ok(status);
    }
    return status;
  }

  ~RemoteServerSessionHarness() {
    iree_hal_remote_server_session_deinitialize_provisionals(
        &session, iree_allocator_system());
    if (resource_table_initialized_) {
      iree_hal_remote_resource_table_deinitialize(&session.resource_table,
                                                  iree_allocator_system());
    }
    if (observed_window_initialized_ && completed_window_initialized_) {
      iree_hal_remote_server_session_deinitialize_windows(&session);
    } else {
      if (observed_window_initialized_) {
        iree_net_sequence_window_deinitialize(
            &session.observed_submission_window);
      }
      if (completed_window_initialized_) {
        iree_net_sequence_window_deinitialize(&session.completed_signal_window);
      }
    }
    iree_net_queue_channel_release(queue_channel);
    if (server_mutex_initialized_) {
      iree_slim_mutex_deinitialize(&server.session_mutex);
    }
  }

  iree_async_axis_t queue_axis = 0x0200;
  uint64_t queue_epoch = 0;
  iree_hal_remote_server_t server = {};
  CapturingEndpoint endpoint;
  iree_net_queue_channel_t* queue_channel = NULL;
  iree_hal_remote_server_session_t session = {};

 private:
  iree_net_session_topology_t topology_ = {};
  TestBufferPool header_pool_;
  bool server_mutex_initialized_ = false;
  bool resource_table_initialized_ = false;
  bool observed_window_initialized_ = false;
  bool completed_window_initialized_ = false;
};

static iree_status_t AssignTestBufferResource(
    RemoteServerSessionHarness* harness, TestResource* resource,
    iree_hal_remote_resource_id_t* out_resource_id) {
  return iree_hal_remote_resource_table_assign(
      &harness->session.resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
      &resource->resource, out_resource_id);
}

static const iree_hal_remote_advance_payload_t* ParseSingleAdvancePayload(
    const std::vector<uint8_t>& advance_frame, iree_async_axis_t expected_axis,
    uint64_t expected_epoch) {
  if (advance_frame.size() < IREE_NET_QUEUE_FRAME_HEADER_SIZE) {
    ADD_FAILURE() << "ADVANCE frame is smaller than queue frame header";
    return nullptr;
  }
  iree_net_queue_frame_header_t header;
  memcpy(&header, advance_frame.data(), sizeof(header));
  EXPECT_EQ(iree_net_queue_frame_header_type(header),
            IREE_NET_QUEUE_FRAME_TYPE_ADVANCE);
  EXPECT_TRUE(iree_all_bits_set(iree_net_queue_frame_header_flags(header),
                                IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER));

  iree_const_byte_span_t payload = iree_make_const_byte_span(
      advance_frame.data() + IREE_NET_QUEUE_FRAME_HEADER_SIZE,
      advance_frame.size() - IREE_NET_QUEUE_FRAME_HEADER_SIZE);
  if (payload.data_length < sizeof(iree_async_frontier_t)) {
    ADD_FAILURE() << "ADVANCE payload is missing signal frontier";
    return nullptr;
  }
  const iree_async_frontier_t* signal_frontier =
      reinterpret_cast<const iree_async_frontier_t*>(payload.data);
  if (signal_frontier->entry_count != 1) {
    ADD_FAILURE() << "ADVANCE signal frontier entry count is "
                  << static_cast<uint32_t>(signal_frontier->entry_count);
    return nullptr;
  }
  EXPECT_EQ(signal_frontier->entries[0].axis, expected_axis);
  EXPECT_EQ(signal_frontier->entries[0].epoch, expected_epoch);

  const iree_host_size_t frontier_size =
      sizeof(iree_async_frontier_t) +
      signal_frontier->entry_count * sizeof(iree_async_frontier_entry_t);
  if (payload.data_length <
      frontier_size + sizeof(iree_hal_remote_advance_payload_t)) {
    ADD_FAILURE() << "ADVANCE payload is missing status payload";
    return nullptr;
  }
  return reinterpret_cast<const iree_hal_remote_advance_payload_t*>(
      payload.data + frontier_size);
}

static void SubmitMissingLeaseFileRead(RemoteServerSessionHarness* harness,
                                       uint64_t signal_epoch) {
  const iree_hal_remote_resource_id_t provisional_file_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, 42);
  iree_hal_remote_file_read_op_t file_read = {};
  file_read.header.type = IREE_HAL_REMOTE_QUEUE_OP_FILE_READ;
  file_read.source_file_id = provisional_file_id;
  file_read.target_buffer_id = 0x101;
  file_read.length = 16;

  iree_async_single_frontier_t signal_frontier_storage;
  iree_async_single_frontier_initialize(&signal_frontier_storage,
                                        harness->queue_axis, signal_epoch);
  IREE_ASSERT_OK(iree_hal_remote_server_on_queue_command(
      &harness->session, /*stream_id=*/0, /*wait_frontier=*/NULL,
      iree_async_single_frontier_as_frontier(&signal_frontier_storage),
      iree_make_const_byte_span(&file_read, sizeof(file_read)),
      /*lease=*/NULL));
}

static void SubmitResourceRelease(RemoteServerSessionHarness* harness,
                                  uint64_t required_observed_epoch,
                                  iree_hal_remote_resource_id_t resource_id) {
  struct ReleasePacket {
    iree_hal_remote_resource_release_op_t op;
    iree_hal_remote_resource_id_t resource_ids[1];
  } release = {};
  release.op.header.type = IREE_HAL_REMOTE_QUEUE_OP_RESOURCE_RELEASE_BATCH;
  release.op.required_observed_epoch = required_observed_epoch;
  release.op.resource_count = IREE_ARRAYSIZE(release.resource_ids);
  release.resource_ids[0] = resource_id;

  IREE_ASSERT_OK(iree_hal_remote_server_on_queue_command(
      &harness->session, /*stream_id=*/0, /*wait_frontier=*/NULL,
      /*signal_frontier=*/NULL,
      iree_make_const_byte_span(&release, sizeof(release)),
      /*lease=*/NULL));
}

TEST(RemoteServerSessionTest, QueueCommandWithoutLeaseSignalsErrorAdvance) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  const iree_hal_remote_resource_id_t provisional_file_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, 42);
  iree_hal_remote_file_read_op_t file_read = {};
  file_read.header.type = IREE_HAL_REMOTE_QUEUE_OP_FILE_READ;
  file_read.source_file_id = provisional_file_id;
  file_read.target_buffer_id = 0x101;
  file_read.length = 16;

  iree_async_single_frontier_t signal_frontier_storage;
  iree_async_single_frontier_initialize(&signal_frontier_storage,
                                        harness.queue_axis, 1);
  IREE_ASSERT_OK(iree_hal_remote_server_on_queue_command(
      &harness.session, /*stream_id=*/0, /*wait_frontier=*/NULL,
      iree_async_single_frontier_as_frontier(&signal_frontier_storage),
      iree_make_const_byte_span(&file_read, sizeof(file_read)),
      /*lease=*/NULL));

  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            1u);
  const iree_hal_remote_advance_payload_t* advance = ParseSingleAdvancePayload(
      harness.endpoint.sends[0].data, harness.queue_axis, 1);
  ASSERT_NE(advance, nullptr);
  EXPECT_EQ(advance->resolution_count, 0);
  EXPECT_EQ(advance->status_code, IREE_STATUS_FAILED_PRECONDITION);
  EXPECT_GT(advance->status_wire_length, 0u);
}

TEST(RemoteServerSessionTest, BackpressuredErrorAdvanceRetriesOnSendReady) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  harness.endpoint.next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;

  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/1);

  EXPECT_TRUE(iree_any_bit_set(harness.session.queue_flags,
                               IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL));
  EXPECT_TRUE(iree_any_bit_set(
      harness.session.queue_flags,
      IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED));
  EXPECT_EQ(harness.session.pending_advances.count, 1u);
  EXPECT_TRUE(harness.endpoint.sends.empty());

  harness.endpoint.InjectSendReady();

  EXPECT_FALSE(iree_any_bit_set(
      harness.session.queue_flags,
      IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED));
  EXPECT_EQ(harness.session.pending_advances.count, 0u);
  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  const iree_hal_remote_advance_payload_t* advance = ParseSingleAdvancePayload(
      harness.endpoint.sends[0].data, harness.queue_axis, 1);
  ASSERT_NE(advance, nullptr);
  EXPECT_EQ(advance->status_code, IREE_STATUS_FAILED_PRECONDITION);
  EXPECT_GT(advance->status_wire_length, 0u);
}

TEST(RemoteServerSessionTest,
     QueueTerminalRetiresLaterCommandsWithoutPublishing) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  harness.endpoint.next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;

  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/1);
  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/2);

  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            2u);
  EXPECT_EQ(harness.session.pending_advances.count, 1u);
  harness.endpoint.InjectSendReady();

  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  const iree_hal_remote_advance_payload_t* advance = ParseSingleAdvancePayload(
      harness.endpoint.sends[0].data, harness.queue_axis, 1);
  ASSERT_NE(advance, nullptr);
  EXPECT_EQ(advance->status_code, IREE_STATUS_FAILED_PRECONDITION);
}

TEST(RemoteServerSessionTest, TeardownReclaimsBackpressuredAdvance) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  harness.endpoint.next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;

  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/1);

  EXPECT_EQ(harness.session.pending_advances.count, 1u);
}

TEST(RemoteServerSessionTest, LateSendCompletionDoesNotWakeReusedSlot) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  harness.endpoint.complete_sends_immediately = false;

  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/1);

  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  EXPECT_FALSE(harness.endpoint.sends[0].completed);

  // Model teardown and reuse while the old channel still owns an admitted
  // send. The late completion belongs to session 1 and must not publish a
  // readiness edge into session 2's active drainer.
  harness.session.session_id = 2;
  harness.session.queue_flags =
      IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE;
  harness.endpoint.CompleteSend(/*send_index=*/0, iree_ok_status());

  EXPECT_EQ(harness.session.queue_flags,
            IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE);
  harness.session.queue_flags = 0;
}

TEST(RemoteServerSessionTest, QueueUnsupportedOpSignalsErrorAdvance) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  iree_hal_remote_queue_op_header_t unsupported_op = {};
  unsupported_op.type = IREE_HAL_REMOTE_QUEUE_OP_QUEUE_EXTENSION;

  iree_async_single_frontier_t signal_frontier_storage;
  iree_async_single_frontier_initialize(&signal_frontier_storage,
                                        harness.queue_axis, 1);
  IREE_ASSERT_OK(iree_hal_remote_server_on_queue_command(
      &harness.session, /*stream_id=*/0, /*wait_frontier=*/NULL,
      iree_async_single_frontier_as_frontier(&signal_frontier_storage),
      iree_make_const_byte_span(&unsupported_op, sizeof(unsupported_op)),
      /*lease=*/NULL));

  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            1u);
  const iree_hal_remote_advance_payload_t* advance = ParseSingleAdvancePayload(
      harness.endpoint.sends[0].data, harness.queue_axis, 1);
  ASSERT_NE(advance, nullptr);
  EXPECT_EQ(advance->resolution_count, 0);
  EXPECT_EQ(advance->status_code, IREE_STATUS_UNIMPLEMENTED);
  EXPECT_GT(advance->status_wire_length, 0u);
}

TEST(RemoteServerSessionTest, QueueTruncatedPayloadSignalsErrorAdvance) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  uint32_t truncated_payload = 0;

  iree_async_single_frontier_t signal_frontier_storage;
  iree_async_single_frontier_initialize(&signal_frontier_storage,
                                        harness.queue_axis, 1);
  IREE_ASSERT_OK(iree_hal_remote_server_on_queue_command(
      &harness.session, /*stream_id=*/0, /*wait_frontier=*/NULL,
      iree_async_single_frontier_as_frontier(&signal_frontier_storage),
      iree_make_const_byte_span(&truncated_payload, sizeof(truncated_payload)),
      /*lease=*/NULL));

  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            1u);
  const iree_hal_remote_advance_payload_t* advance = ParseSingleAdvancePayload(
      harness.endpoint.sends[0].data, harness.queue_axis, 1);
  ASSERT_NE(advance, nullptr);
  EXPECT_EQ(advance->resolution_count, 0);
  EXPECT_EQ(advance->status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_GT(advance->status_wire_length, 0u);
}

TEST(RemoteServerSessionTest, UploadedCommandBufferModeIsServerOwned) {
  iree_hal_command_buffer_mode_t local_mode = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_remote_server_derive_uploaded_command_buffer_mode(
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, &local_mode));
  EXPECT_EQ(local_mode, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);

  IREE_ASSERT_OK(iree_hal_remote_server_derive_uploaded_command_buffer_mode(
      IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED |
          IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      &local_mode));
  EXPECT_EQ(local_mode, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);

  const iree_hal_command_buffer_mode_t metadata_modes =
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA;
  IREE_ASSERT_OK(iree_hal_remote_server_derive_uploaded_command_buffer_mode(
      metadata_modes, &local_mode));
  EXPECT_EQ(local_mode, metadata_modes);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_derive_uploaded_command_buffer_mode(
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, &local_mode));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_derive_uploaded_command_buffer_mode(
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION, &local_mode));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_derive_uploaded_command_buffer_mode(
          UINT32_C(1) << 31, &local_mode));
}

TEST(RemoteServerSessionTest, QueueAtomicRejectsMalformedRecords) {
  iree_hal_remote_server_session_t session = {};
  const iree_hal_semaphore_list_t empty_list = iree_hal_semaphore_list_empty();

  iree_hal_remote_queue_atomic_wait_op_t wait = {};
  wait.header.type = IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_WAIT;
  wait.params.width = IREE_HAL_REMOTE_ATOMIC_WIDTH_64;
  wait.params.condition = IREE_HAL_REMOTE_ATOMIC_WAIT_CONDITION_EQUAL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_wait(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&wait, sizeof(wait) - 1)));
  wait.header.flags = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_wait(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&wait, sizeof(wait))));
  wait.header.flags = 0;
  wait.params.width = 16;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_wait(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&wait, sizeof(wait))));
  wait.params.width = IREE_HAL_REMOTE_ATOMIC_WIDTH_32;
  wait.params.value = UINT64_C(1) << 32;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_wait(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&wait, sizeof(wait))));
  wait.params.value = 0;
  wait.params.condition = 0xFF;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_wait(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&wait, sizeof(wait))));
  wait.params.width = IREE_HAL_REMOTE_ATOMIC_WIDTH_64;
  wait.params.condition = IREE_HAL_REMOTE_ATOMIC_WAIT_CONDITION_EQUAL;
  wait.target.buffer_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_wait(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&wait, sizeof(wait))));

  iree_hal_remote_queue_atomic_store_op_t store = {};
  store.header.type = IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_STORE;
  store.params.width = IREE_HAL_REMOTE_ATOMIC_WIDTH_32;
  store.params.reserved[1] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_store(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&store, sizeof(store))));

  iree_hal_remote_queue_atomic_rmw_op_t rmw = {};
  rmw.header.type = IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_RMW;
  rmw.params.width = IREE_HAL_REMOTE_ATOMIC_WIDTH_64;
  rmw.params.operation = 0xFF;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_atomic_rmw(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&rmw, sizeof(rmw))));
}

TEST(RemoteServerSessionTest, QueueTimestampRejectsMalformedRecords) {
  iree_hal_remote_server_session_t session = {};
  const iree_hal_semaphore_list_t empty_list = iree_hal_semaphore_list_empty();

  iree_hal_remote_queue_timestamp_op_t timestamp = {};
  timestamp.header.type = IREE_HAL_REMOTE_QUEUE_OP_QUEUE_TIMESTAMP;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_timestamp(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&timestamp, sizeof(timestamp) - 1)));

  timestamp.header.type = IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_RMW;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_timestamp(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&timestamp, sizeof(timestamp))));

  timestamp.header.type = IREE_HAL_REMOTE_QUEUE_OP_QUEUE_TIMESTAMP;
  timestamp.header.flags = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_timestamp(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&timestamp, sizeof(timestamp))));

  timestamp.header.flags = 0;
  timestamp.header.reserved = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_timestamp(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&timestamp, sizeof(timestamp))));

  timestamp.header.reserved = 0;
  timestamp.flags = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_timestamp(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&timestamp, sizeof(timestamp))));

  timestamp.flags = IREE_HAL_TIMESTAMP_FLAG_NONE;
  timestamp.target.buffer_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_queue_timestamp(
          &session, /*local_device=*/nullptr, empty_list, empty_list,
          iree_make_const_byte_span(&timestamp, sizeof(timestamp))));
}

TEST(RemoteServerSessionTest, QueueCommandBeforeFailedFileOpenSignalsAdvance) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  const iree_hal_remote_resource_id_t provisional_file_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, 42);
  iree_hal_remote_file_read_op_t file_read = {};
  file_read.header.type = IREE_HAL_REMOTE_QUEUE_OP_FILE_READ;
  file_read.source_file_id = provisional_file_id;
  file_read.target_buffer_id = 0x101;
  file_read.length = 16;

  int lease_release_count = 0;
  iree_async_buffer_lease_t lease = {};
  lease.span = iree_async_span_from_ptr(&file_read, sizeof(file_read));
  lease.release.fn = CountLeaseRelease;
  lease.release.user_data = &lease_release_count;

  iree_async_single_frontier_t signal_frontier_storage;
  iree_async_single_frontier_initialize(&signal_frontier_storage,
                                        harness.queue_axis, 1);
  IREE_ASSERT_OK(iree_hal_remote_server_on_queue_command(
      &harness.session, /*stream_id=*/0, /*wait_frontier=*/NULL,
      iree_async_single_frontier_as_frontier(&signal_frontier_storage),
      iree_make_const_byte_span(&file_read, sizeof(file_read)), &lease));

  EXPECT_EQ(harness.endpoint.sends.size(), 0u);
  EXPECT_EQ(lease.release.fn, nullptr);
  EXPECT_EQ(lease_release_count, 0);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            0u);

  struct FileOpenPacket {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_file_open_request_t request;
    char path[7];
  } file_open = {};
  file_open.envelope.message_type = IREE_HAL_REMOTE_CONTROL_FILE_OPEN;
  file_open.envelope.message_flags =
      IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET;
  file_open.request.provisional_id = provisional_file_id;
  file_open.request.path_length = sizeof(file_open.path);
  file_open.request.mode = IREE_HAL_MEMORY_ACCESS_READ;
  memcpy(file_open.path, "missing", sizeof(file_open.path));

  IREE_ASSERT_OK(iree_hal_remote_server_on_control_data(
      &harness.session, IREE_NET_CONTROL_DATA_FLAG_NONE,
      iree_make_const_byte_span(&file_open, sizeof(file_open)),
      /*lease=*/NULL));

  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  EXPECT_EQ(lease_release_count, 1);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            1u);

  const iree_hal_remote_advance_payload_t* advance = ParseSingleAdvancePayload(
      harness.endpoint.sends[0].data, harness.queue_axis, 1);
  ASSERT_NE(advance, nullptr);
  EXPECT_EQ(advance->resolution_count, 0);
  EXPECT_EQ(advance->status_code, IREE_STATUS_PERMISSION_DENIED);
  EXPECT_GT(advance->status_wire_length, 0u);
}

TEST(RemoteServerSessionTest,
     QueueCommandBeforeUnsupportedFileRegisterSignalsAdvance) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  const iree_hal_remote_resource_id_t provisional_file_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, 42);
  iree_hal_remote_file_read_op_t file_read = {};
  file_read.header.type = IREE_HAL_REMOTE_QUEUE_OP_FILE_READ;
  file_read.source_file_id = provisional_file_id;
  file_read.target_buffer_id = 0x101;
  file_read.length = 16;

  int lease_release_count = 0;
  iree_async_buffer_lease_t lease = {};
  lease.span = iree_async_span_from_ptr(&file_read, sizeof(file_read));
  lease.release.fn = CountLeaseRelease;
  lease.release.user_data = &lease_release_count;

  iree_async_single_frontier_t signal_frontier_storage;
  iree_async_single_frontier_initialize(&signal_frontier_storage,
                                        harness.queue_axis, 1);
  IREE_ASSERT_OK(iree_hal_remote_server_on_queue_command(
      &harness.session, /*stream_id=*/0, /*wait_frontier=*/NULL,
      iree_async_single_frontier_as_frontier(&signal_frontier_storage),
      iree_make_const_byte_span(&file_read, sizeof(file_read)), &lease));

  EXPECT_EQ(harness.endpoint.sends.size(), 0u);
  EXPECT_EQ(lease.release.fn, nullptr);
  EXPECT_EQ(lease_release_count, 0);

  struct FileRegisterPacket {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_file_register_request_t request;
  } file_register = {};
  file_register.envelope.message_type = IREE_HAL_REMOTE_CONTROL_FILE_REGISTER;
  file_register.envelope.message_flags =
      IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET;
  file_register.request.provisional_id = provisional_file_id;
  file_register.request.external_type =
      IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_POSIX_FD;
  file_register.request.access_flags = IREE_HAL_MEMORY_ACCESS_READ;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_remote_server_on_control_data(
          &harness.session, IREE_NET_CONTROL_DATA_FLAG_NONE,
          iree_make_const_byte_span(&file_register, sizeof(file_register)),
          /*lease=*/NULL));

  ASSERT_EQ(harness.endpoint.sends.size(), 1u);
  EXPECT_EQ(lease_release_count, 1);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            1u);

  const iree_hal_remote_advance_payload_t* advance = ParseSingleAdvancePayload(
      harness.endpoint.sends[0].data, harness.queue_axis, 1);
  ASSERT_NE(advance, nullptr);
  EXPECT_EQ(advance->resolution_count, 0);
  EXPECT_EQ(advance->status_code, IREE_STATUS_UNIMPLEMENTED);
  EXPECT_GT(advance->status_wire_length, 0u);
}

TEST(RemoteServerSessionTest, ResourceReleaseWaitsForObservedSubmissionEpoch) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  int destroy_count = 0;
  TestResource resource;
  InitializeTestResource(&resource, &destroy_count);

  iree_hal_remote_resource_id_t resource_id = 0;
  IREE_ASSERT_OK(AssignTestBufferResource(&harness, &resource, &resource_id));
  iree_hal_resource_release(&resource.resource);

  SubmitResourceRelease(&harness, /*required_observed_epoch=*/2, resource_id);
  EXPECT_EQ(destroy_count, 0);
  EXPECT_NE(iree_hal_remote_resource_table_lookup(
                &harness.session.resource_table,
                IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, resource_id),
            nullptr);

  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/1);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            1u);
  EXPECT_EQ(destroy_count, 0);

  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/2);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            2u);
  EXPECT_EQ(destroy_count, 1);
  EXPECT_EQ(iree_hal_remote_resource_table_lookup(
                &harness.session.resource_table,
                IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, resource_id),
            nullptr);
}

TEST(RemoteServerSessionTest,
     CrossTypeReleaseCannotBypassVirtualBufferOwnership) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  int destroy_count = 0;
  TestResource resource;
  InitializeTestResource(&resource, &destroy_count);

  iree_hal_remote_resource_id_t resource_id = 0;
  IREE_ASSERT_OK(AssignTestBufferResource(&harness, &resource, &resource_id));
  iree_hal_resource_release(&resource.resource);
  harness.session.virtual_buffer_map.resource_ids = &resource_id;
  harness.session.virtual_buffer_map.count = 1;
  harness.session.virtual_buffer_map.capacity = 1;

  const iree_hal_remote_resource_id_t forged_file_id =
      (resource_id & UINT64_C(0x00FFFFFFFFFFFFFF)) |
      ((uint64_t)IREE_HAL_REMOTE_RESOURCE_TYPE_FILE << 56);
  SubmitResourceRelease(&harness, /*required_observed_epoch=*/0,
                        forged_file_id);

  EXPECT_EQ(destroy_count, 0);
  EXPECT_EQ(harness.session.virtual_buffer_map.count, 1u);
  EXPECT_EQ(iree_hal_remote_resource_table_lookup(
                &harness.session.resource_table,
                IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, resource_id),
            &resource);

  harness.session.virtual_buffer_map.resource_ids = nullptr;
  harness.session.virtual_buffer_map.count = 0;
  harness.session.virtual_buffer_map.capacity = 0;
  iree_hal_remote_resource_table_release(&harness.session.resource_table,
                                         resource_id);
  EXPECT_EQ(destroy_count, 1);
}

TEST(RemoteServerSessionTest,
     ResourceReleaseWaitsForContiguousObservedSubmissionPrefix) {
  RemoteServerSessionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());
  int destroy_count = 0;
  TestResource resource;
  InitializeTestResource(&resource, &destroy_count);

  iree_hal_remote_resource_id_t resource_id = 0;
  IREE_ASSERT_OK(AssignTestBufferResource(&harness, &resource, &resource_id));
  iree_hal_resource_release(&resource.resource);

  SubmitResourceRelease(&harness, /*required_observed_epoch=*/2, resource_id);
  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/2);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            0u);
  EXPECT_EQ(destroy_count, 0);

  SubmitMissingLeaseFileRead(&harness, /*signal_epoch=*/1);
  EXPECT_EQ(iree_net_sequence_window_observed(
                &harness.session.observed_submission_window),
            2u);
  EXPECT_EQ(destroy_count, 1);
}

}  // namespace
