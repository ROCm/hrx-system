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
#include "iree/hal/remote/server/server.h"
#include "iree/net/channel/queue/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TestBufferPool {
 public:
  TestBufferPool(iree_host_size_t buffer_count, iree_host_size_t buffer_size) {
    const iree_host_size_t total_size = buffer_count * buffer_size;
    void* memory = malloc(total_size);
    memset(memory, 0, total_size);

    region_ =
        static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
    memset(region_, 0, sizeof(*region_));
    iree_atomic_ref_count_init(&region_->ref_count);
    region_->destroy_fn = DestroyRegion;
    region_->base_ptr = memory;
    region_->length = total_size;
    region_->buffer_size = buffer_size;
    region_->buffer_count = static_cast<uint32_t>(buffer_count);

    iree_status_t status = iree_async_buffer_pool_allocate(
        region_, iree_allocator_system(), &pool_);
    IREE_CHECK_OK(status);
    iree_async_region_release(region_);
  }

  ~TestBufferPool() {
    if (pool_) iree_async_buffer_pool_free(pool_);
  }

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
};

struct CapturingEndpoint {
  iree_net_message_endpoint_callbacks_t callbacks = {};
  bool activated = false;
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
    CapturedSend captured;
    captured.user_data = params->user_data;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      iree_async_span_t span = params->data.values[i];
      const uint8_t* data =
          static_cast<const uint8_t*>(iree_async_span_ptr(span));
      captured.data.insert(captured.data.end(), data, data + span.length);
    }
    endpoint->sends.push_back(std::move(captured));
    iree_net_frame_sender_dispatch_carrier_completion(
        /*callback_user_data=*/NULL, params->user_data, iree_ok_status(),
        /*bytes_transferred=*/0, /*recv_lease=*/NULL);
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

class RemoteServerSessionHarness {
 public:
  RemoteServerSessionHarness() {
    topology_.axes = &queue_axis;
    topology_.current_epochs = &queue_epoch;
    topology_.axis_count = 1;
    topology_.machine_index = 1;
    topology_.session_epoch = 1;

    server.host_allocator = iree_allocator_system();
    server.local_topology = topology_;
    iree_slim_mutex_initialize(&server.session_mutex);

    iree_net_queue_channel_callbacks_t callbacks = {};
    callbacks.on_command = UnusedOnCommand;
    IREE_CHECK_OK(iree_net_queue_channel_create(
        endpoint.as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
        header_pool_.release(), callbacks, iree_allocator_system(),
        &queue_channel));
    IREE_CHECK_OK(iree_net_queue_channel_activate(queue_channel));

    session.server = &server;
    session.session = reinterpret_cast<iree_net_session_t*>(1);
    session.session_id = 1;
    session.queue_channel = queue_channel;
    IREE_CHECK_OK(iree_hal_remote_resource_table_initialize(
        /*capacity=*/16, iree_allocator_system(), &session.resource_table));
    IREE_CHECK_OK(iree_net_sequence_window_initialize(
        /*initial_observed_sequence=*/0, /*initial_capacity=*/16,
        iree_allocator_system(), &session.observed_submission_window));
    IREE_CHECK_OK(iree_net_sequence_window_initialize(
        /*initial_observed_sequence=*/0, /*initial_capacity=*/16,
        iree_allocator_system(), &session.completed_signal_window));
  }

  ~RemoteServerSessionHarness() {
    iree_hal_remote_server_session_deinitialize_provisionals(
        &session, iree_allocator_system());
    iree_hal_remote_resource_table_deinitialize(&session.resource_table,
                                                iree_allocator_system());
    iree_net_sequence_window_deinitialize(&session.observed_submission_window);
    iree_net_sequence_window_deinitialize(&session.completed_signal_window);
    iree_net_queue_channel_release(queue_channel);
    iree_slim_mutex_deinitialize(&server.session_mutex);
  }

  iree_async_axis_t queue_axis = 0x0200;
  uint64_t queue_epoch = 0;
  iree_hal_remote_server_t server = {};
  CapturingEndpoint endpoint;
  iree_net_queue_channel_t* queue_channel = NULL;
  iree_hal_remote_server_session_t session = {};

 private:
  iree_net_session_topology_t topology_ = {};
  TestBufferPool header_pool_{/*buffer_count=*/4, /*buffer_size=*/1024};
};

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

TEST(RemoteServerSessionTest, QueueCommandWithoutLeaseSignalsErrorAdvance) {
  RemoteServerSessionHarness harness;
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

TEST(RemoteServerSessionTest, QueueCommandBeforeFailedFileOpenSignalsAdvance) {
  RemoteServerSessionHarness harness;
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

}  // namespace
