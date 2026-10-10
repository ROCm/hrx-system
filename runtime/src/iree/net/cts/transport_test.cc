// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/channel/control/control_channel.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/connection.h"
#include "iree/net/cts/transport_backend.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::cts {
namespace {

enum PollSide {
  kNotPolling = 0,
  kClientPolling = 1,
  kServerPolling = 2,
};

struct ConnectState {
  // Stable caller-owned attempt, kept through callback and initiation return.
  iree_net_transport_connect_operation_t operation;

  ConnectState() {
    iree_net_transport_connect_operation_initialize(&operation);
  }
  ~ConnectState() {
    iree_net_transport_connect_operation_deinitialize(&operation);
  }
  ConnectState(const ConnectState&) = delete;
  ConnectState& operator=(const ConnectState&) = delete;

  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  bool submitted = false;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_net_connection_t* connection = nullptr;

  static void OnConnect(void* user_data, iree_status_t status,
                        iree_net_connection_t* connection) {
    auto* self = static_cast<ConnectState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    self->connection = connection;
    iree_status_free(status);
  }

  iree_net_transport_connect_callback_t callback() {
    return {.fn = OnConnect, .user_data = this};
  }
};

struct AcceptState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_net_connection_t* connection = nullptr;

  static void OnAccept(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto* self = static_cast<AcceptState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    self->connection = connection;
    iree_status_free(status);
  }

  iree_net_listener_accept_callback_t callback() {
    return {.fn = OnAccept, .user_data = this};
  }
};

struct StopState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  bool submitted = false;
  bool completed = false;

  static void OnStopped(void* user_data) {
    auto* self = static_cast<StopState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    self->completed = true;
  }

  iree_net_listener_stopped_callback_t callback() {
    return {.fn = OnStopped, .user_data = this};
  }
};

struct EndpointReadyState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_net_message_endpoint_t endpoint = {};

  static void OnReady(void* user_data, iree_status_t status,
                      iree_net_message_endpoint_t endpoint) {
    auto* self = static_cast<EndpointReadyState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    self->endpoint = endpoint;
    iree_status_free(status);
  }

  iree_net_endpoint_ready_callback_t callback() {
    return {.fn = OnReady, .user_data = this};
  }
};

struct MessageState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  std::vector<std::string> messages;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;

  static iree_status_t OnMessage(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<MessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    self->messages.emplace_back(reinterpret_cast<const char*>(message.data),
                                message.data_length);
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<MessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_message_endpoint_callbacks_t callbacks() {
    return {
        .on_message = OnMessage,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

struct ProtocolHandoffState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  iree_net_message_endpoint_t endpoint = {};
  bool message_callback_active = false;
  std::vector<std::string> bootstrap_messages;
  std::vector<std::string> operational_messages;
  int error_count = 0;

  static iree_status_t OnBootstrapMessage(void* user_data,
                                          iree_const_byte_span_t message,
                                          iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<ProtocolHandoffState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    EXPECT_FALSE(self->message_callback_active);
    self->message_callback_active = true;
    self->bootstrap_messages.emplace_back(
        reinterpret_cast<const char*>(message.data), message.data_length);
    iree_net_message_endpoint_set_callbacks(self->endpoint,
                                            self->operational_callbacks());
    self->message_callback_active = false;
    return iree_ok_status();
  }

  static iree_status_t OnOperationalMessage(void* user_data,
                                            iree_const_byte_span_t message,
                                            iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<ProtocolHandoffState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    EXPECT_FALSE(self->message_callback_active);
    self->message_callback_active = true;
    self->operational_messages.emplace_back(
        reinterpret_cast<const char*>(message.data), message.data_length);
    self->message_callback_active = false;
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<ProtocolHandoffState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    iree_status_free(status);
  }

  iree_net_message_endpoint_callbacks_t bootstrap_callbacks() {
    return {
        .on_message = OnBootstrapMessage,
        .on_error = OnError,
        .user_data = this,
    };
  }

  iree_net_message_endpoint_callbacks_t operational_callbacks() {
    return {
        .on_message = OnOperationalMessage,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

struct ControlMessageState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  std::vector<std::string> messages;
  std::vector<iree_net_control_data_flags_t> flags;
  int goaway_count = 0;
  uint32_t goaway_reason = 0;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;

  static iree_status_t OnData(void* user_data,
                              iree_net_control_data_flags_t flags,
                              iree_const_byte_span_t payload,
                              iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<ControlMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    self->messages.emplace_back(reinterpret_cast<const char*>(payload.data),
                                payload.data_length);
    self->flags.push_back(flags);
    return iree_ok_status();
  }

  static void OnGoaway(void* user_data, uint32_t reason_code) {
    auto* self = static_cast<ControlMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->goaway_count;
    self->goaway_reason = reason_code;
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<ControlMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_control_channel_callbacks_t callbacks() {
    return {
        .on_data = OnData,
        .on_goaway = OnGoaway,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

struct QueueMessageState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int command_count = 0;
  uint32_t command_queue_id = IREE_NET_QUEUE_ID_NONE;
  std::vector<iree_async_frontier_entry_t> command_wait_frontier;
  std::vector<iree_async_frontier_entry_t> command_signal_frontier;
  std::string command_payload;
  int advance_count = 0;
  std::vector<iree_async_frontier_entry_t> advance_signal_frontier;
  std::string advance_payload;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;

  static std::vector<iree_async_frontier_entry_t> CaptureFrontier(
      const iree_net_queue_frontier_view_t* frontier) {
    std::vector<iree_async_frontier_entry_t> entries;
    entries.reserve(frontier->count);
    for (iree_host_size_t i = 0; i < frontier->count; ++i) {
      entries.push_back(iree_net_queue_frontier_view_get(frontier, i));
    }
    return entries;
  }

  static iree_status_t OnCommand(
      void* user_data, uint32_t queue_id,
      const iree_net_queue_frontier_view_t* wait_frontier,
      const iree_net_queue_frontier_view_t* signal_frontier,
      iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<QueueMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    ++self->command_count;
    self->command_queue_id = queue_id;
    self->command_wait_frontier = CaptureFrontier(wait_frontier);
    self->command_signal_frontier = CaptureFrontier(signal_frontier);
    self->command_payload.assign(reinterpret_cast<const char*>(payload.data),
                                 payload.data_length);
    return iree_ok_status();
  }

  static iree_status_t OnAdvance(
      void* user_data, const iree_net_queue_frontier_view_t* signal_frontier,
      iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<QueueMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    ++self->advance_count;
    self->advance_signal_frontier = CaptureFrontier(signal_frontier);
    self->advance_payload.assign(reinterpret_cast<const char*>(payload.data),
                                 payload.data_length);
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<QueueMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_queue_channel_callbacks_t callbacks() {
    return {
        .on_command = OnCommand,
        .on_advance = OnAdvance,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

struct QueueBuildState {
  std::vector<iree_async_frontier_entry_t> wait_frontier;
  std::vector<iree_async_frontier_entry_t> signal_frontier;
  std::string generated_payload;

  static iree_status_t Build(void* user_data,
                             const iree_net_queue_message_builder_t* builder) {
    auto* self = static_cast<QueueBuildState*>(user_data);
    if (builder->wait_frontier.count != self->wait_frontier.size() ||
        builder->signal_frontier.count != self->signal_frontier.size() ||
        builder->generated_payload.data_length !=
            self->generated_payload.size()) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "queue builder target layout mismatch");
    }
    for (iree_host_size_t i = 0; i < self->wait_frontier.size(); ++i) {
      iree_net_queue_frontier_builder_set(&builder->wait_frontier, i,
                                          self->wait_frontier[i]);
    }
    for (iree_host_size_t i = 0; i < self->signal_frontier.size(); ++i) {
      iree_net_queue_frontier_builder_set(&builder->signal_frontier, i,
                                          self->signal_frontier[i]);
    }
    if (!self->generated_payload.empty()) {
      memcpy(builder->generated_payload.data, self->generated_payload.data(),
             self->generated_payload.size());
    }
    return iree_ok_status();
  }

  iree_net_queue_channel_send_params_t params(
      iree_async_span_list_t payload,
      iree_net_send_completion_callback_t completion_callback) {
    return {
        .wait_frontier_count = static_cast<uint8_t>(wait_frontier.size()),
        .signal_frontier_count = static_cast<uint8_t>(signal_frontier.size()),
        .generated_payload_length = generated_payload.size(),
        .build = Build,
        .build_user_data = this,
        .payload = payload,
        .completion_callback = completion_callback,
    };
  }
};

struct BulkMessageState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int start_count = 0;
  uint64_t start_transfer_id = 0;
  uint64_t start_total_length = 0;
  std::vector<uint64_t> data_offsets;
  std::vector<std::string> data_payloads;
  bool retain_next_data = false;
  iree_async_buffer_lease_t retained_data_lease = {};
  int complete_count = 0;
  uint64_t complete_transfer_id = 0;
  int abort_count = 0;
  uint64_t abort_transfer_id = 0;
  std::string abort_detail;
  int credit_count = 0;
  uint64_t credit_delta = 0;
  uint64_t available_credit_count = 0;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_length) {
    auto* self = static_cast<BulkMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->start_count;
    self->start_transfer_id = transfer_id;
    self->start_total_length = total_length;
    return iree_ok_status();
  }

  static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                              uint64_t offset, iree_const_byte_span_t payload,
                              iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<BulkMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    EXPECT_EQ(transfer_id, self->start_transfer_id);
    self->data_offsets.push_back(offset);
    self->data_payloads.emplace_back(
        reinterpret_cast<const char*>(payload.data), payload.data_length);
    if (self->retain_next_data) {
      EXPECT_EQ(self->retained_data_lease.release.fn, nullptr);
      self->retained_data_lease = *lease;
      *lease = {};
      self->retain_next_data = false;
    }
    return iree_ok_status();
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    auto* self = static_cast<BulkMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->complete_count;
    self->complete_transfer_id = transfer_id;
    return iree_ok_status();
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t detail,
                               iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<BulkMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    ++self->abort_count;
    self->abort_transfer_id = transfer_id;
    self->abort_detail.assign(reinterpret_cast<const char*>(detail.data),
                              detail.data_length);
    return iree_ok_status();
  }

  static iree_status_t OnCredit(void* user_data, uint64_t credit_delta,
                                uint64_t available_credit_count) {
    auto* self = static_cast<BulkMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->credit_count;
    self->credit_delta = credit_delta;
    self->available_credit_count = available_credit_count;
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<BulkMessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_bulk_channel_callbacks_t callbacks() {
    return {
        .on_start = OnStart,
        .on_data = OnData,
        .on_complete = OnComplete,
        .on_abort = OnAbort,
        .on_credit = OnCredit,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

struct SendState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  iree_host_size_t expected_bytes = 0;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;

  static void OnComplete(void* user_data, iree_status_t status,
                         iree_host_size_t bytes_transferred) {
    auto* self = static_cast<SendState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_EQ(bytes_transferred, self->expected_bytes);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_send_completion_callback_t callback() {
    return {.fn = OnComplete, .user_data = this};
  }
};

struct DeactivateState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  bool completed = false;

  static void OnDeactivated(void* user_data) {
    auto* self = static_cast<DeactivateState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    self->completed = true;
  }

  iree_net_connection_deactivate_callback_t callback() {
    return {.fn = OnDeactivated, .user_data = this};
  }
};

struct ReceivePoolResources {
  // Host storage backing receive buffers.
  iree_async_slab_t* slab = nullptr;

  // Proactor registration covering |slab|.
  iree_async_region_t* region = nullptr;

  // Receive buffer pool suballocating |region|.
  iree_async_buffer_pool_t* pool = nullptr;
};

static iree_status_t CreateReceivePool(iree_async_proactor_t* proactor,
                                       ReceivePoolResources* out_resources) {
  *out_resources = ReceivePoolResources{};
  iree_async_slab_options_t slab_options = {
      .buffer_size = 64 * 1024,
      .buffer_count = 16,
  };
  iree_status_t status = iree_async_slab_create(
      slab_options, iree_allocator_system(), &out_resources->slab);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_slab(
        proactor, out_resources->slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
        &out_resources->region);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_create(
        out_resources->region, iree_allocator_system(), &out_resources->pool);
  }
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_pool_release(out_resources->pool);
    iree_async_region_release(out_resources->region);
    iree_async_slab_release(out_resources->slab);
    *out_resources = ReceivePoolResources{};
  }
  return status;
}

static void ReleaseReceivePool(ReceivePoolResources* resources) {
  iree_async_buffer_pool_release(resources->pool);
  iree_async_region_release(resources->region);
  iree_async_slab_release(resources->slab);
  *resources = ReceivePoolResources{};
}

class TransportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = &GetTransportBackend();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &client_proactor_));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &server_proactor_));
    IREE_ASSERT_OK(CreateReceivePool(client_proactor_, &client_receive_pool_));
    IREE_ASSERT_OK(CreateReceivePool(server_proactor_, &server_receive_pool_));
    iree_status_t status =
        backend_->create_factory(iree_allocator_system(), &factory_);
    if (iree_status_code(status) == IREE_STATUS_UNAVAILABLE) {
      iree_status_free(status);
      GTEST_SKIP() << backend_->name << " transport unavailable";
    }
    IREE_ASSERT_OK(status);

    connect_state_.current_poll_side = &current_poll_side_;
    connect_state_.expected_poll_side = kClientPolling;
    accept_state_.current_poll_side = &current_poll_side_;
    accept_state_.expected_poll_side = kServerPolling;
    stop_state_.current_poll_side = &current_poll_side_;
    stop_state_.expected_poll_side = kServerPolling;
  }

  void TearDown() override {
    iree_async_buffer_lease_release(&client_bulk_messages_.retained_data_lease);
    iree_async_buffer_lease_release(&server_bulk_messages_.retained_data_lease);
    DrainPendingConnection();
    StopAndFreeListener();
    DeactivateAndRelease(client_connection_, client_proactor_, kClientPolling);
    DeactivateAndRelease(server_connection_, server_proactor_, kServerPolling);
    iree_net_control_channel_free(client_control_channel_);
    iree_net_control_channel_free(server_control_channel_);
    iree_net_queue_channel_free(client_queue_channel_);
    iree_net_queue_channel_free(server_queue_channel_);
    iree_net_bulk_channel_free(client_bulk_channel_);
    iree_net_bulk_channel_free(server_bulk_channel_);
    iree_net_transport_factory_release(factory_);
    ReleaseReceivePool(&client_receive_pool_);
    ReleaseReceivePool(&server_receive_pool_);
    if (owns_client_proactor_) {
      iree_async_proactor_release(client_proactor_);
    }
    if (owns_server_proactor_) {
      iree_async_proactor_release(server_proactor_);
    }
  }

  void Poll(iree_async_proactor_t* proactor, PollSide side) {
    current_poll_side_ = side;
    iree_status_t status = iree_async_proactor_poll(
        proactor, iree_infinite_timeout(), /*out_completed_count=*/nullptr);
    current_poll_side_ = kNotPolling;
    IREE_ASSERT_OK(status);
  }

  void PollImmediate(iree_async_proactor_t* proactor, PollSide side) {
    current_poll_side_ = side;
    iree_status_t status = iree_async_proactor_poll(
        proactor, iree_immediate_timeout(), /*out_completed_count=*/nullptr);
    current_poll_side_ = kNotPolling;
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_free(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  void PollUntil(iree_async_proactor_t* proactor, PollSide side,
                 const std::function<bool()>& condition) {
    while (!condition()) {
      Poll(proactor, side);
    }
  }

  void PollBothUntil(const std::function<bool()>& condition) {
    while (!condition()) {
      PollImmediate(client_proactor_, kClientPolling);
      PollImmediate(server_proactor_, kServerPolling);
    }
  }

  void CreateListener() {
    std::string bind_address;
    IREE_ASSERT_OK(backend_->make_bind_address(&bind_address));
    IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
        factory_,
        iree_make_string_view(bind_address.data(), bind_address.size()),
        server_proactor_, server_receive_pool_.pool, accept_state_.callback(),
        iree_allocator_system(), &listener_));

    std::array<char, 1024> address_storage = {};
    iree_string_view_t bound_address = iree_string_view_empty();
    IREE_ASSERT_OK(iree_net_listener_query_bound_address(
        listener_, address_storage.size(), address_storage.data(),
        &bound_address));
    connect_address_.assign(bound_address.data, bound_address.size);
  }

  void SubmitConnect(iree_string_view_t address) {
    IREE_ASSERT_OK(iree_net_transport_factory_connect(
        factory_, address, client_proactor_, client_receive_pool_.pool,
        connect_state_.callback(), &connect_state_.operation));
    connect_state_.submitted = true;
  }

  void DrainPendingConnection() {
    if (!connect_state_.submitted) {
      return;
    }
    if (connect_state_.callback_count == 0) {
      PollBothUntil([&] { return connect_state_.callback_count == 1; });
    }
    if (connect_state_.status_code == IREE_STATUS_OK && listener_ &&
        accept_state_.callback_count == 0) {
      PollUntil(server_proactor_, kServerPolling,
                [&] { return accept_state_.callback_count == 1; });
    }
    if (!client_connection_ && connect_state_.connection) {
      client_connection_ = connect_state_.connection;
      connect_state_.connection = nullptr;
    }
    if (!server_connection_ && accept_state_.connection) {
      server_connection_ = accept_state_.connection;
      accept_state_.connection = nullptr;
    }
    connect_state_.submitted = false;
  }

  void EstablishConnection() {
    ASSERT_NO_FATAL_FAILURE(CreateListener());
    SubmitConnect(iree_make_string_view(connect_address_.data(),
                                        connect_address_.size()));
    EXPECT_EQ(connect_state_.callback_count, 0);
    EXPECT_EQ(accept_state_.callback_count, 0);
    // Establishment may exchange resource-import records before either peer
    // can publish a usable connection. Both owners must be allowed to progress.
    PollBothUntil([&] { return connect_state_.callback_count == 1; });
    ASSERT_EQ(connect_state_.status_code, IREE_STATUS_OK);
    PollUntil(server_proactor_, kServerPolling,
              [&] { return accept_state_.callback_count == 1; });
    ASSERT_EQ(accept_state_.status_code, IREE_STATUS_OK);
    DrainPendingConnection();
    ASSERT_NE(client_connection_, nullptr);
    ASSERT_NE(server_connection_, nullptr);
    EXPECT_EQ(iree_net_connection_proactor(client_connection_),
              client_proactor_);
    EXPECT_EQ(iree_net_connection_proactor(server_connection_),
              server_proactor_);
  }

  EndpointReadyState* SubmitOpenEndpoint(iree_net_connection_t* connection,
                                         PollSide side) {
    auto state = std::make_unique<EndpointReadyState>();
    state->current_poll_side = &current_poll_side_;
    state->expected_poll_side = side;
    EndpointReadyState* state_ptr = state.get();
    endpoint_ready_states_.push_back(std::move(state));
    iree_status_t status =
        iree_net_connection_open_endpoint(connection, state_ptr->callback());
    const iree_status_code_t status_code = iree_status_code(status);
    IREE_EXPECT_OK(status);
    return status_code == IREE_STATUS_OK ? state_ptr : nullptr;
  }

  std::pair<iree_net_message_endpoint_t, iree_net_message_endpoint_t>
  OpenEndpoints() {
    auto* client = SubmitOpenEndpoint(client_connection_, kClientPolling);
    auto* server = SubmitOpenEndpoint(server_connection_, kServerPolling);
    if (!client || !server) {
      return {};
    }
    // Native setup may need peer geometry before either endpoint is ready.
    PollBothUntil([&] {
      return client->callback_count == 1 && server->callback_count == 1;
    });
    EXPECT_EQ(client->status_code, IREE_STATUS_OK);
    EXPECT_EQ(server->status_code, IREE_STATUS_OK);
    return {client->endpoint, server->endpoint};
  }

  void WaitForSendSlots(iree_net_message_endpoint_t endpoint,
                        iree_host_size_t minimum_slots) {
    PollBothUntil([&] {
      return iree_net_message_endpoint_query_send_budget(endpoint).slots >=
             minimum_slots;
    });
  }

  void StopAndFreeListener() {
    if (!listener_) {
      return;
    }
    if (!stop_state_.submitted) {
      IREE_ASSERT_OK(iree_net_listener_stop(listener_, stop_state_.callback()));
      stop_state_.submitted = true;
    }
    if (!stop_state_.completed) {
      PollUntil(server_proactor_, kServerPolling,
                [&] { return stop_state_.completed; });
    }
    iree_net_listener_free(listener_);
    listener_ = nullptr;
  }

  void DeactivateAndRelease(iree_net_connection_t*& connection,
                            iree_async_proactor_t* proactor, PollSide side) {
    if (!connection) {
      return;
    }
    DeactivateState state;
    state.current_poll_side = &current_poll_side_;
    state.expected_poll_side = side;
    current_poll_side_ = side;
    iree_net_connection_deactivate(connection, state.callback());
    current_poll_side_ = kNotPolling;
    if (!state.completed) {
      PollUntil(proactor, side, [&] { return state.completed; });
    }
    iree_net_connection_release(connection);
    connection = nullptr;
  }

  // Keeps test-local callback targets alive through drain, including when a
  // fatal assertion returns early from the test body.
  struct ScopedConnectionDrain {
    // Fixture owning the connections and their poll owners.
    TransportTest* test;

    ~ScopedConnectionDrain() {
      test->StopAndFreeListener();
      test->DeactivateAndRelease(test->client_connection_,
                                 test->client_proactor_, kClientPolling);
      test->DeactivateAndRelease(test->server_connection_,
                                 test->server_proactor_, kServerPolling);
    }
  };

  const TransportBackend* backend_ = nullptr;
  int current_poll_side_ = kNotPolling;
  bool owns_client_proactor_ = true;
  bool owns_server_proactor_ = true;
  iree_async_proactor_t* client_proactor_ = nullptr;
  iree_async_proactor_t* server_proactor_ = nullptr;
  ReceivePoolResources client_receive_pool_;
  ReceivePoolResources server_receive_pool_;
  iree_net_transport_factory_t* factory_ = nullptr;
  iree_net_listener_t* listener_ = nullptr;
  iree_net_connection_t* client_connection_ = nullptr;
  iree_net_connection_t* server_connection_ = nullptr;
  std::string connect_address_;
  ConnectState connect_state_;
  AcceptState accept_state_;
  StopState stop_state_;
  std::vector<std::unique_ptr<EndpointReadyState>> endpoint_ready_states_;
  MessageState client_messages_;
  MessageState server_messages_;
  SendState client_send_;
  SendState server_send_;
  ControlMessageState client_control_messages_;
  ControlMessageState server_control_messages_;
  iree_net_control_channel_t* client_control_channel_ = nullptr;
  iree_net_control_channel_t* server_control_channel_ = nullptr;
  QueueMessageState client_queue_messages_;
  QueueMessageState server_queue_messages_;
  iree_net_queue_channel_t* client_queue_channel_ = nullptr;
  iree_net_queue_channel_t* server_queue_channel_ = nullptr;
  BulkMessageState client_bulk_messages_;
  BulkMessageState server_bulk_messages_;
  iree_net_bulk_channel_t* client_bulk_channel_ = nullptr;
  iree_net_bulk_channel_t* server_bulk_channel_ = nullptr;
};

TEST_F(TransportTest, ReportsRequiredCapabilities) {
  const iree_net_transport_capabilities_t capabilities =
      iree_net_transport_factory_query_capabilities(factory_);
  EXPECT_TRUE(iree_all_bits_set(capabilities, backend_->required_capabilities));
}

TEST_F(TransportTest, OptionalDirectOpenPreservesMessageOrdinals) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  struct Ready {
    // Owner-side callback witness shared with the fixture.
    int* current_side;
    // Expected executor for this borrowed view.
    PollSide expected_side;
    // Exactly-once accepted result count.
    uint32_t count = 0;
    static void Complete(void* user_data, iree_status_t status,
                         iree_net_direct_endpoint_t endpoint) {
      auto* self = static_cast<Ready*>(user_data);
      EXPECT_EQ(*self->current_side, self->expected_side);
      IREE_CHECK_OK(status);
      EXPECT_NE(endpoint.self, nullptr);
      ++self->count;
    }
  } client{&current_poll_side_, kClientPolling},
      server{&current_poll_side_, kServerPolling};
  ScopedConnectionDrain drain{this};
  iree_status_t client_status = iree_net_connection_open_direct_endpoint(
      client_connection_, {Ready::Complete, &client});
  iree_status_t server_status = iree_net_connection_open_direct_endpoint(
      server_connection_, {Ready::Complete, &server});
  auto code = iree_status_code(client_status);
  EXPECT_EQ(iree_status_code(server_status), code);
  iree_status_free(client_status);
  iree_status_free(server_status);
  ASSERT_TRUE(code == IREE_STATUS_OK || code == IREE_STATUS_UNIMPLEMENTED);
  EXPECT_EQ(client.count, 0u);
  EXPECT_EQ(server.count, 0u);
  if (code == IREE_STATUS_OK) {
    PollBothUntil([&] { return client.count == 1 && server.count == 1; });
  }
  uint32_t message_count =
      iree_net_connection_max_endpoint_count(client_connection_) -
      (code == IREE_STATUS_OK ? 1u : 0u);
  for (uint32_t i = 0; i < message_count; ++i) {
    auto endpoints = OpenEndpoints();
    EXPECT_NE(endpoints.first.self, nullptr);
    EXPECT_NE(endpoints.second.self, nullptr);
  }
  EXPECT_EQ(client.count, code == IREE_STATUS_OK ? 1u : 0u);
  EXPECT_EQ(server.count, client.count);
}

TEST_F(TransportTest, ListenerStopIsAsynchronousAndRefusesConnections) {
  ASSERT_NO_FATAL_FAILURE(CreateListener());
  IREE_ASSERT_OK(iree_net_listener_stop(listener_, stop_state_.callback()));
  stop_state_.submitted = true;
  EXPECT_FALSE(stop_state_.completed);
  PollUntil(server_proactor_, kServerPolling,
            [&] { return stop_state_.completed; });
  iree_net_listener_free(listener_);
  listener_ = nullptr;

  SubmitConnect(
      iree_make_string_view(connect_address_.data(), connect_address_.size()));
  EXPECT_EQ(connect_state_.callback_count, 0);
  PollUntil(client_proactor_, kClientPolling,
            [&] { return connect_state_.callback_count == 1; });
  EXPECT_NE(connect_state_.status_code, IREE_STATUS_OK);
  EXPECT_EQ(connect_state_.connection, nullptr);
}

TEST_F(TransportTest, CancelBeforeKickoffAllowsReuseAfterJoin) {
  ASSERT_NO_FATAL_FAILURE(CreateListener());
  const auto address =
      iree_make_string_view(connect_address_.data(), connect_address_.size());
  SubmitConnect(address);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_transport_factory_connect(
          factory_, address, client_proactor_, client_receive_pool_.pool,
          connect_state_.callback(), &connect_state_.operation));
  iree_net_transport_connect_operation_cancel(&connect_state_.operation);
  iree_net_transport_connect_operation_cancel(&connect_state_.operation);
  EXPECT_EQ(connect_state_.callback_count, 0);
  PollUntil(client_proactor_, kClientPolling,
            [&] { return connect_state_.callback_count == 1; });
  EXPECT_EQ(connect_state_.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(connect_state_.connection, nullptr);
  PollImmediate(server_proactor_, kServerPolling);
  EXPECT_EQ(accept_state_.callback_count, 0);

  // A detached operation is not a pre-cancel token for its next execution.
  iree_net_transport_connect_operation_cancel(&connect_state_.operation);
  SubmitConnect(address);
  PollBothUntil([&] { return connect_state_.callback_count == 2; });
  ASSERT_EQ(connect_state_.status_code, IREE_STATUS_OK);
  PollUntil(server_proactor_, kServerPolling,
            [&] { return accept_state_.callback_count == 1; });
  EXPECT_EQ(accept_state_.status_code, IREE_STATUS_OK);
}

TEST_F(TransportTest, CancelledConnectCallbackDestroysOperationOwner) {
  ASSERT_NO_FATAL_FAILURE(CreateListener());
  struct Owner {
    // Stable attempt storage destroyed by its terminal callback.
    iree_net_transport_connect_operation_t operation;
    // Completion witness outside the destroyed object.
    bool* destroyed;
  };
  bool destroyed = false;
  auto* owner = new Owner{{}, &destroyed};
  iree_net_transport_connect_operation_initialize(&owner->operation);
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_,
      iree_make_string_view(connect_address_.data(), connect_address_.size()),
      client_proactor_, client_receive_pool_.pool,
      {[](void* user_data, iree_status_t status,
          iree_net_connection_t* connection) {
         auto* owner = static_cast<Owner*>(user_data);
         IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
         EXPECT_EQ(connection, nullptr);
         // Detachment precedes publication and the callback holds no mutex.
         iree_net_transport_connect_operation_cancel(&owner->operation);
         iree_net_transport_connect_operation_deinitialize(&owner->operation);
         *owner->destroyed = true;
         delete owner;
       },
       owner},
      &owner->operation));
  iree_net_transport_connect_operation_cancel(&owner->operation);
  EXPECT_FALSE(destroyed);
  PollUntil(client_proactor_, kClientPolling, [&] { return destroyed; });
  StopAndFreeListener();
  EXPECT_EQ(accept_state_.callback_count, 0);
}

TEST_F(TransportTest, ConcurrentCancellationJoinsBeforeCallerStorageReuse) {
  ASSERT_NO_FATAL_FAILURE(CreateListener());
  SubmitConnect(
      iree_make_string_view(connect_address_.data(), connect_address_.size()));
  std::thread canceller([&] {
    for (int i = 0; i < 64; ++i) {
      iree_net_transport_connect_operation_cancel(&connect_state_.operation);
    }
  });
  PollBothUntil([&] { return connect_state_.callback_count == 1; });
  canceller.join();
  EXPECT_TRUE(connect_state_.status_code == IREE_STATUS_OK ||
              connect_state_.status_code == IREE_STATUS_CANCELLED);
  if (connect_state_.status_code == IREE_STATUS_OK) {
    PollUntil(server_proactor_, kServerPolling,
              [&] { return accept_state_.callback_count == 1; });
  }
  // Cancellation can lose after the peer accepted but before client
  // publication. Stop joins any such claimed server callback as well.
  StopAndFreeListener();
  server_connection_ = accept_state_.connection;
  accept_state_.connection = nullptr;
}

TEST_F(TransportTest, CancellationAfterPublicationLeavesConnectionUsable) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  iree_net_transport_connect_operation_cancel(&connect_state_.operation);
  auto endpoints = OpenEndpoints();
  EXPECT_NE(endpoints.first.self, nullptr);
  EXPECT_NE(endpoints.second.self, nullptr);
  EXPECT_EQ(connect_state_.callback_count, 1);
}

TEST_F(TransportTest, RoutesBidirectionalMessagesOnOwningProactors) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();
  ASSERT_NE(client_endpoint.self, nullptr);
  ASSERT_NE(server_endpoint.self, nullptr);

  client_messages_.current_poll_side = &current_poll_side_;
  client_messages_.expected_poll_side = kClientPolling;
  server_messages_.current_poll_side = &current_poll_side_;
  server_messages_.expected_poll_side = kServerPolling;
  iree_net_message_endpoint_set_callbacks(client_endpoint,
                                          client_messages_.callbacks());
  iree_net_message_endpoint_set_callbacks(server_endpoint,
                                          server_messages_.callbacks());
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));

  char client_prefix[] = "client-";
  char client_suffix[] = "message";
  iree_async_span_t client_span =
      iree_async_span_from_ptr(client_suffix, sizeof(client_suffix) - 1);
  client_send_.current_poll_side = &current_poll_side_;
  client_send_.expected_poll_side = kClientPolling;
  client_send_.expected_bytes =
      (sizeof(client_prefix) - 1) + (sizeof(client_suffix) - 1);
  iree_net_message_endpoint_send_params_t send_params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(client_prefix, sizeof(client_prefix) - 1)),
      .data = iree_async_span_list_make(&client_span, 1),
      .completion_callback = client_send_.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(client_endpoint, &send_params));
  client_prefix[0] = 'X';
  PollBothUntil([&] {
    return server_messages_.messages.size() == 1 &&
           client_send_.callback_count == 1;
  });
  EXPECT_EQ(server_messages_.messages[0], "client-message");
  EXPECT_EQ(client_send_.status_code, IREE_STATUS_OK);

  char server_payload[] = "server-message";
  server_send_.current_poll_side = &current_poll_side_;
  server_send_.expected_poll_side = kServerPolling;
  server_send_.expected_bytes = sizeof(server_payload) - 1;
  send_params = {
      .generated_prefix =
          iree_net_send_prefix_from_bytes(iree_make_const_byte_span(
              server_payload, sizeof(server_payload) - 1)),
      .data = iree_async_span_list_empty(),
      .completion_callback = server_send_.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(server_endpoint, &send_params));
  server_payload[0] = 'X';
  PollBothUntil([&] {
    return client_messages_.messages.size() == 1 &&
           server_send_.callback_count == 1;
  });
  EXPECT_EQ(client_messages_.messages[0], "server-message");
  EXPECT_EQ(server_send_.status_code, IREE_STATUS_OK);
  EXPECT_EQ(client_messages_.error_count, 0);
  EXPECT_EQ(server_messages_.error_count, 0);
}

TEST_F(TransportTest, CallbackHandoffPreservesQueuedMessageOrder) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();
  ASSERT_NE(client_endpoint.self, nullptr);
  ASSERT_NE(server_endpoint.self, nullptr);

  client_messages_.current_poll_side = &current_poll_side_;
  client_messages_.expected_poll_side = kClientPolling;
  iree_net_message_endpoint_set_callbacks(client_endpoint,
                                          client_messages_.callbacks());
  ProtocolHandoffState handoff_state;
  handoff_state.current_poll_side = &current_poll_side_;
  handoff_state.expected_poll_side = kServerPolling;
  handoff_state.endpoint = server_endpoint;
  iree_net_message_endpoint_set_callbacks(server_endpoint,
                                          handoff_state.bootstrap_callbacks());
  std::array<std::string, 3> messages = {
      "bootstrap",
      "control-1",
      "control-2",
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));
  // This test requires all messages to be admitted before the server polls so
  // they exercise callback handoff while queued in the transport.
  WaitForSendSlots(client_endpoint, messages.size());

  std::array<SendState, 3> send_states;
  for (iree_host_size_t i = 0; i < messages.size(); ++i) {
    send_states[i].current_poll_side = &current_poll_side_;
    send_states[i].expected_poll_side = kClientPolling;
    send_states[i].expected_bytes = messages[i].size();
    iree_net_message_endpoint_send_params_t send_params = {
        .generated_prefix = iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(messages[i].data(), messages[i].size())),
        .data = iree_async_span_list_empty(),
        .completion_callback = send_states[i].callback(),
    };
    IREE_ASSERT_OK(
        iree_net_message_endpoint_send(client_endpoint, &send_params));
  }

  PollBothUntil([&] {
    const bool all_messages_received =
        handoff_state.bootstrap_messages.size() +
            handoff_state.operational_messages.size() ==
        messages.size();
    const bool all_sends_completed = std::all_of(
        send_states.begin(), send_states.end(),
        [](const SendState& state) { return state.callback_count == 1; });
    return all_messages_received && all_sends_completed;
  });

  EXPECT_FALSE(handoff_state.message_callback_active);
  EXPECT_EQ(handoff_state.bootstrap_messages,
            std::vector<std::string>({"bootstrap"}));
  EXPECT_EQ(handoff_state.operational_messages,
            std::vector<std::string>({"control-1", "control-2"}));
  EXPECT_EQ(handoff_state.error_count, 0);
  for (const SendState& send_state : send_states) {
    EXPECT_EQ(send_state.status_code, IREE_STATUS_OK);
  }
}

TEST_F(TransportTest, CarriesControlDataAndGoaway) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();
  ASSERT_NE(client_endpoint.self, nullptr);
  ASSERT_NE(server_endpoint.self, nullptr);

  client_control_messages_.current_poll_side = &current_poll_side_;
  client_control_messages_.expected_poll_side = kClientPolling;
  server_control_messages_.current_poll_side = &current_poll_side_;
  server_control_messages_.expected_poll_side = kServerPolling;
  IREE_ASSERT_OK(iree_net_control_channel_allocate(
      client_endpoint, client_control_messages_.callbacks(),
      iree_allocator_system(), &client_control_channel_));
  IREE_ASSERT_OK(iree_net_control_channel_allocate(
      server_endpoint, server_control_messages_.callbacks(),
      iree_allocator_system(), &server_control_channel_));
  iree_net_control_channel_attach(client_control_channel_);
  iree_net_control_channel_attach(server_control_channel_);
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));
  WaitForSendSlots(client_endpoint, 2);

  std::string borrowed_payload = "borrowed request";
  iree_async_span_t borrowed_span = iree_async_span_from_ptr(
      borrowed_payload.data(), borrowed_payload.size());
  SendState borrowed_send;
  borrowed_send.current_poll_side = &current_poll_side_;
  borrowed_send.expected_poll_side = kClientPolling;
  borrowed_send.expected_bytes =
      IREE_NET_CONTROL_MESSAGE_HEADER_SIZE + borrowed_payload.size();
  IREE_ASSERT_OK(iree_net_control_channel_send_data(
      client_control_channel_, 3, iree_async_span_list_make(&borrowed_span, 1),
      borrowed_send.callback()));

  std::string copied_prefix(4097, 'p');
  std::string copied_suffix(4096, 's');
  const std::string expected_copy = copied_prefix + copied_suffix;
  iree_async_span_t copied_spans[] = {
      iree_async_span_from_ptr(copied_prefix.data(), copied_prefix.size()),
      iree_async_span_from_ptr(copied_suffix.data(), copied_suffix.size()),
  };
  SendState copied_send;
  copied_send.current_poll_side = &current_poll_side_;
  copied_send.expected_poll_side = kServerPolling;
  copied_send.expected_bytes =
      IREE_NET_CONTROL_MESSAGE_HEADER_SIZE + expected_copy.size();
  IREE_ASSERT_OK(iree_net_control_channel_send_data_copy(
      server_control_channel_, 5, iree_async_span_list_make(copied_spans, 2),
      copied_send.callback()));
  std::fill(copied_prefix.begin(), copied_prefix.end(), 'x');
  std::fill(copied_suffix.begin(), copied_suffix.end(), 'x');

  SendState goaway_send;
  goaway_send.current_poll_side = &current_poll_side_;
  goaway_send.expected_poll_side = kClientPolling;
  goaway_send.expected_bytes = IREE_NET_CONTROL_MESSAGE_HEADER_SIZE;
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      client_control_channel_, 42, goaway_send.callback()));

  PollBothUntil([&] {
    return server_control_messages_.messages.size() == 1 &&
           server_control_messages_.goaway_count == 1 &&
           client_control_messages_.messages.size() == 1 &&
           borrowed_send.callback_count == 1 &&
           copied_send.callback_count == 1 && goaway_send.callback_count == 1;
  });

  EXPECT_EQ(server_control_messages_.messages,
            std::vector<std::string>({"borrowed request"}));
  EXPECT_EQ(server_control_messages_.flags,
            std::vector<iree_net_control_data_flags_t>({3}));
  EXPECT_EQ(server_control_messages_.goaway_reason, 42u);
  EXPECT_EQ(client_control_messages_.messages,
            std::vector<std::string>({expected_copy}));
  EXPECT_EQ(client_control_messages_.flags,
            std::vector<iree_net_control_data_flags_t>({5}));
  EXPECT_EQ(client_control_messages_.error_count, 0);
  EXPECT_EQ(server_control_messages_.error_count, 0);
  EXPECT_EQ(borrowed_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(copied_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(goaway_send.status_code, IREE_STATUS_OK);
}

TEST_F(TransportTest, CarriesQueueCommandsAndAdvances) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();
  ASSERT_NE(client_endpoint.self, nullptr);
  ASSERT_NE(server_endpoint.self, nullptr);

  client_queue_messages_.current_poll_side = &current_poll_side_;
  client_queue_messages_.expected_poll_side = kClientPolling;
  server_queue_messages_.current_poll_side = &current_poll_side_;
  server_queue_messages_.expected_poll_side = kServerPolling;
  IREE_ASSERT_OK(iree_net_queue_channel_allocate(
      client_endpoint, client_queue_messages_.callbacks(),
      iree_allocator_system(), &client_queue_channel_));
  IREE_ASSERT_OK(iree_net_queue_channel_allocate(
      server_endpoint, server_queue_messages_.callbacks(),
      iree_allocator_system(), &server_queue_channel_));
  iree_net_queue_channel_attach(client_queue_channel_);
  iree_net_queue_channel_attach(server_queue_channel_);
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));

  QueueBuildState command_builder = {
      .wait_frontier = {{3, 5}, {7, 11}},
      .signal_frontier = {{9, 13}},
      .generated_payload = "generated:",
  };
  std::string command_suffix = "borrowed-command";
  iree_async_span_t command_span =
      iree_async_span_from_ptr(command_suffix.data(), command_suffix.size());
  SendState command_send;
  command_send.current_poll_side = &current_poll_side_;
  command_send.expected_poll_side = kClientPolling;
  command_send.expected_bytes = IREE_NET_QUEUE_MESSAGE_HEADER_SIZE +
                                (command_builder.wait_frontier.size() +
                                 command_builder.signal_frontier.size()) *
                                    IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE +
                                command_builder.generated_payload.size() +
                                command_suffix.size();
  const iree_net_queue_channel_send_params_t command_params =
      command_builder.params(iree_async_span_list_make(&command_span, 1),
                             command_send.callback());
  IREE_ASSERT_OK(iree_net_queue_channel_send_command(client_queue_channel_, 7,
                                                     &command_params));

  QueueBuildState advance_builder = {
      .wait_frontier = {},
      .signal_frontier = {{9, 13}, {17, 19}},
      .generated_payload = "advance:",
  };
  std::string advance_suffix = "complete";
  iree_async_span_t advance_span =
      iree_async_span_from_ptr(advance_suffix.data(), advance_suffix.size());
  SendState advance_send;
  advance_send.current_poll_side = &current_poll_side_;
  advance_send.expected_poll_side = kServerPolling;
  advance_send.expected_bytes = IREE_NET_QUEUE_MESSAGE_HEADER_SIZE +
                                advance_builder.signal_frontier.size() *
                                    IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE +
                                advance_builder.generated_payload.size() +
                                advance_suffix.size();
  const iree_net_queue_channel_send_params_t advance_params =
      advance_builder.params(iree_async_span_list_make(&advance_span, 1),
                             advance_send.callback());
  IREE_ASSERT_OK(iree_net_queue_channel_send_advance(server_queue_channel_,
                                                     &advance_params));

  PollBothUntil([&] {
    return server_queue_messages_.command_count == 1 &&
           client_queue_messages_.advance_count == 1 &&
           command_send.callback_count == 1 && advance_send.callback_count == 1;
  });

  EXPECT_EQ(server_queue_messages_.command_queue_id, 7u);
  ASSERT_EQ(server_queue_messages_.command_wait_frontier.size(), 2u);
  EXPECT_EQ(server_queue_messages_.command_wait_frontier[0].axis, 3u);
  EXPECT_EQ(server_queue_messages_.command_wait_frontier[0].epoch, 5u);
  EXPECT_EQ(server_queue_messages_.command_wait_frontier[1].axis, 7u);
  EXPECT_EQ(server_queue_messages_.command_wait_frontier[1].epoch, 11u);
  ASSERT_EQ(server_queue_messages_.command_signal_frontier.size(), 1u);
  EXPECT_EQ(server_queue_messages_.command_signal_frontier[0].axis, 9u);
  EXPECT_EQ(server_queue_messages_.command_signal_frontier[0].epoch, 13u);
  EXPECT_EQ(server_queue_messages_.command_payload,
            "generated:borrowed-command");

  ASSERT_EQ(client_queue_messages_.advance_signal_frontier.size(), 2u);
  EXPECT_EQ(client_queue_messages_.advance_signal_frontier[0].axis, 9u);
  EXPECT_EQ(client_queue_messages_.advance_signal_frontier[0].epoch, 13u);
  EXPECT_EQ(client_queue_messages_.advance_signal_frontier[1].axis, 17u);
  EXPECT_EQ(client_queue_messages_.advance_signal_frontier[1].epoch, 19u);
  EXPECT_EQ(client_queue_messages_.advance_payload, "advance:complete");
  EXPECT_EQ(client_queue_messages_.error_count, 0);
  EXPECT_EQ(server_queue_messages_.error_count, 0);
  EXPECT_EQ(command_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(advance_send.status_code, IREE_STATUS_OK);
}

TEST_F(TransportTest, CarriesCreditBoundedBulkTransfer) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();
  ASSERT_NE(client_endpoint.self, nullptr);
  ASSERT_NE(server_endpoint.self, nullptr);

  client_bulk_messages_.current_poll_side = &current_poll_side_;
  client_bulk_messages_.expected_poll_side = kClientPolling;
  server_bulk_messages_.current_poll_side = &current_poll_side_;
  server_bulk_messages_.expected_poll_side = kServerPolling;
  IREE_ASSERT_OK(iree_net_bulk_channel_allocate(
      client_endpoint, client_bulk_messages_.callbacks(),
      iree_allocator_system(), &client_bulk_channel_));
  IREE_ASSERT_OK(iree_net_bulk_channel_allocate(
      server_endpoint, server_bulk_messages_.callbacks(),
      iree_allocator_system(), &server_bulk_channel_));
  iree_net_bulk_channel_attach(client_bulk_channel_);
  iree_net_bulk_channel_attach(server_bulk_channel_);
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));
  // Semantic bulk credit does not imply reciprocal transport readiness. Wait
  // until both endpoints advertise ordinary multi-send capacity before the
  // test issues adjacent START/DATA and COMPLETE/ABORT operations.
  WaitForSendSlots(client_endpoint, 2);
  WaitForSendSlots(server_endpoint, 2);

  SendState initial_credit_send;
  initial_credit_send.current_poll_side = &current_poll_side_;
  initial_credit_send.expected_poll_side = kServerPolling;
  initial_credit_send.expected_bytes = IREE_NET_BULK_MESSAGE_HEADER_SIZE;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_credit(
      server_bulk_channel_, 1, initial_credit_send.callback()));
  PollBothUntil([&] {
    return client_bulk_messages_.credit_count == 1 &&
           initial_credit_send.callback_count == 1;
  });
  EXPECT_EQ(client_bulk_messages_.credit_delta, 1u);
  EXPECT_EQ(client_bulk_messages_.available_credit_count, 1u);
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(client_bulk_channel_),
            1u);

  std::string first_chunk(32 * 1024, 'a');
  std::string second_chunk(32 * 1024, 'b');
  const uint64_t transfer_length =
      static_cast<uint64_t>(first_chunk.size() + second_chunk.size());
  server_bulk_messages_.retain_next_data = true;

  SendState start_send;
  start_send.current_poll_side = &current_poll_side_;
  start_send.expected_poll_side = kClientPolling;
  start_send.expected_bytes = IREE_NET_BULK_MESSAGE_HEADER_SIZE;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_start(
      client_bulk_channel_, 3, transfer_length, start_send.callback()));

  iree_async_span_t first_spans[] = {
      iree_async_span_from_ptr(first_chunk.data(), first_chunk.size() / 2),
      iree_async_span_from_ptr(first_chunk.data() + first_chunk.size() / 2,
                               first_chunk.size() / 2),
  };
  SendState first_data_send;
  first_data_send.current_poll_side = &current_poll_side_;
  first_data_send.expected_poll_side = kClientPolling;
  first_data_send.expected_bytes =
      IREE_NET_BULK_MESSAGE_HEADER_SIZE + first_chunk.size();
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      client_bulk_channel_, 3, 0, iree_async_span_list_make(first_spans, 2),
      first_data_send.callback()));
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(client_bulk_channel_),
            0u);

  iree_async_span_t second_span =
      iree_async_span_from_ptr(second_chunk.data(), second_chunk.size());
  SendState blocked_data_send;
  blocked_data_send.current_poll_side = &current_poll_side_;
  blocked_data_send.expected_poll_side = kClientPolling;
  blocked_data_send.expected_bytes = 0;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      client_bulk_channel_, 3, first_chunk.size(),
      iree_async_span_list_make(&second_span, 1),
      blocked_data_send.callback()));
  EXPECT_EQ(blocked_data_send.callback_count, 0);

  PollBothUntil([&] {
    return server_bulk_messages_.start_count == 1 &&
           server_bulk_messages_.data_payloads.size() == 1 &&
           start_send.callback_count == 1 &&
           first_data_send.callback_count == 1 &&
           blocked_data_send.callback_count == 1;
  });
  EXPECT_EQ(blocked_data_send.status_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  ASSERT_NE(server_bulk_messages_.retained_data_lease.release.fn, nullptr);
  EXPECT_EQ(server_bulk_messages_.start_transfer_id, 3u);
  EXPECT_EQ(server_bulk_messages_.start_total_length, transfer_length);
  EXPECT_EQ(server_bulk_messages_.data_offsets, (std::vector<uint64_t>{0}));
  EXPECT_EQ(server_bulk_messages_.data_payloads,
            (std::vector<std::string>{first_chunk}));

  iree_async_buffer_lease_release(&server_bulk_messages_.retained_data_lease);
  SendState replenished_credit_send;
  replenished_credit_send.current_poll_side = &current_poll_side_;
  replenished_credit_send.expected_poll_side = kServerPolling;
  replenished_credit_send.expected_bytes = IREE_NET_BULK_MESSAGE_HEADER_SIZE;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_credit(
      server_bulk_channel_, 1, replenished_credit_send.callback()));
  PollBothUntil([&] {
    return client_bulk_messages_.credit_count == 2 &&
           replenished_credit_send.callback_count == 1;
  });
  EXPECT_EQ(client_bulk_messages_.credit_delta, 1u);
  EXPECT_EQ(client_bulk_messages_.available_credit_count, 1u);

  SendState second_data_send;
  second_data_send.current_poll_side = &current_poll_side_;
  second_data_send.expected_poll_side = kClientPolling;
  second_data_send.expected_bytes =
      IREE_NET_BULK_MESSAGE_HEADER_SIZE + second_chunk.size();
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      client_bulk_channel_, 3, first_chunk.size(),
      iree_async_span_list_make(&second_span, 1), second_data_send.callback()));
  SendState transfer_complete_send;
  transfer_complete_send.current_poll_side = &current_poll_side_;
  transfer_complete_send.expected_poll_side = kClientPolling;
  transfer_complete_send.expected_bytes = IREE_NET_BULK_MESSAGE_HEADER_SIZE;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_complete(
      client_bulk_channel_, 3, transfer_complete_send.callback()));

  PollBothUntil([&] {
    return server_bulk_messages_.data_payloads.size() == 2 &&
           server_bulk_messages_.complete_count == 1 &&
           second_data_send.callback_count == 1 &&
           transfer_complete_send.callback_count == 1;
  });
  EXPECT_EQ(server_bulk_messages_.data_offsets,
            (std::vector<uint64_t>{0, first_chunk.size()}));
  EXPECT_EQ(server_bulk_messages_.data_payloads,
            (std::vector<std::string>{first_chunk, second_chunk}));
  EXPECT_EQ(server_bulk_messages_.complete_transfer_id, 3u);

  SendState acknowledgment_send;
  acknowledgment_send.current_poll_side = &current_poll_side_;
  acknowledgment_send.expected_poll_side = kServerPolling;
  acknowledgment_send.expected_bytes = IREE_NET_BULK_MESSAGE_HEADER_SIZE;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_complete(
      server_bulk_channel_, 3, acknowledgment_send.callback()));
  std::string abort_detail = "cancelled by receiver";
  iree_async_span_t abort_span =
      iree_async_span_from_ptr(abort_detail.data(), abort_detail.size());
  SendState abort_send;
  abort_send.current_poll_side = &current_poll_side_;
  abort_send.expected_poll_side = kServerPolling;
  abort_send.expected_bytes =
      IREE_NET_BULK_MESSAGE_HEADER_SIZE + abort_detail.size();
  IREE_ASSERT_OK(iree_net_bulk_channel_send_abort(
      server_bulk_channel_, 5, iree_async_span_list_make(&abort_span, 1),
      abort_send.callback()));

  PollBothUntil([&] {
    return client_bulk_messages_.complete_count == 1 &&
           client_bulk_messages_.abort_count == 1 &&
           acknowledgment_send.callback_count == 1 &&
           abort_send.callback_count == 1;
  });

  EXPECT_EQ(client_bulk_messages_.complete_transfer_id, 3u);
  EXPECT_EQ(client_bulk_messages_.abort_transfer_id, 5u);
  EXPECT_EQ(client_bulk_messages_.abort_detail, abort_detail);
  EXPECT_EQ(client_bulk_messages_.error_count, 0);
  EXPECT_EQ(server_bulk_messages_.error_count, 0);
  EXPECT_EQ(start_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(first_data_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(second_data_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(transfer_complete_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(acknowledgment_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(abort_send.status_code, IREE_STATUS_OK);
}

TEST_F(TransportTest, GeneratesLargeTransientPrefixWithoutSizeCliff) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();

  server_messages_.current_poll_side = &current_poll_side_;
  server_messages_.expected_poll_side = kServerPolling;
  iree_net_message_endpoint_set_callbacks(server_endpoint,
                                          server_messages_.callbacks());
  client_messages_.current_poll_side = &current_poll_side_;
  client_messages_.expected_poll_side = kClientPolling;
  iree_net_message_endpoint_set_callbacks(client_endpoint,
                                          client_messages_.callbacks());
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));

  std::string prefix(16 * 1024 + 1, 'p');
  const std::string expected = prefix;
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  send_state.expected_bytes = prefix.size();
  iree_net_message_endpoint_send_params_t send_params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(prefix.data(), prefix.size())),
      .data = iree_async_span_list_empty(),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(client_endpoint, &send_params));
  std::fill(prefix.begin(), prefix.end(), 'x');

  PollBothUntil([&] {
    return server_messages_.messages.size() == 1 &&
           send_state.callback_count == 1;
  });
  EXPECT_EQ(server_messages_.messages[0], expected);
  EXPECT_EQ(send_state.status_code, IREE_STATUS_OK);
}

TEST_F(TransportTest, SaturatedAdmissionResumesFromCompletion) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();
  client_messages_.current_poll_side = &current_poll_side_;
  client_messages_.expected_poll_side = kClientPolling;
  server_messages_.current_poll_side = &current_poll_side_;
  server_messages_.expected_poll_side = kServerPolling;
  iree_net_message_endpoint_set_callbacks(client_endpoint,
                                          client_messages_.callbacks());
  iree_net_message_endpoint_set_callbacks(server_endpoint,
                                          server_messages_.callbacks());
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));
  // Qualify ordinary multi-send admission, not the single preactivation slot
  // a transport may expose while the reciprocal endpoint becomes ready.
  WaitForSendSlots(client_endpoint, 2);

  struct Producer {
    // Endpoint receiving a retry only from an accepted send's completion.
    iree_net_message_endpoint_t endpoint;
    // Current poll owner, checked by every completion callback.
    int* poll_side;
    // Prefix invocations are an observable admission witness.
    size_t writes = 0;
    // Number of terminal callbacks received.
    size_t completions = 0;
    // Whether the rejected send still needs a completion-driven retry.
    bool retry_pending = false;
    // A missing retry edge terminates the witness as a failure, not a hang.
    bool failed = false;

    static iree_status_t Write(void* user_data, iree_byte_span_t target) {
      ++static_cast<Producer*>(user_data)->writes;
      memset(target.data, 'p', target.data_length);
      return iree_ok_status();
    }

    static void Complete(void* user_data, iree_status_t status,
                         iree_host_size_t length) {
      auto* self = static_cast<Producer*>(user_data);
      EXPECT_EQ(*self->poll_side, kClientPolling);
      EXPECT_EQ(length, 32u);
      self->failed |= !iree_status_is_ok(status);
      IREE_EXPECT_OK(status);
      ++self->completions;
      if (!self->retry_pending) {
        return;
      }
      self->retry_pending = false;
      auto budget = iree_net_message_endpoint_query_send_budget(self->endpoint);
      if (!budget.slots || budget.bytes < 32) {
        ADD_FAILURE() << "completion did not restore send admission";
        self->failed = true;
        return;
      }
      status = self->Send();
      self->failed |= !iree_status_is_ok(status);
      IREE_EXPECT_OK(status);
    }

    iree_status_t Send() {
      iree_net_message_endpoint_send_params_t params = {
          .generated_prefix = {32, Write, this},
          .data = iree_async_span_list_empty(),
          .completion_callback = {Complete, this},
      };
      return iree_net_message_endpoint_send(endpoint, &params);
    }
  } producer{client_endpoint, &current_poll_side_};
  ScopedConnectionDrain drain{this};

  const auto initial =
      iree_net_message_endpoint_query_send_budget(client_endpoint);
  ASSERT_GT(initial.slots, 0u);
  ASSERT_GE(initial.bytes, 32u);
  // No polling means accepted operations cannot return their local slots.
  for (uint32_t i = 0; i < initial.slots; ++i) {
    IREE_ASSERT_OK(producer.Send());
  }
  EXPECT_EQ(producer.writes, initial.slots);
  EXPECT_EQ(producer.completions, 0u);
  EXPECT_EQ(iree_net_message_endpoint_query_send_budget(client_endpoint).slots,
            0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, producer.Send());
  EXPECT_EQ(producer.writes, initial.slots);
  EXPECT_EQ(producer.completions, 0u);
  producer.retry_pending = true;
  const size_t expected_count = (size_t)initial.slots + 1;
  PollBothUntil([&] {
    return producer.failed ||
           (producer.completions == expected_count &&
            server_messages_.messages.size() == expected_count);
  });
  EXPECT_FALSE(producer.failed);
  EXPECT_FALSE(producer.retry_pending);
  EXPECT_EQ(producer.writes, expected_count);
  EXPECT_EQ(producer.completions, expected_count);
  ASSERT_EQ(server_messages_.messages.size(), expected_count);
  for (const auto& message : server_messages_.messages) {
    EXPECT_EQ(message, std::string(32, 'p'));
  }
  EXPECT_GE(iree_net_message_endpoint_query_send_budget(client_endpoint).slots,
            initial.slots);
  EXPECT_EQ(client_messages_.error_count, 0);
  EXPECT_EQ(server_messages_.error_count, 0);
}

TEST_F(TransportTest, MovedMessagesSurviveConnectionTeardown) {
  struct RetainedMessages {
    // Original callback views, never replaced with test-owned message copies.
    std::vector<iree_const_byte_span_t> messages;
    // Movable leases owning the corresponding original byte views.
    std::vector<iree_async_buffer_lease_t> leases;
    // Terminal notification count; peer teardown may report its closure.
    int errors = 0;

    ~RetainedMessages() {
      for (auto& lease : leases) {
        iree_async_buffer_lease_release(&lease);
      }
    }

    static iree_status_t Receive(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
      auto* self = static_cast<RetainedMessages*>(user_data);
      self->messages.push_back(message);
      self->leases.push_back(*lease);
      *lease = {};
      return iree_ok_status();
    }

    static void Error(void* user_data, iree_status_t status) {
      ++static_cast<RetainedMessages*>(user_data)->errors;
      iree_status_free(status);
    }
  } retained;

  // Hold more intact messages than the native receive inventory, followed by
  // fragmented messages larger than the entire receive slab. None can be
  // released until later messages arrive, as with bulk DATA then COMPLETE.
  std::vector<std::string> payloads;
  for (int i = 0; i < 64; ++i) {
    payloads.emplace_back(31, static_cast<char>(i));
  }
  payloads.emplace_back(65535, 'b');
  payloads.emplace_back(65537, 'c');
  payloads.emplace_back(1024 * 1024 + 1, 'd');
  std::vector<SendState> sends(payloads.size());
  SendState independent_send;
  ScopedConnectionDrain drain{this};

  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  auto [client_endpoint, server_endpoint] = OpenEndpoints();
  client_messages_.current_poll_side = &current_poll_side_;
  client_messages_.expected_poll_side = kClientPolling;
  iree_net_message_endpoint_set_callbacks(client_endpoint,
                                          client_messages_.callbacks());
  iree_net_message_endpoint_set_callbacks(
      server_endpoint,
      {RetainedMessages::Receive, RetainedMessages::Error, &retained});
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));
  WaitForSendSlots(client_endpoint, 4);

  for (size_t i = 0; i < payloads.size(); ++i) {
    sends[i].current_poll_side = &current_poll_side_;
    sends[i].expected_poll_side = kClientPolling;
    sends[i].expected_bytes = payloads[i].size();
    iree_async_span_t span =
        iree_async_span_from_ptr(payloads[i].data(), payloads[i].size());
    iree_net_message_endpoint_send_params_t params = {
        .generated_prefix = iree_net_send_prefix_empty(),
        .data = iree_async_span_list_make(&span, 1),
        .completion_callback = sends[i].callback(),
    };
    IREE_ASSERT_OK(iree_net_message_endpoint_send(client_endpoint, &params));
    // Acknowledge each delivery before sending another so coalescing cannot
    // hide native-storage pressure behind framing's earlier-frame copies.
    PollBothUntil([&] {
      return retained.messages.size() == i + 1 && sends[i].callback_count == 1;
    });
  }
  EXPECT_EQ(retained.errors, 0);

  // A separate endpoint must progress while the first retains its messages.
  auto [other_client, other_server] = OpenEndpoints();
  server_messages_.current_poll_side = &current_poll_side_;
  server_messages_.expected_poll_side = kServerPolling;
  iree_net_message_endpoint_set_callbacks(other_client,
                                          client_messages_.callbacks());
  iree_net_message_endpoint_set_callbacks(other_server,
                                          server_messages_.callbacks());
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(other_client));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(other_server));
  WaitForSendSlots(other_client, 1);
  independent_send.current_poll_side = &current_poll_side_;
  independent_send.expected_poll_side = kClientPolling;
  independent_send.expected_bytes = 7;
  iree_net_message_endpoint_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span("control", 7)),
      .data = iree_async_span_list_empty(),
      .completion_callback = independent_send.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(other_client, &params));
  PollBothUntil([&] {
    return independent_send.callback_count == 1 &&
           server_messages_.messages.size() == 1;
  });
  EXPECT_EQ(server_messages_.messages[0], "control");
  EXPECT_EQ(retained.errors, 0);

  StopAndFreeListener();
  DeactivateAndRelease(client_connection_, client_proactor_, kClientPolling);
  // Observe actual peer departure while the receiving connection and moved
  // leases are still live, before asking the receiver to deactivate locally.
  PollUntil(server_proactor_, kServerPolling,
            [&] { return retained.errors != 0; });
  EXPECT_EQ(retained.errors, 1);
  DeactivateAndRelease(server_connection_, server_proactor_, kServerPolling);
  iree_net_transport_factory_release(factory_);
  factory_ = nullptr;

  // No connection or endpoint remains. Lease-backed native registrations may
  // still need the poll owner for final release, even after connection drain.
  std::atomic<bool> consumer_done = false;
  std::thread consumer([&] {
    for (size_t offset = 0; offset < payloads.size(); ++offset) {
      size_t i = payloads.size() - offset - 1;
      EXPECT_EQ(retained.messages[i].data_length, payloads[i].size());
      EXPECT_EQ(memcmp(retained.messages[i].data, payloads[i].data(),
                       payloads[i].size()),
                0);
      iree_async_buffer_lease_release(&retained.leases[i]);
    }
    consumer_done.store(true, std::memory_order_release);
    iree_async_proactor_wake(server_proactor_);
  });
  PollUntil(server_proactor_, kServerPolling,
            [&] { return consumer_done.load(std::memory_order_acquire); });
  consumer.join();
  for (const auto& send : sends) {
    EXPECT_EQ(send.callback_count, 1);
  }
}

TEST_F(TransportTest, DeactivationCancelsPendingEndpointReadyCallback) {
  ASSERT_NO_FATAL_FAILURE(EstablishConnection());
  EndpointReadyState* ready_state =
      SubmitOpenEndpoint(client_connection_, kClientPolling);
  ASSERT_NE(ready_state, nullptr);

  DeactivateState deactivate_state;
  deactivate_state.current_poll_side = &current_poll_side_;
  deactivate_state.expected_poll_side = kClientPolling;
  current_poll_side_ = kClientPolling;
  iree_net_connection_deactivate(client_connection_,
                                 deactivate_state.callback());
  current_poll_side_ = kNotPolling;
  EXPECT_EQ(ready_state->callback_count, 0);
  EXPECT_FALSE(deactivate_state.completed);
  PollUntil(client_proactor_, kClientPolling, [&] {
    return ready_state->callback_count == 1 && deactivate_state.completed;
  });
  EXPECT_EQ(ready_state->status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(ready_state->endpoint.self, nullptr);
  iree_net_connection_release(client_connection_);
  client_connection_ = nullptr;
}

TEST_F(TransportTest, PendingSubmissionRetainsFactoryAndProactors) {
  ASSERT_NO_FATAL_FAILURE(CreateListener());
  SubmitConnect(
      iree_make_string_view(connect_address_.data(), connect_address_.size()));

  iree_net_transport_factory_release(factory_);
  factory_ = nullptr;
  ReleaseReceivePool(&client_receive_pool_);
  ReleaseReceivePool(&server_receive_pool_);
  iree_async_proactor_release(client_proactor_);
  owns_client_proactor_ = false;
  iree_async_proactor_release(server_proactor_);
  owns_server_proactor_ = false;

  DrainPendingConnection();
  ASSERT_EQ(connect_state_.status_code, IREE_STATUS_OK);
  ASSERT_EQ(accept_state_.status_code, IREE_STATUS_OK);
  ASSERT_NE(client_connection_, nullptr);
  ASSERT_NE(server_connection_, nullptr);
  auto endpoints = OpenEndpoints();
  EXPECT_NE(endpoints.first.self, nullptr);
  EXPECT_NE(endpoints.second.self, nullptr);
}

}  // namespace
}  // namespace iree::net::cts
