// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/session.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/net/cts/transport_backend.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::cts {
namespace {

enum SessionPollSide {
  kSessionNotPolling = 0,
  kSessionClientPolling = 1,
  kSessionServerPolling = 2,
};

struct SessionAcceptState {
  int* current_poll_side = nullptr;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_net_connection_t* connection = nullptr;

  static void OnAccept(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto* self = static_cast<SessionAcceptState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, kSessionServerPolling);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    self->connection = connection;
    iree_status_free(status);
  }

  iree_net_listener_accept_callback_t callback() {
    return {.fn = OnAccept, .user_data = this};
  }
};

struct SessionStopState {
  int* current_poll_side = nullptr;
  bool submitted = false;
  bool completed = false;

  static void OnStopped(void* user_data) {
    auto* self = static_cast<SessionStopState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, kSessionServerPolling);
    self->completed = true;
  }

  iree_net_listener_stopped_callback_t callback() {
    return {.fn = OnStopped, .user_data = this};
  }
};

struct SessionCallbackState {
  int* current_poll_side = nullptr;
  SessionPollSide expected_poll_side = kSessionNotPolling;
  bool deactivate_on_ready = false;
  bool ready_callback_active = false;
  bool deactivated_during_ready = false;
  int ready_count = 0;
  int goaway_count = 0;
  int error_count = 0;
  int deactivated_count = 0;
  uint32_t goaway_reason = 0;
  uint32_t remote_application_endpoint_count = 0;
  uint32_t remote_axis_count = 0;
  iree_net_bootstrap_capabilities_t negotiated_capabilities = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;
  std::string remote_application_data;
  std::vector<std::string> control_messages;
  std::vector<iree_net_control_data_flags_t> control_flags;

  static void OnReady(
      void* user_data, iree_net_session_t* session,
      const iree_net_bootstrap_peer_info_view_t* remote_peer,
      iree_net_bootstrap_capabilities_t negotiated_capabilities) {
    auto* self = static_cast<SessionCallbackState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_FALSE(self->ready_callback_active);
    self->ready_callback_active = true;
    ++self->ready_count;
    self->remote_application_endpoint_count =
        remote_peer->application_endpoint_count;
    self->remote_axis_count = remote_peer->axes.count;
    self->negotiated_capabilities = negotiated_capabilities;
    self->remote_application_data.assign(
        reinterpret_cast<const char*>(remote_peer->application_data.data),
        remote_peer->application_data.data_length);
    if (self->deactivate_on_ready) {
      iree_net_session_deactivate(session);
    }
    self->ready_callback_active = false;
  }

  static iree_status_t OnControlData(void* user_data,
                                     iree_net_session_t* session,
                                     iree_net_control_data_flags_t flags,
                                     iree_const_byte_span_t payload,
                                     iree_async_buffer_lease_t* lease) {
    (void)session;
    auto* self = static_cast<SessionCallbackState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    self->control_messages.emplace_back(
        reinterpret_cast<const char*>(payload.data), payload.data_length);
    self->control_flags.push_back(flags);
    return iree_ok_status();
  }

  static void OnGoaway(void* user_data, iree_net_session_t* session,
                       uint32_t reason_code) {
    (void)session;
    auto* self = static_cast<SessionCallbackState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->goaway_count;
    self->goaway_reason = reason_code;
  }

  static void OnError(void* user_data, iree_net_session_t* session,
                      iree_status_t status) {
    (void)session;
    auto* self = static_cast<SessionCallbackState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  static void OnDeactivated(void* user_data, iree_net_session_t* session) {
    (void)session;
    auto* self = static_cast<SessionCallbackState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    self->deactivated_during_ready |= self->ready_callback_active;
    ++self->deactivated_count;
  }

  iree_net_session_callbacks_t callbacks() {
    return {
        .on_ready = OnReady,
        .on_control_data = OnControlData,
        .on_goaway = OnGoaway,
        .on_error = OnError,
        .on_deactivated = OnDeactivated,
        .user_data = this,
    };
  }
};

struct SessionSendState {
  int* current_poll_side = nullptr;
  SessionPollSide expected_poll_side = kSessionNotPolling;
  iree_host_size_t expected_bytes = 0;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;

  static void OnComplete(void* user_data, iree_status_t status,
                         iree_host_size_t bytes_transferred) {
    auto* self = static_cast<SessionSendState*>(user_data);
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

struct SessionEndpointReadyState {
  int* current_poll_side = nullptr;
  SessionPollSide expected_poll_side = kSessionNotPolling;
  SessionCallbackState* session_state = nullptr;
  int callback_count = 0;
  bool callback_preceded_deactivation = false;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_net_message_endpoint_t endpoint = {};

  static void OnReady(void* user_data, iree_status_t status,
                      iree_net_message_endpoint_t endpoint) {
    auto* self = static_cast<SessionEndpointReadyState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->callback_count;
    self->callback_preceded_deactivation =
        self->session_state->deactivated_count == 0;
    self->status_code = iree_status_code(status);
    self->endpoint = endpoint;
    iree_status_free(status);
  }

  iree_net_endpoint_ready_callback_t callback() {
    return {.fn = OnReady, .user_data = this};
  }
};

struct SessionReceivePoolResources {
  iree_async_slab_t* slab = nullptr;
  iree_async_region_t* region = nullptr;
  iree_async_buffer_pool_t* pool = nullptr;
};

class OneShotAllocationGate {
 public:
  iree_allocator_t allocator() { return {this, Control}; }

  void ArmAllocationAtLeast(iree_host_size_t minimum_byte_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = true;
    minimum_byte_length_ = minimum_byte_length;
    entered_ = false;
    released_ = false;
    attempt_completed_ = false;
  }

  bool WaitUntilEnteredOrCompleted() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_ || attempt_completed_; });
    return entered_;
  }

  void CompleteAttempt() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    attempt_completed_ = true;
    condition_.notify_all();
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* gate = static_cast<OneShotAllocationGate*>(self);
    const bool is_allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_CALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_REALLOC;
    if (is_allocation) {
      const auto* alloc_params =
          static_cast<const iree_allocator_alloc_params_t*>(params);
      std::unique_lock<std::mutex> lock(gate->mutex_);
      if (gate->armed_ &&
          alloc_params->byte_length >= gate->minimum_byte_length_) {
        gate->armed_ = false;
        gate->entered_ = true;
        gate->condition_.notify_all();
        gate->condition_.wait(lock, [&] { return gate->released_; });
      }
    }
    iree_allocator_t system_allocator = iree_allocator_system();
    return system_allocator.ctl(system_allocator.self, command, params,
                                inout_ptr);
  }

  // Serializes gate state and its notifications.
  std::mutex mutex_;
  // Reports gate entry, release, and an allocation-free attempt.
  std::condition_variable condition_;
  // True when the next allocation command must stop at the gate.
  bool armed_ = false;
  // Minimum allocation extent eligible to enter the gate.
  iree_host_size_t minimum_byte_length_ = 0;
  // True after an allocation command enters the gate.
  bool entered_ = false;
  // True when the gated allocation command may continue.
  bool released_ = false;
  // True when the send attempt returns without entering the gate.
  bool attempt_completed_ = false;
};

static iree_status_t CreateSessionReceivePool(
    iree_async_proactor_t* proactor,
    SessionReceivePoolResources* out_resources) {
  *out_resources = SessionReceivePoolResources{};
  iree_async_slab_options_t slab_options = {};
  slab_options.buffer_size = 64 * 1024;
  slab_options.buffer_count = 16;
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
    *out_resources = SessionReceivePoolResources{};
  }
  return status;
}

static void ReleaseSessionReceivePool(SessionReceivePoolResources* resources) {
  iree_async_buffer_pool_release(resources->pool);
  iree_async_region_release(resources->region);
  iree_async_slab_release(resources->slab);
  *resources = SessionReceivePoolResources{};
}

class SessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = &GetTransportBackend();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &client_proactor_));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &server_proactor_));
    IREE_ASSERT_OK(
        CreateSessionReceivePool(client_proactor_, &client_receive_pool_));
    IREE_ASSERT_OK(
        CreateSessionReceivePool(server_proactor_, &server_receive_pool_));
    iree_status_t status =
        backend_->create_factory(factory_allocator_.allocator(), &factory_);
    if (iree_status_code(status) == IREE_STATUS_UNAVAILABLE) {
      iree_status_free(status);
      GTEST_SKIP() << backend_->name << " transport unavailable";
    }
    IREE_ASSERT_OK(status);
    accept_state_.current_poll_side = &current_poll_side_;
    stop_state_.current_poll_side = &current_poll_side_;
    client_state_.current_poll_side = &current_poll_side_;
    client_state_.expected_poll_side = kSessionClientPolling;
    server_state_.current_poll_side = &current_poll_side_;
    server_state_.expected_poll_side = kSessionServerPolling;
  }

  void TearDown() override {
    DeactivateSessions();
    StopAndFreeListener();
    if (!accepted_connection_ && accept_state_.connection) {
      accepted_connection_ = accept_state_.connection;
      accept_state_.connection = nullptr;
    }
    if (accepted_connection_) {
      DeactivateConnection(accepted_connection_, server_proactor_,
                           kSessionServerPolling);
    }
    iree_net_transport_factory_release(factory_);
    ReleaseSessionReceivePool(&client_receive_pool_);
    ReleaseSessionReceivePool(&server_receive_pool_);
    iree_async_proactor_release(client_proactor_);
    iree_async_proactor_release(server_proactor_);
  }

  void PollImmediate(iree_async_proactor_t* proactor, SessionPollSide side) {
    current_poll_side_ = side;
    iree_status_t status = iree_async_proactor_poll(
        proactor, iree_immediate_timeout(), /*out_completed_count=*/nullptr);
    current_poll_side_ = kSessionNotPolling;
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_free(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  void Poll(iree_async_proactor_t* proactor, SessionPollSide side) {
    current_poll_side_ = side;
    iree_status_t status = iree_async_proactor_poll(
        proactor, iree_infinite_timeout(), /*out_completed_count=*/nullptr);
    current_poll_side_ = kSessionNotPolling;
    IREE_ASSERT_OK(status);
  }

  void PollBothUntil(const std::function<bool()>& condition) {
    while (!condition()) {
      PollImmediate(client_proactor_, kSessionClientPolling);
      PollImmediate(server_proactor_, kSessionServerPolling);
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

  void StopAndFreeListener() {
    if (!listener_) {
      return;
    }
    if (!stop_state_.submitted) {
      IREE_ASSERT_OK(iree_net_listener_stop(listener_, stop_state_.callback()));
      stop_state_.submitted = true;
    }
    while (!stop_state_.completed) {
      Poll(server_proactor_, kSessionServerPolling);
    }
    iree_net_listener_free(listener_);
    listener_ = nullptr;
  }

  static void OnConnectionDeactivated(void* user_data) {
    *static_cast<bool*>(user_data) = true;
  }

  void DeactivateConnection(iree_net_connection_t*& connection,
                            iree_async_proactor_t* proactor,
                            SessionPollSide side) {
    bool completed = false;
    iree_net_connection_deactivate(
        connection, {.fn = OnConnectionDeactivated, .user_data = &completed});
    while (!completed) {
      Poll(proactor, side);
    }
    iree_net_connection_release(connection);
    connection = nullptr;
  }

  void DeactivateSessions() {
    if (client_session_ && iree_net_session_state(client_session_) !=
                               IREE_NET_SESSION_STATE_DEACTIVATED) {
      iree_net_session_deactivate(client_session_);
    }
    if (server_session_ && iree_net_session_state(server_session_) !=
                               IREE_NET_SESSION_STATE_DEACTIVATED) {
      iree_net_session_deactivate(server_session_);
    }
    PollBothUntil([&] {
      return (!client_session_ || iree_net_session_state(client_session_) ==
                                      IREE_NET_SESSION_STATE_DEACTIVATED) &&
             (!server_session_ || iree_net_session_state(server_session_) ==
                                      IREE_NET_SESSION_STATE_DEACTIVATED);
    });
    iree_net_session_release(client_session_);
    client_session_ = nullptr;
    iree_net_session_release(server_session_);
    server_session_ = nullptr;
  }

  void StartSessions(iree_net_session_options_t* client_options,
                     iree_net_session_options_t* server_options) {
    ASSERT_NO_FATAL_FAILURE(CreateListener());
    IREE_ASSERT_OK(iree_net_session_connect(
        factory_,
        iree_make_string_view(connect_address_.data(), connect_address_.size()),
        client_proactor_, client_receive_pool_.pool, client_options,
        client_state_.callbacks(), iree_allocator_system(), &client_session_));
    PollBothUntil([&] { return accept_state_.callback_count == 1; });
    ASSERT_EQ(accept_state_.status_code, IREE_STATUS_OK);
    accepted_connection_ = accept_state_.connection;
    accept_state_.connection = nullptr;
    IREE_ASSERT_OK(iree_net_session_accept(
        accepted_connection_, server_options, server_state_.callbacks(),
        iree_allocator_system(), &server_session_));
    iree_net_connection_release(accepted_connection_);
    accepted_connection_ = nullptr;
  }

  void EstablishSessions(iree_net_session_options_t* client_options,
                         iree_net_session_options_t* server_options) {
    ASSERT_NO_FATAL_FAILURE(StartSessions(client_options, server_options));
    PollBothUntil([&] {
      return (client_state_.ready_count || client_state_.error_count) &&
             (server_state_.ready_count || server_state_.error_count);
    });
    ASSERT_EQ(client_state_.ready_count, 1);
    ASSERT_EQ(server_state_.ready_count, 1);
  }

  iree_status_t AwaitClientControlSendReadiness() {
    uint8_t payload = 0;
    iree_async_span_t payload_span = iree_async_span_from_ptr(&payload, 1);
    SessionSendState send_state;
    send_state.current_poll_side = &current_poll_side_;
    send_state.expected_poll_side = kSessionClientPolling;
    send_state.expected_bytes = IREE_NET_CONTROL_MESSAGE_HEADER_SIZE + 1;

    while (true) {
      iree_status_t status = iree_net_session_send_control_data(
          client_session_, 0, iree_async_span_list_make(&payload_span, 1),
          send_state.callback());
      if (iree_status_is_ok(status)) {
        break;
      }
      if (!iree_status_is_resource_exhausted(status)) {
        return status;
      }
      iree_status_free(status);
      PollImmediate(client_proactor_, kSessionClientPolling);
      PollImmediate(server_proactor_, kSessionServerPolling);
    }

    PollBothUntil([&] { return send_state.callback_count == 1; });
    if (send_state.status_code != IREE_STATUS_OK) {
      return iree_make_status(send_state.status_code,
                              "control readiness send failed");
    }
    return iree_ok_status();
  }

  const TransportBackend* backend_ = nullptr;
  OneShotAllocationGate factory_allocator_;
  int current_poll_side_ = kSessionNotPolling;
  iree_async_proactor_t* client_proactor_ = nullptr;
  iree_async_proactor_t* server_proactor_ = nullptr;
  SessionReceivePoolResources client_receive_pool_;
  SessionReceivePoolResources server_receive_pool_;
  iree_net_transport_factory_t* factory_ = nullptr;
  iree_net_listener_t* listener_ = nullptr;
  iree_net_connection_t* accepted_connection_ = nullptr;
  iree_net_session_t* client_session_ = nullptr;
  iree_net_session_t* server_session_ = nullptr;
  std::string connect_address_;
  SessionAcceptState accept_state_;
  SessionStopState stop_state_;
  SessionCallbackState client_state_;
  SessionCallbackState server_state_;
};

TEST_F(SessionTest, EstablishesAndDrainsOperationalSession) {
  std::string client_metadata = "client metadata";
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.local_peer.capabilities =
      IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED;
  client_options.local_peer.application_endpoint_count = 1;
  client_options.local_peer.application_data =
      iree_make_const_byte_span(client_metadata.data(), client_metadata.size());
  client_options.required_capabilities =
      IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER;

  std::string server_metadata = "server metadata";
  iree_net_session_options_t server_options =
      iree_net_session_options_default();
  server_options.local_peer.capabilities =
      IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER;
  server_options.local_peer.application_endpoint_count = 1;
  server_options.local_peer.application_data =
      iree_make_const_byte_span(server_metadata.data(), server_metadata.size());

  ASSERT_NO_FATAL_FAILURE(EstablishSessions(&client_options, &server_options));
  EXPECT_EQ(client_state_.remote_application_data, "server metadata");
  EXPECT_EQ(server_state_.remote_application_data, "client metadata");
  EXPECT_EQ(client_state_.remote_application_endpoint_count, 1u);
  EXPECT_EQ(server_state_.remote_application_endpoint_count, 1u);
  EXPECT_EQ(client_state_.negotiated_capabilities,
            IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER);
  EXPECT_EQ(server_state_.negotiated_capabilities,
            IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER);

  SessionEndpointReadyState client_endpoint;
  client_endpoint.current_poll_side = &current_poll_side_;
  client_endpoint.expected_poll_side = kSessionClientPolling;
  client_endpoint.session_state = &client_state_;
  SessionEndpointReadyState server_endpoint;
  server_endpoint.current_poll_side = &current_poll_side_;
  server_endpoint.expected_poll_side = kSessionServerPolling;
  server_endpoint.session_state = &server_state_;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(client_session_,
                                                client_endpoint.callback()));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(server_session_,
                                                server_endpoint.callback()));
  PollBothUntil([&] {
    return client_endpoint.callback_count == 1 &&
           server_endpoint.callback_count == 1;
  });
  EXPECT_EQ(client_endpoint.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_endpoint.status_code, IREE_STATUS_OK);
  SessionEndpointReadyState extra_endpoint;
  extra_endpoint.current_poll_side = &current_poll_side_;
  extra_endpoint.expected_poll_side = kSessionClientPolling;
  extra_endpoint.session_state = &client_state_;
  iree_status_t extra_status = iree_net_session_open_endpoint(
      client_session_, extra_endpoint.callback());
  EXPECT_EQ(iree_status_code(extra_status), IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_status_free(extra_status);
  EXPECT_EQ(extra_endpoint.callback_count, 0);

  std::string payload = "copied session DATA";
  iree_async_span_t payload_span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SessionSendState data_send;
  data_send.current_poll_side = &current_poll_side_;
  data_send.expected_poll_side = kSessionClientPolling;
  data_send.expected_bytes = IREE_NET_CONTROL_MESSAGE_HEADER_SIZE +
                             static_cast<iree_host_size_t>(payload.size());
  IREE_ASSERT_OK(iree_net_session_send_control_data_copy(
      client_session_, 7, iree_async_span_list_make(&payload_span, 1),
      data_send.callback()));
  payload.assign(payload.size(), 'x');
  PollBothUntil([&] {
    return data_send.callback_count == 1 &&
           server_state_.control_messages.size() == 1;
  });
  EXPECT_EQ(data_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_state_.control_messages[0], "copied session DATA");
  EXPECT_EQ(server_state_.control_flags[0], 7u);

  SessionSendState goaway_send;
  goaway_send.current_poll_side = &current_poll_side_;
  goaway_send.expected_poll_side = kSessionClientPolling;
  goaway_send.expected_bytes = IREE_NET_CONTROL_MESSAGE_HEADER_SIZE;
  IREE_ASSERT_OK(iree_net_session_send_goaway(client_session_, 42,
                                              goaway_send.callback()));
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DRAINING);
  PollBothUntil([&] {
    return goaway_send.callback_count == 1 && server_state_.goaway_count == 1;
  });
  EXPECT_EQ(server_state_.goaway_reason, 42u);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_DRAINING);

  DeactivateSessions();
  EXPECT_EQ(client_state_.deactivated_count, 1);
  EXPECT_EQ(server_state_.deactivated_count, 1);
  EXPECT_EQ(client_state_.error_count, 0);
  EXPECT_EQ(server_state_.error_count, 0);
}

TEST_F(SessionTest, CopiedControlSendDoesNotSerializeDeactivation) {
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  iree_net_session_options_t server_options =
      iree_net_session_options_default();
  ASSERT_NO_FATAL_FAILURE(EstablishSessions(&client_options, &server_options));

  std::vector<uint8_t> payload(32 * 1024, 0xA5);
  iree_async_span_t payload_span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  std::atomic<int> send_completion_count{0};
  const iree_net_send_completion_callback_t send_callback = {
      .fn =
          +[](void* user_data, iree_status_t status,
              iree_host_size_t bytes_transferred) {
            (void)bytes_transferred;
            auto* completion_count = static_cast<std::atomic<int>*>(user_data);
            completion_count->fetch_add(1, std::memory_order_relaxed);
            iree_status_free(status);
          },
      .user_data = &send_completion_count,
  };
  iree_status_code_t send_status_code = IREE_STATUS_UNKNOWN;

  // Session publication and transport peer activation are independent. A
  // completed accepted send proves the client control endpoint is no longer
  // consuming its one pre-activation send credit.
  IREE_ASSERT_OK(AwaitClientControlSendReadiness());

  factory_allocator_.ArmAllocationAtLeast(payload.size());
  std::thread send_thread([&] {
    iree_status_t status = iree_net_session_send_control_data_copy(
        client_session_, 0, iree_async_span_list_make(&payload_span, 1),
        send_callback);
    send_status_code = iree_status_code(status);
    iree_status_free(status);
    factory_allocator_.CompleteAttempt();
  });
  if (!factory_allocator_.WaitUntilEnteredOrCompleted()) {
    send_thread.join();
    if (send_status_code == IREE_STATUS_OK) {
      PollBothUntil([&] {
        return send_completion_count.load(std::memory_order_relaxed) == 1;
      });
    }
    FAIL() << "copied send did not reach transport storage allocation; status="
           << send_status_code;
  }

  iree_net_session_deactivate(client_session_);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DRAINING);
  factory_allocator_.Release();
  send_thread.join();

  PollBothUntil([&] { return client_state_.deactivated_count == 1; });
  if (send_status_code == IREE_STATUS_OK) {
    EXPECT_EQ(send_completion_count.load(std::memory_order_relaxed), 1);
  } else {
    EXPECT_EQ(send_completion_count.load(std::memory_order_relaxed), 0);
  }
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DEACTIVATED);
}

TEST_F(SessionTest, DeactivatesBeforeConnectCompletes) {
  ASSERT_NO_FATAL_FAILURE(CreateListener());
  iree_net_session_options_t options = iree_net_session_options_default();
  IREE_ASSERT_OK(iree_net_session_connect(
      factory_,
      iree_make_string_view(connect_address_.data(), connect_address_.size()),
      client_proactor_, client_receive_pool_.pool, &options,
      client_state_.callbacks(), iree_allocator_system(), &client_session_));
  iree_net_session_deactivate(client_session_);
  iree_net_session_deactivate(client_session_);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DRAINING);
  while (client_state_.deactivated_count == 0) {
    Poll(client_proactor_, kSessionClientPolling);
  }
  EXPECT_EQ(accept_state_.callback_count, 0);
  EXPECT_EQ(client_state_.ready_count, 0);
  EXPECT_EQ(client_state_.error_count, 0);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DEACTIVATED);
}

TEST_F(SessionTest, DeactivatedCallbackReleasesPendingConnectSession) {
  ASSERT_NO_FATAL_FAILURE(CreateListener());
  auto options = iree_net_session_options_default();
  auto callbacks = client_state_.callbacks();
  callbacks.on_deactivated = [](void* user_data, iree_net_session_t* session) {
    SessionCallbackState::OnDeactivated(user_data, session);
    iree_net_session_release(session);
  };
  iree_net_session_t* session = nullptr;
  IREE_ASSERT_OK(iree_net_session_connect(
      factory_,
      iree_make_string_view(connect_address_.data(), connect_address_.size()),
      client_proactor_, client_receive_pool_.pool, &options, callbacks,
      iree_allocator_system(), &session));
  iree_net_session_deactivate(session);
  while (client_state_.deactivated_count == 0) {
    Poll(client_proactor_, kSessionClientPolling);
  }
  EXPECT_EQ(client_state_.ready_count, 0);
  EXPECT_EQ(client_state_.error_count, 0);
  EXPECT_EQ(accept_state_.callback_count, 0);
}

TEST_F(SessionTest, DeactivatesWhileControlEndpointIsOpening) {
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  iree_net_session_options_t server_options =
      iree_net_session_options_default();
  ASSERT_NO_FATAL_FAILURE(StartSessions(&client_options, &server_options));

  iree_net_session_deactivate(server_session_);
  iree_net_session_deactivate(server_session_);
  PollBothUntil([&] { return server_state_.deactivated_count == 1; });
  EXPECT_EQ(server_state_.ready_count, 0);
  EXPECT_EQ(server_state_.error_count, 0);
  EXPECT_EQ(server_state_.deactivated_count, 1);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_DEACTIVATED);
}

TEST_F(SessionTest, EndpointCountMismatchRejectsAndAutoDeactivates) {
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.local_peer.application_endpoint_count = 1;
  iree_net_session_options_t server_options =
      iree_net_session_options_default();
  server_options.local_peer.application_endpoint_count = 2;
  ASSERT_NO_FATAL_FAILURE(StartSessions(&client_options, &server_options));

  PollBothUntil([&] {
    return client_state_.deactivated_count == 1 &&
           server_state_.deactivated_count == 1;
  });
  EXPECT_EQ(client_state_.ready_count, 0);
  EXPECT_EQ(server_state_.ready_count, 0);
  EXPECT_EQ(client_state_.error_count, 1);
  EXPECT_EQ(server_state_.error_count, 1);
  EXPECT_EQ(client_state_.error_code, IREE_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(server_state_.error_code, IREE_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DEACTIVATED);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_DEACTIVATED);
}

TEST_F(SessionTest, DeactivationFromReadyWaitsForReadyReturn) {
  client_state_.deactivate_on_ready = true;
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  iree_net_session_options_t server_options =
      iree_net_session_options_default();
  ASSERT_NO_FATAL_FAILURE(EstablishSessions(&client_options, &server_options));
  PollBothUntil([&] { return client_state_.deactivated_count == 1; });
  EXPECT_FALSE(client_state_.deactivated_during_ready);
  EXPECT_EQ(client_state_.error_count, 0);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DEACTIVATED);
}

TEST_F(SessionTest, DeactivationWaitsForPendingEndpointCallback) {
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.local_peer.application_endpoint_count = 1;
  iree_net_session_options_t server_options =
      iree_net_session_options_default();
  server_options.local_peer.application_endpoint_count = 1;
  ASSERT_NO_FATAL_FAILURE(EstablishSessions(&client_options, &server_options));

  SessionEndpointReadyState endpoint_state;
  endpoint_state.current_poll_side = &current_poll_side_;
  endpoint_state.expected_poll_side = kSessionClientPolling;
  endpoint_state.session_state = &client_state_;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(client_session_,
                                                endpoint_state.callback()));
  iree_net_session_deactivate(client_session_);
  PollBothUntil([&] {
    return endpoint_state.callback_count == 1 &&
           client_state_.deactivated_count == 1;
  });
  EXPECT_TRUE(endpoint_state.callback_preceded_deactivation);
  EXPECT_EQ(client_state_.deactivated_count, 1);
}

}  // namespace
}  // namespace iree::net::cts
