// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS session tests: bootstrap, topology exchange, control data, shutdown,
// and queue channel round-trips over application endpoints.
//
// These tests exercise the session lifecycle across all transport backends
// that provide factory support. The session manages connection bootstrap
// (HELLO/HELLO_ACK), proxy semaphore creation, and control channel forwarding
// — all of which must work identically regardless of the underlying transport.
//
// The endpoint provisioning tests open application endpoints via the session
// and run queue COMMAND frames end-to-end over those endpoints, validating the
// full path: application → queue frame → message_endpoint → carrier →
// message_endpoint → application.
//
// Registered with the "factory" tag — only instantiated for backends that
// provide factory-level fields in their BackendInfo.

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iree/net/bootstrap.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/session_test_base.h"
#include "iree/net/channel/control/control_channel.h"
#include "iree/net/channel/queue/frame.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/frame_sender.h"

namespace iree::net::carrier::cts {
namespace {

// Creates a buffer pool for queue channel header encoding, with self-contained
// ownership: the region's destroy callback frees the buffer memory. Ownership
// of the returned pool is transferred to the caller (typically a queue
// channel).
static iree::StatusOr<iree_async_buffer_pool_t*> CreateHeaderPool() {
  static constexpr iree_host_size_t kBufferCount = 16;
  static constexpr iree_host_size_t kBufferSize = 256;
  iree_host_size_t total_size = kBufferCount * kBufferSize;

  void* memory = malloc(total_size);
  if (!memory) {
    return iree::Status(iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                         "failed to allocate header memory"));
  }
  memset(memory, 0, total_size);

  iree_async_region_t* region =
      static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
  if (!region) {
    free(memory);
    return iree::Status(iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                         "failed to allocate header region"));
  }
  memset(region, 0, sizeof(*region));
  iree_atomic_ref_count_init(&region->ref_count);
  region->destroy_fn = [](iree_async_region_t* r) {
    free(r->base_ptr);
    free(r);
  };
  region->base_ptr = memory;
  region->length = total_size;
  region->buffer_size = kBufferSize;
  region->buffer_count = kBufferCount;

  iree_async_buffer_pool_t* pool = nullptr;
  iree_status_t status =
      iree_async_buffer_pool_create(region, iree_allocator_system(), &pool);
  iree_async_region_release(region);  // Pool retains it.
  if (!iree_status_is_ok(status)) return iree::Status(std::move(status));
  return pool;
}

// Holds an application callback open while another thread detaches it.
class CallbackGate {
 public:
  static void Wait(void* user_data) {
    auto* self = static_cast<CallbackGate*>(user_data);
    std::unique_lock<std::mutex> lock(self->mutex_);
    self->entered_ = true;
    self->condition_.notify_all();
    self->condition_.wait(lock, [&]() { return self->released_; });
  }

  void NotifyPollFinished() {
    std::lock_guard<std::mutex> lock(mutex_);
    poll_finished_ = true;
    condition_.notify_all();
  }

  bool WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() { return entered_ || poll_finished_; });
    return entered_;
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_ = false;
  bool released_ = false;
  bool poll_finished_ = false;
};

// Minimal real control-channel peer used to inject bootstrap DATA while the
// session under test remains on its production connection path.
struct BootstrapPeer {
  using DataHandler = iree_status_t (*)(BootstrapPeer* peer,
                                        iree_net_control_frame_flags_t flags,
                                        iree_const_byte_span_t payload);

  // Owned connection backing the control channel.
  iree_net_connection_t* connection = nullptr;
  // Owned control channel over the first connection endpoint.
  iree_net_control_channel_t* channel = nullptr;
  // Terminal endpoint and channel setup status.
  iree::Status setup_status;
  // True after the control channel is active.
  bool endpoint_ready = false;
  // True after at least one DATA message arrived.
  bool data_fired = false;
  // Copy of the most recently received DATA payload.
  std::vector<uint8_t> data;
  // Terminal error observed by the peer control channel.
  iree::Status channel_status;
  // True after a peer DATA send completes.
  bool send_completed = false;
  // Completion status of the peer DATA send.
  iree::Status send_status;
  // True after connection deactivation completes.
  bool deactivated = false;
  // Optional handler invoked for received DATA.
  DataHandler data_handler = nullptr;

  static void OnEndpointReady(void* user_data, iree_status_t status,
                              iree_net_message_endpoint_t endpoint) {
    auto* self = static_cast<BootstrapPeer*>(user_data);
    self->setup_status = iree::Status(std::move(status));
    if (!self->setup_status.ok()) return;

    iree_net_control_channel_callbacks_t callbacks = {};
    callbacks.on_data = OnData;
    callbacks.on_error = OnChannelError;
    callbacks.on_transport_error = OnChannelError;
    callbacks.on_send_complete = OnSendComplete;
    callbacks.user_data = self;
    self->setup_status = iree::Status(iree_net_control_channel_create(
        endpoint, IREE_NET_FRAME_SENDER_MAX_SPANS,
        iree_net_control_channel_options_default(), callbacks,
        iree_allocator_system(), &self->channel));
    if (!self->setup_status.ok()) return;
    self->setup_status =
        iree::Status(iree_net_control_channel_activate(self->channel));
    self->endpoint_ready = self->setup_status.ok();
  }

  static iree_status_t OnData(void* user_data,
                              iree_net_control_frame_flags_t flags,
                              iree_const_byte_span_t payload,
                              iree_async_buffer_lease_t*) {
    auto* self = static_cast<BootstrapPeer*>(user_data);
    self->data_fired = true;
    self->data.assign(payload.data, payload.data + payload.data_length);
    if (self->data_handler) return self->data_handler(self, flags, payload);
    return iree_ok_status();
  }

  static void OnSendComplete(void* user_data, uint64_t, iree_status_t status) {
    auto* self = static_cast<BootstrapPeer*>(user_data);
    self->send_completed = true;
    self->send_status = iree::Status(std::move(status));
  }

  static void OnChannelError(void* user_data, iree_status_t status) {
    auto* self = static_cast<BootstrapPeer*>(user_data);
    self->channel_status = iree::Status(std::move(status));
  }

  static void OnAccept(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto* self = static_cast<BootstrapPeer*>(user_data);
    self->setup_status = iree::Status(std::move(status));
    if (!self->setup_status.ok()) {
      iree_net_connection_release(connection);
      return;
    }
    self->Open(connection);
  }

  void Open(iree_net_connection_t* value) {
    connection = value;
    setup_status = iree::Status(
        iree_net_connection_open_endpoint(connection, {OnEndpointReady, this}));
  }

  bool HasAsyncFailure() const {
    return !setup_status.ok() || !channel_status.ok() ||
           (send_completed && !send_status.ok());
  }

  void BeginDeactivation() {
    iree_net_connection_deactivate(
        connection, {[](void* user_data) {
                       static_cast<BootstrapPeer*>(user_data)->deactivated =
                           true;
                     },
                     this});
  }

  void Release() {
    iree_net_control_channel_release(channel);
    channel = nullptr;
    iree_net_connection_release(connection);
    connection = nullptr;
  }
};

static iree_status_t ValidateClientHello(iree_net_control_frame_flags_t flags,
                                         iree_const_byte_span_t payload) {
  if (flags != IREE_NET_CONTROL_DATA_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "client HELLO used unexpected DATA flags");
  }
  iree_net_bootstrap_message_view_t hello;
  IREE_RETURN_IF_ERROR(iree_net_bootstrap_message_parse(payload, &hello));
  if (hello.type != IREE_NET_BOOTSTRAP_TYPE_HELLO) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected HELLO from client, got type %u",
                            (unsigned)hello.type);
  }
  return iree_ok_status();
}

static iree_status_t SendUnofferedHelloAck(BootstrapPeer* peer,
                                           iree_net_control_frame_flags_t flags,
                                           iree_const_byte_span_t payload) {
  IREE_RETURN_IF_ERROR(ValidateClientHello(flags, payload));

  iree_net_bootstrap_hello_ack_t hello_ack = {};
  hello_ack.header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK;
  hello_ack.session_id = 42;
  hello_ack.negotiated_capabilities =
      IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER;
  iree_async_span_t span =
      iree_async_span_from_ptr(&hello_ack, sizeof(hello_ack));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  return iree_net_control_channel_send_data_copy(
      peer->channel, IREE_NET_CONTROL_DATA_FLAG_NONE, span_list,
      /*operation_user_data=*/0);
}

static iree_status_t SendLongReject(BootstrapPeer* peer,
                                    iree_net_control_frame_flags_t flags,
                                    iree_const_byte_span_t payload) {
  IREE_RETURN_IF_ERROR(ValidateClientHello(flags, payload));

  iree_net_bootstrap_reject_t reject = {};
  reject.header.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  reject.reason_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  std::string reason(2048, 'r');
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(&reject, sizeof(reject)),
      iree_async_span_from_ptr(reason.data(), reason.size()),
  };
  return iree_net_control_channel_send_data_copy(
      peer->channel, IREE_NET_CONTROL_DATA_FLAG_NONE,
      iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
      /*operation_user_data=*/0);
}

using SessionTest = SessionTestBase;

//===----------------------------------------------------------------------===//
// Bootstrap
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, BootstrapSucceeds) {
  EstablishDefaultSessionPair();

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
}

TEST_P(SessionTest, BootstrapWithZeroAxes) {
  iree_net_session_topology_t empty_topo = {};
  EstablishSessionPair(empty_topo, empty_topo);

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);

  EXPECT_EQ(client_callbacks_.remote_axis_count, 0u);
  EXPECT_EQ(server_callbacks_.remote_axis_count, 0u);
}

TEST_P(SessionTest, TopologyExchange) {
  // Client: 2 axes with non-zero epochs.
  iree_async_axis_t client_axes[] = {0x0100, 0x0101};
  uint64_t client_epochs[] = {10, 20};
  iree_net_session_topology_t client_topo = {};
  client_topo.axes = client_axes;
  client_topo.current_epochs = client_epochs;
  client_topo.axis_count = 2;
  client_topo.machine_index = 5;
  client_topo.session_epoch = 3;

  // Server: 3 axes with non-zero epochs.
  iree_async_axis_t server_axes[] = {0x0200, 0x0201, 0x0202};
  uint64_t server_epochs[] = {100, 200, 300};
  iree_net_session_topology_t server_topo = {};
  server_topo.axes = server_axes;
  server_topo.current_epochs = server_epochs;
  server_topo.axis_count = 3;
  server_topo.machine_index = 7;
  server_topo.session_epoch = 4;

  EstablishSessionPair(client_topo, server_topo);

  // Client should see server's topology (3 axes).
  ASSERT_EQ(client_callbacks_.remote_axis_count, 3u);
  EXPECT_EQ(client_callbacks_.remote_axes[0], 0x0200u);
  EXPECT_EQ(client_callbacks_.remote_axes[1], 0x0201u);
  EXPECT_EQ(client_callbacks_.remote_axes[2], 0x0202u);
  EXPECT_EQ(client_callbacks_.remote_epochs[0], 100u);
  EXPECT_EQ(client_callbacks_.remote_epochs[1], 200u);
  EXPECT_EQ(client_callbacks_.remote_epochs[2], 300u);
  EXPECT_EQ(client_callbacks_.remote_machine_index, 7);
  EXPECT_EQ(client_callbacks_.remote_session_epoch, 4);

  // Server should see client's topology (2 axes).
  ASSERT_EQ(server_callbacks_.remote_axis_count, 2u);
  EXPECT_EQ(server_callbacks_.remote_axes[0], 0x0100u);
  EXPECT_EQ(server_callbacks_.remote_axes[1], 0x0101u);
  EXPECT_EQ(server_callbacks_.remote_epochs[0], 10u);
  EXPECT_EQ(server_callbacks_.remote_epochs[1], 20u);
  EXPECT_EQ(server_callbacks_.remote_machine_index, 5);
  EXPECT_EQ(server_callbacks_.remote_session_epoch, 3);
}

TEST_P(SessionTest, SessionIdAssignment) {
  EstablishSessionPair(iree_net_session_topology_t{},
                       iree_net_session_topology_t{},
                       /*server_session_id=*/99);

  // Both sides should agree on the server-assigned session ID.
  EXPECT_EQ(iree_net_session_id(client_session_), 99u);
  EXPECT_EQ(iree_net_session_id(server_session_), 99u);
}

TEST_P(SessionTest, RequiredCapabilitiesRejectMissingPeerSupport) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    iree::Status status;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;

  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        ctx->status = iree::Status(std::move(status));

        if (ctx->status.ok()) {
          iree_net_session_options_t server_options =
              iree_net_session_options_default();
          server_options.session_id = 42;
          server_options.capabilities =
              IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER |
              IREE_NET_BOOTSTRAP_CAPABILITY_RDMA;
          server_options.required_capabilities =
              IREE_NET_BOOTSTRAP_CAPABILITY_RDMA;

          ctx->status = iree::Status(iree_net_session_accept(
              connection, ctx->proactor, ctx->tracker, &server_options,
              ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
              ctx->out_session));
        }

        iree_net_connection_release(connection);
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));

  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER;

  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  ASSERT_TRUE(PollUntil([&]() {
    return !accept_ctx.status.ok() || server_callbacks_.error_fired;
  })) << "Server on_error never fired for missing required capability";
  IREE_ASSERT_OK(accept_ctx.status);

  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_ERROR);
  EXPECT_EQ(server_callbacks_.error_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_FALSE(server_callbacks_.ready_fired);

  ASSERT_TRUE(PollUntil([&]() {
    return iree_net_session_state(client_session_) !=
           IREE_NET_SESSION_STATE_BOOTSTRAPPING;
  }));
}

TEST_P(SessionTest, PeerTopologyExceedingTrackerCapacityIsRejected) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    iree::Status status;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;

  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        ctx->status = iree::Status(std::move(status));
        if (ctx->status.ok()) {
          iree_net_session_options_t options =
              iree_net_session_options_default();
          options.session_id = 42;
          ctx->status = iree::Status(iree_net_session_accept(
              connection, ctx->proactor, ctx->tracker, &options,
              ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
              ctx->out_session));
        }
        iree_net_connection_release(connection);
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));
  std::vector<iree_async_axis_t> axes(kAxisTableCapacity + 1);
  std::vector<uint64_t> epochs(kAxisTableCapacity + 1, 0);
  for (uint32_t i = 0; i < axes.size(); ++i) axes[i] = 0x1000 + i;

  iree_net_session_options_t options = iree_net_session_options_default();
  options.local_topology.axes = axes.data();
  options.local_topology.current_epochs = epochs.data();
  options.local_topology.axis_count = axes.size();
  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  ASSERT_TRUE(PollUntil([&]() {
    return !accept_ctx.status.ok() || server_callbacks_.error_fired;
  }));
  IREE_ASSERT_OK(accept_ctx.status);
  EXPECT_EQ(server_callbacks_.error_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_FALSE(server_callbacks_.ready_fired);

  ASSERT_TRUE(PollUntil([&]() {
    return iree_net_session_state(client_session_) !=
           IREE_NET_SESSION_STATE_BOOTSTRAPPING;
  }));
  EXPECT_EQ(client_callbacks_.error_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_FALSE(client_callbacks_.ready_fired);
}

TEST_P(SessionTest, DuplicatePeerAxisDoesNotRetireExistingTrackerAxis) {
  iree_async_axis_t existing_axis = 0x1234;
  iree_async_semaphore_t* existing_semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, /*initial_value=*/0,
      IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, iree_allocator_system(),
      &existing_semaphore));
  IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
      server_tracker_, existing_axis, existing_semaphore));

  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());
  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    iree::Status status;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;

  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        ctx->status = iree::Status(std::move(status));
        if (ctx->status.ok()) {
          iree_net_session_options_t options =
              iree_net_session_options_default();
          options.session_id = 42;
          ctx->status = iree::Status(iree_net_session_accept(
              connection, ctx->proactor, ctx->tracker, &options,
              ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
              ctx->out_session));
        }
        iree_net_connection_release(connection);
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));
  uint64_t existing_epoch = 0;
  iree_net_session_options_t options = iree_net_session_options_default();
  options.local_topology.axes = &existing_axis;
  options.local_topology.current_epochs = &existing_epoch;
  options.local_topology.axis_count = 1;
  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  ASSERT_TRUE(PollUntil([&]() {
    return !accept_ctx.status.ok() || server_callbacks_.error_fired;
  }));
  IREE_ASSERT_OK(accept_ctx.status);
  EXPECT_EQ(server_callbacks_.error_code, IREE_STATUS_ALREADY_EXISTS);
  EXPECT_FALSE(server_callbacks_.ready_fired);
  ASSERT_TRUE(PollUntil([&]() {
    return iree_net_session_state(client_session_) !=
           IREE_NET_SESSION_STATE_BOOTSTRAPPING;
  }));
  EXPECT_EQ(client_callbacks_.error_code, IREE_STATUS_ALREADY_EXISTS);

  iree_async_frontier_tracker_advance(server_tracker_, existing_axis, 1);
  EXPECT_EQ(iree_async_semaphore_query(existing_semaphore), 1u);

  iree_async_frontier_tracker_retire_axis(
      server_tracker_, existing_axis,
      iree_make_status(IREE_STATUS_CANCELLED, "test axis retired"));
  iree_async_semaphore_release(existing_semaphore);
}

TEST_P(SessionTest, BootstrapDataFlagsAreRejected) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());
  listener_ = pair.listener;
  pair.listener = nullptr;

  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 42;
  IREE_ASSERT_OK(
      iree_net_session_accept(pair.server, proactor_, server_tracker_, &options,
                              server_callbacks_.MakeCallbacks(),
                              iree_allocator_system(), &server_session_));
  iree_net_connection_release(pair.server);
  pair.server = nullptr;

  BootstrapPeer peer;
  peer.Open(pair.client);
  pair.client = nullptr;
  ASSERT_TRUE(PollUntil(
      [&]() { return peer.endpoint_ready || !peer.setup_status.ok(); }));
  IREE_ASSERT_OK(peer.setup_status);

  iree_net_bootstrap_hello_t hello = {};
  hello.header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO;
  hello.protocol_version = IREE_NET_BOOTSTRAP_PROTOCOL_VERSION;
  iree_async_span_t span = iree_async_span_from_ptr(&hello, sizeof(hello));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(iree_net_control_channel_send_data_copy(
      peer.channel, /*flags=*/1, span_list, /*operation_user_data=*/0));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.error_fired; }));
  EXPECT_EQ(server_callbacks_.error_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_FALSE(server_callbacks_.ready_fired);

  ASSERT_TRUE(PollUntil(
      [&]() { return peer.data_fired || !peer.channel_status.ok(); }));
  IREE_ASSERT_OK(peer.channel_status);
  ASSERT_TRUE(peer.data_fired);
  iree_net_bootstrap_message_view_t reject;
  IREE_ASSERT_OK(iree_net_bootstrap_message_parse(
      iree_make_const_byte_span(peer.data.data(), peer.data.size()), &reject));
  ASSERT_EQ(reject.type, IREE_NET_BOOTSTRAP_TYPE_REJECT);
  EXPECT_EQ(reject.value.reject.fixed.reason_code,
            IREE_STATUS_INVALID_ARGUMENT);

  DeactivateSession(server_session_, server_deactivation_);
  peer.BeginDeactivation();
  ASSERT_TRUE(PollUntil(
      [&]() { return server_deactivation_.completed && peer.deactivated; }));
  peer.Release();
}

TEST_P(SessionTest, HelloAckCannotAddUnofferedCapabilities) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  BootstrapPeer peer;
  peer.data_handler = SendUnofferedHelloAck;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_, BootstrapPeer::OnAccept,
      &peer, iree_allocator_system(), &listener_));
  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));

  iree_net_session_options_t options = iree_net_session_options_default();
  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  ASSERT_TRUE(PollUntil([&]() {
    return peer.HasAsyncFailure() || client_callbacks_.error_fired;
  }));
  IREE_ASSERT_OK(peer.setup_status);
  IREE_ASSERT_OK(peer.channel_status);
  if (peer.send_completed) IREE_ASSERT_OK(peer.send_status);
  ASSERT_TRUE(peer.endpoint_ready);
  ASSERT_TRUE(peer.data_fired);
  EXPECT_EQ(client_callbacks_.error_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_FALSE(client_callbacks_.ready_fired);
  EXPECT_EQ(iree_net_session_id(client_session_), 0u);

  DeactivateSession(client_session_, client_deactivation_);
  peer.BeginDeactivation();
  ASSERT_TRUE(PollUntil(
      [&]() { return client_deactivation_.completed && peer.deactivated; }));
  peer.Release();
}

TEST_P(SessionTest, RejectDiagnosticIsBoundedWithoutLimitingWireReason) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  BootstrapPeer peer;
  peer.data_handler = SendLongReject;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_, BootstrapPeer::OnAccept,
      &peer, iree_allocator_system(), &listener_));
  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));

  iree_net_session_options_t options = iree_net_session_options_default();
  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  ASSERT_TRUE(PollUntil([&]() {
    return peer.HasAsyncFailure() || client_callbacks_.error_fired;
  }));
  IREE_ASSERT_OK(peer.setup_status);
  IREE_ASSERT_OK(peer.channel_status);
  if (peer.send_completed) IREE_ASSERT_OK(peer.send_status);
  ASSERT_TRUE(peer.endpoint_ready);
  ASSERT_TRUE(peer.data_fired);
  EXPECT_EQ(client_callbacks_.error_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_FALSE(client_callbacks_.ready_fired);
  EXPECT_NE(client_callbacks_.error_message.find(std::string(1024, 'r')),
            std::string::npos);
  EXPECT_EQ(client_callbacks_.error_message.find(std::string(1025, 'r')),
            std::string::npos);

  DeactivateSession(client_session_, client_deactivation_);
  peer.BeginDeactivation();
  ASSERT_TRUE(PollUntil(
      [&]() { return client_deactivation_.completed && peer.deactivated; }));
  peer.Release();
}

//===----------------------------------------------------------------------===//
// Control data forwarding
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, ControlDataClientToServer) {
  EstablishDefaultSessionPair();
  EXPECT_EQ(client_callbacks_.send_complete_count, 0u);
  EXPECT_EQ(server_callbacks_.send_complete_count, 0u);

  const char* message = "hello server";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message, strlen(message));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(iree_net_session_send_control_data(
      client_session_, 0, span_list, UINT64_C(0xD00DFEED12345678)));

  ASSERT_TRUE(PollUntil([&]() {
    return server_callbacks_.control_data_fired &&
           client_callbacks_.send_complete_count == 1;
  })) << "Server never received control data";

  EXPECT_EQ(std::string(server_callbacks_.control_data.begin(),
                        server_callbacks_.control_data.end()),
            "hello server");
  EXPECT_EQ(client_callbacks_.send_operation_user_data,
            UINT64_C(0xD00DFEED12345678));
  EXPECT_EQ(client_callbacks_.send_status_code, IREE_STATUS_OK);
}

TEST_P(SessionTest, CopiedControlDataCompletesWithOpaqueUserData) {
  EstablishDefaultSessionPair();

  const char* message = "copied control data";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message, strlen(message));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(iree_net_session_send_control_data_copy(
      client_session_, 0, span_list, UINT64_C(0xF0E1D2C3B4A59687)));

  ASSERT_TRUE(PollUntil([&]() {
    return server_callbacks_.control_data_fired &&
           client_callbacks_.send_complete_count == 1;
  }));
  EXPECT_EQ(client_callbacks_.send_operation_user_data,
            UINT64_C(0xF0E1D2C3B4A59687));
  EXPECT_EQ(client_callbacks_.send_status_code, IREE_STATUS_OK);
  EXPECT_EQ(std::string(server_callbacks_.control_data.begin(),
                        server_callbacks_.control_data.end()),
            message);
}

TEST_P(SessionTest, ControlDataServerToClient) {
  EstablishDefaultSessionPair();

  const char* message = "hello client";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message, strlen(message));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(
      iree_net_session_send_control_data(server_session_, 0, span_list, 0));

  ASSERT_TRUE(PollUntil([&]() { return client_callbacks_.control_data_fired; }))
      << "Client never received control data";

  EXPECT_EQ(std::string(client_callbacks_.control_data.begin(),
                        client_callbacks_.control_data.end()),
            "hello client");
}

TEST_P(SessionTest, ControlDataBidirectional) {
  EstablishDefaultSessionPair();

  // Send in both directions simultaneously.
  const char* to_server = "ping";
  const char* to_client = "pong";
  iree_async_span_t span_to_server =
      iree_async_span_from_ptr((void*)to_server, strlen(to_server));
  iree_async_span_list_t list_to_server =
      iree_async_span_list_make(&span_to_server, 1);
  IREE_ASSERT_OK(iree_net_session_send_control_data(client_session_, 0,
                                                    list_to_server, 0));

  iree_async_span_t span_to_client =
      iree_async_span_from_ptr((void*)to_client, strlen(to_client));
  iree_async_span_list_t list_to_client =
      iree_async_span_list_make(&span_to_client, 1);
  IREE_ASSERT_OK(iree_net_session_send_control_data(server_session_, 0,
                                                    list_to_client, 0));

  ASSERT_TRUE(PollUntil([&]() {
    return server_callbacks_.control_data_fired &&
           client_callbacks_.control_data_fired;
  })) << "Bidirectional control data did not arrive";

  EXPECT_EQ(std::string(server_callbacks_.control_data.begin(),
                        server_callbacks_.control_data.end()),
            "ping");
  EXPECT_EQ(std::string(client_callbacks_.control_data.begin(),
                        client_callbacks_.control_data.end()),
            "pong");
}

TEST_P(SessionTest, DetachCallbacksWaitsForActiveCallback) {
  CallbackGate callback_gate;
  client_callbacks_.control_data_hook = CallbackGate::Wait;
  client_callbacks_.control_data_hook_user_data = &callback_gate;
  EstablishDefaultSessionPair();

  const char* message = "hold callback";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message, strlen(message));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(
      iree_net_session_send_control_data(server_session_, 0, span_list, 0));

  std::atomic<bool> detach_complete = false;
  bool callback_entered = false;
  bool detach_completed_early = false;
  std::thread detach_thread([&]() {
    callback_entered = callback_gate.WaitUntilEntered();
    iree_net_session_detach_callbacks(
        client_session_, {[](void* user_data) {
                            static_cast<std::atomic<bool>*>(user_data)->store(
                                true, std::memory_order_release);
                          },
                          &detach_complete});
    detach_completed_early =
        callback_entered && detach_complete.load(std::memory_order_acquire);
    callback_gate.Release();
  });

  iree_status_t poll_status = iree_ok_status();
  while (iree_status_is_ok(poll_status) &&
         !detach_complete.load(std::memory_order_acquire)) {
    iree_host_size_t completed = 0;
    poll_status = PollProactorOnce(proactor_, &completed);
  }
  if (!iree_status_is_ok(poll_status)) callback_gate.NotifyPollFinished();
  detach_thread.join();

  IREE_ASSERT_OK(poll_status);
  ASSERT_TRUE(callback_entered);
  EXPECT_FALSE(detach_completed_early);
  EXPECT_TRUE(detach_complete.load(std::memory_order_acquire));
}

TEST_P(SessionTest, DetachCallbacksWaitsForPendingEndpointCallback) {
  EstablishDefaultSessionPair();

  struct CallbackState {
    int next_order = 0;
    int endpoint_order = -1;
    int detach_order = -1;
    iree_status_code_t endpoint_status = IREE_STATUS_UNKNOWN;
  } callback_state;

  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {[](void* user_data, iree_status_t status,
                           iree_net_message_endpoint_t endpoint) {
                          auto* state = static_cast<CallbackState*>(user_data);
                          state->endpoint_order = state->next_order++;
                          state->endpoint_status = iree_status_code(status);
                          iree_status_free(status);
                          (void)endpoint;
                        },
                        &callback_state}));
  iree_net_session_detach_callbacks(
      client_session_, {[](void* user_data) {
                          auto* state = static_cast<CallbackState*>(user_data);
                          state->detach_order = state->next_order++;
                        },
                        &callback_state});

  EXPECT_EQ(callback_state.detach_order, -1);
  ASSERT_TRUE(PollUntil([&]() { return callback_state.detach_order >= 0; }));
  EXPECT_EQ(callback_state.endpoint_status, IREE_STATUS_OK);
  EXPECT_EQ(callback_state.endpoint_order, 0);
  EXPECT_EQ(callback_state.detach_order, 1);
}

TEST_P(SessionTest, DeactivateDrainsConnection) {
  EstablishDefaultSessionPair();

  bool deactivated = false;
  DeactivateSession(
      client_session_, client_deactivation_,
      {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
       &deactivated});

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DRAINING);
  ASSERT_TRUE(PollUntil([&]() { return deactivated; }));

  const char* message = "after deactivation";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message, strlen(message));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_send_control_data(client_session_, 0, span_list, 0));
}

TEST_P(SessionTest, DeactivateWaitsForPendingEndpointCallback) {
  EstablishDefaultSessionPair();

  struct CallbackState {
    int next_order = 0;
    int endpoint_order = -1;
    int deactivate_order = -1;
    iree_status_code_t endpoint_status = IREE_STATUS_UNKNOWN;
  } callback_state;

  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {[](void* user_data, iree_status_t status,
                           iree_net_message_endpoint_t endpoint) {
                          auto* state = static_cast<CallbackState*>(user_data);
                          state->endpoint_order = state->next_order++;
                          state->endpoint_status = iree_status_code(status);
                          iree_status_free(status);
                          (void)endpoint;
                        },
                        &callback_state}));
  DeactivateSession(client_session_, client_deactivation_,
                    {[](void* user_data) {
                       auto* state = static_cast<CallbackState*>(user_data);
                       state->deactivate_order = state->next_order++;
                     },
                     &callback_state});

  EXPECT_EQ(callback_state.deactivate_order, -1);
  ASSERT_TRUE(
      PollUntil([&]() { return callback_state.deactivate_order >= 0; }));
  EXPECT_EQ(callback_state.endpoint_status, IREE_STATUS_OK);
  EXPECT_EQ(callback_state.endpoint_order, 0);
  EXPECT_EQ(callback_state.deactivate_order, 1);
}

//===----------------------------------------------------------------------===//
// Graceful shutdown
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, GracefulShutdownFromClient) {
  EstablishDefaultSessionPair();

  IREE_ASSERT_OK(iree_net_session_shutdown(client_session_, 0,
                                           iree_make_cstring_view("bye")));

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DRAINING);

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.goaway_fired; }))
      << "Server never received GOAWAY";

  EXPECT_EQ(server_callbacks_.goaway_reason_code, 0u);
  EXPECT_EQ(server_callbacks_.goaway_message, "bye");
}

TEST_P(SessionTest, GracefulShutdownFromServer) {
  EstablishDefaultSessionPair();

  IREE_ASSERT_OK(iree_net_session_shutdown(server_session_, 42,
                                           iree_make_cstring_view("done")));

  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_DRAINING);

  ASSERT_TRUE(PollUntil([&]() { return client_callbacks_.goaway_fired; }))
      << "Client never received GOAWAY";

  EXPECT_EQ(client_callbacks_.goaway_reason_code, 42u);
  EXPECT_EQ(client_callbacks_.goaway_message, "done");
}

TEST_P(SessionTest, OperationsFailAfterShutdown) {
  EstablishDefaultSessionPair();

  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  // send_control_data should fail in DRAINING state.
  const char* data = "nope";
  iree_async_span_t span = iree_async_span_from_ptr((void*)data, strlen(data));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_send_control_data(client_session_, 0, span_list, 0));

  // A second shutdown should also fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_shutdown(client_session_, 0, IREE_SV("again")));
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, ServerSessionRequiresNonzeroId) {
  // session_accept must reject session_id=0.
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());
  ASSERT_NE(pair.server, nullptr);

  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 0;

  SessionCallbackTracker callbacks;
  iree_net_session_t* session = nullptr;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_session_accept(pair.server, proactor_, server_tracker_, &options,
                              callbacks.MakeCallbacks(),
                              iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(SessionTest, OnReadyCallbackRequired) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());
  ASSERT_NE(pair.server, nullptr);

  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 1;

  // Missing on_ready (on_control_data provided).
  iree_net_session_callbacks_t bad_callbacks;
  memset(&bad_callbacks, 0, sizeof(bad_callbacks));
  bad_callbacks.on_control_data = SessionCallbackTracker::OnControlData;

  iree_net_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_session_accept(
                            pair.server, proactor_, server_tracker_, &options,
                            bad_callbacks, iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(SessionTest, OnControlDataCallbackRequired) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());
  ASSERT_NE(pair.server, nullptr);

  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 1;

  // Missing on_control_data (on_ready provided).
  iree_net_session_callbacks_t bad_callbacks;
  memset(&bad_callbacks, 0, sizeof(bad_callbacks));
  bad_callbacks.on_ready = SessionCallbackTracker::OnReady;

  iree_net_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_session_accept(
                            pair.server, proactor_, server_tracker_, &options,
                            bad_callbacks, iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(SessionTest, InvalidLocalBootstrapConfigurationIsRejected) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());
  ASSERT_NE(pair.server, nullptr);

  SessionCallbackTracker callbacks;
  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 1;
  options.local_topology.axis_count = 1;

  iree_net_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_session_accept(pair.server, proactor_, server_tracker_, &options,
                              callbacks.MakeCallbacks(),
                              iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_async_axis_t axis = 1;
  options.local_topology.axes = &axis;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_session_accept(pair.server, proactor_, server_tracker_, &options,
                              callbacks.MakeCallbacks(),
                              iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  uint64_t epoch = 0;
  options.local_topology.current_epochs = &epoch;
  options.local_topology.reserved[0] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_session_accept(pair.server, proactor_, server_tracker_, &options,
                              callbacks.MakeCallbacks(),
                              iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  options.local_topology.reserved[0] = 0;
  options.capabilities =
      IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED | (1u << 31);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_session_accept(pair.server, proactor_, server_tracker_, &options,
                              callbacks.MakeCallbacks(),
                              iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

//===----------------------------------------------------------------------===//
// Remote frontier tracking
//===----------------------------------------------------------------------===//

// After bootstrap, the client's frontier tracker should observe the server's
// axes at the exchanged epoch values.
TEST_P(SessionTest, RemoteAxesRegisteredAfterBootstrap) {
  iree_async_axis_t server_axes[] = {0x0200, 0x0201};
  uint64_t server_epochs[] = {10, 20};
  iree_net_session_topology_t server_topo = {};
  server_topo.axes = server_axes;
  server_topo.current_epochs = server_epochs;
  server_topo.axis_count = 2;
  server_topo.machine_index = 1;
  server_topo.session_epoch = 1;

  iree_net_session_topology_t client_topo = {};
  client_topo.machine_index = 0;
  client_topo.session_epoch = 1;

  EstablishSessionPair(client_topo, server_topo);

  // The client's tracker should have the server's axes registered and advanced
  // to the exchanged epoch.
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(
        client_tracker_, server_axes[i], server_epochs[i]))
        << "Server axis 0x" << std::hex << server_axes[i]
        << " should be initialized to epoch " << std::dec << server_epochs[i];
    EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(
        client_tracker_, server_axes[i], server_epochs[i] + 1))
        << "Server axis 0x" << std::hex << server_axes[i]
        << " should not be past epoch " << std::dec << server_epochs[i];
  }
}

// Advancing a remote axis via frontier_tracker_advance() should make the new
// epoch visible to frontier queries.
TEST_P(SessionTest, RemoteAxisVisibleAfterAdvance) {
  EstablishDefaultSessionPair();

  // The client's tracker has server axis 0x0200 at epoch 0. Advance it.
  iree_async_axis_t server_axis = 0x0200;
  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(client_tracker_, server_axis, 42);
  EXPECT_EQ(dispatched, 0u);

  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                      server_axis, 42));
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                       server_axis, 43));
}

// Advancing a remote axis should satisfy frontier tracker waiters that
// reference that axis.
TEST_P(SessionTest, FrontierWaiterSatisfiedByRemoteAxisAdvance) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;

  // Build a single-entry frontier waiting for server_axis to reach epoch 5.
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {server_axis, 5};

  // Register a waiter.
  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      client_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_free(status);
      },
      &result, &waiter));

  // Waiter should not fire yet (epoch is 0, target is 5).
  EXPECT_FALSE(result.fired);

  // Advance to epoch 3 — still below target.
  iree_async_frontier_tracker_advance(client_tracker_, server_axis, 3);
  EXPECT_FALSE(result.fired);

  // Advance to epoch 5 — should satisfy the waiter.
  iree_async_frontier_tracker_advance(client_tracker_, server_axis, 5);
  EXPECT_TRUE(result.fired) << "Waiter should have fired when axis reached "
                               "target epoch";
  EXPECT_EQ(result.status_code, IREE_STATUS_OK);
}

// Monotonic advancement: advancing to a lower epoch should be a no-op.
TEST_P(SessionTest, RemoteAxisMonotonicAdvance) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;

  // Advance to 100.
  iree_async_frontier_tracker_advance(client_tracker_, server_axis, 100);
  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                      server_axis, 100));
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                       server_axis, 101));

  // Advance to 50 — should be a no-op.
  iree_async_frontier_tracker_advance(client_tracker_, server_axis, 50);
  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                      server_axis, 100));
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                       server_axis, 101));

  // Advance to 100 again — also a no-op (not strictly greater).
  iree_async_frontier_tracker_advance(client_tracker_, server_axis, 100);
  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                      server_axis, 100));
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                       server_axis, 101));

  // Advance to 101 — should succeed.
  iree_async_frontier_tracker_advance(client_tracker_, server_axis, 101);
  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(client_tracker_,
                                                      server_axis, 101));
}

// Failing a remote axis should propagate the error to frontier tracker waiters.
TEST_P(SessionTest, AxisFailurePropagatesToWaiters) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;

  // Register a frontier waiter on the remote axis.
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {server_axis, 999};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } waiter_result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      client_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_free(status);
      },
      &waiter_result, &waiter));

  // The waiter should not have fired yet.
  EXPECT_FALSE(waiter_result.fired);

  // Fail the axis — simulates remote disconnect.
  iree_async_frontier_tracker_fail_axis(
      client_tracker_, server_axis,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "connection lost"));

  // The frontier waiter should have fired with an error status.
  EXPECT_TRUE(waiter_result.fired) << "Frontier waiter should fire on axis "
                                      "failure";
  EXPECT_EQ(waiter_result.status_code, IREE_STATUS_UNAVAILABLE);
}

// After axis failure, new waits on the failed axis should fail immediately.
TEST_P(SessionTest, NewWaitsFailAfterAxisFailure) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;

  // Fail the axis.
  iree_async_frontier_tracker_fail_axis(
      client_tracker_, server_axis,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "gone"));

  // A new frontier waiter on the failed axis should fire immediately with
  // the failure status.
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {server_axis, 1};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      client_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_free(status);
      },
      &result, &waiter));

  // Should have fired immediately since the axis is already failed.
  EXPECT_TRUE(result.fired)
      << "Wait on failed axis should dispatch immediately";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, RetainRelease) {
  EstablishDefaultSessionPair();

  // Extra retain/release cycle should not crash or affect state.
  iree_net_session_retain(client_session_);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  iree_net_session_release(client_session_);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
}

TEST_P(SessionTest, RetainReleaseNullSafe) {
  // Both should be no-ops on NULL.
  iree_net_session_retain(nullptr);
  iree_net_session_release(nullptr);
}

//===----------------------------------------------------------------------===//
// Endpoint provisioning
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, OpenEndpointRequiresOperational) {
  EstablishDefaultSessionPair();

  // Shut down first, then verify endpoint opening fails.
  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  EndpointReadyResult endpoint_result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_open_endpoint(
          client_session_, {EndpointReadyResult::Callback, &endpoint_result}));
}

TEST_P(SessionTest, OpenEndpointSucceeds) {
  EstablishDefaultSessionPair();

  // Open application endpoints on both sides (slot 1; slot 0 is the control
  // channel consumed during bootstrap).
  EndpointReadyResult client_endpoint;
  EndpointReadyResult server_endpoint;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_endpoint}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_endpoint}));

  ASSERT_TRUE(PollUntil([&]() {
    return client_endpoint.fired && server_endpoint.fired;
  })) << "Endpoint open did not complete";

  EXPECT_EQ(client_endpoint.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_endpoint.status_code, IREE_STATUS_OK);
  EXPECT_NE(client_endpoint.endpoint.self, nullptr);
  EXPECT_NE(server_endpoint.endpoint.self, nullptr);
}

TEST_P(SessionTest, MultipleEndpointsSucceed) {
  EstablishDefaultSessionPair();

  // Open two application endpoints on each side (slots 1 and 2).
  EndpointReadyResult client_ep1, client_ep2;
  EndpointReadyResult server_ep1, server_ep2;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_ep1}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_ep2}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_ep1}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_ep2}));

  ASSERT_TRUE(PollUntil([&]() {
    return client_ep1.fired && client_ep2.fired && server_ep1.fired &&
           server_ep2.fired;
  })) << "Multiple endpoint opens did not complete";

  EXPECT_EQ(client_ep1.status_code, IREE_STATUS_OK);
  EXPECT_EQ(client_ep2.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_ep1.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_ep2.status_code, IREE_STATUS_OK);

  // Each endpoint should be distinct.
  EXPECT_NE(client_ep1.endpoint.self, client_ep2.endpoint.self);
  EXPECT_NE(server_ep1.endpoint.self, server_ep2.endpoint.self);
}

//===----------------------------------------------------------------------===//
// Queue channel round-trip over application endpoints
//===----------------------------------------------------------------------===//

// Tracks a received queue COMMAND for test assertions.
struct ReceivedQueueCommand {
  bool fired = false;
  uint32_t stream_id = 0;
  std::vector<uint8_t> data;

  static iree_status_t Callback(void* user_data, uint32_t stream_id,
                                const iree_async_frontier_t* wait_frontier,
                                const iree_async_frontier_t* signal_frontier,
                                iree_const_byte_span_t command_data,
                                iree_async_buffer_lease_t* lease) {
    auto* result = static_cast<ReceivedQueueCommand*>(user_data);
    result->fired = true;
    result->stream_id = stream_id;
    result->data.insert(result->data.end(), command_data.data,
                        command_data.data + command_data.data_length);
    return iree_ok_status();
  }
};

TEST_P(SessionTest, QueueChannelCommandRoundTrip) {
  EstablishDefaultSessionPair();

  // Open application endpoints on both sides.
  EndpointReadyResult client_ep_result;
  EndpointReadyResult server_ep_result;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_ep_result}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_ep_result}));

  ASSERT_TRUE(PollUntil([&]() {
    return client_ep_result.fired && server_ep_result.fired;
  })) << "Endpoint open did not complete";
  ASSERT_EQ(client_ep_result.status_code, IREE_STATUS_OK);
  ASSERT_EQ(server_ep_result.status_code, IREE_STATUS_OK);

  // Create queue channels on the application endpoints.
  ReceivedQueueCommand server_command;
  ReceivedQueueCommand client_command;

  iree_net_queue_channel_callbacks_t server_qcb = {};
  server_qcb.on_command = ReceivedQueueCommand::Callback;
  server_qcb.user_data = &server_command;

  iree_net_queue_channel_callbacks_t client_qcb = {};
  client_qcb.on_command = ReceivedQueueCommand::Callback;
  client_qcb.user_data = &client_command;

  iree_net_queue_channel_t* client_channel = nullptr;
  iree_net_queue_channel_t* server_channel = nullptr;
  IREE_ASSERT_OK_AND_ASSIGN(iree_async_buffer_pool_t * client_header_pool,
                            CreateHeaderPool());
  IREE_ASSERT_OK(iree_net_queue_channel_create(
      client_ep_result.endpoint, /*max_send_spans=*/8, client_header_pool,
      client_qcb, iree_allocator_system(), &client_channel));
  client_header_pool = nullptr;
  IREE_ASSERT_OK_AND_ASSIGN(iree_async_buffer_pool_t * server_header_pool,
                            CreateHeaderPool());
  IREE_ASSERT_OK(iree_net_queue_channel_create(
      server_ep_result.endpoint, /*max_send_spans=*/8, server_header_pool,
      server_qcb, iree_allocator_system(), &server_channel));
  server_header_pool = nullptr;

  // Activate both channels (installs recv handlers and activates endpoints).
  IREE_ASSERT_OK(iree_net_queue_channel_activate(client_channel));
  IREE_ASSERT_OK(iree_net_queue_channel_activate(server_channel));

  // Send a COMMAND from client → server.
  const char* payload = "hello queue";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)payload, strlen(payload));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      client_channel, /*stream_id=*/7, /*wait_frontier=*/NULL,
      /*signal_frontier=*/NULL, span_list, /*operation_user_data=*/0));

  ASSERT_TRUE(PollUntil([&]() { return server_command.fired; }))
      << "Server never received queue command";
  EXPECT_EQ(server_command.stream_id, 7u);
  EXPECT_EQ(std::string(server_command.data.begin(), server_command.data.end()),
            "hello queue");

  // Send a COMMAND from server → client.
  const char* reply = "queue reply";
  iree_async_span_t reply_span =
      iree_async_span_from_ptr((void*)reply, strlen(reply));
  iree_async_span_list_t reply_list = iree_async_span_list_make(&reply_span, 1);
  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      server_channel, /*stream_id=*/42, /*wait_frontier=*/NULL,
      /*signal_frontier=*/NULL, reply_list, /*operation_user_data=*/0));

  ASSERT_TRUE(PollUntil([&]() { return client_command.fired; }))
      << "Client never received queue reply";
  EXPECT_EQ(client_command.stream_id, 42u);
  EXPECT_EQ(std::string(client_command.data.begin(), client_command.data.end()),
            "queue reply");

  // Drain pending send completions before releasing channels. SHM carriers
  // fire send completions asynchronously when the peer's SPSC ring consumer
  // advances; the completion may lag one poll cycle behind data receipt.
  ASSERT_TRUE(PollUntil([&]() {
    return !iree_net_queue_channel_has_pending_sends(client_channel) &&
           !iree_net_queue_channel_has_pending_sends(server_channel);
  })) << "Send completions did not drain";

  iree_net_queue_channel_release(server_channel);
  iree_net_queue_channel_release(client_channel);
}

//===----------------------------------------------------------------------===//
// Error state transitions
//===----------------------------------------------------------------------===//

// Protocol version mismatch during bootstrap causes the server session to
// transition to ERROR state and fire on_error with the validation failure.
//
// This exercises the bootstrap error routing fix: handle_hello() returns an
// error, on_data catches it, and routes it through fail() so the session
// transitions directly to ERROR with the specific diagnostic. Without the fix,
// the error would bubble through the carrier's transport error path, losing the
// original message.
TEST_P(SessionTest, ProtocolVersionMismatchCausesServerError) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  // AcceptCtx holds direct pointers to avoid C++ protected member access
  // restrictions (lambdas in TEST_P can't access protected base members
  // through a base class pointer).
  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    iree::Status status;
    bool fired = false;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;

  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        ctx->status = iree::Status(std::move(status));

        if (ctx->status.ok()) {
          iree_net_session_options_t server_options =
              iree_net_session_options_default();
          server_options.session_id = 1;

          ctx->status = iree::Status(iree_net_session_accept(
              connection, ctx->proactor, ctx->tracker, &server_options,
              ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
              ctx->out_session));
        }

        iree_net_connection_release(connection);
        ctx->fired = true;
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));

  // Client uses a wrong protocol version — the server rejects the HELLO.
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.protocol_version = 999;

  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  // Wait for the server to receive the bad HELLO and fire on_error.
  ASSERT_TRUE(PollUntil([&]() {
    return !accept_ctx.status.ok() || server_callbacks_.error_fired;
  })) << "Server on_error never fired after protocol version mismatch";
  IREE_ASSERT_OK(accept_ctx.status);

  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_ERROR);
  // The server should see INVALID_ARGUMENT from the protocol version check.
  EXPECT_EQ(server_callbacks_.error_code, IREE_STATUS_INVALID_ARGUMENT);

  // The server should NOT have reached OPERATIONAL.
  EXPECT_FALSE(server_callbacks_.ready_fired);
}

// After a session enters ERROR state, all operations return
// FAILED_PRECONDITION.
TEST_P(SessionTest, OperationsFailInErrorState) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    iree::Status status;
    bool fired = false;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;

  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        ctx->status = iree::Status(std::move(status));

        if (ctx->status.ok()) {
          iree_net_session_options_t server_options =
              iree_net_session_options_default();
          server_options.session_id = 1;

          ctx->status = iree::Status(iree_net_session_accept(
              connection, ctx->proactor, ctx->tracker, &server_options,
              ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
              ctx->out_session));
        }

        iree_net_connection_release(connection);
        ctx->fired = true;
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));

  // Client with wrong protocol version forces both peers into ERROR state.
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.protocol_version = 999;

  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  ASSERT_TRUE(PollUntil([&]() {
    return !accept_ctx.status.ok() || server_callbacks_.error_fired;
  })) << "Server on_error never fired";
  IREE_ASSERT_OK(accept_ctx.status);

  ASSERT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_ERROR);

  // send_control_data should fail.
  const char* data = "nope";
  iree_async_span_t span = iree_async_span_from_ptr((void*)data, strlen(data));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_send_control_data(server_session_, 0, span_list, 0));

  // shutdown should fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_shutdown(server_session_, 0, IREE_SV("late")));

  // open_endpoint should fail.
  EndpointReadyResult endpoint_result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_open_endpoint(
          server_session_, {EndpointReadyResult::Callback, &endpoint_result}));
}

//===----------------------------------------------------------------------===//
// Proxy semaphore cleanup on shutdown/goaway
//===----------------------------------------------------------------------===//

// When a session shuts down (sends GOAWAY), it synchronously fails all remote
// axes in the frontier tracker. Pending frontier waiters on those axes should
// fire with UNAVAILABLE.
TEST_P(SessionTest, ShutdownCleanupFailsRemoteAxisWaiters) {
  EstablishDefaultSessionPair();

  // The client has server axis 0x0200 in its tracker. Register a waiter on
  // that axis waiting for epoch 999 (will never arrive normally).
  iree_async_axis_t server_axis = 0x0200;
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {server_axis, 999};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      client_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_free(status);
      },
      &result, &waiter));

  EXPECT_FALSE(result.fired);

  // Client shuts down — this synchronously fails remote axes in the client's
  // tracker, which should wake the waiter.
  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  EXPECT_TRUE(result.fired)
      << "Frontier waiter on remote axis should fire when session shuts down";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

// When a session receives GOAWAY from the peer, it fails all remote axes in
// the frontier tracker. Pending frontier waiters on those axes should fire
// with UNAVAILABLE.
TEST_P(SessionTest, GoawayReceivedCleanupFailsRemoteAxisWaiters) {
  EstablishDefaultSessionPair();

  // The server has client axis 0x0100 in its tracker. Register a waiter on
  // that axis waiting for epoch 999 (will never arrive normally).
  iree_async_axis_t client_axis = 0x0100;
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {client_axis, 999};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      server_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_free(status);
      },
      &result, &waiter));

  EXPECT_FALSE(result.fired);

  // Client sends GOAWAY → server receives it → server's cleanup_remote_axes
  // fires → client axis failed in server's tracker.
  IREE_ASSERT_OK(iree_net_session_shutdown(client_session_, 0, IREE_SV("bye")));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.goaway_fired; }))
      << "Server never received GOAWAY";

  EXPECT_TRUE(result.fired)
      << "Frontier waiter on remote axis should fire when GOAWAY is received";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

// Future waits on failed remote axes should fail immediately after GOAWAY.
TEST_P(SessionTest, GoawayReceivedCleanupFailsFutureFrontierWaits) {
  EstablishDefaultSessionPair();

  // Client sends GOAWAY → server cleanup fails the client axis.
  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.goaway_fired; }))
      << "Server never received GOAWAY";

  iree_async_axis_t client_axis = 0x0100;
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {client_axis, 999};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      server_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_free(status);
      },
      &result, &waiter));

  EXPECT_TRUE(result.fired)
      << "Future frontier waiter on remote axis should fail after GOAWAY";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

//===----------------------------------------------------------------------===//
// Bootstrap timeout
//===----------------------------------------------------------------------===//

// When the server accepts a connection but never creates a session, the client
// session should eventually enter ERROR state (either from a transport error
// or from the bootstrap timeout).
//
// Transport behavior varies:
// - Loopback: the server's carrier is never activated, so the client's HELLO
//   send fails immediately with UNAVAILABLE (the timer is cancelled by this
//   transport error, not by expiry).
// - TCP/SHM: the client's HELLO reaches the kernel buffer / shared memory
//   ring and succeeds. The server never responds, so the client waits until
//   the bootstrap timer fires with DEADLINE_EXCEEDED.
//
// Both paths exercise the bootstrap timer lifecycle: start at session creation,
// cancel on error (loopback) or fire on expiry (TCP/SHM).
TEST_P(SessionTest, ClientErrorsWhenServerNeverResponds) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  // Server accepts connections but does NOT create a session. The connection
  // is held alive (retained by the accept callback) so the transport layer
  // doesn't report a disconnect.
  struct HeldConnectionCtx {
    iree::Status status;
    iree_net_connection_t* connection = nullptr;
  } held_connection_ctx;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<HeldConnectionCtx*>(user_data);
        ctx->status = iree::Status(std::move(status));
        if (ctx->status.ok()) {
          // Hold the connection; don't create a session.
          ctx->connection = connection;
        }
      },
      &held_connection_ctx, iree_allocator_system(), &listener_));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener_));

  // This test exercises the bootstrap timeout itself. A one-second deadline
  // leaves instrumented CI enough scheduling headroom after connection setup.
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.bootstrap_timeout_ns = iree_make_duration_ms(1000);

  IREE_ASSERT_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  // Wait for the client to enter ERROR state.
  ASSERT_TRUE(PollUntil([&]() {
    return !held_connection_ctx.status.ok() || client_callbacks_.error_fired;
  })) << "Client on_error never fired (expected UNAVAILABLE or "
         "DEADLINE_EXCEEDED)";
  IREE_ASSERT_OK(held_connection_ctx.status);

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_ERROR);
  EXPECT_FALSE(client_callbacks_.ready_fired)
      << "on_ready should not fire when server never responds";

  // The error code depends on transport: UNAVAILABLE for loopback (immediate
  // transport error), DEADLINE_EXCEEDED for TCP/SHM (bootstrap timeout).
  EXPECT_TRUE(client_callbacks_.error_code == IREE_STATUS_UNAVAILABLE ||
              client_callbacks_.error_code == IREE_STATUS_DEADLINE_EXCEEDED)
      << "Expected UNAVAILABLE or DEADLINE_EXCEEDED, got "
      << client_callbacks_.error_code;

  // Deactivate the held server connection and wait for every carrier operation
  // before releasing it.
  bool held_connection_deactivated = false;
  iree_net_connection_deactivate(
      held_connection_ctx.connection,
      {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
       &held_connection_deactivated});
  ASSERT_TRUE(PollUntil([&]() { return held_connection_deactivated; }));
  iree_net_connection_release(held_connection_ctx.connection);
  held_connection_ctx.connection = nullptr;
}

// Successful bootstrap cancels its infinite timers. Session deactivation is
// the exact join point for those cancellation callbacks; without cancellation
// no deactivation callback can fire.
TEST_P(SessionTest, BootstrapTimerRetiresBeforeDeactivation) {
  EstablishDefaultSessionPair(IREE_DURATION_INFINITE);

  DeactivateSession(server_session_, server_deactivation_);
  DeactivateSession(client_session_, client_deactivation_);
  PollUntilComplete([&]() {
    return server_deactivation_.completed && client_deactivation_.completed;
  });

  EXPECT_FALSE(client_callbacks_.error_fired);
  EXPECT_FALSE(server_callbacks_.error_fired);
}

}  // namespace

CTS_REGISTER_TEST_SUITE_WITH_TAGS(SessionTest, {"factory"}, {});

// SessionTest requires the "factory" tag — backends without factory support
// legitimately have zero instantiations.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SessionTest);

}  // namespace iree::net::carrier::cts
