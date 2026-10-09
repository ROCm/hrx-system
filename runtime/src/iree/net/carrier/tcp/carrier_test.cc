// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif

#include "iree/async/notification.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_WASM)
#include "iree/async/platform/posix/api.h"
#endif
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/net/carrier/tcp/carrier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

struct ReceivePool {
  // Slab providing receive storage.
  iree_async_slab_t* slab = nullptr;

  // Proactor registration for |slab|.
  iree_async_region_t* region = nullptr;

  // Pool presented to the carrier.
  iree_async_buffer_pool_t* pool = nullptr;
};

struct CarrierContext {
  // Bytes delivered through receive callbacks.
  std::vector<uint8_t> received_data;

  // Lease moved out of the next receive callback when requested.
  iree_async_buffer_lease_t retained_lease = {};

  // Number of receive callbacks delivered.
  int receive_count = 0;

  // Number of callbacks whose storage was borrowed rather than movable.
  int borrowed_receive_count = 0;

  // Number of orderly peer send shutdowns delivered.
  int eof_count = 0;

  // Number of terminal error callbacks delivered.
  int error_count = 0;

  // Code from the first terminal error callback.
  iree_status_code_t error_code = IREE_STATUS_OK;

  // Number of upcoming receive callbacks that should move their lease.
  int retain_lease_count = 0;

  // Error returned from nonempty receive callbacks.
  iree_status_code_t receive_error = IREE_STATUS_OK;
};

struct AsyncOperationResult {
  // Completion status code.
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;

  // True after the operation callback fires.
  bool completed = false;
};

struct CarrierDeactivateResult {
  // Fixture completion flag set by the callback.
  bool* deactivated = nullptr;

  // Number of delivered deactivation callbacks.
  int callback_count = 0;
};

struct SendResult {
  // Fixture flag proving callback affinity to proactor polling.
  bool* is_polling = nullptr;

  // Completion status code.
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;

  // Number of bytes reported by the carrier.
  iree_host_size_t bytes_transferred = 0;

  // Number of times the send callback fired.
  int completion_count = 0;
};

struct ReentrantFailureContext {
  // Carrier used to create submitted and queued send ownership phases.
  iree_net_carrier_t* carrier = nullptr;

  // Completion state for the send submitted to the proactor.
  SendResult submitted_send;

  // Completion state for the send left in the carrier queue.
  SendResult queued_send;

  // Stable storage referenced by |submitted_send|.
  std::array<uint8_t, 8> submitted_payload = {};

  // Stable storage referenced by |queued_send|.
  std::array<uint8_t, 8> queued_payload = {};

  // Fixture state marked by the deactivation callback.
  bool* deactivated = nullptr;

  // Number of nonempty receive callbacks delivered.
  int receive_count = 0;

  // Number of terminal error callbacks delivered.
  int error_count = 0;

  // Code from the first terminal error callback.
  iree_status_code_t error_code = IREE_STATUS_OK;
};

struct PrefixWriter {
  // Byte copied into every generated-prefix position.
  uint8_t value = 0;

  // Number of writer invocations.
  int call_count = 0;

  // Alignment observed for transport-owned storage.
  uintptr_t target_alignment = 0;

  static iree_status_t Write(void* user_data, iree_byte_span_t target) {
    auto* self = static_cast<PrefixWriter*>(user_data);
    ++self->call_count;
    self->target_alignment = reinterpret_cast<uintptr_t>(target.data) %
                             IREE_NET_SEND_PREFIX_ALIGNMENT;
    memset(target.data, self->value, target.data_length);
    return iree_ok_status();
  }
};

struct PrefixWriterGate {
  // Serializes writer entry and release.
  std::mutex mutex;

  // Notifies the test when writer state changes.
  std::condition_variable condition;

  // True after the writer receives transport-owned storage.
  bool entered = false;

  // True when the writer may return.
  bool release = false;

  // Status returned after the writer is released.
  iree_status_code_t result_code = IREE_STATUS_OK;

  static iree_status_t Write(void* user_data, iree_byte_span_t target) {
    auto* self = static_cast<PrefixWriterGate*>(user_data);
    std::unique_lock<std::mutex> lock(self->mutex);
    self->entered = true;
    self->condition.notify_all();
    self->condition.wait(lock, [&] { return self->release; });
    memset(target.data, 0xA5, target.data_length);
    return self->result_code == IREE_STATUS_OK
               ? iree_ok_status()
               : iree_make_status(self->result_code,
                                  "generated-prefix failure for testing");
  }
};

struct CountingAllocator {
  // Number of allocation requests issued through this allocator.
  uint32_t allocation_count = 0;

  // Number of free requests issued through this allocator.
  uint32_t free_count = 0;

  // True when the next allocation request should fail.
  bool fail_next_allocation = false;

  iree_allocator_t allocator() { return {this, Control}; }

  void FailNextAllocation() { fail_next_allocation = true; }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<CountingAllocator*>(self);
    switch (command) {
      case IREE_ALLOCATOR_COMMAND_MALLOC:
      case IREE_ALLOCATOR_COMMAND_CALLOC:
        ++allocator->allocation_count;
        if (allocator->fail_next_allocation) {
          allocator->fail_next_allocation = false;
          *inout_ptr = nullptr;
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "injected allocation failure");
        }
        break;
      case IREE_ALLOCATOR_COMMAND_FREE:
        ++allocator->free_count;
        break;
      default:
        break;
    }
    iree_allocator_t system_allocator = iree_allocator_system();
    return system_allocator.ctl(system_allocator.self, command, params,
                                inout_ptr);
  }
};

static void AsyncOperationCompleted(void* user_data,
                                    iree_async_operation_t* operation,
                                    iree_status_t status,
                                    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  auto* result = static_cast<AsyncOperationResult*>(user_data);
  result->status_code = iree_status_code(status);
  result->completed = true;
  iree_status_free(status);
}

static iree_status_t Receive(void* user_data, iree_async_span_t data,
                             iree_async_buffer_lease_t* lease) {
  auto* context = static_cast<CarrierContext*>(user_data);
  if (data.length == 0) {
    ++context->eof_count;
    return iree_ok_status();
  }
  const uint8_t* data_ptr = iree_async_span_ptr(data);
  context->received_data.insert(context->received_data.end(), data_ptr,
                                data_ptr + data.length);
  ++context->receive_count;
  if (!lease) {
    ++context->borrowed_receive_count;
  } else if (context->retain_lease_count > 0) {
    --context->retain_lease_count;
    context->retained_lease = *lease;
    *lease = {};
  }
  return context->receive_error == IREE_STATUS_OK
             ? iree_ok_status()
             : iree_make_status(context->receive_error,
                                "receive rejected for testing");
}

static void Error(void* user_data, iree_status_t status) {
  auto* context = static_cast<CarrierContext*>(user_data);
  if (context->error_count++ == 0) {
    context->error_code = iree_status_code(status);
  }
  iree_status_free(status);
}

static void SendCompleted(void* user_data, iree_status_t status,
                          iree_host_size_t bytes_transferred) {
  auto* result = static_cast<SendResult*>(user_data);
  if (result->is_polling) {
    EXPECT_TRUE(*result->is_polling);
  }
  result->status_code = iree_status_code(status);
  result->bytes_transferred = bytes_transferred;
  ++result->completion_count;
  iree_status_free(status);
}

static void CarrierDeactivated(void* user_data) {
  *static_cast<bool*>(user_data) = true;
}

static void CarrierDeactivationCounted(void* user_data) {
  auto* result = static_cast<CarrierDeactivateResult*>(user_data);
  ++result->callback_count;
  *result->deactivated = true;
}

static iree_status_t QueueSendsAndFail(void* user_data, iree_async_span_t data,
                                       iree_async_buffer_lease_t* lease) {
  (void)lease;
  auto* context = static_cast<ReentrantFailureContext*>(user_data);
  if (data.length == 0) {
    return iree_ok_status();
  }
  ++context->receive_count;

  iree_async_span_t submitted_span = iree_async_span_from_ptr(
      context->submitted_payload.data(), context->submitted_payload.size());
  iree_net_send_params_t submitted_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&submitted_span, 1),
      {SendCompleted, &context->submitted_send},
  };
  IREE_RETURN_IF_ERROR(
      iree_net_carrier_send(context->carrier, &submitted_params));

  iree_async_span_t queued_span = iree_async_span_from_ptr(
      context->queued_payload.data(), context->queued_payload.size());
  iree_net_send_params_t queued_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&queued_span, 1),
      {SendCompleted, &context->queued_send},
  };
  IREE_RETURN_IF_ERROR(iree_net_carrier_send(context->carrier, &queued_params));

  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "receive rejected after queuing sends");
}

static void DeactivateOnError(void* user_data, iree_status_t status) {
  auto* context = static_cast<ReentrantFailureContext*>(user_data);
  if (context->error_count++ == 0) {
    context->error_code = iree_status_code(status);
  }
  iree_status_free(status);
  iree_net_carrier_deactivate(context->carrier, CarrierDeactivated,
                              context->deactivated);
}

struct TcpSendMode {
  // Stable test case suffix.
  const char* name;
  // Socket preference independent of the carrier's send policy.
  iree_async_socket_options_t socket_options;
  // Carrier-specific crossover applied to each native submission.
  iree_host_size_t zero_copy_min_send_size;
};

class TcpCarrierTest : public ::testing::TestWithParam<TcpSendMode> {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));
  }

  void TearDown() override {
    DeactivatePair();
    ReleaseReceivePool(&server_receive_pool_);
    ReleaseReceivePool(&client_receive_pool_);
    iree_async_proactor_release(proactor_);
  }

  void CreateReceivePool(iree_host_size_t buffer_size,
                         iree_host_size_t buffer_count,
                         ReceivePool* out_receive_pool) {
    iree_async_slab_options_t options = {0};
    options.buffer_size = buffer_size;
    options.buffer_count = buffer_count;
    IREE_ASSERT_OK(iree_async_slab_create(options, iree_allocator_system(),
                                          &out_receive_pool->slab));
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, out_receive_pool->slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
        &out_receive_pool->region));
    IREE_ASSERT_OK(iree_async_buffer_pool_create(out_receive_pool->region,
                                                 iree_allocator_system(),
                                                 &out_receive_pool->pool));
  }

  static void ReleaseReceivePool(ReceivePool* receive_pool) {
    iree_async_buffer_pool_release(receive_pool->pool);
    iree_async_region_release(receive_pool->region);
    iree_async_slab_release(receive_pool->slab);
    *receive_pool = {};
  }

  template <typename Predicate>
  void PollUntil(Predicate predicate) {
    while (!predicate()) {
      iree_host_size_t completion_count = 0;
      is_polling_ = true;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), &completion_count);
      is_polling_ = false;
      IREE_ASSERT_OK(status);
    }
  }

  // Forces large test payloads through short platform send completions.
  static void ConstrainSendBuffer(iree_async_socket_t* socket) {
    int send_buffer_size = 1;
#if defined(IREE_PLATFORM_WINDOWS)
    ASSERT_EQ(
        setsockopt(static_cast<SOCKET>(socket->primitive.value.win32_handle),
                   SOL_SOCKET, SO_SNDBUF,
                   reinterpret_cast<const char*>(&send_buffer_size),
                   sizeof(send_buffer_size)),
        0);
#else
    ASSERT_EQ(setsockopt(socket->primitive.value.fd, SOL_SOCKET, SO_SNDBUF,
                         &send_buffer_size, sizeof(send_buffer_size)),
              0);
#endif  // IREE_PLATFORM_WINDOWS
  }

  void EstablishSockets(iree_async_socket_t** out_client_socket,
                        iree_async_socket_t** out_server_socket) {
    *out_client_socket = nullptr;
    *out_server_socket = nullptr;

    iree_async_socket_t* listener = nullptr;
    IREE_ASSERT_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR | GetParam().socket_options,
        &listener));
    iree_async_address_t bind_address;
    IREE_ASSERT_OK(iree_async_address_from_ipv4(
        iree_make_cstring_view("127.0.0.1"), 0, &bind_address));
    IREE_ASSERT_OK(iree_async_socket_bind(listener, &bind_address));
    IREE_ASSERT_OK(iree_async_socket_listen(listener, 1));
    iree_async_address_t listen_address;
    IREE_ASSERT_OK(
        iree_async_socket_query_local_address(listener, &listen_address));

    AsyncOperationResult accept_result;
    iree_async_socket_accept_operation_t accept_operation;
    memset(&accept_operation, 0, sizeof(accept_operation));
    iree_async_operation_initialize(&accept_operation.base,
                                    IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    AsyncOperationCompleted, &accept_result);
    accept_operation.listen_socket = listener;
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &accept_operation.base));

    IREE_ASSERT_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_NO_DELAY | GetParam().socket_options,
        out_client_socket));
    AsyncOperationResult connect_result;
    iree_async_socket_connect_operation_t connect_operation;
    memset(&connect_operation, 0, sizeof(connect_operation));
    iree_async_operation_initialize(&connect_operation.base,
                                    IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    AsyncOperationCompleted, &connect_result);
    connect_operation.socket = *out_client_socket;
    connect_operation.address = listen_address;
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &connect_operation.base));

    PollUntil(
        [&] { return accept_result.completed && connect_result.completed; });
    ASSERT_EQ(accept_result.status_code, IREE_STATUS_OK);
    ASSERT_EQ(connect_result.status_code, IREE_STATUS_OK);
    ASSERT_NE(accept_operation.accepted_socket, nullptr);
    *out_server_socket = accept_operation.accepted_socket;
    ConstrainSendBuffer(*out_client_socket);
    ConstrainSendBuffer(*out_server_socket);
    iree_async_socket_release(listener);
  }

  void CreateCarrierPair(
      uint32_t max_send_operations = 8,
      iree_host_size_t receive_buffer_size = 4096,
      iree_host_size_t receive_buffer_count = 4,
      iree_allocator_t server_host_allocator = iree_allocator_system()) {
    CreateCarrierPairWithHandlers({Receive, Error, &client_context_},
                                  {Receive, Error, &server_context_},
                                  max_send_operations, receive_buffer_size,
                                  receive_buffer_count, server_host_allocator);
  }

  void CreateCarrierPairWithHandlers(
      iree_net_carrier_handlers_t client_handlers,
      iree_net_carrier_handlers_t server_handlers, uint32_t max_send_operations,
      iree_host_size_t receive_buffer_size,
      iree_host_size_t receive_buffer_count,
      iree_allocator_t server_host_allocator) {
    CreateReceivePool(receive_buffer_size, receive_buffer_count,
                      &client_receive_pool_);
    CreateReceivePool(receive_buffer_size, receive_buffer_count,
                      &server_receive_pool_);

    iree_async_socket_t* client_socket = nullptr;
    iree_async_socket_t* server_socket = nullptr;
    EstablishSockets(&client_socket, &server_socket);
    iree_net_tcp_carrier_options_t options =
        iree_net_tcp_carrier_options_default();
    options.max_send_operations = max_send_operations;
    options.zero_copy_min_send_size = GetParam().zero_copy_min_send_size;
    IREE_ASSERT_OK(iree_net_tcp_carrier_create(
        proactor_, client_socket, client_receive_pool_.pool, &options,
        iree_allocator_system(), &client_carrier_));
    IREE_ASSERT_OK(iree_net_tcp_carrier_create(
        proactor_, server_socket, server_receive_pool_.pool, &options,
        server_host_allocator, &server_carrier_));
    iree_async_socket_release(client_socket);
    iree_async_socket_release(server_socket);

    IREE_ASSERT_OK(
        iree_net_carrier_set_handlers(client_carrier_, client_handlers));
    IREE_ASSERT_OK(
        iree_net_carrier_set_handlers(server_carrier_, server_handlers));
    IREE_ASSERT_OK(iree_net_carrier_activate(client_carrier_));
    IREE_ASSERT_OK(iree_net_carrier_activate(server_carrier_));
  }

  void BeginDeactivation(iree_net_carrier_t* carrier, bool* completed) {
    if (!carrier || *completed) {
      return;
    }
    const iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
    if (state == IREE_NET_CARRIER_STATE_CREATED ||
        state == IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_net_carrier_deactivate(carrier, CarrierDeactivated, completed);
    }
  }

  void DeactivatePair() {
    BeginDeactivation(client_carrier_, &client_deactivated_);
    BeginDeactivation(server_carrier_, &server_deactivated_);
    if (client_carrier_ || server_carrier_) {
      PollUntil([&] {
        return (!client_carrier_ || client_deactivated_) &&
               (!server_carrier_ || server_deactivated_);
      });
    }
    iree_net_carrier_release(client_carrier_);
    iree_net_carrier_release(server_carrier_);
    client_carrier_ = nullptr;
    server_carrier_ = nullptr;
  }

  // Platform proactor shared by both test peers.
  iree_async_proactor_t* proactor_ = nullptr;

  // True only while the fixture is polling |proactor_|.
  bool is_polling_ = false;

  // Client receive storage.
  ReceivePool client_receive_pool_;

  // Server receive storage.
  ReceivePool server_receive_pool_;

  // Client carrier under test.
  iree_net_carrier_t* client_carrier_ = nullptr;

  // Server carrier under test.
  iree_net_carrier_t* server_carrier_ = nullptr;

  // Client callback state.
  CarrierContext client_context_;

  // Server callback state.
  CarrierContext server_context_;

  // Client deactivation completion state.
  bool client_deactivated_ = false;

  // Server deactivation completion state.
  bool server_deactivated_ = false;
};

TEST(TcpCarrierOptionsTest, Defaults) {
  iree_net_tcp_carrier_options_t options =
      iree_net_tcp_carrier_options_default();
  EXPECT_EQ(options.max_send_operations,
            IREE_NET_TCP_DEFAULT_MAX_SEND_OPERATIONS);
  EXPECT_EQ(options.generated_prefix_capacity,
            IREE_NET_TCP_DEFAULT_GENERATED_PREFIX_CAPACITY);
}

TEST_P(TcpCarrierTest, PreservesOrderedScatterGatherAndGeneratedPrefixSends) {
  CreateCarrierPair(/*max_send_operations=*/4,
                    /*receive_buffer_size=*/4096,
                    /*receive_buffer_count=*/4);

  std::vector<uint8_t> first(256 * 1024, 0x11);
  std::vector<uint8_t> second(256 * 1024, 0x22);
  std::array<iree_async_span_t, 2> first_spans = {
      iree_async_span_from_ptr(first.data(), first.size() / 2),
      iree_async_span_from_ptr(first.data() + first.size() / 2,
                               first.size() / 2),
  };
  iree_async_span_t second_span =
      iree_async_span_from_ptr(second.data(), second.size());
  SendResult first_result;
  SendResult second_result;
  SendResult direct_result;

  iree_net_send_params_t first_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(first_spans.data(), first_spans.size()),
      {SendCompleted, &first_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &first_params));
  iree_net_send_params_t second_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&second_span, 1),
      {SendCompleted, &second_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &second_params));

  constexpr const char kDirectPayload[] = "direct-send";
  iree_net_send_params_t direct_params = {
      iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(kDirectPayload, sizeof(kDirectPayload))),
      iree_async_span_list_empty(),
      {SendCompleted, &direct_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &direct_params));

  const iree_host_size_t expected_size =
      first.size() + second.size() + sizeof(kDirectPayload);
  PollUntil([&] {
    return first_result.completion_count == 1 &&
           second_result.completion_count == 1 &&
           direct_result.completion_count == 1 &&
           server_context_.received_data.size() == expected_size;
  });

  EXPECT_EQ(first_result.status_code, IREE_STATUS_OK);
  EXPECT_EQ(second_result.status_code, IREE_STATUS_OK);
  EXPECT_EQ(direct_result.status_code, IREE_STATUS_OK);
  EXPECT_EQ(first_result.bytes_transferred, first.size());
  EXPECT_EQ(second_result.bytes_transferred, second.size());
  EXPECT_EQ(direct_result.bytes_transferred, sizeof(kDirectPayload));
  EXPECT_TRUE(std::equal(first.begin(), first.end(),
                         server_context_.received_data.begin()));
  EXPECT_TRUE(std::equal(second.begin(), second.end(),
                         server_context_.received_data.begin() + first.size()));
  EXPECT_EQ(memcmp(server_context_.received_data.data() + first.size() +
                       second.size(),
                   kDirectPayload, sizeof(kDirectPayload)),
            0);
}

TEST_P(TcpCarrierTest, MixedSendExtentsRetireBeforeSourceReuse) {
  const iree_host_size_t lengths[] = {
      32,
      IREE_NET_TCP_DEFAULT_ZERO_COPY_MIN_SEND_SIZE - 1,
      IREE_NET_TCP_DEFAULT_ZERO_COPY_MIN_SEND_SIZE,
      IREE_NET_TCP_DEFAULT_ZERO_COPY_MIN_SEND_SIZE + 1,
      256 * 1024,
      32,
  };
  CreateCarrierPair(/*max_send_operations=*/IREE_ARRAYSIZE(lengths));
  if (GetParam().zero_copy_min_send_size == IREE_HOST_SIZE_MAX) {
    EXPECT_EQ(iree_net_carrier_capabilities(client_carrier_) &
                  IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX,
              0u);
  }

  struct Source {
    // Source storage reused by the terminal callback while receives may remain.
    std::vector<uint8_t> bytes;
    // Per-send accounting independent of callback order.
    SendResult result;

    static void Complete(void* user_data, iree_status_t status,
                         iree_host_size_t bytes_transferred) {
      auto* self = static_cast<Source*>(user_data);
      SendCompleted(&self->result, status, bytes_transferred);
      std::fill(self->bytes.begin(), self->bytes.end(), 0xFF);
    }
  } sources[IREE_ARRAYSIZE(lengths)];
  std::vector<uint8_t> expected;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
    auto& source = sources[i];
    source.bytes.resize(lengths[i], static_cast<uint8_t>(i));
    source.result.is_polling = &is_polling_;
    expected.insert(expected.end(), source.bytes.begin(), source.bytes.end());
    const iree_host_size_t prefix_length = 16;
    iree_async_span_t span =
        iree_async_span_from_ptr(source.bytes.data() + prefix_length,
                                 source.bytes.size() - prefix_length);
    iree_net_send_params_t params = {
        iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(source.bytes.data(), prefix_length)),
        iree_async_span_list_make(&span, 1),
        {Source::Complete, &source},
    };
    IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  }
  PollUntil([&] {
    for (const auto& source : sources) {
      if (source.result.completion_count != 1) {
        return false;
      }
    }
    return server_context_.received_data.size() == expected.size();
  });
  EXPECT_EQ(server_context_.received_data, expected);
  for (const auto& source : sources) {
    EXPECT_EQ(source.result.status_code, IREE_STATUS_OK);
    EXPECT_EQ(source.result.bytes_transferred, source.bytes.size());
    EXPECT_TRUE(std::all_of(source.bytes.begin(), source.bytes.end(),
                            [](uint8_t value) { return value == 0xFF; }));
  }
}

TEST_P(TcpCarrierTest, GeneratedPrefixSupportsFastAndScatterOverflowPaths) {
  CreateCarrierPair(/*max_send_operations=*/2,
                    /*receive_buffer_size=*/4096,
                    /*receive_buffer_count=*/8);

  PrefixWriter small_writer = {
      .value = 0x31,
  };
  SendResult small_result;
  iree_net_send_params_t small_params = {
      {
          .length = 32,
          .write = PrefixWriter::Write,
          .user_data = &small_writer,
      },
      iree_async_span_list_empty(),
      {SendCompleted, &small_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &small_params));

  PrefixWriter overflow_writer = {
      .value = 0x32,
  };
  std::array<uint8_t, IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS> suffix_bytes = {};
  std::array<iree_async_span_t, IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS>
      suffix_spans;
  for (iree_host_size_t i = 0; i < suffix_spans.size(); ++i) {
    suffix_bytes[i] = static_cast<uint8_t>(i);
    suffix_spans[i] = iree_async_span_from_ptr(&suffix_bytes[i], 1);
  }
  constexpr iree_host_size_t kOverflowPrefixLength =
      IREE_NET_TCP_DEFAULT_GENERATED_PREFIX_CAPACITY + 1;
  SendResult overflow_result;
  iree_net_send_params_t overflow_params = {
      {
          .length = kOverflowPrefixLength,
          .write = PrefixWriter::Write,
          .user_data = &overflow_writer,
      },
      iree_async_span_list_make(suffix_spans.data(), suffix_spans.size()),
      {SendCompleted, &overflow_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &overflow_params));

  const iree_host_size_t expected_size =
      32 + kOverflowPrefixLength + suffix_bytes.size();
  PollUntil([&] {
    return small_result.completion_count == 1 &&
           overflow_result.completion_count == 1 &&
           server_context_.received_data.size() == expected_size;
  });

  EXPECT_EQ(small_writer.call_count, 1);
  EXPECT_EQ(small_writer.target_alignment, 0u);
  EXPECT_EQ(overflow_writer.call_count, 1);
  EXPECT_EQ(overflow_writer.target_alignment, 0u);
  EXPECT_TRUE(std::all_of(server_context_.received_data.begin(),
                          server_context_.received_data.begin() + 32,
                          [](uint8_t value) { return value == 0x31; }));
  EXPECT_TRUE(std::all_of(
      server_context_.received_data.begin() + 32,
      server_context_.received_data.begin() + 32 + kOverflowPrefixLength,
      [](uint8_t value) { return value == 0x32; }));
  EXPECT_TRUE(std::equal(
      suffix_bytes.begin(), suffix_bytes.end(),
      server_context_.received_data.begin() + 32 + kOverflowPrefixLength));
}

TEST_P(TcpCarrierTest, OverflowAllocationFailurePrecedesSendAdmission) {
  CountingAllocator server_allocator;
  CreateCarrierPair(/*max_send_operations=*/1,
                    /*receive_buffer_size=*/4096,
                    /*receive_buffer_count=*/4, server_allocator.allocator());

  PrefixWriter writer = {
      .value = 0x31,
  };
  SendResult failed_result;
  constexpr iree_host_size_t kOverflowPrefixLength =
      IREE_NET_TCP_DEFAULT_GENERATED_PREFIX_CAPACITY + 1;
  iree_net_send_params_t failed_params = {
      {
          .length = kOverflowPrefixLength,
          .write = PrefixWriter::Write,
          .user_data = &writer,
      },
      iree_async_span_list_empty(),
      {SendCompleted, &failed_result},
  };
  server_allocator.FailNextAllocation();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_carrier_send(server_carrier_, &failed_params));
  EXPECT_EQ(writer.call_count, 0);
  EXPECT_EQ(failed_result.completion_count, 0);
  EXPECT_EQ(iree_net_carrier_query_send_budget(server_carrier_).slots, 1u);

  std::array<uint8_t, 8> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendResult retry_result;
  iree_net_send_params_t retry_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &retry_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(server_carrier_, &retry_params));
  PollUntil([&] {
    return retry_result.completion_count == 1 &&
           client_context_.received_data.size() == payload.size();
  });
  EXPECT_EQ(retry_result.status_code, IREE_STATUS_OK);
  DeactivatePair();
}

TEST_P(TcpCarrierTest, RetainedReceiveLeasePreservesProgress) {
  CountingAllocator server_host_allocator;
  CreateCarrierPair(/*max_send_operations=*/4,
                    /*receive_buffer_size=*/64,
                    /*receive_buffer_count=*/2,
                    server_host_allocator.allocator());
  server_context_.retain_lease_count = 1;

  // One byte makes each receive indivisible even with a short native send.
  std::array<uint8_t, 1> first = {};
  std::array<uint8_t, 1> second = {};
  first.fill(0x31);
  second.fill(0x32);
  iree_async_span_t first_span =
      iree_async_span_from_ptr(first.data(), first.size());
  iree_async_span_t second_span =
      iree_async_span_from_ptr(second.data(), second.size());
  SendResult first_result;
  SendResult second_result;
  iree_net_send_params_t first_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&first_span, 1),
      {SendCompleted, &first_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &first_params));
  PollUntil([&] {
    return first_result.completion_count == 1 &&
           server_context_.receive_count == 1;
  });
  ASSERT_NE(server_context_.retained_lease.release.fn, nullptr);
  EXPECT_EQ(server_context_.borrowed_receive_count, 0);

  iree_net_send_params_t second_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&second_span, 1),
      {SendCompleted, &second_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &second_params));
  PollUntil([&] {
    return second_result.completion_count == 1 &&
           server_context_.receive_count == 2;
  });
  EXPECT_EQ(server_context_.borrowed_receive_count, 1);
  EXPECT_EQ(memcmp(iree_async_span_ptr(server_context_.retained_lease.span),
                   first.data(), first.size()),
            0);
  EXPECT_EQ(server_context_.received_data.size(), first.size() + second.size());

  std::thread consumer([&] {
    iree_async_buffer_lease_release(&server_context_.retained_lease);
  });
  consumer.join();
  second_result = {};
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &second_params));
  PollUntil([&] {
    return second_result.completion_count == 1 &&
           server_context_.receive_count == 3;
  });
  EXPECT_EQ(server_context_.borrowed_receive_count, 1);
  EXPECT_EQ(server_host_allocator.allocation_count, 1u);
  DeactivatePair();
}

TEST_P(TcpCarrierTest, SingleBufferReceivesRemainBorrowedWithoutAllocation) {
  CountingAllocator server_host_allocator;
  CreateCarrierPair(/*max_send_operations=*/4,
                    /*receive_buffer_size=*/64,
                    /*receive_buffer_count=*/1,
                    server_host_allocator.allocator());
  std::array<uint8_t, 16> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  for (int i = 0; i < 32; ++i) {
    payload.fill(static_cast<uint8_t>(i));
    SendResult result;
    iree_net_send_params_t params = {iree_net_send_prefix_empty(),
                                     iree_async_span_list_make(&span, 1),
                                     {SendCompleted, &result}};
    IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
    PollUntil([&] {
      return result.completion_count == 1 &&
             server_context_.received_data.size() == (i + 1) * payload.size();
    });
    EXPECT_EQ(result.status_code, IREE_STATUS_OK);
    EXPECT_EQ(memcmp(server_context_.received_data.data() + i * payload.size(),
                     payload.data(), payload.size()),
              0);
  }
  EXPECT_EQ(server_context_.borrowed_receive_count,
            server_context_.receive_count);
  EXPECT_EQ(server_host_allocator.allocation_count, 1u);
  DeactivatePair();
}

TEST_P(TcpCarrierTest, RetainedReceiveLeaseExtendsCarrierLifetime) {
  CountingAllocator server_host_allocator;
  CreateCarrierPair(/*max_send_operations=*/4,
                    /*receive_buffer_size=*/64,
                    /*receive_buffer_count=*/2,
                    server_host_allocator.allocator());
  EXPECT_EQ(server_host_allocator.allocation_count, 1u);
  server_context_.retain_lease_count = 1;

  std::array<uint8_t, 16> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendResult send_result;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &send_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  PollUntil([&] {
    return send_result.completion_count == 1 &&
           server_context_.received_data.size() == payload.size();
  });
  ASSERT_NE(server_context_.retained_lease.release.fn, nullptr);

  BeginDeactivation(server_carrier_, &server_deactivated_);
  PollUntil([&] { return server_deactivated_; });
  iree_net_carrier_release(server_carrier_);
  server_carrier_ = nullptr;
  EXPECT_EQ(server_host_allocator.free_count, 0u);

  iree_async_buffer_lease_release(&server_context_.retained_lease);
  EXPECT_EQ(server_host_allocator.free_count, 1u);
}

TEST_P(TcpCarrierTest, LeaseReturnedOffThreadBeforeReceiveCallbackReturns) {
  auto receive_and_return = [](void* user_data, iree_async_span_t data,
                               iree_async_buffer_lease_t* lease) {
    IREE_RETURN_IF_ERROR(Receive(user_data, data, lease));
    if (data.length == 0) {
      return iree_ok_status();
    }
    if (!lease) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "prior early return lost native capacity");
    }
    auto owned_lease = *lease;
    *lease = {};
    std::thread consumer([owned_lease]() mutable {
      iree_async_buffer_lease_release(&owned_lease);
    });
    consumer.join();
    return iree_ok_status();
  };
  CreateCarrierPairWithHandlers(
      {Receive, Error, &client_context_},
      {receive_and_return, Error, &server_context_},
      /*max_send_operations=*/4, /*receive_buffer_size=*/64,
      /*receive_buffer_count=*/2, iree_allocator_system());
  std::array<uint8_t, 16> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  for (int i = 0; i < 32; ++i) {
    SendResult result;
    iree_net_send_params_t params = {iree_net_send_prefix_empty(),
                                     iree_async_span_list_make(&span, 1),
                                     {SendCompleted, &result}};
    IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
    PollUntil([&] {
      return result.completion_count == 1 &&
             server_context_.received_data.size() == (i + 1) * payload.size();
    });
    EXPECT_EQ(result.status_code, IREE_STATUS_OK);
    ASSERT_EQ(server_context_.error_count, 0);
  }
  EXPECT_EQ(server_context_.borrowed_receive_count, 0);
}

TEST_P(TcpCarrierTest,
       GeneratedPrefixFailureCompletesAndRestoresOneSlotAdmission) {
  CreateCarrierPair(/*max_send_operations=*/1);

  PrefixWriterGate writer_gate;
  writer_gate.result_code = IREE_STATUS_INVALID_ARGUMENT;
  SendResult failed_result;
  failed_result.is_polling = &is_polling_;
  iree_net_send_params_t failed_params = {
      {
          32,
          PrefixWriterGate::Write,
          &writer_gate,
      },
      iree_async_span_list_empty(),
      {SendCompleted, &failed_result},
  };
  iree_status_code_t send_status = IREE_STATUS_UNKNOWN;
  std::thread send_thread([&] {
    iree_status_t status =
        iree_net_carrier_send(client_carrier_, &failed_params);
    send_status = iree_status_code(status);
    iree_status_free(status);
  });
  {
    std::unique_lock<std::mutex> lock(writer_gate.mutex);
    writer_gate.condition.wait(lock, [&] { return writer_gate.entered; });
  }

  iree_net_carrier_send_budget_t budget =
      iree_net_carrier_query_send_budget(client_carrier_);
  EXPECT_EQ(budget.bytes, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(budget.slots, 0u);

  std::array<uint8_t, 8> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendResult retry_result;
  retry_result.is_polling = &is_polling_;
  iree_net_send_params_t retry_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &retry_result},
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_carrier_send(client_carrier_, &retry_params));
  EXPECT_EQ(retry_result.completion_count, 0);

  {
    std::lock_guard<std::mutex> lock(writer_gate.mutex);
    writer_gate.release = true;
  }
  writer_gate.condition.notify_all();
  send_thread.join();
  EXPECT_EQ(send_status, IREE_STATUS_OK);
  EXPECT_EQ(failed_result.completion_count, 0);
  PollUntil([&] { return failed_result.completion_count == 1; });
  EXPECT_EQ(failed_result.status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(failed_result.bytes_transferred, 0u);
  budget = iree_net_carrier_query_send_budget(client_carrier_);
  EXPECT_EQ(budget.slots, 1u);

  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &retry_params));
  PollUntil([&] {
    return retry_result.completion_count == 1 &&
           server_context_.received_data.size() == payload.size();
  });
  EXPECT_EQ(retry_result.status_code, IREE_STATUS_OK);
}

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_WASM)
TEST_P(TcpCarrierTest,
       InitialSubmissionFailureCompletesAcceptedSendOnProactor) {
  iree_async_proactor_release(proactor_);
  proactor_ = nullptr;
  iree_async_proactor_options_t proactor_options =
      iree_async_proactor_options_default();
  proactor_options.max_concurrent_operations = 1;
  IREE_ASSERT_OK(iree_async_proactor_create_posix(
      proactor_options, iree_allocator_system(), &proactor_));
  CreateCarrierPair(/*max_send_operations=*/1);

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  std::array<iree_async_notification_signal_operation_t, 64> signal_operations =
      {};
  std::array<AsyncOperationResult, 64> signal_results;
  iree_host_size_t accepted_signal_count = 0;
  for (; accepted_signal_count < signal_operations.size();
       ++accepted_signal_count) {
    iree_async_notification_signal_operation_t* signal_operation =
        &signal_operations[accepted_signal_count];
    iree_async_operation_initialize(
        &signal_operation->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,
        IREE_ASYNC_OPERATION_FLAG_NONE, AsyncOperationCompleted,
        &signal_results[accepted_signal_count]);
    signal_operation->notification = notification;
    signal_operation->wake_count = 1;
    iree_status_t status =
        iree_async_proactor_submit_one(proactor_, &signal_operation->base);
    if (!iree_status_is_ok(status)) {
      EXPECT_EQ(iree_status_code(status), IREE_STATUS_RESOURCE_EXHAUSTED);
      iree_status_free(status);
      break;
    }
  }
  ASSERT_GT(accepted_signal_count, 0u);
  ASSERT_LT(accepted_signal_count, signal_operations.size());

  std::array<uint8_t, 8> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendResult send_result;
  send_result.is_polling = &is_polling_;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &send_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  EXPECT_EQ(send_result.completion_count, 0);
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_carrier_).slots, 0u);

  PollUntil([&] {
    if (send_result.completion_count != 1) {
      return false;
    }
    for (iree_host_size_t i = 0; i < accepted_signal_count; ++i) {
      if (!signal_results[i].completed) {
        return false;
      }
    }
    return true;
  });
  EXPECT_EQ(send_result.status_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(send_result.bytes_transferred, 0u);
  EXPECT_EQ(client_context_.error_count, 1);
  EXPECT_EQ(client_context_.error_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_carrier_).slots, 0u);
  for (iree_host_size_t i = 0; i < accepted_signal_count; ++i) {
    EXPECT_EQ(signal_results[i].status_code, IREE_STATUS_OK);
  }
  iree_async_notification_release(notification);
}
#endif  // !IREE_PLATFORM_WINDOWS && !IREE_PLATFORM_WASM

TEST_P(TcpCarrierTest, GracefulShutdownDrainsAcceptedSendBeforeFin) {
  CreateCarrierPair(/*max_send_operations=*/1);

  constexpr const char kPayload[] = "before-fin";
  SendResult accepted_result;
  iree_net_send_params_t accepted_params = {
      iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(kPayload, sizeof(kPayload))),
      iree_async_span_list_empty(),
      {SendCompleted, &accepted_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &accepted_params));

  IREE_ASSERT_OK(iree_net_carrier_shutdown(client_carrier_));
  iree_net_carrier_send_budget_t budget =
      iree_net_carrier_query_send_budget(client_carrier_);
  EXPECT_EQ(budget.bytes, 0u);
  EXPECT_EQ(budget.slots, 0u);

  std::array<uint8_t, 1> rejected_payload = {};
  iree_async_span_t rejected_span = iree_async_span_from_ptr(
      rejected_payload.data(), rejected_payload.size());
  SendResult rejected_result;
  iree_net_send_params_t rejected_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&rejected_span, 1),
      {SendCompleted, &rejected_result},
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_carrier_send(client_carrier_, &rejected_params));
  EXPECT_EQ(rejected_result.completion_count, 0);

  PollUntil([&] {
    return accepted_result.completion_count == 1 &&
           server_context_.received_data.size() == sizeof(kPayload) &&
           server_context_.eof_count == 1;
  });
  EXPECT_EQ(accepted_result.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_context_.error_count, 0);
  EXPECT_EQ(
      memcmp(server_context_.received_data.data(), kPayload, sizeof(kPayload)),
      0);

  constexpr const char kResponse[] = "after-fin";
  iree_async_span_t response_span =
      iree_async_span_from_ptr(const_cast<char*>(kResponse), sizeof(kResponse));
  SendResult response_result;
  iree_net_send_params_t response_params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&response_span, 1),
      {SendCompleted, &response_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(server_carrier_, &response_params));
  PollUntil([&] {
    return response_result.completion_count == 1 &&
           client_context_.received_data.size() == sizeof(kResponse);
  });
  EXPECT_EQ(response_result.status_code, IREE_STATUS_OK);
  EXPECT_EQ(memcmp(client_context_.received_data.data(), kResponse,
                   sizeof(kResponse)),
            0);
}

TEST_P(TcpCarrierTest, ReceiveFailureBecomesStickyTerminalError) {
  CreateCarrierPair();
  server_context_.receive_error = IREE_STATUS_DATA_LOSS;

  std::array<uint8_t, 8> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendResult send_result;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &send_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  PollUntil([&] {
    return send_result.completion_count == 1 &&
           server_context_.error_count == 1;
  });
  EXPECT_EQ(server_context_.error_code, IREE_STATUS_DATA_LOSS);

  SendResult rejected_result;
  params.completion_callback = {SendCompleted, &rejected_result};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_net_carrier_send(server_carrier_, &params));
  EXPECT_EQ(rejected_result.completion_count, 0);
  EXPECT_EQ(server_context_.error_count, 1);
}

TEST_P(TcpCarrierTest, TerminalFailureRejectsPreparingGeneratedPrefix) {
  CreateCarrierPair();
  server_context_.receive_error = IREE_STATUS_DATA_LOSS;

  PrefixWriterGate writer_gate;
  SendResult preparing_result;
  preparing_result.is_polling = &is_polling_;
  iree_net_send_params_t preparing_params = {
      {
          32,
          PrefixWriterGate::Write,
          &writer_gate,
      },
      iree_async_span_list_empty(),
      {SendCompleted, &preparing_result},
  };
  iree_status_code_t preparing_status = IREE_STATUS_UNKNOWN;
  std::thread preparing_thread([&] {
    iree_status_t status =
        iree_net_carrier_send(server_carrier_, &preparing_params);
    preparing_status = iree_status_code(status);
    iree_status_free(status);
  });
  {
    std::unique_lock<std::mutex> lock(writer_gate.mutex);
    writer_gate.condition.wait(lock, [&] { return writer_gate.entered; });
  }

  uint8_t payload = 0x5A;
  iree_async_span_t span = iree_async_span_from_ptr(&payload, 1);
  SendResult send_result;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &send_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  PollUntil([&] {
    return send_result.completion_count == 1 &&
           server_context_.error_count == 1;
  });

  EXPECT_EQ(server_context_.error_code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(iree_net_carrier_pending_operation_count(server_carrier_), 1);
  {
    std::lock_guard<std::mutex> lock(writer_gate.mutex);
    writer_gate.release = true;
  }
  writer_gate.condition.notify_all();
  preparing_thread.join();
  EXPECT_EQ(preparing_status, IREE_STATUS_OK);
  EXPECT_EQ(preparing_result.completion_count, 0);
  PollUntil([&] { return preparing_result.completion_count == 1; });
  EXPECT_EQ(preparing_result.status_code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(preparing_result.bytes_transferred, 0u);
  EXPECT_EQ(iree_net_carrier_pending_operation_count(server_carrier_), 0);
}

TEST_P(TcpCarrierTest, ReentrantFailureDeactivationDetachesEachSendSlotOnce) {
  ReentrantFailureContext failure_context;
  failure_context.deactivated = &server_deactivated_;
  CreateCarrierPairWithHandlers(
      {Receive, Error, &client_context_},
      {QueueSendsAndFail, DeactivateOnError, &failure_context},
      /*max_send_operations=*/2,
      /*receive_buffer_size=*/4096,
      /*receive_buffer_count=*/4, iree_allocator_system());
  failure_context.carrier = server_carrier_;

  uint8_t payload = 0x5A;
  iree_async_span_t span = iree_async_span_from_ptr(&payload, 1);
  SendResult trigger_result;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &trigger_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  PollUntil([&] {
    return server_deactivated_ && trigger_result.completion_count == 1 &&
           failure_context.submitted_send.completion_count == 1 &&
           failure_context.queued_send.completion_count == 1;
  });

  EXPECT_EQ(failure_context.receive_count, 1);
  EXPECT_EQ(failure_context.error_count, 1);
  EXPECT_EQ(failure_context.error_code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(failure_context.submitted_send.completion_count, 1);
  EXPECT_EQ(failure_context.queued_send.completion_count, 1);
  EXPECT_EQ(failure_context.queued_send.status_code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(failure_context.queued_send.bytes_transferred, 0u);
  EXPECT_EQ(iree_net_carrier_pending_operation_count(server_carrier_), 0);
}

TEST_P(TcpCarrierTest, DeactivationWaitsForGeneratedPrefixWriter) {
  CreateCarrierPair();

  PrefixWriterGate writer_gate;
  SendResult result;
  result.is_polling = &is_polling_;
  iree_net_send_params_t params = {
      {
          128,
          PrefixWriterGate::Write,
          &writer_gate,
      },
      iree_async_span_list_empty(),
      {SendCompleted, &result},
  };
  iree_status_code_t send_status = IREE_STATUS_UNKNOWN;
  std::thread send_thread([&] {
    iree_status_t status = iree_net_carrier_send(client_carrier_, &params);
    send_status = iree_status_code(status);
    iree_status_free(status);
  });
  {
    std::unique_lock<std::mutex> lock(writer_gate.mutex);
    writer_gate.condition.wait(lock, [&] { return writer_gate.entered; });
  }

  CarrierDeactivateResult deactivate_result;
  deactivate_result.deactivated = &client_deactivated_;
  iree_net_carrier_deactivate(client_carrier_, CarrierDeactivationCounted,
                              &deactivate_result);
  EXPECT_FALSE(client_deactivated_);
  {
    std::lock_guard<std::mutex> lock(writer_gate.mutex);
    writer_gate.release = true;
  }
  writer_gate.condition.notify_all();
  send_thread.join();

  PollUntil(
      [&] { return client_deactivated_ && result.completion_count == 1; });
  EXPECT_EQ(deactivate_result.callback_count, 1);
  EXPECT_EQ(send_status, IREE_STATUS_OK);
  EXPECT_EQ(result.status_code, IREE_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(result.bytes_transferred, 0u);
  iree_net_carrier_send_budget_t budget =
      iree_net_carrier_query_send_budget(client_carrier_);
  EXPECT_EQ(budget.bytes, 0u);
  EXPECT_EQ(budget.slots, 0u);
}

TEST_P(TcpCarrierTest, DeactivationCompletesEveryAcceptedSendExactlyOnce) {
  CreateCarrierPair(/*max_send_operations=*/3);

  std::array<std::vector<uint8_t>, 3> payloads = {
      std::vector<uint8_t>(512 * 1024, 0x41),
      std::vector<uint8_t>(512 * 1024, 0x42),
      std::vector<uint8_t>(512 * 1024, 0x43),
  };
  std::array<iree_async_span_t, 3> spans;
  std::array<SendResult, 3> results;
  for (iree_host_size_t i = 0; i < payloads.size(); ++i) {
    spans[i] = iree_async_span_from_ptr(payloads[i].data(), payloads[i].size());
    iree_net_send_params_t params = {
        iree_net_send_prefix_empty(),
        iree_async_span_list_make(&spans[i], 1),
        {SendCompleted, &results[i]},
    };
    IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  }

  BeginDeactivation(client_carrier_, &client_deactivated_);
  PollUntil([&] { return client_deactivated_; });

  for (const SendResult& result : results) {
    EXPECT_EQ(result.completion_count, 1);
  }
}

TEST_P(TcpCarrierTest, PeerDeactivationDeliversOrderlyEof) {
  CreateCarrierPair();
  BeginDeactivation(server_carrier_, &server_deactivated_);
  PollUntil(
      [&] { return server_deactivated_ && client_context_.eof_count == 1; });

  EXPECT_EQ(client_context_.error_count, 0);
}

struct TestRegion {
  // Region passed through the carrier and proactor.
  iree_async_region_t base;

  // Final-release counter owned by the test.
  std::atomic<int>* destroy_count = nullptr;
};

static void DestroyTestRegion(iree_async_region_t* base_region) {
  auto* region = reinterpret_cast<TestRegion*>(base_region);
  ++*region->destroy_count;
}

TEST_P(TcpCarrierTest, RetainsRegisteredRegionThroughSendCompletion) {
  CreateCarrierPair();

  std::array<uint8_t, 32> payload = {};
  std::atomic<int> destroy_count = 0;
  TestRegion region;
  memset(&region.base, 0, sizeof(region.base));
  iree_atomic_ref_count_init(&region.base.ref_count);
  region.base.proactor = proactor_;
  region.base.destroy_fn = DestroyTestRegion;
  region.base.type = IREE_ASYNC_REGION_TYPE_NONE;
  region.base.access_flags = IREE_ASYNC_BUFFER_ACCESS_FLAG_READ;
  region.base.base_ptr = payload.data();
  region.base.length = payload.size();
  region.destroy_count = &destroy_count;

  iree_async_span_t span =
      iree_async_span_make(&region.base, 0, payload.size());
  SendResult result;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  iree_async_region_release(&region.base);
  if (result.completion_count == 0) {
    EXPECT_EQ(destroy_count.load(), 0);
    PollUntil([&] { return result.completion_count == 1; });
  }
  EXPECT_EQ(result.completion_count, 1);
  EXPECT_EQ(result.status_code, IREE_STATUS_OK);
  EXPECT_EQ(destroy_count.load(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    SendModes, TcpCarrierTest,
    ::testing::Values(
        TcpSendMode{"CopiedSocket", IREE_ASYNC_SOCKET_OPTION_NONE,
                    IREE_NET_TCP_DEFAULT_ZERO_COPY_MIN_SEND_SIZE},
        TcpSendMode{"ZeroCopy", IREE_ASYNC_SOCKET_OPTION_ZERO_COPY, 0},
        TcpSendMode{"Adaptive", IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
                    IREE_NET_TCP_DEFAULT_ZERO_COPY_MIN_SEND_SIZE},
        TcpSendMode{"CopiedOverride", IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
                    IREE_HOST_SIZE_MAX}),
    [](const ::testing::TestParamInfo<TcpSendMode>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace iree
