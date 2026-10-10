// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/loopback/carrier.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

enum PollSide {
  kNotPolling = 0,
  kClientPolling = 1,
  kServerPolling = 2,
};

struct ReceiveGate {
  // Serializes entry and release of the receive callback.
  std::mutex mutex;
  // Notifies the test when callback state changes.
  std::condition_variable condition;
  // True after the receive callback has claimed its input.
  bool entered = false;
  // True when the receive callback may return.
  bool release = false;
};

struct EndpointState {
  // Poll-side marker set by the fixture around proactor polling.
  int* current_poll_side = nullptr;
  // Proactor side on which callbacks must execute.
  PollSide expected_poll_side = kNotPolling;
  // Bytes observed across all receive callbacks.
  std::vector<uint8_t> received_bytes;
  // Length of each nonempty received stream chunk.
  std::vector<iree_host_size_t> received_chunk_lengths;
  // Number of terminal error callbacks received.
  int error_count = 0;
  // Most recent terminal error code.
  iree_status_code_t last_error = IREE_STATUS_OK;
  // Status injected from the receive callback.
  iree_status_code_t receive_status = IREE_STATUS_OK;
  // Carrier deactivated synchronously from the next receive callback.
  iree_net_carrier_t* deactivate_during_receive = nullptr;
  // Optional gate that blocks the receive callback after claiming data.
  ReceiveGate* receive_gate = nullptr;
  // True after callback-initiated deactivation completes.
  bool deactivation_completed = false;
  // Whether callback-initiated deactivation completed before receive returned.
  bool deactivation_completed_inside_receive = false;

  static iree_status_t OnReceive(void* user_data, iree_async_span_t data,
                                 iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<EndpointState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_EQ(lease, nullptr);
    const uint8_t* data_ptr = iree_async_span_ptr(data);
    self->received_bytes.insert(self->received_bytes.end(), data_ptr,
                                data_ptr + data.length);
    self->received_chunk_lengths.push_back(data.length);
    if (self->receive_gate) {
      std::unique_lock<std::mutex> lock(self->receive_gate->mutex);
      self->receive_gate->entered = true;
      self->receive_gate->condition.notify_all();
      self->receive_gate->condition.wait(
          lock, [&] { return self->receive_gate->release; });
    }
    if (self->deactivate_during_receive) {
      iree_net_carrier_deactivate(
          self->deactivate_during_receive,
          [](void* callback_user_data) {
            static_cast<EndpointState*>(callback_user_data)
                ->deactivation_completed = true;
          },
          self);
      self->deactivation_completed_inside_receive =
          self->deactivation_completed;
      self->deactivate_during_receive = nullptr;
    }
    return self->receive_status == IREE_STATUS_OK
               ? iree_ok_status()
               : iree_make_status(self->receive_status,
                                  "injected receive failure");
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<EndpointState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    self->last_error = iree_status_code(status);
    iree_status_free(status);
  }
};

struct SendState {
  // Poll-side marker set by the fixture around proactor polling.
  int* current_poll_side = nullptr;
  // Proactor side on which completions must execute.
  PollSide expected_poll_side = kNotPolling;
  // Number of terminal send callbacks received.
  int completion_count = 0;
  // Status code from each send callback.
  std::vector<iree_status_code_t> status_codes;
  // Transferred byte count from each send callback.
  std::vector<iree_host_size_t> byte_counts;
  // Optional action invoked after recording a completion.
  std::function<void()> completion_action;

  static void OnCompletion(void* user_data, iree_status_t status,
                           iree_host_size_t bytes_transferred) {
    auto* self = static_cast<SendState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->completion_count;
    self->status_codes.push_back(iree_status_code(status));
    self->byte_counts.push_back(bytes_transferred);
    iree_status_free(status);
    if (self->completion_action) {
      self->completion_action();
    }
  }

  iree_net_send_completion_callback_t callback() {
    return {.fn = OnCompletion, .user_data = this};
  }
};

struct DeactivationState {
  // Poll-side marker set by the fixture around proactor polling.
  int* current_poll_side = nullptr;
  // Proactor side on which deactivation must complete.
  PollSide expected_poll_side = kNotPolling;
  // True after the deactivation callback executes.
  bool completed = false;

  static void OnDeactivated(void* user_data) {
    auto* self = static_cast<DeactivationState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    self->completed = true;
  }
};

struct PrefixWriter {
  // Byte copied into every generated-prefix position.
  uint8_t value = 0;
  // Number of writer invocations.
  int call_count = 0;
  // Alignment observed for transport-owned storage.
  uintptr_t target_alignment = 0;
  // Status returned after writing the generated prefix.
  iree_status_code_t result_code = IREE_STATUS_OK;

  static iree_status_t Write(void* user_data, iree_byte_span_t target) {
    auto* self = static_cast<PrefixWriter*>(user_data);
    ++self->call_count;
    self->target_alignment = reinterpret_cast<uintptr_t>(target.data) %
                             IREE_NET_SEND_PREFIX_ALIGNMENT;
    memset(target.data, self->value, target.data_length);
    return self->result_code == IREE_STATUS_OK
               ? iree_ok_status()
               : iree_make_status(self->result_code,
                                  "injected prefix writer failure");
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
                                  "injected prefix writer failure");
  }
};

struct TestRegion {
  // Region passed through the carrier API.
  iree_async_region_t base = {};
  // Set when the final retained reference is released.
  bool* destroyed = nullptr;
  // Registered storage referenced by the test span.
  uint8_t storage[16] = {};

  static void Destroy(iree_async_region_t* base_region) {
    auto* self = reinterpret_cast<TestRegion*>(base_region);
    *self->destroyed = true;
    delete self;
  }
};

class LoopbackCarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &client_proactor_));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &server_proactor_));
    client_endpoint_.current_poll_side = &current_poll_side_;
    client_endpoint_.expected_poll_side = kClientPolling;
    server_endpoint_.current_poll_side = &current_poll_side_;
    server_endpoint_.expected_poll_side = kServerPolling;
  }

  void TearDown() override {
    DeactivateAndRelease(client_, client_proactor_, kClientPolling);
    DeactivateAndRelease(server_, server_proactor_, kServerPolling);
    iree_async_proactor_release(client_proactor_);
    iree_async_proactor_release(server_proactor_);
  }

  void CreatePair(
      const iree_net_loopback_carrier_options_t* options = nullptr) {
    IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
        client_proactor_, server_proactor_, options, iree_allocator_system(),
        &client_, &server_));
    IREE_ASSERT_OK(iree_net_carrier_set_handlers(
        client_, {
                     .on_receive = EndpointState::OnReceive,
                     .on_error = EndpointState::OnError,
                     .user_data = &client_endpoint_,
                 }));
    IREE_ASSERT_OK(iree_net_carrier_set_handlers(
        server_, {
                     .on_receive = EndpointState::OnReceive,
                     .on_error = EndpointState::OnError,
                     .user_data = &server_endpoint_,
                 }));
  }

  void ActivateBoth() {
    IREE_ASSERT_OK(iree_net_carrier_activate(client_));
    IREE_ASSERT_OK(iree_net_carrier_activate(server_));
  }

  void Poll(iree_async_proactor_t* proactor, PollSide side) {
    current_poll_side_ = side;
    IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                            /*out_completed_count=*/nullptr));
    current_poll_side_ = kNotPolling;
  }

  void PollUntil(iree_async_proactor_t* proactor, PollSide side,
                 const std::function<bool()>& condition) {
    while (!condition()) {
      Poll(proactor, side);
    }
  }

  void DeactivateAndRelease(iree_net_carrier_t*& carrier,
                            iree_async_proactor_t* proactor, PollSide side) {
    if (!carrier) {
      return;
    }
    iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
    bool deactivation_completed = state == IREE_NET_CARRIER_STATE_DEACTIVATED;
    if (state == IREE_NET_CARRIER_STATE_CREATED ||
        state == IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_net_carrier_deactivate(
          carrier,
          [](void* user_data) { *static_cast<bool*>(user_data) = true; },
          &deactivation_completed);
    }
    if (!deactivation_completed) {
      PollUntil(proactor, side, [&] {
        return iree_net_carrier_state(carrier) ==
               IREE_NET_CARRIER_STATE_DEACTIVATED;
      });
    }
    EXPECT_EQ(iree_net_carrier_state(carrier),
              IREE_NET_CARRIER_STATE_DEACTIVATED);
    iree_net_carrier_release(carrier);
    carrier = nullptr;
  }

  int current_poll_side_ = kNotPolling;
  iree_async_proactor_t* client_proactor_ = nullptr;
  iree_async_proactor_t* server_proactor_ = nullptr;
  iree_net_carrier_t* client_ = nullptr;
  iree_net_carrier_t* server_ = nullptr;
  EndpointState client_endpoint_;
  EndpointState server_endpoint_;
};

TEST_F(LoopbackCarrierTest, UsesOwningProactorsAndRetainsScatterGatherData) {
  CreatePair();
  ActivateBoth();

  EXPECT_TRUE(iree_all_bits_set(iree_net_carrier_capabilities(client_),
                                IREE_NET_CARRIER_CAPABILITY_RELIABLE |
                                    IREE_NET_CARRIER_CAPABILITY_ORDERED |
                                    IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX));
  EXPECT_FALSE(iree_any_bit_set(iree_net_carrier_capabilities(client_),
                                IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_RX));

  char first[] = "alpha";
  char second[] = "beta";
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(first, sizeof(first) - 1),
      iree_async_span_empty(),
      iree_async_span_from_ptr(second, sizeof(second) - 1),
  };
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  // The loopback carrier retains storage instead of copying it at admission.
  first[0] = 'A';
  PollUntil(server_proactor_, kServerPolling, [&] {
    return server_endpoint_.received_chunk_lengths.size() == 2;
  });
  EXPECT_EQ(send_state.completion_count, 0);
  EXPECT_EQ(server_endpoint_.received_chunk_lengths,
            (std::vector<iree_host_size_t>{5, 4}));
  EXPECT_EQ(std::string(server_endpoint_.received_bytes.begin(),
                        server_endpoint_.received_bytes.end()),
            "Alphabeta");

  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 1; });
  EXPECT_EQ(send_state.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
  EXPECT_EQ(
      send_state.byte_counts,
      (std::vector<iree_host_size_t>{sizeof(first) + sizeof(second) - 2}));
}

TEST_F(LoopbackCarrierTest, QueuesSendUntilPeerActivation) {
  CreatePair();
  IREE_ASSERT_OK(iree_net_carrier_activate(client_));

  char payload[] = "bootstrap";
  iree_async_span_t span =
      iree_async_span_from_ptr(payload, sizeof(payload) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  EXPECT_EQ(server_endpoint_.received_bytes.size(), 0u);
  EXPECT_EQ(send_state.completion_count, 0);

  IREE_ASSERT_OK(iree_net_carrier_activate(server_));
  PollUntil(server_proactor_, kServerPolling,
            [&] { return !server_endpoint_.received_bytes.empty(); });
  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 1; });
  EXPECT_EQ(std::string(server_endpoint_.received_bytes.begin(),
                        server_endpoint_.received_bytes.end()),
            "bootstrap");
  EXPECT_EQ(send_state.status_codes[0], IREE_STATUS_OK);
}

TEST_F(LoopbackCarrierTest, IdleDeactivationCompletesOnOwningProactor) {
  CreatePair();
  ActivateBoth();

  DeactivationState state;
  state.current_poll_side = &current_poll_side_;
  state.expected_poll_side = kClientPolling;
  iree_net_carrier_deactivate(client_, DeactivationState::OnDeactivated,
                              &state);
  EXPECT_FALSE(state.completed);

  PollUntil(client_proactor_, kClientPolling, [&] { return state.completed; });
  EXPECT_EQ(iree_net_carrier_state(client_),
            IREE_NET_CARRIER_STATE_DEACTIVATED);
}

TEST_F(LoopbackCarrierTest, RetainsRegisteredRegionUntilSendCompletion) {
  CreatePair();
  ActivateBoth();

  bool region_destroyed = false;
  auto* region = new TestRegion();
  iree_atomic_ref_count_init(&region->base.ref_count);
  region->base.destroy_fn = TestRegion::Destroy;
  region->base.base_ptr = region->storage;
  region->base.length = sizeof(region->storage);
  region->destroyed = &region_destroyed;
  memcpy(region->storage, "registered", 10);

  iree_async_span_t span = iree_async_span_make(&region->base, 0, 10);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  iree_async_region_release(&region->base);
  EXPECT_FALSE(region_destroyed);

  PollUntil(server_proactor_, kServerPolling,
            [&] { return !server_endpoint_.received_bytes.empty(); });
  EXPECT_FALSE(region_destroyed);
  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 1; });
  EXPECT_TRUE(region_destroyed);
  EXPECT_EQ(std::string(server_endpoint_.received_bytes.begin(),
                        server_endpoint_.received_bytes.end()),
            "registered");
}

TEST_F(LoopbackCarrierTest, CreatedPeerDestructionFailsQueuedSend) {
  CreatePair();
  IREE_ASSERT_OK(iree_net_carrier_activate(client_));

  char payload[] = "bootstrap";
  iree_async_span_t span =
      iree_async_span_from_ptr(payload, sizeof(payload) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  iree_net_carrier_release(server_);
  server_ = nullptr;
  PollUntil(client_proactor_, kClientPolling, [&] {
    return client_endpoint_.error_count == 1 &&
           send_state.completion_count == 1;
  });
  EXPECT_EQ(client_endpoint_.last_error, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(send_state.status_codes[0], IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(send_state.byte_counts[0], 0u);
}

TEST_F(LoopbackCarrierTest, EnforcesSlotBackpressureAndCompletionRetryEdge) {
  iree_net_loopback_carrier_options_t options =
      iree_net_loopback_carrier_options_default();
  options.max_send_operations = 2;
  CreatePair(&options);
  ActivateBoth();

  char payload[] = "x";
  iree_async_span_t span =
      iree_async_span_from_ptr(payload, sizeof(payload) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_).slots, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_carrier_send(client_, &params));

  PollUntil(server_proactor_, kServerPolling, [&] {
    return server_endpoint_.received_chunk_lengths.size() == 2;
  });
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_).slots, 0u);
  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 2; });
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_).slots, 2u);
  IREE_EXPECT_OK(iree_net_carrier_send(client_, &params));
  PollUntil(server_proactor_, kServerPolling, [&] {
    return server_endpoint_.received_chunk_lengths.size() == 3;
  });
  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 3; });
}

TEST_F(LoopbackCarrierTest, GeneratedPrefixUsesAlignedTransportStorage) {
  iree_net_loopback_carrier_options_t options =
      iree_net_loopback_carrier_options_default();
  options.max_send_operations = 2;
  CreatePair(&options);
  ActivateBoth();

  PrefixWriter writer = {.value = 0xA5};
  constexpr char kSuffix[] = "borrowed";
  iree_async_span_t suffix =
      iree_async_span_from_ptr((void*)kSuffix, sizeof(kSuffix) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix =
          {
              .length = 32,
              .write = PrefixWriter::Write,
              .user_data = &writer,
          },
      .data = iree_async_span_list_make(&suffix, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  EXPECT_EQ(writer.call_count, 1);
  EXPECT_EQ(writer.target_alignment, 0u);
  PollUntil(server_proactor_, kServerPolling,
            [&] { return !server_endpoint_.received_bytes.empty(); });
  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 1; });
  ASSERT_EQ(server_endpoint_.received_bytes.size(), 32 + sizeof(kSuffix) - 1);
  EXPECT_TRUE(std::all_of(server_endpoint_.received_bytes.begin(),
                          server_endpoint_.received_bytes.begin() + 32,
                          [](uint8_t value) { return value == 0xA5; }));
  EXPECT_EQ(std::string(server_endpoint_.received_bytes.begin() + 32,
                        server_endpoint_.received_bytes.end()),
            kSuffix);
}

TEST_F(LoopbackCarrierTest,
       FailingGeneratedPrefixCompletesAndRefreshesOneSlotBudget) {
  iree_net_loopback_carrier_options_t options =
      iree_net_loopback_carrier_options_default();
  options.max_send_operations = 1;
  CreatePair(&options);
  ActivateBoth();

  PrefixWriterGate writer_gate;
  writer_gate.result_code = IREE_STATUS_CANCELLED;
  SendState first_send;
  first_send.current_poll_side = &current_poll_side_;
  first_send.expected_poll_side = kClientPolling;
  iree_net_send_params_t first_params = {
      .generated_prefix =
          {
              .length = 1,
              .write = PrefixWriterGate::Write,
              .user_data = &writer_gate,
          },
      .data = iree_async_span_list_empty(),
      .completion_callback = first_send.callback(),
  };
  iree_status_code_t first_send_code = IREE_STATUS_UNKNOWN;
  std::thread send_thread([&] {
    iree_status_t status = iree_net_carrier_send(client_, &first_params);
    first_send_code = iree_status_code(status);
    iree_status_free(status);
  });
  {
    std::unique_lock<std::mutex> lock(writer_gate.mutex);
    writer_gate.condition.wait(lock, [&] { return writer_gate.entered; });
  }

  EXPECT_EQ(iree_net_carrier_query_send_budget(client_).slots, 0u);
  char retry_payload = 'x';
  iree_async_span_t retry_span =
      iree_async_span_from_ptr(&retry_payload, sizeof(retry_payload));
  SendState retry_send;
  retry_send.current_poll_side = &current_poll_side_;
  retry_send.expected_poll_side = kClientPolling;
  iree_net_send_params_t retry_params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&retry_span, 1),
      .completion_callback = retry_send.callback(),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_carrier_send(client_, &retry_params));
  EXPECT_EQ(retry_send.completion_count, 0);

  {
    std::lock_guard<std::mutex> lock(writer_gate.mutex);
    writer_gate.release = true;
  }
  writer_gate.condition.notify_all();
  send_thread.join();

  EXPECT_EQ(first_send_code, IREE_STATUS_OK);
  EXPECT_EQ(first_send.completion_count, 0);
  PollUntil(client_proactor_, kClientPolling,
            [&] { return first_send.completion_count == 1; });
  EXPECT_EQ(first_send.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_CANCELLED}));
  EXPECT_EQ(first_send.byte_counts, (std::vector<iree_host_size_t>{0}));
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_).slots, 1u);
  EXPECT_TRUE(server_endpoint_.received_bytes.empty());

  IREE_ASSERT_OK(iree_net_carrier_send(client_, &retry_params));
  PollUntil(server_proactor_, kServerPolling,
            [&] { return server_endpoint_.received_bytes.size() == 1; });
  PollUntil(client_proactor_, kClientPolling,
            [&] { return retry_send.completion_count == 1; });
  EXPECT_EQ(retry_send.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
}

TEST_F(LoopbackCarrierTest, DrainsCompletionQueuedFromCompletionCallback) {
  CreatePair();
  ActivateBoth();

  PrefixWriter failing_writer;
  failing_writer.result_code = IREE_STATUS_CANCELLED;
  SendState nested_send;
  nested_send.current_poll_side = &current_poll_side_;
  nested_send.expected_poll_side = kClientPolling;
  iree_net_send_params_t nested_params = {
      .generated_prefix =
          {
              .length = 1,
              .write = PrefixWriter::Write,
              .user_data = &failing_writer,
          },
      .data = iree_async_span_list_empty(),
      .completion_callback = nested_send.callback(),
  };

  char payload = 'x';
  iree_async_span_t span = iree_async_span_from_ptr(&payload, sizeof(payload));
  SendState initial_send;
  initial_send.current_poll_side = &current_poll_side_;
  initial_send.expected_poll_side = kClientPolling;
  initial_send.completion_action = [&] {
    IREE_EXPECT_OK(iree_net_carrier_send(client_, &nested_params));
  };
  iree_net_send_params_t initial_params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = initial_send.callback(),
  };

  IREE_ASSERT_OK(iree_net_carrier_send(client_, &initial_params));
  PollUntil(server_proactor_, kServerPolling,
            [&] { return server_endpoint_.received_bytes.size() == 1; });
  Poll(client_proactor_, kClientPolling);

  EXPECT_EQ(initial_send.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_OK}));
  EXPECT_EQ(nested_send.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_CANCELLED}));
  EXPECT_EQ(nested_send.byte_counts, (std::vector<iree_host_size_t>{0}));
}

TEST_F(LoopbackCarrierTest, DeactivationCompletesAcceptedGeneratedPrefixSend) {
  CreatePair();
  ActivateBoth();

  PrefixWriterGate writer_gate;
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix =
          {
              .length = 64,
              .write = PrefixWriterGate::Write,
              .user_data = &writer_gate,
          },
      .data = iree_async_span_list_empty(),
      .completion_callback = send_state.callback(),
  };
  iree_status_code_t send_status_code = IREE_STATUS_UNKNOWN;
  std::thread send_thread([&] {
    iree_status_t status = iree_net_carrier_send(client_, &params);
    send_status_code = iree_status_code(status);
    iree_status_free(status);
  });
  {
    std::unique_lock<std::mutex> lock(writer_gate.mutex);
    writer_gate.condition.wait(lock, [&] { return writer_gate.entered; });
  }

  bool deactivated = false;
  iree_net_carrier_deactivate(
      client_, [](void* user_data) { *static_cast<bool*>(user_data) = true; },
      &deactivated);
  EXPECT_FALSE(deactivated);
  {
    std::lock_guard<std::mutex> lock(writer_gate.mutex);
    writer_gate.release = true;
  }
  writer_gate.condition.notify_all();
  send_thread.join();

  PollUntil(client_proactor_, kClientPolling, [&] { return deactivated; });
  EXPECT_EQ(send_status_code, IREE_STATUS_OK);
  EXPECT_EQ(send_state.completion_count, 1);
  EXPECT_EQ(send_state.status_codes,
            (std::vector<iree_status_code_t>{IREE_STATUS_FAILED_PRECONDITION}));
  EXPECT_EQ(send_state.byte_counts, (std::vector<iree_host_size_t>{0}));
  EXPECT_EQ(iree_net_carrier_pending_operation_count(client_), 0);
}

TEST_F(LoopbackCarrierTest, PeerDepartureFailsUndeliveredSend) {
  CreatePair();
  ActivateBoth();

  char payload[] = "queued";
  iree_async_span_t span =
      iree_async_span_from_ptr(payload, sizeof(payload) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  bool server_deactivated = false;
  iree_net_carrier_deactivate(
      server_, [](void* user_data) { *static_cast<bool*>(user_data) = true; },
      &server_deactivated);
  PollUntil(client_proactor_, kClientPolling, [&] {
    return client_endpoint_.error_count == 1 &&
           send_state.completion_count == 1;
  });
  EXPECT_EQ(client_endpoint_.last_error, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(send_state.status_codes[0], IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(send_state.byte_counts[0], 0u);
  EXPECT_TRUE(server_endpoint_.received_bytes.empty());
  PollUntil(server_proactor_, kServerPolling,
            [&] { return server_deactivated; });
}

TEST_F(LoopbackCarrierTest, ReceiveFailureRemainsLocalToReceiver) {
  CreatePair();
  ActivateBoth();
  server_endpoint_.receive_status = IREE_STATUS_DATA_LOSS;

  char payload[] = "malformed frame";
  iree_async_span_t span =
      iree_async_span_from_ptr(payload, sizeof(payload) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  PollUntil(server_proactor_, kServerPolling,
            [&] { return server_endpoint_.error_count == 1; });
  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 1; });

  EXPECT_EQ(server_endpoint_.last_error, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(client_endpoint_.error_count, 0);
  EXPECT_EQ(send_state.status_codes[0], IREE_STATUS_OK);
  EXPECT_EQ(send_state.byte_counts[0], sizeof(payload) - 1);
}

TEST_F(LoopbackCarrierTest, DeactivationWaitsForClaimedReceive) {
  CreatePair();
  ActivateBoth();
  server_endpoint_.deactivate_during_receive = server_;

  char payload[] = "last delivery";
  iree_async_span_t span =
      iree_async_span_from_ptr(payload, sizeof(payload) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  PollUntil(server_proactor_, kServerPolling,
            [&] { return server_endpoint_.deactivation_completed; });
  EXPECT_FALSE(server_endpoint_.deactivation_completed_inside_receive);
  EXPECT_EQ(iree_net_carrier_state(server_),
            IREE_NET_CARRIER_STATE_DEACTIVATED);

  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_state.completion_count == 1; });
  EXPECT_EQ(send_state.status_codes[0], IREE_STATUS_OK);
}

TEST_F(LoopbackCarrierTest, ConcurrentDeactivationWaitsForClaimedReceive) {
  CreatePair();
  ActivateBoth();
  ReceiveGate receive_gate;
  server_endpoint_.receive_gate = &receive_gate;

  char payload[] = "held delivery";
  iree_async_span_t span =
      iree_async_span_from_ptr(payload, sizeof(payload) - 1);
  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  std::atomic<bool> server_deactivated{false};
  std::thread deactivation_thread([&] {
    {
      std::unique_lock<std::mutex> lock(receive_gate.mutex);
      receive_gate.condition.wait(lock, [&] { return receive_gate.entered; });
    }
    iree_net_carrier_deactivate(
        server_,
        [](void* user_data) {
          static_cast<std::atomic<bool>*>(user_data)->store(
              true, std::memory_order_release);
        },
        &server_deactivated);
    EXPECT_FALSE(server_deactivated.load(std::memory_order_acquire));
    {
      std::lock_guard<std::mutex> lock(receive_gate.mutex);
      receive_gate.release = true;
    }
    receive_gate.condition.notify_all();
  });
  Poll(server_proactor_, kServerPolling);
  deactivation_thread.join();

  EXPECT_TRUE(server_deactivated.load(std::memory_order_acquire));
  EXPECT_EQ(iree_net_carrier_state(server_),
            IREE_NET_CARRIER_STATE_DEACTIVATED);
  PollUntil(client_proactor_, kClientPolling, [&] {
    return client_endpoint_.error_count == 1 &&
           send_state.completion_count == 1;
  });
  EXPECT_EQ(send_state.status_codes[0], IREE_STATUS_OK);
}

TEST_F(LoopbackCarrierTest, RejectsInvalidSpanStorageAndRanges) {
  CreatePair();
  ActivateBoth();

  SendState send_state;
  send_state.current_poll_side = &current_poll_side_;
  send_state.expected_poll_side = kClientPolling;
  iree_async_span_t span = iree_async_span_from_ptr(nullptr, 4);
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback = send_state.callback(),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(client_, &params));

  uint8_t storage[4] = {0};
  iree_async_region_t region = {
      .base_ptr = storage,
      .length = sizeof(storage),
  };
  span = iree_async_span_make(&region, 3, 2);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_carrier_send(client_, &params));

  region.base_ptr = nullptr;
  region.length = 4;
  span = iree_async_span_make(&region, 0, 4);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(client_, &params));
  EXPECT_EQ(send_state.completion_count, 0);
}

}  // namespace
}  // namespace iree
