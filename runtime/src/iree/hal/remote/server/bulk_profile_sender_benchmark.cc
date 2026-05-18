// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/server/bulk.h"
#include "iree/hal/remote/server/bulk_profile_sender.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/bulk_test_util.h"
#include "iree/hal/remote/server/profile_relay.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_REMOTE_PROFILE_BENCHMARK_BURST_COUNT 64
#define IREE_HAL_REMOTE_PROFILE_BENCHMARK_PAYLOAD_LENGTH 64

namespace {

using iree::hal::remote::server::testing::MockCarrier;
using iree::hal::remote::server::testing::MockEndpoint;
using iree::hal::remote::server::testing::ParseBulkHeader;
using iree::hal::remote::server::testing::TestBufferPool;

typedef struct benchmark_profile_sink_t {
  // Resource header for iree_hal_profile_sink_t lifetime management.
  iree_hal_resource_t resource;

  // Host allocator used to release this sink.
  iree_allocator_t host_allocator;
} benchmark_profile_sink_t;

static benchmark_profile_sink_t* benchmark_profile_sink_cast(
    iree_hal_profile_sink_t* base_sink) {
  return reinterpret_cast<benchmark_profile_sink_t*>(base_sink);
}

static void benchmark_profile_sink_destroy(iree_hal_profile_sink_t* base_sink) {
  benchmark_profile_sink_t* sink = benchmark_profile_sink_cast(base_sink);
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t benchmark_profile_sink_begin_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)base_sink;
  (void)metadata;
  return iree_ok_status();
}

static iree_status_t benchmark_profile_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  (void)base_sink;
  (void)metadata;
  (void)iovec_count;
  (void)iovecs;
  return iree_ok_status();
}

static iree_status_t benchmark_profile_sink_end_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)base_sink;
  (void)metadata;
  (void)session_status_code;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t benchmark_profile_sink_vtable = {
    /*.destroy=*/benchmark_profile_sink_destroy,
    /*.begin_session=*/benchmark_profile_sink_begin_session,
    /*.write=*/benchmark_profile_sink_write,
    /*.end_session=*/benchmark_profile_sink_end_session,
};

static iree_status_t benchmark_profile_sink_create(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  *out_sink = NULL;
  benchmark_profile_sink_t* sink = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink);
  if (iree_status_is_ok(status)) {
    memset(sink, 0, sizeof(*sink));
    iree_hal_resource_initialize(&benchmark_profile_sink_vtable,
                                 &sink->resource);
    sink->host_allocator = host_allocator;
    *out_sink = reinterpret_cast<iree_hal_profile_sink_t*>(sink);
  }
  return status;
}

static iree_status_t AllocateProfilePayload(uint64_t sequence,
                                            iree_allocator_t host_allocator,
                                            iree_byte_span_t* out_payload) {
  *out_payload = iree_byte_span_empty();
  const iree_host_size_t total_length =
      sizeof(iree_hal_remote_profile_transfer_header_t) +
      IREE_HAL_REMOTE_PROFILE_BENCHMARK_PAYLOAD_LENGTH;
  void* storage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_length, &storage);
  if (iree_status_is_ok(status)) {
    memset(storage, 0, total_length);
    iree_hal_remote_profile_transfer_header_t header = {};
    header.sequence = sequence;
    header.session_id = 1;
    header.payload_length = IREE_HAL_REMOTE_PROFILE_BENCHMARK_PAYLOAD_LENGTH;
    header.physical_device_ordinal = UINT32_MAX;
    header.queue_ordinal = UINT32_MAX;
    header.callback_type = IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK;
    memcpy(storage, &header, sizeof(header));
    uint8_t* body = static_cast<uint8_t*>(storage) + sizeof(header);
    memset(body, (int)(sequence & 0xFF),
           IREE_HAL_REMOTE_PROFILE_BENCHMARK_PAYLOAD_LENGTH);
    *out_payload = iree_make_byte_span(storage, total_length);
  }
  return status;
}

struct ProfileBulkCallbacks {
  // Session slot receiving bulk-channel callbacks.
  iree_hal_remote_server_session_t* session_slot = nullptr;

  // Number of send completions observed by the bulk channel.
  uint64_t send_completion_count = 0;

  // Last send completion status code observed by the bulk channel.
  iree_status_code_t last_send_completion_code = IREE_STATUS_OK;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_size,
                               iree_net_bulk_frame_flags_t flags) {
    (void)user_data;
    (void)transfer_id;
    (void)total_size;
    (void)flags;
    return iree_ok_status();
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
    return iree_ok_status();
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    (void)user_data;
    (void)transfer_id;
    return iree_ok_status();
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t abort_data,
                               iree_async_buffer_lease_t* lease) {
    (void)user_data;
    (void)transfer_id;
    (void)abort_data;
    (void)lease;
    return iree_ok_status();
  }

  static void OnTransportError(void* user_data, iree_status_t status) {
    (void)user_data;
    iree_status_free(status);
  }

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
    ++callbacks->send_completion_count;
    callbacks->last_send_completion_code = iree_status_code(status);
    iree_hal_remote_server_bulk_on_send_complete(callbacks->session_slot,
                                                 operation_user_data, status);
  }

  static void OnCredit(void* user_data, uint32_t credit_delta,
                       uint32_t available_credit_count) {
    (void)credit_delta;
    (void)available_credit_count;
    ProfileBulkCallbacks* callbacks =
        static_cast<ProfileBulkCallbacks*>(user_data);
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

class ProfileBenchmarkContext {
 public:
  iree_status_t Initialize(iree_host_size_t active_transfer_capacity) {
    iree_atomic_ref_count_init(&server_.ref_count);
    server_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&server_.session_mutex);
    server_mutex_initialized_ = true;

    session_.server = &server_;
    session_.session_id = 1;
    session_.session = reinterpret_cast<iree_net_session_t*>(this);
    iree_hal_remote_server_bulk_session_options_t bulk_options =
        iree_hal_remote_server_bulk_session_options_default();
    bulk_options.active_transfer_capacity = active_transfer_capacity;
    iree_status_t status = iree_hal_remote_server_bulk_session_allocate(
        &session_, &bulk_options, iree_allocator_system(),
        &session_.bulk_session);

    if (iree_status_is_ok(status)) {
      carrier_ = MockCarrier::Create();
      carrier_->sends.reserve(IREE_HAL_REMOTE_PROFILE_BENCHMARK_BURST_COUNT *
                              3);
      endpoint_.carrier = carrier_.get();
      channel_callbacks_.session_slot = &session_;
      TestBufferPool buffer_pool;
      status =
          buffer_pool.Initialize(/*buffer_count=*/16, /*buffer_size=*/1024);
      if (iree_status_is_ok(status)) {
        status = iree_net_bulk_channel_create(
            endpoint_.as_endpoint(), nullptr, buffer_pool.release(),
            channel_callbacks_.MakeCallbacks(), iree_allocator_system(),
            &bulk_channel_);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_activate(bulk_channel_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_session_attach_channel(
          &session_, bulk_channel_);
    }
    if (iree_status_is_ok(status)) {
      status = benchmark_profile_sink_create(iree_allocator_system(),
                                             &profile_sink_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_profile_relay_prepare_begin(
          &session_, profile_sink_);
    }
    return status;
  }

  ~ProfileBenchmarkContext() {
    CompleteAllCapturedSends();
    DropPendingAndActiveTransfers();
    if (session_.bulk_session) {
      iree_hal_remote_server_profile_relay_deinitialize(&session_);
      iree_hal_remote_server_bulk_session_free(session_.bulk_session);
      session_.bulk_session = NULL;
    }
    iree_net_bulk_channel_release(bulk_channel_);
    iree_hal_profile_sink_release(profile_sink_);
    if (server_mutex_initialized_) {
      iree_slim_mutex_deinitialize(&server_.session_mutex);
    }
  }

  iree_status_t SubmitProfilePayload(uint64_t sequence) {
    iree_byte_span_t payload = iree_byte_span_empty();
    iree_status_t status =
        AllocateProfilePayload(sequence, server_.host_allocator, &payload);
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_profile_submit_transfer(
          &session_, session_.session_id, profile_sink_, payload);
    }
    return status;
  }

  iree_status_t CompleteNextCapturedSend(iree_host_size_t send_index,
                                         iree_status_t status) {
    if (send_index >= carrier_->sends.size()) {
      iree_status_free(status);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "benchmark expected captured send");
    }
    carrier_->CompleteSend(send_index, status);
    return iree_ok_status();
  }

  void CompleteAllCapturedSends() {
    if (!carrier_) return;
    for (iree_host_size_t i = 0; i < carrier_->sends.size(); ++i) {
      if (!carrier_->sends[i].completed) {
        carrier_->CompleteSend(i, iree_ok_status());
      }
    }
  }

  void DropPendingAndActiveTransfers() {
    if (!session_.bulk_session) return;
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfers = NULL;
    iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(&session_));
    pending_transfers =
        iree_hal_remote_server_profile_take_pending_transfers_locked(&session_);
    if (iree_hal_remote_server_bulk_session_scheduler(&session_)) {
      iree_hal_remote_bulk_transfer_scheduler_clear(
          iree_hal_remote_server_bulk_session_scheduler(&session_));
    }
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(&session_));

    iree_hal_remote_server_profile_pending_transfer_free_list(
        &server_, pending_transfers);
  }

  void ResetCapturedSends() {
    if (!carrier_) return;
    carrier_->sends.clear();
    channel_callbacks_.send_completion_count = 0;
    channel_callbacks_.last_send_completion_code = IREE_STATUS_OK;
  }

  iree_status_t GrantChunkCredit(uint32_t credit_delta) {
    return endpoint_.InjectCredit(credit_delta);
  }

  iree_host_size_t captured_send_count() const {
    return carrier_->sends.size();
  }

  iree_net_bulk_frame_header_t CapturedSendHeader(iree_host_size_t send_index) {
    return ParseBulkHeader(carrier_->sends[send_index].data);
  }

  iree_hal_remote_server_session_t* session() { return &session_; }

 private:
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
  iree_net_bulk_channel_t* bulk_channel_ = NULL;

  // Active profile sink passed to the server profile relay.
  iree_hal_profile_sink_t* profile_sink_ = NULL;

  // True once |server_.session_mutex| has been initialized.
  bool server_mutex_initialized_ = false;
};

IREE_BENCHMARK_FN(BM_SubmitBurstStalledCredit) {
  (void)benchmark_def;

  ProfileBenchmarkContext context;
  iree_status_t status = context.Initialize(/*active_transfer_capacity=*/1);
  uint64_t sequence = 1;

  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(
             benchmark_state, IREE_HAL_REMOTE_PROFILE_BENCHMARK_BURST_COUNT)) {
    for (iree_host_size_t i = 0;
         i < IREE_HAL_REMOTE_PROFILE_BENCHMARK_BURST_COUNT &&
         iree_status_is_ok(status);
         ++i) {
      status = context.SubmitProfilePayload(sequence++);
    }
    iree_optimization_barrier(context.session());

    iree_benchmark_pause_timing(benchmark_state);
    context.CompleteAllCapturedSends();
    context.DropPendingAndActiveTransfers();
    context.ResetCapturedSends();
    iree_benchmark_resume_timing(benchmark_state);
  }
  return status;
}

IREE_BENCHMARK_REGISTER(BM_SubmitBurstStalledCredit);

IREE_BENCHMARK_FN(BM_DrainBurstDelayedCompletions) {
  (void)benchmark_def;

  ProfileBenchmarkContext context;
  iree_status_t status = context.Initialize(/*active_transfer_capacity=*/1);
  uint64_t sequence = 1;

  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(
             benchmark_state, IREE_HAL_REMOTE_PROFILE_BENCHMARK_BURST_COUNT)) {
    for (iree_host_size_t i = 0;
         i < IREE_HAL_REMOTE_PROFILE_BENCHMARK_BURST_COUNT &&
         iree_status_is_ok(status);
         ++i) {
      status = context.SubmitProfilePayload(sequence++);
    }

    iree_host_size_t send_index = 0;
    for (iree_host_size_t i = 0;
         i < IREE_HAL_REMOTE_PROFILE_BENCHMARK_BURST_COUNT &&
         iree_status_is_ok(status);
         ++i) {
      iree_net_bulk_frame_header_t start_header = {};
      if (send_index >= context.captured_send_count()) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "benchmark missing profile START frame");
      }
      if (iree_status_is_ok(status)) {
        start_header = context.CapturedSendHeader(send_index);
      }
      if (iree_status_is_ok(status) &&
          start_header.type != IREE_NET_BULK_FRAME_TYPE_START) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "benchmark expected profile START frame");
      }
      if (iree_status_is_ok(status)) {
        status = context.GrantChunkCredit(/*credit_delta=*/1);
      }
      if (iree_status_is_ok(status)) {
        status =
            context.CompleteNextCapturedSend(send_index++, iree_ok_status());
      }
      if (iree_status_is_ok(status)) {
        status =
            context.CompleteNextCapturedSend(send_index++, iree_ok_status());
      }
      if (iree_status_is_ok(status)) {
        status =
            context.CompleteNextCapturedSend(send_index++, iree_ok_status());
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_remote_server_bulk_on_complete(
            context.session(), start_header.transfer_id);
      }
    }
    if (iree_status_is_ok(status)) {
      context.ResetCapturedSends();
    }
  }
  return status;
}

IREE_BENCHMARK_REGISTER(BM_DrainBurstDelayedCompletions);

}  // namespace
