// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/profile_relay.h"

#include <cstdint>
#include <cstring>

#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/net/channel/util/sequence_window.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

class ProfileRelayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&server_, 0, sizeof(server_));
    memset(&session_, 0, sizeof(session_));
    server_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&server_.session_mutex);
    session_.server = &server_;
    session_.session_id = 1;
    IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_create(
        &session_, /*options=*/nullptr, iree_allocator_system(),
        &session_.bulk_session));
  }

  void TearDown() override {
    iree_hal_remote_server_bulk_session_free(session_.bulk_session);
    session_.bulk_session = nullptr;
    iree_slim_mutex_deinitialize(&server_.session_mutex);
  }

  iree_hal_remote_server_profile_relay_t* profile_relay() {
    return iree_hal_remote_server_bulk_session_profile_relay(&session_);
  }

  iree_hal_remote_server_t server_;
  iree_hal_remote_server_session_t session_;
};

TEST_F(ProfileRelayTest, ObserveRejectsZeroSequence) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_remote_server_profile_relay_observe_transfer(
                            &session_, /*sequence=*/0, iree_ok_status()));
}

TEST_F(ProfileRelayTest, EarliestFailureWins) {
  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_observe_transfer(
      &session_, /*sequence=*/5,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "sequence 5 failed")));
  EXPECT_EQ(profile_relay()->transfer_failure_sequence, 5u);
  EXPECT_EQ(profile_relay()->transfer_failure_code, IREE_STATUS_UNAVAILABLE);

  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_observe_transfer(
      &session_, /*sequence=*/3,
      iree_make_status(IREE_STATUS_ABORTED, "sequence 3 failed")));
  EXPECT_EQ(profile_relay()->transfer_failure_sequence, 3u);
  EXPECT_EQ(profile_relay()->transfer_failure_code, IREE_STATUS_ABORTED);

  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_observe_transfer(
      &session_, /*sequence=*/7,
      iree_make_status(IREE_STATUS_CANCELLED, "sequence 7 failed")));
  EXPECT_EQ(profile_relay()->transfer_failure_sequence, 3u);
  EXPECT_EQ(profile_relay()->transfer_failure_code, IREE_STATUS_ABORTED);
}

TEST_F(ProfileRelayTest, DeferredResponseWaitsForContiguousObservation) {
  iree_hal_remote_control_envelope_t envelope;
  memset(&envelope, 0, sizeof(envelope));
  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_defer_response(
      &session_, &envelope, /*target_sequence=*/3, iree_ok_status()));
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            0u);

  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_observe_transfer(
      &session_, /*sequence=*/2, iree_ok_status()));
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            0u);

  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_observe_transfer(
      &session_, /*sequence=*/1, iree_ok_status()));
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            2u);

  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_observe_transfer(
      &session_, /*sequence=*/3, iree_ok_status()));
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            3u);

  iree_net_sequence_node_t* pending_list = NULL;
  iree_net_sequence_window_take_pending(&profile_relay()->ack_window,
                                        &pending_list);
  EXPECT_EQ(pending_list, nullptr);
}

TEST_F(ProfileRelayTest, PrepareBeginResetsFailureAndAckWindow) {
  IREE_EXPECT_OK(iree_hal_remote_server_profile_relay_observe_transfer(
      &session_, /*sequence=*/3,
      iree_make_status(IREE_STATUS_ABORTED, "prior session failed")));
  EXPECT_TRUE(iree_net_sequence_window_has_observed(
      &profile_relay()->ack_window, /*sequence=*/3));
  EXPECT_EQ(profile_relay()->transfer_failure_sequence, 3u);

  int destroy_count = 0;
  iree_hal_profile_sink_t* sink = NULL;
  IREE_ASSERT_OK(
      test_profile_sink_create(iree_allocator_system(), &destroy_count, &sink));

  session_.session = reinterpret_cast<iree_net_session_t*>(this);
  IREE_EXPECT_OK(
      iree_hal_remote_server_profile_relay_prepare_begin(&session_, sink));
  EXPECT_EQ(profile_relay()->active_sink, sink);
  EXPECT_EQ(profile_relay()->transfer_failure_sequence, 0u);
  EXPECT_EQ(profile_relay()->transfer_failure_code, IREE_STATUS_OK);
  EXPECT_FALSE(iree_net_sequence_window_has_observed(
      &profile_relay()->ack_window, /*sequence=*/3));
  EXPECT_EQ(iree_net_sequence_window_observed(&profile_relay()->ack_window),
            0u);

  iree_hal_profile_sink_t* detached_sink =
      iree_hal_remote_server_profile_relay_detach_active_sink(&session_, sink);
  EXPECT_EQ(detached_sink, sink);
  iree_hal_profile_sink_release(detached_sink);
  iree_hal_profile_sink_release(sink);
  EXPECT_EQ(destroy_count, 1);
  session_.session = NULL;
}

}  // namespace
