// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/send_window.h"

#include <string.h>

#define IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_VALID_MASK \
  IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT

static bool iree_net_rdma_send_window_flags_are_valid(
    iree_net_rdma_send_window_acquire_flags_t flags) {
  return (flags & ~IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_VALID_MASK) == 0;
}

static bool iree_net_rdma_send_window_uses_remote_recv_credit(
    iree_net_rdma_send_window_acquire_flags_t flags) {
  return iree_any_bit_set(
      flags, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT);
}

static bool iree_net_rdma_send_window_credit_limit_is_newer(
    uint32_t current_limit, uint32_t new_limit) {
  uint32_t distance = new_limit - current_limit;
  return distance < 0x80000000u;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_window_initialize(
    uint32_t send_queue_depth, uint32_t initial_remote_recv_credits,
    iree_net_rdma_send_window_t* out_window) {
  if (!out_window) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_window must not be NULL");
  }
  memset(out_window, 0, sizeof(*out_window));
  if (send_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send_queue_depth must be non-zero");
  }
  out_window->send_queue_depth = send_queue_depth;
  out_window->remote_recv_credit_limit = initial_remote_recv_credits;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_net_rdma_send_window_refresh_remote_credits(
    iree_net_rdma_send_window_t* window, uint32_t remote_recv_credit_limit) {
  if (!window) return;
  if (iree_net_rdma_send_window_credit_limit_is_newer(
          window->remote_recv_credit_limit, remote_recv_credit_limit)) {
    window->remote_recv_credit_limit = remote_recv_credit_limit;
  }
}

IREE_API_EXPORT uint32_t iree_net_rdma_send_window_available_send_slots(
    const iree_net_rdma_send_window_t* window) {
  if (!window || window->posted_send_count >= window->send_queue_depth) {
    return 0;
  }
  return window->send_queue_depth - window->posted_send_count;
}

IREE_API_EXPORT uint32_t iree_net_rdma_send_window_available_recv_credits(
    const iree_net_rdma_send_window_t* window) {
  if (!window) return 0;
  return window->remote_recv_credit_limit - window->remote_recv_credit_used;
}

IREE_API_EXPORT uint32_t iree_net_rdma_send_window_available(
    const iree_net_rdma_send_window_t* window,
    iree_net_rdma_send_window_acquire_flags_t flags) {
  uint32_t available_send_slots =
      iree_net_rdma_send_window_available_send_slots(window);
  if (!iree_net_rdma_send_window_flags_are_valid(flags)) return 0;
  if (!iree_net_rdma_send_window_uses_remote_recv_credit(flags)) {
    return available_send_slots;
  }
  uint32_t available_recv_credits =
      iree_net_rdma_send_window_available_recv_credits(window);
  return available_send_slots < available_recv_credits ? available_send_slots
                                                       : available_recv_credits;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_window_acquire(
    iree_net_rdma_send_window_t* window,
    iree_net_rdma_send_window_acquire_flags_t flags) {
  if (!window || window->send_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "window must be initialized");
  }
  if (!iree_net_rdma_send_window_flags_are_valid(flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send window flags 0x%02X are invalid", flags);
  }
  if (iree_net_rdma_send_window_available_send_slots(window) == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "RDMA send queue is full");
  }
  if (iree_net_rdma_send_window_uses_remote_recv_credit(flags) &&
      iree_net_rdma_send_window_available_recv_credits(window) == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "RDMA remote receive credits are exhausted");
  }

  ++window->posted_send_count;
  if (iree_net_rdma_send_window_uses_remote_recv_credit(flags)) {
    ++window->remote_recv_credit_used;
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_window_abort(
    iree_net_rdma_send_window_t* window,
    iree_net_rdma_send_window_acquire_flags_t flags) {
  if (!window || window->send_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "window must be initialized");
  }
  if (!iree_net_rdma_send_window_flags_are_valid(flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send window flags 0x%02X are invalid", flags);
  }
  if (window->posted_send_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no posted send admission to abort");
  }
  if (iree_net_rdma_send_window_uses_remote_recv_credit(flags) &&
      window->remote_recv_credit_used == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no remote receive credit to restore");
  }

  --window->posted_send_count;
  if (iree_net_rdma_send_window_uses_remote_recv_credit(flags)) {
    --window->remote_recv_credit_used;
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_send_window_complete(iree_net_rdma_send_window_t* window) {
  if (!window || window->send_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "window must be initialized");
  }
  if (window->posted_send_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no posted send to complete");
  }
  --window->posted_send_count;
  return iree_ok_status();
}
