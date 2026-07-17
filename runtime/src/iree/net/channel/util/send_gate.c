// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/send_gate.h"

#define IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT ((uint32_t)1u << 31)
#define IREE_NET_CHANNEL_SEND_GATE_FINISHED_BIT ((uint32_t)1u << 30)
#define IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK \
  (IREE_NET_CHANNEL_SEND_GATE_FINISHED_BIT - 1u)

static bool iree_net_channel_send_gate_is_quiescent(void* user_data) {
  iree_net_channel_send_gate_t* gate = (iree_net_channel_send_gate_t*)user_data;
  uint32_t state = iree_atomic_load(&gate->state, iree_memory_order_acquire);
  return (state & IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK) == 0;
}

static bool iree_net_channel_send_gate_is_closed(void* user_data) {
  iree_net_channel_send_gate_t* gate = (iree_net_channel_send_gate_t*)user_data;
  uint32_t state = iree_atomic_load(&gate->state, iree_memory_order_acquire);
  return (state & IREE_NET_CHANNEL_SEND_GATE_FINISHED_BIT) != 0;
}

void iree_net_channel_send_gate_initialize(
    iree_net_channel_send_gate_t* out_gate) {
  iree_atomic_store(&out_gate->state, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&out_gate->notification);
}

void iree_net_channel_send_gate_deinitialize(
    iree_net_channel_send_gate_t* gate) {
  uint32_t state = iree_atomic_load(&gate->state, iree_memory_order_acquire);
  IREE_ASSERT_EQ(state & IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK, 0u,
                 "send gate deinitialized with admitted sends");
  IREE_ASSERT((state & IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT) == 0 ||
                  (state & IREE_NET_CHANNEL_SEND_GATE_FINISHED_BIT) != 0,
              "send gate deinitialized while close is in progress");
  iree_notification_deinitialize(&gate->notification);
}

bool iree_net_channel_send_gate_try_enter(iree_net_channel_send_gate_t* gate) {
  uint32_t state = iree_atomic_load(&gate->state, iree_memory_order_acquire);
  while ((state & IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT) == 0) {
    if (IREE_UNLIKELY((state & IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK) ==
                      IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK)) {
      iree_abort();
    }
    uint32_t desired_state = state + 1;
    if (iree_atomic_compare_exchange_weak(&gate->state, &state, desired_state,
                                          iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

void iree_net_channel_send_gate_leave(iree_net_channel_send_gate_t* gate) {
  uint32_t previous_state =
      iree_atomic_fetch_sub(&gate->state, 1, iree_memory_order_acq_rel);
  IREE_ASSERT_GT(previous_state & IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK, 0u,
                 "send gate leave without admission");
  if ((previous_state & IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT) != 0 &&
      (previous_state & IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK) == 1) {
    iree_notification_post(&gate->notification, IREE_ALL_WAITERS);
  }
}

bool iree_net_channel_send_gate_begin_close(
    iree_net_channel_send_gate_t* gate) {
  uint32_t previous_state =
      iree_atomic_fetch_or(&gate->state, IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT,
                           iree_memory_order_acq_rel);
  return (previous_state & IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT) == 0;
}

void iree_net_channel_send_gate_await_quiescence(
    iree_net_channel_send_gate_t* gate) {
  IREE_ASSERT((iree_atomic_load(&gate->state, iree_memory_order_acquire) &
               IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT) != 0,
              "send gate quiescence awaited before close");
  (void)iree_notification_await(&gate->notification,
                                iree_net_channel_send_gate_is_quiescent, gate,
                                iree_infinite_timeout());
}

void iree_net_channel_send_gate_finish_close(
    iree_net_channel_send_gate_t* gate) {
  uint32_t previous_state = iree_atomic_fetch_or(
      &gate->state, IREE_NET_CHANNEL_SEND_GATE_FINISHED_BIT,
      iree_memory_order_release);
  IREE_ASSERT((previous_state & IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT) != 0,
              "send gate close finished before it began");
  IREE_ASSERT_EQ(previous_state & IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK, 0u,
                 "send gate close finished before quiescence");
  IREE_ASSERT((previous_state & IREE_NET_CHANNEL_SEND_GATE_FINISHED_BIT) == 0,
              "send gate close finished more than once");
  iree_notification_post(&gate->notification, IREE_ALL_WAITERS);
}

void iree_net_channel_send_gate_await_closed(
    iree_net_channel_send_gate_t* gate) {
  IREE_ASSERT((iree_atomic_load(&gate->state, iree_memory_order_acquire) &
               IREE_NET_CHANNEL_SEND_GATE_CLOSED_BIT) != 0,
              "send gate close awaited before it began");
  (void)iree_notification_await(&gate->notification,
                                iree_net_channel_send_gate_is_closed, gate,
                                iree_infinite_timeout());
}

uint32_t iree_net_channel_send_gate_pending_count(
    const iree_net_channel_send_gate_t* gate) {
  uint32_t state = iree_atomic_load(
      &((iree_net_channel_send_gate_t*)gate)->state, iree_memory_order_acquire);
  return state & IREE_NET_CHANNEL_SEND_GATE_COUNT_MASK;
}
