// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_NET_CHANNEL_TRANSFER_H_
#define IREE_HAL_NET_CHANNEL_TRANSFER_H_

#include "iree/async/operations/semaphore.h"
#include "iree/hal/channel_transfer.h"
#include "iree/hal/net/channel_peer.h"
#include "iree/hal/semaphore.h"

enum { IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY = 8 };

typedef enum iree_hal_net_channel_transfer_phase_e {
  IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_IDLE = 0,
  IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_QUEUED,
  IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_WAITING,
  IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_READY,
  IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_ISSUED,
} iree_hal_net_channel_transfer_phase_t;

// Channel-owned prepared state. The HAL reference controls access and retains
// the channel; the slot storage survives until channel destruction.
// This is acyclic ownership: the channel owns storage, not a HAL transfer ref.
struct iree_hal_net_channel_transfer_t {
  // Prepared HAL resource, initialized only when this tag is bound.
  iree_hal_channel_transfer_t base;
  // Stable peer owner, including before the public resource is initialized.
  iree_hal_net_channel_peer_t* peer;
  // Current execution range, retained only through actual transfer retirement.
  iree_hal_buffer_ref_t buffer;
  // Host address resolved after allocation and producer dependencies complete.
  uint8_t* data;
  // Whether preparation has established the immutable storage contract.
  bool prepared;
  // Admission/progress phase, protected by the peer mutex.
  iree_hal_net_channel_transfer_phase_t phase;
  // Last admitted local sequence; advances only on successful submission.
  uint64_t sequence;
  // Exact queue retained through operation retirement.
  iree_hal_queue_t* queue;
  // Captured wait dependencies, retained through the operation's callback.
  struct {
    // Number of active entries.
    iree_host_size_t count;
    // Retained host-observable semaphore interfaces.
    iree_async_semaphore_t* semaphores[IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY];
    // Required values for the active entries.
    uint64_t values[IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY];
  } waits;
  // Captured completion dependencies, retained through signal/failure delivery.
  struct {
    // Number of active entries.
    iree_host_size_t count;
    // Retained HAL semaphore interfaces.
    iree_hal_semaphore_t* semaphores[IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY];
    // Published values on successful completion.
    uint64_t values[IREE_HAL_NET_CHANNEL_SEMAPHORE_CAPACITY];
  } signals;
  // Cross-thread submission rendezvous, reused only after its callback.
  iree_async_nop_operation_t issue_operation;
  // Dependency observer; its callback joins all software timepoint borrowers.
  iree_async_semaphore_wait_operation_t wait_operation;
  // Peer-authorized destination generation and extent for a SEND binding.
  struct {
    // Last received destination grant.
    uint64_t sequence;
    // Exact extent advertised for that generation.
    uint64_t length;
    // Last grant consumed by a successful native send admission.
    uint64_t used_sequence;
  } credit;
  // Credit-message callback remains owned even if payload arrives first.
  bool credit_pending;
  // Target capture owned independently of the credit-message send receipt.
  struct {
    // Stable net receive record, reused after its terminal callback begins.
    iree_net_message_receive_operation_t operation;
    // Poll-owner state: the net endpoint still borrows the current target.
    bool pending;
  } receive;
};

iree_status_t iree_hal_net_channel_prepare_transfer(
    iree_hal_net_channel_peer_t* peer,
    const iree_hal_channel_transfer_params_t* params,
    iree_hal_channel_transfer_t** out_transfer);
void iree_hal_net_channel_transfers_service(iree_hal_net_channel_peer_t* peer);
void iree_hal_net_channel_transfers_abort(iree_hal_net_channel_peer_t* peer);
iree_status_t iree_hal_net_channel_on_command(
    void* user_data, uint32_t queue_id,
    const iree_net_queue_frontier_view_t* waits,
    const iree_net_queue_frontier_view_t* signals,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);
iree_status_t iree_hal_net_channel_on_advance(
    void* user_data, const iree_net_queue_frontier_view_t* signals,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);

#endif  // IREE_HAL_NET_CHANNEL_TRANSFER_H_
