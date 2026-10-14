// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_NET_CHANNEL_PEER_H_
#define IREE_HAL_NET_CHANNEL_PEER_H_

#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/channel.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/session.h"

typedef struct iree_hal_net_channel_transfer_t iree_hal_net_channel_transfer_t;

// Immutable connection role and preparation geometry.
typedef struct iree_hal_net_channel_peer_options_t {
  // Local rank within the owning group.
  int32_t rank;
  // Remote rank within the owning group.
  int32_t peer_rank;
  // Number of slots per directed mode.
  uint32_t tag_capacity;
  // Borrowed local bind address for a listener or remote contact for a client.
  iree_string_view_t address;
} iree_hal_net_channel_peer_options_t;

// Single poll-owner connection state. The mutex only arbitrates admission and
// terminal state with queue submitters; native progress stays on the poll
// owner.
typedef struct iree_hal_net_channel_peer_t {
  // Borrowed rank channel, retained by public transfers.
  iree_hal_channel_t* owner;
  // Allocator owning this object and its prepared transfer table.
  iree_allocator_t host_allocator;
  // Borrowed owner device qualifying local queues and prepared memory.
  iree_hal_device_t* device;
  // Admitted local queue families, independent of peer membership.
  iree_hal_queue_family_affinity_t queue_family_affinity;
  // Borrowed owner progress service.
  iree_async_proactor_t* proactor;
  // Immutable local rank and preparation geometry.
  iree_hal_net_channel_peer_options_t options;
  // Protects admission, operation phases and terminal status.
  iree_slim_mutex_t mutex;
  // First failure and any subsequent cleanup errors, owned until destruction.
  iree_status_t status;
  // Owned prepared slots, indexed by mode * tag_capacity + tag.
  iree_hal_net_channel_transfer_t* transfers;
  // Checked number of slots covering both directed modes.
  iree_host_size_t transfer_count;
  // Accepted operations not yet returned to their submitters.
  iree_host_size_t pending_count;
  // Borrowed session; the rank owner joins its callbacks before destruction.
  iree_net_session_t* session;
  // Protocol adapter borrowing the session's message endpoint.
  iree_net_queue_channel_t* protocol;
  // True once the operational message endpoint is activated.
  bool ready;
  // Admission is closed by failure or explicit deactivation.
  bool closing;
  // Session callbacks have retired or no session has been bound.
  bool session_drained;
  // Native drain has started on the poll owner.
  bool drain_started;
  // Defers the final receipt until a poll-owner service walk has returned.
  bool servicing;
  // All native and dependency callbacks have retired.
  bool drained;
  // A graceful-close record still borrows the session's control endpoint.
  bool goaway_pending;
  // The peer supplied a graceful close record before endpoint retirement.
  bool peer_closed;
  // Owner receipt captured at allocation and borrowed until actual retirement.
  iree_hal_channel_deactivated_callback_t deactivated_callback;
} iree_hal_net_channel_peer_t;

// Allocates prepared-slot storage on the calling poll owner. The owner keeps
// the device, proactor and address storage alive through peer destruction.
iree_status_t iree_hal_net_channel_peer_allocate(
    iree_hal_channel_t* owner, iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_async_proactor_t* proactor,
    const iree_hal_net_channel_peer_options_t* options,
    iree_hal_channel_deactivated_callback_t callback,
    iree_allocator_t host_allocator, iree_hal_net_channel_peer_t** out_peer);

// Opens the operational protocol after validated peer bootstrap.
void iree_hal_net_channel_peer_open(iree_hal_net_channel_peer_t* peer);

// Processes peer closure while retaining outstanding native source borrows.
void iree_hal_net_channel_peer_on_goaway(iree_hal_net_channel_peer_t* peer,
                                         uint32_t reason);

// Frees peer storage after its captured retirement receipt.
void iree_hal_net_channel_peer_free(iree_hal_net_channel_peer_t* peer);

// Poll-owner helpers consume failure and service all independently ready tags.
void iree_hal_net_channel_peer_fail(iree_hal_net_channel_peer_t* peer,
                                    iree_status_t status);
void iree_hal_net_channel_peer_service(iree_hal_net_channel_peer_t* peer);
void iree_hal_net_channel_peer_try_finish(iree_hal_net_channel_peer_t* peer);

#endif  // IREE_HAL_NET_CHANNEL_PEER_H_
