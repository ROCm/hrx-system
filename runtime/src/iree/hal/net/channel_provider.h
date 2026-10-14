// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_NET_CHANNEL_PROVIDER_H_
#define IREE_HAL_NET_CHANNEL_PROVIDER_H_

#include "iree/async/proactor.h"
#include "iree/hal/channel_provider.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Explicit rank membership and bounded preparation policy. Payloads travel
// directly between peers. Each rank owns its buffers and progress service;
// a coordinator supplies contacts and group identity during setup.
typedef struct iree_hal_net_channel_options_t {
  // Local rank in [0, count).
  int32_t rank;
  // Number of participants in the group.
  int32_t count;
  // Local listening address, copied during provider creation.
  iree_string_view_t address;
  // Contacts indexed by peer rank, with exactly |rank| entries. Higher ranks
  // initiate connections to lower ranks during setup. A launcher may
  // create ranks in order, querying each bound address for subsequent ranks.
  // All strings are copied during provider creation.
  iree_string_view_list_t peer_addresses;
  // Number of prepared tags per peer and direction, in [0, tag_capacity).
  uint32_t tag_capacity;
} iree_hal_net_channel_options_t;

// Default rank zero of two, ephemeral loopback listener, and 64 tags per peer
// and direction. Rank zero requires no peer contacts.
IREE_API_EXPORT iree_hal_net_channel_options_t
iree_hal_net_channel_options_default(void);

// Creates a provider retaining |factory| and |proactor|. The caller drives the
// proactor until every channel's deactivation receipt; no thread is created.
// Factory connections must supply reliable ordered message endpoints. Channel
// creation/preparation and address queries occur on the poll owner; queue
// submissions and deactivation may come from other threads.
//
// Transfers qualify a pool's coherent persistent host access during preparation
// without allocating or retaining payload ranges. Queue exchanges capture
// current ranges with that exact contract and resolve their prepared bindings
// after the allocation/producer dependencies complete. No mapping or native
// registration occurs during queue execution;
// their wait/signal edges must permit host access and payload publication.
// There is one admitted exchange per transfer and up to eight wait/signal
// semaphores per exchange. Transport credit stalls do not occupy other tags.
// A binary membership tree is connected at construction so peer failure and
// departure propagate independently of payload preparation. Extra direct peer
// connections are created during preparation. Peer departure closes this
// group's admission; independent groups have independent failure domains.
// Connection failures are asynchronous channel failures, including setup
// rejection after the local listener has been created. The launch owner cancels
// the group if a participant never joins. Membership reserves no payload
// storage.
IREE_API_EXPORT iree_status_t iree_hal_net_channel_provider_create(
    iree_net_transport_factory_t* factory, iree_async_proactor_t* proactor,
    const iree_hal_net_channel_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_channel_provider_t** out_provider);

// Returns the actual bound address of a listening channel from this provider.
// Storage and view ownership follow net listener queries. Valid before polling.
IREE_API_EXPORT iree_status_t iree_hal_net_channel_query_address(
    iree_hal_channel_t* channel, iree_host_size_t capacity, char* storage,
    iree_string_view_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_NET_CHANNEL_PROVIDER_H_
