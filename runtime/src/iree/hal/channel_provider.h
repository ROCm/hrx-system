// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CHANNEL_PROVIDER_H_
#define IREE_HAL_CHANNEL_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/channel.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_channel_provider_t
//===----------------------------------------------------------------------===//

// Channel creation and configuration provider.
// Hosting applications can use this to either configure or completely replace
// the default device channel creation logic.
typedef struct iree_hal_channel_provider_t iree_hal_channel_provider_t;

// Retains the given |channel_provider| for the caller.
IREE_API_EXPORT void iree_hal_channel_provider_retain(
    iree_hal_channel_provider_t* channel_provider);

// Releases the given |channel_provider| from the caller.
IREE_API_EXPORT void iree_hal_channel_provider_release(
    iree_hal_channel_provider_t* channel_provider);

// Creates the channel selected by this provider for |device|. The provider
// must be configured before channel creation; replacing it does not change
// existing channels. A NULL provider or a rendezvous-only provider returns
// UNIMPLEMENTED. Device implementations may supply their own native creation
// when they have no replacing provider.
IREE_API_EXPORT iree_status_t iree_hal_channel_provider_create_channel(
    iree_hal_channel_provider_t* channel_provider, iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel);

// Returns the rank the process represents as a participant in the default
// collective group in `[0, count)` and the total participant count.
// May return IREE_HAL_CHANNEL_RANK_DEFAULT/IREE_HAL_CHANNEL_COUNT_DEFAULT if
// the provider cannot service the request.
IREE_API_EXPORT iree_status_t
iree_hal_channel_provider_query_default_rank_and_count(
    iree_hal_channel_provider_t* channel_provider, int32_t* out_rank,
    int32_t* out_count);

// Exchanges the default channel ID used during initial channel configuration.
// The caller will provide adequate storage in |id| and the implementation
// should do what it needs (ID exchange, etc). The default root participant will
// provide an initialized ID and all others will expect that ID to be populated
// upon return.
IREE_API_EXPORT iree_status_t iree_hal_channel_provider_exchange_default_id(
    iree_hal_channel_provider_t* channel_provider, iree_byte_span_t id);

//===----------------------------------------------------------------------===//
// iree_hal_channel_provider_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_channel_provider_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_channel_provider_t* channel_provider);

  iree_status_t(IREE_API_PTR* query_default_rank_and_count)(
      iree_hal_channel_provider_t* channel_provider, int32_t* out_rank,
      int32_t* out_count);

  iree_status_t(IREE_API_PTR* exchange_default_id)(
      iree_hal_channel_provider_t* channel_provider, iree_byte_span_t id);

  // Optional replacement for device-native channel construction. Consumes
  // already validated device queue-family affinity; retains its own services.
  iree_status_t(IREE_API_PTR* create_channel)(
      iree_hal_channel_provider_t* channel_provider, iree_hal_device_t* device,
      iree_hal_queue_family_affinity_t queue_family_affinity,
      iree_hal_channel_params_t params, iree_hal_channel_t** out_channel);
} iree_hal_channel_provider_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_channel_provider_vtable_t);

IREE_API_EXPORT void iree_hal_channel_provider_destroy(
    iree_hal_channel_provider_t* channel_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CHANNEL_PROVIDER_H_
