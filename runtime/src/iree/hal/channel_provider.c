// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/channel_provider.h"

#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// iree_hal_channel_provider_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(channel_provider, method_name)                 \
  IREE_HAL_VTABLE_DISPATCH(channel_provider, iree_hal_channel_provider, \
                           method_name)

IREE_HAL_API_RETAIN_RELEASE(channel_provider);

IREE_API_EXPORT iree_status_t iree_hal_channel_provider_create_channel(
    iree_hal_channel_provider_t* channel_provider, iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;
  if (!channel_provider ||
      !_VTABLE_DISPATCH(channel_provider, create_channel)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no channel construction provider is installed");
  }
  return _VTABLE_DISPATCH(channel_provider, create_channel)(
      channel_provider, device, queue_family_affinity, params, out_channel);
}

IREE_API_EXPORT iree_status_t
iree_hal_channel_provider_query_default_rank_and_count(
    iree_hal_channel_provider_t* channel_provider, int32_t* out_rank,
    int32_t* out_count) {
  IREE_ASSERT_ARGUMENT(channel_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_rank) {
    *out_rank = IREE_HAL_CHANNEL_RANK_DEFAULT;
  }
  if (out_count) {
    *out_count = IREE_HAL_CHANNEL_COUNT_DEFAULT;
  }
  iree_status_t status =
      _VTABLE_DISPATCH(channel_provider, query_default_rank_and_count)(
          channel_provider, out_rank, out_count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_channel_provider_exchange_default_id(
    iree_hal_channel_provider_t* channel_provider, iree_byte_span_t id) {
  IREE_ASSERT_ARGUMENT(channel_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(
      channel_provider, exchange_default_id)(channel_provider, id);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
