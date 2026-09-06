// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/channel.h"

#include <stddef.h>

#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// iree_hal_channel_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(channel, method_name) \
  IREE_HAL_VTABLE_DISPATCH(channel, iree_hal_channel, method_name)

IREE_HAL_API_RETAIN_RELEASE(channel);

static iree_status_t iree_hal_channel_validate_queue_family_affinity(
    iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity) {
  if (iree_hal_queue_family_affinity_is_any(queue_family_affinity)) {
    return iree_ok_status();
  }

  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  if (IREE_UNLIKELY(queue_spec->family_count > IREE_HAL_MAX_QUEUE_FAMILIES)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device queue family count %" PRIhsz
                            " exceeds channel affinity capacity %" PRIhsz,
                            queue_spec->family_count,
                            (iree_host_size_t)IREE_HAL_MAX_QUEUE_FAMILIES);
  }
  const iree_hal_queue_family_affinity_t supported_affinity =
      queue_spec->family_count == IREE_HAL_MAX_QUEUE_FAMILIES
          ? IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY
          : (((iree_hal_queue_family_affinity_t)1 << queue_spec->family_count) -
             1);
  if (IREE_UNLIKELY(
          iree_hal_queue_family_affinity_is_empty(queue_family_affinity) ||
          !iree_all_bits_set(supported_affinity, queue_family_affinity))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "channel queue family affinity 0x%016" PRIx64
        " is not a non-empty subset of device affinity 0x%016" PRIx64,
        queue_family_affinity, supported_affinity);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_channel_create(
    iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_channel_validate_queue_family_affinity(
      device, queue_family_affinity);
  iree_hal_channel_t* channel = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_channel)(
        device, queue_family_affinity, params, &channel);
  }
  if (iree_status_is_ok(status)) {
    *out_channel = channel;
  } else {
    iree_hal_channel_release(channel);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_channel_split(
    iree_hal_channel_t* base_channel, int32_t color, int32_t key,
    iree_hal_channel_flags_t flags, iree_hal_channel_t** out_split_channel) {
  IREE_ASSERT_ARGUMENT(base_channel);
  IREE_ASSERT_ARGUMENT(out_split_channel);
  *out_split_channel = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, color);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, key);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, flags);
  iree_hal_channel_t* split_channel = NULL;
  iree_status_t status = _VTABLE_DISPATCH(base_channel, split)(
      base_channel, color, key, flags, &split_channel);
  if (iree_status_is_ok(status)) {
    *out_split_channel = split_channel;
  } else {
    iree_hal_channel_release(split_channel);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_channel_query_rank_and_count(
    const iree_hal_channel_t* channel, int32_t* out_rank, int32_t* out_count) {
  IREE_ASSERT_ARGUMENT(channel);
  int32_t rank = 0;
  int32_t count = 0;
  _VTABLE_DISPATCH(channel, query_rank_and_count)(channel, &rank, &count);
  if (out_rank) *out_rank = rank;
  if (out_count) *out_count = count;
}

IREE_API_EXPORT int32_t
iree_hal_channel_rank(const iree_hal_channel_t* channel) {
  int32_t rank = 0;
  iree_hal_channel_query_rank_and_count(channel, &rank, NULL);
  return rank;
}

IREE_API_EXPORT int32_t
iree_hal_channel_count(const iree_hal_channel_t* channel) {
  int32_t count = 0;
  iree_hal_channel_query_rank_and_count(channel, NULL, &count);
  return count;
}
