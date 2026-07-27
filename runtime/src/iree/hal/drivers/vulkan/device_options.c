// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_options.h"

#include <string.h>

IREE_API_EXPORT void iree_hal_vulkan_device_options_initialize(
    iree_hal_vulkan_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->max_cached_bda_replay_instances = 16;
  out_options->max_cached_bda_replay_publication_bytes =
      64ull * 1024ull * 1024ull;
  out_options->retained_cached_bda_replay_instances = 1;
}

iree_status_t iree_hal_vulkan_device_options_verify(
    const iree_hal_vulkan_device_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);

  const iree_hal_vulkan_device_flags_t recognized_device_flags =
      IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE;
  if (iree_any_bit_set(options->flags, ~recognized_device_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unrecognized Vulkan device option flag bits 0x%08x",
        options->flags & ~recognized_device_flags);
  }
  if (options->max_cached_bda_replay_instances != 0 &&
      options->retained_cached_bda_replay_instances >
          options->max_cached_bda_replay_instances) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan retained cached BDA replay instances (%u) must be <= maximum "
        "cached instances (%u)",
        options->retained_cached_bda_replay_instances,
        options->max_cached_bda_replay_instances);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_device_options_parse(
    iree_hal_vulkan_device_options_t* options, iree_string_pair_list_t params) {
  IREE_ASSERT_ARGUMENT(options);
  if (params.count != 0 && !params.pairs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device option list has count %" PRIhsz
                            " but no value storage",
                            params.count);
  }
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    const iree_string_pair_t* param = &params.pairs[i];
    if (iree_string_view_equal(param->key,
                               IREE_SV("cached_bda_replay_instances")) ||
        iree_string_view_equal(param->key,
                               IREE_SV("vulkan_cached_bda_replay_instances"))) {
      if (!iree_string_view_atoi_uint32(
              param->value, &options->max_cached_bda_replay_instances)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid Vulkan cached BDA replay instance count '%.*s'",
            (int)param->value.size, param->value.data);
      }
    } else if (iree_string_view_equal(
                   param->key,
                   IREE_SV("cached_bda_replay_publication_bytes")) ||
               iree_string_view_equal(
                   param->key,
                   IREE_SV("vulkan_cached_bda_replay_publication_bytes"))) {
      if (!iree_string_view_atoi_uint64(
              param->value,
              &options->max_cached_bda_replay_publication_bytes)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid Vulkan cached BDA replay publication byte limit '%.*s'",
            (int)param->value.size, param->value.data);
      }
    } else if (iree_string_view_equal(
                   param->key,
                   IREE_SV("retained_cached_bda_replay_instances")) ||
               iree_string_view_equal(
                   param->key,
                   IREE_SV("vulkan_retained_cached_bda_replay_instances"))) {
      if (!iree_string_view_atoi_uint32(
              param->value, &options->retained_cached_bda_replay_instances)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid Vulkan retained cached BDA replay instance count '%.*s'",
            (int)param->value.size, param->value.data);
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown Vulkan logical device option '%.*s'",
                              (int)param->key.size, param->key.data);
    }
  }
  return iree_hal_vulkan_device_options_verify(options);
}
