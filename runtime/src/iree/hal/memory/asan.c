// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/asan.h"

static iree_status_t iree_hal_asan_validate_required_power_of_two(
    const char* name, iree_device_size_t value) {
  if (value == 0 || !iree_device_size_is_power_of_two(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s must be a power of two and > 0", name);
  }
  return iree_ok_status();
}

IREE_API_EXPORT bool iree_hal_asan_pool_options_is_enabled(
    const iree_hal_asan_pool_options_t* options) {
  return options && options->mode != IREE_HAL_ASAN_POOL_MODE_DISABLED;
}

IREE_API_EXPORT iree_status_t iree_hal_asan_pool_options_validate(
    const iree_hal_asan_pool_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);

  switch (options->mode) {
    case IREE_HAL_ASAN_POOL_MODE_DISABLED:
      return iree_ok_status();
    case IREE_HAL_ASAN_POOL_MODE_SHADOW:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported HAL ASAN pool mode %d",
                              (int)options->mode);
  }

  IREE_RETURN_IF_ERROR(iree_hal_asan_validate_required_power_of_two(
      "shadow_granule_size", options->shadow_granule_size));
  if (options->redzone_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "redzone_size must be > 0");
  }
  if (options->backing_alignment != 0 &&
      !iree_device_size_is_power_of_two(options->backing_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "backing_alignment must be a power of two or 0");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_asan_calculate_allocation_layout(
    const iree_hal_asan_pool_options_t* options, iree_device_size_t user_length,
    iree_device_size_t user_alignment,
    iree_hal_asan_allocation_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  IREE_RETURN_IF_ERROR(iree_hal_asan_pool_options_validate(options));
  if (!iree_hal_asan_pool_options_is_enabled(options)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "ASAN allocation layout requires enabled options");
  }
  if (user_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "user_length must be > 0");
  }
  IREE_RETURN_IF_ERROR(iree_hal_asan_validate_required_power_of_two(
      "user_alignment", user_alignment));

  const iree_device_size_t shadow_granule_size = options->shadow_granule_size;
  iree_device_size_t backing_length_alignment = options->backing_alignment
                                                    ? options->backing_alignment
                                                    : shadow_granule_size;
  backing_length_alignment =
      iree_max(backing_length_alignment, shadow_granule_size);
  iree_device_size_t backing_offset_alignment =
      iree_max(backing_length_alignment, user_alignment);

  iree_device_size_t left_redzone_length = 0;
  if (!iree_device_size_checked_align(options->redzone_size,
                                      backing_offset_alignment,
                                      &left_redzone_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ASAN left redzone overflow aligning %" PRIu64
                            " bytes to a %" PRIu64 "-byte backing alignment",
                            (uint64_t)options->redzone_size,
                            (uint64_t)backing_offset_alignment);
  }

  iree_device_size_t backing_length = 0;
  if (!iree_device_size_checked_add(left_redzone_length, user_length,
                                    &backing_length) ||
      !iree_device_size_checked_add(backing_length, options->redzone_size,
                                    &backing_length) ||
      !iree_device_size_checked_align(backing_length, backing_length_alignment,
                                      &backing_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ASAN backing length overflow for %" PRIu64
                            " user bytes and %" PRIu64 "-byte redzones",
                            (uint64_t)user_length,
                            (uint64_t)options->redzone_size);
  }

  out_layout->backing_offset_alignment = backing_offset_alignment;
  out_layout->backing_length_alignment = backing_length_alignment;
  out_layout->backing_length = backing_length;
  out_layout->user_offset = left_redzone_length;
  out_layout->user_length = user_length;
  out_layout->left_redzone_length = left_redzone_length;
  out_layout->right_redzone_length =
      backing_length - left_redzone_length - user_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_asan_extend_allocation_layout(
    iree_device_size_t backing_length,
    iree_hal_asan_allocation_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(layout);
  if (backing_length < layout->backing_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "extended ASAN backing length %" PRIu64
                            " is smaller than current backing length %" PRIu64,
                            (uint64_t)backing_length,
                            (uint64_t)layout->backing_length);
  }
  if (!iree_device_size_has_alignment(backing_length,
                                      layout->backing_length_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "extended ASAN backing length %" PRIu64 " does not satisfy %" PRIu64
        "-byte backing length alignment",
        (uint64_t)backing_length, (uint64_t)layout->backing_length_alignment);
  }
  if (layout->user_offset > backing_length ||
      layout->user_length > backing_length - layout->user_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "extended ASAN backing length %" PRIu64
        " cannot contain user range at offset %" PRIu64 " with length %" PRIu64,
        (uint64_t)backing_length, (uint64_t)layout->user_offset,
        (uint64_t)layout->user_length);
  }
  layout->backing_length = backing_length;
  layout->right_redzone_length =
      backing_length - layout->user_offset - layout->user_length;
  return iree_ok_status();
}
