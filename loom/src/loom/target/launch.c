// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/launch.h"

#include <inttypes.h>

bool loom_target_workgroup_size_is_empty(
    const loom_target_workgroup_size_t* size) {
  return size->x == 0 && size->y == 0 && size->z == 0;
}

bool loom_target_workgroup_size_is_concrete(
    const loom_target_workgroup_size_t* size) {
  return size->x != 0 && size->y != 0 && size->z != 0;
}

bool loom_target_workgroup_size_is_partial(
    const loom_target_workgroup_size_t* size) {
  return !loom_target_workgroup_size_is_empty(size) &&
         !loom_target_workgroup_size_is_concrete(size);
}

bool loom_target_workgroup_size_flat_product_u32(
    const loom_target_workgroup_size_t* size, uint32_t* out_flat_size) {
  uint64_t flat_size = (uint64_t)size->x * size->y * size->z;
  if (flat_size > UINT32_MAX) {
    *out_flat_size = 0;
    return false;
  }
  *out_flat_size = (uint32_t)flat_size;
  return true;
}

static iree_status_t loom_target_validate_required_workgroup_size(
    const loom_target_workgroup_size_t* limit,
    const loom_target_workgroup_size_t* required) {
  if (limit->x != 0 && required->x > limit->x) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required workgroup x size %" PRIu32
                            " exceeds target limit %" PRIu32,
                            required->x, limit->x);
  }
  if (limit->y != 0 && required->y > limit->y) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required workgroup y size %" PRIu32
                            " exceeds target limit %" PRIu32,
                            required->y, limit->y);
  }
  if (limit->z != 0 && required->z > limit->z) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required workgroup z size %" PRIu32
                            " exceeds target limit %" PRIu32,
                            required->z, limit->z);
  }
  return iree_ok_status();
}

static bool loom_target_mul_u64_overflows(uint64_t lhs, uint64_t rhs,
                                          uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) {
    *out_result = 0;
    return true;
  }
  *out_result = lhs * rhs;
  return false;
}

static iree_status_t loom_target_workgroup_size_flat_product_checked(
    const loom_target_workgroup_size_t* size, uint64_t* out_flat_size) {
  uint64_t flat_size = 1;
  const uint64_t factors[] = {
      size->x,
      size->y,
      size->z,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(factors); ++i) {
    if (loom_target_mul_u64_overflows(flat_size, factors[i], &flat_size)) {
      *out_flat_size = 0;
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL kernel required flat workgroup size overflows uint64");
    }
  }
  *out_flat_size = flat_size;
  return iree_ok_status();
}

iree_status_t loom_target_validate_hal_kernel_launch(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel) {
  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  if (loom_target_workgroup_size_is_partial(required)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel required workgroup size must be all zero or all non-zero");
  }
  const uint32_t max_flat_workgroup_size = snapshot->max_flat_workgroup_size;
  const uint32_t flat_min = hal_kernel->flat_workgroup_size_min;
  const uint32_t flat_max = hal_kernel->flat_workgroup_size_max;
  if ((flat_min == 0) != (flat_max == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel flat workgroup size range must be both zero or both "
        "non-zero");
  }
  if (flat_min != 0) {
    if (flat_min > flat_max) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL kernel flat workgroup size range must be ordered");
    }
    if (max_flat_workgroup_size != 0 && flat_max > max_flat_workgroup_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL kernel flat workgroup size max %" PRIu32
                              " exceeds target limit %" PRIu32,
                              flat_max, max_flat_workgroup_size);
    }
  }
  if (loom_target_workgroup_size_is_empty(required)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_validate_required_workgroup_size(
      &snapshot->max_workgroup_size, required));

  uint64_t flat_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_target_workgroup_size_flat_product_checked(required, &flat_size));
  if (max_flat_workgroup_size != 0 && flat_size > max_flat_workgroup_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required flat workgroup size %" PRIu64
                            " exceeds target limit %" PRIu32,
                            flat_size, max_flat_workgroup_size);
  }
  if (flat_min != 0) {
    if (flat_size < flat_min || flat_size > flat_max) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL kernel required flat workgroup size %" PRIu64
                              " must be inside selected range %" PRIu32
                              "..%" PRIu32,
                              flat_size, flat_min, flat_max);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_target_require_concrete_hal_kernel_launch(
    const loom_target_hal_kernel_abi_t* hal_kernel,
    iree_string_view_t consumer_name) {
  if (loom_target_workgroup_size_is_partial(
          &hal_kernel->required_workgroup_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel required workgroup size must be all zero or all non-zero");
  }
  if (loom_target_workgroup_size_is_concrete(
          &hal_kernel->required_workgroup_size)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "%.*s requires a selected HAL kernel workgroup size",
                          (int)consumer_name.size, consumer_name.data);
}

static bool loom_target_dispatch_workgroup_count_is_concrete(
    const loom_target_dispatch_workgroup_count_t* count) {
  return count->x != 0 && count->y != 0 && count->z != 0;
}

static iree_status_t loom_target_validate_dispatch_workgroup_count_limits(
    const loom_target_workgroup_count_limit_t* limit,
    const loom_target_dispatch_workgroup_count_t* selected) {
  if (limit->x != 0 && selected->x > limit->x) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch workgroup x count %" PRIu32
                            " exceeds target limit %" PRIu32,
                            selected->x, limit->x);
  }
  if (limit->y != 0 && selected->y > limit->y) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch workgroup y count %" PRIu32
                            " exceeds target limit %" PRIu32,
                            selected->y, limit->y);
  }
  if (limit->z != 0 && selected->z > limit->z) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch workgroup z count %" PRIu32
                            " exceeds target limit %" PRIu32,
                            selected->z, limit->z);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_validate_dispatch_grid_size_limits(
    const loom_target_grid_size_t* limit,
    const loom_target_workgroup_size_t* workgroup_size,
    const loom_target_dispatch_workgroup_count_t* workgroup_count) {
  const uint64_t grid_x = (uint64_t)workgroup_size->x * workgroup_count->x;
  if (limit->x != 0 && grid_x > limit->x) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch grid x size %" PRIu64
                            " exceeds target limit %" PRIu32,
                            grid_x, limit->x);
  }
  const uint64_t grid_y = (uint64_t)workgroup_size->y * workgroup_count->y;
  if (limit->y != 0 && grid_y > limit->y) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch grid y size %" PRIu64
                            " exceeds target limit %" PRIu32,
                            grid_y, limit->y);
  }
  const uint64_t grid_z = (uint64_t)workgroup_size->z * workgroup_count->z;
  if (limit->z != 0 && grid_z > limit->z) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch grid z size %" PRIu64
                            " exceeds target limit %" PRIu32,
                            grid_z, limit->z);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_dispatch_flat_grid_size(
    const loom_target_workgroup_size_t* workgroup_size,
    const loom_target_dispatch_workgroup_count_t* workgroup_count,
    uint64_t* out_flat_grid_size) {
  uint64_t flat_grid_size = 1;
  const uint64_t factors[] = {
      workgroup_size->x,  workgroup_count->x, workgroup_size->y,
      workgroup_count->y, workgroup_size->z,  workgroup_count->z,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(factors); ++i) {
    if (loom_target_mul_u64_overflows(flat_grid_size, factors[i],
                                      &flat_grid_size)) {
      *out_flat_grid_size = 0;
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL dispatch flat grid size overflows uint64");
    }
  }
  *out_flat_grid_size = flat_grid_size;
  return iree_ok_status();
}

iree_status_t loom_target_validate_hal_dispatch_workgroup_count(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel,
    const loom_target_dispatch_workgroup_count_t* workgroup_count) {
  if (!loom_target_dispatch_workgroup_count_is_concrete(workgroup_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL dispatch workgroup count must be all non-zero");
  }
  IREE_RETURN_IF_ERROR(loom_target_validate_dispatch_workgroup_count_limits(
      &snapshot->max_workgroup_count, workgroup_count));

  const loom_target_workgroup_size_t minimum_workgroup_size = {
      .x = 1,
      .y = 1,
      .z = 1,
  };
  IREE_RETURN_IF_ERROR(loom_target_validate_dispatch_grid_size_limits(
      &snapshot->max_grid_size, &minimum_workgroup_size, workgroup_count));

  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  if (loom_target_workgroup_size_is_empty(required)) {
    return iree_ok_status();
  }
  if (loom_target_workgroup_size_is_partial(required)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel required workgroup size must be all zero or all non-zero");
  }
  IREE_RETURN_IF_ERROR(loom_target_validate_dispatch_grid_size_limits(
      &snapshot->max_grid_size, required, workgroup_count));

  uint64_t flat_grid_size = 0;
  IREE_RETURN_IF_ERROR(loom_target_dispatch_flat_grid_size(
      required, workgroup_count, &flat_grid_size));
  if (snapshot->max_flat_grid_size != 0 &&
      flat_grid_size > snapshot->max_flat_grid_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch flat grid size %" PRIu64
                            " exceeds target limit %" PRIu64,
                            flat_grid_size, snapshot->max_flat_grid_size);
  }
  return iree_ok_status();
}
