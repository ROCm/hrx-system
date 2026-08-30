// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_library.h"

static int elementwise_mul_dispatch(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  (void)environment;
  (void)workgroup_state;
  if (dispatch_state->binding_count < 3 ||
      dispatch_state->binding_lengths[0] < 4 * sizeof(float) ||
      dispatch_state->binding_lengths[1] < 4 * sizeof(float) ||
      dispatch_state->binding_lengths[2] < 4 * sizeof(float)) {
    return 1;
  }
  const float* lhs = (const float*)dispatch_state->binding_ptrs[0];
  const float* rhs = (const float*)dispatch_state->binding_ptrs[1];
  float* result = (float*)dispatch_state->binding_ptrs[2];
  for (size_t i = 0; i < 4; ++i) {
    result[i] = lhs[i] * rhs[i];
  }
  return 0;
}

static const iree_hal_executable_library_header_t library_header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "elementwise_mul",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

static const iree_hal_executable_dispatch_v0_t dispatch_functions[] = {
    elementwise_mul_dispatch,
};

static const iree_hal_executable_dispatch_attrs_v0_t dispatch_attrs[] = {
    {
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .local_memory_pages = 0,
        .binding_count = 3,
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = 0,
        .constant_byte_length = 0,
    },
};

static const char* dispatch_names[] = {
    "elementwise_mul",
};

static const iree_hal_executable_library_v0_t library = {
    .header = &library_header,
    .exports =
        {
            .count = 1,
            .ptrs = dispatch_functions,
            .attrs = dispatch_attrs,
            .params = NULL,
            .occupancy = NULL,
            .names = dispatch_names,
            .tags = NULL,
            .parameter_names = NULL,
            .source_locations = NULL,
            .stage_locations = NULL,
        },
    .constants =
        {
            .count = 0,
        },
    .sources =
        {
            .count = 0,
            .files = NULL,
        },
};

IREE_HAL_EXECUTABLE_LIBRARY_EXPORT
const iree_hal_executable_library_header_t* const*
iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  (void)environment;
  return max_version >= IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST
             ? &library.header
             : NULL;
}
