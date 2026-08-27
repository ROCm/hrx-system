// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DISPATCH_H_

#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// True when the dispatch is a static empty grid. Callers treat this as an
// ordered no-op: wait/signal (or later recorded commands) still run in order,
// but no NPU work is submitted.
static inline bool iree_hal_amdxdna_dispatch_is_zero_workgroups(
    const iree_hal_dispatch_config_t* config,
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) return false;
  return (config->workgroup_count[0] | config->workgroup_count[1] |
          config->workgroup_count[2]) == 0;
}

// HOST-ONLY / static NPU kernels are 1x1x1. Zero workgroup counts are allowed
// as ordered no-ops. Indirect grids, extra workgroups, and local memory fail
// closed instead of silently dropping dimensions.
static inline iree_status_t iree_hal_amdxdna_validate_dispatch(
    const iree_hal_dispatch_config_t* config,
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_parameters(flags) ||
      iree_hal_dispatch_uses_indirect_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna supports only static 1x1x1 dispatch");
  }
  if (config->dynamic_workgroup_local_memory != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna does not support workgroup local memory");
  }
  for (int i = 0; i < 3; ++i) {
    if (config->workgroup_size[i] != 0 && config->workgroup_size[i] != 1) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna supports only the default or 1x1x1 workgroup size");
    }
  }
  if (iree_hal_amdxdna_dispatch_is_zero_workgroups(config, flags)) {
    return iree_ok_status();
  }
  if (config->workgroup_count[0] != 1 || config->workgroup_count[1] != 1 ||
      config->workgroup_count[2] != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "amdxdna supports only 1x1x1 workgroup counts");
  }
  return iree_ok_status();
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DISPATCH_H_
