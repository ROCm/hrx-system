// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/barrier.h"

#include "iree/hal/command_buffer.h"

// The XDNA queue family has no native cache-control role. Its qualified memory
// pairs place required host and peer cache operations at those owners, while
// XDNA command execution boundaries already provide global system visibility
// for device DMA. Global and local access dependencies therefore add no native
// command. Ranged effects require an exact queue operation that this family
// cannot encode and cannot promote without weakening the prepared contract.
static iree_status_t iree_hal_amd_xdna_barrier_list_validate(
    const iree_hal_barrier_list_t* barriers) {
  if (!barriers) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < barriers->count; ++i) {
    if (iree_hal_memory_effects_requires_resources(
            barriers->values[i].effects)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "XDNA queues do not support ranged memory transitions");
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_queue_barriers_validate(
    const iree_hal_queue_barriers_t* barriers) {
  if (!barriers) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_barrier_list_validate(barriers->before));
  return iree_hal_amd_xdna_barrier_list_validate(barriers->after);
}
