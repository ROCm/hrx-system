// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/signal.h"

void iree_hal_amd_gpu_aql_signal_block_initialize(
    iree_hal_amd_gpu_signal_pool_t* pool, const void* domain,
    const amdf_gpu_event_info_t* event,
    iree_hal_amd_gpu_aql_signal_t* native_signals, uint64_t device_address,
    iree_hal_amd_gpu_signal_t* signals, uint32_t capacity, uint32_t queue_count,
    uint64_t* frontiers, iree_hal_amd_gpu_signal_block_t* out_block) {
  for (uint32_t i = 0; i < capacity; ++i) {
    native_signals[i] = (iree_hal_amd_gpu_aql_signal_t){
        .kind = 1,
        .event_mailbox = event->mailbox_address,
        .event_id = event->interrupt_context,
    };
  }
  iree_hal_amd_gpu_signal_block_initialize(
      pool, domain, native_signals, device_address,
      sizeof(iree_hal_amd_gpu_aql_signal_t),
      offsetof(iree_hal_amd_gpu_aql_signal_t, value), signals, capacity,
      queue_count, frontiers, out_block);
}
