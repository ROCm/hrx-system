// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/buffer.h"

iree_status_t iree_hal_amd_gpu_buffer_reference_initialize(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_hal_amd_gpu_buffer_reference_t* out_reference) {
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  // Public family validation also accepts legacy buffers, which need not
  // provide the native binding contract required by this driver.
  if (!contract) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "buffer has no native binding contract");
  }
  const uint16_t index =
      contract
          ->scopes[family->memory.queue_scope_id +
                   (site - IREE_HAL_MEMORY_SITE_QUEUE)]
          .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS];
  if (index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "GPU address access is not prepared for this queue");
  }
  *out_reference = (iree_hal_amd_gpu_buffer_reference_t){
      .buffer = buffer,
      .offset = offset,
      .slot = {.index = index,
               .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS},
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_buffer_reference_resolve(
    const iree_hal_amd_gpu_buffer_reference_t* reference,
    uint64_t* out_address) {
  const uint64_t base =
      iree_hal_buffer_native_binding(reference->buffer, reference->slot)
          .device_address;
  if (!base) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer has no committed GPU address");
  }
  *out_address = base + reference->offset;
  return iree_ok_status();
}
