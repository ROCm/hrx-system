// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/relocation.h"

iree_status_t loom_serve_block_region_relocate(
    loom_serve_execution_t* execution, loom_serve_virtual_buffer_t* reservation,
    iree_hal_buffer_t* buffer, const loom_serve_block_region_t* region,
    uint32_t block_count, const uint32_t* destinations,
    uint64_t* out_copied_bytes) {
  *out_copied_bytes = 0;
  // Maintenance metadata stays bounded independently of the model's plane
  // and block counts. Successive batches have queue edges, not host waits.
  iree_hal_transfer_operation_t copies[64];
  iree_host_size_t copy_count = 0;
  uint64_t completion = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t plane = 0;
       plane < region->count && iree_status_is_ok(status); ++plane) {
    const iree_device_size_t origin = region->origin + plane * region->stride;
    for (uint32_t source = 0; source < block_count && iree_status_is_ok(status);
         ++source) {
      const uint32_t target = destinations[source];
      if (target == UINT32_MAX || target == source) {
        continue;
      }
      const iree_device_size_t target_offset =
          origin + target * region->block_bytes;
      status = loom_serve_virtual_buffer_commit(reservation, target_offset,
                                                region->block_bytes);
      if (iree_status_is_ok(status)) {
        copies[copy_count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
            .copy = {.source_buffer = buffer,
                     .source_offset = origin + source * region->block_bytes,
                     .target_buffer = buffer,
                     .target_offset = target_offset,
                     .length = region->block_bytes}};
      }
      if (iree_status_is_ok(status) && copy_count == IREE_ARRAYSIZE(copies)) {
        status = loom_serve_execution_transfer(execution, copy_count, copies,
                                               &completion);
        if (iree_status_is_ok(status)) {
          *out_copied_bytes += copy_count * region->block_bytes;
        }
        copy_count = 0;
      }
    }
  }
  if (iree_status_is_ok(status) && copy_count) {
    status = loom_serve_execution_transfer(execution, copy_count, copies,
                                           &completion);
    if (iree_status_is_ok(status)) {
      *out_copied_bytes += copy_count * region->block_bytes;
    }
  }
  return status;
}
