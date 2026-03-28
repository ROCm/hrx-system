// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Synchronous data transfer
//===----------------------------------------------------------------------===//

pyre_status_t pyre_synchronous_h2d(pyre_device_t device, const void* host_src,
                                   pyre_buffer_t dst, size_t dst_offset,
                                   size_t size) {
  if (!device || !host_src || !dst) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  if (dst_offset + size > dst->size) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_RANGE,
                            "transfer exceeds buffer size");
  }
  return pyre_status_from_iree(iree_hal_device_transfer_h2d(
      device->hal_device, host_src, dst->hal_buffer,
      (iree_device_size_t)dst_offset, (iree_device_size_t)size,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
}

pyre_status_t pyre_synchronous_d2h(pyre_device_t device, pyre_buffer_t src,
                                   size_t src_offset, void* host_dst,
                                   size_t size) {
  if (!device || !src || !host_dst) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  if (src_offset + size > src->size) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_RANGE,
                            "transfer exceeds buffer size");
  }
  return pyre_status_from_iree(iree_hal_device_transfer_d2h(
      device->hal_device, src->hal_buffer, (iree_device_size_t)src_offset,
      host_dst, (iree_device_size_t)size,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
}
