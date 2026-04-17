// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Synchronous data transfer
//===----------------------------------------------------------------------===//

hrx_status_t hrx_synchronous_h2d(hrx_device_t device, const void *host_src,
                                 hrx_buffer_t dst, size_t dst_offset,
                                 size_t size) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_synchronous_h2d");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!device || !host_src || !dst) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument"));
  }
  if (dst_offset + size > dst->size) {
    HRX_RETURN_AND_END_ZONE(z0,
                            hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                            "transfer exceeds buffer size"));
  }
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_status_from_iree(iree_hal_device_transfer_h2d(
              device->hal_device, host_src, dst->hal_buffer,
              (iree_device_size_t)dst_offset, (iree_device_size_t)size,
              IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout())));
}

hrx_status_t hrx_synchronous_d2h(hrx_device_t device, hrx_buffer_t src,
                                 size_t src_offset, void *host_dst,
                                 size_t size) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_synchronous_d2h");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!device || !src || !host_dst) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument"));
  }
  if (src_offset + size > src->size) {
    HRX_RETURN_AND_END_ZONE(z0,
                            hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                            "transfer exceeds buffer size"));
  }
  HRX_RETURN_AND_END_ZONE(
      z0,
      hrx_status_from_iree(iree_hal_device_transfer_d2h(
          device->hal_device, src->hal_buffer, (iree_device_size_t)src_offset,
          host_dst, (iree_device_size_t)size,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout())));
}
