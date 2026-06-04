// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <string.h>

#include "hrx_internal.h"

//===----------------------------------------------------------------------===//
// Synchronous data transfer
//===----------------------------------------------------------------------===//

hrx_status_t hrx_synchronous_h2d(hrx_device_t device, const void* host_src,
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
                                 size_t src_offset, void* host_dst,
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

//===----------------------------------------------------------------------===//
// Async stream transfers
//===----------------------------------------------------------------------===//

hrx_status_t hrx_stream_copy_h2d(hrx_stream_t stream, const void* host_src,
                                 hrx_buffer_t dst, size_t dst_offset,
                                 size_t size) {
  if (!host_src || !dst) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  if (!stream) {
    return hrx_synchronous_h2d(dst->device, host_src, dst, dst_offset, size);
  }

  // Flush any pending dispatches before transfer for proper ordering.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) return status;

  // Chunked synchronous H2D (matches streaming layer: 63KB chunks for
  // inline BUFFER_UPDATE path compatibility with remote HAL).
  const size_t chunk_size = 63 * 1024;
  const uint8_t* src_ptr = (const uint8_t*)host_src;
  size_t remaining = size;
  size_t chunk_offset = 0;

  while (remaining > 0) {
    size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;
    iree_status_t iree_status = iree_hal_device_transfer_h2d(
        stream->device->hal_device, src_ptr + chunk_offset, dst->hal_buffer,
        (iree_device_size_t)(dst_offset + chunk_offset),
        (iree_device_size_t)this_chunk, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
    if (!iree_status_is_ok(iree_status)) {
      return hrx_status_from_iree(iree_status);
    }
    chunk_offset += this_chunk;
    remaining -= this_chunk;
  }
  return hrx_ok_status();
}

hrx_status_t hrx_stream_copy_d2h(hrx_stream_t stream, hrx_buffer_t src,
                                 size_t src_offset, void* host_dst,
                                 size_t size) {
  if (!src || !host_dst) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  if (!stream) {
    return hrx_synchronous_d2h(src->device, src, src_offset, host_dst, size);
  }

  // Flush and synchronize stream for D2H (must complete pending writes first).
  hrx_status_t status = hrx_stream_synchronize(stream);
  if (!hrx_status_is_ok(status)) return status;

  // Chunked synchronous D2H (4MB chunks to avoid staging buffer overflows).
  const size_t chunk_size = 4 * 1024 * 1024;
  uint8_t* dst_ptr = (uint8_t*)host_dst;
  size_t remaining = size;
  size_t chunk_offset = 0;

  while (remaining > 0) {
    size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;
    iree_status_t iree_status = iree_hal_device_transfer_d2h(
        stream->device->hal_device, src->hal_buffer,
        (iree_device_size_t)(src_offset + chunk_offset), dst_ptr + chunk_offset,
        (iree_device_size_t)this_chunk, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
    if (!iree_status_is_ok(iree_status)) {
      return hrx_status_from_iree(iree_status);
    }
    chunk_offset += this_chunk;
    remaining -= this_chunk;
  }
  return hrx_ok_status();
}
