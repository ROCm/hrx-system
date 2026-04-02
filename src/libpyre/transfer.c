// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <string.h>

// Forward declaration of stream_begin_cb (defined in stream.c, internal).
static pyre_status_t pyre_transfer_begin_cb(pyre_stream_t stream) {
  if (stream->pending_cb) return pyre_ok_status();
  iree_status_t status = iree_hal_command_buffer_create(
      stream->device->hal_device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER |
          IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      IREE_HAL_QUEUE_AFFINITY_ANY, 0, &stream->pending_cb);
  if (!iree_status_is_ok(status)) return pyre_status_from_iree(status);
  status = iree_hal_command_buffer_begin(stream->pending_cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(stream->pending_cb);
    stream->pending_cb = NULL;
    return pyre_status_from_iree(status);
  }
  return pyre_ok_status();
}

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

//===----------------------------------------------------------------------===//
// Async stream transfers
//===----------------------------------------------------------------------===//

pyre_status_t pyre_stream_copy_h2d(pyre_stream_t stream,
                                   const void* host_src,
                                   pyre_buffer_t dst, size_t dst_offset,
                                   size_t size) {
  if (!host_src || !dst) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  if (!stream) {
    return pyre_synchronous_h2d(dst->device, host_src, dst, dst_offset, size);
  }

  // Flush any pending dispatches before transfer for proper ordering.
  pyre_status_t status = pyre_stream_flush(stream);
  if (!pyre_status_is_ok(status)) return status;

  // Chunked synchronous H2D (matches streaming layer: 63KB chunks for
  // inline BUFFER_UPDATE path compatibility with remote HAL).
  const size_t chunk_size = 63 * 1024;
  const uint8_t* src_ptr = (const uint8_t*)host_src;
  size_t remaining = size;
  size_t chunk_offset = 0;

  while (remaining > 0) {
    size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;
    iree_status_t iree_status = iree_hal_device_transfer_h2d(
        stream->device->hal_device, src_ptr + chunk_offset,
        dst->hal_buffer,
        (iree_device_size_t)(dst_offset + chunk_offset),
        (iree_device_size_t)this_chunk,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    if (!iree_status_is_ok(iree_status)) {
      return pyre_status_from_iree(iree_status);
    }
    chunk_offset += this_chunk;
    remaining -= this_chunk;
  }
  return pyre_ok_status();
}

pyre_status_t pyre_stream_copy_d2h(pyre_stream_t stream,
                                   pyre_buffer_t src, size_t src_offset,
                                   void* host_dst, size_t size) {
  if (!src || !host_dst) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  if (!stream) {
    return pyre_synchronous_d2h(src->device, src, src_offset, host_dst, size);
  }

  // Flush and synchronize stream for D2H (must complete pending writes first).
  pyre_status_t status = pyre_stream_synchronize(stream);
  if (!pyre_status_is_ok(status)) return status;

  // Chunked synchronous D2H (4MB chunks to avoid staging buffer overflows).
  const size_t chunk_size = 4 * 1024 * 1024;
  uint8_t* dst_ptr = (uint8_t*)host_dst;
  size_t remaining = size;
  size_t chunk_offset = 0;

  while (remaining > 0) {
    size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;
    iree_status_t iree_status = iree_hal_device_transfer_d2h(
        stream->device->hal_device, src->hal_buffer,
        (iree_device_size_t)(src_offset + chunk_offset),
        dst_ptr + chunk_offset, (iree_device_size_t)this_chunk,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    if (!iree_status_is_ok(iree_status)) {
      return pyre_status_from_iree(iree_status);
    }
    chunk_offset += this_chunk;
    remaining -= this_chunk;
  }
  return pyre_ok_status();
}
