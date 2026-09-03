// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "transfer.h"

#include "hrx_internal.h"

iree_status_t hrx_hal_queue_transfer_and_wait(
    hrx_device_t device, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(operation_count == 0 || operations);

  iree_hal_queue_t* queue = device->transfer_queue;
  if (IREE_UNLIKELY(!queue)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HRX device has no transfer queue");
  }

  const iree_hal_queue_family_ordinal_t family_ordinal =
      iree_hal_queue_family_ordinal(iree_hal_queue_family(queue));
  iree_hal_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_hal_semaphore_create(
      device->hal_device, iree_hal_make_queue_family_affinity(family_ordinal),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_queue_transfer(queue, iree_hal_semaphore_list_empty(),
                                signal_semaphores, operation_count, operations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

static hrx_status_t hrx_stream_transfer(
    hrx_stream_t stream, const iree_hal_transfer_operation_t* operation) {
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) return status;
  if (IREE_UNLIKELY(stream->timepoint >= IREE_HAL_SEMAPHORE_MAX_VALUE)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "stream timeline value exhausted");
  }

  uint64_t wait_value = stream->timepoint;
  uint64_t signal_value = wait_value + 1;
  iree_hal_semaphore_t* semaphore = stream->semaphore->hal_semaphore;
  const iree_hal_semaphore_list_t wait_semaphores = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &semaphore,
      .payload_values = &wait_value,
  };
  const iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t iree_status = iree_hal_queue_transfer(
      stream->device->transfer_queue, wait_semaphores, signal_semaphores,
      /*operation_count=*/1, operation);
  if (iree_status_is_ok(iree_status)) {
    stream->timepoint = signal_value;
  }
  return hrx_status_from_iree(iree_status);
}

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
  if (dst->device != device) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "destination buffer belongs to another device"));
  }
  if (dst_offset > dst->size || size > dst->size - dst_offset) {
    HRX_RETURN_AND_END_ZONE(z0,
                            hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                            "transfer exceeds buffer size"));
  }

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload =
          {
              .source = host_src,
              .target_buffer = dst->hal_buffer,
              .target_offset = (iree_device_size_t)dst_offset,
              .length = (iree_device_size_t)size,
          },
  };
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_status_from_iree(
              hrx_hal_queue_transfer_and_wait(device, 1, &operation)));
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
  if (src->device != device) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "source buffer belongs to another device"));
  }
  if (src_offset > src->size || size > src->size - src_offset) {
    HRX_RETURN_AND_END_ZONE(z0,
                            hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                            "transfer exceeds buffer size"));
  }

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = src->hal_buffer,
              .source_offset = (iree_device_size_t)src_offset,
              .target = host_dst,
              .length = (iree_device_size_t)size,
          },
  };
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_status_from_iree(
              hrx_hal_queue_transfer_and_wait(device, 1, &operation)));
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
  if (dst->device != stream->device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "destination buffer belongs to another device");
  }
  if (dst_offset > dst->size || size > dst->size - dst_offset) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "transfer exceeds buffer size");
  }

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload =
          {
              .source = host_src,
              .target_buffer = dst->hal_buffer,
              .target_offset = (iree_device_size_t)dst_offset,
              .length = (iree_device_size_t)size,
          },
  };
  return hrx_stream_transfer(stream, &operation);
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
  if (src->device != stream->device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "source buffer belongs to another device");
  }
  if (src_offset > src->size || size > src->size - src_offset) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "transfer exceeds buffer size");
  }

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = src->hal_buffer,
              .source_offset = (iree_device_size_t)src_offset,
              .target = host_dst,
              .length = (iree_device_size_t)size,
          },
  };
  return hrx_stream_transfer(stream, &operation);
}
