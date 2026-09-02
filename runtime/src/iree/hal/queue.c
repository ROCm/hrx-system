// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/queue.h"

#include "iree/hal/buffer.h"
#include "iree/hal/detail.h"
#include "iree/hal/semaphore.h"

#define _VTABLE_DISPATCH(queue, method_name) \
  IREE_HAL_VTABLE_DISPATCH(queue, iree_hal_queue, method_name)

//===----------------------------------------------------------------------===//
// iree_hal_queue_family_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_queue_family_ordinal_t
iree_hal_queue_family_ordinal(const iree_hal_queue_family_t* queue_family) {
  IREE_ASSERT_ARGUMENT(queue_family);
  return queue_family->ordinal;
}

IREE_API_EXPORT void iree_hal_queue_family_initialize(
    iree_hal_queue_family_ordinal_t ordinal,
    iree_hal_queue_family_t* out_queue_family) {
  IREE_ASSERT_ARGUMENT(out_queue_family);
  out_queue_family->ordinal = ordinal;
}

//===----------------------------------------------------------------------===//
// iree_hal_queue_t
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(queue);

IREE_API_EXPORT const iree_hal_queue_family_t* iree_hal_queue_family(
    const iree_hal_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  return queue->queue_family;
}

static iree_status_t iree_hal_queue_validate_semaphore_list(
    const char* usage, const iree_hal_semaphore_list_t semaphore_list) {
  if (IREE_UNLIKELY(semaphore_list.count && (!semaphore_list.semaphores ||
                                             !semaphore_list.payload_values))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s semaphore list storage is null", usage);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (IREE_UNLIKELY(!semaphore_list.semaphores[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s semaphore %" PRIhsz " is null", usage, i);
    }
    if (IREE_UNLIKELY(semaphore_list.payload_values[i] >
                      IREE_HAL_SEMAPHORE_MAX_VALUE)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "%s semaphore %" PRIhsz " payload value %" PRIu64
          " exceeds the maximum HAL semaphore value %" PRIu64,
          usage, i, semaphore_list.payload_values[i],
          (uint64_t)IREE_HAL_SEMAPHORE_MAX_VALUE);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_queue_validate_transfer_buffer(
    const iree_hal_queue_t* queue, iree_hal_buffer_t* buffer,
    iree_device_size_t offset, iree_device_size_t length,
    iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access) {
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transfer buffer is null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(buffer, offset, length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer), required_usage));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer), required_access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));

  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  const iree_hal_queue_family_affinity_t queue_family_affinity =
      placement.queue_family_affinity;
  if (!iree_hal_buffer_placement_is_undefined(placement) &&
      !iree_hal_queue_family_affinity_is_any(queue_family_affinity)) {
    const iree_hal_queue_family_ordinal_t queue_family_ordinal =
        iree_hal_queue_family_ordinal(iree_hal_queue_family(queue));
    if (IREE_UNLIKELY(queue_family_ordinal >= IREE_HAL_MAX_QUEUE_FAMILIES ||
                      !iree_any_bit_set(queue_family_affinity,
                                        iree_hal_make_queue_family_affinity(
                                            queue_family_ordinal)))) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "queue family %u cannot access a buffer with family affinity "
          "0x%016" PRIx64,
          queue_family_ordinal, queue_family_affinity);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_queue_validate_transfer_operation(
    const iree_hal_queue_t* queue, iree_host_size_t operation_index,
    const iree_hal_transfer_operation_t* operation,
    bool* out_has_borrowed_host_range) {
  bool has_borrowed_host_range = false;
  iree_status_t status = iree_ok_status();
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      if (operation->fill.length == 0) break;
      if (IREE_UNLIKELY(operation->fill.flags != IREE_HAL_FILL_FLAG_NONE)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported fill flags: 0x%016" PRIx64,
                                  operation->fill.flags);
      } else if (IREE_UNLIKELY(!operation->fill.pattern)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "fill pattern is null");
      } else if (IREE_UNLIKELY(operation->fill.pattern_length != 1 &&
                               operation->fill.pattern_length != 2 &&
                               operation->fill.pattern_length != 4)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "fill pattern length must be 1, 2, or 4 bytes (got %" PRIhsz ")",
            operation->fill.pattern_length);
      } else if (IREE_UNLIKELY(operation->fill.target_offset %
                                       operation->fill.pattern_length !=
                                   0 ||
                               operation->fill.length %
                                       operation->fill.pattern_length !=
                                   0)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "fill range must be aligned to the pattern length");
      } else {
        status = iree_hal_queue_validate_transfer_buffer(
            queue, operation->fill.target_buffer, operation->fill.target_offset,
            operation->fill.length, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
            IREE_HAL_MEMORY_ACCESS_WRITE);
      }
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      if (operation->update.length == 0) break;
      if (IREE_UNLIKELY(operation->update.flags != IREE_HAL_UPDATE_FLAG_NONE)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported update flags: 0x%016" PRIx64,
                                  operation->update.flags);
      } else if (IREE_UNLIKELY(operation->update.length > IREE_HOST_SIZE_MAX)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "update length exceeds the host addressable range");
      } else if (IREE_UNLIKELY(
                     operation->update.source_offset >
                     IREE_HOST_SIZE_MAX -
                         (iree_host_size_t)operation->update.length)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "update source range exceeds the host addressable range");
      } else if (IREE_UNLIKELY(!operation->update.source_buffer)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "update source is null");
      } else {
        status = iree_hal_queue_validate_transfer_buffer(
            queue, operation->update.target_buffer,
            operation->update.target_offset, operation->update.length,
            IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
            IREE_HAL_MEMORY_ACCESS_WRITE);
      }
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      if (operation->copy.length == 0) break;
      if (IREE_UNLIKELY(operation->copy.flags != IREE_HAL_COPY_FLAG_NONE)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported copy flags: 0x%016" PRIx64,
                                  operation->copy.flags);
      } else {
        status = iree_hal_queue_validate_transfer_buffer(
            queue, operation->copy.source_buffer, operation->copy.source_offset,
            operation->copy.length, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
            IREE_HAL_MEMORY_ACCESS_READ);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_queue_validate_transfer_buffer(
            queue, operation->copy.target_buffer, operation->copy.target_offset,
            operation->copy.length, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
            IREE_HAL_MEMORY_ACCESS_WRITE);
      }
      if (iree_status_is_ok(status) &&
          IREE_UNLIKELY(
              iree_hal_buffer_test_overlap(
                  operation->copy.source_buffer, operation->copy.source_offset,
                  operation->copy.length, operation->copy.target_buffer,
                  operation->copy.target_offset, operation->copy.length) !=
              IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "copy source and target ranges overlap within one allocation");
      }
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      if (operation->upload.length == 0) break;
      if (IREE_UNLIKELY(operation->upload.length > IREE_HOST_SIZE_MAX)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "upload length exceeds the host addressable range");
      } else if (IREE_UNLIKELY(!operation->upload.source)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "upload source is null");
      } else {
        status = iree_hal_queue_validate_transfer_buffer(
            queue, operation->upload.target_buffer,
            operation->upload.target_offset, operation->upload.length,
            IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
            IREE_HAL_MEMORY_ACCESS_WRITE);
      }
      has_borrowed_host_range = iree_status_is_ok(status);
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (operation->download.length == 0) break;
      if (IREE_UNLIKELY(operation->download.length > IREE_HOST_SIZE_MAX)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "download length exceeds the host addressable range");
      } else if (IREE_UNLIKELY(!operation->download.target)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "download target is null");
      } else {
        status = iree_hal_queue_validate_transfer_buffer(
            queue, operation->download.source_buffer,
            operation->download.source_offset, operation->download.length,
            IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE, IREE_HAL_MEMORY_ACCESS_READ);
      }
      has_borrowed_host_range = iree_status_is_ok(status);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown transfer operation type %u",
                                (uint32_t)operation->type);
      break;
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(status, "transfer operation %" PRIhsz,
                                    operation_index);
  } else {
    *out_has_borrowed_host_range = has_borrowed_host_range;
  }
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_queue_transfer(iree_hal_queue_t* queue,
                        const iree_hal_semaphore_list_t wait_semaphore_list,
                        const iree_hal_semaphore_list_t signal_semaphore_list,
                        iree_host_size_t operation_count,
                        const iree_hal_transfer_operation_t* operations) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)operation_count);

  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!queue)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "queue is null");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_queue_validate_semaphore_list("wait", wait_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_queue_validate_semaphore_list("signal", signal_semaphore_list);
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(operation_count && !operations)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "transfer operation storage is null");
  }

  bool has_borrowed_host_range = false;
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    bool operation_has_borrowed_host_range = false;
    status = iree_hal_queue_validate_transfer_operation(
        queue, i, &operations[i], &operation_has_borrowed_host_range);
    has_borrowed_host_range |= operation_has_borrowed_host_range;
  }
  if (iree_status_is_ok(status) && has_borrowed_host_range &&
      signal_semaphore_list.count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transfers borrowing host memory require a signal semaphore");
  }
  if (iree_status_is_ok(status)) {
    status = _VTABLE_DISPATCH(queue, transfer)(queue, wait_semaphore_list,
                                               signal_semaphore_list,
                                               operation_count, operations);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_queue_fill(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
              .pattern = pattern,
              .pattern_length = pattern_length,
              .flags = flags,
          },
  };
  return iree_hal_queue_transfer(queue, wait_semaphore_list,
                                 signal_semaphore_list, 1, &operation);
}

IREE_API_EXPORT iree_status_t iree_hal_queue_update(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
      .update =
          {
              .source_buffer = source_buffer,
              .source_offset = source_offset,
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
              .flags = flags,
          },
  };
  return iree_hal_queue_transfer(queue, wait_semaphore_list,
                                 signal_semaphore_list, 1, &operation);
}

IREE_API_EXPORT iree_status_t iree_hal_queue_copy(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy =
          {
              .source_buffer = source_buffer,
              .source_offset = source_offset,
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
              .flags = flags,
          },
  };
  return iree_hal_queue_transfer(queue, wait_semaphore_list,
                                 signal_semaphore_list, 1, &operation);
}

IREE_API_EXPORT iree_status_t iree_hal_queue_upload(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload =
          {
              .source = source,
              .target_buffer = target_buffer,
              .target_offset = target_offset,
              .length = length,
          },
  };
  return iree_hal_queue_transfer(queue, wait_semaphore_list,
                                 signal_semaphore_list, 1, &operation);
}

IREE_API_EXPORT iree_status_t iree_hal_queue_download(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target, iree_device_size_t length) {
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = source_buffer,
              .source_offset = source_offset,
              .target = target,
              .length = length,
          },
  };
  return iree_hal_queue_transfer(queue, wait_semaphore_list,
                                 signal_semaphore_list, 1, &operation);
}

IREE_API_EXPORT void iree_hal_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_queue_vtable_t* vtable, iree_hal_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_queue);
  iree_hal_resource_initialize(vtable, &out_queue->resource);
  out_queue->queue_family = queue_family;
}
