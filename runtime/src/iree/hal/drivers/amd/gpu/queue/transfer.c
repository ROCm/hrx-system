// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/transfer.h"

void iree_hal_amd_gpu_download_complete(
    const iree_hal_amd_gpu_download_t* download) {
  const iree_hal_amd_gpu_transfer_operation_t* operation = download->operation;
  const uint8_t* source = download->source;
  iree_device_size_t offset = download->offset;
  iree_host_size_t remaining = download->length;
  while (remaining) {
    const iree_host_size_t length =
        iree_min(remaining, operation->length - offset);
    memcpy((uint8_t*)operation->target.host + offset, source, length);
    source += length;
    remaining -= length;
    ++operation;
    offset = 0;
  }
}

void iree_hal_amd_gpu_transfer_profile(
    const iree_hal_amd_gpu_transfer_t* transfer,
    iree_hal_profile_queue_event_t* event) {
  event->type = transfer->count ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER
                                : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
  event->operation_count = (uint32_t)iree_min(transfer->count, UINT32_MAX);
  event->payload_length = 0;
  if (transfer->count == 1) {
    switch (transfer->operations[0].type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY;
        break;
    }
  }
  for (iree_host_size_t i = 0; i < transfer->count; ++i) {
    event->payload_length += iree_min(transfer->operations[i].length,
                                      UINT64_MAX - event->payload_length);
  }
}

static iree_device_size_t iree_hal_amd_gpu_transfer_length(
    const iree_hal_transfer_operation_t* operation) {
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      return operation->fill.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      return operation->update.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      return operation->copy.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      return operation->upload.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      return operation->download.length;
    default:
      IREE_BUILTIN_UNREACHABLE();
      return 0;
  }
}

iree_status_t iree_hal_amd_gpu_transfer_measure(
    iree_host_size_t count, const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_layout_t* out_layout) {
  iree_hal_amd_gpu_transfer_layout_t layout = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_device_size_t length =
        iree_hal_amd_gpu_transfer_length(&operations[i]);
    if (!length) {
      continue;
    }
    ++layout.count;
    if (operations[i].type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE &&
        !iree_host_size_checked_add(layout.byte_length, length,
                                    &layout.byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "captured transfer updates overflow");
    }
  }
  if (!iree_host_size_checked_mul_add(
          layout.byte_length, layout.count,
          sizeof(iree_hal_amd_gpu_transfer_operation_t), &layout.byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "captured transfer descriptors overflow");
  }
  *out_layout = layout;
  return iree_ok_status();
}

static bool iree_hal_amd_gpu_transfer_program_accessible(
    const iree_hal_queue_family_t* family,
    const iree_hal_amd_gpu_buffer_reference_t* reference,
    iree_hal_buffer_usage_t usage) {
  const iree_hal_memory_scope_access_t* access =
      &reference->buffer->memory.contract
           ->scopes[family->memory.queue_scope_id + 1];
  return iree_all_bits_set(access->usage, usage) &&
         access->bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] ==
             reference->slot.index;
}

iree_status_t iree_hal_amd_gpu_transfer_initialize(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    uint32_t maximum_staging_capacity, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_layout_t layout, void* storage,
    iree_hal_amd_gpu_transfer_t* out_transfer) {
  *out_transfer = (iree_hal_amd_gpu_transfer_t){
      .operations = storage,
      .program_accessible = layout.count > 1,
  };
  uint8_t* bytes = (uint8_t*)(out_transfer->operations + layout.count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_transfer_operation_t* source = &operations[i];
    const iree_device_size_t length = iree_hal_amd_gpu_transfer_length(source);
    if (!length) {
      continue;
    }
    iree_hal_amd_gpu_transfer_operation_t operation = {.type = source->type,
                                                       .length = length};
    iree_hal_buffer_t* source_buffer = NULL;
    iree_hal_buffer_t* target_buffer = NULL;
    iree_device_size_t source_offset = 0, target_offset = 0;
    switch (source->type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        target_buffer = source->fill.target_buffer;
        target_offset = source->fill.target_offset;
        memcpy(&operation.pattern, source->fill.pattern,
               source->fill.pattern_length);
        if (source->fill.pattern_length == 1) {
          operation.pattern *= 0x01010101u;
        }
        if (source->fill.pattern_length == 2) {
          operation.pattern |= operation.pattern << 16;
        }
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
        target_buffer = source->update.target_buffer;
        target_offset = source->update.target_offset;
        operation.source.host = bytes;
        memcpy(bytes,
               (const uint8_t*)source->update.source_buffer +
                   source->update.source_offset,
               length);
        bytes += length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
        source_buffer = source->copy.source_buffer;
        source_offset = source->copy.source_offset;
        target_buffer = source->copy.target_buffer;
        target_offset = source->copy.target_offset;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        operation.source.host = (void*)source->upload.source;
        target_buffer = source->upload.target_buffer;
        target_offset = source->upload.target_offset;
        out_transfer->host_wait_required = true;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        source_buffer = source->download.source_buffer;
        source_offset = source->download.source_offset;
        operation.target.host = source->download.target;
        out_transfer->host_completion_required = true;
        break;
    }
    if (source_buffer) {
      status = iree_hal_amd_gpu_buffer_reference_initialize(
          family, site, source_buffer, source_offset, &operation.source.device);
    }
    if (iree_status_is_ok(status) && target_buffer) {
      status = iree_hal_amd_gpu_buffer_reference_initialize(
          family, site, target_buffer, target_offset, &operation.target.device);
    }
    if (iree_status_is_ok(status)) {
      if (source->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY &&
          site != IREE_HAL_MEMORY_SITE_PROGRAM &&
          out_transfer->program_accessible) {
        out_transfer->program_accessible =
            iree_hal_amd_gpu_transfer_program_accessible(
                family, &operation.source.device,
                IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE) &&
            iree_hal_amd_gpu_transfer_program_accessible(
                family, &operation.target.device,
                IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET);
      }
      iree_hal_buffer_retain(source_buffer);
      iree_hal_buffer_retain(target_buffer);
      out_transfer->operations[out_transfer->count++] = operation;
      if (source->type != IREE_HAL_TRANSFER_OPERATION_TYPE_FILL &&
          source->type != IREE_HAL_TRANSFER_OPERATION_TYPE_COPY) {
        out_transfer->staging_capacity =
            iree_max(out_transfer->staging_capacity,
                     (uint32_t)iree_min(length, maximum_staging_capacity));
      }
    }
  }
  return status;
}

void iree_hal_amd_gpu_transfer_deinitialize(
    iree_hal_amd_gpu_transfer_t* transfer) {
  for (iree_host_size_t i = 0; i < transfer->count; ++i) {
    const iree_hal_amd_gpu_transfer_operation_t* operation =
        &transfer->operations[i];
    if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY ||
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      iree_hal_buffer_release(operation->source.device.buffer);
    }
    if (operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      iree_hal_buffer_release(operation->target.device.buffer);
    }
  }
}
