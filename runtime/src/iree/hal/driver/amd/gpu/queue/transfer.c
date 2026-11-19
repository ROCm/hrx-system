// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/transfer.h"

#include "iree/hal/device_spec.h"

void iree_hal_amd_gpu_transfer_copy_host(
    const iree_hal_amd_gpu_transfer_operation_t* operation, uint64_t offset,
    iree_host_size_t length, void* staging) {
  const bool download =
      operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  uint8_t* host = download ? operation->target.host : operation->source.host;
  if (!operation->rectangular) {
    if (download) {
      memcpy(host + offset, staging, length);
    } else {
      memcpy(staging, host + offset, length);
    }
    return;
  }
  if (download) {
    iree_hal_transfer_scatter_host(staging, operation->region, offset, length,
                                   host);
  } else {
    iree_hal_transfer_gather_host(host, operation->region, offset, length,
                                  staging);
  }
}

void iree_hal_amd_gpu_download_complete(
    const iree_hal_amd_gpu_download_t* download) {
  const iree_hal_amd_gpu_transfer_operation_t* operation = download->operation;
  const uint8_t* source = download->source;
  iree_device_size_t offset = download->offset;
  iree_host_size_t remaining = download->length;
  while (remaining) {
    const iree_host_size_t length = iree_min(
        remaining,
        iree_hal_amd_gpu_transfer_operation_length(operation) - offset);
    iree_hal_amd_gpu_transfer_copy_host(operation, offset, length,
                                        (void*)source);
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
    event->payload_length += iree_min(
        iree_hal_amd_gpu_transfer_operation_length(&transfer->operations[i]),
        UINT64_MAX - event->payload_length);
  }
}

// UPDATE sources become dense at capture; fills have no source endpoint.
static bool iree_hal_amd_gpu_transfer_is_rectangular(
    iree_hal_transfer_operation_type_t type,
    const iree_hal_transfer_region_t* region) {
  return !iree_hal_transfer_range_is_linear(
             region->row_length, region->row_count, region->slice_count,
             &region->target_strides) ||
         ((type != IREE_HAL_TRANSFER_OPERATION_TYPE_FILL &&
           type != IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE) &&
          !iree_hal_transfer_range_is_linear(
              region->row_length, region->row_count, region->slice_count,
              &region->source_strides));
}

iree_status_t iree_hal_amd_gpu_transfer_measure(
    iree_host_size_t count, const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_layout_t* out_layout) {
  iree_hal_amd_gpu_transfer_layout_t layout = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_device_size_t length =
        iree_hal_transfer_operation_length(&operations[i]);
    if (!length) {
      continue;
    }
    ++layout.count;
    const iree_hal_transfer_region_t region =
        iree_hal_transfer_operation_region(&operations[i]);
    layout.region_count +=
        iree_hal_amd_gpu_transfer_is_rectangular(operations[i].type, &region);
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
  if (!iree_host_size_checked_mul_add(layout.byte_length, layout.region_count,
                                      sizeof(iree_hal_transfer_region_t),
                                      &layout.byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "captured transfer geometry overflows");
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

static void iree_hal_amd_gpu_transfer_append_operation(
    iree_hal_amd_gpu_transfer_staging_limits_t staging_limits,
    const iree_hal_amd_gpu_transfer_operation_t* operation,
    iree_hal_amd_gpu_transfer_t* transfer) {
  if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY ||
      operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
    iree_hal_buffer_retain(operation->source.device.buffer);
  }
  if (operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
    iree_hal_buffer_retain(operation->target.device.buffer);
  }
  transfer->operations[transfer->count++] = *operation;
  transfer->host_wait_required |=
      operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  transfer->host_completion_required |=
      operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  if (operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_FILL &&
      operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_COPY) {
    const uint32_t maximum_staging_capacity =
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD
            ? staging_limits.download_capacity
            : staging_limits.upload_capacity;
    transfer->staging_capacity =
        iree_max(transfer->staging_capacity,
                 (uint32_t)iree_min(
                     iree_hal_amd_gpu_transfer_operation_length(operation),
                     maximum_staging_capacity));
  }
}

iree_status_t iree_hal_amd_gpu_transfer_initialize(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    iree_hal_amd_gpu_transfer_staging_limits_t staging_limits,
    iree_host_size_t count, const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_layout_t layout, void* storage,
    iree_hal_amd_gpu_transfer_t* out_transfer) {
  *out_transfer = (iree_hal_amd_gpu_transfer_t){
      .operations = storage,
      .program_accessible = iree_any_bit_set(
          family->spec->role_flags, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH),
  };
  iree_hal_transfer_region_t* regions =
      (iree_hal_transfer_region_t*)(out_transfer->operations + layout.count);
  uint8_t* bytes = (uint8_t*)(regions + layout.region_count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_transfer_operation_t* source = &operations[i];
    const iree_device_size_t length =
        iree_hal_transfer_operation_length(source);
    if (!length) {
      continue;
    }
    iree_hal_amd_gpu_transfer_operation_t operation = {.type = source->type,
                                                       .length = length};
    iree_hal_transfer_region_t region =
        iree_hal_transfer_operation_region(source);
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
        iree_hal_transfer_gather_host(
            (const uint8_t*)source->update.source_buffer +
                source->update.source_offset,
            &region, 0, (iree_host_size_t)length, bytes);
        region.source_strides.row_stride = region.row_length;
        region.source_strides.slice_stride =
            region.row_length * region.row_count;
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
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        source_buffer = source->download.source_buffer;
        source_offset = source->download.source_offset;
        operation.target.host = source->download.target;
        break;
    }
    if (iree_hal_amd_gpu_transfer_is_rectangular(source->type, &region)) {
      operation.rectangular = true;
      operation.region = regions;
      *regions++ = region;
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
      if ((source->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY ||
           operation.rectangular) &&
          site != IREE_HAL_MEMORY_SITE_PROGRAM &&
          out_transfer->program_accessible) {
        out_transfer->program_accessible =
            (!source_buffer || iree_hal_amd_gpu_transfer_program_accessible(
                                   family, &operation.source.device,
                                   IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE)) &&
            (!target_buffer || iree_hal_amd_gpu_transfer_program_accessible(
                                   family, &operation.target.device,
                                   IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
      }
      iree_hal_amd_gpu_transfer_append_operation(staging_limits, &operation,
                                                 out_transfer);
    }
  }
  return status;
}

void iree_hal_amd_gpu_transfer_initialize_operations(
    iree_hal_amd_gpu_transfer_staging_limits_t staging_limits,
    iree_host_size_t count,
    const iree_hal_amd_gpu_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_operation_t* storage,
    iree_hal_amd_gpu_transfer_t* out_transfer) {
  *out_transfer = (iree_hal_amd_gpu_transfer_t){.operations = storage};
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_amd_gpu_transfer_append_operation(staging_limits, &operations[i],
                                               out_transfer);
  }
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
