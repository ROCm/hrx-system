// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/transfer.h"

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

static iree_status_t iree_hal_amd_gpu_transfer_qualify(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_hal_amd_gpu_transfer_address_t* out_address) {
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  // Family/domain membership is established by public transfer validation.
  // Legacy buffers can satisfy that check without a native binding contract.
  if (!contract) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "transfer buffer has no native binding contract");
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
  *out_address = (iree_hal_amd_gpu_transfer_address_t){
      .device = {.buffer = buffer,
                 .offset = offset,
                 .slot = {.index = index,
                          .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS}},
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_transfer_initialize(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    uint32_t maximum_staging_capacity, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_layout_t layout, void* storage,
    iree_hal_amd_gpu_transfer_t* out_transfer) {
  *out_transfer = (iree_hal_amd_gpu_transfer_t){.operations = storage};
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
      status = iree_hal_amd_gpu_transfer_qualify(
          family, site, source_buffer, source_offset, &operation.source);
    }
    if (iree_status_is_ok(status) && target_buffer) {
      status = iree_hal_amd_gpu_transfer_qualify(
          family, site, target_buffer, target_offset, &operation.target);
    }
    if (iree_status_is_ok(status)) {
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

iree_status_t iree_hal_amd_gpu_transfer_resolve(
    const iree_hal_amd_gpu_transfer_address_t* address, uint64_t* out_address) {
  const uint64_t base = iree_hal_buffer_native_binding(address->device.buffer,
                                                       address->device.slot)
                            .device_address;
  if (!base) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transfer buffer has no committed GPU address");
  }
  *out_address = base + address->device.offset;
  return iree_ok_status();
}
