// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/transfer.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

static iree_device_size_t iree_hal_amd_gpu_pm4_transfer_length(
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

iree_status_t iree_hal_amd_gpu_pm4_transfer_measure(
    iree_host_size_t count, const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_pm4_transfer_layout_t* out_layout) {
  iree_hal_amd_gpu_pm4_transfer_layout_t layout = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_device_size_t length =
        iree_hal_amd_gpu_pm4_transfer_length(&operations[i]);
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
          sizeof(iree_hal_amd_gpu_pm4_transfer_operation_t),
          &layout.byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "captured transfer descriptors overflow");
  }
  *out_layout = layout;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_transfer_qualify(
    const iree_hal_queue_family_t* family, iree_hal_buffer_t* buffer,
    iree_device_size_t offset,
    iree_hal_amd_gpu_pm4_transfer_address_t* out_address) {
  const iree_hal_memory_contract_t* contract =
      iree_hal_buffer_memory_view(buffer).contract;
  // Family/domain membership is established by public transfer validation.
  // Legacy buffers can satisfy that check without a native binding contract.
  if (!contract) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "transfer buffer has no native binding contract");
  }
  const uint16_t index =
      contract->scopes[family->memory.queue_scope_id]
          .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS];
  if (index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "GPU address access is not prepared for this queue");
  }
  *out_address = (iree_hal_amd_gpu_pm4_transfer_address_t){
      .device = {.buffer = buffer,
                 .offset = offset,
                 .slot = {.index = index,
                          .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS}},
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_transfer_initialize(
    const iree_hal_queue_family_t* family, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_pm4_transfer_layout_t layout, void* storage,
    iree_hal_amd_gpu_pm4_transfer_t* out_transfer) {
  *out_transfer = (iree_hal_amd_gpu_pm4_transfer_t){.operations = storage};
  uint8_t* bytes = (uint8_t*)(out_transfer->operations + layout.count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_transfer_operation_t* source = &operations[i];
    const iree_device_size_t length =
        iree_hal_amd_gpu_pm4_transfer_length(source);
    if (!length) {
      continue;
    }
    iree_hal_amd_gpu_pm4_transfer_operation_t operation = {.type = source->type,
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
      status = iree_hal_amd_gpu_pm4_transfer_qualify(
          family, source_buffer, source_offset, &operation.source);
    }
    if (iree_status_is_ok(status) && target_buffer) {
      status = iree_hal_amd_gpu_pm4_transfer_qualify(
          family, target_buffer, target_offset, &operation.target);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_retain(source_buffer);
      iree_hal_buffer_retain(target_buffer);
      out_transfer->operations[out_transfer->count++] = operation;
    }
  }
  return status;
}

void iree_hal_amd_gpu_pm4_transfer_deinitialize(
    iree_hal_amd_gpu_pm4_transfer_t* transfer) {
  for (iree_host_size_t i = 0; i < transfer->count; ++i) {
    const iree_hal_amd_gpu_pm4_transfer_operation_t* operation =
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

static iree_status_t iree_hal_amd_gpu_pm4_transfer_resolve(
    const iree_hal_amd_gpu_pm4_transfer_address_t* address,
    iree_device_size_t offset, uint64_t* out_address) {
  const iree_hal_buffer_memory_view_t view =
      iree_hal_buffer_memory_view(address->device.buffer);
  const uint64_t base =
      iree_hal_buffer_memory_native_binding(&view, address->device.slot)
          .device_address;
  if (!base) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transfer buffer has no committed GPU address");
  }
  *out_address = base + address->device.offset + offset;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_transfer_prepare_chunk(
    const iree_hal_amd_gpu_pm4_transfer_t* transfer, uint32_t word_capacity,
    iree_byte_span_t staging, uint64_t staging_address,
    iree_hal_amd_gpu_pm4_transfer_chunk_t* out_chunk) {
  out_chunk->word_count = 0;
  out_chunk->staging_length = 0;
  out_chunk->next = transfer->cursor;
  out_chunk->download = (iree_hal_amd_gpu_pm4_download_t){0};
  iree_status_t status = iree_ok_status();
  while (out_chunk->next.index < transfer->count &&
         out_chunk->word_count + 7 <= word_capacity &&
         !out_chunk->download.length && iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_pm4_transfer_operation_t* operation =
        &transfer->operations[out_chunk->next.index];
    const uint64_t offset = out_chunk->next.offset;
    // GFX11+ firmware consumers use a 32-byte-aligned count below 32 KiB.
    // Splitting is independent of the packet's wider representable count.
    uint32_t length = (uint32_t)iree_min(operation->length - offset, 32736u);
    uint64_t source = 0, target = 0;
    if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY ||
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      status = iree_hal_amd_gpu_pm4_transfer_resolve(&operation->source, offset,
                                                     &source);
    }
    if (iree_status_is_ok(status) &&
        operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      status = iree_hal_amd_gpu_pm4_transfer_resolve(&operation->target, offset,
                                                     &target);
    }
    if (!iree_status_is_ok(status)) {
      continue;
    }
    const bool fill = operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    const bool immediate = fill && !(target & 3) && length >= 4;
    const bool staged =
        operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_COPY && !immediate;
    if (immediate) {
      length &= ~3u;
    }
    if (fill && !immediate) {
      length = iree_min(length, 4u - (uint32_t)(target & 3));
    }
    if (staged) {
      length =
          iree_min(length, staging.data_length - out_chunk->staging_length);
      if (!length) {
        break;
      }
      uint8_t* data = staging.data + out_chunk->staging_length;
      const uint64_t address = staging_address + out_chunk->staging_length;
      if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
        target = address;
        out_chunk->download = (iree_hal_amd_gpu_pm4_download_t){
            .source = data,
            .target = (uint8_t*)operation->target.host + offset,
            .length = length,
        };
      } else {
        source = address;
        if (fill) {
          for (uint32_t i = 0; i < length; ++i) {
            data[i] = (uint8_t)(operation->pattern >> (((offset + i) & 3) * 8));
          }
        } else {
          memcpy(data, (const uint8_t*)operation->source.host + offset, length);
        }
      }
      out_chunk->staging_length += length;
    }
    uint32_t* words = out_chunk->words + out_chunk->word_count;
    if (immediate) {
      const uint32_t shift = (uint32_t)(offset & 3) * 8;
      const uint32_t pattern = shift ? (operation->pattern >> shift) |
                                           (operation->pattern << (32 - shift))
                                     : operation->pattern;
      iree_hal_amd_gpu_pm4_emit_dma_fill(words, target, length, pattern);
    } else {
      iree_hal_amd_gpu_pm4_emit_dma_copy(words, source, target, length);
    }
    out_chunk->word_count += 7;
    out_chunk->next.offset += length;
    if (out_chunk->next.offset == operation->length) {
      ++out_chunk->next.index;
      out_chunk->next.offset = 0;
    }
  }
  return status;
}
