// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/transfer.h"

#include "iree/hal/driver/amd/gpu/queue/buffer.h"

uint64_t iree_hal_amd_gpu_aql_transfer_staging_length(
    const iree_hal_amd_gpu_transfer_t* transfer, uint64_t capacity) {
  const uint64_t length = transfer->staging_capacity;
  if (length > capacity) {
    return length;
  }
  if (!transfer->host_completion_required) {
    if (!transfer->staging_capacity ||
        transfer->operations[transfer->cursor.index].length <=
            transfer->staging_capacity) {
      return length;
    }
    return iree_min(length, iree_max(8, capacity / 2));
  }
  const uint64_t count = iree_min(transfer->count - transfer->cursor.index,
                                  IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS);
  const uint64_t window =
      iree_max(8, iree_min(capacity / 2,
                           IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_DOWNLOAD_LENGTH));
  return iree_min(count * transfer->staging_capacity, window);
}

iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_transfer_packet(
    uint64_t kernel, uint64_t arguments, uint64_t length, uint32_t header) {
  const uint32_t groups = iree_hal_amd_gpu_blit_workgroup_count(length);
  const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
      .header = IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (1u << 16) | header,
      .workgroup_size = {64, 1, 1},
      .grid_size = {groups * 64, 1, 1},
      .kernel_object = kernel,
      .kernarg_address = arguments,
  };
  return iree_hal_amd_gpu_aql_packet_dispatch(&dispatch);
}

iree_status_t iree_hal_amd_gpu_aql_transfer_prepare_chunk(
    const iree_hal_amd_gpu_transfer_t* transfer, uint64_t kernel,
    uint32_t packet_capacity, iree_byte_span_t argument_storage,
    uint64_t argument_address, iree_byte_span_t staging,
    uint64_t staging_address,
    iree_hal_amd_gpu_aql_transfer_chunk_t* out_chunk) {
  *out_chunk = (iree_hal_amd_gpu_aql_transfer_chunk_t){
      .next = transfer->cursor,
  };
  iree_status_t status = iree_ok_status();
  while (out_chunk->next.index < transfer->count &&
         out_chunk->packet_count < packet_capacity &&
         iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_transfer_operation_t* operation =
        &transfer->operations[out_chunk->next.index];
    if (out_chunk->download.length &&
        operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      break;
    }
    const bool copy = operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
    const bool fill = operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    const bool download =
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    const uint64_t offset = out_chunk->next.offset;
    iree_hal_amd_gpu_blit_arguments_t arguments = {
        .length = operation->length - offset,
        .pattern = operation->pattern,
        .fill = fill,
    };
    if (!copy && !fill) {
      arguments.length = iree_min(
          arguments.length, staging.data_length - out_chunk->staging_length);
      if (!arguments.length) {
        break;
      }
    }
    if (copy || download) {
      status = iree_hal_amd_gpu_buffer_reference_resolve(
          &operation->source.device, &arguments.source);
      arguments.source += offset;
    }
    if (!download && iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_buffer_reference_resolve(
          &operation->target.device, &arguments.target);
      arguments.target += offset;
    }
    if (iree_status_is_ok(status)) {
      if (!copy && !fill) {
        uint8_t* bytes = staging.data + out_chunk->staging_length;
        const uint64_t address = staging_address + out_chunk->staging_length;
        if (download) {
          arguments.target = address;
          if (!out_chunk->download.length) {
            out_chunk->download = (iree_hal_amd_gpu_download_t){
                .source = bytes,
                .operation = operation,
                .offset = offset,
            };
          }
          out_chunk->download.length += arguments.length;
        } else {
          arguments.source = address;
          memcpy(bytes, (const uint8_t*)operation->source.host + offset,
                 arguments.length);
        }
        out_chunk->staging_length += (uint32_t)arguments.length;
      }
      const uint32_t argument_offset =
          out_chunk->packet_count * sizeof(arguments);
      memcpy(argument_storage.data + argument_offset, &arguments,
             sizeof(arguments));
      out_chunk->packets[out_chunk->packet_count++] =
          iree_hal_amd_gpu_aql_transfer_packet(
              kernel, argument_address + argument_offset, arguments.length, 0);
      out_chunk->next.offset += arguments.length;
      if (out_chunk->next.offset == operation->length) {
        ++out_chunk->next.index;
        out_chunk->next.offset = 0;
      }
    }
  }
  out_chunk->argument_length =
      out_chunk->packet_count * sizeof(iree_hal_amd_gpu_blit_arguments_t);
  return status;
}
