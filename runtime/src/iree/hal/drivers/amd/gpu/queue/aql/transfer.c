// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/transfer.h"

#include "iree/hal/drivers/amd/gpu/queue/buffer.h"

iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_transfer_packet(
    uint64_t kernel, uint64_t arguments, uint64_t length, uint32_t header) {
  const uint32_t groups =
      length > 256 * 1024 ? 256 : (uint32_t)((length + 1023) / 1024);
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
    uint32_t packet_capacity, iree_byte_span_t upload, uint64_t upload_address,
    iree_hal_amd_gpu_aql_transfer_chunk_t* out_chunk) {
  *out_chunk = (iree_hal_amd_gpu_aql_transfer_chunk_t){
      .next = transfer->cursor,
  };
  const uint32_t staging_offset =
      packet_capacity * sizeof(iree_hal_amd_gpu_blit_arguments_t);
  uint32_t staging_length = 0;
  iree_status_t status = iree_ok_status();
  while (out_chunk->next.index < transfer->count &&
         out_chunk->packet_count < packet_capacity &&
         iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_transfer_operation_t* operation =
        &transfer->operations[out_chunk->next.index];
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
      arguments.length =
          iree_min(arguments.length,
                   upload.data_length - staging_offset - staging_length);
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
        uint8_t* bytes = upload.data + staging_offset + staging_length;
        const uint64_t address =
            upload_address + staging_offset + staging_length;
        if (download) {
          arguments.target = address;
          out_chunk->download = (iree_hal_amd_gpu_download_t){
              .source = bytes,
              .target = (uint8_t*)operation->target.host + offset,
              .length = arguments.length,
          };
        } else {
          arguments.source = address;
          memcpy(bytes, (const uint8_t*)operation->source.host + offset,
                 arguments.length);
        }
        staging_length += (uint32_t)arguments.length;
      }
      const uint32_t argument_offset =
          out_chunk->packet_count * sizeof(arguments);
      memcpy(upload.data + argument_offset, &arguments, sizeof(arguments));
      out_chunk->packets[out_chunk->packet_count++] =
          iree_hal_amd_gpu_aql_transfer_packet(
              kernel, upload_address + argument_offset, arguments.length, 0);
      out_chunk->next.offset += arguments.length;
      if (out_chunk->next.offset == operation->length) {
        ++out_chunk->next.index;
        out_chunk->next.offset = 0;
      }
      if (download) {
        break;
      }
    }
  }
  out_chunk->argument_length =
      staging_length
          ? staging_offset + staging_length
          : out_chunk->packet_count * sizeof(iree_hal_amd_gpu_blit_arguments_t);
  return status;
}
