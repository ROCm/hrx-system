// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/transfer.h"

#include "iree/hal/driver/amd/gpu/queue/blit.h"
#include "iree/hal/driver/amd/gpu/queue/buffer.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"

static uint32_t* iree_hal_amd_gpu_pm4_transfer_copy(uint32_t* words,
                                                    uint64_t features,
                                                    uint64_t source,
                                                    uint64_t target,
                                                    uint32_t length) {
  return iree_hal_amd_gpu_pm4_emit_dma_copy(words, source, target, length);
}

static uint32_t* iree_hal_amd_gpu_pm4_transfer_fill(uint32_t* words,
                                                    uint64_t features,
                                                    uint64_t target,
                                                    uint32_t length,
                                                    uint32_t pattern) {
  return iree_hal_amd_gpu_pm4_emit_dma_fill(words, target, length, pattern);
}

const iree_hal_amd_gpu_dma_transfer_encoding_t
    iree_hal_amd_gpu_pm4_transfer_encoding = {
        .max_packet_length = IREE_HAL_AMD_GPU_PM4_DMA_MAX_PACKET_LENGTH,
        .max_packet_words = 7,
        .copy = iree_hal_amd_gpu_pm4_transfer_copy,
        .fill = iree_hal_amd_gpu_pm4_transfer_fill,
};

iree_status_t iree_hal_amd_gpu_pm4_transfer_prepare_program_chunk(
    const iree_hal_amd_gpu_transfer_t* transfer,
    const iree_hal_amd_gpu_pm4_launch_t* launch, uint32_t word_capacity,
    iree_byte_span_t arguments, uint64_t arguments_address,
    iree_byte_span_t staging, uint64_t staging_address,
    iree_hal_amd_gpu_dma_transfer_chunk_t* out_chunk,
    uint32_t* out_argument_length) {
  out_chunk->word_count = 0;
  out_chunk->staging_length = 0;
  *out_argument_length = 0;
  out_chunk->next = transfer->cursor;
  out_chunk->download = (iree_hal_amd_gpu_download_t){0};
  const iree_hal_transfer_operation_type_t first_type =
      transfer->operations[transfer->cursor.index].type;
  const bool staged = first_type != IREE_HAL_TRANSFER_OPERATION_TYPE_COPY &&
                      first_type != IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  iree_status_t status = iree_ok_status();
  while (out_chunk->next.index < transfer->count &&
         out_chunk->word_count + 31 <= word_capacity &&
         !out_chunk->download.length && iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_transfer_operation_t* operation =
        &transfer->operations[out_chunk->next.index];
    const bool copy = operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
    const bool fill = operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    const bool download =
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    if ((!copy && !download && !operation->rectangular) ||
        staged != (!copy && !fill)) {
      break;
    }
    const uint32_t argument_offset = *out_argument_length;
    const uint32_t argument_end =
        argument_offset + sizeof(iree_hal_amd_gpu_blit_arguments_t);
    if (argument_end > arguments.data_length ||
        (staged && out_chunk->staging_length == staging.data_length)) {
      break;
    }
    const uint64_t total_length =
        iree_hal_amd_gpu_transfer_operation_length(operation);
    uint64_t length = total_length - out_chunk->next.offset;
    if (staged) {
      length =
          iree_min(length, staging.data_length - out_chunk->staging_length);
    }
    iree_hal_amd_gpu_blit_arguments_t blit_arguments = {
        .source = out_chunk->next.offset,
        .target = out_chunk->next.offset,
        .length = length,
        .row_count = 1,
        .slice_count = 1,
        .groups_y = 1,
        .groups_z = 1,
    };
    if (operation->rectangular) {
      length = iree_hal_amd_gpu_blit_prepare_region(
          operation->region, out_chunk->next.offset, length, &blit_arguments);
    }
    blit_arguments.pattern = operation->pattern;
    blit_arguments.fill = fill;
    uint64_t source = 0, target = 0;
    if (copy || download) {
      status = iree_hal_amd_gpu_buffer_reference_resolve(
          &operation->source.device, &source);
    }
    if (!download && iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_buffer_reference_resolve(
          &operation->target.device, &target);
    }
    if (iree_status_is_ok(status)) {
      blit_arguments.source += source;
      blit_arguments.target += target;
      if (staged) {
        uint8_t* bytes = staging.data + out_chunk->staging_length;
        const uint64_t address = staging_address + out_chunk->staging_length;
        if (download) {
          blit_arguments.target = address;
          blit_arguments.target_row_stride = blit_arguments.length;
          blit_arguments.target_slice_stride =
              blit_arguments.length * blit_arguments.row_count;
          // One contiguous download extent per publication.
          out_chunk->download = (iree_hal_amd_gpu_download_t){
              .source = bytes,
              .operation = operation,
              .offset = out_chunk->next.offset,
              .length = length,
          };
        } else {
          blit_arguments.source = address;
          blit_arguments.source_row_stride = blit_arguments.length;
          blit_arguments.source_slice_stride =
              blit_arguments.length * blit_arguments.row_count;
          iree_hal_amd_gpu_transfer_copy_host(operation, out_chunk->next.offset,
                                              length, bytes);
        }
      }
      memcpy(arguments.data + argument_offset, &blit_arguments,
             sizeof(blit_arguments));
      const uint32_t size[3] = {64, 1, 1};
      const uint32_t partial[3] = {0, 0, 0};
      const uint32_t groups[3] = {
          iree_hal_amd_gpu_blit_workgroup_count(blit_arguments.length),
          blit_arguments.groups_y, blit_arguments.groups_z};
      const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
          .kernarg_pointer = arguments_address + argument_offset,
      };
      uint32_t* words = iree_hal_amd_gpu_pm4_launch_emit(
          out_chunk->words + out_chunk->word_count, launch, size, partial, 0,
          &inputs);
      words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
          words, groups, launch->dispatch_initiator);
      out_chunk->word_count = (uint32_t)(words - out_chunk->words);
      *out_argument_length = argument_end;
      if (staged) {
        out_chunk->staging_length += (uint32_t)length;
      }
      out_chunk->next.offset += length;
      if (out_chunk->next.offset == total_length) {
        ++out_chunk->next.index;
        out_chunk->next.offset = 0;
      }
    }
  }
  return status;
}
