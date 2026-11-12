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
    iree_byte_span_t arguments, uint64_t argument_address,
    iree_hal_amd_gpu_dma_transfer_chunk_t* out_chunk,
    uint32_t* out_argument_length) {
  out_chunk->word_count = 0;
  out_chunk->staging_length = 0;
  *out_argument_length = 0;
  out_chunk->next = transfer->cursor;
  out_chunk->download = (iree_hal_amd_gpu_download_t){0};
  iree_status_t status = iree_ok_status();
  while (out_chunk->next.index < transfer->count &&
         out_chunk->word_count + 31 <= word_capacity &&
         *out_argument_length + sizeof(iree_hal_amd_gpu_blit_arguments_t) <=
             arguments.data_length &&
         iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_transfer_operation_t* operation =
        &transfer->operations[out_chunk->next.index];
    if (operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_COPY) {
      break;
    }
    iree_hal_amd_gpu_blit_arguments_t blit_arguments = {
        .length = operation->length - out_chunk->next.offset,
    };
    status = iree_hal_amd_gpu_buffer_reference_resolve(
        &operation->source.device, &blit_arguments.source);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_buffer_reference_resolve(
          &operation->target.device, &blit_arguments.target);
    }
    if (iree_status_is_ok(status)) {
      blit_arguments.source += out_chunk->next.offset;
      blit_arguments.target += out_chunk->next.offset;
      memcpy(arguments.data + *out_argument_length, &blit_arguments,
             sizeof(blit_arguments));
      const uint32_t size[3] = {64, 1, 1};
      const uint32_t partial[3] = {0, 0, 0};
      const uint32_t groups[3] = {
          iree_hal_amd_gpu_blit_workgroup_count(blit_arguments.length), 1, 1};
      const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
          .kernarg_pointer = argument_address + *out_argument_length,
      };
      uint32_t* words = iree_hal_amd_gpu_pm4_launch_emit(
          out_chunk->words + out_chunk->word_count, launch, size, partial, 0,
          &inputs);
      words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
          words, groups, launch->dispatch_initiator);
      out_chunk->word_count = (uint32_t)(words - out_chunk->words);
      *out_argument_length += sizeof(blit_arguments);
      ++out_chunk->next.index;
      out_chunk->next.offset = 0;
    }
  }
  return status;
}
