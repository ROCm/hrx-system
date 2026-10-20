// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/transfer.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/transfer.h"

iree_hal_amd_gpu_pm4_transfer_packet_t
iree_hal_amd_gpu_pm4_transfer_emit_packet(
    const iree_hal_amd_gpu_pm4_transfer_span_t* span, uint64_t offset,
    iree_byte_span_t staging, uint64_t staging_address, uint32_t* words) {
  // Keep the conservative packet policy independent of publication batching.
  uint32_t length = (uint32_t)iree_min(
      span->length - offset, IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_PACKET_LENGTH);
  uint64_t source = span->source_address + offset;
  uint64_t target = span->target_address + offset;
  const bool fill = span->type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  const bool immediate = fill && !(target & 3) && length >= 4;
  const bool staged =
      span->type != IREE_HAL_TRANSFER_OPERATION_TYPE_COPY && !immediate;
  if (immediate) {
    length &= ~3u;
  }
  if (fill && !immediate) {
    length = iree_min(length, 4u - (uint32_t)(target & 3));
  }
  if (staged) {
    length = iree_min(length, staging.data_length);
    if (!length) {
      return (iree_hal_amd_gpu_pm4_transfer_packet_t){0};
    }
    if (span->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      target = staging_address;
    } else {
      source = staging_address;
      if (fill) {
        for (uint32_t i = 0; i < length; ++i) {
          staging.data[i] =
              (uint8_t)(span->pattern >> (((offset + i) & 3) * 8));
        }
      } else {
        memcpy(staging.data, span->source_bytes + offset, length);
      }
    }
  }
  if (immediate) {
    const uint32_t shift = (uint32_t)(offset & 3) * 8;
    const uint32_t pattern =
        shift ? (span->pattern >> shift) | (span->pattern << (32 - shift))
              : span->pattern;
    iree_hal_amd_gpu_pm4_emit_dma_fill(words, target, length, pattern);
  } else {
    iree_hal_amd_gpu_pm4_emit_dma_copy(words, source, target, length);
  }
  return (iree_hal_amd_gpu_pm4_transfer_packet_t){
      .length = length,
      .staging_length = staged ? length : 0,
  };
}

iree_status_t iree_hal_amd_gpu_pm4_transfer_prepare_chunk(
    const iree_hal_amd_gpu_transfer_t* transfer, uint32_t word_capacity,
    iree_byte_span_t staging, uint64_t staging_address,
    iree_hal_amd_gpu_pm4_transfer_chunk_t* out_chunk) {
  out_chunk->word_count = 0;
  out_chunk->staging_length = 0;
  out_chunk->next = transfer->cursor;
  out_chunk->download = (iree_hal_amd_gpu_download_t){0};
  iree_status_t status = iree_ok_status();
  while (out_chunk->next.index < transfer->count &&
         out_chunk->word_count + 7 <= word_capacity &&
         iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_transfer_operation_t* operation =
        &transfer->operations[out_chunk->next.index];
    const uint64_t offset = out_chunk->next.offset;
    iree_hal_amd_gpu_pm4_transfer_span_t span = {
        .length = operation->length,
        .pattern = operation->pattern,
        .type = operation->type,
    };
    if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY ||
        operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      status = iree_hal_amd_gpu_transfer_resolve(&operation->source,
                                                 &span.source_address);
    } else {
      span.source_bytes = operation->source.host;
    }
    if (iree_status_is_ok(status) &&
        operation->type != IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
      status = iree_hal_amd_gpu_transfer_resolve(&operation->target,
                                                 &span.target_address);
    }
    if (iree_status_is_ok(status)) {
      uint8_t* data =
          staging.data ? staging.data + out_chunk->staging_length : NULL;
      const iree_hal_amd_gpu_pm4_transfer_packet_t packet =
          iree_hal_amd_gpu_pm4_transfer_emit_packet(
              &span, offset,
              iree_make_byte_span(
                  data, staging.data_length - out_chunk->staging_length),
              staging_address + out_chunk->staging_length,
              out_chunk->words + out_chunk->word_count);
      if (!packet.length) {
        break;
      }
      if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
        if (!out_chunk->download.length) {
          out_chunk->download = (iree_hal_amd_gpu_download_t){
              .source = data,
              .target = (uint8_t*)operation->target.host + offset,
          };
        }
        out_chunk->download.length += packet.length;
      }
      out_chunk->word_count += 7;
      out_chunk->staging_length += packet.staging_length;
      out_chunk->next.offset += packet.length;
      if (out_chunk->next.offset == operation->length) {
        ++out_chunk->next.index;
        out_chunk->next.offset = 0;
        if (out_chunk->download.length) {
          // One retirement record owns one contiguous host destination. A
          // different operation starts a new publication, even if it downloads.
          break;
        }
      }
    }
  }
  return status;
}
