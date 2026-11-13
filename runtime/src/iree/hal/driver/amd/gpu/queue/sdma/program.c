// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/program.h"

static uint64_t iree_hal_amd_gpu_sdma_program_length(
    const iree_hal_amd_gpu_sdma_program_range_t* range,
    const uint64_t* values) {
  return range->length +
         (range->length_value == UINT32_MAX ? 0 : values[range->length_value]);
}

iree_hal_amd_gpu_sdma_program_progress_t iree_hal_amd_gpu_sdma_program_emit(
    const iree_hal_amd_gpu_sdma_program_t* program, const uint64_t* values,
    iree_hal_amd_gpu_sdma_program_cursor_t* cursor, uint32_t word_capacity,
    uint32_t* words, uint32_t* out_word_count) {
  uint32_t* output = words;
  uint32_t* const end = words + word_capacity;
  while (cursor->segment < program->segment_count) {
    const iree_hal_amd_gpu_sdma_program_segment_t* segment =
        &program->segments[cursor->segment];
    if (segment->kind == IREE_HAL_AMD_GPU_SDMA_PROGRAM_RETIRE) {
      ++cursor->segment;
      *out_word_count = (uint32_t)(output - words);
      return IREE_HAL_AMD_GPU_SDMA_PROGRAM_BOUNDARY;
    }
    if (segment->kind == IREE_HAL_AMD_GPU_SDMA_PROGRAM_PACKETS) {
      if ((uint32_t)(end - output) < segment->packets.word_count) {
        break;
      }
      memcpy(output, program->words + segment->packets.word_offset,
             segment->packets.word_count * sizeof(uint32_t));
      iree_hal_amd_gpu_fixup_apply(
          program->fixups + segment->packets.fixup_offset,
          segment->packets.fixup_count, values, output);
      output += segment->packets.word_count;
      ++cursor->segment;
      continue;
    }
    const iree_hal_amd_gpu_sdma_program_range_t* range = &segment->range;
    const uint64_t length = iree_hal_amd_gpu_sdma_program_length(range, values);
    if (cursor->offset == length) {
      ++cursor->segment;
      cursor->offset = 0;
      continue;
    }
    const uint64_t target =
        values[range->target_value] + range->target_offset + cursor->offset;
    uint32_t part = (uint32_t)iree_min(length - cursor->offset,
                                       IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH);
    if (segment->kind == IREE_HAL_AMD_GPU_SDMA_PROGRAM_COPY) {
      if (end - output < 7) {
        break;
      }
      output = iree_hal_amd_gpu_sdma_emit_copy(
          output, program->features,
          values[range->source_value] + range->source_offset + cursor->offset,
          target, part);
    } else {
      // Each edge reads only captured pattern bytes. Aligned interior packets
      // write DWORDs without touching adjacent bytes or expanding the payload.
      const uint32_t head = (uint32_t)(0 - target) & 3u;
      if (head || part < 4) {
        if (end - output < 7) {
          break;
        }
        part = iree_min(part, head ? head : 3u);
        output = iree_hal_amd_gpu_sdma_emit_copy(output, program->features,
                                                 values[range->source_value] +
                                                     range->source_offset +
                                                     (cursor->offset & 3u),
                                                 target, part);
      } else {
        if (end - output < 5) {
          break;
        }
        part &= ~3u;
        const uint32_t shift = (uint32_t)(cursor->offset & 3u) * 8u;
        const uint32_t pattern =
            shift ? (range->pattern >> shift) | (range->pattern << (32 - shift))
                  : range->pattern;
        output = iree_hal_amd_gpu_sdma_emit_fill(output, program->features,
                                                 target, part, pattern);
      }
    }
    cursor->offset += part;
  }
  *out_word_count = (uint32_t)(output - words);
  return cursor->segment == program->segment_count
             ? IREE_HAL_AMD_GPU_SDMA_PROGRAM_END
             : IREE_HAL_AMD_GPU_SDMA_PROGRAM_MORE;
}
