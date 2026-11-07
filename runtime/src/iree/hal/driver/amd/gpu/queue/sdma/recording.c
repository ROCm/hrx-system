// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/recording.h"

void iree_hal_amd_gpu_sdma_recording_initialize(
    amdf_queue_format_features_t features, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sdma_recording_t* out_recording) {
  *out_recording = (iree_hal_amd_gpu_sdma_recording_t){.features = features};
  iree_hal_amd_gpu_recording_initialize(host_allocator,
                                        &out_recording->storage);
}

void iree_hal_amd_gpu_sdma_recording_deinitialize(
    iree_hal_amd_gpu_sdma_recording_t* recording) {
  iree_allocator_free(recording->storage.host_allocator, recording->segments);
  iree_allocator_free(recording->storage.host_allocator, recording->fixups);
  iree_hal_amd_gpu_recording_deinitialize(&recording->storage);
  *recording = (iree_hal_amd_gpu_sdma_recording_t){0};
}

static iree_status_t iree_hal_amd_gpu_sdma_recording_segment(
    iree_hal_amd_gpu_sdma_recording_t* recording,
    iree_hal_amd_gpu_sdma_program_segment_t segment) {
  if (recording->segment_count == recording->segment_capacity) {
    if (recording->segment_capacity > UINT32_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "SDMA recording exceeds its segment index space");
    }
    const uint32_t capacity = iree_max(16u, recording->segment_capacity * 2);
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        recording->storage.host_allocator, capacity,
        sizeof(*recording->segments), (void**)&recording->segments));
    recording->segment_capacity = capacity;
  }
  recording->segments[recording->segment_count++] = segment;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_sdma_recording_commands(
    iree_hal_amd_gpu_sdma_recording_t* recording, uint32_t word_count,
    uint32_t* out_offset, uint32_t** out_words) {
  iree_hal_amd_gpu_sdma_program_segment_t* segment =
      recording->segment_count
          ? &recording->segments[recording->segment_count - 1]
          : NULL;
  if (!segment || segment->kind != IREE_HAL_AMD_GPU_SDMA_PROGRAM_PACKETS ||
      segment->packets.word_count + word_count >
          IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_segment(
        recording,
        (iree_hal_amd_gpu_sdma_program_segment_t){
            .kind = IREE_HAL_AMD_GPU_SDMA_PROGRAM_PACKETS,
            .packets = {.word_offset = recording->storage.commands.length / 4,
                        .fixup_offset = recording->fixup_count}}));
    segment = &recording->segments[recording->segment_count - 1];
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_commands(
      &recording->storage, word_count * 4, 4, out_offset, (void**)out_words));
  segment->packets.word_count += word_count;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_sdma_recording_pointer(
    iree_hal_amd_gpu_sdma_recording_t* recording, uint32_t byte_offset,
    uint32_t value, uint64_t addend) {
  if (recording->fixup_count + 2 > recording->fixup_capacity) {
    if (recording->fixup_capacity > UINT32_MAX / 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "SDMA recording exceeds its replacement index space");
    }
    const uint32_t capacity = iree_max(32u, recording->fixup_capacity * 2);
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        recording->storage.host_allocator, capacity, sizeof(*recording->fixups),
        (void**)&recording->fixups));
    recording->fixup_capacity = capacity;
  }
  iree_hal_amd_gpu_sdma_program_segment_t* segment =
      &recording->segments[recording->segment_count - 1];
  const uint32_t offset = byte_offset - segment->packets.word_offset * 4;
  for (uint32_t i = 0; i < 2; ++i) {
    recording->fixups[recording->fixup_count++] = (iree_hal_amd_gpu_fixup_t){
        .target_offset = offset + i * 4,
        .source = value | ((i * 32) << 24),
        .addend = addend,
    };
  }
  segment->packets.fixup_count += 2;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_sdma_recording_range(
    iree_hal_amd_gpu_sdma_recording_t* recording,
    iree_hal_amd_gpu_sdma_program_segment_kind_t kind,
    iree_hal_amd_gpu_sdma_program_range_t range) {
  return iree_hal_amd_gpu_sdma_recording_segment(
      recording,
      (iree_hal_amd_gpu_sdma_program_segment_t){.kind = kind, .range = range});
}

iree_status_t iree_hal_amd_gpu_sdma_recording_retire(
    iree_hal_amd_gpu_sdma_recording_t* recording) {
  if (recording->segment_count &&
      recording->segments[recording->segment_count - 1].kind ==
          IREE_HAL_AMD_GPU_SDMA_PROGRAM_RETIRE) {
    return iree_ok_status();
  }
  return iree_hal_amd_gpu_sdma_recording_segment(
      recording, (iree_hal_amd_gpu_sdma_program_segment_t){
                     .kind = IREE_HAL_AMD_GPU_SDMA_PROGRAM_RETIRE});
}

iree_status_t iree_hal_amd_gpu_sdma_recording_seal(
    const iree_hal_amd_gpu_sdma_recording_t* recording,
    iree_hal_amd_gpu_sdma_program_t* out_program) {
  *out_program = (iree_hal_amd_gpu_sdma_program_t){0};
  iree_host_size_t size = 0, segments_offset = 0, fixups_offset = 0;
  iree_host_size_t words_offset = 0, data_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD(recording->segment_count,
                        iree_hal_amd_gpu_sdma_program_segment_t,
                        &segments_offset),
      IREE_STRUCT_FIELD(recording->fixup_count, iree_hal_amd_gpu_fixup_t,
                        &fixups_offset),
      IREE_STRUCT_FIELD(recording->storage.commands.length / 4, uint32_t,
                        &words_offset),
      IREE_STRUCT_FIELD(recording->storage.data.length, uint8_t,
                        &data_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      recording->storage.host_allocator, iree_max(size, 1u), (void**)&storage));
  if (recording->segment_count) {
    memcpy(storage + segments_offset, recording->segments,
           recording->segment_count * sizeof(*recording->segments));
  }
  if (recording->fixup_count) {
    memcpy(storage + fixups_offset, recording->fixups,
           recording->fixup_count * sizeof(*recording->fixups));
  }
  // Section-relative private addresses are already explicit dense-input
  // replacements. The shared append storage has no combined-image fixups.
  iree_hal_amd_gpu_recording_write_sections(
      &recording->storage, storage + words_offset, storage + data_offset, NULL);
  *out_program = (iree_hal_amd_gpu_sdma_program_t){
      .storage = storage,
      .features = recording->features,
      .segments = (const void*)(storage + segments_offset),
      .segment_count = recording->segment_count,
      .words = (const void*)(storage + words_offset),
      .fixups = (const void*)(storage + fixups_offset),
      .data = {storage + data_offset, recording->storage.data.length},
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_sdma_recording_barrier(
    iree_hal_amd_gpu_sdma_recording_t* recording,
    iree_hal_amd_gpu_sdma_barrier_profile_t profile,
    iree_hal_amd_gpu_sdma_barrier_t barrier) {
  const uint32_t visibility = barrier & (IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE |
                                         IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE);
  if (profile.submission && visibility) {
    return iree_hal_amd_gpu_sdma_recording_retire(recording);
  }
  const uint32_t cache =
      ((barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE) ? profile.acquire
                                                         : 0) |
      ((barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE) ? profile.release : 0);
  const bool join = (barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN) != 0;
  const uint32_t word_count = (join ? 1u : 0u) + (cache ? 5u : 0u);
  if (!word_count) {
    return iree_ok_status();
  }
  uint32_t offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_commands(
      recording, word_count, &offset, &words));
  if (join) {
    *words++ = 0;
  }
  if (cache) {
    iree_hal_amd_gpu_sdma_emit_cache(words, cache);
  }
  return iree_ok_status();
}
