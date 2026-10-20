// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/recording.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

void iree_hal_amd_gpu_pm4_recording_initialize(
    iree_allocator_t host_allocator, uint32_t instance_value,
    iree_hal_amd_gpu_pm4_recording_t* out_recording) {
  *out_recording = (iree_hal_amd_gpu_pm4_recording_t){
      .instance_value = instance_value,
  };
  iree_hal_amd_gpu_recording_initialize(host_allocator,
                                        &out_recording->storage);
}

// An eight-DWORD extent leaves the four-DWORD terminal edge last. A one-DWORD
// hole needs another alignment unit because type-3 NOPs require two words.
static uint32_t iree_hal_amd_gpu_pm4_recording_padding(uint32_t word_count) {
  const uint32_t padding = (0u - word_count - 4) & 7u;
  return padding == 1 ? padding + 8 : padding;
}

static iree_status_t iree_hal_amd_gpu_pm4_recording_close_block(
    iree_hal_amd_gpu_pm4_recording_t* recording, uint32_t* out_offset,
    uint32_t** out_exit) {
  const uint32_t word_count =
      (recording->storage.commands.length - recording->block_offset) / 4;
  const uint32_t padding = iree_hal_amd_gpu_pm4_recording_padding(word_count);
  uint32_t offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_commands(
      &recording->storage, (padding + 4) * 4, 4, &offset, (void**)&words));
  if (padding) {
    words = iree_hal_amd_gpu_pm4_emit_nop(words, padding);
  }
  const uint32_t complete_word_count = word_count + padding + 4;
  if (recording->pending_branch) {
    // Only temporary host storage is patched. Successor addresses are GPU
    // fixups, so every resident clone follows its own blocks.
    iree_hal_amd_gpu_pm4_emit_indirect_buffer(recording->pending_branch, 0,
                                              complete_word_count, true);
  } else {
    recording->entry_word_count = complete_word_count;
  }
  *out_offset = offset + padding * 4;
  *out_exit = words;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_recording_commands(
    iree_hal_amd_gpu_pm4_recording_t* recording, uint32_t word_count,
    uint32_t* out_offset, uint32_t** out_words) {
  IREE_ASSERT((uint64_t)word_count +
                  iree_hal_amd_gpu_pm4_recording_padding(word_count) + 4 <=
              UINT32_C(0xfffff));
  const uint32_t block_word_count =
      (recording->storage.commands.length - recording->block_offset) / 4 +
      word_count;
  if (block_word_count +
          iree_hal_amd_gpu_pm4_recording_padding(block_word_count) + 4 >
      UINT32_C(0xfffff)) {
    uint32_t offset = 0;
    uint32_t* branch = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_close_block(
        recording, &offset, &branch));
    iree_hal_amd_gpu_pm4_emit_indirect_buffer(branch, 0, 1, true);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_fixup(
        &recording->storage, offset + 4, recording->instance_value,
        recording->storage.commands.length, 0,
        IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE));
    recording->pending_branch = branch;
    recording->block_offset = recording->storage.commands.length;
  }
  return iree_hal_amd_gpu_recording_commands(
      &recording->storage, word_count * 4, 4, out_offset, (void**)out_words);
}

iree_status_t iree_hal_amd_gpu_pm4_recording_end(
    iree_hal_amd_gpu_pm4_recording_t* recording, uint32_t continuation_value,
    bool chain) {
  uint32_t offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_pm4_recording_close_block(recording, &offset, &words));
  if (chain) {
    iree_hal_amd_gpu_pm4_emit_indirect_buffer(words, 0, 32, true);
    return iree_hal_amd_gpu_recording_fixup(
        &recording->storage, offset + 4, continuation_value, 0, 0,
        IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE);
  }
  iree_hal_amd_gpu_pm4_emit_nop(words, 4);
  return iree_ok_status();
}
