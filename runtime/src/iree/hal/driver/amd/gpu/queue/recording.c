// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/recording.h"

#include "iree/base/internal/memory_copy.h"

struct iree_hal_amd_gpu_recording_chunk_t {
  // Following initialized chunk, or NULL at the stream end.
  iree_hal_amd_gpu_recording_chunk_t* next;
  // Logical byte position of the first payload byte.
  uint32_t offset;
  // Initialized payload extent, in bytes.
  uint32_t length;
  // Allocated payload capacity, in bytes.
  uint32_t capacity;
};

typedef struct iree_hal_amd_gpu_recording_fixup_t {
  // Native replacement, with offsets still relative to their sections.
  iree_hal_amd_gpu_fixup_t value;
  // Producer-owned section coordinates and native replacement width.
  iree_hal_amd_gpu_recording_fixup_flags_t flags;
} iree_hal_amd_gpu_recording_fixup_t;

void iree_hal_amd_gpu_recording_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_recording_t* out_recording) {
  *out_recording = (iree_hal_amd_gpu_recording_t){
      .host_allocator = host_allocator,
      .alignment = 64,
  };
}

static void iree_hal_amd_gpu_recording_stream_deinitialize(
    iree_allocator_t allocator, iree_hal_amd_gpu_recording_stream_t* stream) {
  iree_hal_amd_gpu_recording_chunk_t* chunk = stream->head;
  while (chunk) {
    iree_hal_amd_gpu_recording_chunk_t* next = chunk->next;
    iree_allocator_free(allocator, chunk);
    chunk = next;
  }
  *stream = (iree_hal_amd_gpu_recording_stream_t){0};
}

void iree_hal_amd_gpu_recording_deinitialize(
    iree_hal_amd_gpu_recording_t* recording) {
  iree_hal_amd_gpu_recording_stream_deinitialize(recording->host_allocator,
                                                 &recording->commands);
  iree_hal_amd_gpu_recording_stream_deinitialize(recording->host_allocator,
                                                 &recording->data);
  iree_hal_amd_gpu_recording_stream_deinitialize(recording->host_allocator,
                                                 &recording->command_data);
  iree_hal_amd_gpu_recording_stream_deinitialize(recording->host_allocator,
                                                 &recording->fixups);
}

static iree_status_t iree_hal_amd_gpu_recording_append(
    iree_allocator_t allocator, iree_hal_amd_gpu_recording_stream_t* stream,
    uint32_t byte_length, uint32_t alignment, uint32_t* out_offset,
    void** out_data) {
  const uint64_t offset = iree_align_uint64(stream->length, alignment);
  if (offset + byte_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "resident program section exceeds 4 GiB");
  }
  iree_hal_amd_gpu_recording_chunk_t* chunk = stream->tail;
  if (!chunk || offset + byte_length - chunk->offset > chunk->capacity) {
    const uint32_t capacity = iree_max(byte_length, 16384u);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        allocator, sizeof(*chunk) + (iree_host_size_t)capacity,
        (void**)&chunk));
    *chunk = (iree_hal_amd_gpu_recording_chunk_t){
        .offset = (uint32_t)offset,
        .capacity = capacity,
    };
    if (stream->tail) {
      stream->tail->next = chunk;
    } else {
      stream->head = chunk;
    }
    stream->tail = chunk;
  }
  memset((uint8_t*)(chunk + 1) + chunk->length, 0,
         (size_t)(offset - chunk->offset - chunk->length));
  *out_offset = (uint32_t)offset;
  *out_data = (uint8_t*)(chunk + 1) + offset - chunk->offset;
  chunk->length = (uint32_t)offset + byte_length - chunk->offset;
  stream->length = (uint32_t)offset + byte_length;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_recording_commands(
    iree_hal_amd_gpu_recording_t* recording, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_offset, void** out_data) {
  return iree_hal_amd_gpu_recording_append(recording->host_allocator,
                                           &recording->commands, byte_length,
                                           alignment, out_offset, out_data);
}

iree_status_t iree_hal_amd_gpu_recording_data(
    iree_hal_amd_gpu_recording_t* recording, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_offset, void** out_data) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_append(
      recording->host_allocator, &recording->data, byte_length, alignment,
      out_offset, out_data));
  recording->alignment = iree_max(recording->alignment, alignment);
  memset(*out_data, 0, byte_length);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_recording_command_data(
    iree_hal_amd_gpu_recording_t* recording, uint32_t byte_length,
    uint32_t* out_offset, void** out_data) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_append(
      recording->host_allocator, &recording->command_data, byte_length, 4,
      out_offset, out_data));
  memset(*out_data, 0, byte_length);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_recording_fixup(
    iree_hal_amd_gpu_recording_t* recording, uint32_t target_offset,
    uint32_t source, uint64_t addend, uint32_t shift,
    iree_hal_amd_gpu_recording_fixup_flags_t flags) {
  const uint32_t count =
      (flags & IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE) ? 2 : 1;
  uint32_t offset = 0;
  iree_hal_amd_gpu_recording_fixup_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_append(
      recording->host_allocator, &recording->fixups, count * sizeof(*entries),
      iree_alignof(iree_hal_amd_gpu_recording_fixup_t), &offset,
      (void**)&entries));
  for (uint32_t i = 0; i < count; ++i) {
    entries[i] = (iree_hal_amd_gpu_recording_fixup_t){
        .value = {.target_offset = target_offset + i * 4,
                  .source = source | ((shift + i * 32) << 24),
                  .addend = addend},
        .flags = flags,
    };
  }
  return iree_ok_status();
}

uint32_t iree_hal_amd_gpu_recording_data_offset(
    const iree_hal_amd_gpu_recording_t* recording) {
  return iree_host_align(recording->commands.length, recording->alignment);
}

uint32_t iree_hal_amd_gpu_recording_fixup_count(
    const iree_hal_amd_gpu_recording_t* recording) {
  return recording->fixups.length / sizeof(iree_hal_amd_gpu_recording_fixup_t);
}

uint32_t iree_hal_amd_gpu_recording_command_data_offset(
    const iree_hal_amd_gpu_recording_t* recording) {
  const uint32_t offset = iree_hal_amd_gpu_recording_data_offset(recording) +
                          recording->data.length;
  return recording->command_data.length ? iree_host_align(offset, 4) : offset;
}

static void iree_hal_amd_gpu_recording_copy(
    const iree_hal_amd_gpu_recording_stream_t* stream, uint8_t* destination) {
  uint32_t position = 0;
  for (const iree_hal_amd_gpu_recording_chunk_t* chunk = stream->head; chunk;
       chunk = chunk->next) {
    // Chunks already contain initialized bytes, including their internal
    // alignment holes. Only gaps between chunks still need zeroing.
    iree_memset_write_only(destination + position, 0, chunk->offset - position);
    iree_memcpy_write_only(destination + chunk->offset, chunk + 1,
                           chunk->length);
    position = chunk->offset + chunk->length;
  }
}

void iree_hal_amd_gpu_recording_write(
    const iree_hal_amd_gpu_recording_t* recording, uint8_t* destination,
    uint32_t command_data_source, iree_hal_amd_gpu_fixup_t* table) {
  const uint32_t data_offset =
      iree_hal_amd_gpu_recording_data_offset(recording);
  iree_memset_write_only(destination + recording->commands.length, 0,
                         data_offset - recording->commands.length);
  const uint32_t command_data_offset =
      iree_hal_amd_gpu_recording_command_data_offset(recording);
  iree_memset_write_only(
      destination + data_offset + recording->data.length, 0,
      command_data_offset - data_offset - recording->data.length);
  iree_hal_amd_gpu_recording_write_sections(
      recording, destination, destination + data_offset,
      destination + command_data_offset, command_data_source, table);
}

void iree_hal_amd_gpu_recording_write_sections(
    const iree_hal_amd_gpu_recording_t* recording, uint8_t* commands,
    uint8_t* data, uint8_t* command_data, uint32_t command_data_source,
    iree_hal_amd_gpu_fixup_t* table) {
  const uint32_t data_offset =
      iree_hal_amd_gpu_recording_data_offset(recording);
  iree_hal_amd_gpu_recording_copy(&recording->commands, commands);
  iree_hal_amd_gpu_recording_copy(&recording->data, data);
  iree_hal_amd_gpu_recording_copy(&recording->command_data, command_data);
  const uint32_t command_data_offset =
      iree_hal_amd_gpu_recording_command_data_offset(recording);
  for (const iree_hal_amd_gpu_recording_chunk_t* chunk = recording->fixups.head;
       chunk; chunk = chunk->next) {
    const iree_hal_amd_gpu_recording_fixup_t* entries =
        (const void*)(chunk + 1);
    const uint32_t count = chunk->length / sizeof(*entries);
    for (uint32_t i = 0; i < count; ++i) {
      // The destination may be write-combined VRAM. Resolve in host storage
      // before writing the final entry so no adjustment reads it back.
      iree_hal_amd_gpu_fixup_t value = entries[i].value;
      if (entries[i].flags & IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA) {
        value.target_offset += data_offset;
      } else if (entries[i].flags &
                 IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_COMMAND_DATA) {
        value.target_offset += command_data_offset;
      }
      if (entries[i].flags & IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA) {
        value.addend += data_offset;
      } else if (entries[i].flags &
                 IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_COMMAND_DATA) {
        value.addend += command_data_offset;
        value.source = (value.source & 0xFF000000u) | command_data_source;
      }
      iree_memcpy_write_only(table++, &value, sizeof(value));
    }
  }
}
