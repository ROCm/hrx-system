// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_RECORDING_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_RECORDING_H_

#include "iree/hal/driver/amd/gpu/queue/fixup.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_recording_chunk_t
    iree_hal_amd_gpu_recording_chunk_t;

// Temporary append storage. Chunk boundaries have no native execution meaning;
// sealing copies these bytes once into the program's immutable template layout.
typedef struct iree_hal_amd_gpu_recording_stream_t {
  // First initialized chunk, or NULL for an empty stream.
  iree_hal_amd_gpu_recording_chunk_t* head;
  // Current append chunk, or NULL before the first allocation.
  iree_hal_amd_gpu_recording_chunk_t* tail;
  // Complete logical extent, including alignment holes, in bytes.
  uint32_t length;
} iree_hal_amd_gpu_recording_stream_t;

// Recording-time native bytes and relocations, discarded after sealing. There
// are no operation records or invocation cursors. Commands and shader-readable
// data occupy separate ranges so command fetch never walks argument payloads.
typedef struct iree_hal_amd_gpu_recording_t {
  // Allocator for temporary chunks only.
  iree_allocator_t host_allocator;
  // Encoded native commands in execution order.
  iree_hal_amd_gpu_recording_stream_t commands;
  // Kernargs, indirect launch data, and captured update bytes.
  iree_hal_amd_gpu_recording_stream_t data;
  // Indirect native command bodies and their invocation-private control cells.
  iree_hal_amd_gpu_recording_stream_t command_data;
  // Independent fixups with recording-time section-relative offsets.
  iree_hal_amd_gpu_recording_stream_t fixups;
  // Maximum alignment requested for private data.
  uint32_t alignment;
} iree_hal_amd_gpu_recording_t;

void iree_hal_amd_gpu_recording_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_recording_t* out_recording);
void iree_hal_amd_gpu_recording_deinitialize(
    iree_hal_amd_gpu_recording_t* recording);

// Appends trusted native bytes with stable host storage until deinitialization.
// Protocol recorders impose packet boundaries before calling this primitive.
// The caller initializes every returned byte before sealing.
iree_status_t iree_hal_amd_gpu_recording_commands(
    iree_hal_amd_gpu_recording_t* recording, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_offset, void** out_data);

// Appends zero-initialized private data with the requested native alignment.
iree_status_t iree_hal_amd_gpu_recording_data(
    iree_hal_amd_gpu_recording_t* recording, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_offset, void** out_data);

// Appends zero-initialized native command data with DWORD alignment.
iree_status_t iree_hal_amd_gpu_recording_command_data(
    iree_hal_amd_gpu_recording_t* recording, uint32_t byte_length,
    uint32_t* out_offset, void** out_data);

// Section-relative relocation facts discarded when sealing resolves placement.
enum iree_hal_amd_gpu_recording_fixup_flag_bits_e {
  IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA = 1u << 0,
  IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA = 1u << 1,
  IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE = 1u << 2,
  IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_COMMAND_DATA = 1u << 3,
  IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_COMMAND_DATA = 1u << 4,
};
typedef uint32_t iree_hal_amd_gpu_recording_fixup_flags_t;

// Records a complete DWORD replacement, or a pair when WIDE is set. Flags
// select private-data or command-data target/addend offsets; otherwise offsets
// are relative to native command byte zero. shift is zero for pointers and
// eight for native instruction-address register pairs. No destination reads
// are performed.
iree_status_t iree_hal_amd_gpu_recording_fixup(
    iree_hal_amd_gpu_recording_t* recording, uint32_t target_offset,
    uint32_t source, uint64_t addend, uint32_t shift,
    iree_hal_amd_gpu_recording_fixup_flags_t flags);

// Returns the final private-data offset after the complete command extent has
// been appended. The final instance base satisfies recording->alignment.
uint32_t iree_hal_amd_gpu_recording_data_offset(
    const iree_hal_amd_gpu_recording_t* recording);
uint32_t iree_hal_amd_gpu_recording_command_data_offset(
    const iree_hal_amd_gpu_recording_t* recording);
uint32_t iree_hal_amd_gpu_recording_fixup_count(
    const iree_hal_amd_gpu_recording_t* recording);

// Copies admitted recording bytes and resolves section-relative fixups once.
// Destination covers command_data_offset + command_data.length; table covers
// fixup_count. command_data_source selects the dense input holding the native
// command-data base; zero shares the combined instance base with private data.
// Both outputs may be write-combined device memory. Each byte is written once,
// without reading either destination.
void iree_hal_amd_gpu_recording_write(
    const iree_hal_amd_gpu_recording_t* recording, uint8_t* destination,
    uint32_t command_data_source, iree_hal_amd_gpu_fixup_t* table);

// Writes command, private-data, and command-data sections separately. Fixups
// retain the same logical combined offsets as write; the caller supplies the
// resolved base used by their consumers. Each destination covers its section's
// length and may be write-combined memory. Each byte is written once, without
// reading any destination.
void iree_hal_amd_gpu_recording_write_sections(
    const iree_hal_amd_gpu_recording_t* recording, uint8_t* commands,
    uint8_t* data, uint8_t* command_data, uint32_t command_data_source,
    iree_hal_amd_gpu_fixup_t* table);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_RECORDING_H_
