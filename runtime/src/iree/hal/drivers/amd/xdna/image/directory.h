// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded structural directory for AMD XDNA executable images.
//
// The directory validates and retains an immutable ELF byte sequence, then
// exposes its decoded program headers without interpreting target-specific
// payloads. Program-header ordinals remain stable for the directory lifetime.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_DIRECTORY_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_DIRECTORY_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/xdna/image/format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A half-open byte range in the immutable image source sequence.
typedef struct iree_hal_amd_xdna_image_source_range_t {
  // Logical byte offset from the beginning of the source sequence.
  uint64_t offset;
  // Number of bytes in the range.
  uint64_t length;
} iree_hal_amd_xdna_image_source_range_t;

// One decoded ELF program-header directory entry.
typedef struct iree_hal_amd_xdna_image_program_header_t {
  // ELF program type identifying the payload role.
  uint32_t type;
  // Serialized payload bytes in the immutable source sequence.
  iree_hal_amd_xdna_image_source_range_t file_range;
  // Target virtual address associated with the payload.
  uint32_t virtual_address;
  // Target physical address or format-defined placement descriptor.
  uint32_t physical_address;
  // Target memory extent after loading, including any zero-filled tail.
  uint32_t memory_size;
  // ELF read, write, and execute permission bits.
  iree_hal_amd_xdna_elf_program_flags_t flags;
  // Power-of-two ELF load alignment in bytes.
  uint32_t alignment;
} iree_hal_amd_xdna_image_program_header_t;

// An immutable decoded ELF directory retaining its source byte sequence.
typedef struct iree_hal_amd_xdna_image_directory_t
    iree_hal_amd_xdna_image_directory_t;

// Creates a structural directory over |source_sequence|.
//
// The source sequence is borrowed during the call and retained on success.
// Decoding performs bounded allocation proportional only to the program-header
// count. Payload bytes remain in the source sequence and are never copied into
// the directory. Target-specific payload semantics are not validated here.
iree_status_t iree_hal_amd_xdna_image_directory_create(
    iree_byte_sequence_t* source_sequence, iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_directory_t** out_directory);

// Destroys |directory| and releases its retained source sequence.
void iree_hal_amd_xdna_image_directory_destroy(
    iree_hal_amd_xdna_image_directory_t* directory);

// Returns the complete immutable source byte length.
uint64_t iree_hal_amd_xdna_image_directory_source_length(
    const iree_hal_amd_xdna_image_directory_t* directory);

// Returns the target-specific ELF flags for later target qualification.
uint32_t iree_hal_amd_xdna_image_directory_target_flags(
    const iree_hal_amd_xdna_image_directory_t* directory);

// Returns the number of decoded program headers.
iree_host_size_t iree_hal_amd_xdna_image_directory_program_header_count(
    const iree_hal_amd_xdna_image_directory_t* directory);

// Returns the program header at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |directory| is destroyed.
const iree_hal_amd_xdna_image_program_header_t*
iree_hal_amd_xdna_image_directory_program_header(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_host_size_t ordinal);

// Enumerates the portion of source segments intersecting |source_range|.
//
// Segment boundaries are arbitrary and callers must handle fields split across
// callbacks. Segments remain valid only for the callback duration. An error
// returned by the callback stops enumeration and is propagated unchanged.
iree_status_t iree_hal_amd_xdna_image_directory_enumerate_source_range(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_sequence_segment_callback_t callback);

// Copies exactly |source_range.length| bytes into |storage|.
//
// Storage must have exactly the requested length. This is intended for bounded
// headers and tables; large payload consumers should enumerate instead.
iree_status_t iree_hal_amd_xdna_image_directory_read_source_range(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_span_t storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_DIRECTORY_H_
