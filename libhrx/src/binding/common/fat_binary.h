// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Streaming-layer fat-binary / offload-bundle unpacker.
//
// AMDGPU HAL accepts raw AMD HSACO ELF images, nothing fancier. Anything the
// HIP / CUDA compiler toolchain produces wrapped around those ELFs —
// fat-binary wrappers, Clang offload bundles, and CCOB (compressed Clang
// offload bundles, zstd-compressed) — is unwrapped here, at the streaming
// layer, and only matching raw ELF data plus explicit target metadata is
// forwarded down to the HAL. This keeps the HAL minimal and lets the streaming
// layer be the single source of truth for "what formats does HIP/CUDA hand us".

#ifndef IREE_HAL_STREAMING_FAT_BINARY_H_
#define IREE_HAL_STREAMING_FAT_BINARY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  IREE_HAL_STREAMING_FAT_BINARY_FORMAT_CAPACITY = 64,
};

// One ELF image matched out of a fat-binary / offload bundle.
// The span's lifetime is owned by iree_hal_streaming_fat_binary_extract_t
// and remains valid until that object is reset/deinitialized.
typedef struct iree_hal_streaming_fat_binary_elf_t {
  // Raw AMD HSACO ELF bytes ready to hand to the HAL executable cache.
  iree_const_byte_span_t data;
  // Bundle entry triple (empty for raw-ELF inputs). Borrowed.
  iree_string_view_t triple;
  // NUL-terminated HAL AMDGPU executable format derived from the ELF header.
  char executable_format[IREE_HAL_STREAMING_FAT_BINARY_FORMAT_CAPACITY];
} iree_hal_streaming_fat_binary_elf_t;

// Ranked fat-binary target candidate.
typedef struct iree_hal_streaming_fat_binary_target_t {
  // Target string to match against bundle entries.
  iree_string_view_t value;
} iree_hal_streaming_fat_binary_target_t;

// Owns the ELFs produced by a single fat-binary unpack.
// Must be reset with iree_hal_streaming_fat_binary_extract_reset when the
// contained ELF spans are no longer in use. ELF spans may alias either the
// caller-supplied input or an internally-owned decompressed buffer — never
// keep them past the matching reset.
typedef struct iree_hal_streaming_fat_binary_extract_t {
  iree_allocator_t host_allocator;

  // Decompressed buffer backing store (non-NULL only for CCOB inputs).
  void* owned_buffer;
  iree_host_size_t owned_buffer_size;

  // Heap-allocated array of matched ELFs.
  iree_hal_streaming_fat_binary_elf_t* matches;
  iree_host_size_t match_count;
  iree_host_size_t match_capacity;
} iree_hal_streaming_fat_binary_extract_t;

// Cheap sniff: returns true iff |data| starts with a magic that the
// streaming-layer fat-binary unpacker knows how to crack open.
// Raw ELF also counts as "supported" (trivially passthrough).
bool iree_hal_streaming_fat_binary_is_supported(iree_const_byte_span_t data);

// Validates a raw AMDGPU HSACO ELF and derives the HAL AMDGPU executable
// format string from its code-object target metadata.
iree_status_t iree_hal_streaming_fat_binary_describe_amdgpu_elf(
    iree_const_byte_span_t elf_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_elf_size);

// Unwraps a fat-binary / offload-bundle / CCOB / raw ELF blob and returns
// every contained ELF whose bundle triple matches the best-ranked target
// candidate. For a raw-ELF input the ELF itself is returned as a single match
// with an empty triple.
//
// |targets| must be ordered from most-specific to least-specific. Each target
// value is matched against the bundle-entry triple's trailing gfx component
// (after the final "--" or final "-" separator, or the entire string for bare
// gfx targets), with feature specifiers such as ":sramecc+" or ":xnack-"
// stripped from both sides before comparison. If the bundle contains matches
// for several candidates, only matches for the earliest candidate are returned.
//
// On error |out_extract| is left empty (safe to reset).
// On success the caller owns |out_extract| and must call
// iree_hal_streaming_fat_binary_extract_reset once the ELF spans are no
// longer referenced.
iree_status_t iree_hal_streaming_fat_binary_extract_for_targets(
    iree_const_byte_span_t data, iree_host_size_t target_count,
    const iree_hal_streaming_fat_binary_target_t* targets,
    iree_allocator_t host_allocator,
    iree_hal_streaming_fat_binary_extract_t* out_extract);

// Drops any owned buffers and clears |extract| in place. Safe to call on
// an already-zeroed or already-reset extract.
void iree_hal_streaming_fat_binary_extract_reset(
    iree_hal_streaming_fat_binary_extract_t* extract);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STREAMING_FAT_BINARY_H_
