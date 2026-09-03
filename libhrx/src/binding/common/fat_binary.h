// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Streaming-layer fat-binary / offload-bundle unpacker.
//
// AMDGPU HAL accepts raw AMD HSACO ELF images, nothing fancier. Toolchain
// containers around those ELFs — fat-binary wrappers, Clang offload bundles,
// and CCOB (compressed Clang offload bundles, zstd-compressed) — are unwrapped
// here, at the streaming layer, and only matching raw ELF data plus explicit
// target metadata is forwarded down to the HAL. This keeps the HAL minimal and
// lets the streaming layer be the single source of truth for accepted module
// container formats.

#ifndef IREE_HAL_STREAMING_FAT_BINARY_H_
#define IREE_HAL_STREAMING_FAT_BINARY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  IREE_HAL_STREAMING_FAT_BINARY_TARGET_KEY_CAPACITY = 64,
};

// One ELF image matched out of a fat-binary / offload bundle.
// The span's lifetime is managed by iree_hal_streaming_fat_binary_extract_t
// and remains valid until that object is reset.
typedef struct iree_hal_streaming_fat_binary_elf_t {
  // Raw AMD HSACO ELF bytes ready to load on the selected HAL target.
  iree_const_byte_span_t data;
  // Bundle entry triple (empty for raw-ELF inputs). Borrowed.
  iree_string_view_t triple;
  // Selected target borrowed from the source HAL device spec.
  const iree_hal_executable_target_t* executable_target;
  // NUL-terminated AMDGPU code-object target key derived from the ELF header.
  char
      code_object_target_key[IREE_HAL_STREAMING_FAT_BINARY_TARGET_KEY_CAPACITY];
} iree_hal_streaming_fat_binary_elf_t;

// Ranked fat-binary target candidate.
typedef struct iree_hal_streaming_fat_binary_target_t {
  // Executable target borrowed from a HAL device spec.
  const iree_hal_executable_target_t* executable_target;
} iree_hal_streaming_fat_binary_target_t;

// Owns the ELFs produced by a single fat-binary unpack.
// Must be reset with iree_hal_streaming_fat_binary_extract_reset when the
// contained ELF spans are no longer in use. ELF spans may alias either the
// caller-supplied input or an internally-owned decompressed buffer — never
// keep them past the matching reset.
typedef struct iree_hal_streaming_fat_binary_extract_t {
  // Host allocator used for owned storage.
  iree_allocator_t host_allocator;

  // Decompressed/resolved buffer backing store (non-NULL for CCOB inputs and
  // for HIPK out-of-band kpack inputs; the matched ELF spans point into it).
  void* owned_buffer;
  // Size of |owned_buffer| in bytes.
  iree_host_size_t owned_buffer_size;

  // Heap-allocated array of matched ELFs.
  iree_hal_streaming_fat_binary_elf_t* matches;
  // Number of populated entries in |matches|.
  iree_host_size_t match_count;
  // Allocated entry capacity of |matches|.
  iree_host_size_t match_capacity;
} iree_hal_streaming_fat_binary_extract_t;

// Cheap sniff: returns true iff |data| starts with a magic that the
// streaming-layer fat-binary unpacker knows how to crack open.
// Raw ELF also counts as "supported" (trivially passthrough).
bool iree_hal_streaming_fat_binary_is_supported(iree_const_byte_span_t data);

// Validates a raw AMDGPU HSACO ELF and derives the HAL AMDGPU target key from
// its code-object target metadata.
iree_status_t iree_hal_streaming_fat_binary_describe_amdgpu_elf(
    iree_const_byte_span_t elf_data, iree_host_size_t target_key_capacity,
    char* target_key, iree_host_size_t* out_elf_size);

// Callback invoked for each defined global object in an AMDGPU ELF symbol
// table. The name aliases |elf_data| and is only valid for the duration of the
// callback.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_streaming_fat_binary_global_visitor_t)(
    void* user_data, iree_string_view_t name);

// Visits defined global and weak object symbols in an AMDGPU ELF. A symbol
// present in both the static and dynamic symbol tables may be visited more than
// once. Callers should make processing idempotent when both tables are present.
iree_status_t iree_hal_streaming_fat_binary_visit_elf_global_objects(
    iree_const_byte_span_t elf_data,
    iree_hal_streaming_fat_binary_global_visitor_t visitor, void* user_data);

// Unwraps a fat-binary / offload-bundle / CCOB / raw ELF blob and returns
// every contained ELF compatible with the best-ranked target candidate. For a
// raw-ELF input the ELF itself is returned as a single match with an empty
// triple.
//
// |targets| must be ordered from most-specific to least-specific. Each target
// key is matched against the bundle-entry triple's trailing gfx component and
// validated against the code-object target derived from the ELF header. If the
// input contains matches for several candidates, only matches for the earliest
// compatible candidate are returned. Incompatible feature-specialized images
// are skipped.
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
