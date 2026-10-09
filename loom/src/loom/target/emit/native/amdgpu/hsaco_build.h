// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA code-object plan construction.
//
// This layer validates a complete code-object input and resolves AMDGPU ABI
// policy into a final HSACO plan. Metadata notes, symbols, descriptor bytes,
// executable text, fixups, and virtual addresses are all finalized here.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_BUILD_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_BUILD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amdgpu/target_identity.h"
#include "loom/target/emit/native/amdgpu/descriptor.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/emit/native/amdgpu/text_fixup.h"

#ifdef __cplusplus
extern "C" {
#endif

// One kernel entry emitted into an AMDGPU HSA code object.
typedef struct loom_amdgpu_hsaco_kernel_t {
  // Kernel metadata row shared by the AMDGPU note and kernel descriptor.
  loom_amdgpu_metadata_kernel_t metadata;
  // Descriptor-only ABI controls that are not present in metadata.
  loom_amdgpu_kernel_descriptor_options_t descriptor_options;
  // Encoded native instructions for the kernel entry symbol.
  iree_const_byte_span_t text;
  // Text literal patches resolved after final code-object layout is known.
  const loom_amdgpu_hsaco_text_fixup_t* text_fixups;
  // Number of entries in |text_fixups|.
  iree_host_size_t text_fixup_count;
} loom_amdgpu_hsaco_kernel_t;

// Bitfield controlling AMDGPU HSACO data-symbol placement and access.
typedef uint32_t loom_amdgpu_hsaco_data_symbol_flags_t;
enum loom_amdgpu_hsaco_data_symbol_flag_bits_e {
  // Places the symbol in read-only data and marks it read-only.
  LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_NONE = 0u,
  // Places the symbol in the writable data segment instead of read-only data.
  LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE = 1u << 0,
};

// One data symbol emitted into an AMDGPU HSA code object.
typedef struct loom_amdgpu_hsaco_data_symbol_t {
  // Symbol name emitted into the dynamic and ordinary symbol tables.
  iree_string_view_t name;
  // Initial symbol bytes copied into the allocated storage.
  iree_const_byte_span_t initial_contents;
  // Total byte length of the symbol storage in the code object.
  uint64_t byte_length;
  // Required symbol alignment within its containing section.
  uint64_t alignment;
  // Placement and access flags for the symbol.
  loom_amdgpu_hsaco_data_symbol_flags_t flags;
} loom_amdgpu_hsaco_data_symbol_t;

// Data-object placement contract used by the code-object loader.
typedef uint32_t loom_amdgpu_hsaco_data_layout_t;
enum loom_amdgpu_hsaco_data_layout_e {
  // Packs objects according to their authored alignment requirements.
  LOOM_AMDGPU_HSACO_DATA_LAYOUT_PACKED = 0u,
  // Aligns objects to the ASAN v0 shadow granule and appends one redzone.
  LOOM_AMDGPU_HSACO_DATA_LAYOUT_ASAN_GLOBALS_V0 = 1u,
};

// Complete AMDGPU HSA code object description.
typedef struct loom_amdgpu_hsaco_input_t {
  // Resolved compiler target and normalized AMDHSA feature states.
  loom_amdgpu_target_identity_t target_identity;
  // Kernel entries emitted into this code object.
  const loom_amdgpu_hsaco_kernel_t* kernels;
  // Number of entries in |kernels|.
  iree_host_size_t kernel_count;
  // Data symbols emitted into this code object.
  const loom_amdgpu_hsaco_data_symbol_t* data_symbols;
  // Number of entries in |data_symbols|.
  iree_host_size_t data_symbol_count;
  // Placement contract applied to all data symbols in the code object.
  loom_amdgpu_hsaco_data_layout_t data_layout;
} loom_amdgpu_hsaco_input_t;

// Builds a trusted final code-object |out_plan| from |input|.
//
// All payload storage referenced by the plan is allocated from |arena|. The
// input and arena must remain live until the plan has been written.
iree_status_t loom_amdgpu_hsaco_plan_build(
    const loom_amdgpu_hsaco_input_t* input, loom_amdgpu_hsaco_plan_t* out_plan,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_BUILD_H_
