// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA code-object plan construction.
//
// This layer consumes a compiler-owned code-object input and resolves AMDGPU
// ABI policy into a final HSACO plan. Metadata notes, symbols, descriptor
// bytes, executable text, fixups, and virtual addresses are all finalized
// here. The artifact producer owns symbol admission and fixup validity.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_BUILD_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_BUILD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/target/arch/amdgpu/target_identity.h"
#include "loom/target/emit/native/amdgpu/descriptor.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/emit/native/amdgpu/text_fixup.h"

#ifdef __cplusplus
extern "C" {
#endif

// One unresolved direct call within a code-object text fragment.
//
// The producer owns exact instruction-local offsets. Final HSACO layout binds
// |target_function_ordinal| to one module-local function and patches both
// literal words.
typedef struct loom_amdgpu_hsaco_call_fixup_t {
  // Direct index into the code object's module-local function array.
  iree_host_size_t target_function_ordinal;
  // Fragment-local byte offset of the PC captured by s_getpc_b64.
  uint64_t base_pc_byte_offset;
  // Fragment-local byte offset of the s_add_u32 literal word.
  uint64_t low_literal_byte_offset;
  // Fragment-local byte offset of the s_addc_u32 literal word.
  uint64_t high_literal_byte_offset;
} loom_amdgpu_hsaco_call_fixup_t;

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
  // Direct calls resolved after final code-object text layout.
  const loom_amdgpu_hsaco_call_fixup_t* call_fixups;
  // Number of entries in |call_fixups|.
  iree_host_size_t call_fixup_count;
} loom_amdgpu_hsaco_kernel_t;

// One module-local function emitted into an AMDGPU HSA code object.
//
// Functions are executable text and ordinary local ELF symbols. They are not
// loader-visible kernels and therefore have no descriptor or metadata row.
typedef struct loom_amdgpu_hsaco_function_t {
  // Function symbol emitted only into the ordinary symbol table.
  iree_string_view_t name;
  // Encoded native instructions for the function symbol.
  iree_const_byte_span_t text;
  // Text literal patches resolved after final code-object layout is known.
  const loom_amdgpu_hsaco_text_fixup_t* text_fixups;
  // Number of entries in |text_fixups|.
  iree_host_size_t text_fixup_count;
  // Direct calls resolved after final code-object text layout.
  const loom_amdgpu_hsaco_call_fixup_t* call_fixups;
  // Number of entries in |call_fixups|.
  iree_host_size_t call_fixup_count;
} loom_amdgpu_hsaco_function_t;

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
  // Module-local functions emitted once into this code object.
  const loom_amdgpu_hsaco_function_t* functions;
  // Number of entries in |functions|.
  iree_host_size_t function_count;
  // Data symbols emitted into this code object.
  const loom_amdgpu_hsaco_data_symbol_t* data_symbols;
  // Number of entries in |data_symbols|.
  iree_host_size_t data_symbol_count;
  // Placement contract applied to all data symbols in the code object.
  loom_amdgpu_hsaco_data_layout_t data_layout;
} loom_amdgpu_hsaco_input_t;

// Builds a trusted final code-object |out_plan| from |input|.
//
// Input rows come from admitted module symbols and generated runtime globals;
// their counts fit the module's local symbol capacity. Names are admitted to
// u32 string extents, and text producers own valid fragment-local patch
// offsets. This owner admits aggregate section extents and final address
// displacements. All payload storage referenced by the plan is allocated from
// |arena|. The input and arena must remain live until the plan has been
// written. Only an accepted plan can be serialized. Artifact-format rejection
// emits diagnostics and leaves |out_accepted| false even without an emitter
// callback.
iree_status_t loom_amdgpu_hsaco_plan_build(
    const loom_amdgpu_hsaco_input_t* input, iree_arena_allocator_t* arena,
    iree_diagnostic_emitter_t emitter, bool* out_accepted,
    loom_amdgpu_hsaco_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_BUILD_H_
