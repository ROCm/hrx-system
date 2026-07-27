// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sectioned SPIR-V module builder.
//
// The builder owns the target-local binary envelope for structured SPIR-V
// emission. It consumes the shared Loom target bundle, accumulates the SPIR-V
// feature atoms required by the ABI and emitted packets, and exposes ordered
// logical sections for target lowering to populate with declarations and
// functions. Finalization emits the capability, extension, and memory-model
// rows selected by the accumulated feature set.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_BUILDER_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_BUILDER_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/emit/spirv/binary_writer.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_module_section_e {
  // OpCapability rows.
  LOOM_SPIRV_MODULE_SECTION_CAPABILITY = 0,
  // OpExtension rows.
  LOOM_SPIRV_MODULE_SECTION_EXTENSION = 1,
  // OpExtInstImport rows.
  LOOM_SPIRV_MODULE_SECTION_EXTENDED_INSTRUCTION_IMPORT = 2,
  // Exactly one OpMemoryModel row.
  LOOM_SPIRV_MODULE_SECTION_MEMORY_MODEL = 3,
  // OpEntryPoint rows.
  LOOM_SPIRV_MODULE_SECTION_ENTRY_POINT = 4,
  // OpExecutionMode and OpExecutionModeId rows.
  LOOM_SPIRV_MODULE_SECTION_EXECUTION_MODE = 5,
  // Debug and name rows.
  LOOM_SPIRV_MODULE_SECTION_DEBUG = 6,
  // Annotation and decoration rows.
  LOOM_SPIRV_MODULE_SECTION_ANNOTATION = 7,
  // Types, constants, global variables, and other declarations.
  LOOM_SPIRV_MODULE_SECTION_DECLARATION = 8,
  // Function definitions.
  LOOM_SPIRV_MODULE_SECTION_FUNCTION = 9,
  LOOM_SPIRV_MODULE_SECTION_COUNT = 10,
} loom_spirv_module_section_t;

typedef struct loom_spirv_module_binary_t {
  // Allocator-owned SPIR-V words.
  uint32_t* words;
  // Number of initialized 32-bit words in |words|.
  iree_host_size_t word_count;
} loom_spirv_module_binary_t;

typedef struct loom_spirv_module_builder_t {
  // Allocator used for section and final module storage.
  iree_allocator_t allocator;
  // Target name used in feature diagnostics.
  iree_string_view_t target_name;
  // Export ABI selected for this module.
  loom_target_abi_kind_t abi_kind;
  // Feature atoms required by the ABI and emitted packets.
  loom_spirv_feature_bits_t required_feature_bits;
  // Prepared feature set emitted into the module preamble during finalization.
  loom_spirv_feature_set_t feature_set;
  // Upper bound for allocated result IDs, including the invalid zero ID.
  uint32_t id_bound;
  // Ordered logical section writers.
  loom_spirv_binary_writer_t sections[LOOM_SPIRV_MODULE_SECTION_COUNT];
} loom_spirv_module_builder_t;

// Returns |module| as immutable bytes suitable for SPIR-V consumers.
static inline iree_const_byte_span_t loom_spirv_module_binary_byte_span(
    const loom_spirv_module_binary_t* module) {
  return iree_make_const_byte_span((const uint8_t*)module->words,
                                   module->word_count * sizeof(uint32_t));
}

// Releases storage owned by |module|. Safe to call on a zero-initialized
// module object.
void loom_spirv_module_binary_deinitialize(loom_spirv_module_binary_t* module,
                                           iree_allocator_t allocator);

// Initializes a sectioned builder for |target|. |target| must describe a
// SPIR-V binary shader-entry-point or HAL-kernel bundle.
iree_status_t loom_spirv_module_builder_initialize(
    const loom_target_bundle_t* target, iree_allocator_t allocator,
    loom_spirv_module_builder_t* out_builder);

// Releases transient section storage owned by |builder|. Safe to call on a
// zero-initialized builder.
void loom_spirv_module_builder_deinitialize(
    loom_spirv_module_builder_t* builder);

// Adds feature atoms required by emitted module contents.
void loom_spirv_module_builder_require_feature_bits(
    loom_spirv_module_builder_t* builder,
    loom_spirv_feature_bits_t feature_bits);

// Returns the prepared feature set selected during finalization.
static inline const loom_spirv_feature_set_t*
loom_spirv_module_builder_feature_set(
    const loom_spirv_module_builder_t* builder) {
  return &builder->feature_set;
}

// Returns the writer for |section|. The returned writer remains owned by
// |builder| and is finalized in SPIR-V logical-layout order.
loom_spirv_binary_writer_t* loom_spirv_module_builder_section(
    loom_spirv_module_builder_t* builder, loom_spirv_module_section_t section);

// Allocates one fresh SPIR-V result ID from |builder|'s dense ID space.
uint32_t loom_spirv_module_builder_allocate_id(
    loom_spirv_module_builder_t* builder);

// Raises the SPIR-V result-ID bound to cover caller-assigned IDs in
// [1, |id_bound|). This supports deterministic target-lowering ID layouts.
void loom_spirv_module_builder_require_id_bound(
    loom_spirv_module_builder_t* builder, uint32_t id_bound);

// Finalizes all sections into one SPIR-V binary module. The resulting module
// owns its storage and must be deinitialized by the caller.
iree_status_t loom_spirv_module_builder_finalize(
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_binary_t* out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_BUILDER_H_
