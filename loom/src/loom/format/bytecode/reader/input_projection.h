// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Admission-owned projection of template input shapes in source coordinates.

#ifndef LOOM_FORMAT_BYTECODE_READER_INPUT_PROJECTION_H_
#define LOOM_FORMAT_BYTECODE_READER_INPUT_PROJECTION_H_

#include "loom/format/bytecode/reader/attribute.h"
#include "loom/ir/intern_table.h"
#include "loom/util/ordinal_map.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scratch-owned shape projection reused throughout one module's admission.
// No IR, persistent type graph, or per-symbol storage is constructed here.
typedef struct loom_bytecode_input_projection_t {
  // Public decoder establishing source diagnostics.
  loom_bytecode_reader_decoder_t* decoder;
  // Registry owning descriptor-backed type families.
  loom_context_t* context;
  // Already validated source tables and their original type section.
  loom_bytecode_reader_module_view_t* view;
  // Borrowed source TYPES entry ranges.
  const loom_bytecode_table_entry_metadata_t* type_entries;
  // Admission lifetime for canonical rows, indexes, and reused scope storage.
  iree_arena_allocator_t* arena;
  // Reached static type ordinals mapped to shape IDs; zero means non-shaped.
  loom_ordinal_map_t static_types;
  // Canonical source shapes shared across static roots and scoped records.
  loom_intern_table_t shape_index;
  // Stable pointers to canonical variable-sized shape rows.
  loom_segmented_storage_t shapes;
  // Reused completed-node shape IDs for the active callable signature.
  loom_segmented_storage_t bindings;
  // Number of completed nodes in the current signature scope.
  uint64_t binding_count;
  // Canonical sparse signatures, shared by providers with identical inputs.
  loom_intern_table_t signature_index;
  // Stable pointers to the canonical sparse signature records.
  loom_segmented_storage_t signatures;
  // Reused sparse input/shape pairs for the active signature.
  loom_segmented_storage_t arguments;
  // Number of shaped inputs recorded for the active signature.
  uint32_t argument_count;
  // Source symbol ordinals of templates only, in admission order.
  loom_segmented_storage_t template_symbols;
  // Number of recorded template symbols.
  uint32_t template_count;
} loom_bytecode_input_projection_t;

// Initializes an allocation-free projection. All storage borrows |arena|.
void loom_bytecode_input_projection_initialize(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* view,
    const loom_bytecode_table_entry_metadata_t* type_entries,
    iree_arena_allocator_t* arena,
    loom_bytecode_input_projection_t* out_projection);

// Starts a new callable scope while retaining reusable completion storage and
// module-wide static projections. The signature supplies all SSA value bounds.
void loom_bytecode_input_projection_begin_signature(
    loom_bytecode_input_projection_t* projection);

// Consumes the extension after a validated signature type reference. Returns
// zero for a non-shaped root, or a canonical shape ID valid until arena reset.
iree_status_t loom_bytecode_input_projection_read(
    loom_bytecode_input_projection_t* projection,
    loom_bytecode_reader_cursor_t* cursor, uint64_t source_reference,
    const loom_bytecode_attribute_ssa_validation_scope_t* scope,
    uint32_t* out_shape_id);

// Returns a previously published nonzero shape ID without allocation.
const loom_input_shape_t* loom_bytecode_input_projection_shape(
    const loom_bytecode_input_projection_t* projection, uint32_t shape_id);

// Records one input root after decoding its scoped extension. Non-shaped
// inputs require no record. Arguments arrive in increasing ordinal order.
iree_status_t loom_bytecode_input_projection_add_argument(
    loom_bytecode_input_projection_t* projection, uint16_t ordinal,
    uint32_t shape_id);

// Interns the active sparse signature in scratch storage. The returned ordinal
// is temporary and is replaced by its retained contract at finish_module.
iree_status_t loom_bytecode_input_projection_end_signature(
    loom_bytecode_input_projection_t* projection, uint32_t symbol_ordinal,
    uint32_t* out_ordinal);

// Freezes family obligations and provider-only differences after all symbols
// have been read, including forward family references. Only reached input
// requirements survive admission; the source projection remains scratch-owned.
iree_status_t loom_bytecode_input_projection_finish_module(
    loom_bytecode_input_projection_t* projection,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_module_metadata_t* metadata);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_INPUT_PROJECTION_H_
