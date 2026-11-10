// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared type payload decoding and canonical IR materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_TYPE_H_
#define LOOM_FORMAT_BYTECODE_READER_TYPE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/type_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_reader_module_view_t
    loom_bytecode_reader_module_view_t;

// Validated scoped shape before projecting source references into native IR.
// One stack record is consumed at a time; no field owns allocated storage.
typedef struct loom_bytecode_scoped_shape_t {
  // Literal sizes or one-based source value numbers selected by dynamic_axes.
  // A dynamic zero denotes an unbound dimension.
  uint64_t dimensions[LOOM_TYPE_MAX_RANK];
  // Static encoding index or zero-based source value number, per attachment.
  uint64_t encoding_ordinal;
  // Bit i selects a source value number in dimensions[i].
  uint16_t dynamic_axes;
  // Number of initialized dimensions.
  uint8_t rank;
  // Scalar element type, or NONE for a group.
  uint8_t element_type;
  // Wire encoding attachment kind: none, static, or SSA.
  uint8_t attachment;
  // Explicit view access alignment in bytes, or zero for natural alignment.
  uint8_t access_alignment;
} loom_bytecode_scoped_shape_t;

// State required to materialize a validated type plan into a module.
typedef struct loom_bytecode_type_materializer_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Full input bytes containing retained parameterized-type spans.
  iree_const_byte_span_t bytecode;
  // Finalized type and parameterized-type registry context.
  loom_context_t* context;
  // Immutable validated module facts containing the type plan.
  const loom_bytecode_reader_module_view_t* module_view;
  // Resettable scratch storage for composite type construction.
  iree_arena_allocator_t* scratch_arena;
  // Module receiving canonical type-table entries.
  loom_module_t* output_module;
  // Number of ordered plan entries already replaced with canonical IDs.
  loom_type_id_t position;
  // Next sparse fact in the unconsumed type-plan suffix.
  loom_bytecode_type_fact_t* next_fact;
} loom_bytecode_type_materializer_t;

// Validates a wire kind and maps it to the independent native type kind.
iree_status_t loom_bytecode_type_decode_kind(
    loom_bytecode_reader_decoder_t* decoder, uint8_t kind_byte, uint64_t offset,
    loom_type_kind_t* out_kind);

// Reads and validates the two target register carrier words shared by static
// and scoped type records. |type_index| identifies the record in diagnostics.
iree_status_t loom_bytecode_type_read_register_carrier(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t type_index,
    uint64_t out_payload[2]);

// Decodes a group, tensor, tile, vector, or view after its wire kind. Validates
// source references against their owning scope/table without constructing IR.
iree_status_t loom_bytecode_type_read_scoped_shape(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, loom_type_kind_t kind,
    uint64_t value_count, iree_host_size_t encoding_count,
    loom_bytecode_scoped_shape_t* out_shape);

// Interns a structural type from validated parent metadata and canonical child
// IDs. No temporary child payload is assembled. Selected readers project the
// plan's name and dependency IDs into |module| before construction. The result
// owns its payload independently of the plan and fact lifetimes.
iree_status_t loom_bytecode_type_materialize_structural(
    const loom_bytecode_structural_type_plan_t* plan,
    const loom_bytecode_structural_type_fact_t* fact,
    const loom_type_id_t* dependency_ids, loom_module_t* module,
    loom_type_id_t* out_type_id);

// Consumes a validated plan in source order, replacing each entry with its
// canonical output identity. Sparse structural child slots are consumed too.
// The caller supplies a monotonic prefix after constructing its prior encoding
// dependencies. Initialize |next_fact| to the plan's first fact before use.
iree_status_t loom_bytecode_type_materialize_prefix(
    loom_bytecode_type_materializer_t* materializer,
    iree_host_size_t type_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_TYPE_H_
