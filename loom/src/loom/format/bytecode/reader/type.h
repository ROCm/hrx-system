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

// Ordered descriptor parameters shared by static and scoped type readers.
// The owning reader decodes each value before advancing to the next name.
typedef struct loom_bytecode_type_parameters_t {
  // Registered family selected by the source spelling.
  const loom_parameterized_type_descriptor_t* descriptor;
  // Borrowed source strings used by the remaining parameter names.
  const iree_string_view_t* strings;
  // Number of source strings available to parameter references.
  iree_host_size_t string_count;
  // Position of the declared parameter count, for missing-parameter errors.
  uint64_t count_offset;
  // Source type ordinal, or zero for a scope-local type, for diagnostics.
  loom_type_id_t type_id;
  // Number of name/value pairs declared in the source payload.
  uint8_t present_count;
  // First descriptor slot not yet consumed by an ordered parameter name.
  uint8_t next_parameter_index;
  // A required slot skipped before a later present parameter.
  bool missing_required_parameter;
} loom_bytecode_type_parameters_t;

// Reads the registered family and bounded present-parameter count.
bool loom_bytecode_type_parameters_begin(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, loom_context_t* context,
    const iree_string_view_t* strings, iree_host_size_t string_count,
    loom_type_id_t type_id, loom_bytecode_type_parameters_t* out_parameters);

// Diagnoses an invalid source name reference or a name outside the remaining
// descriptor suffix. Kept off the per-parameter decoding path.
IREE_ATTRIBUTE_COLD bool loom_bytecode_type_parameters_reject_name(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_type_parameters_t* parameters, uint64_t name_id,
    uint64_t offset);

// Reads the next ordered name and returns its descriptor slot. The owning
// reader consumes the associated attribute kind and value afterward.
static inline bool loom_bytecode_type_parameters_next(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor,
    loom_bytecode_type_parameters_t* parameters, uint8_t* out_index) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t name_id = 0;
  LOOM_BYTECODE_RETURN_IF_REJECTED(
      loom_bytecode_reader_read_uvarint_inline(decoder, cursor, &name_id));
  if (name_id < parameters->string_count) {
    const iree_string_view_t name = parameters->strings[name_id];
    const loom_parameterized_type_descriptor_t* descriptor =
        parameters->descriptor;
    for (; parameters->next_parameter_index < descriptor->parameter_count;
         ++parameters->next_parameter_index) {
      const loom_attr_descriptor_t* candidate =
          &descriptor->parameter_descriptors[parameters->next_parameter_index];
      if (iree_string_view_equal(loom_attr_descriptor_name(candidate), name)) {
        *out_index = parameters->next_parameter_index++;
        return true;
      }
      parameters->missing_required_parameter |=
          !iree_any_bit_set(candidate->flags, LOOM_ATTR_OPTIONAL);
    }
  }
  return loom_bytecode_type_parameters_reject_name(decoder, cursor, parameters,
                                                   name_id, offset);
}

// Diagnoses a required parameter absent from the decoded source slots.
IREE_ATTRIBUTE_COLD bool loom_bytecode_type_parameters_reject_missing(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_type_parameters_t* parameters);

// Checks required slots after every present value has been consumed, preserving
// malformed-value precedence over missing-parameter diagnostics.
static inline bool loom_bytecode_type_parameters_finish(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_type_parameters_t* parameters) {
  bool missing_required = parameters->missing_required_parameter;
  const loom_parameterized_type_descriptor_t* descriptor =
      parameters->descriptor;
  for (uint8_t i = parameters->next_parameter_index;
       i < descriptor->parameter_count; ++i) {
    missing_required |= !iree_any_bit_set(
        descriptor->parameter_descriptors[i].flags, LOOM_ATTR_OPTIONAL);
  }
  if (missing_required) {
    return loom_bytecode_type_parameters_reject_missing(decoder, cursor,
                                                        parameters);
  }
  return true;
}

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
bool loom_bytecode_type_decode_kind(loom_bytecode_reader_decoder_t* decoder,
                                    uint8_t kind_byte, uint64_t offset,
                                    loom_type_kind_t* out_kind);

// Reads and validates the two target register carrier words shared by static
// and scoped type records. |type_index| identifies the record in diagnostics.
bool loom_bytecode_type_read_register_carrier(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t type_index,
    uint64_t out_payload[2]);

// Decodes the shared function, dialect, or register parent header after its
// wire kind. Names remain in source STRINGS coordinates; child references are
// read separately by their owning static or scoped reference domain. Register
// headers describe one child; the static grammar's following presence byte may
// select an untyped carrier instead. No storage is allocated.
bool loom_bytecode_type_read_structural_header(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, loom_type_kind_t kind,
    uint64_t type_index, iree_host_size_t string_count,
    loom_bytecode_structural_type_plan_t* out_plan, uint64_t out_payload[2]);

// Emits a malformed scoped-record diagnostic without constructing native IR.
bool loom_bytecode_type_emit_invalid_scoped_record(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, iree_string_view_t reason);

// Validates a static type ordinal or a prior completed scope-local node.
static inline bool loom_bytecode_type_validate_completed_reference(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t reference,
    iree_host_size_t static_count, uint64_t completed_count) {
  const uint64_t ordinal = reference >> 1;
  if (reference & 1) {
    if (ordinal >= completed_count) {
      return loom_bytecode_type_emit_invalid_scoped_record(
          decoder, cursor, IREE_SV("incomplete_child"));
    }
  } else if (ordinal >= static_count) {
    return loom_bytecode_type_emit_invalid_scoped_record(
        decoder, cursor, IREE_SV("unknown_static_type"));
  }
  return true;
}

// Opens the bounded extension following a type reference. Static references
// return an empty cursor without entering the scoped decoder. Otherwise the
// cursor starts at the first completed-type record and borrows the input bytes.
static inline bool loom_bytecode_type_open_scoped_records(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t reference,
    loom_bytecode_reader_cursor_t* out_records, uint64_t* out_count) {
  uint64_t length = 0;
  LOOM_BYTECODE_RETURN_IF_REJECTED(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &length));
  if (length == 0) {
    if (reference & 1) {
      return loom_bytecode_type_emit_invalid_scoped_record(
          decoder, cursor, IREE_SV("missing_scoped_record"));
    }
    loom_bytecode_reader_cursor_initialize(NULL, 0, 0, iree_string_view_empty(),
                                           out_records);
    *out_count = 0;
    return true;
  }
  if (reference != 1) {
    return loom_bytecode_type_emit_invalid_scoped_record(
        decoder, cursor, IREE_SV("unexpected_scoped_record"));
  }
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  iree_const_byte_span_t payload;
  LOOM_BYTECODE_RETURN_IF_REJECTED(
      loom_bytecode_reader_read_span(decoder, cursor, length, &payload));
  loom_bytecode_reader_cursor_initialize(payload.data, payload.data_length,
                                         offset, IREE_SV("SCOPED_TYPES"),
                                         out_records);
  return loom_bytecode_reader_read_uvarint(decoder, out_records, out_count);
}

// Finishes a nonempty extension after its records have completed. Returns the
// zero-based root ordinal in the enclosing completed-type scope and requires
// that no bytes remain in the extension.
static inline bool loom_bytecode_type_close_scoped_records(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* records, uint64_t completed_count,
    uint64_t* out_root) {
  uint64_t root = 0;
  LOOM_BYTECODE_RETURN_IF_REJECTED(
      loom_bytecode_reader_read_uvarint(decoder, records, &root));
  if (!root || root > completed_count) {
    return loom_bytecode_type_emit_invalid_scoped_record(decoder, records,
                                                         IREE_SV("root"));
  }
  LOOM_BYTECODE_RETURN_IF_REJECTED(loom_bytecode_reader_expect_empty(
      decoder, records, IREE_SV("SCOPED_TYPES")));
  *out_root = root - 1;
  return true;
}

// Decodes a group, tensor, tile, vector, or view after its wire kind. Validates
// source references against their owning scope/table without constructing IR.
bool loom_bytecode_type_read_scoped_shape(
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
bool loom_bytecode_type_materialize_prefix(
    loom_bytecode_type_materializer_t* materializer,
    iree_host_size_t type_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_TYPE_H_
