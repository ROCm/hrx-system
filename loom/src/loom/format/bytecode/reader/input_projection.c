// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/input_projection.h"

#include <string.h>

#include "loom/format/bytecode/reader/type.h"
#include "loom/format/bytecode/reader/type_validator.h"
#include "loom/ir/structural_hash.h"

#define LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY 128u

typedef struct loom_bytecode_input_argument_t {
  // Ordinary argument coordinate in the source signature.
  uint32_t ordinal;
  // Canonical nonzero source shape identity.
  uint32_t shape_id;
} loom_bytecode_input_argument_t;

typedef struct loom_bytecode_input_signature_t {
  // Number of shaped arguments retained in values.
  uint32_t count;
  // Sparse arguments in source order.
  loom_bytecode_input_argument_t values[];
} loom_bytecode_input_signature_t;

void loom_bytecode_input_projection_initialize(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* view,
    const loom_bytecode_table_entry_metadata_t* type_entries,
    iree_arena_allocator_t* arena,
    loom_bytecode_input_projection_t* out_projection) {
  *out_projection = (loom_bytecode_input_projection_t){
      .decoder = decoder,
      .context = context,
      .view = view,
      .type_entries = type_entries,
      .arena = arena,
  };
  loom_ordinal_map_initialize(arena, &out_projection->static_types);
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY *
          sizeof(const loom_bytecode_input_shape_t*),
      iree_alignof(const loom_bytecode_input_shape_t*),
      &out_projection->shapes);
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY * sizeof(uint32_t),
      iree_alignof(uint32_t), &out_projection->bindings);
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY *
          sizeof(const loom_bytecode_input_signature_t*),
      iree_alignof(const loom_bytecode_input_signature_t*),
      &out_projection->signatures);
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY *
          sizeof(loom_bytecode_input_argument_t),
      iree_alignof(loom_bytecode_input_argument_t), &out_projection->arguments);
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY * sizeof(uint32_t),
      iree_alignof(uint32_t), &out_projection->template_symbols);
}

void loom_bytecode_input_projection_begin_signature(
    loom_bytecode_input_projection_t* projection) {
  projection->binding_count = 0;
  projection->argument_count = 0;
}

const loom_bytecode_input_shape_t* loom_bytecode_input_projection_shape(
    const loom_bytecode_input_projection_t* projection, uint32_t shape_id) {
  const uint32_t index = shape_id - 1;
  const void* const* segment = loom_segmented_storage_const_segment(
      &projection->shapes, index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
  return segment[index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY];
}

// Only a new canonical row needs storage. Keep allocation and table growth
// shared and outside the successful probe path used by repeated signatures.
static IREE_ATTRIBUTE_NOINLINE iree_status_t
loom_bytecode_input_projection_allocate_row(
    iree_arena_allocator_t* arena, loom_intern_table_t* index,
    loom_segmented_storage_t* rows, uint32_t hash, loom_intern_probe_t* probe,
    iree_host_size_t length, void** out_row) {
  if (!index->capacity) {
    IREE_RETURN_IF_ERROR(loom_intern_table_initialize(arena, 32, index));
    probe->slot = loom_intern_table_find_empty_slot(index, hash);
  } else {
    IREE_RETURN_IF_ERROR(
        loom_intern_table_reserve_insert(arena, index, hash, 1, &probe->slot));
  }
  const uint32_t segment_index =
      (uint32_t)(index->count / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
  const void** segment = NULL;
  if (segment_index == rows->segment_count) {
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(rows, arena, (void**)&segment));
  } else {
    segment = loom_segmented_storage_segment(rows, segment_index);
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, length, out_row));
  segment[index->count % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY] = *out_row;
  return iree_ok_status();
}

typedef struct loom_bytecode_input_shape_probe_t {
  // Canonical rows being probed.
  const loom_bytecode_input_projection_t* projection;
  // Fully initialized source key on the caller's stack.
  const loom_bytecode_input_shape_t* shape;
  // Exactly rank dimensions in the source decoder stack record.
  const uint64_t* dimensions;
} loom_bytecode_input_shape_probe_t;

static bool loom_bytecode_input_shape_equal(const void* opaque,
                                            uint32_t index) {
  const loom_bytecode_input_shape_probe_t* probe = opaque;
  const loom_bytecode_input_shape_t* existing =
      loom_bytecode_input_projection_shape(probe->projection, index + 1);
  return existing->rank == probe->shape->rank &&
         memcmp(existing, probe->shape, sizeof(*existing)) == 0 &&
         memcmp(existing->dimensions, probe->dimensions,
                existing->rank * sizeof(*existing->dimensions)) == 0;
}

static iree_status_t loom_bytecode_input_projection_intern(
    loom_bytecode_input_projection_t* projection, loom_type_kind_t kind,
    const loom_bytecode_scoped_shape_t* source, uint32_t* out_shape_id) {
  const loom_bytecode_input_shape_t header = {
      .kind = (uint8_t)kind,
      .element_type = source->element_type,
      .rank = source->rank,
      .access_alignment =
          source->access_alignment ==
                  iree_max(1,
                           loom_scalar_type_bitwidth(source->element_type) / 8)
              ? 0
              : source->access_alignment,
      .dynamic_axes = source->dynamic_axes,
      .attachment = source->attachment,
      .encoding_ordinal = source->encoding_ordinal,
  };
  const loom_bytecode_input_shape_t* key = &header;
  const iree_host_size_t length =
      sizeof(header) + source->rank * sizeof(*source->dimensions);
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u8(hash, key->kind);
  hash = loom_structural_hash_mix_u8(hash, key->element_type);
  hash = loom_structural_hash_mix_u8(hash, key->rank);
  hash = loom_structural_hash_mix_u8(hash, key->access_alignment);
  hash = loom_structural_hash_mix_u16(hash, key->dynamic_axes);
  hash = loom_structural_hash_mix_u8(hash, key->attachment);
  hash = loom_structural_hash_mix_u64(hash, key->encoding_ordinal);
  hash = loom_structural_hash_mix_bytes(
      hash, source->dimensions, source->rank * sizeof(*source->dimensions));
  hash = loom_structural_hash_finalize(hash);
  const loom_bytecode_input_shape_probe_t context = {
      .projection = projection, .shape = key, .dimensions = source->dimensions};
  loom_intern_probe_t probe =
      loom_intern_table_probe(&projection->shape_index, hash,
                              loom_bytecode_input_shape_equal, &context);
  if (probe.index != UINT32_MAX) {
    *out_shape_id = probe.index + 1;
    return iree_ok_status();
  }
  if (projection->shape_index.count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "source shape identity space exhausted");
  }
  const uint32_t index = (uint32_t)projection->shape_index.count;
  loom_bytecode_input_shape_t* shape = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_input_projection_allocate_row(
      projection->arena, &projection->shape_index, &projection->shapes, hash,
      &probe, length, (void**)&shape));
  memcpy(shape, key, sizeof(*key));
  memcpy(shape->dimensions, source->dimensions,
         source->rank * sizeof(*source->dimensions));
  loom_intern_table_insert(&projection->shape_index, probe.slot, hash, index);
  *out_shape_id = index + 1;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_input_projection_static(
    loom_bytecode_input_projection_t* projection, uint32_t type_index,
    uint32_t* out_shape_id) {
  if (loom_ordinal_map_lookup(&projection->static_types, 0, type_index,
                              out_shape_id)) {
    return iree_ok_status();
  }
  const loom_bytecode_table_entry_metadata_t* entry =
      &projection->type_entries[type_index];
  const loom_bytecode_reader_section_t* section =
      projection->view->sections.types;
  const iree_const_byte_span_t bytes = iree_make_const_byte_span(
      section->bytes.data + (entry->entry_offset - section->absolute_offset),
      entry->entry_length);
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(projection->arena);
  loom_bytecode_type_plan_entry_t plan;
  loom_bytecode_type_fact_t* fact = NULL;
  iree_status_t status = loom_bytecode_type_plan_decode_indexed_entry(
      projection->decoder, projection->context, projection->view,
      projection->arena, type_index, bytes, entry->entry_offset, &plan, &fact);
  loom_type_kind_t kind = LOOM_TYPE_NONE;
  loom_bytecode_scoped_shape_t shape = {0};
  if (iree_status_is_ok(status) && !fact &&
      loom_type_has_dimensions(plan.direct_type)) {
    const loom_type_t type = plan.direct_type;
    kind = loom_type_kind(type);
    shape.rank = loom_type_rank(type);
    shape.element_type = loom_type_element_type(type);
    shape.access_alignment =
        kind == LOOM_TYPE_VIEW ? loom_type_view_alignment_override(type) : 0;
    shape.attachment = type.encoding_id
                           ? LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC
                           : LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE;
    shape.encoding_ordinal = type.encoding_id;
    for (uint8_t axis = 0; axis < shape.rank; ++axis) {
      const uint64_t dimension = loom_type_dim(type, axis);
      if (loom_dim_is_dynamic(dimension)) {
        shape.dynamic_axes |= (uint16_t)(1u << axis);
      } else {
        shape.dimensions[axis] = loom_dim_static_size(dimension);
      }
    }
  }
  iree_arena_checkpoint_restore(&checkpoint);
  IREE_RETURN_IF_ERROR(status);
  *out_shape_id = 0;
  if (kind != LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_bytecode_input_projection_intern(
        projection, kind, &shape, out_shape_id));
  }
  return loom_ordinal_map_insert(&projection->static_types, 0, type_index,
                                 *out_shape_id);
}

static uint32_t loom_bytecode_input_projection_binding(
    const loom_bytecode_input_projection_t* projection, uint64_t index) {
  const uint32_t* segment = loom_segmented_storage_const_segment(
      &projection->bindings,
      (uint32_t)(index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY));
  return segment[index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY];
}

static iree_status_t loom_bytecode_input_projection_record(
    loom_bytecode_input_projection_t* projection,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_validation_scope_t* scope,
    uint32_t* out_shape_id) {
  *out_shape_id = 0;
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint8_t kind_byte = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(projection->decoder, cursor, &kind_byte));
  loom_type_kind_t kind = LOOM_TYPE_NONE;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_decode_kind(
      projection->decoder, kind_byte, offset, &kind));
  if (kind == LOOM_TYPE_PARAMETERIZED) {
    loom_bytecode_attribute_validator_t validator = {
        .decoder = projection->decoder,
        .context = projection->context,
        .module_view = projection->view,
    };
    return loom_bytecode_attribute_validate_complete_type(
        &validator, cursor, scope, projection->binding_count);
  }
  if (kind == LOOM_TYPE_GROUP || kind == LOOM_TYPE_TENSOR ||
      kind == LOOM_TYPE_TILE || kind == LOOM_TYPE_VECTOR ||
      kind == LOOM_TYPE_VIEW) {
    loom_bytecode_scoped_shape_t shape;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_read_scoped_shape(
        projection->decoder, cursor, kind, scope->value_count,
        projection->view->encodings.count, &shape));
    return loom_bytecode_input_projection_intern(projection, kind, &shape,
                                                 out_shape_id);
  }
  if (kind != LOOM_TYPE_FUNCTION && kind != LOOM_TYPE_DIALECT &&
      kind != LOOM_TYPE_REGISTER) {
    return loom_bytecode_type_emit_invalid_scoped_record(
        projection->decoder, cursor, IREE_SV("invalid_scoped_type_kind"));
  }
  loom_bytecode_structural_type_plan_t plan;
  uint64_t payload[2];
  IREE_RETURN_IF_ERROR(loom_bytecode_type_read_structural_header(
      projection->decoder, cursor, kind, 0, projection->view->strings.count,
      &plan, payload));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_count_bytes(
      projection->decoder, cursor, IREE_SV("scoped_type_children"),
      plan.dependency_count, /*minimum_entry_length=*/1,
      /*minimum_trailing_length=*/0, offset));
  for (uint32_t i = 0; i < plan.dependency_count; ++i) {
    uint64_t reference = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(projection->decoder,
                                                           cursor, &reference));
    IREE_RETURN_IF_ERROR(loom_bytecode_type_validate_completed_reference(
        projection->decoder, cursor, reference, projection->view->types.count,
        projection->binding_count));
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_input_projection_read(
    loom_bytecode_input_projection_t* projection,
    loom_bytecode_reader_cursor_t* cursor, uint64_t source_reference,
    const loom_bytecode_attribute_ssa_validation_scope_t* scope,
    uint32_t* out_shape_id) {
  loom_bytecode_reader_cursor_t records;
  uint64_t count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_open_scoped_records(
      projection->decoder, cursor, source_reference, &records, &count));
  if (!records.cursor.length) {
    return loom_bytecode_input_projection_static(
        projection, (uint32_t)(source_reference >> 1), out_shape_id);
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t shape_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_input_projection_record(
        projection, &records, scope, &shape_id));
    const uint32_t segment_index =
        (uint32_t)(projection->binding_count /
                   LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
    uint32_t* segment = NULL;
    if (segment_index == projection->bindings.segment_count) {
      IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
          &projection->bindings, projection->arena, (void**)&segment));
    } else {
      segment =
          loom_segmented_storage_segment(&projection->bindings, segment_index);
    }
    segment[projection->binding_count % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY] =
        shape_id;
    ++projection->binding_count;
  }
  uint64_t root = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_close_scoped_records(
      projection->decoder, &records, projection->binding_count, &root));
  *out_shape_id = loom_bytecode_input_projection_binding(projection, root);
  return iree_ok_status();
}

iree_status_t loom_bytecode_input_projection_add_argument(
    loom_bytecode_input_projection_t* projection, uint16_t ordinal,
    uint32_t shape_id) {
  if (!shape_id) {
    return iree_ok_status();
  }
  const uint32_t index = projection->argument_count;
  const uint32_t segment_index = index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY;
  loom_bytecode_input_argument_t* segment = NULL;
  if (segment_index == projection->arguments.segment_count) {
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &projection->arguments, projection->arena, (void**)&segment));
  } else {
    segment =
        loom_segmented_storage_segment(&projection->arguments, segment_index);
  }
  segment[index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY] =
      (loom_bytecode_input_argument_t){.ordinal = ordinal,
                                       .shape_id = shape_id};
  ++projection->argument_count;
  return iree_ok_status();
}

static const loom_bytecode_input_argument_t*
loom_bytecode_input_projection_argument(
    const loom_bytecode_input_projection_t* projection, uint32_t index) {
  const loom_bytecode_input_argument_t* segment =
      loom_segmented_storage_const_segment(
          &projection->arguments, index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
  return &segment[index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY];
}

static const loom_bytecode_input_signature_t*
loom_bytecode_input_projection_signature(
    const loom_bytecode_input_projection_t* projection, uint32_t ordinal) {
  if (!ordinal) {
    return NULL;
  }
  const uint32_t index = ordinal - 1;
  const void* const* segment = loom_segmented_storage_const_segment(
      &projection->signatures, index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
  return segment[index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY];
}

static bool loom_bytecode_input_signature_equal(const void* opaque,
                                                uint32_t index) {
  const loom_bytecode_input_projection_t* projection = opaque;
  const loom_bytecode_input_signature_t* signature =
      loom_bytecode_input_projection_signature(projection, index + 1);
  if (signature->count != projection->argument_count) {
    return false;
  }
  for (uint32_t i = 0; i < signature->count; ++i) {
    const loom_bytecode_input_argument_t* argument =
        loom_bytecode_input_projection_argument(projection, i);
    if (signature->values[i].ordinal != argument->ordinal ||
        signature->values[i].shape_id != argument->shape_id) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_bytecode_input_projection_end_signature(
    loom_bytecode_input_projection_t* projection, uint32_t symbol_ordinal,
    uint32_t* out_ordinal) {
  *out_ordinal = 0;
  const uint32_t symbol_segment_index =
      projection->template_count / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY;
  uint32_t* symbol_segment = NULL;
  if (symbol_segment_index == projection->template_symbols.segment_count) {
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &projection->template_symbols, projection->arena,
        (void**)&symbol_segment));
  } else {
    symbol_segment = loom_segmented_storage_segment(
        &projection->template_symbols, symbol_segment_index);
  }
  symbol_segment[projection->template_count %
                 LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY] = symbol_ordinal;
  ++projection->template_count;
  if (!projection->argument_count) {
    return iree_ok_status();
  }
  uint32_t hash = loom_structural_hash_initialize();
  for (uint32_t i = 0; i < projection->argument_count; ++i) {
    const loom_bytecode_input_argument_t* argument =
        loom_bytecode_input_projection_argument(projection, i);
    hash = loom_structural_hash_mix_u32(hash, argument->ordinal);
    hash = loom_structural_hash_mix_u32(hash, argument->shape_id);
  }
  hash = loom_structural_hash_finalize(hash);
  loom_intern_probe_t probe =
      loom_intern_table_probe(&projection->signature_index, hash,
                              loom_bytecode_input_signature_equal, projection);
  if (probe.index != UINT32_MAX) {
    *out_ordinal = probe.index + 1;
    return iree_ok_status();
  }
  const uint32_t index = (uint32_t)projection->signature_index.count;
  loom_bytecode_input_signature_t* signature = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_input_projection_allocate_row(
      projection->arena, &projection->signature_index, &projection->signatures,
      hash, &probe,
      sizeof(*signature) +
          projection->argument_count * sizeof(*signature->values),
      (void**)&signature));
  signature->count = projection->argument_count;
  for (uint32_t i = 0; i < signature->count; ++i) {
    signature->values[i] =
        *loom_bytecode_input_projection_argument(projection, i);
  }
  loom_intern_table_insert(&projection->signature_index, probe.slot, hash,
                           index);
  *out_ordinal = index + 1;
  return iree_ok_status();
}

// Emits in source input/axis order. A NULL output counts the exact allocation
// before the same producer fills it; no provisional payload copies survive.
static uint32_t loom_bytecode_input_projection_requirements(
    const loom_bytecode_input_projection_t* projection, uint32_t signature_id,
    uint32_t family_id, loom_bytecode_input_requirement_t* output) {
  if (!signature_id || signature_id == family_id) {
    return 0;
  }
  const loom_bytecode_input_signature_t* signature =
      loom_bytecode_input_projection_signature(projection, signature_id);
  const loom_bytecode_input_signature_t* family =
      loom_bytecode_input_projection_signature(projection, family_id);
  uint32_t count = 0, family_position = 0;
  for (uint32_t i = 0; i < signature->count; ++i) {
    const loom_bytecode_input_argument_t* argument = &signature->values[i];
    const loom_bytecode_input_shape_t* shape =
        loom_bytecode_input_projection_shape(projection, argument->shape_id);
    while (family && family_position < family->count &&
           family->values[family_position].ordinal < argument->ordinal) {
      ++family_position;
    }
    const loom_bytecode_input_shape_t* inherited = NULL;
    if (family && family_position < family->count &&
        family->values[family_position].ordinal == argument->ordinal) {
      inherited = loom_bytecode_input_projection_shape(
          projection, family->values[family_position].shape_id);
      if (shape == inherited) {
        continue;
      }
      if (shape->kind != inherited->kind ||
          shape->element_type != inherited->element_type ||
          shape->rank != inherited->rank ||
          shape->access_alignment != inherited->access_alignment) {
        inherited = NULL;
      }
    }
    for (uint8_t axis = 0; axis < shape->rank; ++axis) {
      const bool dynamic = (shape->dynamic_axes & (1u << axis)) != 0;
      const uint64_t expected = shape->dimensions[axis];
      if ((dynamic && !expected) ||
          (inherited && expected == inherited->dimensions[axis] &&
           ((shape->dynamic_axes ^ inherited->dynamic_axes) & (1u << axis)) ==
               0)) {
        continue;
      }
      if (output) {
        output[count] = (loom_bytecode_input_requirement_t){
            .argument = (uint16_t)argument->ordinal,
            .axis = axis,
            .expected_is_argument = dynamic,
            .expected = dynamic ? expected - 1 : expected,
        };
      }
      ++count;
    }
    if (shape->kind == LOOM_TYPE_GROUP || shape->kind == LOOM_TYPE_VECTOR ||
        (inherited && shape->attachment == inherited->attachment &&
         shape->encoding_ordinal == inherited->encoding_ordinal)) {
      continue;
    }
    if (output) {
      output[count] = (loom_bytecode_input_requirement_t){
          .argument = (uint16_t)argument->ordinal,
          .axis = UINT8_MAX,
          .expected_is_argument =
              shape->attachment == LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA,
          .expected = shape->encoding_ordinal,
      };
    }
    ++count;
  }
  return count;
}

typedef struct loom_bytecode_input_contract_key_t {
  // Source and family signature ordinals packed into their 16-bit domains.
  uint32_t signatures;
  // Retained contract ordinal plus one, including zero for an empty difference.
  uint32_t contract;
} loom_bytecode_input_contract_key_t;

typedef struct loom_bytecode_input_contract_probe_t {
  // Stable previously classified source/family pairs.
  const loom_segmented_storage_t* keys;
  // Source/family identity pair being probed.
  uint32_t signatures;
} loom_bytecode_input_contract_probe_t;

static bool loom_bytecode_input_contract_equal(const void* opaque,
                                               uint32_t index) {
  const loom_bytecode_input_contract_probe_t* probe = opaque;
  const loom_bytecode_input_contract_key_t* segment =
      loom_segmented_storage_const_segment(
          probe->keys, index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
  return segment[index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY].signatures ==
         probe->signatures;
}

iree_status_t loom_bytecode_input_projection_finish_module(
    loom_bytecode_input_projection_t* projection,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_module_metadata_t* metadata) {
  loom_intern_table_t contracts_by_signature_pair = {0};
  loom_segmented_storage_t keys;
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY *
          sizeof(loom_bytecode_input_contract_key_t),
      iree_alignof(loom_bytecode_input_contract_key_t), &keys);
  loom_segmented_storage_t contracts;
  loom_segmented_storage_initialize(
      LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY *
          sizeof(const loom_bytecode_input_contract_t*),
      iree_alignof(const loom_bytecode_input_contract_t*), &contracts);
  uint32_t contract_count = 0;
  // Providers consume source family signatures before family slots become
  // retained contract ordinals. A family reference must name a declaration.
  for (unsigned pass = 0; pass < 2; ++pass) {
    for (uint32_t entry = 0; entry < projection->template_count; ++entry) {
      const uint32_t* symbol_segment = loom_segmented_storage_const_segment(
          &projection->template_symbols,
          entry / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
      const uint32_t i =
          symbol_segment[entry % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY];
      loom_bytecode_symbol_metadata_t* symbol = &metadata->symbols[i];
      const bool is_family = symbol->kind == LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL;
      const bool is_provider =
          symbol->kind == LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF ||
          symbol->kind == LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL;
      if (pass == 0 ? !is_provider : !is_family) {
        continue;
      }
      uint32_t family_id = 0;
      if (is_provider) {
        const loom_bytecode_symbol_metadata_t* family =
            &metadata->symbols[symbol->template_family_symbol_ordinal];
        if (family->kind != LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL) {
          return loom_bytecode_reader_emit_invalid_field(
              projection->decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), i,
              IREE_SV("template_family_symbol_ordinal"), symbol->entry_offset,
              IREE_SV("template_family_requires_a_declaration"));
        }
        family_id = family->input_contract_ordinal_plus_one;
      }
      const uint32_t signature_id = symbol->input_contract_ordinal_plus_one;
      if (!signature_id || signature_id == family_id) {
        symbol->input_contract_ordinal_plus_one = 0;
        continue;
      }
      uint32_t ordinal = 0;
      // A module contains fewer than 65536 symbols, so its nonempty
      // canonical signatures and the zero sentinel each fit in 16 bits.
      const loom_bytecode_input_contract_probe_t key = {
          .keys = &keys,
          .signatures = (signature_id << 16) | family_id,
      };
      const uint32_t hash =
          loom_structural_hash_finalize(loom_structural_hash_mix_u32(
              loom_structural_hash_initialize(), key.signatures));
      loom_intern_probe_t probe =
          loom_intern_table_probe(&contracts_by_signature_pair, hash,
                                  loom_bytecode_input_contract_equal, &key);
      if (probe.index == UINT32_MAX) {
        const uint32_t count = loom_bytecode_input_projection_requirements(
            projection, signature_id, family_id, NULL);
        if (count) {
          loom_bytecode_input_contract_t* contract = NULL;
          IREE_RETURN_IF_ERROR(iree_arena_allocate(
              retained_arena,
              sizeof(*contract) + count * sizeof(*contract->values),
              (void**)&contract));
          contract->count = count;
          contract->reserved = 0;
          loom_bytecode_input_projection_requirements(
              projection, signature_id, family_id, contract->values);
          const uint32_t segment_index =
              contract_count / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY;
          const loom_bytecode_input_contract_t** segment = NULL;
          if (segment_index == contracts.segment_count) {
            IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
                &contracts, projection->arena, (void**)&segment));
          } else {
            segment = loom_segmented_storage_segment(&contracts, segment_index);
          }
          segment[contract_count % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY] =
              contract;
          ordinal = ++contract_count;
        }
        if (!contracts_by_signature_pair.capacity) {
          IREE_RETURN_IF_ERROR(loom_intern_table_initialize(
              projection->arena, 32, &contracts_by_signature_pair));
          probe.slot = loom_intern_table_find_empty_slot(
              &contracts_by_signature_pair, hash);
        } else {
          IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
              projection->arena, &contracts_by_signature_pair, hash, 1,
              &probe.slot));
        }
        const uint32_t index = (uint32_t)contracts_by_signature_pair.count;
        const uint32_t segment_index =
            index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY;
        loom_bytecode_input_contract_key_t* segment = NULL;
        if (segment_index == keys.segment_count) {
          IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
              &keys, projection->arena, (void**)&segment));
        } else {
          segment = loom_segmented_storage_segment(&keys, segment_index);
        }
        segment[index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY] =
            (loom_bytecode_input_contract_key_t){.signatures = key.signatures,
                                                 .contract = ordinal};
        loom_intern_table_insert(&contracts_by_signature_pair, probe.slot, hash,
                                 index);
      } else {
        const loom_bytecode_input_contract_key_t* segment =
            loom_segmented_storage_const_segment(
                &keys, probe.index / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
        ordinal = segment[probe.index % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY]
                      .contract;
      }
      symbol->input_contract_ordinal_plus_one = ordinal;
    }
  }
  if (contract_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        retained_arena, contract_count, sizeof(*metadata->input_contracts),
        (void**)&metadata->input_contracts));
    for (uint32_t i = 0; i < contract_count; ++i) {
      const loom_bytecode_input_contract_t* const* segment =
          loom_segmented_storage_const_segment(
              &contracts, i / LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY);
      metadata->input_contracts[i] =
          segment[i % LOOM_BYTECODE_INPUT_SEGMENT_CAPACITY];
    }
  }
  metadata->input_contract_count = contract_count;
  return iree_ok_status();
}
