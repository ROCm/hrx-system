// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_tables.h"

#include <string.h>

#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/selected_attribute.h"
#include "loom/format/bytecode/reader/type.h"
#include "loom/format/bytecode/reader/type_plan.h"
#include "loom/format/bytecode/reader/type_validator.h"

// Construction form of one decoded type before its child identities complete.
typedef enum loom_bytecode_selected_type_form_e {
  LOOM_BYTECODE_SELECTED_TYPE_DIRECT = 0,
  LOOM_BYTECODE_SELECTED_TYPE_STRUCTURAL = 1,
  LOOM_BYTECODE_SELECTED_TYPE_PARAMETERIZED = 2,
} loom_bytecode_selected_type_form_t;

// Decoded payloads remain private to the current dependency root. Child frames
// write directly into their parent-owned slots before canonical publication.
typedef struct loom_bytecode_selected_table_payload_t {
  union {
    // Encoding metadata and scratch-owned parameter slots.
    loom_encoding_t encoding;
    // Location metadata and its directly addressable child slot.
    struct {
      // Decoded location with scratch-owned aggregate payloads.
      loom_location_entry_t value;
      // Scratch-owned child identity for a tagged location.
      loom_location_id_t* child;
    } location;
    // Type construction metadata with directly addressable dependency slots.
    struct {
      // Selects the direct, structural, or parameterized construction payload.
      loom_bytecode_selected_type_form_t form;
      union {
        // By-value type whose optional encoding is projected separately.
        struct {
          // Decoded type, including scratch-owned overflow dimensions.
          loom_type_t value;
          // Scratch-owned static encoding identity, or NULL without one.
          uint32_t* encoding_id;
        } direct;
        // Canonical parent metadata and its completed child identities.
        struct {
          // Parent construction metadata independent of source/native IDs.
          loom_bytecode_structural_type_plan_t plan;
          // Validated parent payload prefix borrowed until construction.
          const loom_bytecode_structural_type_fact_t* fact;
          // Scratch-owned destination slots for projected child types.
          loom_type_id_t* children;
        } structural;
        // Descriptor-backed type whose aggregate slots survive child decoding.
        struct {
          // Context-owned type family and parameter contract.
          const loom_parameterized_type_descriptor_t* descriptor;
          // Scratch-owned parameter slots containing final projected IDs.
          loom_attribute_t* parameters;
        } parameterized;
      };
    } type;
  };
} loom_bytecode_selected_table_payload_t;

// Only suspended entries retain a record across dependency construction.
typedef struct loom_bytecode_selected_table_pending_t {
  // Scratch boundary preceding this entry and every descendant payload.
  iree_arena_checkpoint_t checkpoint;
  // Decoded value whose dependency destinations all reside in scratch.
  loom_bytecode_selected_table_payload_t payload;
} loom_bytecode_selected_table_pending_t;

// One stable entry on the explicit shared-table materialization stack.
typedef struct loom_bytecode_selected_table_frame_t {
  // Shared-table domain containing the source entry.
  loom_bytecode_selected_table_kind_t table_kind;
  // Source-table ordinal of the entry.
  uint32_t source_ordinal;
  // Parent-owned slot receiving the final projected identity.
  uint32_t* destination;
  // Decoded parent payload retained until its child frames complete.
  loom_bytecode_selected_table_pending_t* pending;
} loom_bytecode_selected_table_frame_t;

// Bounded chunks keep frontier growth in pooled arena storage. Frame addresses
// remain stable while children are pushed, and empty chunks serve later roots.
typedef struct loom_bytecode_selected_table_chunk_t {
  // Previous chunk in stack order, or NULL for the first chunk.
  struct loom_bytecode_selected_table_chunk_t* previous;
  // Allocated successor reused when this chunk fills again.
  struct loom_bytecode_selected_table_chunk_t* next;
  // Number of live frames when this chunk is active.
  iree_host_size_t count;
  // Number of trailing frame slots.
  iree_host_size_t capacity;
  // Stable frames in push order.
  loom_bytecode_selected_table_frame_t frames[];
} loom_bytecode_selected_table_chunk_t;

static loom_bytecode_selected_projection_domain_t
loom_bytecode_selected_table_projection_domain(
    loom_bytecode_selected_table_kind_t table_kind) {
  switch (table_kind) {
    case LOOM_BYTECODE_SELECTED_TABLE_ENCODING:
      return LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING;
    case LOOM_BYTECODE_SELECTED_TABLE_TYPE:
      return LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE;
    case LOOM_BYTECODE_SELECTED_TABLE_LOCATION:
      return LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION;
  }
  IREE_ASSERT_UNREACHABLE("selected table kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_bytecode_selected_table_push(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_kind_t table_kind, uint32_t source_ordinal,
    uint32_t* destination) {
  loom_bytecode_selected_table_chunk_t* chunk = materializer->worklist.current;
  if (chunk == NULL || chunk->count == chunk->capacity) {
    if (chunk != NULL && chunk->next != NULL) {
      chunk = chunk->next;
    } else {
      // Small roots retain only four slots. Deeper frontiers grow chunk sizes
      // geometrically up to 64 slots without abandoning or copying frames.
      const iree_host_size_t maximum_allocation_size =
          iree_arena_block_pool_max_allocation_size(
              materializer->retained_arena.block_pool);
      iree_host_size_t maximum_capacity = 1;
      if (maximum_allocation_size > sizeof(*chunk)) {
        maximum_capacity = iree_max((iree_host_size_t)1,
                                    (maximum_allocation_size - sizeof(*chunk)) /
                                        sizeof(chunk->frames[0]));
      }
      maximum_capacity = iree_min(maximum_capacity, (iree_host_size_t)64);
      const iree_host_size_t capacity =
          iree_min(chunk ? chunk->capacity * 2 : 4, maximum_capacity);
      loom_bytecode_selected_table_chunk_t* next = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          &materializer->retained_arena,
          sizeof(*next) + capacity * sizeof(next->frames[0]), (void**)&next));
      *next = (loom_bytecode_selected_table_chunk_t){
          .previous = chunk,
          .capacity = capacity,
      };
      if (chunk != NULL) {
        chunk->next = next;
      } else {
        materializer->worklist.first = next;
      }
      chunk = next;
    }
    chunk->count = 0;
    materializer->worklist.current = chunk;
  }
  chunk->frames[chunk->count++] = (loom_bytecode_selected_table_frame_t){
      .table_kind = table_kind,
      .source_ordinal = source_ordinal,
      .destination = destination,
  };
  ++materializer->worklist.count;
  return iree_ok_status();
}

static void loom_bytecode_selected_table_pop(
    loom_bytecode_selected_table_materializer_t* materializer) {
  loom_bytecode_selected_table_chunk_t* chunk = materializer->worklist.current;
  --chunk->count;
  --materializer->worklist.count;
  if (chunk->count == 0 && chunk->previous != NULL) {
    materializer->worklist.current = chunk->previous;
  }
}

static iree_const_byte_span_t loom_bytecode_selected_table_entry_span(
    const loom_bytecode_selected_table_materializer_t* materializer,
    const loom_bytecode_table_entry_metadata_t* entry) {
  IREE_ASSERT(entry->entry_offset <= materializer->bytecode.data_length);
  IREE_ASSERT(entry->entry_length <=
              materializer->bytecode.data_length - entry->entry_offset);
  return iree_make_const_byte_span(
      materializer->bytecode.data + (iree_host_size_t)entry->entry_offset,
      (iree_host_size_t)entry->entry_length);
}

static iree_status_t loom_bytecode_selected_table_project_source(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, loom_source_id_t* out_target_source_id) {
  IREE_ASSERT(source_ordinal < materializer->metadata->sources.count);
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE, source_ordinal,
          &target_id)) {
    *out_target_source_id = (loom_source_id_t)target_id;
    return iree_ok_status();
  }
  const iree_string_view_t source_name =
      materializer->metadata->sources.values[source_ordinal];
  loom_source_id_t target_source_id = LOOM_SOURCE_ID_INVALID;
  if (materializer->projected_source_count ==
      materializer->output_module->sources.count) {
    IREE_RETURN_IF_ERROR(loom_module_append_source(
        materializer->output_module, source_name, &target_source_id));
  } else {
    IREE_RETURN_IF_ERROR(loom_module_register_source(
        materializer->output_module, source_name, &target_source_id));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_projection_insert(
      &materializer->projection,
      LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE, source_ordinal,
      target_source_id));
  ++materializer->projected_source_count;
  *out_target_source_id = target_source_id;
  return iree_ok_status();
}

void loom_bytecode_selected_table_materializer_initialize(
    loom_bytecode_reader_decoder_t* decoder, iree_const_byte_span_t bytecode,
    loom_context_t* context, const loom_bytecode_module_metadata_t* metadata,
    iree_arena_allocator_t* scratch_arena, loom_module_t* output_module,
    loom_bytecode_selected_symbol_resolver_t symbol_resolver,
    loom_bytecode_selected_table_materializer_t* out_materializer) {
  *out_materializer = (loom_bytecode_selected_table_materializer_t){
      .decoder = decoder,
      .bytecode = bytecode,
      .context = context,
      .metadata = metadata,
      .scratch_arena = scratch_arena,
      .output_module = output_module,
      .symbol_resolver = symbol_resolver,
  };
  iree_arena_initialize(scratch_arena->block_pool,
                        &out_materializer->retained_arena);
  loom_bytecode_selected_projection_initialize(
      &out_materializer->retained_arena, &out_materializer->projection);
}

void loom_bytecode_selected_table_materializer_deinitialize(
    loom_bytecode_selected_table_materializer_t* materializer) {
  iree_arena_deinitialize(&materializer->retained_arena);
  memset(materializer, 0, sizeof(*materializer));
}

iree_status_t loom_bytecode_selected_table_bind_symbol(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, uint16_t target_symbol_id) {
  IREE_ASSERT(source_name_ordinal < materializer->metadata->strings.count);
  return loom_bytecode_selected_projection_insert(
      &materializer->projection,
      LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME, source_name_ordinal,
      target_symbol_id);
}

iree_status_t loom_bytecode_selected_table_intern_string(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_string_ordinal, loom_string_id_t* out_target_string_id) {
  IREE_ASSERT(source_string_ordinal < materializer->metadata->strings.count);
  return loom_module_intern_string(
      materializer->output_module,
      materializer->metadata->strings.values[source_string_ordinal],
      out_target_string_id);
}

bool loom_bytecode_selected_table_lookup_symbol(
    const loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, loom_symbol_ref_t* out_target_symbol_ref) {
  uint32_t target_symbol_id = 0;
  if (!loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME,
          source_name_ordinal, &target_symbol_id)) {
    return false;
  }
  *out_target_symbol_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = (uint16_t)target_symbol_id,
  };
  return true;
}

iree_status_t loom_bytecode_selected_table_resolve_symbol(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, loom_symbol_ref_t* out_target_symbol_ref,
    bool* out_found) {
  *out_target_symbol_ref = loom_symbol_ref_null();
  *out_found = loom_bytecode_selected_table_lookup_symbol(
      materializer, source_name_ordinal, out_target_symbol_ref);
  if (*out_found || materializer->symbol_resolver.fn == NULL) {
    return iree_ok_status();
  }

  uint32_t source_symbol_ordinal = UINT32_MAX;
  if (!loom_bytecode_module_metadata_lookup_symbol_ordinal(
          materializer->metadata, source_name_ordinal,
          &source_symbol_ordinal)) {
    return iree_ok_status();
  }
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      materializer->symbol_resolver.fn(materializer->symbol_resolver.user_data,
                                       source_symbol_ordinal, &target_ref));
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= materializer->output_module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "external bytecode symbol resolver returned an "
                            "invalid output-module reference");
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_bind_symbol(
      materializer, source_name_ordinal, target_ref.symbol_id));
  *out_target_symbol_ref = target_ref;
  *out_found = true;
  return iree_ok_status();
}

iree_status_t loom_bytecode_selected_table_project_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint16_t source_encoding_id, uint32_t* out_target_encoding_id) {
  IREE_ASSERT(source_encoding_id > 0);
  const uint32_t source_ordinal = source_encoding_id - 1;
  IREE_ASSERT(source_ordinal < materializer->metadata->encodings.count);
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING, source_ordinal,
          &target_id)) {
    *out_target_encoding_id = target_id;
    return iree_ok_status();
  }
  *out_target_encoding_id = 0;
  return loom_bytecode_selected_table_push(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_ENCODING, source_ordinal,
      out_target_encoding_id);
}

iree_status_t loom_bytecode_selected_table_project_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_type_id_t source_type_id, loom_type_id_t* out_target_type_id) {
  IREE_ASSERT(source_type_id < materializer->metadata->types.count);
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE, source_type_id,
          &target_id)) {
    *out_target_type_id = (loom_type_id_t)target_id;
    return iree_ok_status();
  }
  *out_target_type_id = LOOM_TYPE_ID_INVALID;
  return loom_bytecode_selected_table_push(materializer,
                                           LOOM_BYTECODE_SELECTED_TABLE_TYPE,
                                           source_type_id, out_target_type_id);
}

iree_status_t loom_bytecode_selected_table_project_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id) {
  IREE_ASSERT(source_location_id < materializer->metadata->locations.count);
  if (source_location_id == LOOM_LOCATION_UNKNOWN) {
    *out_target_location_id = LOOM_LOCATION_UNKNOWN;
    return iree_ok_status();
  }
  uint32_t target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION, source_location_id,
          &target_id)) {
    *out_target_location_id = (loom_location_id_t)target_id;
    return iree_ok_status();
  }
  *out_target_location_id = LOOM_LOCATION_UNKNOWN;
  return loom_bytecode_selected_table_push(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_LOCATION, source_location_id,
      out_target_location_id);
}

static iree_status_t loom_bytecode_selected_table_prepare_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, loom_encoding_t* out_encoding) {
  const loom_bytecode_encoding_metadata_t* metadata =
      &materializer->metadata->encodings.entries[source_ordinal];
  const loom_bytecode_table_entry_metadata_t entry_metadata = {
      .entry_offset = metadata->entry_offset,
      .entry_length = metadata->entry_length,
  };
  const iree_const_byte_span_t entry_bytes =
      loom_bytecode_selected_table_entry_span(materializer, &entry_metadata);
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      entry_bytes.data, entry_bytes.data_length, metadata->entry_offset,
      IREE_SV("ENCODINGS"), &cursor);

  uint64_t unused_family_index = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, &cursor, &unused_family_index));
  uint64_t alias_plus_one = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, &cursor, &alias_plus_one));
  uint64_t parameter_count = 0;
  const uint64_t parameter_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, &cursor, &parameter_count));
  if (parameter_count > UINT8_MAX || parameter_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        materializer->decoder, IREE_SV("encoding_params"), parameter_count,
        UINT8_MAX, parameter_count_offset);
  }

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
      materializer, metadata->name_string_index, &target_name_id));
  loom_string_id_t target_alias_id = LOOM_STRING_ID_INVALID;
  if (alias_plus_one > 0) {
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
        materializer, (uint32_t)(alias_plus_one - 1), &target_alias_id));
  }

  loom_named_attr_t* parameters = NULL;
  if (parameter_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        materializer->scratch_arena, (iree_host_size_t)parameter_count,
        sizeof(*parameters), (void**)&parameters));
  }
  for (uint64_t parameter_index = 0; parameter_index < parameter_count;
       ++parameter_index) {
    uint64_t source_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        materializer->decoder, &cursor, &source_name_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
        materializer, (uint32_t)source_name_id,
        &parameters[parameter_index].name_id));
    parameters[parameter_index].reserved = 0;
    loom_bytecode_attr_kind_t value_kind = LOOM_BYTECODE_ATTR_I64;
    IREE_RETURN_IF_ERROR(loom_bytecode_attribute_read_kind(
        materializer->decoder, &cursor, &value_kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_decode_named(
        materializer, &cursor, /*descriptor=*/NULL, value_kind,
        &parameters[parameter_index].value,
        materializer->metadata->types.count));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_expect_empty(
      materializer->decoder, &cursor, IREE_SV("encoding_entry")));
  *out_encoding = (loom_encoding_t){
      .name_id = target_name_id,
      .alias_id = target_alias_id,
      .attribute_count = (uint8_t)parameter_count,
      .attributes = parameters,
  };
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_project_type_dependencies(
    loom_bytecode_selected_table_materializer_t* materializer,
    const loom_type_id_t* source_type_ids, iree_host_size_t type_count,
    loom_type_id_t** out_target_type_ids) {
  *out_target_type_ids = NULL;
  if (type_count == 0) {
    return iree_ok_status();
  }
  loom_type_id_t* target_type_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      materializer->scratch_arena, type_count, sizeof(*target_type_ids),
      (void**)&target_type_ids));
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_type(
        materializer, source_type_ids[i], &target_type_ids[i]));
  }
  *out_target_type_ids = target_type_ids;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_prepare_structural_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    const loom_bytecode_type_plan_entry_t* entry,
    const loom_bytecode_type_fact_t* fact,
    loom_bytecode_selected_table_payload_t* payload) {
  const loom_bytecode_structural_type_fact_t* structural_fact =
      (const loom_bytecode_structural_type_fact_t*)fact;
  loom_bytecode_structural_type_plan_t plan = entry->structural;
  loom_type_id_t* target_type_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_type_dependencies(
      materializer, structural_fact->type_ids, plan.dependency_count,
      &target_type_ids));
  if (fact->kind == LOOM_TYPE_DIALECT) {
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
        materializer, plan.name_id, &plan.name_id));
  }
  payload->type.form = LOOM_BYTECODE_SELECTED_TYPE_STRUCTURAL;
  payload->type.structural.plan = plan;
  payload->type.structural.fact = structural_fact;
  payload->type.structural.children = target_type_ids;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_prepare_type_plan(
    loom_bytecode_selected_table_materializer_t* materializer,
    const loom_bytecode_type_plan_entry_t* plan_entry,
    const loom_bytecode_type_fact_t* fact,
    loom_bytecode_selected_table_payload_t* payload) {
  if (fact != NULL) {
    return loom_bytecode_selected_table_prepare_structural_type(
        materializer, plan_entry, fact, payload);
  }

  payload->type.form = LOOM_BYTECODE_SELECTED_TYPE_DIRECT;
  loom_type_t type = plan_entry->direct_type;
  if (loom_type_kind(type) == LOOM_TYPE_DIALECT) {
    loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_intern_string(
        materializer, loom_type_dialect_name_id(type), &target_name_id));
    type = loom_type_dialect_opaque(target_name_id);
  }
  payload->type.direct.value = type;
  payload->type.direct.encoding_id = NULL;
  if (loom_type_has_static_encoding(type)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        materializer->scratch_arena, sizeof(*payload->type.direct.encoding_id),
        (void**)&payload->type.direct.encoding_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_encoding(
        materializer, type.encoding_id, payload->type.direct.encoding_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_prepare_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, loom_bytecode_selected_table_payload_t* payload) {
  const loom_bytecode_table_entry_metadata_t* metadata =
      &materializer->metadata->types.entries[source_ordinal];
  const iree_const_byte_span_t entry_bytes =
      loom_bytecode_selected_table_entry_span(materializer, metadata);
  // The index validated the entry tag and exact span. Parameterized payloads
  // decode directly into pending slots instead of a byte-span-only recipe.
  if (entry_bytes.data[0] == LOOM_BYTECODE_TYPE_PARAMETERIZED) {
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(
        entry_bytes.data, entry_bytes.data_length, metadata->entry_offset,
        IREE_SV("TYPES"), &cursor);
    cursor.cursor.position = 1;
    payload->type.form = LOOM_BYTECODE_SELECTED_TYPE_PARAMETERIZED;
    IREE_RETURN_IF_ERROR(loom_bytecode_selected_attribute_decode_static_type(
        materializer, &cursor, source_ordinal,
        &payload->type.parameterized.descriptor,
        &payload->type.parameterized.parameters));
    return loom_bytecode_reader_expect_empty(materializer->decoder, &cursor,
                                             IREE_SV("type_entry"));
  }
  loom_bytecode_reader_module_view_t module_view = {
      .strings =
          {
              .values = materializer->metadata->strings.values,
              .count = materializer->metadata->strings.count,
          },
      .types =
          {
              .count = materializer->metadata->types.count,
          },
      .encodings =
          {
              .count = materializer->metadata->encodings.count,
          },
  };
  loom_bytecode_type_plan_entry_t plan_entry = {0};
  loom_bytecode_type_fact_t* fact = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_type_plan_decode_indexed_entry(
      materializer->decoder, materializer->context, &module_view,
      materializer->scratch_arena, (loom_type_id_t)source_ordinal, entry_bytes,
      metadata->entry_offset, &plan_entry, &fact));
  return loom_bytecode_selected_table_prepare_type_plan(
      materializer, &plan_entry, fact, payload);
}

static iree_status_t loom_bytecode_selected_table_finish_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    const loom_bytecode_selected_table_payload_t* payload,
    loom_type_id_t* out_type_id) {
  switch (payload->type.form) {
    case LOOM_BYTECODE_SELECTED_TYPE_DIRECT: {
      loom_type_t type = payload->type.direct.value;
      if (payload->type.direct.encoding_id != NULL) {
        type.encoding_id = (uint16_t)*payload->type.direct.encoding_id;
      }
      return loom_module_intern_topological_type_id(materializer->output_module,
                                                    type, NULL, 0, out_type_id);
    }
    case LOOM_BYTECODE_SELECTED_TYPE_STRUCTURAL:
      return loom_bytecode_type_materialize_structural(
          &payload->type.structural.plan, payload->type.structural.fact,
          payload->type.structural.children, materializer->output_module,
          out_type_id);
    case LOOM_BYTECODE_SELECTED_TYPE_PARAMETERIZED: {
      loom_type_t type = {0};
      const loom_parameterized_type_descriptor_t* descriptor =
          payload->type.parameterized.descriptor;
      return loom_module_make_parameterized_type(
          materializer->output_module, descriptor,
          payload->type.parameterized.parameters, descriptor->parameter_count,
          &type, out_type_id);
    }
  }
  IREE_ASSERT_UNREACHABLE("selected type construction form");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_bytecode_selected_table_read_location_coordinate(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor, uint32_t source_ordinal,
    iree_string_view_t field_name, uint32_t* out_value) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(materializer->decoder, cursor, &value));
  if (value > LOOM_LOCATION_COORDINATE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
        source_ordinal, field_name, offset,
        IREE_SV("file_location_coordinate_exceeds_supported_range"));
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_selected_table_read_source(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor, loom_source_id_t* out_source_id) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t source_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      materializer->decoder, cursor, &source_ordinal));
  if (source_ordinal >= materializer->metadata->sources.count) {
    return loom_bytecode_reader_emit_table_ref(
        materializer->decoder, IREE_SV("SOURCES"), source_ordinal,
        materializer->metadata->sources.count, offset);
  }
  return loom_bytecode_selected_table_project_source(
      materializer, (uint32_t)source_ordinal, out_source_id);
}

static iree_status_t loom_bytecode_selected_table_copy_location_payload(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_entry_t* entry) {
  switch (entry->kind) {
    case LOOM_LOCATION_FUSED: {
      loom_location_id_t* children = NULL;
      if (entry->fused.count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &materializer->output_module->arena, entry->fused.count,
            sizeof(*children), (void**)&children));
        memcpy(children, entry->fused.children,
               (iree_host_size_t)entry->fused.count * sizeof(*children));
      }
      entry->fused.children = children;
      return iree_ok_status();
    }
    case LOOM_LOCATION_OPAQUE: {
      uint8_t* data = NULL;
      if (entry->opaque.data_length > 0) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(&materializer->output_module->arena,
                                entry->opaque.data_length, (void**)&data));
        memcpy(data, entry->opaque.data, entry->opaque.data_length);
      }
      entry->opaque.data = data;
      return iree_ok_status();
    }
    case LOOM_LOCATION_TAGGED: {
      uint8_t* data = NULL;
      if (entry->tagged.data_length > 0) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(&materializer->output_module->arena,
                                entry->tagged.data_length, (void**)&data));
        memcpy(data, entry->tagged.data, entry->tagged.data_length);
      }
      entry->tagged.data = data;
      return iree_ok_status();
    }
    case LOOM_LOCATION_NONE:
    case LOOM_LOCATION_FILE:
      return iree_ok_status();
    case LOOM_LOCATION_COUNT_:
      break;
  }
  IREE_ASSERT_UNREACHABLE("validated location kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_bytecode_selected_table_prepare_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_ordinal, loom_bytecode_selected_table_payload_t* payload) {
  loom_location_entry_t* entry = &payload->location.value;
  const loom_bytecode_table_entry_metadata_t* metadata =
      &materializer->metadata->locations.entries[source_ordinal];
  const iree_const_byte_span_t entry_bytes =
      loom_bytecode_selected_table_entry_span(materializer, metadata);
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      entry_bytes.data, entry_bytes.data_length, metadata->entry_offset,
      IREE_SV("LOCATIONS"), &cursor);
  uint8_t kind = 0;
  uint8_t flags = 0;
  const uint64_t kind_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(materializer->decoder, &cursor, &kind));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(materializer->decoder, &cursor, &flags));
  if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
    return loom_bytecode_reader_emit_invalid_field(
        materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
        source_ordinal, IREE_SV("flags"), kind_offset + 1,
        IREE_SV("location_has_unsupported_flag_bits"));
  }
  *entry = (loom_location_entry_t){
      .kind = (loom_location_kind_t)kind,
      .flags = flags,
  };
  switch ((loom_location_kind_t)kind) {
    case LOOM_LOCATION_NONE:
      IREE_ASSERT(source_ordinal == 0 && flags == 0);
      return iree_ok_status();
    case LOOM_LOCATION_FILE: {
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_read_source(
          materializer, &cursor, &entry->file.source_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("start_line"),
              &entry->file.start_line));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("start_col"),
              &entry->file.start_col));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("end_line"),
              &entry->file.end_line));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_selected_table_read_location_coordinate(
              materializer, &cursor, source_ordinal, IREE_SV("end_col"),
              &entry->file.end_col));
      break;
    }
    case LOOM_LOCATION_FUSED: {
      const uint64_t child_count_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t child_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &child_count));
      if (child_count > LOOM_LOCATION_FUSED_CHILD_COUNT_MAX ||
          child_count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("location_children"), child_count,
            LOOM_LOCATION_FUSED_CHILD_COUNT_MAX, child_count_offset);
      }
      loom_location_id_t* children = NULL;
      if (child_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            materializer->scratch_arena, (iree_host_size_t)child_count,
            sizeof(*children), (void**)&children));
      }
      for (uint64_t child_index = 0; child_index < child_count; ++child_index) {
        const uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t source_child = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            materializer->decoder, &cursor, &source_child));
        if (source_child >= source_ordinal) {
          return loom_bytecode_reader_emit_table_ref(
              materializer->decoder, IREE_SV("LOCATIONS"), source_child,
              source_ordinal, child_offset);
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_location(
            materializer, (loom_location_id_t)source_child,
            &children[child_index]));
      }
      entry->fused.count = (uint32_t)child_count;
      entry->fused.children = children;
      break;
    }
    case LOOM_LOCATION_OPAQUE: {
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_read_source(
          materializer, &cursor, &entry->opaque.source_id));
      uint64_t data_length = 0;
      const uint64_t data_length_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &data_length));
      if (data_length > LOOM_LOCATION_DATA_LENGTH_MAX ||
          data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("opaque_location_data"), data_length,
            LOOM_LOCATION_DATA_LENGTH_MAX, data_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, &cursor, data_length, &span));
      entry->opaque.data_length = (uint32_t)span.data_length;
      entry->opaque.data = span.data;
      break;
    }
    case LOOM_LOCATION_TAGGED: {
      uint64_t tag = 0;
      const uint64_t tag_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &tag));
      if (tag == LOOM_LOCATION_TAG_INVALID || tag > LOOM_LOCATION_TAG_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            materializer->decoder, IREE_SV("LOCATIONS"), IREE_SV("location"),
            source_ordinal, IREE_SV("tag"), tag_offset,
            IREE_SV("tagged location tag must be in [1, 65535]"));
      }
      uint64_t source_child = 0;
      const uint64_t child_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &source_child));
      if (source_child >= source_ordinal) {
        return loom_bytecode_reader_emit_table_ref(
            materializer->decoder, IREE_SV("LOCATIONS"), source_child,
            source_ordinal, child_offset);
      }
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          materializer->scratch_arena, sizeof(*payload->location.child),
          (void**)&payload->location.child));
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_project_location(
          materializer, (loom_location_id_t)source_child,
          payload->location.child));
      uint64_t data_length = 0;
      const uint64_t data_length_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          materializer->decoder, &cursor, &data_length));
      if (data_length > LOOM_LOCATION_DATA_LENGTH_MAX ||
          data_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            materializer->decoder, IREE_SV("tagged_location_data"), data_length,
            LOOM_LOCATION_DATA_LENGTH_MAX, data_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
          materializer->decoder, &cursor, data_length, &span));
      entry->tagged.tag = (loom_location_tag_t)tag;
      entry->tagged.data_length = (uint32_t)span.data_length;
      entry->tagged.data = span.data;
      break;
    }
    default:
      return loom_bytecode_reader_emit_enum_value(
          materializer->decoder, IREE_SV("location_kind"), kind,
          LOOM_LOCATION_COUNT_, kind_offset);
  }
  return loom_bytecode_reader_expect_empty(materializer->decoder, &cursor,
                                           IREE_SV("location_entry"));
}

static iree_status_t loom_bytecode_selected_table_prepare_entry(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_kind_t table_kind, uint32_t source_ordinal,
    loom_bytecode_selected_table_payload_t* payload) {
  switch (table_kind) {
    case LOOM_BYTECODE_SELECTED_TABLE_ENCODING:
      return loom_bytecode_selected_table_prepare_encoding(
          materializer, source_ordinal, &payload->encoding);
    case LOOM_BYTECODE_SELECTED_TABLE_TYPE:
      return loom_bytecode_selected_table_prepare_type(materializer,
                                                       source_ordinal, payload);
    case LOOM_BYTECODE_SELECTED_TABLE_LOCATION:
      return loom_bytecode_selected_table_prepare_location(
          materializer, source_ordinal, payload);
  }
  IREE_ASSERT_UNREACHABLE("selected table kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_bytecode_selected_table_finish_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_payload_t* payload,
    loom_location_id_t* out_location_id) {
  if (payload->location.value.kind == LOOM_LOCATION_TAGGED) {
    payload->location.value.tagged.child = *payload->location.child;
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_copy_location_payload(
      materializer, &payload->location.value));
  return loom_module_add_location(materializer->output_module,
                                  payload->location.value, out_location_id);
}

static iree_status_t loom_bytecode_selected_table_finish_entry(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_kind_t table_kind, uint32_t source_ordinal,
    loom_bytecode_selected_table_payload_t* payload, uint32_t* destination) {
  uint32_t target_id = 0;
  switch (table_kind) {
    case LOOM_BYTECODE_SELECTED_TABLE_ENCODING: {
      uint16_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_module_add_encoding(
          materializer->output_module, &payload->encoding, &encoding_id));
      target_id = encoding_id;
      break;
    }
    case LOOM_BYTECODE_SELECTED_TABLE_TYPE: {
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_finish_type(
          materializer, payload, &target_id));
      break;
    }
    case LOOM_BYTECODE_SELECTED_TABLE_LOCATION: {
      IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_finish_location(
          materializer, payload, &target_id));
      break;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_projection_insert(
      &materializer->projection,
      loom_bytecode_selected_table_projection_domain(table_kind),
      source_ordinal, target_id));
  *destination = target_id;
  return iree_ok_status();
}

// Keeps dependency construction off the completed-identity lookup path.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_bytecode_selected_table_materialize_unresolved(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_kind_t table_kind, uint32_t source_ordinal,
    uint32_t* out_target_id) {
  IREE_ASSERT(materializer->worklist.count == 0);
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_push(
      materializer, table_kind, source_ordinal, out_target_id));
  const iree_arena_checkpoint_t root_checkpoint =
      iree_arena_checkpoint_save(materializer->scratch_arena);
  iree_status_t status = iree_ok_status();
  while (materializer->worklist.count > 0 && iree_status_is_ok(status)) {
    const iree_host_size_t frame_depth = materializer->worklist.count;
    loom_bytecode_selected_table_chunk_t* chunk =
        materializer->worklist.current;
    loom_bytecode_selected_table_frame_t* frame =
        &chunk->frames[chunk->count - 1];
    loom_bytecode_selected_table_payload_t immediate_payload = {0};
    loom_bytecode_selected_table_payload_t* payload = &immediate_payload;
    iree_arena_checkpoint_t checkpoint;
    if (frame->pending == NULL) {
      // The caller already established that the root is missing. Children
      // may have duplicate queued references completed by an earlier sibling.
      if (frame_depth != 1 &&
          loom_bytecode_selected_projection_lookup(
              &materializer->projection,
              loom_bytecode_selected_table_projection_domain(frame->table_kind),
              frame->source_ordinal, frame->destination)) {
        loom_bytecode_selected_table_pop(materializer);
        continue;
      }
      if (frame_depth == 1) {
        checkpoint = root_checkpoint;
      } else {
        checkpoint = iree_arena_checkpoint_save(materializer->scratch_arena);
      }
      status = loom_bytecode_selected_table_prepare_entry(
          materializer, frame->table_kind, frame->source_ordinal, payload);
      if (iree_status_is_ok(status) &&
          materializer->worklist.count != frame_depth) {
        status = iree_arena_allocate(materializer->scratch_arena,
                                     sizeof(*frame->pending),
                                     (void**)&frame->pending);
        if (iree_status_is_ok(status)) {
          *frame->pending = (loom_bytecode_selected_table_pending_t){
              .checkpoint = checkpoint,
              .payload = *payload,
          };
        }
      }
    } else {
      payload = &frame->pending->payload;
      checkpoint = frame->pending->checkpoint;
    }
    if (iree_status_is_ok(status) &&
        materializer->worklist.count == frame_depth) {
      status = loom_bytecode_selected_table_finish_entry(
          materializer, frame->table_kind, frame->source_ordinal, payload,
          frame->destination);
      if (frame_depth != 1) {
        iree_arena_checkpoint_restore(&checkpoint);
      }
      loom_bytecode_selected_table_pop(materializer);
    }
  }
  materializer->worklist.count = 0;
  materializer->worklist.current = materializer->worklist.first;
  materializer->worklist.first->count = 0;
  iree_arena_checkpoint_restore(&root_checkpoint);
  return status;
}

static iree_status_t loom_bytecode_selected_table_materialize_root(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_selected_table_kind_t table_kind, uint32_t source_ordinal,
    uint32_t* out_target_id) {
  if (loom_bytecode_selected_projection_lookup(
          &materializer->projection,
          loom_bytecode_selected_table_projection_domain(table_kind),
          source_ordinal, out_target_id)) {
    return iree_ok_status();
  }
  return loom_bytecode_selected_table_materialize_unresolved(
      materializer, table_kind, source_ordinal, out_target_id);
}

iree_status_t loom_bytecode_selected_table_materialize_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint16_t source_encoding_id, uint16_t* out_target_encoding_id) {
  IREE_ASSERT(source_encoding_id > 0);
  IREE_ASSERT(source_encoding_id <= materializer->metadata->encodings.count);
  uint32_t target_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_materialize_root(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_ENCODING,
      source_encoding_id - 1, &target_id));
  *out_target_encoding_id = (uint16_t)target_id;
  return iree_ok_status();
}

iree_status_t loom_bytecode_selected_table_materialize_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_type_id_t source_type_id, loom_type_id_t* out_target_type_id) {
  IREE_ASSERT(source_type_id < materializer->metadata->types.count);
  uint32_t target_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_materialize_root(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_TYPE, source_type_id,
      &target_id));
  *out_target_type_id = (loom_type_id_t)target_id;
  return iree_ok_status();
}

iree_status_t loom_bytecode_selected_table_materialize_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id) {
  IREE_ASSERT(source_location_id < materializer->metadata->locations.count);
  if (source_location_id == LOOM_LOCATION_UNKNOWN) {
    *out_target_location_id = LOOM_LOCATION_UNKNOWN;
    return iree_ok_status();
  }
  uint32_t target_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_selected_table_materialize_root(
      materializer, LOOM_BYTECODE_SELECTED_TABLE_LOCATION, source_location_id,
      &target_id));
  *out_target_location_id = (loom_location_id_t)target_id;
  return iree_ok_status();
}
