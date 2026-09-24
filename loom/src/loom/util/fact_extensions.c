// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_extensions.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/structural_hash.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// Interned payloads
//===----------------------------------------------------------------------===//

typedef enum loom_value_fact_extension_kind_e {
  LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT = 1,
  LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA = 2,
  LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK = 3,
  LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES = 4,
  LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE = 5,
  LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE = 6,
  LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY = 7,
  LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD = 8,
} loom_value_fact_extension_kind_t;

typedef struct loom_value_fact_raw_payload_t {
  // Type-domain-owned payload tag. This is interpreted only through the
  // value's type-owned fact domain, never as a process-global schema ID.
  uint8_t tag;

  // Payload byte count.
  iree_host_size_t length;

  // Borrowed payload bytes. Entries interned in a table point into table arena.
  const void* data;
} loom_value_fact_raw_payload_t;

struct loom_value_fact_extension_entry_t {
  // Content hash for the extension kind and payload.
  uint32_t content_hash;
  // Collision-chain next extension ID, or zero for the end of the chain.
  loom_value_fact_extension_id_t next_id;
  // Extension payload kind.
  loom_value_fact_extension_kind_t kind;
  // Reserved for alignment and future flags. Always zero.
  uint32_t reserved;

  // Payload selected by kind.
  union {
    // Uniform-element vector payload.
    loom_value_fact_uniform_element_t uniform_element;
    // Small static vector lane payload.
    loom_value_fact_small_static_lanes_t small_static_lanes;
    // Vector iota payload.
    loom_value_fact_vector_iota_t vector_iota;
    // Vector prefix-mask payload.
    loom_value_fact_vector_prefix_mask_t vector_prefix_mask;
    // SSA encoding summary payload.
    loom_value_fact_encoding_summary_t encoding_summary;
    // Buffer storage-root payload.
    loom_value_fact_buffer_reference_t buffer_reference;
    // Typed view projection payload.
    loom_value_fact_view_reference_t view_reference;
    // Raw payload owned by the value's type-domain schema.
    loom_value_fact_raw_payload_t type_payload;
  } payload;
};

//===----------------------------------------------------------------------===//
// Payload interning
//===----------------------------------------------------------------------===//

static uint32_t loom_value_fact_hash_facts(loom_value_facts_t facts,
                                           uint32_t hash) {
  return loom_structural_hash_mix_bytes(hash, &facts, sizeof(facts));
}

static uint32_t loom_value_fact_hash_address_layout(
    loom_value_fact_address_layout_t layout, uint32_t hash) {
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)layout.kind);
  hash = loom_structural_hash_mix_u32(hash, layout.rank);
  if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
      layout.rank > 0 && layout.strides) {
    hash = loom_structural_hash_mix_bytes(
        hash, layout.strides, layout.rank * sizeof(loom_value_facts_t));
  }
  return hash;
}

static uint32_t loom_value_fact_hash_encoded_operand_schema(
    loom_value_fact_encoded_operand_schema_t schema, uint32_t hash) {
  return loom_structural_hash_mix_bytes(hash, &schema, sizeof(schema));
}

static uint32_t loom_value_fact_hash_storage_schema(
    loom_value_fact_storage_schema_t schema, uint32_t hash) {
  hash = loom_structural_hash_mix_u32(hash, schema.static_spec_encoding_id);
  return loom_value_fact_hash_encoded_operand_schema(schema.encoded_operand,
                                                     hash);
}

static uint32_t loom_value_fact_hash_encoding_summary(
    loom_value_fact_encoding_summary_t summary, uint32_t hash) {
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)summary.role);
  hash = loom_structural_hash_mix_u32(hash, summary.static_spec_encoding_id);
  hash = loom_value_fact_hash_address_layout(summary.address_layout, hash);
  return loom_value_fact_hash_storage_schema(summary.storage_schema, hash);
}

static uint32_t loom_value_fact_hash_reference_origin(
    loom_value_fact_reference_origin_t origin, uint32_t hash) {
  hash = loom_structural_hash_mix_u32(hash, origin.entry_value_id);
  return loom_structural_hash_mix_u32(
      hash, origin.function_symbol_id | ((uint32_t)origin.region_index << 16) |
                ((uint32_t)origin.kind << 24));
}

static uint32_t loom_value_fact_hash_buffer_reference(
    loom_value_fact_buffer_reference_t reference, uint32_t hash) {
  hash = loom_value_fact_hash_facts(reference.maximum_byte_extent, hash);
  hash = loom_structural_hash_mix_u64(hash, reference.minimum_alignment);
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)reference.memory_space);
  hash = loom_structural_hash_mix_u32(hash, reference.root_value_id);
  hash = loom_structural_hash_mix_u32(hash, reference.alias_scope_id);
  hash = loom_structural_hash_mix_u32(hash, reference.nullability);
  return loom_value_fact_hash_reference_origin(reference.origin, hash);
}

static uint32_t loom_value_fact_hash_view_reference(
    loom_value_fact_view_reference_t reference, uint32_t hash) {
  hash = loom_value_fact_hash_facts(reference.base_byte_offset, hash);
  hash = loom_value_fact_hash_facts(reference.footprint_byte_length, hash);
  hash = loom_structural_hash_mix_u64(hash, reference.minimum_alignment);
  hash = loom_structural_hash_mix_u64(hash, reference.root_minimum_alignment);
  hash =
      loom_structural_hash_mix_u64(hash, reference.static_element_byte_count);
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)reference.memory_space);
  hash = loom_structural_hash_mix_u32(hash, reference.root_value_id);
  hash = loom_structural_hash_mix_u32(hash, reference.buffer_value_id);
  hash = loom_structural_hash_mix_u32(hash, reference.alias_scope_id);
  hash = loom_structural_hash_mix_u32(hash, reference.nullability);
  hash = loom_structural_hash_mix_u32(hash, reference.address_bitwidth);
  return loom_value_fact_hash_reference_origin(reference.origin, hash);
}

static uint32_t loom_value_fact_hash_raw_payload(
    loom_value_fact_raw_payload_t payload, uint32_t hash) {
  hash = loom_structural_hash_mix_u32(hash, payload.tag);
  hash = loom_structural_hash_mix_u64(hash, payload.length);
  return loom_structural_hash_mix_bytes(hash, payload.data, payload.length);
}

static bool loom_value_fact_buffer_reference_equal(
    loom_value_fact_buffer_reference_t lhs,
    loom_value_fact_buffer_reference_t rhs) {
  return loom_value_facts_equal(lhs.maximum_byte_extent,
                                rhs.maximum_byte_extent) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.alias_scope_id == rhs.alias_scope_id &&
         lhs.nullability == rhs.nullability &&
         loom_value_fact_reference_origin_equal(lhs.origin, rhs.origin);
}

static bool loom_value_fact_view_reference_equal(
    loom_value_fact_view_reference_t lhs,
    loom_value_fact_view_reference_t rhs) {
  return loom_value_facts_equal(lhs.base_byte_offset, rhs.base_byte_offset) &&
         loom_value_facts_equal(lhs.footprint_byte_length,
                                rhs.footprint_byte_length) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.root_minimum_alignment == rhs.root_minimum_alignment &&
         lhs.static_element_byte_count == rhs.static_element_byte_count &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.buffer_value_id == rhs.buffer_value_id &&
         lhs.alias_scope_id == rhs.alias_scope_id &&
         lhs.nullability == rhs.nullability &&
         lhs.address_bitwidth == rhs.address_bitwidth &&
         loom_value_fact_reference_origin_equal(lhs.origin, rhs.origin);
}

static bool loom_value_fact_address_layout_equal(
    loom_value_fact_address_layout_t lhs,
    loom_value_fact_address_layout_t rhs) {
  if (lhs.kind != rhs.kind || lhs.rank != rhs.rank) {
    return false;
  }
  if (lhs.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return true;
  }
  if (lhs.rank == 0) {
    return true;
  }
  if (!lhs.strides || !rhs.strides) {
    return lhs.strides == rhs.strides;
  }
  return memcmp(lhs.strides, rhs.strides,
                lhs.rank * sizeof(loom_value_facts_t)) == 0;
}

bool loom_value_fact_encoded_operand_schema_equal(
    loom_value_fact_encoded_operand_schema_t lhs,
    loom_value_fact_encoded_operand_schema_t rhs) {
  return memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

bool loom_value_fact_encoded_operand_schema_is_unknown(
    loom_value_fact_encoded_operand_schema_t schema) {
  loom_value_fact_encoded_operand_schema_t unknown;
  memset(&unknown, 0, sizeof(unknown));
  return loom_value_fact_encoded_operand_schema_equal(schema, unknown);
}

bool loom_value_fact_encoded_operand_schema_has_scale(
    loom_value_fact_encoded_operand_schema_t schema) {
  return schema.scale_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE ||
         schema.secondary_scale_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE ||
         schema.scale_topology != LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE ||
         iree_any_bit_set(
             schema.flags,
             LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK) ||
         schema.scale_group.element_count != 0 ||
         schema.scale_group.shape[0] != 0 || schema.scale_operand_count != 0;
}

bool loom_value_fact_encoded_operand_schema_scale_is_complete(
    loom_value_fact_encoded_operand_schema_t schema) {
  if (!loom_value_fact_encoded_operand_schema_has_scale(schema)) {
    return true;
  }
  if (schema.scale_topology == 0 || schema.scale_group.element_count == 0 ||
      schema.scale_operand_count == 0) {
    return false;
  }
  if (iree_any_bit_set(schema.scale_topology,
                       LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
                           LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D)) {
    return schema.scale_group.shape[0] != 0 && schema.scale_group.shape[1] == 0;
  }
  if (iree_any_bit_set(schema.scale_topology,
                       LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D)) {
    return schema.scale_group.shape[0] != 0 &&
           schema.scale_group.shape[1] != 0 && schema.scale_group.shape[2] == 0;
  }
  return true;
}

bool loom_value_fact_encoded_operand_schema_sparsity_is_complete(
    loom_value_fact_encoded_operand_schema_t schema) {
  const bool is_structured =
      schema.sparsity_policy == LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED;
  const uint16_t nonzero_element_count =
      schema.sparsity_group.nonzero_element_count;
  const uint16_t element_count = schema.sparsity_group.element_count;
  if (!is_structured) {
    return nonzero_element_count == 0 && element_count == 0;
  }
  return nonzero_element_count > 0 && nonzero_element_count < element_count;
}

static bool loom_value_fact_storage_schema_equal(
    loom_value_fact_storage_schema_t lhs,
    loom_value_fact_storage_schema_t rhs) {
  return lhs.static_spec_encoding_id == rhs.static_spec_encoding_id &&
         loom_value_fact_encoded_operand_schema_equal(lhs.encoded_operand,
                                                      rhs.encoded_operand);
}

static bool loom_value_fact_encoding_summary_equal(
    loom_value_fact_encoding_summary_t lhs,
    loom_value_fact_encoding_summary_t rhs) {
  return lhs.role == rhs.role &&
         lhs.static_spec_encoding_id == rhs.static_spec_encoding_id &&
         loom_value_fact_address_layout_equal(lhs.address_layout,
                                              rhs.address_layout) &&
         loom_value_fact_storage_schema_equal(lhs.storage_schema,
                                              rhs.storage_schema);
}

static uint32_t loom_value_fact_extension_hash(
    const loom_value_fact_extension_entry_t* entry) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)entry->kind);
  switch (entry->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT:
      return loom_structural_hash_mix_bytes(
          hash, &entry->payload.uniform_element,
          sizeof(entry->payload.uniform_element));
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES:
      hash = loom_structural_hash_mix_u64(
          hash, entry->payload.small_static_lanes.count);
      return loom_structural_hash_mix_bytes(
          hash, entry->payload.small_static_lanes.lanes,
          entry->payload.small_static_lanes.count * sizeof(loom_value_facts_t));
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA:
      return loom_structural_hash_mix_bytes(hash, &entry->payload.vector_iota,
                                            sizeof(entry->payload.vector_iota));
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK:
      return loom_structural_hash_mix_bytes(
          hash, &entry->payload.vector_prefix_mask,
          sizeof(entry->payload.vector_prefix_mask));
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY:
      return loom_value_fact_hash_encoding_summary(
          entry->payload.encoding_summary, hash);
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE:
      return loom_value_fact_hash_buffer_reference(
          entry->payload.buffer_reference, hash);
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE:
      return loom_value_fact_hash_view_reference(entry->payload.view_reference,
                                                 hash);
    case LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD:
      return loom_value_fact_hash_raw_payload(entry->payload.type_payload,
                                              hash);
    default:
      return hash;
  }
}

static bool loom_value_fact_extension_content_equal(
    const loom_value_fact_extension_entry_t* lhs,
    const loom_value_fact_extension_entry_t* rhs) {
  if (lhs->kind != rhs->kind) {
    return false;
  }
  switch (lhs->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT:
      return memcmp(&lhs->payload.uniform_element,
                    &rhs->payload.uniform_element,
                    sizeof(lhs->payload.uniform_element)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES:
      if (lhs->payload.small_static_lanes.count !=
          rhs->payload.small_static_lanes.count) {
        return false;
      }
      if (lhs->payload.small_static_lanes.count == 0) {
        return true;
      }
      return memcmp(lhs->payload.small_static_lanes.lanes,
                    rhs->payload.small_static_lanes.lanes,
                    lhs->payload.small_static_lanes.count *
                        sizeof(loom_value_facts_t)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA:
      return memcmp(&lhs->payload.vector_iota, &rhs->payload.vector_iota,
                    sizeof(lhs->payload.vector_iota)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK:
      return memcmp(&lhs->payload.vector_prefix_mask,
                    &rhs->payload.vector_prefix_mask,
                    sizeof(lhs->payload.vector_prefix_mask)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY:
      return loom_value_fact_encoding_summary_equal(
          lhs->payload.encoding_summary, rhs->payload.encoding_summary);
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE:
      return loom_value_fact_buffer_reference_equal(
          lhs->payload.buffer_reference, rhs->payload.buffer_reference);
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE:
      return loom_value_fact_view_reference_equal(lhs->payload.view_reference,
                                                  rhs->payload.view_reference);
    case LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD:
      if (lhs->payload.type_payload.tag != rhs->payload.type_payload.tag ||
          lhs->payload.type_payload.length !=
              rhs->payload.type_payload.length) {
        return false;
      }
      if (lhs->payload.type_payload.length == 0) {
        return true;
      }
      return memcmp(lhs->payload.type_payload.data,
                    rhs->payload.type_payload.data,
                    lhs->payload.type_payload.length) == 0;
    default:
      return false;
  }
}

static iree_status_t loom_value_fact_table_clone_fact_array(
    loom_value_fact_table_t* table, const loom_value_facts_t* facts,
    iree_host_size_t count, const loom_value_facts_t** out_facts) {
  *out_facts = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  loom_value_facts_t* cloned_facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena, count,
                                                 sizeof(loom_value_facts_t),
                                                 (void**)&cloned_facts));
  memcpy(cloned_facts, facts, count * sizeof(loom_value_facts_t));
  *out_facts = cloned_facts;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_materialize_extension_payload(
    loom_value_fact_table_t* table, loom_value_fact_extension_entry_t* entry) {
  if (entry->kind == LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES) {
    const loom_value_facts_t* lanes = NULL;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_array(
        table, entry->payload.small_static_lanes.lanes,
        entry->payload.small_static_lanes.count, &lanes));
    entry->payload.small_static_lanes.lanes = lanes;
    return iree_ok_status();
  }
  if (entry->kind == LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY &&
      entry->payload.encoding_summary.address_layout.kind ==
          LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    loom_value_fact_address_layout_t* layout =
        &entry->payload.encoding_summary.address_layout;
    const loom_value_facts_t* strides = NULL;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_array(
        table, layout->strides, layout->rank, &strides));
    layout->strides = strides;
    return iree_ok_status();
  }
  if (entry->kind == LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD &&
      entry->payload.type_payload.length > 0) {
    void* data = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        table->transient_arena, entry->payload.type_payload.length, &data));
    memcpy(data, entry->payload.type_payload.data,
           entry->payload.type_payload.length);
    entry->payload.type_payload.data = data;
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_ensure_extension_capacity(
    loom_value_fact_table_t* table, iree_host_size_t minimum_count) {
  if (minimum_count <= table->extensions.capacity) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity =
      table->extensions.capacity > 0 ? table->extensions.capacity * 2 : 64;
  while (new_capacity < minimum_count) {
    if (new_capacity > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fact extension capacity overflow");
    }
    new_capacity *= 2;
  }

  loom_value_fact_extension_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->transient_arena, new_capacity,
      sizeof(loom_value_fact_extension_entry_t), (void**)&new_entries));
  memset(new_entries, 0,
         new_capacity * sizeof(loom_value_fact_extension_entry_t));
  if (table->extensions.count > 0) {
    memcpy(new_entries, table->extensions.entries,
           table->extensions.count * sizeof(loom_value_fact_extension_entry_t));
  }
  table->extensions.entries = new_entries;
  table->extensions.capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_rehash_extensions(
    loom_value_fact_table_t* table, iree_host_size_t new_bucket_count) {
  loom_value_fact_extension_id_t* new_buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->transient_arena, new_bucket_count,
      sizeof(loom_value_fact_extension_id_t), (void**)&new_buckets));
  memset(new_buckets, 0,
         new_bucket_count * sizeof(loom_value_fact_extension_id_t));
  for (iree_host_size_t i = 0; i < table->extensions.count; ++i) {
    loom_value_fact_extension_entry_t* entry = &table->extensions.entries[i];
    loom_value_fact_extension_id_t id = (loom_value_fact_extension_id_t)(i + 1);
    iree_host_size_t bucket_index =
        (iree_host_size_t)entry->content_hash & (new_bucket_count - 1);
    entry->next_id = new_buckets[bucket_index];
    new_buckets[bucket_index] = id;
  }
  table->extensions.buckets = new_buckets;
  table->extensions.bucket_count = new_bucket_count;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_ensure_extension_buckets(
    loom_value_fact_table_t* table, iree_host_size_t minimum_count) {
  iree_host_size_t new_bucket_count = table->extensions.bucket_count;
  if (new_bucket_count == 0) {
    new_bucket_count = 64;
  }
  while (minimum_count > new_bucket_count - new_bucket_count / 4) {
    if (new_bucket_count > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fact extension bucket capacity overflow");
    }
    new_bucket_count *= 2;
  }
  if (new_bucket_count == table->extensions.bucket_count) {
    return iree_ok_status();
  }
  return loom_value_fact_table_rehash_extensions(table, new_bucket_count);
}

static iree_status_t loom_value_fact_table_intern_extension(
    loom_value_fact_table_t* table,
    const loom_value_fact_extension_entry_t* candidate,
    loom_value_fact_extension_id_t* out_id) {
  loom_value_fact_extension_entry_t entry = *candidate;
  entry.content_hash =
      loom_structural_hash_finalize(loom_value_fact_extension_hash(&entry));
  entry.next_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;

  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_extension_buckets(
      table, table->extensions.count));
  iree_host_size_t bucket_index = (iree_host_size_t)entry.content_hash &
                                  (table->extensions.bucket_count - 1);
  for (loom_value_fact_extension_id_t id =
           table->extensions.buckets[bucket_index];
       id != LOOM_VALUE_FACT_EXTENSION_ID_NONE;
       id = table->extensions.entries[id - 1].next_id) {
    const loom_value_fact_extension_entry_t* existing =
        &table->extensions.entries[id - 1];
    if (existing->content_hash == entry.content_hash &&
        loom_value_fact_extension_content_equal(existing, &entry)) {
      *out_id = id;
      return iree_ok_status();
    }
  }

  iree_host_size_t new_count = table->extensions.count + 1;
  if (new_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "fact extension ID capacity exceeded");
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_extension_capacity(table, new_count));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_extension_buckets(table, new_count));
  bucket_index = (iree_host_size_t)entry.content_hash &
                 (table->extensions.bucket_count - 1);
  loom_value_fact_extension_id_t id = (loom_value_fact_extension_id_t)new_count;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_materialize_extension_payload(table, &entry));
  entry.next_id = table->extensions.buckets[bucket_index];
  table->extensions.entries[table->extensions.count] = entry;
  table->extensions.buckets[bucket_index] = id;
  table->extensions.count = new_count;
  *out_id = id;
  return iree_ok_status();
}

static iree_status_t loom_value_facts_make_extension(
    loom_fact_context_t* context,
    const loom_value_fact_extension_entry_t* candidate,
    loom_value_facts_t* out) {
  loom_value_fact_table_t* table = context->table;
  loom_value_fact_extension_id_t extension_id =
      LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_intern_extension(table, candidate, &extension_id));
  *out = loom_value_facts_unknown();
  out->extension_id = extension_id;
  return iree_ok_status();
}

static const loom_value_fact_extension_entry_t*
loom_value_facts_lookup_extension(const loom_fact_context_t* context,
                                  loom_value_facts_t facts) {
  if (!context || !context->table ||
      facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    return NULL;
  }
  const loom_value_fact_table_t* table = context->table;
  if (facts.extension_id > table->extensions.count) {
    return NULL;
  }
  return &table->extensions.entries[facts.extension_id - 1];
}

//===----------------------------------------------------------------------===//
// Cross-table comparison
//===----------------------------------------------------------------------===//

static bool loom_value_fact_table_scalar_fields_equal(loom_value_facts_t lhs,
                                                      loom_value_facts_t rhs) {
  lhs.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  rhs.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  return loom_value_facts_equal(lhs, rhs);
}

static const loom_value_fact_domain_t* loom_value_fact_domain_for_type(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_type_t type) {
  if (!table || !table->context.resolve_type_domain.fn) {
    return NULL;
  }
  return table->context.resolve_type_domain.fn(
      table->context.resolve_type_domain.user_data, &table->context, module,
      type);
}

static bool loom_value_fact_table_fact_array_equal(
    const loom_value_fact_table_t* lhs_table, const loom_value_facts_t* lhs,
    const loom_value_fact_table_t* rhs_table, const loom_value_facts_t* rhs,
    iree_host_size_t count) {
  if (count == 0) {
    return true;
  }
  if (!lhs || !rhs) {
    return lhs == rhs;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!loom_value_fact_table_facts_equal(lhs_table, lhs[i], rhs_table,
                                           rhs[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_value_fact_table_address_layout_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_address_layout_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_address_layout_t rhs) {
  if (lhs.kind != rhs.kind || lhs.rank != rhs.rank) {
    return false;
  }
  if (lhs.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return true;
  }
  return loom_value_fact_table_fact_array_equal(
      lhs_table, lhs.strides, rhs_table, rhs.strides, lhs.rank);
}

static bool loom_value_fact_table_encoding_summary_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_encoding_summary_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_encoding_summary_t rhs) {
  return lhs.role == rhs.role &&
         lhs.static_spec_encoding_id == rhs.static_spec_encoding_id &&
         loom_value_fact_table_address_layout_equal(
             lhs_table, lhs.address_layout, rhs_table, rhs.address_layout) &&
         loom_value_fact_storage_schema_equal(lhs.storage_schema,
                                              rhs.storage_schema);
}

static bool loom_value_fact_table_buffer_reference_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_buffer_reference_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_buffer_reference_t rhs) {
  return loom_value_fact_table_facts_equal(lhs_table, lhs.maximum_byte_extent,
                                           rhs_table,
                                           rhs.maximum_byte_extent) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.alias_scope_id == rhs.alias_scope_id &&
         lhs.nullability == rhs.nullability &&
         loom_value_fact_reference_origin_equal(lhs.origin, rhs.origin);
}

static bool loom_value_fact_table_view_reference_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_view_reference_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_view_reference_t rhs) {
  return loom_value_fact_table_facts_equal(lhs_table, lhs.base_byte_offset,
                                           rhs_table, rhs.base_byte_offset) &&
         loom_value_fact_table_facts_equal(lhs_table, lhs.footprint_byte_length,
                                           rhs_table,
                                           rhs.footprint_byte_length) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.root_minimum_alignment == rhs.root_minimum_alignment &&
         lhs.static_element_byte_count == rhs.static_element_byte_count &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.buffer_value_id == rhs.buffer_value_id &&
         lhs.alias_scope_id == rhs.alias_scope_id &&
         lhs.nullability == rhs.nullability &&
         lhs.address_bitwidth == rhs.address_bitwidth &&
         loom_value_fact_reference_origin_equal(lhs.origin, rhs.origin);
}

static bool loom_value_fact_table_extension_entries_equal(
    const loom_value_fact_table_t* lhs_table,
    const loom_value_fact_extension_entry_t* lhs,
    const loom_value_fact_table_t* rhs_table,
    const loom_value_fact_extension_entry_t* rhs) {
  if (!lhs || !rhs || lhs->kind != rhs->kind) {
    return false;
  }
  switch (lhs->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT:
      return loom_value_fact_table_facts_equal(
          lhs_table, lhs->payload.uniform_element.element, rhs_table,
          rhs->payload.uniform_element.element);
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES:
      if (lhs->payload.small_static_lanes.count !=
          rhs->payload.small_static_lanes.count) {
        return false;
      }
      return loom_value_fact_table_fact_array_equal(
          lhs_table, lhs->payload.small_static_lanes.lanes, rhs_table,
          rhs->payload.small_static_lanes.lanes,
          lhs->payload.small_static_lanes.count);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA:
      return loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_iota.base, rhs_table,
                 rhs->payload.vector_iota.base) &&
             loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_iota.step, rhs_table,
                 rhs->payload.vector_iota.step);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK:
      return loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_prefix_mask.lower_bound,
                 rhs_table, rhs->payload.vector_prefix_mask.lower_bound) &&
             loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_prefix_mask.upper_bound,
                 rhs_table, rhs->payload.vector_prefix_mask.upper_bound) &&
             loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_prefix_mask.step, rhs_table,
                 rhs->payload.vector_prefix_mask.step);
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY:
      return loom_value_fact_table_encoding_summary_equal(
          lhs_table, lhs->payload.encoding_summary, rhs_table,
          rhs->payload.encoding_summary);
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE:
      return loom_value_fact_table_buffer_reference_equal(
          lhs_table, lhs->payload.buffer_reference, rhs_table,
          rhs->payload.buffer_reference);
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE:
      return loom_value_fact_table_view_reference_equal(
          lhs_table, lhs->payload.view_reference, rhs_table,
          rhs->payload.view_reference);
    case LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD:
      if (lhs->payload.type_payload.tag != rhs->payload.type_payload.tag ||
          lhs->payload.type_payload.length !=
              rhs->payload.type_payload.length) {
        return false;
      }
      if (lhs->payload.type_payload.length == 0) {
        return true;
      }
      return memcmp(lhs->payload.type_payload.data,
                    rhs->payload.type_payload.data,
                    lhs->payload.type_payload.length) == 0;
    default:
      return false;
  }
}

bool loom_value_fact_table_extensions_equal(
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs) {
  if (lhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE ||
      rhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    return lhs.extension_id == rhs.extension_id;
  }
  const loom_value_fact_extension_entry_t* lhs_entry =
      loom_value_facts_lookup_extension(lhs_table ? &lhs_table->context : NULL,
                                        lhs);
  const loom_value_fact_extension_entry_t* rhs_entry =
      loom_value_facts_lookup_extension(rhs_table ? &rhs_table->context : NULL,
                                        rhs);
  return loom_value_fact_table_extension_entries_equal(lhs_table, lhs_entry,
                                                       rhs_table, rhs_entry);
}

bool loom_value_fact_table_facts_equal(const loom_value_fact_table_t* lhs_table,
                                       loom_value_facts_t lhs,
                                       const loom_value_fact_table_t* rhs_table,
                                       loom_value_facts_t rhs) {
  return loom_value_fact_table_scalar_fields_equal(lhs, rhs) &&
         loom_value_fact_table_extensions_equal(lhs_table, lhs, rhs_table, rhs);
}

bool loom_value_fact_table_extensions_equal_for_type(
    const loom_module_t* module, loom_type_t type,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs) {
  const loom_value_fact_domain_t* domain =
      loom_value_fact_domain_for_type(lhs_table, module, type);
  if (domain && domain->extensions_equal) {
    return domain->extensions_equal(domain, module, type, lhs_table, lhs,
                                    rhs_table, rhs);
  }
  return loom_value_fact_table_extensions_equal(lhs_table, lhs, rhs_table, rhs);
}

bool loom_value_fact_table_facts_equal_for_type(
    const loom_module_t* module, loom_type_t type,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs) {
  return loom_value_fact_table_scalar_fields_equal(lhs, rhs) &&
         loom_value_fact_table_extensions_equal_for_type(
             module, type, lhs_table, lhs, rhs_table, rhs);
}

//===----------------------------------------------------------------------===//
// Fact extensions
//===----------------------------------------------------------------------===//

iree_status_t loom_value_facts_make_uniform_element(
    loom_fact_context_t* context, loom_value_facts_t element,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT;
  entry.payload.uniform_element.element = element;
  IREE_RETURN_IF_ERROR(loom_value_facts_make_extension(context, &entry, out));
  out->flags |= element.flags & LOOM_VALUE_FACT_FLOAT_PREDICATE_MASK;
  return iree_ok_status();
}

bool loom_value_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_uniform_element_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT) {
    return false;
  }
  if (out) {
    *out = entry->payload.uniform_element;
  }
  return true;
}

iree_status_t loom_value_facts_make_small_static_lanes(
    loom_fact_context_t* context, loom_value_fact_small_static_lanes_t lanes,
    loom_value_facts_t* out) {
  if (lanes.count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    *out = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES;
  entry.payload.small_static_lanes = lanes;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_small_static_lanes(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_small_static_lanes_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES) {
    return false;
  }
  if (out) {
    *out = entry->payload.small_static_lanes;
  }
  return true;
}

bool loom_value_facts_query_all_equal_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_element) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    if (loom_value_facts_is_unknown(uniform.element)) {
      return false;
    }
    *out_element = uniform.element;
    return true;
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(context, facts, &lanes)) {
    if (lanes.count == 0) {
      return false;
    }
    loom_value_facts_t element = lanes.lanes[0];
    if (loom_value_facts_is_unknown(element)) {
      return false;
    }
    for (iree_host_size_t i = 1; i < lanes.count; ++i) {
      if (!loom_value_facts_equal(element, lanes.lanes[i])) {
        return false;
      }
    }
    *out_element = element;
    return true;
  }

  if (facts.extension_id != LOOM_VALUE_FACT_EXTENSION_ID_NONE ||
      loom_value_facts_is_unknown(facts)) {
    return false;
  }
  *out_element = facts;
  return true;
}

iree_status_t loom_value_facts_make_vector_iota(
    loom_fact_context_t* context, loom_value_fact_vector_iota_t iota,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA;
  entry.payload.vector_iota = iota;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_vector_iota(const loom_fact_context_t* context,
                                        loom_value_facts_t facts,
                                        loom_value_fact_vector_iota_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA) {
    return false;
  }
  if (out) {
    *out = entry->payload.vector_iota;
  }
  return true;
}

iree_status_t loom_value_facts_make_vector_prefix_mask(
    loom_fact_context_t* context, loom_value_fact_vector_prefix_mask_t mask,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK;
  entry.payload.vector_prefix_mask = mask;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_vector_prefix_mask(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_vector_prefix_mask_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK) {
    return false;
  }
  if (out) {
    *out = entry->payload.vector_prefix_mask;
  }
  return true;
}

iree_status_t loom_value_facts_make_encoding_summary(
    loom_fact_context_t* context, loom_value_fact_encoding_summary_t summary,
    loom_value_facts_t* out) {
  if (summary.role == LOOM_ENCODING_ROLE_UNKNOWN &&
      summary.static_spec_encoding_id == 0 &&
      summary.address_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN &&
      summary.storage_schema.static_spec_encoding_id == 0 &&
      loom_value_fact_encoded_operand_schema_is_unknown(
          summary.storage_schema.encoded_operand)) {
    *out = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY;
  entry.payload.encoding_summary = summary;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_encoding_summary(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_encoding_summary_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY) {
    return false;
  }
  if (out) {
    *out = entry->payload.encoding_summary;
  }
  return true;
}

iree_status_t loom_value_facts_make_buffer_reference(
    loom_fact_context_t* context, loom_value_fact_buffer_reference_t reference,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE;
  entry.payload.buffer_reference = reference;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_buffer_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_buffer_reference_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE) {
    return false;
  }
  if (out) {
    *out = entry->payload.buffer_reference;
  }
  return true;
}

iree_status_t loom_value_facts_make_view_reference(
    loom_fact_context_t* context, loom_value_fact_view_reference_t reference,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE;
  entry.payload.view_reference = reference;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_view_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_view_reference_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE) {
    return false;
  }
  if (out) {
    *out = entry->payload.view_reference;
  }
  return true;
}

iree_status_t loom_value_facts_make_extension_payload(
    loom_fact_context_t* context, uint8_t payload_tag, const void* payload,
    iree_host_size_t payload_length, loom_value_facts_t* out) {
  if (payload_length > LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT) {
    *out = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD;
  entry.payload.type_payload.tag = payload_tag;
  entry.payload.type_payload.length = payload_length;
  entry.payload.type_payload.data = payload;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_extension_payload(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    uint8_t payload_tag, const void** out_payload,
    iree_host_size_t* out_payload_length) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD ||
      entry->payload.type_payload.tag != payload_tag) {
    return false;
  }
  if (out_payload) {
    *out_payload = entry->payload.type_payload.data;
  }
  if (out_payload_length) {
    *out_payload_length = entry->payload.type_payload.length;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Cross-table transfers
//===----------------------------------------------------------------------===//

// Array payloads borrow bounded scratch until interning decides whether an
// owned copy is needed. Keep that scratch out of scalar/reference transfers.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_value_fact_table_clone_array_extension(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t facts, loom_value_fact_extension_entry_t* entry,
    loom_value_facts_t* out_facts) {
  const loom_value_facts_t** array = NULL;
  iree_host_size_t count = 0;
  if (entry->kind == LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES) {
    array = &entry->payload.small_static_lanes.lanes;
    count = entry->payload.small_static_lanes.count;
  } else {
    array = &entry->payload.encoding_summary.address_layout.strides;
    count = entry->payload.encoding_summary.address_layout.rank;
  }
  static_assert(LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT >= LOOM_TYPE_MAX_RANK,
                "array scratch must cover lanes and layout axes");
  loom_value_facts_t cloned_facts[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT];
  IREE_ASSERT(count <= IREE_ARRAYSIZE(cloned_facts));
  const loom_value_facts_t* source_facts = *array;
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
        target, source, source_facts[i], &cloned_facts[i]));
  }
  *array = cloned_facts;
  loom_value_fact_extension_id_t extension_id =
      LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_intern_extension(target, entry, &extension_id));
  *out_facts = facts;
  out_facts->extension_id = extension_id;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_clone_fact(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t facts, loom_value_facts_t* out_facts) {
  if (target == source ||
      facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    *out_facts = facts;
    return iree_ok_status();
  }

  const loom_value_fact_extension_entry_t* source_entry =
      loom_value_facts_lookup_extension(&source->context, facts);
  if (!source_entry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source fact extension id %u is invalid",
                            (unsigned)facts.extension_id);
  }

  loom_value_fact_extension_entry_t target_entry = *source_entry;
  target_entry.content_hash = 0;
  target_entry.next_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  switch (source_entry->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.uniform_element.element,
          &target_entry.payload.uniform_element.element));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES: {
      return loom_value_fact_table_clone_array_extension(
          target, source, facts, &target_entry, out_facts);
    }
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_iota.base,
          &target_entry.payload.vector_iota.base));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_iota.step,
          &target_entry.payload.vector_iota.step));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_prefix_mask.lower_bound,
          &target_entry.payload.vector_prefix_mask.lower_bound));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_prefix_mask.upper_bound,
          &target_entry.payload.vector_prefix_mask.upper_bound));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_prefix_mask.step,
          &target_entry.payload.vector_prefix_mask.step));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY: {
      if (source_entry->payload.encoding_summary.address_layout.kind ==
          LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
        return loom_value_fact_table_clone_array_extension(
            target, source, facts, &target_entry, out_facts);
      }
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source,
          source_entry->payload.buffer_reference.maximum_byte_extent,
          &target_entry.payload.buffer_reference.maximum_byte_extent));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.view_reference.base_byte_offset,
          &target_entry.payload.view_reference.base_byte_offset));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source,
          source_entry->payload.view_reference.footprint_byte_length,
          &target_entry.payload.view_reference.footprint_byte_length));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_TYPE_PAYLOAD:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown source fact extension kind %u",
                              (unsigned)source_entry->kind);
  }

  loom_value_fact_extension_id_t target_extension_id =
      LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_intern_extension(
      target, &target_entry, &target_extension_id));
  *out_facts = facts;
  out_facts->extension_id = target_extension_id;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_clone_fact_for_type(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module, loom_type_t type, loom_value_facts_t facts,
    loom_value_facts_t* out_facts) {
  *out_facts = facts;
  if (target == source ||
      facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    return iree_ok_status();
  }

  const loom_value_fact_domain_t* domain =
      loom_value_fact_domain_for_type(target, module, type);
  if (domain && domain->clone_extension) {
    out_facts->extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    return domain->clone_extension(domain, module, type, target, source, facts,
                                   out_facts);
  }
  return loom_value_fact_table_clone_fact(target, source, facts, out_facts);
}

iree_status_t loom_value_fact_table_meet_for_type(
    loom_value_fact_table_t* target, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* out_facts) {
  if (loom_value_fact_table_facts_equal_for_type(module, type, lhs_table, lhs,
                                                 rhs_table, rhs)) {
    return loom_value_fact_table_clone_fact_for_type(target, lhs_table, module,
                                                     type, lhs, out_facts);
  }

  loom_value_facts_t lhs_scalar = lhs;
  lhs_scalar.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  loom_value_facts_t rhs_scalar = rhs;
  rhs_scalar.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  loom_value_facts_meet(&lhs_scalar, &rhs_scalar, out_facts);

  const loom_value_fact_domain_t* domain =
      loom_value_fact_domain_for_type(target, module, type);
  if (domain && domain->meet_extension) {
    return domain->meet_extension(domain, module, type, target, lhs_table, lhs,
                                  rhs_table, rhs, out_facts);
  }
  if (loom_value_fact_table_extensions_equal_for_type(module, type, lhs_table,
                                                      lhs, rhs_table, rhs)) {
    loom_value_facts_t cloned_extension = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        target, lhs_table, module, type, lhs, &cloned_extension));
    out_facts->extension_id = cloned_extension.extension_id;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_widen_for_type(
    loom_value_fact_table_t* target, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* previous_table,
    loom_value_facts_t previous, const loom_value_fact_table_t* next_table,
    loom_value_facts_t next, uint32_t iteration,
    loom_value_facts_t* out_facts) {
  if (loom_value_fact_table_facts_equal_for_type(module, type, previous_table,
                                                 previous, next_table, next)) {
    return loom_value_fact_table_clone_fact_for_type(target, next_table, module,
                                                     type, next, out_facts);
  }

  if (iteration < 2) {
    return loom_value_fact_table_meet_for_type(target, module, type,
                                               previous_table, previous,
                                               next_table, next, out_facts);
  }

  // Floating classifications have finite height and need no interval widening.
  loom_value_facts_meet(&previous, &next, out_facts);
  if (!loom_value_facts_is_float(previous) &&
      !loom_value_facts_is_float(next)) {
    // Range growth does not invalidate divisibility. Its join descends through
    // positive divisors, so it converges independently of interval widening.
    const int64_t known_divisor = out_facts->known_divisor;
    int64_t range_lo = INT64_MIN;
    int64_t range_hi = INT64_MAX;
    if (loom_type_is_scalar(type)) {
      loom_value_facts_scalar_type_domain(loom_type_element_type(type),
                                          &range_lo, &range_hi);
    }
    *out_facts = loom_value_facts_make(range_lo, range_hi, known_divisor);
    if (loom_value_facts_is_lane_varying(previous) ||
        loom_value_facts_is_lane_varying(next)) {
      loom_value_facts_mark_lane_distribution_for_type(type, out_facts);
    } else {
      const loom_value_fact_uniform_scope_t uniform_scope =
          iree_min(loom_value_facts_uniform_scope(previous),
                   loom_value_facts_uniform_scope(next));
      loom_value_facts_mark_uniform_at_scope(out_facts, uniform_scope);
    }
  }
  const loom_value_fact_domain_t* domain =
      loom_value_fact_domain_for_type(target, module, type);
  if (domain && domain->widen_extension) {
    return domain->widen_extension(domain, module, type, target, previous_table,
                                   previous, next_table, next, iteration,
                                   out_facts);
  }
  if (loom_value_fact_table_extensions_equal_for_type(
          module, type, previous_table, previous, next_table, next)) {
    loom_value_facts_t cloned_extension = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        target, previous_table, module, type, previous, &cloned_extension));
    out_facts->extension_id = cloned_extension.extension_id;
  }
  return iree_ok_status();
}
