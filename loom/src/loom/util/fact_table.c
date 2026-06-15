// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_table.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

//===----------------------------------------------------------------------===//
// Capacity management
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

static iree_status_t loom_value_fact_table_ensure_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  if (capacity <= table->capacity) return iree_ok_status();
  const iree_host_size_t old_capacity = table->capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      table->arena, table->capacity, capacity, sizeof(loom_value_facts_t),
      &table->capacity, (void**)&table->entries));
  memset(table->entries + old_capacity, 0,
         (table->capacity - old_capacity) * sizeof(loom_value_facts_t));
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_allocate_initial_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  return loom_value_fact_table_ensure_capacity(table, capacity);
}

static iree_status_t loom_value_fact_table_append_touched_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->touched_count >= table->touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->touched_count, table->touched_count + 1,
        sizeof(*table->touched_values), &table->touched_capacity,
        (void**)&table->touched_values));
  }
  table->touched_values[table->touched_count++] = value_id;
  return iree_ok_status();
}

static uint32_t loom_value_fact_hash_bytes(const void* data,
                                           iree_host_size_t length,
                                           uint32_t hash) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_value_fact_hash_u32(uint32_t value, uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_host_size(iree_host_size_t value,
                                               uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_i64(int64_t value, uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_u64(uint64_t value, uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_facts(loom_value_facts_t facts,
                                           uint32_t hash) {
  return loom_value_fact_hash_bytes(&facts, sizeof(facts), hash);
}

static uint32_t loom_value_fact_hash_address_layout(
    loom_value_fact_address_layout_t layout, uint32_t hash) {
  hash = loom_value_fact_hash_u32((uint32_t)layout.kind, hash);
  hash = loom_value_fact_hash_u32(layout.rank, hash);
  if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
      layout.rank > 0 && layout.strides) {
    hash = loom_value_fact_hash_bytes(
        layout.strides, layout.rank * sizeof(loom_value_facts_t), hash);
  }
  return hash;
}

static uint32_t loom_value_fact_hash_encoded_operand_schema(
    loom_value_fact_encoded_operand_schema_t schema, uint32_t hash) {
  return loom_value_fact_hash_bytes(&schema, sizeof(schema), hash);
}

static uint32_t loom_value_fact_hash_storage_schema(
    loom_value_fact_storage_schema_t schema, uint32_t hash) {
  hash = loom_value_fact_hash_u32(schema.static_spec_encoding_id, hash);
  return loom_value_fact_hash_encoded_operand_schema(schema.encoded_operand,
                                                     hash);
}

static uint32_t loom_value_fact_hash_encoding_summary(
    loom_value_fact_encoding_summary_t summary, uint32_t hash) {
  hash = loom_value_fact_hash_u32((uint32_t)summary.role, hash);
  hash = loom_value_fact_hash_u32(summary.static_spec_encoding_id, hash);
  hash = loom_value_fact_hash_address_layout(summary.address_layout, hash);
  return loom_value_fact_hash_storage_schema(summary.storage_schema, hash);
}

static uint32_t loom_value_fact_hash_buffer_reference(
    loom_value_fact_buffer_reference_t reference, uint32_t hash) {
  hash = loom_value_fact_hash_facts(reference.maximum_byte_extent, hash);
  hash = loom_value_fact_hash_u64(reference.minimum_alignment, hash);
  hash = loom_value_fact_hash_u32((uint32_t)reference.memory_space, hash);
  hash = loom_value_fact_hash_u32(reference.root_value_id, hash);
  hash = loom_value_fact_hash_u32(reference.alias_scope_id, hash);
  return loom_value_fact_hash_u32(reference.nullability, hash);
}

static uint32_t loom_value_fact_hash_view_reference(
    loom_value_fact_view_reference_t reference, uint32_t hash) {
  hash = loom_value_fact_hash_facts(reference.base_byte_offset, hash);
  hash = loom_value_fact_hash_facts(reference.footprint_byte_length, hash);
  hash = loom_value_fact_hash_u64(reference.minimum_alignment, hash);
  hash = loom_value_fact_hash_u64(reference.root_minimum_alignment, hash);
  hash = loom_value_fact_hash_i64(reference.static_element_byte_count, hash);
  hash = loom_value_fact_hash_u32((uint32_t)reference.memory_space, hash);
  hash = loom_value_fact_hash_u32(reference.root_value_id, hash);
  hash = loom_value_fact_hash_u32(reference.alias_scope_id, hash);
  return loom_value_fact_hash_u32(reference.nullability, hash);
}

static uint32_t loom_value_fact_hash_raw_payload(
    loom_value_fact_raw_payload_t payload, uint32_t hash) {
  hash = loom_value_fact_hash_u32(payload.tag, hash);
  hash = loom_value_fact_hash_host_size(payload.length, hash);
  return loom_value_fact_hash_bytes(payload.data, payload.length, hash);
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
         lhs.nullability == rhs.nullability;
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
         lhs.alias_scope_id == rhs.alias_scope_id &&
         lhs.nullability == rhs.nullability;
}

static bool loom_value_fact_address_layout_equal(
    loom_value_fact_address_layout_t lhs,
    loom_value_fact_address_layout_t rhs) {
  if (lhs.kind != rhs.kind || lhs.rank != rhs.rank) return false;
  if (lhs.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) return true;
  if (lhs.rank == 0) return true;
  if (!lhs.strides || !rhs.strides) return lhs.strides == rhs.strides;
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

typedef struct loom_value_fact_encoded_operand_scale_schema_t {
  uint64_t scale_format;
  uint64_t secondary_scale_format;
  uint32_t scale_topology;
  uint32_t flags;
  uint16_t scale_group_element_count;
  uint16_t scale_operand_count;
} loom_value_fact_encoded_operand_scale_schema_t;

static loom_value_fact_encoded_operand_scale_schema_t
loom_value_fact_encoded_operand_scale_schema(
    loom_value_fact_encoded_operand_schema_t schema) {
  loom_value_fact_encoded_operand_scale_schema_t scale_schema;
  memset(&scale_schema, 0, sizeof(scale_schema));
  scale_schema.scale_format = schema.scale_format;
  scale_schema.secondary_scale_format = schema.secondary_scale_format;
  scale_schema.scale_topology = schema.scale_topology;
  scale_schema.flags = schema.flags;
  scale_schema.scale_group_element_count = schema.scale_group_element_count;
  scale_schema.scale_operand_count = schema.scale_operand_count;
  return scale_schema;
}

bool loom_value_fact_encoded_operand_schema_has_scale(
    loom_value_fact_encoded_operand_schema_t schema) {
  loom_value_fact_encoded_operand_scale_schema_t scale_schema =
      loom_value_fact_encoded_operand_scale_schema(schema);
  loom_value_fact_encoded_operand_scale_schema_t empty;
  memset(&empty, 0, sizeof(empty));
  return memcmp(&scale_schema, &empty, sizeof(scale_schema)) != 0;
}

bool loom_value_fact_encoded_operand_schema_scale_is_complete(
    loom_value_fact_encoded_operand_schema_t schema) {
  if (!loom_value_fact_encoded_operand_schema_has_scale(schema)) return true;
  return schema.scale_topology != 0 && schema.scale_group_element_count != 0 &&
         schema.scale_operand_count != 0;
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
  uint32_t hash = 2166136261u;
  hash = loom_value_fact_hash_u32((uint32_t)entry->kind, hash);
  switch (entry->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT:
      return loom_value_fact_hash_bytes(&entry->payload.uniform_element,
                                        sizeof(entry->payload.uniform_element),
                                        hash);
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES:
      hash = loom_value_fact_hash_host_size(
          entry->payload.small_static_lanes.count, hash);
      return loom_value_fact_hash_bytes(
          entry->payload.small_static_lanes.lanes,
          entry->payload.small_static_lanes.count * sizeof(loom_value_facts_t),
          hash);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA:
      return loom_value_fact_hash_bytes(&entry->payload.vector_iota,
                                        sizeof(entry->payload.vector_iota),
                                        hash);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK:
      return loom_value_fact_hash_bytes(
          &entry->payload.vector_prefix_mask,
          sizeof(entry->payload.vector_prefix_mask), hash);
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
  if (lhs->kind != rhs->kind) return false;
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
      if (lhs->payload.small_static_lanes.count == 0) return true;
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
      if (lhs->payload.type_payload.length == 0) return true;
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
  if (count == 0) return iree_ok_status();
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
  if (minimum_count <= table->extensions.capacity) return iree_ok_status();

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
  if (new_bucket_count == 0) new_bucket_count = 64;
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

static iree_status_t loom_value_fact_table_intern_extension_impl(
    loom_value_fact_table_t* table,
    const loom_value_fact_extension_entry_t* candidate,
    bool materialize_payload, loom_value_fact_extension_id_t* out_id) {
  loom_value_fact_extension_entry_t entry = *candidate;
  entry.content_hash = loom_value_fact_extension_hash(&entry);
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
  if (materialize_payload) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_materialize_extension_payload(table, &entry));
  }
  entry.next_id = table->extensions.buckets[bucket_index];
  table->extensions.entries[table->extensions.count] = entry;
  table->extensions.buckets[bucket_index] = id;
  table->extensions.count = new_count;
  *out_id = id;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_intern_extension(
    loom_value_fact_table_t* table,
    const loom_value_fact_extension_entry_t* candidate,
    loom_value_fact_extension_id_t* out_id) {
  return loom_value_fact_table_intern_extension_impl(table, candidate, true,
                                                     out_id);
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
  if (facts.extension_id > table->extensions.count) return NULL;
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
  if (count == 0) return true;
  if (!lhs || !rhs) return lhs == rhs;
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
  if (lhs.kind != rhs.kind || lhs.rank != rhs.rank) return false;
  if (lhs.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) return true;
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
         lhs.nullability == rhs.nullability;
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
         lhs.alias_scope_id == rhs.alias_scope_id &&
         lhs.nullability == rhs.nullability;
}

static bool loom_value_fact_table_extension_entries_equal(
    const loom_value_fact_table_t* lhs_table,
    const loom_value_fact_extension_entry_t* lhs,
    const loom_value_fact_table_t* rhs_table,
    const loom_value_fact_extension_entry_t* rhs) {
  if (!lhs || !rhs || lhs->kind != rhs->kind) return false;
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
      if (lhs->payload.type_payload.length == 0) return true;
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
// Scratch buffers
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out) {
  if (count <= table->scratch.facts.capacity) {
    *out = table->scratch.facts.values;
    return iree_ok_status();
  }
  loom_value_facts_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena, count,
                                                 sizeof(loom_value_facts_t),
                                                 (void**)&new_scratch));
  table->scratch.facts.values = new_scratch;
  table->scratch.facts.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out) {
  if (count <= table->scratch.value_ids.capacity) {
    *out = table->scratch.value_ids.values;
    return iree_ok_status();
  }
  loom_value_id_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena, count,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&new_scratch));
  table->scratch.value_ids.values = new_scratch;
  table->scratch.value_ids.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
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
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_uniform_element_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT) {
    return false;
  }
  if (out) *out = entry->payload.uniform_element;
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
  if (out) *out = entry->payload.small_static_lanes;
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
  if (out) *out = entry->payload.vector_iota;
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
  if (out) *out = entry->payload.vector_prefix_mask;
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
  if (out) *out = entry->payload.encoding_summary;
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
  if (out) *out = entry->payload.buffer_reference;
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
  if (out) *out = entry->payload.view_reference;
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
  if (out_payload) *out_payload = entry->payload.type_payload.data;
  if (out_payload_length) {
    *out_payload_length = entry->payload.type_payload.length;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity) {
  return loom_value_fact_table_initialize_with_arenas(table, arena, arena,
                                                      initial_capacity);
}

iree_status_t loom_value_fact_table_initialize_with_arenas(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* transient_arena,
    iree_host_size_t initial_capacity) {
  memset(table, 0, sizeof(*table));
  table->arena = arena;
  table->transient_arena = transient_arena;
  table->context.table = table;
  return loom_value_fact_table_allocate_initial_capacity(table,
                                                         initial_capacity);
}

void loom_value_fact_table_clear_scope(loom_value_fact_table_t* table) {
  for (iree_host_size_t i = 0; i < table->touched_count; ++i) {
    table->entries[table->touched_values[i]] = (loom_value_facts_t){0};
  }
  table->touched_count = 0;
  table->count = 0;
  table->extensions.entries = NULL;
  table->extensions.capacity = 0;
  table->extensions.count = 0;
  table->extensions.buckets = NULL;
  table->extensions.bucket_count = 0;
  table->scratch.facts.values = NULL;
  table->scratch.facts.capacity = 0;
  table->scratch.value_ids.values = NULL;
  table->scratch.value_ids.capacity = 0;
  table->context.table = table;
  table->context.function = (loom_func_like_t){0};
  table->context.target_bundle = NULL;
}

iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts) {
  IREE_ASSERT_NE(facts.known_divisor, 0);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_capacity(
      table, (iree_host_size_t)value_id + 1));
  if (table->entries[value_id].known_divisor == 0) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_value(table, value_id));
  }
  table->entries[value_id] = facts;
  if ((iree_host_size_t)value_id + 1 > table->count) {
    table->count = (iree_host_size_t)value_id + 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_clone_fact_array_between_tables(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_value_facts_t* source_facts, iree_host_size_t count,
    const loom_value_facts_t** out_facts) {
  *out_facts = NULL;
  if (count == 0) return iree_ok_status();
  loom_value_facts_t* cloned_facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(target->transient_arena, count,
                                                 sizeof(loom_value_facts_t),
                                                 (void**)&cloned_facts));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
        target, source, source_facts[i], &cloned_facts[i]));
  }
  *out_facts = cloned_facts;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_clone_fact(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t facts, loom_value_facts_t* out_facts) {
  if (facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
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
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_clone_fact_array_between_tables(
              target, source, source_entry->payload.small_static_lanes.lanes,
              source_entry->payload.small_static_lanes.count,
              &target_entry.payload.small_static_lanes.lanes));
      break;
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
        loom_value_fact_address_layout_t* target_layout =
            &target_entry.payload.encoding_summary.address_layout;
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_clone_fact_array_between_tables(
                target, source,
                source_entry->payload.encoding_summary.address_layout.strides,
                source_entry->payload.encoding_summary.address_layout.rank,
                &target_layout->strides));
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
      if (source_entry->payload.type_payload.length > 0) {
        void* data = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            target->transient_arena, source_entry->payload.type_payload.length,
            &data));
        memcpy(data, source_entry->payload.type_payload.data,
               source_entry->payload.type_payload.length);
        target_entry.payload.type_payload.data = data;
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown source fact extension kind %u",
                              (unsigned)source_entry->kind);
  }

  loom_value_fact_extension_id_t target_extension_id =
      LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_intern_extension_impl(
      target, &target_entry, false, &target_extension_id));
  *out_facts = facts;
  out_facts->extension_id = target_extension_id;
  return iree_ok_status();
}

static bool loom_value_fact_type_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static void loom_value_fact_mark_lane_distribution_for_type(
    loom_type_t type, loom_value_facts_t* facts) {
  if (loom_value_fact_type_is_i1(type)) {
    loom_value_facts_mark_lane_predicate(facts);
  } else {
    loom_value_facts_mark_lane_varying(facts);
  }
}

iree_status_t loom_value_fact_table_clone_fact_for_type(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module, loom_type_t type, loom_value_facts_t facts,
    loom_value_facts_t* out_facts) {
  *out_facts = facts;
  if (facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
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
  if (loom_value_facts_is_float(lhs_scalar) ||
      loom_value_facts_is_float(rhs_scalar)) {
    *out_facts = loom_value_facts_unknown();
    if (loom_value_facts_is_lane_varying(lhs_scalar) ||
        loom_value_facts_is_lane_varying(rhs_scalar)) {
      loom_value_fact_mark_lane_distribution_for_type(type, out_facts);
    } else if (loom_value_facts_is_uniform(lhs_scalar) &&
               loom_value_facts_is_uniform(rhs_scalar)) {
      loom_value_facts_mark_uniform(out_facts);
    }
  } else {
    loom_value_facts_meet(&lhs_scalar, &rhs_scalar, out_facts);
  }

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

  *out_facts = loom_value_facts_unknown();
  if (loom_value_facts_is_lane_varying(previous) ||
      loom_value_facts_is_lane_varying(next)) {
    loom_value_fact_mark_lane_distribution_for_type(type, out_facts);
  } else if (loom_value_facts_is_uniform(previous) &&
             loom_value_facts_is_uniform(next)) {
    loom_value_facts_mark_uniform(out_facts);
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

iree_status_t loom_value_fact_table_clone_defined_facts(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module) {
  for (iree_host_size_t i = 0; i < source->count; ++i) {
    if (source->entries[i].known_divisor == 0) continue;
    IREE_ASSERT_LT(i, (iree_host_size_t)LOOM_VALUE_ID_INVALID);
    const loom_value_id_t value_id = (loom_value_id_t)i;
    loom_value_facts_t cloned_facts = loom_value_facts_unknown();
    if (module && value_id < module->values.count) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
          target, source, module, loom_module_value_type(module, value_id),
          source->entries[i], &cloned_facts));
    } else {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source->entries[i], &cloned_facts));
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(target, value_id, cloned_facts));
  }
  return iree_ok_status();
}

static int64_t loom_value_fact_static_element_byte_count(loom_type_t type) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) return -1;
  return bit_count / 8;
}

static iree_status_t loom_value_fact_table_seed_buffer_arg(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_memory_space_t memory_space) {
  if (!loom_value_facts_is_unknown(
          loom_value_fact_table_lookup(table, value_id))) {
    return iree_ok_status();
  }
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .memory_space = memory_space,
      .root_value_id = value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_buffer_reference(
      &table->context, reference, &facts));
  return loom_value_fact_table_define(table, value_id, facts);
}

static iree_status_t loom_value_fact_table_seed_view_arg(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_type_t type) {
  if (!loom_value_facts_is_unknown(
          loom_value_fact_table_lookup(table, value_id))) {
    return iree_ok_status();
  }
  loom_value_fact_view_reference_t reference = {
      .base_byte_offset = loom_value_facts_exact_i64(0),
      .footprint_byte_length = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .root_minimum_alignment = 1,
      .static_element_byte_count =
          loom_value_fact_static_element_byte_count(type),
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(
      loom_value_facts_make_view_reference(&table->context, reference, &facts));
  return loom_value_fact_table_define(table, value_id, facts);
}

static iree_status_t loom_value_fact_table_define_if_changed(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_facts_t facts, bool* out_changed) {
  loom_value_facts_t old_facts = loom_value_fact_table_lookup(table, value_id);
  if (out_changed && !loom_value_facts_equal(old_facts, facts)) {
    *out_changed = true;
  }
  return loom_value_fact_table_define(table, value_id, facts);
}

static bool loom_value_fact_table_find_result(const loom_value_id_t* results,
                                              uint16_t result_count,
                                              loom_value_id_t value_id,
                                              uint16_t* out_result_index) {
  if (!results) return false;
  for (uint16_t i = 0; i < result_count; ++i) {
    if (results[i] != value_id) continue;
    *out_result_index = i;
    return true;
  }
  return false;
}

static bool loom_value_fact_table_dynamic_extent_type_supported(
    loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(scalar_type);
}

static loom_value_facts_t loom_value_fact_table_clamp_scalar_type_domain(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_facts_t facts) {
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return facts;
  }
  int64_t lo = 0;
  int64_t hi = 0;
  if (!loom_value_facts_scalar_type_domain(loom_type_element_type(type), &lo,
                                           &hi)) {
    return facts;
  }
  return loom_value_facts_clamp_domain(facts, lo, hi);
}

static iree_status_t loom_value_fact_table_seed_dynamic_extent(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, const loom_value_id_t* result_ids,
    uint16_t result_count, loom_value_facts_t* result_facts,
    bool* out_changed) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }
  if (!loom_value_fact_table_dynamic_extent_type_supported(
          loom_module_value_type(module, value_id))) {
    return iree_ok_status();
  }

  uint16_t result_index = 0;
  if (result_facts && loom_value_fact_table_find_result(
                          result_ids, result_count, value_id, &result_index)) {
    result_facts[result_index] = loom_value_fact_table_clamp_scalar_type_domain(
        module, value_id,
        loom_value_facts_non_negative_extent(result_facts[result_index]));
    return iree_ok_status();
  }

  loom_value_facts_t current = loom_value_fact_table_lookup(table, value_id);
  loom_value_facts_t extent = loom_value_fact_table_clamp_scalar_type_domain(
      module, value_id, loom_value_facts_non_negative_extent(current));
  return loom_value_fact_table_define_if_changed(table, value_id, extent,
                                                 out_changed);
}

static iree_status_t loom_value_fact_table_seed_scalar_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id) {
  if (!loom_value_facts_is_unknown(
          loom_value_fact_table_lookup(table, value_id))) {
    return iree_ok_status();
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define(
      table, value_id,
      loom_value_fact_table_clamp_scalar_type_domain(
          module, value_id, loom_value_facts_unknown()));
}

static loom_value_facts_t loom_value_fact_table_unknown_for_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (value_id >= module->values.count) {
    return facts;
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_type_is_scalar(type)) {
    facts =
        loom_value_fact_table_clamp_scalar_type_domain(module, value_id, facts);
  }
  return facts;
}

static iree_status_t loom_value_fact_table_seed_type_extent_facts(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_type_t type, const loom_value_id_t* result_ids, uint16_t result_count,
    loom_value_facts_t* result_facts, bool* out_changed) {
  if (!loom_type_is_shaped(type)) return iree_ok_status();
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (!loom_type_dim_is_dynamic_at(type, i)) continue;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_dynamic_extent(
        table, module, loom_type_dim_value_id_at(type, i), result_ids,
        result_count, result_facts, out_changed));
  }
  return iree_ok_status();
}

static void loom_value_fact_table_apply_operand_distribution(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts, loom_value_id_t result_id,
    loom_value_facts_t* result_facts) {
  if (result_id == LOOM_VALUE_ID_INVALID || result_id >= module->values.count) {
    return;
  }
  if (loom_value_facts_is_exact(*result_facts)) {
    loom_value_facts_mark_uniform(result_facts);
    return;
  }
  if (iree_any_bit_set(
          result_facts->flags,
          LOOM_VALUE_FACT_DISTRIBUTION_MASK | LOOM_VALUE_FACT_LANE_PREDICATE)) {
    return;
  }
  if (op->operand_count == 0) {
    return;
  }

  bool all_operands_uniform = true;
  bool any_operand_lane_varying = false;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_facts_t facts = operand_facts[i];
    if (loom_value_facts_is_lane_varying(facts) ||
        loom_value_facts_is_lane_predicate(facts)) {
      any_operand_lane_varying = true;
    }
    if (!loom_value_facts_is_uniform(facts)) {
      all_operands_uniform = false;
    }
  }

  if (any_operand_lane_varying) {
    loom_value_fact_mark_lane_distribution_for_type(
        loom_module_value_type(module, result_id), result_facts);
  } else if (all_operands_uniform) {
    loom_value_facts_mark_uniform(result_facts);
  }
}

static int64_t loom_value_fact_loop_iv_base_divisor(loom_value_facts_t facts) {
  if (!loom_value_facts_is_exact(facts) || facts.range_lo == INT64_MIN) {
    return facts.known_divisor;
  }
  // Preserve gcd(0, step) so a zero lower bound keeps step divisibility.
  return facts.range_lo >= 0 ? facts.range_lo : -facts.range_lo;
}

static bool loom_value_fact_counted_loop_last_reachable_iv(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step, int64_t* out_hi) {
  if (!loom_value_facts_is_exact(lower_bound) ||
      !loom_value_facts_is_exact(step) || step.range_lo <= 0 ||
      lower_bound.range_lo >= upper_bound.range_hi) {
    return false;
  }

  int64_t last_possible_offset = 0;
  if (!loom_checked_sub_i64(upper_bound.range_hi, lower_bound.range_lo,
                            &last_possible_offset) ||
      !loom_checked_sub_i64(last_possible_offset, 1, &last_possible_offset)) {
    return false;
  }

  int64_t stepped_offset = 0;
  const int64_t trip_index = last_possible_offset / step.range_lo;
  if (!loom_checked_mul_i64(trip_index, step.range_lo, &stepped_offset) ||
      !loom_checked_add_i64(lower_bound.range_lo, stepped_offset, out_hi)) {
    return false;
  }
  return true;
}

static loom_value_facts_t loom_value_fact_counted_loop_iv_facts(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return loom_value_facts_unknown();
  }

  int64_t lower_divisor = loom_value_fact_loop_iv_base_divisor(lower_bound);
  int64_t divisor = loom_gcd_i64(lower_divisor, step.known_divisor);

  int64_t hi = INT64_MAX;
  if (loom_value_fact_counted_loop_last_reachable_iv(lower_bound, upper_bound,
                                                     step, &hi)) {
    return loom_value_facts_make(lower_bound.range_lo, hi, divisor);
  }

  if (loom_checked_sub_i64(upper_bound.range_hi, 1, &hi)) {
    if (lower_bound.range_lo <= hi) {
      return loom_value_facts_make(lower_bound.range_lo, hi, divisor);
    }
  }
  return loom_value_facts_unknown();
}

static iree_status_t loom_value_fact_table_seed_loop_iv_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_loop_like_t loop = loom_loop_like_cast(module, parent_op);
  if (!loom_loop_like_isa(loop) || !loom_loop_like_has_counted_range(loop)) {
    return iree_ok_status();
  }
  if (loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE ||
      loop.vtable->iv_block_arg_index >= block->arg_count) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body || body->block_count == 0 ||
      block != loom_region_const_entry_block(body)) {
    return iree_ok_status();
  }

  loom_value_facts_t lower_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_lower_bound(loop));
  loom_value_facts_t upper_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_upper_bound(loop));
  loom_value_facts_t step =
      loom_value_fact_table_lookup(table, loom_loop_like_step(loop));
  loom_value_facts_t iv_facts =
      loom_value_fact_counted_loop_iv_facts(lower_bound, upper_bound, step);
  loom_value_id_t iv_id =
      loom_block_arg_id(block, loop.vtable->iv_block_arg_index);
  return loom_value_fact_table_define(table, iv_id, iv_facts);
}

static loom_value_fact_memory_space_t
loom_value_fact_table_seeded_buffer_memory_space(const loom_module_t* module,
                                                 const loom_block_t* block,
                                                 const loom_op_t* parent_op) {
  const loom_region_t* region = block->parent_region;
  if (!parent_op || !region) {
    return LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, parent_op);
  loom_region_t* const* regions = loom_op_regions(parent_op);
  for (uint8_t i = 0; i < parent_op->region_count; ++i) {
    if (regions[i] != region) {
      continue;
    }
    const loom_region_descriptor_t* descriptor =
        loom_op_vtable_region_descriptor(vtable, i);
    if (descriptor &&
        iree_any_bit_set(descriptor->flags, LOOM_REGION_GLOBAL_BUFFER_ARGS)) {
      return LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
    }
    return LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  }
  return LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
}

static bool loom_value_fact_table_block_contains_arg(const loom_block_t* block,
                                                     loom_value_id_t value_id) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (loom_block_arg_id(block, i) == value_id) return true;
  }
  return false;
}

static iree_status_t loom_value_fact_table_apply_func_predicates(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_func_like_t function = loom_func_like_cast(module, parent_op);
  if (!loom_func_like_isa(function)) return iree_ok_status();
  loom_region_t* body = loom_func_like_body(function);
  if (!body || block != loom_region_const_entry_block(body)) {
    return iree_ok_status();
  }

  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    if (predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) continue;
    int64_t raw_value_id = predicate->args[0];
    if (raw_value_id < 0 || raw_value_id > UINT32_MAX) continue;
    loom_value_id_t value_id = (loom_value_id_t)raw_value_id;
    if (!loom_value_fact_table_block_contains_arg(block, value_id)) {
      continue;
    }
    loom_value_facts_t facts = loom_value_fact_table_lookup(table, value_id);
    loom_value_facts_apply_predicate(&facts, predicate);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, value_id, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_seed_block_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_value_fact_memory_space_t buffer_memory_space =
      loom_value_fact_table_seeded_buffer_memory_space(module, block,
                                                       parent_op);
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t value_id = loom_block_arg_id(block, i);
    if (value_id >= module->values.count) {
      continue;
    }
    loom_type_t type = loom_module_value_type(module, value_id);
    if (!loom_value_fact_table_has_entry(table, value_id)) {
      if (loom_type_is_buffer(type)) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_buffer_arg(
            table, value_id, buffer_memory_space));
      } else if (loom_type_is_view(type)) {
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_seed_view_arg(table, value_id, type));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_seed_scalar_arg(table, module, value_id));
      }
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, type, /*result_ids=*/NULL, /*result_count=*/0,
        /*result_facts=*/NULL, /*out_changed=*/NULL));
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_apply_func_predicates(
      table, module, block, parent_op));
  return loom_value_fact_table_seed_loop_iv_arg(table, module, block,
                                                parent_op);
}

//===----------------------------------------------------------------------===//
// CFG block argument summaries
//===----------------------------------------------------------------------===//

static bool loom_value_fact_table_branch_payload_for_successor(
    const loom_op_t* terminator, const loom_block_t* successor,
    const loom_value_id_t** out_args, uint16_t* out_arg_count) {
  *out_args = NULL;
  *out_arg_count = 0;
  if (!terminator || terminator->successor_count != 1 ||
      loom_op_const_successors(terminator)[0] != successor) {
    return false;
  }
  if (terminator->operand_count != successor->arg_count) {
    return false;
  }
  *out_args = loom_op_const_operands(terminator);
  *out_arg_count = terminator->operand_count;
  return true;
}

static bool loom_value_fact_table_block_has_backedge(
    const loom_cfg_graph_t* graph, uint16_t block_index) {
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    uint16_t predecessor_index = predecessors.values[i];
    if (loom_cfg_graph_block_is_reachable(graph, predecessor_index) &&
        predecessor_index >= block_index) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_value_fact_table_define_block_arg_facts(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t arg_id, loom_value_facts_t facts, bool* out_changed) {
  if (arg_id >= module->values.count) {
    return iree_ok_status();
  }
  loom_type_t type = loom_module_value_type(module, arg_id);
  if (out_changed &&
      (!loom_value_fact_table_has_entry(table, arg_id) ||
       !loom_value_fact_table_facts_equal_for_type(
           module, type, table, loom_value_fact_table_lookup(table, arg_id),
           table, facts))) {
    *out_changed = true;
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, arg_id, facts));
  return loom_value_fact_table_seed_type_extent_facts(
      table, module, type, /*result_ids=*/NULL, /*result_count=*/0,
      /*result_facts=*/NULL, out_changed);
}

static iree_status_t loom_value_fact_table_join_cfg_block_arg_incoming(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_type_t type, loom_value_id_t source_value, bool* inout_has_facts,
    loom_value_facts_t* inout_facts) {
  // A forward CFG solve can see backedge operands before the producing block
  // has run in the current iteration. Undefined entries are skipped until the
  // producer contributes facts; values that are defined-but-unknown still have
  // entries and participate in the meet below.
  if (!loom_value_fact_table_has_entry(table, source_value)) {
    return iree_ok_status();
  }
  loom_value_facts_t source_facts =
      loom_value_fact_table_lookup(table, source_value);
  if (!*inout_has_facts) {
    *inout_facts = source_facts;
    *inout_has_facts = true;
    return iree_ok_status();
  }
  loom_value_facts_t joined = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
      table, module, type, table, *inout_facts, table, source_facts, &joined));
  *inout_facts = joined;
  return iree_ok_status();
}

static bool loom_value_fact_table_selector_is_lane_varying(
    const loom_value_fact_table_t* table, loom_value_id_t selector_value_id) {
  if (selector_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_value_fact_table_has_entry(table, selector_value_id)) {
    return false;
  }
  const loom_value_facts_t selector_facts =
      loom_value_fact_table_lookup(table, selector_value_id);
  return loom_value_facts_is_lane_varying(selector_facts) ||
         loom_value_facts_is_lane_predicate(selector_facts);
}

static bool loom_value_fact_table_block_has_payload_edge_to_target(
    const loom_cfg_graph_t* graph, uint16_t source_block_index,
    const loom_block_t* target_block, uint16_t arg_index) {
  loom_cfg_edge_index_span_t successor_edges =
      loom_cfg_graph_successor_edges(graph, source_block_index);
  for (iree_host_size_t i = 0; i < successor_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        loom_cfg_graph_edge(graph, successor_edges.values[i]);
    if (edge == NULL) continue;
    const loom_value_id_t* edge_args = NULL;
    uint16_t edge_arg_count = 0;
    if (loom_value_fact_table_branch_payload_for_successor(
            edge->terminator, target_block, &edge_args, &edge_arg_count) &&
        arg_index < edge_arg_count) {
      return true;
    }
  }
  return false;
}

static bool loom_value_fact_table_edge_is_selected_by_lane_varying_control(
    const loom_value_fact_table_t* table, const loom_cfg_graph_t* graph,
    const loom_cfg_edge_info_t* incoming_edge, const loom_block_t* target_block,
    uint16_t arg_index) {
  if (incoming_edge == NULL) return false;

  loom_cfg_edge_index_span_t arm_predecessor_edges =
      loom_cfg_graph_predecessor_edges(graph,
                                       incoming_edge->source_block_index);
  for (iree_host_size_t i = 0; i < arm_predecessor_edges.count; ++i) {
    const loom_cfg_edge_info_t* guard_edge =
        loom_cfg_graph_edge(graph, arm_predecessor_edges.values[i]);
    if (guard_edge == NULL || !loom_value_fact_table_selector_is_lane_varying(
                                  table, guard_edge->selector_value_id)) {
      continue;
    }

    loom_cfg_edge_index_span_t guard_successor_edges =
        loom_cfg_graph_successor_edges(graph, guard_edge->source_block_index);
    for (iree_host_size_t j = 0; j < guard_successor_edges.count; ++j) {
      const loom_cfg_edge_info_t* sibling_edge =
          loom_cfg_graph_edge(graph, guard_successor_edges.values[j]);
      if (sibling_edge == NULL ||
          sibling_edge->terminator != guard_edge->terminator ||
          sibling_edge->target_block_index ==
              incoming_edge->source_block_index) {
        continue;
      }
      if (loom_value_fact_table_block_has_payload_edge_to_target(
              graph, sibling_edge->target_block_index, target_block,
              arg_index)) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t loom_value_fact_table_compute_cfg_block_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_cfg_graph_t* graph, uint16_t block_index, uint16_t arg_index,
    bool has_backedge, uint32_t iteration, bool* out_changed) {
  const loom_block_t* block = graph->blocks[block_index].block;
  if (!block || arg_index >= block->arg_count) {
    return iree_ok_status();
  }
  loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
  if (arg_id >= module->values.count) {
    return iree_ok_status();
  }
  loom_type_t type = loom_module_value_type(module, arg_id);

  bool has_facts = false;
  bool selected_by_lane_varying_control = false;
  bool all_source_values_match = true;
  loom_value_id_t first_source_value = LOOM_VALUE_ID_INVALID;
  loom_value_facts_t incoming_facts = loom_value_facts_unknown();
  loom_cfg_edge_index_span_t predecessor_edges =
      loom_cfg_graph_predecessor_edges(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
    const loom_cfg_edge_info_t* predecessor_edge =
        loom_cfg_graph_edge(graph, predecessor_edges.values[i]);
    if (predecessor_edge == NULL) continue;
    uint16_t predecessor_index = predecessor_edge->source_block_index;
    if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) {
      continue;
    }
    const loom_block_t* predecessor_block =
        graph->blocks[predecessor_index].block;
    const loom_value_id_t* edge_args = NULL;
    uint16_t edge_arg_count = 0;
    if (!predecessor_block ||
        !loom_value_fact_table_branch_payload_for_successor(
            predecessor_block->last_op, block, &edge_args, &edge_arg_count) ||
        arg_index >= edge_arg_count) {
      continue;
    }
    const loom_value_id_t source_value = edge_args[arg_index];
    if (first_source_value == LOOM_VALUE_ID_INVALID) {
      first_source_value = source_value;
    } else if (source_value != first_source_value) {
      all_source_values_match = false;
    }
    selected_by_lane_varying_control =
        selected_by_lane_varying_control ||
        loom_value_fact_table_edge_is_selected_by_lane_varying_control(
            table, graph, predecessor_edge, block, arg_index);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_join_cfg_block_arg_incoming(
        table, module, type, source_value, &has_facts, &incoming_facts));
  }
  if (!has_facts) {
    return iree_ok_status();
  }

  loom_value_facts_t facts = incoming_facts;
  if (has_backedge && loom_value_fact_table_has_entry(table, arg_id)) {
    loom_value_facts_t current_facts =
        loom_value_fact_table_lookup(table, arg_id);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_widen_for_type(
        table, module, type, table, current_facts, table, incoming_facts,
        iteration, &facts));
  }
  if (selected_by_lane_varying_control && !all_source_values_match &&
      !loom_value_facts_is_exact(facts)) {
    loom_value_fact_mark_lane_distribution_for_type(type, &facts);
  }
  return loom_value_fact_table_define_block_arg_facts(table, module, arg_id,
                                                      facts, out_changed);
}

static iree_status_t loom_value_fact_table_compute_cfg_block_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_cfg_graph_t* graph, uint16_t block_index, uint32_t iteration,
    bool* out_changed) {
  const loom_block_t* block = graph->blocks[block_index].block;
  if (!block || !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return iree_ok_status();
  }
  if (block_index == 0) {
    return iree_ok_status();
  }
  const bool has_backedge =
      loom_value_fact_table_block_has_backedge(graph, block_index);
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_arg(
        table, module, graph, block_index, i, has_backedge, iteration,
        out_changed));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Loop-like region summaries
//===----------------------------------------------------------------------===//

#define LOOM_VALUE_FACT_CFG_MAX_ITERATIONS 16
#define LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS 8

static loom_op_t* loom_value_fact_region_terminator(loom_region_t* region) {
  if (!region || region->block_count == 0) return NULL;
  loom_block_t* block = loom_region_entry_block(region);
  return block && block->op_count > 0 ? block->last_op : NULL;
}

static iree_status_t loom_value_fact_table_compute_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op);

static iree_status_t loom_value_fact_table_compute_cfg_block_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, bool* out_changed) {
  loom_op_t* op = NULL;
  loom_block_for_each_op((loom_block_t*)block, op) {
    bool op_changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_and_report(
        table, module, op, &op_changed));
    *out_changed = *out_changed || op_changed;
    const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
    if (vtable && vtable->loop_like) {
      // Loop-like fact callbacks summarize and visit their nested regions.
      continue;
    }
    loom_region_t** regions = loom_op_regions(op);
    for (uint8_t i = 0; i < op->region_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
          table, module, regions[i], op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_cfg_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op) {
  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(module, region, table->transient_arena, &graph));

  if (region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* entry_block = loom_region_entry_block(region);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_block_args(
      table, module, entry_block, parent_op));

  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_CFG_MAX_ITERATIONS;
       ++iteration) {
    bool changed = false;
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = graph.blocks[block_index].block;
      if (!block || !loom_cfg_graph_block_is_reachable(&graph, block_index)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_args(
          table, module, &graph, block_index, iteration, &changed));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_tree(
          table, module, block, &changed));
    }
    if (!changed) {
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op) {
  if (!region) return iree_ok_status();
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return loom_value_fact_table_compute_cfg_region_tree(table, module, region,
                                                         parent_op);
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_seed_block_args(table, module, block, parent_op));
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op(table, module, op));
      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (vtable && vtable->loop_like) {
        // Loop-like fact callbacks summarize and visit their nested regions.
        continue;
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
            table, module, regions[i], op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_seed_projected_func_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  if (!loom_func_like_isa(function) || parent_op != function.op || !region ||
      region->block_count == 0) {
    return iree_ok_status();
  }

  uint8_t region_index = LOOM_REGION_INDEX_NONE;
  for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
    if (loom_func_like_region(function, i) == region) {
      region_index = i;
      break;
    }
  }
  if (region_index == LOOM_REGION_INDEX_NONE ||
      !loom_func_like_region_projects_args(module, function, region_index)) {
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  loom_block_t* entry_block = loom_region_entry_block(region);
  if (!arguments || !entry_block) return iree_ok_status();

  uint16_t projected_count = iree_min(argument_count, entry_block->arg_count);
  for (uint16_t i = 0; i < projected_count; ++i) {
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(table, arguments[i]);
    if (loom_value_facts_is_unknown(facts)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
        table, loom_block_arg_id(entry_block, i), facts));
  }
  return iree_ok_status();
}

static bool loom_value_fact_counted_loop_proven_zero_trip(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return false;
  }
  return lower_bound.range_lo >= upper_bound.range_hi;
}

static bool loom_value_fact_counted_loop_proven_at_least_one_trip(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return false;
  }
  return lower_bound.range_hi < upper_bound.range_lo;
}

static uint16_t loom_value_fact_loop_carried_arg_offset(loom_loop_like_t loop) {
  return loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
             ? 0
             : (uint16_t)loop.vtable->iv_block_arg_index + 1;
}

static uint16_t loom_value_fact_loop_state_count(loom_loop_like_t loop) {
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  uint16_t count = iter_args.count;
  if (count > loop.op->result_count) count = loop.op->result_count;
  return count;
}

static iree_status_t loom_value_fact_table_allocate_fact_array(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out_facts) {
  *out_facts = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(table->transient_arena, count,
                                   sizeof(loom_value_facts_t),
                                   (void**)out_facts);
}

static iree_status_t loom_value_fact_table_initialize_loop_state(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, loom_value_facts_t** out_init_facts,
    loom_value_facts_t** out_current_facts, loom_type_t** out_types) {
  uint16_t count = loom_value_fact_loop_state_count(loop);
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, out_init_facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, count, out_current_facts));
  *out_types = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, count, sizeof(loom_type_t), (void**)out_types));
  }

  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_id_t value_id = iter_args.values[i];
    (*out_init_facts)[i] = loom_value_fact_table_lookup(table, value_id);
    (*out_current_facts)[i] = (*out_init_facts)[i];
    (*out_types)[i] = value_id < module->values.count
                          ? loom_module_value_type(module, value_id)
                          : loom_type_none();
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_define_loop_entry_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, uint16_t arg_offset, const loom_value_facts_t* facts,
    uint16_t count) {
  if (!region || region->block_count == 0) return iree_ok_status();
  loom_block_t* block = loom_region_entry_block(region);
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t arg_index = arg_offset + i;
    if (arg_index >= block->arg_count) break;
    loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
    if (arg_id >= module->values.count) continue;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, arg_id, facts[i]));
  }
  return iree_ok_status();
}

static void loom_value_fact_table_collect_terminator_operands(
    loom_value_fact_table_t* table, const loom_op_t* terminator,
    uint16_t operand_offset, loom_value_facts_t* facts, uint16_t count) {
  const loom_value_id_t* operands =
      terminator ? loom_op_const_operands(terminator) : NULL;
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t operand_index = operand_offset + i;
    facts[i] =
        (terminator && operand_index < terminator->operand_count)
            ? loom_value_fact_table_lookup(table, operands[operand_index])
            : loom_value_facts_unknown();
  }
}

static iree_status_t loom_value_fact_table_join_loop_backedge(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_type_t* types, const loom_value_facts_t* init_facts,
    const loom_value_facts_t* yielded_facts, const loom_value_facts_t* current,
    uint16_t count, uint32_t iteration, loom_value_facts_t* next,
    bool* out_changed) {
  *out_changed = false;
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_facts_t joined = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
        table, module, types[i], table, init_facts[i], table, yielded_facts[i],
        &joined));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_widen_for_type(
        table, module, types[i], table, current[i], table, joined, iteration,
        &next[i]));
    if (!loom_value_fact_table_facts_equal_for_type(
            module, types[i], table, current[i], table, next[i])) {
      *out_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_define_loop_results(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    loom_value_facts_t* result_facts, uint16_t result_count,
    bool* out_changed) {
  const loom_value_id_t* results = loom_op_const_results(op);
  if (result_count < op->result_count) {
    loom_value_facts_t* full_result_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
        table, op->result_count, &full_result_facts));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      full_result_facts[i] =
          i < result_count ? result_facts[i] : loom_value_facts_unknown();
    }
    result_facts = full_result_facts;
    result_count = op->result_count;
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, loom_module_value_type(module, result), results,
        op->result_count, result_facts, out_changed));
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      continue;
    }
    loom_value_facts_t facts =
        i < result_count ? result_facts[i] : loom_value_facts_unknown();
    loom_type_t type = loom_module_value_type(module, result);
    if (out_changed &&
        !loom_value_fact_table_facts_equal_for_type(
            module, type, table, loom_value_fact_table_lookup(table, result),
            table, facts)) {
      *out_changed = true;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, result, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_counted_loop_summary(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body || body->block_count == 0) {
    return iree_ok_status();
  }
  uint16_t count = loom_value_fact_loop_state_count(loop);
  if (count == 0) {
    return loom_value_fact_table_compute_region_tree(table, module, body,
                                                     loop.op);
  }

  loom_value_facts_t* init_facts = NULL;
  loom_value_facts_t* current_facts = NULL;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_state(
      table, module, loop, &init_facts, &current_facts, &types));

  loom_value_facts_t* yielded_facts = NULL;
  loom_value_facts_t* next_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &yielded_facts));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &next_facts));

  uint16_t carried_arg_offset = loom_value_fact_loop_carried_arg_offset(loop);
  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS;
       ++iteration) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, carried_arg_offset, current_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);

    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_join_loop_backedge(
        table, module, types, init_facts, yielded_facts, current_facts, count,
        iteration, next_facts, &changed));
    memcpy(current_facts, next_facts, count * sizeof(loom_value_facts_t));
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, carried_arg_offset, current_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);
  }

  loom_value_facts_t lower_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_lower_bound(loop));
  loom_value_facts_t upper_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_upper_bound(loop));
  loom_value_facts_t step =
      loom_value_fact_table_lookup(table, loom_loop_like_step(loop));
  bool zero_trip = loom_value_fact_counted_loop_proven_zero_trip(
      lower_bound, upper_bound, step);
  bool at_least_one_trip =
      loom_value_fact_counted_loop_proven_at_least_one_trip(lower_bound,
                                                            upper_bound, step);

  loom_value_facts_t* result_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &result_facts));
  for (uint16_t i = 0; i < count; ++i) {
    if (zero_trip) {
      result_facts[i] = init_facts[i];
    } else if (at_least_one_trip) {
      result_facts[i] = yielded_facts[i];
    } else {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          table, module, types[i], table, init_facts[i], table,
          yielded_facts[i], &result_facts[i]));
    }
  }
  return loom_value_fact_table_define_loop_results(
      table, module, loop.op, result_facts, count, out_changed);
}

static iree_status_t loom_value_fact_table_compute_condition_loop_summary(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  loom_region_t* body = loom_loop_like_body(loop);
  if (!condition_region || !body || condition_region->block_count == 0 ||
      body->block_count == 0) {
    return iree_ok_status();
  }
  uint16_t count = loom_value_fact_loop_state_count(loop);
  if (count == 0) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    return loom_value_fact_table_compute_region_tree(table, module, body,
                                                     loop.op);
  }

  loom_value_facts_t* init_facts = NULL;
  loom_value_facts_t* current_facts = NULL;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_state(
      table, module, loop, &init_facts, &current_facts, &types));

  loom_value_facts_t* forwarded_facts = NULL;
  loom_value_facts_t* yielded_facts = NULL;
  loom_value_facts_t* next_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, count, &forwarded_facts));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &yielded_facts));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &next_facts));

  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS;
       ++iteration) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, condition_region, /*arg_offset=*/0, current_facts,
        count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    loom_op_t* condition = loom_value_fact_region_terminator(condition_region);
    loom_value_fact_table_collect_terminator_operands(
        table, condition, /*operand_offset=*/1, forwarded_facts, count);

    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, /*arg_offset=*/0, forwarded_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);

    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_join_loop_backedge(
        table, module, types, init_facts, yielded_facts, current_facts, count,
        iteration, next_facts, &changed));
    memcpy(current_facts, next_facts, count * sizeof(loom_value_facts_t));
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, condition_region, /*arg_offset=*/0, current_facts,
        count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    loom_op_t* condition = loom_value_fact_region_terminator(condition_region);
    loom_value_fact_table_collect_terminator_operands(
        table, condition, /*operand_offset=*/1, forwarded_facts, count);

    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, /*arg_offset=*/0, forwarded_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);
  }

  return loom_value_fact_table_define_loop_results(
      table, module, loop.op, forwarded_facts, count, out_changed);
}

static iree_status_t loom_value_fact_table_compute_loop_like_summary(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    bool* out_changed) {
  loom_loop_like_t loop = loom_loop_like_cast(module, op);
  if (!loom_loop_like_isa(loop)) return iree_ok_status();
  if (loom_loop_like_condition_region(loop)) {
    return loom_value_fact_table_compute_condition_loop_summary(
        table, module, loop, out_changed);
  }
  if (loom_loop_like_has_counted_range(loop)) {
    return loom_value_fact_table_compute_counted_loop_summary(
        table, module, loop, out_changed);
  }
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, regions[i], op));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Forward pass: compute facts for ops
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op) {
  return loom_value_fact_table_compute_op_and_report(table, module, op, NULL);
}

iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed) {
  if (out_changed) *out_changed = false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (vtable && vtable->loop_like) {
    return loom_value_fact_table_compute_loop_like_summary(
        table, module, (loom_op_t*)op, out_changed);
  }
  if (!vtable || !vtable->infer_facts) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (results[i] == LOOM_VALUE_ID_INVALID ||
          results[i] >= module->values.count) {
        continue;
      }
      // A result with no op-specific inference is still defined. Absence from
      // the table is reserved for not-yet-computed or unreachable values.
      loom_value_facts_t unknown_facts =
          loom_value_fact_table_unknown_for_value(module, results[i]);
      if (out_changed &&
          (!loom_value_fact_table_has_entry(table, results[i]) ||
           !loom_value_fact_table_facts_equal_for_type(
               module, loom_module_value_type(module, results[i]), table,
               loom_value_fact_table_lookup(table, results[i]), table,
               unknown_facts))) {
        *out_changed = true;
      }
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, results[i], unknown_facts));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
          table, module, loom_module_value_type(module, results[i]),
          /*result_ids=*/NULL, /*result_count=*/0, /*result_facts=*/NULL,
          out_changed));
    }
    return iree_ok_status();
  }

  // Get scratch for operand + result facts.
  iree_host_size_t total =
      (iree_host_size_t)op->operand_count + op->result_count;
  loom_value_facts_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_facts_scratch(table, total, &scratch));
  loom_value_facts_t* operand_facts = scratch;
  loom_value_facts_t* result_facts = scratch + op->operand_count;

  // Gather operand facts from the table.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    operand_facts[i] = loom_value_fact_table_lookup(table, operands[i]);
  }

  // Call the fact inference function.
  IREE_RETURN_IF_ERROR(vtable->infer_facts(&table->context, module, op,
                                           operand_facts, result_facts));

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_fact_table_apply_operand_distribution(
        module, op, operand_facts, results[i], &result_facts[i]);
  }

  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        results[i] >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, loom_module_value_type(module, results[i]), results,
        op->result_count, result_facts, out_changed));
  }

  // Store result facts.
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      if (out_changed) {
        loom_value_facts_t old_facts =
            loom_value_fact_table_lookup(table, results[i]);
        loom_type_t result_type = loom_module_value_type(module, results[i]);
        if (!loom_value_fact_table_facts_equal_for_type(module, result_type,
                                                        table, old_facts, table,
                                                        result_facts[i])) {
          *out_changed = true;
        }
      }
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, results[i], result_facts[i]));
    }
  }

  return iree_ok_status();
}

iree_status_t loom_value_fact_table_compute_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  table->context.function = function;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_projected_func_args(
      table, module, function, region, parent_op));
  return loom_value_fact_table_compute_region_tree(table, module, region,
                                                   parent_op);
}

iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            const loom_module_t* module,
                                            loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();
  const uint8_t body_region_index = loom_func_like_body_region_index(function);
  for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
    if (i == body_region_index) continue;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region(
        table, module, function, loom_func_like_region(function, i),
        function.op));
  }
  return loom_value_fact_table_compute_region(table, module, function, body,
                                              function.op);
}
