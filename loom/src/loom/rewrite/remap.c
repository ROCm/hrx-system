// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/remap.h"

#include <string.h>

#include "loom/ir/module.h"

// Recursive remapping only follows payload graphs that should be shallow in
// production IR. Caps keep malformed modules from consuming the C stack.
#define LOOM_IR_REMAP_MAX_ENCODING_DEPTH 16
#define LOOM_IR_REMAP_MAX_LOCATION_DEPTH 16

static bool loom_ir_remap_is_initialized(const loom_ir_remap_t* remap) {
  return remap && remap->source_module && remap->target_module && remap->arena;
}

static iree_host_size_t loom_ir_remap_value_hash(loom_value_id_t value_id) {
  uint32_t hash = value_id;
  hash ^= hash >> 16;
  hash *= 0x7feb352du;
  hash ^= hash >> 15;
  hash *= 0x846ca68bu;
  hash ^= hash >> 16;
  return (iree_host_size_t)hash;
}

static iree_status_t loom_ir_remap_value_map_capacity_for_count(
    iree_host_size_t count, iree_host_size_t* out_capacity) {
  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_mul(count, 2, &minimum_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remap value map capacity overflow");
  }
  minimum_capacity = iree_max(minimum_capacity, 16);
  iree_host_size_t capacity =
      iree_host_size_next_power_of_two(minimum_capacity);
  if (capacity < minimum_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remap value map capacity overflow");
  }
  *out_capacity = capacity;
  return iree_ok_status();
}

static void loom_ir_remap_initialize_value_map_entries(
    loom_ir_remap_value_entry_t* entries, iree_host_size_t capacity) {
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    entries[i] = (loom_ir_remap_value_entry_t){
        .source_value = LOOM_VALUE_ID_INVALID,
        .target_value = LOOM_VALUE_ID_INVALID,
    };
  }
}

static loom_ir_remap_value_entry_t* loom_ir_remap_find_value_map_slot(
    loom_ir_remap_value_entry_t* entries, iree_host_size_t capacity,
    loom_value_id_t source_value) {
  IREE_ASSERT(capacity > 0);
  IREE_ASSERT(iree_host_size_is_power_of_two(capacity));
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot =
      loom_ir_remap_value_hash(source_value) & (iree_host_size_t)mask;
  while (true) {
    loom_ir_remap_value_entry_t* entry = &entries[slot];
    if (entry->source_value == LOOM_VALUE_ID_INVALID ||
        entry->source_value == source_value) {
      return entry;
    }
    slot = (slot + 1) & mask;
  }
}

static const loom_ir_remap_value_entry_t*
loom_ir_remap_find_const_value_map_slot(const loom_ir_remap_t* remap,
                                        loom_value_id_t source_value) {
  if (!remap || remap->value_map_entry_capacity == 0) {
    return NULL;
  }
  const loom_ir_remap_value_entry_t* entry = loom_ir_remap_find_value_map_slot(
      remap->value_map_entries, remap->value_map_entry_capacity, source_value);
  return entry->source_value == source_value ? entry : NULL;
}

static void loom_ir_remap_insert_value_map_entry(
    loom_ir_remap_value_entry_t* entries, iree_host_size_t capacity,
    loom_value_id_t source_value, loom_value_id_t target_value) {
  loom_ir_remap_value_entry_t* entry =
      loom_ir_remap_find_value_map_slot(entries, capacity, source_value);
  IREE_ASSERT(entry->source_value == LOOM_VALUE_ID_INVALID);
  *entry = (loom_ir_remap_value_entry_t){
      .source_value = source_value,
      .target_value = target_value,
  };
}

static iree_status_t loom_ir_remap_ensure_value_map_capacity(
    loom_ir_remap_t* remap, iree_host_size_t required_count) {
  if (required_count <= remap->value_map_entry_capacity / 2) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_ir_remap_value_map_capacity_for_count(
      required_count, &new_capacity));
  loom_ir_remap_value_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      remap->arena, new_capacity, sizeof(*new_entries), (void**)&new_entries));
  loom_ir_remap_initialize_value_map_entries(new_entries, new_capacity);
  for (iree_host_size_t i = 0; i < remap->value_map_entry_capacity; ++i) {
    const loom_ir_remap_value_entry_t old_entry = remap->value_map_entries[i];
    if (old_entry.source_value == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_ir_remap_insert_value_map_entry(new_entries, new_capacity,
                                         old_entry.source_value,
                                         old_entry.target_value);
  }
  remap->value_map_entries = new_entries;
  remap->value_map_entry_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t loom_ir_remap_initialize(const loom_module_t* source_module,
                                       loom_module_t* target_module,
                                       iree_arena_allocator_t* arena,
                                       const loom_ir_remap_options_t* options,
                                       loom_ir_remap_t* out_remap) {
  loom_ir_remap_t remap = {
      .source_module = source_module,
      .target_module = target_module,
      .arena = arena,
      .source_value_snapshot_count = source_module->values.count,
      .allow_unmapped_values = options ? options->allow_unmapped_values : false,
      .remap_symbol = options ? options->remap_symbol
                              : loom_ir_remap_symbol_callback_empty(),
  };

  *out_remap = remap;
  return iree_ok_status();
}

iree_status_t loom_ir_remap_map_value(loom_ir_remap_t* remap,
                                      loom_value_id_t source_value,
                                      loom_value_id_t target_value) {
  if (source_value >= remap->source_module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source value %%%u out of range (source module has %" PRIhsz " values)",
        (unsigned)source_value, remap->source_module->values.count);
  }
  if (source_value >= remap->source_value_snapshot_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source value %%%u was defined after this remap's %" PRIhsz
        "-value source snapshot",
        (unsigned)source_value, remap->source_value_snapshot_count);
  }
  if (target_value >= remap->target_module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target value %%%u out of range (target module has %" PRIhsz " values)",
        (unsigned)target_value, remap->target_module->values.count);
  }
  loom_ir_remap_value_entry_t* entry = NULL;
  if (remap->value_map_entry_capacity > 0) {
    entry = loom_ir_remap_find_value_map_slot(remap->value_map_entries,
                                              remap->value_map_entry_capacity,
                                              source_value);
    if (entry->source_value == source_value) {
      entry->target_value = target_value;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_ir_remap_ensure_value_map_capacity(
      remap, remap->mapped_value_count + 1));
  entry = loom_ir_remap_find_value_map_slot(
      remap->value_map_entries, remap->value_map_entry_capacity, source_value);
  IREE_ASSERT(entry->source_value == LOOM_VALUE_ID_INVALID);
  *entry = (loom_ir_remap_value_entry_t){
      .source_value = source_value,
      .target_value = target_value,
  };
  ++remap->mapped_value_count;
  return iree_ok_status();
}

iree_status_t loom_ir_remap_map_values(loom_ir_remap_t* remap,
                                       const loom_value_id_t* source_values,
                                       const loom_value_id_t* target_values,
                                       iree_host_size_t value_count) {
  if (value_count > 0 && (!source_values || !target_values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty source and target value arrays require payloads");
  }
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(remap, source_values[i], target_values[i]));
  }
  return iree_ok_status();
}

bool loom_ir_remap_try_lookup_value(const loom_ir_remap_t* remap,
                                    loom_value_id_t source_value,
                                    loom_value_id_t* out_target_value) {
  if (out_target_value) {
    *out_target_value = LOOM_VALUE_ID_INVALID;
  }
  if (!remap || source_value >= remap->source_value_snapshot_count) {
    return false;
  }
  const loom_ir_remap_value_entry_t* entry =
      loom_ir_remap_find_const_value_map_slot(remap, source_value);
  if (!entry) {
    return false;
  }
  if (out_target_value) {
    *out_target_value = entry->target_value;
  }
  return true;
}

iree_status_t loom_ir_remap_resolve_value(const loom_ir_remap_t* remap,
                                          loom_value_id_t source_value,
                                          loom_value_id_t* out_target_value) {
  *out_target_value = LOOM_VALUE_ID_INVALID;
  if (source_value >= remap->source_module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source value %%%u out of range (source module has %" PRIhsz " values)",
        (unsigned)source_value, remap->source_module->values.count);
  }
  if (source_value >= remap->source_value_snapshot_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source value %%%u was defined after this remap's %" PRIhsz
        "-value source snapshot",
        (unsigned)source_value, remap->source_value_snapshot_count);
  }

  if (loom_ir_remap_try_lookup_value(remap, source_value, out_target_value)) {
    return iree_ok_status();
  }
  if (remap->allow_unmapped_values &&
      remap->source_module == remap->target_module) {
    *out_target_value = source_value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "source value %%%u has no target remap",
                          (unsigned)source_value);
}

static iree_status_t loom_ir_remap_ensure_block_map_capacity(
    loom_ir_remap_t* remap, iree_host_size_t required_count) {
  if (required_count <= remap->block_map_capacity) return iree_ok_status();
  iree_host_size_t source_capacity = remap->block_map_capacity;
  const loom_block_t** sources = remap->block_map_sources;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      remap->arena, remap->block_map_count, required_count,
      sizeof(loom_block_t*), &source_capacity, (void**)&sources));
  iree_host_size_t target_capacity = remap->block_map_capacity;
  loom_block_t** targets = remap->block_map_targets;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      remap->arena, remap->block_map_count, required_count,
      sizeof(loom_block_t*), &target_capacity, (void**)&targets));
  remap->block_map_sources = sources;
  remap->block_map_targets = targets;
  remap->block_map_capacity =
      source_capacity < target_capacity ? source_capacity : target_capacity;
  return iree_ok_status();
}

iree_status_t loom_ir_remap_map_block(loom_ir_remap_t* remap,
                                      const loom_block_t* source_block,
                                      loom_block_t* target_block) {
  for (iree_host_size_t i = 0; i < remap->block_map_count; ++i) {
    if (remap->block_map_sources[i] != source_block) continue;
    remap->block_map_targets[i] = target_block;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_ir_remap_ensure_block_map_capacity(
      remap, remap->block_map_count + 1));
  remap->block_map_sources[remap->block_map_count] = source_block;
  remap->block_map_targets[remap->block_map_count] = target_block;
  ++remap->block_map_count;
  return iree_ok_status();
}

bool loom_ir_remap_try_lookup_block(const loom_ir_remap_t* remap,
                                    const loom_block_t* source_block,
                                    loom_block_t** out_target_block) {
  if (out_target_block) *out_target_block = NULL;
  if (!remap || !source_block) return false;
  for (iree_host_size_t i = 0; i < remap->block_map_count; ++i) {
    if (remap->block_map_sources[i] != source_block) continue;
    if (out_target_block) *out_target_block = remap->block_map_targets[i];
    return true;
  }
  return false;
}

iree_status_t loom_ir_remap_resolve_block(const loom_ir_remap_t* remap,
                                          const loom_block_t* source_block,
                                          loom_block_t** out_target_block) {
  *out_target_block = NULL;
  if (!source_block) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source successor block is NULL");
  }
  if (loom_ir_remap_try_lookup_block(remap, source_block, out_target_block)) {
    return iree_ok_status();
  }
  if (remap->source_module == remap->target_module) {
    *out_target_block = (loom_block_t*)source_block;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "source successor block has no target remap");
}

iree_status_t loom_ir_remap_string_id(loom_ir_remap_t* remap,
                                      loom_string_id_t source_string_id,
                                      bool allow_invalid,
                                      loom_string_id_t* out_string_id) {
  *out_string_id = LOOM_STRING_ID_INVALID;
  if (source_string_id == LOOM_STRING_ID_INVALID) {
    if (allow_invalid) return iree_ok_status();
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source string id is invalid");
  }
  if (source_string_id >= remap->source_module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source string id %u out of range (source module has %" PRIhsz
        " strings)",
        (unsigned)source_string_id, remap->source_module->strings.count);
  }
  if (remap->source_module == remap->target_module) {
    *out_string_id = source_string_id;
    return iree_ok_status();
  }
  return loom_module_intern_string(
      remap->target_module,
      remap->source_module->strings.entries[source_string_id], out_string_id);
}

static iree_status_t loom_ir_remap_source_id(
    loom_ir_remap_t* remap, loom_source_id_t source_id,
    loom_source_id_t* out_target_source_id) {
  *out_target_source_id = LOOM_SOURCE_ID_INVALID;
  if (source_id == LOOM_SOURCE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source id is invalid");
  }
  if (source_id >= remap->source_module->sources.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source id %u out of range (source module has %" PRIhsz " sources)",
        (unsigned)source_id, remap->source_module->sources.count);
  }
  if (remap->source_module == remap->target_module) {
    *out_target_source_id = source_id;
    return iree_ok_status();
  }
  return loom_module_register_source(
      remap->target_module, remap->source_module->sources.entries[source_id],
      out_target_source_id);
}

static iree_status_t loom_ir_remap_location_entry(
    loom_ir_remap_t* remap, loom_location_entry_t source_entry,
    iree_host_size_t depth, loom_location_entry_t* out_target_entry) {
  if (depth >= LOOM_IR_REMAP_MAX_LOCATION_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location nesting exceeds max depth %u",
                            (unsigned)LOOM_IR_REMAP_MAX_LOCATION_DEPTH);
  }
  loom_location_entry_t target_entry = source_entry;
  switch ((loom_location_kind_t)source_entry.kind) {
    case LOOM_LOCATION_NONE:
      *out_target_entry = target_entry;
      return iree_ok_status();

    case LOOM_LOCATION_FILE: {
      IREE_RETURN_IF_ERROR(loom_ir_remap_source_id(
          remap, source_entry.file.source_id, &target_entry.file.source_id));
      if (source_entry.file.field_span_count > 0) {
        if (!source_entry.file.field_spans) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "file location has field spans but a NULL span payload");
        }
        loom_location_field_span_t* target_spans = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &remap->target_module->arena, source_entry.file.field_span_count,
            sizeof(loom_location_field_span_t), (void**)&target_spans));
        memcpy(target_spans, source_entry.file.field_spans,
               (iree_host_size_t)source_entry.file.field_span_count *
                   sizeof(loom_location_field_span_t));
        target_entry.file.field_spans = target_spans;
      } else {
        target_entry.file.field_spans = NULL;
      }
      *out_target_entry = target_entry;
      return iree_ok_status();
    }

    case LOOM_LOCATION_FUSED:
      if (source_entry.fused.count > 0) {
        if (!source_entry.fused.children) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "fused location has children but a NULL child payload");
        }
        loom_location_id_t* target_children = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &remap->target_module->arena, source_entry.fused.count,
            sizeof(loom_location_id_t), (void**)&target_children));
        for (uint32_t i = 0; i < source_entry.fused.count; ++i) {
          loom_location_entry_t target_child_entry = {0};
          loom_location_id_t source_child_id = source_entry.fused.children[i];
          if (source_child_id >= remap->source_module->locations.count) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "fused location child id %u out of range (source module has "
                "%" PRIhsz " locations)",
                source_child_id, remap->source_module->locations.count);
          }
          IREE_RETURN_IF_ERROR(loom_ir_remap_location_entry(
              remap, remap->source_module->locations.entries[source_child_id],
              depth + 1, &target_child_entry));
          IREE_RETURN_IF_ERROR(loom_module_add_location(
              remap->target_module, target_child_entry, &target_children[i]));
        }
        target_entry.fused.children = target_children;
      } else {
        target_entry.fused.children = NULL;
      }
      *out_target_entry = target_entry;
      return iree_ok_status();

    case LOOM_LOCATION_OPAQUE: {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_source_id(remap, source_entry.opaque.source_id,
                                  &target_entry.opaque.source_id));
      if (source_entry.opaque.data_length > 0) {
        if (!source_entry.opaque.data) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "opaque location has data but a NULL data payload");
        }
        uint8_t* target_data = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            &remap->target_module->arena, source_entry.opaque.data_length,
            (void**)&target_data));
        memcpy(target_data, source_entry.opaque.data,
               source_entry.opaque.data_length);
        target_entry.opaque.data = target_data;
      } else {
        target_entry.opaque.data = NULL;
      }
      *out_target_entry = target_entry;
      return iree_ok_status();
    }

    case LOOM_LOCATION_TAGGED: {
      if (source_entry.tagged.tag == LOOM_LOCATION_TAG_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "tagged location has invalid tag 0");
      }
      if (source_entry.tagged.child != LOOM_LOCATION_UNKNOWN) {
        if (source_entry.tagged.child >=
            remap->source_module->locations.count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "tagged location child id %u out of range (source module has "
              "%" PRIhsz " locations)",
              source_entry.tagged.child, remap->source_module->locations.count);
        }
        loom_location_entry_t target_child_entry = {0};
        IREE_RETURN_IF_ERROR(loom_ir_remap_location_entry(
            remap,
            remap->source_module->locations.entries[source_entry.tagged.child],
            depth + 1, &target_child_entry));
        IREE_RETURN_IF_ERROR(
            loom_module_add_location(remap->target_module, target_child_entry,
                                     &target_entry.tagged.child));
      }
      if (source_entry.tagged.data_length > 0) {
        if (!source_entry.tagged.data) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "tagged location has data but a NULL data payload");
        }
        uint8_t* target_data = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            &remap->target_module->arena, source_entry.tagged.data_length,
            (void**)&target_data));
        memcpy(target_data, source_entry.tagged.data,
               source_entry.tagged.data_length);
        target_entry.tagged.data = target_data;
      } else {
        target_entry.tagged.data = NULL;
      }
      *out_target_entry = target_entry;
      return iree_ok_status();
    }

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown location kind %u",
                              (unsigned)source_entry.kind);
  }
}

iree_status_t loom_ir_remap_location_id(
    loom_ir_remap_t* remap, loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id) {
  *out_target_location_id = LOOM_LOCATION_UNKNOWN;
  if (source_location_id == LOOM_LOCATION_UNKNOWN) return iree_ok_status();
  if (source_location_id >= remap->source_module->locations.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source location id %u out of range (source module has %" PRIhsz
        " locations)",
        source_location_id, remap->source_module->locations.count);
  }
  if (remap->source_module == remap->target_module) {
    *out_target_location_id = source_location_id;
    return iree_ok_status();
  }
  loom_location_entry_t target_entry = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_entry(
      remap, remap->source_module->locations.entries[source_location_id],
      /*depth=*/0, &target_entry));
  return loom_module_add_location(remap->target_module, target_entry,
                                  out_target_location_id);
}

static iree_status_t loom_ir_remap_symbol_ref(
    loom_ir_remap_t* remap, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  if (!loom_symbol_ref_is_valid(source_ref)) {
    *out_target_ref = source_ref;
    return iree_ok_status();
  }
  if (source_ref.module_id != 0 ||
      source_ref.symbol_id >= remap->source_module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=%u, symbol=%u} is out of range",
        (unsigned)source_ref.module_id, (unsigned)source_ref.symbol_id);
  }
  if (remap->source_module == remap->target_module) {
    *out_target_ref = source_ref;
    return iree_ok_status();
  }
  if (!remap->remap_symbol.fn) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cross-module symbol reference remapping requires a symbol policy");
  }
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(remap->remap_symbol.fn(
      remap->remap_symbol.user_data, remap->source_module, remap->target_module,
      source_ref, &target_ref));
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= remap->target_module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target symbol ref {module=%u, symbol=%u} is out of range",
        (unsigned)target_ref.module_id, (unsigned)target_ref.symbol_id);
  }
  *out_target_ref = target_ref;
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_type_sequence(
    loom_ir_remap_t* remap, const loom_type_t* source_types,
    uint16_t type_count, loom_type_t** out_target_types) {
  *out_target_types = NULL;
  if (type_count == 0) return iree_ok_status();
  if (!source_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type sequence has %u entries but a NULL payload",
                            (unsigned)type_count);
  }
  loom_type_t* target_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      remap->arena, type_count, sizeof(loom_type_t), (void**)&target_types));
  for (uint16_t i = 0; i < type_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(remap, source_types[i], &target_types[i]));
  }
  *out_target_types = target_types;
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_type_id(loom_ir_remap_t* remap,
                                           loom_type_id_t source_type_id,
                                           loom_type_id_t* out_target_type_id) {
  *out_target_type_id = LOOM_TYPE_ID_INVALID;
  if (source_type_id == LOOM_TYPE_ID_INVALID ||
      source_type_id >= remap->source_module->types.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source type id %u out of range (source module has %" PRIhsz " types)",
        (unsigned)source_type_id, remap->source_module->types.count);
  }
  loom_type_t target_type = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_type(
      remap, remap->source_module->types.entries[source_type_id],
      &target_type));
  return loom_module_intern_type_id(remap->target_module, target_type,
                                    out_target_type_id);
}

iree_status_t loom_ir_remap_type(loom_ir_remap_t* remap,
                                 loom_type_t source_type,
                                 loom_type_t* out_target_type) {
  *out_target_type = source_type;

  loom_type_kind_t kind = loom_type_kind(source_type);
  if (!loom_type_kind_is_valid(kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown type kind %u", (unsigned)kind);
  }

  if (kind == LOOM_TYPE_FUNCTION) {
    const loom_func_type_data_t* source_data = loom_type_func_data(source_type);
    if (!source_data) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function type has a NULL argument/result payload");
    }
    uint16_t type_count =
        (uint16_t)(source_data->arg_count + source_data->result_count);
    loom_type_t* target_types = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_remap_type_sequence(
        remap, source_data->types, type_count, &target_types));
    const loom_type_t* target_arg_types =
        source_data->arg_count > 0 ? target_types : NULL;
    const loom_type_t* target_result_types =
        source_data->result_count > 0 ? target_types + source_data->arg_count
                                      : NULL;
    return loom_module_intern_function_type(
        remap->target_module, target_arg_types, source_data->arg_count,
        target_result_types, source_data->result_count, out_target_type);
  }

  if (kind == LOOM_TYPE_DIALECT) {
    loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_string_id(remap, loom_type_dialect_name_id(source_type),
                                /*allow_invalid=*/false, &target_name_id));
    uint16_t param_count = loom_type_dialect_param_count(source_type);
    loom_type_t* target_params = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_remap_type_sequence(
        remap, loom_type_dialect_params(source_type), param_count,
        &target_params));
    loom_type_t target_type =
        param_count == 0
            ? loom_type_dialect_opaque(target_name_id)
            : loom_type_dialect(target_name_id, param_count, target_params);
    return loom_module_intern_type(remap->target_module, target_type,
                                   out_target_type);
  }

  loom_type_t target_type = source_type;
  bool needs_interned_payload = false;
  loom_overflow_dim_t target_overflow_dims[LOOM_TYPE_MAX_RANK] = {0};
  if (loom_type_is_shaped(source_type) || loom_type_is_pool(source_type)) {
    uint8_t rank = loom_type_rank(source_type);
    if (loom_type_has_inline_dims(source_type)) {
      for (uint8_t i = 0; i < rank; ++i) {
        uint64_t dim = target_type.dims[i];
        if (!loom_dim_is_dynamic(dim)) continue;
        loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
            remap, loom_dim_value_id(dim), &target_value));
        target_type.dims[i] = loom_dim_pack_dynamic(target_value);
      }
    } else if (rank > 0) {
      const loom_overflow_dim_t* source_dims =
          (const loom_overflow_dim_t*)(uintptr_t)source_type.dims[0];
      if (!source_dims) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rank-%u type has a NULL overflow dim payload",
                                (unsigned)rank);
      }
      for (uint8_t i = 0; i < rank; ++i) {
        target_overflow_dims[i] = source_dims[i];
        if (!loom_dim_is_dynamic(target_overflow_dims[i])) continue;
        loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
            remap, loom_dim_value_id(target_overflow_dims[i]), &target_value));
        target_overflow_dims[i] = loom_dim_pack_dynamic(target_value);
      }
      target_type.dims[0] = (uint64_t)(uintptr_t)target_overflow_dims;
      target_type.dims[1] = 0;
      needs_interned_payload = true;
    }
  }

  if (loom_type_has_ssa_encoding(source_type)) {
    loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        remap, loom_type_encoding_value_id(source_type), &target_value));
    if (target_value > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cannot store target value %%%u in a 16-bit SSA encoding reference",
          (unsigned)target_value);
    }
    target_type.encoding_id = (uint16_t)target_value;
  } else if (loom_type_has_static_encoding(source_type)) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_encoding_id(
        remap, source_type.encoding_id, &target_type.encoding_id));
  }

  if (needs_interned_payload) {
    IREE_RETURN_IF_ERROR(loom_module_intern_type(remap->target_module,
                                                 target_type, &target_type));
  }
  *out_target_type = target_type;
  return iree_ok_status();
}

iree_status_t loom_ir_remap_value_types(loom_ir_remap_t* remap,
                                        const loom_value_id_t* source_values,
                                        iree_host_size_t value_count,
                                        loom_type_t** out_target_types) {
  *out_target_types = NULL;
  if (value_count == 0) return iree_ok_status();
  if (!source_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty source value array requires a payload");
  }

  loom_type_t* target_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      remap->arena, value_count, sizeof(loom_type_t), (void**)&target_types));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_value_id_t source_value = source_values[i];
    if (source_value >= remap->source_module->values.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source value %%%u out of range (source module has %" PRIhsz
          " values)",
          (unsigned)source_value, remap->source_module->values.count);
    }
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap, loom_module_value_type(remap->source_module, source_value),
        &target_types[i]));
  }
  *out_target_types = target_types;
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_predicate_list_into(
    loom_ir_remap_t* remap, const loom_predicate_t* source_predicates,
    iree_host_size_t predicate_count, iree_arena_allocator_t* payload_arena,
    loom_predicate_t** out_target_predicates) {
  *out_target_predicates = NULL;
  if (predicate_count == 0) return iree_ok_status();
  if (!source_predicates) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty predicate list has a NULL source pointer");
  }
  if (predicate_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "predicate list has %" PRIhsz " entries, max %u",
                            predicate_count, (unsigned)UINT16_MAX);
  }

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(payload_arena, predicate_count,
                                                 sizeof(loom_predicate_t),
                                                 (void**)&target_predicates));
  for (iree_host_size_t i = 0; i < predicate_count; ++i) {
    target_predicates[i] = source_predicates[i];
    if (target_predicates[i].arg_count >
        IREE_ARRAYSIZE(target_predicates[i].args)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "predicate %" PRIhsz " has %u args, max %" PRIhsz,
                              i, (unsigned)target_predicates[i].arg_count,
                              IREE_ARRAYSIZE(target_predicates[i].args));
    }
    for (uint8_t arg_index = 0; arg_index < target_predicates[i].arg_count;
         ++arg_index) {
      switch (
          (loom_predicate_arg_tag_t)target_predicates[i].arg_tags[arg_index]) {
        case LOOM_PRED_ARG_NONE:
        case LOOM_PRED_ARG_CONST:
          continue;
        case LOOM_PRED_ARG_VALUE:
          break;
        default:
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "predicate %" PRIhsz " arg %u has unknown tag %u", i,
              (unsigned)arg_index,
              (unsigned)target_predicates[i].arg_tags[arg_index]);
      }
      loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
          remap, (loom_value_id_t)target_predicates[i].args[arg_index],
          &target_value));
      target_predicates[i].args[arg_index] = (int64_t)target_value;
    }
  }
  *out_target_predicates = target_predicates;
  return iree_ok_status();
}

iree_status_t loom_ir_remap_predicate_list(
    loom_ir_remap_t* remap, const loom_predicate_t* source_predicates,
    iree_host_size_t predicate_count,
    loom_predicate_t** out_target_predicates) {
  iree_arena_allocator_t* payload_arena =
      loom_ir_remap_is_initialized(remap) ? &remap->target_module->arena : NULL;
  return loom_ir_remap_predicate_list_into(remap, source_predicates,
                                           predicate_count, payload_arena,
                                           out_target_predicates);
}

static iree_status_t loom_ir_remap_attribute_impl(
    loom_ir_remap_t* remap, loom_attribute_t source_attr,
    iree_host_size_t dict_depth, iree_arena_allocator_t* payload_arena,
    loom_attribute_t* out_target_attr) {
  *out_target_attr = source_attr;

  switch ((loom_attr_kind_t)source_attr.kind) {
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
      return iree_ok_status();

    case LOOM_ATTR_STRING:
      return loom_ir_remap_string_id(remap, source_attr.string_id,
                                     /*allow_invalid=*/false,
                                     &out_target_attr->string_id);

    case LOOM_ATTR_SYMBOL:
      return loom_ir_remap_symbol_ref(remap, source_attr.symbol,
                                      &out_target_attr->symbol);

    case LOOM_ATTR_TYPE:
      return loom_ir_remap_type_id(remap, source_attr.type_id,
                                   &out_target_attr->type_id);

    case LOOM_ATTR_ENCODING: {
      if (source_attr.encoding_id > UINT16_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "source encoding id %" PRIu32
                                " exceeds uint16_t range",
                                source_attr.encoding_id);
      }
      uint16_t target_encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_ir_remap_encoding_id(
          remap, (uint16_t)source_attr.encoding_id, &target_encoding_id));
      out_target_attr->encoding_id = target_encoding_id;
      return iree_ok_status();
    }

    case LOOM_ATTR_I64_ARRAY: {
      if (source_attr.count == 0) {
        out_target_attr->i64_array = NULL;
        return iree_ok_status();
      }
      if (!source_attr.i64_array) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty i64 array attribute has a NULL payload");
      }
      int64_t* target_values = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(payload_arena, source_attr.count,
                                    sizeof(int64_t), (void**)&target_values));
      memcpy(target_values, source_attr.i64_array,
             (iree_host_size_t)source_attr.count * sizeof(int64_t));
      out_target_attr->i64_array = target_values;
      return iree_ok_status();
    }

    case LOOM_ATTR_PREDICATE_LIST: {
      loom_predicate_t* target_predicates = NULL;
      IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list_into(
          remap, source_attr.predicate_list, source_attr.count, payload_arena,
          &target_predicates));
      out_target_attr->predicate_list = target_predicates;
      return iree_ok_status();
    }

    case LOOM_ATTR_DICT: {
      if (dict_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute nesting exceeds max depth %u",
                                (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
      }
      if (source_attr.count == 0) {
        *out_target_attr = loom_make_canonical_attr_dict(NULL, 0);
        return iree_ok_status();
      }
      if (!source_attr.dict_entries) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute has a NULL entry pointer");
      }
      loom_named_attr_t* target_entries = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          remap->arena, source_attr.count, sizeof(loom_named_attr_t),
          (void**)&target_entries));
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        target_entries[i].reserved = 0;
        IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
            remap, source_attr.dict_entries[i].name_id,
            /*allow_invalid=*/false, &target_entries[i].name_id));
        IREE_RETURN_IF_ERROR(loom_ir_remap_attribute_impl(
            remap, source_attr.dict_entries[i].value, dict_depth + 1,
            remap->arena, &target_entries[i].value));
      }
      return loom_module_make_canonical_attr_dict(
          remap->target_module,
          loom_make_named_attr_slice(target_entries, source_attr.count),
          out_target_attr);
    }

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown attribute kind %u",
                              (unsigned)source_attr.kind);
  }
}

iree_status_t loom_ir_remap_attribute(loom_ir_remap_t* remap,
                                      loom_attribute_t source_attr,
                                      loom_attribute_t* out_target_attr) {
  iree_arena_allocator_t* payload_arena =
      loom_ir_remap_is_initialized(remap) ? &remap->target_module->arena : NULL;
  return loom_ir_remap_attribute_impl(remap, source_attr,
                                      /*dict_depth=*/0, payload_arena,
                                      out_target_attr);
}

iree_status_t loom_ir_remap_encoding_id(loom_ir_remap_t* remap,
                                        uint16_t source_encoding_id,
                                        uint16_t* out_target_encoding_id) {
  *out_target_encoding_id = 0;
  if (remap->encoding_depth >= LOOM_IR_REMAP_MAX_ENCODING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding nesting exceeds max depth %u",
                            (unsigned)LOOM_IR_REMAP_MAX_ENCODING_DEPTH);
  }
  ++remap->encoding_depth;
  iree_status_t status = iree_ok_status();
  const loom_encoding_t* source_encoding =
      loom_module_encoding(remap->source_module, source_encoding_id);
  if (!source_encoding) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source encoding id %u out of range (source module has %" PRIhsz
        " encodings)",
        (unsigned)source_encoding_id, remap->source_module->encodings.count);
  }

  if (iree_status_is_ok(status) &&
      remap->source_module == remap->target_module) {
    *out_target_encoding_id = source_encoding_id;
  }

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t target_alias_id = LOOM_STRING_ID_INVALID;
  loom_named_attr_t* target_attrs = NULL;
  if (iree_status_is_ok(status) &&
      remap->source_module != remap->target_module) {
    status = loom_ir_remap_string_id(remap, source_encoding->name_id,
                                     /*allow_invalid=*/false, &target_name_id);
  }
  if (iree_status_is_ok(status) &&
      remap->source_module != remap->target_module) {
    status = loom_ir_remap_string_id(remap, source_encoding->alias_id,
                                     /*allow_invalid=*/true, &target_alias_id);
  }
  if (iree_status_is_ok(status) &&
      remap->source_module != remap->target_module &&
      source_encoding->attribute_count > 0) {
    if (!source_encoding->attributes) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "non-empty source encoding attribute list has a NULL payload");
    }
  }
  if (iree_status_is_ok(status) &&
      remap->source_module != remap->target_module &&
      source_encoding->attribute_count > 0) {
    status = iree_arena_allocate_array(
        remap->arena, source_encoding->attribute_count,
        sizeof(loom_named_attr_t), (void**)&target_attrs);
  }
  for (uint8_t i = 0; iree_status_is_ok(status) &&
                      remap->source_module != remap->target_module &&
                      i < source_encoding->attribute_count;
       ++i) {
    target_attrs[i].reserved = 0;
    status = loom_ir_remap_string_id(
        remap, source_encoding->attributes[i].name_id,
        /*allow_invalid=*/false, &target_attrs[i].name_id);
    if (!iree_status_is_ok(status)) continue;
    status = loom_ir_remap_attribute_impl(
        remap, source_encoding->attributes[i].value, /*dict_depth=*/0,
        remap->arena, &target_attrs[i].value);
  }

  if (iree_status_is_ok(status) &&
      remap->source_module != remap->target_module) {
    loom_encoding_t target_encoding = {
        .name_id = target_name_id,
        .alias_id = target_alias_id,
        .attribute_count = source_encoding->attribute_count,
        .attributes = target_attrs,
    };
    status = loom_module_add_encoding(remap->target_module, &target_encoding,
                                      out_target_encoding_id);
  }
  --remap->encoding_depth;
  return status;
}
