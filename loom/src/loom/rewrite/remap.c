// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/remap.h"

#include <string.h>

#include "loom/ir/module.h"

// Location stacks use a bounded recursive representation.
#define LOOM_IR_REMAP_MAX_LOCATION_DEPTH 16

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
      .remap_same_module_symbols =
          options ? options->remap_same_module_symbols : false,
  };
  if (options != NULL) {
    remap.clone_observer = options->clone_observer;
    remap.op_projection.entries = options->op_projection.entries;
    remap.op_projection.count = options->op_projection.count;
  }

  if (remap.op_projection.count != 0 && remap.op_projection.entries == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operation projection count is non-zero but entries are NULL");
  }
  for (iree_host_size_t i = 0; i < remap.op_projection.count; ++i) {
    if (remap.op_projection.entries[i].source_op == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "operation projection entry %zu has no source",
                              i);
    }
    remap.op_projection.entries[i].target_op = NULL;
  }

  const loom_ir_remap_value_map_kind_t value_map_kind =
      options ? options->value_map_kind : LOOM_IR_REMAP_VALUE_MAP_SPARSE;
  switch (value_map_kind) {
    case LOOM_IR_REMAP_VALUE_MAP_SPARSE:
      break;
    case LOOM_IR_REMAP_VALUE_MAP_SOURCE_INDEXED:
      if (remap.source_value_snapshot_count > 0) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate_array(arena, remap.source_value_snapshot_count,
                                      sizeof(*remap.target_values_by_source),
                                      (void**)&remap.target_values_by_source));
        for (iree_host_size_t i = 0; i < remap.source_value_snapshot_count;
             ++i) {
          remap.target_values_by_source[i] = LOOM_VALUE_ID_INVALID;
        }
      }
      break;
    default:
      IREE_ASSERT_UNREACHABLE("unknown remap value map kind");
      IREE_BUILTIN_UNREACHABLE();
  }

  if (source_module != target_module && source_module->sources.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, source_module->sources.count, sizeof(*remap.target_sources),
        (void**)&remap.target_sources));
    for (iree_host_size_t i = 0; i < source_module->sources.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_module_register_source(
          target_module, source_module->sources.entries[i],
          &remap.target_sources[i]));
    }
  }
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
  if (remap->target_values_by_source != NULL) {
    loom_value_id_t* mapped_value =
        &remap->target_values_by_source[source_value];
    if (*mapped_value == LOOM_VALUE_ID_INVALID) {
      ++remap->mapped_value_count;
    }
    *mapped_value = target_value;
    return iree_ok_status();
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
  if (remap->target_values_by_source != NULL) {
    const loom_value_id_t target_value =
        remap->target_values_by_source[source_value];
    if (target_value == LOOM_VALUE_ID_INVALID) {
      return false;
    }
    if (out_target_value) {
      *out_target_value = target_value;
    }
    return true;
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
  if (required_count <= remap->block_map_capacity) {
    return iree_ok_status();
  }
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
    if (remap->block_map_sources[i] != source_block) {
      continue;
    }
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
  if (out_target_block) {
    *out_target_block = NULL;
  }
  if (!remap || !source_block) {
    return false;
  }
  for (iree_host_size_t i = 0; i < remap->block_map_count; ++i) {
    if (remap->block_map_sources[i] != source_block) {
      continue;
    }
    if (out_target_block) {
      *out_target_block = remap->block_map_targets[i];
    }
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
    if (allow_invalid) {
      return iree_ok_status();
    }
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
      loom_string_table_get(&remap->source_module->strings, source_string_id),
      out_string_id);
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
  *out_target_source_id = remap->target_sources[source_id];
  return iree_ok_status();
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
              remap,
              *loom_location_table_const_entry(&remap->source_module->locations,
                                               source_child_id),
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
            *loom_location_table_const_entry(&remap->source_module->locations,
                                             source_entry.tagged.child),
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
  if (source_location_id == LOOM_LOCATION_UNKNOWN) {
    return iree_ok_status();
  }
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
      remap,
      *loom_location_table_const_entry(&remap->source_module->locations,
                                       source_location_id),
      /*depth=*/0, &target_entry));
  return loom_module_add_location(remap->target_module, target_entry,
                                  out_target_location_id);
}

iree_status_t loom_ir_remap_symbol_ref(loom_ir_remap_t* remap,
                                       loom_symbol_ref_t source_ref,
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
  if (remap->source_module == remap->target_module &&
      !remap->remap_same_module_symbols) {
    *out_target_ref = source_ref;
    return iree_ok_status();
  }
  if (!remap->remap_symbol.fn) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "symbol reference remapping requires a symbol policy");
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

iree_status_t loom_ir_remap_value_types(loom_ir_remap_t* remap,
                                        const loom_value_id_t* source_values,
                                        iree_host_size_t value_count,
                                        loom_type_t** out_target_types) {
  *out_target_types = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
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

iree_status_t loom_ir_remap_assign_value_types(
    loom_module_t* module, const loom_value_id_t* source_values,
    const loom_value_id_t* target_values, iree_host_size_t value_count) {
  bool has_dependencies = false;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(module, source_values[i],
                                        &dependencies);
    if (loom_type_dependencies_next(&dependencies) != LOOM_VALUE_ID_INVALID) {
      has_dependencies = true;
      break;
    }
  }

  iree_arena_allocator_t scratch;
  iree_arena_initialize(module->arena.block_pool, &scratch);
  loom_ir_remap_t remap;
  iree_status_t status = iree_ok_status();
  if (has_dependencies) {
    const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
    status =
        loom_ir_remap_initialize(module, module, &scratch, &options, &remap);
    if (iree_status_is_ok(status)) {
      status = loom_ir_remap_map_values(&remap, source_values, target_values,
                                        value_count);
    }
  }
  for (iree_host_size_t i = 0; i < value_count && iree_status_is_ok(status);
       ++i) {
    loom_type_t type = loom_module_value_type(module, source_values[i]);
    if (has_dependencies) {
      status = loom_ir_remap_type(&remap, type, &type);
    }
    if (iree_status_is_ok(status) &&
        !loom_type_equal(type,
                         loom_module_value_type(module, target_values[i]))) {
      status = loom_module_set_value_type(module, target_values[i], type);
    }
  }
  iree_arena_deinitialize(&scratch);
  return status;
}
