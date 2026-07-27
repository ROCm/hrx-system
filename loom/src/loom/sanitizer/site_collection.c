// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/site_collection.h"

#include <string.h>

#include "loom/ops/sanitizer/ops.h"
#include "loom/util/walk.h"

#define LOOM_SANITIZER_SITE_LOCATION_MAX_DEPTH 64

typedef struct loom_sanitizer_site_location_result_t {
  // Tagged location entry that carried payload, or LOOM_LOCATION_UNKNOWN.
  loom_location_id_t payload_location;

  // Child location under the sanitizer tag, or LOOM_LOCATION_UNKNOWN.
  loom_location_id_t source_location;

  // Decoded payload when payload_location is not LOOM_LOCATION_UNKNOWN.
  loom_sanitizer_site_payload_t payload;

  // Number of sanitizer payloads discovered while traversing the location.
  iree_host_size_t payload_count;
} loom_sanitizer_site_location_result_t;

typedef struct loom_sanitizer_site_count_state_t {
  // Number of sanitizer site ops observed so far.
  iree_host_size_t count;
} loom_sanitizer_site_count_state_t;

typedef struct loom_sanitizer_site_collect_state_t {
  // Module that owns op locations.
  const loom_module_t* module;

  // Row array allocated by the caller.
  loom_sanitizer_site_row_t* rows;

  // Next dense site ID and row index to assign.
  iree_host_size_t next_site_id;
} loom_sanitizer_site_collect_state_t;

static bool loom_sanitizer_site_op_isa(const loom_op_t* op) {
  return loom_sanitizer_assert_access_isa(op) ||
         loom_sanitizer_assert_accesses_isa(op) ||
         loom_sanitizer_assert_value_isa(op) ||
         loom_sanitizer_assert_op_isa(op) ||
         loom_sanitizer_assert_layout_isa(op) ||
         loom_sanitizer_race_access_isa(op) || loom_sanitizer_race_sync_isa(op);
}

static iree_status_t loom_sanitizer_site_location_validate_child(
    const loom_module_t* module, loom_location_id_t location_id) {
  if (location_id == LOOM_LOCATION_UNKNOWN) return iree_ok_status();
  if ((iree_host_size_t)location_id >= module->locations.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer site child location id %u out of range "
                            "(%" PRIhsz " locations)",
                            location_id, module->locations.count);
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_site_location_find_payload(
    const loom_module_t* module, loom_location_id_t location_id, uint8_t depth,
    loom_sanitizer_site_location_result_t* result) {
  if (location_id == LOOM_LOCATION_UNKNOWN) return iree_ok_status();
  if (depth >= LOOM_SANITIZER_SITE_LOCATION_MAX_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sanitizer site location tree exceeded maximum depth %u",
        (unsigned)LOOM_SANITIZER_SITE_LOCATION_MAX_DEPTH);
  }
  if ((iree_host_size_t)location_id >= module->locations.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sanitizer site location id %u out of range (%" PRIhsz " locations)",
        location_id, module->locations.count);
  }

  const loom_location_entry_t* entry = &module->locations.entries[location_id];
  switch (loom_location_get_kind(*entry)) {
    case LOOM_LOCATION_NONE:
    case LOOM_LOCATION_FILE:
    case LOOM_LOCATION_OPAQUE:
      return iree_ok_status();

    case LOOM_LOCATION_FUSED: {
      for (uint32_t i = 0; i < entry->fused.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_sanitizer_site_location_find_payload(
            module, entry->fused.children[i], (uint8_t)(depth + 1), result));
      }
      return iree_ok_status();
    }

    case LOOM_LOCATION_TAGGED: {
      if (entry->tagged.tag == LOOM_LOCATION_TAG_SANITIZER_SITE) {
        if (result->payload_count != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "sanitizer site location tree contains multiple site payloads");
        }
        IREE_RETURN_IF_ERROR(loom_sanitizer_site_payload_decode(
            iree_make_const_byte_span(entry->tagged.data,
                                      entry->tagged.data_length),
            &result->payload));
        IREE_RETURN_IF_ERROR(loom_sanitizer_site_location_validate_child(
            module, entry->tagged.child));
        result->payload_location = location_id;
        result->source_location = entry->tagged.child;
        result->payload_count = 1;
        return iree_ok_status();
      }
      return loom_sanitizer_site_location_find_payload(
          module, entry->tagged.child, (uint8_t)(depth + 1), result);
    }

    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "sanitizer site location id %u has unsupported kind %u", location_id,
          (unsigned)entry->kind);
  }
}

static iree_status_t loom_sanitizer_site_count_visitor(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_sanitizer_site_count_state_t* state =
      (loom_sanitizer_site_count_state_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_sanitizer_site_op_isa(op)) return iree_ok_status();
  if (state->count == LOOM_SANITIZER_SITE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sanitizer site collection exceeded max site id %u",
                            LOOM_SANITIZER_SITE_ID_INVALID - 1);
  }
  ++state->count;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_site_collect_visitor(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_sanitizer_site_collect_state_t* state =
      (loom_sanitizer_site_collect_state_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_sanitizer_site_op_isa(op)) return iree_ok_status();

  loom_sanitizer_site_location_result_t location_result = {
      .payload_location = LOOM_LOCATION_UNKNOWN,
      .source_location = LOOM_LOCATION_UNKNOWN,
  };
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_location_find_payload(
      state->module, op->location, 0, &location_result));

  loom_sanitizer_site_row_t* row = &state->rows[state->next_site_id];
  memset(row, 0, sizeof(*row));
  row->site_id = (loom_sanitizer_site_id_t)state->next_site_id;
  row->op = op;
  row->op_kind = op->kind;
  row->location = op->location;
  row->payload_location = location_result.payload_location;
  row->source_location = location_result.source_location;
  if (location_result.payload_count != 0) {
    row->flags |= LOOM_SANITIZER_SITE_ROW_HAS_PAYLOAD;
    row->payload = location_result.payload;
  }

  ++state->next_site_id;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_site_collection_allocate_rows(
    iree_host_size_t row_count, iree_arena_allocator_t* arena,
    loom_sanitizer_site_collection_t* collection) {
  collection->rows = NULL;
  collection->row_count = row_count;
  if (row_count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, row_count,
                                   sizeof(loom_sanitizer_site_row_t),
                                   (void**)&collection->rows);
}

iree_status_t loom_sanitizer_site_collection_build_region(
    const loom_module_t* module, loom_region_t* region,
    iree_arena_allocator_t* arena,
    loom_sanitizer_site_collection_t* out_collection) {
  IREE_ASSERT_ARGUMENT(out_collection);
  *out_collection = (loom_sanitizer_site_collection_t){0};

  loom_sanitizer_site_count_state_t count_state = {0};
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, region, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_sanitizer_site_count_visitor, &count_state},
      arena, &walk_result));

  loom_sanitizer_site_collection_t collection = {0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_collection_allocate_rows(
      count_state.count, arena, &collection));

  loom_sanitizer_site_collect_state_t collect_state = {
      .module = module,
      .rows = collection.rows,
      .next_site_id = 0,
  };
  IREE_RETURN_IF_ERROR(
      loom_walk_region(module, region, LOOM_WALK_PRE_ORDER,
                       (loom_walk_callback_t){
                           loom_sanitizer_site_collect_visitor, &collect_state},
                       arena, &walk_result));
  *out_collection = collection;
  return iree_ok_status();
}

iree_status_t loom_sanitizer_site_collection_build_function(
    const loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena,
    loom_sanitizer_site_collection_t* out_collection) {
  IREE_ASSERT_ARGUMENT(out_collection);
  *out_collection = (loom_sanitizer_site_collection_t){0};

  loom_sanitizer_site_count_state_t count_state = {0};
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_sanitizer_site_count_visitor, &count_state},
      arena, &walk_result));

  loom_sanitizer_site_collection_t collection = {0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_collection_allocate_rows(
      count_state.count, arena, &collection));

  loom_sanitizer_site_collect_state_t collect_state = {
      .module = module,
      .rows = collection.rows,
      .next_site_id = 0,
  };
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_sanitizer_site_collect_visitor,
                             &collect_state},
      arena, &walk_result));
  *out_collection = collection;
  return iree_ok_status();
}
