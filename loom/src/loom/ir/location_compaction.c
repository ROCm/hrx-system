// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/location_compaction.h"

#include <string.h>

#include "loom/ir/module.h"

typedef struct loom_location_compaction_t {
  // Module whose live operation locations are being compacted.
  loom_module_t* module;
  // One byte per old location ID; nonzero identifies a retained row.
  uint8_t* retained;
  // Location-graph worklist, later reused as the old-to-new ID map.
  loom_location_id_t* worklist;
  // Number of entries published in |worklist|.
  iree_host_size_t worklist_count;
  // Next worklist entry whose transitive children have not been marked.
  iree_host_size_t worklist_head;
} loom_location_compaction_t;

static void loom_location_compaction_mark(loom_location_compaction_t* state,
                                          loom_location_id_t location_id) {
  IREE_ASSERT((iree_host_size_t)location_id < state->module->locations.count);
  if (state->retained[location_id]) {
    return;
  }
  state->retained[location_id] = 1;
  state->worklist[state->worklist_count++] = location_id;
}

static void loom_location_compaction_mark_region(
    loom_location_compaction_t* state, loom_region_t* region) {
  if (!region) {
    return;
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_location_compaction_mark(state, op->location);
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        loom_location_compaction_mark_region(state, regions[i]);
      }
    }
  }
}

static void loom_location_compaction_mark_children(
    loom_location_compaction_t* state) {
  while (state->worklist_head < state->worklist_count) {
    const loom_location_id_t location_id =
        state->worklist[state->worklist_head++];
    const loom_location_entry_t* entry =
        loom_location_table_const_entry(&state->module->locations, location_id);
    switch (entry->kind) {
      case LOOM_LOCATION_FUSED:
        IREE_ASSERT(entry->fused.count == 0 || entry->fused.children);
        for (uint32_t i = 0; i < entry->fused.count; ++i) {
          loom_location_compaction_mark(state, entry->fused.children[i]);
        }
        break;
      case LOOM_LOCATION_TAGGED:
        loom_location_compaction_mark(state, entry->tagged.child);
        break;
      default:
        break;
    }
  }
}

static void loom_location_compaction_remap_region(
    const uint8_t* retained, const loom_location_id_t* new_location_ids,
    loom_region_t* region) {
  if (!region) {
    return;
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_ASSERT(retained[op->location]);
      op->location = new_location_ids[op->location];
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        loom_location_compaction_remap_region(retained, new_location_ids,
                                              regions[i]);
      }
    }
  }
}

iree_status_t loom_module_compact_locations(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_removed_count) {
  if (out_removed_count) {
    *out_removed_count = 0;
  }
  const iree_host_size_t old_location_count = module->locations.count;
  if (old_location_count == 0) {
    return iree_ok_status();
  }

  loom_location_compaction_t state = {.module = module};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, old_location_count, sizeof(*state.retained),
      (void**)&state.retained));
  memset(state.retained, 0, old_location_count * sizeof(*state.retained));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, old_location_count, sizeof(*state.worklist),
      (void**)&state.worklist));

  loom_location_compaction_mark(&state, LOOM_LOCATION_UNKNOWN);
  loom_location_compaction_mark_region(&state, module->body);
  loom_location_compaction_mark_children(&state);
  const iree_host_size_t removed_count =
      old_location_count - state.worklist_count;
  if (removed_count == 0) {
    return iree_ok_status();
  }

  loom_location_id_t* new_location_ids = state.worklist;
  iree_host_size_t new_location_count = 0;
  for (iree_host_size_t old_id = 0; old_id < old_location_count; ++old_id) {
    if (state.retained[old_id]) {
      new_location_ids[old_id] = (loom_location_id_t)new_location_count++;
    }
  }

  iree_host_size_t fused_child_count = 0;
  for (iree_host_size_t old_id = 0; old_id < old_location_count; ++old_id) {
    if (!state.retained[old_id]) {
      continue;
    }
    const loom_location_entry_t* entry = loom_location_table_const_entry(
        &module->locations, (loom_location_id_t)old_id);
    if (entry->kind == LOOM_LOCATION_FUSED &&
        !iree_host_size_checked_add(fused_child_count, entry->fused.count,
                                    &fused_child_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "retained fused location children overflow");
    }
  }
  loom_location_id_t* fused_children = NULL;
  if (fused_child_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, fused_child_count, sizeof(*fused_children),
        (void**)&fused_children));
  }
  iree_host_size_t fused_child_index = 0;
  for (iree_host_size_t old_id = 0; old_id < old_location_count; ++old_id) {
    if (!state.retained[old_id]) {
      continue;
    }
    const loom_location_entry_t* entry = loom_location_table_const_entry(
        &module->locations, (loom_location_id_t)old_id);
    if (entry->kind != LOOM_LOCATION_FUSED) {
      continue;
    }
    for (uint32_t i = 0; i < entry->fused.count; ++i) {
      IREE_ASSERT(state.retained[entry->fused.children[i]]);
      fused_children[fused_child_index++] =
          new_location_ids[entry->fused.children[i]];
    }
  }
  IREE_ASSERT(fused_child_index == fused_child_count);

  loom_location_compaction_remap_region(state.retained, new_location_ids,
                                        module->body);
  fused_child_index = 0;
  for (iree_host_size_t old_id = 0; old_id < old_location_count; ++old_id) {
    if (!state.retained[old_id]) {
      continue;
    }
    const loom_location_id_t new_id = new_location_ids[old_id];
    loom_location_entry_t entry = *loom_location_table_const_entry(
        &module->locations, (loom_location_id_t)old_id);
    if (entry.kind == LOOM_LOCATION_FUSED) {
      entry.fused.children =
          entry.fused.count ? &fused_children[fused_child_index] : NULL;
      fused_child_index += entry.fused.count;
    } else if (entry.kind == LOOM_LOCATION_TAGGED) {
      IREE_ASSERT(state.retained[entry.tagged.child]);
      entry.tagged.child = new_location_ids[entry.tagged.child];
    }
    *loom_location_table_entry(&module->locations, new_id) = entry;
  }
  IREE_ASSERT(fused_child_index == fused_child_count);

  for (iree_host_size_t id = new_location_count; id < old_location_count;
       ++id) {
    *loom_location_table_entry(&module->locations, (loom_location_id_t)id) =
        (loom_location_entry_t){0};
  }
  module->locations.count = new_location_count;
  if (out_removed_count) {
    *out_removed_count = removed_count;
  }
  return iree_ok_status();
}
