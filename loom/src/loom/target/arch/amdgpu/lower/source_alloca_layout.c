// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source allocation layout analysis.
//
// Source-to-low lowering must know storage bases for packets that encode a
// source allocation directly, such as LDS and scratch memory packets. Buffer
// planning records selected allocations into this function-local analysis, and
// packet selectors consume the resulting table through O(1) root lookups. They
// do not recover from a missing entry by rescanning source IR.

#include <stdint.h>
#include <string.h>

#include "loom/analysis/control_uniformity.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/low_legality.h"

enum {
  LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_MEMORY_SPACE_COUNT =
      LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC + 1u,
};

typedef uint8_t loom_amdgpu_source_alloca_layout_entry_flags_t;

#define LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET ((uint8_t)1u << 0)

typedef struct loom_amdgpu_source_alloca_layout_footprint_t {
  // Operations that define or use the source allocation and its aliases.
  const loom_op_t** operations;
  // Number of operations in the allocation footprint.
  iree_host_size_t operation_count;
  // Allocated operation pointer capacity.
  iree_host_size_t operation_capacity;
} loom_amdgpu_source_alloca_layout_footprint_t;

typedef struct loom_amdgpu_source_alloca_layout_entry_t {
  // Entry state bits.
  loom_amdgpu_source_alloca_layout_entry_flags_t flags;
  // Memory space containing the source allocation root.
  loom_value_fact_memory_space_t memory_space;
  // Analyzed byte offset assigned to the allocation root.
  uint64_t byte_offset;
  // Physical slot ordinal in the allocation memory-space segment.
  iree_host_size_t slot_ordinal;
  // Complete operation footprint when this value is an allocation root.
  loom_amdgpu_source_alloca_layout_footprint_t* footprint;
} loom_amdgpu_source_alloca_layout_entry_t;

typedef struct loom_amdgpu_source_alloca_layout_occupant_t {
  // Source allocation root value.
  loom_value_id_t root_value_id;
  // Next allocation sharing the physical slot.
  struct loom_amdgpu_source_alloca_layout_occupant_t* next;
} loom_amdgpu_source_alloca_layout_occupant_t;

typedef struct loom_amdgpu_source_alloca_layout_slot_t {
  // Physical byte offset in the memory-space root.
  uint64_t byte_offset;
  // Physical slot capacity in bytes.
  uint64_t byte_size;
  // Maximum base alignment required by allocations sharing the slot.
  uint64_t byte_alignment;
  // Source allocations assigned to the slot.
  loom_amdgpu_source_alloca_layout_occupant_t* occupants;
  // Emitted low-storage root, or INVALID before entry setup.
  loom_value_id_t low_storage_value_id;
} loom_amdgpu_source_alloca_layout_slot_t;

typedef struct loom_amdgpu_source_alloca_layout_segment_t {
  // Next byte offset before applying the next allocation alignment.
  uint64_t byte_size;
  // Physical slots in stable first-allocation order.
  loom_amdgpu_source_alloca_layout_slot_t* slots;
  // Number of initialized physical slots.
  iree_host_size_t slot_count;
  // Allocated physical-slot capacity.
  iree_host_size_t slot_capacity;
} loom_amdgpu_source_alloca_layout_segment_t;

struct loom_amdgpu_source_alloca_layout_t {
  // Source value domain covered by entries.
  const loom_local_value_domain_t* value_domain;
  // Module containing source operations and values.
  const loom_module_t* module;
  // Fact table used to derive allocation sizes.
  const loom_value_fact_table_t* fact_table;
  // Arena owning allocation footprints, slots, and retained control summaries.
  iree_arena_allocator_t* arena;
  // Source function covered by entries.
  const loom_op_t* source_function_op;
  // Allocation-root entries indexed by source value ordinal.
  loom_amdgpu_source_alloca_layout_entry_t* entries;
  // Number of entry slots.
  iree_host_size_t entry_count;
  // Per-memory-space segment cursors for selected allocation layout.
  loom_amdgpu_source_alloca_layout_segment_t
      segments[LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_MEMORY_SPACE_COUNT];
  // Lazily populated control facts used for branch-exclusive slots.
  loom_control_uniformity_info_t control_uniformity;
  // Analysis lifecycle bits.
  uint8_t flags;
};

#define LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_INITIALIZED ((uint8_t)1u << 0)
static int loom_amdgpu_source_alloca_layout_state_key;

static const loom_amdgpu_source_alloca_layout_t
    kLoomAmdgpuSourceAllocaLayoutEmpty = {
        .flags = LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_INITIALIZED,
};

static bool loom_amdgpu_source_alloca_layout_matches(
    const loom_amdgpu_source_alloca_layout_t* layout,
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    loom_func_like_t source_function) {
  return iree_all_bits_set(layout->flags,
                           LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_INITIALIZED) &&
         layout->fact_table == fact_table && layout->module == module &&
         layout->value_domain == value_domain &&
         layout->source_function_op == source_function.op;
}

static void loom_amdgpu_source_alloca_layout_record_entry(
    const loom_local_value_domain_t* value_domain,
    loom_amdgpu_source_alloca_layout_t* layout, loom_value_id_t root_value_id,
    loom_value_fact_memory_space_t memory_space, uint64_t byte_offset,
    iree_host_size_t slot_ordinal) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, root_value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= layout->entry_count) {
    return;
  }
  loom_amdgpu_source_alloca_layout_entry_t* entry =
      &layout->entries[value_ordinal];
  entry->flags = LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET;
  entry->memory_space = memory_space;
  entry->byte_offset = byte_offset;
  entry->slot_ordinal = slot_ordinal;
}

static bool loom_amdgpu_source_alloca_layout_value_root(
    const loom_amdgpu_source_alloca_layout_t* layout, loom_value_id_t value_id,
    loom_value_id_t* out_root_value_id) {
  *out_root_value_id = LOOM_VALUE_ID_INVALID;
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(layout->fact_table, value_id);
  loom_value_fact_buffer_reference_t buffer_reference;
  if (loom_value_facts_query_buffer_reference(&layout->fact_table->context,
                                              facts, &buffer_reference)) {
    *out_root_value_id = loom_value_fact_buffer_reference_resolve_root_value(
        buffer_reference, value_id);
    return true;
  }
  loom_value_fact_view_reference_t view_reference;
  if (loom_value_facts_query_view_reference(&layout->fact_table->context, facts,
                                            &view_reference)) {
    *out_root_value_id = view_reference.root_value_id;
    return true;
  }
  return false;
}

static iree_status_t
loom_amdgpu_source_alloca_layout_append_footprint_operation(
    loom_amdgpu_source_alloca_layout_t* layout,
    loom_amdgpu_source_alloca_layout_footprint_t* footprint,
    const loom_op_t* operation) {
  if (footprint->operation_count >= footprint->operation_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        layout->arena, footprint->operation_count,
        footprint->operation_capacity == 0 ? 8
                                           : footprint->operation_capacity * 2,
        sizeof(*footprint->operations), &footprint->operation_capacity,
        (void**)&footprint->operations));
  }
  footprint->operations[footprint->operation_count++] = operation;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_alloca_layout_initialize_footprints(
    loom_amdgpu_source_alloca_layout_t* layout) {
  if (layout->entry_count == 0) return iree_ok_status();
  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < layout->value_domain->value_count; ++value_ordinal) {
    const loom_value_id_t value_id =
        layout->value_domain->value_ids[value_ordinal];
    loom_value_id_t value_root_id = LOOM_VALUE_ID_INVALID;
    if (!loom_amdgpu_source_alloca_layout_value_root(layout, value_id,
                                                     &value_root_id)) {
      continue;
    }
    const loom_value_ordinal_t root_ordinal =
        loom_local_value_domain_try_ordinal(layout->value_domain,
                                            value_root_id);
    if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID) continue;

    const loom_value_t* root_value =
        loom_module_value(layout->module, value_root_id);
    if (loom_value_is_block_arg(root_value)) continue;
    const loom_op_t* alloca_op = loom_value_def_op(root_value);
    if (!alloca_op || !loom_buffer_alloca_isa(alloca_op) ||
        !alloca_op->parent_block) {
      continue;
    }
    loom_amdgpu_source_alloca_layout_entry_t* entry =
        &layout->entries[root_ordinal];
    if (!entry->footprint) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          layout->arena, sizeof(*entry->footprint), (void**)&entry->footprint));
      *entry->footprint = (loom_amdgpu_source_alloca_layout_footprint_t){0};
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_source_alloca_layout_append_footprint_operation(
              layout, entry->footprint, alloca_op));
    }
    const loom_value_t* value = loom_module_value(layout->module, value_id);
    const loom_use_t* use = NULL;
    loom_value_for_each_use(value, use) {
      const loom_op_t* user_op = loom_use_user_op(*use);
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_source_alloca_layout_append_footprint_operation(
              layout, entry->footprint, user_op));
    }
  }
  return iree_ok_status();
}

static const loom_amdgpu_source_alloca_layout_footprint_t*
loom_amdgpu_source_alloca_layout_lookup_footprint(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_id_t root_value_id) {
  const loom_value_ordinal_t root_ordinal =
      loom_local_value_domain_try_ordinal(layout->value_domain, root_value_id);
  if (root_ordinal == LOOM_VALUE_ORDINAL_INVALID) return NULL;
  return layout->entries[root_ordinal].footprint;
}

static bool loom_amdgpu_source_alloca_layout_slot_can_grow(
    const loom_amdgpu_source_alloca_layout_segment_t* segment,
    const loom_amdgpu_source_alloca_layout_slot_t* slot, uint64_t byte_size) {
  uint64_t slot_end = 0;
  return byte_size > slot->byte_size &&
         iree_checked_add_u64(slot->byte_offset, slot->byte_size, &slot_end) &&
         slot_end == segment->byte_size;
}

static iree_status_t loom_amdgpu_source_alloca_layout_append_occupant(
    loom_amdgpu_source_alloca_layout_t* layout,
    loom_amdgpu_source_alloca_layout_slot_t* slot,
    loom_value_id_t root_value_id) {
  loom_amdgpu_source_alloca_layout_occupant_t* occupant = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(layout->arena, sizeof(*occupant), (void**)&occupant));
  *occupant = (loom_amdgpu_source_alloca_layout_occupant_t){
      .root_value_id = root_value_id,
      .next = slot->occupants,
  };
  slot->occupants = occupant;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_alloca_layout_try_reuse_slot(
    loom_amdgpu_source_alloca_layout_t* layout,
    loom_amdgpu_source_alloca_layout_segment_t* segment,
    loom_amdgpu_source_alloca_layout_slot_t* slot, const loom_op_t* alloca_op,
    loom_value_id_t root_value_id, uint64_t byte_length,
    uint64_t byte_alignment, bool* out_reused) {
  *out_reused = false;
  if (slot->byte_offset % byte_alignment != 0 ||
      (byte_length > slot->byte_size &&
       !loom_amdgpu_source_alloca_layout_slot_can_grow(segment, slot,
                                                       byte_length))) {
    return iree_ok_status();
  }
  const loom_amdgpu_source_alloca_layout_footprint_t* footprint =
      loom_amdgpu_source_alloca_layout_lookup_footprint(layout, root_value_id);
  if (!footprint) return iree_ok_status();
  for (loom_amdgpu_source_alloca_layout_occupant_t* occupant = slot->occupants;
       occupant; occupant = occupant->next) {
    const loom_amdgpu_source_alloca_layout_footprint_t* occupant_footprint =
        loom_amdgpu_source_alloca_layout_lookup_footprint(
            layout, occupant->root_value_id);
    if (!occupant_footprint) return iree_ok_status();
    bool mutually_exclusive = false;
    IREE_RETURN_IF_ERROR(
        loom_control_uniformity_prove_mutually_exclusive_execution(
            &layout->control_uniformity, footprint->operation_count,
            footprint->operations, occupant_footprint->operation_count,
            occupant_footprint->operations,
            LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP, &mutually_exclusive));
    if (!mutually_exclusive) return iree_ok_status();
  }

  if (byte_length > slot->byte_size) {
    uint64_t next_segment_byte_size = 0;
    if (!iree_checked_add_u64(slot->byte_offset, byte_length,
                              &next_segment_byte_size) ||
        next_segment_byte_size > INT64_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source allocation layout exceeds INT64_MAX");
    }
    slot->byte_size = byte_length;
    segment->byte_size = next_segment_byte_size;
  }
  slot->byte_alignment = iree_max(slot->byte_alignment, byte_alignment);
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_append_occupant(
      layout, slot, root_value_id));
  loom_amdgpu_source_alloca_layout_record_entry(
      layout->value_domain, layout, root_value_id,
      loom_buffer_alloca_memory_space(alloca_op), slot->byte_offset,
      (iree_host_size_t)(slot - segment->slots));
  *out_reused = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_alloca_layout_append_slot(
    loom_amdgpu_source_alloca_layout_t* layout,
    loom_amdgpu_source_alloca_layout_segment_t* segment,
    loom_value_fact_memory_space_t memory_space, const loom_op_t* alloca_op,
    loom_value_id_t root_value_id, uint64_t byte_length,
    uint64_t byte_alignment) {
  uint64_t slot_byte_offset = 0;
  if (!iree_is_power_of_two_uint64(byte_alignment) ||
      !iree_checked_align_u64(segment->byte_size, byte_alignment,
                              &slot_byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source allocation layout alignment overflows");
  }
  uint64_t next_segment_byte_size = 0;
  if (!iree_checked_add_u64(slot_byte_offset, byte_length,
                            &next_segment_byte_size) ||
      next_segment_byte_size > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source allocation layout exceeds INT64_MAX");
  }
  const iree_host_size_t minimum_capacity = segment->slot_count + 1;
  if (minimum_capacity > segment->slot_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        layout->arena, segment->slot_count, iree_max(minimum_capacity, 4u),
        sizeof(*segment->slots), &segment->slot_capacity,
        (void**)&segment->slots));
  }
  loom_amdgpu_source_alloca_layout_slot_t* slot =
      &segment->slots[segment->slot_count++];
  *slot = (loom_amdgpu_source_alloca_layout_slot_t){
      .byte_offset = slot_byte_offset,
      .byte_size = byte_length,
      .byte_alignment = byte_alignment,
      .low_storage_value_id = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_append_occupant(
      layout, slot, root_value_id));
  segment->byte_size = next_segment_byte_size;
  loom_amdgpu_source_alloca_layout_record_entry(
      layout->value_domain, layout, root_value_id, memory_space,
      slot_byte_offset, segment->slot_count - 1);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_alloca_layout_record_allocation(
    loom_amdgpu_source_alloca_layout_t* layout, const loom_op_t* alloca_op,
    uint64_t byte_length) {
  const loom_value_fact_memory_space_t memory_space =
      loom_buffer_alloca_memory_space(alloca_op);
  const loom_value_id_t root_value_id = loom_buffer_alloca_result(alloca_op);
  const uint64_t byte_alignment =
      (uint64_t)loom_buffer_alloca_base_alignment(alloca_op);
  if ((uint32_t)memory_space >= IREE_ARRAYSIZE(layout->segments)) {
    return iree_ok_status();
  }
  if (layout->value_domain == NULL ||
      !loom_local_value_domain_is_acquired(layout->value_domain)) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(layout->value_domain, root_value_id);
  if (value_ordinal != LOOM_VALUE_ORDINAL_INVALID &&
      value_ordinal < layout->entry_count &&
      iree_all_bits_set(layout->entries[value_ordinal].flags,
                        LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET)) {
    return iree_ok_status();
  }

  loom_amdgpu_source_alloca_layout_segment_t* segment =
      &layout->segments[memory_space];

  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    for (iree_host_size_t i = 0; i < segment->slot_count; ++i) {
      bool reused = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_try_reuse_slot(
          layout, segment, &segment->slots[i], alloca_op, root_value_id,
          byte_length, byte_alignment, &reused));
      if (reused) return iree_ok_status();
    }
  }
  return loom_amdgpu_source_alloca_layout_append_slot(
      layout, segment, memory_space, alloca_op, root_value_id, byte_length,
      byte_alignment);
}

static iree_status_t loom_amdgpu_source_alloca_layout_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_func_like_t source_function,
    loom_amdgpu_source_alloca_layout_t* layout) {
  layout->value_domain = value_domain;
  layout->module = module;
  layout->fact_table = fact_table;
  layout->arena = arena;
  layout->source_function_op = source_function.op;
  layout->entries = NULL;
  layout->entry_count = value_domain != NULL ? value_domain->value_count : 0;
  layout->flags = 0;
  if (layout->entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, layout->entry_count,
                                                   sizeof(*layout->entries),
                                                   (void**)&layout->entries));
    memset(layout->entries, 0, layout->entry_count * sizeof(*layout->entries));
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(layout->segments); ++i) {
    layout->segments[i] = (loom_amdgpu_source_alloca_layout_segment_t){0};
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_alloca_layout_initialize_footprints(layout));
  loom_control_uniformity_info_initialize(module, fact_table, arena,
                                          &layout->control_uniformity);
  layout->flags = LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_INITIALIZED;
  return iree_ok_status();
}

const loom_amdgpu_source_alloca_layout_t*
loom_amdgpu_source_alloca_layout_empty(void) {
  return &kLoomAmdgpuSourceAllocaLayoutEmpty;
}

static iree_status_t loom_amdgpu_source_alloca_layout_initialize_for_inputs(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_func_like_t source_function,
    loom_amdgpu_source_alloca_layout_t* layout) {
  if (!loom_amdgpu_source_alloca_layout_matches(
          layout, module, fact_table, value_domain, source_function)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_initialize(
        module, fact_table, value_domain, arena, source_function, layout));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_source_alloca_layout_for_lower_context(
    loom_low_lower_context_t* context,
    const loom_amdgpu_source_alloca_layout_t** out_layout) {
  *out_layout = NULL;
  loom_amdgpu_source_alloca_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_source_alloca_layout_state_key, sizeof(*layout),
      (void**)&layout));
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_local_value_domain_t* value_domain =
      loom_low_lower_context_value_domain(context);
  const loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_initialize_for_inputs(
      loom_low_lower_context_module(context), fact_table, value_domain,
      loom_low_lower_context_function_arena(context), source_function, layout));
  *out_layout = layout;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_source_alloca_layout_record_lower_alloca(
    loom_low_lower_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length) {
  const loom_amdgpu_source_alloca_layout_t* const_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &const_layout));
  loom_amdgpu_source_alloca_layout_t* layout =
      (loom_amdgpu_source_alloca_layout_t*)const_layout;
  return loom_amdgpu_source_alloca_layout_record_allocation(layout, alloca_op,
                                                            byte_length);
}

static bool loom_amdgpu_source_alloca_layout_storage_space(
    loom_value_fact_memory_space_t memory_space,
    loom_storage_space_t* out_storage_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      *out_storage_space = LOOM_STORAGE_SPACE_PRIVATE;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      *out_storage_space = LOOM_STORAGE_SPACE_WORKGROUP;
      return true;
    default:
      return false;
  }
}

iree_status_t loom_amdgpu_source_alloca_layout_emit_low_storage_roots(
    loom_low_lower_context_t* context) {
  const loom_amdgpu_source_alloca_layout_t* const_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &const_layout));
  loom_amdgpu_source_alloca_layout_t* layout =
      (loom_amdgpu_source_alloca_layout_t*)const_layout;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(layout->segments); ++i) {
    loom_amdgpu_source_alloca_layout_segment_t* segment = &layout->segments[i];
    loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_STACK;
    if (!loom_amdgpu_source_alloca_layout_storage_space(
            (loom_value_fact_memory_space_t)i, &storage_space)) {
      continue;
    }
    for (iree_host_size_t j = 0; j < segment->slot_count; ++j) {
      loom_amdgpu_source_alloca_layout_slot_t* slot = &segment->slots[j];
      IREE_ASSERT_EQ(slot->low_storage_value_id, LOOM_VALUE_ID_INVALID);
      loom_op_t* storage_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
          builder, (int64_t)slot->byte_size, (int64_t)slot->byte_alignment,
          loom_type_storage(storage_space),
          layout->source_function_op->location, &storage_op));
      slot->low_storage_value_id = loom_low_storage_reserve_storage(storage_op);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_source_alloca_layout_for_low_legality(
    loom_target_low_legality_context_t* context,
    const loom_amdgpu_source_alloca_layout_t** out_layout) {
  *out_layout = NULL;
  loom_amdgpu_source_alloca_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_get_or_allocate_target_state(
      context, &loom_amdgpu_source_alloca_layout_state_key, sizeof(*layout),
      (void**)&layout));
  const loom_value_fact_table_t* fact_table =
      loom_target_low_legality_fact_table(context);
  const loom_func_like_t source_function =
      loom_target_low_legality_function(context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_initialize_for_inputs(
      loom_target_low_legality_module(context), fact_table,
      loom_target_low_legality_value_domain(context),
      loom_target_low_legality_scratch_arena(context), source_function,
      layout));
  *out_layout = layout;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_source_alloca_layout_record_low_legality_alloca(
    loom_target_low_legality_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length) {
  const loom_amdgpu_source_alloca_layout_t* const_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_low_legality(
      context, &const_layout));
  loom_amdgpu_source_alloca_layout_t* layout =
      (loom_amdgpu_source_alloca_layout_t*)const_layout;
  return loom_amdgpu_source_alloca_layout_record_allocation(layout, alloca_op,
                                                            byte_length);
}

bool loom_amdgpu_source_alloca_layout_lookup_root(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    uint64_t* out_byte_offset) {
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(out_byte_offset);
  *out_byte_offset = 0;
  if (layout->value_domain == NULL ||
      !loom_local_value_domain_is_acquired(layout->value_domain)) {
    return false;
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(layout->value_domain, root_value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= layout->entry_count) {
    return false;
  }
  const loom_amdgpu_source_alloca_layout_entry_t* entry =
      &layout->entries[value_ordinal];
  if (!iree_all_bits_set(entry->flags,
                         LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET) ||
      entry->memory_space != memory_space) {
    return false;
  }
  *out_byte_offset = entry->byte_offset;
  return true;
}

void loom_amdgpu_source_alloca_layout_lookup_low_storage(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    loom_value_id_t* out_storage_value_id) {
  IREE_ASSERT_ARGUMENT(out_storage_value_id);
  IREE_ASSERT((uint32_t)memory_space < IREE_ARRAYSIZE(layout->segments));
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(layout->value_domain, root_value_id);
  const loom_amdgpu_source_alloca_layout_entry_t* entry =
      &layout->entries[value_ordinal];
  IREE_ASSERT(
      iree_all_bits_set(entry->flags,
                        LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET) &&
      entry->memory_space == memory_space);
  const loom_amdgpu_source_alloca_layout_segment_t* segment =
      &layout->segments[memory_space];
  IREE_ASSERT_LT(entry->slot_ordinal, segment->slot_count);
  *out_storage_value_id =
      segment->slots[entry->slot_ordinal].low_storage_value_id;
  IREE_ASSERT_NE(*out_storage_value_id, LOOM_VALUE_ID_INVALID);
}
