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

#include "loom/ir/local_value_domain.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/low_legality.h"

enum {
  LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_MEMORY_SPACE_COUNT =
      LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC + 1u,
};

typedef uint8_t loom_amdgpu_source_alloca_layout_entry_flags_t;

#define LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET ((uint8_t)1u << 0)

typedef struct loom_amdgpu_source_alloca_layout_entry_t {
  // Entry state bits.
  loom_amdgpu_source_alloca_layout_entry_flags_t flags;
  // Memory space containing the source allocation root.
  loom_value_fact_memory_space_t memory_space;
  // Analyzed byte offset assigned to the allocation root.
  uint64_t byte_offset;
} loom_amdgpu_source_alloca_layout_entry_t;

typedef uint8_t loom_amdgpu_source_alloca_layout_segment_flags_t;

#define LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_SEGMENT_VALID ((uint8_t)1u << 0)

typedef struct loom_amdgpu_source_alloca_layout_segment_t {
  // Segment state bits for prefix layout validity.
  loom_amdgpu_source_alloca_layout_segment_flags_t flags;
  // Next byte offset before applying the next allocation alignment.
  uint64_t byte_size;
} loom_amdgpu_source_alloca_layout_segment_t;

struct loom_amdgpu_source_alloca_layout_t {
  // Source value domain covered by entries.
  const loom_local_value_domain_t* value_domain;
  // Fact table used to derive allocation sizes.
  const loom_value_fact_table_t* fact_table;
  // Source function covered by entries.
  const loom_op_t* source_function_op;
  // Allocation-root entries indexed by source value ordinal.
  loom_amdgpu_source_alloca_layout_entry_t* entries;
  // Number of entry slots.
  iree_host_size_t entry_count;
  // Per-memory-space segment cursors for selected allocation layout.
  loom_amdgpu_source_alloca_layout_segment_t
      segments[LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_MEMORY_SPACE_COUNT];
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
    const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    loom_func_like_t source_function) {
  return iree_all_bits_set(layout->flags,
                           LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_INITIALIZED) &&
         layout->fact_table == fact_table &&
         layout->value_domain == value_domain &&
         layout->source_function_op == source_function.op;
}

static void loom_amdgpu_source_alloca_layout_record_entry(
    const loom_local_value_domain_t* value_domain,
    loom_amdgpu_source_alloca_layout_t* layout, loom_value_id_t root_value_id,
    loom_value_fact_memory_space_t memory_space, uint64_t byte_offset) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, root_value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= layout->entry_count) {
    return;
  }
  layout->entries[value_ordinal] = (loom_amdgpu_source_alloca_layout_entry_t){
      .flags = LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET,
      .memory_space = memory_space,
      .byte_offset = byte_offset,
  };
}

static void loom_amdgpu_source_alloca_layout_record_allocation(
    loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    uint64_t byte_length, uint64_t byte_alignment) {
  if ((uint32_t)memory_space >= IREE_ARRAYSIZE(layout->segments)) {
    return;
  }
  if (layout->value_domain == NULL ||
      !loom_local_value_domain_is_acquired(layout->value_domain)) {
    return;
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(layout->value_domain, root_value_id);
  if (value_ordinal != LOOM_VALUE_ORDINAL_INVALID &&
      value_ordinal < layout->entry_count &&
      iree_all_bits_set(layout->entries[value_ordinal].flags,
                        LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_ENTRY_HAS_OFFSET)) {
    return;
  }

  loom_amdgpu_source_alloca_layout_segment_t* segment =
      &layout->segments[memory_space];
  if (!iree_all_bits_set(segment->flags,
                         LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_SEGMENT_VALID)) {
    return;
  }

  uint64_t slot_byte_offset = 0;
  if (!iree_is_power_of_two_uint64(byte_alignment) ||
      !iree_checked_align_u64(segment->byte_size, byte_alignment,
                              &slot_byte_offset)) {
    segment->flags &=
        (loom_amdgpu_source_alloca_layout_segment_flags_t)~LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_SEGMENT_VALID;
    return;
  }
  loom_amdgpu_source_alloca_layout_record_entry(layout->value_domain, layout,
                                                root_value_id, memory_space,
                                                slot_byte_offset);
  uint64_t next_segment_byte_size = 0;
  if (!iree_checked_add_u64(slot_byte_offset, byte_length,
                            &next_segment_byte_size)) {
    segment->flags &=
        (loom_amdgpu_source_alloca_layout_segment_flags_t)~LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_SEGMENT_VALID;
    return;
  }
  segment->byte_size = next_segment_byte_size;
}

static iree_status_t loom_amdgpu_source_alloca_layout_initialize(
    const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_func_like_t source_function,
    loom_amdgpu_source_alloca_layout_t* layout) {
  layout->value_domain = value_domain;
  layout->fact_table = fact_table;
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
    layout->segments[i] = (loom_amdgpu_source_alloca_layout_segment_t){
        .flags = LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_SEGMENT_VALID,
    };
  }
  layout->flags = LOOM_AMDGPU_SOURCE_ALLOCA_LAYOUT_INITIALIZED;
  return iree_ok_status();
}

const loom_amdgpu_source_alloca_layout_t*
loom_amdgpu_source_alloca_layout_empty(void) {
  return &kLoomAmdgpuSourceAllocaLayoutEmpty;
}

static iree_status_t loom_amdgpu_source_alloca_layout_initialize_for_inputs(
    const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_func_like_t source_function,
    loom_amdgpu_source_alloca_layout_t* layout) {
  if (!loom_amdgpu_source_alloca_layout_matches(
          layout, fact_table, value_domain, source_function)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_initialize(
        fact_table, value_domain, arena, source_function, layout));
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
      fact_table, value_domain, loom_low_lower_context_scratch_arena(context),
      source_function, layout));
  *out_layout = layout;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_source_alloca_layout_record_lower_alloca(
    loom_low_lower_context_t* context,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    uint64_t byte_length, uint64_t byte_alignment) {
  const loom_amdgpu_source_alloca_layout_t* const_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &const_layout));
  loom_amdgpu_source_alloca_layout_t* layout =
      (loom_amdgpu_source_alloca_layout_t*)const_layout;
  loom_amdgpu_source_alloca_layout_record_allocation(
      layout, memory_space, root_value_id, byte_length, byte_alignment);
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
      fact_table, loom_target_low_legality_value_domain(context),
      loom_target_low_legality_scratch_arena(context), source_function,
      layout));
  *out_layout = layout;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_source_alloca_layout_record_low_legality_alloca(
    loom_target_low_legality_context_t* context,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    uint64_t byte_length, uint64_t byte_alignment) {
  const loom_amdgpu_source_alloca_layout_t* const_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_low_legality(
      context, &const_layout));
  loom_amdgpu_source_alloca_layout_t* layout =
      (loom_amdgpu_source_alloca_layout_t*)const_layout;
  loom_amdgpu_source_alloca_layout_record_allocation(
      layout, memory_space, root_value_id, byte_length, byte_alignment);
  return iree_ok_status();
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
