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

#include "loom/analysis/storage_interference.h"
#include "loom/codegen/low/source_storage_packing.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/low_legality.h"

enum {
  LOOM_AMDGPU_SOURCE_ALLOCA_SPACE_COUNT =
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

typedef struct loom_amdgpu_source_alloca_layout_segment_t {
  // Shared stable byte-range packing for the memory-space arena.
  loom_source_storage_packing_t* packing;
  // Emitted low-storage arena root, or INVALID before entry setup.
  loom_value_id_t low_storage_value_id;
} loom_amdgpu_source_alloca_layout_segment_t;

struct loom_amdgpu_source_alloca_layout_t {
  // Source value domain covered by entries.
  const loom_local_value_domain_t* value_domain;
  // Module containing source operations and values.
  const loom_module_t* module;
  // Fact table used to derive allocation sizes.
  const loom_value_fact_table_t* fact_table;
  // Arena owning packed allocation records and retained interference facts.
  iree_arena_allocator_t* arena;
  // Source function covered by entries.
  const loom_op_t* source_function_op;
  // Allocation-root entries indexed by source value ordinal.
  loom_amdgpu_source_alloca_layout_entry_t* entries;
  // Number of entry slots.
  iree_host_size_t entry_count;
  // Per-memory-space arenas for selected allocation layout.
  loom_amdgpu_source_alloca_layout_segment_t
      segments[LOOM_AMDGPU_SOURCE_ALLOCA_SPACE_COUNT];
  // Shared lifetime and interference facts for source allocations.
  loom_storage_interference_t* interference;
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
    loom_value_fact_memory_space_t memory_space, uint64_t byte_offset) {
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
}

static iree_status_t
loom_amdgpu_source_alloca_layout_query_workgroup_interference(
    void* user_data, loom_value_id_t lhs_root_value_id,
    loom_value_id_t rhs_root_value_id, bool* out_interferes) {
  loom_amdgpu_source_alloca_layout_t* layout =
      (loom_amdgpu_source_alloca_layout_t*)user_data;
  *out_interferes = true;
  bool proven_nonoverlap = false;
  IREE_RETURN_IF_ERROR(loom_storage_interference_prove_workgroup_nonoverlap(
      layout->interference, lhs_root_value_id, rhs_root_value_id,
      &proven_nonoverlap));
  *out_interferes = !proven_nonoverlap;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_source_alloca_layout_query_conservative_interference(
    void* user_data, loom_value_id_t lhs_root_value_id,
    loom_value_id_t rhs_root_value_id, bool* out_interferes) {
  (void)user_data;
  (void)lhs_root_value_id;
  (void)rhs_root_value_id;
  *out_interferes = true;
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
  if (!segment->packing) {
    const loom_source_storage_packing_interference_fn_t interference_fn =
        memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? loom_amdgpu_source_alloca_layout_query_workgroup_interference
            : loom_amdgpu_source_alloca_layout_query_conservative_interference;
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_create(
        loom_source_storage_packing_interference_callback_make(interference_fn,
                                                               layout),
        layout->arena, &segment->packing));
  }
  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_source_storage_packing_append(
      segment->packing, root_value_id, byte_length, byte_alignment,
      &byte_offset));
  loom_amdgpu_source_alloca_layout_record_entry(
      layout->value_domain, layout, root_value_id, memory_space, byte_offset);
  return iree_ok_status();
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
  layout->interference = NULL;
  layout->flags = 0;
  if (layout->entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, layout->entry_count,
                                                   sizeof(*layout->entries),
                                                   (void**)&layout->entries));
    memset(layout->entries, 0, layout->entry_count * sizeof(*layout->entries));
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(layout->segments); ++i) {
    layout->segments[i] = (loom_amdgpu_source_alloca_layout_segment_t){
        .low_storage_value_id = LOOM_VALUE_ID_INVALID,
    };
  }
  if (value_domain != NULL &&
      loom_local_value_domain_is_acquired(value_domain)) {
    IREE_RETURN_IF_ERROR(loom_storage_interference_analyze_function(
        module, fact_table, value_domain, source_function, arena,
        &layout->interference));
  }
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
    if (!segment->packing) {
      continue;
    }
    const loom_source_storage_packing_requirement_t requirement =
        loom_source_storage_packing_requirement(segment->packing);
    IREE_ASSERT_EQ(segment->low_storage_value_id, LOOM_VALUE_ID_INVALID);
    loom_op_t* storage_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
        builder, (int64_t)requirement.byte_length,
        (int64_t)requirement.byte_alignment, loom_type_storage(storage_space),
        layout->source_function_op->location, &storage_op));
    segment->low_storage_value_id =
        loom_low_storage_reserve_storage(storage_op);
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

bool loom_amdgpu_source_alloca_layout_lookup_byte_offset(
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
    loom_value_id_t* out_storage_value_id, int64_t* out_byte_offset) {
  IREE_ASSERT_ARGUMENT(out_storage_value_id);
  IREE_ASSERT_ARGUMENT(out_byte_offset);
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
  IREE_ASSERT_LE(entry->byte_offset, INT64_MAX);
  *out_storage_value_id = segment->low_storage_value_id;
  *out_byte_offset = (int64_t)entry->byte_offset;
  IREE_ASSERT_NE(*out_storage_value_id, LOOM_VALUE_ID_INVALID);
}
