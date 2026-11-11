// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "buffer.h"
#include "hrx_internal.h"
#include "hrx_memory.h"
#include "mem_pool.h"

// Inline public predicates consume these stable semantic bits, not native
// cache encodings. The remaining effects pass through unchanged to HAL.
static_assert((uint32_t)IREE_HAL_MEMORY_EFFECT_UNSUPPORTED ==
                  (UINT32_C(1) << 31),
              "unsupported effect encoding must match the inline predicate");
static_assert(IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK == UINT32_C(0x3C),
              "resource effect encoding must match the inline predicate");
static_assert(IREE_HAL_MEMORY_EFFECT_PROGRAM_EXECUTOR == (UINT32_C(1) << 29),
              "program effect encoding must match the inline predicate");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_TRANSITION_RELEASE,
                           IREE_HAL_MEMORY_TRANSITION_RELEASE,
                           "transition actions must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_TRANSITION_ACQUIRE,
                           IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                           "transition actions must match HAL");

static hrx_memory_transition_table_t hrx_memory_transition_table_from_hal(
    iree_hal_memory_transition_table_t table) {
  return (hrx_memory_transition_table_t){
      .contract = table.contract,
      .cells = table.contract ? table.contract->transitions : NULL,
  };
}

hrx_memory_transition_table_t hrx_mem_pool_transition_table(
    hrx_mem_pool_t pool) {
  return hrx_memory_transition_table_from_hal(
      iree_hal_pool_transition_table(pool->hal_pool));
}

hrx_memory_transition_table_t hrx_buffer_transition_table(hrx_buffer_t buffer) {
  const iree_hal_memory_transition_table_t table = {
      .contract =
          buffer->hal_buffer ? buffer->hal_buffer->memory.contract : NULL,
  };
  return hrx_memory_transition_table_from_hal(table);
}

hrx_status_t hrx_memory_transition_prepare_pair(
    hrx_memory_transition_table_t table, hrx_memory_scope_t producer,
    hrx_memory_scope_t consumer, hrx_memory_transition_action_t action,
    hrx_memory_transition_pair_t* out_pair) {
  if (!out_pair) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "prepared transition output is NULL");
  }
  iree_hal_memory_transition_pair_t pair = {0};
  iree_status_t status = iree_hal_memory_transition_prepare_pair(
      (iree_hal_memory_transition_table_t){table.contract},
      (iree_hal_memory_scope_t){producer.domain, producer.id},
      (iree_hal_memory_scope_t){consumer.domain, consumer.id},
      (iree_hal_memory_transition_action_t)action, &pair);
  out_pair->cell_index = pair.cell_index;
  return hrx_status_from_iree(status);
}

const hrx_memory_transition_recipe_t* hrx_memory_transition_recipe(
    hrx_memory_transition_table_t table, hrx_memory_transition_pair_t pair,
    hrx_memory_transition_action_t action) {
  return (const hrx_memory_transition_recipe_t*)
      iree_hal_memory_transition_recipe(
          (iree_hal_memory_transition_table_t){table.contract},
          (iree_hal_memory_transition_pair_t){pair.cell_index},
          (iree_hal_memory_transition_action_t)action);
}

hrx_status_t hrx_buffer_mapping_memory_barrier(
    hrx_memory_effects_t effects, size_t mapping_count,
    const hrx_buffer_mapping_transition_t* mappings) {
  if (mapping_count && !mappings) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "mapping transition array is NULL");
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(mapping_count,
                                  sizeof(iree_hal_buffer_mapping_transition_t),
                                  &allocation_size)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "mapping transition array is too large");
  }
  for (size_t i = 0; i < mapping_count; ++i) {
    if (!mappings[i].buffer || !mappings[i].buffer->is_mapped) {
      return hrx_status_from_iree(
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "buffer %zu has no active host mapping", i));
    }
  }
  iree_hal_buffer_mapping_transition_t inline_mappings[8];
  iree_hal_buffer_mapping_transition_t* hal_mappings = inline_mappings;
  if (mapping_count > IREE_ARRAYSIZE(inline_mappings)) {
    HRX_RETURN_IF_IREE_ERROR(iree_allocator_malloc(
        iree_allocator_system(), allocation_size, (void**)&hal_mappings));
  }
  for (size_t i = 0; i < mapping_count; ++i) {
    hal_mappings[i] = (iree_hal_buffer_mapping_transition_t){
        .mapping = &mappings[i].buffer->mapping,
        .offset = mappings[i].offset,
        .length = mappings[i].length,
        .recipe =
            (const iree_hal_memory_transition_recipe_t*)mappings[i].recipe,
    };
  }
  iree_status_t status = iree_hal_buffer_mapping_memory_barrier(
      (iree_hal_memory_effects_t){effects.bits}, mapping_count, hal_mappings);
  if (hal_mappings != inline_mappings) {
    iree_allocator_free(iree_allocator_system(), hal_mappings);
  }
  return hrx_status_from_iree(status);
}
