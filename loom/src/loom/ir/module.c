// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ir/structural_hash.h"
#include "loom/ir/type_identity.h"
#include "loom/ir/type_import.h"
#include "loom/util/adaptive_sort.h"

//===----------------------------------------------------------------------===//
// Hash function
//===----------------------------------------------------------------------===//

// Extends an FNV-1a hash with one byte span.
static uint32_t loom_hash_bytes_extend(uint32_t hash, const void* data,
                                       iree_host_size_t length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

// FNV-1a hash over a byte array.
static uint32_t loom_hash_bytes(const void* data, iree_host_size_t length) {
  return loom_hash_bytes_extend(2166136261u, data, length);
}

static uint32_t loom_hash_string_view(iree_string_view_t string) {
  return loom_hash_bytes(string.data, string.size);
}

//===----------------------------------------------------------------------===//
// Growable table helpers
//===----------------------------------------------------------------------===//

static void loom_type_use_heads_initialize(loom_value_type_use_heads_t* heads,
                                           iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    heads[i].provider = 0;
    heads[i].carrier = 0;
  }
}

static void loom_value_segment_initialize(loom_value_segment_t* segment) {
  memset(segment, 0, sizeof(*segment));
  memset(segment->u32_scratch, 0xFF, sizeof(segment->u32_scratch));
  loom_type_use_heads_initialize(segment->type_use_heads,
                                 LOOM_VALUE_SEGMENT_CAPACITY);
}

static iree_status_t loom_value_table_ensure_capacity(loom_module_t* module) {
  loom_value_table_t* table = &module->values;
  if (table->count < loom_value_table_capacity(table)) {
    return iree_ok_status();
  }
  loom_value_segment_t* segment = NULL;
  IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
      &table->segments, &module->arena, (void**)&segment));
  loom_value_segment_initialize(segment);
  return iree_ok_status();
}

static iree_status_t loom_value_table_reserve(loom_module_t* module,
                                              iree_host_size_t count) {
  loom_value_table_t* table = &module->values;
  iree_host_size_t available_capacity =
      loom_value_table_capacity(table) - table->count;
  while (available_capacity < count) {
    loom_value_segment_t* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &table->segments, &module->arena, (void**)&segment));
    loom_value_segment_initialize(segment);
    available_capacity += LOOM_VALUE_SEGMENT_CAPACITY;
  }
  return iree_ok_status();
}

static void loom_value_u32_scratch_fill(loom_value_u32_scratch_t* scratch,
                                        iree_host_size_t value_count,
                                        int byte_value) {
  IREE_ASSERT(value_count <= scratch->value_table->count);
  iree_host_size_t remaining_count = value_count;
  for (uint32_t segment_index = 0; remaining_count > 0; ++segment_index) {
    loom_value_segment_t* segment =
        (loom_value_segment_t*)loom_segmented_storage_segment(
            &scratch->value_table->segments, segment_index);
    const iree_host_size_t segment_value_count = iree_min(
        remaining_count, (iree_host_size_t)LOOM_VALUE_SEGMENT_CAPACITY);
    memset(segment->u32_scratch, byte_value,
           segment_value_count * sizeof(segment->u32_scratch[0]));
    remaining_count -= segment_value_count;
  }
}

static iree_status_t loom_string_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_string_table_t* table) {
  if (table->count < loom_string_table_capacity(table)) {
    return iree_ok_status();
  }
  void* segment = NULL;
  return loom_segmented_storage_append(&table->segments, arena, &segment);
}

// Stages row storage before the last fallible bucket reserve. Successful
// rehash may move existing buckets; only infallible publication follows it.
static iree_status_t loom_module_reserve_type_insert(
    loom_module_t* module, uint32_t hash, iree_host_size_t* inout_slot) {
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(&module->arena);
  loom_type_table_t types = module->types;
  iree_status_t status = iree_ok_status();
  if (types.count == loom_type_table_capacity(&types)) {
    void* segment = NULL;
    status = loom_segmented_storage_append(&types.segments, &module->arena,
                                           &segment);
  }
  if (iree_status_is_ok(status)) {
    status = loom_intern_table_reserve_insert(
        &module->arena, &module->type_intern, hash, inout_slot);
  }
  if (iree_status_is_ok(status)) {
    module->types = types;
  } else {
    iree_arena_checkpoint_restore(&checkpoint);
  }
  return status;
}

static iree_status_t loom_encoding_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_encoding_table_t* table) {
  if (table->count < table->capacity) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, table->count, /*minimum_capacity=*/8, sizeof(loom_encoding_t),
      &table->capacity, (void**)&table->entries));
  return iree_ok_status();
}

static iree_status_t loom_symbol_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_symbol_table_t* table) {
  if (table->count < table->capacity) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, table->count, /*minimum_capacity=*/8, sizeof(loom_symbol_t),
      &table->capacity, (void**)&table->entries));
  return iree_ok_status();
}

static iree_status_t loom_comment_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_comment_table_t* table) {
  if (table->count < table->capacity) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(arena, table->count, /*minimum_capacity=*/16,
                            sizeof(loom_comment_attachment_t), &table->capacity,
                            (void**)&table->entries));
  return iree_ok_status();
}

static iree_status_t loom_module_initialize_block(loom_module_t* module,
                                                  loom_block_t* block) {
  (void)module;
  memset(block, 0, sizeof(*block));
  block->label_id = LOOM_STRING_ID_INVALID;
  block->region_index = LOOM_BLOCK_REGION_INDEX_INVALID;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Region semantic summaries
//===----------------------------------------------------------------------===//

static void loom_region_adjust_summary_count(uint32_t* count, int32_t delta) {
  if (delta < 0) {
    uint32_t decrement = (uint32_t)(-delta);
    IREE_ASSERT(*count >= decrement);
    *count -= decrement;
  } else {
    *count += (uint32_t)delta;
  }
}

static void loom_region_adjust_summary_counts(loom_region_t* region,
                                              int32_t read_delta,
                                              int32_t write_delta,
                                              int32_t convergent_delta,
                                              int32_t observable_delta,
                                              int32_t memory_access_delta) {
  if (read_delta == 0 && write_delta == 0 && convergent_delta == 0 &&
      observable_delta == 0 && memory_access_delta == 0) {
    return;
  }
  if (read_delta != 0) {
    loom_region_adjust_summary_count(&region->read_effect_count, read_delta);
  }
  if (write_delta != 0) {
    loom_region_adjust_summary_count(&region->write_effect_count, write_delta);
  }
  if (convergent_delta != 0) {
    loom_region_adjust_summary_count(&region->convergent_effect_count,
                                     convergent_delta);
  }
  if (observable_delta != 0) {
    loom_region_adjust_summary_count(&region->observable_effect_count,
                                     observable_delta);
  }
  if (memory_access_delta != 0) {
    loom_region_adjust_summary_count(&region->memory_access_count,
                                     memory_access_delta);
  }
}

static void loom_module_adjust_op_ancestor_summary_counts(
    loom_op_t* op, int32_t read_delta, int32_t write_delta,
    int32_t convergent_delta, int32_t observable_delta,
    int32_t memory_access_delta) {
  if (read_delta == 0 && write_delta == 0 && convergent_delta == 0 &&
      observable_delta == 0 && memory_access_delta == 0) {
    return;
  }
  loom_region_t* region =
      op->parent_block ? op->parent_block->parent_region : NULL;
  loom_op_t* parent_op = op->parent_op;
  while (region) {
    loom_region_adjust_summary_counts(region, read_delta, write_delta,
                                      convergent_delta, observable_delta,
                                      memory_access_delta);
    if (!parent_op) {
      break;
    }
    region =
        parent_op->parent_block ? parent_op->parent_block->parent_region : NULL;
    parent_op = parent_op->parent_op;
  }
}

// A nonempty child region contributes one hint source to its containing
// region. Once a region remains nonempty (or remains empty), no ancestor's
// presence changes. Monotone subtree construction/removal therefore propagates
// at most one transition per region, independent of the number of nested hints.
static void loom_module_adjust_op_hint_sources(loom_op_t* op, int32_t delta) {
  loom_region_t* region =
      op->parent_block ? op->parent_block->parent_region : NULL;
  loom_op_t* parent_op = op->parent_op;
  while (region) {
    const bool had_hints = region->hint_source_count != 0;
    loom_region_adjust_summary_count(&region->hint_source_count, delta);
    if ((region->hint_source_count != 0) == had_hints || !parent_op) {
      break;
    }
    region =
        parent_op->parent_block ? parent_op->parent_block->parent_region : NULL;
    parent_op = parent_op->parent_op;
  }
}

static void loom_module_adjust_poison_op_count(loom_module_t* module,
                                               int32_t delta) {
  if (delta < 0) {
    IREE_ASSERT_GT(module->poison_op_count, 0u);
    --module->poison_op_count;
  } else if (delta > 0) {
    ++module->poison_op_count;
  }
}

static void loom_module_adjust_op_direct_summaries(loom_module_t* module,
                                                   loom_op_t* op,
                                                   loom_trait_flags_t traits,
                                                   int32_t direction) {
  int32_t read_delta = loom_traits_may_read(traits) ? direction : 0;
  int32_t write_delta = loom_traits_may_write(traits) ? direction : 0;
  int32_t convergent_delta = loom_traits_are_convergent(traits) ? direction : 0;
  int32_t observable_delta =
      loom_traits_have_observable_effects(traits) ? direction : 0;
  int32_t memory_access_delta =
      loom_traits_may_access_memory(traits) ? direction : 0;
  loom_module_adjust_op_ancestor_summary_counts(
      op, read_delta, write_delta, convergent_delta, observable_delta,
      memory_access_delta);
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT)) {
    loom_module_adjust_op_hint_sources(op, direction);
  }
  if (loom_traits_are_poison(traits)) {
    loom_module_adjust_poison_op_count(module, direction);
  }
}

void loom_module_record_op_summaries(loom_module_t* module, loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD |
                                             LOOM_OP_FLAG_SUMMARIES_COUNTED)) {
    return;
  }
  loom_module_adjust_op_direct_summaries(module, op, op->traits, +1);
  op->flags |= LOOM_OP_FLAG_SUMMARIES_COUNTED;
}

void loom_module_drop_op_summaries(loom_module_t* module, loom_op_t* op) {
  if (!op) {
    return;
  }
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_SUMMARIES_COUNTED)) {
    loom_module_adjust_op_direct_summaries(module, op, op->traits, -1);
    op->flags &= ~LOOM_OP_FLAG_SUMMARIES_COUNTED;
  }
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) {
      continue;
    }
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        loom_module_drop_op_summaries(module, child_op);
      }
    }
  }
}

void loom_module_update_op_direct_summaries(loom_module_t* module,
                                            loom_op_t* op,
                                            loom_trait_flags_t old_traits,
                                            loom_trait_flags_t new_traits) {
  if (!op || !iree_all_bits_set(op->flags, LOOM_OP_FLAG_SUMMARIES_COUNTED)) {
    return;
  }
  int32_t old_read = loom_traits_may_read(old_traits) ? 1 : 0;
  int32_t old_write = loom_traits_may_write(old_traits) ? 1 : 0;
  int32_t old_convergent = loom_traits_are_convergent(old_traits) ? 1 : 0;
  int32_t old_observable =
      loom_traits_have_observable_effects(old_traits) ? 1 : 0;
  int32_t new_read = loom_traits_may_read(new_traits) ? 1 : 0;
  int32_t new_write = loom_traits_may_write(new_traits) ? 1 : 0;
  int32_t new_convergent = loom_traits_are_convergent(new_traits) ? 1 : 0;
  int32_t new_observable =
      loom_traits_have_observable_effects(new_traits) ? 1 : 0;
  int32_t old_memory_access = loom_traits_may_access_memory(old_traits) ? 1 : 0;
  int32_t new_memory_access = loom_traits_may_access_memory(new_traits) ? 1 : 0;
  loom_module_adjust_op_ancestor_summary_counts(
      op, new_read - old_read, new_write - old_write,
      new_convergent - old_convergent, new_observable - old_observable,
      new_memory_access - old_memory_access);
  if (iree_any_bit_set(old_traits ^ new_traits, LOOM_TRAIT_HINT)) {
    loom_module_adjust_op_hint_sources(
        op, iree_any_bit_set(new_traits, LOOM_TRAIT_HINT) ? 1 : -1);
  }
  const int32_t poison_delta = (loom_traits_are_poison(new_traits) ? 1 : 0) -
                               (loom_traits_are_poison(old_traits) ? 1 : 0);
  loom_module_adjust_poison_op_count(module, poison_delta);
}

iree_status_t loom_region_reserve_block_capacity(
    loom_module_t* module, loom_region_t* region,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= region->block_capacity) {
    return iree_ok_status();
  }
  if (minimum_capacity > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "region block count exceeds UINT16_MAX");
  }

  iree_host_size_t old_capacity = region->block_capacity;
  iree_host_size_t new_capacity = old_capacity > 0 ? old_capacity : 4;
  while (new_capacity < minimum_capacity) {
    new_capacity = iree_min(new_capacity * 2, (iree_host_size_t)UINT16_MAX);
  }

  loom_block_t** new_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, new_capacity,
                                                 sizeof(loom_block_t*),
                                                 (void**)&new_blocks));
  memset(new_blocks, 0, new_capacity * sizeof(loom_block_t*));
  if (region->block_count > 0) {
    memcpy(new_blocks, region->blocks,
           (iree_host_size_t)region->block_count * sizeof(loom_block_t*));
  }

  region->blocks = new_blocks;
  region->block_capacity = (uint16_t)new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_region_blocks_ensure_capacity(loom_module_t* module,
                                                        loom_region_t* region) {
  if (region->block_count == UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "region block count exceeds UINT16_MAX");
  }
  return loom_region_reserve_block_capacity(
      module, region, (iree_host_size_t)region->block_count + 1);
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

static iree_status_t loom_module_initialize_tables(
    loom_module_t* module, const loom_module_size_hints_t* hints) {
  iree_arena_allocator_t* arena = &module->arena;

  iree_host_size_t string_capacity = 512;
  iree_host_size_t type_capacity = 64;
  iree_host_size_t encoding_capacity = 0;
  iree_host_size_t source_capacity = 0;
  iree_host_size_t symbol_capacity = 32;

  if (hints) {
    string_capacity =
        (iree_host_size_t)(hints->string_count * LOOM_MODULE_GROWTH_FACTOR);
    type_capacity =
        (iree_host_size_t)(hints->type_count * LOOM_MODULE_GROWTH_FACTOR);
    encoding_capacity =
        (iree_host_size_t)(hints->encoding_count * LOOM_MODULE_GROWTH_FACTOR);
    source_capacity =
        (iree_host_size_t)(hints->source_count * LOOM_MODULE_GROWTH_FACTOR);
    symbol_capacity =
        (iree_host_size_t)(hints->symbol_count * LOOM_MODULE_GROWTH_FACTOR);
    if (string_capacity < 8) {
      string_capacity = 8;
    }
    if (type_capacity < 8) {
      type_capacity = 8;
    }
    if (hints->encoding_count > 0 && encoding_capacity < 8) {
      encoding_capacity = 8;
    }
    if (hints->source_count > 0 && source_capacity < 4) {
      source_capacity = 4;
    }
    if (symbol_capacity < 4) {
      symbol_capacity = 4;
    }
  }

  // Value segments are allocated lazily as values are defined. The matching
  // scratch and type-use head tables share each segment.
  loom_segmented_storage_initialize(sizeof(loom_value_segment_t),
                                    iree_alignof(loom_value_segment_t),
                                    &module->values.segments);
  module->scratch.values.value_table = &module->values;
  module->scratch.values.state =
      LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ORDINALS;
  module->type_uses.value_table = &module->values;
  iree_arena_initialize(module->arena.block_pool, &module->type_uses.arena);

  // Locations. Rows are allocated lazily in stable segments.
  loom_segmented_storage_initialize(sizeof(loom_location_segment_t),
                                    iree_alignof(loom_location_segment_t),
                                    &module->locations.segments);

  // String views are lazy; hints size only the interner's buckets.
  loom_segmented_storage_initialize(sizeof(loom_string_segment_t),
                                    iree_alignof(loom_string_segment_t),
                                    &module->strings.segments);

  // Canonical rows are lazy; type hints size only the interner's buckets.
  loom_segmented_storage_initialize(sizeof(loom_type_segment_t),
                                    iree_alignof(loom_type_segment_t),
                                    &module->types.segments);

  // Encodings. Modules without an encoding count hint retain lazy allocation.
  if (encoding_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, encoding_capacity, sizeof(loom_encoding_t),
        (void**)&module->encodings.entries));
    module->encodings.capacity = encoding_capacity;
  }

  // Sources. Modules without a source count hint retain lazy allocation.
  if (source_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, source_capacity, sizeof(iree_string_view_t),
        (void**)&module->sources.entries));
    module->sources.capacity = source_capacity;
  }

  // Symbols.
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, symbol_capacity, sizeof(loom_symbol_t),
                                (void**)&module->symbols.entries));
  module->symbols.capacity = symbol_capacity;
  memset(module->symbols.entries, 0, symbol_capacity * sizeof(loom_symbol_t));

  // Intern tables sized to match entry capacity.
  iree_host_size_t string_intern_capacity =
      loom_intern_table_capacity_for_entries(string_capacity);
  IREE_RETURN_IF_ERROR(loom_intern_table_initialize(
      arena, string_intern_capacity, &module->string_intern));
  iree_host_size_t type_intern_capacity =
      loom_intern_table_capacity_for_entries(type_capacity);
  IREE_RETURN_IF_ERROR(loom_intern_table_initialize(arena, type_intern_capacity,
                                                    &module->type_intern));
  iree_host_size_t encoding_intern_capacity = 0;
  if (encoding_capacity > 0) {
    encoding_intern_capacity =
        loom_intern_table_capacity_for_entries(encoding_capacity);
  }
  return loom_intern_table_initialize(arena, encoding_intern_capacity,
                                      &module->encoding_intern);
}

iree_status_t loom_module_allocate(loom_context_t* context,
                                   iree_string_view_t name,
                                   iree_arena_block_pool_t* block_pool,
                                   const loom_module_size_hints_t* hints,
                                   iree_allocator_t allocator,
                                   loom_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_module = NULL;

  loom_module_t* module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(loom_module_t), (void**)&module));
  memset(module, 0, sizeof(loom_module_t));

  module->context = context;
  module->allocator = allocator;
  iree_arena_initialize(block_pool, &module->arena);

  iree_status_t status = loom_module_initialize_tables(module, hints);
  if (iree_status_is_ok(status)) {
    status = loom_module_intern_string(module, name, &module->name_id);
  }
  if (iree_status_is_ok(status)) {
    status = loom_module_allocate_region(module, 1, &module->body);
  }
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    loom_module_free(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void loom_module_free(loom_module_t* module) {
  if (!module) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = module->allocator;
  iree_arena_deinitialize(&module->type_uses.arena);
  iree_arena_deinitialize(&module->arena);
  iree_allocator_free(allocator, module);
  IREE_TRACE_ZONE_END(z0);
}

void loom_module_value_ordinal_scratch_acquire(loom_module_t* module) {
  loom_value_u32_scratch_t* scratch = &module->scratch.values;
  IREE_ASSERT(scratch->state != LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ORDINALS);
  IREE_ASSERT(scratch->state != LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ZEROED);
  if (scratch->state != LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ORDINALS) {
    loom_value_u32_scratch_fill(scratch, module->values.count, 0xFF);
  }
  scratch->state = LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ORDINALS;
}

void loom_module_value_ordinal_scratch_release(loom_module_t* module) {
  loom_value_u32_scratch_t* scratch = &module->scratch.values;
  IREE_ASSERT_EQ(scratch->state,
                 LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ORDINALS);
  scratch->state = LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ORDINALS;
}

void loom_value_u32_scratch_acquire_zeroed(loom_value_u32_scratch_t* scratch,
                                           iree_host_size_t value_count) {
  IREE_ASSERT(scratch != NULL);
  IREE_ASSERT(value_count <= scratch->value_table->count);
  IREE_ASSERT(scratch->state != LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ORDINALS);
  IREE_ASSERT(scratch->state != LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ZEROED);
  loom_value_u32_scratch_fill(scratch, value_count, 0);
  scratch->state = LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ZEROED;
}

void loom_value_u32_scratch_release_zeroed(loom_value_u32_scratch_t* scratch) {
  IREE_ASSERT(scratch != NULL);
  IREE_ASSERT_EQ(scratch->state, LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ZEROED);
  scratch->state = LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ZEROED;
}

//===----------------------------------------------------------------------===//
// Encoding table
//===----------------------------------------------------------------------===//

typedef struct loom_encoding_equal_context_t {
  const loom_module_t* module;
  const loom_encoding_t* encoding;
} loom_encoding_equal_context_t;

static bool loom_encoding_equal_fn(const void* context, uint32_t index) {
  const loom_encoding_equal_context_t* equal_context =
      (const loom_encoding_equal_context_t*)context;
  return loom_encoding_equal(&equal_context->module->encodings.entries[index],
                             equal_context->encoding);
}

// Binds freshly canonicalized sparse parameters to their generated descriptor
// ordinals. Both arrays are lexically ordered, so one linear merge establishes
// all metadata and records whether authored names and kinds satisfy the schema.
// Malformed semantic IR is retained for the verifier to diagnose.
static loom_encoding_family_flags_t loom_module_bind_encoding_parameters(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* family_descriptor,
    loom_named_attr_t* parameters, uint8_t parameter_count) {
  bool all_parameters_valid = true;
  uint8_t descriptor_index = 0;
  for (uint8_t parameter_index = 0; parameter_index < parameter_count;
       ++parameter_index) {
    loom_named_attr_t* parameter = &parameters[parameter_index];
    const iree_string_view_t parameter_name =
        loom_string_table_get(&module->strings, parameter->name_id);
    while (descriptor_index < family_descriptor->parameter_count) {
      const iree_string_view_t descriptor_name = loom_attr_descriptor_name(
          &family_descriptor->parameter_descriptors[descriptor_index]);
      const int comparison =
          iree_string_view_compare(parameter_name, descriptor_name);
      if (comparison <= 0) {
        if (comparison == 0) {
          loom_encoding_parameter_bind_descriptor(parameter, descriptor_index);
          const loom_attr_descriptor_t* descriptor =
              &family_descriptor->parameter_descriptors[descriptor_index];
          bool parameter_valid = loom_attr_descriptor_accepts_kind(
              descriptor, (loom_attr_kind_t)parameter->value.kind);
          if (parameter_valid && parameter->value.kind == LOOM_ATTR_ENUM &&
              !iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
            parameter_valid = loom_attr_descriptor_has_enum_case(
                descriptor, loom_attr_as_enum(parameter->value));
          }
          all_parameters_valid &= parameter_valid;
        } else {
          all_parameters_valid = false;
        }
        break;
      }
      ++descriptor_index;
    }
    if (descriptor_index == family_descriptor->parameter_count) {
      all_parameters_valid = false;
    }
  }
  return all_parameters_valid ? LOOM_ENCODING_FAMILY_STATIC_PARAMETERS_VALID
                              : 0;
}

iree_status_t loom_module_add_encoding(loom_module_t* module,
                                       const loom_encoding_t* encoding,
                                       uint16_t* out_encoding_id) {
  if (encoding->name_id == LOOM_STRING_ID_INVALID ||
      encoding->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding family string id %u is out of range "
                            "(module has %" PRIhsz " strings)",
                            encoding->name_id, module->strings.count);
  }
  if (encoding->alias_id != LOOM_STRING_ID_INVALID &&
      encoding->alias_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding alias string id %u is out of range "
                            "(module has %" PRIhsz " strings)",
                            encoding->alias_id, module->strings.count);
  }
  if (encoding->attribute_count > 0 && !encoding->attributes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty encoding parameter list has a NULL entry pointer");
  }

  iree_string_view_t encoding_name =
      loom_string_table_get(&module->strings, encoding->name_id);
  const loom_encoding_name_resolution_t name_resolution =
      loom_context_resolve_encoding_name(module->context, encoding_name);
  const loom_encoding_vtable_t* vtable = loom_context_resolve_encoding_vtable(
      module->context, name_resolution.family_id);
  if (!vtable && module->context->encodings.vtables.count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown encoding family '%.*s'",
                            (int)encoding_name.size, encoding_name.data);
  }

  loom_attribute_t canonical_attr_dict = {0};
  loom_string_id_t canonical_name_id = encoding->name_id;
  if (name_resolution.alias) {
    const loom_encoding_alias_descriptor_t* alias = name_resolution.alias;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, loom_bstring_view(vtable->descriptor->name),
        &canonical_name_id));

    loom_named_attr_t* alias_entries = NULL;
    if (alias->parameter_count > 0) {
      alias_entries = (loom_named_attr_t*)iree_alloca(alias->parameter_count *
                                                      sizeof(*alias_entries));
    }
    for (uint8_t i = 0; i < alias->parameter_count; ++i) {
      const loom_encoding_alias_parameter_t* parameter = &alias->parameters[i];
      const loom_attr_descriptor_t* parameter_descriptor =
          &vtable->descriptor
               ->parameter_descriptors[parameter->parameter_index];
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          module, loom_attr_descriptor_name(parameter_descriptor),
          &alias_entries[i].name_id));
      alias_entries[i].reserved = 0;
      alias_entries[i].value = parameter->value;
    }

    loom_named_attr_update_t* authored_updates = NULL;
    if (encoding->attribute_count > 0) {
      authored_updates = (loom_named_attr_update_t*)iree_alloca(
          encoding->attribute_count * sizeof(*authored_updates));
    }
    for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
      for (uint8_t alias_parameter_index = 0;
           alias_parameter_index < alias->parameter_count;
           ++alias_parameter_index) {
        const loom_encoding_alias_parameter_t* alias_parameter =
            &alias->parameters[alias_parameter_index];
        if (!iree_any_bit_set(alias_parameter->flags,
                              LOOM_ENCODING_ALIAS_PARAMETER_FIXED)) {
          continue;
        }
        if (encoding->attributes[i].name_id !=
            alias_entries[alias_parameter_index].name_id) {
          continue;
        }
        const iree_string_view_t parameter_name = loom_string_table_get(
            &module->strings, alias_entries[alias_parameter_index].name_id);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "encoding alias '%.*s' fixes parameter '%.*s'; the parameter "
            "cannot be restated",
            (int)encoding_name.size, encoding_name.data,
            (int)parameter_name.size, parameter_name.data);
      }
      authored_updates[i] = loom_named_attr_replace(
          encoding->attributes[i].name_id, encoding->attributes[i].value);
    }
    IREE_RETURN_IF_ERROR(loom_module_replace_canonical_attr_dict(
        module,
        loom_make_named_attr_slice(alias_entries, alias->parameter_count),
        (loom_named_attr_update_slice_t){
            .updates = authored_updates,
            .count = encoding->attribute_count,
        },
        &canonical_attr_dict));
  } else {
    IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
        module,
        loom_make_named_attr_slice(encoding->attributes,
                                   encoding->attribute_count),
        &canonical_attr_dict));
  }
  if (canonical_attr_dict.count > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "encoding '%.*s' has %u parameters, max %u",
                            (int)encoding_name.size, encoding_name.data,
                            (unsigned)canonical_attr_dict.count,
                            (unsigned)UINT8_MAX);
  }

  loom_encoding_t canonical_encoding = {
      .name_id = canonical_name_id,
      .alias_id = encoding->alias_id,
      .attribute_count = canonical_attr_dict.count,
      .family.id = name_resolution.family_id,
      .family.flags = vtable ? vtable->descriptor->family_flags : 0,
      .attributes = canonical_attr_dict.dict_entries,
  };

  if (vtable && canonical_encoding.attribute_count > 0) {
    // Canonical dictionary entries are freshly allocated by this function and
    // have not yet been published through their const module-owned view.
    canonical_encoding.family.flags |= loom_module_bind_encoding_parameters(
        module, vtable->descriptor,
        (loom_named_attr_t*)canonical_encoding.attributes,
        canonical_encoding.attribute_count);
  } else if (vtable) {
    canonical_encoding.family.flags |=
        LOOM_ENCODING_FAMILY_STATIC_PARAMETERS_VALID;
  }
  const uint32_t hash = loom_encoding_hash(&canonical_encoding);
  const loom_encoding_equal_context_t equal_context = {
      .module = module,
      .encoding = &canonical_encoding,
  };
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &module->encoding_intern, hash, loom_encoding_equal_fn, &equal_context);
  const uint32_t existing_index = probe.index;

  // Alias names are rare file-local shorthand. Scan only when one is authored
  // and reject collisions against any structurally different encoding.
  if (canonical_encoding.alias_id != LOOM_STRING_ID_INVALID) {
    for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
      if (i == existing_index) {
        continue;
      }
      if (module->encodings.entries[i].alias_id !=
          canonical_encoding.alias_id) {
        continue;
      }
      iree_string_view_t alias_name =
          loom_string_table_get(&module->strings, canonical_encoding.alias_id);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding alias '%.*s' already names a different encoding",
          (int)alias_name.size, alias_name.data);
    }
  }

  if (existing_index != UINT32_MAX) {
    if (module->encodings.entries[existing_index].alias_id ==
            LOOM_STRING_ID_INVALID &&
        canonical_encoding.alias_id != LOOM_STRING_ID_INVALID) {
      module->encodings.entries[existing_index].alias_id =
          canonical_encoding.alias_id;
    }
    *out_encoding_id = (uint16_t)(existing_index + 1);
    return iree_ok_status();
  }

  // Family semantics are an invariant of structural encoding identity. Only
  // classify a new canonical entry; repeated references reuse the interned
  // entry's established family facts.
  if (vtable &&
      loom_encoding_static_parameters_are_valid(&canonical_encoding) &&
      (!vtable->is_static_valid ||
       vtable->is_static_valid(module, &canonical_encoding))) {
    canonical_encoding.family.flags |=
        LOOM_ENCODING_FAMILY_STATIC_SEMANTICS_VALID;
  }

  // Encoding IDs are 1-based uint16_t. ID 0 means "no encoding" and
  // UINT16_MAX is the maximum representable ID, so we can store at
  // most UINT16_MAX entries (IDs 1 through UINT16_MAX).
  if (module->encodings.count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "encoding table full (%" PRIhsz " entries, max %u)",
                            module->encodings.count, (unsigned)UINT16_MAX);
  }

  IREE_RETURN_IF_ERROR(
      loom_encoding_table_ensure_capacity(&module->arena, &module->encodings));
  iree_host_size_t slot = probe.slot;
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      &module->arena, &module->encoding_intern, hash, &slot));

  const uint32_t new_index = (uint32_t)module->encodings.count;
  loom_encoding_t* entry = &module->encodings.entries[new_index];
  *entry = canonical_encoding;
  loom_intern_table_insert(&module->encoding_intern, slot, hash, new_index);

  *out_encoding_id = (uint16_t)(new_index + 1);
  ++module->encodings.count;
  return iree_ok_status();
}

const loom_encoding_vtable_t* loom_module_encoding_vtable(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  return loom_context_resolve_encoding_vtable(module->context,
                                              encoding->family.id);
}

//===----------------------------------------------------------------------===//
// Symbol table
//===----------------------------------------------------------------------===//

uint16_t loom_module_find_symbol(const loom_module_t* module,
                                 loom_string_id_t name_id) {
  // Linear scan — suitable for diagnostics and ad-hoc queries. For
  // bulk lookups during parsing, use loom_symbol_map_t instead.
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (module->symbols.entries[i].name_id == name_id) {
      return (uint16_t)i;
    }
  }
  return LOOM_SYMBOL_ID_INVALID;
}

iree_status_t loom_module_add_symbol(loom_module_t* module,
                                     loom_string_id_t name_id,
                                     uint16_t* out_symbol_id) {
  // Symbol IDs are 0-based uint16_t. LOOM_SYMBOL_ID_INVALID is the null
  // sentinel (loom_symbol_ref_null), so the maximum valid ID is
  // LOOM_SYMBOL_ID_INVALID - 1, giving a hard cap of 65535 symbols per module.
  if (module->symbols.count >= LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol table full (%" PRIhsz " entries, max %u)",
                            module->symbols.count,
                            (unsigned)(LOOM_SYMBOL_ID_INVALID - 1));
  }

  IREE_RETURN_IF_ERROR(
      loom_symbol_table_ensure_capacity(&module->arena, &module->symbols));

  uint16_t symbol_id = (uint16_t)module->symbols.count;
  loom_symbol_t* symbol = &module->symbols.entries[module->symbols.count++];
  memset(symbol, 0, sizeof(*symbol));
  symbol->name_id = name_id;

  *out_symbol_id = symbol_id;
  return iree_ok_status();
}

static iree_status_t loom_module_mark_symbol_references_in_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    uint8_t* referenced_symbols, uint8_t dict_depth) {
  if (!attr) {
    return iree_ok_status();
  }
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t symbol_ref = loom_attr_as_symbol(*attr);
      if (loom_symbol_ref_is_valid(symbol_ref) && symbol_ref.module_id == 0 &&
          symbol_ref.symbol_id < module->symbols.count) {
        referenced_symbols[symbol_ref.symbol_id] = 1;
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      for (uint16_t i = 0; i < attr->count; ++i) {
        loom_symbol_ref_t symbol_ref = attr->symbol_refs[i];
        if (loom_symbol_ref_is_valid(symbol_ref) && symbol_ref.module_id == 0 &&
            symbol_ref.symbol_id < module->symbols.count) {
          referenced_symbols[symbol_ref.symbol_id] = 1;
        }
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_DICT:
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dict attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if (attr->count > 0 && !attr->dict_entries) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute has a NULL entry pointer");
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_attr(
            module, &attr->dict_entries[i].value, referenced_symbols,
            (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    case LOOM_ATTR_PARAMETERIZED:
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if (attr->count > 0 && !attr->parameterized_slots) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty parameterized attribute has a NULL slot pointer");
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_attr(
            module, &attr->parameterized_slots[i], referenced_symbols,
            (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if (attr->count > 0 && !attr->parameterized_array) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty parameterized attribute array has a NULL value "
            "pointer");
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_attr(
            module, &attr->parameterized_array[i], referenced_symbols,
            (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_module_mark_symbol_references_in_attrs(
    const loom_module_t* module, const loom_attribute_t* attrs,
    iree_host_size_t attr_count, uint8_t* referenced_symbols) {
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_attr(
        module, &attrs[i], referenced_symbols, 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_module_mark_symbol_references_in_named_attrs(
    const loom_module_t* module, const loom_named_attr_t* attrs,
    iree_host_size_t attr_count, uint8_t* referenced_symbols) {
  if (attr_count > 0 && !attrs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty named attribute list has a NULL entry pointer");
  }
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_attr(
        module, &attrs[i].value, referenced_symbols, 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_module_mark_symbol_references_in_types(
    const loom_module_t* module, uint8_t* referenced_symbols) {
  for (iree_host_size_t i = 0; i < module->types.count; ++i) {
    loom_type_t type = loom_type_table_get(&module->types, i);
    if (!loom_type_is_parameterized(type)) {
      continue;
    }
    const loom_attribute_t* parameters =
        loom_type_parameterized_parameters(type);
    const uint8_t parameter_count =
        loom_type_parameterized_parameter_count(type);
    IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_attrs(
        module, parameters, parameter_count, referenced_symbols));
  }
  return iree_ok_status();
}

static iree_status_t loom_module_mark_symbol_references_in_region(
    const loom_module_t* module, const loom_region_t* region,
    uint8_t* referenced_symbols) {
  if (!region) {
    return iree_ok_status();
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_attrs(
          module, loom_op_const_attrs(op), op->attribute_count,
          referenced_symbols));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_region(
            module, regions[i], referenced_symbols));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_module_remap_symbol_attr(
    loom_module_t* module, const uint16_t* new_symbol_ids,
    iree_host_size_t old_symbol_count, loom_attribute_t source_attr,
    uint8_t dict_depth, loom_attribute_t* out_target_attr, bool* out_changed) {
  *out_target_attr = source_attr;
  *out_changed = false;
  switch ((loom_attr_kind_t)source_attr.kind) {
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t symbol_ref = loom_attr_as_symbol(source_attr);
      if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
          symbol_ref.symbol_id >= old_symbol_count) {
        return iree_ok_status();
      }
      uint16_t new_symbol_id = new_symbol_ids[symbol_ref.symbol_id];
      if (new_symbol_id == LOOM_SYMBOL_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "symbol attribute references dropped symbol id %u",
            (unsigned)symbol_ref.symbol_id);
      }
      if (new_symbol_id != symbol_ref.symbol_id) {
        out_target_attr->symbol.symbol_id = new_symbol_id;
        *out_changed = true;
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      if (source_attr.count == 0) {
        return iree_ok_status();
      }
      loom_symbol_ref_t* target_refs = NULL;
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        loom_symbol_ref_t source_ref = source_attr.symbol_refs[i];
        if (!loom_symbol_ref_is_valid(source_ref) ||
            source_ref.module_id != 0 ||
            source_ref.symbol_id >= old_symbol_count) {
          continue;
        }
        uint16_t new_symbol_id = new_symbol_ids[source_ref.symbol_id];
        if (new_symbol_id == LOOM_SYMBOL_ID_INVALID) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "symbol array attribute element %u references dropped symbol "
              "id %u",
              (unsigned)i, (unsigned)source_ref.symbol_id);
        }
        if (new_symbol_id == source_ref.symbol_id) {
          continue;
        }
        if (!target_refs) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &module->arena, source_attr.count, sizeof(*target_refs),
              (void**)&target_refs));
          memcpy(target_refs, source_attr.symbol_refs,
                 (iree_host_size_t)source_attr.count * sizeof(*target_refs));
        }
        target_refs[i].symbol_id = new_symbol_id;
      }
      if (target_refs) {
        *out_target_attr =
            source_attr.kind == LOOM_ATTR_SYMBOL_SET
                ? loom_attr_symbol_set(target_refs, source_attr.count)
                : loom_attr_symbol_array(target_refs, source_attr.count);
        *out_changed = true;
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_DICT: {
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dict attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if (source_attr.count == 0) {
        return iree_ok_status();
      }
      if (!source_attr.dict_entries) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute has a NULL entry pointer");
      }
      loom_named_attr_t* target_entries = NULL;
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        loom_attribute_t target_value = {0};
        bool value_changed = false;
        IREE_RETURN_IF_ERROR(loom_module_remap_symbol_attr(
            module, new_symbol_ids, old_symbol_count,
            source_attr.dict_entries[i].value, (uint8_t)(dict_depth + 1),
            &target_value, &value_changed));
        if (!value_changed) {
          continue;
        }
        if (!target_entries) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &module->arena, source_attr.count, sizeof(*target_entries),
              (void**)&target_entries));
          memcpy(target_entries, source_attr.dict_entries,
                 (iree_host_size_t)source_attr.count * sizeof(*target_entries));
        }
        target_entries[i].value = target_value;
      }
      if (target_entries) {
        *out_target_attr =
            loom_make_canonical_attr_dict(target_entries, source_attr.count);
        *out_changed = true;
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_PARAMETERIZED: {
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if (source_attr.count == 0) {
        return iree_ok_status();
      }
      if (!source_attr.parameterized_slots) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty parameterized attribute has a NULL slot pointer");
      }
      loom_attribute_t* target_slots = NULL;
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        loom_attribute_t target_value = {0};
        bool value_changed = false;
        IREE_RETURN_IF_ERROR(loom_module_remap_symbol_attr(
            module, new_symbol_ids, old_symbol_count,
            source_attr.parameterized_slots[i], (uint8_t)(dict_depth + 1),
            &target_value, &value_changed));
        if (!value_changed) {
          continue;
        }
        if (!target_slots) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &module->arena, source_attr.count, sizeof(*target_slots),
              (void**)&target_slots));
          memcpy(target_slots, source_attr.parameterized_slots,
                 (iree_host_size_t)source_attr.count * sizeof(*target_slots));
        }
        target_slots[i] = target_value;
      }
      if (target_slots) {
        *out_target_attr = loom_make_parameterized_attr(
            (loom_parameterized_attr_kind_t)source_attr.reserved_1,
            target_slots, source_attr.count);
        *out_changed = true;
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (dict_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if (source_attr.count == 0) {
        return iree_ok_status();
      }
      if (!source_attr.parameterized_array) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty parameterized attribute array has a NULL value "
            "pointer");
      }
      loom_attribute_t* target_attributes = NULL;
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        loom_attribute_t target_value = {0};
        bool value_changed = false;
        IREE_RETURN_IF_ERROR(loom_module_remap_symbol_attr(
            module, new_symbol_ids, old_symbol_count,
            source_attr.parameterized_array[i], (uint8_t)(dict_depth + 1),
            &target_value, &value_changed));
        if (!value_changed) {
          continue;
        }
        if (!target_attributes) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &module->arena, source_attr.count, sizeof(*target_attributes),
              (void**)&target_attributes));
          memcpy(
              target_attributes, source_attr.parameterized_array,
              (iree_host_size_t)source_attr.count * sizeof(*target_attributes));
        }
        target_attributes[i] = target_value;
      }
      if (target_attributes) {
        *out_target_attr =
            loom_attr_parameterized_array(target_attributes, source_attr.count);
        *out_changed = true;
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_module_remap_symbol_attrs(
    loom_module_t* module, const uint16_t* new_symbol_ids,
    iree_host_size_t old_symbol_count, loom_attribute_t* attrs,
    iree_host_size_t attr_count) {
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    loom_attribute_t target_attr = {0};
    bool changed = false;
    IREE_RETURN_IF_ERROR(
        loom_module_remap_symbol_attr(module, new_symbol_ids, old_symbol_count,
                                      attrs[i], 0, &target_attr, &changed));
    if (changed) {
      attrs[i] = target_attr;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_module_remap_symbol_named_attrs(
    loom_module_t* module, const uint16_t* new_symbol_ids,
    iree_host_size_t old_symbol_count, const loom_named_attr_t** inout_attrs,
    iree_host_size_t attr_count) {
  if (attr_count == 0) {
    return iree_ok_status();
  }
  const loom_named_attr_t* source_attrs = *inout_attrs;
  if (!source_attrs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty named attribute list has a NULL entry pointer");
  }
  loom_named_attr_t* target_attrs = NULL;
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    loom_attribute_t target_value = {0};
    bool value_changed = false;
    IREE_RETURN_IF_ERROR(loom_module_remap_symbol_attr(
        module, new_symbol_ids, old_symbol_count, source_attrs[i].value, 0,
        &target_value, &value_changed));
    if (!value_changed) {
      continue;
    }
    if (!target_attrs) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, attr_count,
                                                     sizeof(*target_attrs),
                                                     (void**)&target_attrs));
      memcpy(target_attrs, source_attrs, attr_count * sizeof(*target_attrs));
    }
    target_attrs[i].value = target_value;
  }
  if (target_attrs) {
    *inout_attrs = target_attrs;
  }
  return iree_ok_status();
}

static iree_status_t loom_module_remap_symbol_region_attrs(
    loom_module_t* module, const uint16_t* new_symbol_ids,
    iree_host_size_t old_symbol_count, loom_region_t* region) {
  if (!region) {
    return iree_ok_status();
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_module_remap_symbol_attrs(
          module, new_symbol_ids, old_symbol_count, loom_op_attrs(op),
          op->attribute_count));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_remap_symbol_region_attrs(
            module, new_symbol_ids, old_symbol_count, regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static bool loom_module_symbol_has_payload(const loom_symbol_t* symbol) {
  return symbol->kind != LOOM_SYMBOL_NONE || symbol->definition ||
         symbol->defining_op || symbol->flags != 0;
}

static iree_host_size_t loom_module_max_preserved_symbol_id(
    const loom_symbol_ref_t* preserved_symbol_refs,
    iree_host_size_t preserved_symbol_ref_count,
    iree_host_size_t old_symbol_count) {
  iree_host_size_t max_preserved_symbol_id = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < preserved_symbol_ref_count; ++i) {
    loom_symbol_ref_t symbol_ref = preserved_symbol_refs[i];
    if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
        symbol_ref.symbol_id >= old_symbol_count) {
      continue;
    }
    if (max_preserved_symbol_id == IREE_HOST_SIZE_MAX ||
        symbol_ref.symbol_id > max_preserved_symbol_id) {
      max_preserved_symbol_id = symbol_ref.symbol_id;
    }
  }
  return max_preserved_symbol_id;
}

iree_status_t loom_module_compact_symbols_preserving_symbol_refs(
    loom_module_t* module, const loom_symbol_ref_t* preserved_symbol_refs,
    iree_host_size_t preserved_symbol_ref_count,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_removed_count) {
  if (out_removed_count) {
    *out_removed_count = 0;
  }
  const iree_host_size_t old_symbol_count = module->symbols.count;
  if (old_symbol_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t max_preserved_symbol_id =
      loom_module_max_preserved_symbol_id(
          preserved_symbol_refs, preserved_symbol_ref_count, old_symbol_count);

  uint8_t* referenced_symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, old_symbol_count, sizeof(*referenced_symbols),
      (void**)&referenced_symbols));
  memset(referenced_symbols, 0, old_symbol_count * sizeof(*referenced_symbols));

  IREE_RETURN_IF_ERROR(
      loom_module_mark_symbol_references_in_types(module, referenced_symbols));
  // Parameterized types are immutable entries in the module type interner. A
  // preserved ordinal prefix keeps their symbol slots valid without rebuilding
  // that interner during symbol compaction.
  for (iree_host_size_t i = 0; i < old_symbol_count; ++i) {
    if (referenced_symbols[i] == 0) {
      continue;
    }
    if (max_preserved_symbol_id == IREE_HOST_SIZE_MAX ||
        i > max_preserved_symbol_id) {
      max_preserved_symbol_id = i;
    }
  }

  IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_region(
      module, module->body, referenced_symbols));
  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    const loom_encoding_t* encoding = &module->encodings.entries[i];
    IREE_RETURN_IF_ERROR(loom_module_mark_symbol_references_in_named_attrs(
        module, encoding->attributes, encoding->attribute_count,
        referenced_symbols));
  }

  uint16_t* new_symbol_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, old_symbol_count, sizeof(*new_symbol_ids),
      (void**)&new_symbol_ids));
  memset(new_symbol_ids, 0xFF, old_symbol_count * sizeof(*new_symbol_ids));

  iree_host_size_t new_symbol_count = 0;
  iree_host_size_t removed_count = 0;
  for (iree_host_size_t i = 0; i < old_symbol_count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    const bool has_payload = loom_module_symbol_has_payload(symbol);
    const bool preserves_external_ref =
        max_preserved_symbol_id != IREE_HOST_SIZE_MAX &&
        i <= max_preserved_symbol_id;
    const bool keep =
        has_payload || referenced_symbols[i] != 0 || preserves_external_ref;
    if (keep) {
      new_symbol_ids[i] = (uint16_t)new_symbol_count++;
    } else {
      ++removed_count;
    }
  }

  if (removed_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_module_remap_symbol_region_attrs(
      module, new_symbol_ids, old_symbol_count, module->body));
  loom_intern_table_clear(&module->encoding_intern);
  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    loom_encoding_t* encoding = &module->encodings.entries[i];
    const loom_named_attr_t* attributes = encoding->attributes;
    IREE_RETURN_IF_ERROR(loom_module_remap_symbol_named_attrs(
        module, new_symbol_ids, old_symbol_count, &attributes,
        encoding->attribute_count));
    encoding->attributes = attributes;
    const uint32_t hash = loom_encoding_hash(encoding);
    const iree_host_size_t slot =
        loom_intern_table_find_empty_slot(&module->encoding_intern, hash);
    loom_intern_table_insert(&module->encoding_intern, slot, hash, (uint32_t)i);
  }

  for (iree_host_size_t old_index = 0; old_index < old_symbol_count;
       ++old_index) {
    uint16_t new_symbol_id = new_symbol_ids[old_index];
    if (new_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    const loom_symbol_t* source_symbol = &module->symbols.entries[old_index];
    if (!loom_module_symbol_has_payload(source_symbol) &&
        referenced_symbols[old_index] == 0) {
      module->symbols.entries[new_symbol_id] = (loom_symbol_t){
          .name_id = LOOM_STRING_ID_INVALID,
      };
      continue;
    }
    module->symbols.entries[new_symbol_id] = *source_symbol;
  }
  memset(&module->symbols.entries[new_symbol_count], 0,
         (old_symbol_count - new_symbol_count) *
             sizeof(module->symbols.entries[0]));
  module->symbols.count = new_symbol_count;
  if (out_removed_count) {
    *out_removed_count = removed_count;
  }
  return iree_ok_status();
}

iree_status_t loom_module_compact_symbols(loom_module_t* module,
                                          iree_arena_allocator_t* scratch_arena,
                                          iree_host_size_t* out_removed_count) {
  return loom_module_compact_symbols_preserving_symbol_refs(
      module, NULL, 0, scratch_arena, out_removed_count);
}

//===----------------------------------------------------------------------===//
// Location table
//===----------------------------------------------------------------------===//

iree_status_t loom_module_add_location(loom_module_t* module,
                                       loom_location_entry_t entry,
                                       loom_location_id_t* out_location_id) {
  // Location IDs are 32-bit and ID 0 is reserved for LOOM_LOCATION_UNKNOWN.
  // IDs 1 through UINT32_MAX are representable, so once count advances past
  // UINT32_MAX the next cast would wrap to 0 and forge the null sentinel.
  if (module->locations.count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "location table full (%" PRIhsz
                            " entries, max id %u)",
                            module->locations.count, (unsigned)UINT32_MAX);
  }

  loom_location_id_t id = (loom_location_id_t)module->locations.count;
  loom_location_segment_t* segment = NULL;
  if ((id & LOOM_LOCATION_SEGMENT_MASK) == 0) {
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &module->locations.segments, &module->arena, (void**)&segment));
    if (id == LOOM_LOCATION_UNKNOWN) {
      segment->rows[0] = (loom_location_entry_t){.kind = LOOM_LOCATION_NONE};
      id = 1;
    }
  } else {
    segment = (loom_location_segment_t*)loom_segmented_storage_segment(
        &module->locations.segments, id >> LOOM_LOCATION_SEGMENT_SHIFT);
  }
  segment->rows[id & LOOM_LOCATION_SEGMENT_MASK] = entry;
  module->locations.count = (iree_host_size_t)id + 1;
  *out_location_id = id;
  return iree_ok_status();
}

iree_status_t loom_module_attach_location_field_spans(
    loom_module_t* module, loom_location_id_t location_id,
    const loom_location_field_span_t* field_spans,
    iree_host_size_t field_span_count) {
  if (location_id == LOOM_LOCATION_UNKNOWN ||
      (iree_host_size_t)location_id >= module->locations.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location_id %u out of range for field span "
                            "attachment (module has %" PRIhsz " locations)",
                            location_id, module->locations.count);
  }
  loom_location_entry_t* entry =
      loom_location_table_entry(&module->locations, location_id);
  if (entry->kind != LOOM_LOCATION_FILE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot attach field spans to non-file location kind %d",
        (int)entry->kind);
  }
  if (entry->file.field_span_count > 0 || entry->file.field_spans) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location %u already has field spans attached",
                            location_id);
  }
  if (field_span_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "location field span count %" PRIhsz
                            " exceeds maximum %u",
                            field_span_count, (unsigned)UINT16_MAX);
  }
  if (field_span_count > 0 && !field_spans) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "field_span_count > 0 requires non-NULL spans");
  }

  loom_location_field_span_t* copied_spans = NULL;
  if (field_span_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, field_span_count, sizeof(*copied_spans),
        (void**)&copied_spans));
    memcpy(copied_spans, field_spans, field_span_count * sizeof(*copied_spans));
  }
  entry->file.field_span_count = (uint16_t)field_span_count;
  entry->file.field_spans = copied_spans;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Source comments
//===----------------------------------------------------------------------===//

static const loom_comment_attachment_t* loom_module_find_comment_attachment(
    const loom_module_t* module, loom_comment_owner_kind_t owner_kind,
    const void* owner) {
  for (iree_host_size_t i = 0; i < module->comments.count; ++i) {
    const loom_comment_attachment_t* attachment = &module->comments.entries[i];
    if (attachment->owner_kind == owner_kind && attachment->owner == owner) {
      return attachment;
    }
  }
  return NULL;
}

// Stores one attachment's views and payload bytes in one module-arena
// allocation.
static iree_status_t loom_module_store_comments(
    loom_module_t* module, const iree_string_view_t* comments,
    iree_host_size_t comment_count, iree_string_view_t** out_comments) {
  *out_comments = NULL;
  if (!comments) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "comment_count > 0 requires non-NULL comments");
  }

  iree_host_size_t comment_payload_size = 0;
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    if (!iree_host_size_checked_add(comment_payload_size, comments[i].size,
                                    &comment_payload_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "comment payload storage size overflow");
    }
  }
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_mul_add(comment_payload_size, comment_count,
                                      sizeof(*comments), &storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "comment storage size overflow");
  }

  iree_string_view_t* stored_comments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&module->arena, storage_size,
                                           (void**)&stored_comments));
  char* stored_payload = (char*)(stored_comments + comment_count);
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    if (comments[i].size > 0) {
      memcpy(stored_payload, comments[i].data, comments[i].size);
      stored_comments[i] =
          iree_make_string_view(stored_payload, comments[i].size);
      stored_payload += comments[i].size;
    } else {
      stored_comments[i] = iree_string_view_empty();
    }
  }
  *out_comments = stored_comments;
  return iree_ok_status();
}

static iree_status_t loom_module_attach_comments(
    loom_module_t* module, loom_comment_owner_kind_t owner_kind,
    const void* owner, const iree_string_view_t* comments,
    iree_host_size_t comment_count) {
  if (comment_count == 0) {
    return iree_ok_status();
  }
  if (!owner) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "comment attachment requires non-NULL owner");
  }
  if (comment_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "comment count %" PRIhsz " exceeds maximum %u",
                            comment_count, (unsigned)UINT16_MAX);
  }
  if (loom_module_find_comment_attachment(module, owner_kind, owner)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "comments are already attached to owner");
  }

  iree_string_view_t* stored_comments = NULL;
  IREE_RETURN_IF_ERROR(loom_module_store_comments(
      module, comments, comment_count, &stored_comments));
  IREE_RETURN_IF_ERROR(
      loom_comment_table_ensure_capacity(&module->arena, &module->comments));
  module->comments.entries[module->comments.count++] =
      (loom_comment_attachment_t){
          .owner = owner,
          .owner_kind = owner_kind,
          .comment_count = (uint16_t)comment_count,
          .comments = stored_comments,
      };
  return iree_ok_status();
}

iree_status_t loom_module_attach_file_header(loom_module_t* module,
                                             const iree_string_view_t* lines,
                                             iree_host_size_t line_count) {
  return loom_module_attach_comments(module, LOOM_COMMENT_OWNER_MODULE, module,
                                     lines, line_count);
}

iree_status_t loom_module_attach_op_comments(loom_module_t* module,
                                             const loom_op_t* op,
                                             const iree_string_view_t* comments,
                                             iree_host_size_t comment_count) {
  return loom_module_attach_comments(module, LOOM_COMMENT_OWNER_OP, op,
                                     comments, comment_count);
}

iree_status_t loom_module_attach_block_comments(
    loom_module_t* module, const loom_block_t* block,
    const iree_string_view_t* comments, iree_host_size_t comment_count) {
  return loom_module_attach_comments(module, LOOM_COMMENT_OWNER_BLOCK, block,
                                     comments, comment_count);
}

static const iree_string_view_t* loom_module_comments(
    const loom_module_t* module, loom_comment_owner_kind_t owner_kind,
    const void* owner, iree_host_size_t* out_comment_count) {
  if (out_comment_count) {
    *out_comment_count = 0;
  }
  if (!owner) {
    return NULL;
  }
  const loom_comment_attachment_t* attachment =
      loom_module_find_comment_attachment(module, owner_kind, owner);
  if (!attachment) {
    return NULL;
  }
  if (out_comment_count) {
    *out_comment_count = attachment->comment_count;
  }
  return attachment->comments;
}

const iree_string_view_t* loom_module_file_header(
    const loom_module_t* module, iree_host_size_t* out_line_count) {
  return loom_module_comments(module, LOOM_COMMENT_OWNER_MODULE, module,
                              out_line_count);
}

const iree_string_view_t* loom_module_op_comments(
    const loom_module_t* module, const loom_op_t* op,
    iree_host_size_t* out_comment_count) {
  return loom_module_comments(module, LOOM_COMMENT_OWNER_OP, op,
                              out_comment_count);
}

const iree_string_view_t* loom_module_block_comments(
    const loom_module_t* module, const loom_block_t* block,
    iree_host_size_t* out_comment_count) {
  return loom_module_comments(module, LOOM_COMMENT_OWNER_BLOCK, block,
                              out_comment_count);
}

//===----------------------------------------------------------------------===//
// Value definition
//===----------------------------------------------------------------------===//

static iree_status_t loom_module_canonicalize_value_type(
    loom_module_t* module, loom_type_t type, loom_type_t* out_type,
    loom_type_dependency_id_t* out_dependencies) {
  *out_dependencies = 0;
  if (loom_type_kind(type) == LOOM_TYPE_NONE) {
    *out_type = type;
    return iree_ok_status();
  }
  loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_type_id(module, type, &type_id));
  *out_type = loom_type_table_get(&module->types, type_id);
  *out_dependencies = loom_type_table_dependencies(&module->types, type_id);
  return iree_ok_status();
}

iree_status_t loom_module_define_value(loom_module_t* module, loom_type_t type,
                                       loom_value_id_t* out_value_id) {
  // Value IDs are 32-bit and LOOM_VALUE_ID_INVALID is the null sentinel.
  // Fail before count reaches the sentinel value so the cast below cannot
  // produce an invalid ID from user-controlled input size.
  if (module->values.count >= LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "value table full (%" PRIhsz " entries, max id %u)",
                            module->values.count,
                            (unsigned)(LOOM_VALUE_ID_INVALID - 1));
  }

  IREE_RETURN_IF_ERROR(loom_value_table_ensure_capacity(module));
  loom_type_t canonical_type = {0};
  loom_type_dependency_id_t dependencies = 0;
  IREE_RETURN_IF_ERROR(loom_module_canonicalize_value_type(
      module, type, &canonical_type, &dependencies));
  const loom_value_id_t id = (loom_value_id_t)module->values.count;
  loom_type_dependency_assignment_t assignment;
  IREE_RETURN_IF_ERROR(
      loom_type_dependencies_prepare(&module->type_uses, id, dependencies,
                                     module->values.count + 1, &assignment));
  ++module->values.count;
  loom_value_t* value = loom_module_value(module, id);
  value->type = canonical_type;
  value->name_id = LOOM_STRING_ID_INVALID;
  value->def = loom_value_def_make_none();
  loom_value_u32_scratch_t* scratch = &module->scratch.values;
  if (scratch->state == LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ZEROED ||
      scratch->state == LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ZEROED) {
    loom_value_u32_scratch_store(scratch, id, 0);
  }

  loom_type_dependencies_commit(&module->type_uses, id, &assignment);

  *out_value_id = id;
  return iree_ok_status();
}

iree_status_t loom_module_define_untyped_values(
    loom_module_t* module, iree_host_size_t count,
    loom_value_id_t* out_base_value_id) {
  *out_base_value_id = LOOM_VALUE_ID_INVALID;
  if (count == 0) {
    return iree_ok_status();
  }
  if (count > LOOM_VALUE_ID_INVALID - module->values.count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "value table full (%" PRIhsz
                            " entries, cannot append %" PRIhsz ")",
                            module->values.count, count);
  }

  IREE_RETURN_IF_ERROR(loom_value_table_reserve(module, count));
  const loom_value_id_t base_value_id = (loom_value_id_t)module->values.count;
  const bool zero_scratch = module->scratch.values.state ==
                                LOOM_VALUE_U32_SCRATCH_STATE_ACQUIRED_ZEROED ||
                            module->scratch.values.state ==
                                LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ZEROED;
  iree_host_size_t remaining_count = count;
  iree_host_size_t value_ordinal = module->values.count;
  module->values.count += count;
  while (remaining_count > 0) {
    loom_value_segment_t* segment = loom_value_table_segment_for_id(
        &module->values, (loom_value_id_t)value_ordinal);
    const iree_host_size_t segment_offset =
        value_ordinal & LOOM_VALUE_SEGMENT_MASK;
    const iree_host_size_t segment_count =
        iree_min(remaining_count, LOOM_VALUE_SEGMENT_CAPACITY - segment_offset);
    for (iree_host_size_t i = 0; i < segment_count; ++i) {
      segment->values[segment_offset + i].name_id = LOOM_STRING_ID_INVALID;
    }
    if (zero_scratch) {
      memset(&segment->u32_scratch[segment_offset], 0,
             segment_count * sizeof(segment->u32_scratch[0]));
    }
    value_ordinal += segment_count;
    remaining_count -= segment_count;
  }
  *out_base_value_id = base_value_id;
  return iree_ok_status();
}

iree_status_t loom_module_set_value_type(loom_module_t* module,
                                         loom_value_id_t value_id,
                                         loom_type_t type) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %%%u out of range (module has %" PRIhsz
                            " values)",
                            (unsigned)value_id, module->values.count);
  }
  loom_value_t* value = loom_module_value(module, value_id);
  loom_type_t canonical_type = {0};
  loom_type_dependency_id_t dependencies = 0;
  IREE_RETURN_IF_ERROR(loom_module_canonicalize_value_type(
      module, type, &canonical_type, &dependencies));
  loom_type_dependency_assignment_t assignment;
  IREE_RETURN_IF_ERROR(
      loom_type_dependencies_prepare(&module->type_uses, value_id, dependencies,
                                     module->values.count, &assignment));
  loom_type_dependencies_commit(&module->type_uses, value_id, &assignment);
  value->type = canonical_type;
  return iree_ok_status();
}

iree_status_t loom_module_set_value_name(loom_module_t* module,
                                         loom_value_id_t value_id,
                                         loom_string_id_t name_id) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %%%u out of range (module has %" PRIhsz
                            " values)",
                            (unsigned)value_id, module->values.count);
  }
  if (name_id != LOOM_STRING_ID_INVALID && name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value name string id %u out of range (module has "
                            "%" PRIhsz " strings)",
                            (unsigned)name_id, module->strings.count);
  }
  loom_module_value(module, value_id)->name_id = name_id;
  return iree_ok_status();
}

iree_status_t loom_module_copy_value_name(loom_module_t* module,
                                          loom_value_id_t source_value_id,
                                          loom_value_id_t target_value_id) {
  if (source_value_id >= module->values.count ||
      target_value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value name copy ids out of range");
  }
  const loom_value_t* source_value = loom_module_value(module, source_value_id);
  if (source_value->name_id == LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }
  loom_value_t* target_value = loom_module_value(module, target_value_id);
  if (target_value->name_id != LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_module_set_value_name(module, target_value_id,
                                    source_value->name_id);
}

iree_status_t loom_module_overwrite_value_name(
    loom_module_t* module, loom_value_id_t source_value_id,
    loom_value_id_t target_value_id) {
  if (source_value_id >= module->values.count ||
      target_value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value name overwrite ids out of range");
  }
  const loom_value_t* source_value = loom_module_value(module, source_value_id);
  if (source_value->name_id == LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_module_set_value_name(module, target_value_id,
                                    source_value->name_id);
}

iree_status_t loom_module_move_value_name(loom_module_t* module,
                                          loom_value_id_t source_value_id,
                                          loom_value_id_t target_value_id) {
  if (source_value_id >= module->values.count ||
      target_value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value name move ids out of range");
  }
  const loom_string_id_t name_id =
      loom_module_value(module, source_value_id)->name_id;
  if (name_id == LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_module_clear_value_name(module, source_value_id));
  return loom_module_set_value_name(module, target_value_id, name_id);
}

iree_status_t loom_module_clear_value_name(loom_module_t* module,
                                           loom_value_id_t value_id) {
  return loom_module_set_value_name(module, value_id, LOOM_STRING_ID_INVALID);
}

iree_status_t loom_module_try_set_derived_value_name(
    loom_module_t* module, loom_value_id_t source_value_id,
    loom_value_id_t target_value_id, iree_string_view_t suffix,
    iree_arena_allocator_t* scratch_arena) {
  if (source_value_id >= module->values.count ||
      target_value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "derived value name ids out of range");
  }
  if (iree_string_view_is_empty(suffix)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "derived value name suffix is empty");
  }
  const loom_value_t* source_value = loom_module_value(module, source_value_id);
  if (source_value->name_id == LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }
  if (source_value->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source value name string id %u out of range "
                            "(module has %" PRIhsz " strings)",
                            (unsigned)source_value->name_id,
                            module->strings.count);
  }
  const loom_value_t* target_value = loom_module_value(module, target_value_id);
  if (target_value->name_id != LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }

  iree_string_view_t source_name =
      loom_string_table_get(&module->strings, source_value->name_id);
  iree_host_size_t name_length = 0;
  if (!iree_host_size_checked_add(source_name.size, 1, &name_length) ||
      !iree_host_size_checked_add(name_length, suffix.size, &name_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "derived value name length overflow");
  }

  char* name_storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, name_length, (void**)&name_storage));
  memcpy(name_storage, source_name.data, source_name.size);
  name_storage[source_name.size] = '_';
  memcpy(name_storage + source_name.size + 1, suffix.data, suffix.size);

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_make_string_view(name_storage, name_length), &name_id));
  return loom_module_set_value_name(module, target_value_id, name_id);
}

iree_status_t loom_module_refresh_value_type_uses(loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %%%u out of range (module has %" PRIhsz
                            " values)",
                            (unsigned)value_id, module->values.count);
  }
  return loom_type_dependencies_refresh(&module->type_uses, value_id);
}

iree_status_t loom_module_recompute_type_uses(loom_module_t* module) {
  return loom_type_dependencies_recompute(&module->type_uses);
}

bool loom_module_value_has_type_uses(const loom_module_t* module,
                                     loom_value_id_t value_id) {
  return loom_type_dependencies_has_users(&module->type_uses, value_id);
}

void loom_module_drop_value_type_uses(loom_module_t* module,
                                      loom_value_id_t value_id) {
  loom_type_dependencies_drop(&module->type_uses, value_id);
}

//===----------------------------------------------------------------------===//
// String interning
//===----------------------------------------------------------------------===//

typedef struct loom_string_equal_context_t {
  const loom_module_t* module;
  iree_string_view_t string;
} loom_string_equal_context_t;

static bool loom_string_equal_fn(const void* context, uint32_t index) {
  const loom_string_equal_context_t* ctx =
      (const loom_string_equal_context_t*)context;
  return iree_string_view_equal(
      loom_string_table_get(&ctx->module->strings, index), ctx->string);
}

iree_status_t loom_module_intern_string(loom_module_t* module,
                                        iree_string_view_t string,
                                        loom_string_id_t* out_string_id) {
  uint32_t hash = loom_hash_string_view(string);
  loom_string_equal_context_t equal_context = {module, string};

  const loom_intern_probe_t probe = loom_intern_table_probe(
      &module->string_intern, hash, loom_string_equal_fn, &equal_context);
  if (probe.index != UINT32_MAX) {
    *out_string_id = (loom_string_id_t)probe.index;
    return iree_ok_status();
  }

  // String IDs are 32-bit and LOOM_STRING_ID_INVALID is the null sentinel.
  // Check only after the duplicate probe so an already-interned spelling
  // still resolves when the table is full.
  if (module->strings.count >= LOOM_STRING_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "string table full (%" PRIhsz " entries, max id %u)",
        module->strings.count, (unsigned)(LOOM_STRING_ID_INVALID - 1));
  }

  // Prepare the row and bytes before the last fallible bucket reserve. The
  // duplicate probe retains the vacant slot unless growth changes placement.
  IREE_RETURN_IF_ERROR(
      loom_string_table_ensure_capacity(&module->arena, &module->strings));

  // New entry: arena-allocate a copy of the string data.
  char* copy = NULL;
  if (string.size > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&module->arena, string.size, (void**)&copy));
    memcpy(copy, string.data, string.size);
  }
  iree_host_size_t slot = probe.slot;
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      &module->arena, &module->string_intern, hash, &slot));
  const uint32_t new_index = (uint32_t)module->strings.count;
  loom_string_segment_t* segment =
      (loom_string_segment_t*)loom_segmented_storage_segment(
          &module->strings.segments, new_index >> LOOM_STRING_SEGMENT_SHIFT);
  segment->entries[new_index & LOOM_STRING_SEGMENT_MASK] =
      iree_make_string_view(copy, string.size);
  module->strings.count++;
  loom_intern_table_insert(&module->string_intern, slot, hash, new_index);

  *out_string_id = (loom_string_id_t)new_index;
  return iree_ok_status();
}

loom_string_id_t loom_module_lookup_string(const loom_module_t* module,
                                           iree_string_view_t string) {
  uint32_t hash = loom_hash_string_view(string);
  loom_string_equal_context_t equal_context = {module, string};
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &module->string_intern, hash, loom_string_equal_fn, &equal_context);
  return (loom_string_id_t)probe.index;
}

//===----------------------------------------------------------------------===//
// Symbol-set attributes
//===----------------------------------------------------------------------===//

static iree_status_t loom_module_resolve_symbol_ref_name(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "symbol set reference {module=%u, symbol=%u} is not a valid local "
        "symbol (module has %" PRIhsz " symbols)",
        (unsigned)ref.module_id, (unsigned)ref.symbol_id,
        module->symbols.count);
  }
  const loom_string_id_t name_id =
      module->symbols.entries[ref.symbol_id].name_id;
  *out_name = loom_string_table_get(&module->strings, name_id);
  return iree_ok_status();
}

static int loom_module_compare_symbol_refs_by_name(const loom_module_t* module,
                                                   loom_symbol_ref_t lhs,
                                                   loom_symbol_ref_t rhs) {
  loom_string_id_t lhs_name_id = module->symbols.entries[lhs.symbol_id].name_id;
  loom_string_id_t rhs_name_id = module->symbols.entries[rhs.symbol_id].name_id;
  return iree_string_view_compare(
      loom_string_table_get(&module->strings, lhs_name_id),
      loom_string_table_get(&module->strings, rhs_name_id));
}

static void loom_module_sift_symbol_ref_heap(const loom_module_t* module,
                                             loom_symbol_ref_t* refs,
                                             iree_host_size_t root,
                                             iree_host_size_t count) {
  while (root * 2 + 1 < count) {
    iree_host_size_t larger = root;
    iree_host_size_t left = root * 2 + 1;
    iree_host_size_t right = left + 1;
    if (loom_module_compare_symbol_refs_by_name(module, refs[larger],
                                                refs[left]) < 0) {
      larger = left;
    }
    if (right < count && loom_module_compare_symbol_refs_by_name(
                             module, refs[larger], refs[right]) < 0) {
      larger = right;
    }
    if (larger == root) {
      return;
    }
    loom_symbol_ref_t temporary = refs[root];
    refs[root] = refs[larger];
    refs[larger] = temporary;
    root = larger;
  }
}

static void loom_module_sort_symbol_refs_by_name(const loom_module_t* module,
                                                 loom_symbol_ref_t* refs,
                                                 iree_host_size_t count) {
  for (iree_host_size_t start = count / 2; start > 0; --start) {
    loom_module_sift_symbol_ref_heap(module, refs, start - 1, count);
  }
  for (iree_host_size_t end = count; end > 1; --end) {
    loom_symbol_ref_t temporary = refs[0];
    refs[0] = refs[end - 1];
    refs[end - 1] = temporary;
    loom_module_sift_symbol_ref_heap(module, refs, 0, end - 1);
  }
}

static iree_status_t loom_module_validate_symbol_set(
    const loom_module_t* module, const loom_symbol_ref_t* refs,
    iree_host_size_t ref_count) {
  if (ref_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol set has %" PRIhsz " elements, max %u",
                            ref_count, (unsigned)UINT16_MAX);
  }
  if (ref_count > 0 && !refs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty symbol set has a NULL payload");
  }
  for (iree_host_size_t i = 0; i < ref_count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_module_resolve_symbol_ref_name(module, refs[i], &name));
  }
  return iree_ok_status();
}

loom_symbol_ref_t loom_module_canonicalize_symbol_set(
    const loom_module_t* module, loom_symbol_ref_t* refs, uint16_t ref_count) {
  loom_module_sort_symbol_refs_by_name(module, refs, ref_count);
  for (uint16_t i = 1; i < ref_count; ++i) {
    if (loom_module_compare_symbol_refs_by_name(module, refs[i - 1], refs[i]) ==
        0) {
      return refs[i];
    }
  }
  return loom_symbol_ref_null();
}

iree_status_t loom_module_try_make_symbol_set(
    loom_module_t* module, loom_symbol_ref_array_t refs,
    loom_symbol_ref_t* out_duplicate_ref, loom_attribute_t* out_attr) {
  *out_duplicate_ref = loom_symbol_ref_null();
  *out_attr = loom_attr_absent();
  IREE_RETURN_IF_ERROR(
      loom_module_validate_symbol_set(module, refs.values, refs.count));
  if (refs.count == 0) {
    *out_attr = loom_attr_symbol_set(NULL, 0);
    return iree_ok_status();
  }

  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(&module->arena);
  loom_symbol_ref_t* sorted_refs = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &module->arena, refs.count, sizeof(*sorted_refs), (void**)&sorted_refs);
  if (!iree_status_is_ok(status)) {
    iree_arena_checkpoint_restore(&checkpoint);
    return status;
  }
  memcpy(sorted_refs, refs.values, refs.count * sizeof(*sorted_refs));
  *out_duplicate_ref = loom_module_canonicalize_symbol_set(
      module, sorted_refs, (uint16_t)refs.count);
  if (loom_symbol_ref_is_valid(*out_duplicate_ref)) {
    iree_arena_checkpoint_restore(&checkpoint);
    return iree_ok_status();
  }

  *out_attr = loom_attr_symbol_set(sorted_refs, (uint16_t)refs.count);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Canonical dictionary attributes
//===----------------------------------------------------------------------===//

static iree_status_t loom_module_resolve_attr_dict_key_name(
    const loom_module_t* module, loom_string_id_t name_id,
    iree_string_view_t* out_name) {
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dict attribute key string id %u is out of range (module has %" PRIhsz
        " strings)",
        name_id, module->strings.count);
  }
  *out_name = loom_string_table_get(&module->strings, name_id);
  return iree_ok_status();
}

// Aggregate values recurse through this helper with a shared hard depth bound.
static iree_status_t loom_module_canonicalize_attr_value(
    loom_module_t* module, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t value, iree_host_size_t depth,
    loom_attribute_t* out_value);

static bool loom_module_attr_dict_key_less(const loom_module_t* module,
                                           const loom_named_attr_t* lhs,
                                           const loom_named_attr_t* rhs) {
  return iree_string_view_compare(
             loom_string_table_get(&module->strings, lhs->name_id),
             loom_string_table_get(&module->strings, rhs->name_id)) < 0;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT_OUTLINED_HEAP(
    loom_module_sort_attr_dict_entries_impl, loom_named_attr_t,
    const loom_module_t*, loom_module_attr_dict_key_less)

void loom_module_sort_attr_dict_entries(const loom_module_t* module,
                                        loom_named_attr_t* entries,
                                        iree_host_size_t count) {
  loom_module_sort_attr_dict_entries_impl(module, entries, count);
}

static iree_status_t loom_module_make_canonical_attr_dict_entries(
    loom_module_t* module, const loom_named_attr_t* entries,
    iree_host_size_t count, iree_host_size_t depth,
    loom_attribute_t* out_attr) {
  if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dict attribute nesting exceeds max depth %u",
                            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
  }
  if (count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dict attribute has %" PRIhsz " entries, max %u",
                            count, (unsigned)UINT16_MAX);
  }
  if (count > 0 && !entries) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute has a NULL entry pointer");
  }

  if (count == 0) {
    *out_attr = loom_make_canonical_attr_dict(NULL, 0);
    return iree_ok_status();
  }

  loom_named_attr_t* canonical_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, count,
                                                 sizeof(loom_named_attr_t),
                                                 (void**)&canonical_entries));

  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_string_view_t key_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
        module, entries[i].name_id, &key_name));

    canonical_entries[i] = (loom_named_attr_t){
        .name_id = entries[i].name_id,
        .value = {0},
    };
    IREE_RETURN_IF_ERROR(loom_module_canonicalize_attr_value(
        module, /*descriptor=*/NULL, entries[i].value, depth + 1,
        &canonical_entries[i].value));
  }

  // Keys are validated once above. Interned spelling identity makes duplicate
  // detection an ID comparison after the bounded spelling-order sort.
  loom_module_sort_attr_dict_entries(module, canonical_entries, count);
  for (iree_host_size_t i = 1; i < count; ++i) {
    if (canonical_entries[i - 1].name_id == canonical_entries[i].name_id) {
      iree_string_view_t key_name =
          loom_string_table_get(&module->strings, canonical_entries[i].name_id);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate dict attribute key '%.*s'",
                              (int)key_name.size, key_name.data);
    }
  }
  *out_attr = loom_make_canonical_attr_dict(canonical_entries, count);
  return iree_ok_status();
}

static iree_status_t loom_module_make_parameterized_attr_slots(
    loom_module_t* module,
    const loom_parameterized_attr_descriptor_t* family_descriptor,
    const loom_attribute_t* parameters, iree_host_size_t parameter_count,
    iree_host_size_t depth, loom_attribute_t* out_attr) {
  if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute nesting exceeds max depth %u",
        (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
  }
  if (parameter_count != family_descriptor->parameter_count) {
    iree_string_view_t family_name = loom_bstring_view(family_descriptor->name);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameterized attribute '%.*s' has %" PRIhsz
                            " slots but its descriptor requires %u",
                            (int)family_name.size, family_name.data,
                            parameter_count,
                            family_descriptor->parameter_count);
  }
  if (parameter_count > 0 && parameters == NULL) {
    iree_string_view_t family_name = loom_bstring_view(family_descriptor->name);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty parameterized attribute '%.*s' has a NULL slot pointer",
        (int)family_name.size, family_name.data);
  }
  if (parameter_count == 0) {
    *out_attr = loom_make_parameterized_attr(family_descriptor->kind, NULL, 0);
    return iree_ok_status();
  }

  loom_attribute_t* canonical_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, parameter_count, sizeof(*canonical_slots),
      (void**)&canonical_slots));
  for (iree_host_size_t i = 0; i < parameter_count; ++i) {
    const loom_attr_descriptor_t* parameter_descriptor =
        &family_descriptor->parameter_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_module_canonicalize_attr_value(
        module, parameter_descriptor, parameters[i], depth + 1,
        &canonical_slots[i]));
  }

  *out_attr = loom_make_parameterized_attr(family_descriptor->kind,
                                           canonical_slots, parameter_count);
  return iree_ok_status();
}

static iree_status_t loom_module_make_parameterized_attr_array_values(
    loom_module_t* module, const loom_attribute_t* attributes,
    iree_host_size_t attribute_count, iree_host_size_t depth,
    loom_attribute_t* out_attr) {
  if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute array nesting exceeds max depth %u",
        (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
  }
  if (attribute_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "parameterized attribute array has %" PRIhsz
                            " elements, max %u",
                            attribute_count, (unsigned)UINT16_MAX);
  }
  if (attribute_count > 0 && attributes == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty parameterized attribute array has a NULL value pointer");
  }
  if (attribute_count == 0) {
    *out_attr = loom_attr_parameterized_array(NULL, 0);
    return iree_ok_status();
  }

  loom_attribute_t* canonical_attributes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, attribute_count, sizeof(*canonical_attributes),
      (void**)&canonical_attributes));
  for (iree_host_size_t i = 0; i < attribute_count; ++i) {
    if (attributes[i].kind != LOOM_ATTR_PARAMETERIZED) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameterized attribute array element %" PRIhsz
                              " has kind %u, expected PARAMETERIZED",
                              i, (unsigned)attributes[i].kind);
    }
    IREE_RETURN_IF_ERROR(loom_module_canonicalize_attr_value(
        module, /*descriptor=*/NULL, attributes[i], depth + 1,
        &canonical_attributes[i]));
  }

  *out_attr =
      loom_attr_parameterized_array(canonical_attributes, attribute_count);
  return iree_ok_status();
}

static iree_status_t loom_module_validate_attr_descriptor_value(
    const loom_attr_descriptor_t* descriptor, loom_attribute_t value) {
  if (!descriptor) {
    return iree_ok_status();
  }
  iree_string_view_t parameter_name = loom_attr_descriptor_name(descriptor);
  if (loom_attr_is_absent(value)) {
    if (iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPTIONAL)) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required parameter '%.*s' is absent",
                            (int)parameter_name.size, parameter_name.data);
  }

  if (!loom_attr_descriptor_accepts_kind(descriptor,
                                         (loom_attr_kind_t)value.kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter '%.*s' has attribute kind %u but requires kind %u",
        (int)parameter_name.size, parameter_name.data, (unsigned)value.kind,
        (unsigned)descriptor->attr_kind);
  }

  if (value.kind == LOOM_ATTR_ENUM &&
      !iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM) &&
      !loom_attr_descriptor_has_enum_case(descriptor,
                                          loom_attr_as_enum(value))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter '%.*s' has undeclared enum value %u",
                            (int)parameter_name.size, parameter_name.data,
                            loom_attr_as_enum(value));
  }
  if (value.kind == LOOM_ATTR_ENUM_ARRAY &&
      !iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
    if (value.count > 0 && value.enum_array == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "non-empty enum array parameter '%.*s' has a NULL payload",
          (int)parameter_name.size, parameter_name.data);
    }
    for (uint16_t i = 0; i < value.count; ++i) {
      if (loom_attr_descriptor_has_enum_case(descriptor, value.enum_array[i])) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter '%.*s' has undeclared enum array value %u",
          (int)parameter_name.size, parameter_name.data, value.enum_array[i]);
    }
  }
  if (value.kind == LOOM_ATTR_SIGNED_ENUM_SET) {
    if (iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "signed enum-set parameter '%.*s' requires a closed enum domain",
          (int)parameter_name.size, parameter_name.data);
    }
    iree_host_size_t canonical_word_count = 0;
    IREE_RETURN_IF_ERROR(loom_signed_enum_set_canonical_word_count(
        loom_attr_as_signed_enum_set(value), &canonical_word_count));
    for (iree_host_size_t word_index = 0; word_index < canonical_word_count;
         ++word_index) {
      const uint64_t asserted_values =
          value.signed_enum_set_words[word_index] |
          value.signed_enum_set_words[value.count + word_index];
      for (uint8_t bit_index = 0; bit_index < 64; ++bit_index) {
        if (!iree_any_bit_set(asserted_values, UINT64_C(1) << bit_index)) {
          continue;
        }
        const uint8_t enum_value = (uint8_t)(word_index * 64 + bit_index);
        if (loom_attr_descriptor_has_enum_case(descriptor, enum_value)) {
          continue;
        }
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter '%.*s' has undeclared signed enum-set value %u",
            (int)parameter_name.size, parameter_name.data,
            (unsigned)enum_value);
      }
    }
  }
  if (value.kind == LOOM_ATTR_PARAMETERIZED &&
      descriptor->reference.parameterized_attr_kind !=
          LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
      value.reserved_1 != descriptor->reference.parameterized_attr_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter '%.*s' has parameterized family kind %u but requires %u",
        (int)parameter_name.size, parameter_name.data,
        (unsigned)value.reserved_1,
        (unsigned)descriptor->reference.parameterized_attr_kind);
  }
  if (value.kind == LOOM_ATTR_PARAMETERIZED_ARRAY) {
    if (value.count > 0 && value.parameterized_array == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "non-empty parameterized attribute array parameter '%.*s' has a "
          "NULL payload",
          (int)parameter_name.size, parameter_name.data);
    }
    for (uint16_t i = 0; i < value.count; ++i) {
      const loom_attribute_t* element = &value.parameterized_array[i];
      if (element->kind != LOOM_ATTR_PARAMETERIZED) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array parameter '%.*s' element %u has "
            "kind %u, expected PARAMETERIZED",
            (int)parameter_name.size, parameter_name.data, (unsigned)i,
            (unsigned)element->kind);
      }
      if (descriptor->reference.parameterized_attr_kind !=
              LOOM_PARAMETERIZED_ATTR_KIND_ANY &&
          element->reserved_1 !=
              descriptor->reference.parameterized_attr_kind) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array parameter '%.*s' element %u has "
            "family kind %u but requires %u",
            (int)parameter_name.size, parameter_name.data, (unsigned)i,
            (unsigned)element->reserved_1,
            (unsigned)descriptor->reference.parameterized_attr_kind);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_module_canonicalize_attr_value(
    loom_module_t* module, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t value, iree_host_size_t depth,
    loom_attribute_t* out_value) {
  IREE_RETURN_IF_ERROR(
      loom_module_validate_attr_descriptor_value(descriptor, value));
  if (loom_attr_is_absent(value)) {
    if (descriptor == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generic attribute value is absent");
    }
    *out_value = loom_attr_absent();
    return iree_ok_status();
  }

  switch ((loom_attr_kind_t)value.kind) {
    case LOOM_ATTR_ABSENT:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "attribute value is absent");
    case LOOM_ATTR_I64:
      *out_value = loom_attr_i64(value.i64);
      return iree_ok_status();
    case LOOM_ATTR_F64:
      *out_value = loom_attr_f64(value.f64);
      return iree_ok_status();
    case LOOM_ATTR_STRING:
      if (value.string_id == LOOM_STRING_ID_INVALID ||
          value.string_id >= module->strings.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "attribute string value id %u is out of range "
                                "(module has %" PRIhsz " strings)",
                                value.string_id, module->strings.count);
      }
      *out_value = loom_attr_string(value.string_id);
      return iree_ok_status();
    case LOOM_ATTR_BOOL: {
      if (value.raw > 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "boolean attribute payload is %" PRIu64
                                ", expected 0 or 1",
                                value.raw);
      }
      *out_value = loom_attr_bool(value.raw != 0);
      return iree_ok_status();
    }
    case LOOM_ATTR_ENUM:
      if (value.raw > UINT8_MAX) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "enum attribute payload is %" PRIu64
                                ", exceeding the uint8_t domain",
                                value.raw);
      }
      *out_value = loom_attr_enum((uint8_t)value.raw);
      return iree_ok_status();
    case LOOM_ATTR_I64_ARRAY: {
      if (value.count == 0) {
        *out_value = loom_attr_i64_array(NULL, 0);
        return iree_ok_status();
      }
      if (value.i64_array == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty i64 array attribute has a NULL payload");
      }
      int64_t* values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &module->arena, value.count, sizeof(*values), (void**)&values));
      memcpy(values, value.i64_array,
             (iree_host_size_t)value.count * sizeof(*values));
      *out_value = loom_attr_i64_array(values, value.count);
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL:
      *out_value = loom_attr_symbol(value.symbol);
      return iree_ok_status();
    case LOOM_ATTR_SYMBOL_ARRAY: {
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol array attributes require a descriptor-backed field");
      }
      if (value.count == 0) {
        *out_value = loom_attr_symbol_array(NULL, 0);
        return iree_ok_status();
      }
      if (value.symbol_refs == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty symbol array attribute has a NULL payload");
      }
      loom_symbol_ref_t* values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &module->arena, value.count, sizeof(*values), (void**)&values));
      memcpy(values, value.symbol_refs,
             (iree_host_size_t)value.count * sizeof(*values));
      *out_value = loom_attr_symbol_array(values, value.count);
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL_SET: {
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol set attributes require a descriptor-backed field");
      }
      loom_symbol_ref_t duplicate_ref = loom_symbol_ref_null();
      IREE_RETURN_IF_ERROR(loom_module_try_make_symbol_set(
          module, loom_make_symbol_ref_array(value.symbol_refs, value.count),
          &duplicate_ref, out_value));
      if (loom_symbol_ref_is_valid(duplicate_ref)) {
        iree_string_view_t duplicate_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_module_resolve_symbol_ref_name(
            module, duplicate_ref, &duplicate_name));
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "symbol set contains duplicate '@%.*s'",
                                (int)duplicate_name.size, duplicate_name.data);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_TYPE:
      if (value.type_id == LOOM_TYPE_ID_INVALID ||
          value.type_id >= module->types.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "attribute type value id %u is out of range "
                                "(module has %" PRIhsz " types)",
                                value.type_id, module->types.count);
      }
      *out_value = loom_attr_type(value.type_id);
      return iree_ok_status();
    case LOOM_ATTR_PREDICATE_LIST: {
      if (value.count == 0) {
        *out_value = loom_attr_predicate_list(NULL, 0);
        return iree_ok_status();
      }
      if (value.predicate_list == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty predicate list attribute has a NULL payload");
      }
      loom_predicate_t* predicates = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(&module->arena, value.count,
                                    sizeof(*predicates), (void**)&predicates));
      memcpy(predicates, value.predicate_list,
             (iree_host_size_t)value.count * sizeof(*predicates));
      *out_value = loom_attr_predicate_list(predicates, value.count);
      return iree_ok_status();
    }
    case LOOM_ATTR_DICT:
      return loom_module_make_canonical_attr_dict_entries(
          module, value.dict_entries, value.count, depth, out_value);
    case LOOM_ATTR_ENCODING:
      if (value.encoding_id == 0 ||
          value.encoding_id > module->encodings.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "attribute encoding value id %u is out of range "
            "(module has %" PRIhsz " encodings)",
            (unsigned)value.encoding_id, module->encodings.count);
      }
      *out_value = loom_attr_encoding((uint16_t)value.encoding_id);
      return iree_ok_status();
    case LOOM_ATTR_BYTES: {
      if (value.reserved_1 == 0) {
        *out_value = loom_attr_bytes(NULL, 0);
        return iree_ok_status();
      }
      if (value.bytes == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "non-empty bytes attribute has a NULL payload");
      }
      uint8_t* bytes = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(&module->arena, value.reserved_1,
                                               (void**)&bytes));
      memcpy(bytes, value.bytes, value.reserved_1);
      *out_value = loom_attr_bytes(bytes, value.reserved_1);
      return iree_ok_status();
    }
    case LOOM_ATTR_SCOPED_ENUM:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "representation-scoped enum is not a generic attribute value");
    case LOOM_ATTR_ANY:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ANY is a descriptor kind, not an attribute");
    case LOOM_ATTR_ENUM_ARRAY: {
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "enum array attributes require a descriptor-backed field");
      }
      if (value.count == 0) {
        *out_value = loom_attr_enum_array(NULL, 0);
        return iree_ok_status();
      }
      if (value.enum_array == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty enum array attribute has a NULL payload");
      }
      uint8_t* values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &module->arena, value.count, sizeof(*values), (void**)&values));
      memcpy(values, value.enum_array,
             (iree_host_size_t)value.count * sizeof(*values));
      *out_value = loom_attr_enum_array(values, value.count);
      return iree_ok_status();
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "signed enum-set attributes require a descriptor-backed field");
      }
      loom_signed_enum_set_t set = loom_attr_as_signed_enum_set(value);
      iree_host_size_t canonical_word_count = 0;
      IREE_RETURN_IF_ERROR(loom_signed_enum_set_canonical_word_count(
          set, &canonical_word_count));
      if (canonical_word_count == 0) {
        *out_value = loom_attr_signed_enum_set(NULL, 0);
        return iree_ok_status();
      }
      uint64_t* words = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(&module->arena, canonical_word_count * 2,
                                    sizeof(*words), (void**)&words));
      memcpy(words, set.words, canonical_word_count * sizeof(*words));
      memcpy(words + canonical_word_count, set.words + set.word_count,
             canonical_word_count * sizeof(*words));
      *out_value =
          loom_attr_signed_enum_set(words, (uint16_t)canonical_word_count);
      return iree_ok_status();
    }
    case LOOM_ATTR_PARAMETERIZED: {
      if (value.reserved_1 > UINT16_MAX) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute family kind %u exceeds uint16_t",
            (unsigned)value.reserved_1);
      }
      const loom_parameterized_attr_descriptor_t* family_descriptor =
          loom_context_resolve_parameterized_attr(
              module->context,
              (loom_parameterized_attr_kind_t)value.reserved_1);
      if (family_descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute family kind %u is not registered",
            (unsigned)value.reserved_1);
      }
      return loom_module_make_parameterized_attr_slots(
          module, family_descriptor, value.parameterized_slots, value.count,
          depth, out_value);
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute arrays require a descriptor-backed "
            "field");
      }
      return loom_module_make_parameterized_attr_array_values(
          module, value.parameterized_array, value.count, depth, out_value);
    case LOOM_ATTR_COUNT_:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "attribute value has unknown kind %u",
                          (unsigned)value.kind);
}

iree_status_t loom_module_make_canonical_attribute(
    loom_module_t* module, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t value, loom_attribute_t* out_value) {
  return loom_module_canonicalize_attr_value(module, descriptor, value,
                                             /*depth=*/0, out_value);
}

iree_status_t loom_module_make_canonical_attr_dict(
    loom_module_t* module, loom_named_attr_slice_t entries,
    loom_attribute_t* out_attr) {
  return loom_module_make_canonical_attr_dict_entries(
      module, entries.entries, entries.count, 0, out_attr);
}

iree_status_t loom_module_make_parameterized_attr(
    loom_module_t* module, loom_parameterized_attr_kind_t family_kind,
    const loom_attribute_t* parameters, iree_host_size_t parameter_count,
    loom_attribute_t* out_attr) {
  const loom_parameterized_attr_descriptor_t* family_descriptor =
      loom_context_resolve_parameterized_attr(module->context, family_kind);
  if (family_descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameterized attribute family kind %u is not registered",
        (unsigned)family_kind);
  }
  return loom_module_make_parameterized_attr_slots(module, family_descriptor,
                                                   parameters, parameter_count,
                                                   /*depth=*/0, out_attr);
}

iree_status_t loom_module_make_parameterized_attr_array(
    loom_module_t* module, loom_parameterized_attr_array_t attributes,
    loom_attribute_t* out_attr) {
  return loom_module_make_parameterized_attr_array_values(
      module, attributes.values, attributes.count, /*depth=*/0, out_attr);
}

// One sorted update and the base range it consumes. The merge records these
// positions once, before allocating the exact result, so materialization only
// copies retained ranges and canonicalizes replacement values.
typedef struct loom_attr_dict_update_step_t {
  // Ordinal of the borrowed update, sorted by its key spelling.
  uint32_t update_index;
  // Unchanged base entries to copy before applying this update.
  uint16_t prefix_count;
  // First unconsumed base entry after this update.
  uint16_t next_base_index;
} loom_attr_dict_update_step_t;

typedef struct loom_attr_dict_update_order_t {
  // Module that owns every validated update key.
  const loom_module_t* module;
  // Borrowed updates whose ordinals are sorted.
  const loom_named_attr_update_t* updates;
} loom_attr_dict_update_order_t;

static bool loom_module_attr_dict_update_less(
    const loom_attr_dict_update_order_t* order,
    const loom_attr_dict_update_step_t* lhs,
    const loom_attr_dict_update_step_t* rhs) {
  return iree_string_view_compare(
             loom_string_table_get(&order->module->strings,
                                   order->updates[lhs->update_index].name_id),
             loom_string_table_get(&order->module->strings,
                                   order->updates[rhs->update_index].name_id)) <
         0;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT_OUTLINED_HEAP(
    loom_module_sort_attr_dict_updates, loom_attr_dict_update_step_t,
    const loom_attr_dict_update_order_t*, loom_module_attr_dict_update_less)

static iree_status_t loom_module_merge_attr_dict_updates(
    loom_module_t* module, loom_named_attr_slice_t base_entries,
    loom_named_attr_update_slice_t updates, loom_attr_dict_update_step_t* steps,
    loom_attribute_t* out_attr) {
  for (iree_host_size_t i = 0; i < updates.count; ++i) {
    steps[i] = (loom_attr_dict_update_step_t){.update_index = (uint32_t)i};
  }
  const loom_attr_dict_update_order_t order = {module, updates.updates};
  loom_module_sort_attr_dict_updates(&order, steps, updates.count);

  iree_host_size_t base_index = 0;
  iree_host_size_t result_count = base_entries.count;
  for (iree_host_size_t i = 0; i < updates.count; ++i) {
    const loom_named_attr_update_t* update =
        &updates.updates[steps[i].update_index];
    iree_string_view_t key =
        loom_string_table_get(&module->strings, update->name_id);
    if (i > 0 &&
        updates.updates[steps[i - 1].update_index].name_id == update->name_id) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate dict attribute update key '%.*s'",
                              (int)key.size, key.data);
    }
    iree_host_size_t prefix_start = base_index;
    while (base_index < base_entries.count &&
           iree_string_view_compare(
               loom_string_table_get(&module->strings,
                                     base_entries.entries[base_index].name_id),
               key) < 0) {
      ++base_index;
    }
    steps[i].prefix_count = (uint16_t)(base_index - prefix_start);
    bool found = base_index < base_entries.count &&
                 base_entries.entries[base_index].name_id == update->name_id;
    base_index += found;
    steps[i].next_base_index = (uint16_t)base_index;
    result_count += !update->remove;
    result_count -= found;
  }
  if (result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dict attribute replacement has %" PRIhsz
                            " entries, exceeding max %u",
                            result_count, (unsigned)UINT16_MAX);
  }

  loom_named_attr_t* result_entries = NULL;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, result_count,
                                                   sizeof(*result_entries),
                                                   (void**)&result_entries));
  }
  base_index = 0;
  iree_host_size_t result_index = 0;
  for (iree_host_size_t i = 0; i < updates.count; ++i) {
    if (steps[i].prefix_count) {
      memcpy(result_entries + result_index, base_entries.entries + base_index,
             steps[i].prefix_count * sizeof(*result_entries));
      result_index += steps[i].prefix_count;
    }
    base_index = steps[i].next_base_index;
    const loom_named_attr_update_t* update =
        &updates.updates[steps[i].update_index];
    if (!update->remove) {
      loom_attribute_t canonical_value = {0};
      IREE_RETURN_IF_ERROR(loom_module_canonicalize_attr_value(
          module, /*descriptor=*/NULL, update->value, /*depth=*/1,
          &canonical_value));
      result_entries[result_index++] = (loom_named_attr_t){
          .name_id = update->name_id,
          .reserved = 0,
          .value = canonical_value,
      };
    }
  }
  if (base_index < base_entries.count) {
    memcpy(result_entries + result_index, base_entries.entries + base_index,
           (base_entries.count - base_index) * sizeof(*result_entries));
  }
  *out_attr = loom_make_canonical_attr_dict(result_entries, result_count);
  return iree_ok_status();
}

iree_status_t loom_module_replace_canonical_attr_dict(
    loom_module_t* module, loom_named_attr_slice_t base_entries,
    loom_named_attr_update_slice_t updates, loom_attribute_t* out_attr) {
  *out_attr = loom_attr_absent();
  if (base_entries.count > 0 && !base_entries.entries) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty base dict attribute has a NULL entry pointer");
  }
  if (updates.count > 0 && !updates.updates) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute update list has a NULL update pointer");
  }
  if (base_entries.count > UINT16_MAX || updates.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dict attribute replacement inputs exceed %u "
                            "entries (%" PRIhsz " base + %" PRIhsz " updates)",
                            (unsigned)UINT16_MAX, base_entries.count,
                            updates.count);
  }
  loom_attribute_t base_attr =
      loom_make_canonical_attr_dict(base_entries.entries, base_entries.count);
  IREE_RETURN_IF_ERROR(
      loom_module_verify_canonical_attr_dict(module, base_attr));
  for (iree_host_size_t i = 0; i < updates.count; ++i) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
        module, updates.updates[i].name_id, &key));
  }

  loom_attr_dict_update_step_t inline_steps[16];
  if (updates.count <= IREE_ARRAYSIZE(inline_steps)) {
    return loom_module_merge_attr_dict_updates(module, base_entries, updates,
                                               inline_steps, out_attr);
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(module->arena.block_pool, &scratch_arena);
  loom_attr_dict_update_step_t* steps = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &scratch_arena, updates.count, sizeof(*steps), (void**)&steps);
  if (iree_status_is_ok(status)) {
    status = loom_module_merge_attr_dict_updates(module, base_entries, updates,
                                                 steps, out_attr);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static iree_status_t loom_module_verify_canonical_attr_header(
    const loom_attribute_t* attr, iree_string_view_t owner_name) {
  if (attr->reserved_0 != 0 || attr->reserved_1 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s has non-zero reserved bits",
                            (int)owner_name.size, owner_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_module_verify_canonical_attr_value(
    const loom_module_t* module, const loom_attr_descriptor_t* descriptor,
    const loom_attribute_t* value, iree_host_size_t depth) {
  IREE_RETURN_IF_ERROR(
      loom_module_validate_attr_descriptor_value(descriptor, *value));
  if (loom_attr_is_absent(*value)) {
    if (descriptor == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generic attribute value is absent");
    }
    return iree_ok_status();
  }

  if (value->reserved_0 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attribute value has non-zero reserved bits");
  }
  switch ((loom_attr_kind_t)value->kind) {
    case LOOM_ATTR_ABSENT:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "attribute value is absent");
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_SYMBOL:
      return loom_module_verify_canonical_attr_header(
          value, IREE_SV("attribute value"));
    case LOOM_ATTR_BOOL: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("boolean attribute")));
      if (value->raw > 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "boolean attribute payload is %" PRIu64
                                ", expected 0 or 1",
                                value->raw);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_ENUM: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("enum attribute")));
      if (value->raw > UINT8_MAX) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "enum attribute payload is %" PRIu64
                                ", exceeding the uint8_t domain",
                                value->raw);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_STRING: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("string attribute")));
      if (value->string_id == LOOM_STRING_ID_INVALID ||
          value->string_id >= module->strings.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "attribute string value id %u is out of range "
                                "(module has %" PRIhsz " strings)",
                                value->string_id, module->strings.count);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_TYPE: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("type attribute")));
      if (value->type_id == LOOM_TYPE_ID_INVALID ||
          value->type_id >= module->types.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "attribute type value id %u is out of range "
                                "(module has %" PRIhsz " types)",
                                value->type_id, module->types.count);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("encoding attribute")));
      if (value->encoding_id == 0 ||
          value->encoding_id > module->encodings.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "attribute encoding value id %u is out of range "
            "(module has %" PRIhsz " encodings)",
            (unsigned)value->encoding_id, module->encodings.count);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("i64 array attribute")));
      if ((value->count > 0) != (value->i64_array != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "i64 array attribute pointer does not match its element count");
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("symbol array attribute")));
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol array attributes require a descriptor-backed field");
      }
      if ((value->count > 0) != (value->symbol_refs != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol array attribute pointer does not match its element count");
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL_SET: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("symbol set attribute")));
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol set attributes require a descriptor-backed field");
      }
      if ((value->count > 0) != (value->symbol_refs != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol set attribute pointer does not match its element count");
      }
      iree_string_view_t previous_name = iree_string_view_empty();
      for (uint16_t i = 0; i < value->count; ++i) {
        iree_string_view_t current_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_module_resolve_symbol_ref_name(
            module, value->symbol_refs[i], &current_name));
        if (i > 0 &&
            iree_string_view_compare(previous_name, current_name) >= 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "symbol set attribute is not sorted and unique");
        }
        previous_name = current_name;
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("predicate list attribute")));
      if ((value->count > 0) != (value->predicate_list != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "predicate list attribute pointer does not match its count");
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_ENUM_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("enum array attribute")));
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "enum array attributes require a descriptor-backed field");
      }
      if ((value->count > 0) != (value->enum_array != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "enum array attribute pointer does not match its element count");
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("signed enum-set attribute")));
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "signed enum-set attributes require a descriptor-backed field");
      }
      iree_host_size_t canonical_word_count = 0;
      IREE_RETURN_IF_ERROR(loom_signed_enum_set_canonical_word_count(
          loom_attr_as_signed_enum_set(*value), &canonical_word_count));
      if (canonical_word_count != value->count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "signed enum-set attribute has %u words per polarity but "
            "canonical form requires %" PRIhsz,
            (unsigned)value->count, canonical_word_count);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_BYTES:
      if (value->count != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "bytes attribute has a non-zero count field");
      }
      if ((value->reserved_1 > 0) != (value->bytes != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "bytes attribute pointer does not match its byte length");
      }
      return iree_ok_status();
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("dict attribute")));
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if ((value->count > 0) != (value->dict_entries != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dict attribute pointer does not match its entry count");
      }
      iree_string_view_t previous_key_name = iree_string_view_empty();
      for (uint16_t i = 0; i < value->count; ++i) {
        const loom_named_attr_t* entry = &value->dict_entries[i];
        if (entry->reserved != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "dict attribute entry %u has non-zero reserved bits", i);
        }
        iree_string_view_t key_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
            module, entry->name_id, &key_name));
        if (i > 0) {
          int comparison =
              iree_string_view_compare(key_name, previous_key_name);
          if (comparison == 0) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "duplicate dict attribute key '%.*s'",
                                    (int)key_name.size, key_name.data);
          }
          if (comparison < 0) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "dict attribute key '%.*s' appears after '%.*s' out of "
                "canonical order",
                (int)key_name.size, key_name.data, (int)previous_key_name.size,
                previous_key_name.data);
          }
        }
        IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_value(
            module, /*descriptor=*/NULL, &entry->value, depth + 1));
        previous_key_name = key_name;
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_PARAMETERIZED: {
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      const loom_parameterized_attr_descriptor_t* family_descriptor =
          loom_context_resolve_parameterized_attr(
              module->context,
              (loom_parameterized_attr_kind_t)value->reserved_1);
      if (family_descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute family kind %u is not registered",
            (unsigned)value->reserved_1);
      }
      if (value->count != family_descriptor->parameter_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute has %u slots but its descriptor "
            "requires %u",
            value->count, family_descriptor->parameter_count);
      }
      if ((value->count > 0) != (value->parameterized_slots != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute pointer does not match its slot count");
      }
      for (uint16_t i = 0; i < value->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_value(
            module, &family_descriptor->parameter_descriptors[i],
            &value->parameterized_slots[i], depth + 1));
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
          value, IREE_SV("parameterized attribute array")));
      if (descriptor == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute arrays require a descriptor-backed "
            "field");
      }
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      if ((value->count > 0) != (value->parameterized_array != NULL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array pointer does not match its "
            "element count");
      }
      loom_attr_descriptor_t element_descriptor = *descriptor;
      element_descriptor.attr_kind = LOOM_ATTR_PARAMETERIZED;
      for (uint16_t i = 0; i < value->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_value(
            module, &element_descriptor, &value->parameterized_array[i],
            depth + 1));
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SCOPED_ENUM:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "representation-scoped enum is not a generic attribute value");
    case LOOM_ATTR_ANY:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ANY is a descriptor kind, not an attribute");
    case LOOM_ATTR_COUNT_:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "attribute value has unknown kind %u",
                          (unsigned)value->kind);
}

iree_status_t loom_module_verify_canonical_attr_dict(
    const loom_module_t* module, loom_attribute_t attr) {
  if (attr.kind != LOOM_ATTR_DICT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a DICT attribute, got kind %u",
                            (unsigned)attr.kind);
  }
  return loom_module_verify_canonical_attr_value(module, /*descriptor=*/NULL,
                                                 &attr, /*depth=*/0);
}

//===----------------------------------------------------------------------===//
// Type interning
//===----------------------------------------------------------------------===//

typedef struct loom_type_equal_context_t {
  const loom_module_t* module;
  loom_type_t type;
} loom_type_equal_context_t;

// One type assembled over canonical immediate dependencies already interned in
// the module. Dependency IDs follow the type's representation order.
typedef struct loom_topological_type_context_t {
  // Module owning every dependency ID.
  const loom_module_t* module;
  // Temporary top-level type representation.
  loom_type_t type;
  // Canonical immediate dependency IDs in representation order.
  const loom_type_id_t* dependency_ids;
  // Number of entries in dependency_ids.
  iree_host_size_t dependency_count;
} loom_topological_type_context_t;

typedef iree_status_t (*loom_module_type_clone_fn_t)(loom_module_t* module,
                                                     const void* clone_context,
                                                     loom_type_t* out_type);

static bool loom_type_has_same_storage(loom_type_t lhs, loom_type_t rhs) {
  return lhs.header == rhs.header && lhs.encoding_id == rhs.encoding_id &&
         lhs.encoding_flags == rhs.encoding_flags &&
         lhs.dims[0] == rhs.dims[0] && lhs.dims[1] == rhs.dims[1];
}

static void loom_module_note_recent_exact_type(loom_module_t* module,
                                               loom_type_id_t type_id) {
  IREE_ASSERT(type_id < module->types.count);
  const uint32_t ordinal = type_id + 1;
  if (module->recent_exact_type_ordinals[0] == ordinal) {
    return;
  }
  module->recent_exact_type_ordinals[1] = module->recent_exact_type_ordinals[0];
  module->recent_exact_type_ordinals[0] = ordinal;
}

static loom_type_id_t loom_module_find_recent_exact_type(
    const loom_module_t* module, loom_type_t type) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(module->recent_exact_type_ordinals); ++i) {
    const uint32_t ordinal = module->recent_exact_type_ordinals[i];
    if (ordinal == 0) {
      continue;
    }
    const loom_type_id_t type_id = ordinal - 1;
    if (loom_type_has_same_storage(loom_type_table_get(&module->types, type_id),
                                   type)) {
      return type_id;
    }
  }
  return LOOM_TYPE_ID_INVALID;
}

static void loom_module_note_recent_register_type(loom_module_t* module,
                                                  loom_type_id_t type_id) {
  IREE_ASSERT(type_id < module->types.count);
  IREE_ASSERT(
      loom_type_is_register(loom_type_table_get(&module->types, type_id)));
  IREE_ASSERT(loom_type_register_has_value_type(
      loom_type_table_get(&module->types, type_id)));
  const uint32_t ordinal = type_id + 1;
  if (module->recent_register_type_ordinals[0] == ordinal) {
    return;
  }
  module->recent_register_type_ordinals[1] =
      module->recent_register_type_ordinals[0];
  module->recent_register_type_ordinals[0] = ordinal;
}

static loom_type_id_t loom_module_find_recent_register_type_structural(
    const loom_module_t* module, loom_type_t type) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(module->recent_register_type_ordinals); ++i) {
    const uint32_t ordinal = module->recent_register_type_ordinals[i];
    if (ordinal == 0) {
      continue;
    }
    const loom_type_id_t type_id = ordinal - 1;
    const loom_register_type_data_t* existing =
        loom_type_register_data(loom_type_table_get(&module->types, type_id));
    const loom_register_type_data_t* candidate = loom_type_register_data(type);
    if (existing->carrier_payload0 == candidate->carrier_payload0 &&
        existing->carrier_payload1 == candidate->carrier_payload1 &&
        loom_type_has_same_storage(existing->value_type,
                                   candidate->value_type)) {
      return type_id;
    }
  }
  return LOOM_TYPE_ID_INVALID;
}

// Compares one interned module type against a temporary by-value candidate.
static bool loom_type_equal_fn(const void* context, uint32_t index) {
  const loom_type_equal_context_t* ctx =
      (const loom_type_equal_context_t*)context;
  return loom_type_equal(loom_type_table_get(&ctx->module->types, index),
                         ctx->type);
}

// Compares only the immediate structure of a topologically assembled type.
// Nested type identities are canonical module entries and therefore compare by
// their exact by-value storage instead of recursively walking their payloads.
static bool loom_topological_type_equal_fn(const void* context,
                                           uint32_t index) {
  const loom_topological_type_context_t* ctx =
      (const loom_topological_type_context_t*)context;
  const loom_type_t existing = loom_type_table_get(&ctx->module->types, index);
  const loom_type_t candidate = ctx->type;
  if (existing.header != candidate.header ||
      existing.encoding_id != candidate.encoding_id ||
      existing.encoding_flags != candidate.encoding_flags) {
    return false;
  }

  switch (loom_type_kind(candidate)) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* existing_data =
          loom_type_func_data(existing);
      const loom_func_type_data_t* candidate_data =
          loom_type_func_data(candidate);
      if (existing_data == NULL || candidate_data == NULL ||
          existing_data->arg_count != candidate_data->arg_count ||
          existing_data->result_count != candidate_data->result_count) {
        return false;
      }
      for (iree_host_size_t i = 0; i < ctx->dependency_count; ++i) {
        if (!loom_type_has_same_storage(
                existing_data->types[i],
                loom_type_table_get(&ctx->module->types,
                                    ctx->dependency_ids[i]))) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_DIALECT: {
      if (loom_type_dialect_name_id(existing) !=
          loom_type_dialect_name_id(candidate)) {
        return false;
      }
      const loom_type_t* existing_parameters =
          loom_type_dialect_params(existing);
      if (ctx->dependency_count > 0 && existing_parameters == NULL) {
        return false;
      }
      for (iree_host_size_t i = 0; i < ctx->dependency_count; ++i) {
        if (!loom_type_has_same_storage(
                existing_parameters[i],
                loom_type_table_get(&ctx->module->types,
                                    ctx->dependency_ids[i]))) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_REGISTER: {
      if (ctx->dependency_count == 0) {
        return loom_type_equal(existing, candidate);
      }
      const loom_register_type_data_t* existing_data =
          loom_type_register_data(existing);
      const loom_register_type_data_t* candidate_data =
          loom_type_register_data(candidate);
      if (existing_data == NULL || candidate_data == NULL ||
          existing_data->carrier_payload0 != candidate_data->carrier_payload0 ||
          existing_data->carrier_payload1 != candidate_data->carrier_payload1) {
        return false;
      }
      if (loom_type_has_same_storage(
              existing_data->value_type,
              loom_type_table_get(&ctx->module->types,
                                  ctx->dependency_ids[0]))) {
        return true;
      }
      return false;
    }
    default:
      return loom_type_equal(existing, candidate);
  }
}

// Computes loom_type_hash() without recursively hashing canonical immediate
// dependencies. Their structural hashes were recorded when the dependencies
// were interned earlier in the topological sequence.
static uint32_t loom_topological_type_hash(
    const loom_topological_type_context_t* context) {
  const loom_type_t type = context->type;
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u32(hash, type.header);
  hash = loom_structural_hash_mix_u16(hash, type.encoding_id);
  hash = loom_structural_hash_mix_u16(hash, type.encoding_flags);

  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      hash = loom_structural_hash_mix_u16(hash, data->arg_count);
      hash = loom_structural_hash_mix_u16(hash, data->result_count);
      hash = loom_structural_hash_mix_u32(hash,
                                          (uint32_t)context->dependency_count);
      for (iree_host_size_t i = 0; i < context->dependency_count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash, loom_type_table_hash(&context->module->types,
                                       context->dependency_ids[i]));
      }
      return loom_structural_hash_finalize(hash);
    }
    case LOOM_TYPE_DIALECT:
      hash =
          loom_structural_hash_mix_u32(hash, loom_type_dialect_name_id(type));
      hash = loom_structural_hash_mix_u16(hash,
                                          (uint16_t)context->dependency_count);
      for (iree_host_size_t i = 0; i < context->dependency_count; ++i) {
        hash = loom_structural_hash_mix_u32(
            hash, loom_type_table_hash(&context->module->types,
                                       context->dependency_ids[i]));
      }
      return loom_structural_hash_finalize(hash);
    case LOOM_TYPE_REGISTER: {
      if (context->dependency_count == 0) {
        return loom_type_hash(type);
      }
      const loom_register_type_data_t* data = loom_type_register_data(type);
      hash = loom_structural_hash_mix_u64(hash, data->carrier_payload0);
      hash = loom_structural_hash_mix_u64(hash, data->carrier_payload1);
      hash = loom_structural_hash_mix_u32(
          hash, loom_type_table_hash(&context->module->types,
                                     context->dependency_ids[0]));
      return loom_structural_hash_finalize(hash);
    }
    default:
      return loom_type_hash(type);
  }
}

// Builds a descriptor-backed payload from a matching parameter array. The
// caller owns any arena rollback and subsequent type-table publication.
static iree_status_t loom_module_clone_parameterized_type_payload(
    loom_module_t* module,
    const loom_parameterized_type_descriptor_t* descriptor,
    const loom_attribute_t* parameters, loom_type_t* out_type) {
  const uint8_t parameter_count = descriptor->parameter_count;
  loom_attribute_t* canonical_parameters = NULL;
  if (parameter_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, parameter_count, sizeof(*canonical_parameters),
        (void**)&canonical_parameters));
    for (uint8_t i = 0; i < parameter_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_module_canonicalize_attr_value(
          module, &descriptor->parameter_descriptors[i], parameters[i],
          /*depth=*/1, &canonical_parameters[i]));
    }
  }
  *out_type = loom_type_parameterized(descriptor, parameter_count,
                                      canonical_parameters);
  return iree_ok_status();
}

// Clones a leaf type payload into module-owned storage. Structural types are
// assembled separately from canonical immediate child IDs.
static iree_status_t loom_module_clone_type_payload(loom_module_t* module,
                                                    loom_type_t type,
                                                    loom_type_t* out_type) {
  *out_type = type;

  switch (loom_type_kind(type)) {
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      if (!descriptor) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized type has a NULL family descriptor");
      }
      uint8_t parameter_count = loom_type_parameterized_parameter_count(type);
      if (parameter_count != descriptor->parameter_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized type has %u slots but its descriptor requires %u",
            parameter_count, descriptor->parameter_count);
      }
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      if (parameter_count > 0 && !parameters) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty parameterized type has a NULL slot pointer");
      }
      return loom_module_clone_parameterized_type_payload(module, descriptor,
                                                          parameters, out_type);
    }

    default:
      break;
  }

  if (loom_type_has_inline_dims(type)) {
    return iree_ok_status();
  }

  uint8_t rank = loom_type_rank(type);
  if (rank == 0) {
    out_type->dims[0] = 0;
    out_type->dims[1] = 0;
    return iree_ok_status();
  }

  const loom_overflow_dim_t* src_dims =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  if (!src_dims) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rank-%u type has a NULL overflow dim payload",
                            rank);
  }

  loom_overflow_dim_t* cloned_dims = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, rank, sizeof(loom_overflow_dim_t), (void**)&cloned_dims));
  memcpy(cloned_dims, src_dims,
         (iree_host_size_t)rank * sizeof(loom_overflow_dim_t));
  out_type->dims[0] = (uint64_t)(uintptr_t)cloned_dims;
  out_type->dims[1] = 0;
  return iree_ok_status();
}

// Clones one temporary by-value type candidate into module-owned payload.
static iree_status_t loom_module_clone_type_from_context(
    loom_module_t* module, const void* clone_context, loom_type_t* out_type) {
  loom_type_t type = *(const loom_type_t*)clone_context;
  return loom_module_clone_type_payload(module, type, out_type);
}

// Retains only the top-level payload of a type whose nested types are already
// canonical module entries.
static iree_status_t loom_module_clone_topological_type_from_context(
    loom_module_t* module, const void* clone_context, loom_type_t* out_type) {
  const loom_topological_type_context_t* ctx =
      (const loom_topological_type_context_t*)clone_context;
  const loom_type_t type = ctx->type;
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* source_data = loom_type_func_data(type);
      iree_host_size_t allocation_size = 0;
      IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
          sizeof(loom_func_type_data_t), &allocation_size,
          IREE_STRUCT_FIELD_FAM(ctx->dependency_count, loom_type_t)));
      loom_func_type_data_t* target_data = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(&module->arena, allocation_size,
                                               (void**)&target_data));
      target_data->arg_count = source_data->arg_count;
      target_data->result_count = source_data->result_count;
      target_data->reserved = 0;
      for (iree_host_size_t i = 0; i < ctx->dependency_count; ++i) {
        target_data->types[i] =
            loom_type_table_get(&module->types, ctx->dependency_ids[i]);
      }
      *out_type = loom_type_function(target_data);
      return iree_ok_status();
    }
    case LOOM_TYPE_DIALECT: {
      loom_type_t* target_parameters = NULL;
      if (ctx->dependency_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &module->arena, ctx->dependency_count, sizeof(*target_parameters),
            (void**)&target_parameters));
      }
      for (iree_host_size_t i = 0; i < ctx->dependency_count; ++i) {
        target_parameters[i] =
            loom_type_table_get(&module->types, ctx->dependency_ids[i]);
      }
      *out_type =
          loom_type_dialect(loom_type_dialect_name_id(type),
                            (uint16_t)ctx->dependency_count, target_parameters);
      return iree_ok_status();
    }
    case LOOM_TYPE_REGISTER: {
      if (ctx->dependency_count == 0) {
        return loom_module_clone_type_payload(module, type, out_type);
      }
      const loom_register_type_data_t* source_data =
          loom_type_register_data(type);
      loom_register_type_data_t* target_data = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          &module->arena, sizeof(*target_data), (void**)&target_data));
      *target_data = (loom_register_type_data_t){
          .carrier_payload0 = source_data->carrier_payload0,
          .carrier_payload1 = source_data->carrier_payload1,
          .value_type =
              loom_type_table_get(&module->types, ctx->dependency_ids[0]),
      };
      *out_type = loom_type_register_payload_with_value_type(target_data);
      return iree_ok_status();
    }
    default:
      return loom_module_clone_type_payload(module, type, out_type);
  }
}

// Looks up or inserts one type candidate in module->type_intern. Candidate
// cloning only runs on a miss so repeated structural hits do not allocate new
// module payloads.
static iree_status_t loom_module_intern_type_impl(
    loom_module_t* module, uint32_t hash, loom_intern_equal_fn_t equal_fn,
    const void* equal_context, loom_module_type_clone_fn_t clone_fn,
    const void* clone_context, loom_type_dependency_id_t dependencies,
    loom_type_t* out_interned_type, loom_type_id_t* out_type_id,
    bool* out_miss) {
  if (out_miss) {
    *out_miss = false;
  }
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &module->type_intern, hash, equal_fn, equal_context);
  const uint32_t existing_index = probe.index;
  if (existing_index != UINT32_MAX) {
    *out_interned_type = loom_type_table_get(&module->types, existing_index);
    if (out_type_id) {
      *out_type_id = (loom_type_id_t)existing_index;
    }
    loom_module_note_recent_exact_type(module, (loom_type_id_t)existing_index);
    return iree_ok_status();
  }
  if (out_miss) {
    *out_miss = true;
  }

  // Type interner slots use 32-bit indices with UINT32_MAX as the empty
  // sentinel in loom_intern_table_t. Reject a new unique type before that
  // sentinel value could be used as a real table index.
  if (module->types.count >= LOOM_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "type table full (%" PRIhsz " entries, max id %u)",
                            module->types.count,
                            (unsigned)(LOOM_TYPE_ID_INVALID - 1));
  }

  loom_type_t type = {0};
  IREE_RETURN_IF_ERROR(clone_fn(module, clone_context, &type));
  IREE_RETURN_IF_ERROR(loom_type_dependencies_collect(
      module, type, dependencies, &dependencies));

  loom_type_identity_insertion_t identity_insertion;
  IREE_RETURN_IF_ERROR(
      loom_type_identity_prepare(module, type, &identity_insertion));

  // Payload preparation does not intern types. Only hash-table growth can
  // invalidate the vacant slot found by the initial probe.
  iree_host_size_t slot = probe.slot;
  if (module->types.count == loom_type_table_capacity(&module->types) ||
      !loom_intern_table_has_insert_capacity(&module->type_intern)) {
    IREE_RETURN_IF_ERROR(loom_module_reserve_type_insert(module, hash, &slot));
  }

  uint32_t new_index = (uint32_t)module->types.count;
  loom_type_segment_t* segment =
      (loom_type_segment_t*)loom_segmented_storage_segment(
          &module->types.segments, new_index >> LOOM_TYPE_SEGMENT_SHIFT);
  const uint32_t row = new_index & LOOM_TYPE_SEGMENT_MASK;
  segment->entries[row] = type;
  segment->hashes[row] = hash;
  segment->dependencies[row] = dependencies;
  module->types.count++;
  loom_type_identity_commit(module, &identity_insertion);
  loom_intern_table_insert(&module->type_intern, slot, hash, new_index);
  *out_interned_type = type;
  if (out_type_id) {
    *out_type_id = (loom_type_id_t)new_index;
  }
  loom_module_note_recent_exact_type(module, (loom_type_id_t)new_index);
  return iree_ok_status();
}

// Retains a module-owned type candidate by value on an interner miss.
static iree_status_t loom_module_retain_type_from_context(
    loom_module_t* module, const void* clone_context, loom_type_t* out_type) {
  (void)module;
  *out_type = *(const loom_type_t*)clone_context;
  return iree_ok_status();
}

// Erases the redundant static spelling of the native dense representation
// before type hashing and interning. Dynamic attachments remain explicit SSA
// references, and all non-default static families retain their module IDs.
static loom_type_t loom_module_canonicalize_shaped_type_attachment(
    loom_type_t type, const loom_encoding_t* encoding) {
  if (!loom_encoding_is_implicit_shaped_attachment(encoding)) {
    return type;
  }
  type.encoding_id = 0;
  type.encoding_flags = 0;
  return type;
}

loom_type_id_t loom_module_lookup_type_id(const loom_module_t* module,
                                          loom_type_t type) {
  if (loom_type_has_static_encoding(type)) {
    const loom_encoding_t* encoding =
        loom_module_encoding(module, type.encoding_id);
    if (!encoding) {
      return LOOM_TYPE_ID_INVALID;
    }
    type = loom_module_canonicalize_shaped_type_attachment(type, encoding);
  }
  loom_type_id_t type_id = loom_module_find_recent_exact_type(module, type);
  if (type_id == LOOM_TYPE_ID_INVALID) {
    type_id = loom_type_identity_find(module, type);
  }
  if (type_id != LOOM_TYPE_ID_INVALID) {
    return type_id;
  }
  const loom_type_equal_context_t context = {module, type};
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &module->type_intern, loom_type_hash(type), loom_type_equal_fn, &context);
  return probe.index == UINT32_MAX ? LOOM_TYPE_ID_INVALID
                                   : (loom_type_id_t)probe.index;
}

loom_type_id_t loom_module_lookup_topological_type_id(
    const loom_module_t* module, loom_type_t type,
    const loom_type_id_t* structural_dependency_ids,
    iree_host_size_t structural_dependency_count) {
  if (loom_type_has_static_encoding(type)) {
    const loom_encoding_t* encoding =
        loom_module_encoding(module, type.encoding_id);
    if (!encoding) {
      return LOOM_TYPE_ID_INVALID;
    }
    type = loom_module_canonicalize_shaped_type_attachment(type, encoding);
  }
  const loom_topological_type_context_t context = {
      .module = module,
      .type = type,
      .dependency_ids = structural_dependency_ids,
      .dependency_count = structural_dependency_count,
  };
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &module->type_intern, loom_topological_type_hash(&context),
      loom_topological_type_equal_fn, &context);
  return probe.index == UINT32_MAX ? LOOM_TYPE_ID_INVALID
                                   : (loom_type_id_t)probe.index;
}

// Retains each shaped scalar dependency once without repeating hash-table
// queries as distinct shaped types reuse the same element kind.
static iree_status_t loom_module_intern_shaped_element_type(
    loom_module_t* module, loom_type_t type) {
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (loom_scalar_type_set_contains(module->shaped_element_types,
                                    element_type)) {
    return iree_ok_status();
  }
  loom_type_t interned_element_type = {0};
  IREE_RETURN_IF_ERROR(loom_module_intern_type(
      module, loom_type_scalar(element_type), &interned_element_type));
  // Construction can retain invalid scalar ordinals for verification. Only
  // concrete scalar types belong to the compact presence set.
  if (loom_scalar_type_is_valid(element_type)) {
    module->shaped_element_types |=
        (loom_scalar_type_set_t)(1u << element_type);
  }
  return iree_ok_status();
}

iree_status_t loom_module_intern_topological_type_id(
    loom_module_t* module, loom_type_t type,
    const loom_type_id_t* structural_dependency_ids,
    iree_host_size_t structural_dependency_count, loom_type_id_t* out_type_id) {
  if (out_type_id == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type id output is NULL");
  }
  *out_type_id = LOOM_TYPE_ID_INVALID;

  if (loom_type_has_static_encoding(type)) {
    type = loom_module_canonicalize_shaped_type_attachment(
        type, loom_module_encoding(module, type.encoding_id));
  }
  iree_host_size_t expected_dependency_count = 0;
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      // Shaped wire records name their scalar element inline. Selective
      // readers may not have reached a separate scalar table entry.
      IREE_RETURN_IF_ERROR(
          loom_module_intern_shaped_element_type(module, type));
      break;
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      IREE_ASSERT(data != NULL);
      expected_dependency_count =
          (iree_host_size_t)data->arg_count + data->result_count;
      break;
    }
    case LOOM_TYPE_DIALECT:
      expected_dependency_count = loom_type_dialect_param_count(type);
      break;
    case LOOM_TYPE_REGISTER:
      expected_dependency_count =
          loom_type_register_has_value_type(type) ? 1 : 0;
      break;
    default:
      break;
  }
  IREE_ASSERT(expected_dependency_count == structural_dependency_count);
  IREE_ASSERT(structural_dependency_count == 0 ||
              structural_dependency_ids != NULL);
  loom_type_dependency_id_t dependencies = 0;
  // No canonical child can have a nonempty set before the first index exists.
  // Static-only modules therefore need no dependency fold over their children.
  if (module->type_uses.index) {
    for (iree_host_size_t i = 0; i < structural_dependency_count; ++i) {
      IREE_ASSERT(structural_dependency_ids[i] < module->types.count);
      IREE_RETURN_IF_ERROR(loom_type_dependencies_union(
          &module->type_uses, dependencies,
          loom_type_table_dependencies(&module->types,
                                       structural_dependency_ids[i]),
          &dependencies));
    }
  }

  const loom_topological_type_context_t context = {
      .module = module,
      .type = type,
      .dependency_ids = structural_dependency_ids,
      .dependency_count = structural_dependency_count,
  };
  const uint32_t hash = loom_topological_type_hash(&context);
  loom_type_t interned_type = {0};
  iree_status_t status = loom_module_intern_type_impl(
      module, hash, loom_topological_type_equal_fn, &context,
      loom_module_clone_topological_type_from_context, &context, dependencies,
      &interned_type, out_type_id, /*out_miss=*/NULL);
  if (iree_status_is_ok(status) && loom_type_is_register(interned_type) &&
      loom_type_register_has_value_type(interned_type)) {
    loom_module_note_recent_register_type(module, *out_type_id);
  }
  return status;
}

static iree_status_t loom_module_intern_type_with_dependencies(
    loom_module_t* module, loom_type_t type, loom_type_t* out_interned_type,
    loom_type_id_t* out_type_id) {
  if (loom_type_has_static_encoding(type)) {
    const loom_encoding_t* encoding =
        loom_module_encoding(module, type.encoding_id);
    if (!encoding) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static encoding id %u is out of range for module with %" PRIhsz
          " encodings",
          (unsigned)type.encoding_id, module->encodings.count);
    }
    type = loom_module_canonicalize_shaped_type_attachment(type, encoding);
  }
  loom_type_id_t recent_type_id =
      loom_module_find_recent_exact_type(module, type);
  if (recent_type_id == LOOM_TYPE_ID_INVALID) {
    recent_type_id = loom_type_identity_find(module, type);
  }
  if (recent_type_id != LOOM_TYPE_ID_INVALID) {
    *out_interned_type = loom_type_table_get(&module->types, recent_type_id);
    if (out_type_id) {
      *out_type_id = recent_type_id;
    }
    return iree_ok_status();
  }
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION:
    case LOOM_TYPE_DIALECT:
    case LOOM_TYPE_REGISTER: {
      if (loom_type_kind(type) == LOOM_TYPE_REGISTER &&
          !loom_type_register_has_value_type(type)) {
        break;
      }
      loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_type_import(module, type, &type_id));
      *out_interned_type = loom_type_table_get(&module->types, type_id);
      if (out_type_id) {
        *out_type_id = type_id;
      }
      return iree_ok_status();
    }
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      IREE_RETURN_IF_ERROR(
          loom_module_intern_shaped_element_type(module, type));
      break;
    }
    default:
      break;
  }

  const uint32_t hash = loom_type_hash(type);
  const loom_type_equal_context_t equal_context = {module, type};
  return loom_module_intern_type_impl(
      module, hash, loom_type_equal_fn, &equal_context,
      loom_module_clone_type_from_context, &type, /*dependencies=*/0,
      out_interned_type, out_type_id, /*out_miss=*/NULL);
}

iree_status_t loom_module_intern_type(loom_module_t* module, loom_type_t type,
                                      loom_type_t* out_interned_type) {
  return loom_module_intern_type_with_dependencies(
      module, type, out_interned_type, /*out_type_id=*/NULL);
}

iree_status_t loom_module_intern_type_id(loom_module_t* module,
                                         loom_type_t type,
                                         loom_type_id_t* out_type_id) {
  if (!out_type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type id output is NULL");
  }
  *out_type_id = LOOM_TYPE_ID_INVALID;
  loom_type_t interned_type = {0};
  return loom_module_intern_type_with_dependencies(module, type, &interned_type,
                                                   out_type_id);
}

iree_status_t loom_module_make_parameterized_type(
    loom_module_t* module,
    const loom_parameterized_type_descriptor_t* descriptor,
    const loom_attribute_t* parameters, iree_host_size_t parameter_count,
    loom_type_t* out_type, loom_type_id_t* out_type_id) {
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameterized type descriptor is NULL");
  }
  if (parameter_count != descriptor->parameter_count) {
    iree_string_view_t type_name = loom_bstring_view(descriptor->name);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameterized type '%.*s' has %" PRIhsz
                            " slots but its descriptor requires %u",
                            (int)type_name.size, type_name.data,
                            parameter_count, descriptor->parameter_count);
  }
  if (parameter_count > 0 && !parameters) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty parameterized type has a NULL slot pointer");
  }

  if (descriptor->ir_kind != LOOM_TYPE_PARAMETERIZED) {
    if (!loom_type_kind_is_valid(descriptor->ir_kind) ||
        descriptor->ir_kind == LOOM_TYPE_DIALECT || parameter_count != 1 ||
        descriptor->parameter_descriptors[0].attr_kind != LOOM_ATTR_ENUM) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "compact parameterized type descriptor has an "
                              "invalid representation");
    }
    IREE_RETURN_IF_ERROR(loom_module_validate_attr_descriptor_value(
        &descriptor->parameter_descriptors[0], parameters[0]));
    const uint8_t payload = loom_attr_is_absent(parameters[0])
                                ? 0
                                : loom_attr_as_enum(parameters[0]);
    loom_type_t type = {0};
    type.header = loom_type_make_raw_header(descriptor->ir_kind, payload, 0,
                                            descriptor->type_flags);
    if (out_type_id) {
      return loom_module_intern_type_with_dependencies(module, type, out_type,
                                                       out_type_id);
    }
    *out_type = type;
    return iree_ok_status();
  }

  iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(&module->arena);
  loom_type_t type = {0};
  iree_status_t status = loom_module_clone_parameterized_type_payload(
      module, descriptor, parameters, &type);
  bool interner_miss = false;
  if (iree_status_is_ok(status)) {
    uint32_t hash = loom_type_hash(type);
    loom_type_equal_context_t equal_context = {module, type};
    status = loom_module_intern_type_impl(
        module, hash, loom_type_equal_fn, &equal_context,
        loom_module_retain_type_from_context, &type, /*dependencies=*/0,
        out_type, out_type_id, &interner_miss);
  }
  if (!iree_status_is_ok(status) || !interner_miss) {
    iree_arena_checkpoint_restore(&checkpoint);
  }
  return status;
}

iree_status_t loom_module_intern_function_type(loom_module_t* module,
                                               const loom_type_t* arg_types,
                                               uint16_t arg_count,
                                               const loom_type_t* result_types,
                                               uint16_t result_count,
                                               loom_type_t* out_interned_type) {
  if (arg_count > 0 && !arg_types) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function type has %u args but a NULL arg type payload",
        (unsigned)arg_count);
  }
  if (result_count > 0 && !result_types) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function type has %u results but a NULL result type payload",
        (unsigned)result_count);
  }
  loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_type_import_function(
      module, arg_types, arg_count, result_types, result_count, &type_id));
  *out_interned_type = loom_type_table_get(&module->types, type_id);
  return iree_ok_status();
}

iree_status_t loom_module_intern_register_type(loom_module_t* module,
                                               uint64_t carrier_payload0,
                                               uint64_t carrier_payload1,
                                               loom_type_t value_type,
                                               loom_type_t* out_interned_type) {
  loom_register_type_data_t data = {
      .carrier_payload0 = carrier_payload0,
      .carrier_payload1 = carrier_payload1,
      .value_type = value_type,
  };
  const loom_type_t type = loom_type_register_payload_with_value_type(&data);
  loom_type_id_t type_id =
      loom_module_find_recent_register_type_structural(module, type);
  if (type_id == LOOM_TYPE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_intern_type_with_dependencies(
        module, type, out_interned_type, &type_id));
  } else {
    *out_interned_type = loom_type_table_get(&module->types, type_id);
  }
  loom_module_note_recent_register_type(module, type_id);
  loom_module_note_recent_exact_type(module, type_id);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block and region creation
//===----------------------------------------------------------------------===//

iree_status_t loom_module_allocate_block(loom_module_t* module,
                                         loom_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&module->arena, sizeof(loom_block_t),
                              (void**)&block));
  iree_status_t status = loom_module_initialize_block(module, block);
  if (iree_status_is_ok(status)) {
    *out_block = block;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_module_allocate_region(loom_module_t* module,
                                          uint16_t block_count,
                                          loom_region_t** out_region) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_region_t* region = NULL;
  iree_status_t status = iree_arena_allocate(
      &module->arena, sizeof(loom_region_t), (void**)&region);
  if (iree_status_is_ok(status)) {
    memset(region, 0, sizeof(loom_region_t));
    region->blocks = region->inline_blocks;
    region->inline_blocks[0] = &region->entry_block;
    region->block_capacity = 1;
    status = loom_module_initialize_block(module, &region->entry_block);
    region->entry_block.parent_region = region;
    if (block_count > 0) {
      region->entry_block.region_index = 0;
    }
  }
  if (iree_status_is_ok(status) && block_count > 1) {
    region->block_capacity = block_count;
    status = iree_arena_allocate_array(&module->arena, block_count,
                                       sizeof(loom_block_t*),
                                       (void**)&region->blocks);
    if (iree_status_is_ok(status)) {
      memset(region->blocks, 0,
             (iree_host_size_t)block_count * sizeof(loom_block_t*));
      region->blocks[0] = &region->entry_block;
    }
    for (uint16_t i = 1; i < block_count && iree_status_is_ok(status); ++i) {
      status = loom_module_allocate_block(module, &region->blocks[i]);
      if (iree_status_is_ok(status)) {
        region->blocks[i]->parent_region = region;
        region->blocks[i]->region_index = i;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    region->block_count = block_count;
    *out_region = region;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_region_append_block(loom_module_t* module,
                                       loom_region_t* region,
                                       loom_block_t** out_block) {
  IREE_RETURN_IF_ERROR(loom_region_blocks_ensure_capacity(module, region));
  loom_block_t* block = &region->entry_block;
  if (region->block_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_allocate_block(module, &block));
  }
  block->parent_region = region;
  block->region_index = region->block_count;
  region->blocks[region->block_count] = block;
  ++region->block_count;
  *out_block = block;
  return iree_ok_status();
}

iree_status_t loom_region_insert_block(loom_module_t* module,
                                       loom_region_t* region,
                                       uint16_t block_index,
                                       loom_block_t** out_block) {
  IREE_ASSERT_ARGUMENT(region);
  if (block_index > region->block_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "block insertion index %u out of range for %u block(s)",
        (unsigned)block_index, (unsigned)region->block_count);
  }
  if (region->block_count > 0 && block_index == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot insert before a region entry block");
  }

  IREE_RETURN_IF_ERROR(loom_region_blocks_ensure_capacity(module, region));
  loom_block_t* block = &region->entry_block;
  if (region->block_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_allocate_block(module, &block));
  }
  block->parent_region = region;
  block->region_index = block_index;
  for (uint16_t i = region->block_count; i > block_index; --i) {
    region->blocks[i] = region->blocks[i - 1];
    region->blocks[i]->region_index = i;
  }
  region->blocks[block_index] = block;
  ++region->block_count;
  *out_block = block;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block arguments
//===----------------------------------------------------------------------===//

static iree_status_t loom_block_arg_ids_ensure_capacity(
    loom_module_t* module, loom_block_t* block, uint32_t required_count) {
  if (required_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "block arg count exceeds UINT16_MAX");
  }
  if (required_count > block->arg_capacity) {
    iree_host_size_t doubled_capacity =
        (iree_host_size_t)block->arg_capacity * 2;
    iree_host_size_t new_capacity =
        doubled_capacity > 4 ? doubled_capacity : (iree_host_size_t)4;
    iree_host_size_t required_capacity = required_count;
    if (new_capacity < required_capacity) {
      new_capacity = required_capacity;
    }
    if (new_capacity > UINT16_MAX) {
      new_capacity = UINT16_MAX;
    }
    loom_value_id_t* new_arg_ids = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, new_capacity,
                                                   sizeof(*new_arg_ids),
                                                   (void**)&new_arg_ids));
    if (block->arg_count > 0) {
      memcpy(new_arg_ids, block->arg_ids,
             (iree_host_size_t)block->arg_count * sizeof(*new_arg_ids));
    }
    block->arg_ids = new_arg_ids;
    block->arg_capacity = (uint16_t)new_capacity;
  }
  return iree_ok_status();
}

iree_status_t loom_block_add_arg(loom_module_t* module, loom_block_t* block,
                                 loom_value_id_t value_id) {
  return loom_block_insert_arg(module, block, block->arg_count, value_id);
}

iree_status_t loom_block_insert_arg(loom_module_t* module, loom_block_t* block,
                                    uint16_t arg_index,
                                    loom_value_id_t value_id) {
  if (arg_index > block->arg_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "block argument insertion index %u out of range for %u argument(s)",
        (unsigned)arg_index, (unsigned)block->arg_count);
  }
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %%%u out of range (module has %" PRIhsz
                            " values)",
                            (unsigned)value_id, module->values.count);
  }

  const uint32_t required_count = (uint32_t)block->arg_count + 1;
  IREE_RETURN_IF_ERROR(
      loom_block_arg_ids_ensure_capacity(module, block, required_count));

  for (uint16_t i = block->arg_count; i > arg_index; --i) {
    const loom_value_id_t shifted_id = block->arg_ids[i - 1];
    block->arg_ids[i] = shifted_id;
    if (shifted_id == LOOM_VALUE_ID_INVALID ||
        shifted_id >= module->values.count) {
      continue;
    }
    loom_value_t* shifted = loom_module_value(module, shifted_id);
    if (loom_value_is_block_arg(shifted) &&
        loom_value_def_block(shifted) == block) {
      shifted->def = loom_value_def_make_block(block, i);
    }
  }

  ++block->arg_count;
  block->arg_ids[arg_index] = value_id;

  loom_value_t* value = loom_module_value(module, value_id);
  value->flags |= LOOM_VALUE_FLAG_BLOCK_ARG;
  value->def = loom_value_def_make_block(block, arg_index);
  return iree_ok_status();
}

iree_status_t loom_block_remove_arg(loom_module_t* module, loom_block_t* block,
                                    uint16_t arg_index) {
  if (arg_index >= block->arg_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "block argument index %u out of range for %u argument(s)",
        (unsigned)arg_index, (unsigned)block->arg_count);
  }

  loom_value_id_t value_id = block->arg_ids[arg_index];
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block argument %u value %%%u is invalid",
                            (unsigned)arg_index, (unsigned)value_id);
  }

  loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value) || loom_value_def_block(value) != block ||
      loom_value_def_index(value) != arg_index) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "value %%%u is not block argument %u of the target block",
        (unsigned)value_id, (unsigned)arg_index);
  }
  if (value->use_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot remove block argument %%%u with %u operand use(s)",
        (unsigned)value_id, (unsigned)value->use_count);
  }
  if (loom_module_value_has_type_uses(module, value_id)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot remove block argument %%%u with incoming type use(s)",
        (unsigned)value_id);
  }
  if (loom_value_has_attribute_uses(value)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot remove block argument %%%u with attribute use(s)",
        (unsigned)value_id);
  }

  loom_module_drop_value_type_uses(module, value_id);
  value->flags &= ~LOOM_VALUE_FLAG_BLOCK_ARG;
  value->def = loom_value_def_make_none();

  for (uint16_t i = (uint16_t)(arg_index + 1); i < block->arg_count; ++i) {
    loom_value_id_t shifted_id = block->arg_ids[i];
    block->arg_ids[i - 1] = shifted_id;
    if (shifted_id == LOOM_VALUE_ID_INVALID ||
        shifted_id >= module->values.count) {
      continue;
    }
    loom_value_t* shifted = loom_module_value(module, shifted_id);
    if (loom_value_is_block_arg(shifted) &&
        loom_value_def_block(shifted) == block) {
      shifted->def = loom_value_def_make_block(block, (uint16_t)(i - 1));
    }
  }

  --block->arg_count;
  block->arg_ids[block->arg_count] = LOOM_VALUE_ID_INVALID;
  return iree_ok_status();
}

uint16_t loom_block_remove_args(loom_module_t* module, loom_block_t* block,
                                const bool* remove_args,
                                uint16_t remove_arg_count) {
  IREE_ASSERT_EQ(remove_arg_count, block->arg_count);

  uint16_t removed_count = 0;
  for (uint16_t arg_index = 0; arg_index < remove_arg_count; ++arg_index) {
    if (!remove_args[arg_index]) {
      continue;
    }
    const loom_value_id_t value_id = block->arg_ids[arg_index];
    IREE_ASSERT_NE(value_id, LOOM_VALUE_ID_INVALID);
    IREE_ASSERT_LT(value_id, module->values.count);
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_ASSERT(loom_value_is_block_arg(value));
    IREE_ASSERT_EQ(loom_value_def_block(value), block);
    IREE_ASSERT_EQ(loom_value_def_index(value), arg_index);
    IREE_ASSERT_EQ(value->use_count, 0u);
    IREE_ASSERT(!loom_value_has_attribute_uses(value));

#ifndef NDEBUG
    loom_type_use_iterator_t users;
    loom_module_value_type_users(module, value_id, &users);
    for (loom_value_id_t carrier = loom_type_users_next(&users);
         carrier != LOOM_VALUE_ID_INVALID;
         carrier = loom_type_users_next(&users)) {
      const loom_value_t* carrier_value = loom_module_value(module, carrier);
      IREE_ASSERT(loom_value_is_block_arg(carrier_value));
      IREE_ASSERT_EQ(loom_value_def_block(carrier_value), block);
      IREE_ASSERT(remove_args[loom_value_def_index(carrier_value)]);
    }
#endif  // !NDEBUG
    ++removed_count;
  }
  if (removed_count == 0) {
    return 0;
  }

  for (uint16_t arg_index = 0; arg_index < remove_arg_count; ++arg_index) {
    if (remove_args[arg_index]) {
      loom_module_drop_value_type_uses(module, block->arg_ids[arg_index]);
    }
  }

  uint16_t kept_count = 0;
  for (uint16_t arg_index = 0; arg_index < remove_arg_count; ++arg_index) {
    const loom_value_id_t value_id = block->arg_ids[arg_index];
    loom_value_t* value = loom_module_value(module, value_id);
    if (remove_args[arg_index]) {
      IREE_ASSERT(!loom_module_value_has_type_uses(module, value_id));
      value->flags &= ~LOOM_VALUE_FLAG_BLOCK_ARG;
      value->def = loom_value_def_make_none();
      continue;
    }
    block->arg_ids[kept_count] = value_id;
    value->def = loom_value_def_make_block(block, kept_count);
    ++kept_count;
  }
  for (uint16_t arg_index = kept_count; arg_index < remove_arg_count;
       ++arg_index) {
    block->arg_ids[arg_index] = LOOM_VALUE_ID_INVALID;
  }
  block->arg_count = kept_count;
  return removed_count;
}

//===----------------------------------------------------------------------===//
// Block op insertion
//===----------------------------------------------------------------------===//

#define LOOM_BLOCK_ORDINAL_STRIDE UINT64_C(0x100000000)
// A directional repair always has at least half of the ordinal arena. Limiting
// the number of gaps to floor(sqrt(UINT64_MAX / 2)) guarantees that the density
// search finds a repair window before reaching its sentinel.
#define LOOM_BLOCK_ORDINAL_MAX_OP_COUNT UINT32_C(3037000499)
static_assert((uint64_t)LOOM_BLOCK_ORDINAL_MAX_OP_COUNT *
                      LOOM_BLOCK_ORDINAL_MAX_OP_COUNT <=
                  UINT64_MAX / 2,
              "block ordinal capacity exceeds the directional repair bound");
static_assert(
    (uint64_t)(LOOM_BLOCK_ORDINAL_MAX_OP_COUNT + 1) *
            (LOOM_BLOCK_ORDINAL_MAX_OP_COUNT + 1) >
        UINT64_MAX / 2,
    "block ordinal capacity must use the full directional repair bound");

static uint64_t loom_block_ordinal_midpoint(uint64_t lower, uint64_t upper) {
  return lower + (upper - lower) / 2;
}

// Chooses the side of a sparse gap that preserves room for a repeated
// insertion stream. The denser outer gap identifies the endpoint that moved on
// the preceding insertion; placing beside it leaves the current gap available
// beside the stable endpoint. Equal-density gaps split at the midpoint.
static uint64_t loom_block_choose_ordinal(const loom_op_t* prev_op,
                                          const loom_op_t* next_op,
                                          uint64_t lower, uint64_t upper) {
  const uint64_t previous_gap =
      prev_op ? lower - (prev_op->prev_op ? prev_op->prev_op->block_ordinal : 0)
              : UINT64_MAX;
  const uint64_t next_gap =
      next_op
          ? (next_op->next_op ? next_op->next_op->block_ordinal : UINT64_MAX) -
                upper
          : UINT64_MAX;
  if (previous_gap < next_gap) {
    return lower + 1;
  }
  if (next_gap < previous_gap) {
    return upper - 1;
  }
  return loom_block_ordinal_midpoint(lower, upper);
}

// Redistributes |count| successive ops across |gap_count| equal-width gaps.
// Quotient/remainder accumulation avoids an overflowing distance * index
// intermediate while producing floor(distance * index / gap_count).
IREE_ATTRIBUTE_NOINLINE static void loom_block_redistribute_ordinals_forward(
    loom_op_t* first_op, uint32_t count, uint64_t lower, uint64_t distance,
    uint32_t gap_count) {
  const uint64_t quotient = distance / gap_count;
  const uint64_t remainder = distance % gap_count;
  uint64_t ordinal = lower;
  uint64_t remainder_accumulator = 0;
  loom_op_t* op = first_op;
  for (uint32_t i = 0; i < count; ++i, op = op->next_op) {
    ordinal += quotient;
    remainder_accumulator += remainder;
    if (remainder_accumulator >= gap_count) {
      ++ordinal;
      remainder_accumulator -= gap_count;
    }
    op->block_ordinal = ordinal;
  }
}

// Opens a congested gap by redistributing a forward run. The first run
// whose ordinal span exceeds the square of its gap count has enough density
// slack to pay for the relabeling amortized across the insertions that consumed
// it. The supported block-size limit guarantees that a run exists.
static uint64_t loom_block_repair_ordinal_gap_forward(loom_op_t* prev_op,
                                                      loom_op_t* next_op) {
  const uint64_t lower = prev_op ? prev_op->block_ordinal : 0;
  loom_op_t* boundary_op = next_op;
  uint32_t gap_count = 1;
  while (true) {
    const uint64_t upper =
        boundary_op ? boundary_op->block_ordinal : UINT64_MAX;
    const uint64_t distance = upper - lower;
    if (distance > (uint64_t)gap_count * gap_count) {
      loom_block_redistribute_ordinals_forward(next_op, gap_count - 1, lower,
                                               distance, gap_count);
      return loom_block_choose_ordinal(
          prev_op, next_op, lower,
          next_op ? next_op->block_ordinal : UINT64_MAX);
    }
    IREE_ASSERT(boundary_op);
    boundary_op = boundary_op->next_op;
    ++gap_count;
  }
}

// Opens a congested gap by redistributing a backward run.
static uint64_t loom_block_repair_ordinal_gap_backward(loom_block_t* block,
                                                       loom_op_t* prev_op,
                                                       loom_op_t* next_op) {
  const uint64_t upper = next_op ? next_op->block_ordinal : UINT64_MAX;
  loom_op_t* boundary_op = prev_op;
  uint32_t gap_count = 1;
  while (true) {
    const uint64_t lower = boundary_op ? boundary_op->block_ordinal : 0;
    const uint64_t distance = upper - lower;
    if (distance > (uint64_t)gap_count * gap_count) {
      loom_op_t* first_op =
          boundary_op ? boundary_op->next_op : block->first_op;
      loom_block_redistribute_ordinals_forward(first_op, gap_count - 1, lower,
                                               distance, gap_count);
      return loom_block_choose_ordinal(
          prev_op, next_op, prev_op ? prev_op->block_ordinal : 0, upper);
    }
    IREE_ASSERT(boundary_op);
    boundary_op = boundary_op->prev_op;
    ++gap_count;
  }
}

// Repairs an exhausted insertion gap. Kept out of line so the sparse-gap and
// append paths do not carry the density search and redistribution machinery.
IREE_ATTRIBUTE_NOINLINE static uint64_t loom_block_repair_ordinal_gap(
    loom_block_t* block, loom_op_t* prev_op, loom_op_t* next_op) {
  const uint64_t lower = prev_op ? prev_op->block_ordinal : 0;
  if (lower < UINT64_MAX / 2) {
    return loom_block_repair_ordinal_gap_forward(prev_op, next_op);
  }
  return loom_block_repair_ordinal_gap_backward(block, prev_op, next_op);
}

static uint64_t loom_block_insert_ordinal(loom_block_t* block,
                                          loom_op_t* prev_op,
                                          loom_op_t* next_op) {
  if (!block->last_op) {
    return LOOM_BLOCK_ORDINAL_STRIDE;
  }

  const uint64_t lower = prev_op ? prev_op->block_ordinal : 0;
  const uint64_t upper = next_op ? next_op->block_ordinal : UINT64_MAX;
  if (upper - lower > 1) {
    if (!next_op && upper - lower > LOOM_BLOCK_ORDINAL_STRIDE) {
      return lower + LOOM_BLOCK_ORDINAL_STRIDE;
    }
    return loom_block_choose_ordinal(prev_op, next_op, lower, upper);
  }
  return loom_block_repair_ordinal_gap(block, prev_op, next_op);
}

static iree_status_t loom_block_link_op_between(loom_module_t* module,
                                                loom_block_t* block,
                                                loom_op_t* prev_op,
                                                loom_op_t* next_op,
                                                loom_op_t* op) {
  (void)module;
  if (block->op_count >= LOOM_BLOCK_ORDINAL_MAX_OP_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "block operation count exceeds sparse ordinal capacity %u",
        (unsigned)LOOM_BLOCK_ORDINAL_MAX_OP_COUNT);
  }
  if (op->parent_block && op->parent_block != block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot insert op already owned by a different block");
  }
  if (op->prev_op || op->next_op || block->first_op == op ||
      block->last_op == op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot insert op already linked into a block");
  }

  const uint64_t ordinal = loom_block_insert_ordinal(block, prev_op, next_op);

  op->parent_block = block;
  op->block_ordinal = ordinal;
  op->prev_op = prev_op;
  op->next_op = next_op;
  if (op->successor_count > 0 && block->parent_region) {
    block->parent_region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  }

  if (prev_op) {
    prev_op->next_op = op;
  } else {
    block->first_op = op;
  }
  if (next_op) {
    next_op->prev_op = op;
  } else {
    block->last_op = op;
  }
  ++block->op_count;
  return iree_ok_status();
}

iree_status_t loom_block_append_op(loom_module_t* module, loom_block_t* block,
                                   loom_op_t* op) {
  return loom_block_link_op_between(module, block, block->last_op, NULL, op);
}

iree_status_t loom_block_insert_before_op(loom_module_t* module,
                                          loom_block_t* block,
                                          loom_op_t* before_op, loom_op_t* op) {
  if (!before_op) {
    return loom_block_append_op(module, block, op);
  }
  if (before_op->parent_block != block ||
      iree_any_bit_set(before_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot insert before op that is not live in the target block");
  }
  return loom_block_link_op_between(module, block, before_op->prev_op,
                                    before_op, op);
}

iree_status_t loom_block_insert_op(loom_module_t* module, loom_block_t* block,
                                   iree_host_size_t index, loom_op_t* op) {
  if (index >= block->op_count) {
    return loom_block_append_op(module, block, op);
  }
  loom_op_t* before_op = block->first_op;
  for (iree_host_size_t i = 0; i < index; ++i) {
    before_op = before_op->next_op;
  }
  return loom_block_insert_before_op(module, block, before_op, op);
}

void loom_block_unlink_op(loom_module_t* module, loom_op_t* op) {
  loom_block_t* block = op->parent_block;
  if (!block || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return;
  }
  loom_module_drop_op_summaries(module, op);

  if (op->prev_op) {
    op->prev_op->next_op = op->next_op;
  } else if (block->first_op == op) {
    block->first_op = op->next_op;
  }

  if (op->next_op) {
    op->next_op->prev_op = op->prev_op;
  } else if (block->last_op == op) {
    block->last_op = op->prev_op;
  }

  op->prev_op = NULL;
  op->next_op = NULL;
  op->block_ordinal = 0;
  if (block->op_count > 0) {
    --block->op_count;
  }
}

iree_host_size_t loom_block_find_op(const loom_block_t* block,
                                    const loom_op_t* op) {
  iree_host_size_t index = 0;
  const loom_op_t* current = NULL;
  loom_block_for_each_op(block, current) {
    if (current == op) {
      return index;
    }
    ++index;
  }
  return IREE_HOST_SIZE_MAX;
}
