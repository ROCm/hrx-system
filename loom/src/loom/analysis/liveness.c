// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/liveness_dataflow.h"
#include "loom/analysis/liveness_events.h"
#include "loom/analysis/liveness_pressure.h"
#include "loom/analysis/liveness_uses.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"
#include "loom/target/registers.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/segmented_storage.h"

#define LOOM_LIVENESS_OPERATION_USES_PER_SEGMENT 1024u
#define LOOM_LIVENESS_OPERATION_USE_SEGMENT_SHIFT 10u
#define LOOM_LIVENESS_OPERATION_USE_SEGMENT_MASK \
  (LOOM_LIVENESS_OPERATION_USES_PER_SEGMENT - 1u)

static_assert((1u << LOOM_LIVENESS_OPERATION_USE_SEGMENT_SHIFT) ==
                  LOOM_LIVENESS_OPERATION_USES_PER_SEGMENT,
              "liveness use segment capacity must match its index shift");

typedef struct loom_liveness_operation_use_segment_t {
  loom_value_ordinal_t ordinals[LOOM_LIVENESS_OPERATION_USES_PER_SEGMENT];
} loom_liveness_operation_use_segment_t;

static_assert(sizeof(loom_liveness_operation_use_segment_t) == 4096,
              "liveness operation uses must occupy 4 KiB segments");

struct loom_liveness_operation_use_table_t {
  // Stable arena-backed operation-use segments.
  loom_segmented_storage_t segments;
};

typedef struct loom_liveness_bitset_t {
  uint64_t* words;
  iree_host_size_t word_count;
} loom_liveness_bitset_t;

typedef struct loom_liveness_mutable_interval_t {
  // Interval fields being assembled.
  loom_liveness_interval_t interval;
  // True once either a definition or live point has established bounds.
  bool has_bounds;
} loom_liveness_mutable_interval_t;

typedef struct loom_liveness_mutable_segment_t {
  // Value owning |segment|.
  loom_value_ordinal_t value_ordinal;
  // Block-local live segment.
  loom_liveness_segment_t segment;
} loom_liveness_mutable_segment_t;

typedef struct loom_liveness_pressure_state_t {
  // Mutable pressure summaries being built.
  loom_liveness_pressure_summary_t* summaries;
  // Number of initialized pressure summaries.
  iree_host_size_t count;
  // Number of summary records allocated.
  iree_host_size_t capacity;
} loom_liveness_pressure_state_t;

typedef struct loom_liveness_point_shape_t {
  // Number of program points occupied by the shaped IR.
  uint32_t point_span;
  // Number of operations represented by the shaped IR.
  uint32_t operation_count;
} loom_liveness_point_shape_t;

typedef struct loom_liveness_build_state_t {
  // Module containing the analyzed region.
  loom_module_t* module;
  // Region being analyzed.
  const loom_region_t* region;
  // Optional operation order for each block.
  loom_liveness_order_t order;
  // Arena owning analysis result storage retained by the caller.
  iree_arena_allocator_t* result_arena;
  // Resettable workspace released before analysis returns.
  iree_arena_allocator_t* scratch_arena;
  // Active local value domain shared with adjacent compiler phases.
  const loom_local_value_domain_t* value_domain;
  // Value IDs indexed by region-local value ordinal. Borrowed from
  // value_domain.
  const loom_value_id_t* value_ids;
  // Number of initialized local value IDs.
  loom_value_ordinal_t value_count;
  // Number of 64-bit words in local value bitsets.
  iree_host_size_t word_count;
  // Mutable intervals indexed by region-local value ordinal.
  loom_liveness_mutable_interval_t* interval_states;
  // Block-local segments collected in increasing block order.
  loom_liveness_mutable_segment_t* segments;
  // Number of initialized entries in |segments|.
  iree_host_size_t segment_count;
  // Number of allocated entries in |segments|.
  iree_host_size_t segment_capacity;
  // Mutable block-local segment starts indexed by value ordinal.
  uint32_t* segment_start_points;
  // Mutable block-local segment ends indexed by value ordinal.
  uint32_t* segment_end_points;
  // Value ordinals touched in the current top-level block.
  loom_value_ordinal_t* touched_segment_value_ordinals;
  // Number of initialized entries in |touched_segment_value_ordinals|.
  iree_host_size_t touched_segment_value_count;
  // True while interval finalization is collecting one top-level block.
  bool collecting_segments;
  // Region-local value ordinal to interval-index table.
  uint32_t* value_interval_indices;
  // Operations recorded in increasing accepted program-point order.
  loom_liveness_operation_point_t* operation_points;
  // Number of initialized operation-point rows.
  iree_host_size_t operation_count;
  // Number of operation-point rows allocated.
  iree_host_size_t operation_capacity;
  // Segmented semantic operation-use ordinals.
  loom_liveness_operation_use_table_t* operation_uses;
  // Number of appended semantic operation uses.
  iree_host_size_t operation_use_count;
  // Reusable set deduplicating one operation-use range.
  loom_liveness_bitset_t operation_use_seen;
  // Mutable pressure summary state.
  loom_liveness_pressure_state_t pressure_state;
} loom_liveness_build_state_t;

//===----------------------------------------------------------------------===//
// Bitsets
//===----------------------------------------------------------------------===//

static iree_host_size_t loom_liveness_word_count(iree_host_size_t bit_count) {
  return (bit_count + 63u) / 64u;
}

static iree_status_t loom_liveness_bitset_allocate(
    iree_arena_allocator_t* arena, iree_host_size_t word_count,
    loom_liveness_bitset_t* out_bitset) {
  out_bitset->word_count = word_count;
  if (word_count == 0) {
    out_bitset->words = NULL;
    return iree_ok_status();
  }
  return iree_arena_allocate_array(arena, word_count,
                                   sizeof(*out_bitset->words),
                                   (void**)&out_bitset->words);
}

static void loom_liveness_bitset_clear_all(loom_liveness_bitset_t bitset) {
  if (bitset.word_count == 0) {
    return;
  }
  memset(bitset.words, 0, bitset.word_count * sizeof(*bitset.words));
}

static bool loom_liveness_bitset_set(loom_liveness_bitset_t bitset,
                                     loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  uint64_t old_word = bitset.words[word_index];
  bitset.words[word_index] = old_word | mask;
  return old_word != bitset.words[word_index];
}

static bool loom_liveness_bitset_reset(loom_liveness_bitset_t bitset,
                                       loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  uint64_t old_word = bitset.words[word_index];
  bitset.words[word_index] = old_word & ~mask;
  return old_word != bitset.words[word_index];
}

//===----------------------------------------------------------------------===//
// Value classification and intervals
//===----------------------------------------------------------------------===//

static loom_liveness_value_class_t loom_liveness_classify_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  loom_type_t type = loom_module_value_type(module, value_id);
  loom_liveness_value_class_t value_class = {
      .type_kind = loom_type_kind(type),
      .element_type = loom_type_element_type(type),
      .register_class_id = LOOM_LOW_REGISTER_CLASS_ID_INVALID,
      .register_descriptor_set_stable_id = 0,
  };
  if (loom_type_is_register(type)) {
    value_class.register_descriptor_set_stable_id =
        loom_low_register_type_descriptor_set_stable_id(type);
    value_class.register_class_id = loom_low_register_type_class_id(type);
  }
  return value_class;
}

static uint32_t loom_liveness_value_unit_count(const loom_module_t* module,
                                               loom_value_id_t value_id) {
  loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_type_is_register(type)) {
    return loom_low_register_type_unit_count(type);
  }
  return 1;
}

bool loom_liveness_value_class_equal(loom_liveness_value_class_t lhs,
                                     loom_liveness_value_class_t rhs) {
  return lhs.type_kind == rhs.type_kind &&
         lhs.element_type == rhs.element_type &&
         lhs.register_descriptor_set_stable_id ==
             rhs.register_descriptor_set_stable_id &&
         lhs.register_class_id == rhs.register_class_id;
}

static loom_value_ordinal_t loom_liveness_value_ordinal(
    loom_liveness_build_state_t* state, loom_value_id_t value_id) {
  return loom_local_value_domain_ordinal(state->value_domain, value_id);
}

static iree_status_t loom_liveness_ensure_interval_by_ordinal(
    loom_liveness_build_state_t* state, loom_value_ordinal_t value_ordinal,
    loom_liveness_mutable_interval_t** out_interval) {
  loom_value_id_t value_id = state->value_ids[value_ordinal];
  uint32_t interval_index = state->value_interval_indices[value_ordinal];
  if (interval_index == UINT32_MAX) {
    interval_index = value_ordinal;
    state->value_interval_indices[value_ordinal] = interval_index;
    loom_liveness_mutable_interval_t* interval_state =
        &state->interval_states[interval_index];
    *interval_state = (loom_liveness_mutable_interval_t){
        .interval =
            {
                .value_id = value_id,
                .start_point = 0,
                .end_point = 0,
                .value_class =
                    loom_liveness_classify_value(state->module, value_id),
                .unit_count =
                    loom_liveness_value_unit_count(state->module, value_id),
            },
        .has_bounds = false,
    };
  }
  *out_interval = &state->interval_states[interval_index];
  return iree_ok_status();
}

static iree_status_t loom_liveness_ensure_interval(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    loom_liveness_mutable_interval_t** out_interval) {
  const loom_value_ordinal_t value_ordinal =
      loom_liveness_value_ordinal(state, value_id);
  return loom_liveness_ensure_interval_by_ordinal(state, value_ordinal,
                                                  out_interval);
}

static void loom_liveness_note_segment_bounds(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    uint32_t start_point, uint32_t end_point) {
  if (!state->collecting_segments) {
    return;
  }
  const loom_value_ordinal_t value_ordinal =
      loom_liveness_value_ordinal(state, value_id);
  uint32_t* segment_start = &state->segment_start_points[value_ordinal];
  uint32_t* segment_end = &state->segment_end_points[value_ordinal];
  if (*segment_start == UINT32_MAX) {
    IREE_ASSERT_LT(state->touched_segment_value_count, state->value_count);
    state
        ->touched_segment_value_ordinals[state->touched_segment_value_count++] =
        value_ordinal;
    *segment_start = start_point;
    *segment_end = end_point;
    return;
  }
  if (start_point < *segment_start) {
    *segment_start = start_point;
  }
  if (end_point > *segment_end) {
    *segment_end = end_point;
  }
}

static iree_status_t loom_liveness_note_definition(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    uint32_t point) {
  loom_liveness_mutable_interval_t* interval_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_liveness_ensure_interval(state, value_id, &interval_state));
  if (!interval_state->has_bounds) {
    interval_state->interval.start_point = point;
    interval_state->interval.end_point = point;
    interval_state->has_bounds = true;
  } else if (point < interval_state->interval.start_point) {
    interval_state->interval.start_point = point;
  }
  loom_liveness_note_segment_bounds(state, value_id, point, point);
  return iree_ok_status();
}

static iree_status_t loom_liveness_note_live_point(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    uint32_t point) {
  if (point == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness live point exceeds uint32_t range");
  }
  loom_liveness_mutable_interval_t* interval_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_liveness_ensure_interval(state, value_id, &interval_state));
  if (!interval_state->has_bounds) {
    interval_state->interval.start_point = point;
    interval_state->interval.end_point = point + 1u;
    interval_state->has_bounds = true;
    loom_liveness_note_segment_bounds(state, value_id, point, point + 1u);
    return iree_ok_status();
  }
  if (point < interval_state->interval.start_point) {
    interval_state->interval.start_point = point;
  }
  if (point + 1u > interval_state->interval.end_point) {
    interval_state->interval.end_point = point + 1u;
  }
  loom_liveness_note_segment_bounds(state, value_id, point, point + 1u);
  return iree_ok_status();
}

static iree_status_t loom_liveness_append_segment(
    loom_liveness_build_state_t* state, loom_value_ordinal_t value_ordinal,
    uint32_t start_point, uint32_t end_point) {
  if (start_point >= end_point) {
    return iree_ok_status();
  }
  if (state->segment_count >= state->segment_capacity) {
    const iree_host_size_t minimum_capacity = state->segment_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->scratch_arena, state->segment_count, minimum_capacity,
        sizeof(*state->segments), &state->segment_capacity,
        (void**)&state->segments));
  }
  state->segments[state->segment_count++] = (loom_liveness_mutable_segment_t){
      .value_ordinal = value_ordinal,
      .segment =
          {
              .start_point = start_point,
              .end_point = end_point,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_liveness_initialize_segment_scratch(
    loom_liveness_build_state_t* state) {
  if (state->value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->scratch_arena, state->value_count,
                                sizeof(*state->segment_start_points),
                                (void**)&state->segment_start_points));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, state->value_count,
      sizeof(*state->segment_end_points), (void**)&state->segment_end_points));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, state->value_count,
      sizeof(*state->touched_segment_value_ordinals),
      (void**)&state->touched_segment_value_ordinals));
  for (iree_host_size_t i = 0; i < state->value_count; ++i) {
    state->segment_start_points[i] = UINT32_MAX;
    state->segment_end_points[i] = UINT32_MAX;
  }
  return iree_ok_status();
}

static void loom_liveness_begin_block_segments(
    loom_liveness_build_state_t* state) {
  IREE_ASSERT_EQ(state->touched_segment_value_count, 0);
  state->collecting_segments = true;
}

static iree_status_t loom_liveness_finish_block_segments(
    loom_liveness_build_state_t* state) {
  state->collecting_segments = false;
  for (iree_host_size_t i = 0; i < state->touched_segment_value_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        state->touched_segment_value_ordinals[i];
    IREE_RETURN_IF_ERROR(loom_liveness_append_segment(
        state, value_ordinal, state->segment_start_points[value_ordinal],
        state->segment_end_points[value_ordinal]));
    state->segment_start_points[value_ordinal] = UINT32_MAX;
    state->segment_end_points[value_ordinal] = UINT32_MAX;
  }
  state->touched_segment_value_count = 0;
  return iree_ok_status();
}

static bool loom_liveness_build_includes_region_tree(
    const loom_liveness_build_state_t* state) {
  return iree_any_bit_set(state->value_domain->flags,
                          LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE);
}

static bool loom_liveness_analysis_flags_include_region_tree(
    loom_liveness_analysis_flags_t flags) {
  return iree_any_bit_set(flags, LOOM_LIVENESS_ANALYSIS_FLAG_REGION_TREE);
}

static iree_status_t loom_liveness_add_span(uint32_t* inout_point,
                                            uint32_t span,
                                            iree_string_view_t subject) {
  if (*inout_point > UINT32_MAX - span) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness %.*s point span exceeds uint32_t",
                            (int)subject.size, subject.data);
  }
  *inout_point += span;
  return iree_ok_status();
}

static iree_status_t loom_liveness_region_point_shape_for_flags(
    loom_liveness_analysis_flags_t flags, const loom_region_t* region,
    loom_liveness_point_shape_t* out_shape);

static iree_status_t loom_liveness_op_point_shape_for_flags(
    loom_liveness_analysis_flags_t flags, const loom_op_t* op,
    loom_liveness_point_shape_t* out_shape) {
  *out_shape = (loom_liveness_point_shape_t){
      .point_span = 1,
      .operation_count = 1,
  };
  if (!loom_liveness_analysis_flags_include_region_tree(flags)) {
    return iree_ok_status();
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_liveness_point_shape_t region_shape = {0};
    IREE_RETURN_IF_ERROR(loom_liveness_region_point_shape_for_flags(
        flags, regions[i], &region_shape));
    if (region_shape.point_span == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_liveness_add_span(&out_shape->point_span,
                                                region_shape.point_span,
                                                IREE_SV("nested region")));
    IREE_RETURN_IF_ERROR(loom_liveness_add_span(&out_shape->point_span, 1,
                                                IREE_SV("nested region gap")));
    IREE_ASSERT_LE(out_shape->operation_count, out_shape->point_span);
    IREE_ASSERT_LE(region_shape.operation_count, region_shape.point_span);
    out_shape->operation_count += region_shape.operation_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_block_point_shape_for_flags(
    loom_liveness_analysis_flags_t flags, const loom_block_t* block,
    loom_liveness_point_shape_t* out_shape) {
  if (!loom_liveness_analysis_flags_include_region_tree(flags)) {
    *out_shape = (loom_liveness_point_shape_t){
        .point_span = block->op_count,
        .operation_count = block->op_count,
    };
    return iree_ok_status();
  }
  loom_liveness_point_shape_t shape = {0};
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    loom_liveness_point_shape_t op_shape = {0};
    IREE_RETURN_IF_ERROR(
        loom_liveness_op_point_shape_for_flags(flags, op, &op_shape));
    IREE_RETURN_IF_ERROR(loom_liveness_add_span(
        &shape.point_span, op_shape.point_span, IREE_SV("block")));
    IREE_ASSERT_LE(shape.operation_count, shape.point_span);
    IREE_ASSERT_LE(op_shape.operation_count, op_shape.point_span);
    shape.operation_count += op_shape.operation_count;
  }
  *out_shape = shape;
  return iree_ok_status();
}

static iree_status_t loom_liveness_region_point_shape_for_flags(
    loom_liveness_analysis_flags_t flags, const loom_region_t* region,
    loom_liveness_point_shape_t* out_shape) {
  *out_shape = (loom_liveness_point_shape_t){0};
  if (region == NULL || region->block_count == 0) {
    return iree_ok_status();
  }
  loom_liveness_point_shape_t shape = {0};
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (block_index != 0) {
      IREE_RETURN_IF_ERROR(loom_liveness_add_span(&shape.point_span, 1,
                                                  IREE_SV("region block gap")));
    }
    loom_liveness_point_shape_t block_shape = {0};
    IREE_RETURN_IF_ERROR(loom_liveness_block_point_shape_for_flags(
        flags, loom_region_const_block(region, block_index), &block_shape));
    IREE_RETURN_IF_ERROR(loom_liveness_add_span(
        &shape.point_span, block_shape.point_span, IREE_SV("region")));
    IREE_ASSERT_LE(shape.operation_count, shape.point_span);
    IREE_ASSERT_LE(block_shape.operation_count, block_shape.point_span);
    shape.operation_count += block_shape.operation_count;
  }
  *out_shape = shape;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Program points and intervals
//===----------------------------------------------------------------------===//

typedef struct loom_liveness_point_use_state_t {
  loom_liveness_build_state_t* build_state;
  loom_liveness_operation_point_t* operation_point;
  uint32_t point;
} loom_liveness_point_use_state_t;

static loom_value_ordinal_t loom_liveness_operation_use_table_ordinal(
    const loom_liveness_operation_use_table_t* table, uint32_t use_index) {
  const uint32_t segment_index =
      use_index >> LOOM_LIVENESS_OPERATION_USE_SEGMENT_SHIFT;
  const loom_liveness_operation_use_segment_t* segment =
      (const loom_liveness_operation_use_segment_t*)
          loom_segmented_storage_const_segment(&table->segments, segment_index);
  return segment
      ->ordinals[use_index & LOOM_LIVENESS_OPERATION_USE_SEGMENT_MASK];
}

static iree_status_t loom_liveness_append_operation_use(
    loom_liveness_build_state_t* state, loom_value_ordinal_t value_ordinal) {
  if (state->operation_use_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness operation uses exceed uint32_t");
  }
  const uint32_t use_index = (uint32_t)state->operation_use_count;
  const uint32_t segment_index =
      use_index >> LOOM_LIVENESS_OPERATION_USE_SEGMENT_SHIFT;
  if ((use_index & LOOM_LIVENESS_OPERATION_USE_SEGMENT_MASK) == 0) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &state->operation_uses->segments, state->result_arena, &segment));
    IREE_ASSERT_EQ(segment_index,
                   state->operation_uses->segments.segment_count - 1u);
  }
  loom_liveness_operation_use_segment_t* segment =
      (loom_liveness_operation_use_segment_t*)loom_segmented_storage_segment(
          &state->operation_uses->segments, segment_index);
  segment->ordinals[use_index & LOOM_LIVENESS_OPERATION_USE_SEGMENT_MASK] =
      value_ordinal;
  ++state->operation_use_count;
  return iree_ok_status();
}

static void loom_liveness_reset_operation_use_range(
    loom_liveness_build_state_t* state, uint32_t use_start) {
  for (iree_host_size_t i = use_start; i < state->operation_use_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_operation_use_table_ordinal(state->operation_uses,
                                                  (uint32_t)i);
    const bool was_set =
        loom_liveness_bitset_reset(state->operation_use_seen, value_ordinal);
    IREE_ASSERT(was_set);
  }
}

static iree_status_t loom_liveness_note_use_at_point(void* user_data,
                                                     loom_value_id_t value_id) {
  loom_liveness_point_use_state_t* state =
      (loom_liveness_point_use_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_liveness_note_live_point(state->build_state,
                                                     value_id, state->point));
  if (state->operation_point != NULL) {
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_value_ordinal(state->build_state, value_id);
    if (loom_liveness_bitset_set(state->build_state->operation_use_seen,
                                 value_ordinal)) {
      IREE_RETURN_IF_ERROR(loom_liveness_append_operation_use(
          state->build_state, value_ordinal));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_note_values_live_point(
    loom_liveness_build_state_t* state, const loom_value_id_t* values,
    iree_host_size_t value_count, uint32_t point) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_liveness_note_live_point(state, values[i], point));
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_block_arguments(
    loom_liveness_build_state_t* state, const loom_block_t* block,
    uint32_t start_point) {
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
    IREE_RETURN_IF_ERROR(
        loom_liveness_note_definition(state, arg_id, start_point));
    loom_liveness_point_use_state_t type_use_state = {
        .build_state = state,
        .point = start_point,
    };
    IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
        state->module, loom_block_arg_type(state->module, block, arg_index),
        loom_liveness_value_callback_make(loom_liveness_note_use_at_point,
                                          &type_use_state)));
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_region_tree_intervals(
    loom_liveness_build_state_t* state, const loom_region_t* region,
    uint32_t start_point, uint32_t parent_operation_index,
    uint32_t* out_end_point);

static loom_liveness_operation_point_t* loom_liveness_append_operation_point(
    loom_liveness_build_state_t* state, const loom_op_t* op,
    uint32_t parent_operation_index, uint32_t start_point,
    uint32_t* out_operation_index) {
  IREE_ASSERT_LT(state->operation_count, state->operation_capacity);
  const uint32_t operation_index = (uint32_t)state->operation_count++;
  loom_liveness_operation_point_t* operation_point =
      &state->operation_points[operation_index];
  *operation_point = (loom_liveness_operation_point_t){
      .op = op,
      .parent_operation_index = parent_operation_index,
      .start_point = start_point,
      .use_start = (uint32_t)state->operation_use_count,
  };
  *out_operation_index = operation_index;
  return operation_point;
}

static iree_status_t loom_liveness_finalize_op_intervals(
    loom_liveness_build_state_t* state, const loom_op_t* op, uint32_t point,
    uint32_t parent_operation_index, uint32_t* out_end_point) {
  uint32_t operation_index = UINT32_MAX;
  loom_liveness_operation_point_t* operation_point =
      loom_liveness_append_operation_point(state, op, parent_operation_index,
                                           point, &operation_index);
  loom_liveness_point_use_state_t use_state = {
      .build_state = state,
      .operation_point = operation_point,
      .point = point,
  };
  IREE_RETURN_IF_ERROR(loom_liveness_for_each_op_direct_use(
      state->module, op,
      loom_liveness_value_callback_make(loom_liveness_note_use_at_point,
                                        &use_state)));
  operation_point->direct_use_count =
      (uint32_t)state->operation_use_count - operation_point->use_start;
  loom_liveness_reset_operation_use_range(state, operation_point->use_start);
  const uint32_t nested_use_start = (uint32_t)state->operation_use_count;
  IREE_RETURN_IF_ERROR(loom_liveness_for_each_nested_external_use(
      state->module, op,
      loom_liveness_value_callback_make(loom_liveness_note_use_at_point,
                                        &use_state)));
  operation_point->use_count =
      (uint32_t)state->operation_use_count - operation_point->use_start;
  loom_liveness_reset_operation_use_range(state, nested_use_start);

  uint32_t result_point = point;
  IREE_RETURN_IF_ERROR(
      loom_liveness_add_span(&result_point, 1, IREE_SV("operation")));
  if (loom_liveness_build_includes_region_tree(state)) {
    loom_region_t* const* regions = loom_op_regions(op);
    for (uint8_t i = 0; i < op->region_count; ++i) {
      uint32_t region_end_point = result_point;
      IREE_RETURN_IF_ERROR(loom_liveness_finalize_region_tree_intervals(
          state, regions[i], result_point, operation_index, &region_end_point));
      if (region_end_point == result_point) {
        continue;
      }
      result_point = region_end_point;
      IREE_RETURN_IF_ERROR(
          loom_liveness_add_span(&result_point, 1, IREE_SV("nested region")));
    }
    const loom_loop_like_t loop =
        loom_loop_like_cast(state->module, (loom_op_t*)op);
    if (loom_loop_like_isa(loop)) {
      // Captured values are read again on the next iteration. Their lexical
      // last use cannot release storage before the implicit backedge.
      const uint32_t backedge_point = result_point - 1;
      for (uint32_t i = operation_point->direct_use_count;
           i < operation_point->use_count; ++i) {
        const loom_value_ordinal_t ordinal =
            loom_liveness_operation_use_table_ordinal(
                state->operation_uses, operation_point->use_start + i);
        IREE_RETURN_IF_ERROR(loom_liveness_note_live_point(
            state, state->value_ids[ordinal], backedge_point));
      }
      if (loom_loop_like_has_counted_range(loop)) {
        // The loop itself reads these values after yielding the body state.
        // The lower bound is only used on entry.
        IREE_RETURN_IF_ERROR(loom_liveness_note_live_point(
            state, loom_loop_like_iv(loop), backedge_point));
        IREE_RETURN_IF_ERROR(loom_liveness_note_live_point(
            state, loom_loop_like_upper_bound(loop), backedge_point));
        IREE_RETURN_IF_ERROR(loom_liveness_note_live_point(
            state, loom_loop_like_step(loop), backedge_point));
      }
    }
  }

  operation_point->end_point = result_point;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    IREE_RETURN_IF_ERROR(loom_liveness_note_definition(
        state, results[result_index], result_point));
  }
  *out_end_point = result_point;
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_source_block_intervals(
    loom_liveness_build_state_t* state, const loom_block_t* block,
    uint32_t start_point, uint32_t parent_operation_index,
    uint32_t* out_end_point) {
  IREE_RETURN_IF_ERROR(
      loom_liveness_finalize_block_arguments(state, block, start_point));
  uint32_t point = start_point;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    IREE_RETURN_IF_ERROR(loom_liveness_finalize_op_intervals(
        state, op, point, parent_operation_index, &point));
  }
  *out_end_point = point;
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_ordered_block_intervals(
    loom_liveness_build_state_t* state, const loom_block_t* block,
    const loom_liveness_block_order_t* block_order, uint32_t start_point,
    uint32_t parent_operation_index, uint32_t* out_end_point) {
  IREE_RETURN_IF_ERROR(
      loom_liveness_finalize_block_arguments(state, block, start_point));
  uint32_t point = start_point;
  for (iree_host_size_t ordered_index = 0;
       ordered_index < block_order->op_count; ++ordered_index) {
    const loom_op_t* op = block_order->ops[ordered_index];
    IREE_RETURN_IF_ERROR(loom_liveness_finalize_op_intervals(
        state, op, point, parent_operation_index, &point));
  }
  *out_end_point = point;
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_region_tree_intervals(
    loom_liveness_build_state_t* state, const loom_region_t* region,
    uint32_t start_point, uint32_t parent_operation_index,
    uint32_t* out_end_point) {
  *out_end_point = start_point;
  if (region == NULL) {
    return iree_ok_status();
  }
  uint32_t point = start_point;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (block_index != 0) {
      IREE_RETURN_IF_ERROR(
          loom_liveness_add_span(&point, 1, IREE_SV("region block gap")));
    }
    uint32_t block_end_point = point;
    IREE_RETURN_IF_ERROR(loom_liveness_finalize_source_block_intervals(
        state, loom_region_const_block(region, block_index), point,
        parent_operation_index, &block_end_point));
    point = block_end_point;
  }
  *out_end_point = point;
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_intervals(
    loom_liveness_build_state_t* state,
    const loom_liveness_block_info_t* block_infos) {
  IREE_RETURN_IF_ERROR(loom_liveness_initialize_segment_scratch(state));
  for (iree_host_size_t block_index = 0;
       block_index < state->region->block_count; ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(state->region, (uint16_t)block_index);
    const loom_liveness_block_info_t* block_info = &block_infos[block_index];
    loom_liveness_begin_block_segments(state);
    IREE_RETURN_IF_ERROR(loom_liveness_note_values_live_point(
        state, block_info->live_in_values, block_info->live_in_count,
        block_info->start_point));
    IREE_RETURN_IF_ERROR(loom_liveness_note_values_live_point(
        state, block_info->live_out_values, block_info->live_out_count,
        block_info->end_point));

    uint32_t block_end_point = block_info->start_point;
    const iree_host_size_t operation_start = state->operation_count;
    if (!loom_liveness_order_is_empty(state->order)) {
      IREE_RETURN_IF_ERROR(loom_liveness_finalize_ordered_block_intervals(
          state, block, &state->order.blocks[block_index],
          block_info->start_point, UINT32_MAX, &block_end_point));
    } else {
      IREE_RETURN_IF_ERROR(loom_liveness_finalize_source_block_intervals(
          state, block, block_info->start_point, UINT32_MAX, &block_end_point));
    }
    if (block_end_point != block_info->end_point) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "liveness block point span changed while "
                              "finalizing intervals");
    }
    if (operation_start != block_info->operation_start ||
        state->operation_count - operation_start !=
            block_info->operation_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "liveness block operation shape changed while "
                              "finalizing intervals");
    }
    IREE_RETURN_IF_ERROR(loom_liveness_finish_block_segments(state));
  }
  IREE_ASSERT_EQ(state->operation_count, state->operation_capacity);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pressure summaries
//===----------------------------------------------------------------------===//

static iree_status_t loom_liveness_pressure_find_or_add(
    loom_liveness_build_state_t* state, loom_liveness_value_class_t value_class,
    iree_host_size_t* out_index) {
  loom_liveness_pressure_state_t* pressure = &state->pressure_state;
  for (iree_host_size_t i = 0; i < pressure->count; ++i) {
    if (loom_liveness_value_class_equal(pressure->summaries[i].value_class,
                                        value_class)) {
      *out_index = i;
      return iree_ok_status();
    }
  }
  if (pressure->count >= pressure->capacity) {
    iree_host_size_t old_capacity = pressure->capacity;
    iree_host_size_t new_capacity = old_capacity == 0 ? 8 : old_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->scratch_arena, old_capacity, new_capacity,
                              sizeof(*pressure->summaries), &new_capacity,
                              (void**)&pressure->summaries));
    memset(pressure->summaries + old_capacity, 0,
           (new_capacity - old_capacity) * sizeof(*pressure->summaries));
    pressure->capacity = new_capacity;
  }
  *out_index = pressure->count++;
  pressure->summaries[*out_index].value_class = value_class;
  return iree_ok_status();
}

typedef struct loom_liveness_pressure_bucket_t {
  loom_liveness_value_class_t value_class;
  uint32_t live_units;
  uint32_t live_values;
} loom_liveness_pressure_bucket_t;

typedef struct loom_liveness_pressure_sweep_t {
  // Analysis build state owning summary output.
  loom_liveness_build_state_t* build_state;
  // Mutable live buckets grouped by pressure class.
  loom_liveness_pressure_bucket_t* buckets;
  // Number of initialized buckets.
  iree_host_size_t count;
  // Allocated bucket capacity.
  iree_host_size_t capacity;
} loom_liveness_pressure_sweep_t;

static iree_status_t loom_liveness_pressure_sweep_bucket(
    loom_liveness_pressure_sweep_t* sweep,
    loom_liveness_value_class_t value_class,
    loom_liveness_pressure_bucket_t** out_bucket) {
  for (iree_host_size_t i = 0; i < sweep->count; ++i) {
    if (loom_liveness_value_class_equal(sweep->buckets[i].value_class,
                                        value_class)) {
      *out_bucket = &sweep->buckets[i];
      return iree_ok_status();
    }
  }
  if (sweep->count >= sweep->capacity) {
    iree_host_size_t old_capacity = sweep->capacity;
    iree_host_size_t new_capacity = old_capacity == 0 ? 8 : old_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        sweep->build_state->scratch_arena, old_capacity, new_capacity,
        sizeof(*sweep->buckets), &new_capacity, (void**)&sweep->buckets));
    memset(sweep->buckets + old_capacity, 0,
           (new_capacity - old_capacity) * sizeof(*sweep->buckets));
    sweep->capacity = new_capacity;
  }
  *out_bucket = &sweep->buckets[sweep->count++];
  **out_bucket = (loom_liveness_pressure_bucket_t){
      .value_class = value_class,
  };
  return iree_ok_status();
}

static iree_status_t loom_liveness_pressure_sweep_apply_event(
    loom_liveness_pressure_sweep_t* sweep, const loom_liveness_event_t* event) {
  const loom_liveness_interval_t* interval =
      &sweep->build_state->interval_states[event->value_ordinal].interval;
  loom_liveness_pressure_bucket_t* bucket = NULL;
  IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_bucket(
      sweep, interval->value_class, &bucket));
  if (event->value_delta < 0) {
    if (bucket->live_units < interval->unit_count || bucket->live_values == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "liveness pressure sweep underflow");
    }
    bucket->live_units -= interval->unit_count;
    --bucket->live_values;
    return iree_ok_status();
  }
  if (bucket->live_units > UINT32_MAX - interval->unit_count ||
      bucket->live_values == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness pressure sweep exceeds uint32_t");
  }
  bucket->live_units += interval->unit_count;
  ++bucket->live_values;
  return iree_ok_status();
}

static iree_status_t loom_liveness_pressure_sweep_record(
    loom_liveness_pressure_sweep_t* sweep,
    const loom_liveness_block_info_t* block_info, const loom_op_t* peak_op,
    uint32_t point) {
  loom_liveness_build_state_t* state = sweep->build_state;
  for (iree_host_size_t i = 0; i < sweep->count; ++i) {
    const loom_liveness_pressure_bucket_t* bucket = &sweep->buckets[i];
    if (bucket->live_values == 0) {
      continue;
    }
    iree_host_size_t summary_index = 0;
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_find_or_add(
        state, bucket->value_class, &summary_index));
    loom_liveness_pressure_summary_t* summary =
        &state->pressure_state.summaries[summary_index];
    if (bucket->live_units > summary->peak_live_units ||
        (bucket->live_units == summary->peak_live_units &&
         bucket->live_values > summary->peak_live_values)) {
      summary->peak_live_units = bucket->live_units;
      summary->peak_live_values = bucket->live_values;
      summary->peak_block = block_info ? block_info->block : NULL;
      summary->peak_op = peak_op;
      summary->peak_point = point;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_pressure_sweep_adjust_value_ordinal(
    loom_liveness_pressure_sweep_t* sweep, loom_value_ordinal_t value_ordinal,
    int8_t value_delta) {
  loom_liveness_build_state_t* state = sweep->build_state;
  loom_liveness_mutable_interval_t* interval_state = NULL;
  IREE_RETURN_IF_ERROR(loom_liveness_ensure_interval_by_ordinal(
      state, value_ordinal, &interval_state));
  const loom_liveness_event_t event = {
      .value_id = interval_state->interval.value_id,
      .value_ordinal = value_ordinal,
      .value_delta = value_delta,
  };
  return loom_liveness_pressure_sweep_apply_event(sweep, &event);
}

static iree_status_t loom_liveness_pressure_sweep_set_live(
    loom_liveness_pressure_sweep_t* sweep, loom_liveness_bitset_t live_values,
    const loom_value_id_t* values, iree_host_size_t value_count) {
  loom_liveness_bitset_clear_all(live_values);
  sweep->count = 0;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_value_ordinal(sweep->build_state, values[i]);
    const bool was_set = loom_liveness_bitset_set(live_values, value_ordinal);
    IREE_ASSERT(was_set);
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_adjust_value_ordinal(
        sweep, value_ordinal, 1));
  }
  return iree_ok_status();
}

typedef struct loom_liveness_add_use_to_pressure_state_t {
  loom_liveness_build_state_t* build_state;
  loom_liveness_pressure_sweep_t* sweep;
  loom_liveness_bitset_t* live_values;
} loom_liveness_add_use_to_pressure_state_t;

static iree_status_t loom_liveness_add_use_to_pressure(
    void* user_data, loom_value_id_t value_id) {
  loom_liveness_add_use_to_pressure_state_t* state =
      (loom_liveness_add_use_to_pressure_state_t*)user_data;
  const loom_value_ordinal_t value_ordinal =
      loom_liveness_value_ordinal(state->build_state, value_id);
  if (!loom_liveness_bitset_set(*state->live_values, value_ordinal)) {
    return iree_ok_status();
  }
  return loom_liveness_pressure_sweep_adjust_value_ordinal(state->sweep,
                                                           value_ordinal, 1);
}

static iree_status_t loom_liveness_remove_result_from_pressure(
    loom_liveness_build_state_t* state, loom_liveness_pressure_sweep_t* sweep,
    loom_liveness_bitset_t live_values, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_liveness_value_ordinal(state, value_id);
  if (!loom_liveness_bitset_reset(live_values, value_ordinal)) {
    return iree_ok_status();
  }
  return loom_liveness_pressure_sweep_adjust_value_ordinal(sweep, value_ordinal,
                                                           -1);
}

static iree_status_t loom_liveness_pressure_sweep_reverse_op(
    loom_liveness_build_state_t* state, loom_liveness_pressure_sweep_t* sweep,
    loom_liveness_bitset_t live_values, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    IREE_RETURN_IF_ERROR(loom_liveness_remove_result_from_pressure(
        state, sweep, live_values, results[result_index]));
  }
  loom_liveness_add_use_to_pressure_state_t add_state = {
      .build_state = state,
      .sweep = sweep,
      .live_values = &live_values,
  };
  return loom_liveness_for_each_op_use(
      state->module, op,
      loom_liveness_value_callback_make(loom_liveness_add_use_to_pressure,
                                        &add_state));
}

static iree_status_t loom_liveness_compute_block_pressure(
    loom_liveness_build_state_t* state,
    const loom_liveness_block_info_t* block_infos) {
  loom_liveness_bitset_t live_values;
  IREE_RETURN_IF_ERROR(loom_liveness_bitset_allocate(
      state->scratch_arena, state->word_count, &live_values));
  loom_liveness_pressure_sweep_t sweep = {
      .build_state = state,
  };
  for (iree_host_size_t block_index = 0;
       block_index < state->region->block_count; ++block_index) {
    const loom_liveness_block_info_t* block_info = &block_infos[block_index];
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_set_live(
        &sweep, live_values, block_info->live_out_values,
        block_info->live_out_count));
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_record(
        &sweep, block_info, NULL, block_info->end_point));

    uint32_t point = block_info->end_point;
    if (!loom_liveness_order_is_empty(state->order)) {
      const loom_liveness_block_order_t* block_order =
          &state->order.blocks[block_index];
      for (iree_host_size_t reverse_index = block_order->op_count;
           reverse_index > 0; --reverse_index) {
        const loom_op_t* op = block_order->ops[reverse_index - 1u];
        IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_reverse_op(
            state, &sweep, live_values, op));
        --point;
        IREE_RETURN_IF_ERROR(
            loom_liveness_pressure_sweep_record(&sweep, block_info, op, point));
      }
      IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_record(
          &sweep, block_info, NULL, block_info->start_point));
      continue;
    }

    const loom_op_t* op = block_info->block->last_op;
    while (op) {
      IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_reverse_op(
          state, &sweep, live_values, op));
      --point;
      IREE_RETURN_IF_ERROR(
          loom_liveness_pressure_sweep_record(&sweep, block_info, op, point));
      op = op->prev_op;
    }
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_record(
        &sweep, block_info, NULL, block_info->start_point));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

static iree_status_t loom_liveness_finalize_block_infos(
    loom_liveness_build_state_t* state,
    const loom_liveness_block_relation_t* block_relations,
    loom_liveness_block_info_t** out_block_infos) {
  loom_liveness_block_info_t* block_infos = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->result_arena, state->region->block_count,
                                sizeof(*block_infos), (void**)&block_infos));

  uint32_t point = 0;
  uint32_t operation_start = 0;
  for (uint16_t block_index = 0; block_index < state->region->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(state->region, block_index);
    const loom_liveness_block_relation_t* block_relation =
        &block_relations[block_index];
    loom_liveness_block_info_t* block_info = &block_infos[block_index];
    block_info->block = block;
    block_info->start_point = point;
    block_info->operation_start = operation_start;
    loom_liveness_point_shape_t block_shape = {0};
    IREE_RETURN_IF_ERROR(loom_liveness_block_point_shape_for_flags(
        loom_liveness_build_includes_region_tree(state)
            ? LOOM_LIVENESS_ANALYSIS_FLAG_REGION_TREE
            : 0,
        block, &block_shape));
    block_info->operation_count = block_shape.operation_count;
    IREE_RETURN_IF_ERROR(loom_liveness_add_span(&point, block_shape.point_span,
                                                IREE_SV("block")));
    block_info->end_point = point;
    operation_start += block_shape.operation_count;
    block_info->live_in_values = block_relation->live_in_values;
    block_info->live_in_count = block_relation->live_in_count;
    block_info->live_out_values = block_relation->live_out_values;
    block_info->live_out_count = block_relation->live_out_count;
    if (block_index + 1u < state->region->block_count) {
      if (block_info->end_point == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "liveness block gap exceeds uint32_t");
      }
      point = block_info->end_point + 1u;
    }
  }
  state->operation_capacity = operation_start;
  *out_block_infos = block_infos;
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_interval_array(
    loom_liveness_build_state_t* state,
    loom_liveness_interval_t** out_intervals, iree_host_size_t* out_count) {
  iree_host_size_t count = 0;
  for (iree_host_size_t value_ordinal = 0; value_ordinal < state->value_count;
       ++value_ordinal) {
    if (state->value_interval_indices[value_ordinal] != UINT32_MAX) {
      const loom_liveness_interval_t* interval =
          &state->interval_states[value_ordinal].interval;
      // Unused arguments need no destination for a defining write. Decide
      // after collecting implicit loop-control reads as well as explicit uses.
      if (interval->start_point == interval->end_point &&
          loom_value_is_block_arg(
              loom_module_value(state->module, interval->value_id)) &&
          !loom_module_value_has_uses(state->module, interval->value_id)) {
        state->value_interval_indices[value_ordinal] = UINT32_MAX;
        continue;
      }
      ++count;
    }
  }
  loom_liveness_interval_t* intervals = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->result_arena, count, sizeof(*intervals), (void**)&intervals));
  }
  iree_host_size_t interval_index = 0;
  for (iree_host_size_t value_ordinal = 0; value_ordinal < state->value_count;
       ++value_ordinal) {
    if (state->value_interval_indices[value_ordinal] == UINT32_MAX) {
      continue;
    }
    state->value_interval_indices[value_ordinal] = (uint32_t)interval_index;
    intervals[interval_index++] =
        state->interval_states[value_ordinal].interval;
  }
  *out_intervals = intervals;
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_segment_array(
    loom_liveness_build_state_t* state, loom_liveness_segment_t** out_segments,
    iree_host_size_t* out_segment_count,
    loom_liveness_segment_range_t** out_value_segment_ranges) {
  *out_segments = NULL;
  *out_segment_count = 0;
  *out_value_segment_ranges = NULL;
  if (state->value_count == 0) {
    return iree_ok_status();
  }
  if (state->segment_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness segment count exceeds uint32_t");
  }

  loom_liveness_segment_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->result_arena, state->value_count,
                                sizeof(*ranges), (void**)&ranges));
  memset(ranges, 0, state->value_count * sizeof(*ranges));
  for (iree_host_size_t i = 0; i < state->segment_count; ++i) {
    const loom_value_ordinal_t value_ordinal = state->segments[i].value_ordinal;
    IREE_ASSERT_LT(value_ordinal, state->value_count);
    IREE_ASSERT_NE(ranges[value_ordinal].count, UINT32_MAX);
    ++ranges[value_ordinal].count;
  }

  uint32_t next_start = 0;
  for (iree_host_size_t i = 0; i < state->value_count; ++i) {
    ranges[i].start = next_start;
    IREE_ASSERT_LE(ranges[i].count, UINT32_MAX - next_start);
    next_start += ranges[i].count;
  }
  IREE_ASSERT_EQ(next_start, (uint32_t)state->segment_count);

  loom_liveness_segment_t* segments = NULL;
  if (state->segment_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->result_arena, state->segment_count,
                                  sizeof(*segments), (void**)&segments));
  }
  uint32_t* cursors = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->scratch_arena, state->value_count,
                                sizeof(*cursors), (void**)&cursors));
  for (iree_host_size_t i = 0; i < state->value_count; ++i) {
    cursors[i] = ranges[i].start;
  }
  for (iree_host_size_t i = 0; i < state->segment_count; ++i) {
    const loom_liveness_mutable_segment_t* mutable_segment =
        &state->segments[i];
    segments[cursors[mutable_segment->value_ordinal]++] =
        mutable_segment->segment;
  }

  *out_segments = segments;
  *out_segment_count = state->segment_count;
  *out_value_segment_ranges = ranges;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

static iree_status_t loom_liveness_validate_order(const loom_region_t* region,
                                                  loom_liveness_order_t order) {
  if (loom_liveness_order_is_empty(order)) {
    return iree_ok_status();
  }
  if (!region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "liveness order requires a region");
  }
  if (order.block_count != region->block_count || order.blocks == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "liveness order must provide one entry per region block");
  }
  for (iree_host_size_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(region, (uint16_t)block_index);
    const loom_liveness_block_order_t* block_order = &order.blocks[block_index];
    if (block_order->block != block) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "liveness order block %zu does not match region block order",
          block_index);
    }
    if (block_order->op_count != block->op_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "liveness order block %zu has %zu op(s) for a block with %u op(s)",
          block_index, block_order->op_count, block->op_count);
    }
    if (block_order->op_count != 0 && block_order->ops == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "liveness order block %zu has no operation order",
                              block_index);
    }
    for (iree_host_size_t op_index = 0; op_index < block_order->op_count;
         ++op_index) {
      const loom_op_t* op = block_order->ops[op_index];
      if (op == NULL || op->parent_block != block) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "liveness order block %zu entry %zu is not owned by the block",
            block_index, op_index);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_liveness_analyze_region(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_liveness_analysis_t* out_analysis) {
  return loom_liveness_analyze_region_with_order(
      module, region, loom_liveness_order_empty(), arena, out_analysis);
}

iree_status_t loom_liveness_analyze_region_with_order(
    loom_module_t* module, const loom_region_t* region,
    loom_liveness_order_t order, iree_arena_allocator_t* arena,
    loom_liveness_analysis_t* out_analysis) {
  memset(out_analysis, 0, sizeof(*out_analysis));

  loom_local_value_domain_t value_domain = {0};
  iree_status_t status = loom_local_value_domain_acquire_for_region(
      module, region, arena, &value_domain);
  if (iree_status_is_ok(status)) {
    status = loom_liveness_analyze_local_value_domain(&value_domain, order,
                                                      arena, out_analysis);
  }
  loom_local_value_domain_release(&value_domain);
  return status;
}

static iree_status_t
loom_liveness_analyze_local_value_domain_with_dataflow_impl(
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_dataflow_t* dataflow, loom_liveness_order_t order,
    iree_arena_allocator_t* result_arena, iree_arena_allocator_t* scratch_arena,
    loom_liveness_analysis_t* out_analysis) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  IREE_ASSERT_ARGUMENT(dataflow);
  memset(out_analysis, 0, sizeof(*out_analysis));
  loom_module_t* module = value_domain->module;
  const loom_region_t* region = value_domain->region;
  loom_liveness_analysis_flags_t analysis_flags = 0;
  if (iree_any_bit_set(value_domain->flags,
                       LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE)) {
    analysis_flags |= LOOM_LIVENESS_ANALYSIS_FLAG_REGION_TREE;
  }

  loom_liveness_build_state_t state = {
      .module = module,
      .region = region,
      .order = order,
      .result_arena = result_arena,
      .scratch_arena = scratch_arena,
      .value_domain = value_domain,
      .value_ids = value_domain->value_ids,
      .value_count = value_domain->value_count,
  };

  iree_status_t status = loom_liveness_validate_order(region, order);
  if (iree_status_is_ok(status)) {
    state.word_count = loom_liveness_word_count(state.value_count);
  }
  if (iree_status_is_ok(status) && state.value_count > 0) {
    status = iree_arena_allocate_array(scratch_arena, state.value_count,
                                       sizeof(*state.interval_states),
                                       (void**)&state.interval_states);
  }
  if (iree_status_is_ok(status) && state.value_count > 0) {
    memset(state.interval_states, 0,
           state.value_count * sizeof(*state.interval_states));
    status = iree_arena_allocate_array(result_arena, state.value_count,
                                       sizeof(*state.value_interval_indices),
                                       (void**)&state.value_interval_indices);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < state.value_count; ++i) {
      state.value_interval_indices[i] = UINT32_MAX;
    }
  }

  loom_liveness_block_info_t* block_infos = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_liveness_finalize_block_infos(&state, dataflow->blocks,
                                                &block_infos);
  }
  if (iree_status_is_ok(status) && state.operation_capacity > 0) {
    status = iree_arena_allocate_array(result_arena, state.operation_capacity,
                                       sizeof(*state.operation_points),
                                       (void**)&state.operation_points);
  }
  if (iree_status_is_ok(status) && state.operation_capacity > 0) {
    status = iree_arena_allocate(result_arena, sizeof(*state.operation_uses),
                                 (void**)&state.operation_uses);
  }
  if (iree_status_is_ok(status) && state.operation_capacity > 0) {
    loom_segmented_storage_initialize(
        sizeof(loom_liveness_operation_use_segment_t),
        iree_alignof(loom_liveness_operation_use_segment_t),
        &state.operation_uses->segments);
  }
  if (iree_status_is_ok(status) && state.operation_capacity > 0) {
    status = loom_liveness_bitset_allocate(scratch_arena, state.word_count,
                                           &state.operation_use_seen);
  }
  if (iree_status_is_ok(status) && state.operation_capacity > 0) {
    loom_liveness_bitset_clear_all(state.operation_use_seen);
  }
  if (iree_status_is_ok(status)) {
    status = loom_liveness_finalize_intervals(&state, block_infos);
  }
  if (iree_status_is_ok(status) &&
      !loom_liveness_build_includes_region_tree(&state)) {
    status = loom_liveness_compute_block_pressure(&state, block_infos);
  }

  loom_liveness_pressure_summary_t* pressure_summaries = NULL;
  if (iree_status_is_ok(status) && state.pressure_state.count > 0) {
    status = iree_arena_allocate_array(result_arena, state.pressure_state.count,
                                       sizeof(*pressure_summaries),
                                       (void**)&pressure_summaries);
    if (iree_status_is_ok(status)) {
      memcpy(pressure_summaries, state.pressure_state.summaries,
             state.pressure_state.count * sizeof(*pressure_summaries));
    }
  }

  loom_liveness_interval_t* intervals = NULL;
  iree_host_size_t interval_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_liveness_finalize_interval_array(&state, &intervals,
                                                   &interval_count);
  }

  loom_liveness_segment_t* segments = NULL;
  iree_host_size_t segment_count = 0;
  loom_liveness_segment_range_t* value_segment_ranges = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_liveness_finalize_segment_array(
        &state, &segments, &segment_count, &value_segment_ranges);
  }

  if (iree_status_is_ok(status)) {
    loom_liveness_analysis_t analysis = {
        .module = module,
        .region = region,
        .flags = analysis_flags,
        .is_cfg =
            iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG),
        .blocks = block_infos,
        .block_count = region->block_count,
        .intervals = intervals,
        .interval_count = interval_count,
        .value_ids = state.value_ids,
        .value_count = state.value_count,
        .value_interval_indices = state.value_interval_indices,
        .segments = segments,
        .segment_count = segment_count,
        .value_segment_ranges = value_segment_ranges,
        .pressure_summaries = pressure_summaries,
        .pressure_summary_count = state.pressure_state.count,
        .operation_points = state.operation_points,
        .operation_count = state.operation_count,
        .operation_uses = state.operation_uses,
        .operation_use_count = state.operation_use_count,
    };
    if (loom_liveness_build_includes_region_tree(&state)) {
      status = loom_liveness_compute_segment_pressure(
          &analysis, scratch_arena, result_arena, &analysis.pressure_summaries,
          &analysis.pressure_summary_count);
    }
    if (iree_status_is_ok(status)) {
      *out_analysis = analysis;
    }
  }

  return status;
}

iree_status_t loom_liveness_analyze_local_value_domain(
    const loom_local_value_domain_t* value_domain, loom_liveness_order_t order,
    iree_arena_allocator_t* arena, loom_liveness_analysis_t* out_analysis) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  memset(out_analysis, 0, sizeof(*out_analysis));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const loom_region_t* region = value_domain->region;
  loom_cfg_graph_t cfg_graph = {
      .module = value_domain->module,
      .region = region,
      .block_count = region->block_count,
  };
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    status = loom_cfg_graph_build(value_domain->module, region, &scratch_arena,
                                  &cfg_graph);
  }
  loom_liveness_dataflow_t dataflow = {0};
  if (iree_status_is_ok(status)) {
    status = loom_liveness_dataflow_analyze(value_domain, &cfg_graph, arena,
                                            &dataflow);
  }
  if (iree_status_is_ok(status)) {
    status = loom_liveness_analyze_local_value_domain_with_dataflow_impl(
        value_domain, &dataflow, order, arena, &scratch_arena, out_analysis);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

iree_status_t loom_liveness_analyze_local_value_domain_with_dataflow(
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_dataflow_t* dataflow, loom_liveness_order_t order,
    iree_arena_allocator_t* arena, loom_liveness_analysis_t* out_analysis) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status =
      loom_liveness_analyze_local_value_domain_with_dataflow_impl(
          value_domain, dataflow, order, arena, &scratch_arena, out_analysis);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

loom_value_ordinal_t loom_liveness_operation_use_ordinal(
    const loom_liveness_analysis_t* analysis, uint32_t use_index) {
  IREE_ASSERT_ARGUMENT(analysis);
  IREE_ASSERT_LT(use_index, analysis->operation_use_count);
  IREE_ASSERT_ARGUMENT(analysis->operation_uses);
  return loom_liveness_operation_use_table_ordinal(analysis->operation_uses,
                                                   use_index);
}

const loom_liveness_interval_t* loom_liveness_interval_for_value(
    const loom_liveness_analysis_t* analysis, loom_value_id_t value_id) {
  if (!analysis) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < analysis->value_count; ++i) {
    if (analysis->value_ids[i] != value_id) {
      continue;
    }
    return loom_liveness_interval_for_value_ordinal(analysis,
                                                    (loom_value_ordinal_t)i);
  }
  return NULL;
}

const loom_liveness_interval_t* loom_liveness_interval_for_value_ordinal(
    const loom_liveness_analysis_t* analysis,
    loom_value_ordinal_t value_ordinal) {
  if (!analysis || !analysis->value_interval_indices ||
      value_ordinal >= analysis->value_count) {
    return NULL;
  }
  uint32_t interval_index = analysis->value_interval_indices[value_ordinal];
  if (interval_index == UINT32_MAX ||
      interval_index >= analysis->interval_count) {
    return NULL;
  }
  return &analysis->intervals[interval_index];
}

loom_liveness_segment_range_t loom_liveness_segment_range_for_value_ordinal(
    const loom_liveness_analysis_t* analysis,
    loom_value_ordinal_t value_ordinal) {
  if (analysis == NULL || analysis->value_segment_ranges == NULL ||
      value_ordinal >= analysis->value_count) {
    return (loom_liveness_segment_range_t){0};
  }
  return analysis->value_segment_ranges[value_ordinal];
}

bool loom_liveness_segment_range_contains(
    const loom_liveness_segment_t* segments,
    loom_liveness_segment_range_t range, uint32_t point) {
  uint32_t first = range.start;
  uint32_t count = range.count;
  while (count != 0) {
    const uint32_t half = count / 2;
    const uint32_t middle = first + half;
    const loom_liveness_segment_t* segment = &segments[middle];
    if (point >= segment->end_point) {
      first = middle + 1;
      count -= half + 1;
    } else if (point < segment->start_point) {
      count = half;
    } else {
      return true;
    }
  }
  return false;
}

bool loom_liveness_segment_ranges_overlap(
    const loom_liveness_segment_t* segments, loom_liveness_segment_range_t lhs,
    loom_liveness_segment_range_t rhs) {
  if (lhs.count == 0 || rhs.count == 0) {
    return false;
  }
  const loom_liveness_segment_t* lhs_segment = &segments[lhs.start];
  const loom_liveness_segment_t* rhs_segment = &segments[rhs.start];
  const loom_liveness_segment_t* lhs_end = lhs_segment + lhs.count;
  const loom_liveness_segment_t* rhs_end = rhs_segment + rhs.count;
  // Both cursors are valid at entry. Only the advanced cursor can become
  // exhausted; the other segment remains available for the next comparison.
  for (;;) {
    if (lhs_segment->end_point <= rhs_segment->start_point) {
      if (++lhs_segment == lhs_end) {
        return false;
      }
    } else if (rhs_segment->end_point <= lhs_segment->start_point) {
      if (++rhs_segment == rhs_end) {
        return false;
      }
    } else {
      return true;
    }
  }
}

const loom_liveness_block_info_t* loom_liveness_block_info_for_block(
    const loom_liveness_analysis_t* analysis, const loom_block_t* block) {
  if (!analysis) {
    return NULL;
  }
  if (!block) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < analysis->block_count; ++i) {
    if (analysis->blocks[i].block == block) {
      return &analysis->blocks[i];
    }
  }
  return NULL;
}

static const loom_liveness_pressure_summary_t*
loom_liveness_pressure_summary_for_class(
    const loom_liveness_analysis_t* analysis,
    loom_liveness_value_class_t value_class) {
  for (iree_host_size_t i = 0; i < analysis->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &analysis->pressure_summaries[i];
    if (loom_liveness_value_class_equal(summary->value_class, value_class)) {
      return summary;
    }
  }
  return NULL;
}

static loom_liveness_pressure_budget_violation_flags_t
loom_liveness_pressure_budget_violation_bits(
    const loom_liveness_pressure_summary_t* summary,
    const loom_liveness_pressure_budget_t* budget) {
  loom_liveness_pressure_budget_violation_flags_t violation_bits = 0;
  if (budget->max_live_units != UINT32_MAX &&
      summary->peak_live_units > budget->max_live_units) {
    violation_bits |= LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_UNITS;
  }
  if (budget->max_live_values != UINT32_MAX &&
      summary->peak_live_values > budget->max_live_values) {
    violation_bits |= LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_VALUES;
  }
  return violation_bits;
}

iree_status_t loom_liveness_collect_pressure_budget_violations(
    const loom_liveness_analysis_t* analysis,
    const loom_liveness_pressure_budget_t* budgets,
    iree_host_size_t budget_count, iree_arena_allocator_t* arena,
    const loom_liveness_pressure_budget_violation_t** out_violations,
    iree_host_size_t* out_violation_count) {
  *out_violations = NULL;
  *out_violation_count = 0;
  if (budget_count == 0) {
    return iree_ok_status();
  }
  if (!budgets) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pressure budgets are required when budget count "
                            "is non-zero");
  }

  iree_host_size_t violation_count = 0;
  for (iree_host_size_t budget_index = 0; budget_index < budget_count;
       ++budget_index) {
    const loom_liveness_pressure_summary_t* summary =
        loom_liveness_pressure_summary_for_class(
            analysis, budgets[budget_index].value_class);
    if (!summary) {
      continue;
    }
    if (loom_liveness_pressure_budget_violation_bits(
            summary, &budgets[budget_index]) != 0) {
      ++violation_count;
    }
  }
  if (violation_count == 0) {
    return iree_ok_status();
  }

  loom_liveness_pressure_budget_violation_t* violations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, violation_count, sizeof(*violations), (void**)&violations));
  iree_host_size_t violation_index = 0;
  for (iree_host_size_t budget_index = 0; budget_index < budget_count;
       ++budget_index) {
    const loom_liveness_pressure_summary_t* summary =
        loom_liveness_pressure_summary_for_class(
            analysis, budgets[budget_index].value_class);
    if (!summary) {
      continue;
    }
    loom_liveness_pressure_budget_violation_flags_t violation_bits =
        loom_liveness_pressure_budget_violation_bits(summary,
                                                     &budgets[budget_index]);
    if (violation_bits == 0) {
      continue;
    }
    violations[violation_index++] = (loom_liveness_pressure_budget_violation_t){
        .budget_index = budget_index,
        .budget = budgets[budget_index],
        .summary = summary,
        .violation_bits = violation_bits,
    };
  }
  *out_violations = violations;
  *out_violation_count = violation_count;
  return iree_ok_status();
}
