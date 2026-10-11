// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/motion_memory.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_motion_memory_scope_t {
  // Loop owning this body; used only as an identity in the completed index.
  const loom_op_t* loop;
  // First effect position in the body, including nested regions.
  uint32_t begin;
  // Position following all effects in the body.
  uint32_t end;
} loom_motion_memory_scope_t;

typedef struct loom_motion_memory_write_t {
  // Source operation retained for rejection diagnostics.
  const loom_op_t* operation;
  // Position in the immutable effect sequence.
  uint32_t position;
} loom_motion_memory_write_t;

typedef struct loom_motion_memory_footprint_t {
  // Exact or conservative root-relative begin expression.
  loom_symbolic_expr_t begin;
  // Exact or conservative root-relative end expression.
  loom_symbolic_expr_t end;
  // Constant byte length for periodic proofs, or zero when not exact.
  int64_t static_byte_length;
  // Borrowed view-region row owning this access's root and alias facts.
  loom_view_region_id_t view_region;
} loom_motion_memory_footprint_t;

typedef struct loom_motion_memory_barrier_t {
  // Operation preventing read motion.
  const loom_op_t* operation;
  // Position in the immutable effect sequence.
  uint32_t position;
  // Memory space ordered by the operation; unknown covers all spaces.
  loom_value_fact_memory_space_t memory_space;
  // Reason for rejecting motion across this operation.
  loom_motion_loop_hoist_rejection_flags_t rejection_bits;
  // Movement classification details for an unsupported write.
  loom_movement_rejection_flags_t movement_rejection_bits;
} loom_motion_memory_barrier_t;

typedef struct loom_motion_memory_range_t {
  // First index in a sorted range.
  uint32_t begin;
  // Position following the sorted range.
  uint32_t end;
} loom_motion_memory_range_t;

typedef struct loom_motion_memory_interval_t {
  // Inclusive minimum root-relative byte offset.
  int64_t begin;
  // Exclusive maximum root-relative byte offset.
  int64_t end;
} loom_motion_memory_interval_t;

typedef struct loom_motion_memory_group_t {
  // Representative view-region row for equal root/alias/origin/space facts.
  loom_view_region_id_t view_region;
  // Indices of this group's writes, sorted by effect position.
  loom_motion_memory_range_t writes;
  // Sorted disjoint union of conservative numeric bounds for these writes.
  loom_motion_memory_range_t intervals;
} loom_motion_memory_group_t;

struct loom_motion_memory_t {
  // Borrowed owner of view-region rows, symbolic expressions, and scratch.
  loom_movement_analysis_t* movement;
  // Retained loop-body scopes, indexed by operation identity.
  struct {
    // Loop bodies sorted by operation identity after construction.
    loom_motion_memory_scope_t* entries;
    // Number of retained loop bodies.
    iree_host_size_t count;
    // Allocated entry capacity during construction.
    iree_host_size_t capacity;
  } scopes;
  // Source write effects and their demand-built footprint index.
  struct {
    // Write effects in source traversal order.
    loom_motion_memory_write_t* entries;
    // Number of writes; unsupported writes become barriers during indexing.
    iree_host_size_t count;
    // Allocated entry capacity during construction.
    iree_host_size_t capacity;
    // Byte footprints paired with entries, allocated exactly once on demand.
    loom_motion_memory_footprint_t* footprints;
    // Entry indices sorted by root facts, then effect position.
    uint32_t* indices;
  } writes;
  // Root partitions and conservative byte unions for indexed writes.
  struct {
    // Groups of equal root facts sorted primarily by root identity.
    loom_motion_memory_group_t* entries;
    // Number of root groups.
    iree_host_size_t count;
    // Group indices sorted by comparable alias scope.
    uint32_t* aliases;
    // Numeric byte unions, with capacity bounded by the number of writes.
    loom_motion_memory_interval_t* intervals;
  } groups;
  // Retained ordering and unknown effects, indexed by memory space.
  struct {
    // Ordering/unknown effects sorted by memory space, then position.
    loom_motion_memory_barrier_t* entries;
    // Number of ordering/unknown effects.
    iree_host_size_t count;
    // Allocated entry capacity during construction.
    iree_host_size_t capacity;
    // Barrier ranges for each declared memory-space category.
    loom_motion_memory_range_t spaces[LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC + 1];
  } barriers;
  // Next position assigned by the effect producer.
  uint32_t position;
};

static iree_status_t loom_motion_memory_append_barrier(
    loom_motion_memory_t* memory, const loom_op_t* operation, uint32_t position,
    loom_value_fact_memory_space_t memory_space,
    loom_motion_loop_hoist_rejection_flags_t rejection_bits,
    loom_movement_rejection_flags_t movement_rejection_bits) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      memory->movement->expression_context.arena, memory->barriers.count,
      memory->barriers.count + 1, sizeof(*memory->barriers.entries),
      &memory->barriers.capacity, (void**)&memory->barriers.entries));
  memory->barriers.entries[memory->barriers.count++] =
      (loom_motion_memory_barrier_t){
          .operation = operation,
          .position = position,
          .memory_space = memory_space,
          .rejection_bits = rejection_bits,
          .movement_rejection_bits = movement_rejection_bits,
      };
  return iree_ok_status();
}

// MemoryFence operations may declare a conventional memory_space enum. A
// fence without one orders every space; this does not infer a storage write.
static loom_value_fact_memory_space_t loom_motion_memory_fence_space(
    const loom_op_vtable_t* vtable, const loom_op_t* op) {
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (descriptor->attr_kind == LOOM_ATTR_ENUM &&
        iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                               IREE_SV("memory_space"))) {
      const uint8_t value = loom_attr_as_enum(loom_op_const_attrs(op)[i]);
      return value <= LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC
                 ? (loom_value_fact_memory_space_t)value
                 : LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
    }
  }
  return LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
}

static iree_status_t loom_motion_memory_capture_effect(
    loom_motion_memory_t* memory, const loom_op_t* op) {
  const loom_module_t* module = memory->movement->expression_context.module;
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    return loom_motion_memory_append_barrier(
        memory, op, memory->position++, LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
        LOOM_MOTION_LOOP_HOIST_REJECTION_UNKNOWN_INTERFERENCE, 0);
  }
  if (loom_traits_order_memory(traits)) {
    IREE_RETURN_IF_ERROR(loom_motion_memory_append_barrier(
        memory, op, memory->position++,
        loom_motion_memory_fence_space(loom_op_vtable(module, op), op),
        LOOM_MOTION_LOOP_HOIST_REJECTION_ORDERING_INTERFERENCE, 0));
  }
  if (!iree_any_bit_set(traits, LOOM_TRAIT_WRITES_MEMORY)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      memory->movement->expression_context.arena, memory->writes.count,
      memory->writes.count + 1, sizeof(*memory->writes.entries),
      &memory->writes.capacity, (void**)&memory->writes.entries));
  memory->writes.entries[memory->writes.count++] = (loom_motion_memory_write_t){
      .operation = op,
      .position = memory->position++,
  };
  return iree_ok_status();
}

static iree_status_t loom_motion_memory_describe_write(
    loom_motion_memory_t* memory, loom_motion_memory_write_t write,
    loom_motion_memory_footprint_t* footprint, bool* out_described) {
  *out_described = false;
  loom_movement_request_t request = {0};
  loom_movement_diagnostic_t diagnostic = {0};
  bool described = false;
  IREE_RETURN_IF_ERROR(loom_movement_request_describe_op(
      memory->movement, write.operation, &request, &diagnostic, &described));
  if (!described || request.dest.kind != LOOM_MOVEMENT_ENDPOINT_VIEW ||
      !iree_any_bit_set(request.dest.flags, LOOM_MOVEMENT_ENDPOINT_WRITE)) {
    return loom_motion_memory_append_barrier(
        memory, write.operation, write.position,
        LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
        LOOM_MOTION_LOOP_HOIST_REJECTION_UNKNOWN_INTERFERENCE,
        diagnostic.rejection_bits);
  }
  const loom_view_region_t* region = NULL;
  const bool found = loom_view_region_table_try_lookup(
      &memory->movement->view_regions, request.dest.value_id, &region);
  // Describing a view endpoint constructs its owning view-region row.
  IREE_ASSERT(found);
  (void)found;
  *footprint = (loom_motion_memory_footprint_t){
      .begin = request.dest.begin_byte_offset,
      .end = request.dest.end_byte_offset,
      .static_byte_length =
          loom_symbolic_expr_is_constant(&request.dest.byte_length)
              ? request.dest.byte_length.constant
              : 0,
      .view_region = region->region_id,
  };
  *out_described = true;
  return iree_ok_status();
}

static iree_status_t loom_motion_memory_capture_region(
    loom_motion_memory_t* memory, const loom_region_t* region,
    bool inside_loop) {
  const loom_module_t* module = memory->movement->expression_context.module;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
        continue;
      }
      if (inside_loop) {
        IREE_RETURN_IF_ERROR(loom_motion_memory_capture_effect(memory, op));
      }
      const loom_loop_like_t loop = loom_loop_like_cast(module, (loom_op_t*)op);
      const loom_region_t* loop_body = loom_loop_like_has_counted_range(loop)
                                           ? loom_loop_like_body(loop)
                                           : NULL;
      loom_region_t* const* children = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (!children[i]) {
          continue;
        }
        const iree_host_size_t scope_index = memory->scopes.count;
        if (children[i] == loop_body) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              memory->movement->expression_context.arena, memory->scopes.count,
              memory->scopes.count + 1, sizeof(*memory->scopes.entries),
              &memory->scopes.capacity, (void**)&memory->scopes.entries));
          memory->scopes.entries[memory->scopes.count++] =
              (loom_motion_memory_scope_t){
                  .loop = op,
                  .begin = memory->position,
              };
        }
        IREE_RETURN_IF_ERROR(loom_motion_memory_capture_region(
            memory, children[i], inside_loop || children[i] == loop_body));
        if (children[i] == loop_body) {
          memory->scopes.entries[scope_index].end = memory->position;
        }
      }
    }
  }
  return iree_ok_status();
}

static bool loom_motion_memory_scope_less(
    const loom_motion_memory_scope_t* lhs,
    const loom_motion_memory_scope_t* rhs) {
  return (uintptr_t)lhs->loop < (uintptr_t)rhs->loop;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_motion_memory_sort_scopes,
                          loom_motion_memory_scope_t,
                          loom_motion_memory_scope_less)

static bool loom_motion_memory_barrier_less(
    const loom_motion_memory_barrier_t* lhs,
    const loom_motion_memory_barrier_t* rhs) {
  return lhs->memory_space != rhs->memory_space
             ? lhs->memory_space < rhs->memory_space
             : lhs->position < rhs->position;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_motion_memory_sort_barriers,
                          loom_motion_memory_barrier_t,
                          loom_motion_memory_barrier_less)

static int loom_motion_memory_compare_roots(const loom_view_region_t* lhs,
                                            const loom_view_region_t* rhs) {
#define LOOM_COMPARE_ROOT_FIELD(field)       \
  if (lhs->field != rhs->field) {            \
    return lhs->field < rhs->field ? -1 : 1; \
  }
  LOOM_COMPARE_ROOT_FIELD(root_value_id)
  LOOM_COMPARE_ROOT_FIELD(alias_scope_id)
  LOOM_COMPARE_ROOT_FIELD(memory_space)
  LOOM_COMPARE_ROOT_FIELD(origin.function_symbol_id)
  LOOM_COMPARE_ROOT_FIELD(origin.region_index)
  LOOM_COMPARE_ROOT_FIELD(origin.kind)
#undef LOOM_COMPARE_ROOT_FIELD
  return 0;
}

static bool loom_motion_memory_write_less(const loom_motion_memory_t* memory,
                                          const uint32_t* lhs_index,
                                          const uint32_t* rhs_index) {
  const loom_motion_memory_footprint_t* lhs =
      &memory->writes.footprints[*lhs_index];
  const loom_motion_memory_footprint_t* rhs =
      &memory->writes.footprints[*rhs_index];
  const loom_view_region_t* regions = memory->movement->view_regions.regions;
  const int comparison = loom_motion_memory_compare_roots(
      &regions[lhs->view_region], &regions[rhs->view_region]);
  return comparison != 0 ? comparison < 0 : *lhs_index < *rhs_index;
}
LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_motion_memory_sort_writes, uint32_t,
                                       const loom_motion_memory_t*,
                                       loom_motion_memory_write_less)

static bool loom_motion_memory_alias_less(const loom_motion_memory_t* memory,
                                          const uint32_t* lhs,
                                          const uint32_t* rhs) {
  const loom_view_region_t* regions = memory->movement->view_regions.regions;
  const loom_value_fact_alias_scope_id_t lhs_alias =
      loom_view_region_comparable_alias_scope(
          &regions[memory->groups.entries[*lhs].view_region]);
  const loom_value_fact_alias_scope_id_t rhs_alias =
      loom_view_region_comparable_alias_scope(
          &regions[memory->groups.entries[*rhs].view_region]);
  return lhs_alias != rhs_alias ? lhs_alias < rhs_alias : *lhs < *rhs;
}
LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_motion_memory_sort_aliases,
                                       uint32_t, const loom_motion_memory_t*,
                                       loom_motion_memory_alias_less)

static bool loom_motion_memory_interval_less(
    const loom_motion_memory_interval_t* lhs,
    const loom_motion_memory_interval_t* rhs) {
  return lhs->begin < rhs->begin;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_motion_memory_sort_intervals,
                          loom_motion_memory_interval_t,
                          loom_motion_memory_interval_less)

// Numeric unions are only a rejection filter for expensive symbolic queries.
// They contain all writes in the root group, including sibling scopes. A miss
// proves disjointness; a hit still uses the exact scoped write footprints.
static iree_status_t loom_motion_memory_index_intervals(
    loom_motion_memory_t* memory) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      memory->movement->expression_context.arena, memory->writes.count,
      sizeof(*memory->groups.intervals), (void**)&memory->groups.intervals));
  for (iree_host_size_t i = 0; i < memory->groups.count; ++i) {
    loom_motion_memory_group_t* group = &memory->groups.entries[i];
    for (uint32_t j = group->writes.begin; j < group->writes.end; ++j) {
      const loom_motion_memory_footprint_t* write =
          &memory->writes.footprints[memory->writes.indices[j]];
      memory->groups.intervals[j] = (loom_motion_memory_interval_t){
          .begin = loom_value_facts_is_float(write->begin.facts)
                       ? INT64_MIN
                       : write->begin.facts.range_lo,
          .end = loom_value_facts_is_float(write->end.facts)
                     ? INT64_MAX
                     : write->end.facts.range_hi,
      };
    }
    loom_motion_memory_sort_intervals(
        memory->groups.intervals + group->writes.begin,
        group->writes.end - group->writes.begin);
    group->intervals = (loom_motion_memory_range_t){
        .begin = group->writes.begin, .end = group->writes.begin + 1};
    for (uint32_t j = group->writes.begin + 1; j < group->writes.end; ++j) {
      loom_motion_memory_interval_t* previous =
          &memory->groups.intervals[group->intervals.end - 1];
      const loom_motion_memory_interval_t current = memory->groups.intervals[j];
      if (current.begin <= previous->end) {
        previous->end = iree_max(previous->end, current.end);
      } else {
        memory->groups.intervals[group->intervals.end++] = current;
      }
    }
  }
  return iree_ok_status();
}

static void loom_motion_memory_index_barriers(loom_motion_memory_t* memory) {
  loom_motion_memory_sort_barriers(memory->barriers.entries,
                                   memory->barriers.count);
  memset(memory->barriers.spaces, 0, sizeof(memory->barriers.spaces));
  for (uint32_t i = 0; i < memory->barriers.count; ++i) {
    loom_motion_memory_range_t* range =
        &memory->barriers.spaces[memory->barriers.entries[i].memory_space];
    if (range->begin == range->end) {
      range->begin = i;
    }
    range->end = i + 1;
  }
}

// Keep one-time footprint construction and sorting outside the per-load query.
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_motion_memory_index_writes(
    loom_motion_memory_t* memory) {
  iree_arena_allocator_t* arena = memory->movement->expression_context.arena;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, memory->writes.count, sizeof(*memory->writes.footprints),
      (void**)&memory->writes.footprints));
  const iree_host_size_t barrier_count = memory->barriers.count;
  uint32_t described_count = 0;
  for (iree_host_size_t i = 0; i < memory->writes.count; ++i) {
    const loom_motion_memory_write_t write = memory->writes.entries[i];
    bool described = false;
    IREE_RETURN_IF_ERROR(loom_motion_memory_describe_write(
        memory, write, &memory->writes.footprints[described_count],
        &described));
    if (described) {
      memory->writes.entries[described_count++] = write;
    }
  }
  memory->writes.count = described_count;
  if (memory->barriers.count != barrier_count) {
    loom_motion_memory_index_barriers(memory);
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, memory->writes.count, sizeof(*memory->writes.indices),
      (void**)&memory->writes.indices));
  for (uint32_t i = 0; i < memory->writes.count; ++i) {
    memory->writes.indices[i] = i;
  }
  loom_motion_memory_sort_writes(memory, memory->writes.indices,
                                 memory->writes.count);
  const loom_view_region_t* regions = memory->movement->view_regions.regions;
  const loom_view_region_t* previous = NULL;
  for (iree_host_size_t i = 0; i < memory->writes.count; ++i) {
    const loom_view_region_t* region =
        &regions[memory->writes.footprints[memory->writes.indices[i]]
                     .view_region];
    if (!previous || loom_motion_memory_compare_roots(previous, region)) {
      ++memory->groups.count;
      previous = region;
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, memory->groups.count, sizeof(*memory->groups.entries),
      (void**)&memory->groups.entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, memory->groups.count, sizeof(*memory->groups.aliases),
      (void**)&memory->groups.aliases));
  uint32_t group_count = 0;
  for (uint32_t i = 0; i < memory->writes.count; ++i) {
    const loom_view_region_id_t region_id =
        memory->writes.footprints[memory->writes.indices[i]].view_region;
    if (!group_count ||
        loom_motion_memory_compare_roots(
            &regions[memory->groups.entries[group_count - 1].view_region],
            &regions[region_id])) {
      memory->groups.entries[group_count] = (loom_motion_memory_group_t){
          .view_region = region_id, .writes = {.begin = i, .end = i}};
      memory->groups.aliases[group_count] = group_count;
      ++group_count;
    }
    memory->groups.entries[group_count - 1].writes.end = i + 1;
  }
  loom_motion_memory_sort_aliases(memory, memory->groups.aliases,
                                  memory->groups.count);
  IREE_RETURN_IF_ERROR(loom_motion_memory_index_intervals(memory));
  return iree_ok_status();
}

iree_status_t loom_motion_memory_create(loom_movement_analysis_t* movement,
                                        loom_motion_memory_t** out_memory) {
  iree_arena_allocator_t* arena = movement->expression_context.arena;
  loom_motion_memory_t* memory = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*memory), (void**)&memory));
  *memory = (loom_motion_memory_t){.movement = movement};
  IREE_RETURN_IF_ERROR(loom_motion_memory_capture_region(
      memory, movement->view_regions.value_domain->region, false));
  loom_motion_memory_sort_scopes(memory->scopes.entries, memory->scopes.count);
  loom_motion_memory_index_barriers(memory);
  *out_memory = memory;
  return iree_ok_status();
}

static loom_motion_memory_scope_t loom_motion_memory_lookup_scope(
    const loom_motion_memory_t* memory, const loom_op_t* loop) {
  iree_host_size_t begin = 0, end = memory->scopes.count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if ((uintptr_t)memory->scopes.entries[middle].loop < (uintptr_t)loop) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return memory->scopes.entries[begin];
}

static uint32_t loom_motion_memory_first_write(
    const loom_motion_memory_t* memory, loom_motion_memory_range_t range,
    uint32_t position) {
  while (range.begin < range.end) {
    const uint32_t middle = range.begin + (range.end - range.begin) / 2;
    if (memory->writes.entries[memory->writes.indices[middle]].position <
        position) {
      range.begin = middle + 1;
    } else {
      range.end = middle;
    }
  }
  return range.begin;
}

static iree_status_t loom_motion_memory_check_group(
    loom_motion_memory_t* memory, const loom_motion_memory_group_t* group,
    loom_motion_memory_scope_t scope, const loom_view_region_t* read,
    loom_motion_loop_hoist_result_t* result) {
  uint32_t index =
      loom_motion_memory_first_write(memory, group->writes, scope.begin);
  if (index == group->writes.end ||
      memory->writes.entries[memory->writes.indices[index]].position >=
          scope.end) {
    return iree_ok_status();
  }
  const loom_view_region_t* representative =
      &memory->movement->view_regions.regions[group->view_region];
  if (loom_view_region_roots_are_disjoint(read, representative)) {
    return iree_ok_status();
  }
  if (read->root_value_id != LOOM_VALUE_ID_INVALID &&
      read->root_value_id == representative->root_value_id &&
      !loom_value_facts_is_float(read->begin_byte_offset.facts) &&
      !loom_value_facts_is_float(read->end_byte_offset.facts)) {
    loom_motion_memory_range_t range = group->intervals;
    while (range.begin < range.end) {
      const uint32_t middle = range.begin + (range.end - range.begin) / 2;
      if (memory->groups.intervals[middle].end <=
          read->begin_byte_offset.facts.range_lo) {
        range.begin = middle + 1;
      } else {
        range.end = middle;
      }
    }
    if (range.begin == group->intervals.end ||
        memory->groups.intervals[range.begin].begin >=
            read->end_byte_offset.facts.range_hi) {
      return iree_ok_status();
    }
  }
  for (; index < group->writes.end; ++index) {
    const loom_motion_memory_write_t* write =
        &memory->writes.entries[memory->writes.indices[index]];
    if (write->position >= scope.end) {
      break;
    }
    const loom_motion_memory_footprint_t* footprint =
        &memory->writes.footprints[memory->writes.indices[index]];
    loom_view_region_t written = *representative;
    written.begin_byte_offset = footprint->begin;
    written.end_byte_offset = footprint->end;
    loom_symbolic_expr_constant(footprint->static_byte_length,
                                &written.byte_length);
    bool disjoint = false;
    IREE_RETURN_IF_ERROR(loom_view_regions_prove_no_overlap(
        &memory->movement->view_regions, read, &written, &disjoint));
    if (!disjoint) {
      result->rejection_bits =
          LOOM_MOTION_LOOP_HOIST_REJECTION_OVERLAPPING_WRITE;
      result->blocking_op = write->operation;
      break;
    }
  }
  return iree_ok_status();
}

static iree_host_size_t loom_motion_memory_first_alias(
    const loom_motion_memory_t* memory,
    loom_value_fact_alias_scope_id_t alias_scope) {
  iree_host_size_t begin = 0, end = memory->groups.count;
  const loom_view_region_t* regions = memory->movement->view_regions.regions;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_motion_memory_group_t* group =
        &memory->groups.entries[memory->groups.aliases[middle]];
    if (loom_view_region_comparable_alias_scope(&regions[group->view_region]) <
        alias_scope) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static bool loom_motion_memory_check_barriers(
    const loom_motion_memory_t* memory, loom_motion_memory_scope_t scope,
    loom_value_fact_memory_space_t memory_space,
    loom_motion_loop_hoist_result_t* out_result) {
  for (uint32_t space = 0; space <= LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC;
       ++space) {
    if (loom_view_memory_spaces_are_disjoint(
            memory_space, (loom_value_fact_memory_space_t)space)) {
      continue;
    }
    loom_motion_memory_range_t range = memory->barriers.spaces[space];
    const uint32_t end = range.end;
    while (range.begin < range.end) {
      const uint32_t middle = range.begin + (range.end - range.begin) / 2;
      if (memory->barriers.entries[middle].position < scope.begin) {
        range.begin = middle + 1;
      } else {
        range.end = middle;
      }
    }
    if (range.begin < end &&
        memory->barriers.entries[range.begin].position < scope.end) {
      const loom_motion_memory_barrier_t* barrier =
          &memory->barriers.entries[range.begin];
      *out_result = (loom_motion_loop_hoist_result_t){
          .rejection_bits = barrier->rejection_bits,
          .movement_rejection_bits = barrier->movement_rejection_bits,
          .blocking_op = barrier->operation,
      };
      return true;
    }
  }
  return false;
}

iree_status_t loom_motion_memory_evaluate_loop(
    loom_motion_memory_t* memory, const loom_op_t* loop,
    const loom_movement_endpoint_t* read,
    loom_motion_loop_hoist_result_t* out_result) {
  const loom_motion_memory_scope_t scope =
      loom_motion_memory_lookup_scope(memory, loop);
  if (loom_motion_memory_check_barriers(memory, scope, read->memory_space,
                                        out_result) ||
      memory->writes.count == 0) {
    return iree_ok_status();
  }
  if (!memory->writes.indices) {
    IREE_RETURN_IF_ERROR(loom_motion_memory_index_writes(memory));
    // Describing unsupported writes contributes additional unconditional
    // blockers. Consult them before querying any retained byte footprint.
    if (loom_motion_memory_check_barriers(memory, scope, read->memory_space,
                                          out_result)) {
      return iree_ok_status();
    }
  }
  loom_view_region_t read_region;
  loom_movement_endpoint_as_view_region(read, &read_region);
  const loom_view_region_t* regions = memory->movement->view_regions.regions;
  if (loom_view_region_comparable_alias_scope(&read_region) ==
      LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    for (iree_host_size_t i = 0; i < memory->groups.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_motion_memory_check_group(
          memory, &memory->groups.entries[i], scope, &read_region, out_result));
      if (out_result->rejection_bits) {
        break;
      }
    }
    return iree_ok_status();
  }

  // Same-root proofs remain necessary even when two accesses carry different
  // alias assumptions. Distinct scoped roots only need the equal-scope and
  // unscoped groups; other scopes are disjoint by the shared root contract.
  iree_host_size_t begin = 0, end = memory->groups.count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (regions[memory->groups.entries[middle].view_region].root_value_id <
        read->root_value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  for (iree_host_size_t i = begin; i < memory->groups.count; ++i) {
    const loom_motion_memory_group_t* group = &memory->groups.entries[i];
    if (regions[group->view_region].root_value_id != read->root_value_id) {
      break;
    }
    IREE_RETURN_IF_ERROR(loom_motion_memory_check_group(
        memory, group, scope, &read_region, out_result));
    if (out_result->rejection_bits) {
      return iree_ok_status();
    }
  }
  const loom_value_fact_alias_scope_id_t aliases[] = {
      read->alias_scope_id, LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE};
  for (iree_host_size_t a = 0; a < IREE_ARRAYSIZE(aliases); ++a) {
    for (iree_host_size_t i =
             loom_motion_memory_first_alias(memory, aliases[a]);
         i < memory->groups.count; ++i) {
      const loom_motion_memory_group_t* group =
          &memory->groups.entries[memory->groups.aliases[i]];
      const loom_view_region_t* region = &regions[group->view_region];
      if (loom_view_region_comparable_alias_scope(region) != aliases[a]) {
        break;
      }
      if (region->root_value_id == read->root_value_id) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_motion_memory_check_group(
          memory, group, scope, &read_region, out_result));
      if (out_result->rejection_bits) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}
