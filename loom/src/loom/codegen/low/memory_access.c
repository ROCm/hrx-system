// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/memory_access.h"

#include <string.h>

#include "iree/base/api.h"
#include "loom/analysis/symbolic_congruence.h"
#include "loom/ir/intern_table.h"

loom_low_memory_space_t loom_low_memory_access_normalize_space(
    loom_low_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return memory_space;
    case LOOM_LOW_MEMORY_SPACE_NONE:
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

bool loom_low_memory_access_spaces_may_alias(loom_low_memory_space_t left,
                                             loom_low_memory_space_t right) {
  left = loom_low_memory_access_normalize_space(left);
  right = loom_low_memory_access_normalize_space(right);
  return left == right || left == LOOM_LOW_MEMORY_SPACE_GENERIC ||
         right == LOOM_LOW_MEMORY_SPACE_GENERIC;
}

const loom_low_memory_access_summary_t*
loom_low_memory_access_summary_for_space(loom_low_memory_space_t memory_space) {
  static const loom_low_memory_access_summary_t generic = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_GENERIC,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
  };
  static const loom_low_memory_access_summary_t global = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  static const loom_low_memory_access_summary_t workgroup = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  static const loom_low_memory_access_summary_t stack = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_STACK,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  static const loom_low_memory_access_summary_t wasm_memory = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_WASM_MEMORY,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  switch (memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
      return &global;
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return &workgroup;
    case LOOM_LOW_MEMORY_SPACE_STACK:
      return &stack;
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return &wasm_memory;
    default:
      return &generic;
  }
}

static bool loom_low_byte_intervals_have_range(
    const loom_low_byte_interval_t* interval) {
  const loom_low_byte_interval_precision_flags_t required_flags =
      LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
      LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE;
  return iree_all_bits_set(interval->precision_flags, required_flags);
}

static bool loom_low_byte_interval_envelopes_are_disjoint(
    const loom_low_byte_interval_t* left,
    const loom_low_byte_interval_t* right) {
  if (!loom_low_byte_intervals_have_range(left) ||
      !loom_low_byte_intervals_have_range(right)) {
    return false;
  }
  return left->end_facts.range_hi <= right->begin_facts.range_lo ||
         right->end_facts.range_hi <= left->begin_facts.range_lo;
}

static bool loom_low_strided_byte_intervals_are_disjoint(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right) {
  const loom_low_memory_access_precision_flags_t required_precision =
      LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
      LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL;
  if (!iree_all_bits_set(left->precision_flags, required_precision) ||
      !iree_all_bits_set(right->precision_flags, required_precision) ||
      left->alias_root_id != right->alias_root_id) {
    return false;
  }
  const loom_low_strided_byte_interval_t* left_interval =
      &left->strided_interval;
  const loom_low_strided_byte_interval_t* right_interval =
      &right->strided_interval;
  if (left_interval->stride_bytes == 0 ||
      left_interval->stride_bytes != right_interval->stride_bytes ||
      left_interval->begin_bytes >= left_interval->end_bytes ||
      left_interval->end_bytes > left_interval->stride_bytes ||
      right_interval->begin_bytes >= right_interval->end_bytes ||
      right_interval->end_bytes > right_interval->stride_bytes) {
    return false;
  }
  return left_interval->end_bytes <= right_interval->begin_bytes ||
         right_interval->end_bytes <= left_interval->begin_bytes;
}

bool loom_low_memory_access_summaries_may_alias(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right,
    loom_low_memory_comparison_t comparison) {
  if (!loom_low_memory_access_spaces_may_alias(left->memory_space,
                                               right->memory_space)) {
    return false;
  }
  const loom_low_memory_relative_interval_t* lhs = left->relative_interval;
  const loom_low_memory_relative_interval_t* rhs = right->relative_interval;
  const bool same_captured_invocation =
      comparison == LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION ||
      comparison == LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION;
  if (same_captured_invocation && lhs && rhs && lhs->scope == rhs->scope &&
      lhs->disjoint_storage_ordinal != 0 &&
      rhs->disjoint_storage_ordinal != 0 &&
      lhs->disjoint_storage_ordinal != rhs->disjoint_storage_ordinal) {
    return false;
  }
  if (same_captured_invocation && lhs && rhs && lhs->scope == rhs->scope &&
      lhs->storage_id == rhs->storage_id) {
    int64_t lower = 0, upper = 0;
    if (iree_checked_sub_i64(lhs->lower, rhs->upper, &lower) &&
        iree_checked_add_i64(lower, 1, &lower) &&
        iree_checked_sub_i64(lhs->upper, rhs->lower, &upper) &&
        iree_checked_sub_i64(upper, 1, &upper) &&
        loom_symbolic_congruence_excludes_difference(&rhs->origin, &lhs->origin,
                                                     lower, upper)) {
      return false;
    }
  }
  if (iree_all_bits_set(left->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
      iree_all_bits_set(right->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
      left->alias_root_id != right->alias_root_id) {
    return false;
  }
  if (iree_all_bits_set(left->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP) &&
      iree_all_bits_set(right->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP) &&
      left->alias_group_id != right->alias_group_id) {
    return false;
  }
  if (loom_low_strided_byte_intervals_are_disjoint(left, right)) {
    return false;
  }
  const loom_low_memory_access_precision_flags_t interval_precision =
      LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
      LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL;
  if (iree_all_bits_set(left->precision_flags, interval_precision) &&
      iree_all_bits_set(right->precision_flags, interval_precision) &&
      left->alias_root_id == right->alias_root_id && left->byte_interval &&
      right->byte_interval &&
      loom_low_byte_interval_envelopes_are_disjoint(left->byte_interval,
                                                    right->byte_interval)) {
    return false;
  }
  return true;
}

bool loom_low_memory_access_summaries_equal(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right) {
  if (left == right) {
    return true;
  }
  if (left->relative_interval != right->relative_interval) {
    return false;
  }
  if (left->memory_space != right->memory_space ||
      left->precision_flags != right->precision_flags) {
    return false;
  }
  if (iree_any_bit_set(left->precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
      left->alias_root_id != right->alias_root_id) {
    return false;
  }
  if (iree_any_bit_set(left->precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP) &&
      left->alias_group_id != right->alias_group_id) {
    return false;
  }
  if (iree_any_bit_set(left->precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL) &&
      (left->strided_interval.stride_bytes !=
           right->strided_interval.stride_bytes ||
       left->strided_interval.begin_bytes !=
           right->strided_interval.begin_bytes ||
       left->strided_interval.end_bytes != right->strided_interval.end_bytes)) {
    return false;
  }
  if (!iree_any_bit_set(left->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL) ||
      left->byte_interval == right->byte_interval) {
    return true;
  }
  if (left->byte_interval == NULL || right->byte_interval == NULL) {
    return false;
  }
  const loom_low_byte_interval_t* left_interval = left->byte_interval;
  const loom_low_byte_interval_t* right_interval = right->byte_interval;
  return left_interval->precision_flags == right_interval->precision_flags &&
         left_interval->begin_expr_id == right_interval->begin_expr_id &&
         left_interval->end_expr_id == right_interval->end_expr_id &&
         loom_value_facts_equal(left_interval->begin_facts,
                                right_interval->begin_facts) &&
         loom_value_facts_equal(left_interval->end_facts,
                                right_interval->end_facts);
}

// Immutable effects belonging to one packet binding.
typedef struct loom_low_memory_effect_binding_t {
  // Next distinct descriptor effect, or NULL.
  struct loom_low_memory_effect_binding_t* next;
  // Descriptor-local effect ordinal.
  uint16_t effect_ordinal;
  // Captured semantic refinement for this effect.
  loom_low_memory_access_summary_t summary;
} loom_low_memory_effect_binding_t;

typedef struct loom_low_memory_binding_t {
  // Packet identity, including erased predecessors until map destruction.
  const loom_op_t* op;
  // Immutable refinements for its concrete effects.
  loom_low_memory_effect_binding_t* effects;
} loom_low_memory_binding_t;

struct loom_low_memory_access_map_t {
  // Owner of bindings and immutable proof payloads.
  iree_arena_allocator_t* arena;
  // Operation identity to binding ordinal.
  loom_intern_table_t index;
  // Dense packet bindings; effect payload addresses remain stable on growth.
  loom_low_memory_binding_t* bindings;
  // Allocated entries in bindings; index.count is the initialized count.
  iree_host_size_t capacity;
};

static uint32_t loom_low_memory_binding_hash(const void* op) {
  uint64_t value = (uint64_t)(uintptr_t)op >> 4;
  value ^= value >> 16;
  return (uint32_t)(value * UINT64_C(0x9e3779b97f4a7c15));
}

typedef struct loom_low_memory_binding_key_t {
  // Map being queried.
  const loom_low_memory_access_map_t* map;
  // Packet identity.
  const loom_op_t* op;
} loom_low_memory_binding_key_t;

static bool loom_low_memory_binding_equal(const void* context, uint32_t index) {
  const loom_low_memory_binding_key_t* key = context;
  return key->map->bindings[index].op == key->op;
}

static const loom_low_memory_binding_t* loom_low_memory_binding_lookup(
    const loom_low_memory_access_map_t* map, const loom_op_t* op) {
  if (map == NULL) {
    return NULL;
  }
  const loom_low_memory_binding_key_t key = {map, op};
  const loom_intern_probe_t probe =
      loom_intern_table_probe(&map->index, loom_low_memory_binding_hash(op),
                              loom_low_memory_binding_equal, &key);
  return probe.index != UINT32_MAX ? &map->bindings[probe.index] : NULL;
}

iree_status_t loom_low_memory_access_map_create(
    iree_arena_allocator_t* arena, loom_low_memory_access_map_t** out_map) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(**out_map), (void**)out_map));
  **out_map = (loom_low_memory_access_map_t){.arena = arena};
  return loom_intern_table_initialize(arena, 0, &(*out_map)->index);
}

static iree_status_t loom_low_memory_copy_expression(
    iree_arena_allocator_t* arena, const loom_symbolic_expr_t* input,
    loom_symbolic_expr_t* output) {
  *output = *input;
  if (input->term_count != 0) {
    loom_symbolic_term_t* terms = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, input->term_count, sizeof(*terms), (void**)&terms));
    memcpy(terms, input->terms, input->term_count * sizeof(*terms));
    output->terms = terms;
  }
  if (input->congruence != NULL) {
    loom_symbolic_congruence_t* congruence = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*congruence), (void**)&congruence));
    congruence->modulus = input->congruence->modulus;
    IREE_RETURN_IF_ERROR(loom_low_memory_copy_expression(
        arena, &input->congruence->expression, &congruence->expression));
    output->congruence = congruence;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_memory_bind_effects(
    loom_low_memory_access_map_t* map, const loom_op_t* op,
    loom_low_memory_effect_binding_t* effects) {
  const uint32_t hash = loom_low_memory_binding_hash(op);
  const loom_low_memory_binding_key_t key = {map, op};
  loom_intern_probe_t probe = loom_intern_table_probe(
      &map->index, hash, loom_low_memory_binding_equal, &key);
  IREE_ASSERT_EQ(probe.index, UINT32_MAX);
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      map->arena, &map->index, hash, /*insertion_count=*/1, &probe.slot));
  if (map->index.count == map->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        map->arena, map->index.count, map->index.count + 1,
        sizeof(*map->bindings), &map->capacity, (void**)&map->bindings));
  }
  const uint32_t ordinal = (uint32_t)map->index.count;
  map->bindings[ordinal] = (loom_low_memory_binding_t){op, effects};
  loom_intern_table_insert(&map->index, probe.slot, hash, ordinal);
  return iree_ok_status();
}

iree_status_t loom_low_memory_access_map_insert(
    loom_low_memory_access_map_t* map, const loom_op_t* op,
    uint16_t effect_ordinal, const loom_low_memory_access_summary_t* summary) {
  const loom_low_memory_binding_t* binding =
      loom_low_memory_binding_lookup(map, op);
  IREE_ASSERT(loom_low_memory_access_map_lookup(map, op, effect_ordinal) ==
              NULL);
  loom_low_memory_effect_binding_t* effect = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(map->arena, sizeof(*effect), (void**)&effect));
  *effect = (loom_low_memory_effect_binding_t){
      .next = binding ? binding->effects : NULL,
      .effect_ordinal = effect_ordinal,
      .summary = *summary,
  };
  if (summary->relative_interval != NULL) {
    loom_low_memory_relative_interval_t* interval = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(map->arena, sizeof(*interval), (void**)&interval));
    *interval = *summary->relative_interval;
    IREE_RETURN_IF_ERROR(loom_low_memory_copy_expression(
        map->arena, &summary->relative_interval->origin, &interval->origin));
    effect->summary.relative_interval = interval;
  }
  if (summary->byte_interval != NULL) {
    loom_low_byte_interval_t* interval = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(map->arena, sizeof(*interval), (void**)&interval));
    *interval = *summary->byte_interval;
    effect->summary.byte_interval = interval;
  }
  if (binding != NULL) {
    map->bindings[binding - map->bindings].effects = effect;
    return iree_ok_status();
  }
  return loom_low_memory_bind_effects(map, op, effect);
}

const loom_low_memory_access_summary_t* loom_low_memory_access_map_lookup(
    const loom_low_memory_access_map_t* map, const loom_op_t* op,
    uint16_t effect_ordinal) {
  const loom_low_memory_binding_t* binding =
      loom_low_memory_binding_lookup(map, op);
  for (const loom_low_memory_effect_binding_t* effect =
           binding ? binding->effects : NULL;
       effect != NULL; effect = effect->next) {
    if (effect->effect_ordinal == effect_ordinal) {
      return &effect->summary;
    }
  }
  return NULL;
}

iree_status_t loom_low_memory_access_map_replace(
    loom_low_memory_access_map_t* map, const loom_op_t* old_op,
    const loom_op_t* new_op) {
  const loom_low_memory_binding_t* binding =
      loom_low_memory_binding_lookup(map, old_op);
  return binding ? loom_low_memory_bind_effects(map, new_op, binding->effects)
                 : iree_ok_status();
}

iree_status_t loom_low_memory_access_map_transfer(
    const loom_low_memory_access_map_t* source,
    loom_low_memory_access_map_t* target) {
  for (iree_host_size_t i = 0; i < source->index.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_memory_bind_effects(
        target, source->bindings[i].op, source->bindings[i].effects));
  }
  return iree_ok_status();
}

typedef struct loom_low_memory_scope_projection_t {
  // Captured source namespace.
  const void* source;
  // Fresh target namespace, owned by the target map's arena.
  const void* target;
} loom_low_memory_scope_projection_t;

struct loom_low_memory_access_clone_t {
  // Immutable captured effects at the clone boundary.
  const loom_low_memory_access_map_t* source;
  // Map receiving operation and scope translations.
  loom_low_memory_access_map_t* target;
  // Temporary scope-correspondence owner.
  iree_arena_allocator_t* scratch_arena;
  // Source scope identity to correspondence row.
  loom_intern_table_t scope_index;
  // Rows indexed by scope_index, grown only at capacity.
  loom_low_memory_scope_projection_t* scopes;
  // Allocated correspondence entries.
  iree_host_size_t scope_capacity;
};

iree_status_t loom_low_memory_access_clone_create(
    const loom_low_memory_access_map_t* source,
    loom_low_memory_access_map_t* target, iree_arena_allocator_t* scratch_arena,
    loom_low_memory_access_clone_t** out_clone) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, sizeof(**out_clone),
                                           (void**)out_clone));
  **out_clone = (loom_low_memory_access_clone_t){
      .source = source,
      .target = target,
      .scratch_arena = scratch_arena,
  };
  return loom_intern_table_initialize(scratch_arena, 0,
                                      &(*out_clone)->scope_index);
}

typedef struct loom_low_memory_scope_key_t {
  // Clone invocation being queried.
  const loom_low_memory_access_clone_t* clone;
  // Captured scope to translate.
  const void* scope;
} loom_low_memory_scope_key_t;

static bool loom_low_memory_scope_equal(const void* context, uint32_t index) {
  const loom_low_memory_scope_key_t* key = context;
  return key->clone->scopes[index].source == key->scope;
}

static iree_status_t loom_low_memory_clone_scope(
    loom_low_memory_access_clone_t* clone, const void* source,
    const void** out_target) {
  const loom_low_memory_scope_key_t key = {clone, source};
  const uint32_t hash = loom_low_memory_binding_hash(source);
  loom_intern_probe_t probe = loom_intern_table_probe(
      &clone->scope_index, hash, loom_low_memory_scope_equal, &key);
  if (probe.index != UINT32_MAX) {
    *out_target = clone->scopes[probe.index].target;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      clone->scratch_arena, &clone->scope_index, hash, /*insertion_count=*/1,
      &probe.slot));
  if (clone->scope_index.count == clone->scope_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        clone->scratch_arena, clone->scope_index.count,
        clone->scope_index.count + 1, sizeof(*clone->scopes),
        &clone->scope_capacity, (void**)&clone->scopes));
  }
  void* target = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(clone->target->arena, 1, &target));
  const uint32_t ordinal = (uint32_t)clone->scope_index.count;
  clone->scopes[ordinal] = (loom_low_memory_scope_projection_t){source, target};
  loom_intern_table_insert(&clone->scope_index, probe.slot, hash, ordinal);
  *out_target = target;
  return iree_ok_status();
}

iree_status_t loom_low_memory_access_clone_op(void* context,
                                              const loom_op_t* source_op,
                                              loom_op_t* target_op) {
  loom_low_memory_access_clone_t* clone = context;
  const loom_low_memory_binding_t* binding =
      loom_low_memory_binding_lookup(clone->source, source_op);
  for (const loom_low_memory_effect_binding_t* effect =
           binding ? binding->effects : NULL;
       effect != NULL; effect = effect->next) {
    loom_low_memory_access_summary_t summary = effect->summary;
    loom_low_memory_relative_interval_t relative;
    if (summary.relative_interval != NULL) {
      relative = *summary.relative_interval;
      IREE_RETURN_IF_ERROR(
          loom_low_memory_clone_scope(clone, relative.scope, &relative.scope));
      summary.relative_interval = &relative;
    }
    IREE_RETURN_IF_ERROR(loom_low_memory_access_map_insert(
        clone->target, target_op, effect->effect_ordinal, &summary));
  }
  return iree_ok_status();
}
