// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_recurrence.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/module.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/fact_table.h"

typedef struct loom_value_fact_recurrence_step_t {
  // Loop-invariant operand added to or subtracted from the previous value.
  loom_value_fact_recurrence_operand_t increment;
  // True for subtraction of the increment from the previous value.
  bool subtract;
} loom_value_fact_recurrence_step_t;

struct loom_value_fact_recurrence_t {
  // Header argument updated by this equation.
  loom_value_id_t value;
  // Value entering the header from outside the loop.
  loom_value_fact_recurrence_operand_t initial;
  // Translation steps in their source evaluation order.
  loom_value_fact_recurrence_step_t* steps;
  // Number of translation steps.
  uint32_t step_count;
};

loom_value_fact_recurrence_operand_t loom_value_fact_recurrence_operand_make(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value) {
  loom_value_fact_recurrence_operand_t operand = {.value = value};
  if (value == LOOM_VALUE_ID_INVALID) {
    return operand;
  }
  const loom_type_t type = loom_module_value_type(module, value);
  int64_t domain_lower = 0, domain_upper = 0;
  if (!loom_type_is_scalar(type) ||
      !loom_value_facts_scalar_type_domain(loom_type_element_type(type),
                                           &domain_lower, &domain_upper)) {
    return operand;
  }
  const loom_value_id_t identity =
      loom_value_fact_table_query_identity(table, value);
  if (loom_type_equal(type, loom_module_value_type(module, identity))) {
    value = identity;
  }
  const loom_value_t* definition = loom_module_value(module, value);
  if (loom_value_is_block_arg(definition)) {
    return operand;
  }
  const loom_op_t* op = loom_value_def_op(definition);
  if (loom_index_constant_isa(op)) {
    operand.is_literal = true;
    operand.literal = loom_attr_as_i64(loom_index_constant_value(op));
  } else if (loom_scalar_constant_isa(op)) {
    operand.is_literal = true;
    operand.literal = loom_attr_as_i64(loom_scalar_constant_value(op));
  }
  return operand;
}

loom_value_facts_t loom_value_fact_recurrence_operand_facts(
    const loom_value_fact_table_t* table,
    loom_value_fact_recurrence_operand_t operand) {
  return operand.is_literal
             ? loom_value_facts_exact_i64(operand.literal)
             : loom_value_fact_table_lookup(table, operand.value);
}

// Nodes are memoized only for update chains demanded by this loop boundary.
// Each translation has one varying predecessor, so an explicit stack suffices.
typedef struct loom_value_fact_recurrence_node_t {
  // Canonical SSA value represented by this node.
  loom_value_id_t value;
  // Header argument at the chain's root, or INVALID for an unsupported chain.
  loom_value_id_t root;
  // Memo index of the preceding translation, or UINT32_MAX for a root.
  uint32_t previous;
  // Number of translated operations between root and this value.
  uint32_t step_count;
  // Translation from previous to this value.
  loom_value_fact_recurrence_step_t step;
  // True after the complete chain has been classified.
  bool ready;
} loom_value_fact_recurrence_node_t;

typedef struct loom_value_fact_recurrence_builder_t {
  // Current identities and numeric facts supplied by the loop owner.
  const loom_value_fact_table_t* table;
  // Module owning the verified update expressions.
  const loom_module_t* module;
  // Block defining the carried arguments.
  const loom_block_t* header;
  // First carried argument within header.
  uint16_t argument_offset;
  // Number of carried arguments.
  uint16_t argument_count;
  // Structural invariance and condition forwarding borrowed during the build.
  const loom_value_fact_recurrence_scope_t* scope;
  // Temporary storage released at the construction boundary.
  iree_arena_allocator_t* arena;
  // Sparse SSA-value index into nodes.
  loom_intern_table_t memo;
  // Memo rows, addressed by stable ordinal while the array grows.
  loom_value_fact_recurrence_node_t* nodes;
  // Number of populated memo rows.
  iree_host_size_t node_count;
  // Allocated memo row capacity.
  iree_host_size_t node_capacity;
  // Explicit stack of translations waiting for their predecessor.
  uint32_t* pending;
  // Allocated pending-stack capacity.
  iree_host_size_t pending_capacity;
} loom_value_fact_recurrence_builder_t;

// Condition-body arguments name the condition's original operands. Numeric
// inputs keep that operand's predicates instead of its canonical identity.
static loom_value_id_t loom_value_fact_recurrence_forwarded_value(
    const loom_value_fact_recurrence_builder_t* builder,
    loom_value_id_t value) {
  const loom_value_t* definition = loom_module_value(builder->module, value);
  if (builder->scope->forwarding_block && loom_value_is_block_arg(definition) &&
      loom_value_def_block(definition) == builder->scope->forwarding_block) {
    return builder->scope->forwarding_values
        .values[loom_value_def_index(definition)];
  }
  return value;
}

static loom_value_id_t loom_value_fact_recurrence_canonical_value(
    const loom_value_fact_recurrence_builder_t* builder,
    loom_value_id_t value) {
  loom_value_id_t identity =
      loom_value_fact_table_query_identity(builder->table, value);
  if (loom_type_equal(loom_module_value_type(builder->module, value),
                      loom_module_value_type(builder->module, identity))) {
    value = identity;
  }
  const loom_value_id_t forwarded =
      loom_value_fact_recurrence_forwarded_value(builder, value);
  if (forwarded != value) {
    value = forwarded;
    identity = loom_value_fact_table_query_identity(builder->table, value);
    if (loom_type_equal(loom_module_value_type(builder->module, value),
                        loom_module_value_type(builder->module, identity))) {
      value = identity;
    }
  }
  return value;
}

typedef struct loom_value_fact_recurrence_key_t {
  // Current memo rows; this pointer is borrowed for one probe.
  const loom_value_fact_recurrence_node_t* nodes;
  // SSA value being looked up.
  loom_value_id_t value;
} loom_value_fact_recurrence_key_t;

static bool loom_value_fact_recurrence_key_equal(const void* context,
                                                 uint32_t index) {
  const loom_value_fact_recurrence_key_t* key = context;
  return key->nodes[index].value == key->value;
}

static iree_status_t loom_value_fact_recurrence_node(
    loom_value_fact_recurrence_builder_t* builder, loom_value_id_t value,
    uint32_t* out_index) {
  value = loom_value_fact_recurrence_canonical_value(builder, value);
  const uint32_t hash = value * UINT32_C(2654435761);
  const loom_value_fact_recurrence_key_t key = {
      .nodes = builder->nodes,
      .value = value,
  };
  loom_intern_probe_t probe = loom_intern_table_probe(
      &builder->memo, hash, loom_value_fact_recurrence_key_equal, &key);
  if (probe.index != UINT32_MAX) {
    *out_index = probe.index;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      builder->arena, &builder->memo, hash, 1, &probe.slot));
  if (builder->node_count == builder->node_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, builder->node_count, builder->node_count + 1,
        sizeof(*builder->nodes), &builder->node_capacity,
        (void**)&builder->nodes));
  }
  const uint32_t index = (uint32_t)builder->node_count++;
  builder->nodes[index] = (loom_value_fact_recurrence_node_t){
      .value = value,
      .root = LOOM_VALUE_ID_INVALID,
      .previous = UINT32_MAX,
  };
  loom_intern_table_insert(&builder->memo, probe.slot, hash, index);
  *out_index = index;
  return iree_ok_status();
}

static bool loom_value_fact_recurrence_is_invariant(
    const loom_value_fact_recurrence_builder_t* builder,
    loom_value_fact_recurrence_operand_t operand) {
  return operand.is_literal || builder->scope->is_invariant(
                                   builder->scope->user_data,
                                   loom_value_fact_recurrence_canonical_value(
                                       builder, operand.value));
}

// Classifies one operation without walking its operands. The builder retains
// and composes the single varying predecessor through the shared memo.
static bool loom_value_fact_recurrence_translation(
    const loom_value_fact_recurrence_builder_t* builder, const loom_op_t* op,
    loom_value_id_t* out_previous,
    loom_value_fact_recurrence_step_t* out_step) {
  bool subtract = false;
  switch (op->kind) {
    case LOOM_OP_INDEX_ADD:
    case LOOM_OP_SCALAR_ADDI:
      break;
    case LOOM_OP_INDEX_SUB:
    case LOOM_OP_SCALAR_SUBI:
      subtract = true;
      break;
    default:
      return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_fact_recurrence_operand_t left =
      loom_value_fact_recurrence_operand_make(
          builder->table, builder->module,
          loom_value_fact_recurrence_forwarded_value(builder, operands[0]));
  const loom_value_fact_recurrence_operand_t right =
      loom_value_fact_recurrence_operand_make(
          builder->table, builder->module,
          loom_value_fact_recurrence_forwarded_value(builder, operands[1]));
  const bool left_invariant =
      loom_value_fact_recurrence_is_invariant(builder, left);
  const bool right_invariant =
      loom_value_fact_recurrence_is_invariant(builder, right);
  if (!left_invariant && right_invariant) {
    *out_previous = left.value;
    *out_step = (loom_value_fact_recurrence_step_t){
        .increment = right,
        .subtract = subtract,
    };
    return true;
  }
  if (!subtract && left_invariant && !right_invariant) {
    *out_previous = right.value;
    *out_step = (loom_value_fact_recurrence_step_t){.increment = left};
    return true;
  }
  return false;
}

static iree_status_t loom_value_fact_recurrence_resolve(
    loom_value_fact_recurrence_builder_t* builder, loom_value_id_t value,
    uint32_t* out_index) {
  IREE_RETURN_IF_ERROR(
      loom_value_fact_recurrence_node(builder, value, out_index));
  uint32_t index = *out_index;
  iree_host_size_t pending_count = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !builder->nodes[index].ready) {
    loom_value_fact_recurrence_node_t* node = &builder->nodes[index];
    const loom_value_t* definition =
        loom_module_value(builder->module, node->value);
    if (loom_value_is_block_arg(definition)) {
      const uint16_t argument_index = loom_value_def_index(definition);
      if (loom_value_def_block(definition) == builder->header &&
          argument_index >= builder->argument_offset &&
          argument_index - builder->argument_offset < builder->argument_count) {
        node->root = node->value;
      }
      node->ready = true;
      break;
    }
    loom_value_id_t previous = LOOM_VALUE_ID_INVALID;
    if (!loom_value_fact_recurrence_translation(
            builder, loom_value_def_op(definition), &previous, &node->step) ||
        !loom_type_equal(loom_module_value_type(builder->module, node->value),
                         loom_module_value_type(builder->module, previous))) {
      node->ready = true;
      break;
    }
    if (pending_count == builder->pending_capacity) {
      status = iree_arena_grow_array(
          builder->arena, pending_count, pending_count + 1,
          sizeof(*builder->pending), &builder->pending_capacity,
          (void**)&builder->pending);
    }
    if (iree_status_is_ok(status)) {
      uint32_t previous_index = 0;
      status =
          loom_value_fact_recurrence_node(builder, previous, &previous_index);
      if (iree_status_is_ok(status)) {
        builder->nodes[index].previous = previous_index;
        builder->pending[pending_count++] = index;
        index = previous_index;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    while (pending_count) {
      loom_value_fact_recurrence_node_t* node =
          &builder->nodes[builder->pending[--pending_count]];
      const loom_value_fact_recurrence_node_t* previous =
          &builder->nodes[node->previous];
      node->root = previous->root;
      node->step_count = previous->step_count + 1;
      node->ready = true;
    }
  }
  return status;
}

static iree_status_t loom_value_fact_recurrence_retain(
    const loom_value_fact_recurrence_builder_t* builder, uint32_t index,
    loom_value_id_t initial, iree_arena_allocator_t* arena,
    const loom_value_fact_recurrence_t** out_recurrence) {
  loom_value_fact_recurrence_t* recurrence = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*recurrence), (void**)&recurrence));
  *recurrence = (loom_value_fact_recurrence_t){
      .value = builder->nodes[index].root,
      .initial = loom_value_fact_recurrence_operand_make(
          builder->table, builder->module, initial),
      .step_count = builder->nodes[index].step_count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, recurrence->step_count,
                                                 sizeof(*recurrence->steps),
                                                 (void**)&recurrence->steps));
  for (uint32_t i = recurrence->step_count; i > 0; --i) {
    recurrence->steps[i - 1] = builder->nodes[index].step;
    index = builder->nodes[index].previous;
  }
  *out_recurrence = recurrence;
  return iree_ok_status();
}

iree_status_t loom_value_fact_recurrence_set_build(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* header_block, uint16_t argument_offset,
    loom_value_slice_t initial_values, loom_value_slice_t next_values,
    const loom_value_fact_recurrence_scope_t* scope,
    iree_arena_allocator_t* arena, loom_value_fact_recurrence_set_t* out_set) {
  *out_set = (loom_value_fact_recurrence_set_t){0};
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_value_fact_recurrence_builder_t builder = {
      .table = table,
      .module = module,
      .header = header_block,
      .argument_offset = argument_offset,
      .argument_count = initial_values.count,
      .scope = scope,
      .arena = &scratch_arena,
  };
  iree_status_t status =
      loom_intern_table_initialize(&scratch_arena, 0, &builder.memo);
  for (uint16_t i = 0; i < initial_values.count && iree_status_is_ok(status);
       ++i) {
    const loom_value_id_t header_value =
        loom_block_arg_id(header_block, argument_offset + i);
    const loom_type_t type = loom_module_value_type(module, header_value);
    int64_t domain_lower = 0, domain_upper = 0;
    if (!loom_type_is_scalar(type) ||
        !loom_value_facts_scalar_type_domain(loom_type_element_type(type),
                                             &domain_lower, &domain_upper)) {
      continue;
    }
    uint32_t index = 0;
    status = loom_value_fact_recurrence_resolve(&builder, next_values.values[i],
                                                &index);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const loom_value_fact_recurrence_node_t* node = &builder.nodes[index];
    // Directly unchanged state is already solved by the forwarding owner.
    if (node->root != header_value || node->step_count == 0) {
      continue;
    }
    if (!out_set->records) {
      status = iree_arena_allocate_array(arena, initial_values.count,
                                         sizeof(*out_set->records),
                                         (void**)&out_set->records);
      if (iree_status_is_ok(status)) {
        memset(out_set->records, 0,
               initial_values.count * sizeof(*out_set->records));
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_value_fact_recurrence_retain(&builder, index,
                                                 initial_values.values[i],
                                                 arena, &out_set->records[i]);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static bool loom_value_fact_recurrence_fits(
    const loom_value_fact_table_t* table, loom_scalar_type_t scalar_type,
    loom_value_facts_t facts) {
  int64_t lower = 0, upper = 0;
  if (loom_value_facts_is_float(facts) ||
      !loom_value_facts_scalar_type_domain(scalar_type, &lower, &upper) ||
      facts.range_lo < lower || facts.range_hi > upper) {
    return false;
  }
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX) {
    return loom_index_value_facts_fit_signed_target_carrier(&table->context,
                                                            scalar_type, facts);
  }
  if (scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    return loom_index_value_facts_fit_unsigned_target_carrier(
        &table->context, scalar_type, facts);
  }
  return true;
}

bool loom_value_fact_recurrence_evaluate(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_recurrence_t* recurrence, uint64_t trip_count,
    loom_loop_trip_count_kind_t trip_count_kind,
    loom_loop_recurrence_facts_t* out_facts) {
  *out_facts = (loom_loop_recurrence_facts_t){
      .values = loom_value_facts_unknown(),
      .body_values = loom_value_facts_unknown(),
      .exit_value = loom_value_facts_unknown(),
      .trip_count = trip_count,
      .trip_count_kind = trip_count_kind,
  };
  const loom_scalar_type_t scalar_type =
      loom_type_element_type(loom_module_value_type(module, recurrence->value));
  const loom_value_facts_t initial =
      loom_value_fact_recurrence_operand_facts(table, recurrence->initial);
  if (!loom_value_fact_recurrence_fits(table, scalar_type, initial)) {
    return false;
  }
  int64_t delta = 0, minimum_prefix = 0, maximum_prefix = 0;
  for (uint32_t i = 0; trip_count != 0 && i < recurrence->step_count; ++i) {
    const loom_value_fact_recurrence_step_t* step = &recurrence->steps[i];
    const loom_value_facts_t facts =
        loom_value_fact_recurrence_operand_facts(table, step->increment);
    int64_t increment = 0;
    if (!loom_value_facts_as_exact_i64(facts, &increment) ||
        !loom_value_fact_recurrence_fits(table, scalar_type, facts) ||
        !(step->subtract ? iree_checked_sub_i64(delta, increment, &delta)
                         : iree_checked_add_i64(delta, increment, &delta))) {
      return false;
    }
    minimum_prefix = iree_min(minimum_prefix, delta);
    maximum_prefix = iree_max(maximum_prefix, delta);
  }
  loom_loop_recurrence_facts_t candidate;
  if (!loom_loop_domain_additive_recurrence_facts(initial, delta, trip_count,
                                                  &candidate) ||
      !loom_value_fact_recurrence_fits(table, scalar_type, candidate.values)) {
    return false;
  }
  if (trip_count != 0) {
    int64_t lower = 0, upper = 0;
    if (!iree_checked_add_i64(candidate.body_values.range_lo, minimum_prefix,
                              &lower) ||
        !iree_checked_add_i64(candidate.body_values.range_hi, maximum_prefix,
                              &upper) ||
        !loom_value_fact_recurrence_fits(
            table, scalar_type, loom_value_facts_make(lower, upper, 1))) {
      return false;
    }
  }
  candidate.trip_count_kind = trip_count_kind;
  if (trip_count_kind == LOOM_LOOP_TRIP_COUNT_BOUNDED) {
    candidate.exit_value = candidate.values;
  }
  *out_facts = candidate;
  return true;
}

loom_value_facts_t loom_value_fact_recurrence_refine(
    loom_value_facts_t current, loom_value_facts_t proven) {
  loom_value_facts_t result =
      loom_value_facts_clamp_domain(current, proven.range_lo, proven.range_hi);
  int64_t divisor = 0;
  if (!iree_math_checked_lcm_i64(result.known_divisor, proven.known_divisor,
                                 &divisor)) {
    // Either proven divisor remains valid when their joint divisor exceeds
    // the fact representation. Keep the stronger individual magnitude.
    divisor = iree_max(result.known_divisor, proven.known_divisor);
  }
  result.known_divisor = divisor;
  result.flags = (result.flags & ~LOOM_VALUE_FACT_DISTRIBUTION_MASK) |
                 (current.flags & LOOM_VALUE_FACT_DISTRIBUTION_MASK);
  return result;
}
