// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/predicate_facts.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"

// At most this many aliases bound each predicate's scan to constant work.
#define LOOM_VALUE_FACT_ALIAS_LINEAR_SCAN_LIMIT 16

// At most this many predicates bound the number of full alias-list scans.
#define LOOM_VALUE_FACT_PREDICATE_LINEAR_SCAN_LIMIT 4

typedef struct loom_value_fact_alias_map_t {
  // Borrowed alias IDs in result-fact order.
  const loom_value_id_t* values;

  // Number of alias IDs and corresponding result facts.
  uint16_t value_count;

  // Optional one-based alias ordinals keyed by SSA value ID.
  const uint16_t* ordinals;

  // Number of addressable entries in ordinals.
  iree_host_size_t ordinal_capacity;
} loom_value_fact_alias_map_t;

static bool loom_value_fact_alias_map_find(
    const loom_value_fact_alias_map_t* map, loom_value_id_t value_id,
    uint16_t* out_ordinal) {
  if (map->ordinals) {
    if (value_id >= map->ordinal_capacity) return false;
    const uint16_t ordinal_plus_one = map->ordinals[value_id];
    if (ordinal_plus_one == 0) return false;
    *out_ordinal = ordinal_plus_one - 1;
    return true;
  }

  for (uint16_t i = 0; i < map->value_count; ++i) {
    if (map->values[i] != value_id) continue;
    *out_ordinal = i;
    return true;
  }
  return false;
}

static iree_status_t loom_value_fact_alias_map_prepare(
    loom_value_fact_table_t* table, const loom_value_id_t* values,
    uint16_t value_count, loom_value_fact_alias_map_t* out_map) {
  iree_host_size_t required_capacity = 0;
  for (uint16_t i = 0; i < value_count; ++i) {
    required_capacity =
        iree_max(required_capacity, (iree_host_size_t)values[i] + 1);
  }

  if (required_capacity > table->scratch.alias_ordinals.capacity) {
    uint16_t* new_ordinals = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, required_capacity, sizeof(*new_ordinals),
        (void**)&new_ordinals));
    memset(new_ordinals, 0, required_capacity * sizeof(*new_ordinals));
    table->scratch.alias_ordinals.values = new_ordinals;
    table->scratch.alias_ordinals.capacity = required_capacity;
  }

  uint16_t* ordinals = table->scratch.alias_ordinals.values;
  for (uint16_t i = 0; i < value_count; ++i) {
    const loom_value_id_t value_id = values[i];
    if (ordinals[value_id] == 0) ordinals[value_id] = (uint16_t)(i + 1);
  }
  *out_map = (loom_value_fact_alias_map_t){
      /*.values=*/values,
      /*.value_count=*/value_count,
      /*.ordinals=*/ordinals,
      /*.ordinal_capacity=*/table->scratch.alias_ordinals.capacity,
  };
  return iree_ok_status();
}

static void loom_value_fact_alias_map_release(loom_value_fact_table_t* table,
                                              const loom_value_id_t* values,
                                              uint16_t value_count) {
  uint16_t* ordinals = table->scratch.alias_ordinals.values;
  for (uint16_t i = 0; i < value_count; ++i) {
    ordinals[values[i]] = 0;
  }
}

static bool loom_value_fact_alias_map_resolve(
    const loom_value_fact_table_t* table,
    const loom_value_fact_alias_map_t* alias_map,
    const loom_value_facts_t* alias_facts, loom_value_id_t value_id,
    bool* out_is_alias, uint16_t* out_alias_ordinal,
    loom_value_facts_t* out_facts) {
  if (out_is_alias) *out_is_alias = false;
  if (out_alias_ordinal) *out_alias_ordinal = 0;
  uint16_t alias_ordinal = 0;
  if (loom_value_fact_alias_map_find(alias_map, value_id, &alias_ordinal)) {
    if (out_is_alias) *out_is_alias = true;
    if (out_alias_ordinal) *out_alias_ordinal = alias_ordinal;
    *out_facts = alias_facts[alias_ordinal];
    return true;
  }
  return loom_value_fact_table_try_lookup(table, value_id, out_facts);
}

static void loom_value_fact_table_apply_alias_predicates_with_map(
    const loom_value_fact_table_t* table,
    const loom_value_fact_alias_map_t* alias_map,
    const loom_predicate_t* predicates, uint16_t predicate_count,
    loom_value_facts_t* inout_facts) {
  // Apply predicates whose bounds are already literals or exact aliases.
  // Predicates retaining non-exact value operands are consumed by the
  // relational interval refinement below.
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    if (predicate->arg_count == 0 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0) {
      continue;
    }
    uint16_t target_ordinal = 0;
    if (!loom_value_fact_alias_map_find(
            alias_map, (loom_value_id_t)predicate->args[0], &target_ordinal)) {
      continue;
    }

    loom_predicate_t resolved_predicate = *predicate;
    for (uint8_t argument_index = 1;
         argument_index < resolved_predicate.arg_count; ++argument_index) {
      if (resolved_predicate.arg_tags[argument_index] != LOOM_PRED_ARG_VALUE ||
          resolved_predicate.args[argument_index] < 0) {
        continue;
      }
      loom_value_facts_t argument_facts = loom_value_facts_unknown();
      if (!loom_value_fact_alias_map_resolve(
              table, alias_map, inout_facts,
              (loom_value_id_t)resolved_predicate.args[argument_index],
              /*out_is_alias=*/NULL, /*out_alias_ordinal=*/NULL,
              &argument_facts)) {
        continue;
      }
      int64_t exact_value = 0;
      if (!loom_value_facts_as_exact_i64(argument_facts, &exact_value)) {
        continue;
      }
      resolved_predicate.arg_tags[argument_index] = LOOM_PRED_ARG_CONST;
      resolved_predicate.args[argument_index] = exact_value;
    }
    loom_value_facts_apply_predicate(&inout_facts[target_ordinal],
                                     &resolved_predicate);
  }

  // Relational predicates constrain aliases from the known interval of their
  // counterpart. External facts are read-only; only aliases have result slots.
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    if (predicate->arg_count != 2 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->arg_tags[1] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0 || predicate->args[1] < 0) {
      continue;
    }

    bool lhs_is_alias = false;
    bool rhs_is_alias = false;
    uint16_t lhs_ordinal = 0;
    uint16_t rhs_ordinal = 0;
    loom_value_facts_t lhs_facts = loom_value_facts_unknown();
    loom_value_facts_t rhs_facts = loom_value_facts_unknown();
    if (!loom_value_fact_alias_map_resolve(
            table, alias_map, inout_facts, (loom_value_id_t)predicate->args[0],
            &lhs_is_alias, &lhs_ordinal, &lhs_facts) ||
        !loom_value_fact_alias_map_resolve(
            table, alias_map, inout_facts, (loom_value_id_t)predicate->args[1],
            &rhs_is_alias, &rhs_ordinal, &rhs_facts) ||
        (!lhs_is_alias && !rhs_is_alias)) {
      continue;
    }

    loom_value_facts_t* lhs_result =
        lhs_is_alias ? &inout_facts[lhs_ordinal] : NULL;
    loom_value_facts_t* rhs_result =
        rhs_is_alias ? &inout_facts[rhs_ordinal] : NULL;
    switch ((loom_predicate_kind_t)predicate->kind) {
      case LOOM_PREDICATE_EQ: {
        const int64_t range_lo =
            iree_max(lhs_facts.range_lo, rhs_facts.range_lo);
        const int64_t range_hi =
            iree_min(lhs_facts.range_hi, rhs_facts.range_hi);
        if (lhs_result) {
          lhs_result->range_lo = range_lo;
          lhs_result->range_hi = range_hi;
        }
        if (rhs_result) {
          rhs_result->range_lo = range_lo;
          rhs_result->range_hi = range_hi;
        }
        break;
      }
      case LOOM_PREDICATE_LT:
        if (lhs_result && rhs_facts.range_hi > INT64_MIN) {
          lhs_result->range_hi =
              iree_min(lhs_result->range_hi, rhs_facts.range_hi - 1);
        }
        if (rhs_result && lhs_facts.range_lo < INT64_MAX) {
          rhs_result->range_lo =
              iree_max(rhs_result->range_lo, lhs_facts.range_lo + 1);
        }
        break;
      case LOOM_PREDICATE_LE:
        if (lhs_result) {
          lhs_result->range_hi =
              iree_min(lhs_result->range_hi, rhs_facts.range_hi);
        }
        if (rhs_result) {
          rhs_result->range_lo =
              iree_max(rhs_result->range_lo, lhs_facts.range_lo);
        }
        break;
      case LOOM_PREDICATE_GT:
        if (lhs_result && rhs_facts.range_lo < INT64_MAX) {
          lhs_result->range_lo =
              iree_max(lhs_result->range_lo, rhs_facts.range_lo + 1);
        }
        if (rhs_result && lhs_facts.range_hi > INT64_MIN) {
          rhs_result->range_hi =
              iree_min(rhs_result->range_hi, lhs_facts.range_hi - 1);
        }
        break;
      case LOOM_PREDICATE_GE:
        if (lhs_result) {
          lhs_result->range_lo =
              iree_max(lhs_result->range_lo, rhs_facts.range_lo);
        }
        if (rhs_result) {
          rhs_result->range_hi =
              iree_min(rhs_result->range_hi, lhs_facts.range_hi);
        }
        break;
      default:
        continue;
    }
    if (lhs_result) loom_value_facts_recompute_flags(lhs_result);
    if (rhs_result && rhs_result != lhs_result) {
      loom_value_facts_recompute_flags(rhs_result);
    }
  }
}

iree_status_t loom_value_fact_table_apply_alias_predicates(
    loom_value_fact_table_t* table, const loom_value_id_t* values,
    uint16_t value_count, const loom_predicate_t* predicates,
    uint16_t predicate_count, loom_value_facts_t* inout_facts) {
  loom_value_fact_alias_map_t alias_map = {
      /*.values=*/values,
      /*.value_count=*/value_count,
      /*.ordinals=*/NULL,
      /*.ordinal_capacity=*/0,
  };
  const bool use_direct_map =
      value_count > LOOM_VALUE_FACT_ALIAS_LINEAR_SCAN_LIMIT &&
      predicate_count > LOOM_VALUE_FACT_PREDICATE_LINEAR_SCAN_LIMIT;
  if (use_direct_map) {
    IREE_RETURN_IF_ERROR(loom_value_fact_alias_map_prepare(
        table, values, value_count, &alias_map));
  }

  loom_value_fact_table_apply_alias_predicates_with_map(
      table, &alias_map, predicates, predicate_count, inout_facts);
  if (use_direct_map) {
    loom_value_fact_alias_map_release(table, values, value_count);
  }
  return iree_ok_status();
}
