// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/kernel_unit.h"

#include <string.h>

#include "loom/analysis/symbol_liveness.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/transforms/cleanup/canonicalize.h"

// Returns the scalar facts that remain meaningful when crossing from a host
// command program into a device kernel invocation. Extension payloads and
// execution-distribution facts belong to the source function's SSA domain and
// cannot be retained by ordinal in a separately linked kernel unit.
static loom_value_facts_t loom_cmd_kernel_unit_boundary_facts(
    loom_value_facts_t facts) {
  facts.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  facts.flags &= ~(
      LOOM_VALUE_FACT_DISTRIBUTION_MASK | LOOM_VALUE_FACT_TOPOLOGY_DOMAIN_MASK |
      LOOM_VALUE_FACT_LANE_PREDICATE | LOOM_VALUE_FACT_SUBGROUP_LANE_MASK);
  return facts;
}

static bool loom_cmd_kernel_unit_value_groups_equivalent(
    const loom_module_t* source_module, loom_value_slice_t lhs_values,
    loom_value_slice_t rhs_values,
    const loom_value_fact_table_t* source_facts) {
  IREE_ASSERT_EQ(lhs_values.count, rhs_values.count);
  for (uint16_t i = 0; i < lhs_values.count; ++i) {
    const loom_type_t type =
        loom_module_value_type(source_module, lhs_values.values[i]);
    if (!loom_type_is_scalar(type)) continue;
    const loom_value_facts_t lhs_facts = loom_cmd_kernel_unit_boundary_facts(
        loom_value_fact_table_lookup(source_facts, lhs_values.values[i]));
    const loom_value_facts_t rhs_facts = loom_cmd_kernel_unit_boundary_facts(
        loom_value_fact_table_lookup(source_facts, rhs_values.values[i]));
    if (!loom_value_facts_equal(lhs_facts, rhs_facts)) return false;
  }
  return true;
}

bool loom_cmd_kernel_unit_boundaries_equivalent(
    const loom_module_t* source_module, const loom_op_t* lhs_launch_op,
    const loom_op_t* rhs_launch_op,
    const loom_value_fact_table_t* source_facts) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(lhs_launch_op);
  IREE_ASSERT_ARGUMENT(rhs_launch_op);
  IREE_ASSERT_ARGUMENT(source_facts);
  IREE_ASSERT(loom_kernel_launch_isa(lhs_launch_op));
  IREE_ASSERT(loom_kernel_launch_isa(rhs_launch_op));

  return loom_cmd_kernel_unit_value_groups_equivalent(
             source_module, loom_kernel_launch_workloads(lhs_launch_op),
             loom_kernel_launch_workloads(rhs_launch_op), source_facts) &&
         loom_cmd_kernel_unit_value_groups_equivalent(
             source_module, loom_kernel_launch_arguments(lhs_launch_op),
             loom_kernel_launch_arguments(rhs_launch_op), source_facts);
}

static bool loom_cmd_kernel_unit_symbol_is_template_provider(
    const loom_symbol_t* symbol) {
  return symbol->kind == LOOM_SYMBOL_TEMPLATE_DEF ||
         symbol->kind == LOOM_SYMBOL_TEMPLATE_UKERNEL;
}

static iree_status_t loom_cmd_kernel_unit_mark_template_providers(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_template_demand_t* demand) {
  (void)user_data;
  const loom_module_t* module = context->module;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loom_cmd_kernel_unit_symbol_is_template_provider(symbol)) continue;
    IREE_ASSERT(symbol->defining_op);
    const loom_func_like_t provider =
        loom_func_like_cast(module, symbol->defining_op);
    IREE_ASSERT(loom_func_like_isa(provider));
    const loom_symbol_ref_t family = loom_func_like_template_family(provider);
    IREE_ASSERT(loom_symbol_ref_is_valid(family));
    IREE_ASSERT_EQ(family.module_id, 0u);
    IREE_ASSERT_LT(family.symbol_id, module->symbols.count);
    if (family.symbol_id != demand->family_symbol_id) continue;
    IREE_RETURN_IF_ERROR(
        loom_symbol_liveness_mark_symbol_id(context, (loom_symbol_id_t)i));
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_kernel_unit_source_prepare(
    const loom_module_t* source_module, loom_op_t* source_kernel_op,
    const loom_symbol_reference_table_t* references,
    iree_arena_allocator_t* arena, loom_cmd_kernel_unit_source_t* out_source) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(source_kernel_op);
  IREE_ASSERT_ARGUMENT(references);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_source);
  IREE_ASSERT_EQ(references->module, source_module);
  IREE_ASSERT_EQ(references->symbol_count, source_module->symbols.count);
  IREE_ASSERT(loom_kernel_def_isa(source_kernel_op));
  *out_source = (loom_cmd_kernel_unit_source_t){0};

  const loom_symbol_ref_t kernel_ref = loom_func_like_callee(
      loom_func_like_cast(source_module, source_kernel_op));
  IREE_ASSERT(loom_symbol_ref_is_valid(kernel_ref));
  IREE_ASSERT_EQ(kernel_ref.module_id, 0u);
  IREE_ASSERT_LT(kernel_ref.symbol_id, source_module->symbols.count);

  const loom_symbol_liveness_contributor_t contributor = {
      .visit_template_demand = loom_cmd_kernel_unit_mark_template_providers,
  };
  const bool has_template_demands = references->template_demands.count != 0;
  const loom_symbol_liveness_options_t options = {
      .contributors = has_template_demands ? &contributor : NULL,
      .contributor_count = has_template_demands ? 1 : 0,
      .root_symbol_ids =
          {
              .values = &kernel_ref.symbol_id,
              .count = 1,
          },
  };
  loom_symbol_liveness_t liveness = {0};
  IREE_RETURN_IF_ERROR(loom_symbol_liveness_compute(
      source_module, references, &options, arena, &liveness));

  iree_host_size_t selected_count = 0;
  for (iree_host_size_t i = 0; i < liveness.symbol_count; ++i) {
    selected_count += liveness.live_symbols[i] != 0;
  }
  iree_host_size_t* selected_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, selected_count,
                                                 sizeof(*selected_ordinals),
                                                 (void**)&selected_ordinals));
  iree_host_size_t selected_ordinal = 0;
  iree_host_size_t kernel_selection_ordinal = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < liveness.symbol_count; ++i) {
    if (!liveness.live_symbols[i]) continue;
    if (i == kernel_ref.symbol_id) {
      kernel_selection_ordinal = selected_ordinal;
    }
    selected_ordinals[selected_ordinal++] = i;
  }
  IREE_ASSERT_EQ(selected_ordinal, selected_count);
  IREE_ASSERT_NE(kernel_selection_ordinal, IREE_HOST_SIZE_MAX);

  *out_source = (loom_cmd_kernel_unit_source_t){
      .module = source_module,
      .kernel_op = source_kernel_op,
      .symbols =
          {
              .ordinals = selected_ordinals,
              .count = selected_count,
              .kernel_selection_ordinal = kernel_selection_ordinal,
          },
  };
  return iree_ok_status();
}

static void loom_cmd_kernel_unit_clear_explicit_export(
    loom_func_like_t kernel) {
  loom_attribute_t* attrs = loom_op_attrs(kernel.op);
  if (kernel.vtable->export_symbol_attr_index != LOOM_ATTR_INDEX_NONE) {
    attrs[kernel.vtable->export_symbol_attr_index] = loom_attr_absent();
  }
  if (kernel.vtable->export_linkage_attr_index != LOOM_ATTR_INDEX_NONE) {
    attrs[kernel.vtable->export_linkage_attr_index] = loom_attr_absent();
  }
  if (kernel.vtable->export_attrs_attr_index != LOOM_ATTR_INDEX_NONE) {
    attrs[kernel.vtable->export_attrs_attr_index] = loom_attr_absent();
  }
}

// Returns true when every SSA value referenced by |predicate| is exact at the
// source launch boundary. Such a predicate has been fully consumed by unit
// specialization and carries no remaining dynamic contract information.
static bool loom_cmd_kernel_unit_predicate_is_exact(
    const loom_predicate_t* predicate,
    const loom_value_fact_table_t* seed_facts) {
  bool has_value = false;
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE) continue;
    has_value = true;
    const int64_t raw_value = predicate->args[i];
    IREE_ASSERT_GE(raw_value, 0);
    IREE_ASSERT_LE(raw_value, UINT32_MAX);
    if (!loom_value_facts_is_exact(loom_value_fact_table_lookup(
            seed_facts, (loom_value_id_t)raw_value))) {
      return false;
    }
  }
  return has_value;
}

// Removes source contract predicates fully discharged by exact unit facts.
// Predicates involving any dynamic value remain attached, preserving
// relational contracts that cannot be represented by per-value assumes.
static iree_status_t loom_cmd_kernel_unit_consume_exact_predicates(
    loom_module_t* module, loom_func_like_t kernel,
    const loom_value_fact_table_t* seed_facts) {
  if (kernel.vtable->predicates_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(kernel, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();

  uint16_t consumed_count = 0;
  for (uint16_t i = 0; i < predicate_count; ++i) {
    if (loom_cmd_kernel_unit_predicate_is_exact(&predicates[i], seed_facts)) {
      ++consumed_count;
    }
  }
  if (consumed_count == 0) return iree_ok_status();

  const uint16_t retained_count = predicate_count - consumed_count;
  loom_predicate_t* retained_predicates = NULL;
  if (retained_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, retained_count, sizeof(*retained_predicates),
        (void**)&retained_predicates));
    uint16_t retained_ordinal = 0;
    for (uint16_t i = 0; i < predicate_count; ++i) {
      if (loom_cmd_kernel_unit_predicate_is_exact(&predicates[i], seed_facts)) {
        continue;
      }
      retained_predicates[retained_ordinal++] = predicates[i];
    }
    IREE_ASSERT_EQ(retained_ordinal, retained_count);
  }

  loom_op_attrs(kernel.op)[kernel.vtable->predicates_attr_index] =
      retained_count > 0
          ? loom_attr_predicate_list(retained_predicates, retained_count)
          : loom_attr_absent();
  return iree_ok_status();
}

static iree_status_t loom_cmd_kernel_unit_seed_argument_group(
    const loom_value_fact_table_t* source_facts,
    loom_value_slice_t source_values, loom_module_t* unit_module,
    const loom_value_id_t* unit_values, uint16_t unit_value_count,
    loom_value_fact_table_t* unit_seed_facts) {
  IREE_ASSERT_EQ(source_values.count, unit_value_count);
  for (uint16_t i = 0; i < source_values.count; ++i) {
    const loom_value_id_t unit_value = unit_values[i];
    const loom_type_t unit_type =
        loom_module_value_type(unit_module, unit_value);
    if (!loom_type_is_scalar(unit_type)) continue;

    loom_value_facts_t facts = loom_cmd_kernel_unit_boundary_facts(
        loom_value_fact_table_lookup(source_facts, source_values.values[i]));
    if (loom_value_facts_is_unknown(facts)) continue;
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(unit_seed_facts, unit_value, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_kernel_unit_seed_facts(
    const loom_op_t* source_launch_op,
    const loom_value_fact_table_t* source_facts, loom_module_t* unit_module,
    loom_func_like_t unit_kernel, loom_value_fact_table_t* unit_seed_facts) {
  const loom_value_slice_t source_workloads =
      loom_kernel_launch_workloads(source_launch_op);
  const loom_value_slice_t unit_workloads =
      loom_kernel_workload_arg_ids(unit_module, unit_kernel.op);
  IREE_RETURN_IF_ERROR(loom_cmd_kernel_unit_seed_argument_group(
      source_facts, source_workloads, unit_module, unit_workloads.values,
      unit_workloads.count, unit_seed_facts));

  const loom_value_slice_t source_arguments =
      loom_kernel_launch_arguments(source_launch_op);
  uint16_t unit_argument_count = 0;
  const loom_value_id_t* unit_argument_ids =
      loom_func_like_arg_ids(unit_kernel, &unit_argument_count);
  return loom_cmd_kernel_unit_seed_argument_group(
      source_facts, source_arguments, unit_module, unit_argument_ids,
      unit_argument_count, unit_seed_facts);
}

static bool loom_cmd_kernel_unit_argument_is_unused(const loom_module_t* module,
                                                    loom_value_id_t argument) {
  IREE_ASSERT_LT(argument, module->values.count);
  const loom_value_t* value = loom_module_value(module, argument);
  return value->use_count == 0 &&
         !loom_module_value_has_predicate_attribute_uses(module, argument) &&
         !loom_module_value_has_type_uses(module, argument);
}

static bool loom_cmd_kernel_unit_select_operand_use(const loom_op_t* user_op,
                                                    void* user_data) {
  (void)user_op;
  (void)user_data;
  return true;
}

// Reifies exact device-ABI facts inside the body so the derived unit remains
// independently compilable after the transient fact table is gone.
static iree_status_t loom_cmd_kernel_unit_materialize_exact_arguments(
    loom_module_t* module, loom_func_like_t kernel,
    const loom_value_id_t* arguments, uint16_t argument_count,
    const loom_value_fact_table_t* seed_facts) {
  if (argument_count == 0) return iree_ok_status();

  const loom_value_t* first_argument = loom_module_value(module, arguments[0]);
  IREE_ASSERT(loom_value_is_block_arg(first_argument));
  loom_block_t* entry_block = loom_value_def_block(first_argument);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  if (entry_block->first_op) {
    loom_builder_set_before(&builder, entry_block->first_op);
  } else {
    builder.ip.parent_op = kernel.op;
  }

  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_value_id_t argument = arguments[i];
    IREE_ASSERT_EQ(loom_value_def_block(loom_module_value(module, argument)),
                   entry_block);
    if (loom_cmd_kernel_unit_argument_is_unused(module, argument)) continue;
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(seed_facts, argument);
    const loom_type_t type = loom_module_value_type(module, argument);
    if (!loom_value_facts_can_materialize_constant(facts, type)) continue;

    loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_constant_build(
        &builder, facts, type, kernel.op->location, &replacement));
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, argument, replacement));
    IREE_RETURN_IF_ERROR(
        loom_module_replace_value_type_uses(module, argument, replacement));
    IREE_RETURN_IF_ERROR(loom_region_replace_attribute_value_references(
        module, loom_func_like_body(kernel), argument, replacement));
    // Keep function-contract predicates on the formal argument. Moving their
    // references to a body-local constant would make the serialized function
    // metadata depend on a value that is not in scope until its body is read.
    IREE_RETURN_IF_ERROR(loom_value_replace_uses_if(
        module, argument, replacement, loom_cmd_kernel_unit_select_operand_use,
        NULL));
  }
  return iree_ok_status();
}

#define LOOM_CMD_KERNEL_UNIT_PREDICATE_CAPACITY 4

static uint16_t loom_cmd_kernel_unit_predicates_from_facts(
    loom_value_id_t value, loom_type_t type, loom_value_facts_t facts,
    loom_predicate_t* predicates) {
  if (!loom_type_is_scalar(type) || loom_value_facts_is_exact(facts) ||
      loom_value_facts_is_float(facts) || loom_value_facts_is_unknown(facts)) {
    return 0;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type != LOOM_SCALAR_TYPE_INDEX &&
      scalar_type != LOOM_SCALAR_TYPE_OFFSET) {
    return 0;
  }

  int64_t domain_lo = 0;
  int64_t domain_hi = 0;
  const bool has_domain =
      loom_value_facts_scalar_type_domain(scalar_type, &domain_lo, &domain_hi);
  IREE_ASSERT(has_domain);
  (void)has_domain;
  uint16_t count = 0;
  if (facts.range_lo > domain_lo || facts.range_hi < domain_hi) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_RANGE,
        .arg_count = 3,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_CONST},
        .args = {value, facts.range_lo, facts.range_hi},
    };
  }
  if (facts.known_divisor > 1) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_MUL,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {value, facts.known_divisor, 0},
    };
  }
  if (loom_value_facts_is_power_of_two(facts)) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_POW2,
        .arg_count = 1,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_NONE,
                     LOOM_PRED_ARG_NONE},
        .args = {value, 0, 0},
    };
  }
  if (loom_value_facts_is_non_zero(facts) && facts.range_lo <= 0 &&
      facts.range_hi >= 0) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_NE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {value, 0, 0},
    };
  }
  IREE_ASSERT_LE(count, LOOM_CMD_KERNEL_UNIT_PREDICATE_CAPACITY);
  return count;
}

// Persists abstract address facts as ordinary identity assumptions in the
// derived body. Consumers use the assumed result while the formal remains
// available as the unit's dynamic device ABI input.
static iree_status_t loom_cmd_kernel_unit_materialize_abstract_arguments(
    loom_module_t* module, loom_func_like_t kernel,
    const loom_value_id_t* arguments, uint16_t argument_count,
    const loom_value_fact_table_t* seed_facts) {
  if (argument_count == 0) return iree_ok_status();

  const loom_value_t* first_argument = loom_module_value(module, arguments[0]);
  IREE_ASSERT(loom_value_is_block_arg(first_argument));
  loom_block_t* entry_block = loom_value_def_block(first_argument);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  if (entry_block->first_op) {
    loom_builder_set_before(&builder, entry_block->first_op);
  } else {
    builder.ip.parent_op = kernel.op;
  }

  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_value_id_t argument = arguments[i];
    IREE_ASSERT_EQ(loom_value_def_block(loom_module_value(module, argument)),
                   entry_block);
    if (loom_cmd_kernel_unit_argument_is_unused(module, argument)) continue;
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(seed_facts, argument);
    const loom_type_t type = loom_module_value_type(module, argument);
    loom_predicate_t predicates[LOOM_CMD_KERNEL_UNIT_PREDICATE_CAPACITY] = {0};
    const uint16_t predicate_count = loom_cmd_kernel_unit_predicates_from_facts(
        argument, type, facts, predicates);
    if (predicate_count == 0) continue;

    loom_op_t* assume_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_assume_build(
        &builder, &argument, 1, predicates, predicate_count, &type, 1,
        kernel.op->location, &assume_op));
    const loom_value_id_t replacement =
        loom_index_assume_results(assume_op).values[0];
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, argument, replacement));
    IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_except(
        module, argument, replacement, assume_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_kernel_unit_prune_argument_group(
    loom_module_t* module, loom_block_t* entry_block,
    const loom_value_id_t* source_argument_ids, uint16_t source_argument_count,
    uint16_t* out_argument_count,
    const uint16_t** out_source_argument_ordinals) {
  uint16_t retained_count = 0;
  for (uint16_t i = 0; i < source_argument_count; ++i) {
    if (!loom_cmd_kernel_unit_argument_is_unused(module,
                                                 source_argument_ids[i])) {
      ++retained_count;
    }
  }

  uint16_t* source_argument_ordinals = NULL;
  if (retained_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, retained_count, sizeof(*source_argument_ordinals),
        (void**)&source_argument_ordinals));
  }
  uint16_t retained_ordinal = 0;
  for (uint16_t i = 0; i < source_argument_count; ++i) {
    if (!loom_cmd_kernel_unit_argument_is_unused(module,
                                                 source_argument_ids[i])) {
      source_argument_ordinals[retained_ordinal++] = i;
    }
  }

  for (uint16_t i = source_argument_count; i > 0; --i) {
    const uint16_t argument_index = (uint16_t)(i - 1);
    if (loom_cmd_kernel_unit_argument_is_unused(
            module, loom_block_arg_id(entry_block, argument_index))) {
      IREE_RETURN_IF_ERROR(
          loom_block_remove_arg(module, entry_block, argument_index));
    }
  }

  *out_argument_count = retained_count;
  *out_source_argument_ordinals = source_argument_ordinals;
  return iree_ok_status();
}

static iree_status_t loom_cmd_kernel_unit_prune_arguments(
    loom_module_t* module, loom_func_like_t kernel,
    loom_cmd_kernel_unit_t* out_unit) {
  loom_value_slice_t source_workloads =
      loom_kernel_workload_arg_ids(module, kernel.op);
  out_unit->source_workload_count = source_workloads.count;
  if (source_workloads.count > 0) {
    loom_block_t* workload_entry = loom_value_def_block(
        loom_module_value(module, source_workloads.values[0]));
    IREE_RETURN_IF_ERROR(loom_cmd_kernel_unit_prune_argument_group(
        module, workload_entry, source_workloads.values, source_workloads.count,
        &out_unit->workload_count, &out_unit->source_workload_ordinals));
  }

  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_argument_ids =
      loom_func_like_arg_ids(kernel, &source_argument_count);
  out_unit->source_argument_count = source_argument_count;
  loom_block_t* body_entry =
      loom_region_entry_block(loom_func_like_body(kernel));
  IREE_RETURN_IF_ERROR(loom_cmd_kernel_unit_prune_argument_group(
      module, body_entry, source_argument_ids, source_argument_count,
      &out_unit->argument_count, &out_unit->source_argument_ordinals));
  return iree_ok_status();
}

// Converts typed source views into the opaque buffer roots carried by native
// device ABIs. The command descriptor retains the exact view subrange, so the
// derived kernel reconstructs the source view at byte offset zero without
// changing the source kernel contract.
static iree_status_t loom_cmd_kernel_unit_materialize_view_arguments(
    loom_module_t* module, loom_func_like_t kernel) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(kernel, &argument_count);
  if (argument_count == 0) return iree_ok_status();

  loom_block_t* entry_block =
      loom_region_entry_block(loom_func_like_body(kernel));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  if (entry_block->first_op) {
    loom_builder_set_before(&builder, entry_block->first_op);
  } else {
    builder.ip.parent_op = kernel.op;
  }

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_value_id_t argument = arguments[i];
    const loom_type_t view_type = loom_module_value_type(module, argument);
    if (!loom_type_is_view(view_type)) continue;

    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, argument, loom_type_buffer()));
    if (zero == LOOM_VALUE_ID_INVALID) {
      loom_op_t* zero_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          &builder, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
          kernel.op->location, &zero_op));
      zero = loom_index_constant_result(zero_op);
    }
    loom_op_t* view_op = NULL;
    IREE_RETURN_IF_ERROR(loom_buffer_view_build(
        &builder, argument, zero, view_type, kernel.op->location, &view_op));
    const loom_value_id_t view = loom_buffer_view_result(view_op);
    IREE_RETURN_IF_ERROR(loom_module_move_value_name(module, argument, view));
    IREE_RETURN_IF_ERROR(
        loom_value_replace_all_uses_except(module, argument, view, view_op));
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_kernel_unit_materialize(
    const loom_cmd_kernel_unit_source_t* source,
    const loom_op_t* source_launch_op,
    const loom_value_fact_table_t* source_facts,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_cmd_kernel_unit_t* out_unit) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(source->module);
  IREE_ASSERT_ARGUMENT(source->kernel_op);
  IREE_ASSERT_ARGUMENT(source_launch_op);
  IREE_ASSERT_ARGUMENT(source_facts);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_unit);
  memset(out_unit, 0, sizeof(*out_unit));
  IREE_ASSERT(loom_kernel_launch_isa(source_launch_op));
  IREE_ASSERT(loom_kernel_def_isa(source->kernel_op));

  const loom_module_t* source_module = source->module;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

  loom_symbol_ref_t* target_symbols = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &scratch_arena, source->symbols.count, sizeof(*target_symbols),
      (void**)&target_symbols);
  loom_linker_t* linker = NULL;
  if (iree_status_is_ok(status)) {
    status =
        loom_linker_create(source_module->context,
                           &(loom_linker_options_t){
                               .module_name = IREE_SV("command_kernel_unit"),
                           },
                           block_pool, allocator, &linker);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_add_module_symbols(
        linker, source_module,
        (loom_linker_source_symbol_list_t){
            .count = source->symbols.count,
            .ordinals = source->symbols.ordinals,
        },
        loom_linker_source_provider_import_list_empty(),
        (loom_linker_target_symbol_list_t){
            .count = source->symbols.count,
            .values = target_symbols,
        });
  }

  loom_module_t* unit_module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_linker_finish(linker, &unit_module);
  }
  loom_linker_free(linker);

  loom_op_t* unit_kernel_op = NULL;
  loom_func_like_t unit_kernel = {0};
  loom_pass_value_fact_owner_t fact_owner;
  bool fact_owner_initialized = false;
  loom_canonicalizer_t canonicalizer;
  bool canonicalizer_initialized = false;
  if (iree_status_is_ok(status)) {
    const loom_symbol_ref_t unit_kernel_ref =
        target_symbols[source->symbols.kernel_selection_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(unit_kernel_ref));
    IREE_ASSERT_EQ(unit_kernel_ref.module_id, 0u);
    IREE_ASSERT_LT(unit_kernel_ref.symbol_id, unit_module->symbols.count);
    unit_kernel_op =
        unit_module->symbols.entries[unit_kernel_ref.symbol_id].defining_op;
    IREE_ASSERT(unit_kernel_op);
    unit_kernel = loom_func_like_cast(unit_module, unit_kernel_op);
    IREE_ASSERT(loom_func_like_is_kernel_entry(unit_kernel));
    loom_cmd_kernel_unit_clear_explicit_export(unit_kernel);
    loom_func_like_set_retained(unit_module, unit_kernel, true);

    loom_pass_value_fact_owner_initialize(block_pool, &fact_owner);
    fact_owner_initialized = true;
    status = loom_canonicalizer_initialize(unit_module, &scratch_arena,
                                           &fact_owner, &canonicalizer);
    canonicalizer_initialized = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status)) {
    status = loom_cmd_kernel_unit_materialize_view_arguments(unit_module,
                                                             unit_kernel);
  }

  loom_value_fact_table_t seed_facts;
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_initialize(&seed_facts, &scratch_arena,
                                              unit_module->values.count);
  }
  if (iree_status_is_ok(status)) {
    loom_type_registry_configure_fact_context(&seed_facts.context);
    status = loom_cmd_kernel_unit_seed_facts(
        source_launch_op, source_facts, unit_module, unit_kernel, &seed_facts);
  }
  if (iree_status_is_ok(status)) {
    uint16_t unit_argument_count = 0;
    const loom_value_id_t* unit_arguments =
        loom_func_like_arg_ids(unit_kernel, &unit_argument_count);
    status = loom_cmd_kernel_unit_materialize_exact_arguments(
        unit_module, unit_kernel, unit_arguments, unit_argument_count,
        &seed_facts);
  }
  if (iree_status_is_ok(status)) {
    uint16_t unit_argument_count = 0;
    const loom_value_id_t* unit_arguments =
        loom_func_like_arg_ids(unit_kernel, &unit_argument_count);
    status = loom_cmd_kernel_unit_materialize_abstract_arguments(
        unit_module, unit_kernel, unit_arguments, unit_argument_count,
        &seed_facts);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_kernel_unit_consume_exact_predicates(
        unit_module, unit_kernel, &seed_facts);
  }
  if (iree_status_is_ok(status)) {
    loom_canonicalizer_result_t result = {0};
    status = loom_canonicalizer_run_function(&canonicalizer, unit_kernel,
                                             &(loom_canonicalizer_options_t){
                                                 .seed_facts = &seed_facts,
                                             },
                                             &result);
  }
  if (iree_status_is_ok(status)) {
    out_unit->module = unit_module;
    out_unit->kernel_op = unit_kernel_op;
    status = loom_cmd_kernel_unit_prune_arguments(unit_module, unit_kernel,
                                                  out_unit);
  }

  if (canonicalizer_initialized) {
    loom_canonicalizer_deinitialize(&canonicalizer);
  }
  if (fact_owner_initialized) {
    loom_pass_value_fact_owner_deinitialize(&fact_owner);
  }
  iree_arena_deinitialize(&scratch_arena);
  if (!iree_status_is_ok(status)) {
    if (unit_module) loom_module_free(unit_module);
    memset(out_unit, 0, sizeof(*out_unit));
  }
  return status;
}

void loom_cmd_kernel_unit_deinitialize(loom_cmd_kernel_unit_t* unit) {
  if (!unit) return;
  if (unit->module) loom_module_free(unit->module);
  memset(unit, 0, sizeof(*unit));
}
