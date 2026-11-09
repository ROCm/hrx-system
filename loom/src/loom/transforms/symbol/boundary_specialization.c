// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/boundary_specialization.h"

#include <stdio.h>
#include <string.h>

#include "loom/analysis/type_refinement.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/util/fact_table.h"

typedef struct loom_refine_boundaries_specialization_group_t {
  // Number of entries in |argument_types|.
  uint16_t argument_count;

  // Fact-specialized argument types normalized to callee argument identities.
  loom_type_t* argument_types;

  // Fact-specialized result types normalized to callee boundary identities.
  loom_type_t* result_types;

  // Calls whose normalized argument and result types match this group.
  loom_op_t** calls;

  // Number of entries in |calls|.
  iree_host_size_t call_count;

  // Allocated call pointer count.
  iree_host_size_t call_capacity;

  // Symbol created for this specialization.
  loom_symbol_ref_t symbol_ref;
} loom_refine_boundaries_specialization_group_t;

typedef struct loom_refine_boundaries_specialization_function_plan_t {
  // Specialization groups for one private function.
  loom_refine_boundaries_specialization_group_t* groups;

  // Number of entries in |groups|.
  iree_host_size_t group_count;

  // Allocated specialization group count.
  iree_host_size_t group_capacity;
} loom_refine_boundaries_specialization_function_plan_t;

struct loom_refine_boundaries_specialization_plans_t {
  // Dense function plans indexed by boundary graph node.
  loom_refine_boundaries_specialization_function_plan_t* functions;

  // Arena owning the plan, groups, normalized types, and call lists.
  iree_arena_allocator_t* arena;

  // True when specialization keys preserve dynamic SSA dimensions.
  bool encodings_only;
};

static bool loom_refine_boundaries_can_specialize_function(
    const loom_refine_boundaries_function_t* function_info) {
  return function_info->can_refine_boundary &&
         loom_func_def_isa(function_info->function.op) &&
         (function_info->argument_count > 0 || function_info->result_count > 0);
}

static iree_status_t loom_refine_boundaries_specialize_call_boundary_types(
    loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results,
    bool encodings_only, iree_arena_allocator_t* arena,
    loom_type_t** out_argument_types, loom_type_t** out_result_types) {
  *out_argument_types = NULL;
  *out_result_types = NULL;

  loom_ir_remap_t remap = {0};
  const loom_ir_remap_options_t remap_options = {
      .allow_unmapped_values = true,
  };
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(module, module, arena, &remap_options, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
      &remap, operands.values, callee_info->argument_ids, operands.count));
  const loom_value_id_t* callee_results =
      loom_op_const_results(callee_info->function.op);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, results.values,
                                                callee_results, results.count));

  loom_type_t* argument_types = NULL;
  if (operands.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, operands.count,
                                                   sizeof(*argument_types),
                                                   (void**)&argument_types));
  }
  for (iree_host_size_t i = 0; i < operands.count; ++i) {
    loom_type_t specialized_type =
        loom_module_value_type(module, operands.values[i]);
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    if (encodings_only) {
      IREE_RETURN_IF_ERROR(loom_type_specialize_encoding_with_value_facts(
          module, specialized_type, facts, &module->arena, &specialized_type,
          &result));
    } else {
      IREE_RETURN_IF_ERROR(loom_type_specialize_with_value_facts(
          module, specialized_type, facts, &module->arena, &specialized_type,
          &result));
    }
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      IREE_ASSERT_UNREACHABLE(
          "verified call boundary facts conflict with its argument type");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, specialized_type, &argument_types[i]));
  }

  loom_type_t* result_types = NULL;
  if (results.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, results.count, sizeof(*result_types), (void**)&result_types));
  }
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    loom_type_t specialized_type =
        loom_module_value_type(module, results.values[i]);
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    if (encodings_only) {
      IREE_RETURN_IF_ERROR(loom_type_specialize_encoding_with_value_facts(
          module, specialized_type, facts, &module->arena, &specialized_type,
          &result));
    } else {
      IREE_RETURN_IF_ERROR(loom_type_specialize_with_value_facts(
          module, specialized_type, facts, &module->arena, &specialized_type,
          &result));
    }
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      IREE_ASSERT_UNREACHABLE(
          "verified call boundary facts conflict with its result type");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, specialized_type, &result_types[i]));
  }

  *out_argument_types = argument_types;
  *out_result_types = result_types;
  return iree_ok_status();
}

static bool loom_refine_boundaries_group_matches_call_boundary(
    const loom_refine_boundaries_specialization_group_t* group,
    const loom_type_t* argument_types, const loom_type_t* result_types,
    uint16_t result_count) {
  for (uint16_t i = 0; i < group->argument_count; ++i) {
    if (!loom_type_equal(group->argument_types[i], argument_types[i])) {
      return false;
    }
  }
  for (uint16_t i = 0; i < result_count; ++i) {
    if (!loom_type_equal(group->result_types[i], result_types[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_refine_boundaries_append_specialization_call(
    loom_refine_boundaries_specialization_group_t* group, loom_op_t* call_op,
    iree_arena_allocator_t* arena) {
  if (group->call_count >= group->call_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, group->call_count, group->call_count + 1, sizeof(*group->calls),
        &group->call_capacity, (void**)&group->calls));
  }
  group->calls[group->call_count++] = call_op;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_add_specialization_group(
    loom_refine_boundaries_specialization_function_plan_t* plan,
    uint16_t argument_count, loom_type_t* argument_types,
    loom_type_t* result_types, iree_arena_allocator_t* arena,
    loom_refine_boundaries_specialization_group_t** out_group) {
  *out_group = NULL;
  if (plan->group_count >= plan->group_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, plan->group_count, plan->group_count + 1, sizeof(*plan->groups),
        &plan->group_capacity, (void**)&plan->groups));
  }

  loom_refine_boundaries_specialization_group_t* group =
      &plan->groups[plan->group_count++];
  memset(group, 0, sizeof(*group));
  group->symbol_ref = loom_symbol_ref_null();
  group->argument_count = argument_count;
  group->argument_types = argument_types;
  group->result_types = result_types;

  *out_group = group;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_record_specialization_call(
    loom_module_t* module, const loom_refine_boundaries_function_t* callee_info,
    loom_refine_boundaries_specialization_function_plan_t* plan,
    loom_op_t* call_op, loom_value_slice_t operands, loom_value_slice_t results,
    const loom_value_fact_table_t* facts, bool encodings_only,
    iree_arena_allocator_t* arena) {
  loom_type_t* argument_types = NULL;
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_specialize_call_boundary_types(
      module, facts, callee_info, operands, results, encodings_only, arena,
      &argument_types, &result_types));

  for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
    loom_refine_boundaries_specialization_group_t* group = &plan->groups[i];
    if (!loom_refine_boundaries_group_matches_call_boundary(
            group, argument_types, result_types, callee_info->result_count)) {
      continue;
    }
    return loom_refine_boundaries_append_specialization_call(group, call_op,
                                                             arena);
  }

  loom_refine_boundaries_specialization_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_add_specialization_group(
      plan, callee_info->argument_count, argument_types, result_types, arena,
      &group));
  return loom_refine_boundaries_append_specialization_call(group, call_op,
                                                           arena);
}

iree_status_t loom_refine_boundaries_specialization_plans_collect_call(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_specialization_plans_t* plans,
    const loom_value_fact_table_t* function_facts, loom_op_t* op) {
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(module, op, NULL, &callee, &operands,
                                        &results)) {
    return iree_ok_status();
  }
  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(graph, callee, &callee_node)) {
    return iree_ok_status();
  }
  const loom_refine_boundaries_function_t* callee_info =
      &graph->functions[callee_node];
  if (!loom_refine_boundaries_can_specialize_function(callee_info)) {
    return iree_ok_status();
  }

  return loom_refine_boundaries_record_specialization_call(
      module, callee_info, &plans->functions[callee_node], op, operands,
      results, function_facts, plans->encodings_only, plans->arena);
}

static iree_status_t loom_refine_boundaries_make_specialization_symbol(
    loom_module_t* module, loom_symbol_ref_t source_ref,
    iree_host_size_t preferred_ordinal, iree_arena_allocator_t* arena,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  const loom_symbol_t* source_symbol =
      &module->symbols.entries[source_ref.symbol_id];
  iree_string_view_t source_name =
      loom_string_table_get(&module->strings, source_symbol->name_id);

  for (iree_host_size_t ordinal = preferred_ordinal;
       ordinal < IREE_HOST_SIZE_MAX; ++ordinal) {
    char suffix[32] = {0};
    int suffix_length =
        snprintf(suffix, sizeof(suffix), "_spec%" PRIhsz, ordinal);
    if (suffix_length < 0 ||
        (iree_host_size_t)suffix_length >= sizeof(suffix)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "specialization symbol suffix overflow");
    }

    iree_host_size_t name_length = 0;
    if (!iree_host_size_checked_add(
            source_name.size, (iree_host_size_t)suffix_length, &name_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "specialization symbol name overflow");
    }

    char* name_storage = NULL;
    if (name_length > 0) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, name_length, (void**)&name_storage));
      memcpy(name_storage, source_name.data, source_name.size);
      memcpy(name_storage + source_name.size, suffix,
             (iree_host_size_t)suffix_length);
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, iree_make_string_view(name_storage, name_length), &name_id));
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_symbol_ref =
        (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "could not create a unique specialization symbol");
}

static iree_status_t loom_refine_boundaries_clone_function_specialization(
    loom_module_t* module,
    const loom_refine_boundaries_function_t* function_info,
    const loom_refine_boundaries_specialization_group_t* group,
    loom_symbol_ref_t target_ref, const loom_op_t* insertion_anchor,
    iree_arena_allocator_t* arena, loom_op_t** out_op) {
  *out_op = NULL;
  loom_op_t* source_op = function_info->function.op;
  loom_region_t* source_body = loom_func_like_body(function_info->function);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, module, arena,
                                                /*options=*/NULL, &remap));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, source_op->parent_block,
                          &builder);
  loom_builder_set_after(&builder, insertion_anchor);

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_location_id(&remap, source_op->location, &target_location));

  loom_op_t* target_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &builder, LOOM_OP_FUNC_DEF, 0, source_op->result_count, 1,
      source_op->tied_result_count, source_op->attribute_count, target_location,
      &target_op));
  target_op->instance_flags = source_op->instance_flags;
  target_op->traits = source_op->traits;
  target_op->intrinsic_write_spaces = source_op->intrinsic_write_spaces;

  loom_attribute_t* target_attrs = loom_op_attrs(target_op);
  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  for (uint8_t i = 0; i < source_op->attribute_count; ++i) {
    target_attrs[i] = source_attrs[i];
  }
  target_attrs[function_info->function.vtable->callee_attr_index] =
      loom_attr_symbol(target_ref);
  if (function_info->function.vtable->predicates_attr_index !=
      LOOM_ATTR_INDEX_NONE) {
    target_attrs[function_info->function.vtable->predicates_attr_index] =
        (loom_attribute_t){0};
  }

  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  loom_value_id_t* target_results = loom_op_results(target_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_define_value(module, loom_type_none(), &target_results[i]));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, source_results[i], target_results[i]));
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(module, source_results[i],
                                                     target_results[i]));
  }

  if (source_op->tied_result_count > 0) {
    memcpy(loom_op_tied_results(target_op), loom_op_tied_results(source_op),
           (iree_host_size_t)source_op->tied_result_count *
               sizeof(loom_tied_result_t));
  }

  loom_builder_ip_t saved_ip = loom_builder_save(&builder);
  builder.ip.parent_op = target_op;
  loom_region_t* target_body = NULL;
  iree_status_t status =
      loom_ir_clone_region(&builder, source_body, &remap, &target_body);
  loom_builder_restore(&builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  loom_op_set_region(target_op, 0, target_body);

  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function_info->function, &predicate_count);
  if (predicate_count > 0) {
    loom_predicate_t* target_predicates = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
        &remap, predicates, predicate_count, &target_predicates));
    target_attrs[function_info->function.vtable->predicates_attr_index] =
        loom_attr_predicate_list(target_predicates, predicate_count);
  }

  for (uint16_t i = 0; i < function_info->argument_count; ++i) {
    loom_value_id_t target_argument = LOOM_VALUE_ID_INVALID;
    if (!loom_ir_remap_try_lookup_value(&remap, function_info->argument_ids[i],
                                        &target_argument)) {
      IREE_ASSERT_UNREACHABLE(
          "cloned function is missing an argument value mapping");
      IREE_BUILTIN_UNREACHABLE();
    }
    loom_type_t target_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, group->argument_types[i], &target_type));
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, target_argument, target_type));
  }
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    loom_type_t target_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, group->result_types[i], &target_type));
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, target_results[i], target_type));
  }

  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(&builder, target_op));
  *out_op = target_op;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_create_specialization_group(
    loom_module_t* module,
    const loom_refine_boundaries_function_t* function_info,
    loom_refine_boundaries_specialization_group_t* group,
    iree_host_size_t group_ordinal, const loom_op_t* insertion_anchor,
    iree_arena_allocator_t* arena, loom_op_t** out_op) {
  *out_op = NULL;
  loom_symbol_ref_t source_ref = loom_func_like_callee(function_info->function);
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_make_specialization_symbol(
      module, source_ref, group_ordinal, arena, &group->symbol_ref));

  return loom_refine_boundaries_clone_function_specialization(
      module, function_info, group, group->symbol_ref, insertion_anchor, arena,
      out_op);
}

static void loom_refine_boundaries_retarget_call(loom_module_t* module,
                                                 loom_op_t* call_op,
                                                 loom_symbol_ref_t callee) {
  loom_call_like_set_callee(module, loom_call_like_cast(module, call_op),
                            callee);
}

static iree_status_t loom_refine_boundaries_create_specializations(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_specialization_function_plan_t* plans,
    iree_arena_allocator_t* arena, int64_t* out_specialization_count) {
  *out_specialization_count = 0;
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    loom_refine_boundaries_specialization_function_plan_t* plan = &plans[node];
    if (plan->group_count < 2) {
      continue;
    }

    const loom_refine_boundaries_function_t* function_info =
        &graph->functions[node];
    loom_op_t* insertion_anchor = function_info->function.op;
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      loom_refine_boundaries_specialization_group_t* group = &plan->groups[i];
      loom_op_t* specialization_op = NULL;
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_create_specialization_group(
          module, function_info, group, i, insertion_anchor, arena,
          &specialization_op));
      insertion_anchor = specialization_op;
      *out_specialization_count += 1;
    }
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      loom_refine_boundaries_specialization_group_t* group = &plan->groups[i];
      for (iree_host_size_t call_index = 0; call_index < group->call_count;
           ++call_index) {
        loom_refine_boundaries_retarget_call(module, group->calls[call_index],
                                             group->symbol_ref);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_refine_boundaries_specialization_plans_allocate(
    const loom_refine_boundaries_graph_t* graph, bool encodings_only,
    iree_arena_allocator_t* arena,
    loom_refine_boundaries_specialization_plans_t** out_plans) {
  *out_plans = NULL;
  loom_refine_boundaries_specialization_plans_t* plans = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*plans), (void**)&plans));
  memset(plans, 0, sizeof(*plans));
  plans->arena = arena;
  plans->encodings_only = encodings_only;
  if (graph->function_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, graph->function_count,
                                                   sizeof(*plans->functions),
                                                   (void**)&plans->functions));
    memset(plans->functions, 0,
           graph->function_count * sizeof(*plans->functions));
  }
  *out_plans = plans;
  return iree_ok_status();
}

iree_status_t loom_refine_boundaries_specialization_plans_apply(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_specialization_plans_t* plans,
    int64_t* out_specialization_count) {
  *out_specialization_count = 0;
  return loom_refine_boundaries_create_specializations(
      module, graph, plans->functions, plans->arena, out_specialization_count);
}
