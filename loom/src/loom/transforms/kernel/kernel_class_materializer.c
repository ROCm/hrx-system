// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_class_materializer.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/rewrite/module_projection.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/symbol/template_rewrite.h"

// Scalar dialect carrying one published assumption group.
typedef uint8_t loom_kernel_class_assume_domain_t;
enum loom_kernel_class_assume_domain_e {
  LOOM_KERNEL_CLASS_ASSUME_DOMAIN_NONE = 0,
  LOOM_KERNEL_CLASS_ASSUME_DOMAIN_INDEX = 1,
  LOOM_KERNEL_CLASS_ASSUME_DOMAIN_SCALAR = 2,
};

// Scratch representation for one independently valid assume operation.
typedef struct loom_kernel_class_assume_group_t {
  // Unique source values referenced by the group's predicates.
  loom_value_id_t* values;

  // Types parallel to |values|.
  loom_type_t* result_types;

  // Contiguous predicate slice owned by the caller's scratch allocation.
  loom_predicate_t* predicates;

  // Number of initialized values and result types.
  uint16_t value_count;

  // Maximum number of values available in the scratch slices.
  uint16_t value_capacity;

  // Number of initialized predicates.
  uint32_t predicate_count;

  // Dialect required for the group's value domain.
  loom_kernel_class_assume_domain_t domain;

  // Reserved byte. Always zero.
  uint8_t reserved;
} loom_kernel_class_assume_group_t;

static bool loom_kernel_class_provider_is_local(
    const loom_kernel_class_classifier_t* classifier,
    const loom_template_provider_summary_t* provider) {
  return provider->module == classifier->module &&
         loom_symbol_ref_is_valid(provider->symbol) &&
         provider->symbol.module_id == 0;
}

static loom_kernel_class_assume_domain_t loom_kernel_class_value_assume_domain(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  IREE_ASSERT(loom_type_is_scalar(type));
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
                 scalar_type == LOOM_SCALAR_TYPE_OFFSET
             ? LOOM_KERNEL_CLASS_ASSUME_DOMAIN_INDEX
             : LOOM_KERNEL_CLASS_ASSUME_DOMAIN_SCALAR;
}

static void loom_kernel_class_assume_group_add_value(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_kernel_class_assume_group_t* group) {
  for (uint16_t i = 0; i < group->value_count; ++i) {
    if (group->values[i] == value_id) return;
  }
  IREE_ASSERT(group->value_count < group->value_capacity);
  group->values[group->value_count] = value_id;
  group->result_types[group->value_count] =
      loom_module_value_type(module, value_id);
  ++group->value_count;
}

// Translates one model predicate into the cloned application value domain.
// Constant-only predicates need no assume op and return the NONE domain.
static loom_kernel_class_assume_domain_t
loom_kernel_class_materialize_predicate(
    const loom_kernel_class_decision_t* decision,
    const loom_module_t* target_module,
    const loom_value_slice_t target_arguments,
    const loom_decision_program_predicate_t* source_predicate,
    loom_predicate_t* out_predicate) {
  *out_predicate = (loom_predicate_t){
      .kind = source_predicate->kind,
      .arg_count = source_predicate->operand_count,
  };
  loom_kernel_class_assume_domain_t domain =
      LOOM_KERNEL_CLASS_ASSUME_DOMAIN_NONE;
  for (uint8_t i = 0; i < source_predicate->operand_count; ++i) {
    const loom_decision_program_operand_ref_t operand_ref =
        source_predicate->operands[i];
    if (loom_decision_program_operand_is_constant(operand_ref)) {
      const uint32_t constant_ordinal =
          loom_decision_program_operand_ordinal(operand_ref);
      IREE_ASSERT(constant_ordinal < decision->model->program.constant_count);
      out_predicate->arg_tags[i] = LOOM_PRED_ARG_CONST;
      out_predicate->args[i] =
          decision->model->program.constants[constant_ordinal];
      continue;
    }

    IREE_ASSERT(!loom_decision_program_operand_is_result(operand_ref));
    const uint32_t argument_ordinal =
        loom_decision_program_operand_ordinal(operand_ref);
    IREE_ASSERT(argument_ordinal < target_arguments.count);
    const loom_value_id_t value_id = target_arguments.values[argument_ordinal];
    const loom_kernel_class_assume_domain_t value_domain =
        loom_kernel_class_value_assume_domain(target_module, value_id);
    IREE_ASSERT(domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_NONE ||
                domain == value_domain);
    domain = value_domain;
    out_predicate->arg_tags[i] = LOOM_PRED_ARG_VALUE;
    out_predicate->args[i] = value_id;
  }
  return domain;
}

static void loom_kernel_class_append_conjunction(
    const loom_kernel_class_decision_t* decision,
    const loom_module_t* target_module,
    const loom_value_slice_t target_arguments,
    loom_decision_program_conjunction_t conjunction,
    loom_predicate_t* predicate_storage, uint32_t* index_cursor,
    uint32_t* scalar_cursor, loom_kernel_class_assume_group_t groups[2]) {
  const uint32_t predicate_end =
      conjunction.first_predicate + conjunction.predicate_count;
  for (uint32_t i = conjunction.first_predicate; i < predicate_end; ++i) {
    loom_predicate_t predicate;
    const loom_kernel_class_assume_domain_t domain =
        loom_kernel_class_materialize_predicate(
            decision, target_module, target_arguments,
            &decision->model->program.predicates[i], &predicate);
    if (domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_NONE) continue;

    loom_kernel_class_assume_group_t* group =
        domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_INDEX ? &groups[0]
                                                        : &groups[1];
    loom_predicate_t* target_predicate = NULL;
    if (domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_INDEX) {
      target_predicate = &predicate_storage[(*index_cursor)++];
    } else {
      target_predicate = &predicate_storage[--(*scalar_cursor)];
    }
    *target_predicate = predicate;
    ++group->predicate_count;
    for (uint8_t j = 0; j < predicate.arg_count; ++j) {
      if (predicate.arg_tags[j] == LOOM_PRED_ARG_VALUE) {
        loom_kernel_class_assume_group_add_value(
            target_module, (loom_value_id_t)predicate.args[j], group);
      }
    }
  }
}

static iree_status_t loom_kernel_class_build_assume(
    loom_rewriter_t* rewriter, loom_op_t* apply_op,
    const loom_kernel_class_assume_group_t* group,
    loom_value_id_t* call_operands) {
  if (group->predicate_count == 0) return iree_ok_status();
  if (group->predicate_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kernel class assumption exceeds the predicate "
                            "attribute limit");
  }

  loom_builder_set_before(&rewriter->builder, apply_op);
  loom_op_t* assume_op = NULL;
  if (group->domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_INDEX) {
    IREE_RETURN_IF_ERROR(loom_index_assume_build(
        &rewriter->builder, group->values, group->value_count,
        group->predicates, group->predicate_count, group->result_types,
        group->value_count, apply_op->location, &assume_op));
  } else {
    IREE_ASSERT(group->domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_SCALAR);
    IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
        &rewriter->builder, group->values, group->value_count,
        group->predicates, group->predicate_count, group->result_types,
        group->value_count, apply_op->location, &assume_op));
  }

  const loom_value_slice_t assumed_values =
      group->domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_INDEX
          ? loom_index_assume_results(assume_op)
          : loom_scalar_assume_results(assume_op);
  IREE_ASSERT(assumed_values.count == group->value_count);
  for (uint16_t i = 0; i < group->value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
        rewriter, group->values[i], assumed_values.values[i],
        IREE_SV("specialized")));
    for (uint16_t j = 0; j < apply_op->operand_count; ++j) {
      if (call_operands[j] == group->values[i]) {
        call_operands[j] = assumed_values.values[i];
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_class_rewrite_application(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_decision_t* decision, uint32_t action_ordinal,
    const loom_ir_module_projection_t* projection, loom_op_t* target_apply_op,
    loom_rewriter_t* rewriter) {
  IREE_ASSERT(loom_template_apply_isa(target_apply_op));
  IREE_ASSERT(action_ordinal < decision->model->providers.count);
  const loom_template_provider_summary_t* provider =
      loom_template_decision_model_provider(decision->model, action_ordinal);
  const bool action_is_source_generic =
      decision->generic_result.kind == LOOM_DECISION_PROGRAM_RESULT_SELECTED &&
      decision->generic_result.action_ordinal == action_ordinal;
  if (!loom_kernel_class_provider_is_local(classifier, provider)) {
    if (action_is_source_generic) return iree_ok_status();
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel class selects a provider outside its source product");
  }

  const loom_value_slice_t target_arguments =
      loom_template_apply_operands(target_apply_op);
  IREE_ASSERT(target_arguments.count == decision->argument_count);
  loom_value_id_t* call_operands = NULL;
  if (target_arguments.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, target_arguments.count, sizeof(*call_operands),
        (void**)&call_operands));
    memcpy(call_operands, target_arguments.values,
           target_arguments.count * sizeof(*call_operands));
  }

  if (!action_is_source_generic) {
    const loom_kernel_class_action_t* action =
        &decision->actions[action_ordinal];
    IREE_ASSERT(action->choice_ordinal < decision->model->program.choice_count);
    const loom_decision_program_conjunction_t action_conjunction =
        decision->model->program.choices[action->choice_ordinal].conjunction;
    const uint32_t maximum_predicate_count =
        decision->model->program.hard_requirements.predicate_count +
        action_conjunction.predicate_count;
    loom_predicate_t* predicate_storage = NULL;
    loom_value_id_t* group_values = NULL;
    loom_type_t* group_types = NULL;
    if (maximum_predicate_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          rewriter->arena, maximum_predicate_count, sizeof(*predicate_storage),
          (void**)&predicate_storage));
    }
    if (target_arguments.count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          rewriter->arena, target_arguments.count * 2, sizeof(*group_values),
          (void**)&group_values));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          rewriter->arena, target_arguments.count * 2, sizeof(*group_types),
          (void**)&group_types));
    }

    loom_kernel_class_assume_group_t groups[2] = {
        {
            .values = group_values,
            .result_types = group_types,
            .predicates = predicate_storage,
            .value_capacity = target_arguments.count,
            .domain = LOOM_KERNEL_CLASS_ASSUME_DOMAIN_INDEX,
        },
        {
            .values = target_arguments.count > 0
                          ? group_values + target_arguments.count
                          : NULL,
            .result_types = target_arguments.count > 0
                                ? group_types + target_arguments.count
                                : NULL,
            .value_capacity = target_arguments.count,
            .domain = LOOM_KERNEL_CLASS_ASSUME_DOMAIN_SCALAR,
        },
    };
    uint32_t index_cursor = 0;
    uint32_t scalar_cursor = maximum_predicate_count;
    loom_kernel_class_append_conjunction(
        decision, rewriter->module, target_arguments,
        decision->model->program.hard_requirements, predicate_storage,
        &index_cursor, &scalar_cursor, groups);
    loom_kernel_class_append_conjunction(
        decision, rewriter->module, target_arguments, action_conjunction,
        predicate_storage, &index_cursor, &scalar_cursor, groups);
    groups[0].predicates = predicate_storage;
    groups[1].predicates =
        maximum_predicate_count > 0 ? predicate_storage + scalar_cursor : NULL;
    IREE_ASSERT(index_cursor + groups[1].predicate_count <=
                maximum_predicate_count);

    IREE_RETURN_IF_ERROR(loom_kernel_class_build_assume(
        rewriter, target_apply_op, &groups[0], call_operands));
    IREE_RETURN_IF_ERROR(loom_kernel_class_build_assume(
        rewriter, target_apply_op, &groups[1], call_operands));
  }

  const loom_symbol_ref_t target_provider =
      loom_ir_module_projection_target_symbol(projection,
                                              provider->symbol.symbol_id);
  return loom_template_rewrite_apply_as_exact_call(
      rewriter, target_apply_op, target_provider, call_operands);
}

iree_status_t loom_kernel_class_materialize(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_collection_t* collection,
    loom_decision_class_ordinal_t class_ordinal,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module) {
  IREE_ASSERT(class_ordinal < collection->class_count);
  *out_module = NULL;

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

  loom_ir_remap_op_projection_t* operation_projections = NULL;
  loom_kernel_class_trace_id_t* selected_trace_ids = NULL;
  iree_status_t status = iree_ok_status();
  if (collection->accepted_decision_count > 0) {
    status = iree_arena_allocate_array(
        &scratch_arena, collection->accepted_decision_count,
        sizeof(*operation_projections), (void**)&operation_projections);
  }
  if (iree_status_is_ok(status) && collection->accepted_decision_count > 0) {
    status = iree_arena_allocate_array(
        &scratch_arena, collection->accepted_decision_count,
        sizeof(*selected_trace_ids), (void**)&selected_trace_ids);
  }

  loom_kernel_class_trace_id_t trace_id =
      collection->classes[class_ordinal].trace_id;
  uint32_t trace_cursor = collection->accepted_decision_count;
  while (iree_status_is_ok(status) &&
         trace_id != LOOM_KERNEL_CLASS_TRACE_ID_INVALID) {
    IREE_ASSERT(trace_cursor > 0);
    IREE_ASSERT(trace_id < collection->trace_count);
    const loom_kernel_class_trace_t* trace = &collection->traces[trace_id];
    IREE_ASSERT(trace->decision_ordinal < classifier->decision_count);
    const loom_kernel_class_decision_t* decision =
        &classifier->decisions[trace->decision_ordinal];
    IREE_ASSERT(trace->action_ordinal < decision->model->providers.count);
    IREE_ASSERT(collection->decision_results[trace->decision_ordinal].state ==
                LOOM_KERNEL_CLASS_DECISION_ACCEPTED);
    const uint32_t projection_ordinal = --trace_cursor;
    operation_projections[projection_ordinal].source_op =
        decision->demand->apply_op;
    operation_projections[projection_ordinal].target_op = NULL;
    selected_trace_ids[projection_ordinal] = trace_id;

    const loom_template_provider_summary_t* provider =
        loom_template_decision_model_provider(decision->model,
                                              trace->action_ordinal);
    const bool action_is_source_generic =
        decision->generic_result.kind ==
            LOOM_DECISION_PROGRAM_RESULT_SELECTED &&
        decision->generic_result.action_ordinal == trace->action_ordinal;
    if (!action_is_source_generic &&
        !loom_kernel_class_provider_is_local(classifier, provider)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel class selects a provider outside its source product");
    }
    trace_id = trace->parent_trace_id;
  }
  IREE_ASSERT(!iree_status_is_ok(status) || trace_cursor == 0);

  loom_ir_module_projection_t module_projection = {0};
  loom_module_t* target_module = NULL;
  if (iree_status_is_ok(status)) {
    const loom_ir_module_clone_options_t clone_options = {
        .operations =
            {
                .entries = operation_projections,
                .count = collection->accepted_decision_count,
            },
    };
    status = loom_ir_module_clone(classifier->module, &clone_options,
                                  block_pool, &scratch_arena, allocator,
                                  &module_projection, &target_module);
  }

  loom_rewriter_t rewriter;
  bool rewriter_is_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_initialize(&rewriter, target_module, &scratch_arena);
    rewriter_is_initialized = iree_status_is_ok(status);
  }
  for (uint32_t i = 0;
       i < collection->accepted_decision_count && iree_status_is_ok(status);
       ++i) {
    const loom_kernel_class_trace_t* trace =
        &collection->traces[selected_trace_ids[i]];
    status = loom_kernel_class_rewrite_application(
        classifier, &classifier->decisions[trace->decision_ordinal],
        trace->action_ordinal, &module_projection,
        operation_projections[i].target_op, &rewriter);
  }

  if (rewriter_is_initialized) loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&scratch_arena);
  if (iree_status_is_ok(status)) {
    *out_module = target_module;
  } else {
    loom_module_free(target_module);
  }
  return status;
}
