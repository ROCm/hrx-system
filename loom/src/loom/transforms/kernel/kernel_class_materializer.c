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
#include "loom/ops/special_values.h"
#include "loom/ops/template/ops.h"
#include "loom/rewrite/module_projection.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/symbol/template_rewrite.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/predicate_facts.h"

void loom_kernel_class_product_deinitialize(
    loom_kernel_class_product_t* product) {
  loom_module_free(product->module);
  *product = (loom_kernel_class_product_t){0};
}

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
    if (group->values[i] == value_id) {
      return;
    }
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
    if (domain == LOOM_KERNEL_CLASS_ASSUME_DOMAIN_NONE) {
      continue;
    }

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
  if (group->predicate_count == 0) {
    return iree_ok_status();
  }
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
  const bool provider_is_local =
      loom_kernel_class_provider_is_local(classifier, provider);
  if (action_is_source_generic && !provider_is_local) {
    return iree_ok_status();
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

  if (provider_is_local) {
    const loom_symbol_ref_t target_provider =
        loom_ir_module_projection_target_symbol(projection,
                                                provider->symbol.symbol_id);
    return loom_template_rewrite_apply_as_exact_call(
        rewriter, target_apply_op, target_provider, call_operands, NULL);
  }

  // Keep external providers as ordinary generic requests. The assumed
  // operands preserve the class contract in IR so a subsequent indexed link
  // can select the same provider without embedding source-owned identities.
  for (uint16_t i = 0; i < target_arguments.count; ++i) {
    if (call_operands[i] != target_arguments.values[i]) {
      IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(rewriter, target_apply_op,
                                                     i, call_operands[i]));
    }
  }
  return iree_ok_status();
}

// Caller facts are preconditions of the published artifact. Keeping them on
// the signature preserves argument identity, dependent types, and lexical
// scope; ordinary specialization subsequently materializes any constants.
static iree_status_t loom_kernel_class_materialize_argument_facts(
    loom_func_like_t kernel, const loom_value_facts_t* inferred_facts,
    const loom_value_facts_t* argument_facts, uint16_t specialization_count,
    loom_rewriter_t* rewriter) {
  loom_block_t* block = loom_region_entry_block(loom_func_like_body(kernel));
  uint16_t source_count = 0;
  const loom_predicate_t* source =
      loom_func_like_predicates(kernel, &source_count);
  const uint32_t capacity =
      source_count +
      (uint32_t)block->arg_count * LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY;
  loom_predicate_t* predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&rewriter->module->arena,
                                                 capacity, sizeof(*predicates),
                                                 (void**)&predicates));
  if (source_count) {
    memcpy(predicates, source, source_count * sizeof(*source));
  }
  uint32_t count = source_count;
  for (uint16_t i = specialization_count; i < block->arg_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(rewriter->module, block->arg_ids[i]);
    int64_t domain_lo = 0;
    int64_t domain_hi = 0;
    if (!loom_type_is_scalar(type) ||
        !loom_value_facts_scalar_type_domain(loom_type_element_type(type),
                                             &domain_lo, &domain_hi)) {
      continue;
    }
    count += loom_value_facts_export_integer_refinements(
        block->arg_ids[i], inferred_facts[i], argument_facts[i],
        predicates + count, LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY);
  }
  if (count == source_count) {
    return iree_ok_status();
  }
  if (count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "kernel argument refinements exceed the predicate limit");
  }
  return loom_rewriter_set_attr(
      rewriter, kernel.op, kernel.vtable->predicates_attr_index,
      loom_attr_predicate_list(predicates, (uint16_t)count));
}

// A required extent can be an affine expression of several arguments whose
// individual values vary within the class. Preserve that expression's proven
// value without falsely specializing its inputs.
static iree_status_t loom_kernel_class_materialize_shape_values(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_t* entry,
    const loom_ir_module_projection_t* projection, loom_func_like_t kernel,
    loom_rewriter_t* rewriter) {
  loom_block_t* block = loom_region_entry_block(loom_func_like_body(kernel));
  loom_builder_set_before(&rewriter->builder, block->first_op);
  for (uint32_t i = 0; i < entry->shape_value_count; ++i) {
    const loom_kernel_class_shape_value_t* value = &entry->shape_values[i];
    if (!loom_value_facts_is_exact(value->facts)) {
      continue;
    }
    const loom_value_id_t source =
        classifier->projections[value->projection].source_value_id;
    // Entry argument facts are already retained on the signature.
    if (loom_value_is_block_arg(
            loom_module_value(classifier->module, source))) {
      continue;
    }
    loom_value_id_t target = LOOM_VALUE_ID_INVALID;
    const bool projected =
        loom_ir_module_projection_try_target_value(projection, source, &target);
    IREE_ASSERT(projected);
    loom_value_id_t constant = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_constant_build(&rewriter->builder, value->facts,
                            loom_module_value_type(rewriter->module, target),
                            kernel.op->location, &constant));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_with(rewriter, target, constant));
  }
  return iree_ok_status();
}

static bool loom_kernel_class_trace_precedes(
    const loom_kernel_class_trace_t* traces,
    const loom_kernel_class_trace_id_t* lhs,
    const loom_kernel_class_trace_id_t* rhs) {
  return traces[*lhs].decision_ordinal < traces[*rhs].decision_ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_kernel_class_sort_trace_ids,
                                       loom_kernel_class_trace_id_t,
                                       const loom_kernel_class_trace_t*,
                                       loom_kernel_class_trace_precedes)

iree_status_t loom_kernel_class_materialize(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_collection_t* collection,
    loom_decision_class_ordinal_t class_ordinal,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_kernel_class_product_t* out_product) {
  IREE_ASSERT(class_ordinal < collection->class_count);
  *out_product = (loom_kernel_class_product_t){0};

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
    selected_trace_ids[--trace_cursor] = trace_id;

    trace_id = trace->parent_trace_id;
  }
  IREE_ASSERT(!iree_status_is_ok(status) || trace_cursor == 0);

  if (iree_status_is_ok(status)) {
    // The clone observer consumes source-order projections. Classification
    // order may reserve required decisions before admitting optional ones.
    loom_kernel_class_sort_trace_ids(collection->traces, selected_trace_ids,
                                     collection->accepted_decision_count);
    for (uint32_t i = 0; i < collection->accepted_decision_count; ++i) {
      const loom_kernel_class_trace_t* trace =
          &collection->traces[selected_trace_ids[i]];
      operation_projections[i] = (loom_ir_remap_op_projection_t){
          .source_op =
              classifier->decisions[trace->decision_ordinal].application,
      };
    }
  }

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

  loom_rewriter_t rewriter = {0};
  if (iree_status_is_ok(status)) {
    loom_rewriter_initialize(&rewriter, target_module, &scratch_arena);
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

  const loom_symbol_ref_t target_kernel =
      target_module != NULL
          ? loom_ir_module_projection_target_symbol(
                &module_projection, classifier->kernel_symbol_id)
          : loom_symbol_ref_null();
  if (iree_status_is_ok(status) && classifier->kernel_argument_count != 0) {
    status = loom_kernel_class_materialize_argument_facts(
        loom_func_like_cast(target_module, target_module->symbols
                                               .entries[target_kernel.symbol_id]
                                               .defining_op),
        classifier->argument_facts,
        collection->classes[class_ordinal].argument_facts,
        classifier->specialization_count, &rewriter);
  }
  if (iree_status_is_ok(status) &&
      collection->classes[class_ordinal].shape_value_count != 0) {
    status = loom_kernel_class_materialize_shape_values(
        classifier, &collection->classes[class_ordinal], &module_projection,
        loom_func_like_cast(target_module, target_module->symbols
                                               .entries[target_kernel.symbol_id]
                                               .defining_op),
        &rewriter);
  }
  if (iree_status_is_ok(status) && classifier->specialization_count != 0) {
    status = loom_kernel_specializations_apply(
        collection->specialization_module,
        collection->classes[class_ordinal].specializations,
        classifier->specialization_count,
        loom_func_like_cast(target_module, target_module->symbols
                                               .entries[target_kernel.symbol_id]
                                               .defining_op),
        &rewriter, &scratch_arena);
  }
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&scratch_arena);
  if (iree_status_is_ok(status)) {
    *out_product = (loom_kernel_class_product_t){
        .module = target_module,
        .kernel = target_kernel,
    };
  } else {
    loom_module_free(target_module);
  }
  return status;
}
