// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/contract_vector.h"
#include "loom/analysis/kernel_async_legality.h"
#include "loom/analysis/vector_memory_footprint.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/lower_internal.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/registers.h"

enum {
  // Default allocation block size for source-to-low report rows.
  LOOM_LOW_LOWER_REPORT_ROW_VEC_DEFAULT_BYTE_LENGTH = 4096,
};

typedef struct loom_low_lower_descriptor_matrix_plan_t {
  // Shared source adapter used by this matrix descriptor plan.
  loom_target_contract_descriptor_matrix_source_t source;
  // Descriptor row selected by the target matrix projection.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Target-independent request facts used to materialize descriptor operands.
  loom_contract_request_t contract_request;
  // Target-owned immediate attributes materialized from request facts.
  loom_named_attr_slice_t attrs;
} loom_low_lower_descriptor_matrix_plan_t;

static bool loom_low_lower_type_is_none(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_NONE;
}

static bool loom_low_lower_abi_argument_kind_is_known(
    loom_low_lower_abi_argument_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT:
    case LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_resource_import_kind_is_known(
    loom_low_resource_import_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_function_attr_present(loom_func_like_t function,
                                                 uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(function.op)[attr_index]);
}

static loom_target_abi_kind_t loom_low_lower_function_abi(
    const loom_low_lower_context_t* context) {
  const uint8_t abi_attr_index =
      context->source_function.vtable->abi_attr_index;
  if (loom_low_lower_function_attr_present(context->source_function,
                                           abi_attr_index)) {
    return (loom_target_abi_kind_t)loom_func_like_abi(context->source_function);
  }
  return context->options->bundle->export_plan->abi_kind;
}

static iree_status_t loom_low_lower_map_direct_argument(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_low_lower_map_value(context, source_op, source_argument_id,
                                  &out_argument->abi_type);
}

static iree_status_t loom_low_lower_map_argument(
    loom_low_lower_context_t* context, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  uint32_t previous_error_count = context->result->error_count;
  if (context->policy->map_argument.fn == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_map_direct_argument(context, context->source_function.op,
                                           source_argument_id, out_argument));
  } else {
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = loom_type_none(),
        .resource_source_type = loom_type_none(),
    };
    IREE_RETURN_IF_ERROR(context->policy->map_argument.fn(
        context->policy->map_argument.user_data, context,
        context->source_function.op, source_argument_index, source_argument_id,
        out_argument));
  }

  IREE_ASSERT(loom_low_lower_abi_argument_kind_is_known(out_argument->kind));
  if (loom_low_lower_type_is_none(out_argument->abi_type)) {
    if (context->result->error_count == previous_error_count) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("argument")),
          loom_param_u64(source_argument_id),
      };
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_target_context_error(
          context, context->source_function.op, LOOM_ERR_TARGET_027, params,
          IREE_ARRAYSIZE(params)));
    }
    return iree_ok_status();
  }
  IREE_ASSERT(loom_low_type_is_register(out_argument->abi_type));

  if (out_argument->kind == LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
    return iree_ok_status();
  }
  IREE_ASSERT(loom_low_lower_resource_import_kind_is_known(
      out_argument->resource_import_kind));
  IREE_ASSERT_GE(out_argument->resource_index, 0);
  if (loom_low_lower_type_is_none(out_argument->resource_source_type)) {
    out_argument->resource_source_type =
        loom_module_value_type(context->module, source_argument_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_initialize_argument_map(
    loom_low_lower_context_t* context) {
  if (context->lowering.argument_map != NULL) {
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  context->lowering.argument_map_count = argument_count;
  if (argument_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, argument_count, sizeof(*context->lowering.argument_map),
      (void**)&context->lowering.argument_map));
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_map_argument(
        context, i, source_arguments[i], &context->lowering.argument_map[i]));
  }
  return iree_ok_status();
}

static uint16_t loom_low_lower_direct_argument_count(
    const loom_low_lower_context_t* context) {
  uint16_t direct_argument_count = 0;
  for (uint16_t i = 0; i < context->lowering.argument_map_count; ++i) {
    if (context->lowering.argument_map[i].kind ==
        LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
      ++direct_argument_count;
    }
  }
  return direct_argument_count;
}

static void loom_low_lower_assert_options(
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options) {
  IREE_ASSERT(module != NULL);
  IREE_ASSERT(loom_func_like_isa(source_function));
  IREE_ASSERT(options != NULL);
  IREE_ASSERT(source_function.op->kind == LOOM_OP_FUNC_DEF ||
              source_function.op->kind == LOOM_OP_KERNEL_DEF);
  IREE_ASSERT(loom_symbol_ref_is_valid(options->target_ref));
  IREE_ASSERT_EQ(options->target_ref.module_id, 0);
  IREE_ASSERT_LT(options->target_ref.symbol_id, module->symbols.count);
  IREE_ASSERT(options->bundle != NULL);
  IREE_ASSERT(options->bundle->snapshot != NULL);
  IREE_ASSERT(options->bundle->export_plan != NULL);
  IREE_ASSERT(options->bundle->config != NULL);
  IREE_ASSERT(options->fact_table != NULL);
  IREE_ASSERT(options->fact_table->context.target_bundle == options->bundle);
  IREE_ASSERT(options->descriptor_registry != NULL);
  IREE_ASSERT(options->policy != NULL);
}

static iree_status_t loom_low_lower_intern_type_id(
    loom_low_lower_context_t* context, loom_type_t type,
    loom_type_id_t* out_type_id) {
  return loom_module_intern_type_id(context->module, type, out_type_id);
}

static iree_status_t loom_low_lower_check_mapped_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_low_type) {
  uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_value_id, out_low_type));
  if (loom_low_lower_type_is_none(*out_low_type)) {
    if (context->result->error_count == previous_error_count) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("source")),
          loom_param_u64(source_value_id),
      };
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_target_context_error(
          context, source_op, LOOM_ERR_TARGET_027, params,
          IREE_ARRAYSIZE(params)));
    }
  }
  return iree_ok_status();
}

static bool loom_low_lower_first_return_operands(
    loom_region_t* source_body, const loom_op_t** out_return_op,
    loom_value_slice_t* out_operands) {
  *out_return_op = NULL;
  *out_operands = (loom_value_slice_t){0};
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_body, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_func_return_isa(op)) {
        continue;
      }
      *out_return_op = op;
      *out_operands = loom_func_return_operands(op);
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_check_function_result(
    loom_low_lower_context_t* context, const loom_op_t* return_op,
    loom_value_slice_t returned_values, uint16_t result_index,
    loom_value_id_t result_id) {
  if (result_index < returned_values.count) {
    loom_type_t low_type = loom_type_none();
    return loom_low_lower_check_mapped_value(
        context, return_op, returned_values.values[result_index], &low_type);
  }

  loom_type_t low_type = loom_type_none();
  return loom_low_lower_check_mapped_value(context, context->source_function.op,
                                           result_id, &low_type);
}

static iree_status_t loom_low_lower_check_function_signature(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));

  const loom_op_t* return_op = NULL;
  loom_value_slice_t returned_values = {0};
  (void)loom_low_lower_first_return_operands(source_body, &return_op,
                                             &returned_values);

  const loom_value_id_t* result_ids =
      loom_op_const_results(context->source_function.op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_check_function_result(
        context, return_op, returned_values, i, result_ids[i]));
  }

  uint16_t predicate_count = 0;
  (void)loom_func_like_predicates(context->source_function, &predicate_count);
  if (predicate_count != 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(predicate_count),
    };
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_target_context_error(
        context, context->source_function.op, LOOM_ERR_TARGET_028, params,
        IREE_ARRAYSIZE(params)));
  }
  if (context->source_function.op->tied_result_count != 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(context->source_function.op->tied_result_count),
    };
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_target_context_error(
        context, context->source_function.op, LOOM_ERR_TARGET_029, params,
        IREE_ARRAYSIZE(params)));
  }
  return iree_ok_status();
}

static bool loom_low_lower_structured_low_enabled(
    const loom_low_lower_context_t* context) {
  return context->options->control_flow_lowering ==
         LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
}

static bool loom_low_lower_supported_structured_source_op(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_low_lower_structured_low_enabled(context)) {
    return false;
  }
  switch (source_op->kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_is_structural(const loom_module_t* module,
                                            const loom_op_t* op) {
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_are_fact_identity(traits) ||
      loom_traits_are_value_alias(traits)) {
    return true;
  }
  switch (op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_FUNC_CALL:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_KERNEL_RETURN:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_YIELD:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_is_source_metadata(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_ENCODING_ASSUME_SPEC:
    case LOOM_OP_ENCODING_DEFINE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED:
    case LOOM_OP_ENCODING_LAYOUT_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_STRIDED:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_uses_policy(const loom_module_t* module,
                                          const loom_op_t* op) {
  return !loom_low_lower_op_is_structural(module, op) &&
         !loom_low_lower_op_is_source_metadata(op->kind);
}

static bool loom_low_lower_op_is_discardable_hint(const loom_module_t* module,
                                                  const loom_op_t* op) {
  if (op->result_count != 0 || op->region_count != 0 ||
      op->tied_result_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  return iree_any_bit_set(traits, LOOM_TRAIT_HINT);
}

static void loom_low_lower_mark_value_storage_required(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  context->lowering.value_storage_flags[source_ordinal] |=
      LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED;
}

void loom_low_lower_require_source_value_storage(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  loom_low_lower_mark_value_storage_required(context, source_value_id);
}

void loom_low_lower_require_source_operands_storage(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    loom_low_lower_mark_value_storage_required(context, operands[i]);
  }
}

static bool loom_low_lower_value_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  return iree_any_bit_set(context->lowering.value_storage_flags[source_ordinal],
                          LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED);
}

static void loom_low_lower_mark_value_slice_storage_required(
    loom_low_lower_context_t* context, loom_value_slice_t values) {
  for (uint16_t i = 0; i < values.count; ++i) {
    loom_low_lower_mark_value_storage_required(context, values.values[i]);
  }
}

static bool loom_low_lower_cfg_cond_br_exact_bool(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition) {
  if (out_condition != NULL) *out_condition = false;
  if (!loom_cfg_cond_br_isa(source_op) ||
      context->lowering.fact_table == NULL) {
    return false;
  }
  const loom_value_facts_t facts = loom_value_fact_table_lookup(
      context->lowering.fact_table, loom_cfg_cond_br_condition(source_op));
  bool condition = false;
  if (!loom_value_facts_as_exact_bool(facts, &condition)) {
    return false;
  }
  if (out_condition != NULL) *out_condition = condition;
  return true;
}

static bool loom_low_lower_rule_value_ref_source_value(
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t* out_source_value_id) {
  *out_source_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      IREE_ASSERT_LT(value_ref->index, source_op->operand_count);
      *out_source_value_id =
          loom_op_const_operands(source_op)[value_ref->index];
      return true;
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT_LT(value_ref->index, source_op->result_count);
      *out_source_value_id = loom_op_const_results(source_op)[value_ref->index];
      return true;
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      return false;
    default:
      IREE_ASSERT_UNREACHABLE("unknown source-low value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_low_lower_rule_value_ref_uses_source_memory_plan(
    const loom_low_lower_rule_set_t* rule_set, uint16_t value_ref_index) {
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_emit_uses_source_memory_plan(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  for (uint16_t operand_ordinal = 0; operand_ordinal < emit->operand_ref_count;
       ++operand_ordinal) {
    const uint16_t value_ref_index =
        (uint16_t)(emit->operand_ref_start + operand_ordinal);
    if (loom_low_lower_rule_value_ref_uses_source_memory_plan(
            rule_set, value_ref_index)) {
      return true;
    }
  }
  return false;
}

static void loom_low_lower_mark_source_memory_access_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* access) {
  for (uint8_t term_ordinal = 0; term_ordinal < access->dynamic_term_count;
       ++term_ordinal) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->dynamic_terms[term_ordinal];
    loom_low_lower_mark_value_storage_required(context, term->index);
    for (uint8_t stride_ordinal = 0; stride_ordinal < term->stride_value_count;
         ++stride_ordinal) {
      loom_low_lower_mark_value_storage_required(
          context, term->stride_values[stride_ordinal]);
    }
  }
}

static void loom_low_lower_mark_rule_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  const loom_low_lower_rule_t* rule = selected_plan->rule;
  IREE_ASSERT(rule_set != NULL);
  IREE_ASSERT(rule != NULL);
  loom_low_source_memory_access_plan_t source_memory_access = {0};
  bool source_memory_access_built = false;
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + emit_ordinal);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    for (uint16_t operand_ordinal = 0;
         operand_ordinal < emit->operand_ref_count; ++operand_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(emit->operand_ref_start + operand_ordinal);
      loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
      if (loom_low_lower_rule_value_ref_source_value(
              rule_set, selected_plan->source_op, value_ref_index,
              &source_value_id)) {
        loom_low_lower_mark_value_storage_required(context, source_value_id);
      }
    }
    if (emit->source_memory_ordinal == 0 ||
        !loom_low_lower_emit_uses_source_memory_plan(rule_set, emit)) {
      continue;
    }
    if (!source_memory_access_built) {
      loom_low_source_memory_access_diagnostic_t diagnostic = {0};
      if (!loom_low_source_memory_access_plan_build(
              context->module, context->lowering.fact_table,
              selected_plan->source_op, &source_memory_access, &diagnostic)) {
        IREE_ASSERT_UNREACHABLE("selected source memory must still plan");
        IREE_BUILTIN_UNREACHABLE();
      }
      source_memory_access_built = true;
    }
    loom_low_lower_mark_source_memory_access_storage_demands(
        context, &source_memory_access);
  }
  for (uint16_t alias_ordinal = 0; alias_ordinal < rule->alias_ref_count;
       ++alias_ordinal) {
    const uint16_t value_ref_index =
        (uint16_t)(rule->alias_ref_start + alias_ordinal * 2);
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    if (loom_low_lower_rule_value_ref_source_value(
            rule_set, selected_plan->source_op, value_ref_index,
            &source_value_id)) {
      loom_low_lower_mark_value_storage_required(context, source_value_id);
    }
  }
  if (rule->alias_ref_count == 0) return;

  // Alias rules can erase projection ops whose operands are still referenced
  // by facts consumed by later plans, such as dynamic byte offsets used during
  // source-memory address emission. Exact operands are already captured by the
  // facts themselves and do not need low SSA storage.
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_value_fact_table_t* fact_table = context->lowering.fact_table;
  if (loom_buffer_view_isa(source_op)) {
    const loom_value_id_t byte_offset = loom_buffer_view_byte_offset(source_op);
    int64_t exact_offset = 0;
    if (fact_table == NULL ||
        !loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(fact_table, byte_offset),
            &exact_offset)) {
      loom_low_lower_mark_value_storage_required(context, byte_offset);
    }
    return;
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(context->module, source_op);
  if (vtable == NULL || !iree_any_bit_set(vtable->vtable_flags,
                                          LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
    return;
  }
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = vtable->fixed_operand_count; i < source_op->operand_count;
       ++i) {
    int64_t exact_value = 0;
    if (fact_table != NULL &&
        loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(fact_table, operands[i]),
            &exact_value)) {
      continue;
    }
    loom_low_lower_mark_value_storage_required(context, operands[i]);
  }
}

static void loom_low_lower_mark_callback_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  if (selected_plan->kind == LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK &&
      context->policy->mark_plan_storage_demands.fn != NULL) {
    context->policy->mark_plan_storage_demands.fn(
        context->policy->mark_plan_storage_demands.user_data, context,
        selected_plan->source_op, selected_plan->plan);
    return;
  }
  loom_low_lower_require_source_operands_storage(context,
                                                 selected_plan->source_op);
}

static void loom_low_lower_mark_structural_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits)) {
    const loom_value_id_t* operands = loom_op_const_operands(source_op);
    for (uint16_t i = 0; i < source_op->operand_count; ++i) {
      loom_low_lower_mark_value_storage_required(context, operands[i]);
    }
    return;
  }
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(source_op->operand_count >= 1);
    loom_low_lower_mark_value_storage_required(
        context, loom_op_const_operands(source_op)[0]);
    return;
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
      loom_low_lower_mark_value_storage_required(
          context, loom_buffer_assume_same_root_buffer(source_op));
      return;
    case LOOM_OP_FUNC_CALL:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_func_call_operands(source_op));
      return;
    case LOOM_OP_FUNC_RETURN:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_func_return_operands(source_op));
      return;
    case LOOM_OP_CFG_BR:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_cfg_br_args(source_op));
      return;
    case LOOM_OP_CFG_COND_BR: {
      if (loom_low_lower_cfg_cond_br_exact_bool(context, source_op, NULL)) {
        return;
      }
      loom_low_lower_mark_value_storage_required(
          context, loom_cfg_cond_br_condition(source_op));
      return;
    }
    case LOOM_OP_SCF_FOR:
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_for_lower_bound(source_op));
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_for_upper_bound(source_op));
      loom_low_lower_mark_value_storage_required(context,
                                                 loom_scf_for_step(source_op));
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_scf_for_iter_args(source_op));
      if (loom_scf_for_unroll_factor_is_present(source_op)) {
        loom_low_lower_mark_value_storage_required(
            context, loom_scf_for_unroll_factor(source_op));
      }
      return;
    case LOOM_OP_SCF_IF:
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_if_condition(source_op));
      return;
    case LOOM_OP_SCF_YIELD:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_scf_yield_values(source_op));
      return;
    case LOOM_OP_KERNEL_RETURN:
    default:
      return;
  }
}

static bool loom_low_lower_source_op_requires_emission(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (source_op->result_count == 0 || source_op->region_count != 0 ||
      source_op->tied_result_count != 0) {
    return true;
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT |
                                   LOOM_TRAIT_UNIQUE_IDENTITY |
                                   LOOM_TRAIT_CONVERGENT)) {
    return true;
  }
  if (loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
      loom_op_regions_have_write_effects(source_op) ||
      loom_op_regions_have_convergent_effects(source_op) ||
      loom_op_regions_have_hints(context->module, source_op)) {
    return true;
  }
  return false;
}

static bool loom_low_lower_source_op_result_storage_required(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_ASSERT_NE(results[i], LOOM_VALUE_ID_INVALID);
    if (loom_low_lower_value_storage_required(context, results[i]) ||
        loom_module_value_has_type_uses(context->module, results[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_selected_plan_storage_required(
    const loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  return loom_low_lower_source_op_requires_emission(context, source_op) ||
         loom_low_lower_source_op_result_storage_required(context, source_op);
}

static bool loom_low_lower_can_elide_source_storage(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return !loom_low_lower_source_op_requires_emission(context, source_op) &&
         !loom_low_lower_source_op_result_storage_required(context, source_op);
}

static void loom_low_lower_mark_selected_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  switch (selected_plan->kind) {
    case LOOM_LOW_LOWER_SELECTED_PLAN_RULE:
      loom_low_lower_mark_rule_storage_demands(context, selected_plan);
      break;
    case LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX:
    case LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK:
      loom_low_lower_mark_callback_plan_storage_demands(context, selected_plan);
      break;
  }
}

static void loom_low_lower_mark_region_structural_storage_demands(
    loom_low_lower_context_t* context, loom_region_t* source_region) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_lower_op_is_structural(context->module, op)) {
        loom_low_lower_mark_structural_storage_demands(context, op);
      }
      if (loom_low_lower_supported_structured_source_op(context, op)) {
        loom_region_t* const* regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (regions[i] != NULL) {
            loom_low_lower_mark_region_structural_storage_demands(context,
                                                                  regions[i]);
          }
        }
      }
    }
  }
}

static void loom_low_lower_analyze_storage_demands(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_low_lower_mark_region_structural_storage_demands(context, source_body);
  for (iree_host_size_t i = context->lowering.selected_plan_count; i > 0; --i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.selected_plans[i - 1];
    if (!iree_any_bit_set(selected_plan->flags,
                          LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED) &&
        loom_low_lower_selected_plan_storage_required(context, selected_plan)) {
      loom_low_lower_mark_selected_plan_storage_demands(context, selected_plan);
    }
  }
  for (iree_host_size_t i = 0; i < context->lowering.selected_plan_count; ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.selected_plans[i];
    if (loom_low_lower_can_elide_source_storage(context,
                                                selected_plan->source_op)) {
      selected_plan->flags |= LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
    }
  }
}

static iree_status_t loom_low_lowering_frame_initialize_value_ordinals(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  // Target-legalization query scopes can inspect source ops before CFG
  // conversion, so nested source-region values must be ordinal-addressable even
  // when the final source-to-low boundary expects CFG.
  return loom_local_value_domain_acquire_for_region_tree(
      context->module, source_body, &context->arena,
      &context->lowering.value_domain);
}

static void loom_low_lowering_frame_deinitialize(
    loom_low_lower_context_t* context) {
  loom_local_value_domain_release(&context->lowering.value_domain);
}

static void loom_low_lower_count_region_plan_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    iree_host_size_t* inout_plan_capacity) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_lower_supported_structured_source_op(context, op)) {
        loom_region_t* const* regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (regions[i] != NULL) {
            loom_low_lower_count_region_plan_ops(context, regions[i],
                                                 inout_plan_capacity);
          }
        }
        continue;
      }
      if (loom_low_lower_op_uses_policy(context->module, op)) {
        ++(*inout_plan_capacity);
      }
    }
  }
}

static iree_status_t loom_low_lower_prepare_plan(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  iree_host_size_t plan_capacity = 0;
  loom_low_lower_count_region_plan_ops(context, source_body, &plan_capacity);
  context->lowering.selected_plan_capacity = plan_capacity;
  context->lowering.selected_plan_count = 0;
  context->lowering.selected_plan_emit_index = 0;
  if (plan_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, plan_capacity, sizeof(*context->lowering.selected_plans),
      (void**)&context->lowering.selected_plans));
  if (context->options->table_arena != NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->options->table_arena, plan_capacity,
        sizeof(*context->lowering.memory_access_records),
        (void**)&context->lowering.memory_access_records));
    context->lowering.memory_access_record_capacity = plan_capacity;
  }
  return iree_ok_status();
}

static void loom_low_lower_record_selected_plan(
    loom_low_lower_context_t* context,
    loom_low_lower_selected_plan_t selected_plan) {
  IREE_ASSERT_LT(context->lowering.selected_plan_count,
                 context->lowering.selected_plan_capacity);
  context->lowering.selected_plans[context->lowering.selected_plan_count++] =
      selected_plan;
}

static void loom_low_lower_record_elided_hint_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
                   .flags = LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = UINT16_MAX,
                   .rule_set = NULL,
                   .rule = NULL,
                   .resolved_emits = NULL,
                   .plan = loom_low_lower_plan_empty(),
               });
}

static iree_status_t loom_low_lower_record_selected_rule_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t rule_set_index, const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_selection_t rule_selection) {
  const loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_resolve_emit_program(
      context, rule_set, rule_selection.rule, &resolved_emits));
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_RULE,
                   .rule_set_index = rule_set_index,
                   .rule_index = rule_selection.rule_index,
                   .rule_set = rule_set,
                   .rule = rule_selection.rule,
                   .resolved_emits = resolved_emits,
                   .plan = loom_low_lower_plan_empty(),
               });
  return iree_ok_status();
}

static bool loom_low_lower_rule_selection_is_better_failure(
    const loom_low_lower_rule_set_t* failed_rule_set,
    loom_low_lower_rule_selection_t failed_rule_selection,
    loom_low_lower_rule_selection_t rule_selection) {
  return rule_selection.has_source_op_span &&
         (failed_rule_set == NULL ||
          rule_selection.matched_guard_count >
              failed_rule_selection.matched_guard_count);
}

static iree_status_t loom_low_lower_query_environment_from_context(
    loom_low_lower_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_target_contract_query_environment_t* out_environment) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_environment = (loom_target_contract_query_environment_t){
      .module = context->module,
      .function = context->source_function,
      .bundle = context->options->bundle,
      .target_data = context->options->target_data,
      .target_ref = context->options->target_ref,
      .descriptor_set = descriptor_set,
      .fact_table = context->lowering.fact_table,
      .view_regions = view_regions,
      .arena = &context->arena,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_contract_query_rejection(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_query_result_t* result) {
  if (result->rejection != NULL) {
    return loom_low_lower_emit_error_ref(
        context, source_op, result->rejection->error_ref,
        result->rejection->params, result->rejection->param_count);
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_status_t loom_low_lower_descriptor_matrix_request_from_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule,
    loom_contract_request_t* out_request) {
  *out_request = (loom_contract_request_t){0};
  if (context->policy->descriptor_matrix.options == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "descriptor-matrix policy has no source adapter");
  }

  switch (matrix_rule->source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA: {
      if (!loom_vector_mma_isa(source_op)) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "descriptor-matrix selected non-vector.mma op");
      }
      loom_target_contract_query_environment_t environment = {0};
      IREE_RETURN_IF_ERROR(loom_low_lower_query_environment_from_context(
          context, context->descriptor_set, &environment));
      loom_contract_vector_mma_options_t options = {0};
      IREE_RETURN_IF_ERROR(context->policy->descriptor_matrix.options(
          context->policy->descriptor_matrix.user_data, &environment,
          matrix_rule, &options));
      loom_contract_diagnostic_t diagnostic = {0};
      if (!loom_contract_request_from_vector_mma_op(
              context->module, context->lowering.fact_table, source_op,
              &options, out_request, &diagnostic)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "descriptor-matrix selected vector.mma request is inconsistent");
      }
      return iree_ok_status();
    }
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown descriptor-matrix source");
  }
}

static iree_status_t loom_low_lower_record_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule,
    const loom_target_contract_query_result_t* query_result) {
  loom_low_lower_descriptor_matrix_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  plan_data->source = matrix_rule->source;
  if (query_result->selected_descriptor == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "descriptor-matrix legal query has no descriptor");
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_row(
      context, query_result->selected_descriptor, &plan_data->descriptor));
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_request_from_source(
      context, source_op, matrix_rule, &plan_data->contract_request));
  plan_data->attrs = loom_named_attr_slice_empty();
  if (plan_data->descriptor.descriptor->immediate_count != 0) {
    if (context->policy->descriptor_matrix.attrs == NULL) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "descriptor-matrix policy has no attrs callback");
    }
    IREE_RETURN_IF_ERROR(context->policy->descriptor_matrix.attrs(
        context->policy->descriptor_matrix.user_data, context, matrix_rule,
        &plan_data->contract_request, plan_data->descriptor.descriptor,
        &plan_data->attrs));
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = query_result->rule_index,
                   .rule_set = NULL,
                   .rule = NULL,
                   .resolved_emits = NULL,
                   .plan = loom_low_lower_plan_make(source_op->kind, plan_data),
               });
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op_from_contract_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_rule_set_t** inout_failed_rule_set,
    loom_low_lower_rule_selection_t* inout_failed_rule_selection,
    bool* out_selected) {
  *out_selected = false;
  const loom_target_contract_index_t* index = &context->contract_index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const uint16_t case_index = (uint16_t)(op_entry.case_start + i);
    const loom_target_contract_case_t* contract_case =
        &index->cases[case_index];
    const loom_target_contract_binding_t* binding =
        &index->bindings[contract_case->binding_index];
    if (contract_case->system ==
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX) {
      const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
          &binding->fragment->descriptor_matrices[contract_case->row_index];
      loom_target_contract_query_result_t query_result =
          loom_target_contract_query_result_empty();
      loom_target_contract_query_environment_t environment = {0};
      IREE_RETURN_IF_ERROR(loom_low_lower_query_environment_from_context(
          context, context->descriptor_set, &environment));
      IREE_RETURN_IF_ERROR(loom_low_lower_query_descriptor_matrix_contract(
          &environment, &context->policy->descriptor_matrix, matrix_rule,
          source_op, &query_result));
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
        query_result.rule_index = contract_case->row_index;
        IREE_RETURN_IF_ERROR(loom_low_lower_record_descriptor_matrix_plan(
            context, source_op, matrix_rule, &query_result));
        *out_selected = true;
        return iree_ok_status();
      }
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED ||
          query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
        IREE_RETURN_IF_ERROR(loom_low_lower_emit_contract_query_rejection(
            context, source_op, &query_result));
        *out_selected = true;
        return iree_ok_status();
      }
      continue;
    }
    uint16_t rule_index = UINT16_MAX;
    if (!loom_low_lower_contract_case_lower_rule_index(index, contract_case,
                                                       &rule_index)) {
      continue;
    }
    const loom_low_lower_rule_set_t* rule_set =
        context->policy->rule_sets.values[binding->rule_set_index];
    loom_low_lower_rule_selection_t rule_selection = {0};
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select_rule_range(
        context, rule_set, source_op, rule_index, 1, &rule_selection));
    if (rule_selection.rule != NULL) {
      IREE_RETURN_IF_ERROR(loom_low_lower_record_selected_rule_plan(
          context, source_op, binding->rule_set_index, rule_set,
          rule_selection));
      *out_selected = true;
      return iree_ok_status();
    }
    if (loom_low_lower_rule_selection_is_better_failure(
            *inout_failed_rule_set, *inout_failed_rule_selection,
            rule_selection)) {
      *inout_failed_rule_set = rule_set;
      *inout_failed_rule_selection = rule_selection;
    }
  }
  return iree_ok_status();
}

typedef struct loom_low_lower_contract_query_state_t {
  // Mutable lowering context whose scoped arena and target policy back the
  // read-only contract query.
  loom_low_lower_context_t* context;
  // Target contract environment provided by target-low legality.
  const loom_target_contract_query_environment_t* environment;
} loom_low_lower_contract_query_state_t;

static iree_status_t loom_low_lower_contract_query_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)match_context;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_low_lower_map_contract_value_callback_t map_contract_value =
      state->context->policy->map_contract_value;
  if (map_contract_value.fn == NULL) {
    return iree_ok_status();
  }
  return map_contract_value.fn(map_contract_value.user_data, state->environment,
                               source_op, source_value_id, out_mapped_value);
}

static bool loom_low_lower_contract_query_can_materialize(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id) {
  (void)match_context;
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const uint16_t materializer_index =
      (uint16_t)(value_ref->materializer_index - 1);
  const loom_low_lower_value_materializer_t* materializer =
      &rule_set->materializers[materializer_index];
  return materializer->can_materialize(state->context, source_value_id);
}

struct loom_low_lower_source_query_scope_t {
  // Read-only lowering context backing source-to-low contract queries.
  loom_low_lower_context_t context;
  // Diagnostic/result scratch required by the lowering context contract.
  loom_low_lower_result_t result;
  // True after context.lowering.value_domain has acquired module scratch.
  bool value_domain_initialized;
};

static iree_status_t loom_low_lower_query_target_contract_from_context(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_low_descriptor_set_t* saved_descriptor_set =
      context->descriptor_set;
  loom_value_fact_table_t* saved_fact_table = context->lowering.fact_table;
  const bool fact_table_changed =
      saved_fact_table != (loom_value_fact_table_t*)environment->fact_table;
  context->descriptor_set = environment->descriptor_set;
  context->lowering.fact_table =
      (loom_value_fact_table_t*)environment->fact_table;
  if (fact_table_changed) {
    context->lowering.view_regions_initialized = false;
    context->lowering.view_regions_analyzed = false;
  }
  iree_status_t status = iree_ok_status();
  loom_target_contract_query_environment_t query_environment = *environment;
  if (query_environment.view_regions == NULL) {
    const loom_view_region_table_t* view_regions = NULL;
    status = loom_low_lower_context_view_regions(context, &view_regions);
    query_environment.view_regions = view_regions;
  }
  loom_low_lower_contract_query_state_t state = {
      .context = context,
      .environment = &query_environment,
  };
  const loom_low_lower_contract_query_options_t query_options = {
      .contract_index = &context->contract_index,
      .rule_sets = context->policy->rule_sets,
      .map_value =
          {
              .fn = loom_low_lower_contract_query_map_value,
              .user_data = &state,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_contract_query_can_materialize,
              .user_data = &state,
          },
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
      .descriptor_matrix = context->policy->descriptor_matrix,
  };

  if (iree_status_is_ok(status)) {
    status = loom_low_lower_query_target_contract(
        &query_environment, &query_options, source_op, out_result);
  }
  context->descriptor_set = saved_descriptor_set;
  context->lowering.fact_table = saved_fact_table;
  if (fact_table_changed) {
    context->lowering.view_regions_initialized = false;
    context->lowering.view_regions_analyzed = false;
  }
  return status;
}

iree_status_t loom_low_lower_source_query_scope_create(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options, iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_scope);
  *out_scope = NULL;
  loom_low_lower_assert_options(module, source_function, options);

  loom_low_lower_source_query_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*scope), (void**)&scope));
  memset(scope, 0, sizeof(*scope));
  scope->result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  if (!iree_allocator_is_null(options->report_allocator)) {
    scope->result.report_allocator = options->report_allocator;
  }
  scope->context = (loom_low_lower_context_t){
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .result = &scope->result,
  };
  scope->context.lowering.fact_table = options->fact_table;
  iree_arena_initialize(module->arena.block_pool, &scope->context.arena);

  iree_status_t status = loom_target_low_descriptor_set_select_for_bundle(
      options->descriptor_registry, options->bundle,
      &scope->context.descriptor_set);
  loom_region_t* source_body = loom_func_like_body(source_function);
  if (iree_status_is_ok(status) && source_body != NULL) {
    status = loom_low_lowering_frame_initialize_value_ordinals(&scope->context,
                                                               source_body);
    scope->value_domain_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_contract_index_compose(
        scope->context.policy->contract_bindings,
        scope->context.policy->contract_binding_count,
        &scope->context.contract_index, &scope->context.arena);
  }
  if (!iree_status_is_ok(status)) {
    loom_low_lower_source_query_scope_destroy(scope);
    return status;
  }

  *out_scope = scope;
  return iree_ok_status();
}

void loom_low_lower_source_query_scope_destroy(
    loom_low_lower_source_query_scope_t* scope) {
  if (scope == NULL) {
    return;
  }
  if (scope->value_domain_initialized) {
    loom_low_lowering_frame_deinitialize(&scope->context);
    scope->value_domain_initialized = false;
  }
  loom_low_lower_result_deinitialize(&scope->result);
  iree_arena_deinitialize(&scope->context.arena);
  memset(scope, 0, sizeof(*scope));
}

loom_target_contract_query_callback_t
loom_low_lower_source_query_scope_callback(
    loom_low_lower_source_query_scope_t* scope) {
  IREE_ASSERT_ARGUMENT(scope);
  return (loom_target_contract_query_callback_t){
      .fn = loom_low_lower_query_target_contract_from_context,
      .user_data = &scope->context,
  };
}

const loom_local_value_domain_t* loom_low_lower_source_query_scope_value_domain(
    const loom_low_lower_source_query_scope_t* scope) {
  IREE_ASSERT_ARGUMENT(scope);
  return scope->value_domain_initialized ? &scope->context.lowering.value_domain
                                         : NULL;
}

static void loom_low_lower_report_row_list_deinitialize(
    iree_allocator_t allocator, loom_low_lower_report_row_list_t* list) {
  loom_low_lower_report_row_vec_t* vec = list->head;
  while (vec != NULL) {
    loom_low_lower_report_row_vec_t* next = vec->next;
    iree_allocator_free(allocator, vec);
    vec = next;
  }
  *list = (loom_low_lower_report_row_list_t){0};
}

void loom_low_lower_result_deinitialize(loom_low_lower_result_t* result) {
  if (result == NULL) {
    return;
  }
  loom_low_lower_report_row_list_deinitialize(result->report_allocator,
                                              &result->report_rows);
  result->report_allocator = iree_allocator_null();
}

static iree_status_t loom_low_lower_report_row_list_append(
    loom_low_lower_report_row_list_t* list, iree_allocator_t allocator,
    const loom_low_lower_report_row_t* row) {
  if (iree_allocator_is_null(allocator)) {
    return iree_ok_status();
  }
  if (list->tail == NULL || list->tail->count == list->tail->capacity) {
    iree_host_size_t capacity =
        (LOOM_LOW_LOWER_REPORT_ROW_VEC_DEFAULT_BYTE_LENGTH -
         sizeof(loom_low_lower_report_row_vec_t)) /
        sizeof(*row);
    capacity = iree_max((iree_host_size_t)1, capacity);
    iree_host_size_t row_bytes = 0;
    if (!iree_host_size_checked_mul(capacity, sizeof(*row), &row_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source-to-low report row block is too large");
    }
    iree_host_size_t block_bytes = 0;
    if (!iree_host_size_checked_add(sizeof(loom_low_lower_report_row_vec_t),
                                    row_bytes, &block_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source-to-low report row block is too large");
    }
    loom_low_lower_report_row_vec_t* vec = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, block_bytes, (void**)&vec));
    *vec = (loom_low_lower_report_row_vec_t){
        .capacity = capacity,
    };
    if (list->tail != NULL) {
      list->tail->next = vec;
    } else {
      list->head = vec;
    }
    list->tail = vec;
  }
  loom_low_lower_report_row_t* rows =
      loom_low_lower_report_row_vec_rows(list->tail);
  rows[list->tail->count++] = *row;
  ++list->count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_record_report_row(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan,
    uint32_t emitted_low_op_count) {
  loom_low_lower_result_t* result = context->result;
  if (iree_allocator_is_null(context->options->report_allocator)) {
    return iree_ok_status();
  }
  ++result->selected_source_op_count;
  result->emitted_low_op_count += emitted_low_op_count;

  loom_low_lower_report_row_t row = {
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name = loom_op_name(context->module, selected_plan->source_op),
      .source_op_kind = selected_plan->source_op->kind,
      .selection_kind = LOOM_LOW_LOWER_REPORT_SELECTION_PLAN,
      .rule_set_index = UINT16_MAX,
      .rule_index = UINT16_MAX,
      .plan_id = selected_plan->plan.id,
      .plan_detail = iree_string_view_empty(),
      .descriptor_id = LOOM_LOW_STABLE_ID_NONE,
      .emitted_low_op_count = emitted_low_op_count,
  };
  if (selected_plan->rule != NULL) {
    row.selection_kind = LOOM_LOW_LOWER_REPORT_SELECTION_RULE;
    row.rule_set_index = selected_plan->rule_set_index;
    row.rule_index = selected_plan->rule_index;
    row.plan_id = LOOM_LOW_LOWER_PLAN_ID_NONE;
    if (selected_plan->rule->emit_count != 0 &&
        selected_plan->resolved_emits != NULL) {
      row.descriptor_id =
          selected_plan->resolved_emits[0].descriptor.descriptor->stable_id;
    }
  } else if (selected_plan->kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX) {
    const loom_low_lower_descriptor_matrix_plan_t* plan =
        (const loom_low_lower_descriptor_matrix_plan_t*)
            selected_plan->plan.target_data;
    row.descriptor_id = plan->descriptor.descriptor->stable_id;
  }
  if (selected_plan->rule == NULL &&
      context->policy->describe_plan.fn != NULL) {
    row.plan_detail = context->policy->describe_plan.fn(
        context->policy->describe_plan.user_data, context,
        selected_plan->source_op, selected_plan->plan);
  }
  return loom_low_lower_report_row_list_append(&result->report_rows,
                                               result->report_allocator, &row);
}

static iree_status_t loom_low_lower_try_select_op_callback(
    loom_low_lower_context_t* context,
    loom_low_lower_select_op_callback_t callback, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  if (callback.fn == NULL) {
    return iree_ok_status();
  }

  loom_low_lower_plan_t plan = loom_low_lower_plan_empty();
  IREE_RETURN_IF_ERROR(
      callback.fn(callback.user_data, context, source_op, &plan));
  if (loom_low_lower_plan_is_empty(plan)) {
    return iree_ok_status();
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = UINT16_MAX,
                   .rule_set = NULL,
                   .rule = NULL,
                   .plan = plan,
               });
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  if (source_op->region_count != 0) {
    if (loom_low_lower_supported_structured_source_op(context, source_op)) {
      return iree_ok_status();
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(source_op->region_count),
    };
    return loom_low_lower_emit_target_context_error(context, source_op,
                                                    LOOM_ERR_TARGET_030, params,
                                                    IREE_ARRAYSIZE(params));
  }
  if (loom_low_lower_op_is_structural(context->module, source_op)) {
    return iree_ok_status();
  }
  if (loom_low_lower_op_is_source_metadata(source_op->kind)) {
    return iree_ok_status();
  }

  bool selected_callback = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, context->policy->preselect_op, source_op, &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_rule_selection = {0};
  bool selected_rule = false;
  if (context->contract_index.case_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_op_from_contract_index(
        context, source_op, &failed_rule_set, &failed_rule_selection,
        &selected_rule));
    if (selected_rule) {
      return iree_ok_status();
    }
    if (failed_rule_set != NULL) {
      return loom_low_lower_rule_set_emit_selection_failure(
          context, failed_rule_set, source_op, failed_rule_selection);
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, context->policy->select_op, source_op, &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  if (failed_rule_set != NULL) {
    return loom_low_lower_rule_set_emit_selection_failure(
        context, failed_rule_set, source_op, failed_rule_selection);
  }
  if (loom_low_lower_op_is_discardable_hint(context->module, source_op)) {
    loom_low_lower_record_elided_hint_plan(context, source_op);
    return iree_ok_status();
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_status_t loom_low_lower_plan_region(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    const loom_op_t* block_arg_context_op, bool skip_entry_block_args) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    if (!(skip_entry_block_args && block_index == 0)) {
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        loom_type_t low_type = loom_type_none();
        IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
            context, block_arg_context_op, block->arg_ids[i], &low_type));
        if (loom_low_lower_context_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_low_lower_plan_op(context, op));
      if (loom_low_lower_context_should_stop(context)) {
        return iree_ok_status();
      }
      if (!loom_low_lower_supported_structured_source_op(context, op)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_lower_plan_region(context, regions[i], op,
                                       /*skip_entry_block_args=*/false));
        if (loom_low_lower_context_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_body(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  IREE_RETURN_IF_ERROR(
      loom_low_lower_check_function_signature(context, source_body));
  if (loom_low_lower_context_should_stop(context)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_prepare_plan(context, source_body));

  IREE_RETURN_IF_ERROR(loom_low_lower_plan_region(
      context, source_body, context->source_function.op,
      /*skip_entry_block_args=*/true));
  loom_low_lower_analyze_storage_demands(context, source_body);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_signature_types(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_type_t** out_arg_types, iree_host_size_t* out_arg_count,
    loom_type_t** out_result_types, iree_host_size_t* out_result_count) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));
  *out_arg_types = NULL;
  *out_arg_count = 0;
  *out_result_types = NULL;
  *out_result_count = 0;

  uint16_t argument_count = 0;
  (void)loom_func_like_arg_ids(context->source_function, &argument_count);
  loom_type_t* arg_types = NULL;
  const uint16_t direct_argument_count =
      loom_low_lower_direct_argument_count(context);
  if (direct_argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&context->arena, direct_argument_count,
                                  sizeof(*arg_types), (void**)&arg_types));
    uint16_t direct_argument_index = 0;
    for (uint16_t i = 0; i < argument_count; ++i) {
      if (context->lowering.argument_map[i].kind !=
          LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
        continue;
      }
      arg_types[direct_argument_index] =
          context->lowering.argument_map[i].abi_type;
      IREE_ASSERT_FALSE(
          loom_low_lower_type_is_none(arg_types[direct_argument_index]));
      ++direct_argument_index;
    }
  }

  const uint16_t result_count = context->source_function.op->result_count;
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &context->arena, result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* result_ids =
        loom_op_const_results(context->source_function.op);
    const loom_op_t* return_op = NULL;
    loom_value_slice_t returned_values = {0};
    (void)loom_low_lower_first_return_operands(source_body, &return_op,
                                               &returned_values);
    for (uint16_t i = 0; i < result_count; ++i) {
      if (i < returned_values.count) {
        IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
            context, return_op, returned_values.values[i], &result_types[i]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_map_value(context, context->source_function.op,
                                     result_ids[i], &result_types[i]));
      }
      IREE_ASSERT_FALSE(loom_low_lower_type_is_none(result_types[i]));
    }
  }

  *out_arg_types = arg_types;
  *out_arg_count = direct_argument_count;
  *out_result_types = result_types;
  *out_result_count = result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_abi_layout(
    loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_named_attr_slice_t* out_abi_layout) {
  *out_abi_layout = loom_named_attr_slice_empty();
  if (context->policy->map_abi_layout.fn == NULL) {
    return iree_ok_status();
  }
  return context->policy->map_abi_layout.fn(
      context->policy->map_abi_layout.user_data, context, layout_kind,
      arg_types, arg_count, result_types, result_count, out_abi_layout);
}

static loom_region_t* loom_low_lower_low_body(
    const loom_low_lower_context_t* context) {
  if (loom_low_func_def_isa(context->low_func_op)) {
    return loom_low_func_def_body(context->low_func_op);
  }
  if (loom_low_kernel_def_isa(context->low_func_op)) {
    return loom_low_kernel_def_body(context->low_func_op);
  }
  return NULL;
}

static bool loom_low_lower_source_is_kernel_def(
    const loom_low_lower_context_t* context) {
  return loom_kernel_def_isa(context->source_function.op);
}

static iree_status_t loom_low_lower_create_func_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(context->source_function, &predicate_count);

  loom_low_func_def_build_flags_t build_flags = 0;
  uint8_t visibility = loom_func_like_visibility(context->source_function);
  uint8_t cc = loom_func_like_cc(context->source_function);
  uint8_t purity = loom_func_like_purity(context->source_function);
  loom_target_abi_kind_t abi = loom_low_lower_function_abi(context);
  loom_named_attr_slice_t abi_attrs =
      loom_func_like_abi_attrs(context->source_function);
  loom_named_attr_slice_t abi_layout = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_abi_layout(
      context, LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC, arg_types, arg_count,
      result_types, result_count, &abi_layout));
  loom_string_id_t export_symbol =
      loom_func_like_export_symbol(context->source_function);
  loom_named_attr_slice_t export_attrs =
      loom_func_like_export_attrs(context->source_function);
  if (visibility != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
  }
  if (cc != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_CC;
  }
  if (purity != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_PURITY;
  }
  if (abi != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI;
  }
  if (export_symbol != LOOM_STRING_ID_INVALID) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }
  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module),
                          &context->builder);
  loom_builder_set_before(&context->builder, context->source_function.op);
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &context->builder, build_flags, visibility, cc, purity,
      /*allocation=*/0, /*schedule=*/0, context->options->target_ref, abi,
      abi_attrs, abi_layout, export_symbol, export_attrs, low_func_ref,
      arg_types, arg_count, result_types, result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, predicates, predicate_count,
      context->source_function.op->location, &context->low_func_op));

  loom_region_t* low_body = loom_low_lower_low_body(context);
  low_body->flags = source_body->flags;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_kernel_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref, const loom_type_t* arg_types,
    iree_host_size_t arg_count) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(context->source_function, &predicate_count);

  loom_low_kernel_def_build_flags_t build_flags = 0;
  loom_string_id_t export_symbol =
      loom_func_like_export_symbol(context->source_function);
  if (export_symbol != LOOM_STRING_ID_INVALID) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }

  uint8_t export_linkage = 0;
  if (loom_func_like_export_linkage(context->source_function,
                                    &export_linkage)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_LINKAGE;
  }

  loom_target_workgroup_size_t workgroup_size = {0};
  if (loom_kernel_def_static_workgroup_size_from_facts(
          context->module, context->source_function.op,
          context->lowering.fact_table, &workgroup_size)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_X |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Y |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Z;
  }

  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module),
                          &context->builder);
  loom_builder_set_before(&context->builder, context->source_function.op);
  loom_named_attr_slice_t abi_layout = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_abi_layout(
      context, LOOM_LOW_LOWER_ABI_LAYOUT_KIND_KERNEL, arg_types, arg_count,
      /*result_types=*/NULL, /*result_count=*/0, &abi_layout));
  IREE_RETURN_IF_ERROR(loom_low_kernel_def_build(
      &context->builder, build_flags, /*allocation=*/0, /*schedule=*/0,
      context->options->target_ref, abi_layout, export_symbol, export_linkage,
      workgroup_size.x, workgroup_size.y, workgroup_size.z, low_func_ref,
      arg_types, arg_count, predicates, predicate_count,
      context->source_function.op->location, &context->low_func_op));

  loom_region_t* low_body = loom_low_lower_low_body(context);
  low_body->flags = source_body->flags;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_function_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref) {
  loom_type_t* arg_types = NULL;
  iree_host_size_t arg_count = 0;
  loom_type_t* result_types = NULL;
  iree_host_size_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_signature_types(
      context, source_body, &arg_types, &arg_count, &result_types,
      &result_count));

  if (loom_low_lower_source_is_kernel_def(context)) {
    IREE_ASSERT_EQ(result_count, 0);
    IREE_RETURN_IF_ERROR(loom_low_lower_create_kernel_op(
        context, source_body, low_func_ref, arg_types, arg_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_lower_create_func_op(
        context, source_body, low_func_ref, arg_types, arg_count, result_types,
        result_count));
  }

  context->result->low_func_op = context->low_func_op;
  context->result->low_func_ref = low_func_ref;

  const loom_value_id_t* source_results =
      loom_op_const_results(context->source_function.op);
  const loom_value_id_t* low_results =
      loom_op_const_results(context->low_func_op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_decl_signature_types(
    loom_low_lower_context_t* context, loom_type_t** out_arg_types,
    iree_host_size_t* out_arg_count, loom_type_t** out_result_types,
    iree_host_size_t* out_result_count) {
  *out_arg_types = NULL;
  *out_arg_count = 0;
  *out_result_types = NULL;
  *out_result_count = 0;

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  loom_type_t* arg_types = NULL;
  if (argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&context->arena, argument_count,
                                  sizeof(*arg_types), (void**)&arg_types));
    for (uint16_t i = 0; i < argument_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
          context, context->source_function.op, argument_ids[i],
          &arg_types[i]));
    }
  }

  const uint16_t result_count = context->source_function.op->result_count;
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &context->arena, result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* result_ids =
        loom_op_const_results(context->source_function.op);
    for (uint16_t i = 0; i < result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
          context, context->source_function.op, result_ids[i],
          &result_types[i]));
    }
  }

  *out_arg_types = arg_types;
  *out_arg_count = argument_count;
  *out_result_types = result_types;
  *out_result_count = result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_copy_decl_signature_names(
    loom_low_lower_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  const loom_value_id_t* low_arguments =
      loom_op_const_operands(context->low_func_op);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_arguments[i], low_arguments[i]));
  }

  const loom_value_id_t* source_results =
      loom_op_const_results(context->source_function.op);
  const loom_value_id_t* low_results =
      loom_op_const_results(context->low_func_op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_import_declaration(
    loom_module_t* module, loom_func_like_t source_declaration,
    const loom_low_lower_options_t* options,
    loom_low_lower_result_t* out_result) {
  IREE_ASSERT(out_result != NULL);
  IREE_ASSERT(loom_func_like_isa(source_declaration));
  IREE_ASSERT(options != NULL);
  IREE_ASSERT(options->policy != NULL);
  IREE_ASSERT_NE(options->policy->import_decl_kind, 0);
  *out_result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  if (!iree_allocator_is_null(options->report_allocator)) {
    out_result->report_allocator = options->report_allocator;
  }

  const loom_symbol_ref_t low_func_ref =
      loom_func_like_callee(source_declaration);
  IREE_ASSERT(loom_symbol_ref_is_valid(low_func_ref));
  IREE_ASSERT_EQ(low_func_ref.module_id, 0);
  IREE_ASSERT_LT(low_func_ref.symbol_id, module->symbols.count);

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      options->descriptor_registry, options->bundle, &descriptor_set));

  loom_low_lower_context_t context = {
      .module = module,
      .source_function = source_declaration,
      .options = options,
      .policy = options->policy,
      .descriptor_set = descriptor_set,
      .result = out_result,
  };
  out_result->descriptor_set = descriptor_set;
  iree_arena_initialize(module->arena.block_pool, &context.arena);

  loom_type_t* arg_types = NULL;
  iree_host_size_t arg_count = 0;
  loom_type_t* result_types = NULL;
  iree_host_size_t result_count = 0;
  iree_status_t status = loom_low_lower_map_decl_signature_types(
      &context, &arg_types, &arg_count, &result_types, &result_count);
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    loom_string_id_t code_symbol =
        loom_func_like_import_symbol(source_declaration);
    if (code_symbol == LOOM_STRING_ID_INVALID) {
      code_symbol = module->symbols.entries[low_func_ref.symbol_id].name_id;
    }
    IREE_ASSERT_NE(code_symbol, LOOM_STRING_ID_INVALID);
    IREE_ASSERT_LT(code_symbol, module->strings.count);
    loom_low_func_decl_build_flags_t build_flags =
        LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_KIND |
        LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CODE_SYMBOL;
    const uint8_t visibility = loom_func_like_visibility(source_declaration);
    const uint8_t cc = loom_func_like_cc(source_declaration);
    const uint8_t purity = loom_func_like_purity(source_declaration);
    const bool has_abi = loom_low_lower_function_attr_present(
        source_declaration, source_declaration.vtable->abi_attr_index);
    const loom_target_abi_kind_t abi =
        (loom_target_abi_kind_t)loom_func_like_abi(source_declaration);
    loom_named_attr_slice_t abi_attrs =
        loom_func_like_abi_attrs(source_declaration);
    loom_named_attr_slice_t abi_layout = loom_named_attr_slice_empty();
    status = loom_low_lower_map_abi_layout(
        &context, LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC, arg_types, arg_count,
        result_types, result_count, &abi_layout);
    loom_string_id_t export_symbol =
        loom_func_like_export_symbol(source_declaration);
    loom_named_attr_slice_t export_attrs =
        loom_func_like_export_attrs(source_declaration);
    if (visibility != 0) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_VISIBILITY;
    }
    if (cc != 0) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CC;
    }
    if (purity != 0) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_PURITY;
    }
    if (has_abi) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI;
    }
    if (export_symbol != LOOM_STRING_ID_INVALID) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_SYMBOL;
    }

    if (iree_status_is_ok(status)) {
      uint16_t predicate_count = 0;
      const loom_predicate_t* predicates =
          loom_func_like_predicates(source_declaration, &predicate_count);
      loom_builder_initialize(module, &module->arena, loom_module_block(module),
                              &context.builder);
      loom_builder_set_before(&context.builder, source_declaration.op);
      status = loom_low_func_decl_build(
          &context.builder, build_flags, visibility, cc, purity,
          /*allocation=*/0, /*schedule=*/0,
          (uint8_t)options->policy->import_decl_kind, code_symbol,
          options->target_ref, abi, abi_attrs, abi_layout, export_symbol,
          export_attrs, low_func_ref, arg_types, arg_count, result_types,
          result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
          predicates, predicate_count, source_declaration.op->location,
          &context.low_func_op);
    }
  }

  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    status = loom_low_lower_copy_decl_signature_names(&context);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    out_result->low_func_op = context.low_func_op;
    out_result->low_func_ref = low_func_ref;
    status = loom_op_erase(module, source_declaration.op);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    loom_module_link_symbol_defining_op(
        module, context.low_func_op,
        loom_op_vtable(module, context.low_func_op));
  }

  iree_arena_deinitialize(&context.arena);
  return status;
}

static iree_status_t loom_low_lower_map_blocks(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_low_body(context);
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->arena, source_body->block_count,
                                sizeof(*context->lowering.block_map),
                                (void**)&context->lowering.block_map));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, source_body->block_count,
      sizeof(*context->lowering.successor_interpositions),
      (void**)&context->lowering.successor_interpositions));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->arena, source_body->block_count,
                                sizeof(*context->lowering.branch_plans),
                                (void**)&context->lowering.branch_plans));
  memset(context->lowering.block_map, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.block_map));
  memset(context->lowering.successor_interpositions, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.successor_interpositions));
  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    context->lowering.branch_plans[i] = loom_low_lower_plan_empty();
  }

  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    loom_block_t* source_block = loom_region_block(source_body, i);
    loom_block_t* low_block = NULL;
    if (i == 0) {
      low_block = loom_region_entry_block(low_body);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(context->module, low_body, &low_block));
    }
    low_block->label_id = source_block->label_id;
    context->lowering.block_map[i] = low_block;
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_block_t* low_block = context->lowering.block_map[block_index];
    if (block_index == 0) {
      const uint16_t direct_argument_count =
          loom_low_lower_direct_argument_count(context);
      IREE_ASSERT_EQ(low_block->arg_count, direct_argument_count);
      uint16_t direct_argument_index = 0;
      for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
           ++arg_index) {
        if (context->lowering.argument_map[arg_index].kind !=
            LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, source_block->arg_ids[arg_index],
            low_block->arg_ids[direct_argument_index]));
        ++direct_argument_index;
      }
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
         ++arg_index) {
      loom_type_t low_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
          context, context->source_function.op,
          source_block->arg_ids[arg_index], &low_type));
      IREE_ASSERT_FALSE(loom_low_lower_type_is_none(low_type));
      loom_value_id_t low_arg = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder, low_block, low_type, &low_arg));
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_block->arg_ids[arg_index], low_arg));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_argument_resource_import(
    loom_low_lower_context_t* context, const loom_value_id_t* source_arguments,
    uint16_t argument_index) {
  const loom_low_lower_abi_argument_t* argument =
      &context->lowering.argument_map[argument_index];
  loom_type_t source_type = argument->resource_source_type;
  if (loom_low_lower_type_is_none(source_type)) {
    source_type = loom_module_value_type(context->module,
                                         source_arguments[argument_index]);
  }
  loom_type_id_t source_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_intern_type_id(context, source_type, &source_type_id));
  loom_op_t* resource_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resource_build(
      &context->builder, argument->resource_build_flags,
      (uint8_t)argument->resource_import_kind, LOOM_VALUE_ID_INVALID,
      argument->resource_index, source_type_id, argument->resource_extent,
      argument->resource_cache_swizzle_stride, argument->abi_type,
      context->source_function.op->location, &resource_op));
  return loom_low_lower_bind_value(context, source_arguments[argument_index],
                                   loom_low_resource_result(resource_op));
}

static iree_status_t loom_low_lower_emit_argument_resource_imports(
    loom_low_lower_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  if (argument_count == 0 ||
      argument_count == loom_low_lower_direct_argument_count(context)) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    if (context->lowering.argument_map[i].kind !=
        LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      continue;
    }
    status = loom_low_lower_emit_argument_resource_import(context,
                                                          source_arguments, i);
  }

  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_preamble(
    loom_low_lower_context_t* context) {
  if (context->policy->emit_preamble.fn == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->emit_preamble.fn(
      context->policy->emit_preamble.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_entry_setup(
    loom_low_lower_context_t* context) {
  if (context->policy->emit_entry_setup.fn == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->emit_entry_setup.fn(
      context->policy->emit_entry_setup.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_prepare_branches(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  if (context->policy->prepare_branch.fn == NULL) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_block(source_body, block_index);
    const loom_op_t* source_terminator = loom_block_const_last_op(source_block);
    if (source_terminator == NULL || source_terminator->successor_count == 0) {
      continue;
    }
    if (loom_low_lower_cfg_cond_br_exact_bool(context, source_terminator,
                                              NULL)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(context->policy->prepare_branch.fn(
        context->policy->prepare_branch.user_data, context, source_terminator));
    if (loom_low_lower_context_should_stop(context)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_remap_values(
    loom_low_lower_context_t* context, const loom_value_id_t* source_values,
    iree_host_size_t value_count, loom_value_id_t** out_low_values) {
  *out_low_values = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, value_count, sizeof(*low_values), (void**)&low_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_values[i], &low_values[i]));
  }
  *out_low_values = low_values;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_bind_identity_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_EQ(source_op->operand_count, source_op->result_count);
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value_alias(
        context, source_operands[i], source_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks);

static iree_status_t loom_low_lower_map_op_result_types(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  if (source_op->result_count == 0) {
    return iree_ok_status();
  }
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->arena, source_op->result_count,
                                sizeof(*result_types), (void**)&result_types));
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
        context, source_op, source_results[i], &result_types[i]));
    if (loom_low_lower_type_is_none(result_types[i])) {
      return iree_ok_status();
    }
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_bind_op_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* low_op) {
  IREE_ASSERT_EQ(source_op->result_count, low_op->result_count);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  const loom_value_id_t* low_results = loom_op_const_results(low_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_scf_yield(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_slice_t values = loom_scf_yield_values(source_op);
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(context, values.values,
                                                   values.count, &low_values));
  loom_op_t* low_yield_op = NULL;
  return loom_low_scf_yield_build(&context->builder, low_values, values.count,
                                  source_op->location, &low_yield_op);
}

static iree_status_t loom_low_lower_emit_scf_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_if_condition(source_op), &low_condition));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (source_op->result_count != 0 && result_types == NULL) {
    return iree_ok_status();
  }

  loom_low_scf_if_build_flags_t build_flags = 0;
  if (loom_scf_if_else_region(source_op) != NULL) {
    build_flags |= LOOM_LOW_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION;
  }
  loom_op_t* low_if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_if_build(
      &context->builder, build_flags, low_condition, result_types,
      source_op->result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_if_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_if_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_if_op, loom_low_scf_if_then_region(low_if_op));
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, loom_scf_if_then_region(source_op),
      /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_region_t* source_else_region = loom_scf_if_else_region(source_op);
  if (source_else_region == NULL) {
    return iree_ok_status();
  }
  saved_ip = loom_builder_enter_region(&context->builder, low_if_op,
                                       loom_low_scf_if_else_region(low_if_op));
  status = loom_low_lower_emit_region_ops(context, source_else_region,
                                          /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_bind_for_body_args(
    loom_low_lower_context_t* context, const loom_op_t* source_for_op,
    loom_op_t* low_for_op) {
  loom_block_t* source_entry =
      loom_region_entry_block(loom_scf_for_body(source_for_op));
  loom_block_t* low_entry =
      loom_region_entry_block(loom_low_scf_for_body(low_for_op));
  IREE_ASSERT_EQ(source_entry->arg_count, low_entry->arg_count);
  for (uint16_t i = 0; i < source_entry->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, source_entry->arg_ids[i], low_entry->arg_ids[i]));
  }
  return iree_ok_status();
}

static bool loom_low_lower_op_attr_present(const loom_op_t* op,
                                           uint8_t attr_index) {
  return !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static iree_status_t loom_low_lower_emit_scf_for(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_lower_bound(source_op), &low_lower_bound));
  loom_value_id_t low_upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_upper_bound(source_op), &low_upper_bound));
  loom_value_id_t low_step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_step(source_op), &low_step));

  loom_value_slice_t iter_args = loom_scf_for_iter_args(source_op);
  loom_value_id_t* low_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, iter_args.values, iter_args.count, &low_iter_args));

  loom_value_id_t low_unroll_factor = LOOM_VALUE_ID_INVALID;
  loom_low_scf_for_build_flags_t build_flags = 0;
  if (loom_scf_for_unroll_factor_is_present(source_op)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_scf_for_unroll_factor(source_op), &low_unroll_factor));
  }

  uint8_t unroll_policy = 0;
  if (loom_low_lower_op_attr_present(source_op,
                                     loom_scf_for_unroll_policy_ATTR_INDEX)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = (uint8_t)loom_scf_for_unroll_policy(source_op);
  }

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (source_op->result_count != 0 && result_types == NULL) {
    return iree_ok_status();
  }

  loom_op_t* low_for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_for_build(
      &context->builder, build_flags, low_lower_bound, low_upper_bound,
      low_step, low_iter_args, iter_args.count, result_types,
      source_op->result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, low_unroll_factor, unroll_policy,
      source_op->location, &low_for_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_for_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_for_body_args(context, source_op, low_for_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_for_op, loom_low_scf_for_body(low_for_op));
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, loom_scf_for_body(source_op), /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_structural_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  *out_handled = true;
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits)) {
    return loom_low_lower_bind_identity_results(context, source_op);
  }
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(source_op->operand_count >= 1);
    IREE_ASSERT(source_op->result_count == 1);
    return loom_low_lower_bind_value_alias(context,
                                           loom_op_const_operands(source_op)[0],
                                           loom_op_const_results(source_op)[0]);
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT: {
      return loom_low_lower_bind_value_alias(
          context, loom_buffer_assume_same_root_buffer(source_op),
          loom_buffer_assume_same_root_result(source_op));
    }
    case LOOM_OP_FUNC_RETURN: {
      loom_value_slice_t values = loom_func_return_operands(source_op);
      loom_value_id_t* low_values = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, values.values, values.count, &low_values));
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, low_values, values.count,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_FUNC_CALL: {
      loom_value_slice_t operands = loom_func_call_operands(source_op);
      loom_value_id_t* low_operands = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, operands.values, operands.count, &low_operands));

      const loom_value_id_t* source_results = loom_op_const_results(source_op);
      loom_type_t* result_types = NULL;
      bool has_unmapped_result = false;
      if (source_op->result_count != 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &context->arena, source_op->result_count, sizeof(*result_types),
            (void**)&result_types));
        for (uint16_t i = 0; i < source_op->result_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
              context, source_op, source_results[i], &result_types[i]));
          has_unmapped_result |= loom_low_lower_type_is_none(result_types[i]);
        }
      }
      if (has_unmapped_result) {
        return iree_ok_status();
      }

      loom_low_func_call_build_flags_t build_flags = 0;
      uint8_t purity = loom_func_call_purity(source_op);
      if (purity != 0) {
        build_flags |= LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_PURITY;
      }
      loom_op_t* low_call_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_func_call_build(
          &context->builder, build_flags, purity,
          loom_func_call_callee(source_op), low_operands, operands.count,
          result_types, source_op->result_count,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &low_call_op));

      const loom_value_id_t* low_results = loom_op_const_results(low_call_op);
      for (uint16_t i = 0; i < source_op->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, source_results[i], low_results[i]));
      }
      return iree_ok_status();
    }
    case LOOM_OP_KERNEL_RETURN: {
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, NULL, 0,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_CFG_BR: {
      loom_block_t* low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_dest));
      loom_value_slice_t args = loom_cfg_br_args(source_op);
      loom_value_id_t* low_args = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_successor_args(
          context, source_op, 0, low_dest, args.values, args.count, &low_args));
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(&context->builder, low_dest, low_args,
                               args.count, source_op->location, &low_br_op);
    }
    case LOOM_OP_CFG_COND_BR: {
      loom_block_t* low_true_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_true_dest));
      loom_block_t* low_false_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 1, &low_false_dest));
      bool condition = false;
      if (loom_low_lower_cfg_cond_br_exact_bool(context, source_op,
                                                &condition)) {
        loom_block_t* low_dest = condition ? low_true_dest : low_false_dest;
        loom_op_t* low_br_op = NULL;
        return loom_low_br_build(&context->builder, low_dest, NULL, 0,
                                 source_op->location, &low_br_op);
      }
      loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, loom_cfg_cond_br_condition(source_op), &low_condition));
      if (context->policy->emit_cond_branch.fn != NULL) {
        return context->policy->emit_cond_branch.fn(
            context->policy->emit_cond_branch.user_data, context, source_op,
            low_condition, low_true_dest, low_false_dest);
      }
      loom_op_t* low_cond_br_op = NULL;
      return loom_low_cond_br_build(&context->builder, low_condition,
                                    low_true_dest, low_false_dest,
                                    source_op->location, &low_cond_br_op);
    }
    case LOOM_OP_SCF_YIELD:
      return loom_low_lower_emit_scf_yield(context, source_op);
    case LOOM_OP_SCF_IF:
      return loom_low_lower_emit_scf_if(context, source_op);
    case LOOM_OP_SCF_FOR:
      return loom_low_lower_emit_scf_for(context, source_op);
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_low_lower_emit_elided_selected_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_elide_value(context, source_results[i]));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_record_report_row(
      context, selected_plan, /*emitted_low_op_count=*/0));
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_sparse_source_value(
    const loom_contract_request_t* request, loom_value_id_t* out_source_value) {
  *out_source_value = LOOM_VALUE_ID_INVALID;
  const loom_contract_operand_t* operands[] = {
      &request->lhs,
      &request->rhs,
      &request->accumulator,
      &request->result,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
    const loom_contract_value_ref_t ref =
        operands[i]->encoded.auxiliary_value_refs
            [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA];
    if (!loom_contract_value_ref_is_present(ref)) {
      continue;
    }
    const loom_value_id_t source_value = loom_contract_value_ref_value_id(ref);
    if (*out_source_value == LOOM_VALUE_ID_INVALID) {
      *out_source_value = source_value;
      continue;
    }
    if (*out_source_value != source_value) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "descriptor-matrix selected sparse source is ambiguous");
    }
  }
  if (*out_source_value == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "descriptor-matrix selected sparse source is "
                            "unavailable");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_auxiliary_source_value(
    const loom_contract_operand_t* operand,
    loom_contract_auxiliary_operand_key_t key, iree_string_view_t field_name,
    loom_value_id_t* out_source_value) {
  *out_source_value = LOOM_VALUE_ID_INVALID;
  const loom_contract_value_ref_t ref =
      operand->encoded.auxiliary_value_refs[key];
  if (!loom_contract_value_ref_is_present(ref)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "descriptor-matrix selected auxiliary operand '%.*s' is unavailable",
        (int)field_name.size, field_name.data);
  }
  *out_source_value = loom_contract_value_ref_value_id(ref);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    iree_string_view_t field_name, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t* out_low_value) {
  if (iree_string_view_equal(field_name, IREE_SV("a"))) {
    *out_low_value = low_lhs;
    return iree_ok_status();
  }
  if (iree_string_view_equal(field_name, IREE_SV("b"))) {
    *out_low_value = low_rhs;
    return iree_ok_status();
  }
  if (iree_string_view_equal(field_name, IREE_SV("acc"))) {
    *out_low_value = low_init;
    return iree_ok_status();
  }
  if (iree_string_view_equal(field_name, IREE_SV("index")) ||
      iree_string_view_equal(field_name, IREE_SV("metadata")) ||
      iree_string_view_equal(field_name, IREE_SV("sparsity"))) {
    loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_sparse_source_value(
        &plan->contract_request, &source_value));
    return loom_low_lower_lookup_value(context, source_value, out_low_value);
  }
  if (iree_string_view_equal(field_name, IREE_SV("scale_src0"))) {
    loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_descriptor_matrix_auxiliary_source_value(
            &plan->contract_request.lhs,
            LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE, field_name,
            &source_value));
    return loom_low_lower_lookup_value(context, source_value, out_low_value);
  }
  if (iree_string_view_equal(field_name, IREE_SV("scale_src1"))) {
    loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_descriptor_matrix_auxiliary_source_value(
            &plan->contract_request.rhs,
            LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE, field_name,
            &source_value));
    return loom_low_lower_lookup_value(context, source_value, out_low_value);
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "descriptor-matrix selected packet field '%.*s' is unmapped",
      (int)field_name.size, field_name.data);
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t** out_operands, iree_host_size_t* out_operand_count,
    const uint16_t** out_descriptor_operand_packet_indices) {
  *out_operands = NULL;
  *out_operand_count = 0;
  *out_descriptor_operand_packet_indices = NULL;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;

  uint16_t* descriptor_operand_packet_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->arena, descriptor->operand_count,
                                sizeof(*descriptor_operand_packet_indices),
                                (void**)&descriptor_operand_packet_indices));
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    descriptor_operand_packet_indices[i] = UINT16_MAX;
  }

  iree_host_size_t operand_count = 0;
  IREE_ASSERT((uint64_t)descriptor->operand_start +
                  (uint64_t)descriptor->operand_count <=
              descriptor_set->operand_count);
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const uint32_t row = descriptor->operand_start + i;
    const loom_low_operand_t* operand = &descriptor_set->operands[row];
    if (operand->role != LOOM_LOW_OPERAND_ROLE_IMPLICIT) {
      descriptor_operand_packet_indices[i] = (uint16_t)operand_count++;
    }
  }

  loom_value_id_t* operands = NULL;
  if (operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &context->arena, operand_count, sizeof(*operands), (void**)&operands));
  }

  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    if (descriptor_operand_packet_indices[i] == UINT16_MAX) {
      continue;
    }
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    iree_string_view_t field_name = loom_low_descriptor_set_string(
        descriptor_set, operand->field_name_string_offset);
    IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_value(
        context, plan, field_name, low_lhs, low_rhs, low_init,
        &operands[descriptor_operand_packet_indices[i]]));
  }

  *out_operands = operands;
  *out_operand_count = operand_count;
  *out_descriptor_operand_packet_indices = descriptor_operand_packet_indices;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_tied_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    const uint16_t* descriptor_operand_packet_indices,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  iree_host_size_t tied_result_count = 0;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t row = descriptor->constraint_start + i;
    if (descriptor_set->constraints[row].kind ==
        LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->arena, tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));
  iree_host_size_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
    if (constraint->lhs_operand_index >= descriptor->result_count ||
        constraint->rhs_operand_index == LOOM_LOW_ID_NONE ||
        constraint->rhs_operand_index >= descriptor->operand_count ||
        descriptor_operand_packet_indices[constraint->rhs_operand_index] ==
            UINT16_MAX) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "descriptor-matrix selected tied result "
                              "constraint is invalid");
    }
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = constraint->lhs_operand_index,
        .operand_index =
            descriptor_operand_packet_indices[constraint->rhs_operand_index],
        .has_type_change = false,
    };
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static bool loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const uint16_t* descriptor_operand_packet_indices,
    uint16_t constraint_index, uint16_t packet_operand_index) {
  for (uint16_t i = 0; i < constraint_index; ++i) {
    const loom_low_constraint_t* previous =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (previous->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(previous->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(previous->rhs_operand_index < descriptor->operand_count);
    if (descriptor_operand_packet_indices[previous->rhs_operand_index] ==
        packet_operand_index) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_descriptor_matrix_copy_destructive_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    const uint16_t* descriptor_operand_packet_indices,
    loom_value_id_t* operands) {
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(constraint->lhs_operand_index < descriptor->result_count);
    IREE_ASSERT(constraint->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(constraint->rhs_operand_index < descriptor->operand_count);
    IREE_ASSERT(
        descriptor_operand_packet_indices[constraint->rhs_operand_index] !=
        UINT16_MAX);
    const uint16_t packet_operand_index =
        descriptor_operand_packet_indices[constraint->rhs_operand_index];
    if (loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
            descriptor_set, descriptor, descriptor_operand_packet_indices, i,
            packet_operand_index)) {
      continue;
    }
    const loom_value_id_t source_value = operands[packet_operand_index];
    const loom_type_t copy_type =
        loom_module_value_type(context->module, source_value);
    IREE_ASSERT(loom_low_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), source_value, copy_type,
        source_op->location, &copy_op));
    operands[packet_operand_index] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_vector_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  if (descriptor->result_count != 1) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "descriptor-matrix vector.mma descriptor result "
                            "count is invalid");
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_lhs(source_op), &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_rhs(source_op), &low_rhs));
  loom_value_id_t low_init = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_mma_init(source_op), &low_init));

  const loom_value_id_t result = loom_vector_mma_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, result, &result_low_type));
  IREE_ASSERT(loom_low_type_is_register(result_low_type));

  loom_value_id_t* operands = NULL;
  iree_host_size_t operand_count = 0;
  const uint16_t* descriptor_operand_packet_indices = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_operands(
      context, plan, low_lhs, low_rhs, low_init, &operands, &operand_count,
      &descriptor_operand_packet_indices));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_descriptor_matrix_copy_destructive_operands(
          context, source_op, plan, descriptor_operand_packet_indices,
          operands));
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_tied_results(
      context, plan, descriptor_operand_packet_indices, &tied_results,
      &tied_result_count));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count, plan->attrs,
      &result_low_type, 1, tied_results, tied_result_count, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(
      context, result, loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  switch (plan->source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA:
      return loom_low_lower_emit_descriptor_matrix_vector_mma(context,
                                                              source_op, plan);
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown descriptor-matrix source");
  }
}

static iree_status_t loom_low_lower_emit_selected_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_LT(context->lowering.selected_plan_emit_index,
                 context->lowering.selected_plan_count);
  const loom_low_lower_selected_plan_t selected_plan =
      context->lowering
          .selected_plans[context->lowering.selected_plan_emit_index++];
  IREE_ASSERT_EQ(selected_plan.source_op, source_op);
  if (iree_any_bit_set(selected_plan.flags,
                       LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED)) {
    return loom_low_lower_emit_elided_selected_plan(context, &selected_plan);
  }
  const bool report_allocator_provided =
      !iree_allocator_is_null(context->options->report_allocator);
  loom_block_t* insertion_block = context->builder.ip.block;
  uint32_t before_op_count = 0;
  if (report_allocator_provided) {
    IREE_ASSERT(insertion_block != NULL);
    before_op_count = insertion_block->op_count;
  }
  if (selected_plan.kind == LOOM_LOW_LOWER_SELECTED_PLAN_RULE) {
    IREE_ASSERT(selected_plan.rule_set != NULL);
    IREE_ASSERT(selected_plan.rule != NULL);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_emit_rule(
        context, selected_plan.rule_set, source_op, selected_plan.rule,
        selected_plan.resolved_emits));
  } else if (selected_plan.kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX) {
    IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(selected_plan.plan));
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_matrix_plan(
        context, source_op,
        (const loom_low_lower_descriptor_matrix_plan_t*)
            selected_plan.plan.target_data));
  } else {
    IREE_ASSERT_FALSE(loom_low_lower_plan_is_empty(selected_plan.plan));
    IREE_ASSERT(context->policy->emit_op.fn != NULL);
    IREE_RETURN_IF_ERROR(
        context->policy->emit_op.fn(context->policy->emit_op.user_data, context,
                                    source_op, selected_plan.plan));
  }
  if (report_allocator_provided) {
    const uint32_t after_op_count = insertion_block->op_count;
    IREE_ASSERT_GE(after_op_count, before_op_count);
    IREE_RETURN_IF_ERROR(loom_low_lower_record_report_row(
        context, &selected_plan, after_op_count - before_op_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks) {
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < source_region->block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_region, block_index);
    if (map_source_blocks) {
      loom_builder_set_block(&context->builder,
                             context->lowering.block_map[block_index]);
    } else if (block_index != 0) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "structured source region with multiple blocks reached target-low "
          "emission");
    }
    loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      bool handled = false;
      status = loom_low_lower_structural_op(context, source_op, &handled);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (!handled) {
        if (loom_low_lower_op_is_source_metadata(source_op->kind)) {
          continue;
        }
        status = loom_low_lower_emit_selected_plan(context, source_op);
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
    }
  }
  return status;
}

static iree_status_t loom_low_lower_emit_body(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, source_body, /*map_source_blocks=*/true);
  loom_builder_restore(&context->builder, saved_ip);
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(context->lowering.selected_plan_emit_index,
                   context->lowering.selected_plan_count);
  }
  return status;
}

iree_status_t loom_low_lower_function(loom_module_t* module,
                                      loom_func_like_t source_function,
                                      const loom_low_lower_options_t* options,
                                      loom_low_lower_result_t* out_result) {
  IREE_ASSERT(out_result != NULL);
  loom_low_lower_assert_options(module, source_function, options);
  *out_result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  if (!iree_allocator_is_null(options->report_allocator)) {
    out_result->report_allocator = options->report_allocator;
  }

  loom_region_t* source_body = loom_func_like_body(source_function);
  IREE_ASSERT(source_body != NULL);

  loom_low_lower_context_t context = {
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .result = out_result,
  };
  context.lowering.fact_table = options->fact_table;
  iree_arena_initialize(module->arena.block_pool, &context.arena);

  iree_status_t status =
      loom_low_lowering_frame_initialize_value_ordinals(&context, source_body);
  if (iree_status_is_ok(status)) {
    status = loom_target_contract_index_compose(
        context.policy->contract_bindings,
        context.policy->contract_binding_count, &context.contract_index,
        &context.arena);
  }

  loom_vector_memory_footprint_result_t footprint_result = {0};
  if (iree_status_is_ok(status)) {
    const loom_vector_memory_footprint_options_t footprint_options = {
        .arena = &context.arena,
        .fact_table = context.lowering.fact_table,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_vector_memory_footprint_verify_function(
        module, source_function, &footprint_options, &footprint_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += footprint_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.arena);
    return iree_ok_status();
  }

  loom_kernel_async_legality_result_t async_legality_result = {0};
  if (iree_status_is_ok(status)) {
    loom_kernel_async_legality_options_t async_legality_options = {
        .arena = &context.arena,
        .fact_table = context.lowering.fact_table,
        .value_domain = &context.lowering.value_domain,
        .emitter = options->emitter,
        .phase_name = IREE_SV("source-low"),
    };
    status = loom_kernel_async_legality_verify_function(module, source_function,
                                                        &async_legality_options,
                                                        &async_legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += async_legality_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.arena);
    return iree_ok_status();
  }

  loom_target_low_legality_result_t legality_result = {};
  if (iree_status_is_ok(status)) {
    loom_target_low_legality_options_t legality_options = {
        .bundle = options->bundle,
        .target_data = options->target_data,
        .target_ref = options->target_ref,
        .descriptor_registry = options->descriptor_registry,
        .error_catalog = options->policy->error_catalog,
        .provider_list = options->legality_provider_list,
        .contract_query =
            {
                .fn = loom_low_lower_query_target_contract_from_context,
                .user_data = &context,
            },
        .type_supported = context.policy->source_type_supported,
        .fact_table = context.lowering.fact_table,
        .value_domain = &context.lowering.value_domain,
        .structural_legality_flags =
            loom_low_lower_structured_low_enabled(&context)
                ? LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_SCF
                : 0,
        .diagnostic_flags = options->legality_diagnostic_flags,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_target_low_verify_function_legality(
        module, source_function, &legality_options, &legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count = legality_result.error_count;
    out_result->remark_count = legality_result.remark_count;
    out_result->descriptor_set = legality_result.descriptor_set;
    context.descriptor_set = legality_result.descriptor_set;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    loom_low_lowering_frame_deinitialize(&context);
    iree_arena_deinitialize(&context.arena);
    return iree_ok_status();
  }

  if (iree_status_is_ok(status) &&
      context.lowering.value_domain.value_count != 0) {
    status = iree_arena_allocate_array(
        &context.arena, context.lowering.value_domain.value_count,
        sizeof(*context.lowering.value_map),
        (void**)&context.lowering.value_map);
  }
  if (iree_status_is_ok(status) &&
      context.lowering.value_domain.value_count != 0) {
    status = iree_arena_allocate_array(
        &context.arena, context.lowering.value_domain.value_count,
        sizeof(*context.lowering.value_storage_flags),
        (void**)&context.lowering.value_storage_flags);
  }
  if (iree_status_is_ok(status)) {
    for (loom_value_ordinal_t i = 0;
         i < context.lowering.value_domain.value_count; ++i) {
      context.lowering.value_map[i] = LOOM_VALUE_ID_INVALID;
      context.lowering.value_storage_flags[i] = 0;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_plan_body(&context, source_body);
  }
  if (iree_status_is_ok(status) && context.result->error_count == 0) {
    loom_symbol_ref_t low_func_ref = loom_func_like_callee(source_function);
    status =
        loom_low_lower_create_function_op(&context, source_body, low_func_ref);
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_map_blocks(&context, source_body);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_prepare_branches(&context, source_body);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_low_lower_emit_preamble(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_low_lower_emit_argument_resource_imports(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_low_lower_emit_entry_setup(&context);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_low_lower_emit_body(&context, source_body);
    }
    if (iree_status_is_ok(status) && context.result->error_count != 0 &&
        context.low_func_op != NULL) {
      status = loom_op_erase(module, context.low_func_op);
      context.low_func_op = NULL;
      out_result->low_func_op = NULL;
      out_result->low_func_ref = loom_symbol_ref_null();
    }
    // The replacement low op carries the source symbol while the source op
    // still owns the symbol table entry. Erase clears that entry; relink it to
    // the replacement so callers keep the same symbol identity.
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      status = loom_op_erase(module, source_function.op);
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0) {
      loom_module_link_symbol_defining_op(
          module, context.low_func_op,
          loom_op_vtable(module, context.low_func_op));
    }
    if (iree_status_is_ok(status) && context.result->error_count == 0 &&
        context.lowering.memory_access_record_count != 0) {
      out_result->memory_access_table = (loom_low_memory_access_table_t){
          .function_op = context.low_func_op,
          .values = context.lowering.memory_access_records,
          .count = context.lowering.memory_access_record_count,
      };
    }
  }

  loom_low_lowering_frame_deinitialize(&context);
  iree_arena_deinitialize(&context.arena);
  return status;
}
