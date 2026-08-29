// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/lower.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/arithmetic.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/lower/control.h"
#include "loom/target/arch/vm/lower/conversion.h"
#include "loom/target/arch/vm/lower/initialization.h"
#include "loom/target/arch/vm/lower/kernel.h"
#include "loom/target/arch/vm/lower/memory.h"
#include "loom/target/arch/vm/lower/resources.h"
#include "loom/target/arch/vm/lower/types.h"
#include "loom/target/arch/vm/lower/vector.h"

#define LOOM_VM_SOURCE_LOWERING_LIMITS(max_operand_count, max_result_count) \
  enum {                                                                    \
    LOOM_VM_SOURCE_LOWERING_MAX_OPERAND_COUNT = max_operand_count,          \
    LOOM_VM_SOURCE_LOWERING_MAX_RESULT_COUNT = max_result_count,            \
    LOOM_VM_SOURCE_LOWERING_MAX_TYPE_COUNT =                                \
        max_operand_count + max_result_count,                               \
  };
#define LOOM_VM_SOURCE_LOWERING_ROW(...)
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_SOURCE_LOWERING_ROW
#undef LOOM_VM_SOURCE_LOWERING_LIMITS

typedef struct loom_vm_source_lowering_row_t {
  // Dense Core VM descriptor ordinal selected by this row.
  uint16_t descriptor_ordinal;
  // Descriptor-local selector immediate ordinal, or UINT8_MAX when absent.
  uint8_t selector_immediate_ordinal;
  // Source enum attribute ordinal copied to the selector, or UINT8_MAX.
  uint8_t selector_source_attr_ordinal;
  // Fixed selector value used when no source attribute supplies the value.
  uint8_t selector_value;
  // Exact operand types followed by result types at generated maximum offsets.
  uint16_t type_keys[LOOM_VM_SOURCE_LOWERING_MAX_TYPE_COUNT];
} loom_vm_source_lowering_row_t;

typedef struct loom_vm_source_lowering_range_t {
  // First row in kVmSourceLoweringRows for the source operation.
  uint16_t row_start;
  // Number of concrete type-signature rows for the source operation.
  uint8_t row_count;
} loom_vm_source_lowering_range_t;

typedef struct loom_vm_source_lowering_dialect_ranges_t {
  // Dense ranges indexed by the operation's dialect-local ordinal.
  const loom_vm_source_lowering_range_t* ranges;
  // Number of entries available in ranges.
  uint16_t range_count;
} loom_vm_source_lowering_dialect_ranges_t;

typedef struct loom_vm_source_constant_t {
  // Source SSA result receiving the constant value.
  loom_value_id_t result;
  // Canonical VM value-cell bits; narrow payloads are zero-extended.
  uint64_t bits;
} loom_vm_source_constant_t;

static const loom_vm_source_lowering_row_t kVmSourceLoweringRows[] = {
#define LOOM_VM_SOURCE_TYPE_KEY(type_kind, scalar_type) \
  ((uint16_t)(type_kind) | ((uint16_t)(scalar_type) << 8))
#define LOOM_VM_SOURCE_LOWERING_LIMITS(...)
#define LOOM_VM_SOURCE_LOWERING_ROW(                                \
    source_op_kind, descriptor_ordinal, selector_immediate_ordinal, \
    selector_source_attr_ordinal, selector_value, ...)              \
  {descriptor_ordinal,                                              \
   selector_immediate_ordinal,                                      \
   selector_source_attr_ordinal,                                    \
   selector_value,                                                  \
   {__VA_ARGS__}},
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_SOURCE_LOWERING_ROW
#undef LOOM_VM_SOURCE_LOWERING_LIMITS
#undef LOOM_VM_SOURCE_TYPE_KEY
};

#define LOOM_VM_SOURCE_LOWERING_DEFINE_RANGES
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_SOURCE_LOWERING_DEFINE_RANGES

static uint16_t loom_vm_source_type_key(loom_type_t type) {
  const loom_type_kind_t type_kind = loom_type_kind(type);
  const uint8_t scalar_type = loom_type_is_scalar(type)
                                  ? loom_type_element_type(type)
                                  : LOOM_SCALAR_TYPE_NONE;
  return (uint16_t)type_kind | ((uint16_t)scalar_type << 8);
}

static bool loom_vm_source_lowering_row_matches(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_vm_source_lowering_row_t* row) {
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, operands[i]);
    if (loom_vm_source_type_key(type) != row->type_keys[i]) {
      return false;
    }
  }
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, results[i]);
    if (loom_vm_source_type_key(type) !=
        row->type_keys[LOOM_VM_SOURCE_LOWERING_MAX_OPERAND_COUNT + i]) {
      return false;
    }
  }
  return true;
}

static bool loom_vm_try_get_source_constant(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_vm_source_constant_t* out_constant) {
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  loom_attribute_t value = loom_attr_absent();
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      result = loom_index_constant_result(source_op);
      value = loom_index_constant_value(source_op);
      break;
    case LOOM_OP_SCALAR_CONSTANT:
      result = loom_scalar_constant_result(source_op);
      value = loom_scalar_constant_value(source_op);
      break;
    default:
      return false;
  }
  const loom_type_t result_type = loom_module_value_type(module, result);
  *out_constant = (loom_vm_source_constant_t){
      .result = result,
      .bits = loom_vm_constant_bits_from_scalar_attr(
          loom_type_element_type(result_type), value),
  };
  return true;
}

static uint16_t loom_vm_register_class_for_abi_bank(
    loom_vm_call_abi_bank_t bank) {
  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      return VM_CORE_REG_CLASS_ID_VALUE;
    case LOOM_VM_CALL_ABI_BANK_REF:
      return VM_CORE_REG_CLASS_ID_REF;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      return VM_CORE_REG_CLASS_ID_FUNCTION;
    default:
      IREE_ASSERT_UNREACHABLE("classified VM call ABI bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_vm_map_type(void* user_data,
                                      loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type) {
  (void)user_data;
  loom_vm_call_abi_register_layout_t register_layout = {0};
  if (loom_vm_call_abi_try_classify_logical_type(
          loom_low_lower_context_module(context), source_type,
          &register_layout)) {
    return loom_low_lower_make_typed_register_type(
        context, loom_vm_register_class_for_abi_bank(register_layout.bank),
        register_layout.unit_count, source_type, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static bool loom_vm_source_type_supported(void* user_data,
                                          const loom_module_t* module,
                                          loom_type_t source_type) {
  (void)user_data;
  loom_vm_call_abi_register_layout_t register_layout = {0};
  return loom_vm_call_abi_try_classify_logical_type(module, source_type,
                                                    &register_layout);
}

static bool loom_vm_source_function_has_presentation(
    const loom_module_t* module, loom_func_like_t function) {
  if (loom_func_like_import_module(function) != LOOM_STRING_ID_INVALID) {
    return true;
  }
  const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
  if (!loom_symbol_ref_is_valid(function_ref) || function_ref.module_id != 0 ||
      function_ref.symbol_id >= module->symbols.count) {
    return false;
  }
  return !iree_string_view_is_empty(loom_func_like_export_name(
      module, &module->symbols.entries[function_ref.symbol_id], function));
}

static const loom_value_id_t* loom_vm_source_function_argument_values(
    loom_low_lower_context_t* context, iree_host_size_t expected_count) {
  uint16_t source_count = 0;
  const loom_value_id_t* source_values = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &source_count);
  IREE_ASSERT_EQ(source_count, expected_count);
  return source_values;
}

static iree_status_t loom_vm_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind,
    const loom_type_t* argument_types, iree_host_size_t argument_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_named_attr_slice_t* out_abi_layout) {
  (void)user_data;
  *out_abi_layout = loom_named_attr_slice_empty();
  const loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const bool has_presentation =
      loom_vm_source_function_has_presentation(module, source_function);
  const loom_value_id_t* argument_values =
      has_presentation
          ? loom_vm_source_function_argument_values(context, argument_count)
          : NULL;
  if (layout_kind == LOOM_LOW_LOWER_ABI_LAYOUT_KIND_KERNEL) {
    IREE_ASSERT_EQ(result_count, 0u);
    return loom_vm_kernel_map_abi_layout(context, argument_types,
                                         argument_values, argument_count,
                                         out_abi_layout);
  }
  IREE_ASSERT_EQ(layout_kind, LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC);
  const loom_value_id_t* result_values =
      has_presentation ? loom_op_const_results(source_function.op) : NULL;
  IREE_ASSERT(!has_presentation ||
              source_function.op->result_count == result_count);
  loom_attribute_t layout_attr = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_attr(
      loom_low_lower_context_module(context),
      (loom_vm_call_abi_source_fields_t){
          .types = argument_types,
          .values = argument_values,
          .count = argument_count,
      },
      (loom_vm_call_abi_source_fields_t){
          .types = result_types,
          .values = result_values,
          .count = result_count,
      },
      loom_type_none(), loom_low_lower_context_function_arena(context),
      &layout_attr));
  *out_abi_layout = loom_attr_as_dict(layout_attr);
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_preamble(void* user_data,
                                           loom_low_lower_context_t* context) {
  (void)user_data;
  if (loom_kernel_def_isa(loom_low_lower_context_source_function(context).op)) {
    return loom_vm_kernel_emit_preamble(context);
  }
  if (loom_func_like_cc(loom_low_lower_context_source_function(context)) !=
      LOOM_FUNC_CC_INITIALIZER) {
    return iree_ok_status();
  }
  return loom_vm_module_resources_emit_initializer_preamble(context);
}

static iree_status_t loom_vm_select_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  if (loom_vm_kernel_try_select_op(source_op, out_plan)) {
    return iree_ok_status();
  }
  if (loom_vm_arithmetic_try_select_op(source_op, out_plan)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_vm_source_constant_t constant = {0};
  if (loom_vm_try_get_source_constant(module, source_op, &constant)) {
    *out_plan = loom_low_lower_plan_make(
        loom_vm_constant_descriptor_ordinal(constant.bits), NULL);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_vm_module_resource_try_select_op(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_vm_memory_try_select_op(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) return iree_ok_status();
  if (loom_vm_vector_try_select_op(module, source_op, out_plan)) {
    return iree_ok_status();
  }

  const uint8_t dialect_id = loom_op_dialect_id(source_op->kind);
  if (dialect_id < IREE_ARRAYSIZE(kVmSourceLoweringDialectRanges)) {
    const loom_vm_source_lowering_dialect_ranges_t* dialect_ranges =
        &kVmSourceLoweringDialectRanges[dialect_id];
    const uint8_t op_index = loom_op_dialect_index(source_op->kind);
    if (op_index < dialect_ranges->range_count) {
      const loom_vm_source_lowering_range_t range =
          dialect_ranges->ranges[op_index];
      for (uint8_t i = 0; i < range.row_count; ++i) {
        const loom_vm_source_lowering_row_t* row =
            &kVmSourceLoweringRows[range.row_start + i];
        if (loom_vm_source_lowering_row_matches(module, source_op, row)) {
          *out_plan = loom_low_lower_plan_make(row->descriptor_ordinal, row);
          return iree_ok_status();
        }
      }
    }
  }
  if (loom_vm_conversion_try_select_op(module, source_op, out_plan)) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

static void loom_vm_mark_plan_storage_demands(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_vm_memory_mark_plan_storage_demands(context, source_op, plan)) {
    return;
  }
  loom_low_lower_require_source_operands_storage(context, source_op);
}

static iree_status_t loom_vm_emit_constant(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t plan) {
  loom_vm_source_constant_t constant = {0};
  const bool is_constant = loom_vm_try_get_source_constant(
      loom_low_lower_context_module(context), source_op, &constant);
  IREE_ASSERT(is_constant);
  (void)is_constant;
  IREE_ASSERT_EQ(plan.id, loom_vm_constant_descriptor_ordinal(constant.bits));

  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, constant.result, &low_result_type));
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_inline_constant_build(
      loom_low_lower_context_builder(context), constant.bits, low_result_type,
      source_op->location, &low_value));
  return loom_low_lower_bind_value(context, constant.result, low_value);
}

static iree_status_t loom_vm_emit_op(void* user_data,
                                     loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan) {
  (void)user_data;
  bool kernel_handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vm_kernel_emit_op(context, source_op, plan, &kernel_handled));
  if (kernel_handled) return iree_ok_status();

  bool resource_handled = false;
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_emit_op(context, source_op, plan,
                                                       &resource_handled));
  if (resource_handled) return iree_ok_status();

  bool arithmetic_handled = false;
  IREE_RETURN_IF_ERROR(loom_vm_arithmetic_emit_op(context, source_op, plan,
                                                  &arithmetic_handled));
  if (arithmetic_handled) return iree_ok_status();

  bool conversion_handled = false;
  IREE_RETURN_IF_ERROR(loom_vm_conversion_emit_op(context, source_op, plan,
                                                  &conversion_handled));
  if (conversion_handled) return iree_ok_status();

  bool memory_handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vm_memory_emit_op(context, source_op, plan, &memory_handled));
  if (memory_handled) return iree_ok_status();

  bool vector_handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vm_vector_emit_op(context, source_op, plan, &vector_handled));
  if (vector_handled) return iree_ok_status();

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, (uint32_t)plan.id);
  IREE_ASSERT(descriptor != NULL);
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  if (descriptor->op_kind == LOOM_LOW_DESCRIPTOR_OP_KIND_CONST) {
    return loom_vm_emit_constant(context, source_op, plan);
  }

  loom_value_id_t low_operands[LOOM_VM_SOURCE_LOWERING_MAX_OPERAND_COUNT];
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, source_operands[i], &low_operands[i]));
  }

  loom_type_t low_result_types[LOOM_VM_SOURCE_LOWERING_MAX_RESULT_COUNT];
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
        context, source_op, source_results[i], &low_result_types[i]));
  }

  const loom_vm_source_lowering_row_t* row = plan.target_data;
  IREE_ASSERT(row != NULL);
  loom_named_attr_t selector_attr = {0};
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (row->selector_immediate_ordinal != UINT8_MAX) {
    IREE_ASSERT_LT(row->selector_immediate_ordinal,
                   descriptor->immediate_count);
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start +
                                    row->selector_immediate_ordinal];
    const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    IREE_RETURN_IF_ERROR(
        loom_builder_intern_string(loom_low_lower_context_builder(context),
                                   immediate_name, &selector_attr.name_id));
    const int64_t selector_value =
        row->selector_source_attr_ordinal == UINT8_MAX
            ? row->selector_value
            : loom_attr_as_enum(loom_op_const_attrs(
                  source_op)[row->selector_source_attr_ordinal]);
    selector_attr.value = loom_attr_i64(selector_value);
    attrs = loom_make_named_attr_slice(&selector_attr, 1);
  }

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_descriptor, low_operands, source_op->operand_count,
      attrs, low_result_types, source_op->result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  const loom_value_id_t* low_results = loom_op_const_results(low_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_finalize_module(
    void* user_data, loom_module_t* module,
    loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  (void)user_data;
  return loom_vm_module_resources_finalize(module, module_state, scratch_arena);
}

static iree_status_t loom_vm_prepare_module(
    void* user_data, loom_module_t* module,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_low_lower_prepare_module_result_t* out_result) {
  (void)user_data;
  return loom_vm_materialize_initializer(module, diagnostic_emitter,
                                         scratch_arena, out_result);
}

static const loom_low_lower_policy_t kVmCoreLowLowerPolicy = {
    .name = IREE_SVL("vm-core-low-lower"),
    .flags = LOOM_LOW_LOWER_POLICY_FLAG_MODULE_IMPORTS,
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_vm_map_type, .user_data = NULL},
    .source_type_supported = {.fn = loom_vm_source_type_supported,
                              .user_data = NULL},
    .emit_preamble = {.fn = loom_vm_emit_preamble, .user_data = NULL},
    .map_abi_layout = {.fn = loom_vm_map_abi_layout, .user_data = NULL},
    .switch_lowering =
        {
            .can_emit = loom_vm_switch_lowering_can_emit,
            .emit = loom_vm_switch_lowering_emit,
            .user_data = NULL,
        },
    .select_op = {.fn = loom_vm_select_op, .user_data = NULL},
    .mark_plan_storage_demands = {.fn = loom_vm_mark_plan_storage_demands,
                                  .user_data = NULL},
    .emit_op = {.fn = loom_vm_emit_op, .user_data = NULL},
    .prepare_module = {.fn = loom_vm_prepare_module, .user_data = NULL},
    .finalize_module = {.fn = loom_vm_finalize_module, .user_data = NULL},
};

void loom_vm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("vm.core"),
          .policy = &kVmCoreLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
