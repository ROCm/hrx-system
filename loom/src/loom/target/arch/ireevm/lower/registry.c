// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ir/scalar_type.h"
#include "loom/target/arch/ireevm/contracts/core.h"
#include "loom/target/arch/ireevm/contracts/core_lower_rules.h"
#include "loom/target/arch/ireevm/descriptors/descriptors.h"
#include "loom/target/arch/ireevm/lower/lower.h"
#include "loom/target/arch/ireevm/ops/ops.h"

static iree_string_view_t loom_ireevm_lower_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static bool loom_ireevm_type_is_none(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_NONE;
}

static bool loom_ireevm_type_is_dialect_named(const loom_module_t* module,
                                              loom_type_t type,
                                              iree_string_view_t name,
                                              uint16_t param_count) {
  if (!loom_type_is_dialect(type) ||
      loom_type_dialect_param_count(type) != param_count) {
    return false;
  }
  return iree_string_view_equal(
      loom_ireevm_lower_module_string(module, loom_type_dialect_name_id(type)),
      name);
}

static bool loom_ireevm_type_is_ref(const loom_module_t* module,
                                    loom_type_t type) {
  return loom_ireevm_type_is_dialect_named(module, type, IREE_SV("ireevm.ref"),
                                           1);
}

static bool loom_ireevm_type_is_buffer(const loom_module_t* module,
                                       loom_type_t type) {
  return loom_ireevm_type_is_dialect_named(module, type,
                                           IREE_SV("ireevm.buffer"), 0);
}

static bool loom_ireevm_type_is_buffer_ref(const loom_module_t* module,
                                           loom_type_t type) {
  if (!loom_ireevm_type_is_ref(module, type)) {
    return false;
  }
  const loom_type_t* params = loom_type_dialect_params(type);
  return params && loom_ireevm_type_is_buffer(module, params[0]);
}

static bool loom_ireevm_type_is_scalar(loom_type_t type,
                                       loom_scalar_type_t scalar_type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == scalar_type;
}

static bool loom_ireevm_type_is_supported_scalar(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  loom_scalar_type_t element_type = loom_type_element_type(type);
  switch (element_type) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return true;
    default:
      return false;
  }
}

static bool loom_ireevm_source_type_supported(void* user_data,
                                              const loom_module_t* module,
                                              loom_type_t source_type) {
  (void)user_data;
  return loom_ireevm_type_is_ref(module, source_type);
}

static iree_status_t loom_ireevm_map_type(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_ireevm_type_is_ref(loom_low_lower_context_module(context),
                              source_type)) {
    return loom_low_lower_make_register_type(
        context, IREEVM_CORE_REG_CLASS_ID_REF, 1, out_low_type);
  }
  if (!loom_ireevm_type_is_supported_scalar(source_type)) {
    return loom_low_lower_emit_source_type_unsupported(
        context, source_op, IREE_SV("source"), source_type);
  }
  switch (loom_type_element_type(source_type)) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I32:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_I32, 1, out_low_type);
    case LOOM_SCALAR_TYPE_I64:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_I64, 2, out_low_type);
    case LOOM_SCALAR_TYPE_F32:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_F32, 1, out_low_type);
    case LOOM_SCALAR_TYPE_F64:
      return loom_low_lower_make_register_type(
          context, IREEVM_CORE_REG_CLASS_ID_F64, 2, out_low_type);
    default:
      IREE_ASSERT_UNREACHABLE(
          "checked by loom_ireevm_type_is_supported_scalar");
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL, "unreachable scalar type");
}

typedef struct loom_ireevm_buffer_lower_info_t {
  // Source IREE VM buffer op kind.
  loom_op_kind_t op_kind;
  // IREE VM low descriptor ordinal to emit.
  uint32_t descriptor_ordinal;
  // Load/length scalar result type, or COUNT_ for stores.
  loom_scalar_type_t result_type;
  // Store scalar value type, or COUNT_ for loads/length.
  loom_scalar_type_t store_value_type;
} loom_ireevm_buffer_lower_info_t;

static const loom_ireevm_buffer_lower_info_t kIreeVmBufferLowerInfos[] = {
    {LOOM_OP_IREEVM_BUFFER_LENGTH, IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LENGTH,
     LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_I8_U,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_I8_U, LOOM_SCALAR_TYPE_I32,
     LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_I8_S,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_I8_S, LOOM_SCALAR_TYPE_I32,
     LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_I16_U,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_I16_U, LOOM_SCALAR_TYPE_I32,
     LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_I16_S,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_I16_S, LOOM_SCALAR_TYPE_I32,
     LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_I32, IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_I32,
     LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_I64, IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_I64,
     LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_F32, IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_F32,
     LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_LOAD_F64, IREEVM_CORE_DESCRIPTOR_REF_BUFFER_LOAD_F64,
     LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_COUNT_},
    {LOOM_OP_IREEVM_BUFFER_STORE_I8, IREEVM_CORE_DESCRIPTOR_REF_BUFFER_STORE_I8,
     LOOM_SCALAR_TYPE_COUNT_, LOOM_SCALAR_TYPE_I32},
    {LOOM_OP_IREEVM_BUFFER_STORE_I16,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_STORE_I16, LOOM_SCALAR_TYPE_COUNT_,
     LOOM_SCALAR_TYPE_I32},
    {LOOM_OP_IREEVM_BUFFER_STORE_I32,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_STORE_I32, LOOM_SCALAR_TYPE_COUNT_,
     LOOM_SCALAR_TYPE_I32},
    {LOOM_OP_IREEVM_BUFFER_STORE_I64,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_STORE_I64, LOOM_SCALAR_TYPE_COUNT_,
     LOOM_SCALAR_TYPE_I64},
    {LOOM_OP_IREEVM_BUFFER_STORE_F32,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_STORE_F32, LOOM_SCALAR_TYPE_COUNT_,
     LOOM_SCALAR_TYPE_F32},
    {LOOM_OP_IREEVM_BUFFER_STORE_F64,
     IREEVM_CORE_DESCRIPTOR_REF_BUFFER_STORE_F64, LOOM_SCALAR_TYPE_COUNT_,
     LOOM_SCALAR_TYPE_F64},
};

static const loom_ireevm_buffer_lower_info_t*
loom_ireevm_lookup_buffer_lower_info(loom_op_kind_t op_kind) {
  if (op_kind < LOOM_OP_IREEVM_BUFFER_LENGTH ||
      op_kind > LOOM_OP_IREEVM_BUFFER_STORE_F64) {
    return NULL;
  }
  const iree_host_size_t index =
      (iree_host_size_t)(op_kind - LOOM_OP_IREEVM_BUFFER_LENGTH);
  if (index >= IREE_ARRAYSIZE(kIreeVmBufferLowerInfos)) {
    return NULL;
  }
  const loom_ireevm_buffer_lower_info_t* info = &kIreeVmBufferLowerInfos[index];
  return info->op_kind == op_kind ? info : NULL;
}

static iree_status_t loom_ireevm_select_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  (void)context;
  switch (source_op->kind) {
    case LOOM_OP_IREEVM_REF_RETAIN:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    case LOOM_OP_IREEVM_REF_RELEASE:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    case LOOM_OP_IREEVM_REF_DISCARD:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    default:
      *out_plan = loom_ireevm_lookup_buffer_lower_info(source_op->kind)
                      ? loom_low_lower_plan_make(source_op->kind, NULL)
                      : loom_low_lower_plan_empty();
      return iree_ok_status();
  }
}

static iree_status_t loom_ireevm_resolve_descriptor_ordinal(
    loom_low_lower_context_t* context, uint32_t descriptor_ordinal,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (descriptor_ordinal >= descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "IREE VM descriptor ordinal is out of range");
  }
  return loom_low_lower_resolve_descriptor_row(
      context, &descriptor_set->descriptors[descriptor_ordinal],
      out_descriptor);
}

static iree_status_t loom_ireevm_check_ref_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_value_id_t source_value_id,
    loom_type_t* out_low_type) {
  *out_low_type = loom_type_none();
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (!loom_ireevm_type_is_ref(module, source_type)) {
    return loom_low_lower_emit_source_type_unsupported(context, source_op,
                                                       field_name, source_type);
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_value_id, out_low_type));
  return iree_ok_status();
}

static iree_status_t loom_ireevm_check_buffer_ref_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_value_id_t source_value_id,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (!loom_ireevm_type_is_buffer_ref(module, source_type)) {
    return loom_low_lower_emit_source_type_unsupported(context, source_op,
                                                       field_name, source_type);
  }
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source_value_id, &low_type));
  if (loom_ireevm_type_is_none(low_type)) {
    return iree_ok_status();
  }
  return loom_low_lower_lookup_value(context, source_value_id, out_low_value);
}

static iree_status_t loom_ireevm_check_scalar_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_value_id_t source_value_id,
    loom_scalar_type_t expected_type, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (!loom_ireevm_type_is_scalar(source_type, expected_type)) {
    return loom_low_lower_emit_source_type_unsupported(context, source_op,
                                                       field_name, source_type);
  }
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source_value_id, &low_type));
  if (loom_ireevm_type_is_none(low_type)) {
    return iree_ok_status();
  }
  return loom_low_lower_lookup_value(context, source_value_id, out_low_value);
}

static iree_status_t loom_ireevm_check_scalar_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_value_id_t source_value_id,
    loom_scalar_type_t expected_type, loom_type_t* out_low_type) {
  *out_low_type = loom_type_none();
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (!loom_ireevm_type_is_scalar(source_type, expected_type)) {
    return loom_low_lower_emit_source_type_unsupported(context, source_op,
                                                       field_name, source_type);
  }
  return loom_low_lower_map_value(context, source_op, source_value_id,
                                  out_low_type);
}

static iree_status_t loom_ireevm_emit_ref_retain(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t resource = loom_ireevm_ref_retain_resource(source_op);
  loom_type_t resource_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_ireevm_check_ref_value(
      context, source_op, IREE_SV("resource"), resource, &resource_low_type));
  if (loom_ireevm_type_is_none(resource_low_type)) {
    return iree_ok_status();
  }
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, resource, &low_resource));

  const loom_value_id_t result = loom_ireevm_ref_retain_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_ireevm_check_ref_value(
      context, source_op, IREE_SV("result"), result, &result_low_type));
  if (loom_ireevm_type_is_none(result_low_type)) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_ireevm_resolve_descriptor_ordinal(
      context, IREEVM_CORE_DESCRIPTOR_REF_REF_RETAIN, &descriptor));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, &low_resource, 1, loom_named_attr_slice_empty(),
      &result_low_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, result, loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_ireevm_emit_ref_release(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t resource = loom_ireevm_ref_release_resource(source_op);
  loom_type_t resource_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_ireevm_check_ref_value(
      context, source_op, IREE_SV("resource"), resource, &resource_low_type));
  if (loom_ireevm_type_is_none(resource_low_type)) {
    return iree_ok_status();
  }
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, resource, &low_resource));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_ireevm_resolve_descriptor_ordinal(
      context, IREEVM_CORE_DESCRIPTOR_REF_REF_RELEASE, &descriptor));
  loom_op_t* low_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, &low_resource, 1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op);
}

static iree_status_t loom_ireevm_emit_ref_discard(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t resource = loom_ireevm_ref_discard_resource(source_op);
  loom_type_t resource_low_type = loom_type_none();
  return loom_ireevm_check_ref_value(context, source_op, IREE_SV("resource"),
                                     resource, &resource_low_type);
}

static iree_status_t loom_ireevm_emit_buffer_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_ireevm_buffer_lower_info_t* info) {
  const bool is_store = info->store_value_type != LOOM_SCALAR_TYPE_COUNT_;
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  loom_value_id_t low_operands[3] = {
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  iree_host_size_t low_operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_ireevm_check_buffer_ref_operand(
      context, source_op, IREE_SV("buffer"), source_operands[0],
      &low_operands[low_operand_count]));
  if (low_operands[low_operand_count] == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  ++low_operand_count;

  if (source_op->kind != LOOM_OP_IREEVM_BUFFER_LENGTH) {
    IREE_RETURN_IF_ERROR(loom_ireevm_check_scalar_operand(
        context, source_op, IREE_SV("element_offset"), source_operands[1],
        LOOM_SCALAR_TYPE_I64, &low_operands[low_operand_count]));
    if (low_operands[low_operand_count] == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    ++low_operand_count;
  }

  if (is_store) {
    IREE_RETURN_IF_ERROR(loom_ireevm_check_scalar_operand(
        context, source_op, IREE_SV("value"), source_operands[2],
        info->store_value_type, &low_operands[low_operand_count]));
    if (low_operands[low_operand_count] == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    ++low_operand_count;
  }

  const loom_value_id_t source_result =
      source_op->result_count == 0 ? LOOM_VALUE_ID_INVALID
                                   : loom_op_const_results(source_op)[0];
  loom_type_t result_type = loom_type_none();
  iree_host_size_t result_count = 0;
  if (!is_store) {
    IREE_RETURN_IF_ERROR(loom_ireevm_check_scalar_result(
        context, source_op, IREE_SV("result"), source_result, info->result_type,
        &result_type));
    if (loom_ireevm_type_is_none(result_type)) {
      return iree_ok_status();
    }
    result_count = 1;
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_ireevm_resolve_descriptor_ordinal(
      context, info->descriptor_ordinal, &descriptor));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, low_operands, low_operand_count,
      loom_named_attr_slice_empty(), result_count == 0 ? NULL : &result_type,
      result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  if (result_count == 0) {
    return iree_ok_status();
  }
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_ireevm_emit_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  (void)user_data;
  switch ((loom_op_kind_t)plan.id) {
    case LOOM_OP_IREEVM_REF_RETAIN:
      return loom_ireevm_emit_ref_retain(context, source_op);
    case LOOM_OP_IREEVM_REF_RELEASE:
      return loom_ireevm_emit_ref_release(context, source_op);
    case LOOM_OP_IREEVM_REF_DISCARD:
      return loom_ireevm_emit_ref_discard(context, source_op);
    default: {
      const loom_ireevm_buffer_lower_info_t* info =
          loom_ireevm_lookup_buffer_lower_info((loom_op_kind_t)plan.id);
      if (info != NULL) {
        return loom_ireevm_emit_buffer_op(context, source_op, info);
      }
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unknown IREE VM lowering plan");
}

static const loom_low_lower_rule_set_t* const kIreeVmRuleSets[] = {
    &loom_ireevm_core_lower_rule_set,
};

static const loom_target_contract_binding_t kIreeVmContractBindings[] = {
    {&loom_ireevm_core_contract_fragment, 0},
};

static const loom_low_lower_policy_t kIreeVmLowLowerPolicy = {
    .name = IREE_SVL("iree-vm-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_ireevm_map_type, .user_data = NULL},
    .source_type_supported = {.fn = loom_ireevm_source_type_supported,
                              .user_data = NULL},
    .import_decl_kind = LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM,
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kIreeVmRuleSets),
            .values = kIreeVmRuleSets,
        },
    .contract_bindings = kIreeVmContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kIreeVmContractBindings),
    .select_op = {.fn = loom_ireevm_select_op, .user_data = NULL},
    .emit_op = {.fn = loom_ireevm_emit_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_ireevm_low_lower_policy(void) {
  return &kIreeVmLowLowerPolicy;
}

void loom_ireevm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("ireevm.core"),
          .policy = &kIreeVmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
