// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/workgroup.h"

#include <stdint.h>

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/abi.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"

typedef enum loom_spirv_workgroup_plan_kind_e {
  LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA = 1,
  LOOM_SPIRV_WORKGROUP_PLAN_VIEW = 2,
} loom_spirv_workgroup_plan_kind_t;

typedef struct loom_spirv_workgroup_alloca_plan_t {
  // Static byte length of the Workgroup allocation.
  int64_t byte_length;
  // Static byte alignment of the Workgroup allocation.
  int64_t byte_alignment;
} loom_spirv_workgroup_alloca_plan_t;

typedef struct loom_spirv_workgroup_view_plan_t {
  // Source storage-root value that has already lowered to low.storage.
  loom_value_id_t root_value_id;
  // Optional source dynamic element index, or invalid for static-only views.
  loom_value_id_t dynamic_index_value_id;
  // Static element addend folded from the source byte offset.
  int64_t static_element_index;
  // SPIR-V scalar element type addressed by the view.
  loom_spirv_scalar_type_t scalar_type;
  // Register class used for the Workgroup array base pointer.
  uint16_t array_pointer_reg_class_id;
  // Register class used for the Workgroup scalar element pointer.
  uint16_t element_pointer_reg_class_id;
  // Descriptor ref for the typed OpAccessChain row.
  uint32_t access_chain_descriptor_ref;
} loom_spirv_workgroup_view_plan_t;

static bool loom_spirv_workgroup_i64_is_power_of_two(int64_t value) {
  return value > 0 && value <= UINT32_MAX && loom_is_power_of_two_i64(value);
}

static bool loom_spirv_workgroup_exact_positive_i64(loom_value_facts_t facts,
                                                    int64_t* out_value) {
  *out_value = 0;
  if (!loom_value_facts_as_exact_i64(facts, out_value) || *out_value <= 0) {
    return false;
  }
  return true;
}

static iree_status_t loom_spirv_workgroup_emit_rejected(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_type_t type,
    loom_spirv_workgroup_plan_kind_t plan_kind,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_make(plan_kind, NULL);
  return loom_low_lower_emit_source_type_unsupported(context, source_op,
                                                     field_name, type);
}

static uint32_t loom_spirv_workgroup_access_chain_descriptor_ref(
    loom_spirv_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SPIRV_SCALAR_TYPE_F16:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_F16_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_F32:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_F32_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_F64:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_F64_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_BF16:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_BF16_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_S8:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_I8_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_S16:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_I16_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_S32:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_I32_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_S64:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_I64_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_U8:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_U8_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_U16:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_U16_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_U32:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_U32_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_U64:
      return SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_ACCESS_CHAIN_WORKGROUP_U64_ELEMENT_INDEX;
    case LOOM_SPIRV_SCALAR_TYPE_UNKNOWN:
      return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  }
  return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

static bool loom_spirv_workgroup_source_index_type_supported(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32);
}

static bool loom_spirv_workgroup_view_scalar_type(
    loom_type_t view_type, loom_spirv_scalar_type_t* out_scalar_type) {
  *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  loom_spirv_value_type_t value_type = {0};
  if (!loom_spirv_abi_value_type_from_source_type(
          loom_type_scalar(loom_type_element_type(view_type)), &value_type) ||
      value_type.value_class != LOOM_SPIRV_VALUE_CLASS_SCALAR) {
    return false;
  }
  *out_scalar_type = value_type.scalar_type;
  return true;
}

static bool loom_spirv_workgroup_view_root_is_alloca(
    const loom_module_t* module, loom_value_id_t root_value_id) {
  if (root_value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* root = loom_module_value(module, root_value_id);
  return !loom_value_is_block_arg(root) &&
         loom_buffer_alloca_isa(loom_value_def_op(root));
}

static bool loom_spirv_workgroup_view_plan_from_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_spirv_workgroup_view_plan_t* out_plan) {
  *out_plan = (loom_spirv_workgroup_view_plan_t){
      .dynamic_index_value_id = LOOM_VALUE_ID_INVALID,
  };
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_value_id_t result = loom_buffer_view_result(source_op);
  loom_low_source_memory_access_plan_t source_plan = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  const loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_low_source_memory_access_plan_build_view(
          module, fact_table, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, result,
          cache_policy, &source_plan, &diagnostic)) {
    return false;
  }
  if (source_plan.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  if (!loom_spirv_workgroup_view_root_is_alloca(module,
                                                source_plan.root_value_id)) {
    return false;
  }

  loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  const loom_type_t view_type = loom_module_value_type(module, result);
  if (!loom_spirv_workgroup_view_scalar_type(view_type, &scalar_type)) {
    return false;
  }
  const loom_spirv_scalar_type_descriptor_t* scalar_descriptor =
      loom_spirv_scalar_type_descriptor(scalar_type);
  if (scalar_descriptor == NULL || (scalar_descriptor->bit_width % 8) != 0) {
    return false;
  }
  const int64_t element_byte_count = scalar_descriptor->bit_width / 8;
  if (source_plan.element_byte_count != element_byte_count ||
      source_plan.vector_lane_count != 1 ||
      source_plan.vector_lane_byte_stride != element_byte_count ||
      source_plan.static_byte_offset < 0 ||
      (source_plan.static_byte_offset % element_byte_count) != 0) {
    return false;
  }
  const int64_t static_element_index =
      source_plan.static_byte_offset / element_byte_count;
  if (static_element_index > INT32_MAX || source_plan.dynamic_term_count > 1) {
    return false;
  }
  if (source_plan.dynamic_term_count == 1) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source_plan.dynamic_terms[0];
    if (term->stride_value_count != 0 ||
        term->byte_stride != element_byte_count ||
        !loom_spirv_workgroup_source_index_type_supported(
            loom_module_value_type(module, term->index))) {
      return false;
    }
    out_plan->dynamic_index_value_id = term->index;
  }

  out_plan->root_value_id = source_plan.root_value_id;
  out_plan->static_element_index = static_element_index;
  out_plan->scalar_type = scalar_type;
  out_plan->array_pointer_reg_class_id =
      loom_spirv_ptr_workgroup_array_reg_class_id(scalar_type);
  out_plan->element_pointer_reg_class_id =
      loom_spirv_ptr_workgroup_reg_class_id(scalar_type);
  out_plan->access_chain_descriptor_ref =
      loom_spirv_workgroup_access_chain_descriptor_ref(scalar_type);
  return out_plan->array_pointer_reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
         out_plan->element_pointer_reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
         out_plan->access_chain_descriptor_ref !=
             LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

static bool loom_spirv_workgroup_view_has_workgroup_facts(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_value_facts_t facts = loom_value_fact_table_lookup(
      fact_table, loom_buffer_view_result(source_op));
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(&fact_table->context, facts,
                                             &view_reference)) {
    return false;
  }
  return view_reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
}

static iree_status_t loom_spirv_select_workgroup_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  if (loom_buffer_alloca_memory_space(source_op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  loom_spirv_workgroup_alloca_plan_t plan = {0};
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (!loom_spirv_workgroup_exact_positive_i64(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(source_op)),
          &plan.byte_length) ||
      !loom_spirv_workgroup_i64_is_power_of_two(
          loom_buffer_alloca_base_alignment(source_op))) {
    return loom_spirv_workgroup_emit_rejected(
        context, source_op, IREE_SV("workgroup_storage"),
        loom_module_value_type(loom_low_lower_context_module(context),
                               loom_buffer_alloca_result(source_op)),
        LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA, out_plan);
  }
  plan.byte_alignment = loom_buffer_alloca_base_alignment(source_op);

  loom_spirv_workgroup_alloca_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  *plan_data = plan;
  *out_plan =
      loom_low_lower_plan_make(LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA, plan_data);
  return iree_ok_status();
}

static iree_status_t loom_spirv_select_workgroup_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  loom_spirv_workgroup_view_plan_t plan = {0};
  if (!loom_spirv_workgroup_view_plan_from_source(context, source_op, &plan)) {
    if (loom_spirv_workgroup_view_has_workgroup_facts(context, source_op)) {
      return loom_spirv_workgroup_emit_rejected(
          context, source_op, IREE_SV("workgroup_view"),
          loom_module_value_type(loom_low_lower_context_module(context),
                                 loom_buffer_view_result(source_op)),
          LOOM_SPIRV_WORKGROUP_PLAN_VIEW, out_plan);
    }
    return iree_ok_status();
  }
  loom_spirv_workgroup_view_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  *plan_data = plan;
  *out_plan =
      loom_low_lower_plan_make(LOOM_SPIRV_WORKGROUP_PLAN_VIEW, plan_data);
  return iree_ok_status();
}

iree_status_t loom_spirv_select_workgroup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_spirv_select_workgroup_alloca(context, source_op, out_plan);
    case LOOM_OP_BUFFER_VIEW:
      return loom_spirv_select_workgroup_view(context, source_op, out_plan);
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_spirv_resolve_descriptor_ref(
    loom_low_lower_context_t* context, uint32_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          loom_spirv_logical_core_descriptor_set(), descriptor_ref);
  return loom_low_lower_resolve_descriptor_row(context, descriptor,
                                               out_descriptor);
}

static iree_status_t loom_spirv_emit_i32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    int64_t value, loom_value_id_t* out_value_id) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_resolve_descriptor_ref(
      context, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_I32, &descriptor));
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("i32_value"), &value_name_id));
  const loom_named_attr_t attrs[] = {
      {
          .name_id = value_name_id,
          .value = loom_attr_i64(value),
      },
  };
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1, &result_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), result_type,
      source_op->location, &low_op));
  *out_value_id = loom_low_const_result(low_op);
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_i32_add(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t lhs,
                                             loom_value_id_t rhs,
                                             loom_value_id_t* out_value_id) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_resolve_descriptor_ref(
      context, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IADD_I32, &descriptor));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1, &result_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op));
  *out_value_id = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_workgroup_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_view_plan_t* plan,
    loom_value_id_t* out_index_value_id) {
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  if (plan->dynamic_index_value_id != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->dynamic_index_value_id, &dynamic_index));
  }
  if (dynamic_index == LOOM_VALUE_ID_INVALID) {
    return loom_spirv_emit_i32_constant(
        context, source_op, plan->static_element_index, out_index_value_id);
  }
  if (plan->static_element_index == 0) {
    *out_index_value_id = dynamic_index;
    return iree_ok_status();
  }
  loom_value_id_t static_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(
      context, source_op, plan->static_element_index, &static_index));
  return loom_spirv_emit_i32_add(context, source_op, dynamic_index,
                                 static_index, out_index_value_id);
}

static iree_status_t loom_spirv_lower_workgroup_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_alloca_plan_t* plan) {
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      loom_low_lower_context_builder(context), plan->byte_length,
      plan->byte_alignment, loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP),
      source_op->location, &storage_op));
  return loom_low_lower_bind_value(
      context, loom_buffer_alloca_result(source_op),
      loom_low_storage_reserve_storage(storage_op));
}

static iree_status_t loom_spirv_lower_workgroup_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_view_plan_t* plan) {
  loom_value_id_t low_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->root_value_id, &low_storage));
  loom_type_t array_pointer_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, plan->array_pointer_reg_class_id, 1, &array_pointer_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      loom_low_lower_context_builder(context), low_storage, /*offset=*/0,
      array_pointer_type, source_op->location, &address_op));

  loom_value_id_t index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_workgroup_index(context, source_op, plan, &index));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_resolve_descriptor_ref(
      context, plan->access_chain_descriptor_ref, &descriptor));
  loom_type_t element_pointer_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, plan->element_pointer_reg_class_id, 1, &element_pointer_type));
  const loom_value_id_t operands[] = {
      loom_low_storage_address_result(address_op),
      index,
  };
  loom_op_t* access_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &element_pointer_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &access_op));
  return loom_low_lower_bind_value(
      context, loom_buffer_view_result(source_op),
      loom_value_slice_get(loom_low_op_results(access_op), 0));
}

iree_status_t loom_spirv_lower_workgroup_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t plan) {
  switch ((loom_spirv_workgroup_plan_kind_t)plan.id) {
    case LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA:
      return loom_spirv_lower_workgroup_alloca(
          context, source_op,
          (const loom_spirv_workgroup_alloca_plan_t*)plan.target_data);
    case LOOM_SPIRV_WORKGROUP_PLAN_VIEW:
      return loom_spirv_lower_workgroup_view(
          context, source_op,
          (const loom_spirv_workgroup_view_plan_t*)plan.target_data);
  }
  IREE_ASSERT_UNREACHABLE("SPIR-V Workgroup plan selected unknown op kind");
  IREE_BUILTIN_UNREACHABLE();
}
