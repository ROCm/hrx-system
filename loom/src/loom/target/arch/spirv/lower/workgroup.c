// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/workgroup.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/util/fact_table.h"

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
  // Register class used for the Workgroup array base pointer.
  uint16_t array_pointer_reg_class_id;
} loom_spirv_workgroup_view_plan_t;

static bool loom_spirv_workgroup_i64_is_power_of_two(int64_t value) {
  return value > 0 && value <= UINT32_MAX &&
         iree_math_is_power_of_two_i64(value);
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

static bool loom_spirv_workgroup_view_scalar_type(
    loom_type_t view_type, loom_spirv_scalar_type_t* out_scalar_type) {
  *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  loom_spirv_value_type_t value_type = {0};
  if (!loom_spirv_value_type_from_loom_type(
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

static bool loom_spirv_workgroup_view_plan_from_facts(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_spirv_workgroup_view_plan_t* out_plan, bool* out_has_workgroup_facts) {
  *out_plan = (loom_spirv_workgroup_view_plan_t){0};
  *out_has_workgroup_facts = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_value_id_t result = loom_buffer_view_result(source_op);
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, result);
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(&fact_table->context, facts,
                                             &view_reference) ||
      view_reference.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  *out_has_workgroup_facts = true;
  if (!loom_spirv_workgroup_view_root_is_alloca(module,
                                                view_reference.root_value_id)) {
    return false;
  }

  loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  const loom_type_t view_type = loom_module_value_type(module, result);
  if (!loom_spirv_workgroup_view_scalar_type(view_type, &scalar_type)) {
    return false;
  }

  out_plan->root_value_id = view_reference.root_value_id;
  out_plan->array_pointer_reg_class_id =
      loom_spirv_ptr_workgroup_array_reg_class_id(scalar_type);
  return out_plan->array_pointer_reg_class_id != LOOM_LOW_REG_CLASS_NONE;
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
  if (!loom_value_facts_as_non_negative_i64_maximum(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(source_op)),
          &plan.byte_length) ||
      plan.byte_length <= 0 ||
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
  bool has_workgroup_facts = false;
  const bool selected = loom_spirv_workgroup_view_plan_from_facts(
      context, source_op, &plan, &has_workgroup_facts);
  if (!selected) {
    if (has_workgroup_facts) {
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

void loom_spirv_mark_workgroup_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  (void)source_op;
  switch ((loom_spirv_workgroup_plan_kind_t)plan.id) {
    case LOOM_SPIRV_WORKGROUP_PLAN_ALLOCA:
      return;
    case LOOM_SPIRV_WORKGROUP_PLAN_VIEW: {
      const loom_spirv_workgroup_view_plan_t* view_plan =
          (const loom_spirv_workgroup_view_plan_t*)plan.target_data;
      loom_low_lower_require_source_value_storage(context,
                                                  view_plan->root_value_id);
      return;
    }
  }
  IREE_ASSERT_UNREACHABLE("SPIR-V Workgroup plan selected unknown op kind");
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
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   loom_low_storage_address_result(address_op));
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
