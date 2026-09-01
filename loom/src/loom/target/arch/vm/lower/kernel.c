// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/kernel.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/arithmetic.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/records/target_records.h"
#include "loom/util/fact_table.h"

typedef uint8_t loom_vm_kernel_plan_kind_t;
enum loom_vm_kernel_plan_kind_e {
  LOOM_VM_KERNEL_PLAN_KIND_NONE = 0,
  LOOM_VM_KERNEL_PLAN_KIND_CONSTANT_ZERO = 1,
  LOOM_VM_KERNEL_PLAN_KIND_CONSTANT_ONE = 2,
  LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_ID = 3,
  LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_SIZE = 4,
  LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_COUNT = 5,
  LOOM_VM_KERNEL_PLAN_KIND_WORKITEM_ID = 6,
  LOOM_VM_KERNEL_PLAN_KIND_WORKITEM_DISPATCH_ID = 7,
  LOOM_VM_KERNEL_PLAN_KIND_SUBGROUP_ID = 8,
  LOOM_VM_KERNEL_PLAN_KIND_SUBGROUP_COUNT = 9,
  LOOM_VM_KERNEL_PLAN_KIND_NOOP = 10,
};

#define LOOM_VM_KERNEL_OP_INDEX(op_kind) ((op_kind) & 0xFFu)

// Scalar-profile behavior indexed by the dense kernel dialect op ordinal.
static const loom_vm_kernel_plan_kind_t
    kLoomVmKernelPlanKinds[LOOM_OP_KERNEL_COUNT_] = {
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_BARRIER)] =
            LOOM_VM_KERNEL_PLAN_KIND_NOOP,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_ASYNC_GROUP)] =
            LOOM_VM_KERNEL_PLAN_KIND_NOOP,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_ASYNC_WAIT)] =
            LOOM_VM_KERNEL_PLAN_KIND_NOOP,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_ID)] =
            LOOM_VM_KERNEL_PLAN_KIND_WORKITEM_ID,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_ID)] =
            LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_ID,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_SIZE)] =
            LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_SIZE,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_COUNT)] =
            LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_COUNT,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID)] =
            LOOM_VM_KERNEL_PLAN_KIND_WORKITEM_DISPATCH_ID,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_ID)] =
            LOOM_VM_KERNEL_PLAN_KIND_SUBGROUP_ID,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_COUNT)] =
            LOOM_VM_KERNEL_PLAN_KIND_SUBGROUP_COUNT,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SIZE)] =
            LOOM_VM_KERNEL_PLAN_KIND_CONSTANT_ONE,
        [LOOM_VM_KERNEL_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_LANE_ID)] =
            LOOM_VM_KERNEL_PLAN_KIND_CONSTANT_ZERO,
};

#undef LOOM_VM_KERNEL_OP_INDEX

enum {
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_ID_X = 0,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_ID_Y = 1,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_ID_Z = 2,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_X = 3,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_Y = 4,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_Z = 5,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_COUNT_X = 6,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_COUNT_Y = 7,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_COUNT_Z = 8,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_X = 9,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_Y = 10,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_Z = 11,
  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT = 12,
};

static const iree_string_view_t kLoomVmKernelLaunchArgumentNames[] = {
    IREE_SVL("workgroup_id_x"),    IREE_SVL("workgroup_id_y"),
    IREE_SVL("workgroup_id_z"),    IREE_SVL("workgroup_size_x"),
    IREE_SVL("workgroup_size_y"),  IREE_SVL("workgroup_size_z"),
    IREE_SVL("workgroup_count_x"), IREE_SVL("workgroup_count_y"),
    IREE_SVL("workgroup_count_z"), IREE_SVL("workitem_id_x"),
    IREE_SVL("workitem_id_y"),     IREE_SVL("workitem_id_z"),
};
static_assert(IREE_ARRAYSIZE(kLoomVmKernelLaunchArgumentNames) ==
                  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT,
              "VM kernel launch arguments must all have names");

// Callback plan IDs occupy a target-local domain outside descriptor ordinals.
static const uint64_t kLoomVmKernelPlanIdBase = UINT64_C(0x100000000);

static loom_vm_kernel_plan_kind_t loom_vm_kernel_plan_kind_for_op(
    const loom_op_t* source_op) {
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_KERNEL) {
    return LOOM_VM_KERNEL_PLAN_KIND_NONE;
  }
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  return op_index < IREE_ARRAYSIZE(kLoomVmKernelPlanKinds)
             ? kLoomVmKernelPlanKinds[op_index]
             : LOOM_VM_KERNEL_PLAN_KIND_NONE;
}

bool loom_vm_kernel_try_select_op(const loom_op_t* source_op,
                                  loom_low_lower_plan_t* out_plan) {
  const loom_vm_kernel_plan_kind_t plan_kind =
      loom_vm_kernel_plan_kind_for_op(source_op);
  if (plan_kind == LOOM_VM_KERNEL_PLAN_KIND_NONE) return false;
  *out_plan = loom_low_lower_plan_make(
      kLoomVmKernelPlanIdBase + loom_op_dialect_index(source_op->kind), NULL);
  return true;
}

static bool loom_vm_kernel_plan_is_selected(loom_low_lower_plan_t plan) {
  return plan.id >= kLoomVmKernelPlanIdBase &&
         plan.id <
             kLoomVmKernelPlanIdBase + IREE_ARRAYSIZE(kLoomVmKernelPlanKinds);
}

static loom_vm_kernel_plan_kind_t loom_vm_kernel_plan_kind_for_plan(
    loom_low_lower_plan_t plan) {
  IREE_ASSERT(loom_vm_kernel_plan_is_selected(plan));
  return kLoomVmKernelPlanKinds[plan.id - kLoomVmKernelPlanIdBase];
}

static loom_value_id_t loom_vm_kernel_query_result(const loom_op_t* op) {
  IREE_ASSERT_EQ(op->result_count, 1);
  return loom_op_const_results(op)[0];
}

static loom_kernel_dimension_t loom_vm_kernel_query_dimension(
    const loom_op_t* op) {
  IREE_ASSERT_GE(op->attribute_count, 1);
  return (loom_kernel_dimension_t)loom_attr_as_enum(loom_op_const_attrs(op)[0]);
}

static iree_status_t loom_vm_kernel_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_vm_target_bundle_is_core(
          loom_target_low_legality_bundle(context))) {
    return iree_ok_status();
  }
  if (loom_vm_kernel_plan_kind_for_op(source_op) ==
      LOOM_VM_KERNEL_PLAN_KIND_NOOP) {
    *out_handled = true;
  }
  return iree_ok_status();
}

const loom_target_low_legality_provider_t loom_vm_kernel_low_legality_provider =
    {
        .name = IREE_SVL("vm-scalar-kernel-profile"),
        .builtin_dialect_bits = UINT64_C(1) << LOOM_DIALECT_KERNEL,
        .try_verify_op = loom_vm_kernel_try_verify_op,
};

static bool loom_vm_kernel_query_exact_value(loom_low_lower_context_t* context,
                                             loom_value_id_t source_result,
                                             uint64_t* out_value) {
  *out_value = 0;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) return false;
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, source_result), &value) ||
      value < 0) {
    return false;
  }
  *out_value = (uint64_t)value;
  return true;
}

static iree_status_t loom_vm_kernel_bind_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint64_t value) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_inline_constant_build(
      loom_low_lower_context_builder(context), value, result_type,
      source_op->location, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

iree_status_t loom_vm_kernel_map_abi_layout(
    loom_low_lower_context_t* context, const loom_type_t* argument_types,
    const loom_value_id_t* argument_values, iree_host_size_t argument_count,
    loom_named_attr_slice_t* out_abi_layout) {
  *out_abi_layout = loom_named_attr_slice_empty();
  iree_host_size_t abi_argument_count = 0;
  if (!iree_host_size_checked_add(argument_count,
                                  LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT,
                                  &abi_argument_count) ||
      abi_argument_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM kernel ABI argument count exceeds u16");
  }

  loom_type_t* abi_argument_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, abi_argument_count, sizeof(*abi_argument_types),
      (void**)&abi_argument_types));
  for (iree_host_size_t i = 0; i < argument_count; ++i) {
    abi_argument_types[i] = argument_types[i];
  }
  loom_type_t launch_argument_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &launch_argument_type));
  for (iree_host_size_t i = argument_count; i < abi_argument_count; ++i) {
    abi_argument_types[i] = launch_argument_type;
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  iree_string_view_t* abi_argument_names = NULL;
  if (argument_values != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, abi_argument_count, sizeof(*abi_argument_names),
        (void**)&abi_argument_names));
    for (iree_host_size_t i = 0; i < argument_count; ++i) {
      abi_argument_names[i] =
          loom_module_value_name(module, argument_values[i]);
    }
    for (iree_host_size_t i = 0; i < LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT;
         ++i) {
      abi_argument_names[argument_count + i] =
          kLoomVmKernelLaunchArgumentNames[i];
    }
  }

  loom_type_t authored_signature = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_module_intern_function_type(
      module, argument_types, (uint16_t)argument_count,
      /*result_types=*/NULL, /*result_count=*/0, &authored_signature));
  loom_attribute_t layout_attr = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_attr(
      module,
      (loom_vm_call_abi_source_fields_t){
          .types = abi_argument_types,
          .presentation_names = abi_argument_names,
          .count = abi_argument_count,
      },
      (loom_vm_call_abi_source_fields_t){0}, authored_signature,
      loom_low_lower_context_function_arena(context), &layout_attr));
  *out_abi_layout = loom_attr_as_dict(layout_attr);
  return iree_ok_status();
}

static iree_status_t loom_vm_kernel_append_launch_arguments(
    loom_low_lower_context_t* context,
    loom_value_id_t out_arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT]) {
  loom_type_t argument_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &argument_type));

  loom_op_t* low_function = loom_low_lower_context_low_function(context);
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_function_body(low_function));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  for (uint16_t i = 0; i < LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, entry_block, argument_type, &out_arguments[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_kernel_name_launch_arguments(
    loom_low_lower_context_t* context,
    const loom_value_id_t arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT]) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT; ++i) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, kLoomVmKernelLaunchArgumentNames[i], &name_id));
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_name(module, arguments[i], name_id));
  }
  return iree_ok_status();
}

static uint16_t loom_vm_kernel_direct_argument_base(
    loom_vm_kernel_plan_kind_t plan_kind) {
  switch (plan_kind) {
    case LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_ID:
      return LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_ID_X;
    case LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_SIZE:
      return LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_X;
    case LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_COUNT:
      return LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_COUNT_X;
    case LOOM_VM_KERNEL_PLAN_KIND_WORKITEM_ID:
      return LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_X;
    default:
      IREE_ASSERT_UNREACHABLE("direct VM kernel launch argument");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_vm_kernel_emit_dispatch_id(
    loom_low_lower_context_t* context,
    const loom_value_id_t arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT],
    loom_kernel_dimension_t dimension, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_result) {
  return loom_vm_arithmetic_emit_madd_i64(
      context,
      arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_ID_X + dimension],
      arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_X + dimension],
      arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_X + dimension],
      result_type, location, out_result);
}

static iree_status_t loom_vm_kernel_emit_subgroup_id(
    loom_low_lower_context_t* context,
    const loom_value_id_t arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT],
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  loom_value_id_t yz = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_arithmetic_emit_madd_i64(
      context, arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_Y],
      arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_Z],
      arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_Y], result_type,
      location, &yz));
  return loom_vm_arithmetic_emit_madd_i64(
      context, arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_X], yz,
      arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKITEM_ID_X], result_type,
      location, out_result);
}

static iree_status_t loom_vm_kernel_emit_subgroup_count(
    loom_low_lower_context_t* context,
    const loom_value_id_t arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT],
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  loom_value_id_t xy = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_arithmetic_emit_mul_i64(
      context, arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_X],
      arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_Y], result_type,
      location, &xy));
  return loom_vm_arithmetic_emit_mul_i64(
      context, xy, arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_WORKGROUP_SIZE_Z],
      result_type, location, out_result);
}

iree_status_t loom_vm_kernel_emit_preamble(loom_low_lower_context_t* context) {
  if (!loom_kernel_def_isa(
          loom_low_lower_context_source_function(context).op)) {
    return iree_ok_status();
  }

  loom_value_id_t launch_arguments[LOOM_VM_KERNEL_LAUNCH_ARGUMENT_COUNT];
  IREE_RETURN_IF_ERROR(
      loom_vm_kernel_append_launch_arguments(context, launch_arguments));

  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided ||
        !loom_vm_kernel_plan_is_selected(selected_plan.plan)) {
      continue;
    }
    const loom_vm_kernel_plan_kind_t plan_kind =
        loom_vm_kernel_plan_kind_for_plan(selected_plan.plan);
    if (plan_kind == LOOM_VM_KERNEL_PLAN_KIND_NOOP) continue;

    const loom_op_t* source_op = selected_plan.source_op;
    const loom_value_id_t source_result =
        loom_vm_kernel_query_result(source_op);
    IREE_ASSERT_NE(source_result, LOOM_VALUE_ID_INVALID);
    uint64_t exact_value = 0;
    if (plan_kind == LOOM_VM_KERNEL_PLAN_KIND_CONSTANT_ZERO) {
      IREE_RETURN_IF_ERROR(loom_vm_kernel_bind_constant(
          context, source_op, source_result, /*value=*/0));
      continue;
    }
    if (plan_kind == LOOM_VM_KERNEL_PLAN_KIND_CONSTANT_ONE) {
      IREE_RETURN_IF_ERROR(loom_vm_kernel_bind_constant(
          context, source_op, source_result, /*value=*/1));
      continue;
    }
    if (loom_vm_kernel_query_exact_value(context, source_result,
                                         &exact_value)) {
      IREE_RETURN_IF_ERROR(loom_vm_kernel_bind_constant(
          context, source_op, source_result, exact_value));
      continue;
    }

    if (plan_kind >= LOOM_VM_KERNEL_PLAN_KIND_WORKGROUP_ID &&
        plan_kind <= LOOM_VM_KERNEL_PLAN_KIND_WORKITEM_ID) {
      const loom_kernel_dimension_t dimension =
          loom_vm_kernel_query_dimension(source_op);
      IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
      const uint16_t argument_base =
          loom_vm_kernel_direct_argument_base(plan_kind);
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_result, launch_arguments[argument_base + dimension]));
      continue;
    }

    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                  source_result, &result_type));
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    switch (plan_kind) {
      case LOOM_VM_KERNEL_PLAN_KIND_WORKITEM_DISPATCH_ID: {
        const loom_kernel_dimension_t dimension =
            loom_vm_kernel_query_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        IREE_RETURN_IF_ERROR(loom_vm_kernel_emit_dispatch_id(
            context, launch_arguments, dimension, result_type,
            source_op->location, &low_result));
        break;
      }
      case LOOM_VM_KERNEL_PLAN_KIND_SUBGROUP_ID: {
        IREE_RETURN_IF_ERROR(loom_vm_kernel_emit_subgroup_id(
            context, launch_arguments, result_type, source_op->location,
            &low_result));
        break;
      }
      case LOOM_VM_KERNEL_PLAN_KIND_SUBGROUP_COUNT: {
        IREE_RETURN_IF_ERROR(loom_vm_kernel_emit_subgroup_count(
            context, launch_arguments, result_type, source_op->location,
            &low_result));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE("derived VM kernel query");
        IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_result, low_result));
  }
  return loom_vm_kernel_name_launch_arguments(context, launch_arguments);
}

iree_status_t loom_vm_kernel_emit_op(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     bool* out_handled) {
  *out_handled = loom_vm_kernel_plan_is_selected(plan);
  if (!*out_handled) return iree_ok_status();

  if (loom_vm_kernel_plan_kind_for_plan(plan) ==
      LOOM_VM_KERNEL_PLAN_KIND_NOOP) {
    const loom_value_id_t* source_results = loom_op_const_results(source_op);
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_elide_value(context, source_results[i]));
    }
    return iree_ok_status();
  }
  const loom_value_id_t source_result = loom_vm_kernel_query_result(source_op);
  IREE_ASSERT_NE(source_result, LOOM_VALUE_ID_INVALID);
  IREE_ASSERT(
      loom_low_lower_source_value_has_low_mapping(context, source_result));
  return iree_ok_status();
}
