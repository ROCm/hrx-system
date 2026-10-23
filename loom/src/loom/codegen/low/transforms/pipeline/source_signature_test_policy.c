// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/source_signature_test_policy.h"

#include <string.h>

#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

typedef struct loom_test_low_source_signature_group_t {
  // First flat logical carrier owned by this source object.
  uint16_t logical_offset;
  // Number of flat logical carriers owned by this source object.
  uint16_t logical_count;
  // One physical register range carrying the complete source object.
  loom_type_t physical_type;
} loom_test_low_source_signature_group_t;

typedef struct loom_test_low_source_signature_plan_t {
  // Flat logical argument types borrowed from the common retained plan.
  const loom_type_t* logical_argument_types;
  // Flat logical result types borrowed from the common retained plan.
  const loom_type_t* logical_result_types;
  // Number of physical argument groups at the start of groups.
  uint16_t argument_group_count;
  // Number of physical result groups following argument groups.
  uint16_t result_group_count;
  // Source-object groups in argument order followed by result order.
  loom_test_low_source_signature_group_t groups[];
} loom_test_low_source_signature_plan_t;

static iree_status_t loom_test_low_source_signature_unsupported(
    loom_low_lower_context_t* context, iree_string_view_t reason) {
  return loom_low_lower_emit_source_signature_unsupported(
      context, loom_low_lower_context_source_function(context).op, reason);
}

static iree_status_t loom_test_low_plan_source_signature_groups(
    loom_low_lower_context_t* context, const loom_c_abi_object_t* objects,
    uint16_t object_count, const loom_type_t* logical_types,
    uint16_t logical_count, loom_type_t* physical_types,
    loom_test_low_source_signature_group_t* groups) {
  uint32_t logical_offset = 0;
  for (uint16_t i = 0; i < object_count; ++i) {
    const loom_c_abi_object_t* object = &objects[i];
    if (object->logical_count == 0 ||
        object->logical_offset != logical_offset ||
        (uint32_t)logical_offset + object->logical_count > logical_count) {
      return loom_test_low_source_signature_unsupported(
          context, IREE_SV("test target requires nonempty source objects"));
    }
    const uint16_t component_count = object->logical_count;
    const loom_type_t first_type = logical_types[logical_offset];
    if (!loom_low_type_is_register(first_type)) {
      return loom_test_low_source_signature_unsupported(
          context, IREE_SV("test target requires register source components"));
    }
    uint64_t unit_count = 0;
    for (uint16_t j = 0; j < component_count; ++j) {
      const loom_type_t component_type = logical_types[logical_offset + j];
      if (!loom_low_register_type_same_class(first_type, component_type)) {
        return loom_test_low_source_signature_unsupported(
            context,
            IREE_SV("test target groups only one register class per object"));
      }
      unit_count += loom_low_register_type_unit_count(component_type);
    }
    if (unit_count == 0 || unit_count > UINT32_MAX) {
      return loom_test_low_source_signature_unsupported(
          context, IREE_SV("test target source object is too large"));
    }
    const loom_type_t physical_type =
        component_count == 1 ? first_type
                             : loom_low_register_carrier_type_with_unit_count(
                                   first_type, (uint32_t)unit_count);
    physical_types[i] = physical_type;
    groups[i] = (loom_test_low_source_signature_group_t){
        .logical_offset = (uint16_t)logical_offset,
        .logical_count = component_count,
        .physical_type = physical_type,
    };
    logical_offset += component_count;
  }
  if (logical_offset != logical_count) {
    return loom_test_low_source_signature_unsupported(
        context, IREE_SV("source object roots do not cover the signature"));
  }
  return iree_ok_status();
}

iree_status_t loom_test_low_plan_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    const loom_c_abi_signature_layout_t* layout,
    const loom_type_t* logical_argument_types, uint16_t logical_argument_count,
    const loom_type_t* logical_result_types, uint16_t logical_result_count,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    loom_low_lower_source_signature_target_plan_t* out_plan) {
  (void)user_data;
  (void)scratch_arena;
  *out_plan = (loom_low_lower_source_signature_target_plan_t){0};
  const loom_func_c_abi_signature_t* signature = &layout->signature;
  if (signature->arguments.count > UINT16_MAX ||
      signature->results.count > UINT16_MAX) {
    return loom_test_low_source_signature_unsupported(
        context, IREE_SV("test target source object count exceeds 65535"));
  }
  const uint16_t argument_group_count = (uint16_t)signature->arguments.count;
  const uint16_t result_group_count = (uint16_t)signature->results.count;
  loom_type_t* physical_arguments = NULL;
  loom_type_t* physical_results = NULL;
  if (argument_group_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        result_arena, argument_group_count, sizeof(*physical_arguments),
        (void**)&physical_arguments));
  }
  if (result_group_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        result_arena, result_group_count, sizeof(*physical_results),
        (void**)&physical_results));
  }
  loom_test_low_source_signature_plan_t* plan = NULL;
  const iree_host_size_t group_count =
      (iree_host_size_t)argument_group_count + result_group_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      result_arena, sizeof(*plan) + group_count * sizeof(*plan->groups),
      (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->logical_argument_types = logical_argument_types;
  plan->logical_result_types = logical_result_types;
  plan->argument_group_count = argument_group_count;
  plan->result_group_count = result_group_count;
  IREE_RETURN_IF_ERROR(loom_test_low_plan_source_signature_groups(
      context, layout->arguments, argument_group_count, logical_argument_types,
      logical_argument_count, physical_arguments, plan->groups));
  IREE_RETURN_IF_ERROR(loom_test_low_plan_source_signature_groups(
      context, layout->results, result_group_count, logical_result_types,
      logical_result_count, physical_results,
      plan->groups + argument_group_count));
  if (loom_low_lower_context_error_count(context) != 0) {
    return iree_ok_status();
  }
  loom_named_attr_t* abi_layout = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(result_arena, sizeof(*abi_layout),
                                           (void**)&abi_layout));
  loom_string_id_t projection_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("source_projection"), &projection_key));
  *abi_layout = (loom_named_attr_t){
      .name_id = projection_key,
      .value = loom_attr_bool(true),
  };
  *out_plan = (loom_low_lower_source_signature_target_plan_t){
      .argument_types = physical_arguments,
      .argument_count = argument_group_count,
      .result_types = physical_results,
      .result_count = result_group_count,
      .abi_layout = loom_make_named_attr_slice(abi_layout, 1),
      .target_data = plan,
  };
  return iree_ok_status();
}

static iree_status_t loom_test_low_pack_source_signature_groups(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_test_low_source_signature_group_t* groups, uint16_t group_count,
    const loom_value_id_t* input_values, uint16_t input_count,
    loom_value_id_t* output_values, uint16_t output_count) {
  IREE_ASSERT_EQ(group_count, output_count);
  uint16_t logical_count = 0;
  for (uint16_t i = 0; i < group_count; ++i) {
    const loom_test_low_source_signature_group_t* group = &groups[i];
    IREE_ASSERT_EQ(group->logical_offset, logical_count);
    if (group->logical_count == 1) {
      output_values[i] = input_values[logical_count];
    } else {
      loom_op_t* concat_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_concat_build(
          loom_low_lower_context_builder(context), input_values + logical_count,
          group->logical_count, group->physical_type, source_op->location,
          &concat_op));
      output_values[i] = loom_low_concat_result(concat_op);
    }
    logical_count += group->logical_count;
  }
  IREE_ASSERT_EQ(logical_count, input_count);
  return iree_ok_status();
}

static iree_status_t loom_test_low_unpack_source_signature_groups(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_test_low_source_signature_group_t* groups, uint16_t group_count,
    const loom_type_t* logical_types, const loom_value_id_t* input_values,
    uint16_t input_count, loom_value_id_t* output_values,
    uint16_t output_count) {
  IREE_ASSERT_EQ(group_count, input_count);
  uint16_t logical_count = 0;
  for (uint16_t i = 0; i < group_count; ++i) {
    const loom_test_low_source_signature_group_t* group = &groups[i];
    IREE_ASSERT_EQ(group->logical_offset, logical_count);
    if (group->logical_count == 1) {
      output_values[logical_count] = input_values[i];
      ++logical_count;
      continue;
    }
    uint32_t unit_offset = 0;
    for (uint16_t j = 0; j < group->logical_count; ++j) {
      const loom_type_t logical_type = logical_types[logical_count];
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_slice_build(
          loom_low_lower_context_builder(context), input_values[i], unit_offset,
          logical_type, source_op->location, &slice_op));
      output_values[logical_count++] = loom_low_slice_result(slice_op);
      unit_offset += loom_low_register_type_unit_count(logical_type);
    }
  }
  IREE_ASSERT_EQ(logical_count, output_count);
  return iree_ok_status();
}

iree_status_t loom_test_low_emit_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_source_signature_emission_phase_t phase,
    const loom_op_t* source_op, const void* target_plan,
    const loom_value_id_t* input_values, uint16_t input_count,
    loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state) {
  (void)user_data;
  (void)state_arena;
  (void)inout_state;
  const loom_test_low_source_signature_plan_t* plan = target_plan;
  switch (phase) {
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_DEFINITION_ENTRY:
      return loom_test_low_unpack_source_signature_groups(
          context, source_op, plan->groups, plan->argument_group_count,
          plan->logical_argument_types, input_values, input_count,
          output_values, output_count);
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_DEFINITION_EXIT:
      return loom_test_low_pack_source_signature_groups(
          context, source_op, plan->groups + plan->argument_group_count,
          plan->result_group_count, input_values, input_count, output_values,
          output_count);
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_CALL_ARGUMENTS:
      return loom_test_low_pack_source_signature_groups(
          context, source_op, plan->groups, plan->argument_group_count,
          input_values, input_count, output_values, output_count);
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_CALL_RESULTS:
      return loom_test_low_unpack_source_signature_groups(
          context, source_op, plan->groups + plan->argument_group_count,
          plan->result_group_count, plan->logical_result_types, input_values,
          input_count, output_values, output_count);
    default:
      IREE_ASSERT_UNREACHABLE("unknown source signature emission phase");
      IREE_BUILTIN_UNREACHABLE();
  }
}
