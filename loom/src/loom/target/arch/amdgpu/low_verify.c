// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/low_verify.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/instruction_constraints.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef struct loom_amdgpu_low_verify_state_t {
  // Target resolved for the current function.
  const loom_low_resolved_target_t* target;
  // Borrowed function name used in diagnostics.
  iree_string_view_t function_name;
  // Normalized compiler-semantic properties for the resolved target record.
  loom_amdgpu_target_properties_t properties;
  // Canonical target selector used only as diagnostic context.
  iree_string_view_t target_name;
} loom_amdgpu_low_verify_state_t;

static iree_status_t loom_amdgpu_low_verify_begin_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state) {
  (void)provider;
  *out_provider_state = NULL;
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(target->target_facts);
  if (target_facts == NULL) {
    return iree_ok_status();
  }

  loom_amdgpu_low_verify_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      loom_low_verify_context_arena(context), sizeof(*state), (void**)&state));
  *state = (loom_amdgpu_low_verify_state_t){
      .target = target,
      .function_name = loom_low_diagnostic_function_name(
          loom_low_verify_context_module(context),
          loom_low_verify_context_function_op(context)),
      .properties = target_facts->properties,
      .target_name = target_facts->identity.target->name,
  };
  *out_provider_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_verify_instruction_constraints(
    loom_low_verify_context_t* context,
    const loom_amdgpu_low_verify_state_t* state,
    const loom_low_descriptor_packet_t* packet) {
  if (state->properties.instruction_constraints == 0) {
    return iree_ok_status();
  }
  const iree_string_view_t descriptor_key =
      loom_low_descriptor_packet_diagnostic_key(state->target->descriptor_set,
                                                packet);
  const uint16_t descriptor_attr_index =
      loom_low_descriptor_packet_attribute_index(packet);
  loom_amdgpu_instruction_constraint_bits_t active_constraints =
      loom_amdgpu_instruction_constraints_for_descriptor(
          state->target->descriptor_set, packet->descriptor) &
      state->properties.instruction_constraints;
  while (active_constraints != 0) {
    // Select the least-significant atom for deterministic diagnostics.
    const loom_amdgpu_instruction_constraint_bit_t constraint_bit =
        (loom_amdgpu_instruction_constraint_bit_t)(active_constraints &
                                                   (~active_constraints + 1u));
    active_constraints &= ~constraint_bit;
    const loom_amdgpu_instruction_constraint_info_t constraint =
        loom_amdgpu_instruction_constraint_info(constraint_bit);
    const loom_diagnostic_param_t params[] = {
        loom_param_string(state->function_name),
        loom_param_with_field_ref(
            loom_param_string(descriptor_key),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                      descriptor_attr_index)),
        loom_param_string(state->target->descriptor_set_key),
        loom_param_string(state->target_name),
        loom_param_string(
            loom_amdgpu_instruction_constraint_kind_name(constraint.kind)),
        loom_param_string(constraint.constraint_key),
        loom_param_string(loom_amdgpu_instruction_constraint_resolution_name(
            constraint.resolution)),
    };
    IREE_RETURN_IF_ERROR(
        loom_low_verify_context_emit(context, packet->op, LOOM_ERR_AMDGPU_047,
                                     params, IREE_ARRAYSIZE(params)));
    if (loom_low_verify_context_should_stop(context)) break;
  }
  return iree_ok_status();
}

static const loom_low_immediate_t* loom_amdgpu_low_find_dpp_control_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  IREE_ASSERT_LE(descriptor->immediate_start, descriptor_set->immediate_count);
  IREE_ASSERT_LE(descriptor->immediate_count,
                 descriptor_set->immediate_count - descriptor->immediate_start);
  const loom_low_immediate_t* dpp_control = NULL;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    if (immediate->encoding_field_id != LOOM_AMDGPU_ENCODING_FIELD_DPP_CTRL) {
      continue;
    }
    IREE_ASSERT(dpp_control == NULL);
    dpp_control = immediate;
  }
  return dpp_control;
}

static bool loom_amdgpu_low_find_immediate_value(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t immediate_name, const loom_low_immediate_t* immediate,
    uint16_t* out_attrs_attr_index, int64_t* out_value) {
  *out_attrs_attr_index = UINT16_MAX;
  *out_value = 0;
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (!loom_low_packet_try_op_attrs(op, &attrs, out_attrs_attr_index)) {
    IREE_ASSERT_UNREACHABLE("resolved low packet has no immediate dictionary");
    return false;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id >= module->strings.count ||
        !iree_string_view_equal(module->strings.entries[attr->name_id],
                                immediate_name)) {
      continue;
    }
    if (attr->value.kind != LOOM_ATTR_I64) {
      return false;
    }
    *out_value = attr->value.i64;
    return true;
  }
  if (iree_any_bit_set(immediate->flags,
                       LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    *out_value = immediate->default_value;
    return true;
  }
  return false;
}

static iree_status_t loom_amdgpu_low_verify_dpp_control(
    loom_low_verify_context_t* context,
    const loom_amdgpu_low_verify_state_t* state,
    const loom_low_descriptor_packet_t* packet) {
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const loom_amdgpu_encoding_format_flags_t format_flags =
      loom_amdgpu_encoding_format_flags(descriptor->encoding_format_id);
  if (!iree_any_bit_set(format_flags,
                        LOOM_AMDGPU_ENCODING_FORMAT_FLAG_DPP_CONTROL)) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->target->descriptor_set;
  const iree_string_view_t descriptor_key =
      loom_low_descriptor_packet_diagnostic_key(descriptor_set, packet);
  const loom_low_immediate_t* immediate =
      loom_amdgpu_low_find_dpp_control_immediate(descriptor_set, descriptor);
  if (immediate == NULL) {
    return iree_ok_status();
  }
  const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  uint16_t attrs_attr_index = UINT16_MAX;
  int64_t value = 0;
  const loom_module_t* module = loom_low_verify_context_module(context);
  if (!loom_amdgpu_low_find_immediate_value(module, packet->op, immediate_name,
                                            immediate, &attrs_attr_index,
                                            &value) ||
      value < 0 || value > 0x1FF) {
    return iree_ok_status();
  }

  loom_amdgpu_dpp_control_decoding_t decoding = {0};
  if (loom_amdgpu_dpp_control_decode((uint16_t)value, &decoding)) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_with_field_ref(
          loom_param_string(descriptor_key),
          loom_diagnostic_field_ref(
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
              loom_low_descriptor_packet_attribute_index(packet))),
      loom_param_string(state->target->descriptor_set_key),
      loom_param_with_field_ref(
          loom_param_string(immediate_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
      loom_param_u32((uint32_t)value),
      loom_param_string(IREE_SV("dpp.control.encoding")),
      loom_param_string(IREE_SV("reserved_encoding")),
  };
  return loom_low_verify_context_emit(context, packet->op, LOOM_ERR_AMDGPU_044,
                                      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_low_verify_storage_address(
    loom_low_verify_context_t* context,
    const loom_amdgpu_low_verify_state_t* state, const loom_op_t* op) {
  if (!loom_low_storage_address_isa(op)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_verify_context_module(context);
  const loom_value_id_t result = loom_low_storage_address_result(op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (loom_low_register_type_descriptor_set_stable_id(result_type) ==
          state->target->descriptor_set->stable_id &&
      loom_low_register_type_class_id(result_type) ==
          LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
      loom_low_register_type_unit_count(result_type) == 1) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(state->target)),
      loom_param_string(loom_low_diagnostic_export_name(state->target)),
      loom_param_string(loom_low_diagnostic_config_key(state->target)),
      loom_param_string(state->function_name),
      loom_param_string(loom_low_diagnostic_operation_name(module, op)),
      loom_param_string(loom_low_diagnostic_value_name(module, result)),
      loom_param_type(result_type),
      loom_param_u32(LOOM_AMDGPU_REG_CLASS_ID_VGPR),
      loom_param_u32(1),
  };
  return loom_low_verify_context_emit(context, op, LOOM_ERR_AMDGPU_049, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_descriptor_packet_t* packet) {
  (void)provider;
  loom_amdgpu_low_verify_state_t* state =
      (loom_amdgpu_low_verify_state_t*)provider_state;
  if (state == NULL || loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_verify_storage_address(context, state, packet->op));
  if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_verify_dpp_control(context, state, packet));
  return loom_amdgpu_low_verify_instruction_constraints(context, state, packet);
}

const loom_low_verify_provider_t loom_amdgpu_low_verify_provider = {
    .name = IREE_SVL("amdgpu"),
    .begin_function = loom_amdgpu_low_verify_begin_function,
    .verify_op = loom_amdgpu_low_verify_op,
};
