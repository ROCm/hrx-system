// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/addressability.h"

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/error/error_catalog.h"

static bool loom_low_addressability_assignment_exceeds_operand_range(
    const loom_low_operand_t* operand,
    const loom_low_allocation_assignment_t* assignment) {
  const uint64_t assigned_end =
      (uint64_t)assignment->location_base + assignment->location_count;
  switch (operand->address_map_kind) {
    case LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET:
      return assigned_end > operand->addressable_unit_count;
    case LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE: {
      const uint32_t addressable_unit_count = operand->addressable_unit_count;
      if (assignment->location_count == 0 || addressable_unit_count == 0) {
        return true;
      }
      const uint64_t assigned_last = assigned_end - 1;
      return assignment->location_base / addressable_unit_count !=
             assigned_last / addressable_unit_count;
    }
    default:
      return false;
  }
}

static iree_status_t loom_low_addressability_emit_error(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet, const loom_low_operand_t* operand,
    loom_diagnostic_field_ref_t field_ref,
    const loom_low_allocation_assignment_t* assignment,
    iree_diagnostic_emitter_t emitter,
    loom_low_addressability_validation_result_t* result) {
  if (packet->packet_index > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "addressability packet index exceeds u32 range");
  }
  ++result->error_count;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const iree_string_view_t packet_key = loom_low_descriptor_set_string(
      descriptor_set, packet->descriptor->key_string_offset);
  const iree_string_view_t operand_field = loom_low_descriptor_set_string(
      descriptor_set, operand->field_name_string_offset);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&schedule->target)),
      loom_param_string(loom_low_diagnostic_export_name(&schedule->target)),
      loom_param_string(loom_low_diagnostic_config_key(&schedule->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          schedule->module, schedule->function_op)),
      loom_param_string(packet_key),
      loom_param_u32((uint32_t)packet->packet_index),
      loom_param_with_field_ref(loom_param_string(operand_field), field_ref),
      loom_param_string(loom_low_diagnostic_value_name(allocation->module,
                                                       assignment->value_id)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          descriptor_set, assignment->value_class)),
      loom_param_u32(assignment->location_base),
      loom_param_u32(assignment->location_count),
      loom_param_string(
          loom_low_operand_address_map_kind_name(operand->address_map_kind)),
      loom_param_u32(operand->addressable_unit_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = packet->node->op,
      .error = LOOM_ERR_BACKEND_020,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_addressability_validate_operand(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet, uint16_t descriptor_operand_index,
    iree_diagnostic_emitter_t emitter,
    loom_low_addressability_validation_result_t* result) {
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const uint32_t operand_row =
      descriptor->operand_start + descriptor_operand_index;
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_row];
  if (!loom_low_operand_requires_low_window_assignment(operand)) {
    return iree_ok_status();
  }
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_descriptor_operand_assignment(allocation, packet,
                                                    descriptor_operand_index);
  if (!loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    return iree_ok_status();
  }
  if (!loom_low_addressability_assignment_exceeds_operand_range(operand,
                                                                assignment)) {
    return iree_ok_status();
  }
  const loom_diagnostic_field_ref_t field_ref = loom_diagnostic_field_ref(
      descriptor_operand_index < descriptor->result_count
          ? LOOM_DIAGNOSTIC_FIELD_RESULT
          : LOOM_DIAGNOSTIC_FIELD_OPERAND,
      operand->source_value_index);
  return loom_low_addressability_emit_error(schedule, allocation, packet,
                                            operand, field_ref, assignment,
                                            emitter, result);
}

static iree_status_t loom_low_addressability_validate_packet(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet, iree_diagnostic_emitter_t emitter,
    loom_low_addressability_validation_result_t* result) {
  if (packet->descriptor == NULL) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < packet->descriptor->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_addressability_validate_operand(
        schedule, allocation, packet, i, emitter, result));
  }
  return iree_ok_status();
}

iree_status_t loom_low_addressability_validate_allocated_packets(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_diagnostic_emitter_t emitter,
    loom_low_addressability_validation_result_t* out_result) {
  *out_result = (loom_low_addressability_validation_result_t){0};
  const iree_host_size_t packet_count = loom_low_packet_count(schedule);
  for (iree_host_size_t packet_index = 0; packet_index < packet_count;
       ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(schedule, packet_index);
    IREE_RETURN_IF_ERROR(loom_low_addressability_validate_packet(
        schedule, allocation, &packet, emitter, out_result));
  }
  return iree_ok_status();
}
