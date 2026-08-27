// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/preflight.h"

#include <inttypes.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/register_class.h"

static const loom_low_storage_space_set_t kLoomAmdgpuNativeStorageSpaces =
    LOOM_LOW_STORAGE_SPACE_SET_SCRATCH | LOOM_LOW_STORAGE_SPACE_SET_PRIVATE |
    LOOM_LOW_STORAGE_SPACE_SET_WORKGROUP;

// AMDGPU unspillable physical classes model singleton architectural state such
// as SCC, EXEC, and M0. They constrain scheduling/allocation but do not
// contribute SGPR/VGPR high-water metadata.
static bool loom_amdgpu_native_preflight_metadata_free_register_class(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= descriptor_set->reg_class_count) {
    return false;
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_id];
  return iree_all_bits_set(reg_class->flags,
                           LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
}

static iree_status_t loom_amdgpu_native_preflight_update_high_water(
    uint32_t location_base, uint32_t location_count, uint32_t* inout_value) {
  const uint64_t next_free = (uint64_t)location_base + location_count;
  if (next_free > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native emission register high-water mark overflows");
  }
  if ((uint32_t)next_free > *inout_value) {
    *inout_value = (uint32_t)next_free;
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_native_preflight_emit_unsupported_storage_space(
    const loom_low_schedule_table_t* schedule, loom_value_id_t storage_value_id,
    const loom_low_storage_layout_reservation_t* reservation,
    const loom_amdgpu_native_preflight_options_t* options,
    loom_amdgpu_native_preflight_t* preflight) {
  const iree_string_view_t storage_space =
      loom_low_storage_type_space_name(reservation->space);
  if (!options || options->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native emission cannot emit storage space '%.*s'",
        (int)storage_space.size, storage_space.data);
  }

  iree_string_view_t supported_storage_space_names[LOOM_STORAGE_SPACE_COUNT_];
  const iree_host_size_t supported_storage_space_count =
      loom_low_storage_space_set_names(
          kLoomAmdgpuNativeStorageSpaces,
          IREE_ARRAYSIZE(supported_storage_space_names),
          supported_storage_space_names);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&schedule->target)),
      loom_param_string(loom_low_diagnostic_export_name(&schedule->target)),
      loom_param_string(loom_low_diagnostic_config_key(&schedule->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          schedule->module, schedule->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(schedule->module, storage_value_id)),
      loom_param_string(storage_space),
      loom_param_string_list(supported_storage_space_names,
                             supported_storage_space_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_low_diagnostic_value_origin_op(
          schedule->module, storage_value_id, schedule->function_op),
      .error = LOOM_ERR_AMDGPU_036,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(options->emitter, &emission));
  ++preflight->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_native_preflight_collect_storage_usage(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_native_preflight_options_t* options,
    loom_amdgpu_native_preflight_t* preflight) {
  const loom_low_storage_layout_t* layout = &schedule->storage_layout;
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    const loom_low_storage_layout_record_t* record = &layout->records[i];
    if (loom_low_storage_space_set_contains(kLoomAmdgpuNativeStorageSpaces,
                                            record->reservation.space)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_native_preflight_emit_unsupported_storage_space(
            schedule, record->storage_value_id, &record->reservation, options,
            preflight));
  }
  return iree_ok_status();
}

static iree_string_view_t
loom_amdgpu_native_preflight_assignment_register_class_name(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->descriptor_reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      assignment->descriptor_reg_class_id >=
          allocation->target.descriptor_set->reg_class_count) {
    return IREE_SV("<unknown>");
  }
  const loom_low_reg_class_t* reg_class =
      &allocation->target.descriptor_set
           ->reg_classes[assignment->descriptor_reg_class_id];
  return loom_low_descriptor_set_string(allocation->target.descriptor_set,
                                        reg_class->name_string_offset);
}

static iree_status_t
loom_amdgpu_native_preflight_emit_unsupported_register_metadata(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    iree_string_view_t metadata_contract,
    loom_amdgpu_native_preflight_t* preflight,
    const loom_amdgpu_native_preflight_options_t* options) {
  const iree_string_view_t register_class =
      loom_amdgpu_native_preflight_assignment_register_class_name(allocation,
                                                                  assignment);
  if (!options || options->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native emission register class '%.*s' requires %.*s metadata",
        (int)register_class.size, register_class.data,
        (int)metadata_contract.size, metadata_contract.data);
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&allocation->target)),
      loom_param_string(loom_low_diagnostic_export_name(&allocation->target)),
      loom_param_string(loom_low_diagnostic_config_key(&allocation->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          allocation->module, allocation->function_op)),
      loom_param_string(loom_low_diagnostic_value_name(allocation->module,
                                                       assignment->value_id)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          allocation->target.descriptor_set, assignment->value_class)),
      loom_param_string(register_class),
      loom_param_string(metadata_contract),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_low_diagnostic_value_origin_op(
          allocation->module, assignment->value_id, allocation->function_op),
      .error = LOOM_ERR_AMDGPU_035,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(options->emitter, &emission));
  ++preflight->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_native_preflight_collect_register_usage(
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_native_preflight_options_t* options,
    loom_amdgpu_native_preflight_t* preflight) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
      const loom_low_reg_class_t* reg_class =
          &allocation->target.descriptor_set
               ->reg_classes[assignment->descriptor_reg_class_id];
      if (loom_low_reg_class_fixed_location_range_contains(
              reg_class, assignment->location_base,
              assignment->location_count)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_update_high_water(
          assignment->location_base, assignment->location_count,
          &preflight->next_free_sgpr));
      continue;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_update_high_water(
          assignment->location_base, assignment->location_count,
          &preflight->next_free_vgpr));
      continue;
    }
    if (loom_amdgpu_register_class_is_agpr(
            allocation->target.descriptor_set,
            assignment->descriptor_reg_class_id)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_native_preflight_emit_unsupported_register_metadata(
              allocation, assignment, IREE_SV("AGPR kernel-descriptor"),
              preflight, options));
      continue;
    }
    if (loom_amdgpu_native_preflight_metadata_free_register_class(
            allocation->target.descriptor_set,
            assignment->descriptor_reg_class_id)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_native_preflight_emit_unsupported_register_metadata(
            allocation, assignment, IREE_SV("kernel-descriptor"), preflight,
            options));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_native_preflight_analyze(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_native_preflight_options_t* options,
    loom_amdgpu_native_preflight_t* out_preflight) {
  *out_preflight = (loom_amdgpu_native_preflight_t){0};
  *out_preflight = (loom_amdgpu_native_preflight_t){
      .schedule = schedule,
      .allocation = allocation,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_collect_storage_usage(
      schedule, options, out_preflight));
  return loom_amdgpu_native_preflight_collect_register_usage(
      allocation, options, out_preflight);
}
