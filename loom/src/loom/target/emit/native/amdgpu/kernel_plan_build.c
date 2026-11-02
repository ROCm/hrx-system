// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_plan_build.h"

#include <stdint.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/function_plan_build.h"
#include "loom/target/emit/native/amdgpu/kernel_record_build.h"
#include "loom/target/emit/native/amdgpu/kernel_report.h"

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_kernel_plan_emit_private_segment_limit(
    const loom_low_emission_frame_t* frame, uint64_t private_bytes,
    uint32_t limit_bytes, iree_diagnostic_emitter_t emitter) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_resolved_target(
          &frame->schedule.target);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(
          frame->module, frame->schedule.function_op)),
      loom_param_string(processor->name),
      loom_param_u32(loom_low_resolved_target_bundle(&frame->schedule.target)
                         ->snapshot->subgroup_size),
      loom_param_u64(private_bytes),
      loom_param_u32(limit_bytes),
  };
  const loom_diagnostic_emission_t emission = {
      .op = frame->schedule.function_op,
      .error = LOOM_ERR_AMDGPU_053,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static loom_target_compile_report_target_resources_t
loom_amdgpu_kernel_plan_target_resources(
    const loom_amdgpu_occupancy_target_resources_t* target_resources) {
  return (loom_target_compile_report_target_resources_t){
      .scalar_register_class = target_resources->scalar_register_class,
      .scalar_register_count = target_resources->scalar_register_count,
      .vector_register_class = target_resources->vector_register_class,
      .vector_register_count = target_resources->vector_register_count,
      .subgroup_size = target_resources->wave_size,
      .max_subgroups_per_simd = target_resources->max_waves_per_simd,
      .resident_subgroups_per_simd = target_resources->resident_waves_per_simd,
      .occupancy_percent = target_resources->occupancy_percent,
      .limiting_resource = target_resources->limiting_resource,
      .residency_summary = target_resources->residency_summary,
  };
}

static iree_status_t loom_amdgpu_kernel_plan_retain_direct_resource_units(
    const uint32_t* source_units, iree_host_size_t source_unit_count,
    const loom_target_residency_direct_resource_override_t* overrides,
    iree_host_size_t override_count, iree_arena_allocator_t* arena,
    const uint32_t** out_direct_resource_units) {
  *out_direct_resource_units = NULL;
  uint32_t* direct_resource_units = NULL;
  if (source_unit_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, source_unit_count, sizeof(*direct_resource_units),
        (void**)&direct_resource_units));
    memcpy(direct_resource_units, source_units,
           source_unit_count * sizeof(*direct_resource_units));
  }
  for (iree_host_size_t i = 0; i < override_count; ++i) {
    IREE_ASSERT_LT(overrides[i].resource_id, source_unit_count);
    direct_resource_units[overrides[i].resource_id] = overrides[i].units;
  }
  *out_direct_resource_units = direct_resource_units;
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_kernel_plan_emit_unsatisfied_residency_requirement(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_kernel_record_t* record,
    const loom_amdgpu_occupancy_target_resources_t* occupancy,
    iree_diagnostic_emitter_t emitter) {
  IREE_ASSERT(emitter.fn != NULL);
  IREE_ASSERT(iree_any_bit_set(
      occupancy->residency_summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_REQUIRED_TIER_UNSATISFIED));
  IREE_ASSERT(record->metadata.has_required_workgroup_size);
  IREE_ASSERT_NE(occupancy->waves_per_workgroup, 0);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_target_key(&frame->schedule.target)),
      loom_param_string(
          loom_low_diagnostic_export_name(&frame->schedule.target)),
      loom_param_string(
          loom_low_diagnostic_config_key(&frame->schedule.target)),
      loom_param_string(loom_low_diagnostic_function_name(
          frame->module, frame->schedule.function_op)),
      loom_param_string(loom_low_diagnostic_operation_name(
          frame->module, frame->schedule.function_op)),
      loom_param_string(record->processor->name),
      loom_param_u32(record->metadata.max_flat_workgroup_size),
      loom_param_u32(occupancy->waves_per_workgroup),
      loom_param_u32(record->metadata.wavefront_size),
      loom_param_u32(occupancy->residency_summary.required_tier),
      loom_param_u32(occupancy->resident_waves_per_simd),
      loom_param_u32(record->metadata.sgpr_count),
      loom_param_u32(record->metadata.vgpr_count),
      loom_param_u32(record->metadata.group_segment_fixed_size),
      loom_param_string(occupancy->limiting_resource),
  };
  const loom_diagnostic_emission_t emission = {
      .op = frame->schedule.function_op,
      .error = LOOM_ERR_AMDGPU_052,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

// Finalizes target resource accounting before native emission. The allocation
// vector owns every descriptor register class; the prepared record supplies
// ABI and hardware-entry floors that are intentionally outside allocatable
// body storage. Only reports and rejected kernels retain a copied exact vector.
static iree_status_t loom_amdgpu_kernel_plan_admit_residency(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_kernel_record_t* record,
    iree_diagnostic_emitter_t emitter, loom_target_compile_report_t* report,
    iree_arena_allocator_t* arena, bool* out_accepted) {
  *out_accepted = false;
  const uint32_t* allocation_direct_resource_units =
      frame->allocation.physical_extents.ends_by_reg_class;
  const iree_host_size_t direct_resource_unit_count =
      frame->allocation.physical_extents.count;
  IREE_ASSERT(frame->residency.model != NULL);
  IREE_ASSERT_EQ(direct_resource_unit_count,
                 frame->residency.model->direct_resources.resource_count);
  IREE_ASSERT_LT(LOOM_AMDGPU_REG_CLASS_ID_SGPR, direct_resource_unit_count);
  IREE_ASSERT_LT(LOOM_AMDGPU_REG_CLASS_ID_VGPR, direct_resource_unit_count);
  IREE_ASSERT_GE(
      record->metadata.sgpr_count,
      allocation_direct_resource_units[LOOM_AMDGPU_REG_CLASS_ID_SGPR]);
  IREE_ASSERT_GE(
      record->architectural_vgpr_count,
      allocation_direct_resource_units[LOOM_AMDGPU_REG_CLASS_ID_VGPR]);
  const loom_target_residency_direct_resource_override_t resource_overrides[] =
      {
          {
              .resource_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR,
              .units = record->metadata.sgpr_count,
          },
          {
              .resource_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,
              .units = record->architectural_vgpr_count,
          },
      };
  const uint32_t final_residency_tier =
      loom_target_residency_evaluate_tier_with_direct_resource_overrides(
          frame->residency, allocation_direct_resource_units,
          direct_resource_unit_count, resource_overrides,
          IREE_ARRAYSIZE(resource_overrides));
  const bool accepted = loom_target_residency_view_meets_required_tier(
      frame->residency, final_residency_tier);
  if (accepted && report == NULL) {
    *out_accepted = true;
    return iree_ok_status();
  }

  const uint32_t* direct_resource_units = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_plan_retain_direct_resource_units(
      allocation_direct_resource_units, direct_resource_unit_count,
      resource_overrides, IREE_ARRAYSIZE(resource_overrides), arena,
      &direct_resource_units));
  loom_target_residency_constraint_list_t residency_constraints = {0};
  const bool capture_residency_constraints =
      accepted &&
      loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_RESIDENCY_CONSTRAINTS);
  loom_amdgpu_occupancy_target_resources_t occupancy = {0};
  const uint32_t flat_workgroup_size =
      record->metadata.has_required_workgroup_size
          ? record->metadata.max_flat_workgroup_size
          : 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_build_target_resources(
      record->processor, record->metadata.wavefront_size, direct_resource_units,
      direct_resource_unit_count, flat_workgroup_size,
      record->metadata.group_segment_fixed_size, arena, &occupancy,
      capture_residency_constraints ? &residency_constraints : NULL));
  if (!accepted) {
    return loom_amdgpu_kernel_plan_emit_unsatisfied_residency_requirement(
        frame, record, &occupancy, emitter);
  }

  const loom_target_compile_report_target_resources_t target_resources =
      loom_amdgpu_kernel_plan_target_resources(&occupancy);
  loom_target_compile_report_record_target_resources(report, &target_resources);
  if (capture_residency_constraints) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_residency_constraints(
            report, &residency_constraints));
  }
  *out_accepted = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_kernel_plan_build(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_kernel_plan_options_t* options,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_amdgpu_kernel_plan_t* out_plan) {
  loom_amdgpu_function_plan_t function_plan = {0};
  bool function_accepted = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_function_plan_build(
      frame, options->emitter, arena, &function_accepted, &function_plan));
  if (!function_accepted) {
    *out_accepted = false;
    *out_plan = (loom_amdgpu_kernel_plan_t){0};
    return iree_ok_status();
  }
  return loom_amdgpu_kernel_plan_build_from_function(
      frame, &function_plan, options, arena, out_accepted, out_plan);
}

iree_status_t loom_amdgpu_kernel_plan_build_from_function(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_function_plan_t* function_plan,
    const loom_amdgpu_kernel_plan_options_t* options,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_amdgpu_kernel_plan_t* out_plan) {
  *out_accepted = false;
  *out_plan = (loom_amdgpu_kernel_plan_t){
      .function = *function_plan,
  };

  // Private arrays and spill reservations share one hardware scratch segment.
  // Admit its final aligned extent before narrowing metadata or planning code.
  const loom_amdgpu_storage_layout_t* storage_layout =
      &function_plan->storage_layout;
  const loom_amdgpu_processor_properties_t* processor_properties =
      loom_amdgpu_target_processor_properties_from_resolved_target(
          &frame->schedule.target);
  const uint32_t wave_size =
      loom_low_resolved_target_bundle(&frame->schedule.target)
          ->snapshot->subgroup_size;
  const uint32_t private_byte_limit =
      processor_properties->wavefront.max_scratch_bytes / wave_size;
  if (storage_layout->segment_sizes.private_segment_fixed_size >
      private_byte_limit) {
    return loom_amdgpu_kernel_plan_emit_private_segment_limit(
        frame, storage_layout->segment_sizes.private_segment_fixed_size,
        private_byte_limit, options->emitter);
  }
  if (storage_layout->segment_sizes.group_segment_fixed_size > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel emission group segment size exceeds metadata limits");
  }

  const loom_amdgpu_kernel_record_options_t record_options = {
      .abi_layout = options->abi_layout,
      .abi_verify = options->abi_verify,
      .preflight = &function_plan->preflight,
      .minimum_vgpr_count =
          function_plan->packet_plan.wait_states.minimum_vgpr_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build(
      &frame->schedule, &frame->allocation, storage_layout, &record_options,
      &out_plan->record, arena));

  bool residency_accepted = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_plan_admit_residency(
      frame, &out_plan->record, options->emitter, options->report, arena,
      &residency_accepted));
  if (!residency_accepted) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_report_record_plan(
      options->report, &function_plan->packet_plan));
  *out_accepted = true;
  return iree_ok_status();
}
