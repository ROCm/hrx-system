// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/occupancy.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/planning/occupancy_model.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/launch.h"
#include "loom/target/types.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_status_t loom_amdgpu_occupancy_round_up_u32(
    uint32_t value, uint32_t multiple, uint32_t* out_rounded_value) {
  if (value == 0 || multiple <= 1) {
    *out_rounded_value = value;
    return iree_ok_status();
  }
  uint32_t remainder = value % multiple;
  if (remainder == 0) {
    *out_rounded_value = value;
    return iree_ok_status();
  }
  const uint32_t delta = multiple - remainder;
  if (value > UINT32_MAX - delta) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU occupancy rounded allocation unit count overflows");
  }
  *out_rounded_value = value + delta;
  return iree_ok_status();
}

static const loom_amdgpu_occupancy_model_t* loom_amdgpu_occupancy_select_model(
    const loom_amdgpu_processor_properties_t* properties, uint32_t wave_size) {
  return loom_amdgpu_occupancy_model_for_properties(properties, wave_size);
}

static const loom_amdgpu_occupancy_model_t*
loom_amdgpu_occupancy_select_target_model(
    const loom_low_resolved_target_t* target) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_resolved_target(target);
  IREE_ASSERT(processor != NULL);
  const uint32_t wave_size =
      loom_low_resolved_target_bundle(target)->snapshot->subgroup_size;
  IREE_ASSERT(loom_amdgpu_processor_properties_support_wavefront_size(
      &processor->properties, wave_size));
  return loom_amdgpu_occupancy_select_model(&processor->properties, wave_size);
}

static uint32_t loom_amdgpu_occupancy_register_class_index(
    const loom_amdgpu_occupancy_model_t* model,
    uint16_t descriptor_reg_class_id) {
  if (descriptor_reg_class_id >= model->descriptor_reg_class_count) {
    return LOOM_AMDGPU_OCCUPANCY_CLASS_NONE;
  }
  const uint16_t class_index =
      model->register_class_indices_by_descriptor_reg_class_id
          [descriptor_reg_class_id];
  if (class_index == UINT16_MAX) {
    return LOOM_AMDGPU_OCCUPANCY_CLASS_NONE;
  }
  IREE_ASSERT(class_index < model->register_class_count);
  return class_index;
}

const loom_target_residency_model_t* loom_amdgpu_occupancy_residency_model(
    const loom_low_resolved_target_t* target) {
  const loom_amdgpu_occupancy_model_t* model =
      loom_amdgpu_occupancy_select_target_model(target);
  return &model->residency_model;
}

static void loom_amdgpu_occupancy_collect_allocations(
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_occupancy_model_t* model,
    loom_amdgpu_occupancy_register_class_t* class_summaries) {
  IREE_ASSERT_EQ(allocation->physical_extents.count,
                 model->descriptor_reg_class_count);
  for (iree_host_size_t i = 0; i < model->register_class_count; ++i) {
    const uint16_t reg_class_id =
        model->register_classes[i].descriptor_reg_class_id;
    class_summaries[i].allocated_units =
        allocation->physical_extents.ends_by_reg_class[reg_class_id];
  }
}

static iree_status_t loom_amdgpu_occupancy_add_u32(uint32_t lhs, uint32_t rhs,
                                                   uint32_t* out_value) {
  uint64_t value = (uint64_t)lhs + rhs;
  if (value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU occupancy summary overflows uint32_t");
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_collect_spills(
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_occupancy_model_t* model,
    loom_amdgpu_occupancy_register_class_t* class_summaries,
    iree_host_size_t class_summary_count,
    loom_amdgpu_occupancy_table_t* table) {
  for (iree_host_size_t i = 0; i < allocation->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan =
        &allocation->spill_plans[i];
    if (spill_plan->assignment_index >= allocation->assignment_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU occupancy spill plan references "
                              "assignment %" PRIu32
                              " but allocation has %zu "
                              "assignment(s)",
                              spill_plan->assignment_index,
                              allocation->assignment_count);
    }
    if (spill_plan->slot_space != LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU occupancy expected scratch spill slots, got '%.*s'",
          (int)loom_low_spill_slot_space_name(spill_plan->slot_space).size,
          loom_low_spill_slot_space_name(spill_plan->slot_space).data);
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[spill_plan->assignment_index];
    const uint32_t class_index = loom_amdgpu_occupancy_register_class_index(
        model, assignment->descriptor_reg_class_id);
    if (class_index >= class_summary_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU occupancy spill plan references non-modeled descriptor "
          "register class ID %" PRIu16,
          assignment->descriptor_reg_class_id);
    }

    loom_amdgpu_occupancy_register_class_t* class_summary =
        &class_summaries[class_index];
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_count, 1, &class_summary->spill_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_bytes, spill_plan->byte_size,
        &class_summary->spill_bytes));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_store_count, spill_plan->store_count,
        &class_summary->spill_store_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_reload_count, spill_plan->reload_count,
        &class_summary->spill_reload_count));

    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(table->spill_count, 1,
                                                       &table->spill_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        table->scratch_spill_bytes, spill_plan->byte_size,
        &table->scratch_spill_bytes));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        table->spill_store_count, spill_plan->store_count,
        &table->spill_store_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        table->spill_reload_count, spill_plan->reload_count,
        &table->spill_reload_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_flat_workgroup_size(
    const loom_target_export_plan_t* export_plan, uint32_t* out_flat_size) {
  *out_flat_size = 0;
  if (!export_plan || export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_ok_status();
  }
  const loom_target_workgroup_size_t size =
      export_plan->hal_kernel.required_workgroup_size;
  if (!loom_target_workgroup_size_flat_product_u32(&size, out_flat_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU workgroup size overflows uint32_t");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_apply_residency_evaluation(
    uint32_t allocation_granularity,
    const loom_target_residency_resource_evaluation_t* resource_evaluation,
    const loom_target_residency_cliff_t* cliffs, iree_host_size_t cliff_count,
    uint32_t best_tier, uint32_t* out_allocated_units,
    uint32_t* out_rounded_units, uint32_t* out_wave_limit,
    uint32_t* out_next_cliff_units, uint32_t* out_units_until_next_cliff) {
  if (resource_evaluation->units > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU occupancy resource '%.*s' exceeds 32-bit allocation units",
        (int)resource_evaluation->name.size, resource_evaluation->name.data);
  }
  *out_allocated_units = (uint32_t)resource_evaluation->units;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_round_up_u32(
      *out_allocated_units, allocation_granularity, out_rounded_units));
  *out_wave_limit = resource_evaluation->tier;

  loom_target_residency_cliff_evaluation_t cliff_evaluation;
  loom_target_residency_evaluate_cliffs(cliffs, cliff_count, best_tier,
                                        resource_evaluation->units,
                                        &cliff_evaluation);
  if (iree_any_bit_set(
          cliff_evaluation.flags,
          LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_WORSE_TIER)) {
    *out_next_cliff_units = cliff_evaluation.worse_cliff_units;
    *out_units_until_next_cliff =
        (uint32_t)cliff_evaluation.additional_units_to_worse_tier;
  } else {
    *out_next_cliff_units = 0;
    *out_units_until_next_cliff = UINT32_MAX;
  }
  return iree_ok_status();
}

static void loom_amdgpu_occupancy_capture_unique_limiting_resource(
    const loom_target_residency_query_t* query,
    const loom_target_residency_resource_evaluation_t* resource,
    loom_target_residency_summary_t* summary) {
  if (query->limiting_resource_count != 1 ||
      !iree_any_bit_set(
          resource->flags,
          LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_LIMITING)) {
    return;
  }
  summary->flags |=
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE;
  summary->limiting_resource = resource->name;
  summary->limiting_resource_units = resource->units;
  summary->limiting_resource_reduction_units_to_next_better_tier =
      resource->reduction_units_to_next_better_tier;
  if (iree_any_bit_set(
          resource->flags,
          LOOM_TARGET_RESIDENCY_RESOURCE_EVALUATION_FLAG_HAS_NEXT_WORSE_TIER)) {
    summary->flags |=
        LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_LIMITING_RESOURCE_NEXT_WORSE_TIER;
    summary->limiting_resource_next_worse_tier = resource->next_worse_tier;
    summary->limiting_resource_next_worse_cliff_units =
        resource->next_worse_cliff_units;
    summary->limiting_resource_additional_units_to_next_worse_tier =
        resource->additional_units_to_next_worse_tier;
  }
}

static iree_status_t loom_amdgpu_occupancy_finalize_register_limits(
    const loom_amdgpu_occupancy_model_t* model,
    loom_amdgpu_occupancy_register_class_t* class_summaries,
    loom_amdgpu_occupancy_pressure_resource_t* pressure_resources,
    iree_arena_allocator_t* arena, loom_amdgpu_occupancy_table_t* table) {
  const loom_target_residency_model_t* residency_model =
      &model->residency_model;
  if (residency_model->best_tier != table->max_waves_per_simd ||
      residency_model->direct_resources.resource_count !=
          model->descriptor_reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "generated AMDGPU occupancy and residency models disagree");
  }

  uint64_t* direct_resource_units = NULL;
  if (residency_model->direct_resources.resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, residency_model->direct_resources.resource_count,
        sizeof(*direct_resource_units), (void**)&direct_resource_units));
    memset(direct_resource_units, 0,
           residency_model->direct_resources.resource_count *
               sizeof(*direct_resource_units));
  }
  for (iree_host_size_t i = 0; i < table->register_class_count; ++i) {
    const uint16_t direct_resource_id =
        model->register_classes[i].descriptor_reg_class_id;
    direct_resource_units[direct_resource_id] =
        class_summaries[i].allocated_units;
  }

  loom_target_residency_query_t residency_query;
  IREE_RETURN_IF_ERROR(loom_target_residency_query(
      residency_model, direct_resource_units,
      residency_model->direct_resources.resource_count, arena,
      &residency_query));
  IREE_ASSERT(residency_query.model_available);
  table->residency_summary = (loom_target_residency_summary_t){
      .flags = LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_VALID |
               (residency_query.has_next_better_tier
                    ? LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER
                    : 0),
      .best_tier = residency_query.best_tier,
      .tier = residency_query.tier,
      .next_better_tier = residency_query.next_better_tier,
      .limiting_resource_count =
          (uint32_t)residency_query.limiting_resource_count,
  };
  table->resident_waves_per_simd = residency_query.tier;
  table->limiting_resource_kind =
      LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES;
  table->limiting_resource_index = LOOM_AMDGPU_OCCUPANCY_RESOURCE_NONE;

  for (iree_host_size_t i = 0; i < table->register_class_count; ++i) {
    loom_amdgpu_occupancy_register_class_t* class_summary = &class_summaries[i];
    const loom_amdgpu_occupancy_register_class_model_t* class_model =
        &model->register_classes[i];
    const loom_target_residency_cliff_range_t pressure_cliff_range =
        loom_target_residency_direct_resource_cliff_range(
            &residency_model->direct_resources,
            class_model->descriptor_reg_class_id);
    const loom_target_residency_cliff_t* pressure_cliffs =
        pressure_cliff_range.count == 0
            ? NULL
            : &residency_model->direct_resources
                   .cliffs[pressure_cliff_range.start];
    uint32_t allocated_units = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_apply_residency_evaluation(
        class_model->allocation_granularity,
        &residency_query.resources[class_model->descriptor_reg_class_id],
        pressure_cliffs, pressure_cliff_range.count, residency_model->best_tier,
        &allocated_units, &class_summary->rounded_units,
        &class_summary->wave_limit, &class_summary->next_cliff_units,
        &class_summary->units_until_next_cliff));
    IREE_ASSERT_EQ(allocated_units, class_summary->allocated_units);
    loom_amdgpu_occupancy_capture_unique_limiting_resource(
        &residency_query,
        &residency_query.resources[class_model->descriptor_reg_class_id],
        &table->residency_summary);
    if (table->limiting_resource_kind ==
            LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES &&
        table->resident_waves_per_simd < table->max_waves_per_simd &&
        class_summary->wave_limit == table->resident_waves_per_simd) {
      table->limiting_resource_kind =
          LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_REGISTER_CLASS;
      table->limiting_resource_index = (uint32_t)i;
    }
  }

  const loom_target_residency_derived_resource_table_t* resource_table =
      &residency_model->derived_resources;
  IREE_ASSERT_EQ(table->pressure_resource_count,
                 resource_table->resource_count);
  for (iree_host_size_t i = 0; i < table->pressure_resource_count; ++i) {
    loom_amdgpu_occupancy_pressure_resource_t* pressure_resource =
        &pressure_resources[i];
    const loom_target_residency_derived_resource_t* resource =
        &resource_table->resources[i];
    const loom_target_residency_cliff_t* pressure_cliffs =
        resource->cliff_count == 0
            ? NULL
            : &resource_table->cliffs[resource->cliff_start];
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_apply_residency_evaluation(
        resource->allocation_granularity,
        &residency_query.resources[residency_query.direct_resource_count + i],
        pressure_cliffs, resource->cliff_count, residency_model->best_tier,
        &pressure_resource->allocated_units, &pressure_resource->rounded_units,
        &pressure_resource->wave_limit, &pressure_resource->next_cliff_units,
        &pressure_resource->units_until_next_cliff));
    loom_amdgpu_occupancy_capture_unique_limiting_resource(
        &residency_query,
        &residency_query.resources[residency_query.direct_resource_count + i],
        &table->residency_summary);
    if (table->limiting_resource_kind ==
            LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES &&
        table->resident_waves_per_simd < table->max_waves_per_simd &&
        pressure_resource->wave_limit == table->resident_waves_per_simd) {
      table->limiting_resource_kind =
          LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_PRESSURE_RESOURCE;
      table->limiting_resource_index = (uint32_t)i;
    }
  }

  return iree_ok_status();
}

static uint32_t loom_amdgpu_occupancy_ceil_div_u32(uint32_t numerator,
                                                   uint32_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

static uint32_t loom_amdgpu_occupancy_wave_limit_for_workgroups(
    const loom_amdgpu_occupancy_model_t* model, uint32_t waves_per_workgroup,
    uint32_t workgroup_count) {
  const uint32_t resident_waves = waves_per_workgroup * workgroup_count;
  const uint32_t resident_waves_per_simd = loom_amdgpu_occupancy_ceil_div_u32(
      resident_waves, model->domain.simd_count);
  return iree_min(resident_waves_per_simd, model->max_waves_per_simd);
}

static void loom_amdgpu_occupancy_apply_launch_limits(
    const loom_amdgpu_occupancy_model_t* model, uint32_t flat_workgroup_size,
    uint32_t local_memory_bytes, loom_amdgpu_occupancy_table_t* table) {
  table->flat_workgroup_size = flat_workgroup_size;
  uint32_t launch_wave_limit = 0;
  if (flat_workgroup_size != 0) {
    const uint32_t waves_per_workgroup = loom_amdgpu_occupancy_ceil_div_u32(
        flat_workgroup_size, table->wave_size);
    table->waves_per_workgroup = waves_per_workgroup;

    const uint32_t wave_slots =
        model->max_waves_per_simd * model->domain.simd_count;
    const uint32_t wave_limited_workgroup_count =
        wave_slots / waves_per_workgroup;
    const uint32_t workgroup_slot_count =
        waves_per_workgroup == 1
            ? wave_slots
            : iree_min(wave_limited_workgroup_count,
                       model->domain.max_barrier_workgroup_count);

    uint32_t local_memory_workgroup_count = wave_slots;
    if (local_memory_bytes != 0) {
      const uint32_t granularity =
          model->domain.local_memory_allocation_granularity;
      const uint32_t rounded_local_memory_bytes =
          loom_amdgpu_occupancy_ceil_div_u32(local_memory_bytes, granularity) *
          granularity;
      IREE_ASSERT_LE(rounded_local_memory_bytes,
                     model->domain.local_memory_bytes);
      local_memory_workgroup_count = iree_min(
          model->domain.local_memory_bytes / rounded_local_memory_bytes,
          wave_slots);
    }

    const uint32_t workgroup_wave_limit =
        loom_amdgpu_occupancy_wave_limit_for_workgroups(
            model, waves_per_workgroup, workgroup_slot_count);
    const uint32_t local_memory_wave_limit =
        loom_amdgpu_occupancy_wave_limit_for_workgroups(
            model, waves_per_workgroup, local_memory_workgroup_count);
    launch_wave_limit = iree_min(workgroup_wave_limit, local_memory_wave_limit);
    if (launch_wave_limit < table->resident_waves_per_simd ||
        (launch_wave_limit == table->resident_waves_per_simd &&
         launch_wave_limit < table->max_waves_per_simd &&
         table->limiting_resource_kind ==
             LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES)) {
      table->resident_waves_per_simd = launch_wave_limit;
      table->limiting_resource_kind =
          local_memory_wave_limit < workgroup_wave_limit
              ? LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_LOCAL_MEMORY
              : LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_WORKGROUP_SLOTS;
      table->limiting_resource_index = LOOM_AMDGPU_OCCUPANCY_RESOURCE_NONE;
    }
  }

  table->occupancy_percent =
      (table->resident_waves_per_simd * 100u) / table->max_waves_per_simd;

  const bool has_exact_transition =
      flat_workgroup_size != 0 &&
      table->resident_waves_per_simd == table->residency_summary.tier &&
      (!iree_any_bit_set(
           table->residency_summary.flags,
           LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER) ||
       launch_wave_limit >= table->residency_summary.next_better_tier);
  if (!has_exact_transition) {
    table->residency_summary = (loom_target_residency_summary_t){0};
  }
}

static iree_string_view_t loom_amdgpu_occupancy_limiting_resource_name(
    const loom_amdgpu_occupancy_table_t* table) {
  switch (table->limiting_resource_kind) {
    case LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES:
      return IREE_SV("max_waves");
    case LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_REGISTER_CLASS:
      if (table->limiting_resource_index >= table->register_class_count) {
        return IREE_SV("unknown");
      }
      return table->register_classes[table->limiting_resource_index]
          .register_class;
    case LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_PRESSURE_RESOURCE:
      if (table->limiting_resource_index >= table->pressure_resource_count) {
        return IREE_SV("unknown");
      }
      return table->pressure_resources[table->limiting_resource_index].resource;
    case LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_WORKGROUP_SLOTS:
      return IREE_SV("amdgpu.workgroup_slots");
    case LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_LOCAL_MEMORY:
      return IREE_SV("amdgpu.lds");
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_amdgpu_occupancy_build_target_resources(
    const loom_amdgpu_processor_info_t* processor, uint32_t wave_size,
    uint32_t scalar_register_count, uint32_t vector_register_count,
    uint32_t flat_workgroup_size, uint32_t local_memory_bytes,
    iree_arena_allocator_t* arena,
    loom_amdgpu_occupancy_target_resources_t* out_resources) {
  *out_resources = (loom_amdgpu_occupancy_target_resources_t){0};
  if (processor == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU final occupancy requires a resolved processor");
  }
  if (!loom_amdgpu_processor_properties_support_wavefront_size(
          &processor->properties, wave_size)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU final occupancy processor '%.*s' does not support wave size "
        "%" PRIu32,
        (int)processor->name.size, processor->name.data, wave_size);
  }

  const loom_amdgpu_occupancy_model_t* model =
      loom_amdgpu_occupancy_select_model(&processor->properties, wave_size);
  const loom_target_residency_derived_resource_table_t* resource_table =
      &model->residency_model.derived_resources;

  loom_amdgpu_occupancy_register_class_t* register_classes = NULL;
  if (model->register_class_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, model->register_class_count, sizeof(*register_classes),
        (void**)&register_classes));
    memset(register_classes, 0,
           model->register_class_count * sizeof(*register_classes));
  }
  loom_amdgpu_occupancy_pressure_resource_t* pressure_resources = NULL;
  if (resource_table->resource_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_table->resource_count, sizeof(*pressure_resources),
        (void**)&pressure_resources));
    memset(pressure_resources, 0,
           resource_table->resource_count * sizeof(*pressure_resources));
  }

  iree_string_view_t scalar_register_class = iree_string_view_empty();
  iree_string_view_t vector_register_class = iree_string_view_empty();
  bool has_complete_direct_resource_values = true;
  for (iree_host_size_t i = 0; i < model->register_class_count; ++i) {
    const loom_amdgpu_occupancy_register_class_model_t* class_model =
        &model->register_classes[i];
    register_classes[i] = (loom_amdgpu_occupancy_register_class_t){
        .register_class = class_model->register_class,
        .descriptor_reg_class_id = class_model->descriptor_reg_class_id,
        .pool_units = class_model->pool_units,
        .allocation_granularity = class_model->allocation_granularity,
        .limits_occupancy = class_model->limits_occupancy,
        .units_until_next_cliff = UINT32_MAX,
    };
    if (class_model->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
      scalar_register_class = class_model->register_class;
      register_classes[i].allocated_units = scalar_register_count;
    } else if (class_model->descriptor_reg_class_id ==
               LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      vector_register_class = class_model->register_class;
      register_classes[i].allocated_units = vector_register_count;
    } else {
      has_complete_direct_resource_values = false;
    }
  }
  IREE_ASSERT(!iree_string_view_is_empty(scalar_register_class),
              "generated AMDGPU occupancy models must define SGPR rows");
  IREE_ASSERT(!iree_string_view_is_empty(vector_register_class),
              "generated AMDGPU occupancy models must define VGPR rows");

  for (uint16_t i = 0; i < resource_table->resource_count; ++i) {
    const loom_target_residency_derived_resource_t* resource =
        &resource_table->resources[i];
    pressure_resources[i] = (loom_amdgpu_occupancy_pressure_resource_t){
        .resource = resource->name,
        .pool_units = resource->pool_units,
        .allocation_granularity = resource->allocation_granularity,
        .units_until_next_cliff = UINT32_MAX,
    };
  }

  loom_amdgpu_occupancy_table_t table = {
      .processor = processor->name,
      .wave_size = wave_size,
      .max_waves_per_simd = model->max_waves_per_simd,
      .resident_waves_per_simd = model->max_waves_per_simd,
      .limiting_resource_kind =
          LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES,
      .limiting_resource_index = LOOM_AMDGPU_OCCUPANCY_RESOURCE_NONE,
      .register_classes = register_classes,
      .register_class_count = model->register_class_count,
      .pressure_resources = pressure_resources,
      .pressure_resource_count = resource_table->resource_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_finalize_register_limits(
      model, register_classes, pressure_resources, arena, &table));
  if (!has_complete_direct_resource_values) {
    table.residency_summary = (loom_target_residency_summary_t){0};
  }
  loom_amdgpu_occupancy_apply_launch_limits(model, flat_workgroup_size,
                                            local_memory_bytes, &table);

  *out_resources = (loom_amdgpu_occupancy_target_resources_t){
      .scalar_register_class = scalar_register_class,
      .scalar_register_count = scalar_register_count,
      .vector_register_class = vector_register_class,
      .vector_register_count = vector_register_count,
      .wave_size = wave_size,
      .max_waves_per_simd = table.max_waves_per_simd,
      .resident_waves_per_simd = table.resident_waves_per_simd,
      .occupancy_percent = table.occupancy_percent,
      .limiting_resource = loom_amdgpu_occupancy_limiting_resource_name(&table),
      .residency_summary = table.residency_summary,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_emit_resource_summary(
    const loom_amdgpu_occupancy_table_t* table,
    iree_string_view_t resource_name, uint32_t pool_units,
    uint32_t allocated_units, iree_string_view_t limiting_resource,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_target_key(&table->allocation->target)),
      loom_param_string(
          loom_low_diagnostic_export_name(&table->allocation->target)),
      loom_param_string(
          loom_low_diagnostic_config_key(&table->allocation->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          table->allocation->module, table->allocation->function_op)),
      loom_param_string(resource_name),
      loom_param_u32(pool_units),
      loom_param_u32(allocated_units),
      loom_param_u32(table->occupancy_percent),
      loom_param_string(limiting_resource),
  };
  loom_diagnostic_emission_t emission = {
      .op = table->allocation->function_op,
      .error = LOOM_ERR_BACKEND_010,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_amdgpu_occupancy_emit_summary(
    const loom_amdgpu_occupancy_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  const iree_string_view_t limiting_resource =
      loom_amdgpu_occupancy_limiting_resource_name(table);
  for (iree_host_size_t i = 0; i < table->register_class_count; ++i) {
    const loom_amdgpu_occupancy_register_class_t* class_summary =
        &table->register_classes[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_emit_resource_summary(
        table, class_summary->register_class, class_summary->pool_units,
        class_summary->allocated_units, limiting_resource, emitter));
  }
  for (iree_host_size_t i = 0; i < table->pressure_resource_count; ++i) {
    const loom_amdgpu_occupancy_pressure_resource_t* pressure_resource =
        &table->pressure_resources[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_emit_resource_summary(
        table, pressure_resource->resource, pressure_resource->pool_units,
        pressure_resource->allocated_units, limiting_resource, emitter));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_occupancy_build(
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_occupancy_options_t* options,
    iree_arena_allocator_t* arena, loom_amdgpu_occupancy_table_t* out_table) {
  *out_table = (loom_amdgpu_occupancy_table_t){0};

  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_resolved_target(&allocation->target);
  if (processor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU occupancy requires an AMDGPU processor "
                            "target record");
  }
  const loom_target_bundle_t* target_bundle =
      loom_low_resolved_target_bundle(&allocation->target);
  const uint32_t wave_size = target_bundle->snapshot->subgroup_size;
  if (wave_size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU occupancy requires a fixed target "
                            "subgroup size");
  }
  IREE_ASSERT(loom_amdgpu_processor_properties_support_wavefront_size(
      &processor->properties, wave_size));

  const loom_amdgpu_occupancy_model_t* model =
      loom_amdgpu_occupancy_select_model(&processor->properties, wave_size);
  const loom_target_residency_derived_resource_table_t* resource_table =
      &model->residency_model.derived_resources;

  loom_amdgpu_occupancy_register_class_t* register_classes = NULL;
  if (model->register_class_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, model->register_class_count, sizeof(*register_classes),
        (void**)&register_classes));
    memset(register_classes, 0,
           model->register_class_count * sizeof(*register_classes));
  }
  loom_amdgpu_occupancy_pressure_resource_t* pressure_resources = NULL;
  if (resource_table->resource_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_table->resource_count, sizeof(*pressure_resources),
        (void**)&pressure_resources));
    memset(pressure_resources, 0,
           resource_table->resource_count * sizeof(*pressure_resources));
  }

  loom_amdgpu_occupancy_table_t table = {
      .allocation = allocation,
      .processor = processor->name,
      .wave_size = wave_size,
      .max_waves_per_simd = model->max_waves_per_simd,
      .resident_waves_per_simd = model->max_waves_per_simd,
      .limiting_resource_kind =
          LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES,
      .limiting_resource_index = LOOM_AMDGPU_OCCUPANCY_RESOURCE_NONE,
      .register_classes = register_classes,
      .register_class_count = model->register_class_count,
      .pressure_resources = pressure_resources,
      .pressure_resource_count = resource_table->resource_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_flat_workgroup_size(
      target_bundle->export_plan, &table.flat_workgroup_size));
  for (iree_host_size_t i = 0; i < model->register_class_count; ++i) {
    register_classes[i] = (loom_amdgpu_occupancy_register_class_t){
        .register_class = model->register_classes[i].register_class,
        .descriptor_reg_class_id =
            model->register_classes[i].descriptor_reg_class_id,
        .pool_units = model->register_classes[i].pool_units,
        .allocation_granularity =
            model->register_classes[i].allocation_granularity,
        .limits_occupancy = model->register_classes[i].limits_occupancy,
        .units_until_next_cliff = UINT32_MAX,
    };
  }
  for (uint16_t i = 0; i < resource_table->resource_count; ++i) {
    const loom_target_residency_derived_resource_t* resource =
        &resource_table->resources[i];
    pressure_resources[i] = (loom_amdgpu_occupancy_pressure_resource_t){
        .resource = resource->name,
        .pool_units = resource->pool_units,
        .allocation_granularity = resource->allocation_granularity,
        .units_until_next_cliff = UINT32_MAX,
    };
  }

  loom_amdgpu_occupancy_collect_allocations(allocation, model,
                                            register_classes);
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_collect_spills(
      allocation, model, register_classes, model->register_class_count,
      &table));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_finalize_register_limits(
      model, register_classes, pressure_resources, arena, &table));
  loom_amdgpu_occupancy_apply_launch_limits(model, table.flat_workgroup_size, 0,
                                            &table);

  if (options && iree_any_bit_set(options->diagnostic_flags,
                                  LOOM_AMDGPU_OCCUPANCY_DIAGNOSTIC_SUMMARY)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_occupancy_emit_summary(&table, options->emitter));
  }
  *out_table = table;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_write_cliff_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint32_t value) {
  if (value == UINT32_MAX || value == 0) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
}

static iree_status_t loom_amdgpu_occupancy_write_register_class(
    const loom_amdgpu_occupancy_register_class_t* register_class,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("register_class"), register_class->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("allocated_units"), register_class->allocated_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("rounded_units"), register_class->rounded_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("pool_units"), register_class->pool_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("allocation_granularity"),
      register_class->allocation_granularity));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("limits_occupancy"), register_class->limits_occupancy));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("wave_limit"), register_class->wave_limit));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_field(
      &object, IREE_SV("next_cliff_units"), register_class->next_cliff_units));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_field(
      &object, IREE_SV("units_until_next_cliff"),
      register_class->units_until_next_cliff));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("spill_count"), register_class->spill_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("spill_bytes"), register_class->spill_bytes));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("spill_store_count"),
                                          register_class->spill_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("spill_reload_count"),
      register_class->spill_reload_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_amdgpu_occupancy_write_pressure_resource(
    const loom_amdgpu_occupancy_pressure_resource_t* pressure_resource,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("resource"), pressure_resource->resource));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("allocated_units"), pressure_resource->allocated_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("rounded_units"), pressure_resource->rounded_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("pool_units"), pressure_resource->pool_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("allocation_granularity"),
      pressure_resource->allocation_granularity));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("wave_limit"), pressure_resource->wave_limit));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_field(
      &object, IREE_SV("next_cliff_units"),
      pressure_resource->next_cliff_units));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_field(
      &object, IREE_SV("units_until_next_cliff"),
      pressure_resource->units_until_next_cliff));
  return loom_json_object_end(&object);
}

iree_status_t loom_amdgpu_occupancy_format_json(
    const loom_amdgpu_occupancy_table_t* table,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.amdgpu.occupancy.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("function"),
      loom_low_diagnostic_function_name(table->allocation->module,
                                        table->allocation->function_op)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target"), table->allocation->target.target_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("descriptor_set"),
      table->allocation->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("processor"), table->processor));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("wave_size"), table->wave_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("max_waves_per_simd"), table->max_waves_per_simd));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flat_workgroup_size"), table->flat_workgroup_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("waves_per_workgroup"), table->waves_per_workgroup));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("resident_waves_per_simd"),
      table->resident_waves_per_simd));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("occupancy_percent"), table->occupancy_percent));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("limiting_resource"),
      loom_amdgpu_occupancy_limiting_resource_name(table)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("spill_count"), table->spill_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("scratch_spill_bytes"), table->scratch_spill_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("spill_store_count"), table->spill_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("spill_reload_count"), table->spill_reload_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("register_classes")));
  loom_json_array_writer_t register_classes;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &register_classes));
  for (iree_host_size_t i = 0; i < table->register_class_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&register_classes));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_register_class(
        &table->register_classes[i], &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&register_classes));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("pressure_resources")));
  loom_json_array_writer_t pressure_resources;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &pressure_resources));
  for (iree_host_size_t i = 0; i < table->pressure_resource_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&pressure_resources));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_pressure_resource(
        &table->pressure_resources[i], &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&pressure_resources));
  return loom_json_object_end(&object);
}
