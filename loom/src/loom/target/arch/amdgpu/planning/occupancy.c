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
#include "loom/target/arch/amdgpu/planning/occupancy_tables.h"
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

static iree_status_t loom_amdgpu_occupancy_wave_limit(
    uint32_t pool_units, uint32_t allocation_granularity,
    uint32_t max_waves_per_simd, uint32_t allocated_units,
    uint32_t* out_rounded_units, uint32_t* out_wave_limit) {
  if (allocated_units == 0) {
    *out_rounded_units = 0;
    *out_wave_limit = max_waves_per_simd;
    return iree_ok_status();
  }
  uint32_t rounded_units = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_round_up_u32(
      allocated_units, allocation_granularity, &rounded_units));
  *out_rounded_units = rounded_units;
  if (rounded_units == 0) {
    *out_wave_limit = 0;
    return iree_ok_status();
  }
  uint32_t wave_limit = pool_units / rounded_units;
  if (wave_limit > max_waves_per_simd) {
    wave_limit = max_waves_per_simd;
  }
  *out_wave_limit = wave_limit;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_next_cliff_units(
    uint32_t pool_units, uint32_t allocation_granularity,
    uint32_t max_waves_per_simd, uint32_t allocated_units,
    uint32_t current_wave_limit, uint32_t* out_next_cliff_units) {
  *out_next_cliff_units = 0;
  if (current_wave_limit == 0) {
    return iree_ok_status();
  }
  const uint64_t start_candidate = (uint64_t)allocated_units + 1;
  uint64_t stop_candidate = (uint64_t)pool_units + allocation_granularity;
  if (stop_candidate > UINT32_MAX) {
    stop_candidate = UINT32_MAX;
  }
  for (uint64_t candidate = start_candidate; candidate <= stop_candidate;
       ++candidate) {
    uint32_t ignored_rounded_units = 0;
    uint32_t candidate_wave_limit = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_wave_limit(
        pool_units, allocation_granularity, max_waves_per_simd,
        (uint32_t)candidate, &ignored_rounded_units, &candidate_wave_limit));
    if (candidate_wave_limit < current_wave_limit) {
      *out_next_cliff_units = (uint32_t)candidate;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_select_model(
    uint16_t descriptor_set_ordinal, iree_string_view_t descriptor_set_key,
    const loom_amdgpu_occupancy_model_t** out_model) {
  *out_model = NULL;
  const loom_amdgpu_occupancy_model_t* model =
      loom_amdgpu_occupancy_model_for_descriptor_set_ordinal(
          descriptor_set_ordinal);
  if (model != NULL) {
    *out_model = model;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU occupancy model is not defined for descriptor set '%.*s' "
      "(ordinal %" PRIu16 ")",
      (int)descriptor_set_key.size, descriptor_set_key.data,
      descriptor_set_ordinal);
}

static iree_status_t loom_amdgpu_occupancy_lookup_register_class(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t register_class, uint16_t* out_descriptor_reg_class_id) {
  *out_descriptor_reg_class_id = LOOM_LOW_ID_NONE;
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* descriptor_reg_class =
        &descriptor_set->reg_classes[i];
    iree_string_view_t descriptor_register_class =
        loom_low_descriptor_set_string(
            descriptor_set, descriptor_reg_class->name_string_offset);
    if (!iree_string_view_equal(descriptor_register_class, register_class)) {
      continue;
    }
    if (i > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU occupancy descriptor register class ordinal overflows");
    }
    *out_descriptor_reg_class_id = (uint16_t)i;
    return iree_ok_status();
  }
  iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU descriptor set '%.*s' does not define occupancy register class "
      "'%.*s'",
      (int)descriptor_set_key.size, descriptor_set_key.data,
      (int)register_class.size, register_class.data);
}

iree_status_t loom_amdgpu_occupancy_build_schedule_pressure_cliffs(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_pressure_cliff_list_t* out_pressure_cliffs) {
  *out_pressure_cliffs = loom_low_schedule_pressure_cliff_list_empty();

  iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  const loom_amdgpu_occupancy_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_select_model(
      descriptor_set->descriptor_set_ordinal, descriptor_set_key, &model));
  if (model->register_class_count == 0 || model->max_waves_per_simd == 0) {
    return iree_ok_status();
  }

  iree_host_size_t max_cliff_count = 0;
  if (!iree_host_size_checked_mul(model->register_class_count,
                                  model->max_waves_per_simd,
                                  &max_cliff_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU schedule pressure cliff capacity exceeds host size");
  }
  loom_low_schedule_pressure_cliff_t* cliffs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, max_cliff_count, sizeof(*cliffs), (void**)&cliffs));
  iree_host_size_t cliff_count = 0;

  for (iree_host_size_t class_index = 0;
       class_index < model->register_class_count; ++class_index) {
    const loom_amdgpu_occupancy_register_class_model_t* class_model =
        &model->register_classes[class_index];
    uint16_t descriptor_reg_class_id = LOOM_LOW_ID_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_lookup_register_class(
        descriptor_set, class_model->register_class, &descriptor_reg_class_id));
    uint32_t previous_wave_limit = model->max_waves_per_simd;
    uint64_t stop_candidate =
        (uint64_t)class_model->pool_units + class_model->allocation_granularity;
    if (stop_candidate > UINT32_MAX) {
      stop_candidate = UINT32_MAX;
    }
    for (uint64_t candidate = 1; candidate <= stop_candidate; ++candidate) {
      uint32_t ignored_rounded_units = 0;
      uint32_t candidate_wave_limit = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_wave_limit(
          class_model->pool_units, class_model->allocation_granularity,
          model->max_waves_per_simd, (uint32_t)candidate,
          &ignored_rounded_units, &candidate_wave_limit));
      if (candidate_wave_limit >= previous_wave_limit) {
        continue;
      }
      if (cliff_count >= max_cliff_count) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU schedule pressure cliff capacity was underestimated");
      }
      cliffs[cliff_count++] = (loom_low_schedule_pressure_cliff_t){
          .descriptor_reg_class_id = descriptor_reg_class_id,
          .cliff_units = (uint32_t)candidate,
          .tier_before = previous_wave_limit,
          .tier_after = candidate_wave_limit,
      };
      previous_wave_limit = candidate_wave_limit;
      if (candidate_wave_limit == 0) {
        break;
      }
    }
  }

  *out_pressure_cliffs = (loom_low_schedule_pressure_cliff_list_t){
      .values = cliffs,
      .count = cliff_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_find_register_class(
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_occupancy_register_class_t* register_classes,
    iree_host_size_t register_class_count, uint16_t descriptor_reg_class_id,
    uint32_t* out_index) {
  for (iree_host_size_t i = 0; i < register_class_count; ++i) {
    if (register_classes[i].descriptor_reg_class_id ==
        descriptor_reg_class_id) {
      if (i > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU occupancy class index overflows");
      }
      *out_index = (uint32_t)i;
      return iree_ok_status();
    }
  }
  if (descriptor_reg_class_id >=
      allocation->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU occupancy assignment references invalid descriptor register "
        "class ID %" PRIu16,
        descriptor_reg_class_id);
  }
  const loom_low_reg_class_t* descriptor_reg_class =
      &allocation->target.descriptor_set->reg_classes[descriptor_reg_class_id];
  iree_string_view_t register_class =
      loom_low_descriptor_set_string(allocation->target.descriptor_set,
                                     descriptor_reg_class->name_string_offset);
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU occupancy model for descriptor set '%.*s' does not cover "
      "register class '%.*s'",
      (int)allocation->target.descriptor_set_key.size,
      allocation->target.descriptor_set_key.data, (int)register_class.size,
      register_class.data);
}

static iree_status_t loom_amdgpu_occupancy_collect_allocations(
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_occupancy_register_class_t* class_summaries,
    iree_host_size_t class_summary_count) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    uint32_t class_index = LOOM_AMDGPU_OCCUPANCY_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_find_register_class(
        allocation, class_summaries, class_summary_count,
        assignment->descriptor_reg_class_id, &class_index));
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    uint64_t allocated_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (allocated_end > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU occupancy allocated register range overflows");
    }
    if ((uint32_t)allocated_end >
        class_summaries[class_index].allocated_units) {
      class_summaries[class_index].allocated_units = (uint32_t)allocated_end;
    }
  }
  return iree_ok_status();
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
    uint32_t class_index = LOOM_AMDGPU_OCCUPANCY_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_find_register_class(
        allocation, class_summaries, class_summary_count,
        assignment->descriptor_reg_class_id, &class_index));

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

static iree_status_t loom_amdgpu_occupancy_finalize_resource_limit(
    uint32_t pool_units, uint32_t allocation_granularity,
    uint32_t max_waves_per_simd, uint32_t allocated_units,
    uint32_t* out_rounded_units, uint32_t* out_wave_limit,
    uint32_t* out_next_cliff_units, uint32_t* out_units_until_next_cliff) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_wave_limit(
      pool_units, allocation_granularity, max_waves_per_simd, allocated_units,
      out_rounded_units, out_wave_limit));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_next_cliff_units(
      pool_units, allocation_granularity, max_waves_per_simd, allocated_units,
      *out_wave_limit, out_next_cliff_units));
  *out_units_until_next_cliff = *out_next_cliff_units == 0
                                    ? UINT32_MAX
                                    : *out_next_cliff_units - allocated_units;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_finalize_limits(
    const loom_amdgpu_occupancy_model_t* model,
    loom_amdgpu_occupancy_register_class_t* class_summaries,
    loom_amdgpu_occupancy_pressure_resource_t* pressure_resources,
    loom_amdgpu_occupancy_table_t* table) {
  table->resident_waves_per_simd = table->max_waves_per_simd;
  table->limiting_resource_kind =
      LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES;
  table->limiting_resource_index = LOOM_AMDGPU_OCCUPANCY_RESOURCE_NONE;
  for (iree_host_size_t i = 0; i < table->register_class_count; ++i) {
    loom_amdgpu_occupancy_register_class_t* class_summary = &class_summaries[i];
    const loom_amdgpu_occupancy_register_class_model_t* class_model =
        &model->register_classes[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_finalize_resource_limit(
        class_model->pool_units, class_model->allocation_granularity,
        table->max_waves_per_simd, class_summary->allocated_units,
        &class_summary->rounded_units, &class_summary->wave_limit,
        &class_summary->next_cliff_units,
        &class_summary->units_until_next_cliff));
    if (class_summary->wave_limit < table->resident_waves_per_simd) {
      table->resident_waves_per_simd = class_summary->wave_limit;
      table->limiting_resource_kind =
          LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_REGISTER_CLASS;
      table->limiting_resource_index = (uint32_t)i;
    }
  }
  for (iree_host_size_t i = 0; i < table->pressure_resource_count; ++i) {
    loom_amdgpu_occupancy_pressure_resource_t* pressure_resource =
        &pressure_resources[i];
    const loom_amdgpu_occupancy_resource_model_t* resource_model =
        &model->resources[i];
    uint32_t allocated_units = 0;
    for (iree_host_size_t j = 0; j < resource_model->member_count; ++j) {
      const loom_amdgpu_occupancy_resource_member_model_t* member =
          &resource_model->members[j];
      uint32_t contribution_units = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_round_up_u32(
          class_summaries[member->register_class_index].allocated_units,
          member->contribution_granularity, &contribution_units));
      IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
          allocated_units, contribution_units, &allocated_units));
    }
    pressure_resource->allocated_units = allocated_units;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_finalize_resource_limit(
        resource_model->pool_units, resource_model->allocation_granularity,
        table->max_waves_per_simd, pressure_resource->allocated_units,
        &pressure_resource->rounded_units, &pressure_resource->wave_limit,
        &pressure_resource->next_cliff_units,
        &pressure_resource->units_until_next_cliff));
    if (pressure_resource->wave_limit < table->resident_waves_per_simd) {
      table->resident_waves_per_simd = pressure_resource->wave_limit;
      table->limiting_resource_kind =
          LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_PRESSURE_RESOURCE;
      table->limiting_resource_index = (uint32_t)i;
    }
  }
  if (table->max_waves_per_simd == 0) {
    table->occupancy_percent = 0;
  } else {
    table->occupancy_percent =
        (table->resident_waves_per_simd * 100u) / table->max_waves_per_simd;
  }
  return iree_ok_status();
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
    default:
      return IREE_SV("unknown");
  }
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

  const loom_amdgpu_occupancy_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_select_model(
      allocation->target.descriptor_set->descriptor_set_ordinal,
      allocation->target.descriptor_set_key, &model));

  loom_amdgpu_occupancy_register_class_t* register_classes = NULL;
  if (model->register_class_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, model->register_class_count, sizeof(*register_classes),
        (void**)&register_classes));
    memset(register_classes, 0,
           model->register_class_count * sizeof(*register_classes));
  }
  loom_amdgpu_occupancy_pressure_resource_t* pressure_resources = NULL;
  if (model->resource_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, model->resource_count, sizeof(*pressure_resources),
        (void**)&pressure_resources));
    memset(pressure_resources, 0,
           model->resource_count * sizeof(*pressure_resources));
  }
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_resolved_target(allocation->module,
                                                        &allocation->target);
  if (processor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU occupancy requires an AMDGPU processor "
                            "target record");
  }
  const uint32_t wave_size =
      allocation->target.bundle_storage.snapshot.subgroup_size;
  if (wave_size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU occupancy requires a fixed target "
                            "subgroup size");
  }
  IREE_ASSERT(
      loom_amdgpu_processor_supports_wavefront_size(processor, wave_size));

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
      .pressure_resource_count = model->resource_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_flat_workgroup_size(
      &allocation->target.bundle_storage.export_plan,
      &table.flat_workgroup_size));
  if (table.flat_workgroup_size != 0 && table.wave_size != 0) {
    table.waves_per_workgroup =
        (table.flat_workgroup_size + table.wave_size - 1) / table.wave_size;
  }
  for (iree_host_size_t i = 0; i < model->register_class_count; ++i) {
    uint16_t descriptor_reg_class_id = LOOM_LOW_ID_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_lookup_register_class(
        allocation->target.descriptor_set,
        model->register_classes[i].register_class, &descriptor_reg_class_id));
    register_classes[i] = (loom_amdgpu_occupancy_register_class_t){
        .register_class = model->register_classes[i].register_class,
        .descriptor_reg_class_id = descriptor_reg_class_id,
        .pool_units = model->register_classes[i].pool_units,
        .allocation_granularity =
            model->register_classes[i].allocation_granularity,
        .units_until_next_cliff = UINT32_MAX,
    };
  }
  for (iree_host_size_t i = 0; i < model->resource_count; ++i) {
    pressure_resources[i] = (loom_amdgpu_occupancy_pressure_resource_t){
        .resource = model->resources[i].resource,
        .pool_units = model->resources[i].pool_units,
        .allocation_granularity = model->resources[i].allocation_granularity,
        .units_until_next_cliff = UINT32_MAX,
    };
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_collect_allocations(
      allocation, register_classes, model->register_class_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_collect_spills(
      allocation, register_classes, model->register_class_count, &table));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_finalize_limits(
      model, register_classes, pressure_resources, &table));

  if (options && iree_any_bit_set(options->diagnostic_flags,
                                  LOOM_AMDGPU_OCCUPANCY_DIAGNOSTIC_SUMMARY)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_occupancy_emit_summary(&table, options->emitter));
  }
  *out_table = table;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_write_cliff_u32_or_null(
    uint32_t value, loom_output_stream_t* stream) {
  if (value == UINT32_MAX || value == 0) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

static iree_status_t loom_amdgpu_occupancy_write_register_class(
    const loom_amdgpu_occupancy_register_class_t* register_class,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"register_class\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, register_class->register_class));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"allocated_units\":%" PRIu32 ",\"rounded_units\":%" PRIu32
      ",\"pool_units\":%" PRIu32 ",\"allocation_granularity\":%" PRIu32
      ",\"wave_limit\":%" PRIu32 ",\"next_cliff_units\":",
      register_class->allocated_units, register_class->rounded_units,
      register_class->pool_units, register_class->allocation_granularity,
      register_class->wave_limit));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_u32_or_null(
      register_class->next_cliff_units, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"units_until_next_cliff\":"));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_u32_or_null(
      register_class->units_until_next_cliff, stream));
  return loom_output_stream_write_format(
      stream,
      ",\"spill_count\":%" PRIu32 ",\"spill_bytes\":%" PRIu32
      ",\"spill_store_count\":%" PRIu32 ",\"spill_reload_count\":%" PRIu32 "}",
      register_class->spill_count, register_class->spill_bytes,
      register_class->spill_store_count, register_class->spill_reload_count);
}

static iree_status_t loom_amdgpu_occupancy_write_pressure_resource(
    const loom_amdgpu_occupancy_pressure_resource_t* pressure_resource,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"resource\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, pressure_resource->resource));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"allocated_units\":%" PRIu32 ",\"rounded_units\":%" PRIu32
      ",\"pool_units\":%" PRIu32 ",\"allocation_granularity\":%" PRIu32
      ",\"wave_limit\":%" PRIu32 ",\"next_cliff_units\":",
      pressure_resource->allocated_units, pressure_resource->rounded_units,
      pressure_resource->pool_units, pressure_resource->allocation_granularity,
      pressure_resource->wave_limit));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_u32_or_null(
      pressure_resource->next_cliff_units, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"units_until_next_cliff\":"));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_u32_or_null(
      pressure_resource->units_until_next_cliff, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t loom_amdgpu_occupancy_format_json(
    const loom_amdgpu_occupancy_table_t* table,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.amdgpu.occupancy.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_diagnostic_function_name(
                   table->allocation->module, table->allocation->function_op)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, table->allocation->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, table->allocation->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"processor\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, table->processor));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"wave_size\":%" PRIu32 ",\"max_waves_per_simd\":%" PRIu32
      ",\"flat_workgroup_size\":%" PRIu32 ",\"waves_per_workgroup\":%" PRIu32
      ",\"resident_waves_per_simd\":%" PRIu32 ",\"occupancy_percent\":%" PRIu32
      ",\"limiting_resource\":",
      table->wave_size, table->max_waves_per_simd, table->flat_workgroup_size,
      table->waves_per_workgroup, table->resident_waves_per_simd,
      table->occupancy_percent));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_amdgpu_occupancy_limiting_resource_name(table)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"spill_count\":%" PRIu32 ",\"scratch_spill_bytes\":%" PRIu32
      ",\"spill_store_count\":%" PRIu32 ",\"spill_reload_count\":%" PRIu32
      ",\"register_classes\":[",
      table->spill_count, table->scratch_spill_bytes, table->spill_store_count,
      table->spill_reload_count));
  for (iree_host_size_t i = 0; i < table->register_class_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_register_class(
        &table->register_classes[i], &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "],\"pressure_resources\":["));
  for (iree_host_size_t i = 0; i < table->pressure_resource_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_pressure_resource(
        &table->pressure_resources[i], &stream));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
