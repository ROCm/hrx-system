// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/target_constraints.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"

static const loom_low_reg_class_t*
loom_low_allocation_target_constraints_reg_class_at(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  IREE_ASSERT(descriptor_set != NULL);
  IREE_ASSERT(reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
              reg_class_id < descriptor_set->reg_class_count);
  return &descriptor_set->reg_classes[reg_class_id];
}

static bool loom_low_allocation_target_constraints_location_range_overlaps(
    uint32_t lhs_base, uint32_t lhs_count, uint32_t rhs_base,
    uint32_t rhs_count) {
  const uint64_t lhs_end = (uint64_t)lhs_base + lhs_count;
  const uint64_t rhs_end = (uint64_t)rhs_base + rhs_count;
  return lhs_base < rhs_end && rhs_base < lhs_end;
}

static bool loom_low_allocation_target_constraints_value_id_is_ignored(
    loom_value_id_t value_id, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count) {
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    if (ignored_value_ids[i] == value_id) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_target_constraints_lookup_budget(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, uint32_t* out_max_units) {
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  for (iree_host_size_t i = 0; i < constraints->budget_count; ++i) {
    const loom_low_allocation_resolved_budget_t* budget =
        &constraints->budgets[i];
    if (!loom_low_allocation_storage_reg_classes_share(
            descriptor_set, budget->descriptor_reg_class_id, reg_class_id)) {
      continue;
    }
    *out_max_units = budget->max_units;
    return true;
  }
  return false;
}

static iree_string_view_t loom_low_allocation_target_constraints_reg_class_name(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id) {
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= descriptor_set->reg_class_count) {
    return IREE_SV("<unknown>");
  }
  const loom_low_reg_class_t* reg_class =
      loom_low_allocation_target_constraints_reg_class_at(descriptor_set,
                                                          reg_class_id);
  return loom_low_descriptor_set_string(descriptor_set,
                                        reg_class->name_string_offset);
}

static iree_status_t loom_low_allocation_target_constraints_emit(
    loom_low_allocation_target_constraints_t* constraints, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op ? op : constraints->function_op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(constraints->emitter, &emission));
  ++constraints->error_count;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_target_constraints_emit_unknown_budget(
    loom_low_allocation_target_constraints_t* constraints,
    iree_string_view_t register_class) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(register_class),
      loom_param_string(constraints->target->descriptor_set_key),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, constraints->function_op, LOOM_ERR_BACKEND_023, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_duplicate_budget(
    loom_low_allocation_target_constraints_t* constraints,
    iree_string_view_t register_class, uint16_t existing_reg_class_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(register_class),
      loom_param_string(loom_low_allocation_target_constraints_reg_class_name(
          constraints, existing_reg_class_id)),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, constraints->function_op, LOOM_ERR_BACKEND_024, params,
      IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_allocation_target_constraints_resolve_reg_class(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_liveness_value_class_t value_class, uint16_t* out_reg_class_id,
    const loom_low_reg_class_t** out_reg_class) {
  IREE_ASSERT_ARGUMENT(constraints);
  if (out_reg_class_id) {
    *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  }
  if (out_reg_class) {
    *out_reg_class = NULL;
  }
  const loom_low_resolved_target_t* target = constraints->target;
  if (value_class.type_kind != LOOM_TYPE_REGISTER ||
      value_class.register_descriptor_set_stable_id !=
          target->descriptor_set->stable_id ||
      value_class.register_class_id >=
          target->descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation value class is not defined by descriptor set '%.*s'",
        (int)target->descriptor_set_key.size, target->descriptor_set_key.data);
  }
  uint16_t reg_class_id = value_class.register_class_id;
  const loom_low_reg_class_t* reg_class =
      &target->descriptor_set->reg_classes[reg_class_id];
  if (out_reg_class_id) {
    *out_reg_class_id = reg_class_id;
  }
  if (out_reg_class) {
    *out_reg_class = reg_class;
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_target_constraints_reg_class_capacity(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_class_capacity_t* out_capacity) {
  IREE_ASSERT_ARGUMENT(constraints);
  IREE_ASSERT_ARGUMENT(out_capacity);
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation cannot resolve invalid register class %" PRIu16,
        reg_class_id);
  }
  const loom_low_reg_class_t* reg_class =
      loom_low_allocation_target_constraints_reg_class_at(descriptor_set,
                                                          reg_class_id);

  uint32_t budget_units = 0;
  const bool has_budget = loom_low_allocation_target_constraints_lookup_budget(
      constraints, reg_class_id, &budget_units);
  const bool has_allocatable_count = reg_class->allocatable_count != 0;
  uint32_t max_units = UINT32_MAX;
  bool is_bounded = false;
  if (has_budget && has_allocatable_count) {
    max_units = budget_units < reg_class->allocatable_count
                    ? budget_units
                    : reg_class->allocatable_count;
    is_bounded = true;
  } else if (has_budget) {
    max_units = budget_units;
    is_bounded = true;
  } else if (has_allocatable_count) {
    max_units = reg_class->allocatable_count;
    is_bounded = true;
  }

  const bool is_spillable =
      !iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
  *out_capacity = (loom_low_allocation_class_capacity_t){
      .descriptor_reg_class_id = reg_class_id,
      .location_kind =
          loom_low_allocation_storage_reg_class_location_kind(reg_class),
      .max_units = max_units,
      .alloc_unit_bits = reg_class->alloc_unit_bits,
      .spill_slot_space =
          (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
      .is_spillable = is_spillable,
      .is_bounded = is_bounded,
  };
  return iree_ok_status();
}

iree_status_t loom_low_allocation_target_constraints_class_capacity(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_liveness_value_class_t value_class,
    loom_low_allocation_class_capacity_t* out_capacity) {
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      constraints, value_class, &reg_class_id, NULL));
  return loom_low_allocation_target_constraints_reg_class_capacity(
      constraints, reg_class_id, out_capacity);
}

static iree_status_t loom_low_allocation_target_constraints_resolve_budgets(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    iree_arena_allocator_t* arena) {
  if (budget_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, budget_count, sizeof(*constraints->budgets),
      (void**)&constraints->budgets));
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  for (iree_host_size_t i = 0; i < budget_count; ++i) {
    const loom_low_allocation_budget_t* budget = &budgets[i];
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    bool found_reg_class = loom_low_descriptor_set_lookup_register_class(
        descriptor_set, budget->register_class, &reg_class_id, NULL);
    if (!found_reg_class) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_unknown_budget(
              constraints, budget->register_class));
      continue;
    }
    for (iree_host_size_t j = 0; j < constraints->budget_count; ++j) {
      if (!loom_low_allocation_storage_reg_classes_share(
              descriptor_set, constraints->budgets[j].descriptor_reg_class_id,
              reg_class_id)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_duplicate_budget(
              constraints, budget->register_class,
              constraints->budgets[j].descriptor_reg_class_id));
      reg_class_id = LOOM_LOW_REG_CLASS_NONE;
      break;
    }
    if (reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
      continue;
    }
    constraints->budgets[constraints->budget_count++] =
        (loom_low_allocation_resolved_budget_t){
            .descriptor_reg_class_id = reg_class_id,
            .max_units = budget->max_units,
        };
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_target_constraints_emit_unsupported_location_kind(
    loom_low_allocation_target_constraints_t* constraints,
    loom_low_allocation_location_kind_t location_kind,
    iree_string_view_t request_kind, const loom_op_t* diagnostic_op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(request_kind),
      loom_param_string(loom_low_allocation_location_kind_name(location_kind)),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, diagnostic_op, LOOM_ERR_BACKEND_027, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_empty_location_range(
    loom_low_allocation_target_constraints_t* constraints,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    iree_string_view_t request_kind, const loom_op_t* diagnostic_op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(request_kind),
      loom_param_string(loom_low_allocation_location_kind_name(location_kind)),
      loom_param_u32(location_base),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, diagnostic_op, LOOM_ERR_BACKEND_028, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_location_range_overflow(
    loom_low_allocation_target_constraints_t* constraints,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, uint64_t location_end,
    iree_string_view_t request_kind, const loom_op_t* diagnostic_op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(request_kind),
      loom_param_string(loom_low_allocation_location_kind_name(location_kind)),
      loom_param_u32(location_base),
      loom_param_u32(location_count),
      loom_param_u64(location_end),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, diagnostic_op, LOOM_ERR_BACKEND_029, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_allocation_target_constraints_validate_range(
    loom_low_allocation_target_constraints_t* constraints,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, iree_string_view_t request_kind,
    const loom_op_t* diagnostic_op, bool* out_valid) {
  *out_valid = false;
  if (!loom_low_allocation_location_kind_is_register_like(location_kind)) {
    return loom_low_allocation_target_constraints_emit_unsupported_location_kind(
        constraints, location_kind, request_kind, diagnostic_op);
  }
  if (location_count == 0) {
    return loom_low_allocation_target_constraints_emit_empty_location_range(
        constraints, location_kind, location_base, request_kind, diagnostic_op);
  }
  const uint64_t location_end = (uint64_t)location_base + location_count;
  if (location_end > UINT32_MAX) {
    return loom_low_allocation_target_constraints_emit_location_range_overflow(
        constraints, location_kind, location_base, location_count, location_end,
        request_kind, diagnostic_op);
  }
  *out_valid = true;
  return iree_ok_status();
}

bool loom_low_allocation_target_constraints_location_range_fits_capacity(
    const loom_low_allocation_class_capacity_t* capacity,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
  IREE_ASSERT_ARGUMENT(capacity);
  if (location_kind != capacity->location_kind || location_count == 0 ||
      (uint64_t)location_base + location_count > UINT32_MAX) {
    return false;
  }
  return !capacity->is_bounded ||
         (uint64_t)location_base + location_count <= capacity->max_units;
}

static iree_status_t
loom_low_allocation_target_constraints_emit_capacity_failure(
    loom_low_allocation_target_constraints_t* constraints, const loom_op_t* op,
    uint16_t reg_class_id, iree_string_view_t request_kind,
    uint32_t location_base, uint32_t location_count, uint64_t location_end,
    uint32_t max_units) {
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  const loom_low_reg_class_t* reg_class =
      loom_low_allocation_target_constraints_reg_class_at(descriptor_set,
                                                          reg_class_id);
  const iree_string_view_t register_class = loom_low_descriptor_set_string(
      descriptor_set, reg_class->name_string_offset);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(request_kind),
      loom_param_string(register_class),
      loom_param_u32(location_base),
      loom_param_u32(location_count),
      loom_param_u64(location_end),
      loom_param_u32(max_units),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op ? op : constraints->function_op,
      .error = LOOM_ERR_BACKEND_022,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(constraints->emitter, &emission));
  ++constraints->error_count;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_target_constraints_emit_location_kind_mismatch(
    loom_low_allocation_target_constraints_t* constraints, const loom_op_t* op,
    uint16_t reg_class_id, iree_string_view_t request_kind,
    loom_low_allocation_location_kind_t location_kind,
    loom_low_allocation_location_kind_t expected_location_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(request_kind),
      loom_param_string(loom_low_allocation_target_constraints_reg_class_name(
          constraints, reg_class_id)),
      loom_param_string(loom_low_allocation_location_kind_name(location_kind)),
      loom_param_string(
          loom_low_allocation_location_kind_name(expected_location_kind)),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, op, LOOM_ERR_BACKEND_030, params, IREE_ARRAYSIZE(params));
}

iree_status_t
loom_low_allocation_target_constraints_validate_register_location_capacity(
    loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind,
    uint32_t location_base, uint32_t location_count, iree_string_view_t subject,
    const loom_op_t* diagnostic_op, bool* out_valid) {
  IREE_ASSERT_ARGUMENT(constraints);
  IREE_ASSERT_ARGUMENT(out_valid);
  *out_valid = false;
  bool valid_range = false;
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_validate_range(
      constraints, location_kind, location_base, location_count, subject,
      diagnostic_op, &valid_range));
  if (!valid_range) {
    return iree_ok_status();
  }

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_target_constraints_reg_class_capacity(
          constraints, reg_class_id, &capacity));
  if (location_kind != capacity.location_kind) {
    return loom_low_allocation_target_constraints_emit_location_kind_mismatch(
        constraints, diagnostic_op, reg_class_id, subject, location_kind,
        capacity.location_kind);
  }
  const uint64_t location_end = (uint64_t)location_base + location_count;
  if (capacity.is_bounded && location_end > capacity.max_units) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_emit_capacity_failure(
            constraints, diagnostic_op, reg_class_id, subject, location_base,
            location_count, location_end, capacity.max_units));
    return iree_ok_status();
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_target_constraints_emit_missing_reserved_range_class(
    loom_low_allocation_target_constraints_t* constraints,
    iree_host_size_t reserved_range_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_u64(reserved_range_index),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, constraints->function_op, LOOM_ERR_BACKEND_025, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_unknown_reserved_range_class(
    loom_low_allocation_target_constraints_t* constraints,
    iree_host_size_t reserved_range_index, iree_string_view_t register_class) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_u64(reserved_range_index),
      loom_param_string(register_class),
      loom_param_string(constraints->target->descriptor_set_key),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, constraints->function_op, LOOM_ERR_BACKEND_026, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_reserved_range_overlap(
    loom_low_allocation_target_constraints_t* constraints,
    iree_host_size_t reserved_range_index,
    const loom_low_allocation_reserved_range_t* reserved_range,
    iree_host_size_t existing_reserved_range_index,
    const loom_low_allocation_resolved_reserved_range_t* existing,
    uint16_t reg_class_id) {
  const uint64_t location_end =
      (uint64_t)reserved_range->location_base + reserved_range->location_count;
  const uint64_t existing_location_end =
      (uint64_t)existing->location_base + existing->location_count;
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_u64(reserved_range_index),
      loom_param_u64(existing_reserved_range_index),
      loom_param_string(loom_low_allocation_target_constraints_reg_class_name(
          constraints, reg_class_id)),
      loom_param_u32(reserved_range->location_base),
      loom_param_u32(reserved_range->location_count),
      loom_param_u64(location_end),
      loom_param_u32(existing->location_base),
      loom_param_u32(existing->location_count),
      loom_param_u64(existing_location_end),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, constraints->function_op, LOOM_ERR_BACKEND_031, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_resolve_reserved_ranges(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_reserved_range_t* reserved_ranges,
    iree_host_size_t reserved_range_count, iree_arena_allocator_t* arena) {
  if (reserved_range_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, reserved_range_count, sizeof(*constraints->reserved_ranges),
      (void**)&constraints->reserved_ranges));
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  for (iree_host_size_t i = 0; i < reserved_range_count; ++i) {
    const loom_low_allocation_reserved_range_t* reserved_range =
        &reserved_ranges[i];
    if (iree_string_view_is_empty(reserved_range->register_class)) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_missing_reserved_range_class(
              constraints, i));
      continue;
    }
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    bool found_reg_class = loom_low_descriptor_set_lookup_register_class(
        descriptor_set, reserved_range->register_class, &reg_class_id, NULL);
    if (!found_reg_class) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_unknown_reserved_range_class(
              constraints, i, reserved_range->register_class));
      continue;
    }
    bool valid_range = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_validate_register_location_capacity(
            constraints, reg_class_id, reserved_range->location_kind,
            reserved_range->location_base, reserved_range->location_count,
            IREE_SV("reserved_range"), constraints->function_op, &valid_range));
    if (!valid_range) {
      continue;
    }
    for (iree_host_size_t j = 0; j < constraints->reserved_range_count; ++j) {
      const loom_low_allocation_resolved_reserved_range_t* existing =
          &constraints->reserved_ranges[j];
      if (reserved_range->location_kind != existing->location_kind ||
          !loom_low_allocation_storage_reg_classes_share(
              descriptor_set, reg_class_id,
              existing->descriptor_reg_class_id)) {
        continue;
      }
      if (loom_low_allocation_target_constraints_location_range_overlaps(
              reserved_range->location_base, reserved_range->location_count,
              existing->location_base, existing->location_count)) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_target_constraints_emit_reserved_range_overlap(
                constraints, i, reserved_range, j, existing, reg_class_id));
        reg_class_id = LOOM_LOW_REG_CLASS_NONE;
        break;
      }
    }
    if (reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
      continue;
    }
    constraints->reserved_ranges[constraints->reserved_range_count++] =
        (loom_low_allocation_resolved_reserved_range_t){
            .descriptor_reg_class_id = reg_class_id,
            .location_kind = reserved_range->location_kind,
            .location_base = reserved_range->location_base,
            .location_count = reserved_range->location_count,
        };
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_target_constraints_initialize(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_resolved_target_t* target,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    const loom_low_allocation_reserved_range_t* reserved_ranges,
    iree_host_size_t reserved_range_count, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena,
    loom_low_allocation_target_constraints_t* out_constraints) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_constraints);
  *out_constraints = (loom_low_allocation_target_constraints_t){
      .module = module,
      .function_op = function_op,
      .target = target,
      .emitter = emitter,
  };
  if (target->descriptor_set->reg_class_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, target->descriptor_set->reg_class_count,
        sizeof(*out_constraints->max_assigned_location_end_by_reg_class),
        (void**)&out_constraints->max_assigned_location_end_by_reg_class));
    memset(
        out_constraints->max_assigned_location_end_by_reg_class, 0,
        target->descriptor_set->reg_class_count *
            sizeof(*out_constraints->max_assigned_location_end_by_reg_class));
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_budgets(
      out_constraints, budgets, budget_count, arena));
  return loom_low_allocation_target_constraints_resolve_reserved_ranges(
      out_constraints, reserved_ranges, reserved_range_count, arena);
}

static iree_status_t
loom_low_allocation_target_constraints_emit_invalid_fixed_value_id(
    loom_low_allocation_target_constraints_t* constraints,
    iree_host_size_t fixed_value_index, loom_value_id_t value_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_u64(fixed_value_index),
      loom_param_u32(value_id),
      loom_param_u64(constraints->module->values.count),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints, constraints->function_op, LOOM_ERR_BACKEND_032, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_unallocatable_fixed_value(
    loom_low_allocation_target_constraints_t* constraints,
    loom_value_id_t value_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(constraints->module, value_id)),
      loom_param_u32(value_id),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints,
      loom_low_diagnostic_value_origin_op(constraints->module, value_id,
                                          constraints->function_op),
      LOOM_ERR_BACKEND_033, params, IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_fixed_value_unit_mismatch(
    loom_low_allocation_target_constraints_t* constraints,
    loom_value_id_t value_id, uint32_t required_unit_count,
    uint32_t location_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(constraints->module, value_id)),
      loom_param_u32(value_id),
      loom_param_u32(required_unit_count),
      loom_param_u32(location_count),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints,
      loom_low_diagnostic_value_origin_op(constraints->module, value_id,
                                          constraints->function_op),
      LOOM_ERR_BACKEND_034, params, IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_fixed_value_misalignment(
    loom_low_allocation_target_constraints_t* constraints,
    loom_value_id_t value_id, uint32_t location_base,
    uint32_t required_alignment) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(constraints->module, value_id)),
      loom_param_u32(value_id),
      loom_param_u32(location_base),
      loom_param_u32(required_alignment),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints,
      loom_low_diagnostic_value_origin_op(constraints->module, value_id,
                                          constraints->function_op),
      LOOM_ERR_BACKEND_035, params, IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_low_allocation_target_constraints_emit_duplicate_fixed_value(
    loom_low_allocation_target_constraints_t* constraints,
    iree_host_size_t fixed_value_index,
    iree_host_size_t existing_fixed_value_index, loom_value_id_t value_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_u64(fixed_value_index),
      loom_param_u64(existing_fixed_value_index),
      loom_param_string(
          loom_low_diagnostic_value_name(constraints->module, value_id)),
      loom_param_u32(value_id),
  };
  return loom_low_allocation_target_constraints_emit(
      constraints,
      loom_low_diagnostic_value_origin_op(constraints->module, value_id,
                                          constraints->function_op),
      LOOM_ERR_BACKEND_036, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_allocation_target_constraints_resolve_fixed_values(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_liveness_analysis_t* liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_low_allocation_fixed_value_t* fixed_values,
    iree_host_size_t fixed_value_count, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(constraints);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT_ARGUMENT(arena);
  if (fixed_value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, fixed_value_count, sizeof(*constraints->fixed_values),
      (void**)&constraints->fixed_values));
  for (iree_host_size_t i = 0; i < fixed_value_count; ++i) {
    const loom_low_allocation_fixed_value_t* fixed_value = &fixed_values[i];
    if (fixed_value->value_id >= constraints->module->values.count) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_invalid_fixed_value_id(
              constraints, i, fixed_value->value_id));
      continue;
    }
    const loom_value_ordinal_t value_ordinal =
        loom_local_value_domain_try_ordinal(value_domain,
                                            fixed_value->value_id);
    const loom_liveness_interval_t* interval =
        value_ordinal == LOOM_VALUE_ORDINAL_INVALID
            ? NULL
            : loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
    if (interval == NULL ||
        !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_unallocatable_fixed_value(
              constraints, fixed_value->value_id));
      continue;
    }
    if (fixed_value->location_count != interval->unit_count) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_fixed_value_unit_mismatch(
              constraints, fixed_value->value_id, interval->unit_count,
              fixed_value->location_count));
      continue;
    }
    const uint32_t alignment =
        loom_low_allocation_live_range_interval_alignment(interval);
    if (fixed_value->location_base % alignment != 0) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_emit_fixed_value_misalignment(
              constraints, fixed_value->value_id, fixed_value->location_base,
              alignment));
      continue;
    }

    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_resolve_reg_class(
            constraints, interval->value_class, &reg_class_id, NULL));
    bool valid_range = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_validate_register_location_capacity(
            constraints, reg_class_id, fixed_value->location_kind,
            fixed_value->location_base, fixed_value->location_count,
            IREE_SV("fixed_value"),
            loom_low_diagnostic_value_origin_op(constraints->module,
                                                fixed_value->value_id,
                                                constraints->function_op),
            &valid_range));
    if (!valid_range) {
      continue;
    }
    bool duplicate_fixed_value = false;
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (fixed_values[j].value_id == fixed_value->value_id) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_target_constraints_emit_duplicate_fixed_value(
                constraints, i, j, fixed_value->value_id));
        duplicate_fixed_value = true;
        break;
      }
    }
    if (duplicate_fixed_value) {
      continue;
    }
    constraints->fixed_values[constraints->fixed_value_count++] =
        (loom_low_allocation_resolved_fixed_value_t){
            .value_id = fixed_value->value_id,
            .value_ordinal = value_ordinal,
            .descriptor_reg_class_id = reg_class_id,
            .interval = interval,
            .location_kind = fixed_value->location_kind,
            .location_base = fixed_value->location_base,
            .location_count = fixed_value->location_count,
        };
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_target_constraints_emit_failure(
    loom_low_allocation_target_constraints_t* constraints, const loom_op_t* op,
    loom_liveness_value_class_t value_class, uint32_t budget_units,
    uint32_t peak_units, iree_string_view_t failure_code) {
  IREE_ASSERT_ARGUMENT(constraints);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_export_name(constraints->target)),
      loom_param_string(loom_low_diagnostic_config_key(constraints->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          constraints->module, constraints->function_op)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          constraints->target->descriptor_set, value_class)),
      loom_param_u32(budget_units),
      loom_param_u32(peak_units),
      loom_param_string(failure_code),
  };
  loom_diagnostic_emission_t emission = {
      .op = op ? op : constraints->function_op,
      .error = LOOM_ERR_BACKEND_005,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(constraints->emitter, &emission));
  ++constraints->error_count;
  return iree_ok_status();
}

const loom_low_allocation_resolved_fixed_value_t*
loom_low_allocation_target_constraints_fixed_value_for_value(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_id_t value_id) {
  IREE_ASSERT_ARGUMENT(constraints);
  for (iree_host_size_t i = 0; i < constraints->fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &constraints->fixed_values[i];
    if (fixed_value->value_id == value_id) {
      return fixed_value;
    }
  }
  return NULL;
}

static uint32_t loom_low_allocation_target_constraints_assignment_location_end(
    const loom_low_allocation_assignment_t* assignment) {
  const uint64_t location_end =
      (uint64_t)assignment->location_base + assignment->location_count;
  return location_end > UINT32_MAX ? UINT32_MAX : (uint32_t)location_end;
}

void loom_low_allocation_target_constraints_record_assignment_location_end(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_assignment_t* assignment) {
  IREE_ASSERT_ARGUMENT(constraints);
  IREE_ASSERT_ARGUMENT(assignment);
  const uint16_t reg_class_id = assignment->descriptor_reg_class_id;
  IREE_ASSERT(reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
              reg_class_id <
                  constraints->target->descriptor_set->reg_class_count);
  const loom_low_reg_class_t* reg_class =
      loom_low_allocation_target_constraints_reg_class_at(
          constraints->target->descriptor_set, reg_class_id);
  if (assignment->location_kind !=
      loom_low_allocation_storage_reg_class_location_kind(reg_class)) {
    return;
  }
  const uint32_t location_end =
      loom_low_allocation_target_constraints_assignment_location_end(
          assignment);
  if (constraints->max_assigned_location_end_by_reg_class[reg_class_id] <
      location_end) {
    constraints->max_assigned_location_end_by_reg_class[reg_class_id] =
        location_end;
  }
}

uint32_t loom_low_allocation_target_constraints_assigned_location_search_limit(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind) {
  IREE_ASSERT_ARGUMENT(constraints);
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  uint32_t max_end = 0;
  for (uint16_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    if (!loom_low_allocation_storage_reg_classes_share(descriptor_set, i,
                                                       reg_class_id)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        loom_low_allocation_target_constraints_reg_class_at(descriptor_set, i);
    if (loom_low_allocation_storage_reg_class_location_kind(reg_class) !=
        location_kind) {
      continue;
    }
    const uint32_t assignment_end =
        constraints->max_assigned_location_end_by_reg_class[i];
    if (assignment_end > max_end) {
      max_end = assignment_end;
    }
  }
  for (iree_host_size_t i = 0; i < constraints->fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &constraints->fixed_values[i];
    if (fixed_value->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_storage_reg_classes_share(
            descriptor_set, fixed_value->descriptor_reg_class_id,
            reg_class_id)) {
      continue;
    }
    uint64_t fixed_value_end =
        (uint64_t)fixed_value->location_base + fixed_value->location_count;
    if (fixed_value_end > UINT32_MAX) {
      return UINT32_MAX;
    }
    if ((uint32_t)fixed_value_end > max_end) {
      max_end = (uint32_t)fixed_value_end;
    }
  }
  for (iree_host_size_t i = 0; i < constraints->reserved_range_count; ++i) {
    const loom_low_allocation_resolved_reserved_range_t* reserved_range =
        &constraints->reserved_ranges[i];
    if (reserved_range->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_storage_reg_classes_share(
            descriptor_set, reserved_range->descriptor_reg_class_id,
            reg_class_id)) {
      continue;
    }
    uint64_t reserved_range_end = (uint64_t)reserved_range->location_base +
                                  reserved_range->location_count;
    if (reserved_range_end > UINT32_MAX) {
      return UINT32_MAX;
    }
    if ((uint32_t)reserved_range_end > max_end) {
      max_end = (uint32_t)reserved_range_end;
    }
  }
  return max_end;
}

bool loom_low_allocation_target_constraints_fixed_value_conflicts(
    const loom_low_allocation_target_constraints_t* constraints,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(constraints);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(candidate);
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  for (iree_host_size_t i = 0; i < constraints->fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &constraints->fixed_values[i];
    if (loom_low_allocation_target_constraints_value_id_is_ignored(
            fixed_value->value_id, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (fixed_value->location_kind != candidate->location_kind) {
      continue;
    }
    if (!loom_low_allocation_storage_reg_classes_share(
            descriptor_set, fixed_value->descriptor_reg_class_id,
            candidate->descriptor_reg_class_id)) {
      continue;
    }
    const loom_low_allocation_assignment_t fixed_assignment = {
        .value_id = fixed_value->value_id,
        .value_class = fixed_value->interval->value_class,
        .descriptor_reg_class_id = fixed_value->descriptor_reg_class_id,
        .start_point = fixed_value->interval->start_point,
        .end_point = loom_low_allocation_live_range_interval_storage_end_point(
            fixed_value->interval),
        .unit_count = fixed_value->interval->unit_count,
        .location_kind = fixed_value->location_kind,
        .location_base = fixed_value->location_base,
        .location_count = fixed_value->location_count,
        .unit_end_point_start =
            loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
                unit_liveness, liveness, fixed_value->value_ordinal),
    };
    if (loom_low_allocation_live_range_assignments_conflict(
            descriptor_set, liveness, unit_liveness->end_points,
            unit_liveness->end_point_count, &fixed_assignment, candidate)) {
      return true;
    }
  }
  return false;
}

bool loom_low_allocation_target_constraints_reserved_range_conflicts(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind,
    uint32_t location_base, uint32_t location_count) {
  IREE_ASSERT_ARGUMENT(constraints);
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  for (iree_host_size_t i = 0; i < constraints->reserved_range_count; ++i) {
    const loom_low_allocation_resolved_reserved_range_t* reserved_range =
        &constraints->reserved_ranges[i];
    if (reserved_range->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_storage_reg_classes_share(
            descriptor_set, reserved_range->descriptor_reg_class_id,
            reg_class_id)) {
      continue;
    }
    if (loom_low_allocation_target_constraints_location_range_overlaps(
            reserved_range->location_base, reserved_range->location_count,
            location_base, location_count)) {
      return true;
    }
  }
  return false;
}
