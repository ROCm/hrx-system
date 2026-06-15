// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_contract.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/target/facts.h"
#include "loom/target/launch.h"

bool loom_target_function_contract_bundles_compatible(
    const loom_target_bundle_t* module_bundle,
    const loom_target_bundle_t* selected_bundle) {
  IREE_ASSERT_ARGUMENT(module_bundle);
  IREE_ASSERT_ARGUMENT(selected_bundle);
  return module_bundle->snapshot->codegen_format ==
             selected_bundle->snapshot->codegen_format &&
         module_bundle->snapshot->artifact_format ==
             selected_bundle->snapshot->artifact_format &&
         iree_string_view_equal(module_bundle->config->contract_set_key,
                                selected_bundle->config->contract_set_key);
}

void loom_target_function_contract_apply_compatible_selection(
    const loom_target_bundle_t* selected_bundle,
    loom_target_bundle_storage_t* bundle_storage) {
  IREE_ASSERT_ARGUMENT(selected_bundle);
  IREE_ASSERT_ARGUMENT(bundle_storage);
  bundle_storage->snapshot = *selected_bundle->snapshot;
  bundle_storage->config = *selected_bundle->config;
  bundle_storage->bundle = *selected_bundle;
  bundle_storage->bundle.snapshot = &bundle_storage->snapshot;
  bundle_storage->bundle.export_plan = &bundle_storage->export_plan;
  bundle_storage->bundle.config = &bundle_storage->config;
}

static iree_string_view_t loom_target_function_contract_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return IREE_SV("<unknown>");
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_target_function_contract_symbol_ref_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return loom_target_function_contract_string_from_id(module, symbol->name_id);
}

static iree_status_t loom_target_function_contract_emit(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_target_function_contract_reject(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    bool* out_valid) {
  *out_valid = false;
  return loom_target_function_contract_emit(
      diagnostic_emitter, func_facts->func_op, error, params, param_count);
}

static bool loom_target_function_contract_mul_u64(uint64_t lhs, uint64_t rhs,
                                                  uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) {
    *out_result = 0;
    return false;
  }
  *out_result = lhs * rhs;
  return true;
}

static iree_status_t loom_target_function_contract_apply_abi_attrs(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, loom_named_attr_slice_t attrs,
    bool* out_valid) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    iree_string_view_t name =
        loom_target_function_contract_string_from_id(module, entry->name_id);
    const loom_diagnostic_param_t params[] = {
        loom_param_string(func_facts->name),
        loom_param_string(name),
    };
    return loom_target_function_contract_reject(
        diagnostic_emitter, func_facts, LOOM_ERR_TARGET_021, params,
        IREE_ARRAYSIZE(params), out_valid);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_function_contract_lookup_target(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    const loom_target_symbol_facts_t** out_target) {
  *out_target = NULL;
  if (!loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(func_facts->name),
    };
    return loom_target_function_contract_reject(
        diagnostic_emitter, func_facts, LOOM_ERR_TARGET_026, params,
        IREE_ARRAYSIZE(params), out_valid);
  }
  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, func_facts->target_symbol, &target_base_facts));
  const loom_target_symbol_facts_t* target =
      loom_target_symbol_facts_cast(target_base_facts);
  if (!target) {
    iree_string_view_t target_name =
        loom_target_function_contract_symbol_ref_name(
            module, func_facts->target_symbol);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified function '@%.*s' target '@%.*s' does not resolve to target "
        "facts",
        (int)func_facts->name.size, func_facts->name.data,
        (int)target_name.size, target_name.data);
  }
  *out_target = target;
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_target_function_contract_reject_constraint(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    iree_string_view_t constraint_key, bool* out_valid) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(target_name),
      loom_param_string(constraint_key),
  };
  return loom_target_function_contract_reject(
      diagnostic_emitter, func_facts, LOOM_ERR_TARGET_022, params,
      IREE_ARRAYSIZE(params), out_valid);
}

static iree_status_t loom_target_function_contract_reject_target_constraint(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts,
    const loom_target_symbol_facts_t* target, iree_string_view_t constraint_key,
    bool* out_valid) {
  return loom_target_function_contract_reject_constraint(
      diagnostic_emitter, func_facts, target->name, constraint_key, out_valid);
}

static iree_status_t loom_target_function_contract_reject_flat_range(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    iree_string_view_t constraint_key, uint32_t minimum, uint32_t maximum,
    uint32_t limit, bool* out_valid) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(target_name),
      loom_param_string(constraint_key),
      loom_param_u32(minimum),
      loom_param_u32(maximum),
      loom_param_u32(limit),
  };
  return loom_target_function_contract_reject(
      diagnostic_emitter, func_facts, LOOM_ERR_TARGET_023, params,
      IREE_ARRAYSIZE(params), out_valid);
}

static iree_status_t loom_target_function_contract_reject_dimension_limit(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    iree_string_view_t axis, uint32_t size, uint32_t limit, bool* out_valid) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(target_name),
      loom_param_string(axis),
      loom_param_u32(size),
      loom_param_u32(limit),
  };
  return loom_target_function_contract_reject(
      diagnostic_emitter, func_facts, LOOM_ERR_TARGET_024, params,
      IREE_ARRAYSIZE(params), out_valid);
}

static iree_status_t loom_target_function_contract_reject_flat_size(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    iree_string_view_t constraint_key, uint64_t size, uint32_t minimum,
    uint32_t maximum, uint32_t limit, bool* out_valid) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(target_name),
      loom_param_string(constraint_key),
      loom_param_u64(size),
      loom_param_u32(minimum),
      loom_param_u32(maximum),
      loom_param_u32(limit),
  };
  return loom_target_function_contract_reject(
      diagnostic_emitter, func_facts, LOOM_ERR_TARGET_025, params,
      IREE_ARRAYSIZE(params), out_valid);
}

static iree_status_t loom_target_function_contract_validate_workgroup_size(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    const loom_target_snapshot_t* snapshot,
    const loom_target_workgroup_size_t* required_workgroup_size,
    uint64_t* out_flat_size, bool* out_valid) {
  *out_flat_size = 0;
  if (loom_target_workgroup_size_is_partial(required_workgroup_size)) {
    return loom_target_function_contract_reject_constraint(
        diagnostic_emitter, func_facts, target_name,
        IREE_SV("required_workgroup_size.complete"), out_valid);
  }
  if (loom_target_workgroup_size_is_empty(required_workgroup_size)) {
    return iree_ok_status();
  }
  const loom_target_workgroup_size_t* limit = &snapshot->max_workgroup_size;
  if (limit->x != 0 && required_workgroup_size->x > limit->x) {
    return loom_target_function_contract_reject_dimension_limit(
        diagnostic_emitter, func_facts, target_name, IREE_SV("x"),
        required_workgroup_size->x, limit->x, out_valid);
  }
  if (limit->y != 0 && required_workgroup_size->y > limit->y) {
    return loom_target_function_contract_reject_dimension_limit(
        diagnostic_emitter, func_facts, target_name, IREE_SV("y"),
        required_workgroup_size->y, limit->y, out_valid);
  }
  if (limit->z != 0 && required_workgroup_size->z > limit->z) {
    return loom_target_function_contract_reject_dimension_limit(
        diagnostic_emitter, func_facts, target_name, IREE_SV("z"),
        required_workgroup_size->z, limit->z, out_valid);
  }
  uint64_t flat_size = 1;
  if (!loom_target_function_contract_mul_u64(
          flat_size, required_workgroup_size->x, &flat_size) ||
      !loom_target_function_contract_mul_u64(
          flat_size, required_workgroup_size->y, &flat_size) ||
      !loom_target_function_contract_mul_u64(
          flat_size, required_workgroup_size->z, &flat_size)) {
    return loom_target_function_contract_reject_constraint(
        diagnostic_emitter, func_facts, target_name,
        IREE_SV("required_flat_workgroup_size.u64"), out_valid);
  }
  const uint32_t max_flat_workgroup_size = snapshot->max_flat_workgroup_size;
  if (max_flat_workgroup_size != 0 && flat_size > max_flat_workgroup_size) {
    return loom_target_function_contract_reject_flat_size(
        diagnostic_emitter, func_facts, target_name,
        IREE_SV("required_flat_workgroup_size.target_limit"), flat_size,
        /*minimum=*/0, /*maximum=*/0, max_flat_workgroup_size, out_valid);
  }
  *out_flat_size = flat_size;
  return iree_ok_status();
}

static iree_status_t loom_target_function_contract_validate_hal_kernel(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel, bool* out_valid) {
  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  uint64_t flat_size = 0;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_validate_workgroup_size(
      diagnostic_emitter, func_facts, target_name, snapshot, required,
      &flat_size, out_valid));
  if (!*out_valid) {
    return iree_ok_status();
  }
  const uint32_t max_flat_workgroup_size = snapshot->max_flat_workgroup_size;
  const uint32_t flat_min = hal_kernel->flat_workgroup_size_min;
  const uint32_t flat_max = hal_kernel->flat_workgroup_size_max;
  if ((flat_min == 0) != (flat_max == 0)) {
    return loom_target_function_contract_reject_constraint(
        diagnostic_emitter, func_facts, target_name,
        IREE_SV("flat_workgroup_range.complete"), out_valid);
  }
  if (flat_min != 0) {
    if (flat_min > flat_max) {
      return loom_target_function_contract_reject_flat_range(
          diagnostic_emitter, func_facts, target_name,
          IREE_SV("flat_workgroup_range.ordered"), flat_min, flat_max,
          max_flat_workgroup_size, out_valid);
    }
    if (max_flat_workgroup_size != 0 && flat_max > max_flat_workgroup_size) {
      return loom_target_function_contract_reject_flat_range(
          diagnostic_emitter, func_facts, target_name,
          IREE_SV("flat_workgroup_range.target_limit"), flat_min, flat_max,
          max_flat_workgroup_size, out_valid);
    }
  }
  if (loom_target_workgroup_size_is_empty(required)) {
    return iree_ok_status();
  }
  if (flat_min != 0 && (flat_size < flat_min || flat_size > flat_max)) {
    return loom_target_function_contract_reject_flat_size(
        diagnostic_emitter, func_facts, target_name,
        IREE_SV("required_flat_workgroup_size.range"), flat_size, flat_min,
        flat_max, max_flat_workgroup_size, out_valid);
  }
  return iree_ok_status();
}

iree_status_t loom_target_function_contract_apply_hal_workgroup_size(
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    const loom_target_workgroup_size_t* required_workgroup_size,
    iree_diagnostic_emitter_t diagnostic_emitter,
    loom_target_bundle_storage_t* bundle_storage, bool* out_valid) {
  *out_valid = false;
  if (bundle_storage->export_plan.abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_target_function_contract_reject_constraint(
        diagnostic_emitter, func_facts, target_name, IREE_SV("abi.hal_kernel"),
        out_valid);
  }
  bundle_storage->export_plan.hal_kernel.required_workgroup_size =
      *required_workgroup_size;
  *out_valid = true;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_validate_hal_kernel(
      diagnostic_emitter, func_facts, target_name, &bundle_storage->snapshot,
      &bundle_storage->export_plan.hal_kernel, out_valid));
  if (!*out_valid) {
    return iree_ok_status();
  }
  loom_target_bundle_storage_rebind(bundle_storage);
  return iree_ok_status();
}

iree_status_t loom_target_function_contract_resolve(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_target_bundle_storage_t* out_bundle_storage) {
  const loom_target_symbol_facts_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_lookup_target(
      module, fact_table, func_facts, diagnostic_emitter, out_valid, &target));
  if (!*out_valid) {
    return iree_ok_status();
  }
  return loom_target_function_contract_resolve_from_bundle(
      module, func_facts, target->name, &target->storage.bundle,
      diagnostic_emitter, out_valid, out_bundle_storage);
}

iree_status_t loom_target_function_contract_resolve_from_bundle(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_string_view_t target_name, const loom_target_bundle_t* base_bundle,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_target_bundle_storage_t* out_bundle_storage) {
  memset(out_bundle_storage, 0, sizeof(*out_bundle_storage));
  *out_valid = false;

  *out_bundle_storage = (loom_target_bundle_storage_t){
      .snapshot = *base_bundle->snapshot,
      .export_plan = *base_bundle->export_plan,
      .config = *base_bundle->config,
      .bundle = *base_bundle,
  };
  loom_target_bundle_storage_rebind(out_bundle_storage);
  out_bundle_storage->export_plan.name = func_facts->name;
  out_bundle_storage->export_plan.export_symbol = iree_string_view_empty();
  if (func_facts->has_abi) {
    out_bundle_storage->export_plan.abi_kind = func_facts->abi_kind;
  }
  if (out_bundle_storage->export_plan.abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
    return loom_target_function_contract_reject_constraint(
        diagnostic_emitter, func_facts, target_name, IREE_SV("abi.concrete"),
        out_valid);
  }
  *out_valid = true;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_apply_abi_attrs(
      module, func_facts, diagnostic_emitter, func_facts->abi_attrs,
      out_valid));
  if (!*out_valid) {
    return iree_ok_status();
  }
  if (out_bundle_storage->export_plan.abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    IREE_RETURN_IF_ERROR(loom_target_function_contract_validate_hal_kernel(
        diagnostic_emitter, func_facts, target_name,
        &out_bundle_storage->snapshot,
        &out_bundle_storage->export_plan.hal_kernel, out_valid));
    if (!*out_valid) {
      return iree_ok_status();
    }
  }
  if (!iree_string_view_is_empty(func_facts->export_symbol)) {
    out_bundle_storage->export_plan.export_symbol = func_facts->export_symbol;
  }
  if (func_facts->has_export_linkage) {
    out_bundle_storage->export_plan.linkage = func_facts->export_linkage;
  }
  *out_valid = true;
  return iree_ok_status();
}
