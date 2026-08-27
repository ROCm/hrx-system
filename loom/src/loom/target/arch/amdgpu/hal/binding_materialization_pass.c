// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal/binding_materialization_pass.h"

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/hal/binding_materialization.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/function_version.h"

#define LOOM_AMDGPU_HAL_KERNEL_ABI_STATISTICS(V, statistics_type) \
  V(statistics_type, errors, "errors",                            \
    "Number of AMDGPU HAL ABI errors emitted.")                   \
  V(statistics_type, functions, "functions",                      \
    "Number of target-low functions materialized.")               \
  V(statistics_type, bindings, "bindings",                        \
    "Number of HAL buffer bindings materialized.")                \
  V(statistics_type, direct_args, "direct-args",                  \
    "Number of direct kernarg values materialized.")              \
  V(statistics_type, descriptors, "descriptors",                  \
    "Number of HAL buffer descriptor pseudos materialized.")

LOOM_PASS_STATISTICS_DEFINE(loom_amdgpu_hal_kernel_abi_statistics,
                            loom_amdgpu_hal_kernel_abi_statistics_t,
                            LOOM_AMDGPU_HAL_KERNEL_ABI_STATISTICS)

static const loom_pass_info_t
    loom_amdgpu_materialize_hal_kernel_abi_pass_info_storage = {
        .name = IREE_SVL("amdgpu-materialize-hal-kernel-abi"),
        .description = IREE_SVL("Materialize AMDGPU HAL kernel ABI resources."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout = &loom_amdgpu_hal_kernel_abi_statistics_layout,
};

static const loom_pass_info_t
    loom_amdgpu_materialize_hal_buffer_descriptors_pass_info_storage = {
        .name = IREE_SVL("amdgpu-materialize-hal-buffer-descriptors"),
        .description =
            IREE_SVL("Materialize AMDGPU HAL buffer descriptor pseudos."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout = &loom_amdgpu_hal_kernel_abi_statistics_layout,
};

const loom_pass_info_t* loom_amdgpu_materialize_hal_kernel_abi_pass_info(void) {
  return &loom_amdgpu_materialize_hal_kernel_abi_pass_info_storage;
}

const loom_pass_info_t*
loom_amdgpu_materialize_hal_buffer_descriptors_pass_info(void) {
  return &loom_amdgpu_materialize_hal_buffer_descriptors_pass_info_storage;
}

static bool loom_amdgpu_materialize_hal_kernel_abi_matches(
    const loom_low_resolved_target_t* target) {
  return target->descriptor_set != NULL &&
         loom_low_resolved_target_bundle(target)->export_plan->abi_kind ==
             LOOM_TARGET_ABI_HAL_KERNEL &&
         loom_amdgpu_target_facts_cast(target->target_facts) != NULL;
}

iree_status_t loom_amdgpu_materialize_hal_kernel_abi_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op)) {
    return iree_ok_status();
  }

  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, pass->arena);
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, function.op,
      loom_target_function_version_target_facts(pass->function_version),
      descriptor_registry, pass->diagnostic_emitter, &target));
  if (!loom_amdgpu_materialize_hal_kernel_abi_matches(&target)) {
    return iree_ok_status();
  }

  loom_amdgpu_hal_kernel_abi_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_low(
      module, function.op, target.descriptor_set, /*max_errors=*/20,
      pass->diagnostic_emitter, &verify_result, pass->arena));
  loom_amdgpu_hal_kernel_abi_statistics_t* statistics =
      loom_amdgpu_hal_kernel_abi_statistics(pass);
  statistics->errors += verify_result.error_count;
  if (verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_amdgpu_hal_binding_materialization_result_t materialization = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize(
      module, function.op, target.descriptor_set, &materialization,
      pass->arena));
  ++statistics->functions;
  statistics->bindings += materialization.materialized_binding_count;
  statistics->direct_args += materialization.materialized_direct_arg_count;
  statistics->descriptors += materialization.materialized_descriptor_count;
  if (materialization.changed) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_materialize_hal_buffer_descriptors_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op)) {
    return iree_ok_status();
  }

  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, pass->arena);
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, function.op,
      loom_target_function_version_target_facts(pass->function_version),
      descriptor_registry, pass->diagnostic_emitter, &target));
  if (!loom_amdgpu_materialize_hal_kernel_abi_matches(&target)) {
    return iree_ok_status();
  }

  iree_host_size_t materialized_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_buffer_descriptors(
      module, function.op, target.descriptor_set, &materialized_count,
      pass->arena));
  if (materialized_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_hal_kernel_abi_statistics_t* statistics =
      loom_amdgpu_hal_kernel_abi_statistics(pass);
  ++statistics->functions;
  statistics->descriptors += materialized_count;
  loom_pass_mark_changed(pass);
  return iree_ok_status();
}
