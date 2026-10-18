// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/configuration.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/pass/report.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/configuration.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/pass_environment.h"

static iree_status_t loom_aie2p_configuration_check_run(loom_pass_t* pass,
                                                        loom_module_t* module) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_function_version_list_t* function_versions =
      loom_target_pass_capability_function_versions(capability);
  const loom_target_low_descriptor_registry_t registry =
      loom_target_environment_low_descriptor_registry(
          loom_target_pass_capability_target_environment(capability));
  loom_target_function_version_snapshot_t versions = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, function_versions, pass->arena, &versions));

  // This boundary fixture has one retained configuration and one core worker.
  // Its source lowering supplies real proofs; no test-authored alias metadata
  // participates in either compilation route.
  const loom_op_t* entry_op = NULL;
  loom_op_t* worker_op = NULL;
  iree_host_size_t entry_count = 0;
  iree_host_size_t worker_count = 0;
  const loom_target_function_version_t* worker_version = NULL;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    loom_op_t* op = symbol->defining_op;
    if (!op || !loom_low_func_def_isa(op)) {
      continue;
    }
    const loom_func_like_t function = loom_func_like_cast(module, op);
    const iree_string_view_t contract = loom_string_table_get(
        &module->strings, loom_func_like_repr_contract(function));
    if (iree_string_view_equal(contract,
                               IREE_SV("amd.xdna.aie2p.configuration")) &&
        iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_RETAIN)) {
      ++entry_count;
      entry_op = op;
    } else if (iree_string_view_equal(contract,
                                      IREE_SV("amd.xdna.aie2p.core"))) {
      ++worker_count;
      worker_op = op;
      worker_version = loom_target_function_version_snapshot_at(
          &versions, loom_func_like_callee(function).symbol_id);
    }
  }
  if (entry_count != 1 || worker_count != 1 || !worker_version ||
      !worker_version->memory_accesses) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "configuration memory check requires one entry and one worker with "
        "source memory proofs");
  }

  loom_symbol_fact_table_t symbol_facts;
  loom_symbol_fact_table_initialize(&symbol_facts, pass->arena);
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, entry_op, NULL, &registry.registry,
      pass->diagnostic_emitter, &target));
  const loom_aie2p_target_facts_t* entry_facts =
      loom_aie2p_target_facts_cast(target.target_facts);
  if (!entry_facts || !entry_facts->device_profile) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "configuration memory check requires a device profile");
  }

  loom_target_compile_report_t configuration_report;
  loom_target_compile_report_initialize(&configuration_report,
                                        iree_allocator_system());
  const loom_aie2p_xdna_artifact_request_t request = {
      .module = module,
      .function_versions = function_versions,
      .low_descriptor_registry = &registry.registry,
      .compile_report = &configuration_report,
      .diagnostic_emitter = pass->diagnostic_emitter,
      .scratch_arena = pass->arena,
      .allocator = iree_allocator_system(),
  };
  loom_aie2p_configuration_plan_t* plan = NULL;
  loom_aie2p_xdna_entry_t entry = {0};
  bool accepted = false;
  iree_status_t status = loom_aie2p_configuration_plan_build(
      &request, &versions, entry_op, entry_facts->device_profile, &plan, &entry,
      &accepted);
  if (iree_status_is_ok(status) && accepted) {
    status = loom_aie2p_configuration_plan_materialize(plan, &accepted);
  }

  loom_target_compile_report_t worker_report;
  loom_target_compile_report_initialize(&worker_report,
                                        iree_allocator_system());
  const loom_aie2p_leaf_compile_options_t worker_options = {
      .descriptor_registry = &registry.registry,
      .function_target_facts = worker_version->function_target_facts,
      .memory_accesses = worker_version->memory_accesses,
      .loop_execution = worker_version->loop_execution,
      .diagnostic_emitter = pass->diagnostic_emitter,
      .compile_report = &worker_report,
  };
  loom_aie2p_leaf_contribution_t contribution = {0};
  if (iree_status_is_ok(status) && accepted) {
    status = loom_aie2p_leaf_compile(module, worker_op, &worker_options,
                                     pass->arena, &accepted, &contribution);
  }
  if (iree_status_is_ok(status) && accepted) {
    const loom_pass_report_detail_field_t fields[] = {
        loom_pass_report_detail_bool_field(
            IREE_SV("memory_dependencies_preserved"),
            configuration_report.schedule_dependency_count ==
                worker_report.schedule_dependency_count),
    };
    status = loom_pass_report_append_detail(pass, IREE_SV("worker_facts"),
                                            fields, IREE_ARRAYSIZE(fields));
  }
  loom_target_compile_report_deinitialize(&worker_report);
  loom_target_compile_report_deinitialize(&configuration_report);
  return status;
}

static const loom_pass_info_t* loom_aie2p_configuration_check_info(void) {
  static const loom_pass_info_t info = {
      .name = IREE_SVL("test.aie2p-configuration-memory"),
      .description =
          IREE_SVL("Compare configuration and direct worker analysis"),
      .kind = LOOM_PASS_MODULE,
  };
  return &info;
}

static const loom_pass_descriptor_t kConfigurationCheckPasses[] = {{
    .key = IREE_SVL("test.aie2p-configuration-memory"),
    .info = loom_aie2p_configuration_check_info,
    .module_run = loom_aie2p_configuration_check_run,
}};

static const loom_pass_registry_t kConfigurationCheckRegistry = {
    .descriptors = kConfigurationCheckPasses,
    .descriptor_count = IREE_ARRAYSIZE(kConfigurationCheckPasses),
};

const loom_target_provider_t loom_aie2p_configuration_check_provider = {
    .pass_registry = &kConfigurationCheckRegistry,
};
