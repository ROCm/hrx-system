// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/src/program_plan.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/binding/c/src/compile_report.h"
#include "loom/binding/c/src/context.h"
#include "loom/binding/c/src/diagnostic.h"
#include "loom/binding/c/src/emit.h"
#include "loom/binding/c/src/module.h"
#include "loom/binding/c/src/option_chain.h"
#include "loom/binding/c/src/result.h"
#include "loom/binding/c/src/source.h"
#include "loom/binding/c/src/target.h"
#include "loom/binding/c/src/workspace.h"
#include "loom/binding/c/target/cmd/provider.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/target/arch/cmd/package.h"
#include "loom/target/facts.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/report.h"
#include "loomc/iree.h"
#include "loomc/target/cmd/program_plan.h"

typedef struct loomc_cmd_program_plan_root_storage_t {
  // Public root name borrowed from the package unit module.
  loomc_string_view_t name;

  // Root-local executable requirements in portable package slot order.
  const loomc_cmd_program_plan_executable_requirement_t*
      executable_requirements;

  // Number of entries in |executable_requirements|.
  loomc_host_size_t executable_requirement_count;
} loomc_cmd_program_plan_root_storage_t;

typedef struct loomc_cmd_program_plan_dependency_storage_t {
  // Stable compilation-unit name borrowed from the first module export.
  loomc_string_view_t name;

  // Export names in the unit-local order referenced by package entries.
  loomc_string_view_t* export_names;

  // Number of entries in |export_names|.
  uint32_t export_count;

  // Public target artifact format fixed by the unit's sealed target facts.
  loomc_string_view_t artifact_format;

  // Generic plan unit producing this dependency executable.
  loomc_program_plan_unit_t unit;
} loomc_cmd_program_plan_dependency_storage_t;

typedef struct loomc_cmd_program_plan_storage_t {
  // Generic program-plan provider header.
  loomc_program_plan_storage_t base;

  // Workspace retained for the compiler plan's shared arena block pool.
  loomc_workspace_t* workspace;

  // Compiler-owned command roots, layouts, and dependency graph.
  //
  // Module pointers borrow from |unit_modules| after construction. All other
  // plan storage is owned by |plan.arena|.
  loom_cmd_program_plan_t plan;

  // Public unit modules in generic plan-unit order.
  loomc_module_t** unit_modules;

  // Number of entries in |unit_modules|.
  loomc_host_size_t unit_count;

  // Root metadata in generic plan-root order.
  loomc_cmd_program_plan_root_storage_t* roots;

  // Dependency metadata in compiler plan dependency order.
  loomc_cmd_program_plan_dependency_storage_t* dependencies;

  // Shared package unit containing every selected command root.
  loomc_program_plan_unit_t package_unit;

  // Shared launch-config unit, or an invalid token when no root requires one.
  loomc_program_plan_unit_t launch_config_unit;
} loomc_cmd_program_plan_storage_t;

typedef struct loomc_cmd_diagnostic_capture_t {
  // Result receiving materialized source diagnostics.
  loomc_result_t* result;

  // Number of error diagnostics observed.
  uint32_t error_count;
} loomc_cmd_diagnostic_capture_t;

static iree_status_t loomc_cmd_capture_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_cmd_diagnostic_capture_t* capture =
      (loomc_cmd_diagnostic_capture_t*)user_data;
  if (emission != NULL && emission->error != NULL &&
      emission->error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++capture->error_count;
  }
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
}

static loomc_string_view_t loomc_cmd_symbol_name(const loom_module_t* module,
                                                 const loom_op_t* op) {
  const loom_symbol_ref_t symbol_ref =
      loom_func_like_callee(loom_func_like_cast(module, (loom_op_t*)op));
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return loomc_string_view_from_iree(module->strings.entries[symbol->name_id]);
}

static loom_op_t* loomc_cmd_find_symbol(loom_module_t* module,
                                        loomc_string_view_t name) {
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, iree_string_view_from_loomc(name));
  if (name_id == LOOM_STRING_ID_INVALID) return NULL;
  const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) return NULL;
  return module->symbols.entries[symbol_id].defining_op;
}

static loomc_status_t loomc_cmd_take_internal_module(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loom_module_t** internal_module, loomc_allocator_t allocator,
    loomc_module_t** out_module) {
  *out_module = NULL;
  loomc_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_create_empty(context, workspace, allocator, &module));
  loomc_status_t status =
      loomc_module_set_loom_module(module, *internal_module);
  if (loomc_status_is_ok(status)) {
    *internal_module = NULL;
    *out_module = module;
    module = NULL;
  }
  loomc_module_release(module);
  return status;
}

static loomc_status_t loomc_cmd_finish_unit_failure(
    loomc_result_t* result, loomc_status_t status,
    loomc_string_view_t diagnostic_code, loomc_result_t** out_result) {
  if (!loomc_status_is_result_diagnostic(status)) {
    loomc_result_release(result);
    return status;
  }
  loomc_status_t add_status = loomc_result_fail_status_diagnostic_consume(
      result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, diagnostic_code,
      status);
  if (loomc_status_is_ok(add_status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return add_status;
}

static loomc_status_t loomc_cmd_compile_module(
    const loomc_module_t* source_module, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, const loomc_pass_program_t* pass_program,
    loomc_string_view_t module_name, loomc_allocator_t allocator,
    loomc_module_t** out_module, loomc_result_t** out_result) {
  *out_module = NULL;
  *out_result = NULL;
  loomc_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_clone(source_module, workspace, allocator, &module));
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .module_name = module_name,
  };
  loomc_result_t* result = NULL;
  loomc_status_t status =
      loomc_compile_module(compiler, workspace, pass_program, module,
                           &compile_options, allocator, &result);
  if (loomc_status_is_ok(status)) {
    *out_module = module;
    *out_result = result;
    module = NULL;
    result = NULL;
  }
  loomc_module_release(module);
  loomc_result_release(result);
  return status;
}

static loomc_status_t loomc_cmd_add_compile_report(
    loomc_result_t* result, const loomc_compile_report_options_t* options,
    loomc_string_view_t default_identifier,
    const loom_target_compile_report_t* report) {
  if (options == NULL || options->mode == LOOMC_COMPILE_REPORT_MODE_NONE) {
    return loomc_ok_status();
  }
  loomc_string_view_t report_identifier = loomc_string_view_empty();
  loomc_status_t status = loomc_compile_report_make_identifier(
      options, default_identifier, loomc_result_allocator(result),
      &report_identifier);
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_report_add_artifact(result, options->mode,
                                               report_identifier, report);
  }
  loomc_allocator_free(loomc_result_allocator(result),
                       (void*)report_identifier.data);
  return status;
}

static loomc_status_t loomc_cmd_serialize_module(const loomc_module_t* module,
                                                 loomc_string_view_t identifier,
                                                 loomc_allocator_t allocator,
                                                 loomc_byte_span_t* out_data) {
  *out_data = loomc_byte_span_empty();
  loomc_source_t* source = NULL;
  loomc_status_t status = loomc_module_serialize_internal_bytecode_to_source(
      module, loomc_module_const_loom_module(module), identifier, allocator,
      &source);
  if (loomc_status_is_ok(status)) {
    status = loomc_source_take_contents(source, out_data);
  }
  loomc_source_release(source);
  return status;
}

static loomc_status_t loomc_cmd_resolve_dependency_artifact_format(
    const loom_target_environment_t* target_environment,
    const loom_module_t* module, loom_op_t* kernel_op,
    loom_symbol_fact_table_t* fact_table,
    loomc_string_view_t* out_artifact_format) {
  *out_artifact_format = loomc_string_view_empty();
  const loom_symbol_ref_t target_ref =
      loom_func_like_target(loom_func_like_cast(module, kernel_op));
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "command dependency has no sealed target");
  }

  loom_symbol_fact_table_reset(fact_table);
  const loom_symbol_facts_base_t* base_facts = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_symbol_fact_table_lookup_ref(
          fact_table, module, target_ref, &base_facts)));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  const loom_target_artifact_format_t target_artifact_format =
      loom_target_facts_bundle(target_facts->projection)
          ->snapshot->artifact_format;

  const loom_target_emitter_t* selected_emitter = NULL;
  const loom_target_emitter_list_t emitters =
      loom_target_environment_emitter_list(target_environment);
  for (iree_host_size_t i = 0; i < emitters.count; ++i) {
    const loom_target_emitter_t* emitter = emitters.values[i];
    if (emitter->target_artifact_format != target_artifact_format) continue;
    if (selected_emitter != NULL) {
      const iree_string_view_t format_name =
          loom_target_artifact_format_name(target_artifact_format);
      return loomc_status_from_iree(iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "command dependency target artifact format '%.*s' is ambiguous",
          (int)format_name.size, format_name.data));
    }
    selected_emitter = emitter;
  }
  if (selected_emitter == NULL) {
    const iree_string_view_t format_name =
        loom_target_artifact_format_name(target_artifact_format);
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "command dependency target artifact format '%.*s' has no linked "
        "emitter",
        (int)format_name.size, format_name.data));
  }
  *out_artifact_format =
      loomc_string_view_from_iree(selected_emitter->public_artifact_format);
  return loomc_ok_status();
}

static void loomc_cmd_program_plan_storage_destroy(
    loomc_program_plan_storage_t* base, loomc_allocator_t allocator) {
  loomc_cmd_program_plan_storage_t* storage =
      (loomc_cmd_program_plan_storage_t*)base;

  // Plan module pointers borrow from the public module handles after
  // construction. Clear them before releasing compiler-owned metadata.
  storage->plan.root_module = NULL;
  storage->plan.launch_module = NULL;
  for (iree_host_size_t i = 0; i < storage->plan.dependency_count; ++i) {
    storage->plan.dependency_units[i].module = NULL;
  }
  loom_cmd_program_plan_deinitialize(&storage->plan);
  for (iree_host_size_t i = 0; i < storage->unit_count; ++i) {
    loomc_module_release(storage->unit_modules[i]);
  }
  loomc_workspace_release(storage->workspace);
  loomc_allocator_free(allocator, storage);
}

static loomc_status_t loomc_cmd_compile_package_unit(
    const loomc_cmd_program_plan_storage_t* storage, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, const loomc_pass_program_t* pass_program,
    const loomc_compile_report_options_t* report_options,
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  const loomc_string_view_t package_name =
      loomc_make_cstring_view("command_programs");
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_compile_module(
      storage->unit_modules[storage->package_unit.value], compiler, workspace,
      pass_program, package_name, allocator, &module, &result));
  if (!loomc_result_succeeded(result)) {
    loomc_module_release(module);
    *out_result = result;
    return loomc_ok_status();
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  const iree_host_size_t root_count = storage->plan.root_count;
  iree_host_size_t entry_count = 0;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (!iree_host_size_checked_add(
            entry_count, storage->plan.roots[i].entry_count, &entry_count)) {
      iree_arena_deinitialize(&scratch_arena);
      loomc_module_release(module);
      return loomc_cmd_finish_unit_failure(
          result,
          loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                            "command package entry table is too large"),
          loomc_make_cstring_view("PROGRAM_PLAN/COMMAND_PACKAGE"), out_result);
    }
  }

  loom_cmd_program_root_t* compiled_roots = NULL;
  loom_cmd_program_t* programs = NULL;
  iree_byte_span_t* program_data = NULL;
  loom_cmd_program_package_source_export_t* package_exports = NULL;
  loom_cmd_program_package_source_entry_t* package_entries = NULL;
  loomc_status_t status = loomc_status_from_iree(iree_arena_allocate_array(
      &scratch_arena, root_count, sizeof(*compiled_roots),
      (void**)&compiled_roots));
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*programs), (void**)&programs));
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*program_data),
        (void**)&program_data));
    if (loomc_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < root_count; ++i) {
        program_data[i] = iree_byte_span_empty();
      }
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*package_exports),
        (void**)&package_exports));
  }
  if (loomc_status_is_ok(status) && entry_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, entry_count, sizeof(*package_entries),
        (void**)&package_entries));
  }
  if (loomc_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < root_count; ++i) {
      programs[i] = (loom_cmd_program_t){0};
    }
  }

  loom_module_t* internal_module = loomc_module_loom_module(module);
  loom_cmd_program_package_source_entry_t* next_entry = package_entries;
  loom_cmd_program_plan_t compiled_plan = storage->plan;
  compiled_plan.root_module = internal_module;
  compiled_plan.roots = compiled_roots;
  iree_host_size_t command_count = 0;
  iree_host_size_t command_storage_size = 0;
  for (iree_host_size_t i = 0; i < root_count && loomc_status_is_ok(status);
       ++i) {
    compiled_roots[i] = storage->plan.roots[i];
    compiled_roots[i].function_op =
        loomc_cmd_find_symbol(internal_module, storage->roots[i].name);
    if (compiled_roots[i].function_op == NULL) {
      status = loomc_status_from_iree(iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled command package is missing selected root '%.*s'",
          (int)storage->roots[i].name.size, storage->roots[i].name.data));
      break;
    }
    status = loomc_status_from_iree(loom_cmd_program_plan_serialize_root(
        &compiled_plan, i, loomc_workspace_block_pool(workspace),
        &program_data[i], &programs[i], iree_allocator_from_loomc(allocator)));
    if (!loomc_status_is_ok(status)) break;

    if (!iree_host_size_checked_add(command_count, programs[i].commands.count,
                                    &command_count) ||
        !iree_host_size_checked_add(command_storage_size,
                                    programs[i].storage.data_length,
                                    &command_storage_size)) {
      status =
          loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                            "command package report counters are too large");
      break;
    }
    const loom_cmd_program_root_t* root = &storage->plan.roots[i];
    for (uint32_t j = 0; j < root->entry_count; ++j) {
      const loom_cmd_program_entry_t* entry = &root->entries[j];
      const uint32_t dependency_index =
          root->executable_unit_indices[entry->executable_index];
      const loomc_cmd_program_plan_dependency_storage_t* dependency =
          &storage->dependencies[dependency_index];
      next_entry[j] = (loom_cmd_program_package_source_entry_t){
          .executable_index = entry->executable_index,
          .name = iree_string_view_from_loomc(
              dependency->export_names[entry->unit_export_index]),
      };
    }
    package_exports[i] = (loom_cmd_program_package_source_export_t){
        .name = iree_string_view_from_loomc(storage->roots[i].name),
        .program = &programs[i],
        .entries = root->entry_count == 0 ? NULL : next_entry,
        .entry_count = root->entry_count,
    };
    if (root->entry_count != 0) next_entry += root->entry_count;
  }

  iree_byte_span_t package_data = iree_byte_span_empty();
  if (loomc_status_is_ok(status)) {
    loom_cmd_program_package_t package = {0};
    status = loomc_status_from_iree(loom_cmd_program_package_build(
        package_exports, root_count, iree_allocator_from_loomc(allocator),
        &package_data, &package));
  }
  const uint64_t package_size = package_data.data_length;
  if (loomc_status_is_ok(status)) {
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE),
        package_name,
        loomc_make_byte_span(package_data.data, package_data.data_length));
    if (loomc_status_is_ok(status)) package_data = iree_byte_span_empty();
  }
  if (loomc_status_is_ok(status) && report_options != NULL &&
      report_options->mode != LOOMC_COMPILE_REPORT_MODE_NONE) {
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(&report,
                                          iree_allocator_from_loomc(allocator));
    report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_COMMAND_PROGRAM;
    report.requested_detail_flags =
        loomc_compile_report_requested_detail_flags(report_options->mode);
    report.function_name = iree_string_view_from_loomc(package_name);
    report.backend_name = IREE_SV("cmd");
    report.artifact_format = IREE_SV(LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
    loom_target_compile_report_record_status(&report, IREE_STATUS_OK);
    loom_target_compile_report_record_artifact_size(&report, package_size);
    loom_target_compile_report_record_emission(
        &report, command_count, command_storage_size, package_size);
    status = loomc_cmd_add_compile_report(result, report_options, package_name,
                                          &report);
    loom_target_compile_report_deinitialize(&report);
  }

  iree_allocator_free(iree_allocator_from_loomc(allocator), package_data.data);
  if (program_data != NULL) {
    for (iree_host_size_t i = 0; i < root_count; ++i) {
      iree_allocator_free(iree_allocator_from_loomc(allocator),
                          program_data[i].data);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  loomc_module_release(module);
  if (!loomc_status_is_ok(status)) {
    return loomc_cmd_finish_unit_failure(
        result, status, loomc_make_cstring_view("PROGRAM_PLAN/COMMAND_PACKAGE"),
        out_result);
  }
  *out_result = result;
  return loomc_ok_status();
}

static loomc_status_t loomc_cmd_compile_launch_config_unit(
    const loomc_cmd_program_plan_storage_t* storage, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, const loomc_pass_program_t* pass_program,
    const loomc_compile_report_options_t* report_options,
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  const loomc_string_view_t launch_name =
      loomc_make_cstring_view("command_launch_config");
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_compile_module(
      storage->unit_modules[storage->launch_config_unit.value], compiler,
      workspace, pass_program, launch_name, allocator, &module, &result));
  if (!loomc_result_succeeded(result)) {
    loomc_module_release(module);
    *out_result = result;
    return loomc_ok_status();
  }

  loomc_byte_span_t artifact_data = loomc_byte_span_empty();
  loomc_status_t status = loomc_cmd_serialize_module(module, launch_name,
                                                     allocator, &artifact_data);
  loomc_module_release(module);
  const uint64_t artifact_size = artifact_data.data_length;
  if (loomc_status_is_ok(status)) {
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_COMMAND_LAUNCH_CONFIG,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
        launch_name, artifact_data);
    if (loomc_status_is_ok(status)) artifact_data = loomc_byte_span_empty();
  }
  loomc_allocator_free(allocator, (void*)artifact_data.data);
  if (loomc_status_is_ok(status) && report_options != NULL &&
      report_options->mode != LOOMC_COMPILE_REPORT_MODE_NONE) {
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(&report,
                                          iree_allocator_from_loomc(allocator));
    report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_LAUNCH_CONFIG;
    report.requested_detail_flags =
        loomc_compile_report_requested_detail_flags(report_options->mode);
    report.function_name = iree_string_view_from_loomc(launch_name);
    report.backend_name = IREE_SV("loom");
    report.artifact_format = IREE_SV(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
    loom_target_compile_report_record_status(&report, IREE_STATUS_OK);
    loom_target_compile_report_record_artifact_size(&report, artifact_size);
    status = loomc_cmd_add_compile_report(result, report_options, launch_name,
                                          &report);
    loom_target_compile_report_deinitialize(&report);
  }
  if (!loomc_status_is_ok(status)) {
    return loomc_cmd_finish_unit_failure(
        result, status, loomc_make_cstring_view("PROGRAM_PLAN/LAUNCH_CONFIG"),
        out_result);
  }
  *out_result = result;
  return loomc_ok_status();
}

static loomc_status_t loomc_cmd_compile_dependency_unit(
    const loomc_cmd_program_plan_storage_t* storage, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, uint32_t dependency_index,
    const loomc_pass_program_t* pass_program,
    const loomc_compile_report_options_t* report_options,
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  const loomc_cmd_program_plan_dependency_storage_t* dependency =
      &storage->dependencies[dependency_index];
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_compile_module(
      storage->unit_modules[dependency->unit.value], compiler, workspace,
      pass_program, dependency->name, allocator, &module, &result));
  if (!loomc_result_succeeded(result)) {
    loomc_module_release(module);
    *out_result = result;
    return loomc_ok_status();
  }

  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = report_options,
      .artifact_format = dependency->artifact_format,
      .identifier = dependency->name,
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_host_size_t primary_artifact_index = LOOMC_HOST_SIZE_MAX;
  loomc_status_t status = loomc_emit_module_append(
      loomc_context_target_environment(loomc_module_context(module)), workspace,
      module, &emit_options, result, &primary_artifact_index);
  loomc_module_release(module);
  if (!loomc_status_is_ok(status)) {
    loomc_result_release(result);
    return status;
  }
  *out_result = result;
  return loomc_ok_status();
}

static const loomc_module_t* loomc_cmd_program_plan_unit_module(
    const loomc_program_plan_storage_t* base, loomc_host_size_t unit_index) {
  const loomc_cmd_program_plan_storage_t* storage =
      (const loomc_cmd_program_plan_storage_t*)base;
  return storage->unit_modules[unit_index];
}

static loomc_status_t loomc_cmd_program_plan_compile_unit(
    const loomc_program_plan_storage_t* base, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, loomc_host_size_t unit_index,
    const loomc_pass_program_t* pass_program,
    const loomc_program_plan_unit_compile_options_t* options,
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  const loomc_cmd_program_plan_storage_t* storage =
      (const loomc_cmd_program_plan_storage_t*)base;
  loomc_option_chain_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
      options ? options->next : NULL, LOOMC_OPTION_CHAIN_ALLOW_COMPILE_REPORT,
      &resolved_options));
  if (unit_index == storage->package_unit.value) {
    return loomc_cmd_compile_package_unit(
        storage, compiler, workspace, pass_program,
        resolved_options.compile_report, allocator, out_result);
  }
  if (loomc_program_plan_unit_is_valid(storage->launch_config_unit) &&
      unit_index == storage->launch_config_unit.value) {
    return loomc_cmd_compile_launch_config_unit(
        storage, compiler, workspace, pass_program,
        resolved_options.compile_report, allocator, out_result);
  }
  const loomc_host_size_t dependency_base =
      loomc_program_plan_unit_is_valid(storage->launch_config_unit) ? 2u : 1u;
  return loomc_cmd_compile_dependency_unit(
      storage, compiler, workspace, (uint32_t)(unit_index - dependency_base),
      pass_program, resolved_options.compile_report, allocator, out_result);
}

static const loomc_program_plan_operations_t kLoomcCmdProgramPlanOperations = {
    .unit_module = loomc_cmd_program_plan_unit_module,
    .compile_unit = loomc_cmd_program_plan_compile_unit,
    .destroy = loomc_cmd_program_plan_storage_destroy,
};

static loomc_status_t loomc_cmd_program_plan_storage_create(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loom_cmd_program_plan_t* internal_plan, loomc_allocator_t allocator,
    loomc_cmd_program_plan_storage_t** out_storage) {
  *out_storage = NULL;
  bool has_launch_config = false;
  iree_host_size_t executable_requirement_count = 0;
  for (iree_host_size_t i = 0; i < internal_plan->root_count; ++i) {
    has_launch_config |= internal_plan->roots[i].launch_tuple_count != 0;
    if (!iree_host_size_checked_add(executable_requirement_count,
                                    internal_plan->roots[i].executable_count,
                                    &executable_requirement_count)) {
      return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                               "command executable table is too large");
    }
  }

  iree_host_size_t dependency_export_count = 0;
  for (iree_host_size_t i = 0; i < internal_plan->dependency_count; ++i) {
    if (!iree_host_size_checked_add(
            dependency_export_count,
            internal_plan->dependency_units[i].export_count,
            &dependency_export_count)) {
      return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                               "command export name table is too large");
    }
  }

  iree_host_size_t unit_count = 1;
  if (!iree_host_size_checked_add(unit_count, has_launch_config ? 1u : 0u,
                                  &unit_count) ||
      !iree_host_size_checked_add(unit_count, internal_plan->dependency_count,
                                  &unit_count) ||
      unit_count > UINT32_MAX) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "command program plan is too large");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t unit_modules_offset = 0;
  iree_host_size_t roots_offset = 0;
  iree_host_size_t dependencies_offset = 0;
  iree_host_size_t executable_requirements_offset = 0;
  iree_host_size_t export_names_offset = 0;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(loomc_cmd_program_plan_storage_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(unit_count, loomc_module_t*,
                                iree_alignof(loomc_module_t*),
                                &unit_modules_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          internal_plan->root_count, loomc_cmd_program_plan_root_storage_t,
          iree_alignof(loomc_cmd_program_plan_root_storage_t), &roots_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          internal_plan->dependency_count,
          loomc_cmd_program_plan_dependency_storage_t,
          iree_alignof(loomc_cmd_program_plan_dependency_storage_t),
          &dependencies_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          executable_requirement_count,
          loomc_cmd_program_plan_executable_requirement_t,
          iree_alignof(loomc_cmd_program_plan_executable_requirement_t),
          &executable_requirements_offset),
      IREE_STRUCT_FIELD_ALIGNED(dependency_export_count, loomc_string_view_t,
                                iree_alignof(loomc_string_view_t),
                                &export_names_offset))));

  loomc_cmd_program_plan_storage_t* storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_allocator_malloc_array(
      iree_allocator_from_loomc(allocator), total_size, 1, (void**)&storage)));
  memset(storage, 0, total_size);
  storage->base.operations = &kLoomcCmdProgramPlanOperations;
  storage->workspace = workspace;
  loomc_workspace_retain(workspace);
  storage->plan = *internal_plan;
  memset(internal_plan, 0, sizeof(*internal_plan));
  storage->unit_modules =
      (loomc_module_t**)((uint8_t*)storage + unit_modules_offset);
  storage->unit_count = unit_count;
  storage->roots = (loomc_cmd_program_plan_root_storage_t*)((uint8_t*)storage +
                                                            roots_offset);
  storage->dependencies =
      (loomc_cmd_program_plan_dependency_storage_t*)((uint8_t*)storage +
                                                     dependencies_offset);
  storage->package_unit = (loomc_program_plan_unit_t){0};
  storage->launch_config_unit = has_launch_config
                                    ? (loomc_program_plan_unit_t){1}
                                    : loomc_program_plan_unit_invalid();

  const loomc_host_size_t dependency_base = has_launch_config ? 2u : 1u;
  loomc_cmd_program_plan_executable_requirement_t* next_requirement =
      (loomc_cmd_program_plan_executable_requirement_t*)((uint8_t*)storage +
                                                         executable_requirements_offset);
  for (iree_host_size_t i = 0; i < storage->plan.root_count; ++i) {
    const loom_cmd_program_root_t* internal_root = &storage->plan.roots[i];
    loomc_cmd_program_plan_root_storage_t* root = &storage->roots[i];
    root->name = loomc_cmd_symbol_name(storage->plan.root_module,
                                       internal_root->function_op);
    root->executable_requirements = next_requirement;
    root->executable_requirement_count = internal_root->executable_count;
    for (uint32_t j = 0; j < internal_root->executable_count; ++j) {
      next_requirement[j] = (loomc_cmd_program_plan_executable_requirement_t){
          .unit = {dependency_base + internal_root->executable_unit_indices[j]},
      };
    }
    next_requirement += internal_root->executable_count;
  }

  loomc_string_view_t* next_export_name =
      (loomc_string_view_t*)((uint8_t*)storage + export_names_offset);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loom_symbol_fact_table_t target_fact_table = {0};
  loom_symbol_fact_table_initialize(&target_fact_table, &scratch_arena);
  const loom_target_environment_t* target_environment =
      loomc_target_environment_loom_target_environment(
          loomc_context_target_environment(context));
  loomc_status_t status = loomc_ok_status();
  for (iree_host_size_t i = 0;
       i < storage->plan.dependency_count && loomc_status_is_ok(status); ++i) {
    loom_cmd_kernel_unit_t* internal_dependency =
        &storage->plan.dependency_units[i];
    loomc_cmd_program_plan_dependency_storage_t* dependency =
        &storage->dependencies[i];
    dependency->export_names = next_export_name;
    dependency->export_count = internal_dependency->export_count;
    dependency->unit = (loomc_program_plan_unit_t){dependency_base + i};
    for (uint32_t j = 0; j < dependency->export_count; ++j) {
      dependency->export_names[j] = loomc_cmd_symbol_name(
          internal_dependency->module, internal_dependency->kernel_ops[j]);
    }
    next_export_name += dependency->export_count;
    dependency->name = dependency->export_names[0];
    status = loomc_cmd_resolve_dependency_artifact_format(
        target_environment, internal_dependency->module,
        internal_dependency->kernel_ops[0], &target_fact_table,
        &dependency->artifact_format);
  }
  iree_arena_deinitialize(&scratch_arena);

  // Transfer module ownership only after all metadata queries have succeeded.
  // During a partial transfer, NULL internal pointers identify modules already
  // owned by |unit_modules| and make failure cleanup unambiguous.
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_take_internal_module(
        context, workspace, &storage->plan.root_module, allocator,
        &storage->unit_modules[storage->package_unit.value]);
  }
  if (has_launch_config && loomc_status_is_ok(status)) {
    status = loomc_cmd_take_internal_module(
        context, workspace, &storage->plan.launch_module, allocator,
        &storage->unit_modules[storage->launch_config_unit.value]);
  } else if (!has_launch_config && storage->plan.launch_module != NULL) {
    loom_module_free(storage->plan.launch_module);
    storage->plan.launch_module = NULL;
  }
  for (iree_host_size_t i = 0;
       i < storage->plan.dependency_count && loomc_status_is_ok(status); ++i) {
    loom_cmd_kernel_unit_t* internal_dependency =
        &storage->plan.dependency_units[i];
    status = loomc_cmd_take_internal_module(
        context, workspace, &internal_dependency->module, allocator,
        &storage->unit_modules[storage->dependencies[i].unit.value]);
  }

  if (loomc_status_is_ok(status)) {
    storage->plan.root_module = (loom_module_t*)loomc_module_const_loom_module(
        storage->unit_modules[storage->package_unit.value]);
    if (has_launch_config) {
      storage->plan.launch_module =
          (loom_module_t*)loomc_module_const_loom_module(
              storage->unit_modules[storage->launch_config_unit.value]);
    }
    for (iree_host_size_t i = 0; i < storage->plan.dependency_count; ++i) {
      storage->plan.dependency_units[i].module =
          (loom_module_t*)loomc_module_const_loom_module(
              storage->unit_modules[storage->dependencies[i].unit.value]);
    }
    *out_storage = storage;
    return loomc_ok_status();
  }

  // Internal module pointers are either still owned or NULL after a transfer.
  loom_cmd_program_plan_deinitialize(&storage->plan);
  for (iree_host_size_t i = 0; i < storage->unit_count; ++i) {
    loomc_module_release(storage->unit_modules[i]);
  }
  loomc_workspace_release(storage->workspace);
  loomc_allocator_free(allocator, storage);
  return status;
}

static loomc_status_t loomc_cmd_program_plan_create_public(
    loomc_cmd_program_plan_storage_t* storage, loomc_workspace_t* workspace,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan) {
  const iree_host_size_t root_count = storage->plan.root_count;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loomc_program_plan_root_create_params_t* roots = NULL;
  loomc_status_t status = loomc_status_from_iree(iree_arena_allocate_array(
      &scratch_arena, root_count, sizeof(*roots), (void**)&roots));
  for (iree_host_size_t i = 0; i < root_count && loomc_status_is_ok(status);
       ++i) {
    const loom_cmd_program_root_t* internal_root = &storage->plan.roots[i];
    iree_host_size_t required_unit_count = 1;
    if (!iree_host_size_checked_add(
            required_unit_count,
            internal_root->launch_tuple_count != 0 ? 1u : 0u,
            &required_unit_count) ||
        !iree_host_size_checked_add(required_unit_count,
                                    internal_root->executable_count,
                                    &required_unit_count)) {
      status = loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                                 "command root closure is too large");
      break;
    }
    loomc_program_plan_unit_t* required_units = NULL;
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, required_unit_count, sizeof(*required_units),
        (void**)&required_units));
    if (!loomc_status_is_ok(status)) break;
    required_units[0] = storage->package_unit;
    iree_host_size_t next_required_unit = 1;
    if (internal_root->launch_tuple_count != 0) {
      required_units[next_required_unit++] = storage->launch_config_unit;
    }
    for (uint32_t j = 0; j < internal_root->executable_count; ++j) {
      required_units[next_required_unit++] =
          storage->roots[i].executable_requirements[j].unit;
    }
    roots[i] = (loomc_program_plan_root_create_params_t){
        .name = storage->roots[i].name,
        .required_units = required_units,
        .required_unit_count = required_unit_count,
    };
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_program_plan_create(
        &(loomc_program_plan_create_params_t){
            .roots = roots,
            .root_count = root_count,
            .unit_count = storage->unit_count,
            .storage = &storage->base,
        },
        allocator, out_program_plan);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static bool loomc_cmd_program_provider_owns_root(const loom_module_t* module,
                                                 const loom_op_t* root_op) {
  if (!loom_command_program_def_isa(root_op)) return false;
  return loom_func_like_visibility(loom_func_like_cast(
             module, (loom_op_t*)root_op)) == LOOM_FUNC_VISIBILITY_PUBLIC;
}

static loomc_status_t loomc_cmd_program_provider_prepare(
    loomc_workspace_t* workspace, const loomc_module_t* sealed_module,
    const loom_op_t* const* selected_root_ops,
    loomc_host_size_t selected_root_count,
    const loomc_program_plan_options_t* options, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan) {
  (void)options;
  *out_program_plan = NULL;

  loom_pass_registry_storage_t pass_registry_storage = {0};
  const loom_pass_registry_t* pass_registry = NULL;
  loomc_status_t status = loomc_target_pass_registry_initialize(
      loomc_context_target_environment(loomc_module_context(sealed_module)),
      &pass_registry_storage, &pass_registry);

  loom_cmd_program_plan_t internal_plan = {0};
  bool valid = false;
  loomc_cmd_diagnostic_capture_t capture = {
      .result = result,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(loom_cmd_program_plan_prepare(
        loomc_module_const_loom_module(sealed_module), selected_root_ops,
        selected_root_count, pass_registry,
        (iree_diagnostic_emitter_t){
            .fn = loomc_cmd_capture_diagnostic,
            .user_data = &capture,
        },
        loomc_workspace_block_pool(workspace), &valid, &internal_plan,
        iree_allocator_from_loomc(allocator)));
  }
  if (!loomc_status_is_ok(status)) {
    loom_cmd_program_plan_deinitialize(&internal_plan);
    if (loomc_status_is_result_diagnostic(status)) {
      return loomc_result_fail_status_diagnostic_consume(
          result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
          loomc_make_cstring_view("PROGRAM_PLAN/COMMAND_PREPARE"), status);
    }
    return status;
  }
  if (!valid || capture.error_count != 0) {
    loom_cmd_program_plan_deinitialize(&internal_plan);
    LOOMC_RETURN_IF_ERROR(
        loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED));
    return loomc_ok_status();
  }

  loomc_cmd_program_plan_storage_t* storage = NULL;
  status = loomc_cmd_program_plan_storage_create(
      loomc_module_context(sealed_module), workspace, &internal_plan, allocator,
      &storage);
  loom_cmd_program_plan_deinitialize(&internal_plan);
  if (!loomc_status_is_ok(status)) {
    if (loomc_status_is_result_diagnostic(status)) {
      return loomc_result_fail_status_diagnostic_consume(
          result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
          loomc_make_cstring_view("PROGRAM_PLAN/COMMAND_UNITS"), status);
    }
    return status;
  }

  status = loomc_cmd_program_plan_create_public(storage, workspace, allocator,
                                                out_program_plan);
  if (!loomc_status_is_ok(status)) {
    loomc_cmd_program_plan_storage_destroy(&storage->base, allocator);
  }
  return status;
}

static const loomc_program_provider_t kLoomcCmdProgramProvider = {
    .owns_root = loomc_cmd_program_provider_owns_root,
    .prepare = loomc_cmd_program_provider_prepare,
};

static const loomc_program_provider_t* const kLoomcCmdProgramProviders[] = {
    &kLoomcCmdProgramProvider,
};

const loomc_program_provider_set_t loomc_cmd_program_provider_set = {
    .values = kLoomcCmdProgramProviders,
    .count = IREE_ARRAYSIZE(kLoomcCmdProgramProviders),
};

loomc_status_t loomc_cmd_program_plan_root_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_cmd_program_plan_root_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  *out_info = (loomc_cmd_program_plan_root_info_t){0};
  const loomc_program_plan_storage_t* base =
      loomc_program_plan_storage(program_plan);
  if (base == NULL || base->operations != &kLoomcCmdProgramPlanOperations) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "program plan was not prepared by the command provider");
  }
  const loomc_cmd_program_plan_storage_t* storage =
      (const loomc_cmd_program_plan_storage_t*)base;
  if (root.value >= storage->plan.root_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program plan root token is out of range");
  }
  const loomc_cmd_program_plan_root_storage_t* root_storage =
      &storage->roots[root.value];
  *out_info = (loomc_cmd_program_plan_root_info_t){
      .package_unit = storage->package_unit,
      .launch_config_unit =
          storage->plan.roots[root.value].launch_tuple_count != 0
              ? storage->launch_config_unit
              : loomc_program_plan_unit_invalid(),
      .executable_requirements = root_storage->executable_requirements,
      .executable_requirement_count =
          root_storage->executable_requirement_count,
  };
  return loomc_ok_status();
}
