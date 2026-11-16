// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/json.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "loom/ir/ir.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/execution/hal/prepared_package.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/issue_report.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/util/stream.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"

IREE_FLAG_NAMED(string, compile_report, "compile-report", "",
                "Detailed report emitted by the owning compile action.");
IREE_FLAG_NAMED(string, prepared_module, "prepared-module", "",
                "Prepared-low Loom bytecode emitted by the compile action.");
IREE_FLAG_NAMED(string, launch_config, "launch-config", "",
                "Launch-config Loom bytecode emitted by the compile action.");
IREE_FLAG(string, target, "", "Exact target profile in family:selector form.");
IREE_FLAG(string, output, "", "Prepared-package output directory.");

static const iree_string_view_t kManifestPath = {"manifest.json",
                                                 sizeof("manifest.json") - 1};
static const iree_string_view_t kPreparedModulePath = {
    "prepared.loombc", sizeof("prepared.loombc") - 1};
static const iree_string_view_t kLaunchConfigPath = {
    "launch.loombc", sizeof("launch.loombc") - 1};

static iree_status_t loom_corpus_observe_result(loomc_result_t* result,
                                                bool* out_succeeded) {
  if (result == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "compiler operation returned no result");
  }
  return loom_tooling_cli_print_loomc_result(stderr, result, out_succeeded);
}

static iree_string_view_t loom_corpus_invocation_name(
    const loom_testbench_invocation_plan_t* invocation) {
  const loom_symbol_t* symbol =
      &invocation->module->symbols.entries[invocation->callee_ref.symbol_id];
  return loom_string_table_get(&invocation->module->strings, symbol->name_id);
}

static iree_status_t loom_corpus_count_trial_domains(
    const loom_testbench_module_plan_t* plan,
    iree_host_size_t* out_trial_domain_count) {
  *out_trial_domain_count = 0;
  for (iree_host_size_t i = 0; i < plan->scenario_count; ++i) {
    if (!iree_host_size_checked_add(*out_trial_domain_count,
                                    plan->scenarios[i].trial_count,
                                    out_trial_domain_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "scenario trial-domain count overflow");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_corpus_collect_prepared_products(
    const loom_testbench_module_plan_t* plan, iree_string_view_t target,
    iree_allocator_t allocator,
    loom_run_hal_testbench_prepared_package_manifest_t* out_manifest) {
  *out_manifest = (loom_run_hal_testbench_prepared_package_manifest_t){0};
  if (plan->issue_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "test module has %" PRIhsz " planning issue(s); first is %.*s",
        plan->issue_count,
        (int)loom_testbench_issue_kind_name(plan->issues[0].kind).size,
        loom_testbench_issue_kind_name(plan->issues[0].kind).data);
  }
  if (plan->case_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared package v0 requires a scenario-only module; found %" PRIhsz
        " check.case record(s)",
        plan->case_count);
  }
  if (plan->scenario_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared package v0 requires a check.scenario");
  }

  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(loom_corpus_count_trial_domains(plan, &capacity));
  if (capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared package v0 requires a check.trial");
  }
  loom_run_hal_testbench_prepared_package_product_t* products = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, capacity, sizeof(*products), (void**)&products));

  iree_status_t status = iree_ok_status();
  iree_host_size_t product_count = 0;
  for (iree_host_size_t i = 0;
       i < plan->scenario_count && iree_status_is_ok(status); ++i) {
    const loom_testbench_scenario_plan_t* scenario = &plan->scenarios[i];
    if (scenario->is_configured) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "prepared package v0 does not support configured scenario @%.*s",
          (int)scenario->name.size, scenario->name.data);
      break;
    }
    for (iree_host_size_t j = 0;
         j < scenario->trial_count && iree_status_is_ok(status); ++j) {
      const loom_testbench_invocation_plan_t* invocation =
          &scenario->trials[j].action.target;
      const iree_string_view_t subject =
          loom_corpus_invocation_name(invocation);
      if (invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "prepared package v0 requires a direct kernel subject; @%.*s "
            "uses invocation kind %u",
            (int)subject.size, subject.data, (unsigned)invocation->kind);
        break;
      }
      bool duplicate = false;
      for (iree_host_size_t k = 0; k < product_count; ++k) {
        if (iree_string_view_equal(products[k].subject, subject)) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        products[product_count++] =
            (loom_run_hal_testbench_prepared_package_product_t){
                .subject = subject,
                .prepared_module_path = kPreparedModulePath,
                .launch_config_path = kLaunchConfigPath,
            };
      }
    }
  }
  if (iree_status_is_ok(status) && product_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "prepared package v0 has no target products");
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, products);
    return status;
  }

  *out_manifest = (loom_run_hal_testbench_prepared_package_manifest_t){
      .host_allocator = allocator,
      .products = products,
      .product_count = product_count,
      .target_profile = target,
  };
  return iree_ok_status();
}

static iree_status_t loom_corpus_parse_artifact_format(
    iree_string_view_t json, iree_string_view_t* out_format) {
  *out_format = iree_string_view_empty();
  iree_string_view_t remaining = json;
  iree_string_view_t root = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_consume_value(&remaining, &root));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&remaining));
  if (!iree_string_view_is_empty(remaining) ||
      iree_string_view_is_empty(root) || root.data[0] != '{') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compile report must be one JSON object");
  }
  iree_string_view_t kind = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(root, IREE_SV("kind"), &kind));
  if (!iree_string_view_equal(kind, IREE_SV("loom.compile_report"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input is not a Loom compile report");
  }
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      root, IREE_SV("public_artifact_format"), out_format));
  if (iree_string_view_is_empty(*out_format)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compile report requires public_artifact_format");
  }
  return iree_ok_status();
}

static iree_status_t loom_corpus_join_package_path(
    iree_string_view_t output_directory, iree_string_view_t relative_path,
    iree_allocator_t allocator, char** out_path) {
  return loom_tooling_file_path_join(output_directory, relative_path, allocator,
                                     out_path);
}

static iree_status_t loom_corpus_copy_file(iree_string_view_t source_path,
                                           iree_string_view_t destination_path,
                                           iree_allocator_t allocator) {
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status =
      loom_tooling_read_input_file(source_path, allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        destination_path, loom_tooling_file_contents_string_view(contents),
        allocator);
  }
  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t loom_corpus_write_package(
    iree_string_view_t output_directory, iree_string_view_t prepared_source,
    iree_string_view_t launch_source,
    loom_run_hal_testbench_prepared_package_manifest_t* manifest,
    iree_allocator_t allocator) {
  IREE_RETURN_IF_ERROR(
      loom_tooling_create_directory_if_needed(output_directory, allocator));

  char* prepared_path = NULL;
  char* launch_path = NULL;
  char* manifest_path = NULL;
  iree_string_builder_t manifest_builder;
  iree_string_builder_initialize(allocator, &manifest_builder);
  iree_status_t status = loom_corpus_join_package_path(
      output_directory, kPreparedModulePath, allocator, &prepared_path);
  if (iree_status_is_ok(status)) {
    status = loom_corpus_join_package_path(output_directory, kLaunchConfigPath,
                                           allocator, &launch_path);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_join_package_path(output_directory, kManifestPath,
                                           allocator, &manifest_path);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_copy_file(
        prepared_source, iree_make_cstring_view(prepared_path), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_copy_file(
        launch_source, iree_make_cstring_view(launch_path), allocator);
  }
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&manifest_builder, &stream);
    status = loom_run_hal_testbench_prepared_package_manifest_write(manifest,
                                                                    &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(manifest_path),
        iree_string_builder_view(&manifest_builder), allocator);
  }

  iree_string_builder_deinitialize(&manifest_builder);
  iree_allocator_free(allocator, manifest_path);
  iree_allocator_free(allocator, launch_path);
  iree_allocator_free(allocator, prepared_path);
  return status;
}

static iree_status_t loom_corpus_assemble_package(
    iree_string_view_t authored_module_path, iree_string_view_t target,
    iree_string_view_t compile_report_path,
    iree_string_view_t prepared_module_path,
    iree_string_view_t launch_config_path, iree_string_view_t output) {
  const iree_allocator_t allocator = iree_allocator_system();
  const loomc_allocator_t loom_allocator = loomc_allocator_from_iree(allocator);
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  iree_io_file_contents_t* compile_report_contents = NULL;
  iree_arena_block_pool_t plan_block_pool;
  iree_arena_allocator_t plan_arena;
  bool plan_arena_initialized = false;
  loom_run_hal_testbench_prepared_package_manifest_t manifest = {0};

  iree_status_t status = iree_status_from_loomc(
      loomc_context_create(NULL, loom_allocator, &context));
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_workspace_create(NULL, loom_allocator, &workspace));
  }
  if (iree_status_is_ok(status)) {
    const loomc_source_load_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = LOOMC_SOURCE_FORMAT_BYTECODE,
    };
    status = iree_status_from_loomc(loomc_source_create_from_path(
        loomc_string_view_from_iree(authored_module_path), &source_options,
        loom_allocator, &source));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        context, workspace, source, NULL, loom_allocator, &module, &result));
  }
  if (iree_status_is_ok(status)) {
    bool succeeded = false;
    status = loom_corpus_observe_result(result, &succeeded);
    if (iree_status_is_ok(status) && !succeeded) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "authored module could not be deserialized");
    }
    loomc_result_release(result);
    result = NULL;
  }

  loomc_module_interop_view_t module_view = {0};
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_get_interop_view(
        module, loom_allocator, &module_view, &result));
  }
  if (iree_status_is_ok(status)) {
    bool succeeded = false;
    status = loom_corpus_observe_result(result, &succeeded);
    if (iree_status_is_ok(status) && !succeeded) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "authored module verification failed");
    }
    loomc_result_release(result);
    result = NULL;
  }
  if (iree_status_is_ok(status)) {
    iree_arena_block_pool_initialize(4096, allocator, &plan_block_pool);
    iree_arena_initialize(&plan_block_pool, &plan_arena);
    plan_arena_initialized = true;
    loom_testbench_module_plan_t plan = {0};
    status = loom_testbench_plan_module(module_view.module, NULL, &plan_arena,
                                        &plan);
    if (iree_status_is_ok(status)) {
      status = loom_corpus_collect_prepared_products(&plan, target, allocator,
                                                     &manifest);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(compile_report_path, allocator,
                                          &compile_report_contents);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_parse_artifact_format(
        loom_tooling_file_contents_string_view(compile_report_contents),
        &manifest.artifact_format);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_write_package(
        output, prepared_module_path, launch_config_path, &manifest, allocator);
  }

  loom_run_hal_testbench_prepared_package_manifest_deinitialize(&manifest);
  if (plan_arena_initialized) {
    iree_arena_deinitialize(&plan_arena);
    iree_arena_block_pool_deinitialize(&plan_block_pool);
  }
  iree_io_file_contents_free(compile_report_contents);
  loomc_result_release(result);
  loomc_module_release(module);
  loomc_source_release(source);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  return status;
}

static iree_status_t loom_corpus_require_file_flag(const char* name,
                                                   const char* value) {
  const iree_string_view_t path = iree_make_cstring_view(value);
  if (iree_string_view_is_empty(path) ||
      loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%s must name a file", name);
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-corpus-assemble-package",
      "Assembles compiler-owned target payloads for one linked scenario "
      "module.\n\n"
      "Usage:\n"
      "  loom-corpus-assemble-package module.loombc "
      "--target=family:selector --compile-report=compile.json "
      "--prepared-module=prepared.loombc --launch-config=launch.loombc "
      "--output=package\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP, &argc,
                           &argv);
  iree_status_t status = iree_ok_status();
  if (argc != 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exactly one linked .loombc input is required");
  }
  const iree_string_view_t input_path =
      argc == 2 ? iree_make_cstring_view(argv[1]) : iree_string_view_empty();
  const iree_string_view_t target = iree_make_cstring_view(FLAG_target);
  const iree_string_view_t output = iree_make_cstring_view(FLAG_output);
  if (iree_status_is_ok(status) &&
      (loom_tooling_file_path_is_stdio(input_path) ||
       !iree_string_view_ends_with(input_path, IREE_SV(".loombc")))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "input must be a linked .loombc file");
  }
  if (iree_status_is_ok(status) && iree_string_view_is_empty(target)) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "--target is required");
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_corpus_require_file_flag("compile-report", FLAG_compile_report);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_corpus_require_file_flag("prepared-module", FLAG_prepared_module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_require_file_flag("launch-config", FLAG_launch_config);
  }
  if (iree_status_is_ok(status) && (iree_string_view_is_empty(output) ||
                                    loom_tooling_file_path_is_stdio(output))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--output must name a package directory");
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_assemble_package(
        input_path, target, iree_make_cstring_view(FLAG_compile_report),
        iree_make_cstring_view(FLAG_prepared_module),
        iree_make_cstring_view(FLAG_launch_config), output);
  }
  if (iree_status_is_ok(status)) {
    return EXIT_SUCCESS;
  }
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return EXIT_FAILURE;
}
