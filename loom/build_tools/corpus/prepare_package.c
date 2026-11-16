// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
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
#include "loomc/target/configured.h"

IREE_FLAG(string, target, "", "Exact target profile in family:selector form.");
IREE_FLAG(string, output, "", "New prepared-package output directory.");

static const iree_string_view_t kManifestPath = {"manifest.json",
                                                 sizeof("manifest.json") - 1};
static const iree_string_view_t kAuthoredModulePath = {
    "authored.loombc", sizeof("authored.loombc") - 1};
static const iree_string_view_t kPreparedModulePath = {
    "prepared.loombc", sizeof("prepared.loombc") - 1};
static const iree_string_view_t kLaunchConfigPath = {
    "launch.loombc", sizeof("launch.loombc") - 1};

typedef struct loom_corpus_prepared_products_t {
  loomc_string_view_t* roots;
  iree_host_size_t root_count;
  loom_run_hal_testbench_prepared_package_manifest_t manifest;
} loom_corpus_prepared_products_t;

typedef struct loom_corpus_compile_artifacts_t {
  const loomc_artifact_t* executable;
  const loomc_artifact_t* prepared_module;
  const loomc_artifact_t* launch_config;
} loom_corpus_compile_artifacts_t;

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
    iree_allocator_t allocator, loom_corpus_prepared_products_t* out_products) {
  *out_products = (loom_corpus_prepared_products_t){0};
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
  iree_host_size_t root_storage_size = 0;
  iree_host_size_t product_storage_size = 0;
  if (!iree_host_size_checked_mul(capacity, sizeof(loomc_string_view_t),
                                  &root_storage_size) ||
      !iree_host_size_checked_mul(
          capacity, sizeof(loom_run_hal_testbench_prepared_package_product_t),
          &product_storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prepared product storage size overflow");
  }
  loomc_string_view_t* roots = NULL;
  loom_run_hal_testbench_prepared_package_product_t* products = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, root_storage_size, (void**)&roots));
  iree_status_t status =
      iree_allocator_malloc(allocator, product_storage_size, (void**)&products);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, roots);
    return status;
  }

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
      const loom_testbench_trial_plan_t* trial = &scenario->trials[j];
      for (iree_host_size_t k = 0; k < trial->recipe_step_count; ++k) {
        const loom_testbench_trial_recipe_step_t* step =
            &trial->recipe_steps[k];
        if (step->kind == LOOM_TESTBENCH_TRIAL_RECIPE_STEP_VALUE_SOURCE &&
            step->value_source.kind ==
                LOOM_TESTBENCH_VALUE_SOURCE_FILE_READ_NPY) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "prepared package v0 does not carry fixture file '%.*s' used "
              "by scenario @%.*s",
              (int)step->value_source.file.path.size,
              step->value_source.file.path.data, (int)scenario->name.size,
              scenario->name.data);
          break;
        }
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      const loom_testbench_invocation_plan_t* invocation =
          &trial->action.target;
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
      if (duplicate) {
        continue;
      }
      roots[product_count] = loomc_string_view_from_iree(subject);
      products[product_count++] =
          (loom_run_hal_testbench_prepared_package_product_t){
              .subject = subject,
              .prepared_module_path = kPreparedModulePath,
              .launch_config_path = kLaunchConfigPath,
          };
    }
  }
  if (iree_status_is_ok(status) && product_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "prepared package v0 has no target products");
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, products);
    iree_allocator_free(allocator, roots);
    return status;
  }

  out_products->roots = roots;
  out_products->root_count = product_count;
  out_products->manifest = (loom_run_hal_testbench_prepared_package_manifest_t){
      .host_allocator = allocator,
      .products = products,
      .product_count = product_count,
      .authored_module_path = kAuthoredModulePath,
      .target_profile = target,
  };
  return iree_ok_status();
}

static void loom_corpus_prepared_products_deinitialize(
    loom_corpus_prepared_products_t* products) {
  if (products == NULL) {
    return;
  }
  const iree_allocator_t allocator = products->manifest.host_allocator;
  loom_run_hal_testbench_prepared_package_manifest_deinitialize(
      &products->manifest);
  iree_allocator_free(allocator, products->roots);
  *products = (loom_corpus_prepared_products_t){0};
}

static iree_status_t loom_corpus_select_compile_artifacts(
    const loomc_result_t* result,
    loom_corpus_compile_artifacts_t* out_artifacts) {
  *out_artifacts = (loom_corpus_compile_artifacts_t){0};
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    const loomc_artifact_t** slot = NULL;
    if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      slot = &out_artifacts->executable;
    } else if (artifact->kind == LOOMC_ARTIFACT_KIND_MODULE &&
               loomc_string_view_equal(
                   artifact->format,
                   loomc_make_cstring_view(
                       LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE))) {
      slot = &out_artifacts->prepared_module;
    } else if (artifact->kind == LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG &&
               loomc_string_view_equal(
                   artifact->format,
                   loomc_make_cstring_view(
                       LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE))) {
      slot = &out_artifacts->launch_config;
    }
    if (slot != NULL && *slot != NULL) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "compiler returned duplicate package artifact");
    }
    if (slot != NULL) {
      *slot = artifact;
    }
  }
  if (out_artifacts->executable == NULL ||
      out_artifacts->prepared_module == NULL ||
      out_artifacts->launch_config == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "compiler omitted an executable, prepared module, or launch config");
  }
  if (out_artifacts->executable->format.size == 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "compiler selected an empty artifact format");
  }
  return iree_ok_status();
}

static iree_status_t loom_corpus_join_package_path(
    iree_string_view_t output_directory, iree_string_view_t relative_path,
    iree_allocator_t allocator, char** out_path) {
  return loom_tooling_file_path_join(output_directory, relative_path, allocator,
                                     out_path);
}

static iree_status_t loom_corpus_write_package(
    iree_string_view_t input_path, iree_string_view_t output_directory,
    const loom_corpus_compile_artifacts_t* artifacts,
    loom_corpus_prepared_products_t* products, iree_allocator_t allocator) {
  bool output_exists = false;
  IREE_RETURN_IF_ERROR(loom_tooling_file_path_is_directory(
      output_directory, allocator, &output_exists));
  if (output_exists) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "output directory '%.*s' already exists",
                            (int)output_directory.size, output_directory.data);
  }
  IREE_RETURN_IF_ERROR(
      loom_tooling_create_directory_if_needed(output_directory, allocator));

  char* authored_path = NULL;
  char* prepared_path = NULL;
  char* launch_path = NULL;
  char* manifest_path = NULL;
  iree_io_file_contents_t* authored_contents = NULL;
  iree_string_builder_t manifest_builder;
  iree_string_builder_initialize(allocator, &manifest_builder);
  iree_status_t status = loom_corpus_join_package_path(
      output_directory, kAuthoredModulePath, allocator, &authored_path);
  if (iree_status_is_ok(status)) {
    status = loom_corpus_join_package_path(
        output_directory, kPreparedModulePath, allocator, &prepared_path);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_join_package_path(output_directory, kLaunchConfigPath,
                                           allocator, &launch_path);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_join_package_path(output_directory, kManifestPath,
                                           allocator, &manifest_path);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_tooling_read_input_file(input_path, allocator, &authored_contents);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(authored_path),
        loom_tooling_file_contents_string_view(authored_contents), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_artifact_write_to_path(
        artifacts->prepared_module, loomc_make_cstring_view(prepared_path),
        loomc_allocator_from_iree(allocator)));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_artifact_write_to_path(
        artifacts->launch_config, loomc_make_cstring_view(launch_path),
        loomc_allocator_from_iree(allocator)));
  }
  products->manifest.artifact_format =
      iree_string_view_from_loomc(artifacts->executable->format);
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&manifest_builder, &stream);
    status = loom_run_hal_testbench_prepared_package_manifest_write(
        &products->manifest, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(manifest_path),
        iree_string_builder_view(&manifest_builder), allocator);
  }

  iree_string_builder_deinitialize(&manifest_builder);
  iree_io_file_contents_free(authored_contents);
  iree_allocator_free(allocator, manifest_path);
  iree_allocator_free(allocator, launch_path);
  iree_allocator_free(allocator, prepared_path);
  iree_allocator_free(allocator, authored_path);
  return status;
}

static iree_status_t loom_corpus_prepare_package(iree_string_view_t input_path,
                                                 iree_string_view_t target,
                                                 iree_string_view_t output) {
  const iree_allocator_t allocator = iree_allocator_system();
  const loomc_allocator_t loom_allocator = loomc_allocator_from_iree(allocator);
  loomc_target_environment_t* target_environment = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_target_profile_t* target_profile = NULL;
  loomc_result_t* result = NULL;
  iree_arena_block_pool_t plan_block_pool;
  iree_arena_allocator_t plan_arena;
  bool plan_arena_initialized = false;
  loom_corpus_prepared_products_t products = {0};
  loom_corpus_compile_artifacts_t artifacts = {0};

  iree_status_t status =
      iree_status_from_loomc(loomc_target_environment_create_configured(
          loom_allocator, &target_environment));
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(
        loomc_context_create(&context_options, loom_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_workspace_create(NULL, loom_allocator, &workspace));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_compiler_create(context, NULL, loom_allocator, &compiler));
  }
  if (iree_status_is_ok(status)) {
    const loomc_source_load_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = LOOMC_SOURCE_FORMAT_BYTECODE,
    };
    status = iree_status_from_loomc(loomc_source_create_from_path(
        loomc_string_view_from_iree(input_path), &source_options,
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
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        target_environment, loomc_string_view_from_iree(target), loom_allocator,
        &target_profile));
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
                                                     &products);
    }
  }

  if (iree_status_is_ok(status)) {
    const loomc_emit_options_t emit_options = {
        .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        .structure_size = sizeof(emit_options),
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    const loomc_compile_artifact_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
        .structure_size = sizeof(compile_options),
        .roots = products.roots,
        .root_count = products.root_count,
        .target_profile = target_profile,
        .emit_options = &emit_options,
        .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE |
                          LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
    };
    status = iree_status_from_loomc(
        loomc_compile_artifact(compiler, workspace, NULL, module,
                               &compile_options, loom_allocator, &result));
  }
  if (iree_status_is_ok(status)) {
    bool succeeded = false;
    status = loom_corpus_observe_result(result, &succeeded);
    if (iree_status_is_ok(status) && !succeeded) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "target package compilation failed");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_select_compile_artifacts(result, &artifacts);
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_write_package(input_path, output, &artifacts,
                                       &products, allocator);
  }

  loom_corpus_prepared_products_deinitialize(&products);
  if (plan_arena_initialized) {
    iree_arena_deinitialize(&plan_arena);
    iree_arena_block_pool_deinitialize(&plan_block_pool);
  }
  loomc_result_release(result);
  loomc_target_profile_release(target_profile);
  loomc_module_release(module);
  loomc_source_release(source);
  loomc_compiler_release(compiler);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-corpus-prepare-package",
      "Prepares one linked scenario-only Loom test module for execution on "
      "an exact target.\n\n"
      "Usage:\n"
      "  loom-corpus-prepare-package module.loombc "
      "--target=family:selector --output=package\n");
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
  if (iree_status_is_ok(status) && (iree_string_view_is_empty(output) ||
                                    loom_tooling_file_path_is_stdio(output))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--output must name a package directory");
  }
  if (iree_status_is_ok(status)) {
    status = loom_corpus_prepare_package(input_path, target, output);
  }
  if (iree_status_is_ok(status)) {
    return EXIT_SUCCESS;
  }
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return EXIT_FAILURE;
}
