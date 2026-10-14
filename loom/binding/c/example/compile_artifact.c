// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "loomc/compile.h"
#include "loomc/iree.h"
#include "loomc/target/configured.h"

static loomc_status_t require_success(const loomc_result_t* result) {
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    fprintf(stderr, "%.*s: %.*s\n", (int)diagnostic->code.size,
            diagnostic->code.data, (int)diagnostic->message.size,
            diagnostic->message.data);
  }
  return loomc_result_succeeded(result)
             ? loomc_ok_status()
             : loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                 "compilation produced diagnostics");
}

// The returned result owns the artifact bytes independently of the compiler,
// module, profile, and scratch workspace destroyed before this call returns.
static loomc_status_t compile_file(const char* input_path, const char* target,
                                   loomc_result_t** out_result) {
  *out_result = NULL;
  const loomc_allocator_t allocator = loomc_allocator_system();
  loomc_target_environment_t* environment = NULL;
  loomc_target_profile_t* profile = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;

  loomc_status_t status =
      loomc_target_environment_create_configured(allocator, &environment);
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = environment,
  };
  const loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_context_create(&context_options, allocator, &context);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_workspace_create(NULL, allocator, &workspace);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_target_profile_select(
        environment, loomc_make_cstring_view(target), allocator, &profile);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compiler_create(context, NULL, allocator, &compiler);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(input_path), NULL,
        allocator, &module, &result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_success(result);
  }
  loomc_result_release(result);
  result = NULL;

  // Roots are inferred from the source. The selected target owns the default
  // pipeline and artifact format, including its kernel ABI materialization.
  const loomc_compile_artifact_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(options),
      .target_profile = profile,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_compile_artifact(compiler, workspace, NULL, module, &options,
                                    allocator, &result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_success(result);
  }

  loomc_module_release(module);
  loomc_compiler_release(compiler);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_profile_release(profile);
  loomc_target_environment_release(environment);
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

static loomc_status_t write_executable(const loomc_result_t* result,
                                       const char* output_path) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      return loomc_artifact_write_to_path(artifact,
                                          loomc_make_cstring_view(output_path),
                                          loomc_allocator_system());
    }
  }
  return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                           "compilation produced no executable artifact");
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: compile_artifact INPUT TARGET OUTPUT\n"
            "Example: compile_artifact fill.loom x86:scalar fill.so\n");
    return 1;
  }
  loomc_result_t* result = NULL;
  loomc_status_t status = compile_file(argv[1], argv[2], &result);
  if (loomc_status_is_ok(status)) {
    status = write_executable(result, argv[3]);
  }
  loomc_result_release(result);
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  iree_status_fprint(stderr, iree_status_from_loomc(status));
  loomc_status_free(status);
  return 1;
}
