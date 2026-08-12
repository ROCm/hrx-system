// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"

static const char kKernelExportName[] = "workload_grid";
static const uint64_t kElementCount = 1009;

typedef struct jit_kernel_state_t {
  // Target package linked into this embedding.
  loomc_target_environment_t* target_environment;

  // Shared context containing the core and target dialects.
  loomc_context_t* context;

  // Per-worker scratch reused across compiler operations.
  loomc_workspace_t* workspace;

  // Immutable input source.
  loomc_source_t* source;

  // Mutable module deserialized for this invocation.
  loomc_module_t* module;

  // Reusable target profile selected for this example.
  loomc_target_profile_t* target_profile;

  // Reusable prepared compiler.
  loomc_compiler_t* compiler;

  // Reusable prepared target pipeline.
  loomc_pass_program_t* pass_program;

  // Loaded host companion for repeated launch evaluation.
  loomc_launch_config_program_t* launch_program;

  // Last compiler operation result.
  loomc_result_t* result;
} jit_kernel_state_t;

static void print_status(loomc_status_t status) {
  char buffer[1024] = {0};
  loomc_host_size_t length = 0;
  loomc_status_format(status, sizeof(buffer), buffer, &length);
  fprintf(stderr, "%.*s\n", (int)length, buffer);
}

static void print_result_diagnostics(const loomc_result_t* result) {
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic == NULL) {
      continue;
    }
    fprintf(stderr, "%.*s: %.*s\n", (int)diagnostic->code.size,
            diagnostic->code.data, (int)diagnostic->message.size,
            diagnostic->message.data);
  }
}

static loomc_status_t require_successful_result(const loomc_result_t* result,
                                                const char* failure_message) {
  if (result != NULL && loomc_result_succeeded(result)) {
    return loomc_ok_status();
  }
  if (result != NULL) {
    print_result_diagnostics(result);
  }
  return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, failure_message);
}

static const loomc_artifact_t* find_result_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != NULL && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, format)) {
      return artifact;
    }
  }
  return NULL;
}

static void jit_kernel_state_initialize(jit_kernel_state_t* state) {
  memset(state, 0, sizeof(*state));
}

static void jit_kernel_state_reset_result(jit_kernel_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
}

static void jit_kernel_state_deinitialize(jit_kernel_state_t* state) {
  loomc_result_release(state->result);
  loomc_launch_config_program_release(state->launch_program);
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_target_profile_release(state->target_profile);
  loomc_module_release(state->module);
  loomc_source_release(state->source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
  loomc_target_environment_release(state->target_environment);
}

// --8<-- [start:prepare]
static loomc_status_t prepare_compiler(jit_kernel_state_t* state,
                                       const char* source_path,
                                       const char* target) {
  // --8<-- [start:target-context]
  loomc_status_t status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &state->target_environment);

  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = state->target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_context_create(&context_options, loomc_allocator_system(),
                                  &state->context);
  }
  // --8<-- [end:target-context]

  // --8<-- [start:source]
  if (loomc_status_is_ok(status)) {
    status = loomc_workspace_create(NULL, loomc_allocator_system(),
                                    &state->workspace);
  }

  loomc_source_load_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create_from_path(
        loomc_make_cstring_view(source_path), &source_options,
        loomc_allocator_system(), &state->source);
  }
  // --8<-- [end:source]

  // --8<-- [start:profile-pipeline]
  loomc_amdgpu_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_make_cstring_view("guide-amdgpu"),
      .identity =
          {
              .target = loomc_make_cstring_view(target),
          },
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_target_profile_create_amdgpu(
        state->target_environment, &profile_options, loomc_allocator_system(),
        &state->target_profile);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compiler_create(state->context, NULL,
                                   loomc_allocator_system(), &state->compiler);
  }

  loomc_target_pipeline_options_t pipeline_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(pipeline_options),
      .identifier = loomc_make_cstring_view("guide-prepared-low"),
      .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .source_to_low_max_errors = 20,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_from_target_pipeline(
        state->context, &pipeline_options, loomc_allocator_system(),
        &state->pass_program, &state->result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "target pipeline preparation failed");
  }
  if (loomc_status_is_ok(status)) {
    jit_kernel_state_reset_result(state);
  }
  // --8<-- [end:profile-pipeline]
  return status;
}
// --8<-- [end:prepare]

static loomc_status_t deserialize_source(jit_kernel_state_t* state) {
  loomc_status_t status = loomc_module_deserialize_from_source(
      state->context, state->workspace, state->source, NULL,
      loomc_allocator_system(), &state->module, &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "source deserialization failed");
  }
  if (loomc_status_is_ok(status)) {
    jit_kernel_state_reset_result(state);
  }
  return status;
}

// --8<-- [start:compile]
static loomc_status_t compile_kernel(jit_kernel_state_t* state) {
  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view(kKernelExportName),
      .target_profile = state->target_profile,
  };
  loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .specializations = &specialization,
      .specialization_count = 1,
  };
  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("guide_jit_kernel"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
  };
  loomc_status_t status = loomc_compile_module(
      state->compiler, state->workspace, state->pass_program, state->module,
      &compile_options, loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status =
        require_successful_result(state->result, "kernel compilation failed");
  }
  return status;
}
// --8<-- [end:compile]

// --8<-- [start:launch-config]
static loomc_status_t prepare_and_evaluate_launch(
    jit_kernel_state_t* state, loomc_launch_config_t* out_launch_config) {
  const loomc_artifact_t* artifact = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE));
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "launch-config artifact was not produced");
  }

  loomc_status_t status = loomc_launch_config_program_load(
      artifact, NULL, NULL, loomc_allocator_system(), &state->launch_program);
  if (loomc_status_is_ok(status)) {
    jit_kernel_state_reset_result(state);
  }
  loomc_launch_config_function_t function =
      loomc_launch_config_function_invalid();
  if (loomc_status_is_ok(status)) {
    status = loomc_launch_config_program_lookup_function(
        state->launch_program, loomc_make_cstring_view(kKernelExportName),
        &function);
  }

  const uint64_t workload_argument_bits[] = {kElementCount};
  if (loomc_status_is_ok(status)) {
    status = loomc_launch_config_program_invoke(state->launch_program, function,
                                                workload_argument_bits, 1,
                                                out_launch_config);
  }
  return status;
}
// --8<-- [end:launch-config]

// --8<-- [start:emit]
static loomc_status_t emit_executable(jit_kernel_state_t* state) {
  loomc_amdgpu_emit_options_t amdgpu_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      .structure_size = sizeof(amdgpu_options),
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = &amdgpu_options,
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .identifier = loomc_make_cstring_view("guide_jit_kernel.hsaco"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_status_t status = loomc_emit_module(
      state->target_environment, state->workspace, state->module, &emit_options,
      loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status =
        require_successful_result(state->result, "executable emission failed");
  }
  return status;
}
// --8<-- [end:emit]

static loomc_status_t inspect_products(
    jit_kernel_state_t* state, const loomc_launch_config_t* launch_config) {
  if (launch_config->workgroup_count.x != 16 ||
      launch_config->workgroup_count.y != 1 ||
      launch_config->workgroup_count.z != 1 ||
      launch_config->workgroup_size.x != 64 ||
      launch_config->workgroup_size.y != 1 ||
      launch_config->workgroup_size.z != 1) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "unexpected launch configuration");
  }

  const loomc_artifact_t* executable = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
  if (executable == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "AMDGPU executable artifact was not produced");
  }
  static const uint8_t kElfMagic[] = {0x7F, 'E', 'L', 'F'};
  if (executable->contents.data_length < sizeof(kElfMagic) ||
      memcmp(executable->contents.data, kElfMagic, sizeof(kElfMagic)) != 0) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "AMDGPU executable is not an ELF artifact");
  }

  printf("workload=%llu workgroups=%ux%ux%u workgroup_size=%ux%ux%u\n",
         (unsigned long long)kElementCount, launch_config->workgroup_count.x,
         launch_config->workgroup_count.y, launch_config->workgroup_count.z,
         launch_config->workgroup_size.x, launch_config->workgroup_size.y,
         launch_config->workgroup_size.z);
  printf("artifact=%.*s format=%.*s bytes=%zu\n",
         (int)executable->identifier.size, executable->identifier.data,
         (int)executable->format.size, executable->format.data,
         (size_t)executable->contents.data_length);
  return loomc_ok_status();
}

static loomc_status_t run_jit_kernel(const char* source_path,
                                     const char* target) {
  jit_kernel_state_t state;
  jit_kernel_state_initialize(&state);

  loomc_status_t status = prepare_compiler(&state, source_path, target);
  if (loomc_status_is_ok(status)) {
    status = deserialize_source(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = compile_kernel(&state);
  }

  loomc_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
  };
  if (loomc_status_is_ok(status)) {
    status = prepare_and_evaluate_launch(&state, &launch_config);
  }
  if (loomc_status_is_ok(status)) {
    status = emit_executable(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = inspect_products(&state, &launch_config);
  }

  jit_kernel_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: jit_kernel <kernel.loom> [target]\n");
    return 64;
  }
  const char* target = argc == 3 ? argv[2] : "gfx11-generic";
  loomc_status_t status = run_jit_kernel(argv[1], target);
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return 1;
}
