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
#include "loomc/target/kernel.h"
#include "loomc/target/vm.h"

static const char kKernelExportName[] = "workload_grid";
static const uint64_t kElementCount = 1009;

typedef struct jit_kernel_state_t {
  // Target package linked into this embedding.
  loomc_target_environment_t* target_environment;

  // Shared context containing the core and target dialects.
  loomc_context_t* context;

  // Per-worker scratch reused across compiler operations.
  loomc_workspace_t* workspace;

  // Immutable ordinary Loom bytecode source.
  loomc_source_t* source;

  // Immutable request naming one host-launchable kernel root.
  loomc_request_t* request;

  // Reusable target profile selected for this example.
  loomc_target_profile_t* target_profile;

  // Reusable prepared compiler.
  loomc_compiler_t* compiler;

  // Reusable prepared target pipeline.
  loomc_pass_program_t* pass_program;

  // Immutable coupled executable and launch-configuration product.
  loomc_product_t* product;

  // Product projection for the requested kernel root.
  loomc_kernel_product_root_t root;

  // Loaded host companion for repeated launch evaluation.
  loomc_vm_launch_config_program_t* launch_program;

  // Program-local launch function bound by product ordinal.
  loomc_vm_launch_config_function_t launch_function;

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

static void jit_kernel_state_initialize(jit_kernel_state_t* state) {
  memset(state, 0, sizeof(*state));
  state->launch_function = loomc_vm_launch_config_function_invalid();
}

static void jit_kernel_state_reset_result(jit_kernel_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
}

static void jit_kernel_state_deinitialize(jit_kernel_state_t* state) {
  loomc_result_release(state->result);
  loomc_vm_launch_config_program_release(state->launch_program);
  loomc_product_release(state->product);
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_target_profile_release(state->target_profile);
  loomc_request_release(state->request);
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
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
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

// --8<-- [start:request]
static loomc_status_t prepare_kernel_request(jit_kernel_state_t* state) {
  loomc_link_index_builder_t* index_builder = NULL;
  loomc_link_index_t* link_index = NULL;
  loomc_status_t status = loomc_link_index_builder_create(
      state->context, NULL, loomc_allocator_system(), &index_builder);

  const loomc_link_index_source_options_t source_options = {
      .provider_name = loomc_make_cstring_view("guide-jit-kernel"),
      .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_add_source(index_builder, state->source,
                                                 &source_options, NULL);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_finish(index_builder, &link_index,
                                             &state->result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "source indexing failed");
  }

  loomc_link_index_symbol_t symbol = {0};
  if (loomc_status_is_ok(status) &&
      !loomc_link_index_lookup_global(
          link_index, loomc_make_cstring_view(kKernelExportName), &symbol)) {
    status = loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                               "kernel export was not found");
  }
  if (loomc_status_is_ok(status) &&
      (symbol.provider_module_ordinal > UINT32_MAX ||
       symbol.module_symbol_ordinal > UINT32_MAX)) {
    status = loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                               "kernel root exceeds the request domain");
  }

  const loomc_request_root_t root = {
      .module_ordinal = (uint32_t)symbol.provider_module_ordinal,
      .symbol_ordinal = (uint32_t)symbol.module_symbol_ordinal,
      .goal = LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_request_create(loomc_kernel_product_descriptor(),
                                  state->source, &root, 1, NULL, 0,
                                  loomc_allocator_system(), &state->request);
  }
  if (loomc_status_is_ok(status)) {
    jit_kernel_state_reset_result(state);
  }

  loomc_link_index_release(link_index);
  loomc_link_index_builder_release(index_builder);
  return status;
}
// --8<-- [end:request]

// --8<-- [start:compile]
static loomc_status_t build_kernel_product(jit_kernel_state_t* state) {
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
  };
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
  loomc_status_t status = loomc_kernel_product_build_request(
      state->compiler, state->workspace, state->pass_program, state->request,
      &compile_options, &emit_options, loomc_allocator_system(),
      &state->product, &state->result);
  if (loomc_status_is_ok(status)) {
    status =
        require_successful_result(state->result, "kernel compilation failed");
  }
  if (loomc_status_is_ok(status) &&
      !loomc_kernel_product_root_at(state->product, 0, &state->root)) {
    status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                               "kernel product has no requested root");
  }
  if (loomc_status_is_ok(status)) {
    jit_kernel_state_reset_result(state);
  }
  return status;
}
// --8<-- [end:compile]

// --8<-- [start:launch-config]
static loomc_status_t prepare_and_evaluate_launch(
    jit_kernel_state_t* state, loomc_launch_config_t* out_launch_config) {
  const loomc_artifact_t* artifact = loomc_product_artifact_at(
      state->product, state->root.launch_config_artifact_ordinal);
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "kernel product has no launch-config artifact");
  }

  loomc_status_t status = loomc_vm_launch_config_program_load(
      artifact, loomc_allocator_system(), &state->launch_program);
  if (loomc_status_is_ok(status)) {
    if (!loomc_vm_launch_config_program_function_at(
            state->launch_program, state->root.launch_config_function_ordinal,
            &state->launch_function)) {
      status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                 "launch function ordinal is invalid");
    }
  }

  const uint64_t workload_argument_bits[] = {kElementCount};
  if (loomc_status_is_ok(status)) {
    status = loomc_vm_launch_config_program_invoke(
        state->launch_program, state->launch_function, workload_argument_bits,
        1, out_launch_config);
  }
  return status;
}
// --8<-- [end:launch-config]

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

  const loomc_artifact_t* executable = loomc_product_artifact_at(
      state->product, state->root.executable_artifact_ordinal);
  if (executable == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "AMDGPU executable artifact was not produced");
  }
  const uint64_t executable_length =
      loomc_byte_sequence_length(executable->contents);
  if (executable_length < 4) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "AMDGPU executable artifact is too small");
  }

  printf("workload=%llu workgroups=%ux%ux%u workgroup_size=%ux%ux%u\n",
         (unsigned long long)kElementCount, launch_config->workgroup_count.x,
         launch_config->workgroup_count.y, launch_config->workgroup_count.z,
         launch_config->workgroup_size.x, launch_config->workgroup_size.y,
         launch_config->workgroup_size.z);
  printf("artifact=%.*s format=%.*s bytes=%llu\n",
         (int)executable->identifier.size, executable->identifier.data,
         (int)executable->format.size, executable->format.data,
         (unsigned long long)executable_length);
  return loomc_ok_status();
}

static loomc_status_t run_jit_kernel(const char* source_path,
                                     const char* target) {
  jit_kernel_state_t state;
  jit_kernel_state_initialize(&state);

  loomc_status_t status = prepare_compiler(&state, source_path, target);
  if (loomc_status_is_ok(status)) {
    status = prepare_kernel_request(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = build_kernel_product(&state);
  }

  loomc_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
  };
  if (loomc_status_is_ok(status)) {
    status = prepare_and_evaluate_launch(&state, &launch_config);
  }
  if (loomc_status_is_ok(status)) {
    status = inspect_products(&state, &launch_config);
  }

  jit_kernel_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: jit_kernel <kernel.loombc> [target]\n");
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
