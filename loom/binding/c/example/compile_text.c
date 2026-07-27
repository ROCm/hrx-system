// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"

static const char kSourceText[] =
    "config.decl @model.hidden_size : %value: index where [range(%value, 0, "
    "8192), mul(%value, 16)]\n"
    "func.def public @entry() -> (index) {\n"
    "  %hidden = config.get @model.hidden_size : index\n"
    "  func.return %hidden : index\n"
    "}\n";

typedef struct compile_text_state_t {
  // Shared API context retained by sources, modules, and prepared tools.
  loomc_context_t* context;

  // Per-worker scratch storage used by deserialize and compile operations.
  loomc_workspace_t* workspace;

  // Immutable source containing the input text.
  loomc_source_t* source;

  // Mutable module produced from source and consumed by compile.
  loomc_module_t* module;

  // Immutable prepared compiler handle.
  loomc_compiler_t* compiler;

  // Immutable prepared pass program shared across compile invocations.
  loomc_pass_program_t* pass_program;

  // Last operation result, reset between phases.
  loomc_result_t* result;
} compile_text_state_t;

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

static void compile_text_state_initialize(compile_text_state_t* state) {
  memset(state, 0, sizeof(*state));
}

static void compile_text_state_deinitialize(compile_text_state_t* state) {
  loomc_result_release(state->result);
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_module_release(state->module);
  loomc_source_release(state->source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
}

static void compile_text_state_reset_result(compile_text_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
}

static loomc_status_t require_successful_result(const loomc_result_t* result,
                                                const char* failure_message) {
  if (loomc_result_succeeded(result)) {
    return loomc_ok_status();
  }
  print_result_diagnostics(result);
  return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, failure_message);
}

static const loomc_artifact_t* find_result_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact == NULL) {
      continue;
    }
    if (artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, format)) {
      return artifact;
    }
  }
  return NULL;
}

static loomc_status_t create_source(compile_text_state_t* state) {
  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("kernel.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  return loomc_source_create(&options, loomc_allocator_system(),
                             &state->source);
}

static loomc_status_t create_resources(compile_text_state_t* state) {
  loomc_status_t status =
      loomc_context_create(NULL, loomc_allocator_system(), &state->context);
  if (loomc_status_is_ok(status)) {
    status = loomc_workspace_create(NULL, loomc_allocator_system(),
                                    &state->workspace);
  }
  if (loomc_status_is_ok(status)) {
    status = create_source(state);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compiler_create(state->context, NULL,
                                   loomc_allocator_system(), &state->compiler);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_from_pipeline_text(
        state->context, loomc_make_cstring_view("canonicalize,dce"), NULL,
        loomc_allocator_system(), &state->pass_program, &state->result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "pass program preparation failed");
  }
  if (loomc_status_is_ok(status)) {
    compile_text_state_reset_result(state);
  }
  return status;
}

static loomc_status_t deserialize_source(compile_text_state_t* state) {
  loomc_status_t status = loomc_module_deserialize_from_source(
      state->context, state->workspace, state->source, NULL,
      loomc_allocator_system(), &state->module, &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "source deserialization failed");
  }
  if (loomc_status_is_ok(status)) {
    compile_text_state_reset_result(state);
  }
  return status;
}

static loomc_status_t compile_module(compile_text_state_t* state) {
  loomc_config_binding_t bindings[] = {
      {
          .key = loomc_make_cstring_view("@model.hidden_size"),
          .value = loomc_make_cstring_view("4096"),
      },
  };
  loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .module_name = loomc_make_cstring_view("jit_kernel"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT |
                        LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
      .config =
          {
              .bindings = bindings,
              .binding_count = 1,
              .flags = LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                       LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
          },
  };
  loomc_status_t status = loomc_compile_module(
      state->compiler, state->workspace, state->pass_program, state->module,
      &options, loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "compilation failed");
  }
  return status;
}

static loomc_status_t write_module_artifact_to_stdout(
    compile_text_state_t* state) {
  const loomc_artifact_t* artifact = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_MODULE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_TEXT));
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "compiled module text artifact was not produced");
  }
  loomc_status_t status = loomc_artifact_write_to_file(artifact, stdout);
  if (loomc_status_is_ok(status) && fflush(stdout) != 0) {
    status =
        loomc_make_status(LOOMC_STATUS_UNKNOWN, "failed to flush module text");
  }
  return status;
}

static loomc_status_t run_compile_text_example(void) {
  compile_text_state_t state;
  compile_text_state_initialize(&state);

  loomc_status_t status = create_resources(&state);
  if (loomc_status_is_ok(status)) {
    status = deserialize_source(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = compile_module(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = write_module_artifact_to_stdout(&state);
  }

  compile_text_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  loomc_status_t status = run_compile_text_example();
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return 1;
}
