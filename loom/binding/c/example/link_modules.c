// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"

static const char kHarnessText[] =
    "config.decl @model.hidden_size : %value: index where [range(%value, 0, "
    "8192), mul(%value, 16)]\n"
    "func.decl @identity(%x: index) -> (index)\n"
    "func.def public @caller() -> (index) {\n"
    "  %hidden = config.get @model.hidden_size : index\n"
    "  %y = func.call @identity(%hidden) : (index) -> (index)\n"
    "  func.return %y : index\n"
    "}\n";

static const char kLibraryText[] =
    "func.def public @identity(%x: index) -> (index) {\n"
    "  func.return %x : index\n"
    "}\n";

typedef struct link_modules_state_t {
  // Shared API context retained by sources, indexes, and prepared tools.
  loomc_context_t* context;

  // Per-worker scratch storage used for the link invocation.
  loomc_workspace_t* workspace;

  // Primary source that defines the caller and imports the library symbol.
  loomc_source_t* harness_source;

  // Library source that provides the imported implementation.
  loomc_source_t* library_source;

  // Mutable builder used only until the frozen link index is finished.
  loomc_link_index_builder_t* index_builder;

  // Frozen source/library index reused by the prepared linker.
  loomc_link_index_t* link_index;

  // Immutable prepared linker handle.
  loomc_linker_t* linker;

  // Immutable prepared compiler handle.
  loomc_compiler_t* compiler;

  // Immutable prepared pass program used by compile invocations.
  loomc_pass_program_t* pass_program;

  // Linked module output owned by this example state.
  loomc_module_t* linked_module;

  // Last operation result, reset between index and link phases.
  loomc_result_t* result;
} link_modules_state_t;

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

static loomc_status_t create_text_source(const char* identifier,
                                         const char* contents,
                                         loomc_source_t** out_source) {
  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view(identifier),
      .contents = loomc_make_byte_span(contents, strlen(contents)),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  return loomc_source_create(&options, loomc_allocator_system(), out_source);
}

static void link_modules_state_initialize(link_modules_state_t* state) {
  memset(state, 0, sizeof(*state));
}

static void link_modules_state_deinitialize(link_modules_state_t* state) {
  loomc_result_release(state->result);
  loomc_module_release(state->linked_module);
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_linker_release(state->linker);
  loomc_link_index_builder_release(state->index_builder);
  loomc_link_index_release(state->link_index);
  loomc_source_release(state->library_source);
  loomc_source_release(state->harness_source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
}

static void link_modules_state_reset_result(link_modules_state_t* state) {
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

static loomc_status_t write_module_artifact_to_stdout(
    link_modules_state_t* state) {
  const loomc_artifact_t* artifact = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_MODULE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_TEXT));
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "compiled module text artifact was not produced");
  }
  loomc_status_t status = loomc_artifact_write_to_file(artifact, stdout);
  if (loomc_status_is_ok(status) && fflush(stdout) != 0) {
    status = loomc_make_status(LOOMC_STATUS_UNKNOWN,
                               "failed to flush linked module text");
  }
  return status;
}

static loomc_status_t create_resources(link_modules_state_t* state) {
  loomc_status_t status =
      loomc_context_create(NULL, loomc_allocator_system(), &state->context);
  if (loomc_status_is_ok(status)) {
    status = loomc_workspace_create(NULL, loomc_allocator_system(),
                                    &state->workspace);
  }
  if (loomc_status_is_ok(status)) {
    status = create_text_source("harness.loom", kHarnessText,
                                &state->harness_source);
  }
  if (loomc_status_is_ok(status)) {
    status = create_text_source("library.loom", kLibraryText,
                                &state->library_source);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_create(
        state->context, NULL, loomc_allocator_system(), &state->index_builder);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_linker_create(state->context, NULL, loomc_allocator_system(),
                                 &state->linker);
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
    link_modules_state_reset_result(state);
  }
  return status;
}

static loomc_status_t add_source_to_index(link_modules_state_t* state,
                                          loomc_source_t* source,
                                          loomc_string_view_t provider_name,
                                          loomc_link_provider_role_t role) {
  loomc_link_index_source_options_t options = {
      .provider_name = provider_name,
      .role = role,
  };
  return loomc_link_index_builder_add_source(state->index_builder, source,
                                             &options, NULL);
}

static loomc_status_t build_link_index(link_modules_state_t* state) {
  loomc_status_t status = add_source_to_index(
      state, state->harness_source, loomc_make_cstring_view("harness"),
      LOOMC_LINK_PROVIDER_ROLE_INPUT);
  if (loomc_status_is_ok(status)) {
    status = add_source_to_index(state, state->library_source,
                                 loomc_make_cstring_view("library"),
                                 LOOMC_LINK_PROVIDER_ROLE_LIBRARY);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_finish(
        state->index_builder, &state->link_index, &state->result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "link index construction failed");
  }
  if (loomc_status_is_ok(status)) {
    link_modules_state_reset_result(state);
  }
  return status;
}

static loomc_status_t link_module(link_modules_state_t* state) {
  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("@caller"),
  };
  loomc_link_options_t link_options = {
      .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
      .structure_size = sizeof(link_options),
      .link_index = state->link_index,
      .root_symbols = roots,
      .root_symbol_count = 1,
  };

  loomc_status_t status =
      loomc_link_module(state->linker, state->workspace, &link_options,
                        &state->linked_module, &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "linking failed");
  }
  if (loomc_status_is_ok(status)) {
    link_modules_state_reset_result(state);
  }
  return status;
}

static loomc_status_t compile_linked_module(link_modules_state_t* state) {
  loomc_config_binding_t bindings[] = {
      {
          .key = loomc_make_cstring_view("@model.hidden_size"),
          .value = loomc_make_cstring_view("4096"),
      },
  };
  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .module_name = loomc_make_cstring_view("jit_kernel"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT |
                        LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
      .config =
          {
              .bindings = bindings,
              .binding_count = 1,
              .json_object =
                  loomc_make_cstring_view("{\"model\":{\"hidden_size\":2048}}"),
              .flags = LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                       LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
          },
  };

  loomc_status_t status = loomc_compile_module(
      state->compiler, state->workspace, state->pass_program,
      state->linked_module, &compile_options, loomc_allocator_system(),
      &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "compilation failed");
  }
  return status;
}

static loomc_status_t run_link_modules_example(void) {
  link_modules_state_t state;
  link_modules_state_initialize(&state);

  loomc_status_t status = create_resources(&state);
  if (loomc_status_is_ok(status)) {
    status = build_link_index(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = link_module(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = compile_linked_module(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = write_module_artifact_to_stdout(&state);
  }

  link_modules_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  loomc_status_t status = run_link_modules_example();
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return 1;
}
