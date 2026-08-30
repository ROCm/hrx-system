// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"
#include "loomc/target/cmd.h"

static const char kSourceText[] =
    "kernel.def @local() {\n"
    "  %one = index.constant 1 : index\n"
    "  kernel.launch.config workgroups(%one, %one, %one) "
    "workgroup_size(%one, %one, %one) : index\n"
    "} launch(%storage: buffer) {\n"
    "  kernel.return\n"
    "}\n"
    "\n"
    "kernel.entry.decl @external(%storage: buffer)\n"
    "\n"
    "command.program.def public @run() launch(%storage: buffer) {\n"
    "  %one = index.constant 1 : index\n"
    "  kernel.launch @local(%storage) : (buffer)\n"
    "  kernel.dispatch @external[%one](%storage) : [index](buffer)\n"
    "  command.return\n"
    "}\n";

typedef struct product_frontier_state_t {
  // Allocator shared by every independently owned object in the example.
  loomc_allocator_t allocator;

  // Shared source and compiler context.
  loomc_context_t* context;

  // Invocation-local mutable compiler scratch.
  loomc_workspace_t* workspace;

  // Immutable source indexed as the command and kernel provider universe.
  loomc_source_t* source;

  // Frozen provider index shared by command-product construction.
  loomc_link_index_t* link_index;

  // Prepared compiler used for independently published kernel requests.
  loomc_compiler_t* compiler;

  // Prepared kernel compilation pipeline.
  loomc_pass_program_t* pass_program;

  // Parent command product retained through result inspection.
  loomc_product_t* command_product;

  // Source-backed kernel request transferred by command construction.
  loomc_request_t* kernel_request;

  // Independently compiled child kernel product.
  loomc_product_t* kernel_product;
} product_frontier_state_t;

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
    if (diagnostic == NULL) continue;
    fprintf(stderr, "%.*s: %.*s\n", (int)diagnostic->code.size,
            diagnostic->code.data, (int)diagnostic->message.size,
            diagnostic->message.data);
  }
}

static loomc_status_t require_successful_result(const loomc_result_t* result,
                                                const char* message) {
  if (result != NULL && loomc_result_succeeded(result)) {
    return loomc_ok_status();
  }
  if (result != NULL) print_result_diagnostics(result);
  return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, message);
}

static loomc_status_t require_condition(bool condition, const char* message) {
  return condition
             ? loomc_ok_status()
             : loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, message);
}

static void product_frontier_state_initialize(product_frontier_state_t* state) {
  memset(state, 0, sizeof(*state));
  state->allocator = loomc_allocator_system();
}

static void product_frontier_state_deinitialize(
    product_frontier_state_t* state) {
  loomc_product_release(state->kernel_product);
  loomc_request_release(state->kernel_request);
  loomc_product_release(state->command_product);
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_link_index_release(state->link_index);
  loomc_source_release(state->source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
}

// --8<-- [start:publish-request]
static loomc_status_t publish_kernel_request(void* user_data,
                                             loomc_request_t* request) {
  product_frontier_state_t* state = (product_frontier_state_t*)user_data;
  if (state->kernel_request != NULL) {
    loomc_request_release(request);
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "example published more than one kernel request");
  }
  // Ownership transfers at callback entry. A concurrent embedding would
  // enqueue this reference and commit its parent binding only after the parent
  // command operation succeeds.
  state->kernel_request = request;
  return loomc_ok_status();
}
// --8<-- [end:publish-request]

// --8<-- [start:prepare]
static loomc_status_t prepare_product_frontier(
    product_frontier_state_t* state, loomc_host_size_t* out_root_ordinal) {
  *out_root_ordinal = 0;
  loomc_status_t status =
      loomc_context_create(NULL, state->allocator, &state->context);
  if (loomc_status_is_ok(status)) {
    status = loomc_workspace_create(NULL, state->allocator, &state->workspace);
  }

  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("product_frontier.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  if (loomc_status_is_ok(status)) {
    status =
        loomc_source_create(&source_options, state->allocator, &state->source);
  }

  loomc_link_index_builder_t* builder = NULL;
  loomc_result_t* index_result = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_create(state->context, NULL,
                                             state->allocator, &builder);
  }
  const loomc_link_index_source_options_t provider_options = {
      .provider_name = loomc_make_cstring_view("product-frontier"),
      .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_add_source(builder, state->source,
                                                 &provider_options, NULL);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_finish(builder, &state->link_index,
                                             &index_result);
  }
  if (loomc_status_is_ok(status)) {
    status =
        require_successful_result(index_result, "provider indexing failed");
  }

  loomc_link_index_symbol_t root_symbol = {0};
  if (loomc_status_is_ok(status) &&
      !loomc_link_index_lookup_global(
          state->link_index, loomc_make_cstring_view("run"), &root_symbol)) {
    status = loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                               "command root @run was not indexed");
  }
  if (loomc_status_is_ok(status)) {
    *out_root_ordinal = root_symbol.ordinal;
    status = loomc_compiler_create(state->context, NULL, state->allocator,
                                   &state->compiler);
  }

  loomc_result_t* pass_result = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_from_pipeline_text(
        state->context, loomc_make_cstring_view("canonicalize,cse"), NULL,
        state->allocator, &state->pass_program, &pass_result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(pass_result,
                                       "kernel pipeline preparation failed");
  }

  loomc_result_release(pass_result);
  loomc_result_release(index_result);
  loomc_link_index_builder_release(builder);
  return status;
}
// --8<-- [end:prepare]

// --8<-- [start:build-command]
static loomc_status_t build_command_product(product_frontier_state_t* state,
                                            loomc_host_size_t root_ordinal) {
  const loomc_cmd_program_product_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      .structure_size = sizeof(options),
      .link_index = state->link_index,
      .root_symbol_ordinals = &root_ordinal,
      .root_symbol_count = 1,
      .request_sink =
          {
              .publish = publish_kernel_request,
              .user_data = state,
          },
  };
  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_cmd_program_product_build(
      state->workspace, &options, state->allocator, &state->command_product,
      &result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(result,
                                       "command-product construction failed");
  }
  if (loomc_status_is_ok(status)) {
    status = require_condition(
        loomc_cmd_program_product_program_count(state->command_product) == 1 &&
            loomc_product_requirement_count(state->command_product) == 2 &&
            state->kernel_request != NULL,
        "command product does not expose the expected frontier");
  }
  loomc_result_release(result);
  return status;
}
// --8<-- [end:build-command]

// --8<-- [start:compile-request]
static loomc_status_t compile_kernel_request(product_frontier_state_t* state) {
  const loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .module_name = loomc_make_cstring_view("product-frontier-kernel"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
  };
  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_compile_request(
      state->compiler, state->workspace, state->pass_program,
      state->kernel_request, &options, state->allocator, &state->kernel_product,
      &result);
  if (loomc_status_is_ok(status)) {
    status =
        require_successful_result(result, "kernel request compilation failed");
  }
  const loomc_artifact_t* artifact = NULL;
  if (loomc_status_is_ok(status)) {
    artifact = loomc_product_artifact_at(state->kernel_product, 0);
    status = require_condition(
        loomc_product_descriptor(state->kernel_product) ==
                loomc_compiled_module_product_descriptor() &&
            loomc_product_export_count(state->kernel_product) == 1 &&
            loomc_product_requirement_count(state->kernel_product) == 0 &&
            artifact != NULL &&
            loomc_string_view_equal(
                artifact->format,
                loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE)),
        "kernel product does not satisfy the child request");
  }
  loomc_result_release(result);
  return status;
}
// --8<-- [end:compile-request]

static loomc_status_t print_product_summary(
    const product_frontier_state_t* state) {
  loomc_cmd_program_t program = {0};
  if (!loomc_cmd_program_product_program_at(state->command_product, 0,
                                            &program)) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "command program metadata is unavailable");
  }
  printf("command=%.*s programs=1 requirements=%zu kernel_products=1\n",
         (int)program.symbol.size, program.symbol.data,
         (size_t)loomc_product_requirement_count(state->command_product));
  return loomc_ok_status();
}

int main(void) {
  product_frontier_state_t state;
  product_frontier_state_initialize(&state);

  loomc_host_size_t root_ordinal = 0;
  loomc_status_t status = prepare_product_frontier(&state, &root_ordinal);
  if (loomc_status_is_ok(status)) {
    status = build_command_product(&state, root_ordinal);
  }
  if (loomc_status_is_ok(status)) {
    status = compile_kernel_request(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = print_product_summary(&state);
  }

  product_frontier_state_deinitialize(&state);
  if (!loomc_status_is_ok(status)) {
    print_status(status);
    loomc_status_free(status);
    return 1;
  }
  return 0;
}
