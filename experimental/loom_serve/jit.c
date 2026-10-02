// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/jit.h"

#include <stdio.h>

#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu/base.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd/program.h"
#include "loomc/target/vm.h"

typedef struct jit_compiler_t {
  // Target package retained by the context and prepared compiler.
  loomc_target_environment_t* environment;
  // Shared immutable parser and dialect configuration.
  loomc_context_t* context;
  // Scratch reused by serialized cold-path compilation calls.
  loomc_workspace_t* workspace;
  // Prepared compiler reused for every kernel in the residency.
  loomc_compiler_t* compiler;
  // Source-to-prepared-low pipeline for the selected target package.
  loomc_pass_program_t* pipeline;
  // Live device facts, or the portable VM profile.
  loomc_target_profile_t* profile;
} jit_compiler_t;

struct loom_serve_jit_t {
  // Allocation policy shared by compiled stages.
  iree_allocator_t allocator;
  // Retained device backing the live target and executable loads.
  iree_hal_device_t* device;
  // Retained exact execution queue.
  iree_hal_queue_t* dispatch;
  // Exact executable target borrowed from device.
  const iree_hal_executable_target_t* target;
  // Model-wide AMDGPU compiler state.
  jit_compiler_t kernel;
  // Frozen source providers, shared across all shape specializations.
  loomc_link_index_t* index;
};

struct loom_serve_jit_stage_t {
  // Allocation policy used for owned bytes and entry storage.
  iree_allocator_t allocator;
  // Owned contiguous portable command image.
  iree_const_byte_span_t image;
  // Parsed program borrowing image.
  loom_cmd_program_t program;
  // Loaded entries in the command program's local slot order.
  loom_serve_command_entry_t* entries;
  // Number of initialized entry slots, including null entries on failure.
  iree_host_size_t entry_count;
};

// Consumes the result even when the operation returned an infrastructure error.
// Domain diagnostics are printed before returning a terminal compile failure.
static iree_status_t jit_result(loomc_status_t operation,
                                loomc_result_t* result) {
  for (iree_host_size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    const loomc_string_view_t source =
        loomc_source_identifier(diagnostic->range.source);
    fprintf(stderr, "%.*s:%u:%u: %.*s: %.*s\n", (int)source.size, source.data,
            diagnostic->range.start_line, diagnostic->range.start_column,
            (int)diagnostic->code.size, diagnostic->code.data,
            (int)diagnostic->message.size, diagnostic->message.data);
  }
  iree_status_t status = iree_status_from_loomc(operation);
  if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Loom JIT failed; see source diagnostics above");
  }
  loomc_result_release(result);
  return status;
}

static void jit_compiler_deinitialize(jit_compiler_t* compiler) {
  loomc_pass_program_release(compiler->pipeline);
  loomc_compiler_release(compiler->compiler);
  loomc_workspace_release(compiler->workspace);
  loomc_context_release(compiler->context);
  loomc_target_profile_release(compiler->profile);
  loomc_target_environment_release(compiler->environment);
}

static iree_status_t jit_compiler_initialize(
    const loomc_sanitizer_options_t* sanitizer, iree_allocator_t allocator,
    jit_compiler_t* compiler) {
  const loomc_allocator_t ca = loomc_allocator_from_iree(allocator);
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .target_environment = compiler->environment,
  };
  const loomc_context_options_t context_options = {.next = &target_options};
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_context_create(&context_options, ca, &compiler->context)));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_workspace_create(NULL, ca, &compiler->workspace)));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_compiler_create(compiler->context, NULL, ca, &compiler->compiler)));
  loomc_result_t* result = NULL;
  const loomc_target_pipeline_options_t pipeline_options = {.next = sanitizer};
  const loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      compiler->context, &pipeline_options, ca, &compiler->pipeline, &result);
  return jit_result(status, result);
}

static iree_status_t jit_read_source(iree_string_view_t path,
                                     iree_allocator_t allocator,
                                     loomc_source_t** out_source) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_read(path, allocator, &contents));
  const loomc_source_options_t options = {
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_string_view_from_iree(path),
      .contents = loomc_byte_span_from_iree(contents->const_buffer),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  iree_status_t status = iree_status_from_loomc(loomc_source_create(
      &options, loomc_allocator_from_iree(allocator), out_source));
  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t jit_index_sources(loom_serve_jit_t* jit,
                                       iree_string_view_t directory) {
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(directory, IREE_SV("sources.txt"),
                                           jit->allocator, &path));
  iree_io_file_contents_t* catalog = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(path), jit->allocator, &catalog);
  iree_allocator_free(jit->allocator, path);
  IREE_RETURN_IF_ERROR(status);
  loomc_link_index_builder_t* builder = NULL;
  status = iree_status_from_loomc(loomc_link_index_builder_create(
      jit->kernel.context, NULL, loomc_allocator_from_iree(jit->allocator),
      &builder));
  iree_string_view_t remaining =
      iree_make_string_view((const char*)catalog->const_buffer.data,
                            catalog->const_buffer.data_length);
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line;
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line) || line.data[0] == '#') {
      continue;
    }
    path = NULL;
    status = iree_file_path_join(directory, line, jit->allocator, &path);
    loomc_source_t* source = NULL;
    if (iree_status_is_ok(status)) {
      status = jit_read_source(iree_make_cstring_view(path), jit->allocator,
                               &source);
    }
    if (iree_status_is_ok(status)) {
      const loomc_link_index_source_options_t options = {
          .provider_name = loomc_string_view_from_iree(line),
          .role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
      };
      status = iree_status_from_loomc(
          loomc_link_index_builder_add_source(builder, source, &options, NULL));
    }
    loomc_source_release(source);
    iree_allocator_free(jit->allocator, path);
  }
  if (iree_status_is_ok(status)) {
    loomc_result_t* result = NULL;
    const loomc_status_t operation =
        loomc_link_index_builder_finish(builder, &jit->index, &result);
    status = jit_result(operation, result);
  }
  loomc_link_index_builder_release(builder);
  iree_io_file_contents_free(catalog);
  return status;
}

void loom_serve_jit_destroy(loom_serve_jit_t* jit) {
  if (!jit) {
    return;
  }
  loomc_link_index_release(jit->index);
  jit_compiler_deinitialize(&jit->kernel);
  iree_hal_queue_release(jit->dispatch);
  iree_hal_device_release(jit->device);
  iree_allocator_free(jit->allocator, jit);
}

iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    iree_string_view_t source_directory,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit) {
  *out_jit = NULL;
  const iree_hal_physical_device_affinity_t affinity =
      iree_hal_queue_family_spec(iree_hal_queue_family(dispatch))
          ->physical_device_affinity;
  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      .physical_device_affinity = affinity,
  };
  const iree_hal_executable_target_selection_result_t selected =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(device), &selection);
  if (selected.outcome !=
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "JIT requires one exact AMDGPU queue target");
  }
  loom_serve_jit_t* jit = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*jit), (void**)&jit));
  jit->allocator = host_allocator;
  jit->device = device;
  iree_hal_device_retain(device);
  jit->dispatch = dispatch;
  iree_hal_queue_retain(dispatch);
  jit->target = selected.target;
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  iree_status_t status = iree_status_from_loomc(
      loomc_target_environment_create_amdgpu(ca, &jit->kernel.environment));
  if (iree_status_is_ok(status)) {
    status = jit_compiler_initialize(sanitizer, host_allocator, &jit->kernel);
  }
  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_iree_hal_profile_options_t options = {
        .device = device, .physical_device_affinity = affinity};
    loomc_result_t* result = NULL;
    const loomc_status_t operation =
        loomc_target_profile_create_amdgpu_iree_hal(
            jit->kernel.environment, &options, ca, &jit->kernel.profile,
            &result);
    status = jit_result(operation, result);
  }
  if (iree_status_is_ok(status)) {
    status = jit_index_sources(jit, source_directory);
  }
  if (iree_status_is_ok(status)) {
    *out_jit = jit;
  } else {
    loom_serve_jit_destroy(jit);
  }
  return status;
}

// Keeps prepared-low function facts in the same module through native emission.
static iree_status_t jit_emit(
    jit_compiler_t* compiler, loomc_source_t* source,
    iree_host_size_t specialization_count,
    const loomc_target_specialization_t* specializations,
    iree_string_view_t format, iree_allocator_t allocator,
    iree_const_byte_span_t* out_image) {
  *out_image = iree_const_byte_span_empty();
  const loomc_allocator_t ca = loomc_allocator_from_iree(allocator);
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  loomc_status_t operation = loomc_module_deserialize_from_source(
      compiler->context, compiler->workspace, source, NULL, ca, &module,
      &result);
  iree_status_t status = jit_result(operation, result);
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .specializations = specializations,
      .specialization_count = specialization_count,
  };
  const loomc_compile_options_t options = {
      .next = &target_options,
      .config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  if (iree_status_is_ok(status)) {
    result = NULL;
    operation =
        loomc_compile_module(compiler->compiler, compiler->workspace,
                             compiler->pipeline, module, &options, ca, &result);
    status = jit_result(operation, result);
  }
  if (iree_status_is_ok(status)) {
    const loomc_emit_options_t emit_options = {
        .artifact_format = loomc_string_view_from_iree(format),
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    result = NULL;
    operation = loomc_emit_module(compiler->environment, compiler->workspace,
                                  module, &emit_options, ca, &result);
    // Retain result until the primary artifact bytes have been copied.
    loomc_result_retain(result);
    status = jit_result(operation, result);
    if (iree_status_is_ok(status)) {
      const loomc_artifact_t* artifact = loomc_result_artifact_at(result, 0);
      loomc_byte_span_t image = loomc_byte_span_empty();
      status = iree_status_from_loomc(
          loomc_byte_sequence_clone(artifact->contents, ca, &image));
      *out_image = iree_const_byte_span_from_loomc(image);
    }
    loomc_result_release(result);
  }
  loomc_module_release(module);
  return status;
}

typedef struct jit_request_t {
  // Next published request, owned by the construction call.
  struct jit_request_t* next;
  // Immutable request transferred by the product builder.
  loomc_request_t* request;
} jit_request_t;

typedef struct jit_requests_t {
  // Allocation policy for request-list nodes.
  iree_allocator_t allocator;
  // Owned requests pending native compilation.
  jit_request_t* head;
} jit_requests_t;

static loomc_status_t jit_publish(void* user_data, loomc_request_t* request) {
  jit_requests_t* requests = user_data;
  jit_request_t* node = NULL;
  iree_status_t status =
      iree_allocator_malloc(requests->allocator, sizeof(*node), (void**)&node);
  if (iree_status_is_ok(status)) {
    node->request = request;
    node->next = requests->head;
    requests->head = node;
  } else {
    loomc_request_release(request);
  }
  return loomc_status_from_iree(status);
}

static iree_status_t jit_load_request(loom_serve_jit_t* jit,
                                      loomc_product_t* product,
                                      loomc_request_t* request,
                                      loom_serve_command_entry_t* entries) {
  const iree_host_size_t count = loomc_request_binding_count(request);
  const iree_host_size_t root_count = loomc_request_root_count(request);
  loomc_target_specialization_t* specializations = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(jit->allocator, root_count,
                                                   sizeof(*specializations),
                                                   (void**)&specializations));
  for (iree_host_size_t i = 0; i < count; ++i) {
    loomc_request_binding_t binding;
    loomc_request_binding_at(request, i, &binding);
    loomc_cmd_entry_requirement_t requirement;
    loomc_cmd_program_product_entry_requirement_at(
        product, binding.requirement_ordinal, &requirement);
    specializations[binding.root_ordinal] = (loomc_target_specialization_t){
        .function_symbol = requirement.symbol,
        .target_profile = jit->kernel.profile,
    };
  }
  iree_const_byte_span_t image;
  iree_status_t status = jit_emit(
      &jit->kernel, loomc_request_source(request), root_count, specializations,
      IREE_SV(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO), jit->allocator, &image);
  iree_allocator_free(jit->allocator, specializations);
  iree_hal_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data = image;
    status = iree_hal_executable_load(iree_hal_queue_family(jit->dispatch),
                                      jit->target, &params, &executable);
  }
  iree_allocator_free(jit->allocator, (void*)image.data);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    loomc_request_binding_t binding;
    loomc_request_binding_at(request, i, &binding);
    loomc_cmd_entry_requirement_t requirement;
    loomc_cmd_program_product_entry_requirement_at(
        product, binding.requirement_ordinal, &requirement);
    loom_serve_command_entry_t* entry = &entries[binding.requirement_ordinal];
    status = iree_hal_executable_lookup_function_by_name(
        executable, iree_string_view_from_loomc(requirement.symbol),
        &entry->function);
    if (iree_status_is_ok(status)) {
      entry->executable = executable;
      iree_hal_executable_retain(executable);
    }
  }
  iree_hal_executable_release(executable);
  return status;
}

void loom_serve_jit_stage_destroy(loom_serve_jit_stage_t* stage) {
  if (!stage) {
    return;
  }
  for (iree_host_size_t i = 0; i < stage->entry_count; ++i) {
    iree_hal_executable_release(stage->entries[i].executable);
  }
  iree_allocator_free(stage->allocator, stage->entries);
  iree_allocator_free(stage->allocator, (void*)stage->image.data);
  iree_allocator_free(stage->allocator, stage);
}

iree_status_t loom_serve_jit_compile(loom_serve_jit_t* jit,
                                     iree_string_view_t root,
                                     const loomc_config_options_t* config,
                                     loom_serve_jit_stage_t** out_stage) {
  *out_stage = NULL;
  const iree_time_t start = iree_time_now();
  loomc_link_index_symbol_t symbol;
  if (!loomc_link_index_lookup_global(
          jit->index, loomc_string_view_from_iree(root), &symbol)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "command root @%.*s",
                            (int)root.size, root.data);
  }
  loom_serve_jit_stage_t* stage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(jit->allocator, sizeof(*stage), (void**)&stage));
  stage->allocator = jit->allocator;
  jit_requests_t requests = {.allocator = jit->allocator};
  const loomc_cmd_program_product_options_t options = {
      .link_index = jit->index,
      .root_symbol_ordinals = &symbol.ordinal,
      .root_symbol_count = 1,
      .config = *config,
      .request_sink = {.publish = jit_publish, .user_data = &requests},
  };
  loomc_product_t* product = NULL;
  loomc_result_t* result = NULL;
  const loomc_status_t operation = loomc_cmd_program_product_build(
      jit->kernel.workspace, &options,
      loomc_allocator_from_iree(jit->allocator), &product, &result);
  iree_status_t status = jit_result(operation, result);
  loomc_cmd_program_t program = {0};
  loom_serve_command_entry_t* requirements = NULL;
  iree_host_size_t requirement_count = 0;
  if (iree_status_is_ok(status)) {
    loomc_cmd_program_product_program_at(product, 0, &program);
    requirement_count = loomc_product_requirement_count(product);
    status = iree_allocator_malloc_array(jit->allocator, requirement_count,
                                         sizeof(*requirements),
                                         (void**)&requirements);
  }
  iree_host_size_t request_count = 0;
  for (jit_request_t* node = requests.head; node && iree_status_is_ok(status);
       node = node->next) {
    status = jit_load_request(jit, product, node->request, requirements);
    ++request_count;
  }
  if (iree_status_is_ok(status)) {
    loomc_byte_span_t image;
    status = iree_status_from_loomc(loomc_byte_sequence_clone(
        program.artifact.contents, loomc_allocator_from_iree(jit->allocator),
        &image));
    if (iree_status_is_ok(status)) {
      stage->image = iree_const_byte_span_from_loomc(image);
      status = loom_cmd_program_parse(stage->image, &stage->program);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        jit->allocator, program.entry_requirement_count,
        sizeof(*stage->entries), (void**)&stage->entries);
  }
  if (iree_status_is_ok(status)) {
    stage->entry_count = program.entry_requirement_count;
    for (iree_host_size_t i = 0; i < stage->entry_count; ++i) {
      stage->entries[i] = requirements[program.entry_requirement_ordinals[i]];
      if (!stage->entries[i].executable) {
        status =
            iree_make_status(IREE_STATUS_NOT_FOUND,
                             "command @%.*s entry %zu has no source provider",
                             (int)root.size, root.data, i);
        break;
      }
      iree_hal_executable_retain(stage->entries[i].executable);
    }
  }
  while (requests.head) {
    jit_request_t* node = requests.head;
    requests.head = node->next;
    loomc_request_release(node->request);
    iree_allocator_free(jit->allocator, node);
  }
  if (requirements) {
    for (iree_host_size_t i = 0; i < requirement_count; ++i) {
      iree_hal_executable_release(requirements[i].executable);
    }
  }
  iree_allocator_free(jit->allocator, requirements);
  loomc_product_release(product);
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "{\"event\":\"jit_stage\",\"root\":\"%.*s\",\"kernels\":%zu,"
            "\"entries\":%zu,\"duration_ms\":%.3f}\n",
            (int)root.size, root.data, request_count, stage->entry_count,
            (double)(iree_time_now() - start) / 1000000.0);
    *out_stage = stage;
  } else {
    loom_serve_jit_stage_destroy(stage);
  }
  return status;
}

const loom_cmd_program_t* loom_serve_jit_stage_program(
    const loom_serve_jit_stage_t* stage) {
  return &stage->program;
}

iree_status_t loom_serve_jit_stage_record(
    const loom_serve_jit_stage_t* stage,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_buffer_t* const* fixed_buffers,
    iree_hal_command_buffer_t** out_command) {
  return loom_serve_command_create(
      queue_family, mode, &stage->program,
      stage->program.requirements.fixed_buffer_count, fixed_buffers,
      stage->entry_count, stage->entries, stage->allocator, out_command);
}

iree_status_t loom_serve_jit_compile_vm(iree_string_view_t source_path,
                                        iree_string_view_t root,
                                        iree_allocator_t host_allocator,
                                        iree_const_byte_span_t* out_image) {
  *out_image = iree_const_byte_span_empty();
  jit_compiler_t compiler = {0};
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  iree_status_t status = iree_status_from_loomc(
      loomc_target_environment_create_vm(ca, &compiler.environment));
  if (iree_status_is_ok(status)) {
    status = jit_compiler_initialize(NULL, host_allocator, &compiler);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        compiler.environment, loomc_make_cstring_view("vm:core"), ca,
        &compiler.profile));
  }
  loomc_source_t* source = NULL;
  if (iree_status_is_ok(status)) {
    status = jit_read_source(source_path, host_allocator, &source);
  }
  if (iree_status_is_ok(status)) {
    const loomc_target_specialization_t specialization = {
        .function_symbol = loomc_string_view_from_iree(root),
        .target_profile = compiler.profile,
    };
    status =
        jit_emit(&compiler, source, 1, &specialization,
                 IREE_SV(LOOMC_ARTIFACT_FORMAT_VM), host_allocator, out_image);
  }
  loomc_source_release(source);
  jit_compiler_deinitialize(&compiler);
  return status;
}
