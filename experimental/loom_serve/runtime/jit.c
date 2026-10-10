// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/jit.h"

#include "iree/base/internal/path.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/io/file_contents.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu/base.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd/program.h"
#include "loomc/target/vm.h"
#include "loomc/task_pool.h"
#include "loomc/task_queue.h"

typedef struct jit_compiler_t {
  // Target package retained by the context and prepared compiler.
  loomc_target_environment_t* environment;
  // Shared immutable parser and dialect configuration.
  loomc_context_t* context;
  // Caller scratch for command products or isolated VM compilation.
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
  // Borrowed evidence accumulator, updated only at joined owner boundaries.
  loom_serve_compilation_report_t* report;
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
  // Concurrent native compilation, independent of serving execution.
  struct {
    // Catalog work queue retaining the application's shared worker executor.
    loomc_task_queue_t* queue;
    // Mutable scratch indexed by the pool's mutually exclusive worker ordinal.
    loomc_workspace_t** workspaces;
    // Number of initialized workspace entries, including during setup failure.
    iree_host_size_t workspace_count;
  } compilation;
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
    loom_serve_compilation_report_t* report, jit_compiler_t* compiler) {
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
  return loom_serve_compilation_result(status, result, report);
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
    status = loom_serve_compilation_result(operation, result, jit->report);
  }
  loomc_link_index_builder_release(builder);
  iree_io_file_contents_free(catalog);
  return status;
}

static iree_status_t jit_initialize_compilation(loom_serve_jit_t* jit,
                                                loomc_task_pool_t* pool) {
  const loomc_allocator_t allocator = loomc_allocator_from_iree(jit->allocator);
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_task_queue_allocate(pool, allocator, &jit->compilation.queue)));
  const iree_host_size_t worker_count = loomc_task_pool_worker_count(pool);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      jit->allocator, worker_count, sizeof(*jit->compilation.workspaces),
      (void**)&jit->compilation.workspaces));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < worker_count && iree_status_is_ok(status);
       ++i) {
    status = iree_status_from_loomc(loomc_workspace_create(
        NULL, allocator, &jit->compilation.workspaces[i]));
    if (iree_status_is_ok(status)) {
      ++jit->compilation.workspace_count;
    }
  }
  return status;
}

void loom_serve_jit_destroy(loom_serve_jit_t* jit) {
  if (!jit) {
    return;
  }
  loomc_task_queue_free(jit->compilation.queue);
  for (iree_host_size_t i = 0; i < jit->compilation.workspace_count; ++i) {
    loomc_workspace_release(jit->compilation.workspaces[i]);
  }
  iree_allocator_free(jit->allocator, jit->compilation.workspaces);
  loomc_link_index_release(jit->index);
  jit_compiler_deinitialize(&jit->kernel);
  iree_hal_queue_release(jit->dispatch);
  iree_hal_device_release(jit->device);
  iree_allocator_free(jit->allocator, jit);
}

iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    loomc_task_pool_t* task_pool,
                                    iree_string_view_t source_directory,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    loom_serve_compilation_report_t* report,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit) {
  *out_jit = NULL;
  const iree_hal_physical_device_affinity_t affinity =
      iree_hal_queue_family_spec(iree_hal_queue_family(dispatch))
          ->physical_device_affinity;
  loom_serve_jit_t* jit = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*jit), (void**)&jit));
  jit->allocator = host_allocator;
  jit->report = report;
  jit->device = device;
  iree_hal_device_retain(device);
  jit->dispatch = dispatch;
  iree_hal_queue_retain(dispatch);
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  iree_status_t status = iree_status_from_loomc(
      loomc_target_environment_create_amdgpu(ca, &jit->kernel.environment));
  if (iree_status_is_ok(status)) {
    status = jit_compiler_initialize(sanitizer, host_allocator, report,
                                     &jit->kernel);
  }
  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_iree_hal_target_options_t options = {
        .device = device, .physical_device_affinity = affinity};
    loomc_iree_hal_target_selection_t selection = {0};
    loomc_result_t* result = NULL;
    const loomc_status_t operation = loomc_target_select_amdgpu_iree_hal(
        jit->kernel.environment, &options, ca, &selection, &result);
    jit->kernel.profile = selection.target_profile;
    jit->target = selection.executable_target;
    status = loom_serve_compilation_result(operation, result, report);
  }
  if (iree_status_is_ok(status)) {
    status = jit_index_sources(jit, source_directory);
  }
  if (iree_status_is_ok(status)) {
    status = jit_initialize_compilation(jit, task_pool);
  }
  if (iree_status_is_ok(status)) {
    *out_jit = jit;
  } else {
    loom_serve_jit_destroy(jit);
  }
  return status;
}

loom_serve_compilation_report_t* loom_serve_jit_report(loom_serve_jit_t* jit) {
  return jit->report;
}

// Keeps prepared-low function facts in the same module through native emission.
static iree_status_t jit_emit(
    const jit_compiler_t* compiler, loomc_workspace_t* workspace,
    loomc_source_t* source, iree_host_size_t specialization_count,
    const loomc_target_specialization_t* specializations,
    iree_string_view_t format, iree_allocator_t allocator,
    loom_serve_compilation_report_t* report,
    iree_const_byte_span_t* out_image) {
  *out_image = iree_const_byte_span_empty();
  const loomc_allocator_t ca = loomc_allocator_from_iree(allocator);
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  loomc_status_t operation = loomc_module_deserialize_from_source(
      compiler->context, workspace, source, NULL, ca, &module, &result);
  iree_status_t status =
      loom_serve_compilation_result(operation, result, report);
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
        loomc_compile_module(compiler->compiler, workspace, compiler->pipeline,
                             module, &options, ca, &result);
    status = loom_serve_compilation_result(operation, result, report);
  }
  if (iree_status_is_ok(status)) {
    const loomc_emit_options_t emit_options = {
        .artifact_format = loomc_string_view_from_iree(format),
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    result = NULL;
    operation = loomc_emit_module(compiler->environment, workspace, module,
                                  &emit_options, ca, &result);
    // Retain result until the primary artifact bytes have been copied.
    loomc_result_retain(result);
    status = loom_serve_compilation_result(operation, result, report);
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
  // Task outcome consumed by the caller after the stage queue drains.
  iree_status_t status;
  // Disjoint bounded evidence, merged only after all native tasks retire.
  loom_serve_compilation_report_t report;
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
                                      loomc_workspace_t* workspace,
                                      loomc_product_t* product,
                                      loomc_request_t* request,
                                      loom_serve_compilation_report_t* report,
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
      &jit->kernel, workspace, loomc_request_source(request), root_count,
      specializations, IREE_SV(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      jit->allocator, report, &image);
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

// A catalog is a library universe, so a named entry must have exactly one
// exported definition. The index's name lookup enumerates declarations too;
// it does not perform definition selection for exact-ordinal product requests.
static iree_status_t jit_lookup_root(loom_serve_jit_t* jit,
                                     iree_string_view_t root,
                                     loomc_host_size_t* out_ordinal) {
  loomc_link_index_symbol_t symbol;
  bool found = false;
  bool has_symbol = loomc_link_index_lookup_global(
      jit->index, loomc_string_view_from_iree(root), &symbol);
  while (has_symbol) {
    if (iree_all_bits_set(symbol.flags,
                          LOOMC_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION |
                              LOOMC_LINK_SYMBOL_FLAG_EXPORT)) {
      if (found) {
        return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "command root @%.*s has multiple exported "
                                "definitions",
                                (int)root.size, root.data);
      }
      *out_ordinal = symbol.ordinal;
      found = true;
    }
    has_symbol =
        loomc_link_index_next_global_duplicate(jit->index, &symbol, &symbol);
  }
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "command root @%.*s has no exported definition",
                            (int)root.size, root.data);
  }
  return iree_ok_status();
}

typedef struct jit_job_t {
  // Initial product-construction task, sharing storage with the joined job.
  loomc_task_t base;
  // Initial task plus accepted native requests. Last release binds the stage.
  iree_atomic_ref_count_t ref_count;
  // Catalog and executor queue borrowed through completion.
  loom_serve_jit_t* jit;
  // Command symbol borrowed through completion.
  iree_string_view_t root;
  // Specialization bindings borrowed through completion.
  const loomc_config_options_t* config;
  // Completion receiving owned result and status after all requests retire.
  loom_serve_jit_completion_fn_t completion;
  // Opaque callback state borrowed through completion.
  void* user_data;
  // Start of accepted compilation in host nanoseconds.
  iree_time_t start;
  // Product-owned native requests and per-request terminal statuses.
  jit_requests_t requests;
  // Owned immutable command product borrowed by native tasks.
  loomc_product_t* product;
  // Parsed command description borrowing product storage.
  loomc_cmd_program_t program;
  // Disjoint native result slots indexed by requirement ordinal.
  loom_serve_command_entry_t* requirements;
  // Number of allocated requirement slots.
  iree_host_size_t requirement_count;
  // Number of native requests submitted for the command.
  iree_host_size_t request_count;
  // Product construction or native submission failure.
  iree_status_t status;
} jit_job_t;

static iree_status_t jit_job_bind(jit_job_t* job,
                                  loom_serve_jit_stage_t** out_stage) {
  loom_serve_jit_t* jit = job->jit;
  loom_serve_jit_stage_t* stage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(jit->allocator, sizeof(*stage), (void**)&stage));
  stage->allocator = jit->allocator;
  loomc_byte_span_t image;
  iree_status_t status = iree_status_from_loomc(loomc_byte_sequence_clone(
      job->program.artifact.contents, loomc_allocator_from_iree(jit->allocator),
      &image));
  if (iree_status_is_ok(status)) {
    stage->image = iree_const_byte_span_from_loomc(image);
    status = loom_cmd_program_parse(stage->image, &stage->program);
  }
  if (iree_status_is_ok(status) && job->program.entry_requirement_count) {
    status = iree_allocator_malloc_array(
        jit->allocator, job->program.entry_requirement_count,
        sizeof(*stage->entries), (void**)&stage->entries);
  }
  if (iree_status_is_ok(status)) {
    stage->entry_count = job->program.entry_requirement_count;
    for (iree_host_size_t i = 0;
         i < stage->entry_count && iree_status_is_ok(status); ++i) {
      stage->entries[i] =
          job->requirements[job->program.entry_requirement_ordinals[i]];
      if (!stage->entries[i].executable) {
        status =
            iree_make_status(IREE_STATUS_NOT_FOUND,
                             "command @%.*s entry %zu has no source provider",
                             (int)job->root.size, job->root.data, i);
      } else {
        iree_hal_executable_retain(stage->entries[i].executable);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_stage = stage;
  } else {
    loom_serve_jit_stage_destroy(stage);
  }
  return status;
}

static void jit_job_release(jit_job_t* job) {
  if (iree_atomic_ref_count_dec(&job->ref_count) != 1) {
    return;
  }
  loom_serve_jit_t* jit = job->jit;
  iree_status_t status = job->status;
  while (job->requests.head) {
    jit_request_t* node = job->requests.head;
    job->requests.head = node->next;
    status = iree_status_join(status, node->status);
    loom_serve_compilation_merge(jit->report, &node->report);
    loomc_request_release(node->request);
    iree_allocator_free(jit->allocator, node);
  }
  loom_serve_jit_stage_t* stage = NULL;
  if (iree_status_is_ok(status)) {
    status = jit_job_bind(job, &stage);
  }
  if (job->requirements) {
    for (iree_host_size_t i = 0; i < job->requirement_count; ++i) {
      iree_hal_executable_release(job->requirements[i].executable);
    }
  }
  iree_allocator_free(jit->allocator, job->requirements);
  loomc_product_release(job->product);
  ++jit->report->commands;
  jit->report->kernels += job->request_count;
  jit->report->entries += stage ? stage->entry_count : 0;
  jit->report->workers =
      iree_max(jit->report->workers, jit->compilation.workspace_count);
  jit->report->duration_ns += iree_time_now() - job->start;
  const loom_serve_jit_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(jit->allocator, job);
  completion(user_data, status, stage);
}

typedef struct jit_task_t {
  // Queue-owned native work record.
  loomc_task_t base;
  // Retained compilation keeping the product and result slots alive.
  jit_job_t* job;
  // Disjoint request and terminal result slot owned by the job.
  jit_request_t* request;
} jit_task_t;

static void jit_task_execute(loomc_task_t* base,
                             loomc_host_size_t worker_ordinal) {
  jit_task_t* task = (jit_task_t*)base;
  jit_job_t* job = task->job;
  task->request->status = jit_load_request(
      job->jit, job->jit->compilation.workspaces[worker_ordinal], job->product,
      task->request->request, &task->request->report, job->requirements);
}

static void jit_task_destroy(loomc_task_t* base) {
  jit_task_t* task = (jit_task_t*)base;
  jit_job_t* job = task->job;
  iree_allocator_free(job->jit->allocator, task);
  jit_job_release(job);
}

static const loomc_task_vtable_t jit_task_vtable = {
    .execute = jit_task_execute,
    .destroy = jit_task_destroy,
};

static iree_status_t jit_submit_request(jit_job_t* job,
                                        jit_request_t* request) {
  jit_task_t* task = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(job->jit->allocator, sizeof(*task), (void**)&task));
  loomc_task_initialize(&jit_task_vtable, &task->base);
  task->job = job;
  task->request = request;
  iree_atomic_ref_count_inc(&job->ref_count);
  iree_status_t status = iree_status_from_loomc(loomc_task_sink_submit(
      loomc_task_queue_sink(job->jit->compilation.queue), &task->base));
  if (!iree_status_is_ok(status)) {
    loomc_task_destroy(&task->base);
  }
  return status;
}

static void jit_job_execute(loomc_task_t* base,
                            loomc_host_size_t worker_ordinal) {
  jit_job_t* job = (jit_job_t*)base;
  loom_serve_jit_t* jit = job->jit;
  loomc_host_size_t root_ordinal = 0;
  iree_status_t status = jit_lookup_root(jit, job->root, &root_ordinal);
  if (iree_status_is_ok(status)) {
    const loomc_cmd_program_product_options_t options = {
        .link_index = jit->index,
        .root_symbol_ordinals = &root_ordinal,
        .root_symbol_count = 1,
        .config = *job->config,
        .request_sink = {.publish = jit_publish, .user_data = &job->requests},
    };
    loomc_result_t* result = NULL;
    const loomc_status_t operation = loomc_cmd_program_product_build(
        jit->kernel.workspace, &options,
        loomc_allocator_from_iree(jit->allocator), &job->product, &result);
    status = loom_serve_compilation_result(operation, result, jit->report);
  }
  if (iree_status_is_ok(status)) {
    loomc_cmd_program_product_program_at(job->product, 0, &job->program);
    job->requirement_count = loomc_product_requirement_count(job->product);
    if (job->requirement_count) {
      status = iree_allocator_malloc_array(
          jit->allocator, job->requirement_count, sizeof(*job->requirements),
          (void**)&job->requirements);
    }
  }
  for (jit_request_t* node = job->requests.head;
       node && iree_status_is_ok(status); node = node->next) {
    status = jit_submit_request(job, node);
    if (iree_status_is_ok(status)) {
      ++job->request_count;
    }
  }
  // The initial task keeps a reference while expanding the native work. Its
  // final release joins the children without occupying a waiting worker.
  job->status = status;
}

static void jit_job_destroy(loomc_task_t* base) {
  jit_job_release((jit_job_t*)base);
}

static const loomc_task_vtable_t jit_job_vtable = {
    .execute = jit_job_execute,
    .destroy = jit_job_destroy,
};

iree_status_t loom_serve_jit_compile_async(
    loom_serve_jit_t* jit, iree_string_view_t root,
    const loomc_config_options_t* config,
    loom_serve_jit_completion_fn_t completion, void* user_data) {
  jit_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(jit->allocator, sizeof(*job), (void**)&job));
  loomc_task_initialize(&jit_job_vtable, &job->base);
  iree_atomic_ref_count_init(&job->ref_count);
  job->jit = jit;
  job->root = root;
  job->config = config;
  job->completion = completion;
  job->user_data = user_data;
  job->start = iree_time_now();
  job->requests.allocator = jit->allocator;
  iree_status_t status = iree_status_from_loomc(loomc_task_sink_submit(
      loomc_task_queue_sink(jit->compilation.queue), &job->base));
  if (!iree_status_is_ok(status)) {
    // Rejection retained ownership and never executed product construction.
    // No completion is owed and no child or product storage exists.
    iree_allocator_free(jit->allocator, job);
  }
  return status;
}

typedef struct jit_join_t {
  // Outer application's wait; never used by a task worker.
  iree_notification_t notification;
  // Joins publication with the callback's final notification access.
  iree_slim_mutex_t mutex;
  // Result/status are ready after the callback has posted its notification.
  bool completed;
  // Owned compilation outcome transferred by the callback.
  iree_status_t status;
  // Owned compiled stage, or NULL on failure.
  loom_serve_jit_stage_t* stage;
} jit_join_t;

static void jit_join_complete(void* user_data, iree_status_t status,
                              loom_serve_jit_stage_t* stage) {
  jit_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  join->status = status;
  join->stage = stage;
  join->completed = true;
  iree_notification_post(&join->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&join->mutex);
}

static bool jit_join_is_complete(void* user_data) {
  jit_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  const bool completed = join->completed;
  iree_slim_mutex_unlock(&join->mutex);
  return completed;
}

iree_status_t loom_serve_jit_compile(loom_serve_jit_t* jit,
                                     iree_string_view_t root,
                                     const loomc_config_options_t* config,
                                     loom_serve_jit_stage_t** out_stage) {
  *out_stage = NULL;
  jit_join_t join = {0};
  iree_notification_initialize(&join.notification);
  iree_slim_mutex_initialize(&join.mutex);
  iree_status_t status =
      loom_serve_jit_compile_async(jit, root, config, jit_join_complete, &join);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&join.notification, jit_join_is_complete, &join,
                            iree_infinite_timeout());
    status = join.status;
    *out_stage = join.stage;
  }
  iree_notification_deinitialize(&join.notification);
  iree_slim_mutex_deinitialize(&join.mutex);
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
                                        iree_host_size_t root_count,
                                        const iree_string_view_t* roots,
                                        loom_serve_compilation_report_t* report,
                                        iree_allocator_t host_allocator,
                                        iree_const_byte_span_t* out_image) {
  *out_image = iree_const_byte_span_empty();
  if (!root_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM program requires at least one source root");
  }
  jit_compiler_t compiler = {0};
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  iree_status_t status = iree_status_from_loomc(
      loomc_target_environment_create_vm(ca, &compiler.environment));
  if (iree_status_is_ok(status)) {
    status = jit_compiler_initialize(NULL, host_allocator, report, &compiler);
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
  loomc_target_specialization_t* specializations = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, root_count,
                                         sizeof(*specializations),
                                         (void**)&specializations);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < root_count; ++i) {
      specializations[i] = (loomc_target_specialization_t){
          .function_symbol = loomc_string_view_from_iree(roots[i]),
          .target_profile = compiler.profile,
      };
    }
    status = jit_emit(&compiler, compiler.workspace, source, root_count,
                      specializations, IREE_SV(LOOMC_ARTIFACT_FORMAT_VM),
                      host_allocator, report, out_image);
  }
  iree_allocator_free(host_allocator, specializations);
  loomc_source_release(source);
  jit_compiler_deinitialize(&compiler);
  return status;
}
