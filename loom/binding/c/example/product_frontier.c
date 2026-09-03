// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"
#include "loomc/target/cmd.h"
#include "loomc/task.h"
#include "loomc/task_pool.h"
#include "loomc/task_queue.h"

static const char kSourceText[] =
    "kernel.def @load() {\n"
    "  %one = index.constant 1 : index\n"
    "  kernel.launch.config workgroups(%one, %one, %one) "
    "workgroup_size(%one, %one, %one) : index\n"
    "} launch(%storage: buffer) {\n"
    "  kernel.return\n"
    "}\n"
    "\n"
    "kernel.def @transform() {\n"
    "  %one = index.constant 1 : index\n"
    "  kernel.launch.config workgroups(%one, %one, %one) "
    "workgroup_size(%one, %one, %one) : index\n"
    "} launch(%storage: buffer) {\n"
    "  kernel.return\n"
    "}\n"
    "\n"
    "kernel.def @store() {\n"
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
    "  kernel.launch @load(%storage) : (buffer)\n"
    "  kernel.launch @transform(%storage) : (buffer)\n"
    "  kernel.launch @store(%storage) : (buffer)\n"
    "  kernel.dispatch @external[%one](%storage) : [index](buffer)\n"
    "  command.return\n"
    "}\n";

enum {
  KERNEL_COMPILE_CAPACITY = 3,
  COMMAND_REQUIREMENT_CAPACITY = 4,
};

typedef struct kernel_compile_output_t {
  // Immutable source request transferred by command construction.
  loomc_request_t* request;

  // Independently compiled child product.
  loomc_product_t* product;

  // Compiler diagnostics retained through batch inspection.
  loomc_result_t* result;

  // Infrastructure status produced by the compile task.
  loomc_status_t status;
} kernel_compile_output_t;

typedef struct jit_scheduler_t {
  // Allocator shared by process, task, and product storage.
  loomc_allocator_t allocator;

  // Shared context for indexing and compilation.
  loomc_context_t* context;

  // Mutable scratch used only by command-product construction.
  loomc_workspace_t* command_workspace;

  // Immutable source containing the command and kernel catalog.
  loomc_source_t* source;

  // Frozen source index reused by command-product builds.
  loomc_link_index_t* link_index;

  // Prepared compiler shared by kernel compile tasks.
  loomc_compiler_t* compiler;

  // Prepared pipeline shared by kernel compile tasks.
  loomc_pass_program_t* pass_program;

  // Optional standard worker population used by this example.
  loomc_task_pool_t* task_pool;

  // Compilation scheduling domain attached to the standard pool.
  loomc_task_queue_t* compile_queue;

  // Scheduler-neutral destination used by the publication callback.
  loomc_task_sink_t compile_sink;

  // Mutable compiler scratch indexed by dense worker ordinal.
  loomc_workspace_t** worker_workspaces;

  // Number of initialized entries in `worker_workspaces`.
  loomc_host_size_t worker_workspace_count;

  // Parent command product retained until child bindings are inspected.
  loomc_product_t* command_product;

  // Accepted kernel compilations in publication order.
  kernel_compile_output_t kernel_compiles[KERNEL_COMPILE_CAPACITY];

  // Number of initialized entries in `kernel_compiles`.
  loomc_host_size_t kernel_compile_count;
} jit_scheduler_t;

typedef struct kernel_compile_task_t {
  // Generic task base transferred to the accepting scheduler.
  loomc_task_t base;

  // Shared prepared state and worker-local workspace table.
  jit_scheduler_t* scheduler;

  // Caller-owned output that outlives task execution.
  kernel_compile_output_t* output;
} kernel_compile_task_t;

static void print_status(loomc_status_t status) {
  char buffer[1024] = {0};
  loomc_host_size_t length = 0;
  loomc_status_format(status, sizeof(buffer), buffer, &length);
  fprintf(stderr, "%.*s\n", (int)length, buffer);
}

static void print_result_diagnostics(const loomc_result_t* result) {
  if (result == NULL) return;
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
  print_result_diagnostics(result);
  return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, message);
}

static loomc_status_t require_condition(bool condition, const char* message) {
  return condition
             ? loomc_ok_status()
             : loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, message);
}

static void jit_scheduler_initialize(jit_scheduler_t* scheduler) {
  memset(scheduler, 0, sizeof(*scheduler));
  scheduler->allocator = loomc_allocator_system();
}

static void jit_scheduler_deinitialize(jit_scheduler_t* scheduler) {
  // Drain tasks before releasing their output slots or worker workspaces.
  loomc_task_queue_free(scheduler->compile_queue);
  for (loomc_host_size_t i = 0; i < scheduler->kernel_compile_count; ++i) {
    kernel_compile_output_t* output = &scheduler->kernel_compiles[i];
    loomc_status_free(output->status);
    loomc_result_release(output->result);
    loomc_product_release(output->product);
    loomc_request_release(output->request);
  }
  for (loomc_host_size_t i = 0; i < scheduler->worker_workspace_count; ++i) {
    loomc_workspace_release(scheduler->worker_workspaces[i]);
  }
  loomc_allocator_free(scheduler->allocator, scheduler->worker_workspaces);
  loomc_task_pool_free(scheduler->task_pool);
  loomc_product_release(scheduler->command_product);
  loomc_pass_program_release(scheduler->pass_program);
  loomc_compiler_release(scheduler->compiler);
  loomc_link_index_release(scheduler->link_index);
  loomc_source_release(scheduler->source);
  loomc_workspace_release(scheduler->command_workspace);
  loomc_context_release(scheduler->context);
}

// --8<-- [start:compile-task]
static void execute_kernel_compile_task(loomc_task_t* base_task,
                                        loomc_host_size_t worker_ordinal) {
  kernel_compile_task_t* task = (kernel_compile_task_t*)base_task;
  jit_scheduler_t* scheduler = task->scheduler;
  kernel_compile_output_t* output = task->output;
  if (worker_ordinal >= scheduler->worker_workspace_count) {
    output->status = loomc_make_status(
        LOOMC_STATUS_OUT_OF_RANGE,
        "scheduler supplied an invalid compiler worker ordinal");
    return;
  }

  loomc_workspace_t* workspace = scheduler->worker_workspaces[worker_ordinal];
  const loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .module_name = loomc_make_cstring_view("scheduled-kernel"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
  };
  output->status = loomc_compile_request(
      scheduler->compiler, workspace, scheduler->pass_program, output->request,
      &options, scheduler->allocator, &output->product, &output->result);
  loomc_workspace_trim(workspace);
}

static void destroy_kernel_compile_task(loomc_task_t* base_task) {
  kernel_compile_task_t* task = (kernel_compile_task_t*)base_task;
  loomc_allocator_free(task->scheduler->allocator, task);
}

static const loomc_task_vtable_t kKernelCompileTaskVtable = {
    .execute = execute_kernel_compile_task,
    .destroy = destroy_kernel_compile_task,
};
// --8<-- [end:compile-task]

static loomc_status_t allocate_kernel_compile_task(
    jit_scheduler_t* scheduler, kernel_compile_output_t* output,
    loomc_task_t** out_task) {
  *out_task = NULL;
  kernel_compile_task_t* task = NULL;
  loomc_status_t status = loomc_allocator_malloc(scheduler->allocator,
                                                 sizeof(*task), (void**)&task);
  if (loomc_status_is_ok(status)) {
    loomc_task_initialize(&kKernelCompileTaskVtable, &task->base);
    task->scheduler = scheduler;
    task->output = output;
    *out_task = &task->base;
  }
  return status;
}

// --8<-- [start:schedule-request]
static loomc_status_t schedule_kernel_request(void* user_data,
                                              loomc_request_t* request) {
  jit_scheduler_t* scheduler = (jit_scheduler_t*)user_data;
  if (loomc_request_product_descriptor(request) !=
      loomc_compiled_module_product_descriptor()) {
    loomc_request_release(request);
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "no compiler route for published request");
  }
  if (scheduler->kernel_compile_count == KERNEL_COMPILE_CAPACITY) {
    loomc_request_release(request);
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "example kernel compile capacity exceeded");
  }

  kernel_compile_output_t* output =
      &scheduler->kernel_compiles[scheduler->kernel_compile_count];
  output->request = request;

  loomc_task_t* task = NULL;
  loomc_status_t status =
      allocate_kernel_compile_task(scheduler, output, &task);
  if (loomc_status_is_ok(status)) {
    status = loomc_task_sink_submit(scheduler->compile_sink, task);
  }
  if (!loomc_status_is_ok(status)) {
    if (task != NULL) loomc_task_destroy(task);
    loomc_request_release(output->request);
    memset(output, 0, sizeof(*output));
    return status;
  }

  // The sink owns `task` after successful submission. An inline scheduler may
  // already have executed and destroyed it, so only the output slot is used.
  ++scheduler->kernel_compile_count;
  return loomc_ok_status();
}
// --8<-- [end:schedule-request]

static loomc_status_t prepare_source_catalog(jit_scheduler_t* scheduler,
                                             loomc_host_size_t* out_root) {
  *out_root = 0;
  loomc_status_t status =
      loomc_context_create(NULL, scheduler->allocator, &scheduler->context);
  if (loomc_status_is_ok(status)) {
    status = loomc_workspace_create(NULL, scheduler->allocator,
                                    &scheduler->command_workspace);
  }

  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("scheduled_jit.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create(&source_options, scheduler->allocator,
                                 &scheduler->source);
  }

  loomc_link_index_builder_t* builder = NULL;
  loomc_result_t* index_result = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_create(scheduler->context, NULL,
                                             scheduler->allocator, &builder);
  }
  const loomc_link_index_source_options_t provider_options = {
      .provider_name = loomc_make_cstring_view("scheduled-jit"),
      .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_add_source(builder, scheduler->source,
                                                 &provider_options, NULL);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_finish(builder, &scheduler->link_index,
                                             &index_result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(index_result, "source indexing failed");
  }

  loomc_link_index_symbol_t root_symbol = {0};
  if (loomc_status_is_ok(status) &&
      !loomc_link_index_lookup_global(scheduler->link_index,
                                      loomc_make_cstring_view("run"),
                                      &root_symbol)) {
    status = loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                               "command root @run was not indexed");
  }
  if (loomc_status_is_ok(status)) *out_root = root_symbol.ordinal;

  loomc_result_release(index_result);
  loomc_link_index_builder_release(builder);
  return status;
}

// --8<-- [start:prepare-scheduler]
static loomc_status_t prepare_compile_scheduler(jit_scheduler_t* scheduler) {
  loomc_status_t status = loomc_compiler_create(
      scheduler->context, NULL, scheduler->allocator, &scheduler->compiler);

  loomc_result_t* pass_result = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_from_pipeline_text(
        scheduler->context, loomc_make_cstring_view("canonicalize,cse"), NULL,
        scheduler->allocator, &scheduler->pass_program, &pass_result);
  }
  if (loomc_status_is_ok(status)) {
    status =
        require_successful_result(pass_result, "pipeline preparation failed");
  }
  loomc_result_release(pass_result);

  const loomc_task_pool_options_t pool_options = {
      .type = LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS,
      .structure_size = sizeof(pool_options),
      .max_worker_count = 4,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_task_pool_allocate(&pool_options, scheduler->allocator,
                                      &scheduler->task_pool);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_task_queue_allocate(
        scheduler->task_pool, scheduler->allocator, &scheduler->compile_queue);
  }
  if (loomc_status_is_ok(status)) {
    scheduler->compile_sink = loomc_task_queue_sink(scheduler->compile_queue);
    scheduler->worker_workspace_count =
        loomc_task_pool_worker_count(scheduler->task_pool);
    status = require_condition(scheduler->worker_workspace_count != 0,
                               "task pool created no workers");
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_allocator_malloc(scheduler->allocator,
                                    scheduler->worker_workspace_count *
                                        sizeof(*scheduler->worker_workspaces),
                                    (void**)&scheduler->worker_workspaces);
  }
  loomc_host_size_t initialized_workspace_count = 0;
  while (loomc_status_is_ok(status) &&
         initialized_workspace_count < scheduler->worker_workspace_count) {
    status = loomc_workspace_create(
        NULL, scheduler->allocator,
        &scheduler->worker_workspaces[initialized_workspace_count]);
    if (loomc_status_is_ok(status)) ++initialized_workspace_count;
  }
  scheduler->worker_workspace_count = initialized_workspace_count;
  return status;
}
// --8<-- [end:prepare-scheduler]

// --8<-- [start:build-command]
static loomc_status_t build_command_product(jit_scheduler_t* scheduler,
                                            loomc_host_size_t root_ordinal) {
  const loomc_cmd_program_product_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      .structure_size = sizeof(options),
      .link_index = scheduler->link_index,
      .root_symbol_ordinals = &root_ordinal,
      .root_symbol_count = 1,
      .request_sink =
          {
              .publish = schedule_kernel_request,
              .user_data = scheduler,
          },
  };
  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_cmd_program_product_build(
      scheduler->command_workspace, &options, scheduler->allocator,
      &scheduler->command_product, &result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(result, "command construction failed");
  }
  if (loomc_status_is_ok(status)) {
    status = require_condition(
        loomc_cmd_program_product_program_count(scheduler->command_product) ==
                1 &&
            loomc_product_requirement_count(scheduler->command_product) ==
                COMMAND_REQUIREMENT_CAPACITY &&
            scheduler->kernel_compile_count == KERNEL_COMPILE_CAPACITY,
        "command construction published an unexpected kernel set");
  }
  loomc_result_release(result);
  return status;
}
// --8<-- [end:build-command]

static loomc_status_t drain_compile_queue(jit_scheduler_t* scheduler) {
  if (scheduler->compile_queue == NULL) return loomc_ok_status();
  loomc_status_t status = loomc_task_queue_shutdown(scheduler->compile_queue);
  if (loomc_status_is_ok(status)) {
    status = loomc_task_queue_await_shutdown(scheduler->compile_queue);
  }
  if (!loomc_status_is_ok(status)) {
    // The void teardown operation still drains accepted tasks, making output
    // storage safe to inspect before preserving the explicit shutdown error.
    loomc_task_queue_free(scheduler->compile_queue);
    scheduler->compile_queue = NULL;
  }
  scheduler->compile_sink = (loomc_task_sink_t){0};
  return status;
}

// --8<-- [start:bind-results]
static loomc_status_t inspect_kernel_products(jit_scheduler_t* scheduler,
                                              bool parent_succeeded) {
  loomc_status_t status = loomc_ok_status();
  bool resolved_requirements[COMMAND_REQUIREMENT_CAPACITY] = {false};
  loomc_host_size_t resolved_requirement_count = 0;

  for (loomc_host_size_t i = 0; i < scheduler->kernel_compile_count; ++i) {
    kernel_compile_output_t* output = &scheduler->kernel_compiles[i];
    const bool compile_call_succeeded = loomc_status_is_ok(output->status);
    status = loomc_status_join(status, output->status);
    output->status = loomc_ok_status();
    if (!compile_call_succeeded) continue;

    if (output->result == NULL || !loomc_result_succeeded(output->result)) {
      print_result_diagnostics(output->result);
      status = loomc_status_join(
          status, loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                    "a scheduled kernel failed compilation"));
      continue;
    }

    const loomc_artifact_t* artifact =
        loomc_product_artifact_at(output->product, 0);
    const bool product_is_valid =
        loomc_product_descriptor(output->product) ==
            loomc_compiled_module_product_descriptor() &&
        loomc_product_export_count(output->product) ==
            loomc_request_root_count(output->request) &&
        loomc_product_requirement_count(output->product) == 0 &&
        loomc_product_artifact_count(output->product) == 1 &&
        artifact != NULL &&
        loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE));
    if (!product_is_valid) {
      status = loomc_status_join(
          status,
          loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                            "a scheduled kernel returned a bad product"));
      continue;
    }

    if (!parent_succeeded) continue;
    const loomc_host_size_t binding_count =
        loomc_request_binding_count(output->request);
    for (loomc_host_size_t j = 0; j < binding_count; ++j) {
      loomc_request_binding_t binding = {0};
      const bool binding_is_valid =
          loomc_request_binding_at(output->request, j, &binding) &&
          binding.requirement_ordinal < COMMAND_REQUIREMENT_CAPACITY &&
          binding.root_ordinal < loomc_product_export_count(output->product) &&
          !resolved_requirements[binding.requirement_ordinal];
      if (!binding_is_valid) {
        status = loomc_status_join(
            status,
            loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                              "a kernel request returned an invalid binding"));
        continue;
      }
      resolved_requirements[binding.requirement_ordinal] = true;
      ++resolved_requirement_count;
    }
  }

  if (!parent_succeeded || !loomc_status_is_ok(status)) return status;

  loomc_host_size_t external_requirement_count = 0;
  for (loomc_host_size_t i = 0; i < COMMAND_REQUIREMENT_CAPACITY; ++i) {
    if (resolved_requirements[i]) continue;
    loomc_cmd_entry_requirement_t requirement = {0};
    if (!loomc_cmd_program_product_entry_requirement_at(
            scheduler->command_product, i, &requirement) ||
        !loomc_string_view_equal(requirement.symbol,
                                 loomc_make_cstring_view("external"))) {
      return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                               "unexpected external kernel requirement");
    }
    ++external_requirement_count;
  }

  loomc_cmd_program_t program = {0};
  if (!loomc_cmd_program_product_program_at(scheduler->command_product, 0,
                                            &program) ||
      resolved_requirement_count != KERNEL_COMPILE_CAPACITY ||
      external_requirement_count != 1) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "kernel products do not satisfy the command");
  }
  printf("command=%.*s requirements=%zu compiled=%zu external=%zu\n",
         (int)program.symbol.size, program.symbol.data,
         (size_t)loomc_product_requirement_count(scheduler->command_product),
         (size_t)resolved_requirement_count,
         (size_t)external_requirement_count);
  return status;
}
// --8<-- [end:bind-results]

static loomc_status_t run_scheduled_jit_example(void) {
  jit_scheduler_t scheduler;
  jit_scheduler_initialize(&scheduler);

  loomc_host_size_t root_ordinal = 0;
  loomc_status_t status = prepare_source_catalog(&scheduler, &root_ordinal);
  if (loomc_status_is_ok(status)) {
    status = prepare_compile_scheduler(&scheduler);
  }
  if (loomc_status_is_ok(status)) {
    status = build_command_product(&scheduler, root_ordinal);
  }
  const bool parent_succeeded = loomc_status_is_ok(status);

  loomc_status_t drain_status = drain_compile_queue(&scheduler);
  status = loomc_status_join(status, drain_status);
  loomc_status_t product_status =
      inspect_kernel_products(&scheduler, parent_succeeded);
  status = loomc_status_join(status, product_status);

  jit_scheduler_deinitialize(&scheduler);
  return status;
}

int main(void) {
  loomc_status_t status = run_scheduled_jit_example();
  if (loomc_status_is_ok(status)) return 0;
  print_status(status);
  loomc_status_free(status);
  return 1;
}
