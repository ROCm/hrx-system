// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"
#include "loomc/task_pool.h"
#include "loomc/task_queue.h"

static const char kSourceText[] =
    "config.decl @variant : %value: index where [range(%value, 1, 9)]\n"
    "func.def public @entry() -> (index) {\n"
    "  %value = config.get @variant : index\n"
    "  func.return %value : index\n"
    "}\n";

enum { JIT_VARIANT_COUNT = 8 };

static const char* const kVariantConfigTexts[JIT_VARIANT_COUNT] = {
    "config.def @variant = 1 : index\n", "config.def @variant = 2 : index\n",
    "config.def @variant = 3 : index\n", "config.def @variant = 4 : index\n",
    "config.def @variant = 5 : index\n", "config.def @variant = 6 : index\n",
    "config.def @variant = 7 : index\n", "config.def @variant = 8 : index\n",
};

typedef struct jit_service_t {
  // Allocator shared by all process- and task-owned objects.
  loomc_allocator_t allocator;

  // Shared context for source deserialization and compilation.
  loomc_context_t* context;

  // Immutable source compiled by each example task.
  loomc_source_t* source;

  // Immutable typed configuration module borrowed by each request.
  loomc_module_t* config_modules[JIT_VARIANT_COUNT];

  // Immutable prepared compiler shared by every worker.
  loomc_compiler_t* compiler;

  // Immutable prepared pass program shared by every worker.
  loomc_pass_program_t* pass_program;

  // Optional standard worker population shared by attached work sources.
  loomc_task_pool_t* task_pool;

  // Compiler task scheduling domain attached to `task_pool`.
  loomc_task_queue_t* compile_queue;

  // Mutable compiler scratch indexed by dense worker ordinal.
  loomc_workspace_t** workspaces;

  // Number of initialized entries in `workspaces`.
  loomc_host_size_t workspace_count;
} jit_service_t;

typedef struct jit_output_t {
  // Terminal infrastructure status produced by the task.
  loomc_status_t status;

  // Terminal compiler result produced by the task.
  loomc_result_t* result;
} jit_output_t;

// --8<-- [start:task-record]
typedef struct jit_task_t {
  // Generic task base owned by the accepting sink.
  loomc_task_t base;

  // Shared immutable service state and worker-local workspace table.
  jit_service_t* service;

  // Immutable typed configuration module for this compilation request.
  const loomc_module_t* config_module;

  // Caller-owned output slot that outlives task execution.
  jit_output_t* output;
} jit_task_t;
// --8<-- [end:task-record]

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

// --8<-- [start:execute]
static void execute_jit_task(loomc_task_t* base_task,
                             loomc_host_size_t worker_ordinal) {
  jit_task_t* task = (jit_task_t*)base_task;
  jit_service_t* service = task->service;
  loomc_workspace_t* workspace = service->workspaces[worker_ordinal];

  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_module_deserialize_text_from_source(
      service->context, workspace, service->source, NULL, service->allocator,
      &module, &result);

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_result_release(result);
    result = NULL;

    const loomc_compile_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .structure_size = sizeof(options),
        .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
        .config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
        .config_module = task->config_module,
    };
    status = loomc_compile_module(service->compiler, workspace,
                                  service->pass_program, module, &options,
                                  service->allocator, &result);
  }

  task->output->status = status;
  task->output->result = result;
  loomc_module_release(module);
  loomc_workspace_trim(workspace);
}
// --8<-- [end:execute]

static void destroy_jit_task(loomc_task_t* base_task) {
  jit_task_t* task = (jit_task_t*)base_task;
  loomc_allocator_free(task->service->allocator, task);
}

static const loomc_task_vtable_t kJitTaskVtable = {
    .execute = execute_jit_task,
    .destroy = destroy_jit_task,
};

static void jit_service_initialize(jit_service_t* service) {
  memset(service, 0, sizeof(*service));
  service->allocator = loomc_allocator_system();
}

static void jit_service_deinitialize(jit_service_t* service) {
  loomc_task_queue_free(service->compile_queue);
  loomc_task_pool_free(service->task_pool);
  for (loomc_host_size_t i = 0; i < JIT_VARIANT_COUNT; ++i) {
    loomc_module_release(service->config_modules[i]);
  }
  for (loomc_host_size_t i = 0; i < service->workspace_count; ++i) {
    loomc_workspace_release(service->workspaces[i]);
  }
  loomc_allocator_free(service->allocator, service->workspaces);
  loomc_pass_program_release(service->pass_program);
  loomc_compiler_release(service->compiler);
  loomc_source_release(service->source);
  loomc_context_release(service->context);
}

static loomc_status_t require_successful_result(const loomc_result_t* result,
                                                const char* message) {
  if (result != NULL && loomc_result_succeeded(result)) {
    return loomc_ok_status();
  }
  print_result_diagnostics(result);
  return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, message);
}

// --8<-- [start:configure]
static loomc_status_t configure_jit_service(jit_service_t* service) {
  loomc_status_t status =
      loomc_context_create(NULL, service->allocator, &service->context);

  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("jit_task_pool.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create(&source_options, service->allocator,
                                 &service->source);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compiler_create(service->context, NULL, service->allocator,
                                   &service->compiler);
  }

  loomc_result_t* pass_result = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_from_pipeline_text(
        service->context, loomc_make_cstring_view("canonicalize,cse,dce"), NULL,
        service->allocator, &service->pass_program, &pass_result);
  }
  if (loomc_status_is_ok(status)) {
    status =
        require_successful_result(pass_result, "pass program creation failed");
  }
  loomc_result_release(pass_result);

  const loomc_task_pool_options_t pool_options = {
      .type = LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS,
      .structure_size = sizeof(pool_options),
      .max_worker_count = 4,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_task_pool_allocate(&pool_options, service->allocator,
                                      &service->task_pool);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_task_queue_allocate(service->task_pool, service->allocator,
                                       &service->compile_queue);
  }
  if (loomc_status_is_ok(status)) {
    const loomc_host_size_t worker_count =
        loomc_task_pool_worker_count(service->task_pool);
    status = loomc_allocator_malloc(service->allocator,
                                    worker_count * sizeof(*service->workspaces),
                                    (void**)&service->workspaces);
    for (loomc_host_size_t i = 0;
         loomc_status_is_ok(status) && i < worker_count; ++i) {
      status = loomc_workspace_create(NULL, service->allocator,
                                      &service->workspaces[i]);
      if (loomc_status_is_ok(status)) ++service->workspace_count;
    }
  }
  for (loomc_host_size_t i = 0;
       loomc_status_is_ok(status) && i < JIT_VARIANT_COUNT; ++i) {
    const char* config_text = kVariantConfigTexts[i];
    const loomc_source_options_t config_source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(config_source_options),
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view("jit_task_pool_config.loom"),
        .contents = loomc_make_byte_span(config_text, strlen(config_text)),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
    };
    loomc_source_t* config_source = NULL;
    loomc_result_t* config_result = NULL;
    status = loomc_source_create(&config_source_options, service->allocator,
                                 &config_source);
    if (loomc_status_is_ok(status)) {
      status = loomc_module_deserialize_text_from_source(
          service->context, service->workspaces[0], config_source, NULL,
          service->allocator, &service->config_modules[i], &config_result);
    }
    if (loomc_status_is_ok(status)) {
      status = require_successful_result(config_result,
                                         "config deserialization failed");
    }
    loomc_result_release(config_result);
    loomc_source_release(config_source);
  }
  if (service->workspace_count != 0) {
    loomc_workspace_trim(service->workspaces[0]);
  }
  return status;
}
// --8<-- [end:configure]

static loomc_status_t allocate_jit_task(jit_service_t* service,
                                        const loomc_module_t* config_module,
                                        jit_output_t* output,
                                        loomc_task_t** out_task) {
  *out_task = NULL;
  jit_task_t* task = NULL;
  loomc_status_t status =
      loomc_allocator_malloc(service->allocator, sizeof(*task), (void**)&task);
  if (loomc_status_is_ok(status)) {
    loomc_task_initialize(&kJitTaskVtable, &task->base);
    task->service = service;
    task->config_module = config_module;
    task->output = output;
    *out_task = &task->base;
  }
  return status;
}

// --8<-- [start:submit]
static loomc_status_t submit_jit_tasks(jit_service_t* service,
                                       jit_output_t* outputs,
                                       loomc_host_size_t output_count) {
  loomc_task_sink_t sink = loomc_task_queue_sink(service->compile_queue);
  loomc_status_t status = loomc_ok_status();
  for (loomc_host_size_t i = 0; loomc_status_is_ok(status) && i < output_count;
       ++i) {
    loomc_task_t* task = NULL;
    status = allocate_jit_task(service, service->config_modules[i], &outputs[i],
                               &task);
    if (loomc_status_is_ok(status)) {
      status = loomc_task_sink_submit(sink, task);
    }
    if (!loomc_status_is_ok(status) && task != NULL) {
      // Rejection preserves the ownership offered to the sink.
      loomc_task_destroy(task);
    }
  }

  // This short-lived example uses shutdown as its batch join. A persistent
  // compiler service tracks request completion separately and shuts its
  // compiler queue down only during service teardown. Other queues and native
  // processes attached to the same pool remain independently runnable.
  if (loomc_status_is_ok(status)) {
    status = loomc_task_queue_shutdown(service->compile_queue);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_task_queue_await_shutdown(service->compile_queue);
  }
  if (!loomc_status_is_ok(status)) {
    // Drain accepted work before caller-owned output storage is inspected or
    // released. The void teardown path preserves the earlier operation error.
    loomc_task_queue_free(service->compile_queue);
    service->compile_queue = NULL;
  }
  return status;
}
// --8<-- [end:submit]

static loomc_status_t inspect_outputs(jit_output_t* outputs,
                                      loomc_host_size_t output_count) {
  loomc_status_t status = loomc_ok_status();
  for (loomc_host_size_t i = 0; i < output_count; ++i) {
    if (!loomc_status_is_ok(outputs[i].status)) {
      if (loomc_status_is_ok(status)) {
        status = outputs[i].status;
        outputs[i].status = loomc_ok_status();
      }
      continue;
    }
    if (!loomc_result_succeeded(outputs[i].result)) {
      print_result_diagnostics(outputs[i].result);
      if (loomc_status_is_ok(status)) {
        status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                   "a JIT task failed compilation");
      }
      continue;
    }
    if (loomc_result_artifact_count(outputs[i].result) != 1 &&
        loomc_status_is_ok(status)) {
      status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                 "a JIT task produced unexpected artifacts");
    }
  }
  return status;
}

static void deinitialize_outputs(jit_output_t* outputs,
                                 loomc_host_size_t output_count) {
  for (loomc_host_size_t i = 0; i < output_count; ++i) {
    loomc_status_free(outputs[i].status);
    loomc_result_release(outputs[i].result);
  }
}

static loomc_status_t run_jit_task_pool_example(void) {
  jit_service_t service;
  jit_service_initialize(&service);
  jit_output_t outputs[JIT_VARIANT_COUNT] = {0};
  const loomc_host_size_t output_count = sizeof(outputs) / sizeof(outputs[0]);

  loomc_status_t status = configure_jit_service(&service);
  if (loomc_status_is_ok(status)) {
    status = submit_jit_tasks(&service, outputs, output_count);
  }
  if (loomc_status_is_ok(status)) {
    status = inspect_outputs(outputs, output_count);
  }
  if (loomc_status_is_ok(status)) {
    printf("compiled=%zu\n", (size_t)output_count);
  }

  deinitialize_outputs(outputs, output_count);
  jit_service_deinitialize(&service);
  return status;
}

int main(void) {
  loomc_status_t status = run_jit_task_pool_example();
  if (loomc_status_is_ok(status)) return 0;
  print_status(status);
  loomc_status_free(status);
  return 1;
}
