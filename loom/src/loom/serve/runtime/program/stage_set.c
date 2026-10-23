// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/stage_set.h"

#include "loom/target/arch/cmd/program.h"
#include "loomc/iree.h"

typedef struct loom_serve_stage_set_job_t loom_serve_stage_set_job_t;

typedef struct loom_serve_stage_set_entry_t {
  // Joined owner retained through this stage's completion callback.
  loom_serve_stage_set_job_t* job;
  // Immutable declaration/result ordinal.
  iree_host_size_t index;
  // Owned stage outcome, consumed at the final join in declaration order.
  iree_status_t status;
} loom_serve_stage_set_entry_t;

struct loom_serve_stage_set_job_t {
  // Submission loop plus each accepted stage callback.
  iree_atomic_ref_count_t ref_count;
  // Initial source indexing task, released before stage compilation.
  loomc_task_t task;
  // Owned result transferred at terminal completion, including on failure.
  loom_serve_stage_set_t* stages;
  // Borrowed device and compiler host through completion.
  loom_serve_device_t* device;
  // Borrowed immutable original declarations and specialization values.
  const loom_serve_preparation_t* preparation;
  // Borrowed provider namespace through source indexing.
  loom_serve_sources_t sources;
  // Retained catalog, either caller-provided or created by source indexing.
  loom_serve_jit_t* jit;
  // Initial task result consumed only after task release.
  iree_status_t status;
  // Optional immutable layout consumer, independent of native readiness.
  loom_serve_stage_set_layout_fn_t layout;
  // Terminal consumer after compiler and source accesses retire.
  loom_serve_stage_set_completion_fn_t completion;
  // Borrowed terminal consumer state.
  void* user_data;
  // One disjoint outcome for every immutable preparation declaration.
  loom_serve_stage_set_entry_t entries[];
};

void loom_serve_stage_set_free(loom_serve_stage_set_t* stages) {
  if (!stages) {
    return;
  }
  for (iree_host_size_t i = 0; i < stages->count; ++i) {
    loom_serve_jit_stage_free(stages->values[i]);
  }
  iree_allocator_free(stages->host_allocator, stages->values);
  iree_allocator_free(stages->host_allocator, stages);
}

static void loom_serve_stage_set_job_release(loom_serve_stage_set_job_t* job) {
  if (!job) {
    return;
  }
  if (iree_atomic_ref_count_dec(&job->ref_count) != 1) {
    return;
  }
  loom_serve_stage_set_t* stages = job->stages;
  iree_status_t status = job->status;
  for (iree_host_size_t i = 0; i < stages->count; ++i) {
    status = iree_status_join(status, job->entries[i].status);
  }
  if (job->jit) {
    loom_serve_jit_report_snapshot(job->jit, &stages->compilation);
  }
  loom_serve_jit_release(job->jit);
  const loom_serve_stage_set_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(stages->host_allocator, job);
  completion(user_data, status, stages);
}

static void loom_serve_stage_set_layout(void* user_data,
                                        const loom_cmd_program_t* program) {
  loom_serve_stage_set_entry_t* entry = user_data;
  entry->job->layout(entry->job->user_data, entry->index, program);
}

static void loom_serve_stage_set_compiled(void* user_data, iree_status_t status,
                                          loom_serve_jit_stage_t* stage) {
  loom_serve_stage_set_entry_t* entry = user_data;
  loom_serve_stage_set_job_t* job = entry->job;
  if (iree_status_is_ok(status)) {
    const loom_cmd_program_t* program = loom_serve_jit_stage_program(stage);
    if (program) {
      status = loom_serve_preparation_stage_check_workspace(
          loom_serve_preparation_stage(job->preparation, entry->index),
          program->requirements.transient.required_byte_length);
    }
  }
  job->stages->values[entry->index] = stage;
  entry->status = status;
  loom_serve_stage_set_job_release(job);
}

static void loom_serve_stage_set_submit(loom_serve_stage_set_job_t* job) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < job->stages->count && iree_status_is_ok(status); ++i) {
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(job->preparation, i);
    loom_serve_stage_set_entry_t* entry = &job->entries[i];
    iree_atomic_ref_count_inc(&job->ref_count);
    status = declaration->kind == LOOM_SERVE_PREPARATION_STAGE_DISPATCH
                 ? loom_serve_jit_compile_dispatch_async(
                       job->jit, loom_serve_jit_task_sink(job->jit),
                       declaration->root, &declaration->config,
                       loom_serve_stage_set_compiled, entry)
                 : loom_serve_jit_compile_async(
                       job->jit, loom_serve_jit_task_sink(job->jit),
                       declaration->root, &declaration->config,
                       job->layout ? loom_serve_stage_set_layout : NULL,
                       loom_serve_stage_set_compiled, entry);
    // The loop retains job even if callbacks finish inline. Rejection owns
    // this outcome and owes no callback; previously accepted stages still join.
    if (!iree_status_is_ok(status)) {
      loom_serve_stage_set_compiled(entry, status, NULL);
    }
  }
  loom_serve_stage_set_job_release(job);
}

static void loom_serve_stage_set_execute(loomc_task_t* task,
                                         loomc_host_size_t worker_ordinal) {
  loom_serve_stage_set_job_t* job =
      (loom_serve_stage_set_job_t*)((uint8_t*)task -
                                    offsetof(loom_serve_stage_set_job_t, task));
  job->status = loom_serve_jit_create(
      loom_serve_device_handle(job->device),
      loom_serve_device_dispatch_queue(job->device),
      loom_serve_device_host(job->device), &job->sources, NULL,
      &job->stages->compilation, job->stages->host_allocator, &job->jit);
}

static void loom_serve_stage_set_release(loomc_task_t* task) {
  loom_serve_stage_set_job_t* job =
      (loom_serve_stage_set_job_t*)((uint8_t*)task -
                                    offsetof(loom_serve_stage_set_job_t, task));
  if (iree_status_is_ok(job->status)) {
    loom_serve_stage_set_submit(job);
  } else {
    loom_serve_stage_set_job_release(job);
  }
}

static const loomc_task_vtable_t loom_serve_stage_set_vtable = {
    .execute = loom_serve_stage_set_execute,
    .destroy = loom_serve_stage_set_release,
};

static iree_status_t loom_serve_stage_set_job_create(
    const loom_serve_preparation_t* preparation,
    loom_serve_stage_set_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator, loom_serve_stage_set_job_t** out_job) {
  *out_job = NULL;
  loom_serve_stage_set_t* stages = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*stages), (void**)&stages));
  stages->host_allocator = host_allocator;
  const iree_host_size_t count =
      loom_serve_preparation_stage_count(preparation);
  loom_serve_stage_set_job_t* job = NULL;
  iree_status_t status = iree_ok_status();
  if (count) {
    status = iree_allocator_malloc_array(host_allocator, count,
                                         sizeof(*stages->values),
                                         (void**)&stages->values);
  }
  if (iree_status_is_ok(status)) {
    stages->count = count;
    status =
        iree_allocator_malloc_struct_array(host_allocator, sizeof(*job), count,
                                           sizeof(*job->entries), (void**)&job);
  }
  if (iree_status_is_ok(status)) {
    loomc_task_initialize(&loom_serve_stage_set_vtable, &job->task);
    iree_atomic_ref_count_init(&job->ref_count);
    job->stages = stages;
    job->preparation = preparation;
    job->completion = completion;
    job->user_data = user_data;
    for (iree_host_size_t i = 0; i < count; ++i) {
      job->entries[i].job = job;
      job->entries[i].index = i;
    }
    *out_job = job;
  } else {
    loom_serve_stage_set_free(stages);
  }
  return status;
}

iree_status_t loom_serve_stage_set_compile_async(
    loom_serve_device_t* device, const loom_serve_preparation_t* preparation,
    const loom_serve_sources_t* sources,
    loom_serve_stage_set_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator) {
  loom_serve_stage_set_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_stage_set_job_create(
      preparation, completion, user_data, host_allocator, &job));
  job->device = device;
  job->sources = *sources;
  iree_status_t status = iree_status_from_loomc(loomc_task_sink_submit(
      loom_serve_host_compiler_sink(loom_serve_device_host(device)),
      &job->task));
  if (!iree_status_is_ok(status)) {
    loom_serve_stage_set_free(job->stages);
    iree_allocator_free(host_allocator, job);
  }
  return status;
}

iree_status_t loom_serve_stage_set_compile_catalog_async(
    loom_serve_jit_t* jit, const loom_serve_preparation_t* preparation,
    loom_serve_stage_set_layout_fn_t layout,
    loom_serve_stage_set_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator) {
  loom_serve_stage_set_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_stage_set_job_create(
      preparation, completion, user_data, host_allocator, &job));
  job->jit = jit;
  job->layout = layout;
  loom_serve_jit_retain(jit);
  loom_serve_stage_set_submit(job);
  return iree_ok_status();
}
