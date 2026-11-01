// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/stages.h"

#include "loom/serve/service/text/state.h"
#include "loomc/iree.h"

static iree_status_t loom_serve_text_check_parameters(
    const loom_cmd_program_t* baseline, const loom_cmd_program_t* candidate) {
  if (baseline->parameter_roots.count != candidate->parameter_roots.count ||
      baseline->parameters.count != candidate->parameters.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage parameter counts differ");
  }
  for (uint32_t i = 0; i < baseline->parameter_roots.count; ++i) {
    const loom_cmd_program_parameter_root_t lhs =
        loom_cmd_program_parameter_root_at(baseline, i);
    const loom_cmd_program_parameter_root_t rhs =
        loom_cmd_program_parameter_root_at(candidate, i);
    if (lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
        lhs.required_byte_length != rhs.required_byte_length ||
        lhs.minimum_alignment != rhs.minimum_alignment) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage parameter root %u differs", i);
    }
  }
  for (uint32_t i = 0; i < baseline->parameters.count; ++i) {
    const loom_cmd_program_parameter_t lhs =
        loom_cmd_program_parameter_at(baseline, i);
    const loom_cmd_program_parameter_t rhs =
        loom_cmd_program_parameter_at(candidate, i);
    if (!iree_string_view_equal(lhs.key, rhs.key) ||
        lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
        lhs.byte_offset != rhs.byte_offset ||
        lhs.byte_length != rhs.byte_length ||
        lhs.minimum_alignment != rhs.minimum_alignment) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage parameter placement %u differs", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_check_layouts(
    const loom_serve_preparation_t* preparation, iree_host_size_t shared_count,
    iree_host_size_t stage_count, const loom_serve_text_stage_t* stages) {
  const loom_cmd_program_t* prefill = &stages[0].program;
  const loom_serve_preparation_stage_t* first =
      loom_serve_preparation_stage(preparation, 0);
  if (!first->parameter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source must declare a checkpoint domain");
  }
  const loom_serve_preparation_parameter_t* checkpoint = &first->parameters[0];
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    const loom_cmd_program_t* program = &stages[i].program;
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(preparation, i);
    IREE_RETURN_IF_ERROR(loom_serve_preparation_stage_check_workspace(
        declaration, program->requirements.transient.required_byte_length));
    const uint32_t binding_count =
        program->requirements.rebindable_binding_count;
    const uint32_t roots = program->requirements.fixed_buffer_count;
    if (program->parameter_roots.count != roots ||
        declaration->parameter_count != roots || binding_count < 1 ||
        binding_count > 8 ||
        (program->requirements.transient.binding_index != UINT32_MAX &&
         program->requirements.transient.binding_index != binding_count - 1) ||
        program->requirements.launch_counts.required_byte_length != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage does not match the token runner bindings");
    }
    for (uint32_t root = 0; root < roots; ++root) {
      const loom_serve_preparation_parameter_t* parameter =
          &declaration->parameters[root];
      if (parameter->binding != root ||
          loom_cmd_program_parameter_root_at(program, root)
                  .fixed_buffer_index != root) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stage parameter bindings must match fixed-root order");
      }
    }
    if (i >= shared_count) {
      continue;
    }
    const bool has_workspace =
        program->requirements.transient.required_byte_length != 0;
    if (roots != 1 || binding_count - has_workspace != TEXT_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared stages must implement the target bindings");
    }
    if (!loom_serve_preparation_parameter_equal(&declaration->parameters[0],
                                                checkpoint)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared target stages must use one checkpoint domain");
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_check_parameters(prefill, program));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_stages_initialize(
    iree_host_size_t stage_count, loom_serve_text_stage_t* stages,
    iree_allocator_t host_allocator, iree_host_size_t* out_root_count) {
  iree_status_t status = iree_ok_status();
  *out_root_count = 0;
  for (iree_host_size_t i = 0; i < stage_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_text_stage_t* stage = &stages[i];
    const uint32_t count = stage->program.requirements.fixed_buffer_count;
    if (count) {
      status = iree_allocator_malloc_array(host_allocator, count,
                                           sizeof(*stage->fixed_buffers),
                                           (void**)&stage->fixed_buffers);
      if (iree_status_is_ok(status)) {
        stage->fixed_buffer_count = count;
        *out_root_count += count;
      }
    }
  }
  return status;
}

// Group roots once during binding. The original stage order preserves the
// identical-layout target prefix in the first domain.
iree_status_t loom_serve_text_stages_bind(
    loom_serve_device_t* device, loom_serve_pool_t* pool,
    const loom_serve_preparation_t* preparation, iree_host_size_t shared_count,
    iree_host_size_t stage_count, loom_serve_text_stage_t* stages,
    loom_serve_text_parameters_t* parameters, iree_allocator_t host_allocator) {
  iree_host_size_t root_count = 0;
  IREE_RETURN_IF_ERROR(loom_serve_text_stages_initialize(
      stage_count, stages, host_allocator, &root_count));
  IREE_RETURN_IF_ERROR(loom_serve_text_check_layouts(preparation, shared_count,
                                                     stage_count, stages));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, root_count,
                                                   sizeof(*parameters->plans),
                                                   (void**)&parameters->plans));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, root_count, sizeof(*parameters->sources),
      (void**)&parameters->sources));
  loom_serve_weight_root_t* roots = NULL;
  iree_host_size_t* domains = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, root_count, sizeof(*roots), (void**)&roots));
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, root_count, sizeof(*domains), (void**)&domains);
  if (iree_status_is_ok(status)) {
    iree_host_size_t root_index = 0;
    for (iree_host_size_t i = 0; i < stage_count; ++i) {
      const loom_serve_preparation_stage_t* declaration =
          loom_serve_preparation_stage(preparation, i);
      for (uint32_t r = 0; r < declaration->parameter_count; ++r) {
        const loom_serve_preparation_parameter_t* source =
            &declaration->parameters[r];
        iree_host_size_t domain = 0;
        for (; domain < parameters->count; ++domain) {
          const loom_serve_preparation_parameter_t* prior =
              parameters->sources[domain];
          if (loom_serve_preparation_parameter_equal(source, prior)) {
            break;
          }
        }
        if (domain == parameters->count) {
          parameters->sources[parameters->count++] = source;
        }
        domains[root_index++] = domain;
      }
    }
    for (iree_host_size_t domain = 0;
         domain < parameters->count && iree_status_is_ok(status); ++domain) {
      iree_host_size_t root_index = 0;
      iree_host_size_t count = 0;
      for (iree_host_size_t i = 0; i < stage_count; ++i) {
        const loom_cmd_program_t* program = &stages[i].program;
        for (uint32_t r = 0; r < program->parameter_roots.count; ++r) {
          if (domains[root_index++] != domain) {
            continue;
          }
          const loom_cmd_program_parameter_root_t root =
              loom_cmd_program_parameter_root_at(program, r);
          roots[count++] = (loom_serve_weight_root_t){
              .program = program,
              .root = root,
              .buffer = &stages[i].fixed_buffers[root.fixed_buffer_index],
          };
        }
      }
      status = loom_serve_weights_allocate(
          loom_serve_device_handle(device),
          loom_serve_device_transfer_queue(device),
          loom_serve_device_dispatch_queue(device),
          loom_serve_pool_storage(pool), loom_serve_pool_memory(pool),
          loom_serve_pool_weight_geometry(pool), NULL,
          loom_serve_pool_workspace(pool), domain == 0 ? shared_count : 0,
          count, roots, host_allocator, &parameters->plans[domain]);
    }
  }
  iree_allocator_free(host_allocator, domains);
  iree_allocator_free(host_allocator, roots);
  return status;
}

iree_status_t loom_serve_text_stages_rebind(
    const loom_serve_preparation_t* preparation, iree_host_size_t shared_count,
    iree_host_size_t stage_count, const loom_serve_text_stage_t* baseline,
    loom_serve_text_stage_t* stages, iree_allocator_t host_allocator) {
  iree_host_size_t root_count = 0;
  IREE_RETURN_IF_ERROR(loom_serve_text_stages_initialize(
      stage_count, stages, host_allocator, &root_count));
  IREE_RETURN_IF_ERROR(loom_serve_text_check_layouts(preparation, shared_count,
                                                     stage_count, stages));
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    const loom_cmd_program_requirements_t* original =
        &baseline[i].program.requirements;
    const loom_cmd_program_requirements_t* replacement =
        &stages[i].program.requirements;
    const uint32_t original_arity =
        original->rebindable_binding_count -
        (original->transient.required_byte_length != 0);
    const uint32_t replacement_arity =
        replacement->rebindable_binding_count -
        (replacement->transient.required_byte_length != 0);
    if (original_arity != replacement_arity) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "replacement stage %zu changes model bindings",
                              i);
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_check_parameters(&baseline[i].program,
                                                          &stages[i].program));
    for (uint32_t j = 0; j < stages[i].fixed_buffer_count; ++j) {
      stages[i].fixed_buffers[j] = baseline[i].fixed_buffers[j];
      iree_hal_buffer_retain(stages[i].fixed_buffers[j]);
    }
  }
  return iree_ok_status();
}

typedef struct loom_serve_text_record_job_t loom_serve_text_record_job_t;

typedef struct loom_serve_text_record_task_t {
  // Queue ownership ends before releasing the task's joined job reference.
  loomc_task_t task;
  // Borrowed job retained until this task's terminal release.
  loom_serve_text_record_job_t* job;
  // Private outcome, joined only after every task has retired.
  iree_status_t status;
} loom_serve_text_record_task_t;

struct loom_serve_text_record_job_t {
  // Submission hold plus one reference for each offered task.
  iree_atomic_ref_count_t ref_count;
  // Next disjoint output slot; failure advances it past all remaining work.
  iree_atomic_int64_t next_index;
  // Allocator owning this job and its bounded inline task array.
  iree_allocator_t host_allocator;
  // Borrowed queue family shared by every stage in the model.
  const iree_hal_queue_family_t* queue_family;
  // Command retention policy selected by the model owner.
  iree_hal_command_buffer_mode_t mode;
  // Number of command slots, independent of the number of tasks.
  iree_host_size_t stage_count;
  // Borrowed immutable inputs and disjoint owned command outputs.
  loom_serve_text_stage_t* stages;
  // Terminal consumer after all borrowed accesses have retired.
  loom_serve_text_stages_completion_fn_t completion;
  // Borrowed completion payload.
  void* user_data;
  // Number of outcome slots in tasks, including unsubmitted zeroed slots.
  iree_host_size_t task_count;
  // Storage bounded by worker concurrency, not command count.
  loom_serve_text_record_task_t tasks[];
};

static void loom_serve_text_record_job_release(
    loom_serve_text_record_job_t* job) {
  if (!job) {
    return;
  }
  if (iree_atomic_ref_count_dec(&job->ref_count) != 1) {
    return;
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < job->task_count; ++i) {
    status = iree_status_join(status, job->tasks[i].status);
  }
  const loom_serve_text_stages_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(job->host_allocator, job);
  completion(user_data, status);
}

static void loom_serve_text_record_execute(loomc_task_t* base,
                                           loomc_host_size_t worker_ordinal) {
  loom_serve_text_record_task_t* task = (void*)base;
  loom_serve_text_record_job_t* job = task->job;
  while (iree_status_is_ok(task->status)) {
    const iree_host_size_t index = (iree_host_size_t)iree_atomic_fetch_add(
        &job->next_index, 1, iree_memory_order_relaxed);
    if (index >= job->stage_count) {
      break;
    }
    loom_serve_text_stage_t* stage = &job->stages[index];
    task->status = loom_serve_jit_stage_record(
        stage->compiled, job->queue_family, job->mode, NULL,
        stage->fixed_buffers, &stage->command);
  }
  if (!iree_status_is_ok(task->status)) {
    iree_atomic_store(&job->next_index, job->stage_count,
                      iree_memory_order_relaxed);
  }
}

static void loom_serve_text_record_release(loomc_task_t* base) {
  loom_serve_text_record_task_t* task = (void*)base;
  loom_serve_text_record_job_release(task->job);
}

static const loomc_task_vtable_t loom_serve_text_record_vtable = {
    .execute = loom_serve_text_record_execute,
    .destroy = loom_serve_text_record_release,
};

iree_status_t loom_serve_text_stages_record_async(
    loomc_task_sink_t sink, iree_host_size_t concurrency,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode, iree_host_size_t stage_count,
    loom_serve_text_stage_t* stages,
    loom_serve_text_stages_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator) {
  const iree_host_size_t task_count = iree_min(stage_count, concurrency);
  loom_serve_text_record_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*job), task_count, sizeof(*job->tasks),
      (void**)&job));
  iree_atomic_ref_count_init(&job->ref_count);
  job->host_allocator = host_allocator;
  job->queue_family = queue_family;
  job->mode = mode;
  job->stage_count = stage_count;
  job->stages = stages;
  job->completion = completion;
  job->user_data = user_data;
  job->task_count = task_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < task_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_text_record_task_t* task = &job->tasks[i];
    loomc_task_initialize(&loom_serve_text_record_vtable, &task->task);
    task->job = job;
    iree_atomic_ref_count_inc(&job->ref_count);
    status = iree_status_from_loomc(loomc_task_sink_submit(sink, &task->task));
    if (!iree_status_is_ok(status)) {
      // Previously accepted tasks still own their references. Stop claims,
      // retain this failure, and consume only the rejected task's ownership.
      iree_atomic_store(&job->next_index, stage_count,
                        iree_memory_order_relaxed);
      task->status = status;
      loomc_task_destroy(&task->task);
    }
  }
  loom_serve_text_record_job_release(job);
  return iree_ok_status();
}

loom_serve_stage_t loom_serve_text_stage_binding(
    const loom_serve_text_stage_t* stage) {
  const loom_cmd_program_requirements_t* requirements =
      &stage->program.requirements;
  return (loom_serve_stage_t){
      .command_buffer = stage->command,
      .binding_count = requirements->rebindable_binding_count -
                       (requirements->transient.required_byte_length != 0),
      .transient = requirements->transient,
  };
}
