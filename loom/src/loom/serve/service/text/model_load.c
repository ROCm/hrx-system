// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cold construction and asynchronous publication of source text models.

#include "iree/async/operations/semaphore.h"
#include "iree/tooling/device_profiling.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/workspace.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/stage_set.h"
#include "loom/serve/runtime/program/tokenizer_load.h"
#include "loom/serve/service/text/control.h"
#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/source.h"
#include "loom/serve/service/text/stages.h"
#include "loom/serve/service/text/state.h"
#include "loom/serve/service/text/storage.h"
#include "loomc/iree.h"

typedef enum loom_serve_text_load_phase_e {
  TEXT_LOAD_PREPARE,
  TEXT_LOAD_COMPILE,
  TEXT_LOAD_BIND,
  TEXT_LOAD_WEIGHTS,
  TEXT_LOAD_RESIDENCY,
  TEXT_LOAD_ACTIVATE,
  TEXT_LOAD_RECORD,
  TEXT_LOAD_LINK,
  TEXT_LOAD_READY,
  TEXT_LOAD_COMPLETE,
} loom_serve_text_load_phase_t;

struct loom_serve_text_load_t {
  // Private bootstrap task; final release publishes completion to the owner.
  loomc_task_t task;
  // Allocation policy for this operation, independent of its model.
  iree_allocator_t host_allocator;
  // Partial model, transferred on every terminal outcome including failure.
  loom_serve_text_model_t* model;
  // Terminal aggregate after every independent compiler producer retires.
  loom_serve_compilation_report_t compilation;
  // Borrowed source paths and shapes, retained through terminal advance.
  loom_serve_text_options_t options;
  // Serialized owner-only phase; callbacks publish results, never transition.
  loom_serve_text_load_phase_t phase;
  // Once cancelled, subsequent advance calls cannot restart preparation.
  bool cancelled;
  // Private task result transferred only by the task's final release.
  iree_status_t task_status;
  // Independent control compilation; its final release publishes readiness.
  struct {
    // Intrusive task borrowing the initialized source and model catalog.
    loomc_task_t task;
    // Private compile result transferred at final task release.
    iree_status_t status;
  } control;
  // Independent tokenizer construction, joined before linking or retirement.
  loom_serve_tokenizer_load_t* tokenizer_load;
  // Terminal model outcome retained while tokenizer work is still retiring.
  iree_status_t terminal_status;
  // Joins worker/proactor final publication with the owner's next advance.
  struct {
    // Protects completion publication and its final notification access.
    iree_slim_mutex_t mutex;
    // Accepted foreground callbacks plus any active submission hold.
    iree_host_size_t pending;
    // Native stage compilation may overlap parameter preparation/activation.
    bool stages_compiling;
    // Control bytecode is unavailable until its independent task retires.
    bool control_compiling;
    // Immutable command layouts published by the accepted stage set.
    iree_host_size_t layout_count;
    // Owned callback outcome, consumed by advance.
    iree_status_t status;
    // Nonblocking serialized-owner notification.
    loom_serve_text_load_wake_fn_t wake;
    // Borrowed notification payload retained through terminal advance.
    void* user_data;
  } event;
  // Initialization upload readiness; failure still requires model retirement.
  struct {
    // Intrusive proactor operation retained through its final callback.
    iree_async_semaphore_wait_operation_t operation;
    // Borrowed semaphores owned by the partial model's execution domain.
    iree_async_semaphore_t** semaphores;
    // Captured compute and feedback values for every execution.
    uint64_t* values;
  } readiness;
};

static void loom_serve_text_load_complete(void* user_data,
                                          iree_status_t status) {
  loom_serve_text_load_t* load = user_data;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status = iree_status_join(load->event.status, status);
  if (--load->event.pending == 0) {
    load->event.wake(load->event.user_data);
  }
  iree_slim_mutex_unlock(&load->event.mutex);
}

// Parameter plans have disjoint preparation state and share immutable source
// catalogs. Preclaim their callbacks and a submission hold so inline completion
// and partial rejection cannot publish readiness before the fanout retires.
static void loom_serve_text_load_prepare_weights(loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  loom_serve_host_t* host = loom_serve_device_host(model->devices[0]);
  const iree_host_size_t count = model->parameters.count;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.pending = count + 1;
  iree_slim_mutex_unlock(&load->event.mutex);
  iree_status_t status = iree_ok_status();
  iree_host_size_t accepted = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_serve_weights_prepare_async(
        model->parameters.plans[i], host,
        model->compiler.catalogs[model->parameters.sources[i]->owner_index],
        model->command_mode, &load->options.sources,
        model->parameters.sources[i], model->source.parameter_sources,
        loom_serve_text_load_complete, load);
    accepted += iree_status_is_ok(status) ? 1 : 0;
  }
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.pending -= count - accepted;
  iree_slim_mutex_unlock(&load->event.mutex);
  loom_serve_text_load_complete(load, status);
}

static iree_status_t loom_serve_text_load_prepare(
    loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  const loom_serve_text_options_t* options = &load->options;
  if (options->mode == LOOM_SERVE_TEXT_MODE_GENERATE) {
    IREE_RETURN_IF_ERROR(loom_serve_tokenizer_load_start(
        loom_serve_device_tokenizer_cache(model->devices[0]),
        loom_serve_host_compiler_sink(
            loom_serve_device_host(model->devices[0])),
        &options->sources, iree_string_view_empty(),
        options->sources.card ? IREE_SV("text") : options->tokenizer_path,
        load->event.wake, load->event.user_data, model->host_allocator,
        &load->tokenizer_load));
  }
  model->shape_count = options->epoch_count;
  model->context_capacity = options->context_capacity;
  if (model->shape_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        model->host_allocator,
        iree_make_const_byte_span(options->epoch_shapes,
                                  model->shape_count * sizeof(*model->shapes)),
        (void**)&model->shapes));
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < model->device_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_device_create_jit(
        model->devices[i], &options->sources, &options->kernel_sanitizer,
        &model->compiler.reports[i], &model->compiler.catalogs[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_source_initialize(
      model->compiler.catalogs[0], model->device_count, options,
      model->host_allocator, &model->source, &model->state.storage));
  model->state.cache.capacity = model->source.pool_capacity;
  model->state.checkpoints.capacity =
      (uint32_t)model->source.cache.checkpoint_capacity;
  const iree_host_size_t stage_count =
      loom_serve_preparation_stage_count(model->source.preparation);
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    if (loom_serve_preparation_stage(model->source.preparation, i)->kind !=
        LOOM_SERVE_PREPARATION_STAGE_COMMAND) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "text stages require command parameter reflection");
    }
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->host_allocator, stage_count, sizeof(*model->stages),
      (void**)&model->stages));
  model->stage_count = stage_count;
  return iree_ok_status();
}

static void loom_serve_text_load_execute(loomc_task_t* task,
                                         loomc_host_size_t worker_ordinal) {
  loom_serve_text_load_t* load = (loom_serve_text_load_t*)task;
  load->task_status = loom_serve_text_load_prepare(load);
}

static void loom_serve_text_load_release(loomc_task_t* task) {
  loom_serve_text_load_t* load = (loom_serve_text_load_t*)task;
  loom_serve_text_load_complete(load, load->task_status);
}

static const loomc_task_vtable_t text_load_vtable = {
    .execute = loom_serve_text_load_execute,
    .destroy = loom_serve_text_load_release,
};

static void loom_serve_text_load_control_execute(
    loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  loom_serve_text_load_t* load =
      (loom_serve_text_load_t*)((uint8_t*)task -
                                offsetof(loom_serve_text_load_t, control.task));
  load->control.status = loom_serve_text_source_compile_control(
      load->model->compiler.catalogs[0], &load->options, &load->model->source);
}

static void loom_serve_text_load_control_release(loomc_task_t* task) {
  loom_serve_text_load_t* load =
      (loom_serve_text_load_t*)((uint8_t*)task -
                                offsetof(loom_serve_text_load_t, control.task));
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status =
      iree_status_join(load->event.status, load->control.status);
  load->event.control_compiling = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static const loomc_task_vtable_t text_load_control_vtable = {
    .execute = loom_serve_text_load_control_execute,
    .destroy = loom_serve_text_load_control_release,
};

static void loom_serve_text_load_layout(void* user_data, iree_host_size_t index,
                                        const loom_cmd_program_t* program) {
  loom_serve_text_load_t* load = user_data;
  iree_slim_mutex_lock(&load->event.mutex);
  load->model->stages[index].program = *program;
  if (++load->event.layout_count == load->model->stage_count) {
    load->event.wake(load->event.user_data);
  }
  iree_slim_mutex_unlock(&load->event.mutex);
}

static void loom_serve_text_load_compiled(void* user_data, iree_status_t status,
                                          loom_serve_stage_set_t* compiled) {
  loom_serve_text_load_t* load = user_data;
  for (iree_host_size_t i = 0; i < compiled->count; ++i) {
    load->model->stages[i].compiled = compiled->values[i];
    compiled->values[i] = NULL;
  }
  loom_serve_stage_set_free(compiled);
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status = iree_status_join(load->event.status, status);
  load->event.stages_compiling = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static void loom_serve_text_load_ready(void* user_data,
                                       iree_async_operation_t* operation,
                                       iree_status_t status,
                                       iree_async_completion_flags_t flags) {
  loom_serve_text_load_complete(user_data, status);
}

static iree_status_t loom_serve_text_load_bind(loom_serve_text_model_t* model) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < model->device_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_device_create_execution(
        model->devices[i], model->pools[i], &model->executions[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_stages_bind(
      model->devices, model->pools, model->source.preparation,
      model->source.shared_stage_count, model->stage_count, model->stages,
      &model->parameters, model->host_allocator));
  loom_serve_weights_t** plans = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->host_allocator, model->parameters.count, sizeof(*plans),
      (void**)&plans));
  for (uint32_t i = 0; i < model->residency.count && iree_status_is_ok(status);
       ++i) {
    iree_host_size_t count = 0;
    for (iree_host_size_t j = 0; j < model->parameters.count; ++j) {
      if (model->residency
              .pool_indices[model->parameters.sources[j]->owner_index] == i) {
        plans[count++] = model->parameters.plans[j];
      }
    }
    status = loom_serve_residency_group_allocate(
        loom_serve_pool_residency_cache(
            model->pools[model->residency.execution_indices[i]]),
        count, plans, model->host_allocator, &model->residency.groups[i]);
  }
  iree_allocator_free(model->host_allocator, plans);
  return status;
}

static iree_status_t loom_serve_text_load_activate(
    loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  loom_serve_device_t* devices[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  loom_serve_pool_t* pools[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  for (uint32_t i = 0; i < model->residency.count; ++i) {
    const uint32_t execution = model->residency.execution_indices[i];
    devices[i] = model->devices[execution];
    pools[i] = model->pools[execution];
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_warmup_allocate(
      model->residency.count, devices, pools, model->residency.groups,
      load->event.wake, load->event.user_data, load->host_allocator,
      &model->warmup));
  loom_serve_text_warmup_poll(model->warmup);
  return loom_serve_text_warmup_status(model->warmup);
}

static iree_status_t loom_serve_text_load_link(loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  iree_arena_initialize(
      loom_serve_host_block_pool(loom_serve_device_host(model->devices[0])),
      &model->row_records);
  iree_status_t status = iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_serve_text_state_allocate(
      &model->state, model->devices[0], model->pools[0], model->executions[0]));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(model->host_allocator, model->row_count,
                                  sizeof(*model->rows), (void**)&model->rows));
  for (iree_host_size_t i = 0;
       i < model->row_count && iree_status_is_ok(status); ++i) {
    status = iree_arena_allocate(&model->row_records, sizeof(**model->rows),
                                 (void**)&model->rows[i]);
    if (iree_status_is_ok(status)) {
      *model->rows[i] = (loom_serve_text_row_t){.model = model,
                                                .state = model->state.rows[i]};
    }
  }
  IREE_RETURN_IF_ERROR(status);
  iree_host_size_t span_capacity = 1;
  for (iree_host_size_t i = 0; i < model->shape_count; ++i) {
    span_capacity = iree_max(span_capacity, model->shapes[i].span_capacity);
  }
  iree_host_size_t total = 0, indices_offset = 0, extents_offset = 0;
  iree_host_size_t next_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total,
      IREE_STRUCT_FIELD(span_capacity, iree_host_size_t, &indices_offset),
      IREE_STRUCT_FIELD(span_capacity, uint32_t, &extents_offset),
      IREE_STRUCT_FIELD(span_capacity, uint32_t, &next_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(model->host_allocator, total, (void**)&storage));
  model->invocation.row_indices = (iree_host_size_t*)(storage + indices_offset);
  model->invocation.extents = (uint32_t*)(storage + extents_offset);
  model->invocation.next_indices = (uint32_t*)(storage + next_offset);
  if (model->state.checkpoints.capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        model->host_allocator, model->state.checkpoints.capacity,
        sizeof(*model->checkpoints), (void**)&model->checkpoints));
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->host_allocator, model->stage_count, sizeof(*model->stage_bindings),
      (void**)&model->stage_bindings));
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    model->stage_bindings[i] = loom_serve_text_stage_binding(
        &model->stages[i],
        loom_serve_preparation_stage(model->source.preparation, i)
            ->execution_index);
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_control_initialize(
      &model->source, model->device_count, model->executions, model->transfer,
      &model->state, model->shape_count, model->shapes, model->stage_count,
      model->stage_bindings, model->host_allocator, &model->control));
  for (uint32_t i = 0; i < model->device_count; ++i) {
    uint64_t* bytes =
        &model->residency.workspace_bytes[model->residency.pool_indices[i]];
    if (!iree_checked_add_u64(*bytes,
                              loom_serve_module_workspace_requirement(
                                  model->control.native_module, i)
                                  .allocation_size,
                              bytes)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "combined command workspace exceeds byte accounting");
    }
  }
  for (uint32_t i = 0; i < model->residency.count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_residency_prepare_working_set(
        model->residency.values[i], model->residency.workspace_bytes[i], 0);
  }
  IREE_RETURN_IF_ERROR(status);
  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(loom_serve_device_host(model->devices[0])),
      0, &proactor));
  for (uint32_t i = 0; i < model->device_count; ++i) {
    const loom_serve_execution_frontier_t frontier =
        loom_serve_execution_frontier(model->executions[i]);
    for (uint32_t j = 0; j < 2; ++j) {
      load->readiness.semaphores[2 * i + j] =
          (iree_async_semaphore_t*)frontier.semaphores[j];
      load->readiness.values[2 * i + j] = frontier.values[j];
    }
  }
  iree_async_semaphore_wait_operation_t* wait = &load->readiness.operation;
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, loom_serve_text_load_ready, load);
  wait->semaphores = load->readiness.semaphores;
  wait->values = load->readiness.values;
  wait->count = 2 * model->device_count;
  wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
  return iree_async_proactor_submit_one(proactor, &wait->base);
}

iree_status_t loom_serve_text_load_allocate(
    iree_host_size_t device_count, loom_serve_device_t* const* devices,
    loom_serve_pool_t* const* pools, const loom_serve_text_options_t* options,
    loom_serve_text_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_text_load_t** out_load) {
  *out_load = NULL;
  if (!device_count || device_count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "text deployment requires 1..%u devices",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  for (iree_host_size_t i = 1; i < device_count; ++i) {
    if (loom_serve_device_group(devices[i]) !=
        loom_serve_device_group(devices[0])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "text execution devices must share a group");
    }
  }
  if ((options->mode != LOOM_SERVE_TEXT_MODE_GENERATE &&
       options->mode != LOOM_SERVE_TEXT_MODE_ENCODE) ||
      (options->mode == LOOM_SERVE_TEXT_MODE_ENCODE &&
       (!options->epoch_count || options->draft_depth ||
        !iree_string_view_is_empty(options->tokenizer_path)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding requires packed epochs without a "
                            "tokenizer or draft stages");
  }
  if (options->row_count < 1 || options->row_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "text resident count must fit a positive i32");
  }
  const iree_host_size_t session_capacity =
      options->cache.session_capacity == LOOM_SERVE_TEXT_CACHE_AUTOMATIC
          ? 0
          : options->cache.session_capacity;
  const iree_host_size_t prefix_capacity =
      options->cache.prefix_capacity == LOOM_SERVE_TEXT_CACHE_AUTOMATIC
          ? 0
          : options->cache.prefix_capacity;
  const bool pooled = options->pool_capacity || options->cache_limit_bytes;
  const bool automatic_cache =
      options->cache.target_bytes &&
      (options->cache.session_capacity == LOOM_SERVE_TEXT_CACHE_AUTOMATIC ||
       options->cache.prefix_capacity == LOOM_SERVE_TEXT_CACHE_AUTOMATIC);
  if (session_capacity >= INT32_MAX || prefix_capacity > INT32_MAX ||
      options->checkpoint_capacity > INT32_MAX ||
      (uint64_t)options->checkpoint_capacity + prefix_capacity +
              (session_capacity ? session_capacity + 1 : 0) >
          INT32_MAX ||
      ((session_capacity || prefix_capacity || automatic_cache) && !pooled)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cache endpoints require pooled storage and a "
                            "combined capacity fitting i32");
  }
  if (options->prefill_capacity < 1 || options->prefill_capacity > INT32_MAX ||
      options->context_capacity < 1 || options->context_capacity > INT32_MAX ||
      options->epoch_count > INT32_MAX ||
      options->epoch_count > SIZE_MAX / (2 * sizeof(int64_t)) ||
      options->draft_depth > 7 ||
      (options->draft_depth && !options->epoch_count) ||
      (uint64_t)options->pool_capacity > INT64_MAX ||
      options->checkpoint_capacity > INT32_MAX ||
      (options->checkpoint_capacity && !pooled) ||
      (options->pool_capacity && options->cache_limit_bytes) ||
      (pooled && !options->epoch_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid model capacities; pooled KV and drafting "
                            "require epoch shapes, "
                            "and draft depth must be in [0, 7]");
  }
  for (iree_host_size_t i = 0; i < options->epoch_count; ++i) {
    const loom_serve_packing_shape_t shape = options->epoch_shapes[i];
    if (shape.token_capacity < 1 || shape.token_capacity > INT32_MAX ||
        shape.span_capacity < 1 || shape.span_capacity > shape.token_capacity) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "epoch %zu has invalid token/span capacities", i);
    }
  }
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  iree_host_size_t load_size = 0, ready_semaphores = 0, ready_values = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_load_t), &load_size,
      IREE_STRUCT_FIELD(2 * device_count, iree_async_semaphore_t*,
                        &ready_semaphores),
      IREE_STRUCT_FIELD_ALIGNED(2 * device_count, uint64_t,
                                iree_alignof(uint64_t), &ready_values)));
  iree_host_size_t model_size = 0, devices_offset = 0, pools_offset = 0;
  iree_host_size_t executions = 0, catalogs = 0, reports = 0;
  iree_host_size_t pool_indices = 0, execution_indices = 0, groups = 0;
  iree_host_size_t residencies = 0, workspace_bytes = 0;
  iree_host_size_t wait_semaphores = 0, wait_values = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_model_t), &model_size,
      IREE_STRUCT_FIELD(device_count, loom_serve_device_t*, &devices_offset),
      IREE_STRUCT_FIELD(device_count, loom_serve_pool_t*, &pools_offset),
      IREE_STRUCT_FIELD(device_count, loom_serve_execution_t*, &executions),
      IREE_STRUCT_FIELD(device_count, loom_serve_jit_t*, &catalogs),
      IREE_STRUCT_FIELD_ALIGNED(device_count, loom_serve_compilation_report_t,
                                iree_alignof(loom_serve_compilation_report_t),
                                &reports),
      IREE_STRUCT_FIELD(device_count, uint32_t, &pool_indices),
      IREE_STRUCT_FIELD(device_count, uint32_t, &execution_indices),
      IREE_STRUCT_FIELD(device_count, loom_serve_residency_group_t*, &groups),
      IREE_STRUCT_FIELD(device_count, loom_serve_residency_t*, &residencies),
      IREE_STRUCT_FIELD_ALIGNED(device_count, uint64_t, iree_alignof(uint64_t),
                                &workspace_bytes),
      IREE_STRUCT_FIELD(2 * device_count, iree_hal_semaphore_t*,
                        &wait_semaphores),
      IREE_STRUCT_FIELD_ALIGNED(2 * device_count, uint64_t,
                                iree_alignof(uint64_t), &wait_values)));
  loom_serve_text_load_t* load = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, load_size, (void**)&load));
  load->host_allocator = host_allocator;
  load->options = *options;
  load->event.wake = wake;
  load->event.user_data = user_data;
  load->readiness.semaphores =
      (iree_async_semaphore_t**)((uint8_t*)load + ready_semaphores);
  load->readiness.values = (uint64_t*)((uint8_t*)load + ready_values);
  iree_slim_mutex_initialize(&load->event.mutex);
  loomc_task_initialize(&text_load_vtable, &load->task);
  loomc_task_initialize(&text_load_control_vtable, &load->control.task);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, model_size, (void**)&load->model);
  if (iree_status_is_ok(status)) {
    loom_serve_text_model_t* model = load->model;
    model->host_allocator = host_allocator;
    model->device_count = (uint32_t)device_count;
    uint8_t* base = (uint8_t*)model;
    model->devices = (loom_serve_device_t**)(base + devices_offset);
    model->pools = (loom_serve_pool_t**)(base + pools_offset);
    model->executions = (loom_serve_execution_t**)(base + executions);
    model->compiler.catalogs = (loom_serve_jit_t**)(base + catalogs);
    model->compiler.reports =
        (loom_serve_compilation_report_t*)(base + reports);
    model->residency.pool_indices = (uint32_t*)(base + pool_indices);
    model->residency.execution_indices = (uint32_t*)(base + execution_indices);
    model->residency.groups = (loom_serve_residency_group_t**)(base + groups);
    model->residency.values = (loom_serve_residency_t**)(base + residencies);
    model->residency.workspace_bytes = (uint64_t*)(base + workspace_bytes);
    model->dependencies.semaphores =
        (iree_hal_semaphore_t**)(base + wait_semaphores);
    model->dependencies.values = (uint64_t*)(base + wait_values);
    for (uint32_t i = 0; i < device_count; ++i) {
      model->devices[i] = devices[i];
      model->pools[i] = pools[i];
      uint32_t pool_index = 0;
      while (pool_index < model->residency.count &&
             pools[model->residency.execution_indices[pool_index]] !=
                 pools[i]) {
        ++pool_index;
      }
      if (pool_index == model->residency.count) {
        model->residency.execution_indices[model->residency.count++] = i;
      }
      model->residency.pool_indices[i] = pool_index;
    }
    model->transfer = loom_serve_device_transfer_queue(devices[0]);
    model->row_count = options->row_count;
    model->draft_depth = options->draft_depth;
    model->command_mode =
        retain_profile_metadata
            ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
            : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
    loom_serve_text_state_initialize(options, host_allocator, &model->state);
    *out_load = load;
  } else {
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(host_allocator, load);
  }
  return status;
}

iree_status_t loom_serve_text_load_advance(loom_serve_text_load_t* load,
                                           loom_serve_text_load_flags_t flags,
                                           bool* out_complete) {
  *out_complete = false;
  load->cancelled |= iree_any_bit_set(flags, LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL);
  iree_slim_mutex_lock(&load->event.mutex);
  if (load->event.pending) {
    iree_slim_mutex_unlock(&load->event.mutex);
    return iree_ok_status();
  }
  iree_status_t status =
      iree_status_join(load->terminal_status, load->event.status);
  load->terminal_status = iree_ok_status();
  load->event.status = iree_ok_status();
  const bool layouts_ready =
      load->event.layout_count == load->model->stage_count;
  const bool stages_compiling = load->event.stages_compiling;
  const bool control_compiling = load->event.control_compiling;
  iree_slim_mutex_unlock(&load->event.mutex);
  if (iree_status_is_ok(status) && load->cancelled) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "model load cancelled");
  }
  loom_serve_text_model_t* model = load->model;
  loom_serve_host_t* host = loom_serve_device_host(model->devices[0]);
  bool submitted = false;
  while (iree_status_is_ok(status) && !submitted &&
         load->phase != TEXT_LOAD_COMPLETE) {
    switch (load->phase) {
      case TEXT_LOAD_PREPARE:
        load->phase = TEXT_LOAD_COMPILE;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = 1;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = iree_status_from_loomc(loomc_task_sink_submit(
            loom_serve_host_compiler_sink(host), &load->task));
        submitted = iree_status_is_ok(status);
        break;
      case TEXT_LOAD_COMPILE:
        load->phase = TEXT_LOAD_BIND;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.control_compiling = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = iree_status_from_loomc(loomc_task_sink_submit(
            loom_serve_host_compiler_sink(host), &load->control.task));
        if (!iree_status_is_ok(status)) {
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.control_compiling = false;
          iree_slim_mutex_unlock(&load->event.mutex);
          break;
        }
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.stages_compiling = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = loom_serve_stage_set_compile_catalog_async(
            model->compiler.catalogs, model->source.preparation,
            loom_serve_text_load_layout, loom_serve_text_load_compiled, load,
            load->host_allocator);
        submitted = iree_status_is_ok(status);
        if (!submitted) {
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.stages_compiling = false;
          iree_slim_mutex_unlock(&load->event.mutex);
        }
        break;
      case TEXT_LOAD_BIND:
        if (!layouts_ready) {
          return iree_ok_status();
        }
        status = loom_serve_text_load_bind(model);
        load->phase = TEXT_LOAD_WEIGHTS;
        break;
      case TEXT_LOAD_WEIGHTS:
        load->phase = TEXT_LOAD_RESIDENCY;
        loom_serve_text_load_prepare_weights(load);
        submitted = true;
        break;
      case TEXT_LOAD_RESIDENCY:
        // Source preparation defines captured-input lane storage. Register
        // the complete cold working set before optional activation, while
        // device compilation and tokenization can still be running.
        for (uint32_t i = 0;
             i < model->residency.count && iree_status_is_ok(status); ++i) {
          const uint32_t execution = model->residency.execution_indices[i];
          status = loom_serve_residency_allocate(
              loom_serve_pool_residency_cache(model->pools[execution]),
              model->executions[execution], 1, &model->residency.groups[i],
              model->host_allocator, &model->residency.values[i]);
        }
        load->phase = TEXT_LOAD_ACTIVATE;
        break;
      case TEXT_LOAD_ACTIVATE:
        if (iree_any_bit_set(flags, LOOM_SERVE_TEXT_LOAD_FLAG_DEMAND)) {
          status = loom_serve_text_load_activate(load);
        }
        load->phase = TEXT_LOAD_RECORD;
        break;
      case TEXT_LOAD_RECORD:
        if (stages_compiling) {
          return iree_ok_status();
        }
        load->phase = TEXT_LOAD_LINK;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = 1;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = loom_serve_text_stages_record_async(
            loom_serve_host_preparation_sink(host),
            loomc_task_pool_worker_count(loom_serve_host_task_pool(host)),
            model->command_mode, model->stage_count, model->stages,
            loom_serve_text_load_complete, load, load->host_allocator);
        submitted = iree_status_is_ok(status);
        break;
      case TEXT_LOAD_LINK:
        if (control_compiling) {
          return iree_ok_status();
        }
        if (!loom_serve_tokenizer_load_take(&load->tokenizer_load, &status,
                                            &model->source.tokenizer)) {
          return iree_ok_status();
        }
        if (iree_status_is_ok(status) &&
            model->source.mode == LOOM_SERVE_TEXT_MODE_GENERATE) {
          status = loom_serve_text_source_resolve_tokenizer(&model->source);
        }
        if (iree_status_is_ok(status)) {
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.pending = 1;
          iree_slim_mutex_unlock(&load->event.mutex);
          status = loom_serve_text_load_link(load);
          submitted = iree_status_is_ok(status);
          load->phase = TEXT_LOAD_READY;
        }
        break;
      case TEXT_LOAD_READY:
        loom_serve_text_storage_release_payloads(&model->state.storage);
        load->phase = TEXT_LOAD_COMPLETE;
        break;
      case TEXT_LOAD_COMPLETE:
        break;
    }
  }
  if (!submitted) {
    // Rejected foreground work owes no callback. Independent compilation still
    // owns source, layout, and model borrows until every producer retires.
    iree_slim_mutex_lock(&load->event.mutex);
    load->event.pending = 0;
    const bool compilation_pending =
        load->event.stages_compiling || load->event.control_compiling;
    iree_slim_mutex_unlock(&load->event.mutex);
    load->phase = TEXT_LOAD_COMPLETE;
    if (compilation_pending) {
      load->terminal_status = status;
      return iree_ok_status();
    }
    if (!loom_serve_tokenizer_load_take(&load->tokenizer_load, &status,
                                        &model->source.tokenizer)) {
      load->terminal_status = status;
      return iree_ok_status();
    }
  }
  if (!submitted) {
    loom_serve_text_warmup_poll(model->warmup);
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_warmup_status(model->warmup);
    }
  }
  if (!iree_status_is_ok(status)) {
    // Binding may already have queued backing allocation. Failure of a later
    // preparation step does not retire that accepted ownership.
    loom_serve_text_model_retire(model, load->event.wake,
                                 load->event.user_data);
    if (!loom_serve_text_model_is_retired(model)) {
      load->terminal_status = status;
      return iree_ok_status();
    }
  }
  *out_complete = load->phase == TEXT_LOAD_COMPLETE;
  if (*out_complete) {
    load->compilation = model->compiler.reports[0];
    for (uint32_t i = 1; i < model->device_count; ++i) {
      loom_serve_compilation_merge(&load->compilation,
                                   &model->compiler.reports[i]);
    }
    // All preparation producers have retired and returned their scratch.
    // Reclaim the burst's idle storage before entering steady serving.
    loom_serve_host_trim_compiler(host);
  }
  return status;
}

const loom_serve_compilation_report_t* loom_serve_text_load_report(
    const loom_serve_text_load_t* load) {
  return &load->compilation;
}

loom_serve_text_model_t* loom_serve_text_load_take_model(
    loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  loom_serve_text_warmup_set_observer(model->warmup, NULL, NULL);
  load->model = NULL;
  return model;
}

void loom_serve_text_load_free(loom_serve_text_load_t* load) {
  if (!load) {
    return;
  }
  IREE_ASSERT(!load->model);
  iree_slim_mutex_deinitialize(&load->event.mutex);
  iree_allocator_free(load->host_allocator, load);
}

static void loom_serve_text_load_notify(void* user_data) {
  iree_notification_post((iree_notification_t*)user_data, IREE_ALL_WAITERS);
}

typedef struct loom_serve_text_load_wait_t {
  // Borrowed cold load, exclusively advanced by this outer owner.
  loom_serve_text_load_t* load;
  // Owned load result, joined with allocation progress before returning.
  iree_status_t status;
} loom_serve_text_load_wait_t;

static bool loom_serve_text_load_wait_complete(void* user_data) {
  loom_serve_text_load_wait_t* wait = user_data;
  bool complete = false;
  wait->status = loom_serve_text_load_advance(wait->load, 0, &complete);
  return complete || !iree_status_is_ok(wait->status);
}

iree_status_t loom_serve_text_model_allocate(
    iree_host_size_t device_count, loom_serve_device_t* const* devices,
    loom_serve_pool_t* const* pools, const loom_serve_text_options_t* options,
    iree_allocator_t host_allocator, loom_serve_text_model_t** out_model) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_load_t* load = NULL;
  iree_status_t status = loom_serve_text_load_allocate(
      device_count, devices, pools, options, loom_serve_text_load_notify,
      &notification, host_allocator, &load);
  if (iree_status_is_ok(status)) {
    loom_serve_text_load_wait_t wait = {.load = load};
    loom_serve_workspace_t* workspaces[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
    for (uint32_t i = 0; i < load->model->residency.count; ++i) {
      workspaces[i] = loom_serve_pool_workspace(
          pools[load->model->residency.execution_indices[i]]);
    }
    status = loom_serve_workspace_await(
        load->model->residency.count, workspaces, &notification,
        loom_serve_text_load_wait_complete, &wait);
    status = iree_status_join(status, wait.status);
  }
  if (load) {
    loom_serve_text_model_t* model = loom_serve_text_load_take_model(load);
    loom_serve_text_load_free(load);
    if (iree_status_is_ok(status)) {
      *out_model = model;
    } else {
      status = iree_status_join(status, loom_serve_text_model_free(model));
    }
  }
  iree_notification_deinitialize(&notification);
  return status;
}
