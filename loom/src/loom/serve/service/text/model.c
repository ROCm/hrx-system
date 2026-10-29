// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared source-defined text model residency and retained row state.

#include "loom/serve/service/text/model.h"

#include <stdlib.h>
#include <string.h>

#include "iree/async/operations/semaphore.h"
#include "iree/tooling/device_profiling.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/stage_set.h"
#include "loom/serve/runtime/program/tokenizer_load.h"
#include "loom/serve/service/text/control.h"
#include "loom/serve/service/text/epoch_plan.h"
#include "loom/serve/service/text/epoch_upload.h"
#include "loom/serve/service/text/media_epoch.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/source.h"
#include "loom/serve/service/text/stages.h"
#include "loom/serve/service/text/state.h"
#include "loom/serve/service/text/storage.h"
#include "loom/serve/service/text/variant.h"
#include "loomc/iree.h"

typedef enum loom_serve_text_load_phase_e {
  TEXT_LOAD_PREPARE,
  TEXT_LOAD_STAGES,
  TEXT_LOAD_BIND,
  TEXT_LOAD_WEIGHTS,
  TEXT_LOAD_RESIDENCY,
  TEXT_LOAD_ACTIVATE,
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
  // Borrowed source paths and shapes, retained through terminal advance.
  loom_serve_text_options_t options;
  // Serialized owner-only phase; callbacks publish results, never transition.
  loom_serve_text_load_phase_t phase;
  // Once cancelled, subsequent advance calls cannot restart preparation.
  bool cancelled;
  // Private task result transferred only by the task's final release.
  iree_status_t task_status;
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
    bool compiling;
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
    iree_async_semaphore_t* semaphores[2];
    // Captured initialization dependency frontier.
    loom_serve_execution_frontier_t frontier;
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
  loom_serve_host_t* host = loom_serve_device_host(model->device_owner);
  const iree_host_size_t count = model->parameters.count;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.pending = count + 1;
  iree_slim_mutex_unlock(&load->event.mutex);
  iree_status_t status = iree_ok_status();
  iree_host_size_t accepted = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_serve_weights_prepare_async(
        model->parameters.plans[i], host, model->jit, model->command_mode,
        &load->options.sources, model->parameters.sources[i],
        model->source.parameter_sources, loom_serve_text_load_complete, load);
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
        loom_serve_device_tokenizer_cache(model->device_owner),
        loom_serve_host_compiler_sink(
            loom_serve_device_host(model->device_owner)),
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
  IREE_RETURN_IF_ERROR(loom_serve_device_create_jit(
      model->device_owner, &options->sources, &options->kernel_sanitizer,
      &model->compilation, &model->jit));
  IREE_RETURN_IF_ERROR(loom_serve_text_source_initialize(
      model->jit, options, model->host_allocator, &model->source,
      &model->state.storage));
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
  load->event.compiling = false;
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
  IREE_RETURN_IF_ERROR(loom_serve_device_create_execution(
      model->device_owner, model->pool, &model->execution));
  IREE_RETURN_IF_ERROR(loom_serve_text_stages_bind(
      model->device_owner, model->pool, model->source.preparation,
      model->source.shared_stage_count, model->stage_count, model->stages,
      &model->parameters, model->host_allocator));
  return loom_serve_residency_group_allocate(
      loom_serve_pool_residency_cache(model->pool), model->parameters.count,
      model->parameters.plans, model->host_allocator, &model->parameter_group);
}

static iree_status_t loom_serve_text_load_activate(
    loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  IREE_RETURN_IF_ERROR(loom_serve_text_warmup_allocate(
      model->device_owner, model->pool, model->parameter_group,
      load->event.wake, load->event.user_data, load->host_allocator,
      &model->warmup));
  loom_serve_text_warmup_poll(model->warmup);
  return loom_serve_text_warmup_status(model->warmup);
}

static iree_status_t loom_serve_text_load_link(loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  iree_arena_initialize(
      loom_serve_host_block_pool(loom_serve_device_host(model->device_owner)),
      &model->row_records);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < model->stage_count && iree_status_is_ok(status); ++i) {
    loom_serve_text_stage_t* stage = &model->stages[i];
    status = loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(model->dispatch),
        model->command_mode, NULL, stage->fixed_buffers, &stage->command);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_state_allocate(
      &model->state, model->device_owner, model->pool, model->execution));
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
    model->stage_bindings[i] = loom_serve_text_stage_binding(&model->stages[i]);
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_control_initialize(
      &model->source, model->execution, model->transfer, &model->state,
      model->shape_count, model->shapes, model->stage_count,
      model->stage_bindings, model->host_allocator, &model->control));
  model->stage_workspace_bytes =
      loom_serve_module_workspace_requirement(model->control.native_module)
          .allocation_size;
  IREE_RETURN_IF_ERROR(loom_serve_residency_prepare_working_set(
      model->residency, model->stage_workspace_bytes, 0));
  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(
          loom_serve_device_host(model->device_owner)),
      0, &proactor));
  load->readiness.frontier = loom_serve_execution_frontier(model->execution);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    load->readiness.semaphores[i] =
        (iree_async_semaphore_t*)load->readiness.frontier.semaphores[i];
  }
  iree_async_semaphore_wait_operation_t* wait = &load->readiness.operation;
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, loom_serve_text_load_ready, load);
  wait->semaphores = load->readiness.semaphores;
  wait->values = load->readiness.frontier.values;
  wait->count = 2;
  wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
  return iree_async_proactor_submit_one(proactor, &wait->base);
}

iree_status_t loom_serve_text_load_allocate(
    loom_serve_device_t* device, loom_serve_pool_t* pool,
    const loom_serve_text_options_t* options,
    loom_serve_text_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_text_load_t** out_load) {
  *out_load = NULL;
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
  if (session_capacity >= INT32_MAX || prefix_capacity > INT32_MAX ||
      options->checkpoint_capacity > INT32_MAX ||
      (uint64_t)options->checkpoint_capacity + prefix_capacity +
              (session_capacity ? session_capacity + 1 : 0) >
          INT32_MAX ||
      ((session_capacity || prefix_capacity) && !options->pool_capacity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cache endpoints require pooled storage and a "
                            "combined capacity fitting i32");
  }
  if (options->prefill_capacity < 1 || options->prefill_capacity > INT32_MAX ||
      options->context_capacity < 1 || options->context_capacity > 262144 ||
      options->epoch_count > INT32_MAX ||
      options->epoch_count > SIZE_MAX / (2 * sizeof(int64_t)) ||
      options->draft_depth > 7 ||
      (options->draft_depth && !options->epoch_count) ||
      (uint64_t)options->pool_capacity > INT64_MAX ||
      options->checkpoint_capacity > INT32_MAX ||
      (options->checkpoint_capacity && !options->pool_capacity) ||
      (options->pool_capacity && !options->epoch_count)) {
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
  loom_serve_text_load_t* load = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*load), (void**)&load));
  load->host_allocator = host_allocator;
  load->options = *options;
  load->event.wake = wake;
  load->event.user_data = user_data;
  iree_slim_mutex_initialize(&load->event.mutex);
  loomc_task_initialize(&text_load_vtable, &load->task);
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*load->model), (void**)&load->model);
  if (iree_status_is_ok(status)) {
    loom_serve_text_model_t* model = load->model;
    model->host_allocator = host_allocator;
    model->device_owner = device;
    model->pool = pool;
    model->device = loom_serve_device_handle(device);
    model->group = loom_serve_device_group(device);
    model->dispatch = loom_serve_device_dispatch_queue(device);
    model->transfer = loom_serve_device_transfer_queue(device);
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
  const bool compiling = load->event.compiling;
  iree_slim_mutex_unlock(&load->event.mutex);
  if (iree_status_is_ok(status) && load->cancelled) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "model load cancelled");
  }
  loom_serve_text_model_t* model = load->model;
  loom_serve_host_t* host = loom_serve_device_host(model->device_owner);
  bool submitted = false;
  while (iree_status_is_ok(status) && !submitted &&
         load->phase != TEXT_LOAD_COMPLETE) {
    switch (load->phase) {
      case TEXT_LOAD_PREPARE:
        load->phase = TEXT_LOAD_STAGES;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = 1;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = iree_status_from_loomc(loomc_task_sink_submit(
            loom_serve_host_compiler_sink(host), &load->task));
        submitted = iree_status_is_ok(status);
        break;
      case TEXT_LOAD_STAGES:
        load->phase = TEXT_LOAD_BIND;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.compiling = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = loom_serve_stage_set_compile_catalog_async(
            model->jit, model->source.preparation, loom_serve_text_load_layout,
            loom_serve_text_load_compiled, load, load->host_allocator);
        submitted = iree_status_is_ok(status);
        if (!submitted) {
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.compiling = false;
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
        status = loom_serve_residency_allocate(
            loom_serve_pool_residency_cache(model->pool), model->execution, 1,
            &model->parameter_group, model->host_allocator, &model->residency);
        load->phase = TEXT_LOAD_ACTIVATE;
        break;
      case TEXT_LOAD_ACTIVATE:
        if (iree_any_bit_set(flags, LOOM_SERVE_TEXT_LOAD_FLAG_DEMAND)) {
          status = loom_serve_text_load_activate(load);
        }
        load->phase = TEXT_LOAD_LINK;
        break;
      case TEXT_LOAD_LINK:
        if (compiling) {
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
    // Rejected foreground work owes no callback. Independent stage work still
    // owns layout bytes and model borrows until its terminal publication.
    iree_slim_mutex_lock(&load->event.mutex);
    load->event.pending = 0;
    const bool stages_pending = load->event.compiling;
    iree_slim_mutex_unlock(&load->event.mutex);
    load->phase = TEXT_LOAD_COMPLETE;
    if (stages_pending) {
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
  return status;
}

const loom_serve_compilation_report_t* loom_serve_text_load_report(
    const loom_serve_text_load_t* load) {
  return &load->model->compilation;
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

iree_status_t loom_serve_text_model_allocate(
    loom_serve_device_t* device, loom_serve_pool_t* pool,
    const loom_serve_text_options_t* options, iree_allocator_t host_allocator,
    loom_serve_text_model_t** out_model) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_load_t* load = NULL;
  iree_status_t status = loom_serve_text_load_allocate(
      device, pool, options, loom_serve_text_load_notify, &notification,
      host_allocator, &load);
  bool complete = false;
  while (iree_status_is_ok(status) && !complete) {
    const uint32_t ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_text_load_advance(load, 0, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
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

void loom_serve_text_model_retire(loom_serve_text_model_t* model,
                                  void (*wake)(void* user_data),
                                  void* user_data) {
  loom_serve_text_warmup_set_observer(model->warmup, wake, user_data);
  if (model->execution) {
    loom_serve_execution_set_retirement_observer(model->execution, wake,
                                                 user_data);
  }
  for (iree_host_size_t i = 0; i < model->parameters.count; ++i) {
    loom_serve_weights_set_retirement_observer(model->parameters.plans[i], wake,
                                               user_data);
  }
  loom_serve_text_control_unlink(&model->control);
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    iree_hal_command_buffer_release(model->stages[i].command);
    model->stages[i].command = NULL;
    for (iree_host_size_t j = 0; j < model->stages[i].fixed_buffer_count; ++j) {
      iree_hal_buffer_release(model->stages[i].fixed_buffers[j]);
      model->stages[i].fixed_buffers[j] = NULL;
    }
  }
  loom_serve_text_state_retire(&model->state, wake, user_data);
}

bool loom_serve_text_model_is_retired(loom_serve_text_model_t* model) {
  if (!model) {
    return true;
  }
  loom_serve_text_warmup_poll(model->warmup);
  bool retired =
      !loom_serve_text_warmup_is_pending(model->warmup) &&
      (!model->execution || loom_serve_execution_is_retired(model->execution));
  for (iree_host_size_t i = 0; i < model->parameters.count; ++i) {
    retired &= loom_serve_weights_is_retired(model->parameters.plans[i]);
  }
  return retired && loom_serve_text_state_is_retired(&model->state);
}

iree_status_t loom_serve_text_model_free_retired(
    loom_serve_text_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  IREE_ASSERT(loom_serve_text_model_is_retired(model));
  iree_status_t warmup_status = loom_serve_text_warmup_free(model->warmup);
  model->warmup = NULL;
  loom_serve_residency_free(model->residency);
  model->residency = NULL;
  loom_serve_residency_group_free(model->parameter_group);
  model->parameter_group = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < model->parameters.count && iree_status_is_ok(status); ++i) {
    status = loom_serve_weights_free(model->parameters.plans[i]);
    if (iree_status_is_ok(status)) {
      model->parameters.plans[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_state_deinitialize(&model->state);
  }
  if (!iree_status_is_ok(status)) {
    return iree_status_join(status, warmup_status);
  }
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    loom_serve_jit_stage_free(model->stages[i].compiled);
    iree_allocator_free(model->host_allocator, model->stages[i].fixed_buffers);
  }
  iree_allocator_free(model->host_allocator, model->shapes);
  iree_allocator_free(model->host_allocator, model->stages);
  iree_allocator_free(model->host_allocator, model->stage_bindings);
  iree_allocator_free(model->host_allocator, model->parameters.plans);
  iree_allocator_free(model->host_allocator, model->parameters.sources);
  loom_serve_jit_release(model->jit);
  loom_serve_text_media_epoch_deinitialize(&model->invocation.media);
  loom_serve_text_control_deinitialize(&model->control);
  iree_allocator_free(model->host_allocator, model->rows);
  iree_arena_deinitialize(&model->row_records);
  iree_allocator_free(model->host_allocator, model->invocation.row_indices);
  iree_allocator_free(model->host_allocator, model->checkpoints);
  loom_serve_execution_release(model->execution);
  loom_serve_text_source_deinitialize(&model->source);
  iree_allocator_free(model->host_allocator, model);
  return warmup_status;
}

iree_status_t loom_serve_text_model_free(loom_serve_text_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_text_model_retire(model, NULL, NULL);
  loom_serve_text_warmup_await(model->warmup);
  iree_status_t status = model->execution
                             ? loom_serve_execution_drain(model->execution)
                             : iree_ok_status();
  for (iree_host_size_t i = 0; i < model->parameters.count; ++i) {
    status = iree_status_join(status, loom_serve_weights_await_retirement(
                                          model->parameters.plans[i]));
  }
  loom_serve_text_state_await_retirement(&model->state);
  return iree_status_join(status, loom_serve_text_model_free_retired(model));
}

loom_serve_device_t* loom_serve_text_model_device(
    const loom_serve_text_model_t* model) {
  return model->device_owner;
}

loom_serve_pool_t* loom_serve_text_model_pool(
    const loom_serve_text_model_t* model) {
  return model->pool;
}

iree_status_t loom_serve_text_model_compile_variant_async(
    loom_serve_text_model_t* model, const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_variant_completion_fn_t completion, void* user_data) {
  return loom_serve_text_variant_compile_async(
      model, model->device_owner, model->source.preparation,
      model->source.shared_stage_count, model->stage_count, model->stages,
      model->command_mode, sources, sanitizer, completion, user_data,
      model->host_allocator);
}

iree_status_t loom_serve_text_model_select_variant(
    loom_serve_text_model_t* model, const loom_serve_text_variant_t* variant) {
  if (variant) {
    return loom_serve_text_variant_bind(
        variant, model, model->control.native_module, model->residency);
  }
  IREE_RETURN_IF_ERROR(loom_serve_residency_check_private_peak(
      model->residency, model->stage_workspace_bytes));
  IREE_RETURN_IF_ERROR(loom_serve_module_bind_stages(
      model->control.native_module, model->stage_count, model->stage_bindings));
  loom_serve_residency_set_private_peak(model->residency,
                                        model->stage_workspace_bytes);
  return iree_ok_status();
}

loom_serve_text_row_t* loom_serve_text_model_row(loom_serve_text_model_t* model,
                                                 iree_host_size_t index) {
  return model->rows[index];
}

loom_serve_memory_statistics_t loom_serve_text_model_memory_statistics(
    const loom_serve_text_model_t* model) {
  return model->state.memory.statistics;
}

loom_serve_memory_usage_t loom_serve_text_model_weight_statistics(
    const loom_serve_text_model_t* model) {
  return loom_serve_residency_group_statistics(model->parameter_group);
}

loom_serve_weight_activity_t loom_serve_text_model_weight_activity(
    const loom_serve_text_model_t* model) {
  return loom_serve_residency_group_activity(model->parameter_group);
}

loom_serve_residency_t* loom_serve_text_model_residency(
    const loom_serve_text_model_t* model) {
  return model->residency;
}

loom_serve_text_warmup_t* loom_serve_text_model_warmup(
    const loom_serve_text_model_t* model) {
  return model->warmup;
}

iree_status_t loom_serve_text_model_activate(loom_serve_text_model_t* model) {
  return loom_serve_residency_activate(model->residency);
}

iree_status_t loom_serve_text_model_deactivate(loom_serve_text_model_t* model,
                                               bool* out_complete) {
  return loom_serve_residency_deactivate(model->residency, out_complete);
}

const iree_tokenizer_t* loom_serve_text_model_tokenizer(
    loom_serve_text_model_t* model) {
  return loom_serve_tokenizer_value(model->source.tokenizer);
}

loom_serve_text_mode_t loom_serve_text_model_mode(
    const loom_serve_text_model_t* model) {
  return model->source.mode;
}

iree_status_t loom_serve_text_model_request_options(
    loom_serve_text_model_t* model, iree_string_view_t body,
    uint64_t request_id, iree_vm_buffer_t** out_options) {
  return loom_serve_text_control_request_options(
      &model->control, &model->source, body, request_id, model->host_allocator,
      out_options);
}

const loom_serve_text_chat_policy_t* loom_serve_text_model_chat_policy(
    const loom_serve_text_model_t* model) {
  return &model->control.chat_policy;
}

iree_host_size_t loom_serve_text_model_context_capacity(
    const loom_serve_text_model_t* model) {
  return model->context_capacity;
}

iree_host_size_t loom_serve_text_model_draft_depth(
    const loom_serve_text_model_t* model) {
  return model->draft_depth;
}

iree_host_size_t loom_serve_text_model_checkpoint_capacity(
    const loom_serve_text_model_t* model) {
  return model->state.checkpoints.capacity;
}

loom_serve_text_cache_selection_t loom_serve_text_model_cache_selection(
    const loom_serve_text_model_t* model) {
  return model->source.cache;
}

loom_serve_text_pool_usage_t loom_serve_text_model_pool_usage(
    const loom_serve_text_model_t* model) {
  loom_serve_text_pool_usage_t usage = {
      .block_size = model->state.storage.block_size,
      .capacity = model->state.cache.capacity,
      .available = (iree_host_size_t)model->state.cache.pool.available *
                   model->state.storage.block_size,
  };
  for (iree_host_size_t i = 0; usage.capacity && i < model->row_count; ++i) {
    const loom_serve_text_state_row_t* row = model->state.rows[i];
    if (row->reserved_extent) {
      const iree_host_size_t live =
          iree_host_size_ceil_div(row->position, usage.block_size);
      usage.pending += (row->block_count - live) * usage.block_size;
    }
  }
  return usage;
}

iree_host_size_t loom_serve_text_model_prefill_capacity(
    const loom_serve_text_model_t* model) {
  return model->source.prefill_capacity;
}

iree_host_size_t loom_serve_text_model_shape_count(
    const loom_serve_text_model_t* model) {
  return model->shape_count;
}

const loom_serve_packing_shape_t* loom_serve_text_model_shapes(
    const loom_serve_text_model_t* model) {
  return model->shapes;
}

iree_status_t loom_serve_text_model_submit_trim(
    loom_serve_text_model_t* model, loom_serve_text_trim_result_t* out_result,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_submit_trim(&model->state, out_result));
  *out_frontier = loom_serve_execution_frontier(model->execution);
  return iree_ok_status();
}

iree_status_t loom_serve_text_model_complete_trim(
    loom_serve_text_model_t* model, loom_serve_text_trim_result_t* result,
    bool* out_complete, loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_complete_trim(&model->state, result, out_complete));
  *out_frontier = loom_serve_execution_frontier(model->execution);
  return iree_ok_status();
}

iree_status_t loom_serve_text_model_trim(
    loom_serve_text_model_t* model, loom_serve_text_trim_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(model->execution));
  loom_serve_execution_frontier_t frontier;
  iree_status_t status =
      loom_serve_text_model_submit_trim(model, out_result, &frontier);
  bool complete = false;
  do {
    status =
        iree_status_join(status, loom_serve_execution_drain(model->execution));
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_model_complete_trim(model, out_result, &complete,
                                                   &frontier);
    }
  } while (iree_status_is_ok(status) && !complete);
  return status;
}

static iree_status_t loom_serve_text_invoke(loom_serve_text_model_t* model,
                                            iree_vm_function_t function,
                                            iree_vm_variant_span_t arguments,
                                            iree_vm_variant_span_t results) {
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                     function, arguments, results);
  iree_vm_variant_span_reset(arguments);
  return status;
}

static iree_status_t loom_serve_text_submit_step(
    loom_serve_text_row_t* row, int32_t initialize,
    loom_serve_execution_frontier_t* out_frontier) {
  loom_serve_text_model_t* model = row->model;
  IREE_RETURN_IF_ERROR(loom_serve_residency_acquire(model->residency));
  iree_vm_variant_t arguments[TEXT_BINDING_COUNT + 4] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               model->source.control_state),
      iree_vm_variant_from_i32(initialize),
      iree_vm_variant_from_i32((int32_t)row->state->source_position),
      iree_vm_variant_from_i32((int32_t)row->step.token_count),
  };
  for (iree_host_size_t i = 0; i < TEXT_BINDING_COUNT; ++i) {
    arguments[i + 4] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->source.types, row->state->buffers[i]);
  }
  iree_vm_variant_t results[2] = {0};
  iree_status_t status = loom_serve_text_invoke(
      model, model->control.step, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results));
  int32_t source_position = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[1], &source_position);
  }
  if (iree_status_is_ok(status)) {
    row->step.source_position = (uint32_t)source_position;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  const iree_hal_transfer_operation_t downloads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->state->buffers[TEXT_TOKENS],
                    .target = &row->transfer.token,
                    .length = sizeof(int32_t)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->state->buffers[TEXT_PROGRESS],
                    .target = row->transfer.progress,
                    .length = sizeof(row->transfer.progress)}},
  };
  uint64_t completion = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_feedback(
        model->execution, IREE_ARRAYSIZE(downloads), downloads, &completion);
  }
  if (iree_status_is_ok(status)) {
    *out_frontier = loom_serve_execution_frontier(model->execution);
  }
  return status;
}

// Only ownership changes visit this path. The source encoder owns device
// record offsets; native state contributes its already validated byte ranges.
static iree_status_t loom_serve_text_publish_state_bindings(
    loom_serve_text_model_t* model) {
  if (!model->state.recurrent.dirty_count) {
    return iree_ok_status();
  }
  uint32_t count = 0;
  for (iree_host_size_t i = 0; i < model->row_count; ++i) {
    loom_serve_text_state_row_t* row = model->state.rows[i];
    if (!row->recurrent.dirty) {
      continue;
    }
    row->recurrent.dirty = false;
    const uint64_t origin =
        iree_hal_buffer_byte_offset(row->buffers[TEXT_RECURRENT]);
    const uint64_t read_origin =
        row->recurrent.anchor == UINT32_MAX
            ? origin
            : iree_hal_buffer_byte_offset(
                  model->state.recurrent.buffers[row->recurrent.anchor]);
    uint8_t* record =
        model->control.host.bytes[TEXT_HOST_STATE_BINDINGS].data + count++ * 32;
    iree_unaligned_store_le_u64(record, i);
    iree_unaligned_store_le_u64(record + 8, read_origin);
    iree_unaligned_store_le_u64(record + 16, origin);
    iree_unaligned_store_le_u64(record + 24, row->position);
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_STATE_BINDINGS]),
      iree_vm_variant_from_i32((int32_t)count),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_STATE_ORIGINS]),
  };
  IREE_RETURN_IF_ERROR(
      loom_serve_text_invoke(model, model->control.encode_state,
                             iree_vm_variant_span_from_array(arguments),
                             iree_vm_variant_span_empty()));
  const iree_byte_span_t origins =
      model->control.host.bytes[TEXT_HOST_STATE_ORIGINS];
  if (origins.data_length) {
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = origins.data,
                   .target_buffer = model->state.epoch.buffers[2],
                   .length = origins.data_length},
    };
    uint64_t completion = 0;
    IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(model->execution, 1,
                                                       &upload, &completion));
  }
  model->state.recurrent.dirty_count = 0;
  return iree_ok_status();
}

iree_status_t loom_serve_text_row_bind_request(loom_serve_text_row_t* row,
                                               iree_vm_buffer_t* options) {
  loom_serve_text_model_t* model = row->model;
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               model->source.control_state),
      iree_vm_variant_from_i32((int32_t)row->state->index),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               options)};
  return loom_serve_text_invoke(model, model->control.bind_request,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_empty());
}

iree_status_t loom_serve_text_model_submit_epoch(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_text_span_t* spans,
    const uint32_t* output_limits,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_text_capture_t** out_capture,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  if (out_capture) {
    *out_capture = NULL;
  }
  const bool encodes = model->source.mode == LOOM_SERVE_TEXT_MODE_ENCODE;
  if (encodes != (out_capture != NULL) ||
      (encodes && (output_limits || continuation))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "known-input encoding requires capture ownership "
                            "without speculative continuation");
  }
  if (continuation && !output_limits) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "continuation requires speculative output limits");
  }
  if (output_limits &&
      !iree_any_bit_set(model->state.flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "draft artifacts were not loaded");
  }
  if (shape_index >= model->shape_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "packed text shape index is not loaded");
  }
  const loom_serve_packing_shape_t shape = model->shapes[shape_index];
  if (!span_count || span_count > shape.span_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "epoch span count exceeds stage capacity");
  }
  // Validate the caller's plan before changing reusable payloads or submitting
  // any work. The device consumes this one established partition invariant.
  iree_host_size_t token_count = 0;
  uint32_t* extents = model->invocation.extents;
  uint32_t* next_indices = model->invocation.next_indices;
  uint8_t* continued = model->control.host.bytes[TEXT_HOST_CONTINUATIONS].data;
  memset(continued, 0, span_count * sizeof(uint32_t));
  bool continued_speculative = false;
  const iree_host_size_t epoch_count = continuation ? 2 : 1;
  const iree_host_size_t width = model->draft_depth + 1;
  bool proposes = false;
  bool has_media_input = false;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    if ((encodes && (span->flags || span->media)) ||
        span->row_index >= model->row_count || !span->token_count ||
        !span->token_ids ||
        span->token_count > shape.token_capacity - token_count ||
        (span->flags & ~(LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                         LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid text epoch span %zu", i);
    }
    if (model->rows[span->row_index]->state->snapshot) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "epoch row %zu is suspended", span->row_index);
    }
    const iree_host_size_t prior = model->rows[span->row_index]->planning.first;
    if (prior < i && spans[prior].row_index == span->row_index) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "epoch repeats resident row %zu",
                              span->row_index);
    }
    if (span->token_count > model->context_capacity -
                                model->rows[span->row_index]->state->position) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "epoch span %zu exceeds remaining context", i);
    }
    const bool generates =
        iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE);
    if (generates && span->media) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "speculative text span %zu carries media", i);
    }
    if (generates && (!output_limits || !output_limits[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "proposal requires a speculative text span");
    }
    if (output_limits && output_limits[i]) {
      const loom_serve_text_row_t* row = model->rows[span->row_index];
      if (output_limits[i] > width * epoch_count ||
          span->token_count != width ||
          !iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT) ||
          !row->has_prediction || row->transfer.progress[7] ||
          span->token_ids[0] != row->transfer.token) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid speculative text span %zu", i);
      }
    }
    extents[i] = (uint32_t)span->token_count;
    model->rows[span->row_index]->planning.first = i;
    token_count += span->token_count;
    proposes |= generates;
    has_media_input |= span->media != NULL;
  }

  if (continuation) {
    if (continuation->shape_index >= model->shape_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "continuation shape index is not loaded");
    }
    const loom_serve_packing_shape_t next_shape =
        model->shapes[continuation->shape_index];
    if (!continuation->span_count ||
        continuation->span_count > next_shape.span_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "continuation span count exceeds capacity");
    }
    iree_host_size_t next_tokens = 0;
    for (iree_host_size_t i = 0; i < continuation->span_count; ++i) {
      const loom_serve_text_span_t* span = &continuation->spans[i];
      if (span->row_index >= model->row_count || !span->token_count ||
          span->token_count > next_shape.token_capacity - next_tokens ||
          (span->flags & ~(LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                           LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid continuation span %zu", i);
      }
      loom_serve_text_row_t* resident = model->rows[span->row_index];
      const iree_host_size_t first = resident->planning.first;
      const iree_host_size_t next = resident->planning.next;
      if (first >= span_count || spans[first].row_index != span->row_index ||
          (next < i &&
           continuation->spans[next].row_index == span->row_index)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "continuation must use distinct first-plan rows");
      }
      const loom_serve_text_span_t* prior = &spans[first];
      const uint32_t limit = continuation->output_limits[i];
      const iree_host_size_t remaining =
          model->context_capacity -
          model->rows[span->row_index]->state->position;
      if (limit) {
        if (limit < 3 || limit > 2 * width || span->token_count != width ||
            span->flags != (LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                            LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE) ||
            span->media ||
            !iree_any_bit_set(prior->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT) ||
            (output_limits[first] && limit > output_limits[first])) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid speculative continuation span %zu",
                                  i);
        }
        // Device continuation requires at least two outputs after the first
        // result. Its largest advancing prefix is therefore limit - 2.
        const iree_host_size_t extent =
            output_limits[first] ? iree_min(2 * width, limit + width - 2)
                                 : prior->token_count + width;
        extents[first] = (uint32_t)iree_min(remaining, extent);
        iree_unaligned_store_le_u32(continued + first * sizeof(uint32_t), 1);
        continued_speculative = true;
      } else {
        if (!span->token_ids || output_limits[first] || prior->flags ||
            iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "known continuation requires unselected known input");
        }
        if (span->token_count > remaining - prior->token_count) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "continuation span %zu exceeds context", i);
        }
        extents[first] += (uint32_t)span->token_count;
      }
      next_indices[i] = (uint32_t)first;
      resident->planning.next = i;
      next_tokens += span->token_count;
      has_media_input |= span->media != NULL;
    }
  }

  uint32_t required_blocks = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_state_row_t* row =
        model->rows[spans[i].row_index]->state;
    if (row->reserved_extent &&
        extents[i] > row->reserved_extent - row->position) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "epoch exceeds request completion reservation");
    }
  }
  for (iree_host_size_t i = 0; model->state.cache.capacity && i < span_count;
       ++i) {
    const loom_serve_text_row_t* row = model->rows[spans[i].row_index];
    required_blocks += loom_serve_text_state_row_growth(row->state, extents[i]);
  }
  if (required_blocks > model->state.cache.pool.available) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "epoch needs %u KV blocks; %u are available",
                            required_blocks, model->state.cache.pool.available);
  }
  loom_serve_text_media_epoch_t media = {0};
  if (has_media_input) {
    IREE_RETURN_IF_ERROR(loom_serve_text_media_epoch_initialize(
        span_count, spans, shape.token_capacity,
        continuation ? continuation->span_count : 0,
        continuation ? continuation->spans : NULL,
        continuation ? model->shapes[continuation->shape_index].token_capacity
                     : 0,
        model->host_allocator, &media));
  }
  iree_status_t status = iree_ok_status();
  iree_host_size_t input_count = loom_serve_text_epoch_plan_write(
      model->state.rows, model->control.host.bytes[TEXT_HOST_PLAN],
      model->control.host.bytes[TEXT_HOST_INPUT], span_count, spans,
      output_limits, NULL, 0, 0);
  if (continuation) {
    input_count = loom_serve_text_epoch_plan_write(
        model->state.rows, model->control.host.bytes[TEXT_HOST_PLAN],
        model->control.host.bytes[TEXT_HOST_INPUT], continuation->span_count,
        continuation->spans, continuation->output_limits, next_indices,
        span_count, input_count);
  }
  loom_serve_text_capture_t* capture = NULL;
  if (encodes) {
    status = loom_serve_text_control_capture(
        &model->control, &model->source, shape_index, span_count,
        iree_hal_buffer_byte_length(model->state.residual),
        model->host_allocator, &capture);
  }
  if (iree_status_is_ok(status) && media.plan) {
    status = loom_serve_execution_reserve_workspace(
        model->execution, media.device_length,
        LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_residency_acquire(model->residency);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_media_epoch_deinitialize(&media);
    loom_serve_text_capture_release(capture);
    return status;
  }
  model->invocation.media = media;
  model->invocation.span_count = span_count;
  model->invocation.flags = (output_limits ? TEXT_EPOCH_FLAG_VERIFY : 0) |
                            (continuation ? TEXT_EPOCH_FLAG_CONTINUATION : 0);
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    model->invocation.row_indices[i] = spans[i].row_index;
  }
  status =
      loom_serve_text_state_grow(&model->state, span_count, spans, extents);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_publish_state_bindings(model);
  }

  loom_serve_text_media_epoch_t* invocation_media = &model->invocation.media;
  const bool has_media_packet = invocation_media->plan != NULL;
  iree_vm_buffer_t* media_plan =
      has_media_packet ? invocation_media->plan
                       : model->control.host.buffers[TEXT_HOST_INPUT];
  iree_vm_buffer_t* input_packet =
      has_media_packet ? invocation_media->packet_records
                       : model->control.host.buffers[TEXT_HOST_TOKENS];
  iree_vm_variant_t encode_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               model->source.control_state),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->control.host.buffers[TEXT_HOST_PLAN]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_INPUT]),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               media_plan),
      iree_vm_variant_from_i32((int32_t)span_count),
      iree_vm_variant_from_i32(continuation ? (int32_t)continuation->span_count
                                            : 0),
      iree_vm_variant_from_i32((int32_t)input_count),
      iree_vm_variant_from_i32((int32_t)invocation_media->first_feature_count),
      iree_vm_variant_from_i32((int32_t)invocation_media->next_feature_count),
      iree_vm_variant_from_i32(
          (int32_t)invocation_media->first_packet_capacity),
      iree_vm_variant_from_i32((int32_t)invocation_media->next_packet_capacity),
      iree_vm_variant_from_i32(model->source.eos_token),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_METADATA]),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               input_packet),
  };
  iree_vm_variant_t encode_results[4] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_invoke(
        model, model->control.encode_epoch,
        iree_vm_variant_span_from_array(encode_arguments),
        iree_vm_variant_span_from_array(encode_results));
  }
  int32_t output_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(encode_results[0], &output_count);
  }
  iree_vm_buffer_t* transfer_plan = NULL;
  int64_t transfer_offset = 0;
  int64_t transfer_length = 0;
  iree_const_byte_span_t transfer_ranges = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->source.vm_types, &encode_results[1], &transfer_plan);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(encode_results[2], &transfer_offset);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(encode_results[3], &transfer_length);
  }
  if (iree_status_is_ok(status) && !transfer_plan) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source epoch transfer plan must not be null");
  }
  if (iree_status_is_ok(status) &&
      ((uint64_t)transfer_offset > IREE_HOST_SIZE_MAX ||
       (uint64_t)transfer_length > IREE_HOST_SIZE_MAX)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source epoch transfer view exceeds host extent");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(transfer_plan, transfer_offset,
                                     transfer_length, &transfer_ranges);
  }
  iree_hal_buffer_t* dynamic_input = NULL;
  if (iree_status_is_ok(status) && has_media_packet) {
    const iree_hal_pool_reservation_request_t request = {
        .allocation_size = invocation_media->device_length,
        .min_alignment = LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT,
    };
    status = loom_serve_execution_alloca(model->execution, 1, &request,
                                         &dynamic_input);
  }
  iree_hal_buffer_t* device_input =
      dynamic_input ? dynamic_input : model->state.epoch.buffers[4];
  uint64_t completion = 0;
  if (iree_status_is_ok(status)) {
    const iree_const_byte_span_t packets[] = {
        iree_const_cast_byte_span(
            model->control.host.bytes[TEXT_HOST_METADATA]),
        iree_const_cast_byte_span(
            has_media_packet ? invocation_media->packet_record_bytes
                             : model->control.host.bytes[TEXT_HOST_TOKENS]),
    };
    iree_hal_buffer_t* targets[] = {model->state.epoch.buffers[1],
                                    device_input};
    status = loom_serve_text_epoch_upload(model->execution, transfer_ranges,
                                          packets, targets);
  }
  iree_vm_buffer_release(transfer_plan);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(encode_results));
  if (iree_status_is_ok(status) && has_media_packet) {
    status = loom_serve_text_media_epoch_enqueue(
        invocation_media, model->execution, device_input, &completion);
  }
  iree_hal_buffer_t* buffers[] = {
      model->state.residual,
      model->state.epoch.buffers[1],
      model->state.epoch.buffers[2],
      model->state.row_arena,
      device_input,
      model->state.epoch.buffers[5],
      model->state.carry,
      model->state.draft.committed,
      model->state.draft.results,
      model->state.draft.cache,
      model->state.draft.row_table,
      model->state.draft.next_results,
  };
  iree_vm_variant_t arguments[11 + IREE_ARRAYSIZE(buffers)] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               model->source.control_state),
      iree_vm_variant_from_i32((int32_t)shape_index),
      iree_vm_variant_from_i32(proposes),
      iree_vm_variant_from_i32(output_limits != NULL),
      iree_vm_variant_from_i32((int32_t)span_count),
      iree_vm_variant_from_i32(output_count),
      iree_vm_variant_from_i32((int32_t)epoch_count),
      iree_vm_variant_from_i32(continuation ? (int32_t)continuation->shape_index
                                            : 0),
      iree_vm_variant_from_i32(continued_speculative),
      iree_vm_variant_from_i32((int32_t)invocation_media->first_feature_count),
      iree_vm_variant_from_i32((int32_t)invocation_media->next_feature_count),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    arguments[11 + i] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->source.types, buffers[i]);
  }
  iree_vm_variant_t results[1] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_invoke(model, model->control.epoch_step,
                                    iree_vm_variant_span_from_array(arguments),
                                    iree_vm_variant_span_from_array(results));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status) && capture) {
    status = loom_serve_text_capture_submit(capture, model->execution,
                                            model->state.residual);
  }
  if (iree_status_is_ok(status) && dynamic_input) {
    status = loom_serve_execution_dealloca(model->execution, dynamic_input,
                                           &completion);
  }
  iree_hal_buffer_release(dynamic_input);
  if (iree_status_is_ok(status)) {
    if (out_capture) {
      *out_capture = capture;
      capture = NULL;
    }
    *out_frontier = loom_serve_execution_frontier(model->execution);
  }
  loom_serve_text_capture_release(capture);
  return status;
}

iree_status_t loom_serve_text_model_complete_epoch(
    loom_serve_text_model_t* model, loom_serve_text_result_t* out_results) {
  const loom_serve_text_media_epoch_t* media = &model->invocation.media;
  iree_vm_buffer_t* media_plan =
      media->plan ? media->plan : model->control.host.buffers[TEXT_HOST_INPUT];
  iree_vm_variant_t publish_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->control.host.buffers[TEXT_HOST_PLAN]),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               media_plan),
      iree_vm_variant_from_i32((int32_t)model->invocation.span_count),
      iree_vm_variant_from_i32((int32_t)media->first_feature_count),
      iree_vm_variant_from_i32((int32_t)media->next_feature_count),
      iree_vm_variant_from_i32(
          iree_any_bit_set(model->invocation.flags, TEXT_EPOCH_FLAG_VERIFY)),
      iree_vm_variant_from_i32(iree_any_bit_set(model->invocation.flags,
                                                TEXT_EPOCH_FLAG_CONTINUATION)),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_CONTINUATIONS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_OUTPUTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_RESULTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_NEXT_RESULTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->control.host.buffers[TEXT_HOST_PROGRESS]),
  };
  IREE_RETURN_IF_ERROR(
      loom_serve_text_invoke(model, model->control.publish_epoch,
                             iree_vm_variant_span_from_array(publish_arguments),
                             iree_vm_variant_span_empty()));
  for (iree_host_size_t i = 0; i < model->invocation.span_count; ++i) {
    const uint8_t* record = model->control.host.bytes[TEXT_HOST_PROGRESS].data +
                            i * TEXT_PROGRESS_FIELD_COUNT * sizeof(int32_t);
    loom_serve_text_result_t result;
    uint32_t source_position = 0;
    loom_serve_text_epoch_plan_read_progress(record, &result, &source_position);
    if (out_results) {
      out_results[i] = result;
    }
    loom_serve_text_row_t* row = model->rows[model->invocation.row_indices[i]];
    row->state->position += result.consumed_count;
    row->state->source_position = source_position;
    loom_serve_text_state_row_trim(row->state);
    row->has_prediction = result.output_count != 0;
    if (row->has_prediction) {
      row->transfer.token = result.tokens[result.output_count - 1];
      row->transfer.progress[0] = (int32_t)result.output_count;
      row->transfer.progress[7] =
          row->transfer.token == model->source.eos_token;
    }
  }
  model->invocation.span_count = 0;
  loom_serve_text_media_epoch_deinitialize(&model->invocation.media);
  loom_serve_residency_release(model->residency);
  return iree_ok_status();
}

static iree_status_t loom_serve_text_epoch(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_text_span_t* spans,
    const uint32_t* output_limits,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_text_result_t* out_results) {
  loom_serve_execution_frontier_t frontier;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_submit_epoch(
      model, shape_index, span_count, spans, output_limits, continuation, NULL,
      &frontier));
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(model->execution));
  return loom_serve_text_model_complete_epoch(model, out_results);
}

iree_status_t loom_serve_text_model_epoch(loom_serve_text_model_t* model,
                                          iree_host_size_t shape_index,
                                          iree_host_size_t span_count,
                                          const loom_serve_text_span_t* spans) {
  return loom_serve_text_epoch(model, shape_index, span_count, spans, NULL,
                               NULL, NULL);
}

iree_status_t loom_serve_text_model_verify(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_text_span_t* spans,
    const uint32_t* output_limits,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_text_result_t* out_results) {
  if (!iree_any_bit_set(model->state.flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "draft artifacts were not loaded");
  }
  return loom_serve_text_epoch(model, shape_index, span_count, spans,
                               output_limits, continuation, out_results);
}

iree_status_t loom_serve_text_row_submit_prefill(
    loom_serve_text_row_t* row, iree_host_size_t count,
    const int32_t* token_ids, loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  loom_serve_text_model_t* model = row->model;
  if (!count || count > model->source.prefill_capacity ||
      count > model->context_capacity - row->state->position ||
      (row->state->reserved_extent &&
       count > row->state->reserved_extent - row->state->position)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prefill exceeds stage, context or reservation");
  }
  row->step.kind = TEXT_STEP_PREFILL;
  row->step.token_count = count;
  row->step.start_time = iree_time_now();
  if (iree_any_bit_set(model->state.flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT) ||
      model->state.cache.capacity) {
    iree_host_size_t shape_index = 0;
    // The public prefill capacity is bounded by the largest prepared shape.
    while (model->shapes[shape_index].token_capacity < count) {
      ++shape_index;
    }
    const loom_serve_text_span_t span = {
        row->state->index,
        count,
        token_ids,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
    };
    return loom_serve_text_model_submit_epoch(model, shape_index, 1, &span,
                                              NULL, NULL, NULL, out_frontier);
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_activate(row->state));
  uint8_t* input = model->control.host.bytes[TEXT_HOST_INPUT].data;
  const iree_host_size_t input_length =
      iree_hal_buffer_byte_length(row->state->buffers[TEXT_TOKENS]);
  memset(input, 0, input_length);
  memcpy(input, token_ids, count * sizeof(int32_t));
  row->transfer.control[0] = (int32_t)count;
  row->transfer.control[1] = (int32_t)row->state->position;
  row->transfer.control[2] = model->source.eos_token;
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.control,
                  .target_buffer = row->state->buffers[TEXT_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = input,
                  .target_buffer = row->state->buffers[TEXT_TOKENS],
                  .length = input_length}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_submit_step(row, 1, out_frontier);
  }
  return status;
}

iree_status_t loom_serve_text_row_submit_decode(
    loom_serve_text_row_t* row, loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  if (!row->has_prediction) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decode requires a selected prediction");
  }
  if (row->state->position == row->model->context_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "text context is full");
  }
  if (row->state->reserved_extent &&
      row->state->position == row->state->reserved_extent) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "decode exceeds request completion reservation");
  }
  row->step.kind = TEXT_STEP_DECODE;
  row->step.token_count = 1;
  row->step.start_time = iree_time_now();
  if (iree_any_bit_set(row->model->state.flags,
                       LOOM_SERVE_TEXT_STATE_FLAG_DRAFT) ||
      row->model->state.cache.capacity) {
    const int32_t token = row->transfer.token;
    const loom_serve_text_span_t span = {
        row->state->index,
        1,
        &token,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
    };
    return loom_serve_text_model_submit_epoch(row->model, 0, 1, &span, NULL,
                                              NULL, NULL, out_frontier);
  }
  // The host row owns the input position and prediction regardless of which
  // schedule produced it. The isolated decode command consumes that snapshot.
  row->transfer.control[0] = (int32_t)row->state->position;
  row->transfer.control[1] = row->transfer.progress[0];
  row->transfer.control[2] = row->model->source.eos_token;
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.control,
                  .target_buffer = row->state->buffers[TEXT_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = &row->transfer.token,
                  .target_buffer = row->state->buffers[TEXT_TOKENS],
                  .length = sizeof(int32_t)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      row->model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_submit_step(row, 0, out_frontier);
  }
  return status;
}

iree_status_t loom_serve_text_row_complete_step(loom_serve_text_row_t* row) {
  loom_serve_text_model_t* model = row->model;
  if (iree_any_bit_set(model->state.flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT) ||
      model->state.cache.capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_text_model_complete_epoch(model, NULL));
  } else {
    row->state->position += row->step.token_count;
    row->state->source_position = row->step.source_position;
    row->has_prediction = true;
    loom_serve_residency_release(model->residency);
  }
  const iree_duration_t duration = iree_time_now() - row->step.start_time;
  if (row->step.kind == TEXT_STEP_PREFILL) {
    row->metrics.prefill_tokens += row->step.token_count;
    ++row->metrics.prefill_steps;
    row->metrics.prefill_duration += duration;
  } else {
    ++row->metrics.decode_steps;
    row->metrics.decode_duration += duration;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_text_row_prefill(loom_serve_text_row_t* row,
                                          iree_host_size_t count,
                                          const int32_t* token_ids) {
  loom_serve_execution_frontier_t frontier;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_submit_prefill(row, count, token_ids, &frontier));
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(row->model->execution));
  return loom_serve_text_row_complete_step(row);
}

iree_status_t loom_serve_text_row_decode(loom_serve_text_row_t* row) {
  loom_serve_execution_frontier_t frontier;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_submit_decode(row, &frontier));
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(row->model->execution));
  return loom_serve_text_row_complete_step(row);
}
