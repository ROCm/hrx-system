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
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/runtime/program/command.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/json.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/program.h"
#include "loom/serve/service/text/source.h"
#include "loom/serve/service/text/stages.h"
#include "loom/serve/service/text/state.h"
#include "loom/serve/service/text/storage.h"
#include "loom/serve/service/text/variant.h"
#include "loomc/iree.h"

enum {
  TEXT_ROW_CAPACITY = LOOM_SERVE_TEXT_ROW_CAPACITY,
  TEXT_HOST_PLAN = 0,
  TEXT_HOST_INPUT = 1,
  TEXT_HOST_METADATA = 2,
  TEXT_HOST_TOKENS = 3,
  TEXT_HOST_OUTPUTS = 4,
  TEXT_HOST_RESULTS = 5,
  TEXT_HOST_NEXT_RESULTS = 6,
  TEXT_HOST_PROGRESS = 7,
  TEXT_HOST_STATE_BINDINGS = 8,
  TEXT_HOST_STATE_ORIGINS = 9,
  TEXT_HOST_BUFFER_COUNT = 10,
};

enum {
  TEXT_PLAN_FIELD_LENGTH = 0,
  TEXT_PLAN_FIELD_POSITION = 1,
  TEXT_PLAN_FIELD_SOURCE_POSITION = 2,
  TEXT_PLAN_FIELD_ROW = 3,
  TEXT_PLAN_FIELD_INPUT_BEGIN = 4,
  TEXT_PLAN_FIELD_INPUT_COUNT = 5,
  TEXT_PLAN_FIELD_FLAGS = 6,
  TEXT_PLAN_FIELD_OUTPUT_LIMIT = 7,
  TEXT_PLAN_FIELD_FIRST_SPAN = 8,
  TEXT_PLAN_FIELD_COUNT = 9,
};

enum {
  TEXT_PROGRESS_FIELD_CONSUMED_COUNT = 0,
  TEXT_PROGRESS_FIELD_KNOWN_COUNT = 1,
  TEXT_PROGRESS_FIELD_OUTPUT_COUNT = 2,
  TEXT_PROGRESS_FIELD_VERIFICATION_COUNT = 3,
  TEXT_PROGRESS_FIELD_SOURCE_POSITION = 4,
  TEXT_PROGRESS_FIELD_TOKENS = 5,
  TEXT_PROGRESS_FIELD_COUNT = 21,
};

enum text_epoch_flag_bits_e {
  TEXT_EPOCH_FLAG_VERIFY = 1u << 0,
  TEXT_EPOCH_FLAG_CONTINUATION = 1u << 1,
};
typedef uint32_t text_epoch_flags_t;

typedef enum text_step_kind_e {
  TEXT_STEP_PREFILL,
  TEXT_STEP_DECODE,
} text_step_kind_t;

struct loom_serve_text_model_t {
  // Host allocation policy used for all owned resources.
  iree_allocator_t host_allocator;
  // Source-owned metadata and bytecode, independent of native model bindings.
  loom_serve_text_source_t source;
  // Borrowed shared owner outliving every model resource and host I/O payload.
  loom_serve_device_t* device_owner;
  // GPU device and allocation domain borrowed from device_owner.
  iree_hal_device_t* device;
  // Semaphore namespace borrowed from device_owner.
  iree_hal_device_group_t* group;
  // Borrowed exact dispatch queue, retained by execution.
  iree_hal_queue_t* dispatch;
  // Borrowed exact transfer queue, retained by execution.
  iree_hal_queue_t* transfer;
  // Owned model stage/input timeline with independently retired feedback.
  loom_serve_execution_t* execution;
  // Cold recording policy for every reusable stage in this residency.
  iree_hal_command_buffer_mode_t command_mode;
  // Shared source index and compiler for this model residency.
  loom_serve_jit_t* jit;
  // Checkpoint domains owning stable parameter reservations across all stages.
  loom_serve_text_parameters_t parameters;
  // Shared-device admission registration borrowing parameter plans until
  // teardown.
  loom_serve_residency_t* residency;
  // Source-declared command programs, owned in declaration order.
  loom_serve_text_stage_t* stages;
  // Number of allocated stages, including partially initialized cold entries.
  iree_host_size_t stage_count;
  // Original native bindings borrowing stages, retained for explicit restore.
  loom_serve_stage_t* stage_bindings;
  // Number of target packed shapes, independent of auxiliary draft stages.
  iree_host_size_t shape_count;
  // Immutable prepared shape capacities, owned in epoch option order.
  loom_serve_packing_shape_t* shapes;
  // Context specialization shared by every compiled stage.
  iree_host_size_t context_capacity;
  // Proposed tokens after each known anchor; zero disables drafting.
  iree_host_size_t draft_depth;
  // Number of preallocated rows.
  iree_host_size_t row_count;
  // Fixed row records, owned by this model.
  loom_serve_text_row_t* rows;
  // Fixed semantic endpoint records paired with the state's checkpoint IDs.
  loom_serve_text_checkpoint_t* checkpoints;
  // Retained device state, mappings and final queue ownership.
  loom_serve_text_state_t state;
  // Native stage submission module retained by program.
  iree_vm_module_t* native_module;
  // Shared source-JIT model program; request state is not VM process state.
  loom_serve_program_t* program;
  // Cold-resolved model step function.
  iree_vm_function_t step;
  // Source-owned packed target, feedback and draft submission sequence.
  iree_vm_function_t epoch_step;
  // Source-owned conversion of semantic spans to opaque device uploads.
  iree_vm_function_t encode_epoch;
  // Source-owned conversion of retired feedback to semantic row progress.
  iree_vm_function_t publish_epoch;
  // Source-owned row-table encoding after recurrent slot ownership changes.
  iree_vm_function_t encode_state;
  // Source-owned request policy binding into a retained row.
  iree_vm_function_t bind_request;
  // Chat formatting functions borrowing the same model-wide VM process.
  loom_serve_text_chat_policy_t chat_policy;
  // Cold retained packet and feedback storage, reused by the single owner.
  struct {
    // Owned VM buffers retained through all accepted work and feedback.
    iree_vm_buffer_t* buffers[TEXT_HOST_BUFFER_COUNT];
    // Stable writable mappings, valid for the corresponding buffer lifetime.
    iree_byte_span_t bytes[TEXT_HOST_BUFFER_COUNT];
  } host;
  // Publication facts retained independently of the submitting caller's plan.
  // One invocation owns the reusable host buffers and a residency pin until
  // successful completion; terminal failure retains them for destruction.
  struct {
    // Number of first-plan spans, zero after successful completion.
    iree_host_size_t span_count;
    // Resident row identity in first-plan caller order.
    iree_host_size_t row_indices[TEXT_ROW_CAPACITY];
    // Source publication mode selected by the submitted plan.
    text_epoch_flags_t flags;
    // First-plan span bits with device-fed speculative continuation.
    uint32_t continued_speculative_mask;
  } invocation;
};

struct loom_serve_text_row_t {
  // Borrowed owning model; all row operations use its single invocation.
  loom_serve_text_model_t* model;
  // Borrowed retained row state, owned by the model-wide state arena.
  loom_serve_text_state_row_t* state;
  // Whether the latest completed work selected a token at the current position.
  bool has_prediction;
  // Completed stage counts and end-to-end host durations.
  loom_serve_text_metrics_t metrics;
  // Single-row publication facts, owned until successful complete_step.
  struct {
    // Source command and completed metric category.
    text_step_kind_t kind;
    // Absolute source-owned position to publish after successful readiness.
    uint32_t source_position;
    // Number of consumed inputs published only after readiness.
    iree_host_size_t token_count;
    // Submission start, including activation and feedback retirement.
    iree_time_t start_time;
  } step;
  // Host payloads remain alive until the model drains accepted transfers.
  struct {
    // Active prefill count, absolute base, and EOS ID.
    int32_t control[3];
    // Pending prediction, retained independently of shared input staging.
    int32_t token;
    // Device-generated count at zero and EOS flag at seven.
    int32_t progress[8];
  } transfer;
};

struct loom_serve_text_checkpoint_t {
  // Borrowed model, outliving this handle.
  loom_serve_text_model_t* model;
  // Serialized handle owners; the last release returns the native endpoint.
  iree_host_size_t references;
  // Endpoint storage owned until the last handle release.
  loom_serve_text_state_checkpoint_t* state;
  // Whether the exact endpoint selected an unconsumed prediction.
  bool has_prediction;
  // Pending token; meaningful only when has_prediction is set.
  int32_t token;
};

static iree_status_t loom_serve_text_create_program(
    loom_serve_text_model_t* runner) {
  // Both dense padding and semantic packed inputs belong to the single model
  // invocation. Only the final prediction persists independently in each row.
  iree_host_size_t input_length = 0;
  for (iree_host_size_t i = 0; i < runner->shape_count; ++i) {
    input_length = iree_max(
        input_length, 2 * runner->shapes[i].token_capacity * sizeof(int32_t));
  }
  const bool packed =
      runner->state.cache.capacity ||
      iree_any_bit_set(runner->state.flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT);
  if (packed &&
      runner->source.prefill_capacity > input_length / (2 * sizeof(int32_t))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source prefill capacity exceeds the epoch catalog");
  }
  for (iree_host_size_t i = 0; i < runner->row_count; ++i) {
    const iree_device_size_t length =
        iree_hal_buffer_byte_length(runner->state.rows[i].buffers[TEXT_TOKENS]);
    const iree_host_size_t minimum =
        (packed ? 1 : runner->source.prefill_capacity) * sizeof(int32_t);
    if (length < minimum || length > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source row %zu has an invalid token extent", i);
    }
    if (!packed) {
      input_length = iree_max(input_length, (iree_host_size_t)length);
    }
  }
  const iree_host_size_t lengths[TEXT_HOST_BUFFER_COUNT] = {
      2 * TEXT_ROW_CAPACITY * TEXT_PLAN_FIELD_COUNT * sizeof(int32_t),
      input_length,
      runner->shape_count
          ? iree_hal_buffer_byte_length(runner->state.epoch.buffers[1])
          : 0,
      runner->shape_count
          ? iree_hal_buffer_byte_length(runner->state.epoch.buffers[4])
          : 0,
      runner->shape_count
          ? iree_hal_buffer_byte_length(runner->state.epoch.buffers[5])
          : 0,
      iree_any_bit_set(runner->state.flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)
          ? runner->state.storage.feedback_split
          : 0,
      iree_any_bit_set(runner->state.flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)
          ? iree_hal_buffer_byte_length(runner->state.draft.next_results)
          : 0,
      TEXT_ROW_CAPACITY * TEXT_PROGRESS_FIELD_COUNT * sizeof(int32_t),
      TEXT_ROW_CAPACITY * 4 * sizeof(int64_t),
      runner->shape_count
          ? runner->state.storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS]
                .data_length
          : 0,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < TEXT_HOST_BUFFER_COUNT && iree_status_is_ok(status); ++i) {
    status =
        iree_vm_buffer_create(lengths[i], iree_alignof(int64_t),
                              runner->host_allocator, &runner->host.buffers[i]);
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_map_write(runner->host.buffers[i], 0, lengths[i],
                                        &runner->host.bytes[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (lengths[TEXT_HOST_STATE_ORIGINS]) {
    memcpy(runner->host.bytes[TEXT_HOST_STATE_ORIGINS].data,
           runner->state.storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS]
               .data,
           lengths[TEXT_HOST_STATE_ORIGINS]);
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      runner->host_allocator, runner->stage_count,
      sizeof(*runner->stage_bindings), (void**)&runner->stage_bindings));
  for (iree_host_size_t i = 0; i < runner->stage_count; ++i) {
    runner->stage_bindings[i] =
        loom_serve_text_stage_binding(&runner->stages[i]);
  }
  const iree_byte_span_t feedback[] = {
      runner->host.bytes[TEXT_HOST_OUTPUTS],
      runner->host.bytes[TEXT_HOST_RESULTS],
      runner->host.bytes[TEXT_HOST_NEXT_RESULTS],
  };
  const loom_serve_module_options_t options = {
      .binding_capacity = 7,
      .stages = {runner->stage_count, runner->stage_bindings},
      .feedback = {IREE_ARRAYSIZE(feedback), feedback},
  };
  status = loom_serve_module_create(&runner->source.types, runner->execution,
                                    options, runner->host_allocator,
                                    &runner->native_module);
  IREE_RETURN_IF_ERROR(status);
  iree_vm_module_t* libraries[] = {runner->native_module, NULL, NULL, NULL};
  status = loom_serve_input_module_create(
      runner->source.environment, runner->source.tokenizer,
      runner->host_allocator, &libraries[1]);
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(
        runner->source.environment, runner->host_allocator, &libraries[2]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_tools_module_create(
        runner->source.environment, runner->host_allocator, &libraries[3]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link(runner->source.module,
                                     iree_vm_module_span_from_array(libraries),
                                     runner->host_allocator, &runner->program);
  }
  iree_vm_module_release(libraries[3]);
  iree_vm_module_release(libraries[2]);
  iree_vm_module_release(libraries[1]);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_policy_initialize(
      runner->source.environment, runner->program, runner->source.options,
      &runner->chat_policy));
  iree_vm_process_t* process = loom_serve_program_process(runner->program);
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("step"), &runner->step));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("encode_epoch"),
      &runner->encode_epoch));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("publish_epoch"),
      &runner->publish_epoch));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("encode_state"),
      &runner->encode_state));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("bind_request"),
      &runner->bind_request));
  return iree_vm_process_lookup_function(process, IREE_SV("model"),
                                         IREE_SV("epoch"), &runner->epoch_step);
}

typedef enum text_load_phase_e {
  TEXT_LOAD_PREPARE,
  TEXT_LOAD_STAGES,
  TEXT_LOAD_WEIGHTS,
  TEXT_LOAD_LINK,
  TEXT_LOAD_READY,
  TEXT_LOAD_COMPLETE,
} text_load_phase_t;

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
  text_load_phase_t phase;
  // Next stage or parameter domain within the current load phase.
  iree_host_size_t index;
  // Once cancelled, subsequent advance calls cannot restart preparation.
  bool cancelled;
  // Private task result transferred only by the task's final release.
  iree_status_t task_status;
  // Joins worker/proactor final publication with the owner's next advance.
  struct {
    // Protects completion publication and its final notification access.
    iree_slim_mutex_t mutex;
    // Accepted work retains this load until its final callback.
    bool pending;
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
  load->event.status = status;
  load->event.pending = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static iree_status_t loom_serve_text_load_prepare(
    loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  const loom_serve_text_options_t* options = &load->options;
  model->shape_count = options->epoch_count;
  model->context_capacity = options->context_capacity;
  if (model->shape_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        model->host_allocator,
        iree_make_const_byte_span(options->epoch_shapes,
                                  model->shape_count * sizeof(*model->shapes)),
        (void**)&model->shapes));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_source_initialize(
      options, model->host_allocator, &model->source, &model->state.storage));
  const iree_host_size_t stage_count =
      loom_serve_preparation_stage_count(model->source.preparation);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->host_allocator, stage_count, sizeof(*model->stages),
      (void**)&model->stages));
  model->stage_count = stage_count;
  return loom_serve_jit_allocate(model->device, model->dispatch,
                                 loom_serve_device_host(model->device_owner),
                                 &options->sources, &options->kernel_sanitizer,
                                 &model->source.compilation,
                                 model->host_allocator, &model->jit);
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

static void loom_serve_text_load_compiled(void* user_data, iree_status_t status,
                                          loom_serve_jit_stage_t* compiled) {
  loom_serve_text_load_t* load = user_data;
  load->model->stages[load->index - 1].compiled = compiled;
  loom_serve_text_load_complete(load, status);
}

static void loom_serve_text_load_ready(void* user_data,
                                       iree_async_operation_t* operation,
                                       iree_status_t status,
                                       iree_async_completion_flags_t flags) {
  loom_serve_text_load_complete(user_data, status);
}

static iree_status_t loom_serve_text_load_bind(loom_serve_text_model_t* model) {
  IREE_RETURN_IF_ERROR(loom_serve_device_create_execution(model->device_owner,
                                                          &model->execution));
  return loom_serve_text_stages_bind(
      model->device_owner, model->source.preparation,
      model->source.shared_stage_count, model->stage_count, model->stages,
      &model->parameters, model->host_allocator);
}

static iree_status_t loom_serve_text_load_link(loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
  iree_status_t status = iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(model->device_owner), model->execution,
      model->parameters.count, model->parameters.plans, model->host_allocator,
      &model->residency));
  for (iree_host_size_t i = 0;
       i < model->stage_count && iree_status_is_ok(status); ++i) {
    loom_serve_text_stage_t* stage = &model->stages[i];
    status = loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(model->dispatch),
        model->command_mode, NULL, stage->fixed_buffers, &stage->command);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_state_allocate(
      &model->state, model->device_owner, model->execution));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(model->host_allocator, model->row_count,
                                  sizeof(*model->rows), (void**)&model->rows));
  for (iree_host_size_t i = 0; i < model->row_count; ++i) {
    model->rows[i].model = model;
    model->rows[i].state = &model->state.rows[i];
  }
  if (model->state.checkpoints.capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        model->host_allocator, model->state.checkpoints.capacity,
        sizeof(*model->checkpoints), (void**)&model->checkpoints));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_create_program(model));
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
    loom_serve_device_t* device, const loom_serve_text_options_t* options,
    loom_serve_text_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_text_load_t** out_load) {
  *out_load = NULL;
  if (options->row_count < 1 || options->row_count > TEXT_ROW_CAPACITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "text row count must be in [1, 16]");
  }
  if (options->prefill_capacity < 1 ||
      options->context_capacity < options->prefill_capacity ||
      options->context_capacity > 262144 || options->epoch_count > INT32_MAX ||
      options->epoch_count > SIZE_MAX / (2 * sizeof(int64_t)) ||
      options->draft_depth > 7 ||
      (options->draft_depth && !options->epoch_count) ||
      options->pool_capacity > 4194304 ||
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
    if (shape.token_capacity < 1 ||
        shape.token_capacity > options->context_capacity ||
        shape.span_capacity < 1 || shape.span_capacity > TEXT_ROW_CAPACITY) {
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
  iree_status_t status = load->event.status;
  load->event.status = iree_ok_status();
  iree_slim_mutex_unlock(&load->event.mutex);
  if (iree_status_is_ok(status) && load->cancelled) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "model load cancelled");
  }
  loom_serve_text_model_t* model = load->model;
  loom_serve_host_t* host = loom_serve_device_host(model->device_owner);
  bool submitted = false;
  while (iree_status_is_ok(status) && !submitted &&
         load->phase != TEXT_LOAD_COMPLETE) {
    // Every phase either advances inline or lends this operation to one task.
    iree_slim_mutex_lock(&load->event.mutex);
    load->event.pending = true;
    iree_slim_mutex_unlock(&load->event.mutex);
    switch (load->phase) {
      case TEXT_LOAD_PREPARE:
        load->phase = TEXT_LOAD_STAGES;
        status = iree_status_from_loomc(loomc_task_sink_submit(
            loom_serve_host_compiler_sink(host), &load->task));
        submitted = iree_status_is_ok(status);
        break;
      case TEXT_LOAD_STAGES:
        if (load->index < model->stage_count) {
          const loom_serve_preparation_stage_t* declaration =
              loom_serve_preparation_stage(model->source.preparation,
                                           load->index++);
          if (declaration->kind != LOOM_SERVE_PREPARATION_STAGE_COMMAND) {
            status = iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "text stages require command parameter reflection");
            break;
          }
          status = loom_serve_jit_compile_async(
              model->jit, declaration->root, &declaration->config,
              loom_serve_text_load_compiled, load);
          submitted = iree_status_is_ok(status);
        } else {
          status = loom_serve_text_load_bind(model);
          load->index = 0;
          load->phase = TEXT_LOAD_WEIGHTS;
        }
        break;
      case TEXT_LOAD_WEIGHTS:
        if (load->index < model->parameters.count) {
          const iree_host_size_t index = load->index++;
          const loom_serve_preparation_parameter_t* checkpoint =
              model->parameters.sources[index];
          status = loom_serve_weights_prepare_async(
              model->parameters.plans[index], host, model->jit,
              model->command_mode, &load->options.sources, checkpoint->paths,
              checkpoint->policy, loom_serve_text_load_complete, load);
          submitted = iree_status_is_ok(status);
        } else {
          load->phase = TEXT_LOAD_LINK;
        }
        break;
      case TEXT_LOAD_LINK:
        status = loom_serve_text_load_link(load);
        submitted = iree_status_is_ok(status);
        load->phase = TEXT_LOAD_READY;
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
    // Rejected work owes no callback; terminal publication is already joined.
    iree_slim_mutex_lock(&load->event.mutex);
    load->event.pending = false;
    iree_slim_mutex_unlock(&load->event.mutex);
    load->phase = TEXT_LOAD_COMPLETE;
  }
  *out_complete = load->phase == TEXT_LOAD_COMPLETE;
  return status;
}

const loom_serve_compilation_report_t* loom_serve_text_load_report(
    const loom_serve_text_load_t* load) {
  return &load->model->source.compilation;
}

loom_serve_text_model_t* loom_serve_text_load_take_model(
    loom_serve_text_load_t* load) {
  loom_serve_text_model_t* model = load->model;
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
    loom_serve_device_t* device, const loom_serve_text_options_t* options,
    iree_allocator_t host_allocator, loom_serve_text_model_t** out_model) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_load_t* load = NULL;
  iree_status_t status = loom_serve_text_load_allocate(
      device, options, loom_serve_text_load_notify, &notification,
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
  if (model->execution) {
    loom_serve_execution_set_retirement_observer(model->execution, wake,
                                                 user_data);
  }
  for (iree_host_size_t i = 0; i < model->parameters.count; ++i) {
    loom_serve_weights_set_retirement_observer(model->parameters.plans[i], wake,
                                               user_data);
  }
  loom_serve_text_chat_policy_deinitialize(&model->chat_policy);
  loom_serve_program_free(model->program);
  model->program = NULL;
  iree_vm_module_release(model->native_module);
  model->native_module = NULL;
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
  bool retired =
      !model->execution || loom_serve_execution_is_retired(model->execution);
  for (iree_host_size_t i = 0; i < model->parameters.count; ++i) {
    retired &= loom_serve_weights_is_retired(model->parameters.plans[i]);
  }
  return retired && loom_serve_retirement_is_idle(&model->state.retirement);
}

iree_status_t loom_serve_text_model_free_retired(
    loom_serve_text_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  IREE_ASSERT(loom_serve_text_model_is_retired(model));
  loom_serve_residency_free(model->residency);
  model->residency = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < model->parameters.count && iree_status_is_ok(status); ++i) {
    status = loom_serve_weights_free(model->parameters.plans[i]);
    if (iree_status_is_ok(status)) {
      model->parameters.plans[i] = NULL;
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_state_deinitialize(&model->state));
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    loom_serve_jit_stage_free(model->stages[i].compiled);
    iree_allocator_free(model->host_allocator, model->stages[i].fixed_buffers);
  }
  iree_allocator_free(model->host_allocator, model->shapes);
  iree_allocator_free(model->host_allocator, model->stages);
  iree_allocator_free(model->host_allocator, model->stage_bindings);
  iree_allocator_free(model->host_allocator, model->parameters.plans);
  iree_allocator_free(model->host_allocator, model->parameters.sources);
  loom_serve_jit_free(model->jit);
  for (iree_host_size_t i = 0; i < TEXT_HOST_BUFFER_COUNT; ++i) {
    iree_vm_buffer_release(model->host.buffers[i]);
  }
  iree_allocator_free(model->host_allocator, model->rows);
  iree_allocator_free(model->host_allocator, model->checkpoints);
  loom_serve_execution_release(model->execution);
  loom_serve_text_source_deinitialize(&model->source);
  iree_allocator_free(model->host_allocator, model);
  return iree_ok_status();
}

iree_status_t loom_serve_text_model_free(loom_serve_text_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_text_model_retire(model, NULL, NULL);
  iree_status_t status = model->execution
                             ? loom_serve_execution_drain(model->execution)
                             : iree_ok_status();
  for (iree_host_size_t i = 0; i < model->parameters.count; ++i) {
    loom_serve_weights_await_retirement(model->parameters.plans[i]);
  }
  loom_serve_retirement_await(&model->state.retirement);
  return iree_status_join(status, loom_serve_text_model_free_retired(model));
}

loom_serve_device_t* loom_serve_text_model_device(
    const loom_serve_text_model_t* model) {
  return model->device_owner;
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
  return variant ? loom_serve_text_variant_bind(variant, model,
                                                model->native_module)
                 : loom_serve_module_bind_stages(model->native_module,
                                                 model->stage_count,
                                                 model->stage_bindings);
}

loom_serve_text_row_t* loom_serve_text_model_row(loom_serve_text_model_t* model,
                                                 iree_host_size_t index) {
  return &model->rows[index];
}

loom_serve_memory_statistics_t loom_serve_text_model_memory_statistics(
    const loom_serve_text_model_t* model) {
  return model->state.memory.statistics;
}

loom_serve_memory_statistics_t loom_serve_text_model_weight_statistics(
    const loom_serve_text_model_t* model) {
  return loom_serve_residency_statistics(model->residency);
}

loom_serve_weight_activity_t loom_serve_text_model_weight_activity(
    const loom_serve_text_model_t* model) {
  return loom_serve_residency_activity(model->residency);
}

loom_serve_residency_t* loom_serve_text_model_residency(
    const loom_serve_text_model_t* model) {
  return model->residency;
}

iree_status_t loom_serve_text_model_activate(loom_serve_text_model_t* model) {
  return loom_serve_residency_activate(model->residency);
}

iree_status_t loom_serve_text_model_deactivate(loom_serve_text_model_t* model) {
  return loom_serve_residency_deactivate(model->residency);
}

iree_tokenizer_t* loom_serve_text_model_tokenizer(
    loom_serve_text_model_t* model) {
  return model->source.tokenizer;
}

const loom_serve_text_chat_policy_t* loom_serve_text_model_chat_policy(
    const loom_serve_text_model_t* model) {
  return &model->chat_policy;
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

loom_serve_text_pool_usage_t loom_serve_text_model_pool_usage(
    const loom_serve_text_model_t* model) {
  loom_serve_text_pool_usage_t usage = {
      .block_size = model->state.storage.block_size,
      .capacity = model->state.cache.capacity,
      .available = (iree_host_size_t)model->state.cache.pool.available *
                   model->state.storage.block_size,
  };
  for (iree_host_size_t i = 0; usage.capacity && i < model->row_count; ++i) {
    const loom_serve_text_state_row_t* row = &model->state.rows[i];
    if (row->reserved_extent) {
      const iree_host_size_t live =
          (row->position + usage.block_size - 1) / usage.block_size;
      usage.pending += (row->block_count - live) * usage.block_size;
    }
  }
  return usage;
}

iree_host_size_t loom_serve_text_row_pool_usage(
    const loom_serve_text_row_t* row) {
  return (iree_host_size_t)row->state->block_count *
         row->model->state.storage.block_size;
}

iree_host_size_t loom_serve_text_row_pool_growth(
    const loom_serve_text_row_t* row, iree_host_size_t extent) {
  if (!row->model->state.cache.capacity) {
    return 0;
  }
  return loom_serve_text_state_row_growth(
             row->state, (uint32_t)(extent - row->state->position)) *
         row->model->state.storage.block_size;
}

iree_host_size_t loom_serve_text_row_pool_reclaimable(
    const loom_serve_text_row_t* row) {
  const loom_serve_text_state_t* state = &row->model->state;
  iree_host_size_t count = 0;
  const iree_host_size_t map = (iree_host_size_t)(row->state - state->rows) *
                               state->storage.blocks_per_row;
  for (uint32_t i = 0; i < row->state->block_count; ++i) {
    count += !loom_serve_block_pool_is_shared(&state->cache.pool,
                                              state->cache.maps[map + i]);
  }
  return count * state->storage.block_size;
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
    loom_serve_text_model_t* model, loom_serve_text_trim_result_t* result) {
  return loom_serve_text_state_complete_trim(&model->state, result);
}

iree_status_t loom_serve_text_model_trim(
    loom_serve_text_model_t* model, loom_serve_text_trim_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(model->execution));
  loom_serve_execution_frontier_t frontier;
  iree_status_t status =
      loom_serve_text_model_submit_trim(model, out_result, &frontier);
  status =
      iree_status_join(status, loom_serve_execution_drain(model->execution));
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_model_complete_trim(model, out_result);
  }
  return status;
}

void loom_serve_text_row_reset_retired(loom_serve_text_row_t* row) {
  const bool reserved = row->state->reserved_extent != 0;
  loom_serve_text_state_row_reset(row->state);
  if (reserved) {
    loom_serve_residency_release(row->model->residency);
  }
  row->has_prediction = false;
  memset(&row->transfer, 0, sizeof(row->transfer));
  memset(&row->metrics, 0, sizeof(row->metrics));
}

iree_status_t loom_serve_text_row_reset(loom_serve_text_row_t* row) {
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(row->model->execution));
  loom_serve_text_row_reset_retired(row);
  return iree_ok_status();
}

iree_host_size_t loom_serve_text_row_suspended_bytes(
    const loom_serve_text_row_t* row) {
  return loom_serve_snapshot_size(row->state->snapshot);
}

iree_status_t loom_serve_text_row_suspend(loom_serve_text_row_t* row) {
  if (row->state->reserved_extent) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot suspend an active request reservation");
  }
  return loom_serve_text_state_row_suspend(row->state);
}

iree_status_t loom_serve_text_row_try_resume(loom_serve_text_row_t* row,
                                             bool* out_resumed) {
  return loom_serve_text_state_row_try_resume(row->state, out_resumed);
}

iree_status_t loom_serve_text_row_submit_pin(
    loom_serve_text_row_t* row, loom_serve_text_checkpoint_t** out_checkpoint,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_checkpoint = NULL;
  *out_frontier = (loom_serve_execution_frontier_t){0};
  if (!row->state->position || row->state->snapshot) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "checkpoint requires a nonempty resident row");
  }
  loom_serve_text_state_checkpoint_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_row_submit_pin(row->state, &state));
  if (!state) {
    return iree_ok_status();
  }
  const iree_host_size_t index =
      (iree_host_size_t)(state - row->model->state.checkpoints.values);
  loom_serve_text_checkpoint_t* checkpoint = &row->model->checkpoints[index];
  *checkpoint =
      (loom_serve_text_checkpoint_t){.model = row->model,
                                     .references = 1,
                                     .state = state,
                                     .has_prediction = row->has_prediction,
                                     .token = row->transfer.token};
  *out_checkpoint = checkpoint;
  *out_frontier = loom_serve_execution_frontier(row->model->execution);
  return iree_ok_status();
}

iree_status_t loom_serve_text_row_try_pin(
    loom_serve_text_row_t* row, loom_serve_text_checkpoint_t** out_checkpoint) {
  *out_checkpoint = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(row->model->execution));
  loom_serve_text_checkpoint_t* checkpoint = NULL;
  loom_serve_execution_frontier_t frontier;
  iree_status_t status =
      loom_serve_text_row_submit_pin(row, &checkpoint, &frontier);
  status = iree_status_join(status,
                            loom_serve_execution_drain(row->model->execution));
  if (iree_status_is_ok(status)) {
    *out_checkpoint = checkpoint;
  }
  return status;
}

iree_status_t loom_serve_text_row_try_restore(
    loom_serve_text_row_t* row, const loom_serve_text_checkpoint_t* checkpoint,
    bool* out_restored) {
  *out_restored = false;
  if (checkpoint->model != row->model) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "checkpoint belongs to a different model");
  }
  if (row->state->reserved_extent) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot replace an active request reservation");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_try_restore(
      row->state, checkpoint->state, out_restored));
  if (*out_restored) {
    row->has_prediction = checkpoint->has_prediction;
    memset(&row->transfer, 0, sizeof(row->transfer));
    row->transfer.token = checkpoint->token;
    row->transfer.progress[0] = checkpoint->has_prediction;
    row->transfer.progress[7] =
        checkpoint->has_prediction &&
        checkpoint->token == row->model->source.eos_token;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_text_row_submit_reserve(
    loom_serve_text_row_t* row, loom_serve_text_reserve_mode_t mode,
    const loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t extent,
    loom_serve_residency_admission_t* out_admission,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
  *out_frontier = (loom_serve_execution_frontier_t){0};
  const iree_host_size_t block_size = row->model->state.storage.block_size;
  const iree_host_size_t maximum =
      ((row->model->context_capacity + block_size - 1) / block_size) *
      block_size;
  if ((mode != LOOM_SERVE_TEXT_RESERVE_FRESH &&
       mode != LOOM_SERVE_TEXT_RESERVE_CONTINUE &&
       mode != LOOM_SERVE_TEXT_RESERVE_CHECKPOINT) ||
      ((checkpoint != NULL) != (mode == LOOM_SERVE_TEXT_RESERVE_CHECKPOINT)) ||
      (checkpoint && checkpoint->model != row->model) || !extent ||
      extent > maximum ||
      (checkpoint && extent < checkpoint->state->position) ||
      (mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE &&
       extent < row->state->position)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid text completion reservation");
  }
  if (row->state->reserved_extent ||
      (mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE && row->state->snapshot)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "reservation requires an idle resident frontier");
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_try_admit(row->model->residency, out_admission));
  if (*out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    return iree_ok_status();
  }
  // Even state-capacity denial retains the activation pin until completion.
  // Cancellation likewise leaves this operation intact until it retires.
  bool reserved = false;
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_submit_reserve(
      row->state, mode, checkpoint ? checkpoint->state : NULL, extent,
      &reserved));
  if (reserved) {
    if (mode == LOOM_SERVE_TEXT_RESERVE_FRESH) {
      row->has_prediction = false;
      memset(&row->transfer, 0, sizeof(row->transfer));
      memset(&row->metrics, 0, sizeof(row->metrics));
    } else if (checkpoint) {
      row->has_prediction = checkpoint->has_prediction;
      memset(&row->transfer, 0, sizeof(row->transfer));
      row->transfer.token = checkpoint->token;
      row->transfer.progress[0] = checkpoint->has_prediction;
      row->transfer.progress[7] =
          checkpoint->has_prediction &&
          checkpoint->token == row->model->source.eos_token;
    }
  }
  *out_frontier = loom_serve_execution_frontier(row->model->execution);
  return iree_ok_status();
}

bool loom_serve_text_row_complete_reserve(loom_serve_text_row_t* row) {
  const bool admitted = loom_serve_text_state_row_complete_reserve(row->state);
  if (!admitted) {
    loom_serve_residency_release(row->model->residency);
  }
  return admitted;
}

iree_status_t loom_serve_text_row_try_reserve(
    loom_serve_text_row_t* row, loom_serve_text_reserve_mode_t mode,
    const loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t extent,
    bool* out_admitted) {
  *out_admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(row->model->execution));
  loom_serve_residency_admission_t admission;
  loom_serve_execution_frontier_t frontier;
  iree_status_t status = loom_serve_text_row_submit_reserve(
      row, mode, checkpoint, extent, &admission, &frontier);
  loom_serve_residency_cancel_admission(row->model->residency);
  status = iree_status_join(status,
                            loom_serve_execution_drain(row->model->execution));
  if (iree_status_is_ok(status) &&
      admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    *out_admitted = loom_serve_text_row_complete_reserve(row);
  }
  return status;
}

void loom_serve_text_row_release_reservation(loom_serve_text_row_t* row) {
  if (!row->state->reserved_extent) {
    return;
  }
  loom_serve_text_state_row_release_reservation(row->state);
  loom_serve_residency_release(row->model->residency);
}

void loom_serve_text_checkpoint_retain(
    loom_serve_text_checkpoint_t* checkpoint) {
  if (checkpoint) {
    ++checkpoint->references;
  }
}

void loom_serve_text_checkpoint_release(
    loom_serve_text_checkpoint_t* checkpoint) {
  if (!checkpoint || --checkpoint->references) {
    return;
  }
  loom_serve_text_state_checkpoint_release(checkpoint->state);
  checkpoint->state = NULL;
}

iree_status_t loom_serve_text_checkpoint_submit_suspend(
    loom_serve_text_checkpoint_t* checkpoint,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_checkpoint_submit_suspend(checkpoint->state));
  *out_frontier = loom_serve_execution_frontier(checkpoint->model->execution);
  return iree_ok_status();
}

void loom_serve_text_checkpoint_complete_suspend(
    loom_serve_text_checkpoint_t* checkpoint) {
  loom_serve_text_state_checkpoint_complete_suspend(checkpoint->state);
}

iree_status_t loom_serve_text_checkpoint_suspend(
    loom_serve_text_checkpoint_t* checkpoint) {
  loom_serve_execution_t* execution = checkpoint->model->execution;
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(execution));
  loom_serve_execution_frontier_t frontier;
  iree_status_t status =
      loom_serve_text_checkpoint_submit_suspend(checkpoint, &frontier);
  status = iree_status_join(status, loom_serve_execution_drain(execution));
  if (iree_status_is_ok(status)) {
    loom_serve_text_checkpoint_complete_suspend(checkpoint);
  }
  return status;
}

iree_host_size_t loom_serve_text_checkpoint_suspended_bytes(
    const loom_serve_text_checkpoint_t* checkpoint) {
  return loom_serve_snapshot_size(checkpoint->state->snapshot);
}

iree_host_size_t loom_serve_text_model_checkpoint_suspended_bytes(
    const loom_serve_text_model_t* model) {
  iree_host_size_t bytes = 0;
  for (iree_host_size_t i = 0; i < model->state.checkpoints.capacity; ++i) {
    const loom_serve_text_checkpoint_t* checkpoint = &model->checkpoints[i];
    if (checkpoint->references) {
      bytes += loom_serve_text_checkpoint_suspended_bytes(checkpoint);
    }
  }
  return bytes;
}

static iree_status_t loom_serve_text_invoke(loom_serve_text_model_t* model,
                                            iree_vm_function_t function,
                                            iree_vm_variant_span_t arguments,
                                            iree_vm_variant_span_t results) {
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->program), function,
                     arguments, results);
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
      model, model->step, iree_vm_variant_span_from_array(arguments),
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
  if (!model->state.recurrent.dirty_rows) {
    return iree_ok_status();
  }
  uint32_t count = 0;
  for (iree_host_size_t i = 0; i < model->row_count; ++i) {
    if (!(model->state.recurrent.dirty_rows & (1u << i))) {
      continue;
    }
    const loom_serve_text_state_row_t* row = &model->state.rows[i];
    const uint64_t origin =
        iree_hal_buffer_byte_offset(row->buffers[TEXT_RECURRENT]);
    const uint64_t read_origin =
        row->recurrent.anchor == UINT32_MAX
            ? origin
            : iree_hal_buffer_byte_offset(
                  model->state.recurrent.buffers[row->recurrent.anchor]);
    uint8_t* record =
        model->host.bytes[TEXT_HOST_STATE_BINDINGS].data + count++ * 32;
    iree_unaligned_store_le_u64(record, i);
    iree_unaligned_store_le_u64(record + 8, read_origin);
    iree_unaligned_store_le_u64(record + 16, origin);
    iree_unaligned_store_le_u64(record + 24, row->position);
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->host.buffers[TEXT_HOST_STATE_BINDINGS]),
      iree_vm_variant_from_i32((int32_t)count),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types,
          model->host.buffers[TEXT_HOST_STATE_ORIGINS]),
  };
  IREE_RETURN_IF_ERROR(loom_serve_text_invoke(
      model, model->encode_state, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_empty()));
  const iree_byte_span_t origins = model->host.bytes[TEXT_HOST_STATE_ORIGINS];
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
  model->state.recurrent.dirty_rows = 0;
  return iree_ok_status();
}

iree_status_t loom_serve_text_row_bind_request(loom_serve_text_row_t* row,
                                               iree_vm_buffer_t* options) {
  loom_serve_text_model_t* model = row->model;
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               model->source.control_state),
      iree_vm_variant_from_i32((int32_t)(row - model->rows)),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               options)};
  return loom_serve_text_invoke(model, model->bind_request,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_empty());
}

// The semantic host record is {length, position, source_position, row,
// input_begin, input_count, flags, output_credit, first_span}. Known IDs are
// copied once into a contiguous host stream. Source owns all device packet
// offsets, logical positions and selection policy.
static iree_host_size_t loom_serve_text_prepare_plan(
    loom_serve_text_model_t* model, iree_host_size_t span_count,
    const loom_serve_text_span_t* spans, const uint32_t* output_limits,
    const uint32_t* first_indices, iree_host_size_t record_begin,
    iree_host_size_t input_begin) {
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    const bool generates =
        iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE);
    const iree_host_size_t input_count =
        span->token_ids && !(first_indices && generates)
            ? (generates ? 1 : span->token_count)
            : 0;
    const loom_serve_text_state_row_t* row = model->rows[span->row_index].state;
    const uint32_t fields[TEXT_PLAN_FIELD_COUNT] = {
        [TEXT_PLAN_FIELD_LENGTH] = (uint32_t)span->token_count,
        [TEXT_PLAN_FIELD_POSITION] = row->position,
        [TEXT_PLAN_FIELD_SOURCE_POSITION] = row->source_position,
        [TEXT_PLAN_FIELD_ROW] = (uint32_t)span->row_index,
        [TEXT_PLAN_FIELD_INPUT_BEGIN] = (uint32_t)input_begin,
        [TEXT_PLAN_FIELD_INPUT_COUNT] = (uint32_t)input_count,
        [TEXT_PLAN_FIELD_FLAGS] = span->flags,
        [TEXT_PLAN_FIELD_OUTPUT_LIMIT] = output_limits ? output_limits[i] : 0,
        [TEXT_PLAN_FIELD_FIRST_SPAN] =
            first_indices ? first_indices[i] : (uint32_t)i,
    };
    uint8_t* record = model->host.bytes[TEXT_HOST_PLAN].data +
                      (record_begin + i) * sizeof(fields);
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(fields); ++j) {
      iree_unaligned_store_le_u32(record + j * sizeof(uint32_t), fields[j]);
    }
    if (input_count) {
      memcpy(model->host.bytes[TEXT_HOST_INPUT].data + input_begin * 4,
             span->token_ids, input_count * sizeof(int32_t));
    }
    input_begin += input_count;
  }
  return input_begin;
}

iree_status_t loom_serve_text_model_submit_epoch(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_text_span_t* spans,
    const uint32_t* output_limits,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
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
  uint32_t resident_mask = 0;
  iree_host_size_t token_count = 0;
  uint32_t extents[TEXT_ROW_CAPACITY];
  uint32_t first_indices[TEXT_ROW_CAPACITY];
  uint32_t next_indices[TEXT_ROW_CAPACITY];
  uint32_t continued_speculative_mask = 0;
  const iree_host_size_t epoch_count = continuation ? 2 : 1;
  const iree_host_size_t width = model->draft_depth + 1;
  bool proposes = false;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    if (span->row_index >= model->row_count || !span->token_count ||
        !span->token_ids ||
        span->token_count > shape.token_capacity - token_count ||
        (span->flags & ~(LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                         LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid text epoch span %zu", i);
    }
    const uint32_t resident_bit = 1u << span->row_index;
    if (model->rows[span->row_index].state->snapshot) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "epoch row %zu is suspended", span->row_index);
    }
    if (resident_mask & resident_bit) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "epoch repeats resident row %zu",
                              span->row_index);
    }
    if (span->token_count > model->context_capacity -
                                model->rows[span->row_index].state->position) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "epoch span %zu exceeds remaining context", i);
    }
    const bool generates =
        iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE);
    if (generates && (!output_limits || !output_limits[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "proposal requires a speculative text span");
    }
    if (output_limits && output_limits[i]) {
      const loom_serve_text_row_t* row = &model->rows[span->row_index];
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
    first_indices[span->row_index] = (uint32_t)i;
    resident_mask |= resident_bit;
    token_count += span->token_count;
    proposes |= generates;
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
    uint32_t next_mask = 0;
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
      const uint32_t bit = 1u << span->row_index;
      if (!(resident_mask & bit) || (next_mask & bit)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "continuation must use distinct first-plan rows");
      }
      const uint32_t first = first_indices[span->row_index];
      const loom_serve_text_span_t* prior = &spans[first];
      const uint32_t limit = continuation->output_limits[i];
      const iree_host_size_t remaining =
          model->context_capacity -
          model->rows[span->row_index].state->position;
      if (limit) {
        if (limit > 2 * width || span->token_count != width ||
            span->flags != (LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                            LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE) ||
            !iree_any_bit_set(prior->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT) ||
            (output_limits[first] && limit > output_limits[first])) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid speculative continuation span %zu",
                                  i);
        }
        const iree_host_size_t extent =
            output_limits[first] ? iree_min(2 * width, limit + width - 1)
                                 : prior->token_count + width;
        extents[first] = (uint32_t)iree_min(remaining, extent);
        continued_speculative_mask |= 1u << first;
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
      next_indices[i] = first;
      next_mask |= bit;
      next_tokens += span->token_count;
    }
  }

  uint32_t required_blocks = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_state_row_t* row =
        model->rows[spans[i].row_index].state;
    if (row->reserved_extent &&
        extents[i] > row->reserved_extent - row->position) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "epoch exceeds request completion reservation");
    }
  }
  for (iree_host_size_t i = 0; model->state.cache.capacity && i < span_count;
       ++i) {
    const loom_serve_text_row_t* row = &model->rows[spans[i].row_index];
    required_blocks += loom_serve_text_state_row_growth(row->state, extents[i]);
  }
  if (required_blocks > model->state.cache.pool.available) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "epoch needs %u KV blocks; %u are available",
                            required_blocks, model->state.cache.pool.available);
  }
  IREE_RETURN_IF_ERROR(loom_serve_residency_acquire(model->residency));
  model->invocation.span_count = span_count;
  model->invocation.flags = (output_limits ? TEXT_EPOCH_FLAG_VERIFY : 0) |
                            (continuation ? TEXT_EPOCH_FLAG_CONTINUATION : 0);
  model->invocation.continued_speculative_mask = continued_speculative_mask;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    model->invocation.row_indices[i] = spans[i].row_index;
  }
  iree_status_t status =
      loom_serve_text_state_grow(&model->state, span_count, spans, extents);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_publish_state_bindings(model);
  }

  const iree_host_size_t input_count = loom_serve_text_prepare_plan(
      model, span_count, spans, output_limits, NULL, 0, 0);
  if (continuation) {
    loom_serve_text_prepare_plan(model, continuation->span_count,
                                 continuation->spans,
                                 continuation->output_limits, next_indices,
                                 TEXT_ROW_CAPACITY, input_count);
  }
  iree_vm_variant_t encode_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               model->source.control_state),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_PLAN]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_INPUT]),
      iree_vm_variant_from_i32((int32_t)span_count),
      iree_vm_variant_from_i32(continuation ? (int32_t)continuation->span_count
                                            : 0),
      iree_vm_variant_from_i32(model->source.eos_token),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_METADATA]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_TOKENS]),
  };
  iree_vm_variant_t encode_results[1] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_invoke(
        model, model->encode_epoch,
        iree_vm_variant_span_from_array(encode_arguments),
        iree_vm_variant_span_from_array(encode_results));
  }
  int32_t output_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(encode_results[0], &output_count);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(encode_results));
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->host.bytes[TEXT_HOST_METADATA].data,
                  .target_buffer = model->state.epoch.buffers[1],
                  .length = model->host.bytes[TEXT_HOST_METADATA].data_length}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->host.bytes[TEXT_HOST_TOKENS].data,
                  .target_buffer = model->state.epoch.buffers[4],
                  .length = model->host.bytes[TEXT_HOST_TOKENS].data_length}},
  };
  uint64_t completion = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_transfer(
        model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  }
  iree_hal_buffer_t* buffers[] = {
      model->state.residual,         model->state.epoch.buffers[1],
      model->state.epoch.buffers[2], model->state.row_arena,
      model->state.epoch.buffers[4], model->state.epoch.buffers[5],
      model->state.draft.carry,      model->state.draft.committed,
      model->state.draft.results,    model->state.draft.cache,
      model->state.draft.row_table,  model->state.draft.next_results,
  };
  iree_vm_variant_t arguments[9 + IREE_ARRAYSIZE(buffers)] = {
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
      iree_vm_variant_from_i32(continued_speculative_mask != 0),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    arguments[9 + i] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->source.types, buffers[i]);
  }
  iree_vm_variant_t results[1] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_invoke(model, model->epoch_step,
                                    iree_vm_variant_span_from_array(arguments),
                                    iree_vm_variant_span_from_array(results));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    *out_frontier = loom_serve_execution_frontier(model->execution);
  }
  return status;
}

iree_status_t loom_serve_text_model_complete_epoch(
    loom_serve_text_model_t* model, loom_serve_text_result_t* out_results) {
  iree_vm_variant_t publish_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_PLAN]),
      iree_vm_variant_from_i32((int32_t)model->invocation.span_count),
      iree_vm_variant_from_i32(
          iree_any_bit_set(model->invocation.flags, TEXT_EPOCH_FLAG_VERIFY)),
      iree_vm_variant_from_i32(iree_any_bit_set(model->invocation.flags,
                                                TEXT_EPOCH_FLAG_CONTINUATION)),
      iree_vm_variant_from_i32(
          (int32_t)model->invocation.continued_speculative_mask),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_OUTPUTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_RESULTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_NEXT_RESULTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->source.vm_types, model->host.buffers[TEXT_HOST_PROGRESS]),
  };
  IREE_RETURN_IF_ERROR(
      loom_serve_text_invoke(model, model->publish_epoch,
                             iree_vm_variant_span_from_array(publish_arguments),
                             iree_vm_variant_span_empty()));
  for (iree_host_size_t i = 0; i < model->invocation.span_count; ++i) {
    const uint8_t* record = model->host.bytes[TEXT_HOST_PROGRESS].data +
                            i * TEXT_PROGRESS_FIELD_COUNT * sizeof(int32_t);
    loom_serve_text_result_t result = {
        .consumed_count = iree_unaligned_load_le_u32(
            record + TEXT_PROGRESS_FIELD_CONSUMED_COUNT * sizeof(int32_t)),
        .known_count = iree_unaligned_load_le_u32(
            record + TEXT_PROGRESS_FIELD_KNOWN_COUNT * sizeof(int32_t)),
        .output_count = iree_unaligned_load_le_u32(
            record + TEXT_PROGRESS_FIELD_OUTPUT_COUNT * sizeof(int32_t)),
        .verification_count = iree_unaligned_load_le_u32(
            record + TEXT_PROGRESS_FIELD_VERIFICATION_COUNT * sizeof(int32_t)),
    };
    const uint32_t source_position = iree_unaligned_load_le_u32(
        record + TEXT_PROGRESS_FIELD_SOURCE_POSITION * sizeof(int32_t));
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(result.tokens); ++j) {
      result.tokens[j] = (int32_t)iree_unaligned_load_le_u32(
          record + (TEXT_PROGRESS_FIELD_TOKENS + j) * sizeof(int32_t));
    }
    if (out_results) {
      out_results[i] = result;
    }
    loom_serve_text_row_t* row = &model->rows[model->invocation.row_indices[i]];
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
      model, shape_index, span_count, spans, output_limits, continuation,
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
        (iree_host_size_t)(row - model->rows),
        count,
        token_ids,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
    };
    return loom_serve_text_model_submit_epoch(model, shape_index, 1, &span,
                                              NULL, NULL, out_frontier);
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_activate(row->state));
  uint8_t* input = model->host.bytes[TEXT_HOST_INPUT].data;
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
        (iree_host_size_t)(row - row->model->rows),
        1,
        &token,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
    };
    return loom_serve_text_model_submit_epoch(row->model, 0, 1, &span, NULL,
                                              NULL, out_frontier);
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

int32_t loom_serve_text_row_token(const loom_serve_text_row_t* row) {
  return row->transfer.token;
}

bool loom_serve_text_row_is_eos(const loom_serve_text_row_t* row) {
  return row->transfer.progress[7] != 0;
}

iree_host_size_t loom_serve_text_row_position(
    const loom_serve_text_row_t* row) {
  return row->state->position;
}

loom_serve_text_metrics_t loom_serve_text_row_metrics(
    const loom_serve_text_row_t* row) {
  return row->metrics;
}
