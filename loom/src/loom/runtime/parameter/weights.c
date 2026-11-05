// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/weights.h"

#include <string.h>

#include "iree/async/semaphore.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/runtime/execution/retirement.h"
#include "loom/runtime/parameter/inputs.h"
#include "loom/runtime/parameter/stream_plan.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/json.h"
#include "loom/runtime/program/preparation.h"
#include "loom/runtime/program/program.h"
#include "loomc/iree.h"

enum { WEIGHT_LANES = 4, WEIGHT_COMPLETIONS = 2 * WEIGHT_LANES };

typedef struct loom_runtime_weight_read_t {
  // Captured source range retained by the input plan.
  const loom_runtime_weight_input_t* input;
  // This lane's byte origin in the shared staging allocation.
  iree_device_size_t staging_offset;
  // Previous input read or preparation, followed by staging allocation.
  iree_hal_semaphore_t* waits[2];
  // Payloads paired with waits, captured by the parameter provider.
  uint64_t wait_values[2];
  // Completed captured-input read on its staging lane.
  uint64_t signal_value;
} loom_runtime_weight_read_t;

struct loom_runtime_weights_t {
  // Allocator owning the plan and cold activation metadata.
  iree_allocator_t host_allocator;
  // Borrowed device enclosing the complete plan lifetime.
  iree_hal_device_t* device;
  // Borrowed native pool for fixed roots; outlives all resolved storage.
  iree_hal_pool_t* storage_pool;
  // Borrowed host preparation queue, valid through final plan retirement.
  loomc_task_sink_t sink;
  // Borrowed exact file transfer queue.
  iree_hal_queue_t* transfer;
  // Borrowed exact in-place preparation queue.
  iree_hal_queue_t* dispatch;
  // Borrowed shared physical budget, or NULL for fixed backing.
  loom_runtime_memory_pool_t* pool;
  // Borrowed shared transient backing, separate from retained parameter budget.
  loom_runtime_workspace_t* workspace;
  // Parameter-only accounting across all unique roots.
  loom_runtime_storage_statistics_t statistics;
  // Accepted streaming/preparation work, never reconstructed from events.
  loom_runtime_parameter_activity_t activity;
  // Final consumer ownership before virtual roots can be unmapped at teardown.
  loom_runtime_retirement_t retirement;
  // Activation lanes retained independently of their submitting host turn.
  struct {
    // Owned physical-slab operations, independent of file readiness failures.
    loom_runtime_memory_operation_t** backing;
    // Accepted operations, including a partially failed activation.
    iree_host_size_t backing_count;
    // Earlier operations have published their results and been freed.
    iree_host_size_t backing_retired;
    // Aggregate native result retained after operation reclamation.
    iree_status_t backing_status;
    // Owned fixed-allocation readiness imported by file-loading lanes.
    iree_hal_semaphore_t* backing_ready;
    // Private fixed-allocation completion, independent of streaming failures.
    iree_async_semaphore_timepoint_t backing_timepoint;
    // Accepted I/O/preparation views, including after failed readiness.
    loom_runtime_retirement_t retirement;
    // Owned lane readiness handles imported into the consuming model timeline.
    iree_hal_semaphore_t* semaphores[WEIGHT_COMPLETIONS];
    // Last submitted preparation or canonical read on each initialized lane.
    uint64_t values[WEIGHT_COMPLETIONS];
    // Initialized handles, including a partially failed activation.
    iree_host_size_t count;
    // Captured-input credit until its queued deallocation completes.
    uint64_t workspace_bytes;
  } activation;
  // Native release borrower, joined before reactivation or destruction.
  loom_runtime_memory_operation_t* deactivation;
  // Owned checkpoint domains, imported-file caches, and captured input ranges.
  loom_runtime_weight_inputs_t* inputs;
  // Owned placement/preparation records, with keys in the owned key storage.
  loom_runtime_weight_span_t* spans;
  // Owned parameter key bytes, independent of source/JIT lifetime.
  char* keys;
  // Number of initialized placement records, including partial creation.
  iree_host_size_t span_count;
  // Owned unique root records; aliasing output slots do not duplicate these.
  loom_runtime_weight_storage_t* storages;
  // Number of initialized unique roots, including partial creation.
  iree_host_size_t storage_count;
  // Backing and bounded file/preparation producers were accepted for
  // activation.
  bool active;
  // Owned terminal producer failure, replayed to later consumers without I/O.
  iree_status_t activation_status;
};

static const loom_runtime_weight_span_t* loom_runtime_weight_find(
    iree_host_size_t count, const loom_runtime_weight_span_t* spans,
    iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(spans[i].parameter.key, key)) {
      return &spans[i];
    }
  }
  return NULL;
}

typedef struct loom_runtime_weight_root_binding_t {
  // Unique storage allocation selected by parameter identity and placement.
  iree_host_size_t storage_index;
  // Byte range of this root within the selected storage allocation.
  iree_hal_buffer_range_t range;
} loom_runtime_weight_root_binding_t;

// Resolves root sharing before submitting any I/O. Only newly allocated roots
// append spans, so tensors shared by multiple stages are read and prepared
// exactly once.
static iree_status_t loom_runtime_weight_resolve_root(
    loom_runtime_weights_t* weights, const loom_cmd_program_t* program,
    loom_cmd_program_parameter_root_t root, iree_host_size_t* span_count,
    loom_runtime_weight_span_t* spans,
    iree_hal_pool_reservation_request_t* requests,
    loom_runtime_weight_root_binding_t* out_binding) {
  const iree_host_size_t prior_count = *span_count;
  const loom_runtime_weight_span_t* shared = NULL;
  iree_device_size_t origin = 0;
  iree_host_size_t count = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    const loom_runtime_weight_span_t* existing =
        loom_runtime_weight_find(prior_count, spans, parameter.key);
    if (!count++) {
      shared = existing;
      if (shared) {
        if (shared->parameter.byte_offset < parameter.byte_offset) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "shared parameter root begins before storage");
        }
        origin = shared->parameter.byte_offset - parameter.byte_offset;
      }
    }
    if ((existing != NULL) != (shared != NULL) ||
        (existing &&
         (existing->storage_index != shared->storage_index ||
          existing->parameter.byte_offset != origin + parameter.byte_offset ||
          existing->parameter.byte_length != parameter.byte_length))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter root cannot share existing placement");
    }
  }
  if (!count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty model parameter root");
  }
  if (shared) {
    *out_binding = (loom_runtime_weight_root_binding_t){
        shared->storage_index, {origin, root.required_byte_length}};
    return iree_ok_status();
  }
  const iree_host_size_t storage_index = weights->storage_count++;
  requests[storage_index] = iree_hal_make_pool_reservation_request(
      root.required_byte_length, root.minimum_alignment);
  *out_binding = (loom_runtime_weight_root_binding_t){
      storage_index, {0, root.required_byte_length}};
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    spans[(*span_count)++] = (loom_runtime_weight_span_t){
        .parameter = parameter,
        .storage_index = storage_index,
    };
  }
  return iree_ok_status();
}

typedef struct loom_runtime_weight_preparer_t {
  // Owned source declaration, independent of the policy invocation.
  loom_runtime_preparation_t* declaration;
  // Owned recorded command retaining its executables independently of the JIT.
  iree_hal_command_buffer_t* command;
  // Final output alone, or final output followed by captured input staging.
  uint32_t binding_count;
} loom_runtime_weight_preparer_t;

typedef struct loom_runtime_weight_preparation_t {
  // Initial CPU task; its destruction transfers ownership to the JIT chain.
  loomc_task_t task;
  // Shared preparation scheduling domain, carried through native children.
  loomc_task_sink_t sink;
  // Exclusively borrowed unactivated plan through terminal completion.
  loom_runtime_weights_t* weights;
  // Borrowed source catalog through terminal completion; other jobs may
  // overlap.
  loom_runtime_jit_t* jit;
  // Private policy compilation evidence merged before native preparation
  // starts.
  loom_runtime_compilation_report_t report;
  // Recording policy applied to each source-selected preparation command.
  iree_hal_command_buffer_mode_t command_mode;
  // Borrowed checkpoint sources and policy, retained through completion.
  const loom_runtime_preparation_parameter_t* parameter;
  // Borrowed immutable source catalog through completion; input recipes retain
  // their selected sources independently on the cold worker.
  loom_runtime_parameter_sources_t* indexed_sources;
  // Copied namespace descriptor borrowing its strings and package through
  // completion. Checkpoints use its filesystem base; policy uses its package.
  loom_runtime_sources_t sources;
  // Owned distinct preparers, queried once before native compilation.
  loom_runtime_weight_preparer_t* preparers;
  // Number of owned preparers, including partial policy failure.
  iree_host_size_t preparer_count;
  // Current native compilation ordinal; equals count at successful completion.
  iree_host_size_t preparer_index;
  // Owned per-span preparer ordinal; SIZE_MAX preserves canonical file bytes.
  iree_host_size_t* span_preparers;
  // CPU policy outcome, consumed by the initial task destructor.
  iree_status_t status;
  // Joins the already-queued fixed root allocation before publication.
  iree_async_semaphore_timepoint_t backing_timepoint;
  // Terminal ownership transfer, invoked after all job storage is released.
  loom_runtime_weights_completion_fn_t completion;
  // Borrowed terminal callback payload.
  void* user_data;
} loom_runtime_weight_preparation_t;

static iree_status_t loom_runtime_weight_query_preparer(
    loom_runtime_program_t* program, const iree_vm_ref_types_t* types,
    iree_vm_function_t function, iree_vm_module_t* library,
    iree_vm_module_t* input_library, loom_cmd_program_parameter_t parameter,
    iree_allocator_t host_allocator,
    loom_runtime_preparation_t** out_preparation,
    loom_runtime_weight_input_recipe_t* out_inputs) {
  *out_preparation = NULL;
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)parameter.key.data, parameter.key.size),
      iree_vm_buffer_release_callback_null(), host_allocator, &key));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(types, &key)};
  iree_vm_variant_t results[1] = {0};
  loom_runtime_preparation_module_begin(library, 0, NULL, 0);
  loom_runtime_weight_inputs_module_begin(input_library);
  iree_status_t status =
      iree_vm_invoke(loom_runtime_program_invocation(program), function,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  loom_runtime_preparation_t* preparation =
      loom_runtime_preparation_module_take(library);
  *out_inputs = loom_runtime_weight_inputs_module_take(input_library);
  const iree_host_size_t stage_count =
      preparation ? loom_runtime_preparation_stage_count(preparation) : 0;
  int64_t byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[0], &byte_length);
  }
  if (iree_status_is_ok(status)) {
    if (stage_count > 1 ||
        (stage_count && loom_runtime_preparation_stage(preparation, 0)->kind !=
                            LOOM_RUNTIME_PREPARATION_STAGE_COMMAND)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "weight policy must declare at most one command");
    } else if (!stage_count && !out_inputs->count && byte_length != 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "canonical weight policy must return zero byte length");
    } else if ((stage_count || out_inputs->count) &&
               (byte_length <= 0 ||
                (uint64_t)byte_length != parameter.byte_length)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "weight policy for '%.*s' requires %" PRId64
                                " bytes, reflected parameter has %" PRIu64,
                                (int)parameter.key.size, parameter.key.data,
                                byte_length, (uint64_t)parameter.byte_length);
    } else if (!stage_count && out_inputs->count &&
               (out_inputs->byte_length != parameter.byte_length ||
                out_inputs->payload_length != parameter.byte_length)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "direct weight inputs must cover the complete "
                                "parameter without gaps");
    }
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    *out_preparation = preparation;
  } else {
    loom_runtime_preparation_free(preparation);
  }
  return status;
}

static iree_status_t loom_runtime_weight_prepare_policy(
    loom_runtime_weight_preparation_t* job) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_runtime_weights_t* weights = job->weights;
  const iree_allocator_t host_allocator = weights->host_allocator;
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_environment_allocate(host_allocator, &environment));
  loom_runtime_program_t* program = NULL;
  iree_vm_module_t* libraries[4] = {NULL, NULL, NULL, NULL};
  const iree_string_view_t roots[] = {IREE_SVL("prepare_weight")};
  iree_status_t status = loom_runtime_preparation_module_create(
      environment, job->sources.card, job->sources.component, host_allocator,
      &libraries[0]);
  if (iree_status_is_ok(status)) {
    status = loom_runtime_weight_inputs_module_create(
        weights->inputs, environment, host_allocator, &libraries[1]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_json_module_create(environment, host_allocator,
                                             &libraries[2]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_input_module_create(environment, NULL, host_allocator,
                                              &libraries[3]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_program_allocate(
        loom_runtime_jit_target_environment(job->jit), environment,
        &job->sources, job->parameter->policy, IREE_ARRAYSIZE(roots), roots,
        (iree_vm_module_span_t){libraries, IREE_ARRAYSIZE(libraries)},
        &job->report, host_allocator, &program);
  }
  iree_vm_ref_types_t types = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types);
  }
  iree_vm_function_t function = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(
        loom_runtime_program_process(program), IREE_SV("model"),
        IREE_SV("prepare_weight"), &function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, weights->span_count,
                                         sizeof(*job->preparers),
                                         (void**)&job->preparers);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, weights->span_count,
                                         sizeof(*job->span_preparers),
                                         (void**)&job->span_preparers);
  }
  IREE_TRACE_ZONE_BEGIN_NAMED(z_queries, "loom_serve_weight_query_preparers");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z_queries, weights->span_count);
  for (iree_host_size_t i = 0;
       i < weights->span_count && iree_status_is_ok(status); ++i) {
    loom_runtime_preparation_t* declaration = NULL;
    status = loom_runtime_weight_query_preparer(
        program, &types, function, libraries[0], libraries[1],
        weights->spans[i].parameter, host_allocator, &declaration,
        &weights->spans[i].inputs);
    job->span_preparers[i] = SIZE_MAX;
    if (iree_status_is_ok(status) && !weights->spans[i].inputs.count) {
      status = loom_runtime_weight_inputs_validate_direct(
          weights->inputs, weights->spans[i].parameter.key,
          weights->spans[i].parameter.byte_length);
    }
    if (iree_status_is_ok(status) && declaration) {
      status = loom_runtime_weight_inputs_reserve_staging(
          weights->inputs, weights->spans[i].inputs);
    }
    if (iree_status_is_ok(status) && declaration) {
      iree_host_size_t p = 0;
      const uint32_t binding_count = weights->spans[i].inputs.count ? 2 : 1;
      while (p < job->preparer_count &&
             (binding_count != job->preparers[p].binding_count ||
              !loom_runtime_preparation_equal(declaration,
                                              job->preparers[p].declaration))) {
        ++p;
      }
      if (p == job->preparer_count) {
        loom_runtime_weight_preparer_t* preparer =
            &job->preparers[job->preparer_count++];
        preparer->declaration = declaration;
        preparer->binding_count = binding_count;
        declaration = NULL;
      }
      job->span_preparers[i] = p;
    }
    loom_runtime_preparation_module_recycle(libraries[0], declaration);
  }
  IREE_TRACE_ZONE_END(z_queries);
  IREE_TRACE_ZONE_BEGIN_NAMED(z_release, "loom_serve_weight_release_policy");
  loom_runtime_program_free(program);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(libraries); ++i) {
    iree_vm_module_release(libraries[i]);
  }
  iree_vm_environment_free(environment);
  IREE_TRACE_ZONE_END(z_release);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_runtime_weight_preparation_finish(
    loom_runtime_weight_preparation_t* job, iree_status_t status) {
  loom_runtime_weights_t* weights = job->weights;
  const iree_allocator_t host_allocator = weights->host_allocator;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      const iree_host_size_t p = job->span_preparers[i];
      if (p != SIZE_MAX) {
        weights->spans[i].prepare = job->preparers[p].command;
        iree_hal_command_buffer_retain(weights->spans[i].prepare);
      }
    }
  }
  for (iree_host_size_t i = 0; i < job->preparer_count; ++i) {
    loom_runtime_preparation_free(job->preparers[i].declaration);
    iree_hal_command_buffer_release(job->preparers[i].command);
  }
  iree_allocator_free(host_allocator, job->preparers);
  iree_allocator_free(host_allocator, job->span_preparers);
  const loom_runtime_weights_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(host_allocator, job);
  completion(user_data, status);
}

static void loom_runtime_weight_preparation_backing_ready(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_runtime_weight_preparation_finish(user_data, status);
}

// Each native completion submits the next preparer without a worker join.
static void loom_runtime_weight_preparation_compiled(
    void* user_data, iree_status_t status, loom_runtime_jit_stage_t* compiled);

static void loom_runtime_weight_preparation_next(
    loom_runtime_weight_preparation_t* job) {
  if (job->preparer_index == job->preparer_count) {
    iree_hal_semaphore_t* backing = job->weights->activation.backing_ready;
    if (!backing) {
      loom_runtime_weight_preparation_finish(job, iree_ok_status());
      return;
    }
    job->backing_timepoint.callback =
        loom_runtime_weight_preparation_backing_ready;
    job->backing_timepoint.user_data = job;
    // Completion may run inline and release job before acceptance returns.
    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)backing, 1, &job->backing_timepoint);
    if (!iree_status_is_ok(status)) {
      loom_runtime_weight_preparation_finish(job, status);
    }
    return;
  }
  const loom_runtime_preparation_stage_t* stage =
      loom_runtime_preparation_stage(
          job->preparers[job->preparer_index].declaration, 0);
  iree_status_t status = loom_runtime_jit_compile_async(
      job->jit, job->sink, stage->root, &stage->config, NULL,
      loom_runtime_weight_preparation_compiled, job);
  // An accepted completion may already have destroyed job. Rejection owes no
  // callback and returns the still-exclusive job to this terminal path.
  if (!iree_status_is_ok(status)) {
    loom_runtime_weight_preparation_finish(job, status);
  }
}

static void loom_runtime_weight_preparation_compiled(
    void* user_data, iree_status_t status, loom_runtime_jit_stage_t* compiled) {
  loom_runtime_weight_preparation_t* job = user_data;
  loom_runtime_weight_preparer_t* preparer =
      &job->preparers[job->preparer_index];
  if (iree_status_is_ok(status)) {
    const loom_cmd_program_requirements_t* requirements =
        &loom_runtime_jit_stage_program(compiled)->requirements;
    if (requirements->fixed_buffer_count ||
        requirements->rebindable_binding_count != preparer->binding_count ||
        requirements->transient.required_byte_length ||
        requirements->launch_counts.required_byte_length) {
      const iree_string_view_t name =
          loom_runtime_preparation_stage(preparer->declaration, 0)->root;
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "weight preparer '%.*s' requires %u mutable bindings "
          "and no global scratch",
          (int)name.size, name.data, preparer->binding_count);
    } else {
      status = loom_runtime_jit_stage_record(
          compiled, iree_hal_queue_family(job->weights->dispatch),
          job->command_mode, NULL, NULL, &preparer->command);
    }
  }
  loom_runtime_jit_stage_free(compiled);
  if (iree_status_is_ok(status)) {
    ++job->preparer_index;
    loom_runtime_weight_preparation_next(job);
  } else {
    loom_runtime_weight_preparation_finish(job, status);
  }
}

static void loom_runtime_weight_preparation_execute(
    loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_runtime_weight_preparation_t* job =
      (loom_runtime_weight_preparation_t*)task;
  const iree_allocator_t host_allocator = job->weights->host_allocator;
  iree_status_t status = loom_runtime_weight_inputs_allocate(
      job->parameter, job->indexed_sources, host_allocator,
      &job->weights->inputs);
  if (iree_status_is_ok(status)) {
    status = loom_runtime_weight_prepare_policy(job);
  }
  if (iree_status_is_ok(status) &&
      loom_runtime_weight_inputs_staging_length(job->weights->inputs) >
          UINT64_MAX / WEIGHT_LANES) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "weight input lanes exceed device address space");
  }
  job->status = status;
  IREE_TRACE_ZONE_END(z0);
}

static void loom_runtime_weight_preparation_destroy(loomc_task_t* task) {
  loom_runtime_weight_preparation_t* job =
      (loom_runtime_weight_preparation_t*)task;
  loom_runtime_jit_report_merge(job->jit, &job->report);
  if (iree_status_is_ok(job->status)) {
    loom_runtime_weight_preparation_next(job);
  } else {
    loom_runtime_weight_preparation_finish(job, job->status);
  }
}

static const loomc_task_vtable_t weight_preparation_vtable = {
    .execute = loom_runtime_weight_preparation_execute,
    .destroy = loom_runtime_weight_preparation_destroy,
};

iree_status_t loom_runtime_weights_prepare_async(
    loom_runtime_weights_t* weights, loom_runtime_host_t* host,
    loom_runtime_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_runtime_sources_t* sources,
    const loom_runtime_preparation_parameter_t* parameter,
    loom_runtime_parameter_sources_t* indexed_sources,
    loom_runtime_weights_completion_fn_t completion, void* user_data) {
  loom_runtime_weight_preparation_t* job = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(weights->host_allocator,
                                             sizeof(*job), (void**)&job));
  loomc_task_initialize(&weight_preparation_vtable, &job->task);
  weights->sink = loom_runtime_host_preparation_sink(host);
  job->sink = weights->sink;
  job->weights = weights;
  job->jit = jit;
  job->command_mode = command_mode;
  job->sources = *sources;
  job->parameter = parameter;
  job->indexed_sources = indexed_sources;
  job->completion = completion;
  job->user_data = user_data;
  iree_status_t status =
      iree_status_from_loomc(loomc_task_sink_submit(job->sink, &job->task));
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(weights->host_allocator, job);
  }
  return status;
}

typedef struct loom_runtime_weight_preparation_join_t {
  // Serializes callback publication with the outer owner's final observation.
  iree_slim_mutex_t mutex;
  // Outer application notification, never waited on by a shared worker.
  iree_notification_t notification;
  // Final preparation result, owned by the outer caller after publication.
  iree_status_t status;
  // Terminal callback has published all of its state.
  bool ready;
} loom_runtime_weight_preparation_join_t;

static void loom_runtime_weight_preparation_ready(void* user_data,
                                                  iree_status_t status) {
  loom_runtime_weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  join->status = status;
  join->ready = true;
  iree_notification_post(&join->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&join->mutex);
}

static bool loom_runtime_weight_preparation_is_ready(void* user_data) {
  loom_runtime_weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  const bool ready = join->ready;
  iree_slim_mutex_unlock(&join->mutex);
  return ready;
}

iree_status_t loom_runtime_weights_prepare(
    loom_runtime_weights_t* weights, loom_runtime_host_t* host,
    loom_runtime_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_runtime_sources_t* sources,
    const loom_runtime_preparation_parameter_t* parameter,
    loom_runtime_parameter_sources_t* indexed_sources) {
  loom_runtime_weight_preparation_join_t join = {0};
  iree_slim_mutex_initialize(&join.mutex);
  iree_notification_initialize(&join.notification);
  iree_status_t status = loom_runtime_weights_prepare_async(
      weights, host, jit, command_mode, sources, parameter, indexed_sources,
      loom_runtime_weight_preparation_ready, &join);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&join.notification,
                            loom_runtime_weight_preparation_is_ready, &join,
                            iree_infinite_timeout());
    status = join.status;
  }
  iree_notification_deinitialize(&join.notification);
  iree_slim_mutex_deinitialize(&join.mutex);
  return status;
}

static iree_status_t loom_runtime_weight_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const loom_runtime_weight_direct_read_t* read = user_data;
  *out_key = read->fragments[i].key;
  *out_span = read->fragments[i].range;
  out_span->buffer_offset -= read->target_offset;
  return iree_ok_status();
}

static iree_status_t loom_runtime_weight_input_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const loom_runtime_weight_read_t* read = user_data;
  *out_key = read->input->entry->key;
  *out_span = read->input->span;
  out_span->buffer_offset += read->staging_offset;
  return iree_ok_status();
}

// The shared scope owns raw allocation/pool lifetime independently of the
// plan's readiness. Tracked views cover every payload read and pack dispatch.
static iree_status_t loom_runtime_weight_allocate_staging(
    loom_runtime_weights_t* weights, loom_runtime_workspace_scope_t** out_scope,
    iree_hal_buffer_t** out_buffer, iree_hal_semaphore_t** out_ready) {
  const iree_hal_pool_reservation_request_t request =
      loom_runtime_weights_activation_workspace(weights);
  IREE_RETURN_IF_ERROR(loom_runtime_workspace_reserve(
      weights->workspace, request.allocation_size, request.min_alignment));
  IREE_RETURN_IF_ERROR(loom_runtime_workspace_scope_allocate(weights->workspace,
                                                             true, out_scope));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_ready));
  iree_hal_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      loom_runtime_workspace_scope_acquire(*out_scope, 1, &pool));
  uint64_t value = 1;
  iree_status_t status =
      iree_hal_queue_alloca(weights->dispatch, iree_hal_semaphore_list_empty(),
                            (iree_hal_semaphore_list_t){1, out_ready, &value},
                            pool, 1, &request, out_buffer);
  if (iree_status_is_ok(status)) {
    loom_runtime_workspace_scope_capture(*out_scope, pool, *out_buffer);
    status = loom_runtime_retirement_track(&weights->activation.retirement,
                                           out_buffer, weights->host_allocator);
  } else {
    iree_hal_pool_release(pool);
  }
  return status;
}

// Each lane keeps two bounded submission windows ahead of execution. The older
// window's completion schedules a refill while the newer window can progress.
// The immutable plan and all lane tasks share one activation ownership claim.
enum { WEIGHT_STREAM_WINDOW = 64 };

typedef struct loom_runtime_weight_stream_t loom_runtime_weight_stream_t;

typedef struct loom_runtime_weight_stream_point_t {
  // Borrowed lane timeline retained by the stream owner.
  iree_hal_semaphore_t* semaphore;
  // Last accepted operation in one submission window.
  uint64_t value;
} loom_runtime_weight_stream_point_t;

typedef struct loom_runtime_weight_stream_lane_t {
  // Reused only after its prior execute/destroy ownership has ended.
  loomc_task_t task;
  // Owned reference protecting the immutable plan and tracked buffer views.
  loom_runtime_weight_stream_t* stream;
  // Direct-read lane first, followed by preparation lanes.
  iree_host_size_t ordinal;
  // Next mapping group or tensor in this lane's strided plan traversal.
  iree_host_size_t cursor;
  // Next source run within a direct-read group.
  iree_host_size_t read_index;
  // Next fragment within that source run, or input within a tensor recipe.
  iree_host_size_t fragment_index;
  // Accepted window endpoints, oldest first.
  loom_runtime_weight_stream_point_t windows[2];
  // Initialized endpoints awaiting replenishment.
  iree_host_size_t window_count;
  // Reused completion registration; never overlaps its submitting task.
  iree_async_semaphore_timepoint_t timepoint;
  // Owned submission or predecessor failure, consumed at task destruction.
  iree_status_t status;
} loom_runtime_weight_stream_lane_t;

struct loom_runtime_weight_stream_t {
  // Construction ownership plus one reference for each accepted lane producer.
  iree_atomic_ref_count_t ref_count;
  // Borrowed plan protected by this stream's activation retirement claim.
  loom_runtime_weights_t* weights;
  // Owned mapped read windows and their gather fragments.
  loom_runtime_weight_stream_plan_t plan;
  // Owned release-tracked final-storage views captured by reads and preparers.
  iree_hal_buffer_t** views;
  // Owned release-tracked staging view until every producer has stopped.
  iree_hal_buffer_t* staging;
  // Owned staging allocation readiness captured by input reads.
  iree_hal_semaphore_t* staging_ready;
  // Owned read/preparation timelines, independent of owner-side deactivation.
  iree_hal_semaphore_t* lanes[WEIGHT_COMPLETIONS];
  // Initialized timelines, including partially constructed activations.
  iree_host_size_t lane_count;
  // Owned captured-input timelines shared by each staging lane's reads.
  iree_hal_semaphore_t* input_lanes[WEIGHT_LANES];
  // Reusable bounded producers, independent of model parameter count.
  loom_runtime_weight_stream_lane_t producers[WEIGHT_COMPLETIONS];
};

typedef struct loom_runtime_weight_stream_window_t {
  // Current provider, shared by every pending gather in this window.
  uint32_t source;
  // Number of pending gathers borrowing the descriptor slots below.
  iree_host_size_t gather_count;
  // Input fragments, mapping barriers, and preparation dispatches admitted.
  iree_host_size_t operation_count;
  // Last submitted or pending operation's lane completion.
  loom_runtime_weight_stream_point_t frontier;
  // Bounded capture descriptors, borrowed only until provider submission ends.
  iree_io_parameter_gather_t gathers[WEIGHT_STREAM_WINDOW];
  // Stable enumerator and timepoint storage for each pending gather.
  union {
    // A bounded consecutive slice of one direct-read source run.
    loom_runtime_weight_direct_read_t direct;
    // One captured input placed in its preparation lane's staging allocation.
    loom_runtime_weight_read_t input;
  } reads[WEIGHT_STREAM_WINDOW];
} loom_runtime_weight_stream_window_t;

static void loom_runtime_weight_stream_fail(loom_runtime_weight_stream_t* job,
                                          iree_status_t status) {
  // Failed readiness stops refills. Accepted I/O still owns its tracked views
  // until actual retirement, independently of these terminal notifications.
  for (iree_host_size_t i = 0; i < job->lane_count; ++i) {
    iree_hal_semaphore_fail(job->lanes[i], iree_status_clone(status));
  }
  for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
    if (job->input_lanes[i]) {
      iree_hal_semaphore_fail(job->input_lanes[i], iree_status_clone(status));
    }
  }
}

static void loom_runtime_weight_stream_release(loom_runtime_weight_stream_t* job) {
  if (!job) {
    return;
  }
  if (iree_atomic_ref_count_dec(&job->ref_count) != 1) {
    return;
  }
  loom_runtime_weights_t* weights = job->weights;
  const iree_allocator_t allocator = weights->host_allocator;
  for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
    iree_hal_semaphore_release(job->input_lanes[i]);
  }
  for (iree_host_size_t i = 0; i < job->lane_count; ++i) {
    iree_hal_semaphore_release(job->lanes[i]);
  }
  iree_hal_semaphore_release(job->staging_ready);
  iree_hal_buffer_release(job->staging);
  for (iree_host_size_t i = 0; job->views && i < weights->storage_count; ++i) {
    iree_hal_buffer_release(job->views[i]);
  }
  iree_allocator_free(allocator, job->views);
  loom_runtime_weight_stream_plan_deinitialize(&job->plan, allocator);
  iree_allocator_free(allocator, job);
  loom_runtime_retirement_release(&weights->activation.retirement);
}

static iree_status_t loom_runtime_weight_stream_flush(
    loom_runtime_weight_stream_t* job,
    loom_runtime_weight_stream_window_t* window) {
  if (!window->gather_count) {
    return iree_ok_status();
  }
  iree_status_t status = iree_io_parameter_provider_gather_batch(
      loom_runtime_weight_inputs_provider(job->weights->inputs, window->source),
      job->weights->device, job->weights->transfer, window->gather_count,
      window->gathers);
  window->gather_count = 0;
  return status;
}

static iree_status_t loom_runtime_weight_stream_source(
    loom_runtime_weight_stream_t* job, loom_runtime_weight_stream_window_t* window,
    uint32_t source) {
  if (window->source != source) {
    IREE_RETURN_IF_ERROR(loom_runtime_weight_stream_flush(job, window));
    window->source = source;
  }
  return iree_ok_status();
}

static iree_status_t loom_runtime_weight_stream_direct(
    loom_runtime_weight_stream_lane_t* lane,
    loom_runtime_weight_stream_window_t* window, iree_host_size_t count) {
  loom_runtime_weight_stream_t* job = lane->stream;
  const loom_runtime_weight_group_t* group = &job->plan.groups[lane->cursor];
  iree_hal_semaphore_t** signal = &job->lanes[lane->ordinal];
  if (!group->count) {
    iree_hal_semaphore_t* mapped = job->weights->pool
                                       ? group->mapped
                                       : job->weights->activation.backing_ready;
    iree_hal_semaphore_t* waits[] = {*signal, mapped};
    uint64_t values[] = {group->signal_value - 1, 1};
    uint64_t signal_value = group->signal_value;
    IREE_RETURN_IF_ERROR(iree_hal_queue_barrier(
        job->weights->transfer,
        (iree_hal_semaphore_list_t){mapped ? 2 : 1, waits, values},
        (iree_hal_semaphore_list_t){1, signal, &signal_value}, NULL,
        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    window->frontier =
        (loom_runtime_weight_stream_point_t){*signal, group->signal_value};
    ++window->operation_count;
    lane->cursor += WEIGHT_LANES;
    return iree_ok_status();
  }
  const loom_runtime_weight_direct_read_t* planned =
      &job->plan.reads[group->read_begin + lane->read_index];
  IREE_RETURN_IF_ERROR(loom_runtime_weight_stream_source(
      job, window, planned->fragments[0].source));
  const iree_host_size_t index = window->gather_count++;
  loom_runtime_weight_direct_read_t* read = &window->reads[index].direct;
  *read = *planned;
  read->fragments += lane->fragment_index;
  read->count = count;
  read->wait_values[0] += lane->fragment_index / WEIGHT_STREAM_WINDOW;
  read->signal_value = read->wait_values[0] + 1;
  window->gathers[index] = (iree_io_parameter_gather_t){
      .target_buffer = group->target ? group->target : job->views[group->storage_index],
      .count = count,
      .enumerator = {loom_runtime_weight_enumerate, read},
      .wait_semaphore_list = {read->waits[1] ? 2 : 1, read->waits,
                              read->wait_values},
      .signal_semaphore_list = {1, signal, &read->signal_value},
  };
  window->frontier =
      (loom_runtime_weight_stream_point_t){*signal, read->signal_value};
  window->operation_count += count;
  lane->fragment_index += count;
  if (lane->fragment_index == planned->count) {
    lane->fragment_index = 0;
    if (++lane->read_index == group->read_end - group->read_begin) {
      lane->read_index = 0;
      lane->cursor += WEIGHT_LANES;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_runtime_weight_stream_input(
    loom_runtime_weight_stream_lane_t* lane,
    loom_runtime_weight_stream_window_t* window) {
  loom_runtime_weight_stream_t* job = lane->stream;
  const loom_runtime_weight_span_t* span = &job->weights->spans[lane->cursor];
  loom_runtime_weight_tensor_plan_t tensor = job->plan.spans[lane->cursor];
  const iree_host_size_t ordinal = lane->ordinal - WEIGHT_LANES;
  const loom_runtime_weight_input_t* input = loom_runtime_weight_inputs_at(
      job->weights->inputs, span->inputs.first + lane->fragment_index);
  IREE_RETURN_IF_ERROR(
      loom_runtime_weight_stream_source(job, window, input->source));
  const iree_host_size_t index = window->gather_count++;
  loom_runtime_weight_read_t* read = &window->reads[index].input;
  const uint64_t prior_input =
      tensor.input_signal - span->inputs.count + lane->fragment_index;
  *read = (loom_runtime_weight_read_t){
      .input = input,
      .staging_offset = ordinal * loom_runtime_weight_inputs_staging_length(
                                      job->weights->inputs),
      .waits = {lane->fragment_index ? job->input_lanes[ordinal]
                                     : job->lanes[lane->ordinal],
                job->staging_ready},
      .wait_values = {lane->fragment_index ? prior_input : tensor.prepare_wait,
                      1},
      .signal_value = prior_input + 1,
  };
  window->gathers[index] = (iree_io_parameter_gather_t){
      .target_buffer = job->staging,
      .count = 1,
      .enumerator = {loom_runtime_weight_input_enumerate, read},
      .wait_semaphore_list = {2, read->waits, read->wait_values},
      .signal_semaphore_list = {1, &job->input_lanes[ordinal],
                                &read->signal_value},
  };
  window->frontier = (loom_runtime_weight_stream_point_t){
      job->input_lanes[ordinal], read->signal_value};
  ++window->operation_count;
  ++lane->fragment_index;
  return iree_ok_status();
}

static iree_status_t loom_runtime_weight_stream_prepare(
    loom_runtime_weight_stream_lane_t* lane,
    loom_runtime_weight_stream_window_t* window) {
  loom_runtime_weight_stream_t* job = lane->stream;
  loom_runtime_weights_t* weights = job->weights;
  const loom_runtime_weight_span_t* span = &weights->spans[lane->cursor];
  loom_runtime_weight_tensor_plan_t tensor = job->plan.spans[lane->cursor];
  iree_hal_semaphore_t* waits[WEIGHT_LANES + 2];
  uint64_t values[WEIGHT_LANES + 2];
  iree_host_size_t wait_count = 0;
  // The last contributing window on a read lane proves all earlier windows.
  for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
    if (tensor.begin == tensor.end) {
      continue;
    }
    const iree_host_size_t last = tensor.end - 1;
    const iree_host_size_t distance =
        (last % WEIGHT_LANES + WEIGHT_LANES - i) % WEIGHT_LANES;
    if (last - tensor.begin >= distance) {
      waits[wait_count] = job->lanes[i];
      values[wait_count++] = job->plan.groups[last - distance].signal_value;
    }
  }
  const iree_host_size_t ordinal = lane->ordinal - WEIGHT_LANES;
  if (span->inputs.count) {
    waits[wait_count] = job->input_lanes[ordinal];
    values[wait_count++] = tensor.input_signal;
  }
  waits[wait_count] = job->lanes[lane->ordinal];
  values[wait_count++] = tensor.prepare_wait;
  const iree_hal_buffer_binding_t bindings[] = {
      {job->views[span->storage_index], span->parameter.byte_offset,
       span->parameter.byte_length},
      {job->staging,
       ordinal * loom_runtime_weight_inputs_staging_length(weights->inputs),
       span->inputs.byte_length}};
  IREE_RETURN_IF_ERROR(iree_hal_queue_execute(
      weights->dispatch, (iree_hal_semaphore_list_t){wait_count, waits, values},
      (iree_hal_semaphore_list_t){1, &job->lanes[lane->ordinal],
                                  &tensor.prepare_signal},
      span->prepare,
      (iree_hal_buffer_binding_table_t){span->inputs.count ? 2 : 1, bindings},
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  window->frontier = (loom_runtime_weight_stream_point_t){
      job->lanes[lane->ordinal], tensor.prepare_signal};
  ++window->operation_count;
  lane->cursor += WEIGHT_LANES;
  lane->fragment_index = 0;
  return iree_ok_status();
}

static bool loom_runtime_weight_stream_lane_finished(
    loom_runtime_weight_stream_lane_t* lane) {
  if (lane->ordinal < WEIGHT_LANES) {
    return lane->cursor >= lane->stream->plan.group_count;
  }
  const loom_runtime_weights_t* weights = lane->stream->weights;
  while (lane->cursor < weights->span_count &&
         !weights->spans[lane->cursor].prepare) {
    lane->cursor += WEIGHT_LANES;
  }
  return lane->cursor >= weights->span_count;
}

static iree_status_t loom_runtime_weight_stream_target(
    loom_runtime_weight_stream_lane_t* lane, bool* out_ready) {
  *out_ready = true;
  loom_runtime_weight_stream_t* job = lane->stream;
  loom_runtime_weight_group_t* group = &job->plan.groups[lane->cursor];
  if (!job->weights->pool || group->target || !group->count) {
    return iree_ok_status();
  }
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(group->mapped, &value));
  if (value < 1) {
    *out_ready = false;
    return iree_ok_status();
  }
  const loom_runtime_weight_group_t* mapping =
      &job->plan.groups[group->mapping_group];
  const iree_host_size_t slab_index =
      mapping->backing.offset /
      loom_runtime_virtual_buffer_mapping_size(mapping->backing.buffer);
  const iree_hal_buffer_binding_t span =
      mapping->operation
          ? loom_runtime_memory_operation_backing(mapping->operation, 0)
          : loom_runtime_virtual_buffer_backing(mapping->backing.buffer,
                                                slab_index);
  // File requests retain this coarse I/O view independently of readiness
  // failure. Page ownership itself remains the directory's reservation value.
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_subspan(span.buffer, span.offset, span.length,
                              job->weights->host_allocator, &group->target));
  return loom_runtime_retirement_track(&job->weights->activation.retirement,
                                       &group->target,
                                       job->weights->host_allocator);
}

static void loom_runtime_weight_stream_execute(
    loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_runtime_weight_stream_lane_t* lane = (void*)task;
  bool waiting_for_mapping = false;
  while (iree_status_is_ok(lane->status) && lane->window_count < 2 &&
         !waiting_for_mapping &&
         !loom_runtime_weight_stream_lane_finished(lane)) {
    loom_runtime_weight_stream_window_t window = {0};
    while (iree_status_is_ok(lane->status) &&
           window.operation_count < WEIGHT_STREAM_WINDOW &&
           !loom_runtime_weight_stream_lane_finished(lane)) {
      if (lane->ordinal < WEIGHT_LANES) {
        bool ready = false;
        lane->status = loom_runtime_weight_stream_target(lane, &ready);
        if (!iree_status_is_ok(lane->status) || !ready) {
          waiting_for_mapping = !ready;
          break;
        }
        const loom_runtime_weight_group_t* group =
            &lane->stream->plan.groups[lane->cursor];
        const iree_host_size_t count =
            group->count
                ? iree_min(WEIGHT_STREAM_WINDOW,
                           lane->stream->plan
                                   .reads[group->read_begin + lane->read_index]
                                   .count -
                               lane->fragment_index)
                : 0;
        if (iree_max(1, count) >
            WEIGHT_STREAM_WINDOW - window.operation_count) {
          break;
        }
        lane->status = loom_runtime_weight_stream_direct(lane, &window, count);
      } else if (lane->fragment_index <
                 lane->stream->weights->spans[lane->cursor].inputs.count) {
        lane->status = loom_runtime_weight_stream_input(lane, &window);
      } else {
        lane->status = loom_runtime_weight_stream_prepare(lane, &window);
      }
    }
    if (iree_status_is_ok(lane->status)) {
      lane->status = loom_runtime_weight_stream_flush(lane->stream, &window);
    }
    if (window.operation_count) {
      lane->windows[lane->window_count++] = window.frontier;
    }
  }
  if (iree_status_is_ok(lane->status) && waiting_for_mapping &&
      !lane->window_count) {
    lane->windows[lane->window_count++] = (loom_runtime_weight_stream_point_t){
        lane->stream->plan.groups[lane->cursor].mapped, 1};
  }
  IREE_TRACE_ZONE_END(z0);
}

static const loomc_task_vtable_t weight_stream_vtable;

static void loom_runtime_weight_stream_ready(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_runtime_weight_stream_lane_t* lane = user_data;
  lane->status = status;
  lane->windows[0] = lane->windows[1];
  lane->window_count = 1;
  loomc_task_initialize(&weight_stream_vtable, &lane->task);
  status = iree_status_from_loomc(
      loomc_task_sink_submit(lane->stream->weights->sink, &lane->task));
  // Acceptance may already have executed and destroyed the lane.
  if (!iree_status_is_ok(status)) {
    lane->status = iree_status_join(lane->status, status);
    loomc_task_destroy(&lane->task);
  }
}

static void loom_runtime_weight_stream_destroy(loomc_task_t* task) {
  loom_runtime_weight_stream_lane_t* lane = (void*)task;
  if (!iree_status_is_ok(lane->status)) {
    loom_runtime_weight_stream_fail(lane->stream, lane->status);
    iree_status_free(lane->status);
    loom_runtime_weight_stream_release(lane->stream);
    return;
  }
  if (loom_runtime_weight_stream_lane_finished(lane)) {
    loom_runtime_weight_stream_release(lane->stream);
    return;
  }
  const loom_runtime_weight_stream_point_t point = lane->windows[0];
  lane->timepoint.callback = loom_runtime_weight_stream_ready;
  lane->timepoint.user_data = lane;
  // The submission stack has retired. An immediate callback can safely reuse
  // the task/timepoint or release the whole stream before registration returns.
  iree_status_t status = iree_async_semaphore_acquire_timepoint(
      (iree_async_semaphore_t*)point.semaphore, point.value, &lane->timepoint);
  if (!iree_status_is_ok(status)) {
    loom_runtime_weight_stream_fail(lane->stream, status);
    iree_status_free(status);
    loom_runtime_weight_stream_release(lane->stream);
  }
}

static const loomc_task_vtable_t weight_stream_vtable = {
    .execute = loom_runtime_weight_stream_execute,
    .destroy = loom_runtime_weight_stream_destroy,
};

static iree_status_t loom_runtime_weight_stream(
    loom_runtime_weights_t* weights) {
  const iree_allocator_t allocator = weights->host_allocator;
  loom_runtime_weight_stream_t* job = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*job), (void**)&job));
  iree_atomic_ref_count_init(&job->ref_count);
  job->weights = weights;
  loom_runtime_retirement_acquire(&weights->activation.retirement);
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_runtime_weight_stream_plan_t* plan = &job->plan;
  iree_status_t status = loom_runtime_weight_stream_plan_initialize(
      weights->span_count, weights->spans, weights->storage_count,
      weights->storages, weights->inputs, weights->pool != NULL, allocator,
      plan);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(allocator, weights->storage_count,
                                    sizeof(*job->views), (void**)&job->views);
  }
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    job->views[i] = weights->storages[i].buffer;
    iree_hal_buffer_retain(job->views[i]);
    status = loom_runtime_retirement_track(&weights->activation.retirement,
                                           &job->views[i], allocator);
  }
  const iree_device_size_t staging_stride =
      loom_runtime_weight_inputs_staging_length(weights->inputs);
  loom_runtime_workspace_scope_t* scope = NULL;
  if (staging_stride && iree_status_is_ok(status)) {
    status = loom_runtime_weight_allocate_staging(
        weights, &scope, &job->staging, &job->staging_ready);
  }
  const iree_host_size_t lane_count =
      plan->has_preparation ? WEIGHT_COMPLETIONS : WEIGHT_LANES;
  iree_hal_semaphore_t** lanes = job->lanes;
  uint64_t* frontiers = weights->activation.values;
  for (iree_host_size_t i = 0; i < lane_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_semaphore_create(
        weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &lanes[i]);
    if (iree_status_is_ok(status)) {
      ++job->lane_count;
      weights->activation.semaphores[i] = lanes[i];
      iree_hal_semaphore_retain(lanes[i]);
      ++weights->activation.count;
    }
  }
  for (iree_host_size_t i = 0;
       staging_stride && i < WEIGHT_LANES && iree_status_is_ok(status); ++i) {
    status = iree_hal_semaphore_create(
        weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &job->input_lanes[i]);
  }
  if (iree_status_is_ok(status) && weights->pool) {
    status = iree_allocator_malloc_array(allocator, plan->backing_count,
                                         sizeof(*weights->activation.backing),
                                         (void**)&weights->activation.backing);
    for (iree_host_size_t i = 0;
         i < plan->group_count && iree_status_is_ok(status); ++i) {
      loom_runtime_weight_group_t* group = &plan->groups[i];
      if (group->mapping_group != i) {
        group->mapped = plan->groups[group->mapping_group].mapped;
        iree_hal_semaphore_retain(group->mapped);
        group->operation = plan->groups[group->mapping_group].operation;
        continue;
      }
      status = iree_hal_semaphore_create(
          weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &group->mapped);
      if (iree_status_is_ok(status)) {
        loom_runtime_memory_admission_t admission;
        status = loom_runtime_memory_pool_commit_async(
            weights->pool, 1, &group->backing, group->mapped, 1, &admission,
            &weights->activation.backing[weights->activation.backing_count]);
        ++weights->activation.backing_count;
        if (iree_status_is_ok(status) &&
            admission != LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE) {
          status =
              iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                               "physical pool cannot back model parameters");
        }
        if (iree_status_is_ok(status)) {
          group->operation = weights->activation.backing[weights->activation.backing_count - 1];
        }
      }
    }
  }
  uint64_t read_bytes = 0, read_count = 0, prepared_bytes = 0;
  if (iree_status_is_ok(status)) {
    // Final readiness is known before any lane starts, without materializing
    // its future queue captures. The plan records exact chunk boundaries.
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      loom_runtime_weight_group_t* group = &plan->groups[i];
      const iree_host_size_t lane = i % WEIGHT_LANES;
      iree_hal_semaphore_t* mapped =
          weights->pool ? group->mapped : weights->activation.backing_ready;
      for (iree_host_size_t r = group->read_begin; r < group->read_end; ++r) {
        loom_runtime_weight_direct_read_t* read = &plan->reads[r];
        read->target_offset = weights->pool ? plan->groups[group->mapping_group].backing.offset : 0;
        read->waits[0] = lanes[lane];
        read->wait_values[0] = frontiers[lane];
        read->waits[1] = mapped;
        read->wait_values[1] = 1;
        const iree_host_size_t chunks =
            (read->count - 1) / WEIGHT_STREAM_WINDOW + 1;
        frontiers[lane] += chunks;
        read->signal_value = frontiers[lane];
        read_bytes += read->byte_length;
        read_count += chunks;
      }
      if (!group->count) {
        ++frontiers[lane];
      }
      group->signal_value = frontiers[lane];
    }
    uint64_t input_values[WEIGHT_LANES] = {0};
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      const loom_runtime_weight_span_t* span = &weights->spans[i];
      if (!span->prepare) {
        continue;
      }
      const iree_host_size_t lane = i % WEIGHT_LANES;
      loom_runtime_weight_tensor_plan_t* tensor = &plan->spans[i];
      tensor->prepare_wait = frontiers[WEIGHT_LANES + lane];
      tensor->prepare_signal = ++frontiers[WEIGHT_LANES + lane];
      input_values[lane] += span->inputs.count;
      tensor->input_signal = input_values[lane];
      read_count += span->inputs.count;
      read_bytes += span->inputs.payload_length;
      prepared_bytes += span->parameter.byte_length;
    }
  }
  if (job->staging && iree_status_is_ok(status)) {
    iree_hal_buffer_t* root = iree_hal_buffer_allocated_buffer(job->staging);
    uint64_t value = frontiers[WEIGHT_LANES] + 1;
    status = iree_hal_queue_dealloca(
        weights->dispatch,
        (iree_hal_semaphore_list_t){WEIGHT_LANES, lanes + WEIGHT_LANES,
                                    frontiers + WEIGHT_LANES},
        (iree_hal_semaphore_list_t){1, &lanes[WEIGHT_LANES], &value}, 1, &root);
    if (iree_status_is_ok(status)) {
      frontiers[WEIGHT_LANES] = value;
    }
  }
  loom_runtime_workspace_scope_retire(scope);
  for (iree_host_size_t i = 0; i < lane_count && iree_status_is_ok(status);
       ++i) {
    loom_runtime_weight_stream_lane_t* lane = &job->producers[i];
    lane->stream = job;
    lane->ordinal = i;
    lane->cursor = i % WEIGHT_LANES;
    loomc_task_initialize(&weight_stream_vtable, &lane->task);
    iree_atomic_ref_count_inc(&job->ref_count);
    status = iree_status_from_loomc(
        loomc_task_sink_submit(weights->sink, &lane->task));
    if (!iree_status_is_ok(status)) {
      lane->status = iree_status_clone(status);
      loomc_task_destroy(&lane->task);
    }
  }
  if (iree_status_is_ok(status)) {
    weights->activity.read_bytes += read_bytes;
    weights->activity.groups += read_count;
    weights->activity.prepared_bytes += prepared_bytes;
  } else {
    loom_runtime_weight_stream_fail(job, status);
  }
  // Construction protects the plan while independently accepted lanes start.
  loom_runtime_weight_stream_release(job);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// A monotonic cursor reclaims each cold backing operation once. Resident
// accounting and retirement queries become constant-time after activation.
static bool loom_runtime_weights_collect_backing(
    loom_runtime_weights_t* weights) {
  // Active read lanes borrow immutable completed operation spans. They and
  // their captured I/O views must retire before those results are released.
  if (!loom_runtime_retirement_is_idle(&weights->activation.retirement)) {
    return false;
  }
  while (weights->activation.backing_retired <
         weights->activation.backing_count) {
    loom_runtime_memory_operation_t** operation =
        &weights->activation.backing[weights->activation.backing_retired];
    if (!loom_runtime_memory_operation_is_complete(*operation)) {
      return false;
    }
    weights->activation.backing_status = iree_status_join(
        weights->activation.backing_status,
        loom_runtime_memory_operation_query_status(*operation));
    loom_runtime_memory_operation_free(*operation);
    *operation = NULL;
    ++weights->activation.backing_retired;
  }
  return true;
}

static void loom_runtime_weights_await_backing(
    loom_runtime_weights_t* weights) {
  for (iree_host_size_t i = weights->activation.backing_retired;
       i < weights->activation.backing_count; ++i) {
    loom_runtime_memory_operation_await(weights->activation.backing[i]);
  }
  loom_runtime_weights_collect_backing(weights);
}

static void loom_runtime_weights_backing_completed(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_runtime_weights_t* weights = user_data;
  // The private semaphore preserves the allocation result for streaming waits.
  iree_status_free(status);
  loom_runtime_retirement_release(&weights->activation.retirement);
}

iree_status_t loom_runtime_weights_allocate(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, iree_hal_pool_t* storage_pool,
    loom_runtime_memory_pool_t* pool, loom_runtime_memory_geometry_t geometry,
    loom_runtime_memory_arena_t* arena, loom_runtime_workspace_t* workspace,
    iree_host_size_t shared_root_count, iree_host_size_t root_count,
    const loom_runtime_weight_root_t* roots, iree_allocator_t host_allocator,
    loom_runtime_weights_t** out_weights) {
  *out_weights = NULL;
  loom_runtime_weights_t* weights = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*weights),
                                             (void**)&weights));
  weights->host_allocator = host_allocator;
  weights->device = device;
  weights->transfer = transfer;
  weights->dispatch = dispatch;
  weights->storage_pool = storage_pool;
  weights->pool = pool;
  weights->workspace = workspace;
  loom_runtime_retirement_initialize(&weights->retirement);
  loom_runtime_retirement_initialize(&weights->activation.retirement);
  *out_weights = weights;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t capacity =
      shared_root_count ? roots[0].program->parameters.count : 0;
  for (iree_host_size_t i = shared_root_count; i < root_count; ++i) {
    capacity += roots[i].program->parameters.count;
  }
  iree_status_t status = iree_allocator_malloc_array(host_allocator, capacity,
                                                     sizeof(*weights->spans),
                                                     (void**)&weights->spans);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, root_count,
                                         sizeof(*weights->storages),
                                         (void**)&weights->storages);
  }
  loom_runtime_weight_root_binding_t* bindings = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, root_count,
        sizeof(*bindings) + sizeof(iree_hal_pool_reservation_request_t) +
            sizeof(iree_hal_buffer_t*),
        (void**)&bindings);
  }
  iree_hal_pool_reservation_request_t* requests =
      bindings ? (iree_hal_pool_reservation_request_t*)(bindings + root_count)
               : NULL;
  iree_hal_buffer_t** buffers =
      requests ? (iree_hal_buffer_t**)(requests + root_count) : NULL;
  if (shared_root_count && iree_status_is_ok(status)) {
    loom_cmd_program_parameter_root_t root = roots[0].root;
    for (iree_host_size_t i = 1; i < shared_root_count; ++i) {
      const loom_cmd_program_parameter_root_t other = roots[i].root;
      root.required_byte_length =
          iree_max(root.required_byte_length, other.required_byte_length);
      root.minimum_alignment =
          iree_max(root.minimum_alignment, other.minimum_alignment);
    }
    status = loom_runtime_weight_resolve_root(
        weights, roots[0].program, root, &weights->span_count, weights->spans,
        requests, &bindings[0]);
  }
  for (iree_host_size_t i = shared_root_count;
       i < root_count && iree_status_is_ok(status); ++i) {
    status = loom_runtime_weight_resolve_root(
        weights, roots[i].program, roots[i].root, &weights->span_count,
        weights->spans, requests, &bindings[i]);
  }
  if (iree_status_is_ok(status) && !pool && weights->storage_count) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &weights->activation.backing_ready);
    if (iree_status_is_ok(status)) {
      weights->activation.backing_timepoint.callback =
          loom_runtime_weights_backing_completed;
      weights->activation.backing_timepoint.user_data = weights;
      loom_runtime_retirement_acquire(&weights->activation.retirement);
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)weights->activation.backing_ready, 1,
          &weights->activation.backing_timepoint);
      if (!iree_status_is_ok(status)) {
        loom_runtime_retirement_release(&weights->activation.retirement);
      }
    }
    if (iree_status_is_ok(status)) {
      uint64_t value = 1;
      status = iree_hal_queue_alloca(
          transfer, iree_hal_semaphore_list_empty(),
          (iree_hal_semaphore_list_t){1, &weights->activation.backing_ready,
                                      &value},
          storage_pool, weights->storage_count, requests, buffers);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_fail(weights->activation.backing_ready,
                                iree_status_clone(status));
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    loom_runtime_weight_storage_t* storage = &weights->storages[i];
    if (pool) {
      status = arena ? loom_runtime_virtual_buffer_allocate_in_arena(
                           arena, requests[i].allocation_size,
                           requests[i].min_alignment, &weights->statistics,
                           &storage->reservation)
                     : loom_runtime_virtual_buffer_allocate(
                           pool, geometry, requests[i].allocation_size,
                           requests[i].min_alignment, &weights->statistics,
                           &storage->reservation);
      if (iree_status_is_ok(status)) {
        buffers[i] = loom_runtime_virtual_buffer_handle(storage->reservation);
        iree_hal_buffer_retain(buffers[i]);
      }
    }
    storage->buffer = buffers[i];
    buffers[i] = NULL;
  }
  for (iree_host_size_t i = 0; i < root_count && iree_status_is_ok(status);
       ++i) {
    if (i && i < shared_root_count) {
      *roots[i].buffer = *roots[0].buffer;
      iree_hal_buffer_retain(*roots[i].buffer);
    } else {
      const loom_runtime_weight_root_binding_t binding = bindings[i];
      status = iree_hal_buffer_subspan(
          weights->storages[binding.storage_index].buffer,
          binding.range.byte_offset, binding.range.byte_length, host_allocator,
          roots[i].buffer);
      if (iree_status_is_ok(status)) {
        status = loom_runtime_retirement_track(&weights->retirement,
                                               roots[i].buffer, host_allocator);
      }
    }
  }
  if (bindings) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  iree_allocator_free(host_allocator, bindings);
  if (iree_status_is_ok(status)) {
    iree_host_size_t key_bytes = 0;
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      key_bytes += weights->spans[i].parameter.key.size;
    }
    status = iree_allocator_malloc(host_allocator, key_bytes,
                                   (void**)&weights->keys);
    if (iree_status_is_ok(status)) {
      char* target = weights->keys;
      for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
        iree_string_view_t* key = &weights->spans[i].parameter.key;
        memcpy(target, key->data, key->size);
        key->data = target;
        target += key->size;
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_host_size_t loom_runtime_weights_parameter_count(
    const loom_runtime_weights_t* weights) {
  return weights->span_count;
}

loom_cmd_program_parameter_t loom_runtime_weights_parameter_at(
    const loom_runtime_weights_t* weights, iree_host_size_t index) {
  return weights->spans[index].parameter;
}

iree_hal_pool_reservation_request_t loom_runtime_weights_activation_workspace(
    const loom_runtime_weights_t* weights) {
  return iree_hal_make_pool_reservation_request(
      loom_runtime_weight_inputs_staging_length(weights->inputs) * WEIGHT_LANES,
      loom_runtime_weight_inputs_staging_alignment(weights->inputs));
}

bool loom_runtime_weights_query_activation_workspace(
    loom_runtime_weights_t* weights, uint64_t* out_bytes) {
  bool pending = false;
  if (weights->activation.workspace_bytes) {
    // The producer's final deallocation joins every preparation lane. A
    // failed timeline can publish a sentinel value without freeing storage;
    // its consumer still owns that error and the accepted payloads.
    iree_async_semaphore_t* semaphore =
        (iree_async_semaphore_t*)weights->activation.semaphores[WEIGHT_LANES];
    const uint64_t value = iree_async_semaphore_query(semaphore);
    if (iree_async_semaphore_query_status(semaphore) == IREE_STATUS_OK) {
      pending = value < weights->activation.values[WEIGHT_LANES];
      if (!pending) {
        weights->activation.workspace_bytes = 0;
      }
    }
  }
  *out_bytes = weights->activation.workspace_bytes;
  return pending;
}

loom_runtime_memory_demand_t loom_runtime_weights_missing_demand(
    const loom_runtime_weights_t* weights) {
  loom_runtime_memory_demand_t demand = {0};
  if (weights->pool) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      loom_runtime_memory_demand_add(&demand,
                                     loom_runtime_virtual_buffer_missing_demand(
                                         weights->storages[i].reservation));
    }
  }
  return demand;
}

static iree_status_t loom_runtime_weights_activate_storage(
    loom_runtime_weights_t* weights) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0,
                                   (int64_t)weights->statistics.reserved_bytes);
  weights->activation.workspace_bytes =
      loom_runtime_weights_activation_workspace(weights).allocation_size;
  iree_status_t status = iree_ok_status();
  if (weights->pool && weights->storage_count) {
    loom_runtime_memory_admission_t admission;
    status = loom_runtime_memory_pool_prepare(
        weights->pool, loom_runtime_weights_missing_demand(weights),
        &admission);
    if (iree_status_is_ok(status) &&
        admission != LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "physical pool cannot back model parameters");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_weight_stream(weights);
  }
  if (iree_status_is_ok(status)) {
    weights->active = true;
    ++weights->activity.activations;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_runtime_weights_activate(loom_runtime_weights_t* weights) {
  IREE_ASSERT(!weights->deactivation);
  if (!iree_status_is_ok(weights->activation_status)) {
    return iree_status_clone(weights->activation_status);
  }
  if (!weights->active) {
    weights->activation_status = loom_runtime_weights_activate_storage(weights);
  }
  return iree_status_clone(weights->activation_status);
}

iree_status_t loom_runtime_weights_import_readiness(
    loom_runtime_weights_t* weights, loom_runtime_execution_t* execution) {
  return loom_runtime_execution_barrier(
      execution, (iree_hal_semaphore_list_t){weights->activation.count,
                                             weights->activation.semaphores,
                                             weights->activation.values});
}

iree_status_t loom_runtime_weights_await(loom_runtime_weights_t* weights) {
  iree_status_t status = iree_ok_status();
  // Every accepted lane must retire even if another lane fails. The outer
  // owner also services cold staging requests from those queues.
  for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
    status = iree_status_join(
        status, loom_runtime_workspace_wait(weights->workspace,
                                            weights->activation.semaphores[i],
                                            weights->activation.values[i]));
  }
  status = iree_status_join(
      status, loom_runtime_workspace_await_retirement(
                  weights->workspace, &weights->activation.retirement));
  loom_runtime_weights_await_backing(weights);
  return iree_status_join(
      status, iree_status_clone(weights->activation.backing_status));
}

iree_status_t loom_runtime_weights_deactivate(loom_runtime_weights_t* weights,
                                              bool* out_complete) {
  *out_complete = false;
  if (!weights->pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fixed parameter backing cannot deactivate");
  }
  if (weights->deactivation) {
    if (loom_runtime_memory_operation_is_complete(weights->deactivation)) {
      iree_status_t status =
          loom_runtime_memory_operation_query_status(weights->deactivation);
      loom_runtime_memory_operation_free(weights->deactivation);
      weights->deactivation = NULL;
      *out_complete = true;
      return status;
    }
    return iree_ok_status();
  }
  if (!weights->active && !weights->activation.backing_count) {
    *out_complete = true;
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)weights->statistics.committed_bytes);
  const bool backing_complete = loom_runtime_weights_collect_backing(weights);
  IREE_ASSERT(backing_complete);
  loom_runtime_virtual_buffer_t** reservations = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      weights->host_allocator, weights->storage_count, sizeof(*reservations),
      (void**)&reservations);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      reservations[i] = weights->storages[i].reservation;
      loom_runtime_virtual_buffer_begin_trim(reservations[i]);
    }
    status = loom_runtime_memory_pool_trim_async(
        weights->pool, weights->storage_count, reservations, NULL, 0,
        &weights->deactivation);
  }
  iree_allocator_free(weights->host_allocator, reservations);
  if (iree_status_is_ok(status)) {
    weights->activity.deactivations += weights->active;
    weights->active = false;
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_release(weights->activation.semaphores[i]);
      weights->activation.semaphores[i] = NULL;
      weights->activation.values[i] = 0;
    }
    weights->activation.count = 0;
    weights->activation.workspace_bytes = 0;
    iree_status_free(weights->activation.backing_status);
    weights->activation.backing_status = iree_ok_status();
    iree_allocator_free(weights->host_allocator, weights->activation.backing);
    weights->activation.backing = NULL;
    weights->activation.backing_count = 0;
    weights->activation.backing_retired = 0;
    iree_hal_semaphore_release(weights->activation.backing_ready);
    weights->activation.backing_ready = NULL;
    *out_complete = !weights->deactivation;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

loom_runtime_storage_statistics_t loom_runtime_weights_statistics(
    loom_runtime_weights_t* weights) {
  // Accounting is published by the serialized owner, never the cold executor.
  loom_runtime_weights_collect_backing(weights);
  loom_runtime_memory_operation_is_complete(weights->deactivation);
  return weights->statistics;
}

loom_runtime_storage_usage_t loom_runtime_weights_memory_usage(
    const loom_runtime_weights_t* weights) {
  return (loom_runtime_storage_usage_t){
      .reserved_bytes = weights->statistics.reserved_bytes,
      .committed_bytes = weights->statistics.committed_bytes,
      .pending_bytes = weights->statistics.pending_bytes,
      .releasing_bytes = weights->statistics.releasing_bytes,
      .released_bytes = weights->statistics.released_bytes,
  };
}

loom_runtime_parameter_activity_t loom_runtime_weights_activity(
    const loom_runtime_weights_t* weights) {
  return weights->activity;
}

void loom_runtime_weights_set_retirement_observer(
    loom_runtime_weights_t* weights, void (*wake)(void* user_data),
    void* user_data) {
  if (weights) {
    for (iree_host_size_t i = weights->activation.backing_retired;
         i < weights->activation.backing_count; ++i) {
      loom_runtime_memory_operation_set_observer(weights->activation.backing[i],
                                                 wake, user_data);
    }
    loom_runtime_memory_operation_set_observer(weights->deactivation, wake,
                                               user_data);
    loom_runtime_retirement_set_observer(&weights->retirement, wake, user_data);
    loom_runtime_retirement_set_observer(&weights->activation.retirement, wake,
                                         user_data);
  }
}

bool loom_runtime_weights_is_retired(loom_runtime_weights_t* weights) {
  return !weights ||
         (loom_runtime_weights_collect_backing(weights) &&
          loom_runtime_memory_operation_is_complete(weights->deactivation) &&
          loom_runtime_retirement_is_idle(&weights->retirement) &&
          loom_runtime_retirement_is_idle(&weights->activation.retirement));
}

iree_status_t loom_runtime_weights_await_retirement(
    loom_runtime_weights_t* weights) {
  if (!weights) {
    return iree_ok_status();
  }
  loom_runtime_weights_await_backing(weights);
  loom_runtime_memory_operation_await(weights->deactivation);
  iree_status_t status = loom_runtime_workspace_await_retirement(
      weights->workspace, &weights->retirement);
  return iree_status_join(
      status, loom_runtime_workspace_await_retirement(
                  weights->workspace, &weights->activation.retirement));
}

iree_status_t loom_runtime_weights_free(loom_runtime_weights_t* weights) {
  if (!weights) {
    return iree_ok_status();
  }
  const bool retired = loom_runtime_weights_is_retired(weights);
  IREE_ASSERT(retired);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    status = loom_runtime_virtual_buffer_free(weights->storages[i].reservation);
    if (iree_status_is_ok(status)) {
      weights->storages[i].reservation = NULL;
      iree_hal_buffer_release(weights->storages[i].buffer);
      weights->storages[i].buffer = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_status_free(weights->activation.backing_status);
    iree_allocator_free(weights->host_allocator, weights->activation.backing);
    loom_runtime_memory_operation_free(weights->deactivation);
    iree_hal_semaphore_release(weights->activation.backing_ready);
    loom_runtime_retirement_deinitialize(&weights->retirement);
    loom_runtime_retirement_deinitialize(&weights->activation.retirement);
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_release(weights->activation.semaphores[i]);
    }
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      iree_hal_command_buffer_release(weights->spans[i].prepare);
    }
    loom_runtime_weight_inputs_free(weights->inputs);
    iree_status_ignore(weights->activation_status);
    iree_allocator_free(weights->host_allocator, weights->storages);
    iree_allocator_free(weights->host_allocator, weights->keys);
    iree_allocator_free(weights->host_allocator, weights->spans);
    iree_allocator_free(weights->host_allocator, weights);
  }
  return status;
}
