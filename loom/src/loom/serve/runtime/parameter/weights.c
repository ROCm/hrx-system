// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/weights.h"

#include <string.h>

#include "iree/async/semaphore.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/tooling/parameter_util.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/program/program.h"
#include "loomc/iree.h"

enum { WEIGHT_LANES = 4, WEIGHT_COMPLETIONS = 2 * WEIGHT_LANES };

typedef struct loom_serve_weight_span_t {
  // Borrowed key and final byte placement from the compiled command.
  loom_cmd_program_parameter_t parameter;
  // Unique final storage index, resolved once before any queue work.
  iree_host_size_t storage_index;
  // Owned cached in-place preparation command, or null for unchanged bytes.
  iree_hal_command_buffer_t* prepare;
} loom_serve_weight_span_t;

typedef struct loom_serve_weight_fragment_t {
  // Borrowed checkpoint identity from the retained parameter plan.
  iree_string_view_t key;
  // File and target byte extents, possibly only part of a tensor.
  iree_io_parameter_span_t range;
} loom_serve_weight_fragment_t;

typedef struct loom_serve_weight_group_t {
  // Borrowed fragments in this group's mapped storage range.
  loom_serve_weight_fragment_t* fragments;
  // Unique storage containing every fragment in this group.
  iree_host_size_t storage_index;
  // Number of fragments exposed to the parameter provider.
  iree_host_size_t count;
  // One physical slab for elastic backing; empty for fixed backing.
  loom_serve_memory_range_t backing;
  // Owned mapping readiness, released after accepted reads capture it.
  iree_hal_semaphore_t* mapped;
  // Previous read on this lane and optional mapping readiness.
  iree_hal_semaphore_t* waits[2];
  // Values paired with waits.
  uint64_t wait_values[2];
  // Read frontier established by this group.
  uint64_t signal_value;
} loom_serve_weight_group_t;

typedef struct loom_serve_weight_group_range_t {
  // First read group contributing bytes to a tensor.
  iree_host_size_t begin;
  // Exclusive last read group contributing bytes to that tensor.
  iree_host_size_t end;
} loom_serve_weight_group_range_t;

typedef struct loom_serve_weight_stream_plan_t {
  // Owned read groups, in physical mapping order for elastic backing.
  loom_serve_weight_group_t* groups;
  // Number of initialized groups, including padding-only slabs.
  iree_host_size_t group_count;
  // Owned contiguous gather fragments borrowed by groups.
  loom_serve_weight_fragment_t* fragments;
  // Owned tensor-to-group intervals consumed by preparation submission.
  loom_serve_weight_group_range_t* spans;
  // Exact checkpoint payload, excluding placement padding.
  uint64_t read_bytes;
  // At least one tensor requires a preparation dispatch.
  bool has_preparation;
} loom_serve_weight_stream_plan_t;

typedef struct loom_serve_weight_storage_t {
  // Owned unique root retained independently of the caller's output views.
  iree_hal_buffer_t* buffer;
  // Owned virtual reservation, or NULL for explicitly fixed backing.
  loom_serve_virtual_buffer_t* reservation;
} loom_serve_weight_storage_t;

struct loom_serve_weights_t {
  // Allocator owning the plan and cold activation metadata.
  iree_allocator_t host_allocator;
  // Borrowed device enclosing the complete plan lifetime.
  iree_hal_device_t* device;
  // Borrowed native pool for fixed roots; outlives all resolved storage.
  iree_hal_pool_t* storage_pool;
  // Borrowed exact file transfer queue.
  iree_hal_queue_t* transfer;
  // Borrowed exact in-place preparation queue.
  iree_hal_queue_t* dispatch;
  // Borrowed shared physical budget, or NULL for fixed backing.
  loom_serve_memory_pool_t* pool;
  // Parameter-only accounting across all unique roots.
  loom_serve_memory_statistics_t statistics;
  // Accepted streaming/preparation work, never reconstructed from events.
  loom_serve_weight_activity_t activity;
  // Final consumer ownership before virtual roots can be unmapped at teardown.
  loom_serve_retirement_t retirement;
  // Activation lanes retained independently of their submitting host turn.
  struct {
    // Owned window operations, joined independently of queue readiness.
    loom_serve_memory_operation_t** backing;
    // Accepted window operations, including partially failed activation.
    iree_host_size_t backing_count;
    // Operations before this cursor have published results and been freed.
    iree_host_size_t backing_retired;
    // Owned aggregate backing result, retained after operation reclamation.
    iree_status_t backing_status;
    // Owned fixed-allocation readiness imported by the file-loading lanes.
    iree_hal_semaphore_t* backing_ready;
    // Private fixed-allocation completion, independent of streaming failures.
    iree_async_semaphore_timepoint_t backing_timepoint;
    // Accepted I/O/preparation views, including after failed readiness.
    loom_serve_retirement_t retirement;
    // Owned lane readiness handles imported into the consuming model timeline.
    iree_hal_semaphore_t* semaphores[WEIGHT_COMPLETIONS];
    // Last submitted read or preparation on each initialized lane.
    uint64_t values[WEIGHT_COMPLETIONS];
    // Initialized handles, including a partially failed activation.
    iree_host_size_t count;
  } activation;
  // Native release borrower, joined before reactivation or destruction.
  loom_serve_memory_operation_t* deactivation;
  // Owned original checkpoint provider and imported-file cache.
  iree_io_parameter_provider_t* provider;
  // Owned placement/preparation records, with keys in the owned key storage.
  loom_serve_weight_span_t* spans;
  // Owned parameter key bytes, independent of source/JIT lifetime.
  char* keys;
  // Number of initialized placement records, including partial creation.
  iree_host_size_t span_count;
  // Owned unique root records; aliasing output slots do not duplicate these.
  loom_serve_weight_storage_t* storages;
  // Number of initialized unique roots, including partial creation.
  iree_host_size_t storage_count;
  // Backing and all file/preparation work were submitted for this activation.
  bool active;
  // Owned terminal producer failure, replayed to later consumers without I/O.
  iree_status_t activation_status;
};

static const loom_serve_weight_span_t* loom_serve_weight_find(
    iree_host_size_t count, const loom_serve_weight_span_t* spans,
    iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(spans[i].parameter.key, key)) {
      return &spans[i];
    }
  }
  return NULL;
}

typedef struct loom_serve_weight_root_binding_t {
  // Unique storage allocation selected by parameter identity and placement.
  iree_host_size_t storage_index;
  // Byte range of this root within the selected storage allocation.
  iree_hal_buffer_range_t range;
} loom_serve_weight_root_binding_t;

// Resolves root sharing before submitting any I/O. Only newly allocated roots
// append spans, so tensors shared by multiple stages are read and prepared
// exactly once.
static iree_status_t loom_serve_weight_resolve_root(
    loom_serve_weights_t* weights, const loom_cmd_program_t* program,
    loom_cmd_program_parameter_root_t root, iree_host_size_t* span_count,
    loom_serve_weight_span_t* spans,
    iree_hal_pool_reservation_request_t* requests,
    loom_serve_weight_root_binding_t* out_binding) {
  const iree_host_size_t prior_count = *span_count;
  const loom_serve_weight_span_t* shared = NULL;
  iree_device_size_t origin = 0;
  iree_host_size_t count = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    const loom_serve_weight_span_t* existing =
        loom_serve_weight_find(prior_count, spans, parameter.key);
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
    *out_binding = (loom_serve_weight_root_binding_t){
        shared->storage_index, {origin, root.required_byte_length}};
    return iree_ok_status();
  }
  const iree_host_size_t storage_index = weights->storage_count++;
  requests[storage_index] = iree_hal_make_pool_reservation_request(
      root.required_byte_length, root.minimum_alignment);
  *out_binding = (loom_serve_weight_root_binding_t){
      storage_index, {0, root.required_byte_length}};
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    spans[(*span_count)++] = (loom_serve_weight_span_t){
        .parameter = parameter,
        .storage_index = storage_index,
    };
  }
  return iree_ok_status();
}

// A cached preparer owns the returned VM string until cold planning completes.
typedef struct loom_serve_weight_preparer_t {
  // Owned VM result backing the root name, independent of the invocation.
  iree_vm_buffer_t* root;
  // Owned recorded command retaining its executables independently of the JIT.
  iree_hal_command_buffer_t* command;
} loom_serve_weight_preparer_t;

typedef struct loom_serve_weight_preparation_t {
  // Initial CPU task; its destruction transfers ownership to the JIT chain.
  loomc_task_t task;
  // Exclusively borrowed unactivated plan through terminal completion.
  loom_serve_weights_t* weights;
  // Exclusively borrowed source catalog through terminal completion.
  loom_serve_jit_t* jit;
  // Recording policy applied to each source-selected preparation command.
  iree_hal_command_buffer_mode_t command_mode;
  // Borrowed checkpoint files, retained by the load owner through completion.
  iree_string_view_list_t weights_paths;
  // Copied namespace descriptor borrowing its strings and package through
  // completion. Checkpoints use its filesystem base; policy uses its package.
  loom_serve_sources_t sources;
  // Borrowed source policy locator with the same lifetime.
  iree_string_view_t policy_path;
  // Owned distinct preparers, queried once before native compilation.
  loom_serve_weight_preparer_t* preparers;
  // Number of owned preparers, including partial policy failure.
  iree_host_size_t preparer_count;
  // Current native compilation ordinal; equals count at successful completion.
  iree_host_size_t preparer_index;
  // Owned per-span preparer ordinal; SIZE_MAX preserves canonical file bytes.
  iree_host_size_t* span_preparers;
  // Stable source configuration borrowed by the current native compile.
  loomc_config_options_t config;
  // CPU policy/indexing outcome, consumed by the initial task destructor.
  iree_status_t status;
  // Joins the already-queued fixed root allocation before publication.
  iree_async_semaphore_timepoint_t backing_timepoint;
  // Terminal ownership transfer, invoked after all job storage is released.
  loom_serve_weights_completion_fn_t completion;
  // Borrowed terminal callback payload.
  void* user_data;
} loom_serve_weight_preparation_t;

static iree_status_t loom_serve_weight_query_preparer(
    loom_serve_program_t* program, const iree_vm_ref_types_t* types,
    iree_vm_function_t function, loom_cmd_program_parameter_t parameter,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_root) {
  *out_root = NULL;
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)parameter.key.data, parameter.key.size),
      iree_vm_buffer_release_callback_null(), host_allocator, &key));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(types, &key)};
  iree_vm_variant_t results[2] = {0};
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(program), function,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  iree_vm_buffer_t* root = NULL;
  int64_t byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(types, &results[0], &root);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &byte_length);
  }
  if (iree_status_is_ok(status)) {
    if (!root || !iree_all_bits_set(iree_vm_buffer_access(root),
                                    IREE_VM_BUFFER_ACCESS_FLAG_READ)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "weight policy must return a readable root");
    } else if (!iree_vm_buffer_length(root) && byte_length != 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unchanged weight policy must return zero byte length");
    } else if (iree_vm_buffer_length(root) &&
               (byte_length <= 0 ||
                (uint64_t)byte_length != parameter.byte_length)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "weight policy for '%.*s' requires %" PRId64
                                " bytes, reflected parameter has %" PRIu64,
                                (int)parameter.key.size, parameter.key.data,
                                byte_length, (uint64_t)parameter.byte_length);
    }
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    *out_root = root;
  } else {
    iree_vm_buffer_release(root);
  }
  return status;
}

static iree_string_view_t loom_serve_weight_preparer_name(
    iree_vm_buffer_t* root) {
  return iree_make_string_view(iree_vm_buffer_const_data(root),
                               iree_vm_buffer_length(root));
}

static iree_status_t loom_serve_weight_prepare_policy(
    loom_serve_weight_preparation_t* job) {
  loom_serve_weights_t* weights = job->weights;
  const iree_allocator_t host_allocator = weights->host_allocator;
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(host_allocator, &environment));
  loom_serve_program_t* program = NULL;
  const iree_string_view_t roots[] = {IREE_SVL("prepare_weight")};
  iree_status_t status = loom_serve_program_allocate(
      loom_serve_jit_target_environment(job->jit), environment, &job->sources,
      job->policy_path, IREE_ARRAYSIZE(roots), roots,
      iree_vm_module_span_empty(), loom_serve_jit_report(job->jit),
      host_allocator, &program);
  iree_vm_ref_types_t types = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types);
  }
  iree_vm_function_t function = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(
        loom_serve_program_process(program), IREE_SV("model"),
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
  for (iree_host_size_t i = 0;
       i < weights->span_count && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* root = NULL;
    status = loom_serve_weight_query_preparer(program, &types, function,
                                              weights->spans[i].parameter,
                                              host_allocator, &root);
    job->span_preparers[i] = SIZE_MAX;
    if (iree_status_is_ok(status) && iree_vm_buffer_length(root)) {
      const iree_string_view_t name = loom_serve_weight_preparer_name(root);
      iree_host_size_t p = 0;
      while (p < job->preparer_count &&
             !iree_string_view_equal(name, loom_serve_weight_preparer_name(
                                               job->preparers[p].root))) {
        ++p;
      }
      if (p == job->preparer_count) {
        loom_serve_weight_preparer_t* preparer =
            &job->preparers[job->preparer_count++];
        preparer->root = root;
        root = NULL;
      }
      job->span_preparers[i] = p;
    }
    iree_vm_buffer_release(root);
  }
  loom_serve_program_free(program);
  iree_vm_environment_free(environment);
  return status;
}

static void loom_serve_weight_preparation_finish(
    loom_serve_weight_preparation_t* job, iree_status_t status) {
  loom_serve_weights_t* weights = job->weights;
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
    iree_vm_buffer_release(job->preparers[i].root);
    iree_hal_command_buffer_release(job->preparers[i].command);
  }
  iree_allocator_free(host_allocator, job->preparers);
  iree_allocator_free(host_allocator, job->span_preparers);
  const loom_serve_weights_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(host_allocator, job);
  completion(user_data, status);
}

static void loom_serve_weight_preparation_backing_ready(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_serve_weight_preparation_finish(user_data, status);
}

// Each native completion submits the next preparer without a worker join.
static void loom_serve_weight_preparation_compiled(
    void* user_data, iree_status_t status, loom_serve_jit_stage_t* compiled);

static void loom_serve_weight_preparation_next(
    loom_serve_weight_preparation_t* job) {
  if (job->preparer_index == job->preparer_count) {
    iree_hal_semaphore_t* backing = job->weights->activation.backing_ready;
    if (!backing) {
      loom_serve_weight_preparation_finish(job, iree_ok_status());
      return;
    }
    job->backing_timepoint.callback =
        loom_serve_weight_preparation_backing_ready;
    job->backing_timepoint.user_data = job;
    // Completion may run inline and release job before acceptance returns.
    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)backing, 1, &job->backing_timepoint);
    if (!iree_status_is_ok(status)) {
      loom_serve_weight_preparation_finish(job, status);
    }
    return;
  }
  const iree_string_view_t root =
      loom_serve_weight_preparer_name(job->preparers[job->preparer_index].root);
  iree_status_t status =
      loom_serve_jit_compile_async(job->jit, root, &job->config,
                                   loom_serve_weight_preparation_compiled, job);
  // An accepted completion may already have destroyed job. Rejection owes no
  // callback and returns the still-exclusive job to this terminal path.
  if (!iree_status_is_ok(status)) {
    loom_serve_weight_preparation_finish(job, status);
  }
}

static void loom_serve_weight_preparation_compiled(
    void* user_data, iree_status_t status, loom_serve_jit_stage_t* compiled) {
  loom_serve_weight_preparation_t* job = user_data;
  loom_serve_weight_preparer_t* preparer = &job->preparers[job->preparer_index];
  if (iree_status_is_ok(status)) {
    const loom_cmd_program_requirements_t* requirements =
        &loom_serve_jit_stage_program(compiled)->requirements;
    if (requirements->fixed_buffer_count ||
        requirements->rebindable_binding_count != 1 ||
        requirements->transient.required_byte_length ||
        requirements->launch_counts.required_byte_length) {
      const iree_string_view_t name =
          loom_serve_weight_preparer_name(preparer->root);
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "weight preparer '%.*s' requires one mutable binding "
          "and no global scratch",
          (int)name.size, name.data);
    } else {
      status = loom_serve_jit_stage_record(
          compiled, iree_hal_queue_family(job->weights->dispatch),
          job->command_mode, NULL, NULL, &preparer->command);
    }
  }
  loom_serve_jit_stage_free(compiled);
  if (iree_status_is_ok(status)) {
    ++job->preparer_index;
    loom_serve_weight_preparation_next(job);
  } else {
    loom_serve_weight_preparation_finish(job, status);
  }
}

static void loom_serve_weight_preparation_execute(
    loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  loom_serve_weight_preparation_t* job = (loom_serve_weight_preparation_t*)task;
  const iree_allocator_t host_allocator = job->weights->host_allocator;
  iree_status_t status = loom_serve_weight_prepare_policy(job);
  iree_io_parameter_index_t* index = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(host_allocator, &index);
  }
  for (iree_host_size_t i = 0;
       i < job->weights_paths.count && iree_status_is_ok(status); ++i) {
    char* weights_path = NULL;
    status = loom_serve_path_resolve(job->sources.base_directory,
                                     job->weights_paths.values[i],
                                     host_allocator, &weights_path);
    if (iree_status_is_ok(status)) {
      status = iree_tooling_append_parameter_file_to_index(
          iree_make_cstring_view(weights_path), index, host_allocator);
    }
    iree_allocator_free(host_allocator, weights_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        iree_string_view_empty(), index, WEIGHT_LANES, host_allocator,
        &job->weights->provider);
  }
  iree_io_parameter_index_release(index);
  job->status = status;
}

static void loom_serve_weight_preparation_destroy(loomc_task_t* task) {
  loom_serve_weight_preparation_t* job = (loom_serve_weight_preparation_t*)task;
  if (iree_status_is_ok(job->status)) {
    loom_serve_weight_preparation_next(job);
  } else {
    loom_serve_weight_preparation_finish(job, job->status);
  }
}

static const loomc_task_vtable_t weight_preparation_vtable = {
    .execute = loom_serve_weight_preparation_execute,
    .destroy = loom_serve_weight_preparation_destroy,
};

iree_status_t loom_serve_weights_prepare_async(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources, iree_string_view_list_t weights_paths,
    iree_string_view_t policy_path,
    loom_serve_weights_completion_fn_t completion, void* user_data) {
  loom_serve_weight_preparation_t* job = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(weights->host_allocator,
                                             sizeof(*job), (void**)&job));
  loomc_task_initialize(&weight_preparation_vtable, &job->task);
  job->weights = weights;
  job->jit = jit;
  job->command_mode = command_mode;
  job->sources = *sources;
  job->weights_paths = weights_paths;
  job->policy_path = policy_path;
  job->config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  job->completion = completion;
  job->user_data = user_data;
  iree_status_t status = iree_status_from_loomc(
      loomc_task_sink_submit(loom_serve_host_compiler_sink(host), &job->task));
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(weights->host_allocator, job);
  }
  return status;
}

typedef struct loom_serve_weight_preparation_join_t {
  // Serializes callback publication with the outer owner's final observation.
  iree_slim_mutex_t mutex;
  // Outer application notification, never waited on by a shared worker.
  iree_notification_t notification;
  // Final preparation result, owned by the outer caller after publication.
  iree_status_t status;
  // Terminal callback has published all of its state.
  bool ready;
} loom_serve_weight_preparation_join_t;

static void loom_serve_weight_preparation_ready(void* user_data,
                                                iree_status_t status) {
  loom_serve_weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  join->status = status;
  join->ready = true;
  iree_notification_post(&join->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&join->mutex);
}

static bool loom_serve_weight_preparation_is_ready(void* user_data) {
  loom_serve_weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  const bool ready = join->ready;
  iree_slim_mutex_unlock(&join->mutex);
  return ready;
}

iree_status_t loom_serve_weights_prepare(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources, iree_string_view_list_t weights_paths,
    iree_string_view_t policy_path) {
  loom_serve_weight_preparation_join_t join = {0};
  iree_slim_mutex_initialize(&join.mutex);
  iree_notification_initialize(&join.notification);
  iree_status_t status = loom_serve_weights_prepare_async(
      weights, host, jit, command_mode, sources, weights_paths, policy_path,
      loom_serve_weight_preparation_ready, &join);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&join.notification,
                            loom_serve_weight_preparation_is_ready, &join,
                            iree_infinite_timeout());
    status = join.status;
  }
  iree_notification_deinitialize(&join.notification);
  iree_slim_mutex_deinitialize(&join.mutex);
  return status;
}

// Builds only cold activation metadata. Elastic groups are physical slab
// windows; fixed groups preserve consecutive canonical tensors and individual
// preparers. Tensor-to-group intervals are recorded here and consumed without
// rescanning.
static iree_status_t loom_serve_weight_stream_plan(
    loom_serve_weights_t* weights, loom_serve_weight_stream_plan_t* plan) {
  const iree_allocator_t allocator = weights->host_allocator;
  iree_host_size_t* storage_groups = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, weights->storage_count, sizeof(*storage_groups),
      (void**)&storage_groups));
  iree_status_t status =
      iree_allocator_malloc_array(allocator, weights->span_count,
                                  sizeof(*plan->spans), (void**)&plan->spans);
  const iree_device_size_t slab_size =
      weights->pool ? loom_serve_memory_pool_slab_size(weights->pool) : 0;
  if (iree_status_is_ok(status)) {
    if (weights->pool) {
      for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
        storage_groups[i] = plan->group_count;
        plan->group_count +=
            iree_hal_buffer_byte_length(weights->storages[i].buffer) /
            slab_size;
      }
    }
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      const loom_serve_weight_span_t* span = &weights->spans[i];
      plan->read_bytes += span->parameter.byte_length;
      plan->has_preparation |= span->prepare != NULL;
      if (weights->pool) {
        const iree_host_size_t base = storage_groups[span->storage_index];
        plan->spans[i] = (loom_serve_weight_group_range_t){
            base + span->parameter.byte_offset / slab_size,
            base + (span->parameter.byte_offset + span->parameter.byte_length +
                    slab_size - 1) /
                       slab_size};
        if (!span->parameter.byte_length) {
          // Empty tensors still resolve their checkpoint key. At the end of a
          // root, the zero-byte read belongs to its final mapped window.
          plan->spans[i].begin =
              base +
              iree_min(span->parameter.byte_offset / slab_size,
                       iree_hal_buffer_byte_length(
                           weights->storages[span->storage_index].buffer) /
                               slab_size -
                           1);
          plan->spans[i].end = plan->spans[i].begin + 1;
        }
      } else {
        if (!i || span->prepare || weights->spans[i - 1].prepare ||
            span->storage_index != weights->spans[i - 1].storage_index) {
          ++plan->group_count;
        }
        plan->spans[i] = (loom_serve_weight_group_range_t){
            plan->group_count - 1, plan->group_count};
      }
    }
    status = iree_allocator_malloc_array(allocator, plan->group_count,
                                         sizeof(*plan->groups),
                                         (void**)&plan->groups);
  }
  iree_host_size_t fragment_count = 0;
  if (iree_status_is_ok(status)) {
    if (weights->pool) {
      for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
        const iree_device_size_t length =
            iree_hal_buffer_byte_length(weights->storages[i].buffer);
        for (iree_device_size_t offset = 0; offset < length;
             offset += slab_size) {
          loom_serve_weight_group_t* group =
              &plan->groups[storage_groups[i] + offset / slab_size];
          group->storage_index = i;
          group->backing = (loom_serve_memory_range_t){
              weights->storages[i].reservation, offset, slab_size};
        }
      }
    }
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      const loom_serve_weight_group_range_t range = plan->spans[i];
      for (iree_host_size_t j = range.begin; j < range.end; ++j) {
        plan->groups[j].storage_index = weights->spans[i].storage_index;
        ++plan->groups[j].count;
        ++fragment_count;
      }
    }
    status = iree_allocator_malloc_array(allocator, fragment_count,
                                         sizeof(*plan->fragments),
                                         (void**)&plan->fragments);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t cursor = 0;
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      plan->groups[i].fragments =
          plan->fragments ? plan->fragments + cursor : NULL;
      cursor += plan->groups[i].count;
      plan->groups[i].count = 0;
    }
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      const loom_cmd_program_parameter_t parameter =
          weights->spans[i].parameter;
      const loom_serve_weight_group_range_t range = plan->spans[i];
      for (iree_host_size_t j = range.begin; j < range.end; ++j) {
        loom_serve_weight_group_t* group = &plan->groups[j];
        const iree_device_size_t begin =
            weights->pool
                ? iree_max(parameter.byte_offset, group->backing.offset)
                : parameter.byte_offset;
        const iree_device_size_t end =
            weights->pool
                ? iree_min(parameter.byte_offset + parameter.byte_length,
                           group->backing.offset + group->backing.length)
                : parameter.byte_offset + parameter.byte_length;
        group->fragments[group->count++] = (loom_serve_weight_fragment_t){
            .key = parameter.key,
            .range = {.parameter_offset = begin - parameter.byte_offset,
                      .buffer_offset = begin,
                      .length = end - begin},
        };
      }
    }
  }
  iree_allocator_free(allocator, storage_groups);
  return status;
}

static iree_status_t loom_serve_weight_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const loom_serve_weight_group_t* group = user_data;
  *out_key = group->fragments[i].key;
  *out_span = group->fragments[i].range;
  return iree_ok_status();
}

static iree_status_t loom_serve_weight_stream(loom_serve_weights_t* weights) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_allocator_t host_allocator = weights->host_allocator;
  loom_serve_weight_stream_plan_t plan = {0};
  iree_status_t status = loom_serve_weight_stream_plan(weights, &plan);
  iree_io_parameter_gather_t* gathers = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, plan.group_count,
                                         sizeof(*gathers), (void**)&gathers);
  }
  iree_hal_buffer_t** views = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, weights->storage_count,
                                         sizeof(*views), (void**)&views);
  }
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    views[i] = weights->storages[i].buffer;
    iree_hal_buffer_retain(views[i]);
    status = loom_serve_retirement_track(&weights->activation.retirement,
                                         &views[i], host_allocator);
  }
  const iree_host_size_t lane_count =
      plan.has_preparation ? WEIGHT_COMPLETIONS : WEIGHT_LANES;
  iree_hal_semaphore_t** lanes = weights->activation.semaphores;
  uint64_t* frontiers = weights->activation.values;
  for (iree_host_size_t i = 0; i < lane_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_semaphore_create(
        weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &lanes[i]);
    if (iree_status_is_ok(status)) {
      ++weights->activation.count;
    }
  }
  if (iree_status_is_ok(status) && weights->pool) {
    status = iree_allocator_malloc_array(host_allocator, plan.group_count,
                                         sizeof(*weights->activation.backing),
                                         (void**)&weights->activation.backing);
    for (iree_host_size_t i = 0;
         i < plan.group_count && iree_status_is_ok(status); ++i) {
      loom_serve_weight_group_t* group = &plan.groups[i];
      status = iree_hal_semaphore_create(
          weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &group->mapped);
      if (iree_status_is_ok(status)) {
        bool admitted = false;
        status = loom_serve_memory_pool_commit_async(
            weights->pool, 1, &group->backing, group->mapped, 1, &admitted,
            &weights->activation.backing[i]);
        ++weights->activation.backing_count;
        if (iree_status_is_ok(status) && !admitted) {
          status =
              iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                               "physical pool cannot back model parameters");
        }
      }
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < plan.group_count; ++i) {
      loom_serve_weight_group_t* group = &plan.groups[i];
      const iree_host_size_t lane = i % WEIGHT_LANES;
      group->waits[0] = lanes[lane];
      group->wait_values[0] = frontiers[lane];
      group->waits[1] =
          weights->pool ? group->mapped : weights->activation.backing_ready;
      group->wait_values[1] = 1;
      group->signal_value = ++frontiers[lane];
      gathers[i] = (iree_io_parameter_gather_t){
          .target_buffer = views[group->storage_index],
          .count = group->count,
          .enumerator = {loom_serve_weight_enumerate, group},
          .wait_semaphore_list = {group->waits[1] ? 2 : 1, group->waits,
                                  group->wait_values},
          .signal_semaphore_list = {1, &lanes[lane], &group->signal_value},
      };
    }
    status = iree_io_parameter_provider_gather_batch(
        weights->provider, weights->device, weights->transfer, plan.group_count,
        gathers);
    if (iree_status_is_ok(status)) {
      weights->activity.read_bytes += plan.read_bytes;
      weights->activity.groups += plan.group_count;
    }
  }
  for (iree_host_size_t i = 0;
       i < weights->span_count && iree_status_is_ok(status); ++i) {
    const loom_serve_weight_span_t* span = &weights->spans[i];
    if (!span->prepare) {
      continue;
    }
    // Each read lane is explicitly ordered. Its last group in this tensor's
    // interval proves every preceding fragment on that lane has arrived.
    const loom_serve_weight_group_range_t range = plan.spans[i];
    iree_hal_semaphore_t* waits[WEIGHT_LANES + 1];
    uint64_t values[WEIGHT_LANES + 1];
    iree_host_size_t wait_count = 0;
    for (iree_host_size_t lane = 0; lane < WEIGHT_LANES; ++lane) {
      if (range.begin == range.end) {
        continue;
      }
      const iree_host_size_t last = range.end - 1;
      const iree_host_size_t distance =
          (last % WEIGHT_LANES + WEIGHT_LANES - lane) % WEIGHT_LANES;
      if (last - range.begin >= distance) {
        waits[wait_count] = lanes[lane];
        values[wait_count++] = (last - distance) / WEIGHT_LANES + 1;
      }
    }
    const iree_host_size_t lane = WEIGHT_LANES + i % WEIGHT_LANES;
    waits[wait_count] = lanes[lane];
    values[wait_count++] = frontiers[lane];
    uint64_t signal_value = ++frontiers[lane];
    const iree_hal_buffer_binding_t binding = {views[span->storage_index],
                                               span->parameter.byte_offset,
                                               span->parameter.byte_length};
    status = iree_hal_queue_execute(
        weights->dispatch,
        (iree_hal_semaphore_list_t){wait_count, waits, values},
        (iree_hal_semaphore_list_t){1, &lanes[lane], &signal_value},
        span->prepare, (iree_hal_buffer_binding_table_t){1, &binding},
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      weights->activity.prepared_bytes += span->parameter.byte_length;
    }
  }
  if (!iree_status_is_ok(status)) {
    // Readiness failure cannot retire native allocations or accepted I/O.
    // Separate backing handles and tracked views retain that ownership.
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_fail(lanes[i], iree_status_clone(status));
    }
  }
  if (views) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      iree_hal_buffer_release(views[i]);
    }
  }
  if (plan.groups) {
    for (iree_host_size_t i = 0; i < plan.group_count; ++i) {
      iree_hal_semaphore_release(plan.groups[i].mapped);
    }
  }
  iree_allocator_free(host_allocator, views);
  iree_allocator_free(host_allocator, gathers);
  iree_allocator_free(host_allocator, plan.fragments);
  iree_allocator_free(host_allocator, plan.spans);
  iree_allocator_free(host_allocator, plan.groups);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// A monotonic cursor reclaims each cold backing operation once. Resident
// accounting and retirement queries become constant-time after activation.
static bool loom_serve_weights_collect_backing(loom_serve_weights_t* weights) {
  while (weights->activation.backing_retired <
         weights->activation.backing_count) {
    loom_serve_memory_operation_t** operation =
        &weights->activation.backing[weights->activation.backing_retired];
    if (!loom_serve_memory_operation_is_complete(*operation)) {
      return false;
    }
    weights->activation.backing_status =
        iree_status_join(weights->activation.backing_status,
                         loom_serve_memory_operation_query_status(*operation));
    loom_serve_memory_operation_free(*operation);
    *operation = NULL;
    ++weights->activation.backing_retired;
  }
  return true;
}

static void loom_serve_weights_await_backing(loom_serve_weights_t* weights) {
  for (iree_host_size_t i = weights->activation.backing_retired;
       i < weights->activation.backing_count; ++i) {
    loom_serve_memory_operation_await(weights->activation.backing[i]);
  }
  loom_serve_weights_collect_backing(weights);
}

static void loom_serve_weights_backing_completed(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_serve_weights_t* weights = user_data;
  // The private semaphore preserves the allocation result for streaming waits.
  iree_status_free(status);
  loom_serve_retirement_release(&weights->activation.retirement);
}

iree_status_t loom_serve_weights_allocate(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, iree_hal_pool_t* storage_pool,
    loom_serve_memory_pool_t* pool, iree_host_size_t shared_root_count,
    iree_host_size_t root_count, const loom_serve_weight_root_t* roots,
    iree_allocator_t host_allocator, loom_serve_weights_t** out_weights) {
  *out_weights = NULL;
  loom_serve_weights_t* weights = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*weights),
                                             (void**)&weights));
  weights->host_allocator = host_allocator;
  weights->device = device;
  weights->transfer = transfer;
  weights->dispatch = dispatch;
  weights->storage_pool = storage_pool;
  weights->pool = pool;
  loom_serve_retirement_initialize(&weights->retirement);
  loom_serve_retirement_initialize(&weights->activation.retirement);
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
  loom_serve_weight_root_binding_t* bindings = NULL;
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
    status = loom_serve_weight_resolve_root(
        weights, roots[0].program, root, &weights->span_count, weights->spans,
        requests, &bindings[0]);
  }
  for (iree_host_size_t i = shared_root_count;
       i < root_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_weight_resolve_root(
        weights, roots[i].program, roots[i].root, &weights->span_count,
        weights->spans, requests, &bindings[i]);
  }
  if (iree_status_is_ok(status) && !pool && weights->storage_count) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &weights->activation.backing_ready);
    if (iree_status_is_ok(status)) {
      weights->activation.backing_timepoint.callback =
          loom_serve_weights_backing_completed;
      weights->activation.backing_timepoint.user_data = weights;
      loom_serve_retirement_acquire(&weights->activation.retirement);
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)weights->activation.backing_ready, 1,
          &weights->activation.backing_timepoint);
      if (!iree_status_is_ok(status)) {
        loom_serve_retirement_release(&weights->activation.retirement);
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
    loom_serve_weight_storage_t* storage = &weights->storages[i];
    if (pool) {
      status = loom_serve_virtual_buffer_allocate(
          pool, requests[i].allocation_size, requests[i].min_alignment,
          &weights->statistics, &storage->reservation);
      if (iree_status_is_ok(status)) {
        buffers[i] = loom_serve_virtual_buffer_handle(storage->reservation);
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
      const loom_serve_weight_root_binding_t binding = bindings[i];
      status = iree_hal_buffer_subspan(
          weights->storages[binding.storage_index].buffer,
          binding.range.byte_offset, binding.range.byte_length, host_allocator,
          roots[i].buffer);
      if (iree_status_is_ok(status)) {
        status = loom_serve_retirement_track(&weights->retirement,
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

static iree_status_t loom_serve_weights_activate_storage(
    loom_serve_weights_t* weights) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0,
                                   (int64_t)weights->statistics.reserved_bytes);
  iree_status_t status = iree_ok_status();
  if (weights->pool && weights->storage_count) {
    loom_serve_memory_admission_t admission;
    status =
        loom_serve_memory_pool_prepare(weights->pool,
                                       weights->statistics.reserved_bytes -
                                           weights->statistics.committed_bytes,
                                       &admission);
    if (iree_status_is_ok(status) &&
        admission != LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "physical pool cannot back model parameters");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_weight_stream(weights);
  }
  if (iree_status_is_ok(status)) {
    weights->active = true;
    ++weights->activity.activations;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_serve_weights_activate(loom_serve_weights_t* weights) {
  IREE_ASSERT(!weights->deactivation);
  if (!iree_status_is_ok(weights->activation_status)) {
    return iree_status_clone(weights->activation_status);
  }
  if (!weights->active) {
    weights->activation_status = loom_serve_weights_activate_storage(weights);
  }
  return iree_status_clone(weights->activation_status);
}

iree_status_t loom_serve_weights_import_readiness(
    loom_serve_weights_t* weights, loom_serve_execution_t* execution) {
  return loom_serve_execution_barrier(
      execution, (iree_hal_semaphore_list_t){weights->activation.count,
                                             weights->activation.semaphores,
                                             weights->activation.values});
}

iree_status_t loom_serve_weights_await(loom_serve_weights_t* weights) {
  iree_status_t status = iree_hal_semaphore_list_wait(
      (iree_hal_semaphore_list_t){weights->activation.count,
                                  weights->activation.semaphores,
                                  weights->activation.values},
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  loom_serve_retirement_await(&weights->activation.retirement);
  loom_serve_weights_await_backing(weights);
  return iree_status_join(
      status, iree_status_clone(weights->activation.backing_status));
}

iree_status_t loom_serve_weights_deactivate(loom_serve_weights_t* weights,
                                            bool* out_complete) {
  *out_complete = false;
  if (!weights->pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fixed parameter backing cannot deactivate");
  }
  if (weights->deactivation) {
    if (loom_serve_memory_operation_is_complete(weights->deactivation)) {
      iree_status_t status =
          loom_serve_memory_operation_query_status(weights->deactivation);
      loom_serve_memory_operation_free(weights->deactivation);
      weights->deactivation = NULL;
      *out_complete = true;
      return status;
    }
    return iree_ok_status();
  }
  if (!weights->active) {
    *out_complete = true;
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)weights->statistics.committed_bytes);
  if (!loom_serve_weights_collect_backing(weights)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  loom_serve_virtual_buffer_t** reservations = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      weights->host_allocator, weights->storage_count, sizeof(*reservations),
      (void**)&reservations);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      reservations[i] = weights->storages[i].reservation;
      loom_serve_virtual_buffer_begin_trim(reservations[i]);
    }
    status = loom_serve_memory_pool_trim_async(
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
    iree_allocator_free(weights->host_allocator, weights->activation.backing);
    weights->activation.backing = NULL;
    weights->activation.backing_count = 0;
    weights->activation.backing_retired = 0;
    iree_status_free(weights->activation.backing_status);
    weights->activation.backing_status = iree_ok_status();
    iree_hal_semaphore_release(weights->activation.backing_ready);
    weights->activation.backing_ready = NULL;
    *out_complete = !weights->deactivation;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

loom_serve_memory_statistics_t loom_serve_weights_statistics(
    loom_serve_weights_t* weights) {
  // Accounting is published by the serialized owner, never the cold executor.
  loom_serve_weights_collect_backing(weights);
  loom_serve_memory_operation_is_complete(weights->deactivation);
  return weights->statistics;
}

loom_serve_weight_activity_t loom_serve_weights_activity(
    const loom_serve_weights_t* weights) {
  return weights->activity;
}

void loom_serve_weights_set_retirement_observer(loom_serve_weights_t* weights,
                                                void (*wake)(void* user_data),
                                                void* user_data) {
  if (weights) {
    for (iree_host_size_t i = weights->activation.backing_retired;
         i < weights->activation.backing_count; ++i) {
      loom_serve_memory_operation_set_observer(weights->activation.backing[i],
                                               wake, user_data);
    }
    loom_serve_memory_operation_set_observer(weights->deactivation, wake,
                                             user_data);
    loom_serve_retirement_set_observer(&weights->retirement, wake, user_data);
    loom_serve_retirement_set_observer(&weights->activation.retirement, wake,
                                       user_data);
  }
}

bool loom_serve_weights_is_retired(loom_serve_weights_t* weights) {
  return !weights ||
         (loom_serve_weights_collect_backing(weights) &&
          loom_serve_memory_operation_is_complete(weights->deactivation) &&
          loom_serve_retirement_is_idle(&weights->retirement) &&
          loom_serve_retirement_is_idle(&weights->activation.retirement));
}

void loom_serve_weights_await_retirement(loom_serve_weights_t* weights) {
  if (weights) {
    loom_serve_weights_await_backing(weights);
    loom_serve_memory_operation_await(weights->deactivation);
    loom_serve_retirement_await(&weights->retirement);
    loom_serve_retirement_await(&weights->activation.retirement);
  }
}

iree_status_t loom_serve_weights_free(loom_serve_weights_t* weights) {
  if (!weights) {
    return iree_ok_status();
  }
  const bool retired = loom_serve_weights_is_retired(weights);
  IREE_ASSERT(retired);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_virtual_buffer_free(weights->storages[i].reservation);
    if (iree_status_is_ok(status)) {
      weights->storages[i].reservation = NULL;
      iree_hal_buffer_release(weights->storages[i].buffer);
      weights->storages[i].buffer = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_allocator_free(weights->host_allocator, weights->activation.backing);
    iree_status_free(weights->activation.backing_status);
    loom_serve_memory_operation_free(weights->deactivation);
    iree_hal_semaphore_release(weights->activation.backing_ready);
    loom_serve_retirement_deinitialize(&weights->retirement);
    loom_serve_retirement_deinitialize(&weights->activation.retirement);
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_release(weights->activation.semaphores[i]);
    }
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      iree_hal_command_buffer_release(weights->spans[i].prepare);
    }
    iree_io_parameter_provider_release(weights->provider);
    iree_status_ignore(weights->activation_status);
    iree_allocator_free(weights->host_allocator, weights->storages);
    iree_allocator_free(weights->host_allocator, weights->keys);
    iree_allocator_free(weights->host_allocator, weights->spans);
    iree_allocator_free(weights->host_allocator, weights);
  }
  return status;
}
