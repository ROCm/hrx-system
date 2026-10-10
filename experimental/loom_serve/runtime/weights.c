// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/weights.h"

#include <string.h>

#include "experimental/loom_serve/runtime/program.h"
#include "experimental/loom_serve/runtime/retirement.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/tooling/parameter_util.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loomc/iree.h"

enum { WEIGHT_LANES = 4 };

typedef struct weight_span_t {
  // Borrowed key and final byte placement from the compiled command.
  loom_cmd_program_parameter_t parameter;
  // Borrowed final allocation or root view receiving the file bytes.
  iree_hal_buffer_t* buffer;
  // Unique final storage index, resolved once before any queue work.
  iree_host_size_t storage_index;
  // Owned cached in-place preparation command, or null for unchanged bytes.
  iree_hal_command_buffer_t* prepare;
} weight_span_t;

typedef struct weight_group_t {
  // Borrowed consecutive spans in one final allocation. Prepared groups own
  // exactly one tensor; consecutive canonical spans share one readiness edge.
  weight_span_t* spans;
  // Transaction-local view retained by all reads and preparations of this
  // group.
  iree_hal_buffer_t* buffer;
  // Number of spans exposed to the parameter provider.
  iree_host_size_t count;
  // Previous preparation frontier on this group's lane.
  uint64_t wait_value;
  // Read and preparation frontier established by this group.
  uint64_t signal_value;
} weight_group_t;

typedef struct weight_storage_t {
  // Borrowed final root retained by the caller's output slot.
  iree_hal_buffer_t* buffer;
  // Owned virtual reservation, or NULL for explicitly fixed backing.
  loom_serve_virtual_buffer_t* reservation;
} weight_storage_t;

struct loom_serve_weights_t {
  // Allocator owning the plan and cold activation metadata.
  iree_allocator_t allocator;
  // Borrowed device enclosing the complete plan lifetime.
  iree_hal_device_t* device;
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
    // Accepted I/O/preparation views, including after failed readiness.
    loom_serve_retirement_t retirement;
    // Owned lane readiness handles imported into the consuming model timeline.
    iree_hal_semaphore_t* semaphores[WEIGHT_LANES];
    // Last submitted preparation or canonical read on each initialized lane.
    uint64_t values[WEIGHT_LANES];
    // Initialized handles, including a partially failed activation.
    iree_host_size_t count;
  } activation;
  // Owned original checkpoint provider and imported-file cache.
  iree_io_parameter_provider_t* provider;
  // Owned placement/preparation records, with keys in the owned key storage.
  weight_span_t* spans;
  // Owned parameter key bytes, independent of source/JIT lifetime.
  char* keys;
  // Number of initialized placement records, including partial creation.
  iree_host_size_t span_count;
  // Owned unique root records; aliasing output slots do not duplicate these.
  weight_storage_t* storages;
  // Number of initialized unique roots, including partial creation.
  iree_host_size_t storage_count;
  // Activation was enqueued into the owning model's execution domain.
  bool active;
};

static iree_status_t weight_allocate(iree_hal_device_t* device,
                                     iree_device_size_t length,
                                     iree_device_size_t alignment,
                                     iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = alignment;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, length, out_buffer);
}

static const weight_span_t* weight_find(iree_host_size_t count,
                                        const weight_span_t* spans,
                                        iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(spans[i].parameter.key, key)) {
      return &spans[i];
    }
  }
  return NULL;
}

// Resolves root sharing before submitting any I/O. Only newly allocated roots
// append spans, so tensors shared by multiple stages are read and prepared
// exactly once.
static iree_status_t weight_resolve_root(loom_serve_weights_t* weights,
                                         const loom_cmd_program_t* program,
                                         loom_cmd_program_parameter_root_t root,
                                         iree_host_size_t* span_count,
                                         weight_span_t* spans,
                                         iree_allocator_t allocator,
                                         iree_hal_buffer_t** out_buffer) {
  const iree_host_size_t prior_count = *span_count;
  const weight_span_t* shared = NULL;
  iree_device_size_t origin = 0;
  iree_host_size_t count = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    const weight_span_t* existing =
        weight_find(prior_count, spans, parameter.key);
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
         (existing->buffer != shared->buffer ||
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
    return iree_hal_buffer_subspan(shared->buffer, origin,
                                   root.required_byte_length, allocator,
                                   out_buffer);
  }
  weight_storage_t* storage = &weights->storages[weights->storage_count];
  if (weights->pool) {
    IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_create(
        weights->pool, root.required_byte_length, root.minimum_alignment,
        &weights->statistics, &storage->reservation));
    *out_buffer = loom_serve_virtual_buffer_handle(storage->reservation);
    iree_hal_buffer_retain(*out_buffer);
  } else {
    IREE_RETURN_IF_ERROR(weight_allocate(weights->device,
                                         root.required_byte_length,
                                         root.minimum_alignment, out_buffer));
  }
  const iree_host_size_t storage_index = weights->storage_count++;
  IREE_RETURN_IF_ERROR(
      loom_serve_retirement_track(&weights->retirement, out_buffer, allocator));
  storage->buffer = *out_buffer;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    spans[(*span_count)++] = (weight_span_t){
        .parameter = parameter,
        .buffer = *out_buffer,
        .storage_index = storage_index,
    };
  }
  return iree_ok_status();
}

// A cached preparer owns the returned VM string until cold planning completes.
typedef struct weight_preparer_t {
  // Owned VM result backing the root name, independent of the invocation.
  iree_vm_buffer_t* root;
  // Owned recorded command retaining its executables independently of the JIT.
  iree_hal_command_buffer_t* command;
} weight_preparer_t;

typedef struct weight_preparation_t {
  // Initial CPU task; its destruction transfers ownership to the JIT chain.
  loomc_task_t task;
  // Exclusively borrowed unactivated plan through terminal completion.
  loom_serve_weights_t* weights;
  // Exclusively borrowed source catalog through terminal completion.
  loom_serve_jit_t* jit;
  // Recording policy applied to each source-selected preparation command.
  iree_hal_command_buffer_mode_t command_mode;
  // Borrowed checkpoint locator, retained by the load owner through completion.
  iree_string_view_t weights_path;
  // Borrowed source policy locator with the same lifetime.
  iree_string_view_t policy_path;
  // Owned distinct preparers, queried once before native compilation.
  weight_preparer_t* preparers;
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
  // Terminal ownership transfer, invoked after all job storage is released.
  loom_serve_weights_completion_fn_t completion;
  // Borrowed terminal callback payload.
  void* user_data;
} weight_preparation_t;

static iree_status_t weight_query_preparer(
    loom_serve_program_t* program, const iree_vm_ref_types_t* types,
    iree_vm_function_t function, loom_cmd_program_parameter_t parameter,
    iree_allocator_t allocator, iree_vm_buffer_t** out_root) {
  *out_root = NULL;
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)parameter.key.data, parameter.key.size),
      iree_vm_buffer_release_callback_null(), allocator, &key));
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

static iree_string_view_t weight_preparer_name(iree_vm_buffer_t* root) {
  return iree_make_string_view(iree_vm_buffer_const_data(root),
                               iree_vm_buffer_length(root));
}

static iree_status_t weight_prepare_policy(weight_preparation_t* job) {
  loom_serve_weights_t* weights = job->weights;
  const iree_allocator_t allocator = weights->allocator;
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_environment_allocate(allocator, &environment));
  loom_serve_program_t* program = NULL;
  const iree_string_view_t roots[] = {IREE_SVL("prepare_weight")};
  iree_status_t status = loom_serve_program_create(
      environment, job->policy_path, IREE_ARRAYSIZE(roots), roots,
      iree_vm_module_span_empty(), loom_serve_jit_report(job->jit), allocator,
      &program);
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
    status = iree_allocator_malloc_array(allocator, weights->span_count,
                                         sizeof(*job->preparers),
                                         (void**)&job->preparers);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, weights->span_count,
                                         sizeof(*job->span_preparers),
                                         (void**)&job->span_preparers);
  }
  for (iree_host_size_t i = 0;
       i < weights->span_count && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* root = NULL;
    status =
        weight_query_preparer(program, &types, function,
                              weights->spans[i].parameter, allocator, &root);
    job->span_preparers[i] = SIZE_MAX;
    if (iree_status_is_ok(status) && iree_vm_buffer_length(root)) {
      const iree_string_view_t name = weight_preparer_name(root);
      iree_host_size_t p = 0;
      while (p < job->preparer_count &&
             !iree_string_view_equal(
                 name, weight_preparer_name(job->preparers[p].root))) {
        ++p;
      }
      if (p == job->preparer_count) {
        weight_preparer_t* preparer = &job->preparers[job->preparer_count++];
        preparer->root = root;
        root = NULL;
      }
      job->span_preparers[i] = p;
    }
    iree_vm_buffer_release(root);
  }
  loom_serve_program_destroy(program);
  iree_vm_environment_free(environment);
  return status;
}

static void weight_preparation_finish(weight_preparation_t* job,
                                      iree_status_t status) {
  loom_serve_weights_t* weights = job->weights;
  const iree_allocator_t allocator = weights->allocator;
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
  iree_allocator_free(allocator, job->preparers);
  iree_allocator_free(allocator, job->span_preparers);
  const loom_serve_weights_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(allocator, job);
  completion(user_data, status);
}

// Each native completion submits the next preparer without a worker join.
static void weight_preparation_compiled(void* user_data, iree_status_t status,
                                        loom_serve_jit_stage_t* compiled);

static void weight_preparation_next(weight_preparation_t* job) {
  if (job->preparer_index == job->preparer_count) {
    weight_preparation_finish(job, iree_ok_status());
    return;
  }
  const iree_string_view_t root =
      weight_preparer_name(job->preparers[job->preparer_index].root);
  iree_status_t status = loom_serve_jit_compile_async(
      job->jit, root, &job->config, weight_preparation_compiled, job);
  // An accepted completion may already have destroyed job. Rejection owes no
  // callback and returns the still-exclusive job to this terminal path.
  if (!iree_status_is_ok(status)) {
    weight_preparation_finish(job, status);
  }
}

static void weight_preparation_compiled(void* user_data, iree_status_t status,
                                        loom_serve_jit_stage_t* compiled) {
  weight_preparation_t* job = user_data;
  weight_preparer_t* preparer = &job->preparers[job->preparer_index];
  if (iree_status_is_ok(status)) {
    const loom_cmd_program_requirements_t* requirements =
        &loom_serve_jit_stage_program(compiled)->requirements;
    if (requirements->fixed_buffer_count ||
        requirements->rebindable_binding_count != 1 ||
        requirements->transient.required_byte_length ||
        requirements->launch_counts.required_byte_length) {
      const iree_string_view_t name = weight_preparer_name(preparer->root);
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "weight preparer '%.*s' requires one mutable binding "
          "and no global scratch",
          (int)name.size, name.data);
    } else {
      status = loom_serve_jit_stage_record(
          compiled, iree_hal_queue_family(job->weights->dispatch),
          job->command_mode, NULL, &preparer->command);
    }
  }
  loom_serve_jit_stage_destroy(compiled);
  if (iree_status_is_ok(status)) {
    ++job->preparer_index;
    weight_preparation_next(job);
  } else {
    weight_preparation_finish(job, status);
  }
}

static void weight_preparation_execute(loomc_task_t* task,
                                       loomc_host_size_t worker_ordinal) {
  (void)worker_ordinal;
  weight_preparation_t* job = (weight_preparation_t*)task;
  const iree_allocator_t allocator = job->weights->allocator;
  iree_status_t status = weight_prepare_policy(job);
  iree_io_parameter_index_t* index = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(allocator, &index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_append_parameter_file_to_index(job->weights_path,
                                                         index, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        iree_string_view_empty(), index, WEIGHT_LANES, allocator,
        &job->weights->provider);
  }
  iree_io_parameter_index_release(index);
  job->status = status;
}

static void weight_preparation_destroy(loomc_task_t* task) {
  weight_preparation_t* job = (weight_preparation_t*)task;
  if (iree_status_is_ok(job->status)) {
    weight_preparation_next(job);
  } else {
    weight_preparation_finish(job, job->status);
  }
}

static const loomc_task_vtable_t weight_preparation_vtable = {
    .execute = weight_preparation_execute,
    .destroy = weight_preparation_destroy,
};

iree_status_t loom_serve_weights_prepare_async(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    iree_string_view_t weights_path, iree_string_view_t policy_path,
    loom_serve_weights_completion_fn_t completion, void* user_data) {
  weight_preparation_t* job = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(weights->allocator, sizeof(*job), (void**)&job));
  loomc_task_initialize(&weight_preparation_vtable, &job->task);
  job->weights = weights;
  job->jit = jit;
  job->command_mode = command_mode;
  job->weights_path = weights_path;
  job->policy_path = policy_path;
  job->config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  job->completion = completion;
  job->user_data = user_data;
  iree_status_t status = iree_status_from_loomc(
      loomc_task_sink_submit(loom_serve_host_compiler_sink(host), &job->task));
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(weights->allocator, job);
  }
  return status;
}

typedef struct weight_preparation_join_t {
  // Serializes callback publication with the outer owner's final observation.
  iree_slim_mutex_t mutex;
  // Outer application notification, never waited on by a shared worker.
  iree_notification_t notification;
  // Final preparation result, owned by the outer caller after publication.
  iree_status_t status;
  // Terminal callback has published all of its state.
  bool ready;
} weight_preparation_join_t;

static void weight_preparation_ready(void* user_data, iree_status_t status) {
  weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  join->status = status;
  join->ready = true;
  iree_notification_post(&join->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&join->mutex);
}

static bool weight_preparation_is_ready(void* user_data) {
  weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  const bool ready = join->ready;
  iree_slim_mutex_unlock(&join->mutex);
  return ready;
}

iree_status_t loom_serve_weights_prepare(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    iree_string_view_t weights_path, iree_string_view_t policy_path) {
  weight_preparation_join_t join = {0};
  iree_slim_mutex_initialize(&join.mutex);
  iree_notification_initialize(&join.notification);
  iree_status_t status = loom_serve_weights_prepare_async(
      weights, host, jit, command_mode, weights_path, policy_path,
      weight_preparation_ready, &join);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&join.notification, weight_preparation_is_ready,
                            &join, iree_infinite_timeout());
    status = join.status;
  }
  iree_notification_deinitialize(&join.notification);
  iree_slim_mutex_deinitialize(&join.mutex);
  return status;
}

static iree_status_t weight_enumerate(void* user_data, iree_host_size_t i,
                                      iree_string_view_t* out_key,
                                      iree_io_parameter_span_t* out_span) {
  const weight_group_t* group = user_data;
  const loom_cmd_program_parameter_t parameter = group->spans[i].parameter;
  *out_key = parameter.key;
  *out_span = (iree_io_parameter_span_t){
      .buffer_offset = parameter.byte_offset,
      .length = parameter.byte_length,
  };
  return iree_ok_status();
}

static iree_status_t weight_stream(loom_serve_weights_t* weights) {
  const iree_allocator_t allocator = weights->allocator;
  const iree_host_size_t span_count = weights->span_count;
  weight_span_t* spans = weights->spans;
  iree_hal_device_t* device = weights->device;
  weight_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, span_count, sizeof(*groups), (void**)&groups));
  iree_io_parameter_gather_t* gathers = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, span_count, sizeof(*gathers), (void**)&gathers);
  iree_hal_buffer_t** views = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, weights->storage_count,
                                         sizeof(*views), (void**)&views);
  }
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    views[i] = weights->storages[i].buffer;
    iree_hal_buffer_retain(views[i]);
    status = loom_serve_retirement_track(&weights->activation.retirement,
                                         &views[i], allocator);
  }
  iree_hal_semaphore_t* reads[WEIGHT_LANES] = {0};
  iree_hal_semaphore_t** ready = weights->activation.semaphores;
  uint64_t* frontiers = weights->activation.values;
  for (iree_host_size_t i = 0; i < WEIGHT_LANES && iree_status_is_ok(status);
       ++i) {
    status =
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &reads[i]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready[i]);
      if (iree_status_is_ok(status)) {
        ++weights->activation.count;
      }
    }
  }
  iree_host_size_t group_count = 0;
  uint64_t loaded_bytes = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < span_count; ++i) {
      const weight_span_t* span = &spans[i];
      if (!i || span->prepare || spans[i - 1].prepare ||
          span->buffer != spans[i - 1].buffer) {
        weight_group_t* group = &groups[group_count++];
        group->spans = &spans[i];
        group->buffer = views[span->storage_index];
      }
      ++groups[group_count - 1].count;
      loaded_bytes += span->parameter.byte_length;
    }
    for (iree_host_size_t i = 0; i < group_count; ++i) {
      weight_group_t* group = &groups[i];
      const iree_host_size_t lane = i % WEIGHT_LANES;
      group->wait_value = frontiers[lane];
      group->signal_value = ++frontiers[lane];
      gathers[i] = (iree_io_parameter_gather_t){
          .target_buffer = group->buffer,
          .count = group->count,
          .enumerator = {weight_enumerate, group},
          .wait_semaphore_list = {1, &ready[lane], &group->wait_value},
          .signal_semaphore_list = {1,
                                    group->spans[0].prepare ? &reads[lane]
                                                            : &ready[lane],
                                    &group->signal_value},
      };
    }
    status = iree_io_parameter_provider_gather_batch(
        weights->provider, device, weights->transfer, group_count, gathers);
    if (iree_status_is_ok(status)) {
      weights->activity.read_bytes += loaded_bytes;
      weights->activity.groups += group_count;
    }
  }
  for (iree_host_size_t i = 0; i < group_count && iree_status_is_ok(status);
       ++i) {
    weight_group_t* group = &groups[i];
    const weight_span_t* span = group->spans;
    if (!span->prepare) {
      continue;
    }
    const iree_host_size_t lane = i % WEIGHT_LANES;
    const iree_hal_buffer_binding_t binding = {group->buffer,
                                               span->parameter.byte_offset,
                                               span->parameter.byte_length};
    status = iree_hal_queue_execute(
        weights->dispatch,
        (iree_hal_semaphore_list_t){1, &reads[lane], &group->signal_value},
        (iree_hal_semaphore_list_t){1, &ready[lane], &group->signal_value},
        span->prepare, (iree_hal_buffer_binding_table_t){1, &binding},
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      weights->activity.prepared_bytes += span->parameter.byte_length;
    }
  }
  if (!iree_status_is_ok(status)) {
    // Release waits for preparations that were never submitted. Queue-owned
    // file, command and buffer references survive until those operations
    // retire.
    for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
      if (reads[i]) {
        iree_hal_semaphore_fail(reads[i], iree_status_clone(status));
      }
      if (ready[i]) {
        iree_hal_semaphore_fail(ready[i], iree_status_clone(status));
      }
    }
  }
  for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
    iree_hal_semaphore_release(reads[i]);
  }
  if (views) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      iree_hal_buffer_release(views[i]);
    }
  }
  iree_allocator_free(allocator, views);
  iree_allocator_free(allocator, gathers);
  iree_allocator_free(allocator, groups);
  return status;
}

iree_status_t loom_serve_weights_create(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, loom_serve_memory_pool_t* pool,
    iree_host_size_t shared_root_count, iree_host_size_t root_count,
    const loom_serve_weight_root_t* roots, loom_serve_weights_t** out_weights,
    iree_allocator_t host_allocator) {
  *out_weights = NULL;
  loom_serve_weights_t* weights = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*weights),
                                             (void**)&weights));
  weights->allocator = host_allocator;
  weights->device = device;
  weights->transfer = transfer;
  weights->dispatch = dispatch;
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
  if (shared_root_count && iree_status_is_ok(status)) {
    loom_cmd_program_parameter_root_t root = roots[0].root;
    for (iree_host_size_t i = 1; i < shared_root_count; ++i) {
      const loom_cmd_program_parameter_root_t other = roots[i].root;
      root.required_byte_length =
          iree_max(root.required_byte_length, other.required_byte_length);
      root.minimum_alignment =
          iree_max(root.minimum_alignment, other.minimum_alignment);
    }
    status = weight_resolve_root(weights, roots[0].program, root,
                                 &weights->span_count, weights->spans,
                                 host_allocator, roots[0].buffer);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 1; i < shared_root_count; ++i) {
      *roots[i].buffer = *roots[0].buffer;
      iree_hal_buffer_retain(*roots[i].buffer);
    }
  }
  for (iree_host_size_t i = shared_root_count;
       i < root_count && iree_status_is_ok(status); ++i) {
    status = weight_resolve_root(weights, roots[i].program, roots[i].root,
                                 &weights->span_count, weights->spans,
                                 host_allocator, roots[i].buffer);
  }
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

iree_status_t loom_serve_weights_activate(loom_serve_weights_t* weights,
                                          loom_serve_execution_t* execution) {
  if (weights->active) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    weight_storage_t* storage = &weights->storages[i];
    if (storage->reservation) {
      status = loom_serve_virtual_buffer_commit(
          storage->reservation, 0,
          iree_hal_buffer_byte_length(storage->buffer));
    }
  }
  if (iree_status_is_ok(status)) {
    status = weight_stream(weights);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_barrier(
        execution, (iree_hal_semaphore_list_t){weights->activation.count,
                                               weights->activation.semaphores,
                                               weights->activation.values});
  }
  if (iree_status_is_ok(status)) {
    weights->active = true;
    ++weights->activity.activations;
  }
  return status;
}

iree_status_t loom_serve_weights_await(loom_serve_weights_t* weights) {
  iree_status_t status = iree_hal_semaphore_list_wait(
      (iree_hal_semaphore_list_t){weights->activation.count,
                                  weights->activation.semaphores,
                                  weights->activation.values},
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  loom_serve_retirement_await(&weights->activation.retirement);
  return status;
}

iree_status_t loom_serve_weights_deactivate(loom_serve_weights_t* weights) {
  if (!weights->pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fixed parameter backing cannot deactivate");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    loom_serve_virtual_buffer_t* reservation = weights->storages[i].reservation;
    loom_serve_virtual_buffer_begin_trim(reservation);
    status = loom_serve_virtual_buffer_trim(reservation);
  }
  if (iree_status_is_ok(status)) {
    weights->activity.deactivations += weights->active;
    weights->active = false;
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_release(weights->activation.semaphores[i]);
      weights->activation.semaphores[i] = NULL;
      weights->activation.values[i] = 0;
    }
    weights->activation.count = 0;
  }
  return status;
}

loom_serve_memory_statistics_t loom_serve_weights_statistics(
    const loom_serve_weights_t* weights) {
  return weights->statistics;
}

loom_serve_weight_activity_t loom_serve_weights_activity(
    const loom_serve_weights_t* weights) {
  return weights->activity;
}

iree_status_t loom_serve_weights_destroy(loom_serve_weights_t* weights) {
  if (!weights) {
    return iree_ok_status();
  }
  loom_serve_retirement_await(&weights->retirement);
  loom_serve_retirement_await(&weights->activation.retirement);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    status =
        loom_serve_virtual_buffer_destroy(weights->storages[i].reservation);
    if (iree_status_is_ok(status)) {
      weights->storages[i].reservation = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_serve_retirement_deinitialize(&weights->retirement);
    loom_serve_retirement_deinitialize(&weights->activation.retirement);
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_release(weights->activation.semaphores[i]);
    }
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      iree_hal_command_buffer_release(weights->spans[i].prepare);
    }
    iree_io_parameter_provider_release(weights->provider);
    iree_allocator_free(weights->allocator, weights->storages);
    iree_allocator_free(weights->allocator, weights->keys);
    iree_allocator_free(weights->allocator, weights->spans);
    iree_allocator_free(weights->allocator, weights);
  }
  return status;
}
