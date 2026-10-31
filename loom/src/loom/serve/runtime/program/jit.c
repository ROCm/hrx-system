// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/jit.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "loom/serve/runtime/program/executable.h"
#include "loom/serve/runtime/program/jit_cache.h"
#include "loom/serve/runtime/program/native_artifact.h"
#include "loom/serve/runtime/program/source_index.h"
#include "loom/serve/runtime/program/target.h"
#include "loom/target/arch/cmd/hal_command_buffer.h"
#include "loomc/compile.h"
#include "loomc/emit.h"
#include "loomc/iree.h"
#include "loomc/launch_config.h"
#include "loomc/link.h"
#include "loomc/target/cmd/program.h"
#include "loomc/target/vm.h"
#include "loomc/task_pool.h"

typedef struct loom_serve_jit_compiler_t {
  // Target package retained by the context and prepared compiler.
  loomc_target_environment_t* environment;
  // Shared immutable parser and dialect configuration.
  loomc_context_t* context;
  // Prepared compiler reused for every kernel in the residency.
  loomc_compiler_t* compiler;
  // Source-to-prepared-low pipeline for the selected target package.
  loomc_pass_program_t* pipeline;
  // Live device facts, or the portable VM profile.
  loomc_target_profile_t* profile;
} loom_serve_jit_compiler_t;

typedef struct loom_serve_jit_job_t loom_serve_jit_job_t;
typedef struct loom_serve_jit_request_t loom_serve_jit_request_t;

// Callback state that can outlive the device-owning source catalog.
// Accepted jobs retain this independently so catalog teardown never runs after
// an external completion has published its result.
typedef struct loom_serve_jit_publication_t {
  // Catalog ownership, accepted jobs and an active publisher.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy shared with jobs and the source catalog.
  iree_allocator_t host_allocator;
  // Serializes report access, the ready list, and the callback publisher.
  iree_slim_mutex_t mutex;
  // Borrowed evidence accumulator, updated in readiness order.
  loom_serve_compilation_report_t* report;
  // First completed job waiting for callback publication.
  loom_serve_jit_job_t* head;
  // Last completed job, or NULL when the list is empty.
  loom_serve_jit_job_t* tail;
  // One worker owns callback publication; reentrant submissions append.
  bool publishing;
} loom_serve_jit_publication_t;

// One catalog's device-local view of an exact shared native input. The catalog
// mutex protects lookup and load completion; compilation and image ownership
// belong to the host domain, independently of this device and its parents.
typedef struct loom_serve_jit_native_t {
  // Cache ownership and each consuming request retain this entry independently.
  iree_atomic_ref_count_t ref_count;
  // Retained immutable compilation outcome, independent of the device load.
  loom_serve_native_artifact_t* artifact;
  // Shared compile completion, keeping pending parent ownership alive.
  loom_serve_native_subscription_t subscription;
  // Waiting requests retain their parent until native completion is published.
  loom_serve_jit_request_t* waiters;
  // Native result is immutable after publication under the catalog mutex.
  bool complete;
  // Owned terminal device-load or cloned compilation status.
  iree_status_t status;
  // Owned loaded-code record, shared with stages after successful publication.
  loom_serve_executable_t* executable;
  // Borrowed catalog retained by pending parents through publication.
  loom_serve_jit_t* jit;
  // First request's scheduling domain for this device's independent load.
  loomc_task_sink_t sink;
  // Emitted-image load submitted after compiler accesses retire.
  loom_serve_executable_request_t load;
} loom_serve_jit_native_t;

struct loom_serve_jit_t {
  // Caller and accepted compiler work ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy shared by compiled stages.
  iree_allocator_t host_allocator;
  // Short native-cache lock; never held across compilation or user callbacks.
  iree_slim_mutex_t mutex;
  // Ordered callbacks that remain valid after catalog/device ownership retires.
  loom_serve_jit_publication_t* publication;
  // Optional shared immutable source generation retained by this catalog.
  loom_serve_jit_generation_t* generation;
  // Retained device backing the live target and executable loads.
  iree_hal_device_t* device;
  // Retained exact execution queue.
  iree_hal_queue_t* dispatch;
  // Exact executable target borrowed from device.
  const iree_hal_executable_target_t* target;
  // Process-lifetime format name paired with the selected HAL loader.
  iree_string_view_t artifact_format;
  // Model-wide compiler state specialized to immutable device facts.
  loom_serve_jit_compiler_t kernel;
  // Exact host compilation domain; local views keep device loading separate.
  loom_serve_native_domain_t* native_domain;
  // Frozen source providers, shared across all shape specializations.
  loomc_link_index_t* index;
  // Immutable linker for direct kernel roots in the shared source catalog.
  loomc_linker_t* linker;
  // Exact output reuse, independent of input specialization and entry mapping.
  loom_serve_executable_cache_t* executables;
  // Exact native reuse, scoped to this catalog's fixed compiler and loader.
  struct {
    // Stable native entries sorted by their exact keys.
    loom_serve_jit_native_t** values;
    // Indexed entries, including the current job's unpublished misses.
    iree_host_size_t count;
    // Allocated pointer capacity, independent of entry storage.
    iree_host_size_t capacity;
  } natives;
  // Concurrent native compilation, independent of serving execution.
  struct {
    // Borrowed shared host compiler sink, never stopped by catalog teardown.
    loomc_task_sink_t sink;
    // Mutable scratch indexed by the pool's mutually exclusive worker ordinal.
    loomc_workspace_t** workspaces;
    // Number of initialized workspace entries, including during setup failure.
    iree_host_size_t workspace_count;
  } compilation;
};

struct loom_serve_jit_stage_t {
  // Cache and each model consumer retain this immutable stage independently.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy used for owned bytes and entry storage.
  iree_allocator_t host_allocator;
  // Borrowed direct launch contract, owned by the sole executable below.
  loom_serve_dispatch_t* dispatch;
  // Owned contiguous portable command image.
  iree_const_byte_span_t image;
  // Parsed program borrowing image.
  loom_cmd_program_t program;
  // Loaded entries in the command program's local slot order.
  loom_cmd_program_hal_entry_t* entries;
  // Number of initialized entry slots, including null entries on failure.
  iree_host_size_t entry_count;
  // Number of initialized loaded-code owners.
  iree_host_size_t executable_count;
  // Loaded-code owners in request order, independent of cache-key lifetime.
  loom_serve_executable_t* executables[];
};

static void loom_serve_jit_compiler_deinitialize(
    loom_serve_jit_compiler_t* compiler) {
  loomc_pass_program_release(compiler->pipeline);
  loomc_compiler_release(compiler->compiler);
  loomc_context_release(compiler->context);
  loomc_target_profile_release(compiler->profile);
  loomc_target_environment_release(compiler->environment);
}

static iree_status_t loom_serve_jit_compiler_initialize(
    const loomc_sanitizer_options_t* sanitizer,
    loomc_target_control_flow_lowering_t control_flow_lowering,
    loom_serve_compilation_report_t* report,
    loom_serve_jit_compiler_t* compiler, iree_allocator_t host_allocator) {
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .target_environment = compiler->environment,
  };
  const loomc_context_options_t context_options = {.next = &target_options};
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_context_create(&context_options, ca, &compiler->context)));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_compiler_create(compiler->context, NULL, ca, &compiler->compiler)));
  loomc_result_t* result = NULL;
  const loomc_target_pipeline_options_t pipeline_options = {
      .next = sanitizer, .control_flow_lowering = control_flow_lowering};
  const loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      compiler->context, &pipeline_options, ca, &compiler->pipeline, &result);
  return loom_serve_compilation_result(status, result, report);
}

// Compilation identities are process-local cache keys, but use a stable byte
// grammar so structure padding, pointer values, and field partitioning never
// authorize reuse. Each field is framed with an explicit little-endian length.
static void loom_serve_jit_identity_update(loom_sha256_state_t* identity,
                                           const void* data,
                                           iree_host_size_t length) {
  uint8_t encoded_length[8];
  iree_unaligned_store_le_u64(encoded_length, length);
  loom_sha256_update(identity, iree_make_const_byte_span(
                                   encoded_length, sizeof(encoded_length)));
  loom_sha256_update(identity, iree_make_const_byte_span(data, length));
}

static void loom_serve_jit_identity_update_view(loom_sha256_state_t* identity,
                                                iree_string_view_t value) {
  loom_serve_jit_identity_update(identity, value.data, value.size);
}

static void loom_serve_jit_identity_update_u64(loom_sha256_state_t* identity,
                                               uint64_t value) {
  uint8_t encoded[8];
  iree_unaligned_store_le_u64(encoded, value);
  loom_serve_jit_identity_update(identity, encoded, sizeof(encoded));
}

static void loom_serve_jit_identity_initialize(
    const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    const loom_serve_target_t* target, loom_sha256_state_t* out_identity) {
  loom_sha256_initialize(out_identity);
  loom_serve_jit_identity_update_view(
      out_identity, IREE_SV("loom.serve.compilation-generation.v2"));
  loom_serve_jit_identity_update_view(
      out_identity, target->selection.executable_target->family);
  loom_serve_jit_identity_update_view(
      out_identity, target->selection.executable_target->target_key);
  loom_serve_jit_identity_update_u64(out_identity,
                                     target->selection.executable_target->kind);
  loom_serve_jit_identity_update_u64(
      out_identity,
      target->selection.executable_target->physical_device_affinity);
  loom_serve_jit_identity_update_u64(
      out_identity, target->selection.executable_target->flags);
  loom_serve_jit_identity_update_view(out_identity, target->artifact_format);
  loom_serve_jit_identity_update_u64(out_identity,
                                     target->control_flow_lowering);
  loom_serve_jit_identity_update_u64(out_identity,
                                     sanitizer ? sanitizer->checks : 0);
  loom_serve_jit_identity_update_u64(out_identity,
                                     sanitizer ? sanitizer->flags : 0);
  loom_serve_jit_identity_update_u64(out_identity,
                                     sanitizer ? sanitizer->reporting_mode : 0);
  loom_serve_jit_identity_update_u64(out_identity, sources->count);
  if (sources->card) {
    const loom_model_selection_t* selection =
        loom_model_card_selection(sources->card);
    loom_serve_jit_identity_update_view(out_identity, selection->config);
    loom_serve_jit_identity_update_u64(out_identity, selection->inputs.count);
    loom_serve_jit_identity_update_u64(out_identity,
                                       selection->libraries.count);
  } else {
    loom_serve_jit_identity_update_view(out_identity, iree_string_view_empty());
    loom_serve_jit_identity_update_u64(out_identity, 0);
    loom_serve_jit_identity_update_u64(out_identity, 0);
  }
  if (sources->component) {
    loom_serve_jit_identity_update_view(out_identity,
                                        sources->component->export_prefix);
    loom_serve_jit_identity_update_u64(out_identity,
                                       sources->component->version);
  } else {
    loom_serve_jit_identity_update_view(out_identity, iree_string_view_empty());
    loom_serve_jit_identity_update_u64(out_identity, 0);
  }
}

static iree_status_t loom_serve_jit_initialize_compilation(
    loom_serve_jit_t* jit, loom_serve_host_t* host) {
  const loomc_allocator_t host_allocator =
      loomc_allocator_from_iree(jit->host_allocator);
  jit->compilation.sink = loom_serve_host_compiler_sink(host);
  const iree_host_size_t worker_count =
      loomc_task_pool_worker_count(loom_serve_host_task_pool(host));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      jit->host_allocator, worker_count, sizeof(*jit->compilation.workspaces),
      (void**)&jit->compilation.workspaces));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < worker_count && iree_status_is_ok(status);
       ++i) {
    status = iree_status_from_loomc(loomc_workspace_create(
        NULL, host_allocator, &jit->compilation.workspaces[i]));
    if (iree_status_is_ok(status)) {
      ++jit->compilation.workspace_count;
    }
  }
  return status;
}

static void loom_serve_jit_native_release(loom_serve_jit_native_t* native,
                                          iree_allocator_t host_allocator) {
  if (!native || iree_atomic_ref_count_dec(&native->ref_count) != 1) {
    return;
  }
  iree_status_free(native->status);
  loom_serve_executable_release(native->executable);
  loom_serve_native_artifact_release(native->artifact);
  iree_allocator_free(host_allocator, native);
}

static iree_status_t loom_serve_jit_publication_create(
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_jit_publication_t** out_publication) {
  *out_publication = NULL;
  loom_serve_jit_publication_t* publication = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*publication), (void**)&publication));
  iree_atomic_ref_count_init(&publication->ref_count);
  publication->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&publication->mutex);
  publication->report = report;
  publication->head = NULL;
  publication->tail = NULL;
  publication->publishing = false;
  *out_publication = publication;
  return iree_ok_status();
}

static void loom_serve_jit_publication_retain(
    loom_serve_jit_publication_t* publication) {
  if (publication) {
    iree_atomic_ref_count_inc(&publication->ref_count);
  }
}

static void loom_serve_jit_publication_release(
    loom_serve_jit_publication_t* publication) {
  if (!publication) {
    return;
  }
  if (iree_atomic_ref_count_dec(&publication->ref_count) != 1) {
    return;
  }
  iree_slim_mutex_deinitialize(&publication->mutex);
  iree_allocator_free(publication->host_allocator, publication);
}

static void loom_serve_jit_destroy(loom_serve_jit_t* jit) {
  for (iree_host_size_t i = 0; i < jit->compilation.workspace_count; ++i) {
    loomc_workspace_release(jit->compilation.workspaces[i]);
  }
  iree_allocator_free(jit->host_allocator, jit->compilation.workspaces);
  for (iree_host_size_t i = 0; i < jit->natives.count; ++i) {
    loom_serve_jit_native_release(jit->natives.values[i], jit->host_allocator);
  }
  iree_allocator_free(jit->host_allocator, jit->natives.values);
  loom_serve_native_domain_release(jit->native_domain);
  loom_serve_executable_cache_free(jit->executables);
  loomc_link_index_release(jit->index);
  loomc_linker_release(jit->linker);
  loom_serve_jit_compiler_deinitialize(&jit->kernel);
  iree_hal_queue_release(jit->dispatch);
  iree_hal_device_release(jit->device);
  loom_serve_jit_generation_release(jit->generation);
  iree_slim_mutex_deinitialize(&jit->mutex);
  loom_serve_jit_publication_release(jit->publication);
  iree_allocator_free(jit->host_allocator, jit);
}

void loom_serve_jit_retain(loom_serve_jit_t* jit) {
  if (jit) {
    iree_atomic_ref_count_inc(&jit->ref_count);
  }
}

void loom_serve_jit_release(loom_serve_jit_t* jit) {
  if (jit && iree_atomic_ref_count_dec(&jit->ref_count) == 1) {
    loom_serve_jit_destroy(jit);
  }
}

loom_serve_jit_cache_storage_t loom_serve_jit_cache_storage(
    const loom_serve_jit_t* jit) {
  const loom_serve_executable_cache_storage_t images =
      loom_serve_executable_cache_storage(jit->executables);
  loom_serve_jit_cache_storage_t storage = {
      .count = jit->natives.count,
      .executable_count = images.count,
      .host_bytes = jit->natives.capacity * sizeof(*jit->natives.values) +
                    images.host_bytes,
      .index_bytes = jit->natives.capacity * sizeof(*jit->natives.values) +
                     images.index_bytes,
      .code_bytes = images.code_bytes};
  for (iree_host_size_t i = 0; i < jit->natives.count; ++i) {
    const loom_serve_jit_native_t* native = jit->natives.values[i];
    storage.host_bytes +=
        sizeof(*native) +
        loom_serve_native_artifact_storage_size(native->artifact) +
        loom_serve_executable_storage_size(native->executable);
  }
  return storage;
}

void loom_serve_jit_trim(loom_serve_jit_t* jit) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < jit->natives.count; ++i) {
    loom_serve_jit_native_t* native = jit->natives.values[i];
    if (!loom_serve_executable_is_shared(native->executable)) {
      loom_serve_jit_native_release(native, jit->host_allocator);
    } else {
      jit->natives.values[count++] = native;
    }
  }
  jit->natives.count = count;
  loom_serve_executable_cache_trim(jit->executables);
  loom_serve_jit_generation_trim(jit->generation);
}

static iree_status_t loom_serve_jit_create_impl(
    loom_serve_jit_cache_t* cache, iree_hal_device_t* device,
    iree_hal_queue_t* dispatch, loom_serve_host_t* host,
    const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_jit_t** out_jit) {
  *out_jit = NULL;
  const iree_hal_physical_device_affinity_t affinity =
      iree_hal_queue_family_spec(iree_hal_queue_family(dispatch))
          ->physical_device_affinity;
  loom_serve_jit_t* jit = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*jit), (void**)&jit));
  iree_atomic_ref_count_init(&jit->ref_count);
  iree_slim_mutex_initialize(&jit->mutex);
  jit->host_allocator = host_allocator;
  jit->publication = NULL;
  jit->generation = NULL;
  iree_status_t status = loom_serve_jit_publication_create(
      report, host_allocator, &jit->publication);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_deinitialize(&jit->mutex);
    iree_allocator_free(host_allocator, jit);
    return status;
  }
  jit->device = device;
  iree_hal_device_retain(device);
  jit->dispatch = dispatch;
  iree_hal_queue_retain(dispatch);
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  jit->kernel.environment = loom_serve_host_target_environment(host);
  if (!jit->kernel.environment) {
    loom_serve_jit_release(jit);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "serving host has no compiler target configuration");
  }
  loomc_target_environment_retain(jit->kernel.environment);
  loom_serve_native_cache_t* native_cache = loom_serve_host_native_cache(host);
  const iree_allocator_t artifact_allocator =
      loom_serve_native_cache_allocator(native_cache);
  iree_host_size_t target_provider_count = 0;
  const loomc_iree_hal_target_provider_t* const* target_providers =
      loom_serve_host_target_providers(host, &target_provider_count);
  loom_serve_target_t target = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_target_select(jit->kernel.environment, target_providers,
                                      target_provider_count, device, affinity,
                                      report, artifact_allocator, &target);
    jit->kernel.profile = target.selection.target_profile;
    jit->target = target.selection.executable_target;
    jit->artifact_format = target.artifact_format;
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_compiler_initialize(
        sanitizer, target.control_flow_lowering, report, &jit->kernel,
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    // Successful pipeline construction accepts only one sanitizer descriptor;
    // its complete semantic fields participate in this compilation domain.
    const loom_serve_native_domain_options_t options = {
        .profile = jit->kernel.profile,
        .artifact_format = jit->artifact_format,
        .control_flow_lowering = target.control_flow_lowering,
        .sanitizer_checks = sanitizer ? sanitizer->checks : 0,
        .sanitizer_flags = sanitizer ? sanitizer->flags : 0,
        .sanitizer_reporting_mode = sanitizer ? sanitizer->reporting_mode : 0,
    };
    status = loom_serve_native_domain_acquire(native_cache, &options,
                                              &jit->native_domain);
  }
  loom_sha256_state_t identity;
  if (iree_status_is_ok(status) && cache) {
    loom_serve_jit_identity_initialize(sources, sanitizer, &target, &identity);
  }
  loom_sha256_t source_identity;
  if (iree_status_is_ok(status)) {
    status = loom_serve_source_index_create(
        jit->kernel.context, sources, NULL, report, host_allocator,
        cache ? &source_identity : NULL, &jit->index);
  }
  if (iree_status_is_ok(status) && cache) {
    loom_serve_jit_identity_update(&identity, source_identity.bytes,
                                   sizeof(source_identity.bytes));
  }
  if (iree_status_is_ok(status) && cache) {
    loom_sha256_t digest;
    loom_sha256_finalize(&identity, &digest);
    status = loom_serve_jit_cache_acquire(cache, &digest, &jit->generation);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_linker_create(jit->kernel.context, NULL, ca, &jit->linker));
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_executable_cache_allocate(
        iree_hal_queue_family(dispatch), jit->target, host_allocator,
        &jit->executables);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_initialize_compilation(jit, host);
  }
  if (iree_status_is_ok(status)) {
    *out_jit = jit;
  } else {
    loom_serve_jit_release(jit);
  }
  return status;
}

iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    loom_serve_host_t* host,
                                    const loom_serve_sources_t* sources,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    loom_serve_compilation_report_t* report,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit) {
  return loom_serve_jit_create_impl(NULL, device, dispatch, host, sources,
                                    sanitizer, report, host_allocator, out_jit);
}

iree_status_t loom_serve_jit_create_cached(
    loom_serve_jit_cache_t* cache, loom_serve_host_t* host,
    const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_compilation_report_t* report, loom_serve_jit_t** out_jit) {
  return loom_serve_jit_create_impl(
      cache, loom_serve_jit_cache_device(cache),
      loom_serve_jit_cache_dispatch(cache), host, sources, sanitizer, report,
      loom_serve_jit_cache_allocator(cache), out_jit);
}

loom_serve_compilation_report_t* loom_serve_jit_report(loom_serve_jit_t* jit) {
  return jit->publication->report;
}

loomc_task_sink_t loom_serve_jit_task_sink(const loom_serve_jit_t* jit) {
  return jit->compilation.sink;
}

void loom_serve_jit_report_snapshot(
    loom_serve_jit_t* jit, loom_serve_compilation_report_t* out_report) {
  iree_slim_mutex_lock(&jit->publication->mutex);
  *out_report = *jit->publication->report;
  iree_slim_mutex_unlock(&jit->publication->mutex);
}

void loom_serve_jit_report_merge(
    loom_serve_jit_t* jit, const loom_serve_compilation_report_t* report) {
  iree_slim_mutex_lock(&jit->publication->mutex);
  loom_serve_compilation_merge(jit->publication->report, report);
  iree_slim_mutex_unlock(&jit->publication->mutex);
}

loomc_target_environment_t* loom_serve_jit_target_environment(
    const loom_serve_jit_t* jit) {
  return jit->kernel.environment;
}

// Keeps prepared-low function facts in the same module through native emission.
static iree_status_t loom_serve_jit_emit_module(
    const loom_serve_jit_compiler_t* compiler, loomc_workspace_t* workspace,
    loomc_module_t* module, iree_host_size_t specialization_count,
    const loomc_target_specialization_t* specializations,
    iree_string_view_t format, loom_serve_compilation_report_t* report,
    iree_allocator_t host_allocator, loomc_byte_sequence_t** out_contents) {
  *out_contents = NULL;
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .specializations = specializations,
      .specialization_count = specialization_count,
  };
  const loomc_compile_options_t options = {
      .next = &target_options,
      .config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  loomc_result_t* result = NULL;
  loomc_status_t operation =
      loomc_compile_module(compiler->compiler, workspace, compiler->pipeline,
                           module, &options, ca, &result);
  iree_status_t status =
      loom_serve_compilation_result(operation, result, report);
  if (iree_status_is_ok(status)) {
    const loomc_emit_options_t emit_options = {
        .artifact_format = loomc_string_view_from_iree(format),
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    result = NULL;
    operation = loomc_emit_module(compiler->environment, workspace, module,
                                  &emit_options, ca, &result);
    // Retain the result while capturing an independent contents reference.
    loomc_result_retain(result);
    status = loom_serve_compilation_result(operation, result, report);
    if (iree_status_is_ok(status)) {
      const loomc_artifact_t* artifact = loomc_result_artifact_at(result, 0);
      loomc_byte_sequence_retain(artifact->contents);
      *out_contents = artifact->contents;
    }
    loomc_result_release(result);
  }
  return status;
}

struct loom_serve_jit_request_t {
  // Next request in parent publication order.
  loom_serve_jit_request_t* next;
  // Next subscriber while this native input is pending.
  loom_serve_jit_request_t* next_waiter;
  // Parent retained only while waiting for native publication.
  loom_serve_jit_job_t* job;
  // Immutable request transferred by the product builder.
  loomc_request_t* request;
  // Retained native result shared by equivalent requests.
  loom_serve_jit_native_t* native;
};

// Keys are process-local byte strings, not serialized cache artifacts. Each
// field carries its length so different source/root partitions cannot alias.
// Append source bytes last: a small metadata suffix could otherwise double
// the string builder's large backing allocation immediately before retention.
static iree_status_t loom_serve_jit_key_append(iree_string_builder_t* key,
                                               const void* data,
                                               iree_host_size_t length) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      key, iree_make_string_view((const char*)&length, sizeof(length))));
  return iree_string_builder_append_string(
      key, iree_make_string_view((const char*)data, length));
}

static iree_status_t loom_serve_jit_request_key(loomc_request_t* request,
                                                iree_string_builder_t* key) {
  const uintptr_t descriptor =
      (uintptr_t)loomc_request_product_descriptor(request);
  loomc_source_t* source = loomc_request_source(request);
  const loomc_source_format_t format = loomc_source_format(source);
  const loomc_string_view_t identifier = loomc_source_identifier(source);
  const loomc_byte_span_t bytes = loomc_source_contents(source);
  const iree_host_size_t count = loomc_request_root_count(request);
  IREE_RETURN_IF_ERROR(
      loom_serve_jit_key_append(key, &descriptor, sizeof(descriptor)));
  IREE_RETURN_IF_ERROR(loom_serve_jit_key_append(key, &format, sizeof(format)));
  IREE_RETURN_IF_ERROR(
      loom_serve_jit_key_append(key, identifier.data, identifier.size));
  IREE_RETURN_IF_ERROR(loom_serve_jit_key_append(key, &count, sizeof(count)));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    loomc_request_root_t root;
    loomc_request_root_at(request, i, &root);
    status = loom_serve_jit_key_append(key, &root.module_ordinal,
                                       sizeof(root.module_ordinal));
    if (iree_status_is_ok(status)) {
      status = loom_serve_jit_key_append(key, &root.symbol_ordinal,
                                         sizeof(root.symbol_ordinal));
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_key_append(key, bytes.data, bytes.data_length);
  }
  return status;
}

// Called under the catalog mutex for both acquisition and failed publication.
static iree_host_size_t loom_serve_jit_native_position(
    loom_serve_jit_t* jit, iree_const_byte_span_t key, bool* out_found) {
  *out_found = false;
  iree_host_size_t begin = 0;
  iree_host_size_t end = jit->natives.count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const iree_const_byte_span_t candidate =
        loom_serve_native_artifact_key(jit->natives.values[middle]->artifact);
    // These framed bytes contain NULs; text comparison would conflate inputs.
    const int order = key.data_length == candidate.data_length
                          ? memcmp(key.data, candidate.data, key.data_length)
                          : (key.data_length < candidate.data_length ? -1 : 1);
    if (order == 0) {
      *out_found = true;
      return middle;
    } else if (order < 0) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  return begin;
}

static void loom_serve_jit_artifact_ready(
    void* user_data, const loom_serve_native_result_t* result);

static iree_status_t loom_serve_jit_acquire_native(
    loom_serve_jit_t* jit, loomc_task_sink_t sink, iree_string_builder_t* key,
    iree_host_size_t root_count, loom_serve_jit_native_t** out_native,
    bool* out_created, bool* out_ready) {
  *out_created = false;
  *out_ready = false;
  const iree_string_view_t key_view = iree_string_builder_view(key);
  bool found = false;
  const iree_host_size_t begin = loom_serve_jit_native_position(
      jit, iree_make_const_byte_span(key_view.data, key_view.size), &found);
  if (found) {
    *out_native = jit->natives.values[begin];
    return iree_ok_status();
  }
  if (jit->natives.count == jit->natives.capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        jit->host_allocator, jit->natives.count + 1,
        sizeof(*jit->natives.values), &jit->natives.capacity,
        (void**)&jit->natives.values));
  }
  loom_serve_jit_native_t* native = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(jit->host_allocator,
                                             sizeof(*native), (void**)&native));
  iree_atomic_ref_count_init(&native->ref_count);
  native->jit = jit;
  native->sink = sink;
  native->subscription = (loom_serve_native_subscription_t){
      .ready = loom_serve_jit_artifact_ready, .user_data = native};
  iree_status_t status = loom_serve_native_artifact_acquire(
      jit->native_domain, key, root_count, &native->subscription,
      &native->artifact, out_created, out_ready);
  if (!iree_status_is_ok(status)) {
    loom_serve_jit_native_release(native, jit->host_allocator);
    return status;
  }
  memmove(&jit->natives.values[begin + 1], &jit->natives.values[begin],
          (jit->natives.count - begin) * sizeof(*jit->natives.values));
  jit->natives.values[begin] = native;
  ++jit->natives.count;
  *out_native = native;
  return iree_ok_status();
}

static void loom_serve_jit_subscribe(loom_serve_jit_request_t* node,
                                     loom_serve_jit_native_t* native);

static iree_status_t loom_serve_jit_acquire_request(
    loom_serve_jit_t* jit, loomc_task_sink_t sink,
    loom_serve_jit_request_t* node, iree_string_builder_t* key,
    iree_host_size_t root_count, bool* out_created) {
  iree_slim_mutex_lock(&jit->mutex);
  loom_serve_jit_native_t* native = NULL;
  bool ready = false;
  iree_status_t status = loom_serve_jit_acquire_native(
      jit, sink, key, root_count, &native, out_created, &ready);
  if (iree_status_is_ok(status)) {
    loom_serve_jit_subscribe(node, native);
  }
  iree_slim_mutex_unlock(&jit->mutex);
  if (ready) {
    loom_serve_jit_artifact_ready(
        native, loom_serve_native_artifact_result(native->artifact));
  }
  return status;
}

static iree_status_t loom_serve_jit_plan_request(loom_serve_jit_t* jit,
                                                 loomc_task_sink_t sink,
                                                 loom_serve_jit_request_t* node,
                                                 bool* out_created) {
  const iree_host_size_t root_count = loomc_request_root_count(node->request);
  iree_string_builder_t key;
  iree_string_builder_initialize(
      loom_serve_native_domain_allocator(jit->native_domain), &key);
  iree_status_t status = loom_serve_jit_request_key(node->request, &key);
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_acquire_request(jit, sink, node, &key, root_count,
                                            out_created);
  }
  iree_string_builder_deinitialize(&key);
  return status;
}

void loom_serve_jit_stage_retain(loom_serve_jit_stage_t* stage) {
  if (stage) {
    iree_atomic_ref_count_inc(&stage->ref_count);
  }
}

bool loom_serve_jit_stage_is_shared(const loom_serve_jit_stage_t* stage) {
  return stage && iree_atomic_ref_count_load(&stage->ref_count) > 1;
}

void loom_serve_jit_stage_free(loom_serve_jit_stage_t* stage) {
  if (!stage || iree_atomic_ref_count_dec(&stage->ref_count) != 1) {
    return;
  }
  for (iree_host_size_t i = 0; i < stage->executable_count; ++i) {
    loom_serve_executable_release(stage->executables[i]);
  }
  iree_allocator_free(stage->host_allocator, stage->entries);
  iree_allocator_free(stage->host_allocator, (void*)stage->image.data);
  iree_allocator_free(stage->host_allocator, stage);
}

iree_host_size_t loom_serve_jit_stage_storage_size(
    const loom_serve_jit_stage_t* stage) {
  return sizeof(*stage) + stage->image.data_length +
         stage->entry_count * sizeof(*stage->entries) +
         stage->executable_count * sizeof(*stage->executables);
}

// A catalog is a library universe, so a named entry must have exactly one
// exported definition. The index's name lookup enumerates declarations too;
// it does not perform definition selection for exact-ordinal product requests.
static iree_status_t loom_serve_jit_lookup_root(
    loom_serve_jit_t* jit, iree_string_view_t root,
    loomc_host_size_t* out_ordinal) {
  loomc_link_index_symbol_t symbol;
  bool found = false;
  bool has_symbol = loomc_link_index_lookup_global(
      jit->index, loomc_string_view_from_iree(root), &symbol);
  while (has_symbol) {
    if (iree_all_bits_set(symbol.flags,
                          LOOMC_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION |
                              LOOMC_LINK_SYMBOL_FLAG_EXPORT)) {
      if (found) {
        return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "stage root @%.*s has multiple exported "
                                "definitions",
                                (int)root.size, root.data);
      }
      *out_ordinal = symbol.ordinal;
      found = true;
    }
    has_symbol =
        loomc_link_index_next_global_duplicate(jit->index, &symbol, &symbol);
  }
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "stage root @%.*s has no exported definition",
                            (int)root.size, root.data);
  }
  return iree_ok_status();
}

struct loom_serve_jit_job_t {
  // Initial task plus accepted native requests. Last release binds the stage.
  iree_atomic_ref_count_t ref_count;
  // Initial product-construction task, sharing storage with the joined job.
  loomc_task_t base;
  // Retained catalog through compiler retirement before external completion.
  loom_serve_jit_t* jit;
  // Scheduling domain inherited by this parent's newly published natives.
  loomc_task_sink_t sink;
  // Retained callback state independent of catalog/device ownership.
  loom_serve_jit_publication_t* publication;
  // Next completed job waiting for callback publication.
  loom_serve_jit_job_t* next;
  // Command symbol borrowed through completion.
  iree_string_view_t root;
  // Selects the native kernel product instead of a portable command product.
  bool direct_dispatch;
  // Complete immutable stage came from the shared source generation.
  bool cache_hit;
  // Exact root and specialization identity within the source generation.
  loom_sha256_t cache_key;
  // Specialization bindings borrowed through completion.
  const loomc_config_options_t* config;
  // Optional early immutable command-layout consumer.
  loom_serve_jit_layout_fn_t layout;
  // Completion receiving owned result and status after all requests retire.
  loom_serve_jit_completion_fn_t completion;
  // Opaque callback state borrowed through completion.
  void* user_data;
  // Start of accepted compilation in host nanoseconds.
  iree_time_t start;
  // Accepted requests and terminal statuses, retained through the final join.
  loom_serve_jit_request_t* requests;
  // Last request, preserving publication order without rescanning.
  loom_serve_jit_request_t* requests_tail;
  // Disjoint parent and joined native evidence, published at completion.
  loom_serve_compilation_report_t report;
  // Owns early immutable command bytes and the final joined executable
  // bindings.
  loom_serve_jit_stage_t* stage;
  // Owned immutable command product used only after parent construction.
  loomc_product_t* product;
  // Parsed command description borrowing product storage.
  loomc_cmd_program_t program;
  // Joined executable bindings indexed by parent requirement ordinal.
  loom_cmd_program_hal_entry_t* requirements;
  // Number of allocated requirement slots.
  iree_host_size_t requirement_count;
  // Number of native requests submitted for the command.
  iree_host_size_t request_count;
  // Requests satisfied by a prior or coalesced native input.
  iree_host_size_t reused_count;
  // Product construction or native submission failure.
  iree_status_t status;
};

static void loom_serve_jit_stage_identity(bool direct_dispatch,
                                          iree_string_view_t root,
                                          const loomc_config_options_t* config,
                                          loom_sha256_t* out_identity) {
  loom_sha256_state_t identity;
  loom_sha256_initialize(&identity);
  loom_serve_jit_identity_update_view(&identity,
                                      IREE_SV("loom.serve.compiled-stage.v1"));
  loom_serve_jit_identity_update_u64(&identity, direct_dispatch);
  loom_serve_jit_identity_update_view(&identity, root);
  loom_serve_jit_identity_update_u64(&identity, config->flags);
  loom_serve_jit_identity_update_u64(&identity, config->binding_count);
  for (iree_host_size_t i = 0; i < config->binding_count; ++i) {
    loom_serve_jit_identity_update_view(
        &identity, iree_string_view_from_loomc(config->bindings[i].key));
    loom_serve_jit_identity_update_view(
        &identity, iree_string_view_from_loomc(config->bindings[i].value));
  }
  loom_serve_jit_identity_update_view(
      &identity, iree_string_view_from_loomc(config->json_object));
  loom_sha256_finalize(&identity, out_identity);
}

static void loom_serve_jit_subscribe(loom_serve_jit_request_t* node,
                                     loom_serve_jit_native_t* native) {
  node->native = native;
  iree_atomic_ref_count_inc(&native->ref_count);
  if (!native->complete) {
    iree_atomic_ref_count_inc(&node->job->ref_count);
    node->next_waiter = native->waiters;
    native->waiters = node;
  }
}

static iree_status_t loom_serve_jit_request_allocate(
    loom_serve_jit_job_t* job, loomc_request_t* request,
    loom_serve_jit_request_t** out_node) {
  loom_serve_jit_request_t* node = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(job->jit->host_allocator,
                                             sizeof(*node), (void**)&node));
  node->job = job;
  node->request = request;
  if (job->requests_tail) {
    job->requests_tail->next = node;
  } else {
    job->requests = node;
  }
  job->requests_tail = node;
  *out_node = node;
  return iree_ok_status();
}

static iree_status_t loom_serve_jit_job_allocate_stage(
    loom_serve_jit_job_t* job) {
  loom_serve_jit_t* jit = job->jit;
  loom_serve_jit_stage_t* stage = NULL;
  const iree_host_size_t executable_count =
      job->request_count + job->reused_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      jit->host_allocator, sizeof(*stage), executable_count,
      sizeof(*stage->executables), (void**)&stage));
  iree_atomic_ref_count_init(&stage->ref_count);
  stage->host_allocator = jit->host_allocator;
  job->stage = stage;
  if (job->direct_dispatch) {
    return iree_ok_status();
  }
  loomc_byte_span_t image;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_byte_sequence_clone(
      job->program.artifact.contents,
      loomc_allocator_from_iree(jit->host_allocator), &image)));
  stage->image = iree_const_byte_span_from_loomc(image);
  return loom_cmd_program_parse(stage->image, &stage->program);
}

static iree_status_t loom_serve_jit_job_bind(loom_serve_jit_job_t* job) {
  loom_serve_jit_t* jit = job->jit;
  if (job->direct_dispatch) {
    IREE_RETURN_IF_ERROR(loom_serve_jit_job_allocate_stage(job));
  }
  loom_serve_jit_stage_t* stage = job->stage;
  for (loom_serve_jit_request_t* node = job->requests; node;
       node = node->next) {
    loom_serve_executable_t* executable = node->native->executable;
    loom_serve_executable_retain(executable);
    stage->executables[stage->executable_count++] = executable;
  }
  if (job->direct_dispatch) {
    stage->dispatch = loom_serve_executable_dispatch(stage->executables[0]);
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (job->program.entry_requirement_count) {
    status = iree_allocator_malloc_array(
        jit->host_allocator, job->program.entry_requirement_count,
        sizeof(*stage->entries), (void**)&stage->entries);
  }
  if (iree_status_is_ok(status)) {
    stage->entry_count = job->program.entry_requirement_count;
    for (iree_host_size_t i = 0;
         i < stage->entry_count && iree_status_is_ok(status); ++i) {
      stage->entries[i] =
          job->requirements[job->program.entry_requirement_ordinals[i]];
      if (!stage->entries[i].executable) {
        status =
            iree_make_status(IREE_STATUS_NOT_FOUND,
                             "command @%.*s entry %zu has no source provider",
                             (int)job->root.size, job->root.data, i);
      }
    }
  }
  return status;
}

// Consumes the completed job and retires its device-owning catalog reference
// before invoking external code. Only the independent publication state lives
// across the callback.
static void loom_serve_jit_job_complete(loom_serve_jit_job_t* job) {
  loom_serve_jit_t* jit = job->jit;
  loom_serve_jit_publication_t* publication = job->publication;
  const loom_serve_jit_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_status_t status = job->status;
  loom_serve_jit_stage_t* stage = job->stage;
  iree_allocator_free(publication->host_allocator, job);
  loom_serve_jit_release(jit);
  if (completion) {
    completion(user_data, status, stage);
  }
  loom_serve_jit_publication_release(publication);
}

// Reports and callbacks publish as jobs become ready, without waiting for an
// unrelated earlier product. Publication ownership permits callback-side
// catalog release and reentrant submission without keeping device ownership
// live after the callback becomes externally visible.
static void loom_serve_jit_job_publish(loom_serve_jit_job_t* completed) {
  loom_serve_jit_publication_t* publication = completed->publication;
  loom_serve_jit_publication_retain(publication);
  iree_slim_mutex_lock(&publication->mutex);
  if (publication->tail) {
    publication->tail->next = completed;
  } else {
    publication->head = completed;
  }
  publication->tail = completed;
  if (!publication->publishing) {
    publication->publishing = true;
    while (publication->head) {
      loom_serve_jit_job_t* job = publication->head;
      publication->head = job->next;
      if (!publication->head) {
        publication->tail = NULL;
      }
      loom_serve_compilation_merge(publication->report, &job->report);
      iree_slim_mutex_unlock(&publication->mutex);
      loom_serve_jit_job_complete(job);
      iree_slim_mutex_lock(&publication->mutex);
    }
    publication->publishing = false;
  }
  iree_slim_mutex_unlock(&publication->mutex);
  loom_serve_jit_publication_release(publication);
}

static void loom_serve_jit_job_release(loom_serve_jit_job_t* job) {
  if (!job) {
    return;
  }
  if (iree_atomic_ref_count_dec(&job->ref_count) != 1) {
    return;
  }
  loom_serve_jit_t* jit = job->jit;
  iree_status_t status = job->status;
  for (loom_serve_jit_request_t* node = job->requests; node;
       node = node->next) {
    if (node->native) {
      status =
          iree_status_join(status, iree_status_clone(node->native->status));
      loom_serve_compilation_diagnostics_merge(
          loom_serve_native_artifact_result(node->native->artifact)
              ->diagnostics,
          &job->report);
    }
  }
  if (iree_status_is_ok(status) && !job->cache_hit) {
    for (loom_serve_jit_request_t* node = job->requests;
         !job->direct_dispatch && node; node = node->next) {
      for (iree_host_size_t i = 0;
           i < loomc_request_binding_count(node->request); ++i) {
        loomc_request_binding_t binding;
        loomc_request_binding_at(node->request, i, &binding);
        loom_cmd_program_hal_entry_t* entry =
            &job->requirements[binding.requirement_ordinal];
        *entry = (loom_cmd_program_hal_entry_t){
            .executable =
                loom_serve_executable_handle(node->native->executable),
            .function = loom_serve_executable_function(node->native->executable,
                                                       binding.root_ordinal),
        };
        iree_hal_executable_retain(entry->executable);
      }
    }
    status = loom_serve_jit_job_bind(job);
  }
  while (job->requests) {
    loom_serve_jit_request_t* node = job->requests;
    job->requests = node->next;
    loomc_request_release(node->request);
    loom_serve_jit_native_release(node->native, jit->host_allocator);
    iree_allocator_free(jit->host_allocator, node);
  }
  if (job->requirements) {
    for (iree_host_size_t i = 0; i < job->requirement_count; ++i) {
      iree_hal_executable_release(job->requirements[i].executable);
    }
  }
  iree_allocator_free(jit->host_allocator, job->requirements);
  loomc_product_release(job->product);
  if (iree_status_is_ok(status) && !job->cache_hit && jit->generation) {
    status = loom_serve_jit_generation_publish_stage(
        jit->generation, &job->cache_key, job->stage, &job->report);
  }
  job->report.commands += !job->direct_dispatch;
  job->report.kernels += job->request_count;
  job->report.reused_kernels += job->reused_count;
  job->report.entries +=
      job->stage ? (job->stage->dispatch ? 1 : job->stage->entry_count) : 0;
  job->report.workers = job->cache_hit ? 0 : jit->compilation.workspace_count;
  job->report.stage_duration_sum_ns += iree_time_now() - job->start;
  job->status = status;
  loom_serve_jit_job_publish(job);
}

static void loom_serve_jit_native_complete(loom_serve_jit_t* jit,
                                           loom_serve_jit_native_t* native) {
  // Each detached subscriber retains its parent job and catalog. The final
  // release may destroy the catalog before invoking its callback, so this path
  // makes no catalog access after releasing the last detached subscriber.
  iree_slim_mutex_lock(&jit->mutex);
  native->complete = true;
  loom_serve_jit_request_t* waiters = native->waiters;
  native->waiters = NULL;
  if (!iree_status_is_ok(native->status)) {
    bool found = false;
    const iree_host_size_t index = loom_serve_jit_native_position(
        jit, loom_serve_native_artifact_key(native->artifact), &found);
    IREE_ASSERT(found && jit->natives.values[index] == native);
    memmove(&jit->natives.values[index], &jit->natives.values[index + 1],
            (--jit->natives.count - index) * sizeof(*jit->natives.values));
    loom_serve_jit_native_release(native, jit->host_allocator);
  }
  iree_slim_mutex_unlock(&jit->mutex);
  while (waiters) {
    loom_serve_jit_request_t* node = waiters;
    waiters = node->next_waiter;
    loom_serve_jit_job_release(node->job);
  }
}

static void loom_serve_jit_native_loaded(void* user_data, iree_status_t status,
                                         loom_serve_executable_t* executable) {
  loom_serve_jit_native_t* native = user_data;
  loom_serve_jit_t* jit = native->jit;
  native->status = status;
  native->executable = executable;
  loom_serve_jit_native_complete(jit, native);
}

static void loom_serve_jit_load_native(loom_serve_jit_native_t* native) {
  loom_serve_jit_t* jit = native->jit;
  native->load.completion = loom_serve_jit_native_loaded;
  native->load.user_data = native;
  loom_serve_executable_cache_load(jit->executables, &native->load);
}

// A device load may complete inline and retire its catalog. This task storage
// therefore belongs to the host allocator and never touches the native view
// after loading returns. Sibling device loads progress on independent workers.
typedef struct loom_serve_jit_load_task_t {
  // Shared queue-owned continuation.
  loomc_task_t base;
  // Host allocation policy, independent of callback-side catalog retirement.
  iree_allocator_t host_allocator;
  // Borrowed view retained by its pending parents through load completion.
  loom_serve_jit_native_t* native;
} loom_serve_jit_load_task_t;

static void loom_serve_jit_load_task_execute(loomc_task_t* base,
                                             loomc_host_size_t worker_ordinal) {
  loom_serve_jit_load_task_t* task = (loom_serve_jit_load_task_t*)base;
  loom_serve_jit_load_native(task->native);
}

static void loom_serve_jit_load_task_destroy(loomc_task_t* base) {
  loom_serve_jit_load_task_t* task = (loom_serve_jit_load_task_t*)base;
  iree_allocator_free(task->host_allocator, task);
}

static const loomc_task_vtable_t jit_load_task_vtable = {
    .execute = loom_serve_jit_load_task_execute,
    .destroy = loom_serve_jit_load_task_destroy,
};

static void loom_serve_jit_artifact_ready(
    void* user_data, const loom_serve_native_result_t* result) {
  loom_serve_jit_native_t* native = user_data;
  if (!iree_status_is_ok(result->status)) {
    loom_serve_jit_native_loaded(native, iree_status_clone(result->status),
                                 NULL);
    return;
  }
  native->load.artifact = native->artifact;
  const iree_allocator_t allocator =
      loom_serve_native_domain_allocator(native->jit->native_domain);
  loom_serve_jit_load_task_t* task = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*task), (void**)&task);
  if (iree_status_is_ok(status)) {
    loomc_task_initialize(&jit_load_task_vtable, &task->base);
    task->host_allocator = allocator;
    task->native = native;
    status = iree_status_from_loomc(
        loomc_task_sink_submit(native->sink, &task->base));
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, task);
    loom_serve_jit_native_loaded(native, status, NULL);
  }
}

static iree_status_t loom_serve_jit_compile_request(
    loom_serve_jit_t* jit, loomc_workspace_t* workspace,
    loom_serve_jit_request_t* node, loom_serve_compilation_report_t* report) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_serve_native_result_t* output =
      loom_serve_native_artifact_result(node->native->artifact);
  const iree_allocator_t artifact_allocator =
      loom_serve_native_domain_allocator(jit->native_domain);
  const loomc_allocator_t host_allocator =
      loomc_allocator_from_iree(jit->host_allocator);
  loomc_string_view_t* roots = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_array(jit->host_allocator, output->root_count,
                                      sizeof(*roots), (void**)&roots));
  loomc_target_specialization_t* specializations = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      jit->host_allocator, output->root_count, sizeof(*specializations),
      (void**)&specializations);
  loomc_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    loomc_result_t* result = NULL;
    const loomc_status_t operation = loomc_module_deserialize_request(
        jit->kernel.context, workspace, node->request, host_allocator, &module,
        roots, &result);
    status = loom_serve_compilation_result(operation, result, report);
  }
  for (iree_host_size_t i = 0;
       i < output->root_count && iree_status_is_ok(status); ++i) {
    // Names must survive module mutation through compile and native emission.
    status = iree_status_from_loomc(loomc_string_view_clone(
        roots[i], loomc_allocator_from_iree(artifact_allocator),
        &specializations[i].function_symbol));
    output->roots[i] =
        iree_string_view_from_loomc(specializations[i].function_symbol);
    specializations[i].target_profile = jit->kernel.profile;
  }
  iree_allocator_free(jit->host_allocator, roots);
  if (iree_status_is_ok(status)) {
    loomc_byte_sequence_t* contents = NULL;
    status = loom_serve_jit_emit_module(
        &jit->kernel, workspace, module, output->root_count, specializations,
        jit->artifact_format, report, artifact_allocator, &contents);
    output->image = iree_byte_sequence_from_loomc(contents);
  }
  loomc_module_release(module);
  iree_allocator_free(jit->host_allocator, specializations);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct loom_serve_jit_task_t {
  // Queue-owned native work record.
  loomc_task_t base;
  // Borrowed compilation retained by its pending native subscription.
  loom_serve_jit_job_t* job;
  // Disjoint request and terminal result slot owned by the job.
  loom_serve_jit_request_t* request;
} loom_serve_jit_task_t;

static void loom_serve_jit_task_execute(loomc_task_t* base,
                                        loomc_host_size_t worker_ordinal) {
  loom_serve_jit_task_t* task = (loom_serve_jit_task_t*)base;
  loom_serve_jit_job_t* job = task->job;
  loom_serve_compilation_report_t report = {0};
  loom_serve_native_result_t* output =
      loom_serve_native_artifact_result(task->request->native->artifact);
  output->status = loom_serve_jit_compile_request(
      job->jit, job->jit->compilation.workspaces[worker_ordinal], task->request,
      &report);
  output->status = iree_status_join(
      output->status,
      loom_serve_compilation_diagnostics_clone(
          &report, loom_serve_native_domain_allocator(job->jit->native_domain),
          &output->diagnostics));
}

static void loom_serve_jit_task_destroy(loomc_task_t* base) {
  loom_serve_jit_task_t* task = (loom_serve_jit_task_t*)base;
  loom_serve_jit_t* jit = task->job->jit;
  loom_serve_native_artifact_t* artifact = task->request->native->artifact;
  iree_allocator_free(jit->host_allocator, task);
  loom_serve_native_artifact_publish(artifact);
}

static const loomc_task_vtable_t jit_task_vtable = {
    .execute = loom_serve_jit_task_execute,
    .destroy = loom_serve_jit_task_destroy,
};

static iree_status_t loom_serve_jit_submit_request(
    loom_serve_jit_job_t* job, loom_serve_jit_request_t* request) {
  loom_serve_jit_task_t* task = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(job->jit->host_allocator,
                                             sizeof(*task), (void**)&task));
  loomc_task_initialize(&jit_task_vtable, &task->base);
  task->job = job;
  task->request = request;
  iree_status_t status =
      iree_status_from_loomc(loomc_task_sink_submit(job->sink, &task->base));
  if (!iree_status_is_ok(status)) {
    // Rejection never executed the task. Its pending entry is terminalized by
    // the publisher, including any concurrent parents already subscribed.
    iree_allocator_free(job->jit->host_allocator, task);
  }
  return status;
}

static loomc_status_t loom_serve_jit_publish(void* user_data,
                                             loomc_request_t* request) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_serve_jit_job_t* job = user_data;
  loom_serve_jit_request_t* node = NULL;
  iree_status_t status = loom_serve_jit_request_allocate(job, request, &node);
  if (iree_status_is_ok(status)) {
    bool created = false;
    status = loom_serve_jit_plan_request(job->jit, job->sink, node, &created);
    if (iree_status_is_ok(status)) {
      if (created) {
        // The parent reference spans publication, including inline completion
        // or rejection. Other requests may immediately reuse this native slot.
        status = loom_serve_jit_submit_request(job, node);
        if (iree_status_is_ok(status)) {
          ++job->request_count;
        } else {
          loom_serve_native_artifact_result(node->native->artifact)->status =
              iree_status_clone(status);
          loom_serve_native_artifact_publish(node->native->artifact);
        }
      } else {
        ++job->reused_count;
      }
    }
  } else {
    loomc_request_release(request);
  }
  IREE_TRACE_ZONE_END(z0);
  return loomc_status_from_iree(status);
}

// Emits code and launch facts without creating device objects. Image loading
// can coalesce with another producer before resolving this input's entries.
static iree_status_t loom_serve_jit_emit_dispatch(
    loom_serve_jit_job_t* job, loomc_workspace_t* workspace,
    loomc_module_t* module, loom_serve_jit_native_t* native,
    loom_serve_compilation_report_t* report) {
  loom_serve_jit_t* jit = job->jit;
  loom_serve_native_result_t* output =
      loom_serve_native_artifact_result(native->artifact);
  const loomc_allocator_t host_allocator = loomc_allocator_from_iree(
      loom_serve_native_domain_allocator(jit->native_domain));
  const loomc_string_view_t root = loomc_string_view_from_iree(job->root);
  loomc_string_view_t retained_root;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_string_view_clone(root, host_allocator, &retained_root)));
  output->roots[0] = iree_string_view_from_loomc(retained_root);
  output->direct_dispatch = true;
  const loomc_emit_options_t emit_options = {
      .artifact_format = loomc_string_view_from_iree(jit->artifact_format),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_compile_artifact_options_t options = {
      .roots = &root,
      .root_count = 1,
      .target_profile = jit->kernel.profile,
      .config = job->config,
      .emit_options = &emit_options,
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
  };
  loomc_result_t* result = NULL;
  ++job->request_count;
  const loomc_status_t operation = loomc_compile_artifact(
      jit->kernel.compiler, workspace, jit->kernel.pipeline, module, &options,
      host_allocator, &result);
  loomc_result_retain(result);
  iree_status_t status =
      loom_serve_compilation_result(operation, result, report);
  loomc_launch_config_program_t* launch = NULL;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < loomc_result_artifact_count(result);
       ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind == LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) {
      status = iree_status_from_loomc(
          loomc_launch_config_program_load(artifact, host_allocator, &launch));
    } else if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      loomc_byte_sequence_retain(artifact->contents);
      output->image = iree_byte_sequence_from_loomc(artifact->contents);
    }
  }
  loomc_result_release(result);
  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_launch_config_program_lookup_function(
        launch, root, &launch_function));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_launch_config_program_invoke(
        launch, launch_function, NULL, 0, &output->launch_config));
  }
  loomc_launch_config_program_release(launch);
  return status;
}

static iree_status_t loom_serve_jit_compile_dispatch(
    loom_serve_jit_job_t* job, loomc_workspace_t* workspace) {
  loom_serve_jit_t* jit = job->jit;
  const loomc_allocator_t host_allocator =
      loomc_allocator_from_iree(jit->host_allocator);
  const loomc_string_view_t root = loomc_string_view_from_iree(job->root);
  const loomc_target_specialization_t specialization = {
      .function_symbol = root,
      .target_profile = jit->kernel.profile,
  };
  const loomc_target_specialization_options_t targets = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .specializations = &specialization,
      .specialization_count = 1,
  };
  const loomc_link_options_t options = {
      .next = &targets,
      .link_index = jit->index,
      .mode = LOOMC_LINK_MODE_LINK,
      .root_symbols = &root,
      .root_symbol_count = 1,
      .config = *job->config,
  };
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  const loomc_status_t operation =
      loomc_link_module(jit->linker, workspace, &options, &module, &result);
  iree_status_t status =
      loom_serve_compilation_result(operation, result, &job->report);
  loomc_source_t* source = NULL;
  const loomc_module_serialize_options_t serialize = {
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
  };
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_serialize_to_source(
        module, &serialize, host_allocator, &source));
  }
  iree_string_builder_t key;
  iree_string_builder_initialize(
      loom_serve_native_domain_allocator(jit->native_domain), &key);
  if (iree_status_is_ok(status)) {
    // Native-root keys have a distinct product domain from command requests.
    const uintptr_t descriptor = 0;
    const loomc_byte_span_t bytes = loomc_source_contents(source);
    status = loom_serve_jit_key_append(&key, &descriptor, sizeof(descriptor));
    if (iree_status_is_ok(status)) {
      status = loom_serve_jit_key_append(&key, root.data, root.size);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_jit_key_append(&key, bytes.data, bytes.data_length);
    }
  }
  loom_serve_jit_request_t* node = NULL;
  bool created = false;
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_request_allocate(job, NULL, &node);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_jit_acquire_request(jit, job->sink, node, &key, 1, &created);
  }
  iree_string_builder_deinitialize(&key);
  loomc_source_release(source);
  if (iree_status_is_ok(status) && created) {
    loom_serve_compilation_report_t report = {0};
    loom_serve_native_result_t* output =
        loom_serve_native_artifact_result(node->native->artifact);
    output->status = loom_serve_jit_emit_dispatch(job, workspace, module,
                                                  node->native, &report);
    output->status = iree_status_join(
        output->status,
        loom_serve_compilation_diagnostics_clone(
            &report, loom_serve_native_domain_allocator(jit->native_domain),
            &output->diagnostics));
  } else if (iree_status_is_ok(status)) {
    ++job->reused_count;
  }
  loomc_module_release(module);
  if (iree_status_is_ok(status) && created) {
    loom_serve_native_artifact_publish(node->native->artifact);
  }
  return status;
}

static void loom_serve_jit_job_execute(loomc_task_t* base,
                                       loomc_host_size_t worker_ordinal) {
  loom_serve_jit_job_t* job =
      (loom_serve_jit_job_t*)((uint8_t*)base -
                              offsetof(loom_serve_jit_job_t, base));
  loom_serve_jit_t* jit = job->jit;
  if (jit->generation) {
    job->stage = loom_serve_jit_generation_lookup_stage(
        jit->generation, &job->cache_key, &job->report);
    if (job->stage) {
      job->cache_hit = true;
      job->reused_count = job->stage->executable_count;
      if (job->layout) {
        job->layout(job->user_data, &job->stage->program);
      }
      return;
    }
  }
  loomc_workspace_t* workspace = jit->compilation.workspaces[worker_ordinal];
  loomc_host_size_t root_ordinal = 0;
  iree_status_t status =
      loom_serve_jit_lookup_root(jit, job->root, &root_ordinal);
  if (job->direct_dispatch) {
    job->status = iree_status_is_ok(status)
                      ? loom_serve_jit_compile_dispatch(job, workspace)
                      : status;
    return;
  }
  if (iree_status_is_ok(status)) {
    const loomc_cmd_program_product_options_t options = {
        .link_index = jit->index,
        .root_symbol_ordinals = &root_ordinal,
        .root_symbol_count = 1,
        .config = *job->config,
        .target_profile = jit->kernel.profile,
        .request_sink = {.publish = loom_serve_jit_publish, .user_data = job},
    };
    loomc_result_t* result = NULL;
    IREE_TRACE_ZONE_BEGIN_NAMED(z_product, "loom_serve_jit_command_product");
    const loomc_status_t operation = loomc_cmd_program_product_build(
        workspace, &options, loomc_allocator_from_iree(jit->host_allocator),
        &job->product, &result);
    IREE_TRACE_ZONE_END(z_product);
    status = loom_serve_compilation_result(operation, result, &job->report);
  }
  if (iree_status_is_ok(status)) {
    loomc_cmd_program_product_program_at(job->product, 0, &job->program);
    job->requirement_count = loomc_product_requirement_count(job->product);
    if (job->requirement_count) {
      status = iree_allocator_malloc_array(
          jit->host_allocator, job->requirement_count,
          sizeof(*job->requirements), (void**)&job->requirements);
    }
  }
  // The initial task keeps a reference while expanding the native work. Its
  // final release joins the children without occupying a waiting worker.
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_job_allocate_stage(job);
  }
  if (iree_status_is_ok(status) && job->layout) {
    job->layout(job->user_data, &job->stage->program);
  }
  job->status = status;
}

static void loom_serve_jit_job_destroy(loomc_task_t* base) {
  loom_serve_jit_job_release(
      (loom_serve_jit_job_t*)((uint8_t*)base -
                              offsetof(loom_serve_jit_job_t, base)));
}

static const loomc_task_vtable_t jit_job_vtable = {
    .execute = loom_serve_jit_job_execute,
    .destroy = loom_serve_jit_job_destroy,
};

static iree_status_t loom_serve_jit_enqueue(
    loom_serve_jit_t* jit, loomc_task_sink_t sink, iree_string_view_t root,
    const loomc_config_options_t* config, bool direct_dispatch,
    loom_serve_jit_layout_fn_t layout,
    loom_serve_jit_completion_fn_t completion, void* user_data) {
  loom_serve_jit_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(jit->host_allocator, sizeof(*job), (void**)&job));
  loomc_task_initialize(&jit_job_vtable, &job->base);
  iree_atomic_ref_count_init(&job->ref_count);
  job->jit = jit;
  job->sink = sink;
  job->publication = jit->publication;
  job->root = root;
  job->direct_dispatch = direct_dispatch;
  loom_serve_jit_stage_identity(direct_dispatch, root, config, &job->cache_key);
  job->config = config;
  job->layout = layout;
  job->completion = completion;
  job->user_data = user_data;
  job->start = iree_time_now();
  loom_serve_jit_retain(jit);
  loom_serve_jit_publication_retain(job->publication);
  iree_status_t status =
      iree_status_from_loomc(loomc_task_sink_submit(sink, &job->base));
  if (!iree_status_is_ok(status)) {
    // Rejection retained ownership and never executed product construction.
    // Retire ownership without a callback or compilation evidence.
    job->completion = NULL;
    loom_serve_jit_job_complete(job);
  }
  return status;
}

iree_status_t loom_serve_jit_compile_async(
    loom_serve_jit_t* jit, loomc_task_sink_t sink, iree_string_view_t root,
    const loomc_config_options_t* config, loom_serve_jit_layout_fn_t layout,
    loom_serve_jit_completion_fn_t completion, void* user_data) {
  return loom_serve_jit_enqueue(jit, sink, root, config, false, layout,
                                completion, user_data);
}

iree_status_t loom_serve_jit_compile_dispatch_async(
    loom_serve_jit_t* jit, loomc_task_sink_t sink, iree_string_view_t root,
    const loomc_config_options_t* config,
    loom_serve_jit_completion_fn_t completion, void* user_data) {
  return loom_serve_jit_enqueue(jit, sink, root, config, true, NULL, completion,
                                user_data);
}

typedef struct loom_serve_jit_join_t {
  // Outer application's wait; never used by a task worker.
  iree_notification_t notification;
  // Joins publication with the callback's final notification access.
  iree_slim_mutex_t mutex;
  // Result/status are ready after the callback has posted its notification.
  bool completed;
  // Owned compilation outcome transferred by the callback.
  iree_status_t status;
  // Owned compiled stage, or NULL on failure.
  loom_serve_jit_stage_t* stage;
} loom_serve_jit_join_t;

static void loom_serve_jit_join_complete(void* user_data, iree_status_t status,
                                         loom_serve_jit_stage_t* stage) {
  loom_serve_jit_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  join->status = status;
  join->stage = stage;
  join->completed = true;
  iree_notification_post(&join->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&join->mutex);
}

static bool loom_serve_jit_join_is_complete(void* user_data) {
  loom_serve_jit_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  const bool completed = join->completed;
  iree_slim_mutex_unlock(&join->mutex);
  return completed;
}

iree_status_t loom_serve_jit_compile(loom_serve_jit_t* jit,
                                     iree_string_view_t root,
                                     const loomc_config_options_t* config,
                                     loom_serve_jit_stage_t** out_stage) {
  *out_stage = NULL;
  loom_serve_jit_join_t join = {0};
  iree_notification_initialize(&join.notification);
  iree_slim_mutex_initialize(&join.mutex);
  iree_status_t status =
      loom_serve_jit_compile_async(jit, jit->compilation.sink, root, config,
                                   NULL, loom_serve_jit_join_complete, &join);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&join.notification, loom_serve_jit_join_is_complete,
                            &join, iree_infinite_timeout());
    status = join.status;
    *out_stage = join.stage;
  }
  iree_notification_deinitialize(&join.notification);
  iree_slim_mutex_deinitialize(&join.mutex);
  return status;
}

const loom_cmd_program_t* loom_serve_jit_stage_program(
    const loom_serve_jit_stage_t* stage) {
  return stage->dispatch ? NULL : &stage->program;
}

loom_serve_dispatch_t* loom_serve_jit_stage_dispatch(
    const loom_serve_jit_stage_t* stage) {
  return stage->dispatch;
}

iree_status_t loom_serve_jit_stage_record(
    const loom_serve_jit_stage_t* stage,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    const loom_cmd_program_command_range_t* range,
    iree_hal_buffer_t* const* fixed_buffers,
    iree_hal_command_buffer_t** out_command) {
  const iree_host_size_t fixed_buffer_count =
      stage->program.requirements.fixed_buffer_count;
  iree_hal_buffer_binding_t* fixed_bindings = NULL;
  iree_status_t status = iree_ok_status();
  if (fixed_buffer_count != 0) {
    status = iree_allocator_malloc_array(
        stage->host_allocator, fixed_buffer_count, sizeof(*fixed_bindings),
        (void**)&fixed_bindings);
  }
  for (iree_host_size_t i = 0;
       i < fixed_buffer_count && iree_status_is_ok(status); ++i) {
    fixed_bindings[i] = (iree_hal_buffer_binding_t){
        .buffer = fixed_buffers[i],
        .offset = 0,
        .length = IREE_HAL_WHOLE_BUFFER,
    };
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_hal_command_buffer_create(
        queue_family, mode, &stage->program, range, fixed_buffer_count,
        fixed_bindings, stage->entry_count, stage->entries,
        stage->host_allocator, out_command);
  }
  iree_allocator_free(stage->host_allocator, fixed_bindings);
  return status;
}

static iree_status_t loom_serve_jit_compile_vm_source(
    loomc_target_environment_t* target_environment,
    const loom_serve_sources_t* sources, loomc_source_t* source,
    iree_host_size_t root_count, const iree_string_view_t* roots,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_image) {
  *out_image = iree_const_byte_span_empty();
  loom_serve_jit_compiler_t compiler = {.environment = target_environment};
  loomc_target_environment_retain(compiler.environment);
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  iree_status_t status = loom_serve_jit_compiler_initialize(
      NULL, LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG, report, &compiler,
      host_allocator);
  loomc_workspace_t* workspace = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_status_from_loomc(loomc_workspace_create(NULL, ca, &workspace));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        compiler.environment, loomc_make_cstring_view("vm:core"), ca,
        &compiler.profile));
  }
  loomc_target_specialization_t* specializations = NULL;
  loomc_string_view_t* root_symbols = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, root_count,
                                         sizeof(*specializations),
                                         (void**)&specializations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, root_count,
                                         sizeof(*root_symbols),
                                         (void**)&root_symbols);
  }
  loomc_link_index_t* index = NULL;
  loomc_linker_t* linker = NULL;
  loomc_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_source_index_create(compiler.context, sources, source,
                                       report, host_allocator, NULL, &index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_linker_create(compiler.context, NULL, ca, &linker));
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < root_count; ++i) {
      root_symbols[i] = loomc_string_view_from_iree(roots[i]);
      specializations[i] = (loomc_target_specialization_t){
          .function_symbol = root_symbols[i],
          .target_profile = compiler.profile,
      };
    }
    const loomc_target_specialization_options_t targets = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .specializations = specializations,
        .specialization_count = root_count,
    };
    const loomc_link_options_t options = {
        .next = &targets,
        .link_index = index,
        .mode = LOOMC_LINK_MODE_LINK,
        .root_symbols = root_symbols,
        .root_symbol_count = root_count,
        .config =
            {.json_object =
                 sources->card
                     ? loomc_string_view_from_iree(
                           loom_model_card_selection(sources->card)->config)
                     : loomc_string_view_empty()},
    };
    loomc_result_t* result = NULL;
    const loomc_status_t operation =
        loomc_link_module(linker, workspace, &options, &module, &result);
    status = loom_serve_compilation_result(operation, result, report);
  }
  if (iree_status_is_ok(status)) {
    loomc_byte_sequence_t* contents = NULL;
    status = loom_serve_jit_emit_module(
        &compiler, workspace, module, root_count, specializations,
        IREE_SV(LOOMC_ARTIFACT_FORMAT_VM), report, host_allocator, &contents);
    if (iree_status_is_ok(status)) {
      loomc_byte_span_t image = loomc_byte_span_empty();
      status = iree_status_from_loomc(
          loomc_byte_sequence_clone(contents, ca, &image));
      *out_image = iree_const_byte_span_from_loomc(image);
    }
    loomc_byte_sequence_release(contents);
  }
  loomc_module_release(module);
  loomc_linker_release(linker);
  loomc_link_index_release(index);
  iree_allocator_free(host_allocator, root_symbols);
  iree_allocator_free(host_allocator, specializations);
  loomc_workspace_release(workspace);
  loom_serve_jit_compiler_deinitialize(&compiler);
  return status;
}

iree_status_t loom_serve_jit_compile_vm(
    loomc_target_environment_t* target_environment,
    const loom_serve_sources_t* sources, iree_string_view_t source_path,
    iree_host_size_t root_count, const iree_string_view_t* roots,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_image) {
  *out_image = iree_const_byte_span_empty();
  if (!root_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM program requires at least one source root");
  }
  loomc_source_t* source = NULL;
  iree_status_t status = iree_ok_status();
  if (source_path.size) {
    status =
        loom_serve_sources_open(sources, source_path, host_allocator, &source);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_compile_vm_source(target_environment, sources,
                                              source, root_count, roots, report,
                                              host_allocator, out_image);
  }
  loomc_source_release(source);
  return status;
}

static void loom_serve_jit_vm_identity(const loomc_source_t* source,
                                       iree_host_size_t root_count,
                                       const iree_string_view_t* roots,
                                       loom_sha256_t* out_identity) {
  loom_sha256_state_t identity;
  loom_sha256_initialize(&identity);
  loom_serve_jit_identity_update_view(&identity,
                                      IREE_SV("loom.serve.compiled-vm.v1"));
  if (source) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    loom_serve_jit_identity_update_u64(&identity, loomc_source_format(source));
    loom_serve_jit_identity_update_view(
        &identity,
        iree_string_view_from_loomc(loomc_source_identifier(source)));
    loom_serve_jit_identity_update(&identity, contents.data,
                                   (iree_host_size_t)contents.data_length);
  } else {
    loom_serve_jit_identity_update_u64(&identity, LOOMC_SOURCE_FORMAT_UNKNOWN);
    loom_serve_jit_identity_update_view(&identity, iree_string_view_empty());
    loom_serve_jit_identity_update(&identity, NULL, 0);
  }
  loom_serve_jit_identity_update_u64(&identity, root_count);
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    loom_serve_jit_identity_update_view(&identity, roots[i]);
  }
  loom_sha256_finalize(&identity, out_identity);
}

iree_status_t loom_serve_jit_compile_vm_cached(
    loom_serve_jit_t* jit, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_image) {
  *out_image = iree_const_byte_span_empty();
  if (!root_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM program requires at least one source root");
  }
  loomc_source_t* source = NULL;
  iree_status_t status = iree_ok_status();
  if (source_path.size) {
    status =
        loom_serve_sources_open(sources, source_path, host_allocator, &source);
  }
  loom_serve_compilation_report_t report = {0};
  if (iree_status_is_ok(status) && jit->generation) {
    loom_sha256_t key;
    loom_serve_jit_vm_identity(source, root_count, roots, &key);
    bool found = false;
    status = loom_serve_jit_generation_lookup_vm(
        jit->generation, &key, &report, host_allocator, &found, out_image);
    if (iree_status_is_ok(status) && !found) {
      status = loom_serve_jit_compile_vm_source(
          jit->kernel.environment, sources, source, root_count, roots, &report,
          host_allocator, out_image);
      if (iree_status_is_ok(status)) {
        status = loom_serve_jit_generation_publish_vm(jit->generation, &key,
                                                      *out_image, &report);
        if (!iree_status_is_ok(status)) {
          iree_allocator_free(host_allocator, (void*)out_image->data);
          *out_image = iree_const_byte_span_empty();
        }
      }
    }
  } else if (iree_status_is_ok(status)) {
    status = loom_serve_jit_compile_vm_source(
        jit->kernel.environment, sources, source, root_count, roots, &report,
        host_allocator, out_image);
  }
  loom_serve_jit_report_merge(jit, &report);
  loomc_source_release(source);
  return status;
}
