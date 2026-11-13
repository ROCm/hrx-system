// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/host/host.h"

#include <stdlib.h>
#include <string.h>

#include "iree/async/util/proactor_thread.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/driver/task/driver.h"
#include "iree/hal/driver/task/executable/loaders/registration/init.h"
#include "iree/task/topology.h"
#include "loom/runtime/program/native_artifact.h"
#include "loomc/iree.h"
#include "loomc/iree/task_pool.h"
#include "loomc/target/vm.h"
#include "loomc/task_queue.h"

typedef struct loom_runtime_host_plan_entry_t {
  struct loom_runtime_host_plan_entry_t* next;
  loom_runtime_plan_provider_v1_t provider;
  iree_string_view_t id;
} loom_runtime_host_plan_entry_t;

struct loom_runtime_host_t {
  iree_slim_mutex_t plan_mutex;
  loom_runtime_host_plan_entry_t* plan_providers;
  bool plans_frozen;
  bool legacy_two_bank;
  // Application and dependent device owners of the shared host resources.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for the host and its shared services.
  iree_allocator_t host_allocator;
  // Reusable CPU scratch blocks shared by independent invocation arenas.
  iree_arena_block_pool_t block_pool;
  // Worker executor shared by compiler and application processes.
  iree_task_executor_t* executor;
  // Task driver retaining the same executor and configured native loaders.
  iree_hal_driver_t* task_driver;
  // Loom task adapter retaining the same executor, with no private workers.
  loomc_task_pool_t* task_pool;
  // Target-independent scratch indexed by mutually exclusive worker ordinal.
  loomc_workspace_t** compiler_workspaces;
  // Initialized workspace entries, including during creation failure cleanup.
  iree_host_size_t compiler_workspace_count;
  // Compiler scheduling domain shared by every independently owned catalog.
  loomc_task_queue_t* compiler_queue;
  // Preparation continuations progress independently of bulk native requests.
  loomc_task_queue_t* preparation_queue;
  // Immutable exact compiler target package shared by every catalog.
  loomc_target_environment_t* target_environment;
  // Weak exact-input and emitted-image indexes shared by matching targets.
  loom_runtime_native_cache_t* native_cache;
  // Owned copy of process-lifetime HAL target provider pointers.
  const struct loomc_iree_hal_target_provider_t** target_providers;
  // Number of entries in target_providers.
  iree_host_size_t target_provider_count;
  // Compiler reuse eligibility shared by workflows and accepted jobs.
  struct {
    // Protects short activity transitions and observer attachment.
    iree_slim_mutex_t mutex;
    // Workflow intent plus compiler jobs not yet fully retired.
    iree_host_size_t count;
    // Monotonic time when the final preparation ended.
    iree_time_t idle_since;
    // Preparation has occurred since the last claimed reclamation.
    bool dirty;
    // One caller is trimming outside the activity lock.
    bool trimming;
    // Stable caller-owned wake targets; no model or compiler ownership.
    loom_runtime_host_preparation_observer_t* observers;
  } preparation;
  // NUMA-local I/O services outliving every attached device.
  iree_async_proactor_pool_t* proactor_pool;
};

#if !IREE_PLATFORM_WASM
static void loom_runtime_host_proactor_failed(void* user_data,
                                              iree_status_t status) {
  // Fatal polling failure cannot retire accepted I/O storage. Operation-level
  // errors still flow through their normal completion callbacks.
  iree_status_abort(status);
}

static iree_status_t loom_runtime_host_proactor_thread_create(
    void* user_data, iree_async_proactor_t* proactor, uint32_t node_id,
    iree_allocator_t host_allocator, void** out_runner) {
  iree_async_proactor_thread_options_t options =
      iree_async_proactor_thread_options_default();
  options.debug_name = IREE_SV("loom-io");
  options.error_callback = (iree_async_proactor_thread_error_callback_t){
      loom_runtime_host_proactor_failed, NULL};
  if (node_id != UINT32_MAX) {
    iree_thread_affinity_set_group_any(node_id, &options.affinity);
  }
  iree_async_proactor_thread_t* thread = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_thread_create(
      proactor, options, host_allocator, &thread));
  *out_runner = thread;
  return iree_ok_status();
}
#endif  // !IREE_PLATFORM_WASM

// CPU queues participate in the same worker budget as JIT and model services.
// The driver retains the supplied executor, loaders and allocation policy.
static iree_status_t loom_runtime_host_create_task_driver(
    loom_runtime_host_t* host) {
  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_create_all_available_executable_loaders(
      IREE_ARRAYSIZE(loaders), host->host_allocator, &loader_count, loaders));
  iree_hal_allocator_t* device_allocator = NULL;
  iree_status_t status =
      iree_hal_allocator_create_heap(host->host_allocator, &device_allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_task_device_params_t params;
    iree_hal_task_device_params_initialize(&params);
    status = iree_hal_task_driver_create(
        IREE_SV("task"), &params, 1, &host->executor, loader_count, loaders,
        device_allocator, host->host_allocator, &host->task_driver);
  }
  iree_hal_allocator_release(device_allocator);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  return status;
}

static iree_status_t loom_runtime_host_create_compiler_environment(
    loom_runtime_target_environment_create_fn_t create,
    iree_allocator_t host_allocator,
    loomc_target_environment_t** out_environment) {
  const loomc_allocator_t allocator = loomc_allocator_from_iree(host_allocator);
  loomc_target_environment_t* environments[2] = {NULL, NULL};
  loomc_status_t status = create(allocator, &environments[0]);
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create_vm(allocator, &environments[1]);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create(
        IREE_ARRAYSIZE(environments), environments, allocator, out_environment);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(environments); ++i) {
    loomc_target_environment_release(environments[i]);
  }
  return iree_status_from_loomc(status);
}

static iree_status_t loom_runtime_host_create_compiler_workspaces(
    loom_runtime_host_t* host) {
  const iree_host_size_t worker_count =
      loomc_task_pool_worker_count(host->task_pool);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host->host_allocator, worker_count, sizeof(*host->compiler_workspaces),
      (void**)&host->compiler_workspaces));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < worker_count && iree_status_is_ok(status);
       ++i) {
    status = iree_status_from_loomc(loomc_workspace_create(
        NULL, loomc_allocator_from_iree(host->host_allocator),
        &host->compiler_workspaces[i]));
    if (iree_status_is_ok(status)) {
      ++host->compiler_workspace_count;
    }
  }
  return status;
}

iree_status_t loom_runtime_host_create(
    iree_host_size_t worker_count,
    const loom_runtime_compiler_options_t* compiler_options,
    iree_allocator_t host_allocator, loom_runtime_host_t** out_host) {
  *out_host = NULL;
  if (!worker_count || worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host worker count is outside executor capacity");
  }
  loom_runtime_host_t* host = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*host), (void**)&host));
  host->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&host->plan_mutex);
  const char* legacy = getenv("LOOM_SERVE_TWO_BANK");
  host->legacy_two_bank = legacy && !strcmp(legacy, "1");
  iree_atomic_ref_count_init(&host->ref_count);
  iree_slim_mutex_initialize(&host->preparation.mutex);
  iree_arena_block_pool_initialize(64 * 1024, host_allocator,
                                   &host->block_pool);
  iree_status_t status = iree_ok_status();
  if (compiler_options) {
    host->target_provider_count = compiler_options->target_provider_count;
    status = iree_allocator_malloc_array(
        host_allocator, host->target_provider_count,
        sizeof(*host->target_providers), (void**)&host->target_providers);
    if (iree_status_is_ok(status)) {
      memcpy(host->target_providers, compiler_options->target_providers,
             host->target_provider_count * sizeof(*host->target_providers));
      status = loom_runtime_host_create_compiler_environment(
          compiler_options->target_environment_create, host_allocator,
          &host->target_environment);
    }
    if (iree_status_is_ok(status)) {
      status = loom_runtime_native_cache_allocate(host_allocator,
                                                  &host->native_cache);
    }
  }
  iree_task_topology_t topology;
  if (iree_status_is_ok(status)) {
    status = iree_task_topology_initialize_from_physical_cores(
        IREE_TASK_TOPOLOGY_NODE_ID_ANY,
        IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, worker_count, &topology);
  }
  if (iree_status_is_ok(status)) {
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    options.worker_stack_size = 2 * 1024 * 1024;
    status = iree_task_executor_create(options, &topology, host_allocator,
                                       &host->executor);
    iree_task_topology_deinitialize(&topology);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_host_create_task_driver(host);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_task_pool_allocate_from_iree_executor(
        host->executor, loomc_allocator_from_iree(host_allocator),
        &host->task_pool));
  }
  if (iree_status_is_ok(status) && compiler_options) {
    status = loom_runtime_host_create_compiler_workspaces(host);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_task_queue_allocate(
        host->task_pool, loomc_allocator_from_iree(host_allocator),
        &host->compiler_queue));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_task_queue_allocate(
        host->task_pool, loomc_allocator_from_iree(host_allocator),
        &host->preparation_queue));
  }
  if (iree_status_is_ok(status)) {
    iree_async_proactor_pool_options_t options =
        iree_async_proactor_pool_options_default();
#if !IREE_PLATFORM_WASM
    options.runner.create = loom_runtime_host_proactor_thread_create;
#endif  // !IREE_PLATFORM_WASM
    // Pool slots retain actual node identities, which may be sparse. Device
    // placement resolves these identities when acquiring its I/O service.
    iree_bitmap_t online_nodes = iree_numa_online_nodes();
    uint32_t node_ids[IREE_NUMA_MAX_NODES];
    iree_host_size_t node_count = 0;
    for (uint32_t node_id = 0; node_id < online_nodes.bit_count; ++node_id) {
      if (iree_bitmap_test(online_nodes, node_id)) {
        node_ids[node_count++] = node_id;
      }
    }
    status = iree_async_proactor_pool_create(
        node_count, node_ids, options, host_allocator, &host->proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_host_register_plan_provider(
        host, IREE_SV("two_bank_v1"), loom_runtime_two_bank_plan_provider());
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_host_register_plan_provider(
        host, IREE_SV("sequential_v1"),
        loom_runtime_sequential_plan_provider());
  }
  if (iree_status_is_ok(status)) {
    *out_host = host;
  } else {
    loom_runtime_host_release(host);
  }
  return status;
}

void loom_runtime_host_retain(loom_runtime_host_t* host) {
  if (IREE_LIKELY(host)) {
    iree_atomic_ref_count_inc(&host->ref_count);
  }
}

void loom_runtime_host_release(loom_runtime_host_t* host) {
  if (IREE_LIKELY(host) && iree_atomic_ref_count_dec(&host->ref_count) == 1) {
    // Preparation scopes and observers borrow the host through final
    // retirement.
    IREE_ASSERT(!host->preparation.count && !host->preparation.observers);
    iree_async_proactor_pool_release(host->proactor_pool);
    loomc_task_queue_free(host->compiler_queue);
    loomc_task_queue_free(host->preparation_queue);
    loomc_task_pool_free(host->task_pool);
    loom_runtime_native_cache_free(host->native_cache);
    iree_hal_driver_release(host->task_driver);
    iree_task_executor_release(host->executor);
    for (iree_host_size_t i = 0; i < host->compiler_workspace_count; ++i) {
      loomc_workspace_release(host->compiler_workspaces[i]);
    }
    iree_allocator_free(host->host_allocator, host->compiler_workspaces);
    loomc_target_environment_release(host->target_environment);
    iree_allocator_free(host->host_allocator, host->target_providers);
    iree_arena_block_pool_deinitialize(&host->block_pool);
    loom_runtime_host_plan_entry_t* entry = host->plan_providers;
    while (entry) {
      loom_runtime_host_plan_entry_t* next = entry->next;
      if (entry->provider.release) {
        entry->provider.release(entry->provider.self);
      }
      iree_allocator_free(host->host_allocator, entry);
      entry = next;
    }
    iree_slim_mutex_deinitialize(&host->plan_mutex);
    iree_slim_mutex_deinitialize(&host->preparation.mutex);
    iree_allocator_free(host->host_allocator, host);
  }
}

iree_task_executor_t* loom_runtime_host_executor(
    const loom_runtime_host_t* host) {
  return host->executor;
}

loomc_task_pool_t* loom_runtime_host_task_pool(
    const loom_runtime_host_t* host) {
  return host->task_pool;
}

loomc_workspace_t* loom_runtime_host_compiler_workspace(
    const loom_runtime_host_t* host, iree_host_size_t worker_ordinal) {
  return host->compiler_workspaces[worker_ordinal];
}

iree_status_t loom_runtime_host_trim_compiler(loom_runtime_host_t* host) {
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < host->compiler_workspace_count; ++i) {
    loomc_workspace_trim(host->compiler_workspaces[i]);
  }
  iree_status_t status = iree_allocator_trim(host->host_allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_runtime_host_preparation_notify_locked(
    loom_runtime_host_t* host) {
  for (loom_runtime_host_preparation_observer_t* observer =
           host->preparation.observers;
       observer; observer = observer->next) {
    observer->callback.wake(observer->callback.user_data);
  }
}

void loom_runtime_host_preparation_begin(loom_runtime_host_t* host) {
  iree_slim_mutex_lock(&host->preparation.mutex);
  ++host->preparation.count;
  host->preparation.dirty = true;
  iree_slim_mutex_unlock(&host->preparation.mutex);
}

void loom_runtime_host_preparation_end(loom_runtime_host_t* host) {
  iree_slim_mutex_lock(&host->preparation.mutex);
  if (!--host->preparation.count) {
    host->preparation.idle_since = iree_time_now();
    loom_runtime_host_preparation_notify_locked(host);
  }
  iree_slim_mutex_unlock(&host->preparation.mutex);
}

void loom_runtime_host_observe_preparation(
    loom_runtime_host_t* host,
    loom_runtime_host_preparation_observer_t* observer) {
  iree_slim_mutex_lock(&host->preparation.mutex);
  observer->next = host->preparation.observers;
  host->preparation.observers = observer;
  iree_slim_mutex_unlock(&host->preparation.mutex);
}

void loom_runtime_host_unobserve_preparation(
    loom_runtime_host_t* host,
    loom_runtime_host_preparation_observer_t* observer) {
  iree_slim_mutex_lock(&host->preparation.mutex);
  loom_runtime_host_preparation_observer_t** link =
      &host->preparation.observers;
  while (*link != observer) {
    link = &(*link)->next;
  }
  *link = observer->next;
  observer->next = NULL;
  iree_slim_mutex_unlock(&host->preparation.mutex);
}

static iree_time_t loom_runtime_host_compiler_idle_deadline_locked(
    loom_runtime_host_t* host, iree_duration_t timeout) {
  if (host->preparation.count || !host->preparation.dirty ||
      host->preparation.trimming || timeout == IREE_DURATION_INFINITE) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  return host->preparation.idle_since +
         iree_min(timeout,
                  IREE_TIME_INFINITE_FUTURE - 1 - host->preparation.idle_since);
}

iree_time_t loom_runtime_host_compiler_idle_deadline(loom_runtime_host_t* host,
                                                     iree_duration_t timeout) {
  iree_slim_mutex_lock(&host->preparation.mutex);
  const iree_time_t deadline =
      loom_runtime_host_compiler_idle_deadline_locked(host, timeout);
  iree_slim_mutex_unlock(&host->preparation.mutex);
  return deadline;
}

iree_status_t loom_runtime_host_trim_compiler_idle(loom_runtime_host_t* host,
                                                   iree_duration_t timeout,
                                                   iree_time_t now) {
  iree_slim_mutex_lock(&host->preparation.mutex);
  const iree_time_t deadline =
      loom_runtime_host_compiler_idle_deadline_locked(host, timeout);
  const bool eligible =
      deadline != IREE_TIME_INFINITE_FUTURE && deadline <= now;
  if (eligible) {
    host->preparation.dirty = false;
    host->preparation.trimming = true;
  }
  iree_slim_mutex_unlock(&host->preparation.mutex);
  if (!eligible) {
    return iree_ok_status();
  }
  iree_status_t status = loom_runtime_host_trim_compiler(host);
  iree_slim_mutex_lock(&host->preparation.mutex);
  host->preparation.trimming = false;
  if (!host->preparation.count && host->preparation.dirty) {
    loom_runtime_host_preparation_notify_locked(host);
  }
  iree_slim_mutex_unlock(&host->preparation.mutex);
  return status;
}

iree_arena_block_pool_t* loom_runtime_host_block_pool(
    loom_runtime_host_t* host) {
  return &host->block_pool;
}

iree_hal_driver_t* loom_runtime_host_task_driver(
    const loom_runtime_host_t* host) {
  return host->task_driver;
}

loomc_task_sink_t loom_runtime_host_compiler_sink(
    const loom_runtime_host_t* host) {
  return loomc_task_queue_sink(host->compiler_queue);
}

loomc_task_sink_t loom_runtime_host_preparation_sink(
    const loom_runtime_host_t* host) {
  return loomc_task_queue_sink(host->preparation_queue);
}

loomc_target_environment_t* loom_runtime_host_target_environment(
    const loom_runtime_host_t* host) {
  return host->target_environment;
}

loom_runtime_native_cache_t* loom_runtime_host_native_cache(
    const loom_runtime_host_t* host) {
  return host->native_cache;
}

const struct loomc_iree_hal_target_provider_t* const*
loom_runtime_host_target_providers(const loom_runtime_host_t* host,
                                   iree_host_size_t* out_provider_count) {
  *out_provider_count = host->target_provider_count;
  return host->target_providers;
}

iree_async_proactor_pool_t* loom_runtime_host_proactor_pool(
    const loom_runtime_host_t* host) {
  return host->proactor_pool;
}

iree_status_t loom_runtime_host_register_plan_provider(
    loom_runtime_host_t* host, iree_string_view_t id,
    const loom_runtime_plan_provider_v1_t* provider) {
  if (!id.size || !provider || provider->struct_size != sizeof(*provider) ||
      provider->abi_version != 1 || !provider->prepare ||
      (!!provider->retain != !!provider->release) ||
      (provider->self && !provider->retain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid plan provider ABI");
  }
  loom_runtime_host_plan_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host->host_allocator, sizeof(*entry), id.size, 1, (void**)&entry));
  entry->id = iree_make_string_view((char*)(entry + 1), id.size);
  memcpy((void*)entry->id.data, id.data, id.size);
  entry->provider = *provider;
  if (provider->retain) {
    provider->retain(provider->self);
  }
  iree_slim_mutex_lock(&host->plan_mutex);
  iree_status_t status = iree_ok_status();
  if (host->plans_frozen) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "plan registry is frozen");
  }
  for (loom_runtime_host_plan_entry_t* prior = host->plan_providers;
       prior && iree_status_is_ok(status); prior = prior->next) {
    if (iree_string_view_equal(prior->id, id)) {
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "plan provider already registered");
    }
  }
  if (iree_status_is_ok(status)) {
    entry->next = host->plan_providers;
    host->plan_providers = entry;
  }
  iree_slim_mutex_unlock(&host->plan_mutex);
  if (!iree_status_is_ok(status)) {
    if (provider->release) {
      provider->release(provider->self);
    }
    iree_allocator_free(host->host_allocator, entry);
  }
  return status;
}

iree_status_t loom_runtime_host_lookup_plan_provider(
    loom_runtime_host_t* host, iree_string_view_t id,
    loom_runtime_plan_provider_v1_t* out_provider) {
  *out_provider = (loom_runtime_plan_provider_v1_t){0};
  iree_slim_mutex_lock(&host->plan_mutex);
  host->plans_frozen = true;
  for (loom_runtime_host_plan_entry_t* entry = host->plan_providers; entry;
       entry = entry->next) {
    if (iree_string_view_equal(entry->id, id)) {
      *out_provider = entry->provider;
      break;
    }
  }
  iree_slim_mutex_unlock(&host->plan_mutex);
  if (!out_provider->prepare) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "execution plan provider '%.*s' is not registered",
                            (int)id.size, id.data);
  }
  if (out_provider->retain) {
    out_provider->retain(out_provider->self);
  }
  return iree_ok_status();
}

bool loom_runtime_host_legacy_two_bank(const loom_runtime_host_t* host) {
  return host->legacy_two_bank;
}
