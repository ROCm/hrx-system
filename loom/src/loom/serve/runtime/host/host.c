// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/host/host.h"

#include <string.h>

#include "iree/async/util/proactor_thread.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/task/driver.h"
#include "iree/hal/drivers/task/executable/loaders/registration/init.h"
#include "iree/task/topology.h"
#include "loomc/iree.h"
#include "loomc/iree/task_pool.h"
#include "loomc/target/vm.h"
#include "loomc/task_queue.h"

struct loom_serve_host_t {
  // Application and dependent device owners of the shared host resources.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for the host and its shared services.
  iree_allocator_t host_allocator;
  // Worker executor shared by compiler and application processes.
  iree_task_executor_t* executor;
  // Task driver retaining the same executor and configured native loaders.
  iree_hal_driver_t* task_driver;
  // Loom task adapter retaining the same executor, with no private workers.
  loomc_task_pool_t* task_pool;
  // Compiler scheduling domain shared by every independently owned catalog.
  loomc_task_queue_t* compiler_queue;
  // Immutable exact compiler target package shared by every catalog.
  loomc_target_environment_t* target_environment;
  // Owned copy of process-lifetime HAL target provider pointers.
  const struct loomc_iree_hal_target_provider_t** target_providers;
  // Number of entries in target_providers.
  iree_host_size_t target_provider_count;
  // NUMA-local I/O services outliving every attached device.
  iree_async_proactor_pool_t* proactor_pool;
};

#if !IREE_PLATFORM_WASM
static void loom_serve_host_proactor_failed(void* user_data,
                                            iree_status_t status) {
  // Fatal polling failure cannot retire accepted I/O storage. Operation-level
  // errors still flow through their normal completion callbacks.
  iree_status_abort(status);
}

static iree_status_t loom_serve_host_proactor_thread_create(
    void* user_data, iree_async_proactor_t* proactor, uint32_t node_id,
    iree_allocator_t host_allocator, void** out_runner) {
  iree_async_proactor_thread_options_t options =
      iree_async_proactor_thread_options_default();
  options.debug_name = IREE_SV("loom-io");
  options.error_callback = (iree_async_proactor_thread_error_callback_t){
      loom_serve_host_proactor_failed, NULL};
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
static iree_status_t loom_serve_host_create_task_driver(
    loom_serve_host_t* host) {
  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_create_all_available_executable_loaders(
      IREE_ARRAYSIZE(loaders), &loader_count, loaders, host->host_allocator));
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

static iree_status_t loom_serve_host_create_compiler_environment(
    loom_serve_target_environment_create_fn_t create,
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

iree_status_t loom_serve_host_create(
    iree_host_size_t worker_count,
    const loom_serve_compiler_options_t* compiler_options,
    iree_allocator_t host_allocator, loom_serve_host_t** out_host) {
  *out_host = NULL;
  if (!worker_count || worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host worker count is outside executor capacity");
  }
  loom_serve_host_t* host = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*host), (void**)&host));
  host->host_allocator = host_allocator;
  iree_atomic_ref_count_init(&host->ref_count);
  iree_status_t status = iree_ok_status();
  if (compiler_options) {
    host->target_provider_count = compiler_options->target_provider_count;
    status = iree_allocator_malloc_array(
        host_allocator, host->target_provider_count,
        sizeof(*host->target_providers), (void**)&host->target_providers);
    if (iree_status_is_ok(status)) {
      memcpy(host->target_providers, compiler_options->target_providers,
             host->target_provider_count * sizeof(*host->target_providers));
      status = loom_serve_host_create_compiler_environment(
          compiler_options->target_environment_create, host_allocator,
          &host->target_environment);
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
    status = loom_serve_host_create_task_driver(host);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_task_pool_allocate_from_iree_executor(
        host->executor, loomc_allocator_from_iree(host_allocator),
        &host->task_pool));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_task_queue_allocate(
        host->task_pool, loomc_allocator_from_iree(host_allocator),
        &host->compiler_queue));
  }
  if (iree_status_is_ok(status)) {
    iree_async_proactor_pool_options_t options =
        iree_async_proactor_pool_options_default();
#if !IREE_PLATFORM_WASM
    options.runner.create = loom_serve_host_proactor_thread_create;
#endif  // !IREE_PLATFORM_WASM
    status =
        iree_async_proactor_pool_create(iree_numa_node_count(), NULL, options,
                                        host_allocator, &host->proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    *out_host = host;
  } else {
    loom_serve_host_release(host);
  }
  return status;
}

void loom_serve_host_retain(loom_serve_host_t* host) {
  if (IREE_LIKELY(host)) {
    iree_atomic_ref_count_inc(&host->ref_count);
  }
}

void loom_serve_host_release(loom_serve_host_t* host) {
  if (IREE_LIKELY(host) && iree_atomic_ref_count_dec(&host->ref_count) == 1) {
    iree_async_proactor_pool_release(host->proactor_pool);
    loomc_task_queue_free(host->compiler_queue);
    loomc_task_pool_free(host->task_pool);
    iree_hal_driver_release(host->task_driver);
    iree_task_executor_release(host->executor);
    loomc_target_environment_release(host->target_environment);
    iree_allocator_free(host->host_allocator, host->target_providers);
    iree_allocator_free(host->host_allocator, host);
  }
}

iree_task_executor_t* loom_serve_host_executor(const loom_serve_host_t* host) {
  return host->executor;
}

loomc_task_pool_t* loom_serve_host_task_pool(const loom_serve_host_t* host) {
  return host->task_pool;
}

iree_hal_driver_t* loom_serve_host_task_driver(const loom_serve_host_t* host) {
  return host->task_driver;
}

loomc_task_sink_t loom_serve_host_compiler_sink(const loom_serve_host_t* host) {
  return loomc_task_queue_sink(host->compiler_queue);
}

loomc_target_environment_t* loom_serve_host_target_environment(
    const loom_serve_host_t* host) {
  return host->target_environment;
}

const struct loomc_iree_hal_target_provider_t* const*
loom_serve_host_target_providers(const loom_serve_host_t* host,
                                 iree_host_size_t* out_provider_count) {
  *out_provider_count = host->target_provider_count;
  return host->target_providers;
}

iree_async_proactor_pool_t* loom_serve_host_proactor_pool(
    const loom_serve_host_t* host) {
  return host->proactor_pool;
}
