// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/net/demo/worker.h"

#include <stdio.h>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/net/channel_provider.h"
#include "iree/net/carrier/tcp/factory.h"

static iree_status_t demo_worker_initialize_memory(demo_worker_t* worker) {
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(worker->queue),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      .requirements = IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST};
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
      .host = {.access = IREE_HAL_MEMORY_ACCESS_ALL,
               .modes = IREE_HAL_MAPPING_MODE_SCOPED |
                        IREE_HAL_MAPPING_MODE_PERSISTENT}};
  iree_hal_slab_pool_options_t storage_options;
  iree_hal_slab_pool_options_initialize(&storage_options);
  IREE_RETURN_IF_ERROR(
      iree_hal_slab_pool_create(worker->group, scope, &storage_options,
                                iree_allocator_system(), &worker->storage));
  iree_hal_tlsf_pool_options_t workspace_options = {0};
  workspace_options.tlsf_options.range_length = 64 * 1024;
  workspace_options.tlsf_options.alignment = 64;
  workspace_options.tlsf_options.frontier_capacity = 2;
  workspace_options.asan =
      iree_hal_device_spec_sanitizer(iree_hal_device_spec(worker->device))
          ->asan.pool_options;
  iree_hal_slab_cache_options_t cache_options = {.max_count = 1};
  IREE_RETURN_IF_ERROR(iree_hal_tlsf_pool_query_backing_request(
      worker->storage, &workspace_options, &cache_options.slab));
  IREE_RETURN_IF_ERROR(
      iree_hal_slab_cache_create(worker->storage, &cache_options,
                                 iree_allocator_system(), &worker->cache));
  return iree_hal_tlsf_pool_create(worker->cache, &workspace_options,
                                   iree_allocator_system(), &worker->workspace);
}

static iree_status_t demo_worker_initialize_channel(
    const iree_hal_net_channel_options_t* options, demo_worker_t* worker) {
  iree_net_tcp_factory_options_t factory_options =
      iree_net_tcp_factory_options_default();
  factory_options.connection_options.carrier_options.max_send_operations = 1;
  factory_options.connection_options.receive_mode =
      IREE_NET_TCP_RECEIVE_MODE_PLACEMENT;
  iree_net_transport_factory_t* factory = NULL;
  IREE_RETURN_IF_ERROR(iree_net_tcp_factory_create(
      &factory_options, iree_allocator_system(), &factory));
  iree_hal_channel_provider_t* provider = NULL;
  iree_status_t status = iree_hal_net_channel_provider_create(
      factory, worker->proactor, options, iree_allocator_system(), &provider);
  iree_net_transport_factory_release(factory);
  if (iree_status_is_ok(status)) {
    iree_hal_device_replace_channel_provider(worker->device, provider);
    const iree_hal_channel_params_t params = {
        .rank = options->rank,
        .count = options->count,
        .group = IREE_SV("cpu-projection")};
    status = iree_hal_channel_create(worker->device,
                                     IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, params,
                                     &worker->channel);
  }
  iree_hal_channel_provider_release(provider);
  if (iree_status_is_ok(status) && options->rank + 1 < options->count) {
    char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
    iree_string_view_t bound_address;
    status = iree_hal_net_channel_query_address(
        worker->channel, sizeof(storage), storage, &bound_address);
    if (iree_status_is_ok(status)) {
      fprintf(stderr, "Listening on %.*s\n", (int)bound_address.size,
              bound_address.data);
      fflush(stderr);
    }
  }
  return status;
}

iree_status_t demo_worker_initialize(
    const iree_hal_net_channel_options_t* options, demo_worker_t* worker) {
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(1, &topology);
  iree_task_executor_options_t executor_options;
  iree_task_executor_options_initialize(&executor_options);
  iree_status_t status = iree_task_executor_create(
      executor_options, &topology, iree_allocator_system(), &worker->executor);
  iree_task_topology_deinitialize(&topology);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(iree_allocator_system(),
                                                      &worker->allocator));
  iree_async_proactor_pool_options_t service_options =
      iree_async_proactor_pool_options_default();
  service_options.runner = (iree_async_proactor_pool_runner_factory_t){0};
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
      1, NULL, service_options, iree_allocator_system(), &worker->services));
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_pool_get(worker->services, 0, &worker->proactor));
  iree_hal_task_device_params_t device_options;
  iree_hal_task_device_params_initialize(&device_options);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = worker->services;
  IREE_RETURN_IF_ERROR(iree_hal_task_device_create(
      IREE_SV("cpu-collective"), &device_options, 1, &worker->executor, 0, NULL,
      worker->allocator, &create_params, iree_allocator_system(),
      &worker->device));
  worker->queue = iree_hal_device_queue(worker->device, 0, 0);
  iree_async_frontier_tracker_t* tracker = NULL;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  status = iree_hal_device_group_create_from_device(
      worker->device, tracker, iree_allocator_system(), &worker->group);
  iree_async_frontier_tracker_release(tracker);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(demo_worker_initialize_memory(worker));
  return demo_worker_initialize_channel(options, worker);
}

void demo_worker_poll(demo_worker_t* worker) {
  IREE_CHECK_OK(iree_async_proactor_poll(worker->proactor,
                                         iree_infinite_timeout(), NULL));
}

static void demo_worker_on_drained(void* user_data, iree_status_t status) {
  demo_worker_t* worker = user_data;
  worker->drain_status = status;
  worker->drained = true;
}

iree_status_t demo_worker_close(demo_worker_t* worker, iree_status_t status) {
  if (worker->channel) {
    IREE_CHECK_OK(
        iree_hal_channel_deactivate(worker->channel, iree_status_clone(status),
                                    (iree_hal_channel_deactivated_callback_t){
                                        demo_worker_on_drained, worker}));
    while (!worker->drained) {
      demo_worker_poll(worker);
    }
    status = iree_status_join(status, worker->drain_status);
    worker->drain_status = iree_ok_status();
  }
  return status;
}

void demo_worker_deinitialize(demo_worker_t* worker) {
  iree_hal_channel_release(worker->channel);
  iree_hal_pool_release(worker->workspace);
  iree_hal_pool_release(worker->cache);
  iree_hal_pool_release(worker->storage);
  iree_hal_device_group_release(worker->group);
  iree_hal_device_release(worker->device);
  if (worker->proactor) {
    iree_async_proactor_end_polling(worker->proactor);
  }
  iree_async_proactor_pool_release(worker->services);
  iree_hal_allocator_release(worker->allocator);
  iree_task_executor_release(worker->executor);
}

iree_status_t demo_completion_initialize(demo_worker_t* worker,
                                         demo_completion_t* completion) {
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      worker->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion->semaphore));
  completion->async_semaphore = (iree_async_semaphore_t*)completion->semaphore;
  return iree_ok_status();
}

static void demo_completion_on_retired(void* user_data,
                                       iree_async_operation_t* operation,
                                       iree_status_t status,
                                       iree_async_completion_flags_t flags) {
  demo_completion_t* completion = user_data;
  completion->status = status;
  completion->pending = false;
}

iree_status_t demo_completion_arm(demo_worker_t* worker,
                                  demo_completion_t* completion) {
  ++completion->value;
  memset(&completion->wait, 0, sizeof(completion->wait));
  completion->wait.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  completion->wait.base.completion_fn = demo_completion_on_retired;
  completion->wait.base.user_data = completion;
  completion->wait.semaphores = &completion->async_semaphore;
  completion->wait.values = &completion->value;
  completion->wait.count = 1;
  completion->wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_submit_one(worker->proactor, &completion->wait.base));
  completion->pending = true;
  return iree_ok_status();
}

iree_status_t demo_completion_admit(demo_completion_t* completion,
                                    iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(completion->semaphore, iree_status_clone(status));
  }
  return status;
}

iree_status_t demo_completion_join(demo_worker_t* worker,
                                   demo_completion_t* completion) {
  while (completion->pending) {
    demo_worker_poll(worker);
  }
  iree_status_t status = completion->status;
  completion->status = iree_ok_status();
  return status;
}
