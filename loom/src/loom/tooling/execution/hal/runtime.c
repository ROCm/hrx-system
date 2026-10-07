// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/runtime.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/tooling/device_util.h"

// Reusable backing granularity for testbench fixture staging. Larger requests
// use the TLSF pool's dedicated-backing path.
static const iree_device_size_t LOOM_RUN_HAL_STAGING_SLAB_LENGTH =
    (iree_device_size_t)64 * 1024 * 1024;

// Preferred suballocation alignment when the selected native route supports
// it. Device addressing requirements remain the hard lower bound.
static const iree_device_size_t LOOM_RUN_HAL_STAGING_PREFERRED_ALIGNMENT = 256;

static iree_status_t loom_run_hal_runtime_select_queue(
    iree_hal_device_t* device,
    iree_hal_queue_family_role_flags_t required_roles, const char* role_name,
    iree_hal_queue_t** out_queue) {
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queue_spec->family_count; ++i) {
    const iree_hal_queue_family_spec_t* family_spec = &queue_spec->families[i];
    if (family_spec->provisioned_queue_count == 0 ||
        !iree_all_bits_set(family_spec->role_flags, required_roles)) {
      continue;
    }

    iree_hal_queue_t* queue = iree_hal_device_queue(
        device, (iree_hal_queue_family_ordinal_t)i, /*queue_ordinal=*/0);
    if (IREE_UNLIKELY(queue == NULL)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HAL device queue specification advertises provisioned %s queue "
          "family %" PRIhsz " but the queue is unavailable",
          role_name, i);
    }
    *out_queue = queue;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "HAL device has no provisioned %s queue family",
                          role_name);
}

static iree_status_t loom_run_hal_runtime_create_staging_pool(
    loom_run_hal_runtime_t* runtime, iree_allocator_t host_allocator) {
  const iree_hal_queue_family_t* dispatch_family =
      iree_hal_queue_family(runtime->dispatch_queue);
  const iree_hal_queue_family_t* transfer_family =
      iree_hal_queue_family(runtime->transfer_queue);
  iree_hal_pool_family_access_t family_accesses[2] = {
      {
          .family = dispatch_family,
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      },
  };
  iree_host_size_t family_count = 1;
  if (transfer_family == dispatch_family) {
    family_accesses[0].usage |= IREE_HAL_BUFFER_USAGE_TRANSFER;
  } else {
    family_accesses[family_count++] = (iree_hal_pool_family_access_t){
        .family = transfer_family,
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
  }
  const iree_hal_pool_scope_t scope = {
      .family_count = family_count,
      .families = family_accesses,
  };
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  source_options.trace_name = IREE_SV("loom-staging-source");

  iree_hal_pool_t* source_pool = NULL;
  iree_status_t status =
      iree_hal_slab_pool_create(runtime->device_group, scope, &source_options,
                                host_allocator, &source_pool);

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(runtime->device);
  const iree_hal_device_dispatch_spec_t* dispatch_spec =
      iree_hal_device_spec_dispatch(device_spec);
  const iree_hal_device_sanitizer_spec_t* sanitizer_spec =
      iree_hal_device_spec_sanitizer(device_spec);
  iree_device_size_t pool_alignment = iree_max(
      (iree_device_size_t)IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT,
      dispatch_spec->addressing.minimum_buffer_device_address_alignment);
  if (iree_status_is_ok(status)) {
    iree_hal_pool_capabilities_t source_capabilities;
    iree_hal_pool_query_capabilities(source_pool, &source_capabilities);
    if (!source_capabilities.max_allocation_alignment ||
        source_capabilities.max_allocation_alignment >=
            LOOM_RUN_HAL_STAGING_PREFERRED_ALIGNMENT) {
      pool_alignment =
          iree_max(pool_alignment, LOOM_RUN_HAL_STAGING_PREFERRED_ALIGNMENT);
    } else if (source_capabilities.max_allocation_alignment >= pool_alignment) {
      pool_alignment = source_capabilities.max_allocation_alignment;
    }
  }
  iree_hal_tlsf_pool_options_t pool_options = {
      .tlsf_options =
          {
              .range_length = LOOM_RUN_HAL_STAGING_SLAB_LENGTH,
              .alignment = pool_alignment,
          },
      .trace_name = IREE_SV("loom-staging"),
  };
  if (iree_any_bit_set(sanitizer_spec->flags,
                       IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN)) {
    pool_options.asan = sanitizer_spec->asan.pool_options;
  }

  iree_hal_pool_reservation_request_t backing_request = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_tlsf_pool_query_backing_request(
        source_pool, &pool_options, &backing_request);
  }
  iree_hal_pool_t* backing_cache = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_slab_cache_options_t cache_options;
    iree_hal_slab_cache_options_initialize(&cache_options);
    cache_options.slab = backing_request;
    cache_options.max_count = 1;
    cache_options.trace_name = IREE_SV("loom-staging-cache");
    status = iree_hal_slab_cache_create(source_pool, &cache_options,
                                        host_allocator, &backing_cache);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_tlsf_pool_create(backing_cache, &pool_options,
                                       host_allocator, &runtime->staging_pool);
  }
  iree_hal_pool_release(backing_cache);
  iree_hal_pool_release(source_pool);
  return status;
}

void loom_run_hal_runtime_options_initialize(
    iree_string_view_t hal_driver_name,
    loom_run_hal_runtime_options_t* out_options) {
  *out_options = (loom_run_hal_runtime_options_t){
      .hal_driver_name = hal_driver_name,
      .event_sink = iree_hal_device_event_sink_stderr(),
  };
}

iree_status_t loom_run_hal_runtime_initialize(
    const loom_run_hal_runtime_options_t* options, iree_allocator_t allocator,
    loom_run_hal_runtime_t* out_runtime) {
  *out_runtime = (loom_run_hal_runtime_t){0};

  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), allocator, &proactor_pool);
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator,
        &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    create_params.event_sink = options->event_sink;
    create_params.runtime_features = options->runtime_features;
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(), options->hal_driver_name,
        &create_params, allocator, &out_runtime->device);
  }
  iree_async_proactor_pool_release(proactor_pool);
  if (iree_status_is_ok(status)) {
    // Group assignment establishes the device frontier and materializes its
    // configured hardware queues, so queue lookup must follow finalization.
    status = iree_hal_device_group_create_from_device(
        out_runtime->device, frontier_tracker, allocator,
        &out_runtime->device_group);
  }
  iree_async_frontier_tracker_release(frontier_tracker);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_runtime_select_queue(
        out_runtime->device, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
        "dispatch", &out_runtime->dispatch_queue);
  }
  if (iree_status_is_ok(status) && !out_runtime->dispatch_queue) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL device has no provisioned dispatch-capable queue");
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_runtime_select_queue(
        out_runtime->device, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
        "transfer", &out_runtime->transfer_queue);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_runtime_create_staging_pool(out_runtime, allocator);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_runtime_deinitialize(out_runtime);
  }
  return status;
}

iree_hal_physical_device_affinity_t
loom_run_hal_runtime_dispatch_physical_device_affinity(
    const loom_run_hal_runtime_t* runtime) {
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(runtime->device));
  const iree_hal_queue_family_ordinal_t family_ordinal =
      iree_hal_queue_family_ordinal(
          iree_hal_queue_family(runtime->dispatch_queue));
  return queue_spec->families[family_ordinal].physical_device_affinity;
}

void loom_run_hal_runtime_deinitialize(loom_run_hal_runtime_t* runtime) {
  if (runtime == NULL) {
    return;
  }
  runtime->dispatch_queue = NULL;
  runtime->transfer_queue = NULL;
  iree_hal_pool_release(runtime->staging_pool);
  iree_hal_device_group_release(runtime->device_group);
  iree_hal_device_release(runtime->device);
  *runtime = (loom_run_hal_runtime_t){0};
}

iree_status_t loom_run_hal_semaphore_wait(iree_hal_semaphore_t* semaphore,
                                          uint64_t value,
                                          iree_timeout_t timeout,
                                          iree_async_wait_flags_t flags) {
  iree_status_t status =
      iree_hal_semaphore_wait(semaphore, value, timeout, flags);
  if (!iree_status_is_ok(status)) {
    // HAL waits may return only a code; the semaphore retains the diagnostic.
    uint64_t semaphore_value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(semaphore, &semaphore_value);
    status = iree_status_join(query_status, status);
  }
  return status;
}
