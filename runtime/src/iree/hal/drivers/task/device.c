// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/task/atomic.h"
#include "iree/hal/drivers/task/command/block_command_buffer.h"
#include "iree/hal/drivers/task/device_spec_builder.h"
#include "iree/hal/drivers/task/executable/environment.h"
#include "iree/hal/drivers/task/profile.h"
#include "iree/hal/drivers/task/queue/queue.h"
#include "iree/hal/drivers/task/semaphore.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/utils/file_registry.h"

typedef struct iree_hal_task_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for small allocations like tasks and submissions.
  iree_arena_block_pool_t small_block_pool;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t large_block_pool;

  iree_host_size_t loader_count;
  iree_hal_executable_loader_t** loaders;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Routes default queue allocations to the best device-owned pool.
  iree_hal_pool_set_t default_pool_set;

  // Shared slab provider backing the default queue-allocation pools.
  iree_hal_slab_provider_t* default_slab_provider;

  // Shared notification published when default-pool reservations are released.
  iree_async_notification_t* default_pool_notification;

  // Suballocating pool used for requests up to the TLSF slab length.
  iree_hal_pool_t* default_tlsf_pool;

  // Direct per-allocation pool used for requests larger than one TLSF slab.
  iree_hal_pool_t* default_oversized_pool;

  // Proactor pool for async I/O. Retained for the lifetime of the device to
  // ensure proactor threads outlive all device resources (semaphores, etc.).
  iree_async_proactor_pool_t* proactor_pool;

  // Proactor selected from the pool for this device's async I/O operations.
  // Borrowed from the pool — valid as long as the pool is retained.
  iree_async_proactor_t* proactor;

  // Sink copied from device creation parameters for device-originated events.
  iree_hal_device_event_sink_t event_sink;

  // Shared frontier tracker for cross-device causal ordering. Retained after
  // topology assignment and released during device destruction.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Immutable device facts captured at creation time.
  iree_hal_device_spec_t* device_spec;

  // Active HAL-native profiling recorder, or NULL when profiling is disabled.
  iree_hal_task_profile_recorder_t* profile_recorder;

  // Next process-local profiling session identifier assigned by this device.
  uint64_t next_profile_session_id;

  // Next profiling submission identifier within the active session.
  iree_atomic_int64_t next_profile_submission_id;

  iree_hal_device_topology_info_t topology_info;

  // Pointer-unique identity of the task queue family.
  iree_hal_queue_family_t queue_family;

  // Number of successfully initialized entries in |queues|.
  iree_host_size_t queue_count;

  // Provisioned task queues embedded in the device allocation.
  iree_hal_task_queue_t queues[];
} iree_hal_task_device_t;

static const iree_hal_device_vtable_t iree_hal_task_device_vtable;

// Logical byte length for the default queue-allocation pool.
#define IREE_HAL_TASK_DEVICE_DEFAULT_POOL_RANGE_LENGTH_DEFAULT \
  (64 * 1024 * 1024)

// Minimum byte alignment for default-pool suballocations.
#define IREE_HAL_TASK_DEVICE_DEFAULT_POOL_ALIGNMENT_DEFAULT \
  IREE_HAL_HEAP_BUFFER_ALIGNMENT

// Maximum death-frontier entries stored per free default-pool block.
#define IREE_HAL_TASK_DEVICE_DEFAULT_POOL_FRONTIER_CAPACITY_DEFAULT \
  IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY

// Catch-all priority for direct allocations in the default pool set.
#define IREE_HAL_TASK_DEVICE_DEFAULT_POOL_PRIORITY_OVERSIZED 0

// Preferred priority for pooled allocations in the default pool set.
#define IREE_HAL_TASK_DEVICE_DEFAULT_POOL_PRIORITY_TLSF 10

static iree_hal_task_device_t* iree_hal_task_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_task_device_vtable);
  return (iree_hal_task_device_t*)base_value;
}

static bool iree_hal_task_device_query_pool_epoch(void* user_data,
                                                  iree_async_axis_t axis,
                                                  uint64_t epoch) {
  iree_hal_task_device_t* device = (iree_hal_task_device_t*)user_data;
  return iree_async_frontier_tracker_query_epoch(device->frontier_tracker, axis,
                                                 epoch);
}

static iree_status_t iree_hal_task_device_create_default_pools(
    iree_hal_task_device_t* device, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator, iree_hal_pool_set_t* out_pool_set,
    iree_hal_slab_provider_t** out_slab_provider,
    iree_async_notification_t** out_notification,
    iree_hal_pool_t** out_tlsf_pool, iree_hal_pool_t** out_oversized_pool) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_pool_set);
  IREE_ASSERT_ARGUMENT(out_slab_provider);
  IREE_ASSERT_ARGUMENT(out_notification);
  IREE_ASSERT_ARGUMENT(out_tlsf_pool);
  IREE_ASSERT_ARGUMENT(out_oversized_pool);
  memset(out_pool_set, 0, sizeof(*out_pool_set));
  *out_slab_provider = NULL;
  *out_notification = NULL;
  *out_tlsf_pool = NULL;
  *out_oversized_pool = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_pool_set_initialize(
      /*initial_capacity=*/2, host_allocator, out_pool_set));

  iree_hal_slab_provider_t* slab_provider = NULL;
  iree_async_notification_t* notification = NULL;
  iree_hal_pool_t* tlsf_pool = NULL;
  iree_hal_pool_t* oversized_pool = NULL;

  iree_status_t status =
      iree_hal_cpu_slab_provider_create(host_allocator, &slab_provider);
  if (iree_status_is_ok(status)) {
    status = iree_async_notification_create(
        proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_tlsf_pool_options_t options = {
        .tlsf_options =
            {
                .range_length =
                    IREE_HAL_TASK_DEVICE_DEFAULT_POOL_RANGE_LENGTH_DEFAULT,
                .alignment =
                    IREE_HAL_TASK_DEVICE_DEFAULT_POOL_ALIGNMENT_DEFAULT,
                .frontier_capacity =
                    IREE_HAL_TASK_DEVICE_DEFAULT_POOL_FRONTIER_CAPACITY_DEFAULT,
            },
        .budget_limit = 0,
    };
    status = iree_hal_tlsf_pool_create(
        options, slab_provider, notification,
        (iree_hal_pool_epoch_query_t){
            .fn = iree_hal_task_device_query_pool_epoch,
            .user_data = device,
        },
        host_allocator, &tlsf_pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_options_t options = {0};
    status = iree_hal_passthrough_pool_create(
        options, slab_provider, notification, host_allocator, &oversized_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_set_register(
        out_pool_set, IREE_HAL_TASK_DEVICE_DEFAULT_POOL_PRIORITY_OVERSIZED,
        oversized_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_set_register(
        out_pool_set, IREE_HAL_TASK_DEVICE_DEFAULT_POOL_PRIORITY_TLSF,
        tlsf_pool);
  }
  if (iree_status_is_ok(status)) {
    *out_slab_provider = slab_provider;
    *out_notification = notification;
    *out_tlsf_pool = tlsf_pool;
    *out_oversized_pool = oversized_pool;
    slab_provider = NULL;
    notification = NULL;
    tlsf_pool = NULL;
    oversized_pool = NULL;
  } else {
    iree_hal_pool_set_deinitialize(out_pool_set);
  }
  iree_hal_pool_release(oversized_pool);
  iree_hal_pool_release(tlsf_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
  return status;
}

void iree_hal_task_device_params_initialize(
    iree_hal_task_device_params_t* out_params) {
  out_params->arena_block_size = 32 * 1024;
  out_params->queue_scope_flags = IREE_TASK_SCOPE_FLAG_NONE;
  // Preserve the 256 KB strategy threshold previously used for scalar fills
  // and copies while applying it to the aggregate transaction payload.
  out_params->inline_transfer_threshold = 256 * 1024;
}

static iree_status_t iree_hal_task_device_check_params(
    const iree_hal_task_device_params_t* params, iree_host_size_t queue_count) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "must have at least one queue");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_task_device_create(
    iree_string_view_t identifier, const iree_hal_task_device_params_t* params,
    iree_host_size_t queue_count, iree_task_executor_t* const* queue_executors,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_allocator_t* device_allocator,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(!queue_count || queue_executors);
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_create_params_verify(create_params));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_task_device_check_params(params, queue_count));

  iree_hal_task_device_t* device = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t loaders_offset = 0;
  iree_host_size_t identifier_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(*device), &total_size,
                             IREE_STRUCT_FIELD_ALIGNED(
                                 queue_count, iree_hal_task_queue_t, 1, NULL),
                             IREE_STRUCT_FIELD_ALIGNED(
                                 loader_count, iree_hal_executable_loader_t*, 1,
                                 &loaders_offset),
                             IREE_STRUCT_FIELD_ALIGNED(identifier.size, char, 1,
                                                       &identifier_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_aligned(host_allocator, total_size,
                                        iree_alignof(iree_hal_task_device_t),
                                        /*offset=*/0, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_task_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + identifier_offset);
  device->host_allocator = host_allocator;
  device->device_allocator = device_allocator;
  iree_hal_allocator_retain(device_allocator);
  iree_atomic_store(&device->next_profile_submission_id, 0,
                    iree_memory_order_relaxed);

  // Retain the proactor pool. Each queue will get a NUMA-correct proactor
  // borrowed from the pool based on its executor's node assignment.
  device->proactor_pool = create_params->proactor_pool;
  device->event_sink = create_params->event_sink;
  iree_async_proactor_pool_retain(device->proactor_pool);

  // Select the device-level proactor from the first queue's executor NUMA
  // node. Used for device-owned pools, files, and semaphores.
  iree_task_topology_node_id_t default_node_id =
      iree_task_executor_node_id(queue_executors[0]);
  iree_status_t status = iree_async_proactor_pool_get_for_node(
      device->proactor_pool, default_node_id, &device->proactor);
  if (iree_status_is_ok(status)) {
    iree_hal_task_device_spec_params_t spec_params = {
        .logical_device_id = identifier,
        .display_name = identifier,
        .driver_id = IREE_SV("task"),
        .backend_id = IREE_SV("task"),
        .queue_count = queue_count,
        .default_queue_worker_count =
            iree_task_executor_worker_count(queue_executors[0]),
        .atomic_capabilities = iree_hal_task_atomic_capabilities(
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL),
        .loader_count = loader_count,
        .loaders = loaders,
    };
    status = iree_hal_task_device_spec_create(&spec_params, host_allocator,
                                              &device->device_spec);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_device_create_default_pools(
        device, device->proactor, device->host_allocator,
        &device->default_pool_set, &device->default_slab_provider,
        &device->default_pool_notification, &device->default_tlsf_pool,
        &device->default_oversized_pool);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_queue_family_initialize(/*ordinal=*/0, &device->queue_family);

    iree_arena_block_pool_initialize(4096, host_allocator,
                                     &device->small_block_pool);
    iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                     &device->large_block_pool);

    device->loader_count = loader_count;
    device->loaders =
        (iree_hal_executable_loader_t**)((uint8_t*)device + loaders_offset);
    for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
      device->loaders[i] = loaders[i];
      iree_hal_executable_loader_retain(device->loaders[i]);
    }

    device->queue_count = 0;
    for (iree_host_size_t i = 0; i < queue_count; ++i) {
      // Select a NUMA-correct proactor for this queue based on its executor's
      // node assignment. Falls back to the first proactor in the pool if the
      // executor's node doesn't have a dedicated proactor.
      iree_async_proactor_t* queue_proactor = NULL;
      iree_task_topology_node_id_t node_id =
          iree_task_executor_node_id(queue_executors[i]);
      status = iree_async_proactor_pool_get_for_node(device->proactor_pool,
                                                     node_id, &queue_proactor);
      if (!iree_status_is_ok(status)) break;

      status = iree_hal_task_queue_initialize(
          device->identifier, (iree_hal_device_t*)device, &device->queue_family,
          params->queue_scope_flags, queue_executors[i], queue_proactor,
          params->inline_transfer_threshold, &device->small_block_pool,
          &device->large_block_pool, device->device_allocator,
          &device->queues[i]);
      if (!iree_status_is_ok(status)) break;
      ++device->queue_count;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_device_clear_topology_info(
    iree_hal_task_device_t* device) {
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    iree_hal_task_queue_retire_frontier(&device->queues[i]);
  }
  iree_async_frontier_tracker_release(device->frontier_tracker);
  device->frontier_tracker = NULL;
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

static void iree_hal_task_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(!device->profile_recorder,
              "profiling sessions must be ended before device destruction");

  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    iree_hal_queue_t* queue = &device->queues[i].base;
    iree_atomic_ref_count_abort_if_uses(&queue->resource.ref_count);
    iree_hal_queue_release(queue);
  }
  iree_async_frontier_tracker_release(device->frontier_tracker);
  device->frontier_tracker = NULL;
  memset(&device->topology_info, 0, sizeof(device->topology_info));

  for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
    iree_hal_executable_loader_release(device->loaders[i]);
  }

  if (device->default_pool_set.entries) {
    iree_hal_pool_set_deinitialize(&device->default_pool_set);
  }
  iree_hal_pool_release(device->default_oversized_pool);
  iree_hal_pool_release(device->default_tlsf_pool);
  iree_hal_slab_provider_release(device->default_slab_provider);
  iree_async_notification_release(device->default_pool_notification);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_hal_device_spec_release(device->device_spec);
  iree_async_proactor_pool_release(device->proactor_pool);

  iree_arena_block_pool_deinitialize(&device->large_block_pool);
  iree_arena_block_pool_deinitialize(&device->small_block_pool);

  iree_allocator_free_aligned(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_task_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_task_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_task_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_task_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_task_device_trim(iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);

  // Before trimming the block pools try to trim subsystems that may be holding
  // on to blocks.
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    iree_hal_task_queue_trim(&device->queues[i]);
  }
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  IREE_RETURN_IF_ERROR(iree_hal_pool_trim(device->default_tlsf_pool));
  IREE_RETURN_IF_ERROR(iree_hal_pool_trim(device->default_oversized_pool));

  iree_arena_block_pool_trim(&device->small_block_pool);
  iree_arena_block_pool_trim(&device->large_block_pool);

  return iree_ok_status();
}

static const iree_hal_device_spec_t* iree_hal_task_device_spec(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return device->device_spec;
}

static const iree_hal_queue_family_t* iree_hal_task_device_queue_family(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return family_ordinal == 0 ? &device->queue_family : NULL;
}

static iree_hal_queue_t* iree_hal_task_device_queue(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal,
    iree_hal_queue_ordinal_t queue_ordinal) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  if (family_ordinal != 0 || queue_ordinal >= device->queue_count) return NULL;
  return &device->queues[queue_ordinal].base;
}

static iree_status_t iree_hal_task_device_sample_observation(
    iree_hal_device_t* base_device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  if (iree_any_bit_set(requested_flags,
                       IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_observation_populate_memory_total_from_spec(
            device->device_spec, out_observation));
  }
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_task_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_task_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_task_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  if (!topology_info) {
    iree_hal_task_device_clear_topology_info(device);
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* frontier_tracker =
      topology_info->frontier.tracker;
  iree_async_axis_t base_axis = topology_info->frontier.base_axis;

  const uint8_t session_epoch = iree_async_axis_session(base_axis);
  const uint8_t machine_index = iree_async_axis_machine(base_axis);
  const uint8_t device_index = iree_async_axis_device_index(base_axis);
  iree_status_t status = iree_ok_status();
  iree_host_size_t assigned_queue_count = 0;
  for (iree_host_size_t i = 0;
       i < device->queue_count && iree_status_is_ok(status); ++i) {
    iree_async_axis_t queue_axis = iree_async_axis_make_queue(
        session_epoch, machine_index, device_index, (uint8_t)i);
    status = iree_hal_task_queue_assign_frontier(&device->queues[i],
                                                 frontier_tracker, queue_axis);
    if (iree_status_is_ok(status)) {
      assigned_queue_count = i + 1;
    }
  }

  if (iree_status_is_ok(status)) {
    device->topology_info = *topology_info;
    device->frontier_tracker = frontier_tracker;
    iree_async_frontier_tracker_retain(device->frontier_tracker);
  } else {
    for (iree_host_size_t i = 0; i < assigned_queue_count; ++i) {
      iree_hal_task_queue_retire_frontier(&device->queues[i]);
    }
  }
  return status;
}

static iree_status_t iree_hal_task_device_create_channel(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_task_device_create_command_buffer(
    iree_hal_device_t* base_device, const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_block_command_buffer_create(
      iree_hal_device_allocator(base_device), queue_family, mode,
      command_categories, binding_capacity, &device->large_block_pool,
      device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_task_device_load_executable(
    iree_hal_device_t* base_device, const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_host_size_t worker_capacity = 0;
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    worker_capacity +=
        iree_task_executor_worker_count(device->queues[i].executor);
  }
  return iree_hal_executable_loader_select_and_load(
      device->loader_count, device->loaders, queue_family, target, load_params,
      worker_capacity, out_executable);
}

static iree_status_t iree_hal_task_device_import_file(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_family_affinity, access,
      handle, device->proactor, iree_hal_device_host_allocator(base_device),
      out_file);
}

static iree_status_t iree_hal_task_device_create_semaphore(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  (void)queue_family_affinity;
  (void)flags;
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_task_semaphore_create(device->proactor, initial_value,
                                        device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_task_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  if (iree_hal_task_semaphore_isa(semaphore)) {
    // Fast-path for semaphores related to this device.
    // TODO(benvanik): ensure the creating devices are compatible as if
    // independent task systems are used things may not work right (ownership
    // confusion).
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  // For now we support all semaphore types as we only need wait sources and
  // all semaphores can be wrapped in those.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
}

static iree_status_t iree_hal_task_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, const iree_hal_queue_family_t* queue_family,
    iree_hal_queue_pool_backend_t* out_backend) {
  (void)queue_family;
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  out_backend->slab_provider = device->default_slab_provider;
  out_backend->notification = device->default_pool_notification;
  out_backend->epoch_query = (iree_hal_pool_epoch_query_t){
      .fn = iree_hal_task_device_query_pool_epoch,
      .user_data = device,
  };
  return iree_ok_status();
}

static uint32_t iree_hal_task_device_profile_count(iree_host_size_t count) {
  return count > UINT32_MAX ? UINT32_MAX : (uint32_t)count;
}

static iree_hal_task_profile_queue_scope_t
iree_hal_task_device_profile_queue_scope(const iree_hal_task_device_t* device,
                                         uint32_t queue_ordinal) {
  const uint32_t physical_device_ordinal =
      device->topology_info.topology ? device->topology_info.topology_index : 0;
  iree_hal_task_profile_queue_scope_t scope = {
      .physical_device_ordinal = physical_device_ordinal,
      .queue_ordinal = queue_ordinal,
      .stream_id = ((uint64_t)physical_device_ordinal << 32) | queue_ordinal,
  };
  return scope;
}

static iree_status_t iree_hal_task_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  if (device->profile_recorder) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot nest task profile captures");
  }

  const uint32_t physical_device_ordinal =
      device->topology_info.topology ? device->topology_info.topology_index : 0;
  iree_hal_profile_device_record_t device_record =
      iree_hal_profile_device_record_default();
  device_record.physical_device_ordinal = physical_device_ordinal;
  device_record.queue_count =
      iree_hal_task_device_profile_count(device->queue_count);

  iree_hal_profile_queue_record_t* queue_records = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      device->host_allocator, device->queue_count, sizeof(*queue_records),
      (void**)&queue_records));
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    const uint32_t queue_ordinal = iree_hal_task_device_profile_count(i);
    const iree_hal_task_profile_queue_scope_t scope =
        iree_hal_task_device_profile_queue_scope(device, queue_ordinal);
    queue_records[i] = iree_hal_profile_queue_record_default();
    queue_records[i].physical_device_ordinal = scope.physical_device_ordinal;
    queue_records[i].queue_ordinal = scope.queue_ordinal;
    queue_records[i].stream_id = scope.stream_id;
  }

  iree_hal_task_profile_recorder_options_t recorder_options = {
      .name = device->identifier,
      .session_id = ++device->next_profile_session_id,
      .device_record_count = 1,
      .device_records = &device_record,
      .queue_record_count = device->queue_count,
      .queue_records = queue_records,
  };
  iree_hal_task_profile_recorder_t* recorder = NULL;
  iree_status_t status = iree_hal_task_profile_recorder_create(
      &recorder_options, options, device->host_allocator, &recorder);
  iree_allocator_free(device->host_allocator, queue_records);
  if (!iree_status_is_ok(status) || !recorder) return status;

  iree_atomic_store(&device->next_profile_submission_id, 1,
                    iree_memory_order_relaxed);
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    iree_hal_task_queue_set_profile_recorder(
        &device->queues[i], recorder,
        iree_hal_task_device_profile_queue_scope(
            device, iree_hal_task_device_profile_count(i)),
        &device->next_profile_submission_id);
  }
  device->profile_recorder = recorder;
  return iree_ok_status();
}

static iree_status_t iree_hal_task_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_task_profile_recorder_flush(device->profile_recorder);
}

static iree_status_t iree_hal_task_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_hal_task_profile_recorder_t* recorder = device->profile_recorder;
  if (!recorder) return iree_ok_status();

  const iree_hal_task_profile_queue_scope_t empty_scope =
      iree_hal_task_profile_queue_scope_default();
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    iree_hal_task_queue_set_profile_recorder(
        &device->queues[i], /*profile_recorder=*/NULL, empty_scope,
        /*submission_counter=*/NULL);
  }
  device->profile_recorder = NULL;
  iree_atomic_store(&device->next_profile_submission_id, 0,
                    iree_memory_order_relaxed);

  iree_status_t status = iree_hal_task_profile_recorder_end(recorder);
  iree_hal_task_profile_recorder_destroy(recorder);
  return status;
}

static const iree_hal_device_vtable_t iree_hal_task_device_vtable = {
    .destroy = iree_hal_task_device_destroy,
    .id = iree_hal_task_device_id,
    .host_allocator = iree_hal_task_device_host_allocator,
    .device_allocator = iree_hal_task_device_allocator,
    .replace_channel_provider = iree_hal_task_replace_channel_provider,
    .trim = iree_hal_task_device_trim,
    .device_spec = iree_hal_task_device_spec,
    .queue_family = iree_hal_task_device_queue_family,
    .queue = iree_hal_task_device_queue,
    .sample_observation = iree_hal_task_device_sample_observation,
    .topology_info = iree_hal_task_device_topology_info,
    .refine_topology_edge = iree_hal_task_device_refine_topology_edge,
    .assign_topology_info = iree_hal_task_device_assign_topology_info,
    .create_channel = iree_hal_task_device_create_channel,
    .create_command_buffer = iree_hal_task_device_create_command_buffer,
    .load_executable = iree_hal_task_device_load_executable,
    .import_file = iree_hal_task_device_import_file,
    .create_semaphore = iree_hal_task_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_task_device_query_semaphore_compatibility,
    .query_queue_pool_backend = iree_hal_task_device_query_queue_pool_backend,
    .profiling_begin = iree_hal_task_device_profiling_begin,
    .profiling_flush = iree_hal_task_device_profiling_flush,
    .profiling_end = iree_hal_task_device_profiling_end,
};
