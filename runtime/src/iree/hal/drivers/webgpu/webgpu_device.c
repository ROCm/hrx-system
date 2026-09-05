// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/webgpu/webgpu_allocator.h"
#include "iree/hal/drivers/webgpu/webgpu_builtins.h"
#include "iree/hal/drivers/webgpu/webgpu_command_buffer.h"
#include "iree/hal/drivers/webgpu/webgpu_executable.h"
#include "iree/hal/drivers/webgpu/webgpu_fd_file.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"
#include "iree/hal/drivers/webgpu/webgpu_queue.h"
#include "iree/hal/drivers/webgpu/webgpu_semaphore.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Flags controlling ownership and behavior.
  iree_hal_webgpu_device_flags_t flags;

  // Bridge handle for the GPUDevice.
  iree_hal_webgpu_handle_t device_handle;

  // Built-in WGSL compute pipelines for buffer operations (fill, copy) that
  // WebGPU's command encoder does not natively support with arbitrary
  // alignment.
  iree_hal_webgpu_builtins_t builtins;

  // True if the execution context supports blocking waits (Atomics.wait).
  // On the browser main thread this is false — Atomics.wait throws TypeError.
  // On Web Workers with cross-origin isolation this is true.
  // Native (Dawn) always supports blocking.
  bool can_block;

  // Proactor pool for async I/O. Retained for the lifetime of the device to
  // ensure proactor threads outlive all device resources (semaphores, etc.).
  iree_async_proactor_pool_t* proactor_pool;

  // Sink copied from device creation parameters for device-originated events.
  iree_hal_device_event_sink_t event_sink;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Immutable device facts captured at creation time.
  iree_hal_device_spec_t* device_spec;

  // Topology information if this device is part of a multi-device topology.
  iree_hal_device_topology_info_t topology_info;

  // Pointer-unique identity of the WebGPU queue family.
  iree_hal_queue_family_t queue_family;

  // Number of successfully initialized provisioned queues.
  iree_host_size_t queue_count;

  // WebGPU exposes exactly one native queue per device.
  iree_hal_webgpu_queue_t queue;

  // + trailing identifier string storage
} iree_hal_webgpu_device_t;

static const iree_hal_device_vtable_t iree_hal_webgpu_device_vtable;
static void iree_hal_webgpu_device_clear_topology_info(
    iree_hal_webgpu_device_t* device);

static iree_hal_webgpu_device_t* iree_hal_webgpu_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_device_vtable);
  return (iree_hal_webgpu_device_t*)base_value;
}

static iree_status_t iree_hal_webgpu_device_spec_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_device_spec_t** out_device_spec) {
  IREE_ASSERT_ARGUMENT(out_device_spec);
  *out_device_spec = NULL;

  iree_hal_physical_device_spec_t physical_device = {
      .identity =
          {
              .display_name = identifier,
              .backend_path = identifier,
              .flags = IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NONE,
          },
      .physical_ordinal = 0,
      .partition_ordinal = 0,
      .partition_count = 1,
      .physical_device_affinity = 1ull,
  };
  iree_hal_device_identity_spec_t identity = {
      .logical_device_id = identifier,
      .display_name = identifier,
      .driver_id = IREE_SV("webgpu"),
      .backend_id = IREE_SV("webgpu"),
      .physical_device_count = 1,
      .physical_devices = &physical_device,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_hal_queue_family_spec_t queue_family = {
      .name = IREE_SV("default"),
      .provisioned_queue_count = 1,
      .priority_count = 1,
      .physical_device_affinity = 1ull,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL,
      .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
  };
  iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &queue_family,
      .flags = IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(&builder, &identity);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_queues(&builder, &queues);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_device_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}

iree_status_t iree_hal_webgpu_device_create(
    iree_string_view_t identifier, iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_device_flags_t flags,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_create_params_verify(create_params));

  iree_hal_webgpu_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_webgpu_device_vtable,
                               &device->resource);
  iree_hal_queue_family_initialize(/*ordinal=*/0, &device->queue_family);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + total_size - identifier.size);
  device->host_allocator = host_allocator;
  device->flags = flags;

  // Store bridge handle. WebGPU has exactly one queue per device.
  device->device_handle = device_handle;
  device->can_block = iree_hal_webgpu_import_can_block() != 0;

  // Retain the proactor pool and acquire a proactor for queue initialization.
  device->proactor_pool = create_params->proactor_pool;
  device->event_sink = create_params->event_sink;
  iree_async_proactor_pool_retain(device->proactor_pool);

  iree_async_proactor_t* proactor = NULL;
  iree_status_t status =
      iree_async_proactor_pool_get(device->proactor_pool, 0, &proactor);
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_device_spec_create(identifier, host_allocator,
                                                &device->device_spec);
  }

  // Create the builtin compute pipelines for fill/copy operations.
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_webgpu_builtins_initialize(device_handle, &device->builtins);
  }

  // Initialize the queue (owns block pool, scratch builder, epoch tracking).
  if (iree_status_is_ok(status)) {
    iree_hal_webgpu_handle_t queue_handle =
        iree_hal_webgpu_import_device_get_queue(device_handle);
    status = iree_hal_webgpu_queue_initialize(
        &device->queue_family, device_handle, queue_handle, &device->builtins,
        proactor,
        /*frontier_tracker=*/NULL, /*axis=*/0, host_allocator, &device->queue);
    if (iree_status_is_ok(status)) device->queue_count = 1;
  }

  // Create the device allocator.
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_allocator_create(device_handle, host_allocator,
                                              &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_webgpu_device_wrap(
    iree_string_view_t identifier, iree_hal_webgpu_handle_t device_handle,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  // Create an inline-mode proactor pool (single node, no threads).
  iree_async_proactor_pool_t* proactor_pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_pool_create(
              /*node_count=*/1, /*node_ids=*/NULL,
              iree_async_proactor_pool_options_default(), host_allocator,
              &proactor_pool));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  // Caller retains ownership of the device handle — no OWNS_DEVICE_HANDLE.
  iree_status_t status = iree_hal_webgpu_device_create(
      identifier, device_handle, IREE_HAL_WEBGPU_DEVICE_FLAG_NONE,
      &create_params, host_allocator, out_device);

  // The device retains the pool — release our reference.
  iree_async_proactor_pool_release(proactor_pool);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_webgpu_device_clear_topology_info(device);
  if (device->queue_count != 0) {
    iree_atomic_ref_count_abort_if_uses(&device->queue.base.resource.ref_count);
    iree_hal_queue_release(&device->queue.base);
    device->queue_count = 0;
  }
  iree_hal_webgpu_builtins_deinitialize(&device->builtins);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_hal_device_spec_release(device->device_spec);
  iree_async_proactor_pool_release(device->proactor_pool);

  // Release the GPUDevice bridge handle only if we own it. When created via
  // iree_hal_webgpu_device_wrap(), the caller retains ownership and the handle
  // must outlive the HAL device.
  if (iree_all_bits_set(device->flags,
                        IREE_HAL_WEBGPU_DEVICE_FLAG_OWNS_DEVICE_HANDLE)) {
    iree_hal_webgpu_import_device_destroy(device->device_handle);
  }

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_webgpu_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_webgpu_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_webgpu_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_webgpu_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_webgpu_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  return iree_ok_status();
}

static const iree_hal_device_spec_t* iree_hal_webgpu_device_spec(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return device->device_spec;
}

static const iree_hal_queue_family_t* iree_hal_webgpu_device_queue_family(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return family_ordinal == 0 ? &device->queue_family : NULL;
}

static iree_hal_queue_t* iree_hal_webgpu_device_queue(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal,
    iree_hal_queue_ordinal_t queue_ordinal) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  if (family_ordinal != 0 || queue_ordinal >= device->queue_count) return NULL;
  return &device->queue.base;
}

static iree_status_t iree_hal_webgpu_device_sample_observation(
    iree_hal_device_t* base_device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  if (iree_any_bit_set(requested_flags,
                       IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_observation_populate_memory_total_from_spec(
            device->device_spec, out_observation));
  }
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_webgpu_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return &device->topology_info;
}

static void iree_hal_webgpu_device_clear_topology_info(
    iree_hal_webgpu_device_t* device) {
  if (device->queue.frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        device->queue.frontier_tracker, device->queue.axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->queue.frontier_tracker);
    device->queue.frontier_tracker = NULL;
    device->queue.axis = 0;
  }
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

static iree_status_t iree_hal_webgpu_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  // WebGPU devices are isolated — no direct peer-to-peer access.
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  if (!topology_info) {
    iree_hal_webgpu_device_clear_topology_info(device);
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* frontier_tracker =
      topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL));
  iree_hal_webgpu_device_clear_topology_info(device);
  device->topology_info = *topology_info;
  device->queue.frontier_tracker = frontier_tracker;
  device->queue.axis = axis;
  iree_async_frontier_tracker_retain(device->queue.frontier_tracker);
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  // WebGPU has no collective communication primitives.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support collective channels");
}

static iree_status_t iree_hal_webgpu_device_create_command_buffer(
    iree_hal_device_t* base_device, const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_command_buffer_create(
      device->device_handle, device->queue.queue_handle, &device->builtins,
      &device->queue.block_pool, iree_hal_device_allocator(base_device),
      queue_family, mode, command_categories, binding_capacity,
      device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_webgpu_device_load_executable(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_executable_create(device->device_handle, load_params,
                                           device->host_allocator,
                                           out_executable);
}

static iree_status_t iree_hal_webgpu_device_import_file(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  switch (primitive.type) {
    case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION:
      // Use generic memory_file — storage_buffer() returns a HOST_LOCAL heap
      // buffer via the heap_buffer_wrap fallback (since WebGPU cannot import
      // host allocations as GPU buffers).
      return iree_hal_memory_file_wrap(
          iree_hal_device_allocator(base_device), queue_family_affinity, access,
          handle, IREE_HAL_MEMORY_FILE_FLAG_NONE,
          iree_hal_device_host_allocator(base_device), out_file);
    case IREE_IO_FILE_HANDLE_TYPE_FD: {
      // Use WebGPU FD file — the fd is a JS file object table index, not a
      // POSIX fd. The standard fd_file uses pread/pwrite (unavailable on wasm).
      uint64_t length =
          iree_hal_webgpu_import_file_get_length((uint32_t)primitive.value.fd);
      return iree_hal_webgpu_fd_file_from_handle(
          access, handle, length, iree_hal_device_host_allocator(base_device),
          out_file);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported file handle type %d",
                              (int)primitive.type);
  }
}

static iree_status_t iree_hal_webgpu_device_create_semaphore(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  (void)queue_family_affinity;
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_semaphore_create(device->queue.proactor, initial_value,
                                          flags, device->host_allocator,
                                          out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_webgpu_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  // WebGPU semaphores are software timeline semaphores. Device wait/signal
  // always works (the proactor advances the timeline on onSubmittedWorkDone
  // completion, and submission checks timeline values before queue.submit).
  // Host signal always works (just a CAS on the timeline value).
  //
  // Host WAIT requires blocking (Atomics.wait), which is only available on
  // Web Workers — not on the browser main thread where Atomics.wait throws
  // TypeError. The can_block flag is queried once at device creation via the
  // bridge and gates HOST_WAIT here so callers never attempt a blocking wait
  // in a context that can't support it.
  iree_hal_semaphore_compatibility_t compatibility =
      IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_ONLY |
      IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL;
  if (device->can_block) {
    compatibility |= IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT;
  }
  return compatibility;
}

//===----------------------------------------------------------------------===//
// Profiling (no-ops for WebGPU)
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // WebGPU has no user-accessible profiling API.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_profiling_flush(
    iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_profiling_end(
    iree_hal_device_t* base_device) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_webgpu_device_vtable = {
    .destroy = iree_hal_webgpu_device_destroy,
    .id = iree_hal_webgpu_device_id,
    .host_allocator = iree_hal_webgpu_device_host_allocator,
    .device_allocator = iree_hal_webgpu_device_allocator,
    .replace_channel_provider = iree_hal_webgpu_replace_channel_provider,
    .trim = iree_hal_webgpu_device_trim,
    .device_spec = iree_hal_webgpu_device_spec,
    .queue_family = iree_hal_webgpu_device_queue_family,
    .queue = iree_hal_webgpu_device_queue,
    .sample_observation = iree_hal_webgpu_device_sample_observation,
    .topology_info = iree_hal_webgpu_device_topology_info,
    .refine_topology_edge = iree_hal_webgpu_device_refine_topology_edge,
    .assign_topology_info = iree_hal_webgpu_device_assign_topology_info,
    .create_channel = iree_hal_webgpu_device_create_channel,
    .create_command_buffer = iree_hal_webgpu_device_create_command_buffer,
    .load_executable = iree_hal_webgpu_device_load_executable,
    .import_file = iree_hal_webgpu_device_import_file,
    .create_semaphore = iree_hal_webgpu_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_webgpu_device_query_semaphore_compatibility,
    .profiling_begin = iree_hal_webgpu_device_profiling_begin,
    .profiling_flush = iree_hal_webgpu_device_profiling_flush,
    .profiling_end = iree_hal_webgpu_device_profiling_end,
};
