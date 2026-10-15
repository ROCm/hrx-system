// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/device.h"

#include <stdio.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/allocation.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/queue.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/hal/utils/file_registry.h"

typedef struct iree_hal_amd_gpu_device_t {
  // Refcounted HAL owner; child family identities remain borrowed.
  iree_hal_resource_t resource;
  // Allocator owning this wrapper and its inline queue/source arrays.
  iree_allocator_t host_allocator;
  // Unique native owner, transferred after successful construction.
  iree_hal_amd_gpu_context_t* context;
  // Immutable public device facts.
  iree_hal_device_spec_t* spec;
  // Owned processor spelling for target views not replaced by table entries.
  char processor_name[32];
  // Qualified physical compiler identity borrowing owned or static strings.
  iree_hal_amd_gpu_target_identity_t target;
  // Canonical exact identity for the explicitly provisioned native family.
  iree_hal_queue_family_t family;
  // Number of fully constructed queues in the inline array.
  uint32_t queue_count;
  // Owned queue references. Every queue joins before any signal backing frees.
  iree_hal_queue_t** queues;
  // Immutable inline sources, one per queue and one final shutdown observer.
  iree_hal_amd_gpu_completion_source_t* sources;
  // Sole native-event completion/retirement worker.
  iree_hal_amd_gpu_completion_t completion;
  // Retained placement-local process infrastructure, independent of the pool.
  iree_async_proactor_pool_entry_t* proactor_entry;
  // Cold lifetime handoff; submission never accesses its mutex/notification.
  struct {
    // Joins the complete final-release wake before the worker frees its owner.
    iree_slim_mutex_t mutex;
    // Parks an already failed completion worker until public ownership ends.
    iree_notification_t notification;
    // Last public release has closed every provisioned queue.
    iree_atomic_int32_t closing;
  } shutdown;
  // Native allocation policy and shared progress supplied to explicit pools.
  struct {
    // One service for cold memory, executable cleanup, and deferred
    // publication.
    iree_hal_memory_maintenance_t* service;
    // Shared queue-allocation captures and transient-root metadata.
    iree_hal_amd_gpu_allocation_context_t allocations;
    // Capacity notification driven by the existing shared proactor.
    iree_async_notification_t* notification;
    // Exact-family address and usage requirements for the shared backend.
    iree_hal_amd_fabric_memory_family_t family;
    // Owned placement choices: physical-local scopes followed by system memory.
    iree_hal_amd_fabric_memory_storage_t* storage;
    // Borrowed factory view, without any default pool or slab allocation.
    iree_hal_amd_fabric_memory_backend_t backend;
    // Persistent native access set used by internal control/image allocations.
    amdf_memory_device_access_t access;
    // Qualified coherent system backing for native control and placed images.
    iree_hal_amd_fabric_memory_options_t control;
    // Address domain qualified by the control profile's GPU access.
    uint32_t address_domain_ordinal;
  } memory;
  // Group assignment; each queue retains the tracker through final retirement.
  iree_hal_device_topology_info_t topology;
} iree_hal_amd_gpu_device_t;

static const iree_hal_device_vtable_t iree_hal_amd_gpu_device_vtable;

void iree_hal_amd_gpu_device_options_initialize(
    iree_hal_amd_gpu_device_options_t* out_options) {
  *out_options = (iree_hal_amd_gpu_device_options_t){
      .queue_count = 1,
      .host_numa_node = UINT32_MAX,
      .pm4 = {.ring_byte_length = 65536,
              .retirement_capacity = 256,
              .argument_capacity = 65536,
              .capture_block_size = 1024,
              .capture_block_count = 64},
  };
}

static iree_string_view_t iree_hal_amd_gpu_device_id(iree_hal_device_t* base) {
  const iree_hal_amd_gpu_device_t* device = (void*)base;
  switch (amdf_gpu_endpoint_architecture(&device->context->target_info)) {
    case AMDF_GPU_ARCHITECTURE_RDNA:
      return IREE_SV("rdna");
    case AMDF_GPU_ARCHITECTURE_CDNA:
      return IREE_SV("cdna");
    default:
      return IREE_SV("amd");
  }
}

static iree_status_t iree_hal_amd_gpu_device_build_spec(
    iree_hal_amd_gpu_device_t* device, iree_string_view_t display_name,
    uint32_t queue_count) {
  const iree_hal_physical_device_spec_t physical = {
      .identity = {.display_name = display_name, .vendor_id = 0x1002},
      .partition_count = 1,
      .physical_device_affinity = 1,
  };
  const iree_hal_device_identity_spec_t identity = {
      .logical_device_id = iree_hal_amd_gpu_device_id((void*)device),
      .display_name = display_name,
      .driver_id = IREE_SV("amd"),
      .backend_id = IREE_SV("libamdf"),
      .vendor_name = IREE_SV("AMD"),
      .vendor_id = 0x1002,
      .physical_device_count = 1,
      .physical_devices = &physical,
  };
  const iree_hal_queue_priority_t priority = IREE_HAL_QUEUE_PRIORITY_NORMAL;
  const iree_hal_queue_family_spec_t family = {
      .name = IREE_SV("pm4"),
      .provisioned_queue_count = queue_count,
      .priority_count = 1,
      .priorities = &priority,
      .physical_device_affinity = 1,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
  };
  const iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &family,
  };
  const iree_hal_device_dispatch_spec_t dispatch = {
      .launch = {.maximum_workgroup_invocations = 1024,
                 .maximum_workgroup_size = {1024, 1024, 1024},
                 .maximum_workgroup_count = {UINT32_MAX, UINT32_MAX,
                                             UINT32_MAX}},
  };
  char target_key[128];
  iree_host_size_t target_key_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_format_artifact_key(
      &device->target, sizeof(target_key), target_key, &target_key_length));
  const iree_hal_executable_target_t target = {
      .family = IREE_SV("amdgpu"),
      .target_key = iree_make_string_view(target_key, target_key_length),
      .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
      .physical_device_affinity = 1,
  };
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(device->host_allocator, &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(&builder, &identity);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_queues(&builder, &queues);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_device_spec_builder_add_executable_target(&builder, &target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, &device->spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}

static iree_status_t iree_hal_amd_gpu_device_prepare_control(
    iree_hal_amd_gpu_device_t* device) {
  const iree_hal_amd_gpu_context_t* context = device->context;
  device->memory.access = (amdf_memory_device_access_t){
      .device = context->device,
      .requirements = {.access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE,
                       .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                                AMDF_MEMORY_FLAG_HOST_COHERENT,
                       .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU},
  };
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(context->api->memory_scope_query_info(
                                        context->fabric->system_scope, &info),
                                    "memory_scope_query_info"));
  iree_status_t status = iree_ok_status();
  bool found = false;
  for (uint32_t i = 0;
       i < info.memory_profile_count && !found && iree_status_is_ok(status);
       ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t access = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(access),
    };
    amdf_status_t native_status =
        context->api->memory_scope_query_device_profile(
            context->fabric->system_scope, i, 1, &device->memory.access,
            &profile, &access);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                           "memory_scope_query_device_profile");
    if (iree_status_is_ok(status) &&
        iree_all_bits_set(profile.roles,
                          AMDF_MEMORY_PROFILE_ROLE_CREATE |
                              AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        iree_any_bit_set(profile.supported_flags,
                         AMDF_MEMORY_FLAG_HOST_VISIBLE)) {
      device->memory.control = (iree_hal_amd_fabric_memory_options_t){
          .scope = context->fabric->system_scope,
          .profile = profile,
          .access_count = 1,
          .accesses = &device->memory.access,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      device->memory.address_domain_ordinal =
          access.device_address.address_domain_ordinal;
      found = true;
    }
  }
  if (iree_status_is_ok(status) && !found) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "GPU has no coherent host-visible control "
                              "and executable backing profile");
  }
  return status;
}

static bool iree_hal_amd_gpu_device_query_pool_epoch(void* user_data,
                                                     iree_async_axis_t axis,
                                                     uint64_t epoch) {
  const iree_hal_amd_gpu_device_t* device = user_data;
  return device->topology.frontier.tracker &&
         iree_async_frontier_tracker_query_epoch(
             device->topology.frontier.tracker, axis, epoch);
}

static void iree_hal_amd_gpu_device_initialize_memory(
    iree_hal_amd_gpu_device_t* device, uint32_t native_family_ordinal) {
  static const iree_hal_amd_fabric_memory_binding_t bindings[] = {
      {.type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
       .address_kind = AMDF_MEMORY_ADDRESS_GPU},
  };
  device->memory.family = (iree_hal_amd_fabric_memory_family_t){
      .queue_family_ordinal = native_family_ordinal,
      .supported_usage =
          IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .binding_count = IREE_ARRAYSIZE(bindings),
      .bindings = bindings,
  };
  const iree_hal_amd_gpu_context_t* context = device->context;
  for (uint32_t i = 0; i < context->memory_scope_count; ++i) {
    device->memory.storage[i] = (iree_hal_amd_fabric_memory_storage_t){
        .scope = context->memory_scopes[i],
        .physical_device_affinity = 1,
        .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    };
  }
  device->memory.storage[context->memory_scope_count] =
      (iree_hal_amd_fabric_memory_storage_t){
          .scope = context->fabric->system_scope,
          .memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      };
  const iree_hal_amd_fabric_memory_backend_options_t options = {
      .api = context->api,
      .instance = context->fabric->instance,
      .device = context->device,
      .address_domain_ordinal = device->memory.address_domain_ordinal,
      .storage_count = context->memory_scope_count + 1,
      .storage = device->memory.storage,
      .event_sink = context->event_sink,
      .families = &device->memory.family,
  };
  iree_hal_amd_fabric_memory_backend_initialize(
      (void*)device, &options, device->memory.notification,
      device->memory.service,
      (iree_hal_pool_epoch_query_t){iree_hal_amd_gpu_device_query_pool_epoch,
                                    device},
      &device->memory.backend);
}

static bool iree_hal_amd_gpu_device_is_closing(void* user_data) {
  iree_hal_amd_gpu_device_t* device = user_data;
  return iree_atomic_load(&device->shutdown.closing, iree_memory_order_acquire);
}

static bool iree_hal_amd_gpu_device_is_idle(iree_hal_amd_gpu_device_t* device) {
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    if (!iree_hal_amd_gpu_pm4_umq_queue_is_idle(device->queues[i])) {
      return false;
    }
  }
  return true;
}

// Joins native owners before freeing any device-owned construction facts. Both
// constructor rollback and completion-worker teardown run outside service.
static bool iree_hal_amd_gpu_device_release_children(
    iree_hal_amd_gpu_device_t* device) {
  if (device->context->cleanup_blocked) {
    // A failed child constructor may have kept native ownership without
    // returning a queue. Its borrowed construction facts must also stay live.
    return false;
  }
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    if (!device->queues[i]) {
      continue;
    }
    amdf_status_t status =
        iree_hal_amd_gpu_pm4_umq_queue_shutdown(device->queues[i]);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(device->context, status,
                                      "PM4 queue native ownership join");
      return false;
    }
    iree_hal_queue_release(device->queues[i]);
    device->queues[i] = NULL;
  }
  if (device->memory.service) {
    iree_hal_amd_gpu_allocation_context_deinitialize(
        &device->memory.allocations);
  }
  iree_hal_memory_maintenance_release(device->memory.service);
  iree_async_notification_release(device->memory.notification);
  iree_async_proactor_pool_entry_release(device->proactor_entry);
  iree_hal_device_spec_release(device->spec);
  iree_notification_deinitialize(&device->shutdown.notification);
  iree_slim_mutex_deinitialize(&device->shutdown.mutex);
  return true;
}

static void iree_hal_amd_gpu_device_completion_exit(void* user_data) {
  iree_hal_amd_gpu_device_t* device = user_data;
  // An early native wait failure cannot consume public construction ownership.
  // The cold mutex also joins notification_post/event_signal's final accesses.
  iree_notification_await(&device->shutdown.notification,
                          iree_hal_amd_gpu_device_is_closing, device,
                          iree_infinite_timeout());
  iree_slim_mutex_lock(&device->shutdown.mutex);
  iree_slim_mutex_unlock(&device->shutdown.mutex);
  // The callback is the worker's final action. Releasing its handle detaches
  // this already joined owner; no state is accessed by the thread afterward.
  iree_thread_release(device->completion.thread);
  device->completion.thread = NULL;
  if (!iree_hal_amd_gpu_device_is_idle(device)) {
    iree_hal_amd_gpu_context_report(
        device->context, amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST),
        "GPU observer exited with unproven native uses; preserving ownership");
    return;
  }
  if (!iree_hal_amd_gpu_device_release_children(device)) {
    return;
  }
  iree_hal_amd_gpu_context_free(device->context);
  iree_allocator_free(device->host_allocator, device);
}

static bool iree_hal_amd_gpu_device_poll_shutdown(void* user_data,
                                                  amdf_status_t wait_status) {
  iree_hal_amd_gpu_device_t* device = user_data;
  if (amdf_status_is_ok(wait_status) &&
      iree_hal_amd_gpu_device_is_closing(device) &&
      iree_hal_amd_gpu_device_is_idle(device)) {
    amdf_status_t status =
        iree_hal_amd_gpu_completion_request_stop(&device->completion);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(device->context, status,
                                      "GPU completion stop wake");
    }
  }
  return false;
}

static void iree_hal_amd_gpu_device_destroy(iree_hal_device_t* base) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    if (iree_atomic_ref_count_load(&device->queues[i]->resource.ref_count) !=
        1) {
      iree_hal_amd_gpu_context_report(
          device->context, amdf_make_api_status(AMDF_STATUS_CODE_BUSY),
          "GPU device released with live queue users; preserving ownership");
      return;
    }
  }
  iree_slim_mutex_lock(&device->shutdown.mutex);
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    iree_hal_amd_gpu_pm4_umq_queue_close(device->queues[i]);
  }
  iree_atomic_store(&device->shutdown.closing, 1, iree_memory_order_release);
  iree_notification_post(&device->shutdown.notification, IREE_ALL_WAITERS);
  amdf_status_t status =
      device->context->gpu->event_signal(device->context->event);
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report(device->context, status,
                                    "GPU shutdown event_signal");
  }
  // The worker may reclaim device immediately after this final access.
  iree_slim_mutex_unlock(&device->shutdown.mutex);
}

iree_status_t iree_hal_amd_gpu_device_create(
    iree_hal_amd_gpu_context_t* context, iree_string_view_t display_name,
    const iree_hal_amd_gpu_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  *out_device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_create_params_verify_runtime_features(
      create_params, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE));
  if (options->queue_family_ordinal >=
          context->endpoint_info.queue_family_count ||
      !options->queue_count || options->queue_count > 256) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GPU provisioning requires an advertised family "
                            "and 1 to 256 queues");
  }
  iree_host_size_t total_size = sizeof(iree_hal_amd_gpu_device_t);
  if (!iree_host_size_checked_mul_add(total_size, options->queue_count,
                                      sizeof(iree_hal_queue_t*), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, options->queue_count + 1,
          sizeof(iree_hal_amd_gpu_completion_source_t), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, (iree_host_size_t)context->memory_scope_count + 1,
          sizeof(iree_hal_amd_fabric_memory_storage_t), &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU device provisioning storage overflows");
  }
  iree_hal_amd_gpu_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  iree_hal_resource_initialize(&iree_hal_amd_gpu_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  device->context = context;
  device->queues = (iree_hal_queue_t**)(device + 1);
  device->sources =
      (iree_hal_amd_gpu_completion_source_t*)(device->queues +
                                              options->queue_count);
  device->memory.storage =
      (iree_hal_amd_fabric_memory_storage_t*)(device->sources +
                                              options->queue_count + 1);
  iree_slim_mutex_initialize(&device->shutdown.mutex);
  iree_notification_initialize(&device->shutdown.notification);
  snprintf(device->processor_name, sizeof(device->processor_name), "gfx%u%u%x",
           context->target_info.gfx_ip.major, context->target_info.gfx_ip.minor,
           context->target_info.gfx_ip.stepping);
  iree_status_t status = iree_hal_amd_gpu_target_identity_parse_processor(
      iree_make_cstring_view(device->processor_name), &device->target);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_target_identity_resolve_physical_target(
        context->target_info.asic_revision, &device->target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_device_build_spec(device, display_name,
                                                options->queue_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_device_prepare_control(device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_acquire_for_node(
        create_params->proactor_pool, options->host_numa_node,
        &device->proactor_entry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_notification_create(
        iree_async_proactor_pool_entry_proactor(device->proactor_entry),
        IREE_ASYNC_NOTIFICATION_FLAG_NONE, &device->memory.notification);
  }
  iree_thread_affinity_t affinity = {0};
  if (options->host_numa_node != UINT32_MAX) {
    iree_thread_affinity_set_group_any(options->host_numa_node, &affinity);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_maintenance_thread_create(affinity, host_allocator,
                                                       &device->memory.service);
    if (iree_status_is_ok(status)) {
      iree_hal_amd_gpu_allocation_context_initialize(
          (iree_hal_device_t*)device, context, device->memory.service,
          host_allocator, &device->memory.allocations);
    }
  }
  iree_hal_amd_gpu_pm4_umq_queue_options_t queue_options = {
      .native = {.type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
                 .structure_size = sizeof(amdf_gpu_user_queue_create_info_t),
                 .queue_family_ordinal = options->queue_family_ordinal,
                 .priority = AMDF_QUEUE_PRIORITY_NORMAL,
                 .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                 .required_capabilities =
                     AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
                 .ring_byte_length = options->pm4.ring_byte_length},
      .memory = &device->memory.control,
      .service = device->memory.service,
      .allocations = &device->memory.allocations,
      .retirement_capacity = options->pm4.retirement_capacity,
      .argument_capacity = options->pm4.argument_capacity,
      .capture_block_size = options->pm4.capture_block_size,
      .capture_block_count = options->pm4.capture_block_count,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_pm4_barrier_profile_initialize(
        device->target.version, &queue_options.barriers);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_queue_family_initialize(
        (void*)device, 0,
        &iree_hal_device_spec_queues(device->spec)->families[0],
        &device->family);
  }
  iree_hal_queue_params_t queue_params;
  iree_hal_queue_params_initialize(&queue_params);
  for (uint32_t i = 0; i < options->queue_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_pm4_umq_queue_create(
        &device->family, context, &queue_options, &queue_params, host_allocator,
        &device->queues[i]);
    if (iree_status_is_ok(status)) {
      device->queue_count = i + 1;
      device->sources[i] =
          iree_hal_amd_gpu_pm4_umq_queue_completion_source(device->queues[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_device_initialize_memory(device,
                                              options->queue_family_ordinal);
    device->sources[device->queue_count] =
        (iree_hal_amd_gpu_completion_source_t){
            iree_hal_amd_gpu_device_poll_shutdown, device};
    // Startup is the final fallible step. Its exit callback cannot consume the
    // device until the caller's eventual last release completes the handoff.
    status = iree_hal_amd_gpu_completion_initialize(
        context, device->queue_count + 1, device->sources,
        (iree_hal_amd_gpu_completion_exit_callback_t){
            iree_hal_amd_gpu_device_completion_exit, device},
        affinity, host_allocator, &device->completion);
  }
  if (iree_status_is_ok(status)) {
    *out_device = (void*)device;
  } else if (iree_hal_amd_gpu_device_release_children(device)) {
    iree_allocator_free(host_allocator, device);
  }
  return status;
}

static iree_allocator_t iree_hal_amd_gpu_device_host_allocator(
    iree_hal_device_t* base) {
  return ((iree_hal_amd_gpu_device_t*)base)->host_allocator;
}

static iree_hal_allocator_t* iree_hal_amd_gpu_device_device_allocator(
    iree_hal_device_t* base) {
  return NULL;
}

static void iree_hal_amd_gpu_device_replace_channel_provider(
    iree_hal_device_t* base, iree_hal_channel_provider_t* provider) {}

static void iree_hal_amd_gpu_device_trim_caches(void* user_data) {
  iree_hal_amd_gpu_device_t* device = user_data;
  iree_arena_block_pool_trim(&device->memory.allocations.blocks);
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    iree_hal_amd_gpu_pm4_umq_queue_trim(device->queues[i]);
  }
}

static iree_status_t iree_hal_amd_gpu_device_trim(iree_hal_device_t* base) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  iree_hal_memory_maintenance_call(device->memory.service,
                                   iree_hal_amd_gpu_device_trim_caches, device);
  return iree_ok_status();
}

static const iree_hal_device_spec_t* iree_hal_amd_gpu_device_device_spec(
    iree_hal_device_t* base) {
  return ((iree_hal_amd_gpu_device_t*)base)->spec;
}

static const iree_hal_queue_family_t* iree_hal_amd_gpu_device_queue_family(
    iree_hal_device_t* base, iree_hal_queue_family_ordinal_t ordinal) {
  return ordinal == 0 ? &((iree_hal_amd_gpu_device_t*)base)->family : NULL;
}

static iree_hal_queue_t* iree_hal_amd_gpu_device_queue(
    iree_hal_device_t* base, iree_hal_queue_family_ordinal_t family,
    iree_hal_queue_ordinal_t ordinal) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  return family == 0 && ordinal < device->queue_count ? device->queues[ordinal]
                                                      : NULL;
}

static const iree_hal_device_topology_info_t*
iree_hal_amd_gpu_device_topology_info(iree_hal_device_t* base) {
  return &((iree_hal_amd_gpu_device_t*)base)->topology;
}

static const iree_hal_memory_backend_t* iree_hal_amd_gpu_device_memory_backend(
    iree_hal_device_t* base) {
  return &((iree_hal_amd_gpu_device_t*)base)->memory.backend.base;
}

static iree_status_t iree_hal_amd_gpu_device_refine_topology_edge(
    iree_hal_device_t* source, iree_hal_device_t* target,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_device_assign_topology_info(
    iree_hal_device_t* base, const iree_hal_device_topology_info_t* topology) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  iree_status_t status = iree_ok_status();
  uint32_t assigned_count = 0;
  for (uint32_t i = 0; i < device->queue_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_pm4_umq_queue_assign_frontier(
        device->queues[i], topology ? topology->frontier.tracker : NULL,
        topology ? topology->frontier.base_axis | ((uint64_t)i << 24) : 0);
    if (iree_status_is_ok(status)) {
      assigned_count = i + 1;
    }
  }
  if (iree_status_is_ok(status)) {
    device->topology =
        topology ? *topology : (iree_hal_device_topology_info_t){0};
  } else {
    for (uint32_t i = 0; i < assigned_count; ++i) {
      status = iree_status_join(status,
                                iree_hal_amd_gpu_pm4_umq_queue_assign_frontier(
                                    device->queues[i], NULL, 0));
    }
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_device_load_executable(
    iree_hal_device_t* base, const iree_hal_queue_family_t* family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_hal_executable_t** out_executable) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  const iree_hal_amd_gpu_executable_options_t options = {
      .target = &device->target,
      .native_family_ordinal = device->memory.family.queue_family_ordinal,
      .memory = &device->memory.control,
      .maintenance = device->memory.service,
  };
  return iree_hal_amd_gpu_executable_create(family, device->context, &options,
                                            params, device->host_allocator,
                                            out_executable);
}

static iree_status_t iree_hal_amd_gpu_device_create_semaphore(
    iree_hal_device_t* base, iree_hal_queue_family_affinity_t affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  return iree_hal_amd_gpu_pm4_semaphore_create(
      iree_async_proactor_pool_entry_proactor(device->proactor_entry),
      initial_value, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_amd_gpu_device_query_semaphore_compatibility(
    iree_hal_device_t* base, iree_hal_semaphore_t* semaphore) {
  // Concrete admission qualifies each producer's actual native signal domain.
  return iree_hal_amd_gpu_pm4_semaphore_isa(semaphore)
             ? IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL
             : IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_amd_gpu_device_acquire_queue(
    iree_hal_device_t* base, const iree_hal_queue_family_t* family,
    const iree_hal_queue_params_t* params, iree_hal_queue_t** out_queue) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "GPU queues are provisioned during device creation");
}

static iree_status_t iree_hal_amd_gpu_device_sample_observation(
    iree_hal_device_t* base, iree_hal_device_observation_flags_t flags,
    iree_hal_device_observation_t* out_observation) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "GPU device observations are unavailable");
}

static iree_status_t iree_hal_amd_gpu_device_create_channel(
    iree_hal_device_t* base, iree_hal_queue_family_affinity_t affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "GPU collective channels are unavailable");
}

static iree_status_t iree_hal_amd_gpu_device_create_command_buffer(
    iree_hal_device_t* base, const iree_hal_queue_family_t* family,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 command-buffer recording is unavailable");
}

static iree_status_t iree_hal_amd_gpu_device_import_file(
    iree_hal_device_t* base, iree_hal_pool_t* pool,
    iree_hal_queue_family_affinity_t affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_external_file_flags_t flags,
    iree_hal_file_t** out_file) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  return iree_hal_file_from_handle(
      pool, access, handle,
      iree_async_proactor_pool_entry_proactor(device->proactor_entry),
      device->host_allocator, out_file);
}

static iree_status_t iree_hal_amd_gpu_device_profiling_begin(
    iree_hal_device_t* base,
    const iree_hal_device_profiling_options_t* options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 queue profiling is unavailable");
}

static iree_status_t iree_hal_amd_gpu_device_profiling_flush(
    iree_hal_device_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_device_profiling_end(
    iree_hal_device_t* base) {
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_amd_gpu_device_vtable = {
    .destroy = iree_hal_amd_gpu_device_destroy,
    .id = iree_hal_amd_gpu_device_id,
    .host_allocator = iree_hal_amd_gpu_device_host_allocator,
    .device_allocator = iree_hal_amd_gpu_device_device_allocator,
    .replace_channel_provider =
        iree_hal_amd_gpu_device_replace_channel_provider,
    .trim = iree_hal_amd_gpu_device_trim,
    .device_spec = iree_hal_amd_gpu_device_device_spec,
    .queue_family = iree_hal_amd_gpu_device_queue_family,
    .queue = iree_hal_amd_gpu_device_queue,
    .acquire_queue = iree_hal_amd_gpu_device_acquire_queue,
    .sample_observation = iree_hal_amd_gpu_device_sample_observation,
    .topology_info = iree_hal_amd_gpu_device_topology_info,
    .memory_backend = iree_hal_amd_gpu_device_memory_backend,
    .refine_topology_edge = iree_hal_amd_gpu_device_refine_topology_edge,
    .assign_topology_info = iree_hal_amd_gpu_device_assign_topology_info,
    .create_channel = iree_hal_amd_gpu_device_create_channel,
    .create_command_buffer = iree_hal_amd_gpu_device_create_command_buffer,
    .load_executable = iree_hal_amd_gpu_device_load_executable,
    .import_file = iree_hal_amd_gpu_device_import_file,
    .create_semaphore = iree_hal_amd_gpu_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_amd_gpu_device_query_semaphore_compatibility,
    .profiling_begin = iree_hal_amd_gpu_device_profiling_begin,
    .profiling_flush = iree_hal_amd_gpu_device_profiling_flush,
    .profiling_end = iree_hal_amd_gpu_device_profiling_end,
};
