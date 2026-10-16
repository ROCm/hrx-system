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
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/queue/allocation.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"
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
  // Immutable queue sources, optional feedback, and a final shutdown observer.
  iree_hal_amd_gpu_completion_source_t* sources;
  // Resident command programs, shared builtins, and idle-instance trimming.
  iree_hal_amd_gpu_pm4_program_pool_t programs;
  // Sole native-event completion/retirement worker.
  iree_hal_amd_gpu_completion_t completion;
  // Optional shared shader reports, drained by the same completion worker.
  iree_hal_amd_gpu_feedback_t feedback;
  // Optional profiling session, owned until externally serialized idle end.
  iree_hal_amd_gpu_profile_t* profile;
  // Monotonic session identity for this device's lifetime.
  uint64_t next_profile_session_id;
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
    // Optional ASAN namespace; absent devices allocate no sanitizer metadata.
    iree_hal_amd_fabric_asan_t* asan;
    // Device-owned shadow records indexed by provisioned queue ordinal.
    iree_hal_amd_gpu_tsan_t* tsan;
    // Initialized or zeroed records requiring native cleanup.
    uint32_t tsan_count;
    // Persistent native access set used by internal control/image allocations.
    amdf_memory_device_access_t access;
    // Qualified coherent system backing for native control and placed images.
    iree_hal_amd_fabric_memory_options_t control;
    // Device access facts qualified by the control profile.
    struct {
      // Native permissions that cannot be withheld from GPU allocations.
      amdf_memory_access_t minimum_access;
      // Ordinary GPU address domain shared by backing profiles.
      uint32_t address_domain_ordinal;
    } capabilities;
  } memory;
  // Group assignment; each queue retains the tracker through final retirement.
  iree_hal_device_topology_info_t topology;
} iree_hal_amd_gpu_device_t;

static const iree_hal_device_vtable_t iree_hal_amd_gpu_device_vtable;

void iree_hal_amd_gpu_device_options_initialize(
    iree_hal_amd_gpu_device_options_t* out_options) {
  *out_options = (iree_hal_amd_gpu_device_options_t){
      .queue_family_ordinal = UINT32_MAX,
      .queue_count = 1,
      .host_numa_node = UINT32_MAX,
      .pm4 = {.ring_byte_length = 65536,
              .retirement_capacity = 256,
              .argument_capacity = 65536,
              .capture_block_size = 1024,
              .capture_block_count = 64},
  };
  iree_hal_amd_fabric_asan_options_initialize(&out_options->asan.memory);
  iree_hal_amd_gpu_tsan_options_initialize(&out_options->tsan.memory);
  out_options->tsan.report_policy =
      IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE;
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
    uint32_t queue_count,
    const iree_hal_amd_fabric_asan_options_t* asan_options) {
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
  if (iree_status_is_ok(status) && asan_options) {
    const iree_hal_device_sanitizer_spec_t sanitizer = {
        .flags = IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN,
        .asan.pool_options =
            {
                .mode = IREE_HAL_ASAN_POOL_MODE_SHADOW,
                .shadow_granule_size = UINT64_C(1)
                                       << asan_options->shadow_scale_shift,
                .redzone_size = asan_options->redzone_size,
                .quarantine_size = asan_options->quarantine_size,
            },
    };
    status = iree_hal_device_spec_builder_set_sanitizer(&builder, &sanitizer);
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
      device->memory.capabilities.minimum_access = access.guaranteed_access;
      device->memory.capabilities.address_domain_ordinal =
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
      .minimum_access = device->memory.capabilities.minimum_access,
      .binding_count = IREE_ARRAYSIZE(bindings),
      .bindings = bindings,
  };
  const iree_hal_amd_gpu_context_t* context = device->context;
  for (uint32_t i = 0; i < context->memory_scope_count; ++i) {
    device->memory.storage[i] = (iree_hal_amd_fabric_memory_storage_t){
        .scope = context->memory_scopes[i],
        .physical_device_affinity = 1,
        .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .minimum_access = device->memory.capabilities.minimum_access,
    };
  }
  device->memory.storage[context->memory_scope_count] =
      (iree_hal_amd_fabric_memory_storage_t){
          .scope = context->fabric->system_scope,
          .memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
          .minimum_access = device->memory.capabilities.minimum_access,
      };
  const iree_hal_amd_fabric_memory_backend_options_t options = {
      .api = context->api,
      .instance = context->fabric->instance,
      .device = context->device,
      .address_domain_ordinal =
          device->memory.capabilities.address_domain_ordinal,
      .storage_count = context->memory_scope_count + 1,
      .storage = device->memory.storage,
      .event_sink = context->event_sink,
      .families = &device->memory.family,
      .asan = device->memory.asan,
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
    if (!iree_hal_amd_gpu_pm4_queue_is_idle(device->queues[i])) {
      return false;
    }
  }
  return true;
}

static void iree_hal_amd_gpu_device_release_programs(void* user_data) {
  iree_hal_amd_gpu_device_t* device = user_data;
  iree_hal_amd_gpu_pm4_program_pool_deinitialize(&device->programs);
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
        iree_hal_amd_gpu_pm4_queue_shutdown(device->queues[i]);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(device->context, status,
                                      "PM4 queue native ownership join");
      return false;
    }
    iree_hal_queue_release(device->queues[i]);
    device->queues[i] = NULL;
  }
  if (device->memory.service) {
    iree_hal_memory_maintenance_call(device->memory.service,
                                     iree_hal_amd_gpu_device_release_programs,
                                     device);
    iree_hal_amd_gpu_allocation_context_deinitialize(
        &device->memory.allocations);
  }
  iree_hal_memory_maintenance_release(device->memory.service);
  iree_status_t profile_status = iree_hal_amd_gpu_profile_end(device->profile);
  device->profile = NULL;
  if (!iree_status_is_ok(profile_status)) {
    iree_hal_amd_gpu_context_report_status(
        device->context, (iree_hal_device_t*)device, profile_status);
    iree_status_free(profile_status);
  }
  // Final executable cleanup on service poisons its globals before shadow
  // teardown.
  for (uint32_t i = 0; i < device->memory.tsan_count; ++i) {
    iree_hal_amd_gpu_tsan_deinitialize(device->context,
                                       &device->memory.tsan[i]);
  }
  iree_allocator_free(device->host_allocator, device->memory.tsan);
  if (device->memory.asan) {
    if (!iree_hal_amd_fabric_asan_deinitialize(device->memory.asan)) {
      device->context->cleanup_blocked = true;
      return false;
    }
    iree_allocator_free(device->host_allocator, device->memory.asan);
  }

  if (!iree_hal_amd_gpu_feedback_deinitialize(&device->feedback)) {
    return false;
  }
  amdf_status_t completion_status =
      iree_hal_amd_gpu_context_deinitialize_completion(device->context);
  if (!amdf_status_is_ok(completion_status)) {
    device->context->cleanup_blocked = true;
    iree_hal_amd_gpu_context_report(device->context, completion_status,
                                    "completion cleanup");
    return false;
  }
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
    if (device->feedback.context) {
      // All producers have retired. An interrupted reservation cannot become
      // ready now; consume the final prefix without waiting for it.
      iree_hal_amd_gpu_feedback_drain(&device->feedback);
    }
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
    iree_hal_amd_gpu_pm4_queue_close(device->queues[i]);
  }
  iree_atomic_store(&device->shutdown.closing, 1, iree_memory_order_release);
  iree_notification_post(&device->shutdown.notification, IREE_ALL_WAITERS);
  amdf_status_t status = iree_hal_amd_gpu_context_signal(device->context);
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report(device->context, status,
                                    "GPU shutdown event_signal");
  }
  // The worker may reclaim device immediately after this final access.
  iree_slim_mutex_unlock(&device->shutdown.mutex);
}

// Queue protocol and publication are native family properties, independent of
// the executable target. Prefer direct publication when this family exposes it.
static amdf_queue_publication_modes_t
iree_hal_amd_gpu_device_family_publication(
    const amdf_queue_family_info_t* family) {
  const amdf_queue_publication_modes_t modes =
      iree_hal_amd_gpu_pm4_queue_publication_modes(family);
  return (modes & AMDF_QUEUE_PUBLICATION_MODE_USER)
             ? AMDF_QUEUE_PUBLICATION_MODE_USER
             : modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
}

typedef struct iree_hal_amd_gpu_device_program_prepare_t {
  // Unpublished device whose construction waits for this service call.
  iree_hal_amd_gpu_device_t* device;
  // Immutable native queue count used by program instance storage.
  uint32_t queue_count;
  // Selected native transport needs terminal chain continuations.
  bool chain;
  // Native preparation result returned to device construction.
  iree_status_t status;
} iree_hal_amd_gpu_device_program_prepare_t;

static void iree_hal_amd_gpu_device_prepare_programs(void* user_data) {
  iree_hal_amd_gpu_device_program_prepare_t* prepare = user_data;
  iree_hal_amd_gpu_device_t* device = prepare->device;
  prepare->status = iree_hal_amd_gpu_pm4_program_pool_initialize(
      device->context, device->memory.service, prepare->queue_count,
      prepare->chain, device->host_allocator, &device->programs);
}

typedef struct iree_hal_amd_gpu_device_feedback_prepare_t {
  // Unpublished device borrowed until the service call joins.
  iree_hal_amd_gpu_device_t* device;
  // Borrowed cold options, consumed before the service call joins.
  const iree_hal_amd_gpu_device_options_t* options;
  // Requested address-sanitizer provisioning.
  bool asan_enabled;
  // Requested queue-local race-sanitizer provisioning.
  bool tsan_enabled;
  // Construction result transferred back to the creating thread.
  iree_status_t status;
} iree_hal_amd_gpu_device_feedback_prepare_t;

static void iree_hal_amd_gpu_device_notify_failure(void* user_data) {
  iree_hal_amd_gpu_device_t* device = user_data;
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    if (device->queues[i]) {
      iree_hal_amd_gpu_pm4_queue_notify_failure(device->queues[i]);
    }
  }
  const amdf_status_t status = iree_hal_amd_gpu_context_signal(device->context);
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report(device->context, status,
                                    "GPU sanitizer failure wake");
  }
}

static void iree_hal_amd_gpu_device_prepare_feedback(void* user_data) {
  iree_hal_amd_gpu_device_feedback_prepare_t* prepare = user_data;
  iree_hal_amd_gpu_device_t* device = prepare->device;
  prepare->status = iree_hal_amd_gpu_feedback_initialize(
      device->context, (iree_hal_device_t*)device, &device->memory.control,
      device->memory.capabilities.address_domain_ordinal,
      device->host_allocator, &device->feedback);
  device->feedback.failure_callback = iree_hal_amd_gpu_device_notify_failure;
  device->feedback.failure_user_data = device;
  const iree_hal_amd_gpu_device_options_t* options = prepare->options;
  if (iree_status_is_ok(prepare->status) && prepare->asan_enabled) {
    amdf_memory_scope_t* scope = device->context->fabric->system_scope;
    if (options->asan.shadow_backing ==
            IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL &&
        device->context->memory_scope_count) {
      scope = device->context->memory_scopes[0];
    }
    prepare->status = iree_allocator_malloc(device->host_allocator,
                                            sizeof(*device->memory.asan),
                                            (void**)&device->memory.asan);
    if (iree_status_is_ok(prepare->status)) {
      prepare->status = iree_hal_amd_fabric_asan_initialize(
          device->context->api, device->context->fabric->instance,
          device->context->device,
          device->memory.capabilities.address_domain_ordinal, scope,
          &options->asan.memory, device->context->event_sink,
          device->host_allocator, device->memory.asan);
      device->memory.asan->failure_callback = iree_hal_amd_gpu_feedback_fail;
      device->memory.asan->failure_user_data = &device->feedback;
      device->feedback.asan_report_policy = options->asan.report_policy;
    }
  }
  if (iree_status_is_ok(prepare->status) && prepare->tsan_enabled) {
    prepare->status = iree_allocator_malloc_array(
        device->host_allocator, options->queue_count,
        sizeof(*device->memory.tsan), (void**)&device->memory.tsan);
    if (iree_status_is_ok(prepare->status)) {
      device->memory.tsan_count = options->queue_count;
      device->feedback.tsan_report_policy = options->tsan.report_policy;
    }
    for (uint32_t i = 0;
         i < device->memory.tsan_count && iree_status_is_ok(prepare->status);
         ++i) {
      prepare->status = iree_hal_amd_gpu_tsan_initialize(
          device->context, &options->tsan.memory, i, &device->memory.tsan[i]);
    }
  }
}

iree_status_t iree_hal_amd_gpu_device_create(
    iree_hal_amd_gpu_context_t* context, iree_string_view_t display_name,
    const iree_hal_amd_gpu_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  *out_device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_create_params_verify_runtime_features(
      create_params, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK |
                         IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN |
                         IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN));
  const bool asan_enabled =
      iree_any_bit_set(create_params->runtime_features,
                       IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN);
  if (asan_enabled &&
      (options->asan.memory.shadow_scale_shift > 7 ||
       (uint32_t)options->asan.shadow_backing >
           IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_HOST_LOCAL ||
       (uint32_t)options->asan.report_policy >
           IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid ASAN device provisioning options");
  }
  const bool tsan_enabled =
      iree_any_bit_set(create_params->runtime_features,
                       IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN);
  if (tsan_enabled &&
      (uint32_t)options->tsan.report_policy >
          IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid TSAN report policy");
  }
  const bool feedback_enabled =
      asan_enabled || tsan_enabled ||
      iree_any_bit_set(create_params->runtime_features,
                       IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK);
  uint32_t family_ordinal = options->queue_family_ordinal;
  if (family_ordinal == UINT32_MAX) {
    for (uint32_t i = 0; i < context->endpoint_info.queue_family_count; ++i) {
      if (iree_hal_amd_gpu_device_family_publication(&context->families[i])) {
        family_ordinal = i;
        break;
      }
    }
    if (family_ordinal == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "GPU has no supported PM4 family");
    }
  }
  if (family_ordinal >= context->endpoint_info.queue_family_count ||
      !options->queue_count || options->queue_count > 256) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GPU provisioning requires an advertised family "
                            "and 1 to 256 queues");
  }
  const amdf_queue_publication_modes_t publication_mode =
      iree_hal_amd_gpu_device_family_publication(
          &context->families[family_ordinal]);
  if (!publication_mode) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "GPU family has no supported PM4 publication");
  }
  iree_host_size_t total_size = sizeof(iree_hal_amd_gpu_device_t);
  const uint32_t source_count = options->queue_count + 1 + feedback_enabled;
  if (!iree_host_size_checked_mul_add(total_size, options->queue_count,
                                      sizeof(iree_hal_queue_t*), &total_size) ||
      !iree_host_size_checked_mul_add(
          total_size, source_count,
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
      (iree_hal_amd_fabric_memory_storage_t*)(device->sources + source_count);
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
    status = iree_hal_amd_gpu_device_build_spec(
        device, display_name, options->queue_count,
        asan_enabled ? &options->asan.memory : NULL);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amd_gpu_context_prepare_completion(context, publication_mode);
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
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_device_program_prepare_t prepare = {
        .device = device,
        .queue_count = options->queue_count,
        .chain = publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
    };
    iree_hal_memory_maintenance_call(device->memory.service,
                                     iree_hal_amd_gpu_device_prepare_programs,
                                     &prepare);
    status = prepare.status;
  }
  if (iree_status_is_ok(status) && feedback_enabled) {
    iree_hal_amd_gpu_device_feedback_prepare_t prepare = {
        .device = device,
        .options = options,
        .asan_enabled = asan_enabled,
        .tsan_enabled = tsan_enabled};
    iree_hal_memory_maintenance_call(device->memory.service,
                                     iree_hal_amd_gpu_device_prepare_feedback,
                                     &prepare);
    status = prepare.status;
    device->memory.allocations.failure = &device->feedback.failure_code;
  }
  iree_hal_amd_gpu_pm4_queue_options_t queue_options = {
      .family_ordinal = family_ordinal,
      .publication_mode = publication_mode,
      .command_byte_capacity = options->pm4.ring_byte_length,
      .memory = &device->memory.control,
      .service = device->memory.service,
      .allocations = &device->memory.allocations,
      .feedback = feedback_enabled ? &device->feedback : NULL,
      .tsan_shadow_slot_count =
          tsan_enabled ? options->tsan.memory.shadow_slot_count : 0,
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
    queue_options.queue_ordinal = i;
    status = iree_hal_amd_gpu_pm4_queue_create(
        &device->family, context, &queue_options, &queue_params, host_allocator,
        &device->queues[i]);
    if (iree_status_is_ok(status)) {
      device->queue_count = i + 1;
      device->sources[i] =
          iree_hal_amd_gpu_pm4_queue_completion_source(device->queues[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_device_initialize_memory(device, family_ordinal);
    // Queue retirement is observed before the final feedback drain so shutdown
    // cannot pass a report published between a drain and native completion.
    if (feedback_enabled) {
      device->sources[device->queue_count] =
          (iree_hal_amd_gpu_completion_source_t){iree_hal_amd_gpu_feedback_poll,
                                                 &device->feedback};
    }
    device->sources[source_count - 1] = (iree_hal_amd_gpu_completion_source_t){
        iree_hal_amd_gpu_device_poll_shutdown, device};
    // Startup is the final fallible step. Its exit callback cannot consume the
    // device until the caller's eventual last release completes the handoff.
    status = iree_hal_amd_gpu_completion_initialize(
        context, source_count, device->sources,
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
  iree_hal_amd_gpu_pm4_program_pool_trim(&device->programs);
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    iree_hal_amd_gpu_pm4_queue_trim(device->queues[i]);
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
    status = iree_hal_amd_gpu_pm4_queue_assign_frontier(
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
      status =
          iree_status_join(status, iree_hal_amd_gpu_pm4_queue_assign_frontier(
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
      .feedback = device->feedback.context ? &device->feedback : NULL,
      .asan = device->memory.asan,
      .tsan_count = device->memory.tsan_count,
      .tsan = device->memory.tsan,
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
  iree_hal_amd_gpu_device_t* device = (void*)base;
  if (iree_any_bit_set(flags, IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_observation_populate_memory_total_from_spec(
            device->spec, out_observation));
  }
  if (iree_any_bit_set(flags, IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER) &&
      device->memory.asan) {
    out_observation->provided_flags |=
        IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER;
    iree_hal_amd_fabric_asan_query_statistics(device->memory.asan,
                                              &out_observation->sanitizer.asan);
  }
  return iree_ok_status();
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
  iree_hal_amd_gpu_device_t* device = (void*)base;
  iree_hal_amd_gpu_pm4_barrier_profile_t profile;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_barrier_profile_initialize(
      device->target.version, &profile));
  return iree_hal_amd_gpu_pm4_command_buffer_create(
      family, &device->programs, mode, categories, binding_capacity,
      (uint32_t)
          device->context->target_info.compute.local_data_share_byte_length,
      device->memory.tsan_count
          ? device->memory.tsan[0].config.shadow_slot_count
          : 0,
      &profile, device->host_allocator, out_command_buffer);
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
  iree_hal_amd_gpu_device_t* device = (void*)base;
  if (device->profile) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "GPU profiling session is already active");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_begin(
      options, ++device->next_profile_session_id, device->queue_count,
      device->host_allocator, &device->profile));
  device->memory.allocations.profile = device->profile;
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    iree_hal_amd_gpu_pm4_queue_set_profile(device->queues[i], device->profile);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_device_profiling_flush(
    iree_hal_device_t* base) {
  return iree_hal_amd_gpu_profile_flush(
      ((iree_hal_amd_gpu_device_t*)base)->profile);
}

static iree_status_t iree_hal_amd_gpu_device_profiling_end(
    iree_hal_device_t* base) {
  iree_hal_amd_gpu_device_t* device = (void*)base;
  iree_hal_amd_gpu_profile_t* profile = device->profile;
  device->profile = NULL;
  device->memory.allocations.profile = NULL;
  for (uint32_t i = 0; i < device->queue_count; ++i) {
    iree_hal_amd_gpu_pm4_queue_set_profile(device->queues[i], NULL);
  }
  return iree_hal_amd_gpu_profile_end(profile);
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
