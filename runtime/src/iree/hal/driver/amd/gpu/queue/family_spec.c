// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/family_spec.h"

#include "iree/hal/driver/amd/fabric/memory.h"
#include "iree/hal/driver/amd/gpu/dispatch.h"
#include "iree/hal/driver/amd/status.h"

iree_status_t iree_hal_amd_gpu_queue_family_spec_initialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_gfxip_version_t target_version, uint32_t family_ordinal,
    amdf_queue_publication_modes_t publication_mode, uint32_t queue_count,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_queue_family_spec_storage_t* out_storage,
    iree_hal_queue_family_spec_t* out_spec) {
  *out_storage = (iree_hal_amd_gpu_queue_family_spec_storage_t){0};
  *out_spec = (iree_hal_queue_family_spec_t){0};
  const amdf_queue_family_info_t* native_family =
      &context->families[family_ordinal];
  const bool compute =
      native_family->command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA;
  amdf_observation_clock_info_t clock = {0};
  if (context->gpu->extension_version >= AMDF_GPU_EXTENSION_VERSION_2) {
    const amdf_status_t status = context->gpu->device_query_queue_clock_info(
        context->device, family_ordinal, &clock);
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
          status, "device_query_queue_clock_info"));
    }
  }
  // Runtime-owned scalar shaders implement the complete semantic family. Each
  // pool intersects this execution support with its exact backing guarantees.
  const iree_hal_atomic_operation_capabilities_t atomic_operations = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
      .device_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
      .system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
  };
  const uint32_t wait_conditions_32 = IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  const uint32_t wait_conditions_64 = IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  amdf_queue_family_placement_info_t native_placement = {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_PLACEMENT_INFO,
      .structure_size = sizeof(native_placement),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->device_query_queue_family_placement_info(
          context->device, family_ordinal, &native_placement),
      "device_query_queue_family_placement_info"));
  // Family support describes packet encodings. Each pool separately qualifies
  // these operations against its exact backing and requested access scopes.
  const amdf_atomic_capabilities_t* native_atomics =
      &native_family->atomic_capabilities;
  const iree_hal_atomic_operation_capabilities_t zero_compute_operations =
      iree_hal_amd_fabric_memory_translate_atomics(
          native_atomics->operations_without_dispatch_32,
          native_atomics->operations_without_dispatch_64,
          (amdf_atomic_reach_t){.scope_32 = AMDF_ATOMIC_SCOPE_SYSTEM,
                                .scope_64 = AMDF_ATOMIC_SCOPE_SYSTEM});
  static_assert(
      (uint32_t)AMDF_ATOMIC_WAIT_CONDITION_EQUAL ==
              (uint32_t)IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_EQUAL &&
          (uint32_t)AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL ==
              (uint32_t)IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NOT_EQUAL &&
          (uint32_t)AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL ==
              (uint32_t)
                  IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_UNSIGNED_GREATER_EQUAL,
      "native and HAL wait predicate bits must match");
  const uint32_t native_wait_conditions_32 =
      (zero_compute_operations.device_scope_32 &
       IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT)
          ? native_atomics->wait_conditions_32 &
                IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL
          : 0;
  const uint32_t native_wait_conditions_64 =
      (zero_compute_operations.device_scope_64 &
       IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT)
          ? native_atomics->wait_conditions_64 &
                IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL
          : 0;
  const bool user_publication =
      publication_mode == AMDF_QUEUE_PUBLICATION_MODE_USER;
  // KERNEL creation has no priority selector. USER creation supplies the
  // independently qualified family priority levels.
  const amdf_queue_priority_capabilities_t priority_capabilities =
      user_publication ? native_family->priority_capabilities
                       : AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL;
  iree_hal_queue_priority_t* priorities = out_storage->priorities;
  iree_host_size_t priority_count = 0;
  if (priority_capabilities & AMDF_QUEUE_PRIORITY_CAPABILITY_LOW) {
    priorities[priority_count++] = -1;
  }
  if (priority_capabilities & AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL) {
    priorities[priority_count++] = IREE_HAL_QUEUE_PRIORITY_NORMAL;
  }
  if (priority_capabilities & AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH) {
    priorities[priority_count++] = 1;
  }
  iree_hal_queue_placement_capabilities_t placements = 0;
  if (user_publication) {
    if (native_placement.placement_capabilities &
        AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL) {
      placements |= IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL;
    }
    if (native_placement.placement_capabilities &
        AMDF_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL) {
      placements |= IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL;
    }
  }
  // Families without exact native selection expose one indivisible resource.
  // Selectable families expand the cached logical geometry before the builder
  // copies the queue specification.
  out_storage->resource_group =
      (iree_hal_queue_execution_resource_group_spec_t){
          .minimum_selected_resource_count = 1};
  out_storage->execution_resource = (iree_hal_queue_execution_resource_spec_t){
      .group_ordinal = 0,
      .first_execution_unit_ordinal = 0,
      .execution_unit_count = context->target_info.compute.compute_unit_count};
  iree_hal_queue_family_spec_t family = {
      .name = !compute ? IREE_SV("sdma")
              : native_family->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                  ? IREE_SV("aql")
                  : IREE_SV("pm4"),
      .flags =
          IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION |
          (clock.frequency_hz
               ? IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS |
                     IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS
               : 0),
      // AQL's GWS initializer still shares its ABI with mutable grid state.
      // PM4 cooperative state has an independent resident owner.
      .supported_queue_features =
          native_family->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
                  user_publication &&
                  (native_family->user_queue_capabilities &
                   AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH) &&
                  iree_hal_amd_gpu_grid_sync_select_strategy(target_version) !=
                      IREE_HAL_AMD_GPU_GRID_SYNC_STRATEGY_NONE
              ? IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH
              : 0,
      .provisioned_queue_count = queue_count,
      .execution_unit_count =
          compute ? out_storage->execution_resource.execution_unit_count : 0,
      .execution_resource_group_count = compute ? 1 : 0,
      .execution_resource_groups =
          compute ? &out_storage->resource_group : NULL,
      .execution_resource_count = compute ? 1 : 0,
      .execution_resources = compute ? &out_storage->execution_resource : NULL,
      .placement_capabilities = placements,
      // The native queue clock query supplies this family's rate. A zero rate
      // leaves timestamp capture unavailable.
      .timestamp_valid_bits = clock.frequency_hz ? 64 : 0,
      .timestamp_frequency_hz = clock.frequency_hz,
      .priority_count = priority_count,
      .priorities = priorities,
      .physical_device_affinity = 1,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC,
      .atomic_capabilities =
          {.operations = atomic_operations,
           .wait_conditions = {.device_scope_32 = wait_conditions_32,
                               .device_scope_64 = wait_conditions_64,
                               .system_scope_32 = wait_conditions_32,
                               .system_scope_64 = wait_conditions_64}},
      .zero_compute_atomic_capabilities =
          {.operations = zero_compute_operations,
           .wait_conditions = {.device_scope_32 = native_wait_conditions_32,
                               .device_scope_64 = native_wait_conditions_64,
                               .system_scope_32 = native_wait_conditions_32,
                               .system_scope_64 = native_wait_conditions_64}},
  };
  static_assert(
      (uint32_t)AMDF_QUEUE_PLACEMENT_COUNT ==
              (uint32_t)IREE_HAL_QUEUE_PLACEMENT_COUNT &&
          (uint32_t)AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN ==
              (uint32_t)IREE_HAL_QUEUE_HOST_PUBLICATION_FLAG_KNOWN &&
          (uint32_t)AMDF_QUEUE_HOST_PUBLICATION_FLAG_DEVICE_READ ==
              (uint32_t)IREE_HAL_QUEUE_HOST_PUBLICATION_FLAG_DEVICE_READ &&
          (uint32_t)AMDF_QUEUE_HOST_PUBLICATION_FLAG_KERNEL_CALL ==
              (uint32_t)IREE_HAL_QUEUE_HOST_PUBLICATION_FLAG_KERNEL_CALL,
      "native and HAL publication requirements agree");
  if (placements) {
    memcpy(family.host_publication, native_placement.host_publication,
           sizeof(family.host_publication));
  }
  if (!compute) {
    family.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                        IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL;
    family.atomic_capabilities = (iree_hal_atomic_capabilities_t){0};
    family.zero_compute_atomic_capabilities =
        (iree_hal_atomic_capabilities_t){0};
  }
  iree_status_t status = iree_ok_status();
  const amdf_gpu_endpoint_info_t* native = &context->target_info;
  iree_hal_queue_execution_resource_group_spec_t* resource_groups = NULL;
  if (compute && user_publication &&
      (native_family->user_queue_capabilities &
       AMDF_USER_QUEUE_CAPABILITY_EXECUTION_RESOURCE_SELECTION)) {
    const amdf_gpu_execution_resource_geometry_t* geometry =
        &native->execution_resources;
    const uint32_t resource_count = native->compute.compute_unit_count /
                                    geometry->compute_units_per_resource;
    status = iree_allocator_malloc(
        host_allocator,
        geometry->group_count * sizeof(*resource_groups) +
            resource_count * sizeof(iree_hal_queue_execution_resource_spec_t),
        (void**)&resource_groups);
    if (iree_status_is_ok(status)) {
      iree_hal_queue_execution_resource_spec_t* resources =
          (void*)(resource_groups + geometry->group_count);
      for (uint32_t i = 0; i < geometry->group_count; ++i) {
        resource_groups[i].minimum_selected_resource_count =
            geometry->minimum_selected_resources_per_group;
      }
      for (uint32_t i = 0; i < resource_count; ++i) {
        resources[i] = (iree_hal_queue_execution_resource_spec_t){
            .group_ordinal = i % geometry->group_count,
            .first_execution_unit_ordinal =
                i * geometry->compute_units_per_resource,
            .execution_unit_count = geometry->compute_units_per_resource};
      }
      out_storage->resource_groups = resource_groups;
      family.execution_resource_group_count = geometry->group_count;
      family.execution_resource_groups = resource_groups;
      family.execution_resource_count = resource_count;
      family.execution_resources = resources;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_spec = family;
  }
  return status;
}

void iree_hal_amd_gpu_queue_family_spec_deinitialize(
    iree_hal_amd_gpu_queue_family_spec_storage_t* storage,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, storage->resource_groups);
}
