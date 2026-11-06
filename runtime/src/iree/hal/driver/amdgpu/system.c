// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/system.h"

#include "iree/hal/driver/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_system_query_gpu_agent_targets(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_agent_target_t* out_targets) {
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_agent_target_query(
        libhsa, topology->gpu_agents[i], host_allocator, &out_targets[i]));
  }

  for (iree_host_size_t i = 1; i < topology->gpu_agent_count; ++i) {
    if (out_targets[i].isa_count != out_targets[0].isa_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "runtime currently requires homogeneous GPU targets; agent[%" PRIhsz
          "] reports %" PRIhsz " ISAs but agent[0] reports %" PRIhsz,
          i, out_targets[i].isa_count, out_targets[0].isa_count);
    }
    for (iree_host_size_t j = 0; j < out_targets[0].isa_count; ++j) {
      const iree_hal_amdgpu_agent_isa_target_t* expected_isa =
          iree_hal_amdgpu_agent_target_isa_at(&out_targets[0], j);
      const iree_hal_amdgpu_agent_isa_target_t* actual_isa =
          iree_hal_amdgpu_agent_target_isa_at(&out_targets[i], j);
      if (iree_hal_amdgpu_target_identity_equal(&expected_isa->identity,
                                                &actual_isa->identity)) {
        continue;
      }
      char expected_target[128] = {0};
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_format_artifact_key(
          &expected_isa->identity, sizeof(expected_target), expected_target,
          /*out_buffer_length=*/NULL));
      char actual_target[128] = {0};
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_format_artifact_key(
          &actual_isa->identity, sizeof(actual_target), actual_target,
          /*out_buffer_length=*/NULL));
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "runtime currently requires homogeneous GPU targets; agent[%" PRIhsz
          "] ISA[%" PRIhsz "] target `%s` does not match agent[0] ISA[%" PRIhsz
          "] target `%s`",
          i, j, actual_target, j, expected_target);
    }
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_system_deinitialize(
    iree_hal_amdgpu_system_t* system);

static iree_status_t iree_hal_amdgpu_system_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_system_options_t options, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t* out_system) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_system);
  IREE_TRACE_ZONE_BEGIN(z0);

  out_system->host_allocator = host_allocator;

  // Query and validate the system information.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_system_info_query(libhsa, &out_system->info));

  // Initialize the platform source used for profile clock correlation.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_device_clock_source_initialize(
              &out_system->device_clock_source));

  // Copy the libhsa symbol table and retain HSA for the lifetime of the system.
  // The caller may destroy the provided libhsa after this call returns.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_libhsa_copy(libhsa, &out_system->libhsa));

  // Copy the topology - today it's a plain-old-data struct and we can just
  // memcpy it. This is an implementation detail, though, and in the future if
  // it allocates anything we'll need to make sure this retains the allocations
  // or does a deep copy.
  memcpy(&out_system->topology, topology, sizeof(out_system->topology));

  // Preserved as the information may be used by components of the system.
  out_system->options = options;

  // Query physical target identity once for each GPU agent. All later
  // consumers share these immutable records instead of independently
  // rediscovering ISA features and ASIC revision.
  iree_status_t status = iree_hal_amdgpu_system_query_gpu_agent_targets(
      &out_system->libhsa, &out_system->topology, out_system->host_allocator,
      out_system->gpu_agent_targets);

  // Initialize the device library, which will load the builtin executable and
  // fail if we don't have a supported arch.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_library_initialize(
        &out_system->libhsa, &out_system->topology,
        out_system->gpu_agent_targets, host_allocator,
        &out_system->device_library);
  }

  // Find common/shared memory pools.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t host_ordinal = 0;
         iree_status_is_ok(status) && host_ordinal < topology->cpu_agent_count;
         ++host_ordinal) {
      status = iree_hal_amdgpu_host_memory_pools_query(
          &out_system->libhsa, topology->cpu_agents[host_ordinal],
          &out_system->host_memory_pools[host_ordinal]);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_system_deinitialize(out_system);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_system_deinitialize(
    iree_hal_amdgpu_system_t* system) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unload the device library - no references to it should remain.
  iree_hal_amdgpu_device_library_deinitialize(&system->device_library);

  for (iree_host_size_t i = 0; i < system->topology.gpu_agent_count; ++i) {
    iree_hal_amdgpu_agent_target_deinitialize(&system->gpu_agent_targets[i]);
  }

  // Release platform clock-sampling state before unloading HSA.
  iree_hal_amdgpu_device_clock_source_deinitialize(
      &system->device_clock_source);

  // This may unload HSA if we were the last retainer in the process.
  iree_hal_amdgpu_libhsa_deinitialize(&system->libhsa);

  memset(system, 0, sizeof(*system));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_system_allocate(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_system_options_t options, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t** out_system) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_system);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_system = NULL;

  iree_hal_amdgpu_system_t* system = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(
              host_allocator,
              sizeof(*system) + topology->cpu_agent_count *
                                    sizeof(system->host_memory_pools[0]),
              (void**)&system));

  iree_status_t status = iree_hal_amdgpu_system_initialize(
      libhsa, topology, options, host_allocator, system);

  if (iree_status_is_ok(status)) {
    *out_system = system;
  } else {
    iree_allocator_free(host_allocator, system);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_system_free(iree_hal_amdgpu_system_t* system) {
  if (!system) {
    return;
  }
  iree_allocator_t host_allocator = system->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_system_deinitialize(system);

  iree_allocator_free(host_allocator, system);

  IREE_TRACE_ZONE_END(z0);
}
