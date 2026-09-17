// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/access_policy.h"

static bool iree_hal_amdgpu_access_agent_list_contains(
    const iree_hal_amdgpu_access_agent_list_t* agent_list, hsa_agent_t agent) {
  for (uint32_t i = 0; i < agent_list->count; ++i) {
    if (agent_list->values[i].handle == agent.handle) return true;
  }
  return false;
}

static iree_status_t iree_hal_amdgpu_access_agent_list_append_unique(
    iree_hal_amdgpu_access_agent_list_t* agent_list, hsa_agent_t agent) {
  if (iree_hal_amdgpu_access_agent_list_contains(agent_list, agent)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(agent_list->count >= IREE_ARRAYSIZE(agent_list->values))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU access agent list capacity exceeded");
  }
  agent_list->values[agent_list->count++] = agent;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_access_agent_list_select_families(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_amdgpu_gpu_agent_mask_t* out_gpu_agent_mask) {
  if (IREE_UNLIKELY(topology->gpu_agent_count > IREE_HAL_MAX_QUEUE_FAMILIES)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU topology GPU agent count %" PRIhsz
                            " exceeds queue-family affinity capacity %" PRIhsz,
                            topology->gpu_agent_count,
                            (iree_host_size_t)IREE_HAL_MAX_QUEUE_FAMILIES);
  }
  const iree_hal_queue_family_affinity_t supported_affinity =
      topology->gpu_agent_count == IREE_HAL_MAX_QUEUE_FAMILIES
          ? IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY
          : (((iree_hal_queue_family_affinity_t)1
              << topology->gpu_agent_count) -
             1);
  if (iree_hal_queue_family_affinity_is_any(queue_family_affinity)) {
    queue_family_affinity = supported_affinity;
  }
  if (IREE_UNLIKELY(
          iree_hal_queue_family_affinity_is_empty(queue_family_affinity) ||
          !iree_all_bits_set(supported_affinity, queue_family_affinity))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU queue family affinity 0x%016" PRIx64
        " is not a non-empty subset of supported families 0x%016" PRIx64,
        queue_family_affinity, supported_affinity);
  }
  *out_gpu_agent_mask = (iree_hal_amdgpu_gpu_agent_mask_t)queue_family_affinity;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_access_agent_list_resolve_memory_agents(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_agent_list);
  memset(out_agent_list, 0, sizeof(*out_agent_list));

  iree_hal_amdgpu_gpu_agent_mask_t gpu_agent_mask = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_select_families(
      topology, queue_family_affinity, &gpu_agent_mask));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t physical_device_ordinal = 0;
       physical_device_ordinal < topology->gpu_agent_count &&
       iree_status_is_ok(status);
       ++physical_device_ordinal) {
    if (!iree_all_bits_set(gpu_agent_mask, ((uint64_t)1)
                                               << physical_device_ordinal)) {
      continue;
    }
    const iree_host_size_t cpu_agent_ordinal =
        topology->gpu_cpu_map[physical_device_ordinal];
    if (IREE_UNLIKELY(cpu_agent_ordinal >= topology->cpu_agent_count)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "AMDGPU topology maps GPU agent ordinal %" PRIhsz
                           " to invalid CPU agent ordinal %" PRIhsz,
                           physical_device_ordinal, cpu_agent_ordinal);
      break;
    }

    status = iree_hal_amdgpu_access_agent_list_append_unique(
        out_agent_list, topology->gpu_agents[physical_device_ordinal]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_access_agent_list_append_unique(
          out_agent_list, topology->cpu_agents[cpu_agent_ordinal]);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_access_agent_list_resolve_queue_family_agents(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_agent_list);
  memset(out_agent_list, 0, sizeof(*out_agent_list));

  iree_hal_amdgpu_gpu_agent_mask_t gpu_agent_mask = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_select_families(
      topology, queue_family_affinity, &gpu_agent_mask));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t physical_device_ordinal = 0;
       physical_device_ordinal < topology->gpu_agent_count &&
       iree_status_is_ok(status);
       ++physical_device_ordinal) {
    if (!iree_all_bits_set(gpu_agent_mask, ((uint64_t)1)
                                               << physical_device_ordinal)) {
      continue;
    }
    status = iree_hal_amdgpu_access_agent_list_append_unique(
        out_agent_list, topology->gpu_agents[physical_device_ordinal]);
  }
  return status;
}

typedef struct iree_hal_amdgpu_ipc_access_query_t {
  // HSA dispatch table used for agent and memory-pool queries.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Exporting allocation's device-local memory pool.
  hsa_amd_memory_pool_t exporting_memory_pool;
  // Destination list populated with proven GPU agents.
  iree_hal_amdgpu_access_agent_list_t* agent_list;
} iree_hal_amdgpu_ipc_access_query_t;

static hsa_status_t iree_hal_amdgpu_ipc_agent_query_memory_access(
    const iree_hal_amdgpu_ipc_access_query_t* query, hsa_agent_t agent,
    bool* out_can_access) {
  *out_can_access = false;
  hsa_amd_memory_pool_access_t access =
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  hsa_status_t status = iree_hsa_amd_agent_memory_pool_get_info_raw(
      query->libhsa, agent, query->exporting_memory_pool,
      HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
  if (status == HSA_STATUS_SUCCESS) {
    *out_can_access = access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  }
  return status;
}

static hsa_status_t iree_hal_amdgpu_access_query_ipc_peer(hsa_agent_t agent,
                                                          void* user_data) {
  iree_hal_amdgpu_ipc_access_query_t* query =
      (iree_hal_amdgpu_ipc_access_query_t*)user_data;
  if (iree_hal_amdgpu_access_agent_list_contains(query->agent_list, agent)) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_device_type_t device_type = 0;
  hsa_status_t status = iree_hsa_agent_get_info_raw(
      query->libhsa, agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (device_type != HSA_DEVICE_TYPE_GPU) {
    return HSA_STATUS_SUCCESS;
  }

  bool can_access = false;
  status =
      iree_hal_amdgpu_ipc_agent_query_memory_access(query, agent, &can_access);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (!can_access) return HSA_STATUS_SUCCESS;

  if (query->agent_list->count == IREE_ARRAYSIZE(query->agent_list->values)) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  query->agent_list->values[query->agent_list->count++] = agent;
  return HSA_STATUS_SUCCESS;
}

iree_status_t iree_hal_amdgpu_access_agent_list_resolve_ipc_memory_agents(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    hsa_amd_memory_pool_t exporting_memory_pool,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_agent_list);
  if (IREE_UNLIKELY(!exporting_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU IPC access resolution requires an exporting memory pool");
  }

  iree_hal_amdgpu_access_agent_list_t selected_agents;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_access_agent_list_resolve_queue_family_agents(
          topology, queue_family_affinity, &selected_agents));
  memset(out_agent_list, 0, sizeof(*out_agent_list));
  iree_hal_amdgpu_ipc_access_query_t query = {
      .libhsa = libhsa,
      .exporting_memory_pool = exporting_memory_pool,
      .agent_list = out_agent_list,
  };

  for (uint32_t i = 0; i < selected_agents.count; ++i) {
    bool can_access = false;
    const hsa_status_t access_status =
        iree_hal_amdgpu_ipc_agent_query_memory_access(
            &query, selected_agents.values[i], &can_access);
    if (IREE_UNLIKELY(access_status != HSA_STATUS_SUCCESS)) {
      return iree_status_from_hsa_status(
          __FILE__, __LINE__, access_status,
          "hsa_amd_agent_memory_pool_get_info",
          "querying selected AMDGPU agent access to IPC memory");
    }
    if (IREE_UNLIKELY(!can_access)) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "selected AMDGPU agent cannot access the exported IPC memory");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_append_unique(
        out_agent_list, selected_agents.values[i]));
  }
  return iree_hsa_iterate_agents(IREE_LIBHSA(libhsa),
                                 iree_hal_amdgpu_access_query_ipc_peer, &query);
}

iree_status_t iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_agent_list);
  memset(out_agent_list, 0, sizeof(*out_agent_list));

  if (IREE_UNLIKELY(
          access_scope == IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_NONE ||
          iree_any_bit_set(access_scope,
                           ~(iree_hal_virtual_memory_access_scope_t)
                               IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU virtual-memory access scope");
  }

  iree_hal_amdgpu_gpu_agent_mask_t gpu_agent_mask = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_select_families(
      topology, queue_family_affinity, &gpu_agent_mask));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t physical_device_ordinal = 0;
       physical_device_ordinal < topology->gpu_agent_count &&
       iree_status_is_ok(status);
       ++physical_device_ordinal) {
    if (!iree_all_bits_set(gpu_agent_mask, ((uint64_t)1)
                                               << physical_device_ordinal)) {
      continue;
    }
    if (iree_any_bit_set(access_scope,
                         IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE)) {
      status = iree_hal_amdgpu_access_agent_list_append_unique(
          out_agent_list, topology->gpu_agents[physical_device_ordinal]);
    }
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(access_scope,
                         IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST)) {
      const iree_host_size_t cpu_agent_ordinal =
          topology->gpu_cpu_map[physical_device_ordinal];
      if (IREE_UNLIKELY(cpu_agent_ordinal >= topology->cpu_agent_count)) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU topology maps GPU agent ordinal %" PRIhsz
                             " to invalid CPU agent ordinal %" PRIhsz,
                             physical_device_ordinal, cpu_agent_ordinal);
        break;
      }
      status = iree_hal_amdgpu_access_agent_list_append_unique(
          out_agent_list, topology->cpu_agents[cpu_agent_ordinal]);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_access_allow_agent_list(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_access_agent_list_t* agent_list, const void* ptr) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(agent_list);
  IREE_ASSERT_ARGUMENT(ptr);
  return iree_hsa_amd_agents_allow_access(IREE_LIBHSA(libhsa),
                                          agent_list->count, agent_list->values,
                                          /*flags=*/NULL, ptr);
}

iree_status_t iree_hal_amdgpu_access_lock_host_allocation_to_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_access_agent_list_t* agent_list,
    hsa_amd_memory_pool_t memory_pool, void* host_ptr,
    iree_device_size_t length, void** out_agent_ptr) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(agent_list);
  IREE_ASSERT_ARGUMENT(host_ptr);
  IREE_ASSERT_ARGUMENT(out_agent_ptr);
  return iree_hsa_amd_memory_lock_to_pool(
      IREE_LIBHSA(libhsa), host_ptr, (size_t)length,
      (hsa_agent_t*)agent_list->values, (int)agent_list->count, memory_pool,
      /*flags=*/0, out_agent_ptr);
}
