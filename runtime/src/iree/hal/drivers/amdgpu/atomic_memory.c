// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/atomic_memory.h"

// Matches the bounded HSA link-path representation used by logical-device
// topology refinement. ROCr currently reports one aggregate record per route.
#define IREE_HAL_AMDGPU_ATOMIC_MEMORY_MAX_LINK_HOPS 16

iree_status_t iree_hal_amdgpu_atomic_memory_select_source_cells(
    const iree_hal_amdgpu_atomic_memory_source_selection_t* selection,
    iree_hal_amdgpu_atomic_memory_cell_flags_t* out_cell_flags) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_cell_flags);
  *out_cell_flags = IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE;

  switch (selection->access) {
    case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:
      return iree_ok_status();
    case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:
    case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:
      break;
    default:
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HSA reported unknown memory pool access mode %u",
                              (uint32_t)selection->access);
  }

  const bool coarse_grained = iree_any_bit_set(
      selection->global_flags, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
  const bool fine_grained = iree_any_bit_set(
      selection->global_flags,
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED);
  if (IREE_UNLIKELY(coarse_grained == fine_grained)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HSA memory pool global flags 0x%08" PRIx32
        " do not identify exactly one coarse- or fine-grained class",
        selection->global_flags);
  }
  if (IREE_UNLIKELY(selection->link_hop_count && !selection->link_hops)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU atomic memory selection requires link records for %" PRIhsz
        " hops",
        selection->link_hop_count);
  }

  bool supports_32bit = true;
  bool supports_64bit = true;
  for (iree_host_size_t i = 0; i < selection->link_hop_count; ++i) {
    supports_32bit &= selection->link_hops[i].atomic_support_32bit;
    supports_64bit &= selection->link_hops[i].atomic_support_64bit;
  }

  iree_hal_amdgpu_atomic_memory_cell_flags_t cell_flags =
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE;
  if (supports_32bit) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32;
  }
  if (supports_64bit) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64;
  }

  const bool supports_system_scope =
      fine_grained && !iree_any_bit_set(selection->allocation_flags,
                                        HSA_AMD_MEMORY_POOL_PCIE_FLAG);
  if (supports_system_scope && supports_32bit) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32;
  }
  if (supports_system_scope && supports_64bit) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_64;
  }

  *out_cell_flags = cell_flags;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_atomic_memory_query_source_masks(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    hsa_amd_memory_pool_t memory_pool, uint32_t allocation_flags,
    iree_hal_amdgpu_atomic_memory_source_masks_t* out_source_masks) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_source_masks);
  memset(out_source_masks, 0, sizeof(*out_source_masks));

  uint32_t global_flags = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
      &global_flags));

  for (iree_host_size_t source_ordinal = 0;
       source_ordinal < topology->gpu_agent_count; ++source_ordinal) {
    hsa_amd_memory_pool_access_t access =
        HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(libhsa), topology->gpu_agents[source_ordinal], memory_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access));

    uint32_t link_hop_count = 0;
    hsa_amd_memory_pool_link_info_t
        link_hops[IREE_HAL_AMDGPU_ATOMIC_MEMORY_MAX_LINK_HOPS];
    if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
          IREE_LIBHSA(libhsa), topology->gpu_agents[source_ordinal],
          memory_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS,
          &link_hop_count));
      if (IREE_UNLIKELY(link_hop_count >
                        IREE_HAL_AMDGPU_ATOMIC_MEMORY_MAX_LINK_HOPS)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "HSA reports %" PRIu32
            " link hops from AMDGPU physical device %" PRIhsz
            " to a memory pool (maximum %u)",
            link_hop_count, source_ordinal,
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_MAX_LINK_HOPS);
      }
      if (link_hop_count) {
        memset(link_hops, 0, sizeof(link_hops[0]) * link_hop_count);
        IREE_RETURN_IF_ERROR(iree_hsa_amd_agent_memory_pool_get_info(
            IREE_LIBHSA(libhsa), topology->gpu_agents[source_ordinal],
            memory_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_hops));
      }
    }

    const iree_hal_amdgpu_atomic_memory_source_selection_t selection = {
        .global_flags = global_flags,
        .allocation_flags = allocation_flags,
        .access = access,
        .link_hop_count = link_hop_count,
        .link_hops = link_hops,
    };
    iree_hal_amdgpu_atomic_memory_cell_flags_t cell_flags =
        IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_atomic_memory_select_source_cells(
        &selection, &cell_flags));

    const iree_hal_amdgpu_gpu_agent_mask_t source_bit =
        ((iree_hal_amdgpu_gpu_agent_mask_t)1) << source_ordinal;
    if (iree_any_bit_set(
            cell_flags,
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32)) {
      out_source_masks->device_scope_32 |= source_bit;
    }
    if (iree_any_bit_set(
            cell_flags,
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64)) {
      out_source_masks->device_scope_64 |= source_bit;
    }
    if (iree_any_bit_set(
            cell_flags,
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32)) {
      out_source_masks->system_scope_32 |= source_bit;
    }
    if (iree_any_bit_set(
            cell_flags,
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_64)) {
      out_source_masks->system_scope_64 |= source_bit;
    }
  }

  return iree_ok_status();
}

iree_hal_amdgpu_atomic_memory_cell_flags_t
iree_hal_amdgpu_atomic_memory_select_device_cells(
    const iree_hal_amdgpu_atomic_memory_source_masks_t* source_masks,
    iree_hal_amdgpu_gpu_agent_mask_t device_mask) {
  if (!device_mask) return IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE;

  iree_hal_amdgpu_atomic_memory_cell_flags_t cell_flags =
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE;
  if (!iree_any_bit_set(device_mask, ~source_masks->device_scope_32)) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32;
  }
  if (!iree_any_bit_set(device_mask, ~source_masks->device_scope_64)) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64;
  }
  if (!iree_any_bit_set(device_mask, ~source_masks->system_scope_32)) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32;
  }
  if (!iree_any_bit_set(device_mask, ~source_masks->system_scope_64)) {
    cell_flags |= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_64;
  }
  return cell_flags;
}

iree_hal_atomic_operation_capabilities_t
iree_hal_amdgpu_atomic_memory_expand_capabilities(
    iree_hal_amdgpu_atomic_memory_cell_flags_t cell_flags) {
  return (iree_hal_atomic_operation_capabilities_t){
      .device_scope_32 =
          iree_any_bit_set(
              cell_flags,
              IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32)
              ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
              : IREE_HAL_ATOMIC_OPERATION_FLAG_NONE,
      .device_scope_64 =
          iree_any_bit_set(
              cell_flags,
              IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64)
              ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
              : IREE_HAL_ATOMIC_OPERATION_FLAG_NONE,
      .system_scope_32 =
          iree_any_bit_set(
              cell_flags,
              IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32)
              ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
              : IREE_HAL_ATOMIC_OPERATION_FLAG_NONE,
      .system_scope_64 =
          iree_any_bit_set(
              cell_flags,
              IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_64)
              ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
              : IREE_HAL_ATOMIC_OPERATION_FLAG_NONE,
  };
}
