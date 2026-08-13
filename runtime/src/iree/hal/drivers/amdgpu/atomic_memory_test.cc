// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/atomic_memory.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_atomic_memory_cell_flags_t SelectSourceCells(
    uint32_t global_flags, uint32_t allocation_flags,
    hsa_amd_memory_pool_access_t access, iree_host_size_t link_hop_count = 0,
    const hsa_amd_memory_pool_link_info_t* link_hops = nullptr) {
  const iree_hal_amdgpu_atomic_memory_source_selection_t selection = {
      /*.global_flags=*/global_flags,
      /*.allocation_flags=*/allocation_flags,
      /*.access=*/access,
      /*.link_hop_count=*/link_hop_count,
      /*.link_hops=*/link_hops,
  };
  iree_hal_amdgpu_atomic_memory_cell_flags_t cell_flags =
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE;
  IREE_EXPECT_OK(iree_hal_amdgpu_atomic_memory_select_source_cells(
      &selection, &cell_flags));
  return cell_flags;
}

TEST(AtomicMemoryTest, CoarseLocalMemorySupportsDeviceScope) {
  EXPECT_EQ(SelectSourceCells(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                              HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
                              HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32 |
                IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64);
}

TEST(AtomicMemoryTest, FinePeerMemoryIntersectsEveryLinkWidth) {
  const hsa_amd_memory_pool_link_info_t link_hops[] = {
      {
          /*.min_latency=*/0,
          /*.max_latency=*/0,
          /*.min_bandwidth=*/0,
          /*.max_bandwidth=*/0,
          /*.atomic_support_32bit=*/true,
          /*.atomic_support_64bit=*/true,
          /*.coherent_support=*/true,
          /*.link_type=*/HSA_AMD_LINK_INFO_TYPE_XGMI,
          /*.numa_distance=*/0,
      },
      {
          /*.min_latency=*/0,
          /*.max_latency=*/0,
          /*.min_bandwidth=*/0,
          /*.max_bandwidth=*/0,
          /*.atomic_support_32bit=*/true,
          /*.atomic_support_64bit=*/false,
          /*.coherent_support=*/true,
          /*.link_type=*/HSA_AMD_LINK_INFO_TYPE_PCIE,
          /*.numa_distance=*/0,
      },
  };
  EXPECT_EQ(SelectSourceCells(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
                              HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
                              HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT,
                              IREE_ARRAYSIZE(link_hops), link_hops),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32 |
                IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32);
}

TEST(AtomicMemoryTest, PcieFineAllocationDoesNotPromiseSystemScope) {
  EXPECT_EQ(SelectSourceCells(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
                              HSA_AMD_MEMORY_POOL_PCIE_FLAG,
                              HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32 |
                IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64);
}

TEST(AtomicMemoryTest, NeverAllowedPathSupportsNoCells) {
  EXPECT_EQ(SelectSourceCells(
                HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED,
                HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
                HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE);
}

TEST(AtomicMemoryTest, RejectsAmbiguousPoolGranularity) {
  const iree_hal_amdgpu_atomic_memory_source_selection_t selection = {
      /*.global_flags=*/HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED |
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
      /*.allocation_flags=*/HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
      /*.access=*/HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT,
      /*.link_hop_count=*/0,
      /*.link_hops=*/nullptr,
  };
  iree_hal_amdgpu_atomic_memory_cell_flags_t cell_flags =
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_atomic_memory_select_source_cells(
                            &selection, &cell_flags));
}

TEST(AtomicMemoryTest, DeviceSelectionRequiresEverySource) {
  const iree_hal_amdgpu_atomic_memory_source_masks_t source_masks = {
      /*.device_scope_32=*/0b111,
      /*.device_scope_64=*/0b011,
      /*.system_scope_32=*/0b101,
      /*.system_scope_64=*/0b001,
  };
  EXPECT_EQ(
      iree_hal_amdgpu_atomic_memory_select_device_cells(&source_masks, 0b101),
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32 |
          IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32);
  EXPECT_EQ(
      iree_hal_amdgpu_atomic_memory_select_device_cells(&source_masks, 0b001),
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL);
  EXPECT_EQ(iree_hal_amdgpu_atomic_memory_select_device_cells(&source_masks, 0),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE);
}

TEST(AtomicMemoryTest, ExpandsCellsToCompleteOperationFamilies) {
  const iree_hal_atomic_operation_capabilities_t capabilities =
      iree_hal_amdgpu_atomic_memory_expand_capabilities(
          IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64 |
          IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32);
  EXPECT_EQ(capabilities.device_scope_32, IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);
  EXPECT_EQ(capabilities.device_scope_64, IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(capabilities.system_scope_32, IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(capabilities.system_scope_64, IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);
}

}  // namespace
}  // namespace iree::hal::amdgpu
