// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/host_memory_pools.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Native handles refer to these dependency-owned records. The production query
// and its real iteration callbacks perform all filtering and selection.
struct Memory {
  // Segment returned by the native property query.
  hsa_region_segment_t segment = HSA_REGION_SEGMENT_GLOBAL;
  // Whether all agents can access this pool without an explicit grant.
  bool accessible = true;
  // Whether the runtime permits allocation from this pool or region.
  bool allocatable = true;
  // Native global flags in the queried API's enum domain.
  uint32_t flags = 0;
  // Property whose native query should fail, or -1 for no failure.
  int failed_property = -1;
};

class HostMemoryPoolsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    libhsa_.hsa_amd_agent_iterate_memory_pools = IteratePools;
    libhsa_.hsa_amd_memory_pool_get_info = QueryPool;
    libhsa_.hsa_agent_iterate_regions = IterateRegions;
    libhsa_.hsa_region_get_info = QueryRegion;
    pools_.resize(3);
    pools_[0].flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    pools_[1].flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
    pools_[2].flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
    regions_.resize(1);
    regions_[0].flags = HSA_REGION_GLOBAL_FLAG_FINE_GRAINED;
  }

  iree_status_t Query(iree_hal_amdgpu_host_memory_pools_t* out_pools) {
    return iree_hal_amdgpu_host_memory_pools_query(
        &libhsa_, hsa_agent_t{reinterpret_cast<uintptr_t>(this)}, out_pools);
  }

  uint64_t PoolHandle(size_t index) const {
    return reinterpret_cast<uintptr_t>(&pools_[index]);
  }

  uint64_t RegionHandle(size_t index) const {
    return reinterpret_cast<uintptr_t>(&regions_[index]);
  }

  static hsa_status_t HSA_API IteratePools(
      hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void*),
      void* user_data) {
    auto* self = reinterpret_cast<HostMemoryPoolsTest*>(agent.handle);
    for (size_t i = 0; i < self->pools_.size(); ++i) {
      ++self->visited_pools_;
      hsa_status_t status =
          callback(hsa_amd_memory_pool_t{self->PoolHandle(i)}, user_data);
      if (status != HSA_STATUS_SUCCESS) {
        return status;
      }
    }
    return self->pool_iteration_status_;
  }

  static hsa_status_t HSA_API IterateRegions(
      hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t, void*),
      void* user_data) {
    auto* self = reinterpret_cast<HostMemoryPoolsTest*>(agent.handle);
    for (size_t i = 0; i < self->regions_.size(); ++i) {
      ++self->visited_regions_;
      hsa_status_t status =
          callback(hsa_region_t{self->RegionHandle(i)}, user_data);
      if (status != HSA_STATUS_SUCCESS) {
        return status;
      }
    }
    return self->region_iteration_status_;
  }

  static hsa_status_t HSA_API QueryPool(hsa_amd_memory_pool_t pool,
                                        hsa_amd_memory_pool_info_t property,
                                        void* value) {
    const auto* memory = reinterpret_cast<const Memory*>(pool.handle);
    if (memory->failed_property == static_cast<int>(property)) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    switch (property) {
      case HSA_AMD_MEMORY_POOL_INFO_SEGMENT:
        *static_cast<hsa_region_segment_t*>(value) = memory->segment;
        break;
      case HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL:
        *static_cast<bool*>(value) = memory->accessible;
        break;
      case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED:
        *static_cast<bool*>(value) = memory->allocatable;
        break;
      case HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS:
        *static_cast<uint32_t*>(value) = memory->flags;
        break;
      default:
        ADD_FAILURE() << "Unexpected pool property " << property;
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    return HSA_STATUS_SUCCESS;
  }

  static hsa_status_t HSA_API QueryRegion(hsa_region_t region,
                                          hsa_region_info_t property,
                                          void* value) {
    const auto* memory = reinterpret_cast<const Memory*>(region.handle);
    if (memory->failed_property == static_cast<int>(property)) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    switch (property) {
      case HSA_REGION_INFO_SEGMENT:
        *static_cast<hsa_region_segment_t*>(value) = memory->segment;
        break;
      case HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED:
        *static_cast<bool*>(value) = memory->allocatable;
        break;
      case HSA_REGION_INFO_GLOBAL_FLAGS:
        *static_cast<hsa_region_global_flag_t*>(value) =
            static_cast<hsa_region_global_flag_t>(memory->flags);
        break;
      default:
        ADD_FAILURE() << "Unexpected region property " << property;
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    return HSA_STATUS_SUCCESS;
  }

  // Native dependency table, with no loaded runtime or device ownership.
  iree_hal_amdgpu_libhsa_t libhsa_ = {};
  // Agent-local native pool records, in enumeration order.
  std::vector<Memory> pools_;
  // Legacy region records, potentially spanning all CPU NUMA agents.
  std::vector<Memory> regions_;
  // Number of native pool callbacks invoked.
  size_t visited_pools_ = 0;
  // Number of native region callbacks invoked.
  size_t visited_regions_ = 0;
  // Native pool iterator result after visiting the records.
  hsa_status_t pool_iteration_status_ = HSA_STATUS_SUCCESS;
  // Native region iterator result after visiting the records.
  hsa_status_t region_iteration_status_ = HSA_STATUS_SUCCESS;
};

TEST_F(HostMemoryPoolsTest, PrefersSeparateFinePool) {
  iree_hal_amdgpu_host_memory_pools_t result;
  IREE_ASSERT_OK(Query(&result));
  EXPECT_EQ(result.coarse_pool.handle, PoolHandle(0));
  EXPECT_EQ(result.kernarg_pool.handle, PoolHandle(1));
  EXPECT_EQ(result.fine_pool.handle, PoolHandle(2));
  EXPECT_EQ(result.fine_region.handle, RegionHandle(0));
}

TEST_F(HostMemoryPoolsTest, KernargFinePoolCanServeBothRoles) {
  pools_.resize(2);
  iree_hal_amdgpu_host_memory_pools_t result;
  IREE_ASSERT_OK(Query(&result));
  EXPECT_EQ(result.fine_pool.handle, PoolHandle(1));
  EXPECT_EQ(result.kernarg_pool.handle, PoolHandle(1));
}

TEST_F(HostMemoryPoolsTest, RegionsAcrossSixteenNumaNodes) {
  regions_.resize(64, regions_[0]);
  iree_hal_amdgpu_host_memory_pools_t result;
  IREE_ASSERT_OK(Query(&result));
  EXPECT_EQ(result.fine_region.handle, RegionHandle(0));
  EXPECT_EQ(visited_regions_, 64u);
}

TEST_F(HostMemoryPoolsTest, PoolCountIsNotBoundedByCollectorStorage) {
  pools_.resize(64, pools_[2]);
  iree_hal_amdgpu_host_memory_pools_t result;
  IREE_ASSERT_OK(Query(&result));
  EXPECT_EQ(result.fine_pool.handle, PoolHandle(2));
  EXPECT_EQ(result.kernarg_pool.handle, PoolHandle(1));
  EXPECT_EQ(visited_pools_, 64u);
}

TEST_F(HostMemoryPoolsTest, FiltersUnsupportedMemoryBeforeSelecting) {
  auto valid_pools = pools_;
  pools_.assign(3, valid_pools[1]);
  pools_[0].segment = HSA_REGION_SEGMENT_GROUP;
  pools_[1].accessible = false;
  pools_[2].allocatable = false;
  pools_.insert(pools_.end(), valid_pools.begin(), valid_pools.end());
  regions_.resize(4, regions_[0]);
  regions_[0].segment = HSA_REGION_SEGMENT_GROUP;
  regions_[1].allocatable = false;
  regions_[2].flags = HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED;
  iree_hal_amdgpu_host_memory_pools_t result;
  IREE_ASSERT_OK(Query(&result));
  EXPECT_EQ(result.coarse_pool.handle, PoolHandle(3));
  EXPECT_EQ(result.kernarg_pool.handle, PoolHandle(4));
  EXPECT_EQ(result.fine_pool.handle, PoolHandle(5));
  EXPECT_EQ(result.fine_region.handle, RegionHandle(3));
}

TEST_F(HostMemoryPoolsTest, SelectsEligibleMemoryBeyondFirstThirtyTwoEntries) {
  const auto valid_pools = pools_;
  pools_.assign(64, Memory{});
  for (size_t i = 0; i < valid_pools.size(); ++i) {
    pools_[61 + i] = valid_pools[i];
  }
  regions_.assign(64, Memory{});
  regions_[63].flags = HSA_REGION_GLOBAL_FLAG_FINE_GRAINED;
  iree_hal_amdgpu_host_memory_pools_t result;
  IREE_ASSERT_OK(Query(&result));
  EXPECT_EQ(result.coarse_pool.handle, PoolHandle(61));
  EXPECT_EQ(result.kernarg_pool.handle, PoolHandle(62));
  EXPECT_EQ(result.fine_pool.handle, PoolHandle(63));
  EXPECT_EQ(result.fine_region.handle, RegionHandle(63));
}

TEST_F(HostMemoryPoolsTest, KeepsFirstEquivalentPool) {
  auto first = pools_;
  pools_.insert(pools_.end(), first.begin(), first.end());
  iree_hal_amdgpu_host_memory_pools_t result;
  IREE_ASSERT_OK(Query(&result));
  EXPECT_EQ(result.coarse_pool.handle, PoolHandle(0));
  EXPECT_EQ(result.kernarg_pool.handle, PoolHandle(1));
  EXPECT_EQ(result.fine_pool.handle, PoolHandle(2));
}

TEST_F(HostMemoryPoolsTest, MissingEachRequiredMemoryRoleFails) {
  const auto valid_pools = pools_;
  const auto valid_regions = regions_;
  for (int role = 0; role < 4; ++role) {
    SCOPED_TRACE(role);
    pools_ = valid_pools;
    regions_ = valid_regions;
    if (role == 0) {
      pools_[0].flags = 0;
    }
    if (role == 1) {
      pools_[1].flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
    }
    if (role == 2) {
      pools_[1].flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
      pools_[2].flags = 0;
    }
    if (role == 3) {
      regions_.clear();
    }
    iree_hal_amdgpu_host_memory_pools_t result = {{1}, {1}, {1}, {1}};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Query(&result));
    EXPECT_EQ(result.coarse_pool.handle, 0u);
    EXPECT_EQ(result.kernarg_pool.handle, 0u);
    EXPECT_EQ(result.fine_pool.handle, 0u);
    EXPECT_EQ(result.fine_region.handle, 0u);
  }
}

TEST_F(HostMemoryPoolsTest, PropagatesNativeIterationFailures) {
  for (bool fail_regions : {false, true}) {
    pool_iteration_status_ =
        fail_regions ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    region_iteration_status_ =
        fail_regions ? HSA_STATUS_ERROR_OUT_OF_RESOURCES : HSA_STATUS_SUCCESS;
    iree_hal_amdgpu_host_memory_pools_t result;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Query(&result));
    EXPECT_EQ(result.fine_pool.handle, 0u);
    EXPECT_EQ(result.fine_region.handle, 0u);
  }
}

TEST_F(HostMemoryPoolsTest, PropagatesEveryPoolQueryFailure) {
  for (auto property : {HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                        HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                        HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                        HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS}) {
    SCOPED_TRACE(property);
    pools_[0].failed_property = property;
    iree_hal_amdgpu_host_memory_pools_t result;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Query(&result));
    EXPECT_EQ(result.fine_pool.handle, 0u);
  }
}

TEST_F(HostMemoryPoolsTest, PropagatesEveryRegionQueryFailure) {
  for (auto property :
       {HSA_REGION_INFO_SEGMENT, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
        HSA_REGION_INFO_GLOBAL_FLAGS}) {
    SCOPED_TRACE(property);
    regions_[0].failed_property = property;
    iree_hal_amdgpu_host_memory_pools_t result;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Query(&result));
    EXPECT_EQ(result.fine_region.handle, 0u);
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
