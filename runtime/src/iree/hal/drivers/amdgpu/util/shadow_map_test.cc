// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/shadow_map.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct FakeMappedSlab {
  uint64_t index = 0;
  uintptr_t base_address = 0;
  iree_device_size_t slab_size = 0;
  uint64_t allocation_handle = 0;
};

struct FakeMapper {
  uintptr_t reservation_base = 0x100000000ull;
  iree_host_size_t reserve_call_count = 0;
  iree_host_size_t release_reservation_call_count = 0;
  iree_host_size_t map_slab_call_count = 0;
  iree_host_size_t unmap_slab_call_count = 0;
  iree_host_size_t fail_map_slab_call_count = 0;
  iree_device_size_t last_reservation_size = 0;
  iree_device_size_t last_reservation_alignment = 0;
  uint64_t next_allocation_handle = 1;
  std::vector<FakeMappedSlab> mapped_slabs;
  std::vector<FakeMappedSlab> unmapped_slabs;
};

static iree_status_t FakeReserve(iree_hal_amdgpu_shadow_map_t* map,
                                 iree_device_size_t reservation_size,
                                 iree_device_size_t alignment,
                                 IREE_AMDGPU_DEVICE_PTR void** out_base_ptr) {
  auto* mapper = static_cast<FakeMapper*>(map->mapper.user_data);
  ++mapper->reserve_call_count;
  mapper->last_reservation_size = reservation_size;
  mapper->last_reservation_alignment = alignment;
  *out_base_ptr =
      reinterpret_cast<IREE_AMDGPU_DEVICE_PTR void*>(mapper->reservation_base);
  return iree_ok_status();
}

static void FakeReleaseReservation(iree_hal_amdgpu_shadow_map_t* map,
                                   IREE_AMDGPU_DEVICE_PTR void* base_ptr,
                                   iree_device_size_t reservation_size) {
  auto* mapper = static_cast<FakeMapper*>(map->mapper.user_data);
  ++mapper->release_reservation_call_count;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(base_ptr), mapper->reservation_base);
  EXPECT_EQ(reservation_size, mapper->last_reservation_size);
}

static iree_status_t FakeMapSlab(
    iree_hal_amdgpu_shadow_map_t* map, IREE_AMDGPU_DEVICE_PTR void* target_ptr,
    iree_device_size_t slab_size, iree_host_size_t access_desc_count,
    const hsa_amd_memory_access_desc_t* access_descs,
    hsa_amd_vmem_alloc_handle_t* out_allocation_handle) {
  auto* mapper = static_cast<FakeMapper*>(map->mapper.user_data);
  ++mapper->map_slab_call_count;
  EXPECT_EQ(access_desc_count, 1u);
  EXPECT_NE(access_descs, nullptr);
  if (mapper->map_slab_call_count == mapper->fail_map_slab_call_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "synthetic shadow slab map failure");
  }

  const uint64_t allocation_handle = mapper->next_allocation_handle++;
  *out_allocation_handle =
      hsa_amd_vmem_alloc_handle_t{/*.handle=*/allocation_handle};
  const uint64_t slab_index =
      (reinterpret_cast<uintptr_t>(target_ptr) - mapper->reservation_base) /
      slab_size;
  mapper->mapped_slabs.push_back(FakeMappedSlab{
      /*.index=*/slab_index,
      /*.base_address=*/reinterpret_cast<uintptr_t>(target_ptr),
      /*.slab_size=*/slab_size,
      /*.allocation_handle=*/allocation_handle,
  });
  return iree_ok_status();
}

static void FakeUnmapSlab(iree_hal_amdgpu_shadow_map_t* map,
                          IREE_AMDGPU_DEVICE_PTR void* target_ptr,
                          iree_device_size_t slab_size,
                          hsa_amd_vmem_alloc_handle_t allocation_handle) {
  auto* mapper = static_cast<FakeMapper*>(map->mapper.user_data);
  ++mapper->unmap_slab_call_count;
  const uint64_t slab_index =
      (reinterpret_cast<uintptr_t>(target_ptr) - mapper->reservation_base) /
      slab_size;
  mapper->unmapped_slabs.push_back(FakeMappedSlab{
      /*.index=*/slab_index,
      /*.base_address=*/reinterpret_cast<uintptr_t>(target_ptr),
      /*.slab_size=*/slab_size,
      /*.allocation_handle=*/allocation_handle.handle,
  });
}

static iree_hal_amdgpu_shadow_map_mapper_t FakeShadowMapMapper(
    FakeMapper* mapper) {
  return iree_hal_amdgpu_shadow_map_mapper_t{
      /*.user_data=*/mapper,
      /*.reserve=*/FakeReserve,
      /*.release_reservation=*/FakeReleaseReservation,
      /*.map_slab=*/FakeMapSlab,
      /*.unmap_slab=*/FakeUnmapSlab,
  };
}

static hsa_amd_memory_access_desc_t FakeAccessDesc() {
  return hsa_amd_memory_access_desc_t{
      /*.permissions=*/HSA_ACCESS_PERMISSION_RW,
      /*.agent_handle=*/hsa_agent_t{/*.handle=*/1},
  };
}

static iree_hal_amdgpu_shadow_map_params_t DefaultParams(
    FakeMapper* mapper, const hsa_amd_memory_access_desc_t* access_desc) {
  const uint32_t shadow_scale_shift =
      IREE_HAL_AMDGPU_SHADOW_MAP_DEFAULT_SCALE_SHIFT;
  return iree_hal_amdgpu_shadow_map_params_t{
      /*.host_allocator=*/iree_allocator_system(),
      /*.shadow_scale_shift=*/shadow_scale_shift,
      /*.application_window_base=*/0x200000ull,
      /*.shadow_size=*/1024,
      /*.slab_size=*/64,
      /*.mapping_mode=*/IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_SPARSE,
      /*.initial_slab_value=*/0,
      /*.access_desc_count=*/1,
      /*.access_descs=*/access_desc,
      /*.mapper=*/FakeShadowMapMapper(mapper),
  };
}

TEST(ShadowMapTest, InitializeComputesGeometry) {
  FakeMapper mapper;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  EXPECT_TRUE(map.initialized);
  EXPECT_EQ(
      map.reservation_base_ptr,
      reinterpret_cast<IREE_AMDGPU_DEVICE_PTR void*>(mapper.reservation_base));
  EXPECT_EQ(map.reservation_size, 1024u);
  EXPECT_EQ(map.slab_size, 64u);
  EXPECT_EQ(map.mapping_mode, IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_SPARSE);
  EXPECT_EQ(map.initial_slab_value, 0u);
  EXPECT_EQ(map.application_window_size, 8192u);
  EXPECT_EQ(map.shadow_base,
            mapper.reservation_base - (params.application_window_base >> 3));
  EXPECT_EQ(mapper.reserve_call_count, 1u);
  EXPECT_EQ(mapper.last_reservation_alignment, 64u);

  iree_hal_amdgpu_shadow_map_deinitialize(&map);
  EXPECT_EQ(mapper.release_reservation_call_count, 1u);
}

TEST(ShadowMapTest, CalculateRangeUsesShadowBytesAndSlabs) {
  FakeMapper mapper;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  iree_hal_amdgpu_shadow_map_range_t range;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_calculate_range(
      &map, params.application_window_base + 63, 2, &range));
  EXPECT_EQ(range.shadow_offset, 7u);
  EXPECT_EQ(range.shadow_length, 2u);
  EXPECT_EQ(range.first_slab_index, 0u);
  EXPECT_EQ(range.slab_count, 1u);
  EXPECT_EQ(range.shadow_address, mapper.reservation_base + 7u);

  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_calculate_range(
      &map, params.application_window_base + 64 * 8 - 1, 2, &range));
  EXPECT_EQ(range.shadow_offset, 63u);
  EXPECT_EQ(range.shadow_length, 2u);
  EXPECT_EQ(range.first_slab_index, 0u);
  EXPECT_EQ(range.slab_count, 2u);

  iree_hal_amdgpu_shadow_map_deinitialize(&map);
}

TEST(ShadowMapTest, MapRangeCoalescesDuplicateSlabs) {
  FakeMapper mapper;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  iree_hal_amdgpu_shadow_map_range_t range;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_map_range(
      &map, params.application_window_base + 64 * 8 - 1, 2, &range));
  EXPECT_EQ(range.slab_count, 2u);
  ASSERT_EQ(map.slab_count, 2u);
  EXPECT_EQ(map.slabs[0].index, 0u);
  EXPECT_EQ(map.slabs[1].index, 1u);
  EXPECT_EQ(mapper.map_slab_call_count, 2u);

  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_map_range(
      &map, params.application_window_base + 64 * 8, 8, &range));
  EXPECT_EQ(range.first_slab_index, 1u);
  EXPECT_EQ(range.slab_count, 1u);
  EXPECT_EQ(map.slab_count, 2u);
  EXPECT_EQ(mapper.map_slab_call_count, 2u);

  iree_hal_amdgpu_shadow_map_deinitialize(&map);
  EXPECT_EQ(mapper.unmap_slab_call_count, 2u);
}

TEST(ShadowMapTest, QueryStatisticsReportsMappedPhysicalSlabs) {
  FakeMapper mapper;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  iree_hal_amdgpu_shadow_map_statistics_t statistics;
  iree_hal_amdgpu_shadow_map_query_statistics(&map, &statistics);
  EXPECT_EQ(statistics.mapped_slab_count, 0u);
  EXPECT_EQ(statistics.committed_size, 0u);

  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_map_slab(&map, 3));
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_map_slab(&map, 7));

  iree_hal_amdgpu_shadow_map_query_statistics(&map, &statistics);
  EXPECT_EQ(statistics.mapped_slab_count, 2u);
  EXPECT_EQ(statistics.committed_size, 2 * params.slab_size);

  iree_hal_amdgpu_shadow_map_deinitialize(&map);
}

TEST(ShadowMapTest, MapExplicitDistantSlab) {
  FakeMapper mapper;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_map_slab(&map, 15));
  ASSERT_EQ(map.slab_count, 1u);
  EXPECT_EQ(map.slabs[0].index, 15u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(map.slabs[0].base_ptr),
            mapper.reservation_base + 15 * params.slab_size);

  iree_hal_amdgpu_shadow_map_deinitialize(&map);
  ASSERT_EQ(mapper.unmapped_slabs.size(), 1u);
  EXPECT_EQ(mapper.unmapped_slabs[0].index, 15u);
}

TEST(ShadowMapTest, MapRangeRollsBackNewSlabsOnFailure) {
  FakeMapper mapper;
  mapper.fail_map_slab_call_count = 2;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  iree_hal_amdgpu_shadow_map_range_t range;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amdgpu_shadow_map_map_range(
          &map, params.application_window_base + 64 * 8 - 1, 2, &range));
  EXPECT_EQ(map.slab_count, 0u);
  EXPECT_EQ(mapper.map_slab_call_count, 2u);
  ASSERT_EQ(mapper.unmapped_slabs.size(), 1u);
  EXPECT_EQ(mapper.unmapped_slabs[0].index, 0u);

  iree_hal_amdgpu_shadow_map_deinitialize(&map);
  EXPECT_EQ(mapper.release_reservation_call_count, 1u);
}

TEST(ShadowMapTest, RejectsOutOfWindowRanges) {
  FakeMapper mapper;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  iree_hal_amdgpu_shadow_map_range_t range;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_shadow_map_calculate_range(
          &map, params.application_window_base - 1, 1, &range));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_shadow_map_calculate_range(
          &map, params.application_window_base + map.application_window_size, 1,
          &range));

  iree_hal_amdgpu_shadow_map_deinitialize(&map);
}

TEST(ShadowMapTest, RejectsInvalidGeometry) {
  FakeMapper mapper;
  hsa_amd_memory_access_desc_t access_desc = FakeAccessDesc();
  iree_hal_amdgpu_shadow_map_params_t params =
      DefaultParams(&mapper, &access_desc);

  iree_hal_amdgpu_shadow_map_t map;
  params.shadow_size = 1000;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_shadow_map_initialize(&params, &map));

  params = DefaultParams(&mapper, &access_desc);
  params.application_window_base += 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_shadow_map_initialize(&params, &map));
}

}  // namespace
}  // namespace iree::hal::amdgpu
