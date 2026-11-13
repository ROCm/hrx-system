// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/memory_backend_test_base.h"

namespace iree::hal::amd::fabric {

TEST_F(MemoryBackendTest, PreparesOneHostSourceForCpuAndPublicHostScopes) {
  native_.mapping_enabled = true;
  native_.host_reservation_enabled = true;
  native_.host_source_enabled = true;
  native_.required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  native_.native_prefix = 4096;
  const uint64_t source_interface = UINT64_C(1)
                                    << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, source_interface,
                            IREE_HAL_MEMORY_ACCESS_ALL,
                            IREE_HAL_BUFFER_USAGE_STORAGE, {}, source_interface,
                            &pool));
  const auto requirements = iree_hal_pool_query_mapping_requirements(pool);
  EXPECT_EQ(requirements.minimum_granularity, 4096u);
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(16384, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(backing, 4096, 8192, allocator_, &view));
  const auto* cpu_family = iree_hal_device_queue_family(cpu_device_, 0);
  const iree_hal_memory_scope_t scopes[] = {
      ResolveScope(IREE_HAL_MEMORY_SITE_HOST),
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, cpu_family),
      ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, cpu_family),
  };
  const void* descriptor = nullptr;
  for (auto scope : scopes) {
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, scope, IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &slot));
    const auto source =
        iree_hal_buffer_native_binding(view, slot).mapping_source;
    ASSERT_NE(source.descriptor, nullptr);
    EXPECT_EQ(source.offset, 8192u);
    EXPECT_EQ(source.granularity, 4096u);
    if (descriptor) {
      EXPECT_EQ(source.descriptor, descriptor);
    }
    descriptor = source.descriptor;
  }
  for (const auto* family : families_) {
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, family),
        IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &slot));
    EXPECT_EQ(
        iree_hal_buffer_native_binding(view, slot).mapping_source.descriptor,
        descriptor);
  }
  iree_hal_buffer_release(view);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, HostVisibilityDoesNotImplyHostMappingSource) {
  native_.host_reservation_enabled = true;
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, 0, IREE_HAL_MEMORY_ACCESS_ALL,
                 IREE_HAL_BUFFER_USAGE_STORAGE, {},
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
                 &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.create_count, 0);
  EXPECT_GT(native_.profile_query_count, 0);
}

TEST_F(MemoryBackendTest, MappingSourcesHonorRequiredBackingPreparation) {
  native_.mapping_enabled = true;
  native_.host_source_enabled = true;
  native_.required_flags = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  // Only one domain needs host-remappable backing; both consume the same
  // allocation. The whole-backing requirement must survive the joint query.
  native_.reservation_source_flags[1] = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  iree_hal_pool_t* pool = nullptr;
  // No CPU family or public mapping requests the host backing property.
  IREE_ASSERT_OK(CreatePool(
      IREE_HAL_BUFFER_USAGE_STORAGE,
      UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, 0, &pool));
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access, 0u);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.create_count, 1);
  // The private file view is prepared without adding HOST_VISIBLE to the
  // allocation's required flags or granting public host access.
  EXPECT_EQ(native_.map_count, 1);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);

  native_.host_source_enabled = false;
  pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, 0,
                 &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.create_count, 1);
}

TEST_F(MemoryBackendTest, IndependentNpuMappingsNeedNoRemappableHostBacking) {
  native_.mapping_enabled = true;
  native_.required_flags = 0;
  iree_hal_pool_t* pool = nullptr;
  // The same XDNA address kind may use independent device translations. Its
  // source profile has no remappable host route. An optional file view does
  // not turn host remapping into a physical backing requirement.
  IREE_ASSERT_OK(CreatePool(
      IREE_HAL_BUFFER_USAGE_STORAGE,
      UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, 0, &pool));
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access, 0u);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.map_count, 1);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest,
       GpuBackingPublishesQualifiedHostSourceWithoutCpuFamily) {
  native_.mapping_enabled = true;
  native_.host_reservation_enabled = true;
  native_.host_source_enabled = true;
  native_.required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(
      IREE_HAL_BUFFER_USAGE_STORAGE,
      UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &pool));
  iree_hal_buffer_native_binding_slot_t host_slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, ResolveScope(IREE_HAL_MEMORY_SITE_HOST),
      IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &host_slot));
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(16384, 4096),
      iree_infinite_timeout(), &backing));
  const auto host_source =
      iree_hal_buffer_native_binding(backing, host_slot).mapping_source;
  EXPECT_NE(host_source.descriptor, nullptr);
  EXPECT_EQ(host_source.granularity, 4096u);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, HostSourceRequiresQualifiedNativeBacking) {
  native_.mapping_enabled = true;
  native_.host_reservation_enabled = true;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(
      IREE_HAL_BUFFER_USAGE_STORAGE,
      UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &pool));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_pool_resolve_binding(
                            pool, ResolveScope(IREE_HAL_MEMORY_SITE_HOST),
                            IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &slot));
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]),
      IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &slot));
  iree_hal_pool_release(pool);
}

}  // namespace iree::hal::amd::fabric
