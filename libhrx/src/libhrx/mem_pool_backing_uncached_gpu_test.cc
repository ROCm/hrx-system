// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "buffer.h"
#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "mem_pool_backing.h"

namespace {

class MemPoolBackingUncachedGpuTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    hrx_status_t status = hrx_gpu_initialize(0);
    initialized_ = hrx_status_is_ok(status);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_get(0, &device_)));
  }

  static void TearDownTestSuite() {
    if (initialized_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_gpu_shutdown()));
      device_ = nullptr;
      initialized_ = false;
    }
  }

  // One GPU lifetime shared across every case in this suite.
  inline static bool initialized_ = false;
  // Borrowed first GPU device, valid while the runtime is initialized.
  inline static hrx_device_t device_ = nullptr;
};

void ExpectCommonAddressAllocation(hrx_device_t device, iree_hal_pool_t* pool) {
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_pool_query_host_access(pool).properties,
                        IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(8, 4096),
      iree_infinite_timeout(), &buffer));

  hrx::runtime::buffer_ptr wrapped_buffer;
  IREE_ASSERT_OK(hrx_buffer_create_from_hal(
      buffer, hrx_device_get_device_group(device),
      (hrx_memory_type_t)iree_hal_buffer_memory_type(buffer),
      (size_t)iree_hal_buffer_byte_length(buffer), nullptr,
      wrapped_buffer.for_output()));
  hrx_memory_scope_t program_scope;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
      device,
      iree_hal_queue_family_ordinal(
          iree_hal_queue_family(device->dispatch_queue)),
      HRX_MEMORY_SITE_PROGRAM, &program_scope)));
  uint64_t device_pointer = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_query_address(
      wrapped_buffer, program_scope, HRX_BUFFER_INTERFACE_DEVICE_ADDRESS,
      &device_pointer)));
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      iree_hal_buffer_allowed_access(buffer),
      IREE_HAL_BUFFER_MAP_FLAG_MAY_ALIAS, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.contents.data), device_pointer);
  IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&mapping));
  wrapped_buffer.reset();
  iree_hal_buffer_release(buffer);
}

TEST_F(MemPoolBackingUncachedGpuTest, HostMappingSourceKeepsBothRequirements) {
  const hrx_memory_type_t memory_type =
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT |
      HRX_MEMORY_TYPE_DEVICE_VISIBLE | HRX_MEMORY_TYPE_DEVICE_UNCACHED;
  iree_hal_pool_t* raw_pool = nullptr;
  IREE_ASSERT_OK(hrx_mem_pool_backing_retain_allocation_source(
      device_, memory_type, HRX_BUFFER_USAGE_MAPPING_SOURCE, &raw_pool));
  std::unique_ptr<iree_hal_pool_t, decltype(&iree_hal_pool_release)> pool(
      raw_pool, iree_hal_pool_release);
  EXPECT_EQ(pool.get(), device_->mem_pool_backing.uncached_host.mapping);
  EXPECT_EQ(device_->mem_pool_backing.host.mapping, nullptr);
  EXPECT_EQ(device_->mem_pool_backing.uncached_device.mapping, nullptr);

  const auto host = iree_hal_pool_query_host_access(pool.get());
  EXPECT_EQ(host.access, IREE_HAL_MEMORY_ACCESS_ALL);
  EXPECT_EQ(host.modes,
            IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool.get(), iree_hal_make_pool_reservation_request(4096, 0),
      iree_infinite_timeout(), &buffer));
  iree_hal_buffer_release(buffer);
}

TEST_F(MemPoolBackingUncachedGpuTest,
       FineGrainedHostSourceSupportsConcurrentLoadStore) {
  iree_hal_pool_t* raw_pool = nullptr;
  IREE_ASSERT_OK(hrx_mem_pool_backing_retain_common_host_source(
      device_, /*fine_grained=*/true, HRX_BUFFER_USAGE_DEFAULT, &raw_pool));
  std::unique_ptr<iree_hal_pool_t, decltype(&iree_hal_pool_release)> pool(
      raw_pool, iree_hal_pool_release);
  EXPECT_EQ(pool.get(), device_->mem_pool_backing.fine_grained_host.ordinary);

  iree_hal_memory_scope_t host_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      device_->hal_device_group,
      iree_hal_memory_site_t{IREE_HAL_MEMORY_SITE_HOST, nullptr}, &host_scope));
  const iree_hal_atomic_operation_capabilities_t host_atomics =
      iree_hal_pool_query_atomic_operations(pool.get(), host_scope);
  EXPECT_EQ(
      host_atomics.system_scope_32 & (IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                                      IREE_HAL_ATOMIC_OPERATION_FLAG_STORE),
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
          IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      device_->hal_device_group,
      iree_hal_memory_site_t{IREE_HAL_MEMORY_SITE_PROGRAM,
                             iree_hal_queue_family(device_->dispatch_queue)},
      &program_scope));
  const iree_hal_atomic_operation_capabilities_t program_atomics =
      iree_hal_pool_query_atomic_operations(pool.get(), program_scope);
  EXPECT_EQ(
      program_atomics.system_scope_32 & (IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                                         IREE_HAL_ATOMIC_OPERATION_FLAG_STORE),
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
          IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);

  ExpectCommonAddressAllocation(device_, pool.get());
}

TEST_F(MemPoolBackingUncachedGpuTest,
       CommonHostSourceUsesCachedCommonAddressMemory) {
  iree_hal_pool_t* raw_pool = nullptr;
  IREE_ASSERT_OK(hrx_mem_pool_backing_retain_common_host_source(
      device_, /*fine_grained=*/false, HRX_BUFFER_USAGE_DEFAULT, &raw_pool));
  std::unique_ptr<iree_hal_pool_t, decltype(&iree_hal_pool_release)> pool(
      raw_pool, iree_hal_pool_release);
  EXPECT_EQ(pool.get(), device_->mem_pool_backing.common_host.ordinary);
  ExpectCommonAddressAllocation(device_, pool.get());
}

}  // namespace
