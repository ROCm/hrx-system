// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/slab_provider.h"

#include <array>
#include <cstring>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/access_policy.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class SlabProviderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (iree_status_is_unavailable(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t SlabProviderTest::host_allocator_;
iree_hal_amdgpu_libhsa_t SlabProviderTest::libhsa_;
iree_hal_amdgpu_topology_t SlabProviderTest::topology_;

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device);
    iree_hal_device_group_release(device_group);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context.params(),
        host_allocator, &base_device));
    return iree_hal_device_group_create_from_device(
        base_device, create_context.frontier_tracker(), host_allocator,
        &device_group);
  }

  iree_hal_amdgpu_logical_device_t* device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device;
  }

  iree_hal_device_t* hal_device() const { return base_device; }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device = NULL;

  // Device group that owns the topology assigned to |base_device|.
  iree_hal_device_group_t* device_group = NULL;
};

TEST_F(SlabProviderTest, CapturesAccessBeforeNativeAcquisition) {
  iree_hal_amdgpu_logical_device_options_t device_options;
  iree_hal_amdgpu_logical_device_options_initialize(&device_options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&device_options, &libhsa_, &topology_,
                                        host_allocator_));
  auto* physical_device = test_device.device()->physical_devices[0];
  auto* queue = iree_hal_device_queue(test_device.hal_device(), 0, 0);
  const auto family_affinity = iree_hal_make_queue_family_affinity(0);
  for (auto memory_pool :
       {physical_device->coarse_block_pools.large.memory_pool,
        physical_device->host_memory_pools.fine_pool}) {
    iree_hal_amdgpu_slab_provider_memory_pool_properties_t properties;
    IREE_ASSERT_OK(iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
        &libhsa_, memory_pool, &properties));
    iree_hal_amdgpu_access_agent_list_t agents;
    IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_memory_agents(
        &topology_, family_affinity, &agents));
    iree_hal_amdgpu_slab_provider_options_t options = {
        .memory_pool = memory_pool,
    };
    options.owned.memory_type = properties.memory_type;
    options.owned.supported_usage = properties.supported_usage;
    options.exposed.memory_type = properties.memory_type;
    options.exposed.supported_usage = properties.supported_usage;
    options.access.queue_family_affinity = family_affinity;
    options.access.agent_count = agents.count;
    options.access.agents = agents.values;
    IREE_ASSERT_OK(iree_hal_amdgpu_atomic_memory_query_source_masks(
        &libhsa_, &topology_, memory_pool, HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
        &options.access.atomic_source_masks));
    options.exposed.atomic_operations =
        iree_hal_amdgpu_atomic_memory_expand_capabilities(
            iree_hal_amdgpu_atomic_memory_select_device_cells(
                &options.access.atomic_source_masks, family_affinity));
    iree_hal_slab_provider_t* provider = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_slab_provider_create(
        test_device.hal_device(), /*import_allocator=*/nullptr, &libhsa_,
        options, 0, &physical_device->materialized_buffer_pool,
        IREE_SV("captured-access"), host_allocator_, &provider));
    // Construction inputs can be reused before the first payload acquisition.
    memset(&agents, 0, sizeof(agents));
    memset(&options, 0, sizeof(options));
    iree_hal_slab_t slab = {};
    IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(
        provider, 128, /*min_alignment=*/1, &slab));
    std::array<uint32_t, 32> values;
    values.fill(0xA5A5A5A5u);
    IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&libhsa_), slab.base_ptr,
                                        values.data(), sizeof(values)));
    iree::hal::cts::Ref<iree_hal_buffer_t> view;
    IREE_ASSERT_OK(iree_hal_slab_provider_wrap_buffer(
        provider, &slab, 32, 64, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_TRANSFER, iree_hal_buffer_release_callback_null(),
        view.out()));
    iree::hal::cts::SemaphoreList filled(test_device.hal_device(), {0}, {1});
    const uint32_t pattern = 0x12345678u;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queue, iree_hal_semaphore_list_empty(), filled, view, 0, 64, &pattern,
        sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(filled, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&libhsa_), values.data(),
                                        slab.base_ptr, sizeof(values)));
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(values[i], i >= 8 && i < 24 ? pattern : 0xA5A5A5A5u);
    }
    view.reset();
    iree_hal_slab_provider_release_slab_sync(provider, &slab);
    iree_hal_slab_provider_release(provider);
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
