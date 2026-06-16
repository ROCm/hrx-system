// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/info.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static void CountingReleaseCallback(void* user_data, iree_hal_buffer_t*) {
  ++*static_cast<int*>(user_data);
}

static iree_status_t QueueFillAndWait(iree_hal_device_t* device,
                                      iree_hal_buffer_t* target_buffer,
                                      iree_device_size_t length,
                                      const void* pattern,
                                      iree_host_size_t pattern_length) {
  iree::hal::cts::SemaphoreList empty_wait;
  iree::hal::cts::SemaphoreList fill_signal(device, {0}, {1});
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, fill_signal,
      target_buffer, /*target_offset=*/0, length, pattern, pattern_length,
      IREE_HAL_FILL_FLAG_NONE));
  return iree_hal_semaphore_list_wait(fill_signal, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t AllocateAndExportDevicePointer(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer,
    uint64_t* out_ptr) {
  *out_buffer = nullptr;
  *out_ptr = 0;

  iree_hal_buffer_t* buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, &buffer);
  iree_hal_external_buffer_t external_buffer = {};
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_export_buffer(
        allocator, buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
    *out_ptr = external_buffer.handle.device_allocation.ptr;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_hal_device_asan_observation_t QueryAsanObservation(
    iree_hal_device_t* device) {
  iree_hal_device_observation_t observation = {};
  IREE_EXPECT_OK(iree_hal_device_sample_observation(
      device, IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER, &observation));
  EXPECT_TRUE(iree_all_bits_set(observation.provided_flags,
                                IREE_HAL_DEVICE_OBSERVATION_FLAG_SANITIZER));
  EXPECT_EQ(IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_ALL,
            observation.sanitizer.asan.flags);
  return observation.sanitizer.asan;
}

static iree_hal_device_asan_spec_t QueryAsanSpec(iree_hal_device_t* device) {
  const iree_hal_device_sanitizer_spec_t* sanitizer =
      iree_hal_device_spec_sanitizer(iree_hal_device_spec(device));
  EXPECT_TRUE(
      iree_all_bits_set(sanitizer->flags, IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN));
  EXPECT_TRUE(
      iree_hal_asan_pool_options_is_enabled(&sanitizer->asan.pool_options));
  return sanitizer->asan;
}

static iree_hal_amdgpu_shadow_map_slab_t* FindShadowMapSlab(
    iree_hal_amdgpu_shadow_map_t* shadow_map, uint64_t slab_index) {
  for (iree_host_size_t i = 0; i < shadow_map->slab_count; ++i) {
    if (shadow_map->slabs[i].index == slab_index) {
      return &shadow_map->slabs[i];
    }
  }
  return nullptr;
}

static iree_device_size_t OversizedAllocationSize(
    const iree_hal_amdgpu_logical_device_t* logical_device) {
  const iree_device_size_t tlsf_range_length =
      logical_device->physical_devices[0]
          ->default_pool_options.tlsf_options.range_length;
  return tlsf_range_length + 1;
}

class AllocatorTest : public ::testing::Test {
 protected:
  class HsaAllocation {
   public:
    explicit HsaAllocation(const iree_hal_amdgpu_libhsa_t* libhsa)
        : libhsa_(libhsa) {}

    ~HsaAllocation() { Reset(); }

    iree_status_t Allocate(hsa_amd_memory_pool_t memory_pool,
                           iree_device_size_t size) {
      Reset();
      return iree_hsa_amd_memory_pool_allocate(
          IREE_LIBHSA(libhsa_), memory_pool, (size_t)size,
          HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &ptr_);
    }

    void Reset() {
      if (!ptr_) return;
      iree_hal_amdgpu_hsa_cleanup_assert_success(
          iree_hsa_amd_memory_pool_free_raw(libhsa_, ptr_));
      ptr_ = nullptr;
    }

    void* ptr() const { return ptr_; }

   private:
    const iree_hal_amdgpu_libhsa_t* libhsa_ = nullptr;
    void* ptr_ = nullptr;
  };

  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_system_info_query(&libhsa_, &system_info_));
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

  class TestLogicalDevice {
   public:
    ~TestLogicalDevice() {
      iree_hal_device_release(base_device_);
      iree_hal_device_group_release(device_group_);
    }

    iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* libhsa,
                             const iree_hal_amdgpu_topology_t* topology,
                             iree_allocator_t host_allocator) {
      iree_hal_amdgpu_logical_device_options_t options;
      iree_hal_amdgpu_logical_device_options_initialize(&options);
      return InitializeWithOptions(&options, libhsa, topology, host_allocator);
    }

    iree_status_t InitializeWithOptions(
        const iree_hal_amdgpu_logical_device_options_t* options,
        const iree_hal_amdgpu_libhsa_t* libhsa,
        const iree_hal_amdgpu_topology_t* topology,
        iree_allocator_t host_allocator) {
      IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
          IREE_SV("amdgpu"), options, libhsa, topology,
          create_context_.params(), host_allocator, &base_device_));
      return iree_hal_device_group_create_from_device(
          base_device_, create_context_.frontier_tracker(), host_allocator,
          &device_group_);
    }

    iree_hal_allocator_t* allocator() const {
      return iree_hal_device_allocator(base_device_);
    }

    iree_hal_device_t* device() const { return base_device_; }

    iree_hal_amdgpu_logical_device_t* logical_device() const {
      return (iree_hal_amdgpu_logical_device_t*)base_device_;
    }

   private:
    // Creation context supplying the proactor pool and frontier tracker.
    iree::hal::cts::DeviceCreateContext create_context_;

    // Test-owned device reference released before the topology-owning group.
    iree_hal_device_t* base_device_ = NULL;

    // Device group that owns the topology assigned to |base_device_|.
    iree_hal_device_group_t* device_group_ = NULL;
  };

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_system_info_t system_info_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t AllocatorTest::host_allocator_;
iree_hal_amdgpu_libhsa_t AllocatorTest::libhsa_;
iree_hal_amdgpu_system_info_t AllocatorTest::system_info_;
iree_hal_amdgpu_topology_t AllocatorTest::topology_;

TEST_F(AllocatorTest, AsanStateReservesDefaultShadowMapWhenEnabled) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  iree_hal_amdgpu_asan_state_t* asan_state =
      &test_device.logical_device()->asan;
  EXPECT_TRUE(iree_hal_amdgpu_asan_state_is_enabled(asan_state));

  iree_hal_amdgpu_shadow_map_t* shadow_map =
      iree_hal_amdgpu_asan_state_shadow_map(asan_state);
  ASSERT_NE(shadow_map, nullptr);
  EXPECT_EQ(shadow_map->shadow_scale_shift, options.asan.shadow_scale_shift);
  EXPECT_EQ(shadow_map->reservation_size, options.asan.shadow_size);
  EXPECT_EQ(shadow_map->application_window_base, 0u);
  EXPECT_EQ(shadow_map->application_window_size,
            options.asan.shadow_size << options.asan.shadow_scale_shift);
  EXPECT_GE(shadow_map->slab_size, options.asan.shadow_slab_size);
  EXPECT_TRUE(iree_device_size_is_power_of_two(shadow_map->slab_size));
  EXPECT_EQ(shadow_map->mapping_mode,
            IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_SPARSE);
  EXPECT_EQ(shadow_map->hsa.memory_type,
            IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_DEFAULT);
  EXPECT_EQ(shadow_map->initial_slab_value, 0xFAu);
  EXPECT_LE(shadow_map->slab_count,
            shadow_map->reservation_size / shadow_map->slab_size);
  EXPECT_NE(asan_state->owned_application_base_ptr, nullptr);
  EXPECT_EQ(asan_state->owned_application_size,
            options.asan.owned_application_size);
  ASSERT_NE(asan_state->application_free_ranges, nullptr);
  const uint64_t owned_application_base =
      reinterpret_cast<uintptr_t>(asan_state->owned_application_base_ptr);
  const uint64_t owned_application_end =
      owned_application_base + asan_state->owned_application_size;
  iree_device_size_t free_length = 0;
  uint64_t previous_range_end = owned_application_base;
  for (const iree_hal_amdgpu_asan_application_range_t* free_range =
           asan_state->application_free_ranges;
       free_range; free_range = free_range->next) {
    ASSERT_GE(free_range->address, owned_application_base);
    ASSERT_GE(free_range->address, previous_range_end);
    ASSERT_LE(free_range->address, owned_application_end);
    ASSERT_LE(free_range->length, owned_application_end - free_range->address);
    ASSERT_LE(free_length,
              asan_state->owned_application_size - free_range->length);
    free_length += free_range->length;
    previous_range_end = free_range->address + free_range->length;
  }
  ASSERT_LE(free_length, asan_state->owned_application_size);
  const iree_device_size_t reserved_application_length =
      asan_state->owned_application_size - free_length;
  if (reserved_application_length > 0) {
    EXPECT_GT(shadow_map->slab_count, 0u);
  }

  iree_hal_amdgpu_asan_config_t config;
  iree_hal_amdgpu_asan_state_populate_config(asan_state, &config);
  EXPECT_EQ(config.record_length, sizeof(config));
  EXPECT_EQ(config.abi_version, IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0);
  EXPECT_EQ(config.flags, IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED);
  EXPECT_EQ(config.shadow_scale_shift, shadow_map->shadow_scale_shift);
  EXPECT_EQ(config.shadow_base, shadow_map->shadow_base);
  EXPECT_EQ(config.application_window_base,
            shadow_map->application_window_base);
  EXPECT_EQ(config.application_window_size,
            shadow_map->application_window_size);
  EXPECT_EQ(config.shadow_size, shadow_map->reservation_size);
  EXPECT_EQ(config.shadow_slab_size, shadow_map->slab_size);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(config.reserved); ++i) {
    EXPECT_EQ(config.reserved[i], 0u);
  }
}

TEST_F(AllocatorTest, AsanHostLocalShadowBackingMapsPinnedHostSlabs) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.shadow_backing = IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_HOST_LOCAL;
  options.asan.shadow_slab_size = 2 * 1024 * 1024;
  options.asan.quarantine_size = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  iree_hal_amdgpu_asan_state_t* asan_state =
      &test_device.logical_device()->asan;
  iree_hal_amdgpu_shadow_map_t* shadow_map =
      iree_hal_amdgpu_asan_state_shadow_map(asan_state);
  ASSERT_NE(shadow_map, nullptr);
  EXPECT_EQ(shadow_map->hsa.memory_type,
            IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST);
  EXPECT_EQ(shadow_map->hsa.memory_pool.handle,
            test_device.logical_device()
                ->physical_devices[0]
                ->coarse_block_pools.large.memory_pool.handle);

  const iree_host_size_t initial_slab_count = shadow_map->slab_count;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_map_slab(shadow_map, 0));
  ASSERT_GE(shadow_map->slab_count, initial_slab_count);

  iree_hal_amdgpu_shadow_map_slab_t* shadow_slab =
      FindShadowMapSlab(shadow_map, 0);
  ASSERT_NE(shadow_slab, nullptr);

  constexpr uint8_t kHeapRedzoneShadowValue = 0xFAu;
  uint8_t shadow_byte = 0;
  IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&libhsa_), &shadow_byte,
                                      shadow_slab->base_ptr,
                                      sizeof(shadow_byte)));
  EXPECT_EQ(shadow_byte, kHeapRedzoneShadowValue);
}

TEST_F(AllocatorTest,
       AsanDeviceLocalAllocationQuarantinesReleasedApplicationRange) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_affinity = 1ull;

  iree_hal_buffer_t* first_buffer = nullptr;
  uint64_t first_ptr = 0;
  IREE_ASSERT_OK(AllocateAndExportDevicePointer(test_device.allocator(), params,
                                                /*allocation_size=*/4096,
                                                &first_buffer, &first_ptr));
  iree_hal_buffer_release(first_buffer);

  iree_hal_buffer_t* second_buffer = nullptr;
  uint64_t second_ptr = 0;
  IREE_ASSERT_OK(AllocateAndExportDevicePointer(test_device.allocator(), params,
                                                /*allocation_size=*/4096,
                                                &second_buffer, &second_ptr));
  EXPECT_NE(second_ptr, first_ptr);
  iree_hal_buffer_release(second_buffer);
}

TEST_F(
    AllocatorTest,
    AsanDeviceLocalAllocationReusesReleasedApplicationRangeWhenUnquarantined) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.quarantine_size = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_affinity = 1ull;

  iree_hal_buffer_t* first_buffer = nullptr;
  uint64_t first_ptr = 0;
  IREE_ASSERT_OK(AllocateAndExportDevicePointer(test_device.allocator(), params,
                                                /*allocation_size=*/4096,
                                                &first_buffer, &first_ptr));
  iree_hal_buffer_release(first_buffer);

  iree_hal_buffer_t* second_buffer = nullptr;
  uint64_t second_ptr = 0;
  IREE_ASSERT_OK(AllocateAndExportDevicePointer(test_device.allocator(), params,
                                                /*allocation_size=*/4096,
                                                &second_buffer, &second_ptr));
  EXPECT_EQ(second_ptr, first_ptr);
  iree_hal_buffer_release(second_buffer);
}

TEST_F(AllocatorTest, AsanDiagnosticsExposeDefaultQuarantineRetention) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.default_pool.range_length = 4096;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  const iree_hal_device_asan_spec_t asan_spec =
      QueryAsanSpec(test_device.device());
  EXPECT_EQ(asan_spec.pool_options.quarantine_size,
            options.asan.quarantine_size);

  iree_hal_device_asan_observation_t asan =
      QueryAsanObservation(test_device.device());
  EXPECT_EQ(asan.quarantine_size, 0u);
  EXPECT_EQ(asan.quarantine_eviction_count, 0u);
  const uint64_t initial_shadow_mapped_slab_count =
      asan.shadow_mapped_slab_count;
  const uint64_t initial_shadow_committed_size = asan.shadow_committed_size;

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_affinity = 1ull;

  iree_hal_buffer_t* buffer = nullptr;
  uint64_t ptr = 0;
  const iree_device_size_t allocation_size =
      OversizedAllocationSize(test_device.logical_device());
  IREE_ASSERT_OK(AllocateAndExportDevicePointer(
      test_device.allocator(), params, allocation_size, &buffer, &ptr));
  iree_hal_buffer_release(buffer);

  asan = QueryAsanObservation(test_device.device());
  EXPECT_GT(asan.quarantine_size, 0u);
  EXPECT_EQ(asan.quarantine_eviction_count, 0u);
  EXPECT_GE(asan.shadow_mapped_slab_count, initial_shadow_mapped_slab_count);
  EXPECT_GE(asan.shadow_committed_size, initial_shadow_committed_size);
}

TEST_F(AllocatorTest, AsanDiagnosticsExposeZeroQuarantineRelease) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.quarantine_size = 0;
  options.default_pool.range_length = 4096;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));
  const iree_hal_device_asan_spec_t asan_spec =
      QueryAsanSpec(test_device.device());
  EXPECT_EQ(asan_spec.pool_options.quarantine_size, 0u);

  iree_hal_device_asan_observation_t asan =
      QueryAsanObservation(test_device.device());
  const uint64_t initial_quarantine_eviction_count =
      asan.quarantine_eviction_count;
  const uint64_t initial_shadow_mapped_slab_count =
      asan.shadow_mapped_slab_count;
  const uint64_t initial_shadow_committed_size = asan.shadow_committed_size;

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_affinity = 1ull;

  iree_hal_buffer_t* buffer = nullptr;
  uint64_t ptr = 0;
  const iree_device_size_t allocation_size =
      OversizedAllocationSize(test_device.logical_device());
  IREE_ASSERT_OK(AllocateAndExportDevicePointer(
      test_device.allocator(), params, allocation_size, &buffer, &ptr));
  iree_hal_buffer_release(buffer);

  asan = QueryAsanObservation(test_device.device());
  EXPECT_EQ(asan.quarantine_size, 0u);
  EXPECT_EQ(asan.quarantine_eviction_count,
            initial_quarantine_eviction_count + 1);
  EXPECT_GE(asan.shadow_mapped_slab_count, initial_shadow_mapped_slab_count);
  EXPECT_GE(asan.shadow_committed_size, initial_shadow_committed_size);
}

TEST_F(AllocatorTest, QueryMemoryHeapsReportsHsaLimits) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_host_size_t heap_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_allocator_query_memory_heaps(
                            test_device.allocator(),
                            /*capacity=*/0, /*heaps=*/NULL, &heap_count));
  ASSERT_GE(heap_count, 2u);
  ASSERT_LE(heap_count, 3u);

  std::array<iree_hal_allocator_memory_heap_t, 3> heaps;
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      test_device.allocator(), heaps.size(), heaps.data(), &heap_count));
  ASSERT_GE(heap_count, 2u);
  ASSERT_LE(heap_count, heaps.size());

  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    const iree_hal_allocator_memory_heap_t& heap = heaps[i];
    EXPECT_NE(heap.max_allocation_size, 0u);
    EXPECT_NE(heap.max_allocation_size, ~(iree_device_size_t)0);
    EXPECT_NE(heap.min_alignment, 0u);
    EXPECT_TRUE(iree_device_size_is_power_of_two(heap.min_alignment));
  }
}

TEST_F(AllocatorTest, OversizedAllocationIsRejectedByCompatibility) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  std::array<iree_hal_allocator_memory_heap_t, 3> heaps;
  iree_host_size_t heap_count = 0;
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      test_device.allocator(), heaps.size(), heaps.data(), &heap_count));
  ASSERT_GE(heap_count, 2u);
  ASSERT_LE(heap_count, heaps.size());
  ASSERT_LT(heaps[0].max_allocation_size, ~(iree_device_size_t)0);

  iree_device_size_t oversized_allocation_size = 0;
  ASSERT_TRUE(iree_device_size_checked_add(heaps[0].max_allocation_size, 1,
                                           &oversized_allocation_size));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, oversized_allocation_size,
          &resolved_params, &resolved_allocation_size);
  EXPECT_EQ(compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorTest, DeviceLocalHostVisibleMemoryIsLowPerformance) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH |
                 IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, /*allocation_size=*/4096,
          &resolved_params, &resolved_allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    GTEST_SKIP() << "device-local host-visible memory is not available";
  }
  EXPECT_TRUE(iree_all_bits_set(
      compatibility, IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
                         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH |
                         IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE));
  EXPECT_TRUE(iree_all_bits_set(resolved_params.type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));

  iree_hal_buffer_params_t preferred_params = {0};
  preferred_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL |
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  preferred_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                           IREE_HAL_BUFFER_USAGE_DISPATCH |
                           IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  iree_hal_buffer_params_t resolved_preferred_params = {0};
  iree_device_size_t resolved_preferred_allocation_size = 0;
  iree_hal_buffer_compatibility_t preferred_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), preferred_params, /*allocation_size=*/4096,
          &resolved_preferred_params, &resolved_preferred_allocation_size);
  EXPECT_TRUE(
      iree_all_bits_set(preferred_compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                            IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
                            IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH |
                            IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE));
  EXPECT_TRUE(iree_all_bits_set(resolved_preferred_params.type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST_F(AllocatorTest, SuppressDeviceFineMemoryRetainsMappedHostMemory) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.suppress_device_fine_memory = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t device_visible_params = {0};
  device_visible_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_visible_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_params_t resolved_device_visible_params = {0};
  iree_device_size_t resolved_device_visible_allocation_size = 0;
  iree_hal_buffer_compatibility_t device_visible_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), device_visible_params,
          /*allocation_size=*/4096, &resolved_device_visible_params,
          &resolved_device_visible_allocation_size);
  EXPECT_EQ(device_visible_compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);

  iree_hal_buffer_params_t host_visible_params = {0};
  host_visible_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL |
                             IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                             IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_visible_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_params_t resolved_host_visible_params = {0};
  iree_device_size_t resolved_host_visible_allocation_size = 0;
  iree_hal_buffer_compatibility_t host_visible_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), host_visible_params,
          /*allocation_size=*/4096, &resolved_host_visible_params,
          &resolved_host_visible_allocation_size);
  EXPECT_TRUE(
      iree_all_bits_set(host_visible_compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                            IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER));
  EXPECT_TRUE(iree_all_bits_set(
      resolved_host_visible_params.type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  EXPECT_TRUE(iree_all_bits_set(resolved_host_visible_params.usage,
                                IREE_HAL_BUFFER_USAGE_MAPPING));
  EXPECT_FALSE(iree_all_bits_set(resolved_host_visible_params.type,
                                 IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      test_device.allocator(), resolved_host_visible_params,
      resolved_host_visible_allocation_size, &buffer));
  EXPECT_TRUE(iree_all_bits_set(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  EXPECT_FALSE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                 IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  iree_hal_buffer_release(buffer);
}

TEST_F(AllocatorTest, OverAlignedAllocationIsRejected) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  std::array<iree_hal_allocator_memory_heap_t, 3> heaps;
  iree_host_size_t heap_count = 0;
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      test_device.allocator(), heaps.size(), heaps.data(), &heap_count));
  ASSERT_GE(heap_count, 2u);
  ASSERT_LE(heap_count, heaps.size());

  const iree_device_size_t over_alignment =
      ~(iree_device_size_t)0 ^ (~(iree_device_size_t)0 >> 1);
  ASSERT_TRUE(iree_device_size_is_power_of_two(over_alignment));
  ASSERT_GT(over_alignment, heaps[0].min_alignment);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = over_alignment;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, /*allocation_size=*/1,
          &resolved_params, &resolved_allocation_size);
  EXPECT_EQ(compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);

  iree_hal_buffer_t* buffer = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_allocate_buffer(test_device.allocator(), params,
                                         /*allocation_size=*/1, &buffer));
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(AllocatorTest, UnsupportedExternalBufferImportsFailLoud) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  std::array<iree_hal_external_buffer_type_t, 2> unsupported_types = {
      IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32,
  };
  for (iree_hal_external_buffer_type_t unsupported_type : unsupported_types) {
    iree_hal_external_buffer_t external_buffer = {};
    external_buffer.type = unsupported_type;
    external_buffer.size = 4096;

    iree_hal_buffer_t* buffer = NULL;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNIMPLEMENTED,
        iree_hal_allocator_import_buffer(
            test_device.allocator(), params, &external_buffer,
            iree_hal_buffer_release_callback_null(), &buffer));
    EXPECT_EQ(buffer, nullptr);
  }
}

TEST_F(AllocatorTest, AmdgpuDeviceSpecExposesRepresentativePhysicalFacts) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(test_device.device());
  ASSERT_NE(device_spec, nullptr);

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(iree_string_view_equal(identity->driver_id, IREE_SV("amdgpu")));
  EXPECT_TRUE(iree_string_view_equal(identity->backend_id, IREE_SV("hsa")));
  ASSERT_EQ(identity->physical_device_count, topology_.gpu_agent_count);
  ASSERT_GT(identity->physical_device_count, 0u);
  EXPECT_TRUE(
      iree_all_bits_set(identity->physical_devices[0].identity.flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS |
                            IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE));

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  ASSERT_NE(memory, nullptr);
  if (system_info_.dmabuf_supported) {
    iree_hal_external_buffer_handle_selection_t selection = {
        /*.handle_type_mask=*/IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF,
        /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
            IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
        /*.buffer_usage=*/
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
        /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_NONE,
        /*.compatible_memory_type_mask=*/UINT32_MAX,
        /*.capability_flags=*/
        IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS |
            IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_OWNING,
    };
    EXPECT_NE(iree_hal_device_spec_find_external_buffer_handle(device_spec,
                                                               &selection),
              nullptr);
  } else {
    EXPECT_EQ(memory->external_buffer_handle_count, 0u);
  }

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_GT(dispatch->execution.unit_count, 0u);
  EXPECT_EQ(dispatch->execution.group_count, topology_.gpu_agent_count);
  EXPECT_TRUE(dispatch->subgroup.default_size == 32 ||
              dispatch->subgroup.default_size == 64);
  EXPECT_EQ(dispatch->subgroup.default_size, dispatch->subgroup.minimum_size);
  EXPECT_EQ(dispatch->subgroup.default_size, dispatch->subgroup.maximum_size);

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_NE(executables, nullptr);
  ASSERT_GT(executables->target_count, 0u);
  EXPECT_TRUE(iree_string_view_equal(executables->targets[0].family,
                                     IREE_SV("amdgpu")));
  EXPECT_FALSE(iree_string_view_is_empty(executables->targets[0].processor));
  EXPECT_FALSE(
      iree_string_view_is_empty(executables->targets[0].loader_target));
}

TEST_F(AllocatorTest, AmdgpuDeviceSpecAllowsCompositeDevices) {
  if (topology_.gpu_agent_count < 2) {
    GTEST_SKIP() << "requires a composite logical device";
  }

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(test_device.device());
  ASSERT_NE(device_spec, nullptr);

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  ASSERT_NE(identity, nullptr);
  ASSERT_EQ(identity->physical_device_count, topology_.gpu_agent_count);
  for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
    const iree_hal_physical_device_spec_t* physical_device =
        &identity->physical_devices[i];
    EXPECT_EQ(physical_device->physical_ordinal, i);
    EXPECT_EQ(physical_device->partition_count, 1u);
    EXPECT_EQ(physical_device->physical_device_affinity, 1ull << i);
    EXPECT_TRUE(iree_all_bits_set(
        physical_device->identity.flags,
        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS |
            IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE));
  }

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->execution.group_count, topology_.gpu_agent_count);
}

TEST_F(AllocatorTest, DeviceAllocationImportRejectsUnknownPointer) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.queue_affinity = 1ull;

  uint32_t host_storage = 0;
  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
  external_buffer.size = sizeof(host_storage);
  external_buffer.handle.device_allocation.ptr =
      (uint64_t)(uintptr_t)&host_storage;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_allocator_import_buffer(
                            test_device.allocator(), params, &external_buffer,
                            iree_hal_buffer_release_callback_null(), &buffer));
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(AllocatorTest, DeviceAllocationImportWrapsHsaAllocation) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  hsa_amd_memory_pool_t memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa_, topology_.gpu_agents[0], &memory_pool));

  constexpr iree_device_size_t kAllocationSize = 4096;
  HsaAllocation allocation(&libhsa_);
  IREE_ASSERT_OK(allocation.Allocate(memory_pool, kAllocationSize));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_affinity = 1ull;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, kAllocationSize, &resolved_params,
          &resolved_allocation_size);
  EXPECT_TRUE(iree_all_bits_set(
      compatibility, IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE |
                         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
                         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH));

  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
  external_buffer.size = kAllocationSize;
  external_buffer.handle.device_allocation.ptr =
      (uint64_t)(uintptr_t)allocation.ptr();

  int release_count = 0;
  iree_hal_buffer_release_callback_t callback = {};
  callback.fn = CountingReleaseCallback;
  callback.user_data = &release_count;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      test_device.allocator(), params, &external_buffer, callback, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), kAllocationSize);

  iree_hal_external_buffer_t exported_buffer = {};
  IREE_ASSERT_OK(iree_hal_allocator_export_buffer(
      test_device.allocator(), buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &exported_buffer));
  EXPECT_EQ(exported_buffer.type,
            IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION);
  EXPECT_EQ(exported_buffer.size, kAllocationSize);
  EXPECT_EQ(exported_buffer.handle.device_allocation.ptr,
            external_buffer.handle.device_allocation.ptr);

  constexpr uint32_t kPattern = 0xACCE551u;
  IREE_ASSERT_OK(QueueFillAndWait(test_device.device(), buffer, kAllocationSize,
                                  &kPattern, sizeof(kPattern)));
  std::array<uint32_t, kAllocationSize / sizeof(uint32_t)> observed = {};
  IREE_ASSERT_OK(iree_hsa_memory_copy(IREE_LIBHSA(&libhsa_), observed.data(),
                                      allocation.ptr(), kAllocationSize));
  for (uint32_t value : observed) {
    EXPECT_EQ(value, kPattern);
  }

  iree_hal_buffer_release(buffer);
  EXPECT_EQ(release_count, 1);
}

TEST_F(AllocatorTest, AsanDeviceAllocationImportPublishesShadow) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  hsa_amd_memory_pool_t memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa_, topology_.gpu_agents[0], &memory_pool));

  constexpr iree_device_size_t kAllocationSize = 4096;
  HsaAllocation allocation(&libhsa_);
  IREE_ASSERT_OK(allocation.Allocate(memory_pool, kAllocationSize));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_affinity = 1ull;

  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
  external_buffer.size = kAllocationSize;
  external_buffer.handle.device_allocation.ptr =
      (uint64_t)(uintptr_t)allocation.ptr();

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      test_device.allocator(), params, &external_buffer,
      iree_hal_buffer_release_callback_null(), &buffer));
  ASSERT_NE(buffer, nullptr);

  iree_hal_amdgpu_asan_state_t* asan_state =
      &test_device.logical_device()->asan;
  iree_hal_amdgpu_shadow_map_t* shadow_map =
      iree_hal_amdgpu_asan_state_shadow_map(asan_state);
  ASSERT_NE(shadow_map, nullptr);

  iree_hal_amdgpu_shadow_map_range_t shadow_range;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_calculate_range(
      shadow_map, external_buffer.handle.device_allocation.ptr,
      external_buffer.size, &shadow_range));
  ASSERT_GT(shadow_range.shadow_length, 0u);

  std::vector<uint8_t> shadow_bytes(shadow_range.shadow_length);
  IREE_ASSERT_OK(iree_hsa_memory_copy(
      IREE_LIBHSA(&libhsa_), shadow_bytes.data(),
      (void*)(uintptr_t)shadow_range.shadow_address, shadow_bytes.size()));
  for (uint8_t shadow_byte : shadow_bytes) {
    EXPECT_EQ(shadow_byte, 0u);
  }

  constexpr uint8_t kHeapRedzoneShadowValue = 0xFAu;
  bool checked_neighbor = false;
  const iree_device_size_t slab_begin_offset =
      shadow_range.first_slab_index * shadow_map->slab_size;
  if (shadow_range.shadow_offset > slab_begin_offset) {
    uint8_t previous_shadow_byte = 0;
    IREE_ASSERT_OK(iree_hsa_memory_copy(
        IREE_LIBHSA(&libhsa_), &previous_shadow_byte,
        (void*)(uintptr_t)(shadow_range.shadow_address - 1),
        sizeof(previous_shadow_byte)));
    EXPECT_EQ(previous_shadow_byte, kHeapRedzoneShadowValue);
    checked_neighbor = true;
  }
  const iree_device_size_t slab_end_offset =
      slab_begin_offset + shadow_map->slab_size;
  if (shadow_range.shadow_offset + shadow_range.shadow_length <
      slab_end_offset) {
    uint8_t next_shadow_byte = 0;
    IREE_ASSERT_OK(
        iree_hsa_memory_copy(IREE_LIBHSA(&libhsa_), &next_shadow_byte,
                             (void*)(uintptr_t)(shadow_range.shadow_address +
                                                shadow_range.shadow_length),
                             sizeof(next_shadow_byte)));
    EXPECT_EQ(next_shadow_byte, kHeapRedzoneShadowValue);
    checked_neighbor = true;
  }
  EXPECT_TRUE(checked_neighbor);

  iree_hal_buffer_release(buffer);
}

TEST_F(AllocatorTest, AsanPremappedShadowModeCoversUnpublishedAddresses) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.shadow_mode = IREE_HAL_AMDGPU_ASAN_SHADOW_MODE_PREMAPPED;
  options.asan.shadow_scale_shift = IREE_HAL_AMDGPU_ASAN_MAX_SHADOW_SCALE_SHIFT;
  options.asan.shadow_size = (iree_device_size_t)1ull << 40;
  options.asan.quarantine_size = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithOptions(
      &options, &libhsa_, &topology_, host_allocator_));

  iree_hal_amdgpu_asan_state_t* asan_state =
      &test_device.logical_device()->asan;
  iree_hal_amdgpu_shadow_map_t* shadow_map =
      iree_hal_amdgpu_asan_state_shadow_map(asan_state);
  ASSERT_NE(shadow_map, nullptr);
  EXPECT_EQ(shadow_map->mapping_mode,
            IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_PREMAPPED);
  EXPECT_NE(shadow_map->hsa.alias_allocation_handle.handle, 0u);

  iree_hal_amdgpu_shadow_map_range_t shadow_range;
  constexpr uint64_t kUnpublishedApplicationAddress = 0x100000000ull;
  IREE_ASSERT_OK(iree_hal_amdgpu_shadow_map_calculate_range(
      shadow_map, kUnpublishedApplicationAddress, /*application_length=*/1,
      &shadow_range));
  ASSERT_EQ(shadow_range.shadow_length, 1u);

  constexpr uint8_t kHeapRedzoneShadowValue = 0xFAu;
  uint8_t shadow_byte = 0;
  IREE_ASSERT_OK(iree_hsa_memory_copy(
      IREE_LIBHSA(&libhsa_), &shadow_byte,
      (void*)(uintptr_t)shadow_range.shadow_address, sizeof(shadow_byte)));
  EXPECT_EQ(shadow_byte, kHeapRedzoneShadowValue);
}

TEST_F(AllocatorTest, DeviceAllocationExportReportsHsaPointer) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.queue_affinity = 1ull;

  iree_hal_buffer_params_t resolved_params = {0};
  iree_device_size_t resolved_allocation_size = 0;
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          test_device.allocator(), params, /*allocation_size=*/4096,
          &resolved_params, &resolved_allocation_size);
  EXPECT_TRUE(iree_all_bits_set(compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                                    IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE));

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      test_device.allocator(), params, /*allocation_size=*/4096, &buffer));

  iree_hal_external_buffer_t external_buffer = {};
  IREE_ASSERT_OK(iree_hal_allocator_export_buffer(
      test_device.allocator(), buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(external_buffer.type,
            IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION);
  EXPECT_NE(external_buffer.handle.device_allocation.ptr, 0u);
  EXPECT_EQ(external_buffer.size, iree_hal_buffer_allocation_size(buffer));

  hsa_amd_pointer_info_t pointer_info = {};
  pointer_info.size = sizeof(pointer_info);
  IREE_ASSERT_OK(iree_hsa_amd_pointer_info(
      IREE_LIBHSA(&libhsa_),
      (const void*)(uintptr_t)external_buffer.handle.device_allocation.ptr,
      &pointer_info, /*alloc=*/NULL, /*num_agents_accessible=*/NULL,
      /*accessible=*/NULL));
  EXPECT_EQ(pointer_info.type, HSA_EXT_POINTER_TYPE_HSA);
  EXPECT_EQ(pointer_info.agentBaseAddress,
            (void*)(uintptr_t)external_buffer.handle.device_allocation.ptr);
  EXPECT_GE(pointer_info.sizeInBytes, external_buffer.size);

  constexpr uint32_t kPattern = 0xA110CA7Eu;
  IREE_ASSERT_OK(QueueFillAndWait(test_device.device(), buffer,
                                  external_buffer.size, &kPattern,
                                  sizeof(kPattern)));
  std::array<uint32_t, 4096 / sizeof(uint32_t)> observed = {};
  IREE_ASSERT_OK(iree_hsa_memory_copy(
      IREE_LIBHSA(&libhsa_), observed.data(),
      (const void*)(uintptr_t)external_buffer.handle.device_allocation.ptr,
      sizeof(observed)));
  for (uint32_t value : observed) {
    EXPECT_EQ(value, kPattern);
  }

  iree_hal_buffer_release(buffer);
}

TEST_F(AllocatorTest, ExternalBufferExportRejectsUnsupportedRequests) {
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      test_device.allocator(), params, /*allocation_size=*/4096, &buffer));

  iree_hal_external_buffer_t external_buffer = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_allocator_export_buffer(
          test_device.allocator(), buffer,
          IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
          IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_allocator_export_buffer(
          test_device.allocator(), buffer,
          IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32,
          IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));

  iree_hal_buffer_release(buffer);
}

}  // namespace
}  // namespace iree::hal::amdgpu
