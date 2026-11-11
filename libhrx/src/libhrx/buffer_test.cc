// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstring>
#include <vector>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CpuBufferTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  void SetUp() override {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
    device_ = hrx_device_group_get_device(group_, 0);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_create(hrx_device_get_queue(device_, 0, 0), 0, &stream_)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
            device_,
            hrx_queue_get_family_ordinal(hrx_device_get_queue(device_, 0, 0)),
            HRX_MEMORY_SITE_PROGRAM, &program_scope_)));
  }

  void TearDown() override {
    hrx_stream_release(stream_);
    hrx_device_group_release(group_);
    group_ = nullptr;
  }

  // Exact prepared program scope, independent of any allocation owner.
  hrx_memory_scope_t program_scope_ = {};
  // CPU device borrowed from the fixture group.
  hrx_device_t device_ = nullptr;
  // Stream providing real allocation and transfer ordering.
  hrx_stream_t stream_ = nullptr;
};

TEST_F(CpuBufferTest,
       AllocatedNativeAddressSurvivesScopedMapAndStreamTransfer) {
  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
      stream_, 64, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED |
          HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
      &buffer)));
  const uint8_t pattern = 0x5A;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
      stream_, buffer, 0, 64, &pattern, sizeof(pattern))));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint64_t device_pointer = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_query_address(
      buffer, program_scope_, HRX_BUFFER_INTERFACE_HOST, &device_pointer)));
  void* mapped_pointer = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_map(
      buffer, HRX_MAP_WRITE | HRX_MAP_MAY_ALIAS, 13, 23, &mapped_pointer)));
  EXPECT_EQ(
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(device_pointer)) + 13,
      mapped_pointer);
  memset(mapped_pointer, 0xA7, 23);
  uint64_t repeated_pointer = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_query_address(
      buffer, program_scope_, HRX_BUFFER_INTERFACE_HOST, &repeated_pointer)));
  EXPECT_EQ(device_pointer, repeated_pointer);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));

  uint8_t result[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h(
      hrx_device_get_queue(device_, 0, 0), buffer, 0, result, sizeof(result))));
  for (size_t i = 0; i < sizeof(result); ++i) {
    EXPECT_EQ(i >= 13 && i < 36 ? 0xA7 : pattern, result[i]);
  }
  hrx_buffer_release(buffer);
}

TEST_F(CpuBufferTest, PublicConstructionPreservesNativePointerAccess) {
  const hrx_buffer_usage_t usages[] = {
      0, HRX_BUFFER_USAGE_DEFAULT,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED};
  for (hrx_buffer_usage_t usage : usages) {
    for (int construction = 0; construction < 3; ++construction) {
      SCOPED_TRACE(usage);
      SCOPED_TRACE(construction);
      alignas(64) uint8_t storage[64];
      memset(storage, 0x5A, sizeof(storage));
      hrx_buffer_t buffer = nullptr;
      const hrx_buffer_params_t params = {
          .type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
          .access = HRX_MEMORY_ACCESS_ALL,
          .usage = usage,
      };
      if (construction == 0) {
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
            stream_, sizeof(storage), params.type, usage, &buffer)));
      } else if (construction == 1) {
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_), params, sizeof(storage), &buffer)));
      } else {
        IREE_ASSERT_OK(hrx_status_to_iree(
            hrx_allocator_import_buffer(hrx_device_allocator(device_), params,
                                        storage, sizeof(storage), &buffer)));
      }
      if (construction != 2) {
        IREE_ASSERT_OK(hrx_status_to_iree(
            hrx_synchronous_h2d(hrx_device_get_queue(device_, 0, 0), storage,
                                buffer, 0, sizeof(storage))));
      }
      uint64_t pointer = 0;
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_query_address(
          buffer, program_scope_, HRX_BUFFER_INTERFACE_HOST, &pointer)));
      ASSERT_NE(pointer, 0u);
      EXPECT_FALSE(buffer->is_mapped);
      EXPECT_EQ(
          0,
          memcmp(reinterpret_cast<const void*>(static_cast<uintptr_t>(pointer)),
                 storage, sizeof(storage)));
      if (construction == 2) {
        EXPECT_EQ(pointer, reinterpret_cast<uintptr_t>(storage));
      }

      void* mapped_pointer = nullptr;
      if (usage & HRX_BUFFER_USAGE_MAPPING_SCOPED) {
        IREE_ASSERT_OK(hrx_status_to_iree(
            hrx_buffer_map(buffer, HRX_MAP_READ | HRX_MAP_MAY_ALIAS, 13, 23,
                           &mapped_pointer)));
        EXPECT_EQ(
            reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(pointer)) + 13,
            mapped_pointer);
      }
      const iree_hal_buffer_mapping_t mapping = buffer->mapping;
      uint64_t repeated_pointer = 0;
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_query_address(
          buffer, program_scope_, HRX_BUFFER_INTERFACE_HOST,
          &repeated_pointer)));
      EXPECT_EQ(pointer, repeated_pointer);
      EXPECT_EQ(0, memcmp(&mapping, &buffer->mapping, sizeof(mapping)));
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));

      uint8_t result[64] = {};
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), buffer, 0,
                              result, sizeof(result))));
      EXPECT_EQ(0, memcmp(result, storage, sizeof(storage)));
      hrx_buffer_release(buffer);
      for (uint8_t value : storage) {
        EXPECT_EQ(0x5A, value);
      }
    }
  }
}

TEST_F(CpuBufferTest, AddressesRequireAnExactPreparedScopeAndInterface) {
  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_buffer_allocate(stream_, 64, HRX_MEMORY_TYPE_HOST_LOCAL,
                          HRX_BUFFER_USAGE_DEFAULT, &buffer)));
  // Keep a real second domain live; a forged pointer cannot establish whether
  // the query incorrectly follows an allocation's implicit owner instead.
  hrx_runtime_t runtime = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
  const hrx_device_create_params_t params = {.uri = "task://"};
  hrx_device_group_t group = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_device_group_create(runtime, 1, &params, &group)));
  hrx_memory_scope_t foreign_scope;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
      hrx_device_group_get_device(group, 0), 0, HRX_MEMORY_SITE_PROGRAM,
      &foreign_scope)));
  for (const auto scope :
       {hrx_memory_scope_t{}, foreign_scope,
        hrx_memory_scope_t{program_scope_.domain, UINT32_MAX}}) {
    uint64_t address = UINT64_MAX;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        hrx_status_to_iree(hrx_buffer_query_address(
            buffer, scope, HRX_BUFFER_INTERFACE_HOST, &address)));
    EXPECT_EQ(0u, address);
  }
  for (auto type :
       {HRX_BUFFER_INTERFACE_DEVICE_ADDRESS, HRX_BUFFER_INTERFACE_XDNA_SHIM_DMA,
        HRX_BUFFER_INTERFACE_XDNA_FIRMWARE}) {
    uint64_t address = UINT64_MAX;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                          hrx_status_to_iree(hrx_buffer_query_address(
                              buffer, program_scope_, type, &address)));
    EXPECT_EQ(0u, address);
  }
  for (auto type :
       {HRX_BUFFER_INTERFACE_VULKAN_BUFFER, HRX_BUFFER_INTERFACE_RDMA,
        HRX_BUFFER_INTERFACE_REGISTERED_IO, HRX_BUFFER_INTERFACE_REMOTE,
        HRX_BUFFER_INTERFACE_WEBGPU_BUFFER, HRX_BUFFER_INTERFACE_MAPPING_SOURCE,
        static_cast<hrx_buffer_interface_t>(UINT16_MAX)}) {
    uint64_t address = UINT64_MAX;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          hrx_status_to_iree(hrx_buffer_query_address(
                              buffer, program_scope_, type, &address)));
    EXPECT_EQ(0u, address);
  }
  hrx_device_group_release(group);
  hrx_runtime_release(runtime);
  uint64_t address = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_query_address(
      buffer, program_scope_, HRX_BUFFER_INTERFACE_HOST, &address)));
  EXPECT_NE(0u, address);
  hrx_buffer_release(buffer);
}

TEST_F(CpuBufferTest, HostUnregisterPreservesRegistrationsAcrossIndexGrowth) {
  constexpr size_t kRegistrationCount = 257;
  std::array<std::array<uint8_t, 64>, kRegistrationCount> storage = {};
  std::array<hrx_buffer_t, kRegistrationCount> handles = {};
  std::array<bool, kRegistrationCount> registered = {};
  for (size_t ordinal = 0; ordinal < kRegistrationCount; ++ordinal) {
    // The prime-sized permutation separates registration order from address
    // order while growing past the table's initial allocation capacity.
    const size_t index = (ordinal * 37) % kRegistrationCount;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_host_memory_register(
        device_, storage[index].data(), storage[index].size(), 0)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_lookup(
        device_, storage[index].data(), &handles[index], nullptr)));
    registered[index] = true;
  }

  for (size_t ordinal = 0; ordinal < kRegistrationCount; ++ordinal) {
    const size_t index =
        ((kRegistrationCount - ordinal - 1) * 37) % kRegistrationCount;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_host_memory_unregister(device_, storage[index].data())));
    registered[index] = false;
    for (size_t probe = 0; probe < kRegistrationCount; ++probe) {
      hrx_buffer_t buffer = nullptr;
      size_t offset = SIZE_MAX;
      iree_status_t status = hrx_status_to_iree(hrx_buffer_lookup(
          device_, storage[probe].data() + 13, &buffer, &offset));
      if (registered[probe]) {
        IREE_ASSERT_OK(status);
        EXPECT_EQ(handles[probe], buffer);
        EXPECT_EQ(13u, offset);
      } else {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
        EXPECT_EQ(nullptr, buffer);
        EXPECT_EQ(0u, offset);
      }
    }
  }
}

TEST_F(CpuBufferTest, PreparedBackingAliasesRemainOrdinaryStreamBuffers) {
  for (hrx_memory_type_t memory_type :
       {HRX_MEMORY_TYPE_HOST_LOCAL, HRX_MEMORY_TYPE_DEVICE_LOCAL}) {
    for (bool stream_allocation : {false, true}) {
      SCOPED_TRACE(memory_type);
      SCOPED_TRACE(stream_allocation);
      const hrx_buffer_usage_t usage =
          HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SOURCE;
      iree_hal_pool_t* pool = nullptr;
      IREE_ASSERT_OK(hrx_mem_pool_backing_retain_allocation_source(
          device_, memory_type, usage, &pool));
      const auto requirements = iree_hal_pool_query_mapping_requirements(pool);
      ASSERT_GT(requirements.minimum_granularity, 0u);
      EXPECT_GE(requirements.recommended_granularity,
                requirements.minimum_granularity);
      const size_t size = requirements.recommended_granularity;
      hrx_buffer_t backing = nullptr;
      if (stream_allocation) {
        IREE_ASSERT_OK(hrx_status_to_iree(
            hrx_buffer_allocate(stream_, size, memory_type, usage, &backing)));
      } else {
        const hrx_buffer_params_t params = {
            .type = memory_type,
            .access = HRX_MEMORY_ACCESS_ALL,
            .usage = usage,
        };
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_), params, size, &backing)));
      }
      const uint8_t pattern = 0x5A;
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
          stream_, backing, 0, size, &pattern, sizeof(pattern))));
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

      const iree_hal_pool_family_access_t family = {
          .family = iree_hal_queue_family(stream_->hal_queue),
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      };
      const iree_hal_pool_scope_t scope = {
          .family_count = 1,
          .families = &family,
          .host =
              {
                  .access = IREE_HAL_MEMORY_ACCESS_ALL,
                  .modes = IREE_HAL_MAPPING_MODE_PERSISTENT,
              },
      };
      const iree_hal_virtual_buffer_options_t options = {
          .byte_length = size,
          .mapping_granularity = size,
      };
      iree_hal_buffer_t* reservation = nullptr;
      IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
          device_->hal_device_group, scope, {1, &pool}, &options,
          iree_allocator_system(), &reservation));
      iree_hal_memory_mapping_t mapping = {
          .target = reservation,
          .scope = {iree_hal_device_group_memory_domain(
                        device_->hal_device_group),
                    IREE_HAL_MEMORY_SCOPE_ANY},
          .target_range = {0, size},
          .source = backing->hal_buffer,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      };
      iree_hal_semaphore_t* semaphore = nullptr;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device_->hal_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
      uint64_t value = 1;
      const iree_hal_semaphore_list_t signals = {1, &semaphore, &value};
      IREE_ASSERT_OK(iree_hal_queue_map(
          stream_->hal_queue, iree_hal_semaphore_list_empty(), signals, 1,
          &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          semaphore, value, iree_infinite_timeout(), /*flags=*/0));

      hrx_buffer_t alias = nullptr;
      IREE_ASSERT_OK(hrx_buffer_create_from_hal((reservation), device_->group,
                                                HRX_MEMORY_TYPE_HOST_LOCAL,
                                                size, nullptr, &alias));
      const uint8_t update = 0xA7;
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
          stream_, alias, 16, 64, &update, sizeof(update))));
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
      std::vector<uint8_t> output(size);
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), backing, 0,
                              output.data(), output.size())));
      for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], i >= 16 && i < 80 ? update : pattern);
      }

      mapping.previous_source = mapping.source;
      mapping.previous_source_offset = mapping.source_offset;
      mapping.source = nullptr;
      mapping.source_offset = 0;
      mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
      mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
      ++value;
      IREE_ASSERT_OK(iree_hal_queue_map(
          stream_->hal_queue, iree_hal_semaphore_list_empty(), signals, 1,
          &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          semaphore, value, iree_infinite_timeout(), /*flags=*/0));
      iree_hal_buffer_release(reservation);
      hrx_buffer_release(alias);
      hrx_buffer_release(backing);
      iree_hal_semaphore_release(semaphore);
      iree_hal_pool_release(pool);
    }
  }
}

TEST_F(CpuBufferTest, HostRegistrationCannotProvideRemappableBacking) {
  uint8_t storage[64] = {};
  const hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SOURCE,
  };
  hrx_buffer_t buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        hrx_status_to_iree(hrx_allocator_import_buffer(
                            hrx_device_allocator(device_), params, storage,
                            sizeof(storage), &buffer)));
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(CpuBufferTest, MappingPermissionsAndDiscardProduceExpectedStreamOutput) {
  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
      stream_, 64, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buffer)));
  const uint8_t pattern = 0x5A;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
      stream_, buffer, 0, 64, &pattern, sizeof(pattern))));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  void* pointer = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_buffer_map(buffer, HRX_MAP_READ | HRX_MAP_WRITE | HRX_MAP_MAY_ALIAS,
                     0, 64, &pointer)));
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(pattern, static_cast<uint8_t*>(pointer)[i]);
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));

  // DISCARD remains an operation request and implies WRITE in the HRX API.
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_buffer_map(buffer, HRX_MAP_DISCARD, 13, 23, &pointer)));
  memset(pointer, 0xA7, 23);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));
  uint8_t result[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h(
      hrx_device_get_queue(device_, 0, 0), buffer, 0, result, sizeof(result))));
  for (size_t i = 0; i < sizeof(result); ++i) {
    EXPECT_EQ(i >= 13 && i < 36 ? 0xA7 : pattern, result[i]);
  }
  hrx_buffer_release(buffer);
}

}  // namespace
