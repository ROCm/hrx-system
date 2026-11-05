// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CpuBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_device_get(0, &device_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_create(device_, 0, &stream_)));
  }

  void TearDown() override {
    hrx_stream_release(stream_);
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }

  // CPU device borrowed from the initialized runtime.
  hrx_device_t device_ = nullptr;
  // Stream providing real allocation and transfer ordering.
  hrx_stream_t stream_ = nullptr;
};

TEST_F(CpuBufferTest,
       NativeAddressesPreserveScopedMappingsAndImportedContents) {
  const iree_hal_buffer_usage_t usages[] = {
      IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
  };
  for (iree_hal_buffer_usage_t usage : usages) {
    alignas(64) uint8_t storage[64];
    for (size_t i = 0; i < sizeof(storage); ++i) {
      storage[i] = i;
    }
    iree_hal_buffer_t* root = nullptr;
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, usage, sizeof(storage),
        iree_make_byte_span(storage, sizeof(storage)),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &root));
    iree_hal_buffer_t* view = nullptr;
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(root, 13, 23, iree_allocator_system(), &view));
    hrx_buffer_t buffer = nullptr;
    IREE_ASSERT_OK(hrx_buffer_create_from_hal(
        view, device_, HRX_MEMORY_TYPE_HOST_LOCAL, 23, nullptr, &buffer));
    iree_hal_buffer_release(view);
    iree_hal_buffer_release(root);

    // Cover no mapping, an interior read-only mapping, and a whole-buffer map.
    // None of these states can change native export permissions or its base.
    for (int mapping_case = 0; mapping_case < 3; ++mapping_case) {
      const size_t map_offset = mapping_case == 1 ? 5 : 0;
      const size_t map_length = mapping_case == 1 ? 7 : 23;
      if (mapping_case != 0) {
        void* mapped_pointer = nullptr;
        IREE_ASSERT_OK(hrx_status_to_iree(
            hrx_buffer_map(buffer, HRX_MAP_READ | HRX_MAP_MAY_ALIAS, map_offset,
                           map_length, &mapped_pointer)));
        EXPECT_EQ(storage + 13 + map_offset, mapped_pointer);
      }
      const iree_hal_buffer_mapping_t mapping = buffer->mapping;
      for (int query = 0; query < 2; ++query) {
        void* pointer = storage;
        iree_status_t status =
            hrx_status_to_iree(hrx_buffer_get_device_ptr(buffer, &pointer));
        if (usage & IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT) {
          IREE_ASSERT_OK(status);
          EXPECT_EQ(storage + 13, pointer);
        } else {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, status);
          EXPECT_EQ(nullptr, pointer);
        }
        EXPECT_EQ(mapping_case != 0, buffer->is_mapped);
        EXPECT_EQ(0, memcmp(&mapping, &buffer->mapping, sizeof(mapping)));
      }
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));
    }
    hrx_buffer_release(buffer);
    for (size_t i = 0; i < sizeof(storage); ++i) {
      EXPECT_EQ(i, storage[i]);
    }
  }
}

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

  void* device_pointer = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_buffer_get_device_ptr(buffer, &device_pointer)));
  void* mapped_pointer = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_map(
      buffer, HRX_MAP_WRITE | HRX_MAP_MAY_ALIAS, 13, 23, &mapped_pointer)));
  EXPECT_EQ(static_cast<uint8_t*>(device_pointer) + 13, mapped_pointer);
  memset(mapped_pointer, 0xA7, 23);
  void* repeated_pointer = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_buffer_get_device_ptr(buffer, &repeated_pointer)));
  EXPECT_EQ(device_pointer, repeated_pointer);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));

  uint8_t result[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(device_, buffer, 0, result, sizeof(result))));
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
            hrx_synchronous_h2d(device_, storage, buffer, 0, sizeof(storage))));
      }
      void* pointer = nullptr;
      IREE_ASSERT_OK(
          hrx_status_to_iree(hrx_buffer_get_device_ptr(buffer, &pointer)));
      ASSERT_NE(pointer, nullptr);
      EXPECT_FALSE(buffer->is_mapped);
      EXPECT_EQ(0, memcmp(pointer, storage, sizeof(storage)));
      if (construction == 2) {
        EXPECT_EQ(pointer, storage);
      }

      void* mapped_pointer = nullptr;
      if (usage & HRX_BUFFER_USAGE_MAPPING_SCOPED) {
        IREE_ASSERT_OK(hrx_status_to_iree(
            hrx_buffer_map(buffer, HRX_MAP_READ | HRX_MAP_MAY_ALIAS, 13, 23,
                           &mapped_pointer)));
        EXPECT_EQ(static_cast<uint8_t*>(pointer) + 13, mapped_pointer);
      }
      const iree_hal_buffer_mapping_t mapping = buffer->mapping;
      void* repeated_pointer = nullptr;
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_buffer_get_device_ptr(buffer, &repeated_pointer)));
      EXPECT_EQ(pointer, repeated_pointer);
      EXPECT_EQ(0, memcmp(&mapping, &buffer->mapping, sizeof(mapping)));
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));

      uint8_t result[64] = {};
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_synchronous_d2h(device_, buffer, 0, result, sizeof(result))));
      EXPECT_EQ(0, memcmp(result, storage, sizeof(storage)));
      hrx_buffer_release(buffer);
      for (uint8_t value : storage) {
        EXPECT_EQ(0x5A, value);
      }
    }
  }
}

TEST_F(CpuBufferTest, HostRegistrationAddressOutlivesRegistryEntry) {
  uint8_t storage[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_host_memory_register(device_, storage, sizeof(storage), 0)));
  hrx_buffer_t buffer = nullptr;
  size_t offset = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_buffer_lookup(device_, storage + 13, &buffer, &offset)));
  EXPECT_EQ(13u, offset);
  hrx_buffer_retain(buffer);
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_host_memory_unregister(device_, storage)));
  void* device_pointer = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_buffer_get_device_ptr(buffer, &device_pointer)));
  EXPECT_EQ(storage, device_pointer);
  void* mapped_pointer = storage;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      hrx_status_to_iree(hrx_buffer_map(buffer, HRX_MAP_READ, 0,
                                        sizeof(storage), &mapped_pointer)));
  EXPECT_EQ(nullptr, mapped_pointer);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_buffer_get_device_ptr(buffer, &device_pointer)));
  EXPECT_EQ(storage, device_pointer);
  hrx_buffer_release(buffer);
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
          .family = iree_hal_queue_family(device_->transfer_queue),
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
          device_->transfer_queue, iree_hal_semaphore_list_empty(), signals, 1,
          &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          semaphore, value, iree_infinite_timeout(), /*flags=*/0));

      hrx_buffer_t alias = nullptr;
      IREE_ASSERT_OK(hrx_buffer_create_from_hal((reservation), device_,
                                                HRX_MEMORY_TYPE_HOST_LOCAL,
                                                size, nullptr, &alias));
      const uint8_t update = 0xA7;
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
          stream_, alias, 16, 64, &update, sizeof(update))));
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
      std::vector<uint8_t> output(size);
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h(
          device_, backing, 0, output.data(), output.size())));
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
          device_->transfer_queue, iree_hal_semaphore_list_empty(), signals, 1,
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
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(device_, buffer, 0, result, sizeof(result))));
  for (size_t i = 0; i < sizeof(result); ++i) {
    EXPECT_EQ(i >= 13 && i < 36 ? 0xA7 : pattern, result[i]);
  }
  hrx_buffer_release(buffer);
}

}  // namespace
