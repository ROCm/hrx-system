// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

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
