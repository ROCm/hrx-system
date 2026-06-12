// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/global_table.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct FakeResolver {
  iree_host_size_t verify_call_count = 0;
  iree_host_size_t create_buffer_call_count = 0;
  iree_host_size_t last_verify_physical_device_ordinal = IREE_HOST_SIZE_MAX;
  iree_host_size_t last_create_physical_device_ordinal = IREE_HOST_SIZE_MAX;
  iree_hal_queue_affinity_t last_create_queue_affinity = 0;
};

static bool TryLookupFakeGlobal(iree_string_view_t name,
                                iree_device_size_t* out_byte_length) {
  if (iree_string_view_equal(name, IREE_SV("global_a"))) {
    *out_byte_length = 64;
    return true;
  }
  if (iree_string_view_equal(name, IREE_SV("global_b"))) {
    *out_byte_length = 128;
    return true;
  }
  *out_byte_length = 0;
  return false;
}

static iree_status_t FakeTryVerify(
    void* user_data, iree_string_view_t name,
    iree_host_size_t verification_physical_device_ordinal, bool* out_found,
    iree_device_size_t* out_byte_length) {
  auto* resolver = static_cast<FakeResolver*>(user_data);
  ++resolver->verify_call_count;
  resolver->last_verify_physical_device_ordinal =
      verification_physical_device_ordinal;
  *out_found = TryLookupFakeGlobal(name, out_byte_length);
  return iree_ok_status();
}

static void FakeBufferRelease(void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_allocator_free_aligned(iree_allocator_system(), user_data);
}

static iree_status_t FakeCreateBuffer(
    void* user_data, iree_string_view_t name,
    iree_device_size_t expected_byte_length,
    iree_hal_queue_affinity_t selected_queue_affinity,
    iree_host_size_t physical_device_ordinal, iree_hal_buffer_t** out_buffer) {
  auto* resolver = static_cast<FakeResolver*>(user_data);
  ++resolver->create_buffer_call_count;
  resolver->last_create_physical_device_ordinal = physical_device_ordinal;
  resolver->last_create_queue_affinity = selected_queue_affinity;
  *out_buffer = nullptr;

  iree_device_size_t byte_length = 0;
  if (!TryLookupFakeGlobal(name, &byte_length)) {
    return iree_make_status(IREE_STATUS_INTERNAL, "fake global not verified");
  }
  if (byte_length != expected_byte_length) {
    return iree_make_status(IREE_STATUS_INTERNAL, "fake global size mismatch");
  }

  void* storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      iree_allocator_system(), expected_byte_length,
      IREE_HAL_HEAP_BUFFER_ALIGNMENT, /*offset=*/0, &storage));
  memset(storage, 0, expected_byte_length);

  iree_hal_buffer_placement_t placement = {
      /*.device=*/nullptr,
      /*.queue_affinity=*/selected_queue_affinity,
      /*.flags=*/IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_release_callback_t release_callback = {
      /*.fn=*/FakeBufferRelease,
      /*.user_data=*/storage,
  };
  iree_status_t status = iree_hal_heap_buffer_wrap(
      placement, IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_DEFAULT, expected_byte_length,
      iree_make_byte_span(storage, expected_byte_length), release_callback,
      iree_allocator_system(), out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free_aligned(iree_allocator_system(), storage);
  }
  return status;
}

static iree_hal_amdgpu_queue_affinity_domain_t TwoDeviceDomain() {
  return (iree_hal_amdgpu_queue_affinity_domain_t){
      /*.supported_affinity=*/0xFull,
      /*.physical_device_count=*/2,
      /*.queue_count_per_physical_device=*/2,
  };
}

class GlobalTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_hal_amdgpu_global_table_params_t params = {
        /*.host_allocator=*/iree_allocator_system(),
        /*.queue_affinity_domain=*/TwoDeviceDomain(),
        /*.loaded_physical_device_mask=*/0x3ull,
        /*.physical_device_count=*/2,
        /*.resolver=*/
        {
            /*.user_data=*/&resolver_,
            /*.try_verify=*/FakeTryVerify,
            /*.create_buffer=*/FakeCreateBuffer,
        },
    };
    IREE_ASSERT_OK(iree_hal_amdgpu_global_table_initialize(&params, &table_));
  }

  void TearDown() override {
    iree_hal_amdgpu_global_table_deinitialize(&table_);
  }

  FakeResolver resolver_;
  iree_hal_amdgpu_global_table_t table_;
};

TEST_F(GlobalTableTest, LookupInternsAndCachesByName) {
  iree_hal_executable_global_t first_global =
      iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_lookup(
      &table_, IREE_SV("global_a"), &first_global));
  EXPECT_TRUE(iree_hal_executable_global_is_valid(first_global));
  EXPECT_EQ(resolver_.verify_call_count, 1u);
  EXPECT_EQ(resolver_.last_verify_physical_device_ordinal, 0u);

  iree_hal_executable_global_t second_global =
      iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_lookup(
      &table_, IREE_SV("global_a"), &second_global));
  EXPECT_EQ(second_global.value, first_global.value);
  EXPECT_EQ(resolver_.verify_call_count, 1u);
}

TEST_F(GlobalTableTest, LookupMissDoesNotIntern) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_amdgpu_global_table_lookup(
                            &table_, IREE_SV("missing"), &global));
  EXPECT_FALSE(iree_hal_executable_global_is_valid(global));

  iree_hal_executable_global_info_t info;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_global_table_info(&table_, global, &info));
}

TEST_F(GlobalTableTest, InfoReturnsBorrowedMetadata) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_lookup(
      &table_, IREE_SV("global_b"), &global));

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_info(&table_, global, &info));
  EXPECT_TRUE(iree_string_view_equal(info.name, IREE_SV("global_b")));
  EXPECT_EQ(info.byte_length, 128u);
}

TEST_F(GlobalTableTest, BufferCachesPerPhysicalDevice) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_lookup(
      &table_, IREE_SV("global_a"), &global));

  iree_hal_buffer_t* first_device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_buffer(
      &table_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &first_device_buffer));
  ASSERT_NE(first_device_buffer, nullptr);
  EXPECT_EQ(resolver_.create_buffer_call_count, 1u);
  EXPECT_EQ(resolver_.last_create_physical_device_ordinal, 0u);
  EXPECT_EQ(resolver_.last_create_queue_affinity, 0x3ull);

  iree_hal_buffer_t* cached_first_device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_buffer(
      &table_, global, IREE_HAL_QUEUE_AFFINITY_ANY,
      &cached_first_device_buffer));
  EXPECT_EQ(cached_first_device_buffer, first_device_buffer);
  EXPECT_EQ(resolver_.create_buffer_call_count, 1u);

  iree_hal_buffer_t* second_device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_buffer(&table_, global, 0xCull,
                                                     &second_device_buffer));
  ASSERT_NE(second_device_buffer, nullptr);
  EXPECT_NE(second_device_buffer, first_device_buffer);
  EXPECT_EQ(resolver_.create_buffer_call_count, 2u);
  EXPECT_EQ(resolver_.last_create_physical_device_ordinal, 1u);
  EXPECT_EQ(resolver_.last_create_queue_affinity, 0xCull);
}

TEST_F(GlobalTableTest, InvalidGlobalAccessRejected) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();

  iree_hal_executable_global_info_t info;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_global_table_info(&table_, global, &info));

  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_global_table_buffer(
          &table_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &buffer));
  EXPECT_EQ(buffer, nullptr);
}

TEST(GlobalTableStandaloneTest, BufferRejectsUnloadedPhysicalDevice) {
  FakeResolver resolver;
  iree_hal_amdgpu_global_table_t table;
  const iree_hal_amdgpu_global_table_params_t params = {
      /*.host_allocator=*/iree_allocator_system(),
      /*.queue_affinity_domain=*/TwoDeviceDomain(),
      /*.loaded_physical_device_mask=*/0x1ull,
      /*.physical_device_count=*/2,
      /*.resolver=*/
      {
          /*.user_data=*/&resolver,
          /*.try_verify=*/FakeTryVerify,
          /*.create_buffer=*/FakeCreateBuffer,
      },
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_initialize(&params, &table));

  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_amdgpu_global_table_lookup(
      &table, IREE_SV("global_a"), &global));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_global_table_buffer(&table, global, 0xCull, &buffer));
  EXPECT_EQ(buffer, nullptr);

  iree_hal_amdgpu_global_table_deinitialize(&table);
}

}  // namespace
}  // namespace iree::hal::amdgpu
