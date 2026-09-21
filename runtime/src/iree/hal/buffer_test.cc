// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/buffer.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::string FormatMemoryType(iree_hal_memory_type_t memory_type) {
  iree_bitfield_string_temp_t temporary;
  const iree_string_view_t value =
      iree_hal_memory_type_format(memory_type, &temporary);
  return std::string(value.data, value.size);
}

TEST(MemoryTypeTest, EncodesLocalityIndependentlyFromCoherence) {
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_HOST_LOCAL, 0x42u);
  EXPECT_EQ(
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      0x46u);
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            0x72u);
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            0x76u);
}

TEST(MemoryTypeTest, RoundTripsOrthogonalLocalityAndCoherence) {
  static const struct {
    iree_hal_memory_type_t memory_type;
    const char* value;
  } cases[] = {
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL, "HOST_LOCAL"},
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
       "HOST_LOCAL|HOST_COHERENT"},
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
       "HOST_LOCAL|DEVICE_LOCAL"},
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
       "HOST_LOCAL|DEVICE_LOCAL|HOST_COHERENT"},
  };
  for (const auto& test_case : cases) {
    EXPECT_EQ(FormatMemoryType(test_case.memory_type), test_case.value);
    iree_hal_memory_type_t parsed_memory_type = 0;
    IREE_EXPECT_OK(iree_hal_memory_type_parse(
        iree_make_cstring_view(test_case.value), &parsed_memory_type));
    EXPECT_EQ(parsed_memory_type, test_case.memory_type);
  }
}

TEST(BufferRangeTest, AcceptsContainedRanges) {
  iree_hal_buffer_t buffer = {};
  buffer.byte_length = 16;

  IREE_EXPECT_OK(iree_hal_buffer_validate_range(&buffer, 0, 16));
  IREE_EXPECT_OK(iree_hal_buffer_validate_range(&buffer, 7, 9));
  IREE_EXPECT_OK(iree_hal_buffer_validate_range(&buffer, 16, 0));
}

TEST(BufferRangeTest, RejectsOutOfRangeAndOverflowingRanges) {
  iree_hal_buffer_t buffer = {};
  buffer.byte_length = 16;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_buffer_validate_range(&buffer, 17, 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_buffer_validate_range(&buffer, 15, 2));

  buffer.byte_length = IREE_DEVICE_SIZE_MAX;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_validate_range(&buffer, IREE_DEVICE_SIZE_MAX - 3, 8));
}

TEST(BufferPermissionTest, ValidatesAccessAndUsage) {
  IREE_EXPECT_OK(iree_hal_buffer_validate_access(IREE_HAL_MEMORY_ACCESS_READ,
                                                 IREE_HAL_MEMORY_ACCESS_READ));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_access(IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_MEMORY_ACCESS_WRITE));

  IREE_EXPECT_OK(
      iree_hal_buffer_validate_usage(IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
                                     IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_usage(IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
                                     IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
}

static void CountBufferRelease(void* user_data, iree_hal_buffer_t* buffer) {
  ++*static_cast<int*>(user_data);
}

struct SubspanAllocatorState {
  // Number of view wrappers allocated but not yet freed.
  iree_host_size_t live_allocation_count = 0;
  // Whether new allocations fail while existing wrappers remain releasable.
  bool fail_allocations = false;
};

static iree_status_t SubspanAllocatorCtl(void* self,
                                         iree_allocator_command_t command,
                                         const void* params, void** inout_ptr) {
  auto* state = static_cast<SubspanAllocatorState*>(self);
  const bool is_allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                             command == IREE_ALLOCATOR_COMMAND_CALLOC;
  if (is_allocation && state->fail_allocations) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  iree_allocator_t system_allocator = iree_allocator_system();
  iree_status_t status =
      system_allocator.ctl(system_allocator.self, command, params, inout_ptr);
  if (iree_status_is_ok(status)) {
    if (is_allocation) {
      ++state->live_allocation_count;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      --state->live_allocation_count;
    }
  }
  return status;
}

class BufferSubspanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING,
        sizeof(storage_), iree_make_byte_span(storage_, sizeof(storage_)),
        ReleaseCallback(), iree_allocator_system(), &root_));
  }

  void TearDown() override {
    iree_hal_buffer_release(root_);
    EXPECT_EQ(0u, allocator_state_.live_allocation_count);
  }

  iree_allocator_t ViewAllocator() {
    return iree_allocator_t{&allocator_state_, SubspanAllocatorCtl};
  }

  iree_hal_buffer_release_callback_t ReleaseCallback() {
    return iree_hal_buffer_release_callback_t{
        +[](void* user_data, iree_hal_buffer_t* buffer) {
          auto* offsets =
              static_cast<std::vector<iree_device_size_t>*>(user_data);
          offsets->push_back(iree_hal_buffer_byte_offset(buffer));
        },
        &released_offsets_};
  }

  // Real backing for all views; callback observations never access retired
  // data.
  alignas(64) uint8_t storage_[1024] = {};
  // Native allocation retained by the fixture unless explicitly released.
  iree_hal_buffer_t* root_ = nullptr;
  // Host allocation accounting for view wrappers only.
  SubspanAllocatorState allocator_state_;
  // Original allocation-relative offsets in callback execution order.
  std::vector<iree_device_size_t> released_offsets_;
};

TEST_F(BufferSubspanTest, NestedViewsAndSiblingsRetainTheReleaseOwner) {
  iree_hal_buffer_t* owner = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      root_, 32, 768, ReleaseCallback(), ViewAllocator(), &owner));
  iree_hal_buffer_t* sibling = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(owner, 0, 16, ViewAllocator(), &sibling));
  iree_hal_buffer_t* intermediate = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(owner, 16, 512, ViewAllocator(), &intermediate));
  iree_hal_buffer_t* child = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(intermediate, 8, 16, ViewAllocator(), &child));
  EXPECT_EQ(root_, iree_hal_buffer_allocated_buffer(child));
  EXPECT_EQ(56u, iree_hal_buffer_byte_offset(child));
  EXPECT_EQ(16u, iree_hal_buffer_byte_length(child));

  iree_hal_buffer_release(owner);
  iree_hal_buffer_release(intermediate);
  EXPECT_EQ(3u, allocator_state_.live_allocation_count);
  EXPECT_TRUE(released_offsets_.empty());
  iree_hal_buffer_release(child);
  EXPECT_EQ(2u, allocator_state_.live_allocation_count);
  EXPECT_TRUE(released_offsets_.empty());
  iree_hal_buffer_release(sibling);
  EXPECT_EQ(0u, allocator_state_.live_allocation_count);
  EXPECT_EQ((std::vector<iree_device_size_t>{32}), released_offsets_);
}

TEST_F(BufferSubspanTest, RepeatedSlicingDoesNotRetainIntermediateViews) {
  for (bool has_owner : {false, true}) {
    SCOPED_TRACE(has_owner);
    iree_hal_buffer_t* view = root_;
    if (has_owner) {
      IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
          root_, 32, 768, ReleaseCallback(), ViewAllocator(), &view));
    } else {
      iree_hal_buffer_retain(view);
    }
    for (int i = 0; i < 256; ++i) {
      iree_hal_buffer_t* child = nullptr;
      IREE_ASSERT_OK(iree_hal_buffer_subspan(view, 1, IREE_HAL_WHOLE_BUFFER,
                                             ViewAllocator(), &child));
      iree_hal_buffer_release(view);
      view = child;
      EXPECT_EQ(root_, iree_hal_buffer_allocated_buffer(view));
      EXPECT_EQ(has_owner ? 2u : 1u, allocator_state_.live_allocation_count);
      EXPECT_TRUE(released_offsets_.empty());
    }
    EXPECT_EQ(has_owner ? 288u : 256u, iree_hal_buffer_byte_offset(view));
    iree_hal_buffer_release(view);
    EXPECT_EQ(0u, allocator_state_.live_allocation_count);
  }
  EXPECT_EQ((std::vector<iree_device_size_t>{32}), released_offsets_);
}

TEST_F(BufferSubspanTest, WholeViewRetainsTheSameOwnerWithoutAllocating) {
  iree_hal_buffer_t* owner = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      root_, 32, 64, ReleaseCallback(), ViewAllocator(), &owner));
  allocator_state_.fail_allocations = true;
  iree_hal_buffer_t* alias = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(owner, 0, IREE_HAL_WHOLE_BUFFER,
                                         ViewAllocator(), &alias));
  EXPECT_EQ(owner, alias);
  iree_hal_buffer_release(owner);
  EXPECT_TRUE(released_offsets_.empty());
  iree_hal_buffer_release(alias);
  EXPECT_EQ((std::vector<iree_device_size_t>{32}), released_offsets_);
}

TEST_F(BufferSubspanTest, IndependentCallbacksComposeInLifetimeOrder) {
  iree_hal_buffer_t* outer = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      root_, 32, 768, ReleaseCallback(), ViewAllocator(), &outer));
  iree_hal_buffer_t* inner = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      outer, 48, 512, ReleaseCallback(), ViewAllocator(), &inner));
  EXPECT_EQ(root_, iree_hal_buffer_allocated_buffer(inner));
  iree_hal_buffer_t* child = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(inner, 8, 16, ViewAllocator(), &child));
  EXPECT_EQ(root_, iree_hal_buffer_allocated_buffer(child));
  EXPECT_EQ(56u, iree_hal_buffer_byte_offset(child));

  iree_hal_buffer_release(root_);
  root_ = nullptr;
  iree_hal_buffer_release(outer);
  iree_hal_buffer_release(inner);
  EXPECT_TRUE(released_offsets_.empty());
  iree_hal_buffer_release(child);
  // The inner callback runs with its containing owner still retained. The
  // final owner releases the native allocation before its own callback.
  EXPECT_EQ((std::vector<iree_device_size_t>{48, 0, 32}), released_offsets_);
}

TEST_F(BufferSubspanTest, AllocationFailureLeavesOwnershipWithTheCaller) {
  iree_hal_buffer_t* owner = nullptr;
  IREE_ASSERT_OK(iree_hal_subspan_buffer_create_with_callback(
      root_, 32, 768, ReleaseCallback(), ViewAllocator(), &owner));
  allocator_state_.fail_allocations = true;
  iree_hal_buffer_t* child = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_subspan_buffer_create_with_callback(
          owner, 48, 64, ReleaseCallback(), ViewAllocator(), &child));
  EXPECT_EQ(nullptr, child);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_buffer_subspan(owner, 16, 64, ViewAllocator(), &child));
  EXPECT_EQ(nullptr, child);
  EXPECT_EQ(1u, allocator_state_.live_allocation_count);
  EXPECT_TRUE(released_offsets_.empty());
  iree_hal_buffer_release(owner);
  EXPECT_EQ(0u, allocator_state_.live_allocation_count);
  EXPECT_EQ((std::vector<iree_device_size_t>{32}), released_offsets_);
}

TEST(BufferExportTest, NestedSubspansBorrowTheExactView) {
  alignas(64) uint8_t storage[128] = {};
  int release_count = 0;
  iree_hal_buffer_release_callback_t release_callback = {
      /*.fn=*/CountBufferRelease,
      /*.user_data=*/&release_count,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
          IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      sizeof(storage), iree_make_byte_span(storage, 96), release_callback,
      iree_allocator_system(), &buffer));

  iree_hal_external_buffer_t external_buffer = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(storage, external_buffer.handle.host_allocation.ptr);
  EXPECT_EQ(96u, external_buffer.size);

  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 16, 64,
                                         iree_allocator_system(), &subspan));
  iree_hal_buffer_t* nested_subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      subspan, 8, 16, iree_allocator_system(), &nested_subspan));
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested_subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(storage + 24, external_buffer.handle.host_allocation.ptr);
  EXPECT_EQ(16u, external_buffer.size);
  const uint8_t pattern = 0xA7;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(
      nested_subspan, 0, IREE_HAL_WHOLE_BUFFER, &pattern, sizeof(pattern)));
  auto* exported_bytes =
      static_cast<uint8_t*>(external_buffer.handle.host_allocation.ptr);
  for (size_t i = 0; i < external_buffer.size; ++i) {
    EXPECT_EQ(pattern, exported_bytes[i]);
  }
  EXPECT_EQ(0, storage[23]);
  EXPECT_EQ(0, storage[40]);

  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested_subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(storage + 24),
            external_buffer.handle.device_allocation.ptr);
  EXPECT_EQ(16u, external_buffer.size);
  EXPECT_EQ(0, release_count);
  iree_hal_buffer_release(nested_subspan);
  EXPECT_EQ(1, release_count);
}

TEST(BufferExportTest, FailureClearsOutputAndPreservesMappingRequirements) {
  alignas(64) uint8_t storage[64] = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      sizeof(storage), iree_make_byte_span(storage, sizeof(storage)),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));

  iree_hal_external_buffer_t external_buffer = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_buffer_export(buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD,
                             IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
                             &external_buffer));
  EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE, external_buffer.type);
  EXPECT_EQ(0u, external_buffer.size);
  EXPECT_EQ(nullptr, external_buffer.handle.host_allocation.ptr);

  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
      sizeof(storage), iree_make_byte_span(storage, sizeof(storage)),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_export(
          buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
          IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE, external_buffer.type);
  EXPECT_EQ(0u, external_buffer.size);
  EXPECT_EQ(nullptr, external_buffer.handle.host_allocation.ptr);
  iree_hal_buffer_release(buffer);
}

}  // namespace
