// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/buffer.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "iree/hal/memory_scope.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BufferBindingTest, PublishesEveryInterfaceAndPreservesNativeFacts) {
  uint8_t storage[256] = {};
  const uint16_t types[] = {
      IREE_HAL_BUFFER_INTERFACE_HOST,
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER,
      IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
      IREE_HAL_BUFFER_INTERFACE_XDNA_FIRMWARE,
      IREE_HAL_BUFFER_INTERFACE_RDMA,
      IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO,
      IREE_HAL_BUFFER_INTERFACE_REMOTE,
  };
  struct NativeTable {
    // Generic representations copied and translated by the shared publisher.
    iree_hal_buffer_native_binding_t bindings[IREE_ARRAYSIZE(types)];
    // Producer-specific value data borrowed unchanged by native consumers.
    uint64_t native_facts;
  } source = {}, target = {};
  source.bindings[0].host_pointer = storage;
  source.bindings[1].device_address = 0x1000;
  source.bindings[2].vulkan = {7, 64};
  source.bindings[3].device_address = 0x2000;
  source.bindings[4].device_address = 0x3000;
  source.bindings[5].rdma = {0x4000, 11, 22};
  source.bindings[6].registered_io = {storage, 33, 128};
  source.bindings[7].remote = {44, 256};
  source.native_facts = 0x123456789ABCDEF0;
  const iree_hal_buffer_binding_layout_t layout = {
      sizeof(source), IREE_ARRAYSIZE(types), 0, types};
  iree_hal_buffer_memory_view_t memory = {};
  memory.bindings = source.bindings;
  memory.binding_offset = 16;
  iree_hal_buffer_memory_copy_bindings(&memory, &layout, target.bindings);
  EXPECT_EQ(target.bindings[0].host_pointer, storage + 16);
  EXPECT_EQ(target.bindings[1].device_address, 0x1010u);
  EXPECT_EQ(target.bindings[2].vulkan.buffer, 7u);
  EXPECT_EQ(target.bindings[2].vulkan.offset, 80u);
  EXPECT_EQ(target.bindings[3].device_address, 0x2010u);
  EXPECT_EQ(target.bindings[4].device_address, 0x3010u);
  EXPECT_EQ(target.bindings[5].rdma.address, 0x4010u);
  EXPECT_EQ(target.bindings[5].rdma.local_key, 11u);
  EXPECT_EQ(target.bindings[5].rdma.remote_key, 22u);
  EXPECT_EQ(target.bindings[6].registered_io.table, storage);
  EXPECT_EQ(target.bindings[6].registered_io.index, 33u);
  EXPECT_EQ(target.bindings[6].registered_io.offset, 144u);
  EXPECT_EQ(target.bindings[7].remote.object_id, 44u);
  EXPECT_EQ(target.bindings[7].remote.offset, 272u);
  EXPECT_EQ(target.native_facts, source.native_facts);

  // A resource without an optional raw address does not acquire one merely
  // because its view begins at a nonzero offset.
  source.bindings[0].host_pointer = nullptr;
  for (uint16_t index : {1, 3, 4}) {
    source.bindings[index].device_address = 0;
  }
  iree_hal_buffer_memory_copy_bindings(&memory, &layout, target.bindings);
  EXPECT_EQ(target.bindings[0].host_pointer, nullptr);
  for (uint16_t index : {1, 3, 4}) {
    EXPECT_EQ(target.bindings[index].device_address, 0u);
  }
}

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

TEST(BufferPermissionTest, OpaqueStorageRequiresEitherDirection) {
  for (iree_hal_buffer_usage_t usage :
       {IREE_HAL_BUFFER_USAGE_STORAGE_READ, IREE_HAL_BUFFER_USAGE_STORAGE_WRITE,
        IREE_HAL_BUFFER_USAGE_STORAGE}) {
    IREE_EXPECT_OK(iree_hal_buffer_validate_usage_any(
        usage, IREE_HAL_BUFFER_USAGE_STORAGE));
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_usage_any(IREE_HAL_BUFFER_USAGE_TRANSFER,
                                         IREE_HAL_BUFFER_USAGE_STORAGE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_usage(IREE_HAL_BUFFER_USAGE_STORAGE_READ,
                                     IREE_HAL_BUFFER_USAGE_STORAGE_WRITE));
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

TEST(BufferPermissionTest, RequiresOnlyReadWritePermissions) {
  for (iree_hal_memory_access_t access :
       {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_MEMORY_ACCESS_ALL}) {
    IREE_EXPECT_OK(
        iree_hal_buffer_validate_access(IREE_HAL_MEMORY_ACCESS_ALL, access));
  }
  for (iree_hal_memory_access_t access :
       {0u, 1u << 4, (1u << 5) | IREE_HAL_MEMORY_ACCESS_WRITE}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_buffer_validate_access(IREE_HAL_MEMORY_ACCESS_ALL, access));
  }
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

class BufferMapTest : public ::testing::TestWithParam<iree_hal_mapping_mode_t> {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < sizeof(storage_); ++i) {
      storage_[i] = i;
    }
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_MAPPING |
            IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
        sizeof(storage_), iree_make_byte_span(storage_, sizeof(storage_)),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &buffer_));
    iree_hal_buffer_t* outer = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer_, 8, 48,
                                           iree_allocator_system(), &outer));
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(outer, 5, 24, iree_allocator_system(), &view_));
    iree_hal_buffer_release(outer);
  }

  void TearDown() override {
    iree_hal_buffer_release(view_);
    iree_hal_buffer_release(buffer_);
  }

  // Imported host storage, including bytes outside the nested logical view.
  alignas(64) uint8_t storage_[64] = {};
  // Root view of the imported allocation.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Nested view starting at byte 13 of storage_.
  iree_hal_buffer_t* view_ = nullptr;
};

TEST_P(BufferMapTest, ReadWritePreservesContentsWithEitherAliasPromise) {
  for (iree_hal_buffer_map_flags_t flags :
       {IREE_HAL_BUFFER_MAP_FLAG_NONE, IREE_HAL_BUFFER_MAP_FLAG_MAY_ALIAS}) {
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        view_, GetParam(), IREE_HAL_MEMORY_ACCESS_ALL, flags, 3, 16, &mapping));
    EXPECT_EQ(storage_ + 16, mapping.contents.data);
    EXPECT_EQ(16u, mapping.contents.data_length);
    for (size_t i = 0; i < sizeof(storage_); ++i) {
      EXPECT_EQ(i, storage_[i]);
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  }
}

TEST_P(BufferMapTest, PreparedDiscardAppliesOnlyWhenCommitted) {
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_prepare_map_range(
      view_, GetParam(), IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_DISCARD, 3, 16, &mapping));
  EXPECT_EQ(nullptr, mapping.contents.data);
  for (size_t i = 0; i < sizeof(storage_); ++i) {
    EXPECT_EQ(i, storage_[i]);
  }
  IREE_ASSERT_OK(iree_hal_buffer_commit_map_range(&mapping));
  EXPECT_EQ(storage_ + 16, mapping.contents.data);
  memset(mapping.contents.data, 0xA5, mapping.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  for (size_t i = 0; i < sizeof(storage_); ++i) {
    EXPECT_EQ(i >= 16 && i < 32 ? 0xA5u : i, storage_[i]);
  }
}

TEST_P(BufferMapTest, RejectsInvalidFlagsBeforeMapping) {
  iree_hal_buffer_mapping_t mapping = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_buffer_map_range(
                            view_, GetParam(), IREE_HAL_MEMORY_ACCESS_READ,
                            IREE_HAL_BUFFER_MAP_FLAG_DISCARD, 0, 8, &mapping));
  EXPECT_EQ(nullptr, mapping.buffer);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_buffer_map_range(view_, GetParam(), IREE_HAL_MEMORY_ACCESS_ALL,
                                static_cast<iree_hal_buffer_map_flags_t>(0x80),
                                0, 8, &mapping));
  EXPECT_EQ(nullptr, mapping.buffer);
  for (size_t i = 0; i < sizeof(storage_); ++i) {
    EXPECT_EQ(i, storage_[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(Modes, BufferMapTest,
                         ::testing::Values(IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MAPPING_MODE_PERSISTENT));

// Models a device allocation whose host cache changes only at the buffer's
// explicit invalidate/flush boundary. The convenience helpers remain the
// production implementation under test.
struct MappedTransferBuffer : iree_hal_buffer_t {
  // Bytes visible to completed device work.
  std::array<uint8_t, 64> device_contents = {};
  // Stale host cache for noncoherent mappings.
  std::array<uint8_t, 64> host_contents = {};
  // Outstanding mappings that must be returned even after a cache error.
  size_t mapping_count = 0;
  // Source transitions performed by the caller.
  size_t invalidate_count = 0;
  // Target transitions performed by the caller.
  size_t flush_count = 0;
  // Last exact native maintenance range, including the mapped view offset.
  struct {
    // Byte offset from the native allocation base.
    iree_device_size_t offset = 0;
    // Number of bytes passed to the native operation.
    iree_device_size_t length = 0;
  } maintenance_range;
  // Injected native invalidation result, without owning a status allocation.
  iree_status_code_t invalidate_code = IREE_STATUS_OK;
  // Injected native publication result, without owning a status allocation.
  iree_status_code_t flush_code = IREE_STATUS_OK;

  explicit MappedTransferBuffer(iree_hal_memory_type_t coherence) {
    static const iree_hal_buffer_vtable_t vtable = {
        /*.recycle=*/iree_hal_buffer_recycle,
        /*.destroy=*/
        [](iree_hal_buffer_t* base) {
          delete static_cast<MappedTransferBuffer*>(base);
        },
        /*.export_range=*/nullptr,
        /*.map_range=*/
        [](iree_hal_buffer_t* base, iree_hal_mapping_mode_t,
           iree_hal_memory_access_t, iree_hal_buffer_map_flags_t,
           iree_device_size_t offset, iree_device_size_t length,
           iree_hal_buffer_mapping_t* mapping) {
          auto* buffer = static_cast<MappedTransferBuffer*>(base);
          ++buffer->mapping_count;
          auto& contents = iree_all_bits_set(iree_hal_buffer_memory_type(base),
                                             IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
                               ? buffer->device_contents
                               : buffer->host_contents;
          mapping->contents =
              iree_make_byte_span(contents.data() + offset, length);
          return iree_ok_status();
        },
        /*.unmap_range=*/
        [](iree_hal_buffer_t* base, iree_device_size_t, iree_device_size_t,
           iree_hal_buffer_mapping_t*) {
          --static_cast<MappedTransferBuffer*>(base)->mapping_count;
          return iree_ok_status();
        },
        /*.invalidate_range=*/
        [](iree_hal_buffer_t* base, iree_device_size_t offset,
           iree_device_size_t length) {
          auto* buffer = static_cast<MappedTransferBuffer*>(base);
          ++buffer->invalidate_count;
          buffer->maintenance_range = {offset, length};
          if (buffer->invalidate_code != IREE_STATUS_OK) {
            return iree_status_from_code(buffer->invalidate_code);
          }
          memcpy(buffer->host_contents.data() + offset,
                 buffer->device_contents.data() + offset, length);
          return iree_ok_status();
        },
        /*.flush_range=*/
        [](iree_hal_buffer_t* base, iree_device_size_t offset,
           iree_device_size_t length) {
          auto* buffer = static_cast<MappedTransferBuffer*>(base);
          ++buffer->flush_count;
          buffer->maintenance_range = {offset, length};
          if (buffer->flush_code != IREE_STATUS_OK) {
            return iree_status_from_code(buffer->flush_code);
          }
          if (!iree_all_bits_set(iree_hal_buffer_memory_type(base),
                                 IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
            memcpy(buffer->device_contents.data() + offset,
                   buffer->host_contents.data() + offset, length);
          }
          return iree_ok_status();
        },
        /*.query_memory=*/nullptr,
        /*.allocation=*/nullptr,
    };
    iree_hal_buffer_initialize(
        iree_hal_buffer_placement_undefined(), this, device_contents.size(), 0,
        device_contents.size(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL | coherence,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING, &vtable,
        this);
  }
};

using MappedTransferBufferPtr =
    std::unique_ptr<MappedTransferBuffer, decltype(&iree_hal_buffer_release)>;

static MappedTransferBufferPtr MakeMappedTransferBuffer(
    iree_hal_memory_type_t coherence = 0) {
  return MappedTransferBufferPtr(new MappedTransferBuffer(coherence),
                                 iree_hal_buffer_release);
}

TEST(BufferMappedTransferTest, ReadObservesEveryCompletedWrite) {
  for (iree_hal_memory_type_t coherence :
       {0u, static_cast<uint32_t>(IREE_HAL_MEMORY_TYPE_HOST_COHERENT)}) {
    auto source = MakeMappedTransferBuffer(coherence);
    for (uint8_t batch = 1; batch <= 4; ++batch) {
      for (size_t i = 0; i < source->device_contents.size(); ++i) {
        source->device_contents[i] = batch * 17 + i;
      }
      std::array<uint8_t, 8> output = {};
      IREE_ASSERT_OK(iree_hal_buffer_map_read(source.get(), 9, output.data(),
                                              output.size()));
      for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], source->device_contents[9 + i]);
      }
      EXPECT_EQ(source->mapping_count, 0u);
    }
    EXPECT_EQ(source->invalidate_count, coherence ? 0u : 4u);
    EXPECT_EQ(source->flush_count, 0u);
  }
}

TEST(BufferMappedTransferTest, CopyTransitionsOnlyTheCopiedRanges) {
  for (iree_hal_memory_type_t source_coherence :
       {0u, static_cast<uint32_t>(IREE_HAL_MEMORY_TYPE_HOST_COHERENT)}) {
    for (iree_hal_memory_type_t target_coherence :
         {0u, static_cast<uint32_t>(IREE_HAL_MEMORY_TYPE_HOST_COHERENT)}) {
      for (iree_device_size_t length :
           {iree_device_size_t{8}, IREE_HAL_WHOLE_BUFFER}) {
        auto source = MakeMappedTransferBuffer(source_coherence);
        auto target = MakeMappedTransferBuffer(target_coherence);
        const size_t copied_length = length == IREE_HAL_WHOLE_BUFFER ? 50 : 8;
        for (uint8_t batch = 1; batch <= 4; ++batch) {
          for (size_t i = 0; i < source->device_contents.size(); ++i) {
            source->device_contents[i] = batch * 17 + i;
          }
          target->device_contents.fill(0xE7);
          target->host_contents.fill(0xA9);
          IREE_ASSERT_OK(iree_hal_buffer_map_copy(source.get(), 9, target.get(),
                                                  14, length));
          for (size_t i = 0; i < target->device_contents.size(); ++i) {
            EXPECT_EQ(target->device_contents[i],
                      i >= 14 && i < 14 + copied_length
                          ? source->device_contents[9 + i - 14]
                          : 0xE7);
            if (!source_coherence) {
              EXPECT_EQ(source->host_contents[i],
                        i >= 9 && i < 9 + copied_length
                            ? source->device_contents[i]
                            : 0);
            }
          }
          EXPECT_EQ(source->mapping_count, 0u);
          EXPECT_EQ(target->mapping_count, 0u);
        }
        EXPECT_EQ(source->invalidate_count, source_coherence ? 0u : 4u);
        EXPECT_EQ(target->flush_count, target_coherence ? 0u : 4u);
        EXPECT_EQ(source->flush_count, 0u);
        EXPECT_EQ(target->invalidate_count, 0u);
      }
    }
  }
}

TEST(BufferMappedTransferTest, FailedInvalidationReturnsEveryMapping) {
  auto source = MakeMappedTransferBuffer();
  auto target = MakeMappedTransferBuffer();
  source->invalidate_code = IREE_STATUS_UNAVAILABLE;
  source->device_contents.fill(0x37);
  target->device_contents.fill(0xE7);
  std::array<uint8_t, 8> output;
  output.fill(0xA9);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_buffer_map_read(source.get(), 9, output.data(), output.size()));
  for (uint8_t value : output) {
    EXPECT_EQ(value, 0xA9);
  }
  EXPECT_EQ(source->mapping_count, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_buffer_map_copy(source.get(), 9, target.get(),
                                                 14, output.size()));
  for (uint8_t value : target->device_contents) {
    EXPECT_EQ(value, 0xE7);
  }
  EXPECT_EQ(source->mapping_count, 0u);
  EXPECT_EQ(target->mapping_count, 0u);
  EXPECT_EQ(source->invalidate_count, 2u);
  EXPECT_EQ(target->flush_count, 0u);
}

// Native qualification is the dependency here. The mapping lifecycle, range
// resolution, recipe capture and host executor use their production APIs.
class BufferMappingTransitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_hal_buffer_binding_layout_t layout = {};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        this, 4, &layout, iree_allocator_system(), &contract_));
    contract_->host.access = IREE_HAL_MEMORY_ACCESS_ALL;
    contract_->scopes[2].interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_HOST;
    contract_->scopes[2].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
    IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
        contract_,
        [](void*, uint32_t producer, uint32_t consumer,
           iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
          out_info->flags = IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE;
          out_info->release.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
          out_info->acquire.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
          if (producer == 1) {
            out_info->release.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
            out_info->release.executor =
                IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
            out_info->release.operation =
                IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH;
            out_info->release.range_granularity = 1;
          }
          if (consumer == 1) {
            out_info->acquire.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
            out_info->acquire.executor =
                IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
            out_info->acquire.operation =
                IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE;
            out_info->acquire.range_granularity = 1;
          }
          return iree_ok_status();
        },
        nullptr));
    const iree_hal_memory_transition_table_t table = {contract_};
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, {this, 1}, {this, 2}, IREE_HAL_MEMORY_TRANSITION_RELEASE,
        &pair));
    release_ = iree_hal_memory_transition_query(table, pair).release;
    release_recipe_ = iree_hal_memory_transition_recipe(
        table, pair, IREE_HAL_MEMORY_TRANSITION_RELEASE);
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, {this, 2}, {this, 1}, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
        &pair));
    acquire_ = iree_hal_memory_transition_query(table, pair).acquire;
    acquire_recipe_ = iree_hal_memory_transition_recipe(
        table, pair, IREE_HAL_MEMORY_TRANSITION_ACQUIRE);
  }

  void TearDown() override { iree_hal_memory_contract_release(contract_); }

  // Captured qualification retained through each host call.
  iree_hal_memory_contract_t* contract_ = nullptr;
  // Required producer-local publication action.
  iree_hal_memory_effects_t release_ = {};
  // Required consumer-local observation action.
  iree_hal_memory_effects_t acquire_ = {};
  // Immutable publication recipe borrowed from the contract.
  const iree_hal_memory_transition_recipe_t* release_recipe_ = nullptr;
  // Immutable observation recipe borrowed from the contract.
  const iree_hal_memory_transition_recipe_t* acquire_recipe_ = nullptr;
};

TEST_F(BufferMappingTransitionTest,
       NativeApiSurvivesCoherenceAndSubviewOffsets) {
  for (iree_hal_memory_type_t coherence :
       {0u, static_cast<uint32_t>(IREE_HAL_MEMORY_TYPE_HOST_COHERENT)}) {
    auto buffer = MakeMappedTransferBuffer(coherence);
    iree_hal_buffer_t* view = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer.get(), 7, 40,
                                           iree_allocator_system(), &view));
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        view, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 3, 24, &mapping));
    memset(mapping.contents.data, 0x57, mapping.contents.data_length);
    iree_hal_buffer_mapping_transition_t transition = {&mapping, 5, 8,
                                                       release_recipe_};
    IREE_ASSERT_OK(
        iree_hal_buffer_mapping_memory_barrier(release_, 1, &transition));
    EXPECT_EQ(buffer->flush_count, 1u);
    EXPECT_EQ(buffer->maintenance_range.offset, 15u);
    EXPECT_EQ(buffer->maintenance_range.length, 8u);
    EXPECT_EQ(buffer->device_contents[15], 0x57);
    buffer->device_contents[16] = 0x92;
    transition.recipe = acquire_recipe_;
    IREE_ASSERT_OK(
        iree_hal_buffer_mapping_memory_barrier(acquire_, 1, &transition));
    EXPECT_EQ(buffer->invalidate_count, 1u);
    EXPECT_EQ(buffer->maintenance_range.offset, 15u);
    EXPECT_EQ(buffer->maintenance_range.length, 8u);
    EXPECT_EQ(mapping.contents.data[6], 0x92);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    iree_hal_buffer_release(view);
    EXPECT_EQ(buffer->mapping_count, 0u);
  }
}

TEST_F(BufferMappingTransitionTest, PreflightsTheEntireBatchBeforeMaintenance) {
  auto buffer = MakeMappedTransferBuffer();
  iree_hal_buffer_mapping_t writable = {}, readable = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer.get(), IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 32, &writable));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer.get(), IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 32, 32, &readable));
  iree_hal_buffer_mapping_transition_t transitions[] = {
      {&writable, 0, IREE_HAL_WHOLE_BUFFER, release_recipe_},
      {&readable, 0, IREE_HAL_WHOLE_BUFFER, release_recipe_},
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_mapping_memory_barrier(release_, 2, transitions));
  EXPECT_EQ(buffer->flush_count, 0u);
  transitions[1] = {&writable, 31, 2, release_recipe_};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_mapping_memory_barrier(release_, 2, transitions));
  EXPECT_EQ(buffer->flush_count, 0u);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&readable));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&writable));
}

TEST_F(BufferMappingTransitionTest,
       NativeFailureStopsBatchAndPreservesOwnership) {
  auto first = MakeMappedTransferBuffer();
  auto second = MakeMappedTransferBuffer();
  second->flush_code = IREE_STATUS_UNAVAILABLE;
  first->host_contents.fill(0x38);
  second->host_contents.fill(0x29);
  iree_hal_buffer_mapping_t mappings[2] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      first.get(), IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 16, &mappings[0]));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      second.get(), IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 16, &mappings[1]));
  iree_hal_buffer_mapping_transition_t transitions[] = {
      {&mappings[0], 0, IREE_HAL_WHOLE_BUFFER, release_recipe_},
      {&mappings[1], 0, IREE_HAL_WHOLE_BUFFER, release_recipe_},
      {&mappings[0], 16, 0, release_recipe_},
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_buffer_mapping_memory_barrier(release_, 3, transitions));
  EXPECT_EQ(first->flush_count, 1u);
  EXPECT_EQ(second->flush_count, 1u);
  EXPECT_EQ(first->device_contents[0], 0x38);
  EXPECT_EQ(second->device_contents[0], 0u);
  EXPECT_EQ(first->mapping_count, 1u);
  EXPECT_EQ(second->mapping_count, 1u);
  for (auto& mapping : mappings) {
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  }
}

}  // namespace
