// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <thread>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::Status;
using ::iree::StatusCode;
using ::iree::testing::status::IsOk;
using ::iree::testing::status::StatusIs;

TEST(AllocatorTrim, NullAllocator) {
  IREE_EXPECT_OK(iree_allocator_trim(iree_allocator_null()));
}

TEST(AllocatorTrim, PreservesInlineStorage) {
  IREE_ALLOCATOR_INLINE_STORAGE(storage, 128);
  iree_allocator_t allocator = iree_allocator_inline_arena(&storage.header);
  void* live = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(allocator, 64, &live));
  std::memset(live, 0x5A, 64);
  IREE_ASSERT_OK(iree_allocator_trim(allocator));
  void* next = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(allocator, 64, &next));
  EXPECT_NE(next, live);
  for (iree_host_size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(live)[i], 0x5A);
  }
}

TEST(AllocatorTrim, PreservesLiveCrossThreadAllocations) {
  constexpr iree_host_size_t kCount = 32;
  constexpr iree_host_size_t kLength = 32768;
  void* allocations[kCount] = {};
  iree_allocator_t allocator = iree_allocator_system();
  std::thread worker([&] {
    for (iree_host_size_t i = 0; i < kCount; ++i) {
      IREE_ASSERT_OK(iree_allocator_malloc_uninitialized(allocator, kLength,
                                                         &allocations[i]));
      std::memset(allocations[i], static_cast<int>(i), kLength);
    }
  });
  worker.join();
  for (iree_host_size_t i = 0; i < kCount; i += 2) {
    iree_allocator_free(allocator, allocations[i]);
  }
  IREE_ASSERT_OK(iree_allocator_trim(allocator));
  for (iree_host_size_t i = 1; i < kCount; i += 2) {
    const auto* bytes = static_cast<const uint8_t*>(allocations[i]);
    for (iree_host_size_t j = 0; j < kLength; ++j) {
      EXPECT_EQ(bytes[j], i);
    }
    iree_allocator_free(allocator, allocations[i]);
  }
  IREE_EXPECT_OK(iree_allocator_trim(allocator));
  void* subsequent = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(allocator, kLength, &subsequent));
  iree_allocator_free(allocator, subsequent);
}

TEST(AllocatorTrim, PropagatesCustomAllocatorFailure) {
  bool called = false;
  iree_allocator_t allocator = {
      &called,
      [](void* self, iree_allocator_command_t command, const void* params,
         void** pointer) -> iree_status_t {
        EXPECT_EQ(command, IREE_ALLOCATOR_COMMAND_TRIM);
        EXPECT_EQ(params, nullptr);
        EXPECT_EQ(pointer, nullptr);
        *static_cast<bool*>(self) = true;
        return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
      },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_allocator_trim(allocator));
  EXPECT_TRUE(called);
}

//===----------------------------------------------------------------------===//
// Byte span tests
//===----------------------------------------------------------------------===//

TEST(ByteSpanTest, Empty) {
  uint8_t storage[1] = {0};
  EXPECT_TRUE(iree_byte_span_is_empty(iree_byte_span_empty()));
  EXPECT_TRUE(iree_byte_span_is_empty(iree_make_byte_span(storage, 0)));
  EXPECT_FALSE(iree_byte_span_is_empty(iree_make_byte_span(NULL, 1)));
}

TEST(ConstByteSpanTest, Empty) {
  const uint8_t storage[1] = {0};
  EXPECT_TRUE(iree_const_byte_span_is_empty(iree_const_byte_span_empty()));
  EXPECT_TRUE(
      iree_const_byte_span_is_empty(iree_make_const_byte_span(storage, 0)));
  EXPECT_FALSE(
      iree_const_byte_span_is_empty(iree_make_const_byte_span(NULL, 1)));
}

//===----------------------------------------------------------------------===//
// Array allocation tests
//===----------------------------------------------------------------------===//

TEST(AllocatorArray, MallocArrayBasic) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(
      iree_allocator_malloc_array(iree_allocator_system(), 10, 8, &ptr));
  ASSERT_NE(ptr, nullptr);
  // Verify memory is zeroed.
  for (int i = 0; i < 80; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(ptr)[i], 0);
  }
  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(AllocatorArray, MallocArrayOverflow) {
  void* ptr = nullptr;
  EXPECT_THAT(Status(iree_allocator_malloc_array(iree_allocator_system(),
                                                 IREE_HOST_SIZE_MAX, 2, &ptr)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST(AllocatorArray, MallocArrayZeroCount) {
  void* ptr = nullptr;
  // Zero count results in zero bytes, which IREE's allocator rejects.
  EXPECT_THAT(
      Status(iree_allocator_malloc_array(iree_allocator_system(), 0, 8, &ptr)),
      StatusIs(StatusCode::kInvalidArgument));
}

TEST(AllocatorArray, MallocArrayUninitializedBasic) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(iree_allocator_malloc_array_uninitialized(
      iree_allocator_system(), 10, 8, &ptr));
  ASSERT_NE(ptr, nullptr);
  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(AllocatorArray, MallocArrayUninitializedOverflow) {
  void* ptr = nullptr;
  EXPECT_THAT(Status(iree_allocator_malloc_array_uninitialized(
                  iree_allocator_system(), IREE_HOST_SIZE_MAX, 2, &ptr)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST(AllocatorArray, ReallocArrayBasic) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(
      iree_allocator_malloc_array(iree_allocator_system(), 10, 8, &ptr));
  ASSERT_NE(ptr, nullptr);
  IREE_EXPECT_OK(
      iree_allocator_realloc_array(iree_allocator_system(), 20, 8, &ptr));
  ASSERT_NE(ptr, nullptr);
  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(AllocatorArray, ReallocArrayOverflow) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(
      iree_allocator_malloc_array(iree_allocator_system(), 10, 8, &ptr));
  ASSERT_NE(ptr, nullptr);
  void* original_ptr = ptr;
  EXPECT_THAT(Status(iree_allocator_realloc_array(iree_allocator_system(),
                                                  IREE_HOST_SIZE_MAX, 2, &ptr)),
              StatusIs(StatusCode::kOutOfRange));
  // Original pointer should be unchanged on failure.
  EXPECT_EQ(ptr, original_ptr);
  iree_allocator_free(iree_allocator_system(), ptr);
}

//===----------------------------------------------------------------------===//
// Struct+trailing allocation tests
//===----------------------------------------------------------------------===//

TEST(AllocatorStruct, MallocWithTrailingBasic) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(iree_allocator_malloc_with_trailing(iree_allocator_system(),
                                                     64, 128, &ptr));
  ASSERT_NE(ptr, nullptr);
  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(AllocatorStruct, MallocWithTrailingOverflow) {
  void* ptr = nullptr;
  // Use values that sum to overflow (each > MAX/2).
  EXPECT_THAT(Status(iree_allocator_malloc_with_trailing(
                  iree_allocator_system(), IREE_HOST_SIZE_MAX / 2 + 100,
                  IREE_HOST_SIZE_MAX / 2 + 100, &ptr)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST(AllocatorStruct, MallocStructArrayBasic) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(iree_allocator_malloc_struct_array(iree_allocator_system(), 64,
                                                    10, 8, &ptr));
  ASSERT_NE(ptr, nullptr);
  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(AllocatorStruct, MallocStructArrayOverflow) {
  void* ptr = nullptr;
  EXPECT_THAT(Status(iree_allocator_malloc_struct_array(
                  iree_allocator_system(), 64, IREE_HOST_SIZE_MAX, 2, &ptr)),
              StatusIs(StatusCode::kOutOfRange));
}

//===----------------------------------------------------------------------===//
// iree_allocator_grow_array tests
//===----------------------------------------------------------------------===//

TEST(GrowArray, BasicGrowth) {
  uint64_t* ptr = nullptr;
  iree_host_size_t capacity = 0;
  // Initial allocation from 0 capacity with minimum of 8.
  IREE_EXPECT_OK(iree_allocator_grow_array(
      iree_allocator_system(), 8, sizeof(uint64_t), &capacity, (void**)&ptr));
  EXPECT_EQ(capacity, 8u);
  EXPECT_NE(ptr, nullptr);

  // Write to verify allocation is valid.
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    ptr[i] = i * 100;
  }

  // Grow from 8 to 16 (doubled).
  IREE_EXPECT_OK(iree_allocator_grow_array(
      iree_allocator_system(), 8, sizeof(uint64_t), &capacity, (void**)&ptr));
  EXPECT_EQ(capacity, 16u);

  // Verify old data preserved (realloc semantics).
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(ptr[i], i * 100);
  }

  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(GrowArray, MinimumCapacityWins) {
  uint64_t* ptr = nullptr;
  iree_host_size_t capacity = 4;
  // Pre-allocate with capacity 4.
  IREE_EXPECT_OK(iree_allocator_malloc_array(iree_allocator_system(), capacity,
                                             sizeof(uint64_t), (void**)&ptr));
  // Grow: max(32, 4*2=8) = 32.
  IREE_EXPECT_OK(iree_allocator_grow_array(
      iree_allocator_system(), 32, sizeof(uint64_t), &capacity, (void**)&ptr));
  EXPECT_EQ(capacity, 32u);
  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(GrowArray, DoubleWins) {
  uint64_t* ptr = nullptr;
  iree_host_size_t capacity = 100;
  // Pre-allocate with capacity 100.
  IREE_EXPECT_OK(iree_allocator_malloc_array(iree_allocator_system(), capacity,
                                             sizeof(uint64_t), (void**)&ptr));
  // Grow: max(8, 100*2=200) = 200.
  IREE_EXPECT_OK(iree_allocator_grow_array(
      iree_allocator_system(), 8, sizeof(uint64_t), &capacity, (void**)&ptr));
  EXPECT_EQ(capacity, 200u);
  iree_allocator_free(iree_allocator_system(), ptr);
}

TEST(GrowArray, OverflowOnDouble) {
  void* ptr = nullptr;
  iree_host_size_t capacity = IREE_HOST_SIZE_MAX;
  EXPECT_THAT(
      Status(iree_allocator_grow_array(iree_allocator_system(), 8,
                                       sizeof(uint64_t), &capacity, &ptr)),
      StatusIs(StatusCode::kOutOfRange));
}

//===----------------------------------------------------------------------===//
// IREE_STRUCT_LAYOUT tests
//===----------------------------------------------------------------------===//

struct TestHeader {
  void* vtable;
  uint64_t flags;
};

struct TestElement {
  uint64_t data[4];
};

TEST(StructLayout, BasicTwoFields) {
  iree_host_size_t total = 0;
  iree_host_size_t field1_offset = 0;
  iree_host_size_t field2_offset = 0;
  IREE_EXPECT_OK(
      IREE_STRUCT_LAYOUT(sizeof(TestHeader), &total,
                         IREE_STRUCT_FIELD(10, TestElement, &field1_offset),
                         IREE_STRUCT_FIELD(5, uint64_t, &field2_offset)));

  EXPECT_EQ(field1_offset, sizeof(TestHeader));
  EXPECT_EQ(field2_offset, sizeof(TestHeader) + 10 * sizeof(TestElement));
  EXPECT_EQ(total, sizeof(TestHeader) + 10 * sizeof(TestElement) +
                       5 * sizeof(uint64_t));
}

TEST(StructLayout, AlignedFields) {
  iree_host_size_t total = 0;
  iree_host_size_t header_offset = 0;
  iree_host_size_t data_offset = 0;
  IREE_EXPECT_OK(IREE_STRUCT_LAYOUT(
      0, &total, IREE_STRUCT_FIELD_ALIGNED(1, TestHeader, 16, &header_offset),
      IREE_STRUCT_FIELD_ALIGNED(100, uint8_t, 64, &data_offset)));

  EXPECT_EQ(header_offset, 0u);
  // TestHeader is 16 bytes, data should be at offset 64 (aligned).
  EXPECT_EQ(data_offset, 64u);
  EXPECT_EQ(total, 64u + 100u);
}

TEST(StructLayout, NullOffsetPointer) {
  iree_host_size_t total = 0;
  IREE_EXPECT_OK(IREE_STRUCT_LAYOUT(sizeof(TestHeader), &total,
                                    IREE_STRUCT_FIELD(10, TestElement, NULL),
                                    IREE_STRUCT_FIELD(5, uint64_t, NULL)));
  EXPECT_EQ(total, sizeof(TestHeader) + 10 * sizeof(TestElement) +
                       5 * sizeof(uint64_t));
}

TEST(StructLayout, ZeroCountField) {
  iree_host_size_t total = 0;
  iree_host_size_t offset = 0;
  IREE_EXPECT_OK(IREE_STRUCT_LAYOUT(
      sizeof(TestHeader), &total, IREE_STRUCT_FIELD(0, TestElement, &offset)));
  EXPECT_EQ(offset, sizeof(TestHeader));
  EXPECT_EQ(total, sizeof(TestHeader));
}

TEST(StructLayout, OverflowOnMultiply) {
  iree_host_size_t total = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      IREE_STRUCT_LAYOUT(0, &total,
                         IREE_STRUCT_FIELD(SIZE_MAX, TestElement, NULL)));
}

TEST(StructLayout, OverflowOnAdd) {
  iree_host_size_t total = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      IREE_STRUCT_LAYOUT(SIZE_MAX - 10, &total,
                         IREE_STRUCT_FIELD(100, uint8_t, NULL)));
}

TEST(StructLayout, SingleField) {
  iree_host_size_t total = 0;
  iree_host_size_t offset = 0;
  IREE_EXPECT_OK(IREE_STRUCT_LAYOUT(
      sizeof(TestHeader), &total, IREE_STRUCT_FIELD(64, TestElement, &offset)));
  EXPECT_EQ(offset, sizeof(TestHeader));
  EXPECT_EQ(total, sizeof(TestHeader) + 64 * sizeof(TestElement));
}

TEST(StructLayout, ThreeFields) {
  iree_host_size_t total = 0;
  iree_host_size_t offset1 = 0, offset2 = 0, offset3 = 0;
  IREE_EXPECT_OK(IREE_STRUCT_LAYOUT(sizeof(TestHeader), &total,
                                    IREE_STRUCT_FIELD(10, uint8_t, &offset1),
                                    IREE_STRUCT_FIELD(20, uint16_t, &offset2),
                                    IREE_STRUCT_FIELD(30, uint32_t, &offset3)));
  EXPECT_EQ(offset1, sizeof(TestHeader));
  EXPECT_EQ(offset2, sizeof(TestHeader) + 10);
  EXPECT_EQ(offset3, sizeof(TestHeader) + 10 + 40);
  EXPECT_EQ(total, sizeof(TestHeader) + 10 + 40 + 120);
}

//===----------------------------------------------------------------------===//
// iree_allocator_malloc_aligned tests
//===----------------------------------------------------------------------===//

TEST(AllocatorAligned, MallocBasic) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(
      iree_allocator_malloc_aligned(iree_allocator_system(), 128, 64, 0, &ptr));
  ASSERT_NE(ptr, nullptr);
  // Verify alignment.
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
  // Verify memory is zeroed.
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(ptr)[i], 0);
  }
  iree_allocator_free_aligned(iree_allocator_system(), ptr);
}

TEST(AllocatorAligned, MallocLargeAlignment) {
  void* ptr = nullptr;
  // Test with 4096-byte (page) alignment.
  IREE_EXPECT_OK(iree_allocator_malloc_aligned(iree_allocator_system(), 256,
                                               4096, 0, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 4096, 0u);
  iree_allocator_free_aligned(iree_allocator_system(), ptr);
}

TEST(AllocatorAligned, MallocWithOffset) {
  // Allocate with an offset so that a specific byte is aligned.
  void* ptr = nullptr;
  iree_host_size_t offset = 32;
  IREE_EXPECT_OK(iree_allocator_malloc_aligned(iree_allocator_system(), 256, 64,
                                               offset, &ptr));
  ASSERT_NE(ptr, nullptr);
  // The byte at 'offset' from ptr should be 64-byte aligned.
  EXPECT_EQ((reinterpret_cast<uintptr_t>(ptr) + offset) % 64, 0u);
  iree_allocator_free_aligned(iree_allocator_system(), ptr);
}

TEST(AllocatorAligned, ReallocGrow) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(
      iree_allocator_malloc_aligned(iree_allocator_system(), 64, 64, 0, &ptr));
  ASSERT_NE(ptr, nullptr);

  // Write pattern to verify data preservation.
  for (int i = 0; i < 64; ++i) {
    static_cast<uint8_t*>(ptr)[i] = static_cast<uint8_t>(i);
  }

  // Grow the allocation.
  IREE_EXPECT_OK(iree_allocator_realloc_aligned(iree_allocator_system(), 256,
                                                64, 0, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);

  // Verify original data preserved.
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(ptr)[i], static_cast<uint8_t>(i));
  }

  iree_allocator_free_aligned(iree_allocator_system(), ptr);
}

TEST(AllocatorAligned, ReallocShrink) {
  void* ptr = nullptr;
  IREE_EXPECT_OK(
      iree_allocator_malloc_aligned(iree_allocator_system(), 256, 64, 0, &ptr));
  ASSERT_NE(ptr, nullptr);

  // Write pattern.
  for (int i = 0; i < 64; ++i) {
    static_cast<uint8_t*>(ptr)[i] = static_cast<uint8_t>(i);
  }

  // Shrink the allocation.
  IREE_EXPECT_OK(
      iree_allocator_realloc_aligned(iree_allocator_system(), 64, 64, 0, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);

  // Verify data preserved in the retained region.
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(ptr)[i], static_cast<uint8_t>(i));
  }

  iree_allocator_free_aligned(iree_allocator_system(), ptr);
}

TEST(AllocatorAligned, ReallocFromNull) {
  void* ptr = nullptr;
  // Realloc with NULL acts like malloc.
  IREE_EXPECT_OK(iree_allocator_realloc_aligned(iree_allocator_system(), 128,
                                                64, 0, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
  iree_allocator_free_aligned(iree_allocator_system(), ptr);
}

TEST(AllocatorAligned, InvalidAlignment) {
  void* ptr = nullptr;
  // Non-power-of-two alignment should fail.
  EXPECT_THAT(Status(iree_allocator_malloc_aligned(iree_allocator_system(), 128,
                                                   63, 0, &ptr)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(AllocatorAligned, ZeroSize) {
  void* ptr = nullptr;
  // Zero-size allocation should fail.
  EXPECT_THAT(Status(iree_allocator_malloc_aligned(iree_allocator_system(), 0,
                                                   64, 0, &ptr)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(AllocatorAligned, OverflowCheck) {
  void* ptr = nullptr;
  // Request a size that would overflow when adding alignment padding.
  EXPECT_THAT(
      Status(iree_allocator_malloc_aligned(
          iree_allocator_system(), IREE_HOST_SIZE_MAX - 10, 64, 0, &ptr)),
      StatusIs(StatusCode::kOutOfRange));
}

}  // namespace
