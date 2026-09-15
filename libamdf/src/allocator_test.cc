// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/allocator.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "gtest/gtest.h"

namespace {

struct TestAllocator {
  // Rejects allocation requests without changing live storage.
  bool fail_allocation = false;
  // Number of allocation requests observed by the callback.
  size_t allocation_count = 0;
  // Number of non-null allocations released through the callback.
  size_t free_count = 0;
  // Number of successful allocations not yet freed.
  size_t live_allocation_count = 0;
  // Alignment supplied to the most recent allocation callback.
  uint64_t requested_alignment = 0;

  static void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                                  uint64_t minimum_alignment) {
    auto* self = static_cast<TestAllocator*>(user_data);
    ++self->allocation_count;
    self->requested_alignment = minimum_alignment;
    if (self->fail_allocation) return nullptr;
#if defined(_WIN32)
    void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                    static_cast<size_t>(minimum_alignment));
#else
    void* pointer = nullptr;
    if (posix_memalign(&pointer, static_cast<size_t>(minimum_alignment),
                       static_cast<size_t>(byte_length)) != 0) {
      pointer = nullptr;
    }
#endif
    if (pointer != nullptr) ++self->live_allocation_count;
    return pointer;
  }

  static void AMDF_CALL Free(void* user_data, void* pointer) {
    auto* self = static_cast<TestAllocator*>(user_data);
    if (pointer == nullptr) return;
    ++self->free_count;
    --self->live_allocation_count;
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
  }

  amdf_allocator_t MakeAllocator() {
    return {
        .user_data = this,
        .allocate = Allocate,
        .free = Free,
    };
  }
};

TEST(AllocatorTest, SystemAllocatorProvidesAlignmentAndZeroing) {
  const amdf_allocator_t allocator = amdf_allocator_system();
  void* pointer = nullptr;
  ASSERT_EQ(amdf_calloc(allocator, 73, 64, &pointer), AMDF_STATUS_OK);
  ASSERT_NE(pointer, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(pointer) & 63u, 0u);
  const uint8_t zeroes[73] = {};
  EXPECT_EQ(std::memcmp(pointer, zeroes, sizeof(zeroes)), 0);
  amdf_free(allocator, pointer);
}

TEST(AllocatorTest, SmallAlignmentIsNormalizedForScalarStorage) {
  TestAllocator state;
  const amdf_allocator_t allocator = state.MakeAllocator();
  void* pointer = nullptr;
  ASSERT_EQ(amdf_malloc(allocator, sizeof(std::max_align_t), 1, &pointer),
            AMDF_STATUS_OK);
  EXPECT_GE(state.requested_alignment, amdf_alignof(std::max_align_t));
  EXPECT_EQ(
      reinterpret_cast<uintptr_t>(pointer) % amdf_alignof(std::max_align_t),
      0u);
  amdf_free(allocator, pointer);
}

TEST(AllocatorTest, ResolveRejectsPartialAllocatorWithoutChangingOutput) {
  TestAllocator state;
  const amdf_allocator_t requested = {
      .user_data = &state,
      .allocate = TestAllocator::Allocate,
  };
  const amdf_allocator_t original = {
      .user_data = reinterpret_cast<void*>(uintptr_t{1}),
      .allocate = reinterpret_cast<amdf_allocator_allocate_fn_t>(uintptr_t{2}),
      .resize = reinterpret_cast<amdf_allocator_resize_fn_t>(uintptr_t{3}),
      .free = reinterpret_cast<amdf_allocator_free_fn_t>(uintptr_t{4}),
  };
  amdf_allocator_t resolved = original;
  EXPECT_EQ(amdf_status_code(amdf_allocator_resolve(&requested, &resolved)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&resolved, &original, sizeof(resolved)), 0);
}

TEST(AllocatorTest, AllocationFailurePreservesOutput) {
  TestAllocator state;
  state.fail_allocation = true;
  const amdf_allocator_t allocator = state.MakeAllocator();
  void* const sentinel = reinterpret_cast<void*>(uintptr_t{1});
  void* pointer = sentinel;
  EXPECT_EQ(amdf_status_code(amdf_malloc(allocator, 64, 16, &pointer)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(pointer, sentinel);
  EXPECT_EQ(state.allocation_count, 1u);
  EXPECT_EQ(state.free_count, 0u);
  EXPECT_EQ(state.live_allocation_count, 0u);
}

TEST(AllocatorTest, OverflowPreservesOutput) {
  const amdf_allocator_t allocator = amdf_allocator_system();
  void* const sentinel = reinterpret_cast<void*>(uintptr_t{1});
  void* pointer = sentinel;
  EXPECT_EQ(amdf_status_code(amdf_calloc_array(allocator, SIZE_MAX, 2,
                                               amdf_max_align_t, &pointer)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(pointer, sentinel);
  EXPECT_EQ(amdf_status_code(amdf_calloc_with_trailing(
                allocator, 2, SIZE_MAX, amdf_max_align_t, &pointer)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(pointer, sentinel);
}

TEST(AllocatorTest, ResizeFallbackPreservesAllocationOnFailure) {
  TestAllocator state;
  const amdf_allocator_t allocator = state.MakeAllocator();
  void* pointer = nullptr;
  ASSERT_EQ(amdf_malloc(allocator, 16, 16, &pointer), AMDF_STATUS_OK);
  ASSERT_NE(pointer, nullptr);
  std::memset(pointer, 0xA5, 16);
  void* const original = pointer;

  state.fail_allocation = true;
  EXPECT_EQ(amdf_status_code(amdf_realloc(allocator, 16, 32, 16, &pointer)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(pointer, original);
  const uint8_t expected[16] = {
      0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5,
      0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5,
  };
  EXPECT_EQ(std::memcmp(pointer, expected, sizeof(expected)), 0);
  EXPECT_EQ(state.free_count, 0u);
  EXPECT_EQ(state.live_allocation_count, 1u);

  state.fail_allocation = false;
  ASSERT_EQ(amdf_realloc(allocator, 16, 32, 16, &pointer), AMDF_STATUS_OK);
  EXPECT_NE(pointer, nullptr);
  EXPECT_EQ(std::memcmp(pointer, expected, sizeof(expected)), 0);
  EXPECT_EQ(state.free_count, 1u);
  EXPECT_EQ(state.live_allocation_count, 1u);
  amdf_free(allocator, pointer);
  EXPECT_EQ(state.free_count, 2u);
  EXPECT_EQ(state.live_allocation_count, 0u);
}

}  // namespace
