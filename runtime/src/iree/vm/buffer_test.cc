// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/buffer.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/test_allocator.h"

namespace {

using iree::vm::testing::CountingAllocator;

struct ReleaseRecord {
  // Number of callback invocations.
  int call_count;
  // Storage pointer received by the callback.
  uint8_t* data;
  // Storage length received by the callback.
  iree_host_size_t length;
};

void RecordRelease(void* user_data, iree_byte_span_t storage) {
  auto* record = static_cast<ReleaseRecord*>(user_data);
  ++record->call_count;
  record->data = storage.data;
  record->length = storage.data_length;
}

TEST(VMBufferTest, CreateUsesOneAlignedSlabAndZeroesPayload) {
  CountingAllocator allocator;
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_create(257, 256, allocator.allocator(), &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(iree_vm_buffer_length(buffer), 257u);
  EXPECT_EQ(iree_vm_buffer_access(buffer),
            IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE);

  void* data = iree_vm_buffer_data(buffer);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % 256, 0u);
  const auto* bytes =
      static_cast<const uint8_t*>(iree_vm_buffer_const_data(buffer));
  ASSERT_NE(bytes, nullptr);
  for (iree_host_size_t i = 0; i < iree_vm_buffer_length(buffer); ++i) {
    EXPECT_EQ(bytes[i], 0u);
  }

  iree_vm_buffer_release(buffer);
  EXPECT_EQ(allocator.free_count(), 1u);
}

TEST(VMBufferTest, CreateRejectsInvalidLayoutWithoutAllocation) {
  CountingAllocator allocator;
  iree_vm_buffer_t* buffer = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_buffer_create(16, 3, allocator.allocator(), &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(allocator.allocation_count(), 0u);

  buffer = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_vm_buffer_create(IREE_HOST_SIZE_MAX, 0,
                                              allocator.allocator(), &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(allocator.allocation_count(), 0u);
}

TEST(VMBufferTest, ClonePreservesBytesAndExactAccess) {
  const std::array<uint8_t, 5> source = {1, 2, 3, 4, 5};
  iree_vm_buffer_t* readable = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_const_byte_span(source.data(), source.size()), 0,
      iree_allocator_system(), &readable));
  EXPECT_EQ(iree_vm_buffer_data(readable), nullptr);
  ASSERT_NE(iree_vm_buffer_const_data(readable), nullptr);
  EXPECT_EQ(std::memcmp(iree_vm_buffer_const_data(readable), source.data(),
                        source.size()),
            0);

  uint8_t sentinel = 0;
  iree_byte_span_t write_span = iree_make_byte_span(&sentinel, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_vm_buffer_map_write(readable, 0, 1, &write_span));
  EXPECT_EQ(write_span.data, &sentinel);
  EXPECT_EQ(write_span.data_length, 1u);
  iree_vm_buffer_release(readable);

  iree_vm_buffer_t* writable = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_const_byte_span(source.data(), source.size()), 0,
      iree_allocator_system(), &writable));
  EXPECT_NE(iree_vm_buffer_data(writable), nullptr);
  EXPECT_EQ(iree_vm_buffer_const_data(writable), nullptr);
  iree_const_byte_span_t read_span = iree_make_const_byte_span(&sentinel, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_vm_buffer_map_read(writable, 0, 1, &read_span));
  EXPECT_EQ(read_span.data, &sentinel);
  EXPECT_EQ(read_span.data_length, 1u);
  iree_vm_buffer_release(writable);
}

TEST(VMBufferTest, WrapTransfersExactStorageOnlyOnSuccess) {
  std::array<uint8_t, 8> storage = {};
  ReleaseRecord record = {};
  iree_vm_buffer_release_callback_t callback = {
      RecordRelease,
      &record,
  };

  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(storage.data(), storage.size()),
                          callback, iree_allocator_system(), &buffer));
  EXPECT_EQ(record.call_count, 0);
  iree_vm_buffer_release(buffer);
  EXPECT_EQ(record.call_count, 1);
  EXPECT_EQ(record.data, storage.data());
  EXPECT_EQ(record.length, storage.size());

  buffer = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_NONE,
                          iree_make_byte_span(storage.data(), storage.size()),
                          callback, iree_allocator_system(), &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(record.call_count, 1);
}

TEST(VMBufferTest, ConstructionRejectsMalformedNonemptySpans) {
  iree_vm_buffer_t* buffer = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                           iree_make_const_byte_span(nullptr, 1), 0,
                           iree_allocator_system(), &buffer));
  EXPECT_EQ(buffer, nullptr);

  buffer = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(nullptr, 1),
                          iree_vm_buffer_release_callback_null(),
                          iree_allocator_system(), &buffer));
  EXPECT_EQ(buffer, nullptr);
}

TEST(VMBufferTest, NestedViewsFlattenOwnershipToTheRoot) {
  CountingAllocator allocator;
  std::array<uint8_t, 16> storage = {};
  for (iree_host_size_t i = 0; i < storage.size(); ++i) {
    storage[i] = static_cast<uint8_t>(i);
  }
  ReleaseRecord record = {};
  iree_vm_buffer_release_callback_t callback = {
      RecordRelease,
      &record,
  };

  iree_vm_buffer_t* root = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_byte_span(storage.data(), storage.size()), callback,
      allocator.allocator(), &root));
  iree_vm_buffer_t* first_view = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_subspan(
      root, 2, 12,
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      allocator.allocator(), &first_view));
  iree_vm_buffer_t* nested_view = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_subspan(first_view, 3, 4,
                                        IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                        allocator.allocator(), &nested_view));
  EXPECT_EQ(allocator.allocation_count(), 3u);

  iree_vm_buffer_release(root);
  iree_vm_buffer_release(first_view);
  EXPECT_EQ(allocator.free_count(), 1u);
  EXPECT_EQ(record.call_count, 0);

  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_buffer_map_read(nested_view, 0, 4, &span));
  ASSERT_EQ(span.data_length, 4u);
  EXPECT_EQ(span.data[0], 5u);
  EXPECT_EQ(span.data[3], 8u);

  iree_vm_buffer_release(nested_view);
  EXPECT_EQ(record.call_count, 1);
  EXPECT_EQ(allocator.free_count(), 3u);
}

TEST(VMBufferTest, WholeRangeSameAccessRetainsIdentity) {
  CountingAllocator allocator;
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(8, 0, allocator.allocator(), &buffer));
  iree_vm_buffer_t* alias = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_subspan(
      buffer, 0, 8,
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      allocator.allocator(), &alias));
  EXPECT_EQ(alias, buffer);
  EXPECT_EQ(allocator.allocation_count(), 1u);

  iree_vm_buffer_release(buffer);
  EXPECT_NE(iree_vm_buffer_data(alias), nullptr);
  EXPECT_EQ(allocator.free_count(), 0u);
  iree_vm_buffer_release(alias);
  EXPECT_EQ(allocator.free_count(), 1u);
}

TEST(VMBufferTest, SubspanAttenuatesAccessAndRejectsInvalidRanges) {
  iree_vm_buffer_t* root = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(8, 0, iree_allocator_system(), &root));

  iree_vm_buffer_t* readable = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_subspan(root, 2, 4,
                                        IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                        iree_allocator_system(), &readable));
  EXPECT_EQ(iree_vm_buffer_access(readable), IREE_VM_BUFFER_ACCESS_FLAG_READ);
  EXPECT_EQ(iree_vm_buffer_data(readable), nullptr);
  EXPECT_NE(iree_vm_buffer_const_data(readable), nullptr);

  iree_vm_buffer_t* invalid = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_vm_buffer_subspan(
          readable, 0, 1,
          IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
          iree_allocator_system(), &invalid));
  EXPECT_EQ(invalid, nullptr);

  invalid = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_vm_buffer_subspan(root, 9, 0, IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_allocator_system(), &invalid));
  EXPECT_EQ(invalid, nullptr);

  invalid = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_vm_buffer_subspan(root, 1, IREE_HOST_SIZE_MAX,
                             IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_allocator_system(), &invalid));
  EXPECT_EQ(invalid, nullptr);

  iree_vm_buffer_release(readable);
  iree_vm_buffer_release(root);
}

TEST(VMBufferTest, EmptyBuffersRemainDistinctObjects) {
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(0, 0, iree_allocator_system(), &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_vm_buffer_length(buffer), 0u);
  EXPECT_EQ(iree_vm_buffer_access(buffer),
            IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE);

  iree_const_byte_span_t span =
      iree_make_const_byte_span(reinterpret_cast<const void*>(uintptr_t{1}), 1);
  IREE_ASSERT_OK(iree_vm_buffer_map_read(buffer, 0, 0, &span));
  EXPECT_EQ(span.data, nullptr);
  EXPECT_EQ(span.data_length, 0u);

  iree_vm_buffer_t* readable = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_subspan(buffer, 0, 0,
                                        IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                        iree_allocator_system(), &readable));
  ASSERT_NE(readable, nullptr);
  EXPECT_NE(readable, buffer);
  EXPECT_EQ(iree_vm_buffer_length(readable), 0u);

  iree_vm_buffer_release(readable);
  iree_vm_buffer_release(buffer);

  iree_vm_buffer_t* wrapped = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                     iree_byte_span_empty(),
                                     iree_vm_buffer_release_callback_null(),
                                     iree_allocator_system(), &wrapped));
  ASSERT_NE(wrapped, nullptr);
  EXPECT_EQ(iree_vm_buffer_const_data(wrapped), nullptr);
  span =
      iree_make_const_byte_span(reinterpret_cast<const void*>(uintptr_t{1}), 1);
  IREE_ASSERT_OK(iree_vm_buffer_map_read(wrapped, 0, 0, &span));
  EXPECT_EQ(span.data, nullptr);
  EXPECT_EQ(span.data_length, 0u);
  iree_vm_buffer_release(wrapped);
}

}  // namespace
