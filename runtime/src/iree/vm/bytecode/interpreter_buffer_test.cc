// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_buffer.h"

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::vm::bytecode {
namespace {

TEST(VMBytecodeInterpreterBufferTest, ChecksExactBufferRefs) {
  iree_vm_ref_types_t types = {};
  IREE_ASSERT_OK(
      iree_vm_ref_types_resolve(iree_vm_buffer_provider_table(), &types));
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(8, 0, iree_allocator_system(), &buffer));

  iree_vm_buffer_t* resolved_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_buffer_check_deref(
      iree_vm_buffer_ref_from_ptr_borrowed(&types, buffer), types.buffer,
      &resolved_buffer));
  EXPECT_EQ(resolved_buffer, buffer);

  resolved_buffer = reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_bytecode_buffer_check_deref(iree_vm_ref_null(), types.buffer,
                                          &resolved_buffer));
  EXPECT_EQ(resolved_buffer, reinterpret_cast<iree_vm_buffer_t*>(uintptr_t{1}));

  iree_vm_buffer_release(buffer);
}

TEST(VMBytecodeInterpreterBufferTest,
     MapsScaledRangesWithoutPublishingFailures) {
  std::array<uint8_t, 16> storage = {};
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(storage.data(), storage.size()),
                          iree_vm_buffer_release_callback_null(),
                          iree_allocator_system(), &buffer));

  iree_byte_span_t span = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_bytecode_buffer_map_lanes(
      buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, /*base=*/2, /*index=*/3,
      /*scale=*/2, /*access_length=*/4, &span));
  EXPECT_EQ(span.data, storage.data() + 8);
  EXPECT_EQ(span.data_length, 4u);

  const iree_byte_span_t sentinel = iree_make_byte_span(storage.data() + 15, 1);
  span = sentinel;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_vm_bytecode_buffer_map_lanes(
          buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, /*base=*/0,
          /*index=*/UINT64_MAX, /*scale=*/2, /*access_length=*/1, &span));
  EXPECT_EQ(span.data, sentinel.data);
  EXPECT_EQ(span.data_length, sentinel.data_length);

  span = sentinel;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_vm_bytecode_buffer_map_lanes(
          buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, /*base=*/UINT64_MAX,
          /*index=*/1, /*scale=*/1, /*access_length=*/1, &span));
  EXPECT_EQ(span.data, sentinel.data);
  EXPECT_EQ(span.data_length, sentinel.data_length);

  span = sentinel;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_vm_bytecode_buffer_map_lanes(
          buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
          /*base=*/0, /*index=*/0, /*scale=*/1, /*access_length=*/1, &span));
  EXPECT_EQ(span.data, sentinel.data);
  EXPECT_EQ(span.data_length, sentinel.data_length);

  iree_vm_buffer_release(buffer);
}

void ExpectAtomicMappingWidth(iree_host_size_t carrier_byte_length) {
  SCOPED_TRACE(carrier_byte_length);
  alignas(8) std::array<uint8_t, 17> storage = {};
  iree_vm_buffer_t* aligned_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_byte_span(storage.data(), 16),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &aligned_buffer));

  uint8_t* address = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_buffer_map_atomic(
      aligned_buffer, carrier_byte_length, carrier_byte_length, &address));
  EXPECT_EQ(address, storage.data() + carrier_byte_length);

  uint8_t* const sentinel = storage.data() + 16;
  address = sentinel;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_vm_bytecode_buffer_map_atomic(
                            aligned_buffer, 17 - carrier_byte_length,
                            carrier_byte_length, &address));
  EXPECT_EQ(address, sentinel);
  iree_vm_buffer_release(aligned_buffer);

  iree_vm_buffer_t* misaligned_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_byte_span(storage.data() + 1, carrier_byte_length),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &misaligned_buffer));
  address = sentinel;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_bytecode_buffer_map_atomic(misaligned_buffer, 0,
                                         carrier_byte_length, &address));
  EXPECT_EQ(address, sentinel);
  iree_vm_buffer_release(misaligned_buffer);

  iree_vm_buffer_t* read_only_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span(storage.data(), carrier_byte_length),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &read_only_buffer));
  address = sentinel;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_vm_bytecode_buffer_map_atomic(read_only_buffer, 0,
                                         carrier_byte_length, &address));
  EXPECT_EQ(address, sentinel);
  iree_vm_buffer_release(read_only_buffer);

  iree_vm_buffer_t* write_only_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_byte_span(storage.data(), carrier_byte_length),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &write_only_buffer));
  address = sentinel;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_vm_bytecode_buffer_map_atomic(write_only_buffer, 0,
                                         carrier_byte_length, &address));
  EXPECT_EQ(address, sentinel);
  iree_vm_buffer_release(write_only_buffer);
}

TEST(VMBytecodeInterpreterBufferTest,
     MapsOnlyCompleteNaturallyAlignedReadWriteCarriers) {
  ExpectAtomicMappingWidth(4);
  ExpectAtomicMappingWidth(8);
}

}  // namespace
}  // namespace iree::vm::bytecode
