// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_stack.h"

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"

namespace iree::vm::bytecode {
namespace {

TEST(VMBytecodeInterpreterStackTest, LoadsEveryMemoryFormatUnaligned) {
  std::array<uint8_t, 66> storage = {};
  for (iree_host_size_t i = 0; i < 64; ++i) {
    storage[i + 1] = static_cast<uint8_t>(i * 17u + 3u);
  }

  for (uint8_t format = IREE_VM_ISA_MEMORY_FORMAT_I8_X1;
       format <= IREE_VM_ISA_MEMORY_FORMAT_I64_X8; ++format) {
    const uint8_t lane_count = static_cast<uint8_t>(1u << (format & 3u));
    const uint8_t element_bytes = static_cast<uint8_t>(1u << (format >> 2));
    std::array<uint64_t, 9> values = {};
    values.fill(UINT64_C(0xDEADBEEFDEADBEEF));
    iree_vm_bytecode_stack_load_lanes(format, storage.data() + 1,
                                      values.data());
    for (uint8_t lane = 0; lane < lane_count; ++lane) {
      uint64_t expected = 0;
      for (uint8_t byte = 0; byte < element_bytes; ++byte) {
        expected |=
            static_cast<uint64_t>(storage[1 + lane * element_bytes + byte])
            << (byte * 8u);
      }
      EXPECT_EQ(values[lane], expected) << "format=" << unsigned(format);
    }
    EXPECT_EQ(values[lane_count], UINT64_C(0xDEADBEEFDEADBEEF));
  }
}

TEST(VMBytecodeInterpreterStackTest, StoresEveryMemoryFormatUnaligned) {
  const std::array<uint64_t, 8> values = {
      UINT64_C(0x0123456789ABCDEF), UINT64_C(0xFEDCBA9876543210),
      UINT64_C(0x1122334455667788), UINT64_C(0x8877665544332211),
      UINT64_C(0xA5A5A5A5A5A5A5A5), UINT64_C(0x5A5A5A5A5A5A5A5A),
      UINT64_C(0xFFEEDDCCBBAA0099), UINT64_C(0x9900AABBCCDDEEFF),
  };

  for (uint8_t format = IREE_VM_ISA_MEMORY_FORMAT_I8_X1;
       format <= IREE_VM_ISA_MEMORY_FORMAT_I64_X8; ++format) {
    const uint8_t lane_count = static_cast<uint8_t>(1u << (format & 3u));
    const uint8_t element_bytes = static_cast<uint8_t>(1u << (format >> 2));
    const uint8_t access_length = lane_count * element_bytes;
    std::array<uint8_t, 66> storage = {};
    storage.fill(0xCD);
    iree_vm_bytecode_stack_store_lanes(format, values.data(),
                                       storage.data() + 1);
    for (uint8_t lane = 0; lane < lane_count; ++lane) {
      for (uint8_t byte = 0; byte < element_bytes; ++byte) {
        const uint8_t expected =
            static_cast<uint8_t>(values[lane] >> (byte * 8u));
        EXPECT_EQ(storage[1 + lane * element_bytes + byte], expected)
            << "format=" << unsigned(format);
      }
    }
    EXPECT_EQ(storage[0], 0xCD);
    EXPECT_EQ(storage[1 + access_length], 0xCD);
  }
}

}  // namespace
}  // namespace iree::vm::bytecode
