// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/encoding/memory_commands.h"

#include <cstring>

namespace pm4 {

size_t WaitMemory32(uint32_t* words, uint64_t address, uint32_t value,
                    Pm4MemoryComparison comparison, uint32_t mask) {
  words[0] = 0xc0053c00u;
  words[1] = static_cast<uint32_t>(comparison) | (1u << 4);
  words[2] = static_cast<uint32_t>(address);
  words[3] = static_cast<uint32_t>(address >> 32);
  words[4] = value;
  words[5] = mask;
  words[6] = 4;
  return 7;
}

size_t WaitMemory64(uint32_t* words, uint64_t address, uint64_t value,
                    Pm4MemoryComparison comparison, uint64_t mask) {
  words[0] = 0xc0079300u;
  words[1] = static_cast<uint32_t>(comparison) | (1u << 4);
  words[2] = static_cast<uint32_t>(address);
  words[3] = static_cast<uint32_t>(address >> 32);
  words[4] = static_cast<uint32_t>(value);
  words[5] = static_cast<uint32_t>(value >> 32);
  words[6] = static_cast<uint32_t>(mask);
  words[7] = static_cast<uint32_t>(mask >> 32);
  words[8] = 4;
  return 9;
}

size_t StoreImmediate(uint32_t* words, uint64_t target_address, uint64_t value,
                      CopyDataWidth width) {
  words[0] = (3u << 30) | (4u << 16) | (0x40u << 8);
  // Immediate source, TC/L2 destination, width and CP write confirmation.
  words[1] = 5u | (2u << 8) | (static_cast<uint32_t>(width) << 16) | (1u << 20);
  words[2] = static_cast<uint32_t>(value);
  words[3] = static_cast<uint32_t>(value >> 32);
  words[4] = static_cast<uint32_t>(target_address);
  words[5] = static_cast<uint32_t>(target_address >> 32);
  return 6;
}

size_t CopyData(uint32_t* words, uint64_t source_address,
                uint64_t target_address, CopyDataWidth width) {
  // Type-3 COPY_DATA has six DWORDs, with its count excluding two DWORDs.
  words[0] = (3u << 30) | (4u << 16) | (0x40u << 8);
  // TC/L2 source and destination, default cache policy and write confirmation.
  words[1] =
      (2u << 0) | (2u << 8) | (1u << 20) | (static_cast<uint32_t>(width) << 16);
  words[2] = static_cast<uint32_t>(source_address);
  words[3] = static_cast<uint32_t>(source_address >> 32);
  words[4] = static_cast<uint32_t>(target_address);
  words[5] = static_cast<uint32_t>(target_address >> 32);
  return 6;
}

static size_t AtomicAdd(uint32_t* words, uint64_t target_address,
                        uint64_t value, uint32_t operation) {
  words[0] = 0xc0071e00u;
  // TC opcode, single pass, LRU policy, and no compare/loop operand.
  words[1] = operation;
  words[2] = static_cast<uint32_t>(target_address);
  words[3] = static_cast<uint32_t>(target_address >> 32);
  words[4] = static_cast<uint32_t>(value);
  words[5] = static_cast<uint32_t>(value >> 32);
  words[6] = 0;
  words[7] = 0;
  words[8] = 0;
  return 9;
}

size_t AtomicAdd32(uint32_t* words, uint64_t target_address, uint32_t value) {
  return AtomicAdd(words, target_address, value, 0x0f);
}

size_t AtomicAdd64(uint32_t* words, uint64_t target_address, uint64_t value) {
  return AtomicAdd(words, target_address, value, 0x2f);
}

size_t CopyGpuClock64(uint32_t* words, uint64_t target_address,
                      CopyDataPolicy policy) {
  words[0] = (3u << 30) | (4u << 16) | (0x40u << 8);
  // GPU clock source, MEMORY destination, 64-bit count and confirmation.
  // The two locality fields occupy the same bits across admitted targets.
  words[1] = 9u | (5u << 8) | (static_cast<uint32_t>(policy) << 13) |
             (1u << 16) | (1u << 20) | (static_cast<uint32_t>(policy) << 25);
  words[2] = 0;
  words[3] = 0;
  words[4] = static_cast<uint32_t>(target_address);
  words[5] = static_cast<uint32_t>(target_address >> 32);
  return 6;
}

size_t WriteData(uint32_t* words, uint64_t target_address,
                 const uint32_t* values, size_t value_count) {
  // Four fixed DWORDs plus payload, with the type-3 count excluding two.
  words[0] = (3u << 30) | (0x37u << 8) |
             (static_cast<uint32_t>(value_count + 2) << 16);
  // Incrementing TC/L2 destination, default cache policy and confirmation.
  words[1] = (2u << 8) | (1u << 20);
  words[2] = static_cast<uint32_t>(target_address);
  words[3] = static_cast<uint32_t>(target_address >> 32);
  std::memcpy(words + 4, values, value_count * sizeof(*values));
  return 4 + value_count;
}

}  // namespace pm4
