// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_PROGRAM_ATOMIC_H_
#define AMDF_CTS_GPU_KERNELS_PROGRAM_ATOMIC_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kernels::program_atomic {

inline constexpr uint32_t kGroupCount = 4;
inline constexpr uint32_t kWorkgroupSize = 64;
inline constexpr uint32_t kProducerCount = kGroupCount * kWorkgroupSize;

// The 32-bit variant accesses only the low word of each eight-byte cell.
struct Control {
  // Untouched leading guard.
  uint64_t leading_guard;
  // Shared compare/exchange destination; exactly one producer replaces it.
  uint64_t reservation;
  // Shared non-returning ADD destination, initialized to exercise carry/wrap.
  uint64_t counter;
  // Host-owned value loaded with acquire by every producer.
  uint64_t tail;
  // Untouched trailing guard.
  uint64_t trailing_guard;
};
static_assert(sizeof(Control) == 40);

struct Record {
  // Value returned by a deliberately mismatching compare/exchange.
  uint64_t failed_value;
  // Value returned by the one contended compare/exchange attempt.
  uint64_t contended_value;
  // Host tail observed by this producer's acquire load.
  uint64_t tail;
  // Producer-specific proposed value XOR the acquired tail.
  uint64_t payload;
  // Release generation; the 32-bit variant preserves the upper guard word.
  uint64_t ready;
};
static_assert(sizeof(Record) == 40);

struct Arguments {
  // Device address of the complete Control, including its leading guard.
  uint64_t control;
  // Device address of the first producer Record, after its guard record.
  uint64_t records;
  // Initial reservation value expected by every contended attempt.
  uint64_t initial;
  // Deliberately mismatching expected value, differing above bit 31 for CAS64.
  uint64_t mismatch;
  // Publication generation for this finite epoch.
  uint64_t generation;
  // Uniform atomic width, either 32 or 64 bits.
  uint32_t bit_width;
};
static_assert(sizeof(Arguments) == 48);

inline constexpr std::array<uint32_t, 6> kArgumentByteOffsets = {
    offsetof(Arguments, control),    offsetof(Arguments, records),
    offsetof(Arguments, initial),    offsetof(Arguments, mismatch),
    offsetof(Arguments, generation), offsetof(Arguments, bit_width)};
inline constexpr std::array<uint32_t, 6> kArgumentByteLengths = {8, 8, 8,
                                                                 8, 8, 4};
inline constexpr std::array<std::string_view, 6> kArgumentValueKinds = {
    "global_buffer", "global_buffer", "by_value",
    "by_value",      "by_value",      "by_value"};

}  // namespace kernels::program_atomic

#endif  // AMDF_CTS_GPU_KERNELS_PROGRAM_ATOMIC_H_
