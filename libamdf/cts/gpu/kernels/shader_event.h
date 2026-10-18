// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_SHADER_EVENT_H_
#define AMDF_CTS_GPU_KERNELS_SHADER_EVENT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kernels::shader_event {

// One finite producer publishes payload before its authoritative generation.
struct Record {
  // Generation transformed by kPayloadMask, written before the release.
  uint64_t payload;
  // Shader release publication acquired by the host after a native wake.
  uint64_t generation;
};
static_assert(sizeof(Record) == 16);

inline constexpr uint64_t kPayloadMask = UINT64_C(0x123456789abcdef0);

// The native mailbox value and interrupt context are independent operands.
struct Arguments {
  // Device address of the caller-owned Record.
  uint64_t record;
  // Device address of the eight-byte KFD signal mailbox.
  uint64_t mailbox;
  // Authoritative generation, independent of the native event operands.
  uint64_t value;
  // Zero-extended native mailbox trigger, written as one eight-byte value.
  uint64_t trigger;
  // Native interrupt payload placed in M0 for MSG_INTERRUPT.
  uint32_t context;
};
static_assert(sizeof(Arguments) == 40);

inline constexpr std::array<uint32_t, 5> kArgumentByteOffsets = {
    offsetof(Arguments, record), offsetof(Arguments, mailbox),
    offsetof(Arguments, value), offsetof(Arguments, trigger),
    offsetof(Arguments, context)};
inline constexpr std::array<uint32_t, 5> kArgumentByteLengths = {
    sizeof(Arguments::record), sizeof(Arguments::mailbox),
    sizeof(Arguments::value), sizeof(Arguments::trigger),
    sizeof(Arguments::context)};
inline constexpr std::array<std::string_view, 5> kArgumentValueKinds = {
    "global_buffer", "global_buffer", "by_value", "by_value", "by_value"};

}  // namespace kernels::shader_event

#endif  // AMDF_CTS_GPU_KERNELS_SHADER_EVENT_H_
