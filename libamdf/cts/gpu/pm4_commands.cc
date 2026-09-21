// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4_commands.h"

#include <cstring>

namespace {

uint32_t MakeHeader(uint32_t opcode, size_t word_count) {
  return (UINT32_C(3) << 30) | (opcode << 8) |
         (static_cast<uint32_t>(word_count - 2) << 16);
}

}  // namespace

void Pm4CommandWriter::SystemBarrier() {
  enum : uint32_t {
    kEventWriteOpcode = 0x46,
    kAcquireMemoryOpcode = 0x58,
    kComputeShaderPartialFlush = 7 | (4 << 8),
    kConservativeGcrControl = (3 << 0) | (1 << 4) | (1 << 5) | (1 << 7) |
                              (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15),
  };
  words_[word_count_++] = MakeHeader(kEventWriteOpcode, 2);
  words_[word_count_++] = kComputeShaderPartialFlush;
  words_[word_count_++] = MakeHeader(kAcquireMemoryOpcode, 8);
  words_[word_count_++] = 0;
  words_[word_count_++] = UINT32_MAX;
  words_[word_count_++] = 0xff;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0x0a;
  words_[word_count_++] = kConservativeGcrControl;
}

void Pm4CommandWriter::CopyData32(uint64_t source_address,
                                  uint64_t target_address) {
  enum : uint32_t {
    kCopyDataOpcode = 0x40,
    kSourceTcL2 = 2 << 0,
    kTargetTcL2 = 2 << 8,
    kWaitForConfirmation = 1 << 20,
  };
  words_[word_count_++] = MakeHeader(kCopyDataOpcode, 6);
  words_[word_count_++] = kSourceTcL2 | kTargetTcL2 | kWaitForConfirmation;
  words_[word_count_++] = static_cast<uint32_t>(source_address);
  words_[word_count_++] = static_cast<uint32_t>(source_address >> 32);
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
}

void Pm4CommandWriter::WriteData32(uint64_t target_address, uint32_t value) {
  enum : uint32_t {
    kWriteDataOpcode = 0x37,
    kTargetTcL2 = 2 << 8,
    kWaitForConfirmation = 1 << 20,
  };
  words_[word_count_++] = MakeHeader(kWriteDataOpcode, 5);
  words_[word_count_++] = kTargetTcL2 | kWaitForConfirmation;
  words_[word_count_++] = static_cast<uint32_t>(target_address);
  words_[word_count_++] = static_cast<uint32_t>(target_address >> 32);
  words_[word_count_++] = value;
}

void Pm4CommandWriter::Noop(size_t word_count) {
  words_[word_count_++] = MakeHeader(0x10, word_count);
  std::memset(words_ + word_count_, 0, (word_count - 1) * sizeof(*words_));
  word_count_ += word_count - 1;
}

void Pm4CommandWriter::PadToEightWords() {
  size_t padding = 8 - word_count_ % 8;
  if (padding == 1) {
    padding += 8;
  }
  Noop(padding);
}
