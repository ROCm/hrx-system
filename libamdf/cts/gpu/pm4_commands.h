// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_COMMANDS_H_
#define AMDF_CTS_GPU_PM4_COMMANDS_H_

#include <cstddef>
#include <cstdint>

// Encodes the CTS memory operations for PM4 format version 1 with ACQUIRE_MEM
// GCR support. Callers supply sufficient command storage, four-byte-aligned
// addresses and a queue admitted for the corresponding transfer/cache roles.
class Pm4CommandWriter {
 public:
  explicit Pm4CommandWriter(uint32_t* words) : words_(words) {}

  void SystemBarrier();
  void CopyData32(uint64_t source_address, uint64_t target_address);
  void WriteData32(uint64_t target_address, uint32_t value);
  void PadToEightWords();
  size_t word_count() const { return word_count_; }

 private:
  // Emits one type-3 NOP of at least two words, including its header.
  void Noop(size_t word_count);

  // Caller-owned command storage, large enough for the known test sequence.
  uint32_t* words_;
  // Number of complete command words emitted into words_.
  size_t word_count_ = 0;
};

#endif  // AMDF_CTS_GPU_PM4_COMMANDS_H_
