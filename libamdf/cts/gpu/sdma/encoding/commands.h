// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_SDMA_ENCODING_COMMANDS_H_
#define AMDF_CTS_GPU_SDMA_ENCODING_COMMANDS_H_

#include <cstddef>
#include <cstdint>

#include "amdf/gpu.h"

// Full-width classic memory comparisons used by completion-value protocols.
// Greater-or-equal alone does not provide a wrap-aware timeline comparison.
enum class SdmaMemoryComparison : uint32_t {
  kEqual = 3,
  kGreaterOrEqual = 5,
};

// SDMA v1 transfer and timestamp commands on coherent system memory. No
// implicit GCR or HDP operations; those require their own admitted cache
// recipe.
class SdmaCommandWriter {
 public:
  SdmaCommandWriter(uint32_t* words, amdf_queue_format_features_t features)
      : words_(words), features_(features) {}
  // Whole-cache data acquire after dependency waits. Requires USER_GCR;
  // range, VMID, and instruction-cache operations are outside this recipe.
  void AcquireFromSystem();
  // Whole-cache data release after transfers and before completion. Requires
  // USER_GCR; emitting a completion packet remains the caller's responsibility.
  void ReleaseToSystem();
  // Emits a one-DWORD NOP. Its pending-transfer ordering contract belongs to
  // the selected native engine; it publishes no completion or cache operation.
  void Noop();
  // A nonempty range within caller-owned allocations and the admitted limit.
  // Scope follows the family; NPD remains clear independently of that layout.
  void CopyLinear(uint64_t source, uint64_t target, uint32_t byte_length);
  // Repeats a DWORD pattern over a nonempty DWORD-aligned owned range. The
  // byte length is at most 0x3ffffc, below the conservative 22-bit count bound.
  // Scope follows the family; fill's separate NPD field remains clear.
  void Fill32(uint64_t target, uint32_t pattern, uint32_t byte_length);
  // Writes an aligned coherent completion word after preceding transfers.
  void Fence32(uint64_t address, uint32_t value);
  // Waits for an aligned coherent word using a full-width comparison. This
  // POLL_REGMEM scope follows the family. The native retry-forever value leaves
  // valid asynchronous work without a deadline.
  void WaitMemory32(
      uint64_t address, uint32_t value,
      SdmaMemoryComparison comparison = SdmaMemoryComparison::kEqual);
  // Writes the raw 64-bit global timestamp after earlier commands complete.
  // Scope follows the family, with a 32-byte-aligned caller-owned destination.
  // Clock conversion and timestamp-write completion are separate contracts.
  void WriteGlobalTimestamp(uint64_t address);
  size_t word_count() const { return word_count_; }

 private:
  // Caller-owned command storage, sufficient for the known command sequence.
  uint32_t* words_;
  // Native fields admitted by the exact queue family.
  amdf_queue_format_features_t features_;
  // Number of complete command words written.
  size_t word_count_ = 0;
};

#endif  // AMDF_CTS_GPU_SDMA_ENCODING_COMMANDS_H_
