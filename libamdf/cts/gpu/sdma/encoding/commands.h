// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_SDMA_ENCODING_COMMANDS_H_
#define AMDF_CTS_GPU_SDMA_ENCODING_COMMANDS_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "amdf/gpu.h"

// A subwindow within a linear allocation. The packet base address is separate;
// row/slice pitches are positive element counts, not encoded minus-one values.
struct SdmaLinearLayout {
  // Starting element within a row.
  uint32_t x;
  // Starting row within a slice.
  uint32_t y;
  // Starting slice.
  uint32_t z;
  // Distance between row starts, in elements.
  uint32_t row_pitch;
  // Distance between slice starts, in elements; the wide form permits 2^32.
  uint64_t slice_pitch;
};

// Positive logical dimensions before count-minus-one packet encoding.
struct SdmaCopyExtent {
  // Elements copied per row.
  uint32_t width;
  // Rows copied per slice.
  uint32_t height;
  // Slices copied.
  uint32_t depth;
};

// Classic memory comparisons used by completion-value protocols. Relational
// comparisons alone do not provide a wrap-aware timeline protocol.
enum class SdmaMemoryComparison : uint32_t {
  kAlways = 0,
  kLess = 1,
  kLessOrEqual = 2,
  kEqual = 3,
  kNotEqual = 4,
  kGreaterOrEqual = 5,
  kGreater = 6,
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
  // Requires COPY_LINEAR_RECT. All geometry fits the family's field widths
  // and distinct caller-owned backing; bases and byte pitches are
  // DWORD-aligned. element_log2 is 0..4. Scope follows the family; NPD and
  // placement hints stay zero. No cache operation or completion is emitted
  // implicitly.
  void CopyLinearRect(uint64_t source, const SdmaLinearLayout& source_layout,
                      uint64_t target, const SdmaLinearLayout& target_layout,
                      const SdmaCopyExtent& extent, uint32_t element_log2);
  // Copies 1..2^20 inline DWORDs into a DWORD-aligned owned range. The
  // command storage receives its own copy of the values, and must have room
  // for the four header DWORDs and all data. Scope follows the family.
  void WriteLinear(uint64_t target, std::span<const uint32_t> values);
  // Repeats a DWORD pattern over a nonempty DWORD-aligned owned range. The
  // byte length is at most 0x3ffffc, below the conservative 22-bit count bound.
  // Scope follows the family; fill's separate NPD field remains clear.
  void Fill32(uint64_t target, uint32_t pattern, uint32_t byte_length);
  // Writes an aligned coherent completion word after preceding transfers.
  void Fence32(uint64_t address, uint32_t value);
  // Publishes a KFD_SIGNAL v1 mailbox and its interrupt after the caller's
  // payload completion. The event context fits the native 28-bit TRAP field.
  // Event ownership extends through consumption of this notification tail.
  void NotifyKfdEvent(const amdf_gpu_event_info_t& event);
  // Waits for an aligned coherent word using a masked comparison. The reference
  // value has no bits outside the mask. POLL_REGMEM scope follows the family;
  // the native retry-forever value leaves valid work without a deadline.
  void WaitMemory32(
      uint64_t address, uint32_t value,
      SdmaMemoryComparison comparison = SdmaMemoryComparison::kEqual,
      uint32_t mask = UINT32_MAX);
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
