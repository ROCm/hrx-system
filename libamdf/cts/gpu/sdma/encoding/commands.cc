// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/sdma/encoding/commands.h"

#include <cstring>

void SdmaCommandWriter::AcquireFromSystem() {
  words_[word_count_++] = 17u | (1u << 8);
  words_[word_count_++] = 0;
  words_[word_count_++] = 0xc3c0u << 16;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
}

void SdmaCommandWriter::ReleaseToSystem() {
  words_[word_count_++] = 17u | (1u << 8);
  words_[word_count_++] = 0;
  words_[word_count_++] = 0x8040u << 16;
  words_[word_count_++] = 0;
  words_[word_count_++] = 0;
}

void SdmaCommandWriter::Noop() { words_[word_count_++] = 0; }

void SdmaCommandWriter::CopyLinear(uint64_t source, uint64_t target,
                                   uint32_t byte_length) {
  const bool scoped =
      (features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  // Scope selection does not imply no-prior-dependency (NPD). Keep NPD clear
  // for streams containing dependent transfers.
  words_[word_count_++] = 1;
  words_[word_count_++] = byte_length - 1;
  words_[word_count_++] = scoped ? (3u << 18) | (3u << 26) : 0;
  words_[word_count_++] = static_cast<uint32_t>(source);
  words_[word_count_++] = static_cast<uint32_t>(source >> 32);
  words_[word_count_++] = static_cast<uint32_t>(target);
  words_[word_count_++] = static_cast<uint32_t>(target >> 32);
}

void SdmaCommandWriter::CopyLinearRect(uint64_t source,
                                       const SdmaLinearLayout& source_layout,
                                       uint64_t target,
                                       const SdmaLinearLayout& target_layout,
                                       const SdmaCopyExtent& extent,
                                       uint32_t element_log2) {
  const uint32_t pitch_shift =
      (features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE) != 0
          ? 16
          : 13;
  const bool scoped =
      (features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  words_[word_count_++] = 1u | (4u << 8) | (element_log2 << 29);
  words_[word_count_++] = static_cast<uint32_t>(source);
  words_[word_count_++] = static_cast<uint32_t>(source >> 32);
  words_[word_count_++] = source_layout.x | (source_layout.y << 16);
  words_[word_count_++] =
      source_layout.z | ((source_layout.row_pitch - 1) << pitch_shift);
  words_[word_count_++] = static_cast<uint32_t>(source_layout.slice_pitch - 1);
  words_[word_count_++] = static_cast<uint32_t>(target);
  words_[word_count_++] = static_cast<uint32_t>(target >> 32);
  words_[word_count_++] = target_layout.x | (target_layout.y << 16);
  words_[word_count_++] =
      target_layout.z | ((target_layout.row_pitch - 1) << pitch_shift);
  words_[word_count_++] = static_cast<uint32_t>(target_layout.slice_pitch - 1);
  words_[word_count_++] = (extent.width - 1) | ((extent.height - 1) << 16);
  words_[word_count_++] =
      (extent.depth - 1) | (scoped ? (3u << 18) | (3u << 26) : 0);
}

void SdmaCommandWriter::WriteLinear(uint64_t target,
                                    std::span<const uint32_t> values) {
  const bool scoped =
      (features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  words_[word_count_++] = 2;
  words_[word_count_++] = static_cast<uint32_t>(target);
  words_[word_count_++] = static_cast<uint32_t>(target >> 32);
  words_[word_count_++] =
      static_cast<uint32_t>(values.size() - 1) | (scoped ? 3u << 26 : 0);
  std::memcpy(words_ + word_count_, values.data(), values.size_bytes());
  word_count_ += values.size();
}

void SdmaCommandWriter::Fill32(uint64_t target, uint32_t pattern,
                               uint32_t byte_length) {
  const bool scoped =
      (features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  words_[word_count_++] = 11u | (2u << 30) | (scoped ? 3u << 24 : 0);
  words_[word_count_++] = static_cast<uint32_t>(target);
  words_[word_count_++] = static_cast<uint32_t>(target >> 32);
  words_[word_count_++] = pattern;
  words_[word_count_++] = byte_length - 1;
}

void SdmaCommandWriter::Fence32(uint64_t address, uint32_t value) {
  uint32_t header = 5;
  if ((features_ & (AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM)) != 0) {
    header |= 3u << 16;  // Uncached MTYPE, where that field is admitted.
  }
  if ((features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM) != 0) {
    header |= 1u << 20;
  }
  if ((features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0) {
    header |= 3u << 24;
  }
  words_[word_count_++] = header;
  words_[word_count_++] = static_cast<uint32_t>(address);
  words_[word_count_++] = static_cast<uint32_t>(address >> 32);
  words_[word_count_++] = value;
}

void SdmaCommandWriter::NotifyKfdEvent(const amdf_gpu_event_info_t& event) {
  Fence32(event.mailbox_address, event.trigger_data);
  words_[word_count_++] = 6;  // TRAP, with zero suboperation and reserved bits.
  words_[word_count_++] = event.interrupt_context;
}

void SdmaCommandWriter::WriteGlobalTimestamp(uint64_t address) {
  const bool scoped =
      (features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  words_[word_count_++] = 13 | (2u << 8) | (scoped ? 3u << 24 : 0);
  words_[word_count_++] = static_cast<uint32_t>(address);
  words_[word_count_++] = static_cast<uint32_t>(address >> 32);
}

void SdmaCommandWriter::WaitMemory32(uint64_t address, uint32_t value,
                                     SdmaMemoryComparison comparison,
                                     uint32_t mask) {
  words_[word_count_++] =
      8 | (static_cast<uint32_t>(comparison) << 28) | (1u << 31);
  words_[word_count_++] = static_cast<uint32_t>(address);
  words_[word_count_++] = static_cast<uint32_t>(address >> 32);
  words_[word_count_++] = value;
  words_[word_count_++] = mask;
  const bool scoped =
      (features_ & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  words_[word_count_++] = (0xfffu << 16) | 4 | (scoped ? 3u << 28 : 0);
}
