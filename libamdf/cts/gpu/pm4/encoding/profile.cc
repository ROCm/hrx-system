// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/encoding/profile.h"

#include <algorithm>

namespace {

// PAL's GFX11 GCR includes metadata and shared-L1 invalidation. Scalar stores
// are absent from these programs, so scalar writeback is not requested.
constexpr Pm4CommandProfile kGfx11 = {
    .system_acquire_gcr = (1u << 0) | (1u << 5) | (1u << 7) | (1u << 8) |
                          (1u << 9) | (1u << 14) | (1u << 15),
    .system_release_gcr =
        (1u << 1) | (1u << 2) | (1u << 3) | (1u << 8) | (1u << 9),
    .gl2_writeback_gcr = 1u << 15,
    .resource3_register = 0x2e28,
    .scratch_size_register = 0x2e18,
    .instruction_prefetch_mask = 0x3f,
    .group_segment_unit_byte_length = 512,
    .supports_wave64 = true,
};

// PAL's GFX12 reserves the former metadata/shared-L1 actions. The common
// scalar/vector/GL2 controls retain their positions and prefetch grows to 8
// bits.
constexpr Pm4CommandProfile kGfx12 = {
    .system_acquire_gcr =
        (1u << 0) | (1u << 7) | (1u << 8) | (1u << 14) | (1u << 15),
    .system_release_gcr = (1u << 2) | (1u << 8) | (1u << 9),
    .gl2_writeback_gcr = 1u << 15,
    .resource3_register = 0x2e28,
    .scratch_size_register = 0x2e18,
    .instruction_prefetch_mask = 0xff,
    .group_segment_unit_byte_length = 512,
    .supports_wave64 = true,
};

// Compiler GFX12.5 uses GC12.1. Linux's gfx_v12_1 packet/register definitions
// add explicit GL2 scope, writable V$, and a moved RSRC3 register. Force-all
// scope covers system backing; forward sequencing drains V$ before GL2.
constexpr Pm4CommandProfile kGfx125 = {
    .system_acquire_gcr = (1u << 0) | (2u << 4) | (1u << 6) | (1u << 7) |
                          (1u << 8) | (1u << 14) | (1u << 15) | (1u << 16),
    .system_release_gcr =
        2u | (1u << 2) | (1u << 8) | (1u << 9) | (1u << 10) | (1u << 12),
    .gl2_writeback_gcr = (2u << 4) | (1u << 15),
    .resource3_register = 0x2e23,
    .scratch_size_register = 0x2e16,
    .instruction_prefetch_mask = 0xff,
    .group_segment_unit_byte_length = 2048,
    .supports_wave64 = false,
};

}  // namespace

const Pm4CommandProfile* Pm4CommandProfile::Find(
    const amdf_gpu_endpoint_info_t& info) {
  const auto& gfx = info.gfx_ip;
  if (gfx.major == 11 && (gfx.minor == 0 || gfx.minor == 5 || gfx.minor == 7)) {
    return &kGfx11;
  }
  if (gfx.major == 12 && gfx.minor == 0) {
    return &kGfx12;
  }
  if (gfx.major == 12 && gfx.minor == 5) {
    return &kGfx125;
  }
  return nullptr;
}

uint64_t Pm4CommandProfile::CodeByteLength(uint32_t image_byte_length,
                                           uint32_t entry_byte_offset,
                                           uint32_t resource3) const {
  const uint64_t prefetch_extent =
      uint64_t{entry_byte_offset} +
      ((resource3 >> 4) & instruction_prefetch_mask) * 128u;
  // PAL backs the complete upload through three additional 64-byte fetch lines,
  // independently of the per-program entry-relative prefetch request.
  const uint64_t image_extent =
      ((uint64_t{image_byte_length} + 63u) & ~UINT64_C(63)) + 192u;
  return std::max(image_extent, prefetch_extent);
}

uint32_t Pm4CommandProfile::EncodeGroupSegmentByteLength(
    uint32_t byte_length) const {
  return byte_length / group_segment_unit_byte_length +
         (byte_length % group_segment_unit_byte_length != 0);
}
