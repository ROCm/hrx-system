// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PM4_ENCODING_PROFILE_H_
#define AMDF_CTS_GPU_PM4_ENCODING_PROFILE_H_

#include <cstdint>

#include "amdf/gpu.h"

// Generation-dependent fields for ordinary PM4 compute commands. Selection
// occurs before native activation; queue and backing capabilities are separate.
struct Pm4CommandProfile {
  // Whole-cache release/acquire GCR after compute completion, including I$.
  uint32_t system_acquire_gcr;
  // RELEASE_MEM GCR before a system-visible marker, in unshifted GCR units.
  uint32_t system_release_gcr;
  // ACQUIRE_MEM GL2 writeback only, including the required scope selection.
  uint32_t gl2_writeback_gcr;
  // COMPUTE_PGM_RSRC3 DWORD address in ordinary shader register space.
  uint32_t resource3_register;
  // COMPUTE_TMPRING_SIZE DWORD address for architected private storage.
  uint32_t scratch_size_register;
  // Unshifted mask of RSRC3.INST_PREF_SIZE, whose unit is 128 bytes.
  uint32_t instruction_prefetch_mask;
  // Byte length represented by one COMPUTE_PGM_RSRC2.LDS_SIZE unit.
  uint32_t group_segment_unit_byte_length;
  // Physical wave64 execution support in addition to the baseline wave32 mode.
  bool supports_wave64;

  // Returns a static profile for a supported physical target, or null.
  static const Pm4CommandProfile* Find(const amdf_gpu_endpoint_info_t& info);

  // Backs the complete image, speculative fetch tail and entry-relative
  // instruction prefetch. Allocation granularity belongs to the memory owner.
  uint64_t CodeByteLength(uint32_t image_byte_length,
                          uint32_t entry_byte_offset, uint32_t resource3) const;

  // Returns the COMPUTE_PGM_RSRC2.LDS_SIZE value covering `byte_length`.
  uint32_t EncodeGroupSegmentByteLength(uint32_t byte_length) const;
};

#endif  // AMDF_CTS_GPU_PM4_ENCODING_PROFILE_H_
