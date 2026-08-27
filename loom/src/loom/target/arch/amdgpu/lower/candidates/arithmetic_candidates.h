// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low arithmetic descriptor candidate rows.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ARITHMETIC_CANDIDATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ARITHMETIC_CANDIDATES_H_

#include <stdint.h>

#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Descriptor refs indexed by the source interpretation for a, b, and c.
typedef loom_amdgpu_descriptor_ref_t loom_amdgpu_fma_mix_descriptor_ref_cube_t
    [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT_]
    [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT_]
    [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT_];

// Descriptor refs indexed by source-0 and source-1 when source-2 is literal.
typedef loom_amdgpu_descriptor_ref_t
    loom_amdgpu_fma_mix_src2_literal_descriptor_ref_table_t
        [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT_]
        [LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT_];

typedef struct loom_amdgpu_packed_ternary_descriptor_candidate_t {
  // Descriptor ref selected when present in the descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Source operand index consumed at each descriptor source position.
  uint8_t source_permutation[LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT];
  // Flags describing selected descriptor packet semantics.
  loom_amdgpu_packed_ternary_flags_t flags;
  // Number of packed 32-bit register units consumed by each packet.
  uint32_t packet_unit_count;
} loom_amdgpu_packed_ternary_descriptor_candidate_t;

// Descriptor refs for f32-result mixed-source V_FMA_MIX_F32 packets.
extern const loom_amdgpu_fma_mix_descriptor_ref_cube_t
    kLoomAmdgpuFmaMixF32DescriptorRefs;

// Descriptor refs for V_FMA_MIX_F32 source-2 literal-zero packets.
extern const loom_amdgpu_fma_mix_src2_literal_descriptor_ref_table_t
    kLoomAmdgpuFmaMixF32Src2LiteralDescriptorRefs;

// Descriptor refs for V_FMA_MIXLO_F16 source-2 literal-zero packets.
extern const loom_amdgpu_fma_mix_src2_literal_descriptor_ref_table_t
    kLoomAmdgpuFmaMixloF16Src2LiteralDescriptorRefs;

// Descriptor refs for V_FMA_MIXHI_F16 source-2 literal-zero packets.
extern const loom_amdgpu_fma_mix_src2_literal_descriptor_ref_table_t
    kLoomAmdgpuFmaMixhiF16Src2LiteralDescriptorRefs;

// Descriptor refs for f32-result mixed-source V_MAD_MIX_F32 packets.
extern const loom_amdgpu_fma_mix_descriptor_ref_cube_t
    kLoomAmdgpuMadMixF32DescriptorRefs;

// Descriptor refs for low-half-result mixed-source V_FMA_MIXLO_F16 packets.
extern const loom_amdgpu_fma_mix_descriptor_ref_cube_t
    kLoomAmdgpuFmaMixloF16DescriptorRefs;

// Descriptor refs for high-half-result mixed-source V_FMA_MIXHI_F16 packets.
extern const loom_amdgpu_fma_mix_descriptor_ref_cube_t
    kLoomAmdgpuFmaMixhiF16DescriptorRefs;

// Descriptor refs for low-half-result mixed-source V_MAD_MIXLO_F16 packets.
extern const loom_amdgpu_fma_mix_descriptor_ref_cube_t
    kLoomAmdgpuMadMixloF16DescriptorRefs;

// Descriptor refs for high-half-result mixed-source V_MAD_MIXHI_F16 packets.
extern const loom_amdgpu_fma_mix_descriptor_ref_cube_t
    kLoomAmdgpuMadMixhiF16DescriptorRefs;

// Descriptor candidates for packed f16 vector.fmaf packets.
extern const loom_amdgpu_packed_ternary_descriptor_candidate_t
    kLoomAmdgpuPackedFmafF16DescriptorCandidates[];

// Number of packed f16 vector.fmaf descriptor candidates.
enum {
  kLoomAmdgpuPackedFmafF16DescriptorCandidateCount = 2u,
};

// Descriptor candidates for packed f32 vector.fmaf packets.
extern const loom_amdgpu_packed_ternary_descriptor_candidate_t
    kLoomAmdgpuPackedFmafF32DescriptorCandidates[];

// Number of packed f32 vector.fmaf descriptor candidates.
enum {
  kLoomAmdgpuPackedFmafF32DescriptorCandidateCount = 1u,
};

// Descriptor candidates for packed i16 vector.fmai with signed preference.
extern const loom_amdgpu_packed_ternary_descriptor_candidate_t
    kLoomAmdgpuPackedFmaiSignedPreferenceDescriptorCandidates[];

// Number of packed i16 vector.fmai candidates with signed preference.
enum {
  kLoomAmdgpuPackedFmaiSignedPreferenceDescriptorCandidateCount = 2u,
};

// Descriptor candidates for packed i16 vector.fmai with unsigned preference.
extern const loom_amdgpu_packed_ternary_descriptor_candidate_t
    kLoomAmdgpuPackedFmaiUnsignedPreferenceDescriptorCandidates[];

// Number of packed i16 vector.fmai candidates with unsigned preference.
enum {
  kLoomAmdgpuPackedFmaiUnsignedPreferenceDescriptorCandidateCount = 2u,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ARITHMETIC_CANDIDATES_H_
