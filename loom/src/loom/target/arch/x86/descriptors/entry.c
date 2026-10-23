// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/descriptors/entry.h"

#include "loom/target/arch/x86/descriptors/avx2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx2_features_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_features_descriptors.h"
#include "loom/target/arch/x86/descriptors/scalar_descriptors.h"

#define LOOM_X86_ENTRY_BASE_FIELDS(prefix)                              \
  .load_u8 = prefix##_DESCRIPTOR_REF_SCALAR_MOVZX_LOAD_U8_GPR32,        \
  .load_u16 = prefix##_DESCRIPTOR_REF_SCALAR_MOVZX_LOAD_U16_GPR32,      \
  .load_u32 = prefix##_DESCRIPTOR_REF_SCALAR_MOV_LOAD_GPR32,            \
  .load_u64 = prefix##_DESCRIPTOR_REF_SCALAR_MOV_LOAD_GPR64,            \
  .store_u8 = prefix##_DESCRIPTOR_REF_SCALAR_MOV_STORE_U8_GPR32,        \
  .store_u16 = prefix##_DESCRIPTOR_REF_SCALAR_MOV_STORE_U16_GPR32,      \
  .store_u32 = prefix##_DESCRIPTOR_REF_SCALAR_MOV_STORE_GPR32,          \
  .store_u64 = prefix##_DESCRIPTOR_REF_SCALAR_MOV_STORE_GPR64,          \
  .add_u64 = prefix##_DESCRIPTOR_REF_SCALAR_ADD_GPR64,                  \
  .mask_u32 = prefix##_DESCRIPTOR_REF_SCALAR_AND_IMM_GPR32,             \
  .widen_u32 = prefix##_DESCRIPTOR_REF_SCALAR_MOVZX_GPR64_GPR32,        \
  .truncate_u64 = prefix##_DESCRIPTOR_REF_SCALAR_MOV_TRUNC_GPR32_GPR64, \
  .shift_left_u64 = prefix##_DESCRIPTOR_REF_SCALAR_SHL_IMM_GPR64,       \
  .shift_right_u64 = prefix##_DESCRIPTOR_REF_SCALAR_SHR_IMM_GPR64,      \
  .or_u64 = prefix##_DESCRIPTOR_REF_SCALAR_OR_GPR64,                    \
  .constant_u32 = prefix##_DESCRIPTOR_REF_SCALAR_MOVIMM_GPR32,          \
  .constant_u64 = prefix##_DESCRIPTOR_REF_SCALAR_MOVIMM_GPR64,          \
  .compare_uge_u32 = prefix##_DESCRIPTOR_REF_SCALAR_CMP_UGE_IMM_GPR32,  \
  .symbol_address = prefix##_DESCRIPTOR_REF_SCALAR_LEA_SYMBOL_GPR64,

#define LOOM_X86_ENTRY_AVX2_FIELDS(prefix)                            \
  .move_u32_xmm = prefix##_DESCRIPTOR_REF_AVX2_VMOVD_XMM_GPR32,       \
  .move_u64_xmm = prefix##_DESCRIPTOR_REF_AVX2_VMOVQ_XMM_GPR64,       \
  .move_xmm_u32 = prefix##_DESCRIPTOR_REF_AVX2_VMOVD_GPR32_XMM,       \
  .move_xmm_u64 = prefix##_DESCRIPTOR_REF_AVX2_VMOVQ_GPR64_XMM,       \
  .load_32_xmm = prefix##_DESCRIPTOR_REF_AVX2_VMOVSS_LOAD_XMM,        \
  .load_64_xmm = prefix##_DESCRIPTOR_REF_AVX2_VMOVSD_LOAD_XMM,        \
  .load_128_xmm = prefix##_DESCRIPTOR_REF_AVX2_VMOVDQU32_LOAD_XMM,    \
  .load_256_ymm = prefix##_DESCRIPTOR_REF_AVX2_VMOVDQU32_LOAD_YMM,    \
  .store_128_xmm = prefix##_DESCRIPTOR_REF_AVX2_VMOVDQU32_STORE_XMM,  \
  .store_256_ymm = prefix##_DESCRIPTOR_REF_AVX2_VMOVDQU32_STORE_YMM,  \
  .zero_xmm = prefix##_DESCRIPTOR_REF_AVX2_VXORPS_ZERO_XMM,           \
  .zero_ymm = prefix##_DESCRIPTOR_REF_AVX2_VXORPS_ZERO_YMM,           \
  .subtract_i8_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPSUBB_XMM,         \
  .subtract_i8_ymm = prefix##_DESCRIPTOR_REF_AVX2_VPSUBB_YMM,         \
  .widen_i8_i64_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBQ_XMM_XMM, \
  .widen_i8_i32_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBD_XMM_XMM, \
  .widen_i8_i16_xmm = prefix##_DESCRIPTOR_REF_AVX2_VPMOVSXBW_XMM_XMM,

#define LOOM_X86_ENTRY_AVX512_FIELDS(prefix)                           \
  .load_512_zmm = prefix##_DESCRIPTOR_REF_AVX512_VMOVDQU32_LOAD_ZMM,   \
  .store_512_zmm = prefix##_DESCRIPTOR_REF_AVX512_VMOVDQU32_STORE_ZMM, \
  .zero_zmm = prefix##_DESCRIPTOR_REF_AVX512_VXORPS_ZERO_ZMM,          \
  .subtract_i8_zmm = prefix##_DESCRIPTOR_REF_AVX512_VPSUBB_ZMM,

#define LOOM_X86_ENTRY_SCALAR(prefix)                                         \
  [prefix##_DESCRIPTOR_SET_ORDINAL] = &(const loom_x86_entry_descriptors_t) { \
    .maximum_simd_byte_length = 0, LOOM_X86_ENTRY_BASE_FIELDS(prefix)         \
  }

#define LOOM_X86_ENTRY_AVX2(prefix)                                           \
  [prefix##_DESCRIPTOR_SET_ORDINAL] = &(const loom_x86_entry_descriptors_t) { \
    .maximum_simd_byte_length = 32,                                           \
    LOOM_X86_ENTRY_BASE_FIELDS(prefix) LOOM_X86_ENTRY_AVX2_FIELDS(prefix)     \
  }

#define LOOM_X86_ENTRY_AVX512(prefix)                                         \
  [prefix##_DESCRIPTOR_SET_ORDINAL] = &(const loom_x86_entry_descriptors_t) { \
    .maximum_simd_byte_length = 64,                                           \
    LOOM_X86_ENTRY_BASE_FIELDS(prefix) LOOM_X86_ENTRY_AVX2_FIELDS(prefix)     \
        LOOM_X86_ENTRY_AVX512_FIELDS(prefix)                                  \
  }

static const loom_x86_entry_descriptors_t* const kEntryDescriptors[] = {
    LOOM_X86_ENTRY_SCALAR(X86_SCALAR_CORE),
    LOOM_X86_ENTRY_AVX2(X86_AVX2_CORE),
    LOOM_X86_ENTRY_AVX2(X86_AVX2_FEATURES_CORE),
    LOOM_X86_ENTRY_AVX512(X86_AVX512_CORE),
    LOOM_X86_ENTRY_AVX512(X86_AVX512_FEATURES_CORE),
};
#undef LOOM_X86_ENTRY_AVX512
#undef LOOM_X86_ENTRY_AVX2
#undef LOOM_X86_ENTRY_SCALAR
#undef LOOM_X86_ENTRY_AVX512_FIELDS
#undef LOOM_X86_ENTRY_AVX2_FIELDS
#undef LOOM_X86_ENTRY_BASE_FIELDS

const loom_x86_entry_descriptors_t* loom_x86_entry_descriptors(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set->descriptor_set_ordinal <
                 IREE_ARRAYSIZE(kEntryDescriptors)
             ? kEntryDescriptors[descriptor_set->descriptor_set_ordinal]
             : NULL;
}
