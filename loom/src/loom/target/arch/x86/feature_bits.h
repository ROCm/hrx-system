// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lightweight x86 processor feature bit vocabulary.
//
// These bits are used by target configs and target contracts that need a dense,
// arch-local feature set without depending on a specific contract matcher.

#ifndef LOOM_TARGET_ARCH_X86_FEATURE_BITS_H_
#define LOOM_TARGET_ARCH_X86_FEATURE_BITS_H_

#include "iree/base/api.h"

// Bitset of x86 processor features available on a selected target.
typedef uint64_t loom_x86_feature_bits_t;

// Target supports AVX-512 VNNI byte/word dot products.
#define LOOM_X86_FEATURE_AVX512_VNNI (UINT64_C(1) << 0)
// Target supports AVX-512 VL encodings for 128-bit and 256-bit forms.
#define LOOM_X86_FEATURE_AVX512_VL (UINT64_C(1) << 1)
// Target supports VEX AVX-VNNI 128-bit and 256-bit forms.
#define LOOM_X86_FEATURE_AVX_VNNI (UINT64_C(1) << 2)
// Target supports AVX-VNNI-INT8 byte dot-product forms.
#define LOOM_X86_FEATURE_AVX_VNNI_INT8 (UINT64_C(1) << 3)
// Target supports AVX-VNNI-INT16 word dot-product forms.
#define LOOM_X86_FEATURE_AVX_VNNI_INT16 (UINT64_C(1) << 4)
// Target supports AVX10.2 extended VNNI dot products.
#define LOOM_X86_FEATURE_AVX10_2 (UINT64_C(1) << 5)
// Target supports AVX-512 BF16 dot products.
#define LOOM_X86_FEATURE_AVX512_BF16 (UINT64_C(1) << 6)

#endif  // LOOM_TARGET_ARCH_X86_FEATURE_BITS_H_
