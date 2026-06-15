// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lightweight IREE VM feature bit vocabulary.
//
// These bits are used by target configs and low descriptor masks. They share
// the IREE VM bytecode FeatureBits bit positions so descriptor requirements can
// flow directly into emitted module and function descriptors.

#ifndef LOOM_TARGET_ARCH_IREEVM_RECORDS_FEATURE_BITS_H_
#define LOOM_TARGET_ARCH_IREEVM_RECORDS_FEATURE_BITS_H_

#include "iree/base/api.h"

// Bitset of IREE VM bytecode features available on a selected target.
typedef uint64_t loom_ireevm_feature_bits_t;

// Target supports 32-bit floating-point bytecode extension opcodes.
#define LOOM_IREEVM_FEATURE_EXT_F32 (UINT64_C(1) << 0)
// Target supports 64-bit floating-point bytecode extension opcodes.
#define LOOM_IREEVM_FEATURE_EXT_F64 (UINT64_C(1) << 1)

#endif  // LOOM_TARGET_ARCH_IREEVM_RECORDS_FEATURE_BITS_H_
