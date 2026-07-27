// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private x86 source-to-target-low lowering helpers.
//
// This header is target-local glue between lower.c and focused x86 leaf
// lowerers. Keep declarations narrow: a new declaration should mean two x86
// lowering invariant clusters genuinely share one contract.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_INTERNAL_H_
#define LOOM_TARGET_ARCH_X86_LOWER_INTERNAL_H_

#include "loom/target/arch/x86/lower/lower.h"
#include "loom/target/arch/x86/register_classes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds a one-unit x86 register type in the current lowering context.
iree_status_t loom_x86_make_register_type(
    loom_low_lower_context_t* context, loom_x86_register_class_t register_class,
    loom_type_t* out_type);

// Maps a packed-dot source value type to the selected x86 register class.
iree_status_t loom_x86_map_packed_dot_type(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type);

// Returns true when |type| is a static vector shape that can live in an x86
// packed-dot vector register, and writes the total register bit width.
bool loom_x86_packed_dot_type_static_vector_bit_width(loom_type_t type,
                                                      uint32_t* out_bit_width);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_INTERNAL_H_
