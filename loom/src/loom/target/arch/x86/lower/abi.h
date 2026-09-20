// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 source-to-Low ABI layout production.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_ABI_H_
#define LOOM_TARGET_ARCH_X86_LOWER_ABI_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Classifies supported object-function signatures into a retained x86-64
// SysV layout. Unsupported signatures remain without a complete ABI layout and
// continue to support target-low fragment workflows.
iree_status_t loom_x86_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind,
    const loom_type_t* argument_types, iree_host_size_t argument_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_named_attr_slice_t* out_abi_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_ABI_H_
