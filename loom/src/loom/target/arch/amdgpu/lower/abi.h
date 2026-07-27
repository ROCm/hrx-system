// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering ABI argument mapping.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ABI_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ABI_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps one source function argument to the low ABI representation selected for
// the active AMDGPU bundle.
iree_status_t loom_amdgpu_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument);

// Maps the authored source kernel signature to the AMDGPU HAL ABI layout
// snapshot attached to low.kernel.def.
iree_status_t loom_amdgpu_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_named_attr_slice_t* out_abi_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ABI_H_
