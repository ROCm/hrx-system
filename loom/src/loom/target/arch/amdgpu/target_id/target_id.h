// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-record identity and AMDHSA target-id formatting.

#ifndef LOOM_TARGET_ARCH_AMDGPU_TARGET_ID_TARGET_ID_H_
#define LOOM_TARGET_ARCH_AMDGPU_TARGET_ID_TARGET_ID_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/target_info.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_resolved_target_t loom_low_resolved_target_t;

// AMDHSA target-id prefix shared by native code-object emission.
extern const iree_string_view_t loom_amdgpu_amdhsa_target_id_prefix;

// Returns the AMDGPU processor selected by |target_op|, or NULL when the op is
// not an AMDGPU target record or verification has not proven the processor.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_processor_from_op(
    const loom_module_t* module, const loom_op_t* target_op);

// Returns the AMDGPU processor selected by |target_ref|, or NULL.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_processor_from_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref);

// Returns the AMDGPU processor selected by a resolved low target, or NULL.
const loom_amdgpu_processor_info_t*
loom_amdgpu_target_processor_from_resolved_target(
    const loom_module_t* module, const loom_low_resolved_target_t* target);

// Appends a full AMDHSA target id for |processor| and optional
// |feature_suffix| to |builder|.
iree_status_t loom_amdgpu_amdhsa_target_id_append(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t feature_suffix, iree_string_builder_t* builder);

// Formats a full AMDHSA target id into |arena|.
iree_status_t loom_amdgpu_amdhsa_target_id_format(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t feature_suffix, iree_arena_allocator_t* arena,
    iree_string_view_t* out_target_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_TARGET_ID_TARGET_ID_H_
