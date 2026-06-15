// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-record helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_
#define LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/target_info.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the AMDGPU processor name selected by |target_op|, or empty.
iree_string_view_t loom_amdgpu_target_record_processor_name(
    const loom_module_t* module, const loom_op_t* target_op);

// Returns the AMDGPU processor row selected by |target_op|, or NULL.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_record_processor(
    const loom_module_t* module, const loom_op_t* target_op);

// Builds a compact AMDGPU target record for |processor|.
//
// The target kind is derived from the processor descriptor-set family. When
// |processor| is not the family default, the target record stores an explicit
// processor override so later verification still checks the family invariant.
iree_status_t loom_amdgpu_target_record_build_for_processor(
    loom_builder_t* builder, const loom_amdgpu_processor_info_t* processor,
    loom_symbol_ref_t symbol, loom_location_id_t location,
    loom_op_t** out_target_op);

// Sets the target-record processor override attr to |processor|.
iree_status_t loom_amdgpu_target_record_set_processor(
    loom_module_t* module, loom_op_t* target_op,
    const loom_amdgpu_processor_info_t* processor);

iree_status_t loom_amdgpu_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_
