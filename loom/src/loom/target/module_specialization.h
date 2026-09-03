// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standalone IR projection of invocation target specialization.

#ifndef LOOM_TARGET_MODULE_SPECIALIZATION_H_
#define LOOM_TARGET_MODULE_SPECIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/specialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Specializes functions in |*inout_module| and replaces it with standalone IR
// carrying the resulting exact target definitions and function target refs.
//
// The source module remains owned by the caller when specialization or
// projection fails. On success the source module is freed and
// |*inout_module| owns the projected replacement. No replacement is made when
// no function versions are produced. Source incompatibilities are emitted
// through |diagnostic_emitter|, reported in |out_error_count|, and return OK.
// Infrastructure and malformed-request failures return a status.
iree_status_t loom_target_specialize_module(
    const loom_target_environment_t* environment,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count);

// Specializes every kernel entry in |*inout_module| to |target_profile|.
//
// This is the homogeneous deployment-product form of
// loom_target_specialize_module. Request rows are derived from the verified
// module and remain private to the call. Modules without kernel entries are
// left unchanged. |product_contract| may be NULL to materialize target-only IR
// without selecting a compiler representation, artifact format, ABI, or
// linkage.
iree_status_t loom_target_specialize_module_kernel_entries(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile,
    const loom_target_product_contract_t* product_contract,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_MODULE_SPECIALIZATION_H_
