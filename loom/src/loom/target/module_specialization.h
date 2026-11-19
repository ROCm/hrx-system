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
// Inputs obey loom_target_specialize_functions' resolved-request contract.
// Allocation and projection failures return a status.
iree_status_t loom_target_specialize_module(
    const loom_target_environment_t* environment,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count);

// Specializes and binds an exclusively owned, unpublished module. A failed
// working module is discarded by its owner. Uses caller scratch for target
// facts and records the durable target definitions directly in |module|.
iree_status_t loom_target_specialize_module_in_place(
    const loom_target_environment_t* environment, loom_module_t* module,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    uint32_t* out_error_count);

// Specializes every declaration-targeted function and every kernel entry in
// |module| to |target_profile|.
//
// This is the homogeneous deployment-product form of
// loom_target_specialize_module_in_place. A non-kernel function opts into the
// invocation profile by carrying an authored target reference that still
// resolves to target.decl; targetless functions remain neutral and concrete
// target records retain their authored context. Request rows are derived from
// verified compiler-owned IR and remain private to the call.
iree_status_t loom_target_specialize_module_deployment_entries(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, loom_module_t* module,
    uint32_t* out_error_count);

// Specializes the deployment entries needed to normalize a command program.
//
// Declaration-targeted functions, targetless kernel entries, and concrete
// kernel entries compatible with |target_profile| are specialized exactly as
// in loom_target_specialize_module_deployment_entries. A concrete kernel entry
// incompatible with the profile remains in its authored target context because
// command specialization may remove every launch reaching it. Callers must
// pair this with loom_target_validate_module_deployment_kernel_entry_refs after
// command normalization and before publishing kernel requests.
iree_status_t loom_target_specialize_module_command_projection_entries(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, loom_module_t* module,
    uint32_t* out_error_count);

// Validates live command-projection kernel entries against |target_profile|.
//
// |kernel_refs| names distinct live kernel entries retained after command
// normalization. Concrete target incompatibilities deferred by
// loom_target_specialize_module_command_projection_entries emit the canonical
// target-specialization diagnostic and increment |out_error_count|. Compatible
// entries require no second specialization. Allocation and diagnostic sink
// failures return a status.
iree_status_t loom_target_validate_module_deployment_kernel_entry_refs(
    const loom_target_environment_t* environment, loom_module_t* module,
    const loom_symbol_ref_t* kernel_refs, iree_host_size_t kernel_count,
    const loom_target_profile_t* target_profile,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, uint32_t* out_error_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_MODULE_SPECIALIZATION_H_
