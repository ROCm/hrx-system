// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lazy function-contract projection over a provider-backed module index.

#ifndef LOOM_LINK_FUNC_CONTRACT_PROJECTION_H_
#define LOOM_LINK_FUNC_CONTRACT_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/link/func_contract.h"
#include "loom/link/module_index.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_link_func_contract_projection_t
    loom_link_func_contract_projection_t;

// Allocates an empty lazy projection over |index|.
//
// Every projected contract is translated into one private identity module.
// Global symbols with the same link name share an identity while provider-local
// private symbols remain distinct. Materialized providers copy only function
// headers and bytecode providers decode only their indexed header payloads;
// function bodies are never visited.
//
// The projection borrows |index| and |block_pool| for its lifetime. It is an
// invocation-local object and requires external synchronization.
iree_status_t loom_link_func_contract_projection_allocate(
    const loom_link_module_index_t* index, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator,
    loom_link_func_contract_projection_t** out_projection);

// Frees |projection| and every projected contract.
void loom_link_func_contract_projection_free(
    loom_link_func_contract_projection_t* projection);

// Returns the shared identity module owning every projected contract.
//
// The returned module is borrowed for the projection lifetime. Callers may
// initialize one same-module remap after loading their complete candidate set
// and reuse it across all contract comparisons.
loom_module_t* loom_link_func_contract_projection_module(
    const loom_link_func_contract_projection_t* projection);

// Loads one indexed function-like symbol contract without reading its body.
//
// |symbol| must belong to the projection index and implement the function-like
// symbol interface. Repeated loads return the same borrowed contract view.
iree_status_t loom_link_func_contract_projection_load(
    loom_link_func_contract_projection_t* projection,
    const loom_link_module_index_symbol_t* symbol,
    const loom_link_func_contract_t** out_contract);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_FUNC_CONTRACT_PROJECTION_H_
