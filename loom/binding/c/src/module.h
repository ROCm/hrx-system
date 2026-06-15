// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_MODULE_STORAGE_H_
#define LOOMC_MODULE_STORAGE_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/workspace.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an empty public module handle with workspace-backed storage.
LOOMC_API_PRIVATE loomc_status_t loomc_module_create_empty(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_allocator_t allocator, loomc_module_t** out_module);

// Returns the allocator owned by the public module handle.
LOOMC_API_PRIVATE loomc_allocator_t
loomc_module_allocator(const loomc_module_t* module);

// Returns the context retained by the public module handle.
LOOMC_API_PRIVATE loomc_context_t* loomc_module_context(
    const loomc_module_t* module);

// Returns the arena block pool backing the internal module.
LOOMC_API_PRIVATE iree_arena_block_pool_t* loomc_module_block_pool(
    loomc_module_t* module);

// Transfers one internal module into an empty public module handle.
LOOMC_API_PRIVATE loomc_status_t loomc_module_set_loom_module(
    loomc_module_t* module, loom_module_t* internal_module);

// Returns the internal module owned by a public module handle.
LOOMC_API_PRIVATE loom_module_t* loomc_module_loom_module(
    loomc_module_t* module);

// Returns the internal module owned by a public module handle.
LOOMC_API_PRIVATE const loom_module_t* loomc_module_const_loom_module(
    const loomc_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_MODULE_STORAGE_H_
