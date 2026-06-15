// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_WORKSPACE_STORAGE_H_
#define LOOMC_WORKSPACE_STORAGE_H_

#include "iree/base/internal/arena.h"
#include "loomc/workspace.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the arena block pool owned by workspace.
LOOMC_API_PRIVATE iree_arena_block_pool_t* loomc_workspace_block_pool(
    loomc_workspace_t* workspace);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_WORKSPACE_STORAGE_H_
