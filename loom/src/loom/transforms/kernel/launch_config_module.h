// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared kernel launch-configuration module extraction.

#ifndef LOOM_TRANSFORMS_KERNEL_LAUNCH_CONFIG_MODULE_H_
#define LOOM_TRANSFORMS_KERNEL_LAUNCH_CONFIG_MODULE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materializes the launch configurations of all source kernel definitions as
// ordinary pure functions in a new module.
//
// Each output function is public, retains its kernel symbol name, accepts
// exactly the kernel workload arguments, and returns the xyz workgroup count
// as three index values. Configuration and target specialization must already
// have selected all symbolic launch dependencies. The normal value-fact engine
// reifies their exact results while extracting the function, leaving workload
// values as the only dynamic inputs. Any unresolved cross-module symbol
// dependency fails materialization instead of being copied into the artifact.
//
// |source_module| is not modified. The returned module shares its Loom context
// with |source_module|, and |block_pool| must outlive it. The caller owns
// |out_module| on success and releases it with loom_module_free.
iree_status_t loom_kernel_launch_config_module_materialize(
    const loom_module_t* source_module, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_LAUNCH_CONFIG_MODULE_H_
