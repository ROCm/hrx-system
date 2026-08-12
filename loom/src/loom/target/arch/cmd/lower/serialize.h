// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Serialization of closed cmd.core low functions.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_SERIALIZE_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_SERIALIZE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serializes one prepared root into a portable command artifact.
//
// |root_index| selects a root in caller order from |plan|. The root's Low
// function, parameter placement, transient requirements, and ABI layout are
// consumed together so independently prepared state cannot be mixed.
//
// Preparation guarantees that the selected root is a closed zero-signature,
// single-block low.func.def using the cmd.core descriptor set and
// command_program ABI. Serialization trusts those plan-owned invariants. It is
// the closed portable-format boundary: a descriptor or Low operation without a
// portable command encoding is rejected. Pure scalar and buffer-reference SSA
// is evaluated once while serializing. The resulting artifact contains
// canonical flat buffer-reference, logical-argument, command, and
// parameter-requirement tables and retains no module, operation, value,
// symbol, or string storage.
//
// |block_pool| supplies invocation-local scratch storage and permits concurrent
// serialization of the same immutable plan from independent workspaces.
// |out_data| is empty on failure. The caller owns returned bytes and must free
// them with |host_allocator|. |out_program| is an infallibly bound trusted view
// borrowing those bytes; only externally loaded bytes require parsing.
iree_status_t loom_cmd_program_plan_serialize_root(
    const loom_cmd_program_plan_t* plan, iree_host_size_t root_index,
    iree_arena_block_pool_t* block_pool, iree_byte_span_t* out_data,
    loom_cmd_program_t* out_program, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_SERIALIZE_H_
