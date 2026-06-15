// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IR-derived move topology queries for allocation and move materialization.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MOVE_TOPOLOGY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MOVE_TOPOLOGY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Counts low.copy operations in |body|.
iree_host_size_t loom_low_allocation_move_topology_count_copy_ops(
    const loom_region_t* body);

// Returns true when a low.concat must materialize its result as packet-local
// storage in |module|. Branch-edge copies can decompose low.concat payloads
// directly into block arguments, so branch-only concats do not require
// packet-local moves.
bool loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
    const loom_module_t* module, const loom_op_t* op);

// Table-backed wrapper for
// loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module.
bool loom_low_allocation_move_topology_concat_requires_packet_materialization(
    const loom_low_allocation_table_t* table, const loom_op_t* op);

// Returns true when |op| may require packet-local moves after allocation.
bool loom_low_allocation_move_topology_op_has_packet_moves(const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_TOPOLOGY_H_
