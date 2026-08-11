// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IR-derived move topology queries for allocation and move materialization.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MOVE_TOPOLOGY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MOVE_TOPOLOGY_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_packet_move_op_kind_e {
  // Operation does not introduce packet-local structural moves.
  LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE = 0,
  // Operation is low.copy.
  LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_COPY = 1,
  // Operation is low.move.
  LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_MOVE = 2,
  // Operation is low.slice.
  LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_SLICE = 3,
  // Operation is low.concat.
  LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT = 4,
} loom_low_allocation_packet_move_op_kind_t;

// Returns true when a low.concat must materialize its result as packet-local
// storage in |module|. Branch-edge copies can decompose low.concat payloads
// directly into block arguments, so branch-only concats do not require
// packet-local moves.
bool loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
    const loom_module_t* module, const loom_op_t* op);

// Classifies low operations that may require packet-local structural moves.
loom_low_allocation_packet_move_op_kind_t
loom_low_allocation_move_topology_packet_move_op_kind(const loom_op_t* op);

// Returns true when |op| may require packet-local moves after allocation.
bool loom_low_allocation_move_topology_op_has_packet_moves(const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_TOPOLOGY_H_
