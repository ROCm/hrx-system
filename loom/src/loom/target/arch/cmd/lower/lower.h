// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Closed source command-program conversion to the portable cmd low ISA.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_LOWER_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/target/arch/cmd/lower/launch_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Portable flattened device-ABI argument kind.
typedef enum loom_cmd_lower_dispatch_argument_kind_e {
  // Source buffer or view represented by a resolved buffer range.
  LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_BUFFER = 0,
  // Exact scalar payload occupying 8 bits.
  LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B8 = 1,
  // Exact scalar payload occupying 16 bits.
  LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B16 = 2,
  // Exact scalar payload occupying 32 bits.
  LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B32 = 3,
  // Exact scalar payload occupying 64 bits.
  LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B64 = 4,
} loom_cmd_lower_dispatch_argument_kind_t;

// One classified device-ABI argument consumed by closed command lowering.
typedef struct loom_cmd_lower_dispatch_argument_t {
  // Portable argument representation.
  loom_cmd_lower_dispatch_argument_kind_t kind;
  // Source value used to reuse its prepared low representation.
  loom_value_id_t source_value;
  // Exact tagless scalar bit pattern; ignored for buffer arguments.
  uint64_t scalar_bits;
} loom_cmd_lower_dispatch_argument_t;

// Resolved executable placement for one aggregate dispatch row.
typedef struct loom_cmd_lower_dispatch_t {
  // Dense executable-table index selected for the dispatch.
  uint32_t executable_index;
  // Dense program entry-table index selecting an executable-local token.
  uint32_t entry_index;
  // Classified device-ABI arguments in entry order.
  const loom_cmd_lower_dispatch_argument_t* arguments;
  // Number of entries in |arguments|.
  uint16_t argument_count;
  // Number of flattened low operand values produced by |arguments|.
  uint32_t operand_value_count;
} loom_cmd_lower_dispatch_t;

// Issue-time binding placement of the aggregate host launch-count table.
typedef struct loom_cmd_lower_launch_count_binding_t {
  // Dense index in the rebindable binding table.
  uint32_t resource_index;
  // Root-relative byte offset of the first xyz tuple.
  uint64_t byte_offset;
} loom_cmd_lower_launch_count_binding_t;

// Compiler-owned facts consumed by closed command-program conversion.
//
// Rows are already resolved by source specialization, dependency-unit
// extraction, binding placement, aggregate launch analysis, and wave planning.
// Conversion preserves these facts; it does not rediscover kernel identity,
// launch arithmetic, or command ordering from the source module.
typedef struct loom_cmd_lower_plan_t {
  // Source launch-binding rows in command-program signature order.
  const loom_cmd_buffer_binding_t* bindings;
  // Number of source launch-binding rows, excluding specialization arguments.
  iree_host_size_t binding_count;
  // Derived source values mapped to root-relative buffer ranges.
  const loom_cmd_buffer_range_t* buffer_ranges;
  // Number of entries in |buffer_ranges|.
  iree_host_size_t buffer_range_count;
  // External resource-table shape emitted on the lowered function.
  loom_cmd_abi_layout_t abi_layout;
  // Aggregate launch graph defining direct and host count placement.
  const loom_cmd_launch_graph_t* launch_graph;
  // Host launch-count table placement, ignored when the graph has no tuples.
  loom_cmd_lower_launch_count_binding_t launch_count_binding;
  // Executable placements in aggregate launch-graph traversal order.
  const loom_cmd_lower_dispatch_t* dispatches;
} loom_cmd_lower_plan_t;

// Replaces one specialized command.program.def with a targetless,
// zero-signature command_program low.func.def using the cmd.core
// representation contract.
//
// The portable issue-time ABI accepts buffer roots and explicitly resolved
// view ranges plus workgroup counts classified by |plan->launch_graph|. Exact
// tuples become direct dispatches. Host tuples become static-indirect
// dispatches referencing one rebindable output table. Buffer and view arguments
// become resolved ranges; exact scalar arguments preserve their tagless ABI
// bits.
// Unsupported kernel-argument forms fail without changing the source program.
// On success the replacement keeps the source symbol identity and is returned
// in |out_low_function|.
iree_status_t loom_cmd_lower_program_to_low(loom_module_t* module,
                                            loom_op_t* program_op,
                                            const loom_cmd_lower_plan_t* plan,
                                            loom_op_t** out_low_function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_LOWER_H_
