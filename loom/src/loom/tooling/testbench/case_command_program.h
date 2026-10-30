// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Projection of one concrete check.case sample into a command program.

#ifndef LOOM_TOOLING_TESTBENCH_CASE_COMMAND_PROGRAM_H_
#define LOOM_TOOLING_TESTBENCH_CASE_COMMAND_PROGRAM_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_case_command_program_t {
  // Generated private command.def symbol.
  loom_symbol_ref_t root_ref;
  // Case value IDs providing storage for command bindings in signature order.
  // Tensor aliases are represented by their ultimate authored storage root.
  // Storage is owned by the mutated module and remains valid until module free.
  const loom_value_id_t* binding_value_ids;
  // Number of entries in |binding_value_ids|.
  iree_host_size_t binding_count;
} loom_testbench_case_command_program_t;

// Appends a private command program representing every kernel launch in one
// concrete check.case sample.
//
// |module_plan| and |case_plan| must have been planned from |module| before
// this call. Sampled parameters and scalar literals required by launch operands
// or dependent types become constants. Shaped case values consume the
// planner's canonical storage identities to become buffer bindings. Launches
// consume the plan's balanced schedule events, preserving the exact authored
// serial and concurrent structure.
//
// Scalar values produced by executable generators or invocations are not
// available to host-only planning. A launch that consumes one fails at that
// operand instead of compiling with an invented value.
iree_status_t loom_testbench_case_command_program_build(
    loom_module_t* module, const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_sample_t sample,
    loom_testbench_case_command_program_t* out_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_CASE_COMMAND_PROGRAM_H_
