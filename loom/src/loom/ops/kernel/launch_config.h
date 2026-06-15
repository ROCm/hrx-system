// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_KERNEL_LAUNCH_CONFIG_H_
#define LOOM_OPS_KERNEL_LAUNCH_CONFIG_H_

#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the launch configuration terminator in |kernel_op|'s config region,
// or NULL if the op is not a source kernel definition or the config region is
// structurally incomplete.
const loom_op_t* loom_kernel_def_launch_config_op(const loom_op_t* kernel_op);

// Returns the workgroup count operand for |dimension| from a launch config
// terminator, or LOOM_VALUE_ID_INVALID when |launch_config| is not a launch
// config terminator or |dimension| is invalid.
loom_value_id_t loom_kernel_launch_config_workgroup_count_operand(
    const loom_op_t* launch_config, loom_kernel_dimension_t dimension);

// Returns the workgroup size operand for |dimension| from a launch config
// terminator, or LOOM_VALUE_ID_INVALID when |launch_config| is not a launch
// config terminator or |dimension| is invalid.
loom_value_id_t loom_kernel_launch_config_workgroup_size_operand(
    const loom_op_t* launch_config, loom_kernel_dimension_t dimension);

// Resolves a fully static required workgroup size from |kernel_op|'s launch
// config terminator. Dynamic dimensions are represented by returning false;
// callers that need a static target contract must emit their own diagnostic at
// the phase boundary that requires it.
bool loom_kernel_def_static_workgroup_size(
    const loom_module_t* module, const loom_op_t* kernel_op,
    loom_target_workgroup_size_t* out_size);

// Resolves a fully static required workgroup size using exact facts for the
// launch config operands. Falls back to direct constant inspection when facts
// are unavailable.
bool loom_kernel_def_static_workgroup_size_from_facts(
    const loom_module_t* module, const loom_op_t* kernel_op,
    const loom_value_fact_table_t* facts,
    loom_target_workgroup_size_t* out_size);

// Resolves a fully static dispatch workgroup count from |kernel_op|'s launch
// config terminator. Dynamic dimensions are represented by returning false.
bool loom_kernel_def_static_workgroup_count(
    const loom_module_t* module, const loom_op_t* kernel_op,
    loom_target_dispatch_workgroup_count_t* out_count);

// Resolves a fully static dispatch workgroup count using exact facts for the
// launch config operands. Falls back to direct constant inspection when facts
// are unavailable.
bool loom_kernel_def_static_workgroup_count_from_facts(
    const loom_module_t* module, const loom_op_t* kernel_op,
    const loom_value_fact_table_t* facts,
    loom_target_dispatch_workgroup_count_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_KERNEL_LAUNCH_CONFIG_H_
