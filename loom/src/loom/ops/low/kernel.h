// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete low.kernel.def helpers.

#ifndef LOOM_OPS_LOW_KERNEL_H_
#define LOOM_OPS_LOW_KERNEL_H_

#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the fixed workgroup size declared by |op|, if |op| is a
// low.kernel.def with concrete workgroup_size attrs.
bool loom_low_kernel_def_static_workgroup_size(
    const loom_op_t* op, loom_target_workgroup_size_t* out_size);

// Returns the fixed dispatch workgroup count declared by |op|, if |op| is a
// low.kernel.def with concrete workgroup_count attrs.
bool loom_low_kernel_def_static_workgroup_count(
    const loom_op_t* op, loom_target_dispatch_workgroup_count_t* out_count);

// Returns the fixed workgroup-cluster size declared by |op|, if |op| is a
// low.kernel.def with concrete cluster_size attrs.
bool loom_low_kernel_def_static_workgroup_cluster_size(
    const loom_op_t* op, loom_target_workgroup_cluster_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_LOW_KERNEL_H_
