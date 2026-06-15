// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/low/kernel.h"

#include "loom/ir/attribute.h"
#include "loom/ops/low/ops.h"

static bool loom_low_optional_attr_is_present(const loom_op_t* op,
                                              uint16_t attr_index) {
  return !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

bool loom_low_kernel_def_static_workgroup_size(
    const loom_op_t* op, loom_target_workgroup_size_t* out_size) {
  *out_size = (loom_target_workgroup_size_t){0};
  if (!loom_low_kernel_def_isa(op) ||
      !loom_low_optional_attr_is_present(
          op, loom_low_kernel_def_workgroup_size_x_ATTR_INDEX)) {
    return false;
  }
  *out_size = (loom_target_workgroup_size_t){
      .x = (uint32_t)loom_low_kernel_def_workgroup_size_x(op),
      .y = (uint32_t)loom_low_kernel_def_workgroup_size_y(op),
      .z = (uint32_t)loom_low_kernel_def_workgroup_size_z(op),
  };
  return true;
}
