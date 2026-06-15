// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/pipeline/pipeline.h"

iree_status_t loom_low_pipeline_build_packetization_preparation(
    loom_builder_t* builder) {
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_run(
      builder, IREE_SV("cse"), loom_named_attr_slice_empty(), &run_op));
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, IREE_SV("low-select-operand-forms"),
                             loom_named_attr_slice_empty(), &run_op));
  return loom_pass_ir_build_run(builder, IREE_SV("low-dce"),
                                loom_named_attr_slice_empty(), &run_op);
}
