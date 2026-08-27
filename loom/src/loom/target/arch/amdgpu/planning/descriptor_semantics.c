// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"

static const loom_low_schedule_structural_state_read_t
    kAmdgpuStructuralStateReads[] = {
        {
            .result_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,
            .state_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_EXEC,
        },
};

loom_low_schedule_structural_state_read_list_t
loom_amdgpu_descriptor_structural_state_reads(void) {
  return (loom_low_schedule_structural_state_read_list_t){
      .values = kAmdgpuStructuralStateReads,
      .count = IREE_ARRAYSIZE(kAmdgpuStructuralStateReads),
  };
}
