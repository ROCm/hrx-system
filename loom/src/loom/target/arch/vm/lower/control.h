// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned VM structural control-flow lowering.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_CONTROL_H_
#define LOOM_TARGET_ARCH_VM_LOWER_CONTROL_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |source_op| is a structurally valid VM switch.
bool loom_vm_switch_lowering_can_emit(void* user_data,
                                      const loom_module_t* module,
                                      const loom_op_t* source_op,
                                      const loom_target_facts_t* target_facts);

// Emits the smallest accepted VM switch representation before allocation.
iree_status_t loom_vm_switch_lowering_emit(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t selector,
                                           loom_block_t* default_dest,
                                           loom_block_t* const* case_dests,
                                           uint16_t case_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_CONTROL_H_
