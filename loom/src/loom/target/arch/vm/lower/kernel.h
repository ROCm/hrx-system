// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM scalar kernel-profile lowering.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_KERNEL_H_
#define LOOM_TARGET_ARCH_VM_LOWER_KERNEL_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a scalar kernel-profile plan for |source_op|.
//
// Returns true when the operation is owned by the kernel profile and writes a
// nonempty callback plan to |out_plan|.
bool loom_vm_kernel_try_select_op(const loom_op_t* source_op,
                                  loom_low_lower_plan_t* out_plan);

// Scalar kernel-profile source legality owned by the VM target.
extern const loom_target_low_legality_provider_t
    loom_vm_kernel_low_legality_provider;

// Builds the complete VM kernel ABI layout from authored kernel arguments.
//
// The runtime signature appends the target-owned launch arguments while the
// authored signature remains available for public presentation. When present,
// |argument_values| supply authored field names and the target supplies stable
// names for its appended fields.
iree_status_t loom_vm_kernel_map_abi_layout(
    loom_low_lower_context_t* context, const loom_type_t* argument_types,
    const loom_value_id_t* argument_values, iree_host_size_t argument_count,
    loom_named_attr_slice_t* out_abi_layout);

// Appends the canonical VM kernel invocation arguments and binds selected
// topology queries to those arguments, derived values, or profile constants.
iree_status_t loom_vm_kernel_emit_preamble(loom_low_lower_context_t* context);

// Emits or erases one selected scalar kernel-profile operation.
//
// |out_handled| is true when |plan| belongs to the kernel profile.
iree_status_t loom_vm_kernel_emit_op(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_KERNEL_H_
