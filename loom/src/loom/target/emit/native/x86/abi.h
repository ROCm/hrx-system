// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 scalar platform-function boundary classification.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_ABI_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_ABI_H_

#include "loom/codegen/low/allocation.h"
#include "loom/target/entry_selection.h"

#ifdef __cplusplus
extern "C" {
#endif

// SysV scalar entry assignments. The register boundary is independent of the
// object format or host OS. Native frame preparation consumes this contract
// before scheduling/allocation; neither the encoder nor the ELF adapter
// selects a calling convention.
typedef struct loom_x86_function_abi_t {
  // ABI registers for live input values, in original parameter order.
  loom_low_allocation_fixed_value_t fixed_values[6];
  // Number of live input assignments; unused parameters still consume ABI
  // slots.
  iree_host_size_t fixed_value_count;
} loom_x86_function_abi_t;

// Admits the logical and physical callable signature and materializes the
// platform's fixed input constraints. Unsupported user boundaries produce a
// diagnostic and leave out_accepted false. Compiler storage is borrowed.
iree_status_t loom_x86_function_abi_prepare(loom_module_t* module,
                                            const loom_target_entry_t* entry,
                                            iree_diagnostic_emitter_t emitter,
                                            bool* out_accepted,
                                            loom_x86_function_abi_t* out_abi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_ABI_H_
