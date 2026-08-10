// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-program Low ABI layout snapshots.

#ifndef LOOM_TARGET_ARCH_CMD_ABI_LAYOUT_H_
#define LOOM_TARGET_ARCH_CMD_ABI_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable external table shape of one command-program Low function.
//
// The layout is established before Low cleanup. Function-local
// low.resource<command_input> operations are use sites within these tables and
// may be removed when unused without changing the declared ABI. Consumers use
// these counts to allocate and validate the external tables even when no
// surviving operation references a table entry.
typedef struct loom_cmd_abi_layout_t {
  // Number of fixed buffer roots supplied during materialization.
  uint32_t fixed_buffer_count;

  // Number of rebindable buffer roots supplied during issue.
  uint32_t rebindable_binding_count;

  // Number of loaded executables supplied during materialization.
  uint32_t executable_count;

  // Number of program-local executable entry tokens.
  uint32_t entry_count;
} loom_cmd_abi_layout_t;

// Builds the canonical low.func.def abi_layout attribute for |layout|.
iree_status_t loom_cmd_abi_layout_make_attr(loom_module_t* module,
                                            const loom_cmd_abi_layout_t* layout,
                                            loom_attribute_t* out_attr);

// Loads the command ABI layout declared by |function_op|.
//
// The function must be a command_program low.func.def with a complete command
// ABI layout. User-authored missing, malformed, or out-of-range fields fail.
iree_status_t loom_cmd_abi_layout_from_low(const loom_module_t* module,
                                           const loom_op_t* function_op,
                                           loom_cmd_abi_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_ABI_LAYOUT_H_
