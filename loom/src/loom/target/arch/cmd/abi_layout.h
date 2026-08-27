// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-program ABI placement and Low layout records.

#ifndef LOOM_TARGET_ARCH_CMD_ABI_LAYOUT_H_
#define LOOM_TARGET_ARCH_CMD_ABI_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materialization role of one command-program buffer root.
typedef enum loom_cmd_buffer_role_e {
  // The buffer range is fixed while the command program remains materialized.
  LOOM_CMD_BUFFER_ROLE_FIXED = 1,
  // The buffer range is supplied through the issue-time binding table.
  LOOM_CMD_BUFFER_ROLE_REBINDABLE = 2,
} loom_cmd_buffer_role_t;

// ABI placement of one source command-program buffer argument.
typedef struct loom_cmd_buffer_binding_t {
  // Materialization role selecting the fixed or rebindable resource table.
  loom_cmd_buffer_role_t role;
  // Dense index in the table selected by |role|.
  uint32_t resource_index;
  // Root-relative byte offset of the range passed to kernel launches.
  uint64_t byte_offset;
  // Byte length of the range, or UINT64_MAX for the remaining root buffer.
  uint64_t byte_length;
} loom_cmd_buffer_binding_t;

// ABI placement of one source view or derived buffer value.
typedef struct loom_cmd_buffer_range_t {
  // Source SSA value mapped to the resulting portable buffer reference.
  loom_value_id_t source_value;
  // Materialization role of the resolved storage root.
  loom_cmd_buffer_role_t role;
  // Dense index in the table selected by |role|.
  uint32_t resource_index;
  // Byte offset relative to the resolved storage root.
  uint64_t byte_offset;
  // Exact byte length of the derived range.
  uint64_t byte_length;
} loom_cmd_buffer_range_t;

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_ABI_LAYOUT_H_
