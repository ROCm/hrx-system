// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Optional presentation data for VM bytecode dump and disassembly tools.

#ifndef IREE_VM_BYTECODE_TOOLING_H_
#define IREE_VM_BYTECODE_TOOLING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the canonical mnemonic for |opcode| or an empty view when unknown.
IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_instruction_name(uint8_t opcode);

// Returns the canonical module-record name or an empty view when unknown.
IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_module_record_name(uint8_t record_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_TOOLING_H_
