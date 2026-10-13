// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_DISASSEMBLER_NATIVE_H_
#define IREE_VM_BYTECODE_DISASSEMBLER_NATIVE_H_

#include "iree/vm/bytecode/disassembler.h"
#include "iree/vm/bytecode/wire/module.h"
#include "iree/vm/native_call.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints an admitted native record as grouped physical selectors. The schema
// and owning function establish all bounds and domain/direction information.
// Traversal uses bounded explicit scratch, not the C stack or expanded DAGs.
iree_status_t iree_vm_bytecode_disassembler_native_call(
    const iree_vm_native_call_schema_t* schema,
    const iree_vm_bytecode_v0_function_row_t* function, const uint32_t* words,
    iree_vm_bytecode_disassembler_write_callback_t output,
    iree_allocator_t scratch_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_VM_BYTECODE_DISASSEMBLER_NATIVE_H_
