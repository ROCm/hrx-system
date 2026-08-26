// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_VERIFICATION_ISA_BUFFER_H_
#define IREE_VM_BYTECODE_VERIFICATION_ISA_BUFFER_H_

#include "iree/vm/bytecode/module_reader.h"

// Verifies one complete executable Core buffer instruction record.
iree_status_t iree_vm_bytecode_verify_buffer_record(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function,
    iree_const_byte_span_t record);

#endif  // IREE_VM_BYTECODE_VERIFICATION_ISA_BUFFER_H_
