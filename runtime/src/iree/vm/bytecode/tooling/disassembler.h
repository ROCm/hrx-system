// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_TOOLING_DISASSEMBLER_H_
#define IREE_VM_BYTECODE_TOOLING_DISASSEMBLER_H_

#include "iree/base/api.h"
#include "iree/vm/bytecode/tooling/dump.h"

// Decodes and emits one structurally bounded function record stream.
iree_status_t iree_vm_bytecode_disassemble_function(
    uint32_t function_ordinal, iree_const_byte_span_t bytecode,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder);

#endif  // IREE_VM_BYTECODE_TOOLING_DISASSEMBLER_H_
