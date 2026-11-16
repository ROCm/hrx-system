// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_SPAN_H_
#define IREE_VM_BYTECODE_INTERPRETER_SPAN_H_

#include "iree/vm/bytecode/interpreter_frame.h"

// Resolves a guest coordinate against incoming storage or a pending buffer pin.
// Reference cells and scalar bytes occupy separate domains. Empty ranges
// preserve a null base without forming a pointer. The returned descriptor is
// frame-owned for incoming storage and null for a managed buffer.
iree_status_t iree_vm_bytecode_span_resolve(
    const iree_vm_bytecode_execution_state_t* state, uint64_t address,
    uint64_t byte_length, bool references, iree_vm_span_access_t access,
    void** out_data, const iree_vm_bytecode_span_region_t** out_region);

#endif  // IREE_VM_BYTECODE_INTERPRETER_SPAN_H_
