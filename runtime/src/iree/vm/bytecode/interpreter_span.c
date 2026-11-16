// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_span.h"

#include "iree/vm/bytecode/interpreter_buffer.h"

iree_status_t iree_vm_bytecode_span_resolve_buffer(
    const iree_vm_bytecode_execution_state_t* state, uint32_t index,
    uint64_t offset, uint64_t byte_length, bool references,
    iree_vm_span_access_t access, void** out_data) {
  if (index >= state->span_buffer_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer span is outside the pending call");
  }
  if (references) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer bytes cannot contain reference cells");
  }
  iree_vm_buffer_access_flags_t buffer_access =
      (access & IREE_VM_SPAN_ACCESS_READ ? IREE_VM_BUFFER_ACCESS_FLAG_READ
                                         : 0) |
      (access & IREE_VM_SPAN_ACCESS_WRITE ? IREE_VM_BUFFER_ACCESS_FLAG_WRITE
                                          : 0);
  iree_byte_span_t span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      state->span_buffers[index].object, buffer_access, offset, byte_length,
      &span));
  *out_data = span.data;
  return iree_ok_status();
}
