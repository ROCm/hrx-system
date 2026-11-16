// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_SPAN_H_
#define IREE_VM_BYTECODE_INTERPRETER_SPAN_H_

#include "iree/vm/bytecode/interpreter_frame.h"

// Resolves a managed buffer pin using the normal buffer mapping contract.
// The caller has identified the buffer region domain; its index and byte range
// remain guest-controlled and are validated here.
iree_status_t iree_vm_bytecode_span_resolve_buffer(
    const iree_vm_bytecode_execution_state_t* state, uint32_t index,
    uint64_t offset, uint64_t byte_length, bool references,
    iree_vm_span_access_t access, void** out_data);

// Resolves a guest coordinate against incoming storage or a pending buffer pin.
// Reference cells and scalar bytes occupy separate domains. Empty ranges
// preserve a null base without forming a pointer. The returned descriptor is
// frame-owned for incoming storage and null for a managed buffer. Keeping the
// incoming path inline specializes domain and access checks at each instruction
// without spilling interpreter state across a native call.
static IREE_ATTRIBUTE_ALWAYS_INLINE iree_status_t iree_vm_bytecode_span_resolve(
    const iree_vm_bytecode_execution_state_t* state, uint64_t address,
    uint64_t byte_length, bool references, iree_vm_span_access_t access,
    void** out_data, const iree_vm_bytecode_span_region_t** out_region) {
  const uint32_t ordinal = address >> IREE_VM_BYTECODE_SPAN_REGION_SHIFT;
  const uint64_t offset = address & IREE_VM_BYTECODE_SPAN_OFFSET_MASK;
  if (ordinal >= IREE_VM_BYTECODE_SPAN_BUFFER_REGION_BASE) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_span_resolve_buffer(
        state, ordinal - IREE_VM_BYTECODE_SPAN_BUFFER_REGION_BASE, offset,
        byte_length, references, access, out_data));
    *out_region = NULL;
    return iree_ok_status();
  }
  if (ordinal == 0 || ordinal > state->span_region_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "span region is outside this frame");
  }
  const iree_vm_bytecode_span_region_t* region =
      &state->span_regions[ordinal - 1];
  if (!region->element_kind || iree_vm_module_signature_type_kind_is_ref(
                                   region->element_kind) != references) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "span storage domain mismatch");
  }
  if (access & ~region->access) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "span access exceeds the borrowed permissions");
  }
  if ((references && (offset % IREE_VM_SPAN_REF_STRIDE != 0 ||
                      byte_length % IREE_VM_SPAN_REF_STRIDE != 0)) ||
      offset > region->byte_length ||
      byte_length > region->byte_length - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "span access is unaligned or outside its storage");
  }
  const uint64_t native_offset =
      references ? offset / IREE_VM_SPAN_REF_STRIDE * sizeof(iree_vm_ref_t)
                 : offset;
  *out_data =
      native_offset ? (uint8_t*)region->data + native_offset : region->data;
  *out_region = region;
  return iree_ok_status();
}

#endif  // IREE_VM_BYTECODE_INTERPRETER_SPAN_H_
