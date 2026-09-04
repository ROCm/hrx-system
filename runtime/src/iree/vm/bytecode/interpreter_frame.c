// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_frame.h"

#include <string.h>

void iree_vm_bytecode_frame_initialize(
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    iree_vm_bytecode_execution_state_t* state) {
  const uint32_t call_ref_scratch_count =
      iree_any_bit_set(function->flags_u16,
                       IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)
          ? IREE_VM_CALL_DIRECT_REGISTER_COUNT
          : 0;
  state->ref_count = function->ref_register_count_u16 +
                     function->local_ref_count_u32 + call_ref_scratch_count;
  if (state->ref_count != 0) {
    memset(state->refs, 0, state->ref_count * sizeof(*state->refs));
  }
  const uint32_t function_count = function->function_register_count_u16 +
                                  function->local_function_count_u32;
  if (function_count != 0) {
    memset(state->functions, 0, function_count * sizeof(*state->functions));
  }

  const uint16_t direct_value_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature->argument_value_count_u16);
  iree_vm_bytecode_frame_copy_direct_values(
      state->values, state->call.value_arguments.direct, direct_value_count);
  const uint16_t direct_ref_count = iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                                             signature->argument_ref_count_u16);
  for (uint16_t i = 0; i < direct_ref_count; ++i) {
    iree_vm_call_ref_argument_load_move(&state->call, i, &state->refs[i]);
  }
  const uint16_t direct_function_count =
      iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
               signature->argument_function_count_u16);
  if (direct_function_count != 0) {
    memcpy(state->functions, state->call.function_arguments.direct,
           direct_function_count * sizeof(*state->functions));
  }
}

// Appends one naturally aligned array to an already valid frame layout.
static iree_host_size_t iree_vm_bytecode_frame_append(
    iree_host_size_t count, iree_host_size_t element_size,
    iree_host_size_t alignment, iree_host_size_t* inout_size) {
  const iree_host_size_t offset = iree_host_align(*inout_size, alignment);
  *inout_size = offset + count * element_size;
  return offset;
}

iree_vm_bytecode_frame_layout_t iree_vm_bytecode_calculate_frame_layout(
    const iree_vm_bytecode_v0_function_row_t* function) {
  iree_vm_bytecode_frame_layout_t layout = {0};
  iree_host_size_t storage_size = sizeof(iree_vm_bytecode_execution_state_t);
  layout.values_offset = iree_vm_bytecode_frame_append(
      function->value_register_count_u16, sizeof(uint64_t),
      iree_alignof(uint64_t), &storage_size);
  layout.refs_offset = iree_vm_bytecode_frame_append(
      function->ref_register_count_u16, sizeof(iree_vm_ref_t),
      iree_alignof(iree_vm_ref_t), &storage_size);
  layout.local_refs_offset = iree_vm_bytecode_frame_append(
      function->local_ref_count_u32, sizeof(iree_vm_ref_t),
      iree_alignof(iree_vm_ref_t), &storage_size);
  const iree_host_size_t call_ref_scratch_count =
      iree_any_bit_set(function->flags_u16,
                       IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)
          ? IREE_VM_CALL_DIRECT_REGISTER_COUNT
          : 0;
  layout.call_ref_scratch_offset = iree_vm_bytecode_frame_append(
      call_ref_scratch_count, sizeof(iree_vm_ref_t),
      iree_alignof(iree_vm_ref_t), &storage_size);
  layout.functions_offset = iree_vm_bytecode_frame_append(
      function->function_register_count_u16, sizeof(iree_vm_function_ref_t),
      iree_alignof(iree_vm_function_ref_t), &storage_size);
  layout.local_functions_offset = iree_vm_bytecode_frame_append(
      function->local_function_count_u32, sizeof(iree_vm_function_ref_t),
      iree_alignof(iree_vm_function_ref_t), &storage_size);
  layout.local_bytes_offset = iree_vm_bytecode_frame_append(
      function->local_byte_length_u16, sizeof(uint8_t), iree_alignof(uint8_t),
      &storage_size);
  layout.frame.storage_size = storage_size;
  layout.frame.storage_alignment =
      iree_max(iree_alignof(iree_vm_bytecode_execution_state_t),
               iree_max(iree_alignof(uint64_t),
                        iree_max(iree_alignof(iree_vm_ref_t),
                                 iree_alignof(iree_vm_function_ref_t))));
  return layout;
}

void iree_vm_bytecode_frame_cleanup(iree_vm_frame_t* frame) {
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)iree_vm_frame_storage(frame);
  for (uint32_t i = 0; i < state->ref_count; ++i) {
    iree_vm_ref_reset(&state->refs[i]);
  }
}
