// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_frame.h"

#include "iree/base/allocator.h"
#include "iree/vm/bytecode/module_reader.h"

void iree_vm_bytecode_frame_initialize(
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    iree_vm_bytecode_execution_state_t* state) {
  const iree_vm_call_packet_t* call = state->call;
  const uint32_t ref_count =
      function->ref_register_count_u16 + function->local_ref_count_u32;
  state->ref_count = ref_count;
  for (uint32_t i = 0; i < ref_count; ++i) {
    state->refs[i] = iree_vm_ref_null();
  }
  const uint32_t function_count = function->function_register_count_u16 +
                                  function->local_function_count_u32;
  if (function_count != 0) {
    memset(state->functions, 0, function_count * sizeof(*state->functions));
  }

  const uint16_t direct_value_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature->argument_value_count_u16);
  iree_vm_bytecode_frame_copy_direct_values(
      state->values, call->value_arguments.direct, direct_value_count);
  const uint16_t direct_ref_count = iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                                             signature->argument_ref_count_u16);
  for (uint16_t i = 0; i < direct_ref_count; ++i) {
    iree_vm_call_ref_argument_load_move(call, i, &state->refs[i]);
  }
  const uint16_t direct_function_count =
      iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
               signature->argument_function_count_u16);
  if (direct_function_count != 0) {
    memcpy(state->functions, call->function_arguments.direct,
           direct_function_count * sizeof(*state->functions));
  }
}

void iree_vm_bytecode_frame_reset_refs(iree_vm_ref_t* refs,
                                       uint32_t ref_count) {
  for (uint32_t i = 0; i < ref_count; ++i) {
    iree_vm_ref_reset(&refs[i]);
  }
}

iree_status_t iree_vm_bytecode_query_frame_layout(
    const iree_vm_bytecode_v0_function_row_t* function,
    iree_vm_bytecode_frame_layout_t* out_layout) {
  iree_vm_bytecode_frame_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_vm_bytecode_execution_state_t), &layout.frame.storage_size,
      IREE_STRUCT_FIELD_ALIGNED(1, iree_vm_call_packet_t,
                                iree_alignof(iree_vm_call_packet_t),
                                &layout.call_offset),
      IREE_STRUCT_FIELD_ALIGNED(function->value_register_count_u16, uint64_t,
                                iree_alignof(uint64_t), &layout.values_offset),
      IREE_STRUCT_FIELD_ALIGNED(function->ref_register_count_u16, iree_vm_ref_t,
                                iree_alignof(iree_vm_ref_t),
                                &layout.refs_offset),
      IREE_STRUCT_FIELD_ALIGNED(function->local_ref_count_u32, iree_vm_ref_t,
                                iree_alignof(iree_vm_ref_t),
                                &layout.local_refs_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          function->function_register_count_u16, iree_vm_function_ref_t,
          iree_alignof(iree_vm_function_ref_t), &layout.functions_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          function->local_function_count_u32, iree_vm_function_ref_t,
          iree_alignof(iree_vm_function_ref_t), &layout.local_functions_offset),
      IREE_STRUCT_FIELD(function->local_byte_length_u16, uint8_t,
                        &layout.local_bytes_offset)));
  layout.frame.storage_alignment = iree_max(
      iree_alignof(iree_vm_bytecode_execution_state_t),
      iree_max(iree_alignof(iree_vm_call_packet_t),
               iree_max(iree_alignof(uint64_t),
                        iree_max(iree_alignof(iree_vm_ref_t),
                                 iree_alignof(iree_vm_function_ref_t)))));
  *out_layout = layout;
  return iree_ok_status();
}

void iree_vm_bytecode_frame_cleanup(iree_vm_frame_t* frame) {
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)iree_vm_frame_storage(frame);
  iree_vm_bytecode_frame_reset_refs(state->refs, state->ref_count);
}
