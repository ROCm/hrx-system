// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_frame.h"

#include <string.h>

void iree_vm_bytecode_frame_initialize(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    iree_vm_bytecode_execution_state_t* state) {
  const uint32_t call_ref_scratch_count =
      iree_any_bit_set(function->flags_u16,
                       IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)
          ? IREE_VM_CALL_DIRECT_REGISTER_COUNT
          : 0;
  state->ref_count = function->ref_register_count_u16 +
                     function->local_ref_count_u32 + call_ref_scratch_count +
                     state->span_buffer_capacity;
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
  if (state->span_region_capacity) {
    memset(state->span_regions, 0,
           state->span_region_capacity * sizeof(*state->span_regions));
    if (signature->argument_value_count_u16 > direct_value_count) {
      memcpy(state->span_argument_values, state->call.value_arguments.overflow,
             (signature->argument_value_count_u16 - direct_value_count) *
                 sizeof(uint64_t));
      state->call.value_arguments.overflow = state->span_argument_values;
    }
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
        image->layout.signatures.descriptors + signature->descriptor_base_u32;
    uint16_t value_ordinal = 0;
    for (uint16_t i = 0; i < signature->argument_count_u16; ++i) {
      const uint16_t kind = descriptors[i].kind_u16;
      if (iree_vm_module_signature_type_kind_is_span(kind)) {
        value_ordinal = (value_ordinal + 1u) / 2u * 2u;
        uint64_t* values = value_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
                               ? state->values + value_ordinal
                               : state->span_argument_values + value_ordinal -
                                     IREE_VM_CALL_DIRECT_REGISTER_COUNT;
        const uint16_t element_kind =
            iree_vm_module_signature_element_kind(kind);
        const bool is_ref =
            iree_vm_module_signature_type_kind_is_ref(element_kind);
        const uint16_t region = value_ordinal / 2u;
        state->span_regions[region] = (iree_vm_bytecode_span_region_t){
            .data = (void*)(uintptr_t)values[0],
            .byte_length = values[1] * (is_ref ? IREE_VM_SPAN_REF_STRIDE
                                               : iree_vm_scalar_type_byte_width(
                                                     element_kind)),
            .ref_type =
                is_ref
                    ? image->resolved_ref_types[descriptors[i].type_ordinal_u16]
                    : NULL,
            .element_kind = element_kind,
            .access = iree_vm_module_signature_span_access(kind),
        };
        values[0] = (uint64_t)(region + 1u)
                    << IREE_VM_BYTECODE_SPAN_REGION_SHIFT;
        value_ordinal += 2;
      } else if (!iree_vm_module_signature_type_kind_is_ref(kind) &&
                 kind != IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION) {
        ++value_ordinal;
      }
    }
  }
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
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature) {
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
  layout.span_buffer_capacity = function->call_value_scratch_count_u16 / 2u;
  layout.span_buffers_offset = iree_vm_bytecode_frame_append(
      layout.span_buffer_capacity, sizeof(iree_vm_ref_t),
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
  if (function->call_value_scratch_count_u16) {
    // Private call values follow local bytes at their natural alignment. The
    // call adapter derives this address; it never enters a guest register.
    iree_vm_bytecode_frame_append(function->call_value_scratch_count_u16,
                                  sizeof(uint64_t), iree_alignof(uint64_t),
                                  &storage_size);
  }
  if (iree_vm_bytecode_signature_has_span_arguments(signature)) {
    layout.span_region_capacity = signature->argument_value_count_u16 / 2u;
    layout.span_regions_offset = iree_vm_bytecode_frame_append(
        layout.span_region_capacity, sizeof(iree_vm_bytecode_span_region_t),
        iree_alignof(iree_vm_bytecode_span_region_t), &storage_size);
    const uint16_t overflow_count =
        signature->argument_value_count_u16 > IREE_VM_CALL_DIRECT_REGISTER_COUNT
            ? signature->argument_value_count_u16 -
                  IREE_VM_CALL_DIRECT_REGISTER_COUNT
            : 0;
    layout.span_argument_values_offset =
        iree_vm_bytecode_frame_append(overflow_count, sizeof(uint64_t),
                                      iree_alignof(uint64_t), &storage_size);
  }
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
