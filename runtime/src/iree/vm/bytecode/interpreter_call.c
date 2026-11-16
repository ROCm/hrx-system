// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_call.h"

void iree_vm_bytecode_call_cleanup_completed(
    iree_vm_bytecode_execution_state_t* state) {
  const iree_vm_bytecode_call_cleanup_t cleanup = state->pending_call_cleanup;
  state->pending_call_cleanup = (iree_vm_bytecode_call_cleanup_t){0};
  for (uint16_t i = 0; i < cleanup.direct_scratch_count; ++i) {
    iree_vm_ref_reset(&cleanup.direct_scratch[i]);
  }
  for (uint16_t i = 0; i < IREE_VM_CALL_DIRECT_REGISTER_COUNT; ++i) {
    if ((cleanup.direct_source_move_mask & ((uint16_t)1u << i)) != 0) {
      iree_vm_ref_reset(&state->refs[i]);
    }
  }
  for (uint16_t i = 0; i < cleanup.overflow_argument_count; ++i) {
    iree_vm_ref_reset(&state->local_refs[i]);
  }
}

static bool iree_vm_bytecode_call_ref_matches(iree_vm_ref_t ref,
                                              iree_vm_ref_type_t type) {
  if (!ref.object) {
    return ref.type_and_state == 0;
  }
  return ((ref.type_and_state & IREE_VM_REF_STATE_MASK) ==
              IREE_VM_REF_STATE_OWNED ||
          (ref.type_and_state & IREE_VM_REF_STATE_BORROWED) != 0) &&
         iree_vm_ref_type(ref) == type;
}

// Validates dynamic arguments and resolves spans in implementation-private
// packet storage. Guest registers retain their encoded coordinates.
static iree_status_t iree_vm_bytecode_call_validate_arguments(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_module_execution_t* execution,
    const iree_vm_bytecode_v0_callable_type_row_t* callable_type,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    const iree_vm_bytecode_execution_state_t* state,
    iree_vm_call_packet_t* packet) {
  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
      iree_vm_bytecode_signature_descriptors(
          &image->layout.signatures, callable_type->signature_ordinal_u16);
  const uint32_t argument_count =
      iree_vm_bytecode_signature_argument_count(signature);
  const iree_vm_bytecode_v0_function_row_t* function = NULL;
  uint64_t* span_values = NULL;
  if (iree_vm_bytecode_signature_has_span_arguments(signature)) {
    function = &image->layout.functions
                    .rows[iree_vm_frame_function_ordinal(state->frame)];
    span_values = (uint64_t*)iree_host_align(
        (uintptr_t)(state->local_bytes + function->local_byte_length_u16),
        iree_alignof(uint64_t));
    const uint16_t direct_count = iree_min(signature->argument_value_count_u16,
                                           IREE_VM_CALL_DIRECT_REGISTER_COUNT);
    memcpy(span_values, state->values, direct_count * sizeof(uint64_t));
    if (signature->argument_value_count_u16 > direct_count) {
      memcpy(span_values + direct_count, state->local_bytes,
             (signature->argument_value_count_u16 - direct_count) *
                 sizeof(uint64_t));
    }
    packet->value_arguments.direct = span_values;
    packet->value_arguments.overflow =
        signature->argument_value_count_u16 > direct_count
            ? span_values + direct_count
            : NULL;
  }
  uint16_t value_ordinal = 0;
  uint16_t ref_ordinal = 0;
  uint16_t function_ordinal = 0;
  for (uint32_t i = 0; i < argument_count; ++i) {
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor =
        &descriptors[i];
    if (iree_vm_module_signature_type_kind_is_ref(descriptor->kind_u16)) {
      const iree_vm_ref_t argument =
          ref_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
              ? state->refs[ref_ordinal]
              : state->local_refs[ref_ordinal -
                                  IREE_VM_CALL_DIRECT_REGISTER_COUNT];
      if (!iree_vm_bytecode_call_ref_matches(
              argument,
              image->resolved_ref_types[descriptor->type_ordinal_u16])) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "call ref argument type mismatch");
      }
      ++ref_ordinal;
    } else if (descriptor->kind_u16 ==
               IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
      const iree_vm_function_ref_t argument =
          function_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
              ? state->functions[function_ordinal]
              : state->local_functions[function_ordinal -
                                       IREE_VM_CALL_DIRECT_REGISTER_COUNT];
      if (!iree_vm_function_ref_matches_callable_type(
              execution, argument, descriptor->type_ordinal_u16)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "call function argument contract mismatch");
      }
      ++function_ordinal;
    } else if (iree_vm_module_signature_type_kind_is_span(
                   descriptor->kind_u16)) {
      value_ordinal = (value_ordinal + 1u) / 2u * 2u;
      const uint64_t offset = span_values[value_ordinal];
      const uint64_t count = span_values[value_ordinal + 1];
      const uint16_t element_kind =
          iree_vm_module_signature_element_kind(descriptor->kind_u16);
      void* data = NULL;
      if (iree_vm_module_signature_type_kind_is_ref(element_kind)) {
        const uint64_t index = offset / 16u;
        if (offset % 16u != 0 || index > function->local_ref_count_u32 ||
            count > function->local_ref_count_u32 - index) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "call reference span exceeds local storage");
        }
        data = state->local_refs + index;
      } else {
        const uint8_t width = iree_vm_scalar_type_byte_width(element_kind);
        if (offset > function->local_byte_length_u16 ||
            count > (function->local_byte_length_u16 - offset) / width) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "call scalar span exceeds local storage");
        }
        data = state->local_bytes + offset;
      }
      span_values[value_ordinal] = (uintptr_t)data;
      value_ordinal += 2;
    } else {
      ++value_ordinal;
    }
  }
  return iree_ok_status();
}

// Preparation for one physical child packet, before ownership is transferred.
typedef struct iree_vm_bytecode_call_plan_t {
  // Complete packet copied into the invocation driver's request.
  iree_vm_call_packet_t packet;
  // Number of direct ref arguments.
  uint16_t direct_ref_argument_count;
  // Direct ref result prefix that replaces caller sources.
  uint16_t direct_ref_result_count;
  // Number of ref arguments in the overflow packet.
  uint16_t overflow_ref_argument_count;
  // Ownership-transfer bits for direct ref arguments.
  uint16_t direct_ref_move_mask;
  // Whether direct arguments use implementation-private scratch.
  bool uses_direct_ref_scratch;
} iree_vm_bytecode_call_plan_t;

static iree_status_t iree_vm_bytecode_call_prepare(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state, uint16_t callable_type_ordinal,
    uint16_t direct_ref_move_mask, iree_vm_bytecode_call_plan_t* out_plan) {
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &image->layout.callable_types.rows[callable_type_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &image->layout.signatures.rows[callable_type->signature_ordinal_u16];
  iree_vm_bytecode_call_plan_t plan = {0};
  plan.direct_ref_argument_count = iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                                            signature->argument_ref_count_u16);
  plan.direct_ref_result_count = iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                                          signature->result_ref_count_u16);
  plan.overflow_ref_argument_count =
      signature->argument_ref_count_u16 - plan.direct_ref_argument_count;
  plan.direct_ref_move_mask = direct_ref_move_mask;
  const uint16_t all_direct_ref_move_mask =
      plan.direct_ref_argument_count == IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? UINT16_MAX
          : (uint16_t)((1u << plan.direct_ref_argument_count) - 1u);
  plan.uses_direct_ref_scratch =
      direct_ref_move_mask != all_direct_ref_move_mask;

  const uint16_t argument_value_overflow =
      signature->argument_value_count_u16 > IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? signature->argument_value_count_u16 -
                IREE_VM_CALL_DIRECT_REGISTER_COUNT
          : 0;
  const uint16_t argument_function_overflow =
      signature->argument_function_count_u16 >
              IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? signature->argument_function_count_u16 -
                IREE_VM_CALL_DIRECT_REGISTER_COUNT
          : 0;
  const uint16_t result_ref_overflow =
      signature->result_ref_count_u16 > IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? signature->result_ref_count_u16 - IREE_VM_CALL_DIRECT_REGISTER_COUNT
          : 0;
  plan.packet = (iree_vm_call_packet_t){
      .value_arguments =
          {
              .direct = state->values,
              .overflow = argument_value_overflow
                              ? (const uint64_t*)state->local_bytes
                              : NULL,
          },
      .ref_arguments =
          {
              .direct = plan.uses_direct_ref_scratch ? state->call_ref_scratch
                                                     : state->refs,
              .overflow =
                  plan.overflow_ref_argument_count ? state->local_refs : NULL,
          },
      .function_arguments =
          {
              .direct = state->functions,
              .overflow =
                  argument_function_overflow ? state->local_functions : NULL,
          },
      .value_results =
          {
              .direct = state->values,
              .overflow =
                  signature->result_value_count_u16 >
                          IREE_VM_CALL_DIRECT_REGISTER_COUNT
                      ? (uint64_t*)state->local_bytes + argument_value_overflow
                      : NULL,
          },
      .ref_results =
          {
              .direct = state->refs,
              .overflow =
                  result_ref_overflow
                      ? state->local_refs + plan.overflow_ref_argument_count
                      : NULL,
          },
      .function_results =
          {
              .direct = state->functions,
              .overflow =
                  signature->result_function_count_u16 >
                          IREE_VM_CALL_DIRECT_REGISTER_COUNT
                      ? state->local_functions + argument_function_overflow
                      : NULL,
          },
  };
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_validate_arguments(
      image, execution, callable_type, signature, state, &plan.packet));
  *out_plan = plan;
  return iree_ok_status();
}

// Commits infallible ref staging after the generic target request succeeds.
static void iree_vm_bytecode_call_commit(
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_bytecode_call_plan_t* plan) {
  iree_vm_bytecode_call_cleanup_t cleanup = {
      .overflow_argument_count = plan->overflow_ref_argument_count,
  };
  if (plan->uses_direct_ref_scratch) {
    for (uint16_t i = 0; i < plan->direct_ref_argument_count; ++i) {
      const iree_vm_ref_t source = state->refs[i];
      if ((plan->direct_ref_move_mask & ((uint16_t)1u << i)) != 0) {
        // Transfer before the child starts so its final release is not held
        // back by a hidden caller owner. Internal borrows remain borrowed.
        state->call_ref_scratch[i] = source;
        state->refs[i] = iree_vm_ref_null();
      } else {
        state->call_ref_scratch[i] = source;
        if (source.object) {
          state->call_ref_scratch[i].type_and_state |=
              IREE_VM_REF_STATE_BORROWED;
        }
      }
    }
    cleanup.direct_scratch = state->call_ref_scratch;
    cleanup.direct_scratch_count = plan->direct_ref_argument_count;
  }

  const uint16_t result_prefix_mask =
      plan->direct_ref_result_count == IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? UINT16_MAX
          : (uint16_t)((1u << plan->direct_ref_result_count) - 1u);
  cleanup.direct_source_move_mask =
      plan->uses_direct_ref_scratch
          ? 0
          : plan->direct_ref_move_mask & ~result_prefix_mask;
  state->pending_call_cleanup = cleanup;
}

iree_status_t iree_vm_bytecode_call_direct(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_bytecode_control_call_t* record,
    iree_vm_execution_outcome_t* out_outcome) {
  uint16_t callable_type_ordinal = 0;
  iree_vm_module_local_function_t local_function = {0};
  if (record->target_kind_u8 == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL) {
    const iree_vm_bytecode_v0_function_row_t* function =
        &image->layout.functions.rows[record->target_ordinal_u16];
    callable_type_ordinal = function->callable_type_ordinal_u16;
    local_function.function_ordinal = record->target_ordinal_u16;
    local_function.callable_type_ordinal = callable_type_ordinal;
    local_function.flags =
        iree_any_bit_set(function->flags_u16,
                         IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD)
            ? IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD
            : IREE_VM_MODULE_FUNCTION_FLAG_NONE;
  } else {
    callable_type_ordinal =
        image->layout.imports.entries[record->target_ordinal_u16]
            .callable_type_ordinal_u16;
  }

  iree_vm_bytecode_call_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_prepare(
      image, execution, state, callable_type_ordinal,
      record->direct_ref_move_mask_u16, &plan));
  iree_status_t status =
      record->target_kind_u8 == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL
          ? iree_vm_invocation_call_local(execution, local_function,
                                          &plan.packet, out_outcome)
          : iree_vm_invocation_call_import(execution,
                                           record->target_ordinal_u16,
                                           &plan.packet, out_outcome);
  if (iree_status_is_ok(status)) {
    iree_vm_bytecode_call_commit(state, &plan);
  }
  return status;
}

iree_status_t iree_vm_bytecode_call_indirect(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_bytecode_control_call_indirect_t* record,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_call_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_prepare(
      image, execution, state, record->callable_type_ordinal_u16,
      record->direct_ref_move_mask_u16, &plan));
  iree_status_t status = iree_vm_invocation_call_function_ref(
      execution, state->functions[record->target_f8],
      record->callable_type_ordinal_u16, &plan.packet, out_outcome);
  if (iree_status_is_ok(status)) {
    iree_vm_bytecode_call_commit(state, &plan);
  }
  return status;
}
