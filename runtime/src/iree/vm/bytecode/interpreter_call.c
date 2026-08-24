// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_call.h"

#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/invocation_storage.h"

void iree_vm_bytecode_call_cleanup_completed(
    iree_vm_bytecode_execution_state_t* state) {
  const iree_vm_bytecode_call_cleanup_t cleanup = state->pending_call_cleanup;
  if (cleanup.direct_argument_count == 0 &&
      cleanup.overflow_argument_count == 0) {
    return;
  }
  state->pending_call_cleanup = (iree_vm_bytecode_call_cleanup_t){0};
  for (uint16_t i = 0; i < cleanup.direct_argument_count; ++i) {
    iree_vm_ref_reset(&cleanup.direct_arguments[i]);
  }
  for (uint16_t i = 0; i < cleanup.overflow_argument_count; ++i) {
    iree_vm_ref_reset(&state->local_refs[i]);
  }
}

// Returns one additional packet owner while preserving internal borrowed
// states exactly. An owned source is retained; a borrow remains dominated by
// the paused invocation frame chain.
static iree_vm_ref_t iree_vm_bytecode_call_stage_ref(iree_vm_ref_t source) {
  if (source.object && (source.type_and_state & IREE_VM_REF_STATE_MASK) ==
                           IREE_VM_REF_STATE_OWNED) {
    iree_vm_ref_object_retain(source.object);
  }
  return source;
}

// Validates the dynamic ref and function arguments of one exact callable. The
// scalar banks carry untagged exact bits and require no execution-time checks.
static iree_status_t iree_vm_bytecode_call_validate_arguments(
    const iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    const iree_vm_bytecode_v0_callable_type_row_t* callable_type,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    const iree_vm_bytecode_execution_state_t* state) {
  if (iree_any_bit_set(callable_type->flags_u16,
                       IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD) &&
      execution->invocation->has_external_borrowed_arguments) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "a potentially yielding call cannot carry borrowed invocation refs");
  }
  if (signature->argument_ref_count_u16 == 0 &&
      signature->argument_function_count_u16 == 0) {
    return iree_ok_status();
  }

  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
      iree_vm_bytecode_signature_descriptors(
          &module->layout.signatures, callable_type->signature_ordinal_u16);
  const uint32_t argument_count =
      iree_vm_bytecode_signature_argument_count(signature);
  uint16_t ref_ordinal = 0;
  uint16_t function_ordinal = 0;
  for (uint32_t i = 0; i < argument_count; ++i) {
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor =
        &descriptors[i];
    if (descriptor->kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
      const iree_vm_ref_t argument = ref_ordinal < 16
                                         ? state->refs[ref_ordinal]
                                         : state->local_refs[ref_ordinal - 16];
      if (argument.object &&
          iree_vm_ref_type(argument) !=
              module->resolved_ref_types[descriptor->type_ordinal_u16]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "call ref argument type mismatch");
      }
      ++ref_ordinal;
    } else if (descriptor->kind_u16 ==
               IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
      const iree_vm_function_ref_t argument =
          function_ordinal < 16 ? state->functions[function_ordinal]
                                : state->local_functions[function_ordinal - 16];
      if (!iree_vm_program_function_ref_matches(
              execution->invocation->process->program, argument,
              execution->linked_module, descriptor->type_ordinal_u16)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "call function argument contract mismatch");
      }
      ++function_ordinal;
    }
  }
  return iree_ok_status();
}

// Describes one preflighted target using its calling-module contract.
typedef struct iree_vm_bytecode_call_target_t {
  // Exact callable contract in the calling module.
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type;
  // Resolved target module in the current program.
  const iree_vm_linked_module_t* linked_module;
  // Resolved module-local target function ordinal.
  uint16_t function_ordinal;
  // True when the resolved target may suspend.
  bool may_yield;
} iree_vm_bytecode_call_target_t;

static iree_status_t iree_vm_bytecode_call_preflight_direct_target(
    const iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    const iree_vm_isa_control_call_record_t* record,
    iree_vm_bytecode_call_target_t* out_target) {
  iree_vm_bytecode_call_target_t target = {0};
  if (record->target_kind_u8 == IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL) {
    const iree_vm_bytecode_v0_function_row_t* function =
        &module->layout.functions.rows[record->target_ordinal_u16];
    target.callable_type =
        iree_vm_bytecode_function_callable_type(&module->layout, function);
    target.linked_module = execution->linked_module;
    target.function_ordinal = record->target_ordinal_u16;
    target.may_yield = iree_any_bit_set(
        function->flags_u16, IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD);
  } else {
    const iree_vm_bytecode_v0_import_entry_row_t* import =
        &module->layout.imports.entries[record->target_ordinal_u16];
    const uint64_t target_bits =
        execution->linked_module
            ->import_target_bits[record->target_ordinal_u16];
    if (target_bits == 0) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "optional import is unresolved");
    }
    target.callable_type =
        &module->layout.callable_types.rows[import->callable_type_ordinal_u16];
    const iree_vm_program_t* program = execution->invocation->process->program;
    target.linked_module =
        &program->linked_modules[iree_vm_program_target_module_ordinal(
            target_bits)];
    target.function_ordinal =
        iree_vm_program_target_function_ordinal(target_bits);
    target.may_yield = iree_vm_program_target_may_yield(target_bits);
  }
  *out_target = target;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_call_preflight_indirect_target(
    const iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    const iree_vm_bytecode_execution_state_t* state,
    const iree_vm_isa_control_call_indirect_record_t* record,
    iree_vm_bytecode_call_target_t* out_target) {
  const iree_vm_function_ref_t function_ref =
      state->functions[record->target_f8];
  if (iree_vm_function_ref_is_null(function_ref)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "indirect call target is null");
  }
  if (!iree_vm_program_function_ref_matches(
          execution->invocation->process->program, function_ref,
          execution->linked_module, record->callable_type_ordinal_u16)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect call target contract mismatch");
  }
  const iree_vm_program_t* program = execution->invocation->process->program;
  *out_target = (iree_vm_bytecode_call_target_t){
      .callable_type = &module->layout.callable_types
                            .rows[record->callable_type_ordinal_u16],
      .linked_module =
          &program->linked_modules[iree_vm_program_target_module_ordinal(
              function_ref.target_bits)],
      .function_ordinal =
          iree_vm_program_target_function_ordinal(function_ref.target_bits),
      .may_yield = iree_vm_program_target_may_yield(function_ref.target_bits),
  };
  return iree_ok_status();
}

// Clears moved caller sources after a successful child entry. Direct result
// slots have already replaced their corresponding sources on synchronous
// completion and are not cleared.
static void iree_vm_bytecode_call_commit_mixed_moves(
    iree_vm_ref_t* refs, uint16_t direct_argument_count,
    uint16_t direct_result_count, uint16_t direct_ref_move_mask,
    iree_vm_execution_outcome_t outcome) {
  const uint16_t first_source =
      outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED
          ? iree_min(direct_argument_count, direct_result_count)
          : 0;
  for (uint16_t i = first_source; i < direct_argument_count; ++i) {
    if (direct_ref_move_mask & ((uint16_t)1u << i)) {
      iree_vm_ref_reset(&refs[i]);
    }
  }
}

static iree_status_t iree_vm_bytecode_call_start_preflighted(
    iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_bytecode_call_target_t* target, uint16_t direct_ref_move_mask,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &module->layout.signatures
           .rows[target->callable_type->signature_ordinal_u16];
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_validate_arguments(
      module, execution, target->callable_type, signature, state));

  const uint16_t direct_ref_argument_count =
      iree_min(16u, signature->argument_ref_count_u16);
  const uint16_t direct_ref_result_count =
      iree_min(16u, signature->result_ref_count_u16);
  const uint16_t valid_ref_move_mask =
      direct_ref_argument_count == 16
          ? UINT16_MAX
          : (uint16_t)((1u << direct_ref_argument_count) - 1u);
  const bool uses_direct_ref_scratch =
      direct_ref_move_mask != valid_ref_move_mask;
  const uint16_t argument_ref_overflow =
      signature->argument_ref_count_u16 - direct_ref_argument_count;
  const uint16_t result_ref_overflow =
      signature->result_ref_count_u16 - direct_ref_result_count;
  iree_vm_ref_t* direct_ref_arguments = state->refs;
  if (uses_direct_ref_scratch) {
    direct_ref_arguments =
        state->local_refs + argument_ref_overflow + result_ref_overflow;
    for (uint16_t i = 0; i < direct_ref_argument_count; ++i) {
      iree_vm_ref_t staged_ref =
          direct_ref_move_mask & ((uint16_t)1u << i)
              ? iree_vm_bytecode_call_stage_ref(state->refs[i])
              : iree_vm_ref_from_ptr_borrowed(state->refs[i].object,
                                              iree_vm_ref_type(state->refs[i]));
      iree_vm_ref_reset(&direct_ref_arguments[i]);
      direct_ref_arguments[i] = staged_ref;
    }
  }

  const uint16_t argument_value_overflow =
      signature->argument_value_count_u16 > 16
          ? signature->argument_value_count_u16 - 16
          : 0;
  const uint16_t argument_function_overflow =
      signature->argument_function_count_u16 > 16
          ? signature->argument_function_count_u16 - 16
          : 0;
  const iree_vm_call_packet_t call = {
      .value_arguments =
          {
              .direct = state->values,
              .overflow = argument_value_overflow
                              ? (const uint64_t*)state->local_bytes
                              : NULL,
          },
      .ref_arguments =
          {
              .direct = direct_ref_arguments,
              .overflow = argument_ref_overflow ? state->local_refs : NULL,
          },
      .value_results =
          {
              .direct = state->values,
              .overflow =
                  signature->result_value_count_u16 > 16
                      ? (uint64_t*)state->local_bytes + argument_value_overflow
                      : NULL,
          },
      .ref_results =
          {
              .direct = state->refs,
              .overflow = result_ref_overflow
                              ? state->local_refs + argument_ref_overflow
                              : NULL,
          },
      .function_arguments =
          {
              .direct = state->functions,
              .overflow =
                  argument_function_overflow ? state->local_functions : NULL,
          },
      .function_results =
          {
              .direct = state->functions,
              .overflow =
                  signature->result_function_count_u16 > 16
                      ? state->local_functions + argument_function_overflow
                      : NULL,
          },
  };

  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_request_call(
      execution->invocation, target->linked_module, target->function_ordinal,
      target->may_yield, &call, &outcome);
  if (!iree_status_is_ok(status)) {
    if (uses_direct_ref_scratch) {
      state->pending_call_cleanup = (iree_vm_bytecode_call_cleanup_t){
          .direct_arguments = direct_ref_arguments,
          .direct_argument_count = direct_ref_argument_count,
      };
      iree_vm_bytecode_call_cleanup_completed(state);
    }
    return status;
  }

  iree_vm_ref_t* direct_cleanup = NULL;
  uint16_t direct_cleanup_count = 0;
  if (uses_direct_ref_scratch) {
    direct_cleanup = direct_ref_arguments;
    direct_cleanup_count = direct_ref_argument_count;
    iree_vm_bytecode_call_commit_mixed_moves(
        state->refs, direct_ref_argument_count, direct_ref_result_count,
        direct_ref_move_mask, outcome);
  } else if (direct_ref_argument_count > direct_ref_result_count) {
    direct_cleanup = state->refs + direct_ref_result_count;
    direct_cleanup_count = direct_ref_argument_count - direct_ref_result_count;
  }
  state->pending_call_cleanup = (iree_vm_bytecode_call_cleanup_t){
      .direct_arguments = direct_cleanup,
      .direct_argument_count = direct_cleanup_count,
      .overflow_argument_count = argument_ref_overflow,
  };
  if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    iree_vm_bytecode_call_cleanup_completed(state);
  }
  *out_outcome = outcome;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_call_start_direct(
    iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_isa_control_call_record_t* record,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_call_target_t target = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_preflight_direct_target(
      module, execution, record, &target));
  return iree_vm_bytecode_call_start_preflighted(
      module, execution, state, &target, record->direct_ref_move_mask_u16,
      out_outcome);
}

iree_status_t iree_vm_bytecode_call_start_indirect(
    iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_isa_control_call_indirect_record_t* record,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_call_target_t target = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_call_preflight_indirect_target(
      module, execution, state, record, &target));
  return iree_vm_bytecode_call_start_preflighted(
      module, execution, state, &target, record->direct_ref_move_mask_u16,
      out_outcome);
}
