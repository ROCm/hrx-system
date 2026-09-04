// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/invocation.h"

#include <string.h>

#include "iree/base/internal/fpu_state.h"
#include "iree/vm/invocation_storage.h"

static const iree_fpu_state_flags_t iree_vm_invocation_fpu_state_flags =
    IREE_FPU_STATE_FLAG_MASK_EXCEPTIONS | IREE_FPU_STATE_FLAG_ROUND_TO_NEAREST;

// Allocation-only owner prefix preceding max-aligned invocation storage.
typedef struct iree_alignas(iree_max_align_t) iree_vm_invocation_allocation_t {
  // Allocator owning the complete prefix and invocation allocation.
  iree_allocator_t host_allocator;
} iree_vm_invocation_allocation_t;

static_assert(sizeof(iree_vm_invocation_allocation_t) % iree_max_align_t == 0,
              "invocation allocation prefix must preserve max alignment");

//===----------------------------------------------------------------------===//
// Storage Lifetime
//===----------------------------------------------------------------------===//

static bool iree_vm_invocation_align_address(uintptr_t address,
                                             iree_host_size_t alignment,
                                             uintptr_t* out_address) {
  iree_host_size_t aligned_address = 0;
  if (!iree_host_size_checked_align((iree_host_size_t)address, alignment,
                                    &aligned_address)) {
    return false;
  }
  *out_address = (uintptr_t)aligned_address;
  return true;
}

bool iree_vm_invocation_is_idle(const iree_vm_invocation_t* invocation) {
  return invocation->state == IREE_VM_INVOCATION_STATE_IDLE;
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_initialize(
    iree_byte_span_t storage, iree_vm_invocation_t** out_invocation) {
  if (!out_invocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_invocation is required");
  }
  *out_invocation = NULL;
  if (!storage.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation storage is required");
  }
  uintptr_t invocation_address = 0;
  if (!iree_vm_invocation_align_address(
          (uintptr_t)storage.data, iree_max_align_t, &invocation_address)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation storage alignment overflows");
  }
  const iree_host_size_t leading_padding =
      (iree_host_size_t)(invocation_address - (uintptr_t)storage.data);
  if (leading_padding > storage.data_length ||
      storage.data_length - leading_padding <
          iree_sizeof_struct(iree_vm_invocation_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation storage cannot contain its header");
  }

  iree_vm_invocation_t* invocation = (iree_vm_invocation_t*)invocation_address;
  memset(invocation, 0, sizeof(*invocation));
  invocation->storage_begin = storage.data;
  invocation->storage_end = storage.data + storage.data_length;
  invocation->stack_cursor = iree_vm_invocation_stack_base(invocation);
  iree_atomic_store(&invocation->cancel_reason,
                    IREE_VM_INVOCATION_CANCEL_REASON_IDLE,
                    iree_memory_order_relaxed);
  *out_invocation = invocation;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_invocation_deinitialize(
    iree_vm_invocation_t* invocation) {
  if (!invocation) return;
  IREE_ASSERT(iree_vm_invocation_is_idle(invocation));
  IREE_ASSERT(!invocation->is_allocated);
  memset(invocation, 0, sizeof(*invocation));
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_allocate(
    iree_host_size_t storage_size, iree_allocator_t host_allocator,
    iree_vm_invocation_t** out_invocation) {
  if (!out_invocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_invocation is required");
  }
  *out_invocation = NULL;
  iree_vm_invocation_allocation_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_with_trailing(
      host_allocator, sizeof(*allocation), storage_size, (void**)&allocation));
  uint8_t* storage = (uint8_t*)allocation + sizeof(*allocation);
  iree_vm_invocation_t* invocation = NULL;
  iree_status_t status = iree_vm_invocation_initialize(
      iree_make_byte_span(storage, storage_size), &invocation);
  if (iree_status_is_ok(status)) {
    allocation->host_allocator = host_allocator;
    invocation->is_allocated = true;
    *out_invocation = invocation;
  } else {
    iree_allocator_free(host_allocator, allocation);
  }
  return status;
}

IREE_API_EXPORT void iree_vm_invocation_free(iree_vm_invocation_t* invocation) {
  if (!invocation) return;
  IREE_ASSERT(iree_vm_invocation_is_idle(invocation));
  IREE_ASSERT(invocation->is_allocated);
  iree_vm_invocation_allocation_t* allocation =
      (iree_vm_invocation_allocation_t*)(invocation->storage_begin -
                                         sizeof(*allocation));
  const iree_allocator_t host_allocator = allocation->host_allocator;
  memset(invocation, 0, sizeof(*invocation));
  iree_allocator_free(host_allocator, allocation);
}

//===----------------------------------------------------------------------===//
// Public Boundary And Root Staging
//===----------------------------------------------------------------------===//

static bool iree_vm_invocation_ranges_overlap(uintptr_t lhs_begin,
                                              uintptr_t lhs_end,
                                              uintptr_t rhs_begin,
                                              uintptr_t rhs_end) {
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

iree_status_t iree_vm_invocation_validate_boundary(
    const iree_vm_invocation_t* invocation, iree_vm_variant_span_t arguments,
    iree_vm_variant_span_t results, iree_byte_span_t outcome_storage) {
  if (IREE_UNLIKELY(!invocation || (arguments.count && !arguments.data) ||
                    (results.count && !results.data) || !outcome_storage.data ||
                    !outcome_storage.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid invocation boundary storage");
  }
  iree_host_size_t argument_size = 0;
  iree_host_size_t result_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(arguments.count,
                                                sizeof(*arguments.data),
                                                &argument_size) ||
                    !iree_host_size_checked_mul(
                        results.count, sizeof(*results.data), &result_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation span size overflows");
  }
  const uintptr_t argument_begin = (uintptr_t)arguments.data;
  const uintptr_t result_begin = (uintptr_t)results.data;
  const uintptr_t outcome_begin = (uintptr_t)outcome_storage.data;
  if (IREE_UNLIKELY(argument_size > UINTPTR_MAX - argument_begin ||
                    result_size > UINTPTR_MAX - result_begin ||
                    outcome_storage.data_length >
                        UINTPTR_MAX - outcome_begin)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation span address overflows");
  }
  const uintptr_t argument_end = argument_begin + argument_size;
  const uintptr_t result_end = result_begin + result_size;
  const uintptr_t outcome_end = outcome_begin + outcome_storage.data_length;
  const uintptr_t invocation_begin = (uintptr_t)invocation->storage_begin;
  const uintptr_t invocation_end = (uintptr_t)invocation->storage_end;

  const bool argument_overlaps =
      argument_size &&
      ((result_size &&
        iree_vm_invocation_ranges_overlap(argument_begin, argument_end,
                                          result_begin, result_end)) ||
       iree_vm_invocation_ranges_overlap(argument_begin, argument_end,
                                         invocation_begin, invocation_end) ||
       iree_vm_invocation_ranges_overlap(argument_begin, argument_end,
                                         outcome_begin, outcome_end));
  const bool result_overlaps =
      result_size &&
      (iree_vm_invocation_ranges_overlap(result_begin, result_end,
                                         invocation_begin, invocation_end) ||
       iree_vm_invocation_ranges_overlap(result_begin, result_end,
                                         outcome_begin, outcome_end));
  const bool outcome_overlaps = iree_vm_invocation_ranges_overlap(
      outcome_begin, outcome_end, invocation_begin, invocation_end);
  if (IREE_UNLIKELY(argument_overlaps || result_overlaps || outcome_overlaps)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation boundary storage overlaps");
  }
  return iree_ok_status();
}

void iree_vm_invocation_consume_arguments(iree_vm_variant_span_t arguments) {
  iree_vm_variant_span_reset(arguments);
}

static iree_vm_variant_t* iree_vm_invocation_variant_slot(
    iree_vm_variant_t* variants, uint32_t variant_offset) {
  return (iree_vm_variant_t*)((uint8_t*)variants + variant_offset);
}

static iree_status_t iree_vm_invocation_validate_arguments(
    const iree_vm_program_t* program, uint64_t target_bits,
    const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments,
    bool* out_has_external_borrowed_arguments) {
  if (IREE_UNLIKELY(
          arguments.count !=
          iree_vm_program_callable_abi_argument_count(callable_abi))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation argument count mismatch");
  }
  for (uint16_t i = 0; i < callable_abi->argument_counts.value_count; ++i) {
    const iree_vm_program_scalar_field_abi_t field =
        callable_abi->value_arguments[i];
    const iree_vm_variant_t argument =
        *iree_vm_invocation_variant_slot(arguments.data, field.variant_offset);
    if (argument.metadata != field.variant_metadata ||
        (argument.payload & ~field.payload_mask) != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invocation scalar argument is not canonical");
    }
  }

  bool has_external_borrowed_arguments = false;
  for (uint16_t i = 0; i < callable_abi->argument_counts.ref_count; ++i) {
    const iree_vm_program_ref_field_abi_t field =
        callable_abi->ref_arguments[i];
    const iree_vm_variant_t argument =
        *iree_vm_invocation_variant_slot(arguments.data, field.variant_offset);
    if (!iree_vm_variant_is_ref(argument) ||
        (argument.payload &&
         iree_vm_variant_ref_type(argument) != field.type) ||
        (!argument.payload &&
         argument.metadata != IREE_VM_VARIANT_TAG_OWNED_REF)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invocation ref argument type mismatch");
    }
    has_external_borrowed_arguments |=
        argument.payload != 0 &&
        (argument.metadata & IREE_VM_VARIANT_TAG_MASK) ==
            IREE_VM_VARIANT_TAG_BORROWED_REF;
  }

  for (uint16_t i = 0; i < callable_abi->argument_counts.function_count; ++i) {
    const iree_vm_program_function_field_abi_t field =
        callable_abi->function_arguments[i];
    const iree_vm_variant_t argument =
        *iree_vm_invocation_variant_slot(arguments.data, field.variant_offset);
    if (!iree_vm_variant_is_function_ref(argument)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invocation function argument type mismatch");
    }
    const iree_vm_function_ref_t function_ref = {
        argument.payload,
        argument.metadata & ~(uint64_t)IREE_VM_VARIANT_TAG_MASK,
    };
    if (!iree_vm_program_function_ref_matches_mapping(program, function_ref,
                                                      field.callable_mapping)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invocation function argument contract mismatch");
    }
  }
  if (has_external_borrowed_arguments &&
      iree_vm_program_target_may_yield(target_bits)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "a possibly yielding function cannot accept borrowed refs");
  }
  *out_has_external_borrowed_arguments = has_external_borrowed_arguments;
  return iree_ok_status();
}

iree_status_t iree_vm_invocation_preflight_root(
    iree_vm_invocation_t* invocation, const iree_vm_program_t* program,
    uint64_t target_bits, const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    bool* out_has_external_borrowed_arguments) {
  if (!iree_vm_invocation_is_idle(invocation)) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation is already active");
  }
  if (results.count !=
      iree_vm_program_callable_abi_result_count(callable_abi)) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation result count mismatch");
  }

  bool has_external_borrowed_arguments = false;
  iree_status_t status = iree_vm_invocation_validate_arguments(
      program, target_bits, callable_abi, arguments,
      &has_external_borrowed_arguments);
  const iree_host_size_t root_storage_size =
      callable_abi->root_layout.storage_size;
  if (iree_status_is_ok(status) &&
      root_storage_size >
          (iree_host_size_t)(invocation->storage_end -
                             iree_vm_invocation_stack_base(invocation))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "root call staging exceeds invocation capacity");
  }
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_consume_arguments(arguments);
    return status;
  }
  *out_has_external_borrowed_arguments = has_external_borrowed_arguments;
  return iree_ok_status();
}

static iree_vm_call_packet_t iree_vm_invocation_make_root_packet(
    uint8_t* storage, const iree_vm_program_callable_abi_t* callable_abi) {
  const iree_vm_program_root_layout_t* layout = &callable_abi->root_layout;
#define IREE_VM_ROOT_BANK(type, offset, count)                               \
  {                                                                          \
    (type*)(storage + (offset)),                                             \
        (type*)(storage +                                                    \
                ((count) > IREE_VM_CALL_DIRECT_REGISTER_COUNT                \
                     ? (offset) +                                            \
                           IREE_VM_CALL_DIRECT_REGISTER_COUNT * sizeof(type) \
                     : 0))                                                   \
  }
  const iree_vm_call_packet_t packet = {
      .value_arguments =
          IREE_VM_ROOT_BANK(uint64_t, layout->value_arguments_offset,
                            callable_abi->argument_counts.value_count),
      .ref_arguments =
          IREE_VM_ROOT_BANK(iree_vm_ref_t, layout->ref_arguments_offset,
                            callable_abi->argument_counts.ref_count),
      .value_results =
          IREE_VM_ROOT_BANK(uint64_t, layout->value_results_offset,
                            callable_abi->result_counts.value_count),
      .ref_results =
          IREE_VM_ROOT_BANK(iree_vm_ref_t, layout->ref_results_offset,
                            callable_abi->result_counts.ref_count),
      .function_arguments = IREE_VM_ROOT_BANK(
          iree_vm_function_ref_t, layout->function_arguments_offset,
          callable_abi->argument_counts.function_count),
      .function_results = IREE_VM_ROOT_BANK(
          iree_vm_function_ref_t, layout->function_results_offset,
          callable_abi->result_counts.function_count),
  };
#undef IREE_VM_ROOT_BANK
  return packet;
}

static iree_vm_call_packet_t iree_vm_invocation_root_packet(
    iree_vm_invocation_t* invocation) {
  return iree_vm_invocation_make_root_packet(
      iree_vm_invocation_stack_base(invocation), invocation->root_callable_abi);
}

static void iree_vm_invocation_initialize_root_banks(
    const iree_vm_program_callable_abi_t* callable_abi,
    const iree_vm_call_packet_t* packet) {
  if (callable_abi->argument_counts.ref_count) {
    memset(packet->ref_arguments.direct, 0,
           callable_abi->argument_counts.ref_count * sizeof(iree_vm_ref_t));
  }
  if (callable_abi->result_counts.ref_count) {
    memset(packet->ref_results.direct, 0,
           callable_abi->result_counts.ref_count * sizeof(iree_vm_ref_t));
  }
  if (callable_abi->result_counts.function_count) {
    memset(packet->function_results.direct, 0,
           callable_abi->result_counts.function_count *
               sizeof(iree_vm_function_ref_t));
  }
}

static void iree_vm_invocation_stage_arguments(
    const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments, const iree_vm_call_packet_t* packet) {
  uint64_t* value_arguments = (uint64_t*)packet->value_arguments.direct;
  for (uint16_t i = 0; i < callable_abi->argument_counts.value_count; ++i) {
    iree_vm_variant_t* argument = iree_vm_invocation_variant_slot(
        arguments.data, callable_abi->value_arguments[i].variant_offset);
    value_arguments[i] = argument->payload;
    *argument = iree_vm_variant_empty();
  }
  iree_vm_ref_t* ref_arguments = packet->ref_arguments.direct;
  for (uint16_t i = 0; i < callable_abi->argument_counts.ref_count; ++i) {
    iree_vm_variant_t* argument = iree_vm_invocation_variant_slot(
        arguments.data, callable_abi->ref_arguments[i].variant_offset);
    if (argument->payload) {
      const uintptr_t ownership =
          (argument->metadata & IREE_VM_VARIANT_TAG_MASK) ==
                  IREE_VM_VARIANT_TAG_BORROWED_REF
              ? IREE_VM_REF_STATE_BORROWED
              : IREE_VM_REF_STATE_OWNED;
      ref_arguments[i].object = (void*)(uintptr_t)argument->payload;
      ref_arguments[i].type_and_state =
          (uintptr_t)iree_vm_variant_ref_type(*argument) | ownership;
    }
    *argument = iree_vm_variant_empty();
  }
  iree_vm_function_ref_t* function_arguments =
      (iree_vm_function_ref_t*)packet->function_arguments.direct;
  for (uint16_t i = 0; i < callable_abi->argument_counts.function_count; ++i) {
    iree_vm_variant_t* argument = iree_vm_invocation_variant_slot(
        arguments.data, callable_abi->function_arguments[i].variant_offset);
    function_arguments[i].program_bits = argument->payload;
    function_arguments[i].target_bits =
        argument->metadata & ~(uint64_t)IREE_VM_VARIANT_TAG_MASK;
    *argument = iree_vm_variant_empty();
  }
}

iree_vm_call_packet_t iree_vm_invocation_commit_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process,
    const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback,
    bool has_external_borrowed_arguments) {
  const iree_vm_call_packet_t packet = iree_vm_invocation_make_root_packet(
      iree_vm_invocation_stack_base(invocation), callable_abi);
  iree_vm_invocation_initialize_root_banks(callable_abi, &packet);
  iree_vm_invocation_stage_arguments(callable_abi, arguments, &packet);
  invocation->process = process;
  invocation->root_callable_abi = callable_abi;
  invocation->stack_cursor = iree_vm_invocation_stack_base(invocation) +
                             callable_abi->root_layout.storage_size;
  invocation->top_frame = NULL;
  invocation->callback_context = NULL;
  invocation->wake_callback = wake_callback;
  invocation->operation = operation;
  invocation->has_external_borrowed_arguments = has_external_borrowed_arguments;
  invocation->state = IREE_VM_INVOCATION_STATE_RUNNING;
  iree_atomic_store(&invocation->cancel_reason, IREE_VM_CANCEL_REASON_NONE,
                    iree_memory_order_release);
  return packet;
}

//===----------------------------------------------------------------------===//
// Module Dispatch
//===----------------------------------------------------------------------===//

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_invocation_validate_dispatch_outcome_slow(
    const iree_vm_invocation_t* invocation, uint8_t* checkpoint_cursor,
    const iree_vm_frame_t* checkpoint_frame, bool may_yield,
    const iree_vm_call_request_t* call_request,
    iree_vm_execution_outcome_t outcome) {
  const bool is_at_checkpoint = invocation->stack_cursor == checkpoint_cursor &&
                                invocation->top_frame == checkpoint_frame;
  const bool has_continuation = invocation->stack_cursor > checkpoint_cursor &&
                                invocation->top_frame != checkpoint_frame;
  const bool has_call_request = call_request->linked_module != NULL;
  if (!is_at_checkpoint && !has_continuation) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "module callback corrupted its frame boundary");
  }
  if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    if (has_call_request) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "completed module callback left a child call request");
    }
    if (!is_at_checkpoint) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "completed module callback did not restore its frame boundary");
    }
    return iree_ok_status();
  }
  if (outcome != IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "module callback returned an invalid outcome");
  }
  if (has_call_request) return iree_ok_status();
  if (!has_continuation) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "suspended module callback did not preserve a continuation");
  }
  if (!may_yield) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "module callback suspended through a non-yielding contract");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_invocation_dispatch_start(
    iree_vm_invocation_t* invocation,
    const iree_vm_linked_module_t* linked_module, uint16_t function_ordinal,
    bool may_yield, const iree_vm_call_packet_t* call,
    iree_vm_call_request_t* call_request,
    iree_vm_execution_outcome_t* out_outcome) {
  uint8_t* checkpoint_cursor = invocation->stack_cursor;
  iree_vm_frame_t* checkpoint_frame = invocation->top_frame;
  call_request->linked_module = NULL;
  const iree_byte_span_t process_storage =
      iree_vm_process_module_state(invocation->process, linked_module);
  const iree_vm_module_function_start_params_t params = {
      .execution =
          {
              .invocation = invocation,
              .linked_module = linked_module,
              .process_storage = process_storage.data,
          },
      .function_ordinal = function_ordinal,
      .call = *call,
  };
  const iree_vm_callback_context_t callback_context = {
      .call_request = call_request,
      .may_yield = may_yield,
  };
  invocation->callback_context = &callback_context;
  iree_vm_execution_outcome_t module_outcome = UINT32_MAX;
  iree_status_t status = linked_module->module->vtable->function_start(
      linked_module->module, &params, &module_outcome);
  invocation->callback_context = NULL;

  const bool has_call_request = call_request->linked_module != NULL;
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(module_outcome != IREE_VM_EXECUTION_OUTCOME_COMPLETED ||
                    invocation->stack_cursor != checkpoint_cursor ||
                    invocation->top_frame != checkpoint_frame ||
                    has_call_request)) {
    status = iree_vm_invocation_validate_dispatch_outcome_slow(
        invocation, checkpoint_cursor, checkpoint_frame, may_yield,
        call_request, module_outcome);
  }
  if (!iree_status_is_ok(status)) {
    call_request->linked_module = NULL;
    iree_vm_invocation_unwind_to(invocation, checkpoint_cursor);
    return status;
  }
  *out_outcome = module_outcome;
  return iree_ok_status();
}

static iree_status_t iree_vm_invocation_dispatch_resume(
    iree_vm_invocation_t* invocation, iree_vm_frame_t* frame,
    iree_vm_call_request_t* call_request,
    iree_vm_execution_outcome_t* out_outcome) {
  uint8_t* checkpoint_cursor = frame->allocation_begin;
  iree_vm_frame_t* checkpoint_frame = frame->parent;
  const iree_vm_linked_module_t* linked_module = frame->linked_module;
  const bool may_yield = frame->may_yield;
  call_request->linked_module = NULL;
  const iree_byte_span_t process_storage =
      iree_vm_process_module_state(invocation->process, linked_module);
  const iree_vm_module_function_resume_params_t params = {
      .execution =
          {
              .invocation = invocation,
              .linked_module = linked_module,
              .process_storage = process_storage.data,
          },
      .frame = frame,
  };
  const iree_vm_callback_context_t callback_context = {
      .call_request = call_request,
      .may_yield = may_yield,
  };
  invocation->callback_context = &callback_context;
  iree_vm_execution_outcome_t module_outcome = UINT32_MAX;
  iree_status_t status = linked_module->module->vtable->function_resume(
      linked_module->module, &params, &module_outcome);
  invocation->callback_context = NULL;

  const bool has_call_request = call_request->linked_module != NULL;
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(module_outcome != IREE_VM_EXECUTION_OUTCOME_COMPLETED ||
                    invocation->stack_cursor != checkpoint_cursor ||
                    invocation->top_frame != checkpoint_frame ||
                    has_call_request)) {
    status = iree_vm_invocation_validate_dispatch_outcome_slow(
        invocation, checkpoint_cursor, checkpoint_frame, may_yield,
        call_request, module_outcome);
  }
  if (!iree_status_is_ok(status)) {
    call_request->linked_module = NULL;
    iree_vm_invocation_unwind_to(invocation, checkpoint_cursor);
    return status;
  }
  *out_outcome = module_outcome;
  return iree_ok_status();
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_invocation_publish_suspension(
    iree_vm_invocation_t* invocation,
    iree_vm_execution_outcome_t* out_outcome) {
  if (invocation->has_external_borrowed_arguments) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "invocation suspended while borrowed root refs were live");
  }
  invocation->state = IREE_VM_INVOCATION_STATE_SUSPENDED;
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  return iree_ok_status();
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_invocation_drive_continuations(
    iree_vm_invocation_t* invocation, iree_vm_call_request_t* call_request,
    iree_vm_execution_outcome_t* out_outcome) {
  while (true) {
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    iree_status_t status = iree_ok_status();
    if (call_request->linked_module) {
      const iree_vm_linked_module_t* linked_module =
          call_request->linked_module;
      const uint16_t function_ordinal = call_request->function_ordinal;
      const bool may_yield = call_request->may_yield;
      status = iree_vm_invocation_dispatch_start(
          invocation, linked_module, function_ordinal, may_yield,
          &call_request->packet, call_request, &outcome);
    } else if (invocation->top_frame) {
      status = iree_vm_invocation_dispatch_resume(
          invocation, invocation->top_frame, call_request, &outcome);
    } else {
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
      return iree_ok_status();
    }
    if (!iree_status_is_ok(status)) return status;
    if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED ||
        call_request->linked_module) {
      continue;
    }
    return iree_vm_invocation_publish_suspension(invocation, out_outcome);
  }
}

iree_status_t iree_vm_invocation_drive_start(
    iree_vm_invocation_t* invocation, uint64_t target_bits,
    const iree_vm_call_packet_t* root_packet,
    iree_vm_execution_outcome_t* out_outcome) {
  const uint16_t module_ordinal =
      iree_vm_program_target_module_ordinal(target_bits);
  const uint16_t function_ordinal =
      iree_vm_program_target_function_ordinal(target_bits);
  const iree_vm_linked_module_t* linked_module =
      &invocation->process->program->linked_modules[module_ordinal];
  const iree_fpu_state_t fpu_state =
      iree_fpu_state_push(iree_vm_invocation_fpu_state_flags);
  iree_vm_call_request_t call_request;
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_dispatch_start(
      invocation, linked_module, function_ordinal,
      iree_vm_program_target_may_yield(target_bits), root_packet, &call_request,
      &outcome);
  if (iree_status_is_ok(status)) {
    if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
      *out_outcome = outcome;
    } else if (call_request.linked_module) {
      status = iree_vm_invocation_drive_continuations(invocation, &call_request,
                                                      out_outcome);
    } else {
      status = iree_vm_invocation_publish_suspension(invocation, out_outcome);
    }
  }
  iree_fpu_state_pop(fpu_state);
  return status;
}

iree_status_t iree_vm_invocation_drive_resume(
    iree_vm_invocation_t* invocation,
    iree_vm_execution_outcome_t* out_outcome) {
  IREE_ASSERT(invocation->top_frame != NULL);
  invocation->state = IREE_VM_INVOCATION_STATE_RUNNING;
  const iree_fpu_state_t fpu_state =
      iree_fpu_state_push(iree_vm_invocation_fpu_state_flags);
  iree_vm_call_request_t call_request;
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_dispatch_resume(
      invocation, invocation->top_frame, &call_request, &outcome);
  if (iree_status_is_ok(status)) {
    if (call_request.linked_module ||
        (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED &&
         invocation->top_frame)) {
      status = iree_vm_invocation_drive_continuations(invocation, &call_request,
                                                      out_outcome);
    } else if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
      *out_outcome = outcome;
    } else {
      status = iree_vm_invocation_publish_suspension(invocation, out_outcome);
    }
  }
  iree_fpu_state_pop(fpu_state);
  return status;
}

//===----------------------------------------------------------------------===//
// Root Completion And Cancellation
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_invocation_validate_root_results(
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* packet) {
  const iree_vm_program_callable_abi_t* callable_abi =
      invocation->root_callable_abi;
  const iree_vm_program_t* program = invocation->process->program;
  const iree_vm_ref_t* ref_results = packet->ref_results.direct;
  for (uint16_t i = 0; i < callable_abi->result_counts.ref_count; ++i) {
    const iree_vm_ref_t ref = ref_results[i];
    if ((!ref.object && ref.type_and_state != 0) ||
        (ref.object &&
         (iree_vm_ref_type(ref) != callable_abi->ref_results[i].type ||
          (ref.type_and_state & IREE_VM_REF_STATE_MASK) >
              IREE_VM_REF_STATE_BORROWED))) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "module returned an invalid ref result");
    }
  }
  const iree_vm_function_ref_t* function_results =
      packet->function_results.direct;
  for (uint16_t i = 0; i < callable_abi->result_counts.function_count; ++i) {
    if (!iree_vm_program_function_ref_matches_mapping(
            program, function_results[i],
            callable_abi->function_results[i].callable_mapping)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "module returned an incompatible function result");
    }
  }
  return iree_ok_status();
}

iree_vm_cancel_reason_t iree_vm_invocation_claim_completion(
    iree_vm_invocation_t* invocation) {
  int32_t expected = IREE_VM_CANCEL_REASON_NONE;
  if (iree_atomic_compare_exchange_strong(&invocation->cancel_reason, &expected,
                                          IREE_VM_INVOCATION_CANCEL_REASON_IDLE,
                                          iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
    return IREE_VM_CANCEL_REASON_NONE;
  }
  iree_atomic_store(&invocation->cancel_reason,
                    IREE_VM_INVOCATION_CANCEL_REASON_IDLE,
                    iree_memory_order_release);
  return (iree_vm_cancel_reason_t)expected;
}

static void iree_vm_invocation_publish_root_results(
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* packet,
    iree_vm_variant_span_t results) {
  const iree_vm_program_callable_abi_t* callable_abi =
      invocation->root_callable_abi;
  const uint64_t* value_results = packet->value_results.direct;
  for (uint16_t i = 0; i < callable_abi->result_counts.value_count; ++i) {
    const iree_vm_program_scalar_field_abi_t field =
        callable_abi->value_results[i];
    *iree_vm_invocation_variant_slot(results.data, field.variant_offset) =
        (iree_vm_variant_t){
            value_results[i] & field.payload_mask,
            field.variant_metadata,
        };
  }
  iree_vm_ref_t* ref_results = packet->ref_results.direct;
  for (uint16_t i = 0; i < callable_abi->result_counts.ref_count; ++i) {
    *iree_vm_invocation_variant_slot(
        results.data, callable_abi->ref_results[i].variant_offset) =
        iree_vm_variant_from_ref_move(&ref_results[i]);
  }
  const iree_vm_function_ref_t* function_results =
      packet->function_results.direct;
  for (uint16_t i = 0; i < callable_abi->result_counts.function_count; ++i) {
    *iree_vm_invocation_variant_slot(
        results.data, callable_abi->function_results[i].variant_offset) =
        iree_vm_variant_from_function_ref(function_results[i]);
  }
}

void iree_vm_invocation_finish(iree_vm_invocation_t* invocation) {
  const iree_vm_call_packet_t packet =
      iree_vm_invocation_root_packet(invocation);
  const iree_vm_program_callable_abi_t* callable_abi =
      invocation->root_callable_abi;
  for (uint16_t i = 0; i < callable_abi->argument_counts.ref_count; ++i) {
    iree_vm_ref_reset(&packet.ref_arguments.direct[i]);
  }
  for (uint16_t i = 0; i < callable_abi->result_counts.ref_count; ++i) {
    iree_vm_ref_reset(&packet.ref_results.direct[i]);
  }
  invocation->callback_context = NULL;
  invocation->top_frame = NULL;
  invocation->stack_cursor = iree_vm_invocation_stack_base(invocation);
  invocation->operation = IREE_VM_INVOCATION_OPERATION_NONE;
  invocation->state = IREE_VM_INVOCATION_STATE_IDLE;
}

void iree_vm_invocation_abort(iree_vm_invocation_t* invocation) {
  iree_atomic_store(&invocation->cancel_reason,
                    IREE_VM_INVOCATION_CANCEL_REASON_IDLE,
                    iree_memory_order_release);
  invocation->callback_context = NULL;
  iree_vm_invocation_unwind_to(invocation,
                               iree_vm_invocation_stack_base(invocation));
  iree_vm_invocation_finish(invocation);
}

iree_status_t iree_vm_invocation_cancel_status(
    iree_vm_cancel_reason_t cancel_reason) {
  if (cancel_reason == IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED) {
    return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                            "VM invocation deadline exceeded");
  }
  return iree_make_status(IREE_STATUS_CANCELLED, "VM invocation was cancelled");
}

IREE_API_EXPORT bool iree_vm_invocation_request_cancel(
    iree_vm_invocation_t* invocation, iree_vm_cancel_reason_t reason) {
  if (!invocation || (reason != IREE_VM_CANCEL_REASON_CANCELLED &&
                      reason != IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED)) {
    return false;
  }
  int32_t expected = IREE_VM_CANCEL_REASON_NONE;
  if (!iree_atomic_compare_exchange_strong(
          &invocation->cancel_reason, &expected, (int32_t)reason,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return false;
  }
  const iree_vm_invocation_wake_callback_t wake_callback =
      invocation->wake_callback;
  if (wake_callback.fn) wake_callback.fn(wake_callback.user_data);
  return true;
}

//===----------------------------------------------------------------------===//
// Public Function Calls
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_invocation_complete_call(
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* root_packet,
    iree_vm_variant_span_t results, iree_vm_execution_outcome_t* out_outcome) {
  iree_status_t status =
      iree_vm_invocation_validate_root_results(invocation, root_packet);
  if (iree_status_is_ok(status)) {
    const iree_vm_cancel_reason_t cancel_reason =
        iree_vm_invocation_claim_completion(invocation);
    if (cancel_reason != IREE_VM_CANCEL_REASON_NONE) {
      status = iree_vm_invocation_cancel_status(cancel_reason);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_vm_invocation_publish_root_results(invocation, root_packet, results);
    iree_vm_invocation_finish(invocation);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  } else {
    iree_vm_invocation_abort(invocation);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_start(
    iree_vm_invocation_t* invocation, iree_vm_function_t function,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_vm_execution_outcome_t* out_outcome) {
  IREE_RETURN_IF_ERROR(iree_vm_invocation_validate_boundary(
      invocation, arguments, results,
      iree_make_byte_span(out_outcome, sizeof(*out_outcome))));
  if (iree_vm_function_is_null(function)) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation function is invalid");
  }
  iree_vm_process_t* process =
      (iree_vm_process_t*)(uintptr_t)function.process_bits;
  const iree_vm_program_t* program = process->program;
  const uint32_t callable_token =
      iree_vm_program_target_callable_token(function.target_bits);
  const iree_vm_program_callable_abi_t* callable_abi =
      &program->callable_abis[callable_token - 1];
  bool has_external_borrowed_arguments = false;
  iree_status_t status = iree_vm_invocation_preflight_root(
      invocation, program, function.target_bits, callable_abi, arguments,
      results, &has_external_borrowed_arguments);
  if (!iree_status_is_ok(status)) return status;
  const iree_vm_call_packet_t root_packet = iree_vm_invocation_commit_root(
      invocation, IREE_VM_INVOCATION_OPERATION_CALL, process, callable_abi,
      arguments, wake_callback, has_external_borrowed_arguments);

  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  status = iree_vm_invocation_drive_start(invocation, function.target_bits,
                                          &root_packet, &outcome);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_abort(invocation);
    return status;
  }
  if (outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    *out_outcome = outcome;
    return iree_ok_status();
  }
  return iree_vm_invocation_complete_call(invocation, &root_packet, results,
                                          out_outcome);
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_resume(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results,
    iree_vm_execution_outcome_t* out_outcome) {
  if (!invocation || !out_outcome ||
      invocation->state != IREE_VM_INVOCATION_STATE_SUSPENDED ||
      invocation->operation != IREE_VM_INVOCATION_OPERATION_CALL ||
      results.count != iree_vm_program_callable_abi_result_count(
                           invocation->root_callable_abi) ||
      (results.count && !results.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid invocation resume boundary");
  }
  IREE_RETURN_IF_ERROR(iree_vm_invocation_validate_boundary(
      invocation, iree_vm_variant_span_empty(), results,
      iree_make_byte_span(out_outcome, sizeof(*out_outcome))));

  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_drive_resume(invocation, &outcome);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_abort(invocation);
    return status;
  }
  if (outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    *out_outcome = outcome;
    return iree_ok_status();
  }
  const iree_vm_call_packet_t root_packet =
      iree_vm_invocation_root_packet(invocation);
  return iree_vm_invocation_complete_call(invocation, &root_packet, results,
                                          out_outcome);
}
