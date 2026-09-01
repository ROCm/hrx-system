// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/invocation.h"

#include <string.h>

#include "iree/base/internal/fpu_state.h"
#include "iree/vm/invocation_storage.h"
#include "iree/vm/program_storage.h"

static const iree_fpu_state_flags_t iree_vm_invocation_fpu_state_flags =
    IREE_FPU_STATE_FLAG_MASK_EXCEPTIONS | IREE_FPU_STATE_FLAG_ROUND_TO_NEAREST;

// Allocation-only owner prefix preceding max-aligned invocation storage.
typedef struct iree_alignas(iree_max_align_t) iree_vm_invocation_allocation_t {
  // Allocator owning the complete prefix and invocation allocation.
  iree_allocator_t host_allocator;
} iree_vm_invocation_allocation_t;

static_assert(sizeof(iree_vm_invocation_allocation_t) % iree_max_align_t == 0,
              "invocation allocation prefix must preserve max alignment");

// Driver-local child call requested by one module callback.
typedef struct iree_vm_call_request_t {
  // Resolved child module, or null when no request is pending.
  const iree_vm_linked_module_t* linked_module;
  // Stable physical banks copied from the requesting callback.
  iree_vm_call_packet_t packet;
  // Module-local child function ordinal.
  uint16_t function_ordinal;
  // True when the child is permitted to yield.
  bool may_yield;
} iree_vm_call_request_t;

// Native-stack context valid only while one module callback is executing.
struct iree_vm_callback_context_t {
  // Linked module currently executing.
  const iree_vm_linked_module_t* linked_module;
  // Driver-local child request populated by callback call helpers.
  iree_vm_call_request_t* call_request;
  // Module-local function currently executing.
  uint16_t function_ordinal;
  // True when the current function is permitted to yield.
  bool may_yield;
};

//===----------------------------------------------------------------------===//
// Invocation Storage
//===----------------------------------------------------------------------===//

bool iree_vm_invocation_is_idle(const iree_vm_invocation_t* invocation) {
  return invocation && invocation->state == IREE_VM_INVOCATION_STATE_IDLE;
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
  if (!iree_vm_invocation_is_idle(invocation) || invocation->is_allocated) {
    return;
  }
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
  if (!iree_vm_invocation_is_idle(invocation) || !invocation->is_allocated) {
    return;
  }
  iree_vm_invocation_allocation_t* allocation =
      (iree_vm_invocation_allocation_t*)(invocation->storage_begin -
                                         sizeof(*allocation));
  const iree_allocator_t host_allocator = allocation->host_allocator;
  memset(invocation, 0, sizeof(*invocation));
  iree_allocator_free(host_allocator, allocation);
}

//===----------------------------------------------------------------------===//
// Composite Frame Stack
//===----------------------------------------------------------------------===//

static void iree_vm_invocation_pop_top_frame(iree_vm_invocation_t* invocation) {
  iree_vm_frame_t* frame = invocation->top_frame;
  if (frame->cleanup) frame->cleanup(frame);
  invocation->top_frame = frame->parent;
  invocation->stack_cursor = frame->allocation_begin;
}

static void iree_vm_invocation_unwind_to(iree_vm_invocation_t* invocation,
                                         uint8_t* stack_cursor) {
  while (invocation->top_frame && invocation->stack_cursor > stack_cursor) {
    iree_vm_invocation_pop_top_frame(invocation);
  }
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_push_frame(
    const iree_vm_module_function_start_params_t* start_params,
    iree_vm_frame_layout_t layout, iree_vm_frame_cleanup_fn_t cleanup,
    iree_vm_frame_t** out_frame) {
  if (!start_params || !out_frame || !start_params->execution.invocation ||
      !start_params->execution.linked_module || layout.storage_alignment == 0 ||
      !iree_host_size_is_power_of_two(layout.storage_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid invocation frame request");
  }
  iree_vm_invocation_t* invocation = start_params->execution.invocation;
  const iree_vm_callback_context_t* callback_context =
      invocation->callback_context;
  if (invocation->state != IREE_VM_INVOCATION_STATE_RUNNING ||
      !callback_context ||
      callback_context->linked_module !=
          start_params->execution.linked_module ||
      callback_context->function_ordinal != start_params->function_ordinal) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "frame request is outside its module callback");
  }

  uintptr_t header_address = 0;
  if (!iree_vm_invocation_align_address((uintptr_t)invocation->stack_cursor,
                                        iree_alignof(iree_vm_frame_t),
                                        &header_address) ||
      header_address > (uintptr_t)invocation->storage_end ||
      (uintptr_t)invocation->storage_end - header_address <
          sizeof(iree_vm_frame_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "invocation frame header exceeds capacity");
  }
  uintptr_t payload_address = 0;
  if (!iree_vm_invocation_align_address(
          header_address + sizeof(iree_vm_frame_t), layout.storage_alignment,
          &payload_address) ||
      payload_address > (uintptr_t)invocation->storage_end ||
      layout.storage_size >
          (uintptr_t)invocation->storage_end - payload_address) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "invocation frame payload exceeds capacity");
  }

  iree_vm_frame_t* frame = (iree_vm_frame_t*)header_address;
  frame->allocation_begin = invocation->stack_cursor;
  frame->storage = (void*)payload_address;
  frame->parent = invocation->top_frame;
  frame->linked_module = start_params->execution.linked_module;
  frame->cleanup = cleanup;
  frame->function_ordinal = start_params->function_ordinal;
  frame->may_yield = callback_context->may_yield;
  invocation->stack_cursor = (uint8_t*)payload_address + layout.storage_size;
  invocation->top_frame = frame;
  *out_frame = frame;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_invocation_pop_frame(
    iree_vm_invocation_t* invocation, iree_vm_frame_t* frame) {
  if (IREE_UNLIKELY(!invocation || invocation->top_frame != frame ||
                    invocation->state != IREE_VM_INVOCATION_STATE_RUNNING)) {
    IREE_ASSERT(false && "only the running top invocation frame can be popped");
    return;
  }
  iree_vm_invocation_pop_top_frame(invocation);
}

IREE_API_EXPORT void* iree_vm_frame_storage(iree_vm_frame_t* frame) {
  return frame->storage;
}

IREE_API_EXPORT uint16_t
iree_vm_frame_function_ordinal(const iree_vm_frame_t* frame) {
  return frame->function_ordinal;
}

//===----------------------------------------------------------------------===//
// Root Call Staging
//===----------------------------------------------------------------------===//

static bool iree_vm_invocation_nonempty_ranges_overlap(uintptr_t lhs_begin,
                                                       uintptr_t lhs_end,
                                                       uintptr_t rhs_begin,
                                                       uintptr_t rhs_end) {
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

iree_status_t iree_vm_invocation_validate_boundary(
    const iree_vm_invocation_t* invocation, iree_vm_variant_span_t arguments,
    iree_vm_variant_span_t results, iree_byte_span_t outcome_storage) {
  if (IREE_UNLIKELY(!invocation || (arguments.count != 0 && !arguments.data) ||
                    (results.count != 0 && !results.data) ||
                    !outcome_storage.data ||
                    outcome_storage.data_length == 0)) {
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
      argument_size != 0 &&
      ((result_size != 0 &&
        iree_vm_invocation_nonempty_ranges_overlap(argument_begin, argument_end,
                                                   result_begin, result_end)) ||
       iree_vm_invocation_nonempty_ranges_overlap(
           argument_begin, argument_end, invocation_begin, invocation_end) ||
       iree_vm_invocation_nonempty_ranges_overlap(argument_begin, argument_end,
                                                  outcome_begin, outcome_end));
  const bool result_overlaps =
      result_size != 0 &&
      (iree_vm_invocation_nonempty_ranges_overlap(
           result_begin, result_end, invocation_begin, invocation_end) ||
       iree_vm_invocation_nonempty_ranges_overlap(result_begin, result_end,
                                                  outcome_begin, outcome_end));
  const bool outcome_overlaps = iree_vm_invocation_nonempty_ranges_overlap(
      outcome_begin, outcome_end, invocation_begin, invocation_end);
  if (IREE_UNLIKELY(argument_overlaps || result_overlaps || outcome_overlaps)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation storage spans overlap");
  }
  return iree_ok_status();
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

static void iree_vm_invocation_consume_arguments(
    iree_vm_variant_span_t arguments) {
  if (arguments.data) iree_vm_variant_span_reset(arguments);
}

static iree_vm_call_packet_t iree_vm_invocation_make_root_packet(
    uint8_t* storage, const iree_vm_program_root_layout_t* layout) {
  const iree_vm_call_packet_t packet = {
      .value_arguments =
          {(uint64_t*)(storage + layout->value_arguments.direct_offset),
           (uint64_t*)(storage + layout->value_arguments.overflow_offset)},
      .ref_arguments =
          {(iree_vm_ref_t*)(storage + layout->ref_arguments.direct_offset),
           (iree_vm_ref_t*)(storage + layout->ref_arguments.overflow_offset)},
      .value_results =
          {(uint64_t*)(storage + layout->value_results.direct_offset),
           (uint64_t*)(storage + layout->value_results.overflow_offset)},
      .ref_results =
          {(iree_vm_ref_t*)(storage + layout->ref_results.direct_offset),
           (iree_vm_ref_t*)(storage + layout->ref_results.overflow_offset)},
      .function_arguments =
          {(iree_vm_function_ref_t*)(storage +
                                     layout->function_arguments.direct_offset),
           (iree_vm_function_ref_t*)(storage + layout->function_arguments
                                                   .overflow_offset)},
      .function_results =
          {(iree_vm_function_ref_t*)(storage +
                                     layout->function_results.direct_offset),
           (iree_vm_function_ref_t*)(storage +
                                     layout->function_results.overflow_offset)},
  };
  return packet;
}

static iree_vm_call_packet_t iree_vm_invocation_root_packet(
    iree_vm_invocation_t* invocation) {
  const iree_vm_program_callable_abi_t* callable_abi =
      invocation->root_call.callable_abi;
  return iree_vm_invocation_make_root_packet(
      iree_vm_invocation_stack_base(invocation), &callable_abi->root_layout);
}

static void iree_vm_invocation_initialize_root_banks(
    const iree_vm_program_callable_abi_t* callable_abi,
    const iree_vm_call_packet_t* packet) {
  if (callable_abi->argument_counts.ref_count) {
    memset(packet->ref_arguments.direct, 0,
           callable_abi->argument_counts.ref_count *
               sizeof(*packet->ref_arguments.direct));
  }
  if (callable_abi->result_counts.ref_count) {
    memset(packet->ref_results.direct, 0,
           callable_abi->result_counts.ref_count *
               sizeof(*packet->ref_results.direct));
  }
  if (callable_abi->result_counts.function_count) {
    memset(packet->function_results.direct, 0,
           callable_abi->result_counts.function_count *
               sizeof(*packet->function_results.direct));
  }
}

static void iree_vm_invocation_stage_arguments(
    const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments, iree_vm_call_packet_t* packet) {
  // Root banks are allocated contiguously. The direct/overflow split only
  // describes the provider-facing register ABI and is not needed while
  // marshalling the complete host boundary.
  uint64_t* value_arguments = (uint64_t*)packet->value_arguments.direct;
  for (uint16_t i = 0; i < callable_abi->argument_counts.value_count; ++i) {
    iree_vm_variant_t* argument = iree_vm_invocation_variant_slot(
        arguments.data, callable_abi->value_arguments[i].variant_offset);
    value_arguments[i] = argument->payload;
    *argument = iree_vm_variant_empty();
  }
  iree_vm_ref_t* ref_arguments = packet->ref_arguments.direct;
  for (uint16_t i = 0; i < callable_abi->argument_counts.ref_count; ++i) {
    iree_vm_variant_t* argument_slot = iree_vm_invocation_variant_slot(
        arguments.data, callable_abi->ref_arguments[i].variant_offset);
    const iree_vm_variant_t argument = *argument_slot;
    iree_vm_ref_t* slot = &ref_arguments[i];
    if (argument.payload) {
      const uintptr_t ownership =
          (argument.metadata & IREE_VM_VARIANT_TAG_MASK) ==
                  IREE_VM_VARIANT_TAG_BORROWED_REF
              ? IREE_VM_REF_STATE_BORROWED
              : IREE_VM_REF_STATE_OWNED;
      slot->object = (void*)(uintptr_t)argument.payload;
      slot->type_and_state =
          (uintptr_t)iree_vm_variant_ref_type(argument) | ownership;
    }
    *argument_slot = iree_vm_variant_empty();
  }
  iree_vm_function_ref_t* function_arguments =
      (iree_vm_function_ref_t*)packet->function_arguments.direct;
  for (uint16_t i = 0; i < callable_abi->argument_counts.function_count; ++i) {
    iree_vm_variant_t* argument_slot = iree_vm_invocation_variant_slot(
        arguments.data, callable_abi->function_arguments[i].variant_offset);
    const iree_vm_variant_t argument = *argument_slot;
    iree_vm_function_ref_t* slot = &function_arguments[i];
    slot->program_bits = argument.payload;
    slot->target_bits = argument.metadata & ~(uint64_t)IREE_VM_VARIANT_TAG_MASK;
    *argument_slot = iree_vm_variant_empty();
  }
}

static void iree_vm_invocation_commit_root_state(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_abi_t* callable_abi, uint8_t* stack_cursor,
    iree_vm_invocation_wake_callback_t wake_callback,
    bool has_external_borrowed_arguments) {
  invocation->process = process;
  invocation->root_call.callable_abi = callable_abi;
  invocation->root_call.target_bits = target_bits;
  invocation->stack_cursor = stack_cursor;
  invocation->top_frame = NULL;
  invocation->callback_context = NULL;
  invocation->wake_callback = wake_callback;
  invocation->operation = operation;
  invocation->has_external_borrowed_arguments = has_external_borrowed_arguments;
  invocation->state = IREE_VM_INVOCATION_STATE_RUNNING;
  iree_atomic_store(&invocation->cancel_reason, IREE_VM_CANCEL_REASON_NONE,
                    iree_memory_order_release);
}

iree_status_t iree_vm_invocation_preflight_root(
    iree_vm_invocation_t* invocation, const iree_vm_program_t* program,
    uint64_t target_bits, const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_root_preflight_t* out_preflight) {
  if (!iree_vm_invocation_is_idle(invocation) || !program || target_bits == 0 ||
      !callable_abi || !out_preflight) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid invocation root boundary");
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
  *out_preflight = (iree_vm_root_preflight_t){
      .root_storage_size = root_storage_size,
      .has_external_borrowed_arguments = has_external_borrowed_arguments,
  };
  return iree_ok_status();
}

iree_vm_call_packet_t iree_vm_invocation_commit_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_vm_root_preflight_t preflight) {
  iree_vm_call_packet_t packet = iree_vm_invocation_make_root_packet(
      iree_vm_invocation_stack_base(invocation), &callable_abi->root_layout);
  iree_vm_invocation_initialize_root_banks(callable_abi, &packet);
  iree_vm_invocation_stage_arguments(callable_abi, arguments, &packet);
  iree_vm_invocation_commit_root_state(
      invocation, operation, process, target_bits, callable_abi,
      iree_vm_invocation_stack_base(invocation) + preflight.root_storage_size,
      wake_callback, preflight.has_external_borrowed_arguments);
  return packet;
}

//===----------------------------------------------------------------------===//
// Module Dispatch
//===----------------------------------------------------------------------===//

static bool iree_vm_invocation_resolve_target(
    iree_vm_invocation_t* invocation, uint64_t target_bits,
    const iree_vm_linked_module_t** out_linked_module,
    uint16_t* out_function_ordinal) {
  if (!target_bits || (target_bits & 3u) != 0) return false;
  const iree_vm_program_t* program = invocation->process->program;
  const uint16_t module_ordinal =
      iree_vm_program_target_module_ordinal(target_bits);
  const uint16_t function_ordinal =
      iree_vm_program_target_function_ordinal(target_bits);
  if (module_ordinal >= program->linked_module_count) return false;
  const iree_vm_linked_module_t* linked_module =
      &program->linked_modules[module_ordinal];
  if (function_ordinal >=
      linked_module->module->descriptor->counts.function_count) {
    return false;
  }
  const uint32_t callable_token =
      iree_vm_program_target_callable_token(target_bits);
  if (!iree_vm_program_callable_token_is_valid(program, callable_token)) {
    return false;
  }
  *out_linked_module = linked_module;
  *out_function_ordinal = function_ordinal;
  return true;
}

static bool iree_vm_invocation_execution_is_current(
    const iree_vm_module_execution_t* execution) {
  if (!execution || !execution->invocation || !execution->linked_module) {
    return false;
  }
  const iree_vm_invocation_t* invocation = execution->invocation;
  const iree_vm_callback_context_t* callback_context =
      invocation->callback_context;
  if (invocation->state != IREE_VM_INVOCATION_STATE_RUNNING ||
      !callback_context ||
      callback_context->linked_module != execution->linked_module) {
    return false;
  }
  const iree_byte_span_t process_storage = iree_vm_process_module_state(
      invocation->process, execution->linked_module);
  return execution->process_storage == process_storage.data;
}

iree_status_t iree_vm_invocation_request_call(
    iree_vm_invocation_t* invocation,
    const iree_vm_linked_module_t* linked_module, uint16_t function_ordinal,
    bool may_yield, const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome) {
  if (!invocation || !linked_module || !call || !out_outcome ||
      invocation->state != IREE_VM_INVOCATION_STATE_RUNNING ||
      !invocation->callback_context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid child call request");
  }
  const iree_vm_callback_context_t* callback_context =
      invocation->callback_context;
  iree_vm_call_request_t* call_request = callback_context->call_request;
  if (call_request->linked_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module callback already requested a child call");
  }
  if (may_yield && !callback_context->may_yield) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "non-yielding module callback requested a yielding child");
  }
  *call_request = (iree_vm_call_request_t){
      .linked_module = linked_module,
      .packet = *call,
      .function_ordinal = function_ordinal,
      .may_yield = may_yield,
  };
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  return iree_ok_status();
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_invocation_validate_dispatch_outcome_slow(
    const iree_vm_invocation_t* invocation, uint8_t* checkpoint_cursor,
    const iree_vm_frame_t* checkpoint_frame, bool may_yield,
    bool has_call_request, iree_vm_execution_outcome_t outcome) {
  const bool is_at_checkpoint = invocation->stack_cursor == checkpoint_cursor &&
                                invocation->top_frame == checkpoint_frame;
  const bool has_continuation = invocation->stack_cursor > checkpoint_cursor &&
                                invocation->top_frame != checkpoint_frame;
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
  if (has_call_request) {
    return iree_ok_status();
  }
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
      .linked_module = linked_module,
      .call_request = call_request,
      .function_ordinal = function_ordinal,
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
        has_call_request, module_outcome);
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
      .linked_module = linked_module,
      .call_request = call_request,
      .function_ordinal = frame->function_ordinal,
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
        has_call_request, module_outcome);
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
        "invocation suspended while external borrowed refs were live");
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
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* root_packet,
    iree_vm_execution_outcome_t* out_outcome) {
  const uint16_t module_ordinal =
      iree_vm_program_target_module_ordinal(invocation->root_call.target_bits);
  const uint16_t function_ordinal = iree_vm_program_target_function_ordinal(
      invocation->root_call.target_bits);
  const iree_vm_linked_module_t* linked_module =
      &invocation->process->program->linked_modules[module_ordinal];
  const iree_fpu_state_t fpu_state =
      iree_fpu_state_push(iree_vm_invocation_fpu_state_flags);
  iree_vm_call_request_t call_request;
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_dispatch_start(
      invocation, linked_module, function_ordinal,
      iree_vm_program_target_may_yield(invocation->root_call.target_bits),
      root_packet, &call_request, &outcome);
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
  invocation->state = IREE_VM_INVOCATION_STATE_RUNNING;
  if (!invocation->top_frame) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "suspended invocation has no continuation");
  }
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
// Nested Module Calls
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_vm_invocation_call_local(const iree_vm_module_execution_t* execution,
                              iree_vm_module_local_function_t local_function,
                              const iree_vm_call_packet_t* call,
                              iree_vm_execution_outcome_t* out_outcome) {
  if (!iree_vm_invocation_execution_is_current(execution) || !call ||
      !out_outcome) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid local call boundary");
  }
  const iree_vm_module_descriptor_t* descriptor =
      execution->linked_module->module->descriptor;
  if (local_function.function_ordinal >= descriptor->counts.function_count ||
      local_function.callable_type_ordinal >=
          descriptor->counts.callable_type_count ||
      (local_function.flags & ~(iree_vm_module_function_flags_t)
                                  IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD) !=
          0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local function descriptor is invalid");
  }
  const uint32_t mapping =
      execution->invocation->process->program
          ->callable_mappings[execution->linked_module->callable_base +
                              local_function.callable_type_ordinal];
  const bool may_yield = iree_any_bit_set(
      local_function.flags, IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD);
  if (may_yield && !iree_vm_program_callable_may_yield(mapping)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local function behavior exceeds its callable contract");
  }
  return iree_vm_invocation_request_call(
      execution->invocation, execution->linked_module,
      local_function.function_ordinal, may_yield, call, out_outcome);
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_call_import(
    const iree_vm_module_execution_t* execution, uint16_t import_ordinal,
    const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome) {
  if (!iree_vm_invocation_execution_is_current(execution) || !call ||
      !out_outcome ||
      import_ordinal >=
          execution->linked_module->module->descriptor->counts.import_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid import call boundary");
  }
  const uint64_t target_bits =
      execution->linked_module->import_target_bits[import_ordinal];
  if (!target_bits) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "optional import is unresolved");
  }
  const iree_vm_linked_module_t* target_module = NULL;
  uint16_t function_ordinal = 0;
  if (!iree_vm_invocation_resolve_target(execution->invocation, target_bits,
                                         &target_module, &function_ordinal)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "resolved import target is invalid");
  }
  return iree_vm_invocation_request_call(
      execution->invocation, target_module, function_ordinal,
      iree_vm_program_target_may_yield(target_bits), call, out_outcome);
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_call_function_ref(
    const iree_vm_module_execution_t* execution,
    iree_vm_function_ref_t function_ref,
    uint16_t expected_callable_type_ordinal, const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome) {
  if (!iree_vm_invocation_execution_is_current(execution) || !call ||
      !out_outcome || iree_vm_function_ref_is_null(function_ref) ||
      expected_callable_type_ordinal >=
          execution->linked_module->module->descriptor->counts
              .callable_type_count ||
      !iree_vm_program_function_ref_matches(
          execution->invocation->process->program, function_ref,
          execution->linked_module, expected_callable_type_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid function-ref call boundary");
  }
  const iree_vm_linked_module_t* target_module = NULL;
  uint16_t function_ordinal = 0;
  if (!iree_vm_invocation_resolve_target(execution->invocation,
                                         function_ref.target_bits,
                                         &target_module, &function_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function-ref target is invalid");
  }
  return iree_vm_invocation_request_call(
      execution->invocation, target_module, function_ordinal,
      iree_vm_program_target_may_yield(function_ref.target_bits), call,
      out_outcome);
}

IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_local_function(
    const iree_vm_module_execution_t* execution,
    iree_vm_module_local_function_t local_function,
    iree_vm_function_ref_t* out_function_ref) {
  if (!iree_vm_invocation_execution_is_current(execution) ||
      !out_function_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid local function-ref boundary");
  }
  const iree_vm_module_descriptor_t* descriptor =
      execution->linked_module->module->descriptor;
  if (local_function.function_ordinal >= descriptor->counts.function_count ||
      local_function.callable_type_ordinal >=
          descriptor->counts.callable_type_count ||
      (local_function.flags & ~(iree_vm_module_function_flags_t)
                                  IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD) !=
          0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local function descriptor is invalid");
  }
  uint32_t mapping =
      execution->invocation->process->program
          ->callable_mappings[execution->linked_module->callable_base +
                              local_function.callable_type_ordinal];
  const bool may_yield = iree_any_bit_set(
      local_function.flags, IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD);
  if (may_yield && !iree_vm_program_callable_may_yield(mapping)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local function behavior exceeds its callable contract");
  }
  mapping &= ~IREE_VM_PROGRAM_CALLABLE_MAY_YIELD;
  if (may_yield) mapping |= IREE_VM_PROGRAM_CALLABLE_MAY_YIELD;
  const iree_host_size_t module_ordinal =
      (iree_host_size_t)(execution->linked_module -
                         execution->invocation->process->program
                             ->linked_modules);
  const iree_vm_function_ref_t function_ref = {
      (uint64_t)(uintptr_t)execution->invocation->process->program,
      iree_vm_program_pack_target_bits(
          (uint16_t)module_ordinal, local_function.function_ordinal, mapping),
  };
  *out_function_ref = function_ref;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_import(
    const iree_vm_module_execution_t* execution, uint16_t import_ordinal,
    iree_vm_function_ref_t* out_function_ref) {
  if (!iree_vm_invocation_execution_is_current(execution) ||
      !out_function_ref ||
      import_ordinal >=
          execution->linked_module->module->descriptor->counts.import_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid import function-ref boundary");
  }
  const iree_vm_function_ref_t function_ref = {
      (uint64_t)(uintptr_t)execution->invocation->process->program,
      execution->linked_module->import_target_bits[import_ordinal],
  };
  *out_function_ref =
      function_ref.target_bits ? function_ref : iree_vm_function_ref_null();
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Root Result Transaction And Cancellation
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_invocation_validate_root_results(
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* packet) {
  const iree_vm_program_callable_abi_t* callable_abi =
      invocation->root_call.callable_abi;
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
    const iree_vm_function_ref_t function_ref = function_results[i];
    if (!iree_vm_program_function_ref_matches_mapping(
            program, function_ref,
            callable_abi->function_results[i].callable_mapping)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "module returned an incompatible function result");
    }
  }
  return iree_ok_status();
}

bool iree_vm_invocation_try_claim_completion(
    iree_vm_invocation_t* invocation,
    iree_vm_cancel_reason_t* out_cancel_reason) {
  int32_t expected = IREE_VM_CANCEL_REASON_NONE;
  if (iree_atomic_compare_exchange_strong(&invocation->cancel_reason, &expected,
                                          IREE_VM_INVOCATION_CANCEL_REASON_IDLE,
                                          iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
    *out_cancel_reason = IREE_VM_CANCEL_REASON_NONE;
    return true;
  }
  *out_cancel_reason = (iree_vm_cancel_reason_t)expected;
  iree_atomic_store(&invocation->cancel_reason,
                    IREE_VM_INVOCATION_CANCEL_REASON_IDLE,
                    iree_memory_order_release);
  return false;
}

static IREE_ATTRIBUTE_NOINLINE void iree_vm_invocation_publish_root_results(
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* packet,
    iree_vm_variant_span_t results) {
  const iree_vm_program_callable_abi_t* callable_abi =
      invocation->root_call.callable_abi;
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
      invocation->root_call.callable_abi;
  const uint16_t argument_ref_count = callable_abi->argument_counts.ref_count;
  for (uint16_t i = 0; i < argument_ref_count; ++i) {
    iree_vm_ref_reset(&packet.ref_arguments.direct[i]);
  }
  const uint16_t result_ref_count = callable_abi->result_counts.ref_count;
  for (uint16_t i = 0; i < result_ref_count; ++i) {
    iree_vm_ref_reset(&packet.ref_results.direct[i]);
  }
  invocation->callback_context = NULL;
  invocation->top_frame = NULL;
  invocation->stack_cursor = iree_vm_invocation_stack_base(invocation);
  invocation->state = IREE_VM_INVOCATION_STATE_IDLE;
  invocation->operation = IREE_VM_INVOCATION_OPERATION_NONE;
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

IREE_API_EXPORT iree_vm_invocation_wake_callback_t
iree_vm_invocation_wake_callback(iree_vm_invocation_t* invocation) {
  return invocation->wake_callback;
}

IREE_API_EXPORT iree_vm_cancel_reason_t
iree_vm_invocation_cancel_reason(const iree_vm_invocation_t* invocation) {
  const int32_t reason =
      iree_atomic_load(&invocation->cancel_reason, iree_memory_order_acquire);
  return reason == IREE_VM_INVOCATION_CANCEL_REASON_IDLE
             ? IREE_VM_CANCEL_REASON_NONE
             : (iree_vm_cancel_reason_t)reason;
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
  if (wake_callback.function) {
    wake_callback.function(wake_callback.user_data);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Public Root Invocation
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_invocation_resolve_root_function(
    iree_vm_function_t function, iree_vm_process_t** out_process,
    const iree_vm_program_callable_abi_t** out_callable_abi) {
  if (iree_vm_function_is_null(function) || (function.target_bits & 3u) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation function is invalid");
  }
  iree_vm_process_t* process =
      (iree_vm_process_t*)(uintptr_t)function.process_bits;
  // Binding validated the VM-produced target against this process. A forged
  // process pointer cannot be made safe here, but the canonical token directly
  // indexes the ABI resolved while linking the program.
  const uint32_t callable_token =
      iree_vm_program_target_callable_token(function.target_bits);
  const iree_vm_program_callable_abi_t* callable_abi =
      iree_vm_program_resolve_callable_abi(process->program, callable_token);
  if (!callable_abi) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation callable token is invalid");
  }
  *out_process = process;
  *out_callable_abi = callable_abi;
  return iree_ok_status();
}

static iree_status_t iree_vm_invocation_complete_call(
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* root_packet,
    iree_vm_variant_span_t results, iree_vm_execution_outcome_t* out_outcome) {
  iree_status_t status =
      iree_vm_invocation_validate_root_results(invocation, root_packet);
  iree_vm_cancel_reason_t cancel_reason = IREE_VM_CANCEL_REASON_NONE;
  if (iree_status_is_ok(status) &&
      !iree_vm_invocation_try_claim_completion(invocation, &cancel_reason)) {
    status = iree_vm_invocation_cancel_status(cancel_reason);
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
  iree_vm_process_t* process = NULL;
  const iree_vm_program_callable_abi_t* callable_abi = NULL;
  iree_status_t status = iree_vm_invocation_resolve_root_function(
      function, &process, &callable_abi);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_consume_arguments(arguments);
    return status;
  }
  iree_vm_root_preflight_t preflight = {0};
  status = iree_vm_invocation_preflight_root(invocation, process->program,
                                             function.target_bits, callable_abi,
                                             arguments, results, &preflight);
  if (!iree_status_is_ok(status)) return status;
  const iree_vm_call_packet_t root_packet = iree_vm_invocation_commit_root(
      invocation, IREE_VM_INVOCATION_OPERATION_CALL, process,
      function.target_bits, callable_abi, arguments, wake_callback, preflight);

  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  status = iree_vm_invocation_drive_start(invocation, &root_packet, &outcome);
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
                           invocation->root_call.callable_abi) ||
      (results.count != 0 && !results.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid invocation resume boundary");
  }
  const iree_vm_variant_span_t no_arguments = iree_vm_variant_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_invocation_validate_boundary(
      invocation, no_arguments, results,
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
