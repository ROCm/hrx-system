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
  invocation->stack_base =
      (uint8_t*)invocation + iree_sizeof_struct(*invocation);
  invocation->stack_cursor = invocation->stack_base;
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
  IREE_ASSERT(!invocation->allocation_base);
  if (!iree_vm_invocation_is_idle(invocation) || invocation->allocation_base) {
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
  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, storage_size, &allocation));
  iree_vm_invocation_t* invocation = NULL;
  iree_status_t status = iree_vm_invocation_initialize(
      iree_make_byte_span(allocation, storage_size), &invocation);
  if (iree_status_is_ok(status)) {
    invocation->allocation_base = allocation;
    invocation->host_allocator = host_allocator;
    *out_invocation = invocation;
  } else {
    iree_allocator_free(host_allocator, allocation);
  }
  return status;
}

IREE_API_EXPORT void iree_vm_invocation_free(iree_vm_invocation_t* invocation) {
  if (!invocation) return;
  IREE_ASSERT(iree_vm_invocation_is_idle(invocation));
  IREE_ASSERT(invocation->allocation_base);
  if (!iree_vm_invocation_is_idle(invocation) || !invocation->allocation_base) {
    return;
  }
  void* allocation_base = invocation->allocation_base;
  const iree_allocator_t host_allocator = invocation->host_allocator;
  memset(invocation, 0, sizeof(*invocation));
  iree_allocator_free(host_allocator, allocation_base);
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
  if (invocation->state != IREE_VM_INVOCATION_STATE_RUNNING ||
      invocation->executing_linked_module !=
          start_params->execution.linked_module ||
      invocation->executing_function_ordinal !=
          start_params->function_ordinal) {
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
  frame->may_yield = invocation->executing_may_yield;
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

static uint32_t iree_vm_invocation_scalar_bit_width(
    iree_vm_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case IREE_VM_SCALAR_TYPE_I8:
    case IREE_VM_SCALAR_TYPE_F8E4M3FN:
    case IREE_VM_SCALAR_TYPE_F8E5M2:
      return 8;
    case IREE_VM_SCALAR_TYPE_I16:
    case IREE_VM_SCALAR_TYPE_F16:
    case IREE_VM_SCALAR_TYPE_BF16:
      return 16;
    case IREE_VM_SCALAR_TYPE_I32:
    case IREE_VM_SCALAR_TYPE_F32:
      return 32;
    default:
      return 64;
  }
}

static bool iree_vm_invocation_ranges_overlap(const void* lhs_data,
                                              iree_host_size_t lhs_size,
                                              const void* rhs_data,
                                              iree_host_size_t rhs_size) {
  if (lhs_size == 0 || rhs_size == 0) return false;
  const uintptr_t lhs_begin = (uintptr_t)lhs_data;
  const uintptr_t rhs_begin = (uintptr_t)rhs_data;
  if (lhs_size > UINTPTR_MAX - lhs_begin ||
      rhs_size > UINTPTR_MAX - rhs_begin) {
    return true;
  }
  return lhs_begin < rhs_begin + rhs_size && rhs_begin < lhs_begin + lhs_size;
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
  const iree_host_size_t invocation_size =
      (iree_host_size_t)(invocation->storage_end - invocation->storage_begin);
  if (IREE_UNLIKELY(
          iree_vm_invocation_ranges_overlap(arguments.data, argument_size,
                                            results.data, result_size) ||
          iree_vm_invocation_ranges_overlap(arguments.data, argument_size,
                                            invocation->storage_begin,
                                            invocation_size) ||
          iree_vm_invocation_ranges_overlap(results.data, result_size,
                                            invocation->storage_begin,
                                            invocation_size) ||
          iree_vm_invocation_ranges_overlap(outcome_storage.data,
                                            outcome_storage.data_length,
                                            arguments.data, argument_size) ||
          iree_vm_invocation_ranges_overlap(outcome_storage.data,
                                            outcome_storage.data_length,
                                            results.data, result_size) ||
          iree_vm_invocation_ranges_overlap(
              outcome_storage.data, outcome_storage.data_length,
              invocation->storage_begin, invocation_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation storage spans overlap");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_invocation_validate_arguments(
    const iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_linked_module_t* signature_module,
    iree_vm_module_signature_type_span_t types,
    iree_vm_program_bank_counts_t argument_counts,
    iree_vm_variant_span_t arguments,
    bool* out_has_external_borrowed_arguments) {
  if (IREE_UNLIKELY(arguments.count != types.count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation argument count mismatch");
  }

  // Scalar-only signatures are the dominant leaf-call shape. Validate their
  // exact tags and canonical payload bits without entering the mixed carrier
  // machinery required for refs and function values.
  if (argument_counts.value_count == types.count) {
    for (iree_host_size_t i = 0; i < types.count; ++i) {
      const iree_vm_scalar_type_t scalar_type =
          (iree_vm_scalar_type_t)types.data[i].kind;
      const iree_vm_variant_t argument = arguments.data[i];
      const uint64_t expected_metadata =
          ((uint64_t)scalar_type << 2) | IREE_VM_VARIANT_TAG_SCALAR;
      const uint32_t bit_width =
          iree_vm_invocation_scalar_bit_width(scalar_type);
      if (IREE_UNLIKELY(
              argument.metadata != expected_metadata ||
              (bit_width < 64 && (argument.payload >> bit_width) != 0))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invocation scalar argument is not canonical");
      }
    }
    *out_has_external_borrowed_arguments = false;
    return iree_ok_status();
  }

  bool has_external_borrowed_arguments = false;
  for (iree_host_size_t i = 0; i < types.count; ++i) {
    const iree_vm_module_signature_type_t type = types.data[i];
    const iree_vm_variant_t argument = arguments.data[i];
    if (type.kind > IREE_VM_SCALAR_TYPE_INVALID &&
        type.kind <= IREE_VM_SCALAR_TYPE_F64) {
      const uint64_t expected_metadata =
          ((uint64_t)type.kind << 2) | IREE_VM_VARIANT_TAG_SCALAR;
      if (argument.metadata != expected_metadata) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invocation scalar argument type mismatch");
      }
      const uint32_t bit_width = iree_vm_invocation_scalar_bit_width(type.kind);
      if (bit_width < 64 && (argument.payload >> bit_width) != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invocation scalar argument has noncanonical high bits");
      }
    } else if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      const iree_vm_ref_type_t expected_type =
          signature_module->module->descriptor->ref_types
              .data[type.type_ordinal];
      if (!iree_vm_variant_is_ref(argument) ||
          (argument.payload &&
           iree_vm_variant_ref_type(argument) != expected_type) ||
          (!argument.payload &&
           argument.metadata != IREE_VM_VARIANT_TAG_OWNED_REF)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invocation ref argument type mismatch");
      }
      has_external_borrowed_arguments |=
          argument.payload != 0 &&
          (argument.metadata & IREE_VM_VARIANT_TAG_MASK) ==
              IREE_VM_VARIANT_TAG_BORROWED_REF;
    } else {
      iree_vm_function_ref_t function_ref = iree_vm_function_ref_null();
      if (!iree_vm_variant_is_function_ref(argument)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invocation function argument type mismatch");
      }
      function_ref.program_bits = argument.payload;
      function_ref.target_bits =
          argument.metadata & ~(uint64_t)IREE_VM_VARIANT_TAG_MASK;
      if (!iree_vm_program_function_ref_matches(process->program, function_ref,
                                                signature_module,
                                                type.type_ordinal)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invocation function argument contract mismatch");
      }
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

static iree_status_t iree_vm_invocation_layout_root_call(
    iree_vm_invocation_t* invocation,
    iree_vm_program_bank_counts_t argument_counts,
    iree_vm_program_bank_counts_t result_counts,
    iree_vm_call_packet_t* out_packet, uint8_t** out_stack_cursor) {
  // Module validation caps each signature side at UINT16_MAX logical values.
  // All bank elements have power-of-two native alignment, the stack base is
  // max-aligned, and every bank byte length preserves that alignment. The
  // resulting two-side layout therefore fits host size without checked
  // arithmetic or per-bank padding on every invocation.
  iree_host_size_t total_size = 0;
  const iree_host_size_t value_arguments_offset = total_size;
  total_size += argument_counts.value_count * sizeof(uint64_t);
  const iree_host_size_t ref_arguments_offset = total_size;
  total_size += argument_counts.ref_count * sizeof(iree_vm_ref_t);
  const iree_host_size_t function_arguments_offset = total_size;
  total_size += argument_counts.function_count * sizeof(iree_vm_function_ref_t);
  const iree_host_size_t value_results_offset = total_size;
  total_size += result_counts.value_count * sizeof(uint64_t);
  const iree_host_size_t ref_results_offset = total_size;
  total_size += result_counts.ref_count * sizeof(iree_vm_ref_t);
  const iree_host_size_t function_results_offset = total_size;
  total_size += result_counts.function_count * sizeof(iree_vm_function_ref_t);
  if (total_size >
      (iree_host_size_t)(invocation->storage_end - invocation->stack_base)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "root call staging exceeds invocation capacity");
  }

  uint8_t* storage = invocation->stack_base;
  uint64_t* value_arguments =
      argument_counts.value_count
          ? (uint64_t*)(storage + value_arguments_offset)
          : NULL;
  iree_vm_ref_t* ref_arguments =
      argument_counts.ref_count
          ? (iree_vm_ref_t*)(storage + ref_arguments_offset)
          : NULL;
  iree_vm_function_ref_t* function_arguments =
      argument_counts.function_count
          ? (iree_vm_function_ref_t*)(storage + function_arguments_offset)
          : NULL;
  uint64_t* value_results = result_counts.value_count
                                ? (uint64_t*)(storage + value_results_offset)
                                : NULL;
  iree_vm_ref_t* ref_results =
      result_counts.ref_count ? (iree_vm_ref_t*)(storage + ref_results_offset)
                              : NULL;
  iree_vm_function_ref_t* function_results =
      result_counts.function_count
          ? (iree_vm_function_ref_t*)(storage + function_results_offset)
          : NULL;
  if (ref_arguments) {
    memset(ref_arguments, 0,
           argument_counts.ref_count * sizeof(*ref_arguments));
  }
  if (ref_results) {
    memset(ref_results, 0, result_counts.ref_count * sizeof(*ref_results));
  }
  if (function_results) {
    memset(function_results, 0,
           result_counts.function_count * sizeof(*function_results));
  }
  const iree_vm_call_packet_t packet = {
      .value_arguments = {value_arguments, argument_counts.value_count > 16
                                               ? value_arguments + 16
                                               : NULL},
      .ref_arguments = {ref_arguments, argument_counts.ref_count > 16
                                           ? ref_arguments + 16
                                           : NULL},
      .value_results = {value_results, result_counts.value_count > 16
                                           ? value_results + 16
                                           : NULL},
      .ref_results = {ref_results,
                      result_counts.ref_count > 16 ? ref_results + 16 : NULL},
      .function_arguments = {function_arguments,
                             argument_counts.function_count > 16
                                 ? function_arguments + 16
                                 : NULL},
      .function_results = {function_results, result_counts.function_count > 16
                                                 ? function_results + 16
                                                 : NULL},
  };
  *out_packet = packet;
  *out_stack_cursor = storage + total_size;
  return iree_ok_status();
}

static iree_vm_ref_t* iree_vm_call_ref_argument_slot(
    const iree_vm_call_packet_t* packet, uint16_t ordinal) {
  return ordinal < 16 ? &packet->ref_arguments.direct[ordinal]
                      : &packet->ref_arguments.overflow[ordinal - 16];
}

static uint64_t iree_vm_call_value_result_load(
    const iree_vm_call_packet_t* packet, uint16_t ordinal) {
  return ordinal < 16 ? packet->value_results.direct[ordinal]
                      : packet->value_results.overflow[ordinal - 16];
}

static iree_vm_ref_t* iree_vm_call_ref_result_slot(
    const iree_vm_call_packet_t* packet, uint16_t ordinal) {
  return ordinal < 16 ? &packet->ref_results.direct[ordinal]
                      : &packet->ref_results.overflow[ordinal - 16];
}

static iree_vm_function_ref_t* iree_vm_call_function_result_slot(
    const iree_vm_call_packet_t* packet, uint16_t ordinal) {
  return ordinal < 16 ? &packet->function_results.direct[ordinal]
                      : &packet->function_results.overflow[ordinal - 16];
}

static void iree_vm_invocation_stage_arguments(
    iree_vm_module_signature_type_span_t types,
    iree_vm_program_bank_counts_t argument_counts,
    iree_vm_variant_span_t arguments, iree_vm_call_packet_t* packet) {
  if (argument_counts.value_count == types.count) {
    uint64_t* value_arguments = (uint64_t*)packet->value_arguments.direct;
    for (iree_host_size_t i = 0; i < types.count; ++i) {
      value_arguments[i] = arguments.data[i].payload;
      arguments.data[i] = iree_vm_variant_empty();
    }
    return;
  }
  uint16_t value_ordinal = 0;
  uint16_t ref_ordinal = 0;
  uint16_t function_ordinal = 0;
  for (iree_host_size_t i = 0; i < types.count; ++i) {
    const iree_vm_module_signature_type_t type = types.data[i];
    iree_vm_variant_t argument = arguments.data[i];
    if (type.kind > IREE_VM_SCALAR_TYPE_INVALID &&
        type.kind <= IREE_VM_SCALAR_TYPE_F64) {
      uint64_t* slot =
          value_ordinal < 16
              ? (uint64_t*)&packet->value_arguments.direct[value_ordinal]
              : (uint64_t*)&packet->value_arguments
                    .overflow[value_ordinal - 16];
      *slot = argument.payload;
      ++value_ordinal;
    } else if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      iree_vm_ref_t* slot =
          iree_vm_call_ref_argument_slot(packet, ref_ordinal++);
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
    } else {
      iree_vm_function_ref_t* slot =
          function_ordinal < 16
              ? (iree_vm_function_ref_t*)&packet->function_arguments
                    .direct[function_ordinal]
              : (iree_vm_function_ref_t*)&packet->function_arguments
                    .overflow[function_ordinal - 16];
      slot->program_bits = argument.payload;
      slot->target_bits =
          argument.metadata & ~(uint64_t)IREE_VM_VARIANT_TAG_MASK;
      ++function_ordinal;
    }
    arguments.data[i] = iree_vm_variant_empty();
  }
}

static void iree_vm_invocation_commit_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_t* callable, iree_vm_call_packet_t packet,
    uint8_t* stack_cursor, iree_vm_invocation_wake_callback_t wake_callback,
    bool has_external_borrowed_arguments) {
  invocation->process = process;
  invocation->root_target_bits = target_bits;
  invocation->root_call.callable = callable;
  invocation->root_call.packet = packet;
  invocation->root_call.allocation_begin = invocation->stack_base;
  invocation->stack_cursor = stack_cursor;
  invocation->wake_callback = wake_callback;
  invocation->operation = operation;
  invocation->has_external_borrowed_arguments = has_external_borrowed_arguments;
  invocation->state = IREE_VM_INVOCATION_STATE_RUNNING;
  iree_atomic_store(&invocation->cancel_reason, IREE_VM_CANCEL_REASON_NONE,
                    iree_memory_order_release);
}

static iree_status_t iree_vm_invocation_prepare_scalar_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_t* callable,
    iree_vm_module_signature_t signature, iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback) {
  if (IREE_UNLIKELY(arguments.count != signature.arguments.count)) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation argument count mismatch");
  }
  for (iree_host_size_t i = 0; i < signature.arguments.count; ++i) {
    const iree_vm_scalar_type_t scalar_type =
        (iree_vm_scalar_type_t)signature.arguments.data[i].kind;
    const iree_vm_variant_t argument = arguments.data[i];
    const uint64_t expected_metadata =
        ((uint64_t)scalar_type << 2) | IREE_VM_VARIANT_TAG_SCALAR;
    const uint32_t bit_width = iree_vm_invocation_scalar_bit_width(scalar_type);
    if (IREE_UNLIKELY(
            argument.metadata != expected_metadata ||
            (bit_width < 64 && (argument.payload >> bit_width) != 0))) {
      iree_vm_invocation_consume_arguments(arguments);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invocation scalar argument is not canonical");
    }
  }

  const iree_host_size_t argument_count = signature.arguments.count;
  const iree_host_size_t result_count = signature.results.count;
  const iree_host_size_t total_size =
      (argument_count + result_count) * sizeof(uint64_t);
  if (IREE_UNLIKELY(total_size > (iree_host_size_t)(invocation->storage_end -
                                                    invocation->stack_base))) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "root call staging exceeds invocation capacity");
  }

  uint64_t* value_arguments = (uint64_t*)invocation->stack_base;
  uint64_t* value_results = value_arguments + argument_count;
  iree_vm_call_packet_t packet = {0};
  packet.value_arguments.direct = value_arguments;
  packet.value_arguments.overflow =
      argument_count > 16 ? value_arguments + 16 : NULL;
  packet.value_results.direct = value_results;
  packet.value_results.overflow = result_count > 16 ? value_results + 16 : NULL;
  for (iree_host_size_t i = 0; i < argument_count; ++i) {
    value_arguments[i] = arguments.data[i].payload;
    arguments.data[i] = iree_vm_variant_empty();
  }
  iree_vm_invocation_commit_root(
      invocation, operation, process, target_bits, callable, packet,
      invocation->stack_base + total_size, wake_callback,
      /*has_external_borrowed_arguments=*/false);
  return iree_ok_status();
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_invocation_prepare_mixed_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_t* callable,
    iree_vm_module_signature_t signature, iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback) {
  bool has_external_borrowed_arguments = false;
  iree_status_t status = iree_vm_invocation_validate_arguments(
      process, target_bits, callable->signature_module, signature.arguments,
      callable->argument_counts, arguments, &has_external_borrowed_arguments);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_consume_arguments(arguments);
    return status;
  }

  iree_vm_call_packet_t packet = {0};
  uint8_t* stack_cursor = NULL;
  status = iree_vm_invocation_layout_root_call(
      invocation, callable->argument_counts, callable->result_counts, &packet,
      &stack_cursor);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_consume_arguments(arguments);
    return status;
  }

  iree_vm_invocation_stage_arguments(
      signature.arguments, callable->argument_counts, arguments, &packet);
  iree_vm_invocation_commit_root(invocation, operation, process, target_bits,
                                 callable, packet, stack_cursor, wake_callback,
                                 has_external_borrowed_arguments);
  return iree_ok_status();
}

iree_status_t iree_vm_invocation_prepare_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_t* callable,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_invocation_wake_callback_t wake_callback) {
  if (!iree_vm_invocation_is_idle(invocation) || !process ||
      operation == IREE_VM_INVOCATION_OPERATION_NONE || target_bits == 0 ||
      !callable) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid invocation root boundary");
  }
  const iree_vm_module_signature_t signature =
      iree_vm_program_callable_signature(callable);
  if (results.count != signature.results.count) {
    iree_vm_invocation_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation result count mismatch");
  }
  if (iree_vm_program_callable_is_scalar_only(callable)) {
    return iree_vm_invocation_prepare_scalar_root(
        invocation, operation, process, target_bits, callable, signature,
        arguments, wake_callback);
  }
  return iree_vm_invocation_prepare_mixed_root(invocation, operation, process,
                                               target_bits, callable, signature,
                                               arguments, wake_callback);
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
  if (!iree_vm_program_resolve_callable(program, callable_token)) {
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
  if (invocation->state != IREE_VM_INVOCATION_STATE_RUNNING ||
      invocation->executing_linked_module != execution->linked_module) {
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
      !invocation->executing_linked_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid child call request");
  }
  if (invocation->has_call_request) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module callback already requested a child call");
  }
  if (may_yield && !invocation->executing_may_yield) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "non-yielding module callback requested a yielding child");
  }
  invocation->call_request = (iree_vm_call_request_t){
      .linked_module = linked_module,
      .packet = *call,
      .function_ordinal = function_ordinal,
      .may_yield = may_yield,
  };
  invocation->has_call_request = true;
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  return iree_ok_status();
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_invocation_validate_dispatch_outcome_slow(
    const iree_vm_invocation_t* invocation, uint8_t* checkpoint_cursor,
    const iree_vm_frame_t* checkpoint_frame, bool may_yield,
    iree_vm_execution_outcome_t outcome) {
  const bool is_at_checkpoint = invocation->stack_cursor == checkpoint_cursor &&
                                invocation->top_frame == checkpoint_frame;
  const bool has_continuation = invocation->stack_cursor > checkpoint_cursor &&
                                invocation->top_frame != checkpoint_frame;
  if (!is_at_checkpoint && !has_continuation) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "module callback corrupted its frame boundary");
  }
  if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    if (invocation->has_call_request) {
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
  if (invocation->has_call_request) {
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
    iree_vm_execution_outcome_t* out_outcome) {
  uint8_t* checkpoint_cursor = invocation->stack_cursor;
  iree_vm_frame_t* checkpoint_frame = invocation->top_frame;
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
  invocation->executing_linked_module = linked_module;
  invocation->executing_function_ordinal = function_ordinal;
  invocation->executing_may_yield = may_yield;
  iree_vm_execution_outcome_t module_outcome = UINT32_MAX;
  iree_status_t status = linked_module->module->vtable->function_start(
      linked_module->module, &params, &module_outcome);
  invocation->executing_linked_module = NULL;

  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(module_outcome != IREE_VM_EXECUTION_OUTCOME_COMPLETED ||
                    invocation->stack_cursor != checkpoint_cursor ||
                    invocation->top_frame != checkpoint_frame ||
                    invocation->has_call_request)) {
    status = iree_vm_invocation_validate_dispatch_outcome_slow(
        invocation, checkpoint_cursor, checkpoint_frame, may_yield,
        module_outcome);
  }
  if (!iree_status_is_ok(status)) {
    invocation->has_call_request = false;
    iree_vm_invocation_unwind_to(invocation, checkpoint_cursor);
    return status;
  }
  *out_outcome = module_outcome;
  return iree_ok_status();
}

static iree_status_t iree_vm_invocation_dispatch_resume(
    iree_vm_invocation_t* invocation, iree_vm_frame_t* frame,
    iree_vm_execution_outcome_t* out_outcome) {
  uint8_t* checkpoint_cursor = frame->allocation_begin;
  iree_vm_frame_t* checkpoint_frame = frame->parent;
  const iree_vm_linked_module_t* linked_module = frame->linked_module;
  const bool may_yield = frame->may_yield;
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
  invocation->executing_linked_module = linked_module;
  invocation->executing_function_ordinal = frame->function_ordinal;
  invocation->executing_may_yield = may_yield;
  iree_vm_execution_outcome_t module_outcome = UINT32_MAX;
  iree_status_t status = linked_module->module->vtable->function_resume(
      linked_module->module, &params, &module_outcome);
  invocation->executing_linked_module = NULL;

  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(module_outcome != IREE_VM_EXECUTION_OUTCOME_COMPLETED ||
                    invocation->stack_cursor != checkpoint_cursor ||
                    invocation->top_frame != checkpoint_frame ||
                    invocation->has_call_request)) {
    status = iree_vm_invocation_validate_dispatch_outcome_slow(
        invocation, checkpoint_cursor, checkpoint_frame, may_yield,
        module_outcome);
  }
  if (!iree_status_is_ok(status)) {
    invocation->has_call_request = false;
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
    iree_vm_invocation_t* invocation,
    iree_vm_execution_outcome_t* out_outcome) {
  while (true) {
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    iree_status_t status = iree_ok_status();
    if (invocation->has_call_request) {
      const iree_vm_linked_module_t* linked_module =
          invocation->call_request.linked_module;
      const uint16_t function_ordinal =
          invocation->call_request.function_ordinal;
      const bool may_yield = invocation->call_request.may_yield;
      const iree_vm_call_packet_t* packet = &invocation->call_request.packet;
      invocation->has_call_request = false;
      status = iree_vm_invocation_dispatch_start(invocation, linked_module,
                                                 function_ordinal, may_yield,
                                                 packet, &outcome);
    } else if (invocation->top_frame) {
      status = iree_vm_invocation_dispatch_resume(
          invocation, invocation->top_frame, &outcome);
    } else {
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
      return iree_ok_status();
    }
    if (!iree_status_is_ok(status)) return status;
    if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED ||
        invocation->has_call_request) {
      continue;
    }
    return iree_vm_invocation_publish_suspension(invocation, out_outcome);
  }
}

iree_status_t iree_vm_invocation_drive_start(
    iree_vm_invocation_t* invocation,
    iree_vm_execution_outcome_t* out_outcome) {
  const uint16_t module_ordinal =
      iree_vm_program_target_module_ordinal(invocation->root_target_bits);
  const uint16_t function_ordinal =
      iree_vm_program_target_function_ordinal(invocation->root_target_bits);
  const iree_vm_linked_module_t* linked_module =
      &invocation->process->program->linked_modules[module_ordinal];
  const iree_fpu_state_t fpu_state =
      iree_fpu_state_push(iree_vm_invocation_fpu_state_flags);
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_dispatch_start(
      invocation, linked_module, function_ordinal,
      iree_vm_program_target_may_yield(invocation->root_target_bits),
      &invocation->root_call.packet, &outcome);
  if (iree_status_is_ok(status)) {
    if (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
      *out_outcome = outcome;
    } else if (invocation->has_call_request) {
      status = iree_vm_invocation_drive_continuations(invocation, out_outcome);
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
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_status_t status = iree_vm_invocation_dispatch_resume(
      invocation, invocation->top_frame, &outcome);
  if (iree_status_is_ok(status)) {
    if (invocation->has_call_request ||
        (outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED &&
         invocation->top_frame)) {
      status = iree_vm_invocation_drive_continuations(invocation, out_outcome);
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
          ->callables[execution->linked_module->callable_base +
                      local_function.callable_type_ordinal]
          .mapping;
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
  uint32_t mapping = execution->invocation->process->program
                         ->callables[execution->linked_module->callable_base +
                                     local_function.callable_type_ordinal]
                         .mapping;
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

iree_status_t iree_vm_invocation_validate_root_results(
    const iree_vm_invocation_t* invocation) {
  if (invocation->root_call.callable->result_counts.ref_count == 0 &&
      invocation->root_call.callable->result_counts.function_count == 0) {
    return iree_ok_status();
  }
  const iree_vm_program_t* program = invocation->process->program;
  const iree_vm_linked_module_t* signature_module =
      invocation->root_call.callable->signature_module;
  const iree_vm_module_signature_t signature =
      iree_vm_program_callable_signature(invocation->root_call.callable);

  uint16_t ref_ordinal = 0;
  uint16_t function_ordinal = 0;
  for (iree_host_size_t i = 0; i < signature.results.count; ++i) {
    const iree_vm_module_signature_type_t type = signature.results.data[i];
    if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      const iree_vm_ref_t ref = *iree_vm_call_ref_result_slot(
          &invocation->root_call.packet, ref_ordinal++);
      const iree_vm_ref_type_t expected_type =
          signature_module->module->descriptor->ref_types
              .data[type.type_ordinal];
      if ((!ref.object && ref.type_and_state != 0) ||
          (ref.object && (iree_vm_ref_type(ref) != expected_type ||
                          (ref.type_and_state & IREE_VM_REF_STATE_MASK) >
                              IREE_VM_REF_STATE_BORROWED))) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "module returned an invalid ref result");
      }
    } else if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION) {
      const iree_vm_function_ref_t function_ref =
          *iree_vm_call_function_result_slot(&invocation->root_call.packet,
                                             function_ordinal++);
      if (!iree_vm_program_function_ref_matches(
              program, function_ref, signature_module, type.type_ordinal)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "module returned an incompatible function result");
      }
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

static iree_vm_variant_t iree_vm_invocation_scalar_result(
    iree_vm_scalar_type_t scalar_type, uint64_t bits) {
  const uint32_t bit_width = iree_vm_invocation_scalar_bit_width(scalar_type);
  if (bit_width < 64) bits &= (UINT64_C(1) << bit_width) - 1;
  const iree_vm_variant_t variant = {
      bits,
      ((uint64_t)scalar_type << 2) | IREE_VM_VARIANT_TAG_SCALAR,
  };
  return variant;
}

static IREE_ATTRIBUTE_NOINLINE void
iree_vm_invocation_publish_uniform_scalar_results(
    const uint64_t* IREE_RESTRICT value_results, iree_host_size_t result_count,
    iree_vm_scalar_type_t scalar_type,
    iree_vm_variant_t* IREE_RESTRICT results) {
  const uint64_t metadata =
      ((uint64_t)scalar_type << 2) | IREE_VM_VARIANT_TAG_SCALAR;
  const uint32_t bit_width = iree_vm_invocation_scalar_bit_width(scalar_type);
  const uint64_t bit_mask =
      bit_width < 64 ? (UINT64_C(1) << bit_width) - 1 : UINT64_MAX;
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    results[i] = (iree_vm_variant_t){value_results[i] & bit_mask, metadata};
  }
}

static IREE_ATTRIBUTE_NOINLINE void iree_vm_invocation_publish_root_results(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results) {
  const iree_vm_module_signature_t signature =
      iree_vm_program_callable_signature(invocation->root_call.callable);
  if (invocation->root_call.callable->result_counts.value_count ==
      signature.results.count) {
    const uint64_t* value_results =
        invocation->root_call.packet.value_results.direct;
    for (iree_host_size_t i = 0; i < signature.results.count; ++i) {
      const iree_vm_scalar_type_t scalar_type =
          (iree_vm_scalar_type_t)signature.results.data[i].kind;
      results.data[i] =
          scalar_type == IREE_VM_SCALAR_TYPE_I64 ||
                  scalar_type == IREE_VM_SCALAR_TYPE_F64
              ? (iree_vm_variant_t){
                    value_results[i],
                    ((uint64_t)scalar_type << 2) | IREE_VM_VARIANT_TAG_SCALAR,
                }
              : iree_vm_invocation_scalar_result(scalar_type, value_results[i]);
    }
    return;
  }
  uint16_t value_ordinal = 0;
  uint16_t ref_ordinal = 0;
  uint16_t function_ordinal = 0;
  for (iree_host_size_t i = 0; i < signature.results.count; ++i) {
    const iree_vm_module_signature_type_t type = signature.results.data[i];
    if (type.kind > IREE_VM_SCALAR_TYPE_INVALID &&
        type.kind <= IREE_VM_SCALAR_TYPE_F64) {
      results.data[i] = iree_vm_invocation_scalar_result(
          type.kind, iree_vm_call_value_result_load(
                         &invocation->root_call.packet, value_ordinal++));
    } else if (type.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      iree_vm_ref_t* ref = iree_vm_call_ref_result_slot(
          &invocation->root_call.packet, ref_ordinal++);
      results.data[i] = iree_vm_variant_from_ref_move(ref);
    } else {
      results.data[i] =
          iree_vm_variant_from_function_ref(*iree_vm_call_function_result_slot(
              &invocation->root_call.packet, function_ordinal++));
    }
  }
}

void iree_vm_invocation_abort(iree_vm_invocation_t* invocation) {
  iree_atomic_store(&invocation->cancel_reason,
                    IREE_VM_INVOCATION_CANCEL_REASON_IDLE,
                    iree_memory_order_release);
  invocation->has_call_request = false;
  iree_vm_invocation_unwind_to(invocation,
                               invocation->root_call.allocation_begin);
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
    const iree_vm_program_callable_t** out_callable) {
  if (iree_vm_function_is_null(function) || (function.target_bits & 3u) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation function is invalid");
  }
  iree_vm_process_t* process =
      (iree_vm_process_t*)(uintptr_t)function.process_bits;
  // Binding validated the VM-produced target against this process. Repeating
  // those table bounds and token checks cannot make a forged process pointer
  // safe and would turn cold binding work into a tax on every call.
  const uint32_t callable_token =
      iree_vm_program_target_callable_token(function.target_bits);
  const iree_vm_program_callable_t* callable =
      &process->program->callables[callable_token - 1];
  *out_process = process;
  *out_callable = callable;
  return iree_ok_status();
}

static iree_status_t iree_vm_invocation_complete_scalar_call(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_cancel_reason_t cancel_reason = IREE_VM_CANCEL_REASON_NONE;
  if (!iree_vm_invocation_try_claim_completion(invocation, &cancel_reason)) {
    invocation->state = IREE_VM_INVOCATION_STATE_IDLE;
    invocation->operation = IREE_VM_INVOCATION_OPERATION_NONE;
    return iree_vm_invocation_cancel_status(cancel_reason);
  }

  const iree_vm_program_callable_t* callable = invocation->root_call.callable;
  if (callable->uniform_result_scalar_type != IREE_VM_SCALAR_TYPE_INVALID) {
    iree_vm_invocation_publish_uniform_scalar_results(
        invocation->root_call.packet.value_results.direct,
        callable->result_counts.value_count,
        callable->uniform_result_scalar_type, results.data);
  } else {
    iree_vm_invocation_publish_root_results(invocation, results);
  }
  invocation->state = IREE_VM_INVOCATION_STATE_IDLE;
  invocation->operation = IREE_VM_INVOCATION_OPERATION_NONE;
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static iree_status_t iree_vm_invocation_complete_call(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_status_t status = iree_vm_invocation_validate_root_results(invocation);
  iree_vm_cancel_reason_t cancel_reason = IREE_VM_CANCEL_REASON_NONE;
  if (iree_status_is_ok(status) &&
      !iree_vm_invocation_try_claim_completion(invocation, &cancel_reason)) {
    status = iree_vm_invocation_cancel_status(cancel_reason);
  }
  if (iree_status_is_ok(status)) {
    const iree_vm_program_callable_t* callable = invocation->root_call.callable;
    if (callable->uniform_result_scalar_type != IREE_VM_SCALAR_TYPE_INVALID) {
      iree_vm_invocation_publish_uniform_scalar_results(
          invocation->root_call.packet.value_results.direct,
          callable->result_counts.value_count,
          callable->uniform_result_scalar_type, results.data);
    } else {
      iree_vm_invocation_publish_root_results(invocation, results);
    }
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
  const iree_vm_program_callable_t* callable = NULL;
  iree_status_t status =
      iree_vm_invocation_resolve_root_function(function, &process, &callable);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_consume_arguments(arguments);
    return status;
  }
  status = iree_vm_invocation_prepare_root(
      invocation, IREE_VM_INVOCATION_OPERATION_CALL, process,
      function.target_bits, callable, arguments, results, wake_callback);
  if (!iree_status_is_ok(status)) return status;

  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  status = iree_vm_invocation_drive_start(invocation, &outcome);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_abort(invocation);
    return status;
  }
  if (outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    *out_outcome = outcome;
    return iree_ok_status();
  }
  if (iree_vm_program_callable_is_scalar_only(callable)) {
    return iree_vm_invocation_complete_scalar_call(invocation, results,
                                                   out_outcome);
  }
  return iree_vm_invocation_complete_call(invocation, results, out_outcome);
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_resume(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results,
    iree_vm_execution_outcome_t* out_outcome) {
  if (!invocation || !out_outcome ||
      invocation->state != IREE_VM_INVOCATION_STATE_SUSPENDED ||
      invocation->operation != IREE_VM_INVOCATION_OPERATION_CALL ||
      results.count !=
          iree_vm_program_callable_signature(invocation->root_call.callable)
              .results.count ||
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
  return iree_vm_invocation_complete_call(invocation, results, out_outcome);
}
