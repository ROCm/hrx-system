// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter.h"

#include <string.h>

#include "iree/vm/bytecode/interpreter/data.inl"
#include "iree/vm/bytecode/interpreter_atomic.h"
#include "iree/vm/bytecode/interpreter_buffer.h"
#include "iree/vm/bytecode/interpreter_call.h"
#include "iree/vm/bytecode/interpreter_conversion.h"
#include "iree/vm/bytecode/interpreter_float.h"
#include "iree/vm/bytecode/interpreter_frame.h"
#include "iree/vm/bytecode/interpreter_integer.h"
#include "iree/vm/bytecode/interpreter_memory.h"
#include "iree/vm/bytecode/process.h"

// Clang and GCC labels-as-values remove the shared loop branch and give each
// instruction site its own predictable successor. Relative label offsets keep
// the dispatch table read-only and free of dynamic pointer relocations. MSVC
// and Wasm use a portable switch over the same handler labels.
#if !defined(IREE_PLATFORM_WASM) && !defined(IREE_COMPILER_MSVC_COMPAT) && \
    (defined(IREE_COMPILER_CLANG) || defined(IREE_COMPILER_GCC))
#define IREE_VM_BYTECODE_DISPATCH_TABLE_ENTRY(opcode, label, record_type) \
  [IREE_VM_BYTECODE_OPCODE_##opcode] =                                    \
      (int32_t)((const uint8_t*) &&                                       \
                iree_vm_bytecode_dispatch_##label - (const uint8_t*) &&   \
                iree_vm_bytecode_dispatch_invalid),
#define IREE_VM_BYTECODE_DISPATCH_BEGIN()                                   \
  static const int32_t dispatch_table[IREE_VM_BYTECODE_OPCODE_CAPACITY] = { \
      IREE_VM_BYTECODE_INTERPRETER_OPCODE_LIST(                             \
          IREE_VM_BYTECODE_DISPATCH_TABLE_ENTRY)};                          \
  goto*((const uint8_t*) &&                                                 \
        iree_vm_bytecode_dispatch_invalid + dispatch_table[record_data[0]])
#define IREE_VM_BYTECODE_DISPATCH() \
  goto*((const uint8_t*) &&         \
        iree_vm_bytecode_dispatch_invalid + dispatch_table[record_data[0]])
#else
// The portable switch uses goto only to reach direct-threaded handlers.
#define IREE_VM_BYTECODE_DISPATCH_SWITCH_GOTO(label) \
  goto iree_vm_bytecode_dispatch_##label
#define IREE_VM_BYTECODE_DISPATCH_SWITCH_CASE(opcode, label, record_type) \
  case IREE_VM_BYTECODE_OPCODE_##opcode:                                  \
    IREE_VM_BYTECODE_DISPATCH_SWITCH_GOTO(label);
#define IREE_VM_BYTECODE_DISPATCH_BEGIN() IREE_VM_BYTECODE_DISPATCH()
#define IREE_VM_BYTECODE_DISPATCH()                     \
  do {                                                  \
    switch (record_data[0]) {                           \
      IREE_VM_BYTECODE_INTERPRETER_OPCODE_LIST(         \
          IREE_VM_BYTECODE_DISPATCH_SWITCH_CASE)        \
      default:                                          \
        IREE_VM_BYTECODE_DISPATCH_SWITCH_GOTO(invalid); \
    }                                                   \
  } while (false)
#endif  // computed goto dispatch

// Returns whether a function can execute directly in its caller result bank.
// Every other function uses one durable frame so calls and suspension preserve
// the exact same state independent of which path reaches them.
static bool iree_vm_bytecode_function_is_scalar_leaf(
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature) {
  return !iree_any_bit_set(function->flags_u16,
                           IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD |
                               IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL) &&
         function->local_byte_length_u16 == 0 &&
         function->ref_register_count_u16 == 0 &&
         function->local_ref_count_u32 == 0 &&
         function->function_register_count_u16 == 0 &&
         function->local_function_count_u32 == 0 &&
         function->value_register_count_u16 <=
             IREE_VM_CALL_DIRECT_REGISTER_COUNT &&
         function->value_register_count_u16 ==
             signature->result_value_count_u16;
}

// Maps one durable frame payload into the fixed execution state and banks.
static iree_vm_bytecode_execution_state_t* iree_vm_bytecode_map_execution_state(
    iree_vm_frame_t* frame, iree_vm_bytecode_frame_layout_t layout,
    iree_vm_call_packet_t call, const uint8_t* program_counter) {
  uint8_t* storage = (uint8_t*)iree_vm_frame_storage(frame);
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)storage;
  *state = (iree_vm_bytecode_execution_state_t){
      .call = call,
      .program_counter = program_counter,
      .frame = frame,
      .values = (uint64_t*)(storage + layout.values_offset),
      .refs = (iree_vm_ref_t*)(storage + layout.refs_offset),
      .local_refs = (iree_vm_ref_t*)(storage + layout.local_refs_offset),
      .call_ref_scratch =
          (iree_vm_ref_t*)(storage + layout.call_ref_scratch_offset),
      .functions = (iree_vm_function_ref_t*)(storage + layout.functions_offset),
      .local_functions =
          (iree_vm_function_ref_t*)(storage + layout.local_functions_offset),
      .local_bytes = storage + layout.local_bytes_offset,
  };
  return state;
}

#define IREE_VM_BYTECODE_DISPATCH_NEXT(record_type) \
  do {                                              \
    record_data += sizeof(record_type);             \
    IREE_VM_BYTECODE_DISPATCH();                    \
  } while (false)

// Replaces |target| with an already-safe complete ref state. The new state is
// installed before the old owner is released so self-replacement stays safe.
static inline void iree_vm_bytecode_ref_replace(iree_vm_ref_t* target,
                                                iree_vm_ref_t new_ref) {
  iree_vm_ref_t old_ref = *target;
  *target = new_ref;
  iree_vm_ref_reset(&old_ref);
}

// Retains |source| before replacing |target| so the two may alias.
static inline void iree_vm_bytecode_ref_retain(iree_vm_ref_t* target,
                                               iree_vm_ref_t source) {
  iree_vm_bytecode_ref_replace(target, iree_vm_ref_retain(source));
}

// Transfers one internal null, borrowed, or owned state exactly. Unlike a
// public escaping move, an internal borrowed state stays borrowed.
static inline void iree_vm_bytecode_ref_move(iree_vm_ref_t* target,
                                             iree_vm_ref_t* source) {
  iree_vm_ref_t moved_ref = *source;
  *source = iree_vm_ref_null();
  iree_vm_bytecode_ref_replace(target, moved_ref);
}

// Resolves one verified signed word displacement from the record end.
static inline const uint8_t* iree_vm_bytecode_direct_target(
    const uint8_t* record_data, iree_host_size_t record_length,
    int32_t target_word_offset) {
  return record_data + record_length +
         (ptrdiff_t)((int64_t)target_word_offset * 4);
}

// Returns the byte length selected by one verified memory format.
static inline uint8_t iree_vm_bytecode_memory_access_length(uint8_t format) {
  return (uint8_t)(1u << ((format >> 2) + (format & 3u)));
}

// Resolves one dynamically indexed access within verified local-byte bounds.
static inline iree_status_t iree_vm_bytecode_stack_resolve_index(
    uint16_t local_byte_length, uint16_t base, uint8_t access_length,
    uint64_t index, uint8_t scale, uint16_t* out_effective_base) {
  const uint32_t available = (uint32_t)local_byte_length - access_length - base;
  if (IREE_UNLIKELY(index > available / scale)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "indexed stack access is out of range");
  }
  *out_effective_base = (uint16_t)(base + (uint32_t)index * scale);
  return iree_ok_status();
}

// Fills i32 stack cells from one sign-extended s16 immediate.
static inline void iree_vm_bytecode_stack_const_s16_i32(uint8_t* target,
                                                        uint16_t count,
                                                        int16_t immediate) {
  const uint32_t value = (uint32_t)(int32_t)immediate;
  for (uint16_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u32(target + i * sizeof(value), value);
  }
}

// Fills i64 stack cells from one sign-extended s16 immediate.
static inline void iree_vm_bytecode_stack_const_s16_i64(uint8_t* target,
                                                        uint16_t count,
                                                        int16_t immediate) {
  const uint64_t value = (uint64_t)(int64_t)immediate;
  for (uint16_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u64(target + i * sizeof(value), value);
  }
}

// Packs zero-extended u16 immediates into i32 stack cells.
static inline void iree_vm_bytecode_stack_pack_i32(uint8_t* target,
                                                   const uint16_t* immediates,
                                                   uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u32(target + i * sizeof(uint32_t), immediates[i]);
  }
}

// Packs zero-extended u32 immediates into i64 stack cells.
static inline void iree_vm_bytecode_stack_pack_i64(uint8_t* target,
                                                   const uint32_t* immediates,
                                                   uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u64(target + i * sizeof(uint64_t), immediates[i]);
  }
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
iree_vm_bytecode_make_integer_division_status(
    iree_vm_bytecode_integer_division_failure_t failure) {
  if (failure == IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "integer division by zero");
  }
  if (failure == IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "signed integer division overflow");
  }
  IREE_ASSERT_UNREACHABLE("integer division returned an invalid failure");
  IREE_BUILTIN_UNREACHABLE();
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
iree_vm_bytecode_make_conversion_status(
    iree_vm_bytecode_conversion_failure_t failure) {
  if (failure == IREE_VM_BYTECODE_CONVERSION_FAILURE_NAN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot convert a floating NaN to an integer");
  }
  if (failure == IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "floating value is outside the selected integer interval");
  }
  IREE_ASSERT_UNREACHABLE("float conversion returned an invalid failure");
  IREE_BUILTIN_UNREACHABLE();
}

// Captures optional readable diagnostic bytes before frame unwind. Invalid
// diagnostic state never replaces the architectural failure code.
IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
iree_vm_bytecode_make_program_status(iree_status_code_t code,
                                     iree_vm_ref_type_t buffer_type,
                                     iree_vm_ref_t message_ref) {
  iree_string_view_t message = iree_string_view_empty();
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  if (message_ref.object && iree_vm_ref_type(message_ref) == buffer_type) {
    const iree_vm_buffer_t* buffer =
        (const iree_vm_buffer_t*)message_ref.object;
    const iree_host_size_t length = iree_vm_buffer_length(buffer);
    const void* data = iree_vm_buffer_const_data(buffer);
    if (data || length == 0) {
      message = iree_make_string_view((const char*)data, length);
    }
  }
#else
  (void)buffer_type;
  (void)message_ref;
#endif  // IREE_STATUS_FEATURE_ANNOTATIONS
  return iree_status_allocate_copy(code, iree_make_cstring_view(__FILE__),
                                   __LINE__, message);
}

// Returns whether |ref| is one canonical null or a valid exact typed state.
static inline bool iree_vm_bytecode_ref_matches_type(
    iree_vm_ref_t ref, iree_vm_ref_type_t expected_type) {
  if (!ref.object) return ref.type_and_state == 0;
  return (ref.type_and_state & IREE_VM_REF_STATE_MASK) <=
             IREE_VM_REF_STATE_BORROWED &&
         iree_vm_ref_type(ref) == expected_type;
}

static inline bool iree_vm_bytecode_ref_matches_global(
    iree_vm_ref_t ref, iree_vm_ref_type_t expected_type,
    const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor) {
  if (iree_vm_ref_is_null(ref)) {
    return iree_any_bit_set(descriptor->flags_u16,
                            IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE);
  }
  return iree_vm_bytecode_ref_matches_type(ref, expected_type);
}

static inline bool iree_vm_bytecode_function_matches_global(
    const iree_vm_module_execution_t* execution,
    iree_vm_function_ref_t function_ref,
    const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor) {
  if (iree_vm_function_ref_is_null(function_ref)) {
    return iree_any_bit_set(descriptor->flags_u16,
                            IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE);
  }
  return iree_vm_function_ref_matches_callable_type(
      execution, function_ref, descriptor->callable_type_ordinal_u16);
}

// Validates every dynamic result before publishing any caller-visible state.
static iree_status_t iree_vm_bytecode_publish_results(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state) {
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      iree_vm_bytecode_function_callable_type(&image->layout, function);
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &image->layout.signatures.rows[callable_type->signature_ordinal_u16];
  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
      iree_vm_bytecode_signature_descriptors(
          &image->layout.signatures, callable_type->signature_ordinal_u16) +
      iree_vm_bytecode_signature_argument_count(signature);

  uint16_t ref_ordinal = 0;
  uint16_t function_ordinal = 0;
  const uint32_t result_count =
      iree_vm_bytecode_signature_result_count(signature);
  for (uint32_t i = 0; i < result_count; ++i) {
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor =
        &descriptors[i];
    if (descriptor->kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
      const iree_vm_ref_t result =
          ref_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
              ? state->refs[ref_ordinal]
              : state->call.ref_results
                    .overflow[ref_ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
      if (!iree_vm_bytecode_ref_matches_type(
              result,
              image->resolved_ref_types[descriptor->type_ordinal_u16])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "bytecode function returned the wrong ref type");
      }
      ++ref_ordinal;
    } else if (descriptor->kind_u16 ==
               IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
      const iree_vm_function_ref_t result =
          function_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
              ? state->functions[function_ordinal]
              : state->call.function_results
                    .overflow[function_ordinal -
                              IREE_VM_CALL_DIRECT_REGISTER_COUNT];
      if (!iree_vm_function_ref_matches_callable_type(
              execution, result, descriptor->type_ordinal_u16)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "bytecode function returned an incompatible function value");
      }
      ++function_ordinal;
    }
  }

  const uint16_t direct_value_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature->result_value_count_u16);
  if (state->call.value_results.direct != state->values) {
    iree_vm_bytecode_frame_copy_direct_values(
        state->call.value_results.direct, state->values, direct_value_count);
  }
  const uint16_t direct_ref_count = iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                                             signature->result_ref_count_u16);
  for (uint16_t i = 0; i < direct_ref_count; ++i) {
    iree_vm_call_ref_result_store_move(&state->call, i, &state->refs[i]);
  }
  const uint16_t direct_function_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature->result_function_count_u16);
  if (direct_function_count != 0 &&
      state->call.function_results.direct != state->functions) {
    memcpy(state->call.function_results.direct, state->functions,
           direct_function_count * sizeof(*state->functions));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_dispatch(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_t* invocation = execution->invocation;
  iree_vm_call_packet_t* call = &state->call;
  uint64_t* values = state->values;
  iree_vm_ref_t* refs = state->refs;
  iree_vm_ref_t* local_refs = state->local_refs;
  iree_vm_function_ref_t* functions = state->functions;
  iree_vm_function_ref_t* local_functions = state->local_functions;
  uint8_t* local_bytes = state->local_bytes;
  const uint8_t* function_bytecode =
      image->layout.functions.bytecode_data + function->bytecode_offset_u32;
  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets =
      iree_vm_bytecode_function_switch_targets(&image->layout.functions,
                                               function);
  const uint8_t* record_data = state->program_counter;

  IREE_VM_BYTECODE_DISPATCH_BEGIN();

iree_vm_bytecode_dispatch_control_block: {
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_control_block_t);
}
iree_vm_bytecode_dispatch_control_return: {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_publish_results(image, function, execution, state));
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}
iree_vm_bytecode_dispatch_control_yield_s32: {
  const iree_vm_bytecode_control_yield_s32_t* record =
      (const iree_vm_bytecode_control_yield_s32_t*)record_data;
  if (iree_vm_invocation_has_external_borrowed_arguments(invocation)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "bytecode function yielded while external borrowed refs were live");
  }
  state->program_counter = iree_vm_bytecode_direct_target(
      record_data, sizeof(*record), record->target_word_offset_s32);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  const iree_vm_invocation_wake_callback_t wake_callback =
      iree_vm_invocation_wake_callback(invocation);
  if (wake_callback.fn) wake_callback.fn(wake_callback.user_data);
  return iree_ok_status();
}
iree_vm_bytecode_dispatch_control_branch_s16: {
  const iree_vm_bytecode_control_branch_s16_t* record =
      (const iree_vm_bytecode_control_branch_s16_t*)record_data;
  record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                               record->target_word_offset_s16);
  IREE_VM_BYTECODE_DISPATCH();
}
iree_vm_bytecode_dispatch_control_branch_s32: {
  const iree_vm_bytecode_control_branch_s32_t* record =
      (const iree_vm_bytecode_control_branch_s32_t*)record_data;
  record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                               record->target_word_offset_s32);
  IREE_VM_BYTECODE_DISPATCH();
}
iree_vm_bytecode_dispatch_control_branch_if_s16: {
  const iree_vm_bytecode_control_branch_if_s16_t* record =
      (const iree_vm_bytecode_control_branch_if_s16_t*)record_data;
  record_data =
      values[record->condition_v8]
          ? iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                           record->target_word_offset_s16)
          : record_data + sizeof(*record);
  IREE_VM_BYTECODE_DISPATCH();
}
iree_vm_bytecode_dispatch_control_branch_if_s32: {
  const iree_vm_bytecode_control_branch_if_s32_t* record =
      (const iree_vm_bytecode_control_branch_if_s32_t*)record_data;
  record_data =
      values[record->condition_v8]
          ? iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                           record->target_word_offset_s32)
          : record_data + sizeof(*record);
  IREE_VM_BYTECODE_DISPATCH();
}
iree_vm_bytecode_dispatch_control_branch_unless_s16: {
  const iree_vm_bytecode_control_branch_unless_s16_t* record =
      (const iree_vm_bytecode_control_branch_unless_s16_t*)record_data;
  record_data =
      !values[record->condition_v8]
          ? iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                           record->target_word_offset_s16)
          : record_data + sizeof(*record);
  IREE_VM_BYTECODE_DISPATCH();
}
iree_vm_bytecode_dispatch_control_branch_unless_s32: {
  const iree_vm_bytecode_control_branch_unless_s32_t* record =
      (const iree_vm_bytecode_control_branch_unless_s32_t*)record_data;
  record_data =
      !values[record->condition_v8]
          ? iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                           record->target_word_offset_s32)
          : record_data + sizeof(*record);
  IREE_VM_BYTECODE_DISPATCH();
}
iree_vm_bytecode_dispatch_control_switch: {
  const iree_vm_bytecode_control_switch_t* record =
      (const iree_vm_bytecode_control_switch_t*)record_data;
  const uint64_t selector = values[record->selector_v8];
  if (selector < record->target_count_u16) {
    const uint32_t target_word_offset =
        switch_targets[record->target_base_u32 + (uint32_t)selector]
            .target_word_offset_u32;
    record_data = function_bytecode + (iree_host_size_t)target_word_offset * 4;
    IREE_VM_BYTECODE_DISPATCH();
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_control_switch_t);
}
iree_vm_bytecode_dispatch_control_call: {
  const iree_vm_bytecode_control_call_t* record =
      (const iree_vm_bytecode_control_call_t*)record_data;
  iree_status_t status = iree_vm_bytecode_call_direct(image, execution, state,
                                                      record, out_outcome);
  if (iree_status_is_ok(status)) {
    state->program_counter = record_data + sizeof(*record);
  }
  return status;
}
iree_vm_bytecode_dispatch_control_call_indirect: {
  const iree_vm_bytecode_control_call_indirect_t* record =
      (const iree_vm_bytecode_control_call_indirect_t*)record_data;
  iree_status_t status = iree_vm_bytecode_call_indirect(image, execution, state,
                                                        record, out_outcome);
  if (iree_status_is_ok(status)) {
    state->program_counter = record_data + sizeof(*record);
  }
  return status;
}
iree_vm_bytecode_dispatch_control_assert: {
  const iree_vm_bytecode_control_assert_t* record =
      (const iree_vm_bytecode_control_assert_t*)record_data;
  if (IREE_LIKELY(values[record->condition_v8] != 0)) {
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_control_assert_t);
  }
  return iree_vm_bytecode_make_program_status(
      IREE_STATUS_FAILED_PRECONDITION, image->buffer_type,
      refs[record->message_r8_nullable]);
}
iree_vm_bytecode_dispatch_control_fail: {
  const iree_vm_bytecode_control_fail_t* record =
      (const iree_vm_bytecode_control_fail_t*)record_data;
  return iree_vm_bytecode_make_program_status(
      (iree_status_code_t)record->status_u8, image->buffer_type,
      refs[record->message_r8_nullable]);
}

iree_vm_bytecode_dispatch_value_copy: {
  const iree_vm_bytecode_value_copy_t* record =
      (const iree_vm_bytecode_value_copy_t*)record_data;
  values[record->destination_v8] = values[record->source_v8];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_value_copy_t);
}
iree_vm_bytecode_dispatch_value_select: {
  const iree_vm_bytecode_value_select_t* record =
      (const iree_vm_bytecode_value_select_t*)record_data;
  const uint64_t condition = values[record->condition_v8];
  const uint64_t selected =
      condition ? values[record->true_v8] : values[record->false_v8];
  values[record->destination_v8] = selected;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_value_select_t);
}

iree_vm_bytecode_dispatch_constant_zero: {
  const iree_vm_bytecode_constant_zero_t* record =
      (const iree_vm_bytecode_constant_zero_t*)record_data;
  values[record->destination_v8] = 0;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_constant_zero_t);
}
iree_vm_bytecode_dispatch_constant_s16: {
  const iree_vm_bytecode_constant_s16_t* record =
      (const iree_vm_bytecode_constant_s16_t*)record_data;
  values[record->destination_v8] = (uint64_t)(int64_t)record->immediate_i16;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_constant_s16_t);
}
iree_vm_bytecode_dispatch_constant_i32: {
  const iree_vm_bytecode_constant_i32_t* record =
      (const iree_vm_bytecode_constant_i32_t*)record_data;
  values[record->destination_v8] = record->bits_u32;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_constant_i32_t);
}
iree_vm_bytecode_dispatch_constant_i64: {
  const iree_vm_bytecode_constant_i64_t* record =
      (const iree_vm_bytecode_constant_i64_t*)record_data;
  values[record->destination_v8] =
      (uint64_t)record->bits_low_u32 | ((uint64_t)record->bits_high_u32 << 32);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_constant_i64_t);
}
iree_vm_bytecode_dispatch_constant_pool_load_i32: {
  const iree_vm_bytecode_constant_pool_load_i32_t* record =
      (const iree_vm_bytecode_constant_pool_load_i32_t*)record_data;
  values[record->destination_v8] =
      (uint32_t)image->layout.constants.cells[record->constant_pool_ordinal_u16]
          .bits_u64;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_constant_pool_load_i32_t);
}
iree_vm_bytecode_dispatch_constant_pool_load_i64: {
  const iree_vm_bytecode_constant_pool_load_i64_t* record =
      (const iree_vm_bytecode_constant_pool_load_i64_t*)record_data;
  values[record->destination_v8] =
      image->layout.constants.cells[record->constant_pool_ordinal_u16].bits_u64;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_constant_pool_load_i64_t);
}

iree_vm_bytecode_dispatch_func_null: {
  const iree_vm_bytecode_func_null_t* record =
      (const iree_vm_bytecode_func_null_t*)record_data;
  functions[record->destination_f8] = iree_vm_function_ref_null();
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_null_t);
}
iree_vm_bytecode_dispatch_func_compare_null: {
  const iree_vm_bytecode_func_compare_null_t* record =
      (const iree_vm_bytecode_func_compare_null_t*)record_data;
  values[record->destination_v8] =
      iree_vm_function_ref_is_null(functions[record->source_f8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_compare_null_t);
}
iree_vm_bytecode_dispatch_func_copy: {
  const iree_vm_bytecode_func_copy_t* record =
      (const iree_vm_bytecode_func_copy_t*)record_data;
  functions[record->destination_f8] = functions[record->source_f8];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_copy_t);
}
iree_vm_bytecode_dispatch_func_address: {
  const iree_vm_bytecode_func_address_t* record =
      (const iree_vm_bytecode_func_address_t*)record_data;
  iree_vm_function_ref_t function_ref = iree_vm_function_ref_null();
  if (record->target_kind_u8 == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL) {
    const iree_vm_bytecode_v0_function_row_t* target_function =
        &image->layout.functions.rows[record->target_ordinal_u16];
    const iree_vm_module_local_function_t local_function = {
        .function_ordinal = record->target_ordinal_u16,
        .callable_type_ordinal = record->callable_type_ordinal_u16,
        .flags = iree_any_bit_set(target_function->flags_u16,
                                  IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD)
                     ? IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD
                     : IREE_VM_MODULE_FUNCTION_FLAG_NONE,
    };
    IREE_RETURN_IF_ERROR(iree_vm_function_ref_from_local_function(
        execution, local_function, &function_ref));
  } else {
    IREE_RETURN_IF_ERROR(iree_vm_function_ref_from_import(
        execution, record->target_ordinal_u16, &function_ref));
  }
  functions[record->destination_f8] = function_ref;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_address_t);
}
iree_vm_bytecode_dispatch_func_import_resolved: {
  const iree_vm_bytecode_func_import_resolved_t* record =
      (const iree_vm_bytecode_func_import_resolved_t*)record_data;
  iree_vm_function_ref_t function_ref = iree_vm_function_ref_null();
  IREE_RETURN_IF_ERROR(iree_vm_function_ref_from_import(
      execution, record->import_ordinal_u16, &function_ref));
  values[record->destination_v8] = !iree_vm_function_ref_is_null(function_ref);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_import_resolved_t);
}
iree_vm_bytecode_dispatch_func_stack_load: {
  const iree_vm_bytecode_func_stack_load_t* record =
      (const iree_vm_bytecode_func_stack_load_t*)record_data;
  functions[record->destination_f8] =
      local_functions[record->local_ordinal_u16];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_stack_load_t);
}
iree_vm_bytecode_dispatch_func_stack_store: {
  const iree_vm_bytecode_func_stack_store_t* record =
      (const iree_vm_bytecode_func_stack_store_t*)record_data;
  local_functions[record->local_ordinal_u16] = functions[record->source_f8];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_stack_store_t);
}

iree_vm_bytecode_dispatch_global_value_immutable_load: {
  const iree_vm_bytecode_global_value_immutable_load_t* record =
      (const iree_vm_bytecode_global_value_immutable_load_t*)record_data;
  iree_vm_bytecode_process_header_t* process_header =
      iree_vm_bytecode_process_header(execution->process_storage);
  uint64_t* set_bits = iree_vm_bytecode_process_value_set_bits(
      image, execution->process_storage);
  if (process_header->construction_state ==
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN &&
      !iree_vm_bytecode_process_bit_test(set_bits, record->global_u16)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "immutable value global is unset during process construction");
  }
  values[record->destination_v8] = iree_vm_bytecode_process_values(
      image, execution->process_storage)[record->global_u16];
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_global_value_immutable_load_t);
}
iree_vm_bytecode_dispatch_global_value_immutable_store: {
  const iree_vm_bytecode_global_value_immutable_store_t* record =
      (const iree_vm_bytecode_global_value_immutable_store_t*)record_data;
  iree_vm_bytecode_process_header_t* process_header =
      iree_vm_bytecode_process_header(execution->process_storage);
  uint64_t* set_bits = iree_vm_bytecode_process_value_set_bits(
      image, execution->process_storage);
  if (process_header->construction_state !=
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN ||
      iree_vm_bytecode_process_bit_test(set_bits, record->global_u16)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "immutable value global cannot be initialized");
  }
  iree_vm_bytecode_process_values(
      image, execution->process_storage)[record->global_u16] =
      values[record->source_v8];
  iree_vm_bytecode_process_bit_set(set_bits, record->global_u16);
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_global_value_immutable_store_t);
}
iree_vm_bytecode_dispatch_global_value_mutable_load: {
  const iree_vm_bytecode_global_value_mutable_load_t* record =
      (const iree_vm_bytecode_global_value_mutable_load_t*)record_data;
  values[record->destination_v8] = iree_vm_bytecode_process_values(
      image, execution->process_storage)[record->global_u16];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_global_value_mutable_load_t);
}
iree_vm_bytecode_dispatch_global_value_mutable_store: {
  const iree_vm_bytecode_global_value_mutable_store_t* record =
      (const iree_vm_bytecode_global_value_mutable_store_t*)record_data;
  iree_vm_bytecode_process_values(
      image, execution->process_storage)[record->global_u16] =
      values[record->source_v8];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_global_value_mutable_store_t);
}
iree_vm_bytecode_dispatch_global_ref_immutable_load_borrow: {
  const iree_vm_bytecode_global_ref_immutable_load_borrow_t* record =
      (const iree_vm_bytecode_global_ref_immutable_load_borrow_t*)record_data;
  iree_vm_bytecode_process_header_t* process_header =
      iree_vm_bytecode_process_header(execution->process_storage);
  uint64_t* set_bits =
      iree_vm_bytecode_process_ref_set_bits(image, execution->process_storage);
  if (process_header->construction_state ==
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN &&
      !iree_vm_bytecode_process_bit_test(set_bits, record->global_u16)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "immutable ref global is unset during process construction");
  }
  const iree_vm_ref_t source = iree_vm_bytecode_process_refs(
      image, execution->process_storage)[record->global_u16];
  const iree_vm_ref_t borrowed =
      source.object ? iree_vm_ref_from_ptr_borrowed(source.object,
                                                    iree_vm_ref_type(source))
                    : iree_vm_ref_null();
  iree_vm_bytecode_ref_replace(&refs[record->destination_r8], borrowed);
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_global_ref_immutable_load_borrow_t);
}
iree_vm_bytecode_dispatch_global_ref_immutable_store_move: {
  const iree_vm_bytecode_global_ref_immutable_store_move_t* record =
      (const iree_vm_bytecode_global_ref_immutable_store_move_t*)record_data;
  iree_vm_bytecode_process_header_t* process_header =
      iree_vm_bytecode_process_header(execution->process_storage);
  uint64_t* set_bits =
      iree_vm_bytecode_process_ref_set_bits(image, execution->process_storage);
  if (process_header->construction_state !=
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN ||
      iree_vm_bytecode_process_bit_test(set_bits, record->global_u16)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "immutable ref global cannot be initialized");
  }
  const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor =
      &image->layout.globals.refs[record->global_u16];
  if (!iree_vm_bytecode_ref_matches_global(
          refs[record->source_r8],
          image->resolved_ref_types[descriptor->ref_type_ordinal_u16],
          descriptor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "immutable ref global type does not match");
  }
  iree_vm_bytecode_process_refs(
      image, execution->process_storage)[record->global_u16] =
      iree_vm_ref_move(&refs[record->source_r8]);
  iree_vm_bytecode_process_bit_set(set_bits, record->global_u16);
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_global_ref_immutable_store_move_t);
}
iree_vm_bytecode_dispatch_global_ref_mutable_load_retain: {
  const iree_vm_bytecode_global_ref_mutable_load_retain_t* record =
      (const iree_vm_bytecode_global_ref_mutable_load_retain_t*)record_data;
  const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor =
      &image->layout.globals.refs[record->global_u16];
  const iree_vm_ref_t source = iree_vm_bytecode_process_refs(
      image, execution->process_storage)[record->global_u16];
  if (iree_vm_bytecode_process_header(execution->process_storage)
              ->construction_state ==
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN &&
      iree_vm_ref_is_null(source) &&
      !iree_any_bit_set(descriptor->flags_u16,
                        IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "nonnullable mutable ref global is temporarily null");
  }
  iree_vm_bytecode_ref_retain(&refs[record->destination_r8], source);
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_global_ref_mutable_load_retain_t);
}
iree_vm_bytecode_dispatch_global_ref_mutable_store_move: {
  const iree_vm_bytecode_global_ref_mutable_store_move_t* record =
      (const iree_vm_bytecode_global_ref_mutable_store_move_t*)record_data;
  const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor =
      &image->layout.globals.refs[record->global_u16];
  if (!iree_vm_bytecode_ref_matches_global(
          refs[record->source_r8],
          image->resolved_ref_types[descriptor->ref_type_ordinal_u16],
          descriptor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mutable ref global type does not match");
  }
  iree_vm_ref_t* target = &iree_vm_bytecode_process_refs(
      image, execution->process_storage)[record->global_u16];
  iree_vm_bytecode_ref_replace(target,
                               iree_vm_ref_move(&refs[record->source_r8]));
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_global_ref_mutable_store_move_t);
}
iree_vm_bytecode_dispatch_global_func_immutable_load: {
  const iree_vm_bytecode_global_func_immutable_load_t* record =
      (const iree_vm_bytecode_global_func_immutable_load_t*)record_data;
  iree_vm_bytecode_process_header_t* process_header =
      iree_vm_bytecode_process_header(execution->process_storage);
  uint64_t* set_bits = iree_vm_bytecode_process_function_set_bits(
      image, execution->process_storage);
  if (process_header->construction_state ==
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN &&
      !iree_vm_bytecode_process_bit_test(set_bits, record->global_u16)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "immutable function global is unset during process construction");
  }
  functions[record->destination_f8] = iree_vm_bytecode_process_functions(
      image, execution->process_storage)[record->global_u16];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_global_func_immutable_load_t);
}
iree_vm_bytecode_dispatch_global_func_immutable_store: {
  const iree_vm_bytecode_global_func_immutable_store_t* record =
      (const iree_vm_bytecode_global_func_immutable_store_t*)record_data;
  iree_vm_bytecode_process_header_t* process_header =
      iree_vm_bytecode_process_header(execution->process_storage);
  uint64_t* set_bits = iree_vm_bytecode_process_function_set_bits(
      image, execution->process_storage);
  if (process_header->construction_state !=
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN ||
      iree_vm_bytecode_process_bit_test(set_bits, record->global_u16)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "immutable function global cannot be initialized");
  }
  const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor =
      &image->layout.globals.functions[record->global_u16];
  if (!iree_vm_bytecode_function_matches_global(
          execution, functions[record->source_f8], descriptor)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "immutable function global contract does not match");
  }
  iree_vm_bytecode_process_functions(
      image, execution->process_storage)[record->global_u16] =
      functions[record->source_f8];
  iree_vm_bytecode_process_bit_set(set_bits, record->global_u16);
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_global_func_immutable_store_t);
}
iree_vm_bytecode_dispatch_global_func_mutable_load: {
  const iree_vm_bytecode_global_func_mutable_load_t* record =
      (const iree_vm_bytecode_global_func_mutable_load_t*)record_data;
  const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor =
      &image->layout.globals.functions[record->global_u16];
  const iree_vm_function_ref_t source = iree_vm_bytecode_process_functions(
      image, execution->process_storage)[record->global_u16];
  if (iree_vm_bytecode_process_header(execution->process_storage)
              ->construction_state ==
          IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN &&
      iree_vm_function_ref_is_null(source) &&
      !iree_any_bit_set(descriptor->flags_u16,
                        IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "nonnullable mutable function global is temporarily null");
  }
  functions[record->destination_f8] = source;
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_global_func_mutable_load_t);
}
iree_vm_bytecode_dispatch_global_func_mutable_store: {
  const iree_vm_bytecode_global_func_mutable_store_t* record =
      (const iree_vm_bytecode_global_func_mutable_store_t*)record_data;
  const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor =
      &image->layout.globals.functions[record->global_u16];
  if (!iree_vm_bytecode_function_matches_global(
          execution, functions[record->source_f8], descriptor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mutable function global contract does not match");
  }
  iree_vm_bytecode_process_functions(
      image, execution->process_storage)[record->global_u16] =
      functions[record->source_f8];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_global_func_mutable_store_t);
}

#define IREE_VM_BYTECODE_DISPATCH_INTEGER_TOTAL(opcode, label, record_type) \
  iree_vm_bytecode_dispatch_##label : {                                     \
    const record_type* record = (const record_type*)record_data;            \
    iree_vm_bytecode_execute_##label(record, values);                       \
    IREE_VM_BYTECODE_DISPATCH_NEXT(record_type);                            \
  }
  IREE_VM_BYTECODE_INTERPRETER_INTEGER_TOTAL_LIST(
      IREE_VM_BYTECODE_DISPATCH_INTEGER_TOTAL)
#undef IREE_VM_BYTECODE_DISPATCH_INTEGER_TOTAL

#define IREE_VM_BYTECODE_DISPATCH_INTEGER_FALLIBLE(opcode, label, record_type) \
  iree_vm_bytecode_dispatch_##label : {                                        \
    const record_type* record = (const record_type*)record_data;               \
    const iree_vm_bytecode_integer_division_failure_t failure =                \
        iree_vm_bytecode_execute_##label(record, values);                      \
    if (IREE_UNLIKELY(failure !=                                               \
                      IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE)) {       \
      return iree_vm_bytecode_make_integer_division_status(failure);           \
    }                                                                          \
    IREE_VM_BYTECODE_DISPATCH_NEXT(record_type);                               \
  }
  IREE_VM_BYTECODE_INTERPRETER_INTEGER_FALLIBLE_LIST(
      IREE_VM_BYTECODE_DISPATCH_INTEGER_FALLIBLE)
#undef IREE_VM_BYTECODE_DISPATCH_INTEGER_FALLIBLE

#define IREE_VM_BYTECODE_DISPATCH_FLOAT(opcode, label, record_type) \
  iree_vm_bytecode_dispatch_##label : {                             \
    const record_type* record = (const record_type*)record_data;    \
    iree_vm_bytecode_execute_##label(record, values);               \
    IREE_VM_BYTECODE_DISPATCH_NEXT(record_type);                    \
  }
  IREE_VM_BYTECODE_INTERPRETER_FLOAT_LIST(IREE_VM_BYTECODE_DISPATCH_FLOAT)
#undef IREE_VM_BYTECODE_DISPATCH_FLOAT

#define IREE_VM_BYTECODE_DISPATCH_CONVERSION_TOTAL(opcode, label, record_type) \
  iree_vm_bytecode_dispatch_##label : {                                        \
    const record_type* record = (const record_type*)record_data;               \
    iree_vm_bytecode_execute_##label(record, values);                          \
    IREE_VM_BYTECODE_DISPATCH_NEXT(record_type);                               \
  }
  IREE_VM_BYTECODE_INTERPRETER_CONVERSION_TOTAL_LIST(
      IREE_VM_BYTECODE_DISPATCH_CONVERSION_TOTAL)
#undef IREE_VM_BYTECODE_DISPATCH_CONVERSION_TOTAL

#define IREE_VM_BYTECODE_DISPATCH_CONVERSION_FALLIBLE(opcode, label,          \
                                                      record_type)            \
  iree_vm_bytecode_dispatch_##label : {                                       \
    const record_type* record = (const record_type*)record_data;              \
    const iree_vm_bytecode_conversion_failure_t failure =                     \
        iree_vm_bytecode_execute_##label(record, values);                     \
    if (IREE_UNLIKELY(failure != IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE)) { \
      return iree_vm_bytecode_make_conversion_status(failure);                \
    }                                                                         \
    IREE_VM_BYTECODE_DISPATCH_NEXT(record_type);                              \
  }
  IREE_VM_BYTECODE_INTERPRETER_CONVERSION_FALLIBLE_LIST(
      IREE_VM_BYTECODE_DISPATCH_CONVERSION_FALLIBLE)
#undef IREE_VM_BYTECODE_DISPATCH_CONVERSION_FALLIBLE

iree_vm_bytecode_dispatch_stack_load: {
  const iree_vm_bytecode_stack_load_t* record =
      (const iree_vm_bytecode_stack_load_t*)record_data;
  iree_vm_bytecode_memory_load_lanes(record->format_u8,
                                     local_bytes + record->base_u16,
                                     values + record->destination_v8);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_load_t);
}
iree_vm_bytecode_dispatch_stack_store: {
  const iree_vm_bytecode_stack_store_t* record =
      (const iree_vm_bytecode_stack_store_t*)record_data;
  iree_vm_bytecode_memory_store_lanes(record->format_u8,
                                      values + record->source_v8,
                                      local_bytes + record->base_u16);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_store_t);
}
iree_vm_bytecode_dispatch_stack_load_indexed: {
  const iree_vm_bytecode_stack_load_indexed_t* record =
      (const iree_vm_bytecode_stack_load_indexed_t*)record_data;
  uint16_t effective_base = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_stack_resolve_index(
      function->local_byte_length_u16, record->base_u16,
      iree_vm_bytecode_memory_access_length(record->format_u8),
      values[record->index_v8], record->scale_u8, &effective_base));
  iree_vm_bytecode_memory_load_lanes(record->format_u8,
                                     local_bytes + effective_base,
                                     values + record->destination_v8);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_load_indexed_t);
}
iree_vm_bytecode_dispatch_stack_store_indexed: {
  const iree_vm_bytecode_stack_store_indexed_t* record =
      (const iree_vm_bytecode_stack_store_indexed_t*)record_data;
  uint16_t effective_base = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_stack_resolve_index(
      function->local_byte_length_u16, record->base_u16,
      iree_vm_bytecode_memory_access_length(record->format_u8),
      values[record->index_v8], record->scale_u8, &effective_base));
  iree_vm_bytecode_memory_store_lanes(record->format_u8,
                                      values + record->source_v8,
                                      local_bytes + effective_base);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_store_indexed_t);
}
iree_vm_bytecode_dispatch_stack_fill: {
  const iree_vm_bytecode_stack_fill_t* record =
      (const iree_vm_bytecode_stack_fill_t*)record_data;
  if (record->length_u16 != 0) {
    iree_vm_bytecode_fill_pattern(
        local_bytes + record->target_base_u16, record->length_u16,
        values[record->pattern_v8], record->pattern_width_u8);
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_fill_t);
}
iree_vm_bytecode_dispatch_stack_copy: {
  const iree_vm_bytecode_stack_copy_t* record =
      (const iree_vm_bytecode_stack_copy_t*)record_data;
  if (record->length_u16 != 0) {
    memmove(local_bytes + record->target_u16, local_bytes + record->source_u16,
            record->length_u16);
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_copy_t);
}
iree_vm_bytecode_dispatch_stack_compare: {
  const iree_vm_bytecode_stack_compare_t* record =
      (const iree_vm_bytecode_stack_compare_t*)record_data;
  const int ordering =
      record->length_u16 == 0
          ? 0
          : memcmp(local_bytes + record->left_u16,
                   local_bytes + record->right_u16, record->length_u16);
  values[record->destination_v8] = ordering < 0   ? UINT32_MAX
                                   : ordering > 0 ? UINT32_C(1)
                                                  : UINT32_C(0);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_compare_t);
}
iree_vm_bytecode_dispatch_stack_copy_rodata: {
  const iree_vm_bytecode_stack_copy_rodata_t* record =
      (const iree_vm_bytecode_stack_copy_rodata_t*)record_data;
  if (record->length_u16 != 0) {
    const uint8_t* source = (const uint8_t*)iree_vm_buffer_const_data(
        &image->rodata_roots[record->rodata_u16]);
    memcpy(local_bytes + record->target_u16, source + record->source_offset_u32,
           record->length_u16);
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_copy_rodata_t);
}
iree_vm_bytecode_dispatch_stack_copy_from_buffer: {
  const iree_vm_bytecode_stack_copy_from_buffer_t* record =
      (const iree_vm_bytecode_stack_copy_from_buffer_t*)record_data;
  iree_vm_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &source_buffer));
  iree_byte_span_t source = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      source_buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ,
      values[record->source_offset_v8], record->length_u16, &source));
  if (record->length_u16 != 0) {
    memcpy(local_bytes + record->target_u16, source.data, record->length_u16);
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_copy_from_buffer_t);
}
iree_vm_bytecode_dispatch_stack_copy_to_buffer: {
  const iree_vm_bytecode_stack_copy_to_buffer_t* record =
      (const iree_vm_bytecode_stack_copy_to_buffer_t*)record_data;
  iree_vm_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &target_buffer));
  iree_byte_span_t target = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      target_buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      values[record->target_offset_v8], record->length_u16, &target));
  if (record->length_u16 != 0) {
    memcpy(target.data, local_bytes + record->source_u16, record->length_u16);
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_copy_to_buffer_t);
}
iree_vm_bytecode_dispatch_stack_const_s16_i32: {
  const iree_vm_bytecode_stack_const_s16_i32_t* record =
      (const iree_vm_bytecode_stack_const_s16_i32_t*)record_data;
  iree_vm_bytecode_stack_const_s16_i32(local_bytes + record->target_u16,
                                       record->count_u16,
                                       record->immediate_i16);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_const_s16_i32_t);
}
iree_vm_bytecode_dispatch_stack_const_s16_i64: {
  const iree_vm_bytecode_stack_const_s16_i64_t* record =
      (const iree_vm_bytecode_stack_const_s16_i64_t*)record_data;
  iree_vm_bytecode_stack_const_s16_i64(local_bytes + record->target_u16,
                                       record->count_u16,
                                       record->immediate_i16);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_const_s16_i64_t);
}
iree_vm_bytecode_dispatch_stack_pack_i32_u16_x2: {
  const iree_vm_bytecode_stack_pack_i32_u16_x2_t* record =
      (const iree_vm_bytecode_stack_pack_i32_u16_x2_t*)record_data;
  iree_vm_bytecode_stack_pack_i32(local_bytes + record->target_u16,
                                  record->immediates_u16, 2);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_pack_i32_u16_x2_t);
}
iree_vm_bytecode_dispatch_stack_pack_i32_u16_x4: {
  const iree_vm_bytecode_stack_pack_i32_u16_x4_t* record =
      (const iree_vm_bytecode_stack_pack_i32_u16_x4_t*)record_data;
  iree_vm_bytecode_stack_pack_i32(local_bytes + record->target_u16,
                                  record->immediates_u16, 4);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_pack_i32_u16_x4_t);
}
iree_vm_bytecode_dispatch_stack_pack_i32_u16_x8: {
  const iree_vm_bytecode_stack_pack_i32_u16_x8_t* record =
      (const iree_vm_bytecode_stack_pack_i32_u16_x8_t*)record_data;
  iree_vm_bytecode_stack_pack_i32(local_bytes + record->target_u16,
                                  record->immediates_u16, 8);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_pack_i32_u16_x8_t);
}
iree_vm_bytecode_dispatch_stack_pack_i64_u32_x2: {
  const iree_vm_bytecode_stack_pack_i64_u32_x2_t* record =
      (const iree_vm_bytecode_stack_pack_i64_u32_x2_t*)record_data;
  iree_vm_bytecode_stack_pack_i64(local_bytes + record->target_u16,
                                  record->immediates_u32, 2);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_pack_i64_u32_x2_t);
}
iree_vm_bytecode_dispatch_stack_pack_i64_u32_x4: {
  const iree_vm_bytecode_stack_pack_i64_u32_x4_t* record =
      (const iree_vm_bytecode_stack_pack_i64_u32_x4_t*)record_data;
  iree_vm_bytecode_stack_pack_i64(local_bytes + record->target_u16,
                                  record->immediates_u32, 4);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_pack_i64_u32_x4_t);
}
iree_vm_bytecode_dispatch_stack_pack_i64_u32_x8: {
  const iree_vm_bytecode_stack_pack_i64_u32_x8_t* record =
      (const iree_vm_bytecode_stack_pack_i64_u32_x8_t*)record_data;
  iree_vm_bytecode_stack_pack_i64(local_bytes + record->target_u16,
                                  record->immediates_u32, 8);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_stack_pack_i64_u32_x8_t);
}

iree_vm_bytecode_dispatch_value_abi_argument_load: {
  const iree_vm_bytecode_value_abi_argument_load_t* record =
      (const iree_vm_bytecode_value_abi_argument_load_t*)record_data;
  values[record->destination_v8] =
      call->value_arguments.overflow[record->slot_u16];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_value_abi_argument_load_t);
}
iree_vm_bytecode_dispatch_value_abi_result_store: {
  const iree_vm_bytecode_value_abi_result_store_t* record =
      (const iree_vm_bytecode_value_abi_result_store_t*)record_data;
  call->value_results.overflow[record->slot_u16] = values[record->source_v8];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_value_abi_result_store_t);
}
iree_vm_bytecode_dispatch_ref_abi_argument_load_borrow: {
  const iree_vm_bytecode_ref_abi_argument_load_borrow_t* record =
      (const iree_vm_bytecode_ref_abi_argument_load_borrow_t*)record_data;
  iree_vm_call_ref_argument_load_borrow(
      call, record->slot_u16 + IREE_VM_CALL_DIRECT_REGISTER_COUNT,
      &refs[record->destination_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(
      iree_vm_bytecode_ref_abi_argument_load_borrow_t);
}
iree_vm_bytecode_dispatch_ref_abi_argument_load_move: {
  const iree_vm_bytecode_ref_abi_argument_load_move_t* record =
      (const iree_vm_bytecode_ref_abi_argument_load_move_t*)record_data;
  iree_vm_call_ref_argument_load_move(
      call, record->slot_u16 + IREE_VM_CALL_DIRECT_REGISTER_COUNT,
      &refs[record->destination_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_abi_argument_load_move_t);
}
iree_vm_bytecode_dispatch_ref_abi_result_store_move: {
  const iree_vm_bytecode_ref_abi_result_store_move_t* record =
      (const iree_vm_bytecode_ref_abi_result_store_move_t*)record_data;
  iree_vm_call_ref_result_store_move(
      call, record->slot_u16 + IREE_VM_CALL_DIRECT_REGISTER_COUNT,
      &refs[record->source_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_abi_result_store_move_t);
}
iree_vm_bytecode_dispatch_func_abi_argument_load: {
  const iree_vm_bytecode_func_abi_argument_load_t* record =
      (const iree_vm_bytecode_func_abi_argument_load_t*)record_data;
  functions[record->destination_f8] =
      call->function_arguments.overflow[record->slot_u16];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_abi_argument_load_t);
}
iree_vm_bytecode_dispatch_func_abi_result_store: {
  const iree_vm_bytecode_func_abi_result_store_t* record =
      (const iree_vm_bytecode_func_abi_result_store_t*)record_data;
  call->function_results.overflow[record->slot_u16] =
      functions[record->source_f8];
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_func_abi_result_store_t);
}

iree_vm_bytecode_dispatch_ref_null: {
  const iree_vm_bytecode_ref_null_t* record =
      (const iree_vm_bytecode_ref_null_t*)record_data;
  iree_vm_ref_reset(&refs[record->destination_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_null_t);
}
iree_vm_bytecode_dispatch_ref_compare_null: {
  const iree_vm_bytecode_ref_compare_null_t* record =
      (const iree_vm_bytecode_ref_compare_null_t*)record_data;
  values[record->destination_v8] = iree_vm_ref_is_null(refs[record->source_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_compare_null_t);
}
iree_vm_bytecode_dispatch_ref_compare_eq: {
  const iree_vm_bytecode_ref_compare_eq_t* record =
      (const iree_vm_bytecode_ref_compare_eq_t*)record_data;
  const iree_vm_ref_t left = refs[record->left_r8];
  const iree_vm_ref_t right = refs[record->right_r8];
  values[record->destination_v8] =
      left.object == right.object &&
      (!left.object || iree_vm_ref_type(left) == iree_vm_ref_type(right));
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_compare_eq_t);
}
iree_vm_bytecode_dispatch_ref_retain: {
  const iree_vm_bytecode_ref_retain_t* record =
      (const iree_vm_bytecode_ref_retain_t*)record_data;
  iree_vm_bytecode_ref_retain(&refs[record->destination_r8],
                              refs[record->source_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_retain_t);
}
iree_vm_bytecode_dispatch_ref_move: {
  const iree_vm_bytecode_ref_move_t* record =
      (const iree_vm_bytecode_ref_move_t*)record_data;
  iree_vm_bytecode_ref_move(&refs[record->destination_r8],
                            &refs[record->source_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_move_t);
}
iree_vm_bytecode_dispatch_ref_discard: {
  const iree_vm_bytecode_ref_discard_t* record =
      (const iree_vm_bytecode_ref_discard_t*)record_data;
  iree_vm_ref_reset(&refs[record->source_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_discard_t);
}
iree_vm_bytecode_dispatch_ref_stack_load_retain: {
  const iree_vm_bytecode_ref_stack_load_retain_t* record =
      (const iree_vm_bytecode_ref_stack_load_retain_t*)record_data;
  iree_vm_bytecode_ref_retain(&refs[record->destination_r8],
                              local_refs[record->slot_u16]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_stack_load_retain_t);
}
iree_vm_bytecode_dispatch_ref_stack_load_move: {
  const iree_vm_bytecode_ref_stack_load_move_t* record =
      (const iree_vm_bytecode_ref_stack_load_move_t*)record_data;
  iree_vm_bytecode_ref_move(&refs[record->destination_r8],
                            &local_refs[record->slot_u16]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_stack_load_move_t);
}
iree_vm_bytecode_dispatch_ref_stack_store_retain: {
  const iree_vm_bytecode_ref_stack_store_retain_t* record =
      (const iree_vm_bytecode_ref_stack_store_retain_t*)record_data;
  iree_vm_bytecode_ref_retain(&local_refs[record->slot_u16],
                              refs[record->source_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_stack_store_retain_t);
}
iree_vm_bytecode_dispatch_ref_stack_store_move: {
  const iree_vm_bytecode_ref_stack_store_move_t* record =
      (const iree_vm_bytecode_ref_stack_store_move_t*)record_data;
  iree_vm_bytecode_ref_move(&local_refs[record->slot_u16],
                            &refs[record->source_r8]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_stack_store_move_t);
}
iree_vm_bytecode_dispatch_ref_stack_discard: {
  const iree_vm_bytecode_ref_stack_discard_t* record =
      (const iree_vm_bytecode_ref_stack_discard_t*)record_data;
  iree_vm_ref_reset(&local_refs[record->slot_u16]);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_ref_stack_discard_t);
}

iree_vm_bytecode_dispatch_buffer_allocate: {
  const iree_vm_bytecode_buffer_allocate_t* record =
      (const iree_vm_bytecode_buffer_allocate_t*)record_data;
  const uint64_t length = values[record->length_v8];
  if (length > IREE_HOST_SIZE_MAX ||
      record->minimum_alignment_log2_u8 >= sizeof(iree_host_size_t) * 8u) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "vm.buffer allocation is not host-representable");
  }
  const iree_host_size_t minimum_alignment =
      (iree_host_size_t)1u << record->minimum_alignment_log2_u8;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      (iree_host_size_t)length, minimum_alignment,
      iree_vm_invocation_host_allocator(invocation), &buffer));
  iree_vm_bytecode_ref_replace(
      &refs[record->destination_r8],
      (iree_vm_ref_t){buffer, (uintptr_t)image->buffer_type});
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_allocate_t);
}
iree_vm_bytecode_dispatch_buffer_length: {
  const iree_vm_bytecode_buffer_length_t* record =
      (const iree_vm_bytecode_buffer_length_t*)record_data;
  const iree_vm_ref_t buffer_ref = refs[record->buffer_r8];
  if (iree_vm_ref_is_null(buffer_ref)) {
    values[record->destination_v8] = 0;
  } else {
    iree_vm_buffer_t* buffer = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
        buffer_ref, image->buffer_type, &buffer));
    values[record->destination_v8] = iree_vm_buffer_length(buffer);
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_length_t);
}
iree_vm_bytecode_dispatch_buffer_subspan: {
  const iree_vm_bytecode_buffer_subspan_t* record =
      (const iree_vm_bytecode_buffer_subspan_t*)record_data;
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &source));
  const uint64_t offset = values[record->offset_v8];
  const uint64_t length = values[record->length_v8];
  if (offset > IREE_HOST_SIZE_MAX || length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "vm.buffer subspan range is not host-representable");
  }
  const iree_vm_buffer_access_flags_t access = iree_vm_buffer_access(source);
  if (access == IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "vm.buffer source is closed");
  }
  iree_vm_buffer_t* result = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_subspan(
      source, (iree_host_size_t)offset, (iree_host_size_t)length, access,
      iree_vm_invocation_host_allocator(invocation), &result));
  iree_vm_bytecode_ref_replace(
      &refs[record->destination_r8],
      (iree_vm_ref_t){result, (uintptr_t)image->buffer_type});
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_subspan_t);
}
iree_vm_bytecode_dispatch_buffer_load: {
  const iree_vm_bytecode_buffer_load_t* record =
      (const iree_vm_bytecode_buffer_load_t*)record_data;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &buffer));
  iree_byte_span_t source = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_lanes(
      buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, values[record->base_v8],
      values[record->index_v8], record->scale_u8,
      iree_vm_bytecode_memory_access_length(record->format_u8), &source));
  iree_vm_bytecode_memory_load_lanes(record->format_u8, source.data,
                                     values + record->destination_v8);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_load_t);
}
iree_vm_bytecode_dispatch_buffer_store: {
  const iree_vm_bytecode_buffer_store_t* record =
      (const iree_vm_bytecode_buffer_store_t*)record_data;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &buffer));
  iree_byte_span_t target = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_lanes(
      buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, values[record->base_v8],
      values[record->index_v8], record->scale_u8,
      iree_vm_bytecode_memory_access_length(record->format_u8), &target));
  iree_vm_bytecode_memory_store_lanes(record->format_u8,
                                      values + record->source_v8, target.data);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_store_t);
}
iree_vm_bytecode_dispatch_buffer_atomic_reduce: {
  const iree_vm_bytecode_buffer_atomic_reduce_t* record =
      (const iree_vm_bytecode_buffer_atomic_reduce_t*)record_data;
  const uint64_t offset = values[record->offset_v8];
  const uint64_t operand_bits = values[record->operand_v8];
  const iree_vm_bytecode_buffer_atomic_kind_t kind =
      record->selector0_u8 & 0x0Fu;
  const iree_vm_bytecode_buffer_atomic_carrier_t carrier =
      record->selector0_u8 >> 7;
  const iree_vm_bytecode_buffer_atomic_ordering_t ordering =
      record->selector1_u8 & 0x07u;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &buffer));
  uint8_t* address = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_atomic(
      buffer, offset, (iree_host_size_t)4u << carrier, &address));
  (void)iree_vm_bytecode_atomic_apply(address, operand_bits, kind, carrier,
                                      ordering);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_atomic_reduce_t);
}
iree_vm_bytecode_dispatch_buffer_atomic_rmw: {
  const iree_vm_bytecode_buffer_atomic_rmw_t* record =
      (const iree_vm_bytecode_buffer_atomic_rmw_t*)record_data;
  const uint64_t offset = values[record->offset_v8];
  const uint64_t operand_bits = values[record->operand_v8];
  const iree_vm_bytecode_buffer_atomic_kind_t kind =
      record->selector0_u8 & 0x0Fu;
  const iree_vm_bytecode_buffer_atomic_carrier_t carrier =
      record->selector0_u8 >> 7;
  const iree_vm_bytecode_buffer_atomic_ordering_t ordering =
      record->selector1_u8 & 0x07u;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &buffer));
  uint8_t* address = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_atomic(
      buffer, offset, (iree_host_size_t)4u << carrier, &address));
  values[record->old_v8] = iree_vm_bytecode_atomic_apply(
      address, operand_bits, kind, carrier, ordering);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_atomic_rmw_t);
}
iree_vm_bytecode_dispatch_buffer_atomic_cmpxchg: {
  const iree_vm_bytecode_buffer_atomic_cmpxchg_t* record =
      (const iree_vm_bytecode_buffer_atomic_cmpxchg_t*)record_data;
  const uint64_t offset = values[record->offset_v8];
  const uint64_t expected_bits = values[record->expected_v8];
  const uint64_t replacement_bits = values[record->replacement_v8];
  const iree_vm_bytecode_buffer_atomic_carrier_t carrier =
      record->selector0_u8 >> 7;
  const iree_vm_bytecode_buffer_atomic_ordering_t success_ordering =
      record->selector0_u8 & 0x07u;
  const iree_vm_bytecode_buffer_atomic_ordering_t failure_ordering =
      (record->selector0_u8 >> 3) & 0x07u;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &buffer));
  uint8_t* address = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_atomic(
      buffer, offset, (iree_host_size_t)4u << carrier, &address));
  values[record->old_v8] = iree_vm_bytecode_atomic_compare_exchange(
      address, expected_bits, replacement_bits, carrier, success_ordering,
      failure_ordering);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_atomic_cmpxchg_t);
}
iree_vm_bytecode_dispatch_buffer_fill: {
  const iree_vm_bytecode_buffer_fill_t* record =
      (const iree_vm_bytecode_buffer_fill_t*)record_data;
  const uint64_t offset = values[record->offset_v8];
  const uint64_t length = values[record->length_v8];
  const uint64_t pattern = values[record->pattern_v8];
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->buffer_r8], image->buffer_type, &buffer));
  iree_byte_span_t target = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, offset, length, &target));
  if (length != 0) {
    iree_vm_bytecode_fill_pattern(target.data, target.data_length, pattern,
                                  record->pattern_width_u8);
  }
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_fill_t);
}
iree_vm_bytecode_dispatch_buffer_copy: {
  const iree_vm_bytecode_buffer_copy_t* record =
      (const iree_vm_bytecode_buffer_copy_t*)record_data;
  const uint64_t target_offset = values[record->target_offset_v8];
  const uint64_t source_offset = values[record->source_offset_v8];
  const uint64_t length = values[record->length_v8];
  iree_vm_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->target_r8], image->buffer_type, &target_buffer));
  iree_vm_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->source_r8], image->buffer_type, &source_buffer));
  iree_byte_span_t target = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      target_buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, target_offset, length,
      &target));
  iree_byte_span_t source = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      source_buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, source_offset, length,
      &source));
  if (length != 0) memmove(target.data, source.data, target.data_length);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_copy_t);
}
iree_vm_bytecode_dispatch_buffer_compare: {
  const iree_vm_bytecode_buffer_compare_t* record =
      (const iree_vm_bytecode_buffer_compare_t*)record_data;
  const uint64_t left_offset = values[record->left_offset_v8];
  const uint64_t right_offset = values[record->right_offset_v8];
  const uint64_t length = values[record->length_v8];
  iree_vm_buffer_t* left_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->left_r8], image->buffer_type, &left_buffer));
  iree_vm_buffer_t* right_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->right_r8], image->buffer_type, &right_buffer));
  iree_byte_span_t left = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      left_buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, left_offset, length,
      &left));
  iree_byte_span_t right = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      right_buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, right_offset, length,
      &right));
  const int ordering =
      length == 0 ? 0 : memcmp(left.data, right.data, left.data_length);
  values[record->destination_v8] = ordering < 0   ? UINT32_MAX
                                   : ordering > 0 ? UINT32_C(1)
                                                  : UINT32_C(0);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_compare_t);
}
iree_vm_bytecode_dispatch_buffer_copy_rodata: {
  const iree_vm_bytecode_buffer_copy_rodata_t* record =
      (const iree_vm_bytecode_buffer_copy_rodata_t*)record_data;
  const uint64_t target_offset = values[record->target_offset_v8];
  const uint64_t length = values[record->length_v8];
  iree_vm_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_check_deref(
      refs[record->target_r8], image->buffer_type, &target_buffer));
  iree_byte_span_t target = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      target_buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, target_offset, length,
      &target));
  iree_byte_span_t source = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_buffer_map_range(
      &image->rodata_roots[record->rodata_u16], IREE_VM_BUFFER_ACCESS_FLAG_READ,
      record->source_offset_u32, length, &source));
  if (length != 0) memcpy(target.data, source.data, target.data_length);
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_copy_rodata_t);
}
iree_vm_bytecode_dispatch_buffer_rodata_load: {
  const iree_vm_bytecode_buffer_rodata_load_t* record =
      (const iree_vm_bytecode_buffer_rodata_load_t*)record_data;
  iree_vm_bytecode_ref_replace(
      &refs[record->destination_r8],
      iree_vm_ref_from_ptr_borrowed(&image->rodata_roots[record->rodata_u16],
                                    image->buffer_type));
  IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_bytecode_buffer_rodata_load_t);
}

iree_vm_bytecode_dispatch_invalid:
  IREE_ASSERT_UNREACHABLE("verified bytecode contains an invalid opcode");
  IREE_BUILTIN_UNREACHABLE();
}

#undef IREE_VM_BYTECODE_DISPATCH_NEXT
#undef IREE_VM_BYTECODE_DISPATCH
#undef IREE_VM_BYTECODE_DISPATCH_BEGIN
#if !defined(IREE_PLATFORM_WASM) && !defined(IREE_COMPILER_MSVC_COMPAT) && \
    (defined(IREE_COMPILER_CLANG) || defined(IREE_COMPILER_GCC))
#undef IREE_VM_BYTECODE_DISPATCH_TABLE_ENTRY
#else
#undef IREE_VM_BYTECODE_DISPATCH_SWITCH_CASE
#endif  // computed goto dispatch

iree_status_t iree_vm_bytecode_interpreter_start(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(module);
  const iree_vm_bytecode_v0_function_row_t* function =
      &image->layout.functions.rows[params->function_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      iree_vm_bytecode_function_signature(&image->layout, function);
  const uint8_t* program_counter =
      image->layout.functions.bytecode_data + function->bytecode_offset_u32;

  if (IREE_LIKELY(
          iree_vm_bytecode_function_is_scalar_leaf(function, signature))) {
    iree_vm_bytecode_execution_state_t state = {
        .call = params->call,
        .program_counter = program_counter,
        .values = params->call.value_results.direct,
    };
    iree_vm_bytecode_frame_copy_direct_values(
        state.values, params->call.value_arguments.direct,
        signature->argument_value_count_u16);
    return iree_vm_bytecode_dispatch(image, function, &params->execution,
                                     &state, out_outcome);
  }

  const iree_vm_bytecode_frame_layout_t layout =
      iree_vm_bytecode_calculate_frame_layout(function);
  iree_vm_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_invocation_push_frame(
      params, layout.frame, iree_vm_bytecode_frame_cleanup, &frame));
  iree_vm_bytecode_execution_state_t* state =
      iree_vm_bytecode_map_execution_state(frame, layout, params->call,
                                           program_counter);
  iree_vm_bytecode_frame_initialize(function, signature, state);
  iree_status_t status = iree_vm_bytecode_dispatch(
      image, function, &params->execution, state, out_outcome);
  if (iree_status_is_ok(status) &&
      *out_outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    iree_vm_invocation_pop_frame(params->execution.invocation, frame);
  }
  return status;
}

iree_status_t iree_vm_bytecode_interpreter_resume(
    iree_vm_module_t* module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  IREE_RETURN_IF_ERROR(
      iree_vm_invocation_check_cancelled(params->execution.invocation));

  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(module);
  const uint16_t function_ordinal =
      iree_vm_frame_function_ordinal(params->frame);
  const iree_vm_bytecode_v0_function_row_t* function =
      &image->layout.functions.rows[function_ordinal];
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)iree_vm_frame_storage(params->frame);
  iree_vm_bytecode_call_cleanup_completed(state);
  iree_status_t status = iree_vm_bytecode_dispatch(
      image, function, &params->execution, state, out_outcome);
  if (iree_status_is_ok(status) &&
      *out_outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    iree_vm_invocation_pop_frame(params->execution.invocation, params->frame);
  }
  return status;
}
