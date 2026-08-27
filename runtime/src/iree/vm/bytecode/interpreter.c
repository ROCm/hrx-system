// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter.h"

#include <string.h>

#include "iree/vm/bytecode/interpreter_atomic.h"
#include "iree/vm/bytecode/interpreter_buffer.h"
#include "iree/vm/bytecode/interpreter_call.h"
#include "iree/vm/bytecode/interpreter_conversion.h"
#include "iree/vm/bytecode/interpreter_float.h"
#include "iree/vm/bytecode/interpreter_float_math.h"
#include "iree/vm/bytecode/interpreter_frame.h"
#include "iree/vm/bytecode/interpreter_integer.h"
#include "iree/vm/bytecode/interpreter_stack.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/wire/core/abi.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/float.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/ref.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/core/stack.h"
#include "iree/vm/bytecode/wire/core/value.h"
#include "iree/vm/invocation_storage.h"

#define IREE_VM_BYTECODE_DEFINE_EXECUTABLE_OPCODE_LIST
#include "iree/vm/bytecode/execution_tables.inl"
#undef IREE_VM_BYTECODE_DEFINE_EXECUTABLE_OPCODE_LIST

#define IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(name)            \
  IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_ISA_CONTROL_STATUS_##name, \
                             IREE_STATUS_##name,                \
                             "VM control status must match iree_status_t")
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(CANCELLED);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(UNKNOWN);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(INVALID_ARGUMENT);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(DEADLINE_EXCEEDED);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(NOT_FOUND);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(ALREADY_EXISTS);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(PERMISSION_DENIED);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(RESOURCE_EXHAUSTED);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(FAILED_PRECONDITION);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(ABORTED);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(OUT_OF_RANGE);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(UNIMPLEMENTED);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(INTERNAL);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(UNAVAILABLE);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(DATA_LOSS);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(UNAUTHENTICATED);
IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS(INCOMPATIBLE);
#undef IREE_VM_BYTECODE_ASSERT_CONTROL_STATUS

// Clang and GCC labels-as-values remove the shared loop branch and let each
// opcode site predict its own successor. Other compilers retain the portable
// switch loop over the same handler bodies.
#if !defined(IREE_PLATFORM_WASM) && \
    (defined(IREE_COMPILER_CLANG) || defined(IREE_COMPILER_GCC))
#define IREE_VM_BYTECODE_DISPATCH_TABLE_ENTRY(opcode, label) \
  [IREE_VM_ISA_CORE_OPCODE_##opcode] = &&iree_vm_bytecode_dispatch_##label,
#define IREE_VM_BYTECODE_DISPATCH_BEGIN()          \
  static const void* const dispatch_table[256] = { \
      IREE_VM_BYTECODE_EXECUTABLE_OPCODE_LIST(     \
          IREE_VM_BYTECODE_DISPATCH_TABLE_ENTRY)}; \
  const uint8_t* record_data = program_counter;    \
  do {                                             \
  goto* dispatch_table[record_data[0]]
#define IREE_VM_BYTECODE_DISPATCH_CASE(opcode, label) \
  iree_vm_bytecode_dispatch_##label:
#define IREE_VM_BYTECODE_DISPATCH_NEXT(record_type) \
  record_data += sizeof(record_type);               \
  goto* dispatch_table[record_data[0]]
#define IREE_VM_BYTECODE_DISPATCH_IS_SAME(opcode) \
  (record_data[0] == IREE_VM_ISA_CORE_OPCODE_##opcode)
#define IREE_VM_BYTECODE_DISPATCH_CONTINUE() \
  goto* dispatch_table[record_data[0]]
#define IREE_VM_BYTECODE_DISPATCH_TERMINATE() break
#define IREE_VM_BYTECODE_DISPATCH_END()                                     \
  status =                                                                  \
      iree_make_status(IREE_STATUS_INTERNAL,                                \
                       "verified bytecode fell through its function body"); \
  }                                                                         \
  while (false)
#else
#define IREE_VM_BYTECODE_DISPATCH_BEGIN()                      \
  const uint8_t* record_data = program_counter;                \
  const uint8_t* bytecode_end =                                \
      function_bytecode + function->bytecode_length_u32;       \
  bool dispatch_terminated = false;                            \
  while (!dispatch_terminated && record_data < bytecode_end) { \
    switch (record_data[0]) {
#define IREE_VM_BYTECODE_DISPATCH_CASE(opcode, label) \
  case IREE_VM_ISA_CORE_OPCODE_##opcode:
#define IREE_VM_BYTECODE_DISPATCH_NEXT(record_type) \
  record_data += sizeof(record_type);               \
  break
#define IREE_VM_BYTECODE_DISPATCH_IS_SAME(opcode) false
#define IREE_VM_BYTECODE_DISPATCH_CONTINUE() break
#define IREE_VM_BYTECODE_DISPATCH_TERMINATE() \
  dispatch_terminated = true;                 \
  break
#define IREE_VM_BYTECODE_DISPATCH_END()                                        \
  default:                                                                     \
    status = iree_make_status(IREE_STATUS_INTERNAL,                            \
                              "verified bytecode contains an unknown opcode"); \
    dispatch_terminated = true;                                                \
    break;                                                                     \
    }                                                                          \
    }                                                                          \
    if (!dispatch_terminated) {                                                \
      status = iree_make_status(                                               \
          IREE_STATUS_INTERNAL,                                                \
          "verified bytecode fell through its function body");                 \
    }
#endif  // computed goto dispatch

// Replaces |target| with an already-safe complete ref state. Installing the
// new state before releasing the old owner keeps self-replacement safe.
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

// Transfers one internal null, borrowed, or owned state exactly. Unlike the
// public escaping move, an internal borrowed state remains borrowed.
static inline void iree_vm_bytecode_ref_move(iree_vm_ref_t* target,
                                             iree_vm_ref_t* source) {
  iree_vm_ref_t moved_ref = *source;
  *source = iree_vm_ref_null();
  iree_vm_bytecode_ref_replace(target, moved_ref);
}

// Resolves one verified signed word displacement from the record end.
static inline const uint8_t* iree_vm_bytecode_direct_target(
    const uint8_t* record_data, iree_host_size_t record_length,
    int32_t target_word_delta) {
  return record_data + record_length +
         (ptrdiff_t)((int64_t)target_word_delta * 4);
}

// Materializes the terminal status for an integer division failure off the
// successful instruction path.
IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
iree_vm_bytecode_make_integer_division_status(
    iree_vm_bytecode_integer_division_failure_t failure) {
  switch (failure) {
    case IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "integer division by zero");
    case IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW:
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "signed integer division overflow");
    case IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "invalid integer division failure");
}

// Captures an optional readable vm.buffer message before frame unwind. Invalid
// diagnostic state never replaces the primary architectural failure code.
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
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
  return iree_status_allocate_copy(code, iree_make_cstring_view(__FILE__),
                                   __LINE__, message);
}

// Returns whether |ref| satisfies one resolved ref-global descriptor.
static inline bool iree_vm_bytecode_ref_matches_global(
    iree_vm_ref_t ref, iree_vm_ref_type_t expected_type,
    const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor) {
  if (iree_vm_ref_is_null(ref)) {
    return iree_any_bit_set(descriptor->flags_u16,
                            IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE);
  }
  return iree_vm_ref_type(ref) == expected_type;
}

// Returns whether |function_ref| satisfies one function-global descriptor.
static inline bool iree_vm_bytecode_function_matches_global(
    const iree_vm_program_t* program, iree_vm_function_ref_t function_ref,
    const iree_vm_linked_module_t* signature_module,
    const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor) {
  if (iree_vm_function_ref_is_null(function_ref)) {
    return iree_any_bit_set(descriptor->flags_u16,
                            IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE);
  }
  return iree_vm_program_function_ref_matches(
      program, function_ref, signature_module,
      descriptor->callable_type_ordinal_u16);
}

static void iree_vm_bytecode_frame_reset(iree_vm_ref_t* refs,
                                         uint32_t ref_count) {
  for (uint32_t i = 0; i < ref_count; ++i) {
    iree_vm_ref_reset(&refs[i]);
  }
}

// Copies one direct physical value bank in fixed native-vector-sized chunks.
// Direct banks have a fixed small bound, so this remains a bounded sequence
// with no variable-length memcpy call or per-cell bounds branch.
static inline void iree_vm_bytecode_copy_direct_values(uint64_t* target,
                                                       const uint64_t* source,
                                                       uint16_t count) {
  while (count >= 4) {
    memcpy(target, source, 4 * sizeof(uint64_t));
    target += 4;
    source += 4;
    count -= 4;
  }
  if (count >= 2) {
    memcpy(target, source, 2 * sizeof(uint64_t));
    target += 2;
    source += 2;
    count -= 2;
  }
  if (count) {
    *target = *source;
  }
}

static void iree_vm_bytecode_initialize_state(
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
  iree_vm_bytecode_copy_direct_values(
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

static iree_status_t iree_vm_bytecode_publish_results(
    const iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_module_execution_t* execution,
    const iree_vm_call_packet_t* call, uint64_t* values, iree_vm_ref_t* refs,
    iree_vm_function_ref_t* functions) {
  const iree_vm_bytecode_v0_signature_row_t* signature =
      iree_vm_bytecode_function_signature(&module->layout, function);
  if (signature->result_ref_count_u16 || signature->result_function_count_u16) {
    const uint32_t argument_count =
        iree_vm_bytecode_signature_argument_count(signature);
    const uint32_t result_count =
        iree_vm_bytecode_signature_result_count(signature);
    const iree_vm_bytecode_v0_signature_descriptor_row_t* result_descriptors =
        iree_vm_bytecode_signature_descriptors(
            &module->layout.signatures,
            iree_vm_bytecode_function_callable_type(&module->layout, function)
                ->signature_ordinal_u16) +
        argument_count;
    uint16_t ref_ordinal = 0;
    uint16_t function_ordinal = 0;
    for (uint32_t i = 0; i < result_count; ++i) {
      const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor =
          &result_descriptors[i];
      if (descriptor->kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
        const iree_vm_ref_t result_ref =
            ref_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
                ? refs[ref_ordinal]
                : call->ref_results
                      .overflow[ref_ordinal -
                                IREE_VM_CALL_DIRECT_REGISTER_COUNT];
        if (result_ref.object &&
            iree_vm_ref_type(result_ref) !=
                module->resolved_ref_types[descriptor->type_ordinal_u16]) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "bytecode function returned the wrong ref type");
        }
        ++ref_ordinal;
      } else if (descriptor->kind_u16 ==
                 IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
        const iree_vm_function_ref_t result_function =
            function_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
                ? functions[function_ordinal]
                : call->function_results
                      .overflow[function_ordinal -
                                IREE_VM_CALL_DIRECT_REGISTER_COUNT];
        if (!iree_vm_program_function_ref_matches(
                execution->invocation->process->program, result_function,
                execution->linked_module, descriptor->type_ordinal_u16)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "bytecode function returned an incompatible function value");
        }
        ++function_ordinal;
      }
    }
  }

  const uint16_t direct_value_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature->result_value_count_u16);
  if (call->value_results.direct != values) {
    iree_vm_bytecode_copy_direct_values(call->value_results.direct, values,
                                        direct_value_count);
  }
  const uint16_t direct_ref_count = iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                                             signature->result_ref_count_u16);
  for (uint16_t i = 0; i < direct_ref_count; ++i) {
    iree_vm_call_ref_result_store_move(call, i, &refs[i]);
  }
  const uint16_t direct_function_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature->result_function_count_u16);
  if (direct_function_count != 0 &&
      call->function_results.direct != functions) {
    memcpy(call->function_results.direct, functions,
           direct_function_count * sizeof(*functions));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_execute(
    iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_t* invocation = execution->invocation;
  const iree_vm_call_packet_t* call = state->call;
  uint64_t* values = state->values;
  iree_vm_ref_t* refs = state->refs;
  iree_vm_ref_t* local_refs = state->local_refs;
  iree_vm_function_ref_t* functions = state->functions;
  iree_vm_function_ref_t* local_functions = state->local_functions;
  uint8_t* local_bytes = state->local_bytes;
  void* process_storage = execution->process_storage;
  uint64_t* global_values = NULL;
  iree_vm_ref_t* global_refs = NULL;
  iree_vm_function_ref_t* global_functions = NULL;
  uint64_t* global_value_set_bits = NULL;
  uint64_t* global_ref_set_bits = NULL;
  uint64_t* global_function_set_bits = NULL;
  iree_vm_bytecode_process_state_t* process_state = NULL;
  if (module->layout.globals.header) {
    global_values = iree_vm_bytecode_process_values(module, process_storage);
    global_refs = iree_vm_bytecode_process_refs(module, process_storage);
    global_functions =
        iree_vm_bytecode_process_functions(module, process_storage);
    global_value_set_bits =
        iree_vm_bytecode_process_value_set_bits(module, process_storage);
    global_ref_set_bits =
        iree_vm_bytecode_process_ref_set_bits(module, process_storage);
    global_function_set_bits =
        iree_vm_bytecode_process_function_set_bits(module, process_storage);
    process_state = iree_vm_bytecode_process_state(process_storage);
  }
  const iree_vm_bytecode_v0_constant_cell_t* constant_cells =
      module->layout.constants.cells;
  const uint8_t* function_bytecode =
      module->layout.functions.bytecode_data + function->bytecode_offset_u32;
  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets =
      module->layout.functions.switch_targets +
      function->switch_target_base_u32;
  const uint8_t* program_counter = state->program_counter;
  iree_status_t status = iree_ok_status();

  IREE_VM_BYTECODE_DISPATCH_BEGIN();
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BLOCK, control_block) {
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_control_block_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_RETURN, control_return) {
    status = iree_vm_bytecode_publish_results(module, function, execution, call,
                                              values, refs, functions);
    if (iree_status_is_ok(status)) {
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    }
    IREE_VM_BYTECODE_DISPATCH_TERMINATE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_YIELD_S32, control_yield_s32) {
    const iree_vm_isa_control_yield_s32_record_t* record =
        (const iree_vm_isa_control_yield_s32_record_t*)record_data;
    if (!state->frame) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "bytecode function yielded without a durable frame");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    if (invocation->has_external_borrowed_arguments) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "bytecode function yielded while external borrowed refs were live");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    state->program_counter = iree_vm_bytecode_direct_target(
        record_data, sizeof(*record), record->target_rel32);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
    const iree_vm_invocation_wake_callback_t wake_callback =
        iree_vm_invocation_wake_callback(invocation);
    if (wake_callback.function) {
      wake_callback.function(wake_callback.user_data);
    }
    IREE_VM_BYTECODE_DISPATCH_TERMINATE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BRANCH_S16, control_branch_s16) {
    const iree_vm_isa_control_branch_s16_record_t* record =
        (const iree_vm_isa_control_branch_s16_record_t*)record_data;
    record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                                 record->target_rel16);
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BRANCH_S32, control_branch_s32) {
    const iree_vm_isa_control_branch_s32_record_t* record =
        (const iree_vm_isa_control_branch_s32_record_t*)record_data;
    record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                                 record->target_rel32);
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BRANCH_IF_S16, control_branch_if_s16) {
    const iree_vm_isa_control_branch_if_s16_record_t* record =
        (const iree_vm_isa_control_branch_if_s16_record_t*)record_data;
    if (values[record->condition_v8] != 0) {
      record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                                   record->target_rel16);
      IREE_VM_BYTECODE_DISPATCH_CONTINUE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_control_branch_if_s16_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BRANCH_IF_S32, control_branch_if_s32) {
    const iree_vm_isa_control_branch_if_s32_record_t* record =
        (const iree_vm_isa_control_branch_if_s32_record_t*)record_data;
    if (values[record->condition_v8] != 0) {
      record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                                   record->target_rel32);
      IREE_VM_BYTECODE_DISPATCH_CONTINUE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_control_branch_if_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BRANCH_UNLESS_S16,
                                 control_branch_unless_s16) {
    const iree_vm_isa_control_branch_unless_s16_record_t* record =
        (const iree_vm_isa_control_branch_unless_s16_record_t*)record_data;
    if (values[record->condition_v8] == 0) {
      record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                                   record->target_rel16);
      IREE_VM_BYTECODE_DISPATCH_CONTINUE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_control_branch_unless_s16_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BRANCH_UNLESS_S32,
                                 control_branch_unless_s32) {
    const iree_vm_isa_control_branch_unless_s32_record_t* record =
        (const iree_vm_isa_control_branch_unless_s32_record_t*)record_data;
    if (values[record->condition_v8] == 0) {
      record_data = iree_vm_bytecode_direct_target(record_data, sizeof(*record),
                                                   record->target_rel32);
      IREE_VM_BYTECODE_DISPATCH_CONTINUE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_control_branch_unless_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_SWITCH, control_switch) {
    const iree_vm_isa_control_switch_record_t* record =
        (const iree_vm_isa_control_switch_record_t*)record_data;
    const uint64_t selector = values[record->selector_v8];
    if (selector < record->target_count_u16) {
      const uint32_t target_word_offset =
          switch_targets[record->target_base_u32 + (uint32_t)selector];
      record_data =
          function_bytecode + (iree_host_size_t)target_word_offset * 4;
      IREE_VM_BYTECODE_DISPATCH_CONTINUE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_control_switch_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_CALL, control_call) {
    const iree_vm_isa_control_call_record_t* record =
        (const iree_vm_isa_control_call_record_t*)record_data;
    state->program_counter = record_data + sizeof(*record);
    iree_vm_execution_outcome_t call_outcome = UINT32_MAX;
    status = iree_vm_bytecode_call_start_direct(module, execution, state,
                                                record, &call_outcome);
    if (!iree_status_is_ok(status) ||
        call_outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
      if (iree_status_is_ok(status)) {
        *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
      }
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    record_data = state->program_counter;
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_CALL_INDIRECT, control_call_indirect) {
    const iree_vm_isa_control_call_indirect_record_t* record =
        (const iree_vm_isa_control_call_indirect_record_t*)record_data;
    state->program_counter = record_data + sizeof(*record);
    iree_vm_execution_outcome_t call_outcome = UINT32_MAX;
    status = iree_vm_bytecode_call_start_indirect(module, execution, state,
                                                  record, &call_outcome);
    if (!iree_status_is_ok(status) ||
        call_outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
      if (iree_status_is_ok(status)) {
        *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
      }
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    record_data = state->program_counter;
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_ASSERT, control_assert) {
    const iree_vm_isa_control_assert_record_t* record =
        (const iree_vm_isa_control_assert_record_t*)record_data;
    if (IREE_LIKELY(values[record->condition_v8] != 0)) {
      IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_control_assert_record_t);
    }
    status = iree_vm_bytecode_make_program_status(
        IREE_STATUS_FAILED_PRECONDITION, module->buffer_type,
        iree_vm_ref_null());
    IREE_VM_BYTECODE_DISPATCH_TERMINATE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_FAIL, control_fail) {
    const iree_vm_isa_control_fail_record_t* record =
        (const iree_vm_isa_control_fail_record_t*)record_data;
    status = iree_vm_bytecode_make_program_status(
        (iree_status_code_t)record->status_u8, module->buffer_type,
        refs[record->message_r8_nullable]);
    IREE_VM_BYTECODE_DISPATCH_TERMINATE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(VALUE_ABI_ARGUMENT_LOAD,
                                 value_abi_argument_load) {
    const iree_vm_isa_value_abi_argument_load_record_t* record =
        (const iree_vm_isa_value_abi_argument_load_record_t*)record_data;
    values[record->dst_v8] = call->value_arguments.overflow[record->slot_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_value_abi_argument_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(VALUE_ABI_RESULT_STORE,
                                 value_abi_result_store) {
    const iree_vm_isa_value_abi_result_store_record_t* record =
        (const iree_vm_isa_value_abi_result_store_record_t*)record_data;
    call->value_results.overflow[record->slot_u16] = values[record->src_v8];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_value_abi_result_store_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_ABI_ARGUMENT_LOAD_BORROW,
                                 ref_abi_argument_load_borrow) {
    const iree_vm_isa_ref_abi_argument_load_borrow_record_t* record =
        (const iree_vm_isa_ref_abi_argument_load_borrow_record_t*)record_data;
    const iree_vm_ref_t source = call->ref_arguments.overflow[record->slot_u16];
    const iree_vm_ref_t borrowed =
        source.object ? iree_vm_ref_from_ptr_borrowed(source.object,
                                                      iree_vm_ref_type(source))
                      : iree_vm_ref_null();
    iree_vm_bytecode_ref_replace(&refs[record->dst_r8], borrowed);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_ref_abi_argument_load_borrow_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_ABI_ARGUMENT_LOAD_MOVE,
                                 ref_abi_argument_load_move) {
    const iree_vm_isa_ref_abi_argument_load_move_record_t* record =
        (const iree_vm_isa_ref_abi_argument_load_move_record_t*)record_data;
    iree_vm_bytecode_ref_move(&refs[record->dst_r8],
                              &call->ref_arguments.overflow[record->slot_u16]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_ref_abi_argument_load_move_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_ABI_RESULT_STORE_MOVE,
                                 ref_abi_result_store_move) {
    const iree_vm_isa_ref_abi_result_store_move_record_t* record =
        (const iree_vm_isa_ref_abi_result_store_move_record_t*)record_data;
    iree_vm_ref_t new_result = iree_vm_ref_move(&refs[record->src_r8]);
    iree_vm_bytecode_ref_replace(&call->ref_results.overflow[record->slot_u16],
                                 new_result);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_ref_abi_result_store_move_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_ABI_ARGUMENT_LOAD,
                                 func_abi_argument_load) {
    const iree_vm_isa_func_abi_argument_load_record_t* record =
        (const iree_vm_isa_func_abi_argument_load_record_t*)record_data;
    functions[record->dst_f8] =
        call->function_arguments.overflow[record->slot_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_abi_argument_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_ABI_RESULT_STORE, func_abi_result_store) {
    const iree_vm_isa_func_abi_result_store_record_t* record =
        (const iree_vm_isa_func_abi_result_store_record_t*)record_data;
    call->function_results.overflow[record->slot_u16] =
        functions[record->src_f8];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_abi_result_store_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONSTANT_ZERO, constant_zero) {
    do {
      const iree_vm_isa_constant_zero_record_t* record =
          (const iree_vm_isa_constant_zero_record_t*)record_data;
      values[record->dst_v8] = 0;
      record_data += sizeof(*record);
    } while (IREE_VM_BYTECODE_DISPATCH_IS_SAME(CONSTANT_ZERO));
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONSTANT_S16, constant_s16) {
    do {
      const iree_vm_isa_constant_s16_record_t* record =
          (const iree_vm_isa_constant_s16_record_t*)record_data;
      values[record->dst_v8] = (uint64_t)(int64_t)record->immediate_i16;
      record_data += sizeof(*record);
    } while (IREE_VM_BYTECODE_DISPATCH_IS_SAME(CONSTANT_S16));
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONSTANT_I32, constant_i32) {
    const iree_vm_isa_constant_i32_record_t* record =
        (const iree_vm_isa_constant_i32_record_t*)record_data;
    values[record->dst_v8] = record->bits_u32le;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_constant_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONSTANT_I64, constant_i64) {
    const iree_vm_isa_constant_i64_record_t* record =
        (const iree_vm_isa_constant_i64_record_t*)record_data;
    values[record->dst_v8] = (uint64_t)record->bits_low_u32le |
                             ((uint64_t)record->bits_high_u32le << 32);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_constant_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONSTANT_POOL_LOAD_I32,
                                 constant_pool_load_i32) {
    const iree_vm_isa_constant_pool_load_i32_record_t* record =
        (const iree_vm_isa_constant_pool_load_i32_record_t*)record_data;
    values[record->dst_v8] = (uint32_t)constant_cells[record->pool_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_constant_pool_load_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONSTANT_POOL_LOAD_I64,
                                 constant_pool_load_i64) {
    const iree_vm_isa_constant_pool_load_i64_record_t* record =
        (const iree_vm_isa_constant_pool_load_i64_record_t*)record_data;
    values[record->dst_v8] = constant_cells[record->pool_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_constant_pool_load_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(VALUE_COPY, value_copy) {
    do {
      const iree_vm_isa_value_copy_record_t* record =
          (const iree_vm_isa_value_copy_record_t*)record_data;
      values[record->dst_v8] = values[record->src_v8];
      record_data += sizeof(*record);
    } while (IREE_VM_BYTECODE_DISPATCH_IS_SAME(VALUE_COPY));
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(VALUE_SELECT, value_select) {
    do {
      const iree_vm_isa_value_select_record_t* record =
          (const iree_vm_isa_value_select_record_t*)record_data;
      const uint64_t condition = values[record->condition_v8];
      const uint64_t selected_value =
          condition ? values[record->true_v8] : values[record->false_v8];
      values[record->dst_v8] = selected_value;
      record_data += sizeof(*record);
    } while (IREE_VM_BYTECODE_DISPATCH_IS_SAME(VALUE_SELECT));
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_VALUE_IMMUTABLE_LOAD,
                                 global_value_immutable_load) {
    const iree_vm_isa_global_value_immutable_load_record_t* record =
        (const iree_vm_isa_global_value_immutable_load_record_t*)record_data;
    if (process_state->construction_state ==
            IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN &&
        !iree_vm_bytecode_bit_test(global_value_set_bits, record->global_u16)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "immutable value global is unset during construction");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    values[record->dst_v8] = global_values[record->global_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_value_immutable_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_VALUE_IMMUTABLE_STORE,
                                 global_value_immutable_store) {
    const iree_vm_isa_global_value_immutable_store_record_t* record =
        (const iree_vm_isa_global_value_immutable_store_record_t*)record_data;
    if (process_state->construction_state !=
            IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN ||
        iree_vm_bytecode_bit_test(global_value_set_bits, record->global_u16)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "immutable value global cannot be initialized");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    global_values[record->global_u16] = values[record->src_v8];
    iree_vm_bytecode_bit_set(global_value_set_bits, record->global_u16);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_value_immutable_store_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_VALUE_MUTABLE_LOAD,
                                 global_value_mutable_load) {
    const iree_vm_isa_global_value_mutable_load_record_t* record =
        (const iree_vm_isa_global_value_mutable_load_record_t*)record_data;
    values[record->dst_v8] = global_values[record->global_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_value_mutable_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_VALUE_MUTABLE_STORE,
                                 global_value_mutable_store) {
    const iree_vm_isa_global_value_mutable_store_record_t* record =
        (const iree_vm_isa_global_value_mutable_store_record_t*)record_data;
    global_values[record->global_u16] = values[record->src_v8];
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_value_mutable_store_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_REF_IMMUTABLE_LOAD_BORROW,
                                 global_ref_immutable_load_borrow) {
    const iree_vm_isa_global_ref_immutable_load_borrow_record_t* record =
        (const iree_vm_isa_global_ref_immutable_load_borrow_record_t*)
            record_data;
    if (process_state->construction_state ==
            IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN &&
        !iree_vm_bytecode_bit_test(global_ref_set_bits, record->global_u16)) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "immutable ref global is unset during construction");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const iree_vm_ref_t source = global_refs[record->global_u16];
    const iree_vm_ref_t borrowed =
        source.object ? iree_vm_ref_from_ptr_borrowed(source.object,
                                                      iree_vm_ref_type(source))
                      : iree_vm_ref_null();
    iree_vm_bytecode_ref_replace(&refs[record->dst_r8], borrowed);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_ref_immutable_load_borrow_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_REF_IMMUTABLE_STORE_MOVE,
                                 global_ref_immutable_store_move) {
    const iree_vm_isa_global_ref_immutable_store_move_record_t* record =
        (const iree_vm_isa_global_ref_immutable_store_move_record_t*)
            record_data;
    if (process_state->construction_state !=
            IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN ||
        iree_vm_bytecode_bit_test(global_ref_set_bits, record->global_u16)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "immutable ref global cannot be initialized");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor =
        &module->layout.globals.refs[record->global_u16];
    const iree_vm_ref_type_t expected_type =
        module->resolved_ref_types[descriptor->ref_type_ordinal_u16];
    if (!iree_vm_bytecode_ref_matches_global(refs[record->src_r8],
                                             expected_type, descriptor)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "immutable ref global type does not match");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const iree_vm_ref_t new_ref = iree_vm_ref_move(&refs[record->src_r8]);
    iree_vm_bytecode_ref_replace(&global_refs[record->global_u16], new_ref);
    iree_vm_bytecode_bit_set(global_ref_set_bits, record->global_u16);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_ref_immutable_store_move_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_REF_MUTABLE_LOAD_RETAIN,
                                 global_ref_mutable_load_retain) {
    const iree_vm_isa_global_ref_mutable_load_retain_record_t* record =
        (const iree_vm_isa_global_ref_mutable_load_retain_record_t*)record_data;
    const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor =
        &module->layout.globals.refs[record->global_u16];
    const iree_vm_ref_t source = global_refs[record->global_u16];
    if (!source.object &&
        !iree_any_bit_set(descriptor->flags_u16,
                          IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "nonnullable mutable ref global is temporarily null");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_bytecode_ref_retain(&refs[record->dst_r8], source);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_ref_mutable_load_retain_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_REF_MUTABLE_STORE_MOVE,
                                 global_ref_mutable_store_move) {
    const iree_vm_isa_global_ref_mutable_store_move_record_t* record =
        (const iree_vm_isa_global_ref_mutable_store_move_record_t*)record_data;
    const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor =
        &module->layout.globals.refs[record->global_u16];
    const iree_vm_ref_type_t expected_type =
        module->resolved_ref_types[descriptor->ref_type_ordinal_u16];
    if (!iree_vm_bytecode_ref_matches_global(refs[record->src_r8],
                                             expected_type, descriptor)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "mutable ref global type does not match");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const iree_vm_ref_t new_ref = iree_vm_ref_move(&refs[record->src_r8]);
    iree_vm_bytecode_ref_replace(&global_refs[record->global_u16], new_ref);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_ref_mutable_store_move_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_FUNC_IMMUTABLE_LOAD,
                                 global_func_immutable_load) {
    const iree_vm_isa_global_func_immutable_load_record_t* record =
        (const iree_vm_isa_global_func_immutable_load_record_t*)record_data;
    if (process_state->construction_state ==
            IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN &&
        !iree_vm_bytecode_bit_test(global_function_set_bits,
                                   record->global_u16)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "immutable function global is unset during construction");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    functions[record->dst_f8] = global_functions[record->global_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_func_immutable_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_FUNC_IMMUTABLE_STORE,
                                 global_func_immutable_store) {
    const iree_vm_isa_global_func_immutable_store_record_t* record =
        (const iree_vm_isa_global_func_immutable_store_record_t*)record_data;
    if (process_state->construction_state !=
            IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN ||
        iree_vm_bytecode_bit_test(global_function_set_bits,
                                  record->global_u16)) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "immutable function global cannot be initialized");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor =
        &module->layout.globals.functions[record->global_u16];
    if (!iree_vm_bytecode_function_matches_global(
            invocation->process->program, functions[record->src_f8],
            execution->linked_module, descriptor)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "immutable function global contract does not match");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    global_functions[record->global_u16] = functions[record->src_f8];
    iree_vm_bytecode_bit_set(global_function_set_bits, record->global_u16);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_func_immutable_store_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_FUNC_MUTABLE_LOAD,
                                 global_func_mutable_load) {
    const iree_vm_isa_global_func_mutable_load_record_t* record =
        (const iree_vm_isa_global_func_mutable_load_record_t*)record_data;
    const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor =
        &module->layout.globals.functions[record->global_u16];
    const iree_vm_function_ref_t source = global_functions[record->global_u16];
    if (process_state->construction_state ==
            IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN &&
        iree_vm_function_ref_is_null(source) &&
        !iree_any_bit_set(descriptor->flags_u16,
                          IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "nonnullable mutable function global is temporarily null");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    functions[record->dst_f8] = source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_func_mutable_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(GLOBAL_FUNC_MUTABLE_STORE,
                                 global_func_mutable_store) {
    const iree_vm_isa_global_func_mutable_store_record_t* record =
        (const iree_vm_isa_global_func_mutable_store_record_t*)record_data;
    const iree_vm_bytecode_v0_global_function_descriptor_row_t* descriptor =
        &module->layout.globals.functions[record->global_u16];
    if (!iree_vm_bytecode_function_matches_global(
            invocation->process->program, functions[record->src_f8],
            execution->linked_module, descriptor)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "mutable function global contract does not "
                                "match");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    global_functions[record->global_u16] = functions[record->src_f8];
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_global_func_mutable_store_record_t);
  }
#include "iree/vm/bytecode/interpreter_integer_dispatch.c.inc"
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_ADD_F32, float_add_f32) {
    const iree_vm_isa_float_add_f32_record_t* record =
        (const iree_vm_isa_float_add_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_add_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_add_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_ADD_F64, float_add_f64) {
    const iree_vm_isa_float_add_f64_record_t* record =
        (const iree_vm_isa_float_add_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_add_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_add_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_SUB_F32, float_sub_f32) {
    const iree_vm_isa_float_sub_f32_record_t* record =
        (const iree_vm_isa_float_sub_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_sub_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_sub_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_SUB_F64, float_sub_f64) {
    const iree_vm_isa_float_sub_f64_record_t* record =
        (const iree_vm_isa_float_sub_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_sub_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_sub_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MUL_F32, float_mul_f32) {
    const iree_vm_isa_float_mul_f32_record_t* record =
        (const iree_vm_isa_float_mul_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_mul_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_mul_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MUL_F64, float_mul_f64) {
    const iree_vm_isa_float_mul_f64_record_t* record =
        (const iree_vm_isa_float_mul_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_mul_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_mul_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_DIV_F32, float_div_f32) {
    const iree_vm_isa_float_div_f32_record_t* record =
        (const iree_vm_isa_float_div_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_div_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_div_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_DIV_F64, float_div_f64) {
    const iree_vm_isa_float_div_f64_record_t* record =
        (const iree_vm_isa_float_div_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_div_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_div_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_REM_F32, float_rem_f32) {
    const iree_vm_isa_float_rem_f32_record_t* record =
        (const iree_vm_isa_float_rem_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_rem_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_rem_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_REM_F64, float_rem_f64) {
    const iree_vm_isa_float_rem_f64_record_t* record =
        (const iree_vm_isa_float_rem_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_rem_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_rem_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_NEG_F32, float_neg_f32) {
    const iree_vm_isa_float_neg_f32_record_t* record =
        (const iree_vm_isa_float_neg_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_neg_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_neg_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_NEG_F64, float_neg_f64) {
    const iree_vm_isa_float_neg_f64_record_t* record =
        (const iree_vm_isa_float_neg_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_neg_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_neg_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_ABS_F32, float_abs_f32) {
    const iree_vm_isa_float_abs_f32_record_t* record =
        (const iree_vm_isa_float_abs_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_abs_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_abs_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_ABS_F64, float_abs_f64) {
    const iree_vm_isa_float_abs_f64_record_t* record =
        (const iree_vm_isa_float_abs_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_abs_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_abs_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MINMAX_F32, float_minmax_f32) {
    const iree_vm_isa_float_minmax_f32_record_t* record =
        (const iree_vm_isa_float_minmax_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_minmax_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_minmax_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MINMAX_F64, float_minmax_f64) {
    const iree_vm_isa_float_minmax_f64_record_t* record =
        (const iree_vm_isa_float_minmax_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_minmax_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_minmax_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_COMPARE_F32, float_compare_f32) {
    const iree_vm_isa_float_compare_f32_record_t* record =
        (const iree_vm_isa_float_compare_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_compare_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_compare_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_COMPARE_F64, float_compare_f64) {
    const iree_vm_isa_float_compare_f64_record_t* record =
        (const iree_vm_isa_float_compare_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_compare_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_compare_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_CLASSIFY_F32, float_classify_f32) {
    const iree_vm_isa_float_classify_f32_record_t* record =
        (const iree_vm_isa_float_classify_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_classify_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_classify_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_CLASSIFY_F64, float_classify_f64) {
    const iree_vm_isa_float_classify_f64_record_t* record =
        (const iree_vm_isa_float_classify_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_classify_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_classify_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_CLAMP_F32, float_clamp_f32) {
    const iree_vm_isa_float_clamp_f32_record_t* record =
        (const iree_vm_isa_float_clamp_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_clamp_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_clamp_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_CLAMP_F64, float_clamp_f64) {
    const iree_vm_isa_float_clamp_f64_record_t* record =
        (const iree_vm_isa_float_clamp_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_clamp_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_clamp_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_COPYSIGN_F32, float_copysign_f32) {
    const iree_vm_isa_float_copysign_f32_record_t* record =
        (const iree_vm_isa_float_copysign_f32_record_t*)record_data;
    iree_vm_bytecode_execute_float_copysign_f32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_copysign_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_COPYSIGN_F64, float_copysign_f64) {
    const iree_vm_isa_float_copysign_f64_record_t* record =
        (const iree_vm_isa_float_copysign_f64_record_t*)record_data;
    iree_vm_bytecode_execute_float_copysign_f64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_copysign_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MATH_UNARY_F32, float_math_unary_f32) {
    const iree_vm_isa_float_math_unary_f32_record_t* record =
        (const iree_vm_isa_float_math_unary_f32_record_t*)record_data;
    values[record->dst_v8] = iree_vm_bytecode_float_math_unary_f32(
        record->selector_u8, (uint32_t)values[record->src_v8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_math_unary_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MATH_UNARY_F64, float_math_unary_f64) {
    const iree_vm_isa_float_math_unary_f64_record_t* record =
        (const iree_vm_isa_float_math_unary_f64_record_t*)record_data;
    values[record->dst_v8] = iree_vm_bytecode_float_math_unary_f64(
        record->selector_u8, values[record->src_v8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_math_unary_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MATH_BINARY_F32, float_math_binary_f32) {
    const iree_vm_isa_float_math_binary_f32_record_t* record =
        (const iree_vm_isa_float_math_binary_f32_record_t*)record_data;
    values[record->dst_v8] = iree_vm_bytecode_float_math_binary_f32(
        record->selector_u8, (uint32_t)values[record->lhs_v8],
        (uint32_t)values[record->rhs_v8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_math_binary_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MATH_BINARY_F64, float_math_binary_f64) {
    const iree_vm_isa_float_math_binary_f64_record_t* record =
        (const iree_vm_isa_float_math_binary_f64_record_t*)record_data;
    values[record->dst_v8] = iree_vm_bytecode_float_math_binary_f64(
        record->selector_u8, values[record->lhs_v8], values[record->rhs_v8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_math_binary_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MATH_TERNARY_F32,
                                 float_math_ternary_f32) {
    const iree_vm_isa_float_math_ternary_f32_record_t* record =
        (const iree_vm_isa_float_math_ternary_f32_record_t*)record_data;
    values[record->dst_v8] = iree_vm_bytecode_float_math_ternary_f32(
        record->selector_u8, (uint32_t)values[record->a_v8],
        (uint32_t)values[record->b_v8], (uint32_t)values[record->c_v8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_math_ternary_f32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FLOAT_MATH_TERNARY_F64,
                                 float_math_ternary_f64) {
    const iree_vm_isa_float_math_ternary_f64_record_t* record =
        (const iree_vm_isa_float_math_ternary_f64_record_t*)record_data;
    values[record->dst_v8] = iree_vm_bytecode_float_math_ternary_f64(
        record->selector_u8, values[record->a_v8], values[record->b_v8],
        values[record->c_v8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_float_math_ternary_f64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_NULL, func_null) {
    const iree_vm_isa_func_null_record_t* record =
        (const iree_vm_isa_func_null_record_t*)record_data;
    functions[record->dst_f8] = iree_vm_function_ref_null();
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_null_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_COMPARE_NULL, func_compare_null) {
    const iree_vm_isa_func_compare_null_record_t* record =
        (const iree_vm_isa_func_compare_null_record_t*)record_data;
    values[record->dst_v8] =
        iree_vm_function_ref_is_null(functions[record->src_f8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_compare_null_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_COPY, func_copy) {
    const iree_vm_isa_func_copy_record_t* record =
        (const iree_vm_isa_func_copy_record_t*)record_data;
    functions[record->dst_f8] = functions[record->src_f8];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_copy_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_ADDRESS, func_address) {
    const iree_vm_isa_func_address_record_t* record =
        (const iree_vm_isa_func_address_record_t*)record_data;
    const iree_vm_program_t* program = invocation->process->program;
    const iree_vm_linked_module_t* linked_module = execution->linked_module;
    if (record->target_kind_u8 == IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL) {
      uint32_t mapping =
          program->callable_mappings[linked_module->callable_base +
                                     record->callable_type_ordinal_u16];
      mapping &= ~IREE_VM_PROGRAM_CALLABLE_MAY_YIELD;
      if (iree_any_bit_set(
              module->layout.functions.rows[record->target_ordinal_u16]
                  .flags_u16,
              IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD)) {
        mapping |= IREE_VM_PROGRAM_CALLABLE_MAY_YIELD;
      }
      const uint16_t module_ordinal =
          (uint16_t)(linked_module - program->linked_modules);
      functions[record->dst_f8] = (iree_vm_function_ref_t){
          (uint64_t)(uintptr_t)program,
          iree_vm_program_pack_target_bits(module_ordinal,
                                           record->target_ordinal_u16, mapping),
      };
    } else {
      const uint64_t target_bits =
          linked_module->import_target_bits[record->target_ordinal_u16];
      functions[record->dst_f8] = target_bits
                                      ? (iree_vm_function_ref_t){
                                            (uint64_t)(uintptr_t)program,
                                            target_bits,
                                        }
                                      : iree_vm_function_ref_null();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_address_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_IMPORT_RESOLVED, func_import_resolved) {
    const iree_vm_isa_func_import_resolved_record_t* record =
        (const iree_vm_isa_func_import_resolved_record_t*)record_data;
    values[record->dst_v8] =
        execution->linked_module
            ->import_target_bits[record->import_ordinal_u16] != 0;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_import_resolved_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_STACK_LOAD, func_stack_load) {
    const iree_vm_isa_func_stack_load_record_t* record =
        (const iree_vm_isa_func_stack_load_record_t*)record_data;
    functions[record->dst_f8] = local_functions[record->local_ordinal_u16];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_stack_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(FUNC_STACK_STORE, func_stack_store) {
    const iree_vm_isa_func_stack_store_record_t* record =
        (const iree_vm_isa_func_stack_store_record_t*)record_data;
    local_functions[record->local_ordinal_u16] = functions[record->src_f8];
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_func_stack_store_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_NULL, ref_null) {
    const iree_vm_isa_ref_null_record_t* record =
        (const iree_vm_isa_ref_null_record_t*)record_data;
    iree_vm_ref_reset(&refs[record->dst_r8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_null_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_COMPARE_NULL, ref_compare_null) {
    const iree_vm_isa_ref_compare_null_record_t* record =
        (const iree_vm_isa_ref_compare_null_record_t*)record_data;
    values[record->dst_v8] = iree_vm_ref_is_null(refs[record->src_r8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_compare_null_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_COMPARE_EQ, ref_compare_eq) {
    const iree_vm_isa_ref_compare_eq_record_t* record =
        (const iree_vm_isa_ref_compare_eq_record_t*)record_data;
    const iree_vm_ref_t lhs = refs[record->lhs_r8];
    const iree_vm_ref_t rhs = refs[record->rhs_r8];
    values[record->dst_v8] =
        lhs.object == rhs.object &&
        (!lhs.object || iree_vm_ref_type(lhs) == iree_vm_ref_type(rhs));
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_compare_eq_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_RETAIN, ref_retain) {
    const iree_vm_isa_ref_retain_record_t* record =
        (const iree_vm_isa_ref_retain_record_t*)record_data;
    iree_vm_bytecode_ref_retain(&refs[record->dst_r8], refs[record->src_r8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_retain_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_MOVE, ref_move) {
    const iree_vm_isa_ref_move_record_t* record =
        (const iree_vm_isa_ref_move_record_t*)record_data;
    iree_vm_bytecode_ref_move(&refs[record->dst_r8], &refs[record->src_r8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_move_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_DISCARD, ref_discard) {
    const iree_vm_isa_ref_discard_record_t* record =
        (const iree_vm_isa_ref_discard_record_t*)record_data;
    iree_vm_ref_reset(&refs[record->src_r8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_discard_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_STACK_LOAD_RETAIN, ref_stack_load_retain) {
    const iree_vm_isa_ref_stack_load_retain_record_t* record =
        (const iree_vm_isa_ref_stack_load_retain_record_t*)record_data;
    iree_vm_bytecode_ref_retain(&refs[record->dst_r8],
                                local_refs[record->slot_u16]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_stack_load_retain_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_STACK_LOAD_MOVE, ref_stack_load_move) {
    const iree_vm_isa_ref_stack_load_move_record_t* record =
        (const iree_vm_isa_ref_stack_load_move_record_t*)record_data;
    iree_vm_bytecode_ref_move(&refs[record->dst_r8],
                              &local_refs[record->slot_u16]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_stack_load_move_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_STACK_STORE_RETAIN,
                                 ref_stack_store_retain) {
    const iree_vm_isa_ref_stack_store_retain_record_t* record =
        (const iree_vm_isa_ref_stack_store_retain_record_t*)record_data;
    iree_vm_bytecode_ref_retain(&local_refs[record->slot_u16],
                                refs[record->src_r8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_stack_store_retain_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_STACK_STORE_MOVE, ref_stack_store_move) {
    const iree_vm_isa_ref_stack_store_move_record_t* record =
        (const iree_vm_isa_ref_stack_store_move_record_t*)record_data;
    iree_vm_bytecode_ref_move(&local_refs[record->slot_u16],
                              &refs[record->src_r8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_stack_store_move_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(REF_STACK_DISCARD, ref_stack_discard) {
    const iree_vm_isa_ref_stack_discard_record_t* record =
        (const iree_vm_isa_ref_stack_discard_record_t*)record_data;
    iree_vm_ref_reset(&local_refs[record->slot_u16]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_ref_stack_discard_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_RODATA_LOAD, buffer_rodata_load) {
    const iree_vm_isa_buffer_rodata_load_record_t* record =
        (const iree_vm_isa_buffer_rodata_load_record_t*)record_data;
    iree_vm_bytecode_ref_replace(
        &refs[record->dst_r8],
        iree_vm_ref_from_ptr_borrowed(&module->rodata_roots[record->rodata_u16],
                                      module->buffer_type));
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_rodata_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_ALLOCATE, buffer_allocate) {
    const iree_vm_isa_buffer_allocate_record_t* record =
        (const iree_vm_isa_buffer_allocate_record_t*)record_data;
    const uint64_t length = values[record->length_v8];
    if (!iree_vm_bytecode_buffer_allocation_is_representable(length)) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "vm.buffer allocation is not host-representable");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_create(
        (iree_host_size_t)length,
        /*minimum_alignment=*/0, invocation->process->host_allocator, &buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_bytecode_ref_replace(
        &refs[record->dst_r8],
        (iree_vm_ref_t){buffer, (uintptr_t)module->buffer_type});
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_allocate_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_LENGTH, buffer_length) {
    const iree_vm_isa_buffer_length_record_t* record =
        (const iree_vm_isa_buffer_length_record_t*)record_data;
    const iree_vm_ref_t buffer_ref = refs[record->buffer_r8];
    if (buffer_ref.object) {
      iree_vm_buffer_t* buffer = NULL;
      status = iree_vm_bytecode_buffer_check_deref(
          buffer_ref, module->buffer_type, &buffer);
      if (!iree_status_is_ok(status)) {
        IREE_VM_BYTECODE_DISPATCH_TERMINATE();
      }
      values[record->dst_v8] = buffer->length;
    } else {
      values[record->dst_v8] = 0;
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_length_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_SUBSPAN, buffer_subspan) {
    const iree_vm_isa_buffer_subspan_record_t* record =
        (const iree_vm_isa_buffer_subspan_record_t*)record_data;
    iree_vm_buffer_t* source = NULL;
    status = iree_vm_bytecode_buffer_check_deref(refs[record->buffer_r8],
                                                 module->buffer_type, &source);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const uint64_t offset = values[record->offset_v8];
    const uint64_t length = values[record->length_v8];
    const iree_vm_buffer_access_flags_t access =
        iree_vm_buffer_effective_access(source);
    if (access == IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "vm.buffer source is closed");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    if (offset > source->length || length > source->length - offset) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "vm.buffer subspan range is out of bounds");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_buffer_t* result = NULL;
    status = iree_vm_buffer_subspan(
        source, (iree_host_size_t)offset, (iree_host_size_t)length, access,
        invocation->process->host_allocator, &result);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_bytecode_ref_replace(
        &refs[record->dst_r8],
        (iree_vm_ref_t){result, (uintptr_t)module->buffer_type});
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_subspan_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_LOAD, buffer_load) {
    const iree_vm_isa_buffer_load_record_t* record =
        (const iree_vm_isa_buffer_load_record_t*)record_data;
    const uint64_t base = values[record->base_v8];
    const uint64_t index = values[record->index_v8];
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(refs[record->buffer_r8],
                                                 module->buffer_type, &buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const uint8_t access_length =
        (uint8_t)(1u << ((record->format_u8 >> 2) + (record->format_u8 & 3u)));
    iree_byte_span_t source = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_lanes(
        buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, base, index, record->scale_u8,
        access_length, &source);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_bytecode_stack_load_lanes(record->format_u8, source.data,
                                      values + record->dst_v8);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_STORE, buffer_store) {
    const iree_vm_isa_buffer_store_record_t* record =
        (const iree_vm_isa_buffer_store_record_t*)record_data;
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(refs[record->buffer_r8],
                                                 module->buffer_type, &buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const uint8_t access_length =
        (uint8_t)(1u << ((record->format_u8 >> 2) + (record->format_u8 & 3u)));
    iree_byte_span_t target = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_lanes(
        buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, values[record->base_v8],
        values[record->index_v8], record->scale_u8, access_length, &target);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_bytecode_stack_store_lanes(record->format_u8,
                                       values + record->src_v8, target.data);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_store_record_t);
  }
  // Host atomics operate at system scope. Verified thread through device
  // scopes are conservatively strengthened and need no execution-time branch.
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_ATOMIC_REDUCE, buffer_atomic_reduce) {
    const iree_vm_isa_buffer_atomic_reduce_record_t* record =
        (const iree_vm_isa_buffer_atomic_reduce_record_t*)record_data;
    const uint64_t offset = values[record->offset_v8];
    const uint64_t operand_bits = values[record->operand_v8];
    const iree_vm_isa_buffer_atomic_kind_t kind = record->selector0_u8 & 0x0Fu;
    const iree_vm_isa_buffer_atomic_carrier_t carrier =
        record->selector0_u8 >> 7;
    const iree_vm_isa_buffer_atomic_ordering_t ordering =
        record->selector1_u8 & 0x07u;
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(refs[record->buffer_r8],
                                                 module->buffer_type, &buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    uint8_t* address = NULL;
    status = iree_vm_bytecode_buffer_map_atomic(
        buffer, offset, (iree_host_size_t)4u << carrier, &address);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    (void)iree_vm_bytecode_atomic_apply(address, operand_bits, kind, carrier,
                                        ordering);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_atomic_reduce_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_ATOMIC_RMW, buffer_atomic_rmw) {
    const iree_vm_isa_buffer_atomic_rmw_record_t* record =
        (const iree_vm_isa_buffer_atomic_rmw_record_t*)record_data;
    const uint64_t offset = values[record->offset_v8];
    const uint64_t operand_bits = values[record->operand_v8];
    const iree_vm_isa_buffer_atomic_kind_t kind = record->selector0_u8 & 0x0Fu;
    const iree_vm_isa_buffer_atomic_carrier_t carrier =
        record->selector0_u8 >> 7;
    const iree_vm_isa_buffer_atomic_ordering_t ordering =
        record->selector1_u8 & 0x07u;
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(refs[record->buffer_r8],
                                                 module->buffer_type, &buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    uint8_t* address = NULL;
    status = iree_vm_bytecode_buffer_map_atomic(
        buffer, offset, (iree_host_size_t)4u << carrier, &address);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    values[record->old_v8] = iree_vm_bytecode_atomic_apply(
        address, operand_bits, kind, carrier, ordering);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_atomic_rmw_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_ATOMIC_CMPXCHG, buffer_atomic_cmpxchg) {
    const iree_vm_isa_buffer_atomic_cmpxchg_record_t* record =
        (const iree_vm_isa_buffer_atomic_cmpxchg_record_t*)record_data;
    const uint64_t offset = values[record->offset_v8];
    const uint64_t expected_bits = values[record->expected_v8];
    const uint64_t replacement_bits = values[record->replacement_v8];
    const iree_vm_isa_buffer_atomic_carrier_t carrier =
        record->selector0_u8 >> 7;
    const iree_vm_isa_buffer_atomic_ordering_t success_ordering =
        record->selector0_u8 & 0x07u;
    const iree_vm_isa_buffer_atomic_ordering_t failure_ordering =
        (record->selector0_u8 >> 3) & 0x07u;
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(refs[record->buffer_r8],
                                                 module->buffer_type, &buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    uint8_t* address = NULL;
    status = iree_vm_bytecode_buffer_map_atomic(
        buffer, offset, (iree_host_size_t)4u << carrier, &address);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    values[record->old_v8] = iree_vm_bytecode_atomic_compare_exchange(
        address, expected_bits, replacement_bits, carrier, success_ordering,
        failure_ordering);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_atomic_cmpxchg_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_FILL, buffer_fill) {
    const iree_vm_isa_buffer_fill_record_t* record =
        (const iree_vm_isa_buffer_fill_record_t*)record_data;
    const uint64_t offset = values[record->offset_v8];
    const uint64_t length = values[record->length_v8];
    const uint64_t pattern = values[record->pattern_v8];
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(refs[record->buffer_r8],
                                                 module->buffer_type, &buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t target = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(
        buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, offset, length, &target);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    if (length != 0) {
      iree_vm_bytecode_fill_pattern(target.data, target.data_length, pattern,
                                    record->pattern_width_u8);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_fill_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_COPY, buffer_copy) {
    const iree_vm_isa_buffer_copy_record_t* record =
        (const iree_vm_isa_buffer_copy_record_t*)record_data;
    const uint64_t target_offset = values[record->target_offset_v8];
    const uint64_t source_offset = values[record->source_offset_v8];
    const uint64_t length = values[record->length_v8];
    iree_vm_buffer_t* target_buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(
        refs[record->target_r8], module->buffer_type, &target_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_buffer_t* source_buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(
        refs[record->source_r8], module->buffer_type, &source_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t target = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(target_buffer,
                                               IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
                                               target_offset, length, &target);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t source = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(source_buffer,
                                               IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                               source_offset, length, &source);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    if (length != 0) {
      memmove(target.data, source.data, target.data_length);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_copy_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_COMPARE, buffer_compare) {
    const iree_vm_isa_buffer_compare_record_t* record =
        (const iree_vm_isa_buffer_compare_record_t*)record_data;
    const uint64_t lhs_offset = values[record->lhs_offset_v8];
    const uint64_t rhs_offset = values[record->rhs_offset_v8];
    const uint64_t length = values[record->length_v8];
    iree_vm_buffer_t* lhs_buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(
        refs[record->lhs_r8], module->buffer_type, &lhs_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_buffer_t* rhs_buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(
        refs[record->rhs_r8], module->buffer_type, &rhs_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t lhs = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(
        lhs_buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, lhs_offset, length, &lhs);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t rhs = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(
        rhs_buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, rhs_offset, length, &rhs);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    const int ordering =
        length == 0 ? 0 : memcmp(lhs.data, rhs.data, lhs.data_length);
    values[record->dst_v8] = ordering < 0   ? UINT32_MAX
                             : ordering > 0 ? UINT32_C(1)
                                            : UINT32_C(0);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_compare_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_COPY_RODATA, buffer_copy_rodata) {
    const iree_vm_isa_buffer_copy_rodata_record_t* record =
        (const iree_vm_isa_buffer_copy_rodata_record_t*)record_data;
    const uint64_t target_offset = values[record->target_offset_v8];
    const uint64_t length = values[record->length_v8];
    iree_vm_buffer_t* target_buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(
        refs[record->target_r8], module->buffer_type, &target_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t target = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(target_buffer,
                                               IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
                                               target_offset, length, &target);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t source = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(
        &module->rodata_roots[record->rodata_u16],
        IREE_VM_BUFFER_ACCESS_FLAG_READ, record->source_offset_u32, length,
        &source);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    if (length != 0) {
      memcpy(target.data, source.data, target.data_length);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_copy_rodata_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_INTEGER, conversion_integer) {
    do {
      const iree_vm_isa_conversion_integer_record_t* record =
          (const iree_vm_isa_conversion_integer_record_t*)record_data;
      iree_vm_bytecode_execute_conversion_integer(record, values);
      record_data += sizeof(*record);
    } while (IREE_VM_BYTECODE_DISPATCH_IS_SAME(CONVERSION_INTEGER));
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_FLOAT_EXTEND,
                                 conversion_float_extend) {
    const iree_vm_isa_conversion_float_extend_record_t* record =
        (const iree_vm_isa_conversion_float_extend_record_t*)record_data;
    iree_vm_bytecode_execute_conversion_float_extend(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_conversion_float_extend_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_FLOAT_TRUNCATE,
                                 conversion_float_truncate) {
    const iree_vm_isa_conversion_float_truncate_record_t* record =
        (const iree_vm_isa_conversion_float_truncate_record_t*)record_data;
    iree_vm_bytecode_execute_conversion_float_truncate(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_conversion_float_truncate_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_FLOAT_WIDTH,
                                 conversion_float_width) {
    const iree_vm_isa_conversion_float_width_record_t* record =
        (const iree_vm_isa_conversion_float_width_record_t*)record_data;
    iree_vm_bytecode_execute_conversion_float_width(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_conversion_float_width_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_INTEGER_TO_FLOAT,
                                 conversion_integer_to_float) {
    const iree_vm_isa_conversion_integer_to_float_record_t* record =
        (const iree_vm_isa_conversion_integer_to_float_record_t*)record_data;
    iree_vm_bytecode_execute_conversion_integer_to_float(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_conversion_integer_to_float_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_FLOAT_TO_INTEGER,
                                 conversion_float_to_integer) {
    const iree_vm_isa_conversion_float_to_integer_record_t* record =
        (const iree_vm_isa_conversion_float_to_integer_record_t*)record_data;
    const iree_vm_bytecode_conversion_failure_t failure =
        iree_vm_bytecode_execute_conversion_float_to_integer(record, values);
    if (failure != IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE) {
      status = failure == IREE_VM_BYTECODE_CONVERSION_FAILURE_NAN
                   ? iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "cannot convert a floating NaN to an "
                                      "integer")
                   : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                      "floating value is outside the selected "
                                      "integer interval");
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_conversion_float_to_integer_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_LOAD, stack_load) {
    const iree_vm_isa_stack_load_record_t* record =
        (const iree_vm_isa_stack_load_record_t*)record_data;
    iree_vm_bytecode_stack_load_lanes(record->format_u8,
                                      local_bytes + record->base_u16,
                                      values + record->dst_v8);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_STORE, stack_store) {
    const iree_vm_isa_stack_store_record_t* record =
        (const iree_vm_isa_stack_store_record_t*)record_data;
    iree_vm_bytecode_stack_store_lanes(record->format_u8,
                                       values + record->src_v8,
                                       local_bytes + record->base_u16);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_store_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_LOAD_INDEXED, stack_load_indexed) {
    const iree_vm_isa_stack_load_indexed_record_t* record =
        (const iree_vm_isa_stack_load_indexed_record_t*)record_data;
    const uint8_t access_length =
        (uint8_t)(1u << ((record->format_u8 >> 2) + (record->format_u8 & 3u)));
    uint16_t effective_base = 0;
    status = iree_vm_bytecode_stack_resolve_index(
        function->local_byte_length_u16, record->base_u16, access_length,
        values[record->index_v8], record->scale_u8, &effective_base);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_bytecode_stack_load_lanes(record->format_u8,
                                      local_bytes + effective_base,
                                      values + record->dst_v8);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_load_indexed_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_STORE_INDEXED, stack_store_indexed) {
    const iree_vm_isa_stack_store_indexed_record_t* record =
        (const iree_vm_isa_stack_store_indexed_record_t*)record_data;
    const uint8_t access_length =
        (uint8_t)(1u << ((record->format_u8 >> 2) + (record->format_u8 & 3u)));
    uint16_t effective_base = 0;
    status = iree_vm_bytecode_stack_resolve_index(
        function->local_byte_length_u16, record->base_u16, access_length,
        values[record->index_v8], record->scale_u8, &effective_base);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_vm_bytecode_stack_store_lanes(record->format_u8,
                                       values + record->src_v8,
                                       local_bytes + effective_base);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_store_indexed_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_FILL, stack_fill) {
    const iree_vm_isa_stack_fill_record_t* record =
        (const iree_vm_isa_stack_fill_record_t*)record_data;
    if (record->length_u16 != 0) {
      iree_vm_bytecode_fill_pattern(
          local_bytes + record->target_base_u16, record->length_u16,
          values[record->pattern_v8], record->pattern_width_u8);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_fill_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_COPY, stack_copy) {
    const iree_vm_isa_stack_copy_record_t* record =
        (const iree_vm_isa_stack_copy_record_t*)record_data;
    if (record->length_u16 != 0) {
      memmove(local_bytes + record->target_u16,
              local_bytes + record->source_u16, record->length_u16);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_copy_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_COMPARE, stack_compare) {
    const iree_vm_isa_stack_compare_record_t* record =
        (const iree_vm_isa_stack_compare_record_t*)record_data;
    int ordering = 0;
    if (record->length_u16 != 0) {
      ordering = memcmp(local_bytes + record->lhs_u16,
                        local_bytes + record->rhs_u16, record->length_u16);
    }
    values[record->dst_v8] = ordering < 0   ? UINT32_MAX
                             : ordering > 0 ? UINT32_C(1)
                                            : UINT32_C(0);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_compare_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_COPY_RODATA, stack_copy_rodata) {
    const iree_vm_isa_stack_copy_rodata_record_t* record =
        (const iree_vm_isa_stack_copy_rodata_record_t*)record_data;
    if (record->length_u16 != 0) {
      const uint8_t* source = (const uint8_t*)iree_vm_buffer_const_data(
          &module->rodata_roots[record->rodata_u16]);
      memcpy(local_bytes + record->target_u16,
             source + record->source_offset_u32, record->length_u16);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_copy_rodata_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_COPY_FROM_BUFFER,
                                 stack_copy_from_buffer) {
    const iree_vm_isa_stack_copy_from_buffer_record_t* record =
        (const iree_vm_isa_stack_copy_from_buffer_record_t*)record_data;
    const uint64_t source_offset = values[record->source_offset_v8];
    iree_vm_buffer_t* source_buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(
        refs[record->buffer_r8], module->buffer_type, &source_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t source = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(
        source_buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, source_offset,
        record->length_u16, &source);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    if (record->length_u16 != 0) {
      memcpy(local_bytes + record->target_u16, source.data, record->length_u16);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_copy_from_buffer_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_COPY_TO_BUFFER, stack_copy_to_buffer) {
    const iree_vm_isa_stack_copy_to_buffer_record_t* record =
        (const iree_vm_isa_stack_copy_to_buffer_record_t*)record_data;
    const uint64_t target_offset = values[record->target_offset_v8];
    iree_vm_buffer_t* target_buffer = NULL;
    status = iree_vm_bytecode_buffer_check_deref(
        refs[record->buffer_r8], module->buffer_type, &target_buffer);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    iree_byte_span_t target = iree_byte_span_empty();
    status = iree_vm_bytecode_buffer_map_range(
        target_buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, target_offset,
        record->length_u16, &target);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    if (record->length_u16 != 0) {
      memcpy(target.data, local_bytes + record->source_u16, record->length_u16);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_copy_to_buffer_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_CONST_S16_I32, stack_const_s16_i32) {
    const iree_vm_isa_stack_const_s16_i32_record_t* record =
        (const iree_vm_isa_stack_const_s16_i32_record_t*)record_data;
    if (record->count_u16 != 0) {
      iree_vm_bytecode_stack_const_s16_i32(local_bytes + record->target_u16,
                                           record->count_u16,
                                           record->immediate_i16);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_const_s16_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_CONST_S16_I64, stack_const_s16_i64) {
    const iree_vm_isa_stack_const_s16_i64_record_t* record =
        (const iree_vm_isa_stack_const_s16_i64_record_t*)record_data;
    if (record->count_u16 != 0) {
      iree_vm_bytecode_stack_const_s16_i64(local_bytes + record->target_u16,
                                           record->count_u16,
                                           record->immediate_i16);
    }
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_const_s16_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_PACK_I32_U16_X2, stack_pack_i32_u16_x2) {
    const iree_vm_isa_stack_pack_i32_u16_x2_record_t* record =
        (const iree_vm_isa_stack_pack_i32_u16_x2_record_t*)record_data;
    iree_vm_bytecode_stack_pack_i32(local_bytes + record->target_u16,
                                    record->immediates_le, 2);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_pack_i32_u16_x2_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_PACK_I32_U16_X4, stack_pack_i32_u16_x4) {
    const iree_vm_isa_stack_pack_i32_u16_x4_record_t* record =
        (const iree_vm_isa_stack_pack_i32_u16_x4_record_t*)record_data;
    iree_vm_bytecode_stack_pack_i32(local_bytes + record->target_u16,
                                    record->immediates_le, 4);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_pack_i32_u16_x4_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_PACK_I32_U16_X8, stack_pack_i32_u16_x8) {
    const iree_vm_isa_stack_pack_i32_u16_x8_record_t* record =
        (const iree_vm_isa_stack_pack_i32_u16_x8_record_t*)record_data;
    iree_vm_bytecode_stack_pack_i32(local_bytes + record->target_u16,
                                    record->immediates_le, 8);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_pack_i32_u16_x8_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_PACK_I64_U32_X2, stack_pack_i64_u32_x2) {
    const iree_vm_isa_stack_pack_i64_u32_x2_record_t* record =
        (const iree_vm_isa_stack_pack_i64_u32_x2_record_t*)record_data;
    iree_vm_bytecode_stack_pack_i64(local_bytes + record->target_u16,
                                    record->immediates_le, 2);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_pack_i64_u32_x2_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_PACK_I64_U32_X4, stack_pack_i64_u32_x4) {
    const iree_vm_isa_stack_pack_i64_u32_x4_record_t* record =
        (const iree_vm_isa_stack_pack_i64_u32_x4_record_t*)record_data;
    iree_vm_bytecode_stack_pack_i64(local_bytes + record->target_u16,
                                    record->immediates_le, 4);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_pack_i64_u32_x4_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(STACK_PACK_I64_U32_X8, stack_pack_i64_u32_x8) {
    const iree_vm_isa_stack_pack_i64_u32_x8_record_t* record =
        (const iree_vm_isa_stack_pack_i64_u32_x8_record_t*)record_data;
    iree_vm_bytecode_stack_pack_i64(local_bytes + record->target_u16,
                                    record->immediates_le, 8);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_stack_pack_i64_u32_x8_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_END();

  return status;
}

static bool iree_vm_bytecode_function_is_scalar_leaf(
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature) {
  // Verification requires the register bank to cover every result, so the
  // final comparison is equality for every published image.
  return function->local_byte_length_u16 == 0 &&
         function->ref_register_count_u16 == 0 &&
         function->local_ref_count_u32 == 0 &&
         function->function_register_count_u16 == 0 &&
         function->local_function_count_u32 == 0 &&
         function->value_register_count_u16 <=
             IREE_VM_CALL_DIRECT_REGISTER_COUNT &&
         function->value_register_count_u16 <=
             signature->result_value_count_u16;
}

static iree_status_t iree_vm_bytecode_function_start_transient(
    iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_t* invocation = params->execution.invocation;
  uint8_t* frame_checkpoint = invocation->stack_cursor;
  iree_vm_bytecode_execution_state_t state = {
      .call = &params->call,
      .program_counter = module->layout.functions.bytecode_data +
                         function->bytecode_offset_u32 +
                         sizeof(iree_vm_isa_control_block_record_t),
      .frame = NULL,
  };
  if (iree_vm_bytecode_function_is_scalar_leaf(function, signature)) {
    // Result banks are invocation-owned staging until the root succeeds. A
    // scalar-only leaf whose complete register bank fits the direct result
    // prefix can execute in place without reserving or copying a frame.
    state.values = params->call.value_results.direct;
    const uint16_t direct_value_count =
        iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                 signature->argument_value_count_u16);
    iree_vm_bytecode_copy_direct_values(
        state.values, params->call.value_arguments.direct, direct_value_count);
  } else {
    const uint32_t ref_count =
        function->ref_register_count_u16 + function->local_ref_count_u32;
    const uint32_t function_count = function->function_register_count_u16 +
                                    function->local_function_count_u32;
    const iree_host_size_t frame_storage_size =
        function->value_register_count_u16 * sizeof(uint64_t) +
        (iree_host_size_t)ref_count * sizeof(iree_vm_ref_t) +
        (iree_host_size_t)function_count * sizeof(iree_vm_function_ref_t) +
        function->local_byte_length_u16;
    uint8_t* frame_storage = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_invocation_stack_reserve(
        invocation, frame_storage_size,
        iree_max(iree_max(iree_alignof(uint64_t), iree_alignof(iree_vm_ref_t)),
                 iree_alignof(iree_vm_function_ref_t)),
        &frame_checkpoint, &frame_storage));
    state.values = (uint64_t*)frame_storage;
    state.refs =
        (iree_vm_ref_t*)(frame_storage +
                         function->value_register_count_u16 * sizeof(uint64_t));
    state.local_refs = state.refs + function->ref_register_count_u16;
    state.functions = (iree_vm_function_ref_t*)(state.local_refs +
                                                function->local_ref_count_u32);
    state.local_functions =
        state.functions + function->function_register_count_u16;
    state.local_bytes =
        (uint8_t*)(state.local_functions + function->local_function_count_u32);
    iree_vm_bytecode_initialize_state(function, signature, &state);
  }
  iree_status_t status = iree_vm_bytecode_execute(
      module, function, &params->execution, &state, out_outcome);
  const uint32_t ref_count =
      function->ref_register_count_u16 + function->local_ref_count_u32;
  iree_vm_bytecode_frame_reset(state.refs, ref_count);
  iree_vm_invocation_stack_rewind(invocation, frame_checkpoint);
  return status;
}

static iree_status_t iree_vm_bytecode_function_start_durable(
    iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_frame_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_query_frame_layout(function, &layout));
  iree_vm_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_invocation_push_frame(
      params, layout.frame, iree_vm_bytecode_frame_cleanup, &frame));

  uint8_t* frame_storage = (uint8_t*)iree_vm_frame_storage(frame);
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)frame_storage;
  iree_vm_call_packet_t* call =
      (iree_vm_call_packet_t*)(frame_storage + layout.call_offset);
  *call = params->call;
  *state = (iree_vm_bytecode_execution_state_t){
      .call = call,
      .program_counter = module->layout.functions.bytecode_data +
                         function->bytecode_offset_u32 +
                         sizeof(iree_vm_isa_control_block_record_t),
      .frame = frame,
      .values = (uint64_t*)(frame_storage + layout.values_offset),
      .refs = (iree_vm_ref_t*)(frame_storage + layout.refs_offset),
      .local_refs = (iree_vm_ref_t*)(frame_storage + layout.local_refs_offset),
      .functions =
          (iree_vm_function_ref_t*)(frame_storage + layout.functions_offset),
      .local_functions =
          (iree_vm_function_ref_t*)(frame_storage +
                                    layout.local_functions_offset),
      .local_bytes = frame_storage + layout.local_bytes_offset,
  };
  iree_vm_bytecode_initialize_state(function, signature, state);

  iree_status_t status = iree_vm_bytecode_execute(
      module, function, &params->execution, state, out_outcome);
  if (iree_status_is_ok(status) &&
      *out_outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    iree_vm_invocation_pop_frame(params->execution.invocation, frame);
  }
  return status;
}

iree_status_t iree_vm_bytecode_function_start(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  const iree_vm_bytecode_v0_function_row_t* function =
      &module->layout.functions.rows[params->function_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      iree_vm_bytecode_function_signature(&module->layout, function);
  return iree_any_bit_set(function->flags_u16,
                          IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD |
                              IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)
             ? iree_vm_bytecode_function_start_durable(
                   module, function, signature, params, out_outcome)
             : iree_vm_bytecode_function_start_transient(
                   module, function, signature, params, out_outcome);
}

iree_status_t iree_vm_bytecode_function_resume(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_cancel_reason_t cancel_reason =
      iree_vm_invocation_cancel_reason(params->execution.invocation);
  if (cancel_reason != IREE_VM_CANCEL_REASON_NONE) {
    return iree_vm_invocation_cancel_status(cancel_reason);
  }

  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  const uint16_t function_ordinal =
      iree_vm_frame_function_ordinal(params->frame);
  const iree_vm_bytecode_v0_function_row_t* function =
      &module->layout.functions.rows[function_ordinal];
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)iree_vm_frame_storage(params->frame);
  iree_vm_bytecode_call_cleanup_completed(state);
  iree_status_t status = iree_vm_bytecode_execute(
      module, function, &params->execution, state, out_outcome);
  if (iree_status_is_ok(status) &&
      *out_outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    iree_vm_invocation_pop_frame(params->execution.invocation, params->frame);
  }
  return status;
}

#undef IREE_VM_BYTECODE_DISPATCH_TABLE_ENTRY
#undef IREE_VM_BYTECODE_DISPATCH_BEGIN
#undef IREE_VM_BYTECODE_DISPATCH_CASE
#undef IREE_VM_BYTECODE_DISPATCH_NEXT
#undef IREE_VM_BYTECODE_DISPATCH_IS_SAME
#undef IREE_VM_BYTECODE_DISPATCH_CONTINUE
#undef IREE_VM_BYTECODE_DISPATCH_TERMINATE
#undef IREE_VM_BYTECODE_DISPATCH_END
#undef IREE_VM_BYTECODE_EXECUTABLE_OPCODE_LIST
