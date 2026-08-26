// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter.h"

#include <string.h>

#include "iree/vm/bytecode/interpreter_float.h"
#include "iree/vm/bytecode/interpreter_integer.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/float.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/value.h"
#include "iree/vm/invocation_storage.h"

#define IREE_VM_BYTECODE_DEFINE_EXECUTABLE_OPCODE_LIST
#include "iree/vm/bytecode/execution_tables.inl"
#undef IREE_VM_BYTECODE_DEFINE_EXECUTABLE_OPCODE_LIST

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
  const uint8_t* record_data = bytecode;           \
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
#define IREE_VM_BYTECODE_DISPATCH_BEGIN()                                   \
  const uint8_t* record_data = bytecode;                                    \
  const uint8_t* bytecode_end = bytecode + function->bytecode_length_u32 -  \
                                sizeof(iree_vm_isa_control_block_record_t); \
  bool dispatch_terminated = false;                                         \
  while (!dispatch_terminated && record_data < bytecode_end) {              \
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

static void iree_vm_bytecode_frame_reset(iree_vm_ref_t* refs,
                                         uint16_t ref_register_count) {
  for (uint16_t i = 0; i < ref_register_count; ++i) {
    iree_vm_ref_reset(&refs[i]);
  }
}

static uint32_t iree_vm_bytecode_bf16_to_f32_bits(uint16_t source_bits) {
  uint32_t result_bits = (uint32_t)source_bits << 16;
  const bool is_nan =
      (source_bits & 0x7F80u) == 0x7F80u && (source_bits & 0x007Fu) != 0;
  if (is_nan) result_bits |= 0x00400000u;
  return result_bits;
}

static iree_status_t iree_vm_bytecode_f32_to_u32(uint32_t source_bits,
                                                 uint32_t* out_result) {
  const bool is_nan = (source_bits & 0x7F800000u) == 0x7F800000u &&
                      (source_bits & 0x007FFFFFu) != 0;
  if (is_nan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot convert an f32 NaN to u32");
  }
  float source = 0.0f;
  memcpy(&source, &source_bits, sizeof(source));
  if (!(source > -1.0f && source < 0x1p32f)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "f32 value is outside the u32 interval");
  }
  *out_result = (uint32_t)source;
  return iree_ok_status();
}

// Copies one direct physical value bank in fixed native-vector-sized chunks.
// Direct banks are capped at 16 cells, so this remains a bounded sequence with
// no variable-length memcpy call or per-cell bounds branch.
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

static iree_status_t iree_vm_bytecode_publish_results(
    const iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_call_packet_t* call, uint64_t* values, iree_vm_ref_t* refs) {
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &module->layout.signatures.rows[function->signature_ordinal_u16];
  if (signature->result_ref_count_u16) {
    const uint32_t argument_count =
        iree_vm_bytecode_signature_argument_count(signature);
    const uint32_t result_count =
        iree_vm_bytecode_signature_result_count(signature);
    const iree_vm_bytecode_v0_signature_descriptor_row_t* result_descriptors =
        iree_vm_bytecode_signature_descriptors(
            &module->layout.signatures, function->signature_ordinal_u16) +
        argument_count;
    uint16_t ref_ordinal = 0;
    for (uint32_t i = 0; i < result_count; ++i) {
      const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor =
          &result_descriptors[i];
      if (descriptor->kind_u16 != IREE_VM_BYTECODE_SIGNATURE_KIND_REF) continue;
      if (refs[ref_ordinal].object &&
          iree_vm_ref_type(refs[ref_ordinal]) !=
              module->resolved_ref_types[descriptor->type_ordinal_u16]) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "bytecode function returned the wrong ref type");
      }
      ++ref_ordinal;
    }
  }

  const uint16_t direct_value_count = signature->result_value_count_u16 < 16
                                          ? signature->result_value_count_u16
                                          : 16;
  if (call->value_results.direct != values) {
    iree_vm_bytecode_copy_direct_values(call->value_results.direct, values,
                                        direct_value_count);
    if (signature->result_value_count_u16 > 16) {
      memcpy(call->value_results.overflow, values + 16,
             (signature->result_value_count_u16 - 16) * sizeof(uint64_t));
    }
  }
  for (uint16_t i = 0; i < signature->result_ref_count_u16; ++i) {
    iree_vm_call_ref_result_store_move(call, i, &refs[i]);
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_function_start(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  const iree_vm_bytecode_v0_function_row_t* function =
      &module->layout.functions.rows[params->function_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &module->layout.signatures.rows[function->signature_ordinal_u16];
  iree_vm_invocation_t* invocation = params->execution.invocation;
  uint8_t* frame_checkpoint = invocation->stack_cursor;
  uint64_t* values = NULL;
  iree_vm_ref_t* refs = NULL;
  // Verification requires the register bank to cover every result, so the
  // final comparison is equality for every published image.
  if (function->ref_register_count_u16 == 0 &&
      function->value_register_count_u16 <= 16 &&
      function->value_register_count_u16 <= signature->result_value_count_u16) {
    // Result banks are invocation-owned staging until the root succeeds. A
    // scalar-only leaf whose complete register bank fits the direct result
    // prefix can execute in place without reserving or copying a frame.
    values = params->call.value_results.direct;
  } else {
    const iree_host_size_t frame_storage_size =
        function->value_register_count_u16 * sizeof(uint64_t) +
        function->ref_register_count_u16 * sizeof(iree_vm_ref_t);
    uint8_t* frame_storage = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_invocation_stack_reserve(
        invocation, frame_storage_size,
        iree_max(iree_alignof(uint64_t), iree_alignof(iree_vm_ref_t)),
        &frame_checkpoint, &frame_storage));
    values = (uint64_t*)frame_storage;
    if (function->ref_register_count_u16) {
      refs = (iree_vm_ref_t*)(values + function->value_register_count_u16);
    }
  }
  for (uint16_t i = 0; i < function->ref_register_count_u16; ++i) {
    refs[i] = iree_vm_ref_null();
  }
  const uint16_t direct_value_count = signature->argument_value_count_u16 < 16
                                          ? signature->argument_value_count_u16
                                          : 16;
  iree_vm_bytecode_copy_direct_values(
      values, params->call.value_arguments.direct, direct_value_count);
  if (signature->argument_value_count_u16 > 16) {
    memcpy(values + 16, params->call.value_arguments.overflow,
           (signature->argument_value_count_u16 - 16) * sizeof(uint64_t));
  }
  for (uint16_t i = 0; i < signature->argument_ref_count_u16; ++i) {
    iree_vm_call_ref_argument_load_move(&params->call, i, &refs[i]);
  }

  void* process_storage = params->execution.process_storage;
  uint64_t* global_values =
      module->layout.globals.header
          ? iree_vm_bytecode_process_values(module, process_storage)
          : NULL;
  uint64_t* global_value_set_bits =
      module->layout.globals.header
          ? iree_vm_bytecode_process_value_set_bits(module, process_storage)
          : NULL;
  iree_vm_bytecode_process_state_t* process_state =
      module->layout.globals.header
          ? iree_vm_bytecode_process_state(process_storage)
          : NULL;
  const iree_vm_bytecode_v0_constant_cell_t* constant_cells =
      module->layout.constants.cells;
  const uint8_t* bytecode = module->layout.functions.bytecode_data +
                            function->bytecode_offset_u32 +
                            sizeof(iree_vm_isa_control_block_record_t);
  iree_status_t status = iree_ok_status();

  IREE_VM_BYTECODE_DISPATCH_BEGIN();
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_BLOCK, control_block) {
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_control_block_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONTROL_RETURN, control_return) {
    status = iree_vm_bytecode_publish_results(module, function, &params->call,
                                              values, refs);
    if (iree_status_is_ok(status)) {
      *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    }
    IREE_VM_BYTECODE_DISPATCH_TERMINATE();
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
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ADD_I32, integer_add_i32) {
    const iree_vm_isa_integer_add_i32_record_t* record =
        (const iree_vm_isa_integer_add_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs + rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_add_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ADD_I64, integer_add_i64) {
    const iree_vm_isa_integer_add_i64_record_t* record =
        (const iree_vm_isa_integer_add_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs + rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_add_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_SUB_I32, integer_sub_i32) {
    const iree_vm_isa_integer_sub_i32_record_t* record =
        (const iree_vm_isa_integer_sub_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs - rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_sub_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_SUB_I64, integer_sub_i64) {
    const iree_vm_isa_integer_sub_i64_record_t* record =
        (const iree_vm_isa_integer_sub_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs - rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_sub_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MUL_I32, integer_mul_i32) {
    const iree_vm_isa_integer_mul_i32_record_t* record =
        (const iree_vm_isa_integer_mul_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs * rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_mul_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MUL_I64, integer_mul_i64) {
    const iree_vm_isa_integer_mul_i64_record_t* record =
        (const iree_vm_isa_integer_mul_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs * rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_mul_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_NEG_I32, integer_neg_i32) {
    const iree_vm_isa_integer_neg_i32_record_t* record =
        (const iree_vm_isa_integer_neg_i32_record_t*)record_data;
    const uint32_t source = (uint32_t)values[record->src_v8];
    values[record->dst_v8] = UINT32_C(0) - source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_neg_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_NEG_I64, integer_neg_i64) {
    const iree_vm_isa_integer_neg_i64_record_t* record =
        (const iree_vm_isa_integer_neg_i64_record_t*)record_data;
    const uint64_t source = values[record->src_v8];
    values[record->dst_v8] = UINT64_C(0) - source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_neg_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ABS_S32, integer_abs_s32) {
    const iree_vm_isa_integer_abs_s32_record_t* record =
        (const iree_vm_isa_integer_abs_s32_record_t*)record_data;
    const uint32_t source = (uint32_t)values[record->src_v8];
    values[record->dst_v8] =
        source & UINT32_C(0x80000000) ? UINT32_C(0) - source : source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_abs_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_ABS_S64, integer_abs_s64) {
    const iree_vm_isa_integer_abs_s64_record_t* record =
        (const iree_vm_isa_integer_abs_s64_record_t*)record_data;
    const uint64_t source = values[record->src_v8];
    values[record->dst_v8] =
        source & UINT64_C(0x8000000000000000) ? UINT64_C(0) - source : source;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_abs_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_S32, integer_min_s32) {
    const iree_vm_isa_integer_min_s32_record_t* record =
        (const iree_vm_isa_integer_min_s32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] =
        (lhs ^ UINT32_C(0x80000000)) < (rhs ^ UINT32_C(0x80000000)) ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_S64, integer_min_s64) {
    const iree_vm_isa_integer_min_s64_record_t* record =
        (const iree_vm_isa_integer_min_s64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = (lhs ^ UINT64_C(0x8000000000000000)) <
                                     (rhs ^ UINT64_C(0x8000000000000000))
                                 ? lhs
                                 : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_U32, integer_min_u32) {
    const iree_vm_isa_integer_min_u32_record_t* record =
        (const iree_vm_isa_integer_min_u32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs < rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MIN_U64, integer_min_u64) {
    const iree_vm_isa_integer_min_u64_record_t* record =
        (const iree_vm_isa_integer_min_u64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs < rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_min_u64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_S32, integer_max_s32) {
    const iree_vm_isa_integer_max_s32_record_t* record =
        (const iree_vm_isa_integer_max_s32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] =
        (lhs ^ UINT32_C(0x80000000)) > (rhs ^ UINT32_C(0x80000000)) ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_s32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_S64, integer_max_s64) {
    const iree_vm_isa_integer_max_s64_record_t* record =
        (const iree_vm_isa_integer_max_s64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = (lhs ^ UINT64_C(0x8000000000000000)) >
                                     (rhs ^ UINT64_C(0x8000000000000000))
                                 ? lhs
                                 : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_s64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_U32, integer_max_u32) {
    const iree_vm_isa_integer_max_u32_record_t* record =
        (const iree_vm_isa_integer_max_u32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs > rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_MAX_U64, integer_max_u64) {
    const iree_vm_isa_integer_max_u64_record_t* record =
        (const iree_vm_isa_integer_max_u64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs > rhs ? lhs : rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_max_u64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_AND_I32, integer_and_i32) {
    const iree_vm_isa_integer_and_i32_record_t* record =
        (const iree_vm_isa_integer_and_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs & rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_and_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_AND_I64, integer_and_i64) {
    const iree_vm_isa_integer_and_i64_record_t* record =
        (const iree_vm_isa_integer_and_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs & rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_and_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_OR_I32, integer_or_i32) {
    const iree_vm_isa_integer_or_i32_record_t* record =
        (const iree_vm_isa_integer_or_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs | rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_or_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_OR_I64, integer_or_i64) {
    const iree_vm_isa_integer_or_i64_record_t* record =
        (const iree_vm_isa_integer_or_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs | rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_or_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_XOR_I32, integer_xor_i32) {
    const iree_vm_isa_integer_xor_i32_record_t* record =
        (const iree_vm_isa_integer_xor_i32_record_t*)record_data;
    const uint32_t lhs = (uint32_t)values[record->lhs_v8];
    const uint32_t rhs = (uint32_t)values[record->rhs_v8];
    values[record->dst_v8] = lhs ^ rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_xor_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_XOR_I64, integer_xor_i64) {
    const iree_vm_isa_integer_xor_i64_record_t* record =
        (const iree_vm_isa_integer_xor_i64_record_t*)record_data;
    const uint64_t lhs = values[record->lhs_v8];
    const uint64_t rhs = values[record->rhs_v8];
    values[record->dst_v8] = lhs ^ rhs;
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_xor_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_COMPARE_I32, integer_compare_i32) {
    const iree_vm_isa_integer_compare_i32_record_t* record =
        (const iree_vm_isa_integer_compare_i32_record_t*)record_data;
    iree_vm_bytecode_execute_integer_compare_i32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_compare_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_COMPARE_I64, integer_compare_i64) {
    const iree_vm_isa_integer_compare_i64_record_t* record =
        (const iree_vm_isa_integer_compare_i64_record_t*)record_data;
    iree_vm_bytecode_execute_integer_compare_i64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_compare_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_LEA_I32, integer_lea_i32) {
    const iree_vm_isa_integer_lea_i32_record_t* record =
        (const iree_vm_isa_integer_lea_i32_record_t*)record_data;
    iree_vm_bytecode_execute_integer_lea_i32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_lea_i32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_LEA_I64, integer_lea_i64) {
    const iree_vm_isa_integer_lea_i64_record_t* record =
        (const iree_vm_isa_integer_lea_i64_record_t*)record_data;
    iree_vm_bytecode_execute_integer_lea_i64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_integer_lea_i64_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_CEILDIV_POW2_U32,
                                 integer_ceildiv_pow2_u32) {
    const iree_vm_isa_integer_ceildiv_pow2_u32_record_t* record =
        (const iree_vm_isa_integer_ceildiv_pow2_u32_record_t*)record_data;
    iree_vm_bytecode_execute_integer_ceildiv_pow2_u32(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_integer_ceildiv_pow2_u32_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(INTEGER_CEILDIV_POW2_U64,
                                 integer_ceildiv_pow2_u64) {
    const iree_vm_isa_integer_ceildiv_pow2_u64_record_t* record =
        (const iree_vm_isa_integer_ceildiv_pow2_u64_record_t*)record_data;
    iree_vm_bytecode_execute_integer_ceildiv_pow2_u64(record, values);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_integer_ceildiv_pow2_u64_record_t);
  }
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
  IREE_VM_BYTECODE_DISPATCH_CASE(BUFFER_RODATA_LOAD, buffer_rodata_load) {
    const iree_vm_isa_buffer_rodata_load_record_t* record =
        (const iree_vm_isa_buffer_rodata_load_record_t*)record_data;
    iree_vm_ref_t old_ref = refs[record->dst_r8];
    refs[record->dst_r8] = iree_vm_ref_from_ptr_borrowed(
        &module->rodata_roots[record->rodata_u16], module->buffer_type);
    iree_vm_ref_reset(&old_ref);
    IREE_VM_BYTECODE_DISPATCH_NEXT(iree_vm_isa_buffer_rodata_load_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_INTEGER, conversion_integer) {
    do {
      const iree_vm_isa_conversion_integer_record_t* record =
          (const iree_vm_isa_conversion_integer_record_t*)record_data;
      const uint32_t source = (uint32_t)values[record->src_v8];
      if (record->selector_u8 == IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64) {
        values[record->dst_v8] =
            source & 0x80000000u
                ? (uint64_t)source | UINT64_C(0xFFFFFFFF00000000)
                : source;
      } else {
        values[record->dst_v8] = source;
      }
      record_data += sizeof(*record);
    } while (IREE_VM_BYTECODE_DISPATCH_IS_SAME(CONVERSION_INTEGER));
    IREE_VM_BYTECODE_DISPATCH_CONTINUE();
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_FLOAT_EXTEND,
                                 conversion_float_extend) {
    const iree_vm_isa_conversion_float_extend_record_t* record =
        (const iree_vm_isa_conversion_float_extend_record_t*)record_data;
    values[record->dst_v8] =
        iree_vm_bytecode_bf16_to_f32_bits((uint16_t)values[record->src_v8]);
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_conversion_float_extend_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_CASE(CONVERSION_FLOAT_TO_INTEGER,
                                 conversion_float_to_integer) {
    const iree_vm_isa_conversion_float_to_integer_record_t* record =
        (const iree_vm_isa_conversion_float_to_integer_record_t*)record_data;
    uint32_t result = 0;
    status =
        iree_vm_bytecode_f32_to_u32((uint32_t)values[record->src_v8], &result);
    if (!iree_status_is_ok(status)) {
      IREE_VM_BYTECODE_DISPATCH_TERMINATE();
    }
    values[record->dst_v8] = result;
    IREE_VM_BYTECODE_DISPATCH_NEXT(
        iree_vm_isa_conversion_float_to_integer_record_t);
  }
  IREE_VM_BYTECODE_DISPATCH_END();

  iree_vm_bytecode_frame_reset(refs, function->ref_register_count_u16);
  iree_vm_invocation_stack_rewind(invocation, frame_checkpoint);
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
