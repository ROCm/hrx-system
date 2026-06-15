// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM function-body bytecode emission from target-low packet tables.
//
// This emits the bytecode stream stored inside a VM function definition, not a
// complete VMFB/module archive. The surrounding module table, import/export
// records, and FlatBuffer container are owned by the later VM module emitter.

#ifndef LOOM_TARGET_EMIT_IREEVM_FUNCTION_BYTECODE_H_
#define LOOM_TARGET_EMIT_IREEVM_FUNCTION_BYTECODE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_ireevm_module_plan_t loom_ireevm_module_plan_t;

typedef enum loom_ireevm_register_bank_e {
  LOOM_IREEVM_REGISTER_BANK_I32 = 0,
  LOOM_IREEVM_REGISTER_BANK_REF = 1,
} loom_ireevm_register_bank_t;

typedef struct loom_ireevm_register_class_layout_t {
  // Human-readable target-low register class name used in diagnostics.
  iree_string_view_t class_name;
  // VM bytecode register bank storing values of this class.
  loom_ireevm_register_bank_t bank;
  // Number of VM bank slots occupied by one value of this class.
  uint16_t unit_count;
  // Required base ordinal alignment, in VM bank slots.
  uint16_t alignment;
} loom_ireevm_register_class_layout_t;

// Returns the VM bytecode storage layout for an ireevm.core target-low register
// class. The VM ABI has i32 and ref banks; i64/f64 values are aligned two-slot
// spans in the i32 bank instead of an independent bank.
iree_status_t loom_ireevm_register_class_layout(
    uint16_t register_class_id,
    loom_ireevm_register_class_layout_t* out_layout);

typedef struct loom_ireevm_function_bytecode_t {
  // Allocator-owned padded bytecode storage.
  uint8_t* data;
  // Number of bytes in |data|, including VM function-body padding.
  iree_host_size_t data_length;
  // Number of semantic bytecode bytes before trailing function-body padding.
  iree_host_size_t bytecode_length;
  // Number of VM basic blocks in the function body.
  uint16_t block_count;
  // Number of VM i32 register slots required by the emitted function.
  uint16_t i32_register_count;
  // Number of VM ref register slots required by the emitted function.
  uint16_t ref_register_count;
  // IREE VM FeatureBits required by opcodes in this function body.
  uint32_t feature_requirements;
} loom_ireevm_function_bytecode_t;

// Releases storage owned by |bytecode|. Safe to call on a zero-initialized
// bytecode object.
void loom_ireevm_function_bytecode_deinitialize(
    loom_ireevm_function_bytecode_t* bytecode, iree_allocator_t allocator);

// Emits IREE VM function-body bytecode for one scheduled and allocated
// low.func.def. The tables must describe the same function and target. The
// emitter supports unspilled ireevm.core target-id allocation for scalar
// register-bank packets; unsupported packets fail loud instead of silently
// producing partial bytecode.
iree_status_t loom_ireevm_emit_function_bytecode(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_ireevm_module_plan_t* module_plan, iree_allocator_t allocator,
    loom_ireevm_function_bytecode_t* out_bytecode);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_IREEVM_FUNCTION_BYTECODE_H_
