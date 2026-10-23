// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86-64 SysV source-value classification shared by lowering and emission.

#ifndef LOOM_TARGET_ARCH_X86_CALL_ABI_H_
#define LOOM_TARGET_ARCH_X86_CALL_ABI_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_x86_call_abi_class_e {
  LOOM_X86_CALL_ABI_CLASS_INTEGER = 0,
  LOOM_X86_CALL_ABI_CLASS_SSE = 1,
} loom_x86_call_abi_class_t;

// Inbound action required after transporting one platform value into its Low
// carrier. Outbound transport only exposes the significant low bits.
typedef enum loom_x86_call_abi_value_action_e {
  LOOM_X86_CALL_ABI_VALUE_ACTION_NONE = 0,
  LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I1 = 1,
  LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I8 = 2,
  LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I16 = 3,
} loom_x86_call_abi_value_action_t;

// Physical classes and stack properties for one source-typed ABI value.
typedef struct loom_x86_call_abi_classification_t {
  // Independent SysV register bank consumed by the value.
  loom_x86_call_abi_class_t abi_class;
  // Low register class used inside the function body.
  uint16_t carrier_register_class;
  // Platform register class used at the function boundary.
  uint16_t boundary_register_class;
  // Significant byte count in a register or stack slot.
  uint16_t byte_length;
  // Natural stack alignment in bytes.
  uint8_t byte_alignment;
  // loom_x86_call_abi_value_action_t required after inbound transport.
  uint8_t action;
} loom_x86_call_abi_classification_t;

// Exact physical placement retained after classifying one complete SysV
// function boundary. The signature describes each physical Low value while
// the parallel placement arrays identify its platform register or stack slot.
// A register value has a non-negative register index and offset -1. A stack
// argument has register -1 and a non-negative byte offset. Results are always
// direct registers; indirect source results are projected to an explicit
// hidden pointer argument before this contract is formed.
typedef struct loom_x86_call_abi_layout_t {
  // Physical source-semantic signature corresponding one-to-one with the Low
  // arguments and results.
  const loom_func_type_data_t* signature;
  // Exact platform register index for each physical argument, or -1 for stack.
  loom_i64_array_t argument_registers;
  // Exact incoming stack byte offset for each argument, or -1 for a register.
  loom_i64_array_t argument_offsets;
  // Exact platform register index for each physical result.
  loom_i64_array_t result_registers;
  // Complete incoming stack argument area in bytes.
  uint32_t stack_bytes;
  // Required alignment of the stack argument area, or zero when empty.
  uint8_t stack_alignment;
  // Whether exact SysV placement is present. A signature-only layout leaves
  // physical placement to the ordinary native ABI planner.
  bool is_exact_sysv;
} loom_x86_call_abi_layout_t;

// Builds the canonical exact SysV layout consumed by native x86 emission.
// Placement arrays correspond one-to-one with the physical semantic types and
// only need to remain live for the duration of this call. The returned slice
// borrows canonical module storage.
iree_status_t loom_x86_call_abi_layout_make(
    loom_module_t* module, const loom_type_t* argument_types,
    uint16_t argument_count, const loom_type_t* result_types,
    uint16_t result_count, int64_t* argument_registers,
    int64_t* argument_offsets, int64_t* result_registers, uint32_t stack_bytes,
    uint8_t stack_alignment, loom_named_attr_slice_t* out_layout);

// Parses and validates the shared x86 ABI layout vocabulary. Empty and
// signature-only layouts are admitted for ordinary Loom call boundaries.
// Exact layouts use a nested `sysv` dictionary with argument/result placement
// arrays and stack geometry. Returns false for malformed authored Low IR.
bool loom_x86_call_abi_layout_parse(const loom_module_t* module,
                                    loom_named_attr_slice_t layout,
                                    uint16_t argument_count,
                                    uint16_t result_count,
                                    loom_x86_call_abi_layout_t* out_layout);

// Classifies one source type admitted by the native x86-64 SysV boundary.
// Source views are transported as their projected data address. Compiler-only
// values and vectors without an exact architectural register width fail.
bool loom_x86_call_abi_classify_source_type(
    loom_type_t source_type,
    loom_x86_call_abi_classification_t* out_classification);

// Classifies one source type using an exact Low carrier register class. Most
// source types have one carrier. Narrow floating values additionally admit XMM
// so feature profiles with native scalar arithmetic can retain the platform
// representation inside the function body.
bool loom_x86_call_abi_classify_source_carrier(
    loom_type_t source_type, uint16_t carrier_register_class,
    loom_x86_call_abi_classification_t* out_classification);

// Returns the byte width of a general-purpose or SIMD call register class.
uint16_t loom_x86_call_abi_register_byte_length(uint16_t register_class);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_CALL_ABI_H_
