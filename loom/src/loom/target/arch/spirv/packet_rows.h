// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned SPIR-V packet rows.
//
// These rows describe the exact value types and binary instruction forms for
// selected target-low descriptors. They are keyed by dense descriptor ordinal
// so target-low verification and emission can dispatch without string
// comparisons or descriptor-key switches.

#ifndef LOOM_TARGET_ARCH_SPIRV_PACKET_ROWS_H_
#define LOOM_TARGET_ARCH_SPIRV_PACKET_ROWS_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/value_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SPIRV_PACKET_IMMEDIATE_NONE UINT8_MAX

typedef enum loom_spirv_packet_form_e {
  LOOM_SPIRV_PACKET_FORM_UNSUPPORTED = 0,
  LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT = 1,
  LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE = 2,
  LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN = 3,
  LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED = 4,
  LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED = 5,
  LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD = 6,
  LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE = 7,
  LOOM_SPIRV_PACKET_FORM_SELECT = 8,
  LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT = 9,
  LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN = 10,
  LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD = 11,
  LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_STORE = 12,
  LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_MUL_ADD = 13,
  LOOM_SPIRV_PACKET_FORM_CONTROL_BARRIER = 14,
  LOOM_SPIRV_PACKET_FORM_ACCESS_CHAIN = 15,
} loom_spirv_packet_form_t;

typedef struct loom_spirv_packet_row_t {
  // SPIR-V instruction opcode.
  uint32_t opcode;
  // Emission algorithm selected for this descriptor.
  loom_spirv_packet_form_t form;
  // Result value type, or UNKNOWN for result-less packets.
  loom_spirv_value_type_t result_type;
  // Required value type for each packet operand.
  loom_spirv_value_type_t operand_types[3];
  // Expected packet result count.
  uint8_t result_count;
  // Expected packet operand count.
  uint8_t operand_count;
  // Descriptor-local immediate index read by the row.
  uint8_t immediate_index;
  // Number of literal words emitted for INTEGER_CONSTANT rows.
  uint8_t literal_word_count;
  // Alignment operand for aligned memory access rows.
  uint8_t memory_alignment;
  // SPIR-V BuiltIn enumerant for built-in load rows.
  uint32_t builtin;
  // Vector component extracted from built-in input rows.
  uint8_t component_index;
  // SPIR-V execution scope literal for barrier rows.
  uint32_t execution_scope;
  // SPIR-V memory scope literal for barrier rows.
  uint32_t memory_scope;
  // SPIR-V memory semantics mask literal for barrier rows.
  uint32_t memory_semantics;
  // Cooperative matrix layout literal for load/store rows.
  uint32_t cooperative_matrix_layout;
  // Cooperative matrix stride literal for load/store rows.
  uint32_t cooperative_matrix_stride;
  // Cooperative matrix operands mask literal for mul-add rows.
  uint32_t cooperative_matrix_operands;
} loom_spirv_packet_row_t;

// Returns the packet row for |descriptor_ordinal|, or NULL when the descriptor
// has no binary emission row yet.
const loom_spirv_packet_row_t* loom_spirv_packet_row_for_descriptor_ordinal(
    uint32_t descriptor_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_PACKET_ROWS_H_
