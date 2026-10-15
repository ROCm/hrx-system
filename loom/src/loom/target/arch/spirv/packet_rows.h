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
#include "loom/target/arch/spirv/extended_instruction.h"
#include "loom/target/arch/spirv/value_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SPIRV_PACKET_MAX_OPERAND_COUNT 4
#define LOOM_SPIRV_PACKET_OPERAND_TYPE_CAPACITY 3
// Maximum result width carried by a native OpVectorShuffle packet.
#define LOOM_SPIRV_PACKET_VECTOR_SHUFFLE_MAX_RESULT_LANES 4

enum loom_spirv_packet_form_e {
  LOOM_SPIRV_PACKET_FORM_UNSUPPORTED = 0,
  LOOM_SPIRV_PACKET_FORM_SCALAR_CONSTANT = 1,
  // Binary instruction whose result has the first operand's type. The row
  // owns both operand types, which need not match (OpVectorTimesScalar).
  LOOM_SPIRV_PACKET_FORM_BINARY_LHS_TYPE = 2,
  LOOM_SPIRV_PACKET_FORM_PHYSICAL_STORAGE_BUFFER_BYTE_OFFSET = 3,
  LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED = 4,
  LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED = 5,
  LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD = 6,
  LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE = 7,
  LOOM_SPIRV_PACKET_FORM_SELECT = 8,
  LOOM_SPIRV_PACKET_FORM_UNARY_TYPED = 9,
  LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN = 10,
  LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD = 11,
  LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_STORE = 12,
  LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_MUL_ADD = 13,
  LOOM_SPIRV_PACKET_FORM_CONTROL_BARRIER = 14,
  LOOM_SPIRV_PACKET_FORM_ACCESS_CHAIN = 15,
  LOOM_SPIRV_PACKET_FORM_BOOLEAN_CONSTANT = 16,
  LOOM_SPIRV_PACKET_FORM_COMPOSITE_CONSTRUCT = 17,
  LOOM_SPIRV_PACKET_FORM_COMPOSITE_EXTRACT = 18,
  LOOM_SPIRV_PACKET_FORM_COMPOSITE_INSERT = 19,
  LOOM_SPIRV_PACKET_FORM_ATOMIC = 20,
  LOOM_SPIRV_PACKET_FORM_ATOMIC_COMPARE_EXCHANGE = 21,
  LOOM_SPIRV_PACKET_FORM_ATOMIC_FLOAT_BITCAST = 22,
  LOOM_SPIRV_PACKET_FORM_ATOMIC_FLOAT_CAS = 23,
  LOOM_SPIRV_PACKET_FORM_ATOMIC_FLOAT_COMPARE_EXCHANGE = 24,
  LOOM_SPIRV_PACKET_FORM_EXTENDED_INSTRUCTION = 25,
  // Typed subgroup instruction with one value operand after its scope.
  LOOM_SPIRV_PACKET_FORM_GROUP_NON_UNIFORM_UNARY = 26,
  // Typed subgroup arithmetic with a GroupOperation literal before its value.
  LOOM_SPIRV_PACKET_FORM_GROUP_NON_UNIFORM_ARITHMETIC = 27,
  // Typed two-input vector shuffle with one selector per result lane.
  LOOM_SPIRV_PACKET_FORM_VECTOR_SHUFFLE = 28,
};
typedef uint8_t loom_spirv_packet_form_t;

// Ordinal into the generated packet value-type table. Zero names UNKNOWN.
typedef uint16_t loom_spirv_packet_value_type_ref_t;

#define LOOM_SPIRV_PACKET_VALUE_TYPE_REF_UNKNOWN \
  ((loom_spirv_packet_value_type_ref_t)0)

enum loom_spirv_packet_flag_bits_e {
  // Preserves the operation's separate rounding and arithmetic ordering.
  LOOM_SPIRV_PACKET_FLAG_NO_CONTRACTION = 1u << 0,
};
typedef uint8_t loom_spirv_packet_flags_t;

// Dense module-cache and entry-interface slots for shader Input builtins.
enum loom_spirv_builtin_variable_e {
  LOOM_SPIRV_BUILTIN_VARIABLE_WORKGROUP_ID = 0,
  LOOM_SPIRV_BUILTIN_VARIABLE_WORKITEM_ID = 1,
  LOOM_SPIRV_BUILTIN_VARIABLE_WORKITEM_DISPATCH_ID = 2,
  LOOM_SPIRV_BUILTIN_VARIABLE_WORKGROUP_COUNT = 3,
  LOOM_SPIRV_BUILTIN_VARIABLE_SUBGROUP_LANE_ID = 4,
  LOOM_SPIRV_BUILTIN_VARIABLE_SUBGROUP_SIZE = 5,
  LOOM_SPIRV_BUILTIN_VARIABLE_SUBGROUP_ID = 6,
  LOOM_SPIRV_BUILTIN_VARIABLE_SUBGROUP_COUNT = 7,
  LOOM_SPIRV_BUILTIN_VARIABLE_COUNT = 8,
};
typedef uint8_t loom_spirv_builtin_variable_t;

typedef struct loom_spirv_builtin_variable_info_t {
  // Debug name assigned to the SPIR-V Input variable.
  iree_string_view_t name;
  // SPIR-V BuiltIn enumerant decorating the variable.
  uint32_t builtin;
  // Number of u32 components in the builtin value.
  uint8_t component_count;
} loom_spirv_builtin_variable_info_t;

// Generated variable facts indexed by loom_spirv_builtin_variable_t.
extern const loom_spirv_builtin_variable_info_t
    loom_spirv_builtin_variables[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT];

// Form-specific packet literals interned independently of descriptor and type.
// The packet row's form selects the active member.
typedef union loom_spirv_packet_payload_t {
  // Logical access-chain coordinate units.
  struct {
    // Exact right shift converting a byte offset to an element index.
    // Byte-coordinate packets require an offset divisible by the element
    // width. Zero also represents an already-materialized element index.
    uint8_t coordinate_byte_shift;
  } access_chain;
  // Scalar-constant packet literals.
  struct {
    // Number of literal words emitted for the constant.
    uint8_t literal_word_count;
  } scalar_constant;
  // Built-in input packet literals.
  struct {
    // Dense builtin-variable table and cache slot.
    loom_spirv_builtin_variable_t variable;
    // Vector component extracted from the built-in input.
    uint8_t component_index;
  } builtin_load;
  // Control-barrier packet literals.
  struct {
    // SPIR-V execution scope literal.
    uint32_t execution_scope;
    // SPIR-V memory scope literal.
    uint32_t memory_scope;
    // SPIR-V memory semantics mask literal.
    uint32_t memory_semantics;
  } barrier;
  // Group-non-uniform instruction literals.
  struct {
    // SPIR-V execution scope literal.
    uint32_t execution_scope;
    // SPIR-V GroupOperation literal for arithmetic packets.
    uint32_t operation;
  } group_non_uniform;
  // Cooperative-matrix packet literals.
  struct {
    // Cooperative matrix operands mask literal for mul-add rows.
    uint32_t operands;
  } cooperative_matrix;
  // Atomic instruction literals.
  struct {
    // SPIR-V memory Scope enumerant.
    uint32_t memory_scope;
    // Storage-class Memory Semantics bit used for ordered accesses.
    uint32_t storage_semantics;
    // Success ordering ordinal for compare-exchange packets.
    uint8_t success_ordering;
    // Integer scalar type used to bitcast floating-point atomics.
    uint8_t integer_scalar;
    // Floating-point combine operation selected for a CAS loop.
    uint8_t float_operation;
  } atomic;
  // Extended-instruction packet literals.
  struct {
    // Imported instruction set selected for OpExtInst.
    loom_spirv_extended_instruction_set_t instruction_set;
    // Instruction enumerant defined by the selected instruction set.
    uint32_t instruction;
  } extended_instruction;
} loom_spirv_packet_payload_t;

static_assert(sizeof(loom_spirv_packet_payload_t) == 12,
              "SPIR-V packet payloads must remain compact");

typedef struct loom_spirv_packet_row_t {
  // SPIR-V instruction opcode, occupying the low 16 bits of its binary header.
  uint16_t opcode;
  // Ordinal into the shared payload table. Zero names the empty payload.
  uint16_t payload_ref;
  // Sole result value-type table ref, or UNKNOWN for result-less packets.
  loom_spirv_packet_value_type_ref_t result_type_ref;
  // Required value-type refs for packet operands. Four-operand packets use one
  // repeated type because the inline type-ref capacity is three.
  loom_spirv_packet_value_type_ref_t
      operand_type_refs[LOOM_SPIRV_PACKET_OPERAND_TYPE_CAPACITY];
  // Emission algorithm selected for this descriptor.
  loom_spirv_packet_form_t form;
  // Expected packet operand count.
  uint8_t operand_count;
  // Alignment operand for aligned memory access rows.
  uint8_t memory_alignment;
  // Semantic emission requirements of the selected packet.
  loom_spirv_packet_flags_t flags;
} loom_spirv_packet_row_t;

static_assert(sizeof(loom_spirv_packet_row_t) == 16,
              "SPIR-V packet rows must remain compact");

// Generated native literals interned across all packet rows.
extern const loom_spirv_packet_payload_t loom_spirv_packet_payloads[];

// Generated value types interned across all packet rows.
extern const loom_spirv_value_type_t loom_spirv_packet_value_types[];

// Returns the form-specific literals required by |row|.
static inline const loom_spirv_packet_payload_t* loom_spirv_packet_row_payload(
    const loom_spirv_packet_row_t* row) {
  return &loom_spirv_packet_payloads[row->payload_ref];
}

// Returns the result value type required by |row|.
static inline loom_spirv_value_type_t loom_spirv_packet_row_result_type(
    const loom_spirv_packet_row_t* row) {
  return loom_spirv_packet_value_types[row->result_type_ref];
}

// Returns the required value type for packet operand |operand_index|.
static inline loom_spirv_value_type_t loom_spirv_packet_row_operand_type(
    const loom_spirv_packet_row_t* row, uint8_t operand_index) {
  const uint8_t type_index =
      row->operand_count > LOOM_SPIRV_PACKET_OPERAND_TYPE_CAPACITY
          ? 0
          : operand_index;
  return loom_spirv_packet_value_types[row->operand_type_refs[type_index]];
}

// Returns the packet row for |descriptor_ordinal|, or NULL when the descriptor
// has no binary emission row yet.
const loom_spirv_packet_row_t* loom_spirv_packet_row_for_descriptor_ordinal(
    uint32_t descriptor_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_PACKET_ROWS_H_
