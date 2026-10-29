// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/approximate_f32.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

static_assert((uint8_t)LOOM_SCALAR_CMPF_PREDICATE_OEQ ==
                  (uint8_t)LOOM_VECTOR_CMPF_PREDICATE_OEQ,
              "scalar and vector ordered-equal predicates must match");
static_assert((uint8_t)LOOM_SCALAR_CMPF_PREDICATE_OGT ==
                  (uint8_t)LOOM_VECTOR_CMPF_PREDICATE_OGT,
              "scalar and vector ordered-greater predicates must match");
static_assert((uint8_t)LOOM_SCALAR_CMPF_PREDICATE_OLT ==
                  (uint8_t)LOOM_VECTOR_CMPF_PREDICATE_OLT,
              "scalar and vector ordered-less predicates must match");
static_assert((uint8_t)LOOM_SCALAR_CMPI_PREDICATE_EQ ==
                  (uint8_t)LOOM_VECTOR_CMPI_PREDICATE_EQ,
              "scalar and vector equal predicates must match");
static_assert((uint8_t)LOOM_SCALAR_CMPI_PREDICATE_NE ==
                  (uint8_t)LOOM_VECTOR_CMPI_PREDICATE_NE,
              "scalar and vector not-equal predicates must match");
static_assert((uint8_t)LOOM_SCALAR_CMPI_PREDICATE_ULT ==
                  (uint8_t)LOOM_VECTOR_CMPI_PREDICATE_ULT,
              "scalar and vector unsigned-less predicates must match");
static_assert((uint8_t)LOOM_SCALAR_CMPI_PREDICATE_UGE ==
                  (uint8_t)LOOM_VECTOR_CMPI_PREDICATE_UGE,
              "scalar and vector unsigned-greater-equal predicates must match");

typedef uint8_t loom_math_approximate_f32_attribute_mode_t;
enum loom_math_approximate_f32_attribute_mode_e {
  LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE = 0,
  LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_CONSTANT_F32 = 1,
  LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_CONSTANT_I32 = 2,
  LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_ENUM = 3,
};

typedef uint8_t loom_math_approximate_f32_opcode_t;

typedef struct loom_math_approximate_f32_instruction_descriptor_t {
  // Operation emitted when evaluating a scalar source value.
  loom_op_kind_t scalar_op_kind;
  // Operation emitted when evaluating a vector source value.
  loom_op_kind_t vector_op_kind;
  // Number of leading instruction arguments that reference SSA values.
  uint8_t operand_count;
  // Element type of the instruction's single result.
  loom_scalar_type_t result_element_type;
  // Interpretation of the instruction's non-operand argument, when present.
  loom_math_approximate_f32_attribute_mode_t attribute_mode;
  // Source fast-math flags admitted by this operation kind.
  uint8_t fastmath_mask;
} loom_math_approximate_f32_instruction_descriptor_t;
static_assert(sizeof(loom_math_approximate_f32_instruction_descriptor_t) == 8,
              "approximate F32 descriptors must remain compact");

typedef struct loom_math_approximate_f32_instruction_t {
  // Index of the shared scalar/vector operation descriptor.
  loom_math_approximate_f32_opcode_t opcode;
  // Local SSA value indices or an attribute payload index.
  uint8_t arguments[3];
} loom_math_approximate_f32_instruction_t;
static_assert(sizeof(loom_math_approximate_f32_instruction_t) == 4,
              "approximate F32 instructions must remain compact");

typedef struct loom_math_approximate_f32_program_t {
  // First instruction in the shared instruction table.
  uint16_t instruction_offset;
  // Number of instructions in this program.
  uint8_t instruction_count;
  // Local SSA value index returned by this program.
  uint8_t result_value;
} loom_math_approximate_f32_program_t;
static_assert(sizeof(loom_math_approximate_f32_program_t) == 4,
              "approximate F32 programs must remain compact");

// Shared exact bit patterns referenced by the three recipe programs.
#define LOOM_MATH_APPROXIMATE_F32_CONSTANTS(_) \
  _(ZERO_BITS, 0x00000000)                     \
  _(I32_ONE, 0x00000001)                       \
  _(I32_TWO, 0x00000002)                       \
  _(I32_THREE, 0x00000003)                     \
  _(I32_SIX, 0x00000006)                       \
  _(EXPONENT_SHIFT, 0x00000017)                \
  _(QUARTER_PHASE_SHIFT, 0x0000001E)           \
  _(SIGN_SHIFT, 0x0000001F)                    \
  _(SIGNIFICAND_FRACTION_MASK, 0x007FFFFF)     \
  _(SIGNIFICAND_HIDDEN_BIT, 0x00800000)        \
  _(EXP2_COEFFICIENT_0, 0x3AACCBBD)            \
  _(EXP2_COEFFICIENT_1, 0x3C1E74F2)            \
  _(EXP2_COEFFICIENT_2, 0x3D635E99)            \
  _(SINE_COEFFICIENT_4, 0x3DA83C18)            \
  _(EXP2_COEFFICIENT_3, 0x3E75FCDE)            \
  _(F32_HALF, 0x3F000000)                      \
  _(EXP2_COEFFICIENT_4, 0x3F317211)            \
  _(F32_ONE, 0x3F800000)                       \
  _(F32_TWO, 0x40000000)                       \
  _(SINE_COEFFICIENT_2, 0x402335E3)            \
  _(SINE_COEFFICIENT_0, 0x40490FDB)            \
  _(EXP2_ORDINARY_BOUND, 0x42FD0000)           \
  _(EXP2_FAR_BOUND, 0x43400000)                \
  _(TRIG_CARDINAL_BOUND_BITS, 0x4A000000)      \
  _(TRIG_MIDDLE_BOUND_BITS, 0x4A800000)        \
  _(TRIG_UPPER_BOUND_BITS, 0x4B000000)         \
  _(MAGIC_ROUND_BIAS, 0x4B400000)              \
  _(EXP2_SPLIT_BIAS_BITS, 0x7F000000)          \
  _(F32_INFINITY_BITS, 0x7F800000)             \
  _(F32_QUIET_NAN_BITS, 0x7FC00000)            \
  _(F32_MAGNITUDE_MASK, 0x7FFFFFFF)            \
  _(F32_SIGN_MASK, 0x80000000)                 \
  _(SINE_COEFFICIENT_5, 0xBBF183A8)            \
  _(SINE_COEFFICIENT_3, 0xBF196966)            \
  _(SINE_COEFFICIENT_1, 0xC0A55DE7)

#define LOOM_MATH_APPROXIMATE_F32_CONSTANT_ENUM(name, bits) \
  LOOM_MATH_APPROXIMATE_F32_CONSTANT_##name,
enum loom_math_approximate_f32_constant_e {
  LOOM_MATH_APPROXIMATE_F32_CONSTANTS(LOOM_MATH_APPROXIMATE_F32_CONSTANT_ENUM)
      LOOM_MATH_APPROXIMATE_F32_CONSTANT_COUNT_,
};
#undef LOOM_MATH_APPROXIMATE_F32_CONSTANT_ENUM

#define LOOM_MATH_APPROXIMATE_F32_CONSTANT_ROW(name, bits) \
  [LOOM_MATH_APPROXIMATE_F32_CONSTANT_##name] = UINT32_C(bits),
static const uint32_t
    kApproximateF32Constants[LOOM_MATH_APPROXIMATE_F32_CONSTANT_COUNT_] = {
        LOOM_MATH_APPROXIMATE_F32_CONSTANTS(
            LOOM_MATH_APPROXIMATE_F32_CONSTANT_ROW)};
#undef LOOM_MATH_APPROXIMATE_F32_CONSTANT_ROW
#undef LOOM_MATH_APPROXIMATE_F32_CONSTANTS

// Scalar/vector operation pairs admitted by the checked-in recipe programs.
#define LOOM_MATH_APPROXIMATE_F32_OPS(_)                                       \
  _(CONSTANT_F32, LOOM_OP_SCALAR_CONSTANT, LOOM_OP_VECTOR_SPLATF, 0,           \
    LOOM_SCALAR_TYPE_F32, LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_CONSTANT_F32, 0) \
  _(CONSTANT_I32, LOOM_OP_SCALAR_CONSTANT, LOOM_OP_VECTOR_SPLATI, 0,           \
    LOOM_SCALAR_TYPE_I32, LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_CONSTANT_I32, 0) \
  _(BITCAST_F32, LOOM_OP_SCALAR_BITCAST, LOOM_OP_VECTOR_BITCAST, 1,            \
    LOOM_SCALAR_TYPE_F32, LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)         \
  _(BITCAST_I32, LOOM_OP_SCALAR_BITCAST, LOOM_OP_VECTOR_BITCAST, 1,            \
    LOOM_SCALAR_TYPE_I32, LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)         \
  _(ABSF, LOOM_OP_SCALAR_ABSF, LOOM_OP_VECTOR_ABSF, 1, LOOM_SCALAR_TYPE_F32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE,                                  \
    LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)                                       \
  _(ADDF, LOOM_OP_SCALAR_ADDF, LOOM_OP_VECTOR_ADDF, 2, LOOM_SCALAR_TYPE_F32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE,                                  \
    LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)                                       \
  _(SUBF, LOOM_OP_SCALAR_SUBF, LOOM_OP_VECTOR_SUBF, 2, LOOM_SCALAR_TYPE_F32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE,                                  \
    LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)                                       \
  _(MULF, LOOM_OP_SCALAR_MULF, LOOM_OP_VECTOR_MULF, 2, LOOM_SCALAR_TYPE_F32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE,                                  \
    LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)                                       \
  _(FMAF, LOOM_OP_SCALAR_FMAF, LOOM_OP_VECTOR_FMAF, 3, LOOM_SCALAR_TYPE_F32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE,                                  \
    LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)                                       \
  _(ADDI, LOOM_OP_SCALAR_ADDI, LOOM_OP_VECTOR_ADDI, 2, LOOM_SCALAR_TYPE_I32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)                               \
  _(SUBI, LOOM_OP_SCALAR_SUBI, LOOM_OP_VECTOR_SUBI, 2, LOOM_SCALAR_TYPE_I32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)                               \
  _(ANDI, LOOM_OP_SCALAR_ANDI, LOOM_OP_VECTOR_ANDI, 2, LOOM_SCALAR_TYPE_I32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)                               \
  _(ORI, LOOM_OP_SCALAR_ORI, LOOM_OP_VECTOR_ORI, 2, LOOM_SCALAR_TYPE_I32,      \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)                               \
  _(XORI, LOOM_OP_SCALAR_XORI, LOOM_OP_VECTOR_XORI, 2, LOOM_SCALAR_TYPE_I32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)                               \
  _(SHLI, LOOM_OP_SCALAR_SHLI, LOOM_OP_VECTOR_SHLI, 2, LOOM_SCALAR_TYPE_I32,   \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)                               \
  _(SHRUI, LOOM_OP_SCALAR_SHRUI, LOOM_OP_VECTOR_SHRUI, 2,                      \
    LOOM_SCALAR_TYPE_I32, LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)         \
  _(CMPF, LOOM_OP_SCALAR_CMPF, LOOM_OP_VECTOR_CMPF, 2, LOOM_SCALAR_TYPE_I1,    \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_ENUM,                                  \
    LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)                                       \
  _(CMPI, LOOM_OP_SCALAR_CMPI, LOOM_OP_VECTOR_CMPI, 2, LOOM_SCALAR_TYPE_I1,    \
    LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_ENUM, 0)                               \
  _(SELECT_F32, LOOM_OP_SCF_SELECT, LOOM_OP_VECTOR_SELECT, 3,                  \
    LOOM_SCALAR_TYPE_F32, LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)         \
  _(SELECT_I32, LOOM_OP_SCF_SELECT, LOOM_OP_VECTOR_SELECT, 3,                  \
    LOOM_SCALAR_TYPE_I32, LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE, 0)

#define LOOM_MATH_APPROXIMATE_F32_OPCODE_ENUM(name, scalar_op, vector_op,    \
                                              operand_count, result_type,    \
                                              attribute_mode, fastmath_mask) \
  LOOM_MATH_APPROXIMATE_F32_OPCODE_##name,
enum loom_math_approximate_f32_opcode_e {
  LOOM_MATH_APPROXIMATE_F32_OPS(LOOM_MATH_APPROXIMATE_F32_OPCODE_ENUM)
      LOOM_MATH_APPROXIMATE_F32_OPCODE_COUNT_,
};
#undef LOOM_MATH_APPROXIMATE_F32_OPCODE_ENUM

#define LOOM_MATH_APPROXIMATE_F32_DESCRIPTOR_ROW(                             \
    name, scalar_op, vector_op, operand_count_, result_type, attribute_mode_, \
    fastmath_mask_)                                                           \
  [LOOM_MATH_APPROXIMATE_F32_OPCODE_##name] = {                               \
      .scalar_op_kind = (scalar_op),                                          \
      .vector_op_kind = (vector_op),                                          \
      .operand_count = (operand_count_),                                      \
      .result_element_type = (result_type),                                   \
      .attribute_mode = (attribute_mode_),                                    \
      .fastmath_mask = (fastmath_mask_),                                      \
  },
static const loom_math_approximate_f32_instruction_descriptor_t
    kApproximateF32InstructionDescriptors
        [LOOM_MATH_APPROXIMATE_F32_OPCODE_COUNT_] = {
            LOOM_MATH_APPROXIMATE_F32_OPS(
                LOOM_MATH_APPROXIMATE_F32_DESCRIPTOR_ROW)};
#undef LOOM_MATH_APPROXIMATE_F32_DESCRIPTOR_ROW
#undef LOOM_MATH_APPROXIMATE_F32_OPS

// Assigns local value IDs directly from each authored recipe row order.
#define LOOM_MATH_APPROXIMATE_F32_CONSTANT(...) \
  LOOM_MATH_APPROXIMATE_F32_VALUE(__VA_ARGS__)
#define LOOM_MATH_APPROXIMATE_F32_UNARY(...) \
  LOOM_MATH_APPROXIMATE_F32_VALUE(__VA_ARGS__)
#define LOOM_MATH_APPROXIMATE_F32_BINARY(...) \
  LOOM_MATH_APPROXIMATE_F32_VALUE(__VA_ARGS__)
#define LOOM_MATH_APPROXIMATE_F32_TERNARY(...) \
  LOOM_MATH_APPROXIMATE_F32_VALUE(__VA_ARGS__)
#define LOOM_MATH_APPROXIMATE_F32_COMPARE(...) \
  LOOM_MATH_APPROXIMATE_F32_VALUE(__VA_ARGS__)

#define LOOM_MATH_APPROXIMATE_F32_VALUE(result, ...) \
  LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_##result,
enum loom_math_approximate_f32_exp2_value_e {
  LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_VALUE = 0,
#include "loom/transforms/math/approximate_f32_exp2.inl"
  LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_COUNT_,
};
#undef LOOM_MATH_APPROXIMATE_F32_VALUE

#define LOOM_MATH_APPROXIMATE_F32_VALUE(result, ...) \
  LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_##result,
enum loom_math_approximate_f32_sin_value_e {
  LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_TURNS = 0,
#include "loom/transforms/math/approximate_f32_sin_turns.inl"
  LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_COUNT_,
};
#undef LOOM_MATH_APPROXIMATE_F32_VALUE

#define LOOM_MATH_APPROXIMATE_F32_VALUE(result, ...) \
  LOOM_MATH_APPROXIMATE_F32_COS_VALUE_##result,
enum loom_math_approximate_f32_cos_value_e {
  LOOM_MATH_APPROXIMATE_F32_COS_VALUE_TURNS = 0,
#include "loom/transforms/math/approximate_f32_cos_turns.inl"
  LOOM_MATH_APPROXIMATE_F32_COS_VALUE_COUNT_,
};
#undef LOOM_MATH_APPROXIMATE_F32_VALUE

#undef LOOM_MATH_APPROXIMATE_F32_COMPARE
#undef LOOM_MATH_APPROXIMATE_F32_TERNARY
#undef LOOM_MATH_APPROXIMATE_F32_BINARY
#undef LOOM_MATH_APPROXIMATE_F32_UNARY
#undef LOOM_MATH_APPROXIMATE_F32_CONSTANT

// Rejects forward references at build time. The evaluator can consequently
// consume the checked-in programs as trusted compiler data.
#define LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, operand) \
  static_assert(LOOM_MATH_APPROXIMATE_F32_VALUE(operand) <          \
                    LOOM_MATH_APPROXIMATE_F32_VALUE(result),        \
                #operand " must precede " #result)
#define LOOM_MATH_APPROXIMATE_F32_CONSTANT(result, type, constant)
#define LOOM_MATH_APPROXIMATE_F32_UNARY(result, opcode, input) \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, input);
#define LOOM_MATH_APPROXIMATE_F32_BINARY(result, opcode, lhs, rhs) \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, lhs);         \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, rhs);
#define LOOM_MATH_APPROXIMATE_F32_TERNARY(result, opcode, operand0, operand1, \
                                          operand2)                           \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, operand0);               \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, operand1);               \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, operand2);
#define LOOM_MATH_APPROXIMATE_F32_COMPARE(result, opcode, predicate, lhs, rhs) \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, lhs);                     \
  LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND(result, rhs);

#define LOOM_MATH_APPROXIMATE_F32_VALUE(name) \
  LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_##name
#include "loom/transforms/math/approximate_f32_exp2.inl"
#undef LOOM_MATH_APPROXIMATE_F32_VALUE
#define LOOM_MATH_APPROXIMATE_F32_VALUE(name) \
  LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_##name
#include "loom/transforms/math/approximate_f32_sin_turns.inl"
#undef LOOM_MATH_APPROXIMATE_F32_VALUE
#define LOOM_MATH_APPROXIMATE_F32_VALUE(name) \
  LOOM_MATH_APPROXIMATE_F32_COS_VALUE_##name
#include "loom/transforms/math/approximate_f32_cos_turns.inl"
#undef LOOM_MATH_APPROXIMATE_F32_VALUE

#undef LOOM_MATH_APPROXIMATE_F32_COMPARE
#undef LOOM_MATH_APPROXIMATE_F32_TERNARY
#undef LOOM_MATH_APPROXIMATE_F32_BINARY
#undef LOOM_MATH_APPROXIMATE_F32_UNARY
#undef LOOM_MATH_APPROXIMATE_F32_CONSTANT
#undef LOOM_MATH_APPROXIMATE_F32_VALIDATE_OPERAND

enum {
  LOOM_MATH_APPROXIMATE_F32_EXP2_INSTRUCTION_COUNT =
      LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_COUNT_ - 1,
  LOOM_MATH_APPROXIMATE_F32_SIN_INSTRUCTION_COUNT =
      LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_COUNT_ - 1,
  LOOM_MATH_APPROXIMATE_F32_COS_INSTRUCTION_COUNT =
      LOOM_MATH_APPROXIMATE_F32_COS_VALUE_COUNT_ - 1,
  LOOM_MATH_APPROXIMATE_F32_EXP2_INSTRUCTION_OFFSET = 0,
  LOOM_MATH_APPROXIMATE_F32_SIN_INSTRUCTION_OFFSET =
      LOOM_MATH_APPROXIMATE_F32_EXP2_INSTRUCTION_COUNT,
  LOOM_MATH_APPROXIMATE_F32_COS_INSTRUCTION_OFFSET =
      LOOM_MATH_APPROXIMATE_F32_SIN_INSTRUCTION_OFFSET +
      LOOM_MATH_APPROXIMATE_F32_SIN_INSTRUCTION_COUNT,
  LOOM_MATH_APPROXIMATE_F32_INSTRUCTION_COUNT =
      LOOM_MATH_APPROXIMATE_F32_COS_INSTRUCTION_OFFSET +
      LOOM_MATH_APPROXIMATE_F32_COS_INSTRUCTION_COUNT,
  LOOM_MATH_APPROXIMATE_F32_VALUE_CAPACITY =
      LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_COUNT_,
};
static_assert((LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_COUNT_ + 0) >=
                      (LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_COUNT_ + 0) &&
                  (LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_COUNT_ + 0) >=
                      (LOOM_MATH_APPROXIMATE_F32_COS_VALUE_COUNT_ + 0),
              "sine recipe must define the maximum value capacity");
static_assert(LOOM_MATH_APPROXIMATE_F32_VALUE_CAPACITY <= UINT8_MAX,
              "approximate F32 value indices must fit in one byte");
static_assert(LOOM_MATH_APPROXIMATE_F32_CONSTANT_COUNT_ <= UINT8_MAX,
              "approximate F32 constant indices must fit in one byte");

#define LOOM_MATH_APPROXIMATE_F32_CONSTANT_ID(name) \
  LOOM_MATH_APPROXIMATE_F32_CONSTANT_##name
#define LOOM_MATH_APPROXIMATE_F32_OPCODE(name) \
  LOOM_MATH_APPROXIMATE_F32_OPCODE_##name
#define LOOM_MATH_APPROXIMATE_F32_PREDICATE_(opcode, predicate) \
  LOOM_SCALAR_##opcode##_PREDICATE_##predicate
#define LOOM_MATH_APPROXIMATE_F32_PREDICATE(opcode, predicate) \
  LOOM_MATH_APPROXIMATE_F32_PREDICATE_(opcode, predicate)
#define LOOM_MATH_APPROXIMATE_F32_INSTRUCTION(opcode_, arg0, arg1, arg2) \
  {                                                                      \
      .opcode = (uint8_t)LOOM_MATH_APPROXIMATE_F32_OPCODE(opcode_),      \
      .arguments = {(uint8_t)(arg0), (uint8_t)(arg1), (uint8_t)(arg2)},  \
  },
#define LOOM_MATH_APPROXIMATE_F32_CONSTANT(result, type, constant)      \
  LOOM_MATH_APPROXIMATE_F32_INSTRUCTION(                                \
      CONSTANT_##type, LOOM_MATH_APPROXIMATE_F32_CONSTANT_ID(constant), \
      UINT8_MAX, UINT8_MAX)
#define LOOM_MATH_APPROXIMATE_F32_UNARY(result, opcode, input) \
  LOOM_MATH_APPROXIMATE_F32_INSTRUCTION(                       \
      opcode, LOOM_MATH_APPROXIMATE_F32_VALUE(input), UINT8_MAX, UINT8_MAX)
#define LOOM_MATH_APPROXIMATE_F32_BINARY(result, opcode, lhs, rhs) \
  LOOM_MATH_APPROXIMATE_F32_INSTRUCTION(                           \
      opcode, LOOM_MATH_APPROXIMATE_F32_VALUE(lhs),                \
      LOOM_MATH_APPROXIMATE_F32_VALUE(rhs), UINT8_MAX)
#define LOOM_MATH_APPROXIMATE_F32_TERNARY(result, opcode, operand0, operand1, \
                                          operand2)                           \
  LOOM_MATH_APPROXIMATE_F32_INSTRUCTION(                                      \
      opcode, LOOM_MATH_APPROXIMATE_F32_VALUE(operand0),                      \
      LOOM_MATH_APPROXIMATE_F32_VALUE(operand1),                              \
      LOOM_MATH_APPROXIMATE_F32_VALUE(operand2))
#define LOOM_MATH_APPROXIMATE_F32_COMPARE(result, opcode, predicate, lhs, rhs) \
  LOOM_MATH_APPROXIMATE_F32_INSTRUCTION(                                       \
      opcode, LOOM_MATH_APPROXIMATE_F32_VALUE(lhs),                            \
      LOOM_MATH_APPROXIMATE_F32_VALUE(rhs),                                    \
      LOOM_MATH_APPROXIMATE_F32_PREDICATE(opcode, predicate))

static const loom_math_approximate_f32_instruction_t
    kApproximateF32Instructions[] = {
#define LOOM_MATH_APPROXIMATE_F32_VALUE(name) \
  LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_##name
#include "loom/transforms/math/approximate_f32_exp2.inl"
#undef LOOM_MATH_APPROXIMATE_F32_VALUE
#define LOOM_MATH_APPROXIMATE_F32_VALUE(name) \
  LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_##name
#include "loom/transforms/math/approximate_f32_sin_turns.inl"
#undef LOOM_MATH_APPROXIMATE_F32_VALUE
#define LOOM_MATH_APPROXIMATE_F32_VALUE(name) \
  LOOM_MATH_APPROXIMATE_F32_COS_VALUE_##name
#include "loom/transforms/math/approximate_f32_cos_turns.inl"
#undef LOOM_MATH_APPROXIMATE_F32_VALUE
};

#undef LOOM_MATH_APPROXIMATE_F32_COMPARE
#undef LOOM_MATH_APPROXIMATE_F32_TERNARY
#undef LOOM_MATH_APPROXIMATE_F32_BINARY
#undef LOOM_MATH_APPROXIMATE_F32_UNARY
#undef LOOM_MATH_APPROXIMATE_F32_CONSTANT
#undef LOOM_MATH_APPROXIMATE_F32_INSTRUCTION
#undef LOOM_MATH_APPROXIMATE_F32_PREDICATE
#undef LOOM_MATH_APPROXIMATE_F32_PREDICATE_
#undef LOOM_MATH_APPROXIMATE_F32_OPCODE
#undef LOOM_MATH_APPROXIMATE_F32_CONSTANT_ID

static_assert(IREE_ARRAYSIZE(kApproximateF32Instructions) ==
                  LOOM_MATH_APPROXIMATE_F32_INSTRUCTION_COUNT,
              "approximate F32 instruction table must cover every recipe");

static const loom_math_approximate_f32_program_t
    kApproximateF32Programs[LOOM_MATH_APPROXIMATE_F32_OP_COUNT_] = {
        [LOOM_MATH_APPROXIMATE_F32_OP_EXP2] =
            {
                .instruction_offset =
                    LOOM_MATH_APPROXIMATE_F32_EXP2_INSTRUCTION_OFFSET,
                .instruction_count =
                    LOOM_MATH_APPROXIMATE_F32_EXP2_INSTRUCTION_COUNT,
                .result_value = LOOM_MATH_APPROXIMATE_F32_EXP2_VALUE_RESULT,
            },
        [LOOM_MATH_APPROXIMATE_F32_OP_SIN_TURNS] =
            {
                .instruction_offset =
                    LOOM_MATH_APPROXIMATE_F32_SIN_INSTRUCTION_OFFSET,
                .instruction_count =
                    LOOM_MATH_APPROXIMATE_F32_SIN_INSTRUCTION_COUNT,
                .result_value = LOOM_MATH_APPROXIMATE_F32_SIN_VALUE_RESULT,
            },
        [LOOM_MATH_APPROXIMATE_F32_OP_COS_TURNS] =
            {
                .instruction_offset =
                    LOOM_MATH_APPROXIMATE_F32_COS_INSTRUCTION_OFFSET,
                .instruction_count =
                    LOOM_MATH_APPROXIMATE_F32_COS_INSTRUCTION_COUNT,
                .result_value = LOOM_MATH_APPROXIMATE_F32_COS_VALUE_RESULT,
            },
};

static loom_type_t loom_math_approximate_f32_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  loom_type_t result_type = source_type;
  result_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return result_type;
}

static loom_attribute_t loom_math_approximate_f32_constant_f32(uint32_t bits) {
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return loom_attr_f64((double)value);
}

static loom_attribute_t loom_math_approximate_f32_constant_i32(uint32_t bits) {
  int32_t value = 0;
  memcpy(&value, &bits, sizeof(value));
  return loom_attr_i64((int64_t)value);
}

static loom_attribute_t loom_math_approximate_f32_instruction_attribute(
    const loom_math_approximate_f32_instruction_descriptor_t* descriptor,
    const loom_math_approximate_f32_instruction_t* instruction) {
  switch (descriptor->attribute_mode) {
    case LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_CONSTANT_F32:
      return loom_math_approximate_f32_constant_f32(
          kApproximateF32Constants[instruction->arguments[0]]);
    case LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_CONSTANT_I32:
      return loom_math_approximate_f32_constant_i32(
          kApproximateF32Constants[instruction->arguments[0]]);
    case LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_ENUM:
      return loom_attr_enum(instruction->arguments[2]);
    case LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("attribute-free approximate F32 instruction");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_math_approximate_f32_build(
    loom_math_evaluation_t* evaluation, loom_math_approximate_f32_op_t op,
    const loom_math_evaluation_value_t* input,
    loom_math_evaluation_value_t* out_value) {
  const loom_math_approximate_f32_program_t* program =
      &kApproximateF32Programs[op];
  const bool is_vector = loom_type_is_vector(evaluation->value_type);
  loom_builder_t* builder = &evaluation->rewriter->builder;
  loom_value_id_t values[LOOM_MATH_APPROXIMATE_F32_VALUE_CAPACITY];
  values[0] = loom_math_evaluation_value_id(input);

  for (uint8_t instruction_ordinal = 0;
       instruction_ordinal < program->instruction_count;
       ++instruction_ordinal) {
    const loom_math_approximate_f32_instruction_t* instruction =
        &kApproximateF32Instructions[program->instruction_offset +
                                     instruction_ordinal];
    const loom_math_approximate_f32_instruction_descriptor_t* descriptor =
        &kApproximateF32InstructionDescriptors[instruction->opcode];
    const loom_op_kind_t op_kind =
        is_vector ? descriptor->vector_op_kind : descriptor->scalar_op_kind;
    const uint8_t attribute_count =
        descriptor->attribute_mode == LOOM_MATH_APPROXIMATE_F32_ATTRIBUTE_NONE
            ? 0
            : 1;
    loom_op_t* emitted_op = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, op_kind, descriptor->operand_count, 1, 0, 0, attribute_count,
        evaluation->location, &emitted_op));
    emitted_op->instance_flags =
        evaluation->fastmath_flags & descriptor->fastmath_mask;
    for (uint8_t operand_ordinal = 0;
         operand_ordinal < descriptor->operand_count; ++operand_ordinal) {
      loom_op_operands(emitted_op)[operand_ordinal] =
          values[instruction->arguments[operand_ordinal]];
    }
    if (attribute_count != 0) {
      loom_op_attrs(emitted_op)[0] =
          loom_math_approximate_f32_instruction_attribute(descriptor,
                                                          instruction);
    }
    const loom_type_t result_type = loom_math_approximate_f32_type_with_element(
        evaluation->value_type, descriptor->result_element_type);
    IREE_RETURN_IF_ERROR(loom_builder_define_result(
        builder, result_type, &loom_op_results(emitted_op)[0]));
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, emitted_op));
    values[instruction_ordinal + 1] = loom_op_results(emitted_op)[0];
  }

  *out_value =
      loom_math_evaluation_value_from_id(values[program->result_value]);
  return iree_ok_status();
}
