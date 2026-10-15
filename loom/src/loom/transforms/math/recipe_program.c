// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/recipe_program.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

enum loom_math_recipe_attribute_mode_e {
  LOOM_MATH_RECIPE_ATTR_NONE = 0,
  LOOM_MATH_RECIPE_ATTR_CONSTANT_F32 = 1,
  LOOM_MATH_RECIPE_ATTR_CONSTANT_I32 = 2,
  LOOM_MATH_RECIPE_ATTR_ENUM = 3,
};
typedef uint8_t loom_math_recipe_attribute_mode_t;

typedef struct loom_math_recipe_instruction_descriptor_t {
  // Operation emitted when evaluating a scalar source value.
  loom_op_kind_t scalar_op_kind;
  // Operation emitted when evaluating a vector source value.
  loom_op_kind_t vector_op_kind;
  // Number of leading instruction arguments that reference SSA values.
  uint8_t operand_count;
  // Element type of the instruction's single result.
  loom_scalar_type_t result_element_type;
  // Interpretation of the instruction's non-operand argument, when present.
  loom_math_recipe_attribute_mode_t attribute_mode;
} loom_math_recipe_instruction_descriptor_t;
static_assert(sizeof(loom_math_recipe_instruction_descriptor_t) == 8,
              "math recipe descriptors must remain compact");

typedef struct loom_math_recipe_instruction_t {
  // Index of the shared scalar/vector operation descriptor.
  uint8_t descriptor_index;
  // Local SSA value indices or an attribute payload index.
  uint8_t arguments[3];
} loom_math_recipe_instruction_t;
static_assert(sizeof(loom_math_recipe_instruction_t) == 4,
              "math recipe instructions must remain compact");

typedef struct loom_math_recipe_program_t {
  // First instruction in the shared instruction table.
  uint16_t instruction_offset;
  // Number of instructions in this program.
  uint8_t instruction_count;
  // Local SSA value index returned by this program.
  uint8_t result_value;
} loom_math_recipe_program_t;
static_assert(sizeof(loom_math_recipe_program_t) == 4,
              "math recipe programs must remain compact");

#include "loom/transforms/math/approximate_f32_recipe_tables.inl"

static loom_type_t loom_math_recipe_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  loom_type_t result_type = source_type;
  result_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return result_type;
}

static loom_attribute_t loom_math_recipe_constant_f32(uint32_t bits) {
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return loom_attr_f64((double)value);
}

static loom_attribute_t loom_math_recipe_constant_i32(uint32_t bits) {
  int32_t value = 0;
  memcpy(&value, &bits, sizeof(value));
  return loom_attr_i64((int64_t)value);
}

static loom_attribute_t loom_math_recipe_instruction_attribute(
    const loom_math_recipe_instruction_descriptor_t* descriptor,
    const loom_math_recipe_instruction_t* instruction) {
  switch (descriptor->attribute_mode) {
    case LOOM_MATH_RECIPE_ATTR_CONSTANT_F32:
      return loom_math_recipe_constant_f32(
          kMathRecipeConstants[instruction->arguments[0]]);
    case LOOM_MATH_RECIPE_ATTR_CONSTANT_I32:
      return loom_math_recipe_constant_i32(
          kMathRecipeConstants[instruction->arguments[0]]);
    case LOOM_MATH_RECIPE_ATTR_ENUM:
      return loom_attr_enum(instruction->arguments[2]);
    case LOOM_MATH_RECIPE_ATTR_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("attribute-free recipe instruction");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_math_recipe_program_evaluate(
    loom_math_evaluation_t* evaluation,
    loom_math_recipe_program_id_t program_id,
    const loom_math_evaluation_value_t* input,
    loom_math_evaluation_value_t* out_value) {
  const loom_math_recipe_program_t* program = &kMathRecipePrograms[program_id];
  const bool is_vector = loom_type_is_vector(evaluation->value_type);
  loom_builder_t* builder = &evaluation->rewriter->builder;
  loom_value_id_t values[LOOM_MATH_RECIPE_PROGRAM_VALUE_CAPACITY];
  values[0] = loom_math_evaluation_value_id(input);

  for (uint8_t instruction_ordinal = 0;
       instruction_ordinal < program->instruction_count;
       ++instruction_ordinal) {
    const loom_math_recipe_instruction_t* instruction =
        &kMathRecipeInstructions[program->instruction_offset +
                                 instruction_ordinal];
    const loom_math_recipe_instruction_descriptor_t* descriptor =
        &kMathRecipeInstructionDescriptors[instruction->descriptor_index];
    const loom_op_kind_t op_kind =
        is_vector ? descriptor->vector_op_kind : descriptor->scalar_op_kind;
    const uint8_t attribute_count =
        descriptor->attribute_mode == LOOM_MATH_RECIPE_ATTR_NONE ? 0 : 1;
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, op_kind, descriptor->operand_count, 1, 0, 0, attribute_count,
        evaluation->location, &op));
    for (uint8_t operand_ordinal = 0;
         operand_ordinal < descriptor->operand_count; ++operand_ordinal) {
      loom_op_operands(op)[operand_ordinal] =
          values[instruction->arguments[operand_ordinal]];
    }
    if (attribute_count != 0) {
      loom_op_attrs(op)[0] =
          loom_math_recipe_instruction_attribute(descriptor, instruction);
    }
    const loom_type_t result_type = loom_math_recipe_type_with_element(
        evaluation->value_type, descriptor->result_element_type);
    IREE_RETURN_IF_ERROR(loom_builder_define_result(builder, result_type,
                                                    &loom_op_results(op)[0]));
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
    values[instruction_ordinal + 1] = loom_op_results(op)[0];
  }

  *out_value =
      loom_math_evaluation_value_from_id(values[program->result_value]);
  return iree_ok_status();
}
