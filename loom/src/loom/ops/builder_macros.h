// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builder macros for common op patterns.
//
// Generated per-dialect builders.c files use these macros to stamp out
// builder function implementations for ops that match standard patterns
// (binary, unary, cast, comparison). Complex ops get explicit
// implementations instead.
//
// Each macro defines a function with the signature:
//   iree_status_t <name>(loom_builder_t* builder, ...,
//                        loom_location_id_t location,
//                        loom_op_t** out_op);
//
// Usage in generated builders.c:
//
//   LOOM_DEFINE_BINARY_OP_BUILDER(loom_test_addi_build, LOOM_OP_TEST_ADDI)
//   LOOM_DEFINE_UNARY_OP_BUILDER(loom_test_neg_build, LOOM_OP_TEST_NEG)

#ifndef LOOM_OPS_BUILDER_MACROS_H_
#define LOOM_OPS_BUILDER_MACROS_H_

#include "loom/ops/op_defs.h"

// Defines a builder for a binary op: 2 operands, 1 result, no attrs/regions.
// Pattern: %result = op %lhs, %rhs : type
#define LOOM_DEFINE_BINARY_OP_BUILDER(func_name, kind_enum)                    \
  iree_status_t func_name(loom_builder_t* builder, loom_value_id_t lhs,        \
                          loom_value_id_t rhs, loom_type_t result_type,        \
                          loom_location_id_t location, loom_op_t** out_op) {   \
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, (kind_enum), 2, 1,  \
                                                  0, 0, 0, location, out_op)); \
    loom_op_operands(*out_op)[0] = lhs;                                        \
    loom_op_operands(*out_op)[1] = rhs;                                        \
    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;                        \
    IREE_RETURN_IF_ERROR(                                                      \
        loom_builder_define_value(builder, result_type, &_result_id));         \
    loom_op_results(*out_op)[0] = _result_id;                                  \
    return loom_builder_finalize_op(builder, *out_op);                         \
  }

// Defines a builder for a unary op: 1 operand, 1 result, no attrs/regions.
// Pattern: %result = op %input : type
#define LOOM_DEFINE_UNARY_OP_BUILDER(func_name, kind_enum)                     \
  iree_status_t func_name(loom_builder_t* builder, loom_value_id_t input,      \
                          loom_type_t result_type,                             \
                          loom_location_id_t location, loom_op_t** out_op) {   \
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, (kind_enum), 1, 1,  \
                                                  0, 0, 0, location, out_op)); \
    loom_op_operands(*out_op)[0] = input;                                      \
    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;                        \
    IREE_RETURN_IF_ERROR(                                                      \
        loom_builder_define_value(builder, result_type, &_result_id));         \
    loom_op_results(*out_op)[0] = _result_id;                                  \
    return loom_builder_finalize_op(builder, *out_op);                         \
  }

// Defines a builder for a cast op: 1 operand, 1 result, no attrs/regions.
// Pattern: %result = op %input : input_type to result_type
#define LOOM_DEFINE_CAST_OP_BUILDER(func_name, kind_enum)                      \
  iree_status_t func_name(loom_builder_t* builder, loom_value_id_t input,      \
                          loom_type_t input_type, loom_type_t result_type,     \
                          loom_location_id_t location, loom_op_t** out_op) {   \
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, (kind_enum), 1, 1,  \
                                                  0, 0, 0, location, out_op)); \
    loom_op_operands(*out_op)[0] = input;                                      \
    (void)input_type;                                                          \
    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;                        \
    IREE_RETURN_IF_ERROR(                                                      \
        loom_builder_define_value(builder, result_type, &_result_id));         \
    loom_op_results(*out_op)[0] = _result_id;                                  \
    return loom_builder_finalize_op(builder, *out_op);                         \
  }

// Defines a builder for a comparison op: 2 operands, 1 result, 1 enum attr.
// Pattern: %result = op predicate, %lhs, %rhs : operand_type
// The result_type is separate from operand_type (comparisons typically
// return i1/bool, not the operand type).
#define LOOM_DEFINE_COMPARISON_OP_BUILDER(func_name, kind_enum)                \
  iree_status_t func_name(loom_builder_t* builder, uint8_t predicate,          \
                          loom_value_id_t lhs, loom_value_id_t rhs,            \
                          loom_type_t operand_type, loom_type_t result_type,   \
                          loom_location_id_t location, loom_op_t** out_op) {   \
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, (kind_enum), 2, 1,  \
                                                  0, 0, 1, location, out_op)); \
    loom_op_operands(*out_op)[0] = lhs;                                        \
    loom_op_operands(*out_op)[1] = rhs;                                        \
    loom_op_attrs(*out_op)[0] = loom_attr_enum(predicate);                     \
    (void)operand_type;                                                        \
    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;                        \
    IREE_RETURN_IF_ERROR(                                                      \
        loom_builder_define_value(builder, result_type, &_result_id));         \
    loom_op_results(*out_op)[0] = _result_id;                                  \
    return loom_builder_finalize_op(builder, *out_op);                         \
  }

// Defines a builder for a binary op with per-instance flags.
// Pattern: %result = op<flags> %lhs, %rhs : type
#define LOOM_DEFINE_BINARY_OP_WITH_FLAGS_BUILDER(func_name, kind_enum)         \
  iree_status_t func_name(loom_builder_t* builder, uint8_t instance_flags,     \
                          loom_value_id_t lhs, loom_value_id_t rhs,            \
                          loom_type_t result_type,                             \
                          loom_location_id_t location, loom_op_t** out_op) {   \
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, (kind_enum), 2, 1,  \
                                                  0, 0, 0, location, out_op)); \
    (*out_op)->instance_flags = instance_flags;                                \
    loom_op_operands(*out_op)[0] = lhs;                                        \
    loom_op_operands(*out_op)[1] = rhs;                                        \
    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;                        \
    IREE_RETURN_IF_ERROR(                                                      \
        loom_builder_define_value(builder, result_type, &_result_id));         \
    loom_op_results(*out_op)[0] = _result_id;                                  \
    return loom_builder_finalize_op(builder, *out_op);                         \
  }

// Defines a builder for a unary op with per-instance flags.
// Pattern: %result = op<flags> %input : type
#define LOOM_DEFINE_UNARY_OP_WITH_FLAGS_BUILDER(func_name, kind_enum)          \
  iree_status_t func_name(loom_builder_t* builder, uint8_t instance_flags,     \
                          loom_value_id_t input, loom_type_t result_type,      \
                          loom_location_id_t location, loom_op_t** out_op) {   \
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, (kind_enum), 1, 1,  \
                                                  0, 0, 0, location, out_op)); \
    (*out_op)->instance_flags = instance_flags;                                \
    loom_op_operands(*out_op)[0] = input;                                      \
    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;                        \
    IREE_RETURN_IF_ERROR(                                                      \
        loom_builder_define_value(builder, result_type, &_result_id));         \
    loom_op_results(*out_op)[0] = _result_id;                                  \
    return loom_builder_finalize_op(builder, *out_op);                         \
  }

// Defines a builder for a comparison op with per-instance flags.
// Pattern: %result = op<flags> predicate, %lhs, %rhs : operand_type
#define LOOM_DEFINE_COMPARISON_OP_WITH_FLAGS_BUILDER(func_name, kind_enum)     \
  iree_status_t func_name(loom_builder_t* builder, uint8_t instance_flags,     \
                          uint8_t predicate, loom_value_id_t lhs,              \
                          loom_value_id_t rhs, loom_type_t operand_type,       \
                          loom_type_t result_type,                             \
                          loom_location_id_t location, loom_op_t** out_op) {   \
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, (kind_enum), 2, 1,  \
                                                  0, 0, 1, location, out_op)); \
    (*out_op)->instance_flags = instance_flags;                                \
    loom_op_operands(*out_op)[0] = lhs;                                        \
    loom_op_operands(*out_op)[1] = rhs;                                        \
    loom_op_attrs(*out_op)[0] = loom_attr_enum(predicate);                     \
    (void)operand_type;                                                        \
    loom_value_id_t _result_id = LOOM_VALUE_ID_INVALID;                        \
    IREE_RETURN_IF_ERROR(                                                      \
        loom_builder_define_value(builder, result_type, &_result_id));         \
    loom_op_results(*out_op)[0] = _result_id;                                  \
    return loom_builder_finalize_op(builder, *out_op);                         \
  }

#endif  // LOOM_OPS_BUILDER_MACROS_H_
