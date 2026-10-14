// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/materialize_assertions.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"

static const loom_pass_info_t
    loom_sanitizer_materialize_assertions_pass_info_storage = {
        .name = IREE_SVL("sanitizer-materialize-assertions"),
        .description = IREE_SVL(
            "Materialize semantic sanitizer assertions as executable IR."),
        .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_sanitizer_materialize_assertions_pass_info(void) {
  return &loom_sanitizer_materialize_assertions_pass_info_storage;
}

typedef struct loom_sanitizer_predicate_materializer_t {
  // Module owning every referenced and newly materialized value.
  loom_module_t* module;
  // Builder positioned immediately before the sanitizer assertion.
  loom_builder_t* builder;
  // Source assertion location copied to every executable predicate operation.
  loom_location_id_t location;
} loom_sanitizer_predicate_materializer_t;

static bool loom_sanitizer_is_address_scalar(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_sanitizer_integer_comparison_is_unsigned(loom_type_t type) {
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         scalar_type == LOOM_SCALAR_TYPE_I1;
}

static iree_status_t loom_sanitizer_build_numeric_constant(
    loom_sanitizer_predicate_materializer_t* materializer, loom_type_t type,
    int64_t value, loom_value_id_t* out_value) {
  loom_op_t* constant_op = NULL;
  if (loom_sanitizer_is_address_scalar(type)) {
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(materializer->builder, loom_attr_i64(value),
                                  type, materializer->location, &constant_op));
    *out_value = loom_index_constant_result(constant_op);
  } else {
    const loom_scalar_type_t scalar_type = loom_type_element_type(type);
    const loom_attribute_t attribute =
        loom_scalar_type_is_float(scalar_type) ? loom_attr_f64((double)value)
        : scalar_type == LOOM_SCALAR_TYPE_I1   ? loom_attr_bool(value != 0)
                                               : loom_attr_i64(value);
    IREE_RETURN_IF_ERROR(
        loom_scalar_constant_build(materializer->builder, attribute, type,
                                   materializer->location, &constant_op));
    *out_value = loom_scalar_constant_result(constant_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_materialize_predicate_argument(
    loom_sanitizer_predicate_materializer_t* materializer,
    const loom_predicate_t* predicate, uint8_t argument_index, loom_type_t type,
    loom_value_id_t* out_value) {
  if (predicate->arg_tags[argument_index] == LOOM_PRED_ARG_VALUE) {
    *out_value = (loom_value_id_t)predicate->args[argument_index];
    return iree_ok_status();
  }
  IREE_ASSERT(predicate->arg_tags[argument_index] == LOOM_PRED_ARG_CONST);
  return loom_sanitizer_build_numeric_constant(
      materializer, type, predicate->args[argument_index], out_value);
}

static loom_index_cmp_predicate_t loom_sanitizer_index_comparison_predicate(
    loom_predicate_kind_t kind, bool unsigned_comparison) {
  switch (kind) {
    case LOOM_PREDICATE_EQ:
      return LOOM_INDEX_CMP_PREDICATE_EQ;
    case LOOM_PREDICATE_NE:
      return LOOM_INDEX_CMP_PREDICATE_NE;
    case LOOM_PREDICATE_LT:
      return unsigned_comparison ? LOOM_INDEX_CMP_PREDICATE_ULT
                                 : LOOM_INDEX_CMP_PREDICATE_SLT;
    case LOOM_PREDICATE_LE:
      return unsigned_comparison ? LOOM_INDEX_CMP_PREDICATE_ULE
                                 : LOOM_INDEX_CMP_PREDICATE_SLE;
    case LOOM_PREDICATE_GT:
      return unsigned_comparison ? LOOM_INDEX_CMP_PREDICATE_UGT
                                 : LOOM_INDEX_CMP_PREDICATE_SGT;
    case LOOM_PREDICATE_GE:
      return unsigned_comparison ? LOOM_INDEX_CMP_PREDICATE_UGE
                                 : LOOM_INDEX_CMP_PREDICATE_SGE;
    default:
      IREE_ASSERT_UNREACHABLE("expected an integer relation predicate");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static loom_scalar_cmpi_predicate_t loom_sanitizer_scalar_comparison_predicate(
    loom_predicate_kind_t kind, bool unsigned_comparison) {
  switch (kind) {
    case LOOM_PREDICATE_EQ:
      return LOOM_SCALAR_CMPI_PREDICATE_EQ;
    case LOOM_PREDICATE_NE:
      return LOOM_SCALAR_CMPI_PREDICATE_NE;
    case LOOM_PREDICATE_LT:
      return unsigned_comparison ? LOOM_SCALAR_CMPI_PREDICATE_ULT
                                 : LOOM_SCALAR_CMPI_PREDICATE_SLT;
    case LOOM_PREDICATE_LE:
      return unsigned_comparison ? LOOM_SCALAR_CMPI_PREDICATE_ULE
                                 : LOOM_SCALAR_CMPI_PREDICATE_SLE;
    case LOOM_PREDICATE_GT:
      return unsigned_comparison ? LOOM_SCALAR_CMPI_PREDICATE_UGT
                                 : LOOM_SCALAR_CMPI_PREDICATE_SGT;
    case LOOM_PREDICATE_GE:
      return unsigned_comparison ? LOOM_SCALAR_CMPI_PREDICATE_UGE
                                 : LOOM_SCALAR_CMPI_PREDICATE_SGE;
    default:
      IREE_ASSERT_UNREACHABLE("expected an integer relation predicate");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_sanitizer_build_integer_comparison(
    loom_sanitizer_predicate_materializer_t* materializer, loom_type_t type,
    loom_predicate_kind_t kind, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t* out_condition) {
  bool unsigned_comparison =
      loom_sanitizer_integer_comparison_is_unsigned(type);
  switch (kind) {
    case LOOM_PREDICATE_ULT:
      kind = LOOM_PREDICATE_LT;
      unsigned_comparison = true;
      break;
    case LOOM_PREDICATE_ULE:
      kind = LOOM_PREDICATE_LE;
      unsigned_comparison = true;
      break;
    case LOOM_PREDICATE_UGT:
      kind = LOOM_PREDICATE_GT;
      unsigned_comparison = true;
      break;
    case LOOM_PREDICATE_UGE:
      kind = LOOM_PREDICATE_GE;
      unsigned_comparison = true;
      break;
    default:
      break;
  }
  loom_op_t* comparison_op = NULL;
  if (loom_sanitizer_is_address_scalar(type)) {
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        materializer->builder,
        loom_sanitizer_index_comparison_predicate(kind, unsigned_comparison),
        lhs, rhs, materializer->location, &comparison_op));
    *out_condition = loom_index_cmp_result(comparison_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        materializer->builder,
        loom_sanitizer_scalar_comparison_predicate(kind, unsigned_comparison),
        lhs, rhs, materializer->location, &comparison_op));
    *out_condition = loom_scalar_cmpi_result(comparison_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_numeric_comparison(
    loom_sanitizer_predicate_materializer_t* materializer, loom_type_t type,
    loom_predicate_kind_t kind, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t* out_condition) {
  if (!loom_scalar_type_is_float(loom_type_element_type(type))) {
    return loom_sanitizer_build_integer_comparison(materializer, type, kind,
                                                   lhs, rhs, out_condition);
  }
  IREE_ASSERT(kind == LOOM_PREDICATE_EQ || kind == LOOM_PREDICATE_NE);
  loom_op_t* comparison_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(
      materializer->builder, /*instance_flags=*/0,
      kind == LOOM_PREDICATE_EQ ? LOOM_SCALAR_CMPF_PREDICATE_OEQ
                                : LOOM_SCALAR_CMPF_PREDICATE_UNE,
      lhs, rhs, materializer->location, &comparison_op));
  *out_condition = loom_scalar_cmpf_result(comparison_op);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_conjunction(
    loom_sanitizer_predicate_materializer_t* materializer, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_condition) {
  loom_op_t* conjunction_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      materializer->builder, lhs, rhs, loom_type_scalar(LOOM_SCALAR_TYPE_I1),
      materializer->location, &conjunction_op));
  *out_condition = loom_scalar_andi_result(conjunction_op);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_negation(
    loom_sanitizer_predicate_materializer_t* materializer,
    loom_value_id_t value, loom_value_id_t* out_condition) {
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_value_id_t true_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_numeric_constant(
      materializer, i1_type, 1, &true_value));
  loom_op_t* negation_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_xori_build(materializer->builder, value, true_value, i1_type,
                             materializer->location, &negation_op));
  *out_condition = loom_scalar_xori_result(negation_op);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_materialize_numeric_relation(
    loom_sanitizer_predicate_materializer_t* materializer,
    const loom_predicate_t* predicate, loom_type_t type,
    loom_predicate_kind_t relation_kind, uint8_t lhs_argument_index,
    uint8_t rhs_argument_index, loom_value_id_t* out_condition) {
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate_argument(
      materializer, predicate, lhs_argument_index, type, &lhs));
  IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate_argument(
      materializer, predicate, rhs_argument_index, type, &rhs));
  return loom_sanitizer_build_numeric_comparison(
      materializer, type, relation_kind, lhs, rhs, out_condition);
}

static iree_status_t loom_sanitizer_materialize_multiple_predicate(
    loom_sanitizer_predicate_materializer_t* materializer,
    const loom_predicate_t* predicate, loom_type_t type,
    loom_value_id_t* out_condition) {
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    return loom_sanitizer_build_numeric_constant(materializer, type, 1,
                                                 out_condition);
  }

  loom_value_id_t subject = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
  loom_type_t arithmetic_type = type;
  IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate_argument(
      materializer, predicate, 0, type, &subject));
  if (loom_sanitizer_is_address_scalar(type)) {
    arithmetic_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    loom_op_t* cast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        materializer->builder, subject, type, arithmetic_type,
        materializer->location, &cast_op));
    subject = loom_index_cast_result(cast_op);
    IREE_RETURN_IF_ERROR(loom_sanitizer_build_numeric_constant(
        materializer, arithmetic_type, predicate->args[1], &divisor));
  } else {
    IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate_argument(
        materializer, predicate, 1, type, &divisor));
  }

  loom_op_t* remainder_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_remsi_build(
      materializer->builder, subject, divisor, arithmetic_type,
      materializer->location, &remainder_op));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_numeric_constant(
      materializer, arithmetic_type, 0, &zero));
  return loom_sanitizer_build_integer_comparison(
      materializer, arithmetic_type, LOOM_PREDICATE_EQ,
      loom_scalar_remsi_result(remainder_op), zero, out_condition);
}

static iree_status_t loom_sanitizer_materialize_power_of_two_predicate(
    loom_sanitizer_predicate_materializer_t* materializer,
    const loom_predicate_t* predicate, loom_type_t type,
    loom_value_id_t* out_condition) {
  loom_value_id_t subject = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate_argument(
      materializer, predicate, 0, type, &subject));
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    *out_condition = subject;
    return iree_ok_status();
  }

  loom_type_t arithmetic_type = type;
  if (scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    arithmetic_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    loom_op_t* cast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        materializer->builder, subject, type, arithmetic_type,
        materializer->location, &cast_op));
    subject = loom_index_cast_result(cast_op);
  }

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_numeric_constant(
      materializer, arithmetic_type, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_numeric_constant(
      materializer, arithmetic_type, 1, &one));

  loom_value_id_t positive = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_integer_comparison(
      materializer, arithmetic_type, LOOM_PREDICATE_GT, subject, zero,
      &positive));

  loom_op_t* population_count_op = NULL;
  loom_value_id_t population_count = LOOM_VALUE_ID_INVALID;
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX) {
    IREE_RETURN_IF_ERROR(loom_index_ctpopi_build(materializer->builder, subject,
                                                 materializer->location,
                                                 &population_count_op));
    population_count = loom_index_ctpopi_result(population_count_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_ctpopi_build(
        materializer->builder, subject, arithmetic_type, materializer->location,
        &population_count_op));
    population_count = loom_scalar_ctpopi_result(population_count_op);
  }

  loom_value_id_t single_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_integer_comparison(
      materializer, arithmetic_type, LOOM_PREDICATE_EQ, population_count, one,
      &single_bit));
  return loom_sanitizer_build_conjunction(materializer, positive, single_bit,
                                          out_condition);
}

static iree_status_t loom_sanitizer_materialize_float_predicate(
    loom_sanitizer_predicate_materializer_t* materializer,
    const loom_predicate_t* predicate, loom_value_id_t* out_condition) {
  const loom_value_id_t subject = (loom_value_id_t)predicate->args[0];
  loom_op_t* classifier_op = NULL;
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_NOT_NAN: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_isnanf_build(materializer->builder, subject,
                                   materializer->location, &classifier_op));
      return loom_sanitizer_build_negation(
          materializer, loom_scalar_isnanf_result(classifier_op),
          out_condition);
    }
    case LOOM_PREDICATE_NOT_INF: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_isinff_build(materializer->builder, subject,
                                   materializer->location, &classifier_op));
      return loom_sanitizer_build_negation(
          materializer, loom_scalar_isinff_result(classifier_op),
          out_condition);
    }
    case LOOM_PREDICATE_FINITE: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_isfinitef_build(materializer->builder, subject,
                                      materializer->location, &classifier_op));
      *out_condition = loom_scalar_isfinitef_result(classifier_op);
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE("expected a floating-point class predicate");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_sanitizer_materialize_predicate(
    loom_sanitizer_predicate_materializer_t* materializer,
    const loom_predicate_t* predicate, loom_value_id_t* out_condition) {
  const loom_value_id_t subject = (loom_value_id_t)predicate->args[0];
  const loom_type_t type =
      loom_module_value_type(materializer->module, subject);
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_ULT:
    case LOOM_PREDICATE_ULE:
    case LOOM_PREDICATE_UGT:
    case LOOM_PREDICATE_UGE:
      return loom_sanitizer_materialize_numeric_relation(
          materializer, predicate, type, (loom_predicate_kind_t)predicate->kind,
          0, 1, out_condition);
    case LOOM_PREDICATE_MULTIPLE_OF:
      return loom_sanitizer_materialize_multiple_predicate(
          materializer, predicate, type, out_condition);
    case LOOM_PREDICATE_POWER_OF_TWO:
      return loom_sanitizer_materialize_power_of_two_predicate(
          materializer, predicate, type, out_condition);
    case LOOM_PREDICATE_RANGE: {
      loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
      loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_numeric_relation(
          materializer, predicate, type, LOOM_PREDICATE_GE, 0, 1,
          &lower_bound));
      IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_numeric_relation(
          materializer, predicate, type, LOOM_PREDICATE_LE, 0, 2,
          &upper_bound));
      return loom_sanitizer_build_conjunction(materializer, lower_bound,
                                              upper_bound, out_condition);
    }
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return loom_sanitizer_materialize_float_predicate(materializer, predicate,
                                                        out_condition);
    default:
      IREE_ASSERT_UNREACHABLE("verified sanitizer predicate kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_sanitizer_materialize_predicate_list(
    loom_sanitizer_predicate_materializer_t* materializer,
    loom_attribute_t predicates, loom_value_id_t* out_condition) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  for (uint16_t i = 0; i < predicates.count; ++i) {
    loom_value_id_t predicate_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate(
        materializer, &predicates.predicate_list[i], &predicate_condition));
    if (condition == LOOM_VALUE_ID_INVALID) {
      condition = predicate_condition;
    } else {
      IREE_RETURN_IF_ERROR(loom_sanitizer_build_conjunction(
          materializer, condition, predicate_condition, &condition));
    }
  }
  *out_condition = condition;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_kernel_assert(
    loom_sanitizer_predicate_materializer_t* materializer,
    loom_value_id_t condition) {
  loom_op_t* assert_op = NULL;
  return loom_kernel_assert_build(materializer->builder, /*build_flags=*/0,
                                  condition, LOOM_STRING_ID_INVALID,
                                  materializer->location, &assert_op);
}

static iree_status_t loom_sanitizer_emit_materialization_failure(
    loom_pass_t* pass, const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t reason) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(pass->info->name),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_LOWERING_064,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

typedef enum loom_sanitizer_assume_group_kind_e {
  LOOM_SANITIZER_ASSUME_GROUP_ADDRESS = 0,
  LOOM_SANITIZER_ASSUME_GROUP_PAYLOAD = 1,
} loom_sanitizer_assume_group_kind_t;

typedef struct loom_sanitizer_assume_group_t {
  // Source assertion operands represented by this assume operation.
  loom_value_id_t* values;
  // Result types corresponding positionally to values.
  loom_type_t* result_types;
  // Source assertion result ordinal corresponding to each group result.
  uint16_t* result_indices;
  // Number of populated value, result type, and result-index entries.
  uint16_t value_count;
  // Predicates whose subject belongs to this scalar domain.
  loom_predicate_t* predicates;
  // Number of populated predicates.
  uint16_t predicate_count;
} loom_sanitizer_assume_group_t;

static iree_status_t loom_sanitizer_allocate_assume_group_storage(
    iree_arena_allocator_t* arena, uint16_t value_capacity,
    uint16_t predicate_capacity, loom_sanitizer_assume_group_t* group) {
  if (value_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_capacity, sizeof(*group->values), (void**)&group->values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_capacity, sizeof(*group->result_types),
        (void**)&group->result_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_capacity, sizeof(*group->result_indices),
        (void**)&group->result_indices));
  }
  if (predicate_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, predicate_capacity,
                                                   sizeof(*group->predicates),
                                                   (void**)&group->predicates));
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_assume_group(
    loom_rewriter_t* rewriter, loom_sanitizer_assume_group_kind_t kind,
    loom_location_id_t location, const loom_sanitizer_assume_group_t* group,
    loom_value_id_t* replacements) {
  if (group->predicate_count == 0) {
    return iree_ok_status();
  }
  loom_op_t* assume_op = NULL;
  loom_value_slice_t results = {0};
  if (kind == LOOM_SANITIZER_ASSUME_GROUP_ADDRESS) {
    IREE_RETURN_IF_ERROR(loom_index_assume_build(
        &rewriter->builder, group->values, group->value_count,
        group->predicates, group->predicate_count, group->result_types,
        group->value_count, location, &assume_op));
    results = loom_index_assume_results(assume_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
        &rewriter->builder, group->values, group->value_count,
        group->predicates, group->predicate_count, group->result_types,
        group->value_count, location, &assume_op));
    results = loom_scalar_assume_results(assume_op);
  }
  for (uint16_t i = 0; i < group->value_count; ++i) {
    replacements[group->result_indices[i]] = results.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_materialize_value_assertion(
    loom_module_t* module, loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_value_slice_t values = loom_sanitizer_assert_value_values(op);
  const loom_value_slice_t results = loom_sanitizer_assert_value_results(op);
  const loom_attribute_t predicates =
      loom_sanitizer_assert_value_predicates(op);
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  if (results.count == 0) {
    return loom_rewriter_erase(rewriter, op);
  }

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, results.count,
                                                 sizeof(*replacements),
                                                 (void**)&replacements));
  for (uint16_t i = 0; i < values.count; ++i) {
    replacements[i] = values.values[i];
  }
  if (predicates.count == 0) {
    return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                    results.count);
  }

  loom_sanitizer_predicate_materializer_t materializer = {
      .module = module,
      .builder = &rewriter->builder,
      .location = op->location,
  };
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate_list(
      &materializer, predicates, &condition));
  IREE_RETURN_IF_ERROR(
      loom_sanitizer_build_kernel_assert(&materializer, condition));

  loom_sanitizer_assume_group_t address_group = {0};
  loom_sanitizer_assume_group_t payload_group = {0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_allocate_assume_group_storage(
      rewriter->arena, values.count, predicates.count, &address_group));
  IREE_RETURN_IF_ERROR(loom_sanitizer_allocate_assume_group_storage(
      rewriter->arena, values.count, predicates.count, &payload_group));
  for (uint16_t i = 0; i < values.count; ++i) {
    const loom_type_t type = loom_module_value_type(module, values.values[i]);
    loom_sanitizer_assume_group_t* group = NULL;
    if (loom_sanitizer_is_address_scalar(type)) {
      group = &address_group;
    } else if (loom_type_is_scalar(type)) {
      group = &payload_group;
    } else {
      continue;
    }
    const uint16_t group_index = group->value_count++;
    group->values[group_index] = values.values[i];
    group->result_types[group_index] =
        loom_module_value_type(module, results.values[i]);
    group->result_indices[group_index] = i;
  }
  for (uint16_t i = 0; i < predicates.count; ++i) {
    const loom_predicate_t predicate = predicates.predicate_list[i];
    const loom_type_t subject_type =
        loom_module_value_type(module, (loom_value_id_t)predicate.args[0]);
    loom_sanitizer_assume_group_t* group =
        loom_sanitizer_is_address_scalar(subject_type) ? &address_group
                                                       : &payload_group;
    group->predicates[group->predicate_count++] = predicate;
  }

  IREE_RETURN_IF_ERROR(loom_sanitizer_build_assume_group(
      rewriter, LOOM_SANITIZER_ASSUME_GROUP_ADDRESS, op->location,
      &address_group, replacements));
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_assume_group(
      rewriter, LOOM_SANITIZER_ASSUME_GROUP_PAYLOAD, op->location,
      &payload_group, replacements));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, results.count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  results.count);
}

static iree_status_t loom_sanitizer_materialize_operation_assertion(
    loom_module_t* module, loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_attribute_t predicates = loom_sanitizer_assert_op_predicates(op);
  if (predicates.count == 0) {
    return loom_rewriter_erase(rewriter, op);
  }
  loom_builder_set_before(&rewriter->builder, op);
  loom_sanitizer_predicate_materializer_t materializer = {
      .module = module,
      .builder = &rewriter->builder,
      .location = op->location,
  };
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_materialize_predicate_list(
      &materializer, predicates, &condition));
  IREE_RETURN_IF_ERROR(
      loom_sanitizer_build_kernel_assert(&materializer, condition));
  return loom_rewriter_erase(rewriter, op);
}

static iree_status_t loom_sanitizer_materialize_layout_assertion(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  const loom_value_id_t source = loom_sanitizer_assert_layout_view(op);
  const loom_value_id_t result = loom_sanitizer_assert_layout_result(op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_type_encoding_equals(source_type, result_type)) {
    return loom_sanitizer_emit_materialization_failure(
        pass, module, op,
        IREE_SV("the result encoding is not proven identical to the source "
                "encoding"));
  }
  const uint8_t source_alignment = loom_type_view_alignment(source_type);
  const uint8_t result_alignment = loom_type_view_alignment(result_type);
  if (source_alignment % result_alignment != 0) {
    return loom_sanitizer_emit_materialization_failure(
        pass, module, op,
        IREE_SV("the result access alignment is stronger than the source "
                "contract"));
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_sanitizer_predicate_materializer_t materializer = {
      .module = module,
      .builder = &rewriter->builder,
      .location = op->location,
  };
  loom_value_id_t shape_condition = LOOM_VALUE_ID_INVALID;
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const uint8_t rank = loom_type_rank(source_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    const bool source_dynamic = loom_type_dim_is_dynamic_at(source_type, axis);
    const bool result_dynamic = loom_type_dim_is_dynamic_at(result_type, axis);
    if (!source_dynamic && !result_dynamic) {
      continue;
    }

    loom_value_id_t source_dimension = LOOM_VALUE_ID_INVALID;
    loom_value_id_t result_dimension = LOOM_VALUE_ID_INVALID;
    if (source_dynamic) {
      source_dimension = loom_type_dim_value_id_at(source_type, axis);
    } else {
      IREE_RETURN_IF_ERROR(loom_sanitizer_build_numeric_constant(
          &materializer, index_type,
          loom_type_dim_static_size_at(source_type, axis), &source_dimension));
    }
    if (result_dynamic) {
      result_dimension = loom_type_dim_value_id_at(result_type, axis);
    } else {
      IREE_RETURN_IF_ERROR(loom_sanitizer_build_numeric_constant(
          &materializer, index_type,
          loom_type_dim_static_size_at(result_type, axis), &result_dimension));
    }
    if (source_dimension == result_dimension) {
      continue;
    }

    loom_value_id_t axis_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_sanitizer_build_integer_comparison(
        &materializer, index_type, LOOM_PREDICATE_EQ, source_dimension,
        result_dimension, &axis_condition));
    if (shape_condition == LOOM_VALUE_ID_INVALID) {
      shape_condition = axis_condition;
    } else {
      IREE_RETURN_IF_ERROR(loom_sanitizer_build_conjunction(
          &materializer, shape_condition, axis_condition, &shape_condition));
    }
  }
  if (shape_condition != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_sanitizer_build_kernel_assert(&materializer, shape_condition));
  }

  loom_op_t* refine_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_refine_build(
      &rewriter->builder, source, result_type, op->location, &refine_op));
  const loom_value_id_t replacement = loom_view_refine_result(refine_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static bool loom_sanitizer_is_semantic_assertion(const loom_op_t* op) {
  return loom_sanitizer_assert_value_isa(op) ||
         loom_sanitizer_assert_op_isa(op) ||
         loom_sanitizer_assert_layout_isa(op);
}

iree_status_t loom_sanitizer_materialize_assertions_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_status_t status = loom_rewriter_seed_function(&rewriter, function);
  while (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass)) {
    loom_op_t* op = loom_rewriter_pop(&rewriter);
    if (!op) {
      break;
    }
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
        !loom_sanitizer_is_semantic_assertion(op)) {
      continue;
    }
    if (!loom_func_like_is_kernel(function)) {
      status = loom_sanitizer_emit_materialization_failure(
          pass, module, op,
          IREE_SV("semantic assertions require a dispatchable kernel"));
      continue;
    }
    if (loom_sanitizer_assert_value_isa(op)) {
      status =
          loom_sanitizer_materialize_value_assertion(module, &rewriter, op);
    } else if (loom_sanitizer_assert_op_isa(op)) {
      status =
          loom_sanitizer_materialize_operation_assertion(module, &rewriter, op);
    } else {
      status = loom_sanitizer_materialize_layout_assertion(pass, module,
                                                           &rewriter, op);
    }
  }
  if (iree_status_is_ok(status) &&
      (rewriter.created_op_count != 0 || rewriter.erased_op_count != 0 ||
       iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED))) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
