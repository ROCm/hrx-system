// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_value_constraints.h"

#include <limits>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/attribute.h"

namespace {

TEST(SymbolValueConstraintsTest, ChecksExactIntegerValue) {
  const loom_value_id_t contract_value = 7;
  loom_predicate_t predicates[] = {
      {
          .kind = LOOM_PREDICATE_GE,
          .arg_count = 2,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
          .reserved = {},
          .args = {contract_value, 32},
      },
      {
          .kind = LOOM_PREDICATE_MULTIPLE_OF,
          .arg_count = 2,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
          .reserved = {},
          .args = {contract_value, 16},
      },
      {
          .kind = LOOM_PREDICATE_POWER_OF_TWO,
          .arg_count = 1,
          .arg_tags = {LOOM_PRED_ARG_VALUE},
          .reserved = {},
          .args = {contract_value},
      },
  };

  IREE_ASSERT_OK(loom_symbol_value_constraints_check_exact(
      IREE_SV("tile_size"), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      contract_value, loom_attr_i64(64),
      loom_attr_predicate_list(predicates, IREE_ARRAYSIZE(predicates))));
}

TEST(SymbolValueConstraintsTest, RejectsViolatedIntegerPredicate) {
  const loom_value_id_t contract_value = 3;
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {contract_value, 16, 63},
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_value_constraints_check_exact(
          IREE_SV("tile_size"), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
          contract_value, loom_attr_i64(64),
          loom_attr_predicate_list(&predicate, 1)));
}

TEST(SymbolValueConstraintsTest, ChecksUnsignedCarrierOrder) {
  const loom_value_id_t contract_value = 3;
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_ULE,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {contract_value, -1},
  };

  IREE_ASSERT_OK(loom_symbol_value_constraints_check_exact(
      IREE_SV("tile_size"), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      contract_value, loom_attr_i64(INT32_MIN),
      loom_attr_predicate_list(&predicate, 1)));

  predicate.args[1] = INT32_MIN;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_value_constraints_check_exact(
          IREE_SV("tile_size"), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
          contract_value, loom_attr_i64(-1),
          loom_attr_predicate_list(&predicate, 1)));
}

TEST(SymbolValueConstraintsTest, RejectsNonPositiveMultipleDivisor) {
  const loom_value_id_t contract_value = 3;
  for (int64_t divisor : {INT64_C(-16), INT64_C(0)}) {
    loom_predicate_t predicate = {
        .kind = LOOM_PREDICATE_MULTIPLE_OF,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
        .reserved = {},
        .args = {contract_value, divisor},
    };

    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_symbol_value_constraints_check_exact(
            IREE_SV("tile_size"), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
            contract_value, loom_attr_i64(64),
            loom_attr_predicate_list(&predicate, 1)));
  }
}

TEST(SymbolValueConstraintsTest, ChecksExactFloatingClassifications) {
  static constexpr loom_scalar_type_t kScalarTypes[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  static constexpr loom_predicate_kind_t kPredicateKinds[] = {
      LOOM_PREDICATE_NOT_NAN,
      LOOM_PREDICATE_NOT_INF,
      LOOM_PREDICATE_FINITE,
  };
  static constexpr double kValues[] = {
      1.5,
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };
  const loom_value_id_t contract_value = 5;

  for (const loom_scalar_type_t scalar_type : kScalarTypes) {
    for (const loom_predicate_kind_t predicate_kind : kPredicateKinds) {
      for (iree_host_size_t value_index = 0;
           value_index < IREE_ARRAYSIZE(kValues); ++value_index) {
        SCOPED_TRACE(loom_scalar_type_name(scalar_type));
        SCOPED_TRACE(loom_predicate_kind_name(predicate_kind));
        SCOPED_TRACE(value_index);
        loom_predicate_t predicate = {
            .kind = predicate_kind,
            .arg_count = 1,
            .arg_tags = {LOOM_PRED_ARG_VALUE},
            .reserved = {},
            .args = {contract_value},
        };

        // E4M3 has finite-only overflow semantics: infinities saturate to its
        // largest finite values before the predicate observes them.
        const bool is_nan = value_index == 1;
        const bool is_infinity = value_index >= 2;
        const bool saturates_infinity =
            scalar_type == LOOM_SCALAR_TYPE_F8E4M3 && is_infinity;
        bool expected = true;
        if (predicate_kind == LOOM_PREDICATE_NOT_NAN) {
          expected = !is_nan;
        } else if (predicate_kind == LOOM_PREDICATE_NOT_INF) {
          expected = !is_infinity || saturates_infinity;
        } else if (predicate_kind == LOOM_PREDICATE_FINITE) {
          expected = (!is_nan && !is_infinity) || saturates_infinity;
        }

        iree_status_t status = loom_symbol_value_constraints_check_exact(
            IREE_SV("scale"), loom_type_scalar(scalar_type), contract_value,
            loom_attr_f64(kValues[value_index]),
            loom_attr_predicate_list(&predicate, 1));
        if (expected) {
          IREE_EXPECT_OK(status);
        } else {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
        }
      }
    }
  }
}

TEST(SymbolValueConstraintsTest, ChecksExactFloatingEquality) {
  const loom_value_id_t contract_value = 5;
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_EQ,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {contract_value, 2},
  };

  IREE_ASSERT_OK(loom_symbol_value_constraints_check_exact(
      IREE_SV("scale"), loom_type_scalar(LOOM_SCALAR_TYPE_F32), contract_value,
      loom_attr_f64(2.0), loom_attr_predicate_list(&predicate, 1)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_value_constraints_check_exact(
          IREE_SV("scale"), loom_type_scalar(LOOM_SCALAR_TYPE_F32),
          contract_value, loom_attr_f64(2.5),
          loom_attr_predicate_list(&predicate, 1)));
}

TEST(SymbolValueConstraintsTest, FloatingEqualityUsesDeclaredTypeValue) {
  const loom_value_id_t contract_value = 5;
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_EQ,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {contract_value, 2048},
  };

  // 2049 rounds to 2048 in f16 before it becomes an exact config value.
  IREE_ASSERT_OK(loom_symbol_value_constraints_check_exact(
      IREE_SV("scale"), loom_type_scalar(LOOM_SCALAR_TYPE_F16), contract_value,
      loom_attr_f64(2049.0), loom_attr_predicate_list(&predicate, 1)));

  predicate.args[1] = 2049;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_value_constraints_check_exact(
          IREE_SV("scale"), loom_type_scalar(LOOM_SCALAR_TYPE_F16),
          contract_value, loom_attr_f64(2049.0),
          loom_attr_predicate_list(&predicate, 1)));
}

TEST(SymbolValueConstraintsTest, IgnoresPredicatesForOtherValues) {
  const loom_value_id_t contract_value = 3;
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_EQ,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {contract_value + 1, 0},
  };

  IREE_ASSERT_OK(loom_symbol_value_constraints_check_exact(
      IREE_SV("tile_size"), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      contract_value, loom_attr_i64(64),
      loom_attr_predicate_list(&predicate, 1)));
}

}  // namespace
