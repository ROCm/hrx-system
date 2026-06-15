// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/op_defs.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(DialectTableHelpers, ReturnVtableArraysAndCounts) {
  const loom_op_vtable_t vtable = {};
  const loom_op_vtable_t* const vtables[] = {
      &vtable,
  };

  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* result =
      loom_dialect_vtable_array(vtables, IREE_ARRAYSIZE(vtables), &count);
  EXPECT_EQ(result, vtables);
  EXPECT_EQ(count, 1u);

  EXPECT_EQ(loom_dialect_vtable_array(vtables, IREE_ARRAYSIZE(vtables),
                                      /*out_count=*/nullptr),
            vtables);
}

TEST(DialectTableHelpers, ReturnSemanticArraysAndCounts) {
  loom_op_semantics_t semantics[] = {
      loom_op_semantics_empty(),
      loom_op_semantics_empty(),
  };
  semantics[1].phase = LOOM_OP_PHASE_EXECUTABLE;
  semantics[1].contract_families = LOOM_CONTRACT_VECTOR_COORDINATE;

  iree_host_size_t count = 0;
  const loom_op_semantics_t* result = loom_dialect_semantics_array(
      semantics, IREE_ARRAYSIZE(semantics), &count);
  EXPECT_EQ(result, semantics);
  EXPECT_EQ(count, 2u);

  EXPECT_EQ(loom_dialect_semantics_array(semantics, IREE_ARRAYSIZE(semantics),
                                         /*out_count=*/nullptr),
            semantics);
}

TEST(DialectTableHelpers, LookupSemanticsByDialectAndIndex) {
  loom_op_semantics_t semantics[] = {
      loom_op_semantics_empty(),
      loom_op_semantics_empty(),
  };
  semantics[1].phase = LOOM_OP_PHASE_EXECUTABLE;
  semantics[1].contract_families = LOOM_CONTRACT_VECTOR_COORDINATE;

  loom_op_semantics_t found = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_TEST, 1), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(found.phase, LOOM_OP_PHASE_EXECUTABLE);
  EXPECT_EQ(found.contract_families, LOOM_CONTRACT_VECTOR_COORDINATE);

  loom_op_semantics_t wrong_dialect = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 1), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(wrong_dialect.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(wrong_dialect.contract_families, 0u);

  loom_op_semantics_t out_of_range = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_TEST, 2), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(out_of_range.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(out_of_range.contract_families, 0u);
}

}  // namespace
}  // namespace loom
