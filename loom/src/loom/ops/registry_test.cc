// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/test/registry.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Op registry
//===----------------------------------------------------------------------===//

TEST(OpRegistry, RegistersProductionContextSurface) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_op_kind_t kind = 0;
  const loom_op_vtable_t* vtable = loom_context_lookup_op_by_name(
      &context, iree_make_cstring_view("func.def"), &kind);
  ASSERT_NE(vtable, nullptr);
  EXPECT_EQ(kind, LOOM_OP_FUNC_DEF);

  vtable = loom_context_lookup_op_by_name(
      &context, iree_make_cstring_view("index.constant"), &kind);
  ASSERT_NE(vtable, nullptr);
  EXPECT_EQ(kind, LOOM_OP_INDEX_CONSTANT);

  kind = 0xFFFF;
  EXPECT_EQ(loom_context_lookup_op_by_name(
                &context, iree_make_cstring_view("nonexistent.op"), &kind),
            nullptr);
  EXPECT_EQ(kind, 0xFFFF);

  EXPECT_EQ(
      loom_context_lookup_op_by_name(&context, iree_string_view_empty(), &kind),
      nullptr);

  loom_op_semantics_t iota_semantics =
      loom_context_resolve_op_semantics(&context, LOOM_OP_VECTOR_IOTA);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      iota_semantics.contract_families, LOOM_CONTRACT_VECTOR_COORDINATE));

  loom_encoding_family_id_t dense_family_id =
      loom_context_resolve_encoding_name(&context,
                                         IREE_SV("encoding.layout.dense"))
          .family_id;
  EXPECT_NE(dense_family_id, LOOM_ENCODING_FAMILY_ID_INVALID);
  EXPECT_NE(loom_context_resolve_encoding_vtable(&context, dense_family_id),
            nullptr);
  EXPECT_EQ(loom_context_lookup_op_by_name(
                &context, iree_make_cstring_view("test.addi"), &kind),
            nullptr);

  loom_context_deinitialize(&context);
}

//===----------------------------------------------------------------------===//
// Semantic metadata
//===----------------------------------------------------------------------===//

TEST(OpSemantics, DialectTablesAreDenseAndQueryable) {
  iree_host_size_t count = 0;
  const loom_op_semantics_t* buffer_semantics =
      loom_buffer_dialect_op_semantics(&count);
  ASSERT_GT(count,
            (iree_host_size_t)loom_op_dialect_index(LOOM_OP_BUFFER_ALLOCA));

  loom_op_semantics_t alloca_semantics =
      loom_buffer_op_semantics(LOOM_OP_BUFFER_ALLOCA);
  EXPECT_EQ(alloca_semantics.phase, LOOM_OP_PHASE_EXECUTABLE);
  EXPECT_EQ(alloca_semantics.contract_families, 0u);
  EXPECT_EQ(
      buffer_semantics[loom_op_dialect_index(LOOM_OP_BUFFER_ALLOCA)].phase,
      LOOM_OP_PHASE_EXECUTABLE);

  EXPECT_EQ(loom_scalar_op_semantics(LOOM_OP_SCALAR_CEILDIVSI).phase,
            LOOM_OP_PHASE_EXECUTABLE);
  EXPECT_EQ(loom_scalar_op_semantics(LOOM_OP_SCALAR_POISON).phase,
            LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(loom_scalar_op_semantics(LOOM_OP_SCALAR_ASSUME).phase,
            LOOM_OP_PHASE_UNSPECIFIED);

  loom_op_semantics_t iota_semantics =
      loom_vector_op_semantics(LOOM_OP_VECTOR_IOTA);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      iota_semantics.contract_families, LOOM_CONTRACT_VECTOR_COORDINATE));

  loom_op_semantics_t tensor_descriptor_semantics =
      loom_kernel_op_semantics(LOOM_OP_KERNEL_TENSOR_LDS_DESCRIPTOR);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      tensor_descriptor_semantics.contract_families,
      LOOM_CONTRACT_TENSOR_MEMORY));

  loom_op_semantics_t tensor_load_semantics =
      loom_kernel_op_semantics(LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      tensor_load_semantics.contract_families,
      LOOM_CONTRACT_KERNEL_ASYNC | LOOM_CONTRACT_TENSOR_MEMORY));

  loom_op_semantics_t wrong_dialect =
      loom_vector_op_semantics(LOOM_OP_KERNEL_BARRIER);
  EXPECT_EQ(wrong_dialect.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(wrong_dialect.contract_families, 0u);
}

TEST(OpSemantics, AsyncContractAndCachePolicyIdentifyTransfers) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  struct ExpectedAsyncOp {
    loom_op_kind_t kind;
    bool is_transfer;
  };
  static constexpr ExpectedAsyncOp kExpectedAsyncOps[] = {
      {LOOM_OP_KERNEL_ASYNC_COPY, true},
      {LOOM_OP_KERNEL_ASYNC_COPY_MASK, true},
      {LOOM_OP_KERNEL_ASYNC_GATHER, true},
      {LOOM_OP_KERNEL_ASYNC_GATHER_MASK, true},
      {LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS, true},
      {LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS, true},
      {LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER, true},
      {LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK, true},
      {LOOM_OP_KERNEL_ASYNC_GROUP, false},
      {LOOM_OP_KERNEL_ASYNC_WAIT, false},
  };
  for (const auto& expected : kExpectedAsyncOps) {
    SCOPED_TRACE(expected.kind);
    const loom_op_semantics_t semantics =
        loom_context_resolve_op_semantics(&context, expected.kind);
    EXPECT_TRUE(loom_contract_family_set_has_any(semantics.contract_families,
                                                 LOOM_CONTRACT_KERNEL_ASYNC));
    const loom_op_vtable_t* vtable =
        loom_context_resolve_op(&context, expected.kind);
    ASSERT_NE(vtable, nullptr);
    EXPECT_EQ(vtable->cache_policy.available, expected.is_transfer);
  }

  iree_host_size_t vtable_count = 0;
  const loom_op_vtable_t* const* vtables =
      loom_kernel_dialect_vtables(&vtable_count);
  iree_host_size_t semantics_count = 0;
  const loom_op_semantics_t* semantics =
      loom_kernel_dialect_op_semantics(&semantics_count);
  ASSERT_EQ(vtable_count, semantics_count);
  iree_host_size_t async_op_count = 0;
  for (iree_host_size_t i = 0; i < semantics_count; ++i) {
    if (!loom_contract_family_set_has_any(semantics[i].contract_families,
                                          LOOM_CONTRACT_KERNEL_ASYNC)) {
      continue;
    }
    EXPECT_NE(vtables[i], nullptr);
    ++async_op_count;
  }
  EXPECT_EQ(async_op_count, IREE_ARRAYSIZE(kExpectedAsyncOps));

  loom_context_deinitialize(&context);
}

TEST(OpSemantics, IntegerArithmeticConditionsCoverEveryDialectForm) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  struct ExpectedCondition {
    loom_op_kind_t kind;
    loom_execution_condition_t condition;
  };
  static constexpr ExpectedCondition kExpectedConditions[] = {
      {LOOM_OP_SCALAR_ADDI, LOOM_EXECUTION_CONDITION_INTEGER_ADD},
      {LOOM_OP_VECTOR_ADDI, LOOM_EXECUTION_CONDITION_INTEGER_ADD},
      {LOOM_OP_INDEX_ADD, LOOM_EXECUTION_CONDITION_INTEGER_ADD},
      {LOOM_OP_SCALAR_SUBI, LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT},
      {LOOM_OP_VECTOR_SUBI, LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT},
      {LOOM_OP_INDEX_SUB, LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT},
      {LOOM_OP_SCALAR_MULI, LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY},
      {LOOM_OP_VECTOR_MULI, LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY},
      {LOOM_OP_INDEX_MUL, LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY},
      {LOOM_OP_SCALAR_FMAI, LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD},
      {LOOM_OP_VECTOR_FMAI, LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD},
      {LOOM_OP_INDEX_MADD, LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD},
  };
  for (const auto& expected : kExpectedConditions) {
    EXPECT_EQ(loom_context_resolve_op_semantics(&context, expected.kind)
                  .execution_condition,
              expected.condition);
  }

  loom_context_deinitialize(&context);
}

TEST(OpSemantics, NumericSpeculationDistinguishesTotalAndPartialFamilies) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  // These families are total for every type accepted by their verifier. Keep
  // the complete cross-dialect matrix together so a new spelling cannot
  // silently lose LICM and if-conversion legality.
  static constexpr loom_op_kind_t kTotalNumericOps[] = {
      // Integer extrema.
      LOOM_OP_SCALAR_MINSI,
      LOOM_OP_SCALAR_MAXSI,
      LOOM_OP_SCALAR_MINUI,
      LOOM_OP_SCALAR_MAXUI,
      LOOM_OP_VECTOR_MINSI,
      LOOM_OP_VECTOR_MAXSI,
      LOOM_OP_VECTOR_MINUI,
      LOOM_OP_VECTOR_MAXUI,
      LOOM_OP_INDEX_MIN,
      LOOM_OP_INDEX_MAX,
      // Bitwise logic.
      LOOM_OP_SCALAR_ANDI,
      LOOM_OP_SCALAR_ORI,
      LOOM_OP_SCALAR_XORI,
      LOOM_OP_VECTOR_ANDI,
      LOOM_OP_VECTOR_ORI,
      LOOM_OP_VECTOR_XORI,
      LOOM_OP_INDEX_ANDI,
      LOOM_OP_INDEX_ORI,
      LOOM_OP_INDEX_XORI,
      // Modulo-width rotates.
      LOOM_OP_SCALAR_ROTLI,
      LOOM_OP_SCALAR_ROTRI,
      LOOM_OP_VECTOR_ROTLI,
      LOOM_OP_VECTOR_ROTRI,
      LOOM_OP_INDEX_ROTLI,
      LOOM_OP_INDEX_ROTRI,
      // Bit counts, including zero inputs.
      LOOM_OP_SCALAR_CTLZI,
      LOOM_OP_SCALAR_CTTZI,
      LOOM_OP_SCALAR_CTPOPI,
      LOOM_OP_VECTOR_CTLZI,
      LOOM_OP_VECTOR_CTTZI,
      LOOM_OP_VECTOR_CTPOPI,
      LOOM_OP_INDEX_CTLZI,
      LOOM_OP_INDEX_CTTZI,
      LOOM_OP_INDEX_CTPOPI,
      // Comparisons and selection.
      LOOM_OP_SCALAR_CMPI,
      LOOM_OP_SCALAR_CMPF,
      LOOM_OP_VECTOR_CMPI,
      LOOM_OP_VECTOR_CMPF,
      LOOM_OP_VECTOR_SELECT,
      LOOM_OP_INDEX_CMP,
      // Width and numeric conversions.
      LOOM_OP_SCALAR_SITOFP,
      LOOM_OP_SCALAR_UITOFP,
      LOOM_OP_SCALAR_EXTF,
      LOOM_OP_SCALAR_FPTRUNC,
      LOOM_OP_SCALAR_EXTSI,
      LOOM_OP_SCALAR_EXTUI,
      LOOM_OP_SCALAR_TRUNCI,
      LOOM_OP_SCALAR_BITCAST,
      LOOM_OP_VECTOR_SITOFP,
      LOOM_OP_VECTOR_UITOFP,
      LOOM_OP_VECTOR_EXTF,
      LOOM_OP_VECTOR_FPTRUNC,
      LOOM_OP_VECTOR_EXTSI,
      LOOM_OP_VECTOR_EXTUI,
      LOOM_OP_VECTOR_TRUNCI,
      LOOM_OP_VECTOR_BITCAST,
  };
  for (loom_op_kind_t kind : kTotalNumericOps) {
    SCOPED_TRACE(kind);
    const loom_op_vtable_t* vtable = loom_context_resolve_op(&context, kind);
    ASSERT_NE(vtable, nullptr);
    EXPECT_TRUE(loom_traits_are_safe_to_speculate(vtable->traits));
  }

  // Neighboring partial operations require operand facts or target-carrier
  // proofs. A family-wide annotation must not erase that distinction.
  static constexpr loom_op_kind_t kPartialNumericOps[] = {
      LOOM_OP_SCALAR_FPTOSI, LOOM_OP_SCALAR_FPTOUI, LOOM_OP_VECTOR_FPTOSI,
      LOOM_OP_VECTOR_FPTOUI, LOOM_OP_INDEX_CAST,    LOOM_OP_SCALAR_SHLI,
      LOOM_OP_SCALAR_SHRSI,  LOOM_OP_SCALAR_SHRUI,  LOOM_OP_VECTOR_SHLI,
      LOOM_OP_VECTOR_SHRSI,  LOOM_OP_VECTOR_SHRUI,  LOOM_OP_INDEX_SHLI,
      LOOM_OP_INDEX_SHRSI,   LOOM_OP_INDEX_SHRUI,   LOOM_OP_SCALAR_DIVSI,
      LOOM_OP_SCALAR_DIVUI,  LOOM_OP_SCALAR_REMSI,  LOOM_OP_SCALAR_REMUI,
      LOOM_OP_VECTOR_DIVSI,  LOOM_OP_VECTOR_DIVUI,  LOOM_OP_VECTOR_REMSI,
      LOOM_OP_VECTOR_REMUI,  LOOM_OP_INDEX_DIV,     LOOM_OP_INDEX_REM,
  };
  for (loom_op_kind_t kind : kPartialNumericOps) {
    SCOPED_TRACE(kind);
    const loom_op_vtable_t* vtable = loom_context_resolve_op(&context, kind);
    ASSERT_NE(vtable, nullptr);
    EXPECT_FALSE(loom_traits_are_safe_to_speculate(vtable->traits));
  }

  loom_context_deinitialize(&context);
}

TEST(OpSemantics, EagerExecutionCostSeparatesLegalityFromProfitability) {
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  struct ExpectedCost {
    loom_op_kind_t kind;
    uint8_t cost;
  };
  static constexpr ExpectedCost kExpectedCosts[] = {
      {LOOM_OP_SCALAR_FMAF, 1},   {LOOM_OP_VECTOR_ROUNDEVENF, 1},
      {LOOM_OP_SCALAR_SILUF, 0},  {LOOM_OP_VECTOR_MMA, 0},
      {LOOM_OP_VECTOR_REDUCE, 0},
  };
  for (const auto& expected : kExpectedCosts) {
    SCOPED_TRACE(expected.kind);
    const loom_op_vtable_t* vtable =
        loom_context_resolve_op(&context, expected.kind);
    ASSERT_NE(vtable, nullptr);
    EXPECT_TRUE(loom_traits_are_safe_to_speculate(vtable->traits));
    EXPECT_EQ(loom_context_resolve_op_semantics(&context, expected.kind)
                  .eager_execution_cost,
              expected.cost);
  }

  loom_context_deinitialize(&context);
}

TEST(TypeRegistry, DescriptorsCarrySemantics) {
  const loom_type_descriptor_t* token = loom_type_registry_lookup(
      nullptr, iree_make_cstring_view("kernel.async.token"));
  ASSERT_NE(token, nullptr);
  EXPECT_EQ(token->semantics.semantic, LOOM_TYPE_SEMANTIC_CONTROL_TOKEN);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      token->semantics.contract_families, LOOM_CONTRACT_KERNEL_ASYNC));

  const loom_type_descriptor_t* tensor_descriptor = loom_type_registry_lookup(
      nullptr, iree_make_cstring_view("kernel.tensor.lds.descriptor"));
  ASSERT_NE(tensor_descriptor, nullptr);
  EXPECT_EQ(tensor_descriptor->semantics.semantic,
            LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      tensor_descriptor->semantics.contract_families,
      LOOM_CONTRACT_TENSOR_MEMORY));

  const loom_type_descriptor_t* tile =
      loom_type_registry_lookup(nullptr, iree_make_cstring_view("tile"));
  ASSERT_NE(tile, nullptr);
  EXPECT_EQ(tile->semantics.semantic, LOOM_TYPE_SEMANTIC_ORDINARY);
  EXPECT_EQ(tile->semantics.contract_families, 0u);
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

TEST(TypeConstraint, ScalarAddressFamiliesAreExplicit) {
  loom_type_t scalar_i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t scalar_index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t scalar_offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);

  EXPECT_STREQ("scalar",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_SCALAR));
  EXPECT_STREQ("index", loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_INDEX));
  EXPECT_STREQ("offset",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_OFFSET));
  EXPECT_STREQ("address",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_ADDRESS));

  EXPECT_TRUE(
      loom_type_satisfies_constraint(scalar_i32, LOOM_TYPE_CONSTRAINT_SCALAR));
  EXPECT_TRUE(loom_type_satisfies_constraint(scalar_index,
                                             LOOM_TYPE_CONSTRAINT_SCALAR));
  EXPECT_TRUE(loom_type_satisfies_constraint(scalar_offset,
                                             LOOM_TYPE_CONSTRAINT_SCALAR));
  EXPECT_TRUE(loom_type_satisfies_constraint(scalar_index,
                                             LOOM_TYPE_CONSTRAINT_ADDRESS));
  EXPECT_TRUE(loom_type_satisfies_constraint(scalar_offset,
                                             LOOM_TYPE_CONSTRAINT_ADDRESS));
  EXPECT_FALSE(
      loom_type_satisfies_constraint(scalar_i32, LOOM_TYPE_CONSTRAINT_ADDRESS));

  EXPECT_TRUE(
      loom_type_satisfies_constraint(scalar_index, LOOM_TYPE_CONSTRAINT_INDEX));
  EXPECT_FALSE(loom_type_satisfies_constraint(scalar_offset,
                                              LOOM_TYPE_CONSTRAINT_INDEX));
  EXPECT_TRUE(loom_type_satisfies_constraint(scalar_offset,
                                             LOOM_TYPE_CONSTRAINT_OFFSET));
  EXPECT_FALSE(loom_type_satisfies_constraint(scalar_index,
                                              LOOM_TYPE_CONSTRAINT_OFFSET));
  EXPECT_FALSE(
      loom_type_satisfies_constraint(scalar_i32, LOOM_TYPE_CONSTRAINT_OFFSET));
}

TEST(TypeConstraint, RegisterFamilyIsExplicit) {
  loom_type_t reg = loom_type_register_payload(42, 4);
  loom_type_t scalar_i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t vector_i32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);

  EXPECT_STREQ("register",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_REGISTER));
  EXPECT_TRUE(
      loom_type_satisfies_constraint(reg, LOOM_TYPE_CONSTRAINT_REGISTER));
  EXPECT_FALSE(loom_type_satisfies_constraint(scalar_i32,
                                              LOOM_TYPE_CONSTRAINT_REGISTER));
  EXPECT_FALSE(loom_type_satisfies_constraint(vector_i32,
                                              LOOM_TYPE_CONSTRAINT_REGISTER));
}

TEST(TypeConstraint, ElementFamiliesRequireShapedTypes) {
  const uint64_t lanes = loom_dim_pack_static(4);
  loom_type_t vector_i8 =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, lanes, 0);
  loom_type_t vector_i32 =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, lanes, 0);
  loom_type_t vector_i16 =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I16, lanes, 0);
  loom_type_t vector_f16 =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F16, lanes, 0);
  loom_type_t vector_bf16 =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_BF16, lanes, 0);
  loom_type_t vector_f32 =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, lanes, 0);
  loom_type_t vector_i1 =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, lanes, 0);
  loom_type_t tile_i32 =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, lanes, 0);
  loom_type_t tensor_i32 =
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_I32, lanes, 0);
  loom_type_t view_i32 =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32, lanes, 0);
  loom_type_t tile_f32 =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, lanes, 0);
  loom_type_t scalar_i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  EXPECT_STREQ("integer_element",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT));
  EXPECT_STREQ("float_element",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT));
  EXPECT_STREQ("i1_element",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_I1_ELEMENT));
  EXPECT_STREQ("i8 element type",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_I8_ELEMENT));
  EXPECT_STREQ("i32 element type",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_I32_ELEMENT));
  EXPECT_STREQ(
      "f16 or bf16 element type",
      loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT));
  EXPECT_STREQ("f32 element type",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_F32_ELEMENT));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_i32, LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      tile_i32, LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      tensor_i32, LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      view_i32, LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_f32, LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_i32, LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_f32, LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      tile_f32, LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_i32, LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT));

  EXPECT_TRUE(loom_type_satisfies_constraint(vector_i1,
                                             LOOM_TYPE_CONSTRAINT_I1_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(vector_i32,
                                              LOOM_TYPE_CONSTRAINT_I1_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(vector_i8,
                                             LOOM_TYPE_CONSTRAINT_I8_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(vector_i16,
                                              LOOM_TYPE_CONSTRAINT_I8_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(vector_i32,
                                             LOOM_TYPE_CONSTRAINT_I32_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_i8, LOOM_TYPE_CONSTRAINT_I32_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_f16, LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_bf16, LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_f32, LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(vector_f32,
                                             LOOM_TYPE_CONSTRAINT_F32_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_f16, LOOM_TYPE_CONSTRAINT_F32_ELEMENT));
}

TEST(TypeConstraint, IndexOrNonI1IntegerFamiliesAreExplicit) {
  loom_type_t scalar_index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t scalar_offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t scalar_i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t scalar_i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t scalar_f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t vector_index = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_INDEX, loom_dim_pack_static(4), 0);
  loom_type_t vector_offset = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_OFFSET, loom_dim_pack_static(4), 0);
  loom_type_t vector_i1 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, loom_dim_pack_static(4), 0);
  loom_type_t vector_i8 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(4), 0);
  loom_type_t vector_f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  EXPECT_STREQ("index or non-i1 integer scalar",
               loom_type_constraint_name(
                   LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR));
  EXPECT_STREQ("index or non-i1 integer element type",
               loom_type_constraint_name(
                   LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      scalar_index, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      scalar_i32, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_i1, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_offset, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_f32, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_i8, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_index, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_i8, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_i1, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_offset, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_f32, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_i32, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT));
}

TEST(TypeConstraint, BitwiseFamiliesAcceptIntegerAndFloatPayloads) {
  loom_type_t scalar_index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t scalar_offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_type_t scalar_i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t scalar_i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t scalar_f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t vector_i1 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, loom_dim_pack_static(4), 0);
  loom_type_t vector_i8 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(4), 0);
  loom_type_t vector_f32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  EXPECT_TRUE(loom_type_satisfies_constraint(
      scalar_index, LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      scalar_i32, LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      scalar_f32, LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_i1, LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_offset, LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_i8, LOOM_TYPE_CONSTRAINT_BITWISE_ELEMENT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_f32, LOOM_TYPE_CONSTRAINT_BITWISE_ELEMENT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_i1, LOOM_TYPE_CONSTRAINT_BITWISE_ELEMENT));
}

TEST(TypeConstraint, VectorShapeFamiliesRequireVectorTypes) {
  loom_type_t vector_1d = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t vector_dynamic_1d = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(0), 0);
  loom_type_t vector_2d =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(2), loom_dim_pack_static(2), 0);
  loom_type_t tile_1d = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  EXPECT_STREQ("rank-1 vector",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR));
  EXPECT_STREQ(
      "all-static vector shape",
      loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR));
  EXPECT_STREQ("all-static rank-1 vector",
               loom_type_constraint_name(
                   LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_1d, LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_dynamic_1d, LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_2d, LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      tile_1d, LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_2d, LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_dynamic_1d, LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      tile_1d, LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      vector_1d, LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_2d, LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_dynamic_1d, LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR));
}

TEST(TypeConstraint, EncodingRolesAreExplicit) {
  loom_type_t any_encoding = loom_type_encoding();
  loom_type_t layout =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  loom_type_t schema =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_STORAGE_SCHEMA);
  loom_type_t storage =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_PHYSICAL_STORAGE);
  loom_type_t transform =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM);
  loom_type_t scalar_i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  EXPECT_STREQ("encoding",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_ANY_ENCODING));
  EXPECT_STREQ("encoding<layout>",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT));
  EXPECT_STREQ("encoding<schema>",
               loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA));
  EXPECT_STREQ("encoding<storage>", loom_type_constraint_name(
                                        LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE));
  EXPECT_STREQ(
      "encoding<transform>",
      loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      any_encoding, LOOM_TYPE_CONSTRAINT_ANY_ENCODING));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      layout, LOOM_TYPE_CONSTRAINT_ANY_ENCODING));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      scalar_i32, LOOM_TYPE_CONSTRAINT_ANY_ENCODING));

  EXPECT_TRUE(loom_type_satisfies_constraint(
      layout, LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      schema, LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      storage, LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE));
  EXPECT_TRUE(loom_type_satisfies_constraint(
      transform, LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM));

  EXPECT_FALSE(loom_type_satisfies_constraint(
      any_encoding, LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      schema, LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT));
}

//===----------------------------------------------------------------------===//
// Type registry
//===----------------------------------------------------------------------===//

TEST(TypeRegistry, LookupBuiltinTypes) {
  const loom_type_descriptor_t* desc;

  desc = loom_type_registry_lookup(nullptr, iree_make_cstring_view("tile"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_TILE);
  EXPECT_EQ(loom_type_registry_lookup_builtin(LOOM_TYPE_TILE), desc);
  EXPECT_EQ(desc->param_count, 3);
  EXPECT_NE(desc->format_elements, nullptr);
  EXPECT_GT(desc->format_element_count, 0);

  desc = loom_type_registry_lookup(nullptr, iree_make_cstring_view("tensor"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_TENSOR);
  EXPECT_EQ(desc->param_count, 3);

  desc = loom_type_registry_lookup(nullptr, iree_make_cstring_view("vector"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_VECTOR);
  EXPECT_EQ(desc->param_count, 2);
  EXPECT_NE(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 3);

  desc = loom_type_registry_lookup(nullptr, iree_make_cstring_view("view"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_VIEW);
  EXPECT_EQ(desc->param_count, 4);
  EXPECT_NE(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 13);

  desc = loom_type_registry_lookup(nullptr, iree_make_cstring_view("buffer"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_BUFFER);
  EXPECT_EQ(desc->param_count, 0);
  EXPECT_EQ(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 0);

  desc = loom_type_registry_lookup(nullptr, iree_make_cstring_view("pool"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_POOL);
  EXPECT_EQ(desc->param_count, 0);
  EXPECT_EQ(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 0);

  desc = loom_type_registry_lookup(nullptr, iree_make_cstring_view("encoding"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_ENCODING);
  EXPECT_EQ(loom_type_registry_lookup_builtin(LOOM_TYPE_ENCODING), desc);
  EXPECT_EQ(desc->parameterized, &loom_encoding_type_parameterized_descriptor);
  EXPECT_TRUE(iree_string_view_equal(
      loom_encoding_type_role_name(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      IREE_SV("layout")));

  desc =
      loom_type_registry_lookup(nullptr, iree_make_cstring_view("low.storage"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_STORAGE);
  EXPECT_EQ(loom_type_registry_lookup_builtin(LOOM_TYPE_STORAGE), desc);
  EXPECT_EQ(desc->parameterized,
            &loom_low_storage_type_parameterized_descriptor);
  loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_COUNT_;
  EXPECT_TRUE(
      loom_low_storage_type_space_parse(IREE_SV("workgroup"), &storage_space));
  EXPECT_EQ(storage_space, LOOM_STORAGE_SPACE_WORKGROUP);

  EXPECT_EQ(loom_type_registry_lookup_builtin(LOOM_TYPE_DIALECT), nullptr);
  EXPECT_EQ(loom_type_registry_lookup_builtin(LOOM_TYPE_PARAMETERIZED),
            nullptr);
  EXPECT_EQ(loom_type_registry_lookup_builtin(LOOM_TYPE_COUNT_), nullptr);
}

TEST(TypeRegistry, RegisteredNamesCarryOpeningAngleByte) {
  const loom_type_registry_entry_t* entries = loom_type_registry_entries();
  const iree_host_size_t entry_count = loom_type_registry_count();
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const loom_type_descriptor_t* descriptor = entries[i].descriptor;
    const iree_string_view_t name = loom_bstring_view(descriptor->name);
    EXPECT_EQ(descriptor->name[name.size + 1], '<');
    if (descriptor->parameterized) {
      EXPECT_EQ(descriptor->parameterized->type_descriptor, descriptor);
      EXPECT_EQ(descriptor->parameterized->parameter_count,
                descriptor->param_count);
    }
  }
}

TEST(TypeRegistry, LookupDialectType) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(nullptr, iree_make_cstring_view("hal.buffer"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_DIALECT);
  EXPECT_EQ(desc->param_count, 0);
  EXPECT_EQ(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 0);
}

TEST(TypeRegistry, DialectOwnedTypesRequireDialectRegistration) {
  EXPECT_EQ(loom_type_registry_lookup(nullptr, IREE_SV("test.scope")), nullptr);

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_test_dialect_register(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  const loom_type_descriptor_t* descriptor =
      loom_type_registry_lookup(&context, IREE_SV("test.scope"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->ir_kind, LOOM_TYPE_PARAMETERIZED);
  EXPECT_EQ(loom_type_registry_lookup(&context, IREE_SV("tile")),
            loom_type_registry_lookup_builtin(LOOM_TYPE_TILE));

  for (iree_host_size_t i = 0; i < context.registered_types.count; ++i) {
    const auto* registered = context.registered_types.entries[i].descriptor;
    if (registered->parameterized) {
      EXPECT_EQ(registered->parameterized->type_descriptor, registered);
      EXPECT_EQ(registered->parameterized->parameter_count,
                registered->param_count);
    }
  }

  loom_context_deinitialize(&context);
}

TEST(TypeRegistry, ConfiguresFactContextResolver) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena, 8));
  ASSERT_EQ(table.context.resolve_type_domain.fn, nullptr);

  loom_type_registry_configure_fact_context(&table.context);
  EXPECT_EQ(table.context.resolve_type_domain.fn,
            &loom_type_registry_resolve_fact_domain);
  EXPECT_EQ(table.context.resolve_type_domain.user_data, nullptr);

  loom_type_registry_configure_fact_context(nullptr);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(TypeRegistry, LookupUnknownReturnsNull) {
  EXPECT_EQ(loom_type_registry_lookup(
                nullptr, iree_make_cstring_view("nonexistent.type")),
            nullptr);
}

TEST(TypeRegistry, LookupEmptyString) {
  EXPECT_EQ(loom_type_registry_lookup(nullptr, iree_make_cstring_view("")),
            nullptr);
}

TEST(TypeRegistry, EntriesAreSorted) {
  const loom_type_registry_entry_t* entries = loom_type_registry_entries();
  iree_host_size_t count = loom_type_registry_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 1; i < count; ++i) {
    EXPECT_LT(iree_string_view_compare(entries[i - 1].name, entries[i].name), 0)
        << "entries[" << i - 1 << "] (\""
        << std::string(entries[i - 1].name.data, entries[i - 1].name.size)
        << "\") >= entries[" << i << "] (\""
        << std::string(entries[i].name.data, entries[i].name.size) << "\")";
  }
}

TEST(TypeRegistry, AllEntriesValid) {
  const loom_type_registry_entry_t* entries = loom_type_registry_entries();
  iree_host_size_t count = loom_type_registry_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 0; i < count; ++i) {
    EXPECT_GT(entries[i].name.size, 0u) << "entry " << i << " has empty name";
    const loom_type_descriptor_t* desc =
        loom_type_registry_lookup(nullptr, entries[i].name);
    EXPECT_NE(desc, nullptr)
        << "entry " << i << " (\""
        << std::string(entries[i].name.data, entries[i].name.size)
        << "\") not found by lookup";
  }
}

TEST(TypeRegistry, TileFormatElements) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(nullptr, iree_make_cstring_view("tile"));
  ASSERT_NE(desc, nullptr);
  ASSERT_EQ(desc->format_element_count, 6);
  // ShapeOf, Keyword(x), ScalarOf, Optional, Keyword(,), EncodingOf.
  EXPECT_EQ(desc->format_elements[0].kind, LOOM_TYPE_FMT_SHAPE);
  EXPECT_EQ(desc->format_elements[1].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[1].data, LOOM_KW_X);
  EXPECT_EQ(desc->format_elements[2].kind, LOOM_TYPE_FMT_SCALAR);
  EXPECT_EQ(desc->format_elements[3].kind, LOOM_TYPE_FMT_OPTIONAL);
  EXPECT_EQ(desc->format_elements[4].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[4].data, LOOM_KW_COMMA);
  EXPECT_EQ(desc->format_elements[5].kind, LOOM_TYPE_FMT_ENCODING);
}

TEST(TypeRegistry, VectorFormatElements) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(nullptr, iree_make_cstring_view("vector"));
  ASSERT_NE(desc, nullptr);
  ASSERT_EQ(desc->format_element_count, 3);
  // ShapeOf, Keyword(x), ScalarOf.
  EXPECT_EQ(desc->format_elements[0].kind, LOOM_TYPE_FMT_SHAPE);
  EXPECT_EQ(desc->format_elements[1].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[1].data, LOOM_KW_X);
  EXPECT_EQ(desc->format_elements[2].kind, LOOM_TYPE_FMT_SCALAR);
}

TEST(TypeRegistry, ViewFormatElements) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(nullptr, iree_make_cstring_view("view"));
  ASSERT_NE(desc, nullptr);
  ASSERT_EQ(desc->format_element_count, 13);
  // ShapeOf, Keyword(x), ScalarOf, optional encoding and alignment clauses.
  EXPECT_EQ(desc->format_elements[0].kind, LOOM_TYPE_FMT_SHAPE);
  EXPECT_EQ(desc->format_elements[1].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[1].data, LOOM_KW_X);
  EXPECT_EQ(desc->format_elements[2].kind, LOOM_TYPE_FMT_SCALAR);
  EXPECT_EQ(desc->format_elements[3].kind, LOOM_TYPE_FMT_OPTIONAL);
  EXPECT_EQ(desc->format_elements[4].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[4].data, LOOM_KW_COMMA);
  EXPECT_EQ(desc->format_elements[5].kind, LOOM_TYPE_FMT_ENCODING);
  EXPECT_EQ(desc->format_elements[6].kind, LOOM_TYPE_FMT_OPTIONAL);
  EXPECT_EQ(desc->format_elements[6].field_index, 3);
  EXPECT_EQ(desc->format_elements[6].data, (6 << 8) | 3);
  EXPECT_EQ(desc->format_elements[7].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[7].data, LOOM_KW_COMMA);
  EXPECT_EQ(desc->format_elements[8].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[8].data, LOOM_KW_ALIGN);
  EXPECT_EQ(desc->format_elements[9].kind, LOOM_TYPE_FMT_GLUE);
  EXPECT_EQ(desc->format_elements[10].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[10].data, LOOM_KW_LPAREN);
  EXPECT_EQ(desc->format_elements[11].kind, LOOM_TYPE_FMT_ALIGNMENT);
  EXPECT_EQ(desc->format_elements[11].field_index, 3);
  EXPECT_EQ(desc->format_elements[12].kind, LOOM_TYPE_FMT_KEYWORD);
  EXPECT_EQ(desc->format_elements[12].data, LOOM_KW_RPAREN);
}

}  // namespace
}  // namespace loom
