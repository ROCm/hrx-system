// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_registry.h"
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

  EXPECT_NE(loom_context_lookup_encoding_vtable(
                &context, iree_make_cstring_view("dense")),
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

TEST(TypeRegistry, DescriptorsCarrySemantics) {
  const loom_type_descriptor_t* token =
      loom_type_registry_lookup(iree_make_cstring_view("kernel.async.token"));
  ASSERT_NE(token, nullptr);
  EXPECT_EQ(token->semantics.semantic, LOOM_TYPE_SEMANTIC_CONTROL_TOKEN);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      token->semantics.contract_families, LOOM_CONTRACT_KERNEL_ASYNC));

  const loom_type_descriptor_t* tensor_descriptor = loom_type_registry_lookup(
      iree_make_cstring_view("kernel.tensor.lds.descriptor"));
  ASSERT_NE(tensor_descriptor, nullptr);
  EXPECT_EQ(tensor_descriptor->semantics.semantic,
            LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE);
  EXPECT_TRUE(loom_contract_family_set_has_any(
      tensor_descriptor->semantics.contract_families,
      LOOM_CONTRACT_TENSOR_MEMORY));

  const loom_type_descriptor_t* tile =
      loom_type_registry_lookup(iree_make_cstring_view("tile"));
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

  desc = loom_type_registry_lookup(iree_make_cstring_view("tile"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_TILE);
  EXPECT_EQ(desc->param_count, 3);
  EXPECT_NE(desc->format_elements, nullptr);
  EXPECT_GT(desc->format_element_count, 0);

  desc = loom_type_registry_lookup(iree_make_cstring_view("tensor"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_TENSOR);
  EXPECT_EQ(desc->param_count, 3);

  desc = loom_type_registry_lookup(iree_make_cstring_view("vector"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_VECTOR);
  EXPECT_EQ(desc->param_count, 2);
  EXPECT_NE(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 3);

  desc = loom_type_registry_lookup(iree_make_cstring_view("view"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_VIEW);
  EXPECT_EQ(desc->param_count, 3);
  EXPECT_NE(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 6);

  desc = loom_type_registry_lookup(iree_make_cstring_view("buffer"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_BUFFER);
  EXPECT_EQ(desc->param_count, 0);
  EXPECT_EQ(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 0);

  desc = loom_type_registry_lookup(iree_make_cstring_view("pool"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_POOL);
  EXPECT_EQ(desc->param_count, 1);

  desc = loom_type_registry_lookup(iree_make_cstring_view("group"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_GROUP);
  EXPECT_EQ(desc->param_count, 1);
}

TEST(TypeRegistry, LookupDialectType) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(iree_make_cstring_view("hal.buffer"));
  ASSERT_NE(desc, nullptr);
  EXPECT_EQ(desc->ir_kind, LOOM_TYPE_DIALECT);
  EXPECT_EQ(desc->param_count, 0);
  EXPECT_EQ(desc->format_elements, nullptr);
  EXPECT_EQ(desc->format_element_count, 0);
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
            loom_type_registry_resolve_fact_domain);
  EXPECT_EQ(table.context.resolve_type_domain.user_data, nullptr);

  loom_type_registry_configure_fact_context(nullptr);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(TypeRegistry, LookupUnknownReturnsNull) {
  EXPECT_EQ(
      loom_type_registry_lookup(iree_make_cstring_view("nonexistent.type")),
      nullptr);
}

TEST(TypeRegistry, LookupEmptyString) {
  EXPECT_EQ(loom_type_registry_lookup(iree_make_cstring_view("")), nullptr);
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
        loom_type_registry_lookup(entries[i].name);
    EXPECT_NE(desc, nullptr)
        << "entry " << i << " (\""
        << std::string(entries[i].name.data, entries[i].name.size)
        << "\") not found by lookup";
  }
}

TEST(TypeRegistry, TileFormatElements) {
  const loom_type_descriptor_t* desc =
      loom_type_registry_lookup(iree_make_cstring_view("tile"));
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
      loom_type_registry_lookup(iree_make_cstring_view("vector"));
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
      loom_type_registry_lookup(iree_make_cstring_view("view"));
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

}  // namespace
}  // namespace loom
