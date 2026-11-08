// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/signature.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <array>

#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"

namespace loom::cxx_import {
namespace {

TEST_F(ValueBuilderTest, BindsFunctionArgumentsAndResultsToTheirOwnIdentities) {
  Source source(
      IREE_SV("namespace loom { namespace encoding { "
              "enum class role { layout }; } namespace type { "
              "using size_type = __SIZE_TYPE__; "
              "inline constexpr size_type dynamic = ~size_type{0}; "
              "template <loom::encoding::role Role, size_type Rank> "
              "struct [[loom::type(\"encoding\")]] encoding { "
              "size_type strides[Rank]; }; "
              "template <size_type... Extents> "
              "struct [[loom::type(\"shape\")]] shape {}; "
              "template <class Shape, class T, class Encoding = void> "
              "struct [[loom::type(\"view\")]] view { T* data; }; "
              "} } using Plane = loom::type::view<"
              "loom::type::shape<loom::type::dynamic, 32>, "
              "const float, loom::type::encoding<"
              "loom::encoding::role::layout, 2>>;"),
      IREE_SV("signature.cpp"), options());
  Types types(source.unit(), source.diagnostics());
  auto symbols = source.unit().globalScope()->find("Plane");
  ASSERT_NE(symbols.begin(), symbols.end());
  const cxx::Type* plane = (*symbols.begin())->type();
  std::array<const cxx::Type*, 2> sources = {plane, plane};

  auto signature =
      bind_signature(types, sources, source.unit().ast(), &builder_);
  ASSERT_EQ(signature.types.size(), 6u);
  ASSERT_EQ(signature.identities.size(), 6u);
  loom_string_id_t name;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("forward"), &name));
  loom_symbol_id_t symbol;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* op;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, {0, symbol},
                                      signature.types.data(), 3,
                                      signature.types.data() + 3, 3, nullptr, 0,
                                      nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));

  auto* entry = loom_region_entry_block(loom_test_func_body(op));
  ASSERT_EQ(entry->arg_count, 3u);
  for (size_t index = 0; index < 3; ++index) {
    EXPECT_EQ(entry->arg_ids[index], signature.identities[index]);
    EXPECT_EQ(loom_op_results(op)[index], signature.identities[index + 3]);
  }
  auto argument_view = loom_module_value_type(module_, entry->arg_ids[2]);
  EXPECT_EQ(loom_type_dim_value_id_at(argument_view, 0), entry->arg_ids[0]);
  EXPECT_EQ(loom_type_encoding_value_id(argument_view), entry->arg_ids[1]);
  auto result_view = loom_module_value_type(module_, loom_op_results(op)[2]);
  EXPECT_EQ(loom_type_dim_value_id_at(result_view, 0), loom_op_results(op)[0]);
  EXPECT_EQ(loom_type_encoding_value_id(result_view), loom_op_results(op)[1]);
}

TEST_F(ValueBuilderTest, BindsDependentVectorsToEachCallableIdentity) {
  Source source(IREE_SV(R"(
    namespace loom { namespace type {
    using size_type = __SIZE_TYPE__;
    inline constexpr size_type dynamic = ~size_type{0};
    template <size_type... Extents>
    struct [[loom::type("shape")]] shape {};
    template <class Shape, class T>
    struct [[loom::type("vector")]] vector {};
    } }
    using Fragment = loom::type::vector<
        loom::type::shape<loom::type::dynamic>, float>;
  )"),
                IREE_SV("vector_signature.cxx"), options());
  Types types(source.unit(), source.diagnostics());
  auto symbols = source.unit().globalScope()->find("Fragment");
  ASSERT_NE(symbols.begin(), symbols.end());
  const cxx::Type* fragment = (*symbols.begin())->type();
  std::array<const cxx::Type*, 2> sources = {fragment, fragment};

  auto signature =
      bind_signature(types, sources, source.unit().ast(), &builder_);
  ASSERT_EQ(signature.types.size(), 4u);
  ASSERT_EQ(signature.identities.size(), 4u);
  loom_string_id_t name;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("forward"), &name));
  loom_symbol_id_t symbol;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* op;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, {0, symbol},
                                      signature.types.data(), 2,
                                      signature.types.data() + 2, 2, nullptr, 0,
                                      nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));

  auto* entry = loom_region_entry_block(loom_test_func_body(op));
  ASSERT_EQ(entry->arg_count, 2u);
  EXPECT_EQ(entry->arg_ids[0], signature.identities[0]);
  EXPECT_EQ(entry->arg_ids[1], signature.identities[1]);
  EXPECT_EQ(loom_op_results(op)[0], signature.identities[2]);
  EXPECT_EQ(loom_op_results(op)[1], signature.identities[3]);
  auto argument_vector = loom_module_value_type(module_, entry->arg_ids[1]);
  EXPECT_EQ(loom_type_dim_value_id_at(argument_vector, 0), entry->arg_ids[0]);
  auto result_vector = loom_module_value_type(module_, loom_op_results(op)[1]);
  EXPECT_EQ(loom_type_dim_value_id_at(result_vector, 0),
            loom_op_results(op)[0]);
}

TEST_F(ValueBuilderTest, ReservesStaticIdentitiesForCallableContracts) {
  Source source(IREE_SV("unsigned forward(unsigned input);"),
                IREE_SV("signature.cxx"), options());
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  cxx::SimpleDeclarationAST* simple = nullptr;
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    if ((simple = cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration))) {
      break;
    }
  }
  ASSERT_NE(simple, nullptr);
  auto* declarator = simple->initDeclaratorList->value;
  auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(declarator->symbol);
  auto* function_type = cxx::type_cast<cxx::FunctionType>(function->type());
  std::array<const cxx::Type*, 2> sources = {function->parameters()[0]->type(),
                                             function_type->returnType()};
  Types types(source.unit(), source.diagnostics());

  auto signature = bind_signature(types, sources, declarator, &builder_,
                                  SignatureIdentityRequirement::Required);
  ASSERT_EQ(signature.types.size(), 2u);
  ASSERT_EQ(signature.identities.size(), 2u);

  loom_string_id_t name;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("forward"), &name));
  loom_symbol_id_t symbol;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* op;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, {0, symbol},
                                      signature.types.data(), 1,
                                      signature.types.data() + 1, 1, nullptr, 0,
                                      nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));

  auto* entry = loom_region_entry_block(loom_test_func_body(op));
  ASSERT_EQ(entry->arg_count, 1u);
  EXPECT_EQ(entry->arg_ids[0], signature.identities[0]);
  EXPECT_EQ(loom_op_results(op)[0], signature.identities[1]);
}

TEST_F(ValueBuilderTest, LeavesStaticConditionLoopSignaturesInferred) {
  std::array<const cxx::Type*, 1> sources = {
      source_.unit().control()->getUnsignedIntType()};

  auto condition = bind_condition_loop_signatures(
      types_, sources, sources, source_.unit().ast(), &builder_);
  EXPECT_TRUE(condition.identities.empty());
  EXPECT_TRUE(condition.header_types.empty());
  EXPECT_TRUE(condition.result_types.empty());
  EXPECT_EQ(builder_.reserved_value_count, 0u);

  auto differing_arity = bind_condition_loop_signatures(
      types_, sources, sources, source_.unit().ast(), &builder_, {},
      /*require_result_types=*/true);
  EXPECT_TRUE(differing_arity.identities.empty());
  EXPECT_TRUE(differing_arity.header_types.empty());
  ASSERT_EQ(differing_arity.result_types.size(), 1u);
  EXPECT_EQ(builder_.reserved_value_count, 0u);
}

TEST_F(ValueBuilderTest, BindsConditionLoopHeaderAndResultsIndependently) {
  Source source(
      IREE_SV("namespace loom { namespace encoding { "
              "enum class role { layout }; } namespace type { "
              "using size_type = __SIZE_TYPE__; "
              "inline constexpr size_type dynamic = ~size_type{0}; "
              "template <size_type... Extents> "
              "struct [[loom::type(\"shape\")]] shape {}; "
              "template <loom::encoding::role Role, size_type Rank> "
              "struct [[loom::type(\"encoding\")]] encoding { "
              "size_type strides[Rank]; }; "
              "template <class Shape, class T, class Encoding = void> "
              "struct [[loom::type(\"view\")]] view { T* data; }; "
              "} } using Row = loom::type::view<"
              "loom::type::shape<loom::type::dynamic>, "
              "const float, loom::type::encoding<"
              "loom::encoding::role::layout, 1>>;"),
      IREE_SV("condition_loop_signature.cxx"), options());
  Types types(source.unit(), source.diagnostics());
  auto symbols = source.unit().globalScope()->find("Row");
  ASSERT_NE(symbols.begin(), symbols.end());
  const cxx::Type* row = (*symbols.begin())->type();
  std::array<const cxx::Type*, 1> sources = {row};

  loom_value_id_t extent;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &extent));
  loom_value_id_t layout;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_),
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      &layout));
  std::array<loom_value_id_t, 3> initial_identities = {extent, layout,
                                                       LOOM_VALUE_ID_INVALID};
  BoundTypeStorage type_storage;
  std::vector<loom_type_t> initial_types;
  types.append_bound(row, source.unit().ast(), initial_identities, type_storage,
                     initial_types);
  ASSERT_EQ(initial_types.size(), 3u);
  loom_value_id_t view;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_), initial_types[2], &view));
  loom_value_id_t active;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), &active));

  const std::array<loom_type_t, 1> header_suffix = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I1)};
  auto signatures = bind_condition_loop_signatures(
      types, sources, sources, source.unit().ast(), &builder_, header_suffix,
      /*require_result_types=*/true);
  ASSERT_EQ(signatures.identities.size(), 7u);
  ASSERT_EQ(signatures.header_types.size(), 4u);
  ASSERT_EQ(signatures.result_types.size(), 3u);
  EXPECT_EQ(loom_type_dim_value_id_at(signatures.header_types[2], 0),
            signatures.identities[0]);
  EXPECT_EQ(loom_type_encoding_value_id(signatures.header_types[2]),
            signatures.identities[1]);
  EXPECT_EQ(loom_type_dim_value_id_at(signatures.result_types[2], 0),
            signatures.identities[4]);
  EXPECT_EQ(loom_type_encoding_value_id(signatures.result_types[2]),
            signatures.identities[5]);

  const std::array<loom_value_id_t, 4> initial = {extent, layout, view, active};
  loom_op_t* loop;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder_, initial.data(), initial.size(), signatures.header_types.data(),
      signatures.result_types.data(), 3, nullptr, 0, LOOM_LOCATION_UNKNOWN,
      &loop));
  auto* before = loom_region_entry_block(loom_scf_while_before(loop));
  auto* after = loom_region_entry_block(loom_scf_while_after(loop));
  ASSERT_EQ(before->arg_count, 4u);
  ASSERT_EQ(after->arg_count, 3u);
  for (size_t i = 0; i < before->arg_count; ++i) {
    EXPECT_EQ(before->arg_ids[i], signatures.identities[i]);
  }
  for (size_t i = 0; i < after->arg_count; ++i) {
    EXPECT_EQ(loom_op_results(loop)[i], signatures.identities[i + 4]);
  }
  auto before_view = loom_module_value_type(module_, before->arg_ids[2]);
  EXPECT_EQ(loom_type_dim_value_id_at(before_view, 0), before->arg_ids[0]);
  EXPECT_EQ(loom_type_encoding_value_id(before_view), before->arg_ids[1]);
  auto after_view = loom_module_value_type(module_, after->arg_ids[2]);
  EXPECT_EQ(loom_type_dim_value_id_at(after_view, 0), after->arg_ids[0]);
  EXPECT_EQ(loom_type_encoding_value_id(after_view), after->arg_ids[1]);

  // A post-test loop may first construct its dependent result in the before
  // region, with no incoming handle or shape identities at all.
  auto fresh = bind_condition_loop_signatures(
      types, {}, sources, source.unit().ast(), &builder_, {}, true);
  ASSERT_TRUE(fresh.header_types.empty());
  ASSERT_EQ(fresh.result_types.size(), 3u);
  ASSERT_EQ(fresh.identities.size(), 3u);
  loom_op_t* fresh_loop;
  IREE_ASSERT_OK(loom_scf_while_build(&builder_, nullptr, 0, nullptr,
                                      fresh.result_types.data(), 3, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &fresh_loop));
  auto* fresh_before =
      loom_region_entry_block(loom_scf_while_before(fresh_loop));
  auto* fresh_after = loom_region_entry_block(loom_scf_while_after(fresh_loop));
  EXPECT_EQ(fresh_before->arg_count, 0u);
  ASSERT_EQ(fresh_after->arg_count, 3u);
  auto fresh_view = loom_module_value_type(module_, fresh_after->arg_ids[2]);
  EXPECT_EQ(loom_type_dim_value_id_at(fresh_view, 0), fresh_after->arg_ids[0]);
  EXPECT_EQ(loom_type_encoding_value_id(fresh_view), fresh_after->arg_ids[1]);
}

}  // namespace
}  // namespace loom::cxx_import
