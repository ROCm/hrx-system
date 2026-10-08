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
#include "loom/ops/test/ops.h"

namespace loom::cxx_import {
namespace {

TEST_F(ValueBuilderTest, BindsFunctionArgumentsAndResultsToTheirOwnIdentities) {
  Source source(IREE_SV("namespace loom { namespace encoding { "
                        "enum class role { layout }; } namespace type { "
                        "using size_type = __SIZE_TYPE__; "
                        "inline constexpr size_type dynamic = ~size_type{0}; "
                        "template <loom::encoding::role Role, size_type Rank> "
                        "struct [[loom::type(\"encoding\")]] encoding { "
                        "size_type strides[Rank]; }; "
                        "template <size_type... Extents> "
                        "struct [[loom::type(\"shape\")]] shape {}; "
                        "template <class Shape, class T, "
                        "loom::encoding::role Role = "
                        "loom::encoding::role::layout> "
                        "struct [[loom::type(\"view\")]] view { T* data; }; "
                        "} } using Plane = loom::type::view<"
                        "loom::type::shape<loom::type::dynamic, 32>, "
                        "const float>;"),
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

}  // namespace
}  // namespace loom::cxx_import
