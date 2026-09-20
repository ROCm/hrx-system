// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/functions.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"

namespace loom::cxx_import {
namespace {
using FunctionsTest = ValueBuilderTest;

class Calls : public cxx::ASTVisitor {
 public:
  void visit(cxx::CallExpressionAST* call) override {
    auto* id = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
    symbols.push_back(cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol));
    cxx::ASTVisitor::visit(call);
  }
  // Resolved direct calls in the selected source body.
  std::vector<cxx::FunctionSymbol*> symbols;
};

TEST_F(FunctionsTest, RootsQueueEachConcreteHelperOnceWithPrivateVisibility) {
  Source source(IREE_SV(R"(
    template<unsigned N> static int scale(int x) { return x * N; }
    int unused(int x) { return x; }
    int entry(int x) { return scale<3>(x) + scale<3>(x); }
  )"),
                IREE_SV("functions.cpp"), options());
  Types types(source.unit(), source.diagnostics());
  Locations locations(source.unit(), source.diagnostics(), module_);
  Intrinsics intrinsics(source.unit(), source.diagnostics(), types);
  LaunchContracts launches(source.unit(), source.diagnostics());
  Scalars scalars(source.unit(), source.diagnostics(), types, locations,
                  builder_);
  SymbolNames names(source.unit(), source.diagnostics());
  Configs configs(source.unit(), source.diagnostics(), types, scalars,
                  locations, names);
  Functions functions(source.unit(), source.diagnostics(), module_, intrinsics,
                      launches, configs, names);
  const iree_string_view_t roots[] = {IREE_SV("entry")};
  functions.select(roots);
  ASSERT_EQ(functions.pending().size(), 1u);
  auto entry =
      functions.define(functions.pending()[0], types, locations, &builder_);
  EXPECT_EQ(entry.kind, FunctionKind::Ordinary);
  EXPECT_EQ(loom_func_def_visibility(entry.operation),
            LOOM_FUNC_VISIBILITY_PUBLIC);
  EXPECT_EQ(loom_region_entry_arg_count(entry.region), 1u);
  Calls calls;
  calls.accept(entry.body);
  ASSERT_EQ(calls.symbols.size(), 2u);
  auto first = functions.declare(calls.symbols[0]);
  auto second = functions.declare(calls.symbols[1]);
  EXPECT_EQ(first.symbol_id, second.symbol_id);
  ASSERT_EQ(functions.pending().size(), 2u);
  auto helper =
      functions.define(functions.pending()[1], types, locations, &builder_);
  EXPECT_EQ(loom_func_def_visibility(helper.operation), 0);
  auto name_id = module_->symbols.entries[first.symbol_id].name_id;
  auto name = module_->strings.entries[name_id];
  EXPECT_EQ(std::string_view(name.data, name.size), "scale_3");
}

TEST_F(FunctionsTest,
       DefaultSelectionAndKernelDefinitionExposeTheBodyContract) {
  Source source(IREE_SV(R"(
    namespace { int local() { return 1; } }
    static int hidden() { return 2; }
    [[gnu::visibility("hidden")]] int invisible() { return 3; }
    [[loom::kernel, loom::workgroup_size(64, 1, 1)]] void kernel(float* output) {}
    int ordinary() { return 4; }
  )"),
                IREE_SV("visibility.cpp"), options());
  Types types(source.unit(), source.diagnostics());
  Locations locations(source.unit(), source.diagnostics(), module_);
  Intrinsics intrinsics(source.unit(), source.diagnostics(), types);
  LaunchContracts launches(source.unit(), source.diagnostics());
  Scalars scalars(source.unit(), source.diagnostics(), types, locations,
                  builder_);
  SymbolNames names(source.unit(), source.diagnostics());
  Configs configs(source.unit(), source.diagnostics(), types, scalars,
                  locations, names);
  Functions functions(source.unit(), source.diagnostics(), module_, intrinsics,
                      launches, configs, names);
  functions.select({});
  ASSERT_EQ(functions.pending().size(), 2u);
  auto definition =
      functions.define(functions.pending()[0], types, locations, &builder_);
  EXPECT_EQ(definition.kind, FunctionKind::Kernel);
  EXPECT_TRUE(loom_kernel_def_isa(definition.operation));
  EXPECT_EQ(definition.region, loom_kernel_def_body(definition.operation));
  EXPECT_EQ(definition.return_type->kind(), cxx::TypeKind::kVoid);
  EXPECT_EQ(loom_region_entry_arg_count(definition.region), 1u);
  EXPECT_EQ(definition.source->symbol, functions.pending()[0]);
}

}  // namespace
}  // namespace loom::cxx_import
