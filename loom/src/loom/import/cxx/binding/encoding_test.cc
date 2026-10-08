// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/encoding.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>
#include <cxx/views/symbols.h>

#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/encoding/ops.h"

namespace loom::cxx_import {
namespace {
using EncodingTest = ValueBuilderTest;

TEST_F(EncodingTest, RetainsTheRegisteredFamilyAndItsCanonicalAlias) {
  Source source(
      IREE_SV("template<auto Spec> "
              "[[loom::op(\"encoding.define\", \"encoding.f4e2m1\")]] "
              "int definition();"),
      IREE_SV("encoding.cxx"), options());
  auto functions = cxx::views::each_function(
      *source.unit().globalScope()->find("definition").begin());
  auto* function = *functions.begin();
  const auto& attribute = (*function->attributes())[0];
  auto family =
      EncodingIntrinsic::admit(source.unit(), source.diagnostics(), &context_,
                               function, attribute, source.unit().ast());
  ASSERT_TRUE(family);
  EXPECT_EQ(family->operation, EncodingIntrinsic::Operation::StaticSchema);
  EXPECT_EQ(family->name, "encoding.f4e2m1");
  ASSERT_NE(family->alias, nullptr);
  EXPECT_TRUE(loom_bstring_equal(family->descriptor->name,
                                 IREE_SV("encoding.operand")));
  EXPECT_EQ(family->descriptor->role, LOOM_ENCODING_ROLE_STORAGE_SCHEMA);
  EXPECT_EQ(family->descriptor,
            loom_context_resolve_encoding_vtable(
                &context_, loom_context_resolve_encoding_name(
                               &context_, IREE_SV("encoding.operand"))
                               .family_id)
                ->descriptor);
}

TEST_F(EncodingTest, IdentifiesDynamicPhysicalStorageComposition) {
  Source source(
      IREE_SV("[[loom::op(\"encoding.define\")]] int compose(int, int);"),
      IREE_SV("encoding.cxx"), options());
  auto functions = cxx::views::each_function(
      *source.unit().globalScope()->find("compose").begin());
  auto* function = *functions.begin();
  const auto& attribute = (*function->attributes())[0];
  auto family =
      EncodingIntrinsic::admit(source.unit(), source.diagnostics(), &context_,
                               function, attribute, source.unit().ast());
  ASSERT_TRUE(family);
  EXPECT_EQ(family->operation, EncodingIntrinsic::Operation::PhysicalStorage);
  EXPECT_EQ(family->name, "encoding.storage");
  EXPECT_EQ(family->descriptor, nullptr);
  EXPECT_EQ(family->alias, nullptr);
}

}  // namespace
}  // namespace loom::cxx_import
