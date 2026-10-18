// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/locations.h"

#include <cxx/ast.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/context.h"

namespace loom::cxx_import {
namespace {

cxx::AST* find_value(cxx::TranslationUnit& unit) {
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(unit.ast());
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    auto* simple = cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration);
    if (simple && simple->initDeclaratorList &&
        cxx::to_string(simple->initDeclaratorList->value->symbol->name()) ==
            "value") {
      return declaration;
    }
  }
  return nullptr;
}

class LocationsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_cxx_import_options_initialize(&options_);
  }
  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  // Dialect context outlives the module.
  loom_context_t context_ = {};
  // Arena pool outlives the module.
  iree_arena_block_pool_t pool_ = {};
  // Owns source identities and locations after the AST expires.
  loom_module_t* module_ = nullptr;
  // Source configuration for each independent translation unit.
  loom_cxx_import_options_t options_ = {};
};

TEST_F(LocationsTest, RetainedRangeAndFilenameOutliveSource) {
  loom_location_id_t location;
  {
    Source source(IREE_SV("int value;\n"), IREE_SV("/app/source.cpp"),
                  options_);
    Locations locations(source.unit(), source.diagnostics(), module_);
    auto* declaration = find_value(source.unit());
    ASSERT_NE(declaration, nullptr);
    location = locations.get(declaration);
    auto count = module_->locations.count;
    EXPECT_EQ(locations.get(declaration), location);
    EXPECT_EQ(module_->locations.count, count);
  }
  ASSERT_NE(location, LOOM_LOCATION_UNKNOWN);
  const auto& entry =
      *loom_location_table_const_entry(&module_->locations, location);
  EXPECT_EQ(entry.kind, LOOM_LOCATION_FILE);
  auto filename = module_->sources.entries[entry.file.source_id];
  EXPECT_EQ(string(filename), "/app/source.cpp");
  EXPECT_EQ(entry.file.start_line, 1);
  EXPECT_EQ(entry.file.start_col, 1);
  EXPECT_EQ(entry.file.end_line, 1);
  EXPECT_EQ(entry.file.end_col, 11);
}

TEST_F(LocationsTest, ObserverReceivesExactHeaderSource) {
  struct Capture {
    // Physical filename of the observed translated source.
    std::string filename;
    // Exact bytes retained for that physical source.
    std::string source;
  } capture;
  options_.source_provider = {
      [](void*, iree_string_view_t path, bool* out_found,
         iree_string_view_t* out_source) {
        *out_found = iree_string_view_equal(path, IREE_SV("/app/value.h"));
        *out_source =
            *out_found ? IREE_SV("int value;\n") : iree_string_view_empty();
        return iree_ok_status();
      },
      nullptr};
  options_.source_observer = {
      [](void* user_data, loom_source_id_t, iree_string_view_t filename,
         iree_string_view_t source) {
        auto& capture = *static_cast<Capture*>(user_data);
        capture.filename = string(filename);
        capture.source = string(source);
        return iree_ok_status();
      },
      &capture};
  Source source(IREE_SV("#include \"value.h\"\n"), IREE_SV("/app/source.cpp"),
                options_);
  Locations locations(source.unit(), source.diagnostics(), module_,
                      options_.source_observer);
  auto* declaration = find_value(source.unit());
  ASSERT_NE(declaration, nullptr);
  EXPECT_NE(locations.get(declaration), LOOM_LOCATION_UNKNOWN);
  EXPECT_EQ(capture.filename, "/app/value.h");
  EXPECT_EQ(capture.source, "int value;\n");
}

TEST_F(LocationsTest, PreservesCoordinatesBeyondCompactRange) {
  std::string contents(LOOM_LOCATION_COMPACT_COORDINATE_MAX, '\n');
  contents.append(LOOM_LOCATION_COMPACT_COORDINATE_MAX, ' ');
  contents += "int value;\n";
  Source source(view(contents), IREE_SV("large.cpp"), options_);
  Locations locations(source.unit(), source.diagnostics(), module_);
  auto* declaration = find_value(source.unit());
  ASSERT_NE(declaration, nullptr);
  const loom_location_id_t location = locations.get(declaration);
  ASSERT_NE(location, LOOM_LOCATION_UNKNOWN);
  EXPECT_FALSE(source.diagnostics().has_error());
  const auto* entry =
      loom_location_table_const_entry(&module_->locations, location);
  ASSERT_EQ(entry->kind, LOOM_LOCATION_FILE);
  EXPECT_EQ(entry->file.start_line, LOOM_LOCATION_COMPACT_COORDINATE_MAX + 1);
  EXPECT_EQ(entry->file.start_col, LOOM_LOCATION_COMPACT_COORDINATE_MAX + 1);
  EXPECT_EQ(entry->file.end_line, LOOM_LOCATION_COMPACT_COORDINATE_MAX + 1);
  EXPECT_EQ(entry->file.end_col, LOOM_LOCATION_COMPACT_COORDINATE_MAX + 11);
}

}  // namespace
}  // namespace loom::cxx_import
