// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/source.h"

#include <cxx/archive.h>
#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/private/semantic_codec.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/preamble.h"

namespace loom::cxx_import {
namespace {

TEST(SourceTest, LayoutAndMutableSemanticStateBelongToEachSource) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source first(IREE_SV("int value = 1;"), IREE_SV("first.cpp"), options);
  options.data_model = LOOM_CXX_DATA_MODEL_LLP64;
  Source second(IREE_SV("int value = 2;"), IREE_SV("second.cpp"), options);
  options.data_model = LOOM_CXX_DATA_MODEL_ILP32;
  Source third(IREE_SV("int value = 3;"), IREE_SV("third.cpp"), options);
  EXPECT_EQ(first.unit().control()->memoryLayout()->sizeOfLong(), 8);
  EXPECT_EQ(second.unit().control()->memoryLayout()->sizeOfLong(), 4);
  EXPECT_EQ(third.unit().control()->memoryLayout()->sizeOfPointer(), 4);
  EXPECT_EQ(third.unit().control()->memoryLayout()->sizeOfLongLong(), 8);
  EXPECT_NE(first.unit().ast(), second.unit().ast());
  EXPECT_NE(first.unit().globalScope(), second.unit().globalScope());
}

TEST(SourceTest, NarrowFloatIdentitySurvivesSemanticArchive) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  std::vector<uint8_t> bytes;
  {
    Source source(IREE_SV("__bf16 convert(_Float16, __float8_e4m3fn, "
                          "__float8_e5m2);"),
                  IREE_SV("types.cpp"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.globalScope = source.unit().globalScope();
    roots.ast = source.unit().ast();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }
  Source destination(IREE_SV(""), IREE_SV("restored.cpp"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots roots;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, roots)) << decoder.error();
  auto symbols = roots.globalScope->find("convert");
  ASSERT_NE(symbols.begin(), symbols.end());
  auto functions = cxx::views::each_function(*symbols.begin());
  ASSERT_EQ(std::ranges::distance(functions), 1);
  auto* type = cxx::type_cast<cxx::FunctionType>((*functions.begin())->type());
  ASSERT_NE(type, nullptr);
  EXPECT_EQ(type->returnType(),
            destination.unit().control()->getBFloat16Type());
  ASSERT_EQ(type->parameterTypes().size(), 3u);
  EXPECT_EQ(type->parameterTypes()[0],
            destination.unit().control()->getFloat16Type());
  EXPECT_EQ(type->parameterTypes()[1],
            destination.unit().control()->getFloat8E4M3FNType());
  EXPECT_EQ(type->parameterTypes()[2],
            destination.unit().control()->getFloat8E5M2Type());
}

TEST(SourceTest, VectorConversionSurvivesSemanticArchiveAndClone) {
  class Conversions final : public cxx::ASTVisitor {
   public:
    void visit(cxx::BuiltinConvertVectorExpressionAST* ast) override {
      expressions.push_back(ast);
      cxx::ASTVisitor::visit(ast);
    }

    // Conversion expressions borrowed from the visited source arena.
    std::vector<cxx::BuiltinConvertVectorExpressionAST*> expressions;
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  std::vector<uint8_t> bytes;
  {
    Source source(
        IREE_SV(
            "using Bytes = unsigned char __attribute__((ext_vector_type(4)));"
            "using Floats = float __attribute__((ext_vector_type(4)));"
            "Floats convert(Bytes value) {"
            "return __builtin_convertvector(value, Floats); }"),
        IREE_SV("vectors.cpp"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.globalScope = source.unit().globalScope();
    roots.ast = source.unit().ast();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }
  Source destination(IREE_SV(""), IREE_SV("restored.cpp"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots roots;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, roots)) << decoder.error();
  for (auto* ast : {roots.ast, roots.ast->clone(destination.unit().arena())}) {
    Conversions conversions;
    conversions.accept(ast);
    ASSERT_EQ(conversions.expressions.size(), 1u);
    auto* conversion = conversions.expressions.front();
    ASSERT_NE(conversion->expression, nullptr);
    ASSERT_NE(conversion->typeId, nullptr);
    EXPECT_TRUE(conversion->convertVectorLoc);
    EXPECT_TRUE(conversion->rparenLoc);
    EXPECT_EQ(conversion->valueCategory, cxx::ValueCategory::kPrValue);
    EXPECT_EQ(conversion->typeId->type, conversion->type);
    auto* input =
        cxx::unqualified_cast<cxx::VectorType>(conversion->expression->type);
    auto* output = cxx::unqualified_cast<cxx::VectorType>(conversion->type);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(input->elementCount(), 4);
    EXPECT_EQ(output->elementCount(), 4);
    EXPECT_EQ(input->elementType(),
              destination.unit().control()->getUnsignedCharType());
    EXPECT_EQ(output->elementType(),
              destination.unit().control()->getFloatType());
  }
}

TEST(SourceTest, ProviderBytesAreCopiedBeforeTheNextCallback) {
  struct Store {
    // Mutable provider buffer invalidated by every subsequent lookup.
    std::string scratch;
    // Candidate request counts, including missing paths.
    std::map<std::string, unsigned> requests;
  } store;
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.source_provider = {
      [](void* user_data, iree_string_view_t path, bool* out_found,
         iree_string_view_t* out_source) {
        auto& store = *static_cast<Store*>(user_data);
        std::string filename(path.data, path.size);
        ++store.requests[filename];
        store.scratch.assign(256, '?');
        if (filename == "/include/first.h") {
          store.scratch =
              "#pragma once\n#include <second.h>\n"
              "constexpr int first = second + 1;\n";
        } else if (filename == "/include/second.h") {
          store.scratch = "#pragma once\nconstexpr int second = 41;\n";
        } else {
          *out_found = false;
          *out_source = iree_string_view_empty();
          return iree_ok_status();
        }
        *out_found = true;
        *out_source = view(store.scratch);
        return iree_ok_status();
      },
      &store};
  iree_string_view_t directory = IREE_SV("/include");
  options.include_paths = &directory;
  options.include_path_count = 1;
  Source source(
      IREE_SV("#include <first.h>\n#include <first.h>\n"
              "#if __has_include(<missing.h>)\n#error missing\n#endif\n"
              "#if __has_include(<missing.h>)\n#error missing\n#endif\n"
              "static_assert(first == 42);\n"),
      IREE_SV("/app/source.cpp"), options);
  store.scratch.clear();
  ASSERT_NE(source.unit().ast(), nullptr);
  EXPECT_EQ(store.requests.at("/include/first.h"), 1u);
  EXPECT_EQ(store.requests.at("/include/second.h"), 1u);
  EXPECT_EQ(store.requests.at("/include/missing.h"), 1u);
  EXPECT_EQ(source.contents().source("/include/first.h"),
            "#pragma once\n#include <second.h>\n"
            "constexpr int first = second + 1;\n");
  EXPECT_EQ(source.contents().source("/include/second.h"),
            "#pragma once\nconstexpr int second = 41;\n");
}

TEST(SourceTest, PhysicalRangesMapUtf8ColumnsToExactBytes) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  const std::string text =
      "/* \xcf\x80 */ constexpr auto fragment = R\"asm(first\nsecond)asm\";\n";
  Source source(view(text), IREE_SV("assembly.cpp"), options);
  for (const auto& token : source.unit().tokens()) {
    if (token.kind() != cxx::TokenKind::T_STRING_LITERAL) {
      continue;
    }
    auto first = source.unit().preprocessor()->tokenStartPosition(token);
    auto last = source.unit().preprocessor()->tokenEndPosition(token);
    auto physical = source.contents().range(first, last);
    EXPECT_EQ(
        physical.source.substr(physical.start, physical.end - physical.start),
        "R\"asm(first\nsecond)asm\"");
    return;
  }
  FAIL() << "Expected one raw string literal";
}

TEST(SourceTest, RestoredPreamblePreservesConsumerAndHeaderPhysicalSource) {
  static constexpr std::string_view kHeader =
      "constexpr auto assembly_text = R\"loom(first\nsecond)loom\";\n";
  static constexpr std::string_view kSource =
      "#include \"literal.h\"\n"
      "constexpr auto consumer_file = __FILE__;\n";
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.source_provider = {
      [](void*, iree_string_view_t path, bool* out_found,
         iree_string_view_t* out_source) {
        *out_found = iree_string_view_equal(path, IREE_SV("/app/literal.h"));
        *out_source = *out_found ? view(kHeader) : iree_string_view_empty();
        return iree_ok_status();
      },
      nullptr};
  loom_cxx_preamble_t* raw_preamble = nullptr;
  IREE_ASSERT_OK(loom_cxx_preamble_create(
      view(kSource), IREE_SV("/app/producer.cpp"), &options,
      iree_allocator_system(), &raw_preamble));
  std::unique_ptr<loom_cxx_preamble_t, decltype(&loom_cxx_preamble_release)>
      preamble(raw_preamble, loom_cxx_preamble_release);
  ASSERT_NE(preamble, nullptr);

  options.preamble = preamble.get();
  Source source(view(kSource), IREE_SV("/app/source.cpp"), options);
  bool found_consumer_file = false;
  for (const auto& token : source.unit().tokens()) {
    if (token.kind() == cxx::TokenKind::T_STRING_LITERAL &&
        token.spell() == "\"/app/source.cpp\"") {
      found_consumer_file = true;
    }
  }
  EXPECT_TRUE(found_consumer_file);

  class Literals final : public cxx::ASTVisitor {
   public:
    void visit(cxx::StringLiteralExpressionAST* ast) override {
      literals.push_back(ast);
      cxx::ASTVisitor::visit(ast);
    }

    // Literal expressions borrowed from the visited translation unit.
    std::vector<cxx::StringLiteralExpressionAST*> literals;
  } literals;
  literals.accept(source.unit().ast());
  auto found = std::ranges::find_if(literals.literals, [](const auto* literal) {
    return literal->literal->value().starts_with("R\"loom(");
  });
  ASSERT_NE(found, literals.literals.end());
  const auto location = (*found)->firstSourceLocation();
  EXPECT_FALSE(source.unit().ownsLocation(location));
  const auto first = source.unit().tokenStartPosition(location);
  const auto last = source.unit().tokenEndPosition(location);
  const auto physical = source.contents().range(first, last);
  EXPECT_EQ(first.fileName, "/app/literal.h");
  EXPECT_EQ(physical.source, kHeader);
  EXPECT_EQ(
      physical.source.substr(physical.start, physical.end - physical.start),
      "R\"loom(first\nsecond)loom\"");
}

TEST(SourceTest, RejectedSourceLeavesTheNextInvocationIndependent) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  EXPECT_THROW(Source(IREE_SV("int invalid = ;"), IREE_SV("bad.cpp"), options),
               SourceRejected);
  Source source(IREE_SV("int valid = 7;"), IREE_SV("good.cpp"), options);
  EXPECT_FALSE(source.diagnostics().has_error());
  ASSERT_NE(source.unit().ast(), nullptr);
}

TEST(SourceTest, IncludeDirectoryRetainsFilesystemRoot) {
  const auto root = std::filesystem::current_path().root_path();
  const auto directory = root.string();
  const auto include_path = view(directory);
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.include_paths = &include_path;
  options.include_path_count = 1;
  Source source(IREE_SV("int value = 1;"), IREE_SV("source.cpp"), options);
  const auto& paths = source.unit().preprocessor()->userIncludePaths();
  ASSERT_EQ(paths.size(), 1u);
  EXPECT_EQ(paths.front(), root.generic_string());
}

TEST(SourceTest, BuiltinStringSpellingRetainsThePhysicalExpansionRange) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const char* macro : {"__FILE__", "__DATE__", "__TIME__"}) {
    SCOPED_TRACE(macro);
    const std::string contents =
        "\n#line 700 \"presumed.cc\"\nconstexpr auto value = " +
        std::string(macro) + ";\n";
    Source source(view(contents), IREE_SV("physical.cpp"), options);
    auto* preprocessor = source.unit().preprocessor();
    unsigned literal_count = 0;
    for (const auto& token : source.unit().tokens()) {
      if (token.fileId() != preprocessor->mainSourceFileId() ||
          token.kind() != cxx::TokenKind::T_STRING_LITERAL) {
        continue;
      }
      ++literal_count;
      EXPECT_EQ(token.offset(), contents.find(macro));
      EXPECT_EQ(token.length(), std::string_view(macro).size());
      auto first = preprocessor->tokenStartPosition(token);
      auto last = preprocessor->tokenEndPosition(token);
      EXPECT_EQ(first.fileName, "physical.cpp");
      EXPECT_EQ(first.line, 3u);
      EXPECT_EQ(first.column, 24u);
      EXPECT_EQ(last.fileName, first.fileName);
      EXPECT_EQ(last.line, first.line);
      EXPECT_EQ(last.column, first.column + token.length());
      auto presumed = preprocessor->presumedTokenStartPosition(token);
      EXPECT_EQ(presumed.fileName, "presumed.cc");
      EXPECT_EQ(presumed.line, 700u);
      const auto spelling = token.spell();
      if (std::string_view(macro) == "__FILE__") {
        EXPECT_EQ(spelling, "\"presumed.cc\"");
      } else {
        // Date and time vary, but their spelling remains a quoted literal.
        EXPECT_EQ(spelling.size(),
                  std::string_view(macro) == "__DATE__" ? 13u : 10u);
        EXPECT_EQ(spelling.front(), '"');
        EXPECT_EQ(spelling.back(), '"');
      }
    }
    EXPECT_EQ(literal_count, 1u);
  }
}

TEST(SourceTest, StringizedTokensRetainInvocationRangesAndSeparateSpelling) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("#define QUOTE(value) #value\n\n"
                        "constexpr auto a = QUOTE(ab);\n"
                        "constexpr auto b = QUOTE();\n"),
                IREE_SV("operators.cpp"), options);
  auto* preprocessor = source.unit().preprocessor();
  unsigned literal_count = 0;
  for (const auto& token : source.unit().tokens()) {
    if (token.fileId() != preprocessor->mainSourceFileId() ||
        token.kind() != cxx::TokenKind::T_STRING_LITERAL) {
      continue;
    }
    auto first = preprocessor->tokenStartPosition(token);
    auto last = preprocessor->tokenEndPosition(token);
    EXPECT_EQ(first.fileName, "operators.cpp");
    EXPECT_EQ(first.line, 3u + literal_count);
    EXPECT_EQ(first.column, 20u);
    EXPECT_EQ(last.line, first.line);
    EXPECT_EQ(last.column, 25u);
    EXPECT_EQ(token.length(), 5u);
    EXPECT_EQ(token.spell(), literal_count == 0 ? "\"ab\"" : "\"\"");
    ++literal_count;
  }
  EXPECT_EQ(literal_count, 2u);
}

TEST(SourceTest, ExplicitXvalueRetainsItsTypeIdAndObjectIdentity) {
  class Casts final : public cxx::ASTVisitor {
   public:
    void visit(cxx::CppCastExpressionAST* ast) override {
      expressions.push_back(ast);
      cxx::ASTVisitor::visit(ast);
    }

    std::vector<cxx::CppCastExpressionAST*> expressions;
  } casts;
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("struct Value { unsigned member; };"
                        "unsigned read(Value value) {"
                        "Value moved = static_cast<Value&&>((value));"
                        "return static_cast<unsigned>(moved.member); }"),
                IREE_SV("xvalue.cpp"), options);
  casts.accept(source.unit().ast());
  ASSERT_EQ(casts.expressions.size(), 2u);
  auto* xvalue = casts.expressions[0];
  ASSERT_NE(xvalue->typeId, nullptr);
  EXPECT_EQ(xvalue->valueCategory, cxx::ValueCategory::kXValue);
  EXPECT_TRUE(
      source.unit().typeTraits().is_rvalue_reference(xvalue->typeId->type));
  EXPECT_EQ(explicit_xvalue_operand(source.unit(), xvalue), xvalue->expression);
  EXPECT_TRUE(cxx::ast_cast<cxx::IdExpressionAST>(
      unwrap_object_identity(source.unit(), xvalue)));
  EXPECT_EQ(explicit_xvalue_operand(source.unit(), casts.expressions[1]),
            nullptr);
}

TEST(SourceTest, DiagnosticSinkFailureCrossesTheParserSafeBoundary) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.diagnostic_sink = {[](void*, const loom_diagnostic_t*) {
                               return iree_make_status(IREE_STATUS_CANCELLED,
                                                       "consumer stopped");
                             },
                             nullptr};
  try {
    Source source(IREE_SV("int invalid = ;"), IREE_SV("bad.cpp"), options);
    FAIL() << "Expected the retained diagnostic sink failure";
  } catch (StatusError& error) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, error.release());
  }
}

TEST(SourceTest, CopiedProviderFailureRetainsOriginalStatus) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  uintptr_t provided_status_identity = 0;
  options.source_provider = {
      [](void* user_data, iree_string_view_t, bool*, iree_string_view_t*) {
        auto status =
            iree_make_status(IREE_STATUS_UNAVAILABLE, "provider failed");
        // Retain only the identity; the source boundary owns the status.
        *static_cast<uintptr_t*>(user_data) =
            reinterpret_cast<uintptr_t>(status);
        return status;
      },
      &provided_status_identity};
  std::optional<StatusError> retained_error;
  try {
    Source source(IREE_SV("#include \"missing.h\"\n"),
                  IREE_SV("/app/source.cpp"), options);
    FAIL() << "Expected the provider failure";
  } catch (const StatusError& error) {
    retained_error.emplace(error);
  }
  ASSERT_TRUE(retained_error.has_value());
  auto status = retained_error->release();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(status), provided_status_identity);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
}

}  // namespace
}  // namespace loom::cxx_import
