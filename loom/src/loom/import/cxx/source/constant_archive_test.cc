// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/archive.h>
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/private/semantic_codec.h>
#include <cxx/symbols.h>
#include <cxx/views/symbols.h>

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

struct SubobjectPathCollector final : cxx::ASTVisitor {
  // Paths retained by resolved member expressions.
  std::vector<std::vector<cxx::Symbol*>> member_paths;
  // Paths retained by resolved derived-to-base conversions.
  std::vector<std::vector<cxx::Symbol*>> cast_paths;

  bool preVisit(cxx::AST* ast) override {
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast);
        member && member->subobjectPath) {
      member_paths.emplace_back(cxx::ListView(member->subobjectPath).begin(),
                                cxx::ListView(member->subobjectPath).end());
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(ast);
        cast && cast->subobjectPath) {
      cast_paths.emplace_back(cxx::ListView(cast->subobjectPath).begin(),
                              cxx::ListView(cast->subobjectPath).end());
    }
    return true;
  }
};

bool hasBasePath(const std::vector<std::vector<cxx::Symbol*>>& paths,
                 std::initializer_list<cxx::Symbol*> targets) {
  for (const auto& path : paths) {
    if (path.size() != targets.size()) {
      continue;
    }
    bool matches = true;
    auto target = targets.begin();
    for (auto* step : path) {
      auto* base = cxx::symbol_cast<cxx::BaseClassSymbol>(step);
      if (!base || base->symbol() != *target++) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return true;
    }
  }
  return false;
}

TEST(ConstantArchiveTest, PreservesComplexAndIndeterminateValues) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  const std::array<cxx::ConstValue, 2> values = {
      std::make_shared<cxx::ConstComplex>(1.25f, -2.5f),
      cxx::IndeterminateValue{},
  };
  for (const auto& value : values) {
    SCOPED_TRACE(value.index());
    std::vector<std::uint8_t> bytes;
    {
      Source source(IREE_SV("int constant;"), IREE_SV("constant.cxx"), options);
      auto symbols = source.unit().globalScope()->find("constant");
      ASSERT_FALSE(symbols.begin() == symbols.end());
      auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(*symbols.begin());
      ASSERT_NE(variable, nullptr);
      if (std::holds_alternative<std::shared_ptr<cxx::ConstComplex>>(value)) {
        variable->setType(source.unit().control()->getComplexType(
            source.unit().control()->getFloatType()));
      }
      variable->setConstValue(value);
      cxx::ArchiveWriter writer;
      cxx::SemanticArchiveRoots roots;
      roots.ast = source.unit().ast();
      roots.globalScope = source.unit().globalScope();
      cxx::SemanticEncoder encoder(&source.unit());
      ASSERT_TRUE(encoder(roots, writer));
      bytes = writer();
    }
    Source destination(IREE_SV(""), IREE_SV("restored.cxx"), options);
    cxx::ArchiveReader reader;
    ASSERT_TRUE(reader(bytes)) << reader.error();
    cxx::SemanticArchiveRoots restored;
    cxx::SemanticDecoder decoder(&destination.unit());
    ASSERT_TRUE(decoder(reader, restored)) << decoder.error();
    auto symbols = restored.globalScope->find("constant");
    ASSERT_FALSE(symbols.begin() == symbols.end());
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(*symbols.begin());
    ASSERT_NE(variable, nullptr);
    ASSERT_TRUE(variable->constValue());
    ASSERT_EQ(variable->constValue()->index(), value.index());
    if (auto* complex = std::get_if<std::shared_ptr<cxx::ConstComplex>>(
            &*variable->constValue())) {
      ASSERT_NE(*complex, nullptr);
      EXPECT_EQ(std::get<float>((*complex)->real()), 1.25f);
      EXPECT_EQ(std::get<float>((*complex)->imag()), -2.5f);
    }
  }
}

TEST(ConstantArchiveTest, PreservesTypedBitsAfterSourceDestruction) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  std::vector<std::uint8_t> bytes;
  {
    Source source(
        IREE_SV("using Halves2 = _Float16 __attribute__((ext_vector_type(2)));"
                "constexpr Halves2 packed = __builtin_bit_cast(Halves2, "
                "0x7c018000u);"
                "constexpr double wide = __builtin_bit_cast(double, "
                "0x7ff0000012345678ULL);"),
        IREE_SV("payloads.cxx"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.ast = source.unit().ast();
    roots.globalScope = source.unit().globalScope();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }
  Source destination(IREE_SV(""), IREE_SV("restored.cxx"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots restored;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, restored)) << decoder.error();
  cxx::ASTInterpreter interpreter(&destination.unit());
  for (const char* name : {"packed", "wide"}) {
    auto symbols = restored.globalScope->find(name);
    ASSERT_FALSE(symbols.begin() == symbols.end());
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(*symbols.begin());
    ASSERT_NE(variable, nullptr);
    ASSERT_TRUE(variable->constValue());
    auto copied = interpreter.cloneValue(*variable->constValue());
    if (auto* floating = std::get_if<cxx::ConstFloat>(&copied)) {
      EXPECT_EQ(floating->format(), cxx::ConstFloat::Format::kDouble);
      EXPECT_EQ(floating->bits(), 0x7ff0000012345678ULL);
    } else {
      auto elements = std::get<std::shared_ptr<cxx::InitializerList>>(copied);
      ASSERT_EQ(elements->elements.size(), 2u);
      auto first =
          std::get<cxx::ConstFloat>(std::get<0>(elements->elements[0]));
      auto second =
          std::get<cxx::ConstFloat>(std::get<0>(elements->elements[1]));
      EXPECT_EQ(first.format(), cxx::ConstFloat::Format::kFloat16);
      EXPECT_EQ(second.format(), cxx::ConstFloat::Format::kFloat16);
      EXPECT_EQ(first.bits(), 0x8000u);
      EXPECT_EQ(second.bits(), 0x7c01u);
    }
  }
}

TEST(ConstantArchiveTest, PreservesReferenceTemplateIdentity) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  std::vector<std::uint8_t> bytes;
  {
    Source source(
        IREE_SV("constexpr unsigned left = 5, right = 5;"
                "template<const unsigned& N> struct Ref {};"
                "using Left = Ref<left>; using Right = Ref<right>;"
                "constexpr const unsigned* left_address = &left;"
                "constexpr const unsigned* right_address = &right;"
                "static_assert(!__is_same(Left, Right));"
                "constexpr unsigned table[2] = {5, 5};"
                "using First = Ref<table[0]>;"
                "using Second = Ref<table[1]>;"
                "constexpr const auto& alias = table;"
                "constexpr const unsigned* first_address = &table[0];"
                "constexpr const unsigned* second_address = &alias[1];"),
        IREE_SV("references.cxx"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.ast = source.unit().ast();
    roots.globalScope = source.unit().globalScope();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }

  Source destination(IREE_SV(""), IREE_SV("restored.cxx"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots restored;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, restored)) << decoder.error();

  auto primary_symbols = restored.globalScope->find("Ref");
  ASSERT_FALSE(primary_symbols.begin() == primary_symbols.end());
  auto* primary = cxx::symbol_cast<cxx::ClassSymbol>(*primary_symbols.begin());
  ASSERT_NE(primary, nullptr);
  ASSERT_EQ(primary->specializations().size(), 4u);

  const std::array<const char*, 4> object_names = {"left", "right", "table",
                                                   "table"};
  const std::array<const char*, 4> address_names = {
      "left_address", "right_address", "first_address", "second_address"};
  const std::array<std::intmax_t, 4> offsets = {0, 0, 0, 1};
  for (size_t index = 0; index < object_names.size(); ++index) {
    SCOPED_TRACE(object_names[index]);
    auto objects = restored.globalScope->find(object_names[index]);
    ASSERT_FALSE(objects.begin() == objects.end());
    auto addresses = restored.globalScope->find(address_names[index]);
    ASSERT_FALSE(addresses.begin() == addresses.end());
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(*addresses.begin());
    ASSERT_NE(variable, nullptr);
    ASSERT_TRUE(variable->constValue());
    auto address =
        std::get<std::shared_ptr<cxx::ConstAddress>>(*variable->constValue());
    ASSERT_NE(address, nullptr);
    EXPECT_EQ(address->symbol(), *objects.begin());
    EXPECT_EQ(address->offset(), offsets[index]);

    // The independent address constant must find the restored specialization,
    // even though it has a different ConstAddress allocation from its key.
    const auto& specialization = primary->specializations()[index];
    const std::vector<cxx::TemplateArgument> arguments = {
        *variable->constValue()};
    EXPECT_EQ(primary->findSpecialization(&destination.unit(), arguments),
              specialization.symbol);
    EXPECT_TRUE(cxx::compare_single_arg(&destination.unit(),
                                        specialization.arguments.front(),
                                        arguments.front()));
    EXPECT_FALSE(cxx::compare_single_arg(
        &destination.unit(),
        primary->specializations()[(index + 1) % object_names.size()]
            .arguments.front(),
        arguments.front()));
  }
}

TEST(ConstantArchiveTest, PreservesRootedSubobjectIdentity) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  std::vector<std::uint8_t> bytes;
  {
    Source source(
        IREE_SV("struct Parameters {"
                "  unsigned scale, bias;"
                "  constexpr const unsigned& get() const { return scale; }"
                "};"
                "constexpr Parameters first{3, 5}, second{3, 5};"
                "extern Parameters external;"
                "template<const unsigned& Value> struct Ref {};"
                "using First = Ref<first.scale>;"
                "using Second = Ref<second.get()>;"
                "using Bias = Ref<first.bias>;"
                "using External = Ref<external.get()>;"
                "static_assert(__is_same(Ref<first.get()>, First));"
                "static_assert(!__is_same(First, Second));"
                "constexpr const unsigned* first_address = &first.get();"
                "constexpr const unsigned* second_address = &second.scale;"
                "constexpr const unsigned* bias_address = &first.bias;"
                "constexpr const unsigned* external_address = "
                "    &external.scale;"),
        IREE_SV("fields.cxx"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.ast = source.unit().ast();
    roots.globalScope = source.unit().globalScope();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }

  Source destination(IREE_SV(""), IREE_SV("restored.cxx"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots restored;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, restored)) << decoder.error();

  auto templates = restored.globalScope->find("Ref");
  ASSERT_FALSE(templates.begin() == templates.end());
  auto* primary = cxx::symbol_cast<cxx::ClassSymbol>(*templates.begin());
  ASSERT_NE(primary, nullptr);
  ASSERT_EQ(primary->specializations().size(), 4u);

  constexpr std::array<const char*, 4> address_names = {
      "first_address", "second_address", "bias_address", "external_address"};
  std::array<cxx::Symbol*, address_names.size()> resolved_specializations = {};
  for (size_t index = 0; index < address_names.size(); ++index) {
    SCOPED_TRACE(address_names[index]);
    auto symbols = restored.globalScope->find(address_names[index]);
    ASSERT_FALSE(symbols.begin() == symbols.end());
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(*symbols.begin());
    ASSERT_NE(variable, nullptr);
    ASSERT_TRUE(variable->constValue());
    auto address =
        std::get<std::shared_ptr<cxx::ConstAddress>>(*variable->constValue());
    ASSERT_NE(address, nullptr);
    ASSERT_NE(address->parent(), nullptr);

    const std::vector<cxx::TemplateArgument> arguments = {
        *variable->constValue()};
    resolved_specializations[index] =
        primary->findSpecialization(&destination.unit(), arguments);
    ASSERT_NE(resolved_specializations[index], nullptr);
    for (size_t previous = 0; previous < index; ++previous) {
      EXPECT_NE(resolved_specializations[index],
                resolved_specializations[previous]);
    }
  }
}

TEST(ConstantArchiveTest, PreservesSelectedSubobjectPaths) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  std::vector<std::uint8_t> bytes;
  {
    Source source(IREE_SV("struct Base { unsigned value; };"
                          "struct Left : Base {}; struct Right : Base {};"
                          "struct Both : Left, Right {};"
                          "extern Both object;"
                          "template<const unsigned&> struct Ref {};"
                          "using Member = Ref<object.Left::value>;"
                          "constexpr const unsigned& select(const Left& left) {"
                          "  return left.value;"
                          "}"
                          "using Converted = "
                          "    Ref<select(static_cast<const Left&>(object))>;"
                          "static_assert(__is_same(Member, Converted));"),
                  IREE_SV("selected_paths.cxx"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.ast = source.unit().ast();
    roots.globalScope = source.unit().globalScope();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }

  Source destination(IREE_SV(""), IREE_SV("restored.cxx"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots restored;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, restored)) << decoder.error();

  auto base_symbols = restored.globalScope->find("Base");
  ASSERT_FALSE(base_symbols.begin() == base_symbols.end());
  auto left_symbols = restored.globalScope->find("Left");
  ASSERT_FALSE(left_symbols.begin() == left_symbols.end());
  auto* base = cxx::symbol_cast<cxx::ClassSymbol>(*base_symbols.begin());
  auto* left = cxx::symbol_cast<cxx::ClassSymbol>(*left_symbols.begin());
  ASSERT_NE(base, nullptr);
  ASSERT_NE(left, nullptr);

  SubobjectPathCollector restored_paths;
  restored_paths.accept(restored.ast);
  EXPECT_TRUE(hasBasePath(restored_paths.member_paths, {left, base}));
  EXPECT_TRUE(hasBasePath(restored_paths.cast_paths, {left}));

  auto* cloned = restored.ast->clone(destination.unit().arena());
  SubobjectPathCollector cloned_paths;
  cloned_paths.accept(cloned);
  EXPECT_TRUE(hasBasePath(cloned_paths.member_paths, {left, base}));
  EXPECT_TRUE(hasBasePath(cloned_paths.cast_paths, {left}));
}

}  // namespace
}  // namespace loom::cxx_import
