// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/archive.h>
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_visitor.h>
#include <cxx/attributes.h>
#include <cxx/private/semantic_codec.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class LayoutTest : public ::testing::TestWithParam<loom_cxx_data_model_t> {};

TEST_P(LayoutTest, PragmaPackedBitPositionsBelongToTheSourceLayout) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.data_model = GetParam();
  Source source(IREE_SV("#pragma pack(push, 1)\n"
                        "struct Packet { unsigned first:31, second:2; };\n"
                        "#pragma pack(pop)\n"),
                IREE_SV("layout.cpp"), options);
  auto records = source.unit().globalScope()->find("Packet");
  ASSERT_FALSE(records.begin() == records.end());
  auto* record = cxx::symbol_cast<cxx::ClassSymbol>(*records.begin());
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record->sizeInBytes(), 5);
  EXPECT_EQ(record->alignment(), 1);
  auto fields = record->find("second");
  ASSERT_FALSE(fields.begin() == fields.end());
  auto* second = cxx::symbol_cast<cxx::FieldSymbol>(*fields.begin());
  ASSERT_NE(second, nullptr);
  auto position = record->layout()->getFieldInfo(second);
  ASSERT_TRUE(position.has_value());
  EXPECT_EQ(position->offset * 8 + position->bitOffset, 31u);
  EXPECT_EQ(position->bitWidth, 2u);
  EXPECT_EQ(position->allocUnitSizeBytes, 5u);
  EXPECT_EQ(second->localOffset() * 8 + second->bitFieldOffset(), 31);
}

// Inspects the decoded syntax and its ordinary arena clone, independently of
// the field layout facts stored on symbols.
class BitfieldAttributeVisitor final : public cxx::ASTVisitor {
 public:
  void visit(cxx::BitfieldDeclaratorAST* ast) override {
    ++count;
    EXPECT_NE(ast->attributeList, nullptr);
    EXPECT_NE(ast->trailingAttributeList, nullptr);
    EXPECT_GT(ast->lastSourceLocation().index(),
              ast->sizeExpression->lastSourceLocation().index());
  }

  // Number of bitfield declarations reached through normal AST visitation.
  unsigned count = 0;
};

TEST_P(LayoutTest, MemberPackingCrossesTheDeclaredAllocationUnit) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.data_model = GetParam();
  Source source(IREE_SV("struct Bits { unsigned first:31, second:2 "
                        "__attribute__((packed)); };"),
                IREE_SV("bits.cpp"), options);
  auto records = source.unit().globalScope()->find("Bits");
  ASSERT_FALSE(records.begin() == records.end());
  auto* record = cxx::symbol_cast<cxx::ClassSymbol>(*records.begin());
  ASSERT_NE(record, nullptr);
  auto fields = record->find("second");
  ASSERT_FALSE(fields.begin() == fields.end());
  auto* second = cxx::symbol_cast<cxx::FieldSymbol>(*fields.begin());
  ASSERT_NE(second, nullptr);
  EXPECT_TRUE(second->isPacked());
  auto position = record->layout()->getFieldInfo(second);
  ASSERT_TRUE(position.has_value());
  EXPECT_EQ(position->offset * 8 + position->bitOffset, 31u);
  auto first = record->find("first");
  ASSERT_FALSE(first.begin() == first.end());
  EXPECT_FALSE(cxx::symbol_cast<cxx::FieldSymbol>(*first.begin())->isPacked());
}

TEST_P(LayoutTest, LayoutRequestsAndResolvedLayoutSurviveSemanticArchives) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.data_model = GetParam();
  Source source(
      IREE_SV(
          "enum [[gnu::packed]] Byte { byte = 255 };\n"
          "typedef unsigned AlignedWord __attribute__((aligned(32)));\n"
          "struct AliasPacket { unsigned char tag; AlignedWord value; };\n"
          "struct [[gnu::packed, gnu::aligned(64)]] Packet {\n"
          "  unsigned char tag;\n"
          "  alignas(16) unsigned value;\n"
          "  unsigned tail [[gnu::packed]];\n"
          "  unsigned bits [[gnu::packed]] : 3 __attribute__((aligned(2)));\n"
          "};\n"),
      IREE_SV("layout.cpp"), options);
  cxx::ArchiveWriter writer;
  cxx::SemanticArchiveRoots roots;
  roots.globalScope = source.unit().globalScope();
  roots.ast = source.unit().ast();
  cxx::SemanticEncoder encoder(&source.unit());
  ASSERT_TRUE(encoder(roots, writer));
  auto bytes = writer();
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();

  Source destination(IREE_SV(""), IREE_SV("destination.cpp"), options);
  cxx::SemanticArchiveRoots restored;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, restored)) << decoder.error();
  BitfieldAttributeVisitor attributes;
  attributes.accept(restored.ast);
  EXPECT_EQ(attributes.count, 1u);
  auto* clone = restored.ast->clone(destination.unit().arena());
  attributes.accept(clone);
  EXPECT_EQ(attributes.count, 2u);
  auto enumerations = restored.globalScope->find("Byte");
  ASSERT_FALSE(enumerations.begin() == enumerations.end());
  auto* enumeration = cxx::symbol_cast<cxx::EnumSymbol>(*enumerations.begin());
  ASSERT_NE(enumeration, nullptr);
  EXPECT_TRUE(enumeration->isPacked());
  EXPECT_EQ(enumeration->underlyingType()->kind(),
            cxx::TypeKind::kUnsignedChar);
  ASSERT_NE(enumeration->promotionType(), nullptr);
  EXPECT_EQ(enumeration->promotionType()->kind(), cxx::TypeKind::kInt);
  EXPECT_EQ(destination.unit().typeTraits().promoted_integer_type(
                enumeration->type()),
            enumeration->promotionType());
  auto aliases = restored.globalScope->find("AlignedWord");
  ASSERT_FALSE(aliases.begin() == aliases.end());
  auto* alias = cxx::symbol_cast<cxx::TypeAliasSymbol>(*aliases.begin());
  ASSERT_NE(alias, nullptr);
  auto* aligned = cxx::type_cast<cxx::AlignedType>(alias->type());
  ASSERT_NE(aligned, nullptr);
  EXPECT_EQ(aligned->alignment(), 32u);
  EXPECT_NE(cxx::type_cast<cxx::UnsignedIntType>(aligned->elementType()),
            nullptr);
  auto aliasRecords = restored.globalScope->find("AliasPacket");
  ASSERT_FALSE(aliasRecords.begin() == aliasRecords.end());
  auto* aliasRecord = cxx::symbol_cast<cxx::ClassSymbol>(*aliasRecords.begin());
  ASSERT_NE(aliasRecord, nullptr);
  EXPECT_EQ(aliasRecord->sizeInBytes(), 64);
  EXPECT_EQ(aliasRecord->alignment(), 32);
  auto aliasFields = aliasRecord->find("value");
  ASSERT_FALSE(aliasFields.begin() == aliasFields.end());
  auto* aliasValue = cxx::symbol_cast<cxx::FieldSymbol>(*aliasFields.begin());
  ASSERT_NE(aliasValue, nullptr);
  ASSERT_TRUE(aliasRecord->layout()->getFieldInfo(aliasValue).has_value());
  EXPECT_EQ(aliasRecord->layout()->getFieldInfo(aliasValue)->offset, 32u);
  auto records = restored.globalScope->find("Packet");
  ASSERT_FALSE(records.begin() == records.end());
  auto* record = cxx::symbol_cast<cxx::ClassSymbol>(*records.begin());
  ASSERT_NE(record, nullptr);
  EXPECT_NE(cxx::findAttribute(record->attributes(), "packed"), nullptr);
  EXPECT_EQ(record->packAlignment(), 0);
  EXPECT_EQ(record->minimumAlignment(), 64);
  EXPECT_EQ(record->sizeInBytes(), 64);
  EXPECT_EQ(record->alignment(), 64);
  auto fields = record->find("value");
  ASSERT_FALSE(fields.begin() == fields.end());
  auto* value = cxx::symbol_cast<cxx::FieldSymbol>(*fields.begin());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->explicitAlignment(), 16);
  ASSERT_TRUE(record->layout()->getFieldInfo(value).has_value());
  EXPECT_EQ(record->layout()->getFieldInfo(value)->offset, 16u);
  auto tails = record->find("tail");
  ASSERT_FALSE(tails.begin() == tails.end());
  auto* tail = cxx::symbol_cast<cxx::FieldSymbol>(*tails.begin());
  ASSERT_NE(tail, nullptr);
  EXPECT_TRUE(tail->isPacked());
}

// Queries decoded syntax directly so evaluation cannot depend on the source
// invocation or a reconstruction of its member-designator chain.
class OffsetValueVisitor final : public cxx::ASTVisitor {
 public:
  explicit OffsetValueVisitor(cxx::TranslationUnit& unit)
      : interpreter_(&unit) {}

  void visit(cxx::BuiltinOffsetofExpressionAST* ast) override {
    values.push_back(ast->value);
    auto value = interpreter_.evaluate(ast);
    EXPECT_EQ(value.has_value(), ast->value.has_value());
    if (value) {
      EXPECT_EQ(interpreter_.toUInt(*value), ast->value);
    }
  }

  // Resolved or dependent offsets in ordinary syntax visitation order.
  std::vector<std::optional<std::uint64_t>> values;

 private:
  // Evaluator borrowing the decoded source invocation.
  cxx::ASTInterpreter interpreter_;
};

TEST_P(LayoutTest, CompoundOffsetsSurviveSourceDestructionArchivesAndClones) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.data_model = GetParam();
  std::vector<std::uint8_t> bytes;
  {
    Source source(
        IREE_SV(
            "struct [[gnu::packed]] Block { char tag; unsigned payload[3]; };"
            "struct [[gnu::packed]] Packet { char tag; Block blocks[3]; };"
            "auto offset() { return "
            "__builtin_offsetof(Packet, blocks[2].payload[1]); }"
            "struct Address { char bytes[1]; };"
            "auto wide_offset() { return __builtin_offsetof(Address, "
            "bytes[sizeof(void*) == 4 ? 0xffffffffULL : "
            "0x100000005ULL]); }"
            "template<unsigned Index> constexpr auto deferred() { return "
            "__builtin_offsetof(Packet, blocks[Index].payload[1]); }"),
        IREE_SV("offset.cpp"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.globalScope = source.unit().globalScope();
    roots.ast = source.unit().ast();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }
  Source destination(IREE_SV(""), IREE_SV("destination.cpp"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots restored;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, restored)) << decoder.error();
  OffsetValueVisitor offsets(destination.unit());
  std::vector<std::optional<std::uint64_t>> expected = {
      32,
      GetParam() == LOOM_CXX_DATA_MODEL_ILP32
          ? std::numeric_limits<std::uint32_t>::max()
          : UINT64_C(0x100000005),
      std::nullopt};
  offsets.accept(restored.ast);
  EXPECT_EQ(offsets.values, expected);
  offsets.values.clear();
  offsets.accept(restored.ast->clone(destination.unit().arena()));
  EXPECT_EQ(offsets.values, expected);
}

INSTANTIATE_TEST_SUITE_P(DataModels, LayoutTest,
                         ::testing::Values(LOOM_CXX_DATA_MODEL_LP64,
                                           LOOM_CXX_DATA_MODEL_LLP64,
                                           LOOM_CXX_DATA_MODEL_ILP32));

}  // namespace
}  // namespace loom::cxx_import
