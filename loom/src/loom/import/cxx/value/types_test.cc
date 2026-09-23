// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/types.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

TEST(TypesTest, ProjectsTheConfiguredDataModelAndRetainsSignedness) {
  for (auto model : {LOOM_CXX_DATA_MODEL_LP64, LOOM_CXX_DATA_MODEL_LLP64,
                     LOOM_CXX_DATA_MODEL_ILP32}) {
    loom_cxx_import_options_t options;
    loom_cxx_import_options_initialize(&options);
    options.data_model = model;
    Source source(IREE_SV("int entry();"), IREE_SV("types.cpp"), options);
    Types types(source.unit(), source.diagnostics());
    auto* control = source.unit().control();
    auto* owner = source.unit().ast();
    EXPECT_EQ(
        loom_type_element_type(types.get(control->getLongIntType(), owner)),
        model == LOOM_CXX_DATA_MODEL_LP64 ? LOOM_SCALAR_TYPE_I64
                                          : LOOM_SCALAR_TYPE_I32);
    EXPECT_TRUE(
        loom_type_equal(types.get(control->getIntType(), owner),
                        types.get(control->getUnsignedIntType(), owner)));
    EXPECT_FALSE(types.is_unsigned(control->getIntType()));
    EXPECT_TRUE(types.is_unsigned(control->getUnsignedIntType()));
    EXPECT_EQ(loom_type_kind(types.get(
                  control->getPointerType(control->getFloatType()), owner)),
              LOOM_TYPE_BUFFER);
    EXPECT_EQ(
        loom_type_element_type(types.get(control->getBFloat16Type(), owner)),
        LOOM_SCALAR_TYPE_BF16);
    EXPECT_EQ(types.storage_size(control->getBFloat16Type(), owner), 2);
    EXPECT_TRUE(types.is_float(control->getBFloat16Type()));
    EXPECT_EQ(loom_type_element_type(
                  types.get(control->getFloat8E4M3FNType(), owner)),
              LOOM_SCALAR_TYPE_F8E4M3);
    EXPECT_EQ(
        loom_type_element_type(types.get(control->getFloat8E5M2Type(), owner)),
        LOOM_SCALAR_TYPE_F8E5M2);
    EXPECT_EQ(types.storage_size(control->getFloat8E4M3FNType(), owner), 1);
    EXPECT_EQ(types.storage_size(control->getFloat8E5M2Type(), owner), 1);
    EXPECT_TRUE(types.is_float(control->getFloat8E4M3FNType()));
    EXPECT_TRUE(types.is_float(control->getFloat8E5M2Type()));
  }
}

TEST(TypesTest, ObjectPointersDoNotRequirePointeeStorage) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("struct Opaque; union Payload { int a; float b; }; "
                        "struct Owned { ~Owned() {} }; "
                        "using Callback = void (*)();"),
                IREE_SV("opaque.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* control = source.unit().control();
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  const cxx::Type* pointees[] = {
      control->getVoidType(),
      source.unit().typeTraits().add_const(control->getVoidType()),
      source_type("Opaque"),
      source_type("Payload"),
      source_type("Owned"),
      control->getBoolType(),
      control->getPointerType(control->getIntType()),
  };
  for (auto* pointee : pointees) {
    auto* pointer = control->getPointerType(pointee);
    EXPECT_EQ(loom_type_kind(types.get(pointer, owner)), LOOM_TYPE_BUFFER);
    EXPECT_EQ(types.partition(pointer, owner).kind, ValueKind::Pointer);
    std::vector<loom_type_t> components;
    types.append(pointer, owner, components);
    ASSERT_EQ(components.size(), 2u);
    EXPECT_EQ(loom_type_kind(components[0]), LOOM_TYPE_BUFFER);
    EXPECT_EQ(loom_type_element_type(components[1]), LOOM_SCALAR_TYPE_OFFSET);
    EXPECT_THROW(types.storage_size(pointee, owner), SourceRejected);
  }
  EXPECT_THROW(types.get(source_type("Callback"), owner), SourceRejected);
}

TEST(TypesTest, RecordPartitionsRetainNominalMembersAndStaticTransport) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("struct Empty {}; struct Pair { unsigned a, b; }; "
                        "struct Other { unsigned a, b; }; "
                        "struct Packet { Empty empty; Pair pair; "
                        "const unsigned* pointer; const bool valid; };"),
                IREE_SV("records.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  const auto* packet = types.record(source_type("Packet"), owner);
  ASSERT_NE(packet, nullptr);
  ASSERT_EQ(packet->members.size(), 4u);
  EXPECT_EQ(packet->component_count, 5u);
  EXPECT_EQ(packet->members[0].partition->component_count, 0u);
  EXPECT_EQ(packet->members[1].component_offset, 0u);
  EXPECT_EQ(packet->members[2].component_offset, 2u);
  EXPECT_EQ(packet->members[3].component_offset, 4u);
  EXPECT_EQ(types.member(packet->members[2].field, owner).component_offset, 2u);
  EXPECT_EQ(packet->component_names,
            (std::vector<std::string>{"pair_a", "pair_b", "pointer",
                                      "pointer_byte_offset", "valid"}));
  EXPECT_NE(packet->members[1].partition,
            &types.partition(source_type("Other"), owner));
  EXPECT_EQ(types.partition(source_type("Pair"), owner).kind,
            ValueKind::Record);
  EXPECT_EQ(packet->members[2].partition->kind, ValueKind::Pointer);
  EXPECT_TRUE(
      source.unit().typeTraits().is_const(packet->members[3].field->type()));
  EXPECT_EQ(types.record(source_type("Empty"), owner)->source->sizeInBytes(),
            1);
  std::vector<loom_type_t> signature;
  types.append(source_type("Packet"), owner, signature);
  ASSERT_EQ(signature.size(), 5u);
  EXPECT_EQ(loom_type_element_type(signature[0]), LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(loom_type_kind(signature[2]), LOOM_TYPE_BUFFER);
  EXPECT_EQ(loom_type_element_type(signature[3]), LOOM_SCALAR_TYPE_OFFSET);
  EXPECT_EQ(loom_type_element_type(signature[4]), LOOM_SCALAR_TYPE_I1);
}

TEST(TypesTest, RecordMemoryUsesSourceLayoutIndependentlyOfValuePartitions) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Padded { unsigned char tag; float value; unsigned short count; };
    struct Nested { unsigned prefix; Padded payload; };
    struct Observed { volatile unsigned value; };
    struct [[gnu::packed]] Packed { unsigned char tag; float value; };
    struct FieldPacked { unsigned char tag; unsigned value [[gnu::packed]]; };
    #pragma pack(push, 2)
    struct Capped { unsigned char tag; double value; };
    #pragma pack(pop)
    struct Transport { unsigned* pointer; bool valid; };
    template<class T> struct Box { unsigned char tag; T value; };
    using Concrete = Box<unsigned>;
  )"),
                IREE_SV("record_memory.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto* control = source.unit().control();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  auto* padded = source_type("Padded");
  EXPECT_EQ(types.storage_size(padded, owner), 12);
  EXPECT_EQ(types.storage_size(source_type("Nested"), owner), 16);
  EXPECT_EQ(types.storage_size(source_type("Concrete"), owner), 8);
  EXPECT_EQ(types.storage_size(source_type("Packed"), owner), 5);
  EXPECT_EQ(types.storage_size(source_type("FieldPacked"), owner), 5);
  EXPECT_EQ(types.storage_size(source_type("Capped"), owner), 10);
  EXPECT_EQ(loom_type_kind(types.get(control->getPointerType(padded), owner)),
            LOOM_TYPE_BUFFER);
  auto* partition = types.record(padded, owner);
  ASSERT_NE(partition, nullptr);
  ASSERT_EQ(partition->members.size(), 3u);
  EXPECT_EQ(partition->members[1].field->offsetInClass(), 4u);
  EXPECT_EQ(partition->members[1].component_offset, 1u);
  EXPECT_EQ(types.storage_size(padded, owner), 12);
  auto* observed = source_type("Observed");
  EXPECT_EQ(types.storage_size(observed, owner), 4);
  EXPECT_EQ(types.partition(control->getPointerType(observed), owner).kind,
            ValueKind::Pointer);
  EXPECT_THROW(types.partition(observed, owner), SourceRejected);
  auto* transport = source_type("Transport");
  EXPECT_EQ(types.partition(transport, owner).component_count, 3u);
  EXPECT_THROW(types.storage_size(transport, owner), SourceRejected);
}

TEST(TypesTest, FixedArrayStorageRetainsNestedSourceLayout) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    using Float4 = float __attribute__((ext_vector_type(4)));
    struct Block {
      _Float16 scale;
      unsigned short scales_high;
      unsigned char scales_low[4];
      unsigned char quants[128];
    };
    struct Records { unsigned prefix; Block blocks[3]; };
    struct Vectors { unsigned prefix; Float4 values[2][3]; };
  )"),
                IREE_SV("array_storage.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto* control = source.unit().control();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  auto* block = source_type("Block");
  EXPECT_EQ(types.storage_size(block, owner), 136);
  EXPECT_EQ(types.storage_size(source_type("Records"), owner), 412);
  EXPECT_EQ(types.storage_size(source_type("Vectors"), owner), 112);
  auto* array = control->getBoundedArrayType(block, 3);
  EXPECT_EQ(types.storage_size(array, owner), 408);
  EXPECT_EQ(types.partition(control->getPointerType(array), owner).kind,
            ValueKind::Pointer);
  EXPECT_THROW(types.partition(block, owner), SourceRejected);
  EXPECT_THROW(types.storage_size(control->getUnboundedArrayType(block), owner),
               SourceRejected);
}

TEST(TypesTest, ViewPartitionsBindEachDestinationShapeAndLayoutIdentity) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("namespace loom { namespace encoding { "
                        "enum class role { layout }; } namespace type { "
                        "using size_type = __SIZE_TYPE__; "
                        "inline constexpr size_type dynamic = ~size_type{0}; "
                        "template <loom::encoding::role Role, size_type Rank> "
                        "struct [[loom::type(\"encoding\")]] encoding { "
                        "size_type strides[Rank]; }; "
                        "template <class T, size_type Rows = dynamic, "
                        "size_type Columns = dynamic> "
                        "struct [[loom::type(\"view\")]] view { T* data; "
                        "size_type shape[2]; "
                        "encoding<loom::encoding::role::layout, 2> layout; }; "
                        "} } "
                        "using Layout = loom::type::encoding<"
                        "loom::encoding::role::layout, 2>; "
                        "using Plane = loom::type::view<const float, "
                        "loom::type::dynamic, 32>; "
                        "using Observed = loom::type::view<const volatile "
                        "float, loom::type::dynamic, 32>; "
                        "struct Packet { Plane plane; unsigned tag; };"),
                IREE_SV("views.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  const auto& layout = types.partition(source_type("Layout"), owner);
  EXPECT_EQ(layout.kind, ValueKind::Encoding);
  EXPECT_EQ(layout.component_count, 1u);
  EXPECT_FALSE(types.requires_binding(source_type("Layout"), owner));
  std::vector<loom_type_t> layout_types;
  types.append(source_type("Layout"), owner, layout_types);
  ASSERT_EQ(layout_types.size(), 1u);
  EXPECT_EQ(loom_type_kind(layout_types[0]), LOOM_TYPE_ENCODING);
  EXPECT_EQ(loom_type_encoding_role(layout_types[0]),
            LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);

  const auto& plane = types.partition(source_type("Plane"), owner);
  EXPECT_EQ(plane.kind, ValueKind::View);
  EXPECT_EQ(plane.component_count, 3u);
  EXPECT_TRUE(types.requires_binding(source_type("Plane"), owner));
  EXPECT_EQ(static_cast<const ViewPartition&>(plane).access_flags, 0);
  const auto& observed = types.partition(source_type("Observed"), owner);
  EXPECT_EQ(static_cast<const ViewPartition&>(observed).access_flags,
            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  const loom_value_id_t plane_ids[] = {11, 12, 13};
  std::vector<loom_type_t> plane_types;
  types.append_bound(source_type("Plane"), owner, plane_ids, plane_types);
  ASSERT_EQ(plane_types.size(), 3u);
  EXPECT_EQ(loom_type_element_type(plane_types[0]), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_encoding_role(plane_types[1]),
            LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_EQ(loom_type_kind(plane_types[2]), LOOM_TYPE_VIEW);
  EXPECT_EQ(loom_type_rank(plane_types[2]), 2u);
  EXPECT_EQ(loom_type_element_type(plane_types[2]), LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(loom_type_dim_value_id_at(plane_types[2], 0), 11u);
  EXPECT_EQ(loom_type_dim_static_size_at(plane_types[2], 1), 32);
  EXPECT_TRUE(loom_type_has_ssa_encoding(plane_types[2]));
  EXPECT_EQ(loom_type_encoding_value_id(plane_types[2]), 12u);
  std::vector<loom_type_t> unbound;
  EXPECT_THROW(types.append(source_type("Plane"), owner, unbound),
               SourceRejected);

  const auto* packet = types.record(source_type("Packet"), owner);
  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(packet->component_count, 4u);
  EXPECT_EQ(
      packet->component_names,
      (std::vector<std::string>{"plane_rows", "plane_layout", "plane", "tag"}));
  EXPECT_TRUE(types.requires_binding(source_type("Packet"), owner));
  const loom_value_id_t packet_ids[] = {21, 22, 23, 24};
  std::vector<loom_type_t> packet_types;
  types.append_bound(source_type("Packet"), owner, packet_ids, packet_types);
  ASSERT_EQ(packet_types.size(), 4u);
  EXPECT_EQ(loom_type_dim_value_id_at(packet_types[2], 0), 21u);
  EXPECT_EQ(loom_type_encoding_value_id(packet_types[2]), 22u);
  EXPECT_EQ(loom_type_element_type(packet_types[3]), LOOM_SCALAR_TYPE_I32);
}

TEST(TypesTest, RejectsRepresentationsThatLoseSourceSemantics) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("int entry();"), IREE_SV("types.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* control = source.unit().control();
  auto* owner = source.unit().ast();
  EXPECT_THROW(types.storage_size(control->getBoolType(), owner),
               SourceRejected);
  EXPECT_THROW(
      types.get(control->getLvalueReferenceType(control->getIntType()), owner),
      SourceRejected);
  EXPECT_THROW(types.get(control->getLongDoubleType(), owner), SourceRejected);
}

TEST(TypesTest, SeparatesLoadedRepresentationFromVolatileObjectAdmission) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("int entry();"), IREE_SV("types.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* control = source.unit().control();
  auto* owner = source.unit().ast();
  auto* integer = control->getUnsignedIntType();
  auto* observed = control->getQualType(integer, cxx::CvQualifiers::kVolatile);
  auto* pointer = control->getPointerType(observed);
  EXPECT_TRUE(
      loom_type_equal(types.get(observed, owner), types.get(integer, owner)));
  EXPECT_EQ(types.memory_access_flags(integer), 0);
  EXPECT_EQ(types.memory_access_flags(pointer), 0);
  EXPECT_EQ(types.memory_access_flags(observed),
            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  EXPECT_EQ(types.partition(pointer, owner).kind, ValueKind::Pointer);
  EXPECT_THROW(types.partition(observed, owner), SourceRejected);
  EXPECT_THROW(
      types.partition(
          control->getQualType(pointer, cxx::CvQualifiers::kVolatile), owner),
      SourceRejected);
}

TEST(TypesTest, EnumProjectionPreservesSourceWidthSignednessAndIdentity) {
  struct Case {
    // Complete source definition used to establish the enum representation.
    const char* source;
    // Scalar storage after projection into Loom.
    loom_scalar_type_t element;
    // Source interpretation of the projected integer bits.
    bool is_unsigned;
  };
  for (auto test : {
           Case{"enum class Value : unsigned char { high = 255 };",
                LOOM_SCALAR_TYPE_I8, true},
           Case{"enum class Value : signed char { low = -128 };",
                LOOM_SCALAR_TYPE_I8, false},
           Case{"enum class Value : unsigned short { high = 65535 };",
                LOOM_SCALAR_TYPE_I16, true},
           Case{"enum Value { negative = -1, positive = 2 };",
                LOOM_SCALAR_TYPE_I32, false},
           Case{"enum Value { high = 0x80000000u };", LOOM_SCALAR_TYPE_I32,
                true},
           Case{"enum Value { high = 1ULL << 40 };", LOOM_SCALAR_TYPE_I64,
                false},
           Case{"enum Value { high = 0xffffffffffffffffULL };",
                LOOM_SCALAR_TYPE_I64, true},
       }) {
    SCOPED_TRACE(test.source);
    loom_cxx_import_options_t options;
    loom_cxx_import_options_initialize(&options);
    Source source(iree_make_cstring_view(test.source), IREE_SV("enum.cpp"),
                  options);
    auto symbols = source.unit().globalScope()->find("Value");
    ASSERT_FALSE(symbols.begin() == symbols.end());
    auto* type = (*symbols.begin())->type();
    ASSERT_TRUE(source.unit().typeTraits().is_enum(type));
    Types types(source.unit(), source.diagnostics());
    auto* owner = source.unit().ast();
    EXPECT_EQ(loom_type_element_type(types.get(type, owner)), test.element);
    EXPECT_EQ(types.is_unsigned(type), test.is_unsigned);
    EXPECT_EQ(types.unqualified(type), type);
    auto* control = source.unit().control();
    auto* constant = control->getQualType(type, cxx::CvQualifiers::kConst);
    EXPECT_EQ(types.is_unsigned(constant), test.is_unsigned);
    EXPECT_TRUE(
        loom_type_equal(types.get(constant, owner), types.get(type, owner)));
    EXPECT_THROW(types.require_mutable(constant, owner), SourceRejected);
    EXPECT_EQ(loom_type_kind(types.get(control->getPointerType(type), owner)),
              LOOM_TYPE_BUFFER);
  }
}

TEST(TypesTest, BooleanEnumsKeepTheBooleanStorageContract) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("enum class Flag : bool { no, yes };"),
                IREE_SV("enum.cpp"), options);
  auto symbols = source.unit().globalScope()->find("Flag");
  ASSERT_FALSE(symbols.begin() == symbols.end());
  auto* type = (*symbols.begin())->type();
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  EXPECT_EQ(loom_type_element_type(types.get(type, owner)),
            LOOM_SCALAR_TYPE_I1);
  EXPECT_EQ(loom_type_kind(types.get(
                source.unit().control()->getPointerType(type), owner)),
            LOOM_TYPE_BUFFER);
  EXPECT_THROW(types.storage_size(type, owner), SourceRejected);
}

TEST(TypesTest, VectorProjectionKeepsLaneShapeAndRejectsPackedBoolAndPadding) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("int entry();"), IREE_SV("vectors.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* control = source.unit().control();
  auto* owner = source.unit().ast();
  auto* vector = control->getVectorType(control->getUnsignedIntType(), 16,
                                        cxx::VectorKind::kGnu);
  EXPECT_TRUE(loom_type_equal(
      types.get(vector, owner),
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, 16, 0)));
  EXPECT_EQ(loom_type_kind(types.get(control->getPointerType(vector), owner)),
            LOOM_TYPE_BUFFER);
  EXPECT_THROW(types.get(control->getVectorType(control->getBoolType(), 16,
                                                cxx::VectorKind::kExt),
                         owner),
               SourceRejected);
  EXPECT_THROW(types.get(control->getVectorType(control->getFloatType(), 3,
                                                cxx::VectorKind::kExt),
                         owner),
               SourceRejected);
}

TEST(TypesTest, MutationUsesTheObjectQualifierInsteadOfItsPointee) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("int entry();"), IREE_SV("types.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* control = source.unit().control();
  auto* owner = source.unit().ast();
  auto* constant =
      control->getQualType(control->getIntType(), cxx::CvQualifiers::kConst);
  auto* pointer = control->getPointerType(control->getIntType());
  EXPECT_THROW(types.require_mutable(constant, owner), SourceRejected);
  EXPECT_THROW(
      types.require_mutable(
          control->getQualType(pointer, cxx::CvQualifiers::kConst), owner),
      SourceRejected);
  EXPECT_NO_THROW(types.require_mutable(control->getIntType(), owner));
  EXPECT_NO_THROW(
      types.require_mutable(control->getPointerType(constant), owner));
}

}  // namespace
}  // namespace loom::cxx_import
