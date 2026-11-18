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

#include <string>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/type_builders.h"

namespace loom::cxx_import {
namespace {

using TypesWithModuleTest = ValueBuilderTest;

TEST_F(TypesWithModuleTest, ClassifiesTypeDisjointMemoryObservations) {
  Source source(IREE_SV(R"(
    namespace std { enum class byte : unsigned char {}; }
    enum PlainEnumeration { plain_value };
    enum class ScopedEnumeration { scoped_value };
    enum __attribute__((may_alias)) AliasingEnumeration { alias_value };
    typedef int AliasingInteger __attribute__((may_alias));
    using NativeVector = int __attribute__((ext_vector_type(4)));
    struct Record { int value; };
    namespace enclosing {
    namespace std { enum class byte : unsigned char {}; }
    }
    using ShadowByte = enclosing::std::byte;
  )"),
                IREE_SV("memory_observations.cxx"), options());
  Types types(source.unit(), source.diagnostics());
  auto* control = source.unit().control();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  auto expect_disjoint = [&](const cxx::Type* type) {
    EXPECT_EQ(types.memory_access_flags(type),
              LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT)
        << cxx::to_string(type);
  };
  auto expect_universal = [&](const cxx::Type* type) {
    EXPECT_EQ(types.memory_access_flags(type), 0) << cxx::to_string(type);
  };

  const cxx::Type* disjoint_types[] = {
      control->getBoolType(),
      control->getSignedCharType(),
      control->getShortIntType(),
      control->getUnsignedShortIntType(),
      control->getIntType(),
      control->getUnsignedIntType(),
      control->getLongIntType(),
      control->getUnsignedLongIntType(),
      control->getLongLongIntType(),
      control->getUnsignedLongLongIntType(),
      control->getInt128Type(),
      control->getUnsignedInt128Type(),
      control->getBitIntType(37),
      control->getUnsignedBitIntType(37),
      control->getChar8Type(),
      control->getChar16Type(),
      control->getChar32Type(),
      control->getWideCharType(),
      control->getFloatType(),
      control->getDoubleType(),
      control->getLongDoubleType(),
      control->getHalfType(),
      control->getFloat16Type(),
      control->getBFloat16Type(),
      control->getFloat8E4M3FNType(),
      control->getFloat8E5M2Type(),
      control->getFloat128Type(),
      source_type("PlainEnumeration"),
      source_type("ScopedEnumeration"),
      source_type("ShadowByte"),
  };
  for (const cxx::Type* type : disjoint_types) {
    expect_disjoint(type);
  }

  auto* standard = cxx::symbol_cast<cxx::NamespaceSymbol>(
      *source.unit().globalScope()->find("std").begin());
  ASSERT_NE(standard, nullptr);
  const cxx::Type* universal_types[] = {
      control->getCharType(),
      control->getUnsignedCharType(),
      (*standard->find("byte").begin())->type(),
      source_type("AliasingEnumeration"),
      source_type("AliasingInteger"),
      source_type("NativeVector"),
      source_type("Record"),
      control->getAtomicType(control->getIntType()),
      control->getPointerType(control->getIntType()),
      control->getNullptrType(),
  };
  for (const cxx::Type* type : universal_types) {
    expect_universal(type);
  }

  expect_disjoint(control->getQualType(control->getIntType(),
                                       cxx::CvQualifiers::kConstRestrict));
  EXPECT_EQ(
      types.memory_access_flags(control->getQualType(
          control->getIntType(), cxx::CvQualifiers::kVolatile)),
      LOOM_MEMORY_ACCESS_FLAG_VOLATILE | LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT);
  expect_disjoint(
      control->getLvalueReferenceType(control->getUnsignedIntType()));
  EXPECT_EQ(
      types.memory_access_flags(
          control->getLvalueReferenceType(control->getQualType(
              control->getIntType(), cxx::CvQualifiers::kVolatile))),
      LOOM_MEMORY_ACCESS_FLAG_VOLATILE | LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT);
  expect_universal(nullptr);

  auto c_options = options();
  c_options.standard = IREE_SV("c23");
  Source c_source(IREE_SV("signed char value;"),
                  IREE_SV("memory_observations.c"), c_options);
  Types c_types(c_source.unit(), c_source.diagnostics());
  EXPECT_EQ(c_types.memory_access_flags(
                c_source.unit().control()->getSignedCharType()),
            0);
  EXPECT_EQ(
      c_types.memory_access_flags(c_source.unit().control()->getIntType()), 0);
}

TEST_F(TypesWithModuleTest,
       ObjectPropertiesFollowOwnershipWithoutValueAdmission) {
  Source source(IREE_SV(R"(
    struct Plain { unsigned value; };
    struct Observed { volatile unsigned value; };
    struct Derived : Observed { Plain other; };
    struct Owned { Derived elements[3]; };
    struct Borrowed { Owned& reference; volatile unsigned* pointer; };
    struct VolatilePointer { unsigned* volatile pointer; };
    using Qualified = const volatile Plain;
    using Array = volatile unsigned[4];
  )"),
                IREE_SV("object_properties.cxx"), options());
  Types types(source.unit(), source.diagnostics());
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  for (const char* name :
       {"Observed", "Derived", "Owned", "Qualified", "Array"}) {
    EXPECT_TRUE(types.has_volatile_subobject(source_type(name))) << name;
    EXPECT_FALSE(types.requires_value_transport(source_type(name))) << name;
  }
  EXPECT_FALSE(types.has_volatile_subobject(source_type("Plain")));
  EXPECT_FALSE(types.has_volatile_subobject(source_type("Borrowed")));
  EXPECT_TRUE(types.requires_value_transport(source_type("Borrowed")));
  EXPECT_TRUE(types.has_volatile_subobject(source_type("VolatilePointer")));
  EXPECT_TRUE(types.requires_value_transport(source_type("VolatilePointer")));
  // The qualified and unqualified objects share a representation schema.
  EXPECT_EQ(&types.partition(source_type("Plain"), source.unit().ast()),
            &types.partition(source_type("Qualified"), source.unit().ast()));
}

TEST_F(TypesWithModuleTest, ConstructionGroupsKeepDependentHandlesWhole) {
  Source source(IREE_SV(R"(
    namespace loom::type {
    using size_type = __SIZE_TYPE__;
    inline constexpr size_type dynamic = ~size_type{0};
    template <size_type... Extents>
    struct [[loom::type("shape")]] shape {};
    template <class Shape, class T, class Encoding = void>
    struct [[loom::type("view")]] view {};
    }
    using View = loom::type::view<loom::type::shape<loom::type::dynamic>, float>;
    struct Field { View value; unsigned position; };
    struct Pair { Field first; Field second; };
    struct Native { unsigned value; unsigned other; };
    struct NativeStorage { unsigned bits : 3; };
    struct Binding { unsigned& value; const float* pointer; unsigned count; };
  )"),
                IREE_SV("construction_groups.cxx"), options());
  Types types(source.unit(), source.diagnostics());
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  EXPECT_TRUE(types.requires_value_transport(source_type("Pair")));
  EXPECT_FALSE(types.requires_value_transport(source_type("Native")));
  // Storage classification must not require value admission of native types.
  EXPECT_FALSE(types.requires_value_transport(source_type("NativeStorage")));
  auto groups = types.projections(source_type("Pair"), source.unit().ast());
  ASSERT_EQ(groups.size(), 4u);
  EXPECT_EQ(groups[0].partition->kind, ValueKind::View);
  EXPECT_EQ(groups[0].partition->component_count, 2u);
  EXPECT_EQ(groups[0].component_offset, 0u);
  EXPECT_EQ(groups[1].component_offset, 2u);
  EXPECT_EQ(groups[2].partition, groups[0].partition);
  EXPECT_EQ(groups[2].component_offset, 3u);
  EXPECT_EQ(groups[3].component_offset, 5u);
  EXPECT_EQ(groups.data(),
            types.projections(source_type("Pair"), source.unit().ast()).data());
  auto bindings =
      types.projections(source_type("Binding"), source.unit().ast());
  ASSERT_EQ(bindings.size(), 3u);
  EXPECT_EQ(bindings[0].partition, &kPointerPartition);
  EXPECT_EQ(&types.partition(bindings[0].type, source.unit().ast()),
            &kPointerPartition);
  EXPECT_EQ(bindings[1].component_offset, 2u);
  EXPECT_EQ(bindings[2].component_offset, 4u);
}

TEST_F(TypesWithModuleTest, KernelRecordsRetainArgumentAndSourcePositions) {
  Source source(IREE_SV(R"(
    using Words = unsigned __attribute__((ext_vector_type(4)));
    template<class T> struct [[loom::type("buffer")]] Buffer { T* data; };
    struct [[loom::type("index")]] Index {};
    struct Empty {};
    struct Input { const unsigned* data; unsigned count; };
    struct Arguments : Empty {
      Input inputs[2];
      Words weights;
      Buffer<unsigned> output;
      Index position;
    };
  )"),
                IREE_SV("kernel_records.cxx"), options());
  Types types(source.unit(), source.diagnostics(), module_);
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  auto* arguments_type = source_type("Arguments");
  const auto* projection = types.kernel_record(arguments_type, owner);
  ASSERT_NE(projection, nullptr);
  EXPECT_EQ(projection, types.kernel_record(arguments_type, owner));
  EXPECT_EQ(
      projection->argument_indices,
      (std::vector<size_t>{0, KernelRecordProjection::kZeroOrigin, 1, 2,
                           KernelRecordProjection::kZeroOrigin, 3, 4, 5, 6}));
  ASSERT_EQ(projection->types.size(), 7u);
  EXPECT_EQ(loom_type_kind(projection->types[0]), LOOM_TYPE_BUFFER);
  EXPECT_EQ(loom_type_element_type(projection->types[1]), LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(loom_type_kind(projection->types[2]), LOOM_TYPE_BUFFER);
  EXPECT_EQ(loom_type_element_type(projection->types[3]), LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(loom_type_kind(projection->types[4]), LOOM_TYPE_VECTOR);
  EXPECT_EQ(loom_type_kind(projection->types[5]), LOOM_TYPE_BUFFER);
  EXPECT_EQ(loom_type_element_type(projection->types[6]),
            LOOM_SCALAR_TYPE_INDEX);
  auto* empty_type = source_type("Empty");
  EXPECT_TRUE(types.kernel_record(empty_type, owner)->types.empty());
  EXPECT_TRUE(types.kernel_record(empty_type, owner)->argument_indices.empty());
  EXPECT_EQ(types.kernel_record(source.unit().control()->getIntType(), owner),
            nullptr);
}

TEST_F(TypesWithModuleTest, ProjectsSpecialAndRegisteredOpaqueHandles) {
  Source source(IREE_SV(R"(
    template<class T> struct [[loom::type("buffer")]] Buffer { T* data; };
    using WordBuffer = Buffer<unsigned>;
    using Words4 = unsigned __attribute__((vector_size(16)));
    using VectorBuffer = Buffer<Words4>;
    struct [[loom::type("index")]] Index {
      operator long long() const;
    };
    struct [[loom::type("offset")]] Offset {
      operator unsigned long long() const;
    };
    struct [[loom::type("check.entropy")]] Entropy {};
    struct [[loom::type("buffer")]] MalformedBuffer {};
    struct [[loom::type("unknown.value")]] Unknown {};
  )"),
                IREE_SV("dialect_types.cxx"), options());
  Types types(source.unit(), source.diagnostics(), module_);
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  const auto& buffer = types.partition(source_type("WordBuffer"), owner);
  ASSERT_EQ(buffer.kind, ValueKind::Buffer);
  ASSERT_EQ(buffer.component_count, 1u);
  EXPECT_EQ(loom_type_kind(types.get(source_type("WordBuffer"), owner)),
            LOOM_TYPE_BUFFER);
  EXPECT_EQ(types.unqualified(
                static_cast<const BufferPartition&>(buffer).element_type),
            source.unit().control()->getUnsignedIntType());
  std::vector<loom_type_t> buffer_signature;
  types.append(source_type("WordBuffer"), owner, buffer_signature);
  ASSERT_EQ(buffer_signature.size(), 1u);
  EXPECT_EQ(loom_type_kind(buffer_signature[0]), LOOM_TYPE_BUFFER);

  const auto& vector_buffer = static_cast<const BufferPartition&>(
      types.partition(source_type("VectorBuffer"), owner));
  EXPECT_EQ(vector_buffer.component_count, 1u);
  EXPECT_EQ(types.unqualified(vector_buffer.element_type),
            types.unqualified(source_type("Words4")));
  EXPECT_EQ(loom_type_kind(types.get(source_type("VectorBuffer"), owner)),
            LOOM_TYPE_BUFFER);

  const auto& index = types.partition(source_type("Index"), owner);
  ASSERT_EQ(index.kind, ValueKind::Address);
  ASSERT_EQ(index.component_count, 1u);
  auto index_type = types.get(source_type("Index"), owner);
  EXPECT_EQ(loom_type_element_type(index_type), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_TRUE(types.is_index(source_type("Index"), owner));
  EXPECT_FALSE(types.is_offset(source_type("Index"), owner));
  std::vector<loom_type_t> index_signature;
  types.append(source_type("Index"), owner, index_signature);
  ASSERT_EQ(index_signature.size(), 1u);
  EXPECT_TRUE(loom_type_equal(index_signature[0], index_type));

  const auto& offset = types.partition(source_type("Offset"), owner);
  ASSERT_EQ(offset.kind, ValueKind::Address);
  ASSERT_EQ(offset.component_count, 1u);
  auto offset_type = types.get(source_type("Offset"), owner);
  EXPECT_EQ(loom_type_element_type(offset_type), LOOM_SCALAR_TYPE_OFFSET);
  EXPECT_TRUE(types.is_offset(source_type("Offset"), owner));
  EXPECT_FALSE(types.is_index(source_type("Offset"), owner));
  std::vector<loom_type_t> offset_signature;
  types.append(source_type("Offset"), owner, offset_signature);
  ASSERT_EQ(offset_signature.size(), 1u);
  EXPECT_TRUE(loom_type_equal(offset_signature[0], offset_type));

  const auto& entropy = types.partition(source_type("Entropy"), owner);
  ASSERT_EQ(entropy.kind, ValueKind::OpaqueDialect);
  ASSERT_EQ(entropy.component_count, 1u);
  auto projected = types.get(source_type("Entropy"), owner);
  ASSERT_TRUE(loom_type_is_dialect(projected));
  EXPECT_TRUE(
      types.is_opaque_dialect(source_type("Entropy"), "check.entropy", owner));
  EXPECT_FALSE(
      types.is_opaque_dialect(source_type("Entropy"), "async.token", owner));
  std::vector<loom_type_t> signature;
  types.append(source_type("Entropy"), owner, signature);
  ASSERT_EQ(signature.size(), 1u);
  EXPECT_TRUE(loom_type_equal(signature[0], projected));

  EXPECT_THROW(types.get(source_type("MalformedBuffer"), owner),
               SourceRejected);
  EXPECT_THROW(types.get(source_type("Unknown"), owner), SourceRejected);
}

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
    EXPECT_EQ(loom_type_element_type(components[1]), LOOM_SCALAR_TYPE_INDEX);
  }
  EXPECT_THROW(types.storage_size(control->getVoidType(), owner),
               SourceRejected);
  EXPECT_THROW(types.storage_size(source_type("Opaque"), owner),
               SourceRejected);
  EXPECT_THROW(types.storage_size(source_type("Payload"), owner),
               SourceRejected);
  EXPECT_EQ(types.storage_size(source_type("Owned"), owner), 1);
  EXPECT_THROW(
      types.storage_size(control->getPointerType(control->getIntType()), owner),
      SourceRejected);
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
  EXPECT_EQ(types.member(packet->members[2].source, owner).component_offset,
            2u);
  EXPECT_EQ(packet->component_names,
            (std::vector<std::string>{"pair_a", "pair_b", "pointer",
                                      "pointer_byte_offset", "valid"}));
  EXPECT_NE(packet->members[1].partition,
            &types.partition(source_type("Other"), owner));
  EXPECT_EQ(types.partition(source_type("Pair"), owner).kind,
            ValueKind::Record);
  EXPECT_EQ(packet->members[2].partition->kind, ValueKind::Pointer);
  EXPECT_TRUE(source.unit().typeTraits().is_const(packet->members[3].type()));
  EXPECT_EQ(types.record(source_type("Empty"), owner)->source->sizeInBytes(),
            1);
  std::vector<loom_type_t> signature;
  types.append(source_type("Packet"), owner, signature);
  ASSERT_EQ(signature.size(), 5u);
  EXPECT_EQ(loom_type_element_type(signature[0]), LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(loom_type_kind(signature[2]), LOOM_TYPE_BUFFER);
  EXPECT_EQ(loom_type_element_type(signature[3]), LOOM_SCALAR_TYPE_INDEX);
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
  EXPECT_EQ(partition->members[1].byte_offset(), 4u);
  EXPECT_EQ(partition->members[1].component_offset, 1u);
  EXPECT_EQ(types.storage_size(padded, owner), 12);
  auto* observed = source_type("Observed");
  EXPECT_EQ(types.storage_size(observed, owner), 4);
  EXPECT_EQ(types.partition(control->getPointerType(observed), owner).kind,
            ValueKind::Pointer);
  EXPECT_EQ(types.partition(observed, owner).kind, ValueKind::Record);
  EXPECT_TRUE(types.has_volatile_subobject(observed));
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
  EXPECT_EQ(types.partition(block, owner).component_count, 134u);
  const auto& array_partition = types.partition(array, owner);
  EXPECT_EQ(array_partition.kind, ValueKind::Array);
  EXPECT_EQ(array_partition.component_count, 402u);
  EXPECT_EQ(static_cast<const ArrayPartition&>(array_partition).element,
            &types.partition(block, owner));
  const auto* records = types.record(source_type("Records"), owner);
  EXPECT_EQ(records->component_count, 403u);
  EXPECT_EQ(records->component_names.back(), "blocks_2_quants_127");
  const auto* vectors = types.record(source_type("Vectors"), owner);
  EXPECT_EQ(vectors->component_count, 7u);
  EXPECT_EQ(vectors->component_names.back(), "values_1_2");
  EXPECT_FALSE(types.requires_binding(source_type("Vectors"), owner));
  EXPECT_THROW(types.storage_size(control->getUnboundedArrayType(block), owner),
               SourceRejected);
}

TEST(TypesTest, RejectsRepresentationsThatLoseSourceSemantics) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("int entry();"), IREE_SV("types.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* control = source.unit().control();
  auto* owner = source.unit().ast();
  EXPECT_THROW(
      types.get(control->getLvalueReferenceType(control->getIntType()), owner),
      SourceRejected);
  EXPECT_THROW(types.get(control->getLongDoubleType(), owner), SourceRejected);
}

TEST(TypesTest, SeparatesRepresentationFromVolatileObservation) {
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
  EXPECT_EQ(types.memory_access_flags(integer),
            LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT);
  EXPECT_EQ(types.memory_access_flags(pointer), 0);
  EXPECT_EQ(
      types.memory_access_flags(observed),
      LOOM_MEMORY_ACCESS_FLAG_VOLATILE | LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT);
  EXPECT_EQ(types.partition(pointer, owner).kind, ValueKind::Pointer);
  EXPECT_EQ(types.partition(observed, owner).kind, ValueKind::SSA);
  EXPECT_TRUE(types.has_volatile_subobject(observed));
  auto* observed_pointer =
      control->getQualType(pointer, cxx::CvQualifiers::kVolatile);
  EXPECT_EQ(types.partition(observed_pointer, owner).kind, ValueKind::Pointer);
  EXPECT_TRUE(types.has_volatile_subobject(observed_pointer));
  EXPECT_FALSE(types.has_volatile_subobject(pointer));
  EXPECT_THROW(types.storage_size(observed_pointer, owner), SourceRejected);
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
                true},
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
  EXPECT_EQ(types.storage_size(type, owner), 1);
}

TEST(TypesTest, BooleanObjectsHaveByteLayoutAndPredicateValues) {
  for (auto model : {LOOM_CXX_DATA_MODEL_LP64, LOOM_CXX_DATA_MODEL_LLP64,
                     LOOM_CXX_DATA_MODEL_ILP32}) {
    loom_cxx_import_options_t options;
    loom_cxx_import_options_initialize(&options);
    options.data_model = model;
    Source source(IREE_SV("int entry();"), IREE_SV("types.cpp"), options);
    Types types(source.unit(), source.diagnostics());
    auto* control = source.unit().control();
    auto* owner = source.unit().ast();
    const cxx::Type* elements[] = {
        control->getBoolType(),
        control->getQualType(control->getBoolType(), cxx::CvQualifiers::kConst),
        control->getQualType(control->getBoolType(),
                             cxx::CvQualifiers::kVolatile),
    };
    for (auto* element : elements) {
      EXPECT_EQ(loom_type_element_type(types.get(element, owner)),
                LOOM_SCALAR_TYPE_I1);
      EXPECT_EQ(types.storage_size(element, owner), 1);
      EXPECT_EQ(types.partition(control->getPointerType(element), owner).kind,
                ValueKind::Pointer);
      auto* array = control->getBoundedArrayType(element, 7);
      EXPECT_EQ(loom_type_kind(types.get(array, owner)), LOOM_TYPE_BUFFER);
      EXPECT_EQ(types.storage_size(array, owner), 7);
    }
  }
}

TEST(TypesTest, CharacterCodeUnitsRetainTheSourceDataModel) {
  for (auto model : {LOOM_CXX_DATA_MODEL_LP64, LOOM_CXX_DATA_MODEL_LLP64,
                     LOOM_CXX_DATA_MODEL_ILP32}) {
    loom_cxx_import_options_t options;
    loom_cxx_import_options_initialize(&options);
    options.data_model = model;
    Source source(IREE_SV("int entry();"), IREE_SV("types.cpp"), options);
    Types types(source.unit(), source.diagnostics());
    auto* control = source.unit().control();
    auto* owner = source.unit().ast();
    const bool windows = model == LOOM_CXX_DATA_MODEL_LLP64;
    const struct {
      // Source code-unit type, independent of its integer representation.
      const cxx::Type* type;
      // Expected signless IR scalar representation.
      loom_scalar_type_t scalar;
      // Expected source ABI byte width.
      int64_t bytes;
      // Whether widening source arithmetic zero-extends the code unit.
      bool is_unsigned;
    } cases[] = {
        {control->getChar8Type(), LOOM_SCALAR_TYPE_I8, 1, true},
        {control->getChar16Type(), LOOM_SCALAR_TYPE_I16, 2, true},
        {control->getChar32Type(), LOOM_SCALAR_TYPE_I32, 4, true},
        {control->getWideCharType(),
         static_cast<loom_scalar_type_t>(windows ? LOOM_SCALAR_TYPE_I16
                                                 : LOOM_SCALAR_TYPE_I32),
         windows ? 2 : 4, windows},
    };
    for (const auto& test : cases) {
      EXPECT_EQ(loom_type_element_type(types.get(test.type, owner)),
                test.scalar);
      EXPECT_EQ(types.is_unsigned(test.type), test.is_unsigned);
      EXPECT_EQ(types.storage_size(test.type, owner), test.bytes);
      EXPECT_EQ(
          types.storage_size(control->getBoundedArrayType(test.type, 3), owner),
          3 * test.bytes);
    }
  }
}

TEST(TypesTest, VectorProjectionSeparatesLogicalLanesFromObjectLayout) {
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
  const struct {
    const cxx::Type* lane;
    loom_scalar_type_t element;
    int64_t lane_bytes;
  } cases[] = {
      {control->getSignedCharType(), LOOM_SCALAR_TYPE_I8, 1},
      {control->getUnsignedCharType(), LOOM_SCALAR_TYPE_I8, 1},
      {control->getShortIntType(), LOOM_SCALAR_TYPE_I16, 2},
      {control->getUnsignedShortIntType(), LOOM_SCALAR_TYPE_I16, 2},
      {control->getIntType(), LOOM_SCALAR_TYPE_I32, 4},
      {control->getUnsignedIntType(), LOOM_SCALAR_TYPE_I32, 4},
      {control->getLongLongIntType(), LOOM_SCALAR_TYPE_I64, 8},
      {control->getUnsignedLongLongIntType(), LOOM_SCALAR_TYPE_I64, 8},
      {control->getFloat8E4M3FNType(), LOOM_SCALAR_TYPE_F8E4M3, 1},
      {control->getFloat8E5M2Type(), LOOM_SCALAR_TYPE_F8E5M2, 1},
      {control->getFloat16Type(), LOOM_SCALAR_TYPE_F16, 2},
      {control->getBFloat16Type(), LOOM_SCALAR_TYPE_BF16, 2},
      {control->getFloatType(), LOOM_SCALAR_TYPE_F32, 4},
      {control->getDoubleType(), LOOM_SCALAR_TYPE_F64, 8},
  };
  const uint64_t lane_counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16};
  for (const auto& test : cases) {
    for (auto lane_count : lane_counts) {
      uint64_t object_lane_count = 1;
      while (object_lane_count < lane_count) {
        object_lane_count *= 2;
      }
      auto* ext =
          control->getVectorType(test.lane, lane_count, cxx::VectorKind::kExt);
      EXPECT_TRUE(loom_type_equal(
          types.get(ext, owner),
          loom_type_shaped_1d(LOOM_TYPE_VECTOR, test.element, lane_count, 0)));
      EXPECT_EQ(types.storage_size(ext, owner),
                test.lane_bytes * object_lane_count);
      EXPECT_EQ(types.has_vector_object_padding(ext, owner),
                lane_count != object_lane_count);
    }
  }
}

TEST(TypesTest, RetainsSelectedSubobjectPositionsIndependentlyOfByteLayout) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Leaf { double value; };
    struct Middle { char tag; Leaf leaf; };
    struct Outer { unsigned count; Middle first, second; };
  )"),
                IREE_SV("subobjects.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto* outer = types.record(
      (*source.unit().globalScope()->find("Outer").begin())->type(), owner);
  const auto& second = outer->members[2];
  const auto& middle = static_cast<const RecordPartition&>(*second.partition);
  auto* path = cxx::make_list_node(source.unit().arena(), second.source);
  path->next =
      cxx::make_list_node(source.unit().arena(), middle.members[1].source);
  EXPECT_EQ(types.subobject_offset(path), 32u);
  const auto& selected = types.projection(path, owner);
  EXPECT_EQ(selected.root->source, second.source);
  EXPECT_EQ(selected.partition, middle.members[1].partition);
  EXPECT_EQ(selected.component_offset, 4u);
  EXPECT_EQ(selected.byte_offset, 32u);
  EXPECT_EQ(&types.projection(path, owner), &selected);
  EXPECT_EQ(types.subobject_offset(path), selected.byte_offset);
}

TEST(TypesTest, BasePartitionsPreserveRepeatedIdentityAndOverlappingPadding) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Empty {};
    struct Left : Empty {};
    struct Right : Empty {};
    struct Both : Left, Right { unsigned value; };
    struct Tail {
      unsigned value;
      unsigned char tag;
      Tail() : value(0), tag(0) {}
    };
    struct Derived : Tail { unsigned char extra; };
  )"),
                IREE_SV("base_layout.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };
  auto* both = types.record(source_type("Both"), owner);
  ASSERT_EQ(both->members.size(), 3u);
  EXPECT_EQ(both->component_count, 1u);
  EXPECT_NE(both->members[0].source, both->members[1].source);
  EXPECT_EQ(both->members[0].byte_offset(), 0u);
  EXPECT_EQ(both->members[1].byte_offset(), 1u);
  EXPECT_EQ(both->members[2].byte_offset(), 0u);
  EXPECT_EQ(both->members[0].partition->component_count, 0u);
  EXPECT_EQ(both->members[1].partition->component_count, 0u);
  EXPECT_EQ(types.storage_size(source_type("Both"), owner), 4);

  auto* derived = types.record(source_type("Derived"), owner);
  ASSERT_EQ(derived->members.size(), 2u);
  EXPECT_EQ(derived->component_count, 3u);
  EXPECT_EQ(derived->members[1].component_offset, 2u);
  EXPECT_EQ(derived->members[1].byte_offset(), 5u);
  EXPECT_EQ(types.storage_size(source_type("Tail"), owner), 8);
  EXPECT_EQ(types.storage_size(source_type("Derived"), owner), 8);
}

TEST(TypesTest, RetainsMutableSubobjectsThroughBasesAndArrays) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Leaf { mutable unsigned value; };
    struct Plain { unsigned value; };
    struct Derived : Leaf { Plain plain; };
    struct Nested { Leaf leaves[2][3]; };
    struct Borrowed { const Leaf* pointer; };
  )"),
                IREE_SV("mutable_subobjects.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto record = [&](const char* name) {
    return types.record(
        (*source.unit().globalScope()->find(name).begin())->type(), owner);
  };
  EXPECT_TRUE(record("Derived")->has_mutable_subobject);
  EXPECT_TRUE(record("Nested")->has_mutable_subobject);
  EXPECT_FALSE(record("Plain")->has_mutable_subobject);
  EXPECT_FALSE(record("Borrowed")->has_mutable_subobject);
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
