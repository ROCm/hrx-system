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

TEST_F(TypesWithModuleTest, ProjectsPipelinePayloadAndHandleFamilies) {
  Source source(IREE_SV(R"(
    namespace loom {
    namespace encoding {
    enum class role { layout, schema, storage, transform };
    }
    namespace channel {
    enum class mode { immutable = 0, mutable_ = 1 };
    }
    namespace type {
    using size_type = __SIZE_TYPE__;
    inline constexpr size_type dynamic = ~size_type{0};
    template<size_type... Extents>
    struct [[loom::type("shape")]] shape {};
    template<loom::encoding::role Role, size_type Rank>
    struct [[loom::type("encoding")]] encoding {};
    template<class Shape, class T, class Encoding = void>
    struct [[loom::type("tile")]] tile {};
    struct [[loom::type("pool")]] pool {};
    template<class Payload>
    struct [[loom::type("channel")]] channel {};
    template<class Payload, loom::channel::mode Mode =
                                loom::channel::mode::immutable>
    struct [[loom::type("read")]] read {};
    template<class Payload>
    struct [[loom::type("write")]] write {};
    }
    }
    using Layout = loom::type::encoding<loom::encoding::role::layout, 2>;
    using StaticTile = loom::type::tile<loom::type::shape<4, 8>, float>;
    using DynamicTile = loom::type::tile<
        loom::type::shape<loom::type::dynamic, 8>, float, Layout>;
    using StaticChannel = loom::type::channel<StaticTile>;
    using DynamicChannel = loom::type::channel<DynamicTile>;
    using ImmutableRead = loom::type::read<StaticTile>;
    using MutableRead = loom::type::read<
        StaticTile, loom::channel::mode::mutable_>;
    using DynamicWrite = loom::type::write<DynamicTile>;
    using Pool = loom::type::pool;
  )"),
                IREE_SV("pipeline_types.cxx"), options());
  Types types(source.unit(), source.diagnostics(), module_);
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  const auto& static_tile = static_cast<const TilePartition&>(
      types.partition(source_type("StaticTile"), owner));
  EXPECT_EQ(static_tile.kind, ValueKind::Tile);
  EXPECT_EQ(static_tile.component_count, 1u);
  EXPECT_FALSE(static_tile.requires_binding());
  EXPECT_EQ(loom_type_kind(static_tile.static_type), LOOM_TYPE_TILE);
  EXPECT_EQ(loom_type_dim_static_size_at(static_tile.static_type, 0), 4);
  EXPECT_EQ(loom_type_dim_static_size_at(static_tile.static_type, 1), 8);

  const auto& dynamic_tile = static_cast<const TilePartition&>(
      types.partition(source_type("DynamicTile"), owner));
  EXPECT_EQ(dynamic_tile.component_count, 3u);
  EXPECT_TRUE(dynamic_tile.requires_binding());
  EXPECT_EQ(dynamic_tile.component_names,
            (std::vector<std::string>{"dim0", "encoding", ""}));
  const loom_value_id_t dynamic_ids[] = {11, 12, 13};
  BoundTypeStorage tile_storage;
  std::vector<loom_type_t> tile_types;
  types.append_bound(source_type("DynamicTile"), owner, dynamic_ids,
                     tile_storage, tile_types);
  ASSERT_EQ(tile_types.size(), 3u);
  EXPECT_EQ(loom_type_element_type(tile_types[0]), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_encoding_role(tile_types[1]),
            LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_EQ(loom_type_kind(tile_types[2]), LOOM_TYPE_TILE);
  EXPECT_EQ(loom_type_dim_value_id_at(tile_types[2], 0), 11u);
  EXPECT_EQ(loom_type_dim_static_size_at(tile_types[2], 1), 8);
  EXPECT_EQ(loom_type_encoding_value_id(tile_types[2]), 12u);

  std::vector<loom_type_t> static_channel_types;
  types.append(source_type("StaticChannel"), owner, static_channel_types);
  ASSERT_EQ(static_channel_types.size(), 1u);
  ASSERT_TRUE(loom_channel_type_isa(static_channel_types[0]));
  auto static_payload = loom_type_table_get(
      &module_->types, loom_channel_type_payload(static_channel_types[0]));
  EXPECT_TRUE(loom_type_equal(static_payload, static_tile.static_type));

  const auto& dynamic_channel = static_cast<const ChannelPartition&>(
      types.partition(source_type("DynamicChannel"), owner));
  EXPECT_EQ(dynamic_channel.component_count, 3u);
  EXPECT_TRUE(dynamic_channel.requires_binding());
  EXPECT_EQ(dynamic_channel.component_names,
            (std::vector<std::string>{"dim0", "encoding", ""}));
  BoundTypeStorage channel_storage;
  std::vector<loom_type_t> channel_types;
  types.append_bound(source_type("DynamicChannel"), owner, dynamic_ids,
                     channel_storage, channel_types);
  ASSERT_EQ(channel_types.size(), 3u);
  ASSERT_TRUE(loom_channel_type_isa(channel_types[2]));
  auto dynamic_payload = loom_type_table_get(
      &module_->types, loom_channel_type_payload(channel_types[2]));
  EXPECT_EQ(loom_type_dim_value_id_at(dynamic_payload, 0), 11u);
  EXPECT_EQ(loom_type_encoding_value_id(dynamic_payload), 12u);

  std::vector<loom_type_t> immutable_read_types;
  types.append(source_type("ImmutableRead"), owner, immutable_read_types);
  ASSERT_EQ(immutable_read_types.size(), 1u);
  EXPECT_TRUE(loom_read_type_isa(immutable_read_types[0]));
  EXPECT_FALSE(loom_read_type_has_mode(immutable_read_types[0]));
  std::vector<loom_type_t> mutable_read_types;
  types.append(source_type("MutableRead"), owner, mutable_read_types);
  ASSERT_EQ(mutable_read_types.size(), 1u);
  EXPECT_TRUE(loom_read_type_has_mode(mutable_read_types[0]));
  EXPECT_EQ(loom_read_type_mode(mutable_read_types[0]),
            LOOM_READ_TYPE_MODE_MUTABLE);

  BoundTypeStorage write_storage;
  std::vector<loom_type_t> write_types;
  types.append_bound(source_type("DynamicWrite"), owner, dynamic_ids,
                     write_storage, write_types);
  ASSERT_EQ(write_types.size(), 3u);
  EXPECT_TRUE(loom_write_type_isa(write_types[2]));

  const auto& pool = types.partition(source_type("Pool"), owner);
  EXPECT_EQ(pool.kind, ValueKind::Pool);
  EXPECT_EQ(loom_type_kind(types.get(source_type("Pool"), owner)),
            LOOM_TYPE_POOL);
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
    EXPECT_EQ(loom_type_element_type(components[1]), LOOM_SCALAR_TYPE_OFFSET);
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
  EXPECT_EQ(partition->members[1].byte_offset(), 4u);
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

TEST(TypesTest, ViewPartitionsPreservePlainAndEncodedSourceTypes) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(
      IREE_SV("namespace loom { namespace encoding { "
              "enum class role { layout, schema, storage, transform }; "
              "} namespace type { "
              "using size_type = __SIZE_TYPE__; "
              "inline constexpr size_type dynamic = ~size_type{0}; "
              "template <loom::encoding::role Role, size_type Rank> "
              "struct [[loom::type(\"encoding\")]] encoding { "
              "size_type strides[Rank]; }; "
              "template <size_type... Extents> "
              "struct [[loom::type(\"shape\")]] shape {}; "
              "template <class Shape, class T, class Encoding = void> "
              "struct [[loom::type(\"view\")]] view { T* data; }; "
              "} } "
              "using Layout = loom::type::encoding<"
              "loom::encoding::role::layout, 2>; "
              "using Storage3 = loom::type::encoding<"
              "loom::encoding::role::storage, 3>; "
              "using Plain = loom::type::view<loom::type::shape<"
              "loom::type::dynamic, 32>, const float>; "
              "using Plane = loom::type::view<loom::type::shape<"
              "loom::type::dynamic, 32>, const float, Layout>; "
              "using Observed = loom::type::view<loom::type::shape<"
              "loom::type::dynamic, 32>, const volatile float>; "
              "using Volume = loom::type::view<loom::type::shape<"
              "loom::type::dynamic, 4, loom::type::dynamic>, "
              "const float, Storage3>; "
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

  const auto& plain = types.partition(source_type("Plain"), owner);
  EXPECT_EQ(plain.kind, ValueKind::View);
  EXPECT_EQ(plain.component_count, 2u);
  EXPECT_EQ(static_cast<const ViewPartition&>(plain).encoding, nullptr);
  const loom_value_id_t plain_ids[] = {9, 10};
  BoundTypeStorage plain_storage;
  std::vector<loom_type_t> plain_types;
  types.append_bound(source_type("Plain"), owner, plain_ids, plain_storage,
                     plain_types);
  ASSERT_EQ(plain_types.size(), 2u);
  EXPECT_EQ(loom_type_kind(plain_types[1]), LOOM_TYPE_VIEW);
  EXPECT_FALSE(loom_type_has_encoding(plain_types[1]));
  EXPECT_EQ(loom_type_dim_value_id_at(plain_types[1], 0), 9u);

  const auto& plane = types.partition(source_type("Plane"), owner);
  EXPECT_EQ(plane.kind, ValueKind::View);
  EXPECT_EQ(plane.component_count, 3u);
  EXPECT_TRUE(types.requires_binding(source_type("Plane"), owner));
  EXPECT_EQ(static_cast<const ViewPartition&>(plane).encoding,
            &static_cast<const EncodingPartition&>(layout));
  EXPECT_EQ(static_cast<const ViewPartition&>(plane).access_flags, 0);
  const auto& observed = types.partition(source_type("Observed"), owner);
  EXPECT_EQ(static_cast<const ViewPartition&>(observed).access_flags,
            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  const loom_value_id_t plane_ids[] = {11, 12, 13};
  BoundTypeStorage plane_storage;
  std::vector<loom_type_t> plane_types;
  types.append_bound(source_type("Plane"), owner, plane_ids, plane_storage,
                     plane_types);
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

  const auto& volume = types.partition(source_type("Volume"), owner);
  EXPECT_EQ(volume.kind, ValueKind::View);
  EXPECT_EQ(volume.component_count, 4u);
  const loom_value_id_t volume_ids[] = {31, 32, 33, 34};
  BoundTypeStorage volume_storage;
  std::vector<loom_type_t> volume_types;
  types.append_bound(source_type("Volume"), owner, volume_ids, volume_storage,
                     volume_types);
  ASSERT_EQ(volume_types.size(), 4u);
  EXPECT_EQ(loom_type_encoding_role(volume_types[2]),
            LOOM_ENCODING_ROLE_PHYSICAL_STORAGE);
  EXPECT_EQ(loom_type_rank(volume_types[3]), 3u);
  EXPECT_EQ(loom_type_dim_value_id_at(volume_types[3], 0), 31u);
  EXPECT_EQ(loom_type_dim_static_size_at(volume_types[3], 1), 4);
  EXPECT_EQ(loom_type_dim_value_id_at(volume_types[3], 2), 32u);
  EXPECT_EQ(loom_type_encoding_value_id(volume_types[3]), 33u);
  std::vector<loom_type_t> unbound;
  EXPECT_THROW(types.append(source_type("Plane"), owner, unbound),
               SourceRejected);

  const auto* packet = types.record(source_type("Packet"), owner);
  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(packet->component_count, 4u);
  EXPECT_EQ(
      packet->component_names,
      (std::vector<std::string>{"plane_dim0", "plane_layout", "plane", "tag"}));
  EXPECT_TRUE(types.requires_binding(source_type("Packet"), owner));
  const loom_value_id_t packet_ids[] = {21, 22, 23, 24};
  BoundTypeStorage packet_storage;
  std::vector<loom_type_t> packet_types;
  types.append_bound(source_type("Packet"), owner, packet_ids, packet_storage,
                     packet_types);
  ASSERT_EQ(packet_types.size(), 4u);
  EXPECT_EQ(loom_type_dim_value_id_at(packet_types[2], 0), 21u);
  EXPECT_EQ(loom_type_encoding_value_id(packet_types[2]), 22u);
  EXPECT_EQ(loom_type_element_type(packet_types[3]), LOOM_SCALAR_TYPE_I32);
}

TEST(TypesTest, BindsRuntimeTensorExtentAcrossValueBoundaries) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(
      IREE_SV("namespace loom { namespace type { "
              "using size_type = __SIZE_TYPE__; "
              "inline constexpr size_type dynamic = ~size_type{0}; "
              "template <size_type... Extents> "
              "struct [[loom::type(\"shape\")]] shape {}; "
              "template <class Shape, class T> "
              "struct [[loom::type(\"tensor\")]] tensor { T* data; }; "
              "} } "
              "using Static = loom::type::tensor<loom::type::shape<7>, int>; "
              "using Dynamic = loom::type::tensor<"
              "loom::type::shape<loom::type::dynamic>, float>; "
              "struct Packet { Dynamic values; unsigned tag; };"),
      IREE_SV("tensors.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  const auto& static_tensor = static_cast<const TensorPartition&>(
      types.partition(source_type("Static"), owner));
  EXPECT_EQ(static_tensor.component_count, 1u);
  EXPECT_FALSE(static_tensor.is_dynamic());
  EXPECT_FALSE(types.requires_binding(source_type("Static"), owner));
  std::vector<loom_type_t> static_types;
  types.append(source_type("Static"), owner, static_types);
  ASSERT_EQ(static_types.size(), 1u);
  EXPECT_EQ(loom_type_kind(static_types[0]), LOOM_TYPE_TENSOR);
  EXPECT_EQ(loom_type_dim_static_size_at(static_types[0], 0), 7);
  EXPECT_EQ(loom_type_element_type(static_types[0]), LOOM_SCALAR_TYPE_I32);

  const auto& dynamic_tensor = static_cast<const TensorPartition&>(
      types.partition(source_type("Dynamic"), owner));
  EXPECT_EQ(dynamic_tensor.component_count, 2u);
  EXPECT_TRUE(dynamic_tensor.is_dynamic());
  EXPECT_TRUE(types.requires_binding(source_type("Dynamic"), owner));
  const loom_value_id_t dynamic_ids[] = {41, 42};
  BoundTypeStorage dynamic_storage;
  std::vector<loom_type_t> dynamic_types;
  types.append_bound(source_type("Dynamic"), owner, dynamic_ids,
                     dynamic_storage, dynamic_types);
  ASSERT_EQ(dynamic_types.size(), 2u);
  EXPECT_EQ(loom_type_element_type(dynamic_types[0]), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_kind(dynamic_types[1]), LOOM_TYPE_TENSOR);
  EXPECT_EQ(loom_type_dim_value_id_at(dynamic_types[1], 0), 41u);
  EXPECT_EQ(loom_type_element_type(dynamic_types[1]), LOOM_SCALAR_TYPE_F32);

  const auto* packet = types.record(source_type("Packet"), owner);
  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(packet->component_count, 3u);
  EXPECT_EQ(packet->component_names,
            (std::vector<std::string>{"values_dim0", "values", "tag"}));
  EXPECT_TRUE(types.requires_binding(source_type("Packet"), owner));
}

TEST(TypesTest, BindsRankGenericSemanticVectorsAndNestedRecords) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    namespace loom { namespace type {
    using size_type = __SIZE_TYPE__;
    inline constexpr size_type dynamic = ~size_type{0};
    template <size_type... Extents>
    struct [[loom::type("shape")]] shape {};
    template <class Shape, class T>
    struct [[loom::type("vector")]] vector {};
    } }
    using Static = loom::type::vector<loom::type::shape<2, 3, 4>, float>;
    using Dynamic = loom::type::vector<
        loom::type::shape<loom::type::dynamic>, float>;
    using Mixed = loom::type::vector<loom::type::shape<
        loom::type::dynamic, 8, loom::type::dynamic>, unsigned>;
    using Mask = loom::type::vector<
        loom::type::shape<loom::type::dynamic>, bool>;
    using Qualified = loom::type::vector<loom::type::shape<4>, const float>;
    using Nested = loom::type::vector<loom::type::shape<4>, Static>;
    struct Packet { Mixed values; unsigned tag; };
  )"),
                IREE_SV("semantic_vectors.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  const auto& static_vector = static_cast<const VectorPartition&>(
      types.partition(source_type("Static"), owner));
  EXPECT_EQ(static_vector.component_count, 1u);
  EXPECT_FALSE(static_vector.is_dynamic());
  EXPECT_FALSE(types.requires_binding(source_type("Static"), owner));
  auto static_type = types.get(source_type("Static"), owner);
  EXPECT_EQ(loom_type_kind(static_type), LOOM_TYPE_VECTOR);
  EXPECT_EQ(loom_type_rank(static_type), 3u);
  EXPECT_EQ(loom_type_dim_static_size_at(static_type, 0), 2);
  EXPECT_EQ(loom_type_dim_static_size_at(static_type, 1), 3);
  EXPECT_EQ(loom_type_dim_static_size_at(static_type, 2), 4);
  EXPECT_EQ(loom_type_element_type(static_type), LOOM_SCALAR_TYPE_F32);
  std::vector<loom_type_t> static_types;
  types.append(source_type("Static"), owner, static_types);
  ASSERT_EQ(static_types.size(), 1u);
  EXPECT_TRUE(loom_type_equal(static_types[0], static_type));

  const auto& dynamic_vector = static_cast<const VectorPartition&>(
      types.partition(source_type("Dynamic"), owner));
  EXPECT_EQ(dynamic_vector.component_count, 2u);
  EXPECT_TRUE(dynamic_vector.is_dynamic());
  EXPECT_TRUE(types.requires_binding(source_type("Dynamic"), owner));
  EXPECT_EQ(dynamic_vector.component_names,
            (std::vector<std::string>{"dim0", ""}));
  const loom_value_id_t dynamic_ids[] = {41, 42};
  BoundTypeStorage dynamic_storage;
  std::vector<loom_type_t> dynamic_types;
  types.append_bound(source_type("Dynamic"), owner, dynamic_ids,
                     dynamic_storage, dynamic_types);
  ASSERT_EQ(dynamic_types.size(), 2u);
  EXPECT_EQ(loom_type_element_type(dynamic_types[0]), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_kind(dynamic_types[1]), LOOM_TYPE_VECTOR);
  EXPECT_EQ(loom_type_dim_value_id_at(dynamic_types[1], 0), 41u);
  EXPECT_EQ(loom_type_element_type(dynamic_types[1]), LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(dynamic_vector.vector_value(dynamic_ids), 42u);
  EXPECT_EQ(types.native_vector(source_type("Dynamic")), nullptr);
  EXPECT_THROW(types.get(source_type("Dynamic"), owner), SourceRejected);

  const auto& mixed_vector = static_cast<const VectorPartition&>(
      types.partition(source_type("Mixed"), owner));
  EXPECT_EQ(mixed_vector.component_count, 3u);
  EXPECT_EQ(mixed_vector.component_names,
            (std::vector<std::string>{"dim0", "dim2", ""}));
  const loom_value_id_t mixed_ids[] = {51, 52, 53};
  BoundTypeStorage mixed_storage;
  std::vector<loom_type_t> mixed_types;
  types.append_bound(source_type("Mixed"), owner, mixed_ids, mixed_storage,
                     mixed_types);
  ASSERT_EQ(mixed_types.size(), 3u);
  EXPECT_EQ(loom_type_element_type(mixed_types[0]), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_element_type(mixed_types[1]), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_rank(mixed_types[2]), 3u);
  EXPECT_EQ(loom_type_dim_value_id_at(mixed_types[2], 0), 51u);
  EXPECT_EQ(loom_type_dim_static_size_at(mixed_types[2], 1), 8);
  EXPECT_EQ(loom_type_dim_value_id_at(mixed_types[2], 2), 52u);
  EXPECT_EQ(loom_type_element_type(mixed_types[2]), LOOM_SCALAR_TYPE_I32);

  const auto* packet = types.record(source_type("Packet"), owner);
  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(packet->component_count, 4u);
  EXPECT_EQ(packet->component_names,
            (std::vector<std::string>{"values_dim0", "values_dim2", "values",
                                      "tag"}));
  EXPECT_TRUE(types.requires_binding(source_type("Packet"), owner));
  const loom_value_id_t packet_ids[] = {61, 62, 63, 64};
  BoundTypeStorage packet_storage;
  std::vector<loom_type_t> packet_types;
  types.append_bound(source_type("Packet"), owner, packet_ids, packet_storage,
                     packet_types);
  ASSERT_EQ(packet_types.size(), 4u);
  EXPECT_EQ(loom_type_dim_value_id_at(packet_types[2], 0), 61u);
  EXPECT_EQ(loom_type_dim_value_id_at(packet_types[2], 2), 62u);
  EXPECT_EQ(loom_type_element_type(packet_types[3]), LOOM_SCALAR_TYPE_I32);

  const auto& mask = static_cast<const VectorPartition&>(
      types.partition(source_type("Mask"), owner));
  EXPECT_EQ(mask.element, LOOM_SCALAR_TYPE_I1);
  EXPECT_THROW(types.partition(source_type("Qualified"), owner),
               SourceRejected);
  EXPECT_THROW(types.partition(source_type("Nested"), owner), SourceRejected);
}

TEST(TypesTest, AppendsVectorLanesToTensorObjectShape) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
        namespace loom { namespace type {
        using size_type = __SIZE_TYPE__;
        inline constexpr size_type dynamic = ~size_type{0};
        template <size_type... Extents>
        struct [[loom::type("shape")]] shape {};
        template <class Shape, class T>
        struct [[loom::type("tensor")]] tensor { T* data; };
        } }
        using GnuWords = unsigned __attribute__((vector_size(16)));
        using ExtBytes = unsigned char __attribute__((ext_vector_type(16)));
        using Static = loom::type::tensor<loom::type::shape<7>, GnuWords>;
        using Dynamic = loom::type::tensor<
            loom::type::shape<loom::type::dynamic>, ExtBytes>;
      )"),
                IREE_SV("vector_tensors.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  const auto& static_tensor = static_cast<const TensorPartition&>(
      types.partition(source_type("Static"), owner));
  EXPECT_EQ(static_tensor.component_count, 1u);
  EXPECT_EQ(static_tensor.vector_lane_count, 4u);
  EXPECT_EQ(static_tensor.element_bytes, 16u);
  EXPECT_EQ(types.unqualified(static_tensor.lane_type),
            source.unit().control()->getUnsignedIntType());
  std::vector<loom_type_t> static_types;
  types.append(source_type("Static"), owner, static_types);
  ASSERT_EQ(static_types.size(), 1u);
  EXPECT_EQ(loom_type_kind(static_types[0]), LOOM_TYPE_TENSOR);
  EXPECT_EQ(loom_type_rank(static_types[0]), 2u);
  EXPECT_EQ(loom_type_dim_static_size_at(static_types[0], 0), 7);
  EXPECT_EQ(loom_type_dim_static_size_at(static_types[0], 1), 4);
  EXPECT_EQ(loom_type_element_type(static_types[0]), LOOM_SCALAR_TYPE_I32);

  const auto& dynamic_tensor = static_cast<const TensorPartition&>(
      types.partition(source_type("Dynamic"), owner));
  EXPECT_EQ(dynamic_tensor.component_count, 2u);
  EXPECT_EQ(dynamic_tensor.vector_lane_count, 16u);
  EXPECT_EQ(dynamic_tensor.element_bytes, 16u);
  EXPECT_EQ(types.unqualified(dynamic_tensor.lane_type),
            source.unit().control()->getUnsignedCharType());
  const loom_value_id_t dynamic_ids[] = {41, 42};
  BoundTypeStorage dynamic_storage;
  std::vector<loom_type_t> dynamic_types;
  types.append_bound(source_type("Dynamic"), owner, dynamic_ids,
                     dynamic_storage, dynamic_types);
  ASSERT_EQ(dynamic_types.size(), 2u);
  EXPECT_EQ(loom_type_element_type(dynamic_types[0]), LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(loom_type_kind(dynamic_types[1]), LOOM_TYPE_TENSOR);
  EXPECT_EQ(loom_type_rank(dynamic_types[1]), 2u);
  EXPECT_EQ(loom_type_dim_value_id_at(dynamic_types[1], 0), 41u);
  EXPECT_EQ(loom_type_dim_static_size_at(dynamic_types[1], 1), 16);
  EXPECT_EQ(loom_type_element_type(dynamic_types[1]), LOOM_SCALAR_TYPE_I8);
}

TEST(TypesTest, ProjectsEverySupportedViewRankAndEncodingRole) {
  std::string source_text = R"(
    namespace loom { namespace encoding {
    enum class role { layout, schema, storage, transform };
    } namespace type {
    using size_type = __SIZE_TYPE__;
    inline constexpr size_type dynamic = ~size_type{0};
    template <loom::encoding::role Role, size_type Rank>
    struct [[loom::type("encoding")]] encoding { size_type values[Rank + 1]; };
    template <size_type... Extents>
    struct [[loom::type("shape")]] shape {};
    template <class Shape, class T, class Encoding = void>
    struct [[loom::type("view")]] view { T* data; };
    } }
    using Layout = loom::type::encoding<loom::encoding::role::layout, 15>;
    using Schema = loom::type::encoding<loom::encoding::role::schema, 0>;
    using Storage = loom::type::encoding<loom::encoding::role::storage, 15>;
    using Transform = loom::type::encoding<loom::encoding::role::transform, 0>;
    using StaticVolume = loom::type::view<
        loom::type::shape<2, 3, 5>, const float>;
  )";
  for (size_t rank = 1; rank <= LOOM_TYPE_MAX_RANK; ++rank) {
    source_text += "using Rank" + std::to_string(rank) +
                   " = loom::type::view<loom::type::shape<"
                   "loom::type::dynamic";
    for (size_t axis = 1; axis < rank; ++axis) {
      source_text += ", " + std::to_string(axis + 1);
    }
    source_text += ">, const float, loom::type::encoding<";
    source_text += rank % 2 ? "loom::encoding::role::layout, "
                            : "loom::encoding::role::storage, ";
    source_text += std::to_string(rank) + ">>;\n";
  }

  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(iree_make_string_view(source_text.data(), source_text.size()),
                IREE_SV("view_family.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const std::string& name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  struct EncodingCase {
    // Source alias naming the encoding specialization.
    const char* name;
    // Expected High encoding role.
    loom_encoding_role_t role;
    // Expected logical rank.
    size_t rank;
  };
  for (const auto& test : {
           EncodingCase{"Layout", LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, 15},
           EncodingCase{"Schema", LOOM_ENCODING_ROLE_STORAGE_SCHEMA, 0},
           EncodingCase{"Storage", LOOM_ENCODING_ROLE_PHYSICAL_STORAGE, 15},
           EncodingCase{"Transform", LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM, 0},
       }) {
    const auto& encoding = static_cast<const EncodingPartition&>(
        types.partition(source_type(test.name), owner));
    EXPECT_EQ(encoding.role, test.role);
    EXPECT_EQ(encoding.rank, test.rank);
  }

  std::vector<loom_type_t> static_volume_types;
  types.append(source_type("StaticVolume"), owner, static_volume_types);
  ASSERT_EQ(static_volume_types.size(), 1u);
  EXPECT_EQ(loom_type_rank(static_volume_types[0]), 3u);
  EXPECT_EQ(loom_type_dim_static_size_at(static_volume_types[0], 0), 2);
  EXPECT_EQ(loom_type_dim_static_size_at(static_volume_types[0], 1), 3);
  EXPECT_EQ(loom_type_dim_static_size_at(static_volume_types[0], 2), 5);

  for (size_t rank = 1; rank <= LOOM_TYPE_MAX_RANK; ++rank) {
    auto* input = source_type("Rank" + std::to_string(rank));
    const auto& view =
        static_cast<const ViewPartition&>(types.partition(input, owner));
    ASSERT_EQ(view.extents.size(), rank);
    ASSERT_NE(view.encoding, nullptr);
    EXPECT_EQ(view.encoding->role, rank % 2
                                       ? LOOM_ENCODING_ROLE_ADDRESS_LAYOUT
                                       : LOOM_ENCODING_ROLE_PHYSICAL_STORAGE);
    EXPECT_EQ(view.extents[0], -1);
    for (size_t axis = 1; axis < rank; ++axis) {
      EXPECT_EQ(view.extents[axis], static_cast<int64_t>(axis + 1));
    }

    const loom_value_id_t first_id = 100 + rank * 4;
    const loom_value_id_t identities[] = {first_id, first_id + 1, first_id + 2};
    BoundTypeStorage storage;
    std::vector<loom_type_t> projected;
    types.append_bound(input, owner, identities, storage, projected);
    ASSERT_EQ(projected.size(), 3u);
    EXPECT_EQ(loom_type_encoding_role(projected[1]), view.encoding->role);
    EXPECT_EQ(loom_type_rank(projected[2]), rank);
    EXPECT_EQ(loom_type_dim_value_id_at(projected[2], 0), first_id);
    for (size_t axis = 1; axis < rank; ++axis) {
      EXPECT_EQ(loom_type_dim_static_size_at(projected[2], axis),
                static_cast<int64_t>(axis + 1));
    }
    EXPECT_EQ(loom_type_encoding_value_id(projected[2]), first_id + 1);
  }
}

TEST(TypesTest, RejectsInvalidViewRanksAndEncodingRoles) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    namespace loom { namespace encoding {
    enum class role { layout, schema, storage, transform };
    } namespace type {
    using size_type = __SIZE_TYPE__;
    template <loom::encoding::role Role, size_type Rank>
    struct [[loom::type("encoding")]] encoding { size_type value; };
    template <size_type... Extents>
    struct [[loom::type("shape")]] shape {};
    template <class Shape, class T, class Encoding = void>
    struct [[loom::type("view")]] view { T* data; };
    } }
    using RankZero = loom::type::view<loom::type::shape<>, float>;
    using RankSixteen = loom::type::view<loom::type::shape<
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>, float>;
    using SchemaView = loom::type::view<loom::type::shape<4>, float,
        loom::type::encoding<loom::encoding::role::schema, 0>>;
    using TransformView = loom::type::view<loom::type::shape<4>, float,
        loom::type::encoding<loom::encoding::role::transform, 0>>;
    using MismatchedLayout = loom::type::view<loom::type::shape<4>, float,
        loom::type::encoding<loom::encoding::role::layout, 2>>;
    using ScalarEncoding = loom::type::view<loom::type::shape<4>, float, int>;
    using LayoutZero = loom::type::encoding<loom::encoding::role::layout, 0>;
    using StorageZero = loom::type::encoding<loom::encoding::role::storage, 0>;
    using SchemaRanked = loom::type::encoding<loom::encoding::role::schema, 1>;
    using TransformRanked = loom::type::encoding<
        loom::encoding::role::transform, 1>;
  )"),
                IREE_SV("invalid_views.cpp"), options);
  Types types(source.unit(), source.diagnostics());
  auto* owner = source.unit().ast();
  auto source_type = [&](const char* name) {
    return (*source.unit().globalScope()->find(name).begin())->type();
  };

  for (const char* name :
       {"RankZero", "RankSixteen", "SchemaView", "TransformView",
        "MismatchedLayout", "ScalarEncoding", "LayoutZero", "StorageZero",
        "SchemaRanked", "TransformRanked"}) {
    EXPECT_THROW(types.partition(source_type(name), owner), SourceRejected);
  }
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
