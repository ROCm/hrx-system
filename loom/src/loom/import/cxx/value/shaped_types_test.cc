// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <string>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/builder_test.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ops/type_builders.h"

namespace loom::cxx_import {
namespace {

using ShapedTypesWithModuleTest = ValueBuilderTest;

TEST_F(ShapedTypesWithModuleTest, ProjectsPipelinePayloadAndHandleFamilies) {
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

  // Both direct operation construction and flattened signatures enforce the
  // encoding representation limit before narrowing a module-local identity.
  const loom_value_id_t maximum_dependencies[] = {11, UINT16_MAX};
  auto maximum_tile =
      dynamic_tile.bind_type(maximum_dependencies, tile_storage);
  EXPECT_EQ(loom_type_encoding_value_id(maximum_tile), UINT16_MAX);
  const loom_value_id_t overflowing_dependencies[] = {11, UINT16_MAX + 1u};
  try {
    dynamic_tile.bind_type(overflowing_dependencies, tile_storage);
    FAIL() << "encoding identity was truncated";
  } catch (StatusError& error) {
    iree::Status status(error.release());
    EXPECT_EQ(status.code(), iree::StatusCode::kResourceExhausted);
  }

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

TEST(ShapedTypesTest, ViewPartitionsPreservePlainAndEncodedSourceTypes) {
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

  const auto& plane_partition = static_cast<const ViewPartition&>(plane);
  const loom_value_id_t maximum_dependencies[] = {11, UINT16_MAX};
  auto maximum_view =
      plane_partition.bind_type(maximum_dependencies, plane_storage);
  EXPECT_EQ(loom_type_encoding_value_id(maximum_view), UINT16_MAX);
  const loom_value_id_t overflowing_dependencies[] = {11, UINT16_MAX + 1u};
  EXPECT_THROW(
      plane_partition.bind_type(overflowing_dependencies, plane_storage),
      StatusError);
  const loom_value_id_t overflowing_ids[] = {11, UINT16_MAX + 1u,
                                             UINT16_MAX + 2u};
  EXPECT_THROW(types.append_bound(source_type("Plane"), owner, overflowing_ids,
                                  plane_storage, plane_types),
               StatusError);

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

TEST(ShapedTypesTest, BindsRuntimeTensorExtentAcrossValueBoundaries) {
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

TEST(ShapedTypesTest, BindsRankGenericSemanticVectorsAndNestedRecords) {
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

TEST(ShapedTypesTest, AppendsVectorLanesToTensorObjectShape) {
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

TEST(ShapedTypesTest, ProjectsEverySupportedViewRankAndEncodingRole) {
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

TEST(ShapedTypesTest, RejectsInvalidViewRanksAndEncodingRoles) {
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

}  // namespace
}  // namespace loom::cxx_import
