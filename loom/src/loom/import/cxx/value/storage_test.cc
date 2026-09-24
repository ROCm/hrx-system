// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/storage.h"

#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/views/symbol_chain.h>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/builder_test.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {
using StorageTest = ValueBuilderTest;

TEST_F(StorageTest, MemberProjectionRetainsRootAndNestedSourceOffsets) {
  Source source(IREE_SV("struct Inner { char tag; float value; }; "
                        "struct Outer { unsigned prefix; Inner inner; };"),
                IREE_SV("storage_records.cpp"), options());
  Types types(source.unit(), source.diagnostics());
  Locations locations(source.unit(), source.diagnostics(), module_);
  Scalars scalars(source.unit(), source.diagnostics(), types, locations,
                  builder_);
  Storage storage(source.unit(), source.diagnostics(), types, scalars,
                  locations, builder_);
  auto* owner = source.unit().ast();
  auto* outer = cxx::symbol_cast<cxx::ClassSymbol>(
      *source.unit().globalScope()->find("Outer").begin());
  ASSERT_NE(outer, nullptr);
  EXPECT_EQ(types.storage_size(outer->type(), owner), 12);
  auto* inner_field =
      cxx::symbol_cast<cxx::FieldSymbol>(*outer->find("inner").begin());
  ASSERT_NE(inner_field, nullptr);
  auto* inner =
      cxx::type_cast<cxx::ClassType>(inner_field->type())->definition();
  auto* value_field =
      cxx::symbol_cast<cxx::FieldSymbol>(*inner->find("value").begin());
  ASSERT_NE(value_field, nullptr);
  auto allocation =
      storage.allocate(source.unit().control()->getBoundedArrayType(
                           source.unit().control()->getUnsignedCharType(), 64),
                       LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 4, owner);
  auto parent =
      storage.member(storage.project(allocation.pointer, outer->type(), owner),
                     inner_field, owner);
  auto field = storage.member(parent, value_field, owner);
  EXPECT_EQ(parent.pointer.root, allocation.pointer.root);
  EXPECT_EQ(field.pointer.root, allocation.pointer.root);
  EXPECT_EQ(parent.alignment, 4u);
  EXPECT_EQ(field.alignment, 4u);
  auto* offset = producer(field.pointer.byte_offset);
  ASSERT_TRUE(loom_index_add_isa(offset));
  EXPECT_EQ(loom_op_operands(offset)[0], parent.pointer.byte_offset);
  EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(
                producer(loom_op_operands(offset)[1]))),
            4);
  auto access = storage.dereference(field, value_field->type(), owner);
  EXPECT_EQ(loom_buffer_view_buffer(producer(access.view)), field.pointer.root);
  EXPECT_EQ(loom_buffer_view_byte_offset(producer(access.view)),
            field.pointer.byte_offset);
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, access.view),
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, 1, 0)));
}

TEST_F(StorageTest, RetainsArrayShapeAndExplicitAlignment) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* array_type = control->getBoundedArrayType(control->getFloatType(), 64);
  auto allocation = storage.allocate(
      array_type, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 64, owner);
  EXPECT_EQ(
      loom_buffer_alloca_base_alignment(producer(allocation.pointer.root)), 64);
  auto index = scalars.integer(17, LOOM_SCALAR_TYPE_I32);
  auto access = storage.subscript(
      storage.project(allocation.pointer, array_type, owner), index, array_type,
      control->getUnsignedIntType(), owner);
  EXPECT_EQ(access.view, allocation.view);
  ASSERT_TRUE(access.index.has_value());
  EXPECT_TRUE(loom_index_cast_isa(producer(*access.index)));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, access.view),
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, 64, 0)));
}

TEST_F(StorageTest, ArrayAliasesRetainTheirElementTypeAndInteriorOrigin) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* bytes =
      control->getBoundedArrayType(control->getUnsignedCharType(), 256);
  auto allocation = storage.allocate(
      bytes, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 16, owner);
  auto* row = control->getBoundedArrayType(control->getFloatType(), 32);
  auto interior = storage.advance(
      storage.project(allocation.pointer, row, owner),
      scalars.integer(1, LOOM_SCALAR_TYPE_I32), control->getPointerType(row),
      control->getIntType(), cxx::TokenKind::T_PLUS, owner);
  auto access =
      storage.subscript(interior, scalars.integer(17, LOOM_SCALAR_TYPE_I32),
                        row, control->getIntType(), owner);
  EXPECT_FALSE(access.index.has_value());
  auto* view = producer(access.view);
  ASSERT_TRUE(loom_buffer_view_isa(view));
  EXPECT_EQ(loom_buffer_view_buffer(view), allocation.pointer.root);
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, access.view),
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, 1, 0)));
  auto* origin = producer(loom_buffer_view_byte_offset(view));
  auto* assumed = producer(loom_index_cast_input(origin));
  auto* sum = producer(loom_op_operands(assumed)[0]);
  EXPECT_EQ(loom_index_cast_input(producer(loom_op_operands(sum)[0])),
            interior.pointer.byte_offset);
}

TEST_F(StorageTest, InteriorPointersRetainSignedDisplacementsAndRootIdentity) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto allocation = storage.allocate(
      control->getBoundedArrayType(control->getFloatType(), 64),
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 0, owner);
  auto* pointer_type = control->getPointerType(control->getFloatType());
  auto interior = storage.advance(
      storage.project(allocation.pointer, control->getFloatType(), owner),
      scalars.integer(17, LOOM_SCALAR_TYPE_I64), pointer_type,
      control->getLongLongIntType(), cxx::TokenKind::T_PLUS, owner);
  auto retreat = scalars.integer(-1, LOOM_SCALAR_TYPE_I32);
  auto access = storage.subscript(interior, retreat, pointer_type,
                                  control->getIntType(), owner);
  EXPECT_FALSE(access.index.has_value());
  auto* view = producer(access.view);
  ASSERT_TRUE(loom_buffer_view_isa(view));
  EXPECT_EQ(loom_buffer_view_buffer(view), allocation.pointer.root);
  auto* offset = producer(loom_buffer_view_byte_offset(view));
  ASSERT_TRUE(loom_index_cast_isa(offset));
  auto* assumed_origin = producer(loom_index_cast_input(offset));
  ASSERT_TRUE(loom_scalar_assume_isa(assumed_origin));
  auto predicates = loom_scalar_assume_predicates(assumed_origin);
  ASSERT_EQ(predicates.count, 1u);
  EXPECT_EQ(predicates.predicate_list[0].kind, LOOM_PREDICATE_RANGE);
  EXPECT_EQ(predicates.predicate_list[0].args[0],
            loom_op_operands(assumed_origin)[0]);
  EXPECT_EQ(predicates.predicate_list[0].args[1], 0);
  EXPECT_EQ(predicates.predicate_list[0].args[2], INT64_MAX);
  auto* sum = producer(loom_op_operands(assumed_origin)[0]);
  ASSERT_TRUE(loom_scalar_addi_isa(sum));
  auto* scale = producer(loom_op_operands(sum)[1]);
  ASSERT_TRUE(loom_scalar_muli_isa(scale));
  auto* extension = producer(loom_op_operands(scale)[0]);
  EXPECT_TRUE(loom_scalar_extsi_isa(extension));
  EXPECT_EQ(loom_op_operands(extension)[0], retreat);
  EXPECT_EQ(loom_type_element_type(
                loom_module_value_type(module_, loom_op_results(scale)[0])),
            LOOM_SCALAR_TYPE_I64);

  // Two aliases share their root but retain independent origins when flattened
  // for calls and control-flow edges.
  Value first(allocation.pointer);
  Value second(interior.pointer);
  std::vector<loom_value_id_t> arguments;
  first.append_to(arguments);
  second.append_to(arguments);
  ValueArena values;
  Value restored =
      values.capture(kPointerPartition,
                     std::span<const loom_value_id_t>(arguments).subspan(2));
  EXPECT_EQ(restored.pointer().root, first.pointer().root);
  EXPECT_EQ(restored.pointer().byte_offset, interior.pointer.byte_offset);
  EXPECT_NE(restored.pointer().byte_offset, first.pointer().byte_offset);
}

TEST_F(StorageTest, ILP32PointersPublishTheirSourceRepresentationRange) {
  auto source_options = options();
  source_options.data_model = LOOM_CXX_DATA_MODEL_ILP32;
  Source source(IREE_SV("int entry();"), IREE_SV("storage_ilp32.cpp"),
                source_options);
  Types types(source.unit(), source.diagnostics());
  Locations locations(source.unit(), source.diagnostics(), module_);
  Scalars scalars(source.unit(), source.diagnostics(), types, locations,
                  builder_);
  Storage storage(source.unit(), source.diagnostics(), types, scalars,
                  locations, builder_);
  auto* control = source.unit().control();
  auto* owner = source.unit().ast();
  auto allocation =
      storage.allocate(control->getBoundedArrayType(control->getIntType(), 64),
                       LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 0, owner);
  auto origin = scalars.integer(16, LOOM_SCALAR_TYPE_OFFSET);
  auto constrained =
      storage.constrain_origin({allocation.pointer.root, origin}, owner);
  EXPECT_EQ(constrained.root, allocation.pointer.root);
  auto* assumed = producer(constrained.byte_offset);
  ASSERT_TRUE(loom_index_assume_isa(assumed));
  auto predicates = loom_index_assume_predicates(assumed);
  ASSERT_EQ(predicates.count, 1u);
  EXPECT_EQ(predicates.predicate_list[0].args[0], loom_op_operands(assumed)[0]);
  EXPECT_EQ(predicates.predicate_list[0].args[1], 0);
  EXPECT_EQ(predicates.predicate_list[0].args[2], UINT32_MAX);
  EXPECT_EQ(loom_op_operands(assumed)[0], origin);

  auto* pointer_type = control->getPointerType(control->getIntType());
  auto advanced =
      storage.advance(storage.project({allocation.pointer.root, origin},
                                      control->getIntType(), owner),
                      scalars.integer(1, LOOM_SCALAR_TYPE_I32), pointer_type,
                      control->getIntType(), cxx::TokenKind::T_PLUS, owner);
  EXPECT_TRUE(advanced.pointer_width_constrained);
  auto* advanced_assume =
      producer(loom_index_cast_input(producer(advanced.pointer.byte_offset)));
  ASSERT_TRUE(loom_scalar_assume_isa(advanced_assume));
  predicates = loom_scalar_assume_predicates(advanced_assume);
  ASSERT_EQ(predicates.count, 1u);
  EXPECT_EQ(predicates.predicate_list[0].args[2], UINT32_MAX);
}

TEST_F(StorageTest, UnsignedDisplacementsExtendBeforeScaling) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto allocation = storage.allocate(
      control->getBoundedArrayType(control->getFloatType(), 64),
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 0, owner);
  auto* pointer = control->getPointerType(control->getFloatType());
  auto advanced = storage.advance(
      storage.project(allocation.pointer, control->getFloatType(), owner),
      scalars.integer(17, LOOM_SCALAR_TYPE_I32), pointer,
      control->getUnsignedIntType(), cxx::TokenKind::T_PLUS, owner);
  auto* assumed_origin =
      producer(loom_op_operands(producer(advanced.pointer.byte_offset))[0]);
  auto* sum = producer(loom_op_operands(assumed_origin)[0]);
  auto* scale = producer(loom_op_operands(sum)[1]);
  EXPECT_TRUE(loom_scalar_extui_isa(producer(loom_op_operands(scale)[0])));
  auto invalid = scalars.integer(1, LOOM_SCALAR_TYPE_I32);
  EXPECT_THROW(storage.advance(storage.project(allocation.pointer,
                                               control->getFloatType(), owner),
                               invalid, pointer, control->getIntType(),
                               cxx::TokenKind::T_STAR, owner),
               SourceRejected);
}

TEST_F(StorageTest, ArrayIndicesRetainDeclaredBounds) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* array = control->getBoundedArrayType(control->getIntType(), 64);
  auto allocation =
      storage.allocate(array, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 0, owner);
  const cxx::Type* index_types[] = {control->getIntType(),
                                    control->getUnsignedIntType(),
                                    control->getUnsignedLongLongIntType()};
  for (auto* type : index_types) {
    auto input =
        scalars.integer(63, loom_type_element_type(types_.get(type, owner)));
    auto access =
        storage.subscript(storage.project(allocation.pointer, array, owner),
                          input, array, type, owner);
    ASSERT_TRUE(access.index.has_value());
    auto* assumed_index =
        producer(loom_index_cast_input(producer(*access.index)));
    ASSERT_TRUE(loom_scalar_assume_isa(assumed_index));
    auto wide_input = loom_op_operands(assumed_index)[0];
    EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, wide_input),
                                loom_type_scalar(LOOM_SCALAR_TYPE_I64)));
    auto predicates = loom_scalar_assume_predicates(assumed_index);
    ASSERT_EQ(predicates.count, 1u);
    EXPECT_EQ(predicates.predicate_list[0].kind, LOOM_PREDICATE_RANGE);
    EXPECT_EQ(predicates.predicate_list[0].args[0], wide_input);
    EXPECT_EQ(predicates.predicate_list[0].args[1], 0);
    EXPECT_EQ(predicates.predicate_list[0].args[2], 63);
  }
}

TEST_F(StorageTest, ResolvedArrayAccessRetainsIndexAndMemoryQualifiers) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* element = control->getQualType(control->getUnsignedIntType(),
                                       cxx::CvQualifiers::kVolatile);
  auto* array = control->getBoundedArrayType(element, 64);
  auto allocation =
      storage.allocate(array, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 0, owner);
  auto access =
      storage.subscript(storage.project(allocation.pointer, array, owner),
                        scalars.integer(17, LOOM_SCALAR_TYPE_I32), array,
                        control->getUnsignedIntType(), owner);
  auto value = storage.load(access, element, owner);
  auto* load = producer(value);
  ASSERT_TRUE(loom_view_load_isa(load));
  EXPECT_EQ(loom_view_load_view(load), allocation.view);
  EXPECT_EQ(loom_view_load_memory_flags(load),
            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  ASSERT_EQ(loom_view_load_indices(load).count, 1u);
  EXPECT_EQ(loom_view_load_indices(load).values[0], *access.index);
  storage.store(access, value, element, owner);
  auto* store = loom_block_const_last_op(loom_module_block(module_));
  ASSERT_TRUE(loom_view_store_isa(store));
  EXPECT_EQ(loom_view_store_view(store), allocation.view);
  EXPECT_EQ(loom_view_store_value(store), value);
  EXPECT_EQ(loom_view_store_memory_flags(store),
            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  ASSERT_EQ(loom_view_store_indices(store).count, 1u);
  EXPECT_EQ(loom_view_store_indices(store).values[0], *access.index);
}

TEST_F(StorageTest, ResolvedVectorAccessPreservesItsFootprint) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* integer = control->getUnsignedIntType();
  auto allocation =
      storage.allocate(control->getBoundedArrayType(integer, 4),
                       LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 16, owner);
  auto* vector = control->getQualType(
      control->getVectorType(integer, 4, cxx::VectorKind::kGnu),
      cxx::CvQualifiers::kVolatile);
  auto access = storage.dereference(
      storage.project(allocation.pointer, vector, owner), vector, owner);
  auto value = storage.load(access, vector, owner);
  auto* load = producer(value);
  ASSERT_TRUE(loom_vector_load_isa(load));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, value),
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, 4, 0)));
  EXPECT_EQ(loom_vector_load_memory_flags(load),
            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  EXPECT_EQ(loom_vector_load_indices(load).count, 0u);
  storage.store(access, value, vector, owner);
  auto* store = loom_block_const_last_op(loom_module_block(module_));
  ASSERT_TRUE(loom_vector_store_isa(store));
  EXPECT_EQ(loom_vector_store_view(store), loom_vector_load_view(load));
  EXPECT_EQ(loom_vector_store_memory_flags(store),
            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
}

}  // namespace
}  // namespace loom::cxx_import
