// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/storage.h"

#include <cxx/token.h>

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

TEST_F(StorageTest, RetainsArrayShapeAndExplicitAlignment) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* array_type = control->getBoundedArrayType(control->getFloatType(), 64);
  auto allocation = storage.workgroup(array_type, 64, owner);
  EXPECT_EQ(
      loom_buffer_alloca_base_alignment(producer(allocation.pointer.root)), 64);
  auto index = scalars.integer(17, LOOM_SCALAR_TYPE_I32);
  auto access = storage.subscript(allocation.pointer, index, array_type,
                                  control->getUnsignedIntType(), owner);
  EXPECT_EQ(access.view, allocation.view);
  ASSERT_TRUE(access.index.has_value());
  EXPECT_TRUE(loom_index_cast_isa(producer(*access.index)));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, access.view),
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, 64, 0)));
}

TEST_F(StorageTest, InteriorPointersRetainSignedDisplacementsAndRootIdentity) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto allocation = storage.workgroup(
      control->getBoundedArrayType(control->getFloatType(), 64), 0, owner);
  auto* pointer_type = control->getPointerType(control->getFloatType());
  auto interior = storage.advance(allocation.pointer,
                                  scalars.integer(17, LOOM_SCALAR_TYPE_I64),
                                  pointer_type, control->getLongLongIntType(),
                                  cxx::TokenKind::T_PLUS, owner);
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
  Value second(interior);
  std::vector<loom_value_id_t> arguments;
  first.append_to(arguments);
  second.append_to(arguments);
  ValueArena values;
  Value restored =
      values.capture(kPointerPartition,
                     std::span<const loom_value_id_t>(arguments).subspan(2));
  EXPECT_EQ(restored.pointer().root, first.pointer().root);
  EXPECT_EQ(restored.pointer().byte_offset, interior.byte_offset);
  EXPECT_NE(restored.pointer().byte_offset, first.pointer().byte_offset);
}

TEST_F(StorageTest, UnsignedDisplacementsExtendBeforeScaling) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto allocation = storage.workgroup(
      control->getBoundedArrayType(control->getFloatType(), 64), 0, owner);
  auto* pointer = control->getPointerType(control->getFloatType());
  auto advanced = storage.advance(
      allocation.pointer, scalars.integer(17, LOOM_SCALAR_TYPE_I32), pointer,
      control->getUnsignedIntType(), cxx::TokenKind::T_PLUS, owner);
  auto* assumed_origin =
      producer(loom_op_operands(producer(advanced.byte_offset))[0]);
  auto* sum = producer(loom_op_operands(assumed_origin)[0]);
  auto* scale = producer(loom_op_operands(sum)[1]);
  EXPECT_TRUE(loom_scalar_extui_isa(producer(loom_op_operands(scale)[0])));
  auto invalid = scalars.integer(1, LOOM_SCALAR_TYPE_I32);
  EXPECT_THROW(
      storage.advance(allocation.pointer, invalid, pointer,
                      control->getIntType(), cxx::TokenKind::T_STAR, owner),
      SourceRejected);
}

TEST_F(StorageTest, WideArrayIndicesRetainDeclaredBounds) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Storage storage(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* array = control->getBoundedArrayType(control->getIntType(), 64);
  auto allocation = storage.workgroup(array, 0, owner);
  auto input = scalars.integer(63, LOOM_SCALAR_TYPE_I64);
  auto access = storage.subscript(allocation.pointer, input, array,
                                  control->getUnsignedLongLongIntType(), owner);
  ASSERT_TRUE(access.index.has_value());
  auto* offset = producer(loom_index_cast_input(producer(*access.index)));
  ASSERT_TRUE(loom_index_cast_isa(offset));
  auto* assumed_index = producer(loom_index_cast_input(offset));
  ASSERT_TRUE(loom_scalar_assume_isa(assumed_index));
  auto predicates = loom_scalar_assume_predicates(assumed_index);
  ASSERT_EQ(predicates.count, 1u);
  EXPECT_EQ(predicates.predicate_list[0].kind, LOOM_PREDICATE_RANGE);
  EXPECT_EQ(predicates.predicate_list[0].args[0], input);
  EXPECT_EQ(predicates.predicate_list[0].args[1], 0);
  EXPECT_EQ(predicates.predicate_list[0].args[2], 63);
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
  auto allocation = storage.workgroup(array, 0, owner);
  auto access = storage.subscript(allocation.pointer,
                                  scalars.integer(17, LOOM_SCALAR_TYPE_I32),
                                  array, control->getUnsignedIntType(), owner);
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
      storage.workgroup(control->getBoundedArrayType(integer, 4), 16, owner);
  auto* vector = control->getQualType(
      control->getVectorType(integer, 4, cxx::VectorKind::kGnu),
      cxx::CvQualifiers::kVolatile);
  auto access = storage.dereference(allocation.pointer, vector, owner);
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
