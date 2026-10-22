// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/bitcast.h"

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {
namespace {
using BitCastTest = ValueBuilderTest;

TEST_F(BitCastTest, ScalarEndpointsPreserveTheOperandAndStaticLane) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* word = control->getUnsignedIntType();
  auto* bytes = control->getVectorType(control->getUnsignedCharType(), 4,
                                       cxx::VectorKind::kExt);
  BitCast unpack(source_.unit(), source_.diagnostics(), types_, word, bytes,
                 owner);
  BitCast pack(source_.unit(), source_.diagnostics(), types_, bytes, word,
               owner);
  auto input = scalars.integer(0x76543210, LOOM_SCALAR_TYPE_I32);
  auto lanes = unpack.build(builder_, input, LOOM_LOCATION_UNKNOWN);
  auto* unpack_op = producer(lanes);
  ASSERT_TRUE(loom_vector_bitcast_isa(unpack_op));
  auto* singleton = producer(loom_vector_bitcast_input(unpack_op));
  ASSERT_TRUE(loom_vector_from_elements_isa(singleton));
  EXPECT_EQ(loom_op_operands(singleton)[0], input);

  auto result = pack.build(builder_, lanes, LOOM_LOCATION_UNKNOWN);
  auto* extract = producer(result);
  ASSERT_TRUE(loom_vector_extract_isa(extract));
  EXPECT_EQ(loom_vector_extract_indices(extract).count, 0u);
  EXPECT_EQ(loom_vector_extract_static_indices(extract).count, 1u);
  EXPECT_EQ(loom_vector_extract_static_indices(extract).i64_array[0], 0);
  EXPECT_TRUE(
      loom_vector_bitcast_isa(producer(loom_vector_extract_source(extract))));
  EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, result),
                              loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
}

TEST_F(BitCastTest, IdenticalRepresentationsReuseTheOriginalValue) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  auto* control = source_.unit().control();
  BitCast cast(source_.unit(), source_.diagnostics(), types_,
               control->getIntType(), control->getUnsignedIntType(),
               source_.unit().ast());
  auto input = scalars.integer(-1, LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(cast.build(builder_, input, LOOM_LOCATION_UNKNOWN), input);
}

TEST_F(BitCastTest, PaddedVectorObjectBytesRemainOutsideSSAValues) {
  auto* control = source_.unit().control();
  auto* padded = control->getVectorType(control->getFloat16Type(), 3,
                                        cxx::VectorKind::kExt);
  EXPECT_THROW(
      BitCast(source_.unit(), source_.diagnostics(), types_, padded,
              control->getUnsignedLongLongIntType(), source_.unit().ast()),
      SourceRejected);
}

}  // namespace
}  // namespace loom::cxx_import
