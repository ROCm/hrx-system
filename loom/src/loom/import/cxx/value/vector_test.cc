// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/vector.h"

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {
namespace {
using VectorsTest = ValueBuilderTest;

TEST_F(VectorsTest, NarrowUnsignedLanesAndFullWidthComparisonMasks) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Vectors vectors(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* byte = control->getUnsignedCharType();
  auto* source_type = control->getVectorType(byte, 16, cxx::VectorKind::kGnu);
  auto* mask_type = control->getVectorType(control->getSignedCharType(), 16,
                                           cxx::VectorKind::kGnu);
  auto left = vectors.convert(scalars.integer(-1, LOOM_SCALAR_TYPE_I8), byte,
                              source_type, owner);
  auto right = vectors.convert(scalars.integer(1, LOOM_SCALAR_TYPE_I8), byte,
                               source_type, owner);
  auto sum = vectors.binary(cxx::TokenKind::T_PLUS, left, right, source_type,
                            source_type, owner);
  EXPECT_TRUE(loom_vector_addi_isa(producer(sum)));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, sum),
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, 16, 0)));
  auto mask = vectors.binary(cxx::TokenKind::T_LESS, left, right, source_type,
                             mask_type, owner);
  auto* extension = producer(mask);
  ASSERT_TRUE(loom_vector_extsi_isa(extension));
  auto* comparison = producer(loom_vector_extsi_input(extension));
  ASSERT_TRUE(loom_vector_cmpi_isa(comparison));
  EXPECT_EQ(loom_vector_cmpi_predicate(comparison),
            LOOM_VECTOR_CMPI_PREDICATE_ULT);
}

TEST_F(VectorsTest, VectorConversionsReinterpretBitsInsteadOfNumericLanes) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Vectors vectors(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* input =
      control->getVectorType(control->getIntType(), 4, cxx::VectorKind::kGnu);
  auto* output =
      control->getVectorType(control->getFloatType(), 4, cxx::VectorKind::kGnu);
  auto value =
      vectors.convert(scalars.integer(0x3f800000, LOOM_SCALAR_TYPE_I32),
                      control->getIntType(), input, owner);
  auto converted = vectors.convert(value, input, output, owner);
  EXPECT_TRUE(loom_vector_bitcast_isa(producer(converted)));
}

TEST_F(VectorsTest, VectorReinterpretationUsesLogicalValueBits) {
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  Vectors vectors(source_.unit(), source_.diagnostics(), types_, scalars,
                  locations, builder_);
  auto* control = source_.unit().control();
  auto* owner = source_.unit().ast();
  auto* padded = control->getVectorType(control->getFloat16Type(), 3,
                                        cxx::VectorKind::kExt);
  auto* words = control->getVectorType(control->getUnsignedShortIntType(), 3,
                                       cxx::VectorKind::kExt);
  auto* bits = control->getVectorType(control->getUnsignedCharType(), 8,
                                      cxx::VectorKind::kExt);
  auto value =
      vectors.convert(scalars.integer(0, LOOM_SCALAR_TYPE_I16),
                      control->getUnsignedShortIntType(), padded, owner);
  auto converted = vectors.convert(value, padded, words, owner);
  EXPECT_TRUE(loom_vector_bitcast_isa(producer(converted)));
  EXPECT_THROW(vectors.convert(value, padded, bits, owner), SourceRejected);
}

}  // namespace
}  // namespace loom::cxx_import
