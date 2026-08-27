// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/registers.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(RegistersTest, CarrierPropertiesAreIndependent) {
  const loom_type_t lhs = loom_low_register_type(
      /*descriptor_set_stable_id=*/42, /*register_class_id=*/3,
      /*unit_count=*/2);
  const loom_type_t same_class = loom_low_register_type(
      /*descriptor_set_stable_id=*/42, /*register_class_id=*/3,
      /*unit_count=*/4);
  const loom_type_t same_unit_count = loom_low_register_type(
      /*descriptor_set_stable_id=*/42, /*register_class_id=*/7,
      /*unit_count=*/2);

  EXPECT_TRUE(loom_low_register_type_same_class(lhs, same_class));
  EXPECT_FALSE(loom_low_register_type_same_unit_count(lhs, same_class));
  EXPECT_FALSE(loom_low_register_type_same_class(lhs, same_unit_count));
  EXPECT_TRUE(loom_low_register_type_same_unit_count(lhs, same_unit_count));
}

TEST(RegistersTest, CarrierProjectionDiscardsSemanticValueType) {
  const loom_type_t value_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_register_type_data_t register_data = {
      /*.carrier_payload0=*/42,
      /*.carrier_payload1=*/
      loom_low_register_type_pack_payload1(
          /*register_class_id=*/3, /*unit_count=*/2),
      /*.value_type=*/value_type,
  };
  const loom_type_t typed_type =
      loom_type_register_payload_with_value_type(&register_data);

  const loom_type_t carrier_type =
      loom_low_register_carrier_type_with_unit_count(typed_type,
                                                     /*unit_count=*/4);
  EXPECT_TRUE(loom_low_register_type_same_class(typed_type, carrier_type));
  EXPECT_EQ(loom_low_register_type_unit_count(carrier_type), 4u);
  EXPECT_FALSE(loom_type_register_has_value_type(carrier_type));
}

}  // namespace
}  // namespace loom
