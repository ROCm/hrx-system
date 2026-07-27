// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/register_classes.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/x86/descriptors/avx10_2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_bf16_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_packed_dot_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_vnni_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_int16_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_int8_descriptors.h"
#include "loom/target/arch/x86/descriptors/packed_dot_descriptors.h"
#include "loom/target/arch/x86/descriptors/scalar_descriptors.h"
#include "loom/target/arch/x86/descriptors/simd128_descriptors.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

void ExpectDescriptorClass(const loom_low_descriptor_set_t* descriptor_set,
                           loom_x86_register_class_t register_class) {
  uint16_t descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_ASSERT_OK(loom_x86_descriptor_set_register_class_id(
      descriptor_set, register_class, &descriptor_reg_class_id));

  loom_x86_register_class_t resolved_register_class =
      LOOM_X86_REGISTER_CLASS_GPR32;
  IREE_ASSERT_OK(loom_x86_descriptor_set_logical_register_class(
      descriptor_set, descriptor_reg_class_id, &resolved_register_class));
  EXPECT_EQ(resolved_register_class, register_class);

  iree_string_view_t expected_name = iree_string_view_empty();
  IREE_ASSERT_OK(loom_x86_register_class_name(register_class, &expected_name));
  ASSERT_LT(descriptor_reg_class_id, descriptor_set->reg_class_count);
  const loom_low_reg_class_t& descriptor_reg_class =
      descriptor_set->reg_classes[descriptor_reg_class_id];
  EXPECT_EQ(ToString(loom_low_descriptor_set_string(
                descriptor_set, descriptor_reg_class.name_string_offset)),
            ToString(expected_name));
}

void ExpectDescriptorPresent(const loom_low_descriptor_set_t* descriptor_set,
                             iree_string_view_t descriptor_key) {
  EXPECT_NE(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, descriptor_key),
      LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
      << ToString(descriptor_key);
}

void ExpectDescriptorMissing(const loom_low_descriptor_set_t* descriptor_set,
                             iree_string_view_t descriptor_key) {
  EXPECT_EQ(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, descriptor_key),
      LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
      << ToString(descriptor_key);
}

void ExpectOperandAddressMap(const loom_low_descriptor_set_t* descriptor_set,
                             iree_string_view_t descriptor_key,
                             iree_string_view_t field_name,
                             loom_low_operand_address_map_kind_t expected_kind,
                             uint32_t expected_addressable_unit_count) {
  const uint16_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, descriptor_key);
  ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
      << ToString(descriptor_key);
  ASSERT_LT(descriptor_ordinal, descriptor_set->descriptor_count);
  const loom_low_descriptor_t& descriptor =
      descriptor_set->descriptors[descriptor_ordinal];
  ASSERT_LE((uint64_t)descriptor.operand_start + descriptor.operand_count,
            descriptor_set->operand_count);
  for (uint16_t i = 0; i < descriptor.operand_count; ++i) {
    const loom_low_operand_t& operand =
        descriptor_set->operands[descriptor.operand_start + i];
    if (!iree_string_view_equal(
            loom_low_descriptor_set_string(descriptor_set,
                                           operand.field_name_string_offset),
            field_name)) {
      continue;
    }
    EXPECT_EQ(operand.address_map_kind, expected_kind)
        << ToString(descriptor_key) << " " << ToString(field_name);
    EXPECT_EQ(operand.addressable_unit_count, expected_addressable_unit_count)
        << ToString(descriptor_key) << " " << ToString(field_name);
    return;
  }
  ADD_FAILURE() << "descriptor " << ToString(descriptor_key)
                << " has no operand field " << ToString(field_name);
}

TEST(X86RegisterClassesTest, SharedScalarClassesAcrossViews) {
  const loom_low_descriptor_set_t* scalar_descriptor_set =
      loom_x86_scalar_core_descriptor_set();
  const loom_low_descriptor_set_t* simd128_descriptor_set =
      loom_x86_simd128_core_descriptor_set();
  const loom_low_descriptor_set_t* avx2_descriptor_set =
      loom_x86_avx2_core_descriptor_set();
  const loom_low_descriptor_set_t* avx512_descriptor_set =
      loom_x86_avx512_core_descriptor_set();

  ExpectDescriptorClass(scalar_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR32);
  ExpectDescriptorClass(scalar_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  ExpectDescriptorClass(simd128_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR32);
  ExpectDescriptorClass(simd128_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  ExpectDescriptorClass(avx2_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR32);
  ExpectDescriptorClass(avx2_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR32);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64);
}

TEST(X86RegisterClassesTest, VectorClassesAcrossProfileViews) {
  const loom_low_descriptor_set_t* simd128_descriptor_set =
      loom_x86_simd128_core_descriptor_set();
  const loom_low_descriptor_set_t* avx2_descriptor_set =
      loom_x86_avx2_core_descriptor_set();
  const loom_low_descriptor_set_t* avx512_descriptor_set =
      loom_x86_avx512_core_descriptor_set();
  const loom_low_descriptor_set_t* packed_dot_descriptor_set =
      loom_x86_packed_dot_core_descriptor_set();
  const loom_low_descriptor_set_t* avx512_packed_dot_descriptor_set =
      loom_x86_avx512_packed_dot_core_descriptor_set();

  ExpectDescriptorClass(simd128_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);

  ExpectDescriptorClass(avx2_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx2_descriptor_set, LOOM_X86_REGISTER_CLASS_YMM);

  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_ZMM);
  ExpectDescriptorClass(avx512_descriptor_set, LOOM_X86_REGISTER_CLASS_K);

  ExpectDescriptorClass(packed_dot_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(packed_dot_descriptor_set, LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorClass(packed_dot_descriptor_set, LOOM_X86_REGISTER_CLASS_ZMM);

  ExpectDescriptorClass(avx512_packed_dot_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx512_packed_dot_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorClass(avx512_packed_dot_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_ZMM);
}

TEST(X86RegisterClassesTest, VexRowsImportedIntoWideViewsStayLow16) {
  const loom_low_descriptor_set_t* avx512_descriptor_set =
      loom_x86_avx512_core_descriptor_set();
  ExpectOperandAddressMap(avx512_descriptor_set, IREE_SV("x86.avx2.vpaddd.xmm"),
                          IREE_SV("dst"),
                          LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET, 16);
  ExpectOperandAddressMap(avx512_descriptor_set, IREE_SV("x86.avx2.vpaddd.xmm"),
                          IREE_SV("lhs"),
                          LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET, 16);
  ExpectOperandAddressMap(avx512_descriptor_set, IREE_SV("x86.avx2.vpaddd.xmm"),
                          IREE_SV("rhs"),
                          LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET, 16);
  ExpectOperandAddressMap(avx512_descriptor_set,
                          IREE_SV("x86.avx512.vpaddd.zmm"), IREE_SV("dst"),
                          LOOM_LOW_OPERAND_ADDRESS_MAP_DIRECT, 0);

  const loom_low_descriptor_set_t* avx512_packed_dot_descriptor_set =
      loom_x86_avx512_packed_dot_core_descriptor_set();
  ExpectOperandAddressMap(avx512_packed_dot_descriptor_set,
                          IREE_SV("x86.avx_vnni.vpdpbusd.ymm"), IREE_SV("dst"),
                          LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET, 16);
  ExpectOperandAddressMap(
      avx512_packed_dot_descriptor_set, IREE_SV("x86.avx512_vnni.vpdpbusd.zmm"),
      IREE_SV("dst"), LOOM_LOW_OPERAND_ADDRESS_MAP_DIRECT, 0);
}

TEST(X86RegisterClassesTest,
     PackedDotFeatureViewsExposeTheirDescriptorFamilies) {
  const loom_low_descriptor_set_t* avx512_vnni_descriptor_set =
      loom_x86_avx512_vnni_core_descriptor_set();
  const loom_low_descriptor_set_t* avx512_bf16_descriptor_set =
      loom_x86_avx512_bf16_core_descriptor_set();
  const loom_low_descriptor_set_t* avx_vnni_descriptor_set =
      loom_x86_avx_vnni_core_descriptor_set();
  const loom_low_descriptor_set_t* avx_vnni_int8_descriptor_set =
      loom_x86_avx_vnni_int8_core_descriptor_set();
  const loom_low_descriptor_set_t* avx_vnni_int16_descriptor_set =
      loom_x86_avx_vnni_int16_core_descriptor_set();
  const loom_low_descriptor_set_t* avx10_2_descriptor_set =
      loom_x86_avx10_2_core_descriptor_set();

  ExpectDescriptorClass(avx512_vnni_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx512_vnni_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorClass(avx512_vnni_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_ZMM);
  ExpectDescriptorPresent(avx512_vnni_descriptor_set,
                          IREE_SV("x86.avx512_vnni.vpdpbusd.zmm"));
  ExpectDescriptorClass(avx512_bf16_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_ZMM);
  ExpectDescriptorPresent(avx512_bf16_descriptor_set,
                          IREE_SV("x86.avx512_bf16.vdpbf16ps.zmm"));

  ExpectDescriptorClass(avx_vnni_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx_vnni_descriptor_set, LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorPresent(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx_vnni.vpdpbusd.ymm"));
  ExpectDescriptorMissing(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx512.vaddps.zmm"));
  ExpectDescriptorMissing(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx512_vnni.vpdpbusd.zmm"));
  ExpectDescriptorMissing(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx10_2.vpdpbssd.zmm"));
  ExpectDescriptorClass(avx_vnni_int8_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorPresent(avx_vnni_int8_descriptor_set,
                          IREE_SV("x86.avx_vnni_int8.vpdpbssd.ymm"));
  ExpectDescriptorMissing(avx_vnni_int8_descriptor_set,
                          IREE_SV("x86.avx10_2.vpdpbssd.zmm"));
  ExpectDescriptorClass(avx_vnni_int16_descriptor_set,
                        LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorPresent(avx_vnni_int16_descriptor_set,
                          IREE_SV("x86.avx_vnni_int16.vpdpwsud.ymm"));

  ExpectDescriptorClass(avx10_2_descriptor_set, LOOM_X86_REGISTER_CLASS_XMM);
  ExpectDescriptorClass(avx10_2_descriptor_set, LOOM_X86_REGISTER_CLASS_YMM);
  ExpectDescriptorClass(avx10_2_descriptor_set, LOOM_X86_REGISTER_CLASS_ZMM);
  ExpectDescriptorPresent(avx10_2_descriptor_set,
                          IREE_SV("x86.avx10_2.vpdpbssd.zmm"));
}

TEST(X86RegisterClassesTest, VectorWidthProjection) {
  loom_x86_register_class_t register_class = LOOM_X86_REGISTER_CLASS_GPR32;
  EXPECT_TRUE(
      loom_x86_register_class_for_vector_bit_width(128, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_XMM);
  EXPECT_TRUE(
      loom_x86_register_class_for_vector_bit_width(256, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_YMM);
  EXPECT_TRUE(
      loom_x86_register_class_for_vector_bit_width(512, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_ZMM);
  EXPECT_FALSE(
      loom_x86_register_class_for_vector_bit_width(64, &register_class));
  EXPECT_EQ(register_class, LOOM_X86_REGISTER_CLASS_GPR32);
}

TEST(X86RegisterClassesTest, InvalidProjectionFailsLoudly) {
  iree_string_view_t register_class_name = iree_string_view_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_x86_register_class_name((loom_x86_register_class_t)99,
                                   &register_class_name));

  loom_x86_register_class_t register_class = LOOM_X86_REGISTER_CLASS_GPR32;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_x86_descriptor_set_logical_register_class(
          loom_x86_scalar_core_descriptor_set(), UINT16_MAX, &register_class));
}

}  // namespace
}  // namespace loom
