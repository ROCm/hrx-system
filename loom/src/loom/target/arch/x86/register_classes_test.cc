// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/register_classes.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
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
                                           operand.field_name_string_ref),
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

TEST(X86RegisterClassesTest, ViewsPreserveRegisterVocabularyAndCapacity) {
  struct Case {
    // Actual generated descriptor view consumed by parsing and allocation.
    const loom_low_descriptor_set_t* descriptor_set;
    // Counts in logical register-class order; zero denotes an absent class.
    uint16_t capacities[6];
  };
  const Case cases[] = {
      {loom_x86_scalar_core_descriptor_set(), {16, 16, 0, 0, 0, 0}},
      {loom_x86_simd128_core_descriptor_set(), {16, 16, 16, 0, 0, 0}},
      {loom_x86_avx2_core_descriptor_set(), {16, 16, 16, 16, 0, 0}},
      {loom_x86_avx512_core_descriptor_set(), {16, 16, 32, 32, 32, 8}},
      {loom_x86_avx512_packed_dot_core_descriptor_set(),
       {16, 16, 32, 32, 32, 8}},
      {loom_x86_packed_dot_core_descriptor_set(), {0, 0, 32, 32, 32, 0}},
      {loom_x86_avx512_vnni_core_descriptor_set(), {0, 0, 32, 32, 32, 0}},
      {loom_x86_avx512_bf16_core_descriptor_set(), {0, 0, 32, 32, 32, 0}},
      {loom_x86_avx_vnni_core_descriptor_set(), {0, 0, 16, 16, 0, 0}},
      {loom_x86_avx_vnni_int8_core_descriptor_set(), {0, 0, 16, 16, 0, 0}},
      {loom_x86_avx_vnni_int16_core_descriptor_set(), {0, 0, 16, 16, 0, 0}},
      {loom_x86_avx10_2_core_descriptor_set(), {0, 0, 32, 32, 32, 0}},
  };
  constexpr loom_x86_register_class_t register_classes[] = {
      LOOM_X86_REGISTER_CLASS_GPR32, LOOM_X86_REGISTER_CLASS_GPR64,
      LOOM_X86_REGISTER_CLASS_XMM,   LOOM_X86_REGISTER_CLASS_YMM,
      LOOM_X86_REGISTER_CLASS_ZMM,   LOOM_X86_REGISTER_CLASS_K,
  };
  const auto* storage = loom_x86_avx512_packed_dot_core_descriptor_set();
  for (const Case& test_case : cases) {
    const auto* descriptor_set = test_case.descriptor_set;
    SCOPED_TRACE(ToString(loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->key_string_ref)));
    for (uint16_t kind = 0; kind < IREE_ARRAYSIZE(register_classes); ++kind) {
      const auto register_class = register_classes[kind];
      const uint16_t storage_id = register_class;
      const iree_string_view_t name = loom_low_descriptor_set_string(
          storage, storage->reg_classes[storage_id].name_string_ref);
      SCOPED_TRACE(ToString(name));
      uint16_t class_id = LOOM_LOW_REG_CLASS_NONE;
      const loom_low_reg_class_t* reg_class = nullptr;
      bool found = loom_low_descriptor_set_lookup_register_class(
          descriptor_set, name, &class_id, &reg_class);
      loom_type_t type = loom_type_none();
      if (test_case.capacities[kind] == 0) {
        EXPECT_FALSE(found);
        EXPECT_EQ(class_id, LOOM_LOW_REG_CLASS_NONE);
        EXPECT_EQ(reg_class, nullptr);
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_NOT_FOUND,
            loom_low_build_register_type(descriptor_set, storage_id, 1, &type));
      } else {
        ASSERT_TRUE(found);
        EXPECT_EQ(class_id, storage_id);
        EXPECT_EQ(reg_class->allocatable_count, test_case.capacities[kind]);
        IREE_ASSERT_OK(
            loom_low_build_register_type(descriptor_set, storage_id, 1, &type));
        EXPECT_EQ(loom_x86_logical_register_class(class_id), register_class);
      }
    }
  }
  EXPECT_EQ(loom_x86_avx_vnni_core_descriptor_set()->reg_classes,
            loom_x86_avx_vnni_int8_core_descriptor_set()->reg_classes);
  EXPECT_EQ(loom_x86_avx512_core_descriptor_set()->operands,
            loom_x86_avx2_core_descriptor_set()->operands);
}

TEST(X86RegisterClassesTest, CountClassesAliasTheSamePhysicalRegister) {
  for (const auto* descriptor_set : {
           loom_x86_scalar_core_descriptor_set(),
           loom_x86_simd128_core_descriptor_set(),
           loom_x86_avx2_core_descriptor_set(),
           loom_x86_avx512_core_descriptor_set(),
           loom_x86_avx512_packed_dot_core_descriptor_set(),
       }) {
    SCOPED_TRACE(ToString(loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->key_string_ref)));
    uint16_t general_class_id = LOOM_LOW_REG_CLASS_NONE;
    const loom_low_reg_class_t* general_class = nullptr;
    ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
        descriptor_set, IREE_SV("x86.gpr64"), &general_class_id,
        &general_class));
    for (const auto count_name : {IREE_SV("x86.ecx"), IREE_SV("x86.rcx")}) {
      SCOPED_TRACE(ToString(count_name));
      uint16_t count_class_id = LOOM_LOW_REG_CLASS_NONE;
      const loom_low_reg_class_t* count_class = nullptr;
      ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
          descriptor_set, count_name, &count_class_id, &count_class));
      EXPECT_EQ(count_class->allocatable_count, 1);
      EXPECT_EQ(count_class->alias_set_id, general_class->alias_set_id);
      EXPECT_TRUE(iree_any_bit_set(count_class->flags,
                                   LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE));
      EXPECT_EQ(loom_low_descriptor_set_physical_register_candidate(
                    descriptor_set, count_class_id, 0),
                1);
      const bool is_word =
          iree_string_view_equal(count_name, IREE_SV("x86.ecx"));
      EXPECT_EQ(count_class->alloc_unit_bits, is_word ? 32 : 64);
      EXPECT_EQ(loom_x86_logical_register_class(count_class_id),
                is_word ? LOOM_X86_REGISTER_CLASS_GPR32
                        : LOOM_X86_REGISTER_CLASS_GPR64);
    }
  }
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

  ExpectDescriptorPresent(avx512_vnni_descriptor_set,
                          IREE_SV("x86.avx512_vnni.vpdpbusd.zmm"));
  ExpectDescriptorPresent(avx512_bf16_descriptor_set,
                          IREE_SV("x86.avx512_bf16.vdpbf16ps.zmm"));

  ExpectDescriptorPresent(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx_vnni.vpdpbusd.ymm"));
  ExpectDescriptorMissing(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx512.vaddps.zmm"));
  ExpectDescriptorMissing(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx512_vnni.vpdpbusd.zmm"));
  ExpectDescriptorMissing(avx_vnni_descriptor_set,
                          IREE_SV("x86.avx10_2.vpdpbssd.zmm"));
  ExpectDescriptorPresent(avx_vnni_int8_descriptor_set,
                          IREE_SV("x86.avx_vnni_int8.vpdpbssd.ymm"));
  ExpectDescriptorMissing(avx_vnni_int8_descriptor_set,
                          IREE_SV("x86.avx10_2.vpdpbssd.zmm"));
  ExpectDescriptorPresent(avx_vnni_int16_descriptor_set,
                          IREE_SV("x86.avx_vnni_int16.vpdpwsud.ymm"));

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

}  // namespace
}  // namespace loom
