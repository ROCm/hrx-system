// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

namespace {

TEST(MachineTest, CompleteTableCounts) {
  EXPECT_EQ(loom_aie2p_machine_atomic_unit_count(), 207);
  EXPECT_EQ(loom_aie2p_machine_physical_register_count(), 359);
  EXPECT_EQ(loom_aie2p_machine_register_class_count(), 369);
  EXPECT_EQ(loom_aie2p_machine_register_adapter_count(), 61);
  EXPECT_EQ(loom_aie2p_machine_immediate_count(), 21);
  EXPECT_EQ(loom_aie2p_machine_form_count(), 880);
}

TEST(MachineTest, PhysicalRegistersRetainAtomicSubregisters) {
  const loom_aie2p_physical_register_id_t x0 =
      loom_aie2p_machine_find_physical_register(IREE_SV("x0"));
  const loom_aie2p_physical_register_id_t wl0 =
      loom_aie2p_machine_find_physical_register(IREE_SV("wl0"));
  const loom_aie2p_physical_register_id_t wh0 =
      loom_aie2p_machine_find_physical_register(IREE_SV("wh0"));
  ASSERT_NE(x0, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);

  loom_aie2p_physical_register_info_t info;
  ASSERT_TRUE(loom_aie2p_machine_query_physical_register(x0, &info));
  EXPECT_TRUE(iree_string_view_equal(info.assembly_name, IREE_SV("x0")));
  EXPECT_EQ(info.hardware_encoding, 0);
  EXPECT_EQ(info.atomic_unit_count, 2);
  EXPECT_EQ(info.subregister_count, 2);

  loom_aie2p_subregister_info_t subregister;
  ASSERT_TRUE(loom_aie2p_machine_query_subregister(x0, 0, &subregister));
  EXPECT_EQ(subregister.register_id, wl0);
  EXPECT_TRUE(
      iree_string_view_equal(subregister.index_name, IREE_SV("sub_256_lo")));
  ASSERT_TRUE(loom_aie2p_machine_query_subregister(x0, 1, &subregister));
  EXPECT_EQ(subregister.register_id, wh0);
  EXPECT_TRUE(
      iree_string_view_equal(subregister.index_name, IREE_SV("sub_256_hi")));

  EXPECT_EQ(loom_aie2p_machine_physical_register_atomic_unit(x0, 0),
            loom_aie2p_machine_physical_register_atomic_unit(wh0, 0));
  EXPECT_EQ(loom_aie2p_machine_physical_register_atomic_unit(x0, 1),
            loom_aie2p_machine_physical_register_atomic_unit(wl0, 0));
}

TEST(MachineTest, RegisterClassesPreserveCandidateOrder) {
  const loom_aie2p_register_class_id_t xa =
      loom_aie2p_machine_find_register_class(IREE_SV("mXa"));
  ASSERT_NE(xa, LOOM_AIE2P_REGISTER_CLASS_ID_INVALID);
  loom_aie2p_register_class_info_t info;
  ASSERT_TRUE(loom_aie2p_machine_query_register_class(xa, &info));
  EXPECT_EQ(info.register_size_bits, 512);
  EXPECT_EQ(info.candidate_count, 12);
  EXPECT_TRUE(
      iree_any_bit_set(info.flags, LOOM_AIE2P_REGISTER_CLASS_FLAG_ALLOCATABLE));
  EXPECT_EQ(loom_aie2p_machine_register_class_candidate(xa, 0),
            loom_aie2p_machine_find_physical_register(IREE_SV("x0")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_aie2p_machine_register_class_value_type(xa, 0), IREE_SV("v64i8")));
}

TEST(MachineTest, RegisterEncodingsAreOperandLocal) {
  EXPECT_EQ(loom_aie2p_machine_find_register_adapter(
                IREE_SV("not_an_aie2p_register_adapter")),
            LOOM_AIE2P_REGISTER_ADAPTER_ID_INVALID);
  const loom_aie2p_physical_register_id_t p1 =
      loom_aie2p_machine_find_physical_register(IREE_SV("p1"));
  const loom_aie2p_register_adapter_id_t destination =
      loom_aie2p_machine_find_register_adapter(IREE_SV("OP_mAguDst"));
  const loom_aie2p_register_adapter_id_t source =
      loom_aie2p_machine_find_register_adapter(IREE_SV("OP_mAguSrc"));
  uint8_t value = 0;
  ASSERT_TRUE(loom_aie2p_machine_encode_register(destination, p1, &value));
  EXPECT_EQ(value, 12);
  ASSERT_TRUE(loom_aie2p_machine_encode_register(source, p1, &value));
  EXPECT_EQ(value, 18);
  ASSERT_TRUE(loom_aie2p_machine_encode_register(
      LOOM_AIE2P_REGISTER_ADAPTER_ID_DIRECT, p1, &value));
  EXPECT_EQ(value, 1);
}

TEST(MachineTest, QRegisterAdaptersFollowArchitecturalDecoder) {
  const iree_string_view_t adapter_names[] = {
      IREE_SV("OP_mQQsa"),
      IREE_SV("OP_mQQsm"),
      IREE_SV("OP_mQQss"),
  };
  const iree_string_view_t register_names[] = {
      IREE_SV("q0"),
      IREE_SV("q1"),
      IREE_SV("q2"),
      IREE_SV("q3"),
  };
  for (iree_string_view_t adapter_name : adapter_names) {
    const loom_aie2p_register_adapter_id_t adapter =
        loom_aie2p_machine_find_register_adapter(adapter_name);
    ASSERT_NE(adapter, LOOM_AIE2P_REGISTER_ADAPTER_ID_DIRECT);
    for (uint8_t i = 0; i < IREE_ARRAYSIZE(register_names); ++i) {
      const loom_aie2p_physical_register_id_t register_id =
          loom_aie2p_machine_find_physical_register(register_names[i]);
      uint8_t value = UINT8_MAX;
      ASSERT_TRUE(
          loom_aie2p_machine_encode_register(adapter, register_id, &value));
      EXPECT_EQ(value, i);
    }
  }
}

TEST(MachineTest, CrossWidthPredicateAdaptersProjectScalarHalves) {
  const loom_aie2p_register_class_id_t predicate_class =
      loom_aie2p_machine_find_register_class(IREE_SV("eLPredicate"));
  ASSERT_NE(predicate_class, LOOM_AIE2P_REGISTER_CLASS_ID_INVALID);
  loom_aie2p_register_class_info_t class_info;
  ASSERT_TRUE(
      loom_aie2p_machine_query_register_class(predicate_class, &class_info));
  EXPECT_EQ(class_info.candidate_count, 8);

  const loom_aie2p_register_adapter_id_t low_adapter =
      loom_aie2p_machine_find_register_adapter(IREE_SV("LOOM_eL_low32"));
  const loom_aie2p_register_adapter_id_t high_adapter =
      loom_aie2p_machine_find_register_adapter(IREE_SV("LOOM_eL_high32"));
  const loom_aie2p_register_adapter_id_t lda_high_adapter =
      loom_aie2p_machine_find_register_adapter(
          IREE_SV("LOOM_eL_high32_OP_mLdaCg"));
  ASSERT_NE(low_adapter, LOOM_AIE2P_REGISTER_ADAPTER_ID_INVALID);
  ASSERT_NE(high_adapter, LOOM_AIE2P_REGISTER_ADAPTER_ID_INVALID);
  ASSERT_NE(lda_high_adapter, LOOM_AIE2P_REGISTER_ADAPTER_ID_INVALID);

  for (uint8_t i = 0; i < class_info.candidate_count; ++i) {
    const loom_aie2p_physical_register_id_t register_id =
        loom_aie2p_machine_register_class_candidate(predicate_class, i);
    const std::string register_name = "l" + std::to_string(i + 8);
    EXPECT_EQ(register_id,
              loom_aie2p_machine_find_physical_register(iree_make_string_view(
                  register_name.data(), register_name.size())));

    uint8_t value = UINT8_MAX;
    ASSERT_TRUE(
        loom_aie2p_machine_encode_register(low_adapter, register_id, &value));
    EXPECT_EQ(value, 16 + 2 * i);
    ASSERT_TRUE(
        loom_aie2p_machine_encode_register(high_adapter, register_id, &value));
    EXPECT_EQ(value, 17 + 2 * i);
    ASSERT_TRUE(loom_aie2p_machine_encode_register(lda_high_adapter,
                                                   register_id, &value));
    EXPECT_EQ(value, 68 + 8 * i);
  }
}

TEST(MachineTest, ImmediateBoundariesRoundTrip) {
  for (iree_host_size_t i = 1; i <= loom_aie2p_machine_immediate_count(); ++i) {
    const loom_aie2p_immediate_id_t immediate_id = (loom_aie2p_immediate_id_t)i;
    loom_aie2p_immediate_info_t info;
    ASSERT_TRUE(loom_aie2p_machine_query_immediate(immediate_id, &info));
    uint8_t fixed_zero_bits = 0;
    for (uint32_t step = info.step; step > 1; step >>= 1) {
      ++fixed_zero_bits;
    }
    int64_t minimum = 0;
    int64_t maximum = 0;
    if (iree_any_bit_set(info.flags, LOOM_AIE2P_IMMEDIATE_FLAG_NEGATIVE)) {
      minimum = -(INT64_C(1) << (info.encoded_width_bits + fixed_zero_bits));
      maximum = -(int64_t)info.step;
    } else if (iree_any_bit_set(info.flags, LOOM_AIE2P_IMMEDIATE_FLAG_SIGNED)) {
      const uint8_t value_bits = info.encoded_width_bits + fixed_zero_bits;
      minimum = -(INT64_C(1) << (value_bits - 1));
      maximum = (INT64_C(1) << (value_bits - 1)) - info.step;
    } else {
      minimum = 0;
      maximum = (INT64_C(1) << (info.encoded_width_bits + fixed_zero_bits)) -
                info.step;
    }
    for (int64_t value : {minimum, maximum}) {
      uint64_t encoded_value = 0;
      IREE_ASSERT_OK(loom_aie2p_machine_encode_immediate(immediate_id, value,
                                                         &encoded_value));
      int64_t decoded_value = 0;
      IREE_ASSERT_OK(loom_aie2p_machine_decode_immediate(
          immediate_id, encoded_value, &decoded_value));
      EXPECT_EQ(decoded_value, value)
          << std::string(info.name.data, info.name.size);
    }
  }
}

TEST(MachineTest, RejectsInvalidImmediateValues) {
  const loom_aie2p_immediate_id_t negative =
      loom_aie2p_machine_find_immediate(IREE_SV("c12n_step4"));
  uint64_t encoded_value = 0;
  EXPECT_THAT(loom_aie2p_machine_encode_immediate(negative, 0, &encoded_value),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
  EXPECT_THAT(loom_aie2p_machine_encode_immediate(negative, -3, &encoded_value),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
}

TEST(MachineTest, MachineFormsAlignWithEncodingForms) {
  ASSERT_EQ(loom_aie2p_machine_form_count(),
            loom_aie2p_encoding_instruction_count());
  for (iree_host_size_t i = 1; i <= loom_aie2p_machine_form_count(); ++i) {
    loom_aie2p_machine_form_info_t machine_info;
    loom_aie2p_instruction_info_t encoding_info;
    ASSERT_TRUE(loom_aie2p_machine_query_form((loom_aie2p_machine_form_id_t)i,
                                              &machine_info));
    ASSERT_TRUE(loom_aie2p_encoding_query_instruction_info(
        (loom_aie2p_instruction_id_t)i, &encoding_info));
    EXPECT_TRUE(iree_string_view_equal(machine_info.name, encoding_info.name));
  }
  EXPECT_EQ(loom_aie2p_machine_find_form(IREE_SV("MOV_OR")),
            LOOM_AIE2P_MACHINE_FORM_ID_INVALID);
}

TEST(MachineTest, MachineFormsRetainOperandsAndTies) {
  const loom_aie2p_machine_form_id_t divs =
      loom_aie2p_machine_find_form(IREE_SV("DIVS"));
  loom_aie2p_machine_form_info_t info;
  ASSERT_TRUE(loom_aie2p_machine_query_form(divs, &info));
  EXPECT_EQ(info.output_count, 2);
  EXPECT_EQ(info.input_count, 3);
  EXPECT_EQ(info.tie_count, 1);

  loom_aie2p_machine_operand_info_t operand;
  ASSERT_TRUE(loom_aie2p_machine_query_form_operand(divs, 1, &operand));
  EXPECT_TRUE(iree_string_view_equal(operand.name, IREE_SV("sd_out")));
  loom_aie2p_machine_tie_info_t tie;
  ASSERT_TRUE(loom_aie2p_machine_query_form_tie(divs, 0, &tie));
  EXPECT_EQ(tie.definition_ordinal, 1);
  EXPECT_EQ(tie.use_ordinal, 0);
}

}  // namespace
