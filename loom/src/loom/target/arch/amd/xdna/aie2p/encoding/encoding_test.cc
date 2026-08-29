// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct EncodingIds {
  loom_aie2p_instruction_id_t nop;
  loom_aie2p_instruction_id_t ret;
  loom_aie2p_instruction_id_t vector_add_8;
  loom_aie2p_instruction_id_t vector_add_16;
  loom_aie2p_instruction_id_t vector_add_32;
  loom_aie2p_instruction_id_t vector_load_a;
  loom_aie2p_instruction_id_t vector_load_b;
  loom_aie2p_instruction_id_t vector_store;
  loom_aie2p_encoding_field_id_t d;
  loom_aie2p_encoding_field_id_t dst;
  loom_aie2p_encoding_field_id_t imm;
  loom_aie2p_encoding_field_id_t ptr;
  loom_aie2p_encoding_field_id_t s1;
  loom_aie2p_encoding_field_id_t s2;
  loom_aie2p_encoding_field_id_t src;
  loom_aie2p_bundle_format_id_t i16_nop;
  loom_aie2p_bundle_format_id_t i32_alu;
  loom_aie2p_bundle_format_id_t i32_mv;
  loom_aie2p_bundle_format_id_t i32_st;
  loom_aie2p_bundle_format_id_t i48_lda_ldb;
};

EncodingIds ResolveEncodingIds() {
  return {
      .nop = loom_aie2p_encoding_find_instruction(IREE_SV("NOP")),
      .ret = loom_aie2p_encoding_find_instruction(IREE_SV("RET")),
      .vector_add_8 = loom_aie2p_encoding_find_instruction(IREE_SV("VADD_8")),
      .vector_add_16 = loom_aie2p_encoding_find_instruction(IREE_SV("VADD_16")),
      .vector_add_32 = loom_aie2p_encoding_find_instruction(IREE_SV("VADD_32")),
      .vector_load_a = loom_aie2p_encoding_find_instruction(
          IREE_SV("VLDA_dmx_lda_x_idx_imm")),
      .vector_load_b = loom_aie2p_encoding_find_instruction(
          IREE_SV("VLDB_dmx_ldb_x_idx_imm")),
      .vector_store = loom_aie2p_encoding_find_instruction(
          IREE_SV("VST_dmx_sts_x_idx_imm")),
      .d = loom_aie2p_encoding_find_field(IREE_SV("d")),
      .dst = loom_aie2p_encoding_find_field(IREE_SV("dst")),
      .imm = loom_aie2p_encoding_find_field(IREE_SV("imm")),
      .ptr = loom_aie2p_encoding_find_field(IREE_SV("ptr")),
      .s1 = loom_aie2p_encoding_find_field(IREE_SV("s1")),
      .s2 = loom_aie2p_encoding_find_field(IREE_SV("s2")),
      .src = loom_aie2p_encoding_find_field(IREE_SV("src")),
      .i16_nop = loom_aie2p_encoding_find_bundle_format(IREE_SV("I16_NOP")),
      .i32_alu = loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_ALU")),
      .i32_mv = loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_MV")),
      .i32_st = loom_aie2p_encoding_find_bundle_format(IREE_SV("I32_ST")),
      .i48_lda_ldb =
          loom_aie2p_encoding_find_bundle_format(IREE_SV("I48_LDA_LDB")),
  };
}

iree_status_t AppendBundle(loom_aie2p_bundle_format_id_t format,
                           const loom_aie2p_encoded_slot_t* slots,
                           iree_host_size_t slot_count,
                           std::vector<uint8_t>* program) {
  loom_aie2p_encoding_packet_t packet;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_encoding_pack_bundle(format, slots, slot_count, &packet));
  program->insert(program->end(), packet.data,
                  packet.data + packet.data_length);
  return iree_ok_status();
}

iree_status_t BuildVectorAddLeaf(const EncodingIds& ids,
                                 loom_aie2p_instruction_id_t vector_add,
                                 std::vector<uint8_t>* out_program) {
  out_program->clear();

  const loom_aie2p_encoding_field_value_t load_a_fields[] = {
      {ids.dst, 0},
      {ids.imm, 0},
      {ids.ptr, 0},
  };
  uint64_t load_a = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      ids.vector_load_a, load_a_fields, IREE_ARRAYSIZE(load_a_fields),
      &load_a));

  const loom_aie2p_encoding_field_value_t load_b_fields[] = {
      {ids.dst, 2},
      {ids.imm, 0},
      {ids.ptr, 1},
  };
  uint64_t load_b = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      ids.vector_load_b, load_b_fields, IREE_ARRAYSIZE(load_b_fields),
      &load_b));

  uint64_t nop = 0;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_encoding_pack_instruction(ids.nop, /*field_values=*/nullptr,
                                           /*field_value_count=*/0, &nop));
  uint64_t ret = 0;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_encoding_pack_instruction(ids.ret, /*field_values=*/nullptr,
                                           /*field_value_count=*/0, &ret));

  const loom_aie2p_encoding_field_value_t vector_add_fields[] = {
      {ids.d, 0},
      {ids.s1, 2},
      {ids.s2, 0},
  };
  uint64_t vector_add_value = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      vector_add, vector_add_fields, IREE_ARRAYSIZE(vector_add_fields),
      &vector_add_value));

  const loom_aie2p_encoding_field_value_t store_fields[] = {
      {ids.imm, 0},
      {ids.ptr, 2},
      {ids.src, 0},
  };
  uint64_t store = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_instruction(
      ids.vector_store, store_fields, IREE_ARRAYSIZE(store_fields), &store));

  const loom_aie2p_encoded_slot_t dual_load_slots[] = {
      {LOOM_AIE2P_SLOT_LDA, load_a},
      {LOOM_AIE2P_SLOT_LDB, load_b},
  };
  IREE_RETURN_IF_ERROR(AppendBundle(ids.i48_lda_ldb, dual_load_slots,
                                    IREE_ARRAYSIZE(dual_load_slots),
                                    out_program));

  const loom_aie2p_encoded_slot_t nop_slot[] = {
      {LOOM_AIE2P_SLOT_NOP, nop},
  };
  for (int i = 0; i < 4; ++i) {
    IREE_RETURN_IF_ERROR(AppendBundle(ids.i16_nop, nop_slot,
                                      IREE_ARRAYSIZE(nop_slot), out_program));
  }

  const loom_aie2p_encoded_slot_t ret_slot[] = {
      {LOOM_AIE2P_SLOT_ALU, ret},
  };
  IREE_RETURN_IF_ERROR(AppendBundle(ids.i32_alu, ret_slot,
                                    IREE_ARRAYSIZE(ret_slot), out_program));
  IREE_RETURN_IF_ERROR(AppendBundle(ids.i16_nop, nop_slot,
                                    IREE_ARRAYSIZE(nop_slot), out_program));

  const loom_aie2p_encoded_slot_t vector_add_slot[] = {
      {LOOM_AIE2P_SLOT_MV, vector_add_value},
  };
  IREE_RETURN_IF_ERROR(AppendBundle(ids.i32_mv, vector_add_slot,
                                    IREE_ARRAYSIZE(vector_add_slot),
                                    out_program));
  IREE_RETURN_IF_ERROR(AppendBundle(ids.i16_nop, nop_slot,
                                    IREE_ARRAYSIZE(nop_slot), out_program));

  const loom_aie2p_encoded_slot_t store_slot[] = {
      {LOOM_AIE2P_SLOT_ST, store},
  };
  IREE_RETURN_IF_ERROR(AppendBundle(ids.i32_st, store_slot,
                                    IREE_ARRAYSIZE(store_slot), out_program));
  IREE_RETURN_IF_ERROR(AppendBundle(ids.i16_nop, nop_slot,
                                    IREE_ARRAYSIZE(nop_slot), out_program));
  return iree_ok_status();
}

TEST(EncodingTest, DenseIdsRoundTripStableNames) {
  for (iree_host_size_t i = 1; i <= loom_aie2p_encoding_instruction_count();
       ++i) {
    const loom_aie2p_instruction_id_t instruction =
        (loom_aie2p_instruction_id_t)i;
    loom_aie2p_instruction_info_t info;
    ASSERT_TRUE(loom_aie2p_encoding_query_instruction_info(instruction, &info));
    EXPECT_EQ(loom_aie2p_encoding_find_instruction(info.name), instruction);
  }
  EXPECT_EQ(loom_aie2p_encoding_find_instruction(IREE_SV("missing")),
            LOOM_AIE2P_INSTRUCTION_ID_INVALID);

  for (iree_host_size_t i = 1; i <= loom_aie2p_encoding_field_count(); ++i) {
    const loom_aie2p_encoding_field_id_t field =
        (loom_aie2p_encoding_field_id_t)i;
    const iree_string_view_t name = loom_aie2p_encoding_field_name(field);
    ASSERT_FALSE(iree_string_view_is_empty(name));
    EXPECT_EQ(loom_aie2p_encoding_find_field(name), field);
  }
  EXPECT_EQ(loom_aie2p_encoding_find_field(IREE_SV("missing")),
            LOOM_AIE2P_ENCODING_FIELD_ID_INVALID);

  for (iree_host_size_t i = 1; i <= loom_aie2p_encoding_bundle_format_count();
       ++i) {
    const loom_aie2p_bundle_format_id_t format =
        (loom_aie2p_bundle_format_id_t)i;
    loom_aie2p_bundle_format_info_t info;
    ASSERT_TRUE(loom_aie2p_encoding_query_bundle_format_info(format, &info));
    EXPECT_EQ(loom_aie2p_encoding_find_bundle_format(info.name), format);
  }
  EXPECT_EQ(loom_aie2p_encoding_find_bundle_format(IREE_SV("missing")),
            LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
}

TEST(EncodingTest, BundleFormatsAreSelectedByExactSlotSet) {
  const loom_aie2p_slot_t dual_load_slots[] = {
      LOOM_AIE2P_SLOT_LDB,
      LOOM_AIE2P_SLOT_LDA,
  };
  const loom_aie2p_bundle_format_id_t dual_load_format =
      loom_aie2p_encoding_find_bundle_format_for_slots(
          dual_load_slots, IREE_ARRAYSIZE(dual_load_slots));
  EXPECT_EQ(dual_load_format,
            loom_aie2p_encoding_find_bundle_format(IREE_SV("I48_LDA_LDB")));

  const loom_aie2p_slot_t duplicate_slots[] = {
      LOOM_AIE2P_SLOT_LDA,
      LOOM_AIE2P_SLOT_LDA,
  };
  EXPECT_EQ(loom_aie2p_encoding_find_bundle_format_for_slots(
                duplicate_slots, IREE_ARRAYSIZE(duplicate_slots)),
            LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);

  const loom_aie2p_slot_t unsupported_slots[] = {
      LOOM_AIE2P_SLOT_NOP,
      LOOM_AIE2P_SLOT_ALU,
  };
  EXPECT_EQ(loom_aie2p_encoding_find_bundle_format_for_slots(
                unsupported_slots, IREE_ARRAYSIZE(unsupported_slots)),
            LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID);
}

TEST(EncodingTest, ReproducesRetainedVectorLeaves) {
  const EncodingIds ids = ResolveEncodingIds();
  ASSERT_EQ(loom_aie2p_encoding_instruction_count(), 880);
  ASSERT_EQ(loom_aie2p_encoding_field_count(), 28);
  ASSERT_EQ(loom_aie2p_encoding_bundle_format_count(), 77);
  ASSERT_NE(ids.vector_add_8, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  ASSERT_NE(ids.vector_add_16, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  ASSERT_NE(ids.vector_add_32, LOOM_AIE2P_INSTRUCTION_ID_INVALID);

  struct TestCase {
    loom_aie2p_instruction_id_t instruction;
    std::array<uint8_t, 32> expected;
  };
  const TestCase test_cases[] = {
      {
          ids.vector_add_32,
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x2d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
      {
          ids.vector_add_16,
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x1d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
      {
          ids.vector_add_8,
          {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x18, 0x00, 0x28, 0x10, 0x00, 0x00, 0x78, 0x0d,
           0x10, 0x18, 0x00, 0x00, 0x18, 0x13, 0x04, 0x0a, 0x00, 0x00},
      },
  };

  for (const TestCase& test_case : test_cases) {
    std::vector<uint8_t> program;
    IREE_ASSERT_OK(BuildVectorAddLeaf(ids, test_case.instruction, &program));
    EXPECT_EQ(program, std::vector<uint8_t>(test_case.expected.begin(),
                                            test_case.expected.end()));
  }
}

TEST(EncodingTest, DecodesBundleSlotsAndInstructionFields) {
  const EncodingIds ids = ResolveEncodingIds();
  const uint8_t encoded_packet[] = {0x3c, 0x68, 0x09, 0x72, 0x83, 0x00};
  loom_aie2p_decoded_bundle_t bundle;
  IREE_ASSERT_OK(loom_aie2p_encoding_decode_bundle(
      iree_make_const_byte_span(encoded_packet, sizeof(encoded_packet)),
      &bundle));

  EXPECT_EQ(bundle.format, ids.i48_lda_ldb);
  ASSERT_EQ(bundle.slot_count, 2);
  EXPECT_EQ(bundle.slots[0].slot, LOOM_AIE2P_SLOT_LDA);
  EXPECT_EQ(bundle.slots[0].value, 2103);
  EXPECT_EQ(bundle.slots[1].slot, LOOM_AIE2P_SLOT_LDB);
  EXPECT_EQ(bundle.slots[1].value, 16685);

  loom_aie2p_instruction_id_t candidates[2];
  EXPECT_EQ(loom_aie2p_encoding_query_instruction_candidates(
                bundle.slots[0].slot, bundle.slots[0].value,
                IREE_ARRAYSIZE(candidates), candidates),
            1);
  EXPECT_EQ(candidates[0], ids.vector_load_a);

  loom_aie2p_encoding_field_value_t fields[3];
  iree_host_size_t field_count = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_unpack_instruction(
      candidates[0], bundle.slots[0].value, IREE_ARRAYSIZE(fields), fields,
      &field_count));
  ASSERT_EQ(field_count, 3);
  EXPECT_EQ(fields[0].field_id, ids.dst);
  EXPECT_EQ(fields[0].value, 0);
  EXPECT_EQ(fields[1].field_id, ids.imm);
  EXPECT_EQ(fields[1].value, 0);
  EXPECT_EQ(fields[2].field_id, ids.ptr);
  EXPECT_EQ(fields[2].value, 0);
}

TEST(EncodingTest, RejectsIncompleteOrOutOfRangeInputs) {
  const EncodingIds ids = ResolveEncodingIds();
  const loom_aie2p_encoding_field_value_t missing_field[] = {
      {ids.d, 0},
      {ids.s1, 2},
  };
  uint64_t encoded_value = 0;
  EXPECT_THAT(
      loom_aie2p_encoding_pack_instruction(ids.vector_add_32, missing_field,
                                           IREE_ARRAYSIZE(missing_field),
                                           &encoded_value),
      iree::testing::status::StatusIs(iree::StatusCode::kInvalidArgument));

  const loom_aie2p_encoding_field_value_t out_of_range[] = {
      {ids.d, 16},
      {ids.s1, 2},
      {ids.s2, 0},
  };
  EXPECT_THAT(loom_aie2p_encoding_pack_instruction(
                  ids.vector_add_32, out_of_range, IREE_ARRAYSIZE(out_of_range),
                  &encoded_value),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
}

TEST(EncodingTest, VerifiedPackingConsumesDeclaredLowFieldBits) {
  const loom_aie2p_instruction_id_t vector_select =
      loom_aie2p_encoding_find_instruction(IREE_SV("VSEL_16"));
  const loom_aie2p_encoding_field_id_t d =
      loom_aie2p_encoding_find_field(IREE_SV("d"));
  const loom_aie2p_encoding_field_id_t s1 =
      loom_aie2p_encoding_find_field(IREE_SV("s1"));
  const loom_aie2p_encoding_field_id_t s2 =
      loom_aie2p_encoding_find_field(IREE_SV("s2"));
  const loom_aie2p_encoding_field_id_t sel =
      loom_aie2p_encoding_find_field(IREE_SV("sel"));
  const loom_aie2p_encoding_field_value_t source_values[] = {
      {d, 0},
      {s1, 0},
      {s2, 0},
      {sel, 31},
  };
  const loom_aie2p_encoded_slot_t source_slot =
      loom_aie2p_encoding_pack_verified_instruction(
          vector_select, source_values, IREE_ARRAYSIZE(source_values));
  EXPECT_EQ(source_slot.slot, LOOM_AIE2P_SLOT_MV);

  const loom_aie2p_encoding_field_value_t architectural_values[] = {
      {d, 0},
      {s1, 0},
      {s2, 0},
      {sel, 15},
  };
  uint64_t architectural_value = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
      vector_select, architectural_values, IREE_ARRAYSIZE(architectural_values),
      &architectural_value));
  EXPECT_EQ(source_slot.value, architectural_value);

  uint64_t rejected_value = 0;
  EXPECT_THAT(loom_aie2p_encoding_pack_instruction(
                  vector_select, source_values, IREE_ARRAYSIZE(source_values),
                  &rejected_value),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
}

TEST(EncodingTest, RetainsArchitecturalDelayWindowAndProvenance) {
  const EncodingIds ids = ResolveEncodingIds();
  loom_aie2p_instruction_info_t info;
  ASSERT_TRUE(loom_aie2p_encoding_query_instruction_info(ids.ret, &info));
  EXPECT_EQ(info.slot, LOOM_AIE2P_SLOT_ALU);
  EXPECT_EQ(info.delay_slot_count, 5);
  EXPECT_TRUE(iree_string_view_equal(
      loom_aie2p_encoding_llvm_aie_source_commit(),
      IREE_SV("ce8c0f8fd66bff15b347351c67e9fb4fe0a17205")));
}

}  // namespace
