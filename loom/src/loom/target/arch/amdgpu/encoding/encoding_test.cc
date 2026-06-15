// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/encoding/encoding.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace {

#define LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(variable, ordinal, key)    \
  const loom_amdgpu_encoding_table_t* variable =                      \
      loom_amdgpu_encoding_table_for_descriptor_set_ordinal(ordinal); \
  if (variable == nullptr) {                                          \
    GTEST_SKIP() << "AMDGPU encoding table not selected: " << key;    \
  }

iree_status_t PackVMovB32Dpp(const loom_amdgpu_encoding_table_t* table,
                             uint16_t format,
                             loom_amdgpu_encoding_packet_t* out_packet) {
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
          /*.reserved=*/{},
          /*.value=*/1,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0,
          /*.reserved=*/{},
          /*.value=*/250,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VSRC0,
          /*.reserved=*/{},
          /*.value=*/2,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_DPP_CTRL,
          /*.reserved=*/{},
          /*.value=*/0x140,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_ROW_MASK,
          /*.reserved=*/{},
          /*.value=*/0xF,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_BANK_MASK,
          /*.reserved=*/{},
          /*.value=*/0xF,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_BOUND_CTRL,
          /*.reserved=*/{},
          /*.value=*/1,
      },
  };
  return loom_amdgpu_encoding_pack(table, format, /*opcode=*/1, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t PackVMovB32Sdwa(const loom_amdgpu_encoding_table_t* table,
                              bool sign_extend,
                              loom_amdgpu_encoding_packet_t* out_packet) {
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
          /*.reserved=*/{},
          /*.value=*/1,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0,
          /*.reserved=*/{},
          /*.value=*/249,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VSRC0,
          /*.reserved=*/{},
          /*.value=*/2,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_DST_SEL,
          /*.reserved=*/{},
          /*.value=*/6,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_DST_UNUSED,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0_SEL,
          /*.reserved=*/{},
          /*.value=*/1,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0_SEXT,
          /*.reserved=*/{},
          /*.value=*/sign_extend ? 1u : 0u,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_CLAMP,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_OMOD,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_S0,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_S1,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0_ABS,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0_NEG,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC1_ABS,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC1_NEG,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC1_SEL,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC1_SEXT,
          /*.reserved=*/{},
          /*.value=*/0,
      },
  };
  return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
                                   /*opcode=*/1, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

TEST(AmdgpuEncodingTest, VMovB32UsesInlineSourceForSmallU32) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_v_mov_b32_u32(table, /*vdst=*/1,
                                                         /*imm32=*/2, &packet));
  EXPECT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.bit_count, 32u);
}

TEST(AmdgpuEncodingTest, VMovB32UsesLiteralForLargeU32) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_v_mov_b32_u32(
      table, /*vdst=*/1, /*imm32=*/65536, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
}

TEST(AmdgpuEncodingTest, SMovB32UsesInlineSourceForSmallU32) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_s_mov_b32_u32(table, /*sdst=*/1,
                                                         /*imm32=*/2, &packet));
  EXPECT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.bit_count, 32u);
}

TEST(AmdgpuEncodingTest, SMovB32UsesLiteralForLargeU32) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_s_mov_b32_u32(
      table, /*sdst=*/1, /*imm32=*/65536, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
}

TEST(AmdgpuEncodingTest, Vop2U32VgprUsesInlineSourceForSmallU32) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vop2_u32_vgpr(
      table, /*opcode=*/0, /*vdst=*/1, /*imm32=*/8, /*vsrc1=*/2, &packet));
  EXPECT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.bit_count, 32u);
}

TEST(AmdgpuEncodingTest, Vop2U32VgprUsesLiteralForLargeU32) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vop2_u32_vgpr(
      table, /*opcode=*/0, /*vdst=*/1, /*imm32=*/65536, /*vsrc1=*/2, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
}

TEST(AmdgpuEncodingTest, ReportsGeneratedTableFormatSupport) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  EXPECT_TRUE(loom_amdgpu_encoding_table_has_format(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1));
  EXPECT_TRUE(loom_amdgpu_encoding_table_has_format(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL));
  EXPECT_TRUE(loom_amdgpu_encoding_table_has_format(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY));
  EXPECT_TRUE(loom_amdgpu_encoding_table_has_format(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY_LITERAL));
  EXPECT_FALSE(loom_amdgpu_encoding_table_has_format(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3PX2));
  EXPECT_FALSE(loom_amdgpu_encoding_table_has_format(
      nullptr, LOOM_AMDGPU_ENCODING_FORMAT_VOP1));
}

TEST(AmdgpuEncodingTest, NamesVopdFormats) {
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY),
      IREE_SV("vopdxy")));
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY_LITERAL),
                             IREE_SV("vopdxy_literal")));
}

TEST(AmdgpuEncodingTest, NamesDppFormats) {
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP),
      IREE_SV("vop1_dpp")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP16),
      IREE_SV("vop1_dpp16")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP1_SDWA),
      IREE_SV("vop1_sdwa")));
}

TEST(AmdgpuEncodingTest, NamesScalarLiteralFormats) {
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_SOP1_LITERAL),
                             IREE_SV("sop1_literal")));
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_SOP2_LITERAL),
                             IREE_SV("sop2_literal")));
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_SOPC_LITERAL),
                             IREE_SV("sopc_literal")));
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_SOPK_LITERAL),
                             IREE_SV("sopk_literal")));
}

TEST(AmdgpuEncodingTest, NamesGfx1250SupplementalFormats) {
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL),
                             IREE_SV("vop3p_literal")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP3PX2),
      IREE_SV("vop3px2")));
}

TEST(AmdgpuEncodingTest, ScaleSourcesUseUnifiedSourceSelectors) {
  EXPECT_TRUE(loom_amdgpu_encoding_field_uses_unified_source(
      LOOM_AMDGPU_ENCODING_FIELD_SCALE_SRC0));
  EXPECT_TRUE(loom_amdgpu_encoding_field_uses_unified_source(
      LOOM_AMDGPU_ENCODING_FIELD_SCALE_SRC1));
}

TEST(AmdgpuEncodingTest, MapsVopFieldsToVgprMsbSlots) {
  EXPECT_EQ(loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VOP3,
                                               LOOM_AMDGPU_ENCODING_FIELD_SRC0),
            LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0);
  EXPECT_EQ(
      loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VOP3,
                                         LOOM_AMDGPU_ENCODING_FIELD_VSRC1),
      LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1);
  EXPECT_EQ(
      loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VOP3,
                                         LOOM_AMDGPU_ENCODING_FIELD_VSRC2),
      LOOM_AMDGPU_VGPR_MSB_SLOT_SRC2);
  EXPECT_EQ(loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VOP1,
                                               LOOM_AMDGPU_ENCODING_FIELD_VDST),
            LOOM_AMDGPU_VGPR_MSB_SLOT_DST);
}

TEST(AmdgpuEncodingTest, MapsMemoryFieldsToVgprMsbSlots) {
  EXPECT_EQ(loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VDS,
                                               LOOM_AMDGPU_ENCODING_FIELD_ADDR),
            LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0);
  EXPECT_EQ(
      loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VDS,
                                         LOOM_AMDGPU_ENCODING_FIELD_DATA0),
      LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1);
  EXPECT_EQ(
      loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VDS,
                                         LOOM_AMDGPU_ENCODING_FIELD_DATA1),
      LOOM_AMDGPU_VGPR_MSB_SLOT_SRC2);
  EXPECT_EQ(
      loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL,
                                         LOOM_AMDGPU_ENCODING_FIELD_VDST),
      LOOM_AMDGPU_VGPR_MSB_SLOT_DST);
  EXPECT_EQ(
      loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER,
                                         LOOM_AMDGPU_ENCODING_FIELD_VDATA),
      LOOM_AMDGPU_VGPR_MSB_SLOT_DST);
}

TEST(AmdgpuEncodingTest, LeavesUncontrolledFieldsWithoutVgprMsbSlot) {
  EXPECT_EQ(
      loom_amdgpu_encoding_vgpr_msb_slot(LOOM_AMDGPU_ENCODING_FORMAT_VOP3PX2,
                                         LOOM_AMDGPU_ENCODING_FIELD_SCALE_SRC0),
      LOOM_AMDGPU_VGPR_MSB_SLOT_NONE);
}

TEST(AmdgpuEncodingTest, MapsVgprMsbSlotsToModeShifts) {
  EXPECT_EQ(loom_amdgpu_vgpr_msb_slot_shift(LOOM_AMDGPU_VGPR_MSB_SLOT_NONE),
            0u);
  EXPECT_EQ(loom_amdgpu_vgpr_msb_slot_shift(LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0),
            0u);
  EXPECT_EQ(loom_amdgpu_vgpr_msb_slot_shift(LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1),
            2u);
  EXPECT_EQ(loom_amdgpu_vgpr_msb_slot_shift(LOOM_AMDGPU_VGPR_MSB_SLOT_SRC2),
            4u);
  EXPECT_EQ(loom_amdgpu_vgpr_msb_slot_shift(LOOM_AMDGPU_VGPR_MSB_SLOT_DST), 6u);
}

TEST(AmdgpuEncodingTest, PacksRdna3VMovB32Dpp16LaneControl) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(
      PackVMovB32Dpp(table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP16, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0x7e0202fa));
  EXPECT_EQ(packet.words[1], UINT32_C(0xff094002));
}

TEST(AmdgpuEncodingTest, PacksCdna4VMovB32DppLaneControl) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4, "amdgpu.cdna4.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(
      PackVMovB32Dpp(table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0x7e0202fa));
  EXPECT_EQ(packet.words[1], UINT32_C(0xff094002));
}

TEST(AmdgpuEncodingTest, PacksGfx1250SupplementalSwmmac) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
          /*.reserved=*/{},
          /*.value=*/24,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0,
          /*.reserved=*/{},
          /*.value=*/0x100,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC1,
          /*.reserved=*/{},
          /*.value=*/0x108,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC2,
          /*.reserved=*/{},
          /*.value=*/0x120,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_OPSEL_HI,
          /*.reserved=*/{},
          /*.value=*/3,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_INDEX_KEY_16BIT,
          /*.reserved=*/{},
          /*.value=*/1,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_A_REUSE,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_B_REUSE,
          /*.reserved=*/{},
          /*.value=*/0,
      },
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, /*opcode=*/0x65, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xcc650818));
  EXPECT_EQ(packet.words[1], UINT32_C(0x1c821100));
}

TEST(AmdgpuEncodingTest, PacksGfx1250SupplementalScaledWmma) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_X2ENCODING,
          /*.reserved=*/{},
          /*.value=*/0x35,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0,
          /*.reserved=*/{},
          /*.value=*/0x108,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC1,
          /*.reserved=*/{},
          /*.value=*/0x118,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC2,
          /*.reserved=*/{},
          /*.value=*/0x128,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SCALE_SRC0,
          /*.reserved=*/{},
          /*.value=*/0x101,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SCALE_SRC1,
          /*.reserved=*/{},
          /*.value=*/0x102,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_A_FMT,
          /*.reserved=*/{},
          /*.value=*/1,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_B_FMT,
          /*.reserved=*/{},
          /*.value=*/2,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_A_SCALE,
          /*.reserved=*/{},
          /*.value=*/1,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_B_SCALE,
          /*.reserved=*/{},
          /*.value=*/1,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_A_SCALE_FMT,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_B_SCALE_FMT,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_A_REUSE,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_MATRIX_B_REUSE,
          /*.reserved=*/{},
          /*.value=*/0,
      },
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3PX2, /*opcode=*/0x33, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 4u);
  EXPECT_EQ(packet.bit_count, 128u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xcc350800));
  EXPECT_EQ(packet.words[1], UINT32_C(0x0c020501));
  EXPECT_EQ(packet.words[2], UINT32_C(0xcc330800));
  EXPECT_EQ(packet.words[3], UINT32_C(0x14a23108));
}

TEST(AmdgpuEncodingTest, PacksCdna4VMovB32SdwaByteExtract) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4, "amdgpu.cdna4.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(PackVMovB32Sdwa(table, /*sign_extend=*/false, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0x7e0202f9));
  EXPECT_EQ(packet.words[1], UINT32_C(0x00010602));
}

TEST(AmdgpuEncodingTest, PacksCdna4VMovB32SdwaSignedByteExtract) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4, "amdgpu.cdna4.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(PackVMovB32Sdwa(table, /*sign_extend=*/true, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0x7e0202f9));
  EXPECT_EQ(packet.words[1], UINT32_C(0x00090602));
}

TEST(AmdgpuEncodingTest, PacksVopdxyDualFmacPair) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 0;
  fields.op_y = 0;
  fields.src0_x = 0x104;
  fields.vsrc1_x = 2;
  fields.vdst_x = 255;
  fields.src0_y = 0x101;
  fields.vsrc1_y = 3;
  fields.vdst_y = 6;
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vopdxy(table, &fields, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xc8000504));
  EXPECT_EQ(packet.words[1], UINT32_C(0xff060701));
}

TEST(AmdgpuEncodingTest, PacksVopdxyDualMovPair) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  uint16_t source_42 = 0;
  ASSERT_TRUE(loom_amdgpu_encoding_inline_u32_source(table, 42, &source_42));
  uint16_t source_0 = 0;
  ASSERT_TRUE(loom_amdgpu_encoding_inline_u32_source(table, 0, &source_0));

  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 8;
  fields.op_y = 8;
  fields.src0_x = source_42;
  fields.vsrc1_x = 0;
  fields.vdst_x = 0;
  fields.src0_y = source_0;
  fields.vsrc1_y = 0;
  fields.vdst_y = 1;
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vopdxy(table, &fields, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xca1000aa));
  EXPECT_EQ(packet.words[1], UINT32_C(0x00000080));
}

TEST(AmdgpuEncodingTest, PacksVopdxyLiteralDualFmaakPair) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 1;
  fields.op_y = 1;
  fields.src0_x = 0x101;
  fields.vsrc1_x = 2;
  fields.vdst_x = 0;
  fields.src0_y = 0x104;
  fields.vsrc1_y = 5;
  fields.vdst_y = 3;
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vopdxy_literal(
      table, &fields, UINT32_C(0x3f800000), &packet));
  EXPECT_EQ(packet.word_count, 3u);
  EXPECT_EQ(packet.bit_count, 96u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xc8420501));
  EXPECT_EQ(packet.words[1], UINT32_C(0x00020b04));
  EXPECT_EQ(packet.words[2], UINT32_C(0x3f800000));
}

TEST(AmdgpuEncodingTest, RejectsSameParityVopdxyDestinations) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 0;
  fields.op_y = 0;
  fields.src0_x = 0x104;
  fields.vsrc1_x = 2;
  fields.vdst_x = 255;
  fields.src0_y = 0x101;
  fields.vsrc1_y = 3;
  fields.vdst_y = 7;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_encoding_pack_vopdxy(table, &fields, &packet));
}

TEST(AmdgpuEncodingTest, RejectsOutOfRangeVopdxyOp) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 16;
  fields.op_y = 0;
  fields.src0_x = 0x104;
  fields.vsrc1_x = 2;
  fields.vdst_x = 255;
  fields.src0_y = 0x101;
  fields.vsrc1_y = 3;
  fields.vdst_y = 6;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_amdgpu_encoding_pack_vopdxy(table, &fields, &packet));
}

TEST(AmdgpuEncodingTest, InlineF32SourceMapsBitPatternToSourceSelector) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  uint16_t source = 0;
  EXPECT_TRUE(loom_amdgpu_encoding_inline_f32_source(
      table, UINT32_C(0x3f800000), &source));
  EXPECT_EQ(source, 242u);
}

TEST(AmdgpuEncodingTest, InlineF32SourceRejectsUnsupportedBitPattern) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  uint16_t source = 1;
  EXPECT_FALSE(loom_amdgpu_encoding_inline_f32_source(
      table, UINT32_C(0x40400000), &source));
  EXPECT_EQ(source, 0u);
}

}  // namespace
