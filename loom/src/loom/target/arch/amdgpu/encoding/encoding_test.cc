// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/encoding/encoding.h"

#include <iomanip>

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

TEST(AmdgpuEncodingTest, DecodesExactlyArchitecturalDppControls) {
  bool expected[0x200] = {};
  for (uint16_t value = 0x000; value <= 0x0FF; ++value) expected[value] = true;
  for (uint16_t value = 0x101; value <= 0x10F; ++value) expected[value] = true;
  for (uint16_t value = 0x111; value <= 0x11F; ++value) expected[value] = true;
  for (uint16_t value = 0x121; value <= 0x12F; ++value) expected[value] = true;
  for (uint16_t value = 0x130; value <= 0x13C; value += 4) {
    expected[value] = true;
  }
  for (uint16_t value = 0x140; value <= 0x143; ++value) expected[value] = true;
  for (uint16_t value = 0x150; value <= 0x16F; ++value) expected[value] = true;

  for (uint16_t value = 0; value < IREE_ARRAYSIZE(expected); ++value) {
    loom_amdgpu_dpp_control_decoding_t decoding = {};
    EXPECT_EQ(loom_amdgpu_dpp_control_decode(value, &decoding), expected[value])
        << "DPP control 0x" << std::hex << value;
  }
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

TEST(AmdgpuEncodingTest, PacksSDelayAluSoppImmediate) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_sopp_simm16(
      table, /*opcode=*/0x007, /*immediate=*/0x0214, &packet));
  EXPECT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.bit_count, 32u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xbf870214));
}

TEST(AmdgpuEncodingTest, PacksRdnaVop3UnusedSourcesAsInlineZero) {
  const uint16_t descriptor_set_ordinals[] = {
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_GFX11_GENERIC,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_GFX12_GENERIC,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_GFX12_5_GENERIC,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3_5,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX1250_A0,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX1251,
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
  };
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x101},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x102},
  };
  bool tested_table = false;
  for (uint16_t descriptor_set_ordinal : descriptor_set_ordinals) {
    const loom_amdgpu_encoding_table_t* table =
        loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
            descriptor_set_ordinal);
    if (table == nullptr) continue;
    tested_table = true;
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3, /*opcode=*/0x32C, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    EXPECT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.bit_count, 64u);
    EXPECT_EQ(packet.words[0], UINT32_C(0xd72c0000));
    EXPECT_EQ(packet.words[1], UINT32_C(0x02020501));
  }
  if (!tested_table) GTEST_SKIP() << "No RDNA encoding table selected";
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

TEST(AmdgpuEncodingTest, SelectedTablesMatchDescriptorSetFacts) {
  for (uint16_t ordinal = 0;
       ordinal < loom_amdgpu_target_info_descriptor_set_count(); ++ordinal) {
    const loom_amdgpu_encoding_table_t* table =
        loom_amdgpu_encoding_table_for_descriptor_set_ordinal(ordinal);
    if (table == nullptr) continue;
    const loom_amdgpu_descriptor_set_info_t* descriptor_set =
        loom_amdgpu_target_info_descriptor_set_at(ordinal);
    ASSERT_NE(descriptor_set, nullptr);
    EXPECT_EQ(table->descriptor_set_ordinal, descriptor_set->ordinal);
    EXPECT_TRUE(
        iree_string_view_equal(table->descriptor_set_key, descriptor_set->key));
  }
}

TEST(AmdgpuEncodingTest, PacksGfx125XPackedFp8Vop1Words) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  struct {
    uint16_t opcode;
    uint16_t destination;
    uint16_t source;
    uint32_t expected_word;
  } cases[] = {
      {0xEB, 0, 1, UINT32_C(0x7e00eb01)},
      {0xEB, 126, 127, UINT32_C(0x7efceb7f)},
      {0xED, 4, 5, UINT32_C(0x7e08ed05)},
      {0xED, 127, 126, UINT32_C(0x7efeed7e)},
  };
  for (const auto& test_case : cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {
            /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
            /*.reserved=*/{},
            /*.value=*/test_case.destination,
        },
        {
            /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VSRC0,
            /*.reserved=*/{},
            /*.value=*/test_case.source,
        },
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1_VGPR, test_case.opcode,
        field_values, IREE_ARRAYSIZE(field_values), &packet));
    EXPECT_EQ(packet.word_count, 1u);
    EXPECT_EQ(packet.bit_count, 32u);
    EXPECT_EQ(packet.words[0], test_case.expected_word);
  }
}

TEST(AmdgpuEncodingTest, PacksGfx125XScalarFp8Vop1Words) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  struct {
    uint16_t opcode;
    uint32_t expected_word;
  } cases[] = {
      {0x77, UINT32_C(0x7e02ef02)},
      {0x78, UINT32_C(0x7e02f102)},
  };
  for (const auto& test_case : cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {
            /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
            /*.reserved=*/{},
            /*.value=*/1,
        },
        {
            /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0,
            /*.reserved=*/{},
            /*.value=*/258,
        },
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    EXPECT_EQ(packet.word_count, 1u);
    EXPECT_EQ(packet.bit_count, 32u);
    EXPECT_EQ(packet.words[0], test_case.expected_word);
  }
}

TEST(AmdgpuEncodingTest, PacksGfx125XScalarFp8Vop3Words) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  struct {
    uint16_t opcode;
    uint16_t op_sel;
    uint32_t expected_word_0;
  } cases[] = {
      {0x1F7, 2, UINT32_C(0xd5f71001)}, {0x1F7, 1, UINT32_C(0xd5f70801)},
      {0x1F7, 3, UINT32_C(0xd5f71801)}, {0x1F8, 2, UINT32_C(0xd5f81001)},
      {0x1F8, 1, UINT32_C(0xd5f80801)}, {0x1F8, 3, UINT32_C(0xd5f81801)},
  };
  for (const auto& test_case : cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {
            /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
            /*.reserved=*/{},
            /*.value=*/1,
        },
        {
            /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_SRC0,
            /*.reserved=*/{},
            /*.value=*/258,
        },
        {
            /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_OPSEL,
            /*.reserved=*/{},
            /*.value=*/test_case.op_sel,
        },
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    EXPECT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.bit_count, 64u);
    EXPECT_EQ(packet.words[0], test_case.expected_word_0);
    EXPECT_EQ(packet.words[1], UINT32_C(0x02010102));
  }
}

TEST(AmdgpuEncodingTest, PacksRdna4mFp8DecodeWords) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4M, "amdgpu.rdna4m.core");

  loom_amdgpu_encoding_packet_t packet = {};
  struct {
    uint16_t opcode;
    uint32_t expected_word;
  } vop1_cases[] = {
      {0x6C, UINT32_C(0x7e02d903)},
      {0x6D, UINT32_C(0x7e02db03)},
      {0x6E, UINT32_C(0x7e02dd03)},
      {0x6F, UINT32_C(0x7e02df03)},
  };
  for (const auto& test_case : vop1_cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 1},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x103},
    };
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 1u);
    EXPECT_EQ(packet.words[0], test_case.expected_word);
  }

  struct {
    uint16_t opcode;
    uint16_t op_sel;
    uint32_t expected_word_0;
  } vop3_cases[] = {
      {0x1EC, 2, UINT32_C(0xd5ec1001)}, {0x1EC, 1, UINT32_C(0xd5ec0801)},
      {0x1EC, 3, UINT32_C(0xd5ec1801)}, {0x1EE, 1, UINT32_C(0xd5ee0801)},
      {0x1ED, 2, UINT32_C(0xd5ed1001)}, {0x1ED, 1, UINT32_C(0xd5ed0801)},
      {0x1ED, 3, UINT32_C(0xd5ed1801)}, {0x1EF, 1, UINT32_C(0xd5ef0801)},
  };
  for (const auto& test_case : vop3_cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 1},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x103},
        {LOOM_AMDGPU_ENCODING_FIELD_OP_SEL, {}, test_case.op_sel},
    };
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0], test_case.expected_word_0);
    EXPECT_EQ(packet.words[1], UINT32_C(0x02010103));
  }
}

TEST(AmdgpuEncodingTest, PacksRdna4mMinmaxWords) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4M, "amdgpu.rdna4m.core");

  const struct {
    uint16_t opcode;
    uint32_t expected_word;
  } vop2_cases[] = {
      {0x00F, UINT32_C(0x1e020702)},
      {0x010, UINT32_C(0x20020702)},
      {0x03A, UINT32_C(0x74020702)},
      {0x039, UINT32_C(0x72020702)},
  };
  for (const auto& test_case : vop2_cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 1},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x102},
        {LOOM_AMDGPU_ENCODING_FIELD_VSRC1, {}, 3},
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP2, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 1u);
    EXPECT_EQ(packet.words[0], test_case.expected_word);
  }

  const struct {
    uint16_t opcode;
    uint16_t source0;
    uint16_t source1;
    uint32_t expected_word_0;
    uint32_t expected_word_1;
  } vop3_binary_cases[] = {
      {0x329, 0x103, 0x105, UINT32_C(0xd7290001), UINT32_C(0x02020b03)},
      {0x32A, 0x103, 0x105, UINT32_C(0xd72a0001), UINT32_C(0x02020b03)},
      {0x365, 0x102, 0x103, UINT32_C(0xd7650001), UINT32_C(0x02020702)},
      {0x366, 0x102, 0x103, UINT32_C(0xd7660001), UINT32_C(0x02020702)},
      {0x367, 0x102, 0x103, UINT32_C(0xd7670001), UINT32_C(0x02020702)},
      {0x368, 0x102, 0x103, UINT32_C(0xd7680001), UINT32_C(0x02020702)},
      {0x341, 0x103, 0x105, UINT32_C(0xd7410001), UINT32_C(0x02020b03)},
      {0x342, 0x103, 0x105, UINT32_C(0xd7420001), UINT32_C(0x02020b03)},
  };
  for (const auto& test_case : vop3_binary_cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 1},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, test_case.source0},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, test_case.source1},
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0], test_case.expected_word_0);
    EXPECT_EQ(packet.words[1], test_case.expected_word_1);
  }

  const struct {
    uint16_t opcode;
    uint32_t expected_word_0;
  } vop3p_cases[] = {
      {0x011, UINT32_C(0xcc114001)},
      {0x012, UINT32_C(0xcc124001)},
      {0x01D, UINT32_C(0xcc1d4001)},
      {0x01E, UINT32_C(0xcc1e4001)},
  };
  for (const auto& test_case : vop3p_cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 1},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x102},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x103},
        {LOOM_AMDGPU_ENCODING_FIELD_OP_SEL_HI, {}, 0x7},
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, test_case.opcode,
        field_values, IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0], test_case.expected_word_0);
    EXPECT_EQ(packet.words[1], UINT32_C(0x1a020702));
  }

  const struct {
    uint16_t opcode;
    uint32_t expected_word_0;
  } vop3_ternary_cases[] = {
      {0x219, UINT32_C(0xd6190001)}, {0x21C, UINT32_C(0xd61c0001)},
      {0x249, UINT32_C(0xd6490001)}, {0x24C, UINT32_C(0xd64c0001)},
      {0x231, UINT32_C(0xd6310001)}, {0x232, UINT32_C(0xd6320001)},
      {0x22D, UINT32_C(0xd62d0001)}, {0x22E, UINT32_C(0xd62e0001)},
      {0x22F, UINT32_C(0xd62f0001)}, {0x230, UINT32_C(0xd6300001)},
      {0x25F, UINT32_C(0xd65f0001)}, {0x25E, UINT32_C(0xd65e0001)},
      {0x261, UINT32_C(0xd6610001)}, {0x260, UINT32_C(0xd6600001)},
      {0x26C, UINT32_C(0xd66c0001)}, {0x26D, UINT32_C(0xd66d0001)},
      {0x26E, UINT32_C(0xd66e0001)}, {0x26F, UINT32_C(0xd66f0001)},
  };
  for (const auto& test_case : vop3_ternary_cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 1},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x102},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x103},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x104},
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0], test_case.expected_word_0);
    EXPECT_EQ(packet.words[1], UINT32_C(0x04120702));
  }
}

TEST(AmdgpuEncodingTest, PacksRdna4mFp8EncodeWords) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4M, "amdgpu.rdna4m.core");
  struct {
    uint16_t opcode;
    uint16_t destination_op_sel;
    uint32_t expected_word_0;
  } cases[] = {
      {0x369, 0, UINT32_C(0xd7690001)},
      {0x369, 1 << 3, UINT32_C(0xd7694001)},
      {0x36A, 0, UINT32_C(0xd76a0001)},
      {0x36A, 1 << 3, UINT32_C(0xd76a4001)},
  };
  for (const auto& test_case : cases) {
    const loom_amdgpu_encoding_field_value_t field_values[] = {
        {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 1},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x102},
        {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x103},
        {LOOM_AMDGPU_ENCODING_FIELD_OP_SEL, {}, test_case.destination_op_sel},
    };
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3, test_case.opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0], test_case.expected_word_0);
    EXPECT_EQ(packet.words[1], UINT32_C(0x02020702));
  }
}

TEST(AmdgpuEncodingTest, PacksRdna4mFp8DotWords) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4M, "amdgpu.rdna4m.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x101},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x102},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x103},
      {LOOM_AMDGPU_ENCODING_FIELD_OP_SEL_HI, {}, 0x7},
  };
  for (uint16_t opcode = 0x24; opcode <= 0x27; ++opcode) {
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, opcode, field_values,
        IREE_ARRAYSIZE(field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0], UINT32_C(0xcc000000) |
                                   (static_cast<uint32_t>(opcode) << 16) |
                                   UINT32_C(0x00004000));
    EXPECT_EQ(packet.words[1], UINT32_C(0x1c0e0501));
  }
}

TEST(AmdgpuEncodingTest, PacksRdna4mMatrixWords) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4M, "amdgpu.rdna4m.core");
  const loom_amdgpu_encoding_field_value_t wmma_field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 8},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x100},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x104},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x108},
      {LOOM_AMDGPU_ENCODING_FIELD_OP_SEL_HI, {}, 0x7},
  };
  for (uint16_t opcode = 0x40; opcode <= 0x4A; ++opcode) {
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, opcode, wmma_field_values,
        IREE_ARRAYSIZE(wmma_field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0],
              UINT32_C(0xcc004008) | (static_cast<uint32_t>(opcode) << 16));
    EXPECT_EQ(packet.words[1], UINT32_C(0x1c220900));
  }

  const loom_amdgpu_encoding_field_value_t swmmac_field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 12},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x100},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x104},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x114},
      {LOOM_AMDGPU_ENCODING_FIELD_OP_SEL_HI, {}, 0x7},
  };
  for (uint16_t opcode = 0x50; opcode <= 0x5A; ++opcode) {
    loom_amdgpu_encoding_packet_t packet = {};
    IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
        table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, opcode, swmmac_field_values,
        IREE_ARRAYSIZE(swmmac_field_values), &packet));
    ASSERT_EQ(packet.word_count, 2u);
    EXPECT_EQ(packet.words[0],
              UINT32_C(0xcc00400c) | (static_cast<uint32_t>(opcode) << 16));
    EXPECT_EQ(packet.words[1], UINT32_C(0x1c520900));
  }
}

TEST(AmdgpuEncodingTest, PacksGfx125XVNop) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_v_nop(table, &packet));
  EXPECT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.bit_count, 32u);
  EXPECT_EQ(packet.words[0], UINT32_C(0x7e000000));
}

TEST(AmdgpuEncodingTest, RejectsGfx125XPackedFp8HighSource) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VDST,
          /*.reserved=*/{},
          /*.value=*/0,
      },
      {
          /*.field_id=*/LOOM_AMDGPU_ENCODING_FIELD_VSRC0,
          /*.reserved=*/{},
          /*.value=*/128,
      },
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1_VGPR,
                                /*opcode=*/0xEB, field_values,
                                IREE_ARRAYSIZE(field_values), &packet));
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
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP8),
      IREE_SV("vop1_dpp8")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP2_DPP8),
      IREE_SV("vop2_dpp8")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_DPP8),
      IREE_SV("vop3p_dpp8")));
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_VOP3_SDST_DPP8),
                             IREE_SV("vop3_sdst_dpp8")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP3_DPP8),
      IREE_SV("vop3_dpp8")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOPC_DPP8),
      IREE_SV("vopc_dpp8")));
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
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOP1_VGPR),
      IREE_SV("vop1_vgpr")));
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

TEST(AmdgpuEncodingTest, PacksRdna35VAddF32Dpp8LaneSelectors) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3_5, "amdgpu.rdna3_5.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 5},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 233},
      {LOOM_AMDGPU_ENCODING_FIELD_VSRC0, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 2},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_0, {}, 7},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_1, {}, 6},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_2, {}, 5},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_3, {}, 4},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_4, {}, 3},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_5, {}, 2},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_6, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_LANE_SEL_7, {}, 0},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3_DPP8, /*opcode=*/0x103,
      field_values, IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 3u);
  EXPECT_EQ(packet.bit_count, 96u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xd5030005));
  // The unused SRC2 field remains inline zero instead of naming a VGPR.
  EXPECT_EQ(packet.words[1], UINT32_C(0x020004e9));
  EXPECT_EQ(packet.words[2], UINT32_C(0x05397701));
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

TEST(AmdgpuEncodingTest, PacksCdna3SparseMatrixFma) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3, "amdgpu.cdna3.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 8},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x10C},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x120},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x10E},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA, /*opcode=*/0x78,
      field_values, IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xd3f80008));
  EXPECT_EQ(packet.words[1], UINT32_C(0x043a410c));
}

TEST(AmdgpuEncodingTest, PacksCdna4SparseMatrixFma) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4, "amdgpu.cdna4.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 32},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x138},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x140},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x126},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA, /*opcode=*/0x43,
      field_values, IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xd3c30020));
  EXPECT_EQ(packet.words[1], UINT32_C(0x049a8138));
}

TEST(AmdgpuEncodingTest, PacksRdna3SignedIntegerWmma) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3, "amdgpu.rdna3.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 8},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x100},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x104},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x108},
      {LOOM_AMDGPU_ENCODING_FIELD_OP_SEL_HI, {}, 7},
      {LOOM_AMDGPU_ENCODING_FIELD_NEG, {}, 3},
      {LOOM_AMDGPU_ENCODING_FIELD_NEG_HI, {}, 3},
      {LOOM_AMDGPU_ENCODING_FIELD_CLAMP, {}, 0},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, /*opcode=*/0x44, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xcc444308));
  EXPECT_EQ(packet.words[1], UINT32_C(0x7c220900));
}

TEST(AmdgpuEncodingTest, PacksRdna4SignedClampedIntegerSwmmac) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4, "amdgpu.rdna4.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 6},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x100},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x102},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x10E},
      {LOOM_AMDGPU_ENCODING_FIELD_OPSEL_HI, {}, 7},
      {LOOM_AMDGPU_ENCODING_FIELD_NEG, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_CLAMP, {}, 1},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, /*opcode=*/0x54, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xcc54c006));
  EXPECT_EQ(packet.words[1], UINT32_C(0x3c3a0500));
}

TEST(AmdgpuEncodingTest, PacksGfx1250SignedClampedIntegerSwmmac) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 24},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC0, {}, 0x100},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC1, {}, 0x108},
      {LOOM_AMDGPU_ENCODING_FIELD_SRC2, {}, 0x120},
      {LOOM_AMDGPU_ENCODING_FIELD_OPSEL_HI, {}, 3},
      {LOOM_AMDGPU_ENCODING_FIELD_INDEX_KEY_16BIT, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_NEG, {}, 2},
      {LOOM_AMDGPU_ENCODING_FIELD_CLAMP, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_MATRIX_A_REUSE, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_MATRIX_B_REUSE, {}, 0},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P, /*opcode=*/0x7B, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xcc7b8018));
  EXPECT_EQ(packet.words[1], UINT32_C(0x5c821100));
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

TEST(AmdgpuEncodingTest, PacksGfx1250TensorLoadToLdsD2) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_DIM, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_DMASK, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR4, {}, 0x7C},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR0, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR1, {}, 4},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR2, {}, 0x7C},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR3, {}, 0x7C},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VIMAGE, /*opcode=*/0xC4, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 3u);
  EXPECT_EQ(packet.bit_count, 96u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xd0710001));
  EXPECT_EQ(packet.words[1], UINT32_C(0x7c000000));
  EXPECT_EQ(packet.words[2], UINT32_C(0x7c7c0400));
}

TEST(AmdgpuEncodingTest, PacksGfx1250TensorLoadToLdsD4) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_DIM, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_DMASK, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR4, {}, 0x7C},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR0, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR1, {}, 8},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR2, {}, 16},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR3, {}, 20},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VIMAGE, /*opcode=*/0xC4, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 3u);
  EXPECT_EQ(packet.bit_count, 96u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xd0710001));
  EXPECT_EQ(packet.words[1], UINT32_C(0x7c000000));
  EXPECT_EQ(packet.words[2], UINT32_C(0x14100800));
}

TEST(AmdgpuEncodingTest, PacksGfx1250TensorWaitCounts) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_sopp_simm16(
      table, /*opcode=*/0x4B, /*immediate=*/0, &packet));
  ASSERT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xbfcb0000));

  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_sopp_simm16(
      table, /*opcode=*/0x4B, /*immediate=*/UINT16_MAX, &packet));
  ASSERT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xbfcbffff));
}

TEST(AmdgpuEncodingTest, PacksGfx1250ClusterAsyncLoadToLds) {
  LOOM_AMDGPU_REQUIRE_ENCODING_TABLE(
      table, LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X,
      "amdgpu.rdna4.gfx125x.core");
  const loom_amdgpu_encoding_field_value_t field_values[] = {
      {LOOM_AMDGPU_ENCODING_FIELD_VDST, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_VADDR, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_SADDR, {}, 4},
      {LOOM_AMDGPU_ENCODING_FIELD_IOFFSET, {}, 0},
      {LOOM_AMDGPU_ENCODING_FIELD_NV, {}, 1},
      {LOOM_AMDGPU_ENCODING_FIELD_SCOPE, {}, 2},
      {LOOM_AMDGPU_ENCODING_FIELD_TH, {}, 2},
  };
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL, /*opcode=*/0x6C, field_values,
      IREE_ARRAYSIZE(field_values), &packet));
  EXPECT_EQ(packet.word_count, 3u);
  EXPECT_EQ(packet.bit_count, 96u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xee1b0084));
  EXPECT_EQ(packet.words[1], UINT32_C(0x00280000));
  EXPECT_EQ(packet.words[2], UINT32_C(0x00000001));
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
  EXPECT_EQ(packet.words[1], UINT32_C(0x0a020501));
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
  uint16_t zero_source = 0;
  EXPECT_TRUE(loom_amdgpu_encoding_inline_f32_source(
      table, UINT32_C(0x00000000), &zero_source));
  EXPECT_EQ(zero_source, 128u);

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
