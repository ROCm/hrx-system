// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-owned low descriptor encoding format identifiers.
//
// These values are written into loom_low_descriptor_t::encoding_format_id by
// AMDGPU descriptor overlays. They are intentionally compact and stable within
// Loom; vendor XML encoding names stay in the generator layer instead of being
// linked into native emitters as strings.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ENCODING_ENCODING_H_
#define LOOM_TARGET_ARCH_AMDGPU_ENCODING_ENCODING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT 4u
// Sentinel passed to loom_amdgpu_encoding_pack when the descriptor supplies all
// opcode-like fields through explicit encoding field values.
#define LOOM_AMDGPU_ENCODING_OPCODE_NONE UINT16_MAX

typedef enum loom_amdgpu_encoding_format_e {
  // Descriptor does not carry an AMDGPU machine encoding format.
  LOOM_AMDGPU_ENCODING_FORMAT_NONE = 0,
  // Scalar one-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP1 = 1,
  // Scalar one-source instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP1_LITERAL = 38,
  // Scalar two-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP2 = 2,
  // Scalar two-source instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP2_LITERAL = 39,
  // Scalar program-flow or wait instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPP = 3,
  // Vector two-source 32-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP2 = 4,
  // VOP2 form with a mandatory literal payload.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL = 5,
  // Vector three-source 64-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3 = 6,
  // Packed vector three-source 64-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3P = 7,
  // Scalar memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SMEM = 8,
  // Memory buffer instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_MUBUF = 9,
  // RDNA4 VBUFFER instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER = 10,
  // Local data share instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_DS = 11,
  // Flat generic-address memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_FLAT = 13,
  // CDNA global/flat memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL = 14,
  // RDNA3 global/flat memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL = 15,
  // RDNA3 scratch/private-memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_FLAT_SCRATCH = 16,
  // Scalar compare instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPC = 20,
  // Scalar compare instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPC_LITERAL = 40,
  // Scalar one-destination immediate instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPK = 21,
  // Scalar one-destination immediate instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPK_LITERAL = 41,
  // RDNA4 local data share instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VDS = 22,
  // RDNA4 flat generic-address memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VFLAT = 25,
  // RDNA4 global memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL = 26,
  // RDNA4 scratch/private-memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VSCRATCH = 34,
  // Vector one-source 32-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP1 = 30,
  // RDNA4 scaled WMMA 128-bit paired VOP3P instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3PX2 = 31,
  // Vector one-source 32-bit instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL = 42,
  // Vector one-source 32-bit instruction format with legacy DPP lane control.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP = 43,
  // Vector one-source 32-bit instruction format with DPP16 lane control.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP1_DPP16 = 44,
  // Vector one-source 32-bit instruction format with CDNA SDWA modifiers.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP1_SDWA = 46,
  // Packed vector three-source 96-bit instruction format with mandatory
  // literal.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL = 52,
  // CDNA MFMA packed vector three-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA = 53,
  // Vector three-source 64-bit instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3_LITERAL = 56,
  // Vector three-source 64-bit instruction format with scalar carry result.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3_SDST = 57,
  // VOP3 scalar-destination form with a mandatory literal payload.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3_SDST_LITERAL = 58,
  // Wave32 dual-VALU packet format without a literal payload.
  LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY = 67,
  // Wave32 dual-VALU packet format with a shared literal payload.
  LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY_LITERAL = 68,
} loom_amdgpu_encoding_format_t;

#include "loom/target/arch/amdgpu/encoding/encoding_field_ids.h"

typedef struct loom_amdgpu_encoding_bit_range_t {
  // Target bit offset populated by this range.
  uint8_t bit_offset;
  // Number of encoded bits copied into the target packet.
  uint8_t bit_count;
  // Source value bit offset before optional padding for this range.
  uint8_t source_bit_offset;
  // Number of source padding bits that must match padding_value.
  uint8_t padding_bit_count;
  // Required value for the low padding bits in this range.
  uint64_t padding_value;
} loom_amdgpu_encoding_bit_range_t;

typedef struct loom_amdgpu_encoding_field_layout_t {
  // Stable target-owned encoding field identifier.
  uint16_t field_id;
  // First bit-range row for this field.
  uint16_t range_start;
  // Number of bit-range rows for this field.
  uint8_t range_count;
  // Number of low source bits consumed by this field including padding.
  uint8_t value_bit_count;
  // Target-owned field flags; bit 0 marks a conditional XML field.
  uint16_t flags;
} loom_amdgpu_encoding_field_layout_t;

typedef struct loom_amdgpu_encoding_format_layout_t {
  // Stable target-owned encoding format identifier.
  uint16_t format_id;
  // Total encoded packet width in bits.
  uint16_t bit_count;
  // Number of 32-bit words occupied by this packet.
  uint16_t word_count;
  // First field-layout row for this format.
  uint16_t field_start;
  // Number of field-layout rows for this format.
  uint16_t field_count;
  // Base identifier words loaded before fields are overlaid.
  uint32_t identifier_words[LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT];
  // Identifier-mask words from the vendor XML.
  uint32_t identifier_mask_words[LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT];
} loom_amdgpu_encoding_format_layout_t;

typedef struct loom_amdgpu_encoding_inline_f32_source_t {
  // IEEE f32 bit pattern accepted by the descriptor immediate.
  uint32_t bit_pattern;
  // Target source selector for the inline f32 bit pattern.
  uint16_t source;
} loom_amdgpu_encoding_inline_f32_source_t;

typedef struct loom_amdgpu_encoding_table_t {
  // Descriptor-set key this table can encode.
  iree_string_view_t descriptor_set_key;
  // Dense generated descriptor-set ordinal this table can encode.
  uint16_t descriptor_set_ordinal;
  // SOP1 opcode used for target-inserted s_mov_b32 register/literal moves.
  uint16_t s_mov_b32_opcode;
  // VOP1 opcode used for target-inserted v_mov_b32 register/immediate moves.
  uint16_t v_mov_b32_opcode;
  // Unified source field value selecting the literal word after a packet.
  uint16_t source_literal;
  // Scalar source field value selecting unsigned inline integer zero.
  uint16_t scalar_inline_u32_zero;
  // Number of contiguous unsigned inline integer scalar source values.
  uint16_t scalar_inline_u32_count;
  // Sorted inline f32 source-selector rows keyed by bit pattern.
  const loom_amdgpu_encoding_inline_f32_source_t* inline_f32_sources;
  // Number of inline f32 source-selector rows.
  uint16_t inline_f32_source_count;
  // Unified source field value selecting VGPR zero.
  uint16_t vector_source_vgpr0;
  // Number of contiguous VGPR unified-source values.
  uint16_t vector_source_vgpr_count;
  // Sorted format-layout rows.
  const loom_amdgpu_encoding_format_layout_t* formats;
  // Number of format-layout rows.
  uint32_t format_count;
  // Dense field-layout rows referenced by format rows.
  const loom_amdgpu_encoding_field_layout_t* fields;
  // Number of field-layout rows.
  uint32_t field_count;
  // Dense bit-range rows referenced by field rows.
  const loom_amdgpu_encoding_bit_range_t* bit_ranges;
  // Number of bit-range rows.
  uint32_t bit_range_count;
} loom_amdgpu_encoding_table_t;

typedef struct loom_amdgpu_encoding_field_value_t {
  // Stable target-owned encoding field identifier.
  uint16_t field_id;
  // Reserved for alignment and future flags.
  uint16_t reserved;
  // Unsigned field value before XML padding/range extraction.
  uint64_t value;
} loom_amdgpu_encoding_field_value_t;

typedef struct loom_amdgpu_encoding_packet_t {
  // Packed little-endian 32-bit instruction words.
  uint32_t words[LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT];
  // Total encoded packet width in bits.
  uint16_t bit_count;
  // Number of valid words in words.
  uint16_t word_count;
} loom_amdgpu_encoding_packet_t;

typedef enum loom_amdgpu_vgpr_msb_slot_e {
  // Encoding field is not controlled by S_SET_VGPR_MSB.
  LOOM_AMDGPU_VGPR_MSB_SLOT_NONE = 0,
  // Encoding field reads the SRC0 VGPR high-bank selector.
  LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0 = 1,
  // Encoding field reads the SRC1 VGPR high-bank selector.
  LOOM_AMDGPU_VGPR_MSB_SLOT_SRC1 = 2,
  // Encoding field reads the SRC2 VGPR high-bank selector.
  LOOM_AMDGPU_VGPR_MSB_SLOT_SRC2 = 3,
  // Encoding field reads the destination/data VGPR high-bank selector.
  LOOM_AMDGPU_VGPR_MSB_SLOT_DST = 4,
} loom_amdgpu_vgpr_msb_slot_t;

enum {
  // Number of low VGPR indices addressed by one S_SET_VGPR_MSB selector bank.
  LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE = 256,
};

// Returns the two-bit S_SET_VGPR_MSB mode shift for |slot|.
uint8_t loom_amdgpu_vgpr_msb_slot_shift(loom_amdgpu_vgpr_msb_slot_t slot);

typedef struct loom_amdgpu_encoding_vopdxy_fields_t {
  // X-slot VOPD operation id.
  uint16_t op_x;
  // Y-slot VOPD operation id.
  uint16_t op_y;
  // X-slot unified SRC0 selector.
  uint16_t src0_x;
  // X-slot VGPR source 1 register.
  uint16_t vsrc1_x;
  // X-slot VGPR destination register.
  uint16_t vdst_x;
  // Y-slot unified SRC0 selector.
  uint16_t src0_y;
  // Y-slot VGPR source 1 register.
  uint16_t vsrc1_y;
  // Y-slot VGPR destination register.
  uint16_t vdst_y;
} loom_amdgpu_encoding_vopdxy_fields_t;

// Returns a short stable diagnostic name for |encoding_format|.
iree_string_view_t loom_amdgpu_encoding_format_name(uint16_t encoding_format);

// Returns true when |table| has a generated bit-layout row for
// |encoding_format|.
bool loom_amdgpu_encoding_table_has_format(
    const loom_amdgpu_encoding_table_t* table, uint16_t encoding_format);

// Returns true when |field_id| uses AMDGPU's unified scalar/vector source
// register encoding, where VGPR operands are biased by 0x100.
bool loom_amdgpu_encoding_field_uses_unified_source(uint16_t field_id);

// Returns true when |field_id| is the first vector source selector.
bool loom_amdgpu_encoding_field_is_src0(uint16_t field_id);

// Returns true when |field_id| is the SDWA destination selector field.
bool loom_amdgpu_encoding_field_is_dst_sel(uint16_t field_id);

// Returns true when |field_id| is the literal payload field.
bool loom_amdgpu_encoding_field_is_literal(uint16_t field_id);

// Returns true when |selector| writes a sub-DWORD destination through SDWA.
bool loom_amdgpu_sdwa_dst_selector_writes_subdword(uint32_t selector);

// Returns the S_SET_VGPR_MSB selector slot controlling |encoding_field_id| in
// |encoding_format_id|, or NONE when the field does not read the MODE VGPR-MSB
// state.
loom_amdgpu_vgpr_msb_slot_t loom_amdgpu_encoding_vgpr_msb_slot(
    uint16_t encoding_format_id, uint16_t encoding_field_id);

// Maps a verified inline unsigned integer into the target's scalar/vector
// source selector domain.
bool loom_amdgpu_encoding_inline_u32_source(
    const loom_amdgpu_encoding_table_t* table, uint32_t value,
    uint16_t* out_source);

// Maps a verified inline f32 bit pattern into the target's scalar/vector source
// selector domain.
bool loom_amdgpu_encoding_inline_f32_source(
    const loom_amdgpu_encoding_table_t* table, uint32_t bit_pattern,
    uint16_t* out_source);

// Returns the generated encoding table for |descriptor_set_ordinal|, or NULL
// when this binary was not linked with a matching table.
const loom_amdgpu_encoding_table_t*
loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
    uint16_t descriptor_set_ordinal);

// Packs a native AMDGPU packet by applying |opcode| and target field values to
// the XML-derived encoding layout. When |opcode| is
// LOOM_AMDGPU_ENCODING_OPCODE_NONE, no implicit OP field is written and callers
// must supply any opcode-like fields explicitly. Field values are architectural
// numbers; XML padding bits are validated before encoded ranges are extracted.
iree_status_t loom_amdgpu_encoding_pack(
    const loom_amdgpu_encoding_table_t* table, uint16_t encoding_format,
    uint16_t opcode, const loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a SOPP packet with its 16-bit immediate field.
iree_status_t loom_amdgpu_encoding_pack_sopp_simm16(
    const loom_amdgpu_encoding_table_t* table, uint16_t opcode,
    uint16_t immediate, loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit SGPR-to-SGPR s_mov_b32 packet using the target table's SOP1
// layout.
iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_sgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint16_t ssrc0,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit immediate-to-SGPR s_mov_b32 packet using an inline scalar
// source when possible and a literal SOP1 form otherwise.
iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit VGPR-to-VGPR v_mov_b32 packet using the target table's VOP1
// layout.
iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_vgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint16_t vsrc0,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit immediate-to-VGPR v_mov_b32 packet using an inline vector
// source when possible and a literal VOP1 form otherwise.
iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit VOP2 packet whose first source is an immediate and second
// source is a VGPR. The immediate uses an inline source when possible and a
// literal VOP2 form otherwise.
iree_status_t loom_amdgpu_encoding_pack_vop2_u32_vgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t opcode, uint16_t vdst,
    uint32_t imm32, uint16_t vsrc1, loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 64-bit VOPDXY packet without a literal payload.
iree_status_t loom_amdgpu_encoding_pack_vopdxy(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_vopdxy_fields_t* fields,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 96-bit VOPDXY packet with one shared 32-bit literal payload.
iree_status_t loom_amdgpu_encoding_pack_vopdxy_literal(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_vopdxy_fields_t* fields, uint32_t literal_u32,
    loom_amdgpu_encoding_packet_t* out_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ENCODING_ENCODING_H_
