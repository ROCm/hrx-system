# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Owned AIE2P instruction and bundle encoding facts.

The compact records normalize the physical llvm-aie encodings at the pinned
commit below. LLVM's codegen-only ``MOV_OR`` selector alias is represented by
the byte-identical ``OR`` form instead of becoming a Loom physical instruction.
LLVM-AIE is an extraction and differential-testing oracle only: normal Loom
builds have no dependency on its compiler, TableGen runtime, or source tree.
"""

from __future__ import annotations

from collections.abc import Sequence

from loom.target.arch.amd.xdna.aie.encoding import (
    BitMapping,
    BundleFieldEncoding,
    BundleFormatEncoding,
    BundleInstance,
    EncodingTable,
    EncodingWitness,
    InstructionEncoding,
    InstructionFieldEncoding,
    InstructionInstance,
    bit_range,
)

# Exact LLVM-AIE revision used as the extraction oracle for the owned records
# below. This is provenance for diagnostics and reproducibility, not an AIE2P
# target identity or compatibility key.
LLVM_AIE_SOURCE_COMMIT = "ce8c0f8fd66bff15b347351c67e9fb4fe0a17205"

SLOT_BIT_COUNTS = {
    "alu": 20,
    "lda": 20,
    "ldb": 17,
    "lng": 42,
    "mv": 22,
    "nop": 1,
    "st": 20,
    "vec": 26,
}

# name|slot|fixed value|delay slots|field@target:value:count,...;...
_INSTRUCTION_ENCODING_RECORDS = """\
ABS|alu|220|0|d0@10:0:5;s0@15:0:5
ACQ_COND_mLockId_imm|alu|c10|0|id@14:0:6;s1@5:0:5
ACQ_COND_mLockId_reg|alu|2c10|0|id@15:0:5;s1@5:0:5
ACQ_mLockId_imm|alu|410|0|id@14:0:6;s1@5:0:5
ACQ_mLockId_reg|alu|2410|0|id@15:0:5;s1@5:0:5
ADC|alu|5|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
ADD_NC_mv_add_ri|mv|0|0|dst@15:0:7;imm@2:0:8;s0@10:0:5
ADD_NC_mv_add_rr|mv|a|0|dst@15:0:7;s0@10:0:5;s1@5:0:5
ADD_add_r_ri|alu|6|0|d0@10:0:5;imm@3:0:7;s0@15:0:5
ADD_alu_r_rr|alu|1|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
AND|alu|9|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
ASHL|alu|1d|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
CLB|alu|20|0|d0@10:0:5;s0@15:0:5
CLZ|alu|60|0|d0@10:0:5;s0@15:0:5
DIVS|alu|18|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
DONE|alu|1000|0|-
EQ|alu|f|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
EQZ|alu|1a0|0|d0@10:0:5;s0@15:0:5
EVENT_ERROR|alu|c2000|0|-
EVENT_WARNING|alu|82000|0|-
EVENT_event0|alu|2000|0|-
EVENT_event1|alu|42000|0|-
EXTEND_s16|alu|e0|0|d0@10:0:5;s0@15:0:5
EXTEND_s8|alu|a0|0|d0@10:0:5;s0@15:0:5
EXTEND_u16|alu|160|0|d0@10:0:5;s0@15:0:5
EXTEND_u8|alu|120|0|d0@10:0:5;s0@15:0:5
GE|alu|13|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
GEU|alu|17|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
INVSQRT_mOptConvDel_mRx|alu|188|0|d0@10:0:5;s0@15:0:5
INVSQRT_mOptConvDel_mRx_mOptConv|alu|1c8|0|d0@10:0:5;s0@15:0:5
INVSQRT_mRx|alu|108|0|d0@10:0:5;s0@15:0:5
INVSQRT_mRx_mOptConv|alu|148|0|d0@10:0:5;s0@15:0:5
INV_mOptConvDel_mRx|alu|288|0|d0@10:0:5;s0@15:0:5
INV_mOptConvDel_mRx_mOptConv|alu|2c8|0|d0@10:0:5;s0@15:0:5
INV_mRx|alu|208|0|d0@10:0:5;s0@15:0:5
INV_mRx_mOptConv|alu|248|0|d0@10:0:5;s0@15:0:5
JL_alumv_or|alu|6000|5|a@7:0:3
JL_lng|lng|4|5|i@17:0:20
JNZ|lng|10006|5|i@17:0:20;s0@37:0:5
JNZD|alu|40|5|a@7:0:3;d0@10:0:5;s0@15:0:5
JZ|lng|6|5|i@17:0:20;s0@37:0:5
J_alumv_or|alu|4000|5|a@7:0:3
J_lng|lng|2|5|i@17:0:20
LDA_2D_dms_lda|lda|200d|0|dst@4:0:7;mod@14:0:3;ptr@17:0:3
LDA_2D_dmv_lda_q|lda|204f|0|dst@9:0:2;mod@14:0:3;ptr@17:0:3
LDA_2D_s16|lda|2025|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_2D_s8|lda|2005|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_2D_u16|lda|2035|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_2D_u8|lda|2015|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_3D_dms_lda|lda|300d|0|dst@4:0:7;mod@14:0:2;ptr@17:0:3
LDA_3D_dmv_lda_q|lda|304f|0|dst@9:0:2;mod@14:0:2;ptr@17:0:3
LDA_3D_s16|lda|3025|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
LDA_3D_s8|lda|3005|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
LDA_3D_u16|lda|3035|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
LDA_3D_u8|lda|3015|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
LDA_TM_2D|lda|2027|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_TM_3D|lda|3027|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
LDA_TM_idx|lda|27|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
LDA_TM_idx_imm|lda|827|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_TM_pstm_nrm|lda|1027|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_TM_pstm_nrm_imm|lda|1827|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_dms_lda_idx|lda|d|0|dj@14:0:3;dst@4:0:7;ptr@17:0:3
LDA_dms_lda_idx_imm|lda|80d|0|dst@4:0:7;imm@13:0:4;ptr@17:0:3
LDA_dms_lda_pstm_nrm|lda|100d|0|dst@4:0:7;mod@14:0:3;ptr@17:0:3
LDA_dms_lda_pstm_nrm_imm|lda|180d|0|dst@4:0:7;imm@13:0:4;ptr@17:0:3
LDA_dms_lda_spill|lda|2|0|dst@4:0:7;imm@11:0:9
LDA_dmv_lda_q_idx|lda|4f|0|dj@14:0:3;dst@9:0:2;ptr@17:0:3
LDA_dmv_lda_q_idx_imm|lda|84f|0|dst@9:0:2;imm@13:0:4;ptr@17:0:3
LDA_dmv_lda_q_pstm_nrm|lda|104f|0|dst@9:0:2;mod@14:0:3;ptr@17:0:3
LDA_dmv_lda_q_pstm_nrm_imm|lda|184f|0|dst@9:0:2;imm@13:0:4;ptr@17:0:3
LDA_dmv_lda_q_spill|lda|14f|0|dst@9:0:2;imm@11:0:9
LDA_s16_idx|lda|25|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
LDA_s16_idx_imm|lda|825|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_s16_pstm_nrm|lda|1025|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_s16_pstm_nrm_imm|lda|1825|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_s8_idx|lda|5|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
LDA_s8_idx_imm|lda|805|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_s8_pstm_nrm|lda|1005|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_s8_pstm_nrm_imm|lda|1805|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_u16_idx|lda|35|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
LDA_u16_idx_imm|lda|835|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_u16_pstm_nrm|lda|1035|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_u16_pstm_nrm_imm|lda|1835|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_u8_idx|lda|15|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
LDA_u8_idx_imm|lda|815|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LDA_u8_pstm_nrm|lda|1015|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
LDA_u8_pstm_nrm_imm|lda|1815|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
LSHL|alu|1b|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
LT|alu|15|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
LTU|alu|19|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
MAC|alu|c|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
MOVA|lda|0|0|dst@2:0:7;i@9:0:11
MOVS|st|16|0|dst@14:0:6;src@7:0:7
MOVXM|lng|1|0|dst@15:0:7;i@3:0:12,22:12:20
MOVX_alu_cg|alu|2|0|dst@9:0:6;i@3:0:6,15:6:5
MOVX_mvx_cr_imm|alu|3000|0|dst@7:0:5;src@15:0:5
MOVX_mvx_cr_r|alu|7000|0|dst@7:0:5;src@15:0:5
MOV_CPH_mMStream_tlast_imm|st|4|0|addr@11:0:3;id@6:0:5;nw@4:0:2;op@14:0:2
MOV_CPH_mMStream_tlast_reg|st|40004|0|addr@11:0:3;id@6:0:5;nw@4:0:2;op@14:0:2
MOV_CPH_nb_mMStream_tlast_imm|st|80004|0|addr@11:0:3;id@6:0:5;nw@4:0:2;op@14:0:2
MOV_CPH_nb_mMStream_tlast_reg|st|c0004|0|addr@11:0:3;id@6:0:5;nw@4:0:2;op@14:0:2
MOV_CPH_nb_tlast|st|a0004|0|addr@11:0:3;id@6:0:5;nw@4:0:2;op@14:0:2
MOV_CPH_tlast|st|20004|0|addr@11:0:3;id@6:0:5;nw@4:0:2;op@14:0:2
MOV_PH_mMStream_tlast_imm|st|10004|0|id@6:0:5;pcktType@11:0:3
MOV_PH_mMStream_tlast_reg|st|50004|0|id@6:0:5;pcktType@11:0:3
MOV_PH_nb_mMStream_tlast_imm|st|90004|0|id@6:0:5;pcktType@11:0:3
MOV_PH_nb_mMStream_tlast_reg|st|d0004|0|id@6:0:5;pcktType@11:0:3
MOV_PH_nb_tlast|st|b0004|0|id@6:0:5;pcktType@11:0:3
MOV_PH_tlast|st|30004|0|id@6:0:5;pcktType@11:0:3
MOV_alu_mv_alu_flt2fx|alu|28|0|d0@10:0:5;s0@15:0:5
MOV_alu_mv_alu_fx2flt|alu|68|0|d0@10:0:5;s0@15:0:5
MOV_alu_mv_mv_mv_cg|mv|5|0|dst@15:0:7;i@4:0:11
MOV_alu_mv_mv_mv_cntr2l|mv|a57|0|dst@18:0:4
MOV_alu_mv_mv_mv_e_mv_eh_to_eh|mv|208e7|0|dst@18:0:4;src@12:0:4
MOV_alu_mv_mv_mv_e_mv_eh_to_el|mv|8e7|0|dst@18:0:4;src@12:0:4
MOV_alu_mv_mv_mv_e_mv_eh_to_r|mv|108e7|0|dst@17:0:5;src@12:0:4
MOV_alu_mv_mv_mv_e_mv_el_to_eh|mv|200e7|0|dst@18:0:4;src@12:0:4
MOV_alu_mv_mv_mv_e_mv_el_to_el|mv|e7|0|dst@18:0:4;src@12:0:4
MOV_alu_mv_mv_mv_e_mv_el_to_r|mv|100e7|0|dst@17:0:5;src@12:0:4
MOV_alu_mv_mv_mv_e_mv_r_to_eh|mv|204e7|0|dst@18:0:4;src@11:0:5
MOV_alu_mv_mv_mv_e_mv_r_to_el|mv|4e7|0|dst@18:0:4;src@11:0:5
MOV_alu_mv_mv_mv_scl|mv|7|0|dst@15:0:7;src@8:0:7
MOV_d1|mv|2b|0|dst@17:0:5;src@8:0:7
MOV_d2|mv|ab|0|dst@17:0:5;src@8:0:7
MOV_d3|mv|802b|0|dst@17:0:5;src@8:0:7
MOV_d4|mv|80ab|0|dst@17:0:5;src@8:0:7
MOV_d5|mv|1002b|0|dst@17:0:5;src@8:0:7
MOV_d6|mv|100ab|0|dst@17:0:5;src@8:0:7
MOV_lda|lda|7|0|dst@6:0:5
MOV_nb_lda|lda|80007|0|dst@6:0:5
MOV_nb_st_mMStream_tlast_imm|st|94004|0|src@4:0:7
MOV_nb_st_mMStream_tlast_reg|st|d4004|0|src@4:0:7
MOV_nb_tlast|st|b4004|0|src@4:0:7
MOV_st_mMStream_tlast_imm|st|14004|0|src@4:0:7
MOV_st_mMStream_tlast_reg|st|54004|0|src@4:0:7
MOV_tlast|st|34004|0|src@4:0:7
MSC|alu|1c|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
MUL|alu|1f|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
NE|alu|11|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
NEZ|alu|1e0|0|d0@10:0:5;s0@15:0:5
NOP|nop|0|0|-
NOPA|lda|2cf|0|-
NOPB|ldb|4|0|-
NOPM|mv|1a57|0|-
NOPS|st|2b6|0|-
NOPV|vec|70|0|-
NOPX|alu|0|0|-
NOPXM|lng|0|0|-
OR|alu|b|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
PADDA_2D|lda|20cf|0|mod@14:0:3;ptr@17:0:3
PADDA_3D|lda|30cf|0|mod@14:0:2;ptr@17:0:3
PADDA_pstm_nrm|lda|10cf|0|mod@14:0:3;ptr@17:0:3
PADDA_pstm_nrm_imm|lda|18cf|0|imm@13:0:4;ptr@17:0:3
PADDB_2D|ldb|4e4|0|mod@11:0:3;ptr@14:0:3
PADDB_3D|ldb|6e4|0|mod@11:0:2;ptr@14:0:3
PADDB_pstm_nrm|ldb|2e4|0|mod@11:0:3;ptr@14:0:3
PADDB_pstm_nrm_imm|ldb|3e4|0|imm@10:0:4;ptr@14:0:3
PADDS_2D|st|20b6|0|mod@14:0:3;ptr@17:0:3
PADDS_3D|st|30b6|0|mod@14:0:2;ptr@17:0:3
PADDS_pstm_nrm|st|10b6|0|mod@14:0:3;ptr@17:0:3
PADDS_pstm_nrm_imm|st|18b6|0|imm@13:0:4;ptr@17:0:3
PADDXM_pstm_sp|lng|10000007|0|mod@39:0:3
PADDXM_pstm_sp_imm|lng|7|0|imm@29:0:13
POPCOUNT|alu|260|0|d0@10:0:5;s0@15:0:5
REL_COND_mLockId_imm|alu|810|0|id@14:0:6;s1@5:0:5
REL_COND_mLockId_reg|alu|2810|0|id@15:0:5;s1@5:0:5
REL_mLockId_imm|alu|10|0|id@14:0:6;s1@5:0:5
REL_mLockId_reg|alu|2010|0|id@15:0:5;s1@5:0:5
RET|alu|5000|5|-
SBC|alu|7|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
SEL_EQZ|alu|4|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
SEL_NEZ|alu|14|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
SQRT_mOptConvDel_mRx|alu|88|0|d0@10:0:5;s0@15:0:5
SQRT_mOptConvDel_mRx_mOptConv|alu|c8|0|d0@10:0:5;s0@15:0:5
SQRT_mRx|alu|8|0|d0@10:0:5;s0@15:0:5
SQRT_mRx_mOptConv|alu|48|0|d0@10:0:5;s0@15:0:5
ST_2D_dms_sts|st|2003|0|mod@14:0:3;ptr@17:0:3;src@4:0:7
ST_2D_dmv_sts_q|st|2036|0|mod@14:0:3;ptr@17:0:3;src@9:0:2
ST_2D_s16|lda|202e|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
ST_2D_s8|lda|200e|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
ST_3D_dms_sts|st|3003|0|mod@14:0:2;ptr@17:0:3;src@4:0:7
ST_3D_dmv_sts_q|st|3036|0|mod@14:0:2;ptr@17:0:3;src@9:0:2
ST_3D_s16|lda|302e|0|mod@14:0:2;ptr@17:0:3;src@6:0:5
ST_3D_s8|lda|300e|0|mod@14:0:2;ptr@17:0:3;src@6:0:5
ST_TM_2D|st|203d|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
ST_TM_3D|st|303d|0|mod@14:0:2;ptr@17:0:3;src@6:0:5
ST_TM_idx|st|3d|0|dj@14:0:3;ptr@17:0:3;src@6:0:5
ST_TM_idx_imm|st|83d|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
ST_TM_pstm_nrm|st|103d|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
ST_TM_pstm_nrm_imm|st|183d|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
ST_dms_sts_idx|st|3|0|dj@14:0:3;ptr@17:0:3;src@4:0:7
ST_dms_sts_idx_imm|st|803|0|imm@13:0:4;ptr@17:0:3;src@4:0:7
ST_dms_sts_pstm_nrm|st|1003|0|mod@14:0:3;ptr@17:0:3;src@4:0:7
ST_dms_sts_pstm_nrm_imm|st|1803|0|imm@13:0:4;ptr@17:0:3;src@4:0:7
ST_dms_sts_spill|st|b|0|imm@11:0:9;src@4:0:7
ST_dmv_sts_q_idx|st|36|0|dj@14:0:3;ptr@17:0:3;src@9:0:2
ST_dmv_sts_q_idx_imm|st|836|0|imm@13:0:4;ptr@17:0:3;src@9:0:2
ST_dmv_sts_q_pstm_nrm|st|1036|0|mod@14:0:3;ptr@17:0:3;src@9:0:2
ST_dmv_sts_q_pstm_nrm_imm|st|1836|0|imm@13:0:4;ptr@17:0:3;src@9:0:2
ST_dmv_sts_q_spill|st|136|0|imm@11:0:9;src@9:0:2
ST_s16_idx|lda|2e|0|dj@14:0:3;ptr@17:0:3;src@6:0:5
ST_s16_idx_imm|lda|82e|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
ST_s16_pstm_nrm|lda|102e|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
ST_s16_pstm_nrm_imm|lda|182e|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
ST_s8_idx|lda|e|0|dj@14:0:3;ptr@17:0:3;src@6:0:5
ST_s8_idx_imm|lda|80e|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
ST_s8_pstm_nrm|lda|100e|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
ST_s8_pstm_nrm_imm|lda|180e|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
SUB|alu|3|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
VABS_GTZ_16_vaddSign0|mv|1597|0|d@18:0:4;s2@14:0:4
VABS_GTZ_16_vaddSign1|mv|1d97|0|d@18:0:4;s2@14:0:4
VABS_GTZ_32_vaddSign0|mv|597|0|d@18:0:4;s2@14:0:4
VABS_GTZ_32_vaddSign1|mv|d97|0|d@18:0:4;s2@14:0:4
VABS_GTZ_8_vaddSign0|mv|2597|0|d@18:0:4;s2@14:0:4
VABS_GTZ_8_vaddSign1|mv|2d97|0|d@18:0:4;s2@14:0:4
VADDMAC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_X_X|vec|1|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_Y_Y|vec|41|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@12:0:3;s2@8:0:3
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_X_X|vec|1c0001|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_Y_Y|vec|1c0041|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@8:0:3
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_X_X|vec|180001|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_Y_Y|vec|180041|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@8:0:3
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EX|vec|21|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EY|vec|11|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@8:0:3
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_QEY|vec|51|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@10:0:1
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EY_QEX|vec|61|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@12:0:3;s2@9:0:2
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EX|vec|1c0021|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EY|vec|1c0011|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@8:0:3
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_QEY|vec|1c0051|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@10:0:1
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EY_QEX|vec|1c0061|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@9:0:2
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EX|vec|180021|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EY|vec|180011|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@8:0:3
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_QEY|vec|180051|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@10:0:1
VADDMAC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EY_QEX|vec|180061|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@9:0:2
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_QX|vec|40|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@9:0:2
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_QY|vec|a0|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@10:0:1
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X|vec|0|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_Y|vec|20|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@8:0:3
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_QX|vec|860|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@9:0:2
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_QY|vec|d0|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@10:0:1
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_X|vec|60|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@7:0:4
VADDMAC_vmac_cm2_add_reg_vmul_cm_core_Y_Y|vec|50|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@8:0:3
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QX|vec|38040|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@9:0:2
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QY|vec|380a0|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@10:0:1
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_X|vec|38000|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_X_Y|vec|38020|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@8:0:3
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QX|vec|38860|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@9:0:2
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QY|vec|380d0|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@10:0:1
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_X|vec|38060|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@7:0:4
VADDMAC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_Y|vec|38050|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@8:0:3
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_QX|vec|30040|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@9:0:2
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_QY|vec|300a0|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@10:0:1
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_X|vec|30000|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@7:0:4
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_X_Y|vec|30020|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@8:0:3
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_QX|vec|30860|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@9:0:2
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_QY|vec|300d0|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@10:0:1
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_X|vec|30060|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@7:0:4
VADDMAC_vmac_cm2_add_scd_vmul_cm_core_Y_Y|vec|30050|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@8:0:3
VADDMSC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_X_X|vec|9|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_f_vaddmac_bf_vmac_cm2_add_reg_vmul_bf_core_Y_Y|vec|49|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@12:0:3;s2@8:0:3
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_X_X|vec|1c0009|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_incr_vmul_bf_core_Y_Y|vec|1c0049|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@8:0:3
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_X_X|vec|180009|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_f_vaddmac_bf_vmac_cm2_add_scd_vmul_bf_core_Y_Y|vec|180049|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@8:0:3
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EX|vec|29|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_EY|vec|19|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@8:0:3
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EX_QEY|vec|59|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@11:0:4;s2@10:0:1
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_reg_vmul_bfp_core_EY_QEX|vec|69|0|acc@21:0:5;acc1@15:0:3;acc2@18:0:3;s1@12:0:3;s2@9:0:2
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EX|vec|1c0029|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_EY|vec|1c0019|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@8:0:3
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EX_QEY|vec|1c0059|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@10:0:1
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_incr_vmul_bfp_core_EY_QEX|vec|1c0069|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@9:0:2
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EX|vec|180029|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_EY|vec|180019|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@8:0:3
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EX_QEY|vec|180059|0|acc@21:0:5;acc1@15:0:3;s1@11:0:4;s2@10:0:1
VADDMSC_f_vaddmac_bfp_vmac_cm2_add_scd_vmul_bfp_core_EY_QEX|vec|180069|0|acc@21:0:5;acc1@15:0:3;s1@12:0:3;s2@9:0:2
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_QX|vec|48|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@9:0:2
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_QY|vec|a8|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@10:0:1
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_X|vec|8|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_X_Y|vec|28|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@11:0:4;s2@8:0:3
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_QX|vec|868|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@9:0:2
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_QY|vec|d8|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@10:0:1
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_X|vec|68|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@7:0:4
VADDMSC_vmac_cm2_add_reg_vmul_cm_core_Y_Y|vec|58|0|acc@21:0:5;acc1@18:0:3;acc2@15:0:3;s1@12:0:3;s2@8:0:3
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QX|vec|38048|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@9:0:2
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_QY|vec|380a8|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@10:0:1
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_X|vec|38008|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_X_Y|vec|38028|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@8:0:3
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QX|vec|38868|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@9:0:2
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_QY|vec|380d8|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@10:0:1
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_X|vec|38068|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@7:0:4
VADDMSC_vmac_cm2_add_scd_incr_vmul_cm_core_Y_Y|vec|38058|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@8:0:3
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_QX|vec|30048|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@9:0:2
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_QY|vec|300a8|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@10:0:1
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_X|vec|30008|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@7:0:4
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_X_Y|vec|30028|0|acc@21:0:5;acc1@18:0:3;s1@11:0:4;s2@8:0:3
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_QX|vec|30868|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@9:0:2
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_QY|vec|300d8|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@10:0:1
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_X|vec|30068|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@7:0:4
VADDMSC_vmac_cm2_add_scd_vmul_cm_core_Y_Y|vec|30058|0|acc@21:0:5;acc1@18:0:3;s1@12:0:3;s2@8:0:3
VADDSUB_16|mv|19|0|d@18:0:4;s1@14:0:4;s2@10:0:4;sel@6:0:4
VADDSUB_32|mv|9|0|d@18:0:4;s1@14:0:4;s2@10:0:4;sel@6:0:4
VADDSUB_8|mv|29|0|d@18:0:4;s1@14:0:4;s2@10:0:4;sel@6:0:4
VADD_16|mv|eb|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VADD_32|mv|16b|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VADD_8|mv|6b|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VADD_f_vmac_cm2_add_reg|vec|f5|0|acc@21:0:5;acc1@15:0:3;acc2@12:0:3;dst@18:0:3
VADD_f_vmac_cm2_add_scd|vec|60f5|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VADD_f_vmac_cm2_add_scd_incr|vec|70f5|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VADD_vmac_cm2_add_reg|vec|f4|0|acc@21:0:5;acc1@15:0:3;acc2@12:0:3;dst@18:0:3
VADD_vmac_cm2_add_scd|vec|60f4|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VADD_vmac_cm2_add_scd_incr|vec|70f4|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VBAND|mv|15b|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VBCSTSHFL_16_vec_broadcast_shuffle_bm|mv|a97|0|dst@18:0:4;src@13:0:5
VBCSTSHFL_16_vec_broadcast_shuffle_x|mv|e97|0|dst@18:0:4;src@13:0:5
VBCSTSHFL_32_vec_broadcast_shuffle_bm|mv|1297|0|dst@18:0:4;src@13:0:5
VBCSTSHFL_32_vec_broadcast_shuffle_x|mv|1697|0|dst@18:0:4;src@13:0:5
VBCSTSHFL_64_vec_broadcast_shuffle_bm|mv|1a97|0|dst@18:0:4;src@14:0:4
VBCSTSHFL_64_vec_broadcast_shuffle_x|mv|1e97|0|dst@18:0:4;src@14:0:4
VBCSTSHFL_8_vec_broadcast_shuffle_bm|mv|297|0|dst@18:0:4;src@13:0:5
VBCSTSHFL_8_vec_broadcast_shuffle_x|mv|697|0|dst@18:0:4;src@13:0:5
VBCST_16|mv|b97|0|dst@18:0:4;src@13:0:5
VBCST_32|mv|1397|0|dst@18:0:4;src@13:0:5
VBCST_64|mv|1b97|0|dst@18:0:4;src@14:0:4
VBCST_8|mv|397|0|dst@18:0:4;src@13:0:5
VBNEG_LTZ_s16|mv|1057|0|d@18:0:4;s2@14:0:4
VBNEG_LTZ_s32|mv|57|0|d@18:0:4;s2@14:0:4
VBNEG_LTZ_s8|mv|2057|0|d@18:0:4;s2@14:0:4
VBOR|mv|35b|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VCLR|vec|38074|0|dst@18:0:3
VCONV_bf16_fp32_mv_w_srs_bf|st|2c|0|dst@15:0:5;src@6:0:5
VCONV_bf16_fp32_mv_x_srs_bf|st|202c|0|dst@16:0:4;src@7:0:4
VCONV_bfp16ebs16_ebs8|st|102c|0|dst@16:0:4;src@7:0:4
VCONV_bfp16ebs16_fp32|st|10ec|0|dst@16:0:4;src@8:0:3
VCONV_bfp16ebs8_fp32|st|106c|0|dst@16:0:4;src@8:0:3
VCONV_fp32_bf16_mv_ups_wbf|mv|197|0|dst@17:0:5;src@11:0:5
VCONV_fp32_bf16_mv_ups_xbf|mv|c57|0|dst@18:0:4;src@12:0:4
VEQZ_16|mv|1257|0|cmp@18:0:4;s2@14:0:4
VEQZ_32|mv|2257|0|cmp@18:0:4;s2@14:0:4
VEQZ_8|mv|257|0|cmp@18:0:4;s2@14:0:4
VEXP2|mv|797|0|dst@17:0:5;src@12:0:5
VEXTBCSTSHFL_128|mv|1b|0|dst@18:0:4;idx@9:0:5;s1@14:0:4
VEXTBCSTSHFL_16|mv|9b|0|dst@18:0:4;idx@9:0:5;s1@14:0:4
VEXTBCSTSHFL_32|mv|11b|0|dst@18:0:4;idx@9:0:5;s1@14:0:4
VEXTBCSTSHFL_64|mv|19b|0|dst@18:0:4;idx@9:0:5;s1@14:0:4
VEXTBCST_128_vec_extract_broadcast_imm|mv|201a|0|dst@18:0:4;idx@5:0:6;s1@14:0:4
VEXTBCST_128_vec_extract_broadcast_r|mv|2033|0|dst@18:0:4;idx@6:0:5;s1@14:0:4
VEXTBCST_16_vec_extract_broadcast_imm|mv|81a|0|dst@18:0:4;idx@5:0:6;s1@14:0:4
VEXTBCST_16_vec_extract_broadcast_r|mv|833|0|dst@18:0:4;idx@6:0:5;s1@14:0:4
VEXTBCST_32_vec_extract_broadcast_imm|mv|101a|0|dst@18:0:4;idx@5:0:6;s1@14:0:4
VEXTBCST_32_vec_extract_broadcast_r|mv|1033|0|dst@18:0:4;idx@6:0:5;s1@14:0:4
VEXTBCST_64_vec_extract_broadcast_imm|mv|181a|0|dst@18:0:4;idx@5:0:6;s1@14:0:4
VEXTBCST_64_vec_extract_broadcast_r|mv|1833|0|dst@18:0:4;idx@6:0:5;s1@14:0:4
VEXTBCST_8_vec_extract_broadcast_imm|mv|1a|0|dst@18:0:4;idx@5:0:6;s1@14:0:4
VEXTBCST_8_vec_extract_broadcast_r|mv|33|0|dst@18:0:4;idx@6:0:5;s1@14:0:4
VEXTRACT_16_vec_extract_imm_vaddSign0|mv|80d|0|dst@17:0:5;idx@4:0:6;s1@13:0:4
VEXTRACT_16_vec_extract_imm_vaddSign1|mv|c0d|0|dst@17:0:5;idx@4:0:6;s1@13:0:4
VEXTRACT_16_vec_extract_r_vaddSign0|mv|806|0|dst@17:0:5;idx@5:0:5;s1@13:0:4
VEXTRACT_16_vec_extract_r_vaddSign1|mv|c06|0|dst@17:0:5;idx@5:0:5;s1@13:0:4
VEXTRACT_32_vec_extract_imm_vaddSign0|mv|100d|0|dst@17:0:5;idx@4:0:6;s1@13:0:4
VEXTRACT_32_vec_extract_imm_vaddSign1|mv|140d|0|dst@17:0:5;idx@4:0:6;s1@13:0:4
VEXTRACT_32_vec_extract_r_vaddSign0|mv|1006|0|dst@17:0:5;idx@5:0:5;s1@13:0:4
VEXTRACT_32_vec_extract_r_vaddSign1|mv|1406|0|dst@17:0:5;idx@5:0:5;s1@13:0:4
VEXTRACT_64_vec_extract_imm_vaddSign0|mv|180d|0|dst@18:0:4;idx@4:0:6;s1@13:0:4
VEXTRACT_64_vec_extract_imm_vaddSign1|mv|1c0d|0|dst@18:0:4;idx@4:0:6;s1@13:0:4
VEXTRACT_64_vec_extract_r_vaddSign0|mv|1806|0|dst@18:0:4;idx@5:0:5;s1@13:0:4
VEXTRACT_64_vec_extract_r_vaddSign1|mv|1c06|0|dst@18:0:4;idx@5:0:5;s1@13:0:4
VEXTRACT_8_vec_extract_imm_vaddSign0|mv|d|0|dst@17:0:5;idx@4:0:6;s1@13:0:4
VEXTRACT_8_vec_extract_imm_vaddSign1|mv|40d|0|dst@17:0:5;idx@4:0:6;s1@13:0:4
VEXTRACT_8_vec_extract_r_vaddSign0|mv|6|0|dst@17:0:5;idx@5:0:5;s1@13:0:4
VEXTRACT_8_vec_extract_r_vaddSign1|mv|406|0|dst@17:0:5;idx@5:0:5;s1@13:0:4
VFLOOR_s32_bf16_mv_float_to_int_bm|st|c02c|0|dst@16:0:4;imm@11:0:1;shft@12:0:2;src@6:0:5
VFLOOR_s32_bf16_mv_float_to_int_w|st|402c|0|dst@16:0:4;shft@12:0:2;src@6:0:5
VGE_16_vaddSign0|mv|253|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VGE_16_vaddSign1|mv|353|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VGE_32_vaddSign0|mv|293|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VGE_32_vaddSign1|mv|393|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VGE_8_vaddSign0|mv|213|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VGE_8_vaddSign1|mv|313|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VGE_bf16|mv|267|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VINSERT_16_mIdxImm0|mv|4b|0|dst@18:0:4;s1@14:0:4;src@8:0:5
VINSERT_16_mR29_insert|mv|204b|0|dst@18:0:4;s1@14:0:4;src@8:0:5
VINSERT_32_mIdxImm0|mv|8b|0|dst@18:0:4;s1@14:0:4;src@8:0:5
VINSERT_32_mR29_insert|mv|208b|0|dst@18:0:4;s1@14:0:4;src@8:0:5
VINSERT_64_mIdxImm0|mv|cb|0|dst@18:0:4;s1@14:0:4;src@9:0:4
VINSERT_64_mR29_insert|mv|20cb|0|dst@18:0:4;s1@14:0:4;src@9:0:4
VINSERT_8_mIdxImm0|mv|b|0|dst@18:0:4;s1@14:0:4;src@8:0:5
VINSERT_8_mR29_insert|mv|200b|0|dst@18:0:4;s1@14:0:4;src@8:0:5
VLDA_128_dmv_lda_w_idx|lda|3|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
VLDA_128_dmv_lda_w_idx_imm|lda|803|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
VLDA_128_dmv_lda_w_pstm_nrm|lda|1003|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
VLDA_128_dmv_lda_w_pstm_nrm_imm|lda|1803|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
VLDA_128_dmv_lda_w_spill|lda|23|0|dst@6:0:5;imm@11:0:9
VLDA_2D_128|lda|2003|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
VLDA_2D_CONV_fp32_bf16_dmw_lda_ups_bf|lda|2013|0|mod@14:0:3;op@6:0:5;ptr@17:0:3
VLDA_2D_CONV_fp32_bf16_dmx_lda_ups_bf|lda|2057|0|mod@14:0:3;op@7:0:4;ptr@17:0:3
VLDA_2D_UPS_2x_dmw_lda_ups_w2b_upsSign0|lda|2001|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3;su@4:0:2
VLDA_2D_UPS_2x_dmw_lda_ups_w2b_upsSign1|lda|2009|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3;su@4:0:2
VLDA_2D_UPS_2x_dmx_lda_ups_x2c_upsSign0|lda|2006|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_2D_UPS_2x_dmx_lda_ups_x2c_upsSign1|lda|2016|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_2D_UPS_4x_dmw_lda_ups_w2c_upsSign0|lda|200a|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_2D_UPS_4x_dmw_lda_ups_w2c_upsSign1|lda|201a|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_2D_UPS_4x_dmx_lda_ups_x2d_upsSign0|lda|201e|0|dst@8:0:3;mod@14:0:3;ptr@17:0:3;su@6:0:2
VLDA_2D_UPS_4x_dmx_lda_ups_x2d_upsSign1|lda|203e|0|dst@8:0:3;mod@14:0:3;ptr@17:0:3;su@6:0:2
VLDA_2D_dmw_lda_w|lda|2033|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
VLDA_2D_dmx_lda_bm|lda|202b|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
VLDA_2D_dmx_lda_fifohl|lda|200f|0|dst@8:0:3;mod@14:0:3;ptr@17:0:3
VLDA_2D_dmx_lda_x|lda|2037|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3
VLDA_3D_128|lda|3003|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
VLDA_3D_CONV_fp32_bf16_dmw_lda_ups_bf|lda|3013|0|mod@14:0:2;op@6:0:5;ptr@17:0:3
VLDA_3D_CONV_fp32_bf16_dmx_lda_ups_bf|lda|3057|0|mod@14:0:2;op@7:0:4;ptr@17:0:3
VLDA_3D_UPS_2x_dmw_lda_ups_w2b_upsSign0|lda|3001|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3;su@4:0:2
VLDA_3D_UPS_2x_dmw_lda_ups_w2b_upsSign1|lda|3009|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3;su@4:0:2
VLDA_3D_UPS_2x_dmx_lda_ups_x2c_upsSign0|lda|3006|0|dst@7:0:4;mod@14:0:2;ptr@17:0:3;su@5:0:2
VLDA_3D_UPS_2x_dmx_lda_ups_x2c_upsSign1|lda|3016|0|dst@7:0:4;mod@14:0:2;ptr@17:0:3;su@5:0:2
VLDA_3D_UPS_4x_dmw_lda_ups_w2c_upsSign0|lda|300a|0|dst@7:0:4;mod@14:0:2;ptr@17:0:3;su@5:0:2
VLDA_3D_UPS_4x_dmw_lda_ups_w2c_upsSign1|lda|301a|0|dst@7:0:4;mod@14:0:2;ptr@17:0:3;su@5:0:2
VLDA_3D_UPS_4x_dmx_lda_ups_x2d_upsSign0|lda|301e|0|dst@8:0:3;mod@14:0:2;ptr@17:0:3;su@6:0:2
VLDA_3D_UPS_4x_dmx_lda_ups_x2d_upsSign1|lda|303e|0|dst@8:0:3;mod@14:0:2;ptr@17:0:3;su@6:0:2
VLDA_3D_dmw_lda_w|lda|3033|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
VLDA_3D_dmx_lda_bm|lda|302b|0|dst@6:0:5;mod@14:0:2;ptr@17:0:3
VLDA_3D_dmx_lda_fifohl|lda|300f|0|dst@8:0:3;mod@14:0:2;ptr@17:0:3
VLDA_3D_dmx_lda_x|lda|3037|0|dst@7:0:4;mod@14:0:2;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_idx|lda|13|0|dj@14:0:3;op@6:0:5;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_idx_imm|lda|813|0|imm@13:0:4;op@6:0:5;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_pstm_nrm|lda|1013|0|mod@14:0:3;op@6:0:5;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmw_lda_ups_bf_pstm_nrm_imm|lda|1813|0|imm@13:0:4;op@6:0:5;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_idx|lda|57|0|dj@14:0:3;op@7:0:4;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_idx_imm|lda|857|0|imm@13:0:4;op@7:0:4;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm|lda|1057|0|mod@14:0:3;op@7:0:4;ptr@17:0:3
VLDA_CONV_fp32_bf16_dmx_lda_ups_bf_pstm_nrm_imm|lda|1857|0|imm@13:0:4;op@7:0:4;ptr@17:0:3
VLDA_FILL_512|lda|2817|0|ptr@17:0:1
VLDA_POP_512_2D|lda|42017|0|dst@7:0:4;mod@14:0:3;ptr@17:0:1
VLDA_POP_512_3D|lda|c0017|0|dst@7:0:4;mod@14:0:2;ptr@17:0:1
VLDA_POP_512_fifo_1d_pop|lda|81017|0|dst@7:0:4;mod@14:0:3;ptr@17:0:1
VLDA_POP_512_normal_pop|lda|17|0|dst@7:0:4;ptr@17:0:1
VLDA_POP_544_2D|lda|1017|0|dst@7:0:4;mod@14:0:3;ptr@17:0:1
VLDA_POP_544_3D|lda|82017|0|dst@7:0:4;mod@14:0:2;ptr@17:0:1
VLDA_POP_544_fifo_1d_pop|lda|c1017|0|dst@7:0:4;mod@14:0:3;ptr@17:0:1
VLDA_POP_544_normal_pop|lda|40017|0|dst@7:0:4;ptr@17:0:1
VLDA_POP_576_2D|lda|c2017|0|dst@7:0:4;mod@14:0:3;ptr@17:0:1
VLDA_POP_576_3D|lda|2017|0|dst@7:0:4;mod@14:0:2;ptr@17:0:1
VLDA_POP_576_fifo_1d_pop|lda|41017|0|dst@7:0:4;mod@14:0:3;ptr@17:0:1
VLDA_POP_576_normal_pop|lda|80017|0|dst@7:0:4;ptr@17:0:1
VLDA_POP_640_2D|lda|817|0|dst@9:0:2;mod@14:0:3;ptr@17:0:1
VLDA_POP_640_3D|lda|43017|0|dst@9:0:2;mod@14:0:2;ptr@17:0:1
VLDA_POP_640_fifo_1d_pop|lda|40817|0|dst@9:0:2;mod@14:0:3;ptr@17:0:1
VLDA_POP_640_normal_pop|lda|3017|0|dst@9:0:2;ptr@17:0:1
VLDA_POP_704_2D|lda|80817|0|dst@9:0:2;mod@14:0:3;ptr@17:0:1
VLDA_POP_704_3D|lda|c3017|0|dst@9:0:2;mod@14:0:2;ptr@17:0:1
VLDA_POP_704_fifo_1d_pop|lda|c0817|0|dst@9:0:2;mod@14:0:3;ptr@17:0:1
VLDA_POP_704_normal_pop|lda|83017|0|dst@9:0:2;ptr@17:0:1
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_imm_upsSign0|lda|801|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_imm_upsSign1|lda|809|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_upsSign0|lda|1|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmw_lda_ups_w2b_idx_upsSign1|lda|9|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_imm_upsSign0|lda|1801|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_imm_upsSign1|lda|1809|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_upsSign0|lda|1001|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmw_lda_ups_w2b_pstm_nrm_upsSign1|lda|1009|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3;su@4:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_imm_upsSign0|lda|806|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_imm_upsSign1|lda|816|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_upsSign0|lda|6|0|dj@14:0:3;dst@7:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_idx_upsSign1|lda|16|0|dj@14:0:3;dst@7:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_imm_upsSign0|lda|1806|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_imm_upsSign1|lda|1816|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_upsSign0|lda|1006|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_UPS_2x_dmx_lda_ups_x2c_pstm_nrm_upsSign1|lda|1016|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_imm_upsSign0|lda|80a|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_imm_upsSign1|lda|81a|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_upsSign0|lda|a|0|dj@14:0:3;dst@7:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_idx_upsSign1|lda|1a|0|dj@14:0:3;dst@7:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_imm_upsSign0|lda|180a|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_imm_upsSign1|lda|181a|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_upsSign0|lda|100a|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmw_lda_ups_w2c_pstm_nrm_upsSign1|lda|101a|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3;su@5:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_imm_upsSign0|lda|81e|0|dst@8:0:3;imm@13:0:4;ptr@17:0:3;su@6:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_imm_upsSign1|lda|83e|0|dst@8:0:3;imm@13:0:4;ptr@17:0:3;su@6:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_upsSign0|lda|1e|0|dj@14:0:3;dst@8:0:3;ptr@17:0:3;su@6:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_idx_upsSign1|lda|3e|0|dj@14:0:3;dst@8:0:3;ptr@17:0:3;su@6:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_imm_upsSign0|lda|181e|0|dst@8:0:3;imm@13:0:4;ptr@17:0:3;su@6:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_imm_upsSign1|lda|183e|0|dst@8:0:3;imm@13:0:4;ptr@17:0:3;su@6:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_upsSign0|lda|101e|0|dst@8:0:3;mod@14:0:3;ptr@17:0:3;su@6:0:2
VLDA_UPS_4x_dmx_lda_ups_x2d_pstm_nrm_upsSign1|lda|103e|0|dst@8:0:3;mod@14:0:3;ptr@17:0:3;su@6:0:2
VLDA_dmw_lda_w_idx|lda|33|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
VLDA_dmw_lda_w_idx_imm|lda|833|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
VLDA_dmw_lda_w_pstm_nrm|lda|1033|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
VLDA_dmw_lda_w_pstm_nrm_imm|lda|1833|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
VLDA_dmw_lda_w_spill|lda|b|0|dst@6:0:5;imm@11:0:9
VLDA_dmx_lda_bm_idx|lda|2b|0|dj@14:0:3;dst@6:0:5;ptr@17:0:3
VLDA_dmx_lda_bm_idx_imm|lda|82b|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
VLDA_dmx_lda_bm_pstm_nrm|lda|102b|0|dst@6:0:5;mod@14:0:3;ptr@17:0:3
VLDA_dmx_lda_bm_pstm_nrm_imm|lda|182b|0|dst@6:0:5;imm@13:0:4;ptr@17:0:3
VLDA_dmx_lda_bm_spill|lda|1b|0|dst@6:0:5;imm@11:0:9
VLDA_dmx_lda_fifohl_idx|lda|f|0|dj@14:0:3;dst@8:0:3;ptr@17:0:3
VLDA_dmx_lda_fifohl_idx_imm|lda|80f|0|dst@8:0:3;imm@13:0:4;ptr@17:0:3
VLDA_dmx_lda_fifohl_pstm_nrm|lda|100f|0|dst@8:0:3;mod@14:0:3;ptr@17:0:3
VLDA_dmx_lda_fifohl_pstm_nrm_imm|lda|180f|0|dst@8:0:3;imm@13:0:4;ptr@17:0:3
VLDA_dmx_lda_fifohl_spill|lda|8f|0|dst@8:0:3;imm@11:0:9
VLDA_dmx_lda_x_idx|lda|37|0|dj@14:0:3;dst@7:0:4;ptr@17:0:3
VLDA_dmx_lda_x_idx_imm|lda|837|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3
VLDA_dmx_lda_x_pstm_nrm|lda|1037|0|dst@7:0:4;mod@14:0:3;ptr@17:0:3
VLDA_dmx_lda_x_pstm_nrm_imm|lda|1837|0|dst@7:0:4;imm@13:0:4;ptr@17:0:3
VLDA_dmx_lda_x_spill|lda|77|0|dst@7:0:4;imm@11:0:9
VLDB_128_idx|ldb|0|0|dj@11:0:3;dst@3:0:5;ptr@14:0:3
VLDB_128_idx_imm|ldb|100|0|dst@3:0:5;imm@10:0:4;ptr@14:0:3
VLDB_128_pstm_nrm|ldb|200|0|dst@3:0:5;mod@11:0:3;ptr@14:0:3
VLDB_128_pstm_nrm_imm|ldb|300|0|dst@3:0:5;imm@10:0:4;ptr@14:0:3
VLDB_2D_128|ldb|400|0|dst@3:0:5;mod@11:0:3;ptr@14:0:3
VLDB_2D_UNPACK_dmw_ldb_unpack_unpackSign0|ldb|402|0|dst@4:0:4;mod@11:0:3;ptr@14:0:3
VLDB_2D_UNPACK_dmw_ldb_unpack_unpackSign1|ldb|40a|0|dst@4:0:4;mod@11:0:3;ptr@14:0:3
VLDB_2D_UNPACK_dmx_ldb_unpack_unpackSign0|ldb|403|0|dst@5:0:3;mod@11:0:3;ptr@14:0:3
VLDB_2D_UNPACK_dmx_ldb_unpack_unpackSign1|ldb|40b|0|dst@5:0:3;mod@11:0:3;ptr@14:0:3
VLDB_2D_dmw_ldb|ldb|401|0|dst@3:0:5;mod@11:0:3;ptr@14:0:3
VLDB_2D_dmx_ldb_x|ldb|40d|0|dst@4:0:4;mod@11:0:3;ptr@14:0:3
VLDB_3D_128|ldb|600|0|dst@3:0:5;mod@11:0:2;ptr@14:0:3
VLDB_3D_UNPACK_dmw_ldb_unpack_unpackSign0|ldb|602|0|dst@4:0:4;mod@11:0:2;ptr@14:0:3
VLDB_3D_UNPACK_dmw_ldb_unpack_unpackSign1|ldb|60a|0|dst@4:0:4;mod@11:0:2;ptr@14:0:3
VLDB_3D_UNPACK_dmx_ldb_unpack_unpackSign0|ldb|603|0|dst@5:0:3;mod@11:0:2;ptr@14:0:3
VLDB_3D_UNPACK_dmx_ldb_unpack_unpackSign1|ldb|60b|0|dst@5:0:3;mod@11:0:2;ptr@14:0:3
VLDB_3D_dmw_ldb|ldb|601|0|dst@3:0:5;mod@11:0:2;ptr@14:0:3
VLDB_3D_dmx_ldb_x|ldb|60d|0|dst@4:0:4;mod@11:0:2;ptr@14:0:3
VLDB_4x16_hi|ldb|904|0|dst@3:0:5;src@12:0:5
VLDB_4x16_lo|ldb|104|0|dst@3:0:5;src@12:0:5
VLDB_4x32_hi|ldb|b04|0|dst@3:0:5;src@12:0:5
VLDB_4x32_lo|ldb|304|0|dst@3:0:5;src@12:0:5
VLDB_4x64_hi|ldb|d04|0|dst@3:0:5;src@12:0:5
VLDB_4x64_lo|ldb|504|0|dst@3:0:5;src@12:0:5
VLDB_FILLX_512|ldb|8505|0|ptr@14:0:1
VLDB_FILL_512|ldb|10505|0|ptr@14:0:1
VLDB_POPX_512|ldb|10005|0|dst@4:0:4;ptr@14:0:1
VLDB_POP_512_2D|ldb|18405|0|dst@4:0:4;mod@11:0:3;ptr@14:0:1
VLDB_POP_512_3D|ldb|405|0|dst@4:0:4;mod@11:0:2;ptr@14:0:1
VLDB_POP_512_fifo_1d_pop|ldb|8205|0|dst@4:0:4;mod@11:0:3;ptr@14:0:1
VLDB_POP_512_normal_pop|ldb|5|0|dst@4:0:4;ptr@14:0:1
VLDB_POP_544_2D|ldb|10205|0|dst@4:0:4;mod@11:0:3;ptr@14:0:1
VLDB_POP_544_3D|ldb|8405|0|dst@4:0:4;mod@11:0:2;ptr@14:0:1
VLDB_POP_544_fifo_1d_pop|ldb|605|0|dst@4:0:4;mod@11:0:3;ptr@14:0:1
VLDB_POP_544_normal_pop|ldb|18005|0|dst@4:0:4;ptr@14:0:1
VLDB_POP_576_2D|ldb|205|0|dst@4:0:4;mod@11:0:3;ptr@14:0:1
VLDB_POP_576_3D|ldb|10405|0|dst@4:0:4;mod@11:0:2;ptr@14:0:1
VLDB_POP_576_fifo_1d_pop|ldb|18205|0|dst@4:0:4;mod@11:0:3;ptr@14:0:1
VLDB_POP_576_normal_pop|ldb|8005|0|dst@4:0:4;ptr@14:0:1
VLDB_POP_640_2D|ldb|10105|0|dst@6:0:2;mod@11:0:3;ptr@14:0:1
VLDB_POP_640_3D|ldb|18605|0|dst@6:0:2;mod@11:0:2;ptr@14:0:1
VLDB_POP_640_fifo_1d_pop|ldb|18105|0|dst@6:0:2;mod@11:0:3;ptr@14:0:1
VLDB_POP_640_normal_pop|ldb|10605|0|dst@6:0:2;ptr@14:0:1
VLDB_POP_704_2D|ldb|8105|0|dst@6:0:2;mod@11:0:3;ptr@14:0:1
VLDB_POP_704_3D|ldb|105|0|dst@6:0:2;mod@11:0:2;ptr@14:0:1
VLDB_POP_704_fifo_1d_pop|ldb|505|0|dst@6:0:2;mod@11:0:3;ptr@14:0:1
VLDB_POP_704_normal_pop|ldb|8605|0|dst@6:0:2;ptr@14:0:1
VLDB_UNPACK_dmw_ldb_unpack_idx_imm_unpackSign0|ldb|102|0|dst@4:0:4;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmw_ldb_unpack_idx_imm_unpackSign1|ldb|10a|0|dst@4:0:4;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmw_ldb_unpack_idx_unpackSign0|ldb|2|0|dj@11:0:3;dst@4:0:4;ptr@14:0:3
VLDB_UNPACK_dmw_ldb_unpack_idx_unpackSign1|ldb|a|0|dj@11:0:3;dst@4:0:4;ptr@14:0:3
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_imm_unpackSign0|ldb|302|0|dst@4:0:4;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_imm_unpackSign1|ldb|30a|0|dst@4:0:4;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_unpackSign0|ldb|202|0|dst@4:0:4;mod@11:0:3;ptr@14:0:3
VLDB_UNPACK_dmw_ldb_unpack_pstm_nrm_unpackSign1|ldb|20a|0|dst@4:0:4;mod@11:0:3;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_idx_imm_unpackSign0|ldb|103|0|dst@5:0:3;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_idx_imm_unpackSign1|ldb|10b|0|dst@5:0:3;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_idx_unpackSign0|ldb|3|0|dj@11:0:3;dst@5:0:3;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_idx_unpackSign1|ldb|b|0|dj@11:0:3;dst@5:0:3;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign0|ldb|303|0|dst@5:0:3;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1|ldb|30b|0|dst@5:0:3;imm@10:0:4;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_unpackSign0|ldb|203|0|dst@5:0:3;mod@11:0:3;ptr@14:0:3
VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_unpackSign1|ldb|20b|0|dst@5:0:3;mod@11:0:3;ptr@14:0:3
VLDB_dmw_ldb_idx|ldb|1|0|dj@11:0:3;dst@3:0:5;ptr@14:0:3
VLDB_dmw_ldb_idx_imm|ldb|101|0|dst@3:0:5;imm@10:0:4;ptr@14:0:3
VLDB_dmw_ldb_pstm_nrm|ldb|201|0|dst@3:0:5;mod@11:0:3;ptr@14:0:3
VLDB_dmw_ldb_pstm_nrm_imm|ldb|301|0|dst@3:0:5;imm@10:0:4;ptr@14:0:3
VLDB_dmx_ldb_x_idx|ldb|d|0|dj@11:0:3;dst@4:0:4;ptr@14:0:3
VLDB_dmx_ldb_x_idx_imm|ldb|10d|0|dst@4:0:4;imm@10:0:4;ptr@14:0:3
VLDB_dmx_ldb_x_pstm_nrm|ldb|20d|0|dst@4:0:4;mod@11:0:3;ptr@14:0:3
VLDB_dmx_ldb_x_pstm_nrm_imm|ldb|30d|0|dst@4:0:4;imm@10:0:4;ptr@14:0:3
VLT_16_vaddSign0|mv|53|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VLT_16_vaddSign1|mv|153|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VLT_32_vaddSign0|mv|93|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VLT_32_vaddSign1|mv|193|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VLT_8_vaddSign0|mv|13|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VLT_8_vaddSign1|mv|113|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VLT_bf16|mv|67|0|cmp@18:0:4;s1@14:0:4;s2@10:0:4
VMAC_f_vmac_bf_vmul_bf_core_X_X|vec|5|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMAC_f_vmac_bf_vmul_bf_core_Y_Y|vec|45|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@8:0:3
VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX|vec|25|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMAC_f_vmac_bfp_vmul_bfp_core_EX_EY|vec|15|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@8:0:3
VMAC_f_vmac_bfp_vmul_bfp_core_EX_QEY|vec|55|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@10:0:1
VMAC_f_vmac_bfp_vmul_bfp_core_EY_QEX|vec|65|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@9:0:2
VMAC_vmul_cm_core_X_QX|vec|44|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@9:0:2
VMAC_vmul_cm_core_X_QY|vec|a4|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@10:0:1
VMAC_vmul_cm_core_X_X|vec|4|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMAC_vmul_cm_core_X_Y|vec|24|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@8:0:3
VMAC_vmul_cm_core_Y_QX|vec|864|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@9:0:2
VMAC_vmul_cm_core_Y_QY|vec|d4|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@10:0:1
VMAC_vmul_cm_core_Y_X|vec|64|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@7:0:4
VMAC_vmul_cm_core_Y_Y|vec|54|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@8:0:3
VMAXDIFF_LT_16_vaddSign0|mv|102|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAXDIFF_LT_16_vaddSign1|mv|112|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAXDIFF_LT_32_vaddSign0|mv|2|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAXDIFF_LT_32_vaddSign1|mv|12|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAXDIFF_LT_8_vaddSign0|mv|202|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAXDIFF_LT_8_vaddSign1|mv|212|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAX_LT_16_vaddSign0|mv|182|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAX_LT_16_vaddSign1|mv|192|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAX_LT_32_vaddSign0|mv|82|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAX_LT_32_vaddSign1|mv|92|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAX_LT_8_vaddSign0|mv|282|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAX_LT_8_vaddSign1|mv|292|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMAX_LT_bf16|mv|367|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMIN_GE_16_vaddSign0|mv|162|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMIN_GE_16_vaddSign1|mv|172|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMIN_GE_32_vaddSign0|mv|62|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMIN_GE_32_vaddSign1|mv|72|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMIN_GE_8_vaddSign0|mv|262|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMIN_GE_8_vaddSign1|mv|272|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMIN_GE_bf16|mv|167|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VMOV_0_mv_scd_cm|lda|103b|0|dst@7:0:4
VMOV_0_mv_scd_dm_imm|lda|303b|0|dst@8:0:3
VMOV_1_mv_scd_cm|lda|107b|0|dst@7:0:4
VMOV_1_mv_scd_dm_imm|lda|307b|0|dst@8:0:3
VMOV_2|lda|30bb|0|dst@8:0:3
VMOV_3|lda|30fb|0|dst@8:0:3
VMOV_D|vec|74|0|acc1@15:0:3;dst@18:0:3
VMOV_alu_mv_mv_cm|mv|457|0|dst@18:0:4;src@12:0:4
VMOV_alu_mv_mv_ex|mv|2e7|0|dst@18:0:4;src@12:0:4
VMOV_alu_mv_mv_q|mv|105e7|0|dst@18:0:2;src@12:0:2
VMOV_alu_mv_mv_q_to_w|mv|7e7|0|dst@17:0:5;src@12:0:2
VMOV_alu_mv_mv_qex|mv|17|0|dst@18:0:2;src@12:0:2
VMOV_alu_mv_mv_qx|mv|217|0|dst@18:0:2;src@12:0:2
VMOV_alu_mv_mv_w|mv|117|0|dst@17:0:5;src@11:0:5
VMOV_alu_mv_mv_w_to_q|mv|10317|0|dst@18:0:2;src@11:0:5
VMOV_alu_mv_mv_x|mv|97|0|dst@16:0:6;src@10:0:6
VMOV_lda_mv_scd_bm|lda|3b|0|dst@6:0:5
VMOV_lda_mv_scd_dm_dyn|lda|283b|0|dst@8:0:3
VMOV_lda_mv_scd_dm_reg|lda|83b|0|dst@8:0:3
VMOV_lda_mv_scd_x|lda|203b|0|dst@7:0:4
VMOV_st_mv_mcd_bm|st|18004|0|src@6:0:5
VMOV_st_mv_mcd_x|st|1c004|0|src@7:0:4
VMSC_f_vmac_bf_vmul_bf_core_X_X|vec|d|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMSC_f_vmac_bf_vmul_bf_core_Y_Y|vec|4d|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@8:0:3
VMSC_f_vmac_bfp_vmul_bfp_core_EX_EX|vec|2d|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMSC_f_vmac_bfp_vmul_bfp_core_EX_EY|vec|1d|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@8:0:3
VMSC_f_vmac_bfp_vmul_bfp_core_EX_QEY|vec|5d|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@10:0:1
VMSC_f_vmac_bfp_vmul_bfp_core_EY_QEX|vec|6d|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@9:0:2
VMSC_vmul_cm_core_X_QX|vec|4c|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@9:0:2
VMSC_vmul_cm_core_X_QY|vec|ac|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@10:0:1
VMSC_vmul_cm_core_X_X|vec|c|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMSC_vmul_cm_core_X_Y|vec|2c|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@11:0:4;s2@8:0:3
VMSC_vmul_cm_core_Y_QX|vec|86c|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@9:0:2
VMSC_vmul_cm_core_Y_QY|vec|dc|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@10:0:1
VMSC_vmul_cm_core_Y_X|vec|6c|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@7:0:4
VMSC_vmul_cm_core_Y_Y|vec|5c|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3;s1@12:0:3;s2@8:0:3
VMUL_f_vmul_bf_vmul_bf_core_X_X|vec|38005|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMUL_f_vmul_bf_vmul_bf_core_Y_Y|vec|38045|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@8:0:3
VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX|vec|38025|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMUL_f_vmul_bfp_vmul_bfp_core_EX_EY|vec|38015|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@8:0:3
VMUL_f_vmul_bfp_vmul_bfp_core_EX_QEY|vec|38055|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@10:0:1
VMUL_f_vmul_bfp_vmul_bfp_core_EY_QEX|vec|38065|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@9:0:2
VMUL_vmul_cm_core_X_QX|vec|38044|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@9:0:2
VMUL_vmul_cm_core_X_QY|vec|380a4|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@10:0:1
VMUL_vmul_cm_core_X_X|vec|38004|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@7:0:4
VMUL_vmul_cm_core_X_Y|vec|38024|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@8:0:3
VMUL_vmul_cm_core_Y_QX|vec|38864|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@9:0:2
VMUL_vmul_cm_core_Y_QY|vec|380d4|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@10:0:1
VMUL_vmul_cm_core_Y_X|vec|38064|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@7:0:4
VMUL_vmul_cm_core_Y_Y|vec|38054|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@8:0:3
VNEG|vec|7c|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VNEGMUL_f_vmul_bf_vmul_bf_core_X_X|vec|3800d|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@7:0:4
VNEGMUL_f_vmul_bf_vmul_bf_core_Y_Y|vec|3804d|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@8:0:3
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_EX|vec|3802d|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@7:0:4
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_EY|vec|3801d|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@8:0:3
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EX_QEY|vec|3805d|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@10:0:1
VNEGMUL_f_vmul_bfp_vmul_bfp_core_EY_QEX|vec|3806d|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@9:0:2
VNEGMUL_vmul_cm_core_X_QX|vec|3804c|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@9:0:2
VNEGMUL_vmul_cm_core_X_QY|vec|380ac|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@10:0:1
VNEGMUL_vmul_cm_core_X_X|vec|3800c|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@7:0:4
VNEGMUL_vmul_cm_core_X_Y|vec|3802c|0|acc@21:0:5;dst@18:0:3;s1@11:0:4;s2@8:0:3
VNEGMUL_vmul_cm_core_Y_QX|vec|3886c|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@9:0:2
VNEGMUL_vmul_cm_core_Y_QY|vec|380dc|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@10:0:1
VNEGMUL_vmul_cm_core_Y_X|vec|3806c|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@7:0:4
VNEGMUL_vmul_cm_core_Y_Y|vec|3805c|0|acc@21:0:5;dst@18:0:3;s1@12:0:3;s2@8:0:3
VNEG_GTZ_16|mv|1857|0|d@18:0:4;s2@14:0:4
VNEG_GTZ_32|mv|857|0|d@18:0:4;s2@14:0:4
VNEG_GTZ_8|mv|2857|0|d@18:0:4;s2@14:0:4
VNEG_f|vec|7d|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VPACK_mv_pack_w_packSign0|st|302c|0|dst@15:0:5;src@7:0:4
VPACK_mv_pack_w_packSign1|st|306c|0|dst@15:0:5;src@7:0:4
VPACK_mv_pack_x_packSign0|st|82c|0|dst@16:0:4;src@8:0:3
VPACK_mv_pack_x_packSign1|st|86c|0|dst@16:0:4;src@8:0:3
VPUSH_hi_16|mv|bb|0|d@18:0:4;s1@14:0:4;s2@9:0:5
VPUSH_hi_32|mv|13b|0|d@18:0:4;s1@14:0:4;s2@9:0:5
VPUSH_hi_64|mv|1bb|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VPUSH_hi_8|mv|3b|0|d@18:0:4;s1@14:0:4;s2@9:0:5
VPUSH_lo_16|mv|fb|0|d@18:0:4;s1@13:0:5;s2@9:0:4
VPUSH_lo_32|mv|17b|0|d@18:0:4;s1@13:0:5;s2@9:0:4
VPUSH_lo_64|mv|1fb|0|d@18:0:4;s1@14:0:4;s2@9:0:4
VPUSH_lo_8|mv|7b|0|d@18:0:4;s1@13:0:5;s2@9:0:4
VSEL_16|mv|11|0|d@18:0:4;s1@14:0:4;s2@10:0:4;sel@6:0:4
VSEL_32|mv|1|0|d@18:0:4;s1@14:0:4;s2@10:0:4;sel@6:0:4
VSEL_8|mv|21|0|d@18:0:4;s1@14:0:4;s2@10:0:4;sel@6:0:4
VSHIFT|mv|16|0|d@18:0:4;s1@14:0:4;s2@10:0:4;shift@5:0:5
VSHIFT_ALIGN|mv|a7|0|d@18:0:4;pre@8:0:2;s1@14:0:4;s2@10:0:4
VSHUFFLE_vec_shuffle_bm|mv|e|0|dst@18:0:4;mod@5:0:5;s1@14:0:4;s2@10:0:4
VSHUFFLE_vec_shuffle_ex|mv|1e|0|dst@18:0:4;mod@5:0:5;s1@14:0:4;s2@10:0:4
VSHUFFLE_vec_shuffle_x|mv|3|0|dst@18:0:4;mod@5:0:5;s1@14:0:4;s2@10:0:4
VSRS_2x_mv_w_srs_bm_srsSign0|st|c|0|dst@15:0:5;src@6:0:5;su@12:0:2
VSRS_2x_mv_w_srs_bm_srsSign1|st|80c|0|dst@15:0:5;src@6:0:5;su@12:0:2
VSRS_2x_mv_x_srs_cm_srsSign0|st|404c|0|dst@16:0:4;src@7:0:4;su@12:0:2
VSRS_2x_mv_x_srs_cm_srsSign1|st|484c|0|dst@16:0:4;src@7:0:4;su@12:0:2
VSRS_4x_mv_w_srs_cm_srsSign0|st|400c|0|dst@15:0:5;src@7:0:4;su@12:0:2
VSRS_4x_mv_w_srs_cm_srsSign1|st|480c|0|dst@15:0:5;src@7:0:4;su@12:0:2
VSRS_4x_mv_x_srs_dm_srsSign0|st|c04c|0|dst@16:0:4;src@8:0:3;su@12:0:2
VSRS_4x_mv_x_srs_dm_srsSign1|st|c84c|0|dst@16:0:4;src@8:0:3;su@12:0:2
VST_128_dmv_sts_w_idx|st|1c|0|dj@14:0:3;ptr@17:0:3;src@6:0:5
VST_128_dmv_sts_w_idx_imm|st|81c|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_128_dmv_sts_w_pstm_nrm|st|101c|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_128_dmv_sts_w_pstm_nrm_imm|st|181c|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_128_dmv_sts_w_spill|st|3c|0|imm@11:0:9;src@6:0:5
VST_2D_128|st|201c|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_2D_CONV_bf16_fp32_dmw_sts_srs_bf|st|2025|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_2D_CONV_bf16_fp32_dmx_sts_srs_bf|st|2046|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_2D_PACK_dmw_sts_pack_packSign0|st|2005|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_2D_PACK_dmw_sts_pack_packSign1|st|2045|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_2D_PACK_dmx_sts_pack_packSign0|st|201d|0|mod@14:0:3;ptr@17:0:3;src@8:0:3
VST_2D_PACK_dmx_sts_pack_packSign1|st|205d|0|mod@14:0:3;ptr@17:0:3;src@8:0:3
VST_2D_SRS_2x_dm_sts_srs_cm_srsSign0|st|2040|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_2D_SRS_2x_dm_sts_srs_cm_srsSign1|st|2048|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_2D_SRS_2x_dmw_sts_srs_bm_srsSign0|st|2001|0|mod@14:0:3;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_2D_SRS_2x_dmw_sts_srs_bm_srsSign1|st|2009|0|mod@14:0:3;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_2D_SRS_4x_dm_sts_srs_cm_srsSign0|st|2000|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_2D_SRS_4x_dm_sts_srs_cm_srsSign1|st|2008|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_2D_SRS_4x_dmx_sts_srs_dm_srsSign0|st|2002|0|mod@14:0:3;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_2D_SRS_4x_dmx_sts_srs_dm_srsSign1|st|200a|0|mod@14:0:3;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_2D_dmw_sts_w|st|2015|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_2D_dmx_sts_bm|st|200d|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_2D_dmx_sts_fifohl|st|2056|0|mod@14:0:3;ptr@17:0:3;src@8:0:3
VST_2D_dmx_sts_x|st|2026|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_3D_128|st|301c|0|mod@14:0:2;ptr@17:0:3;src@6:0:5
VST_3D_CONV_bf16_fp32_dmw_sts_srs_bf|st|3025|0|mod@14:0:2;ptr@17:0:3;src@6:0:5
VST_3D_CONV_bf16_fp32_dmx_sts_srs_bf|st|3046|0|mod@14:0:2;ptr@17:0:3;src@7:0:4
VST_3D_PACK_dmw_sts_pack_packSign0|st|3005|0|mod@14:0:2;ptr@17:0:3;src@7:0:4
VST_3D_PACK_dmw_sts_pack_packSign1|st|3045|0|mod@14:0:2;ptr@17:0:3;src@7:0:4
VST_3D_PACK_dmx_sts_pack_packSign0|st|301d|0|mod@14:0:2;ptr@17:0:3;src@8:0:3
VST_3D_PACK_dmx_sts_pack_packSign1|st|305d|0|mod@14:0:2;ptr@17:0:3;src@8:0:3
VST_3D_SRS_2x_dm_sts_srs_cm_srsSign0|st|3040|0|mod@14:0:2;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_3D_SRS_2x_dm_sts_srs_cm_srsSign1|st|3048|0|mod@14:0:2;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_3D_SRS_2x_dmw_sts_srs_bm_srsSign0|st|3001|0|mod@14:0:2;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_3D_SRS_2x_dmw_sts_srs_bm_srsSign1|st|3009|0|mod@14:0:2;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_3D_SRS_4x_dm_sts_srs_cm_srsSign0|st|3000|0|mod@14:0:2;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_3D_SRS_4x_dm_sts_srs_cm_srsSign1|st|3008|0|mod@14:0:2;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_3D_SRS_4x_dmx_sts_srs_dm_srsSign0|st|3002|0|mod@14:0:2;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_3D_SRS_4x_dmx_sts_srs_dm_srsSign1|st|300a|0|mod@14:0:2;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_3D_dmw_sts_w|st|3015|0|mod@14:0:2;ptr@17:0:3;src@6:0:5
VST_3D_dmx_sts_bm|st|300d|0|mod@14:0:2;ptr@17:0:3;src@6:0:5
VST_3D_dmx_sts_fifohl|st|3056|0|mod@14:0:2;ptr@17:0:3;src@8:0:3
VST_3D_dmx_sts_x|st|3026|0|mod@14:0:2;ptr@17:0:3;src@7:0:4
VST_CONV_bf16_fp32_dmw_sts_srs_bf_idx|st|25|0|dj@14:0:3;ptr@17:0:3;src@6:0:5
VST_CONV_bf16_fp32_dmw_sts_srs_bf_idx_imm|st|825|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_CONV_bf16_fp32_dmw_sts_srs_bf_pstm_nrm|st|1025|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_CONV_bf16_fp32_dmw_sts_srs_bf_pstm_nrm_imm|st|1825|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_CONV_bf16_fp32_dmx_sts_srs_bf_idx|st|46|0|dj@14:0:3;ptr@17:0:3;src@7:0:4
VST_CONV_bf16_fp32_dmx_sts_srs_bf_idx_imm|st|846|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm|st|1046|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_CONV_bf16_fp32_dmx_sts_srs_bf_pstm_nrm_imm|st|1846|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_FLUSH_512_2D|st|40006|0|mod@14:0:3
VST_FLUSH_512_3D|st|80006|0|mod@14:0:2
VST_FLUSH_512_CONV_2D|st|60006|0|mod@14:0:3
VST_FLUSH_512_CONV_3D|st|a0006|0|mod@14:0:2
VST_FLUSH_512_CONV_fifo_1d_flush|st|e0006|0|mod@14:0:3
VST_FLUSH_512_CONV_normal_flush|st|20006|0|-
VST_FLUSH_512_fifo_1d_flush|st|c0006|0|mod@14:0:3
VST_FLUSH_512_normal_flush|st|6|0|-
VST_PACK_dmw_sts_pack_idx_imm_packSign0|st|805|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_PACK_dmw_sts_pack_idx_imm_packSign1|st|845|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_PACK_dmw_sts_pack_idx_packSign0|st|5|0|dj@14:0:3;ptr@17:0:3;src@7:0:4
VST_PACK_dmw_sts_pack_idx_packSign1|st|45|0|dj@14:0:3;ptr@17:0:3;src@7:0:4
VST_PACK_dmw_sts_pack_pstm_nrm_imm_packSign0|st|1805|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_PACK_dmw_sts_pack_pstm_nrm_imm_packSign1|st|1845|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_PACK_dmw_sts_pack_pstm_nrm_packSign0|st|1005|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_PACK_dmw_sts_pack_pstm_nrm_packSign1|st|1045|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_PACK_dmx_sts_pack_idx_imm_packSign0|st|81d|0|imm@13:0:4;ptr@17:0:3;src@8:0:3
VST_PACK_dmx_sts_pack_idx_imm_packSign1|st|85d|0|imm@13:0:4;ptr@17:0:3;src@8:0:3
VST_PACK_dmx_sts_pack_idx_packSign0|st|1d|0|dj@14:0:3;ptr@17:0:3;src@8:0:3
VST_PACK_dmx_sts_pack_idx_packSign1|st|5d|0|dj@14:0:3;ptr@17:0:3;src@8:0:3
VST_PACK_dmx_sts_pack_pstm_nrm_imm_packSign0|st|181d|0|imm@13:0:4;ptr@17:0:3;src@8:0:3
VST_PACK_dmx_sts_pack_pstm_nrm_imm_packSign1|st|185d|0|imm@13:0:4;ptr@17:0:3;src@8:0:3
VST_PACK_dmx_sts_pack_pstm_nrm_packSign0|st|101d|0|mod@14:0:3;ptr@17:0:3;src@8:0:3
VST_PACK_dmx_sts_pack_pstm_nrm_packSign1|st|105d|0|mod@14:0:3;ptr@17:0:3;src@8:0:3
VST_PUSH_512|st|90006|0|src@7:0:4
VST_PUSH_544|st|94006|0|src@7:0:4
VST_PUSH_544_CONV_bfp16ebs16_ebs8|st|b4006|0|src@7:0:4
VST_PUSH_544_CONV_bfp16ebs16_fp32|st|b8006|0|src@8:0:3
VST_PUSH_576|st|98006|0|src@7:0:4
VST_PUSH_576_CONV_bfp16ebs8_fp32|st|b0006|0|src@8:0:3
VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign0|st|840|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign1|st|848|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dm_sts_srs_cm_idx_srsSign0|st|40|0|dj@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dm_sts_srs_cm_idx_srsSign1|st|48|0|dj@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign0|st|1840|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1|st|1848|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_srsSign0|st|1040|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_srsSign1|st|1048|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_idx_imm_srsSign0|st|801|0|imm@13:0:4;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_idx_imm_srsSign1|st|809|0|imm@13:0:4;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_idx_srsSign0|st|1|0|dj@14:0:3;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_idx_srsSign1|st|9|0|dj@14:0:3;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_imm_srsSign0|st|1801|0|imm@13:0:4;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_imm_srsSign1|st|1809|0|imm@13:0:4;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_srsSign0|st|1001|0|mod@14:0:3;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_2x_dmw_sts_srs_bm_pstm_nrm_srsSign1|st|1009|0|mod@14:0:3;ptr@17:0:3;src@6:0:5;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_idx_imm_srsSign0|st|800|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_idx_imm_srsSign1|st|808|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_idx_srsSign0|st|0|0|dj@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_idx_srsSign1|st|8|0|dj@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_imm_srsSign0|st|1800|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1|st|1808|0|imm@13:0:4;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_srsSign0|st|1000|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dm_sts_srs_cm_pstm_nrm_srsSign1|st|1008|0|mod@14:0:3;ptr@17:0:3;src@7:0:4;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_idx_imm_srsSign0|st|802|0|imm@13:0:4;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_idx_imm_srsSign1|st|80a|0|imm@13:0:4;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_idx_srsSign0|st|2|0|dj@14:0:3;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_idx_srsSign1|st|a|0|dj@14:0:3;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign0|st|1802|0|imm@13:0:4;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1|st|180a|0|imm@13:0:4;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_srsSign0|st|1002|0|mod@14:0:3;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_srsSign1|st|100a|0|mod@14:0:3;ptr@17:0:3;src@8:0:3;su@4:0:2
VST_dmw_sts_w_idx|st|15|0|dj@14:0:3;ptr@17:0:3;src@6:0:5
VST_dmw_sts_w_idx_imm|st|815|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_dmw_sts_w_pstm_nrm|st|1015|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_dmw_sts_w_pstm_nrm_imm|st|1815|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_dmw_sts_w_spill|st|35|0|imm@11:0:9;src@6:0:5
VST_dmx_sts_bm_idx|st|d|0|dj@14:0:3;ptr@17:0:3;src@6:0:5
VST_dmx_sts_bm_idx_imm|st|80d|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_dmx_sts_bm_pstm_nrm|st|100d|0|mod@14:0:3;ptr@17:0:3;src@6:0:5
VST_dmx_sts_bm_pstm_nrm_imm|st|180d|0|imm@13:0:4;ptr@17:0:3;src@6:0:5
VST_dmx_sts_bm_spill|st|2d|0|imm@11:0:9;src@6:0:5
VST_dmx_sts_fifohl_idx|st|56|0|dj@14:0:3;ptr@17:0:3;src@8:0:3
VST_dmx_sts_fifohl_idx_imm|st|856|0|imm@13:0:4;ptr@17:0:3;src@8:0:3
VST_dmx_sts_fifohl_pstm_nrm|st|1056|0|mod@14:0:3;ptr@17:0:3;src@8:0:3
VST_dmx_sts_fifohl_pstm_nrm_imm|st|1856|0|imm@13:0:4;ptr@17:0:3;src@8:0:3
VST_dmx_sts_fifohl_spill|st|d6|0|imm@11:0:9;src@8:0:3
VST_dmx_sts_x_idx|st|26|0|dj@14:0:3;ptr@17:0:3;src@7:0:4
VST_dmx_sts_x_idx_imm|st|826|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_dmx_sts_x_pstm_nrm|st|1026|0|mod@14:0:3;ptr@17:0:3;src@7:0:4
VST_dmx_sts_x_pstm_nrm_imm|st|1826|0|imm@13:0:4;ptr@17:0:3;src@7:0:4
VST_dmx_sts_x_spill|st|66|0|imm@11:0:9;src@7:0:4
VSUB_16|mv|2eb|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_32|mv|36b|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_8|mv|26b|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_GE_16_vaddSign0|mv|142|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_GE_16_vaddSign1|mv|152|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_GE_32_vaddSign0|mv|42|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_GE_32_vaddSign1|mv|52|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_GE_8_vaddSign0|mv|242|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_GE_8_vaddSign1|mv|252|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_LT_16_vaddSign0|mv|122|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_LT_16_vaddSign1|mv|132|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_LT_32_vaddSign0|mv|22|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_LT_32_vaddSign1|mv|32|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_LT_8_vaddSign0|mv|222|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_LT_8_vaddSign1|mv|232|0|d@18:0:4;s1@14:0:4;s2@10:0:4
VSUB_f_vmac_cm2_add_reg|vec|fd|0|acc@21:0:5;acc1@15:0:3;acc2@12:0:3;dst@18:0:3
VSUB_f_vmac_cm2_add_scd|vec|60fd|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VSUB_f_vmac_cm2_add_scd_incr|vec|70fd|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VSUB_vmac_cm2_add_reg|vec|fc|0|acc@21:0:5;acc1@15:0:3;acc2@12:0:3;dst@18:0:3
VSUB_vmac_cm2_add_scd|vec|60fc|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VSUB_vmac_cm2_add_scd_incr|vec|70fc|0|acc@21:0:5;acc1@15:0:3;dst@18:0:3
VTANH|mv|f97|0|dst@17:0:5;src@12:0:5
VUNPACK_mv_unpack_w_unpackSign0|ldb|204|0|dst@4:0:4;src@12:0:5
VUNPACK_mv_unpack_w_unpackSign1|ldb|20c|0|dst@4:0:4;src@12:0:5
VUNPACK_mv_unpack_x_unpackSign0|ldb|604|0|dst@5:0:3;src@13:0:4
VUNPACK_mv_unpack_x_unpackSign1|ldb|60c|0|dst@5:0:3;src@13:0:4
VUPS_2x_mv_ups_w2b_upsSign0|mv|87|0|dst@17:0:5;src@11:0:5;su@9:0:2
VUPS_2x_mv_ups_w2b_upsSign1|mv|187|0|dst@17:0:5;src@11:0:5;su@9:0:2
VUPS_2x_mv_ups_x2c_upsSign0|mv|c7|0|dst@18:0:4;src@12:0:4;su@9:0:2
VUPS_2x_mv_ups_x2c_upsSign1|mv|1c7|0|dst@18:0:4;src@12:0:4;su@9:0:2
VUPS_4x_mv_ups_w2c_upsSign0|mv|47|0|dst@18:0:4;src@11:0:5;su@9:0:2
VUPS_4x_mv_ups_w2c_upsSign1|mv|147|0|dst@18:0:4;src@11:0:5;su@9:0:2
VUPS_4x_mv_ups_x2d_upsSign0|mv|27|0|dst@19:0:3;src@12:0:4;su@9:0:2
VUPS_4x_mv_ups_x2d_upsSign1|mv|127|0|dst@19:0:3;src@12:0:4;su@9:0:2
XOR|alu|d|0|d0@10:0:5;s0@15:0:5;s1@5:0:5
"""

# name|bit count|fixed value|slot@target:value:count,...;...
_BUNDLE_ENCODING_RECORDS = """\
I112_LDA_LDB_ALU_MV_ST|112|10000007e|alu@55:0:20;lda@92:0:20;ldb@75:0:17;mv@33:0:22;st@12:0:20
I112_LDA_LDB_ALU_MV_VEC|112|10000000e|alu@55:0:20;lda@92:0:20;ldb@75:0:17;mv@33:0:22;vec@6:0:26
I112_LDA_LDB_ALU_ST_VEC|112|2e|alu@55:0:20;lda@92:0:20;ldb@75:0:17;st@33:0:20;vec@6:0:26
I112_LDA_LDB_LNG_ST|112|7e|lda@92:0:20;ldb@75:0:17;lng@33:0:42;st@12:0:20
I112_LDA_LDB_LNG_VEC|112|e|lda@92:0:20;ldb@75:0:17;lng@33:0:42;vec@6:0:26
I112_LDA_ST_MV_VEC|112|4000000000002e|lda@92:0:20;mv@32:0:22;st@72:0:20;vec@6:0:26
I112_ST_LDB_ALU_MV_VEC|112|10000001e|alu@55:0:20;ldb@75:0:17;mv@33:0:22;st@92:0:20;vec@6:0:26
I112_ST_LDB_LNG_VEC|112|1e|ldb@75:0:17;lng@33:0:42;st@92:0:20;vec@6:0:26
I128_LDA_LDB_ST_ALU_MV_VEC|128|8000001|alu@50:0:20;lda@108:0:20;ldb@91:0:17;mv@28:0:22;st@71:0:20;vec@1:0:26
I128_LDA_LDB_ST_LNG_VEC|128|1|lda@108:0:20;ldb@91:0:17;lng@28:0:42;st@71:0:20;vec@1:0:26
I16_NOP|16|0|nop@15:0:1
I32_ALU|32|10000018|alu@7:0:20
I32_LDA|32|18|lda@7:0:20
I32_LDB|32|38000018|ldb@10:0:17
I32_MV|32|18000018|mv@5:0:22
I32_ST|32|8000018|st@7:0:20
I32_VEC|32|8|vec@6:0:26
I48_ALU_MV|48|24|alu@28:0:20;mv@6:0:22
I48_LDA_ALU|48|2c|alu@8:0:20;lda@28:0:20
I48_LDA_LDB|48|3c|lda@28:0:20;ldb@11:0:17
I48_LDA_MV|48|14|lda@28:0:20;mv@6:0:22
I48_LDA_ST|48|c|lda@28:0:20;st@8:0:20
I48_LDB_ALU|48|1c|alu@8:0:20;ldb@31:0:17
I48_LDB_MV|48|34|ldb@31:0:17;mv@6:0:22
I48_LDB_ST|48|4c|ldb@31:0:17;st@8:0:20
I48_LNG|48|4|lng@6:0:42
I48_ST_ALU|48|5c|alu@8:0:20;st@28:0:20
I64_ALU_VEC|64|100000022|alu@34:0:20;vec@6:0:26
I64_LDA_LDB_ALU|64|12|alu@7:0:20;lda@44:0:20;ldb@27:0:17
I64_LDA_LDB_ST|64|32|lda@44:0:20;ldb@27:0:17;st@7:0:20
I64_LDA_VEC|64|400000022|lda@44:0:20;vec@6:0:26
I64_LDB_VEC|64|22|ldb@42:0:17;vec@6:0:26
I64_MV_VEC|64|600000022|mv@37:0:22;vec@6:0:26
I64_ST_LDB_ALU|64|52|alu@7:0:20;ldb@27:0:17;st@44:0:20
I64_ST_MV|64|2|mv@12:0:22;st@44:0:20
I64_ST_VEC|64|200000022|st@44:0:20;vec@6:0:26
I80_ALU_MV_VEC|80|10000001a|alu@55:0:20;mv@33:0:22;vec@6:0:26
I80_LDA_ALU_MV|80|8ba|alu@34:0:20;lda@60:0:20;mv@12:0:22
I80_LDA_ALU_VEC|80|2a|alu@34:0:20;lda@60:0:20;vec@6:0:26
I80_LDA_LDB_MV|80|eba|lda@60:0:20;ldb@42:0:17;mv@12:0:22
I80_LDA_LDB_VEC|80|1d0000000a|lda@60:0:20;ldb@42:0:17;vec@6:0:26
I80_LDA_LNG|80|ba|lda@60:0:20;lng@12:0:42
I80_LDA_MV_VEC|80|20000000a|lda@60:0:20;mv@37:0:22;vec@6:0:26
I80_LDA_ST_ALU|80|7a|alu@7:0:20;lda@60:0:20;st@39:0:20
I80_LDA_ST_MV|80|2ba|lda@60:0:20;mv@12:0:22;st@39:0:20
I80_LDA_ST_VEC|80|90000000a|lda@60:0:20;st@39:0:20;vec@6:0:26
I80_LDB_ALU_MV|80|83a|alu@34:0:20;ldb@59:0:17;mv@12:0:22
I80_LDB_ALU_VEC|80|20000002a|alu@34:0:20;ldb@59:0:17;vec@6:0:26
I80_LDB_LNG|80|3a|ldb@59:0:17;lng@12:0:42
I80_LDB_MV_VEC|80|60000000a|ldb@59:0:17;mv@37:0:22;vec@6:0:26
I80_LNG_VEC|80|1a|lng@33:0:42;vec@6:0:26
I80_ST_ALU_MV|80|93a|alu@34:0:20;mv@12:0:22;st@60:0:20
I80_ST_ALU_VEC|80|10000002a|alu@34:0:20;st@60:0:20;vec@6:0:26
I80_ST_LDB_MV|80|6ba|ldb@42:0:17;mv@12:0:22;st@60:0:20
I80_ST_LDB_VEC|80|10000000a|ldb@42:0:17;st@60:0:20;vec@6:0:26
I80_ST_LNG|80|13a|lng@12:0:42;st@60:0:20
I80_ST_MV_VEC|80|40000000a|mv@37:0:22;st@60:0:20;vec@6:0:26
I96_LDA_ALU_MV_VEC|96|100000006|alu@55:0:20;lda@76:0:20;mv@33:0:22;vec@6:0:26
I96_LDA_LDB_ALU_MV|96|8b6|alu@34:0:20;lda@76:0:20;ldb@59:0:17;mv@12:0:22
I96_LDA_LDB_ALU_ST|96|136|alu@34:0:20;lda@76:0:20;ldb@59:0:17;st@14:0:20
I96_LDA_LDB_ALU_VEC|96|300000026|alu@34:0:20;lda@76:0:20;ldb@59:0:17;vec@6:0:26
I96_LDA_LDB_LNG|96|b6|lda@76:0:20;ldb@59:0:17;lng@12:0:42
I96_LDA_LDB_MV_VEC|96|200000026|lda@76:0:20;ldb@59:0:17;mv@37:0:22;vec@6:0:26
I96_LDA_LDB_ST_MV|96|f6|lda@76:0:20;ldb@59:0:17;mv@12:0:22;st@39:0:20
I96_LDA_LDB_ST_VEC|96|e00000026|lda@76:0:20;ldb@59:0:17;st@38:0:20;vec@6:0:26
I96_LDA_LNG_VEC|96|6|lda@76:0:20;lng@33:0:42;vec@6:0:26
I96_LDA_ST_ALU_MV|96|876|alu@34:0:20;lda@76:0:20;mv@12:0:22;st@55:0:20
I96_LDA_ST_ALU_VEC|96|26|alu@34:0:20;lda@76:0:20;st@55:0:20;vec@6:0:26
I96_LDA_ST_LNG|96|76|lda@76:0:20;lng@12:0:42;st@55:0:20
I96_LDB_ALU_MV_VEC|96|1a00000026|alu@59:0:20;ldb@79:0:17;mv@37:0:22;vec@6:0:26
I96_LDB_LNG_VEC|96|a00000026|ldb@79:0:17;lng@37:0:42;vec@6:0:26
I96_ST_ALU_MV_VEC|96|100000016|alu@55:0:20;mv@33:0:22;st@76:0:20;vec@6:0:26
I96_ST_LDB_ALU_MV|96|836|alu@34:0:20;ldb@59:0:17;mv@12:0:22;st@76:0:20
I96_ST_LDB_ALU_VEC|96|100000026|alu@34:0:20;ldb@59:0:17;st@76:0:20;vec@6:0:26
I96_ST_LDB_LNG|96|36|ldb@59:0:17;lng@12:0:42;st@76:0:20
I96_ST_LDB_MV_VEC|96|600000026|ldb@59:0:17;mv@37:0:22;st@76:0:20;vec@6:0:26
I96_ST_LNG_VEC|96|16|lng@33:0:42;st@76:0:20;vec@6:0:26
"""


def _parse_mappings(encoded_mappings: str) -> tuple[BitMapping, ...]:
    mappings: tuple[BitMapping, ...] = ()
    for encoded_segment in encoded_mappings.split(","):
        target_bit, value_bit, count = (
            int(value) for value in encoded_segment.split(":")
        )
        mappings += bit_range(target_bit, value_bit, count)
    return mappings


def _parse_instruction_fields(
    encoded_fields: str,
) -> tuple[InstructionFieldEncoding, ...]:
    if encoded_fields == "-":
        return ()
    return tuple(
        InstructionFieldEncoding(name, _parse_mappings(encoded_mappings))
        for encoded_field in encoded_fields.split(";")
        for name, encoded_mappings in (encoded_field.split("@", 1),)
    )


def _parse_bundle_fields(encoded_fields: str) -> tuple[BundleFieldEncoding, ...]:
    if encoded_fields == "-":
        return ()
    return tuple(
        BundleFieldEncoding(slot, _parse_mappings(encoded_mappings))
        for encoded_field in encoded_fields.split(";")
        for slot, encoded_mappings in (encoded_field.split("@", 1),)
    )


def _derive_fixed_mask(
    bit_count: int,
    fields: Sequence[InstructionFieldEncoding | BundleFieldEncoding],
) -> int:
    variable_mask = 0
    for field in fields:
        for mapping in field.mappings:
            variable_mask |= 1 << mapping.target_bit
    return ((1 << bit_count) - 1) & ~variable_mask


def _parse_instruction_encodings() -> tuple[InstructionEncoding, ...]:
    instructions = []
    for encoded_record in _INSTRUCTION_ENCODING_RECORDS.splitlines():
        name, slot, fixed_value, delay_slot_count, encoded_fields = (
            encoded_record.split("|", 4)
        )
        bit_count = SLOT_BIT_COUNTS[slot]
        fields = _parse_instruction_fields(encoded_fields)
        instructions.append(
            InstructionEncoding(
                name=name,
                slot=slot,
                bit_count=bit_count,
                fixed_mask=_derive_fixed_mask(bit_count, fields),
                fixed_value=int(fixed_value, 16),
                fields=fields,
                delay_slot_count=int(delay_slot_count),
            )
        )
    return tuple(instructions)


def _parse_bundle_encodings() -> tuple[BundleFormatEncoding, ...]:
    bundle_formats = []
    for encoded_record in _BUNDLE_ENCODING_RECORDS.splitlines():
        name, encoded_bit_count, fixed_value, encoded_fields = encoded_record.split(
            "|", 3
        )
        bit_count = int(encoded_bit_count)
        fields = _parse_bundle_fields(encoded_fields)
        bundle_formats.append(
            BundleFormatEncoding(
                name=name,
                bit_count=bit_count,
                fixed_mask=_derive_fixed_mask(bit_count, fields),
                fixed_value=int(fixed_value, 16),
                fields=fields,
            )
        )
    return tuple(bundle_formats)


CORE_ENCODING_TABLE = EncodingTable(
    instructions=_parse_instruction_encodings(),
    bundle_formats=_parse_bundle_encodings(),
)


_NOP = InstructionInstance("NOP")
_RET = InstructionInstance("RET")
_VLDA = InstructionInstance(
    "VLDA_dmx_lda_x_idx_imm",
    (
        ("dst", 0),
        ("imm", 0),
        ("ptr", 0),
    ),
)
_VLDB = InstructionInstance(
    "VLDB_dmx_ldb_x_idx_imm",
    (
        ("dst", 2),
        ("imm", 0),
        ("ptr", 1),
    ),
)
_VST = InstructionInstance(
    "VST_dmx_sts_x_idx_imm",
    (
        ("imm", 0),
        ("ptr", 2),
        ("src", 0),
    ),
)


def _vector_add_witness(
    symbol: str,
    instruction_name: str,
    expected_hex: str,
) -> EncodingWitness:
    vector_add = InstructionInstance(
        instruction_name,
        (
            ("d", 0),
            ("s1", 2),
            ("s2", 0),
        ),
    )
    nop_bundle = BundleInstance("I16_NOP", (_NOP,))
    return EncodingWitness(
        symbol=symbol,
        expected_bytes=bytes.fromhex(expected_hex),
        bundles=(
            BundleInstance("I48_LDA_LDB", (_VLDA, _VLDB)),
            nop_bundle,
            nop_bundle,
            nop_bundle,
            nop_bundle,
            BundleInstance("I32_ALU", (_RET,)),
            nop_bundle,
            BundleInstance("I32_MV", (vector_add,)),
            nop_bundle,
            BundleInstance("I32_ST", (_VST,)),
            nop_bundle,
        ),
    )


CORE_ENCODING_WITNESSES = (
    _vector_add_witness(
        "vector_add_i32x16",
        "VADD_32",
        "3c68097283000000000000000000180028100000782d101800001813040a0000",
    ),
    _vector_add_witness(
        "vector_add_i16x32",
        "VADD_16",
        "3c68097283000000000000000000180028100000781d101800001813040a0000",
    ),
    _vector_add_witness(
        "vector_add_i8x64",
        "VADD_8",
        "3c68097283000000000000000000180028100000780d101800001813040a0000",
    ),
)
