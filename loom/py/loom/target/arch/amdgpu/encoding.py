# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable AMDGPU encoding identifiers shared by generators.

Vendor XML names stay in Python. Generated C tables carry dense numeric ids so
native emitters can pack instruction bits without string dispatch.
"""

from __future__ import annotations

from enum import IntEnum

AMDGPU_ENCODING_FORMAT_NONE = 0

_AMDGPU_ENCODING_FORMAT_ID_PAIRS = (
    ("ENC_SOP1", 1),
    ("ENC_SOP2", 2),
    ("ENC_SOPP", 3),
    ("ENC_VOP2", 4),
    ("VOP2_INST_LITERAL", 5),
    ("ENC_VOP3", 6),
    ("ENC_VOP3P", 7),
    ("ENC_SMEM", 8),
    ("ENC_MUBUF", 9),
    ("ENC_VBUFFER", 10),
    ("ENC_DS", 11),
    ("ENC_EXP", 12),
    ("ENC_FLAT", 13),
    ("ENC_FLAT_GLBL", 14),
    ("ENC_FLAT_GLOBAL", 15),
    ("ENC_FLAT_SCRATCH", 16),
    ("ENC_LDSDIR", 17),
    ("ENC_MIMG", 18),
    ("ENC_MTBUF", 19),
    ("ENC_SOPC", 20),
    ("ENC_SOPK", 21),
    ("ENC_VDS", 22),
    ("ENC_VDSDIR", 23),
    ("ENC_VEXPORT", 24),
    ("ENC_VFLAT", 25),
    ("ENC_VGLOBAL", 26),
    ("ENC_VIMAGE", 27),
    ("ENC_VINTERP", 28),
    ("ENC_VINTRP", 29),
    ("ENC_VOP1", 30),
    ("ENC_VOP3PX2", 31),
    ("ENC_VOPC", 32),
    ("ENC_VSAMPLE", 33),
    ("ENC_VSCRATCH", 34),
    ("MIMG_NSA1", 35),
    ("MIMG_NSA2", 36),
    ("MIMG_NSA3", 37),
    ("SOP1_INST_LITERAL", 38),
    ("SOP2_INST_LITERAL", 39),
    ("SOPC_INST_LITERAL", 40),
    ("SOPK_INST_LITERAL", 41),
    ("VOP1_INST_LITERAL", 42),
    ("VOP1_VOP_DPP", 43),
    ("VOP1_VOP_DPP16", 44),
    ("VOP1_VOP_DPP8", 45),
    ("VOP1_VOP_SDWA", 46),
    ("VOP2_VOP_DPP", 47),
    ("VOP2_VOP_DPP16", 48),
    ("VOP2_VOP_DPP8", 49),
    ("VOP2_VOP_SDWA", 50),
    ("VOP2_VOP_SDWA_SDST_ENC", 51),
    ("VOP3P_INST_LITERAL", 52),
    ("VOP3P_MFMA", 53),
    ("VOP3P_VOP_DPP16", 54),
    ("VOP3P_VOP_DPP8", 55),
    ("VOP3_INST_LITERAL", 56),
    ("VOP3_SDST_ENC", 57),
    ("VOP3_SDST_ENC_INST_LITERAL", 58),
    ("VOP3_SDST_ENC_VOP_DPP16", 59),
    ("VOP3_SDST_ENC_VOP_DPP8", 60),
    ("VOP3_VOP_DPP16", 61),
    ("VOP3_VOP_DPP8", 62),
    ("VOPC_INST_LITERAL", 63),
    ("VOPC_VOP_DPP16", 64),
    ("VOPC_VOP_DPP8", 65),
    ("VOPC_VOP_SDWA_SDST_ENC", 66),
    ("VOPDXY", 67),
    ("VOPDXY_INST_LITERAL", 68),
    ("ENC_VOP1_VGPR", 69),
)

AMDGPU_ENCODING_FORMAT_IDS = dict(_AMDGPU_ENCODING_FORMAT_ID_PAIRS)
AMDGPU_ENCODING_FORMAT_XML_NAMES = tuple(
    name for name, _format_id in _AMDGPU_ENCODING_FORMAT_ID_PAIRS
)
AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID = {
    format_id: name for name, format_id in _AMDGPU_ENCODING_FORMAT_ID_PAIRS
}

_AMDGPU_SUPPLEMENTAL_ENCODING_FORMAT_NAMES_BY_TARGET = {
    "gfx12_generic": ("ENC_VOP3PX2",),
    "gfx12_5_generic": ("ENC_VOP3PX2", "ENC_VOP1_VGPR"),
    "rdna4": ("ENC_VOP3PX2",),
    "rdna4_gfx1250_a0": ("ENC_VOP3PX2", "ENC_VOP1_VGPR"),
    "rdna4_gfx1251": ("ENC_VOP3PX2", "ENC_VOP1_VGPR"),
    "rdna4_gfx125x": ("ENC_VOP3PX2", "ENC_VOP1_VGPR"),
}


def amdgpu_supplemental_encoding_format_names(target: str) -> tuple[str, ...]:
    """Returns encoding formats supplied outside of the target ISA XML."""

    return _AMDGPU_SUPPLEMENTAL_ENCODING_FORMAT_NAMES_BY_TARGET.get(target, ())


AMDGPU_ENCODING_FORMAT_SOP1 = AMDGPU_ENCODING_FORMAT_IDS["ENC_SOP1"]
AMDGPU_ENCODING_FORMAT_SOP1_LITERAL = AMDGPU_ENCODING_FORMAT_IDS["SOP1_INST_LITERAL"]
AMDGPU_ENCODING_FORMAT_SOP2 = AMDGPU_ENCODING_FORMAT_IDS["ENC_SOP2"]
AMDGPU_ENCODING_FORMAT_SOP2_LITERAL = AMDGPU_ENCODING_FORMAT_IDS["SOP2_INST_LITERAL"]
AMDGPU_ENCODING_FORMAT_SOPP = AMDGPU_ENCODING_FORMAT_IDS["ENC_SOPP"]
AMDGPU_ENCODING_FORMAT_SOPC = AMDGPU_ENCODING_FORMAT_IDS["ENC_SOPC"]
AMDGPU_ENCODING_FORMAT_VOP2 = AMDGPU_ENCODING_FORMAT_IDS["ENC_VOP2"]
AMDGPU_ENCODING_FORMAT_VOP2_LITERAL = AMDGPU_ENCODING_FORMAT_IDS["VOP2_INST_LITERAL"]
AMDGPU_ENCODING_FORMAT_VOP3 = AMDGPU_ENCODING_FORMAT_IDS["ENC_VOP3"]
AMDGPU_ENCODING_FORMAT_VOP3_LITERAL = AMDGPU_ENCODING_FORMAT_IDS["VOP3_INST_LITERAL"]
AMDGPU_ENCODING_FORMAT_VOP3_DPP16 = AMDGPU_ENCODING_FORMAT_IDS["VOP3_VOP_DPP16"]
AMDGPU_ENCODING_FORMAT_VOP3_DPP8 = AMDGPU_ENCODING_FORMAT_IDS["VOP3_VOP_DPP8"]
AMDGPU_ENCODING_FORMAT_VOP3P_DPP16 = AMDGPU_ENCODING_FORMAT_IDS["VOP3P_VOP_DPP16"]
AMDGPU_ENCODING_FORMAT_VOP3P_DPP8 = AMDGPU_ENCODING_FORMAT_IDS["VOP3P_VOP_DPP8"]
AMDGPU_ENCODING_FORMAT_VOP3_SDST_DPP16 = AMDGPU_ENCODING_FORMAT_IDS[
    "VOP3_SDST_ENC_VOP_DPP16"
]
AMDGPU_ENCODING_FORMAT_VOP3_SDST_DPP8 = AMDGPU_ENCODING_FORMAT_IDS[
    "VOP3_SDST_ENC_VOP_DPP8"
]
AMDGPU_ENCODING_FORMAT_VOPC_DPP16 = AMDGPU_ENCODING_FORMAT_IDS["VOPC_VOP_DPP16"]
AMDGPU_ENCODING_FORMAT_VOPC_DPP8 = AMDGPU_ENCODING_FORMAT_IDS["VOPC_VOP_DPP8"]
AMDGPU_ENCODING_FORMAT_VOP3P = AMDGPU_ENCODING_FORMAT_IDS["ENC_VOP3P"]
AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL = AMDGPU_ENCODING_FORMAT_IDS["VOP3P_INST_LITERAL"]
AMDGPU_ENCODING_FORMAT_VOP3PX2 = AMDGPU_ENCODING_FORMAT_IDS["ENC_VOP3PX2"]
AMDGPU_ENCODING_FORMAT_VOP3_SDST = AMDGPU_ENCODING_FORMAT_IDS["VOP3_SDST_ENC"]
AMDGPU_ENCODING_FORMAT_SMEM = AMDGPU_ENCODING_FORMAT_IDS["ENC_SMEM"]
AMDGPU_ENCODING_FORMAT_MUBUF = AMDGPU_ENCODING_FORMAT_IDS["ENC_MUBUF"]
AMDGPU_ENCODING_FORMAT_VBUFFER = AMDGPU_ENCODING_FORMAT_IDS["ENC_VBUFFER"]
AMDGPU_ENCODING_FORMAT_DS = AMDGPU_ENCODING_FORMAT_IDS["ENC_DS"]
AMDGPU_ENCODING_FORMAT_FLAT = AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT"]
AMDGPU_ENCODING_FORMAT_VDS = AMDGPU_ENCODING_FORMAT_IDS["ENC_VDS"]
AMDGPU_ENCODING_FORMAT_VFLAT = AMDGPU_ENCODING_FORMAT_IDS["ENC_VFLAT"]
AMDGPU_ENCODING_FORMAT_VGLOBAL = AMDGPU_ENCODING_FORMAT_IDS["ENC_VGLOBAL"]
AMDGPU_ENCODING_FORMAT_VIMAGE = AMDGPU_ENCODING_FORMAT_IDS["ENC_VIMAGE"]
AMDGPU_ENCODING_FORMAT_VOP1 = AMDGPU_ENCODING_FORMAT_IDS["ENC_VOP1"]
AMDGPU_ENCODING_FORMAT_VOP1_LITERAL = AMDGPU_ENCODING_FORMAT_IDS["VOP1_INST_LITERAL"]
AMDGPU_ENCODING_FORMAT_VOP1_DPP = AMDGPU_ENCODING_FORMAT_IDS["VOP1_VOP_DPP"]
AMDGPU_ENCODING_FORMAT_VOP1_DPP16 = AMDGPU_ENCODING_FORMAT_IDS["VOP1_VOP_DPP16"]
AMDGPU_ENCODING_FORMAT_VOP1_DPP8 = AMDGPU_ENCODING_FORMAT_IDS["VOP1_VOP_DPP8"]
AMDGPU_ENCODING_FORMAT_VOP1_SDWA = AMDGPU_ENCODING_FORMAT_IDS["VOP1_VOP_SDWA"]
AMDGPU_ENCODING_FORMAT_VOP2_DPP = AMDGPU_ENCODING_FORMAT_IDS["VOP2_VOP_DPP"]
AMDGPU_ENCODING_FORMAT_VOP2_DPP16 = AMDGPU_ENCODING_FORMAT_IDS["VOP2_VOP_DPP16"]
AMDGPU_ENCODING_FORMAT_VOP2_DPP8 = AMDGPU_ENCODING_FORMAT_IDS["VOP2_VOP_DPP8"]
AMDGPU_ENCODING_FORMAT_VSCRATCH = AMDGPU_ENCODING_FORMAT_IDS["ENC_VSCRATCH"]
AMDGPU_ENCODING_FORMAT_VOPDXY = AMDGPU_ENCODING_FORMAT_IDS["VOPDXY"]
AMDGPU_ENCODING_FORMAT_VOPDXY_LITERAL = AMDGPU_ENCODING_FORMAT_IDS[
    "VOPDXY_INST_LITERAL"
]
AMDGPU_ENCODING_FORMAT_VOP1_VGPR = AMDGPU_ENCODING_FORMAT_IDS["ENC_VOP1_VGPR"]

AMDGPU_VOP3_ENCODING_FORMAT_NAMES = frozenset(
    (
        "ENC_VOP3",
        "ENC_VOP3P",
        "ENC_VOP3PX2",
        "VOP3P_INST_LITERAL",
        "VOP3P_MFMA",
        "VOP3P_VOP_DPP16",
        "VOP3P_VOP_DPP8",
        "VOP3_INST_LITERAL",
        "VOP3_SDST_ENC",
        "VOP3_SDST_ENC_INST_LITERAL",
        "VOP3_SDST_ENC_VOP_DPP16",
        "VOP3_SDST_ENC_VOP_DPP8",
        "VOP3_VOP_DPP16",
        "VOP3_VOP_DPP8",
    )
)

AMDGPU_DPP_CONTROL_ENCODING_FORMAT_IDS = frozenset(
    (
        AMDGPU_ENCODING_FORMAT_VOP1_DPP,
        AMDGPU_ENCODING_FORMAT_VOP1_DPP16,
        AMDGPU_ENCODING_FORMAT_VOP2_DPP,
        AMDGPU_ENCODING_FORMAT_VOP2_DPP16,
        AMDGPU_ENCODING_FORMAT_VOP3P_DPP16,
        AMDGPU_ENCODING_FORMAT_VOP3_SDST_DPP16,
        AMDGPU_ENCODING_FORMAT_VOP3_DPP16,
        AMDGPU_ENCODING_FORMAT_VOPC_DPP16,
    )
)


def amdgpu_dpp_control_is_valid(value: int) -> bool:
    """Returns whether value is an architectural DPP control encoding."""

    if 0 <= value <= 0x0FF:
        return True
    group = value >> 4
    selector = value & 0xF
    if group in (0x10, 0x11, 0x12):
        return selector != 0
    if group == 0x13:
        return selector <= 0xC and selector % 4 == 0
    if group == 0x14:
        return selector <= 3
    return group in (0x15, 0x16)


class AmdgpuVgprMsbFormatClass(IntEnum):
    """Encoding field family controlled by S_SET_VGPR_MSB."""

    NONE = 0
    VOP = 1
    DS = 2
    FLAT = 3
    BUFFER = 4


class AmdgpuVgprMsbSlot(IntEnum):
    """S_SET_VGPR_MSB selector slot used by an encoding field."""

    NONE = 0
    SRC0 = 1
    SRC1 = 2
    SRC2 = 3
    DST = 4


AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE = 256
AMDGPU_GFX125X_VGPR_MSB_BANK_COUNT = 4


AMDGPU_GFX125X_VOP_VGPR_MSB_FORMAT_NAMES = (
    "ENC_VOP1",
    "ENC_VOP1_VGPR",
    "ENC_VOP2",
    "ENC_VOP3",
    "ENC_VOP3P",
    "ENC_VOP3PX2",
    "ENC_VOPC",
    "VOP1_INST_LITERAL",
    "VOP1_VOP_DPP",
    "VOP1_VOP_DPP16",
    "VOP1_VOP_DPP8",
    "VOP1_VOP_SDWA",
    "VOP2_INST_LITERAL",
    "VOP2_VOP_DPP",
    "VOP2_VOP_DPP16",
    "VOP2_VOP_DPP8",
    "VOP2_VOP_SDWA",
    "VOP2_VOP_SDWA_SDST_ENC",
    "VOP3P_INST_LITERAL",
    "VOP3P_MFMA",
    "VOP3P_VOP_DPP16",
    "VOP3P_VOP_DPP8",
    "VOP3_INST_LITERAL",
    "VOP3_SDST_ENC",
    "VOP3_SDST_ENC_INST_LITERAL",
    "VOP3_SDST_ENC_VOP_DPP16",
    "VOP3_SDST_ENC_VOP_DPP8",
    "VOP3_VOP_DPP16",
    "VOP3_VOP_DPP8",
    "VOPC_INST_LITERAL",
    "VOPC_VOP_DPP16",
    "VOPC_VOP_DPP8",
    "VOPC_VOP_SDWA_SDST_ENC",
)


AMDGPU_GFX125X_VGPR_MSB_FORMAT_CLASSES_BY_ID = {
    **dict.fromkeys(
        tuple(
            AMDGPU_ENCODING_FORMAT_IDS[name]
            for name in AMDGPU_GFX125X_VOP_VGPR_MSB_FORMAT_NAMES
        ),
        AmdgpuVgprMsbFormatClass.VOP,
    ),
    **dict.fromkeys(
        (AMDGPU_ENCODING_FORMAT_DS, AMDGPU_ENCODING_FORMAT_VDS),
        AmdgpuVgprMsbFormatClass.DS,
    ),
    **dict.fromkeys(
        (
            AMDGPU_ENCODING_FORMAT_FLAT,
            AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT_GLBL"],
            AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT_GLOBAL"],
            AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT_SCRATCH"],
            AMDGPU_ENCODING_FORMAT_VFLAT,
            AMDGPU_ENCODING_FORMAT_VGLOBAL,
            AMDGPU_ENCODING_FORMAT_VSCRATCH,
        ),
        AmdgpuVgprMsbFormatClass.FLAT,
    ),
    **dict.fromkeys(
        (
            AMDGPU_ENCODING_FORMAT_MUBUF,
            AMDGPU_ENCODING_FORMAT_IDS["ENC_MTBUF"],
            AMDGPU_ENCODING_FORMAT_VBUFFER,
        ),
        AmdgpuVgprMsbFormatClass.BUFFER,
    ),
}

AMDGPU_VGPR_MSB_FIELD_SLOTS_BY_FORMAT_CLASS = {
    AmdgpuVgprMsbFormatClass.VOP: {
        "SRC0": AmdgpuVgprMsbSlot.SRC0,
        "VSRC0": AmdgpuVgprMsbSlot.SRC0,
        "SRC1": AmdgpuVgprMsbSlot.SRC1,
        "VSRC1": AmdgpuVgprMsbSlot.SRC1,
        "SRC2": AmdgpuVgprMsbSlot.SRC2,
        "VSRC2": AmdgpuVgprMsbSlot.SRC2,
        "VDST": AmdgpuVgprMsbSlot.DST,
    },
    AmdgpuVgprMsbFormatClass.DS: {
        "ADDR": AmdgpuVgprMsbSlot.SRC0,
        "DATA0": AmdgpuVgprMsbSlot.SRC1,
        "DATA1": AmdgpuVgprMsbSlot.SRC2,
        "VDST": AmdgpuVgprMsbSlot.DST,
    },
    AmdgpuVgprMsbFormatClass.FLAT: {
        "ADDR": AmdgpuVgprMsbSlot.SRC0,
        "VADDR": AmdgpuVgprMsbSlot.SRC0,
        "DATA": AmdgpuVgprMsbSlot.SRC1,
        "VDATA": AmdgpuVgprMsbSlot.SRC1,
        "VSRC": AmdgpuVgprMsbSlot.SRC1,
        "VDST": AmdgpuVgprMsbSlot.DST,
    },
    AmdgpuVgprMsbFormatClass.BUFFER: {
        "VADDR": AmdgpuVgprMsbSlot.SRC0,
        "VDATA": AmdgpuVgprMsbSlot.DST,
        "VSRC": AmdgpuVgprMsbSlot.DST,
        "VDST": AmdgpuVgprMsbSlot.DST,
    },
}


AMDGPU_GFX125X_VGPR_MSB_SLOT_OVERRIDES_BY_DESCRIPTOR_KEY = {
    descriptor_key: {
        "SRC1": AmdgpuVgprMsbSlot.SRC2,
        "VSRC1": AmdgpuVgprMsbSlot.SRC2,
    }
    for descriptor_key in (
        "amdgpu.v_fmamk_f16",
        "amdgpu.v_fmamk_f32",
        "amdgpu.v_fmamk_f64",
    )
}


def amdgpu_gfx125x_vgpr_msb_slot(
    descriptor_key: str, encoding_format_id: int, encoding_field_name: str
) -> AmdgpuVgprMsbSlot:
    """Returns the VGPR-MSB slot controlling an encoding field."""

    descriptor_overrides = AMDGPU_GFX125X_VGPR_MSB_SLOT_OVERRIDES_BY_DESCRIPTOR_KEY.get(
        descriptor_key, {}
    )
    slot_override = descriptor_overrides.get(encoding_field_name)
    if slot_override is not None:
        return slot_override
    format_class = AMDGPU_GFX125X_VGPR_MSB_FORMAT_CLASSES_BY_ID.get(
        encoding_format_id, AmdgpuVgprMsbFormatClass.NONE
    )
    return AMDGPU_VGPR_MSB_FIELD_SLOTS_BY_FORMAT_CLASS.get(format_class, {}).get(
        encoding_field_name, AmdgpuVgprMsbSlot.NONE
    )


AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_OFFSET = 11
AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_COUNT = 4

AMDGPU_ENCODING_FIELD_NAMES = (
    "A16",
    "ABID",
    "ABS",
    "ACC",
    "ACC_CD",
    "ADDR",
    "ATTR",
    "ATTRCHAN",
    "ATTR_CHAN",
    "BANK_MASK",
    "BLGP",
    "BOUND_CTRL",
    "BREAK_SPAN",
    "Bits",
    "CBSZ",
    "CLAMP",
    "COMPR",
    "D16",
    "DA",
    "DATA",
    "DATA0",
    "DATA1",
    "DFMT",
    "DIM",
    "DLC",
    "DMASK",
    "DONE",
    "DONT_WAIT_EXPORT_READY",
    "DPP_CTRL",
    "DS",
    "DST_SEL",
    "DST_UNUSED",
    "DURATION",
    "Descriptor",
    "EN",
    "ENCODING",
    "EXP",
    "EXPORT_READY",
    "Exponent",
    "FI",
    "FORMAT",
    "GDS",
    "GLC",
    "GSOP",
    "HAS_RTN",
    "HOLD_CNT",
    "ID",
    "IDXEN",
    "IMM",
    "INSTID0",
    "INSTID1",
    "INSTSKIP",
    "IOFFSET",
    "Integer",
    "LANE_SEL_0",
    "LANE_SEL_1",
    "LANE_SEL_2",
    "LANE_SEL_3",
    "LANE_SEL_4",
    "LANE_SEL_5",
    "LANE_SEL_6",
    "LANE_SEL_7",
    "LDS",
    "LENGTH",
    "LGKM",
    "LITERAL",
    "LWE",
    "MDP",
    "MEM",
    "MSG",
    "Mantissa",
    "NEG",
    "NEG_HI",
    "NFMT",
    "NSA",
    "NT",
    "NV",
    "OFFEN",
    "OFFSET",
    "OFFSET0",
    "OFFSET1",
    "OMOD",
    "OP",
    "OPSEL",
    "OPSEL_HI",
    "OPX",
    "OPY",
    "OP_SEL",
    "OP_SEL_HI",
    "OP_SEL_HI_2",
    "R128",
    "ROW_EN",
    "ROW_MASK",
    "RSRC",
    "S0",
    "S1",
    "SADDR",
    "SAMP",
    "SA_SDST",
    "SBASE",
    "SC0",
    "SC1",
    "SCALE_SRC0",
    "SCALE_SRC1",
    "SCC",
    "SCOPE",
    "SD",
    "SDATA",
    "SDST",
    "SEG",
    "SIMM16",
    "SIZE",
    "SLC",
    "SLEEP_FOREVER",
    "SOFFSET",
    "SOFFSET_EN",
    "SRC0",
    "SRC0_ABS",
    "SRC0_NEG",
    "SRC0_SEL",
    "SRC0_SEXT",
    "SRC1",
    "SRC1_ABS",
    "SRC1_NEG",
    "SRC1_SEL",
    "SRC1_SEXT",
    "SRC2",
    "SRCX0",
    "SRCY0",
    "SRSRC",
    "SSAMP",
    "SSRC0",
    "SSRC1",
    "STREAMID",
    "SVE",
    "SYSTEM",
    "Sign",
    "TARGET",
    "TFE",
    "TGT",
    "TH",
    "UNORM",
    "VADDR",
    "VADDR0",
    "VADDR1",
    "VADDR2",
    "VADDR3",
    "VADDR4",
    "VADDRA",
    "VADDRB",
    "VADDRC",
    "VADDRD",
    "VADDRE",
    "VADDRF",
    "VADDRG",
    "VADDRH",
    "VADDRI",
    "VADDRJ",
    "VADDRK",
    "VADDRL",
    "VALUE",
    "VA_SDST",
    "VA_SSRC",
    "VA_VCC",
    "VA_VDST",
    "VDATA",
    "VDST",
    "VDSTX",
    "VDSTY",
    "VERSION",
    "VM",
    "VM_VSRC",
    "VSRC",
    "VSRC0",
    "VSRC1",
    "VSRC2",
    "VSRC3",
    "VSRCX1",
    "VSRCY1",
    "W32",
    "W64",
    "WAIT_EXP",
    "WAIT_VA_VDST",
    "WAIT_VDST",
    "WAIT_VM_VSRC",
    "X2ENCODING",
    "INDEX_KEY_16BIT",
    "MATRIX_A_FMT",
    "MATRIX_A_REUSE",
    "MATRIX_A_SCALE",
    "MATRIX_A_SCALE_FMT",
    "MATRIX_B_FMT",
    "MATRIX_B_REUSE",
    "MATRIX_B_SCALE",
    "MATRIX_B_SCALE_FMT",
    "SCALE_SEL",
)

AMDGPU_ENCODING_FIELD_IDS = {
    name: field_id for field_id, name in enumerate(AMDGPU_ENCODING_FIELD_NAMES, start=1)
}
AMDGPU_ENCODING_FIELD_NAMES_BY_ID = {
    field_id: name for name, field_id in AMDGPU_ENCODING_FIELD_IDS.items()
}


def amdgpu_encoding_format_id(xml_name: str) -> int:
    try:
        return AMDGPU_ENCODING_FORMAT_IDS[xml_name]
    except KeyError as exc:
        raise KeyError(f"unmapped AMDGPU encoding format '{xml_name}'") from exc


def amdgpu_encoding_field_id(field_name: str) -> int:
    try:
        return AMDGPU_ENCODING_FIELD_IDS[field_name]
    except KeyError as exc:
        raise KeyError(f"unmapped AMDGPU encoding field '{field_name}'") from exc


def amdgpu_encoding_field_name(field_id: int) -> str:
    try:
        return AMDGPU_ENCODING_FIELD_NAMES_BY_ID[field_id]
    except KeyError as exc:
        raise KeyError(f"unmapped AMDGPU encoding field id {field_id}") from exc
