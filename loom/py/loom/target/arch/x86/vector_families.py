# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Immutable x86 vector-family rows shared by descriptor and contract generation."""

from __future__ import annotations

from dataclasses import dataclass

AVX2_VECTOR_BIT_WIDTHS = (128, 256)
# The x86 call ABI carries subregister vectors in the low bits of an XMM
# register. These payloads use AVX2 construction and lane-movement instructions
# even though they are narrower than an architecturally native packed arithmetic
# width.
AVX2_CARRIER_VECTOR_BIT_WIDTHS = (8, 16, 32, 64, *AVX2_VECTOR_BIT_WIDTHS)
# Packed floating-point instructions operate lane-wise over the low-XMM carrier,
# so 64-bit logical packets use the same XMM descriptors as architecturally
# native packed widths. One-lane f32 packets retain their scalar projection.
AVX2_FLOAT_VECTOR_BIT_WIDTHS = (64, *AVX2_VECTOR_BIT_WIDTHS)
AVX512_VECTOR_BIT_WIDTHS = (512,)
AVX512_CARRIER_VECTOR_BIT_WIDTHS = (
    *AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    *AVX512_VECTOR_BIT_WIDTHS,
)
AVX512VL_VECTOR_BIT_WIDTHS = AVX2_VECTOR_BIT_WIDTHS
# One-lane vectors use the scalar FP16 family. Packed forms start at two lanes.
AVX512_FP16_VECTOR_BIT_WIDTHS = (32, 64, 128, 256, 512)
AVX2_ROOT_VECTOR_BIT_WIDTHS = AVX2_FLOAT_VECTOR_BIT_WIDTHS
AVX512_ROOT_VECTOR_BIT_WIDTHS = (
    *AVX2_ROOT_VECTOR_BIT_WIDTHS,
    *AVX512_VECTOR_BIT_WIDTHS,
)
AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS = (
    *AVX2_VECTOR_BIT_WIDTHS,
    *AVX512_VECTOR_BIT_WIDTHS,
)


@dataclass(frozen=True, slots=True)
class VectorElement:
    name: str
    bit_width: int

    def lane_count(self, vector_bit_width: int) -> int:
        if vector_bit_width % self.bit_width:
            raise ValueError(
                f"vector width {vector_bit_width} does not contain whole "
                f"{self.name} lanes"
            )
        return vector_bit_width // self.bit_width


@dataclass(frozen=True, slots=True)
class VectorBinaryFamily:
    source_operation: str
    mnemonic: str
    semantic: str
    element: VectorElement


@dataclass(frozen=True, slots=True)
class FloatRootFamily:
    """Square-root operation row with its source precision contract."""

    source_operation: str
    mnemonic: str
    semantic: str
    element: VectorElement
    approximate: bool


@dataclass(frozen=True, slots=True)
class VectorLaneFamily:
    """Bit-compatible scalar lane movement shared by source element types."""

    element_bit_width: int
    element_names: tuple[str, ...]
    extract_mnemonic: str
    insert_mnemonic: str


@dataclass(frozen=True, slots=True)
class VectorStorageFamily:
    """Bit-compatible vector storage shared by source element types."""

    element_bit_width: int
    element_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VectorIntegerWideningFamily:
    """One signed and unsigned packed-integer widening pair."""

    source: VectorElement
    result: VectorElement
    signed_mnemonic: str
    unsigned_mnemonic: str


@dataclass(frozen=True, slots=True)
class VectorIntegerNarrowingFamily:
    """One packed-integer narrowing pair."""

    source: VectorElement
    result: VectorElement
    mnemonic: str

    @property
    def avx2_half_result_bit_width(self) -> int:
        """Bit width produced from one 128-bit AVX2 source lane."""

        return 128 * self.result.bit_width // self.source.bit_width


INTEGER_ELEMENTS = (
    VectorElement("i8", 8),
    VectorElement("i16", 16),
    VectorElement("i32", 32),
    VectorElement("i64", 64),
)

X86_VECTOR_INTEGER_WIDENING_FAMILIES = (
    VectorIntegerWideningFamily(
        INTEGER_ELEMENTS[0], INTEGER_ELEMENTS[1], "vpmovsxbw", "vpmovzxbw"
    ),
    VectorIntegerWideningFamily(
        INTEGER_ELEMENTS[0], INTEGER_ELEMENTS[2], "vpmovsxbd", "vpmovzxbd"
    ),
    VectorIntegerWideningFamily(
        INTEGER_ELEMENTS[0], INTEGER_ELEMENTS[3], "vpmovsxbq", "vpmovzxbq"
    ),
    VectorIntegerWideningFamily(
        INTEGER_ELEMENTS[1], INTEGER_ELEMENTS[2], "vpmovsxwd", "vpmovzxwd"
    ),
    VectorIntegerWideningFamily(
        INTEGER_ELEMENTS[1], INTEGER_ELEMENTS[3], "vpmovsxwq", "vpmovzxwq"
    ),
    VectorIntegerWideningFamily(
        INTEGER_ELEMENTS[2], INTEGER_ELEMENTS[3], "vpmovsxdq", "vpmovzxdq"
    ),
)

X86_VECTOR_INTEGER_NARROWING_FAMILIES = (
    VectorIntegerNarrowingFamily(INTEGER_ELEMENTS[1], INTEGER_ELEMENTS[0], "vpmovwb"),
    VectorIntegerNarrowingFamily(INTEGER_ELEMENTS[2], INTEGER_ELEMENTS[0], "vpmovdb"),
    VectorIntegerNarrowingFamily(INTEGER_ELEMENTS[3], INTEGER_ELEMENTS[0], "vpmovqb"),
    VectorIntegerNarrowingFamily(INTEGER_ELEMENTS[2], INTEGER_ELEMENTS[1], "vpmovdw"),
    VectorIntegerNarrowingFamily(INTEGER_ELEMENTS[3], INTEGER_ELEMENTS[1], "vpmovqw"),
    VectorIntegerNarrowingFamily(INTEGER_ELEMENTS[3], INTEGER_ELEMENTS[2], "vpmovqd"),
)

FLOAT_ELEMENTS = (
    VectorElement("f32", 32),
    VectorElement("f64", 64),
)

FP16_ELEMENT = VectorElement("f16", 16)

STORAGE_ELEMENTS = (
    VectorElement("f8E4M3", 8),
    VectorElement("f8E5M2", 8),
    FP16_ELEMENT,
    VectorElement("bf16", 16),
)

X86_LANE_FAMILIES = (
    VectorLaneFamily(
        8,
        ("i8", "f8E4M3", "f8E5M2"),
        "vpextrb",
        "vpinsrb",
    ),
    VectorLaneFamily(
        16,
        ("i16", "f16", "bf16"),
        "vpextrw",
        "vpinsrw",
    ),
    VectorLaneFamily(32, ("i32",), "vpextrd", "vpinsrd"),
    VectorLaneFamily(64, ("i64",), "vpextrq", "vpinsrq"),
)

X86_VECTOR_STORAGE_FAMILIES = (
    VectorStorageFamily(8, ("i8", "f8E4M3", "f8E5M2")),
    VectorStorageFamily(16, ("i16", "f16", "bf16")),
    VectorStorageFamily(32, ("i32", "f32")),
    VectorStorageFamily(64, ("i64", "f64")),
)

AVX2_SHUFFLE_FAMILIES = X86_VECTOR_STORAGE_FAMILIES

AVX2_INTEGER_BINARY_FAMILIES = (
    *(
        VectorBinaryFamily("addi", f"vpadd{suffix}", "integer.add", element)
        for element, suffix in zip(INTEGER_ELEMENTS, ("b", "w", "d", "q"), strict=True)
    ),
    *(
        VectorBinaryFamily("subi", f"vpsub{suffix}", "integer.sub", element)
        for element, suffix in zip(INTEGER_ELEMENTS, ("b", "w", "d", "q"), strict=True)
    ),
    VectorBinaryFamily("muli", "vpmullw", "integer.mul", INTEGER_ELEMENTS[1]),
    VectorBinaryFamily("muli", "vpmulld", "integer.mul", INTEGER_ELEMENTS[2]),
    *(
        VectorBinaryFamily("minsi", mnemonic, "integer.mins", element)
        for element, mnemonic in zip(
            INTEGER_ELEMENTS[:3], ("vpminsb", "vpminsw", "vpminsd"), strict=True
        )
    ),
    *(
        VectorBinaryFamily("maxsi", mnemonic, "integer.maxs", element)
        for element, mnemonic in zip(
            INTEGER_ELEMENTS[:3], ("vpmaxsb", "vpmaxsw", "vpmaxsd"), strict=True
        )
    ),
    *(
        VectorBinaryFamily("minui", mnemonic, "integer.minu", element)
        for element, mnemonic in zip(
            INTEGER_ELEMENTS[:3], ("vpminub", "vpminuw", "vpminud"), strict=True
        )
    ),
    *(
        VectorBinaryFamily("maxui", mnemonic, "integer.maxu", element)
        for element, mnemonic in zip(
            INTEGER_ELEMENTS[:3], ("vpmaxub", "vpmaxuw", "vpmaxud"), strict=True
        )
    ),
    VectorBinaryFamily("shli", "vpsllvd", "integer.shl", INTEGER_ELEMENTS[2]),
    VectorBinaryFamily("shli", "vpsllvq", "integer.shl", INTEGER_ELEMENTS[3]),
    VectorBinaryFamily("shrsi", "vpsravd", "integer.shrs", INTEGER_ELEMENTS[2]),
    VectorBinaryFamily("shrui", "vpsrlvd", "integer.shru", INTEGER_ELEMENTS[2]),
    VectorBinaryFamily("shrui", "vpsrlvq", "integer.shru", INTEGER_ELEMENTS[3]),
)

# Immediate and uniform-count shifts are distinct from per-lane variable shifts.
AVX2_UNIFORM_SHIFT_FAMILIES = tuple(
    VectorBinaryFamily(operation, f"{mnemonic}{suffix}", semantic, element)
    for operation, mnemonic, semantic in (
        ("shli", "vpsll", "integer.shl"),
        ("shrsi", "vpsra", "integer.shrs"),
        ("shrui", "vpsrl", "integer.shru"),
    )
    for element, suffix in zip(INTEGER_ELEMENTS[1:], ("w", "d", "q"), strict=True)
    if not (operation == "shrsi" and element.bit_width == 64)
)
AVX512VL_UNIFORM_SHIFT_FAMILIES = (
    VectorBinaryFamily("shrsi", "vpsraq", "integer.shrs", INTEGER_ELEMENTS[3]),
)
AVX512_UNIFORM_SHIFT_FAMILIES = (
    *AVX2_UNIFORM_SHIFT_FAMILIES,
    *AVX512VL_UNIFORM_SHIFT_FAMILIES,
)

# AVX-512BW extends the word variable shifts while AVX-512DQ completes the
# native i64 multiply, extrema, and arithmetic-shift cells. Core AVX-512 has no
# native byte-shift or byte-multiply forms, so those use a distinct composed
# lowering mechanism.
AVX512_INTEGER_BINARY_FAMILIES = (
    *AVX2_INTEGER_BINARY_FAMILIES,
    VectorBinaryFamily("muli", "vpmullq", "integer.mul", INTEGER_ELEMENTS[3]),
    *(
        VectorBinaryFamily(source_operation, mnemonic, semantic, INTEGER_ELEMENTS[3])
        for source_operation, mnemonic, semantic in (
            ("minsi", "vpminsq", "integer.mins"),
            ("maxsi", "vpmaxsq", "integer.maxs"),
            ("minui", "vpminuq", "integer.minu"),
            ("maxui", "vpmaxuq", "integer.maxu"),
        )
    ),
    *(
        VectorBinaryFamily(source_operation, mnemonic, semantic, INTEGER_ELEMENTS[1])
        for source_operation, mnemonic, semantic in (
            ("shli", "vpsllvw", "integer.shl"),
            ("shrsi", "vpsravw", "integer.shrs"),
            ("shrui", "vpsrlvw", "integer.shru"),
        )
    ),
    VectorBinaryFamily("shrsi", "vpsravq", "integer.shrs", INTEGER_ELEMENTS[3]),
)

_AVX2_INTEGER_BINARY_KEYS = frozenset(
    (family.source_operation, family.element.name)
    for family in AVX2_INTEGER_BINARY_FAMILIES
)
AVX512VL_INTEGER_BINARY_FAMILIES = tuple(
    family
    for family in AVX512_INTEGER_BINARY_FAMILIES
    if (family.source_operation, family.element.name) not in _AVX2_INTEGER_BINARY_KEYS
)

AVX2_FLOAT_BINARY_FAMILIES = tuple(
    VectorBinaryFamily(source_operation, f"v{stem}{suffix}", semantic, element)
    for source_operation, stem, semantic in (
        ("addf", "add", "float.add"),
        ("subf", "sub", "float.sub"),
        ("mulf", "mul", "float.mul"),
        ("divf", "div", "float.div"),
    )
    for element, suffix in zip(FLOAT_ELEMENTS, ("ps", "pd"), strict=True)
)

AVX512_FLOAT_BINARY_FAMILIES = AVX2_FLOAT_BINARY_FAMILIES

AVX512_FP16_FLOAT_BINARY_FAMILIES = tuple(
    VectorBinaryFamily(source_operation, f"v{stem}ph", semantic, FP16_ELEMENT)
    for source_operation, stem, semantic in (
        ("addf", "add", "float.add"),
        ("subf", "sub", "float.sub"),
        ("mulf", "mul", "float.mul"),
        ("divf", "div", "float.div"),
    )
)

AVX2_SCALAR_FLOAT_BINARY_FAMILIES = tuple(
    VectorBinaryFamily(source_operation, f"v{stem}{suffix}", semantic, element)
    for source_operation, stem, semantic in (
        ("addf", "add", "float.add"),
        ("subf", "sub", "float.sub"),
        ("mulf", "mul", "float.mul"),
        ("divf", "div", "float.div"),
    )
    for element, suffix in zip(FLOAT_ELEMENTS, ("ss", "sd"), strict=True)
)

AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES = tuple(
    VectorBinaryFamily(source_operation, f"v{stem}sh", semantic, FP16_ELEMENT)
    for source_operation, stem, semantic in (
        ("addf", "add", "float.add"),
        ("subf", "sub", "float.sub"),
        ("mulf", "mul", "float.mul"),
        ("divf", "div", "float.div"),
    )
)

AVX2_FLOAT_ROOT_FAMILIES = (
    FloatRootFamily("sqrtf", "vsqrtps", "float.sqrt", FLOAT_ELEMENTS[0], False),
    FloatRootFamily("sqrtf", "vsqrtpd", "float.sqrt", FLOAT_ELEMENTS[1], False),
    FloatRootFamily(
        "rsqrtf", "vrsqrtps", "float.rsqrt.approx", FLOAT_ELEMENTS[0], True
    ),
)

AVX2_SCALAR_FLOAT_ROOT_FAMILIES = (
    FloatRootFamily("sqrtf", "vsqrtss", "float.sqrt", FLOAT_ELEMENTS[0], False),
    FloatRootFamily("sqrtf", "vsqrtsd", "float.sqrt", FLOAT_ELEMENTS[1], False),
    FloatRootFamily(
        "rsqrtf", "vrsqrtss", "float.rsqrt.approx", FLOAT_ELEMENTS[0], True
    ),
)

AVX512_FLOAT_ROOT_FAMILIES = (
    *tuple(row for row in AVX2_FLOAT_ROOT_FAMILIES if not row.approximate),
    FloatRootFamily(
        "rsqrtf", "vrsqrt14ps", "float.rsqrt.approx", FLOAT_ELEMENTS[0], True
    ),
    FloatRootFamily(
        "rsqrtf", "vrsqrt14pd", "float.rsqrt.approx", FLOAT_ELEMENTS[1], True
    ),
)

AVX512_SCALAR_FLOAT_ROOT_FAMILIES = (
    FloatRootFamily(
        "rsqrtf", "vrsqrt14ss", "float.rsqrt.approx", FLOAT_ELEMENTS[0], True
    ),
    FloatRootFamily(
        "rsqrtf", "vrsqrt14sd", "float.rsqrt.approx", FLOAT_ELEMENTS[1], True
    ),
)

AVX512_FP16_FLOAT_ROOT_FAMILIES = (
    FloatRootFamily("sqrtf", "vsqrtph", "float.sqrt", FP16_ELEMENT, False),
    FloatRootFamily("rsqrtf", "vrsqrtph", "float.rsqrt.approx", FP16_ELEMENT, True),
)

AVX512_FP16_SCALAR_FLOAT_ROOT_FAMILIES = (
    FloatRootFamily("sqrtf", "vsqrtsh", "float.sqrt", FP16_ELEMENT, False),
    FloatRootFamily("rsqrtf", "vrsqrtsh", "float.rsqrt.approx", FP16_ELEMENT, True),
)

AVX2_BITWISE_FAMILIES = (
    ("andi", "vpand", "bits.and"),
    ("ori", "vpor", "bits.or"),
    ("xori", "vpxor", "bits.xor"),
)

# The EVEX dword forms operate on the complete 512-bit payload regardless of
# the source element interpretation.
AVX512_BITWISE_FAMILIES = (
    ("andi", "vpandd", "bits.and"),
    ("ori", "vpord", "bits.or"),
    ("xori", "vpxord", "bits.xor"),
)

AVX2_INTEGER_REDUCTION_FAMILIES = (
    *(
        row
        for row in AVX2_INTEGER_BINARY_FAMILIES
        if row.source_operation in ("addi", "muli", "minsi", "maxsi", "minui", "maxui")
    ),
    *(
        VectorBinaryFamily(source_operation, mnemonic, semantic, element)
        for source_operation, mnemonic, semantic in AVX2_BITWISE_FAMILIES
        for element in INTEGER_ELEMENTS
    ),
)

AVX512_INTEGER_REDUCTION_FAMILIES = (
    *(
        row
        for row in AVX512_INTEGER_BINARY_FAMILIES
        if row.source_operation in ("addi", "muli", "minsi", "maxsi", "minui", "maxui")
    ),
    *(
        VectorBinaryFamily(source_operation, mnemonic, semantic, element)
        for source_operation, mnemonic, semantic in AVX512_BITWISE_FAMILIES
        for element in INTEGER_ELEMENTS
    ),
)

AVX2_INTEGER_COMPARE_MNEMONICS = {
    element.name: (f"vpcmpeq{suffix}", f"vpcmpgt{suffix}")
    for element, suffix in zip(INTEGER_ELEMENTS, ("b", "w", "d", "q"), strict=True)
}

AVX2_FLOAT_COMPARE_MNEMONICS = {
    "f32": "vcmpps",
    "f64": "vcmppd",
}

X86_FLOAT_COMPARE_IMMEDIATES = {
    "oeq": 0,
    "ogt": 30,
    "oge": 29,
    "olt": 17,
    "ole": 18,
    "one": 12,
    "ord": 7,
    "ueq": 8,
    "ugt": 22,
    "uge": 21,
    "ult": 25,
    "ule": 26,
    "une": 4,
    "uno": 3,
}

AVX2_FLOAT_FMA_MNEMONICS = {
    "f32": "vfmadd231ps",
    "f64": "vfmadd231pd",
}

AVX512_FLOAT_FMA_MNEMONICS = AVX2_FLOAT_FMA_MNEMONICS

AVX512_INTEGER_COMPARE_MNEMONICS = {
    element.name: (f"vpcmp{suffix}", f"vpcmpu{suffix}")
    for element, suffix in zip(INTEGER_ELEMENTS, ("b", "w", "d", "q"), strict=True)
}

AVX512_FLOAT_COMPARE_MNEMONICS = AVX2_FLOAT_COMPARE_MNEMONICS

AVX512_FP16_FLOAT_COMPARE_MNEMONIC = "vcmpph"
AVX512_FP16_SCALAR_FLOAT_COMPARE_MNEMONIC = "vcmpsh"

AVX512_SELECT_MNEMONICS = {
    **{
        element.name: f"vpblendm{suffix}"
        for element, suffix in zip(INTEGER_ELEMENTS, ("b", "w", "d", "q"), strict=True)
    },
    **{
        element.name: f"vpblendm{suffix}"
        for element, suffix in zip(STORAGE_ELEMENTS, ("b", "b", "w", "w"), strict=True)
    },
    "f32": "vblendmps",
    "f64": "vblendmpd",
}

AVX2_SCALAR_FLOAT_FMA_MNEMONICS = {
    "f32": "vfmadd231ss",
    "f64": "vfmadd231sd",
}

AVX512_FP16_FLOAT_FMA_MNEMONIC = "vfmadd231ph"
AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC = "vfmadd231sh"

FLOAT_EXTREMA_OPERATIONS = (
    "minimumf",
    "maximumf",
    "minnumf",
    "maxnumf",
)

FLOAT_EXTREMA_MNEMONICS = {
    "minimumf": {"f32": "vminps", "f64": "vminpd"},
    "maximumf": {"f32": "vmaxps", "f64": "vmaxpd"},
    "minnumf": {"f32": "vminps", "f64": "vminpd"},
    "maxnumf": {"f32": "vmaxps", "f64": "vmaxpd"},
}

AVX2_SCALAR_FLOAT_EXTREMA_MNEMONICS = {
    "minimumf": {"f32": "vminss", "f64": "vminsd"},
    "maximumf": {"f32": "vmaxss", "f64": "vmaxsd"},
    "minnumf": {"f32": "vminss", "f64": "vminsd"},
    "maxnumf": {"f32": "vmaxss", "f64": "vmaxsd"},
}

AVX512_FP16_FLOAT_EXTREMA_MNEMONICS = {
    "minimumf": "vminph",
    "maximumf": "vmaxph",
    "minnumf": "vminph",
    "maxnumf": "vmaxph",
}

AVX512_FP16_SCALAR_FLOAT_EXTREMA_MNEMONICS = {
    "minimumf": "vminsh",
    "maximumf": "vmaxsh",
    "minnumf": "vminsh",
    "maxnumf": "vmaxsh",
}

AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS = ("addf", "mulf")

AVX2_FLOAT_REDUCTION_OPERATIONS = (
    *AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS,
    *FLOAT_EXTREMA_OPERATIONS,
)

AVX2_PAYLOAD_ELEMENT_NAMES = tuple(
    element.name for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
)


def validate_vector_families() -> None:
    widening_pairs = {
        (family.source.name, family.result.name)
        for family in X86_VECTOR_INTEGER_WIDENING_FAMILIES
    }
    if widening_pairs != {
        (source.name, result.name)
        for source_index, source in enumerate(INTEGER_ELEMENTS[:-1])
        for result in INTEGER_ELEMENTS[source_index + 1 :]
    }:
        raise ValueError(
            "x86 integer widening rows must cover every widening element pair"
        )
    widening_mnemonics = {
        mnemonic
        for family in X86_VECTOR_INTEGER_WIDENING_FAMILIES
        for mnemonic in (family.signed_mnemonic, family.unsigned_mnemonic)
    }
    if len(widening_mnemonics) != 2 * len(X86_VECTOR_INTEGER_WIDENING_FAMILIES):
        raise ValueError("x86 integer widening rows must use unique mnemonics")
    keys = [
        (family.source_operation, family.element.name)
        for family in (*AVX2_INTEGER_BINARY_FAMILIES, *AVX2_FLOAT_BINARY_FAMILIES)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate AVX2 source-operation and element family")
    if set(AVX2_INTEGER_COMPARE_MNEMONICS) != {
        element.name for element in INTEGER_ELEMENTS
    }:
        raise ValueError(
            "AVX2 integer comparison rows must cover every integer element"
        )
    if set(AVX2_FLOAT_COMPARE_MNEMONICS) != {
        element.name for element in FLOAT_ELEMENTS
    }:
        raise ValueError("AVX2 floating comparison rows must cover every float element")
    if set(FLOAT_EXTREMA_MNEMONICS) != set(FLOAT_EXTREMA_OPERATIONS):
        raise ValueError("packed extrema rows must cover every float semantic")
    if set(AVX2_SCALAR_FLOAT_EXTREMA_MNEMONICS) != set(FLOAT_EXTREMA_OPERATIONS):
        raise ValueError("AVX2 scalar extrema rows must cover every float semantic")
    avx512_keys = [
        (family.source_operation, family.element.name)
        for family in AVX512_INTEGER_BINARY_FAMILIES
    ]
    if len(avx512_keys) != len(set(avx512_keys)):
        raise ValueError("duplicate AVX-512 source-operation and element family")
    if {
        (family.source_operation, family.element.name)
        for family in AVX512VL_INTEGER_BINARY_FAMILIES
    } != set(avx512_keys) - _AVX2_INTEGER_BINARY_KEYS:
        raise ValueError("AVX-512VL rows must contain exactly the non-AVX2 cells")
    if set(AVX512_INTEGER_COMPARE_MNEMONICS) != {
        element.name for element in INTEGER_ELEMENTS
    }:
        raise ValueError(
            "AVX-512 integer comparison rows must cover every integer element"
        )
    if set(AVX512_SELECT_MNEMONICS) != set(AVX2_PAYLOAD_ELEMENT_NAMES):
        raise ValueError("AVX-512 select rows must cover every payload element")
    fp16_binary_operations = {
        family.source_operation for family in AVX512_FP16_FLOAT_BINARY_FAMILIES
    }
    if fp16_binary_operations != {"addf", "subf", "mulf", "divf"}:
        raise ValueError("AVX-512 FP16 binary rows must cover the arithmetic family")
    if {
        family.source_operation for family in AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES
    } != fp16_binary_operations:
        raise ValueError("AVX-512 FP16 scalar and packed arithmetic must agree")
    avx2_root_cells = {
        (family.source_operation, family.element.name, family.approximate)
        for family in AVX2_FLOAT_ROOT_FAMILIES
    }
    if avx2_root_cells != {
        ("sqrtf", "f32", False),
        ("sqrtf", "f64", False),
        ("rsqrtf", "f32", True),
    }:
        raise ValueError("AVX2 root rows must cover exact roots and native rsqrt")
    if {
        (family.source_operation, family.element.name, family.approximate)
        for family in AVX512_FLOAT_ROOT_FAMILIES
    } != avx2_root_cells | {("rsqrtf", "f64", True)}:
        raise ValueError("AVX-512 root rows must complete native F32/F64 roots")
    if {
        (family.source_operation, family.approximate)
        for family in AVX512_FP16_FLOAT_ROOT_FAMILIES
    } != {("sqrtf", False), ("rsqrtf", True)}:
        raise ValueError("AVX-512 FP16 rows must cover exact sqrt and native rsqrt")
    if {
        (family.source_operation, family.element.name, family.approximate)
        for family in AVX2_SCALAR_FLOAT_ROOT_FAMILIES
    } != avx2_root_cells:
        raise ValueError("AVX2 scalar and packed root rows must agree")
    if {
        (family.source_operation, family.element.name, family.approximate)
        for family in AVX512_SCALAR_FLOAT_ROOT_FAMILIES
    } != {
        ("rsqrtf", "f32", True),
        ("rsqrtf", "f64", True),
    }:
        raise ValueError("AVX-512 scalar root rows must cover VRSQRT14")
    if {
        (family.source_operation, family.approximate)
        for family in AVX512_FP16_SCALAR_FLOAT_ROOT_FAMILIES
    } != {("sqrtf", False), ("rsqrtf", True)}:
        raise ValueError("AVX-512 FP16 scalar and packed root rows must agree")
    if set(AVX512_FP16_FLOAT_EXTREMA_MNEMONICS) != set(FLOAT_EXTREMA_OPERATIONS) or set(
        AVX512_FP16_SCALAR_FLOAT_EXTREMA_MNEMONICS
    ) != set(FLOAT_EXTREMA_OPERATIONS):
        raise ValueError("AVX-512 FP16 extrema rows must cover every float semantic")


validate_vector_families()
