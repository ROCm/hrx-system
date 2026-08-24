# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    EQUALS,
    GLUE,
    LBRACE,
    LBRACKET,
    RBRACE,
    RBRACKET,
    AlignedRefs,
    Attr,
    AttrDict,
    Clause,
    Ref,
    Refs,
    ResultType,
    ResultTypeList,
    TemplateParam,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.memory import MemorySpace
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    BUFFER,
    BYTE_PATTERN_SCALAR,
    FACT_IDENTITY,
    I32,
    OFFSET,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    SAFE_TO_SPECULATE,
    VIEW,
    AttrDef,
    Dialect,
    LegacyFormat,
    Op,
    Operand,
    OpPhase,
    Reads,
    Result,
    SameType,
    VariadicValuesMatch,
    Writes,
)

# ============================================================================
# Group
# ============================================================================

buffer_ops = Dialect(
    "buffer",
    dialect_id=0x0C,
    doc="Opaque storage roots and typed view construction.",
    default_phase=OpPhase.EXECUTABLE,
)

# ============================================================================
# buffer.alloca — fixed-frame scratch allocation root
# ============================================================================

buffer_alloca = Op(
    name="buffer.alloca",
    group=buffer_ops,
    doc=(
        "Create a fixed-frame scratch buffer root in an allocatable memory space. "
        "Each execution produces a distinct storage identity; identical allocas "
        "must not be commoned. The byte length is the requested physical byte "
        "count for the execution. Targets requiring a static frame reserve its "
        "proven finite non-negative maximum. base_alignment is the minimum byte "
        "alignment of the root storage base. Target lowering determines which "
        "allocatable spaces are legal for the containing program kind."
    ),
    operands=[
        Operand(
            "byte_length",
            OFFSET,
            doc="Requested physical byte length of the scratch root.",
        ),
    ],
    results=[
        Result(
            "result",
            BUFFER,
            doc="Fresh opaque scratch storage root.",
            allocates=True,
        ),
    ],
    attrs=[
        AttrDef(
            "base_alignment",
            ATTR_TYPE_I64,
            doc="Minimum byte alignment of the root storage base.",
        ),
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=MemorySpace,
            doc="Allocatable scratch memory space for the root.",
        ),
    ],
    verify="loom_buffer_alloca_verify",
    facts="loom_buffer_alloca_facts",
    format=[
        TemplateParam("memory_space"),
        Clause("align", Attr("base_alignment")),
        Ref("byte_length"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%scratch = buffer.alloca<workgroup> align(64) %bytes : buffer",
    ],
)

# ============================================================================
# buffer.pack — lay out simultaneously live byte ranges in one slab
# ============================================================================

buffer_pack = Op(
    name="buffer.pack",
    group=buffer_ops,
    doc=(
        "Lay out simultaneously live physical byte ranges in one dense slab. "
        "Ranges retain operand order, begin at offsets satisfying their "
        "minimum alignments, and never alias. The total byte length is rounded "
        "up to the greatest range alignment so the result can be used as a "
        "repeatable record stride. Byte lengths may remain dynamic through "
        "specialization. Allocation lifetime packing is a separate compiler "
        "responsibility and must not be encoded with this operation."
    ),
    operands=[
        Operand(
            "byte_lengths",
            OFFSET,
            doc="Physical byte length of each range in slab order.",
            variadic=True,
        ),
    ],
    results=[
        Result(
            "total_byte_length",
            OFFSET,
            doc="Aligned physical byte length of the complete slab.",
        ),
        Result(
            "byte_offsets",
            OFFSET,
            doc="Slab-relative byte offset of each range.",
            variadic=True,
        ),
    ],
    attrs=[
        AttrDef(
            "minimum_alignments",
            ATTR_TYPE_I64_ARRAY,
            doc="Positive power-of-two minimum byte alignment for each range.",
        ),
    ],
    constraints=[
        VariadicValuesMatch("byte_lengths", "byte_offsets"),
        SameType("total_byte_length", "byte_offsets"),
    ],
    traits=[PURE],
    verify="loom_buffer_pack_verify",
    facts="loom_buffer_pack_facts",
    format=[
        AlignedRefs("byte_lengths", "minimum_alignments"),
        COLON,
        ResultTypeList("total_byte_length", parens=False, uniform=True),
    ],
    examples=[
        ("%total, %header_offset, %payload_offset = buffer.pack [align(16) %header_bytes, align(256) %payload_bytes] : offset"),
    ],
)

# ============================================================================
# buffer.assume.alignment — refine buffer root alignment facts
# ============================================================================

buffer_assume_alignment = Op(
    name="buffer.assume.alignment",
    group=buffer_ops,
    doc=(
        "Refine existing buffer roots with an explicit minimum byte alignment "
        "contract. The result preserves the same storage identity, extent, "
        "memory-space, alias, and nullability facts while strengthening the "
        "root base alignment fact."
    ),
    operands=[Operand("buffers", BUFFER, doc="Buffer roots to refine.", variadic=True)],
    results=[
        Result(
            "results",
            BUFFER,
            doc="Same buffer roots with refined alignment.",
            variadic=True,
        )
    ],
    attrs=[
        AttrDef(
            "minimum_alignment",
            ATTR_TYPE_I64,
            doc="Minimum byte alignment guaranteed for each root storage base.",
        ),
    ],
    constraints=[VariadicValuesMatch("buffers", "results")],
    traits=[PURE, FACT_IDENTITY],
    verify="loom_buffer_assume_alignment_verify",
    facts="loom_buffer_assume_alignment_facts",
    format=[
        Refs("buffers"),
        AttrDict(),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%aligned = buffer.assume.alignment %buffer {minimum_alignment = 16} : buffer",
        ("%lhs_a, %rhs_a = buffer.assume.alignment %lhs, %rhs {minimum_alignment = 16} : buffer, buffer"),
    ],
)

# ============================================================================
# buffer.assume.memory_space — refine a buffer root memory-space fact
# ============================================================================

buffer_assume_memory_space = Op(
    name="buffer.assume.memory_space",
    group=buffer_ops,
    doc=("Refine an existing buffer root with a concrete target-independent memory-space fact while preserving the same storage identity, extent, alignment, and nullability facts."),
    operands=[Operand("buffer", BUFFER, doc="Buffer root to refine.")],
    results=[
        Result("result", BUFFER, doc="Same buffer root with refined memory-space facts."),
    ],
    attrs=[
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=MemorySpace,
            doc="Concrete memory space to assume.",
        ),
    ],
    traits=[PURE, FACT_IDENTITY],
    verify="loom_buffer_assume_memory_space_verify",
    facts="loom_buffer_assume_memory_space_facts",
    format=[
        TemplateParam("memory_space"),
        Ref("buffer"),
        COLON,
        TypeOf("result"),
    ],
    legacy_formats=[
        LegacyFormat(
            "buffer.assume.memory_space.attr_dict",
            format=[
                Ref("buffer"),
                LBRACE,
                kw("memory_space"),
                EQUALS,
                Attr("memory_space"),
                RBRACE,
                COLON,
                TypeOf("result"),
            ],
            replaced_by="loom-source-format-2026-06-09",
        )
    ],
    examples=[
        "%global = buffer.assume.memory_space<global> %buffer : buffer",
    ],
)

# ============================================================================
# buffer.assume.noalias — refine a buffer root with a comparable noalias scope
# ============================================================================

buffer_assume_noalias = Op(
    name="buffer.assume.noalias",
    group=buffer_ops,
    doc=(
        "Refine an existing buffer root with an explicit noalias contract. "
        "The result preserves the same storage identity, extent, memory-space, "
        "alignment, and nullability facts, and marks the root identity as "
        "comparable for disjointness proofs. External buffer arguments do not "
        "gain this proof by default."
    ),
    operands=[Operand("buffers", BUFFER, doc="Buffer roots to refine.", variadic=True)],
    results=[
        Result(
            "results",
            BUFFER,
            doc="Same buffer roots with noalias scopes.",
            variadic=True,
        )
    ],
    constraints=[VariadicValuesMatch("buffers", "results")],
    traits=[PURE, FACT_IDENTITY],
    facts="loom_buffer_assume_noalias_facts",
    format=[
        Refs("buffers"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%unique = buffer.assume.noalias %buffer : buffer",
        ("%lhs_unique, %rhs_unique = buffer.assume.noalias %lhs, %rhs : buffer, buffer"),
    ],
)

# ============================================================================
# buffer.assume.same_root — refine a buffer root to share another root identity
# ============================================================================

buffer_assume_same_root = Op(
    name="buffer.assume.same_root",
    group=buffer_ops,
    doc=(
        "Refine an existing buffer root to share another buffer's storage root. "
        "This is a dominance-scoped assertion for internally specialized "
        "dispatches that know two incoming handles refer to the same allocation. "
        "The result keeps the first operand's value while inheriting the second "
        "operand's root identity and comparable alias scope."
    ),
    operands=[
        Operand("buffer", BUFFER, doc="Buffer value to refine."),
        Operand("root", BUFFER, doc="Buffer whose storage root is shared."),
    ],
    results=[Result("result", BUFFER, doc="Same buffer value with refined root identity.")],
    traits=[PURE],
    facts="loom_buffer_assume_same_root_facts",
    format=[
        Ref("buffer"),
        COMMA,
        Ref("root"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%same = buffer.assume.same_root %buffer, %root : buffer",
    ],
)

# ============================================================================
# buffer.view — form a typed view from a buffer root
# ============================================================================

buffer_view = Op(
    name="buffer.view",
    group=buffer_ops,
    doc=("Form a typed non-owning view from an opaque buffer root and base byte offset. The result view type carries the address layout."),
    operands=[
        Operand("buffer", BUFFER, doc="Opaque storage root."),
        Operand("byte_offset", OFFSET, doc="Base byte offset from the buffer root."),
    ],
    results=[Result("result", VIEW, doc="Typed logical view over the buffer.")],
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    verify="loom_buffer_view_verify",
    facts="loom_buffer_view_facts",
    format=[
        Ref("buffer"),
        GLUE,
        LBRACKET,
        Ref("byte_offset"),
        RBRACKET,
        COLON,
        TypeOf("buffer"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%view = buffer.view %buffer[%offset] : buffer -> view<[%M]xf32, %layout>",
    ],
)

# ============================================================================
# buffer.length — query physical byte length
# ============================================================================

buffer_length = Op(
    name="buffer.length",
    group=buffer_ops,
    doc=("Query the physical byte length of a buffer root without accessing its payload. Returns zero when the buffer is null."),
    operands=[Operand("buffer", BUFFER, doc="Opaque storage root, which may be null.")],
    results=[
        Result(
            "byte_length",
            OFFSET,
            doc="Physical byte length of the complete buffer root.",
        ),
    ],
    traits=[PURE, SAFE_TO_SPECULATE],
    facts="loom_buffer_length_facts",
    format=[Ref("buffer")],
    examples=[
        "%byte_length = buffer.length %buffer",
    ],
)

# ============================================================================
# buffer.load.i8.u/buffer.store.i8 — canonical raw byte access
# ============================================================================

buffer_load_i8_u = Op(
    name="buffer.load.i8.u",
    group=buffer_ops,
    builder_name="load_i8_u",
    doc=("Load one unsigned byte from a buffer root and zero-extend it to the canonical i32 carrier. The byte offset must identify an accessible byte in the buffer."),
    operands=[
        Operand("source", BUFFER, doc="Buffer root read by the load."),
        Operand("byte_offset", OFFSET, doc="Byte offset into the source buffer."),
    ],
    results=[
        Result("result", I32, doc="Loaded unsigned byte zero-extended to i32."),
    ],
    effects=[Reads("source")],
    facts="loom_buffer_load_i8_u_facts",
    format=[
        Ref("source"),
        GLUE,
        LBRACKET,
        Ref("byte_offset"),
        RBRACKET,
    ],
    examples=[
        "%byte = buffer.load.i8.u %source[%byte_offset]",
    ],
)

buffer_store_i8 = Op(
    name="buffer.store.i8",
    group=buffer_ops,
    builder_name="store_i8",
    doc=("Store the low eight bits of an i32 carrier to one byte in a buffer root. The byte offset must identify an accessible byte in the buffer."),
    operands=[
        Operand("value", I32, doc="i32 carrier whose low eight bits are stored."),
        Operand("target", BUFFER, doc="Buffer root written by the store."),
        Operand("byte_offset", OFFSET, doc="Byte offset into the target buffer."),
    ],
    effects=[Writes("target")],
    format=[
        Ref("value"),
        COMMA,
        Ref("target"),
        GLUE,
        LBRACKET,
        Ref("byte_offset"),
        RBRACKET,
    ],
    examples=[
        "buffer.store.i8 %byte, %target[%byte_offset]",
    ],
)

# ============================================================================
# buffer.copy — copy a non-overlapping byte range
# ============================================================================

buffer_copy = Op(
    name="buffer.copy",
    group=buffer_ops,
    doc=(
        "Copy an exact non-overlapping byte range between buffer roots. "
        "Source and target ranges must not overlap; programs may not depend "
        "on an overlap-safe target implementation. A zero byte length "
        "performs no byte access."
    ),
    operands=[
        Operand("source", BUFFER, doc="Buffer root read by the copy."),
        Operand("source_offset", OFFSET, doc="Source byte offset."),
        Operand("target", BUFFER, doc="Buffer root written by the copy."),
        Operand("target_offset", OFFSET, doc="Target byte offset."),
        Operand("byte_length", OFFSET, doc="Number of bytes to copy."),
    ],
    effects=[Reads("source"), Writes("target")],
    format=[
        Ref("source"),
        GLUE,
        LBRACKET,
        Ref("source_offset"),
        RBRACKET,
        COMMA,
        Ref("target"),
        GLUE,
        LBRACKET,
        Ref("target_offset"),
        RBRACKET,
        COMMA,
        Ref("byte_length"),
    ],
    examples=[
        "buffer.copy %source[%source_offset], %target[%target_offset], %byte_length",
    ],
)

# ============================================================================
# buffer.fill — repeat a raw scalar byte pattern
# ============================================================================

buffer_fill = Op(
    name="buffer.fill",
    group=buffer_ops,
    doc=(
        "Repeat the raw little-endian bytes of an 8-, 16-, 32-, or 64-bit "
        "integer or floating-point scalar across an exact writable byte "
        "range. A final partial repetition writes the low-address prefix of "
        "the pattern bytes. A zero byte length performs no byte access."
    ),
    operands=[
        Operand(
            "pattern",
            BYTE_PATTERN_SCALAR,
            doc="Raw scalar bit pattern to repeat.",
        ),
        Operand("target", BUFFER, doc="Buffer root receiving the pattern."),
        Operand("target_offset", OFFSET, doc="Target byte offset."),
        Operand("byte_length", OFFSET, doc="Number of bytes to fill."),
    ],
    effects=[Writes("target")],
    format=[
        Ref("pattern"),
        COMMA,
        Ref("target"),
        GLUE,
        LBRACKET,
        Ref("target_offset"),
        RBRACKET,
        COMMA,
        Ref("byte_length"),
        COLON,
        TypeOf("pattern"),
    ],
    examples=[
        "buffer.fill %pattern, %target[%target_offset], %byte_length : bf16",
    ],
)

# ============================================================================
# buffer.compare — lexicographically compare byte ranges
# ============================================================================

buffer_compare = Op(
    name="buffer.compare",
    group=buffer_ops,
    doc=("Lexicographically compare equal-length ranges as unsigned bytes and return canonical i32 -1, 0, or +1. A zero byte length returns zero and performs no byte access."),
    operands=[
        Operand("lhs", BUFFER, doc="Left buffer root."),
        Operand("lhs_offset", OFFSET, doc="Left byte offset."),
        Operand("rhs", BUFFER, doc="Right buffer root."),
        Operand("rhs_offset", OFFSET, doc="Right byte offset."),
        Operand("byte_length", OFFSET, doc="Number of bytes to compare."),
    ],
    results=[
        Result("order", I32, doc="Canonical signed ordering result."),
    ],
    effects=[Reads("lhs"), Reads("rhs")],
    facts="loom_buffer_compare_facts",
    format=[
        Ref("lhs"),
        GLUE,
        LBRACKET,
        Ref("lhs_offset"),
        RBRACKET,
        COMMA,
        Ref("rhs"),
        GLUE,
        LBRACKET,
        Ref("rhs_offset"),
        RBRACKET,
        COMMA,
        Ref("byte_length"),
    ],
    examples=[
        "%order = buffer.compare %lhs[%lhs_offset], %rhs[%rhs_offset], %byte_length",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_BUFFER_OPS: tuple[Op, ...] = (
    buffer_alloca,
    buffer_assume_alignment,
    buffer_assume_memory_space,
    buffer_assume_noalias,
    buffer_assume_same_root,
    buffer_view,
    buffer_pack,
    buffer_length,
    buffer_load_i8_u,
    buffer_store_i8,
    buffer_copy,
    buffer_fill,
    buffer_compare,
)
