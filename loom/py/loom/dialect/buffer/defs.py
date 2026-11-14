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
    OptionalGroup,
    Ref,
    Refs,
    ResultType,
    ResultTypeList,
    TemplateParam,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.atomic import AtomicFenceOrdering, AtomicScope
from loom.dialect.memory import MemorySpace
from loom.dsl import (
    ANY,
    ATTR_TYPE_BYTES,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    BUFFER,
    BYTE_PATTERN_SCALAR,
    CONSTANT_LIKE,
    FACT_IDENTITY,
    I1,
    I32,
    INTEGER,
    MEMORY_FENCE,
    OFFSET,
    POOL,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    SAFE_TO_SPECULATE,
    VIEW,
    AttrDef,
    Borrow,
    BorrowedResult,
    CachePolicyInterface,
    Dialect,
    EnumCase,
    EnumDef,
    LegacyFormat,
    MemoryAccessInterface,
    Op,
    Operand,
    OperandRole,
    OpPhase,
    Reads,
    Result,
    SameType,
    VariadicValuesMatch,
    Writes,
    value_duplication_op,
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
# buffer.constant — anonymous immutable executable storage
# ============================================================================

buffer_constant = Op(
    name="buffer.constant",
    group=buffer_ops,
    doc=(
        "Borrow anonymous immutable executable storage containing the exact "
        "payload bytes. Evaluation performs no per-invocation allocation or "
        "initialization. Equal contents may share storage, with no distinct "
        "address identity guarantee. The byte extent excludes alignment padding; "
        "UTF-8 adds no terminator or normalization. Empty payloads produce a "
        "nonnull zero-length buffer. Omitted alignment means one byte. Named "
        "objects with identity or linkage requirements use global.rodata.def."
    ),
    results=[Result("result", BUFFER, doc="Borrowed readonly storage root.")],
    attrs=[
        AttrDef("contents", ATTR_TYPE_BYTES, doc="Exact immutable payload bytes."),
        AttrDef(
            "alignment",
            ATTR_TYPE_I64,
            optional=True,
            doc="Minimum power-of-two byte alignment, defaulting to one.",
        ),
    ],
    traits=[PURE, CONSTANT_LIKE],
    eager_execution_cost=1,
    verify="loom_buffer_constant_verify",
    facts="loom_buffer_constant_facts",
    format=[
        OptionalGroup([Clause("align", Attr("alignment"))], anchor="alignment"),
        Attr("contents"),
        COLON,
        ResultType("result"),
    ],
    examples=['%message = buffer.constant utf8("ready") : buffer'],
)

buffer_null = Op(
    name="buffer.null",
    group=buffer_ops,
    doc=(
        "Produce the null buffer, which identifies no storage. The result has "
        "zero byte length, allocates nothing, and carries no release obligation. "
        "It may be selected, passed, returned, and tested for null, but cannot "
        "back a memory access. Null is distinct from every allocated buffer, "
        "including zero-length storage. Its physical representation is target "
        "and memory-space dependent; it is not an integer address."
    ),
    results=[Result("result", BUFFER, doc="Null storage reference.")],
    traits=[PURE, CONSTANT_LIKE, SAFE_TO_SPECULATE],
    facts="loom_buffer_null_facts",
    eager_execution_cost=1,
    format=[COLON, ResultType("result")],
    examples=["%empty = buffer.null : buffer"],
)

buffer_compare_null = Op(
    name="buffer.compare.null",
    group=buffer_ops,
    doc=("Test whether a buffer is null. Returns true exactly when the operand identifies no storage, independently of its byte length. The query does not access memory or change ownership."),
    operands=[Operand("buffer", BUFFER, doc="Storage reference to test.")],
    results=[Result("result", I1, doc="True for null, false for storage.")],
    traits=[PURE, SAFE_TO_SPECULATE],
    ownership_effects=[Borrow("buffer")],
    facts="loom_buffer_compare_null_facts",
    eager_execution_cost=1,
    format=[Ref("buffer"), COLON, TypeOf("buffer")],
    examples=["%missing = buffer.compare.null %input : buffer"],
)

BufferAddressPredicate = EnumDef(
    "BufferAddressPredicate",
    [
        EnumCase("eq", 0, doc="Equal positions."),
        EnumCase("ne", 1, doc="Unequal positions."),
    ],
    doc="Equality predicates for opaque buffer positions.",
)

buffer_compare_address = Op(
    name="buffer.compare.address",
    group=buffer_ops,
    doc=(
        "Compare two byte positions without exposing their addresses. Each "
        "nonnull buffer must be live and its offset must be in bounds or at "
        "the one-past endpoint. A null buffer permits only offset zero and "
        "differs from every nonnull buffer, including empty storage. Aliased "
        "buffers compare by their selected positions, not wrapper identity. "
        "An empty root without native storage retains a distinct logical "
        "position shared by its subspans. The operation reads no payload "
        "bytes and changes no ownership."
    ),
    operands=[
        Operand("lhs", BUFFER, doc="Left storage reference."),
        Operand("lhs_offset", OFFSET, doc="Left byte position."),
        Operand("rhs", BUFFER, doc="Right storage reference."),
        Operand("rhs_offset", OFFSET, doc="Right byte position."),
    ],
    results=[Result("result", I1, doc="Result of the selected predicate.")],
    attrs=[AttrDef("predicate", ATTR_TYPE_ENUM, enum_def=BufferAddressPredicate)],
    traits=[PURE],
    ownership_effects=[Borrow("lhs"), Borrow("rhs")],
    canonicalize="loom_buffer_compare_address_canonicalize",
    facts="loom_buffer_compare_address_facts",
    format=[
        Attr("predicate"),
        COMMA,
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
    ],
    examples=[
        "%done = buffer.compare.address eq, %cursor[%position], %end[%limit]",
        "%more = buffer.compare.address ne, %cursor[%position], %end[%limit]",
    ],
)

# ============================================================================
# buffer.distance — signed displacement within one buffer root
# ============================================================================

buffer_distance = Op(
    name="buffer.distance",
    group=buffer_ops,
    phase=OpPhase.SOURCE_STRUCTURE,
    doc=(
        "Return lhs_offset minus rhs_offset as a signed byte distance. Both "
        "positions must derive from the same buffer byte origin, proven before "
        "target lowering; otherwise compilation fails. Sharing an allocation "
        "without a common byte origin is insufficient. Nonnull positions must "
        "be live and in bounds or one-past. Two null positions at offset zero "
        "have distance zero. Offset values lie in [0, INT64_MAX], so their "
        "signed difference fits i64 and must also fit the signed result type. "
        "No address or payload bytes are observed."
    ),
    operands=[
        Operand("lhs", BUFFER, doc="Left storage root."),
        Operand("lhs_offset", OFFSET, doc="Left root-relative byte position."),
        Operand("rhs", BUFFER, doc="Right storage root."),
        Operand("rhs_offset", OFFSET, doc="Right root-relative byte position."),
    ],
    results=[Result("result", INTEGER, doc="Signed left-minus-right byte distance.")],
    traits=[PURE],
    ownership_effects=[Borrow("lhs"), Borrow("rhs")],
    canonicalize="loom_buffer_distance_canonicalize",
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
        COLON,
        TypeOf("result"),
    ],
    examples=["%bytes = buffer.distance %buffer[%end], %buffer[%begin] : i64"],
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
        "allocatable spaces are legal for the containing program kind. The "
        "execution frame owns the allocation; the result borrows that storage "
        "and has no explicit release obligation. An explicit pool selects the "
        "backing allocation resource; memory_space remains an access requirement "
        "that resource must satisfy. The pool is borrowed and does not extend "
        "the allocation lifetime. Omitting it uses the containing execution's "
        "allocation resource for that space, which must be unambiguous before "
        "materialization. Lowering cannot replace an explicit pool with an "
        "ambient allocation resource."
    ),
    operands=[
        Operand(
            "pool",
            POOL,
            optional=True,
            doc="Borrowed resource providing the allocation's backing storage.",
        ),
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
    ownership_effects=[Borrow("pool"), BorrowedResult("result")],
    facts="loom_buffer_alloca_facts",
    format=[
        TemplateParam("memory_space"),
        OptionalGroup([Clause("pool", Ref("pool"))], anchor="pool"),
        Clause("align", Attr("base_alignment")),
        Ref("byte_length"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%scratch = buffer.alloca<workgroup> align(64) %bytes : buffer",
        "%packets = buffer.alloca<workgroup> pool(%tile_memory) align(64) %bytes : buffer",
    ],
)

# ============================================================================
# buffer.alloca.init — literal-initialized fixed-frame storage
# ============================================================================

buffer_alloca_init = Op(
    name="buffer.alloca.init",
    group=buffer_ops,
    builder_name="alloca_init",
    doc=(
        "Create fresh fixed-frame storage initialized with the exact payload "
        "bytes. Extent is the payload length, without implicit terminators or "
        "alignment padding. Ownership, lifetime, alignment, and backing-pool "
        "selection follow buffer.alloca; equal initializers never merge writable "
        "allocations. Workgroup initialization is collective: every workitem "
        "encounters the same dynamic instance in workgroup-uniform control, "
        "and the initialized bytes are visible to the whole workgroup before "
        "participants proceed. Other allocatable spaces initialize each "
        "invocation's independent storage without collective synchronization."
    ),
    operands=[
        Operand("pool", POOL, optional=True, doc="Borrowed backing allocation resource."),
    ],
    results=[Result("result", BUFFER, allocates=True, doc="Fresh initialized storage root.")],
    attrs=[
        AttrDef("base_alignment", ATTR_TYPE_I64, doc="Minimum byte alignment of the storage base."),
        AttrDef("memory_space", ATTR_TYPE_ENUM, enum_def=MemorySpace, doc="Allocatable scratch memory space."),
        AttrDef("contents", ATTR_TYPE_BYTES, doc="Exact initial payload bytes."),
    ],
    ownership_effects=[Borrow("pool"), BorrowedResult("result")],
    effective_traits="loom_buffer_alloca_init_effective_traits",
    eager_execution_cost=0,
    verify="loom_buffer_alloca_init_verify",
    facts="loom_buffer_alloca_init_facts",
    format=[
        TemplateParam("memory_space"),
        OptionalGroup([Clause("pool", Ref("pool"))], anchor="pool"),
        Clause("align", Attr("base_alignment")),
        Attr("contents"),
        COLON,
        ResultType("result"),
    ],
    examples=['%scratch = buffer.alloca.init<private> align(1) utf8("ready") : buffer'],
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
        "specialization. The complete aligned extent must be representable "
        "in the selected target's offset domain. Allocation lifetime packing is a separate compiler "
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
    facts_from_operands=True,
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
        "gain this proof by default. The promise separates distinct scoped "
        "roots in the same execution. Repeated executions may bind a root to "
        "different storage; noalias alone does not establish disjointness "
        "across executions."
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
# buffer.restrict.begin/end — scoped source access independence
# ============================================================================

buffer_restrict_begin = Op(
    name="buffer.restrict.begin",
    group=buffer_ops,
    doc=(
        "Begin a source-language restrict access scope over buffer roots and "
        "typed views. Restricted references name independent access paths; "
        "peers name the other incoming storage paths covered by the same "
        "scope. The results preserve physical storage identity and differ only "
        "in access-provenance facts. A restrict scope may prove that "
        "overlapping accesses do not occur in a defined program only when at "
        "least one access writes. It does not prove that roots are physically "
        "disjoint, and shared read-only storage remains valid. "
        "buffer.restrict.end removes the scope from values that leave it."
    ),
    operands=[
        Operand(
            "restricted",
            ANY,
            doc=("Buffer or view references accessed through distinct restricted paths."),
            variadic=True,
        ),
        Operand(
            "peers",
            ANY,
            doc=("Other incoming buffer or view references covered by the same access scope."),
            variadic=True,
        ),
    ],
    results=[
        Result(
            "results",
            ANY,
            doc="Same references carrying the active access-scope association.",
            variadic=True,
        )
    ],
    traits=[PURE, FACT_IDENTITY],
    verify="loom_buffer_restrict_begin_verify",
    facts="loom_buffer_restrict_begin_facts",
    format=[
        Refs("restricted"),
        OptionalGroup([kw("peers"), Refs("peers")], anchor="peers"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%source_r = buffer.restrict.begin %source : buffer",
        ("%source_r, %output_r, %scratch_peer = buffer.restrict.begin %source, %output peers %scratch : buffer, view<16xf32>, view<16xf32>"),
    ],
)

buffer_restrict_end = Op(
    name="buffer.restrict.end",
    group=buffer_ops,
    doc=(
        "End source-language restrict access scopes carried by buffer roots or "
        "typed views. The results preserve every physical storage fact while "
        "removing the scoped access association, preventing it from escaping "
        "a lexical or invocation boundary."
    ),
    operands=[
        Operand(
            "references",
            ANY,
            doc="Buffer or view references leaving their access scopes.",
            variadic=True,
        )
    ],
    results=[
        Result(
            "results",
            ANY,
            doc="Same references without restrict access associations.",
            variadic=True,
        )
    ],
    constraints=[VariadicValuesMatch("references", "results")],
    traits=[PURE, FACT_IDENTITY],
    facts="loom_buffer_restrict_end_facts",
    verify="loom_buffer_restrict_end_verify",
    format=[Refs("references"), COLON, TypesOf("results")],
    examples=[
        "%returned = buffer.restrict.end %scoped : buffer",
        "%lhs_out, %rhs_out = buffer.restrict.end %lhs, %rhs : buffer, view<16xf32>",
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
        Operand("buffer", BUFFER, doc="Opaque storage root.", role=OperandRole.ACCESS_PATH),
        Operand(
            "byte_offset",
            OFFSET,
            doc="Base byte offset from the buffer root.",
            role=OperandRole.ACCESS_PATH,
        ),
    ],
    results=[Result("result", VIEW, doc="Typed logical view over the buffer.", reference_source="buffer")],
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
    operands=[Operand("buffer", BUFFER, doc="Opaque storage root, which may be null.", observes_reference=True)],
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
    eager_execution_cost=1,
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
        Operand(
            "source",
            BUFFER,
            doc="Buffer root read by the load.",
            role=OperandRole.ACCESS_PATH,
        ),
        Operand(
            "byte_offset",
            OFFSET,
            doc="Byte offset into the source buffer.",
            role=OperandRole.ACCESS_PATH,
        ),
    ],
    results=[
        Result("result", I32, doc="Loaded unsigned byte zero-extended to i32."),
    ],
    effects=[Reads("source")],
    interfaces=[CachePolicyInterface(None, None), MemoryAccessInterface(view="source")],
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
        Operand(
            "target",
            BUFFER,
            doc="Buffer root written by the store.",
            role=OperandRole.ACCESS_PATH,
        ),
        Operand(
            "byte_offset",
            OFFSET,
            doc="Byte offset into the target buffer.",
            role=OperandRole.ACCESS_PATH,
        ),
    ],
    effects=[Writes("target")],
    interfaces=[CachePolicyInterface(None, None), MemoryAccessInterface(view="target")],
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
        Operand("source", BUFFER, doc="Buffer root read by the copy.", role=OperandRole.ACCESS_PATH),
        Operand("source_offset", OFFSET, doc="Source byte offset.", role=OperandRole.ACCESS_PATH),
        Operand("target", BUFFER, doc="Buffer root written by the copy.", role=OperandRole.ACCESS_PATH),
        Operand("target_offset", OFFSET, doc="Target byte offset.", role=OperandRole.ACCESS_PATH),
        Operand("byte_length", OFFSET, doc="Number of bytes to copy.", role=OperandRole.ACCESS_PATH),
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
        Operand("target", BUFFER, doc="Buffer root receiving the pattern.", role=OperandRole.ACCESS_PATH),
        Operand("target_offset", OFFSET, doc="Target byte offset.", role=OperandRole.ACCESS_PATH),
        Operand("byte_length", OFFSET, doc="Number of bytes to fill.", role=OperandRole.ACCESS_PATH),
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
        Operand("lhs", BUFFER, doc="Left buffer root.", role=OperandRole.ACCESS_PATH),
        Operand("lhs_offset", OFFSET, doc="Left byte offset.", role=OperandRole.ACCESS_PATH),
        Operand("rhs", BUFFER, doc="Right buffer root.", role=OperandRole.ACCESS_PATH),
        Operand("rhs_offset", OFFSET, doc="Right byte offset.", role=OperandRole.ACCESS_PATH),
        Operand("byte_length", OFFSET, doc="Number of bytes to compare.", role=OperandRole.ACCESS_PATH),
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
# buffer.fence — thread memory ordering across storage roots
# ============================================================================

buffer_fence = Op(
    name="buffer.fence",
    group=buffer_ops,
    doc=(
        "Order the executing thread's memory accesses across storage roots "
        "using an explicit atomic ordering and synchronization scope. This "
        "applies to ordinary functions and kernels, including accesses through "
        "typed views and vectors. Acquire and release fences participate in "
        "synchronization through matching atomic observations and publications; "
        "they do not rendezvous with other invocations or complete independent "
        "asynchronous transfers. The backing memory and target must support "
        "the requested synchronization domain."
    ),
    attrs=[
        AttrDef("scope", ATTR_TYPE_ENUM, enum_def=AtomicScope, doc="Synchronization domain of the ordered memory effects."),
        AttrDef("ordering", ATTR_TYPE_ENUM, enum_def=AtomicFenceOrdering, doc="Acquire, release, acquire-release, or sequentially consistent ordering."),
    ],
    traits=[MEMORY_FENCE],
    format=[Clause("scope", Attr("scope")), Clause("ordering", Attr("ordering"))],
    examples=[
        "buffer.fence scope(system) ordering(acquire)",
        "buffer.fence scope(device) ordering(release)",
    ],
)

# ============================================================================
# Registry
# ============================================================================

buffer_duplicate = value_duplication_op(
    "buffer.duplicate",
    group=buffer_ops,
    carrier=BUFFER,
    facts="loom_buffer_duplicate_facts",
    doc=("Create another local owner of a buffer root. The duplicate identifies the same opaque storage and may lower to the same physical carrier, but remains a distinct SSA ownership identity."),
    examples=["%copy = buffer.duplicate %source : buffer"],
)

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
    buffer_fence,
    buffer_constant,
    buffer_alloca_init,
    buffer_duplicate,
    buffer_null,
    buffer_compare_null,
    buffer_compare_address,
    buffer_distance,
    buffer_restrict_begin,
    buffer_restrict_end,
)
