# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable persistent-pipeline type and operation definitions."""

from loom.assembly import (
    COLON,
    COMMA,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    FuncArgs,
    IndexList,
    OptionalGroup,
    PredicateList,
    Refs,
    Region,
    ResultType,
    Scope,
    SymbolRef,
    TemplateParam,
    TypesOf,
    kw,
)
from loom.dialect.memory import MemorySpace
from loom.dialect.symbol import Retain, Visibility
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    INDEX,
    ISOLATED_FROM_ABOVE,
    POOL,
    PURE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    AttrDef,
    CallLikeInterface,
    CallLikeKind,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    HasAncestor,
    HasParent,
    ImplicitTerminator,
    NoAncestor,
    Op,
    Operand,
    OpPhase,
    RegionDef,
    Result,
    SymbolDefinition,
    SymbolDefinitionFlag,
    SymbolReference,
    TypeDef,
)

pipeline_ops = Dialect(
    "pipeline",
    dialect_id=0x23,
    doc="Portable persistent dataflow programs over typed record streams.",
)

PipelineScope = EnumDef(
    "PipelineScope",
    [
        EnumCase(
            "kernel",
            1,
            doc="Must materialize as one directly loadable executable artifact.",
        ),
    ],
    doc=("Required materialization boundary. An absent scope permits a generic pipeline program that may span targets and runtime operations."),
)

StrandParticipation = EnumDef(
    "StrandParticipation",
    [
        EnumCase(
            "cooperative",
            1,
            doc="Selected workers cooperate in one strand instance and one ownership action per channel record.",
        ),
    ],
    doc="Channel participation within a strand. Absent participation creates an independent instance per selected worker.",
)

_PIPELINE_MODIFIER_FORMAT = [
    OptionalGroup([TemplateParam("scope")], anchor="scope"),
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    OptionalGroup([Attr("retain")], anchor="retain"),
    OptionalGroup(
        [kw("target"), GLUE, LPAREN, SymbolRef("target"), GLUE, RPAREN],
        anchor="target",
    ),
]

_PIPELINE_SIGNATURE_FORMAT = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs(
                "args",
                group="specializations",
                end_attr="specialization_count",
            ),
            kw("run"),
            FuncArgs(
                "args",
                group="bindings",
                start_attr="specialization_count",
            ),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_PIPELINE_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef("scope", ATTR_TYPE_ENUM, enum_def=PipelineScope, optional=True),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("retain", "enum", enum_def=Retain, optional=True),
    AttrDef(
        "target",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef("predicates", "predicate_list", optional=True),
    AttrDef(
        "specialization_count",
        ATTR_TYPE_I64,
        doc="Number of leading staged pipeline arguments.",
    ),
]

pipeline_def = Op(
    "pipeline.def",
    group=pipeline_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Persistent dataflow program. Leading specialization arguments remain "
        "ordinary SSA values and typed run arguments are supplied by each "
        "invocation. Their source types are independent of the eventual target "
        "export ABI. The optional scope fixes the artifact boundary that "
        "lowering must satisfy."
    ),
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE, ImplicitTerminator("pipeline.finish")],
    attrs=list(_PIPELINE_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="pipeline",
        interfaces=["func_like", "pipeline"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
        product_carrier="scope",
    ),
    regions=[
        RegionDef(
            "body",
            doc="Construction of invocation-owned storage and independently progressing strands.",
            terminator="pipeline.finish",
            buffer_arg_memory_space="global",
        )
    ],
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            visibility="visibility",
            target="target",
            predicates="predicates",
            specialization_count="specialization_count",
            body="body",
        )
    ],
    verify="loom_pipeline_signature_verify",
    format=[
        *_PIPELINE_MODIFIER_FORMAT,
        *_PIPELINE_SIGNATURE_FORMAT,
        Region("body"),
    ],
    examples=[
        "pipeline.def<kernel> target(@array) @resident() run(%input: buffer, %output: buffer) {\n}",
        "pipeline.def @heterogeneous(%batch: index) run(%input: buffer) {\n}",
    ],
)

_PIPELINE_GRAPH_TRAITS = [HasAncestor("pipeline.def")]

pipeline_memory = Op(
    "pipeline.memory",
    group=pipeline_ops,
    doc=(
        "Select a borrowed memory pool at worker coordinates in the enclosing "
        "pipeline invocation. Coordinates use the strand worker domain; "
        "the invocation supplies the execution instance, not the target symbol. "
        "Repeated selections of the same memory identify the same pool. The "
        "query neither allocates storage nor synchronizes access: buffer.alloca "
        "creates fresh allocation roots and channel operations govern record "
        "ownership. Each accessing worker must have a legal mapping to the "
        "selected backing. The pool remains valid for the invocation and may "
        "be passed to generic construction helpers. Selection belongs to "
        "construction outside strand bodies. Composed children borrow the "
        "enclosing invocation's coordinate space and backing pools; the "
        "materializing target verifies that the selected pool exists. A memory "
        "selection does not require the child to own an executable artifact."
    ),
    operands=[Operand("coordinates", INDEX, variadic=True)],
    results=[Result("result", POOL, doc="Borrowed invocation-local memory pool.")],
    attrs=[
        AttrDef("memory_space", ATTR_TYPE_ENUM, enum_def=MemorySpace),
        AttrDef(
            "static_coordinates",
            ATTR_TYPE_I64_ARRAY,
            doc="Worker coordinates; INT64_MIN entries refer to dynamic coordinates.",
        ),
    ],
    traits=[PURE, *_PIPELINE_GRAPH_TRAITS, NoAncestor("pipeline.strand")],
    verify="loom_pipeline_memory_verify",
    format=[
        TemplateParam("memory_space"),
        IndexList("coordinates", "static_coordinates"),
        COLON,
        ResultType("result"),
    ],
    examples=["%memory = pipeline.memory<workgroup>[%column, 3] : pool"],
)

pipeline_decl = Op(
    "pipeline.decl",
    group=pipeline_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Bodyless persistent pipeline contract. The leading specialization arguments, invocation arguments, target, predicates, and optional artifact scope have the same meaning as on pipeline.def."
    ),
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    attrs=list(_PIPELINE_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="pipeline",
        interfaces=["func_like", "pipeline"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
        product_carrier="scope",
        flags=[SymbolDefinitionFlag.DECLARATION],
    ),
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            visibility="visibility",
            target="target",
            predicates="predicates",
            specialization_count="specialization_count",
            args="args",
        )
    ],
    verify="loom_pipeline_signature_verify",
    format=[*_PIPELINE_MODIFIER_FORMAT, *_PIPELINE_SIGNATURE_FORMAT],
    examples=[
        "pipeline.decl<kernel> @resident(%width: index) run(%input: view<[%width]xi32>, %output: buffer)",
        "pipeline.decl @stage(%width: index) run(%input: channel<tile<[%width]xi32>>)",
    ],
)

pipeline_compose = Op(
    "pipeline.compose",
    group=pipeline_ops,
    doc=(
        "Compose a child pipeline into the enclosing invocation. Each composition "
        "constructs fresh child storage and independently progressing strands; "
        "it never submits another invocation. When the nearest schedule is "
        "pipeline.serial the child completes before that schedule advances. "
        "In a concurrent graph, composition adds no sibling dependency; the "
        "enclosing group owns its completion boundary. "
        "Specialization and run operands substitute the child's complete typed "
        "signature. Child target and materialization requirements remain binding."
    ),
    operands=[
        Operand("specializations", ANY, variadic=True),
        Operand("bindings", ANY, variadic=True),
    ],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("pipeline", ["pipeline"]),
        ),
    ],
    traits=[UNKNOWN_EFFECTS, *_PIPELINE_GRAPH_TRAITS],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="specializations",
            results=None,
            kind=CallLikeKind.COMPOSITION,
        ),
    ],
    verify="loom_pipeline_compose_verify",
    format=[
        SymbolRef("callee"),
        OptionalGroup(
            [GLUE, LBRACKET, Refs("specializations"), RBRACKET],
            anchor="specializations",
        ),
        GLUE,
        LPAREN,
        Refs("bindings"),
        RPAREN,
        COLON,
        OptionalGroup(
            [LBRACKET, TypesOf("specializations"), RBRACKET, GLUE],
            anchor="specializations",
        ),
        LPAREN,
        TypesOf("bindings"),
        RPAREN,
    ],
    examples=[
        "pipeline.compose @column[%column](%input, %output) : [index](channel<tile<16xi32>>, channel<tile<16xi32>>)",
    ],
)

pipeline_serial = Op(
    "pipeline.serial",
    group=pipeline_ops,
    doc=(
        "Order complete child pipelines within one resident invocation. Each "
        "composition in the region completes its strands and asynchronous "
        "memory operations before the schedule advances. Structured control "
        "repeats or selects child lifetimes; nested schedule regions complete "
        "before their enclosing schedule advances. Completed writes are visible "
        "to subsequent children. Outer storage and published channel records "
        "survive the boundary; completion neither empties outer channels nor "
        "orders unrelated concurrent strands. Child completion cannot retain "
        "uncompleted transfers or owned read/write capabilities. Strands belong "
        "to composed child definitions or concurrent regions."
    ),
    regions=[
        RegionDef(
            "body",
            doc="Resident child compositions and their structured schedule.",
            terminator="pipeline.yield",
        ),
    ],
    traits=[
        UNKNOWN_EFFECTS,
        *_PIPELINE_GRAPH_TRAITS,
        NoAncestor("pipeline.strand"),
        ImplicitTerminator("pipeline.yield"),
    ],
    format=[Region("body")],
    examples=["pipeline.serial {\n}"],
)

pipeline_concurrent = Op(
    "pipeline.concurrent",
    group=pipeline_ops,
    doc=(
        "Construct a concurrent child graph within one resident invocation. "
        "Sibling strands, compositions, and nested schedules progress without "
        "dependency edges between siblings. The complete group joins before "
        "its enclosing serial schedule advances. Local storage and channels "
        "belong to the group; outer storage and published records survive its "
        "completion. The single-block graph has lexical captures and no "
        "results. Runtime repetition and selection belong in nested serial "
        "branches; nesting serial and concurrent groups preserves each group's "
        "completion boundary."
    ),
    regions=[
        RegionDef(
            "body",
            doc="Concurrent child graph with independently progressing branches.",
            single_block=True,
            terminator="pipeline.yield",
        ),
    ],
    traits=[
        UNKNOWN_EFFECTS,
        *_PIPELINE_GRAPH_TRAITS,
        NoAncestor("pipeline.strand"),
        ImplicitTerminator("pipeline.yield"),
    ],
    format=[Region("body")],
    examples=["pipeline.concurrent {\n  pipeline.serial {\n  }\n}"],
)

pipeline_yield = Op(
    "pipeline.yield",
    group=pipeline_ops,
    doc="Complete a resident schedule region. Implicit at the end of its region.",
    traits=[TERMINATOR],
    verify="loom_pipeline_yield_verify",
    examples=["pipeline.yield"],
)

pipeline_strand = Op(
    "pipeline.strand",
    group=pipeline_ops,
    doc=(
        "Construct strand instances over a target-relative worker domain. By "
        "default each selected worker enters an independent instance once. A "
        "cooperative strand instead enters one instance with all selected workers: "
        "channel actions collectively acquire, publish or release one record, "
        "while ordinary payload operations execute per participant. Channel "
        "actions require converged participation, including through calls. "
        "Ordinary loops in the body express repeated work. The region captures lexical SSA "
        "values and declares work rather than executing inline. Multiple strands "
        "may share a worker, with channel dependencies controlling their progress. "
        "Origins, counts and strides select coordinates as origin + index * stride. "
        "An omitted target inherits the enclosing pipeline's target environment."
    ),
    operands=[
        Operand("origins", INDEX, variadic=True, doc="Dynamic worker origins."),
        Operand("counts", INDEX, variadic=True, doc="Dynamic worker counts."),
        Operand("strides", INDEX, variadic=True, doc="Dynamic worker strides."),
    ],
    attrs=[
        AttrDef(
            "target",
            "symbol",
            optional=True,
            symbol_ref=SymbolReference("target", ["target"]),
        ),
        AttrDef(
            "static_origins",
            ATTR_TYPE_I64_ARRAY,
            doc="Worker origins; INT64_MIN entries refer to dynamic origins.",
        ),
        AttrDef(
            "static_counts",
            ATTR_TYPE_I64_ARRAY,
            doc="Worker counts; INT64_MIN entries refer to dynamic counts.",
        ),
        AttrDef(
            "static_strides",
            ATTR_TYPE_I64_ARRAY,
            doc="Worker strides; INT64_MIN entries refer to dynamic strides.",
        ),
        AttrDef(
            "participation",
            ATTR_TYPE_ENUM,
            enum_def=StrandParticipation,
            optional=True,
        ),
    ],
    regions=[
        RegionDef(
            "body",
            doc="One complete strand instance, with lexical captures.",
            terminator="pipeline.end",
            execution_target="target",
        ),
    ],
    traits=[
        UNKNOWN_EFFECTS,
        *_PIPELINE_GRAPH_TRAITS,
        ImplicitTerminator("pipeline.end"),
    ],
    verify="loom_pipeline_strand_verify",
    format=[
        OptionalGroup([TemplateParam("participation")], anchor="participation"),
        OptionalGroup(
            [kw("target"), GLUE, LPAREN, SymbolRef("target"), GLUE, RPAREN],
            anchor="target",
        ),
        kw("workers"),
        GLUE,
        LPAREN,
        IndexList("origins", "static_origins"),
        COMMA,
        IndexList("counts", "static_counts", glue=False),
        COMMA,
        IndexList("strides", "static_strides", glue=False),
        GLUE,
        RPAREN,
        Region("body"),
    ],
    examples=[
        "pipeline.strand target(@npu) workers([%column, 2], [1, 1], [1, 1]) {\n}",
        "pipeline.strand<cooperative> workers([0], [4], [1]) {\n}",
    ],
)

pipeline_end = Op(
    "pipeline.end",
    group=pipeline_ops,
    doc="Complete a strand instance. Implicit at the end of its region.",
    traits=[TERMINATOR, HasParent("pipeline.strand")],
    examples=["pipeline.end"],
)

pipeline_finish = Op(
    "pipeline.finish",
    group=pipeline_ops,
    doc=("Complete the pipeline body. Invocation completion follows the execution and communication obligations of the materialized program."),
    traits=[TERMINATOR, HasParent("pipeline.def")],
    examples=["pipeline.finish"],
)

ALL_PIPELINE_TYPES: tuple[TypeDef, ...] = ()
ALL_PIPELINE_OPS: tuple[Op, ...] = (
    pipeline_def,
    pipeline_finish,
    pipeline_strand,
    pipeline_end,
    pipeline_compose,
    pipeline_memory,
    pipeline_decl,
    pipeline_serial,
    pipeline_yield,
    pipeline_concurrent,
)
