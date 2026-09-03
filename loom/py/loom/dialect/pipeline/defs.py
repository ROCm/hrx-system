# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable persistent-pipeline type and operation definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    FuncArgs,
    OptionalGroup,
    Param,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TemplateParam,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.func.defs import Retain, Visibility
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    INDEX,
    ISOLATED_FROM_ABOVE,
    PURE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    VIEW,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    HasAncestor,
    HasParent,
    Op,
    Operand,
    OpPhase,
    RegionDef,
    Result,
    SameType,
    SymbolDefinition,
    SymbolReference,
    TypeDef,
)

pipeline_ops = Dialect(
    "pipeline",
    dialect_id=0x23,
    doc="Portable persistent dataflow programs over typed record streams.",
)

pipeline_flow_type = TypeDef(
    "pipeline.flow",
    params=[AttrDef("element_type", "type")],
    format=[Param("element_type")],
    doc=("Typed ordered record stream between scheduling groups. The element type describes one transferred value and is normally tile<...>."),
)

PipelineScope = EnumDef(
    "PipelineScope",
    [
        EnumCase(
            "kernel",
            1,
            doc="Must materialize as one directly loadable executable artifact.",
        ),
        EnumCase(
            "command",
            2,
            doc="Must materialize as one reusable command program.",
        ),
    ],
    doc=("Required materialization boundary. An absent scope permits a generic pipeline program that may span targets and runtime operations."),
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
            kw("launch"),
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
        "ordinary SSA values and launch bindings are supplied when the "
        "materialized pipeline is issued. The optional scope fixes the artifact "
        "boundary that lowering must satisfy."
    ),
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_PIPELINE_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="pipeline",
        interfaces=["func_like", "pipeline"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
    ),
    regions=[
        RegionDef(
            "body",
            doc="Portable scheduling groups, flows, and stage graph.",
            terminator="pipeline.return",
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
    verify="loom_pipeline_def_verify",
    format=[
        *_PIPELINE_MODIFIER_FORMAT,
        *_PIPELINE_SIGNATURE_FORMAT,
        Region("body"),
    ],
    examples=[
        "pipeline.def<kernel> target(@array) @resident() launch(%input: buffer, %output: buffer) {\n  pipeline.return\n}",
        "pipeline.def @heterogeneous(%batch: index) launch(%input: buffer) {\n  pipeline.return\n}",
    ],
)

_PIPELINE_GRAPH_TRAITS = [HasAncestor("pipeline.def")]

pipeline_scatter = Op(
    "pipeline.scatter",
    group=pipeline_ops,
    doc=("Partition the leading dimension of a source view across a scheduling group, producing one suffix-shaped tile record per lane."),
    operands=[
        Operand("source", VIEW, doc="Dense source view with a lane dimension."),
        Operand("group", ANY, doc="Destination scheduling group."),
    ],
    results=[Result("result", ANY, doc="Per-lane tile flow.")],
    traits=[PURE, *_PIPELINE_GRAPH_TRAITS],
    verify="loom_pipeline_scatter_verify",
    format=[
        Ref("source"),
        kw("across"),
        Ref("group"),
        COLON,
        TypeOf("source"),
        COMMA,
        TypeOf("group"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%tiles = pipeline.scatter %source across %workers : view<2x8x8xi8>, group -> pipeline.flow<tile<8x8xi8>>"],
)

pipeline_read = Op(
    "pipeline.read",
    group=pipeline_ops,
    doc="Read one complete source-view tile record for each destination lane.",
    operands=[
        Operand("source", VIEW, doc="Source view containing one tile record."),
        Operand("group", ANY, doc="Destination scheduling group."),
    ],
    results=[Result("result", ANY, doc="Tile flow delivered to the group.")],
    traits=[PURE, *_PIPELINE_GRAPH_TRAITS],
    verify="loom_pipeline_read_verify",
    format=[
        Ref("source"),
        kw("on"),
        Ref("group"),
        COLON,
        TypeOf("source"),
        COMMA,
        TypeOf("group"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%bias = pipeline.read %source on %reducers : view<8x8xi32>, group -> pipeline.flow<tile<8x8xi32>>"],
)

pipeline_stage = Op(
    "pipeline.stage",
    group=pipeline_ops,
    doc=("Instantiate one stage invocation per group lane. Inputs and outputs are lane-wise typed flows; the referenced callable supplies one record-firing implementation."),
    operands=[
        Operand("group", ANY, doc="Scheduling group executing the stage."),
        Operand("inputs", ANY, variadic=True, doc="Lane-wise input flows."),
    ],
    results=[Result("outputs", ANY, variadic=True, doc="Lane-wise output flows.")],
    attrs=[
        AttrDef(
            "entry",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        )
    ],
    traits=[UNKNOWN_EFFECTS, *_PIPELINE_GRAPH_TRAITS],
    verify="loom_pipeline_stage_verify",
    format=[
        SymbolRef("entry"),
        kw("on"),
        Ref("group"),
        GLUE,
        LPAREN,
        Refs("inputs"),
        RPAREN,
        COLON,
        LPAREN,
        TypeOf("group"),
        OptionalGroup([COMMA, TypesOf("inputs")], anchor="inputs"),
        RPAREN,
        ARROW,
        ResultTypeList("outputs"),
    ],
    examples=["%partials = pipeline.stage @product on %workers(%lhs, %rhs) : (group, pipeline.flow<tile<8x8xi8>>, pipeline.flow<tile<8x8xi8>>) -> (pipeline.flow<tile<8x8xi32>>)"],
)

pipeline_buffer = Op(
    "pipeline.buffer",
    group=pipeline_ops,
    doc=("Require an independently buffered flow with SSA-defined minimum record capacity. Targets may select a greater capacity."),
    operands=[
        Operand("source", ANY, doc="Input tile flow."),
        Operand("capacity", INDEX, doc="Required minimum record capacity."),
    ],
    results=[
        Result(
            "result",
            ANY,
            allocates=True,
            doc="Distinct buffered tile flow.",
        )
    ],
    constraints=[SameType("source", "result")],
    traits=[*_PIPELINE_GRAPH_TRAITS],
    verify="loom_pipeline_buffer_verify",
    format=[
        Ref("source"),
        kw("capacity"),
        Ref("capacity"),
        COLON,
        LPAREN,
        TypeOf("source"),
        COMMA,
        TypeOf("capacity"),
        RPAREN,
        ARROW,
        ResultType("result"),
    ],
    examples=["%buffered = pipeline.buffer %partials capacity %ring_capacity : (pipeline.flow<tile<8x8xi32>>, index) -> pipeline.flow<tile<8x8xi32>>"],
)

pipeline_reduce = Op(
    "pipeline.reduce",
    group=pipeline_ops,
    doc=("Gather each source-group lane record into one target-group stage firing. Target inputs remain pointwise with the target group."),
    operands=[
        Operand("source_group", ANY, doc="Group whose lane records are gathered."),
        Operand("source_inputs", ANY, variadic=True, doc="Gathered source-group flows."),
        Operand("target_group", ANY, doc="Group executing the reduction stage."),
        Operand("target_inputs", ANY, variadic=True, doc="Pointwise target-group flows."),
    ],
    results=[Result("outputs", ANY, variadic=True, doc="Target-group output flows.")],
    attrs=[
        AttrDef(
            "entry",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        )
    ],
    traits=[UNKNOWN_EFFECTS, *_PIPELINE_GRAPH_TRAITS],
    verify="loom_pipeline_reduce_verify",
    format=[
        SymbolRef("entry"),
        kw("from"),
        Ref("source_group"),
        GLUE,
        LPAREN,
        Refs("source_inputs"),
        RPAREN,
        kw("to"),
        Ref("target_group"),
        GLUE,
        LPAREN,
        Refs("target_inputs"),
        RPAREN,
        COLON,
        LPAREN,
        TypeOf("source_group"),
        OptionalGroup([COMMA, TypesOf("source_inputs")], anchor="source_inputs"),
        RPAREN,
        kw("to"),
        LPAREN,
        TypeOf("target_group"),
        OptionalGroup([COMMA, TypesOf("target_inputs")], anchor="target_inputs"),
        RPAREN,
        ARROW,
        ResultTypeList("outputs"),
    ],
    examples=[
        "%result = pipeline.reduce @sum from %products(%partials) to %reducers(%bias) : (group, pipeline.flow<tile<8x8xi32>>) to (group, pipeline.flow<tile<8x8xi32>>) -> (pipeline.flow<tile<8x8xi32>>)"
    ],
)

pipeline_write = Op(
    "pipeline.write",
    group=pipeline_ops,
    doc="Write each source-group tile record to a destination view.",
    operands=[
        Operand("source", ANY, doc="Source tile flow."),
        Operand("target", VIEW, doc="Destination view."),
    ],
    traits=[UNKNOWN_EFFECTS, *_PIPELINE_GRAPH_TRAITS],
    verify="loom_pipeline_write_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("target"),
        COLON,
        TypeOf("source"),
        COMMA,
        TypeOf("target"),
    ],
    examples=["pipeline.write %result to %output : pipeline.flow<tile<8x8xi32>>, view<8x8xi32>"],
)

pipeline_return = Op(
    "pipeline.return",
    group=pipeline_ops,
    doc="Terminate a pipeline definition.",
    traits=[TERMINATOR, HasParent("pipeline.def")],
    examples=["pipeline.return"],
)

ALL_PIPELINE_TYPES: tuple[TypeDef, ...] = (pipeline_flow_type,)
ALL_PIPELINE_OPS: tuple[Op, ...] = (
    pipeline_def,
    pipeline_scatter,
    pipeline_read,
    pipeline_stage,
    pipeline_buffer,
    pipeline_reduce,
    pipeline_write,
    pipeline_return,
)
