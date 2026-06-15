# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE VM target-family record dialect."""

from typing import Any

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    AttrDict,
    FormatElement,
    FuncArgs,
    OptionalGroup,
    PredicateList,
    Ref,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TemplateParam,
    TypeOf,
    kw,
)
from loom.dialect.func.defs import CallingConv, Purity, Visibility
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ANY,
    SYMBOL_DEFINE,
    AttrDef,
    Borrow,
    Dialect,
    Discard,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    Op,
    Operand,
    OpPhase,
    Reads,
    Release,
    Result,
    Retain,
    RetainedResult,
    SameType,
    SymbolDefinition,
    SymbolReference,
    TargetLikeInterface,
    TypeDef,
    TypeParam,
    Writes,
)

ireevm_ops = Dialect(
    "ireevm",
    dialect_id=0x1A,
    doc="IREE VM target-family records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/ireevm/ops",
    register_by_default=False,
)

IreeVmTargetKind = EnumDef(
    "IreeVmTargetKind",
    [
        EnumCase("core", 1, doc="IREE VM core target row."),
    ],
    doc="IREE VM target row selected by ireevm.target.",
)

ireevm_ref_type = TypeDef(
    name="ireevm.ref",
    params=[TypeParam("object", ANY)],
    format=[TypeOf("object")],
    doc="IREE VM reference-counted object reference.",
)

ireevm_list_type = TypeDef(
    name="ireevm.list",
    params=[TypeParam("element", ANY)],
    format=[TypeOf("element")],
    doc="IREE VM list heap object type used behind ireevm.ref.",
)

ireevm_buffer_type = TypeDef(
    name="ireevm.buffer",
    doc="IREE VM host byte-buffer heap object type used behind ireevm.ref.",
)

_IMPORT_DECL_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef(
        "target",
        "symbol",
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef("import_symbol", "string"),
    AttrDef("cc", "enum", enum_def=CallingConv, optional=True),
    AttrDef("purity", "enum", enum_def=Purity, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_IMPORT_DECL_MODIFIER_FORMAT: list[FormatElement] = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    kw("target"),
    GLUE,
    LPAREN,
    SymbolRef("target"),
    GLUE,
    RPAREN,
    kw("symbol"),
    GLUE,
    LPAREN,
    Attr("import_symbol"),
    GLUE,
    RPAREN,
    OptionalGroup([Attr("cc")], anchor="cc"),
    OptionalGroup([Attr("purity")], anchor="purity"),
]

_IMPORT_DECL_SIGNATURE_FORMAT: list[FormatElement] = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs("args"),
            OptionalGroup(
                [ARROW, ResultTypeList("results")],
                anchor="results",
            ),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_IMPORT_DECL_FUNC_LIKE: dict[str, Any] = dict(
    callee="callee",
    target="target",
    import_symbol="import_symbol",
    visibility="visibility",
    cc="cc",
    purity="purity",
    predicates="predicates",
)

ireevm_target = Op(
    "ireevm.target",
    group=ireevm_ops,
    doc=(
        "IREE VM target-family record. The selector chooses a VM emission row; "
        "optional attrs structurally override authored common target fields."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_ireevm_target_bundles",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(IreeVmTargetKind),
    verify="loom_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "ireevm.target<core> @vm",
    ],
)

ireevm_import_decl = Op(
    "ireevm.import.decl",
    group=ireevm_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "IREE VM imported function declaration. Callable by name via func.call "
        "and lowered to a VM module import during target materialization."
    ),
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    attrs=list(_IMPORT_DECL_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
        fact_domain="loom_func_symbol_fact_domain",
    ),
    results=[Result("results", ANY, variadic=True)],
    interfaces=[FuncLikeInterface(**_IMPORT_DECL_FUNC_LIKE, args_as_operands=True)],
    verify="loom_ireevm_import_decl_verify",
    format=[
        *_IMPORT_DECL_MODIFIER_FORMAT,
        *_IMPORT_DECL_SIGNATURE_FORMAT,
    ],
    examples=[
        'ireevm.import.decl target(@vm) symbol("hal.buffer.length") '
        "@hal_buffer_length(%buffer: ireevm.ref<ireevm.list<i32>>) -> (i64)",
        'ireevm.import.decl public target(@vm) symbol("hal.buffer.map") '
        "@hal_buffer_map(%buffer: i32) -> (i64)",
    ],
)

ireevm_ref_retain = Op(
    "ireevm.ref.retain",
    group=ireevm_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Retain an additional owned IREE VM reference.",
    operands=[Operand("resource", ANY, doc="Reference value to retain.")],
    results=[Result("result", ANY, doc="Retained owned reference.")],
    constraints=[SameType("resource", "result")],
    ownership_effects=[Retain("resource"), RetainedResult("result")],
    format=[
        Ref("resource"),
        COLON,
        TypeOf("resource"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%owned = ireevm.ref.retain %resource : "
        "ireevm.ref<ireevm.list<i32>> -> ireevm.ref<ireevm.list<i32>>",
    ],
)

ireevm_ref_release = Op(
    "ireevm.ref.release",
    group=ireevm_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Release an owned IREE VM reference.",
    operands=[Operand("resource", ANY, doc="Reference value to release.")],
    ownership_effects=[Release("resource")],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "ireevm.ref.release %resource : ireevm.ref<ireevm.list<i32>>",
    ],
)

ireevm_ref_discard = Op(
    "ireevm.ref.discard",
    group=ireevm_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Discard compiler ownership of an IREE VM reference without releasing it.",
    operands=[Operand("resource", ANY, doc="Reference value to discard.")],
    ownership_effects=[Discard("resource")],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "ireevm.ref.discard %resource : ireevm.ref<ireevm.list<i32>>",
    ],
)

_BUFFER_LOAD_FORMAT: list[FormatElement] = [
    Ref("buffer"),
    GLUE,
    LBRACKET,
    Ref("element_offset"),
    RBRACKET,
    COLON,
    TypeOf("buffer"),
    COMMA,
    TypeOf("element_offset"),
    ARROW,
    ResultType("result"),
]

_BUFFER_STORE_FORMAT: list[FormatElement] = [
    Ref("value"),
    COMMA,
    Ref("buffer"),
    GLUE,
    LBRACKET,
    Ref("element_offset"),
    RBRACKET,
    COLON,
    TypeOf("value"),
    COMMA,
    TypeOf("buffer"),
    COMMA,
    TypeOf("element_offset"),
]


ireevm_buffer_length = Op(
    "ireevm.buffer.length",
    group=ireevm_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Return the byte length of an IREE VM host byte buffer.",
    operands=[Operand("buffer", ANY, doc="Buffer reference to query.")],
    results=[Result("result", ANY, doc="Byte length as an i64 scalar.")],
    ownership_effects=[Borrow("buffer")],
    effects=[Reads("buffer")],
    verify="loom_ireevm_buffer_op_verify",
    format=[
        Ref("buffer"),
        COLON,
        TypeOf("buffer"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%length = ireevm.buffer.length %buffer : ireevm.ref<ireevm.buffer> -> i64",
    ],
)


def _buffer_load_op(name: str, result_type: str, result_doc: str) -> Op:
    return Op(
        f"ireevm.buffer.load.{name}",
        group=ireevm_ops,
        phase=OpPhase.EXECUTABLE,
        doc=f"Load {result_doc} from an IREE VM host byte buffer.",
        operands=[
            Operand("buffer", ANY, doc="Source buffer reference."),
            Operand(
                "element_offset",
                ANY,
                doc="Element offset into the source buffer.",
            ),
        ],
        results=[Result("result", ANY, doc="Loaded scalar value.")],
        ownership_effects=[Borrow("buffer")],
        effects=[Reads("buffer")],
        verify="loom_ireevm_buffer_op_verify",
        format=_BUFFER_LOAD_FORMAT,
        examples=[
            f"%value = ireevm.buffer.load.{name} %buffer[%offset] : "
            f"ireevm.ref<ireevm.buffer>, i64 -> {result_type}",
        ],
    )


ireevm_buffer_load_i8_u = _buffer_load_op("i8.u", "i32", "an unsigned i8 value as i32")
ireevm_buffer_load_i8_s = _buffer_load_op(
    "i8.s", "i32", "a sign-extended i8 value as i32"
)
ireevm_buffer_load_i16_u = _buffer_load_op(
    "i16.u", "i32", "an unsigned i16 value as i32"
)
ireevm_buffer_load_i16_s = _buffer_load_op(
    "i16.s", "i32", "a sign-extended i16 value as i32"
)
ireevm_buffer_load_i32 = _buffer_load_op("i32", "i32", "an i32 value")
ireevm_buffer_load_i64 = _buffer_load_op("i64", "i64", "an i64 value")
ireevm_buffer_load_f32 = _buffer_load_op("f32", "f32", "an f32 value")
ireevm_buffer_load_f64 = _buffer_load_op("f64", "f64", "an f64 value")


def _buffer_store_op(name: str, value_type: str, value_doc: str) -> Op:
    return Op(
        f"ireevm.buffer.store.{name}",
        group=ireevm_ops,
        phase=OpPhase.EXECUTABLE,
        doc=f"Store {value_doc} into an IREE VM host byte buffer.",
        operands=[
            Operand("buffer", ANY, doc="Target buffer reference."),
            Operand(
                "element_offset",
                ANY,
                doc="Element offset into the target buffer.",
            ),
            Operand("value", ANY, doc="Scalar value to store."),
        ],
        ownership_effects=[Borrow("buffer")],
        effects=[Writes("buffer")],
        verify="loom_ireevm_buffer_op_verify",
        format=_BUFFER_STORE_FORMAT,
        examples=[
            f"ireevm.buffer.store.{name} %value, %buffer[%offset] : "
            f"{value_type}, ireevm.ref<ireevm.buffer>, i64",
        ],
    )


ireevm_buffer_store_i8 = _buffer_store_op("i8", "i32", "the low 8 bits of an i32 value")
ireevm_buffer_store_i16 = _buffer_store_op(
    "i16", "i32", "the low 16 bits of an i32 value"
)
ireevm_buffer_store_i32 = _buffer_store_op("i32", "i32", "an i32 value")
ireevm_buffer_store_i64 = _buffer_store_op("i64", "i64", "an i64 value")
ireevm_buffer_store_f32 = _buffer_store_op("f32", "f32", "an f32 value")
ireevm_buffer_store_f64 = _buffer_store_op("f64", "f64", "an f64 value")

ALL_IREEVM_TYPES: tuple[TypeDef, ...] = (
    ireevm_ref_type,
    ireevm_list_type,
    ireevm_buffer_type,
)

ALL_IREEVM_OPS: tuple[Op, ...] = (
    ireevm_target,
    ireevm_import_decl,
    ireevm_ref_retain,
    ireevm_ref_release,
    ireevm_ref_discard,
    ireevm_buffer_length,
    ireevm_buffer_load_i8_u,
    ireevm_buffer_load_i8_s,
    ireevm_buffer_load_i16_u,
    ireevm_buffer_load_i16_s,
    ireevm_buffer_load_i32,
    ireevm_buffer_load_i64,
    ireevm_buffer_load_f32,
    ireevm_buffer_load_f64,
    ireevm_buffer_store_i8,
    ireevm_buffer_store_i16,
    ireevm_buffer_store_i32,
    ireevm_buffer_store_i64,
    ireevm_buffer_store_f32,
    ireevm_buffer_store_f64,
)
