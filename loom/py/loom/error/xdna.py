# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""XDNA domain — XDNA-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_XDNA_001: Array worker fold has an empty record sequence.
ERR_XDNA_001 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=1,
    severity=Severity.ERROR,
    summary="Array worker fold has an empty record sequence.",
    message=("array worker fold requires a positive record count; got {record_count}"),
    params=(ErrorParam("record_count", ParamKind.U32),),
    fix_hint="Provide a non-empty record sequence for the worker fold.",
)

# ERR_XDNA_005: AIE2P worker channel cycle requires interleaved phases.
ERR_XDNA_005 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=5,
    severity=Severity.ERROR,
    summary="AIE2P worker channel cycle requires interleaved phases.",
    message=(
        "AIE2P worker {worker} (group {group} lane {lane}) participates in a "
        "channel cycle but waits for all inputs before publishing any output"
    ),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("group", ParamKind.U32),
        ErrorParam("lane", ParamKind.U32),
    ),
    fix_hint="Place the stages in groups whose worker dependencies are acyclic.",
)

# ERR_XDNA_006: An AIE2P channel cannot acquire physical resources.
ERR_XDNA_006 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=6,
    severity=Severity.ERROR,
    summary="AIE2P channel cannot acquire physical resources.",
    message=(
        "AIE2P channel {channel} at worker ({column}, {row}) cannot acquire "
        "physical resources with capacity {capacity} and {record_bytes}-byte "
        "records: {reason}"
    ),
    params=(
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("column", ParamKind.U32),
        ErrorParam("row", ParamKind.U32),
        ErrorParam("capacity", ParamKind.U32),
        ErrorParam("record_bytes", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Reduce the channel capacity or record size, reduce competing channels, "
        "or change worker placement."
    ),
)

# ERR_XDNA_007: An array stream cannot be routed within link capacity.
ERR_XDNA_007 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=7,
    severity=Severity.ERROR,
    summary="Array stream link capacity is exhausted.",
    message=(
        "AIE2P array channel {channel} cannot be routed because a stream link "
        "has no free channels"
    ),
    params=(ErrorParam("channel", ParamKind.U32),),
    fix_hint="Reduce independent streams crossing the link or change worker placement.",
)

# ERR_XDNA_013: A worker coordinate lies outside the physical array.
ERR_XDNA_013 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=13,
    severity=Severity.ERROR,
    summary="Array worker coordinate is outside the device.",
    message=(
        "AIE2P worker coordinate ({column}, {row}) is outside the "
        "{column_count}-column, {row_count}-row array"
    ),
    params=(
        ErrorParam("column", ParamKind.U32),
        ErrorParam("row", ParamKind.U32),
        ErrorParam("column_count", ParamKind.U32),
        ErrorParam("row_count", ParamKind.U32),
    ),
    fix_hint="Place the worker on a compute tile within the physical array.",
)

# ERR_XDNA_014: AIE2P array body contains an operation outside its topology.
ERR_XDNA_014 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=14,
    severity=Severity.ERROR,
    summary="AIE2P array body contains a non-topology operation.",
    message=(
        "AIE2P array programs contain topology packets and low.return; "
        "'{op_name}' is not an array topology operation"
    ),
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Place executable computation in a resident core worker.",
)

# ERR_XDNA_015: AIE2P array packet result requires a tile payload.
ERR_XDNA_015 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=15,
    severity=Severity.ERROR,
    summary="AIE2P array packet result requires a tile payload.",
    message=(
        "AIE2P array descriptor '{descriptor}' requires a tile-valued result "
        "register; got {result_type}"
    ),
    params=(
        ErrorParam("descriptor", ParamKind.STRING),
        ErrorParam("result_type", ParamKind.TYPE),
    ),
    fix_hint="Specify the transported tile type in the result register.",
)

# ERR_XDNA_016: AIE2P resident worker materialization requires a definition.
ERR_XDNA_016 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=16,
    severity=Severity.ERROR,
    summary="AIE2P resident worker materialization requires a definition.",
    message=(
        "AIE2P resident worker '@{entry}' requires a local core function "
        "definition before array materialization"
    ),
    params=(ErrorParam("entry", ParamKind.STRING),),
    fix_hint=(
        "Link the core worker definition into the module before emitting the array."
    ),
)

# ERR_XDNA_017: A physical DMA cannot represent a channel record size.
ERR_XDNA_017 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=17,
    severity=Severity.ERROR,
    summary="Physical DMA cannot represent a channel record size.",
    message=(
        "channel {channel} has {record_bytes} byte records, but the selected "
        "DMA engine accepts records from {minimum_bytes} through "
        "{maximum_bytes} bytes in {granularity_bytes}-byte units"
    ),
    params=(
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("record_bytes", ParamKind.U32),
        ErrorParam("minimum_bytes", ParamKind.U64),
        ErrorParam("maximum_bytes", ParamKind.U64),
        ErrorParam("granularity_bytes", ParamKind.U32),
    ),
    fix_hint=(
        "Pad or split the channel records, or select a transport with a "
        "compatible record-size domain."
    ),
)

# ERR_XDNA_018: A resident array has no executable channel graph.
ERR_XDNA_018 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=18,
    severity=Severity.ERROR,
    summary="Resident array has no executable channel graph.",
    message=(
        "AIE2P resident array has {worker_count} workers and {channel_count} "
        "channels; both counts must be positive"
    ),
    params=(
        ErrorParam("worker_count", ParamKind.U32),
        ErrorParam("channel_count", ParamKind.U32),
    ),
    fix_hint="Connect at least one resident worker through an array channel.",
)

# ERR_XDNA_019: A worker group has invalid lane membership.
ERR_XDNA_019 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=19,
    severity=Severity.ERROR,
    summary="Worker group has invalid lane membership.",
    message=(
        "AIE2P group {group} declares {lane_count} lanes, but lane {lane} has "
        "{worker_count} workers; each declared lane requires exactly one worker "
        "and no other lane is valid"
    ),
    params=(
        ErrorParam("group", ParamKind.U32),
        ErrorParam("lane", ParamKind.U32),
        ErrorParam("lane_count", ParamKind.U32),
        ErrorParam("worker_count", ParamKind.U32),
    ),
    fix_hint="Instantiate every declared group lane exactly once.",
)

# ERR_XDNA_020: An array worker has invalid placement cardinality.
ERR_XDNA_020 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=20,
    severity=Severity.ERROR,
    summary="Array worker has invalid placement cardinality.",
    message=(
        "AIE2P worker {worker} has {location_count} location constraints; "
        "exactly one is required"
    ),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("location_count", ParamKind.U32),
    ),
    fix_hint="Provide exactly one physical location for the resident worker.",
)

# ERR_XDNA_021: An array worker cannot occupy its selected tile.
ERR_XDNA_021 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=21,
    severity=Severity.ERROR,
    summary="Array worker cannot occupy its selected tile.",
    message=("AIE2P worker {worker} cannot occupy tile ({column}, {row}): {reason}"),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("column", ParamKind.U32),
        ErrorParam("row", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint="Select one unoccupied compute tile for each resident worker.",
)

# ERR_XDNA_022: A resident worker port has no unique active/resource mapping.
ERR_XDNA_022 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=22,
    severity=Severity.ERROR,
    summary="Resident worker port does not match its leaf resource ABI.",
    message=(
        "AIE2P worker {worker} entry '@{entry}' port {port} has "
        "{active_endpoint_count} active topology endpoints and "
        "{resource_count} leaf resources; each port requires exactly one active "
        "endpoint and at most one resource"
    ),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("entry", ParamKind.STRING),
        ErrorParam("port", ParamKind.U64),
        ErrorParam("active_endpoint_count", ParamKind.U32),
        ErrorParam("resource_count", ParamKind.U32),
    ),
    fix_hint=(
        "Define the topology port once and match each leaf resource to that "
        "port, or omit the resource for a synchronization-only port."
    ),
)

# ERR_XDNA_023: An active binding view has invalid record geometry.
ERR_XDNA_023 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=23,
    severity=Severity.ERROR,
    summary="Active binding view has invalid record geometry.",
    message=(
        "AIE2P binding view {view_type} lane {lane} of {lane_count} cannot "
        "select records from {source_type}: {reason}"
    ),
    params=(
        ErrorParam("view_type", ParamKind.TYPE),
        ErrorParam("lane", ParamKind.U32),
        ErrorParam("lane_count", ParamKind.U32),
        ErrorParam("source_type", ParamKind.TYPE),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint="Make the active view select a representable suffix of one binding tile.",
)

# ERR_XDNA_024: A logical channel record has no representable byte footprint.
ERR_XDNA_024 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=24,
    severity=Severity.ERROR,
    summary="Logical channel record has no representable byte footprint.",
    message=(
        "AIE2P channel {channel} record type {record_type} has no representable "
        "byte footprint: {reason}"
    ),
    params=(
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("record_type", ParamKind.TYPE),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint="Use a non-empty exact tile shape whose whole-byte size fits in u32.",
)

# ERR_XDNA_025: A logical channel ring violates a required relationship.
ERR_XDNA_025 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=25,
    severity=Severity.ERROR,
    summary="Logical channel ring has an invalid size relationship.",
    message=(
        "AIE2P channel {channel} has {quantity} {actual}; {relationship} is {required}"
    ),
    params=(
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("quantity", ParamKind.STRING),
        ErrorParam("actual", ParamKind.U32),
        ErrorParam("relationship", ParamKind.STRING),
        ErrorParam("required", ParamKind.U64),
    ),
    fix_hint="Make the channel ring match the stated topology relationship.",
)

# ERR_XDNA_026: A logical channel connects incompatible endpoint owners.
ERR_XDNA_026 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=26,
    severity=Severity.ERROR,
    summary="Logical channel connects incompatible endpoint owners.",
    message=(
        "AIE2P channel {channel} cannot connect "
        "{sender_kind}[{sender_owner}]:{sender_port} to "
        "{receiver_kind}[{receiver_owner}]:{receiver_port}: {reason}"
    ),
    params=(
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("sender_kind", ParamKind.STRING),
        ErrorParam("sender_owner", ParamKind.U32),
        ErrorParam("sender_port", ParamKind.U32),
        ErrorParam("receiver_kind", ParamKind.STRING),
        ErrorParam("receiver_owner", ParamKind.U32),
        ErrorParam("receiver_port", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Connect the channel through a resident worker and compatible binding access."
    ),
)

# ERR_XDNA_027: A receiver endpoint consumes more than one source channel.
ERR_XDNA_027 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=27,
    severity=Severity.ERROR,
    summary="Receiver endpoint consumes more than one source channel.",
    message=(
        "AIE2P receiver endpoint {endpoint} is consumed by channels "
        "{first_channel} and {channel}; an active receiver accepts exactly "
        "one source channel"
    ),
    params=(
        ErrorParam("endpoint", ParamKind.U32),
        ErrorParam("first_channel", ParamKind.U32),
        ErrorParam("channel", ParamKind.U32),
    ),
    fix_hint="Give each source channel a distinct receiver endpoint.",
)

# ERR_XDNA_028: A temporal fold declares an invalid output range.
ERR_XDNA_028 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=28,
    severity=Severity.ERROR,
    summary="Temporal fold output range is invalid.",
    message=(
        "AIE2P worker {worker} fold declares {output_count} outputs from port "
        "{output_port}; {quantity} {actual} {requirement}"
    ),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("output_port", ParamKind.U32),
        ErrorParam("output_count", ParamKind.U32),
        ErrorParam("quantity", ParamKind.STRING),
        ErrorParam("actual", ParamKind.U64),
        ErrorParam("requirement", ParamKind.STRING),
    ),
    fix_hint=(
        "Make active sender ports exactly cover one non-empty representable "
        "output range."
    ),
)

# ERR_XDNA_029: A temporal fold cannot be materialized.
ERR_XDNA_029 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=29,
    severity=Severity.ERROR,
    summary="Temporal fold cannot be materialized.",
    message=(
        "AIE2P worker {worker} cannot fold channel {channel} record type "
        "{record_type} with combiner {combiner}: {reason}"
    ),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("record_type", ParamKind.TYPE),
        ErrorParam("combiner", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Provide one compatible input cadence and use addf over one f32 element "
        "or a multiple of 16 f32 elements."
    ),
)

# ERR_XDNA_030: An AIE2P array ABI layout field is unsupported.
ERR_XDNA_030 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=30,
    severity=Severity.ERROR,
    summary="AIE2P array ABI layout field is unsupported.",
    message=(
        "AIE2P array function '@{function_name}' ABI layout field "
        "'{field_name}' is unsupported"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
    ),
    fix_hint="Use only the binding_count field in an AIE2P array ABI layout.",
)

# ERR_XDNA_031: An active binding ordinal violates the dense external ABI.
ERR_XDNA_031 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=31,
    severity=Severity.ERROR,
    summary="Active binding ordinal violates the array ABI.",
    message=(
        "AIE2P binding ordinal {ordinal} is {reason}; the array ABI declares "
        "{binding_count} dense slots"
    ),
    params=(
        ErrorParam("ordinal", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
        ErrorParam("binding_count", ParamKind.U32),
    ),
    fix_hint=(
        "Give each active binding one unique ordinal within the declared dense "
        "binding table."
    ),
)

# ERR_XDNA_032: An external binding transfer is not representable by shim DMA.
ERR_XDNA_032 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=32,
    severity=Severity.ERROR,
    summary="External binding transfer is not representable by shim DMA.",
    message=(
        "AIE2P channel {channel} cannot transfer {record_type} records from "
        "binding source {source_type} at byte offset {byte_offset} with shim "
        "DMA: {reason}"
    ),
    params=(
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("record_type", ParamKind.TYPE),
        ErrorParam("source_type", ParamKind.TYPE),
        ErrorParam("byte_offset", ParamKind.U64),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use an exact source layout whose partition offset, transfer span, "
        "repeats, and address dimensions fit the shim DMA fields."
    ),
)

# ERR_XDNA_033: A resident worker's local state cannot be placed.
ERR_XDNA_033 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=33,
    severity=Severity.ERROR,
    summary="Resident worker local state cannot be placed.",
    message=(
        "AIE2P worker {worker} at tile ({column}, {row}) cannot place "
        "{purpose} storage: {requested_bytes} bytes with {alignment_bytes}-byte "
        "alignment do not fit the {capacity_bytes}-byte banked local-memory domain"
    ),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("column", ParamKind.U32),
        ErrorParam("row", ParamKind.U32),
        ErrorParam("purpose", ParamKind.STRING),
        ErrorParam("requested_bytes", ParamKind.U64),
        ErrorParam("alignment_bytes", ParamKind.U64),
        ErrorParam("capacity_bytes", ParamKind.U32),
    ),
    fix_hint=(
        "Reduce the worker-local state, split the worker, or select a tile with "
        "sufficient local memory."
    ),
)

# ERR_XDNA_034: Read-only data bank conflicts cannot be placed.
ERR_XDNA_034 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=34,
    severity=Severity.ERROR,
    summary="Resident read-only data bank conflicts cannot be placed.",
    message=(
        "AIE2P worker {worker} at tile ({column}, {row}) cannot place "
        "read-only data '@{symbol}' in any of {bank_count} programmer banks "
        "without overlapping a conflicting resident definition"
    ),
    params=(
        ErrorParam("worker", ParamKind.U32),
        ErrorParam("column", ParamKind.U32),
        ErrorParam("row", ParamKind.U32),
        ErrorParam("symbol", ParamKind.STRING),
        ErrorParam("bank_count", ParamKind.U32),
    ),
    fix_hint=(
        "Reduce the conflicting data footprint, split the worker, or select a "
        "tile with enough independent local-memory banks."
    ),
)

ERR_XDNA_050 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=50,
    severity=Severity.ERROR,
    summary="Configuration function has a runtime signature.",
    message=(
        "configuration function '@{function_name}' has {argument_count} arguments "
        "and {result_count} results; configuration functions have neither"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("argument_count", ParamKind.U32),
        ErrorParam("result_count", ParamKind.U32),
    ),
)

ERR_XDNA_035 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=35,
    severity=Severity.ERROR,
    summary="Configuration function contains runtime control or storage.",
    message=(
        "configuration function '@{function_name}' requires a single sequence "
        "of configuration commands; '{op_name}' requires core execution"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

ERR_XDNA_036 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=36,
    severity=Severity.ERROR,
    summary="Configuration references an undefined function body.",
    message=(
        "configuration reference '{field_name}' to '@{symbol_name}' "
        "requires a {definition_kind} definition"
    ),
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("definition_kind", ParamKind.STRING),
    ),
)

ERR_XDNA_037 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=37,
    severity=Severity.ERROR,
    summary="Configuration reference requires a closed function signature.",
    message=(
        "configuration reference '{field_name}' to '@{symbol_name}' supplies "
        "no arguments or return continuation, but the function has "
        "{argument_count} arguments and {result_count} results"
    ),
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("argument_count", ParamKind.U32),
        ErrorParam("result_count", ParamKind.U32),
    ),
)

ERR_XDNA_038 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=38,
    severity=Severity.ERROR,
    summary="Configuration command is unavailable in the selected phase.",
    message=(
        "configuration function '@{function_name}' is used for {phase}, "
        "which cannot execute '{command}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("phase", ParamKind.STRING),
        ErrorParam("command", ParamKind.STRING),
    ),
)

ERR_XDNA_039 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=39,
    severity=Severity.ERROR,
    summary="Configuration entry count does not match its function role.",
    message=(
        "configuration function '@{function_name}' requires {expected_count} "
        "entry commands, but has {actual_count}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
        ErrorParam("actual_count", ParamKind.U32),
    ),
)

ERR_XDNA_040 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=40,
    severity=Severity.ERROR,
    summary="Loaded tile program requires an unbound physical resource.",
    message=(
        "loaded tile program '@{function_name}' contains '{op_name}'; "
        "program.load requires imported resource addresses "
        "to be explicit in the core program"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

ERR_XDNA_041 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=41,
    severity=Severity.ERROR,
    summary="Configuration operand is outside its physical range.",
    message=(
        "configuration operand '{operand_name}' is {value}; "
        "its physical range is [{minimum}, {maximum}]"
    ),
    params=(
        ErrorParam("operand_name", ParamKind.STRING),
        ErrorParam("value", ParamKind.U64),
        ErrorParam("minimum", ParamKind.U64),
        ErrorParam("maximum", ParamKind.U64),
    ),
)

ERR_XDNA_042 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=42,
    severity=Severity.ERROR,
    summary="Configuration operand violates an alignment contract.",
    message=(
        "configuration operand '{operand_name}' is {value}; "
        "it must be a multiple of {alignment}"
    ),
    params=(
        ErrorParam("operand_name", ParamKind.STRING),
        ErrorParam("value", ParamKind.U64),
        ErrorParam("alignment", ParamKind.U64),
    ),
)

ERR_XDNA_043 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=43,
    severity=Severity.ERROR,
    summary="Invocation binding ordinal is not dense.",
    message=(
        "invocation binding ordinal is {actual}; "
        "the next declared binding is {expected}"
    ),
    params=(ErrorParam("actual", ParamKind.U64), ErrorParam("expected", ParamKind.U64)),
)

ERR_XDNA_044 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=44,
    severity=Severity.ERROR,
    summary="Invocation range exceeds the declared binding extent.",
    message=(
        "binding {binding} has {extent} bytes; "
        "range offset {offset} and length {length} exceed that extent"
    ),
    params=(
        ErrorParam("binding", ParamKind.U32),
        ErrorParam("extent", ParamKind.U64),
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("length", ParamKind.U64),
    ),
)

ERR_XDNA_045 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=45,
    severity=Severity.ERROR,
    summary="Configuration register access crosses a physical tile aperture.",
    message=(
        "configuration register access at address {address} has {word_count} words "
        "and exceeds one tile in the {columns}-column partition"
    ),
    params=(
        ErrorParam("address", ParamKind.U32),
        ErrorParam("word_count", ParamKind.U64),
        ErrorParam("columns", ParamKind.U32),
    ),
)

ERR_XDNA_046 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=46,
    severity=Severity.ERROR,
    summary="Program load does not select a compute tile.",
    message=(
        "program.load coordinate ({column}, {row}) is not a compute tile "
        "in the {columns}-column partition"
    ),
    params=(
        ErrorParam("column", ParamKind.U64),
        ErrorParam("row", ParamKind.U64),
        ErrorParam("columns", ParamKind.U32),
    ),
)

ERR_XDNA_047 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=47,
    severity=Severity.ERROR,
    summary="Shim descriptor flags overlap its relocated address.",
    message=(
        "shim descriptor flags {flags} set low address bits "
        "reserved for the range operand"
    ),
    params=(ErrorParam("flags", ParamKind.U32),),
)

ERR_XDNA_048 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=48,
    severity=Severity.ERROR,
    summary="Binding alignment is not a power of two.",
    message="invocation binding alignment {alignment} must be a nonzero power of two",
    params=(ErrorParam("alignment", ParamKind.U64),),
)

ERR_XDNA_049 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=49,
    severity=Severity.ERROR,
    summary="Resident data and worker storage exceed tile memory.",
    message=(
        "tile [{column}, {row}] requires {required} bytes for resident data "
        "and loaded worker storage, but has {capacity} bytes"
    ),
    params=(
        ErrorParam("column", ParamKind.U32),
        ErrorParam("row", ParamKind.U32),
        ErrorParam("required", ParamKind.U64),
        ErrorParam("capacity", ParamKind.U64),
    ),
)

ERR_XDNA_052 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=52,
    severity=Severity.ERROR,
    summary="Register wait requires bits outside its comparison mask.",
    message=(
        "wait.mask32 value {value} sets bits outside mask {mask}; "
        "the masked register cannot match this value"
    ),
    params=(ErrorParam("mask", ParamKind.U32), ErrorParam("value", ParamKind.U32)),
)

ERR_XDNA_051 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=51,
    severity=Severity.ERROR,
    summary="Resident worker realization cannot satisfy this program.",
    message="AIE2P resident worker realization requires {requirement}",
    params=(ErrorParam("requirement", ParamKind.STRING),),
)

# ERR_XDNA_053: An XDNA artifact root is not an artifact entry.
ERR_XDNA_053 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=53,
    severity=Severity.ERROR,
    summary="XDNA artifact root is not an artifact entry.",
    message=(
        "XDNA artifact root '@{function_name}' uses representation "
        "'{representation}'; expected 'amd.xdna.aie2p.array' or "
        "'amd.xdna.aie2p.configuration'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("representation", ParamKind.STRING),
    ),
    fix_hint="Select an AIE2P array or configuration root for XDNA emission.",
)

# ERR_XDNA_054: An XDNA artifact has no exact deployment profile.
ERR_XDNA_054 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=54,
    severity=Severity.ERROR,
    summary="XDNA artifact has no exact deployment profile.",
    message=(
        "AIE2P artifact entry '@{function_name}' uses a generic target and no "
        "artifact entry selects the exact device profile required for emission"
    ),
    params=(ErrorParam("function_name", ParamKind.STRING),),
    fix_hint="Select one exact XDNA device profile for the artifact entry.",
)

# ERR_XDNA_055: XDNA artifact entries select different deployment profiles.
ERR_XDNA_055 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=55,
    severity=Severity.ERROR,
    summary="XDNA artifact entries select different deployment profiles.",
    message=(
        "AIE2P artifact entry '@{function_name}' selects device profile "
        "'{actual_profile}', but the artifact already uses '{expected_profile}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("actual_profile", ParamKind.STRING),
        ErrorParam("expected_profile", ParamKind.STRING),
    ),
    fix_hint="Compile entries for different device profiles as separate artifacts.",
)

# ERR_XDNA_056: An AIE2P product exceeds an image-format limit.
ERR_XDNA_056 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=56,
    severity=Severity.ERROR,
    summary="AIE2P product exceeds an image-format limit.",
    message=(
        "AIE2P product requires {actual} {quantity}; the XDNA image format "
        "supports at most {maximum}"
    ),
    params=(
        ErrorParam("quantity", ParamKind.STRING),
        ErrorParam("actual", ParamKind.U64),
        ErrorParam("maximum", ParamKind.U64),
    ),
    fix_hint="Reduce the number or encoded size of entries in this artifact.",
)

ERR_XDNA_057 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=57,
    severity=Severity.ERROR,
    summary="XDNA measurement boundaries do not form one complete interval.",
    message=(
        "configuration function '@{function_name}' requires at most one "
        "ordered measure.begin/measure.end pair: {reason}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint="Place one complete measurement interval in the invocation function.",
)

# ERR_XDNA_058: An AIE2P tile program exceeds core program memory.
ERR_XDNA_058 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=58,
    severity=Severity.ERROR,
    summary="AIE2P tile program exceeds core program memory.",
    message=(
        "AIE2P tile program '@{function_name}' requires {required} code bytes, "
        "but the core has {capacity} bytes of program memory"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("required", ParamKind.U64),
        ErrorParam("capacity", ParamKind.U64),
    ),
    fix_hint="Reduce or split the program's native instruction footprint.",
)

ALL_XDNA_ERRORS = (
    ERR_XDNA_001,
    ERR_XDNA_005,
    ERR_XDNA_006,
    ERR_XDNA_007,
    ERR_XDNA_013,
    ERR_XDNA_014,
    ERR_XDNA_015,
    ERR_XDNA_016,
    ERR_XDNA_017,
    ERR_XDNA_018,
    ERR_XDNA_019,
    ERR_XDNA_020,
    ERR_XDNA_021,
    ERR_XDNA_022,
    ERR_XDNA_023,
    ERR_XDNA_024,
    ERR_XDNA_025,
    ERR_XDNA_026,
    ERR_XDNA_027,
    ERR_XDNA_028,
    ERR_XDNA_029,
    ERR_XDNA_030,
    ERR_XDNA_031,
    ERR_XDNA_032,
    ERR_XDNA_033,
    ERR_XDNA_034,
    ERR_XDNA_050,
    ERR_XDNA_035,
    ERR_XDNA_036,
    ERR_XDNA_037,
    ERR_XDNA_038,
    ERR_XDNA_039,
    ERR_XDNA_040,
    ERR_XDNA_041,
    ERR_XDNA_042,
    ERR_XDNA_043,
    ERR_XDNA_044,
    ERR_XDNA_045,
    ERR_XDNA_046,
    ERR_XDNA_047,
    ERR_XDNA_048,
    ERR_XDNA_049,
    ERR_XDNA_052,
    ERR_XDNA_051,
    ERR_XDNA_053,
    ERR_XDNA_054,
    ERR_XDNA_055,
    ERR_XDNA_056,
    ERR_XDNA_057,
    ERR_XDNA_058,
)
