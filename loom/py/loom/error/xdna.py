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

# ERR_XDNA_002: AIE2P worker requires a frame-completion phase.
ERR_XDNA_002 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=2,
    severity=Severity.ERROR,
    summary="AIE2P worker requires a frame-completion phase.",
    message=(
        "AIE2P pipeline group {group} has frame-completion stages that "
        "require a phased worker program"
    ),
    params=(ErrorParam("group", ParamKind.U32),),
    fix_hint="Place completion stages in a separate group.",
)

# ERR_XDNA_003: AIE2P worker outputs require different firing phases.
ERR_XDNA_003 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=3,
    severity=Severity.ERROR,
    summary="AIE2P worker outputs require different firing phases.",
    message=(
        "AIE2P pipeline group {group} requires compatible folds on every "
        "boundary output; mixed cadences require a phased worker program"
    ),
    params=(ErrorParam("group", ParamKind.U32),),
    fix_hint="Place recordwise and folded outputs in separate groups.",
)

# ERR_XDNA_004: AIE2P internal buffered flow requires a ring state machine.
ERR_XDNA_004 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=4,
    severity=Severity.ERROR,
    summary="AIE2P internal buffered flow requires a ring state machine.",
    message=(
        "AIE2P pipeline group {group} flow {flow} has capacity {capacity}; "
        "buffered same-group flow requires a composite ring state machine"
    ),
    params=(
        ErrorParam("group", ParamKind.U32),
        ErrorParam("flow", ParamKind.U32),
        ErrorParam("capacity", ParamKind.U32),
    ),
    fix_hint="Place the buffered flow producer and consumer in separate groups.",
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

# ERR_XDNA_006: AIE2P channel has no available compute endpoint.
ERR_XDNA_006 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=6,
    severity=Severity.ERROR,
    summary="AIE2P channel has no available compute endpoint.",
    message=(
        "AIE2P channel {channel} needs {capacity} records of {record_bytes} bytes, "
        "compatible DMA channels, descriptors and locks on a compute tile visible "
        "to worker ({column}, {row}); no candidate has all requested resources"
    ),
    params=(
        ErrorParam("channel", ParamKind.U32),
        ErrorParam("column", ParamKind.U32),
        ErrorParam("row", ParamKind.U32),
        ErrorParam("capacity", ParamKind.U32),
        ErrorParam("record_bytes", ParamKind.U32),
    ),
    fix_hint="Reduce the channel capacity or record size, or change worker placement.",
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

# ERR_XDNA_008: AIE2P array pipeline requires kernel materialization scope.
ERR_XDNA_008 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=8,
    severity=Severity.ERROR,
    summary="AIE2P array pipeline requires kernel materialization scope.",
    message=(
        "AIE2P array pipeline requires kernel materialization scope; got '{scope}'"
    ),
    params=(ErrorParam("scope", ParamKind.STRING),),
    fix_hint="Declare pipeline.def<kernel> for one resident array executable.",
)

# ERR_XDNA_009: AIE2P pipeline exceeds resident compute capacity.
ERR_XDNA_009 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=9,
    severity=Severity.ERROR,
    summary="AIE2P pipeline exceeds resident compute capacity.",
    message=(
        "AIE2P pipeline requires {instance_count} resident instances "
        "but has {compute_tile_count} compute tiles"
    ),
    params=(
        ErrorParam("instance_count", ParamKind.U32),
        ErrorParam("compute_tile_count", ParamKind.U32),
    ),
    fix_hint="Reduce the total resident group lane count.",
)

# ERR_XDNA_010: AIE2P composite stages have different core targets.
ERR_XDNA_010 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=10,
    severity=Severity.ERROR,
    summary="AIE2P composite stages have different core targets.",
    message=(
        "AIE2P pipeline group {group} stage '@{entry}' uses target "
        "'@{actual_target}', but its other stages use '@{expected_target}'"
    ),
    params=(
        ErrorParam("group", ParamKind.U32),
        ErrorParam("entry", ParamKind.STRING),
        ErrorParam("actual_target", ParamKind.STRING),
        ErrorParam("expected_target", ParamKind.STRING),
    ),
    fix_hint="Use one exact core target for all stages in a resident group.",
)

# ERR_XDNA_011: AIE2P composite flow has no representable private record.
ERR_XDNA_011 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=11,
    severity=Severity.ERROR,
    summary="AIE2P composite flow has no representable private record.",
    message=(
        "AIE2P pipeline group {group} internal flow {flow} requires a "
        "whole-byte tile whose bit count fits in 64 bits; got {tile_type}"
    ),
    params=(
        ErrorParam("group", ParamKind.U32),
        ErrorParam("flow", ParamKind.U32),
        ErrorParam("tile_type", ParamKind.TYPE),
    ),
    fix_hint="Use byte-complete internal records with a representable total size.",
)

# ERR_XDNA_012: AIE2P composite worker exceeds callable ABI capacity.
ERR_XDNA_012 = ErrorDef(
    domain=ErrorDomain.XDNA,
    code=12,
    severity=Severity.ERROR,
    summary="AIE2P composite worker exceeds callable ABI capacity.",
    message=(
        "AIE2P pipeline group {group} requires {port_count} composite buffer "
        "arguments; the callable ABI supports at most {maximum}"
    ),
    params=(
        ErrorParam("group", ParamKind.U32),
        ErrorParam("port_count", ParamKind.U32),
        ErrorParam("maximum", ParamKind.U32),
    ),
    fix_hint="Split the stages across groups or reduce distinct boundary flows.",
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

ALL_XDNA_ERRORS = (
    ERR_XDNA_001,
    ERR_XDNA_002,
    ERR_XDNA_003,
    ERR_XDNA_004,
    ERR_XDNA_005,
    ERR_XDNA_006,
    ERR_XDNA_007,
    ERR_XDNA_008,
    ERR_XDNA_009,
    ERR_XDNA_010,
    ERR_XDNA_011,
    ERR_XDNA_012,
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
)
