# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""STRUCTURE domain — count errors, terminators, and regions."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_STRUCTURE_001: Wrong operand count.
ERR_STRUCTURE_001 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=1,
    severity=Severity.ERROR,
    summary="Wrong operand count.",
    message="'{op_name}' has {actual_count} operands, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_002: Wrong result count.
ERR_STRUCTURE_002 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=2,
    severity=Severity.ERROR,
    summary="Wrong result count.",
    message="'{op_name}' has {actual_count} results, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_003: Wrong attribute count.
ERR_STRUCTURE_003 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=3,
    severity=Severity.ERROR,
    summary="Wrong attribute count.",
    message="'{op_name}' has {actual_count} attributes, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_004: Wrong region count.
ERR_STRUCTURE_004 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=4,
    severity=Severity.ERROR,
    summary="Wrong region count.",
    message="'{op_name}' has {actual_count} regions, expected {expected_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_005: Missing terminator in block.
ERR_STRUCTURE_005 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=5,
    severity=Severity.ERROR,
    summary="Missing terminator in block.",
    message="block in '{op_name}' region {region_index} is missing a terminator",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("region_index", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_006: Single-block region has wrong block count.
ERR_STRUCTURE_006 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=6,
    severity=Severity.ERROR,
    summary="Single-block region has wrong block count.",
    message="'{op_name}' region {region_index} must have exactly one "
    "block, has {block_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("region_index", ParamKind.U32),
        ErrorParam("block_count", ParamKind.U32),
    ),
)

# ERR_STRUCTURE_007: BlockArgCount constraint violated.
ERR_STRUCTURE_007 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=7,
    severity=Severity.ERROR,
    summary="Region block argument count mismatch.",
    message="region has {actual_count} block arguments, expected "
    "{expected_count} (one per input tile)",
    params=(
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Ensure the region has exactly one block argument per input tile",
)

# ERR_STRUCTURE_008: YieldCountMatchesResults violated.
ERR_STRUCTURE_008 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=8,
    severity=Severity.ERROR,
    summary="Yield must produce the correct number of values.",
    message="yield has {actual_count} operands, expected {expected_count}",
    params=(
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Yield exactly {expected_count} values",
)

# ERR_STRUCTURE_009: Unknown op kind.
ERR_STRUCTURE_009 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=9,
    severity=Severity.ERROR,
    summary="Unknown op kind.",
    message="unknown op kind {op_kind} in '{op_name}'",
    params=(
        ErrorParam("op_kind", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_STRUCTURE_010: Enum attribute value out of range.
ERR_STRUCTURE_010 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=10,
    severity=Severity.ERROR,
    summary="Enum attribute value out of range.",
    message="attribute '{attr_name}' has enum value {actual_value}, "
    "but only {enum_case_count} values are defined",
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.U32),
        ErrorParam("enum_case_count", ParamKind.U32),
    ),
    fix_hint="'{attr_name}' must have a value in range [0, {enum_case_count})",
)

# ERR_STRUCTURE_011: Non-symbol op at module level.
ERR_STRUCTURE_011 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=11,
    severity=Severity.ERROR,
    summary="Non-symbol-defining op at module level.",
    message="'{op_name}' cannot appear at module level (only symbol-defining "
    "ops like func.def and func.decl are allowed here)",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Move '{op_name}' inside a function body",
)

# ERR_STRUCTURE_012: Operation appears after a block terminator.
ERR_STRUCTURE_012 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=12,
    severity=Severity.ERROR,
    summary="Operation appears after a block terminator.",
    message="'{op_name}' appears after a block terminator",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Move '{op_name}' before the block terminator or remove the "
    "unreachable op",
)

# ERR_STRUCTURE_013: Two variadic fields disagree on element count.
ERR_STRUCTURE_013 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=13,
    severity=Severity.ERROR,
    summary="Variadic field count mismatch.",
    message="'{field_a}' has {actual_count} values but '{field_b}' has "
    "{expected_count}",
    params=(
        ErrorParam("field_a", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("field_b", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Ensure '{field_a}' and '{field_b}' have the same number of values",
)

# ERR_STRUCTURE_014: Attribute value violates a structural constraint.
ERR_STRUCTURE_014 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=14,
    severity=Severity.ERROR,
    summary="Attribute value violates a structural constraint.",
    message="attribute '{attr_name}' value {actual_value} must satisfy "
    "{expected_constraint}",
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.I64),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="Choose an attribute value satisfying '{expected_constraint}'",
)

# ERR_STRUCTURE_015: Static threshold table is not ordered.
ERR_STRUCTURE_015 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=15,
    severity=Severity.ERROR,
    summary="Static threshold table is not ordered.",
    message="threshold table '{field_name}' is not ordered at entries "
    "{left_index} and {right_index}",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("left_index", ParamKind.U32),
        ErrorParam("right_index", ParamKind.U32),
    ),
    fix_hint="Provide static thresholds in nondecreasing order",
)

# ERR_STRUCTURE_016: Operation traits are mutually incompatible.
ERR_STRUCTURE_016 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=16,
    severity=Severity.ERROR,
    summary="Operation traits are mutually incompatible.",
    message="'{op_name}' has incompatible traits {trait_a} and {trait_b}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("trait_a", ParamKind.STRING),
        ErrorParam("trait_b", ParamKind.STRING),
    ),
    fix_hint="Remove one of the incompatible traits from '{op_name}'",
)

# ERR_STRUCTURE_017: Pure function body has effects.
ERR_STRUCTURE_017 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=17,
    severity=Severity.ERROR,
    summary="Pure function body has effects.",
    message=(
        "'{op_name}' declares pure but its body has {read_count} read-like, "
        "{write_count} write-like, and {convergent_count} convergent effect(s)"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("read_count", ParamKind.U32),
        ErrorParam("write_count", ParamKind.U32),
        ErrorParam("convergent_count", ParamKind.U32),
    ),
    fix_hint="Remove the pure modifier or remove the observable effects",
)

# ERR_STRUCTURE_018: Region terminator has the wrong op kind.
ERR_STRUCTURE_018 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=18,
    severity=Severity.ERROR,
    summary="Region terminator has the wrong op kind.",
    message=(
        "block in '{op_name}' region {region_index} must terminate with "
        "'{expected_terminator}' but found '{actual_terminator}'"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("region_index", ParamKind.U32),
        ErrorParam("expected_terminator", ParamKind.STRING),
        ErrorParam("actual_terminator", ParamKind.STRING),
    ),
    fix_hint="Use '{expected_terminator}' as the region terminator",
)

# ERR_STRUCTURE_019: Predicate list payload is missing.
ERR_STRUCTURE_019 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=19,
    severity=Severity.ERROR,
    summary="Predicate list payload is missing.",
    message=(
        "predicate list attribute '{attr_name}' has {predicate_count} "
        "entries but no payload"
    ),
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("predicate_count", ParamKind.U32),
    ),
    fix_hint="Provide a predicate payload for every list entry",
)

# ERR_STRUCTURE_020: Predicate kind is out of range.
ERR_STRUCTURE_020 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=20,
    severity=Severity.ERROR,
    summary="Predicate kind is out of range.",
    message=(
        "predicate list attribute '{attr_name}' entry {predicate_index} has "
        "kind {actual_kind}, but only {kind_count} kinds are defined"
    ),
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("predicate_index", ParamKind.U32),
        ErrorParam("actual_kind", ParamKind.U32),
        ErrorParam("kind_count", ParamKind.U32),
    ),
    fix_hint="Use a predicate kind from the predicate vocabulary",
)

# ERR_STRUCTURE_021: Predicate argument count mismatch.
ERR_STRUCTURE_021 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=21,
    severity=Severity.ERROR,
    summary="Predicate argument count mismatch.",
    message=(
        "predicate list attribute '{attr_name}' entry {predicate_index} "
        "('{predicate_name}') expects {expected_count} arguments, got "
        "{actual_count}"
    ),
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("predicate_index", ParamKind.U32),
        ErrorParam("predicate_name", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
        ErrorParam("actual_count", ParamKind.U32),
    ),
    fix_hint="Use the predicate arity defined by the predicate vocabulary",
)

# ERR_STRUCTURE_022: Predicate argument tag is out of range.
ERR_STRUCTURE_022 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=22,
    severity=Severity.ERROR,
    summary="Predicate argument tag is out of range.",
    message=(
        "predicate list attribute '{attr_name}' entry {predicate_index} "
        "argument {argument_index} has tag {actual_tag}, but only {tag_count} "
        "tags are defined"
    ),
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("predicate_index", ParamKind.U32),
        ErrorParam("argument_index", ParamKind.U32),
        ErrorParam("actual_tag", ParamKind.U32),
        ErrorParam("tag_count", ParamKind.U32),
    ),
    fix_hint="Use a predicate argument tag from the predicate vocabulary",
)

# ERR_STRUCTURE_023: Successor target is missing.
ERR_STRUCTURE_023 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=23,
    severity=Severity.ERROR,
    summary="Successor target is missing.",
    message="successor {successor_index} of '{op_name}' has no target block",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("successor_index", ParamKind.U32),
    ),
    fix_hint="Set every successor field to a block in the op's enclosing region",
)

# ERR_STRUCTURE_024: Successor target is outside the enclosing region.
ERR_STRUCTURE_024 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=24,
    severity=Severity.ERROR,
    summary="Successor target is outside the enclosing region.",
    message=(
        "successor {successor_index} of '{op_name}' targets a block outside "
        "the op's enclosing region"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("successor_index", ParamKind.U32),
    ),
    fix_hint="Branch only to blocks in the same region as the branch op",
)

# ERR_STRUCTURE_025: Successor argument count mismatch.
ERR_STRUCTURE_025 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=25,
    severity=Severity.ERROR,
    summary="Successor argument count mismatch.",
    message=(
        "successor {successor_index} of '{op_name}' passes {actual_count} "
        "arguments, expected {expected_count}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("successor_index", ParamKind.U32),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Pass one value per destination block argument",
)

# ERR_STRUCTURE_026: Successor argument type mismatch.
ERR_STRUCTURE_026 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=26,
    severity=Severity.ERROR,
    summary="Successor argument type mismatch.",
    message=(
        "successor {successor_index} argument {argument_index} of '{op_name}' "
        "has type {actual_type}, expected {expected_type}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("successor_index", ParamKind.U32),
        ErrorParam("argument_index", ParamKind.U32),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_type", ParamKind.TYPE),
    ),
    fix_hint="Pass a value whose type matches the destination block argument",
)

# ERR_STRUCTURE_027: String attribute value violates a structural constraint.
ERR_STRUCTURE_027 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=27,
    severity=Severity.ERROR,
    summary="String attribute value violates a structural constraint.",
    message=(
        "attribute '{attr_name}' value \"{actual_value}\" must satisfy "
        "{expected_constraint}"
    ),
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.STRING),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="Choose a string value satisfying '{expected_constraint}'",
)

# ERR_STRUCTURE_028: Pass callback failed.
ERR_STRUCTURE_028 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=28,
    severity=Severity.ERROR,
    summary="Pass callback failed.",
    message=(
        "pass '{pass_key}' failed while running at {anchor_kind} anchor "
        "on symbol '{symbol_name}' in pipeline '@{pipeline_name}'"
    ),
    params=(
        ErrorParam("pass_key", ParamKind.STRING),
        ErrorParam("anchor_kind", ParamKind.STRING),
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("pipeline_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Inspect the pass-specific diagnostic or run the named pipeline on IR "
        "that satisfies pass '{pass_key}'"
    ),
)

# ERR_STRUCTURE_029: Operation placement constraint violated.
ERR_STRUCTURE_029 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=29,
    severity=Severity.ERROR,
    summary="Operation placement constraint violated.",
    message=(
        "'{op_name}' violates {constraint_kind} ancestor placement for "
        "'{ancestor_name}' (observed ancestor: '{actual_ancestor}')"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("constraint_kind", ParamKind.STRING),
        ErrorParam("ancestor_name", ParamKind.STRING),
        ErrorParam("actual_ancestor", ParamKind.STRING),
    ),
    fix_hint="Move '{op_name}' to a region satisfying its placement contract",
)

# ERR_STRUCTURE_030: Operation order constraint violated.
ERR_STRUCTURE_030 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=30,
    severity=Severity.ERROR,
    summary="Operation order constraint violated.",
    message="'{op_name}' must appear {placement} '{reference_name}'",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("placement", ParamKind.STRING),
        ErrorParam("reference_name", ParamKind.STRING),
    ),
    fix_hint="Move '{op_name}' to the required position in its block",
)

# ERR_STRUCTURE_031: Operation block placement constraint violated.
ERR_STRUCTURE_031 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=31,
    severity=Severity.ERROR,
    summary="Operation block placement constraint violated.",
    message="'{op_name}' must appear in {expected_block}, found {actual_block}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("expected_block", ParamKind.STRING),
        ErrorParam("actual_block", ParamKind.STRING),
    ),
    fix_hint="Move '{op_name}' to the required block",
)

# ERR_STRUCTURE_032: Referenced value origin is structurally invalid.
ERR_STRUCTURE_032 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=32,
    severity=Severity.ERROR,
    summary="Referenced value origin is structurally invalid.",
    message=("'{op_name}' field '{field_name}' must reference {required_origin}"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("required_origin", ParamKind.STRING),
    ),
    fix_hint="Use a value produced by the required structural origin",
)

# ERR_STRUCTURE_033: Referenced value owner does not match.
ERR_STRUCTURE_033 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=33,
    severity=Severity.ERROR,
    summary="Referenced value owner does not match.",
    message=(
        "'{op_name}' field '{field_name}' references a value owned by a "
        "different {owner_kind}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("owner_kind", ParamKind.STRING),
    ),
    fix_hint="Use a value owned by the same enclosing operation",
)

# ERR_STRUCTURE_034: Pure call targets an impure callee.
ERR_STRUCTURE_034 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=34,
    severity=Severity.ERROR,
    summary="Pure call targets an impure callee.",
    message=(
        "pure call callee '@{callee_name}' through '{boundary_name}' does not "
        "have a pure contract"
    ),
    params=(
        ErrorParam("callee_name", ParamKind.STRING),
        ErrorParam("boundary_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Remove the pure marker, mark the direct callee pure after proving its "
        "body has no observable effects, or route the call through a boundary "
        "whose contract is explicitly pure"
    ),
)

# ERR_STRUCTURE_035: Counted loop step is not proven positive.
ERR_STRUCTURE_035 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=35,
    severity=Severity.ERROR,
    summary="Counted loop step is not proven positive.",
    message="{pass_name} requires {op_name} step to be fact-proven positive",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the loop step facts or normalize the loop before converting "
        "structured control flow to CFG"
    ),
)

# ERR_STRUCTURE_036: CFG conversion cannot preserve tied result ownership.
ERR_STRUCTURE_036 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=36,
    severity=Severity.ERROR,
    summary="CFG conversion cannot preserve tied result ownership.",
    message="{pass_name} cannot preserve tied result ownership on {op_name}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Lower ownership transfers before CFG conversion or keep the "
        "structured op until CFG block arguments can model the transfer"
    ),
)

# ERR_STRUCTURE_037: Low packet descriptor ordinal conflicts with target.
ERR_STRUCTURE_037 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=37,
    severity=Severity.ERROR,
    summary="Low packet descriptor ordinal conflicts with target.",
    message=(
        "low function '{function_name}' packet '{packet_key}' stores descriptor "
        "ordinal {descriptor_ordinal}, which resolves to '{descriptor_key}' in "
        "target contract '{descriptor_set_key}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("packet_key", ParamKind.STRING),
        ErrorParam("descriptor_ordinal", ParamKind.U32),
        ErrorParam("descriptor_key", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Regenerate the packet for the selected target contract or leave the "
        "descriptor ordinal unresolved so the verifier can resolve the key"
    ),
)

# ERR_STRUCTURE_038: Workgroup barrier is under divergent control.
ERR_STRUCTURE_038 = ErrorDef(
    domain=ErrorDomain.STRUCTURE,
    code=38,
    severity=Severity.ERROR,
    summary="Workgroup barrier is under divergent control.",
    message=(
        "'{op_name}' with workgroup scope is control-dependent on "
        "lane-varying {control_kind} '{control_value}' from '{control_op_name}'"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("control_kind", ParamKind.STRING),
        ErrorParam("control_value", ParamKind.STRING),
        ErrorParam("control_op_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Move the workgroup barrier outside lane-varying control and guard "
        "only the memory access or final store"
    ),
)

ALL_STRUCTURE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_STRUCTURE_001,
    ERR_STRUCTURE_002,
    ERR_STRUCTURE_003,
    ERR_STRUCTURE_004,
    ERR_STRUCTURE_005,
    ERR_STRUCTURE_006,
    ERR_STRUCTURE_007,
    ERR_STRUCTURE_008,
    ERR_STRUCTURE_009,
    ERR_STRUCTURE_010,
    ERR_STRUCTURE_011,
    ERR_STRUCTURE_012,
    ERR_STRUCTURE_013,
    ERR_STRUCTURE_014,
    ERR_STRUCTURE_015,
    ERR_STRUCTURE_016,
    ERR_STRUCTURE_017,
    ERR_STRUCTURE_018,
    ERR_STRUCTURE_019,
    ERR_STRUCTURE_020,
    ERR_STRUCTURE_021,
    ERR_STRUCTURE_022,
    ERR_STRUCTURE_023,
    ERR_STRUCTURE_024,
    ERR_STRUCTURE_025,
    ERR_STRUCTURE_026,
    ERR_STRUCTURE_027,
    ERR_STRUCTURE_028,
    ERR_STRUCTURE_029,
    ERR_STRUCTURE_030,
    ERR_STRUCTURE_031,
    ERR_STRUCTURE_032,
    ERR_STRUCTURE_033,
    ERR_STRUCTURE_034,
    ERR_STRUCTURE_035,
    ERR_STRUCTURE_036,
    ERR_STRUCTURE_037,
    ERR_STRUCTURE_038,
)
