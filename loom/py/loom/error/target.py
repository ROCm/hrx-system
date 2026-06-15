# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TARGET domain — shared target contract and legality diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

_TARGET_CONTEXT_PARAMS = (
    ErrorParam("target_key", ParamKind.STRING),
    ErrorParam("export_name", ParamKind.STRING),
    ErrorParam("config_key", ParamKind.STRING),
    ErrorParam("function_name", ParamKind.STRING),
    ErrorParam("op_name", ParamKind.STRING),
)

# ERR_TARGET_001: Target has no lowering contract for an operation.
ERR_TARGET_001 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=1,
    severity=Severity.ERROR,
    summary="Target has no lowering contract for an operation.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "has no target-low contract for '{op_name}' in '@{function_name}'"
    ),
    params=_TARGET_CONTEXT_PARAMS,
    fix_hint=(
        "Select a target contract that handles '{op_name}' or refine the "
        "operation before target-low lowering"
    ),
)

# ERR_TARGET_002: Target contract rejected a value type constraint.
ERR_TARGET_002 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=2,
    severity=Severity.ERROR,
    summary="Target contract rejected a value type constraint.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "type {actual_type} does not match {expected_type}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_type", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine '{field_name}' to a type accepted by the selected target contract"
    ),
)

# ERR_TARGET_003: Target contract guard constraint is not satisfied.
ERR_TARGET_003 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=3,
    severity=Severity.ERROR,
    summary="Target contract guard constraint is not satisfied.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {subject_role} '{subject_name}' in "
        "'@{function_name}': constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("subject_role", ParamKind.STRING),
        ErrorParam("subject_name", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_004: Target contract rejected a count constraint.
ERR_TARGET_004 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=4,
    severity=Severity.ERROR,
    summary="Target contract rejected a count constraint.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {subject_role} '{subject_name}' in "
        "'@{function_name}': constraint '{constraint_key}' requires count "
        "{expected_count}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("subject_role", ParamKind.STRING),
        ErrorParam("subject_name", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_TARGET_005: Target contract rejected a range constraint.
ERR_TARGET_005 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=5,
    severity=Severity.ERROR,
    summary="Target contract rejected a range constraint.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {subject_role} '{subject_name}' in "
        "'@{function_name}': constraint '{constraint_key}' requires range "
        "[{minimum}, {maximum}]"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("subject_role", ParamKind.STRING),
        ErrorParam("subject_name", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.I64),
        ErrorParam("maximum", ParamKind.I64),
    ),
)

# ERR_TARGET_006: Target contract rejected an element range constraint.
ERR_TARGET_006 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=6,
    severity=Severity.ERROR,
    summary="Target contract rejected an element range constraint.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {subject_role} '{subject_name}' element "
        "{element} in '@{function_name}': constraint '{constraint_key}' "
        "requires range [{minimum}, {maximum}]"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("subject_role", ParamKind.STRING),
        ErrorParam("subject_name", ParamKind.STRING),
        ErrorParam("element", ParamKind.U32),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.I64),
        ErrorParam("maximum", ParamKind.I64),
    ),
)

# ERR_TARGET_007: Target contract rejected a relation constraint.
ERR_TARGET_007 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=7,
    severity=Severity.ERROR,
    summary="Target contract rejected a relation constraint.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {subject_role} '{subject_name}' and "
        "'{other_subject_name}' in '@{function_name}': constraint "
        "'{constraint_key}' was not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("subject_role", ParamKind.STRING),
        ErrorParam("subject_name", ParamKind.STRING),
        ErrorParam("other_subject_name", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_008: Target contract rejected a source memory access.
ERR_TARGET_008 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=8,
    severity=Severity.ERROR,
    summary="Target contract rejected a source memory access.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' source memory access in '@{function_name}': "
        "operation '{operation_kind}' does not match the selected contract"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("operation_kind", ParamKind.STRING),
    ),
)

# ERR_TARGET_009: Target pipeline entry has no target record.
ERR_TARGET_009 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=9,
    severity=Severity.ERROR,
    summary="Target pipeline entry has no target record.",
    message=(
        "target pipeline '{pipeline_name}' entry '@{function_name}' must "
        "declare a target record"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Attach a target record compatible with pipeline '{pipeline_name}' "
        "to '@{function_name}'"
    ),
)

# ERR_TARGET_010: Target pipeline rejected the selected target bundle.
ERR_TARGET_010 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=10,
    severity=Severity.ERROR,
    summary="Target pipeline rejected the selected target bundle.",
    message=(
        "target pipeline '{pipeline_name}' cannot compile '@{function_name}' "
        "with target '{target_key}' export '{export_name}' config "
        "'{config_key}' ({codegen_format}/{artifact_format}/{abi_kind})"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("codegen_format", ParamKind.STRING),
        ErrorParam("artifact_format", ParamKind.STRING),
        ErrorParam("abi_kind", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a target record/export whose artifact kind is consumed by "
        "pipeline '{pipeline_name}'"
    ),
)

# ERR_TARGET_011: Target pipeline found no compatible entry.
ERR_TARGET_011 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=11,
    severity=Severity.ERROR,
    summary="Target pipeline found no compatible entry.",
    message=(
        "module contains no function with a target record compatible with "
        "target pipeline '{pipeline_name}'"
    ),
    params=(ErrorParam("pipeline_name", ParamKind.STRING),),
    fix_hint=(
        "Select an entry symbol explicitly or add a target record compatible "
        "with pipeline '{pipeline_name}'"
    ),
)

# ERR_TARGET_012: Target pipeline found multiple compatible entries.
ERR_TARGET_012 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=12,
    severity=Severity.ERROR,
    summary="Target pipeline found multiple compatible entries.",
    message=(
        "module contains {candidate_count} functions compatible with target "
        "pipeline '{pipeline_name}'"
    ),
    params=(
        ErrorParam("pipeline_name", ParamKind.STRING),
        ErrorParam("candidate_count", ParamKind.U32),
    ),
    fix_hint="Select one entry symbol explicitly for pipeline '{pipeline_name}'",
)

# ERR_TARGET_021: Function ABI override field is unsupported.
ERR_TARGET_021 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=21,
    severity=Severity.ERROR,
    summary="Function ABI override field is unsupported.",
    message=(
        "function '@{function_name}' ABI override field '{field_name}' is not supported"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
    ),
)

# ERR_TARGET_022: Function target contract constraint is not satisfied.
ERR_TARGET_022 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=22,
    severity=Severity.ERROR,
    summary="Function target contract constraint is not satisfied.",
    message=(
        "function '@{function_name}' target record '@{target_name}' does not "
        "satisfy target contract constraint '{constraint_key}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_023: HAL kernel flat workgroup range constraint is not satisfied.
ERR_TARGET_023 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=23,
    severity=Severity.ERROR,
    summary="HAL kernel flat workgroup range constraint is not satisfied.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' flat "
        "workgroup range [{minimum}, {maximum}] does not satisfy "
        "'{constraint_key}' with target limit {limit}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.U32),
        ErrorParam("maximum", ParamKind.U32),
        ErrorParam("limit", ParamKind.U32),
    ),
)

# ERR_TARGET_024: HAL kernel required workgroup dimension exceeds target limit.
ERR_TARGET_024 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=24,
    severity=Severity.ERROR,
    summary="HAL kernel required workgroup dimension exceeds target limit.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "required workgroup {axis} size {size} exceeds target limit {limit}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("axis", ParamKind.STRING),
        ErrorParam("size", ParamKind.U32),
        ErrorParam("limit", ParamKind.U32),
    ),
)

# ERR_TARGET_025: HAL kernel required flat workgroup size constraint is not satisfied.
ERR_TARGET_025 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=25,
    severity=Severity.ERROR,
    summary="HAL kernel required flat workgroup size constraint is not satisfied.",
    message=(
        "kernel entry '@{function_name}' target record '@{target_name}' "
        "required flat workgroup size {size} does not satisfy "
        "'{constraint_key}' with range [{minimum}, {maximum}] and target "
        "limit {limit}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("size", ParamKind.U64),
        ErrorParam("minimum", ParamKind.U32),
        ErrorParam("maximum", ParamKind.U32),
        ErrorParam("limit", ParamKind.U32),
    ),
)

# ERR_TARGET_026: Function has no target record.
ERR_TARGET_026 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=26,
    severity=Severity.ERROR,
    summary="Function has no target record.",
    message="function '@{function_name}' must declare a target record",
    params=(ErrorParam("function_name", ParamKind.STRING),),
)

# ERR_TARGET_027: Target-low ABI policy did not map a value.
ERR_TARGET_027 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=27,
    severity=Severity.ERROR,
    summary="Target-low ABI policy did not map a value.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' {value_kind} value {value_id} in "
        "'@{function_name}': target-low ABI policy did not produce a "
        "register type"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("value_kind", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U64),
    ),
)

# ERR_TARGET_028: Function predicates reached target-low lowering.
ERR_TARGET_028 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=28,
    severity=Severity.ERROR,
    summary="Function predicates reached target-low lowering.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': {predicate_count} "
        "function predicate(s) require value remapping before target-low "
        "lowering"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("predicate_count", ParamKind.U32),
    ),
)

# ERR_TARGET_029: Tied function results reached target-low lowering.
ERR_TARGET_029 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=29,
    severity=Severity.ERROR,
    summary="Tied function results reached target-low lowering.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': {tied_result_count} "
        "tied result(s) require explicit ABI ownership lowering"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("tied_result_count", ParamKind.U32),
    ),
)

# ERR_TARGET_030: Operation with regions reached target-low source lowering.
ERR_TARGET_030 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=30,
    severity=Severity.ERROR,
    summary="Operation with regions reached target-low source lowering.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': {region_count} nested "
        "region(s) must be lowered away before target-low source lowering"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("region_count", ParamKind.U32),
    ),
)

# ERR_TARGET_031: Target-low type constraint is not satisfied.
ERR_TARGET_031 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=31,
    severity=Severity.ERROR,
    summary="Target-low type constraint is not satisfied.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' type {actual_type} in '@{function_name}': "
        "type constraint '{type_constraint}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("type_constraint", ParamKind.STRING),
    ),
)

# ERR_TARGET_032: Target-low operation constraint is not satisfied.
ERR_TARGET_032 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=32,
    severity=Severity.ERROR,
    summary="Target-low operation constraint is not satisfied.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': operation constraint "
        "'{op_constraint}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("op_constraint", ParamKind.STRING),
    ),
)

# ERR_TARGET_033: Target-low policy has no source value type mapping.
ERR_TARGET_033 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=33,
    severity=Severity.ERROR,
    summary="Target-low policy has no source value type mapping.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field '{field_name}' in '@{function_name}': "
        "source type {actual_type} has no target-low register mapping in the "
        "selected policy"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

# ERR_TARGET_034: Target branch lowering constraint is not satisfied.
ERR_TARGET_034 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=34,
    severity=Severity.ERROR,
    summary="Target branch lowering constraint is not satisfied.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' in '@{function_name}': branch lowering "
        "constraint '{branch_constraint}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("branch_constraint", ParamKind.STRING),
    ),
)

# ERR_TARGET_035: Target branch condition type constraint is not satisfied.
ERR_TARGET_035 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=35,
    severity=Severity.ERROR,
    summary="Target branch condition type constraint is not satisfied.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' field 'condition' in '@{function_name}': "
        "type {actual_type} does not satisfy '{type_constraint}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("type_constraint", ParamKind.STRING),
    ),
)

# ERR_TARGET_036: Target has no projection for an emitter.
ERR_TARGET_036 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=36,
    severity=Severity.ERROR,
    summary="Target has no projection for an emitter.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "has no '{emitter_key}' projection for codegen format "
        "'{codegen_format}' and ABI '{abi_kind}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("emitter_key", ParamKind.STRING),
        ErrorParam("codegen_format", ParamKind.STRING),
        ErrorParam("abi_kind", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a target with a '{emitter_key}' projection or link a provider "
        "that projects this target record"
    ),
)

# ERR_TARGET_037: Target emitter preflight constraint failed.
ERR_TARGET_037 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=37,
    severity=Severity.ERROR,
    summary="Target emitter preflight constraint failed.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot run emitter '{emitter_key}': legality code '{legality_code}' "
        "failed constraint '{constraint_key}' for subject '{subject_key}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("emitter_key", ParamKind.STRING),
        ErrorParam("legality_code", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("subject_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_038: Target emitter rejected an operation.
ERR_TARGET_038 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=38,
    severity=Severity.ERROR,
    summary="Target emitter rejected an operation.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "emitter '{emitter_key}' provider '{provider_name}' rejected "
        "'{op_name}': legality code '{legality_code}' failed constraint "
        "'{constraint_key}' for subject '{subject_key}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("emitter_key", ParamKind.STRING),
        ErrorParam("provider_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("legality_code", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("subject_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_039: Matrix source contract did not satisfy target constraints.
ERR_TARGET_039 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=39,
    severity=Severity.ERROR,
    summary="Matrix source contract did not satisfy target constraints.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' matrix contract source '{source_contract}' in "
        "'@{function_name}': constraint '{matrix_constraint}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("source_contract", ParamKind.STRING),
        ErrorParam("matrix_constraint", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine matrix fragment roles, shapes, schemas, numeric formats, and "
        "capability facts before target matrix lowering"
    ),
)

# ERR_TARGET_040: Low function call target does not match the caller target.
ERR_TARGET_040 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=40,
    severity=Severity.ERROR,
    summary="Low function call target does not match the caller target.",
    message=(
        "'{op_name}' field '{field_name}' calls a low function targeting "
        "'@{callee_target_name}' from an entry targeting '@{caller_target_name}'"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("callee_target_name", ParamKind.STRING),
        ErrorParam("caller_target_name", ParamKind.STRING),
    ),
    fix_hint="Call a low function owned by the same target as the enclosing entry.",
)

# ERR_TARGET_041: Low resource import kind conflicts with function ABI.
ERR_TARGET_041 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=41,
    severity=Severity.ERROR,
    summary="Low resource import kind conflicts with function ABI.",
    message=(
        "'{op_name}' field '{field_name}' imports {import_kind}, requiring "
        "ABI '{expected_abi}', but the enclosing function ABI is '{actual_abi}'"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("import_kind", ParamKind.STRING),
        ErrorParam("expected_abi", ParamKind.STRING),
        ErrorParam("actual_abi", ParamKind.STRING),
    ),
    fix_hint="Use a resource import kind accepted by the enclosing function ABI.",
)

# ERR_TARGET_042: Low register type has no target descriptor register class.
ERR_TARGET_042 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=42,
    severity=Severity.ERROR,
    summary="Low register type has no target descriptor register class.",
    message=(
        "low function '@{function_name}' {value_kind} '{value_name}' has "
        "type {actual_type}, whose register class is not defined by descriptor "
        "set '{descriptor_set}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_kind", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("descriptor_set", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a register class declared by descriptor set '{descriptor_set}' "
        "or select a target config whose descriptor set owns this ABI type"
    ),
)

# ERR_TARGET_043: Low ABI register unit count exceeds target descriptor class.
ERR_TARGET_043 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=43,
    severity=Severity.ERROR,
    summary="Low ABI register unit count exceeds target descriptor class.",
    message=(
        "low function '@{function_name}' {value_kind} '{value_name}' has "
        "type {actual_type}, requiring {unit_count} allocatable unit(s), but "
        "descriptor set '{descriptor_set}' provides only {allocatable_count}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_kind", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("unit_count", ParamKind.U32),
        ErrorParam("descriptor_set", ParamKind.STRING),
        ErrorParam("allocatable_count", ParamKind.U32),
    ),
    fix_hint=(
        "Use a smaller ABI register value or select a descriptor set whose "
        "register class has enough allocatable units."
    ),
)

# ERR_TARGET_044: Low descriptor set is not available.
ERR_TARGET_044 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=44,
    severity=Severity.ERROR,
    summary="Low descriptor set is not available.",
    message=(
        "low function '@{function_name}' target '@{target_name}' requires "
        "descriptor set '{descriptor_set_key}', but the descriptor registry "
        "does not provide it"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint="Link the descriptor package named by '{descriptor_set_key}' or "
    "choose a target config whose descriptor set is available",
)

# ERR_TARGET_045: Low descriptor is not available.
ERR_TARGET_045 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=45,
    severity=Severity.ERROR,
    summary="Low descriptor is not available.",
    message=(
        "low function '@{function_name}' uses descriptor '{opcode}', but "
        "descriptor set '{descriptor_set_key}' does not define it"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint="Choose an opcode from descriptor set '{descriptor_set_key}' or "
    "select a target config with a descriptor set that defines '{opcode}'",
)

# ERR_TARGET_046: Low descriptor feature is not enabled.
ERR_TARGET_046 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=46,
    severity=Severity.ERROR,
    summary="Low descriptor feature is not enabled.",
    message=(
        "low function '@{function_name}' uses descriptor '{opcode}' from set "
        "'{descriptor_set_key}', but feature word {feature_word_index} is "
        "missing bits {missing_feature_bits}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
        ErrorParam("feature_word_index", ParamKind.U32),
        ErrorParam("missing_feature_bits", ParamKind.U64),
    ),
    fix_hint="Enable the missing target feature bits or choose a descriptor "
    "legal for the selected target config",
)

# ERR_TARGET_047: Low descriptor immediate attribute is missing.
ERR_TARGET_047 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=47,
    severity=Severity.ERROR,
    summary="Low descriptor immediate attribute is missing.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' requires "
        "immediate attribute '{immediate_name}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("immediate_name", ParamKind.STRING),
    ),
    fix_hint="Provide attribute '{immediate_name}' in the low packet attrs "
    "for descriptor '{opcode}'",
)

# ERR_TARGET_048: Low descriptor immediate attribute is not declared.
ERR_TARGET_048 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=48,
    severity=Severity.ERROR,
    summary="Low descriptor immediate attribute is not declared.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' has "
        "attribute '{attr_name}', but the descriptor declares no such "
        "immediate"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("attr_name", ParamKind.STRING),
    ),
    fix_hint="Remove attribute '{attr_name}' or add a descriptor immediate "
    "row that owns it",
)

# ERR_TARGET_049: Low descriptor immediate attribute kind is invalid.
ERR_TARGET_049 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=49,
    severity=Severity.ERROR,
    summary="Low descriptor immediate attribute kind is invalid.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' immediate "
        "'{immediate_name}' has attribute kind {actual_kind}, expected "
        "{expected_kind}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("immediate_name", ParamKind.STRING),
        ErrorParam("actual_kind", ParamKind.U32),
        ErrorParam("expected_kind", ParamKind.STRING),
    ),
    fix_hint="Encode immediate '{immediate_name}' using {expected_kind}",
)

# ERR_TARGET_050: Target contract requires stronger source memory alignment.
ERR_TARGET_050 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=50,
    severity=Severity.ERROR,
    summary="Target contract requires stronger source memory alignment.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected '{op_name}' source memory access in '@{function_name}': "
        "access to {value_type} requires {required_alignment}-byte alignment, "
        "but source facts prove {known_alignment}-byte alignment"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("value_type", ParamKind.TYPE),
        ErrorParam("required_alignment", ParamKind.U32),
        ErrorParam("known_alignment", ParamKind.U32),
    ),
)

# ERR_TARGET_051: Low workgroup storage use exceeds target limit.
ERR_TARGET_051 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=51,
    severity=Severity.ERROR,
    summary="Low workgroup storage use exceeds target limit.",
    message=(
        "low function '@{function_name}' target '@{target_name}' reserves "
        "{reserved_bytes} byte(s) of workgroup storage, exceeding target limit "
        "{limit_bytes}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("reserved_bytes", ParamKind.U64),
        ErrorParam("limit_bytes", ParamKind.U64),
    ),
    fix_hint=(
        "Reduce workgroup storage reservations or select a target profile with "
        "a larger workgroup storage limit."
    ),
)

# ERR_TARGET_052: Function target conflicts with invocation target.
ERR_TARGET_052 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=52,
    severity=Severity.ERROR,
    summary="Function target conflicts with invocation target.",
    message=(
        "function '@{function_name}' target record '@{authored_target_name}' "
        "is incompatible with invocation target '{selected_target_name}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("authored_target_name", ParamKind.STRING),
        ErrorParam("selected_target_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Compile this function with a compatible target selection or remove "
        "the explicit target record from the source function."
    ),
)

# ERR_TARGET_053: Target emitter has no low descriptor mapping.
ERR_TARGET_053 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=53,
    severity=Severity.ERROR,
    summary="Target emitter has no low descriptor mapping.",
    message=(
        "low function '@{function_name}' descriptor '{opcode}' from set "
        "'{descriptor_set_key}' has no emitter mapping for '{emitter_key}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("opcode", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
        ErrorParam("emitter_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a descriptor supported by '{emitter_key}' or add an emitter "
        "mapping for '{opcode}'"
    ),
)

# ERR_TARGET_054: Target emitter cannot represent a low function shape.
ERR_TARGET_054 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=54,
    severity=Severity.ERROR,
    summary="Target emitter cannot represent a low function shape.",
    message=(
        "low function '@{function_name}' cannot be emitted by "
        "'{emitter_key}': {subject_kind} count is {actual_count}, expected "
        "{expected_count}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("emitter_key", ParamKind.STRING),
        ErrorParam("subject_kind", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
)

# ERR_TARGET_055: Target emitter cannot represent a low descriptor set.
ERR_TARGET_055 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=55,
    severity=Severity.ERROR,
    summary="Target emitter cannot represent a low descriptor set.",
    message=(
        "low function '@{function_name}' descriptor set "
        "'{descriptor_set_key}' cannot be emitted by '{emitter_key}', "
        "expected '{expected_descriptor_set_key}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
        ErrorParam("emitter_key", ParamKind.STRING),
        ErrorParam("expected_descriptor_set_key", ParamKind.STRING),
    ),
)

# ERR_TARGET_056: Target emitter cannot represent a low value type.
ERR_TARGET_056 = ErrorDef(
    domain=ErrorDomain.TARGET,
    code=56,
    severity=Severity.ERROR,
    summary="Target emitter cannot represent a low value type.",
    message=(
        "low function '@{function_name}' cannot emit {value_kind} "
        "'{value_name}' with type {actual_type} through '{emitter_key}', "
        "expected '{expected_constraint}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_kind", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("emitter_key", ParamKind.STRING),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
)

ALL_TARGET_ERRORS = (
    ERR_TARGET_001,
    ERR_TARGET_002,
    ERR_TARGET_003,
    ERR_TARGET_004,
    ERR_TARGET_005,
    ERR_TARGET_006,
    ERR_TARGET_007,
    ERR_TARGET_008,
    ERR_TARGET_009,
    ERR_TARGET_010,
    ERR_TARGET_011,
    ERR_TARGET_012,
    ERR_TARGET_021,
    ERR_TARGET_022,
    ERR_TARGET_023,
    ERR_TARGET_024,
    ERR_TARGET_025,
    ERR_TARGET_026,
    ERR_TARGET_027,
    ERR_TARGET_028,
    ERR_TARGET_029,
    ERR_TARGET_030,
    ERR_TARGET_031,
    ERR_TARGET_032,
    ERR_TARGET_033,
    ERR_TARGET_034,
    ERR_TARGET_035,
    ERR_TARGET_036,
    ERR_TARGET_037,
    ERR_TARGET_038,
    ERR_TARGET_039,
    ERR_TARGET_040,
    ERR_TARGET_041,
    ERR_TARGET_042,
    ERR_TARGET_043,
    ERR_TARGET_044,
    ERR_TARGET_045,
    ERR_TARGET_046,
    ERR_TARGET_047,
    ERR_TARGET_048,
    ERR_TARGET_049,
    ERR_TARGET_050,
    ERR_TARGET_051,
    ERR_TARGET_052,
    ERR_TARGET_053,
    ERR_TARGET_054,
    ERR_TARGET_055,
    ERR_TARGET_056,
)
