# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LOWERING domain — pass legality and unsupported mappings."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_LOWERING_022: Kernel async group has no wait in the current stream.
ERR_LOWERING_022 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=22,
    severity=Severity.ERROR,
    summary="Kernel async group has no wait in the current stream.",
    message=(
        "{phase_name} requires {op_name} to be waited in the current "
        "straight-line async stream"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the async group in the same straight-line stream or lower it "
        "with a pipeline-aware async strategy"
    ),
)

# ERR_LOWERING_023: Kernel async group is carried outside the stream.
ERR_LOWERING_023 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=23,
    severity=Severity.ERROR,
    summary="Kernel async group is carried outside the stream.",
    message=(
        "{phase_name} cannot lower {op_name} whose group value has a non-wait use"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Keep async groups in a straight-line group/wait stream or run a "
        "pipeline-aware legality path before lowering"
    ),
)

# ERR_LOWERING_024: Kernel async movement cannot be described.
ERR_LOWERING_024 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=24,
    severity=Severity.ERROR,
    summary="Kernel async movement cannot be described.",
    message=(
        "{phase_name} cannot describe the movement endpoints for {op_name}; "
        "movement rejection bits are {rejection_bits}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("rejection_bits", ParamKind.U64),
    ),
    fix_hint=(
        "Refine async transfer operands so their source and destination view "
        "regions can be described by movement analysis"
    ),
)

# ERR_LOWERING_025: Kernel async token producer is not an async view movement.
ERR_LOWERING_025 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=25,
    severity=Severity.ERROR,
    summary="Kernel async token producer is not an async view movement.",
    message=(
        "{phase_name} requires the token producer for {op_name} to be an async "
        "view movement"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Group only tokens produced by kernel async transfer operations whose "
        "destination is a view"
    ),
)

# ERR_LOWERING_026: Kernel async destination overlaps a pending destination.
ERR_LOWERING_026 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=26,
    severity=Severity.ERROR,
    summary="Kernel async destination overlaps a pending destination.",
    message=(
        "{phase_name} found {op_name} whose destination may overlap an earlier "
        "uncompleted async destination"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the earlier async group before issuing an overlapping async "
        "destination or prove the destination views are disjoint"
    ),
)

# ERR_LOWERING_027: Synchronous write overlaps a pending async destination.
ERR_LOWERING_027 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=27,
    severity=Severity.ERROR,
    summary="Synchronous write overlaps a pending async destination.",
    message=(
        "{phase_name} found {op_name} writing a view that may overlap a "
        "pending async destination before wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the pending async group before the synchronous write or prove the "
        "written view is disjoint"
    ),
)

# ERR_LOWERING_028: Synchronous read overlaps a pending async destination.
ERR_LOWERING_028 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=28,
    severity=Severity.ERROR,
    summary="Synchronous read overlaps a pending async destination.",
    message=(
        "{phase_name} found {op_name} reading a view that may observe a "
        "pending async destination before wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the pending async group before the synchronous read or prove the "
        "read view is disjoint"
    ),
)

# ERR_LOWERING_029: Kernel async wait references an uncommitted group.
ERR_LOWERING_029 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=29,
    severity=Severity.ERROR,
    summary="Kernel async wait references an uncommitted group.",
    message=(
        "{phase_name} requires {op_name} to wait a group committed in the "
        "current straight-line async stream"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Move the wait after the matching kernel.async.group in the same block "
        "or use a pipeline-aware async lowering path"
    ),
)

# ERR_LOWERING_030: Kernel async wait references an already completed group.
ERR_LOWERING_030 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=30,
    severity=Severity.ERROR,
    summary="Kernel async wait references an already completed group.",
    message=(
        "{phase_name} found {op_name} waiting an async group already completed "
        "by an earlier wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Remove the duplicate wait or wait a younger group that is still outstanding"
    ),
)

# ERR_LOWERING_031: Kernel async wait count does not match stream depth.
ERR_LOWERING_031 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=31,
    severity=Severity.ERROR,
    summary="Kernel async wait count does not match stream depth.",
    message=(
        "{phase_name} found {op_name} with newer_groups {actual_newer_groups}, "
        "but {expected_newer_groups} younger async groups remain outstanding"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("actual_newer_groups", ParamKind.I64),
        ErrorParam("expected_newer_groups", ParamKind.U64),
    ),
    fix_hint=(
        "Set newer_groups to the number of younger uncompleted groups that "
        "remain after this wait"
    ),
)

# ERR_LOWERING_032: Kernel async group leaves a block before wait.
ERR_LOWERING_032 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=32,
    severity=Severity.ERROR,
    summary="Kernel async group leaves a block before wait.",
    message=("{phase_name} requires {op_name} to be waited before leaving its block"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the async group along every control-flow path before leaving the "
        "block or move the async stream into a pipeline-aware region"
    ),
)

# ERR_LOWERING_033: Kernel async group token has no producer op.
ERR_LOWERING_033 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=33,
    severity=Severity.ERROR,
    summary="Kernel async group token has no producer op.",
    message=(
        "{phase_name} requires every token operand of {op_name} to be produced "
        "by a local async transfer op"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Commit only tokens produced in the current function by kernel async "
        "transfer operations"
    ),
)

# ERR_LOWERING_034: Math legalization policy is unavailable.
ERR_LOWERING_034 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=34,
    severity=Severity.ERROR,
    summary="Math legalization policy is unavailable.",
    message=(
        "{phase_name} requires a target math policy for {contract_key} before "
        "legalizing {op_name}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("contract_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Attach a target with a linked math policy or run legalize-math only "
        "after target selection"
    ),
)

# ERR_LOWERING_035: Math legalization recipe is unavailable.
ERR_LOWERING_035 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=35,
    severity=Severity.ERROR,
    summary="Math legalization recipe is unavailable.",
    message=(
        "{phase_name} selected recipe {recipe_name} for {op_name} through "
        "{policy_name}, but no recipe row handles {math_op} in {lane_domain} "
        "lanes of {element_type} under {constraint_key}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("policy_name", ParamKind.STRING),
        ErrorParam("recipe_name", ParamKind.STRING),
        ErrorParam("math_op", ParamKind.STRING),
        ErrorParam("lane_domain", ParamKind.STRING),
        ErrorParam("element_type", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Link the selected math recipe shard or change the target policy to "
        "keep or reject the operation"
    ),
)

# ERR_LOWERING_036: Math legalization policy rejected an operation.
ERR_LOWERING_036 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=36,
    severity=Severity.ERROR,
    summary="Math legalization policy rejected an operation.",
    message=(
        "{phase_name} policy {policy_name} rejected {op_name} for {math_op} "
        "in {lane_domain} lanes of {element_type} under {constraint_key}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("policy_name", ParamKind.STRING),
        ErrorParam("math_op", ParamKind.STRING),
        ErrorParam("lane_domain", ParamKind.STRING),
        ErrorParam("element_type", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a supported variant, add the required fast-math permission, or "
        "implement a stricter target recipe for the reported constraint"
    ),
)

# ERR_LOWERING_037: Write overlaps a pending async source.
ERR_LOWERING_037 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=37,
    severity=Severity.ERROR,
    summary="Write overlaps a pending async source.",
    message=(
        "{phase_name} found {op_name} writing a view that may overlap a "
        "pending async source before wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the pending async group before the write or prove the written "
        "view is disjoint from every pending source"
    ),
)

# ERR_LOWERING_038: Kernel async group does not commit the current transfers.
ERR_LOWERING_038 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=38,
    severity=Severity.ERROR,
    summary="Kernel async group does not commit the current transfers.",
    message=(
        "{phase_name} requires {op_name} to commit the exact uncommitted "
        "transfer token sequence in program order"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Commit every transfer token since the previous group exactly once "
        "and in source program order"
    ),
)

# ERR_LOWERING_039: Kernel async transfer leaves a block before group commit.
ERR_LOWERING_039 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=39,
    severity=Severity.ERROR,
    summary="Kernel async transfer leaves a block before group commit.",
    message=(
        "{phase_name} requires {op_name} to be committed to a local async "
        "group before leaving its block"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Commit the transfer token to the next kernel.async.group in the same "
        "straight-line block"
    ),
)

# ERR_LOWERING_040: Kernel async transfer token escapes the local stream.
ERR_LOWERING_040 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=40,
    severity=Severity.ERROR,
    summary="Kernel async transfer token escapes the local stream.",
    message=(
        "{phase_name} requires the token produced by {op_name} to have exactly "
        "one local kernel.async.group use"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Use the transfer token exactly once in the next local kernel.async.group"
    ),
)

# ERR_LOWERING_043: Boundary fact refinement did not converge.
ERR_LOWERING_043 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=43,
    severity=Severity.ERROR,
    summary="Boundary fact refinement did not converge.",
    message=(
        "{pass_name} did not converge boundary facts in {max_iterations} iteration(s)"
    ),
    params=(
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("max_iterations", ParamKind.U32),
    ),
    fix_hint=(
        "Specialize recursive/SCC summaries or raise the pass iteration limit "
        "only after proving additional iterations are bounded"
    ),
)

# ERR_LOWERING_044: Required callable inline failed.
ERR_LOWERING_044 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=44,
    severity=Severity.ERROR,
    summary="Required callable inline failed.",
    message=("{phase_name} cannot inline {op_name} to @{callee_name}: {failure_code}"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("callee_name", ParamKind.STRING),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
    fix_hint=(
        "Remove the required inline policy, provide an inlineable same-module "
        "body, or resolve the reported policy conflict"
    ),
)

# ERR_LOWERING_045: Template application resolution failed.
ERR_LOWERING_045 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=45,
    severity=Severity.ERROR,
    summary="Template application resolution failed.",
    message=(
        "{phase_name} cannot resolve {op_name} against template family "
        "<{contract_key}>: {failure_code}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("contract_key", ParamKind.STRING),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
    fix_hint=(
        "Add a matching template.def provider, make target, caller context, "
        "and value predicates resolvable before final selection, or correct "
        "the constraints on an exact template.call"
    ),
)

# ERR_LOWERING_046: Sanitizer observation cannot represent memory operation.
ERR_LOWERING_046 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=46,
    severity=Severity.ERROR,
    summary="Sanitizer observation cannot represent memory operation.",
    message=("{phase_name} cannot observe {op_name}: {reason}"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Lower the memory operation to sanitizer-observable scalar accesses "
        "before observation insertion or add a sanitizer observation form that "
        "preserves the operation's lane activity"
    ),
)

# ERR_LOWERING_047: Vector bank scalarization contract failed.
ERR_LOWERING_047 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=47,
    severity=Severity.ERROR,
    summary="Vector bank scalarization contract failed.",
    message=(
        "{phase_name} cannot scalarize carried slot {slot} of {op_name}: {reason}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("slot", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Address the bank only through one static leading-index prefix and "
        "keep the aggregate inside its scf.for recurrence"
    ),
)

# ERR_LOWERING_048: Template application target condition is unresolved.
ERR_LOWERING_048 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=48,
    severity=Severity.ERROR,
    summary="Template application target condition is unresolved.",
    message=(
        "{phase_name} cannot select an implementation for {op_name}"
        "<{contract_key}> because @{constraint_owner_name} has unresolved condition "
        "#{condition_name}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("contract_key", ParamKind.STRING),
        ErrorParam("constraint_owner_name", ParamKind.STRING),
        ErrorParam("condition_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Supply target facts that decide the condition, add a provider whose "
        "applicability is proven, or correct the constraints on an exact "
        "template.call"
    ),
)

# ERR_LOWERING_049: Command-program composition is recursive.
ERR_LOWERING_049 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=49,
    severity=Severity.ERROR,
    summary="Command-program composition is recursive.",
    message=(
        "command-program preparation cannot flatten recursive composition "
        "through @{program_name}"
    ),
    params=(ErrorParam("program_name", ParamKind.STRING),),
    fix_hint=(
        "Make command.program.launch dependencies acyclic before materializing "
        "the command program"
    ),
)

# ERR_LOWERING_050: Command program launch has no kernel definition.
ERR_LOWERING_050 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=50,
    severity=Severity.ERROR,
    summary="Command program launch has no kernel definition.",
    message=(
        "command-program preparation cannot lower the unresolved kernel "
        "declaration @{kernel_name}"
    ),
    params=(ErrorParam("kernel_name", ParamKind.STRING),),
    fix_hint=(
        "Link a kernel.def implementation for this declaration before "
        "preparing the command program"
    ),
)

ALL_LOWERING_ERRORS: tuple[ErrorDef, ...] = (
    ERR_LOWERING_022,
    ERR_LOWERING_023,
    ERR_LOWERING_024,
    ERR_LOWERING_025,
    ERR_LOWERING_026,
    ERR_LOWERING_027,
    ERR_LOWERING_028,
    ERR_LOWERING_029,
    ERR_LOWERING_030,
    ERR_LOWERING_031,
    ERR_LOWERING_032,
    ERR_LOWERING_033,
    ERR_LOWERING_034,
    ERR_LOWERING_035,
    ERR_LOWERING_036,
    ERR_LOWERING_037,
    ERR_LOWERING_038,
    ERR_LOWERING_039,
    ERR_LOWERING_040,
    ERR_LOWERING_043,
    ERR_LOWERING_044,
    ERR_LOWERING_045,
    ERR_LOWERING_046,
    ERR_LOWERING_047,
    ERR_LOWERING_048,
    ERR_LOWERING_049,
    ERR_LOWERING_050,
)
