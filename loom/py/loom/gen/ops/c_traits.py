# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C trait and placement metadata helpers for generated op tables."""

from __future__ import annotations

from loom.dsl import EffectKind, Op, RegionDef
from loom.fields import compute_layout
from loom.gen.ops.c_enums import TRAIT_MAP
from loom.gen.ops.c_names import c_enum_name


def implicit_terminator_kind(op: Op, ops_by_name: dict[str, Op]) -> str:
    """Returns the C op kind for this op's implicit terminator trait."""
    terminator_traits = [trait for trait in op.traits if trait.name == "ImplicitTerminator"]
    if not terminator_traits:
        return "LOOM_OP_KIND_UNKNOWN"
    if len(terminator_traits) > 1:
        raise ValueError(f"Op '{op.name}': duplicate ImplicitTerminator traits are not supported")
    trait = terminator_traits[0]
    if len(trait.args) != 1:
        raise ValueError(f"Op '{op.name}': ImplicitTerminator requires one op name argument")
    terminator_name = trait.args[0]
    terminator_op = ops_by_name.get(terminator_name)
    if terminator_op is None:
        raise ValueError(f"Op '{op.name}': ImplicitTerminator '{terminator_name}' must name an op in the '{op.namespace}' dialect")
    if not any(trait.name == "Terminator" for trait in terminator_op.traits):
        raise ValueError(f"Op '{op.name}': ImplicitTerminator '{terminator_name}' is not marked with the Terminator trait")
    terminator_layout = compute_layout(terminator_op)
    if terminator_layout.fixed_operand_count != 0 or terminator_layout.fixed_result_count != 0 or terminator_op.attrs or terminator_op.regions:
        raise ValueError(f"Op '{op.name}': ImplicitTerminator '{terminator_name}' must be instantiable with zero operands, results, attrs, and regions")
    return c_enum_name(terminator_op)


def region_terminator_kind(op: Op, region: RegionDef, ops_by_name: dict[str, Op]) -> str:
    """Returns the C op kind required for explicit terminators in a region."""
    if region.terminator is None:
        return "LOOM_OP_KIND_UNKNOWN"
    terminator_op = ops_by_name.get(region.terminator)
    if terminator_op is None:
        raise ValueError(f"Op '{op.name}' region '{region.name}': terminator '{region.terminator}' must name a registered op")
    if not any(trait.name == "Terminator" for trait in terminator_op.traits):
        raise ValueError(f"Op '{op.name}' region '{region.name}': terminator '{region.terminator}' is not marked with the Terminator trait")
    return c_enum_name(terminator_op)


def trait_op_kinds(
    op: Op,
    ops_by_name: dict[str, Op],
    trait_name: str,
) -> list[str]:
    """Returns op-kind enum names referenced by parameterized placement traits."""
    kinds: list[str] = []
    for trait in op.traits:
        if trait.name != trait_name:
            continue
        if len(trait.args) != 1:
            raise ValueError(f"Op '{op.name}': {trait_name} requires one op name argument")
        ancestor_name = trait.args[0]
        ancestor_op = ops_by_name.get(ancestor_name)
        if ancestor_op is None:
            raise ValueError(f"Op '{op.name}': {trait_name} '{ancestor_name}' must name an op in the '{op.namespace}' dialect")
        kinds.append(c_enum_name(ancestor_op))
    return kinds


def trait_flags(op: Op) -> str:
    """Returns the C trait bitfield expression for an op."""
    bits = []
    for trait in op.traits:
        c_name = TRAIT_MAP.get(trait.name)
        if c_name:
            bits.append(c_name)

    has_read = False
    has_write = False
    for effect in op.effects:
        if effect.kind in (EffectKind.READ, EffectKind.READWRITE):
            has_read = True
        if effect.kind in (EffectKind.WRITE, EffectKind.READWRITE):
            has_write = True
    if has_read:
        bits.append("LOOM_TRAIT_READS_MEMORY")
    if has_write:
        bits.append("LOOM_TRAIT_WRITES_MEMORY")

    has_allocating_result = any(result.allocates for result in op.results)
    has_explicit_unique_identity = any(trait.name == "UniqueIdentity" for trait in op.traits)
    if has_allocating_result and not has_explicit_unique_identity:
        bits.append("LOOM_TRAIT_UNIQUE_IDENTITY")

    explicit_pure = any(trait.name == "Pure" for trait in op.traits)
    has_non_deterministic = any(trait.name == "NonDeterministic" for trait in op.traits)
    has_unknown_effects = any(trait.name == "UnknownEffects" for trait in op.traits)
    has_hint = any(trait.name == "Hint" for trait in op.traits)
    if (
        not explicit_pure
        and not op.effects
        and not op.ownership_effects
        and not has_non_deterministic
        and not has_unknown_effects
        and not has_hint
        and not has_allocating_result
        and not has_explicit_unique_identity
        and not has_read
        and not has_write
    ):
        bits.append("LOOM_TRAIT_PURE")

    if not bits:
        return "0"
    return " | ".join(bits)
