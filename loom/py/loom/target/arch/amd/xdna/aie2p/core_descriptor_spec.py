# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Semantic selection of an AIE2P physical form for the Low descriptor model."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.low_descriptors import DescriptorOpKind, Effect


@dataclass(frozen=True, slots=True)
class _DescriptorSpec:
    """Semantic selection of one physical form and its exact itinerary."""

    # Owned physical machine form providing operands, constraints and encoding.
    form_name: str
    # Stable descriptor key in the AIE2P core vocabulary.
    key: str
    # Semantic operation selected by source lowering and equivalent aliases.
    semantic_tag: str
    # Owned scheduling itinerary, specialized for the selected register banks.
    itinerary: str
    # Operand-local storage domains within the native encoding domains.
    storage_overrides: tuple[tuple[str, str], ...] = ()
    # Structural Low operation carrying the descriptor.
    op_kind: DescriptorOpKind = DescriptorOpKind.OP
    # Machine outputs fixed to architectural state instead of SSA results.
    implicit_outputs: tuple[str, ...] = ()
    # Machine inputs fixed to architectural state instead of SSA operands.
    implicit_inputs: tuple[str, ...] = ()
    # Authored Low spelling when the physical mnemonic is ambiguous.
    asm_mnemonic: str | None = None
    # Named partial-register views accessed by individual machine operands.
    operand_register_parts: tuple[tuple[str, str], ...] = ()
    # Native encoding adapters for the selected storage and register parts.
    encoding_adapter_overrides: tuple[tuple[str, str], ...] = ()
    # Disjoint part preserved through the first result's tied storage input.
    storage_continuation_part: str | None = None
    # Input aggregates updated by co-indexed native outputs. The first output
    # names the aggregate SSA result tied to the input's storage.
    aggregate_updates: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Semantically equivalent descriptors available to instruction scheduling.
    schedule_alternatives: tuple[str, ...] = ()
    # Memory transfer width in bits, independent of register storage width.
    memory_width_bits: int | None = None
    # Whether individual memory accesses are independently observable.
    ordered_memory: bool = False
    # Additional effects beyond the machine form's memory and control effects.
    effects: tuple[Effect, ...] = ()
    # Whether allocation may use the descriptor for physical register copies.
    allocation_move: bool = False
    # Whether the pure result can be recreated near uses under register pressure.
    rematerializable: bool = False
    # Exposes native srCarry reads and writes as fixed-register SSA values.
    expose_carry: bool = False
