# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native input/output ties and co-indexed heterogeneous register tuples."""

from __future__ import annotations

from itertools import combinations

from loom.target.arch.amd.xdna.aie.machine import (
    MachineForm,
    MachineOperand,
    MachineOperandKind,
)
from loom.target.arch.amd.xdna.aie.schedule import bypass_class
from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import CORE_ENCODING_TABLE
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.arch.amd.xdna.aie2p.core_schedule_data import CORE_SCHEDULE_TABLE
from loom.target.low_descriptors import Constraint, ConstraintKind

_MACHINE_CLASSES = {row.name: row for row in CORE_MACHINE_TABLE.register_classes}
_MACHINE_REGISTERS = {row.name: row for row in CORE_MACHINE_TABLE.physical_registers}
_MACHINE_ADAPTERS = {row.name: row for row in CORE_MACHINE_TABLE.register_adapters}
_INSTRUCTION_ENCODINGS = {row.name: row for row in CORE_ENCODING_TABLE.instructions}
_ITINERARIES = {row.name: row for row in CORE_SCHEDULE_TABLE.itineraries}


def _native_class(operand: MachineOperand) -> str:
    return (
        _MACHINE_ADAPTERS[operand.type_name].register_class
        if operand.kind is MachineOperandKind.REGISTER_ADAPTER
        else operand.type_name
    )


def descriptor_constraints(
    spec: _DescriptorSpec,
    form: MachineForm,
    operand_names: tuple[str, ...],
) -> tuple[Constraint, ...]:
    """Returns direct update ties and heterogeneous tuple constraints."""

    operand_indices = {
        name: operand_index for operand_index, name in enumerate(operand_names)
    }
    tied_updates = set(spec.tied_updates)
    destructive_updates = set(spec.destructive_updates)
    if len(tied_updates) != len(spec.tied_updates) or len(destructive_updates) != len(
        spec.destructive_updates
    ):
        raise ValueError(f"{form.name}: descriptor update pairs must be unique")
    if tied_updates & destructive_updates:
        raise ValueError(
            f"{form.name}: destructive updates already imply same-storage ties"
        )
    result = [
        Constraint(
            ConstraintKind.TIED,
            operand_indices[tie.definition],
            operand_indices[tie.use],
        )
        for tie in form.ties
        if tie.use not in spec.overwritten_inputs
    ]
    for result_name, input_name in spec.tied_updates:
        if result_name not in operand_indices or input_name not in operand_indices:
            raise ValueError(
                f"{form.name}: tied update names unknown operands "
                f"{result_name!r}, {input_name!r}"
            )
        result.append(
            Constraint(
                ConstraintKind.TIED,
                operand_indices[result_name],
                operand_indices[input_name],
            )
        )
    for result_name, input_name in spec.destructive_updates:
        if result_name not in operand_indices or input_name not in operand_indices:
            raise ValueError(
                f"{form.name}: destructive update names unknown operands "
                f"{result_name!r}, {input_name!r}"
            )
        result_index = operand_indices[result_name]
        input_index = operand_indices[input_name]
        result.extend(
            (
                Constraint(ConstraintKind.TIED, result_index, input_index),
                Constraint(ConstraintKind.DESTRUCTIVE, result_index, input_index),
            )
        )
    result.extend(
        Constraint(
            ConstraintKind.TIED,
            operand_indices[outputs[0]],
            operand_indices[input_name],
        )
        for input_name, outputs in spec.aggregate_updates
    )

    # AIE2P load FIFO forms update one heterogeneous physical-register tuple.
    # LLVM models these through a target hook instead of TableGen Constraints,
    # so they are absent from the imported machine-form ties. Recover the
    # direct state updates from the stable operand family and constrain every
    # tuple member pair: the allocator may visit the three classes in any
    # order, and each partial assignment must select the same aggregate row.
    update_group = ("ptr", "fifo_reg", "pos")
    output_names = tuple(f"{name}_out" for name in update_group)
    if all(name in operand_indices for name in output_names):
        existing_ties = {
            (constraint.lhs_operand_index, constraint.rhs_operand_index)
            for constraint in result
        }
        for input_name, output_name in zip(update_group, output_names, strict=True):
            pair = (operand_indices[output_name], operand_indices[input_name])
            if pair not in existing_ties:
                result.append(Constraint(ConstraintKind.TIED, *pair))
        result.extend(
            Constraint(ConstraintKind.SAME_REGISTER_ORDINAL, lhs, rhs)
            for lhs, rhs in combinations(
                (operand_indices[name] for name in update_group), 2
            )
        )
    return tuple(result)


def descriptor_register_outputs(
    spec: _DescriptorSpec, form: MachineForm
) -> tuple[MachineOperand, ...]:
    """Retains one tied aggregate result for co-indexed component updates."""

    outputs = {
        operand.name: operand
        for operand in form.outputs
        if operand.kind is not MachineOperandKind.IMMEDIATE
        and operand.name not in spec.implicit_outputs
    }
    inputs = {
        operand.name: operand
        for operand in form.inputs
        if operand.kind is not MachineOperandKind.IMMEDIATE
        and operand.name not in spec.implicit_inputs
        and operand.name not in spec.overwritten_inputs
    }
    storage_overrides = dict(spec.storage_overrides)
    grouped = set()
    encoded = {field.name for field in _INSTRUCTION_ENCODINGS[form.name].fields}
    for input_name, group in spec.aggregate_updates:
        if input_name not in inputs or not group or not set(group) <= outputs.keys():
            raise ValueError(
                f"{form.name}: aggregate update requires explicit register operands"
            )
        leader = outputs[group[0]]
        storage = _MACHINE_CLASSES[
            storage_overrides.get(leader.name, _native_class(leader))
        ]
        itinerary = _ITINERARIES[spec.itinerary]
        leader_index = form.outputs.index(leader)
        leader_timing = (
            itinerary.operand_cycles[leader_index],
            bypass_class(itinerary, leader_index),
        )
        if storage.name != storage_overrides.get(
            input_name, _native_class(inputs[input_name])
        ):
            raise ValueError(
                f"{form.name}: aggregate result must retain its input storage class"
            )
        for name in group:
            operand = outputs[name]
            native = _MACHINE_CLASSES[_native_class(operand)]
            if name in grouped or name in encoded:
                raise ValueError(
                    f"{form.name}: grouped outputs must be unique and unencoded"
                )
            index = form.outputs.index(operand)
            timing = (itinerary.operand_cycles[index], bypass_class(itinerary, index))
            if timing != leader_timing:
                raise ValueError(
                    f"{form.name}: grouped outputs must have identical timing"
                )
            if len(native.candidates) != len(storage.candidates) or any(
                not set(_MACHINE_REGISTERS[source].atomic_units)
                <= set(_MACHINE_REGISTERS[target].atomic_units)
                for source, target in zip(
                    native.candidates, storage.candidates, strict=True
                )
            ):
                raise ValueError(
                    f"{form.name}: aggregate result must own each co-indexed "
                    "native output"
                )
            grouped.add(name)
    omitted = {name for _, group in spec.aggregate_updates for name in group[1:]}
    return tuple(operand for name, operand in outputs.items() if name not in omitted)
