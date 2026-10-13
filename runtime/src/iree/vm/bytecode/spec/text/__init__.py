# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical physical text declarations for VM bytecode."""

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec import isa
from iree.vm.bytecode.spec.schema import NumericTable


class InstructionPosition(enum.IntEnum):
    """Placement of one field in canonical instruction text."""

    RESULT = 0
    OPERAND = 1
    ATTRIBUTE = 2


class InstructionFieldKind(enum.IntEnum):
    """Closed textual codecs used by physical instruction fields."""

    VALUE_REGISTER = 0
    VALUE_REGISTER_RANGE = 1
    VALUE_REGISTER_FORMAT_RANGE = 2
    REF_REGISTER = 3
    FUNCTION_REGISTER = 4
    SIGNED = 5
    UNSIGNED = 6
    HEX = 7
    COMBINED_HEX = 8
    SELECTOR = 9
    PACKED_SELECTOR = 10
    BLOCK_TARGET = 11
    ORDINAL = 12
    LOCAL_BYTES = 13
    LOCAL_BYTES_RANGE = 14
    LOCAL_BYTES_REPEATED_RANGE = 15
    LOCAL_BYTES_FIXED_RANGE = 16
    REF_SLOT = 17
    MODULE_SYMBOL = 18
    RODATA_RANGE = 19
    DIRECT_TARGET = 20
    SWITCH_SLICE = 21


class SymbolDomain(enum.Enum):
    """Ordinal spaces and their canonical generated symbol prefixes."""

    STRING = "string"
    REF_GROUP = "ref_group"
    REF_TYPE = "ref_type"
    SIGNATURE = "signature"
    CALLABLE = "callable"
    IMPORT_GROUP = "import_group"
    IMPORT = "import"
    EXPORT = "export"
    FUNCTION = "function"
    CONSTANT = "constant"
    GLOBAL_VALUE = "gv"
    GLOBAL_REF = "gr"
    GLOBAL_FUNCTION = "gf"
    RODATA = "rodata"
    SWITCH_TARGET = "switch_target"
    BLOCK = "bb"
    DIRECT_TARGET = "direct_target"
    NATIVE_LAYOUT = "native_layout"
    NATIVE_IMPORT = "native_import"
    NATIVE_CALLSITE = "native_callsite"


class InstructionTextField(NamedTuple):
    """One canonical text item backed by one or more wire fields."""

    name: str
    position: InstructionPosition
    kind: InstructionFieldKind
    primary_field: str
    related_field: str | None = None
    numeric_table: NumericTable | None = None
    symbol_domain: SymbolDomain | None = None
    bit_offset: int = 0
    bit_length: int = 0
    fixed_value: int = 0


class LaneFormat(NamedTuple):
    """One encoded lane format and its two canonical text components."""

    encoded_value: int
    element_name: str
    lane_count: int


class MnemonicProjection(NamedTuple):
    """Selector split between a mnemonic suffix and register range."""

    field_name: str
    formats: tuple[LaneFormat, ...]


class DirectTargetKind(NamedTuple):
    """One direct-target selector and its matching symbol declaration."""

    encoded_value: int
    symbol_domain: SymbolDomain
    symbol_flags_mask: int = 0
    symbol_flags_value: int = 0


class InstructionText(NamedTuple):
    """Complete canonical text declaration for one physical instruction."""

    instruction: isa.Instruction
    fields: tuple[InstructionTextField, ...]
    mnemonic_projection: MnemonicProjection | None = None
    direct_target_kinds: tuple[DirectTargetKind, ...] = ()


class InstructionTextFormat(NamedTuple):
    """Canonical textual forms for one versioned physical ISA."""

    instructions: tuple[InstructionText, ...]
