# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declarative instruction identities, layouts, and complete semantics."""

from __future__ import annotations

import dataclasses
import enum
import re

from model.schema import (
    FieldReference,
    RuleUse,
    ScalarEncoding,
    ValidationScope,
    validate_rule_use,
)
from model.specification import (
    ARCHITECTURAL_EXTENSION_PAGE_MIN,
    RESERVED_EXTENDED_ESCAPE_PAGE_ID,
    Entity,
    Specification,
)

_FIELD_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_MNEMONIC_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")
_REF_CONTRACT_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


class InstructionFieldRole(enum.Enum):
    """Presentation and semantic role of an encoded instruction field."""

    RESULT = "result"
    OPERAND = "operand"
    IMMEDIATE = "immediate"
    RANGE_BASE = "range_base"
    RANGE_COUNT = "range_count"
    CONSTRAINT_MEMBER = "constraint_member"
    PADDING = "padding"


class ControlFlow(enum.Enum):
    """Static successor class of an instruction record."""

    SEQUENTIAL = "sequential"
    BLOCK = "block"
    RETURN = "return"
    YIELD = "yield"
    BRANCH = "branch"
    CONDITIONAL_BRANCH = "conditional_branch"
    SWITCH = "switch"
    CALL = "call"
    FAIL = "fail"


class Suspension(enum.Enum):
    """Whether instruction execution may suspend its invocation."""

    NEVER = "never"
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    TARGET_DEPENDENT = "target_dependent"


class StateAccess(enum.Enum):
    """Kind of access an instruction may make to architectural state."""

    UNKNOWN = "unknown"
    READ = "read"
    WRITE = "write"
    ALLOCATE = "allocate"
    RELEASE = "release"
    SYNCHRONIZE = "synchronize"


class StateResource(enum.Enum):
    """Non-register architectural state resource affected by an instruction."""

    ANY = "any"
    INVOCATION_ARGUMENTS = "invocation.arguments"
    INVOCATION_RESULTS = "invocation.results"
    FRAME_LOCALS = "frame.locals"
    PROCESS_GLOBALS = "process.globals"
    BUFFER = "buffer"
    HAL_DEVICE = "hal.device"
    HAL_DEVICE_GROUP = "hal.device_group"
    HAL_EXECUTABLE = "hal.executable"
    HAL_POOL = "hal.pool"
    HAL_COMMAND_BUFFER = "hal.command_buffer"
    HAL_QUEUE = "hal.queue"
    HAL_SEMAPHORE = "hal.semaphore"
    HAL_CHANNEL = "hal.channel"
    IO_FILE = "io.file"


@dataclasses.dataclass(frozen=True, slots=True)
class StateEffect:
    """One conservative may-effect on non-register architectural state.

    Effects include state changes scheduled for deferred execution. Resource
    fields name the encoded fields that identify the affected resource. An
    empty field tuple denotes implicit or domain-wide state.
    """

    access: StateAccess
    resource: StateResource
    resource_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.access, StateAccess):
            raise ValueError("invalid state access")
        if not isinstance(self.resource, StateResource):
            raise ValueError("invalid state resource")
        is_unknown = self.access == StateAccess.UNKNOWN
        is_any_resource = self.resource == StateResource.ANY
        if is_unknown != is_any_resource:
            raise ValueError("unknown state access and any resource must be paired")
        if is_any_resource and self.resource_fields:
            raise ValueError("any-resource state effect cannot name resource fields")
        if len(set(self.resource_fields)) != len(self.resource_fields):
            raise ValueError("duplicate state-effect resource field")
        for field_name in self.resource_fields:
            if not _FIELD_PATTERN.fullmatch(field_name):
                raise ValueError(f"invalid state-effect resource field {field_name!r}")


class RefNullPolicy(enum.Enum):
    """Dynamic null/type interpretation of a ref operand or result."""

    DESCRIPTOR = "descriptor"
    DIRECT_OR_SLOT = "direct_or_slot"
    NULLABLE = "nullable"
    REQUIRED = "required"
    RESULT_NONNULL = "result_nonnull"


class RefOwnership(enum.Enum):
    """Ownership transition applied by a ref operand or result."""

    BORROW = "borrow"
    CLEAR = "clear"
    CONSUME = "consume"
    DIAGNOSTIC_BORROW = "diagnostic.borrow"
    INSPECT = "inspect"
    MOVE = "move"
    PUBLISH_MOVE = "publish.move"
    REPLACE_BORROW = "replace.borrow"
    REPLACE_MOVE = "replace.move"
    REPLACE_OWNER = "replace.owner"
    REPLACE_RETAIN = "replace.retain"
    RETAIN = "retain"


class InstructionRangeStorage(enum.Enum):
    """Frame storage domain addressed by a counted instruction range."""

    LOCAL_BYTES = "local.bytes"
    LOCAL_REFS = "local.refs"


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeRefPolicy:
    """Dynamic type, nullability, and ownership contract for one ref use."""

    type_contract: str
    null_policy: RefNullPolicy
    ownership: RefOwnership

    def __post_init__(self) -> None:
        if not _REF_CONTRACT_PATTERN.fullmatch(self.type_contract):
            raise ValueError(f"invalid ref type contract {self.type_contract!r}")
        if not isinstance(self.null_policy, RefNullPolicy):
            raise ValueError("invalid ref null policy")
        if not isinstance(self.ownership, RefOwnership):
            raise ValueError("invalid ref ownership policy")


@dataclasses.dataclass(frozen=True, slots=True)
class InstructionField:
    """One complete named byte range in an instruction record."""

    name: str
    offset: int
    encoding_id: str
    role: InstructionFieldRole
    description: str
    validation: tuple[RuleUse, ...]
    runtime_ref_policy: RuntimeRefPolicy | None = None
    array_length: int = 1

    def __post_init__(self) -> None:
        if not _FIELD_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid instruction field name {self.name!r}")
        if self.offset < 0:
            raise ValueError(f"{self.name}: negative instruction offset")
        if self.array_length <= 0:
            raise ValueError(f"{self.name}: non-positive array length")
        if not self.description.strip():
            raise ValueError(f"{self.name}: missing field description")
        if not self.validation:
            raise ValueError(f"{self.name}: missing field validation")

    def referenced_entity_ids(self) -> tuple[str, ...]:
        references = [self.encoding_id]
        for rule_use in self.validation:
            references.extend(rule_use.referenced_entity_ids())
        return tuple(sorted(set(references)))


@dataclasses.dataclass(frozen=True, slots=True)
class InstructionRangeMember:
    """One array base governed by a shared instruction count field."""

    base_field: str
    storage: InstructionRangeStorage
    element_byte_length: int
    element_alignment: int
    runtime_ref_policy: RuntimeRefPolicy | None = None

    def __post_init__(self) -> None:
        if not _FIELD_PATTERN.fullmatch(self.base_field):
            raise ValueError(f"invalid range base field {self.base_field!r}")
        if not isinstance(self.storage, InstructionRangeStorage):
            raise ValueError(f"{self.base_field}: invalid range storage")
        if self.element_byte_length <= 0:
            raise ValueError(f"{self.base_field}: invalid element byte length")
        if not _is_power_of_two(self.element_alignment):
            raise ValueError(f"{self.base_field}: invalid element alignment")
        if self.element_byte_length % self.element_alignment:
            raise ValueError(f"{self.base_field}: element length violates alignment")


@dataclasses.dataclass(frozen=True, slots=True)
class InstructionRangeGroup:
    """One checked count shared by one or more local-storage arrays."""

    name: str
    count_field: str
    members: tuple[InstructionRangeMember, ...]

    def __post_init__(self) -> None:
        if not _FIELD_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid range group name {self.name!r}")
        if not _FIELD_PATTERN.fullmatch(self.count_field):
            raise ValueError(f"invalid range count field {self.count_field!r}")
        if not self.members:
            raise ValueError(f"{self.name}: empty range group")


@dataclasses.dataclass(frozen=True, slots=True)
class FailureCase:
    """One dynamic terminal failure and its mutation boundary."""

    status: str
    condition: str
    atomicity: str

    def __post_init__(self) -> None:
        if not self.status.strip():
            raise ValueError("missing failure status")
        if not self.condition.strip():
            raise ValueError("missing failure condition")
        if not self.atomicity.strip():
            raise ValueError("missing failure atomicity")


@dataclasses.dataclass(frozen=True, slots=True)
class InstructionSemantics:
    """Complete observable behavior of one physical instruction."""

    description: str
    verification: tuple[str, ...]
    preconditions: tuple[str, ...]
    success: tuple[str, ...]
    failures: tuple[FailureCase, ...]
    ownership: tuple[str, ...]
    assembly: tuple[str, ...]
    pseudocode: str
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("missing instruction description")
        if not self.verification or any(
            not statement.strip() for statement in self.verification
        ):
            raise ValueError("missing instruction verification behavior")
        if any(not statement.strip() for statement in self.preconditions):
            raise ValueError("empty instruction precondition")
        if any(not statement.strip() for statement in self.success):
            raise ValueError("empty instruction success behavior")
        if any(not statement.strip() for statement in self.ownership):
            raise ValueError("empty instruction ownership behavior")
        if not self.assembly or any(
            not statement.strip() for statement in self.assembly
        ):
            raise ValueError("missing instruction assembly example")
        if not self.pseudocode.strip():
            raise ValueError("missing instruction pseudocode")
        if tuple(sorted(self.dependencies)) != self.dependencies:
            raise ValueError("instruction semantic dependencies are not sorted")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("duplicate instruction semantic dependency")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InstructionFamily(Entity):
    """One generated-document family and its inherited normative contract."""

    document_order: int
    normative_text: str

    def __post_init__(self) -> None:
        super(InstructionFamily, self).__post_init__()
        if self.document_order < 0:
            raise ValueError(f"{self.entity_id}: negative document order")
        if not self.normative_text.strip():
            raise ValueError(f"{self.entity_id}: missing family contract")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Instruction(Entity):
    """One complete immutable physical instruction declaration."""

    opcode: int
    mnemonic: str
    byte_length: int
    family_id: str
    fields: tuple[InstructionField, ...]
    range_groups: tuple[InstructionRangeGroup, ...]
    constraints: tuple[RuleUse, ...]
    control_flow: ControlFlow
    suspension: Suspension
    state_effects: tuple[StateEffect, ...]
    semantics: InstructionSemantics

    def referenced_entity_ids(self) -> tuple[str, ...]:
        references = [self.family_id, *self.semantics.dependencies]
        for field in self.fields:
            references.extend(field.referenced_entity_ids())
        for constraint in self.constraints:
            references.extend(constraint.referenced_entity_ids())
        return tuple(sorted(set(references)))

    def validate(self, specification: Specification) -> None:
        entities_by_id = specification.entity_map()
        domain = specification.domain_map()[self.since.domain]
        family = entities_by_id.get(self.family_id)
        if not isinstance(family, InstructionFamily):
            raise ValueError(
                f"{self.entity_id}: family {self.family_id!r} is not declared"
            )
        if family.since.domain != self.since.domain:
            raise ValueError(f"{self.entity_id}: family belongs to another domain")
        if not 1 <= self.opcode <= 0xFF:
            raise ValueError(f"{self.entity_id}: invalid opcode {self.opcode:#x}")
        if domain.page_id == 0:
            if (
                ARCHITECTURAL_EXTENSION_PAGE_MIN
                <= self.opcode
                <= RESERVED_EXTENDED_ESCAPE_PAGE_ID
            ):
                raise ValueError(
                    f"{self.entity_id}: core opcode {self.opcode:#x} is a "
                    "reserved page prefix"
                )
        if not _MNEMONIC_PATTERN.fullmatch(self.mnemonic):
            raise ValueError(f"{self.entity_id}: invalid mnemonic {self.mnemonic!r}")
        if self.byte_length <= 0 or self.byte_length % 4:
            raise ValueError(f"{self.entity_id}: instruction is not four-byte framed")
        if self.control_flow == ControlFlow.FAIL:
            if self.semantics.success:
                raise ValueError(
                    f"{self.entity_id}: terminal failure has success effects"
                )
            if not self.semantics.failures:
                raise ValueError(
                    f"{self.entity_id}: terminal failure has no failure case"
                )
        elif not self.semantics.success:
            raise ValueError(f"{self.entity_id}: missing success behavior")

        state_effect_keys: set[tuple[StateAccess, StateResource, frozenset[str]]] = (
            set()
        )
        for state_effect in self.state_effects:
            if not isinstance(state_effect, StateEffect):
                raise ValueError(f"{self.entity_id}: invalid state effect")
            key = (
                state_effect.access,
                state_effect.resource,
                frozenset(state_effect.resource_fields),
            )
            if key in state_effect_keys:
                raise ValueError(f"{self.entity_id}: duplicate state effect {key}")
            state_effect_keys.add(key)
        if len(self.state_effects) > 1 and any(
            state_effect.access == StateAccess.UNKNOWN
            for state_effect in self.state_effects
        ):
            raise ValueError(
                f"{self.entity_id}: unknown state effect must be the only effect"
            )

        for other in specification.entities:
            if not isinstance(other, Instruction) or other is self:
                continue
            other_domain = specification.domain_map()[other.since.domain]
            if other_domain.page_id == domain.page_id and other.opcode == self.opcode:
                raise ValueError(
                    f"{self.entity_id}: opcode collides with {other.entity_id}"
                )
            if other.mnemonic == self.mnemonic:
                raise ValueError(
                    f"{self.entity_id}: mnemonic collides with {other.entity_id}"
                )

        header_byte_length = 1 if domain.page_id == 0 else 2
        occupied: list[str | None] = [None] * self.byte_length
        for byte_offset in range(header_byte_length):
            occupied[byte_offset] = "instruction_header"
        field_names: set[str] = set()
        for field in self.fields:
            if field.name in field_names:
                raise ValueError(f"{self.entity_id}: duplicate field {field.name}")
            field_names.add(field.name)
            encoding = entities_by_id.get(field.encoding_id)
            if not isinstance(encoding, ScalarEncoding):
                raise ValueError(
                    f"{self.entity_id}.{field.name}: encoding is not scalar"
                )
            if field.offset % encoding.alignment:
                raise ValueError(f"{self.entity_id}.{field.name}: naturally misaligned")
            field_end = field.offset + encoding.byte_length * field.array_length
            if field.offset < header_byte_length or field_end > self.byte_length:
                raise ValueError(f"{self.entity_id}.{field.name}: outside instruction")
            for byte_offset in range(field.offset, field_end):
                previous_owner = occupied[byte_offset]
                if previous_owner is not None:
                    raise ValueError(
                        f"{self.entity_id}.{field.name}: overlaps "
                        f"{previous_owner} at byte {byte_offset}"
                    )
                occupied[byte_offset] = field.name
        uncovered = [
            byte_offset for byte_offset, owner in enumerate(occupied) if owner is None
        ]
        if uncovered:
            raise ValueError(f"{self.entity_id}: uncovered bytes {uncovered}")

        frozen_field_names = frozenset(field_names)
        fields_by_name = {field.name: field for field in self.fields}
        for state_effect in self.state_effects:
            for field_name in state_effect.resource_fields:
                field = fields_by_name.get(field_name)
                if field is None:
                    raise ValueError(
                        f"{self.entity_id}: state effect references unknown field "
                        f"{field_name}"
                    )
                if field.role == InstructionFieldRole.PADDING:
                    raise ValueError(
                        f"{self.entity_id}: state effect references padding field "
                        f"{field_name}"
                    )
        for field in self.fields:
            runtime_ref_policy = field.runtime_ref_policy
            if runtime_ref_policy is not None:
                self._validate_field_ref_policy(field.name, runtime_ref_policy)
            for rule_use in field.validation:
                validate_rule_use(
                    f"{self.entity_id}.{field.name}",
                    rule_use,
                    ValidationScope.FIELD,
                    specification,
                    field_names=frozen_field_names,
                )
        for constraint in self.constraints:
            validate_rule_use(
                self.entity_id,
                constraint,
                ValidationScope.RECORD,
                specification,
                field_names=frozen_field_names,
            )

        self._validate_ranges(field_names)
        self._validate_constraint_members(field_names)

    def _validate_field_ref_policy(
        self,
        field_name: str,
        runtime_ref_policy: RuntimeRefPolicy,
    ) -> None:
        mode = (
            runtime_ref_policy.null_policy,
            runtime_ref_policy.ownership,
        )
        allowed_modes = {
            (RefNullPolicy.DESCRIPTOR, RefOwnership.BORROW),
            (RefNullPolicy.DESCRIPTOR, RefOwnership.PUBLISH_MOVE),
            (RefNullPolicy.DESCRIPTOR, RefOwnership.RETAIN),
            (RefNullPolicy.DIRECT_OR_SLOT, RefOwnership.BORROW),
            (RefNullPolicy.NULLABLE, RefOwnership.BORROW),
            (RefNullPolicy.NULLABLE, RefOwnership.CLEAR),
            (RefNullPolicy.NULLABLE, RefOwnership.DIAGNOSTIC_BORROW),
            (RefNullPolicy.NULLABLE, RefOwnership.INSPECT),
            (RefNullPolicy.NULLABLE, RefOwnership.MOVE),
            (RefNullPolicy.NULLABLE, RefOwnership.PUBLISH_MOVE),
            (RefNullPolicy.NULLABLE, RefOwnership.REPLACE_MOVE),
            (RefNullPolicy.NULLABLE, RefOwnership.REPLACE_RETAIN),
            (RefNullPolicy.NULLABLE, RefOwnership.RETAIN),
            (RefNullPolicy.REQUIRED, RefOwnership.BORROW),
            (RefNullPolicy.REQUIRED, RefOwnership.CONSUME),
            (RefNullPolicy.RESULT_NONNULL, RefOwnership.REPLACE_BORROW),
            (RefNullPolicy.RESULT_NONNULL, RefOwnership.REPLACE_OWNER),
        }
        if mode not in allowed_modes:
            raise ValueError(
                f"{self.entity_id}.{field_name}: unsupported ref policy "
                f"{mode[0].value}/{mode[1].value}"
            )

    def _validate_ranges(self, field_names: set[str]) -> None:
        groups: set[str] = set()
        owned_fields: set[str] = set()
        for group in self.range_groups:
            if group.name in groups:
                raise ValueError(
                    f"{self.entity_id}: duplicate range group {group.name}"
                )
            groups.add(group.name)
            if group.count_field not in field_names:
                raise ValueError(
                    f"{self.entity_id}: missing range count {group.count_field}"
                )
            if group.count_field in owned_fields:
                raise ValueError(
                    f"{self.entity_id}: multiply owned range field {group.count_field}"
                )
            owned_fields.add(group.count_field)
            for member in group.members:
                if member.base_field not in field_names:
                    raise ValueError(
                        f"{self.entity_id}: missing range base {member.base_field}"
                    )
                if member.base_field in owned_fields:
                    raise ValueError(
                        f"{self.entity_id}: multiply owned range field "
                        f"{member.base_field}"
                    )
                if member.storage == InstructionRangeStorage.LOCAL_REFS:
                    if member.runtime_ref_policy is None:
                        raise ValueError(
                            f"{self.entity_id}.{member.base_field}: missing ref "
                            "range policy"
                        )
                    mode = (
                        member.runtime_ref_policy.null_policy,
                        member.runtime_ref_policy.ownership,
                    )
                    if mode not in {
                        (RefNullPolicy.NULLABLE, RefOwnership.BORROW),
                        (RefNullPolicy.REQUIRED, RefOwnership.BORROW),
                    }:
                        raise ValueError(
                            f"{self.entity_id}.{member.base_field}: unsupported "
                            "ref range policy"
                        )
                elif member.runtime_ref_policy is not None:
                    raise ValueError(
                        f"{self.entity_id}.{member.base_field}: byte range has "
                        "a ref policy"
                    )
                owned_fields.add(member.base_field)
        declared_fields = {
            field.name
            for field in self.fields
            if field.role
            in {InstructionFieldRole.RANGE_BASE, InstructionFieldRole.RANGE_COUNT}
        }
        if owned_fields != declared_fields:
            raise ValueError(
                f"{self.entity_id}: range ownership mismatch; declared "
                f"{sorted(declared_fields)}, owned {sorted(owned_fields)}"
            )

    def _validate_constraint_members(self, field_names: set[str]) -> None:
        referenced_fields: set[str] = set()

        def collect(value: object) -> None:
            if isinstance(value, FieldReference):
                referenced_fields.add(value.field_name)
            elif isinstance(value, tuple):
                for element in value:
                    collect(element)

        for constraint in self.constraints:
            for argument in constraint.arguments:
                collect(argument)
        declared_fields = {
            field.name
            for field in self.fields
            if field.role == InstructionFieldRole.CONSTRAINT_MEMBER
        }
        if not declared_fields <= referenced_fields:
            raise ValueError(
                f"{self.entity_id}: unowned constraint members "
                f"{sorted(declared_fields - referenced_fields)}"
            )
        if not referenced_fields <= field_names:
            raise ValueError(
                f"{self.entity_id}: constraint references unknown fields "
                f"{sorted(referenced_fields - field_names)}"
            )
