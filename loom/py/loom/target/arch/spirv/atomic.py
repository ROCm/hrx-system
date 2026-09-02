# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for Vulkan SPIR-V atomic instructions."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_value


@dataclass(frozen=True, slots=True)
class AtomicIntegerScalar:
    source_type: str
    suffix: str
    scalar_enum: str
    byte_width: int
    storage_buffer_feature_atoms: tuple[str, ...] = ()
    workgroup_feature_atoms: tuple[str, ...] = ()

    def feature_atoms(self, storage_class: AtomicStorageClass) -> tuple[str, ...]:
        if storage_class.suffix == "storage_buffer":
            return self.storage_buffer_feature_atoms
        if storage_class.suffix == "workgroup":
            return self.workgroup_feature_atoms
        raise ValueError(f"unknown atomic storage class '{storage_class.suffix}'")


@dataclass(frozen=True, slots=True)
class AtomicIntegerOperation:
    source_kind: str
    suffix: str
    mnemonic: str
    opcode: str
    supports_reduce: bool = True


@dataclass(frozen=True, slots=True)
class AtomicFloatScalar:
    source_type: str
    suffix: str
    scalar_enum: str
    byte_width: int
    integer_source_type: str | None
    integer_suffix: str | None
    integer_scalar_enum: str | None
    storage_buffer_basic_feature_atom: str
    workgroup_basic_feature_atom: str
    storage_buffer_add_feature_atom: str
    workgroup_add_feature_atom: str
    storage_buffer_cas_feature_atoms: tuple[str, ...] = ()
    workgroup_cas_feature_atoms: tuple[str, ...] = ()

    def native_feature_atoms(
        self,
        storage_class: AtomicStorageClass,
        operation: AtomicFloatOperation,
    ) -> tuple[str, ...]:
        if operation.source_kind == "xchgf":
            feature_atom = (
                self.storage_buffer_basic_feature_atom
                if storage_class.suffix == "storage_buffer"
                else self.workgroup_basic_feature_atom
            )
        elif operation.source_kind == "addf":
            feature_atom = (
                self.storage_buffer_add_feature_atom
                if storage_class.suffix == "storage_buffer"
                else self.workgroup_add_feature_atom
            )
        else:
            return ()
        return (feature_atom,)

    def cas_feature_atoms(self, storage_class: AtomicStorageClass) -> tuple[str, ...]:
        if self.integer_scalar_enum is None:
            return ()
        if storage_class.suffix == "storage_buffer":
            return self.storage_buffer_cas_feature_atoms
        if storage_class.suffix == "workgroup":
            return self.workgroup_cas_feature_atoms
        raise ValueError(f"unknown atomic storage class '{storage_class.suffix}'")


@dataclass(frozen=True, slots=True)
class AtomicFloatOperation:
    source_kind: str
    suffix: str
    native_mnemonic: str | None
    native_opcode: str | None
    cas_operation: int
    supports_reduce: bool = True


@dataclass(frozen=True, slots=True)
class AtomicStorageClass:
    suffix: str
    memory_semantics: str


@dataclass(frozen=True, slots=True)
class AtomicOrdering:
    source_keyword: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class AtomicScope:
    source_keyword: str
    spirv_scope: str
    orderings: tuple[AtomicOrdering, ...]
    feature_atoms: tuple[str, ...] = ()


ATOMIC_STORAGE_CLASSES = (
    AtomicStorageClass(
        suffix="storage_buffer",
        memory_semantics="LOOM_SPIRV_MEMORY_SEMANTICS_UNIFORM_MEMORY_MASK",
    ),
    AtomicStorageClass(
        suffix="workgroup",
        memory_semantics="LOOM_SPIRV_MEMORY_SEMANTICS_WORKGROUP_MEMORY_MASK",
    ),
)

ATOMIC_ORDERINGS = (
    AtomicOrdering("relaxed", 0),
    AtomicOrdering("acquire", 1),
    AtomicOrdering("release", 2),
    AtomicOrdering("acq_rel", 3),
)

ATOMIC_SCOPES = (
    AtomicScope(
        source_keyword="thread",
        spirv_scope="LOOM_SPIRV_SCOPE_INVOCATION",
        orderings=ATOMIC_ORDERINGS[:1],
    ),
    AtomicScope(
        source_keyword="subgroup",
        spirv_scope="LOOM_SPIRV_SCOPE_SUBGROUP",
        orderings=ATOMIC_ORDERINGS,
        feature_atoms=("group_non_uniform",),
    ),
    AtomicScope(
        source_keyword="workgroup",
        spirv_scope="LOOM_SPIRV_SCOPE_WORKGROUP",
        orderings=ATOMIC_ORDERINGS,
    ),
    AtomicScope(
        source_keyword="device",
        spirv_scope="LOOM_SPIRV_SCOPE_DEVICE",
        orderings=ATOMIC_ORDERINGS,
        feature_atoms=("vulkan_memory_model_device_scope",),
    ),
)

ATOMIC_INTEGER_SCALARS = (
    AtomicIntegerScalar(
        source_type="i32",
        suffix="i32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
        byte_width=4,
    ),
    AtomicIntegerScalar(
        source_type="i64",
        suffix="i64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S64",
        byte_width=8,
        storage_buffer_feature_atoms=("storage_buffer_int64_atomics",),
        workgroup_feature_atoms=("workgroup_int64_atomics",),
    ),
)

ATOMIC_INTEGER_OPERATIONS = (
    AtomicIntegerOperation(
        "xchgi",
        "exchange",
        "OpAtomicExchange",
        "LOOM_SPIRV_OP_ATOMIC_EXCHANGE",
        supports_reduce=False,
    ),
    AtomicIntegerOperation(
        "addi", "iadd", "OpAtomicIAdd", "LOOM_SPIRV_OP_ATOMIC_I_ADD"
    ),
    AtomicIntegerOperation(
        "subi", "isub", "OpAtomicISub", "LOOM_SPIRV_OP_ATOMIC_I_SUB"
    ),
    AtomicIntegerOperation("andi", "and", "OpAtomicAnd", "LOOM_SPIRV_OP_ATOMIC_AND"),
    AtomicIntegerOperation("ori", "or", "OpAtomicOr", "LOOM_SPIRV_OP_ATOMIC_OR"),
    AtomicIntegerOperation("xori", "xor", "OpAtomicXor", "LOOM_SPIRV_OP_ATOMIC_XOR"),
    AtomicIntegerOperation(
        "minsi", "smin", "OpAtomicSMin", "LOOM_SPIRV_OP_ATOMIC_S_MIN"
    ),
    AtomicIntegerOperation(
        "maxsi", "smax", "OpAtomicSMax", "LOOM_SPIRV_OP_ATOMIC_S_MAX"
    ),
    AtomicIntegerOperation(
        "minui", "umin", "OpAtomicUMin", "LOOM_SPIRV_OP_ATOMIC_U_MIN"
    ),
    AtomicIntegerOperation(
        "maxui", "umax", "OpAtomicUMax", "LOOM_SPIRV_OP_ATOMIC_U_MAX"
    ),
)

ATOMIC_FLOAT_SCALARS = (
    AtomicFloatScalar(
        source_type="f16",
        suffix="f16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F16",
        byte_width=2,
        integer_source_type=None,
        integer_suffix=None,
        integer_scalar_enum=None,
        storage_buffer_basic_feature_atom="storage_buffer_float16_atomics",
        workgroup_basic_feature_atom="workgroup_float16_atomics",
        storage_buffer_add_feature_atom="storage_buffer_float16_atomic_add",
        workgroup_add_feature_atom="workgroup_float16_atomic_add",
    ),
    AtomicFloatScalar(
        source_type="f32",
        suffix="f32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F32",
        byte_width=4,
        integer_source_type="i32",
        integer_suffix="i32",
        integer_scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
        storage_buffer_basic_feature_atom="storage_buffer_float32_atomics",
        workgroup_basic_feature_atom="workgroup_float32_atomics",
        storage_buffer_add_feature_atom="storage_buffer_float32_atomic_add",
        workgroup_add_feature_atom="workgroup_float32_atomic_add",
    ),
    AtomicFloatScalar(
        source_type="f64",
        suffix="f64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F64",
        byte_width=8,
        integer_source_type="i64",
        integer_suffix="i64",
        integer_scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S64",
        storage_buffer_basic_feature_atom="storage_buffer_float64_atomics",
        workgroup_basic_feature_atom="workgroup_float64_atomics",
        storage_buffer_add_feature_atom="storage_buffer_float64_atomic_add",
        workgroup_add_feature_atom="workgroup_float64_atomic_add",
        storage_buffer_cas_feature_atoms=(
            "float64",
            "storage_buffer_int64_atomics",
        ),
        workgroup_cas_feature_atoms=(
            "float64",
            "workgroup_int64_atomics",
        ),
    ),
)

# Vulkan's native floating-point minimum and maximum atomics do not preserve
# Loom's distinct NaN and signed-zero contracts. Those operations therefore use
# exact integer compare-exchange loops instead of the native extension opcodes.
ATOMIC_FLOAT_OPERATIONS = (
    AtomicFloatOperation(
        source_kind="xchgf",
        suffix="exchange",
        native_mnemonic="OpAtomicExchange",
        native_opcode="LOOM_SPIRV_OP_ATOMIC_EXCHANGE",
        cas_operation=0,
        supports_reduce=False,
    ),
    AtomicFloatOperation(
        source_kind="addf",
        suffix="fadd",
        native_mnemonic="OpAtomicFAddEXT",
        native_opcode="LOOM_SPIRV_OP_ATOMIC_F_ADD_EXT",
        cas_operation=1,
    ),
    AtomicFloatOperation(
        source_kind="minimumf",
        suffix="fminimum",
        native_mnemonic=None,
        native_opcode=None,
        cas_operation=2,
    ),
    AtomicFloatOperation(
        source_kind="maximumf",
        suffix="fmaximum",
        native_mnemonic=None,
        native_opcode=None,
        cas_operation=3,
    ),
    AtomicFloatOperation(
        source_kind="minnumf",
        suffix="fmin",
        native_mnemonic=None,
        native_opcode=None,
        cas_operation=4,
    ),
    AtomicFloatOperation(
        source_kind="maxnumf",
        suffix="fmax",
        native_mnemonic=None,
        native_opcode=None,
        cas_operation=5,
    ),
)


def atomic_feature_bits(
    scalar: AtomicIntegerScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
) -> int:
    return feature_bits_value(
        (*scalar.feature_atoms(storage_class), *scope.feature_atoms)
    )


def float_atomic_native_feature_bits(
    scalar: AtomicFloatScalar,
    operation: AtomicFloatOperation,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
) -> int:
    return feature_bits_value(
        (*scalar.native_feature_atoms(storage_class, operation), *scope.feature_atoms)
    )


def float_atomic_cas_feature_bits(
    scalar: AtomicFloatScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
) -> int:
    return feature_bits_value(
        (*scalar.cas_feature_atoms(storage_class), *scope.feature_atoms)
    )


def atomic_descriptor_key(
    form: str,
    scalar: AtomicIntegerScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
    *,
    operation: AtomicIntegerOperation | None = None,
    success_ordering: AtomicOrdering | None = None,
) -> str:
    if form == "cmpxchg":
        if operation is not None or success_ordering is None:
            raise ValueError("compare-exchange keys require only a success ordering")
        operation_suffix = f"cmpxchg.{success_ordering.source_keyword}"
    else:
        if operation is None or success_ordering is not None:
            raise ValueError(f"{form} keys require only an atomic operation")
        operation_suffix = operation.suffix
    return (
        f"spirv.atomic.{form}.{operation_suffix}.{storage_class.suffix}."
        f"{scalar.suffix}.{scope.source_keyword}"
    )


def float_atomic_descriptor_key(
    form: str,
    strategy: str,
    scalar: AtomicFloatScalar,
    storage_class: AtomicStorageClass,
    scope: AtomicScope,
    *,
    operation: AtomicFloatOperation | None = None,
    success_ordering: AtomicOrdering | None = None,
) -> str:
    if form == "cmpxchg":
        if operation is not None or success_ordering is None:
            raise ValueError("compare-exchange keys require only a success ordering")
        operation_suffix = f"cmpxchg.{success_ordering.source_keyword}"
    else:
        if operation is None or success_ordering is not None:
            raise ValueError(f"{form} keys require only an atomic operation")
        operation_suffix = operation.suffix
    return (
        f"spirv.atomic.{form}.{operation_suffix}.{strategy}."
        f"{storage_class.suffix}.{scalar.suffix}.{scope.source_keyword}"
    )


def cmpxchg_failure_orderings(
    success_ordering: AtomicOrdering,
) -> tuple[AtomicOrdering, ...]:
    if success_ordering.source_keyword in ("relaxed", "release"):
        return ATOMIC_ORDERINGS[:1]
    if success_ordering.source_keyword in ("acquire", "acq_rel"):
        return ATOMIC_ORDERINGS[:2]
    raise ValueError(
        f"unsupported compare-exchange success ordering "
        f"'{success_ordering.source_keyword}'"
    )
