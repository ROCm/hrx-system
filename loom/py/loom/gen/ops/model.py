# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python DSL model loading for Loom C op table generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from loom.dsl import Op


@dataclass(frozen=True)
class DialectGeneration:
    """Generated C metadata inputs for one dialect."""

    dialect: Any
    ops: list[Op]
    table_shards: Sequence[tuple[Any, Sequence[Op]]] | None


@dataclass(frozen=True)
class GenerationModel:
    """Complete op/type model consumed by C table generation."""

    dialects: list[DialectGeneration]
    types: list[Any]


DialectGenerationLoader = Callable[[], DialectGeneration]


def _load_test_generation() -> DialectGeneration:
    from loom.dialect.test import ALL_TEST_OPS, test_ops

    return DialectGeneration(test_ops, list(ALL_TEST_OPS), None)


def _load_scalar_generation() -> DialectGeneration:
    from loom.dialect.scalar import ALL_SCALAR_OPS, SCALAR_OP_CATEGORY_GROUPS, scalar_ops

    return DialectGeneration(scalar_ops, list(ALL_SCALAR_OPS), SCALAR_OP_CATEGORY_GROUPS)


def _load_func_generation() -> DialectGeneration:
    from loom.dialect.func import ALL_FUNC_OPS, func_ops

    return DialectGeneration(func_ops, list(ALL_FUNC_OPS), None)


def _load_encoding_generation() -> DialectGeneration:
    from loom.dialect.encoding import ALL_ENCODING_OPS, encoding_ops

    return DialectGeneration(encoding_ops, list(ALL_ENCODING_OPS), None)


def _load_pool_generation() -> DialectGeneration:
    from loom.dialect.pool import ALL_POOL_OPS, pool_ops

    return DialectGeneration(pool_ops, list(ALL_POOL_OPS), None)


def _load_global_generation() -> DialectGeneration:
    from loom.dialect.globals import ALL_GLOBAL_OPS, global_ops

    return DialectGeneration(global_ops, list(ALL_GLOBAL_OPS), None)


def _load_scf_generation() -> DialectGeneration:
    from loom.dialect.scf import ALL_SCF_OPS, scf_ops

    return DialectGeneration(scf_ops, list(ALL_SCF_OPS), None)


def _load_cfg_generation() -> DialectGeneration:
    from loom.dialect.cfg import ALL_CFG_OPS, cfg_ops

    return DialectGeneration(cfg_ops, list(ALL_CFG_OPS), None)


def _load_check_generation() -> DialectGeneration:
    from loom.dialect.check import ALL_CHECK_OPS, check_ops

    return DialectGeneration(check_ops, list(ALL_CHECK_OPS), None)


def _load_buffer_generation() -> DialectGeneration:
    from loom.dialect.buffer import ALL_BUFFER_OPS, buffer_ops

    return DialectGeneration(buffer_ops, list(ALL_BUFFER_OPS), None)


def _load_view_generation() -> DialectGeneration:
    from loom.dialect.view import ALL_VIEW_OPS, view_ops

    return DialectGeneration(view_ops, list(ALL_VIEW_OPS), None)


def _load_vector_generation() -> DialectGeneration:
    from loom.dialect.vector import ALL_VECTOR_OPS, VECTOR_OP_CATEGORY_GROUPS, vector_ops

    return DialectGeneration(vector_ops, list(ALL_VECTOR_OPS), VECTOR_OP_CATEGORY_GROUPS)


def _load_index_generation() -> DialectGeneration:
    from loom.dialect.index import ALL_INDEX_OPS, index_ops

    return DialectGeneration(index_ops, list(ALL_INDEX_OPS), None)


def _load_kernel_generation() -> DialectGeneration:
    from loom.dialect.kernel import ALL_KERNEL_OPS, kernel_ops

    return DialectGeneration(kernel_ops, list(ALL_KERNEL_OPS), None)


def _load_llvmir_generation() -> DialectGeneration:
    from loom.dialect.llvmir import ALL_LLVMIR_OPS, llvmir_ops

    return DialectGeneration(llvmir_ops, list(ALL_LLVMIR_OPS), None)


def _load_target_generation() -> DialectGeneration:
    from loom.dialect.target import ALL_TARGET_OPS, target_ops

    return DialectGeneration(target_ops, list(ALL_TARGET_OPS), None)


def _load_low_generation() -> DialectGeneration:
    from loom.dialect.low import ALL_LOW_OPS, low_ops

    return DialectGeneration(low_ops, list(ALL_LOW_OPS), None)


def _load_pass_generation() -> DialectGeneration:
    from loom.dialect.pass_ import ALL_PASS_OPS, pass_ops

    return DialectGeneration(pass_ops, list(ALL_PASS_OPS), None)


def _load_config_generation() -> DialectGeneration:
    from loom.dialect.config import ALL_CONFIG_OPS, config_ops

    return DialectGeneration(config_ops, list(ALL_CONFIG_OPS), None)


def _load_sanitizer_generation() -> DialectGeneration:
    from loom.dialect.sanitizer import ALL_SANITIZER_OPS, sanitizer_ops

    return DialectGeneration(sanitizer_ops, list(ALL_SANITIZER_OPS), None)


def _load_amdgpu_generation() -> DialectGeneration:
    from loom.target.arch.amdgpu.dialect import ALL_AMDGPU_OPS, amdgpu_ops

    return DialectGeneration(amdgpu_ops, list(ALL_AMDGPU_OPS), None)


def _load_x86_generation() -> DialectGeneration:
    from loom.target.arch.x86.dialect import ALL_X86_OPS, x86_ops

    return DialectGeneration(x86_ops, list(ALL_X86_OPS), None)


def _load_spirv_generation() -> DialectGeneration:
    from loom.target.arch.spirv.dialect import ALL_SPIRV_OPS, spirv_ops

    return DialectGeneration(spirv_ops, list(ALL_SPIRV_OPS), None)


def _load_wasm_generation() -> DialectGeneration:
    from loom.target.arch.wasm.dialect import ALL_WASM_OPS, wasm_ops

    return DialectGeneration(wasm_ops, list(ALL_WASM_OPS), None)


def _load_ireevm_generation() -> DialectGeneration:
    from loom.target.arch.ireevm.dialect import ALL_IREEVM_OPS, ireevm_ops

    return DialectGeneration(ireevm_ops, list(ALL_IREEVM_OPS), None)


_DIALECT_GENERATION_LOADERS: tuple[tuple[str, DialectGenerationLoader], ...] = (
    ("test", _load_test_generation),
    ("scalar", _load_scalar_generation),
    ("func", _load_func_generation),
    ("encoding", _load_encoding_generation),
    ("pool", _load_pool_generation),
    ("global", _load_global_generation),
    ("scf", _load_scf_generation),
    ("cfg", _load_cfg_generation),
    ("check", _load_check_generation),
    ("buffer", _load_buffer_generation),
    ("view", _load_view_generation),
    ("vector", _load_vector_generation),
    ("index", _load_index_generation),
    ("kernel", _load_kernel_generation),
    ("llvmir", _load_llvmir_generation),
    ("target", _load_target_generation),
    ("low", _load_low_generation),
    ("pass", _load_pass_generation),
    ("config", _load_config_generation),
    ("sanitizer", _load_sanitizer_generation),
    ("amdgpu", _load_amdgpu_generation),
    ("x86", _load_x86_generation),
    ("spirv", _load_spirv_generation),
    ("wasm", _load_wasm_generation),
    ("ireevm", _load_ireevm_generation),
)


def dialect_names() -> tuple[str, ...]:
    """Returns dialect names accepted by selected C table generation."""
    return tuple(name for name, _ in _DIALECT_GENERATION_LOADERS)


def _load_all_types() -> list[Any]:
    from loom.builtin_types import ALL_BUILTIN_TYPES
    from loom.dialect.hal import ALL_HAL_TYPES
    from loom.dialect.kernel import ALL_KERNEL_TYPES
    from loom.target.arch.ireevm.dialect import ALL_IREEVM_TYPES

    return [
        *ALL_BUILTIN_TYPES,
        *ALL_HAL_TYPES,
        *ALL_KERNEL_TYPES,
        *ALL_IREEVM_TYPES,
    ]


def load_generation_model() -> GenerationModel:
    """Loads all Python DSL declarations needed for C table generation."""
    return GenerationModel(
        dialects=[loader() for _, loader in _DIALECT_GENERATION_LOADERS],
        types=_load_all_types(),
    )


def load_dialect_generation(dialect_name: str) -> DialectGeneration:
    """Loads the Python DSL declarations for one dialect."""
    loaders = dict(_DIALECT_GENERATION_LOADERS)
    loader = loaders.get(dialect_name)
    if loader is None:
        raise ValueError(f"unknown dialect {dialect_name!r}; expected one of {', '.join(dialect_names())}")
    return loader()
