# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Registry for checked-in low descriptor sets.

This module only registers descriptor sets whose source of truth is checked-in
Python data. Targets that require vendor inputs, such as AMDGPU ISA XML, keep
their own generator entry points so they can make the external data dependency
explicit in the build graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import cast

from loom.target.low_descriptors import DescriptorSet


@dataclass(frozen=True)
class DescriptorSetRegistration:
    key: str
    module_name: str
    symbol_name: str
    aliases: tuple[str, ...] = ()
    generates_checked_in_c: bool = True

    def load(self) -> DescriptorSet:
        module = import_module(self.module_name)
        return cast(DescriptorSet, getattr(module, self.symbol_name))


DESCRIPTOR_SET_REGISTRATIONS = (
    DescriptorSetRegistration(
        key="ireevm.core",
        module_name="loom.target.arch.ireevm.descriptors",
        symbol_name="IREEVM_CORE_DESCRIPTOR_SET",
        aliases=("ireevm_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="llvmir.generic.core",
        module_name="loom.target.arch.llvmir.descriptors",
        symbol_name="LLVMIR_GENERIC_CORE_DESCRIPTOR_SET",
        aliases=("llvmir_generic_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="wasm.core.simd128",
        module_name="loom.target.arch.wasm.descriptors",
        symbol_name="WASM_CORE_SIMD128_DESCRIPTOR_SET",
        aliases=("wasm_core_simd128",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="spirv.logical.core",
        module_name="loom.target.arch.spirv.descriptors",
        symbol_name="SPIRV_LOGICAL_CORE_DESCRIPTOR_SET",
        aliases=("spirv_logical_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.scalar.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_SCALAR_DESCRIPTOR_SET",
        aliases=("x86_scalar_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.simd128.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_SIMD128_DESCRIPTOR_SET",
        aliases=("x86_simd128_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx2.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX2_DESCRIPTOR_SET",
        aliases=("x86_avx2_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx512.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX512_CORE_DESCRIPTOR_SET",
        aliases=("x86_avx512_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.packed_dot.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_PACKED_DOT_DESCRIPTOR_SET",
        aliases=("x86_packed_dot_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx512_vnni.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX512_VNNI_DESCRIPTOR_SET",
        aliases=("x86_avx512_vnni_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx512_bf16.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX512_BF16_DESCRIPTOR_SET",
        aliases=("x86_avx512_bf16_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx_vnni.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX_VNNI_DESCRIPTOR_SET",
        aliases=("x86_avx_vnni_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx_vnni_int8.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX_VNNI_INT8_DESCRIPTOR_SET",
        aliases=("x86_avx_vnni_int8_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx_vnni_int16.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX_VNNI_INT16_DESCRIPTOR_SET",
        aliases=("x86_avx_vnni_int16_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx10_2.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX10_2_DESCRIPTOR_SET",
        aliases=("x86_avx10_2_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="x86.avx512_packed_dot.core",
        module_name="loom.target.arch.x86.descriptors",
        symbol_name="X86_AVX512_PACKED_DOT_DESCRIPTOR_SET",
        aliases=("x86_avx512_packed_dot_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="test.low.core",
        module_name="loom.target.test.descriptors",
        symbol_name="TEST_LOW_CORE_DESCRIPTOR_SET",
        aliases=("test_low_core",),
        generates_checked_in_c=False,
    ),
    DescriptorSetRegistration(
        key="test.low.alt",
        module_name="loom.target.test.descriptors",
        symbol_name="TEST_LOW_ALT_DESCRIPTOR_SET",
        aliases=("test_low_alt",),
        generates_checked_in_c=False,
    ),
)


def iter_descriptor_set_registrations() -> Iterable[DescriptorSetRegistration]:
    return DESCRIPTOR_SET_REGISTRATIONS


def iter_descriptor_sets() -> Iterable[DescriptorSet]:
    for registration in DESCRIPTOR_SET_REGISTRATIONS:
        yield registration.load()


def iter_checked_in_c_descriptor_sets() -> Iterable[DescriptorSet]:
    for registration in DESCRIPTOR_SET_REGISTRATIONS:
        if registration.generates_checked_in_c:
            yield registration.load()


def descriptor_set_names() -> tuple[str, ...]:
    names: list[str] = []
    for registration in DESCRIPTOR_SET_REGISTRATIONS:
        names.append(registration.key)
        names.extend(registration.aliases)
    return tuple(names)


def resolve_descriptor_set(name: str) -> DescriptorSet:
    for registration in DESCRIPTOR_SET_REGISTRATIONS:
        if name == registration.key or name in registration.aliases:
            return registration.load()
    supported = ", ".join(descriptor_set_names())
    raise ValueError(
        f"unknown low descriptor set '{name}'; expected one of: {supported}"
    )
