# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU source value materializers used by contract fragments."""

from loom.target.contracts import ValueMaterializer

I32_VGPR_MATERIALIZER = ValueMaterializer(
    name="i32_vgpr",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_i32",
    materialize="loom_amdgpu_lookup_or_materialize_vgpr_i32",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

F32_VGPR_MATERIALIZER = ValueMaterializer(
    name="f32_vgpr",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_f32",
    materialize="loom_amdgpu_lookup_or_materialize_vgpr_f32",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

I64_VGPR_MATERIALIZER = ValueMaterializer(
    name="i64_vgpr",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_i64",
    materialize="loom_amdgpu_lookup_or_materialize_vgpr_i64",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

ADDRESS_VGPR_MATERIALIZER = ValueMaterializer(
    name="address_vgpr",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_address",
    materialize="loom_amdgpu_lookup_or_materialize_vgpr_address",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

I1_NATIVE_MASK_MATERIALIZER = ValueMaterializer(
    name="i1_native_mask",
    can_materialize="loom_amdgpu_value_can_materialize_as_native_i1_mask",
    materialize="loom_amdgpu_lookup_or_materialize_native_i1_mask",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)
