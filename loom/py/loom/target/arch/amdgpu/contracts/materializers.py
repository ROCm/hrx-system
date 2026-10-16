# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU source value materializers used by contract fragments."""

from loom.target.contracts import ValueMaterializer

REGISTERS_VGPR_MATERIALIZER = ValueMaterializer(
    name="registers_vgpr",
    result_type="loom_amdgpu_materialized_vgpr_register_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_registers",
    materialize="loom_amdgpu_lookup_or_materialize_vgpr_registers",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

VOP3_BINARY_RHS_MATERIALIZER = ValueMaterializer(
    name="vop3_binary_rhs",
    result_type="loom_amdgpu_materialized_vop3_binary_rhs_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_vop3_binary_rhs",
    materialize="loom_amdgpu_lookup_or_materialize_vop3_binary_rhs",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

I32_VGPR_MATERIALIZER = ValueMaterializer(
    name="i32_vgpr",
    result_type="loom_amdgpu_materialized_vgpr_register_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_i32",
    prepare="loom_amdgpu_prepare_vgpr_literal",
    materialize="loom_amdgpu_emit_prepared_vgpr_value",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

F32_VGPR_MATERIALIZER = ValueMaterializer(
    name="f32_vgpr",
    result_type="loom_amdgpu_materialized_vgpr_register_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_f32",
    prepare="loom_amdgpu_prepare_vgpr_literal",
    materialize="loom_amdgpu_emit_prepared_vgpr_value",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

I64_VGPR_MATERIALIZER = ValueMaterializer(
    name="i64_vgpr",
    result_type="loom_amdgpu_materialized_vgpr_register_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_i64",
    materialize="loom_amdgpu_lookup_or_materialize_vgpr_i64",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

ADDRESS_VGPR_MATERIALIZER = ValueMaterializer(
    name="address_vgpr",
    result_type="loom_amdgpu_materialized_vgpr_address_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_address",
    prepare="loom_amdgpu_prepare_vgpr_literal",
    materialize="loom_amdgpu_emit_prepared_vgpr_address",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

ADDRESS_REGISTER_MATERIALIZER = ValueMaterializer(
    name="address_register",
    result_type="loom_amdgpu_materialized_address_register_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_vgpr_address",
    materialize="loom_amdgpu_lookup_address_register",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

ADDRESS_SGPR_MATERIALIZER = ValueMaterializer(
    name="address_sgpr",
    result_type="loom_amdgpu_materialized_sgpr_address_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_sgpr_address",
    materialize="loom_amdgpu_lookup_or_materialize_sgpr_address",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)

I1_NATIVE_MASK_MATERIALIZER = ValueMaterializer(
    name="i1_native_mask",
    result_type="loom_amdgpu_materialized_native_i1_mask_type",
    can_materialize="loom_amdgpu_value_can_materialize_as_native_i1_mask",
    prepare="loom_amdgpu_prepare_native_i1_mask",
    materialize="loom_amdgpu_emit_prepared_native_i1_mask",
    header="loom/target/arch/amdgpu/lower/materializers.h",
)
