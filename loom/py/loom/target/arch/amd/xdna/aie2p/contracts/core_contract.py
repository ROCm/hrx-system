# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P core contract composition."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.target.arch.amd.xdna.aie2p.contracts import core as core_rules
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    AIE2P_CONVERSION_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.f32 import AIE2P_F32_RULES
from loom.target.arch.amd.xdna.aie2p.contracts.f32_compare import (
    AIE2P_F32_COMPARE_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.floating import (
    AIE2P_BF16_MATRIX_RULES,
    AIE2P_FLOATING_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.i64 import AIE2P_I64_RULES
from loom.target.arch.amd.xdna.aie2p.contracts.index_conversion import (
    AIE2P_INDEX_CONVERSION_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.memory import AIE2P_MEMORY_RULES
from loom.target.arch.amd.xdna.aie2p.contracts.nonlinear import (
    AIE2P_NONLINEAR_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packed_dot import (
    AIE2P_PACKED_DOT_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.reduction import (
    AIE2P_REDUCTION_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.structural import (
    AIE2P_STRUCTURAL_RULES,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractCase,
    ContractFragment,
    DescriptorMatrixRule,
    ValueAliasRule,
    ValueProject,
    ValueRef,
)


def aie2p_core_cases() -> Sequence[ContractCase]:
    # Specialized cases precede general cases because the compact runtime table
    # is queried in authored order.
    return (
        ValueAliasRule(
            source_op=buffer.buffer_view,
            source=ValueRef.operand("buffer"),
            result=ValueRef.result("result"),
        ),
        ValueAliasRule(
            source_op=view.view_subview,
            source=ValueRef.operand("source"),
            result=ValueRef.result("result"),
        ),
        ValueAliasRule(
            source_op=view.view_refine,
            source=ValueRef.operand("source"),
            result=ValueRef.result("result"),
        ),
        *AIE2P_BF16_MATRIX_RULES,
        DescriptorMatrixRule(
            source_op=vector.vector_mma,
            source="vector_mma",
        ),
        core_rules._matrix_fragment_store_rule(),
        *AIE2P_PACKED_DOT_RULES,
        *AIE2P_REDUCTION_RULES,
        *AIE2P_STRUCTURAL_RULES,
        *AIE2P_I64_RULES,
        *AIE2P_MEMORY_RULES,
        *(
            core_rules._address_constant_rule(
                result_type,
                descriptor_key,
                minimum,
                maximum,
            )
            for result_type, minimum, maximum in (
                (core_rules._INDEX, core_rules._SHORT_MIN, core_rules._SHORT_MAX),
                (core_rules._OFFSET, 0, core_rules._SHORT_MAX),
            )
            for descriptor_key in ("amd.xdna.aie2p.constant.i32.short",)
        ),
        *(
            core_rules._address_constant_rule(
                result_type,
                "amd.xdna.aie2p.constant.i32",
                minimum,
                maximum,
            )
            for result_type, minimum, maximum in (
                (core_rules._INDEX, core_rules._I32_MIN, core_rules._I32_MAX),
                (core_rules._OFFSET, 0, core_rules._I32_MAX),
            )
        ),
        *AIE2P_INDEX_CONVERSION_RULES,
        *(
            core_rules._binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, descriptor_key in (
                (index.index_add, "amd.xdna.aie2p.add.i32"),
                (index.index_sub, "amd.xdna.aie2p.sub.i32"),
            )
            for type_pattern in (core_rules._INDEX, core_rules._OFFSET)
        ),
        core_rules._binary_rule(
            index.index_mul,
            core_rules._INDEX,
            "amd.xdna.aie2p.mul.i32",
        ),
        core_rules._index_scale_rule(),
        core_rules._unsigned_division_rule(
            index.index_div,
            core_rules._INDEX,
            return_quotient=True,
        ),
        core_rules._unsigned_division_rule(
            index.index_rem,
            core_rules._INDEX,
            return_quotient=False,
        ),
        core_rules._integer_minmax_rule(
            index.index_min,
            core_rules._INDEX,
            "amd.xdna.aie2p.cmp.slt.i32.select",
            swap_compare_operands=False,
        ),
        core_rules._integer_minmax_rule(
            index.index_max,
            core_rules._INDEX,
            "amd.xdna.aie2p.cmp.slt.i32.select",
            swap_compare_operands=True,
        ),
        core_rules._madd_rule(index.index_madd, core_rules._INDEX),
        *(
            core_rules._binary_rule(source_op, core_rules._INDEX, descriptor_key)
            for source_op, descriptor_key in (
                (index.index_andi, "amd.xdna.aie2p.and.i32"),
                (index.index_ori, "amd.xdna.aie2p.or.i32"),
                (index.index_xori, "amd.xdna.aie2p.xor.i32"),
                (index.index_shli, "amd.xdna.aie2p.lshl.i32"),
            )
        ),
        core_rules._right_shift_rule(
            index.index_shrsi,
            core_rules._INDEX,
            "amd.xdna.aie2p.ashl.i32",
        ),
        core_rules._right_shift_rule(
            index.index_shrui,
            core_rules._INDEX,
            "amd.xdna.aie2p.lshl.i32",
        ),
        core_rules._rotate_rule(index.index_rotli, core_rules._INDEX, rotate_left=True),
        core_rules._rotate_rule(
            index.index_rotri, core_rules._INDEX, rotate_left=False
        ),
        core_rules._unary_rule(
            index.index_ctlzi,
            core_rules._INDEX,
            "amd.xdna.aie2p.clz.i32",
        ),
        core_rules._count_trailing_zeros_rule(index.index_cttzi, core_rules._INDEX),
        core_rules._unary_rule(
            index.index_ctpopi,
            core_rules._INDEX,
            "amd.xdna.aie2p.popcount.i32",
        ),
        *(
            core_rules._compare_rule(
                index.index_cmp,
                type_pattern,
                predicate,
                descriptor_key,
                swap_operands=swap_operands,
            )
            for type_pattern in (core_rules._INDEX, core_rules._OFFSET)
            for predicate, descriptor_key, swap_operands in (
                ("eq", "amd.xdna.aie2p.cmp.eq.i32", False),
                ("ne", "amd.xdna.aie2p.cmp.ne.i32", False),
                ("slt", "amd.xdna.aie2p.cmp.slt.i32", False),
                ("sle", "amd.xdna.aie2p.cmp.sge.i32", True),
                ("sgt", "amd.xdna.aie2p.cmp.slt.i32", True),
                ("sge", "amd.xdna.aie2p.cmp.sge.i32", False),
                ("ult", "amd.xdna.aie2p.cmp.ult.i32", False),
                ("ule", "amd.xdna.aie2p.cmp.uge.i32", True),
                ("ugt", "amd.xdna.aie2p.cmp.ult.i32", True),
                ("uge", "amd.xdna.aie2p.cmp.uge.i32", False),
            )
        ),
        core_rules._logical_constant_rule(),
        core_rules._constant_rule(
            core_rules._I8,
            "amd.xdna.aie2p.constant.i32.short",
            core_rules._I8_MIN,
            core_rules._I8_MAX,
        ),
        core_rules._constant_rule(
            core_rules._I16,
            "amd.xdna.aie2p.constant.i32.short",
            core_rules._SHORT_MIN,
            core_rules._SHORT_MAX,
        ),
        core_rules._constant_rule(
            core_rules._I16,
            "amd.xdna.aie2p.constant.i32",
            core_rules._I16_MIN,
            core_rules._I16_MAX,
        ),
        core_rules._constant_rule(
            core_rules._I32,
            "amd.xdna.aie2p.constant.i32.short",
            core_rules._SHORT_MIN,
            core_rules._SHORT_MAX,
        ),
        core_rules._constant_rule(
            core_rules._I32,
            "amd.xdna.aie2p.constant.i32",
            core_rules._I32_MIN,
            core_rules._I32_MAX,
        ),
        core_rules._float_constant_rule(
            core_rules._F16, ValueProject.float_as_f16_bits("result")
        ),
        core_rules._float_constant_rule(
            core_rules._BF16, ValueProject.float_as_bf16_bits("result")
        ),
        core_rules._float_constant_rule(
            core_rules._F32, ValueProject.float_as_f32_i32("result")
        ),
        *AIE2P_CONVERSION_RULES,
        core_rules._vector_constant_rule(
            core_rules._I8_VECTOR,
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i8x64",
            core_rules._I8_MIN,
            core_rules._I8_MAX,
        ),
        core_rules._vector_constant_rule(
            core_rules._I16_VECTOR,
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i16x32",
            core_rules._SHORT_MIN,
            core_rules._SHORT_MAX,
        ),
        core_rules._vector_constant_rule(
            core_rules._I16_VECTOR,
            "amd.xdna.aie2p.constant.i32",
            "amd.xdna.aie2p.splat.i16x32",
            core_rules._I16_MIN,
            core_rules._I16_MAX,
        ),
        core_rules._vector_constant_rule(
            core_rules._I32_VECTOR,
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i32x16",
            core_rules._SHORT_MIN,
            core_rules._SHORT_MAX,
        ),
        core_rules._vector_constant_rule(
            core_rules._I32_VECTOR,
            "amd.xdna.aie2p.constant.i32",
            "amd.xdna.aie2p.splat.i32x16",
            core_rules._I32_MIN,
            core_rules._I32_MAX,
        ),
        core_rules._float_vector_constant_rule(
            core_rules._F16_VECTOR,
            "amd.xdna.aie2p.splat.i16x32",
            ValueProject.float_as_f16_bits("result"),
        ),
        core_rules._float_vector_constant_rule(
            core_rules._BF16_VECTOR,
            "amd.xdna.aie2p.splat.i16x32",
            ValueProject.float_as_bf16_bits("result"),
        ),
        core_rules._float_vector_constant_rule(
            core_rules._F32_VECTOR,
            "amd.xdna.aie2p.splat.i32x16",
            ValueProject.float_as_f32_i32("result"),
        ),
        *core_rules._vector_broadcast_alias_rules(),
        *(
            core_rules._vector_broadcast_rule(
                element_type, maximum_lanes, descriptor_key
            )
            for element_type, maximum_lanes, descriptor_key in (
                (
                    "i8",
                    64,
                    "amd.xdna.aie2p.broadcast.i8x64.from-vector",
                ),
                (
                    "i16",
                    32,
                    "amd.xdna.aie2p.broadcast.i16x32.from-vector",
                ),
                (
                    "i32",
                    16,
                    "amd.xdna.aie2p.broadcast.i32x16.from-vector",
                ),
            )
        ),
        *(
            rule
            for (
                scalar_type,
                vector_type,
                maximum_index,
                immediate_key,
                register_key,
            ) in (
                (
                    core_rules._I8,
                    core_rules._I8_VECTOR,
                    63,
                    "amd.xdna.aie2p.extract.i8.immediate",
                    "amd.xdna.aie2p.extract.i8.register",
                ),
                (
                    core_rules._F8E4M3,
                    core_rules._F8E4M3_VECTOR,
                    63,
                    "amd.xdna.aie2p.extract.i8.immediate",
                    "amd.xdna.aie2p.extract.i8.register",
                ),
                (
                    core_rules._F8E5M2,
                    core_rules._F8E5M2_VECTOR,
                    63,
                    "amd.xdna.aie2p.extract.i8.immediate",
                    "amd.xdna.aie2p.extract.i8.register",
                ),
                (
                    core_rules._I16,
                    core_rules._I16_VECTOR,
                    31,
                    "amd.xdna.aie2p.extract.i16.immediate",
                    "amd.xdna.aie2p.extract.i16.register",
                ),
                (
                    core_rules._F16,
                    core_rules._F16_VECTOR,
                    31,
                    "amd.xdna.aie2p.extract.i16.immediate",
                    "amd.xdna.aie2p.extract.i16.register",
                ),
                (
                    core_rules._BF16,
                    core_rules._BF16_VECTOR,
                    31,
                    "amd.xdna.aie2p.extract.i16.immediate",
                    "amd.xdna.aie2p.extract.i16.register",
                ),
                (
                    core_rules._I32,
                    core_rules._I32_VECTOR,
                    15,
                    "amd.xdna.aie2p.extract.i32.immediate",
                    "amd.xdna.aie2p.extract.i32.register",
                ),
                (
                    core_rules._F32,
                    core_rules._F32_VECTOR,
                    15,
                    "amd.xdna.aie2p.extract.i32.immediate",
                    "amd.xdna.aie2p.extract.i32.register",
                ),
            )
            for rule in (
                core_rules._vector_extract_static_rule(
                    vector_type,
                    scalar_type,
                    maximum_index,
                    immediate_key,
                ),
                core_rules._vector_extract_dynamic_rule(
                    vector_type,
                    scalar_type,
                    register_key,
                ),
            )
        ),
        core_rules._vector_predicate_extract_rule(dynamic_index=False),
        core_rules._vector_predicate_extract_rule(dynamic_index=True),
        core_rules._vector_extract_static_rule(
            core_rules._I1X2X64_VECTOR,
            core_rules._I1_VECTOR,
            1,
            "amd.xdna.aie2p.extract.predicate64.immediate",
        ),
        core_rules._vector_insert_zero_rule(
            core_rules._BF16,
            core_rules._BF16X8_VECTOR,
            "amd.xdna.aie2p.insert.bf16x8.zero",
        ),
        core_rules._vector_insert_static_rule(
            core_rules._BF16,
            core_rules._BF16X8_VECTOR,
            7,
            "amd.xdna.aie2p.insert.bf16x8.register",
        ),
        core_rules._vector_insert_dynamic_rule(
            core_rules._BF16,
            core_rules._BF16X8_VECTOR,
            "amd.xdna.aie2p.insert.bf16x8.register",
        ),
        *(
            rule
            for scalar_type, vector_type, maximum_index, zero_key, register_key in (
                (
                    core_rules._I8,
                    core_rules._I8_VECTOR,
                    63,
                    "amd.xdna.aie2p.insert.i8.zero",
                    "amd.xdna.aie2p.insert.i8.register",
                ),
                (
                    core_rules._F8E4M3,
                    core_rules._F8E4M3_VECTOR,
                    63,
                    "amd.xdna.aie2p.insert.i8.zero",
                    "amd.xdna.aie2p.insert.i8.register",
                ),
                (
                    core_rules._F8E5M2,
                    core_rules._F8E5M2_VECTOR,
                    63,
                    "amd.xdna.aie2p.insert.i8.zero",
                    "amd.xdna.aie2p.insert.i8.register",
                ),
                (
                    core_rules._I16,
                    core_rules._I16_VECTOR,
                    31,
                    "amd.xdna.aie2p.insert.i16.zero",
                    "amd.xdna.aie2p.insert.i16.register",
                ),
                (
                    core_rules._F16,
                    core_rules._F16_VECTOR,
                    31,
                    "amd.xdna.aie2p.insert.i16.zero",
                    "amd.xdna.aie2p.insert.i16.register",
                ),
                (
                    core_rules._BF16,
                    core_rules._BF16_VECTOR,
                    31,
                    "amd.xdna.aie2p.insert.i16.zero",
                    "amd.xdna.aie2p.insert.i16.register",
                ),
                (
                    core_rules._I32,
                    core_rules._I32_VECTOR,
                    15,
                    "amd.xdna.aie2p.insert.i32.zero",
                    "amd.xdna.aie2p.insert.i32.register",
                ),
                (
                    core_rules._F32,
                    core_rules._F32_VECTOR,
                    15,
                    "amd.xdna.aie2p.insert.i32.zero",
                    "amd.xdna.aie2p.insert.i32.register",
                ),
            )
            for rule in (
                core_rules._vector_insert_zero_rule(scalar_type, vector_type, zero_key),
                core_rules._vector_insert_static_rule(
                    scalar_type,
                    vector_type,
                    maximum_index,
                    register_key,
                ),
                core_rules._vector_insert_dynamic_rule(
                    scalar_type,
                    vector_type,
                    register_key,
                ),
            )
        ),
        core_rules._binary_rule(
            scalar_arithmetic.scalar_addi,
            core_rules._I32,
            "amd.xdna.aie2p.add.i32",
        ),
        core_rules._binary_rule(
            scalar_arithmetic.scalar_subi,
            core_rules._I32,
            "amd.xdna.aie2p.sub.i32",
        ),
        core_rules._binary_rule(
            scalar_arithmetic.scalar_muli,
            core_rules._I32,
            "amd.xdna.aie2p.mul.i32",
        ),
        core_rules._unsigned_division_rule(
            scalar_arithmetic.scalar_divui,
            core_rules._I32,
            return_quotient=True,
        ),
        core_rules._unsigned_division_rule(
            scalar_arithmetic.scalar_remui,
            core_rules._I32,
            return_quotient=False,
        ),
        core_rules._unary_rule(
            scalar_arithmetic.scalar_absi,
            core_rules._I32,
            "amd.xdna.aie2p.abs.i32",
        ),
        *(
            core_rules._integer_minmax_rule(
                source_op,
                core_rules._I32,
                descriptor_key,
                swap_compare_operands=swap_compare_operands,
            )
            for source_op, descriptor_key, swap_compare_operands in (
                (
                    scalar_arithmetic.scalar_minsi,
                    "amd.xdna.aie2p.cmp.slt.i32.select",
                    False,
                ),
                (
                    scalar_arithmetic.scalar_maxsi,
                    "amd.xdna.aie2p.cmp.slt.i32.select",
                    True,
                ),
                (
                    scalar_arithmetic.scalar_minui,
                    "amd.xdna.aie2p.cmp.ult.i32.select",
                    False,
                ),
                (
                    scalar_arithmetic.scalar_maxui,
                    "amd.xdna.aie2p.cmp.ult.i32.select",
                    True,
                ),
            )
        ),
        core_rules._madd_rule(scalar_arithmetic.scalar_fmai, core_rules._I32),
        *(
            core_rules._binary_rule(source_op, type_pattern, descriptor_key)
            for type_pattern in (core_rules._I8, core_rules._I16)
            for source_op, descriptor_key in (
                (scalar_arithmetic.scalar_addi, "amd.xdna.aie2p.add.i32"),
                (scalar_arithmetic.scalar_subi, "amd.xdna.aie2p.sub.i32"),
                (scalar_arithmetic.scalar_muli, "amd.xdna.aie2p.mul.i32"),
            )
        ),
        *(
            core_rules._vector_binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, type_pattern, descriptor_key in (
                (
                    vector.vector_addi,
                    core_rules._I8_VECTOR,
                    "amd.xdna.aie2p.add.i8x64",
                ),
                (
                    vector.vector_subi,
                    core_rules._I8_VECTOR,
                    "amd.xdna.aie2p.sub.i8x64",
                ),
                (
                    vector.vector_addi,
                    core_rules._I16_VECTOR,
                    "amd.xdna.aie2p.add.i16x32",
                ),
                (
                    vector.vector_subi,
                    core_rules._I16_VECTOR,
                    "amd.xdna.aie2p.sub.i16x32",
                ),
                (
                    vector.vector_addi,
                    core_rules._I32_VECTOR,
                    "amd.xdna.aie2p.add.i32x16",
                ),
                (
                    vector.vector_subi,
                    core_rules._I32_VECTOR,
                    "amd.xdna.aie2p.sub.i32x16",
                ),
            )
        ),
        *(
            core_rules._vector_binary_rule(source_op, vector_type, descriptor_key)
            for width, vector_type in (
                (8, core_rules._I8_VECTOR),
                (16, core_rules._I16_VECTOR),
                (32, core_rules._I32_VECTOR),
            )
            for source_op, operation, signedness in (
                (vector.vector_minsi, "min", "signed"),
                (vector.vector_maxsi, "max", "signed"),
                (vector.vector_minui, "min", "unsigned"),
                (vector.vector_maxui, "max", "unsigned"),
            )
            for descriptor_key in (
                f"amd.xdna.aie2p.{operation}.{signedness}.i{width}x{512 // width}",
            )
        ),
        core_rules._vector_multiply_i16_rule(),
        core_rules._vector_bitunpack_i4_rule(
            vector.vector_bitunpacku,
            "amd.xdna.aie2p.unpack.u4x64.to.u8x64.configured",
        ),
        core_rules._vector_bitunpack_i4_rule(
            vector.vector_bitunpacks,
            "amd.xdna.aie2p.unpack.s4x64.to.s8x64.configured",
        ),
        core_rules._vector_bitunpack_i1_alias_rule(),
        *AIE2P_F32_COMPARE_RULES,
        *AIE2P_F32_RULES,
        *AIE2P_NONLINEAR_RULES,
        core_rules._matrix_accumulator_zero_rule(),
        *AIE2P_FLOATING_RULES,
        *(
            core_rules._vector_binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, type_pattern, descriptor_key in (
                (
                    vector.vector_andi,
                    core_rules._I8_VECTOR,
                    "amd.xdna.aie2p.and.bits512",
                ),
                (
                    vector.vector_andi,
                    core_rules._I16_VECTOR,
                    "amd.xdna.aie2p.and.bits512",
                ),
                (
                    vector.vector_andi,
                    core_rules._I32_VECTOR,
                    "amd.xdna.aie2p.and.bits512",
                ),
                (
                    vector.vector_ori,
                    core_rules._I8_VECTOR,
                    "amd.xdna.aie2p.or.bits512",
                ),
                (
                    vector.vector_ori,
                    core_rules._I16_VECTOR,
                    "amd.xdna.aie2p.or.bits512",
                ),
                (
                    vector.vector_ori,
                    core_rules._I32_VECTOR,
                    "amd.xdna.aie2p.or.bits512",
                ),
            )
        ),
        *(
            core_rules._vector_xor_rule(type_pattern, subtract_descriptor_key)
            for type_pattern, subtract_descriptor_key in (
                (core_rules._I8_VECTOR, "amd.xdna.aie2p.sub.i8x64"),
                (core_rules._I16_VECTOR, "amd.xdna.aie2p.sub.i16x32"),
                (core_rules._I32_VECTOR, "amd.xdna.aie2p.sub.i32x16"),
            )
        ),
        *(
            core_rules._vector_splat_rule(scalar_type, result_type, descriptor_key)
            for scalar_type, result_type, descriptor_key in (
                (core_rules._I8, core_rules._I8_VECTOR, "amd.xdna.aie2p.splat.i8x64"),
                (
                    core_rules._F8E4M3,
                    core_rules._F8E4M3_VECTOR,
                    "amd.xdna.aie2p.splat.i8x64",
                ),
                (
                    core_rules._F8E5M2,
                    core_rules._F8E5M2_VECTOR,
                    "amd.xdna.aie2p.splat.i8x64",
                ),
                (
                    core_rules._I16,
                    core_rules._I16_VECTOR,
                    "amd.xdna.aie2p.splat.i16x32",
                ),
                (
                    core_rules._F16,
                    core_rules._F16_VECTOR,
                    "amd.xdna.aie2p.splat.i16x32",
                ),
                (
                    core_rules._BF16,
                    core_rules._BF16_VECTOR,
                    "amd.xdna.aie2p.splat.i16x32",
                ),
                (
                    core_rules._I32,
                    core_rules._I32_VECTOR,
                    "amd.xdna.aie2p.splat.i32x16",
                ),
                (
                    core_rules._F32,
                    core_rules._F32_VECTOR,
                    "amd.xdna.aie2p.splat.i32x16",
                ),
            )
        ),
        core_rules._vector_predicate_splat_rule(),
        *(
            core_rules._vector_select_rule(value_type, descriptor_key)
            for value_type, descriptor_key in (
                (core_rules._I8_VECTOR, "amd.xdna.aie2p.select.i8x64"),
                (core_rules._F8E4M3_VECTOR, "amd.xdna.aie2p.select.i8x64"),
                (core_rules._F8E5M2_VECTOR, "amd.xdna.aie2p.select.i8x64"),
                (core_rules._I16_VECTOR, "amd.xdna.aie2p.select.i16x32.mask64"),
                (core_rules._F16_VECTOR, "amd.xdna.aie2p.select.i16x32.mask64"),
                (core_rules._BF16_VECTOR, "amd.xdna.aie2p.select.i16x32.mask64"),
                (core_rules._I32_VECTOR, "amd.xdna.aie2p.select.i32x16.mask64"),
                (core_rules._F32_VECTOR, "amd.xdna.aie2p.select.i32x16.mask64"),
            )
        ),
        *(
            core_rules._vector_predicate_binary_rule(source_op, operation)
            for source_op, operation in (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
            )
        ),
        *(
            core_rules._vector_compare_rule(predicate, operand_type, width)
            for width, operand_type in (
                (8, core_rules._I8_VECTOR),
                (16, core_rules._I16_VECTOR),
                (32, core_rules._I32_VECTOR),
            )
            for predicate in (
                "eq",
                "ne",
                "slt",
                "sle",
                "sgt",
                "sge",
                "ult",
                "ule",
                "ugt",
                "uge",
            )
        ),
        *(
            core_rules._scalar_select_rule(result_type)
            for result_type in (
                core_rules._I1,
                core_rules._I8,
                core_rules._F8E4M3,
                core_rules._F8E5M2,
                core_rules._I16,
                core_rules._F16,
                core_rules._BF16,
                core_rules._I32,
                core_rules._F32,
                core_rules._INDEX,
                core_rules._OFFSET,
            )
        ),
        *(
            core_rules._whole_vector_select_rule(result_type)
            for result_type in core_rules._BITCAST_VECTOR_TYPES
        ),
        *core_rules._scalar_bitcast_alias_rules(),
        *core_rules._vector_bitcast_alias_rules(),
        *(
            core_rules._binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, type_pattern, descriptor_key in (
                (
                    scalar_bitwise.scalar_andi,
                    core_rules._I32,
                    "amd.xdna.aie2p.and.i32",
                ),
                (
                    scalar_bitwise.scalar_ori,
                    core_rules._I32,
                    "amd.xdna.aie2p.or.i32",
                ),
                (
                    scalar_bitwise.scalar_xori,
                    core_rules._I32,
                    "amd.xdna.aie2p.xor.i32",
                ),
                (
                    scalar_bitwise.scalar_andi,
                    core_rules._I1,
                    "amd.xdna.aie2p.and.i32",
                ),
                (
                    scalar_bitwise.scalar_ori,
                    core_rules._I1,
                    "amd.xdna.aie2p.or.i32",
                ),
                (
                    scalar_bitwise.scalar_xori,
                    core_rules._I1,
                    "amd.xdna.aie2p.xor.i32",
                ),
            )
        ),
        *(
            core_rules._binary_rule(source_op, type_pattern, descriptor_key)
            for type_pattern in (core_rules._I8, core_rules._I16)
            for source_op, descriptor_key in (
                (scalar_bitwise.scalar_andi, "amd.xdna.aie2p.and.i32"),
                (scalar_bitwise.scalar_ori, "amd.xdna.aie2p.or.i32"),
                (scalar_bitwise.scalar_xori, "amd.xdna.aie2p.xor.i32"),
            )
        ),
        core_rules._binary_rule(
            scalar_bitwise.scalar_shli,
            core_rules._I32,
            "amd.xdna.aie2p.lshl.i32",
        ),
        core_rules._right_shift_rule(
            scalar_bitwise.scalar_shrsi,
            core_rules._I32,
            "amd.xdna.aie2p.ashl.i32",
        ),
        core_rules._right_shift_rule(
            scalar_bitwise.scalar_shrui,
            core_rules._I32,
            "amd.xdna.aie2p.lshl.i32",
        ),
        core_rules._rotate_rule(
            scalar_bitwise.scalar_rotli,
            core_rules._I32,
            rotate_left=True,
        ),
        core_rules._rotate_rule(
            scalar_bitwise.scalar_rotri,
            core_rules._I32,
            rotate_left=False,
        ),
        core_rules._unary_rule(
            scalar_bitwise.scalar_ctlzi,
            core_rules._I32,
            "amd.xdna.aie2p.clz.i32",
        ),
        core_rules._count_trailing_zeros_rule(
            scalar_bitwise.scalar_cttzi, core_rules._I32
        ),
        core_rules._unary_rule(
            scalar_bitwise.scalar_ctpopi,
            core_rules._I32,
            "amd.xdna.aie2p.popcount.i32",
        ),
        core_rules._narrow_left_shift_rule(
            core_rules._I8,
            "amd.xdna.aie2p.extend.unsigned.i8",
        ),
        core_rules._narrow_right_shift_rule(
            scalar_bitwise.scalar_shrsi,
            core_rules._I8,
            "amd.xdna.aie2p.extend.signed.i8",
            "amd.xdna.aie2p.extend.unsigned.i8",
            "amd.xdna.aie2p.ashl.i32",
        ),
        core_rules._narrow_right_shift_rule(
            scalar_bitwise.scalar_shrui,
            core_rules._I8,
            "amd.xdna.aie2p.extend.unsigned.i8",
            "amd.xdna.aie2p.extend.unsigned.i8",
            "amd.xdna.aie2p.lshl.i32",
        ),
        core_rules._narrow_left_shift_rule(
            core_rules._I16,
            "amd.xdna.aie2p.extend.unsigned.i16",
        ),
        core_rules._narrow_right_shift_rule(
            scalar_bitwise.scalar_shrsi,
            core_rules._I16,
            "amd.xdna.aie2p.extend.signed.i16",
            "amd.xdna.aie2p.extend.unsigned.i16",
            "amd.xdna.aie2p.ashl.i32",
        ),
        core_rules._narrow_right_shift_rule(
            scalar_bitwise.scalar_shrui,
            core_rules._I16,
            "amd.xdna.aie2p.extend.unsigned.i16",
            "amd.xdna.aie2p.extend.unsigned.i16",
            "amd.xdna.aie2p.lshl.i32",
        ),
        core_rules._bitfield_rule(
            scalar_bitwise.scalar_bitfield_extractu,
            "amd.xdna.aie2p.lshl.i32",
        ),
        core_rules._bitfield_rule(
            scalar_bitwise.scalar_bitfield_extracts,
            "amd.xdna.aie2p.ashl.i32",
        ),
        core_rules._zero_compare_rule(
            "eq",
            "amd.xdna.aie2p.cmp.eqz.i32",
            zero_field="rhs",
        ),
        core_rules._zero_compare_rule(
            "eq",
            "amd.xdna.aie2p.cmp.eqz.i32",
            zero_field="lhs",
        ),
        core_rules._zero_compare_rule(
            "ne",
            "amd.xdna.aie2p.cmp.nez.i32",
            zero_field="rhs",
        ),
        core_rules._zero_compare_rule(
            "ne",
            "amd.xdna.aie2p.cmp.nez.i32",
            zero_field="lhs",
        ),
        *(
            core_rules._compare_rule(
                scalar_comparison.scalar_cmpi,
                core_rules._I32,
                predicate,
                descriptor_key,
                swap_operands=swap_operands,
            )
            for predicate, descriptor_key, swap_operands in (
                ("eq", "amd.xdna.aie2p.cmp.eq.i32", False),
                ("ne", "amd.xdna.aie2p.cmp.ne.i32", False),
                ("slt", "amd.xdna.aie2p.cmp.slt.i32", False),
                ("sle", "amd.xdna.aie2p.cmp.sge.i32", True),
                ("sgt", "amd.xdna.aie2p.cmp.slt.i32", True),
                ("sge", "amd.xdna.aie2p.cmp.sge.i32", False),
                ("ult", "amd.xdna.aie2p.cmp.ult.i32", False),
                ("ule", "amd.xdna.aie2p.cmp.uge.i32", True),
                ("ugt", "amd.xdna.aie2p.cmp.ult.i32", True),
                ("uge", "amd.xdna.aie2p.cmp.uge.i32", False),
            )
        ),
    )


AIE2P_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

AIE2P_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="amd.xdna.aie2p.core",
    descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/amd/xdna/aie2p/contracts/core.h",
    cases=aie2p_core_cases(),
)
