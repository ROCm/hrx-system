# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang/TIR importer coverage manifest."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum


class CoverageState(Enum):
    """Importer coverage state for one TileLang/TIR construct."""

    SUPPORTED = "supported"
    NORMALIZED = "normalized"
    IGNORED_METADATA = "ignored_metadata"
    TARGET_ONLY = "target_only"
    OPAQUE_FOREIGN = "opaque_foreign"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class OpFamily(Enum):
    """Namespace family for one TileLang/TIR construct."""

    TIR_NODE = "tir_node"
    TIR_OP = "tir_op"
    TILELANG_OP = "tilelang_op"
    TILELANG_TILEOP = "tilelang_tileop"
    ATTRIBUTE = "attribute"


@dataclass(frozen=True, slots=True)
class OpCoverage:
    """Coverage row for one TileLang/TIR construct."""

    name: str
    family: OpFamily
    state: CoverageState
    note: str


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    """Difference between an external TileLang/TVM inventory and the manifest."""

    known: tuple[str, ...]
    missing: tuple[str, ...]


TILELANG_OP_COVERAGE: tuple[OpCoverage, ...] = (
    OpCoverage(
        "tir.PrimFunc",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Kernel/function container; imports to a Loom kernel definition.",
    ),
    OpCoverage(
        "tir.Block",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Structured block wrapper; static alloc_buffers become scratch views "
        "and region contents import directly.",
    ),
    OpCoverage(
        "tir.BlockRealize",
        OpFamily.TIR_NODE,
        CoverageState.NORMALIZED,
        "Block realization wrapper; true-predicate blocks normalize to Block.",
    ),
    OpCoverage(
        "tir.For",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Loop structure; serial and thread-bound forms map to scf/kernel ops.",
    ),
    OpCoverage(
        "tir.IfThenElse",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Structured control flow; imports to scf.if once values are mapped.",
    ),
    OpCoverage(
        "tir.SeqStmt",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Statement list; preserves source order.",
    ),
    OpCoverage(
        "tir.LetStmt",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar binding; imports as the producing expression plus SSA name.",
    ),
    OpCoverage(
        "tir.Evaluate",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Effect statement wrapper for calls and annotations.",
    ),
    OpCoverage(
        "tir.BufferLoad",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Buffer read; maps to view/vector loads depending on result type.",
    ),
    OpCoverage(
        "tir.BufferStore",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Buffer write; maps to view/vector stores depending on value type.",
    ),
    OpCoverage(
        "tir.Allocate",
        OpFamily.TIR_NODE,
        CoverageState.DEFERRED,
        "Local memory allocation requires storage-space and lifetime mapping.",
    ),
    OpCoverage(
        "tir.AttrStmt",
        OpFamily.TIR_NODE,
        CoverageState.NORMALIZED,
        "Thread extent, assumptions, and storage attributes are decoded into "
        "kernel facts.",
    ),
    OpCoverage(
        "tir.Var",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "SSA, loop, buffer, and symbolic dimension references.",
    ),
    OpCoverage(
        "tir.IntImm",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Integer constants.",
    ),
    OpCoverage(
        "tir.FloatImm",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Floating-point constants.",
    ),
    OpCoverage(
        "tir.Cast",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or same-shape vector cast.",
    ),
    OpCoverage(
        "tir.Ramp",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Vector lane ramp; maps to vector.iota or contiguous vector memory bases.",
    ),
    OpCoverage(
        "tir.Broadcast",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Vector broadcast; maps to vector.splat.",
    ),
    OpCoverage(
        "tir.Add",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector arithmetic.",
    ),
    OpCoverage(
        "tir.Sub",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector arithmetic.",
    ),
    OpCoverage(
        "tir.Mul",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector arithmetic.",
    ),
    OpCoverage(
        "tir.Div",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector arithmetic; exact signedness semantics come from dtype.",
    ),
    OpCoverage(
        "tir.FloorDiv",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Index/scalar/vector integer floor division.",
    ),
    OpCoverage(
        "tir.FloorMod",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Index/scalar/vector integer floor remainder.",
    ),
    OpCoverage(
        "tir.Mod",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Index/scalar/vector integer remainder.",
    ),
    OpCoverage(
        "tir.Min",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector min expression.",
    ),
    OpCoverage(
        "tir.Max",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector max expression.",
    ),
    OpCoverage(
        "tir.EQ",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector comparison.",
    ),
    OpCoverage(
        "tir.NE",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector comparison.",
    ),
    OpCoverage(
        "tir.LT",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector comparison.",
    ),
    OpCoverage(
        "tir.LE",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector comparison.",
    ),
    OpCoverage(
        "tir.GT",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector comparison.",
    ),
    OpCoverage(
        "tir.GE",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector comparison.",
    ),
    OpCoverage(
        "tir.And",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector boolean conjunction.",
    ),
    OpCoverage(
        "tir.Or",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector boolean disjunction.",
    ),
    OpCoverage(
        "tir.Not",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector boolean negation.",
    ),
    OpCoverage(
        "tir.Select",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar or vector select expression.",
    ),
    OpCoverage(
        "tir.Call",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Structured call node; dispatches by intrinsic/operator name.",
    ),
    OpCoverage(
        "tir.abs",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar absolute value; unsigned inputs normalize to identity.",
    ),
    OpCoverage(
        "tir.acos",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.acosh",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.asin",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.asinh",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.atan",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.atan2",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.atanh",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.bitwise_and",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index bitwise operation.",
    ),
    OpCoverage(
        "tir.bitwise_or",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index bitwise operation.",
    ),
    OpCoverage(
        "tir.bitwise_xor",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index bitwise operation.",
    ),
    OpCoverage(
        "tir.bitwise_not",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index bitwise not; normalizes to xor with an all-ones constant.",
    ),
    OpCoverage(
        "tir.ceil",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point rounding intrinsic.",
    ),
    OpCoverage(
        "tir.ceildiv",
        OpFamily.TIR_OP,
        CoverageState.NORMALIZED,
        "Scalar ceildiv maps directly; index ceildiv expands to add/sub/div.",
    ),
    OpCoverage(
        "tir.clz",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index count-leading-zero operation.",
    ),
    OpCoverage(
        "tir.copysign",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point sign injection.",
    ),
    OpCoverage(
        "tir.cos",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.cosh",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.div",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar division with C/TIR truncation semantics.",
    ),
    OpCoverage(
        "tir.erf",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.erfc",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.exp",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.exp2",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.exp10",
        OpFamily.TIR_OP,
        CoverageState.DEFERRED,
        "Needs either scalar.exp10f or a reviewed lowering sequence.",
    ),
    OpCoverage(
        "tir.expm1",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.floor",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point rounding intrinsic.",
    ),
    OpCoverage(
        "tir.floordiv",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index floor division.",
    ),
    OpCoverage(
        "tir.floormod",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index floor remainder.",
    ),
    OpCoverage(
        "tir.fma",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar fused multiply-add.",
    ),
    OpCoverage(
        "tir.fmod",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point remainder.",
    ),
    OpCoverage(
        "tir.hypot",
        OpFamily.TIR_OP,
        CoverageState.DEFERRED,
        "Needs either scalar.hypotf or a reviewed lowering sequence.",
    ),
    OpCoverage(
        "tir.if_then_else",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Lazy expression-level selection maps to result-producing scf.if.",
    ),
    OpCoverage(
        "tir.indexdiv",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Non-negative index division.",
    ),
    OpCoverage(
        "tir.indexmod",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Non-negative index remainder.",
    ),
    OpCoverage(
        "tir.isfinite",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point classification.",
    ),
    OpCoverage(
        "tir.isinf",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point classification.",
    ),
    OpCoverage(
        "tir.isnan",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point classification.",
    ),
    OpCoverage(
        "tir.ldexp",
        OpFamily.TIR_OP,
        CoverageState.DEFERRED,
        "Needs either scalar.ldexpf or a reviewed target-neutral lowering.",
    ),
    OpCoverage(
        "tir.likely",
        OpFamily.TIR_OP,
        CoverageState.NORMALIZED,
        "Branch prediction hint; imports the wrapped condition/value directly.",
    ),
    OpCoverage(
        "tir.log",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.log10",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.log1p",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.log2",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.nearbyint",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Nearest-even scalar floating-point rounding.",
    ),
    OpCoverage(
        "tir.nextafter",
        OpFamily.TIR_OP,
        CoverageState.DEFERRED,
        "Needs scalar.nextafterf or a reviewed lowering sequence.",
    ),
    OpCoverage(
        "tir.popcount",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index population count.",
    ),
    OpCoverage(
        "tir.pow",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point power.",
    ),
    OpCoverage(
        "tir.power",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point power.",
    ),
    OpCoverage(
        "tir.reinterpret",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/vector bit reinterpretation; maps to Loom bitcast ops.",
    ),
    OpCoverage(
        "tir.round",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point rounding.",
    ),
    OpCoverage(
        "tir.rsqrt",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point reciprocal square root.",
    ),
    OpCoverage(
        "tir.shift_left",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index left shift.",
    ),
    OpCoverage(
        "tir.shift_right",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar/index right shift.",
    ),
    OpCoverage(
        "tir.sigmoid",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Imports to Loom logistic activation.",
    ),
    OpCoverage(
        "tir.sin",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.sinh",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.sqrt",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.tan",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.tanh",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point math intrinsic.",
    ),
    OpCoverage(
        "tir.trunc",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar floating-point rounding toward zero.",
    ),
    OpCoverage(
        "tir.truncdiv",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar integer division with truncation semantics.",
    ),
    OpCoverage(
        "tir.truncmod",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Scalar integer remainder with truncation semantics.",
    ),
    OpCoverage(
        "tir.assume",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        (
            "Effect assumption call; simple integer predicates map to Loom "
            "assume after materializing pure operand expressions."
        ),
    ),
    OpCoverage(
        "tir.tvm_storage_sync",
        OpFamily.TIR_OP,
        CoverageState.SUPPORTED,
        "Shared-memory workgroup barrier; shared/shared.dyn map to kernel.barrier.",
    ),
    OpCoverage(
        "tir.thread_return",
        OpFamily.TIR_OP,
        CoverageState.NORMALIZED,
        "Per-thread kernel exit; top-level prefix guards map to kernel.exit.",
    ),
    OpCoverage(
        "tl.sync_grid",
        OpFamily.TILELANG_OP,
        CoverageState.REJECTED,
        "Cooperative-grid synchronization is recognized and rejected with a "
        "grid-contract diagnostic until Loom has a grid sync operation and "
        "launch residency contract.",
    ),
    OpCoverage(
        "tl.device_assert",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Runtime kernel assertion imports to kernel.assert, not an assume.",
    ),
    OpCoverage(
        "tl.device_assert_with_msg",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Runtime kernel assertion with message imports to kernel.assert.",
    ),
    OpCoverage(
        "tl.sync_warp",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Full-mask warp synchronization; maps to subgroup-scoped kernel.barrier.",
    ),
    OpCoverage(
        "tl.shfl_xor_sync",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Full-mask warp xor shuffle; maps to kernel.subgroup.shuffle<xor>.",
    ),
    OpCoverage(
        "tl.shfl_down_sync",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Full-mask warp down shuffle; maps to kernel.subgroup.shuffle<down>.",
    ),
    OpCoverage(
        "tl.shfl_up_sync",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Full-mask warp up shuffle; maps to kernel.subgroup.shuffle<up>.",
    ),
    OpCoverage(
        "tl.shfl_sync",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Full-mask warp index shuffle; maps to kernel.subgroup.shuffle<index>.",
    ),
    OpCoverage(
        "tl.warp_reduce_sum",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Warp scalar sum reduction; maps to kernel.subgroup.reduce<add*>.",
    ),
    OpCoverage(
        "tl.warp_reduce_max",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Warp scalar max reduction; maps to kernel.subgroup.reduce<max*>.",
    ),
    OpCoverage(
        "tl.warp_reduce_min",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Warp scalar min reduction; maps to kernel.subgroup.reduce<min*>.",
    ),
    OpCoverage(
        "tl.warp_reduce_bitand",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Warp scalar bitwise-and reduction; maps to kernel.subgroup.reduce<andi>.",
    ),
    OpCoverage(
        "tl.warp_reduce_bitor",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Warp scalar bitwise-or reduction; maps to kernel.subgroup.reduce<ori>.",
    ),
    OpCoverage(
        "tl.match_any_sync",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Full-mask warp match-any; maps to kernel.subgroup.match.any.",
    ),
    OpCoverage(
        "tl.access_ptr",
        OpFamily.TILELANG_OP,
        CoverageState.NORMALIZED,
        "Pointer metadata helper; atomic import recovers the referenced view "
        "and indices.",
    ),
    OpCoverage(
        "tl.atomic_add_elem_op",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Scalar element atomic add; maps to view.atomic.reduce<add*>.",
    ),
    OpCoverage(
        "tl.atomic_add_ret_elem_op",
        OpFamily.TILELANG_OP,
        CoverageState.SUPPORTED,
        "Scalar element atomic add returning the old value; maps to "
        "view.atomic.rmw<add*>.",
    ),
    OpCoverage(
        "tl.atomic_addx2_elem_op",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed/vector atomic add requires vector atomic access import.",
    ),
    OpCoverage(
        "tl.atomic_addx4_elem_op",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed/vector atomic add requires vector atomic access import.",
    ),
    OpCoverage(
        "tl.__ldg",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Read-only cache load should become a target-independent access encoding fact.",
    ),
    OpCoverage(
        "tl.__log",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.__log2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.__log10",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.__tan",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.__cos",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.__sin",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.__exp10",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.__exp",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fast-math intrinsic needs an explicit Loom fastmath import policy.",
    ),
    OpCoverage(
        "tl.pow_of_int",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Needs a reviewed integer-exponent lowering preserving TileLang semantics.",
    ),
    OpCoverage(
        "tl.add2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed x2 arithmetic requires a vector/packed-scalar representation choice.",
    ),
    OpCoverage(
        "tl.sub2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed x2 arithmetic requires a vector/packed-scalar representation choice.",
    ),
    OpCoverage(
        "tl.mul2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed x2 arithmetic requires a vector/packed-scalar representation choice.",
    ),
    OpCoverage(
        "tl.fma2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed x2 arithmetic requires a vector/packed-scalar representation choice.",
    ),
    OpCoverage(
        "tl.max2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed x2 arithmetic requires a vector/packed-scalar representation choice.",
    ),
    OpCoverage(
        "tl.min2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed x2 arithmetic requires a vector/packed-scalar representation choice.",
    ),
    OpCoverage(
        "tl.abs2",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Packed x2 arithmetic requires a vector/packed-scalar representation choice.",
    ),
    OpCoverage(
        "tir.call_extern",
        OpFamily.TIR_OP,
        CoverageState.OPAQUE_FOREIGN,
        "Opaque external call; requires an explicit foreign-op policy.",
    ),
    OpCoverage(
        "tir.type_annotation",
        OpFamily.TIR_OP,
        CoverageState.NORMALIZED,
        "Pointer type metadata helper; access-pointer import validates and drops it.",
    ),
    OpCoverage(
        "tir.tvm_tuple",
        OpFamily.TIR_OP,
        CoverageState.IGNORED_METADATA,
        "Attribute value tuple; known metadata attributes decode the wrapper shape.",
    ),
    OpCoverage(
        "tir.call_packed",
        OpFamily.TIR_OP,
        CoverageState.REJECTED,
        "Host/runtime call is not a kernel-body operation.",
    ),
    OpCoverage(
        "tir.tvm_thread_allreduce",
        OpFamily.TIR_OP,
        CoverageState.DEFERRED,
        "Needs a Loom subgroup/workgroup reduction representation.",
    ),
    OpCoverage(
        "tir.tvm_access_ptr",
        OpFamily.TIR_OP,
        CoverageState.NORMALIZED,
        "TVM pointer view helper; recover buffer/view facts instead.",
    ),
    OpCoverage(
        "thread_extent",
        OpFamily.ATTRIBUTE,
        CoverageState.NORMALIZED,
        "Launch topology fact; imports to kernel workgroup/subgroup metadata.",
    ),
    OpCoverage(
        "tl.assume",
        OpFamily.ATTRIBUTE,
        CoverageState.SUPPORTED,
        "TileLang scoped integer predicate; maps to index.assume/scalar.assume.",
    ),
    OpCoverage(
        "tilelang_assume",
        OpFamily.ATTRIBUTE,
        CoverageState.SUPPORTED,
        "TileLang scoped integer predicate alias; maps to Loom assume.",
    ),
    OpCoverage(
        "tir.kernel_launch_params",
        OpFamily.ATTRIBUTE,
        CoverageState.IGNORED_METADATA,
        "Late launch ABI metadata; Loom derives launch metadata structurally.",
    ),
    OpCoverage(
        "global_symbol",
        OpFamily.ATTRIBUTE,
        CoverageState.IGNORED_METADATA,
        "Debug/import naming hint only; HAL symbol names are not semantic.",
    ),
    OpCoverage(
        "calling_conv",
        OpFamily.ATTRIBUTE,
        CoverageState.IGNORED_METADATA,
        "TVM lowering artifact; Loom kernels use HAL ABI rules.",
    ),
    OpCoverage(
        "tl.readonly_param_indices",
        OpFamily.ATTRIBUTE,
        CoverageState.NORMALIZED,
        "Access fact; should become a target-independent buffer access encoding.",
    ),
    OpCoverage(
        "tl.non_restrict_params",
        OpFamily.ATTRIBUTE,
        CoverageState.NORMALIZED,
        "Aliasing fact; should become a target-independent buffer access encoding.",
    ),
    OpCoverage(
        "dyn_shared_memory_buf",
        OpFamily.ATTRIBUTE,
        CoverageState.TARGET_ONLY,
        "Late dynamic shared-memory ABI hook; source allocation facts are preferred.",
    ),
    OpCoverage(
        "tl.copy",
        OpFamily.TILELANG_OP,
        CoverageState.NORMALIZED,
        "High-level authoring copy normalizes to tl.tileop.copy in structured TIR.",
    ),
    OpCoverage(
        "tl.gemm",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Matrix multiply tile op; maps through vector.mma and fragment encodings.",
    ),
    OpCoverage(
        "tl.alloc_shared",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Shared-memory allocation; depends on storage-space and layout encodings.",
    ),
    OpCoverage(
        "tl.alloc_fragment",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fragment allocation; depends on vector.fragment encoding import.",
    ),
    OpCoverage(
        "tl.tileop.region",
        OpFamily.TILELANG_TILEOP,
        CoverageState.SUPPORTED,
        "Tile region descriptor; decoded from buffer load, access mask, and extents.",
    ),
    OpCoverage(
        "tl.tileop.copy",
        OpFamily.TILELANG_TILEOP,
        CoverageState.SUPPORTED,
        (
            "Region-to-region copy; imports as structured scf.for plus "
            "view.load/store. Singleton dimensions normalize across source "
            "and destination regions, and scalar integer widening/truncation "
            "is explicit. disable_tma normalizes to this copy path; other "
            "scheduling annotations remain deferred."
        ),
    ),
    OpCoverage(
        "tl.tileop.fill",
        OpFamily.TILELANG_TILEOP,
        CoverageState.SUPPORTED,
        "Region fill; imports as structured scf.for plus view.store.",
    ),
    OpCoverage(
        "tl.tileop.reduce",
        OpFamily.TILELANG_TILEOP,
        CoverageState.SUPPORTED,
        (
            "Vectorized single-axis tile reductions import through vector.reduce, "
            "including absolute-value reductions; reducer-state forms remain "
            "separate."
        ),
    ),
    OpCoverage(
        "tl.tileop.finalize_reducer",
        OpFamily.TILELANG_TILEOP,
        CoverageState.SUPPORTED,
        (
            "replication=none finalizers normalize to no-op; replication=all "
            "maps scalar sum/min/max reducers through kernel.workgroup.reduce."
        ),
    ),
    OpCoverage(
        "tl.tileop.cumsum",
        OpFamily.TILELANG_TILEOP,
        CoverageState.SUPPORTED,
        (
            "Rank-1 shared-memory cumulative sum maps to kernel.workgroup.scan; "
            "higher-rank, non-shared, and non-workgroup-sized regions remain "
            "diagnostic cases."
        ),
    ),
    OpCoverage(
        "tl.tileop.gemm",
        OpFamily.TILELANG_TILEOP,
        CoverageState.SUPPORTED,
        (
            "Dense 16x16x16 f16/f32 GEMM maps through vector.fragment and "
            "vector.mma; transpose, packed, and asynchronous variants remain "
            "separate bridge cases."
        ),
    ),
)


def coverage_by_name() -> Mapping[str, OpCoverage]:
    coverage: dict[str, OpCoverage] = {}
    for row in TILELANG_OP_COVERAGE:
        if row.name in coverage:
            raise ValueError(f"duplicate TileLang coverage row {row.name!r}")
        coverage[row.name] = row
    return coverage


def coverage_row(op_name: str) -> OpCoverage | None:
    """Returns the exact or wildcard coverage row for a TileLang/TIR op name."""

    rows = coverage_by_name()
    exact = rows.get(op_name)
    if exact is not None:
        return exact
    for row in rows.values():
        if row.name.endswith(".*") and op_name.startswith(row.name[:-1]):
            return row
    return None


def audit_coverage(names: Iterable[str]) -> CoverageAudit:
    known_rows = coverage_by_name()
    known: list[str] = []
    missing: list[str] = []
    for name in sorted(set(names)):
        if name in known_rows or _matches_wildcard(name, known_rows):
            known.append(name)
        else:
            missing.append(name)
    return CoverageAudit(
        known=tuple(known),
        missing=tuple(missing),
    )


def _matches_wildcard(name: str, rows: Mapping[str, OpCoverage]) -> bool:
    for row_name in rows:
        if not row_name.endswith(".*"):
            continue
        if name.startswith(row_name[:-1]):
            return True
    return False
