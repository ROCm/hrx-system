# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 processor scheduling models over canonical Low descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from loom.target.arch.x86.target_info import X86_TARGET_TUNE_INFOS
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    IssueUse,
    IssueUseKind,
    LatencyKind,
    ModelQuality,
    OperandRole,
    Resource,
    ScheduleClass,
    ScheduleModel,
)


@dataclass(frozen=True, slots=True)
class X86ScheduleBehaviorGroup:
    """One processor behavior shared by several native instruction forms.

    Micro-operation count remains in the checked-in source data even though the
    runtime model currently applies only latency and reciprocal throughput.
    Physical emission may therefore share rows whose applied behavior matches;
    the retained source fact is available when a complete front-end resource
    model can consume it.
    """

    # Result dependency latency in cycles.
    latency_cycles: int
    # Reciprocal throughput counted in half-cycle units.
    reciprocal_throughput_half_cycles: int
    # Front-end micro-operations reported by the source machine model.
    micro_operation_count: int
    # Exact native descriptor identities with this behavior.
    descriptor_keys: tuple[str, ...]


# Offline translation of LLVM 23.1.1's checked x86 scheduling models queried with
# llvm-mca --instruction-tables=full. Generation consumes only this checked-in
# data and never runs LLVM or a Loom compiler. LLVM currently gives znver5 and
# znver6 identical results for the complete register-only floating family.
X86_ZNVER5_FLOAT_SCHEDULE_GROUPS: tuple[X86ScheduleBehaviorGroup, ...] = (
    X86ScheduleBehaviorGroup(
        1,
        1,
        1,
        (
            "x86.avx2.vblendvpd.xmm",
            "x86.avx2.vblendvps.xmm",
            "x86.avx2.vblendvps.ymm",
            "x86.avx2.vbroadcastss.xmm",
            "x86.avx2.vinsertps.xmm",
            "x86.avx2.vmovddup.xmm",
            "x86.avx2.vpermilps.xmm",
            "x86.avx2.vshufpd.xmm",
            "x86.avx512.vcvttsd2usi.gpr32.xmm",
            "x86.avx512.vcvttsd2usi.gpr64.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        2,
        1,
        1,
        (
            "x86.avx2.vcmppd.ymm",
            "x86.avx2.vcmpps.ymm",
            "x86.avx2.vmaxpd.xmm",
            "x86.avx2.vmaxpd.ymm",
            "x86.avx2.vmaxps.xmm",
            "x86.avx2.vmaxps.ymm",
            "x86.avx2.vminpd.xmm",
            "x86.avx2.vminpd.ymm",
            "x86.avx2.vminps.xmm",
            "x86.avx2.vminps.ymm",
            "x86.avx512.vcmppd.xmm",
            "x86.avx512.vcmppd.ymm",
            "x86.avx512.vcmpps.xmm",
            "x86.avx512.vcmpps.ymm",
            "x86.avx512_fp16.vcmpph.xmm",
            "x86.avx512_fp16.vcmpph.ymm",
            "x86.avx512_fp16.vmaxph.xmm",
            "x86.avx512_fp16.vmaxph.ymm",
            "x86.avx512_fp16.vminph.xmm",
            "x86.avx512_fp16.vminph.ymm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        2,
        2,
        1,
        (
            "x86.avx2.vbroadcastsd.ymm",
            "x86.avx2.vbroadcastss.ymm",
            "x86.avx2.vcmppd.xmm",
            "x86.avx2.vcmpps.xmm",
            "x86.avx2.vmaxsd.xmm",
            "x86.avx2.vmaxss.xmm",
            "x86.avx2.vminsd.xmm",
            "x86.avx2.vminss.xmm",
            "x86.avx512.vbroadcastsd.zmm",
            "x86.avx512.vbroadcastss.zmm",
            "x86.avx512.vcmppd.zmm",
            "x86.avx512.vcmpps.zmm",
            "x86.avx512.vextractf32x4.xmm.zmm",
            "x86.avx512.vmaxpd.zmm",
            "x86.avx512.vmaxps.zmm",
            "x86.avx512.vminpd.zmm",
            "x86.avx512.vminps.zmm",
            "x86.avx512_fp16.vcmpph.zmm",
            "x86.avx512_fp16.vcmpsh.xmm",
            "x86.avx512_fp16.vmaxph.zmm",
            "x86.avx512_fp16.vmaxsh.xmm",
            "x86.avx512_fp16.vminph.zmm",
            "x86.avx512_fp16.vminsh.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        3,
        1,
        1,
        (
            "x86.avx2.vaddpd.xmm",
            "x86.avx2.vaddpd.ymm",
            "x86.avx2.vaddps.xmm",
            "x86.avx2.vaddps.ymm",
            "x86.avx2.vaddsd.xmm",
            "x86.avx2.vaddss.xmm",
            "x86.avx2.vcvtdq2pd.xmm.xmm",
            "x86.avx2.vcvtdq2ps.xmm.xmm",
            "x86.avx2.vcvtdq2ps.ymm.ymm",
            "x86.avx2.vcvtpd2ps.xmm.xmm",
            "x86.avx2.vcvtps2pd.xmm.xmm",
            "x86.avx2.vcvtsd2ss.xmm",
            "x86.avx2.vcvtss2sd.xmm",
            "x86.avx2.vcvttps2dq.xmm.xmm",
            "x86.avx2.vmulpd.xmm",
            "x86.avx2.vmulpd.ymm",
            "x86.avx2.vmulps.xmm",
            "x86.avx2.vmulps.ymm",
            "x86.avx2.vmulsd.xmm",
            "x86.avx2.vmulss.xmm",
            "x86.avx2.vsubpd.xmm",
            "x86.avx2.vsubpd.ymm",
            "x86.avx2.vsubps.xmm",
            "x86.avx2.vsubps.ymm",
            "x86.avx2.vsubsd.xmm",
            "x86.avx2.vsubss.xmm",
            "x86.avx512.vcvtpd2ps.xmm.xmm",
            "x86.avx512.vcvtph2ps.xmm.xmm",
            "x86.avx512.vcvtps2pd.xmm.xmm",
            "x86.avx512.vcvtqq2pd.xmm.xmm",
            "x86.avx512.vcvtqq2ps.xmm.xmm",
            "x86.avx512.vcvtqq2ps.xmm.ymm",
            "x86.avx512.vcvtsd2ss.xmm",
            "x86.avx512.vcvtss2sd.xmm",
            "x86.avx512.vcvttps2qq.xmm.xmm",
            "x86.avx512.vcvttps2udq.xmm.xmm",
            "x86.avx512.vcvttps2uqq.xmm.xmm",
            "x86.avx512.vcvtudq2pd.xmm.xmm",
            "x86.avx512.vcvtudq2ps.xmm.xmm",
            "x86.avx512.vcvtudq2ps.ymm.ymm",
            "x86.avx512.vcvtuqq2pd.xmm.xmm",
            "x86.avx512.vcvtuqq2ps.xmm.xmm",
            "x86.avx512.vcvtuqq2ps.xmm.ymm",
            "x86.avx512_bf16.vcvtne2ps2bf16.merge.xmm",
            "x86.avx512_bf16.vcvtne2ps2bf16.xmm",
            "x86.avx512_bf16.vcvtne2ps2bf16.zero.xmm",
            "x86.avx512_bf16.vcvtneps2bf16.merge.xmm.xmm",
            "x86.avx512_bf16.vcvtneps2bf16.xmm.xmm",
            "x86.avx512_bf16.vcvtneps2bf16.zero.xmm.xmm",
            "x86.avx512_fp16.vaddph.xmm",
            "x86.avx512_fp16.vaddph.ymm",
            "x86.avx512_fp16.vaddsh.xmm",
            "x86.avx512_fp16.vcvtph2psx.xmm.xmm",
            "x86.avx512_fp16.vcvtps2phx.xmm.xmm",
            "x86.avx512_fp16.vcvtsh2ss.xmm",
            "x86.avx512_fp16.vcvtss2sh.xmm",
            "x86.avx512_fp16.vmulph.xmm",
            "x86.avx512_fp16.vmulph.ymm",
            "x86.avx512_fp16.vmulsh.xmm",
            "x86.avx512_fp16.vsubph.xmm",
            "x86.avx512_fp16.vsubph.ymm",
            "x86.avx512_fp16.vsubsh.xmm",
            "x86.avx_ne_convert.vcvtneps2bf16.xmm.xmm",
            "x86.f16c.vcvtph2ps.xmm.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        3,
        2,
        1,
        (
            "x86.avx2.vcvttpd2dq.xmm.xmm",
            "x86.avx512.vaddpd.zmm",
            "x86.avx512.vaddps.zmm",
            "x86.avx512.vcvtps2ph.xmm.xmm",
            "x86.avx512.vcvttpd2qq.xmm.xmm",
            "x86.avx512.vcvttpd2udq.xmm.xmm",
            "x86.avx512.vcvttpd2uqq.xmm.xmm",
            "x86.avx512.vmulpd.zmm",
            "x86.avx512.vmulps.zmm",
            "x86.avx512.vsubpd.zmm",
            "x86.avx512.vsubps.zmm",
            "x86.avx512_fp16.vaddph.zmm",
            "x86.avx512_fp16.vmulph.zmm",
            "x86.avx512_fp16.vsubph.zmm",
            "x86.f16c.vcvtps2ph.xmm.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        3,
        2,
        2,
        (
            "x86.avx2.vcvtdq2pd.ymm.xmm",
            "x86.avx2.vcvttpd2dq.xmm.ymm",
            "x86.avx512.vcvtdq2ps.zmm.zmm",
            "x86.avx512.vcvtqq2pd.ymm.ymm",
            "x86.avx512.vcvtqq2ps.ymm.zmm",
            "x86.avx512.vcvttpd2qq.ymm.ymm",
            "x86.avx512.vcvttpd2udq.xmm.ymm",
            "x86.avx512.vcvttpd2uqq.ymm.ymm",
            "x86.avx512.vcvtudq2pd.ymm.xmm",
            "x86.avx512.vcvtudq2ps.zmm.zmm",
            "x86.avx512.vcvtuqq2pd.ymm.ymm",
            "x86.avx512.vcvtuqq2ps.ymm.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        3,
        4,
        2,
        (
            "x86.avx512.vcvttpd2dq.ymm.zmm",
            "x86.avx512.vcvttpd2qq.zmm.zmm",
            "x86.avx512.vcvttpd2udq.ymm.zmm",
            "x86.avx512.vcvttpd2uqq.zmm.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        4,
        1,
        1,
        (
            "x86.avx10_2.vdpphps.xmm",
            "x86.avx10_2.vdpphps.ymm",
            "x86.avx2.vcvttps2dq.ymm.ymm",
            "x86.avx2.vfmadd231pd.xmm",
            "x86.avx2.vfmadd231pd.ymm",
            "x86.avx2.vfmadd231ps.xmm",
            "x86.avx2.vfmadd231ps.ymm",
            "x86.avx512.vcvttps2qq.ymm.xmm",
            "x86.avx512.vcvttps2udq.ymm.ymm",
            "x86.avx512.vcvttps2uqq.ymm.xmm",
            "x86.avx512_fp16.vfmadd231ph.xmm",
            "x86.avx512_fp16.vfmadd231ph.ymm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        4,
        2,
        1,
        (
            "x86.avx10_2.vdpphps.zmm",
            "x86.avx2.vextractf128.xmm.ymm",
            "x86.avx2.vfmadd231sd.xmm",
            "x86.avx2.vfmadd231ss.xmm",
            "x86.avx2.vrsqrtps.xmm",
            "x86.avx2.vrsqrtps.ymm",
            "x86.avx2.vrsqrtss.xmm",
            "x86.avx512.vfmadd231pd.zmm",
            "x86.avx512.vfmadd231ps.zmm",
            "x86.avx512.vrsqrt14sd.xmm",
            "x86.avx512.vrsqrt14ss.xmm",
            "x86.avx512_fp16.vfmadd231ph.zmm",
            "x86.avx512_fp16.vfmadd231sh.xmm",
            "x86.avx512_fp16.vrsqrtph.xmm",
            "x86.avx512_fp16.vrsqrtph.ymm",
            "x86.avx512_fp16.vrsqrtsh.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        4,
        2,
        2,
        (
            "x86.avx2.vcvtps2pd.ymm.xmm",
            "x86.avx2.vcvtsi2ss.xmm.gpr32",
            "x86.avx2.vcvtsi2ss.xmm.gpr64",
            "x86.avx512.vcvtph2ps.ymm.xmm",
            "x86.avx512.vcvtps2pd.ymm.xmm",
            "x86.avx512.vcvttps2dq.zmm.zmm",
            "x86.avx512.vcvttps2qq.zmm.ymm",
            "x86.avx512.vcvttps2udq.zmm.zmm",
            "x86.avx512.vcvttps2uqq.zmm.ymm",
            "x86.avx512.vcvtusi2ss.xmm.gpr32",
            "x86.avx512.vcvtusi2ss.xmm.gpr64",
            "x86.avx512_fp16.vcvtph2psx.ymm.xmm",
            "x86.f16c.vcvtph2ps.ymm.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        4,
        4,
        4,
        (
            "x86.avx512.vcvtdq2pd.zmm.ymm",
            "x86.avx512.vcvtph2ps.zmm.ymm",
            "x86.avx512.vcvtqq2pd.zmm.zmm",
            "x86.avx512.vcvtudq2pd.zmm.ymm",
            "x86.avx512.vcvtuqq2pd.zmm.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        5,
        2,
        1,
        (
            "x86.avx512.vrsqrt14pd.xmm",
            "x86.avx512.vrsqrt14pd.ymm",
            "x86.avx512.vrsqrt14pd.zmm",
            "x86.avx512.vrsqrt14ps.xmm",
            "x86.avx512.vrsqrt14ps.ymm",
            "x86.avx512.vrsqrt14ps.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        5,
        2,
        2,
        (
            "x86.avx2.vcvtsi2sd.xmm.gpr32",
            "x86.avx2.vcvtsi2sd.xmm.gpr64",
            "x86.avx512.vcvtusi2sd.xmm.gpr32",
            "x86.avx512.vcvtusi2sd.xmm.gpr64",
        ),
    ),
    X86ScheduleBehaviorGroup(5, 4, 1, ("x86.avx512_fp16.vrsqrtph.zmm",)),
    X86ScheduleBehaviorGroup(
        5,
        5,
        2,
        (
            "x86.avx2.vcvttsd2si.gpr32.xmm",
            "x86.avx2.vcvttsd2si.gpr64.xmm",
            "x86.avx2.vcvttss2si.gpr32.xmm",
            "x86.avx2.vcvttss2si.gpr64.xmm",
            "x86.avx512.vcvttss2usi.gpr32.xmm",
            "x86.avx512.vcvttss2usi.gpr64.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        6,
        2,
        2,
        (
            "x86.avx2.vcvtpd2ps.xmm.ymm",
            "x86.avx512.vcvtpd2ps.xmm.ymm",
            "x86.avx512.vcvtps2ph.xmm.ymm",
            "x86.avx512.vcvtps2ph.ymm.zmm",
            "x86.avx512_bf16.vcvtne2ps2bf16.merge.ymm",
            "x86.avx512_bf16.vcvtne2ps2bf16.ymm",
            "x86.avx512_bf16.vcvtne2ps2bf16.zero.ymm",
            "x86.avx512_bf16.vcvtneps2bf16.merge.xmm.ymm",
            "x86.avx512_bf16.vcvtneps2bf16.xmm.ymm",
            "x86.avx512_bf16.vcvtneps2bf16.zero.xmm.ymm",
            "x86.avx512_fp16.vcvtps2phx.xmm.ymm",
            "x86.avx_ne_convert.vcvtneps2bf16.xmm.ymm",
            "x86.f16c.vcvtps2ph.xmm.ymm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        6,
        4,
        4,
        (
            "x86.avx512.vcvtpd2ps.ymm.zmm",
            "x86.avx512.vcvtps2pd.zmm.ymm",
            "x86.avx512_bf16.vcvtne2ps2bf16.merge.zmm",
            "x86.avx512_bf16.vcvtne2ps2bf16.zero.zmm",
            "x86.avx512_bf16.vcvtne2ps2bf16.zmm",
            "x86.avx512_bf16.vcvtneps2bf16.merge.ymm.zmm",
            "x86.avx512_bf16.vcvtneps2bf16.ymm.zmm",
            "x86.avx512_bf16.vcvtneps2bf16.zero.ymm.zmm",
            "x86.avx512_fp16.vcvtph2psx.zmm.ymm",
            "x86.avx512_fp16.vcvtps2phx.ymm.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        6,
        6,
        2,
        (
            "x86.avx512_bf16.vdpbf16ps.xmm",
            "x86.avx512_bf16.vdpbf16ps.ymm",
            "x86.avx512_bf16.vdpbf16ps.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        11,
        6,
        1,
        (
            "x86.avx2.vdivps.xmm",
            "x86.avx2.vdivps.ymm",
            "x86.avx2.vdivss.xmm",
            "x86.avx512_fp16.vdivph.xmm",
            "x86.avx512_fp16.vdivph.ymm",
            "x86.avx512_fp16.vdivsh.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        11,
        12,
        1,
        (
            "x86.avx512.vdivps.zmm",
            "x86.avx512_fp16.vdivph.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        13,
        10,
        1,
        (
            "x86.avx2.vdivpd.xmm",
            "x86.avx2.vdivpd.ymm",
            "x86.avx2.vdivsd.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(13, 20, 1, ("x86.avx512.vdivpd.zmm",)),
    X86ScheduleBehaviorGroup(
        15,
        10,
        1,
        (
            "x86.avx2.vsqrtps.xmm",
            "x86.avx2.vsqrtps.ymm",
            "x86.avx2.vsqrtss.xmm",
            "x86.avx512_fp16.vsqrtph.xmm",
            "x86.avx512_fp16.vsqrtph.ymm",
            "x86.avx512_fp16.vsqrtsh.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        15,
        20,
        1,
        (
            "x86.avx512.vsqrtps.zmm",
            "x86.avx512_fp16.vsqrtph.zmm",
        ),
    ),
    X86ScheduleBehaviorGroup(
        21,
        18,
        1,
        (
            "x86.avx2.vsqrtpd.xmm",
            "x86.avx2.vsqrtpd.ymm",
            "x86.avx2.vsqrtsd.xmm",
        ),
    ),
    X86ScheduleBehaviorGroup(21, 36, 1, ("x86.avx512.vsqrtpd.zmm",)),
)

# The same offline query covers every scalar integer multiply without an
# attached memory access. Memory forms retain their load-sensitive classes.
X86_ZNVER5_INTEGER_MULTIPLY_SCHEDULE_GROUPS: tuple[X86ScheduleBehaviorGroup, ...] = (
    X86ScheduleBehaviorGroup(
        3,
        2,
        1,
        (
            "x86.scalar.imul.gpr32",
            "x86.scalar.imul.gpr64",
            "x86.scalar.imul.imm.gpr32",
            "x86.scalar.imul.imm.gpr64",
        ),
    ),
    X86ScheduleBehaviorGroup(3, 6, 2, ("x86.scalar.mul.high.gpr64",)),
)


def _validate_behavior_groups(
    groups: tuple[X86ScheduleBehaviorGroup, ...],
) -> Mapping[str, X86ScheduleBehaviorGroup]:
    behaviors: dict[str, X86ScheduleBehaviorGroup] = {}
    for group in groups:
        if group.latency_cycles <= 0:
            raise ValueError("x86 schedule-model latency must be positive")
        if group.reciprocal_throughput_half_cycles <= 0:
            raise ValueError("x86 schedule-model throughput must be positive")
        if group.micro_operation_count <= 0:
            raise ValueError(
                "x86 schedule-model micro-operation count must be positive"
            )
        if not group.descriptor_keys:
            raise ValueError("x86 schedule-model behavior group must not be empty")
        for descriptor_key in group.descriptor_keys:
            if descriptor_key in behaviors:
                raise ValueError(
                    f"x86 schedule-model descriptor '{descriptor_key}' is duplicated"
                )
            behaviors[descriptor_key] = group
    return MappingProxyType(behaviors)


X86_ZNVER5_FLOAT_SCHEDULE_BEHAVIORS = _validate_behavior_groups(
    X86_ZNVER5_FLOAT_SCHEDULE_GROUPS
)
X86_ZNVER5_SCHEDULE_BEHAVIORS = _validate_behavior_groups(
    X86_ZNVER5_FLOAT_SCHEDULE_GROUPS + X86_ZNVER5_INTEGER_MULTIPLY_SCHEDULE_GROUPS
)

_X86_SCHEDULE_BEHAVIORS_BY_MODEL_KEY = MappingProxyType(
    {
        "x86.znver5": X86_ZNVER5_SCHEDULE_BEHAVIORS,
        "x86.znver6": X86_ZNVER5_SCHEDULE_BEHAVIORS,
    }
)


def _is_floating_schedule_descriptor(descriptor: Descriptor) -> bool:
    semantic_tag = descriptor.semantic_tag
    if semantic_tag is None:
        return False
    return semantic_tag.startswith(
        (
            "float.",
            "integer.sitofp.",
            "integer.uitofp.",
            "dot.bf16bf16.",
            "dot.f16f16.",
        )
    )


def _is_register_only(descriptor: Descriptor) -> bool:
    return not descriptor.effects and all(
        operand.role is not OperandRole.RESOURCE for operand in descriptor.operands
    )


def x86_register_only_floating_descriptor_keys(
    descriptor_set: DescriptorSet,
) -> frozenset[str]:
    """Returns floating forms with fixed instruction-only schedule behavior.

    Memory-attached forms retain their existing variable schedule classes.
    Their completion includes cache-dependent load behavior that a fixed native
    instruction latency cannot represent.
    """

    return frozenset(
        descriptor.key
        for descriptor in descriptor_set.descriptors
        if _is_floating_schedule_descriptor(descriptor)
        and _is_register_only(descriptor)
    )


def x86_register_only_integer_multiply_descriptor_keys(
    descriptor_set: DescriptorSet,
) -> frozenset[str]:
    """Returns the complete register-only scalar integer multiply family."""

    return frozenset(
        descriptor.key
        for descriptor in descriptor_set.descriptors
        if descriptor.key.startswith("x86.scalar.")
        and descriptor.semantic_tag is not None
        and descriptor.semantic_tag.startswith("integer.mul.")
        and _is_register_only(descriptor)
    )


def _throughput_issue_use(
    descriptor_key: str,
    issue_use: IssueUse,
    resource: Resource,
    behavior: X86ScheduleBehaviorGroup,
) -> IssueUse:
    half_cycles = behavior.reciprocal_throughput_half_cycles
    if half_cycles % 2 == 0:
        cycles = half_cycles // 2
        units = resource.capacity_per_cycle
    else:
        if resource.capacity_per_cycle % 2 != 0:
            raise ValueError(
                f"x86 descriptor '{descriptor_key}' requires half-cycle throughput "
                f"on odd-capacity resource '{resource.name}'"
            )
        cycles = half_cycles
        units = resource.capacity_per_cycle // 2
    return replace(issue_use, cycles=cycles, units=units)


def _machine_schedule_class(
    schedule_class: ScheduleClass,
    resources_by_name: Mapping[str, Resource],
    behavior: X86ScheduleBehaviorGroup,
) -> ScheduleClass:
    if len(schedule_class.issue_uses) != 1:
        raise ValueError(
            f"x86 modeled schedule class '{schedule_class.name}' must have one "
            "issue use"
        )
    issue_use = schedule_class.issue_uses[0]
    if issue_use.kind is not IssueUseKind.REQUIRED or issue_use.stage != 0:
        raise ValueError(
            f"x86 modeled schedule class '{schedule_class.name}' must have one "
            "required stage-zero issue use"
        )
    resource = resources_by_name[issue_use.resource]
    return replace(
        schedule_class,
        latency_kind=LatencyKind.ESTIMATE,
        model_quality=ModelQuality.ESTIMATED,
        latency_cycles=behavior.latency_cycles,
        schedule_distance_cycles=0,
        minimum_issue_separation_cycles=behavior.latency_cycles,
        issue_uses=(
            _throughput_issue_use(
                schedule_class.name,
                issue_use,
                resource,
                behavior,
            ),
        ),
    )


def project_x86_machine_schedule_classes(
    descriptor_set: DescriptorSet,
) -> DescriptorSet:
    """Gives every modeled instruction form a stable canonical class."""

    if descriptor_set.schedule_models:
        raise ValueError(
            f"x86 descriptor set '{descriptor_set.key}' already has schedule models"
        )

    inline_classes_by_name = {
        schedule_class.name: schedule_class
        for schedule_class in descriptor_set.schedule_classes
    }
    inline_schedule_classes = list(descriptor_set.schedule_classes)
    projected_descriptors = []
    required_behavior_keys = x86_register_only_floating_descriptor_keys(
        descriptor_set
    ) | x86_register_only_integer_multiply_descriptor_keys(descriptor_set)
    for descriptor in descriptor_set.descriptors:
        behavior = X86_ZNVER5_SCHEDULE_BEHAVIORS.get(descriptor.key)
        if behavior is None:
            if descriptor.key in required_behavior_keys:
                raise ValueError(
                    f"x86 modeled descriptor '{descriptor.key}' has no processor "
                    "schedule data"
                )
            projected_descriptors.append(descriptor)
            continue
        source_class = inline_classes_by_name[descriptor.schedule_class]
        descriptor_class = replace(source_class, name=descriptor.key)
        inline_schedule_classes.append(descriptor_class)
        projected_descriptors.append(
            replace(descriptor, schedule_class=descriptor_class.name)
        )

    return replace(
        descriptor_set,
        descriptors=tuple(projected_descriptors),
        schedule_classes=tuple(inline_schedule_classes),
    )


def attach_x86_machine_schedule_models(
    descriptor_set: DescriptorSet,
) -> DescriptorSet:
    """Attaches complete processor models to a projected class namespace."""

    if descriptor_set.schedule_models:
        raise ValueError(
            f"x86 descriptor set '{descriptor_set.key}' already has schedule models"
        )
    tune_model_keys = tuple(tune.schedule_model_key for tune in X86_TARGET_TUNE_INFOS)
    if tuple(_X86_SCHEDULE_BEHAVIORS_BY_MODEL_KEY) != tune_model_keys:
        raise ValueError("x86 target tunes and schedule-model data disagree")

    resources_by_name = {
        resource.name: resource for resource in descriptor_set.resources
    }
    schedule_models = []
    for model_key, behaviors in _X86_SCHEDULE_BEHAVIORS_BY_MODEL_KEY.items():
        schedule_models.append(
            ScheduleModel(
                key=model_key,
                resources=descriptor_set.resources,
                schedule_classes=tuple(
                    _machine_schedule_class(
                        schedule_class,
                        resources_by_name,
                        behaviors[schedule_class.name],
                    )
                    if schedule_class.name in behaviors
                    else schedule_class
                    for schedule_class in descriptor_set.schedule_classes
                ),
                timing_events=descriptor_set.timing_events,
                event_separations=descriptor_set.event_separations,
            )
        )

    return replace(
        descriptor_set,
        schedule_models=tuple(schedule_models),
    )


def apply_x86_machine_schedule_models(descriptor_set: DescriptorSet) -> DescriptorSet:
    """Adds stable per-instruction classes and complete processor models."""

    return attach_x86_machine_schedule_models(
        project_x86_machine_schedule_classes(descriptor_set)
    )
