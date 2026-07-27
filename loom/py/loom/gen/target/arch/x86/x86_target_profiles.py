# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: x86 profile catalogs -> native and LLVMIR projection rows."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import c_string_literal as _c_string_literal  # noqa: E402
from loom.gen.support.files import write_text_file as _write_text  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.x86.packed_dot_data import (  # noqa: E402
    FEATURE_AVX10_2,
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_VL,
    FEATURE_AVX512_VNNI,
    FEATURE_AVX_VNNI,
    FEATURE_AVX_VNNI_INT8,
    FEATURE_AVX_VNNI_INT16,
)
from loom.target.arch.x86.target_info import (  # noqa: E402
    X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS,
    X86TargetProfileInfo,
    x86_target_profile_info_by_key,
)
from loom.target.emit.llvmir.x86.target_info import (  # noqa: E402
    X86_LLVMIR_PROJECTION_INFOS,
    X86LlvmirProjectionInfo,
    x86_llvmir_target_feature_string,
)

_FEATURE_BIT_C_NAMES = (
    (FEATURE_AVX512_VNNI, "LOOM_X86_FEATURE_AVX512_VNNI"),
    (FEATURE_AVX512_VL, "LOOM_X86_FEATURE_AVX512_VL"),
    (FEATURE_AVX_VNNI, "LOOM_X86_FEATURE_AVX_VNNI"),
    (FEATURE_AVX_VNNI_INT8, "LOOM_X86_FEATURE_AVX_VNNI_INT8"),
    (FEATURE_AVX_VNNI_INT16, "LOOM_X86_FEATURE_AVX_VNNI_INT16"),
    (FEATURE_AVX10_2, "LOOM_X86_FEATURE_AVX10_2"),
    (FEATURE_AVX512_BF16, "LOOM_X86_FEATURE_AVX512_BF16"),
)


def _c_arg(value: str) -> str:
    return f'"{_c_string_literal(value)}"'


def _symbol_suffix(profile_key: str) -> str:
    stem = profile_key.removeprefix("x86.")
    return "".join(part.capitalize() for part in re.split(r"[^0-9A-Za-z]+", stem))


def _feature_bits_expr(feature_bits: int) -> str:
    remaining_bits = feature_bits
    parts: list[str] = []
    for bit, c_name in _FEATURE_BIT_C_NAMES:
        if remaining_bits & bit:
            parts.append(c_name)
            remaining_bits &= ~bit
    if remaining_bits:
        raise ValueError(f"unknown x86 feature bits 0x{remaining_bits:x}")
    return " | ".join(parts) if parts else "0"


def _snapshot_name(native_bundle_key: str) -> str:
    return f"x86_64-{native_bundle_key.removeprefix('x86-')}-low"


def _native_target_profiles() -> tuple[X86TargetProfileInfo, ...]:
    profiles: list[X86TargetProfileInfo] = []
    for profile_key in X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS:
        if profile_key is None:
            continue
        profile = x86_target_profile_info_by_key(profile_key)
        if profile.native_bundle_key is None:
            raise ValueError(f"x86 profile {profile_key} has no native bundle key")
        profiles.append(profile)
    return tuple(profiles)


def _emit_native_profiles_inl(profiles: tuple[X86TargetProfileInfo, ...]) -> str:
    lines = [
        *line_comment_header("//", generator="loom.gen.target.arch.x86.x86_target_profiles"),
        "// clang-format off",
        "",
        "#ifdef LOOM_X86_NATIVE_TARGET_PROFILE",
    ]
    for profile in profiles:
        native_bundle_key = profile.native_bundle_key
        if native_bundle_key is None:
            raise ValueError(f"x86 profile {profile.profile_key} has no native bundle")
        lines.append(
            "LOOM_X86_NATIVE_TARGET_PROFILE("
            f"{_symbol_suffix(profile.profile_key)}, "
            f"{_c_arg(native_bundle_key)}, "
            f"{_c_arg(_snapshot_name(native_bundle_key))}, "
            f"{_c_arg(profile.descriptor_set_key)}, "
            f"{_feature_bits_expr(profile.contract_feature_bits)})"
        )
    lines.extend(["#endif", "", "// clang-format on", ""])
    return "\n".join(lines)


def _emit_llvmir_profiles_inl(
    projections: tuple[X86LlvmirProjectionInfo, ...],
) -> str:
    lines = [
        *line_comment_header("//", generator="loom.gen.target.arch.x86.x86_target_profiles"),
        "// clang-format off",
        "",
        "#ifdef LOOM_LLVMIR_X86_TARGET_PROFILE",
    ]
    for projection in projections:
        profile = projection.target_profile
        lines.append(
            "LOOM_LLVMIR_X86_TARGET_PROFILE("
            f"{_symbol_suffix(profile.profile_key)}, "
            f"{_c_arg(profile.descriptor_set_key)}, "
            f"{_c_arg(projection.debug_profile_key)}, "
            f"{_c_arg(x86_llvmir_target_feature_string(projection))}, "
            f"{_feature_bits_expr(profile.contract_feature_bits)})"
        )
    lines.extend(["#endif", "", "#ifdef LOOM_LLVMIR_X86_TARGET_FIXTURE"])
    for profile_key in ("x86.scalar", "x86.packed_dot"):
        projection = next(info for info in projections if info.profile_key == profile_key)
        profile = projection.target_profile
        lines.append(f"LOOM_LLVMIR_X86_TARGET_FIXTURE({_symbol_suffix(profile.profile_key)}, {_c_arg(profile.descriptor_set_key)}, {_feature_bits_expr(profile.contract_feature_bits)})")
    lines.append("#endif")
    lines.extend(["", "// clang-format on", ""])
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates x86 target profile include snippets.")
    parser.add_argument(
        "--native-output",
        type=Path,
        help="Path to write the native x86 profile include snippet.",
    )
    parser.add_argument(
        "--llvmir-output",
        type=Path,
        help="Path to write the LLVMIR x86 projection include snippet.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generation inputs without writing any output files.",
    )
    args = parser.parse_args(argv)
    if args.check and (args.native_output is not None or args.llvmir_output is not None):
        parser.error("--check cannot be combined with output flags")
    if not args.check and args.native_output is None and args.llvmir_output is None:
        parser.error("at least one output flag is required unless --check is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    native_contents = _emit_native_profiles_inl(_native_target_profiles())
    llvmir_contents = _emit_llvmir_profiles_inl(X86_LLVMIR_PROJECTION_INFOS)

    if args.check:
        return 0
    if args.native_output is not None:
        _write_text(args.native_output, native_contents)
    if args.llvmir_output is not None:
        _write_text(args.llvmir_output, llvmir_contents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
