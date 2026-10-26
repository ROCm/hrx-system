# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: x86 profile catalogs -> native target rows."""

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
from loom.target.arch.x86.feature_catalog import X86_FEATURE_INFOS  # noqa: E402
from loom.target.arch.x86.target_info import (  # noqa: E402
    X86_DEPLOYMENT_PROFILE_INFOS,
    X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS,
    X86_TARGET_TUNE_INFOS,
    X86DeploymentProfileInfo,
    X86TargetProfileInfo,
    X86TargetTuneInfo,
    sorted_deployment_profile_infos,
    x86_deployment_profile_executable_target_profiles,
    x86_deployment_profile_preferred_target_profile,
    x86_target_profile_info_by_key,
)
from loom.target.low_descriptors import descriptor_stable_id, schedule_model_stable_id  # noqa: E402


def _c_arg(value: str) -> str:
    return f'"{_c_string_literal(value)}"'


def _symbol_suffix(profile_key: str) -> str:
    stem = profile_key.removeprefix("x86.")
    return "".join(part.capitalize() for part in re.split(r"[^0-9A-Za-z]+", stem))


def _enum_suffix(spelling: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", spelling).upper()


def _processor_vendor_words(vendor: str) -> tuple[int, int]:
    vendor_bytes = vendor.encode("ascii")
    if len(vendor_bytes) != 12:
        raise ValueError(f"x86 processor vendor {vendor!r} must be 12 ASCII bytes")
    return (
        int.from_bytes(vendor_bytes[:8], "little"),
        int.from_bytes(vendor_bytes[8:], "little"),
    )


def _feature_bits_expr(feature_bits: int) -> str:
    remaining_bits = feature_bits
    parts: list[str] = []
    for ordinal, info in enumerate(X86_FEATURE_INFOS):
        bit = 1 << ordinal
        if remaining_bits & bit:
            parts.append(info.c_feature_macro)
            remaining_bits &= ~bit
    if remaining_bits:
        raise ValueError(f"unknown x86 feature bits 0x{remaining_bits:x}")
    return " | ".join(parts) if parts else "0"


def _cpu_feature_bits_expr(features: tuple[str, ...]) -> str:
    return " | ".join(f"LOOM_X86_CPU_FIELD0_BIT({feature})" for feature in features) or "0"


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


def _emit_native_profiles_inl(
    profiles: tuple[X86TargetProfileInfo, ...],
    tunes: tuple[X86TargetTuneInfo, ...] = X86_TARGET_TUNE_INFOS,
    deployment_profiles: tuple[X86DeploymentProfileInfo, ...] = X86_DEPLOYMENT_PROFILE_INFOS,
) -> str:
    lines = [
        *line_comment_header("//", generator="loom.gen.target.arch.x86.x86_target_profiles"),
        "// clang-format off",
        "",
        "#ifdef LOOM_X86_TARGET_FEATURE",
    ]
    lines.extend(f"LOOM_X86_TARGET_FEATURE({info.c_enum_suffix}, {info.c_feature_macro})" for info in X86_FEATURE_INFOS)
    lines.extend(["#endif", "", "#ifdef LOOM_X86_CPU_FEATURE_MAPPING"])
    for info in X86_FEATURE_INFOS:
        cpu_feature_position = f"LOOM_X86_CPU_FIELD0_BIT_POSITION({info.runtime_cpu_suffix})" if info.runtime_cpu_suffix is not None else "UINT8_MAX"
        lines.append(f"LOOM_X86_CPU_FEATURE_MAPPING({cpu_feature_position})")
    lines.extend(["#endif", "", "#ifdef LOOM_X86_TARGET_TUNE"])
    lines.extend(f"LOOM_X86_TARGET_TUNE({_enum_suffix(tune.spelling)}, {tune.stable_value}, UINT64_C(0x{schedule_model_stable_id(tune.schedule_model_key):016x}))" for tune in tunes)
    profiles_by_key = {profile.profile_key: profile for profile in profiles}
    target_kinds_by_key = {profile.profile_key: target_kind for target_kind, profile in enumerate(profiles, start=1)}
    lines.extend(["#endif", "", "#ifdef LOOM_X86_DEPLOYMENT_PROFILE"])
    for ordinal, deployment_profile in enumerate(sorted(deployment_profiles, key=lambda info: info.spelling), start=1):
        executable_profiles = x86_deployment_profile_executable_target_profiles(deployment_profile, profiles)
        executable_target_kinds = sum(1 << target_kinds_by_key[profile.profile_key] for profile in executable_profiles)
        preferred_profile = x86_deployment_profile_preferred_target_profile(deployment_profile, profiles)
        preferred_profile = profiles_by_key[preferred_profile.profile_key]
        preferred_target_kind = target_kinds_by_key[preferred_profile.profile_key]
        lines.append(
            "LOOM_X86_DEPLOYMENT_PROFILE("
            f"{_enum_suffix(deployment_profile.spelling)}, "
            f"{ordinal}, "
            f"{len(deployment_profile.spelling.encode('ascii'))}, "
            f"{_c_arg(deployment_profile.spelling)}, "
            f"{_cpu_feature_bits_expr(deployment_profile.cpu_features)}, "
            f"UINT32_C(0x{executable_target_kinds:08x}), "
            f"{_symbol_suffix(preferred_profile.profile_key)}, "
            f"{preferred_target_kind}, "
            f"{_enum_suffix(deployment_profile.tune_spelling)})"
        )
    lines.extend(["#endif", "", "#ifdef LOOM_X86_PROCESSOR_DEPLOYMENT_MATCH"])
    for deployment_profile in deployment_profiles:
        for match in deployment_profile.processor_matches:
            vendor_low, vendor_high = _processor_vendor_words(match.vendor)
            lines.append(
                "LOOM_X86_PROCESSOR_DEPLOYMENT_MATCH("
                f"{_enum_suffix(deployment_profile.spelling)}, "
                f"UINT64_C(0x{vendor_low:016x}), "
                f"UINT32_C(0x{vendor_high:08x}), "
                f"UINT16_C(0x{match.family:03x}), "
                f"UINT8_C(0x{match.first_model:02x}), "
                f"UINT8_C(0x{match.last_model:02x}))"
            )
    lines.extend(["#endif", "", "#ifdef LOOM_X86_NATIVE_TARGET_PROFILE"])
    for target_kind, profile in enumerate(profiles, start=1):
        native_bundle_key = profile.native_bundle_key
        if native_bundle_key is None:
            raise ValueError(f"x86 profile {profile.profile_key} has no native bundle")
        feature_contract_set_key = profile.feature_contract_set_key
        if feature_contract_set_key is None:
            raise ValueError(f"x86 profile {profile.profile_key} has no feature contract")
        lines.append(
            "LOOM_X86_NATIVE_TARGET_PROFILE("
            f"{_symbol_suffix(profile.profile_key)}, "
            f"{target_kind}, "
            f"{_c_arg(profile.profile_key.removeprefix('x86.'))}, "
            f"{_c_arg(native_bundle_key)}, "
            f"{_c_arg(_snapshot_name(native_bundle_key))}, "
            f"{_c_arg(profile.descriptor_set_key)}, "
            f"UINT64_C(0x{descriptor_stable_id(profile.descriptor_set_key):016x}), "
            f"{_c_arg(feature_contract_set_key)}, "
            f"UINT64_C(0x{descriptor_stable_id(feature_contract_set_key):016x}), "
            f"{_feature_bits_expr(profile.contract_feature_bits)}, "
            f"{_feature_bits_expr(profile.supported_feature_bits)}, "
            f"{_cpu_feature_bits_expr(profile.required_cpu_features)}, "
            f"{profile.automatic_priority})"
        )
    lines.extend(["#endif", "", "// clang-format on", ""])
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates x86 target profile include snippets.")
    parser.add_argument(
        "--native-output",
        type=Path,
        help="Path to write the native x86 profile include snippet.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generation inputs without writing any output files.",
    )
    args = parser.parse_args(argv)
    if args.check and args.native_output is not None:
        parser.error("--check cannot be combined with --native-output")
    if not args.check and args.native_output is None:
        parser.error("--native-output is required unless --check is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    native_contents = _emit_native_profiles_inl(
        _native_target_profiles(),
        deployment_profiles=sorted_deployment_profile_infos(),
    )

    if args.check:
        return 0
    if args.native_output is not None:
        _write_text(args.native_output, native_contents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
