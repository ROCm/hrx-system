# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest
from build_tools.bazel import configure as bazel_configure


def generate(arguments: list[str]) -> str:
    return bazel_configure.generate_config(bazel_configure.parse_arguments(arguments))


def test_media_options_have_equivalent_portable_and_native_spellings() -> None:
    portable = generate(
        [
            "-DLOOM_SERVE_IMAGE_INPUT=OFF",
            "-DLOOM_SERVE_IMAGE_CODECS=png;jpeg",
            "-DLOOM_SERVE_AUDIO_CODECS=wav;flac",
        ]
    )
    native = generate(
        [
            "--//loom/config/serve:image_input=false",
            "--//loom/config/serve:image_codecs=png,jpeg",
            "--//loom/config/serve:audio_codecs=wav,flac",
        ]
    )

    assert portable == native
    assert "build --//loom/config/serve:image_input=false\n" in portable
    assert "build --//loom/config/serve:image_codecs=png,jpeg\n" in portable
    assert "build --//loom/config/serve:audio_codecs=wav,flac\n" in portable


@pytest.mark.parametrize(
    "arguments",
    [
        ["-DLOOM_SERVE_IMAGE_CODECS=png;typo"],
        ["--//loom/config/serve:audio_codecs=typo"],
        [
            "-DLOOM_SERVE_IMAGE_INPUT=ON",
            "--//loom/config/serve:image_input=false",
        ],
        [
            "-DLOOM_SERVE_AUDIO_CODECS=wav",
            "--//loom/config/serve:audio_codecs=mp3",
        ],
    ],
)
def test_media_options_reject_unknown_values_and_mixed_spellings(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        generate(arguments)


@pytest.mark.parametrize(
    ("arguments", "architectures", "artifacts", "execution", "selection"),
    [
        (
            [],
            "aie2p,amdgpu,spirv,vm,x86",
            "elf,spirv,vm",
            "iree_hal,iree_vm",
            "default",
        ),
        (
            ["-DLOOM_TARGET_ALL_CPU=ON"],
            "vm,x86",
            "elf,vm",
            "iree_hal,iree_vm",
            "cpu",
        ),
        (
            ["-DLOOM_TARGET_ALL_CPU=ON", "-DLOOM_TARGET_ALL_AMD_NPU=ON"],
            "aie2p,vm,x86",
            "elf,vm",
            "iree_hal,iree_vm",
            "cpu,amd_npu",
        ),
    ],
)
def test_presets_replace_the_default_and_compose_by_union(
    arguments: list[str],
    architectures: str,
    artifacts: str,
    execution: str,
    selection: str,
) -> None:
    config = generate(arguments)
    assert "build --//loom/config/arch:enable=" + architectures + "\n" in config
    assert "build --//loom/config/emit:enable=" + artifacts + "\n" in config
    assert "build --//loom/config/execute:enable=" + execution + "\n" in config
    assert "# Loom target selection: " + selection + "\n" in config


def test_explicit_leaves_replace_a_disabled_default() -> None:
    config = generate(
        [
            "-DLOOM_TARGET_DEFAULT=OFF",
            "-DLOOM_ARCH_X86=ON",
            "-DLOOM_EMIT_ELF=ON",
            "-DLOOM_EXECUTE_IREE_HAL=ON",
        ]
    )

    assert "build --//loom/config/arch:enable=x86\n" in config
    assert "build --//loom/config/emit:enable=elf\n" in config
    assert "build --//loom/config/execute:enable=iree_hal\n" in config
    assert "# Loom target selection: explicit leaves\n" in config


def test_explicit_leaves_extend_but_do_not_subtract_from_a_preset() -> None:
    config = generate(
        [
            "-DLOOM_ARCH_WASM=ON",
            "-DLOOM_EMIT_WASM=ON",
            "-DLOOM_EXECUTE_WASM_HOST=ON",
            "-DLOOM_ARCH_AMDGPU=OFF",
            "-DLOOM_EMIT_SPIRV=OFF",
            "-DLOOM_EXECUTE_IREE_HAL=OFF",
        ]
    )

    assert (
        "build --//loom/config/arch:enable=aie2p,amdgpu,spirv,vm,wasm,x86\n" in config
    )
    assert "build --//loom/config/emit:enable=elf,spirv,vm,wasm\n" in config
    assert "build --//loom/config/execute:enable=iree_hal,iree_vm,wasm_host\n" in config


def test_native_capability_selection_is_exact_and_complete() -> None:
    config = generate(
        [
            "--//loom/config/arch:enable=vm,x86",
            "--//loom/config/emit:enable=elf,vm",
            "--//loom/config/execute:enable=iree_hal,iree_vm",
        ]
    )
    assert "build --//loom/config/arch:enable=vm,x86\n" in config
    assert "# Loom target selection: exact native lists\n" in config

    with pytest.raises(SystemExit, match="must name complete"):
        generate(["--//loom/config/arch:enable=x86"])
    with pytest.raises(SystemExit, match="Do not mix Loom target presets"):
        generate(
            [
                "-DLOOM_TARGET_ALL_CPU=ON",
                "--//loom/config/arch:enable=x86",
                "--//loom/config/emit:enable=elf",
                "--//loom/config/execute:enable=iree_hal",
            ]
        )


def test_importer_options_and_amdgpu_targets_reach_native_settings() -> None:
    config = generate(
        [
            "-DLOOM_IMPORT_TILELANG=ON",
            "--//loom/config/arch/amdgpu:targets=gfx942,gfx125X-all",
        ]
    )
    assert "build --//loom/config/import:enable=tilelang\n" in config
    assert "build --//loom/config/arch/amdgpu:targets=gfx125X-all,gfx942\n" in config


@pytest.mark.parametrize(
    "arguments",
    [
        ["-DLOOM_ARCH_AMDGPU=ON", "--//loom/config/arch:enable=spirv"],
        ["-DLOOM_IMPORT_TILELANG=ON", "--//loom/config/import:enable=mlir"],
        [
            "-DLOOM_EXECUTE_IREE_HAL=ON",
            "--//loom/config/execute:enable=iree_hal",
        ],
    ],
)
def test_portable_and_native_selection_spellings_conflict(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit, match="Do not mix portable"):
        generate(arguments)
