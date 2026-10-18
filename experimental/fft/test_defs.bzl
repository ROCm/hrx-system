# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small real-FFT checks with independently constructed direct-DFT fixtures."""

load("//loom/build_tools/bazel:build_defs.bzl", "loom_config_compatible_with")
load("//loom/build_tools/bazel:defs.bzl", "loom_execution_profile", "loom_test", "loom_test_module")
load("//loom/requirements:defs.bzl", "EXECUTE_IREE_HAL", "TARGET_ARCH_X86")
load("//loom/target/amdgpu:execution_profiles.bzl", "AMDGPU_HARDWARE_PROFILE")
load("//loom/target/vm:execution_profiles.bzl", "VM_REFERENCE_PROFILE")

_STORAGE = {"bf16": "std::bfloat16_t", "f16": "std::float16_t", "f32": "float"}
_NATIVE_PROFILE = loom_execution_profile(
    name = "x86_avx512",
    target_family = "x86",
    target_class = "cpu",
    executor = "hardware",
    build_requirements = [EXECUTE_IREE_HAL, TARGET_ARCH_X86],
    runner_args = ["--device=task", "--target=x86:avx512", "--task_topology_max_group_count=2"],
)

def workgroup_fft_test(name, length, frames = 3, input_type = "f32", output_type = "f32"):
    """Checks one cooperative schedule with independent DFT fixtures.

    Args:
        name: Fixture and test prefix.
        length: Exact real transform length.
        frames: Runtime frame count.
        input_type: Input storage spelling.
        output_type: Output storage spelling.
    """
    fixtures = [name + suffix for suffix in [
        "-input.npy",
        "-window.npy",
        "-forward-expected.npy",
        "-power-expected.npy",
    ]]
    native.genrule(
        name = name + "_data",
        testonly = True,
        outs = fixtures + [name + ".loom"],
        cmd = "$(execpath :workgroup_test_data) $(@D)/%s %s %s %s %s" % (name, length, frames, input_type, output_type),
        tools = [":workgroup_test_data"],
    )
    compatible = loom_config_compatible_with(["//loom/config/import:cxx"])
    loom_test_module(
        name = name + "_module",
        srcs = ["workgroup_test.cxx", name + ".loom"],
        data = [":headers"] + fixtures,
        inputopts = ["cxx:" + " ".join([
            "D=FFT_LENGTH=%s" % length,
            "D=FFT_INPUT=%s" % _STORAGE[input_type],
            "D=FFT_OUTPUT=%s" % _STORAGE[output_type],
        ])],
        target_compatible_with = compatible,
    )
    for mode in ["forward", "power"]:
        loom_test(
            name = name + "_" + mode,
            module = ":" + name + "_module",
            case = "@" + mode,
            execution_profiles = [AMDGPU_HARDWARE_PROFILE],
            benchmark_smoke = False,
            tags = ["manual"],
            target_compatible_with = compatible,
        )

def fft_test(name, length, width, frames, input_type = "f32", output_type = "f32", frequency_only = False):
    """Separates each direction so default checks retain a small JIT surface.

    Args:
        name: Test family name and fixture prefix.
        length: Exact real transform length.
        width: Maximum number of frames in one packet.
        frames: Runtime frame count supplied by the fixture.
        input_type: Input storage spelling, f32, f16, or bf16.
        output_type: Output storage spelling, f32, f16, or bf16.
        frequency_only: Uses the single-frame schedule and its table suffix.
    """
    fixtures = [name + suffix for suffix in [
        "-table.npy",
        "-forward-input.npy",
        "-forward-expected.npy",
        "-inverse-input.npy",
        "-inverse-expected.npy",
    ]]
    native.genrule(
        name = name + "_data",
        testonly = True,
        outs = fixtures + [name + ".loom"],
        cmd = "$(execpath :test_data) $(@D)/%s %s %s %s %s %s%s" % (name, length, width, frames, input_type, output_type, " --frequency-only" if frequency_only else ""),
        tools = [":test_data"],
    )
    compatible = loom_config_compatible_with(["//loom/config/import:cxx"])
    loom_test_module(
        name = name + "_module",
        srcs = ["fft_test.cxx", name + ".loom"],
        deps = [":layout"],
        data = [":headers"] + fixtures,
        inputopts = ["cxx:" + " ".join([
            "D=FFT_LENGTH=%s" % length,
            "D=FFT_WIDTH=%s" % width,
            "D=FFT_INPUT=%s" % _STORAGE[input_type],
            "D=FFT_OUTPUT=%s" % _STORAGE[output_type],
            "D=FFT_FREQUENCY_ONLY=%s" % (1 if frequency_only else 0),
        ])],
        target_compatible_with = compatible,
    )
    for direction in ["forward", "inverse"]:
        loom_test(
            name = name + "_" + direction,
            module = ":" + name + "_module",
            case = "@" + direction,
            execution_profiles = [VM_REFERENCE_PROFILE],
            benchmark_smoke = False,
            target_compatible_with = compatible,
        )

        # AVX-512 is a qualification-machine requirement, not a generic CPU CI
        # assumption. Explicit runs use the same small source and fixtures.
        loom_test(
            name = name + "_" + direction + "_native",
            module = ":" + name + "_module",
            case = "@" + direction,
            execution_profiles = [_NATIVE_PROFILE],
            benchmark_smoke = False,
            tags = ["manual"],
            target_compatible_with = compatible + ["@platforms//cpu:x86_64"],
        )

def fft_composition_test(name):
    """Checks separately imported callers sharing one provider library.

    Args:
        name: Test name and linked-module prefix.
    """
    compatible = loom_config_compatible_with(["//loom/config/import:cxx"])
    loom_test_module(
        name = name + "_module",
        srcs = ["fft_test.cxx", "composition_test.cxx", "composition_test.loom"],
        data = [":headers", "f32_f32-table.npy"],
        deps = [":layout"],
        inputopts = ["cxx:D=FFT_LENGTH=10 D=FFT_WIDTH=2 D=FFT_INPUT=float D=FFT_OUTPUT=float"],
        target_compatible_with = compatible,
    )
    loom_test(
        name = name,
        module = ":" + name + "_module",
        execution_profiles = [VM_REFERENCE_PROFILE],
        benchmark_smoke = False,
        target_compatible_with = compatible,
    )
    loom_test(
        name = name + "_native",
        module = ":" + name + "_module",
        execution_profiles = [_NATIVE_PROFILE],
        benchmark_smoke = False,
        tags = ["manual"],
        target_compatible_with = compatible + ["@platforms//cpu:x86_64"],
    )
