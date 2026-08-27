# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU device binary target selection."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load(
    "//build_tools/amdgpu:selectors.bzl",
    "IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY",
    "iree_amdgpu_expand_target_selectors",
    "iree_amdgpu_selectors_for_device_binary_target",
    "iree_amdgpu_target_label_fragment",
    "iree_amdgpu_target_selector_config_settings",
)
load("//build_tools/amdgpu:target_map.bzl", "IREE_AMDGPU_DEVICE_BINARY_TARGETS")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

_BUILD_MODE_PREBUILT = "prebuilt"
_BUILD_MODE_SOURCE = "source"
_VALID_BUILD_MODES = [
    _BUILD_MODE_PREBUILT,
    _BUILD_MODE_SOURCE,
]

_PREBUILT_DEVICE_BINARY_TARGETS = {
    "gfx10-1-generic": True,
    "gfx10-3-generic": True,
    "gfx11-generic": True,
    "gfx12-generic": True,
    "gfx9-4-generic": True,
    "gfx9-generic": True,
    "gfx90a": True,
}

# Keep intentionally unbuildable helper targets out of //... CI enumeration.
_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

def iree_hal_amdgpu_expand_device_binary_targets(targets):
    return iree_amdgpu_expand_target_selectors(
        targets,
        mode = IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY,
    )

def _device_binary_build_mode_flag_impl(ctx):
    if ctx.build_setting_value not in _VALID_BUILD_MODES:
        fail("Unknown AMDGPU device binary build mode '{}'. Available: {}".format(
            ctx.build_setting_value,
            ", ".join(_VALID_BUILD_MODES),
        ))
    return BuildSettingInfo(value = ctx.build_setting_value)

_device_binary_build_mode_flag = rule(
    implementation = _device_binary_build_mode_flag_impl,
    build_setting = config.string(flag = True),
)

def iree_hal_amdgpu_device_binaries(
        name,
        target_selectors_flag = "//runtime/src/iree/hal/drivers/amdgpu:targets",
        target = "amdgcn-amd-amdhsa"):
    """Embeds selected AMDGPU builtin device binaries into the runtime.

    Args:
      name: C embed-data target name.
      target_selectors_flag: Label of the AMDGPU target selector build setting.
      target: LLVM target triple prefix used in blob filenames.
    """
    _device_binary_build_mode_flag(
        name = "build_mode",
        build_setting_default = _BUILD_MODE_PREBUILT,
    )

    native.config_setting(
        name = "build_mode_prebuilt",
        flag_values = {
            ":build_mode": _BUILD_MODE_PREBUILT,
        },
    )

    native.config_setting(
        name = "build_mode_source",
        flag_values = {
            ":build_mode": _BUILD_MODE_SOURCE,
        },
    )

    target_selection = iree_amdgpu_target_selector_config_settings(
        name = "target",
        flag = target_selectors_flag,
    )

    device_binary_requested = dict(target_selection.requested)
    for device_binary_target in IREE_AMDGPU_DEVICE_BINARY_TARGETS:
        if device_binary_target in device_binary_requested:
            continue
        requested_name = "target_%s_requested" % (
            iree_amdgpu_target_label_fragment(device_binary_target),
        )
        selects.config_setting_group(
            name = requested_name,
            match_any = [
                target_selection.selected[selector]
                for selector in iree_amdgpu_selectors_for_device_binary_target(
                    device_binary_target,
                )
            ],
        )
        device_binary_requested[device_binary_target] = ":%s" % (requested_name,)

    binary_srcs = []
    for device_binary_target in IREE_AMDGPU_DEVICE_BINARY_TARGETS:
        binary_name = "%s--%s" % (target, device_binary_target)
        selects.config_setting_group(
            name = "%s_requested_prebuilt" % (iree_amdgpu_target_label_fragment(device_binary_target),),
            match_all = [
                device_binary_requested[device_binary_target],
                ":build_mode_prebuilt",
            ],
        )
        selects.config_setting_group(
            name = "%s_requested_source" % (iree_amdgpu_target_label_fragment(device_binary_target),),
            match_all = [
                device_binary_requested[device_binary_target],
                ":build_mode_source",
            ],
        )
        if device_binary_target in _PREBUILT_DEVICE_BINARY_TARGETS:
            prebuilt_label = "//runtime/src/iree/hal/drivers/amdgpu/device/binaries/prebuilt:%s.so" % (binary_name,)
        else:
            missing_prebuilt_name = "missing_prebuilt_%s" % (iree_amdgpu_target_label_fragment(device_binary_target),)
            native.filegroup(
                name = missing_prebuilt_name,
                target_compatible_with = _INCOMPATIBLE_TARGET,
            )
            prebuilt_label = ":%s" % (missing_prebuilt_name,)
        source_label = "//runtime/src/iree/hal/drivers/amdgpu/device/binaries/source:%s.so" % (binary_name,)
        binary_srcs += select({
            ":%s_requested_prebuilt" % (iree_amdgpu_target_label_fragment(device_binary_target),): [prebuilt_label],
            ":%s_requested_source" % (iree_amdgpu_target_label_fragment(device_binary_target),): [source_label],
            "//conditions:default": [],
        })
    iree_c_embed_data(
        name = name,
        srcs = binary_srcs,
        c_file_output = "toc.c",
        flatten = True,
        h_file_output = "toc.h",
        identifier = "iree_hal_amdgpu_device_binaries",
    )
