# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom AMDGPU descriptor-set build helpers."""

load("//loom/build_tools/bazel:build_defs.bzl", "loom_config_compatible_with")
load(
    ":target_config.bzl",
    "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES",
    "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES_BY_STORAGE_GENERATOR_TARGET",
    "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_KEY",
    "LOOM_AMDGPU_DESCRIPTOR_SET_DEFINES",
    "LOOM_AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS",
)

_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

def _descriptor_set_config_label(capability):
    if capability not in LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES:
        fail("Unknown Loom AMDGPU descriptor-set capability: {}".format(capability))
    return "//loom/config/target/amdgpu:" + capability

def loom_amdgpu_descriptor_set_compatible_with(capability):
    """Returns target compatibility for a selected descriptor-set capability.

    Returns:
      A `target_compatible_with` value that disables the target unless
      `capability` is selected by Loom's AMDGPU target configuration.
    """
    return loom_config_compatible_with([_descriptor_set_config_label(capability)])

def loom_amdgpu_descriptor_table_compatible_with(storage_generator_target):
    """Returns compatibility for contracts backed by a descriptor table.

    Descriptor contracts may be views over a shared generated table. The
    storage library is compatible when its own contract or any view backed by
    it is selected.

    Args:
      storage_generator_target: Generator target owning the descriptor table.

    Returns:
      A `target_compatible_with` value for every contract using that table.
    """
    capabilities = LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES_BY_STORAGE_GENERATOR_TARGET.get(
        storage_generator_target,
    )
    if not capabilities:
        fail("Unknown AMDGPU descriptor storage target: {}".format(storage_generator_target))
    compatibility = {
        "//conditions:default": _INCOMPATIBLE_TARGET,
    }
    for capability in capabilities:
        compatibility[_descriptor_set_config_label(capability)] = []
    return select(compatibility)

def loom_amdgpu_selected_descriptor_set_defines():
    """Returns C defines for the selected descriptor-set capabilities.

    Returns:
      A configurable list of preprocessor defines, one per selected descriptor
      set.
    """
    defines = []
    for capability in LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES:
        defines = defines + select({
            _descriptor_set_config_label(capability): [
                LOOM_AMDGPU_DESCRIPTOR_SET_DEFINES[capability],
            ],
            "//conditions:default": [],
        })
    return defines

def loom_amdgpu_selected_descriptor_set_values(values):
    """Selects list values keyed by descriptor-set capability.

    Args:
      values: Dictionary from descriptor-set capability to a string or list of
        strings.

    Returns:
      A configurable list containing only values for selected capabilities.
    """
    result = []
    for capability in LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES:
        selected_values = values.get(capability, [])
        if type(selected_values) == type(""):
            selected_values = [selected_values]
        result = result + select({
            _descriptor_set_config_label(capability): selected_values,
            "//conditions:default": [],
        })
    return result

def loom_amdgpu_selected_descriptor_set_generator_args(args):
    """Selects generator args keyed by descriptor-set capability.

    Args:
      args: Dictionary from descriptor-set capability to a string or list of
        strings. Args may use `$(rootpath ...)` so bazel_to_cmake can map
        fetched descriptor data to the matching source directory.

    Returns:
      A configurable list containing only args for selected capabilities.
    """
    bazel_args = {}
    for capability, selected_args in args.items():
        if type(selected_args) == type(""):
            selected_args = [selected_args]
        bazel_args[capability] = [
            arg.replace("$(rootpath ", "$(location ").replace(
                "$(execpath ",
                "$(location ",
            )
            for arg in selected_args
        ]
    return loom_amdgpu_selected_descriptor_set_values(bazel_args)

def loom_amdgpu_selected_descriptor_set_key_args():
    """Returns generator args naming every selected descriptor set."""
    return loom_amdgpu_selected_descriptor_set_values({
        capability: "--descriptor-set=" + descriptor_set_key
        for descriptor_set_key, capability in LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_KEY.items()
    })

def loom_amdgpu_selected_descriptor_set_deps(targets):
    """Selects dependency labels keyed by descriptor-set capability.

    Args:
      targets: Dictionary from descriptor-set capability to dependency labels.

    Returns:
      A dependency list containing only targets for selected capabilities.
    """
    return loom_amdgpu_selected_descriptor_set_values(targets)

def loom_amdgpu_low_descriptor_deps():
    """Returns selected local low-descriptor table library deps.

    Returns:
      A configurable dependency list for descriptor table libraries in the
      current package.
    """
    return loom_amdgpu_selected_descriptor_set_deps({
        capability: ":" + LOOM_AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS[capability]
        for capability in LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES
    })

def loom_amdgpu_encoding_table_deps():
    """Returns selected local encoding table library deps.

    Returns:
      A configurable dependency list for encoding table libraries in the
      current package.
    """
    return loom_amdgpu_selected_descriptor_set_deps({
        capability: ":" + LOOM_AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS[capability] + "_encoding_tables"
        for capability in LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES
    })
