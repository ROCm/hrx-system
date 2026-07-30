# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared AMDGPU target selector helpers."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load(
    "//build_tools/amdgpu:target_map.bzl",
    "IREE_AMDGPU_CODE_OBJECT_TARGETS",
    "IREE_AMDGPU_DEFAULT_TARGET_SELECTORS",
    "IREE_AMDGPU_DEVICE_BINARY_TARGETS",
    "IREE_AMDGPU_EXACT_TARGETS",
    "IREE_AMDGPU_EXACT_TARGET_CODE_OBJECTS",
    "IREE_AMDGPU_EXACT_TARGET_DEVICE_BINARY_VARIANTS",
    "IREE_AMDGPU_TARGET_FAMILIES",
    "IREE_AMDGPU_TARGET_FAMILY_NAMES",
)

IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT = "code-object"
IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY = "device-binary"
IREE_AMDGPU_TARGET_EXPANSION_EXACT = "exact"

def _append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)

def iree_amdgpu_valid_selectors():
    """Returns all accepted AMDGPU target selectors.

    Returns:
      Exact targets, code-object targets, and selector families.
    """
    selectors = []
    _append_unique(selectors, IREE_AMDGPU_EXACT_TARGETS)
    _append_unique(selectors, IREE_AMDGPU_CODE_OBJECT_TARGETS)
    _append_unique(selectors, IREE_AMDGPU_TARGET_FAMILY_NAMES)
    return selectors

def iree_amdgpu_target_label_fragment(target):
    """Returns a target string fragment usable in labels and filenames.

    Args:
      target: Exact target, code-object target, or selector family.

    Returns:
      The target with punctuation that is awkward in generated labels replaced
      by underscores.
    """
    return target.replace("-", "_").replace(".", "_")

def _code_object_target_for_exact(target):
    return IREE_AMDGPU_EXACT_TARGET_CODE_OBJECTS[target]

def _exact_targets_for_code_object(code_object_target):
    exact_targets = []
    for exact_target in IREE_AMDGPU_EXACT_TARGETS:
        if IREE_AMDGPU_EXACT_TARGET_CODE_OBJECTS[exact_target] == code_object_target:
            exact_targets.append(exact_target)
    return exact_targets

def _unknown_selector_error(selector):
    fail("Unknown AMDGPU target selector '{}'. Available: {}".format(
        selector,
        ", ".join(iree_amdgpu_valid_selectors()),
    ))

def _expand_code_object_target_selector(selector):
    if selector in IREE_AMDGPU_CODE_OBJECT_TARGETS:
        return [selector]
    if selector in IREE_AMDGPU_EXACT_TARGETS:
        return [_code_object_target_for_exact(selector)]
    if selector in IREE_AMDGPU_TARGET_FAMILIES:
        expanded_targets = []
        for exact_target in IREE_AMDGPU_TARGET_FAMILIES[selector]:
            _append_unique(expanded_targets, [_code_object_target_for_exact(exact_target)])
        return expanded_targets
    return _unknown_selector_error(selector)

def _expand_exact_target_selector(selector):
    if selector in IREE_AMDGPU_EXACT_TARGETS:
        return [selector]
    if selector in IREE_AMDGPU_CODE_OBJECT_TARGETS:
        return _exact_targets_for_code_object(selector)
    if selector in IREE_AMDGPU_TARGET_FAMILIES:
        return IREE_AMDGPU_TARGET_FAMILIES[selector]
    return _unknown_selector_error(selector)

def _expand_device_binary_target_selector(selector):
    expanded_targets = []
    for exact_target in _expand_exact_target_selector(selector):
        _append_unique(
            expanded_targets,
            IREE_AMDGPU_EXACT_TARGET_DEVICE_BINARY_VARIANTS.get(
                exact_target,
                [],
            ),
        )
        _append_unique(
            expanded_targets,
            [_code_object_target_for_exact(exact_target)],
        )
    return expanded_targets

def iree_amdgpu_expand_target_selectors(
        targets,
        mode = IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT):
    """Expands AMDGPU target selectors to exact or code-object targets.

    Args:
      targets: List of exact targets, code-object targets, or selector families.
      mode: Expansion mode. `code-object` expands to the smallest known set of
        compatible code-object targets. `device-binary` includes any
        exact-target artifact variants required before compatible code-object
        fallbacks. `exact` expands to exact HSA ISA target names.

    Returns:
      Deduplicated list of expanded targets in target-map order.
    """
    expanded_targets = []
    for target in targets:
        if mode == IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT:
            _append_unique(expanded_targets, _expand_code_object_target_selector(target))
        elif mode == IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY:
            _append_unique(expanded_targets, _expand_device_binary_target_selector(target))
        elif mode == IREE_AMDGPU_TARGET_EXPANSION_EXACT:
            _append_unique(expanded_targets, _expand_exact_target_selector(target))
        else:
            fail("Unknown AMDGPU target expansion mode '{}'. Available: {}, {}, {}".format(
                mode,
                IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT,
                IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY,
                IREE_AMDGPU_TARGET_EXPANSION_EXACT,
            ))
    return expanded_targets

def iree_amdgpu_selectors_for_code_object_target(code_object_target):
    """Returns every selector that requests a code-object target.

    Args:
      code_object_target: Code-object target to match.

    Returns:
      Selectors that should make the code-object target available.
    """
    selectors = [code_object_target]
    for exact_target in _exact_targets_for_code_object(code_object_target):
        _append_unique(selectors, [exact_target])
    for family in IREE_AMDGPU_TARGET_FAMILY_NAMES:
        if code_object_target in iree_amdgpu_expand_target_selectors([family]):
            _append_unique(selectors, [family])
    return selectors

def iree_amdgpu_selectors_for_exact_target(exact_target):
    """Returns every selector that requests an exact target.

    Args:
      exact_target: Exact HSA ISA target to match.

    Returns:
      Selectors that should make the exact target available.
    """
    if exact_target not in IREE_AMDGPU_EXACT_TARGETS:
        fail("Unknown AMDGPU exact target '{}'. Available: {}".format(
            exact_target,
            ", ".join(IREE_AMDGPU_EXACT_TARGETS),
        ))

    selectors = [exact_target]
    code_object_target = _code_object_target_for_exact(exact_target)
    _append_unique(selectors, [code_object_target])
    for family in IREE_AMDGPU_TARGET_FAMILY_NAMES:
        if exact_target in iree_amdgpu_expand_target_selectors(
            [family],
            mode = IREE_AMDGPU_TARGET_EXPANSION_EXACT,
        ):
            _append_unique(selectors, [family])
    return selectors

def iree_amdgpu_selectors_for_device_binary_target(device_binary_target):
    """Returns every selector that requests a device-binary target.

    Args:
      device_binary_target: Code-object fallback or target-overlay artifact.

    Returns:
      Public target selectors that require the device-binary artifact.
    """
    if device_binary_target not in IREE_AMDGPU_DEVICE_BINARY_TARGETS:
        fail("Unknown AMDGPU device-binary target '{}'. Available: {}".format(
            device_binary_target,
            ", ".join(IREE_AMDGPU_DEVICE_BINARY_TARGETS),
        ))
    if device_binary_target in IREE_AMDGPU_CODE_OBJECT_TARGETS:
        return iree_amdgpu_selectors_for_code_object_target(device_binary_target)

    selectors = []
    for exact_target in IREE_AMDGPU_EXACT_TARGETS:
        if device_binary_target in IREE_AMDGPU_EXACT_TARGET_DEVICE_BINARY_VARIANTS.get(
            exact_target,
            [],
        ):
            _append_unique(
                selectors,
                iree_amdgpu_selectors_for_exact_target(exact_target),
            )
    return selectors

def _target_selectors_flag_impl(ctx):
    valid_selectors = iree_amdgpu_valid_selectors()
    invalid_selectors = [
        selector
        for selector in ctx.build_setting_value
        if selector not in valid_selectors
    ]
    if invalid_selectors:
        fail("Unknown AMDGPU target selector(s) [{}]. Available: {}".format(
            ", ".join(invalid_selectors),
            ", ".join(valid_selectors),
        ))
    return BuildSettingInfo(value = ctx.build_setting_value)

_target_selectors_flag = rule(
    implementation = _target_selectors_flag_impl,
    build_setting = config.string_list(flag = True),
)

def iree_amdgpu_target_selectors_flag(
        name,
        build_setting_default = IREE_AMDGPU_DEFAULT_TARGET_SELECTORS):
    """Defines an AMDGPU target selector string-list build setting.

    Args:
      name: Build setting target name.
      build_setting_default: Default selector list.
    """
    _target_selectors_flag(
        name = name,
        build_setting_default = build_setting_default,
    )

def iree_amdgpu_target_selector_config_settings(
        name,
        flag,
        code_object_targets = IREE_AMDGPU_CODE_OBJECT_TARGETS):
    """Creates config_settings for selectors and requested code objects.

    Args:
      name: Prefix for generated config_setting targets.
      flag: Label string of an iree_amdgpu_target_selectors_flag target.
      code_object_targets: Code-object targets to create requested groups for.

    Returns:
      struct with `selected` and `requested` dictionaries. `selected` maps any
      selector to its config_setting label. `requested` maps each code-object
      target to a config_setting_group that matches that target, any exact
      target covered by it, or any family covering it.
    """
    selected = {}
    for selector in iree_amdgpu_valid_selectors():
        setting_name = "{}_{}_selected".format(
            name,
            iree_amdgpu_target_label_fragment(selector),
        )
        native.config_setting(
            name = setting_name,
            flag_values = {
                flag: selector,
            },
        )
        selected[selector] = ":" + setting_name

    requested = {}
    for code_object_target in code_object_targets:
        if code_object_target not in IREE_AMDGPU_CODE_OBJECT_TARGETS:
            fail("Unknown AMDGPU code-object target '{}'. Available: {}".format(
                code_object_target,
                ", ".join(IREE_AMDGPU_CODE_OBJECT_TARGETS),
            ))
        requested_name = "{}_{}_requested".format(
            name,
            iree_amdgpu_target_label_fragment(code_object_target),
        )
        selects.config_setting_group(
            name = requested_name,
            match_any = [
                selected[selector]
                for selector in iree_amdgpu_selectors_for_code_object_target(
                    code_object_target,
                )
            ],
        )
        requested[code_object_target] = ":" + requested_name

    return struct(
        requested = requested,
        selected = selected,
    )

def iree_amdgpu_exact_target_selector_config_settings(
        name,
        flag,
        exact_targets = IREE_AMDGPU_EXACT_TARGETS):
    """Creates config_settings for requested exact AMDGPU targets.

    Args:
      name: Prefix for generated config_setting targets.
      flag: Label string of an iree_amdgpu_target_selectors_flag target.
      exact_targets: Exact targets to create requested groups for.

    Returns:
      Dictionary mapping each exact target to a config_setting_group that
      matches that exact target, a covering code-object target, or a covering
      target family.
    """
    selected = {}
    for selector in iree_amdgpu_valid_selectors():
        setting_name = "{}_{}_selected".format(
            name,
            iree_amdgpu_target_label_fragment(selector),
        )
        native.config_setting(
            name = setting_name,
            flag_values = {
                flag: selector,
            },
        )
        selected[selector] = ":" + setting_name

    requested = {}
    for exact_target in exact_targets:
        if exact_target not in IREE_AMDGPU_EXACT_TARGETS:
            fail("Unknown AMDGPU exact target '{}'. Available: {}".format(
                exact_target,
                ", ".join(IREE_AMDGPU_EXACT_TARGETS),
            ))
        requested_name = "{}_{}_requested".format(
            name,
            iree_amdgpu_target_label_fragment(exact_target),
        )
        selects.config_setting_group(
            name = requested_name,
            match_any = [
                selected[selector]
                for selector in iree_amdgpu_selectors_for_exact_target(exact_target)
            ],
        )
        requested[exact_target] = ":" + requested_name

    return requested
