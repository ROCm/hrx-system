# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom AMDGPU target selector helpers."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load(
    "//build_tools/amdgpu:selectors.bzl",
    "IREE_AMDGPU_TARGET_EXPANSION_EXACT",
    "iree_amdgpu_expand_target_selectors",
    "iree_amdgpu_selectors_for_exact_target",
    "iree_amdgpu_target_label_fragment",
    "iree_amdgpu_valid_selectors",
)
load(
    ":target_config.bzl",
    "LOOM_AMDGPU_DEFAULT_TARGET_SELECTORS",
    "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES",
    "LOOM_AMDGPU_DESCRIPTOR_SET_EXACT_PROCESSORS",
    "LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS",
    "LOOM_AMDGPU_TARGET_SOURCE_IREE_HAL",
    "LOOM_AMDGPU_TARGET_SOURCE_LOOM_DEFAULTS",
    "LOOM_AMDGPU_TARGET_SOURCE_SELECTORS",
)

def _append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)

def _is_supported_exact_target(target):
    return target in LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS

def _selector_expands_only_to_supported_processors(selector):
    expanded_targets = iree_amdgpu_expand_target_selectors(
        [selector],
        mode = IREE_AMDGPU_TARGET_EXPANSION_EXACT,
    )
    if not expanded_targets:
        return False
    for target in expanded_targets:
        if not _is_supported_exact_target(target):
            return False
    return True

def loom_amdgpu_valid_target_selectors():
    """Returns selectors accepted by Loom's AMDGPU compile-target config.

    Returns:
      Source selectors and AMDGPU hardware selectors whose exact expansion is
      fully supported by Loom.
    """
    valid_selectors = []
    _append_unique(valid_selectors, LOOM_AMDGPU_TARGET_SOURCE_SELECTORS)
    for selector in iree_amdgpu_valid_selectors():
        if _selector_expands_only_to_supported_processors(selector):
            _append_unique(valid_selectors, [selector])
    return valid_selectors

def _target_selectors_flag_impl(ctx):
    valid_selectors = loom_amdgpu_valid_target_selectors()
    invalid_selectors = [
        selector
        for selector in ctx.build_setting_value
        if selector not in valid_selectors
    ]
    if invalid_selectors:
        fail("Unknown or unsupported Loom AMDGPU target selector(s) [{}]. Available: {}".format(
            ", ".join(invalid_selectors),
            ", ".join(valid_selectors),
        ))
    return BuildSettingInfo(value = ctx.build_setting_value)

_target_selectors_flag = rule(
    implementation = _target_selectors_flag_impl,
    build_setting = config.string_list(flag = True),
)

def loom_amdgpu_target_selectors_flag(
        name,
        build_setting_default = LOOM_AMDGPU_DEFAULT_TARGET_SELECTORS):
    """Defines a Loom AMDGPU target selector build setting."""
    _target_selectors_flag(
        name = name,
        build_setting_default = build_setting_default,
    )

def _processor_label(processor, prefix = ""):
    label = "processor_" + iree_amdgpu_target_label_fragment(processor)
    if prefix:
        return prefix + "_" + label
    return label

def _descriptor_set_label(capability, prefix = ""):
    if prefix:
        return prefix + "_" + capability
    return capability

def _selectors_for_exact_target(exact_target, valid_selectors):
    selectors = []
    for selector in iree_amdgpu_selectors_for_exact_target(exact_target):
        if selector in valid_selectors:
            _append_unique(selectors, [selector])
    return selectors

def _exact_target_selector_config_settings(
        name,
        flag,
        exact_targets,
        valid_selectors):
    selected = {}
    for exact_target in exact_targets:
        for selector in _selectors_for_exact_target(exact_target, valid_selectors):
            if selector in selected:
                continue
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
        selectors = _selectors_for_exact_target(exact_target, valid_selectors)
        if not selectors:
            fail("No valid selector reaches Loom AMDGPU target {}".format(exact_target))
        requested_name = "{}_{}_requested".format(
            name,
            iree_amdgpu_target_label_fragment(exact_target),
        )
        selects.config_setting_group(
            name = requested_name,
            match_any = [selected[selector] for selector in selectors],
        )
        requested[exact_target] = ":" + requested_name

    return requested

def loom_amdgpu_target_config_settings(
        name,
        flag,
        iree_hal_flag = None):
    """Creates public Loom AMDGPU target capability config settings.

    Args:
      name: Logical name for the selector settings group. The generated target
        labels intentionally stay canonical because consumers depend on them by
        capability name.
      flag: Label string of a `loom_amdgpu_target_selectors_flag`.
      iree_hal_flag: Optional runtime IREE HAL AMDGPU selector flag mirrored by
        the `iree_hal` source selector and the direct `iree_hal_*` capability
        labels. Leave unset for standalone Loom target configuration that must
        not depend on the IREE HAL package graph.

    Returns:
      struct with `descriptor_sets` and `iree_hal_descriptor_sets`
      dictionaries mapping capability names to public config labels.
    """
    if not name:
        fail("loom_amdgpu_target_config_settings requires a non-empty name")

    native.config_setting(
        name = "target_source_" + LOOM_AMDGPU_TARGET_SOURCE_LOOM_DEFAULTS,
        flag_values = {
            flag: LOOM_AMDGPU_TARGET_SOURCE_LOOM_DEFAULTS,
        },
    )

    if iree_hal_flag:
        native.config_setting(
            name = "target_source_" + LOOM_AMDGPU_TARGET_SOURCE_IREE_HAL,
            flag_values = {
                flag: LOOM_AMDGPU_TARGET_SOURCE_IREE_HAL,
            },
        )

    explicit_loom_selectors = [
        selector
        for selector in loom_amdgpu_valid_target_selectors()
        if selector not in LOOM_AMDGPU_TARGET_SOURCE_SELECTORS
    ]
    loom_processors = _exact_target_selector_config_settings(
        name = "selected_processor",
        flag = flag,
        exact_targets = LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS,
        valid_selectors = explicit_loom_selectors,
    )
    iree_hal_processors = {}
    if iree_hal_flag:
        iree_hal_processors = _exact_target_selector_config_settings(
            name = "iree_hal_selected_processor",
            flag = iree_hal_flag,
            exact_targets = LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS,
            valid_selectors = iree_amdgpu_valid_selectors(),
        )

    for processor in LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS:
        processor_matches = [
            ":target_source_" + LOOM_AMDGPU_TARGET_SOURCE_LOOM_DEFAULTS,
            loom_processors[processor],
        ]
        if iree_hal_flag:
            iree_hal_processor = _processor_label(processor, prefix = "iree_hal")
            iree_hal_processor_from_source = _processor_label(
                processor,
                prefix = "target_source_iree_hal",
            )
            selects.config_setting_group(
                name = iree_hal_processor,
                match_any = [iree_hal_processors[processor]],
            )
            selects.config_setting_group(
                name = iree_hal_processor_from_source,
                match_all = [
                    ":target_source_" + LOOM_AMDGPU_TARGET_SOURCE_IREE_HAL,
                    ":" + iree_hal_processor,
                ],
            )
            processor_matches.append(":" + iree_hal_processor_from_source)
        selects.config_setting_group(
            name = _processor_label(processor),
            match_any = processor_matches,
        )

    descriptor_sets = {}
    iree_hal_descriptor_sets = {}
    for capability in LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES:
        descriptor_set_matches = [
            ":target_source_" + LOOM_AMDGPU_TARGET_SOURCE_LOOM_DEFAULTS,
        ] + [
            loom_processors[processor]
            for processor in LOOM_AMDGPU_DESCRIPTOR_SET_EXACT_PROCESSORS[capability]
        ]
        if iree_hal_flag:
            iree_hal_capability = _descriptor_set_label(capability, prefix = "iree_hal")
            iree_hal_from_source = _descriptor_set_label(
                capability,
                prefix = "target_source_iree_hal",
            )
            selects.config_setting_group(
                name = iree_hal_capability,
                match_any = [
                    iree_hal_processors[processor]
                    for processor in LOOM_AMDGPU_DESCRIPTOR_SET_EXACT_PROCESSORS[capability]
                ],
            )
            selects.config_setting_group(
                name = iree_hal_from_source,
                match_all = [
                    ":target_source_" + LOOM_AMDGPU_TARGET_SOURCE_IREE_HAL,
                    ":" + iree_hal_capability,
                ],
            )
            descriptor_set_matches.append(":" + iree_hal_from_source)
            iree_hal_descriptor_sets[capability] = ":" + iree_hal_capability
        selects.config_setting_group(
            name = capability,
            match_any = descriptor_set_matches,
        )
        descriptor_sets[capability] = ":" + capability

    return struct(
        descriptor_sets = descriptor_sets,
        iree_hal_descriptor_sets = iree_hal_descriptor_sets,
    )
