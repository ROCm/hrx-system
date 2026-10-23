# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

from loom.target.arch.x86.descriptors import X86_AVX512_FEATURES_DESCRIPTOR_SET
from loom.target.arch.x86.machine_models import (
    X86_ZNVER5_FLOAT_SCHEDULE_BEHAVIORS,
    X86_ZNVER5_SCHEDULE_BEHAVIORS,
    apply_x86_machine_schedule_models,
    x86_register_only_floating_descriptor_keys,
    x86_register_only_integer_multiply_descriptor_keys,
)


def test_float_schedule_data_covers_complete_register_family() -> None:
    assert set(X86_ZNVER5_FLOAT_SCHEDULE_BEHAVIORS) == set(
        x86_register_only_floating_descriptor_keys(X86_AVX512_FEATURES_DESCRIPTOR_SET)
    )


def test_integer_multiply_schedule_data_covers_complete_register_family() -> None:
    assert set(X86_ZNVER5_SCHEDULE_BEHAVIORS).difference(
        X86_ZNVER5_FLOAT_SCHEDULE_BEHAVIORS
    ) == set(
        x86_register_only_integer_multiply_descriptor_keys(
            X86_AVX512_FEATURES_DESCRIPTOR_SET
        )
    )


def test_machine_models_preserve_inline_behavior_by_descriptor() -> None:
    source = X86_AVX512_FEATURES_DESCRIPTOR_SET
    projected = apply_x86_machine_schedule_models(source)
    source_classes = {
        schedule_class.name: schedule_class
        for schedule_class in source.schedule_classes
    }
    projected_descriptors = {
        descriptor.key: descriptor for descriptor in projected.descriptors
    }
    projected_classes = {
        schedule_class.name: schedule_class
        for schedule_class in projected.schedule_classes
    }
    for descriptor_key in X86_ZNVER5_SCHEDULE_BEHAVIORS:
        source_descriptor = source.descriptor_by_key(descriptor_key)
        assert source_descriptor is not None
        assert projected_descriptors[descriptor_key].schedule_class == descriptor_key
        assert projected_classes[descriptor_key] == replace(
            source_classes[source_descriptor.schedule_class],
            name=descriptor_key,
        )


def test_machine_models_encode_latency_and_half_cycle_throughput() -> None:
    projected = apply_x86_machine_schedule_models(X86_AVX512_FEATURES_DESCRIPTOR_SET)
    resources = {resource.name: resource for resource in projected.resources}
    assert tuple(model.key for model in projected.schedule_models) == (
        "x86.znver5",
        "x86.znver6",
    )
    for model in projected.schedule_models:
        classes = {
            schedule_class.name: schedule_class
            for schedule_class in model.schedule_classes
        }
        for descriptor_key, behavior in X86_ZNVER5_SCHEDULE_BEHAVIORS.items():
            schedule_class = classes[descriptor_key]
            assert schedule_class.latency_cycles == behavior.latency_cycles
            assert (
                schedule_class.minimum_issue_separation_cycles
                == behavior.latency_cycles
            )
            assert len(schedule_class.issue_uses) == 1
            issue_use = schedule_class.issue_uses[0]
            resource = resources[issue_use.resource]
            assert (
                2 * issue_use.cycles * issue_use.units
                == behavior.reciprocal_throughput_half_cycles
                * resource.capacity_per_cycle
            )


def test_zen5_and_zen6_current_behavior_tables_pool_exactly() -> None:
    projected = apply_x86_machine_schedule_models(X86_AVX512_FEATURES_DESCRIPTOR_SET)
    assert (
        projected.schedule_models[0].schedule_classes
        == projected.schedule_models[1].schedule_classes
    )
