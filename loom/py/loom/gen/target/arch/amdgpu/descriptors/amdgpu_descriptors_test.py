# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from loom.gen.target.arch.amdgpu.descriptors import amdgpu_descriptors
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AmdgpuDescriptorSetInfo,
)
from loom.target.low_descriptors import Descriptor, DescriptorSet


def _descriptor(index: int) -> Descriptor:
    return Descriptor(
        key=f"amdgpu.test{index}",
        mnemonic=None,
        semantic_tag=None,
        operands=(),
        schedule_class="amdgpu.test",
    )


def _descriptor_set(target: str, descriptor_count: int) -> DescriptorSet:
    return DescriptorSet(
        key=f"amdgpu.{target.replace('_', '.')}.core",
        target_key="amdgpu",
        feature_key=f"amdgpu.{target}.v1",
        c_header_path=Path(f"{target}_descriptors.h"),
        c_source_path=Path(f"{target}_descriptors.c"),
        header_guard=f"{target.upper()}_DESCRIPTORS_H_",
        public_header=f"loom/target/arch/amdgpu/{target}_descriptors.h",
        function_name=f"loom_amdgpu_{target}_core_descriptor_set",
        c_table_prefix=f"Amdgpu{target.title()}Core",
        c_enum_prefix=f"AMDGPU_{target.upper()}_CORE",
        generator_version=1,
        reg_classes=(),
        resources=(),
        schedule_classes=(),
        descriptors=tuple(_descriptor(index) for index in range(descriptor_count)),
    )


def _descriptor_set_info(
    target: str,
    *,
    storage_target: str | None = None,
) -> AmdgpuDescriptorSetInfo:
    return AmdgpuDescriptorSetInfo(
        generator_target=target,
        key=f"amdgpu.{target}.core",
        isa_xml_key="test",
        isa_architecture_name="AMDGPU Test",
        isa_architecture_id=0,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        storage_generator_target=storage_target,
    )


def test_storage_generation_reuses_parsed_isa_for_declared_views() -> None:
    storage_target = "test_storage"
    view_target = "test_view"
    storage_info = _descriptor_set_info(storage_target)
    view_info = _descriptor_set_info(view_target, storage_target=storage_target)
    parsed_spec = object()
    parse_calls: list[Path] = []
    build_calls: list[tuple[str, object]] = []

    def parse_xml(path: Path) -> object:
        parse_calls.append(path)
        return parsed_spec

    def build_descriptor_set(target: str, spec: object) -> DescriptorSet:
        build_calls.append((target, spec))
        descriptor_count = 2 if target == view_target else 1
        return _descriptor_set(target, descriptor_count)

    def generate_descriptor_set(descriptor_set: DescriptorSet) -> SimpleNamespace:
        return SimpleNamespace(header=f"// {descriptor_set.key}\n")

    def generate_descriptor_set_shared_source(
        storage_descriptor_set: DescriptorSet,
        view_descriptor_sets: tuple[DescriptorSet, ...],
    ) -> str:
        return "// shared\n"

    def descriptor_set_info_by_target(target: str) -> AmdgpuDescriptorSetInfo:
        if target == storage_target:
            return storage_info
        if target == view_target:
            return view_info
        raise ValueError(target)

    def storage_info_by_target(target: str) -> AmdgpuDescriptorSetInfo:
        if target in (storage_target, view_target):
            return storage_info
        raise ValueError(target)

    def view_infos_by_storage_target(
        target: str,
    ) -> tuple[AmdgpuDescriptorSetInfo, ...]:
        if target == storage_target:
            return (view_info,)
        return ()

    with (
        TemporaryDirectory() as temporary_directory,
        mock.patch.object(
            amdgpu_descriptors,
            "AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS",
            (storage_target, view_target),
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "amdgpu_descriptor_set_info_by_generator_target",
            descriptor_set_info_by_target,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "amdgpu_descriptor_set_storage_info_by_generator_target",
            storage_info_by_target,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "amdgpu_descriptor_set_view_infos_by_storage_generator_target",
            view_infos_by_storage_target,
        ),
        mock.patch.object(amdgpu_descriptors, "parse_amdgpu_isa_xml_path", parse_xml),
        mock.patch.object(
            amdgpu_descriptors,
            "build_amdgpu_core_descriptor_set_from_spec",
            build_descriptor_set,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "generate_descriptor_set",
            generate_descriptor_set,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "generate_descriptor_set_shared_source",
            generate_descriptor_set_shared_source,
        ),
    ):
        tmp_path = Path(temporary_directory)
        xml_path = tmp_path / "amdgpu_isa_test.xml"
        view_header_path = tmp_path / "test_view_descriptors.h"
        assert (
            amdgpu_descriptors.main(
                [
                    f"--target={storage_target}",
                    f"--xml={xml_path}",
                    f"--header={tmp_path / 'test_storage_descriptors.h'}",
                    f"--source={tmp_path / 'test_storage_descriptors.c'}",
                    f"--view-header={view_target}={view_header_path}",
                ]
            )
            == 0
        )

    assert parse_calls == [xml_path]
    assert build_calls == [
        (storage_target, parsed_spec),
        (view_target, parsed_spec),
    ]
