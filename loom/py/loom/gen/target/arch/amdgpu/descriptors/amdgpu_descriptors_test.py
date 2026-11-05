# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from loom.gen.target.arch.amdgpu.amdgpu_target_table_family import (
    AmdgpuTargetTableFamily,
    amdgpu_target_table_family,
)
from loom.gen.target.arch.amdgpu.descriptors import amdgpu_descriptors
from loom.gen.target.low.compiled import DescriptorIdentityDomain
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_SOPP_OPCODE_INFO_RDNA,
    AmdgpuDescriptorSetInfo,
    AmdgpuDescriptorSetIsaInfo,
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
        inline_schedule_model_key="amdgpu.generic",
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
        isa_infos=(
            AmdgpuDescriptorSetIsaInfo(
                isa_xml_key="test",
                isa_architecture_name="AMDGPU Test",
                isa_architecture_id=1,
                sopp_opcodes=AMDGPU_SOPP_OPCODE_INFO_RDNA,
            ),
        ),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        storage_generator_target=storage_target,
    )


def test_exact_storage_families_include_portable_source_views() -> None:
    rdna3_family = amdgpu_target_table_family("rdna3")
    rdna3_5_family = amdgpu_target_table_family("rdna3_5")
    cdna3_family = amdgpu_target_table_family("cdna3")
    cdna4_family = amdgpu_target_table_family("cdna4")

    assert [info.generator_target for info in rdna3_family.view_infos] == ["gfx11_generic"]
    assert rdna3_family.representation_infos == ()
    assert rdna3_5_family.view_infos == ()
    assert [info.generator_target for info in rdna3_5_family.representation_infos] == ["gfx11_generic"]
    assert [info.generator_target for info in cdna3_family.view_infos] == ["gfx9_4_generic"]
    assert cdna3_family.representation_infos == ()
    assert cdna4_family.view_infos == ()
    assert [info.generator_target for info in cdna4_family.representation_infos] == ["gfx9_4_generic"]


def test_alternate_portable_view_uses_storage_qualified_provider() -> None:
    storage_info = _descriptor_set_info("test_storage")
    representation_info = _descriptor_set_info("test_portable")
    family = AmdgpuTargetTableFamily(
        storage_info=storage_info,
        view_infos=(),
        representation_infos=(representation_info,),
    )
    storage_set = replace(_descriptor_set("test_storage", 2), descriptor_set_ordinal=7)
    representation_set = replace(_descriptor_set("test_portable", 1), descriptor_set_ordinal=2)
    generated = SimpleNamespace(source="// shared\n", view_headers=("// storage\n",))
    expected_identity_domain = DescriptorIdentityDomain(keys=("amdgpu.test0", "amdgpu.test1"))

    with (
        mock.patch.object(
            amdgpu_descriptors,
            "amdgpu_descriptor_identity_domain",
            return_value=expected_identity_domain,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "generate_descriptor_set_family",
            return_value=generated,
        ) as generate_descriptor_set_family,
    ):
        assert (
            amdgpu_descriptors.generate_amdgpu_descriptor_table_family(
                family,
                {
                    "test_storage": storage_set,
                    "test_portable": representation_set,
                },
                source_public_header=storage_set.public_header,
            )
            is generated
        )

    generated_views = generate_descriptor_set_family.call_args.args[1]
    assert generated_views[0] is storage_set
    assert generated_views[1].function_name == ("loom_amdgpu_test_portable_core_descriptor_set_from_test_storage_storage")
    assert generated_views[1].descriptor_set_ordinal == 7
    assert generate_descriptor_set_family.call_args.kwargs["identity_domain"] is expected_identity_domain


def test_storage_generation_reuses_parsed_isa_for_declared_views() -> None:
    storage_target = "test_storage"
    view_target = "test_view"
    storage_info = _descriptor_set_info(storage_target)
    view_info = _descriptor_set_info(view_target, storage_target=storage_target)
    parsed_spec = object()
    parse_calls: list[Path] = []
    build_calls: list[tuple[tuple[str, ...], object]] = []
    expected_identity_domain = DescriptorIdentityDomain(keys=("amdgpu.test0", "amdgpu.test1"))

    def parse_xml(paths: dict[str, Path], instruction_names: dict[str, tuple[str, ...]]) -> dict[str, object]:
        assert instruction_names == {"test": ("TEST",)}
        parse_calls.extend(paths.values())
        return {"test": parsed_spec}

    def build_descriptor_sets(targets: tuple[str, ...], specs: dict[str, object]) -> dict[str, DescriptorSet]:
        build_calls.append((targets, specs))
        return {
            target: _descriptor_set(
                target,
                2 if target == view_target else 1,
            )
            for target in targets
        }

    def generate_descriptor_set_family(
        storage_descriptor_set: DescriptorSet,
        view_descriptor_sets: tuple[DescriptorSet, ...],
        *,
        primary_strings=None,
        shared_strings=None,
        identity_domain=None,
    ) -> SimpleNamespace:
        assert primary_strings is None
        assert shared_strings is None
        assert identity_domain is expected_identity_domain
        return SimpleNamespace(
            source="// shared\n",
            view_headers=tuple(f"// {descriptor_set.key}\n" for descriptor_set in view_descriptor_sets),
        )

    with (
        TemporaryDirectory() as temporary_directory,
        mock.patch.object(
            amdgpu_descriptors,
            "AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS",
            (storage_target, view_target),
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "amdgpu_target_table_family",
            return_value=AmdgpuTargetTableFamily(
                storage_info=storage_info,
                view_infos=(view_info,),
            ),
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "amdgpu_core_descriptor_set_instruction_names_by_isa_key",
            return_value={"test": ("TEST",)},
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "parse_amdgpu_isa_xml_paths_for_instructions",
            parse_xml,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "build_amdgpu_core_descriptor_sets_from_specs",
            build_descriptor_sets,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "generate_descriptor_set_family",
            generate_descriptor_set_family,
        ),
        mock.patch.object(
            amdgpu_descriptors,
            "amdgpu_descriptor_identity_domain",
            return_value=expected_identity_domain,
        ),
    ):
        tmp_path = Path(temporary_directory)
        xml_path = tmp_path / "amdgpu_isa_test.xml"
        view_header_path = tmp_path / "test_view_descriptors.h"
        assert (
            amdgpu_descriptors.main(
                [
                    f"--target={storage_target}",
                    f"--isa-xml=test:{xml_path}",
                    f"--header={tmp_path / 'test_storage_descriptors.h'}",
                    f"--source={tmp_path / 'test_storage_descriptors.c'}",
                    f"--view-header={view_target}={view_header_path}",
                ]
            )
            == 0
        )

    assert parse_calls == [xml_path]
    assert build_calls == [
        ((storage_target, view_target), {"test": parsed_spec}),
    ]
