# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from loom.builders import DialectBuilder

class TargetBuilder(DialectBuilder):
    def generic(
        self,
        *,
        kind: str,
        symbol: str,
        codegen_format: str | None = ...,
        artifact_format: str | None = ...,
        default_pointer_bitwidth: int | None = ...,
        index_bitwidth: int | None = ...,
        offset_bitwidth: int | None = ...,
        max_workgroup_size_x: int | None = ...,
        max_workgroup_size_y: int | None = ...,
        max_workgroup_size_z: int | None = ...,
        max_flat_workgroup_size: int | None = ...,
        subgroup_size: int | None = ...,
        max_grid_size_x: int | None = ...,
        max_grid_size_y: int | None = ...,
        max_grid_size_z: int | None = ...,
        max_flat_grid_size: int | None = ...,
        max_workgroup_count_x: int | None = ...,
        max_workgroup_count_y: int | None = ...,
        max_workgroup_count_z: int | None = ...,
        memory_space_generic: int | None = ...,
        memory_space_global: int | None = ...,
        memory_space_workgroup: int | None = ...,
        memory_space_constant: int | None = ...,
        memory_space_private: int | None = ...,
        memory_space_host: int | None = ...,
        memory_space_descriptor: int | None = ...,
        abi: str | None = ...,
        export_symbol: str | None = ...,
        linkage: str | None = ...,
        hal_buffer_resource_flags: int | None = ...,
        contract_set_key: str | None = ...,
        contract_feature_bits: int | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
