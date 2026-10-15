# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Semantic and readability contracts for Loom C++ rendering."""

import pytest

from loom.format.cxx.registry import load_module
from loom.format.cxx.renderer import CxxRenderError, render_module
from loom.format.cxx.test_support import LOOP_MODULE, SOURCE_MODULE
from loom.ir import Module


def _module_with_float_comparison(predicate: str) -> Module:
    source = SOURCE_MODULE.replace(
        "  %value = view.load %slice[0] : view<1xf32> -> f32\n",
        "  %value = view.load %slice[0] : view<1xf32> -> f32\n"
        "  %limit = scalar.constant 0.0 : f32\n"
        f"  %comparison = scalar.cmpf {predicate}, %value, %limit : f32\n",
    )
    return load_module(source.encode())


def test_projects_dynamic_subview_with_index_identity() -> None:
    result = render_module(load_module(SOURCE_MODULE.encode()))

    assert (
        result
        == """\
#include <loomcxx/kernel.h>

[[loom::kernel, loom::workgroup_count(1, 1, 1), loom::workgroup_size(1, 1, 1)]]
void copy_one(const float* input, float* output) {
  auto input_view = loom::buffer::view<4>(input, {}, loom::encoding::layout::dense<1>());
  unsigned workitem = static_cast<unsigned>(loom::kernel::workitem::id.x);
  loom::type::index v43 = loom::index::cast(workitem);
  auto slice = loom::view::subview<1>(input_view, loom::type::indices{v43}, {});
  float value = loom::view::load(slice, 0);
  auto output_view = loom::buffer::view<4>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(value, output_view, v43);
}
"""
    )
    assert "v42" not in result
    assert "unsigned long long" not in result
    assert "9223372036854775807" not in result


def test_projects_multiple_kernels_in_one_translation_unit() -> None:
    result = render_module(load_module((SOURCE_MODULE + "\n" + LOOP_MODULE).encode()))

    assert result.count("#include <loomcxx/kernel.h>") == 1
    assert result.index("void copy_one(") < result.index("void accumulate()")
    assert "}\n\n[[loom::kernel" in result
    assert "output_1" not in result


def test_projects_distinct_kernel_targets_in_one_translation_unit() -> None:
    source = (
        "amdgpu.target<gfx1100> @first_target\n"
        "amdgpu.target<gfx942> @second_target\n"
        + SOURCE_MODULE.replace(
            "kernel.def @copy_one()", "kernel.def target(@first_target) @copy_one()"
        )
        + "\n"
        + LOOP_MODULE.replace(
            "kernel.def @accumulate()",
            "kernel.def target(@second_target) @accumulate()",
        )
    )

    result = render_module(load_module(source.encode()))

    assert result.count("#include <loomcxx/target/amdgpu.h>") == 1
    assert (
        'constexpr loom::amdgpu::target rendered_target_first_target{.kind = "gfx1100"};'
        in result
    )
    assert (
        'constexpr loom::amdgpu::target rendered_target_second_target{.kind = "gfx942"};'
        in result
    )
    assert "loom::target(rendered_target_first_target)" in result
    assert "loom::target(rendered_target_second_target)" in result


def test_preserves_scalar_kernel_arguments() -> None:
    source = SOURCE_MODULE.replace(
        "} launch(%input: buffer, %output: buffer) {",
        "} launch(%element_count: i32, %input: buffer, %output: buffer) {",
    )

    result = render_module(load_module(source.encode()))

    assert "unsigned element_count," in result


def test_projects_explicit_dense_layout_at_shaped_use() -> None:
    source = SOURCE_MODULE.replace(
        "  %input_view = buffer.view %input[%zero] : buffer -> view<4xf32>\n",
        "  %input_layout = encoding.layout.dense : encoding<layout>\n"
        "  %input_view = buffer.view %input[%zero] : buffer -> "
        "view<4xf32, %input_layout>\n",
    ).replace(
        "  %slice = view.subview %input_view[%43] : view<4xf32> -> view<1xf32>\n"
        "  %value = view.load %slice[0] : view<1xf32> -> f32\n",
        "  %slice = view.subview %input_view[%43] : "
        "view<4xf32, %input_layout> -> view<1xf32, %input_layout>\n"
        "  %value = view.load %slice[0] : view<1xf32, %input_layout> -> f32\n",
    )

    result = render_module(load_module(source.encode()))

    assert (
        "loom::buffer::view<4>(input, {}, loom::encoding::layout::dense<1>())" in result
    )


def test_projects_explicit_strided_layout_values() -> None:
    source = SOURCE_MODULE.replace(
        "  %input_view = buffer.view %input[%zero] : buffer -> view<4xf32>\n",
        "  %row_stride = index.constant 8 : index\n"
        "  %input_layout = encoding.layout.strided [%row_stride] : "
        "encoding<layout>\n"
        "  %input_view = buffer.view %input[%zero] : buffer -> "
        "view<4xf32, %input_layout>\n",
    ).replace(
        "  %slice = view.subview %input_view[%43] : view<4xf32> -> view<1xf32>\n"
        "  %value = view.load %slice[0] : view<1xf32> -> f32\n",
        "  %slice = view.subview %input_view[%43] : "
        "view<4xf32, %input_layout> -> view<1xf32, %input_layout>\n"
        "  %value = view.load %slice[0] : view<1xf32, %input_layout> -> f32\n",
    )

    result = render_module(load_module(source.encode()))

    assert (
        "auto input_layout = loom::encoding::layout::strided(loom::type::index{8});"
    ) in result
    assert "loom::buffer::view<4>(input, {}, input_layout)" in result


def test_projects_exposed_kernel_topology_family() -> None:
    source = SOURCE_MODULE.replace(
        "  %zero = index.constant 0 : offset\n",
        "  %zero = index.constant 0 : offset\n"
        "  %workgroup_count_index = kernel.workgroup.count<x> : index\n"
        "  %workgroup_count = index.cast %workgroup_count_index : index to i32\n"
        "  %workgroup_size_index = kernel.workgroup.size<y> : index\n"
        "  %workgroup_size = index.cast %workgroup_size_index : index to i32\n"
        "  %subgroup_count_index = kernel.subgroup.count : index\n"
        "  %subgroup_count = index.cast %subgroup_count_index : index to i32\n"
        "  %subgroup_size_index = kernel.subgroup.size : index\n"
        "  %subgroup_size = index.cast %subgroup_size_index : index to i32\n"
        "  %lane_index = kernel.subgroup.lane.id : index\n"
        "  %lane = index.cast %lane_index : index to i32\n",
    )

    result = render_module(load_module(source.encode()))

    assert (
        "unsigned workgroup_count = "
        "static_cast<unsigned>(loom::kernel::workgroup::count.x);"
    ) in result
    assert (
        "unsigned workgroup_size = "
        "static_cast<unsigned>(loom::kernel::workgroup::size.y);"
    ) in result
    assert ("unsigned subgroup_count = loom::kernel::subgroup::count();") in result
    assert ("unsigned subgroup_size = loom::kernel::subgroup::size();") in result
    assert "unsigned lane = loom::kernel::subgroup::lane_id();" in result


def test_projects_loop_carried_ssa_as_mutable_cxx_state() -> None:
    result = render_module(load_module(LOOP_MODULE.encode()))

    assert "float sum = 0.0f;" in result
    assert (
        "[[loom::unroll(loom::type::index{4}), "
        "loom::pipeline(loom::type::index{2}), "
        'loom::schedule("recurrence")]]' in result
    )
    assert "for (unsigned index = 0u; index < 4u; ++index) {" in result
    assert "sum = sum + 1.0f;" in result
    assert "next" not in result


def test_stages_vector_insert_before_replacing_a_distinct_loop_carrier() -> None:
    source = """\
kernel.def @replace_lane() {
  %count = index.constant 1 : index
  kernel.launch.config workgroups(%count, %count, %count) workgroup_size(%count, %count, %count) : index
} launch() {
  %zero = scalar.constant 0 : i32
  %one = scalar.constant 1 : i32
  %initial = vector.splat %zero : vector<4xi32>
  %source = vector.splat %one : vector<4xi32>
  %lower = index.constant 0 : offset
  %upper = index.constant 4 : offset
  %step = index.constant 1 : offset
  %result = scf.for %index = [%lower to %upper step %step](%carried = %initial : vector<4xi32>) -> (vector<4xi32>) {
    %lane = vector.extract %carried[0] : vector<4xi32> -> i32
    %next = vector.insert %lane into %source[1] : i32, vector<4xi32>
    scf.yield %next : vector<4xi32>
  }
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "unsigned lane = result[0];" in result
    assert "auto next_next = source;" in result
    assert "next_next[1] = lane;" in result
    assert "result = next_next;" in result


def test_projects_while_regions_as_guarded_cxx_state() -> None:
    source = """\
kernel.def @bounded_loop() {
  %count = index.constant 1 : index
  kernel.launch.config workgroups(%count, %count, %count) workgroup_size(%count, %count, %count) : index
} launch() {
  %begin = index.constant 0 : index
  %end = index.constant 4 : index
  %step = index.constant 1 : index
  %terminal = scf.while(%before = %begin : index) -> (index) {
    %more = index.cmp ult, %before, %end : index
    scf.condition %more, %before : i1, index
  } do(%position: index) {
    %next = index.add %position, %step : index
    scf.yield %next : index
  }
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "loom::type::index terminal = loom::type::index{0};" in result
    assert "bool loop_continues = true;" in result
    assert "while (loop_continues) {" in result
    assert "bool more = loom::index::ult(terminal, loom::type::index{4});" in result
    assert "loop_continues = more;" in result
    assert "if (loop_continues) {" in result
    assert "terminal = terminal + loom::type::index{1};" in result


def test_projects_vector_absolute_value_as_typed_intrinsic() -> None:
    source = SOURCE_MODULE.replace(
        "  %value = view.load %slice[0] : view<1xf32> -> f32\n",
        "  %loaded = view.load %slice[0] : view<1xf32> -> f32\n"
        "  %values = vector.splat %loaded : vector<4xf32>\n"
        "  %absolute = vector.absf %values : vector<4xf32>\n"
        "  %value = vector.extract %absolute[0] : vector<4xf32> -> f32\n",
    )

    result = render_module(load_module(source.encode()))

    assert "#include <loomcxx/vector.h>" in result
    assert "Float32x4 absolute = loom::vector::absf(values);" in result


def test_projects_static_encoding_through_the_public_schema_facade() -> None:
    source = SOURCE_MODULE.replace(
        "  %zero = index.constant 0 : offset\n",
        "  %schema = encoding.define #encoding.f4e2m1<"
        "affine=scale_only, payload_elements=64, "
        "payload_packing=little_endian_nibbles, payload_registers=8, "
        "scale_format=e8m0, scale_group_elements=32, scale_operands=1, "
        "scale_topology=block_1d, zero_scale_fallback=true> : "
        "encoding<schema>\n"
        "  %zero = index.constant 0 : offset\n",
    )

    result = render_module(load_module(source.encode()))

    assert "#include <loomcxx/encoding.h>" in result
    assert (
        "auto schema = loom::encoding::define<loom::encoding::f4e2m1{"
        ".payload_elements = 64}>();" in result
    )


def test_projects_named_decode_auxiliary_through_a_record() -> None:
    source = """\
kernel.def @decode() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  %zero = scalar.constant 0 : i32
  %payload = vector.splat %zero : vector<4xi32>
  %scale = vector.splat %zero : vector<1xi32>
  %schema = encoding.define #encoding.f4e2m1<affine=scale_only, payload_elements=32, payload_packing=little_endian_nibbles, payload_registers=4, scale_format=e8m0, scale_group_elements=32, scale_operands=1, scale_topology=block_1d, zero_scale_fallback=true> : encoding<schema>
  %decoded = vector.decode %payload using %schema {scale = %scale : vector<1xi32>} : vector<4xi32>, encoding<schema> -> vector<32xbf16>
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "struct DecodeAuxiliary {\n  UInt32x1 scale;\n};" in result
    assert (
        "BFloat16x32 decoded = loom::vector::decode<BFloat16x32>("
        "payload, schema, DecodeAuxiliary{scale});" in result
    )


def test_projects_register_slices_with_address_domain_offsets() -> None:
    source = """\
kernel.def @slice() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%offset: index) {
  %zero = scalar.constant 0.0 : f32
  %source = vector.splat %zero : vector<16xf32>
  %dynamic = vector.slice %source[%offset] : vector<16xf32> -> vector<4xf32>
  %static = vector.slice %source[4] : vector<16xf32> -> vector<4xf32>
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert (
        "Float32x4 dynamic = loom::vector::slice<Float32x4>(source, offset);" in result
    )
    assert (
        "Float32x4 static_value = loom::vector::slice<Float32x4>("
        "source, loom::type::index{4});" in result
    )


def test_projects_matrix_fragment_roles_shapes_parameters_and_mma() -> None:
    source = """\
kernel.def @matrix() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%m: index, %n: index, %k: index) {
  %zero = scalar.constant 0.0 : f32
  %lhs_data = vector.splat %zero : vector<4xf32>
  %rhs_data = vector.splat %zero : vector<4xf32>
  %init_data = vector.splat %zero : vector<4xf32>
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] : vector<4xf32>
  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] : vector<4xf32>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<4xf32>
  %result = vector.mma %lhs, %rhs, %init : vector<4xf32>, vector<4xf32>, vector<4xf32>
  %zero_word = scalar.constant 0 : i32
  %payload = vector.splat %zero_word : vector<8xi32>
  %scale = vector.splat %zero_word : vector<2xi32>
  %schema = encoding.define #encoding.f4e2m1<affine=scale_only, payload_elements=64, payload_packing=target_fragment, payload_registers=8, scale_format=f8e4m3, scale_group_elements=16, scale_operands=1, scale_topology=block_1d, zero_scale_fallback=true> : encoding<schema>
  %encoded_lhs = vector.fragment<lhs> %payload shape [%m, %k] using {scale = %scale : vector<2xi32>, schema = %schema : encoding<schema>} : vector<8xi32>
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert (
        "struct FragmentParameters {\n"
        "  UInt32x2 scale;\n"
        "  loom::type::encoding<loom::encoding::role::schema> schema;\n"
        "};" in result
    )
    assert (
        "Float32x4 lhs = loom::vector::fragment::attach<"
        "loom::vector::fragment::role::lhs>(lhs_data, "
        "loom::vector::fragment::shape{m, k});" in result
    )
    assert "Float32x4 result = loom::vector::mma(lhs, rhs, init);" in result
    assert (
        "UInt32x8 encoded_lhs = loom::vector::fragment::attach<"
        "loom::vector::fragment::role::lhs>(payload, "
        "loom::vector::fragment::shape{m, k}, "
        "FragmentParameters{scale, schema});" in result
    )


def test_projects_scoped_buffer_alignment_without_strengthening_the_parameter() -> None:
    source = """\
kernel.def @scoped_alignment() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%active: i1, %input: buffer) {
  scf.if %active {
    %aligned = buffer.assume.alignment %input {minimum_alignment = 16} : buffer
  }
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "[[loom::assume_aligned" not in result
    assert "if (active) {" in result
    assert "auto aligned = loom::buffer::assume_alignment<16>(input);" in result


def test_projects_buffer_alias_contracts_at_their_lexical_scope() -> None:
    source = """\
kernel.def @aliases() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%active: i1, %input: buffer, %output: buffer) {
  %input_unique, %output_unique = buffer.assume.noalias %input, %output : buffer, buffer
  scf.if %active {
    %scoped_input, %scoped_output = buffer.assume.noalias %input_unique, %output_unique : buffer, buffer
  }
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "[[loom::noalias]] const unsigned char* input_unique" in result
    assert "[[loom::noalias]] const unsigned char* output_unique" in result
    assert "auto scoped_input = loom::buffer::assume_noalias(input_unique);" in result
    assert "auto scoped_output = loom::buffer::assume_noalias(output_unique);" in result


def test_projects_atomic_memory_operations_and_fences() -> None:
    source = """\
kernel.def @atomics() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%values: buffer) {
  %zero = index.constant 0 : offset
  %values_view = buffer.view %values[%zero] : buffer -> view<2x2xi32>
  %replacement = scalar.constant 1 : i32
  buffer.fence scope(device) ordering(release)
  view.atomic.store %replacement, %values_view[0, 0] {ordering = relaxed, scope = device} : i32, view<2x2xi32>
  %observed = view.atomic.load %values_view[0, 1] {ordering = acquire, scope = device} : view<2x2xi32> -> i32
  %old = view.atomic.rmw<addi> %replacement, %values_view[1, 0] {ordering = acq_rel, scope = device} : i32, view<2x2xi32> -> i32
  view.atomic.reduce<maxui> %old, %values_view[1, 1] {ordering = relaxed, scope = device} : i32, view<2x2xi32>
  %compared = view.atomic.cmpxchg %observed, %old, %values_view[0, 0] {failure_ordering = acquire, scope = device, success_ordering = acq_rel} : i32, view<2x2xi32> -> i32
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "#include <loomcxx/atomic.h>" in result
    assert (
        "loom::buffer::fence<loom::atomic::ordering::release, "
        "loom::atomic::scope::device>();" in result
    )
    assert (
        "loom::view::atomic::store<loom::atomic::ordering::relaxed, "
        "loom::atomic::scope::device>(1u, values_view, 0, 0);" in result
    )
    assert (
        "unsigned observed = loom::view::atomic::load<"
        "loom::atomic::ordering::acquire, loom::atomic::scope::device>("
        "values_view, 0, 1);" in result
    )
    assert (
        "unsigned old = loom::view::atomic::rmw<loom::atomic::kind::addi, "
        "loom::atomic::ordering::acq_rel, loom::atomic::scope::device>(1u, "
        "values_view, 1, 0);" in result
    )
    assert (
        "loom::view::atomic::reduce<loom::atomic::kind::maxui, "
        "loom::atomic::ordering::relaxed, loom::atomic::scope::device>(old, "
        "values_view, 1, 1);" in result
    )
    assert (
        "(void)loom::view::atomic::cmpxchg<"
        "loom::atomic::ordering::acq_rel, loom::atomic::ordering::acquire, "
        "loom::atomic::scope::device>(observed, old, values_view, 0, 0);" in result
    )


def test_projects_index_arithmetic_without_scalarizing_its_domain() -> None:
    source = SOURCE_MODULE.replace(
        "  %42 = scalar.extui %workitem : i32 to i64\n"
        "  %bounded = scalar.assume %42 [range(%42, 0, 9223372036854775807)] : i64\n"
        "  %43 = index.cast %bounded : i64 to index\n",
        "  %one = index.constant 1 : index\n"
        "  %43 = index.mul %workitem_index, %one : index\n",
    )
    result = render_module(load_module(source.encode()))

    assert (
        "loom::type::index v43 = static_cast<loom::type::index>(workitem_index) * "
        "loom::type::index{1};"
    ) in result
    assert "loom::type::indices{v43}" in result


def test_projects_topology_coordinates_without_losing_index_identity() -> None:
    source = """\
kernel.def @topology_identity() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%output: buffer) {
  %zero = index.constant 0 : offset
  %output_view = buffer.view %output[%zero] : buffer -> view<64xi32>
  %coordinate = kernel.workitem.id<x> : index
  %coordinate$1 = index.assume %coordinate [range(%coordinate, 0, 62)] : index
  %one = index.constant 1 : index
  %shifted = index.add %coordinate$1, %one : index
  %lane = index.cast %coordinate$1 : index to i32
  %wide = index.cast %coordinate$1 : index to i64
  view.store %lane, %output_view[%coordinate$1] : i32, view<64xi32>
  view.store %lane, %output_view[%shifted] : i32, view<64xi32>
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert (
        result
        == """\
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>

[[loom::kernel, loom::workgroup_count(1, 1, 1), loom::workgroup_size(1, 1, 1)]]
void topology_identity(unsigned* output) {
  auto output_view = loom::buffer::view<64>(output, {}, loom::encoding::layout::dense<1>());
  auto coordinate = loom::kernel::workitem::id.x;
  loom::assume(loom::predicate::range(coordinate, 0, 62));
  loom::type::index shifted = static_cast<loom::type::index>(coordinate) + loom::type::index{1};
  unsigned lane = static_cast<unsigned>(coordinate);
  unsigned long long wide = loom::index::cast<unsigned long long>(static_cast<loom::type::index>(coordinate));
  loom::view::store(lane, output_view, coordinate);
  loom::view::store(lane, output_view, shifted);
}
"""
    )


def test_preserves_unsigned_extension_when_folding_address_constants() -> None:
    source = SOURCE_MODULE.replace(
        "  %workitem_index = kernel.workitem.id<x> : index\n"
        "  %workitem = index.cast %workitem_index : index to i32\n"
        "  %42 = scalar.extui %workitem : i32 to i64\n"
        "  %bounded = scalar.assume %42 [range(%42, 0, 9223372036854775807)] : i64\n"
        "  %43 = index.cast %bounded : i64 to index\n",
        "  %negative = scalar.constant -1 : i32\n"
        "  %wide = scalar.extui %negative : i32 to i64\n"
        "  %43 = index.cast %wide : i64 to index\n",
    )

    result = render_module(load_module(source.encode()))

    assert "loom::type::index{4294967295}" in result
    assert "loom::type::index{-1}" not in result


def test_preserves_unsigned_extension_of_a_signed_runtime_value() -> None:
    source = SOURCE_MODULE.replace(
        "} launch(%input: buffer, %output: buffer) {",
        "} launch(%signed_input: i32, %input: buffer, %output: buffer) {",
    ).replace(
        "  %workitem_index = kernel.workitem.id<x> : index\n"
        "  %workitem = index.cast %workitem_index : index to i32\n"
        "  %42 = scalar.extui %workitem : i32 to i64\n"
        "  %bounded = scalar.assume %42 [range(%42, 0, 9223372036854775807)] : i64\n"
        "  %43 = index.cast %bounded : i64 to index\n",
        "  %signed_wide = scalar.extsi %signed_input : i32 to i64\n"
        "  %unsigned_wide = scalar.extui %signed_input : i32 to i64\n"
        "  %43 = index.cast %unsigned_wide : i64 to index\n",
    )

    result = render_module(load_module(source.encode()))

    assert "int signed_input" in result
    assert (
        "static_cast<unsigned long long>(static_cast<unsigned>(signed_input))" in result
    )


def test_projects_address_operation_forms() -> None:
    source = """\
kernel.def @address_math() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%lhs: index, %rhs: index, %amount: index, %stride: offset, %physical: offset) {
  %sum = index.add %lhs, %rhs : index
  %difference = index.sub %lhs, %rhs : index
  %product = index.mul %lhs, %rhs : index
  %quotient = index.div %lhs, %rhs : index
  %remainder = index.rem %lhs, %rhs : index
  %bit_and = index.andi %lhs, %rhs : index
  %bit_or = index.ori %lhs, %rhs : index
  %bit_xor = index.xori %lhs, %rhs : index
  %shift_left = index.shli %lhs, %amount : index
  %shift_right = index.shrsi %lhs, %amount : index
  %unsigned_shift = index.shrui %lhs, %amount : index
  %minimum = index.min %lhs, %rhs : index
  %maximum = index.max %lhs, %rhs : index
  %multiply_add = index.madd %lhs, %rhs, %amount : index
  %rotate_left = index.rotli %lhs, %amount : index
  %rotate_right = index.rotri %lhs, %amount : index
  %leading_zeros = index.ctlzi %lhs : index
  %trailing_zeros = index.cttzi %lhs : index
  %population = index.ctpopi %lhs : index
  %combined_quotient, %combined_remainder = index.divrem %lhs, %rhs : index
  %bytes = index.scale %lhs, %stride : index, offset -> offset
  %offset_sum = index.add %physical, %bytes : offset
  %offset_difference = index.sub %physical, %bytes : offset
  %logical = index.cast %physical : offset to index
  %physical_again = index.cast %lhs : index to offset
  %logical_fixed = index.cast %lhs : index to i32
  %physical_fixed = index.cast %physical : offset to i64
  %required_lhs = index.constexpr %lhs : index
  %required_physical = index.constexpr %physical : offset
  %signed_less = index.cmp slt, %lhs, %rhs : index
  %unsigned_less = index.cmp ult, %lhs, %rhs : index
  %offset_less = index.cmp ult, %physical, %bytes : offset
  %offset_signed_less = index.cmp slt, %physical, %bytes : offset
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert (
        result
        == """\
#include <loomcxx/kernel.h>

[[loom::kernel, loom::workgroup_count(1, 1, 1), loom::workgroup_size(1, 1, 1)]]
void address_math(loom::type::index lhs, loom::type::index rhs, loom::type::index amount, loom::type::offset stride, loom::type::offset physical) {
  loom::type::index sum = lhs + rhs;
  loom::type::index difference = lhs - rhs;
  loom::type::index product = lhs * rhs;
  loom::type::index quotient = lhs / rhs;
  loom::type::index remainder = lhs % rhs;
  loom::type::index bit_and = lhs & rhs;
  loom::type::index bit_or = lhs | rhs;
  loom::type::index bit_xor = lhs ^ rhs;
  loom::type::index shift_left = lhs << amount;
  loom::type::index shift_right = lhs >> amount;
  loom::type::index unsigned_shift = loom::index::shrui(lhs, amount);
  loom::type::index minimum = loom::index::min(lhs, rhs);
  loom::type::index maximum = loom::index::max(lhs, rhs);
  loom::type::index multiply_add = loom::index::madd(lhs, rhs, amount);
  loom::type::index rotate_left = loom::index::rotli(lhs, amount);
  loom::type::index rotate_right = loom::index::rotri(lhs, amount);
  loom::type::index leading_zeros = loom::index::ctlzi(lhs);
  loom::type::index trailing_zeros = loom::index::cttzi(lhs);
  loom::type::index population = loom::index::ctpopi(lhs);
  auto quotient_remainder = loom::index::divrem(lhs, rhs);
  loom::type::offset bytes = loom::index::scale(lhs, stride);
  loom::type::offset offset_sum = physical + bytes;
  loom::type::offset offset_difference = physical - bytes;
  loom::type::index logical = loom::index::cast(physical);
  loom::type::offset physical_again = loom::offset::cast(lhs);
  unsigned logical_fixed = loom::index::cast<unsigned>(lhs);
  unsigned long long physical_fixed = loom::offset::cast<unsigned long long>(physical);
  [[loom::constexpr]] const loom::type::index required_lhs = lhs;
  [[loom::constexpr]] const loom::type::offset required_physical = physical;
  bool signed_less = lhs < rhs;
  bool unsigned_less = loom::index::ult(lhs, rhs);
  bool offset_less = physical < bytes;
  bool offset_signed_less = loom::offset::slt(physical, bytes);
}
"""
    )


def test_projects_byte_and_logical_scratch_extents() -> None:
    source = """\
kernel.def @scratch() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  %bytes = index.constant 128 : offset
  %raw = buffer.alloca<workgroup> align(16) %bytes : buffer
  %count = index.constant 32 : index
  %stride = index.constant 1 : offset
  %scaled = index.scale %count, %stride : index, offset -> offset
  %logical = buffer.alloca<private> align(4) %scaled : buffer
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert (
        "auto* raw = loom::buffer::alloca<loom::memory_space::workgroup, "
        "16>(loom::type::offset{128});"
    ) in result
    assert (
        "auto* logical = loom::buffer::alloca<unsigned char, "
        "loom::memory_space::private_, 4>(loom::type::index{32});"
    ) in result


@pytest.mark.parametrize(
    ("domain", "predicate", "expression"),
    [
        ("index", "eq", "lhs == rhs"),
        ("index", "ne", "lhs != rhs"),
        ("index", "slt", "lhs < rhs"),
        ("index", "sle", "lhs <= rhs"),
        ("index", "sgt", "lhs > rhs"),
        ("index", "sge", "lhs >= rhs"),
        ("index", "ult", "loom::index::ult(lhs, rhs)"),
        ("index", "ule", "loom::index::ule(lhs, rhs)"),
        ("index", "ugt", "loom::index::ugt(lhs, rhs)"),
        ("index", "uge", "loom::index::uge(lhs, rhs)"),
        ("offset", "eq", "lhs == rhs"),
        ("offset", "ne", "lhs != rhs"),
        ("offset", "ult", "lhs < rhs"),
        ("offset", "ule", "lhs <= rhs"),
        ("offset", "ugt", "lhs > rhs"),
        ("offset", "uge", "lhs >= rhs"),
        ("offset", "slt", "loom::offset::slt(lhs, rhs)"),
        ("offset", "sle", "loom::offset::sle(lhs, rhs)"),
        ("offset", "sgt", "loom::offset::sgt(lhs, rhs)"),
        ("offset", "sge", "loom::offset::sge(lhs, rhs)"),
    ],
)
def test_projects_every_address_comparison(
    domain: str, predicate: str, expression: str
) -> None:
    source = f"""\
kernel.def @compare() {{
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
}} launch(%lhs: {domain}, %rhs: {domain}) {{
  %comparison = index.cmp {predicate}, %lhs, %rhs : {domain}
  kernel.return
}}
"""

    result = render_module(load_module(source.encode()))

    assert f"bool comparison = {expression};" in result


def test_projects_address_assumptions_with_plain_constant_bounds() -> None:
    source = """\
kernel.def @bounded() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%value: index, %bytes: offset) {
  %bounded_value = index.assume %value [range(%value, 0, 15)] : index
  %bounded_bytes = index.assume %bytes [range(%bytes, 0, 255)] : offset
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "#include <loomcxx/predicate.h>" in result
    assert "loom::assume(loom::predicate::range(value, 0, 15));" in result
    assert "loom::assume(loom::predicate::range(bytes, 0, 255));" in result


def test_projects_floating_point_permissions_on_the_exact_expression() -> None:
    source = SOURCE_MODULE.replace(
        "  %value = view.load %slice[0] : view<1xf32> -> f32\n",
        "  %loaded = view.load %slice[0] : view<1xf32> -> f32\n"
        "  %value = scalar.addf<fast> %loaded, %loaded : f32\n",
    )
    result = render_module(load_module(source.encode()))

    assert "[[loom::fastmath]] float value = loaded + loaded;" in result


def test_rejects_memory_policy_instead_of_dropping_it() -> None:
    source = SOURCE_MODULE.replace(
        "%value = view.load %slice[0] : view<1xf32> -> f32",
        "%value = view.load %slice[0] {cache_scope = device, "
        "cache_temporal = non_temporal} : view<1xf32> -> f32",
    )
    module = load_module(source.encode())

    with pytest.raises(
        CxxRenderError,
        match=(
            r"cannot project view.load: attributes have no exact C\+\+ projection: "
            "cache_scope, cache_temporal"
        ),
    ):
        render_module(module)


def test_projects_clustered_subgroup_reduction_contract() -> None:
    source = SOURCE_MODULE.replace(
        "%value = view.load %slice[0] : view<1xf32> -> f32",
        "%loaded = view.load %slice[0] : view<1xf32> -> f32\n"
        "  %value = kernel.subgroup.reduce<addf> %loaded "
        "{cluster_size = 4, cluster_stride = 2} : f32",
    )

    result = render_module(load_module(source.encode()))

    assert "loom::kernel::subgroup::reduce::addf<4, 2>(loaded)" in result


@pytest.mark.parametrize(
    ("predicate", "operator"),
    [
        ("oeq", "=="),
        ("ogt", ">"),
        ("oge", ">="),
        ("olt", "<"),
        ("ole", "<="),
        ("une", "!="),
    ],
)
def test_projects_exact_native_float_comparison_semantics(
    predicate: str, operator: str
) -> None:
    result = render_module(_module_with_float_comparison(predicate))

    assert f"bool comparison = value {operator} 0.0f;" in result


@pytest.mark.parametrize(
    "predicate",
    ["one", "ord", "ueq", "ugt", "uge", "ult", "ule", "uno"],
)
def test_rejects_float_comparisons_without_native_binary_operators(
    predicate: str,
) -> None:
    with pytest.raises(
        CxxRenderError,
        match=(
            rf"cannot project scalar.cmpf: floating comparison predicate "
            rf"'{predicate}' has no native C\+\+ binary operator"
        ),
    ):
        render_module(_module_with_float_comparison(predicate))


def test_rejects_top_level_definitions_that_would_be_omitted() -> None:
    module = load_module(
        ("func.def @helper() {\n  func.return\n}\n\n" + SOURCE_MODULE).encode()
    )

    with pytest.raises(
        CxxRenderError,
        match=(
            "cannot project func.def: a top-level definition would be omitted "
            r"from the C\+\+ projection"
        ),
    ):
        render_module(module)
