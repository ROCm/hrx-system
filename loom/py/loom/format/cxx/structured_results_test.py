# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete source projections of structured branch and loop results."""

import pytest

from loom.format.cxx.registry import load_module
from loom.format.cxx.renderer import render_module


@pytest.mark.parametrize(
    ("value_type", "source_type", "initializer", "preamble"),
    [
        ("i1", "bool", "{}", "\n"),
        ("i8", "unsigned char", "{}", "\n"),
        ("i16", "unsigned short", "{}", "\n"),
        ("i32", "unsigned", "{}", "\n"),
        ("i64", "unsigned long long", "{}", "\n"),
        ("f16", "std::float16_t", "{}", "\n#include <stdfloat>\n\n"),
        ("bf16", "std::bfloat16_t", "{}", "\n#include <stdfloat>\n\n"),
        ("f32", "float", "{}", "\n"),
        ("f64", "double", "{}", "\n"),
        (
            "f8E4M3",
            "loom::type::float8_e4m3fn_t",
            "{}",
            "#include <loomcxx/numeric.h>\n\n",
        ),
        (
            "f8E5M2",
            "loom::type::float8_e5m2_t",
            "{}",
            "#include <loomcxx/numeric.h>\n\n",
        ),
        ("index", "loom::type::index", "{0}", "#include <loomcxx/index.h>\n\n"),
        ("offset", "loom::type::offset", "{0}", "#include <loomcxx/index.h>\n\n"),
        (
            "vector<4xi32>",
            "UInt32x4",
            "{}",
            "\nusing UInt32x4 = unsigned __attribute__((ext_vector_type(4)));\n\n",
        ),
        (
            "vector<4xi1>",
            "Boolx4",
            "",
            "#include <loomcxx/vector.h>\n\n"
            "using Boolx4 = loom::type::vector<loom::type::shape<4>, bool>;\n\n",
        ),
        (
            "vector<2x2xi32>",
            "UInt32x2x2",
            "",
            "#include <loomcxx/vector.h>\n\n"
            "using UInt32x2x2 = loom::type::vector<loom::type::shape<2, 2>, unsigned>;\n\n",
        ),
    ],
)
def test_declares_numeric_results_before_assigning_each_branch(
    value_type: str, source_type: str, initializer: str, preamble: str
) -> None:
    source = f"""\
func.def public @choose(%condition: i1, %first: {value_type}, %second: {value_type}) -> ({value_type}) {{
  %result = scf.if %condition -> ({value_type}) {{
    scf.yield %first : {value_type}
  }} else {{
    scf.yield %second : {value_type}
  }}
  func.return %result : {value_type}
}}
"""

    result = render_module(load_module(source.encode()))

    assert (
        result
        == preamble
        + f"""\
[[loom::symbol("choose")]]
{source_type} choose(bool condition, {source_type} first, {source_type} second);

[[loom::symbol("choose")]]
{source_type} choose(bool condition, {source_type} first, {source_type} second) {{
  {source_type} result{initializer};
  if (condition) {{
    result = first;
  }} else {{
    result = second;
  }}
  return result;
}}
"""
    )


def test_projects_distinct_header_and_result_view_types() -> None:
    source = """\
kernel.def @widened_view() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%input: buffer, %output: buffer) {
  %zero = index.constant 0 : offset
  %small = buffer.view %input[%zero] : buffer -> view<4xi32>
  %large = buffer.view %input[%zero] : buffer -> view<8xi32>
  %result = scf.while(%before = %small : view<4xi32>) -> (view<8xi32>) {
    %active = view.load %before[0] : view<4xi32> -> i32
    %flag = scalar.trunci %active : i32 to i1
    scf.condition %flag, %large : i1, view<8xi32>
  } do(%after: view<8xi32>) {
    scf.yield %small : view<4xi32>
  }
  %value = view.load %result[0] : view<8xi32> -> i32
  %destination = buffer.view %output[%zero] : buffer -> view<1xi32>
  view.store %value, %destination[0] : i32, view<1xi32>
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert (
        result
        == """\
#include <loomcxx/kernel.h>

[[loom::kernel, loom::workgroup_count(1, 1, 1), loom::workgroup_size(1, 1, 1)]]
void widened_view(const unsigned* input, unsigned* output) {
  auto small = loom::buffer::view<4>(input, {});
  auto large = loom::buffer::view<8>(input, {});
  auto before = small;
  loom::type::view<loom::type::shape<8>, const unsigned> result;
  bool loop_continues = true;
  do {
    unsigned active = loom::view::load(before, 0);
    bool flag = (active & static_cast<unsigned>(1)) != static_cast<unsigned>(0);
    loop_continues = flag;
    result = large;
    if (loop_continues) {
      before = small;
    }
  } while (loop_continues);
  unsigned value = loom::view::load(result, 0);
  auto destination = loom::buffer::view<1>(output, {});
  loom::view::store(value, destination, 0);
}
"""
    )


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

    assert (
        result
        == """\
#include <loomcxx/kernel.h>

[[loom::kernel, loom::workgroup_count(1, 1, 1), loom::workgroup_size(1, 1, 1)]]
void bounded_loop() {
  loom::type::index terminal = loom::type::index{0};
  bool loop_continues = true;
  do {
    bool more = loom::index::ult(terminal, loom::type::index{4});
    loop_continues = more;
    if (loop_continues) {
      terminal = terminal + loom::type::index{1};
    }
  } while (loop_continues);
}
"""
    )


def test_projects_expression_only_while_header_as_natural_cxx() -> None:
    source = """\
kernel.def @bounded_loop() {
  %count = index.constant 1 : index
  kernel.launch.config workgroups(%count, %count, %count) workgroup_size(%count, %count, %count) : index
} launch() {
  %begin = index.constant 0 : index
  %terminal = scf.while(%before = %begin : index) -> (index) {
    %0 = index.constant 4 : index
    %1 = index.cmp slt, %before, %0 : index
    scf.condition %1, %before : i1, index
  } do(%position: index) {
    %2 = index.constant 1 : index
    %3 = index.add %position, %2 : index
    scf.yield %3 : index
  }
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert "while (terminal < loom::type::index{4}) {" in result
    assert "terminal = terminal + loom::type::index{1};" in result
    assert "loop_continues" not in result


def test_projects_identity_header_while_without_a_synthetic_guard() -> None:
    source = """\
kernel.def @identity_header_loop() {
  %count = index.constant 1 : index
  kernel.launch.config workgroups(%count, %count, %count) workgroup_size(%count, %count, %count) : index
} launch() {
  %initial_active = scalar.constant 1 : i1
  %initial_position = index.constant 0 : index
  %limit = index.constant 4 : index
  %step = index.constant 1 : index
  %active, %position = scf.while(%before_active = %initial_active : i1, %before_position = %initial_position : index) -> (i1, index) {
    scf.condition %before_active, %before_active, %before_position : i1, i1, index
  } do(%body_active: i1, %body_position: index) {
    %next_position = index.add %body_position, %step : index
    %next_active = index.cmp ult, %next_position, %limit : index
    scf.yield %next_active, %next_position : i1, index
  }
  kernel.return
}
"""

    result = render_module(load_module(source.encode()))

    assert result.count("while (") == 1
    assert "bool active = 1;" in result
    assert "while (active) {" in result
    assert "loop_continues" not in result
