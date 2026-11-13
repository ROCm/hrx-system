# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executes predicate conversions and comparisons after public C++ projection."""

import json
import operator
import subprocess
import sys

from loom.format.cxx import load_module, render_module

INTEGERS = {
    "i8": "unsigned char",
    "i16": "unsigned short",
    "i32": "unsigned",
    "i64": "unsigned long long",
}
FLOATS = {
    "f8E4M3": "loom::type::float8_e4m3fn_t",
    "f8E5M2": "loom::type::float8_e5m2_t",
    "f16": "std::float16_t",
    "bf16": "std::bfloat16_t",
    "f32": "float",
    "f64": "double",
}
SIGNED_INTEGERS = {
    "i8": "signed char",
    "i16": "short",
    "i32": "int",
    "i64": "long long",
}
SOURCE_TYPES = {"i1": "bool", **INTEGERS, **FLOATS}


def conversions():
    for integer in INTEGERS:
        yield "extui", "i1", integer, ((0, 0), (1, 1))
        yield "extsi", "i1", integer, ((0, 0), (1, -1))
        yield (
            "trunci",
            integer,
            "i1",
            ((0, 0), (1, 1), (2, 0), (3, 1), (-2, 0), (-1, 1)),
        )
    for floating in FLOATS:
        yield "uitofp", "i1", floating, ((0, 0), (1, 1))
        yield "sitofp", "i1", floating, ((0, 0), (1, -1))
        # Inputs exercise rounding toward zero within the i1 result domain.
        yield (
            "fptoui",
            floating,
            "i1",
            ((-0.75, 0), (0, 0), (0.75, 0), (1, 1), (1.75, 1)),
        )
        yield (
            "fptosi",
            floating,
            "i1",
            ((-1.75, 1), (-1, 1), (-0.75, 0), (0, 0), (0.75, 0)),
        )


def ir_type(element: str, shape: tuple[int, ...]) -> str:
    dimensions = "x".join(map(str, shape))
    return f"vector<{dimensions}x{element}>" if shape else element


def comparison_programs() -> tuple[list[str], list[str], int]:
    functions, checks = [], []
    expectation_count = 0
    relations = {
        "eq": operator.eq,
        "ne": operator.ne,
        "slt": operator.lt,
        "sle": operator.le,
        "sgt": operator.gt,
        "sge": operator.ge,
        "ult": operator.lt,
        "ule": operator.le,
        "ugt": operator.gt,
        "uge": operator.ge,
    }
    for width in (1, 8, 16, 32, 64):
        modulus = 1 << width
        source_type = SOURCE_TYPES[f"i{width}"]
        samples = [
            (0, 0),
            (modulus - 1, 0),
            (0, modulus - 1),
            (modulus - 1, modulus - 1),
            (modulus // 2, modulus // 2 - 1),
        ]
        for predicate, relation in relations.items():
            name = f"compare_{predicate}_i{width}"
            functions.append(
                f"func.def public noinline @{name}(%lhs: i{width}, %rhs: i{width}) -> (i1) {{\n"
                f"  %signed_use = scalar.sitofp %lhs : i{width} to f64\n"
                f"  %result = scalar.cmpi {predicate}, %lhs, %rhs : i{width}\n"
                "  func.return %result : i1\n}\n"
            )
            declarations, expectations = [], []
            for ordinal, (lhs, rhs) in enumerate(samples):
                left = (
                    lhs - modulus
                    if predicate.startswith("s") and lhs >= modulus // 2
                    else lhs
                )
                right = (
                    rhs - modulus
                    if predicate.startswith("s") and rhs >= modulus // 2
                    else rhs
                )
                expected = "true" if relation(left, right) else "false"
                declarations.append(
                    f"  const auto actual_{ordinal} = {name}(static_cast<{source_type}>({lhs}ull), static_cast<{source_type}>({rhs}ull));"
                )
                expectations.append(
                    f"  loom::check::expect_equal(actual_{ordinal}, {expected});"
                )
            checks.append(
                f"LOOM_CHECK_CASE(check_{name}) {{\n"
                + "\n".join(declarations + expectations)
                + "\n}"
            )
            expectation_count += len(expectations)
    functions.append("""
func.def public noinline @carried_maximum(%lhs: i64, %rhs: i64) -> (i64) {
  %result = scf.while(%value = %lhs : i64) -> (i64) {
    %less = scalar.cmpi slt, %value, %rhs : i64
    scf.condition %less, %value : i1, i64
  } do(%current: i64) {
    scf.yield %rhs : i64
  }
  func.return %result : i64
}
""")
    checks.append("""
static long long observe_carried_maximum(long long lhs, long long rhs) {
  return static_cast<long long>(carried_maximum(lhs, rhs));
}
LOOM_CHECK_CASE(signed_comparison_of_unsigned_loop_carrier) {
  const auto crossed = observe_carried_maximum(-7ll, 9ll);
  const auto negative = observe_carried_maximum(-7ll, -9ll);
  const auto unchanged = observe_carried_maximum(9ll, -7ll);
  loom::check::expect_equal(crossed, 9ll);
  loom::check::expect_equal(negative, -7ll);
  loom::check::expect_equal(unchanged, 9ll);
}
""")
    expectation_count += 3
    return functions, checks, expectation_count


def programs() -> tuple[str, str, int, int]:
    functions, checks, expectation_count = comparison_programs()
    observations = []
    for shape in ((), (4,), (2, 2)):
        prefix = "x".join(map(str, shape))
        dialect = "vector" if shape else "scalar"

        for member, source, result, samples in conversions():
            name = f"{member}_{source}_{result}_{prefix or 'scalar'}"
            source_type, result_type = ir_type(source, shape), ir_type(result, shape)
            functions.append(
                f"func.def public noinline @{name}(%input: {source_type}) -> ({result_type}) {{\n"
                f"  %result = {dialect}.{member} %input : {source_type} to {result_type}\n"
                f"  func.return %result : {result_type}\n}}\n"
            )
            input_element = SOURCE_TYPES[source]
            output_element = SOURCE_TYPES[result]
            if shape:
                if len(shape) == 1 and source != "i1":
                    observations.append(
                        f"using Input_{name} = {input_element} __attribute__((ext_vector_type(4)));"
                    )
                    input_expression = f"(Input_{name}{{}} + input)"
                else:
                    dimensions = ", ".join(map(str, shape))
                    observations.append(
                        f"using Input_{name} = loom::type::vector<loom::type::shape<{dimensions}>, {input_element}>;"
                    )
                    input_expression = f"loom::vector::splat<Input_{name}>({{}}, input)"
                coordinates = "lane" if len(shape) == 1 else "lane / 2u, lane % 2u"
                observed = (
                    f"loom::vector::extract<{output_element}>(result, {coordinates})"
                )
            else:
                input_expression, observed = "input", "result"
            if member == "extsi":
                observed = f"static_cast<{SIGNED_INTEGERS[result]}>({observed})"
            observations.append(
                f"static long long observe_{name}({input_element} input, unsigned lane) {{\n"
                f"  const auto result = {name}({input_expression});\n"
                f"  return static_cast<long long>({observed});\n}}"
            )
            declarations, expectations = [], []
            for ordinal, (value, expected) in enumerate(samples):
                for lane in range(4 if shape else 1):
                    actual = f"actual_{ordinal}_{lane}"
                    declarations.append(
                        f"  const auto {actual} = observe_{name}(static_cast<{input_element}>({value}), {lane}u);"
                    )
                    expectations.append(
                        f"  loom::check::expect_equal({actual}, {expected}ll);"
                    )
            checks.append(
                f"LOOM_CHECK_CASE(check_{name}) {{\n"
                + "\n".join(declarations + expectations)
                + "\n}"
            )
            expectation_count += len(expectations)
    functions.append("""
func.def noinline @forward_signed_bits(%input: vector<2x2xi32>) -> (vector<2x2xi32>) {
  func.return %input : vector<2x2xi32>
}
func.def public noinline @signed_transport(%input: vector<2x2xi1>, %direct: i1) -> (vector<2x2xi32>) {
  %extended = vector.extsi %input : vector<2x2xi1> to vector<2x2xi32>
  %forwarded = func.call @forward_signed_bits(%extended) : (vector<2x2xi32>) -> (vector<2x2xi32>)
  %selected = scf.if %direct -> (vector<2x2xi32>) {
    scf.yield %extended : vector<2x2xi32>
  } else {
    scf.yield %forwarded : vector<2x2xi32>
  }
  func.return %selected : vector<2x2xi32>
}
""")
    observations.append("""
static int observe_signed_transport(bool input, bool direct) {
  using Flags = loom::type::vector<loom::type::shape<2, 2>, bool>;
  const auto value = signed_transport(loom::vector::splat<Flags>({}, input), direct);
  return static_cast<int>(loom::vector::extract<unsigned>(value, 1u, 1u));
}
""")
    checks.append("""
LOOM_CHECK_CASE(signed_vector_bits_cross_calls_and_control_flow) {
  const auto direct_set = observe_signed_transport(true, true);
  const auto called_set = observe_signed_transport(true, false);
  const auto direct_clear = observe_signed_transport(false, true);
  const auto called_clear = observe_signed_transport(false, false);
  loom::check::expect_equal(direct_set, -1);
  loom::check::expect_equal(called_set, -1);
  loom::check::expect_equal(direct_clear, 0);
  loom::check::expect_equal(called_clear, 0);
}
""")
    expectation_count += 4
    return (
        "\n".join(functions),
        "\n".join(observations + checks),
        len(checks),
        expectation_count,
    )


def main(importer: str, runner: str) -> None:
    source, assertions, case_count, expectation_count = programs()
    projected = render_module(load_module(source.encode()))
    checked = (
        "#include <loomcxx/check.h>\n#include <loomcxx/vector.h>\n"
        + projected
        + assertions
    )
    imported = subprocess.run(
        [importer, "--cleanup=false", "--to=bc", "-"],
        input=checked.encode(),
        capture_output=True,
        check=False,
    )
    if imported.returncode:
        raise AssertionError(imported.stderr.decode())
    executed = subprocess.run(
        [runner, "--input-format=loombc", "-"],
        input=imported.stdout,
        capture_output=True,
        check=False,
    )
    if executed.returncode:
        raise AssertionError(executed.stdout.decode() + executed.stderr.decode())
    report = json.loads(executed.stdout)
    assert report["case_count"] == case_count, report
    assert report["sample_count"] == case_count, report
    assert report["skipped_case_count"] == 0, report
    assert report["planning_issue_count"] == 0, report
    assert (
        sum(sample["expectations"]["expectation_count"] for sample in report["samples"])
        == expectation_count
    ), report
    sys.stdout.write(
        f"{case_count} named predicate cases, {expectation_count} expectations passed\n"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
