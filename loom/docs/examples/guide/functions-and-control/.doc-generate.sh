#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Executes the loop-tuning walkthrough's hardware-independent commands and
# publishes source and actual report excerpts beside the generated page.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
output_dir="${1:-${repo_root}/build/loom-docs/examples/guide/functions-and-control}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

"${repo_root}/build_tools/bin/iree-bazel-build" --config=asan \
  //loom/src/loom/tools/loom-format \
  //loom/src/loom/tools/loom-link \
  //loom/src/loom/tools/loom-compile \
  //loom/src/loom/tools/iree-benchmark-loom \
  //loom/py/loom/tools:loom-compile-report
loom_format="${repo_root}/bazel-bin/loom/src/loom/tools/loom-format/loom-format"
loom_link="${repo_root}/bazel-bin/loom/src/loom/tools/loom-link/loom-link"
loom_compile="${repo_root}/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile"
loom_report="${repo_root}/bazel-bin/loom/py/loom/tools/loom-compile-report"
loom_benchmark="${repo_root}/bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom"

cp -- "${script_dir}/read-ahead.loom" "${output_dir}/read-ahead.loom"
cp -- "${script_dir}/read-ahead-tests.loom" "${output_dir}/read-ahead-tests.loom"
cp -- "${script_dir}/vector-read-ahead.loom" "${output_dir}/vector-read-ahead.loom"
cp -- "${script_dir}/vector-read-ahead-tests.loom" "${output_dir}/vector-read-ahead-tests.loom"
cp -- "${script_dir}/paired-read-ahead.loom" "${output_dir}/paired-read-ahead.loom"
cp -- "${script_dir}/paired-read-ahead-tests.loom" "${output_dir}/paired-read-ahead-tests.loom"
cp -- "${script_dir}/guarded-read-ahead.loom" "${output_dir}/guarded-read-ahead.loom"
cp -- "${script_dir}/guarded-read-ahead-tests.loom" "${output_dir}/guarded-read-ahead-tests.loom"
cp -- "${repo_root}/loom/src/loom/test/corpus/checked_benchmarks/streaming_packed_s8_dot.loom" \
  "${output_dir}/streaming-packed-dot.loom"
cp -- "${repo_root}/loom/src/loom/test/corpus/checked_benchmarks/routed_row_combine_f32.loom" \
  "${output_dir}/routed-row-combine.loom"

cd -- "${output_dir}"
"${loom_format}" --check guarded-read-ahead.loom
"${loom_format}" --check guarded-read-ahead-tests.loom
"${loom_link}" guarded-read-ahead.loom guarded-read-ahead-tests.loom \
  --mode=merge --to=bc --output=guarded-read-ahead.loombc
"${loom_compile}" guarded-read-ahead.loombc --root=@sum_guarded_rows \
  --target=amdgpu:gfx11-generic --format=amdgpu-hsaco \
  --output=guarded-rows.hsaco --compile-report=details \
  --compile-report-output=guarded-rows.report.json
"${loom_report}" show guarded-rows.report.json >guarded-rows.show.txt
"${loom_report}" suggest guarded-rows.report.json >guarded-rows.suggest.txt
sed -n '/^Source loop pipelines/,$p' guarded-rows.show.txt >guarded-pipeline-schedule.txt
grep -Fq 'depth=3 queue_records=2' guarded-pipeline-schedule.txt
grep -Fq 'scf.if producer iteration_lookahead=2' guarded-pipeline-schedule.txt
"${loom_compile}" guarded-read-ahead.loombc --root=@guarded_rows_composed \
  --target=spirv:vulkan1.3+bda --format=spirv-binary --output=guarded-rows.spv
"${loom_benchmark}" guarded-read-ahead.loombc --benchmark=@guarded_rows_time \
  --dry-run --output=guarded-rows.plan.json

"${loom_format}" --check vector-read-ahead.loom
"${loom_format}" --check vector-read-ahead-tests.loom
"${loom_link}" vector-read-ahead.loom vector-read-ahead-tests.loom \
  --mode=merge --to=bc --output=vector-read-ahead.loombc
"${loom_compile}" vector-read-ahead.loombc --root=@sum_vector_rows \
  --target=amdgpu:gfx11-generic --format=amdgpu-hsaco \
  --output=vector-rows.hsaco --compile-report=details \
  --compile-report-output=vector-rows.report.json
"${loom_report}" show vector-rows.report.json >vector-rows.show.txt
"${loom_report}" suggest vector-rows.report.json >vector-rows.suggest.txt
sed -n '/^\[scf.compare_pipeline_depth\]/,/^$/p' vector-rows.suggest.txt >vector-pipeline-suggest.txt
test -s vector-pipeline-suggest.txt
"${loom_compile}" vector-read-ahead.loombc --root=@sum_vector_rows_composed \
  --target=amdgpu:gfx11-generic --format=amdgpu-hsaco \
  --output=composed-rows.hsaco --compile-report=details \
  --compile-report-output=composed-rows.report.json
"${loom_report}" show composed-rows.report.json >composed-rows.show.txt
grep -Fq 'depth=4 queue_records=3' composed-rows.show.txt
grep -Fq 'depth=1 queue_records=0' composed-rows.show.txt
"${loom_compile}" vector-read-ahead.loombc --root=@sum_vector_rows_composed \
  --target=spirv:vulkan1.3+bda --format=spirv-binary --output=composed-rows.spv
"${loom_benchmark}" vector-read-ahead.loombc --benchmark=@sum_vector_rows_time \
  --dry-run --output=vector-rows.plan.json

"${loom_format}" --check read-ahead.loom
"${loom_format}" --check read-ahead-tests.loom
"${loom_link}" read-ahead.loom read-ahead-tests.loom \
  --mode=merge --to=bc --output=read-ahead.loombc

for depth in 1 3; do
  "${loom_compile}" read-ahead.loombc --root=@sum_rows \
    --target=amdgpu:gfx11-generic --format=amdgpu-hsaco \
    --config="read_ahead.depth=${depth}" --config=read_ahead.unroll=4 \
    --output="sum-rows-d${depth}.hsaco" --compile-report=details \
    --compile-report-output="sum-rows-d${depth}.report.json"
  "${loom_report}" show "sum-rows-d${depth}.report.json" \
    >"sum-rows-d${depth}.show.txt"
  "${loom_benchmark}" read-ahead.loombc --benchmark=@sum_rows_64 \
    --config="read_ahead.depth=${depth}" --config=read_ahead.unroll=4 \
    --dry-run --output="sum-rows-d${depth}.plan.json"
done

"${loom_report}" suggest sum-rows-d3.report.json >sum-rows-d3.suggest.txt
"${loom_report}" diff sum-rows-d1.report.json sum-rows-d3.report.json \
  --force >sum-rows.diff.txt
sed -n '/^Source loop pipelines/,$p' sum-rows-d3.show.txt >pipeline-schedule.txt
sed -n '/^\[scf.compare_pipeline_depth\]/,/^$/p' sum-rows-d3.suggest.txt >pipeline-suggest.txt
grep -Fq 'depth=1 queue_records=0' sum-rows-d1.show.txt
grep -Fq 'depth=3 queue_records=2' pipeline-schedule.txt
grep -Fq 'read_ahead.depth' sum-rows.diff.txt
test -s pipeline-suggest.txt

"${loom_compile}" streaming-packed-dot.loom \
  --root=@streaming_packed_s8_dot_read_ahead \
  --target=amdgpu:gfx11-generic --format=amdgpu-hsaco \
  --config=packed_stream.depth=4 --config=packed_stream.unroll=2 \
  --output=packed-dot.hsaco --compile-report=details \
  --compile-report-output=packed-dot.report.json
"${loom_report}" show packed-dot.report.json >packed-dot.show.txt
"${loom_report}" suggest packed-dot.report.json >packed-dot.suggest.txt
sed -n '/^\[amdgpu.pipeline_copy_waits\]/,/^$/p' packed-dot.suggest.txt >pipeline-copy-waits.txt
test -s pipeline-copy-waits.txt
"${loom_benchmark}" streaming-packed-dot.loom \
  --benchmark=@streaming_packed_s8_dot_read_ahead_n128_time \
  --config=packed_stream.depth=4 --config=packed_stream.unroll=2 \
  --dry-run --output=packed-dot.plan.json

# Publish the matched dependent-load controls and their real compile reports.
"${loom_format}" --check routed-row-combine.loom
for policy in serial pipelined; do
  "${loom_compile}" routed-row-combine.loom \
    --root="@routed_row_combine_${policy}" \
    --target=amdgpu:gfx11-generic --format=amdgpu-hsaco \
    --output="routed-${policy}.hsaco" --compile-report=details \
    --compile-report-output="routed-${policy}.report.json"
  "${loom_benchmark}" routed-row-combine.loom \
    --benchmark="@routed_row_combine_${policy}_n8_t256" \
    --dry-run --output="routed-${policy}.plan.json"
done
"${loom_report}" diff routed-serial.report.json routed-pipelined.report.json \
  --force >routed-lookahead.diff.txt

# Compile the independent caller grid and retain the bounded evidence readers use.
"${loom_format}" --check paired-read-ahead.loom
"${loom_format}" --check paired-read-ahead-tests.loom
"${loom_link}" vector-read-ahead.loom paired-read-ahead.loom \
  paired-read-ahead-tests.loom --mode=merge --to=bc --output=paired-read-ahead.loombc
mkdir -p paired
compile_pair() {
  local left_depth="$1" left_factor="$2" right_depth="$3" right_factor="$4"
  local candidate="paired/l${1}u${2}-r${3}u${4}"
  "${loom_compile}" paired-read-ahead.loombc --root=@sum_paired_rows \
    --target=amdgpu:gfx1151 --format=amdgpu-hsaco \
    --config="paired_rows.left_lookahead=$((left_depth - 1))" \
    --config="paired_rows.left_unroll=${left_factor}" \
    --config="paired_rows.right_lookahead=$((right_depth - 1))" \
    --config="paired_rows.right_unroll=${right_factor}" \
    --output="${candidate}.hsaco" --compile-report=details \
    --compile-report-output="${candidate}.report.json"
  "${loom_report}" show "${candidate}.report.json" --format=json >"${candidate}.view.json"
  "${loom_report}" suggest "${candidate}.report.json" >"${candidate}.suggest.txt"
}
for left_depth in 1 4; do
  for left_factor in 1 4; do
    for right_depth in 1 4; do
      for right_factor in 1 4; do
        compile_pair "${left_depth}" "${left_factor}" "${right_depth}" "${right_factor}"
      done
    done
  done
done
for depth in 8 16 32; do
  compile_pair "${depth}" 4 "${depth}" 4
done
"${loom_report}" diff paired/l1u4-r1u4.report.json paired/l4u4-r1u4.report.json \
  --force >paired/depth.diff.txt
grep -Fq 'paired_rows.left_lookahead' paired/depth.diff.txt
sed -n '/^\[scf.compare_pipeline_depth\]/,/^$/p' paired/l4u4-r1u1.suggest.txt \
  >paired-pipeline-suggest.txt
test -s paired-pipeline-suggest.txt
"${loom_compile}" paired-read-ahead.loombc --root=@sum_paired_rows \
  --target=spirv:vulkan1.3+bda --format=spirv-binary --output=paired-rows.spv
"${loom_benchmark}" paired-read-ahead.loombc --benchmark=@paired_rows_time \
  --dry-run --output=paired-rows.plan.json

python3 - <<'PY'
import itertools
import json
from pathlib import Path

policies = list(itertools.product((1, 4), repeat=4))
policies += [(depth, 4, depth, 4) for depth in (8, 16, 32)]
views = {}
for left_depth, left_factor, right_depth, right_factor in policies:
    name = f"l{left_depth}u{left_factor}-r{right_depth}u{right_factor}"
    view = json.loads(Path(f"paired/{name}.view.json").read_text())
    expected = {
        "paired_rows.left_lookahead": str(left_depth - 1),
        "paired_rows.left_unroll": str(left_factor),
        "paired_rows.right_lookahead": str(right_depth - 1),
        "paired_rows.right_unroll": str(right_factor),
    }
    assert view["identity"]["config_bindings"] == [
        {"key": key, "value": value} for key, value in sorted(expected.items())
    ], name
    assert [row["depth"] for row in view["loop_pipelines"]["rows"]] == [
        left_depth, right_depth
    ], name
    assert len(view["entries"]) == 1, name
    views[name] = view["entries"][0]

lines = [
    "| Left policy | Right policy | Code bytes | VGPRs | Modeled residency | Spills |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for left_depth, left_factor, right_depth, right_factor in (
    (1, 1, 1, 1), (1, 4, 1, 4), (4, 1, 4, 1), (4, 4, 1, 1),
    (4, 4, 4, 4), (8, 4, 8, 4), (16, 4, 16, 4), (32, 4, 32, 4),
):
    entry = views[f"l{left_depth}u{left_factor}-r{right_depth}u{right_factor}"]
    facts = entry["artifact_facts"]
    analysis = entry["compiler_analysis"]
    lines.append(
        f"| {left_depth} / {left_factor} | {right_depth} / {right_factor} | "
        f"{facts['code_byte_count']} | {analysis['vector_register_count']} | "
        f"{analysis['occupancy_percent']}% | {analysis['allocation_spill_count']} |"
    )
Path("paired-resources.md").write_text("\n".join(lines) + "\n")
PY
