# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prepare local gfx1201 kernels for sdma_pipeline_benchmark.

Build the host tool with the normal HRX build target sdma_pipeline_benchmark.
Then, using a ROCm SDK with HIP, rocBLAS/hipBLASLt, and LLVM tools:

  python prepare.py --rocm /path/to/rocm --output /tmp/sdma-data
  python run.py --binary /path/to/sdma_pipeline_benchmark \
      --data /tmp/sdma-data --rocm /path/to/rocm --output /tmp/sdma-results
  python plot.py --results /tmp/sdma-results  # requires Matplotlib

Default runs compare direct-ring SDMA against shader copies. To also run IBs,
add --modes sdma-ib sdma-ring shader --allow-experimental-ib to run.py on a
machine whose KFD SDMA queue setup enables IB processing. Normal capabilities
remain disabled. The experimental VMID/context-save setup is not qualified for
preemption or eviction.

The workload is eight double-buffered 32 MiB H2D weight copies and 4096-square
FP16-input/FP32-output GEMMs. It uses the local library's optimized kernel via
native PM4, with no HIP/rocBLAS calls in the timed pipeline. This is a fixed
shape/launch ABI experiment, not a general library-kernel loading interface.
Preparation fails if the selected kernel differs from the supported ABI.
Generated code objects, launch templates, logs, and results stay in --output;
none are required in the source tree.

Record GPU clock/power policy alongside results: automatic clock selection can
affect comparisons between shader and SDMA traffic. These tools do not change
system power settings. Inspect per-run medians as well as aggregate timings.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rocm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="c++", help="Host C++ compiler")
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    sdk, out = args.rocm.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    common = [
        args.cxx,
        "-std=c++17",
        "-O3",
        "-D__HIP_PLATFORM_AMD__",
        "-I" + str(sdk / "include"),
    ]
    subprocess.run(
        common
        + [
            str(source_dir / "reference.cc"),
            "-L" + str(sdk / "lib"),
            "-lamdhip64",
            "-lrocblas",
            "-o",
            str(out / "reference"),
        ],
        check=True,
    )
    subprocess.run(
        common
        + [
            "-shared",
            "-fPIC",
            str(source_dir / "capture.cc"),
            "-ldl",
            "-o",
            str(out / "capture.so"),
        ],
        check=True,
    )
    env["LD_PRELOAD"] = str(out / "capture.so")
    # Never reuse outputs from an earlier successful capture.
    for name in ["captured-kernel.txt", "captured-args.bin", "optimized-manifest.json"]:
        (out / name).unlink(missing_ok=True)
    with (out / "capture.log").open("w") as log:
        subprocess.run(
            [str(out / "reference")],
            cwd=out,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=60,
        )
    name, launch = (out / "captured-kernel.txt").read_text().splitlines()
    if list(map(int, launch.split())) != [131072, 1, 1, 128, 1, 1, 176]:
        raise RuntimeError("Selected GEMM launch geometry is unsupported")
    modules = list(
        dict.fromkeys(
            re.findall(r"^MODULE_FILE (.*)$", (out / "capture.log").read_text(), re.M)
        )
    )
    # Select the module actually containing the captured entry point.
    block = None
    for module in modules:
        candidate = out / "candidate.hsaco"
        result = subprocess.run(
            [
                str(sdk / "llvm/bin/clang-offload-bundler"),
                "--type=o",
                "--unbundle",
                "--targets=hipv4-amdgcn-amd-amdhsa--gfx1201",
                "--input=" + module,
                "--output=" + str(candidate),
            ],
            capture_output=True,
        )
        if result.returncode:
            continue
        metadata = subprocess.check_output(
            [str(sdk / "llvm/bin/llvm-readobj"), "--notes", str(candidate)], text=True
        )
        for section in metadata.split("  - .args:")[1:]:
            if re.search(r"^    \.name:\s+" + re.escape(name) + r"\s*$", section, re.M):
                block = "  - .args:" + section
                module_path = Path(module)
                candidate.replace(out / "optimized.hsaco")
                break
        if block:
            break
    if not block or not re.search(r"\.private_segment_fixed_size:\s+0\b", block):
        raise RuntimeError("Cannot find a supported scratch-free GEMM code object")
    pointers = re.findall(
        r"\.name:\s+(\w+)\s+\.offset:\s+(\d+)\s+"
        r"\.size:\s+8\s+\.value_kind:\s+global_buffer",
        block,
    )
    if pointers[:4] != [("D", "32"), ("C", "40"), ("A", "48"), ("B", "56")]:
        raise RuntimeError("Selected GEMM pointer ABI is unsupported")
    arguments = (out / "captured-args.bin").read_bytes()
    if len(arguments) != 176 or any(arguments[32:64]):
        raise RuntimeError("Invalid sanitized GEMM argument template")
    # Any additional global-buffer argument must be null for this recipe.
    for _, offset in pointers[4:]:
        if int(offset) + 8 > len(arguments) or any(
            arguments[int(offset) : int(offset) + 8]
        ):
            raise RuntimeError("Selected GEMM requires an unsupported auxiliary buffer")
    (out / "optimized-args-template.bin").write_bytes(arguments)
    (out / "captured-args.bin").unlink()
    (out / "optimized-kernel-metadata.txt").write_text(block)
    subprocess.run(
        [
            str(sdk / "llvm/bin/clang"),
            "-target",
            "amdgcn-amd-amdhsa",
            "-mcpu=gfx1201",
            "-mno-wavefrontsize64",
            "-nogpulib",
            "-O3",
            "-std=c2x",
            "-c",
            str(source_dir.parent / "sdma_pipeline_benchmark_testdata.c"),
            "-o",
            str(out / "blit.o"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(sdk / "llvm/bin/ld.lld"),
            "-shared",
            str(out / "blit.o"),
            "-o",
            str(out / "blit.hsaco"),
        ],
        check=True,
    )
    files = [
        out / filename
        for filename in [
            "optimized.hsaco",
            "blit.hsaco",
            "optimized-args-template.bin",
            "captured-kernel.txt",
        ]
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / "optimized-manifest.json").write_text(
        json.dumps(
            {
                "kernel": name,
                "shape": [4096, 4096, 4096],
                "target": "gfx1201",
                "source_module": module_path.name,
                "source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print("Prepared benchmark data in", out)


if __name__ == "__main__":
    main()
