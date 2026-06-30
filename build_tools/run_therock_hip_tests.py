#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fetch TheRock HIP test binaries and run them against a local HRX build."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import ci_core_linux as ci

DEFAULT_OUTPUT_DIR_NAME = Path("build") / "therock-hip-tests"
DEFAULT_ARTIFACT_SET = "core-with-upstream-hip"
DEFAULT_HIPTEST_COMPONENTS = ("test",)
HIPTEST_ARTIFACT_NAME = "core-hiptests"
HIP_SYMBOL_RE = re.compile(r"\b((?:__)?hip[A-Za-z0-9_]+)(?:@|$)")
TEST_RESULT_STATUSES = ("pass", "fail", "crash", "hang", "skipped")
TEST_RESULT_BAD_STATUSES = ("fail", "crash", "hang")
DEFAULT_COMPARE_EXAMPLE_LIMIT = 8


@dataclass(frozen=True)
class CTestCase:
    name: str
    command: list[str]
    working_directory: Path
    labels: list[str]
    skip_return_code: int | None
    environment: list[str]
    environment_modifications: list[str]


@dataclass(frozen=True)
class IsolatedTestResult:
    name: str
    command: list[str]
    working_directory: Path
    labels: list[str]
    status: str
    returncode: int | None
    duration_seconds: float
    log_path: Path | None


@dataclass(frozen=True)
class ResultComparison:
    baseline_name: str
    tip_name: str
    baseline: dict[str, object]
    tip: dict[str, object]
    test_names: list[str]
    baseline_results: dict[str, dict[str, object]]
    tip_results: dict[str, dict[str, object]]


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, os.fspath(default)))


def env_optional_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return Path(value)


def prepend_env_paths(env: dict[str, str], name: str, paths: list[Path]) -> None:
    values = [os.fspath(path) for path in paths]
    current = env.get(name)
    if current:
        values.append(current)
    env[name] = ":".join(values)


def hiptest_artifacts(components: list[str]) -> list[str]:
    return [
        f"{HIPTEST_ARTIFACT_NAME}_{component}_generic" for component in components
    ]


def wanted_artifacts(artifact_set: str, hiptest_components: list[str]) -> list[str]:
    return ci.wanted_artifacts(artifact_set) + hiptest_artifacts(hiptest_components)


def s3_bucket_url(bucket: str, params: dict[str, str]) -> str:
    return f"https://{bucket}.s3.us-east-2.amazonaws.com/?{urlencode(params)}"


def s3_object_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.us-east-2.amazonaws.com/{quote(key)}"


def list_s3_prefixes(bucket: str) -> list[str]:
    prefixes: list[str] = []
    params = {"list-type": "2", "delimiter": "/"}
    while True:
        root = read_s3_listing(bucket, params)
        prefixes.extend(
            node.findtext("{http://s3.amazonaws.com/doc/2006-03-01/}Prefix", "")
            for node in root.findall(
                "{http://s3.amazonaws.com/doc/2006-03-01/}CommonPrefixes"
            )
        )
        token = root.findtext(
            "{http://s3.amazonaws.com/doc/2006-03-01/}NextContinuationToken"
        )
        if not token:
            return prefixes
        params["continuation-token"] = token


def read_s3_listing(bucket: str, params: dict[str, str]) -> ET.Element:
    with urlopen(s3_bucket_url(bucket, params), timeout=60) as response:
        return ET.fromstring(response.read())


def list_s3_objects(bucket: str, prefix: str) -> list[ci.S3Object]:
    objects: list[ci.S3Object] = []
    params = {"list-type": "2", "prefix": prefix}
    while True:
        root = read_s3_listing(bucket, params)
        for node in root.findall("{http://s3.amazonaws.com/doc/2006-03-01/}Contents"):
            key = node.findtext("{http://s3.amazonaws.com/doc/2006-03-01/}Key")
            size = node.findtext("{http://s3.amazonaws.com/doc/2006-03-01/}Size")
            last_modified = node.findtext(
                "{http://s3.amazonaws.com/doc/2006-03-01/}LastModified"
            )
            if key is None or size is None or last_modified is None:
                raise RuntimeError("Malformed S3 listing response")
            objects.append(
                ci.S3Object(
                    key=key,
                    size=int(size),
                    last_modified=last_modified,
                )
            )
        token = root.findtext(
            "{http://s3.amazonaws.com/doc/2006-03-01/}NextContinuationToken"
        )
        if not token:
            return objects
        params["continuation-token"] = token


def download_object(bucket: str, obj: ci.S3Object, cache_dir: Path) -> Path:
    dest = cache_dir / Path(obj.key).name
    if dest.exists() and dest.stat().st_size == obj.size:
        ci.log(f"  == Cached {dest.name}")
        return dest
    ci.log(f"  ++ Downloading {obj.key}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urlopen(s3_object_url(bucket, obj.key), timeout=60) as response:
        with tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
    tmp.replace(dest)
    return dest


def download_checksum(bucket: str, key: str, archive_path: Path) -> Path | None:
    checksum_dest = archive_path.with_name(archive_path.name + ".sha256sum")
    try:
        with urlopen(s3_object_url(bucket, f"{key}.sha256sum"), timeout=60) as response:
            checksum_dest.write_bytes(response.read())
    except HTTPError:
        return None
    return checksum_dest


def discover_latest_run_id(release_type: str, wanted: list[str]) -> str:
    bucket = ci.release_bucket(release_type, "artifacts")
    candidates: list[int] = []
    for common_prefix in list_s3_prefixes(bucket):
        match = re.match(r"^(\d+)-linux/$", common_prefix)
        if match:
            candidates.append(int(match.group(1)))
    for run_id in sorted(candidates, reverse=True):
        prefix = f"{run_id}-{ci.PLATFORM}/"
        available = list_s3_objects(bucket, prefix)
        _, missing = ci.select_available(available, prefix, wanted)
        if not missing:
            return str(run_id)
    raise RuntimeError(
        f"Could not discover a {release_type} Linux run with TheRock HIP tests. "
        "Pass --run-id explicitly."
    )


def write_fetch_manifest(
    path: Path,
    *,
    release_type: str,
    run_id: str,
    bucket: str,
    artifact_set: str,
    hiptest_components: list[str],
    artifacts: list[ci.S3Object],
) -> None:
    data = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "release_type": release_type,
        "run_id": run_id,
        "platform": ci.PLATFORM,
        "bucket": bucket,
        "artifact_set": artifact_set,
        "hiptest_components": hiptest_components,
        "artifacts": [obj.__dict__ for obj in artifacts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def fetch_therock_artifacts(args: argparse.Namespace) -> None:
    wanted = wanted_artifacts(args.artifact_set, args.hiptest_component)
    run_id = args.run_id
    if args.latest:
        run_id = discover_latest_run_id(args.release_type, wanted)
        ci.log(f"Resolved latest {args.release_type} Linux run id: {run_id}")
    if not run_id:
        raise RuntimeError("Pass --run-id or leave --latest enabled")

    bucket = ci.release_bucket(args.release_type, "artifacts")
    prefix = f"{run_id}-{ci.PLATFORM}/"
    available = list_s3_objects(bucket, prefix)
    if not available:
        raise RuntimeError(f"No artifacts found at s3://{bucket}/{prefix}")

    selected, missing = ci.select_available(available, prefix, wanted)
    if missing:
        raise RuntimeError("Missing required artifacts:\n  " + "\n  ".join(missing))

    ci.log("Artifacts selected:")
    for obj in selected:
        ci.log(f"  {obj.key} ({obj.size / 1024 / 1024:.1f} MiB)")

    output_dir = args.rocm_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.download_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[tuple[ci.S3Object, Path]] = []
    with ThreadPoolExecutor(max_workers=args.download_concurrency) as executor:
        futures = {
            executor.submit(download_object, bucket, obj, cache_dir): obj
            for obj in selected
        }
        for future in as_completed(futures):
            obj = futures[future]
            downloaded.append((obj, future.result()))

    for obj, archive_path in sorted(downloaded, key=lambda item: item[1].name):
        checksum = download_checksum(bucket, obj.key, archive_path)
        ci.verify_checksum(archive_path, checksum)
        ci.log(f"  ++ Flattening {archive_path.name}")
        ci.flatten_therock_artifact(archive_path, output_dir)

    write_fetch_manifest(
        output_dir / ".hrx-therock-hip-tests-artifacts.json",
        release_type=args.release_type,
        run_id=run_id,
        bucket=bucket,
        artifact_set=args.artifact_set,
        hiptest_components=args.hiptest_component,
        artifacts=selected,
    )
    ci.log(f"TheRock ROCm root with HIP tests ready: {output_dir}")


def build_hrx(args: argparse.Namespace) -> None:
    build_args = argparse.Namespace(
        rocm_root=args.rocm_root,
        build_dir=args.hrx_build_dir,
        public_install_dir=args.hrx_install_dir,
        tests_install_dir=args.hrx_tests_install_dir,
        public_deps_dir=args.hrx_public_deps_dir,
        composed_install_dir=args.hrx_composed_install_dir,
        package_smoke_build_dir=args.hrx_package_smoke_build_dir,
        public_component="HrxPublicDist",
        tests_component="HrxTestsDist",
        build_type=args.build_type,
        target=args.build_target,
        sanitizer="none",
        assertions=args.assertions,
        ctest_regex="",
        ctest_exclude_regex="",
        ctest_label_regex="",
        ctest_label_exclude_regex="",
        ctest_parallelism=0,
        cts_device="",
        test_tmpdir=None,
        gpu=False,
        prepare_public_deps=True,
        package_smoke=False,
        package=False,
        package_suffix="",
        passthrough=False,
        amdgpu=True,
        cmake_option=args.cmake_option,
        package_output_dir=args.output_dir / "dist",
    )
    ci.build_core(build_args)


def hiptests_dir(rocm_root: Path) -> Path:
    return rocm_root / "share" / "hip" / "catch_tests"


def ctest_filter_args(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = []
    if args.test_type == "quick":
        cmd.extend(["-L", "smoke"])
    if args.ctest_regex:
        cmd.extend(["-R", args.ctest_regex])
    if args.ctest_exclude_regex:
        cmd.extend(["--exclude-regex", args.ctest_exclude_regex])
    return cmd


def runtime_env(args: argparse.Namespace) -> dict[str, str]:
    rocm_root = args.rocm_root.resolve()
    hrx_root = args.hrx_install_dir.resolve()
    hip_library = hrx_root / "lib" / "libamdhip64.so"
    ci.require_path(hiptests_dir(rocm_root) / "CTestTestfile.cmake", "HIP CTest file")
    ci.require_path(hip_library, "HRX libamdhip64.so")

    env = dict(os.environ)
    prepend_env_paths(env, "PATH", [hrx_root / "bin", rocm_root / "bin"])
    prepend_env_paths(
        env,
        "LD_LIBRARY_PATH",
        [
            hrx_root / "lib",
            rocm_root / "lib",
            rocm_root / "lib" / "rocm_sysdeps" / "lib",
        ],
    )
    prepend_env_paths(env, "CMAKE_PREFIX_PATH", [hrx_root, rocm_root])
    env["ROCM_PATH"] = os.fspath(rocm_root)
    env["THEROCK_BIN_DIR"] = os.fspath(rocm_root / "bin")
    env.setdefault("HRX_GPU_DRIVER", "amdgpu")
    env["LD_PRELOAD"] = ":".join(
        [os.fspath(hip_library)]
        + ([env["LD_PRELOAD"]] if env.get("LD_PRELOAD") else [])
    )
    if args.amdgpu_families:
        env["AMDGPU_FAMILIES"] = args.amdgpu_families
    return env


def hiptests_command(args: argparse.Namespace) -> list[str | Path]:
    cmd: list[str | Path] = [
        "ctest",
        "--tests-information",
        f"{args.shard_index},,{args.total_shards}",
        "--test-dir",
        hiptests_dir(args.rocm_root.resolve()),
        "--output-on-failure",
    ]
    if args.ctest_parallelism:
        cmd.extend(["--parallel", str(args.ctest_parallelism)])
    if args.ctest_timeout:
        cmd.extend(["--timeout", str(args.ctest_timeout)])
    cmd.extend(ctest_filter_args(args))
    return cmd


def property_value(test: dict[str, object], name: str) -> object | None:
    for prop in test.get("properties", []):
        if not isinstance(prop, dict):
            continue
        if prop.get("name") == name:
            return prop.get("value")
    return None


def string_list_property(test: dict[str, object], name: str) -> list[str]:
    value = property_value(test, name)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [part for part in str(value).split(";") if part]


def int_property(test: dict[str, object], name: str) -> int | None:
    value = property_value(test, name)
    if value is None or value == "":
        return None
    return int(value)


def path_property(test: dict[str, object], name: str, default: Path) -> Path:
    value = property_value(test, name)
    if value is None or value == "":
        return default
    return Path(str(value))


def discover_ctest_tests(
    args: argparse.Namespace, env: dict[str, str]
) -> list[CTestCase]:
    test_dir = hiptests_dir(args.rocm_root.resolve())
    cmd: list[str | Path] = [
        "ctest",
        "--show-only=json-v1",
        "--test-dir",
        test_dir,
    ]
    cmd.extend(ctest_filter_args(args))
    result = subprocess.run(
        [os.fspath(arg) for arg in cmd],
        cwd=ci.REPO_ROOT,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not discover TheRock HIP CTest tests:\n"
            f"Command: {' '.join(os.fspath(arg) for arg in cmd)}\n"
            f"Result: {result.returncode}\n"
            f"Output:\n{result.stdout}{result.stderr}"
        )
    data = json.loads(result.stdout)
    tests: list[CTestCase] = []
    for test in data.get("tests", []):
        if not isinstance(test, dict):
            continue
        command_value = test.get("command")
        if not isinstance(command_value, list) or not command_value:
            continue
        tests.append(
            CTestCase(
                name=str(test["name"]),
                command=[str(arg) for arg in command_value],
                working_directory=path_property(test, "WORKING_DIRECTORY", test_dir),
                labels=string_list_property(test, "LABELS"),
                skip_return_code=int_property(test, "SKIP_RETURN_CODE"),
                environment=string_list_property(test, "ENVIRONMENT"),
                environment_modifications=string_list_property(
                    test, "ENVIRONMENT_MODIFICATION"
                ),
            )
        )
    if args.total_shards > 1:
        tests = [
            test
            for index, test in enumerate(tests, start=1)
            if ((index - 1) % args.total_shards) + 1 == args.shard_index
        ]
    return tests


def apply_test_environment(test: CTestCase, base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    for assignment in test.environment:
        name, separator, value = assignment.partition("=")
        if not separator:
            raise RuntimeError(
                f"Malformed CTest ENVIRONMENT entry for {test.name}: {assignment}"
            )
        env[name] = value
    for modification in test.environment_modifications:
        apply_environment_modification(env, modification)
    return env


def apply_environment_modification(env: dict[str, str], modification: str) -> None:
    name, separator, operation_and_value = modification.partition("=")
    if not separator:
        raise RuntimeError(f"Malformed CTest ENVIRONMENT_MODIFICATION: {modification}")
    operation, separator, value = operation_and_value.partition(":")
    if not separator and operation not in ("reset", "unset"):
        raise RuntimeError(f"Malformed CTest ENVIRONMENT_MODIFICATION: {modification}")

    if operation == "reset":
        env.pop(name, None)
    elif operation == "set":
        env[name] = value
    elif operation == "unset":
        env.pop(name, None)
    elif operation == "string_append":
        env[name] = env.get(name, "") + value
    elif operation == "string_prepend":
        env[name] = value + env.get(name, "")
    elif operation == "path_list_append":
        append_env_value(env, name, value, os.pathsep, append=True)
    elif operation == "path_list_prepend":
        append_env_value(env, name, value, os.pathsep, append=False)
    elif operation == "cmake_list_append":
        append_env_value(env, name, value, ";", append=True)
    elif operation == "cmake_list_prepend":
        append_env_value(env, name, value, ";", append=False)
    else:
        raise RuntimeError(f"Unsupported CTest ENVIRONMENT_MODIFICATION: {modification}")


def append_env_value(
    env: dict[str, str], name: str, value: str, separator: str, *, append: bool
) -> None:
    current = env.get(name, "")
    if not current:
        env[name] = value
    elif append:
        env[name] = current + separator + value
    else:
        env[name] = value + separator + current


def safe_log_name(index: int, test_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", test_name).strip("._")
    if not sanitized:
        sanitized = "test"
    return f"{index:05d}-{sanitized[:160]}.log"


def classify_test_result(
    *, timed_out: bool, returncode: int | None, skip_return_code: int | None
) -> str:
    if timed_out:
        return "hang"
    if returncode == skip_return_code:
        return "skipped"
    if returncode == 0:
        return "pass"
    if returncode is not None and returncode < 0:
        return "crash"
    return "fail"


def run_isolated_test(
    index: int,
    test_count: int,
    test: CTestCase,
    *,
    base_env: dict[str, str],
    timeout: int,
    report_dir: Path,
    log_passing_tests: bool,
) -> IsolatedTestResult:
    ci.log(f"[{index}/{test_count}] {test.name}")
    start = time.monotonic()
    timed_out = False
    output = ""
    returncode: int | None = None
    env = apply_test_environment(test, base_env)
    process = subprocess.Popen(
        test.command,
        cwd=test.working_directory,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout if timeout else None)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output_after_kill, _ = process.communicate()
        output = (output or "") + (output_after_kill or "")
        returncode = process.returncode

    duration = time.monotonic() - start
    status = classify_test_result(
        timed_out=timed_out,
        returncode=returncode,
        skip_return_code=test.skip_return_code,
    )
    log_path: Path | None = None
    if log_passing_tests or status not in ("pass", "skipped"):
        log_dir = report_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / safe_log_name(index, test.name)
        log_path.write_text(
            "\n".join(
                [
                    f"name: {test.name}",
                    f"status: {status}",
                    f"returncode: {returncode}",
                    f"duration_seconds: {duration:.3f}",
                    f"working_directory: {test.working_directory}",
                    "command:",
                    "  " + " ".join(test.command),
                    "",
                    "output:",
                    output or "",
                ]
            )
        )
    return IsolatedTestResult(
        name=test.name,
        command=test.command,
        working_directory=test.working_directory,
        labels=test.labels,
        status=status,
        returncode=returncode,
        duration_seconds=duration,
        log_path=log_path,
    )


def result_to_json(result: IsolatedTestResult, report_dir: Path) -> dict[str, object]:
    log_path = None
    if result.log_path:
        log_path = os.fspath(result.log_path.relative_to(report_dir))
    return {
        "name": result.name,
        "command": result.command,
        "working_directory": os.fspath(result.working_directory),
        "labels": result.labels,
        "status": result.status,
        "returncode": result.returncode,
        "duration_seconds": round(result.duration_seconds, 3),
        "log_path": log_path,
    }


def write_isolated_reports(
    report_dir: Path,
    args: argparse.Namespace,
    results: list[IsolatedTestResult],
    *,
    selected_test_count: int,
    complete: bool,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = {status: 0 for status in TEST_RESULT_STATUSES}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    data = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "complete": complete,
        "summary": counts,
        "selected_test_count": selected_test_count,
        "completed_test_count": len(results),
        "rocm_root": os.fspath(args.rocm_root.resolve()),
        "hrx_install_dir": os.fspath(args.hrx_install_dir.resolve()),
        "test_type": args.test_type,
        "ctest_regex": args.ctest_regex,
        "ctest_exclude_regex": args.ctest_exclude_regex,
        "shard_index": args.shard_index,
        "total_shards": args.total_shards,
        "isolated_test_timeout": args.isolated_test_timeout,
        "results": [result_to_json(result, report_dir) for result in results],
    }
    json_text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    json_report_path = report_dir / "hip-test-results.json"
    json_report_path.write_text(json_text)
    if args.isolated_json_output:
        json_output_path = args.isolated_json_output.resolve()
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        if json_output_path != json_report_path.resolve():
            json_output_path.write_text(json_text)

    lines = [
        "TheRock HIP isolated test results",
        f"complete: {str(complete).lower()}",
        f"selected_test_count: {selected_test_count}",
        f"completed_test_count: {len(results)}",
        "summary:",
        *[f"  {status}: {counts.get(status, 0)}" for status in TEST_RESULT_STATUSES],
        "",
        "non-passing tests:",
    ]
    non_passing = [
        result for result in results if result.status not in ("pass", "skipped")
    ]
    if not non_passing:
        lines.append("  none")
    for result in non_passing:
        log_suffix = f" ({result.log_path.relative_to(report_dir)})" if result.log_path else ""
        lines.append(f"  {result.status}: {result.name}{log_suffix}")
    lines.extend(["", "all tests:"])
    for result in results:
        lines.append(f"  {result.status}: {result.name}")
    (report_dir / "hip-test-results.txt").write_text("\n".join(lines) + "\n")


def load_result_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse result JSON {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"Result JSON {path} must contain a JSON object")
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"Result JSON {path} is missing a 'results' list")
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise RuntimeError(f"Result JSON {path} result #{index} is not an object")
        if not isinstance(result.get("name"), str):
            raise RuntimeError(f"Result JSON {path} result #{index} is missing a name")
        if not isinstance(result.get("status"), str):
            raise RuntimeError(f"Result JSON {path} result #{index} is missing a status")
    return data


def result_map(data: dict[str, object], *, path: Path) -> dict[str, dict[str, object]]:
    mapped: dict[str, dict[str, object]] = {}
    results = data["results"]
    if not isinstance(results, list):
        raise RuntimeError(f"Result JSON {path} is missing a 'results' list")
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError(f"Result JSON {path} contains a non-object result")
        name = result["name"]
        if not isinstance(name, str):
            raise RuntimeError(f"Result JSON {path} contains a result without a name")
        if name in mapped:
            raise RuntimeError(f"Result JSON {path} contains duplicate test {name!r}")
        mapped[name] = result
    return mapped


def result_set_name(path: Path) -> str:
    if path.stem == "hip-test-results":
        return path.parent.name
    return path.stem


def compare_result_json(
    baseline_path: Path,
    tip_path: Path,
    *,
    baseline_name: str | None,
    tip_name: str | None,
) -> ResultComparison:
    baseline = load_result_json(baseline_path)
    tip = load_result_json(tip_path)
    baseline_results = result_map(baseline, path=baseline_path)
    tip_results = result_map(tip, path=tip_path)
    missing_from_tip = sorted(set(baseline_results) - set(tip_results))
    missing_from_baseline = sorted(set(tip_results) - set(baseline_results))
    if missing_from_tip or missing_from_baseline:
        details = []
        if missing_from_tip:
            details.append(
                f"{len(missing_from_tip)} tests only in baseline "
                f"(first: {missing_from_tip[0]})"
            )
        if missing_from_baseline:
            details.append(
                f"{len(missing_from_baseline)} tests only in tip "
                f"(first: {missing_from_baseline[0]})"
            )
        raise RuntimeError(
            "Result JSON files do not contain the same tests: " + "; ".join(details)
        )
    tip_results_list = tip["results"]
    if not isinstance(tip_results_list, list):
        raise RuntimeError(f"Result JSON {tip_path} is missing a 'results' list")
    test_names = []
    for result in tip_results_list:
        if not isinstance(result, dict) or not isinstance(result.get("name"), str):
            raise RuntimeError(f"Result JSON {tip_path} contains a malformed result")
        test_names.append(result["name"])
    return ResultComparison(
        baseline_name=baseline_name or result_set_name(baseline_path),
        tip_name=tip_name or result_set_name(tip_path),
        baseline=baseline,
        tip=tip,
        test_names=test_names,
        baseline_results=baseline_results,
        tip_results=tip_results,
    )


def result_status_counts(results: dict[str, dict[str, object]]) -> dict[str, int]:
    counts = {status: 0 for status in TEST_RESULT_STATUSES}
    for result in results.values():
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def status_count_text(counts: dict[str, int]) -> str:
    return ", ".join(
        f"{status}={counts.get(status, 0)}" for status in TEST_RESULT_STATUSES
    )


def test_family(result: dict[str, object]) -> str:
    name = str(result["name"])
    labels = result.get("labels", [])
    label_text = (
        " ".join(str(label) for label in labels) if isinstance(labels, list) else ""
    )
    text = f"{name} {label_text}".lower()
    if name.startswith("Unit_HRR_"):
        return "HRR capture/replay"
    if "graph" in text or "capture" in text:
        return "Graphs and stream capture"
    if (
        "texture" in text
        or "tex" in text
        or "surface" in text
        or "array" in text
        or "mipmapped" in text
    ):
        return "Textures, surfaces, and arrays"
    if "ipc" in text or "mempool" in text or "virtual" in text or "vmm" in text:
        return "IPC, mempool, and VMM"
    if (
        "memcpy" in text
        or "memset" in text
        or "malloc" in text
        or "hostregister" in text
        or "hostmalloc" in text
        or "memory" in text
        or "memgetinfo" in text
        or "coherency" in text
    ):
        return "Memory and copies"
    if "rtc" in text or "hiprtc" in text or "linker" in text:
        return "RTC and compilation"
    if (
        "module" in text
        or "library" in text
        or "kernel" in text
        or "occupancy" in text
    ):
        return "Module, library, and kernel launch"
    if "stream" in text or "event" in text or "callback" in text:
        return "Streams and events"
    if (
        "device" in text
        or "warp" in text
        or "shuffle" in text
        or "syncthreads" in text
        or "atomic" in text
        or "coop" in text
        or "cooperative" in text
        or "math" in text
        or "half" in text
    ):
        return "Device intrinsics, math, atomics, and cooperative groups"
    if "nogputst" in text:
        return "No-GPU/error-path behavior"
    return "Other"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def sample_names(names: list[str], limit: int) -> str:
    if not names:
        return "none"
    sample = ", ".join(names[:limit])
    if len(names) > limit:
        sample += f", ... {len(names) - limit} more"
    return sample


def labels_for_result(result: dict[str, object]) -> list[str]:
    labels = result.get("labels", [])
    if not isinstance(labels, list):
        return ["<none>"]
    return [str(label) for label in labels] or ["<none>"]


def render_result_comparison_markdown(
    comparison: ResultComparison,
    *,
    example_limit: int,
) -> str:
    baseline_counts = result_status_counts(comparison.baseline_results)
    tip_counts = result_status_counts(comparison.tip_results)
    transition_counts: dict[tuple[str, str], int] = {}
    family_improvements: dict[str, list[str]] = {}
    family_regressions: dict[str, list[str]] = {}
    newly_crash_or_hang: list[str] = []
    common_hangs: list[str] = []
    fixed_hangs: list[str] = []
    new_hangs: list[str] = []
    label_totals: dict[str, dict[str, int]] = {}

    for name in comparison.test_names:
        baseline_result = comparison.baseline_results[name]
        tip_result = comparison.tip_results[name]
        baseline_status = str(baseline_result["status"])
        tip_status = str(tip_result["status"])
        transition_counts[(baseline_status, tip_status)] = (
            transition_counts.get((baseline_status, tip_status), 0) + 1
        )
        if baseline_status != "pass" and tip_status == "pass":
            family_improvements.setdefault(test_family(tip_result), []).append(name)
        if baseline_status == "pass" and tip_status != "pass":
            family_regressions.setdefault(test_family(tip_result), []).append(name)
        if baseline_status not in ("crash", "hang") and tip_status in ("crash", "hang"):
            newly_crash_or_hang.append(name)
        if baseline_status == "hang" and tip_status == "hang":
            common_hangs.append(name)
        elif baseline_status == "hang" and tip_status != "hang":
            fixed_hangs.append(name)
        elif baseline_status != "hang" and tip_status == "hang":
            new_hangs.append(name)
        for label in labels_for_result(tip_result):
            row = label_totals.setdefault(
                label,
                {
                    "total": 0,
                    "baseline_pass": 0,
                    "tip_pass": 0,
                    "newly_passing": 0,
                    "pass_regressions": 0,
                },
            )
            row["total"] += 1
            if baseline_status == "pass":
                row["baseline_pass"] += 1
            if tip_status == "pass":
                row["tip_pass"] += 1
            if baseline_status != "pass" and tip_status == "pass":
                row["newly_passing"] += 1
            if baseline_status == "pass" and tip_status != "pass":
                row["pass_regressions"] += 1

    baseline_non_pass = sum(
        baseline_counts.get(status, 0) for status in TEST_RESULT_BAD_STATUSES
    )
    tip_non_pass = sum(
        tip_counts.get(status, 0) for status in TEST_RESULT_BAD_STATUSES
    )
    newly_passing_count = sum(len(names) for names in family_improvements.values())
    pass_regression_count = sum(len(names) for names in family_regressions.values())
    crash_hang_fixed_count = sum(
        count
        for (baseline_status, tip_status), count in transition_counts.items()
        if baseline_status in ("crash", "hang") and tip_status not in ("crash", "hang")
    )

    lines = [
        "# TheRock HIP isolated result comparison: "
        f"{comparison.baseline_name} vs {comparison.tip_name}",
        "",
        f"Generated at: {dt.datetime.now(dt.UTC).isoformat()}",
        "",
        "## Inputs",
        "",
        f"- Baseline: `{comparison.baseline_name}`",
        f"- Tip: `{comparison.tip_name}`",
        "- Baseline selected/completed: "
        f"{comparison.baseline.get('selected_test_count', 'unknown')}/"
        f"{comparison.baseline.get('completed_test_count', 'unknown')}",
        "- Tip selected/completed: "
        f"{comparison.tip.get('selected_test_count', 'unknown')}/"
        f"{comparison.tip.get('completed_test_count', 'unknown')}",
        "",
        "## Totals",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Tree", "pass", "fail", "crash", "hang", "skipped", "non-pass"],
            [
                [
                    comparison.baseline_name,
                    str(baseline_counts.get("pass", 0)),
                    str(baseline_counts.get("fail", 0)),
                    str(baseline_counts.get("crash", 0)),
                    str(baseline_counts.get("hang", 0)),
                    str(baseline_counts.get("skipped", 0)),
                    str(baseline_non_pass),
                ],
                [
                    comparison.tip_name,
                    str(tip_counts.get("pass", 0)),
                    str(tip_counts.get("fail", 0)),
                    str(tip_counts.get("crash", 0)),
                    str(tip_counts.get("hang", 0)),
                    str(tip_counts.get("skipped", 0)),
                    str(tip_non_pass),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "Net movement:",
            "",
            "- "
            + status_count_text(
                {
                    status: tip_counts.get(status, 0)
                    - baseline_counts.get(status, 0)
                    for status in TEST_RESULT_STATUSES
                }
            ),
            f"- {newly_passing_count} tests moved from non-pass to pass.",
            f"- {pass_regression_count} tests moved from pass to non-pass.",
            f"- {crash_hang_fixed_count} tests moved out of crash/hang.",
            f"- {len(newly_crash_or_hang)} tests newly crash/hang.",
            "",
            "## Status Transitions",
            "",
        ]
    )
    transition_rows = []
    transition_statuses = sorted(
        set(TEST_RESULT_STATUSES) | {status for pair in transition_counts for status in pair}
    )
    for baseline_status in transition_statuses:
        for tip_status in transition_statuses:
            count = transition_counts.get((baseline_status, tip_status), 0)
            if count:
                transition_rows.append([f"{baseline_status} -> {tip_status}", str(count)])
    lines.extend(markdown_table(["transition", "count"], transition_rows))

    lines.extend(["", "## Biggest Improvements", ""])
    if family_improvements:
        for family, names in sorted(
            family_improvements.items(), key=lambda item: len(item[1]), reverse=True
        ):
            lines.append(
                f"- {family}: {len(names)} newly passing. "
                f"Examples: {sample_names(names, example_limit)}."
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Regressions", ""])
    if family_regressions:
        for family, names in sorted(
            family_regressions.items(), key=lambda item: len(item[1]), reverse=True
        ):
            lines.append(
                f"- {family}: {len(names)} pass-to-nonpass. "
                f"Examples: {sample_names(names, example_limit)}."
            )
    else:
        lines.append("- none")
    lines.append(f"- Newly crash/hang: {sample_names(newly_crash_or_hang, example_limit)}.")

    lines.extend(
        [
            "",
            "## Hangs",
            "",
            f"- Common hangs ({len(common_hangs)}): "
            f"{sample_names(common_hangs, example_limit)}.",
            f"- Baseline hangs fixed ({len(fixed_hangs)}): "
            f"{sample_names(fixed_hangs, example_limit)}.",
            f"- New tip hangs ({len(new_hangs)}): {sample_names(new_hangs, example_limit)}.",
            "",
            "## Upstream Label Deltas",
            "",
        ]
    )
    label_rows = []
    for label, counts in label_totals.items():
        pass_delta = counts["tip_pass"] - counts["baseline_pass"]
        if (
            counts["newly_passing"]
            or counts["pass_regressions"]
            or counts["total"] >= 25
        ):
            label_rows.append(
                [
                    label,
                    str(counts["total"]),
                    str(counts["baseline_pass"]),
                    str(counts["tip_pass"]),
                    f"{pass_delta:+d}",
                    str(counts["newly_passing"]),
                    str(counts["pass_regressions"]),
                ]
            )
    label_rows.sort(key=lambda row: int(row[5]) - int(row[6]), reverse=True)
    lines.extend(
        markdown_table(
            [
                "label",
                "total",
                "baseline pass",
                "tip pass",
                "pass delta",
                "newly passing",
                "pass regressions",
            ],
            label_rows,
        )
    )
    lines.append("")
    return "\n".join(lines)


def compare_results(args: argparse.Namespace) -> None:
    baseline_path, tip_path = [path.resolve() for path in args.compare_results]
    comparison = compare_result_json(
        baseline_path,
        tip_path,
        baseline_name=args.compare_baseline_name,
        tip_name=args.compare_tip_name,
    )
    markdown = render_result_comparison_markdown(
        comparison,
        example_limit=args.compare_example_limit,
    )
    if args.compare_output:
        output_path = args.compare_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown + "\n")
        ci.log(f"Wrote comparison markdown: {output_path}")
    else:
        print(markdown)


def repo_relative_path(path: Path | None, repo_root: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return repo_root / path


def run_isolated_hiptests(args: argparse.Namespace, env: dict[str, str]) -> None:
    tests = discover_ctest_tests(args, env)
    if not tests:
        raise RuntimeError("No TheRock HIP tests matched the selected filters")

    report_dir = args.isolated_report_dir.resolve()
    ci.log(
        f"Running {len(tests)} TheRock HIP tests in isolated mode; "
        f"report directory: {report_dir}"
    )
    results: list[IsolatedTestResult] = []
    complete = False
    try:
        for index, test in enumerate(tests, start=1):
            result = run_isolated_test(
                index,
                len(tests),
                test,
                base_env=env,
                timeout=args.isolated_test_timeout,
                report_dir=report_dir,
                log_passing_tests=args.isolated_log_passing_tests,
            )
            results.append(result)
            write_isolated_reports(
                report_dir,
                args,
                results,
                selected_test_count=len(tests),
                complete=False,
            )
            ci.log(
                f"  {result.status} in {result.duration_seconds:.2f}s"
                + (f" (log: {result.log_path})" if result.log_path else "")
            )
        complete = True
    finally:
        write_isolated_reports(
            report_dir,
            args,
            results,
            selected_test_count=len(tests),
            complete=complete,
        )

    counts = {status: 0 for status in TEST_RESULT_STATUSES}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    ci.log(
        "Isolated TheRock HIP summary: "
        + ", ".join(f"{status}={counts.get(status, 0)}" for status in TEST_RESULT_STATUSES)
    )
    bad_count = counts["fail"] + counts["crash"] + counts["hang"]
    if bad_count:
        raise RuntimeError(
            f"{bad_count} TheRock HIP tests did not pass. "
            f"See {report_dir / 'hip-test-results.txt'}"
        )


def strip_symbol_version(symbol: str) -> str:
    return symbol.split("@", 1)[0]


def require_tool(name: str) -> str:
    tool_path = shutil.which(name)
    if not tool_path:
        raise RuntimeError(f"Missing required executable on PATH: {name}")
    return tool_path


def defined_dynamic_symbols(library: Path, *, required: bool = True) -> set[str]:
    result = subprocess.run(
        [require_tool("nm"), "-D", "--defined-only", os.fspath(library)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if required:
            raise RuntimeError(
                f"Could not read dynamic symbols from {library}:\n"
                f"{result.stderr.strip()}"
            )
        return set()
    symbols: set[str] = set()
    for line in result.stdout.splitlines():
        columns = line.split()
        if columns:
            symbols.add(strip_symbol_version(columns[-1]))
    return symbols


def undefined_dynamic_hip_symbols(executable: Path) -> set[str]:
    result = subprocess.run(
        [require_tool("readelf"), "-Ws", os.fspath(executable)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return set()
    symbols: set[str] = set()
    for line in result.stdout.splitlines():
        if " UND " not in line:
            continue
        match = HIP_SYMBOL_RE.search(line)
        if match:
            symbols.add(match.group(1))
    return symbols


def hiptest_executables(test_dir: Path) -> list[Path]:
    executables: list[Path] = []
    for path in sorted(test_dir.iterdir()):
        if path.is_file() and os.access(path, os.X_OK):
            executables.append(path)
    return executables


def runtime_export_libraries(args: argparse.Namespace) -> list[Path]:
    rocm_root = args.rocm_root.resolve()
    hip_library = args.hrx_install_dir.resolve() / "lib" / "libamdhip64.so"
    libraries = [hip_library]
    seen = {hip_library.resolve()}
    for library_dir in [rocm_root / "lib", rocm_root / "lib" / "rocm_sysdeps" / "lib"]:
        if not library_dir.exists():
            continue
        for library in sorted(library_dir.glob("*.so*")):
            if library.name.startswith("libamdhip64.so"):
                continue
            resolved = library.resolve()
            if resolved in seen or not library.is_file():
                continue
            seen.add(resolved)
            libraries.append(library)
    return libraries


def missing_hip_imports(args: argparse.Namespace) -> dict[Path, list[str]]:
    test_dir = hiptests_dir(args.rocm_root.resolve())
    hip_library = args.hrx_install_dir.resolve() / "lib" / "libamdhip64.so"
    ci.require_path(test_dir, "HIP catch_tests directory")
    ci.require_path(hip_library, "HRX libamdhip64.so")

    exports: set[str] = set()
    for index, library in enumerate(runtime_export_libraries(args)):
        exports.update(defined_dynamic_symbols(library, required=(index == 0)))
    missing: dict[Path, list[str]] = {}
    for executable in hiptest_executables(test_dir):
        missing_symbols = sorted(undefined_dynamic_hip_symbols(executable) - exports)
        if missing_symbols:
            missing[executable] = missing_symbols
    return missing


def write_missing_abi_report(path: Path, missing: dict[Path, list[str]]) -> None:
    lines: list[str] = []
    for executable, symbols in missing.items():
        lines.append(f"{executable}:")
        lines.extend(f"  {symbol}" for symbol in symbols)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def check_hip_abi(args: argparse.Namespace) -> None:
    if args.abi_check == "off":
        return
    missing = missing_hip_imports(args)
    if not missing:
        ci.log("TheRock HIP test ABI preflight passed")
        return

    report_path = args.output_dir.resolve() / "hip-test-missing-abi-symbols.txt"
    write_missing_abi_report(report_path, missing)
    missing_symbols = sorted({symbol for symbols in missing.values() for symbol in symbols})
    summary = [
        "The runtime library set is missing "
        f"{len(missing_symbols)} HIP symbols imported by "
        f"{len(missing)} TheRock HIP test executables.",
        "CTest/Catch discovery executes those binaries before running tests, "
        "so discovery will fail with dynamic-loader errors until the ABI gap "
        "is fixed.",
        f"Full missing-symbol report: {report_path}",
        "First missing symbols:",
        *[f"  {symbol}" for symbol in missing_symbols[:20]],
    ]
    if len(missing_symbols) > 20:
        summary.append(f"  ... {len(missing_symbols) - 20} more")
    message = "\n".join(summary)
    if args.abi_check == "warn":
        ci.log(message)
        return
    raise RuntimeError(message)


def run_hiptests(args: argparse.Namespace) -> None:
    env = runtime_env(args)
    check_hip_abi(args)
    if args.list_tests:
        tests = discover_ctest_tests(args, env)
        ci.log(f"Listing {len(tests)} TheRock HIP tests with HRX HIP: {env['LD_PRELOAD']}")
        for index, test in enumerate(tests, start=1):
            print(f"{index:5d}: {test.name}")
        return
    if args.test_mode == "isolated":
        ci.log(f"Running TheRock HIP tests with isolated processes: {env['LD_PRELOAD']}")
        run_isolated_hiptests(args, env)
        return
    cmd = hiptests_command(args)
    ci.log(f"Running TheRock HIP tests with HRX HIP: {env['LD_PRELOAD']}")
    ci.run(cmd, cwd=ci.REPO_ROOT, env=env, stderr_to_stdout=True)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=env_optional_path("HRX_REPO_ROOT"),
        help=(
            "HRX source checkout to build and test. Defaults to the checkout "
            "containing this runner."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=env_optional_path("HRX_THEROCK_HIP_TEST_OUTPUT_DIR"),
    )
    parser.add_argument(
        "--release-type",
        default=env_default("HRX_RELEASE_TYPE", "nightly"),
        choices=["dev", "nightly", "prerelease"],
    )
    parser.add_argument("--run-id", default=env_default("HRX_RUN_ID", ""))
    parser.add_argument(
        "--latest",
        action=argparse.BooleanOptionalAction,
        default=not bool(env_default("HRX_RUN_ID", "")),
        help="Resolve the latest complete Linux run when --run-id is empty.",
    )
    parser.add_argument(
        "--artifact-set",
        default=env_default("HRX_ARTIFACT_SET", DEFAULT_ARTIFACT_SET),
        choices=sorted(ci.ARTIFACT_SETS),
        help="TheRock runtime artifact closure to fetch before HIP tests.",
    )
    parser.add_argument(
        "--hiptest-component",
        action="append",
        default=list(DEFAULT_HIPTEST_COMPONENTS),
        choices=["test", "lib", "dev", "doc", "dbg"],
        help="TheRock core-hiptests component to fetch. Repeat to fetch more.",
    )
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=env_int("HRX_DOWNLOAD_CONCURRENCY", 8),
    )
    parser.add_argument(
        "--rocm-root",
        type=Path,
        default=env_optional_path("HRX_ROCM_ROOT"),
    )
    parser.add_argument(
        "--download-cache-dir",
        type=Path,
        default=env_optional_path("HRX_DOWNLOAD_CACHE_DIR"),
    )
    parser.add_argument(
        "--hrx-build-dir",
        type=Path,
        default=env_optional_path("HRX_BUILD_DIR"),
    )
    parser.add_argument(
        "--hrx-install-dir",
        type=Path,
        default=env_optional_path("HRX_PUBLIC_INSTALL_DIR"),
    )
    parser.add_argument(
        "--hrx-tests-install-dir",
        type=Path,
        default=env_optional_path("HRX_TESTS_INSTALL_DIR"),
    )
    parser.add_argument(
        "--hrx-public-deps-dir",
        type=Path,
        default=env_optional_path("HRX_PUBLIC_DEPS_DIR"),
    )
    parser.add_argument(
        "--hrx-composed-install-dir",
        type=Path,
        default=env_optional_path("HRX_COMPOSED_INSTALL_DIR"),
    )
    parser.add_argument(
        "--hrx-package-smoke-build-dir",
        type=Path,
        default=env_optional_path("HRX_PACKAGE_SMOKE_BUILD_DIR"),
    )
    parser.add_argument(
        "--build-type", default=env_default("HRX_BUILD_TYPE", "RelWithDebInfo")
    )
    parser.add_argument("--build-target", default=env_default("HRX_BUILD_TARGET", "all"))
    parser.add_argument(
        "--assertions",
        action=argparse.BooleanOptionalAction,
        default=ci.env_bool("HRX_ASSERTIONS", False),
    )
    parser.add_argument("-D", dest="cmake_option", action="append", default=[])
    parser.add_argument(
        "--test-type",
        default=env_default("TEST_TYPE", "standard"),
        choices=["quick", "standard", "comprehensive", "full"],
    )
    parser.add_argument(
        "--amdgpu-families", default=env_default("AMDGPU_FAMILIES", "")
    )
    parser.add_argument("--shard-index", type=int, default=env_int("SHARD_INDEX", 1))
    parser.add_argument("--total-shards", type=int, default=env_int("TOTAL_SHARDS", 1))
    parser.add_argument(
        "--ctest-parallelism",
        type=int,
        default=env_int("HRX_THEROCK_HIP_CTEST_PARALLELISM", 1),
    )
    parser.add_argument(
        "--ctest-timeout",
        type=int,
        default=env_int("HRX_THEROCK_HIP_CTEST_TIMEOUT", 7200),
    )
    parser.add_argument(
        "--test-mode",
        default=env_default("HRX_THEROCK_HIP_TEST_MODE", "ctest"),
        choices=["ctest", "isolated"],
        help=(
            "Use normal CTest execution or run each discovered CTest test in "
            "a separate process for pass/fail/crash/hang accounting."
        ),
    )
    parser.add_argument(
        "--isolated-test-timeout",
        type=int,
        default=env_int("HRX_THEROCK_HIP_ISOLATED_TEST_TIMEOUT", 300),
        help="Per-test wall-clock timeout in seconds for --test-mode=isolated.",
    )
    parser.add_argument(
        "--isolated-report-dir",
        type=Path,
        default=env_optional_path("HRX_THEROCK_HIP_ISOLATED_REPORT_DIR"),
        help="Directory for isolated-mode JSON/text reports and failing logs.",
    )
    parser.add_argument(
        "--isolated-json-output",
        type=Path,
        default=env_optional_path("HRX_THEROCK_HIP_ISOLATED_JSON_OUTPUT"),
        help=(
            "Optional extra path for the isolated-mode JSON results. The runner "
            "always writes hip-test-results.json under --isolated-report-dir."
        ),
    )
    parser.add_argument(
        "--isolated-log-passing-tests",
        action="store_true",
        help="Write per-test logs for passing/skipped isolated tests too.",
    )
    parser.add_argument("--ctest-regex", default=env_default("HRX_CTEST_REGEX", ""))
    parser.add_argument(
        "--ctest-exclude-regex", default=env_default("HRX_CTEST_EXCLUDE_REGEX", "")
    )
    parser.add_argument(
        "--abi-check",
        default=env_default("HRX_THEROCK_HIP_ABI_CHECK", "fail"),
        choices=["fail", "warn", "off"],
        help="Check HRX exports against TheRock HIP test imports before CTest.",
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List discovered TheRock HIP CTest tests without running them.",
    )
    parser.add_argument(
        "--compare-results",
        nargs=2,
        metavar=("BASELINE_JSON", "TIP_JSON"),
        type=Path,
        help=(
            "Compare two isolated-mode hip-test-results.json files and emit a "
            "markdown delta report. This mode exits before fetch/build/test."
        ),
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        help="Markdown output path for --compare-results. Defaults to stdout.",
    )
    parser.add_argument(
        "--compare-baseline-name",
        help="Display name for the first --compare-results JSON file.",
    )
    parser.add_argument(
        "--compare-tip-name",
        help="Display name for the second --compare-results JSON file.",
    )
    parser.add_argument(
        "--compare-example-limit",
        type=int,
        default=DEFAULT_COMPARE_EXAMPLE_LIMIT,
        help="Maximum example test names to include per comparison bucket.",
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.repo_root is None:
        args.repo_root = ci.REPO_ROOT
    args.repo_root = args.repo_root.resolve()
    ci.REPO_ROOT = args.repo_root
    if args.output_dir is None:
        args.output_dir = args.repo_root / DEFAULT_OUTPUT_DIR_NAME
    else:
        args.output_dir = repo_relative_path(args.output_dir, args.repo_root)
    output_dir = args.output_dir
    if args.rocm_root is None:
        args.rocm_root = output_dir / "rocm-root"
    else:
        args.rocm_root = repo_relative_path(args.rocm_root, args.repo_root)
    if args.download_cache_dir is None:
        args.download_cache_dir = output_dir / "downloads"
    else:
        args.download_cache_dir = repo_relative_path(
            args.download_cache_dir, args.repo_root
        )
    if args.hrx_build_dir is None:
        args.hrx_build_dir = output_dir / "build" / "hrx"
    else:
        args.hrx_build_dir = repo_relative_path(args.hrx_build_dir, args.repo_root)
    if args.hrx_install_dir is None:
        args.hrx_install_dir = output_dir / "install" / "hrx-public"
    else:
        args.hrx_install_dir = repo_relative_path(args.hrx_install_dir, args.repo_root)
    if args.hrx_tests_install_dir is None:
        args.hrx_tests_install_dir = output_dir / "install" / "hrx-tests"
    else:
        args.hrx_tests_install_dir = repo_relative_path(
            args.hrx_tests_install_dir, args.repo_root
        )
    if args.hrx_public_deps_dir is None:
        args.hrx_public_deps_dir = output_dir / "install" / "hrx-public-deps"
    else:
        args.hrx_public_deps_dir = repo_relative_path(
            args.hrx_public_deps_dir, args.repo_root
        )
    if args.hrx_composed_install_dir is None:
        args.hrx_composed_install_dir = output_dir / "install" / "hrx-composed"
    else:
        args.hrx_composed_install_dir = repo_relative_path(
            args.hrx_composed_install_dir, args.repo_root
        )
    if args.hrx_package_smoke_build_dir is None:
        args.hrx_package_smoke_build_dir = output_dir / "build" / "package-smoke"
    else:
        args.hrx_package_smoke_build_dir = repo_relative_path(
            args.hrx_package_smoke_build_dir, args.repo_root
        )
    if args.isolated_report_dir is None:
        args.isolated_report_dir = output_dir / "isolated-results"
    else:
        args.isolated_report_dir = repo_relative_path(
            args.isolated_report_dir, args.repo_root
        )
    args.isolated_json_output = repo_relative_path(
        args.isolated_json_output, args.repo_root
    )
    if args.shard_index < 1:
        parser.error("--shard-index must be at least 1")
    if args.total_shards < 1:
        parser.error("--total-shards must be at least 1")
    if args.shard_index > args.total_shards:
        parser.error("--shard-index must be <= --total-shards")
    if args.ctest_parallelism < 0:
        parser.error("--ctest-parallelism must be non-negative")
    if args.ctest_timeout < 0:
        parser.error("--ctest-timeout must be non-negative")
    if args.isolated_test_timeout < 0:
        parser.error("--isolated-test-timeout must be non-negative")
    if args.compare_example_limit < 1:
        parser.error("--compare-example-limit must be at least 1")
    compare_only_args = [
        args.compare_output,
        args.compare_baseline_name,
        args.compare_tip_name,
    ]
    if not args.compare_results and any(value is not None for value in compare_only_args):
        parser.error("--compare-output and --compare-*-name require --compare-results")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.compare_results:
        compare_results(args)
        return 0
    if not args.skip_fetch:
        fetch_therock_artifacts(args)
    if not args.skip_build:
        build_hrx(args)
    if not args.skip_test:
        run_hiptests(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        ci.log(f"error: {e}")
        raise SystemExit(1) from None
