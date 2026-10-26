# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Measures final SDK wheels and reports their product-size history."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from build_tools.ci import github_api
from loom.packaging.python import release

SCHEMA = "loom.binary-size-history.v2"
MAIN_ARTIFACT_NAME = "loom-binary-size-history-main-v2"
MAX_MAIN_OBSERVATIONS = 40
SPARKLINE_GLYPHS = "▁▂▃▄▅▆▇█"


class BinarySizeError(RuntimeError):
    """Raised when a size witness or history record violates its contract."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(os.fspath(arg) for arg in command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _command_version(command: Sequence[str]) -> str:
    result = _run([*command, "--version"], capture_output=True)
    return result.stdout.strip() or result.stderr.strip()


def normalize_tool_identity(value: str, llvm_root: Path) -> str:
    """Removes the per-fetch install path from otherwise stable tool output."""
    resolved_root = llvm_root.resolve()
    root_spellings = {
        os.fspath(llvm_root),
        llvm_root.as_posix(),
        os.fspath(resolved_root),
        resolved_root.as_posix(),
    }
    for root_spelling in sorted(root_spellings, key=len, reverse=True):
        value = value.replace(root_spelling, "$LLVM_ROOT")
    return value


def _toolchain_identity(llvm_bin: Path, environment_id: str) -> dict[str, str]:
    compiler = shlex.split(os.environ.get("CC", "cc"))
    cxx_compiler = shlex.split(os.environ.get("CXX", "c++"))
    llvm_root = llvm_bin.resolve().parent
    identity = {
        "environment": environment_id,
        "cc": _command_version(compiler),
        "cxx": _command_version(cxx_compiler),
        "linker": _command_version([os.fspath(llvm_bin / "ld.lld")]),
        "elf_reader": _command_version([os.fspath(llvm_bin / "llvm-readobj")]),
    }
    return {
        name: normalize_tool_identity(value, llvm_root)
        for name, value in identity.items()
    }


def _fingerprint(platform: str, profile: str, toolchain: dict[str, str]) -> str:
    payload = {
        "schema": SCHEMA,
        "product": "hrx-loom",
        "configuration": ["loom-sdk", profile],
        "platform": platform,
        "toolchain": toolchain,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _read_elf_json(llvm_readobj: Path, path: Path, option: str) -> dict[str, Any]:
    result = _run(
        [
            os.fspath(llvm_readobj),
            "--elf-output-style=JSON",
            option,
            os.fspath(path),
        ],
        capture_output=True,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BinarySizeError(f"llvm-readobj returned invalid JSON for {path}") from exc
    if not isinstance(payload, list) or len(payload) != 1:
        raise BinarySizeError(
            f"llvm-readobj returned an unexpected file list for {path}"
        )
    if not isinstance(payload[0], dict):
        raise BinarySizeError(
            f"llvm-readobj returned an invalid file record for {path}"
        )
    return payload[0]


def classify_elf_sections(
    file_record: dict[str, Any], file_size: int
) -> dict[str, int]:
    """Classifies file-backed ELF bytes and reconciles them to exact size."""
    try:
        section_records = file_record["Sections"]
    except KeyError as exc:
        raise BinarySizeError("ELF JSON has no Sections array") from exc
    if not isinstance(section_records, list):
        raise BinarySizeError("ELF Sections is not an array")

    classes = {
        "executable": 0,
        "read_only": 0,
        "writable": 0,
        "unwind": 0,
        "other": 0,
    }
    section_bytes = 0
    for wrapped_section in section_records:
        try:
            section = wrapped_section["Section"]
            name = section["Name"]["Name"]
            section_type = section["Type"]["Name"]
            size = int(section["Size"])
            flag_names = {flag["Name"] for flag in section["Flags"]["Flags"]}
        except (KeyError, TypeError, ValueError) as exc:
            raise BinarySizeError(
                "ELF JSON contains an invalid section record"
            ) from exc
        if size < 0:
            raise BinarySizeError(f"ELF section {name!r} has a negative size")
        if section_type == "SHT_NOBITS":
            continue
        section_bytes += size
        if "SHF_ALLOC" not in flag_names:
            category = "other"
        elif name in {".eh_frame", ".eh_frame_hdr", ".gcc_except_table"}:
            category = "unwind"
        elif "SHF_EXECINSTR" in flag_names:
            category = "executable"
        elif "SHF_WRITE" in flag_names:
            category = "writable"
        else:
            category = "read_only"
        classes[category] += size

    overhead = file_size - section_bytes
    if overhead < 0:
        raise BinarySizeError(
            f"ELF section payload ({section_bytes}) exceeds file size ({file_size})"
        )
    classes["other"] += overhead
    if sum(classes.values()) != file_size:
        raise BinarySizeError("ELF section classes do not reconcile to file size")
    return classes


def measure_wheel(
    wheel: release.Wheel, llvm_readobj: Path, toolchain: dict[str, str]
) -> list[dict[str, Any]]:
    """Reads the shipping archive without rebuilding, stripping, or repairing it."""
    prefix = f"{wheel.platform}/{wheel.profile}"
    fingerprint = _fingerprint(wheel.platform, wheel.profile, toolchain)

    def record(name: str, title: str, size: int, digest: str) -> dict[str, Any]:
        return {
            "id": f"{prefix}/{name}",
            "title": f"{wheel.platform} / {title}",
            "fingerprint": fingerprint,
            "bytes": size,
            "sha256": digest,
        }

    with (
        zipfile.ZipFile(wheel.path) as archive,
        tempfile.TemporaryDirectory() as temporary,
    ):
        members = [member for member in archive.infolist() if not member.is_dir()]
        native = [member for member in members if ".data/scripts/" in member.filename]
        if not native:
            raise BinarySizeError(f"SDK wheel has no native payload: {wheel.path}")
        # Aggregate rows identify their containing archive; member rows identify
        # the exact uncompressed file whose bytes they measure.
        observations = [
            record("wheel", "Download", wheel.path.stat().st_size, wheel.digest),
            record(
                "installed",
                "Installed payload",
                sum(member.file_size for member in members),
                wheel.digest,
            ),
            record(
                "native",
                "Native payload",
                sum(member.file_size for member in native),
                wheel.digest,
            ),
        ]
        for member in sorted(native, key=lambda member: member.filename):
            name = Path(member.filename).name
            contents = archive.read(member)
            measurement = record(
                name, name, len(contents), hashlib.sha256(contents).hexdigest()
            )
            if contents.startswith(b"\x7fELF"):
                binary = Path(temporary) / name
                binary.write_bytes(contents)
                measurement["sections"] = classify_elf_sections(
                    _read_elf_json(llvm_readobj, binary, "--sections"), len(contents)
                )
            observations.append(measurement)
        return observations


def collect(args: argparse.Namespace) -> None:
    package = args.repo_root.resolve() / "loom/packaging/python"
    identity = release.read_identity(package / "version.json")
    readme = (package / "PYPI_README.md").read_text(encoding="utf-8")
    toolchain = _toolchain_identity(args.llvm_bin, args.environment_id)
    observations = []
    for path in args.wheel:
        wheel = release.inspect_wheel(path, identity, readme)
        observations.extend(
            measure_wheel(wheel, args.llvm_bin / "llvm-readobj", toolchain)
        )
    current = {
        "schema": SCHEMA,
        "observation": {
            "commit": args.commit,
            "repository": args.repository,
            "ref": args.ref,
            "run_id": args.run_id,
            "measured_at": args.measured_at
            or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            "toolchain": toolchain,
            "witnesses": observations,
        },
    }
    validate_current(current)
    _write_json(args.output_dir / "current.json", current)


def _require_sha(value: Any, description: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise BinarySizeError(f"{description} must be a full lowercase Git SHA")
    return value


def _validate_witness(witness: Any) -> None:
    if not isinstance(witness, dict):
        raise BinarySizeError("witness record is not an object")
    for key in ("id", "title", "fingerprint", "sha256"):
        if not isinstance(witness.get(key), str) or not witness[key]:
            raise BinarySizeError(f"witness {key} must be a non-empty string")
    if not isinstance(witness.get("bytes"), int) or witness["bytes"] < 0:
        raise BinarySizeError("witness bytes must be a non-negative integer")
    sections = witness.get("sections")
    if sections is None:
        return
    expected_sections = {"executable", "read_only", "writable", "unwind", "other"}
    if not isinstance(sections, dict) or set(sections) != expected_sections:
        raise BinarySizeError("witness sections have an unexpected shape")
    if any(not isinstance(value, int) or value < 0 for value in sections.values()):
        raise BinarySizeError("witness section sizes must be non-negative integers")
    if sum(sections.values()) != witness["bytes"]:
        raise BinarySizeError("witness section sizes do not reconcile")


def _validate_observation(observation: Any) -> None:
    if not isinstance(observation, dict):
        raise BinarySizeError("observation is not an object")
    _require_sha(observation.get("commit"), "observation commit")
    for key in ("repository", "ref", "run_id", "measured_at"):
        if not isinstance(observation.get(key), str):
            raise BinarySizeError(f"observation {key} must be a string")
    if not isinstance(observation.get("toolchain"), dict):
        raise BinarySizeError("observation toolchain must be an object")
    witnesses = observation.get("witnesses")
    if not isinstance(witnesses, list) or not witnesses:
        raise BinarySizeError("observation witnesses must be a non-empty array")
    ids = []
    for witness in witnesses:
        _validate_witness(witness)
        ids.append(witness["id"])
    if len(ids) != len(set(ids)):
        raise BinarySizeError("observation contains duplicate witness IDs")


def validate_current(current: Any) -> None:
    if not isinstance(current, dict) or current.get("schema") != SCHEMA:
        raise BinarySizeError(f"current record must use schema {SCHEMA}")
    if set(current) != {"schema", "observation"}:
        raise BinarySizeError("current record has an unexpected shape")
    _validate_observation(current["observation"])


def empty_history() -> dict[str, Any]:
    return {"schema": SCHEMA, "history": []}


def validate_history(history: Any) -> None:
    if not isinstance(history, dict) or history.get("schema") != SCHEMA:
        raise BinarySizeError(f"history must use schema {SCHEMA}")
    if set(history) != {"schema", "history"}:
        raise BinarySizeError("history record has an unexpected shape")
    observations = history["history"]
    if not isinstance(observations, list):
        raise BinarySizeError("history observations must be an array")
    commits = []
    for observation in observations:
        _validate_observation(observation)
        commits.append(observation["commit"])
    if len(commits) != len(set(commits)):
        raise BinarySizeError("history contains duplicate commits")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BinarySizeError(f"could not read JSON from {path}") from exc


def merge_history(
    history: dict[str, Any],
    current: dict[str, Any],
    *,
    update_main: bool,
) -> dict[str, Any]:
    validate_history(history)
    validate_current(current)
    merged = list(history["history"])
    if update_main:
        observation = current["observation"]
        merged = [item for item in merged if item["commit"] != observation["commit"]]
        merged.append(observation)
        merged = merged[-MAX_MAIN_OBSERVATIONS:]
    return {"schema": SCHEMA, "history": merged}


def _witness_by_id(
    observation: dict[str, Any], witness_id: str
) -> dict[str, Any] | None:
    return next(
        (
            witness
            for witness in observation["witnesses"]
            if witness["id"] == witness_id
        ),
        None,
    )


def compatible_baseline(
    history: dict[str, Any], current_witness: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for observation in reversed(history["history"]):
        witness = _witness_by_id(observation, current_witness["id"])
        if witness and witness["fingerprint"] == current_witness["fingerprint"]:
            return observation, witness
    return None


def sparkline(values: Sequence[int]) -> str:
    if not values:
        return "—"
    low = min(values)
    high = max(values)
    if low == high:
        return SPARKLINE_GLYPHS[len(SPARKLINE_GLYPHS) // 2] * len(values)
    scale = len(SPARKLINE_GLYPHS) - 1
    return "".join(
        SPARKLINE_GLYPHS[round((value - low) * scale / (high - low))]
        for value in values
    )


def _format_bytes(value: int, *, signed: bool = False) -> str:
    sign = ""
    magnitude = value
    if signed:
        sign = "+" if value >= 0 else "-"
        magnitude = abs(value)
    if magnitude >= 1024 * 1024:
        formatted = f"{magnitude / (1024 * 1024):.2f} MiB"
    elif magnitude >= 1024:
        formatted = f"{magnitude / 1024:.1f} KiB"
    else:
        formatted = f"{magnitude} B"
    return sign + formatted


def render_summary(history: dict[str, Any], current: dict[str, Any]) -> str:
    validate_history(history)
    validate_current(current)
    observation = current["observation"]
    rows = [
        "# Loom SDK product size",
        "",
        "| Product | Current artifact | Main baseline | Delta | Recent main |",
        "| --- | ---: | ---: | ---: | :--- |",
    ]
    for witness in observation["witnesses"]:
        baseline = compatible_baseline(history, witness)
        compatible_values = []
        for historical_observation in history["history"]:
            historical_witness = _witness_by_id(historical_observation, witness["id"])
            if (
                historical_witness
                and historical_witness["fingerprint"] == witness["fingerprint"]
            ):
                compatible_values.append(historical_witness["bytes"])
        compatible_values = compatible_values[-11:] + [witness["bytes"]]
        current_size = f"{_format_bytes(witness['bytes'])} ({witness['bytes']:,} B)"
        if baseline:
            _, baseline_witness = baseline
            baseline_bytes = baseline_witness["bytes"]
            difference = witness["bytes"] - baseline_bytes
            percentage = difference * 100 / baseline_bytes if baseline_bytes else 0.0
            baseline_size = _format_bytes(baseline_bytes)
            delta = f"{_format_bytes(difference, signed=True)} ({percentage:+.2f}%)"
        else:
            baseline_size = "new series"
            delta = "—"
        rows.append(
            f"| {witness['title']} | {current_size} | {baseline_size} | "
            f"{delta} | `{sparkline(compatible_values)}` |"
        )

    rows.extend(
        [
            "",
            "Sizes describe the exact wheel and its uncompressed members. Shared "
            "libraries are counted once in the native payload; installed totals "
            "include Python and metadata. No binary is rebuilt or stripped. "
            "ELF section detail is included in the observation. A changed profile, "
            "platform, or toolchain fingerprint starts a new series.",
            "",
            "<details>",
            "<summary>Current machine-readable observation</summary>",
            "",
            "```json",
            json.dumps(current, indent=2, sort_keys=True),
            "```",
            "",
            "</details>",
            "",
        ]
    )
    return "\n".join(rows)


def report(args: argparse.Namespace) -> None:
    current = _load_json(args.current)
    validate_current(current)
    history = _load_json(args.baseline) if args.baseline.is_file() else empty_history()
    validate_history(history)
    summary = render_summary(history, current)
    output_history = merge_history(history, current, update_main=args.update_main)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "current.json", current)
    _write_json(args.output_dir / "history.json", output_history)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")


def _github_json(url: str, token: str) -> Any:
    return github_api.get_json(url, token)


def _github_bytes(url: str, token: str) -> bytes:
    return github_api.get_bytes(url, token)


def list_main_artifacts(
    repository: str,
    artifact_name: str,
    token: str,
    *,
    get_json: Callable[[str, str], Any] = _github_json,
) -> list[dict[str, Any]]:
    artifacts = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {"name": artifact_name, "per_page": 100, "page": page}
        )
        url = f"https://api.github.com/repos/{repository}/actions/artifacts?{query}"
        payload = get_json(url, token)
        page_artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        if not isinstance(page_artifacts, list):
            raise BinarySizeError("GitHub artifact listing has an unexpected shape")
        for artifact in page_artifacts:
            workflow_run = artifact.get("workflow_run", {})
            if (
                artifact.get("name") == artifact_name
                and not artifact.get("expired", False)
                and workflow_run.get("head_branch") == "main"
            ):
                artifacts.append(artifact)
        if len(page_artifacts) < 100:
            return artifacts
        page += 1


def _git_distance(repo_root: Path, ancestor: str, anchor: str) -> int | None:
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, anchor],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode == 1:
        return None
    if ancestry.returncode != 0:
        raise BinarySizeError(
            f"git could not test ancestry for {ancestor} and {anchor}: "
            f"{ancestry.stderr.decode(errors='replace')}"
        )
    result = _run(
        ["git", "rev-list", "--count", f"{ancestor}..{anchor}"],
        cwd=repo_root,
        capture_output=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise BinarySizeError("git rev-list returned an invalid distance") from exc


def select_nearest_artifact(
    artifacts: Iterable[dict[str, Any]],
    *,
    anchor: str,
    exclude_commit: str | None,
    distance: Callable[[str, str], int | None],
) -> dict[str, Any] | None:
    _require_sha(anchor, "comparison anchor")
    if exclude_commit:
        _require_sha(exclude_commit, "excluded commit")
    candidates = []
    for artifact in artifacts:
        workflow_run = artifact.get("workflow_run", {})
        commit = workflow_run.get("head_sha")
        if commit == exclude_commit:
            continue
        try:
            commit = _require_sha(commit, "artifact head commit")
        except BinarySizeError:
            continue
        commit_distance = distance(commit, anchor)
        if commit_distance is not None:
            candidates.append(
                (commit_distance, artifact.get("created_at", ""), artifact)
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=False)
    nearest_distance = candidates[0][0]
    nearest = [item for item in candidates if item[0] == nearest_distance]
    return max(nearest, key=lambda item: item[1])[2]


def history_from_zip(payload: bytes, expected_head_sha: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = archive.namelist()
            if names.count("history.json") != 1:
                raise BinarySizeError("history artifact must contain one history.json")
            history = json.loads(archive.read("history.json"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise BinarySizeError("history artifact is not a valid history ZIP") from exc
    validate_history(history)
    if not history["history"]:
        raise BinarySizeError("main history artifact contains no observations")
    if history["history"][-1]["commit"] != expected_head_sha:
        raise BinarySizeError(
            "main history artifact head does not match its workflow commit"
        )
    return history


def fetch_history(args: argparse.Namespace) -> None:
    token = os.environ.get(args.token_env)
    if not token:
        raise BinarySizeError(
            f"{args.token_env} is required to read workflow artifacts"
        )
    artifacts = list_main_artifacts(
        args.repository,
        args.artifact_name,
        token,
    )
    selected = select_nearest_artifact(
        artifacts,
        anchor=args.anchor,
        exclude_commit=args.exclude_commit,
        distance=lambda ancestor, anchor: _git_distance(
            args.repo_root, ancestor, anchor
        ),
    )
    if selected is None:
        print("No compatible ancestor main artifact was found; starting a new history.")
        history = empty_history()
    else:
        workflow_run = selected["workflow_run"]
        head_sha = _require_sha(workflow_run.get("head_sha"), "artifact head commit")
        archive_url = selected.get("archive_download_url")
        if not isinstance(archive_url, str) or not archive_url:
            raise BinarySizeError("selected artifact has no archive download URL")
        history = history_from_zip(_github_bytes(archive_url, token), head_sha)
        print(
            f"Using main size history from {head_sha[:12]} "
            f"(artifact {selected.get('id')})."
        )
    _write_json(args.output, history)


def _add_collect_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("collect")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, action="append", required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--measured-at", default="")
    parser.set_defaults(func=collect)


def _add_fetch_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("fetch-history")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--repository", required=True)
    parser.add_argument("--artifact-name", default=MAIN_ARTIFACT_NAME)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--exclude-commit")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=fetch_history)


def _add_report_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("report")
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--update-main", action="store_true")
    parser.set_defaults(func=report)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    _add_collect_parser(subparsers)
    _add_fetch_parser(subparsers)
    _add_report_parser(subparsers)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    try:
        args = parse_arguments(argv)
        args.func(args)
        return 0
    except (
        BinarySizeError,
        ValueError,
        OSError,
        zipfile.BadZipFile,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"loom binary size: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
