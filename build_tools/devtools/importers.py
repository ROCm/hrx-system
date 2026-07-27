# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/licenses/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional importer Python environment helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from build_tools.devtools.command_plan import (
    CommandPlan,
    CommandStep,
    EnsureDirectoryStep,
    quote_command,
)
from build_tools.devtools.environment import (
    LOCAL_TMP_ROOT,
    REPO_ROOT,
    ToolEnvironment,
    venv_bin_dir,
)


@dataclass(frozen=True)
class ImporterEnvironmentSpec:
    name: str
    lock_file: Path
    import_modules: tuple[str, ...]
    bazel_config: str
    cmake_options: tuple[str, ...]

    @property
    def state_dir(self) -> Path:
        return LOCAL_TMP_ROOT / "importers" / self.name

    @property
    def root(self) -> Path:
        return self.state_dir / "venv"

    @property
    def manifest_path(self) -> Path:
        return self.state_dir / "environment.json"

    @property
    def python(self) -> Path:
        if os.name == "nt":
            return venv_bin_dir(self.root) / "python.exe"
        return venv_bin_dir(self.root) / "python"


IMPORTER_ENVIRONMENTS: dict[str, ImporterEnvironmentSpec] = {
    "mlir": ImporterEnvironmentSpec(
        name="mlir",
        lock_file=REPO_ROOT / "requirements-importers-mlir.lock.txt",
        import_modules=(
            "iree.compiler",
            "iree.compiler.ir",
            "iree.compiler.dialects.affine",
            "iree.compiler.dialects.amdgpu",
            "iree.compiler.dialects.arith",
            "iree.compiler.dialects.builtin",
            "iree.compiler.dialects.func",
            "iree.compiler.dialects.gpu",
            "iree.compiler.dialects.hal",
            "iree.compiler.dialects.iree_codegen",
            "iree.compiler.dialects.iree_gpu",
            "iree.compiler.dialects.memref",
            "iree.compiler.dialects.scf",
            "iree.compiler.dialects.vector",
        ),
        bazel_config="loom-importer-mlir",
        cmake_options=("LOOM_IMPORT_MLIR=ON",),
    ),
    "tilelang": ImporterEnvironmentSpec(
        name="tilelang",
        lock_file=REPO_ROOT / "requirements-importers-tilelang.lock.txt",
        import_modules=(
            "tilelang",
            "tilelang.language",
            "torch",
            "tvm",
            "tvm.tirx",
            "tvm.te",
            "tvm_ffi",
        ),
        bazel_config="loom-importer-tilelang",
        cmake_options=("LOOM_IMPORT_TILELANG=ON",),
    ),
}


def names() -> tuple[str, ...]:
    return tuple(sorted(IMPORTER_ENVIRONMENTS))


def spec(name: str) -> ImporterEnvironmentSpec:
    try:
        return IMPORTER_ENVIRONMENTS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown importer environment {name!r}; expected one of {', '.join(names())}"
        ) from exc


def parse_names(values: list[str]) -> tuple[str, ...]:
    parsed_names = []
    seen_names = set()
    for value in values:
        for name in value.split(","):
            name = name.strip()
            if not name:
                continue
            if name == "all":
                expanded_names = names()
            else:
                expanded_names = (spec(name).name,)
            for expanded_name in expanded_names:
                if expanded_name in seen_names:
                    continue
                parsed_names.append(expanded_name)
                seen_names.add(expanded_name)
    return tuple(parsed_names)


def extract_options(args: list[str]) -> tuple[tuple[str, ...], list[str]]:
    importer_values = []
    stripped_args = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--importer-env":
            index += 1
            if index >= len(args):
                raise ValueError("--importer-env requires a value")
            importer_values.append(args[index])
        elif arg.startswith("--importer-env="):
            importer_values.append(arg.split("=", 1)[1])
        else:
            stripped_args.append(arg)
        index += 1
    return parse_names(importer_values), stripped_args


def lock_hash(spec: ImporterEnvironmentSpec) -> str:
    import hashlib

    return hashlib.sha256(spec.lock_file.read_bytes()).hexdigest()


def load_manifest(spec: ImporterEnvironmentSpec) -> dict[str, object]:
    try:
        manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"importer environment {spec.name!r} is not set up; run "
            f"`python dev.py importers setup {spec.name}`"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"importer environment manifest is not valid JSON: {spec.manifest_path}"
        ) from exc
    if manifest.get("lock_sha256") != lock_hash(spec):
        raise ValueError(
            f"importer environment {spec.name!r} is stale relative to "
            f"{spec.lock_file.name}; run `python dev.py importers setup {spec.name}`"
        )
    if not manifest.get("ok"):
        raise ValueError(
            f"importer environment {spec.name!r} did not pass probes; run "
            f"`python dev.py importers doctor {spec.name}`"
        )
    site_packages = manifest.get("site_packages")
    if not isinstance(site_packages, str) or not Path(site_packages).is_dir():
        raise ValueError(
            f"importer environment {spec.name!r} manifest has no usable "
            f"site-packages path; run `python dev.py importers setup {spec.name}`"
        )
    return manifest


def selected_manifests(
    importer_names: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(load_manifest(spec(name)) for name in importer_names)


def pythonpath_for_manifests(manifests: tuple[dict[str, object], ...]) -> str:
    paths = []
    seen_paths = set()
    for manifest in manifests:
        site_packages = str(manifest["site_packages"])
        if site_packages in seen_paths:
            continue
        paths.append(site_packages)
        seen_paths.add(site_packages)
    return os.pathsep.join(paths)


def env_with_importers(
    importer_names: tuple[str, ...],
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    manifests = selected_manifests(importer_names)
    importer_pythonpath = pythonpath_for_manifests(manifests)
    if importer_pythonpath:
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            env["PYTHONPATH"] = importer_pythonpath + os.pathsep + existing_pythonpath
        else:
            env["PYTHONPATH"] = importer_pythonpath
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def bazel_configs(importer_names: tuple[str, ...]) -> list[str]:
    if len(importer_names) > 1:
        sorted_importer_names = tuple(sorted(importer_names))
        if sorted_importer_names == names():
            return ["--config=loom-importers"]
        return [
            f"--//loom/config/import:enable={','.join(sorted_importer_names)}",
        ]
    return [f"--config={spec(name).bazel_config}" for name in importer_names]


def bazel_test_env_args(importer_names: tuple[str, ...]) -> list[str]:
    manifests = selected_manifests(importer_names)
    pythonpath = pythonpath_for_manifests(manifests)
    if not pythonpath:
        return []
    return [
        f"--test_env=PYTHONPATH={pythonpath}",
        "--test_env=PYTHONDONTWRITEBYTECODE=1",
    ]


def bazel_configure_args(importer_names: tuple[str, ...]) -> list[str]:
    defines = []
    for importer_name in importer_names:
        for option in spec(importer_name).cmake_options:
            defines.append("-D" + option)
    return defines


def cmake_configure_args(importer_names: tuple[str, ...]) -> list[str]:
    args = []
    for importer_name in importer_names:
        for option in spec(importer_name).cmake_options:
            args.append("-D" + option)
    return args


class ImporterProbeStep:
    def __init__(self, spec: ImporterEnvironmentSpec) -> None:
        self.spec = spec

    def describe(self) -> str:
        return f"# probe importer environment {self.spec.name} and write {self.spec.manifest_path}"

    def run(self, verbose: bool = False) -> int:
        if not self.spec.python.is_file():
            print(
                f"dev.py: importer environment {self.spec.name!r} has no Python: "
                f"{self.spec.python}",
                file=sys.stderr,
            )
            return 1
        command = [
            str(self.spec.python),
            "-c",
            _PROBE_SCRIPT,
            self.spec.name,
            str(self.spec.lock_file),
            lock_hash(self.spec),
            str(self.spec.root),
            str(self.spec.manifest_path),
            json.dumps(self.spec.import_modules),
        ]
        if verbose:
            print("dev.py: " + quote_command(command))
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = result.stdout.rstrip()
        if output:
            print(output)
        return result.returncode


class ImporterManifestStep:
    def __init__(self, spec: ImporterEnvironmentSpec, *, output_format: str) -> None:
        self.spec = spec
        self.output_format = output_format

    def describe(self) -> str:
        return f"# print importer environment {self.spec.name} manifest"

    def run(self, verbose: bool = False) -> int:
        del verbose
        try:
            manifest = load_manifest(self.spec)
        except ValueError as exc:
            print(f"dev.py: {exc}", file=sys.stderr)
            return 1
        if self.output_format == "json":
            print(json.dumps(manifest, indent=2, sort_keys=True))
        elif self.output_format == "shell":
            print(f"PYTHONPATH={manifest['site_packages']}")
        else:
            print(f"dev.py: unsupported env output format: {self.output_format}")
            return 1
        return 0


def setup_plan(importer_name: str, tool_env: ToolEnvironment) -> CommandPlan:
    importer_spec = spec(importer_name)
    return CommandPlan(
        [
            EnsureDirectoryStep(importer_spec.state_dir),
            CommandStep(
                [
                    tool_env.python,
                    "-m",
                    "venv",
                    str(importer_spec.root),
                ],
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label=f"create {importer_name} importer environment",
            ),
            CommandStep(
                [
                    str(importer_spec.python),
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "-r",
                    str(importer_spec.lock_file),
                ],
                cwd=REPO_ROOT,
                env={
                    **tool_env.path_env(),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                label=f"install {importer_name} importer packages",
            ),
            ImporterProbeStep(importer_spec),
        ]
    )


def doctor_plan(importer_name: str) -> CommandPlan:
    return CommandPlan([ImporterProbeStep(spec(importer_name))])


def env_plan(importer_name: str, *, output_format: str) -> CommandPlan:
    return CommandPlan(
        [ImporterManifestStep(spec(importer_name), output_format=output_format)]
    )


_PROBE_SCRIPT = r"""
import importlib
import importlib.metadata
import json
import sys
import sysconfig
from pathlib import Path

name = sys.argv[1]
lock_file = Path(sys.argv[2])
lock_hash = sys.argv[3]
root = Path(sys.argv[4])
manifest_path = Path(sys.argv[5])
modules = json.loads(sys.argv[6])

module_results = {}
ok = True
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        ok = False
        module_results[module_name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        module_results[module_name] = {
            "ok": True,
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }

packages = {}
for distribution in importlib.metadata.distributions():
    package_name = distribution.metadata.get("Name")
    if package_name:
        packages[package_name.lower()] = distribution.version

manifest = {
    "name": name,
    "ok": ok,
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "root": str(root),
    "site_packages": sysconfig.get_paths()["purelib"],
    "lock_file": str(lock_file),
    "lock_sha256": lock_hash,
    "modules": module_results,
    "packages": packages,
}

manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if ok:
    print(f"Importer environment {name!r} is ready: {manifest['site_packages']}")
else:
    print(f"Importer environment {name!r} failed probes; see {manifest_path}", file=sys.stderr)
    sys.exit(1)
"""
