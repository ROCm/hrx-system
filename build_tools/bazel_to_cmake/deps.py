#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates the CMake dependency lock from Bazel module fragments.

The Bazel module graph is the source of truth for third-party source
dependencies. This tool reads the root `MODULE.bazel` include graph, collects
source dependencies from `*/third_party/deps.MODULE.bazel` fragments, resolves
Bazel Central Registry source metadata for `bazel_dep()` entries, and writes a
root `MODULE.cmake.lock` file consumed by CMake FetchContent adapters.

`--check` is deliberately offline. It validates the Bazel fragments against the
checked-in lock file and fails if BCR metadata would need to be resolved.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import re
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_REGISTRY_URL = "https://bcr.bazel.build"
ROOT_MODULE_FILE = "MODULE.bazel"
ROOT_CMAKE_LOCK_FILE = "MODULE.cmake.lock"


@dataclasses.dataclass(frozen=True)
class Dependency:
    """A source dependency declared in a third-party Bazel module fragment."""

    name: str
    kind: str
    owner: str
    module_name: str
    repo_name: str
    version: str
    dev_dependency: bool
    urls: tuple[str, ...] = ()
    sha256: str = ""
    strip_prefix: str = ""
    source_url: str = ""
    build_file: str = ""


class DummyModuleExtension:
    """No-op object used for Bazel module extension declarations."""

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return self._ignore

    def _ignore(self, **_kwargs: Any) -> None:
        return None


class ModuleParser:
    """Executes the Python-shaped subset of root Bazel module files."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.dependencies: list[Dependency] = []
        self._active_files: set[Path] = set()
        self._parsed_files: set[Path] = set()

    def parse(self, module_file: Path) -> list[Dependency]:
        self._parse_file(module_file.resolve())
        return self.dependencies

    def _parse_file(self, module_file: Path) -> None:
        if module_file in self._parsed_files:
            return
        if module_file in self._active_files:
            relpath = self._repo_relpath(module_file)
            raise ValueError(f"MODULE include cycle involving {relpath}")
        self._active_files.add(module_file)
        collect_source_deps = self._is_third_party_dependency_fragment(module_file)
        globals_map = self._globals_for_file(module_file, collect_source_deps)
        try:
            code = module_file.read_text(encoding="utf-8")
            exec(compile(code, str(module_file), "exec"), globals_map, {})
        finally:
            self._active_files.remove(module_file)
        self._parsed_files.add(module_file)

    def _globals_for_file(
        self,
        module_file: Path,
        collect_source_deps: bool,
    ) -> dict[str, Any]:
        return {
            "__builtins__": {},
            "bazel_dep": lambda **kwargs: self._bazel_dep(
                module_file, collect_source_deps, **kwargs
            ),
            "include": lambda label: self._include(label),
            "local_path_override": self._ignore,
            "module": self._ignore,
            "multiple_version_override": self._ignore,
            "override_repo": self._ignore,
            "single_version_override": self._ignore,
            "use_extension": lambda *_args, **_kwargs: DummyModuleExtension(),
            "use_repo": self._ignore,
            "use_repo_rule": lambda repo_rule_label, repo_rule_name: self._repo_rule(
                module_file,
                collect_source_deps,
                repo_rule_label=repo_rule_label,
                repo_rule_name=repo_rule_name,
            ),
        }

    def _include(self, label: str) -> None:
        self._parse_file(self._resolve_label(label))

    def _bazel_dep(
        self,
        module_file: Path,
        collect_source_deps: bool,
        **kwargs: Any,
    ) -> None:
        if not collect_source_deps:
            return
        _validate_known_fields(
            kwargs,
            context="bazel_dep",
            known_fields={
                "dev_dependency",
                "max_compatibility_level",
                "name",
                "repo_name",
                "version",
            },
        )
        name = _required_string(kwargs, "name", "bazel_dep")
        version = _required_string(kwargs, "version", f"bazel_dep({name})")
        repo_name = _optional_string(kwargs, "repo_name", name)
        dev_dependency = bool(kwargs.get("dev_dependency", False))
        self.dependencies.append(
            Dependency(
                name=name,
                kind="bazel_dep",
                owner=self._owner_for_module_file(module_file),
                module_name=name,
                repo_name=repo_name,
                version=version,
                dev_dependency=dev_dependency,
            )
        )

    def _repo_rule(
        self,
        module_file: Path,
        collect_source_deps: bool,
        *,
        repo_rule_label: str,
        repo_rule_name: str,
    ) -> Callable[..., None]:
        if repo_rule_name == "http_archive":
            return lambda **kwargs: self._http_archive(
                module_file, collect_source_deps, **kwargs
            )
        if repo_rule_name == "rocm_repository":
            return lambda **kwargs: self._rocm_repository(
                module_file, collect_source_deps, **kwargs
            )
        if collect_source_deps:
            relpath = self._repo_relpath(module_file)
            raise ValueError(
                f"{relpath} uses unsupported repo rule {repo_rule_name} "
                f"from {repo_rule_label}; teach deps.py how to lock it"
            )
        return self._ignore

    def _http_archive(
        self,
        module_file: Path,
        collect_source_deps: bool,
        **kwargs: Any,
    ) -> None:
        if not collect_source_deps:
            return
        _validate_known_fields(
            kwargs,
            context="http_archive",
            known_fields={
                "build_file",
                "integrity",
                "name",
                "sha256",
                "strip_prefix",
                "url",
                "urls",
            },
        )
        name = _required_string(kwargs, "name", "http_archive")
        urls = _urls_from_kwargs(kwargs, f"http_archive({name})")
        sha256 = _sha256_from_kwargs(kwargs, f"http_archive({name})")
        strip_prefix = _optional_string(kwargs, "strip_prefix", "")
        build_file = _optional_string(kwargs, "build_file", "")
        self.dependencies.append(
            Dependency(
                name=name,
                kind="http_archive",
                owner=self._owner_for_module_file(module_file),
                module_name=name,
                repo_name=name,
                version="",
                dev_dependency=False,
                urls=tuple(urls),
                sha256=sha256,
                strip_prefix=strip_prefix,
                build_file=build_file,
            )
        )

    def _rocm_repository(
        self,
        module_file: Path,
        collect_source_deps: bool,
        **kwargs: Any,
    ) -> None:
        if not collect_source_deps:
            return
        _validate_known_fields(
            kwargs,
            context="rocm_repository",
            known_fields={
                "build_file",
                "display_name",
                "integrity",
                "name",
                "sha256",
                "strip_prefix",
                "url",
                "urls",
            },
        )
        name = _required_string(kwargs, "name", "rocm_repository")
        if "url" not in kwargs and "urls" not in kwargs:
            return
        urls = _urls_from_kwargs(kwargs, f"rocm_repository({name})")
        sha256 = _sha256_from_kwargs(kwargs, f"rocm_repository({name})")
        strip_prefix = _optional_string(kwargs, "strip_prefix", "")
        build_file = _optional_string(kwargs, "build_file", "")
        self.dependencies.append(
            Dependency(
                name=name,
                kind="rocm_repository",
                owner=self._owner_for_module_file(module_file),
                module_name=name,
                repo_name=name,
                version="",
                dev_dependency=False,
                urls=tuple(urls),
                sha256=sha256,
                strip_prefix=strip_prefix,
                build_file=build_file,
            )
        )

    def _resolve_label(self, label: str) -> Path:
        match = re.fullmatch(r"//([^:]*):(.+)", label)
        if not match:
            raise ValueError(f"unsupported MODULE include label: {label}")
        package, filename = match.groups()
        return (self.repo_root / package / filename).resolve()

    def _repo_relpath(self, path: Path) -> str:
        return path.resolve().relative_to(self.repo_root.resolve()).as_posix()

    def _is_third_party_dependency_fragment(self, path: Path) -> bool:
        relpath = self._repo_relpath(path)
        return relpath.endswith("build_tools/third_party/deps.MODULE.bazel")

    def _owner_for_module_file(self, path: Path) -> str:
        relpath = self._repo_relpath(path)
        suffix = "/build_tools/third_party/deps.MODULE.bazel"
        if relpath == "build_tools/third_party/deps.MODULE.bazel":
            return "shared"
        if relpath.endswith(suffix):
            return relpath[: -len(suffix)]
        raise ValueError(f"cannot determine dependency owner for {relpath}")

    def _ignore(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class LockResolver:
    """Resolves source metadata for dependencies."""

    def __init__(self, registry_url: str, existing_lock: dict[str, dict[str, Any]]):
        self.registry_url = registry_url.rstrip("/")
        self.existing_lock = existing_lock

    def resolve_for_update(self, dependency: Dependency) -> Dependency:
        if dependency.kind in {"http_archive", "rocm_repository"}:
            return dependency
        if dependency.kind != "bazel_dep":
            raise ValueError(f"unsupported dependency kind: {dependency.kind}")
        source_url = self._bcr_source_url(dependency)
        source = self._read_json_url(source_url)
        urls = _source_urls(source, dependency.name)
        sha256 = _sha256_from_source(source, dependency.name)
        strip_prefix = _optional_json_string(source, "strip_prefix", "")
        return dataclasses.replace(
            dependency,
            urls=tuple(urls),
            sha256=sha256,
            strip_prefix=strip_prefix,
            source_url=source_url,
        )

    def resolve_for_check(self, dependency: Dependency) -> Dependency:
        if dependency.kind in {"http_archive", "rocm_repository"}:
            return dependency
        if dependency.kind != "bazel_dep":
            raise ValueError(f"unsupported dependency kind: {dependency.kind}")
        lock_entry = self.existing_lock.get(dependency.name)
        if not lock_entry:
            raise ValueError(
                f"{dependency.name} is missing from {ROOT_CMAKE_LOCK_FILE}; "
                "run deps.py update mode to resolve BCR metadata"
            )
        _validate_lock_identity(dependency, lock_entry)
        urls = tuple(lock_entry.get("URLS", ()))
        sha256 = str(lock_entry.get("SHA256", ""))
        if not urls or not sha256:
            raise ValueError(
                f"{dependency.name} is missing locked BCR source metadata in "
                f"{ROOT_CMAKE_LOCK_FILE}; run deps.py update mode"
            )
        _validate_sha256(sha256, dependency.name)
        return dataclasses.replace(
            dependency,
            urls=urls,
            sha256=sha256,
            strip_prefix=str(lock_entry.get("STRIP_PREFIX", "")),
            source_url=str(lock_entry.get("SOURCE_URL", "")),
        )

    def _bcr_source_url(self, dependency: Dependency) -> str:
        return (
            f"{self.registry_url}/modules/{dependency.module_name}/"
            f"{dependency.version}/source.json"
        )

    def _read_json_url(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "iree-deps-lock/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        except urllib.error.URLError as exc:
            raise ValueError(f"failed to resolve {url}: {exc}") from exc
        try:
            decoded = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON from {url}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"{url} did not contain a JSON object")
        return decoded


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or verify the root CMake dependency lock."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify MODULE.cmake.lock without writing files or making network "
            "requests. Fails if BCR metadata is missing from the checked-in lock."
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print generated lock contents without writing MODULE.cmake.lock.",
    )
    parser.add_argument(
        "--module",
        default=ROOT_MODULE_FILE,
        help="Root Bazel module file to parse.",
    )
    parser.add_argument(
        "--output",
        default=ROOT_CMAKE_LOCK_FILE,
        help="CMake lock file to write or verify.",
    )
    parser.add_argument(
        "--registry_url",
        default=DEFAULT_REGISTRY_URL,
        help="Bazel registry URL used to resolve bazel_dep source metadata.",
    )
    args = parser.parse_args()
    if args.check and args.preview:
        parser.error("--check and --preview are mutually exclusive")
    return args


def main() -> int:
    sys.dont_write_bytecode = True
    args = parse_arguments()
    repo_root = Path.cwd()
    module_path = (repo_root / args.module).resolve()
    output_path = (repo_root / args.output).resolve()

    parser = ModuleParser(repo_root)
    dependencies = parser.parse(module_path)
    _validate_unique_dependencies(dependencies)

    existing_lock = parse_cmake_lock(output_path) if output_path.exists() else {}
    resolver = LockResolver(args.registry_url, existing_lock)
    if args.check:
        resolved = [resolver.resolve_for_check(dep) for dep in dependencies]
    else:
        resolved = [resolver.resolve_for_update(dep) for dep in dependencies]

    contents = render_cmake_lock(resolved)
    if args.preview:
        print(contents, end="")
        return 0
    if args.check:
        if not output_path.exists():
            print(f"ERROR: {args.output} does not exist", file=sys.stderr)
            return 1
        existing_contents = output_path.read_text(encoding="utf-8")
        if existing_contents != contents:
            print(f"ERROR: {args.output} is stale; run deps.py", file=sys.stderr)
            return 1
        return 0

    output_path.write_text(contents, encoding="utf-8")
    return 0


def parse_cmake_lock(path: Path) -> dict[str, dict[str, Any]]:
    """Parses the generated CMake lock well enough for offline check mode."""

    if not path.exists():
        return {}
    variables = _parse_cmake_sets(path.read_text(encoding="utf-8"))
    dependency_ids = variables.get("IREE_DEPENDENCY_IDS", [])
    if isinstance(dependency_ids, str):
        dependency_ids = [dependency_ids]
    entries: dict[str, dict[str, Any]] = {}
    for dependency_id in dependency_ids:
        prefix = f"IREE_DEP_{_cmake_identifier(dependency_id)}_"
        entry: dict[str, Any] = {}
        for variable_name, value in variables.items():
            if variable_name.startswith(prefix):
                entry[variable_name[len(prefix) :]] = value
        entries[dependency_id] = entry
    return entries


def _parse_cmake_sets(contents: str) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    lines = iter(contents.splitlines())
    for line in lines:
        inline_match = re.fullmatch(
            r'set\(([A-Za-z0-9_]+) "((?:[^"\\]|\\.)*)"\)',
            line,
        )
        if inline_match:
            variables[inline_match.group(1)] = _cmake_unescape(inline_match.group(2))
            continue

        multiline_match = re.fullmatch(r"set\(([A-Za-z0-9_]+)", line)
        if not multiline_match:
            continue
        name = multiline_match.group(1)
        values = []
        for value_line in lines:
            if value_line == ")":
                break
            value_match = re.fullmatch(r'  "((?:[^"\\]|\\.)*)"', value_line)
            if not value_match:
                raise ValueError(f"cannot parse MODULE.cmake.lock line: {value_line}")
            values.append(_cmake_unescape(value_match.group(1)))
        variables[name] = values
    return variables


def render_cmake_lock(dependencies: Iterable[Dependency]) -> str:
    dependencies = list(dependencies)
    lines = [
        "# Copyright 2026 The IREE Authors",
        "#",
        "# Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "# See https://llvm.org/LICENSE.txt for license information.",
        "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "# Generated by build_tools/bazel_to_cmake/deps.py. Do not edit.",
        "#",
        "# Source dependency declarations live in owner-scoped deps.MODULE.bazel",
        "# fragments included by MODULE.bazel. Run this generator in update mode",
        "# when those declarations change.",
        "",
        "set(IREE_DEPENDENCY_IDS",
    ]
    for dependency in dependencies:
        lines.append(f'  "{_cmake_escape(dependency.name)}"')
    lines.extend([")", ""])

    for dependency in dependencies:
        identifier = _cmake_identifier(dependency.name)
        _append_cmake_scalar(lines, identifier, "KIND", dependency.kind)
        _append_cmake_scalar(lines, identifier, "OWNER", dependency.owner)
        _append_cmake_scalar(lines, identifier, "MODULE_NAME", dependency.module_name)
        _append_cmake_scalar(lines, identifier, "REPO_NAME", dependency.repo_name)
        _append_cmake_scalar(lines, identifier, "VERSION", dependency.version)
        _append_cmake_scalar(
            lines,
            identifier,
            "DEV_DEPENDENCY",
            "TRUE" if dependency.dev_dependency else "FALSE",
        )
        _append_cmake_scalar(lines, identifier, "SOURCE_URL", dependency.source_url)
        _append_cmake_list(lines, identifier, "URLS", dependency.urls)
        _append_cmake_scalar(lines, identifier, "SHA256", dependency.sha256)
        _append_cmake_scalar(lines, identifier, "STRIP_PREFIX", dependency.strip_prefix)
        _append_cmake_scalar(lines, identifier, "BUILD_FILE", dependency.build_file)
        lines.append("")
    return "\n".join(lines)


def _append_cmake_scalar(
    lines: list[str],
    dependency_identifier: str,
    property_name: str,
    value: str,
) -> None:
    lines.append(
        f'set(IREE_DEP_{dependency_identifier}_{property_name} "{_cmake_escape(value)}")'
    )


def _append_cmake_list(
    lines: list[str],
    dependency_identifier: str,
    property_name: str,
    values: Iterable[str],
) -> None:
    lines.append(f"set(IREE_DEP_{dependency_identifier}_{property_name}")
    for value in values:
        lines.append(f'  "{_cmake_escape(value)}"')
    lines.append(")")


def _validate_unique_dependencies(dependencies: Iterable[Dependency]) -> None:
    seen: dict[str, Dependency] = {}
    for dependency in dependencies:
        existing = seen.get(dependency.name)
        if existing:
            raise ValueError(
                textwrap.dedent(
                    f"""\
                    duplicate source dependency {dependency.name}:
                      {existing.owner}: {existing.kind}
                      {dependency.owner}: {dependency.kind}
                    promote the dependency to a shared owner instead of declaring it twice
                    """
                ).strip()
            )
        seen[dependency.name] = dependency


def _validate_lock_identity(
    dependency: Dependency,
    lock_entry: dict[str, Any],
) -> None:
    expected = {
        "KIND": dependency.kind,
        "OWNER": dependency.owner,
        "MODULE_NAME": dependency.module_name,
        "REPO_NAME": dependency.repo_name,
        "VERSION": dependency.version,
        "DEV_DEPENDENCY": "TRUE" if dependency.dev_dependency else "FALSE",
    }
    for field_name, expected_value in expected.items():
        actual_value = str(lock_entry.get(field_name, ""))
        if actual_value != expected_value:
            raise ValueError(
                f"{dependency.name} lock field {field_name} is {actual_value!r}; "
                f"expected {expected_value!r}"
            )


def _required_string(kwargs: dict[str, Any], name: str, context: str) -> str:
    value = kwargs.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires string field {name}")
    return value


def _optional_string(kwargs: dict[str, Any], name: str, default: str) -> str:
    value = kwargs.get(name, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_json_string(source: dict[str, Any], name: str, default: str) -> str:
    value = source.get(name, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"source.json field {name} must be a string")
    return value


def _urls_from_kwargs(kwargs: dict[str, Any], context: str) -> list[str]:
    if "urls" in kwargs:
        urls = kwargs["urls"]
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError(f"{context} requires urls to be a list of strings")
        if not urls:
            raise ValueError(f"{context} requires at least one URL")
        return urls
    if "url" in kwargs:
        url = kwargs["url"]
        if not isinstance(url, str) or not url:
            raise ValueError(f"{context} requires url to be a string")
        return [url]
    raise ValueError(f"{context} requires url or urls")


def _source_urls(source: dict[str, Any], dependency_name: str) -> list[str]:
    if "urls" in source:
        urls = source["urls"]
        if (
            isinstance(urls, list)
            and all(isinstance(url, str) for url in urls)
            and urls
        ):
            return urls
        raise ValueError(f"BCR source for {dependency_name} has invalid urls")
    url = source.get("url")
    if isinstance(url, str) and url:
        return [url]
    raise ValueError(f"BCR source for {dependency_name} does not declare url or urls")


def _sha256_from_kwargs(kwargs: dict[str, Any], context: str) -> str:
    sha256 = kwargs.get("sha256")
    if isinstance(sha256, str) and sha256:
        _validate_sha256(sha256, context)
        return sha256
    integrity = kwargs.get("integrity")
    if isinstance(integrity, str) and integrity:
        return _sha256_from_integrity(integrity)
    raise ValueError(f"{context} requires sha256 or integrity")


def _sha256_from_source(source: dict[str, Any], dependency_name: str) -> str:
    sha256 = source.get("sha256")
    if isinstance(sha256, str) and sha256:
        _validate_sha256(sha256, dependency_name)
        return sha256
    integrity = source.get("integrity")
    if isinstance(integrity, str) and integrity:
        return _sha256_from_integrity(integrity)
    raise ValueError(f"BCR source for {dependency_name} has no sha256/integrity")


def _sha256_from_integrity(integrity: str) -> str:
    prefix = "sha256-"
    if not integrity.startswith(prefix):
        raise ValueError(f"unsupported integrity format: {integrity}")
    try:
        digest = base64.b64decode(integrity[len(prefix) :], validate=True)
    except ValueError as exc:
        raise ValueError(f"invalid sha256 integrity: {integrity}") from exc
    if len(digest) != hashlib.sha256().digest_size:
        raise ValueError(f"invalid sha256 integrity length: {integrity}")
    return digest.hex()


def _validate_sha256(sha256: str, context: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError(f"{context} requires a lowercase hex SHA256 digest")


def _validate_known_fields(
    kwargs: dict[str, Any],
    *,
    context: str,
    known_fields: set[str],
) -> None:
    unknown_fields = sorted(set(kwargs) - known_fields)
    if unknown_fields:
        raise ValueError(
            f"{context} has unsupported fields: {', '.join(unknown_fields)}"
        )


def _cmake_identifier(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", name.upper())


def _cmake_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _cmake_unescape(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
