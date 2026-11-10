# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Derives dependency identity from Bazel and source declarations."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from build_tools.bazel_to_cmake.deps import Dependency

SCHEMA = "hrx.dependency.inventory.v0"

# These dependencies support repository development and exported test APIs but
# do not participate in a shipping runtime or compiler artifact.
_DEVELOPMENT_SOURCE_DEPENDENCIES = frozenset(
    {
        "catch2",
        "google_benchmark",
        "googletest",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Declaration:
    """One canonical declaration contributing to a dependency identity."""

    name: str
    kind: str
    owner: str
    source_location: str
    urls: tuple[str, ...] = ()
    sha256: str = ""
    patches: tuple[str, ...] = ()

    def as_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "owner": self.owner,
            "source_location": self.source_location,
        }
        if self.urls:
            value["urls"] = list(self.urls)
        if self.sha256:
            value["sha256"] = self.sha256
        if self.patches:
            value["patches"] = list(self.patches)
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class InventoryEntry:
    """One versioned package identity derived from canonical declarations."""

    package_url: str
    scope: str
    declarations: tuple[Declaration, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "package_url": self.package_url,
            "relationship": "direct",
            "scope": self.scope,
            "declarations": [value.as_json() for value in self.declarations],
        }


@dataclasses.dataclass(slots=True)
class _EntryBuilder:
    package_url: str
    scope: str = "development"
    declarations: dict[tuple[object, ...], Declaration] = dataclasses.field(
        default_factory=dict
    )

    def add(self, declaration: Declaration, scope: str) -> None:
        if scope == "runtime":
            self.scope = "runtime"
        key = (
            declaration.name,
            declaration.kind,
            declaration.owner,
            declaration.source_location,
            declaration.urls,
            declaration.sha256,
            declaration.patches,
        )
        self.declarations[key] = declaration

    def finish(self) -> InventoryEntry:
        return InventoryEntry(
            package_url=self.package_url,
            scope=self.scope,
            declarations=tuple(
                sorted(
                    self.declarations.values(),
                    key=lambda value: (
                        value.source_location,
                        value.name,
                        value.kind,
                    ),
                )
            ),
        )


def build_inventory(
    bazel_graph: Mapping[str, Any],
    source_dependencies: Iterable[Dependency],
) -> dict[str, object]:
    """Builds an ephemeral inventory without duplicating checked-in versions."""

    source_dependencies = tuple(source_dependencies)
    source_bazel_versions = {
        dependency.module_name: dependency.version
        for dependency in source_dependencies
        if dependency.kind == "bazel_dep"
    }
    builders: dict[str, _EntryBuilder] = {}
    graph_versions: dict[str, str] = {}

    for module in _direct_bazel_modules(bazel_graph):
        name = module["name"]
        version = module["version"]
        if name in graph_versions:
            raise ValueError(f"duplicate direct Bazel module {name!r}")
        graph_versions[name] = version
        declared_version = source_bazel_versions.get(name)
        if declared_version is not None and declared_version != version:
            raise ValueError(
                f"Bazel resolved {name}@{version}, but the source dependency "
                f"parser resolved {name}@{declared_version}"
            )
        package_url = _bazel_package_url(name, version)
        scope = (
            "runtime"
            if name in source_bazel_versions
            and name not in _DEVELOPMENT_SOURCE_DEPENDENCIES
            else "development"
        )
        _builder(builders, package_url).add(
            Declaration(
                name=str(module.get("apparentName") or name),
                kind="bazel_module",
                owner="root",
                source_location="MODULE.bazel",
                urls=(
                    f"https://registry.bazel.build/modules/"
                    f"{quote(name, safe='._~-')}/{quote(version, safe='._~-')}",
                ),
            ),
            scope,
        )

    for dependency in source_dependencies:
        if dependency.kind == "bazel_dep":
            graph_version = graph_versions.get(dependency.module_name)
            if graph_version is None:
                raise ValueError(
                    f"source dependency {dependency.module_name!r} is absent "
                    "from the resolved direct Bazel graph"
                )
        package_url = source_package_url(dependency)
        scope = (
            "development"
            if dependency.name in _DEVELOPMENT_SOURCE_DEPENDENCIES
            else "runtime"
        )
        _builder(builders, package_url).add(
            Declaration(
                name=dependency.name,
                kind=dependency.kind,
                owner=dependency.owner,
                source_location=_source_location(dependency.owner),
                urls=dependency.urls,
                sha256=dependency.sha256,
                patches=dependency.patches,
            ),
            scope,
        )

    dependencies = {
        package_url: builders[package_url].finish().as_json()
        for package_url in sorted(builders)
    }
    return {
        "schema": SCHEMA,
        "dependencies": dependencies,
    }


def source_package_url(dependency: Dependency) -> str:
    """Returns the package identity represented by a source declaration."""

    if dependency.kind == "bazel_dep":
        return _bazel_package_url(dependency.module_name, dependency.version)

    for url in dependency.urls:
        parsed = urlsplit(url)
        if parsed.netloc.lower() == "github.com":
            match = re.fullmatch(
                r"/([^/]+)/([^/]+)/archive/(?:refs/tags/)?(.+)",
                unquote(parsed.path),
            )
            if match:
                owner, repository, archive_name = match.groups()
                version = _strip_archive_suffix(archive_name)
                return _github_package_url(owner, repository, version)
        if parsed.netloc.lower() == "raw.githubusercontent.com":
            match = re.match(r"/([^/]+)/([^/]+)/([^/]+)/", unquote(parsed.path))
            if match:
                owner, repository, revision = match.groups()
                return _github_package_url(owner, repository, revision)

    if dependency.name == "libdrm_headers":
        match = re.fullmatch(r"libdrm-(.+)", dependency.strip_prefix)
        if not match:
            raise ValueError(
                "libdrm_headers requires a versioned libdrm-* strip_prefix"
            )
        return _generic_package_url("libdrm", match.group(1))

    raise ValueError(
        f"cannot derive a package identity for source dependency "
        f"{dependency.name!r} from {dependency.urls!r}"
    )


def validate_inventory(value: Mapping[str, Any]) -> None:
    """Validates the stable boundary consumed by inventory projections."""

    if value.get("schema") != SCHEMA:
        raise ValueError(f"expected inventory schema {SCHEMA!r}")
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ValueError("inventory dependencies must be an object")
    for key, dependency in dependencies.items():
        if not isinstance(key, str) or not isinstance(dependency, dict):
            raise ValueError("inventory dependencies must map strings to objects")
        if dependency.get("package_url") != key:
            raise ValueError(f"inventory key does not match package_url: {key!r}")
        if dependency.get("relationship") != "direct":
            raise ValueError(f"dependency {key!r} is not direct")
        if dependency.get("scope") not in {"runtime", "development"}:
            raise ValueError(f"dependency {key!r} has an invalid scope")
        declarations = dependency.get("declarations")
        if not isinstance(declarations, list) or not declarations:
            raise ValueError(f"dependency {key!r} has no declarations")


def _direct_bazel_modules(graph: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if graph.get("root") is not True:
        raise ValueError("Bazel dependency graph must describe the root module")
    dependencies = graph.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("Bazel dependency graph has no direct dependencies")
    modules: list[Mapping[str, Any]] = []
    for module in dependencies:
        if not isinstance(module, dict):
            raise ValueError("Bazel direct dependency must be an object")
        name = module.get("name")
        version = module.get("version")
        if not isinstance(name, str) or not name:
            raise ValueError("Bazel direct dependency has no name")
        if not isinstance(version, str) or not version:
            raise ValueError(f"Bazel direct dependency {name!r} has no version")
        modules.append(module)
    return tuple(modules)


def _builder(builders: dict[str, _EntryBuilder], package_url: str) -> _EntryBuilder:
    return builders.setdefault(package_url, _EntryBuilder(package_url))


def _source_location(owner: str) -> str:
    if owner == "shared":
        return "build_tools/third_party/deps.MODULE.bazel"
    return f"{owner}/build_tools/third_party/deps.MODULE.bazel"


def _bazel_package_url(name: str, version: str) -> str:
    # GitHub's dependency-submission API currently rejects registered
    # pkg:bazel identities. Preserve the ecosystem honestly as a generic
    # namespace until the API accepts the registered type.
    return f"pkg:generic/bazel/{quote(name, safe='._~-')}@{quote(version, safe='._~-')}"


def _github_package_url(owner: str, repository: str, version: str) -> str:
    repository = repository.removesuffix(".git")
    return (
        "pkg:github/"
        f"{quote(owner, safe='._~-')}/{quote(repository, safe='._~-')}"
        f"@{quote(version, safe='._~-')}"
    )


def _generic_package_url(name: str, version: str) -> str:
    return f"pkg:generic/{quote(name, safe='._~-')}@{quote(version, safe='._~-')}"


def _strip_archive_suffix(value: str) -> str:
    for suffix in (".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".zip"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    raise ValueError(f"unsupported GitHub archive suffix: {value!r}")
