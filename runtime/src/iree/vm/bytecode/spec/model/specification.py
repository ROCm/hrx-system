# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specification identity, version, dependency, and projection model.

Numeric identities are always explicit in concrete specification entities.
This module owns the VM version domains and the graph mechanics used to validate
and project concrete entities. It deliberately contains no VM opcodes, section
identifiers, or execution behavior.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping

_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_DOMAIN_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SPECIFICATION_PATTERN = re.compile(r"[a-z][a-z0-9_.]*")

CORE_PAGE_ID = 0x00
ARCHITECTURAL_EXTENSION_PAGE_MIN = 0xF0
ARCHITECTURAL_EXTENSION_PAGE_MAX = 0xFD
RESERVED_EXPERIMENT_PAGE_ID = 0xFE
RESERVED_EXTENDED_ESCAPE_PAGE_ID = 0xFF


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class Version:
    """One exact major/minor release in a named compatibility domain."""

    domain: str
    major: int
    minor: int

    def __post_init__(self) -> None:
        if not _DOMAIN_PATTERN.fullmatch(self.domain):
            raise ValueError(f"invalid version domain {self.domain!r}")
        if not 0 <= self.major <= 0xFFFF:
            raise ValueError(f"invalid {self.domain} major version {self.major}")
        if not 0 <= self.minor <= 0xFFFF:
            raise ValueError(f"invalid {self.domain} minor version {self.minor}")


@dataclasses.dataclass(frozen=True, slots=True)
class VersionDomain:
    """One independently negotiated format or instruction authority."""

    domain: str
    page_id: int
    major: int
    latest_minor: int
    summary: str

    def __post_init__(self) -> None:
        Version(self.domain, self.major, self.latest_minor)
        if self.page_id == CORE_PAGE_ID:
            if self.domain != "core":
                raise ValueError(
                    f"{self.domain}: only the core domain may use page ID 0x0"
                )
        elif not (
            ARCHITECTURAL_EXTENSION_PAGE_MIN
            <= self.page_id
            <= ARCHITECTURAL_EXTENSION_PAGE_MAX
        ):
            raise ValueError(f"{self.domain}: page ID {self.page_id:#x} is unavailable")
        if self.domain == "core" and self.page_id != CORE_PAGE_ID:
            raise ValueError("core: the core domain must use page ID 0x0")
        if not self.summary.strip():
            raise ValueError(f"{self.domain}: missing version-domain summary")

    def version(self, minor: int | None = None) -> Version:
        """Returns the latest or selected version in this domain."""

        return Version(
            self.domain,
            self.major,
            self.latest_minor if minor is None else minor,
        )


CORE_DOMAIN = VersionDomain(
    domain="core",
    page_id=0x00,
    major=0,
    latest_minor=0,
    summary="Core module format and mandatory instruction set.",
)
HAL_DOMAIN = VersionDomain(
    domain="hal",
    page_id=0xF0,
    major=0,
    latest_minor=0,
    summary="Optional Hardware Abstraction Layer instruction page.",
)
VERSION_DOMAINS = (CORE_DOMAIN, HAL_DOMAIN)

CORE_0 = CORE_DOMAIN.version(0)
HAL_0 = HAL_DOMAIN.version(0)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Entity:
    """Base declaration for one stable, version-owned specification fact."""

    entity_id: str
    since: Version
    summary: str
    minimum_consumer_version: Version | None = None
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.entity_id):
            raise ValueError(f"invalid entity ID {self.entity_id!r}")
        if not self.summary.strip():
            raise ValueError(f"{self.entity_id}: missing summary")
        if tuple(sorted(self.dependencies)) != self.dependencies:
            raise ValueError(f"{self.entity_id}: dependencies are not sorted")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"{self.entity_id}: duplicate dependency")
        if self.entity_id in self.dependencies:
            raise ValueError(f"{self.entity_id}: self dependency")
        minimum_consumer_version = self.minimum_consumer_version
        if minimum_consumer_version is None:
            minimum_consumer_version = self.since
            object.__setattr__(
                self,
                "minimum_consumer_version",
                minimum_consumer_version,
            )
        if minimum_consumer_version.domain != self.since.domain:
            raise ValueError(
                f"{self.entity_id}: consumer and introduction domains differ"
            )
        if minimum_consumer_version.major != self.since.major:
            raise ValueError(
                f"{self.entity_id}: consumer and introduction majors differ"
            )
        if minimum_consumer_version.minor > self.since.minor:
            raise ValueError(
                f"{self.entity_id}: minimum consumer version follows its introduction"
            )
        for dependency in self.dependencies:
            if not _IDENTIFIER_PATTERN.fullmatch(dependency):
                raise ValueError(f"{self.entity_id}: invalid dependency {dependency!r}")

    @property
    def normative_anchor(self) -> str:
        """Returns the stable generated-document anchor for this entity."""

        return "spec-" + self.entity_id.replace(".", "-").replace("_", "-")

    def referenced_entity_ids(self) -> tuple[str, ...]:
        """Returns entity references encoded by a concrete declaration."""

        return ()

    def dependency_ids(self) -> tuple[str, ...]:
        """Returns explicit and structurally derived dependencies."""

        combined = (*self.dependencies, *self.referenced_entity_ids())
        return tuple(sorted(set(combined)))

    def validate(self, specification: Specification) -> None:
        """Validates concrete declaration invariants against the whole graph."""

        del specification


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NormativeClause(Entity):
    """One reusable version-owned normative contract inherited by entities."""

    normative_text: str

    def __post_init__(self) -> None:
        super(NormativeClause, self).__post_init__()
        if not self.normative_text.strip():
            raise ValueError(f"{self.entity_id}: missing normative text")


@dataclasses.dataclass(frozen=True, slots=True)
class Projection:
    """One dependency-closed historical view of a specification."""

    specification_name: str
    versions: tuple[Version, ...]
    domains: tuple[VersionDomain, ...]
    entities: tuple[Entity, ...]

    def entity_map(self) -> dict[str, Entity]:
        """Returns projected entities keyed by stable identity."""

        return {entity.entity_id: entity for entity in self.entities}

    def require_entity(self, entity_id: str) -> Entity:
        """Returns an available entity or rejects use outside this view."""

        entity = self.entity_map().get(entity_id)
        if entity is None:
            raise KeyError(
                f"{entity_id!r} is unavailable in projection {self.version_label()}"
            )
        return entity

    def version_label(self) -> str:
        """Returns a deterministic human-readable version selection."""

        return ", ".join(
            f"{version.domain}={version.major}.{version.minor}"
            for version in self.versions
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReleaseDiff:
    """Exact stable-identity difference between two projected releases."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def is_additive(self) -> bool:
        """Returns whether the later view only adds new stable identities."""

        return not self.removed and not self.changed

    def require_additive(self) -> None:
        """Rejects removal or reinterpretation within a compatible major."""

        if self.removed or self.changed:
            raise ValueError(
                "release is not additive: "
                f"removed={list(self.removed)}, changed={list(self.changed)}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Specification:
    """Complete current specification and all of its historical facts."""

    name: str
    domains: tuple[VersionDomain, ...]
    entities: tuple[Entity, ...]

    def __post_init__(self) -> None:
        if not _SPECIFICATION_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid specification name {self.name!r}")
        if not self.domains:
            raise ValueError(f"{self.name}: no version domains")

        domains_by_name: dict[str, VersionDomain] = {}
        domains_by_page: dict[int, VersionDomain] = {}
        for domain in self.domains:
            if domain.domain in domains_by_name:
                raise ValueError(f"duplicate version domain {domain.domain}")
            if domain.page_id in domains_by_page:
                raise ValueError(
                    f"duplicate version page {domain.page_id:#x} for "
                    f"{domain.domain} and {domains_by_page[domain.page_id].domain}"
                )
            domains_by_name[domain.domain] = domain
            domains_by_page[domain.page_id] = domain

        entities_by_id: dict[str, Entity] = {}
        anchors: dict[str, str] = {}
        for entity in self.entities:
            if entity.entity_id in entities_by_id:
                raise ValueError(f"duplicate entity ID {entity.entity_id}")
            domain = domains_by_name.get(entity.since.domain)
            if domain is None:
                raise ValueError(
                    f"{entity.entity_id}: unknown version domain {entity.since.domain}"
                )
            if entity.since.major != domain.major:
                raise ValueError(
                    f"{entity.entity_id}: major {entity.since.major} does not "
                    f"match {domain.domain} major {domain.major}"
                )
            if entity.since.minor > domain.latest_minor:
                raise ValueError(
                    f"{entity.entity_id}: introduced in unavailable "
                    f"{entity.since.domain} minor {entity.since.minor}"
                )
            prior_anchor_owner = anchors.get(entity.normative_anchor)
            if prior_anchor_owner is not None:
                raise ValueError(
                    f"{entity.entity_id}: normative anchor collides with "
                    f"{prior_anchor_owner}"
                )
            anchors[entity.normative_anchor] = entity.entity_id
            entities_by_id[entity.entity_id] = entity

        for entity in self.entities:
            dependency_ids = entity.dependency_ids()
            if entity.entity_id in dependency_ids:
                raise ValueError(f"{entity.entity_id}: self dependency")
            for dependency_id in dependency_ids:
                if not _IDENTIFIER_PATTERN.fullmatch(dependency_id):
                    raise ValueError(
                        f"{entity.entity_id}: invalid dependency {dependency_id!r}"
                    )
                dependency = entities_by_id.get(dependency_id)
                if dependency is None:
                    raise ValueError(
                        f"{entity.entity_id}: missing dependency {dependency_id}"
                    )
                if (
                    dependency.since.domain == entity.since.domain
                    and dependency.since.major == entity.since.major
                    and dependency.since.minor > entity.since.minor
                ):
                    raise ValueError(
                        f"{entity.entity_id}: introduced before same-domain "
                        f"dependency {dependency_id}"
                    )
        self._validate_acyclic(entities_by_id)
        for entity in self.entities:
            entity.validate(self)

    def domain_map(self) -> dict[str, VersionDomain]:
        """Returns version domains keyed by stable domain name."""

        return {domain.domain: domain for domain in self.domains}

    def entity_map(self) -> dict[str, Entity]:
        """Returns all entities keyed by stable identity."""

        return {entity.entity_id: entity for entity in self.entities}

    def project(self, versions: Iterable[Version]) -> Projection:
        """Projects and validates one exact set of domain versions."""

        selected_versions = tuple(sorted(versions, key=lambda value: value.domain))
        selected_by_domain: dict[str, Version] = {}
        domains_by_name = self.domain_map()
        for version in selected_versions:
            if version.domain in selected_by_domain:
                raise ValueError(f"duplicate projected domain {version.domain}")
            domain = domains_by_name.get(version.domain)
            if domain is None:
                raise ValueError(f"unknown projected domain {version.domain}")
            if version.major != domain.major:
                raise ValueError(
                    f"{version.domain}: requested major {version.major}, "
                    f"supported major {domain.major}"
                )
            if version.minor > domain.latest_minor:
                raise ValueError(
                    f"{version.domain}: requested minor {version.minor}, "
                    f"latest minor {domain.latest_minor}"
                )
            selected_by_domain[version.domain] = version

        projected_entities = tuple(
            entity
            for entity in self.entities
            if (
                (selected := selected_by_domain.get(entity.since.domain)) is not None
                and entity.since.major == selected.major
                and entity.since.minor <= selected.minor
            )
        )
        projected_ids = {entity.entity_id for entity in projected_entities}
        for entity in projected_entities:
            missing = set(entity.dependency_ids()) - projected_ids
            if missing:
                raise ValueError(
                    f"{entity.entity_id}: projection omits dependencies "
                    f"{sorted(missing)}"
                )
        projected_domains = tuple(
            self.domain_map()[version.domain] for version in selected_versions
        )
        return Projection(
            self.name,
            selected_versions,
            projected_domains,
            projected_entities,
        )

    def derive_projection_versions(
        self,
        entity_ids: Iterable[str],
    ) -> tuple[Version, ...]:
        """Derives versions that expose a transitive declaration set."""

        return self._derive_versions(entity_ids, use_consumer_versions=False)

    def derive_requirements(self, entity_ids: Iterable[str]) -> tuple[Version, ...]:
        """Derives minimum consumer versions for a transitive feature set."""

        return self._derive_versions(entity_ids, use_consumer_versions=True)

    def _derive_versions(
        self,
        entity_ids: Iterable[str],
        *,
        use_consumer_versions: bool,
    ) -> tuple[Version, ...]:
        """Reduces one dependency closure to exact domain versions."""

        entities_by_id = self.entity_map()
        requirements: dict[str, Version] = {}
        visited: set[str] = set()

        def visit(entity_id: str) -> None:
            if entity_id in visited:
                return
            entity = entities_by_id.get(entity_id)
            if entity is None:
                raise KeyError(f"unknown specification entity {entity_id!r}")
            visited.add(entity_id)
            version = (
                entity.minimum_consumer_version
                if use_consumer_versions
                else entity.since
            )
            if version is None:
                raise AssertionError(f"{entity.entity_id}: missing consumer version")
            current = requirements.get(version.domain)
            if current is None or version.minor > current.minor:
                requirements[version.domain] = version
            for dependency_id in entity.dependency_ids():
                visit(dependency_id)

        for entity_id in entity_ids:
            visit(entity_id)
        return tuple(requirements[key] for key in sorted(requirements))

    @staticmethod
    def _validate_acyclic(entities_by_id: Mapping[str, Entity]) -> None:
        visiting: list[str] = []
        visiting_set: set[str] = set()
        visited: set[str] = set()

        def visit(entity_id: str) -> None:
            if entity_id in visited:
                return
            if entity_id in visiting_set:
                cycle_start = visiting.index(entity_id)
                cycle = visiting[cycle_start:] + [entity_id]
                raise ValueError(
                    "specification dependency cycle: " + " -> ".join(cycle)
                )
            visiting.append(entity_id)
            visiting_set.add(entity_id)
            for dependency_id in entities_by_id[entity_id].dependency_ids():
                visit(dependency_id)
            visiting.pop()
            visiting_set.remove(entity_id)
            visited.add(entity_id)

        for entity_id in sorted(entities_by_id):
            visit(entity_id)


def compare_projections(before: Projection, after: Projection) -> ReleaseDiff:
    """Returns exact stable-identity and content changes between two views."""

    if before.specification_name != after.specification_name:
        raise ValueError(
            "cannot compare projections from different specifications: "
            f"{before.specification_name!r} and {after.specification_name!r}"
        )
    before_map = before.entity_map()
    after_map = after.entity_map()
    before_ids = set(before_map)
    after_ids = set(after_map)
    return ReleaseDiff(
        added=tuple(sorted(after_ids - before_ids)),
        removed=tuple(sorted(before_ids - after_ids)),
        changed=tuple(
            sorted(
                entity_id
                for entity_id in before_ids & after_ids
                if before_map[entity_id] != after_map[entity_id]
            )
        ),
    )
