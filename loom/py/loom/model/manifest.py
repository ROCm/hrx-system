# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable model selection, independent of compilation, acquisition, and hosts."""

from __future__ import annotations

import base64
import copy
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit


class ManifestError(ValueError):
    """An invalid external model description or unavailable selection."""


def _fail(path: str, message: str) -> None:
    raise ManifestError(f"{path}: {message}")


def _object(value: Any, path: str, required=(), optional=()) -> dict:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing:
        _fail(path, f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        _fail(path, f"unknown fields: {', '.join(sorted(unknown))}")
    return value


def _mapping(value: Any, path: str, *, empty: bool = False) -> dict:
    if not isinstance(value, dict) or (not value and not empty):
        _fail(path, "expected a named object" + ("" if empty else " with entries"))
    for key in value:
        _string(key, path)
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        _fail(path, "expected a nonempty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(path, "expected valid UTF-8")
    return value


def _path(value: Any, path: str, *, directory: bool = False) -> str:
    _string(value, path)
    parsed = PurePosixPath(value)
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or ".." in parsed.parts
        or str(parsed) != value
        or (value == "." and not directory)
    ):
        _fail(path, "expected a normalized relative path")
    return value


def _strings(value: Any, path: str, *, empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        _fail(path, "expected an array" + ("" if empty else " with entries"))
    for index, item in enumerate(value):
        _string(item, f"{path}[{index}]")
    if len(value) != len(set(value)):
        _fail(path, "duplicate entries")
    return value


def _reference(value: Any, table: dict, path: str) -> str:
    _string(value, path)
    if value not in table:
        _fail(path, f"unknown reference {value!r}")
    return value


def _json_value(value: Any, path: str) -> None:
    """Reject values that cannot cross the JSON/UTF-8 public boundary."""
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "numbers must be finite")
    elif isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            _fail(path, "expected valid UTF-8")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _string(key, path)
            _json_value(item, path + "." + key)
    else:
        _fail(path, "expected a JSON value")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("$", f"duplicate object key {key!r}")
        result[key] = value
    return result


def parse_json(source: str) -> Any:
    """Reads plain finite JSON, rejecting duplicate keys and JSONC extensions."""

    def reject_constant(value):
        _fail("$", f"non-JSON numeric constant {value}")

    try:
        value = json.loads(
            source, object_pairs_hook=_unique_object, parse_constant=reject_constant
        )
    except json.JSONDecodeError as error:
        raise ManifestError(f"$: {error}") from error
    _json_value(value, "$")
    return value


def _url(value: Any, path: str) -> None:
    _string(value, path)
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        _fail(path, "URL whitespace and control characters must be escaped")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        _fail(path, "invalid HTTP(S) URL")
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        _fail(path, "expected an HTTP(S) URL without credentials or fragment")


def _origin_object(value: Any, path: str, *, root: bool) -> dict:
    _mapping(value, path)
    kind = value.get("kind")
    fields = {
        "hf": (("kind", "repository"), ("revision",) if root else ("revision", "path")),
        "url": (("kind", "url"), ()),
        "local": (("kind", "path"), ()),
    }
    if not root:
        fields["inline"] = (("kind", "base64"), ())
    if not isinstance(kind, str) or kind not in fields:
        _fail(path + ".kind", f"unsupported origin {kind!r}")
    _object(value, path, *fields[kind])
    if kind == "hf":
        _path(value["repository"], path + ".repository")
        if "revision" in value:
            _string(value["revision"], path + ".revision")
        if not root:
            _path(value.get("path"), path + ".path")
    elif kind == "url":
        _url(value["url"], path + ".url")
        if root and urlsplit(value["url"]).query:
            _fail(path + ".url", "directory origins cannot contain a query")
    elif kind == "local":
        # Acquisition knows if host-local coordinates came from a local caller
        # or an untrusted remote document; parsing alone cannot authorize them.
        _string(value["path"], path + ".path")
    else:
        try:
            base64.b64decode(value["base64"], validate=True)
        except (TypeError, ValueError):
            _fail(path + ".base64", "invalid base64 content")
    return value


def _origin(value: Any, roots: dict, path: str, *, group: bool = False) -> None:
    if isinstance(value, dict):
        if group:
            _fail(path, "group origins require a directory locator")
        _origin_object(value, path, root=False)
        return
    _string(value, path)
    if value.startswith("@"):
        name, separator, suffix = value[1:].partition("/")
        _reference(name, roots, path)
        if not separator or (not suffix and not group):
            _fail(path, "origin aliases require @name/path")
        if suffix:
            _path(suffix.removesuffix("/") if group else suffix, path)
    elif "://" in value:
        _url(value, path)
        if group and urlsplit(value).query:
            _fail(path, "directory origins cannot contain a query")
    else:
        _path(value.removesuffix("/") if group else value, path, directory=group)
    if group and not value.endswith("/"):
        _fail(path, "group origin must end with '/'")


def _transform(value: Any, path: str) -> dict:
    _object(value, path, ("kind", "version"), ("alignment", "tensors"))
    if (
        value["kind"] != "safetensors"
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        _fail(path, "supported transform: safetensors version 1")
    alignment = value.get("alignment", 8)
    if (
        type(alignment) is not int
        or not 8 <= alignment <= 1024 * 1024
        or alignment & (alignment - 1)
    ):
        _fail(path + ".alignment", "expected a power of two from 8 through 1048576")
    result = {"kind": "safetensors", "version": 1, "alignment": alignment}
    if "tensors" in value:
        tensors = value["tensors"]
        if (
            not isinstance(tensors, list)
            or not tensors
            or any(not isinstance(name, str) for name in tensors)
            or len(set(tensors)) != len(tensors)
        ):
            _fail(path + ".tensors", "expected a nonempty array of unique tensor names")
        result["tensors"] = sorted(value["tensors"])
    return result


def _file(
    value: Any,
    roots: dict,
    path: str,
    *,
    provider: bool = False,
    parameter: bool = False,
) -> dict:
    record = {"path": value} if isinstance(value, str) else value
    _object(
        record,
        path,
        ("path",),
        ("origin", "format", "options", "data")
        if provider
        else ("origin", "format", "transform")
        if parameter
        else ("origin", "format"),
    )
    _path(record["path"], path + ".path")
    if "format" in record:
        _string(record["format"], path + ".format")
    if "origin" in record:
        _origin(record["origin"], roots, path + ".origin")
    if "transform" in record:
        record["transform"] = _transform(record["transform"], path + ".transform")
        if record.get("format") != "safetensors":
            _fail(
                path + ".format",
                "safetensors transform requires explicit format 'safetensors'",
            )
    if provider:
        _strings(record.get("options", []), path + ".options")
        data = record.get("data", [])
        if not isinstance(data, list):
            _fail(path + ".data", "expected an array")
        for index, item in enumerate(data):
            _file(item, roots, f"{path}.data[{index}]")
    return record


def _config(value: Any, path: str) -> None:
    for name, item in _mapping(value, path, empty=True).items():
        if type(item) not in (str, bool, int, float):
            _fail(path + "." + name, "expected a scalar compiler value")


def _service_contract(service: dict, path: str) -> None:
    version = service.get("version", 1)
    if type(version) is not int or not 1 <= version <= 0xFFFFFFFF:
        _fail(path + ".version", "expected a service ABI version in [1, 2^32-1]")
    _strings(service.get("options", []), path + ".options")
    for key in ("interface", "constraints"):
        _mapping(service.get(key, {}), path + "." + key, empty=True)


def _memory(value: Any, path: str) -> None:
    _object(value, path, ("workload",), ("weights_bytes", "transient_bytes"))
    _string(value["workload"], path + ".workload")
    if not ({"weights_bytes", "transient_bytes"} & value.keys()):
        _fail(path, "requires at least one memory estimate")
    for key in ("weights_bytes", "transient_bytes"):
        if key in value and (
            type(value[key]) is not int or not 0 <= value[key] < 2**64
        ):
            _fail(path + "." + key, "expected integer bytes in [0, 2^64-1]")


@dataclass(frozen=True)
class ModelFile:
    """One destination with explicit source intent and all selected consumers."""

    # Normalized relative destination under the card directory.
    path: str
    # Logical locator or provider descriptor, independent of destination naming.
    origin: str | dict
    # Role-specific explicit encoding, or None for inference.
    format: str | None
    # Consumers determine integrity requirements; any code role remains code.
    roles: frozenset[str]
    # Normalized acquisition operation; absent for an unchanged source file.
    transform: dict | None = None


@dataclass(frozen=True)
class Selection:
    """One owned dependency closure with no compiler, device, or network state."""

    # Original training-source identity.
    model_id: str
    # Canonical preset name, or None for explicit choices.
    preset: str | None
    # Service identity, separate from its contract kind and version.
    service_name: str
    # Parameter/configuration variant identity.
    variant: str
    # Arbitrary named executable partition, or None for an unpartitioned variant.
    partition: str | None
    # Provider-set identity with no encoding semantics.
    programs: str
    # Selected service contract.
    service: dict
    # Selected provider set with ordered input/library roles.
    program: dict
    # Variant-over-program fixed JIT configuration.
    config: dict
    # Code-facing parameter index declarations.
    parameters: dict
    # Code-facing tokenizer declarations.
    tokenizers: dict
    # Canonical execution, request, workload, and preparation options.
    settings: dict
    # Deduplicated destinations with resolved group-origin inheritance.
    files: dict[str, ModelFile]
    # Portable card containing only this selection, preserving source intent.
    document: dict


class ModelCard:
    """An owned, validated model publication with independent service revisions."""

    def __init__(self, document: dict):
        self._document = copy.deepcopy(document)
        _json_value(self._document, "$")
        self._validate()

    @classmethod
    def parse(cls, source: str) -> ModelCard:
        return cls(parse_json(source))

    @property
    def document(self) -> dict:
        return copy.deepcopy(self._document)

    def _validate(self) -> None:
        doc = _mapping(self._document, "$")
        if type(doc.get("schema_version")) is not int or doc["schema_version"] != 1:
            _fail("$.schema_version", "unsupported model-card version; supported: 1")
        _object(
            doc,
            "$",
            ("schema_version", "model", "programs", "services", "variants"),
            ("$schema", "origins", "presets"),
        )
        if "$schema" in doc:
            _string(doc["$schema"], "$.$schema")
        model = _object(
            doc["model"],
            "$.model",
            ("id", "name"),
            ("description", "license", "origin", "implementation"),
        )
        for key in ("id", "name"):
            _string(model[key], "$.model." + key)
        for key in ("description", "license"):
            if key in model and not isinstance(model[key], str):
                _fail("$.model." + key, "expected a string")
        if "origin" in model:
            _origin_object(model["origin"], "$.model.origin", root=True)
        if "implementation" in model:
            implementation = _object(
                model["implementation"],
                "$.model.implementation",
                ("repository", "revision", "path"),
            )
            for key in ("repository", "revision"):
                _string(implementation[key], "$.model.implementation." + key)
            _path(implementation["path"], "$.model.implementation.path")
        origins = _mapping(doc.get("origins", {}), "$.origins", empty=True)
        for name, origin in origins.items():
            if "/" in name or name.startswith("@"):
                _fail("$.origins", "alias names cannot contain '/' or start with '@'")
            _origin_object(origin, "$.origins." + name, root=True)
        programs = _mapping(doc["programs"], "$.programs")
        for name, program in programs.items():
            path = "$.programs." + name
            _object(
                program,
                path,
                ("inputs", "libraries"),
                ("directory", "origin", "config"),
            )
            _path(program.get("directory", "."), path + ".directory", directory=True)
            if "origin" in program:
                _origin(program["origin"], origins, path + ".origin", group=True)
            _config(program.get("config", {}), path + ".config")
            seen = set()
            for role in ("inputs", "libraries"):
                if not isinstance(program[role], list):
                    _fail(path + "." + role, "expected an array")
                for index, item in enumerate(program[role]):
                    provider = _file(
                        item, origins, f"{path}.{role}[{index}]", provider=True
                    )
                    if provider["path"] in seen:
                        _fail(path, f"duplicate provider {provider['path']!r}")
                    seen.add(provider["path"])
            if not program["inputs"]:
                _fail(path + ".inputs", "requires at least one input")
        services = _mapping(doc["services"], "$.services")
        for name, service in services.items():
            path = "$.services." + name
            _object(
                service,
                path,
                ("kind",),
                ("version", "options", "interface", "constraints", "components"),
            )
            kind = _string(service["kind"], path + ".kind")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kind):
                _fail(path + ".kind", "expected a service symbol identifier")
            _service_contract(service, path)
            for role, component in _mapping(
                service.get("components", {}), path + ".components", empty=True
            ).items():
                component_path = path + ".components." + role
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
                    _fail(component_path, "expected a component symbol identifier")
                _object(
                    component,
                    component_path,
                    (),
                    ("version", "options", "interface", "constraints"),
                )
                _service_contract(component, component_path)
        self._validate_variants(origins, services, programs)
        for name, preset in _mapping(
            doc.get("presets", {}), "$.presets", empty=True
        ).items():
            path = "$.presets." + name
            _object(
                preset,
                path,
                ("service", "variant", "programs"),
                (
                    "options",
                    "execution",
                    "request",
                    "workload",
                    "partition",
                    "memory",
                    "components",
                ),
            )
            for key in ("options", "execution", "request", "workload"):
                _mapping(preset.get(key, {}), path + "." + key, empty=True)
            for role, component in _mapping(
                preset.get("components", {}), path + ".components", empty=True
            ).items():
                component_path = path + ".components." + role
                _object(component, component_path, (), ("options", "execution"))
                for key in ("options", "execution"):
                    _mapping(
                        component.get(key, {}), component_path + "." + key, empty=True
                    )
            if "partition" in preset:
                _string(preset["partition"], path + ".partition")
            if "memory" in preset:
                _memory(preset["memory"], path + ".memory")
            self.select(name)

    def _validate_variants(self, origins: dict, services: dict, programs: dict):
        doc = self._document
        variants = _mapping(doc["variants"], "$.variants")
        for name, variant in variants.items():
            path = "$.variants." + name
            _object(variant, path, (), ("description", "services", "partitions"))
            if ("services" in variant) == ("partitions" in variant):
                _fail(path, "declare exactly one of services or partitions")
            if "description" in variant and not isinstance(variant["description"], str):
                _fail(path + ".description", "expected a string")
            scopes = variant.get("partitions", {None: variant})
            if "partitions" in variant:
                _mapping(scopes, path + ".partitions")
            for partition, partition_definition in scopes.items():
                partition_path = path
                if partition is not None:
                    partition_path += ".partitions." + partition
                    _object(
                        partition_definition,
                        partition_path,
                        ("services",),
                        ("description",),
                    )
                    if "description" in partition_definition and not isinstance(
                        partition_definition["description"], str
                    ):
                        _fail(partition_path + ".description", "expected a string")
                for service_name, variant_service in _mapping(
                    partition_definition["services"], partition_path + ".services"
                ).items():
                    current = partition_path + ".services." + service_name
                    _reference(service_name, services, current)
                    _object(
                        variant_service,
                        current,
                        ("programs",),
                        ("config", "parameters", "tokenizers"),
                    )
                    for program in _strings(
                        variant_service["programs"], current + ".programs", empty=False
                    ):
                        _reference(program, programs, current + ".programs")
                    _config(variant_service.get("config", {}), current + ".config")
                    for scope, parameter in _mapping(
                        variant_service.get("parameters", {}),
                        current + ".parameters",
                        empty=True,
                    ).items():
                        scope_path = current + ".parameters." + scope
                        _object(parameter, scope_path, ("files",), ("origin",))
                        if "origin" in parameter:
                            _origin(
                                parameter["origin"],
                                origins,
                                scope_path + ".origin",
                                group=True,
                            )
                        if (
                            not isinstance(parameter["files"], list)
                            or not parameter["files"]
                        ):
                            _fail(
                                scope_path + ".files", "expected a nonempty shard array"
                            )
                        for index, item in enumerate(parameter["files"]):
                            _file(
                                item,
                                origins,
                                f"{scope_path}.files[{index}]",
                                parameter=True,
                            )
                    for identity, tokenizer in _mapping(
                        variant_service.get("tokenizers", {}),
                        current + ".tokenizers",
                        empty=True,
                    ).items():
                        _file(tokenizer, origins, current + ".tokenizers." + identity)

    def choices(self) -> list[dict[str, str]]:
        """Enumerates declared compatible choices without reading any payload."""
        choices = []
        for variant, definition in self._document["variants"].items():
            for partition, scope in definition.get(
                "partitions", {None: definition}
            ).items():
                for service, supported in scope["services"].items():
                    for programs in supported["programs"]:
                        choice = {
                            "variant": variant,
                            "service": service,
                            "programs": programs,
                        }
                        if partition is not None:
                            choice["partition"] = partition
                        choices.append(choice)
        return choices

    def select(
        self,
        preset: str | None = None,
        *,
        service: str | None = None,
        variant: str | None = None,
        programs: str | None = None,
        partition: str | None = None,
    ) -> Selection:
        doc = self._document

        def choose(table, requested, label):
            if requested is None:
                if len(table) != 1:
                    _fail(label, f"select explicitly ({len(table)} choices)")
                return next(iter(table))
            return _reference(requested, table, label)

        presets = doc.get("presets", {})
        if (
            preset is None
            and all(value is None for value in (service, variant, programs, partition))
            and presets
        ):
            preset = choose(presets, None, "preset")
        settings = {}
        if preset is not None:
            _reference(preset, presets, "preset")
            settings = presets[preset]
        variant = choose(
            doc["variants"],
            variant if variant is not None else settings.get("variant"),
            "variant",
        )
        variant_scope = doc["variants"][variant]
        requested_partition = (
            partition if partition is not None else settings.get("partition")
        )
        if "partitions" in variant_scope:
            partition = choose(
                variant_scope["partitions"], requested_partition, "partition"
            )
            variant_scope = variant_scope["partitions"][partition]
        elif requested_partition is not None:
            _fail("partition", "selected variant is unpartitioned")
        choices = variant_scope["services"]
        service = choose(
            choices,
            service if service is not None else settings.get("service"),
            "service",
        )
        variant_service = choices[service]
        programs = choose(
            dict.fromkeys(variant_service["programs"]),
            programs if programs is not None else settings.get("programs"),
            "programs",
        )
        service_definition = doc["services"][service]
        unknown_options = settings.get("options", {}).keys() - set(
            service_definition.get("options", [])
        )
        if unknown_options:
            _fail(
                "options",
                "undeclared model options: " + ", ".join(sorted(unknown_options)),
            )
        for role, component in settings.get("components", {}).items():
            definition = service_definition.get("components", {}).get(role)
            if definition is None:
                _fail("components." + role, "preset names an undeclared component")
            unknown = component.get("options", {}).keys() - set(
                definition.get("options", [])
            )
            if unknown:
                _fail(
                    "components." + role + ".options",
                    "undeclared model options: " + ", ".join(sorted(unknown)),
                )
        program = doc["programs"][programs]
        files = {}

        def add(value, role, group_origin=""):
            record = {"path": value} if isinstance(value, str) else value
            name = record["path"]
            origin = record.get(
                "origin", ("" if group_origin == "./" else group_origin) + name
            )
            candidate = ModelFile(
                name,
                copy.deepcopy(origin),
                record.get("format"),
                frozenset({role}),
                copy.deepcopy(record.get("transform")),
            )
            if name in files:
                previous = files[name]
                if (
                    previous.origin != candidate.origin
                    or previous.transform != candidate.transform
                    or (
                        previous.format
                        and candidate.format
                        and previous.format != candidate.format
                    )
                ):
                    _fail("files." + name, "conflicting selected file declarations")
                candidate = replace(
                    previous,
                    roles=previous.roles | candidate.roles,
                    format=previous.format or candidate.format,
                )
            files[name] = candidate

        for role in ("inputs", "libraries"):
            for value in program[role]:
                add(value, "program", program.get("origin", ""))
                provider = {"path": value} if isinstance(value, str) else value
                for auxiliary in provider.get("data", []):
                    add(auxiliary, "auxiliary", program.get("origin", ""))
        for parameter in variant_service.get("parameters", {}).values():
            for value in parameter["files"]:
                add(value, "parameter", parameter.get("origin", ""))
        for value in variant_service.get("tokenizers", {}).values():
            add(value, "tokenizer")
        paths = set(files)
        for path in paths:
            if any(str(parent) in paths for parent in PurePosixPath(path).parents):
                _fail(
                    "files." + path,
                    "destination collides with another file's directory",
                )
        projected = {
            key: copy.deepcopy(doc[key])
            for key in ("$schema", "schema_version", "model", "origins")
            if key in doc
        }
        projected["programs"] = {programs: copy.deepcopy(program)}
        projected["services"] = {service: copy.deepcopy(service_definition)}
        projected_variant_service = copy.deepcopy(variant_service)
        projected_variant_service["programs"] = [programs]
        projected_scope = {"services": {service: projected_variant_service}}
        if partition is not None:
            if "description" in variant_scope:
                projected_scope["description"] = variant_scope["description"]
            projected_scope = {"partitions": {partition: projected_scope}}
        projected["variants"] = {variant: projected_scope}
        if "description" in doc["variants"][variant]:
            projected["variants"][variant]["description"] = doc["variants"][variant][
                "description"
            ]
        settings = copy.deepcopy(settings)
        if any(
            settings.get(key, value) != value
            for key, value in (
                ("variant", variant),
                ("partition", partition),
                ("service", service),
                ("programs", programs),
            )
        ):
            settings.pop("memory", None)
        settings.update(service=service, variant=variant, programs=programs)
        if partition is not None:
            settings["partition"] = partition
        projected["presets"] = {preset: settings} if preset else {}
        return Selection(
            model_id=doc["model"]["id"],
            preset=preset,
            service_name=service,
            variant=variant,
            partition=partition,
            programs=programs,
            service=copy.deepcopy(service_definition),
            program=copy.deepcopy(program),
            config=copy.deepcopy(
                program.get("config", {}) | variant_service.get("config", {})
            ),
            parameters=copy.deepcopy(variant_service.get("parameters", {})),
            tokenizers=copy.deepcopy(variant_service.get("tokenizers", {})),
            settings=settings,
            files=files,
            document=projected,
        )
