# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lossless deterministic JSON projection for tooling consumers."""

from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Mapping

from model.specification import Entity, Projection


def _json_value(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if dataclasses.is_dataclass(value):
        payload = {
            "kind": type(value).__name__,
            **{
                field.name: _json_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
        if isinstance(value, Entity):
            payload["normative_anchor"] = value.normative_anchor
        return payload
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(element)
            for key, element in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(element) for element in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot project {type(value).__name__} to JSON")


def projection_payload(projection: Projection) -> dict[str, object]:
    """Returns a complete self-describing projection payload."""

    return {
        "schema": "iree.vm.bytecode.specification",
        "schema_version": 1,
        "specification": projection.specification_name,
        "versions": [_json_value(version) for version in projection.versions],
        "domains": [_json_value(domain) for domain in projection.domains],
        "entities": [_json_value(entity) for entity in projection.entities],
    }


def render_projection_json(projection: Projection) -> str:
    """Renders one dependency-closed specification projection as JSON."""

    return (
        json.dumps(
            projection_payload(projection),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
