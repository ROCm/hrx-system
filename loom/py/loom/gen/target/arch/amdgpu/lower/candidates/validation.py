# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validation helpers for generated AMDGPU source-to-low candidate tables."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

from loom.target.arch.amdgpu.descriptors import amdgpu_descriptor_ref_keys
from loom.target.low_descriptors import target_relative_name

_UINT16_MAX = 0xFFFF


def c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def descriptor_ref_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{c_identifier(target_relative_name('amdgpu', key))}"


def require_descriptor_refs(
    owner: str,
    descriptor_keys: Iterable[str],
    descriptor_ref_key_set: set[str] | None = None,
) -> None:
    known_refs = descriptor_ref_key_set if descriptor_ref_key_set is not None else set(amdgpu_descriptor_ref_keys())
    missing_refs = sorted({key for key in descriptor_keys if key not in known_refs})
    if missing_refs:
        raise ValueError(f"{owner} requires missing descriptor refs: {', '.join(missing_refs)}")


def required_descriptor_ref_constant_name(
    owner: str,
    key: str,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    require_descriptor_refs(owner, (key,), descriptor_ref_key_set)
    return descriptor_ref_constant_name(key)


def optional_descriptor_ref_constant_name(
    key: str,
    descriptor_ref_key_set: set[str],
) -> str:
    if key not in descriptor_ref_key_set:
        return "LOOM_AMDGPU_DESCRIPTOR_REF_NONE"
    return descriptor_ref_constant_name(key)


def _validate_uint16_field(owner: str, field_name: str, value: int) -> None:
    if value < 0 or value > _UINT16_MAX:
        raise ValueError(f"{owner} has {field_name} {value}, which does not fit uint16_t")


def dense_candidate_ranges[T, K: Hashable](
    candidates: Sequence[T],
    key_fn: Callable[[T], K],
    *,
    owner: str,
    range_sort_key: Callable[[K], Any] | None = None,
) -> tuple[tuple[K, int, int], ...]:
    ranges: dict[K, tuple[int, int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        _validate_uint16_field(owner, "candidate index", candidate_index)
        key = key_fn(candidate)
        previous_range = ranges.get(key)
        if previous_range is None:
            ranges[key] = (candidate_index, 1)
            continue
        first_candidate, candidate_count = previous_range
        if first_candidate + candidate_count != candidate_index:
            raise ValueError(f"{owner} must be contiguous by range key {key}")
        next_candidate_count = candidate_count + 1
        _validate_uint16_field(owner, "candidate count", next_candidate_count)
        ranges[key] = (first_candidate, next_candidate_count)
    return tuple(
        (key, first_candidate, candidate_count)
        for key, (first_candidate, candidate_count) in sorted(
            ranges.items(),
            key=lambda item: range_sort_key(item[0]) if range_sort_key is not None else item[0],
        )
    )
