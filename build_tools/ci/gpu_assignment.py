# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolves a Linux runner's GPU grant to ordered DRM character devices."""

import errno
import os
import re
import stat
from pathlib import Path

DRM_CLASS = Path("/sys/class/drm")
DRI = Path("/dev/dri")
KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def native_identity(number: int) -> str:
    return f"linux_device:{os.major(number)}:{os.minor(number)}"


def _render_name(value: str, dri: Path) -> str:
    path = Path(value)
    if path.parent == dri / "by-path":
        path = path.resolve(strict=True)
    if path.parent != dri or not re.fullmatch(r"renderD[0-9]+", path.name):
        raise ValueError(f"Expected a DRM render node under {dri}, got {value!r}")
    return path.name


def _rocr_names(tokens: list[str], dri: Path, topology: Path) -> list[str]:
    # ROCr ranks compute-bearing KFD nodes in numeric topology order, before
    # applying its visible-device mask. The thunk removes render nodes hidden
    # by the cgroup or mount. Beyond the first such node, a parent ordinal and
    # a container ordinal can denote different GPUs.
    generation_file = topology.parent / "generation_id"
    generation = generation_file.read_text()
    devices = []
    first_hidden = None
    for node in sorted(topology.iterdir(), key=lambda path: int(path.name)):
        try:
            properties = dict(
                line.split() for line in (node / "properties").read_text().splitlines()
            )
        except PermissionError as error:
            if error.errno == errno.EPERM:
                if first_hidden is None:
                    first_hidden = len(devices)
                continue
            raise
        if not int(properties["simd_count"]):
            continue
        name = f"renderD{int(properties['drm_render_minor'])}"
        try:
            (dri / name).stat()
        except FileNotFoundError:
            if first_hidden is None:
                first_hidden = len(devices)
            continue
        devices.append((name, int(properties["unique_id"])))

    if generation_file.read_text() != generation:
        raise ValueError("KFD topology changed while resolving the runner allocation")

    names = []
    for token in tokens:
        if re.fullmatch(r"[0-9]+", token):
            rank = int(token)
            if first_hidden is not None and rank >= first_hidden:
                raise ValueError(
                    f"ROCR_VISIBLE_DEVICES rank {rank} crosses a hidden KFD node; supply a physical GPU assignment"
                )
            if rank >= len(devices):
                raise ValueError(
                    f"ROCR_VISIBLE_DEVICES rank {rank} exceeds the visible KFD inventory"
                )
            names.append(devices[rank][0])
        elif re.fullmatch(r"GPU-[0-9a-f]{1,16}", token, re.IGNORECASE):
            matches = [
                name
                for name, uuid in devices
                if uuid and f"{uuid:016x}".startswith(token[4:].lower())
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"ROCR_VISIBLE_DEVICES token {token!r} matches {len(matches)} KFD nodes"
                )
            names.extend(matches)
        else:
            raise ValueError(
                f"Invalid or unavailable ROCR_VISIBLE_DEVICES token {token!r}"
            )
    return names


def _native_name(identity: str) -> str:
    match = re.fullmatch(r"linux_device:([0-9]+:[0-9]+)", identity)
    if not match:
        raise ValueError(f"Invalid Linux GPU identity {identity!r}")
    number = match[1]
    name = "renderD" + number.split(":")[1]
    if (DRM_CLASS / name / "dev").read_text().strip() != number:
        raise ValueError(f"{identity} does not identify {name}")
    return name


def _grant(variable: str) -> list[str]:
    tokens = [part.strip() for part in os.environ[variable].split(",")]
    if not all(tokens):
        raise ValueError(f"{variable} contains an empty GPU assignment")
    if variable == "ROCR_VISIBLE_DEVICES":
        names = _rocr_names(tokens, DRI, KFD_NODES)
    elif variable in ("CHILD_GPU_DEVICES", "PARENT_GPU_DEVICES"):
        names = [_render_name(token, DRI) for token in tokens]
    else:
        if variable == "AMDF_CTS_GPU_NATIVE_IDENTITY" and len(tokens) != 1:
            raise ValueError(f"{variable} selects exactly one GPU")
        names = [_native_name(token) for token in tokens]
    if len(set(names)) != len(names):
        raise ValueError(f"{variable} repeats a GPU")
    return names


def assigned_render_nodes() -> dict[Path, int]:
    """Returns the intersection of the job's grants, without opening a GPU."""
    if os.environ.get("AMDF_CTS_GPU_PEER_NATIVE_IDENTITY"):
        raise ValueError("A peer GPU is outside this single-GPU job's assignment")
    # The child's order wins when several grants describe the same devices.
    sources = (
        "CHILD_GPU_DEVICES",
        "HAL_AMD_GPU_ASSIGNMENT",
        "ROCR_VISIBLE_DEVICES",
        "PARENT_GPU_DEVICES",
    )
    grants = {name: _grant(name) for name in sources if name in os.environ}
    if (
        "CHILD_GPU_DEVICES" in grants
        and "PARENT_GPU_DEVICES" in grants
        and not set(grants["CHILD_GPU_DEVICES"]).issubset(grants["PARENT_GPU_DEVICES"])
    ):
        raise ValueError(f"Child GPU allocation exceeds its parent: {grants}")
    if not grants:
        # TheRock's Azure GPU scale set gives one ephemeral single-GPU VM to
        # each job and deliberately supplies an empty isolation env-file.
        if not re.fullmatch(
            r"azure-linux-[a-z0-9]+-gfx[0-9a-f]+-gpu-rocm-scale-runner-[a-z0-9]{4}",
            os.environ.get("RUNNER_NAME", ""),
        ):
            raise ValueError(
                "No runner GPU assignment; supply HAL_AMD_GPU_ASSIGNMENT or assigned render nodes"
            )
        devices = [
            entry.name
            for entry in DRM_CLASS.glob("renderD[0-9]*")
            if int((entry / "device/vendor").read_text(), 16) == 0x1002
        ]
        if len(devices) != 1:
            raise ValueError(
                f"Dedicated single-GPU runner exposes {len(devices)} AMD render nodes: {devices}"
            )
        grants["dedicated VM"] = devices
    if "AMDF_CTS_GPU_NATIVE_IDENTITY" in os.environ:
        grants["AMDF_CTS_GPU_NATIVE_IDENTITY"] = _grant("AMDF_CTS_GPU_NATIVE_IDENTITY")
    names = [
        name
        for name in next(iter(grants.values()))
        if all(name in grant for grant in grants.values())
    ]
    if not names:
        raise ValueError(f"Conflicting runner GPU assignments: {grants}")
    result = {}
    for name in names:
        path = DRI / name
        metadata = path.stat()
        major, minor = (DRM_CLASS / name / "dev").read_text().strip().split(":")
        if not stat.S_ISCHR(metadata.st_mode) or metadata.st_rdev != os.makedev(
            int(major), int(minor)
        ):
            raise ValueError(
                f"{path} does not refer to the kernel's render node {major}:{minor}"
            )
        result[path] = metadata.st_rdev
    print(
        "Runner GPU assignment: "
        + ", ".join(
            f"{path} ({native_identity(number)})" for path, number in result.items()
        ),
        flush=True,
    )
    return result
