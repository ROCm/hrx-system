# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds a declared provider closure and emits a portable model card.

Bazel owns providers and their auxiliary inputs, plus any explicitly supplied
runtime fixtures. Unsupplied runtime assets retain their acquisition descriptors
for loom-fetch. This action never acquires files or invents service roots.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from loom.model import ManifestError, ModelCard


def build(plan: dict, linker: str | None = None) -> None:
    selection = ModelCard.parse(Path(plan["card"]).read_text(encoding="utf-8")).select(
        plan["preset"],
        programs=plan.get("programs"),
        partition=plan.get("partition"),
    )
    providers = {
        name for name, file in selection.files.items() if "program" in file.roles
    }
    data = {
        name for name, file in selection.files.items() if "auxiliary" in file.roles
    } - providers
    assets = set(selection.files) - providers - data
    for role, expected in (("files", providers), ("data", data)):
        supplied = set(plan[role])
        if supplied != expected:
            raise ManifestError(
                f"{role} must match the selected closure: "
                f"missing {sorted(expected - supplied)}, "
                f"unused {sorted(supplied - expected)}"
            )
    unused_assets = set(plan["assets"]) - assets
    if unused_assets:
        raise ManifestError(
            f"assets contain unused runtime files: {sorted(unused_assets)}"
        )
    materialized = providers | data | set(plan["assets"])
    reserved = {"model.json", "serve.json"}
    for path in selection.files:
        if path.split("/", 1)[0] in reserved:
            raise ManifestError(f"file {path!r} conflicts with a generated document")
    encoding = plan["encoding"]
    if encoding not in ("bc", "text", "preserve"):
        raise ManifestError(f"unsupported provider encoding {encoding!r}")
    if encoding != "preserve" and not linker:
        raise ManifestError("provider normalization requires --linker")

    # Only produced files change identity. Runtime assets that remain outside
    # the build keep their logical sources, formats, and acquisition transforms.
    document = selection.document
    aliases = {
        file.origin[1:].split("/", 1)[0]
        for name, file in selection.files.items()
        if name not in materialized
        and isinstance(file.origin, str)
        and file.origin.startswith("@")
    }
    if aliases:
        document["origins"] = {
            name: origin
            for name, origin in document["origins"].items()
            if name in aliases
        }
    else:
        document.pop("origins", None)
    document.pop("$schema", None)
    program = document["programs"][selection.programs]
    program.pop("origin", None)

    def output_file(value):
        record = {"path": value} if isinstance(value, str) else value
        if record["path"] not in materialized:
            return record | {"origin": selection.files[record["path"]].origin}
        if isinstance(value, str):
            return value
        return {key: value[key] for key in ("path", "format") if key in value}

    for role in ("inputs", "libraries"):
        normalized = []
        for value in program[role]:
            provider = {"path": value} if isinstance(value, str) else value
            name = provider["path"]
            if encoding == "preserve":
                result = {
                    key: provider[key]
                    for key in ("path", "format", "options")
                    if key in provider
                }
                if provider.get("data"):
                    result["data"] = [output_file(item) for item in provider["data"]]
                normalized.append(result)
                continue
            binding = plan["files"][name]
            arguments = [linker, "--mode=merge", binding["input"]]
            format_name = provider.get("format")
            if format_name in ("loom", "loombc"):
                arguments.extend(
                    [
                        "--input-format=loom",
                        "--from=" + ("bc" if format_name == "loombc" else "text"),
                    ]
                )
            elif format_name:
                arguments.append("--input-format=" + format_name)
            arguments.extend(
                "--input-options=" + option for option in provider.get("options", [])
            )
            arguments.extend(
                [
                    "--source-prefix-map=" + plan["input_root"] + "/=",
                    "--to=" + plan["encoding"],
                    "--output=" + binding["output"],
                ]
            )
            subprocess.run(arguments, check=True)
            result = {
                "path": name,
                "format": "loombc" if plan["encoding"] == "bc" else "loom",
            }
            if provider.get("data"):
                result["data"] = [output_file(item) for item in provider["data"]]
            normalized.append(result)
        program[role] = normalized
    scope = document["variants"][selection.variant]
    if selection.partition is not None:
        scope = scope["partitions"][selection.partition]
    variant = scope["services"][selection.service_name]
    for parameter in variant.get("parameters", {}).values():
        parameter.pop("origin", None)
        parameter["files"] = [output_file(item) for item in parameter["files"]]
    variant["tokenizers"] = {
        name: output_file(value)
        for name, value in variant.get("tokenizers", {}).items()
    }
    Path(plan["output_card"]).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    deployment = {
        "name": plan["name"],
        "model": {"path": "model.json", "preset": selection.preset},
        "definition": {"kind": selection.service["kind"]},
    }
    Path(plan["output_deployment"]).write_text(
        json.dumps(deployment, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--linker")
    args = parser.parse_args()
    build(json.loads(args.plan.read_text(encoding="utf-8")), args.linker)


if __name__ == "__main__":
    main()
