# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Metadata-only model references for publishers and SDK users."""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote, urljoin

from loom.model.lock import FileIdentity, ModelLock
from loom.model.manifest import ManifestError, ModelCard, ModelFile, Selection, _url
from loom.model.memory import describe_memory


def _text(value: object) -> str:
    """Keeps author text literal in headings, paragraphs, and table cells."""
    return "".join(
        "<br>"
        if character == "\n"
        else f"&#{ord(character)};"
        if character in "\\`*_{}[]#!|~+-"
        or ord(character) < 32
        or ord(character) == 127
        else html.escape(character, quote=True)
        for character in str(value)
    )


def _code(value: object) -> str:
    return "<code>" + _text(value) + "</code>"


def _link(label: str, url: str) -> str:
    # Angle destinations admit parentheses. Encode their delimiters and escape
    # entities without changing existing URL escapes or international hostnames.
    destination = re.sub(r'[<>"\\\s`|]', lambda match: quote(match[0], safe=""), url)
    return "[" + _text(label) + "](<" + html.escape(destination, quote=False) + ">)"


def _json(value: object) -> list[str]:
    # JSON quotes strings and escapes their line endings, so author content
    # cannot introduce a closing fence on its own line.
    return ["```json", json.dumps(value, indent=2, ensure_ascii=False), "```", ""]


def _source(
    file: ModelFile,
    roots: dict,
    lock: ModelLock | None,
    identity: FileIdentity | None,
    card_url: str | None,
) -> str:
    origin = file.origin
    revision = identity.resolved["revision"] if identity and identity.resolved else None
    if isinstance(origin, str):
        if origin.startswith("@"):
            alias, suffix = origin[1:].split("/", 1)
            descriptor = roots[alias]
            kind = descriptor["kind"]
            if kind == "hf":
                origin = descriptor | {"path": suffix}
                if lock:
                    revision = lock.origins[alias]["resolved"]["revision"]
            elif kind == "url":
                origin = {
                    "kind": "url",
                    "url": urljoin(
                        descriptor["url"].rstrip("/") + "/", quote(suffix, safe="/")
                    ),
                }
            else:
                origin = {
                    "kind": "local",
                    "path": descriptor["path"].rstrip("/\\") + "/" + suffix,
                }
        elif "://" in origin:
            origin = {"kind": "url", "url": origin}
        elif card_url:
            return _link(origin, urljoin(card_url, quote(origin, safe="/")))
        else:
            return _code(origin) + " (relative to model card)"
    kind = origin["kind"]
    if kind == "hf":
        revision = revision or origin.get("revision", "main")
        url = (
            "https://huggingface.co/"
            + quote(origin["repository"], safe="/")
            + "/resolve/"
            + quote(revision, safe="")
            + "/"
            + quote(origin["path"], safe="/")
        )
        return (
            _link(origin["repository"] + "/" + origin["path"], url)
            + " at "
            + _code(revision)
        )
    if kind == "url":
        return _link(origin["url"], origin["url"])
    if kind == "inline":
        return f"Inline data ({len(base64.b64decode(origin['base64']))} bytes)"
    return _code(origin["path"]) + " (local source)"


def _uses(selection: Selection) -> tuple[dict[str, list[str]], dict]:
    """Retains ordered shard/provider positions and their explicit consumers."""
    uses: dict[str, list[str]] = {}
    options = {}

    def add(value, use):
        record = {"path": value} if isinstance(value, str) else value
        path = record["path"]
        uses.setdefault(path, []).append(use)
        if record.get("options"):
            options[path] = record["options"]

    for scope, parameter in selection.parameters.items():
        for index, value in enumerate(parameter["files"]):
            add(value, f"parameters.{scope}[{index}]")
    for name, tokenizer in selection.tokenizers.items():
        add(tokenizer, "tokenizers." + name)
    for name, voice in selection.voices.items():
        add(voice["file"], "voices." + name)
    for role in ("inputs", "libraries"):
        for index, value in enumerate(selection.program[role]):
            add(value, f"{role}[{index}]")
            if isinstance(value, dict):
                for auxiliary in value.get("data", []):
                    add(auxiliary, "data for " + value["path"])
    return uses, options


def render_markdown(
    card: ModelCard,
    *,
    lock: ModelLock | None = None,
    card_url: str | None = None,
    overview: str | None = None,
) -> str:
    """Renders every compatible choice without acquisition or payload access.

    A supplied lock must cover every displayed choice. Its immutable revisions
    and output commitments are displayed, without verifying any bytes. Relative
    sources become absolute links only when a publication card URL is supplied.
    An optional publisher-authored Markdown overview is included verbatim before
    the generated contract. It owns user guidance and uses publication-safe links.
    """
    if card_url is not None:
        _url(card_url, "card_url")
    document = card.document
    model = document["model"]
    choices = card.choices()
    lines = ["# " + _text(model["name"]), ""]
    if model.get("description"):
        lines += [_text(model["description"]), ""]
    lines += [
        "Model: "
        + _code(model["id"])
        + ". Card schema: "
        + str(document["schema_version"])
        + ".",
        "",
    ]
    if "license" in model:
        lines += ["Model license: " + _text(model["license"]) + ".", ""]
    if card_url:
        lines += ["Model card: " + _link("model.json", card_url) + ".", ""]
    sources = []
    if "origin" in model:
        sources.append(("Original model", model["origin"], None))
    for name, origin in document.get("origins", {}).items():
        resolved = lock.origins.get(name, {}).get("resolved", {}) if lock else {}
        sources.append((name, origin, resolved.get("revision")))
    for name, origin, resolved_revision in sources:
        if origin["kind"] == "hf":
            url = "https://huggingface.co/" + quote(origin["repository"], safe="/")
            source = _link(origin["repository"], url)
            revision = resolved_revision or origin.get("revision")
            if revision:
                source += (
                    " ("
                    + _link(
                        revision,
                        url + "/tree/" + quote(revision, safe=""),
                    )
                    + ")"
                )
        elif origin["kind"] == "url":
            source = _link(origin["url"], origin["url"])
        else:
            source = _code(origin["path"]) + " (local source)"
        lines += [_text(name) + ": " + source + ".", ""]
    if overview:
        lines += [overview.strip(), ""]
    lines += [
        "## Model contract",
        "",
        "This reference describes the card's declared choices. "
        "Execution qualification belongs to the model's published checks.",
        "",
        "Source links use the supplied lock's immutable revisions; byte counts and "
        "SHA-256 values commit to output files. Generation does not verify payloads."
        if lock
        else "Source links use authored revisions, including moving names when "
        "declared. No acquisition lock was supplied.",
        "",
    ]
    provenance = {
        key: model[key] for key in ("origin", "implementation") if key in model
    }
    if provenance:
        lines += ["## Provenance", "", *_json(provenance)]
    lines += [
        "## Declared choices",
        "",
        "| Variant | Partition | Service | Programs |",
        "| --- | --- | --- | --- |",
    ]
    for choice in choices:
        lines.append(
            "| "
            + " | ".join(
                _text(choice.get(key, "—"))
                for key in ("variant", "partition", "service", "programs")
            )
            + " |"
        )
    lines += ["", "## Canonical presets", ""]
    for name, preset in document.get("presets", {}).items():
        if "memory" in preset:
            lines += [
                _code(name) + ": " + _text(describe_memory(preset["memory"])),
                "",
            ]
    lines += (
        _json(document["presets"])
        if document.get("presets")
        else ["No canonical presets declared.", ""]
    )
    lines += [
        "## Services",
        "",
        "An omitted service version selects the unversioned development ABI.",
        "",
        *_json(document["services"]),
    ]
    for choice in choices:
        selection = card.select(**choice)
        identities = lock.cover(selection) if lock else {}
        title = " / ".join(
            choice[key]
            for key in ("variant", "partition", "service", "programs")
            if key in choice
        )
        lines += ["## Files: " + _text(title), ""]
        variant = document["variants"][selection.variant]
        for description in (
            variant.get("description"),
            variant.get("partitions", {})
            .get(selection.partition, {})
            .get("description"),
        ):
            if description:
                lines += [_text(description), ""]
        lines += [
            "Program directory: "
            + _code(selection.program.get("directory", "."))
            + ".",
            "",
        ]
        if selection.config:
            lines += ["Effective JIT configuration:", "", *_json(selection.config)]
        lines += ["| Local path | Use | Format | Source |", "| --- | --- | --- | --- |"]
        uses, options = _uses(selection)
        for path, consumers in uses.items():
            file = selection.files[path]
            source = _source(
                file, document.get("origins", {}), lock, identities.get(path), card_url
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _code(path),
                        _text(", ".join(consumers)),
                        _text(file.format or "inferred"),
                        source,
                    )
                )
                + " |"
            )
        lines.append("")
        if selection.voices:
            lines += ["Original voice recordings:", "", *_json(selection.voices)]
        if options:
            lines += ["Importer options:", "", *_json(options)]
        transforms = {
            path: file.transform
            for path, file in selection.files.items()
            if file.transform
        }
        if transforms:
            lines += [
                "Acquisition transforms (source links identify the input; "
                "locks identify the resulting archive):",
                "",
                *_json(transforms),
            ]
        if lock:
            lines += [
                "### Locked output files",
                "",
                "| Local path | Bytes | SHA-256 |",
                "| --- | --- | --- |",
            ]
            for path in uses:
                identity = identities[path]
                lines.append(
                    f"| {_code(path)} | {identity.size} | {_code(identity.sha256)} |"
                )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a model reference without reading or fetching payloads."
    )
    parser.add_argument(
        "--card", type=Path, required=True, help="Local model card JSON"
    )
    parser.add_argument(
        "--lockfile", type=Path, help="Lock covering every displayed choice"
    )
    parser.add_argument(
        "--card-url", help="Publication URL for resolving card-relative source links"
    )
    parser.add_argument(
        "--overview",
        type=Path,
        help="Publisher-authored Markdown included before the generated contract",
    )
    parser.add_argument(
        "--output", type=Path, help="UTF-8 Markdown output (default: stdout)"
    )
    args = parser.parse_args(argv)
    try:
        if (
            args.output
            and args.output.exists()
            and any(
                args.output.samefile(path)
                for path in (args.card, args.lockfile, args.overview)
                if path
            )
        ):
            raise ManifestError("output must not overwrite an input document")
        card = ModelCard.parse(args.card.read_text(encoding="utf-8"))
        lock = (
            ModelLock.parse(args.lockfile.read_text(encoding="utf-8"))
            if args.lockfile
            else None
        )
        overview = args.overview.read_text(encoding="utf-8") if args.overview else None
        result = render_markdown(
            card, lock=lock, card_url=args.card_url, overview=overview
        )
        if args.output:
            args.output.write_text(result, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(result)
    except (ManifestError, OSError, UnicodeError) as error:
        sys.stderr.write(f"model documentation: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
