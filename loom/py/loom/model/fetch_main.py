# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""loom-fetch command entry point; acquisition is also available as a Python API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from loom.model import ManifestError, ModelCard
from loom.model.catalog import ModelCatalog
from loom.model.fetch import check_output, install, plan_fetch, plan_lock, update_lock
from loom.model.fetch_console import AccessWaiter, ConsoleProgress, display_text
from loom.model.fetch_hf import AccessRequired
from loom.model.fetch_picker import Picker
from loom.model.fetch_publication import (
    PublicationSnapshot,
    installed_catalog,
    load_publication,
)
from loom.model.fetch_source import FetchError
from loom.model.fetch_storage import ContentCache
from loom.model.memory import describe_memory


def _cache_directory() -> Path:
    if "LOOM_CACHE_HOME" in os.environ:
        return Path(os.environ["LOOM_CACHE_HOME"])
    if os.name == "nt":
        if "LOCALAPPDATA" in os.environ:
            root = Path(os.environ["LOCALAPPDATA"])
        else:
            root = Path.home() / "AppData" / "Local"
    else:
        if "XDG_CACHE_HOME" in os.environ:
            root = Path(os.environ["XDG_CACHE_HOME"])
        else:
            root = Path.home() / ".cache"
    return root / "loom"


def _overrides(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, location = value.partition("=")
        if not separator or not name or not location:
            raise ManifestError("--origin requires NAME=LOCAL_DIRECTORY_OR_URL")
        if name in result:
            raise ManifestError(f"duplicate --origin for {name!r}")
        result[name] = location
    return result


def _default_output(model_id: str) -> Path:
    # Model IDs are opaque identifiers in the card. Only this implicit CLI
    # destination gives them filesystem semantics, portable across supported hosts.
    for part in model_id.split("/"):
        if (
            part in ("", ".", "..")
            or part.endswith((".", " "))
            or any(
                not character.isprintable() or character in '<>:"\\|?*'
                for character in part
            )
            or re.fullmatch(
                r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³]",
                part.split(".", 1)[0].rstrip(" ").upper(),
            )
        ):
            raise ManifestError(
                f"model ID {model_id!r} is not a portable relative output path; "
                "specify --output=DIR"
            )
    directory = Path(model_id)
    if any(
        path.is_symlink() or path.is_junction()
        for path in (directory, *directory.parents)
    ):
        raise ManifestError(
            "default output traverses a symbolic link or junction; "
            "specify --output=DIR to select that storage explicitly"
        )
    return directory


def _update_all_variants_lock(args, document, card, cache, snapshot):
    plan = plan_lock(
        document,
        card,
        cache,
        programs=args.programs,
        overrides=_overrides(args.origin),
        lock_location=args.lockfile,
        offline=args.offline,
        lock_scope=args.lock_scope or "assets",
    )
    result = plan.summary()
    if snapshot is not None:
        result["catalog"] = snapshot.summary()
    if not args.dry_run:
        update_lock(plan, cache, preserve_sources=args.preserve_sources)
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif args.dry_run:
        sys.stdout.write(
            f"{display_text(plan.model_id)}: "
            f"{len(plan.selections)} choices, "
            f"{len(plan.sources)} unique outputs\n"
        )
        for source in plan.sources.values():
            sys.stdout.write("  " + display_text(json.dumps(source.origin)) + "\n")
        for output in plan.derived.values():
            sys.stdout.write(
                "  "
                + display_text(
                    f"{json.dumps(output.origin)} (derived at load from {output.source})"
                )
                + "\n"
            )
    else:
        sys.stdout.write(display_text(str(plan.path)) + "\n")
    return 0


def _list_card(card: ModelCard, *, json_output: bool) -> None:
    listing = card.document
    if json_output:
        sys.stdout.write(json.dumps(listing, indent=2) + "\n")
        return
    sys.stdout.write(display_text(listing["model"]["name"]) + "\n")
    for section in ("presets", "variants", "services", "programs"):
        sys.stdout.write(
            f"{section}: "
            + ", ".join(display_text(value) for value in listing.get(section, {}))
            + "\n"
        )
    for name, preset in listing.get("presets", {}).items():
        if "memory" in preset:
            sys.stdout.write(
                display_text(name + ": " + describe_memory(preset["memory"])) + "\n"
            )


def _catalog_model(catalog: ModelCatalog, picker: Picker | None) -> str:
    choices = catalog.document["models"]
    if not choices:
        raise ManifestError("catalog contains no models")
    if len(choices) == 1:
        return next(iter(choices))
    if picker is not None:
        return picker.model(catalog.document)
    raise ManifestError(
        "catalog has multiple models; select --model=ID from: " + ", ".join(choices)
    )


def _list_catalog(snapshot: PublicationSnapshot, *, json_output: bool) -> None:
    listing = snapshot.content.document
    if json_output:
        sys.stdout.write(
            json.dumps({**listing, "snapshot": snapshot.summary()}, indent=2) + "\n"
        )
    else:
        if snapshot.revision:
            sys.stderr.write(f"Catalog revision: {snapshot.revision}\n")
        for identity, entry in listing["models"].items():
            sys.stdout.write(
                f"{display_text(identity)}\t{display_text(entry['name'])}\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loom-fetch",
        description="Acquire a portable Loom model's selected files. "
        "Without a card or model ID, an interactive terminal opens the model picker.",
    )
    parser.add_argument(
        "card",
        nargs="?",
        metavar="SOURCE",
        help="local model JSON, HTTP(S) model-card or repository URL, or "
        "PROVIDER:OWNER/REPOSITORY[:PRESET][@REVISION] "
        "(providers: huggingface, github)",
    )
    parser.add_argument("--model", help="model ID to select from the publication")
    parser.add_argument(
        "--catalog",
        help="local catalog JSON, Hugging Face catalog URL, "
        "or explicit repository reference (selects its root catalog.json); "
        "defaults to this SDK's association",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list declared choices without acquiring payloads",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit structured JSON on stdout"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress on stderr"
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="disable terminal and browser interaction",
    )
    parser.add_argument(
        "--wait-for-access",
        action="store_true",
        help="wait and recheck provider access even without an interactive terminal",
    )
    parser.add_argument("--preset", help="named canonical preset")
    parser.add_argument("--service", help="named service")
    parser.add_argument("--variant", help="named parameter variant")
    parser.add_argument("--partition", help="named executable partition of the variant")
    parser.add_argument(
        "--programs", help="named program set (independent of file format)"
    )
    parser.add_argument(
        "--origin",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="replace one source root with a local directory or HTTP(S) directory",
    )
    parser.add_argument(
        "--output", type=Path, help="installation directory (default: ./MODEL_ID)"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=_cache_directory(),
        help="shared content and document cache",
    )
    parser.add_argument(
        "--lockfile", help="explicit lock path or URL; defaults to CARD.lock"
    )
    parser.add_argument(
        "--offline", action="store_true", help="forbid all network requests"
    )
    locking = parser.add_mutually_exclusive_group()
    locking.add_argument(
        "--no-lock", action="store_true", help="neither read nor write locks"
    )
    locking.add_argument(
        "--update-lock",
        action="store_true",
        help="replace the lock with exactly the selected verified sources",
    )
    parser.add_argument(
        "--lock-scope",
        choices=("assets", "all"),
        help="with --update-lock: assets (default) leaves programs editable; "
        "all also records program and auxiliary file identities for publication",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="with --update-lock, publish a lock for every declared variant, "
        "partition, service, and program set without creating an installation; "
        "--programs restricts publication to that set's compatible choices",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and show the plan without acquiring payloads",
    )
    parser.add_argument(
        "--materialize",
        choices=("auto", "share", "copy"),
        default="auto",
        help="auto (default) copies programs and their auxiliary files while "
        "sharing assets, warning and copying if links are unavailable; "
        "share requires links for every file; copy makes every file independent",
    )
    parser.add_argument(
        "--preserve-sources",
        action="store_true",
        help="retain acquired transform input ranges; "
        "existing local/HF cache files remain shared",
    )
    parser.add_argument(
        "--no-verify-parameters",
        action="store_true",
        help="developer mode: skip publisher parameter expectations while retaining "
        "code and other input checks; requires a new installation",
    )
    args = parser.parse_args(argv)
    if args.lock_scope is not None and not args.update_lock:
        parser.error("--lock-scope requires --update-lock")
    if args.no_verify_parameters and (args.no_lock or args.update_lock):
        parser.error(
            "--no-verify-parameters cannot be combined with --no-lock or --update-lock"
        )
    if args.all_variants:
        if not args.update_lock:
            parser.error("--all-variants requires --update-lock")
        if any(
            (
                args.preset,
                args.service,
                args.variant,
                args.partition,
            )
        ):
            parser.error("--all-variants permits only --programs as a selection flag")
        if args.output is not None or args.materialize != "auto" or args.list:
            parser.error(
                "--all-variants publishes only a lock; "
                "--output, --materialize=share/copy, and --list do not apply"
            )
    if args.card is not None and args.catalog is not None:
        parser.error("a source cannot be combined with --catalog")
    interactive = (
        not args.no_input
        and not args.json
        and sys.stdin.isatty()
        and sys.stderr.isatty()
    )
    if (
        args.card is None
        and args.catalog is None
        and args.model is None
        and not args.list
        and not interactive
    ):
        parser.error("select a card or --model=ID; --list shows the available catalog")
    if args.no_lock and args.lockfile:
        parser.error("--no-lock cannot be combined with --lockfile")
    try:
        picker = Picker(sys.stdin, sys.stderr) if interactive else None
        wait = (
            AccessWaiter(interactive=interactive, input=sys.stdin, output=sys.stderr)
            if interactive or args.wait_for_access
            else None
        )
        cache = ContentCache(
            args.cache,
            wait_for_access=wait,
            progress=ConsoleProgress(sys.stderr, quiet=args.quiet),
        )
        publication = load_publication(
            args.card if args.card is not None else args.catalog or installed_catalog(),
            cache,
            offline=args.offline,
            catalog=args.card is None,
        )
        if publication.preset is not None:
            if args.preset is not None and args.preset != publication.preset:
                raise ManifestError("repository preset conflicts with --preset")
            if args.all_variants:
                raise ManifestError("--all-variants cannot select a repository preset")
            args.preset = publication.preset
        snapshot = (
            publication if isinstance(publication.content, ModelCatalog) else None
        )
        if snapshot is not None:
            if args.model is None and args.list:
                _list_catalog(snapshot, json_output=args.json)
                return 0
            if args.model is None:
                args.model = _catalog_model(snapshot.content, picker)
        else:
            args.model = args.model or publication.content.document["model"]["id"]
        document, card = publication.read_model(args.model, cache, offline=args.offline)
        if args.list:
            _list_card(card, json_output=args.json)
            return 0
        if args.all_variants:
            return _update_all_variants_lock(args, document, card, cache, snapshot)
        select = (
            card.select
            if picker is None
            else lambda **values: picker.selection(card, **values)
        )
        selection = select(
            preset=args.preset,
            service=args.service,
            variant=args.variant,
            partition=args.partition,
            programs=args.programs,
        )
        if args.output is None:
            args.output = _default_output(selection.model_id)
        cache.progress.emit("destination", path=str(args.output.absolute()))
        plan = plan_fetch(
            document,
            selection,
            cache,
            overrides=_overrides(args.origin),
            lock_location=args.lockfile,
            offline=args.offline,
            no_lock=args.no_lock,
            update_lock=args.update_lock,
            lock_scope=args.lock_scope,
            no_verify_parameters=args.no_verify_parameters,
            catalog=snapshot.summary() if snapshot else None,
        )
        result = plan.summary()
        if "memory" in result and not args.json:
            sys.stderr.write(display_text(describe_memory(result["memory"])) + "\n")
        if plan.unverified_parameters and not args.json:
            sys.stderr.write(
                "Skipping publisher parameter expectations: "
                + ", ".join(
                    display_text(path) for path in sorted(plan.unverified_parameters)
                )
                + "\n"
            )
        if snapshot is not None:
            result["catalog"] = snapshot.summary()
        result["output"] = str(args.output.absolute())
        if args.dry_run:
            check_output(plan, args.output)
        else:
            result["card"] = str(
                install(
                    plan,
                    cache,
                    args.output,
                    materialize=args.materialize,
                    preserve_sources=args.preserve_sources,
                )
            )
        if args.json:
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
        elif args.dry_run:
            sys.stdout.write(
                display_text(
                    f"{selection.model_id}: {selection.variant} / "
                    f"{selection.programs} / {selection.service_name}"
                )
                + "\n"
            )
            for file in result["files"]:
                size = (
                    "unknown size" if file["size"] is None else f"{file['size']} bytes"
                )
                if file.get("derived"):
                    size = (
                        "derived at load from " + file["derived"]["source"]
                        if file["size"] is None
                        else f"derived at load from {file['derived']['source']}; "
                        f"expected {file['size']} bytes"
                    )
                sys.stdout.write(display_text(f"  {file['path']} ({size})") + "\n")
        else:
            sys.stdout.write(display_text(result["card"]) + "\n")
        return 0
    except AccessRequired as error:
        if args.json:
            sys.stdout.write(json.dumps(error.document(), indent=2) + "\n")
        sys.stderr.write(f"loom-fetch: {display_text(str(error))}\n")
        return 1
    except (ManifestError, FetchError, OSError) as error:
        if args.json:
            sys.stdout.write(
                json.dumps({"kind": "error", "message": str(error)}) + "\n"
            )
        sys.stderr.write(f"loom-fetch: {display_text(str(error))}\n")
        return 1
    except KeyboardInterrupt:
        if args.json:
            sys.stdout.write(json.dumps({"kind": "cancelled"}) + "\n")
        sys.stderr.write("loom-fetch: cancelled\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
