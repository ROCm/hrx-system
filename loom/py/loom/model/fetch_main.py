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
import sys
from pathlib import Path

from loom.model import ManifestError, ModelCard
from loom.model.fetch import check_output, install, plan_fetch, plan_lock, update_lock
from loom.model.fetch_catalog import installed_catalog, load_catalog
from loom.model.fetch_console import AccessWaiter, display_text
from loom.model.fetch_hf import AccessRequired
from loom.model.fetch_picker import Picker
from loom.model.fetch_source import FetchError, document_location
from loom.model.fetch_storage import ContentCache
from loom.model.memory import describe_memory


def _cache_directory() -> Path:
    if "LOOM_CACHE_HOME" in os.environ:
        return Path(os.environ["LOOM_CACHE_HOME"])
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
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


def _update_publication_lock(args, document, card, cache, snapshot):
    publication = plan_lock(
        document,
        card,
        cache,
        programs=args.programs,
        overrides=_overrides(args.origin),
        lock_location=args.lockfile,
        offline=args.offline,
    )
    result = publication.summary()
    if snapshot is not None:
        result["catalog"] = snapshot.summary()
    if not args.dry_run:
        update_lock(publication, cache, preserve_sources=args.preserve_sources)
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif args.dry_run:
        sys.stdout.write(
            f"{display_text(publication.model_id)}: "
            f"{len(publication.selections)} choices, "
            f"{len(publication.sources)} unique outputs\n"
        )
        for source in publication.sources.values():
            sys.stdout.write("  " + display_text(json.dumps(source.origin)) + "\n")
    else:
        sys.stdout.write(display_text(str(publication.path)) + "\n")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loom-fetch",
        description="Acquire a portable Loom model's selected files. "
        "Without a card or model ID, an interactive terminal opens the model picker.",
    )
    parser.add_argument(
        "card", nargs="?", help="local model JSON or a raw HTTP(S) model-card URL"
    )
    parser.add_argument("--model", help="model ID in the selected catalog")
    parser.add_argument(
        "--catalog",
        help="local catalog JSON or github:OWNER/REPOSITORY@TAG_OR_COMMIT; "
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
    parser.add_argument("--output", type=Path, help="installation directory")
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
        choices=("share", "copy"),
        default="share",
        help="share source/cache files or explicitly create independent copies",
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
        if args.output is not None or args.materialize != "share" or args.list:
            parser.error(
                "--all-variants publishes only a lock; "
                "--output, --materialize=copy, and --list do not apply"
            )
    if args.card is not None and (args.model is not None or args.catalog is not None):
        parser.error("a direct card cannot be combined with --model or --catalog")
    interactive = (
        not args.no_input
        and not args.json
        and sys.stdin.isatty()
        and sys.stderr.isatty()
    )
    if args.card is None and args.model is None and not args.list and not interactive:
        parser.error("select a card or --model=ID; --list shows the available catalog")
    if (
        not args.list
        and not args.all_variants
        and args.output is None
        and not interactive
    ):
        parser.error("--output is required for acquisition or --dry-run")
    if args.no_lock and args.lockfile:
        parser.error("--no-lock cannot be combined with --lockfile")
    try:
        picker = Picker(sys.stdin, sys.stderr) if interactive else None
        wait = (
            AccessWaiter(interactive=interactive, input=sys.stdin, output=sys.stderr)
            if interactive or args.wait_for_access
            else None
        )
        cache = ContentCache(args.cache, wait_for_access=wait)
        snapshot = None
        if args.card is None:
            snapshot = load_catalog(
                args.catalog or installed_catalog(), cache, offline=args.offline
            )
            if args.model is None and args.list:
                listing = snapshot.catalog.document
                if args.json:
                    sys.stdout.write(
                        json.dumps(
                            {**listing, "snapshot": snapshot.summary()}, indent=2
                        )
                        + "\n"
                    )
                else:
                    if snapshot.revision:
                        sys.stderr.write(f"Catalog revision: {snapshot.revision}\n")
                    for identity, entry in listing["models"].items():
                        sys.stdout.write(
                            f"{display_text(identity)}\t{display_text(entry['name'])}\n"
                        )
                return 0
            if args.model is None:
                args.model = picker.model(snapshot.catalog.document)
            document, card = snapshot.read_model(
                args.model, cache, offline=args.offline
            )
        else:
            document = cache.read_document(
                document_location(args.card), offline=args.offline
            )
            card = ModelCard.parse(document.text)
        if args.list:
            _list_card(card, json_output=args.json)
            return 0
        if args.all_variants:
            return _update_publication_lock(args, document, card, cache, snapshot)
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
            args.output = picker.directory()
        plan = plan_fetch(
            document,
            selection,
            cache,
            overrides=_overrides(args.origin),
            lock_location=args.lockfile,
            offline=args.offline,
            no_lock=args.no_lock,
            update_lock=args.update_lock,
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
                sys.stdout.write(f"  {display_text(file['path'])} ({size})\n")
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
