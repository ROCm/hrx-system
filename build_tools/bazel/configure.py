#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates local Bazel configuration for this checkout."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_tools.bazel import configure_project
from build_tools.bazel.configure_api import ConfigRequest


def windows_symbolic_links_available() -> bool:
    """Returns whether this process can create Windows symbolic links."""
    with tempfile.TemporaryDirectory(prefix="hrx-bazel-symlink-") as temp_directory:
        directory = Path(temp_directory)
        target = directory / "target"
        link = directory / "link"
        target.write_text("", encoding="utf-8")
        try:
            link.symlink_to(target)
        except OSError:
            return False
        return link.is_symlink()


def require_windows_bazel_host(
    *,
    platform_name: str | None = None,
    symbolic_link_probe: Callable[[], bool] | None = None,
) -> None:
    """Fails with setup instructions for missing Windows Bazel capabilities."""
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "win32":
        return
    if symbolic_link_probe is None:
        symbolic_link_probe = windows_symbolic_links_available
    if symbolic_link_probe():
        return
    raise SystemExit(
        "Windows Bazel host requirements are not satisfied:\n\n"
        "Symbolic-link creation is unavailable. rules_python constructs "
        "Windows runfiles and runtime virtual environments with symbolic "
        "links. Enable Windows Developer Mode or grant this account the "
        "'Create symbolic links' user right.\n\n"
        "CI jobs without administrator rights must use a base image where "
        "symbolic-link creation is already provisioned."
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Writes .bazelrc.configured for this checkout. Defaults to the "
            "repository's ordinary recursive build scope."
        ),
        epilog="""Examples:
  python build_tools/bazel/configure.py
  python build_tools/bazel/configure.py -DNAME=VALUE
  python build_tools/bazel/configure.py --//package:setting=value

Portable project options and friendly presets are documented in BUILDING.md.
Other Bazel-native overrides belong in .bazelrc.local.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".bazelrc.configured"),
        help="Path to write. Defaults to .bazelrc.configured.",
    )
    parser.add_argument(
        "-D",
        action="append",
        default=[],
        dest="defines",
        metavar="NAME=VALUE",
        help="Portable project configuration option documented in BUILDING.md.",
    )
    parser.add_argument(
        "--check-windows-host",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args, native_bazel_args = parser.parse_known_args(argv)
    args.native_bazel_args = native_bazel_args
    return args


def parse_define(define: str) -> tuple[str, str]:
    if "=" not in define:
        raise SystemExit(f"-D{define} must use NAME=VALUE.")
    name, value = define.split("=", 1)
    name = name.split(":", 1)[0]
    if not name:
        raise SystemExit(f"-D{define} has no option name.")
    return name, value


def apply_define(request: ConfigRequest, define: str) -> None:
    name, value = parse_define(define)
    if configure_project.apply_define(request, name, value):
        return
    raise SystemExit(
        f"Unsupported Bazel configure option -D{name}. Published portable "
        "configuration options are documented in BUILDING.md."
    )


def apply_native_bazel_arg(request: ConfigRequest, arg: str) -> None:
    if arg == "--":
        raise SystemExit("build_tools/bazel/configure.py does not use a -- separator.")
    if configure_project.apply_native_bazel_arg(request, arg):
        return
    if arg.startswith("--repo_env="):
        raise SystemExit(
            f"Unsupported repository environment setting {arg!r}. Generated "
            "configuration accepts only documented project settings; put other "
            "Bazel-native overrides in .bazelrc.local."
        )
    raise SystemExit(
        f"Unsupported Bazel configure argument {arg!r}. Use documented -D "
        "project options or .bazelrc.local for checkout-specific Bazel overrides."
    )


def request_from_args(args: argparse.Namespace) -> ConfigRequest:
    request = ConfigRequest(output=args.output)
    configure_project.initialize_request(request)
    for define in args.defines:
        apply_define(request, define)
    for native_bazel_arg in args.native_bazel_args:
        apply_native_bazel_arg(request, native_bazel_arg)
    configure_project.resolve_request(request)
    configure_project.validate_request(request)
    return request


def render_config(request: ConfigRequest) -> str:
    lines = [
        "# AUTO-GENERATED by build_tools/bazel/configure.py; do not edit.",
        "# Re-run python build_tools/bazel/configure.py or use .bazelrc.local for local overrides.",
    ]
    for section in configure_project.config_sections(request):
        lines.extend(("", "# " + section.heading, *section.lines))
    lines.append("")
    return "\n".join(lines)


def generate_config(args: argparse.Namespace) -> str:
    return render_config(request_from_args(args))


def main() -> int:
    args = parse_arguments()
    require_windows_bazel_host()
    if args.check_windows_host:
        return 0
    request = request_from_args(args)
    config = render_config(request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config, encoding="utf-8")
    for line in configure_project.request_summary(request):
        print(line)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
