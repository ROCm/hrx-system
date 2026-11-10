# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line entry point for normalized Dependabot advisory events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from build_tools.dependencies.github_alerts import (
    build_advisory_events,
    normalize_alert_pages,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize saved GitHub Dependabot alerts as JSONL."
    )
    parser.add_argument(
        "--alerts",
        default="-",
        help="GitHub alert response JSON, or - for stdin.",
    )
    args = parser.parse_args()

    if args.alerts == "-":
        value = json.load(sys.stdin)
    else:
        with Path(args.alerts).open(encoding="utf-8") as file:
            value = json.load(file)
    for event in build_advisory_events(normalize_alert_pages(value)):
        print(json.dumps(event, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
