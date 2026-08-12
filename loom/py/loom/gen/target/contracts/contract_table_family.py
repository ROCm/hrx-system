# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Python target contract fragment -> complete C table family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[4]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.target.contracts.contract_fragments import (  # noqa: E402
    generate_contract_fragment_from_lower_rules,
)
from loom.gen.target.contracts.lower_rules import (  # noqa: E402
    generate_lower_rule_set_from_compiled,
)
from loom.target.contract_fragments import resolve_contract_fragment  # noqa: E402
from loom.target.contracts import compile_lower_rule_set  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate target contract and lower-rule C tables from Python data.")
    parser.add_argument(
        "--contract-fragment",
        action="append",
        required=True,
        metavar="KEY",
        help="Contract fragment key or alias to generate; may be repeated.",
    )
    parser.add_argument(
        "--contract-header",
        action="append",
        required=True,
        type=Path,
        help="Generated contract fragment header path; may be repeated.",
    )
    parser.add_argument(
        "--contract-source",
        action="append",
        required=True,
        type=Path,
        help="Generated contract fragment source path; may be repeated.",
    )
    parser.add_argument(
        "--lower-rule-header",
        action="append",
        required=True,
        type=Path,
        help="Generated lower-rule header path; may be repeated.",
    )
    parser.add_argument(
        "--lower-rule-source",
        action="append",
        required=True,
        type=Path,
        help="Generated lower-rule source path; may be repeated.",
    )
    args = parser.parse_args(argv)

    family_count = len(args.contract_fragment)
    for flag_name, paths in (
        ("--contract-header", args.contract_header),
        ("--contract-source", args.contract_source),
        ("--lower-rule-header", args.lower_rule_header),
        ("--lower-rule-source", args.lower_rule_source),
    ):
        if len(paths) != family_count:
            parser.error(f"{flag_name} has {len(paths)} values for {family_count} contract fragments")

    for (
        contract_fragment,
        contract_header,
        contract_source,
        lower_rule_header,
        lower_rule_source,
    ) in zip(
        args.contract_fragment,
        args.contract_header,
        args.contract_source,
        args.lower_rule_header,
        args.lower_rule_source,
        strict=True,
    ):
        try:
            registration = resolve_contract_fragment(contract_fragment)
        except ValueError as exc:
            parser.error(str(exc))
        fragment = registration.load()
        dialect_ops = registration.load_dialect_ops()
        lower_rules = compile_lower_rule_set(fragment, dialect_ops=dialect_ops)
        generated_contract = generate_contract_fragment_from_lower_rules(
            fragment,
            dialect_ops=dialect_ops,
            lower_rules=lower_rules,
        )
        generated_lower_rules = generate_lower_rule_set_from_compiled(
            fragment,
            compiled=lower_rules,
        )
        for path, contents in (
            (contract_header, generated_contract.header),
            (contract_source, generated_contract.source),
            (lower_rule_header, generated_lower_rules.header),
            (lower_rule_source, generated_lower_rules.source),
        ):
            write_text_file(path, contents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
