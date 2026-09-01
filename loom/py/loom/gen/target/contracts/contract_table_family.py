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
    compile_contract_fragment_from_lower_rules,
    generate_contract_fragment_from_compiled,
)
from loom.gen.target.contracts.contract_sets import (  # noqa: E402
    ContractSetGenerationInput,
    generate_contract_sets,
)
from loom.gen.target.contracts.lower_rules import (  # noqa: E402
    generate_lower_rule_set_from_compiled,
)
from loom.target.contract_fragments import resolve_contract_fragment  # noqa: E402
from loom.target.contracts import (  # noqa: E402
    CONTRACT_ROW_NONE,
    compile_contract_set,
    compile_lower_rule_set,
)


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
        "--contract-fragment-stem",
        action="append",
        required=True,
        metavar="STEM",
        help="Output stem paired with each contract fragment.",
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
    parser.add_argument(
        "--contract-set",
        action="append",
        default=[],
        metavar="NAME=STEM[,STEM...]",
        help="Ordered contract set assembled from generated fragment stems.",
    )
    parser.add_argument(
        "--contract-set-header",
        type=Path,
        help="Generated contract-set family header path.",
    )
    parser.add_argument(
        "--contract-set-source",
        type=Path,
        help="Generated contract-set family source path.",
    )
    args = parser.parse_args(argv)

    family_count = len(args.contract_fragment)
    for flag_name, paths in (
        ("--contract-fragment-stem", args.contract_fragment_stem),
        ("--contract-header", args.contract_header),
        ("--contract-source", args.contract_source),
        ("--lower-rule-header", args.lower_rule_header),
        ("--lower-rule-source", args.lower_rule_source),
    ):
        if len(paths) != family_count:
            parser.error(f"{flag_name} has {len(paths)} values for {family_count} contract fragments")

    if args.contract_set:
        if args.contract_set_header is None or args.contract_set_source is None:
            parser.error("--contract-set-header and --contract-set-source are required with --contract-set")
    elif args.contract_set_header is not None or args.contract_set_source is not None:
        parser.error("contract-set outputs require at least one --contract-set")

    known_fragment_stems = set(args.contract_fragment_stem)
    contract_set_specs = []
    for contract_set_spec in args.contract_set:
        set_name, separator, fragment_stem_list = contract_set_spec.partition("=")
        fragment_stems = tuple(fragment_stem_list.split(","))
        if not separator or not set_name or not fragment_stems or "" in fragment_stems:
            parser.error(f"invalid contract set '{contract_set_spec}'; expected NAME=STEM[,STEM...]")
        if len(set(fragment_stems)) != len(fragment_stems):
            parser.error(f"contract set '{set_name}' contains duplicate fragments")
        unknown_stems = [stem for stem in fragment_stems if stem not in known_fragment_stems]
        if unknown_stems:
            parser.error(f"contract set '{set_name}' references unknown fragment stems: " + ", ".join(unknown_stems))
        contract_set_specs.append((set_name, fragment_stems))

    fragments_by_stem = {}
    compiled_fragments_by_stem = {}
    lower_rules_by_stem = {}

    for (
        contract_fragment,
        contract_fragment_stem,
        contract_header,
        contract_source,
        lower_rule_header,
        lower_rule_source,
    ) in zip(
        args.contract_fragment,
        args.contract_fragment_stem,
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
        compiled_fragment = compile_contract_fragment_from_lower_rules(
            fragment,
            dialect_ops=dialect_ops,
            lower_rules=lower_rules,
        )
        generated_contract = generate_contract_fragment_from_compiled(
            fragment,
            compiled=compiled_fragment,
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

        if contract_fragment_stem in fragments_by_stem:
            parser.error(f"contract fragment stem '{contract_fragment_stem}' is duplicated")
        fragments_by_stem[contract_fragment_stem] = fragment
        compiled_fragments_by_stem[contract_fragment_stem] = compiled_fragment
        lower_rules_by_stem[contract_fragment_stem] = lower_rules

    set_inputs = []
    for set_name, fragment_stems in contract_set_specs:
        fragments = tuple(fragments_by_stem[stem] for stem in fragment_stems)
        compiled_fragments = tuple(compiled_fragments_by_stem[stem] for stem in fragment_stems)
        lower_rules = tuple(lower_rules_by_stem[stem] for stem in fragment_stems)
        compiled_set = compile_contract_set(
            set_name,
            compiled_fragments,
        )
        for fragment_stem, binding, rules in zip(
            fragment_stems,
            compiled_set.bindings,
            lower_rules,
            strict=True,
        ):
            index_has_lower_rules = binding.rule_set_index != CONTRACT_ROW_NONE
            if index_has_lower_rules != bool(rules.rules):
                parser.error(f"contract fragment '{fragment_stem}' case index and lower-rule table disagree")
        set_inputs.append(
            ContractSetGenerationInput(
                compiled=compiled_set,
                fragments=fragments,
            )
        )

    if set_inputs:
        generated_sets = generate_contract_sets(
            set_inputs,
            public_header=args.contract_set_header.name,
        )
        write_text_file(args.contract_set_header, generated_sets.header)
        write_text_file(args.contract_set_source, generated_sets.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
