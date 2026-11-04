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
from loom.gen.support.string_pool import CStringBank  # noqa: E402
from loom.gen.target.contracts.contract_fragments import (  # noqa: E402
    generate_contract_fragment_from_lower_rules,
)
from loom.gen.target.contracts.contract_index import (  # noqa: E402
    ContractCasePool,
    generate_contract_case_pool,
    generate_contract_index,
)
from loom.gen.target.contracts.lower_rules import (  # noqa: E402
    diagnostic_descriptor_names,
    generate_lower_rule_set_from_compiled,
)
from loom.target.contract_fragments import resolve_contract_fragment  # noqa: E402
from loom.target.contracts import compile_lower_rule_set  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate target contract and lower-rule C tables from Python data.",
        fromfile_prefix_chars="@",
    )
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
    parser.add_argument(
        "--contract-index",
        action="append",
        default=[],
        metavar="NAME:KEY,KEY",
        help="C initializer name and ordered fragment keys; may be repeated.",
    )
    parser.add_argument("--index-output", type=Path, help="Private policy index include.")
    parser.add_argument("--shared-strings-header", type=Path)
    parser.add_argument("--shared-strings-source", type=Path)
    parser.add_argument("--shared-strings-layout", type=Path)
    parser.add_argument("--shared-strings-public-header")
    parser.add_argument("--shared-strings-symbol")
    args = parser.parse_args(argv)
    if bool(args.contract_index) != bool(args.index_output):
        parser.error("--contract-index and --index-output must be supplied together")
    shared_string_args = (args.shared_strings_header, args.shared_strings_source, args.shared_strings_layout, args.shared_strings_public_header, args.shared_strings_symbol)
    if any(shared_string_args) and not all(shared_string_args):
        parser.error("all shared-string output paths, public header, and symbol must be supplied together")

    family_count = len(args.contract_fragment)
    for flag_name, paths in (
        ("--contract-header", args.contract_header),
        ("--contract-source", args.contract_source),
        ("--lower-rule-header", args.lower_rule_header),
        ("--lower-rule-source", args.lower_rule_source),
    ):
        if len(paths) != family_count:
            parser.error(f"{flag_name} has {len(paths)} values for {family_count} contract fragments")

    source_fragments = []
    for contract_fragment in args.contract_fragment:
        try:
            registration = resolve_contract_fragment(contract_fragment)
        except ValueError as exc:
            parser.error(str(exc))
        fragment = registration.load()
        dialect_ops = registration.load_dialect_ops()
        lower_rules = compile_lower_rule_set(fragment, dialect_ops=dialect_ops)
        source_fragments.append((registration.key, fragment, dialect_ops, lower_rules))

    shared_strings = None
    if args.shared_strings_layout:
        names = frozenset().union(*(diagnostic_descriptor_names(lower_rules) for _, _, _, lower_rules in source_fragments))
        if not names:
            parser.error("shared-string output requires descriptor diagnostic spellings")
        shared_strings = CStringBank.build(names, symbol=args.shared_strings_symbol, header=args.shared_strings_public_header)
        write_text_file(args.shared_strings_header, shared_strings.emit_header())
        write_text_file(args.shared_strings_source, shared_strings.emit_source())
        write_text_file(args.shared_strings_layout, shared_strings.manifest())

    compiled_fragments = {}
    for (
        (fragment_key, fragment, dialect_ops, lower_rules),
        contract_header,
        contract_source,
        lower_rule_header,
        lower_rule_source,
    ) in zip(
        source_fragments,
        args.contract_header,
        args.contract_source,
        args.lower_rule_header,
        args.lower_rule_source,
        strict=True,
    ):
        generated_contract = generate_contract_fragment_from_lower_rules(
            fragment,
            dialect_ops=dialect_ops,
            lower_rules=lower_rules,
        )
        generated_lower_rules = generate_lower_rule_set_from_compiled(
            fragment,
            compiled=lower_rules,
            shared_strings=shared_strings,
        )
        compiled_fragments[fragment_key] = (
            fragment,
            generated_contract.compiled,
            lower_rules,
        )
        for path, contents in (
            (contract_header, generated_contract.header),
            (contract_source, generated_contract.source),
            (lower_rule_header, generated_lower_rules.header),
            (lower_rule_source, generated_lower_rules.source),
        ):
            write_text_file(path, contents)
    index_specs = []
    initializer_names = set()
    family_binding_indices = {}
    for specification in args.contract_index:
        name, separator, keys = specification.partition(":")
        if not separator or not keys or name in initializer_names:
            parser.error(f"invalid or duplicate contract index: {specification}")
        initializer_names.add(name)
        fragments = []
        binding_indices = []
        for key in keys.split(","):
            registration = resolve_contract_fragment(key)
            if registration.key not in compiled_fragments:
                # Metadata-only fragments have no ordinary rule pool output.
                fragment = registration.load()
                dialect_ops = registration.load_dialect_ops()
                lower_rules = compile_lower_rule_set(fragment, dialect_ops=dialect_ops)
                generated = generate_contract_fragment_from_lower_rules(
                    fragment,
                    dialect_ops=dialect_ops,
                    lower_rules=lower_rules,
                )
                compiled_fragments[registration.key] = (
                    fragment,
                    generated.compiled,
                    None,
                )
            fragments.append(compiled_fragments[registration.key])
            if registration.key not in family_binding_indices:
                family_binding_indices[registration.key] = len(family_binding_indices)
            binding_indices.append(family_binding_indices[registration.key])
        index_specs.append((name, fragments, binding_indices))
    if args.index_output:
        case_pool = ContractCasePool("kLoomTargetContractFamilyCases")
        selection_blob_symbols = {}
        index_sources = [
            generate_contract_index(
                name,
                fragments,
                binding_indices,
                case_pool,
                selection_blob_symbols,
            )
            for name, fragments, binding_indices in index_specs
        ]
        write_text_file(
            args.index_output,
            "\n".join((generate_contract_case_pool(case_pool), *index_sources)),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
