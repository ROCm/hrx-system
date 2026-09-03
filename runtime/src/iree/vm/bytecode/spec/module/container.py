# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM image envelope, sections, and structural obligations."""

from iree.vm.bytecode.spec.module import (
    ModuleFormat,
    RecordFieldReference,
    Section,
    StructuralConstraint,
    WireRecord,
)
from iree.vm.bytecode.spec.module import records as module_records
from iree.vm.bytecode.spec.module.numeric import NUMERIC_TABLES
from iree.vm.bytecode.spec.version import CORE_0


def _ref(record: WireRecord, field_name: str | None = None) -> RecordFieldReference:
    return RecordFieldReference(record, field_name)


def _constraint(
    name: str,
    contract: str,
    *inputs: RecordFieldReference,
) -> StructuralConstraint:
    return StructuralConstraint(name, CORE_0, inputs, contract)


def _section(
    name: str,
    section_type: int,
    summary: str,
    contract: str,
    records: tuple[WireRecord, ...],
    constraints: tuple[StructuralConstraint, ...],
    *,
    required_flags: int = 0,
) -> Section:
    return Section(
        name,
        section_type,
        required_flags,
        CORE_0,
        summary,
        contract,
        records,
        constraints,
    )


REQUIREMENTS = _section(
    "requirements",
    0x0001,
    "Declares the non-Core architectural pages required by the image.",
    (
        "The section is absent for a Core-only image. When present it is a nonempty "
        "exact array of requirement rows strictly increasing by page_id. A page ID "
        "must be in the architectural extension range 0xF0 through 0xFD. Loading "
        "rejects an unavailable page, a different major, or a required minor newer "
        "than the registered page. Every used page instruction and required "
        "page-owned section must have been introduced no later than the declared "
        "version. A supported declaration may conservatively name a newer minor "
        "than the image strictly uses; the loader does not reconstruct a "
        "producer-minimal requirement."
    ),
    (module_records.REQUIREMENT_ROW,),
    (
        _constraint(
            "extent",
            "Require a nonzero payload whose length is an exact multiple of six.",
            _ref(module_records.REQUIREMENT_ROW),
        ),
        _constraint(
            "order",
            "Require strictly increasing non-Core page IDs.",
            _ref(module_records.REQUIREMENT_ROW, "page_id_u16"),
        ),
        _constraint(
            "capabilities",
            (
                "Match every declared page ID and major against one registered page "
                "whose supported minor is at least the required minor."
            ),
            _ref(module_records.REQUIREMENT_ROW, "page_id_u16"),
            _ref(module_records.REQUIREMENT_ROW, "major_u16"),
            _ref(module_records.REQUIREMENT_ROW, "required_minor_u16"),
        ),
        _constraint(
            "feature_coverage",
            (
                "Reject page instructions or required sections newer than the "
                "declared page requirement. A declaration may conservatively name a "
                "newer supported minor; verification does not infer a minimum."
            ),
            _ref(module_records.REQUIREMENT_ROW),
        ),
    ),
)

STRINGS = _section(
    "strings",
    0x0002,
    "Stores the module's shared length-delimited UTF-8 strings.",
    (
        "string_count is in [1, 65535]. The count-plus-one offsets begin at zero, "
        "are monotonically nondecreasing and in range, and end at the exact trailing "
        "byte length, which is at most UINT32_MAX. Each indexed value is NUL-free "
        "valid UTF-8; empty and duplicate values are valid. Identity contexts "
        "additionally require nonempty bytes. Ordering and uniqueness compare raw "
        "UTF-8 bytes, not ordinals. Required strings use direct u16 ordinals. Only "
        "explicitly nullable fields accept 0xFFFF; it is distinct from an empty "
        "string and is not a generic ordinal convention."
    ),
    (module_records.STRINGS_HEADER, module_records.STRING_OFFSET),
    (
        _constraint(
            "index_extent",
            "Derive and bounds-check the count-plus-one offset array.",
            _ref(module_records.STRINGS_HEADER, "string_count_u32"),
            _ref(module_records.STRING_OFFSET),
        ),
        _constraint(
            "offsets",
            (
                "Require offset zero, monotonic in-range offsets, and exact final "
                "coverage of the trailing byte area."
            ),
            _ref(module_records.STRING_OFFSET, "byte_offset_u32"),
        ),
        _constraint(
            "text",
            "Require every indexed byte range to be valid NUL-free UTF-8.",
            _ref(module_records.STRING_OFFSET),
        ),
    ),
)

REF_TYPES = _section(
    "ref_types",
    0x0003,
    "Declares the module-local reference-type ordinal space.",
    (
        "The section is absent when no ref type is used. group_count is in [1, "
        "65535]; every group is nonempty; and the checked sum of entries is in [1, "
        "65536]. Groups are strictly ordered and unique by raw namespace bytes, and "
        "entries within each group are strictly ordered and unique by raw local-name "
        "bytes. The flat ordinal is the zero-based traversal position through all "
        "groups; no base or sentinel is serialized. Stable type identity is the pair "
        "{namespace, local_name}, independent of callable module names. Module "
        "creation resolves each namespace once and stores exact descriptors in one "
        "flat module-local array."
    ),
    (
        module_records.REF_TYPES_HEADER,
        module_records.REF_TYPE_GROUP_ROW,
        module_records.REF_TYPE_ENTRY_ROW,
    ),
    (
        _constraint(
            "extent",
            (
                "Derive group and entry ranges from checked prefix sums and consume "
                "the section exactly."
            ),
            _ref(module_records.REF_TYPES_HEADER),
            _ref(module_records.REF_TYPE_GROUP_ROW),
            _ref(module_records.REF_TYPE_ENTRY_ROW),
        ),
        _constraint(
            "counts",
            "Require nonempty groups and a total entry count in [1, 65536].",
            _ref(module_records.REF_TYPES_HEADER, "group_count_u32"),
            _ref(module_records.REF_TYPE_GROUP_ROW, "entry_count_u32"),
        ),
        _constraint(
            "group_order",
            "Require namespace groups strictly ordered and unique by raw UTF-8 bytes.",
            _ref(module_records.REF_TYPE_GROUP_ROW, "namespace_string_u16"),
        ),
        _constraint(
            "entry_order",
            (
                "Require local names strictly ordered and unique by raw bytes within "
                "each namespace group."
            ),
            _ref(module_records.REF_TYPE_ENTRY_ROW, "type_name_string_u16"),
        ),
    ),
)

SIGNATURES = _section(
    "signatures",
    0x0004,
    "Declares source-ordered logical function signatures.",
    (
        "signature_count is in [1, 65536]. For each signature, argument descriptors "
        "occur first in source order and results follow in source order; descriptor "
        "blocks occur in signature ordinal order. descriptor_base is the checked "
        "running prefix and the final block consumes the section exactly. Scalars "
        "require type ordinal zero; refs and functions require exact local type "
        "ordinals and admit canonical null at typed runtime boundaries. Verification "
        "recomputes and exactly matches all six value/ref/function argument and result "
        "counts. Each logical argument and result total is at most 65535. In each "
        "bank the first 16 positions use direct registers and the remainder use the "
        "canonical overflow packet. Duplicate structurally equal signatures are valid "
        "and require no decoded signature table."
    ),
    (
        module_records.SIGNATURES_HEADER,
        module_records.SIGNATURE_ROW,
        module_records.SIGNATURE_DESCRIPTOR_ROW,
    ),
    (
        _constraint(
            "extent",
            (
                "Derive the fixed rows and descriptor tail with checked arithmetic "
                "and consume the section exactly."
            ),
            _ref(module_records.SIGNATURES_HEADER),
            _ref(module_records.SIGNATURE_ROW),
            _ref(module_records.SIGNATURE_DESCRIPTOR_ROW),
        ),
        _constraint(
            "descriptors",
            (
                "Require argument descriptors in source order followed by result "
                "descriptors in source order, with blocks in signature ordinal order. "
                "Scalar kinds require a zero type ordinal; ref and function kinds "
                "require exact local ref-type and callable-type ordinals."
            ),
            _ref(module_records.SIGNATURE_DESCRIPTOR_ROW),
        ),
        _constraint(
            "physical_layout",
            (
                "Require canonical descriptor bases, exact three-bank counts, bounded "
                "logical totals, and exact overflow extents."
            ),
            _ref(module_records.SIGNATURE_ROW),
            _ref(module_records.SIGNATURE_DESCRIPTOR_ROW),
        ),
    ),
)

CALLABLE_TYPES = _section(
    "callable_types",
    0x0005,
    "Declares canonical structural callable contracts.",
    (
        "The section is absent when no callable contract is used; otherwise the count "
        "is in [1, 65536]. MAY_YIELD is permission: a non-yielding target satisfies "
        "either contract, while a yielding target cannot satisfy a contract with the "
        "flag clear. Rows are dense, unique, and strictly ordered by nesting depth, "
        "structural signature, then flags. Structural signatures compare argument "
        "count and source-ordered argument types followed by result count and "
        "source-ordered result types. Scalar kinds compare by ID, ref types by "
        "canonical namespace and local-name order, and nested callable types by an "
        "earlier canonical ordinal. Every FUNCTION descriptor in row N names an "
        "ordinal below N. A row's nesting depth is zero without FUNCTION descriptors "
        "and otherwise one plus the maximum referenced depth. Recursive callable "
        "types are invalid in Core 0.0."
    ),
    (module_records.CALLABLE_TYPES_HEADER, module_records.CALLABLE_TYPE_ROW),
    (
        _constraint(
            "extent",
            "Require exactly callable_type_count fixed rows and no trailing bytes.",
            _ref(module_records.CALLABLE_TYPES_HEADER, "callable_type_count_u32"),
            _ref(module_records.CALLABLE_TYPE_ROW),
        ),
        _constraint(
            "rows",
            "Validate every signature ordinal, permission flag, and reserved field.",
            _ref(module_records.CALLABLE_TYPE_ROW),
        ),
        _constraint(
            "topology",
            "Require nested FUNCTION descriptors to name only earlier callable types.",
            _ref(module_records.CALLABLE_TYPE_ROW, "signature_ordinal_u16"),
            _ref(module_records.SIGNATURE_DESCRIPTOR_ROW),
        ),
        _constraint(
            "depth",
            (
                "Require depth zero for leaf signatures and one plus the maximum child "
                "depth for signatures containing FUNCTION descriptors."
            ),
            _ref(module_records.CALLABLE_TYPE_ROW, "nesting_depth_u16"),
            _ref(module_records.SIGNATURE_DESCRIPTOR_ROW),
        ),
        _constraint(
            "order",
            (
                "Require strictly increasing nesting-depth, structural-signature, and "
                "flags order with no duplicate callable contracts."
            ),
            _ref(module_records.CALLABLE_TYPE_ROW),
            _ref(module_records.SIGNATURE_ROW),
            _ref(module_records.SIGNATURE_DESCRIPTOR_ROW),
        ),
    ),
)

IMPORTS = _section(
    "imports",
    0x0006,
    "Declares external callable dependencies grouped by target module.",
    (
        "The section is absent without imports. group_count is in [1, 65535], each "
        "group is nonempty, and the checked entry sum is in [1, 65536]. Groups are "
        "strictly ordered and unique by target-module bytes. Entries within one group "
        "are strictly ordered by symbol bytes then callable-type ordinal, rejecting "
        "only duplicate exact rows. Flat import ordinal is the zero-based traversal "
        "position and has no sentinel or serialized base. OPTIONAL tolerates only an "
        "absent module or export; a present export must satisfy the exact structural "
        "callable type and may-yield implication. Program creation resolves one "
        "target module per group and one function per entry, stores canonical zero for "
        "unresolved optional imports, fails unresolved required imports, and leaves "
        "calls with flat targets and no runtime string or group lookup."
    ),
    (
        module_records.IMPORTS_HEADER,
        module_records.IMPORT_GROUP_ROW,
        module_records.IMPORT_ENTRY_ROW,
    ),
    (
        _constraint(
            "extent",
            (
                "Derive group and entry ranges from checked prefix sums and consume "
                "the section exactly."
            ),
            _ref(module_records.IMPORTS_HEADER),
            _ref(module_records.IMPORT_GROUP_ROW),
            _ref(module_records.IMPORT_ENTRY_ROW),
        ),
        _constraint(
            "counts",
            "Require nonempty groups and a total import count in [1, 65536].",
            _ref(module_records.IMPORTS_HEADER, "group_count_u32"),
            _ref(module_records.IMPORT_GROUP_ROW, "entry_count_u32"),
        ),
        _constraint(
            "group_order",
            "Require module groups strictly ordered and unique by raw UTF-8 bytes.",
            _ref(module_records.IMPORT_GROUP_ROW, "module_name_string_u16"),
        ),
        _constraint(
            "entry_order",
            (
                "Require entries to be strictly ordered by raw symbol bytes then "
                "callable-type ordinal."
            ),
            _ref(module_records.IMPORT_ENTRY_ROW, "symbol_name_string_u16"),
            _ref(module_records.IMPORT_ENTRY_ROW, "callable_type_ordinal_u16"),
        ),
    ),
)

EXPORTS = _section(
    "exports",
    0x0007,
    "Declares the module's public callable names.",
    (
        "The section is absent without exports; otherwise export_count is in [1, "
        "65535]. Rows are strictly ordered and unique by raw public-name bytes. Each "
        "row is the complete public signature and conservative may-yield contract, "
        "and the named local function must uphold both. Distinct aliases may name one "
        "deduplicated function while retaining independent callable, presentation, "
        "and metadata declarations. The process initializer is selected by the exact "
        "ordinary export name initialize."
    ),
    (module_records.EXPORTS_HEADER, module_records.EXPORT_ROW),
    (
        _constraint(
            "extent",
            "Require exactly export_count fixed rows and no trailing bytes.",
            _ref(module_records.EXPORTS_HEADER, "export_count_u32"),
            _ref(module_records.EXPORT_ROW),
        ),
        _constraint(
            "order",
            "Require public names strictly ordered and unique by raw UTF-8 bytes.",
            _ref(module_records.EXPORT_ROW, "name_string_u16"),
        ),
    ),
)

FUNCTIONS = _section(
    "functions",
    0x0008,
    "Stores bytecode function declarations, switch targets, and record streams.",
    (
        "The section is absent without bytecode functions; otherwise function_count "
        "is in [1, 65536]. All function-local switch-target ranges occur in function "
        "ordinal order, followed by all bytecode ranges in function ordinal order. "
        "switch_target_base and bytecode_offset are exact checked running prefixes "
        "and the final bytecode end consumes the section. bytecode_length is a "
        "nonzero multiple of four. block_count is in [1, 65536] and exactly equals "
        "the number of decoded control.block records. Each register count is in [0, "
        "256] and covers the larger direct argument/result prefix for its signature "
        "bank, capped at 16. Declared byte/ref/function local extents include the "
        "canonical outgoing call packet and must fit every decoded call. Every switch "
        "entry is a u32 four-byte-word offset to an exact decoded control.block in its "
        "owning function, including unused entries. Switch subranges may overlap. An "
        "out-of-range selector follows the next instruction. Verification decodes the "
        "complete stream, validates every field and local extent, requires "
        "control.block first, rejects fallthrough past the final record, and proves "
        "every direct target. The mapped rows and canonical prefixes permit direct "
        "lookup without a decoded function table."
    ),
    (
        module_records.FUNCTIONS_HEADER,
        module_records.FUNCTION_ROW,
        module_records.SWITCH_TARGET_ENTRY,
    ),
    (
        _constraint(
            "extent",
            (
                "Derive function rows, all target ranges, and all bytecode ranges with "
                "checked prefix sums. Require every bytecode length to be a nonzero "
                "multiple of four and consume the section exactly."
            ),
            _ref(module_records.FUNCTIONS_HEADER),
            _ref(module_records.FUNCTION_ROW),
            _ref(module_records.SWITCH_TARGET_ENTRY),
        ),
        _constraint(
            "rows",
            (
                "Validate callable types, flags, register counts, local extents, and "
                "reserved fields."
            ),
            _ref(module_records.FUNCTION_ROW),
        ),
        _constraint(
            "signature_prefixes",
            (
                "For each value, ref, and function bank, require the register count "
                "to cover min(16, max(argument_count, result_count)) from the "
                "function's source-ordered signature descriptors."
            ),
            _ref(module_records.FUNCTION_ROW, "callable_type_ordinal_u16"),
            _ref(module_records.FUNCTION_ROW, "value_register_count_u16"),
            _ref(module_records.FUNCTION_ROW, "ref_register_count_u16"),
            _ref(module_records.FUNCTION_ROW, "function_register_count_u16"),
        ),
        _constraint(
            "call_packet",
            (
                "Require every decoded call's target-signature packet to fit the "
                "caller's declared local storage. Resolve direct calls to their local "
                "or import signatures and use the encoded signature for indirect "
                "calls. Each register bank covers min(16, max(argument_count, "
                "result_count)). local_byte_length covers 8 times the combined value "
                "argument and result overflow beyond 16; local_ref_count and "
                "local_function_count cover the analogous overflow cells. All "
                "arithmetic is checked before comparison."
            ),
            _ref(module_records.FUNCTION_ROW, "local_byte_length_u16"),
            _ref(module_records.FUNCTION_ROW, "local_ref_count_u32"),
            _ref(module_records.FUNCTION_ROW, "local_function_count_u32"),
        ),
        _constraint(
            "decode",
            (
                "Decode and structurally validate every instruction record with exact "
                "byte coverage and no fallthrough past the final record."
            ),
            _ref(module_records.FUNCTION_ROW, "bytecode_length_u32"),
        ),
        _constraint(
            "block_count",
            (
                "Require the declared block count to equal the number of decoded "
                "control.block records."
            ),
            _ref(module_records.FUNCTION_ROW, "block_count_u32"),
        ),
        _constraint(
            "entry_block",
            "Require the first decoded record to be control.block.",
            _ref(module_records.FUNCTION_ROW, "bytecode_length_u32"),
        ),
        _constraint(
            "targets",
            (
                "Require every direct target and every function-local table entry to "
                "name an exact decoded control.block record."
            ),
            _ref(module_records.FUNCTION_ROW, "switch_target_base_u32"),
            _ref(module_records.FUNCTION_ROW, "switch_target_entry_count_u32"),
            _ref(module_records.SWITCH_TARGET_ENTRY, "target_word_offset_u32"),
        ),
    ),
)

CONSTANTS = _section(
    "constants",
    0x0009,
    "Stores module-local untyped constant value cells.",
    (
        "The payload is an exact nonempty array of naturally aligned raw "
        "little-endian u64 cells with a count in [1, 65536]. Every u16 value is a "
        "direct ordinal at the maximum count; no sentinel exists. Cells carry "
        "arbitrary bits and equal cells at distinct ordinals are valid. An i32 pool "
        "load canonicalizes the low 32 bits and an i64 load preserves the complete "
        "cell. The format requires neither interning nor reachability and does not "
        "prefer pool over inline constants."
    ),
    (module_records.CONSTANT_CELL,),
    (
        _constraint(
            "extent",
            "Require [1, 65536] complete eight-byte cells and no trailing bytes.",
            _ref(module_records.CONSTANT_CELL),
        ),
    ),
)

GLOBALS = _section(
    "globals",
    0x000A,
    "Declares private per-process value, reference, and function globals.",
    (
        "The section is absent when all three totals are zero; a present all-zero "
        "header is invalid. Each total is in [0, 65536], each immutable count is at "
        "most its total, and the exact length is 32 + 4*ref_count + "
        "4*function_count. Ref rows and then function rows occur in ordinal order and "
        "name exact local types; value globals need no rows because each is an "
        "untyped u64 cell. Each domain's immutable globals are its dense prefix and "
        "the remaining suffix is mutable. These layouts are private bytecode state "
        "inside the generic module's opaque process slice. Zeroed state starts with "
        "zero values and canonical-null refs/functions. During construction immutable "
        "stores are set-once and immutable loads require the set bit. Sealing requires "
        "every immutable value, permits omitted nullable ref/function globals as null, "
        "rejects required null globals, and thereafter forbids immutable stores. Ref "
        "cells are owning roots released on detach; function cells are non-owning "
        "process-bound values. No initializer payload, public name, process address, "
        "reflection record, or generic global layout is serialized."
    ),
    (
        module_records.GLOBALS_HEADER,
        module_records.GLOBAL_REF_DESCRIPTOR_ROW,
        module_records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
    ),
    (
        _constraint(
            "extent",
            (
                "Use the explicit ref and function counts to require exact descriptor "
                "array coverage after the 32-byte header."
            ),
            _ref(module_records.GLOBALS_HEADER),
            _ref(module_records.GLOBAL_REF_DESCRIPTOR_ROW),
            _ref(module_records.GLOBAL_FUNCTION_DESCRIPTOR_ROW),
        ),
        _constraint(
            "counts",
            (
                "Bound all three totals, require one nonzero total, and require each "
                "immutable prefix length not to exceed its domain total."
            ),
            _ref(module_records.GLOBALS_HEADER),
        ),
        _constraint(
            "refs",
            "Validate every exact ref type and NULLABLE flag.",
            _ref(module_records.GLOBAL_REF_DESCRIPTOR_ROW),
        ),
        _constraint(
            "functions",
            "Validate every exact callable type and NULLABLE flag.",
            _ref(module_records.GLOBAL_FUNCTION_DESCRIPTOR_ROW),
        ),
    ),
)

RODATA = _section(
    "rodata",
    0x000B,
    "Stores aligned module-local read-only byte blocks.",
    (
        "block_count is in [1, 65536]. Every descriptor carries an exact byte length "
        "and positive power-of-two minimum alignment; the section row's payload "
        "alignment equals the maximum of eight and every block alignment. After the "
        "header and complete descriptor array, each block is derived in ordinal order "
        "by aligning the running cursor, requiring skipped padding to be zero, and "
        "taking the declared length; the final block end consumes the section. "
        "Zero-length and duplicate blocks remain distinct valid ordinals. Module load "
        "creates one READ-only vm.buffer root per block whose initial owner is held by "
        "the module and whose storage lifetime survives any escaped owner. A block "
        "maps the image directly when its actual address meets the declared alignment; "
        "otherwise module creation copies it into an aligned tail of the same exact "
        "module slab. buffer.rodata.load publishes an internal borrow that may be "
        "promoted to an ordinary owner without an ownership cycle."
    ),
    (module_records.RODATA_HEADER, module_records.RODATA_BLOCK_DESCRIPTOR),
    (
        _constraint(
            "extent",
            "Derive and bounds-check the complete block-descriptor array.",
            _ref(module_records.RODATA_HEADER, "block_count_u32"),
            _ref(module_records.RODATA_BLOCK_DESCRIPTOR),
        ),
        _constraint(
            "blocks",
            (
                "Derive minimum-aligned block views, prove zero padding, and consume "
                "the section exactly. Require the directory payload alignment to "
                "equal the maximum of eight and all declared block alignments."
            ),
            _ref(module_records.RODATA_BLOCK_DESCRIPTOR, "byte_length_u64"),
            _ref(module_records.RODATA_BLOCK_DESCRIPTOR, "minimum_alignment_u32"),
            _ref(module_records.SECTION_DIRECTORY_ROW, "payload_alignment_u32"),
        ),
    ),
)

PRESENTATION = _section(
    "presentation",
    0x000C,
    "Carries optional human-facing data for public declarations.",
    (
        "This skippable observational section is absent without presentation data and "
        "otherwise has entry_count in [1, 131071]. Sparse rows are strictly ordered "
        "by (declaration_kind, declaration_ordinal), with imports before exports and "
        "no duplicates. Each row owns one field row for every machine argument in "
        "source order followed by every machine result in source order of its "
        "declaration's callable signature. When ABI lowering expands one authored "
        "aggregate into multiple machine fields, its optional name and authored type "
        "anchor on the first field and continuation fields use canonical null strings. "
        "field_base is the checked running prefix and the final field end consumes the "
        "section. Every string field uses 0xFFFF as canonical null and each entry has "
        "at least one non-null function- or field-level value. Presentation belongs "
        "to the authored import or export declaration, so aliases retain independent "
        "data; internal functions have none. Authored types are human-readable source "
        "contracts, not machine-parsed VM types. Consumers treat every string as "
        "untrusted text and perform language-specific name, collision, keyword, and "
        "escaping work."
    ),
    (
        module_records.PRESENTATION_HEADER,
        module_records.PRESENTATION_ENTRY_ROW,
        module_records.PRESENTATION_FIELD_ROW,
    ),
    (
        _constraint(
            "extent",
            (
                "Derive each field block from its declaration's callable signature "
                "and consume the section exactly. Require arguments in source order "
                "followed by results in source order and field_base equal to the "
                "checked running prefix."
            ),
            _ref(module_records.PRESENTATION_HEADER),
            _ref(module_records.PRESENTATION_ENTRY_ROW),
            _ref(module_records.PRESENTATION_FIELD_ROW),
        ),
        _constraint(
            "order",
            (
                "Require valid declaration identities and strictly increasing "
                "(declaration_kind, declaration_ordinal) keys."
            ),
            _ref(module_records.PRESENTATION_ENTRY_ROW, "declaration_kind_u16"),
            _ref(module_records.PRESENTATION_ENTRY_ROW, "declaration_ordinal_u16"),
        ),
        _constraint(
            "nonempty",
            "Require at least one non-null presentation field for every entry.",
            _ref(module_records.PRESENTATION_ENTRY_ROW, "documentation_string_u16"),
            _ref(module_records.PRESENTATION_ENTRY_ROW, "authored_type_string_u16"),
            _ref(module_records.PRESENTATION_FIELD_ROW, "name_string_u16"),
            _ref(module_records.PRESENTATION_FIELD_ROW, "authored_type_string_u16"),
        ),
    ),
    required_flags=1,
)

METADATA = _section(
    "metadata",
    0x000D,
    "Carries optional typed metadata for the module and public declarations.",
    (
        "This skippable observational section is absent without metadata and "
        "otherwise has nonzero total_entry_count. Module entries occupy the first "
        "range; import scopes, then export scopes, strictly increase by declaration "
        "ordinal and partition the remaining entries through canonical entry_base "
        "prefixes. Scope counts cannot exceed their declaration domains and every "
        "scope is nonempty. Within each scope, keys are nonempty and strictly ordered "
        "and unique by raw UTF-8 bytes. Zero padding aligns the count-plus-one u64 "
        "offset array to eight. Offsets begin at zero, are nondecreasing and in range, "
        "and end at the exact trailing value-byte length; individual values have no "
        "alignment promise. BOOL is exactly one canonical byte; I64, U64, and F64 are "
        "exactly eight little-endian bytes, with every F64 bit pattern including NaN "
        "payloads preserved; UTF8 is any-length valid UTF-8 and may contain NUL; "
        "BYTES is opaque. Unknown nonzero types remain valid opaque typed spans. Equal "
        "adjacent offsets are legal only for variable-width or unknown types."
    ),
    (
        module_records.METADATA_HEADER,
        module_records.METADATA_SCOPE_ROW,
        module_records.METADATA_ENTRY_ROW,
        module_records.METADATA_VALUE_OFFSET,
    ),
    (
        _constraint(
            "extent",
            (
                "Derive both scope arrays, entries, zero alignment padding, the "
                "count-plus-one offsets, and exact trailing value bytes."
            ),
            _ref(module_records.METADATA_HEADER),
            _ref(module_records.METADATA_SCOPE_ROW),
            _ref(module_records.METADATA_ENTRY_ROW),
            _ref(module_records.METADATA_VALUE_OFFSET),
        ),
        _constraint(
            "scopes",
            (
                "Require ordered valid declaration scopes and canonical entry bases "
                "partitioning the shared entry array."
            ),
            _ref(module_records.METADATA_HEADER),
            _ref(module_records.METADATA_SCOPE_ROW),
        ),
        _constraint(
            "entries",
            (
                "Require nonempty keys strictly byte-sorted and unique within every "
                "metadata scope."
            ),
            _ref(module_records.METADATA_ENTRY_ROW),
        ),
        _constraint(
            "values",
            (
                "Require canonical offsets and exact known typed payload encodings "
                "while preserving unknown nonzero types as opaque spans."
            ),
            _ref(module_records.METADATA_ENTRY_ROW, "value_type_u16"),
            _ref(module_records.METADATA_VALUE_OFFSET, "byte_offset_u64"),
        ),
    ),
    required_flags=1,
)

SECTIONS = (
    REQUIREMENTS,
    STRINGS,
    REF_TYPES,
    SIGNATURES,
    CALLABLE_TYPES,
    IMPORTS,
    EXPORTS,
    FUNCTIONS,
    CONSTANTS,
    GLOBALS,
    RODATA,
    PRESENTATION,
    METADATA,
)

MODULE_CONSTRAINTS = (
    _constraint(
        "image_storage",
        "Require an eight-byte-aligned, immutable, lifetime-stable image span.",
        _ref(module_records.IMAGE_HEADER),
    ),
    _constraint(
        "image_header",
        "Require the complete header and validate every fixed header field.",
        _ref(module_records.IMAGE_HEADER),
    ),
    _constraint(
        "image_directory_extent",
        "Derive the directory end with checked widened multiplication and addition.",
        _ref(module_records.IMAGE_HEADER, "section_count_u16"),
        _ref(module_records.SECTION_DIRECTORY_ROW),
    ),
    _constraint(
        "directory_order",
        "Require strictly increasing section types with no duplicates.",
        _ref(module_records.SECTION_DIRECTORY_ROW, "section_type_u16"),
    ),
    _constraint(
        "directory_rows",
        "Validate every directory row before deriving a payload view.",
        _ref(module_records.SECTION_DIRECTORY_ROW),
    ),
    _constraint(
        "directory_known_sections",
        "Require exact known-section flags and canonical nonempty payloads.",
        _ref(module_records.SECTION_DIRECTORY_ROW, "section_type_u16"),
        _ref(module_records.SECTION_DIRECTORY_ROW, "section_flags_u16"),
        _ref(module_records.SECTION_DIRECTORY_ROW, "byte_length_u64"),
    ),
    _constraint(
        "directory_unknown_sections",
        "Accept an unknown section only when its flags equal SKIPPABLE.",
        _ref(module_records.SECTION_DIRECTORY_ROW, "section_type_u16"),
        _ref(module_records.SECTION_DIRECTORY_ROW, "section_flags_u16"),
    ),
    _constraint(
        "directory_payload_packing",
        (
            "Starting at the checked directory end, derive each row-aligned payload, "
            "prove skipped padding is zero, and consume the image exactly."
        ),
        _ref(module_records.IMAGE_HEADER, "section_count_u16"),
        _ref(module_records.SECTION_DIRECTORY_ROW, "byte_length_u64"),
        _ref(module_records.SECTION_DIRECTORY_ROW, "payload_alignment_u32"),
    ),
    _constraint(
        "cross_section_presence",
        (
            "Require every live string, type, signature, function, pool, global, and "
            "rodata reference to have its owning section."
        ),
    ),
    _constraint(
        "cross_section_versions",
        (
            "Require each non-Core section authority to be in 0xF0 through 0xFD and "
            "have an exact Requirements declaration. Every used instruction, "
            "selector, flag, and section feature must be declared by its owning "
            "supported version authority."
        ),
        _ref(module_records.IMAGE_HEADER, "core_required_minor_u16"),
        _ref(module_records.REQUIREMENT_ROW),
    ),
    _constraint(
        "cross_section_publication",
        (
            "Publish no bytecode module until every section-local and cross-section "
            "check succeeds."
        ),
    ),
)

_IMAGE_CONTRACT = (
    "The complete image is canonical little-endian immutable storage. The fixed "
    "header begins at byte zero, the section directory immediately follows it, and "
    "payloads occur in strictly increasing directory order without serialized "
    "payload offsets. Starting at the checked end of the directory, align the running "
    "cursor to each row's declared payload alignment, require every skipped padding "
    "byte to be zero, take exactly byte_length bytes, and repeat; the final unaligned "
    "payload end must equal the image length. All size, offset, count, alignment, and "
    "address calculations use widened checked arithmetic and reject before pointer "
    "formation. Each payload alignment is a power of two at least eight. Reserved and "
    "alignment-padding bytes are zero. Fixed table fields use natural alignment and "
    "ordinary C layouts; Core 0.0 supports only little-endian hosts and defines no "
    "byte-swapping representation. Instruction streams retain four-byte framing and "
    "do not become native object arrays. A header with zero directory rows is a valid "
    "empty module. An absent known collection section denotes an empty collection, a "
    "present known section is nonempty, and a live cross-reference requires its "
    "owning section. An unknown skippable section may be empty because its authority, "
    "not an older reader, defines its payload contract. A section type's high byte "
    "names its page authority: zero names Core and 0xF0 through 0xFD name architectural "
    "extension pages, each with an exact Requirements declaration. Every other "
    "nonzero authority is invalid even for skippable sections: 0xFE is reserved for "
    "noncanonical experiments and 0xFF for a future extended escape. The image "
    "contains no module name, host pointers, relocations, resolved type descriptors, "
    "process addresses, or native module records."
)

_LIFECYCLE_CONTRACT = (
    "Bytecode module creation is the single verification boundary for every fact "
    "knowable from one image and its ref-type environment. It validates bounded "
    "planning fields before allocation, derives exact resolved-ref and rodata-root "
    "multiplicities, allocates one module slab, validates every section and "
    "cross-section relation, resolves ref types, and publishes nothing until every "
    "check succeeds. Only the module name, flat resolved-type handles, and rodata root "
    "objects need variable allocation tails; declarations and instructions remain "
    "mapped and require no decoded-table copies. Structural verification proves bytes, "
    "versions, ordinals, extents, declarations, instruction records, and exact control "
    "targets. It deliberately does not build a call graph, propagate may-yield, prove "
    "reachability or POD definite assignment, or compute ref-ownership dataflow. "
    "Unused declarations and duplicate payload values remain valid where their section "
    "permits them. The bytecode factory pairs immutable image storage with one "
    "caller-supplied nonempty generic module name. Private ref-counted image storage "
    "owns the span and deallocator and dominates the bytecode module plus every escaped "
    "image-backed rodata root; it has no public lifecycle. Program creation alone "
    "resolves cross-module imports, callable tokens, process-storage offsets, and the "
    "executable initialize export. Only the program's executable module is searched "
    "for that exact export name; linked libraries have no implicit initialization. An "
    "absent initializer is a no-op. A present initializer accepts ordinary "
    "process-creation arguments, returns no results, and may yield only when its "
    "callable contract permits it. Process creation attaches zeroed module slices and "
    "invokes that function exactly once with construction authority across transitive "
    "calls, so explicitly called library code may set its own immutable globals. "
    "Success seals every slice and publishes only complete state; failure unwinds and "
    "detaches every slice without publishing a process. The loader keeps image bytes "
    "bitwise stable through the module and every escaped image-backed rodata lifetime. "
    "Mutation after verification is outside the VM contract and execution performs no "
    "per-instruction integrity check. This is not a sandbox boundary; isolation between "
    "mutually untrusted programs belongs at the process or machine boundary."
)

MODULE_FORMAT = ModuleFormat(
    CORE_0,
    8,
    8,
    "Canonical mmap-compatible IREE VM bytecode module container.",
    _IMAGE_CONTRACT + "\n\n" + _LIFECYCLE_CONTRACT,
    NUMERIC_TABLES,
    (module_records.IMAGE_HEADER, module_records.SECTION_DIRECTORY_ROW),
    SECTIONS,
    MODULE_CONSTRAINTS,
)
