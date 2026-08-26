# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 module envelope, section grammar, and verification obligations."""

from __future__ import annotations

from model.module import (
    EnvelopeRecord,
    ModuleFormat,
    RecordFieldReference,
    Section,
    SectionRecord,
    ValidationObligation,
)
from model.module.v0.records import RECORDS_BY_KEY
from model.schema import ValidationScope
from model.specification import CORE_0, NormativeClause

FORMAT = ModuleFormat(
    entity_id="core.module.format",
    since=CORE_0,
    summary="IREE VM bytecode module container.",
    image_alignment=8,
    section_alignment=8,
    normative_text=(
        "The complete image is canonical little-endian immutable storage. The "
        "fixed header begins at byte zero, the section directory immediately "
        "follows it, and payloads occur in strictly increasing directory order "
        "without serialized payload offsets. Starting at the checked end of "
        "the directory, align the running cursor to the declared section "
        "alignment, require every skipped padding byte to be zero, take exactly "
        "the row's byte_length bytes, and repeat; the final unaligned payload "
        "end must equal the image length. All size, offset, count, alignment, "
        "and address calculations use widened checked arithmetic and reject "
        "before pointer formation. Reserved and alignment bytes are zero. "
        "Fixed table fields use natural alignment and ordinary C layouts; "
        "version zero supports only little-endian hosts and defines no "
        "byte-swapping representation. Instruction streams retain their own "
        "four-byte framing and do not become native object arrays. A header with "
        "zero directory rows is a valid empty module. An absent known collection "
        "section denotes an empty collection, a present known section is "
        "nonempty, and a live cross-reference requires its owning section. An "
        "unknown skippable section may be empty because its owning authority, "
        "not the older reader, defines its payload contract. A section type's "
        "high byte names its page authority: zero names Core, 0xF0..0xFD name "
        "architectural extension pages, and every extension authority must have "
        "an exact Requirements declaration. Every other nonzero authority is "
        "invalid even for skippable sections: 0xFE is reserved for noncanonical "
        "experiments and 0xFF for a future extended escape. The image contains "
        "no module name, host pointers, relocations, resolved type descriptors, "
        "process addresses, or native module records."
    ),
)

LIFECYCLE = NormativeClause(
    entity_id="core.module.contract.verification_lifetime",
    since=CORE_0,
    summary="Module verification, publication, and immutable-image lifetime.",
    dependencies=(FORMAT.entity_id,),
    normative_text=(
        "Bytecode module creation is the single verification boundary for "
        "every fact knowable from one image and its ref-type environment. It "
        "validates bounded planning fields before allocation, derives exact "
        "resolved-ref and rodata-root multiplicities, allocates one module "
        "slab, validates every section and cross-section relation, resolves "
        "ref types, and publishes nothing until every check succeeds. Only the "
        "module name, flat resolved-type handles, and rodata root objects need "
        "variable allocation tails; declarations and instructions remain "
        "mapped and require no decoded-table copies. Structural verification "
        "proves bytes, versions, ordinals, extents, declarations, instruction "
        "records, and exact control targets. It deliberately does not build a "
        "call graph, propagate may-yield, prove reachability or POD definite "
        "assignment, or compute ref-ownership dataflow. Unused declarations "
        "and duplicate payload values remain valid where their section permits "
        "them. The bytecode factory pairs immutable image storage with one caller-"
        "supplied nonempty generic module name. Private ref-counted image storage "
        "owns the span and deallocator and dominates the bytecode module plus "
        "every escaped image-backed rodata root; it has no public lifecycle. "
        "Program creation alone resolves cross-module imports, callable tokens, "
        "process-storage offsets, and the executable initialize export. Only the "
        "program's executable module is searched for the exact export name "
        "initialize; linked libraries have no implicit initialization. An absent "
        "initializer is a no-op. A present initializer accepts the process-"
        "creation arguments described by its ordinary callable signature, returns "
        "no results, and may yield only when its callable contract permits it. "
        "Process creation attaches zeroed module slices and invokes that function "
        "exactly once with construction authority across its transitive calls, so "
        "explicitly called library code may set its own immutable globals. Success "
        "seals every slice and publishes only complete state; failure unwinds and "
        "detaches every slice without publishing a process. The loader must keep "
        "the image bytes bitwise stable through the module and every escaped "
        "image-backed rodata lifetime. Mutation after verification is outside "
        "the VM contract and execution performs no per-instruction integrity "
        "check. This is not a sandbox boundary; isolation between mutually "
        "untrusted programs belongs at the process or machine boundary."
    ),
)

ENVELOPE_RECORDS = (
    EnvelopeRecord(
        entity_id="core.module.envelope.image_header",
        since=CORE_0,
        summary="Image envelope use of image_header.",
        format_id=FORMAT.entity_id,
        record_id=RECORDS_BY_KEY["image_header"].entity_id,
        document_order=0,
    ),
    EnvelopeRecord(
        entity_id="core.module.envelope.section_directory_row",
        since=CORE_0,
        summary="Image envelope use of section_directory_row.",
        format_id=FORMAT.entity_id,
        record_id=RECORDS_BY_KEY["section_directory_row"].entity_id,
        document_order=1,
    ),
)

SECTIONS = (
    Section(
        entity_id="core.module.section.requirements",
        since=CORE_0,
        summary="Module requirements section.",
        section_type=1,
        required_flags=0,
        grammar="nonempty fixed array of six-byte requirement rows",
        normative_text=(
            "The section is absent for a core-only image. When present it is a "
            "nonempty exact array of requirement rows strictly increasing by "
            "page_id. A page ID must be in the architectural extension range "
            "0xF0..0xFD. Loading rejects an unavailable page, a different major, "
            "or a required minor newer than the registered page. Every used "
            "page instruction and required page-owned section must have been "
            "introduced no later than the declared version. A supported "
            "declaration may conservatively name a newer minor than the image "
            "strictly uses; the loader does not reconstruct producer-minimal "
            "requirements."
        ),
    ),
    Section(
        entity_id="core.module.section.strings",
        since=CORE_0,
        summary="Module strings section.",
        section_type=2,
        required_flags=0,
        grammar="count header, count-plus-one u32 offsets, and exact UTF-8 byte tail",
        normative_text=(
            "string_count is in [1, 65535]. The count-plus-one offsets begin at "
            "zero, are monotonically nondecreasing and in range, and end at the "
            "exact trailing-byte length, which is therefore at most UINT32_MAX. "
            "Each indexed value is NUL-free valid UTF-8; empty and duplicate "
            "values are valid. Identity contexts additionally require nonempty "
            "bytes. Ordering and uniqueness compare referenced raw UTF-8 bytes, "
            "not ordinals. Required strings use direct u16 ordinals. Only fields "
            "explicitly declared nullable accept 0xFFFF; it is distinct from a "
            "present empty string and is not a generic ordinal convention."
        ),
    ),
    Section(
        entity_id="core.module.section.ref_types",
        since=CORE_0,
        summary="Module ref types section.",
        section_type=3,
        required_flags=0,
        grammar="group count, group rows, and prefix-summed type entry rows",
        normative_text=(
            "The section is absent when no ref type is used. group_count is in "
            "[1, 65535]; every group is nonempty; and the checked sum of entries "
            "is in [1, 65536]. Groups are strictly ordered and unique by raw "
            "namespace bytes, and entries within each group are strictly ordered "
            "and unique by raw local-name bytes. The flat module-local type "
            "ordinal is the zero-based traversal position through every group in "
            "row order; no base or sentinel is serialized. Stable type identity "
            "is the pair {namespace, local_name}, independent of callable module "
            "names. Module creation resolves each namespace once and stores the "
            "resulting exact descriptors in one flat module-local array."
        ),
    ),
    Section(
        entity_id="core.module.section.signatures",
        since=CORE_0,
        summary="Module signatures section.",
        section_type=4,
        required_flags=0,
        grammar="signature count, signature rows, and prefix-summed descriptors",
        normative_text=(
            "signature_count is in [1, 65536]. For each signature, argument "
            "descriptors occur first in source order and results follow in "
            "source order; signature descriptor blocks occur in signature "
            "ordinal order. descriptor_base is the checked running prefix and "
            "the final block consumes the section exactly. Scalars require type "
            "ordinal zero; refs and functions require exact local type ordinals "
            "and admit canonical null at typed runtime boundaries. Verification "
            "recomputes and exactly matches all six value/ref/function argument "
            "and result counts. Each logical argument and result total is at most "
            "65535. In each bank the first 16 positions use direct registers and "
            "the remainder use the canonical overflow packet. Duplicate "
            "structurally equal signatures are valid and no decoded signature "
            "table is required."
        ),
    ),
    Section(
        entity_id="core.module.section.callable_types",
        since=CORE_0,
        summary="Module callable types section.",
        section_type=5,
        required_flags=0,
        grammar="callable-type count followed by exactly that many fixed rows",
        normative_text=(
            "The section is absent when no callable contract is used; otherwise "
            "callable_type_count is in [1, 65536]. MAY_YIELD is permission: a "
            "non-yielding target satisfies either contract, while a yielding "
            "target cannot satisfy a contract with MAY_YIELD clear. Rows are "
            "dense and topologically ordered: every FUNCTION descriptor in the "
            "signature named by row N must name a callable ordinal below N. "
            "Recursive callable types are invalid in version zero. Structurally "
            "equal rows may have distinct ordinals; program creation may intern "
            "their structure without changing the image."
        ),
    ),
    Section(
        entity_id="core.module.section.imports",
        since=CORE_0,
        summary="Module imports section.",
        section_type=6,
        required_flags=0,
        grammar="group count, module group rows, and prefix-summed import entries",
        normative_text=(
            "The section is absent without imports. group_count is in [1, "
            "65535], each group is nonempty, and the checked entry sum is in "
            "[1, 65536]. Groups are strictly ordered and unique by target-module "
            "bytes. Entries within one group are strictly ordered by symbol "
            "bytes then callable-type ordinal, rejecting only duplicate exact "
            "rows. Flat import ordinal is the zero-based traversal position and "
            "has no sentinel or serialized base. OPTIONAL tolerates only an "
            "absent module or export; a present export must still satisfy the "
            "exact structural callable type and may-yield implication. Program "
            "creation resolves one target module per group and one function per "
            "entry, stores canonical zero for unresolved optional imports, fails "
            "unresolved required imports, and leaves calls with flat targets and "
            "no runtime string or group lookup."
        ),
    ),
    Section(
        entity_id="core.module.section.exports",
        since=CORE_0,
        summary="Module exports section.",
        section_type=7,
        required_flags=0,
        grammar="export count followed by exactly that many export rows",
        normative_text=(
            "The section is absent without exports; otherwise export_count is in "
            "[1, 65535]. Rows are strictly ordered and unique by raw public-name "
            "bytes. Each callable row is the complete public signature and "
            "conservative may-yield contract, and the named local function must "
            "uphold both. Distinct aliases may name one deduplicated function "
            "while retaining independent callable, presentation, and metadata "
            "declarations. The process initializer is selected by the ordinary "
            "exact export name initialize."
        ),
    ),
    Section(
        entity_id="core.module.section.functions",
        since=CORE_0,
        summary="Module functions section.",
        section_type=8,
        required_flags=0,
        grammar="function rows, all target entries, then all record streams",
        normative_text=(
            "The section is absent without bytecode functions; otherwise "
            "function_count is in [1, 65536]. All function-local switch-target "
            "ranges occur in function ordinal order, followed by all bytecode "
            "ranges in function ordinal order. switch_target_base and "
            "bytecode_offset are exact checked running prefixes and the final "
            "bytecode end consumes the section. bytecode_length is a nonzero "
            "multiple of four. block_count is in [1, 65536] and exactly equals "
            "the number of decoded control.block records in the function. Each "
            "register count is in [0, 256] and covers the larger direct "
            "argument/result prefix for its signature bank, capped at 16. "
            "Declared byte/ref/function local extents include the canonical "
            "outgoing call packet and must fit every decoded call. "
            "Every switch-table entry is a u32 four-byte-word offset to an exact "
            "decoded control.block in its owning function, including unused "
            "entries. Switch subranges may overlap and unused valid entries are "
            "allowed. Each switch checks target_base + target_count against the "
            "function-local table; an out-of-range selector follows the next "
            "instruction. Verification decodes the complete stream, validates "
            "every field and local extent, requires control.block first, rejects "
            "fallthrough past the final record, and proves every direct target. "
            "The mapped rows and canonical prefixes permit direct lookup without "
            "a decoded function table."
        ),
    ),
    Section(
        entity_id="core.module.section.constants",
        since=CORE_0,
        summary="Module constants section.",
        section_type=9,
        required_flags=0,
        grammar="nonempty fixed array of naturally aligned u64 cells",
        normative_text=(
            "The payload is an exact nonempty array of naturally aligned raw "
            "little-endian u64 cells with a count in [1, 65536]. Every u16 value "
            "is a direct ordinal at the maximum count; no sentinel exists. Cells "
            "carry arbitrary bits and equal cells at distinct ordinals are "
            "valid. An i32 pool load canonicalizes the low 32 bits and an i64 "
            "load preserves the complete cell. The format requires neither "
            "interning nor reachability and does not prefer pool over inline "
            "constants."
        ),
    ),
    Section(
        entity_id="core.module.section.globals",
        since=CORE_0,
        summary="Module globals section.",
        section_type=10,
        required_flags=0,
        grammar="32-byte three-domain header followed by ref and function descriptors",
        normative_text=(
            "The section is absent when all value, ref, and function counts are "
            "zero; a present all-zero header is invalid. Each total is in [0, "
            "65536], each immutable count is at most its total, and the exact "
            "section length is 32 + 4*ref_count + 4*function_count. Ref rows and "
            "then function rows occur in ordinal order and name exact local "
            "types; value globals need no rows because each is an untyped u64 "
            "cell. Each domain's immutable globals are its dense prefix and the "
            "remaining suffix is mutable. These layouts are private bytecode "
            "state inside the generic module's opaque process slice. Zeroed "
            "state starts with zero values and canonical-null refs/functions. "
            "During construction immutable stores are set-once and immutable "
            "loads require the set bit. Sealing requires every immutable value, "
            "permits omitted nullable ref/function globals as null, rejects "
            "required null globals, and thereafter forbids immutable stores. Ref "
            "cells are owning roots released on detach; function cells are "
            "non-owning process-bound values. No initializer payload, public "
            "name, process address, reflection record, or generic global layout "
            "is serialized."
        ),
    ),
    Section(
        entity_id="core.module.section.rodata",
        since=CORE_0,
        summary="Module rodata section.",
        section_type=11,
        required_flags=0,
        grammar="block-count header, u64 lengths, then aligned block payloads",
        normative_text=(
            "block_count is in [1, 65536]. After the header and complete u64 "
            "length array, derive each block in ordinal order by aligning the "
            "running cursor to eight, requiring skipped padding to be zero, and "
            "taking the exact declared length; the final block end consumes the "
            "section. Each block therefore has exactly eight-byte image-relative "
            "alignment. Zero-length and duplicate blocks remain distinct valid "
            "ordinals. Version zero promises no stronger alignment. Module load "
            "creates one READ-only vm.buffer root per block whose initial owner "
            "is held by the module and whose image-storage lifetime survives any "
            "escaped owner. buffer.rodata.load publishes an internal borrow that "
            "may be promoted to an ordinary owner without an ownership cycle."
        ),
    ),
    Section(
        entity_id="core.module.section.presentation",
        since=CORE_0,
        summary="Module presentation section.",
        section_type=12,
        required_flags=1,
        grammar="entry count, sparse declaration rows, and signature-derived field rows",
        normative_text=(
            "This skippable observational section is absent without presentation "
            "and otherwise has entry_count in [1, 131071]. Sparse rows are "
            "strictly ordered by the composite key (declaration_kind, "
            "declaration_ordinal), with imports before exports and no duplicates. "
            "Each row owns one field row for every logical argument in source "
            "order followed by every result in source order of its declaration's "
            "callable signature. field_base is the checked running prefix and the "
            "final field end consumes the section. Every string field uses "
            "0xFFFF as canonical null and each entry has at least one non-null "
            "function- or field-level value. Presentation belongs to the authored "
            "import/export declaration, so aliases of one function retain "
            "independent data; internal functions have none. Authored types are "
            "human-readable source contracts, not machine-parsed VM types. Loom "
            "captures the public logical type immediately before ABI-changing "
            "lowering and aligns field data with final logical VM positions. "
            "Consumers treat every string as untrusted text and perform their "
            "own language-specific name, collision, keyword, and escaping work."
        ),
    ),
    Section(
        entity_id="core.module.section.metadata",
        since=CORE_0,
        summary="Module metadata section.",
        section_type=13,
        required_flags=1,
        grammar="scope counts, sparse scope rows, entries, aligned offsets, and typed value bytes",
        normative_text=(
            "This skippable observational section is absent without metadata and "
            "otherwise has nonzero total_entry_count. Module entries occupy the "
            "first range; import scopes, then export scopes, strictly increase by "
            "declaration ordinal and partition the remaining entries through "
            "canonical entry_base prefixes. Scope counts cannot exceed their "
            "declaration domains and every scope is nonempty. Within each scope, "
            "keys are nonempty and strictly ordered and unique by raw UTF-8 "
            "bytes. Zero padding aligns the count-plus-one u64 offset array to "
            "eight. Offsets begin at zero, are nondecreasing and in range, and "
            "end at the exact trailing value-byte length; individual values have "
            "no alignment promise. BOOL is exactly one canonical byte; I64, U64, "
            "and F64 are exactly eight little-endian bytes, with every F64 bit "
            "pattern including NaN payloads preserved; UTF8 is any-length "
            "valid UTF-8 and may contain NUL; BYTES is opaque. Unknown nonzero "
            "types remain valid opaque typed spans. Equal adjacent offsets are "
            "legal only for variable-width or unknown types."
        ),
    ),
)
SECTIONS_BY_KEY = {
    section.entity_id.removeprefix("core.module.section."): section
    for section in SECTIONS
}

SECTION_RECORDS = (
    SectionRecord(
        entity_id="core.module.section.requirements.record.requirement_row",
        since=CORE_0,
        summary="requirements section use of requirement_row.",
        section_id=SECTIONS_BY_KEY["requirements"].entity_id,
        record_id=RECORDS_BY_KEY["requirement_row"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.strings.record.strings_header",
        since=CORE_0,
        summary="strings section use of strings_header.",
        section_id=SECTIONS_BY_KEY["strings"].entity_id,
        record_id=RECORDS_BY_KEY["strings_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.strings.record.string_offset",
        since=CORE_0,
        summary="strings section use of string_offset.",
        section_id=SECTIONS_BY_KEY["strings"].entity_id,
        record_id=RECORDS_BY_KEY["string_offset"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.ref_types.record.ref_types_header",
        since=CORE_0,
        summary="ref_types section use of ref_types_header.",
        section_id=SECTIONS_BY_KEY["ref_types"].entity_id,
        record_id=RECORDS_BY_KEY["ref_types_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.ref_types.record.ref_type_group_row",
        since=CORE_0,
        summary="ref_types section use of ref_type_group_row.",
        section_id=SECTIONS_BY_KEY["ref_types"].entity_id,
        record_id=RECORDS_BY_KEY["ref_type_group_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.ref_types.record.ref_type_entry_row",
        since=CORE_0,
        summary="ref_types section use of ref_type_entry_row.",
        section_id=SECTIONS_BY_KEY["ref_types"].entity_id,
        record_id=RECORDS_BY_KEY["ref_type_entry_row"].entity_id,
        document_order=2,
    ),
    SectionRecord(
        entity_id="core.module.section.signatures.record.signatures_header",
        since=CORE_0,
        summary="signatures section use of signatures_header.",
        section_id=SECTIONS_BY_KEY["signatures"].entity_id,
        record_id=RECORDS_BY_KEY["signatures_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.signatures.record.signature_row",
        since=CORE_0,
        summary="signatures section use of signature_row.",
        section_id=SECTIONS_BY_KEY["signatures"].entity_id,
        record_id=RECORDS_BY_KEY["signature_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.signatures.record.signature_descriptor_row",
        since=CORE_0,
        summary="signatures section use of signature_descriptor_row.",
        section_id=SECTIONS_BY_KEY["signatures"].entity_id,
        record_id=RECORDS_BY_KEY["signature_descriptor_row"].entity_id,
        document_order=2,
    ),
    SectionRecord(
        entity_id="core.module.section.callable_types.record.callable_types_header",
        since=CORE_0,
        summary="callable_types section use of callable_types_header.",
        section_id=SECTIONS_BY_KEY["callable_types"].entity_id,
        record_id=RECORDS_BY_KEY["callable_types_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.callable_types.record.callable_type_row",
        since=CORE_0,
        summary="callable_types section use of callable_type_row.",
        section_id=SECTIONS_BY_KEY["callable_types"].entity_id,
        record_id=RECORDS_BY_KEY["callable_type_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.imports.record.imports_header",
        since=CORE_0,
        summary="imports section use of imports_header.",
        section_id=SECTIONS_BY_KEY["imports"].entity_id,
        record_id=RECORDS_BY_KEY["imports_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.imports.record.import_group_row",
        since=CORE_0,
        summary="imports section use of import_group_row.",
        section_id=SECTIONS_BY_KEY["imports"].entity_id,
        record_id=RECORDS_BY_KEY["import_group_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.imports.record.import_entry_row",
        since=CORE_0,
        summary="imports section use of import_entry_row.",
        section_id=SECTIONS_BY_KEY["imports"].entity_id,
        record_id=RECORDS_BY_KEY["import_entry_row"].entity_id,
        document_order=2,
    ),
    SectionRecord(
        entity_id="core.module.section.exports.record.exports_header",
        since=CORE_0,
        summary="exports section use of exports_header.",
        section_id=SECTIONS_BY_KEY["exports"].entity_id,
        record_id=RECORDS_BY_KEY["exports_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.exports.record.export_row",
        since=CORE_0,
        summary="exports section use of export_row.",
        section_id=SECTIONS_BY_KEY["exports"].entity_id,
        record_id=RECORDS_BY_KEY["export_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.functions.record.functions_header",
        since=CORE_0,
        summary="functions section use of functions_header.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
        record_id=RECORDS_BY_KEY["functions_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.functions.record.function_row",
        since=CORE_0,
        summary="functions section use of function_row.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
        record_id=RECORDS_BY_KEY["function_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.functions.record.switch_target_entry",
        since=CORE_0,
        summary="functions section use of switch_target_entry.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
        record_id=RECORDS_BY_KEY["switch_target_entry"].entity_id,
        document_order=2,
    ),
    SectionRecord(
        entity_id="core.module.section.constants.record.constant_cell",
        since=CORE_0,
        summary="constants section use of constant_cell.",
        section_id=SECTIONS_BY_KEY["constants"].entity_id,
        record_id=RECORDS_BY_KEY["constant_cell"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.globals.record.globals_header",
        since=CORE_0,
        summary="globals section use of globals_header.",
        section_id=SECTIONS_BY_KEY["globals"].entity_id,
        record_id=RECORDS_BY_KEY["globals_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.globals.record.global_ref_descriptor_row",
        since=CORE_0,
        summary="globals section use of global_ref_descriptor_row.",
        section_id=SECTIONS_BY_KEY["globals"].entity_id,
        record_id=RECORDS_BY_KEY["global_ref_descriptor_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.globals.record.global_function_descriptor_row",
        since=CORE_0,
        summary="globals section use of global_function_descriptor_row.",
        section_id=SECTIONS_BY_KEY["globals"].entity_id,
        record_id=RECORDS_BY_KEY["global_function_descriptor_row"].entity_id,
        document_order=2,
    ),
    SectionRecord(
        entity_id="core.module.section.rodata.record.rodata_header",
        since=CORE_0,
        summary="rodata section use of rodata_header.",
        section_id=SECTIONS_BY_KEY["rodata"].entity_id,
        record_id=RECORDS_BY_KEY["rodata_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.rodata.record.rodata_block_length",
        since=CORE_0,
        summary="rodata section use of rodata_block_length.",
        section_id=SECTIONS_BY_KEY["rodata"].entity_id,
        record_id=RECORDS_BY_KEY["rodata_block_length"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.presentation.record.presentation_header",
        since=CORE_0,
        summary="presentation section use of presentation_header.",
        section_id=SECTIONS_BY_KEY["presentation"].entity_id,
        record_id=RECORDS_BY_KEY["presentation_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.presentation.record.presentation_entry_row",
        since=CORE_0,
        summary="presentation section use of presentation_entry_row.",
        section_id=SECTIONS_BY_KEY["presentation"].entity_id,
        record_id=RECORDS_BY_KEY["presentation_entry_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.presentation.record.presentation_field_row",
        since=CORE_0,
        summary="presentation section use of presentation_field_row.",
        section_id=SECTIONS_BY_KEY["presentation"].entity_id,
        record_id=RECORDS_BY_KEY["presentation_field_row"].entity_id,
        document_order=2,
    ),
    SectionRecord(
        entity_id="core.module.section.metadata.record.metadata_header",
        since=CORE_0,
        summary="metadata section use of metadata_header.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
        record_id=RECORDS_BY_KEY["metadata_header"].entity_id,
        document_order=0,
    ),
    SectionRecord(
        entity_id="core.module.section.metadata.record.metadata_scope_row",
        since=CORE_0,
        summary="metadata section use of metadata_scope_row.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
        record_id=RECORDS_BY_KEY["metadata_scope_row"].entity_id,
        document_order=1,
    ),
    SectionRecord(
        entity_id="core.module.section.metadata.record.metadata_entry_row",
        since=CORE_0,
        summary="metadata section use of metadata_entry_row.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
        record_id=RECORDS_BY_KEY["metadata_entry_row"].entity_id,
        document_order=2,
    ),
    SectionRecord(
        entity_id="core.module.section.metadata.record.metadata_value_offset",
        since=CORE_0,
        summary="metadata section use of metadata_value_offset.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
        record_id=RECORDS_BY_KEY["metadata_value_offset"].entity_id,
        document_order=3,
    ),
)

VALIDATION_OBLIGATIONS = (
    ValidationObligation(
        entity_id="core.validation.module.image.storage",
        since=CORE_0,
        summary="Require an eight-byte-aligned, immutable, lifetime-stable image span.",
        scope=ValidationScope.IMAGE,
        kind="validate_storage",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["image_header"].entity_id),),
        normative_text="Require an eight-byte-aligned, immutable, lifetime-stable image span.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.image.header",
        since=CORE_0,
        summary="Require the complete header and validate every fixed header field.",
        scope=ValidationScope.IMAGE,
        kind="validate_fixed_record",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["image_header"].entity_id),),
        normative_text="Require the complete header and validate every fixed header field.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.image.directory_extent",
        since=CORE_0,
        summary="Derive the directory end with checked widened multiplication/addition.",
        scope=ValidationScope.IMAGE,
        kind="checked_extent",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["image_header"].entity_id,
                "section_count_u16",
            ),
            RecordFieldReference(RECORDS_BY_KEY["section_directory_row"].entity_id),
        ),
        normative_text="Derive the directory end with checked widened multiplication/addition.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.directory.order",
        since=CORE_0,
        summary="Require strictly increasing section types with no duplicates.",
        scope=ValidationScope.DIRECTORY,
        kind="strict_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["section_directory_row"].entity_id,
                "section_type_u16",
            ),
        ),
        normative_text="Require strictly increasing section types with no duplicates.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.directory.rows",
        since=CORE_0,
        summary="Validate every directory row before deriving a payload view.",
        scope=ValidationScope.DIRECTORY,
        kind="validate_rows",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["section_directory_row"].entity_id),
        ),
        normative_text="Validate every directory row before deriving a payload view.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.directory.known_sections",
        since=CORE_0,
        summary="Require exact known-section flags and canonical nonempty payloads.",
        scope=ValidationScope.DIRECTORY,
        kind="validate_known_section_contract",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["section_directory_row"].entity_id,
                "section_type_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["section_directory_row"].entity_id,
                "section_flags_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["section_directory_row"].entity_id,
                "byte_length_u64",
            ),
        ),
        normative_text="Require exact known-section flags and canonical nonempty payloads.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.directory.unknown_sections",
        since=CORE_0,
        summary="Accept an unknown section only when its flags equal SKIPPABLE.",
        scope=ValidationScope.DIRECTORY,
        kind="validate_unknown_section_contract",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["section_directory_row"].entity_id,
                "section_type_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["section_directory_row"].entity_id,
                "section_flags_u16",
            ),
        ),
        normative_text="Accept an unknown section only when its flags equal SKIPPABLE.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.directory.payload_packing",
        since=CORE_0,
        summary="Derive eight-byte-aligned payloads, prove zero padding, and consume the image exactly.",
        scope=ValidationScope.DIRECTORY,
        kind="derive_packed_payloads",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["image_header"].entity_id,
                "section_count_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["section_directory_row"].entity_id,
                "byte_length_u64",
            ),
        ),
        normative_text="Derive eight-byte-aligned payloads, prove zero padding, and consume the image exactly.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.requirements.extent",
        since=CORE_0,
        summary="Require a nonzero payload whose length is an exact multiple of six.",
        scope=ValidationScope.SECTION,
        kind="fixed_array_extent",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["requirement_row"].entity_id),),
        normative_text="Require a nonzero payload whose length is an exact multiple of six.",
        section_id=SECTIONS_BY_KEY["requirements"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.requirements.order",
        since=CORE_0,
        summary="Require strictly increasing non-core page IDs.",
        scope=ValidationScope.SECTION,
        kind="strict_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["requirement_row"].entity_id,
                "page_id_u16",
            ),
        ),
        normative_text="Require strictly increasing non-core page IDs.",
        section_id=SECTIONS_BY_KEY["requirements"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.requirements.capabilities",
        since=CORE_0,
        summary="Match every declared page against one registered major/minor capability.",
        scope=ValidationScope.SECTION,
        kind="match_page_capabilities",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["requirement_row"].entity_id,
                "page_id_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["requirement_row"].entity_id,
                "major_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["requirement_row"].entity_id,
                "required_minor_u16",
            ),
        ),
        normative_text="Match every declared page against one registered major/minor capability.",
        section_id=SECTIONS_BY_KEY["requirements"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.requirements.feature_coverage",
        since=CORE_0,
        summary=(
            "Require every page feature to fit its declaration without "
            "reconstructing a minimum."
        ),
        scope=ValidationScope.SECTION,
        kind="validate_declared_features",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["requirement_row"].entity_id),),
        normative_text=(
            "Reject page instructions or required sections newer than the "
            "declared page requirement. A supported "
            "declaration may name a page or minor newer than the image strictly "
            "uses; the loader does not reconstruct or enforce producer-minimal "
            "requirements."
        ),
        section_id=SECTIONS_BY_KEY["requirements"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.strings.index_extent",
        since=CORE_0,
        summary="Derive and bounds-check the count-plus-one offset array.",
        scope=ValidationScope.SECTION,
        kind="checked_counted_extent",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["strings_header"].entity_id,
                "string_count_u32",
            ),
            RecordFieldReference(RECORDS_BY_KEY["string_offset"].entity_id),
        ),
        normative_text="Derive and bounds-check the count-plus-one offset array.",
        section_id=SECTIONS_BY_KEY["strings"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.strings.offsets",
        since=CORE_0,
        summary="Require offset zero, monotonic offsets, in-range values, and exact final byte coverage.",
        scope=ValidationScope.SECTION,
        kind="validate_offsets",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["string_offset"].entity_id,
                "byte_offset_u32",
            ),
        ),
        normative_text="Require offset zero, monotonic offsets, in-range values, and exact final byte coverage.",
        section_id=SECTIONS_BY_KEY["strings"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.strings.text",
        since=CORE_0,
        summary="Require every indexed byte range to be valid NUL-free UTF-8.",
        scope=ValidationScope.SECTION,
        kind="validate_utf8",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["string_offset"].entity_id),),
        normative_text="Require every indexed byte range to be valid NUL-free UTF-8.",
        section_id=SECTIONS_BY_KEY["strings"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.ref_types.extent",
        since=CORE_0,
        summary="Derive group and entry ranges from checked prefix sums and consume the section exactly.",
        scope=ValidationScope.SECTION,
        kind="checked_grouped_extent",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["ref_types_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["ref_type_group_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["ref_type_entry_row"].entity_id),
        ),
        normative_text="Derive group and entry ranges from checked prefix sums and consume the section exactly.",
        section_id=SECTIONS_BY_KEY["ref_types"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.ref_types.counts",
        since=CORE_0,
        summary="Require nonempty groups and a total entry count in one through 65,536.",
        scope=ValidationScope.SECTION,
        kind="validate_group_counts",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["ref_types_header"].entity_id,
                "group_count_u32",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["ref_type_group_row"].entity_id,
                "entry_count_u32",
            ),
        ),
        normative_text="Require nonempty groups and a total entry count in one through 65,536.",
        section_id=SECTIONS_BY_KEY["ref_types"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.ref_types.group_order",
        since=CORE_0,
        summary="Require namespace groups strictly ordered and unique by raw UTF-8 bytes.",
        scope=ValidationScope.SECTION,
        kind="strict_string_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["ref_type_group_row"].entity_id,
                "namespace_string_u16",
            ),
        ),
        normative_text="Require namespace groups strictly ordered and unique by raw UTF-8 bytes.",
        section_id=SECTIONS_BY_KEY["ref_types"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.ref_types.entry_order",
        since=CORE_0,
        summary="Require local names strictly ordered and unique by raw bytes within each group.",
        scope=ValidationScope.SECTION,
        kind="strict_group_entry_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["ref_type_entry_row"].entity_id,
                "type_name_string_u16",
            ),
        ),
        normative_text="Require local names strictly ordered and unique by raw bytes within each group.",
        section_id=SECTIONS_BY_KEY["ref_types"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.signatures.extent",
        since=CORE_0,
        summary="Derive the fixed rows and descriptor tail with checked arithmetic and consume the section exactly.",
        scope=ValidationScope.SECTION,
        kind="checked_counted_tail",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["signatures_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["signature_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["signature_descriptor_row"].entity_id),
        ),
        normative_text="Derive the fixed rows and descriptor tail with checked arithmetic and consume the section exactly.",
        section_id=SECTIONS_BY_KEY["signatures"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.signatures.descriptors",
        since=CORE_0,
        summary="Validate source-ordered argument and result descriptors and their exact types.",
        scope=ValidationScope.SECTION,
        kind="validate_signature_descriptors",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["signature_descriptor_row"].entity_id),
        ),
        normative_text=(
            "Validate scalar kinds or exact local ref/callable types in source "
            "order. For each signature, require every argument descriptor in "
            "source order followed by every result descriptor in source order. "
            "Require descriptor blocks in signature ordinal order. Validate "
            "each scalar kind with a zero type ordinal and each ref or function "
            "kind with an exact local ref-type or callable-type ordinal."
        ),
        section_id=SECTIONS_BY_KEY["signatures"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.signatures.physical_layout",
        since=CORE_0,
        summary="Require canonical descriptor bases, exact three-bank counts, bounded logical totals, and exact overflow extents.",
        scope=ValidationScope.SECTION,
        kind="validate_signature_layout",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["signature_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["signature_descriptor_row"].entity_id),
        ),
        normative_text="Require canonical descriptor bases, exact three-bank counts, bounded logical totals, and exact overflow extents.",
        section_id=SECTIONS_BY_KEY["signatures"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.callable_types.extent",
        since=CORE_0,
        summary="Require exactly callable_type_count fixed rows and no trailing bytes.",
        scope=ValidationScope.SECTION,
        kind="checked_counted_extent",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["callable_types_header"].entity_id,
                "callable_type_count_u32",
            ),
            RecordFieldReference(RECORDS_BY_KEY["callable_type_row"].entity_id),
        ),
        normative_text="Require exactly callable_type_count fixed rows and no trailing bytes.",
        section_id=SECTIONS_BY_KEY["callable_types"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.callable_types.rows",
        since=CORE_0,
        summary="Validate every signature ordinal and callable permission flag.",
        scope=ValidationScope.SECTION,
        kind="validate_callable_rows",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["callable_type_row"].entity_id),),
        normative_text="Validate every signature ordinal and callable permission flag.",
        section_id=SECTIONS_BY_KEY["callable_types"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.callable_types.topology",
        since=CORE_0,
        summary="Require nested FUNCTION descriptors to name only earlier callable types.",
        scope=ValidationScope.SECTION,
        kind="validate_callable_topology",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["callable_type_row"].entity_id,
                "signature_ordinal_u16",
            ),
            RecordFieldReference(RECORDS_BY_KEY["signature_descriptor_row"].entity_id),
        ),
        normative_text="Require nested FUNCTION descriptors to name only earlier callable types.",
        section_id=SECTIONS_BY_KEY["callable_types"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.imports.extent",
        since=CORE_0,
        summary="Derive group and entry ranges from checked prefix sums and consume the section exactly.",
        scope=ValidationScope.SECTION,
        kind="checked_grouped_extent",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["imports_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["import_group_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["import_entry_row"].entity_id),
        ),
        normative_text="Derive group and entry ranges from checked prefix sums and consume the section exactly.",
        section_id=SECTIONS_BY_KEY["imports"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.imports.counts",
        since=CORE_0,
        summary="Require nonempty groups and a total import count in one through 65,536.",
        scope=ValidationScope.SECTION,
        kind="validate_group_counts",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["imports_header"].entity_id,
                "group_count_u32",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["import_group_row"].entity_id,
                "entry_count_u32",
            ),
        ),
        normative_text="Require nonempty groups and a total import count in one through 65,536.",
        section_id=SECTIONS_BY_KEY["imports"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.imports.group_order",
        since=CORE_0,
        summary="Require module groups strictly ordered and unique by raw UTF-8 bytes.",
        scope=ValidationScope.SECTION,
        kind="strict_string_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["import_group_row"].entity_id,
                "module_name_string_u16",
            ),
        ),
        normative_text="Require module groups strictly ordered and unique by raw UTF-8 bytes.",
        section_id=SECTIONS_BY_KEY["imports"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.imports.entry_order",
        since=CORE_0,
        summary="Require entries to be strictly ordered by raw symbol bytes then callable-type ordinal.",
        scope=ValidationScope.SECTION,
        kind="strict_import_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["import_entry_row"].entity_id,
                "symbol_name_string_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["import_entry_row"].entity_id,
                "callable_type_ordinal_u16",
            ),
        ),
        normative_text="Require entries to be strictly ordered by raw symbol bytes then callable-type ordinal.",
        section_id=SECTIONS_BY_KEY["imports"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.exports.extent",
        since=CORE_0,
        summary="Require exactly export_count fixed rows and no trailing bytes.",
        scope=ValidationScope.SECTION,
        kind="checked_counted_extent",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["exports_header"].entity_id,
                "export_count_u32",
            ),
            RecordFieldReference(RECORDS_BY_KEY["export_row"].entity_id),
        ),
        normative_text="Require exactly export_count fixed rows and no trailing bytes.",
        section_id=SECTIONS_BY_KEY["exports"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.exports.order",
        since=CORE_0,
        summary="Require public names strictly ordered and unique by raw UTF-8 bytes.",
        scope=ValidationScope.SECTION,
        kind="strict_string_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["export_row"].entity_id,
                "name_string_u16",
            ),
        ),
        normative_text="Require public names strictly ordered and unique by raw UTF-8 bytes.",
        section_id=SECTIONS_BY_KEY["exports"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.extent",
        since=CORE_0,
        summary="Derive function rows, all target ranges, and all bytecode ranges with checked prefix sums.",
        scope=ValidationScope.SECTION,
        kind="checked_function_extent",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["functions_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["function_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["switch_target_entry"].entity_id),
        ),
        normative_text="Derive function rows, all target ranges, and all bytecode ranges with checked prefix sums.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.rows",
        since=CORE_0,
        summary="Validate signature, flags, register counts, local extents, and reserved fields.",
        scope=ValidationScope.SECTION,
        kind="validate_function_rows",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["function_row"].entity_id),),
        normative_text="Validate signature, flags, register counts, local extents, and reserved fields.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.signature_prefixes",
        since=CORE_0,
        summary="Require each register bank to cover its signature-derived direct argument and result prefixes.",
        scope=ValidationScope.SECTION,
        kind="validate_direct_prefix_capacity",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "signature_ordinal_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "value_register_count_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "ref_register_count_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "function_register_count_u16",
            ),
        ),
        normative_text=(
            "Require each register bank to cover its signature-derived direct "
            "argument and result prefixes. For each value, ref, and function "
            "bank, require the function's "
            "register count to be at least min(16, max(argument_count, "
            "result_count)) for that bank. Counts are derived from the "
            "function's own source-ordered signature descriptors."
        ),
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.call_packet",
        since=CORE_0,
        summary="Require every decoded call's target-signature packet to fit the caller's declared local storage.",
        scope=ValidationScope.SECTION,
        kind="validate_call_packet_fit",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "local_byte_length_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "local_ref_count_u32",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "local_function_count_u32",
            ),
        ),
        normative_text=(
            "Require every decoded call's target-signature packet to fit the "
            "caller's declared local storage. Resolve each direct call to its "
            "local function signature or import "
            "callable signature, and use the encoded callable signature for an "
            "indirect call. For each bank, require caller registers to cover "
            "min(16, max(argument_count, result_count)). Require "
            "local_byte_length to cover 8 * (max(argument_value_count - 16, "
            "0) + max(result_value_count - 16, 0)); require local_ref_count "
            "and local_function_count to cover the analogous sums of ref and "
            "function overflow cells. All arithmetic is checked before "
            "comparison."
        ),
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.decode",
        since=CORE_0,
        summary="Decode and structurally validate every instruction record with exact byte coverage.",
        scope=ValidationScope.SECTION,
        kind="decode_record_stream",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "bytecode_length_u32",
            ),
        ),
        normative_text="Decode and structurally validate every instruction record with exact byte coverage.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.block_count",
        since=CORE_0,
        summary="Require the declared block count to equal the number of decoded control.block records.",
        scope=ValidationScope.SECTION,
        kind="validate_block_count",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "block_count_u32",
            ),
        ),
        normative_text="Require the declared block count to equal the number of decoded control.block records.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.entry_block",
        since=CORE_0,
        summary="Require the first decoded record to be control.block.",
        scope=ValidationScope.SECTION,
        kind="validate_entry_block",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "bytecode_length_u32",
            ),
        ),
        normative_text="Require the first decoded record to be control.block.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.functions.targets",
        since=CORE_0,
        summary="Require every direct target and every function-local table entry to name a decoded control.block record.",
        scope=ValidationScope.SECTION,
        kind="validate_control_targets",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "switch_target_base_u32",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["function_row"].entity_id,
                "switch_target_entry_count_u32",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["switch_target_entry"].entity_id,
                "target_word_offset_u32",
            ),
        ),
        normative_text="Require every direct target and every function-local table entry to name a decoded control.block record.",
        section_id=SECTIONS_BY_KEY["functions"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.constants.extent",
        since=CORE_0,
        summary="Require one through 65,536 complete eight-byte cells and no trailing bytes.",
        scope=ValidationScope.SECTION,
        kind="fixed_array_extent",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["constant_cell"].entity_id),),
        normative_text="Require one through 65,536 complete eight-byte cells and no trailing bytes.",
        section_id=SECTIONS_BY_KEY["constants"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.globals.extent",
        since=CORE_0,
        summary="Use the explicit ref/function counts to require exact row-array coverage after the 32-byte header.",
        scope=ValidationScope.SECTION,
        kind="checked_global_extent",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["globals_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["global_ref_descriptor_row"].entity_id),
            RecordFieldReference(
                RECORDS_BY_KEY["global_function_descriptor_row"].entity_id
            ),
        ),
        normative_text="Use the explicit ref/function counts to require exact row-array coverage after the 32-byte header.",
        section_id=SECTIONS_BY_KEY["globals"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.globals.counts",
        since=CORE_0,
        summary="Bound all three totals, require one nonzero total, and validate every immutable prefix length.",
        scope=ValidationScope.SECTION,
        kind="validate_global_counts",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["globals_header"].entity_id),),
        normative_text="Bound all three totals, require one nonzero total, and validate every immutable prefix length.",
        section_id=SECTIONS_BY_KEY["globals"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.globals.refs",
        since=CORE_0,
        summary="Validate every exact ref type and NULLABLE flag.",
        scope=ValidationScope.SECTION,
        kind="validate_ref_descriptors",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["global_ref_descriptor_row"].entity_id),
        ),
        normative_text="Validate every exact ref type and NULLABLE flag.",
        section_id=SECTIONS_BY_KEY["globals"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.globals.functions",
        since=CORE_0,
        summary="Validate every exact callable type and NULLABLE flag.",
        scope=ValidationScope.SECTION,
        kind="validate_function_descriptors",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["global_function_descriptor_row"].entity_id
            ),
        ),
        normative_text="Validate every exact callable type and NULLABLE flag.",
        section_id=SECTIONS_BY_KEY["globals"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.rodata.extent",
        since=CORE_0,
        summary="Derive and bounds-check the complete u64 block-length array.",
        scope=ValidationScope.SECTION,
        kind="checked_counted_extent",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["rodata_header"].entity_id,
                "block_count_u32",
            ),
            RecordFieldReference(RECORDS_BY_KEY["rodata_block_length"].entity_id),
        ),
        normative_text="Derive and bounds-check the complete u64 block-length array.",
        section_id=SECTIONS_BY_KEY["rodata"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.rodata.blocks",
        since=CORE_0,
        summary="Derive eight-byte-aligned block views, prove zero padding, and consume the section exactly.",
        scope=ValidationScope.SECTION,
        kind="derive_aligned_blocks",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["rodata_block_length"].entity_id,
                "byte_length_u64",
            ),
        ),
        normative_text="Derive eight-byte-aligned block views, prove zero padding, and consume the section exactly.",
        section_id=SECTIONS_BY_KEY["rodata"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.presentation.extent",
        since=CORE_0,
        summary="Derive each field block from its import/export callable signature and consume the section exactly.",
        scope=ValidationScope.SECTION,
        kind="checked_presentation_extent",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["presentation_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["presentation_entry_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["presentation_field_row"].entity_id),
        ),
        normative_text=(
            "Derive each field block from its import/export callable signature "
            "and consume the section exactly. "
            "Require one field row for every argument in source order followed "
            "by every result in source order, require field_base to equal the "
            "checked running field prefix."
        ),
        section_id=SECTIONS_BY_KEY["presentation"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.presentation.order",
        since=CORE_0,
        summary="Require valid declaration identities and strictly increasing composite keys.",
        scope=ValidationScope.SECTION,
        kind="strict_declaration_order",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["presentation_entry_row"].entity_id,
                "declaration_kind_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["presentation_entry_row"].entity_id,
                "declaration_ordinal_u16",
            ),
        ),
        normative_text="Require valid declaration identities and strictly increasing composite keys.",
        section_id=SECTIONS_BY_KEY["presentation"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.presentation.nonempty",
        since=CORE_0,
        summary="Require at least one non-null presentation field for every row.",
        scope=ValidationScope.SECTION,
        kind="validate_semantic_presence",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["presentation_entry_row"].entity_id,
                "documentation_string_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["presentation_entry_row"].entity_id,
                "authored_type_string_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["presentation_field_row"].entity_id,
                "name_string_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["presentation_field_row"].entity_id,
                "authored_type_string_u16",
            ),
        ),
        normative_text="Require at least one non-null presentation field for every row.",
        section_id=SECTIONS_BY_KEY["presentation"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.metadata.extent",
        since=CORE_0,
        summary="Derive both scope arrays, entries, zero alignment padding, count-plus-one offsets, and exact value bytes.",
        scope=ValidationScope.SECTION,
        kind="checked_metadata_extent",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["metadata_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["metadata_scope_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["metadata_entry_row"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["metadata_value_offset"].entity_id),
        ),
        normative_text="Derive both scope arrays, entries, zero alignment padding, count-plus-one offsets, and exact value bytes.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.metadata.scopes",
        since=CORE_0,
        summary="Require ordered valid declaration scopes and canonical entry bases partitioning the shared entry array.",
        scope=ValidationScope.SECTION,
        kind="validate_metadata_scopes",
        inputs=(
            RecordFieldReference(RECORDS_BY_KEY["metadata_header"].entity_id),
            RecordFieldReference(RECORDS_BY_KEY["metadata_scope_row"].entity_id),
        ),
        normative_text="Require ordered valid declaration scopes and canonical entry bases partitioning the shared entry array.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.metadata.entries",
        since=CORE_0,
        summary="Require nonempty keys strictly byte-sorted and unique within every metadata scope.",
        scope=ValidationScope.SECTION,
        kind="validate_metadata_entries",
        inputs=(RecordFieldReference(RECORDS_BY_KEY["metadata_entry_row"].entity_id),),
        normative_text="Require nonempty keys strictly byte-sorted and unique within every metadata scope.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.metadata.values",
        since=CORE_0,
        summary="Require canonical offsets and validate known typed payloads while preserving unknown nonzero types as opaque spans.",
        scope=ValidationScope.SECTION,
        kind="validate_metadata_values",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["metadata_entry_row"].entity_id,
                "value_type_u16",
            ),
            RecordFieldReference(
                RECORDS_BY_KEY["metadata_value_offset"].entity_id,
                "byte_offset_u64",
            ),
        ),
        normative_text="Require canonical offsets and validate known typed payloads while preserving unknown nonzero types as opaque spans.",
        section_id=SECTIONS_BY_KEY["metadata"].entity_id,
    ),
    ValidationObligation(
        entity_id="core.validation.module.cross_section.presence",
        since=CORE_0,
        summary="Require every live string, type, signature, function, pool, global, and rodata reference to have its owning section.",
        scope=ValidationScope.CROSS_SECTION,
        kind="validate_live_presence",
        inputs=(),
        normative_text="Require every live string, type, signature, function, pool, global, and rodata reference to have its owning section.",
    ),
    ValidationObligation(
        entity_id="core.validation.module.cross_section.versions",
        since=CORE_0,
        summary="Require every page authority and versioned feature to be declared and supported.",
        scope=ValidationScope.CROSS_SECTION,
        kind="validate_feature_versions",
        inputs=(
            RecordFieldReference(
                RECORDS_BY_KEY["image_header"].entity_id,
                "core_required_minor_u16",
            ),
            RecordFieldReference(RECORDS_BY_KEY["requirement_row"].entity_id),
        ),
        normative_text=(
            "Require each non-Core section authority to be in 0xF0..0xFD and "
            "have an exact Requirements declaration, "
            "and require every used instruction, selector, flag, and section "
            "feature to be declared by its owning supported version authority."
        ),
    ),
    ValidationObligation(
        entity_id="core.validation.module.cross_section.publication",
        since=CORE_0,
        summary="Publish no bytecode module until all section-local and cross-section checks succeed.",
        scope=ValidationScope.CROSS_SECTION,
        kind="publish_atomically",
        inputs=(),
        normative_text="Publish no bytecode module until all section-local and cross-section checks succeed.",
    ),
)

ENTITIES = (
    FORMAT,
    LIFECYCLE,
    *ENVELOPE_RECORDS,
    *SECTIONS,
    *SECTION_RECORDS,
    *VALIDATION_OBLIGATIONS,
)
