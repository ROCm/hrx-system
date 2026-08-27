# Loom Agent Guide

This guide applies to work under `loom/`. More specific subtree guides can add
target rules, but they inherit the ownership split here.

## Responsibility Boundaries

Python owns source-of-truth data that already lives naturally in Python:
dialect declarations, descriptor schemas, target facts, vendor machine-readable
ISA data, generated contract rows, spelling tables, and build-time validation.
Invariants known when the compiler is generated belong in Python and fail the
build or generator test there.

C owns the shipping compiler surface: structs, enums, public and private
headers, semantic comments, function declarations, hot-path interpreters,
lowering mechanics, value-fact queries, diagnostics, and policy that depends on
user IR. Checked-in C headers are design documents and should be the place a
human or agent finds the representation contract by search.

Generated files are not design documents. A generator that emits a public struct
or enum for a backend table is usually hiding the contract in the wrong place.
Move the type to a checked-in header and have Python emit rows that initialize
that type.

## IR Represents Programs, Not Pass State

Loom IR is never an inter-pass mailbox. An operation, attribute, operand, type,
or region belongs in the IR only when it represents structurally coherent
program semantics or an explicit user-facing compilation contract. The
representation must remain meaningful when authored in text, parsed from
bytecode, cloned, canonicalized, transformed independently, and inspected
without hidden knowledge of the pass that produced it or the later pass that
plans to consume it.

Analysis and planning state lives in analysis results, pipeline-owned plan
objects, or explicit lowering interfaces with defined ownership, lifetime,
invalidation, and translation rules. An analysis never inserts synthetic
identity operations, zero-result sentinels, hint operations, fake operands, or
metadata attributes merely to make later passes remember its decisions. Using
SSA use lists or ordinary cloning to transport compiler bookkeeping is a
representation failure: it changes the program seen by every intervening pass
and makes those passes accidental participants in the analysis protocol.

Names such as `compiler_owned`, `temporary`, `marker`, `hint`, `annotation`, or
`metadata` grant no exception. If a construct would be incoherent or useless
when written by a user, it is not an IR construct. A narrow exception exists
for deliberate user control over compilation, such as a schedule or resource
contract owned by the operation it governs. Such a contract is part of the
documented dialect semantics, verifies without private producer state, and
fails loudly when the selected pipeline cannot implement it. Candidate lists,
analysis snapshots, repair recipes, provenance checksums, and pass-local
decisions remain compiler state even when exposing them as IR would make
cloning convenient.

State that must survive rewriting creates an infrastructure question rather
than permission to encode a side table as IR. The design must identify the
state owner, mutation and invalidation boundaries, how cloning or lowering
translates it, and when it is consumed. A general provenance or rewrite
tracking facility must be justified as reusable compiler infrastructure and
must remain separate from program semantics.

## Operation IR Boundaries

Every operation field carries stable public semantics; compiler caches,
resolved facts, provider or table IDs, feature masks, and pass state remain
outside IR. Raw encodings such as `contract_feature_bits` are invalid even in
tests, while typed attributes may use private physical payloads only when text
and bytecode preserve the stable public value. Work that authors or changes
operations follows `loom/py/loom/dialect/README.md`.

## Assembly Formatting Is A Source Contract

A Low function's `target<representation_contract>` binds its assembly
vocabulary independently of its optional target symbol witness. When the
function is marked `asm`, its body is a durable, readable source form rather
than a disposable view of canonical low IR. Parsing and printing that form
must preserve its target mnemonics, canonical structural operations, compiler
hints, comments, and contract-selected register spellings without replacing
the body with generic low syntax. Downstream authoring, review, diagnostic, and
comparison workflows rely on that stable representation.

A change that makes valid operations unprintable inside target assembly, drops
them while formatting, or weakens a round-trip test by rewriting the input into
generic low syntax is a release-blocking format regression. New generic
metadata and compiler hints interleaved with target packets extend the
trait-based canonical fallback in the low assembly parser and printer; they do
not receive target-opcode special cases. Target emission may intentionally
erase a zero-cost hint only after the readable low assembly form has been
preserved and tested.

## Template Provider Applicability

A template provider's applicability is defined only by its declared family
signature and explicit provider metadata: target binding, `requires` and
`where` constraints, and priority. Selection never inspects a template body to
accept, reject, rank, or deduplicate a provider. Structurally identical
definitions remain distinct providers.

Template bodies may contain operations whose structural placement cannot be
verified until the selected definition is materialized and inlined. That
verification is deliberately deferred to the resulting program. An invalid
selected body fails verification there; its contents never retroactively alter
which template was applicable.

## Reviewable Pipeline Changes

A pull request boundary is an architectural cut, not a file-count or
commit-count cut. Each PR must present a coherent end-to-end behavior that can
be evaluated critically on its own. A producer that writes an internal
contract in one PR while a later PR supplies the only meaningful consumer or
enforcement is not an independently valid change. Green CI on a dormant,
silently discarded, or unreachable contract does not validate the design.

When correctness or architecture can only be judged across a stack, the stack
stays on one unpublished branch until the integrated design has been reviewed.
It may be split afterward only at interfaces whose semantics are complete in
every cut: supported behavior works end to end, unsupported behavior fails
loudly, and no later PR retroactively makes an earlier representation valid.

## Generated Tables

Generated backend tables should be data. The normal shape is a checked-in
header and checked-in C interpreter or lookup helper consuming generated rows.
Large target tables usually fit as `.inl` fragments included by checked-in `.c`
files, or as generated `.c` files whose checked-in header owns all types and
declarations. The table-generator contract excludes bespoke control flow,
callbacks, switch trees, lookup functions, and new semantic APIs. Schema-level
surfaces such as dialect op bindings are the exception: there the generated
artifact is itself the canonical API.

Generators earn their place when they cover data sets large enough or volatile
enough that hand maintenance is worse: vendor ISA records, many descriptor rows,
many source-to-low rule rows, cross-product feature matrices, or tables whose
invalid combinations can be exhaustively rejected during generation. The
payoff is meaningful handwritten duplication deleted, stronger build-time
validation, or hot-path runtime work removed.

A tiny stable table belongs in C. A singleton generated row, an unconditional
extern selected by address, or a few enum constants wrapped in hundreds of lines
of Python and build/test wiring is code bloat unless it lands as part of the
same coherent change that expands to a real family and removes enough C
complexity to make the trade obvious.

## Runtime Cost

Loom's JIT goal makes target lowering hot-path code. Generated metadata should
let C do less work at runtime, not add indirection around constants that were
already known. Descriptor consistency, row count limits, opcode-family
compatibility, feature coverage, and impossible operand forms are generation
time checks. Runtime code evaluates user-dependent facts, resolves descriptor
availability, emits diagnostics, and walks compact trusted data.

## Compiler Trust Boundary

The `loomc` API and command-line tools own external input validation. Parsing,
verification, target selection, and explicit API preconditions establish facts
that later JIT stages consume as trusted compiler state. Successful schedules,
allocations, emission frames, and target packet plans are not untrusted inputs
merely because their declarations live in non-static headers.

Internal transforms and fact queries expose that ownership in their signatures:
an infallible query returns its value, and an infallible transform returns
`void`. An `iree_status_t` below the public boundary identifies a possible
failure owned by that operation, such as arena allocation, output growth,
user-dependent target support, or a representational format limit. Rechecking
producer-owned table bounds, assignment presence, op shape, or cross-stage
identity as recoverable statuses both adds hot-path work and disguises compiler
bugs as runtime conditions.

Debug assertions mark only non-obvious memory or encoding joins where violating
a producer invariant would otherwise cause unsafe access or silent corruption.
They are not the default replacement for deleted validation. Malformed
schedule, allocation, or packet-plan fixtures are not useful consumer coverage;
tests exercise authored input through the real verification and compilation
boundary and check failures at the stage that owns them.

## Cleaning Existing Code

Existing generators are in scope for cleanup when they violate these boundaries.
Finding a generated header that owns backend table structs, generated lookup
code that could be a checked-in generic helper over rows, or a generator whose
only output is a tiny stable table is a design finding, not precedent. Repair
the ownership split before building new work on top of it.

## Review Questions

Before adding or extending a generator, answer:

- What checked-in C contract consumes the generated data?
- Which handwritten duplication or runtime work disappears?
- Which invalid combinations become build-time failures?
- How many rows or families are expected now, and what concrete pressure makes
  that grow?
- Would this be clearer and faster as a normal C constant table?

If those answers are weak, the durable change is probably a small checked-in C
table, a better fact query, a structured diagnostic, or no change at all.
