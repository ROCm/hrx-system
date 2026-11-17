# Target contributions and shared legalization

Adding an optional compiler target changes the vocabulary and native lowering
available for that target. It leaves compilation for other targets unchanged.
A Wasm-only embedding and a compiler containing Wasm, AMDGPU, and XDNA use the
same portable arithmetic algorithms for a Wasm function.

## Ownership

`loom_target_provider_t` supplies a target's descriptor sets, source-to-Low and
math policies, scoped legalization and diagnostic callbacks, and emission
mechanics. The common compiler owns registry composition and dispatch.

Portable rewrites live under `transforms/` with the dialect or representation
contract they implement. `codegen/low/pipeline/legalizer_registry.c` registers
their providers once, independently of the optional targets. For example,
wide integer division expressed with scalar, index, and SCF operations belongs
there even when the first production use is on a GPU without native division.

Target callbacks own decisions that actually depend on the target: selecting
a physical packet width, preserving an available native instruction, choosing
an ABI carrier, or applying an ISA-specific constraint. They can call shared
rewrite utilities without owning those utilities' portable fallback
registration. A callback whose only action is to invoke a portable recipe is
evidence that the shared registration is missing a case.

The existing families illustrate the boundary:

| Decision | Owner |
| --- | --- |
| Decode FP8 into BF16 using integer operations | Shared scalar reference provider |
| Stage F16/FP8-to-I64 through exact F32 and I32/U32 | Shared conversion reference provider |
| Compose subgroup matching from broadcasts, comparisons, votes, and masks | Shared kernel reference provider |
| Select a physical vector packet or an available native collective | Target capability and representation policy |
| Admit an exact floating recipe under a target's floating-control guarantees | Target math/legality policy |

The subgroup recipes consume the existing indexed subgroup-size snapshot;
they neither query a target by name nor discover target facts by walking IR.
The common provider owns their registration and payload eligibility. Leaf
broadcast, vote, and mask contracts retain their capability diagnostics.

## Scope and selection policy

Scope describes applicability. A provider names a target family, an exact Low
descriptor set, or `LOOM_LOW_DESCRIPTOR_SCOPE_SHARED`. Common dispatch tests
that scope before invoking the callback. Target contributions name descriptor
families or sets supplied by their owning package; a shared provider belongs
to common registry composition. An omitted scope is invalid.

Strategy describes why a rewrite is selected. `TARGET` realizes an operation
using the selected target's native capabilities. `REFERENCE` supplies a
portable semantic fallback. Shared ownership does not imply reference strategy:
shared vector packet construction is driven by the selected target's packet
policy and is a native realization.

The default policy prefers an accepted native contract, then target rewrites,
then reference fallback. `require-native` rejects reference expansion;
`reference-only` skips target rewrites and selects applicable reference
recipes. A registration's directory or name does not determine this policy.

Rule flags refine that ordering:

- `REWRITE_LEGAL` permits an intentional rewrite of an already accepted op.
- `REQUIRE_CONTRACT_REJECTION` waits for an explicit target rejection. An
  unanswered contract can still reach a native lowering without decomposition.
- `REQUIRE_CONTRACT_UNHANDLED` provides a fallback only for an unanswered
  contract, preserving a target's explicit semantic rejection.

`reference-only` deliberately overrides the latter two guards. Type masks and
match predicates select the supported semantic domain before the rewrite runs.
These distinctions preserve native code quality and precise errors when a
portable recipe is introduced or moved.

## Registration and qualification

A portable family extends its owning shared provider and the corresponding
common transform tests. A genuinely new shared provider is added to the common
registry and its BUILD dependencies. Target registration arrays contain only
scoped target decisions. Source-to-Low and math policy keys name descriptor
sets supplied by the same target and have one owner in a composed environment.

`configured/compiler_provider_set_test.cc` qualifies these ownership rules
over the actual enabled providers: explicit scopes, owned descriptor identities,
unique contributions, and unique policy keys. These are immutable compiler
construction invariants, checked in tests rather than repeatedly on the JIT
path. The common legalizer registry test covers shared availability without
optional providers and ordering ahead of competing decompositions.

The production witness determines the remaining qualification. An isolated
target environment and the fully configured environment compile the same
source to the same result. A second applicable target establishes that a
portable algorithm is actually reusable. Existing common corpus execution
checks semantics; exact, compact transform or instruction fixtures establish
the mechanism and native-selection controls. Native-preferred, reference-only,
and require-native behavior each remain part of the contract.

Cost accounting includes minimal builds as well as fully configured builds.
Moving an algorithm can remove duplicate registry entries while adding code
to an embedding that previously omitted it. Registry setup, rewrite expansion,
scratch use, and final program quality are separate costs; scope qualification
does not add work to operation dispatch beyond its existing identity test.
