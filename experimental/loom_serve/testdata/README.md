# Frozen retained source-review workload

`source_review.json` supplies eight concurrent two-turn reviews to
`benchmark_service.py --workload`. Each response has a 192-token ceiling. The
second request includes the actual first response, so this is closed-loop HTTP
replay with retained state, not an identical token trajectory across models.
There is no agent process, tool execution, or hidden reasoning in the timed loop.

The embedded file-read results are public repository source, byte-matched
against commit `3ffe90573b` under `loom/src/loom/verify/test/`:
`function_entry`, `operand_dictionary`, `unreachable_definitions`,
`attribute_visibility`, `type_visibility`, `func_like_exits`, `isolation`, and
`dominance` (all `.loom-test`). Each session reviews one file, then revisits its
proposal using another. Prompts require evidence and distinguish guesses; model
answers are not used as correctness oracles for the reviewed code.

These copies remain frozen when compiler tests evolve. The workload identity is
its exact bytes, SHA256
`2694163152fcaecab7389e5060fd702becf67da3e01e4cd36b68c0704eb3bdaa`.
Changing the corpus creates a different experiment rather than silently updating
a prior score. The source excerpts carry the repository's Apache-2.0 WITH
LLVM-exception license; no private session transcripts or machine paths are
included.

The [performance guide](../docs/PERFORMANCE.md) supplies the source-JIT server
configuration, measurement procedure, and interpretation limits.
