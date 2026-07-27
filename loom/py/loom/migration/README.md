# Loom Source Migration

This package owns Loom source migration: moving checked and user-authored
`.loom` and `.loom-test` text from older spellings to the current source
format. It is intentionally separate from normal op authoring and from bytecode
compatibility.

The migration layer is for source-format continuity. A future reader changing
an op format should be able to answer four questions here:

- What kind of compatibility change is this?
- Does the op declaration need a `LegacyFormat` entry?
- Can the rewrite be generated structurally, or does it need a hook?
- Which tests prove the rule is safe to keep until its compatibility window
  expires?

## Compatibility Surfaces

Loom has several compatibility surfaces with different costs and promises.

| Surface | Migration shape |
| --- | --- |
| `.loom` source | Precision source edits driven by op `legacy_formats`. |
| `.loom-test` source | Migration applies only to input IR sections; expected blocks are updated by `loom-check --update`. |
| `.loombc` bytecode | The current tool checks the bytecode version and reports stale files; compatibility readers are a separate design. |
| In-memory IR shape | Requires parser/builder support or a semantic rewrite hook when old source cannot construct the current op directly. |
| Comments and formatting | Precision edits preserve surrounding source; comments inside a rewrite span currently produce diagnostics instead of guessed edits. |

New ops, new attributes accepted through an existing `AttrDict()`, new docs,
new examples, and implementation-only changes usually have no source migration
surface. Changing the canonical text spelling of an existing stable op does.

Experimental dialects can stay fast-moving. Stable dialects and source used by
checked corpora need migration rules when their canonical text spelling
changes, because those rules are what let agents and users update files without
reverse-engineering the old grammar.

## Baselines And Windows

Text migration coordinates are named source-format baselines. The current
development baseline is declared in [manifest.py](manifest.py):

```python
CURRENT_TEXT_BASELINE = "loom-source-format-2026-06-09"
```

Rules extracted from op declarations carry:

- `rule_id`: stable identifier for diagnostics and tests, usually
  `<op-name>.<reason>`.
- `introduced`: baseline where the legacy spelling was valid. Defaults to
  `pre-release`.
- `replaced_by`: baseline where the current spelling replaced the legacy one.
- `expires_after`: optional baseline after which the rule can be removed. Empty
  means the removal window is not decided yet.

`replaced_by` and `expires_after` name different facts. `replaced_by` is when
the new spelling became current. `expires_after` is a retention policy decision
for deleting the compatibility rule. Pre-release rules often leave
`expires_after` empty until we have real releases or a deliberate cleanup window.

## Structural Rules

Most text spelling changes should be structural. The op declaration states the
legacy format and the current format, and migration code derives the rewrite.

The first real rule is `buffer.assume.memory_space.attr_dict` in
[loom/py/loom/dialect/buffer/defs.py](/loom/py/loom/dialect/buffer/defs.py).
It migrates:

```loom
%global = buffer.assume.memory_space %buffer {memory_space = global} : buffer
```

to:

```loom
%global = buffer.assume.memory_space<global> %buffer : buffer
```

The declaration is local to the op:

```python
legacy_formats=[
    LegacyFormat(
        "buffer.assume.memory_space.attr_dict",
        format=[
            Ref("buffer"),
            LBRACE,
            kw("memory_space"),
            EQUALS,
            Attr("memory_space"),
            RBRACE,
            COLON,
            TypeOf("result"),
        ],
        replaced_by="loom-source-format-2026-06-09",
    )
]
```

The structural migrator matches the legacy format, captures fields by name,
maps them through any `LegacyFieldMapping` entries, applies
`LegacyFieldDefault` values, and renders the current op format. This keeps the
easy case declarative and prevents a small syntax change from becoming a custom
rewrite function.

Structural support is intentionally finite. Today it covers simple fixed-token
formats using elements such as `Keyword`, `Ref`, `Attr`, `TemplateParam`, and
`TypeOf`. Unsupported elements fail loudly when default migration rules are
built unless the rule names a rewrite hook.

## Rewrite Hooks

Hooks are the escape hatch for migrations that cannot be inferred from two
format lists. Typical hook-shaped changes include:

- The old source needs semantic interpretation before constructing current IR.
- A required current field has no local default and must be derived from nearby
  IR.
- The old spelling used a format element the structural migrator does not
  support yet.
- The migration needs custom diagnostics or fixup hints for partially
  recognized legacy input.

The op declaration names the hook; the migration driver or test registers the
implementation:

```python
LegacyFormat(
    "example.semantic.old",
    format=[AttrDict("attrs")],
    replaced_by="loom-source-format-2026-06-09",
    rewrite_hook="example_semantic_rewrite",
)
```

```python
rules = migration_rules_from_ops(
    (example_op,),
    rewrite_hooks={"example_semantic_rewrite": example_semantic_rewrite},
)
```

A hook is still a migration rule. It should return precision edits and
structured diagnostics through `MigrationRuleApplication`, not edit files
directly and not update `.loom-test` expected blocks itself.

## `.loom-test` Handling

`.loom-test` files contain input IR and expected output/diagnostic blocks. The
migration driver rewrites only the input IR sections discovered by the
`loom-check` case parser.

Expected blocks are owned by the check/update workflow. After a source
migration changes input IR, run the relevant `loom-check` tests with `--update`
when expected output intentionally changes. The tool workflow is documented in
[loom/src/loom/tools/loom-check/README.md](/loom/src/loom/tools/loom-check/README.md).

This separation keeps migration from accidentally blessing output drift. The
migration tool changes source; `loom-check` regenerates expectations.

## Rule Tests

An active rule should have focused tests that document the old spelling, the
current spelling, and the failure shape. The buffer rule tests in
[rules_test.py](rules_test.py) are the first worked example.

Useful coverage:

- Legacy input rewrites to exact current source.
- Current syntax is a no-op.
- Running migration twice is idempotent.
- Rewritten source parses with the current strict parser.
- Malformed legacy syntax reports the rule id and a source range.
- `.loom-test` input sections are migrated without touching expected blocks.
- Hook-backed rules prove both registered-hook dispatch and missing-hook
  diagnostics.

Manifest tests in [manifest_test.py](manifest_test.py) cover baseline ordering,
unknown baseline diagnostics, duplicate rule ids, required active-rule test
metadata, and expiration warnings.

## Tool Workflow

The CLI entry point is `loom-migrate`:

```bash
loom-migrate path/to/file.loom
loom-migrate path/to/file.loom --output /tmp/current.loom
loom-migrate --root loom --check
loom-migrate --root loom --in-place
```

`--check` reports files that require migration and is the shape intended for
presubmit integration. `--in-place` updates source files. `--json` emits
structured per-file diagnostics for agents and automation.

These examples assume `loom-migrate` is available on `PATH`. It may come from
an installed Loom release, a CMake build, a Bazel build, or another package
layout.

## In-Tree Developer Checks

When editing the migration package inside the Loom source tree, one focused
Bazel validation shape is:

```bash
python dev.py bazel test //loom/py/loom/migration:migration_test
python dev.py bazel run //loom/py/loom/migration:loom-migrate -- --root loom --check
```

When op declarations or generated C tables changed as part of a source-format
migration, also run the owning dialect/generator tests.

## Pointers

- [loom/py/loom/dialect/README.md](/loom/py/loom/dialect/README.md): normal op
  and dialect authoring.
- [manifest.py](manifest.py): release table, current source-format baseline,
  bytecode version, and rule metadata linting.
- [rules.py](rules.py): structural rule generation and hook registration.
- [driver.py](driver.py): file discovery, source migration, `.loom-test`
  container handling, bytecode version checks, and parser validation.
- [cli.py](cli.py): command-line wrapper for `loom-migrate`.
- [rules_test.py](rules_test.py): worked examples for structural and hook-backed
  rules.
