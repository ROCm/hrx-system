# Loom Documentation Contract

This guide applies to the public website source under `loom/docs/`. The parent
`loom/AGENTS.md` continues to own compiler and IR design; this file owns how
those contracts become user-facing documentation.

## Audience And Evidence

Public pages begin with the capability a Loom user is trying to exercise and
the observable contract that makes the result trustworthy. A reader with
installed Loom binaries can identify the right tool, form a valid command,
interpret its output, and find the next deeper reference without knowing the
source tree or compiler pass structure.

Stable user evidence includes source and artifact identities, selected targets,
structured reports, correctness results, and explicitly defined measurement
boundaries. Compiler implementation details appear only when they are part of a
public diagnostic or are required to explain observable behavior. Session
transcripts, change history, temporary research state, machine-specific setup,
and source-build commands are not user documentation.

## Information Shape

Workflow pages own task-oriented journeys. Guide pages own concepts that span
tools. Reference pages own exact syntax and semantics. A focused page links to
those deeper owners instead of copying their full contents, and related pages
use the same term for the same public concept.

The ordinary successful path appears first. Specialized control, raw report
fields, compiler traces, and target-specific archaeology follow through
progressive disclosure at the point where a concrete question requires them.
Research techniques remain clearly separated from the acceptance gates for
maintained programs.

## Commands And Examples

Commands are valid for installed binaries on `PATH` and use public flags.
Repository wrappers, Bazel labels, checkout layout, and local toolchain paths
belong in developer documentation. Examples use realistic symbols and files,
state any required preconditions, and remain copyable as shown.

Canonical IR examples teach one reason for a construct at a time. Orthogonal
contracts such as authored Low, fixed physical allocation, and locked source
order receive separate examples unless their interaction is the subject being
explained. Generated dialect reference examples inherit this requirement from
their Python declarations.

## Help And Agent Guidance

Public `--help`, `--agents_md`, and website documentation describe one coherent
tool contract. Agent guidance is a compact decision aid: the common commands,
the evidence they produce, a small set of useful structured-output queries, and
links to focused website pages. Long field catalogs and specialized `jq`
programs live with the report schema or workflow that explains when they are
useful.

Flag interactions are described at their real abstraction boundary. For
example, `--pipeline=none` disables all compiler transformations; individual
pipeline-fed features are not presented as isolated exceptions to that rule.

## Review Boundary

A documentation change is complete when its commands match current tool help,
its links resolve under the MkDocs navigation, its claims name the evidence that
supports them, and its examples do not promote an investigation artifact into a
maintained source form. Generated reference changes begin in the canonical
dialect declarations and are regenerated rather than edited in place.
