# Loom programming guide

!!! info "Programming guide preview"
    This preview establishes the site structure and generated references. The
    landing-page narrative and executable tutorial sequence are placeholders
    while the checked guide programs are completed.

<div class="loom-hero" markdown>

## One program, specialized all the way down

Loom is a source-first compiler for kernels and asynchronous device programs.
It keeps program structure, target facts, specialization choices, correctness
cases, benchmark workloads, and compile evidence connected from readable IR to
native artifacts.

[Get Loom](getting-started/acquiring-loom.md){ .md-button .md-button--primary }
[Explore the reference](reference/index.md){ .md-button }

</div>

## The authoring loop

Loom programs are ordinary checked source files. The same source can be
formatted, verified, specialized for a concrete target, run against embedded
correctness cases, measured with embedded benchmark workloads, and inspected
through a structured compile report. Kernel authors do not need a separate
test language or a target-specific build graph to complete that loop.

<div class="grid cards" markdown>

-   :octicons-code-24: **Author readable IR**

    Keep algorithms, reusable motifs, launch boundaries, and specialization
    choices explicit in canonical `.loom` source.

-   :octicons-check-circle-24: **Prove behavior beside the program**

    `check.case` and `check.benchmark` describe correctness and performance
    workloads without hiding them in a framework harness.

-   :octicons-workflow-24: **Specialize with target facts intact**

    Link reusable providers, resolve compile-time choices, and lower only after
    the facts needed by the selected target are known.

-   :octicons-graph-24: **Inspect the compiler's answer**

    Structured reports connect native resource use and instruction behavior
    back to the specialization and IR that produced them.

</div>

## A compiler surface, not a kernel island

The smallest useful Loom artifact is a kernel. The same linking,
specialization, diagnostics, and target machinery also represents launch
configuration and command programs containing whole reusable subgraphs. This
lets adoption begin at one performance-critical boundary without creating a
new one-off runtime or giving up the path to larger program ownership.

The guide develops that model from the bottom up: first a program you can run,
then reusable functions and templates, explicit testing and measurement,
target-aware tuning, library composition, command programs, and embedding
through the stable C API.
