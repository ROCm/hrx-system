# Loom AMDGPU Python Target Guide

Read `loom/src/loom/target/arch/amdgpu/AGENTS.md` before changing this tree.
That file is the canonical AMDGPU target guide and applies to these Python
descriptor, contract, alias, and target-info sources.

The most important local rule is ownership: Python carries the target truth that
can be validated once when Loom is generated. Descriptor packet consistency,
asm spellings, feature rows, source-to-low contract shapes, alias legality, and
table completeness belong here rather than as runtime C special cases.
