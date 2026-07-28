# Qwen MoE Agent Guide

Read `experimental/qwen_moe/README.md` before editing this experiment.

Each production `.loom` file is the complete authoring source. It retains the
mathematical and physical-layout comments, nonzero `check.case` coverage,
production `check.benchmark` declarations, launch configuration, and target
specialization needed to modify and port the kernel. Generated stripped source
may be embedded or packaged, but it is never the checked-in source of truth.

Kernels consume the packed Q4_K and Q6_K layouts exposed by GGUF unless a
separately measured repacking experiment justifies a backend-owned replacement
allocation. Checked-in tests use bounded synthetic packed data or tiny
representative extracts; model files and large tensor captures remain external.

Fusion boundaries follow model dataflow, not llama.cpp's current dispatch list.
Every fusion must retain an executable numerical comparison at its semantic
output and benchmark equivalent work against the pinned reference backend.

Logical operation contracts are shared across targets. Gfx1100 and gfx1151 may
select different templates, tile sizes, workgroup layouts, or instruction
families without changing model-facing kernel names or C/C++ routing logic.
