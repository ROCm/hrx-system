# Qwen3 MoE Owned C Runtime

This experiment runs the fixed Qwen3-30B-A3B-Q4_K_M model directly through
IREE HAL and JIT-compiled Loom kernels. It owns the complete model lifecycle:
GGUF parameter indexing and gathering, resident device storage, kernel
specialization, reusable command-buffer recording, request state, queue
submission, correctness witnesses, a generation CLI, and a separate Google
Benchmark runner.

The implementation is intentionally model-specific. The tensor schema, layer
count, layouts, and kernel schedule are compile-time contracts rather than a
generic graph or model format. Raw F32, Q4_K, and Q6_K parameter payloads are
gathered into one deterministic resident slab without repacking. Kernel
identity is independent of weight identity: all 48 layers reuse prepared
executables and select their weights through recorded slab subranges.

`experimental/qwen_moe` remains the source of truth for authored Loom kernels.
Qwen-owned declaration modules add the runtime HAL export contracts, and the
runtime embeds the resulting selectively linked source. `experimental/id4` is
reference evidence only; this package has no source, build, or ABI dependency
on it.

See [KERNEL_INTEGRATION.md](KERNEL_INTEGRATION.md) for the exact authored-source
map, prefill/decode routing, JIT and export boundary, and the compiler facts
exposed by each transient bring-up workaround.

The first completed integration milestone was one complete layer-0 prefill-512
execution:

1. validate and gather the real GGUF parameter set;
2. specialize and load the required Loom exports for the live HAL device;
3. record the attention and routed-expert schedule into one reusable indirect
   command buffer;
4. submit through explicit queue dependencies and compare captured semantic
   outputs; and
5. exercise that same runtime path from a CLI smoke and a filterable
   `Qwen/Layer0/Prefill/512` benchmark row.

The `qwen-cli` target is the text-generation owner. It loads a matching
HuggingFace `tokenizer.json`, applies the ordinary Qwen system/user/assistant
chat form, uploads the encoded prompt once, and streams greedily selected UTF-8
until EOS or `--max_tokens`. Each full-model issue publishes its selected
token into device-local request input before signaling completion. The host
reads a separate observation copy for text and EOS only; decode never waits on
a host token upload.

```sh
build_tools/bin/iree-bazel-run //experimental/qwen/binding/cli:qwen-cli -- \
  --device=amdgpu://0 \
  --parameters=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
  --tokenizer=/path/to/Qwen3-30B-A3B/tokenizer.json \
  --prompt='Reply with one word: hello' \
  --max_tokens=16
```

The bring-up generation lane prepares one decode program for the 64-row shape
class containing the first continuation position. The active position lives
only in device request control and advances at the greedy endpoint, so every
position in that class reuses the same recorded command buffer. A later class
is prepared lazily only when generation crosses a 64-row boundary. The
cooperative decode-attention adapter currently bounds this scheme to 2048
rows. `--max_tokens` bounds generation; it does not change startup JIT or
command-buffer preparation work.

## Temporary bring-up workarounds

`kernels/router_top8_f32_bringup_workaround.loom` makes the owned model's
compact route-ID rows structural while the generic authored kernel omits its
required `route_count <= route_id_stride` relation. The Qwen program fixes both
values to eight. This fork retains that exact behavior, carries one function,
and is deleted when the upstream kernel establishes and validates its generic
stride contract. The same fork temporarily selects the authored wide
four-subgroup geometry because source-to-low compilation cannot see the
workload-evaluated decode-or-prefill launch choice. Prefill measurements remain
representative of that geometry; decode measurements do not until the fork is
deleted.

`kernels/router_projection_top8_fused_bringup_workaround.loom` is the fused
decode router's one-function instance of the same route-stride defect. It
retains the canonical projection, device-scope last-arrival sequence, and
shared top-8 row helper, but passes the configured route count as that helper's
physical stride because this model fixes compact `8 == 8` rows. It intentionally
ignores the public stride argument and is valid only for this owned model.
Delete it with the standalone top-8 fork when the generic helper represents
and verifies `route_count <= route_id_stride`.

`kernels/expert_table_bringup_workaround.loom` makes Qwen's fixed route count
of eight structural in expert-assignment enumeration. AMDGPU lowering requires
an exact divisor, but the compile boundary currently loses the exact workload
value already used during launch evaluation. This one-function fork is deleted
when workload specialization reaches target lowering.

`kernels/attention_metadata_bringup_workaround.loom` is a temporary
one-function device-state producer while the Qwen kernel corpus has no
canonical attention-metadata kernel. It derives positions, separate K/V cache
indices, and dense causal-mask bits from one compact context-base control word.
It is not a kernel framework or a generator and must not accumulate alternate
indexing policies. Delete it when the canonical producer lands.

`kernels/flash_attention_decode_split_bringup_workaround.loom` retains the
canonical split-K decode algorithm bodies. The runtime temporarily binds only
the upper bound of a 64-row shape class as
`qwen.decode.key_value_capacity` because launch topology still requires a
structural bound. Request metadata masks rows beyond the device-owned active
context, and request creation zero-initializes unwritten K/V rows. The adapter
can therefore apply the canonical producer and cooperative reducer to the
whole class without position-specific specialization or unsafe suffix reads.
It contains no attention algorithm body or generated variants and is deleted
when the unmodified public kernel supports a dynamic active context within a
bounded launch capacity.

Pure-tail prompt attention currently uses a second, explicitly temporary
runtime workaround: contexts below the 64-token key tile are recorded as one
canonical general-attention dispatch per query row. There is no alternate Loom
source for this path. The multirow specialization corrupts nonleading query
rows, while the same kernel specialized for one row matches the real layer-0
oracle. On the 14-token chat prompt, row-by-row recording reduced the
post-attention maximum absolute error from 0.183716 to 0.000976562, reduced the
complete-layer maximum absolute error to 0.000900745, and reproduced eight
greedy reference token IDs exactly. This non-sanctioned schedule adds one
dispatch per prompt row per layer; prefill-512 and split-K decode retain their
established schedules. Delete it when the canonical pure-tail multirow path
passes the same differential.

The three direct feed-forward forks
`kernels/routed_gate_up_next_q8_bringup_workaround.loom`,
`kernels/routed_down_q4_next_q8_bringup_workaround.loom`, and
`kernels/routed_down_q6_next_q8_bringup_workaround.loom` retain the canonical
decode ABIs and device bodies while making this model's exact one-token,
eight-route dimensions structural. The gate/up variant publishes its F32
SwiGLU rows and packed Q8_1 x4 groups together; the routed-down variants
publish the following layer's normalized Q8_1 x4 row. They exist only because
workload facts and nested device-template requirements do not yet cross every
source-to-low boundary. Delete all three when those compiler boundaries are
repaired; they are not model-generated kernel variants.

`kernels/vocabulary_q6k_partial_argmax.loom` owns only the model endpoint. It
applies the canonical `ggml.linear_q6k_q8_1_x4.body` contraction and reduces
each workgroup's eight finite logits to one deterministic maximum pair. The
compact argmax kernel finalizes those 18,992 pairs with lowest-token tie
breaking, publishing the winner into both the device-local request token slot
and mapped host-visible observation storage. Decode consumes the device token
through the request timeline; it does not read and reupload the observation
copy.

The working full-model programs now record all 48 layers, embedding, final
normalization, partial vocabulary projection, and compact greedy finalization
into reusable command buffers. Prefill retains the grouped F16 WMMA
feed-forward route;
decode uses fused split-K attention, fused normalization and Q8_1 packing,
direct attention output with fused feed-forward F32/Q8_1 publication, fused
router projection and normalized top-8 selection, direct raw-Q4_K gate/up with
fused SwiGLU Q8_1 publication, and direct Q4_K or Q6_K down contraction with
route weighting, reduction, and residual publication fused.
The layer runner remains a first-class optimization surface so model work can
be isolated to a stable benchmark row without forking execution logic.
