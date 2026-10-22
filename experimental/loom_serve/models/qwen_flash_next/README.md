# Qwen3.8 Flash-Next

This package runs all 48 text layers of Qwen3.8 Flash-Next through `loom-serve`.
The model traversal, original-format weight reads, recurrent state, sparse
attention, and sampling are Loom programs. The shared server supplies hybrid
batch scheduling, physical cache pages, retained endpoints, and HTTP ownership.
There is no model-specific native runner or separate prefill/decode execution
path.

## Assets and deployment

Supply the original Unsloth `UD-Q4_K_XL` text checkpoint and its matching
`tokenizer.json`. The `weights` argument is the directory containing these four
files:

```text
Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
```

Preparation explicitly declares all four files in one parameter domain. The
first contains metadata, and the other three contain tensor payloads. All
specialized batch shapes share that domain. Renaming a differently quantized or
modified model does not make its tensor layouts compatible. Vision is separate;
the optional MTP predictor uses the additional asset described below.

The parameter bank occupies approximately **103.68 GiB**, including the
**26.82 GiB** IQ4_NL n-gram table. Original packed weights remain packed; there
is no whole-model dequantized copy. A 192 GB unified-memory machine has room for
the resident model, cache pools, recurrent endpoints, and working storage.
Account for the device's actual memory budget and other applications when
choosing capacities.

From the repository root, create a deployment document that preserves the source
provider list:

```sh
flash_weights=/path/to/UD-Q4_K_XL
flash_tokenizer=/path/to/tokenizer.json
jq --arg directory "$PWD/experimental/loom_serve/models/qwen_flash_next" \
   --arg weights "$flash_weights" --arg tokenizer "$flash_tokenizer" \
   '.base_directory = $directory | .definition += {
      weights: $weights, tokenizer: $tokenizer
    }' experimental/loom_serve/models/qwen_flash_next/model.json \
   > /tmp/flash-next-deployment.json
```

After building the [supported server](../../README.md#build-and-load):

```sh
bazel-bin/loom/src/loom/tools/loom-serve/loom-serve \
  --device=rdna --listen=127.0.0.1:8080 \
  --model=/tmp/flash-next-deployment.json

# From another terminal:
curl --fail --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Loom-Session: example' \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"What is 2+2?"}],"enable_thinking":false,"temperature":0,"max_tokens":64}'
```

Registration compiles the source programs; parameter activation loads the
checkpoint on first use. `/v1/status` distinguishes those operations. Reusing
`X-Loom-Session` with the complete matching conversation retains model state.
The shared [Qwen chat policy](../qwen/README.md#device-sampling) supplies thinking,
tool formatting, and device sampling, including `definition.options.sampling`.

Copy this directory together with `../qwen/` and `../../motifs/`,
preserving their relative layout. Model changes do not require rebuilding the
server.

## Multi-token prediction

The optional predictor uses the original `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`
sidecar. Enable three recursive proposals by extending the deployment document:

```sh
flash_draft=/path/to/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
jq --arg draft "$flash_draft" \
   '.definition.draft_depth = 3 |
    .definition.options.draft_weights = $draft' \
   /tmp/flash-next-deployment.json > /tmp/flash-next-mtp-deployment.json
```

Use that document with `--model`. The sidecar's embedding and vocabulary tensors
are identical to the target's tensors and share their device allocations. Only
2,775,621,632 additional parameter bytes are needed, bringing the logical
parameter bank to approximately **106.26 GiB**. Physical backing granularity
and mutable state add capacity independently. The predictor reads its Q8_0
expert and injection weights directly.

The predictor proposes greedily, and the target verifies under the request's
sampling policy. Accepted prefixes publish target state and refresh the
predictor with accepted target hidden rows. Rejected proposals remain outside
the committed cache frontier. Verification retains compact transition operands
and replays the accepted prefix through the existing recurrent kernels; it does
not copy a complete recurrent endpoint for each candidate. Predictor carry adds
21,248 bytes per retained owner, and predictor cache adds 2,112 bytes per pooled
token, plus a 768-byte working tail per execution row.

The deployment defaults to `continuation_epochs: 1`. Set it to `2` to pre-issue
a second packed epoch without an intermediate host round trip. Device
acceptance advances positions, filters EOS and exhausted output credit, and
compacts surviving spans before the second traversal. A final single output
uses ordinary decode, so continuation fits the same request reservation as
host-scheduled work. MTP shares the hybrid scheduler's mixed prompt and
generated spans, checkpoint forks, EOS, and cancellation.

The default remains `draft_depth: 0`: the gain depends on accepted proposals
repaying proposal, verification, and catch-up work. Compare the same prompts,
sampling policy, output budget, and retained state with and without MTP. Server
epoch records expose proposed and accepted draft-input counts independently of
selected output tokens.

## Packed execution and retained state

Every epoch carries one immutable table of active token spans. Prompt chunks,
single-token continuations, and independent sessions use the same program.
`prefill_capacity` bounds an epoch, not a prompt: longer inputs are chunked.
The package requires pooled cache storage, a context capacity divisible by 64,
at most 32,768 tokens per epoch, and draft depth zero or three.

The default document provides four execution rows, a 16K per-session context
ceiling, a shared 32K-token page pool, a 16K-token epoch ceiling, and four
checkpoint slots. KV storage is 25,344 bytes per pool token, including compressed sparse
attention indexes. A recurrent endpoint is 115,651,840 bytes: 36 GDN states,
PLE token and convolution history, and incomplete sparse-index blocks. Endpoint
capacity therefore matters independently of KV capacity.

For 128K contexts on a 192 GB machine, this deployment provides eight execution
rows, eight checkpoint slots, and a shared 256K-token page pool while retaining
16K-token epochs:

```sh
jq '.definition += {
      rows: 8, context_capacity: 131072, pool_capacity: 262144,
      prefill_capacity: 16384, checkpoint_capacity: 8
    }' /tmp/flash-next-deployment.json \
  > /tmp/flash-next-128k-deployment.json
```

Use the resulting document with `--model`. Apply the same override to the MTP
deployment to combine long context with prediction. The context ceiling includes
both input and generated tokens. The page pool is shared across live sessions
and retained checkpoints; eight rows do not reserve eight independent 128K
histories. Shared prefixes retain their pages, while new continuation tokens
consume additional pool slots. Recurrent endpoints have their own storage and
lifetime, independently of those shared pages.

The scheduler chooses among prepared token/span capacities for each epoch;
raising `prefill_capacity` does not force every request to use that largest
shape. Larger prompt batches can share more expert weights and amortize
dispatch cost, but also keep the device occupied longer before newly ready
requests receive their next turn. Measure first-text latency alongside total
cohort throughput when selecting the capacity for interactive workloads.
The default 16K epoch ceiling allows large prompts to reuse weights across
many tokens. A lower ceiling trades that reuse for more frequent scheduler
turns when sharing the device with latency-sensitive requests.

MTP has a separate tradeoff: accepted proposals save target traversals, while
drafting and verification add work. A configuration that improves one client's
generation rate can reduce aggregate throughput when many clients already
fill the target batch. Compare ordinary and speculative execution on the same
arrival pattern, prompt lengths, and output budgets. Complete-cohort throughput
includes prompt processing and queueing; it differs from individual generation
rates measured after first text or after preparing every prompt.

Pinned readers and private writers have distinct recurrent origins. At a fork,
the first token reads its immutable anchor; subsequent tokens read the writer.
Only that writer is updated. Main K/V and index keys use the scheduler's physical
page IDs. Partial four-token index blocks remain with the recurrent endpoint,
so a fork at an arbitrary token boundary preserves both histories. Fresh spans
synthesize empty state instead of depending on cleared recurrent storage.

## Source map

| Source | Responsibility |
|---|---|
| `prepare.loom`, `weights.loom` | Stage specialization, explicit checkpoint files, and storage geometry. |
| `../qwen/control.loom` | Chat policy, sampling policy, epoch submission, and feedback publication. |
| `epoch.loom` | Four residual streams through 48 layers, including PLE at layer 1 and QSA every fourth layer. |
| `propose.loom`, `verify.loom`, `programs/mtp.loom` | Recursive proposals, target verification, and accepted-prefix predictor catch-up. |
| `programs/` | Hyperconnection, recurrent attention, sparse attention, routed/shared experts, and output-head composition. |
| `kernels/gdn.loom`, `kernels/gdn_window.loom`, `kernels/work.loom` | Recurrent updates and immutable span ownership. |
| `kernels/gdn_fusion.loom` | QKV contraction, tile-local convolution, compact boundaries, and speculative capture. |
| `kernels/speculation.loom`, `kernels/mtp_state.loom` | Compact target transition capture, accepted-prefix replay, and predictor carry ownership. |
| `kernels/qsa_*.loom`, `kernels/pages.loom` | Paged K/V, four-token compression, exact top-512 block selection, and causal attention. |
| `kernels/ple_*.loom`, `kernels/iq4.loom` | EOS-aware bigram/trigram hashes, IQ4_NL reads, and dilated convolution history. |
| `kernels/quantized.loom`, `kernels/q8.loom`, `kernels/experts.loom` | Q4_K, Q5_K, Q5_1, and Q8_0 expert projections in the checkpoint's actual encodings. |
| `kernels/q8_matrix.loom`, `kernels/hyperconnection_matrix.loom` | Shared original-Q8 matrix contraction and fused hyperconnection publication. |
| `kernels/expert_buckets.loom` | Device-owned expert grouping and finite workers that reuse weights across routed rows. |
| `kernels/expert_matrix.loom`, `kernels/expert_block.loom` | Matrix schedules and register-resident packed blocks for routed gate/up and down projections. |

The text model uses 36 GDN layers and 12 sparse-attention layers, with 512
experts and ten selected experts plus a shared expert in each layer. Dense
projections and embeddings use Q8_0; index projections use BF16. Accumulation is
F32, with explicit BF16 activation publication boundaries. The output head
projects only rows that require token selection.

Recurrent spans keep the full F32 state in registers until publishing their
private writer. At epoch capacities of 32 or more, two workgroups per head
share four-token Q/K/V windows through 10,304 bytes of double-buffered LDS.
Four lanes cooperate on a value row, and each lane retains two rows. Three
raw-input windows remain in flight so global loads overlap the state updates;
normalization and gates run at LDS publication, after those loads have matured.
The LDS key layout interleaves four contiguous key segments across banks.

Small epochs and speculative replay use the wave recurrence. Both bodies use
the same accurate near-one decay: small exponential errors would otherwise
compound across long-lived state and make transitions between bodies drift.
The physical state layout, immutable anchor, private writer, and publication
permission are identical. The window path adds no global workspace or dispatch.

Sparse index scoring shares four queries and 64 pooled keys per workgroup.
The four heads of each query form the rows of a BF16 matrix product, followed
by F32 ReLU and head summation. A span dimension establishes common resident
cache ownership before sharing keys. Neighboring spans that cross a query tile
publish disjoint score rows; each query retains its own causal extent. The
16-by-136 query panel and 64-by-136 key panel occupy 21,760 bytes of LDS.
Capacities below 32 retain the one-wave dot-product scorer.

By default, scoring and exact top-512 selection process 512 queries at a time
while the complete epoch can contain 16K or more tokens. The score workspace consumes
`min(epoch_capacity, qsa.query_tile) * context_capacity` bytes: 16 MiB at 32K
context or 64 MiB at 128K with the default 512-query cohort. Selection finishes
before that workspace is reused. Dense queries do not read scores, and masked
cells need no clearing. Larger cohorts expose parallel selection work without
allocating a whole-epoch score matrix. The selector preserves ordered logical
IDs and the earliest logical block on exact score ties.

Sparse attention keeps each query's exact top-512 four-token blocks and its
incomplete causal tail. At capacities of at least 32 tokens, one workgroup
shares a selected K/V tile across the twelve query heads that consume it.
Four waves retain their query panels in registers, merge Q*K partials, and
compute P*V through BF16/F32 matrix instructions. Twelve heads occupy a padded
16-row matrix; the remaining rows have no external reads or writes. The two
K/V panels and four partial-score planes occupy 23,040 bytes of shared memory,
with no dense mask, attention matrix, or extra global workspace.

Softmax and accumulation remain F32. Probabilities use a BF16 high component
and a BF16 rounding residual for the matrix contractions; using only the high
component loses accuracy in cancellation-sensitive outputs. Attended values
still round to BF16 before sigmoid gating and final BF16 publication. Smaller
cohorts use sixteen waves to partition one head's keys, providing parallelism
when too few independent queries are available to occupy the device.

The 36 GDN QKV tensors use an in-place preparation pass. Each workgroup owns
four complete output channels, captures their original bytes into 10,880 bytes
of shared memory, and separates their scales and codes into contiguous planes.
The allocation keeps exactly the original size. The loader publishes readiness
after preparation and repeats it after reloading the original checkpoint; all
QKV consumers, including speculative verification, use the prepared layout.

At epoch capacities of at least 128, one workgroup contracts 128 tokens against
one 128-channel Q/K/V head. Exact integer codes and existing BF16 activations
feed matrix instructions; the original scales apply in F32 after each K32
partial product. The shared Q8 contraction specializes its addressing for the
prepared scale/code planes and keeps two stages in flight. Its F32 result rounds
to BF16 in a padded 128-by-132 LDS tile, where interior tokens immediately pass
through four-tap convolution, SiLU, and Q/K normalization. The complete raw QKV
array is never written to global memory.

Each intersection of a session and a token tile exports only its first and last
three raw rows. A separate head/session worker resolves the prefixes against
retained history and carries each suffix into the next tile. It consumes all
boundaries before updating the private writer, including when that writer is
also the history source. Compact storage needs at most
`ceil(token_capacity / 128) + span_capacity - 1` six-row slots; unused slots
require no clearing. Speculative spans publish their raw rows directly into
their verification capture slots and leave retained history unchanged.

Smaller epochs retain 64-channel by 32-token matrix tiles or packed dot
instructions. Sparse accepted-prefix replay retains its wave implementation.
Both paths share recurrence and gating after the same BF16 convolution
publication. Prompt and generated tokens still use one epoch contract.

Other dense Q8 projections retain the original checkpoint layout. One matrix
template accepts the tile dimensions, wave geometry, and pipeline depth as SSA
values. Shape selection keeps dot products for tiny epochs, uses smaller tiles
when a narrow output needs more independent workgroups, and increases reuse to
128-by-128 output tiles for wide prefill projections. The wide schedule uses
two stages of read-ahead; smaller schedules use one. All instantiate the same
contraction through 128-thread and 256-thread launch entries. Selection
specializes during command compilation; no CPU decision
or parameter repacking is added to an epoch. The immutable epoch header bounds
active rows, so partial tiles preserve inactive storage and cannot mix padding
into valid results.

Hyperconnections use the same contraction at capacities of at least 128 tokens.
Each 128-by-128 tile reuses original Q8 coefficients across its token panel.
Down projection applies quarter-scaled SiLU to the F32 fragments before storing
BF16. Up projection stages all four streams of a feature panel into the same
wave, applies sigmoid, and combines the four normalized values in F32 before
the final BF16 store. No global gate tensor or decoded weight bank is allocated.
The separate four-row injection retains its original F32 or Q8 arithmetic;
smaller epochs and selected output-head rows retain packed dot products.

MTP catch-up also uses the shared Q8 contraction at capacities of at least 128.
The original projection rows contain adjacent embedding and hidden halves;
explicit row stride and byte origin select each half without copying weights.
The embedding projection is computed once per token and shared by its four
hidden streams. Each projection rounds independently to BF16 before their final
addition. Small speculative cohorts retain the original wave reduction, which
avoids a temporary embedding result and underfilled matrix dispatches.

Catch-up normalization reads accepted target rows through the existing token
map and a resident-row origins table. First-row inputs come from committed
carry restored into the residual buffer; normalization finishes before the
projection overwrites that buffer. Later inputs stay in the immutable target
plane, eliminating the full hidden-state copy. Recursive proposals retain
their private hidden inputs and share the same normalization and projection.

Expert projections group routes by their selected expert. One GPU workgroup
builds a histogram, scans record offsets, and scatters original
`token * 10 + slot` identities into records. Each record carries its expert,
live row count, and original identities. Capacity specialization chooses the
record width and contraction schedule:

| Prepared token capacity | Expert schedule |
| --- | --- |
| 1–4 | Direct per-route projections. |
| 5–255 | Four-route records and packed dot products. |
| 256–2,047 | 64-route records and matrix contractions. |
| 2,048–32,768 | 128-route records and matrix contractions. |

A finite pool of workgroups visits live records and output features. Gate/up
and down reuse the same table, and the final combination accumulates the
original ten slots in order. Wider records amortize packed-weight reads and
decoding; compact records limit padded matrix work when expert populations
are smaller. These choices specialize into the command and kernel programs,
without a host routing decision or a distinct prefill execution path.

The 128-row gate/up schedule assigns eight wave32 subgroups to 64 channels
per branch. Each lane retains packed weight codes and their original headers
in registers: one K256 block for Q4_K/Q5_K, or four K32 blocks for Q8_0.
Gate and up share a padded K64 activation window in LDS. Masked loads fetch
the following window while the current one computes, suppressing inactive
rows and the final refill. Weight blocks remain resident across those windows,
so each K32 contraction reuses their headers and code packets.

Integer weight codes are represented exactly in BF16 and consumed by matrix
instructions. Original scales and affine corrections are applied in F32 after
each K32 partial product; fused gate/up retains F32 values through SiLU and
the product before publishing BF16. The smaller routed, shared, and down
matrix schedules stage both codes and activations in padded LDS panels, with
tile and wave geometry passed as SSA values. All schedules consume original
Q4_K, Q5_K, Q5_1, or Q8_0 storage without an expanded parameter bank.

The producer's dispatch dependency publishes the complete table to consumers.
The count word is an extent, and the command owns record storage until both
consumers finish. There is no CPU route readback. For capacity `C` and record
width `R`, the allocation bounds records by
`min(10 * C, floor(10 * C / R) + 512)`: at most one record per route, or full
records plus one partial record per expert. Rank scratch uses `40 * C` bytes;
the record table uses four header bytes plus `4 * (R + 2)` bytes per allocated
record. Together they occupy 8,964 bytes at capacity 32 and
1,587,204 bytes at the default maximum capacity of 16,384. Only live records
are consumed, and only their unused tail fields require initialization, so
storage reuse needs no global clear.

The shared expert uses the same contraction with contiguous token rows and
original Q8_0 gate/up weights. At capacities of 64 or more, 64-token by
64-feature tiles reuse each activation panel across both projections and keep
the nonlinear result in F32 until BF16 publication. The immutable epoch header
supplies the live row count directly; shared work needs no routing table.
Smaller capacities retain packed dot products, including the default predictor
proposal and verification shapes. Shared down uses the dense Q8 matrix policy.

Both dot and matrix consumers preserve the original packed weights, BF16
activations, and publication boundaries. Their accumulation order differs;
independent F64 references check the original-value numerical contract instead
of requiring identical rounding between schedules.

## Exact sparse attention selection

Each query scores completed four-token blocks, then selects at most 512 logical
block IDs. Selection preserves the original score ordering, resolves equal
scores by earliest logical position, and publishes IDs in ascending order.
The attention consumer resolves those IDs through the retained page map.

For longer histories, a 4,096-bin histogram narrows the threshold. A sample
places the first window, but every causal score is counted; a clipped winning
bin triggers a complete-range histogram. Up to 4,096 candidate keys stay in
LDS while eight-bit refinements find the exact threshold. Larger candidate
sets stream the original row. Scratch remains bounded independently of context
length, and selection introduces no score approximation.

Each lane then consumes 32 consecutive scores and forms selection masks.
Workgroup scans assign disjoint output intervals, including the earliest
threshold ties. IDs go directly to the existing output buffer, without an
intermediate global mask, CPU readback, or separate compaction dispatch.
Narrow cohorts use 1,024-thread workgroups; wider cohorts use 256 threads.
Histories with no more than 512 completed blocks keep the direct-ID path.

## Qualification

The checkpoint-free kernel suite uses independent scalar arithmetic and
ownership checks, including partitioned recurrent execution, shared-anchor
forks, incomplete index blocks, EOS history, inactive storage guards, IQ4_NL
decoding, stable top-512 selection, and exact route ownership through empty and
ragged expert buckets. An independent F64 oracle covers every expert encoding,
routed row count, and empty or ragged contiguous shared-expert tile:

```sh
iree-bazel-test --config=asan //experimental/loom_serve/models/qwen_flash_next/...
```

The real-checkpoint service check exercises useful text, concurrent branches,
retained continuation, fresh replay, and weight reactivation through HTTP:

```sh
PYTHONPATH=loom/src/loom/serve/testing python3 -B \
  -m experimental.loom_serve.models.qwen_flash_next.check_service \
  --url=http://127.0.0.1:8080 --model=flash-next
```

Run it against a dedicated model with at least two rows, a 256-token context,
and one checkpoint slot. It temporarily drains and reactivates that model. This
is a correctness check, not a performance benchmark or broad language-quality
evaluation. Numerical differences near an expert-selection cutoff can change
the routed expert set; full-model evaluation must measure output distributions
and task quality in addition to individual-kernel error.
