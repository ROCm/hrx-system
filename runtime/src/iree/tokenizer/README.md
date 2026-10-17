# IREE Tokenizer

A high-performance streaming tokenizer library for ML inference and training,
written in C. Loads HuggingFace `tokenizer.json` and OpenAI tiktoken
`.tiktoken` files and implements BPE, WordPiece, and Unigram models.

## What This Is

The IREE tokenizer is a format-agnostic tokenization engine with a pull-based
streaming architecture. It accepts input and produces tokens incrementally,
with independent ownership for input storage, unresolved model state, and
output delivery.

The tokenizer object is immutable after construction and can be shared across
threads. Each encoder owns its progress, output credit, and workspace. Callers
provide fixed state and transform storage, plus an explicit allocator for model
workspace that grows with unresolved input. Reset reuses that capacity, making
pooled encoders useful for repeated requests. Bounded allocators can enforce a
request's memory budget; exhaustion fails explicitly without changing tokens.

### Key Properties

- **Streaming-first**: `encode(text)` is just `initialize() -> feed(text) ->
  finalize()`. Exact Unigram path selection is independent of input chunk
  boundaries and output capacity.
- **Explicit memory ownership**: Fixed state and transform buffers are
  caller-provided. Exact Unigram paths and fused unknown spellings can outgrow
  inline storage; their workspace uses the encoder's allocator and survives
  reset. Deinitialization releases it.
- **Stable model offsets**: Token spans use logical transformed-input bytes,
  independent of ring wrap or reclamation. Byte-fallback tokens share the span
  of the selected unknown piece. Recovering original-input coordinates through
  length-changing normalization requires source maps from those transforms;
  the model's byte positions alone do not provide that mapping.
- **HuggingFace formats**: Loads vocabulary, normalization, segmentation,
  model, decoder, and post-processing configuration from `tokenizer.json`.
- **Tiktoken compatible**: Loads OpenAI tiktoken `.tiktoken` files
  (cl100k_base, o200k_base, r50k_base, p50k_base) and produces token-for-token
  identical output to the tiktoken Python library. Merge pairs are
  reconstructed from ranks via BPE simulation — verified 100% accurate against
  HuggingFace's explicit merge lists.
- **Safe for untrusted input**: All arithmetic uses checked overflow helpers.
  No undefined behavior on malformed UTF-8, pathological regex patterns, or
  adversarial input.

## Performance

Throughput in MiB/s (one-shot encode and decode, pool rotation with 64
copies defeating cache warming, Clang `-O3 -march=native` with ThinLTO,
AMD EPYC 5.4 GHz). Corpus: 593KB ASCII (Sherlock Holmes), 1.8MB CJK
(Chinese classic), 1.4MB Code (concatenated C source):

| Model | Algo | Vocab | ASCII Enc | CJK Enc | Code Enc | ASCII Dec | CJK Dec | Code Dec |
|-------|------|------:|----------:|--------:|---------:|----------:|--------:|---------:|
| GPT-2             | BPE       |  50K |  28 |  15 |  44 |   587 | 1,588 |   630 |
| Llama 3           | BPE       | 128K |  42 |   9 |  54 |   709 | 1,236 |   781 |
| Gemma 2B          | BPE       | 256K |  40 |  67 |  15 |   213 |   366 |   215 |
| Qwen 2.5          | BPE       | 151K |  24 |   8 |  44 |   736 | 1,270 |   835 |
| BLOOM             | BPE       | 250K |  24 |   7 |   2 |   621 |   984 |   762 |
| Mistral NeMo      | BPE       | 131K |  39 |  10 |  39 |   601 | 1,221 |   780 |
| DeepSeek V3       | BPE       | 129K |  13 |  10 |  16 |   658 | 1,667 | 2,125 |
| Whisper           | BPE       |  51K |  26 |  11 |  43 |   588 | 1,439 |   596 |
| BERT              | WordPiece |  30K |  44 |  36 |  40 |   814 | 3,061 |   792 |
| T5                | Unigram   |  32K |  18 |  34 |   3 |   646 | 1,808 |   556 |

Streaming encode produces bit-exact output regardless of buffer size and
tracks one-shot throughput closely. DeepSeek V3 streaming encode on ASCII
(same settings as above, pool rotation, Sherlock Holmes corpus):

| Buffer Size | Throughput | Tokens | vs One-Shot |
|------------:|-----------:|-------:|------------:|
|        1 KB |  10.7 MiB/s | 143,053 | 0.82x |
|        4 KB |  10.7 MiB/s | 143,053 | 0.82x |
|       16 KB |  11.6 MiB/s | 143,053 | 0.89x |
|       64 KB |  11.6 MiB/s | 143,053 | 0.89x |
|    One-Shot |  13.1 MiB/s | 143,053 | 1.00x |

Throughput is flat across all buffer sizes. The 0.82-0.89x streaming ratio is
the cost of the pass-through probe: when a segmenter child finds no matches
(e.g., Numbers split on English text), the pipeline probes the final child's
DFA to determine the natural segment boundary before expanding through the
full child chain. This avoids forcing word splits at arbitrary buffer
boundaries.

### Running Benchmarks

```bash
# Single model (HuggingFace JSON):
iree-bazel-run --copt=-O3 --copt=-march=native --features=thin_lto \
  //runtime/src/iree/tokenizer/tools:comprehensive_benchmark -- \
  --tokenizer_json=path/to/tokenizer.json --benchmark_min_time=1s

# Single model (tiktoken — encoding inferred from filename):
iree-bazel-run --copt=-O3 --copt=-march=native --features=thin_lto \
  //runtime/src/iree/tokenizer/tools:comprehensive_benchmark -- \
  --tokenizer_json=path/to/cl100k_base.tiktoken --benchmark_min_time=1s

# All 10 models (downloads tokenizers automatically):
runtime/src/iree/tokenizer/tools/run_benchmarks.sh --benchmark_min_time=1s
```

The `--rotate` flag allocates 64 copies of each corpus at separate heap
addresses and cycles through them. The total working set exceeds L3 cache,
defeating cache warming. Off by default.

## Architecture

### Streaming Encode Pipeline

```
Input ──┐
(bytes) │
        v
 ┌─────────────┐       (by config)
 │ UTF-8       │  OR  ┌─────────────┐
 │ Decoder     │      │ ByteLevel   │
 └──────┬──────┘      │ Encoder     │
        │             └──────┬──────┘
        └───────┬────────────┘
                │ codepoints (batched)
                v
 ┌──────────────────────────────┐
 │ Normalizer Chain             │
 │ (NFC, lowercase, strip, ...) │
 └──────────────┬───────────────┘
                │ codepoints (batched)
                v
         ┌─────────────┐
         │ UTF-8       │
         │ Encoder     │
         └──────┬──────┘
                │ bytes
                v
 ┌───────────────────────────────────┐
 │ Transform Buffer (ring buffer)    │
 │ [normalized bytes]                │
 └────────────────┬──────────────────┘
                  │
                  v
 ┌──────────────────────────────┐
 │ Segmenter                    │
 │ (Metaspace, ByteLevel, BERT, │
 │  whitespace, regex split)    │
 └──────────────┬───────────────┘
                │ segments (byte ranges)
                v
 ┌──────────────────────────────┐
 │ BPE / WordPiece / Unigram    │
 │ (subword tokenization)       │
 └──────────────┬───────────────┘
                │
                v
        Token IDs + Offsets
```

The pipeline is **pull-based**: output buffer capacity drives processing. When
the caller calls `feed()`, the encoder pulls data through each stage only as
needed to fill the output. Each stage maintains a small batch buffer (default
64 codepoints) to amortize indirect call overhead. For 1M codepoints through
4 pipeline stages, this means ~63K indirect calls instead of 4M.

### Streaming Decode Pipeline

Token IDs are looked up in the vocabulary and passed through token-local
transforms such as ByteFallback, Metaspace, or WordPiece. When those transforms
permit it, immutable predecoded vocabulary entries replace repeated decoding.
First-position forms and byte-fallback state remain separate so incremental
calls preserve the same text as one-shot decoding.

HuggingFace `Fuse` joins the logical token strings. Transforms after that boundary
consume one continuing text stream: for example, `Replace(▁, " ")`,
`ByteFallback`, `Fuse`, `Strip(" ", 1, 0)` removes one leading space from the
utterance while preserving spaces between words. ByteLevel also establishes a
joined-string boundary. The native builder expresses the text stage with
`iree_tokenizer_builder_set_decode_transform`.

The token decoder retains accepted input, and a bounded byte buffer retains its
output until the text transform accepts it. Finalization finishes the token
decoder, drains that buffer, and then finalizes the text transform. Output credit
can stop any stage without ending the logical string. A suffix strip therefore
removes the utterance's suffix, independently of feed boundaries.

An absent or null HuggingFace decoder joins tokens with ASCII spaces. An explicit
`Fuse` or empty `Sequence` concatenates them. Native builders with no decoder
create encode-only tokenizers.

### Data Flow

```
User                    Tokenizer                 Pipeline Stages
 │                          │                            │
 │  initialize(state,       │                            │
 │    buffer, capacity)     │                            │
 │─────────────────────────>│                            │
 │                          │                            │
 │  feed(state, chunk1)     │                            │
 │─────────────────────────>│  pull codepoints           │
 │                          │───────────────────────────>│
 │                          │  batched codepoints        │
 │                          │<───────────────────────────│
 │                          │  [transform, split, BPE]   │
 │  (partial tokens)        │                            │
 │<─────────────────────────│                            │
 │                          │                            │
 │  feed(state, chunk2)     │                            │
 │─────────────────────────>│            ...             │
 │  (more tokens)           │                            │
 │<─────────────────────────│                            │
 │                          │                            │
 │  finalize(state)         │  flush remaining           │
 │─────────────────────────>│───────────────────────────>│
 │  (final tokens)          │<───────────────────────────│
 │<─────────────────────────│                            │
 │                          │                            │
 │  deinitialize(state)     │                            │
 │─────────────────────────>│                            │
```

## Design Choices

### Cache and Locality Efficiency

The pipeline is organized to keep working data in L1/L2 cache:

- **Batched pull model**: Each stage processes 64 items at a time from a small
  internal buffer (~256 bytes), which fits in L1. The upstream indirect call
  happens once per batch, not once per item.
- **Ring buffer transform**: Normalized bytes flow through a power-of-two ring
  buffer. The segmenter reads from one region while the normalizer writes to
  another, keeping both hot in cache.
- **Compact vocabulary**: Token strings are stored contiguously in a single
  allocation with offset indexing. Vocabulary lookup is a single indexed load,
  not a hash table chase.
- **Merge hash table**: BPE merge lookups use an open-addressing hash table
  with linear probing, keeping probe sequences cache-line-local.

### Algorithmic Efficiency

- **BPE**: Priority-queue based merge with a fixed-size heap. Processes one
  segment at a time with O(n log n) merges where n is the segment length
  (typically 5-20 bytes, not the full input length). The heap is embedded in
  the encode state, not heap-allocated.
- **Unigram**: F64 Viterbi paths retain one best predecessor per future end.
  Only prefixes shared by every surviving path can publish before a segment
  ends. Vocabulary width bounds the frontier and input lookahead; unresolved
  paths retain their own workspace. Selected unknown pieces are fused before
  vocabulary lookup or byte expansion.
- **Trie-based vocabulary**: Token lookup during encoding uses a trie for
  longest-prefix matching, giving O(k) lookup where k is the max token length
  (bounded, typically < 50 bytes).
- **DFA regex**: The regex engine compiles patterns to DFA at tokenizer load
  time. Matching is O(n) in input length with no backtracking, making it safe
  for untrusted input patterns.

### Memory Consumption

Fixed storage is sized from the tokenizer configuration. Exact Unigram adds
workspace only when retained paths or selected unknown text exceed inline space:

| Component | Size | Notes |
|-----------|------|-------|
| Encode state | Configuration-dependent | Pipeline stages, model frontier, and inline workspace |
| Unigram workspace | Input-dependent | Live ambiguous paths and fused unknown text; caller allocator |
| Transform buffer | 4-64 KB | Caller-provided, tunable |
| Decode state | Configuration-dependent | Token decoder state, optional text transform, and a retained edge sized from the vocabulary |
| Tokenizer (shared) | Varies | Vocabulary + merge table + regex DFA |

The fixed slabs and decode state are independent of input length. Unigram's
retained path/output workspace depends on unresolved ambiguity. For example,
with `a` scored -1 and `aa` scored -1.5, an odd-length run begins with `a` and an
even-length run begins with `aa`: no fixed lookahead can settle its first token.
The encoder retains that evidence until the segment ends. Ordinary resolved
prefixes retire as input arrives, without retokenizing earlier text.

### Composability with Streaming Workflows

Incremental input and output have independent ownership:

- **Speculative decoding**: Feed accepted tokens into the continuing decode
  state after verification. The state retains the committed prefix; it has no
  rollback operation for tentative tokens that have already been fed.
- **Streaming chat**: Feed each LLM-generated token to the decoder as it
  arrives. UTF-8 output is produced as soon as complete characters are
  available.
- **Chunked encoding**: Feed input in arbitrary-sized chunks (network packets,
  file read buffers). The tokenizer handles chunk boundaries transparently,
  including mid-codepoint splits.
- **Multi-turn context**: Control the `AT_INPUT_START` flag to handle
  Metaspace prepend behavior correctly across conversation turns without
  re-encoding history.

## Features

### Tokenization Models

- **BPE** (Byte-Pair Encoding): GPT-2, Llama, Gemma, Qwen, Mistral,
  DeepSeek, BLOOM, Whisper, etc.
- **WordPiece**: BERT and variants (DistilBERT, RoBERTa, DeBERTa, etc.)
- **Unigram**: T5, XLM-RoBERTa, CamemBERT, etc.

### Normalizers

NFC, NFD, NFKD, lowercase, strip, strip accents, BERT normalization,
precompiled (HuggingFace format), prepend, replace, regex replace, sequences
of the above.

### Segmenters (Pre-tokenizers)

ByteLevel (GPT-2), Metaspace (SentencePiece), BERT (whitespace + punctuation +
CJK), whitespace, digits, punctuation, regex split, sequences of the above.
All five HuggingFace split behaviors (Removed, Isolated, MergedWithPrevious,
MergedWithNext, Contiguous) are supported.

### Decoders

ByteLevel, ByteFallback, Metaspace, WordPiece (## strip), CTC, Replace, Strip,
sequences of the above.

### Post-processing

Template-based special token insertion (BOS, EOS, CLS, SEP) with
HuggingFace-compatible template pair support and type ID assignment.

### Format Loaders

**HuggingFace `tokenizer.json`**: Loads normalizer and pre-tokenizer
configurations, model types (BPE, WordPiece, Unigram), decoder chains, added
tokens, and post-processor templates. A single `tokenizer.json` is
self-contained — it carries the vocabulary, merge list, regex patterns, special
tokens, and all pipeline configuration.

**OpenAI tiktoken `.tiktoken`**: Loads the `.tiktoken` BPE vocabulary format
used by GPT-4, GPT-4o, GPT-3, and Codex models. Tiktoken files store only
base64-encoded byte tokens with integer ranks — no explicit merge list, no
regex pattern, no special tokens. The loader reconstructs the full BPE merge
table from ranks via simulation: for each multi-byte token at rank R, it
simulates BPE encoding of that token's raw bytes using only merges with rank
< R, yielding the exact merge pair. Predefined configs provide the regex
pattern and special tokens for each standard encoding (cl100k_base, o200k_base,
r50k_base, p50k_base). Cross-validated against HuggingFace's explicit merge
lists with 100% token-for-token accuracy.

### Offset Tracking

Forward-propagated byte offsets through the full normalization pipeline.
Offsets are stored as run-length encoded maps (O(discontinuities) memory, not
O(input_length)), making them practical for arbitrarily long inputs.

### Regex Engine

Self-contained regex compiler and DFA execution engine. Compiles patterns at
tokenizer load time; no runtime regex dependency. O(n) matching with no
backtracking.

## Directory Layout

```
tokenizer/
  tokenizer.h/c          Core API, construction, and encode state
  decode.c               Decode execution and streaming state
  types.h                Shared types (token_id_t, offset_t, etc.)
  model.h/c              Model vtable interface
  normalizer.h/c         Normalizer vtable interface
  segmenter.h/c          Segmenter vtable interface
  decoder.h/c            Decoder vtable interface
  postprocessor.h/c      Template-based special token insertion
  special_tokens.h/c     Added token matching (pre/post normalization)

  model/                 Tokenization algorithms
    bpe.h/c              BPE with priority-queue merge
    wordpiece.h/c        WordPiece greedy longest-match
    unigram.h/c          Unigram (Viterbi) model

  normalizer/            Codepoint transforms
    nfc.c, nfd.c, nfkd.c, lowercase.c, strip.c, strip_accents.c,
    bert.c, precompiled.c, prepend.c, replace.c, regex_replace.c,
    sequence.c, passthrough.c

  segmenter/             Pre-tokenization (word boundaries)
    bert.c, whitespace.c, metaspace.c, digits.c, punctuation.c,
    split.c, sequence.c, passthrough.c

  decoder/               Token-to-text transforms
    byte_level.c, byte_fallback.c, metaspace.c, wordpiece.c,
    ctc.c, replace.c, strip.c, sequence.c, passthrough.c

  vocab/                 Vocabulary management
    vocab.h/c            Vocab lookup (id->string, string->id)
    vocab_builder.h/c    Incremental vocab construction
    vocab_hash.h/c       String-to-ID hash table
    vocab_merge_hash.h/c BPE merge pair hash table
    vocab_trie.h/c       Trie for longest-prefix matching

  regex/                 Self-contained regex engine
    compile.h/c          Pattern -> NFA -> DFA compilation
    exec.h/c             DFA execution
    internal/            Lexer, parser, NFA, DFA internals

  format/                Format loaders (tokenizer core is format-agnostic)
    huggingface/         HuggingFace tokenizer.json loader
      tokenizer_json.h/c   Top-level JSON parser
      model_json.h/c       BPE/WordPiece/Unigram JSON
      normalizer_json.h/c  Normalizer chain JSON
      segmenter_json.h/c   Pre-tokenizer JSON
      decoder_json.h/c     Decoder chain JSON
      added_tokens_json.h/c  Special token JSON
      postprocessor_json.h/c Post-processor template JSON
    tiktoken/            OpenAI tiktoken .tiktoken loader
      tiktoken.h/c         Parser, vocab/merge reconstruction, assembly

  tools/                 Benchmarks and test infrastructure
    comprehensive_benchmark.cc  Google Benchmark suite
    run_benchmarks.sh           Downloads models, runs full suite
    huggingface_smoketest.py    Correctness tests against HuggingFace
    run_smoketest.sh            Wrapper for smoketest dependencies

  testing/               Test utilities
    scoped_resource.h    RAII wrappers for test cleanup
```

The `format/` directory contains only loaders: they parse a specific file
format and populate the tokenizer builder. The core tokenizer has no knowledge
of any file format. Future formats (SentencePiece protobuf, a custom mmap
format for zero-copy loading) would be additional subdirectories under
`format/`.

## Quick Start

```c
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/format/tiktoken/tiktoken.h"
#include "iree/tokenizer/tokenizer.h"

// Load from HuggingFace JSON:
iree_tokenizer_t* tokenizer = NULL;
iree_status_t status = iree_tokenizer_from_huggingface_json(
    json_contents, iree_allocator_system(), &tokenizer);

// -- OR load from tiktoken (e.g., cl100k_base for GPT-4):
// iree_status_t status = iree_tokenizer_from_tiktoken(
//     tiktoken_contents, iree_tokenizer_tiktoken_config_cl100k_base(),
//     iree_allocator_system(), &tokenizer);

// One-shot encode.
iree_tokenizer_token_id_t tokens[1024];
iree_host_size_t count = 0;
status = iree_tokenizer_encode(
    tokenizer, iree_make_cstring_view("Hello, world!"),
    IREE_TOKENIZER_ENCODE_FLAG_NONE,
    iree_tokenizer_make_token_output(tokens, NULL, NULL, 1024),
    iree_allocator_system(), &count);

// One-shot decode.
char text[4096];
iree_host_size_t text_length = 0;
status = iree_tokenizer_decode(
    tokenizer, iree_tokenizer_make_token_id_list(tokens, count),
    IREE_TOKENIZER_DECODE_FLAG_NONE,
    iree_make_mutable_string_view(text, sizeof(text)),
    iree_allocator_system(), &text_length);

iree_tokenizer_free(tokenizer);
```

### Streaming Encode

```c
// Calculate required state size.
iree_host_size_t state_size = 0;
iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size);

// Allocate state + transform buffer (caller controls placement).
uint8_t* state_storage = malloc(state_size);
size_t buffer_size = iree_tokenizer_transform_buffer_recommended_size(4096);
uint8_t* transform_buffer = malloc(buffer_size);

// Initialize streaming state.
iree_tokenizer_encode_state_t* state = NULL;
iree_tokenizer_encode_state_initialize(
    tokenizer,
    iree_make_byte_span(state_storage, state_size),
    iree_make_byte_span(transform_buffer, buffer_size),
    iree_tokenizer_offset_run_list_empty(),
    IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START,
    iree_allocator_system(),
    &state);

// Feed chunks as they arrive (network, file, etc.).
while (has_more_input()) {
    iree_string_view_t chunk = get_next_chunk();
    while (chunk.size > 0) {
        iree_host_size_t bytes_consumed = 0, tokens_written = 0;
        iree_tokenizer_encode_state_feed(
            state, chunk, output, &bytes_consumed, &tokens_written);
        chunk.data += bytes_consumed;
        chunk.size -= bytes_consumed;
        process_tokens(output, tokens_written);
    }
}

// Flush remaining tokens.
iree_host_size_t final_count = 0;
iree_tokenizer_encode_state_finalize(state, output, &final_count);
process_tokens(output, final_count);

iree_tokenizer_encode_state_deinitialize(state);
free(transform_buffer);
free(state_storage);
```

### Streaming Decode (Token-at-a-Time)

```c
iree_tokenizer_decode_state_t* state = NULL;
iree_tokenizer_decode_state_initialize(
    tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE,
    iree_make_byte_span(state_storage, state_size), &state);

// Feed each token as the LLM generates it.
while (llm_has_more_tokens()) {
    iree_tokenizer_token_id_t token = llm_sample_next();
    char text[64];
    iree_host_size_t consumed = 0, written = 0;
    iree_tokenizer_decode_state_feed(
        state, iree_tokenizer_make_token_id_list(&token, 1),
        iree_make_mutable_string_view(text, sizeof(text)),
        &consumed, &written);
    if (written > 0) {
        stream_to_user(text, written);  // UTF-8 safe for display
    }
}

iree_host_size_t final_bytes = 0;
iree_tokenizer_decode_state_finalize(
    state, iree_make_mutable_string_view(text, sizeof(text)), &final_bytes);
iree_tokenizer_decode_state_deinitialize(state);
```
