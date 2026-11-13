# Launch a published profile

`loom-launch` is the launcher for published Loom profiles: it finds a package
that fits this machine, installs it, serves it behind an instrumented
OpenAI-compatible endpoint, and benchmarks it against the recipe the publisher
recorded. The native server and the acquisition command remain what they are;
the launcher chooses, supervises and observes.

Every command speaks `--json` and keeps its state under one directory
(`--state`, `$LOOM_LAUNCH_STATE`, or `~/.local/state/loom-launch`).

## Find a profile

A published profile is a repository with a root `model.json`, tagged for Loom
serving, and usually carrying `info.json` measurements and an `llm_config.json`
chat policy. Search reads those documents and reports what matters before a
download:

```sh
loom-launch search llama --limit 5
```

```text
profile                                              quant     ctx    memory    decode   fit
----------------------------------------------------------------------------------------------
loom-fun/unsloth-Llama-3.2-3B-Instruct-loom          Q4_K_M   131072   4.1 GiB     92.4   0.95
```

`fit` scores the profile for this machine: whether the memory the card was
validated with fits, whether the card was measured on this GPU architecture, and
whether its recorded gates passed. `--no-details` lists names only; `--offline`
answers from cached documents.

Inspect one profile, including the prompt its chat policy renders:

```sh
loom-launch show loom-fun/unsloth-Llama-3.2-3B-Instruct-loom \
  --prompt "Name three databases." --thinking off
```

## Install

```sh
loom-launch install loom-fun/unsloth-Llama-3.2-3B-Instruct-loom
```

Installation runs `loom-fetch`, so locks, caches, gated-repository approval and
materialization behave exactly as the publisher documented; the launcher adds a
default destination under its state directory and records the installation.
`--materialize`, `--output`, `--preset` and `--revision` are passed through.

## Serve

```sh
loom-launch serve loom-fun/unsloth-Llama-3.2-3B-Instruct-loom --context 32768
```

```text
launching loom-fun/unsloth-Llama-3.2-3B-Instruct-loom
  run       ~/.local/state/loom-launch/runs/20261112T101500-4242
  deployment loom-fun-unsloth-Llama-3.2-3B-Instruct-loom  device amdgpu://
  context   32768 tokens, max 4096 output, 4 rows
  endpoint  http://127.0.0.1:54177
  server    http://127.0.0.1:8080 (instrumented by the proxy)
  prepared in 41.6s; ^C stops it
```

The launcher composes a deployment document from the profile's preset and the
requested capacities, starts `loom-serve`, waits until `/v1/models` lists the
deployment, then starts the instrumentation proxy. Context is configurable
within what the package declares — never above it — and an output limit that
would leave no room is refused before the server starts.

| Option | Effect |
| --- | --- |
| `--context N` | `context_capacity`; refuses more than the package's own limit. |
| `--max-tokens N` | The deployment's default output limit. |
| `--parallel N` | Resident sessions (`rows` and `max_sessions`). |
| `--draft-depth N` | Speculative draft depth; `0` disables drafting. |
| `--session-cache N`, `--prefix-cache N`, `--checkpoint-capacity N` | Retained and pinned state. |
| `--device URI`, `--port`, `--workers`, `--memory-bytes`, `--pool-backing` | The server's own resource selection. |
| `--no-proxy` | Serve the native endpoint without instrumentation. |
| `--api-key TOKEN` | Require a bearer token from clients. |
| `--detach` | Run in the background; `status`, `logs` and `stop` control it. |

`--port` selects the native listener. The instrumentation proxy binds its own
port (`--proxy-port`, or an ephemeral one), and the `endpoint` the launcher
prints — repeated by `loom-launch status` — is the address clients use.

Clients use the endpoint like any OpenAI-compatible one:

```sh
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"loom-fun-unsloth-Llama-3.2-3B-Instruct-loom",
       "messages":[{"role":"user","content":"Say hello."}]}'
```

With the launcher's own client, thinking controls and template previews come
from the package's policy:

```sh
loom-launch chat loom-fun/unsloth-Llama-3.2-3B-Instruct-loom --think off
loom-launch chat --base-url http://127.0.0.1:8080 --prompt "Say hello."
```

## Instrumentation

Every request the proxy carries is one JSON object in
`runs/<run>/requests.jsonl`: route, model, status, time to first token, decode
rate, token usage, cached tokens, bytes and the retained-session name when one
was used. Three routes expose the live picture:

```sh
curl --fail http://127.0.0.1:8080/loom/metrics      # Prometheus text
curl --fail http://127.0.0.1:8080/loom/status       # launch, upstream, snapshot
curl --fail http://127.0.0.1:8080/loom/requests     # recent requests
```

`loom-launch status` summarizes the current launch; `loom-launch logs --follow`
shows the server's own output next to its JSONL events; `loom-launch stop` asks
the recorded process to drain and exit.

## Benchmark

`bench` measures the rows a card publishes, so a local run answers the same
questions the publisher answered:

```sh
loom-launch bench loom-fun/unsloth-Llama-3.2-3B-Instruct-loom --suite standard
```

```text
metric                      value  published   ratio  roofline
--------------------------------------------------------------
decode                       88.1       92.4   0.95x       71%
decode_2k                    81.0       84.0   0.96x       68%
prefill_1k                 2712.4     2600.0   1.04x       35%
prefill_2k                 2680.1     2500.0   1.07x       34%
prefill_8k                 2210.7     1900.0   1.16x       29%
streams4                    152.3      140.0   1.09x       61%
```

Rows come from generated documents of exact size (the reported `prompt_tokens`
is what the server counted), measured client-side over the same endpoint a user
would call. Suites are `quick` (decode, prefill 1k and 2k), `standard` (adds
decode at 2k, prefill 8k, four streams) and `full` (adds 16k and 32k, decode at
32k). `--repeats N` reports the median of N samples and keeps each sample.

Stored runs stay under `<state>/benchmarks`, are listed with `bench --list`, and
printed with `bench --show RUN_ID`. The `comparison` block pairs every measured
row with the card's published value and roofline ceiling; a run of a different
harness version records its workload identity alongside the numbers, so two
records are never compared as if they were the same measurement.

## Diagnose

```sh
loom-launch hardware        # machine, GPU architecture, memory, device URI
loom-launch doctor          # loom-serve, loom-fetch, hub, state directory, device
```

`doctor` reports a missing native server as a failure with the installation
command that provides it, so an incomplete environment is diagnosed before a
launch is attempted.
