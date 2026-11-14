# Loom launcher

`loom-launch` finds, installs, serves and benchmarks published Loom profiles
behind an instrumented OpenAI-compatible endpoint. It runs on top of the
native server and the ordinary acquisition command: `loom-serve` owns devices,
compilation and inference, `loom-fetch` owns acquisition, and this package owns
discovery, deployment choices, supervision and instrumentation.

```sh
loom-launch search llama --limit 5        # what is published, and does it fit here
loom-launch show loom-fun/OWNER-model-loom
loom-launch install loom-fun/OWNER-model-loom
loom-launch serve loom-fun/OWNER-model-loom --context 32768
```

## What a profile is

A profile is one published Loom package. The launcher reads four documents and
never invents what they omit:

| Document | The launcher reads |
| --- | --- |
| `model.json` | Preset, service kind, and the execution capacities the package declares. |
| `info.json` (`loom-card/2`) | Features, gates, measurement sessions, roofline ceilings, the memory the card was validated with. |
| `llm_config.json` | Architecture, checkpoint name (quantization), native context, the chat policy and the drafter. |
| `README.md` | The human summary, used for display only. |

`profile.Profile` merges those documents; `profile.Measurement` is one published
metric with its baseline and roofline; `profile.ChatPolicy` renders the package's
own framing for previews and explains which template controls it consumes.

## Commands

| Command | Purpose |
| --- | --- |
| `search` | Lists matching hub profiles, annotates each with quantization, context, memory, published decode rate and a fit score for this machine. |
| `show` | Explains one profile: features, gates, preset, chat policy, published measurements, rendered prompt. |
| `install` | Acquires a profile through `loom-fetch` (`--revision` pins it as `@<revision>`), adds the profile's own `deployment.json`/`devices.json` when it publishes them (see below), and records it in the launcher's registry. `--materialize`, `--output`, `--preset`, `--loom-fetch`, `--offline`. |
| `serve` | Composes a deployment document (or runs an installed scenario's own documents), starts `loom-serve`, waits for the model to prepare, and fronts it with the instrumentation proxy. |
| `chat` | A terminal conversation with thinking controls, prompt preview, and usage/context reporting. |
| `bench` | Runs the card's metric vocabulary locally, stores the result, and compares it with the published rows and the roofline. |
| `status`, `logs`, `stop` | Observe and stop the launch recorded in the state directory. |
| `hardware`, `doctor` | Report the machine and check the launcher's environment. |

## Configuration

Context and concurrency are chosen by the launcher, then expressed in the
deployment document the server owns:

```sh
loom-launch serve OWNER/model-loom \
  --context 65536          # context_capacity, never above the package's own limit
  --max-tokens 4096        # the deployment's default output limit
  --parallel 2             # rows and max_sessions
  --draft-depth 0          # disables the package's speculative drafting
  --session-cache 4        # retained idle histories
  --device amdgpu://       # otherwise detected
```

A launch refuses a context larger than the package was prepared for and an
output limit that leaves no room in the context, instead of discovering the
problem during preparation.

## Installed scenarios

An installed scenario carries its own deployment documents. A package holding
both `deployment.json` and `devices.json` was prepared for one machine shape (for
example a GPU and an XDNA NPU sharing an exchange pool), and `serve` runs it as
prepared, with no composed deployment and no `--device`:

```sh
loom-serve --devices=<pkg>/devices.json --model=<pkg>/deployment.json --listen=127.0.0.1:<port>
```

`loom-fetch` installs only what the card declares, which never includes those
two files, so after a successful fetch `install` looks for them, in order: in
the installed directory; in the local Hugging Face cache's snapshot of the
pinned revision; and (not with `--offline`) on the hub at that revision. Most
profiles publish neither: they install exactly as before, silently. A profile
publishing only one of them, or a hub that cannot be read, gets a `warning:` on
stderr, and the install that already succeeded stands (it serves a single-device
deployment until both files are next to the card). A found pair is reported as

```text
  scenario  deployment qwen3.8-27b on amdgpu + xdna (deployment.json + devices.json from the hub at <revision>)
```

and recorded in `installs.json` as `"scenario": true` with `"scenario_revision"`.
`show OWNER/NAME` then explains the installed package, scenario line included.
The scenario owns its capacities and devices, so `--context`, `--max-tokens`,
`--parallel`, `--draft-depth`, `--checkpoint-capacity`, `--session-cache`,
`--prefix-cache`, `--prefill-capacity` and `--device` are refused for it;
`--memory-bytes`/`--pool-backing` change a run-local copy of `devices.json`.
The native launcher (`loom/rust/loom-launch`) behaves the same way.

## Instrumentation

`serve` starts a proxy in front of the native listener unless `--no-proxy` is
given. The proxy records one JSON object per request — time to first token,
decode rate, token usage, status, bytes, session — under the run directory, and
serves the launcher's own routes:

| Route | Content |
| --- | --- |
| `/loom/metrics` | Prometheus text: request counters by route and status, token totals, TTFT and decode-rate quantiles, in-flight and upstream gauges. |
| `/loom/status` | The launch (repository, deployment, device, run directory) and the proxy's snapshot. |
| `/loom/requests` | The most recent instrumented requests. |

`--api-key` requires a bearer token from clients. Real server routes
(`/v1/chat/completions`, `/v1/models`, `/v1/status`, `/metrics`, `/logz`,
`/dashboard/`, checkpoints) pass through unchanged.

This launcher still ties the endpoint to the model: `serve` is one process that
supervises `loom-serve` and runs the proxy in front of it, so `stop` (or a
model that exits) ends the endpoint and its `/loom/*` routes with the model,
and there are no `launcher`, `start` or `unload` verbs. The native launcher
(`loom/rust/loom-launch`) splits the two: its launcher service keeps the
endpoint, dashboard and instrumentation up with no model loaded (`/v1/*`
answers 503 `no model is loaded`), loads and unloads models under it, and
records itself in `launcher.json` next to `current.json`. Both launchers share
the state directory and `current.json`; this launcher ignores `launcher.json`,
so stop a native launcher with its own `loom-launch launcher stop`.

## Benchmarks

`bench` measures the same rows a card publishes: decode at a short prompt and at
2k and 32k context, prefill over 1k through 32k documents, four concurrent
streams. Workloads are generated from a seeded word stream
(`workload.WORKLOAD_IDENTITY`), so two runs of this harness are comparable; each
row reports the prompt tokens the server counted rather than an estimate.

```sh
loom-launch bench OWNER/model-loom --suite standard --repeats 2
loom-launch bench --list
loom-launch bench --show 20261112T101500-4242
```

Results are stored under `<state>/benchmarks/<profile>/<run>.json` with the
command, device, workload identity and per-row samples. `comparison` pairs each
measured row with the card's published value and its roofline ceiling, which is
what makes a local run answerable against the recipe it came from.

## State

Everything the launcher keeps lives under one directory
(`--state`, `$LOOM_LAUNCH_STATE`, or `~/.local/state/loom-launch`):

```text
installs.json            installed profiles and their cards (scenario, scenario_revision
                         when the install carries deployment.json + devices.json)
models/<slug>/           default installation targets
runs/<run>/serve.json    the deployment document that was served
runs/<run>/serve.log     every line the native server wrote
runs/<run>/events.jsonl  the server's JSONL events, plus launcher events
runs/<run>/requests.jsonl instrumented client requests
runs/<run>/meta.json     command, device, capacities, timings, exit status
current.json             the live launch (pid, endpoints, deployment)
benchmarks/              stored local runs
cache/profiles/          cached profile documents for `--offline`
```

## Design notes

- The launcher is standard-library only. Hub access is a small `urllib` client;
  acquisition delegates to `loom-fetch`, so locks, caches, gated repositories
  and materialization behave exactly as the publisher tested.
- The server's contract is honoured rather than duplicated: capacities are
  written into a deployment document, and unsupported values fail at launch.
- Instrumentation is client-side by construction. A request's time to first
  token and decode rate are measured where the client sees them, which is the
  same evidence a published card reports.
- `testing.FakeOpenAIServer` is the endpoint the launcher's own tests run
  against: streaming and complete completions, configurable latency, token
  counts and failures, with no device or checkpoint.
