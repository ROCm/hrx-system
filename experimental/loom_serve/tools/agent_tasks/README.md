# Live pi repair workload

Eight distinct, small numerical repairs exercise real file tools, failed tests,
retained tool histories and follow-up requests. Success means the repaired
function passes the protected executable checker and the agent recalls a result
from its test output. This is a bounded coding-task control, not a general
coding benchmark, unseen reasoning evaluation or kernel-authoring score.

The corpus covers exclusive scan, page splitting, range coalescing, stable
softmax, RMS normalization, masked mean, transpose and segmented sum. Each seed
fails its numerical cases. Agents receive the contract and buggy source, not a
reference implementation. Test outputs include the later recall case.

## Run against an existing server

The runner uses an external installation of `@earendil-works/pi-coding-agent`
(tested with 0.82.1), Node 22.19 or newer, and Python 3 on a POSIX host. It neither
installs packages nor reads the user's pi configuration, credentials, extensions
or skills. The server owns model loading and configuration independently.

```sh
node experimental/loom_serve/tools/agent_tasks/run.mjs \
  /path/to/node_modules/@earendil-works/pi-coding-agent \
  /path/to/new-results-directory \
  http://127.0.0.1:8080/v1 model-name 8 loom
```

The six arguments are the installed package directory, new output directory,
OpenAI-compatible API base URL, model name, task count (1–8), and engine
(`loom`, `llama`, `vllm` or `gufo`). Engine selection maps only the nonthinking
request option. Each run uses greedy generation, an 8,192-token client context, a
384-token per-response ceiling, at most eight provider requests and ten tool
calls per task. Compaction and retries are disabled. A truncated or failed
response cannot count as a successful task. Real local tool execution stays
inside each newly created task directory; only `candidate.py` is editable,
and the only shell command is `python3 check.py`.

For warm performance comparisons, first complete a separate request to activate
the weights on every engine; readiness alone can precede residency. Record that
warmup separately. Match checkpoint, speculation, context allowance and hardware
policy, isolate the runner, and alternate configurations. On installations using
`benchmark-lock`, wrap the command in its ordinary lease; the lease identity is
recorded when present. The runner itself imposes no inference timeout.

The [performance guide](../../docs/PERFORMANCE.md) describes measurement
boundaries and controlled comparisons. Memory observation covers the server
lifetime, including loading and warmup, independently of the warm task clock.
On unified memory, host RSS and GPU allocation counters can overlap; adding
them does not establish a physical-memory peak.

## What the result means

Tasks start 75 ms apart. After completing its repair, each agent waits
`100 + 150 * task_index` ms before its retained recall. The elapsed cohort clock
includes tool execution and these pauses, but not server startup or an external
warmup. It starts before fixture/session construction, so that small client setup
cost is included. These are closed-loop trajectories: a different model reply
can change later requests, repairs and token work. The primary score is complete
tasks per cohort, with failures and actual work counts retained. Equal requested
output caps are not equal generated work.

The initial HTTP messages and tool definitions are common across engines; pi's
physical working-directory suffix is removed from the submitted system message.
Rendered chat templates can still differ. This workload does not establish
equal-token kernel speed or quality beyond its explicit cases.

`summary.json` records every task outcome. Task directories retain candidate
source, independent evaluation, exact provider payloads, pi sessions, compact
stream events and the shared `loom-pi-calls-v1` timing/usage recording. The
manifest identifies the corpus, runner, checker and installed pi version.
Progress is JSONL on stdout. Any failed task produces a nonzero runner exit;
successful peers and failed candidates remain available for inspection.

Per-task `first_text_ms` includes only replies that emit a text delta. Tool-only
replies do not contribute, so this field is not an all-request TTFT statistic.
The call recordings retain start/end times for every provider request; stream
events preserve the separate text and tool-call progress.

The cohort summary can be inspected without a model or pi installation:

```sh
jq '{elapsed_ms, planned_tasks, completed_tasks,
     requests: ([.tasks[].requests] | add),
     tools: ([.tasks[].tools] | add),
     output_tokens: ([.tasks[].output_tokens] | add),
     failed_tasks: [.tasks[] | select(.success != true) | {task, errors}],
     completed_tasks_per_minute:
       (if .completed_tasks == .planned_tasks
        then 60000 * .completed_tasks / .elapsed_ms else null end)}' \
  /path/to/results/summary.json
```

For a complete successful run, the existing trace exporter also validates the
request recordings and emits content-free token counts, retained-prefix counts,
request durations and inter-request delays:

```sh
iree-bazel-run //experimental/loom_serve/tools:agent_trace -- \
  --output /path/to/new-trace.json /path/to/results/*/calls.jsonl
```

It rejects unfinished recordings instead of treating missing work as a fast
completion. The trace clock begins with the first provider call, whereas the
cohort summary includes fixture/session construction; their elapsed times are
not interchangeable. Successful client outcomes do not establish clean server
retirement, thermal stability or resource isolation; those observations belong
beside the client record in a performance comparison.

The checker accepts a deliberately small pure-function language and gives each
candidate subprocess two CPU seconds and 256 MiB of address space. Those limits
bound erroneous candidate code, not GPU work or asynchronous serving. This is
a controlled fixture runner, not a general-purpose security sandbox for arbitrary
agent code. The checker and its independent case coverage run through Bazel:

```sh
iree-bazel-test --config=asan \
  //experimental/loom_serve/tools/agent_tasks:check_test
```
