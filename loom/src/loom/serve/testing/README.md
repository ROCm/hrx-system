# Serving test support

These `testonly` libraries support native lifecycle tests and model-authoring
checks. The server has no Python dependency, and this package is not a client
SDK. Integration tests stay beside the server or service they exercise.

`model.h` owns decoded model documents for native tests. `stalled_stderr.h`
reproduces a full Linux stderr pipe so a lifecycle test can prove that a blocked
diagnostic sink cannot stop inference on its only worker.

The local `loom_serve_testing` Python package contains three independent helpers:

- `http` owns child-process startup, HTTP control calls, joined shutdown and
  explicitly fallible telemetry history. Readiness and completion come from
  server state and counters, not sleeps or promised event retention.
- `logs` owns an observer's cursors. Gaps, model removal and stream reincarnation
  remain explicit; a reader never owns server resources.
- `chat` consumes complete chat/SSE responses, validates terminal outcomes and
  records monotonic request, first-text and output-chunk times.

Bazel callers depend on the required library; its `imports` attribute supplies
the Python package root. CMake carries the same import directory through the
dependency. A standalone model-check environment can add this directory to
`PYTHONPATH`, without installing or modifying the Loom compiler's Python
authoring package. Each helper's unit tests live beside its implementation;
the real command and modality tests exercise the same helpers end to end.
