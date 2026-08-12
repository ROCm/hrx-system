# Loom bytecode

This package owns Loom's canonical binary format, structured malformed-input
diagnostics, retained metadata index, and full module reader and writer.
Bytecode is untrusted at the public `loom_bytecode_read_*` boundary. Successful
reads produce verified bounds and table relationships that downstream reader
stages consume directly.

## Diagnosed-error unwinding

The public reader distinguishes user-authored format errors from failures of
the compiler process. Malformed bytecode emits a structured `BYTECODE`
diagnostic and is reported through `loom_bytecode_read_result_t::error_count`;
the public read returns OK after the diagnostic is accepted. Allocation,
diagnostic-sink, and other infrastructure failures return their original
non-OK status.

Private reader functions use `IREE_STATUS_DEFERRED` to unwind after an error
diagnostic has been emitted. This is a bytecode-local exception to the normal
rule that `iree_status_t` is not a control-flow token:

- `loom_bytecode_reader_emit_error` increments the error count, emits the
  structured diagnostic, and returns code-only `IREE_STATUS_DEFERRED` when the
  sink accepts it.
- The marker is created with `iree_status_from_code`; it has no allocation,
  message, payload, or captured backtrace.
- Private functions propagate it with ordinary status handling, including
  `IREE_RETURN_IF_ERROR` and `IREE_RETURN_AND_END_ZONE_IF_ERROR`. Outputs from a
  call returning non-OK are never consumed.
- The metadata, retained-index, and shared module public boundaries are the
  only places that consume the exact code-only marker, and only when an error
  was counted by the bytecode decoder. The shared module boundary does this
  before invoking the general IR verifier. The boundaries publish the
  diagnostic result, withhold invalid output objects, and return OK.
- Every other non-OK status propagates unchanged. `IREE_STATUS_DEFERRED` is
  reserved within the private bytecode-reader call graph, so a diagnostic sink
  returning it is rejected as an invalid callback result.

This contract keeps malformed-input diagnostics structured while preserving a
single internal unwind channel. Internal parsing never polls the cumulative
error count after a call; that count belongs to public result publication, not
private control flow. A new public reader entry point includes the same terminal
normalization so the private marker cannot escape the bytecode API.
