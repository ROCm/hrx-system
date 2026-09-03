# Query compile evidence

`loom-compile-report show`, `diff`, and `suggest` provide bounded views before a
reader needs the complete report schema. Raw JSON remains valuable when one of
those views identifies a specific scheduling, allocation, wait, memory, target,
or legalization question. This page maps those questions to query shapes for
the current report version.

Compile reports are version-zero same-checkout diagnostics. Regenerate a report
with the compiler used to inspect it; schema mismatch is evidence that the
report and tool do not share a contract.

## Understand the report shape

A single-entry report exposes its selected `function`, lowered symbol, emitted
artifact summary, and compiler analysis at the top level. A multi-entry
artifact additionally uses `.entries.rows[]` as the per-entry summary index.
Those rows intentionally use compact flat names such as
`allocation_spill_count`, while top-level sections retain richer nested
objects.

Missing fields mean the compiler or target did not establish that evidence.
They are not zeros. `// null` is useful when comparing optional evidence
without manufacturing a number.

Detail mode populates focused collections rather than one undifferentiated
trace:

| Collection | Question |
| --- | --- |
| `pressure_rows.rows[]` and `pressure_origin_rows.rows[]` | Which values and source families established a pressure peak? |
| `allocation_high_water_rows.rows[]` and `spill_rows.rows[]` | Which placements and spill actions followed from that pressure? |
| `source_low.rows[]`, `source_low.memory_rows[]`, and `source_low.selection_summaries.rows[]` | Which source operations selected each Low representation and memory route? |
| `schedule_band_summary_rows.rows[]` | Which semantic instruction families occupy each schedule band? |
| `wait_reason_summary_rows.rows[]` and `wait_action_rows.rows[]` | Why was each wait family required, and where was it placed? |
| `math_legalization.rows[]` and `target_legalization.rows[]` | Which source operations required representation or target repair? |
| `target_capability_rows.rows[]` | Which selected target facts constrained lowering? |

Read one single-entry summary and its multi-entry index:

```shell
jq '{function,
     lowered,
     target_export,
     code_bytes: .emission.code_byte_count,
     valu: .static_instruction_mix.vector_alu_count,
     matrix: .static_instruction_mix.matrix_count,
     wmma: .static_instruction_mix.wmma_count,
     mfma: .static_instruction_mix.mfma_count,
     planned_spills: .allocation.spill_count,
     planned_spill_slots: .allocation.spill_plan_count,
     materialized_spill_storage:
       .allocation.materialized_spill_storage_count,
     materialized_spill_bytes:
       .allocation.materialized_spill_storage_bytes,
     materialized_spill_stores:
       .allocation.materialized_spill_store_count,
     materialized_reloads: .allocation.materialized_reload_count,
     materialized_reload_bytes: .allocation.materialized_reload_bytes,
     source_packets:
       (.economics.memory.source_low.packet_count // null),
     source_unknown_dynamic_packets:
       (.economics.memory.source_low.unknown_dynamic_packet_count // null),
     low_dynamic_issued_bytes:
       (.economics.memory.dispatch_issued.total_bytes // null),
     low_global_loads:
       (.economics.memory.dispatch_issued.global_load_count // null),
     low_global_stores:
       (.economics.memory.dispatch_issued.global_store_count // null),
     final_vgprs: .target_resources.vector.final.register_count,
     scheduled_vgpr_pressure:
       .target_resources.vector.scheduled_pressure.peak_live_units,
     private_bytes: .memory.private_bytes},
    (.entries.rows[]? |
      {function,
       source_function,
       target_export_symbol,
       code_byte_count,
       low_dynamic_issued_bytes:
         (.economics.memory.dispatch_issued.total_bytes // null),
       valu: .static_instruction_mix.vector_alu_count,
       matrix: .static_instruction_mix.matrix_count,
       wmma: .static_instruction_mix.wmma_count,
       mfma: .static_instruction_mix.mfma_count,
       allocation_spill_count,
       allocation_spill_plan_count,
       final_vgprs: .target_resources.vector.final.register_count,
       scheduled_vgpr_pressure:
         .target_resources.vector.scheduled_pressure.peak_live_units,
       allocation_materialized_spill_storage_count,
       allocation_materialized_spill_store_count,
       allocation_materialized_reload_count,
       private_memory_bytes})' report.json
```

Select every object associated with one source or lowered function when the
relevant evidence may live in several detail arrays:

```shell
jq --arg f branchy '
  .. | objects |
  select(.function? == $f or
         .source_function? == $f or
         .function_name? == $f)' report.json
```

## Explain a register or spill cliff

Summary mode identifies the existence of scheduled pressure, final register
usage, planned spills, materialized spill traffic, and private memory. Detail
mode adds the provenance needed to explain the cliff:

`target_resources.{scalar,vector}.scheduled_pressure.peak_live_units` is
scheduled virtual pressure before final allocation metadata.
`target_resources.{scalar,vector}.final.register_count` is the emitted physical
register count used for occupancy. Their difference is evidence about the
allocation boundary rather than a contradiction.

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=gfx1151 \
  --output=kernel.hsaco \
  --compile-report=details \
  --compile-report-output=report.details.json
```

Find the values live at each pressure peak:

```shell
jq '.pressure_origin_rows.rows[]? |
    {function,
     register_class,
     peak_block,
     peak_operation,
     origin,
     origin_operation,
     semantic_tag,
     sample_value,
     live_units,
     live_values} |
    with_entries(select(.value != null))' report.details.json
```

Inspect the allocations establishing a high-water mark and the storage that
blocked a lower placement:

```shell
jq '.allocation_high_water_rows.rows[]? |
    {function,
     value,
     register_class,
     origin,
     semantic_tag,
     required_unit_count,
     location_base,
     location_count,
     high_water_units,
     lower_largest_free_run_unit_count,
     lower_pressure_releasable_largest_free_run_unit_count,
     active_assignment_blocker_units,
     active_storage_lease_blocker_units} |
    with_entries(select(.value != null))' report.details.json
```

Read individual spill actions or group them by the source live-range family
that produced them:

```shell
jq '.spill_rows.rows[]? |
    {function,
     kind,
     value,
     register_class,
     origin,
     semantic_tag,
     byte_size,
     store_count,
     store_bytes,
     reload_count,
     reload_bytes} |
    with_entries(select(.value != null))' report.details.json

jq '[.spill_rows.rows[]?] |
    group_by(.origin + ":" + (.semantic_tag // ""))[] |
    {count: length,
     function: .[0].function,
     origin: .[0].origin,
     semantic_tag: .[0].semantic_tag,
     bytes: (map(.byte_size) | add),
     stores: (map(.store_count) | add),
     reloads: (map(.reload_count) | add)} |
    with_entries(select(.value != null))' report.details.json
```

Repeated rows with one origin and semantic tag usually describe one live-range
family, not independent defects.

## Rank source-to-Low selections

`source_low.selection_summaries.rows[]` aggregates target selection under the
authored source operation. It remains available in summary reports so large
generated packet families can be ranked before requesting detail rows.

```shell
jq '.source_low.selection_summaries.rows[]? |
    {function,
     source_op,
     selection,
     plan_key,
     descriptor_key,
     descriptor_semantic_tag,
     selected_op_count,
     emitted_low_op_count} |
    with_entries(select(.value != null))' report.json

jq '[.source_low.selection_summaries.rows[]? |
      {function,
       source_op,
       selection,
       plan_key,
       descriptor_key,
       descriptor_semantic_tag,
       selected_op_count,
       emitted_low_op_count} |
      with_entries(select(.value != null))] |
    sort_by(.emitted_low_op_count // 0) |
    reverse |
    .[:20][]' report.json
```

Limit the view to matrix-family choices:

```shell
jq '.source_low.selection_summaries.rows[]? |
    select((.descriptor_semantic_tag // "") |
           startswith("matrix.")) |
    {function,
     source_op,
     plan_key,
     descriptor_key,
     descriptor_semantic_tag,
     selected_op_count,
     emitted_low_op_count} |
    with_entries(select(.value != null))' report.json
```

## Inspect schedule bands

Schedule-band summaries retain semantic tags, band width, node counts,
instruction-family counts, and result pressure. When exact fixed-trip
multiplicity is proven, `dynamic_instruction_mix` also reports repeated packet
counts. Its absence means dynamic multiplicity was not proven.

Read matrix-family bands:

```shell
jq '.schedule_band_summary_rows.rows[]? |
    select((.semantic_tag // "") | startswith("matrix.")) |
    {function,
     block,
     semantic_tag,
     band_count,
     node_count,
     matrix: .static_instruction_mix.matrix_count,
     wmma: .static_instruction_mix.wmma_count,
     mfma: .static_instruction_mix.mfma_count,
     swmmac: .static_instruction_mix.swmmac_count,
     smfmac: .static_instruction_mix.smfmac_count,
     dynamic_matrix: (.dynamic_instruction_mix.matrix_count // null),
     dynamic_wmma: (.dynamic_instruction_mix.wmma_count // null),
     dynamic_mfma: (.dynamic_instruction_mix.mfma_count // null),
     dynamic_swmmac: (.dynamic_instruction_mix.swmmac_count // null),
     dynamic_smfmac: (.dynamic_instruction_mix.smfmac_count // null),
     result_unit_count} |
    with_entries(select(.value != null))' report.json
```

Read global and buffer memory bands:

```shell
jq '.schedule_band_summary_rows.rows[]? |
    select(((.semantic_tag // "") | startswith("memory.global.")) or
           ((.semantic_tag // "") | startswith("memory.buffer."))) |
    {function,
     block,
     semantic_tag,
     band_count,
     node_count,
     max_band_node_count,
     global_loads: .static_instruction_mix.global_load_count,
     global_stores: .static_instruction_mix.global_store_count,
     buffer_loads: .static_instruction_mix.buffer_load_count,
     buffer_stores: .static_instruction_mix.buffer_store_count,
     dynamic_global_loads:
       (.dynamic_instruction_mix.global_load_count // null),
     dynamic_global_stores:
       (.dynamic_instruction_mix.global_store_count // null),
     dynamic_buffer_loads:
       (.dynamic_instruction_mix.buffer_load_count // null),
     dynamic_buffer_stores:
       (.dynamic_instruction_mix.buffer_store_count // null)} |
    with_entries(select(.value != null))' report.json
```

`max_band_node_count` is the widest consecutive band represented by the row;
it is structural schedule evidence rather than a hardware-cycle estimate.

## Attribute waits

Compact reports group waits by counter and reason. This is the first view for
ranking full drains and partial waits:

```shell
jq '.wait_reason_summary_rows.rows[]? |
    {function,
     counter,
     reason,
     action_count: .summary.action_count,
     full_drains: .summary.full_drain_count,
     partial_waits: .summary.partial_wait_count,
     max_outstanding_before: .summary.max_outstanding_before,
     max_drained_count: .summary.max_drained_count} |
    with_entries(select(.value != null))' report.json

jq '[.wait_reason_summary_rows.rows[]?] |
    sort_by(-.summary.full_drain_count)[] |
    {function,
     counter,
     reason,
     full_drains: .summary.full_drain_count,
     partial_waits: .summary.partial_wait_count,
     max_outstanding_before: .summary.max_outstanding_before,
     max_drained_count: .summary.max_drained_count} |
    with_entries(select(.value != null))' report.json
```

Detail mode connects each action to scheduled producer and consumer semantics:

```shell
jq '.wait_action_rows.rows[]? |
    {function,
     counter,
     action,
     reason,
     scheduled_ordinal,
     producer: .producer_semantic_tag,
     consumer: .consumer_semantic_tag,
     target_count,
     outstanding_before,
     outstanding_after,
     drained_count} |
    with_entries(select(.value != null))' report.details.json
```

## Account for memory traffic

Top-level economics separate authored logical traffic from target-Low issued
traffic. Dynamic counts and interval envelopes appear only when the compiler
proves their multiplicity and address coverage.

`economics.memory.per_workitem_issued` and
`economics.memory.dispatch_issued` estimate target-Low dynamic issued traffic.
They count modeled operation effects and descriptor widths rather than cache
transactions or executed branch paths.

```shell
jq '{source_issued_bytes:
       (.economics.memory.source_low.dispatch_issued.total_bytes // null),
     source_logical_bytes:
       (.economics.memory.source_low.dispatch_source.total_bytes // null),
     source_dynamic_bytes:
       (.economics.memory.source_low.dynamic_source_byte_count // null),
     source_read_dynamic_bytes:
       (.economics.memory.source_low.dynamic_read_byte_count // null),
     source_write_dynamic_bytes:
       (.economics.memory.source_low.dynamic_write_byte_count // null),
     source_unique_bytes:
       (.economics.memory.source_low.interval_envelope.unique_byte_count //
        null),
     source_read_unique_bytes:
       (.economics.memory.source_low.read_interval_envelope.unique_byte_count //
        null),
     source_write_unique_bytes:
       (.economics.memory.source_low.write_interval_envelope.unique_byte_count //
        null),
     low_dynamic_issued_bytes:
       (.economics.memory.dispatch_issued.total_bytes // null),
     low_global_loads:
       (.economics.memory.dispatch_issued.global_load_count // null),
     low_global_stores:
       (.economics.memory.dispatch_issued.global_store_count // null)}' \
  report.json
```

Strategy rows explain which target packet and fallback route served a memory
operation:

```shell
jq '.source_low.memory.strategies[]? |
    {function,
     memory_space,
     operation,
     packet,
     strategy,
     fallback_reason,
     storage,
     packet_count,
     dispatch_source,
     dispatch_issued,
     scalar_packet_count,
     vector_packet_count} |
    with_entries(select(.value != null))' report.json
```

Per-argument rows group traffic by source binding:

```shell
jq '[.source_low.memory.arguments[]? |
      {function,
       source_root,
       source_root_argument_index,
       memory_space,
       packet_count,
       load_packet_count,
       store_packet_count,
       dispatch_source,
       dispatch_issued,
       dynamic_source_byte_count,
       dynamic_read_byte_count,
       dynamic_write_byte_count,
       dynamic_issued_read_byte_count,
       dynamic_issued_write_byte_count,
       scalar_packet_count,
       vector_packet_count,
       read_unique_bytes: .read_interval_envelope.unique_byte_count,
       write_unique_bytes: .write_interval_envelope.unique_byte_count} |
      with_entries(select(.value != null))] |
    sort_by(-(.dispatch_issued.total_bytes // 0))[]' report.json
```

Source-root and target-strategy rollups appear under
`.source_low.memory.roots[]` and `.source_low.memory.strategies[]` when that
evidence is available.

`argument_packets` is the cross product of binding, source root, operation,
packet, selected strategy, fallback reason, and storage contract:

```shell
jq '[.source_low.memory.argument_packets[]? |
      {function,
       source_root,
       source_root_argument_index,
       memory_space,
       operation,
       packet,
       strategy,
       fallback_reason,
       storage,
       packet_count,
       load_packet_count,
       store_packet_count,
       dispatch_source,
       dispatch_issued,
       dynamic_source_byte_count,
       dynamic_read_byte_count,
       dynamic_write_byte_count,
       dynamic_issued_read_byte_count,
       dynamic_issued_write_byte_count,
       scalar_packet_count,
       vector_packet_count,
       read_unique_bytes: .read_interval_envelope.unique_byte_count,
       write_unique_bytes: .write_interval_envelope.unique_byte_count} |
      with_entries(select(.value != null))] |
    sort_by(-(.dispatch_issued.total_bytes // 0))[]' report.json
```

The physical storage schema supplies semantic facts that scalar element names
such as `f8E4M3`, `f8E5M2`, or BF8 cannot. For data known to exclude non-finite
payloads, `rounding=finite_only` records that contract. Storage that also
flushes subnormals uses `rounding=finite_flush_subnormal`. Strategy and fallback
rows then distinguish a proven packed decode from a repair path; raw scalar
spelling alone cannot establish that choice. For example,
`fp8_packed_bf16_decode_repair_zero_subnormal` and
`fp8_packed_bf16_decode_repair_zero` name different selected routes, while a
fallback such as `missing_finite_not_subnormal` names the storage fact that was
not proven.

## Read target capabilities

Target-capability rows identify the selected processor, subgroup, index width,
matrix profiles, and other provider facts used by lowering:

```shell
jq '.target_capability_rows.rows[]? |
    select(.namespace == "amdgpu" or .namespace == "target") |
    {function,
     namespace,
     key,
     value_kind,
     value_u64,
     value_bool,
     value_string} |
    with_entries(select(.value != null))' report.json
```

AMDGPU narrow matrix rows describe whether a selected type has no native
packet, an unscaled packet, a scaled packet, or both:

```shell
jq '.target_capability_rows.rows[]? |
    select(.namespace == "amdgpu" and
           (.key | startswith("matrix_") and
                   endswith("_native_kind"))) |
    {function, key, native_kind: .value_string} |
    with_entries(select(.value != null))' report.json
```

The keys currently include `matrix_fp8_native_kind`,
`matrix_bf8_native_kind`, `matrix_fp6_native_kind`,
`matrix_bf6_native_kind`, and `matrix_fp4_native_kind`. Values distinguish
`none`, `unscaled`, `scaled`, and `unscaled_scaled` without requiring
disassembly.

## Follow an IR trace

IR traces answer a different question from compile reports: they preserve the
module at selected pass boundaries rather than aggregating compiler evidence.
`loom-compile` and `loom-opt` share the `--dump-ir-*` tracing flags. Each JSONL
trace row contains a whole-module snapshot in `.ir`, even when the recorded pass
was anchored on one function.

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=kernel.hsaco \
  --dump-ir-after-all \
  --dump-ir-format=jsonl \
  --dump-ir-output=trace.jsonl

jq 'select(.stage == "prepared-low") | .pass' trace.jsonl
jq -r 'select(.stage == "prepared-low") | .ir' trace.jsonl
```

When the exact compiler boundary is known, a focused capture such as
`--dump-ir-after=source-to-low` avoids retaining every intermediate module.

A prepared-Low snapshot can be assembled with `--pipeline=none` only when it
is an ABI-complete emitter input. That includes target live-ins such as the
AMDGPU kernarg segment pointer when the selected ABI uses kernargs. The
supported reconstruction and acceptance boundary is described in
[Raise a native schedule into Loom](oracles/native-schedule.md).

## Query reports retained by a benchmark

Debug and full benchmark bundles write per-candidate report sidecars. Their
paths are the cleanest route into the recipes above:

```shell
jq 'select(.row == "compile" and .compile_report_path) |
    {candidate_id, path: .compile_report_path}' results.jsonl
```

When a report is embedded directly, retain the candidate identity around the
query result:

```shell
jq 'select(.row == "compile" and .compile_report) as $event |
    $event.compile_report.pressure_origin_rows.rows[]? |
    {candidate_id: $event.candidate_id,
     function,
     register_class,
     origin,
     semantic_tag,
     live_units}' results.jsonl
```

Some consumers retain the same report below a completed benchmark result. The
adapter changes, but the report query does not:

```shell
jq 'select(.row == "benchmark" and
           .benchmark_result.compile_report) as $event |
    $event.benchmark_result.compile_report |
    {candidate_id: $event.candidate_id,
     sample_id: $event.sample_id,
     code_bytes: .emission.code_byte_count,
     final_vgprs: .target_resources.vector.final.register_count,
     scheduled_vgpr_pressure:
       .target_resources.vector.scheduled_pressure.peak_live_units,
     materialized_spills:
       .allocation.materialized_spill_storage_count}' results.jsonl
```

Timing establishes whether performance changed. Report evidence explains which
emitted or modeled properties changed. A controlled conclusion needs both.
