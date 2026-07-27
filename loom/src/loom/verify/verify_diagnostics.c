// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_diagnostics.h"

#include "loom/format/text/printer.h"

static bool loom_verify_resolve_location_id(const loom_verify_state_t* state,
                                            loom_location_id_t location,
                                            loom_source_range_t* out_range) {
  if (!loom_source_resolve(state->source_resolver, state->module, location,
                           out_range)) {
    return false;
  }
  if (out_range->provenance == LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE &&
      out_range->source.size > 0) {
    out_range->provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  }
  return true;
}

// Resolves an op's location to a source range via the configured resolver.
static bool loom_verify_resolve_location(const loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         loom_source_range_t* out_range) {
  if (!op) return false;
  return loom_verify_resolve_location_id(state, op->location, out_range);
}

// Maximum per-token highlight ranges per diagnostic.
#define LOOM_VERIFY_MAX_HIGHLIGHTS 8

typedef struct loom_verify_highlight_target_t {
  // Callback ref emitted by the text printer.
  loom_print_field_ref_t printer_field_ref;

  // Sink-facing field ref stored on the output highlight.
  loom_diagnostic_field_ref_t diagnostic_field_ref;

  // Diagnostic param index that requested this highlight.
  iree_host_size_t param_index;
} loom_verify_highlight_target_t;

typedef struct loom_highlight_collector_t {
  // Highlight targets derived from structured diagnostic param metadata.
  const loom_verify_highlight_target_t* wanted_targets;

  // Number of entries in wanted_targets.
  iree_host_size_t wanted_count;

  // Per-target occurrence counters for repeated op fields.
  uint16_t target_occurrences[LOOM_VERIFY_MAX_HIGHLIGHTS];

  // Caller-owned output highlight storage.
  loom_highlight_range_t* highlights;

  // Number of entries written to highlights.
  iree_host_size_t highlight_count;
} loom_highlight_collector_t;

// Finds the first target whose field kind/index match |field_ref| and whose
// requested occurrence matches the number of equal field refs seen so far for
// that target. |target_occurrences| is updated for every equal field ref so
// repeated spans can be selected deterministically.
static const loom_verify_highlight_target_t* loom_verify_match_highlight_target(
    const loom_verify_highlight_target_t* wanted_targets,
    iree_host_size_t wanted_count, loom_print_field_ref_t field_ref,
    uint16_t* target_occurrences) {
  const loom_verify_highlight_target_t* matching_target = NULL;
  for (iree_host_size_t i = 0; i < wanted_count; ++i) {
    if (!loom_print_field_ref_equal(wanted_targets[i].printer_field_ref,
                                    field_ref)) {
      continue;
    }
    uint16_t occurrence = target_occurrences[i]++;
    if (!matching_target &&
        wanted_targets[i].diagnostic_field_ref.occurrence == occurrence) {
      matching_target = &wanted_targets[i];
    }
  }
  return matching_target;
}

static void loom_highlight_field_callback(void* user_data,
                                          loom_print_field_ref_t field_ref,
                                          iree_host_size_t start,
                                          iree_host_size_t end) {
  loom_highlight_collector_t* collector =
      (loom_highlight_collector_t*)user_data;
  if (collector->highlight_count >= LOOM_VERIFY_MAX_HIGHLIGHTS) {
    return;
  }
  const loom_verify_highlight_target_t* target =
      loom_verify_match_highlight_target(collector->wanted_targets,
                                         collector->wanted_count, field_ref,
                                         collector->target_occurrences);
  if (!target) {
    return;
  }
  collector->highlights[collector->highlight_count].start = start;
  collector->highlights[collector->highlight_count].end = end;
  collector->highlights[collector->highlight_count].field_ref =
      target->diagnostic_field_ref;
  collector->highlights[collector->highlight_count].param_index =
      target->param_index;
  ++collector->highlight_count;
}

// Converts diagnostic field-ref metadata to the printer callback ref used by
// loom_text_print_operation_with_field_callback. Returns false when
// the diagnostic ref does not identify an op field that the printer can label.
static bool loom_verify_print_field_ref(
    loom_diagnostic_field_ref_t diagnostic_field_ref,
    loom_print_field_ref_t* out_field_ref) {
  loom_print_field_kind_t kind = LOOM_PRINT_FIELD_OPERAND;
  switch (diagnostic_field_ref.kind) {
    case LOOM_DIAGNOSTIC_FIELD_OPERAND:
      kind = LOOM_PRINT_FIELD_OPERAND;
      break;
    case LOOM_DIAGNOSTIC_FIELD_RESULT:
      kind = LOOM_PRINT_FIELD_RESULT;
      break;
    case LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE:
      kind = LOOM_PRINT_FIELD_ATTR;
      break;
    case LOOM_DIAGNOSTIC_FIELD_REGION:
      kind = LOOM_PRINT_FIELD_REGION;
      break;
    case LOOM_DIAGNOSTIC_FIELD_SUCCESSOR:
      kind = LOOM_PRINT_FIELD_SUCCESSOR;
      break;
    case LOOM_DIAGNOSTIC_FIELD_NONE:
    default:
      return false;
  }
  *out_field_ref = loom_print_field_ref(kind, diagnostic_field_ref.index);
  return true;
}

// Collects field highlight targets directly from structured diagnostic params.
static iree_host_size_t loom_collect_highlight_targets(
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    loom_verify_highlight_target_t* out_targets,
    iree_host_size_t* out_omitted_count) {
  *out_omitted_count = 0;
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < param_count; ++i) {
    if (!loom_diagnostic_field_ref_is_set(params[i].field_ref)) {
      continue;
    }
    loom_print_field_ref_t printer_field_ref = {0};
    if (!loom_verify_print_field_ref(params[i].field_ref, &printer_field_ref)) {
      continue;
    }
    if (count >= LOOM_VERIFY_MAX_HIGHLIGHTS) {
      ++*out_omitted_count;
      continue;
    }
    out_targets[count++] = (loom_verify_highlight_target_t){
        .printer_field_ref = printer_field_ref,
        .diagnostic_field_ref = params[i].field_ref,
        .param_index = i,
    };
  }
  return count;
}

static bool loom_verify_print_field_ref_from_location_field(
    const loom_location_field_span_t* field_span,
    loom_print_field_ref_t* out_field_ref) {
  loom_print_field_kind_t kind = LOOM_PRINT_FIELD_OPERAND;
  switch (field_span->kind) {
    case LOOM_LOCATION_FIELD_OPERAND:
      kind = LOOM_PRINT_FIELD_OPERAND;
      break;
    case LOOM_LOCATION_FIELD_RESULT:
      kind = LOOM_PRINT_FIELD_RESULT;
      break;
    case LOOM_LOCATION_FIELD_ATTRIBUTE:
      kind = LOOM_PRINT_FIELD_ATTR;
      break;
    case LOOM_LOCATION_FIELD_REGION:
      kind = LOOM_PRINT_FIELD_REGION;
      break;
    case LOOM_LOCATION_FIELD_SUCCESSOR:
      kind = LOOM_PRINT_FIELD_SUCCESSOR;
      break;
    default:
      return false;
  }
  *out_field_ref = loom_print_field_ref(kind, field_span->index);
  return true;
}

// Collects parser-provided highlights from the op's file location after
// resolving the whole-op source range to original source text.
static iree_host_size_t loom_collect_source_backed_highlights(
    const loom_module_t* module, const loom_op_t* op,
    const loom_source_range_t* source_location,
    const loom_verify_highlight_target_t* wanted_targets,
    iree_host_size_t wanted_count, loom_highlight_range_t* out_highlights,
    iree_host_size_t max_highlight_count) {
  if (!op || wanted_count == 0 || source_location->source.size == 0 ||
      op->location == LOOM_LOCATION_UNKNOWN ||
      (iree_host_size_t)op->location >= module->locations.count) {
    return 0;
  }

  const loom_location_entry_t* location =
      &module->locations.entries[op->location];
  if (location->kind != LOOM_LOCATION_FILE ||
      location->file.field_span_count == 0 || !location->file.field_spans) {
    return 0;
  }

  uint16_t target_occurrences[LOOM_VERIFY_MAX_HIGHLIGHTS] = {0};
  iree_host_size_t highlight_count = 0;
  for (uint16_t span_index = 0; span_index < location->file.field_span_count &&
                                highlight_count < max_highlight_count;
       ++span_index) {
    loom_print_field_ref_t span_field_ref = {0};
    if (!loom_verify_print_field_ref_from_location_field(
            &location->file.field_spans[span_index], &span_field_ref)) {
      continue;
    }

    const loom_verify_highlight_target_t* target =
        loom_verify_match_highlight_target(wanted_targets, wanted_count,
                                           span_field_ref, target_occurrences);
    if (!target) continue;

    const loom_location_field_span_t* field_span =
        &location->file.field_spans[span_index];
    iree_host_size_t start_offset = loom_verify_source_byte_offset(
        source_location->source, field_span->start_line, field_span->start_col);
    iree_host_size_t end_offset = loom_verify_source_byte_offset(
        source_location->source, field_span->end_line, field_span->end_col);
    if (start_offset >= end_offset || start_offset < source_location->start ||
        end_offset > source_location->end) {
      continue;
    }

    out_highlights[highlight_count++] = (loom_highlight_range_t){
        .start = start_offset,
        .end = end_offset,
        .field_ref = target->diagnostic_field_ref,
        .param_index = target->param_index,
    };
  }

  return highlight_count;
}

// Converts a packed verifier constraint field ref into the diagnostic-local
// metadata representation. |element_offset| lets callers point at a specific
// element inside a variadic field while preserving the human-facing name shape.
loom_diagnostic_field_ref_t loom_verify_diagnostic_field_ref(
    uint8_t field_ref, uint16_t element_offset) {
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (LOOM_FIELD_REF_CATEGORY(field_ref)) {
    case LOOM_FIELD_OPERAND:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                       (uint16_t)(index + element_offset));
    case LOOM_FIELD_RESULT:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT,
                                       (uint16_t)(index + element_offset));
    case LOOM_FIELD_ATTR:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                       (uint16_t)(index + element_offset));
    case LOOM_FIELD_REGION:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_REGION,
                                       (uint16_t)(index + element_offset));
    default:
      return loom_diagnostic_field_ref_none();
  }
}

// Returns a string param annotated with the corresponding structured field ref.
loom_diagnostic_param_t loom_verify_param_string_for_diagnostic_field(
    iree_string_view_t value, loom_diagnostic_field_kind_t field_kind,
    uint16_t field_index) {
  return loom_param_with_field_ref(
      loom_param_string(value),
      loom_diagnostic_field_ref(field_kind, field_index));
}

// Returns a string param annotated with a packed verifier field ref.
loom_diagnostic_param_t loom_verify_param_string_for_field(
    iree_string_view_t value, uint8_t field_ref) {
  return loom_param_with_field_ref(
      loom_param_string(value), loom_verify_diagnostic_field_ref(field_ref, 0));
}

// Returns an indexed field-name string param annotated with the concrete
// element ref inside a variadic field.
loom_diagnostic_param_t loom_verify_param_string_for_indexed_field(
    iree_string_view_t value, uint8_t field_ref, uint16_t element_offset) {
  return loom_param_with_field_ref(
      loom_param_string(value),
      loom_verify_diagnostic_field_ref(field_ref, element_offset));
}

// Resolves labeled related ops through the source resolver.
// Returns the number of note entries written to |out_related_locations|.
static iree_host_size_t loom_verify_collect_related_locations(
    const loom_verify_state_t* state,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count,
    loom_diagnostic_related_location_t* out_related_locations,
    loom_highlight_range_t* out_related_highlights,
    iree_host_size_t* out_omitted_count) {
  *out_omitted_count = 0;
  if (!related_ops || related_op_count == 0) {
    return 0;
  }

  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0; i < related_op_count; ++i) {
    if (!related_ops[i].op) continue;
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_verify_resolve_location(state, related_ops[i].op,
                                      &source_location)) {
      continue;
    }
    if (related_location_count >= LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS) {
      ++*out_omitted_count;
      continue;
    }
    loom_diagnostic_related_location_t* related_location =
        &out_related_locations[related_location_count];
    *related_location = (loom_diagnostic_related_location_t){
        .label = related_ops[i].label,
        .source_location = source_location,
    };
    if (loom_diagnostic_field_ref_is_set(related_ops[i].field_ref)) {
      loom_print_field_ref_t printer_field_ref = {0};
      if (loom_verify_print_field_ref(related_ops[i].field_ref,
                                      &printer_field_ref)) {
        const loom_verify_highlight_target_t highlight_target = {
            .printer_field_ref = printer_field_ref,
            .diagnostic_field_ref = related_ops[i].field_ref,
            .param_index = 0,
        };
        related_location->highlight_count =
            loom_collect_source_backed_highlights(
                state->module, related_ops[i].op,
                &related_location->source_location, &highlight_target,
                /*wanted_count=*/1,
                &out_related_highlights[related_location_count],
                /*max_highlight_count=*/1);
        if (related_location->highlight_count > 0) {
          related_location->highlights =
              &out_related_highlights[related_location_count];
        }
      }
    }
    ++related_location_count;
  }
  return related_location_count;
}

// Emits one structured diagnostic request through the configured sink.
//
// Source resolution strategy:
//   1. Try the configured source resolver (original source text).
//   2. If that fails, print the op to text and use the printed
//      representation as the diagnostic source.
//
// Per-token highlighting is derived automatically from structured field refs
// attached to diagnostic params. Those refs are passed to the printer's field
// callback, which records byte ranges and preserves the originating param index
// for caret output and machine JSON.
void loom_verify_emit_diagnostic(loom_verify_state_t* state,
                                 const loom_diagnostic_emission_t* emission) {
  if (!iree_status_is_ok(state->diagnostic_status)) return;

  if (emission->error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++state->result->error_count;
  } else if (emission->error->severity == LOOM_DIAGNOSTIC_WARNING) {
    ++state->result->warning_count;
  }

  if (!state->sink.fn) return;

  loom_diagnostic_t diagnostic = {0};
  diagnostic.severity = emission->error->severity;
  diagnostic.error = emission->error;
  diagnostic.params = emission->params;
  diagnostic.param_count = emission->param_count;
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.origin.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
  diagnostic.source_location.provenance =
      LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;

  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  loom_highlight_range_t
      related_highlights[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  diagnostic.related_location_count = loom_verify_collect_related_locations(
      state, emission->related_ops, emission->related_op_count,
      related_locations, related_highlights,
      &diagnostic.related_location_omitted_count);
  if (diagnostic.related_location_count > 0) {
    diagnostic.related_locations = related_locations;
  }

  // Collect structured field refs from params for per-token highlighting.
  loom_verify_highlight_target_t highlight_targets[LOOM_VERIFY_MAX_HIGHLIGHTS];
  iree_host_size_t highlight_target_count = loom_collect_highlight_targets(
      emission->params, emission->param_count, highlight_targets,
      &diagnostic.highlight_omitted_count);
  loom_highlight_range_t highlights[LOOM_VERIFY_MAX_HIGHLIGHTS];

  // Try the source resolver first (original source text).
  bool resolved = loom_verify_resolve_location(state, emission->op,
                                               &diagnostic.source_location);
  if (resolved) {
    diagnostic.origin = diagnostic.source_location;
    diagnostic.highlight_count = loom_collect_source_backed_highlights(
        state->module, emission->op, &diagnostic.source_location,
        highlight_targets, highlight_target_count, highlights,
        IREE_ARRAYSIZE(highlights));
    if (diagnostic.highlight_count > 0) {
      diagnostic.highlights = highlights;
    }
  }

  // Fallback: print the op and use the printed text as the source.
  // The printer's field callback records byte ranges for the derived
  // field refs, giving per-token caret underlines.
  iree_string_builder_t op_text_builder;
  loom_highlight_collector_t collector = {0};
  bool printed_op = false;
  if (!resolved && emission->op) {
    iree_string_builder_initialize(state->module->context->allocator,
                                   &op_text_builder);

    collector.wanted_targets = highlight_targets;
    collector.wanted_count = highlight_target_count;
    collector.highlights = highlights;
    collector.highlight_count = 0;

    loom_print_field_callback_t field_callback = {
        .fn = highlight_target_count > 0 ? loom_highlight_field_callback : NULL,
        .user_data = &collector,
    };
    iree_status_t print_status = loom_text_print_operation_with_field_callback(
        state->module, emission->op, &op_text_builder,
        LOOM_TEXT_PRINT_USE_ALIASES, field_callback);
    if (!iree_status_is_ok(print_status)) {
      // Printing failed (OOM, etc.). Use a static fallback so the
      // diagnostic still has something to display with carets.
      iree_status_free(print_status);
      iree_string_builder_deinitialize(&op_text_builder);
      static const char kFallback[] = "<failed to print op>";
      loom_source_range_t fallback_range = {
          .provenance = LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK,
          .filename = IREE_SV("<verifier>"),
          .source = iree_make_string_view(kFallback, sizeof(kFallback) - 1),
          .start = 0,
          .end = sizeof(kFallback) - 1,
          .start_line = 1,
          .start_column = 1,
          .end_line = 1,
          .end_column = (uint32_t)sizeof(kFallback),
      };
      diagnostic.origin = fallback_range;
      diagnostic.source_location = fallback_range;
    } else if (iree_string_builder_size(&op_text_builder) > 0) {
      iree_host_size_t text_length = iree_string_builder_size(&op_text_builder);
      loom_source_range_t op_range = {0};
      op_range.provenance = LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK;
      op_range.filename = IREE_SV("<verifier>");
      op_range.source = iree_make_string_view(
          iree_string_builder_buffer(&op_text_builder), text_length);
      op_range.start = 0;
      op_range.end = text_length;
      op_range.start_line = 1;
      op_range.start_column = 1;
      op_range.end_line = 1;
      op_range.end_column = (uint32_t)text_length + 1;
      diagnostic.origin = op_range;
      diagnostic.source_location = op_range;
      printed_op = true;

      if (collector.highlight_count > 0) {
        diagnostic.highlights = highlights;
        diagnostic.highlight_count = collector.highlight_count;
      }
    } else {
      iree_string_builder_deinitialize(&op_text_builder);
    }
  }

  iree_status_t emit_status = loom_diagnostic_emit(&state->sink, &diagnostic);

  if (printed_op) {
    iree_string_builder_deinitialize(&op_text_builder);
  }

  loom_verify_record_diagnostic_status(state, emit_status);
}

void loom_verify_emit_structured(loom_verify_state_t* state,
                                 const loom_op_t* op,
                                 const loom_error_def_t* error,
                                 const loom_diagnostic_param_t* params,
                                 iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  loom_verify_emit_diagnostic(state, &emission);
}

iree_status_t loom_verify_diagnostic_emitter_fn(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_verify_state_t* state = (loom_verify_state_t*)user_data;
  loom_verify_emit_diagnostic(state, emission);
  return loom_verify_pending_diagnostic_status(state);
}
