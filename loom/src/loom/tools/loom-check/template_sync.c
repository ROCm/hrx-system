// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/template_sync.h"

#include <string.h>

#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/op_defs.h"

typedef struct loom_check_template_sync_symbol_t {
  // Arena-owned symbol name.
  iree_string_view_t name;

  // Generated roles carried by the symbol's defining operation.
  loom_symbol_definition_flags_t definition_flags;

  // One-based first source line of the defining operation.
  uint16_t start_line;

  // One-based last source line of the defining operation.
  uint16_t end_line;
} loom_check_template_sync_symbol_t;

typedef struct loom_check_template_sync_case_t {
  // Arena-owned function symbol name used as the case key.
  iree_string_view_t key;

  // Arena-owned operation name of the func-like case definition.
  iree_string_view_t definition_op_name;

  // Parsed case carrying the input to synchronize.
  const loom_test_case_t* test_case;

  // Arena-owned metadata array for every symbol in the case input.
  loom_check_template_sync_symbol_t* symbols;

  // Number of entries in |symbols|.
  iree_host_size_t symbol_count;
} loom_check_template_sync_case_t;

typedef struct loom_check_template_sync_annotation_anchor_t {
  // Source range of the annotation comment line in the target file.
  loom_test_source_range_t annotation_range;

  // Target input line that the annotation is expected to match.
  iree_string_view_t target_line;

  // One-based occurrence of target_line within the target case input.
  iree_host_size_t target_line_occurrence;

  // One-based occurrence of target_line seen so far in the template case input.
  iree_host_size_t template_line_occurrence;

  // True when the annotation line originally followed its target line.
  bool insert_after_target_line;

  // True once the annotation has been emitted into the synchronized case.
  bool emitted;
} loom_check_template_sync_annotation_anchor_t;

typedef struct loom_check_template_sync_target_overlay_t {
  // Target case that owns this overlay.
  const loom_check_template_sync_case_t* source_record;

  // Target-specific definitions before the func-like op, such as the target
  // record used by its target(@...) annotation.
  iree_string_view_t case_prelude;

  // Target-specific clauses between the func-like op name and case symbol.
  iree_string_view_t definition_prefix;

  // Symbols available from target-owned definitions before the case body.
  const loom_check_template_sync_symbol_t* symbols;

  // Number of entries in |symbols|.
  iree_host_size_t symbol_count;
} loom_check_template_sync_target_overlay_t;

static bool loom_check_template_sync_target_overlay_has_content(
    const loom_check_template_sync_target_overlay_t* overlay) {
  return !iree_string_view_is_empty(
             iree_string_view_trim(overlay->case_prelude)) ||
         !iree_string_view_is_empty(
             iree_string_view_trim(overlay->definition_prefix));
}

static iree_status_t loom_check_template_sync_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_string) {
  if (source.size == 0) {
    *out_string = source;
    return iree_ok_status();
  }
  char* target_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, source.size, (void**)&target_data));
  memcpy(target_data, source.data, source.size);
  *out_string = iree_make_string_view(target_data, source.size);
  return iree_ok_status();
}

static bool loom_check_template_sync_input_is_empty(
    const loom_test_case_t* test_case) {
  return iree_string_view_is_empty(iree_string_view_trim(test_case->input));
}

static iree_status_t loom_check_template_sync_append_with_trailing_newline(
    iree_string_builder_t* builder, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, text));
  if (text.size == 0 || text.data[text.size - 1] != '\n') {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_ensure_trailing_newline(
    iree_string_builder_t* builder) {
  const iree_host_size_t size = iree_string_builder_size(builder);
  if (size > 0 && iree_string_builder_buffer(builder)[size - 1] != '\n') {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_ensure_trailing_blank_line(
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_check_template_sync_ensure_trailing_newline(builder));
  const iree_host_size_t size = iree_string_builder_size(builder);
  if (size > 0 &&
      (size == 1 || iree_string_builder_buffer(builder)[size - 2] != '\n')) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_check_template_sync_consume_line(
    iree_string_view_t* remaining) {
  iree_host_size_t newline_pos =
      iree_string_view_find(*remaining, IREE_SV("\n"), 0);
  if (newline_pos == IREE_STRING_VIEW_NPOS) {
    iree_string_view_t line = *remaining;
    *remaining = iree_string_view_empty();
    return line;
  }
  iree_string_view_t line = iree_string_view_substr(*remaining, 0, newline_pos);
  *remaining =
      iree_string_view_substr(*remaining, newline_pos + 1, IREE_HOST_SIZE_MAX);
  return line;
}

static iree_status_t loom_check_template_sync_append_line(
    iree_string_builder_t* builder, iree_string_view_t line) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, line));
  return iree_string_builder_append_cstring(builder, "\n");
}

static iree_string_view_t loom_check_template_sync_trim_trailing_blank_lines(
    iree_string_view_t text) {
  iree_host_size_t end = text.size;
  while (end > 0) {
    iree_host_size_t line_end = end;
    if (line_end > 0 && text.data[line_end - 1] == '\n') {
      --line_end;
    }
    iree_host_size_t line_start = line_end;
    while (line_start > 0 && text.data[line_start - 1] != '\n') {
      --line_start;
    }
    iree_string_view_t line =
        iree_string_view_substr(text, line_start, line_end - line_start);
    if (!iree_string_view_is_empty(iree_string_view_trim(line))) {
      break;
    }
    end = line_start;
  }
  return iree_string_view_substr(text, 0, end);
}

static bool loom_check_template_sync_line_at(iree_string_view_t source,
                                             iree_host_size_t target_line,
                                             iree_string_view_t* out_line) {
  if (target_line == 0) {
    return false;
  }
  iree_host_size_t current_line = 1;
  iree_string_view_t remaining = source;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_template_sync_consume_line(&remaining);
    if (current_line == target_line) {
      *out_line = line;
      return true;
    }
    ++current_line;
  }
  return false;
}

static bool loom_check_template_sync_line_occurrence_at(
    iree_string_view_t source, iree_host_size_t target_line,
    iree_string_view_t target_line_text, iree_host_size_t* out_occurrence) {
  *out_occurrence = 0;
  if (target_line == 0) {
    return false;
  }
  iree_string_view_t trimmed_target_line =
      iree_string_view_trim(target_line_text);
  iree_host_size_t current_line = 1;
  iree_host_size_t occurrence = 0;
  iree_string_view_t remaining = source;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_template_sync_consume_line(&remaining);
    if (iree_string_view_equal(iree_string_view_trim(line),
                               trimmed_target_line)) {
      ++occurrence;
    }
    if (current_line == target_line) {
      *out_occurrence = occurrence;
      return occurrence > 0;
    }
    ++current_line;
  }
  return false;
}

static bool loom_check_template_sync_is_symbol_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '$' || c == '.' || c == '-';
}

static bool loom_check_template_sync_is_space(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static bool loom_check_template_sync_find_symbol_token(
    iree_string_view_t line, iree_string_view_t key,
    iree_host_size_t* out_symbol_position) {
  *out_symbol_position = IREE_STRING_VIEW_NPOS;
  if (iree_string_view_is_empty(key)) {
    return false;
  }
  iree_host_size_t position = 0;
  while (position < line.size) {
    position = iree_string_view_find(line, key, position);
    if (position == IREE_STRING_VIEW_NPOS) {
      return false;
    }
    const iree_host_size_t at_position =
        position == 0 ? IREE_STRING_VIEW_NPOS : position - 1;
    const iree_host_size_t end_position = position + key.size;
    const bool has_symbol_prefix =
        at_position != IREE_STRING_VIEW_NPOS && line.data[at_position] == '@';
    const bool has_symbol_suffix =
        end_position < line.size &&
        loom_check_template_sync_is_symbol_char(line.data[end_position]);
    if (has_symbol_prefix && !has_symbol_suffix) {
      *out_symbol_position = at_position;
      return true;
    }
    ++position;
  }
  return false;
}

static bool loom_check_template_sync_definition_line_info(
    iree_string_view_t line, iree_string_view_t op_name, iree_string_view_t key,
    iree_host_size_t* out_op_end, iree_host_size_t* out_symbol_position) {
  *out_op_end = IREE_STRING_VIEW_NPOS;
  *out_symbol_position = IREE_STRING_VIEW_NPOS;
  iree_string_view_t trimmed = iree_string_view_trim(line);
  const iree_host_size_t leading_space =
      (iree_host_size_t)(trimmed.data - line.data);
  if (iree_string_view_is_empty(op_name) ||
      !iree_string_view_starts_with(trimmed, op_name)) {
    return false;
  }
  const iree_host_size_t op_end = leading_space + op_name.size;
  if (op_end < line.size &&
      !loom_check_template_sync_is_space(line.data[op_end])) {
    return false;
  }
  iree_host_size_t symbol_position = IREE_STRING_VIEW_NPOS;
  if (!loom_check_template_sync_find_symbol_token(line, key,
                                                  &symbol_position)) {
    return false;
  }
  if (symbol_position < op_end) {
    return false;
  }
  *out_op_end = op_end;
  *out_symbol_position = symbol_position;
  return true;
}

static loom_check_template_sync_target_overlay_t
loom_check_template_sync_collect_case_overlay(
    const loom_check_template_sync_case_t* record) {
  loom_check_template_sync_target_overlay_t overlay = {0};
  if (record == NULL) return overlay;

  const loom_test_case_t* test_case = record->test_case;
  overlay.source_record = record;
  overlay.symbols = record->symbols;
  overlay.symbol_count = record->symbol_count;

  iree_host_size_t offset = 0;
  iree_string_view_t remaining =
      loom_check_template_sync_trim_trailing_blank_lines(test_case->input);
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t before = remaining;
    iree_string_view_t line = loom_check_template_sync_consume_line(&remaining);
    iree_host_size_t op_end = IREE_STRING_VIEW_NPOS;
    iree_host_size_t symbol_position = IREE_STRING_VIEW_NPOS;
    if (loom_check_template_sync_definition_line_info(
            line, record->definition_op_name, record->key, &op_end,
            &symbol_position)) {
      overlay.case_prelude = loom_check_template_sync_trim_trailing_blank_lines(
          iree_string_view_substr(test_case->input, 0, offset));
      overlay.definition_prefix =
          iree_string_view_substr(line, op_end, symbol_position - op_end);
      return overlay;
    }
    offset += before.size - remaining.size;
  }
  return overlay;
}

static iree_string_view_t
loom_check_template_sync_strip_materialized_template_prelude(
    iree_string_view_t target_prelude, iree_string_view_t template_prelude) {
  // A synchronized case is emitted as the target-owned prelude followed by
  // the authoritative template input. Strip that exact template suffix before
  // preserving the target overlay. Repeating the operation also repairs files
  // produced by the former non-idempotent updater.
  target_prelude =
      loom_check_template_sync_trim_trailing_blank_lines(target_prelude);
  template_prelude =
      loom_check_template_sync_trim_trailing_blank_lines(template_prelude);
  while (!iree_string_view_is_empty(template_prelude) &&
         iree_string_view_ends_with(target_prelude, template_prelude)) {
    target_prelude =
        iree_string_view_remove_suffix(target_prelude, template_prelude.size);
    target_prelude =
        loom_check_template_sync_trim_trailing_blank_lines(target_prelude);
  }
  return target_prelude;
}

static loom_check_template_sync_target_overlay_t
loom_check_template_sync_collect_target_overlay(
    const loom_check_template_sync_case_t* target_record,
    const loom_check_template_sync_case_t* template_record) {
  loom_check_template_sync_target_overlay_t target_overlay =
      loom_check_template_sync_collect_case_overlay(target_record);
  const loom_check_template_sync_target_overlay_t template_overlay =
      loom_check_template_sync_collect_case_overlay(template_record);
  target_overlay.case_prelude =
      loom_check_template_sync_strip_materialized_template_prelude(
          target_overlay.case_prelude, template_overlay.case_prelude);
  return target_overlay;
}

static iree_status_t loom_check_template_sync_source_line_number(
    iree_string_view_t source, loom_test_source_range_t input_range,
    loom_test_source_range_t line_range, iree_host_size_t* out_line) {
  *out_line = 0;
  if (line_range.start_byte < input_range.start_byte ||
      line_range.start_byte > input_range.end_byte ||
      line_range.start_byte > source.size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "template sync annotation range is outside input");
  }
  iree_host_size_t line = 1;
  for (iree_host_size_t i = input_range.start_byte; i < line_range.start_byte;
       ++i) {
    if (source.data[i] == '\n') {
      ++line;
    }
  }
  *out_line = line;
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_extract_case_metadata(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    loom_check_template_sync_case_t* out_record) {
  *out_record = (loom_check_template_sync_case_t){
      .test_case = test_case,
  };

  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(host_allocator, &stripped_input);
  iree_status_t status =
      loom_test_file_strip_comments(test_case->input, &stripped_input);

  loom_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_text_parse(iree_string_builder_view(&stripped_input),
                             filename, context, block_pool, NULL, &module);
  }
  if (iree_status_is_ok(status) && module == NULL) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template synchronization requires each case input to parse cleanly");
  }

  iree_string_view_t key = iree_string_view_empty();
  iree_string_view_t definition_op_name = iree_string_view_empty();
  iree_host_size_t func_like_count = 0;
  if (iree_status_is_ok(status)) {
    if (module->symbols.count > 0) {
      status =
          iree_arena_allocate_array(arena, module->symbols.count,
                                    sizeof(loom_check_template_sync_symbol_t),
                                    (void**)&out_record->symbols);
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &module->symbols.entries[i];
      loom_check_template_sync_symbol_t* symbol_record =
          &out_record->symbols[out_record->symbol_count];
      status = loom_check_template_sync_copy_string(
          arena, module->strings.entries[symbol->name_id],
          &symbol_record->name);
      if (!iree_status_is_ok(status)) break;
      symbol_record->definition_flags = symbol->definition->flags;
      const loom_location_entry_t* location =
          &module->locations.entries[symbol->defining_op->location];
      if (location->kind == LOOM_LOCATION_FILE) {
        symbol_record->start_line = location->file.start_line;
        symbol_record->end_line = location->file.end_line;
      }
      ++out_record->symbol_count;

      loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
      if (!loom_func_like_isa(func) || !loom_func_like_body(func)) {
        continue;
      }
      ++func_like_count;
      key = module->strings.entries[symbol->name_id];
      definition_op_name = loom_op_name(module, symbol->defining_op);
    }
    if (iree_status_is_ok(status) &&
        (func_like_count != 1 || iree_string_view_is_empty(key) ||
         iree_string_view_is_empty(definition_op_name))) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template synchronization requires each case to contain exactly one "
          "func-like definition, got %" PRIhsz,
          func_like_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_template_sync_copy_string(arena, key, &out_record->key);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_template_sync_copy_string(
        arena, definition_op_name, &out_record->definition_op_name);
  }

  loom_module_free(module);
  iree_string_builder_deinitialize(&stripped_input);
  return status;
}

static iree_status_t loom_check_template_sync_collect_cases(
    const loom_test_file_t* file, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    bool allow_empty_cases, bool reject_case_run_directives,
    loom_check_template_sync_case_t** out_cases,
    iree_host_size_t* out_case_count) {
  *out_cases = NULL;
  *out_case_count = 0;
  if (file->case_count == 0) {
    return iree_ok_status();
  }

  loom_check_template_sync_case_t* cases = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, file->case_count, sizeof(*cases), (void**)&cases));
  iree_host_size_t case_count = 0;
  for (iree_host_size_t i = 0; i < file->case_count; ++i) {
    const loom_test_case_t* test_case = &file->cases[i];
    if (allow_empty_cases &&
        loom_check_template_sync_input_is_empty(test_case)) {
      continue;
    }
    if (reject_case_run_directives && test_case->has_run_directive) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template-synchronized target cases must inherit the file-level RUN "
          "directive");
    }
    loom_check_template_sync_case_t record = {0};
    IREE_RETURN_IF_ERROR(loom_check_template_sync_extract_case_metadata(
        test_case, filename, context, block_pool, arena, host_allocator,
        &record));
    for (iree_host_size_t j = 0; j < case_count; ++j) {
      if (iree_string_view_equal(cases[j].key, record.key)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate template-synchronized case for func-like definition "
            "@%.*s",
            (int)record.key.size, record.key.data);
      }
    }
    cases[case_count++] = record;
  }

  *out_cases = cases;
  *out_case_count = case_count;
  return iree_ok_status();
}

static const loom_check_template_sync_case_t*
loom_check_template_sync_find_case(const loom_check_template_sync_case_t* cases,
                                   iree_host_size_t case_count,
                                   iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < case_count; ++i) {
    if (iree_string_view_equal(cases[i].key, key)) {
      return &cases[i];
    }
  }
  return NULL;
}

static loom_check_template_sync_target_overlay_t
loom_check_template_sync_collect_default_target_overlay(
    const loom_check_template_sync_case_t* target_cases,
    iree_host_size_t target_case_count,
    const loom_check_template_sync_case_t* template_cases,
    iree_host_size_t template_case_count) {
  for (iree_host_size_t i = 0; i < target_case_count; ++i) {
    const loom_check_template_sync_case_t* template_record =
        loom_check_template_sync_find_case(template_cases, template_case_count,
                                           target_cases[i].key);
    const loom_check_template_sync_target_overlay_t candidate =
        loom_check_template_sync_collect_target_overlay(&target_cases[i],
                                                        template_record);
    if (loom_check_template_sync_target_overlay_has_content(&candidate)) {
      return candidate;
    }
  }
  return (loom_check_template_sync_target_overlay_t){0};
}

static iree_status_t loom_check_template_sync_append_source_range(
    iree_string_view_t source, loom_test_source_range_t range,
    iree_string_builder_t* builder) {
  if (range.end_byte < range.start_byte || range.end_byte > source.size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "template sync source range is out of bounds");
  }
  return iree_string_builder_append_string(
      builder, iree_string_view_substr(source, range.start_byte,
                                       range.end_byte - range.start_byte));
}

static iree_status_t loom_check_template_sync_append_target_directive(
    iree_string_view_t target_source, loom_test_source_range_t range,
    iree_string_builder_t* builder) {
  if (loom_test_source_range_is_empty(range)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_source_range(
      target_source, range, builder));
  return iree_string_builder_append_cstring(builder, "\n");
}

static iree_status_t loom_check_template_sync_append_target_directives(
    iree_string_view_t target_source, const loom_test_case_t* target_case,
    iree_string_builder_t* builder) {
  if (!target_case) {
    return iree_ok_status();
  }
  loom_test_source_range_t first = target_case->requires_directive_range;
  loom_test_source_range_t second = target_case->xfail_directive_range;
  if (!loom_test_source_range_is_empty(first) &&
      !loom_test_source_range_is_empty(second) &&
      second.start_byte < first.start_byte) {
    loom_test_source_range_t temporary = first;
    first = second;
    second = temporary;
  }
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_directive(
      target_source, first, builder));
  return loom_check_template_sync_append_target_directive(target_source, second,
                                                          builder);
}

static void loom_check_template_sync_advance_annotation_line_matches(
    loom_check_template_sync_annotation_anchor_t* anchors,
    iree_host_size_t anchor_count, iree_string_view_t template_line) {
  iree_string_view_t trimmed_template_line =
      iree_string_view_trim(template_line);
  for (iree_host_size_t i = 0; i < anchor_count; ++i) {
    loom_check_template_sync_annotation_anchor_t* anchor = &anchors[i];
    if (iree_string_view_equal(trimmed_template_line,
                               iree_string_view_trim(anchor->target_line))) {
      ++anchor->template_line_occurrence;
    }
  }
}

static iree_status_t loom_check_template_sync_append_target_annotations(
    iree_string_view_t target_source,
    loom_check_template_sync_annotation_anchor_t* anchors,
    iree_host_size_t anchor_count, iree_string_view_t template_line,
    bool insert_after_target_line, iree_string_builder_t* builder) {
  iree_string_view_t trimmed_template_line =
      iree_string_view_trim(template_line);
  for (iree_host_size_t i = 0; i < anchor_count; ++i) {
    loom_check_template_sync_annotation_anchor_t* anchor = &anchors[i];
    if (anchor->emitted ||
        anchor->insert_after_target_line != insert_after_target_line ||
        anchor->template_line_occurrence != anchor->target_line_occurrence ||
        !iree_string_view_equal(trimmed_template_line,
                                iree_string_view_trim(anchor->target_line))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_directive(
        target_source, anchor->annotation_range, builder));
    anchor->emitted = true;
  }
  return iree_ok_status();
}

static bool loom_check_template_sync_overlay_defines_symbol(
    const loom_check_template_sync_target_overlay_t* overlay,
    iree_string_view_t symbol_name) {
  if (overlay == NULL || overlay->source_record == NULL) return false;
  uint16_t case_definition_start_line = 0;
  for (iree_host_size_t i = 0; i < overlay->symbol_count; ++i) {
    const loom_check_template_sync_symbol_t* symbol = &overlay->symbols[i];
    if (iree_string_view_equal(symbol->name, overlay->source_record->key)) {
      case_definition_start_line = symbol->start_line;
      break;
    }
  }
  if (case_definition_start_line == 0) return false;
  for (iree_host_size_t i = 0; i < overlay->symbol_count; ++i) {
    const loom_check_template_sync_symbol_t* symbol = &overlay->symbols[i];
    if (!iree_string_view_equal(symbol->name, symbol_name) ||
        iree_any_bit_set(symbol->definition_flags,
                         LOOM_SYMBOL_DEFINITION_FLAG_DECLARATION) ||
        symbol->end_line == 0 ||
        symbol->end_line >= case_definition_start_line) {
      continue;
    }
    return true;
  }
  return false;
}

static bool loom_check_template_sync_template_line_is_satisfied_declaration(
    const loom_check_template_sync_case_t* template_record,
    const loom_check_template_sync_target_overlay_t* overlay,
    iree_host_size_t line_number) {
  for (iree_host_size_t i = 0; i < template_record->symbol_count; ++i) {
    const loom_check_template_sync_symbol_t* symbol =
        &template_record->symbols[i];
    if (!iree_any_bit_set(symbol->definition_flags,
                          LOOM_SYMBOL_DEFINITION_FLAG_DECLARATION) ||
        line_number < symbol->start_line || line_number > symbol->end_line) {
      continue;
    }
    if (loom_check_template_sync_overlay_defines_symbol(overlay,
                                                        symbol->name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_check_template_sync_append_template_input_line(
    iree_string_view_t template_line, iree_host_size_t line_number,
    const loom_check_template_sync_case_t* template_record,
    const loom_check_template_sync_target_overlay_t* overlay,
    iree_string_builder_t* builder) {
  if (loom_check_template_sync_template_line_is_satisfied_declaration(
          template_record, overlay, line_number)) {
    return iree_ok_status();
  }
  if (overlay == NULL ||
      iree_string_view_is_empty(overlay->definition_prefix)) {
    return loom_check_template_sync_append_line(builder, template_line);
  }
  iree_host_size_t op_end = IREE_STRING_VIEW_NPOS;
  iree_host_size_t symbol_position = IREE_STRING_VIEW_NPOS;
  if (!loom_check_template_sync_definition_line_info(
          template_line, template_record->definition_op_name,
          template_record->key, &op_end, &symbol_position)) {
    return loom_check_template_sync_append_line(builder, template_line);
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, iree_string_view_substr(template_line, 0, op_end)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, overlay->definition_prefix));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, iree_string_view_substr(template_line, symbol_position,
                                       IREE_HOST_SIZE_MAX)));
  return iree_string_builder_append_cstring(builder, "\n");
}

static iree_status_t loom_check_template_sync_append_target_overlay_prelude(
    const loom_check_template_sync_target_overlay_t* overlay,
    iree_string_builder_t* builder) {
  if (overlay == NULL || iree_string_view_is_empty(overlay->case_prelude)) {
    return iree_ok_status();
  }
  return loom_check_template_sync_append_with_trailing_newline(
      builder, overlay->case_prelude);
}

static iree_status_t loom_check_template_sync_collect_annotation_anchors(
    iree_string_view_t target_source, const loom_test_case_t* target_case,
    iree_arena_allocator_t* arena,
    loom_check_template_sync_annotation_anchor_t** out_anchors,
    iree_host_size_t* out_anchor_count) {
  *out_anchors = NULL;
  *out_anchor_count = 0;
  if (!target_case || target_case->annotation_count == 0) {
    return iree_ok_status();
  }
  loom_check_template_sync_annotation_anchor_t* anchors = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, target_case->annotation_count,
                                sizeof(*anchors), (void**)&anchors));
  for (iree_host_size_t i = 0; i < target_case->annotation_count; ++i) {
    const loom_test_annotation_t* annotation = &target_case->annotations[i];
    iree_string_view_t target_line = iree_string_view_empty();
    if (!loom_check_template_sync_line_at(
            target_case->input, annotation->target_line, &target_line)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "template sync annotation target line %" PRIhsz
                              " is out of range",
                              annotation->target_line);
    }
    iree_host_size_t target_line_occurrence = 0;
    if (!loom_check_template_sync_line_occurrence_at(
            target_case->input, annotation->target_line, target_line,
            &target_line_occurrence)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template sync annotation target line occurrence is out of range");
    }
    iree_host_size_t annotation_line = 0;
    IREE_RETURN_IF_ERROR(loom_check_template_sync_source_line_number(
        target_source, target_case->input_range, annotation->source_range,
        &annotation_line));
    anchors[i] = (loom_check_template_sync_annotation_anchor_t){
        .annotation_range = annotation->source_range,
        .target_line = target_line,
        .target_line_occurrence = target_line_occurrence,
        .insert_after_target_line = annotation_line > annotation->target_line,
    };
  }
  *out_anchors = anchors;
  *out_anchor_count = target_case->annotation_count;
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_append_input_with_annotations(
    iree_string_view_t target_source,
    const loom_check_template_sync_case_t* template_record,
    const loom_check_template_sync_case_t* target_record,
    const loom_check_template_sync_target_overlay_t* default_overlay,
    iree_arena_allocator_t* arena, iree_string_builder_t* builder) {
  const loom_test_case_t* template_case = template_record->test_case;
  const loom_test_case_t* target_case =
      target_record ? target_record->test_case : NULL;
  loom_check_template_sync_annotation_anchor_t* anchors = NULL;
  iree_host_size_t anchor_count = 0;
  IREE_RETURN_IF_ERROR(loom_check_template_sync_collect_annotation_anchors(
      target_source, target_case, arena, &anchors, &anchor_count));
  loom_check_template_sync_target_overlay_t overlay = {0};
  if (target_record) {
    overlay = loom_check_template_sync_collect_target_overlay(target_record,
                                                              template_record);
  }
  if (!loom_check_template_sync_target_overlay_has_content(&overlay) &&
      default_overlay) {
    overlay = *default_overlay;
  }
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_overlay_prelude(
      &overlay, builder));

  iree_string_view_t remaining =
      loom_check_template_sync_trim_trailing_blank_lines(template_case->input);
  iree_host_size_t line_number = 1;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_template_sync_consume_line(&remaining);
    loom_check_template_sync_advance_annotation_line_matches(
        anchors, anchor_count, line);
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_annotations(
        target_source, anchors, anchor_count, line,
        /*insert_after_target_line=*/false, builder));
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_template_input_line(
        line, line_number, template_record, &overlay, builder));
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_annotations(
        target_source, anchors, anchor_count, line,
        /*insert_after_target_line=*/true, builder));
    ++line_number;
  }
  for (iree_host_size_t i = 0; i < anchor_count; ++i) {
    if (!anchors[i].emitted) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template sync could not anchor target annotation in template case");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_append_case(
    iree_string_view_t target_source,
    const loom_check_template_sync_case_t* template_record,
    const loom_check_template_sync_case_t* target_record,
    const loom_check_template_sync_target_overlay_t* default_overlay,
    bool is_first_case, iree_arena_allocator_t* arena,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_check_template_sync_ensure_trailing_blank_line(builder));
  if (!is_first_case) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "// ====\n\n"));
  }
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_target_directives(
      target_source, target_record ? target_record->test_case : NULL, builder));
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_input_with_annotations(
      target_source, template_record, target_record, default_overlay, arena,
      builder));
  if (target_record && target_record->test_case->has_expected_section) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "// ----\n"));
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_with_trailing_newline(
        builder, target_record->test_case->expected));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_parse_template(
    iree_string_view_t template_source, iree_arena_allocator_t* arena,
    loom_test_file_t* out_template_file) {
  IREE_RETURN_IF_ERROR(
      loom_test_file_parse(template_source, arena, out_template_file));
  if (out_template_file->has_template_directive) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "a loom-check TEMPLATE source must not itself declare TEMPLATE");
  }
  return iree_ok_status();
}

static iree_status_t loom_check_template_sync_parse_overlay_module(
    iree_string_view_t source, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loom_module_t** out_module) {
  *out_module = NULL;
  iree_string_builder_t stripped_source;
  iree_string_builder_initialize(host_allocator, &stripped_source);
  iree_status_t status =
      loom_test_file_strip_comments(source, &stripped_source);
  if (iree_status_is_ok(status)) {
    status = loom_text_parse(iree_string_builder_view(&stripped_source),
                             filename, context, block_pool, NULL, out_module);
  }
  iree_string_builder_deinitialize(&stripped_source);
  return status;
}

static iree_status_t loom_check_template_sync_validate_overlay_contracts(
    const loom_check_template_sync_case_t* template_record,
    const loom_check_template_sync_target_overlay_t* overlay,
    iree_string_view_t target_filename, iree_string_view_t template_filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator) {
  if (overlay == NULL || overlay->source_record == NULL) {
    return iree_ok_status();
  }

  const loom_check_template_sync_target_overlay_t template_overlay =
      loom_check_template_sync_collect_case_overlay(template_record);
  loom_module_t* template_module = NULL;
  loom_module_t* target_module = NULL;
  loom_module_t* linked_module = NULL;
  iree_status_t status = loom_check_template_sync_parse_overlay_module(
      template_overlay.case_prelude, template_filename, context, block_pool,
      host_allocator, &template_module);
  if (iree_status_is_ok(status)) {
    status = loom_check_template_sync_parse_overlay_module(
        overlay->case_prelude, target_filename, context, block_pool,
        host_allocator, &target_module);
  }
  if (iree_status_is_ok(status)) {
    const loom_module_t* source_modules[] = {template_module, target_module};
    const loom_link_options_t options = {
        .module_name = IREE_SV("loom_check_template_overlay"),
    };
    status = loom_link_materialized_modules(
        source_modules, IREE_ARRAYSIZE(source_modules), &options, block_pool,
        host_allocator, &linked_module);
  }

  loom_module_free(linked_module);
  loom_module_free(target_module);
  loom_module_free(template_module);
  return status;
}

iree_status_t loom_check_template_sync_build_source(
    iree_string_view_t target_source, const loom_test_file_t* target_file,
    iree_string_view_t target_filename, iree_string_view_t template_source,
    iree_string_view_t template_filename, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    iree_allocator_t host_allocator, iree_string_builder_t* new_source,
    bool* out_changed) {
  iree_string_builder_reset(new_source);
  *out_changed = false;

  if (!target_file->has_template_directive) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template synchronization requires a TEMPLATE directive");
  }

  loom_test_file_t template_file = {0};
  IREE_RETURN_IF_ERROR(loom_check_template_sync_parse_template(
      template_source, arena, &template_file));

  loom_check_template_sync_case_t* template_cases = NULL;
  iree_host_size_t template_case_count = 0;
  IREE_RETURN_IF_ERROR(loom_check_template_sync_collect_cases(
      &template_file, template_filename, context, block_pool, arena,
      host_allocator, /*allow_empty_cases=*/false,
      /*reject_case_run_directives=*/false, &template_cases,
      &template_case_count));
  if (template_case_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template synchronization requires at least one template case");
  }

  loom_check_template_sync_case_t* target_cases = NULL;
  iree_host_size_t target_case_count = 0;
  IREE_RETURN_IF_ERROR(loom_check_template_sync_collect_cases(
      target_file, target_filename, context, block_pool, arena, host_allocator,
      /*allow_empty_cases=*/true, /*reject_case_run_directives=*/true,
      &target_cases, &target_case_count));
  const loom_check_template_sync_target_overlay_t default_overlay =
      loom_check_template_sync_collect_default_target_overlay(
          target_cases, target_case_count, template_cases, template_case_count);

  iree_host_size_t preamble_end = target_source.size;
  if (target_file->case_count > 0) {
    const loom_test_case_t* first_case = &target_file->cases[0];
    preamble_end = loom_test_source_range_is_empty(first_case->separator_range)
                       ? first_case->source_range.start_byte
                       : first_case->separator_range.start_byte;
  }
  const loom_test_source_range_t preamble_range = {.start_byte = 0,
                                                   .end_byte = preamble_end};
  IREE_RETURN_IF_ERROR(loom_check_template_sync_append_source_range(
      target_source, preamble_range, new_source));

  for (iree_host_size_t i = 0; i < template_case_count; ++i) {
    const loom_check_template_sync_case_t* template_record = &template_cases[i];
    const loom_check_template_sync_case_t* target_record =
        loom_check_template_sync_find_case(target_cases, target_case_count,
                                           template_record->key);
    loom_check_template_sync_target_overlay_t overlay = {0};
    if (target_record) {
      overlay = loom_check_template_sync_collect_target_overlay(
          target_record, template_record);
    }
    if (!loom_check_template_sync_target_overlay_has_content(&overlay)) {
      overlay = default_overlay;
    }
    IREE_RETURN_IF_ERROR(loom_check_template_sync_validate_overlay_contracts(
        template_record, &overlay, target_filename, template_filename, context,
        block_pool, host_allocator));
    IREE_RETURN_IF_ERROR(loom_check_template_sync_append_case(
        target_source, template_record, target_record, &default_overlay,
        /*is_first_case=*/i == 0, arena, new_source));
  }

  iree_string_view_t rebuilt_source = iree_string_builder_view(new_source);
  *out_changed = !iree_string_view_equal(target_source, rebuilt_source);
  return iree_ok_status();
}
