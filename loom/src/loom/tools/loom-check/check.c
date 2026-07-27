// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/check.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns the next line from |*remaining|, advancing past the newline.
// The returned view does NOT include the trailing '\n' or '\r\n'.
// When no newline is found, returns the remainder and sets |*remaining|
// to empty. Trailing '\r' is stripped so that CRLF input produces the
// same line views as LF input.
static iree_string_view_t loom_check_consume_line(
    iree_string_view_t* remaining) {
  iree_string_view_t line;
  intptr_t newline = iree_string_view_find_char(*remaining, '\n', 0);
  if (newline < 0) {
    line = *remaining;
    *remaining = iree_string_view_empty();
  } else {
    line = iree_string_view_substr(*remaining, 0, (iree_host_size_t)newline);
    *remaining = iree_string_view_substr(
        *remaining, (iree_host_size_t)newline + 1, IREE_HOST_SIZE_MAX);
  }
  if (line.size > 0 && line.data[line.size - 1] == '\r') {
    line.size--;
  }
  return line;
}

// Returns the byte pointer just past the end of |scanner|, accounting
// for the possibility that scanner.data is NULL when the source is
// exhausted.
static const char* loom_check_scanner_end(iree_string_view_t scanner,
                                          const char* fallback_end) {
  return scanner.data ? scanner.data : fallback_end;
}

static loom_check_source_range_t loom_check_source_range_from_pointers(
    const char* source_start, const char* start, const char* end) {
  IREE_ASSERT(start <= end);
  return (loom_check_source_range_t){
      .start_byte = (iree_host_size_t)(start - source_start),
      .end_byte = (iree_host_size_t)(end - source_start),
  };
}

static loom_check_source_range_t loom_check_source_range_from_view(
    const char* source_start, iree_string_view_t view) {
  if (!view.data) return loom_check_source_range_empty();
  return loom_check_source_range_from_pointers(source_start, view.data,
                                               view.data + view.size);
}

// Returns true if |line| is a case separator: starts with "// ====".
static bool loom_check_is_case_separator(iree_string_view_t line) {
  return iree_string_view_starts_with(line, iree_make_cstring_view("// ===="));
}

// Returns true if |line| is an input/expected separator: exactly "// ----"
// with optional trailing whitespace.
static bool loom_check_is_expected_separator(iree_string_view_t line) {
  iree_string_view_t trimmed = iree_string_view_trim(line);
  return iree_string_view_equal(trimmed, iree_make_cstring_view("// ----"));
}

// Returns true if |line| starts with "// RUN:" (with or without the
// trailing space). Used for stray directive detection — any line
// starting with this prefix in the body is always a mistake.
static bool loom_check_looks_like_run(iree_string_view_t line) {
  return iree_string_view_starts_with(line, iree_make_cstring_view("// RUN:"));
}

// Returns true if |line| is a well-formed RUN directive with the
// required space after the colon.
static bool loom_check_is_run_directive(iree_string_view_t line) {
  return iree_string_view_starts_with(line, iree_make_cstring_view("// RUN: "));
}

// Returns true if |line| starts with "// XFAIL:" (with or without
// the trailing space).
static bool loom_check_looks_like_xfail(iree_string_view_t line) {
  return iree_string_view_starts_with(line,
                                      iree_make_cstring_view("// XFAIL:"));
}

// Returns true if |line| is a well-formed XFAIL directive.
static bool loom_check_is_xfail_directive(iree_string_view_t line) {
  return iree_string_view_starts_with(line,
                                      iree_make_cstring_view("// XFAIL: "));
}

// Returns true if |line| starts with "// REQUIRES:" (with or without
// the trailing space).
static bool loom_check_looks_like_requires(iree_string_view_t line) {
  return iree_string_view_starts_with(line,
                                      iree_make_cstring_view("// REQUIRES:"));
}

// Returns true if |line| is a well-formed REQUIRES directive.
static bool loom_check_is_requires_directive(iree_string_view_t line) {
  return iree_string_view_starts_with(line,
                                      iree_make_cstring_view("// REQUIRES: "));
}

// Returns true if |line| starts with "// TEMPLATE:" (with or without
// the trailing space).
static bool loom_check_looks_like_template(iree_string_view_t line) {
  return iree_string_view_starts_with(line,
                                      iree_make_cstring_view("// TEMPLATE:"));
}

// Returns true if |line| is a well-formed TEMPLATE directive.
static bool loom_check_is_template_directive(iree_string_view_t line) {
  return iree_string_view_starts_with(line,
                                      iree_make_cstring_view("// TEMPLATE: "));
}

// Returns true if |line| starts with "// CASE:" (with or without
// the trailing space). CASE is intentionally unsupported because case identity
// comes from function symbols in the IR.
static bool loom_check_looks_like_case(iree_string_view_t line) {
  return iree_string_view_starts_with(line, iree_make_cstring_view("// CASE:"));
}

//===----------------------------------------------------------------------===//
// Directive parsing
//===----------------------------------------------------------------------===//

// Parses a RUN directive value into mode and optional arguments.
// |value| is the part after "// RUN: ".
static iree_status_t loom_check_parse_run_modifiers(
    iree_string_view_t* value, loom_check_output_flags_t* out_output_flags) {
  *out_output_flags = 0;
  *value = iree_string_view_trim(*value);
  while (!iree_string_view_is_empty(*value)) {
    iree_string_view_t token;
    iree_string_view_t rest;
    iree_string_view_split(*value, ' ', &token, &rest);
    if (!iree_string_view_equal(token, IREE_SV("with-locations"))) {
      break;
    }
    if (iree_all_bits_set(*out_output_flags, LOOM_CHECK_OUTPUT_LOCATIONS)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate RUN modifier: 'with-locations'");
    }
    *out_output_flags |= LOOM_CHECK_OUTPUT_LOCATIONS;
    *value = iree_string_view_trim(rest);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_verify_run_modifiers(
    loom_check_mode_t mode, loom_check_output_flags_t output_flags) {
  if (iree_all_bits_set(output_flags, LOOM_CHECK_OUTPUT_LOCATIONS) &&
      mode != LOOM_CHECK_MODE_ROUNDTRIP && mode != LOOM_CHECK_MODE_PASS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "RUN modifier 'with-locations' only applies to roundtrip and pass "
        "modes");
  }
  return iree_ok_status();
}

static iree_status_t loom_check_parse_run_directive(
    iree_string_view_t value, loom_check_mode_t* out_mode,
    loom_check_output_flags_t* out_output_flags,
    iree_string_view_t* out_pipeline, iree_string_view_t* out_format_target,
    iree_string_view_t* out_emit_target) {
  *out_pipeline = iree_string_view_empty();
  *out_format_target = iree_string_view_empty();
  *out_emit_target = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_check_parse_run_modifiers(&value, out_output_flags));

  if (iree_string_view_equal(value, iree_make_cstring_view("roundtrip")) ||
      iree_string_view_is_empty(value)) {
    *out_mode = LOOM_CHECK_MODE_ROUNDTRIP;
    return loom_check_verify_run_modifiers(*out_mode, *out_output_flags);
  }

  if (iree_string_view_equal(value, iree_make_cstring_view("verify"))) {
    *out_mode = LOOM_CHECK_MODE_VERIFY;
    return loom_check_verify_run_modifiers(*out_mode, *out_output_flags);
  }

  if (iree_string_view_consume_prefix(&value,
                                      iree_make_cstring_view("pass "))) {
    *out_mode = LOOM_CHECK_MODE_PASS;
    *out_pipeline = iree_string_view_trim(value);
    if (iree_string_view_is_empty(*out_pipeline)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RUN: pass requires a pipeline argument");
    }
    return loom_check_verify_run_modifiers(*out_mode, *out_output_flags);
  }

  if (iree_string_view_consume_prefix(&value,
                                      iree_make_cstring_view("pass-report "))) {
    *out_mode = LOOM_CHECK_MODE_PASS_REPORT;
    *out_pipeline = iree_string_view_trim(value);
    if (iree_string_view_is_empty(*out_pipeline)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RUN: pass-report requires a pipeline argument");
    }
    return loom_check_verify_run_modifiers(*out_mode, *out_output_flags);
  }

  if (iree_string_view_consume_prefix(&value,
                                      iree_make_cstring_view("format "))) {
    *out_mode = LOOM_CHECK_MODE_FORMAT;
    *out_format_target = iree_string_view_trim(value);
    if (iree_string_view_is_empty(*out_format_target)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RUN: format requires a target argument");
    }
    return loom_check_verify_run_modifiers(*out_mode, *out_output_flags);
  }

  if (iree_string_view_consume_prefix(&value,
                                      iree_make_cstring_view("emit "))) {
    *out_mode = LOOM_CHECK_MODE_EMIT;
    *out_emit_target = iree_string_view_trim(value);
    if (iree_string_view_is_empty(*out_emit_target)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RUN: emit requires a target argument");
    }
    return loom_check_verify_run_modifiers(*out_mode, *out_output_flags);
  }

  iree_string_view_t keyword;
  iree_string_view_t rest;
  iree_string_view_split(value, ' ', &keyword, &rest);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown RUN mode: '%.*s'", (int)keyword.size,
                          keyword.data);
}

static bool loom_check_is_requirement_separator(char c) {
  return c == ',' || c == ' ' || c == '\t';
}

static bool loom_check_is_requirement_name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
}

static bool loom_check_template_path_contains_parent_segment(
    iree_string_view_t path) {
  if (iree_string_view_equal(path, IREE_SV("..")) ||
      iree_string_view_starts_with(path, IREE_SV("../")) ||
      iree_string_view_ends_with(path, IREE_SV("/.."))) {
    return true;
  }
  return iree_string_view_find(path, IREE_SV("/../"), 0) !=
         IREE_STRING_VIEW_NPOS;
}

static bool loom_check_is_template_path_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' || c == '/';
}

static iree_status_t loom_check_parse_template_path(
    iree_string_view_t value, iree_string_view_t* out_template_path) {
  *out_template_path = iree_string_view_trim(value);
  if (iree_string_view_is_empty(*out_template_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TEMPLATE requires a root-relative path");
  }
  if (iree_string_view_starts_with_char(*out_template_path, '/')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TEMPLATE path must be relative, got '%.*s'",
                            (int)out_template_path->size,
                            out_template_path->data);
  }
  if (iree_string_view_starts_with(*out_template_path, IREE_SV("./"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TEMPLATE path must be root-relative, got '%.*s'",
                            (int)out_template_path->size,
                            out_template_path->data);
  }
  if (loom_check_template_path_contains_parent_segment(*out_template_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TEMPLATE path must not contain '..', got '%.*s'",
                            (int)out_template_path->size,
                            out_template_path->data);
  }
  for (iree_host_size_t i = 0; i < out_template_path->size; ++i) {
    if (!loom_check_is_template_path_char(out_template_path->data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid character '%c' in TEMPLATE path '%.*s'",
                              out_template_path->data[i],
                              (int)out_template_path->size,
                              out_template_path->data);
    }
  }
  return iree_ok_status();
}

static bool loom_check_requirement_list_contains(
    const iree_string_view_t* requirements, iree_host_size_t requirement_count,
    iree_string_view_t requirement) {
  for (iree_host_size_t i = 0; i < requirement_count; ++i) {
    if (iree_string_view_equal(requirements[i], requirement)) return true;
  }
  return false;
}

static iree_status_t loom_check_parse_requirement_list(
    iree_string_view_t value, iree_arena_allocator_t* arena,
    iree_string_view_t** out_requirements,
    iree_host_size_t* out_requirement_count) {
  *out_requirements = NULL;
  *out_requirement_count = 0;
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "REQUIRES requires at least one requirement name");
  }

  iree_host_size_t requirement_count = 0;
  iree_string_view_t scanner = value;
  while (!iree_string_view_is_empty(scanner)) {
    scanner = iree_string_view_trim(scanner);
    if (iree_string_view_is_empty(scanner)) break;
    if (scanner.data[0] == ',') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty requirement name in REQUIRES directive");
    }
    iree_host_size_t token_length = 0;
    while (token_length < scanner.size &&
           !loom_check_is_requirement_separator(scanner.data[token_length])) {
      if (!loom_check_is_requirement_name_char(scanner.data[token_length])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid character '%c' in REQUIRES requirement name",
            scanner.data[token_length]);
      }
      ++token_length;
    }
    if (token_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty requirement name in REQUIRES directive");
    }
    ++requirement_count;
    scanner =
        iree_string_view_substr(scanner, token_length, IREE_HOST_SIZE_MAX);
    scanner = iree_string_view_trim(scanner);
    if (iree_string_view_starts_with_char(scanner, ',')) {
      scanner = iree_string_view_substr(scanner, 1, IREE_HOST_SIZE_MAX);
      if (iree_string_view_is_empty(iree_string_view_trim(scanner))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "empty requirement name in REQUIRES directive");
      }
    }
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, requirement_count,
                                                 sizeof(iree_string_view_t),
                                                 (void**)out_requirements));
  *out_requirement_count = requirement_count;

  scanner = value;
  iree_host_size_t requirement_index = 0;
  while (!iree_string_view_is_empty(scanner)) {
    scanner = iree_string_view_trim(scanner);
    iree_host_size_t token_length = 0;
    while (token_length < scanner.size &&
           !loom_check_is_requirement_separator(scanner.data[token_length])) {
      ++token_length;
    }
    (*out_requirements)[requirement_index++] =
        iree_string_view_substr(scanner, 0, token_length);
    scanner =
        iree_string_view_substr(scanner, token_length, IREE_HOST_SIZE_MAX);
    scanner = iree_string_view_trim(scanner);
    if (iree_string_view_starts_with_char(scanner, ',')) {
      scanner = iree_string_view_substr(scanner, 1, IREE_HOST_SIZE_MAX);
    }
  }
  for (iree_host_size_t i = 0; i < requirement_count; ++i) {
    if (loom_check_requirement_list_contains(*out_requirements, i,
                                             (*out_requirements)[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate requirement name '%.*s' in REQUIRES directive",
          (int)(*out_requirements)[i].size, (*out_requirements)[i].data);
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_check_combine_requirement_lists(
    const iree_string_view_t* lhs_requirements,
    iree_host_size_t lhs_requirement_count,
    const iree_string_view_t* rhs_requirements,
    iree_host_size_t rhs_requirement_count, iree_arena_allocator_t* arena,
    iree_string_view_t** out_requirements,
    iree_host_size_t* out_requirement_count) {
  *out_requirements = NULL;
  *out_requirement_count = 0;

  iree_host_size_t unique_requirement_count = 0;
  for (iree_host_size_t i = 0; i < lhs_requirement_count; ++i) {
    if (!loom_check_requirement_list_contains(lhs_requirements, i,
                                              lhs_requirements[i])) {
      ++unique_requirement_count;
    }
  }
  for (iree_host_size_t i = 0; i < rhs_requirement_count; ++i) {
    if (!loom_check_requirement_list_contains(
            lhs_requirements, lhs_requirement_count, rhs_requirements[i]) &&
        !loom_check_requirement_list_contains(rhs_requirements, i,
                                              rhs_requirements[i])) {
      ++unique_requirement_count;
    }
  }
  if (unique_requirement_count == 0) return iree_ok_status();

  iree_string_view_t* requirements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unique_requirement_count, sizeof(iree_string_view_t),
      (void**)&requirements));

  iree_host_size_t requirement_index = 0;
  for (iree_host_size_t i = 0; i < lhs_requirement_count; ++i) {
    if (!loom_check_requirement_list_contains(lhs_requirements, i,
                                              lhs_requirements[i])) {
      requirements[requirement_index++] = lhs_requirements[i];
    }
  }
  for (iree_host_size_t i = 0; i < rhs_requirement_count; ++i) {
    if (!loom_check_requirement_list_contains(
            lhs_requirements, lhs_requirement_count, rhs_requirements[i]) &&
        !loom_check_requirement_list_contains(rhs_requirements, i,
                                              rhs_requirements[i])) {
      requirements[requirement_index++] = rhs_requirements[i];
    }
  }

  *out_requirements = requirements;
  *out_requirement_count = requirement_index;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Annotation parsing
//===----------------------------------------------------------------------===//

// Tries to parse a severity prefix from |*text|. Returns true if found
// and advances |*text| past "ERROR", "WARNING", or "REMARK".
// Annotations use uppercase severity keywords to distinguish them from
// ordinary comments (which use natural language).
static bool loom_check_parse_severity(
    iree_string_view_t* text, loom_diagnostic_severity_t* out_severity) {
  if (iree_string_view_consume_prefix(text, iree_make_cstring_view("ERROR"))) {
    *out_severity = LOOM_DIAGNOSTIC_ERROR;
    return true;
  }
  if (iree_string_view_consume_prefix(text,
                                      iree_make_cstring_view("WARNING"))) {
    *out_severity = LOOM_DIAGNOSTIC_WARNING;
    return true;
  }
  if (iree_string_view_consume_prefix(text, iree_make_cstring_view("REMARK"))) {
    *out_severity = LOOM_DIAGNOSTIC_REMARK;
    return true;
  }
  return false;
}

// Returns true if |comment_text| (the part after "// ") looks like a
// diagnostic annotation — starts with a severity keyword followed by
// ':' or '@'. This distinguishes annotations from natural language
// containing words like "ERROR" (which would lack the colon/offset).
static bool loom_check_looks_like_annotation(iree_string_view_t comment_text) {
  iree_string_view_t text = iree_string_view_trim(comment_text);
  loom_diagnostic_severity_t unused;
  if (!loom_check_parse_severity(&text, &unused)) return false;
  return iree_string_view_starts_with_char(text, ':') ||
         iree_string_view_starts_with_char(text, '@');
}

// Extracts comment text from a standalone comment line. Returns the
// text after "// " if the trimmed line starts with "// ", otherwise
// returns empty. Annotations must be on standalone comment lines;
// trailing comments on IR lines are not recognized.
static iree_string_view_t loom_check_extract_comment_text(
    iree_string_view_t line) {
  iree_string_view_t trimmed = iree_string_view_trim(line);
  if (iree_string_view_starts_with(trimmed, iree_make_cstring_view("// "))) {
    return iree_string_view_substr(trimmed, 3, IREE_HOST_SIZE_MAX);
  }
  if (iree_string_view_starts_with(trimmed, iree_make_cstring_view("//"))) {
    return iree_string_view_trim(
        iree_string_view_substr(trimmed, 2, IREE_HOST_SIZE_MAX));
  }
  return iree_string_view_empty();
}

// Returns true if |trimmed_line| is a standalone annotation line.
// Uses extract_comment_text to handle both "// " and "//" prefixes
// consistently — a line that would be extracted as a comment and
// parsed as an annotation must be recognized here too.
static bool loom_check_is_annotation_line(iree_string_view_t trimmed_line) {
  iree_string_view_t comment = loom_check_extract_comment_text(trimmed_line);
  if (iree_string_view_is_empty(comment)) return false;
  return loom_check_looks_like_annotation(comment);
}

// Parses one quoted string from |text|, returning the unquoted contents
// in |*out_substring| and the trimmed remainder (everything after the
// closing quote) in |*out_remainder|. Errors if |text| does not start
// with a quote or the quote is unterminated.
static iree_status_t loom_check_parse_quoted_substring(
    iree_string_view_t text, iree_string_view_t* out_substring,
    iree_string_view_t* out_remainder) {
  if (!iree_string_view_starts_with_char(text, '"')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected quoted string at '%.*s'", (int)text.size,
                            text.data);
  }
  iree_host_size_t close = iree_string_view_find_char(text, '"', 1);
  if (close == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated string in annotation matcher");
  }
  *out_substring = iree_string_view_substr(text, 1, close - 1);
  *out_remainder = iree_string_view_trim(
      iree_string_view_substr(text, close + 1, IREE_HOST_SIZE_MAX));
  return iree_ok_status();
}

// Consumes a sequence of quoted substrings from |text| and appends them
// to the annotation's substring array. |text| must contain only quoted
// strings separated by whitespace, or be empty. Errors on unterminated
// quotes, non-quote text, or overflow of LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS.
static iree_status_t loom_check_parse_substring_list(
    iree_string_view_t text, loom_check_annotation_t* out) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    if (out->message_substring_count >= LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "annotation has more than %d quoted substrings; raise "
          "LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS if this is intentional",
          (int)LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS);
    }
    if (!iree_string_view_starts_with_char(text, '"')) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unexpected text after annotation matcher: '%.*s' "
          "(expected a quoted substring)",
          (int)text.size, text.data);
    }
    iree_string_view_t substring;
    iree_string_view_t remainder;
    IREE_RETURN_IF_ERROR(
        loom_check_parse_quoted_substring(text, &substring, &remainder));
    out->message_substrings[out->message_substring_count++] = substring;
    text = remainder;
  }
  return iree_ok_status();
}

static bool loom_check_is_param_name_start(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

static bool loom_check_is_param_name_char(char c) {
  return loom_check_is_param_name_start(c) || (c >= '0' && c <= '9');
}

static iree_status_t loom_check_parse_param_name(
    iree_string_view_t text, iree_string_view_t* out_name,
    iree_string_view_t* out_remainder) {
  *out_name = iree_string_view_empty();
  *out_remainder = text;
  if (iree_string_view_is_empty(text) ||
      !loom_check_is_param_name_start(text.data[0])) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected diagnostic param name at '%.*s'",
                            (int)text.size, text.data);
  }
  iree_host_size_t length = 1;
  while (length < text.size &&
         loom_check_is_param_name_char(text.data[length])) {
    ++length;
  }
  *out_name = iree_string_view_substr(text, 0, length);
  *out_remainder = iree_string_view_trim(
      iree_string_view_substr(text, length, IREE_HOST_SIZE_MAX));
  return iree_ok_status();
}

// Consumes an optional structured parameter matcher object:
//
//   {field_a="lhs", type_a="i32"}
//
// Values are quoted string comparisons against the diagnostic's rendered
// parameter values. The matcher is structured by parameter name; it is not a
// diagnostic message substring.
static iree_status_t loom_check_parse_param_matchers(
    iree_string_view_t text, loom_check_annotation_t* out,
    iree_string_view_t* out_remainder) {
  text = iree_string_view_trim(text);
  *out_remainder = text;
  if (!iree_string_view_consume_prefix_char(&text, '{')) {
    return iree_ok_status();
  }
  text = iree_string_view_trim(text);
  while (true) {
    if (iree_string_view_consume_prefix_char(&text, '}')) {
      *out_remainder = iree_string_view_trim(text);
      return iree_ok_status();
    }
    if (out->param_match_count >= LOOM_CHECK_MAX_ANNOTATION_PARAM_MATCHES) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "annotation has more than %d diagnostic param matchers; raise "
          "LOOM_CHECK_MAX_ANNOTATION_PARAM_MATCHES if this is intentional",
          (int)LOOM_CHECK_MAX_ANNOTATION_PARAM_MATCHES);
    }

    iree_string_view_t name;
    IREE_RETURN_IF_ERROR(loom_check_parse_param_name(text, &name, &text));
    if (!iree_string_view_consume_prefix_char(&text, '=')) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "diagnostic param matcher '%.*s' missing '='",
                              (int)name.size, name.data);
    }
    text = iree_string_view_trim(text);

    iree_string_view_t value;
    IREE_RETURN_IF_ERROR(
        loom_check_parse_quoted_substring(text, &value, &text));
    out->param_matches[out->param_match_count++] =
        (loom_check_annotation_param_match_t){
            .name = name,
            .value = value,
        };

    if (iree_string_view_consume_prefix_char(&text, ',')) {
      text = iree_string_view_trim(text);
      continue;
    }
    if (iree_string_view_starts_with_char(text, '}')) {
      continue;
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "diagnostic param matcher expected ',' or '}' after value at '%.*s'",
        (int)text.size, text.data);
  }
}

// Parses the matcher part of an annotation: DOMAIN/CODE followed by an optional
// structured param matcher and zero or more quoted substrings, OR just one or
// more quoted substrings, OR a bare DOMAIN. All components are independently
// optional. Multiple substrings are AND-combined against the rendered
// diagnostic message. Multiple parameter matchers are AND-combined against
// structured diagnostic params.
//
// Examples:
//   TYPE/001                         — domain + code only
//   TYPE/001 {field_a="lhs"}          — domain + code + param matcher
//   TYPE/001 "operand 3" "result 0"  — domain + code + two substrings
//   "fragment"                       — substring only (any diagnostic)
//   PARSE                            — domain only (any code, any message)
static iree_status_t loom_check_parse_matcher(iree_string_view_t text,
                                              loom_check_annotation_t* out) {
  out->domain = LOOM_ERROR_DOMAIN_COUNT_;  // Sentinel: match any domain.
  out->code = 0;
  out->message_substring_count = 0;
  out->param_match_count = 0;
  text = iree_string_view_trim(text);

  if (iree_string_view_is_empty(text)) {
    return iree_ok_status();
  }

  // Substring-only matcher: text starts with a quote.
  if (iree_string_view_starts_with_char(text, '"')) {
    return loom_check_parse_substring_list(text, out);
  }

  // Otherwise expect DOMAIN or DOMAIN/CODE, optionally followed by structured
  // param matchers and quoted substrings. Isolate the DOMAIN/CODE token (ends
  // at space or end of string).
  iree_string_view_t domain_code = text;
  iree_string_view_t after_code = iree_string_view_empty();
  iree_host_size_t space = iree_string_view_find_char(text, ' ', 0);
  if (space != IREE_STRING_VIEW_NPOS) {
    domain_code = iree_string_view_substr(text, 0, space);
    after_code = iree_string_view_trim(
        iree_string_view_substr(text, space + 1, IREE_HOST_SIZE_MAX));
  }

  // Split DOMAIN/CODE on '/'. No slash means domain-only.
  iree_string_view_t domain_str;
  iree_string_view_t code_str;
  intptr_t slash =
      iree_string_view_split(domain_code, '/', &domain_str, &code_str);
  if (slash >= 0) {
    if (!loom_error_domain_from_name(domain_str, &out->domain)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown error domain in annotation: '%.*s'",
                              (int)domain_str.size, domain_str.data);
    }
    if (iree_string_view_is_empty(code_str)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty code after '/' in annotation: '%.*s'",
                              (int)domain_code.size, domain_code.data);
    }
    int32_t code_value = 0;
    if (!iree_string_view_atoi_int32_base(code_str, 10, &code_value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "non-numeric code in annotation: '%.*s'",
                              (int)code_str.size, code_str.data);
    }
    if (code_value < 0 || code_value > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "annotation code out of range: %d", code_value);
    }
    out->code = (uint16_t)code_value;
  } else {
    // Bare domain name without a code (e.g., "// ERROR: TYPE").
    if (!loom_error_domain_from_name(domain_code, &out->domain)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown error domain in annotation: '%.*s'",
                              (int)domain_code.size, domain_code.data);
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_check_parse_param_matchers(after_code, out, &after_code));
  return loom_check_parse_substring_list(after_code, out);
}

// Tries to parse an annotation from the text after "// " in a line.
// Returns true via |*out_found| if the text is a valid annotation,
// populating |*out|. Returns false (without error) if the text is an
// ordinary comment. |current_line| is the 1-based line number within
// the input section, used to compute target_line from @offsets.
static iree_status_t loom_check_try_parse_annotation(
    iree_string_view_t comment_text, iree_host_size_t current_line,
    loom_check_annotation_t* out, bool* out_found) {
  *out_found = false;
  memset(out, 0, sizeof(*out));
  out->domain = LOOM_ERROR_DOMAIN_COUNT_;  // Sentinel: match any domain.

  iree_string_view_t text = iree_string_view_trim(comment_text);

  // An annotation starts with an uppercase severity keyword. If we
  // don't find one, this is an ordinary comment — not an error.
  loom_diagnostic_severity_t severity;
  if (!loom_check_parse_severity(&text, &severity)) {
    return iree_ok_status();
  }
  out->severity = severity;

  // The severity keyword may be followed by an @offset that redirects
  // the annotation to a different line. Without an offset, the
  // annotation targets its own line.
  iree_host_size_t target_line = current_line;
  if (iree_string_view_consume_prefix_char(&text, '@')) {
    bool negative = false;
    if (iree_string_view_consume_prefix_char(&text, '+')) {
      negative = false;
    } else if (iree_string_view_consume_prefix_char(&text, '-')) {
      negative = true;
    }
    iree_host_size_t colon = iree_string_view_find_char(text, ':', 0);
    if (colon == IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "annotation with @ offset missing ':'");
    }
    iree_string_view_t offset_str = iree_string_view_substr(text, 0, colon);
    int32_t offset_value = 0;
    if (!iree_string_view_atoi_int32(offset_str, &offset_value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid offset in annotation: '%.*s'",
                              (int)offset_str.size, offset_str.data);
    }
    if (offset_value < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "annotation offset must be non-negative: %d",
                              offset_value);
    }
    if (negative) {
      if ((iree_host_size_t)offset_value >= current_line) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "annotation offset @-%d on line %zu "
                                "would target before line 1",
                                offset_value, current_line);
      }
      target_line = current_line - (iree_host_size_t)offset_value;
    } else {
      target_line = current_line + (iree_host_size_t)offset_value;
    }
    text = iree_string_view_substr(text, colon, IREE_HOST_SIZE_MAX);
  }

  // A colon must follow the severity (and optional offset). Without it,
  // this is a word like "ERROR" appearing in normal text.
  if (!iree_string_view_consume_prefix_char(&text, ':')) {
    return iree_ok_status();
  }
  text = iree_string_view_trim(text);

  out->target_line = target_line;

  IREE_RETURN_IF_ERROR(loom_check_parse_matcher(text, out));

  *out_found = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Annotation extraction
//===----------------------------------------------------------------------===//

// Returns true if |text| contains any lines with diagnostic annotations.
static bool loom_check_text_contains_annotations(iree_string_view_t text) {
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t line = loom_check_consume_line(&text);
    iree_string_view_t comment = loom_check_extract_comment_text(line);
    if (!iree_string_view_is_empty(comment) &&
        loom_check_looks_like_annotation(comment)) {
      return true;
    }
  }
  return false;
}

// Extracts annotations from |input| by scanning standalone comment
// lines. Line numbers are 1-based within |input|. Annotations are
// counted first, then arena-allocated and populated in a second scan.
static iree_status_t loom_check_extract_annotations(
    const char* source_start, iree_string_view_t input,
    iree_arena_allocator_t* arena, loom_check_annotation_t** out_annotations,
    iree_host_size_t* out_annotation_count) {
  *out_annotations = NULL;
  *out_annotation_count = 0;

  // Count annotations so we can allocate the array.
  iree_host_size_t annotation_count = 0;
  iree_host_size_t line_number = 0;
  iree_string_view_t scanner = input;
  while (!iree_string_view_is_empty(scanner)) {
    iree_string_view_t line = loom_check_consume_line(&scanner);
    ++line_number;
    iree_string_view_t comment_text = loom_check_extract_comment_text(line);
    if (iree_string_view_is_empty(comment_text)) continue;
    loom_check_annotation_t annotation;
    bool found = false;
    IREE_RETURN_IF_ERROR(loom_check_try_parse_annotation(
        comment_text, line_number, &annotation, &found));
    if (found) ++annotation_count;
  }

  if (annotation_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, annotation_count, sizeof(loom_check_annotation_t),
      (void**)out_annotations));
  *out_annotation_count = annotation_count;

  // Populate the allocated array with a second scan.
  scanner = input;
  line_number = 0;
  iree_host_size_t annotation_index = 0;
  while (!iree_string_view_is_empty(scanner)) {
    const char* line_start = scanner.data;
    iree_string_view_t line = loom_check_consume_line(&scanner);
    ++line_number;
    iree_string_view_t comment_text = loom_check_extract_comment_text(line);
    if (iree_string_view_is_empty(comment_text)) continue;
    loom_check_annotation_t annotation;
    bool found = false;
    IREE_RETURN_IF_ERROR(loom_check_try_parse_annotation(
        comment_text, line_number, &annotation, &found));
    if (found) {
      annotation.source_range = loom_check_source_range_from_pointers(
          source_start, line_start, line.data + line.size);
      (*out_annotations)[annotation_index++] = annotation;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Case section splitting
//===----------------------------------------------------------------------===//

// Splits the raw case text (everything between case separators) into
// directives, input, and expected sections.
//
// Directives (// RUN:, // REQUIRES:, // XFAIL:) are extracted from the top
// of the case. File-level directives (// TEMPLATE:) are only accepted when
// |allow_file_directives| is true. The header region consists of directives,
// blank lines, and non-annotation comment lines — all consumed. The first
// annotation line (// ERROR/WARNING/REMARK) or non-comment line starts the
// body.
//
// Any line starting with "// RUN:", "// REQUIRES:", or "// XFAIL:" in the
// body is always a mistake (misplaced or malformed directive) and produces an
// error. This catches the silent misconfiguration where a comment before a
// directive pushes it into the body.
//
// Within the body, a // ---- line splits input from expected output;
// without it, expected equals input (round-trip identity).
static iree_status_t loom_check_parse_case_sections(
    const char* source_start, iree_string_view_t case_text,
    loom_check_source_range_t separator_range, loom_check_case_t* out_case,
    bool allow_file_directives, iree_arena_allocator_t* arena) {
  memset(out_case, 0, sizeof(*out_case));
  out_case->mode = LOOM_CHECK_MODE_ROUNDTRIP;

  // Guard against NULL-backed empty views (iree_string_view_empty has
  // data == NULL). Pointer arithmetic on NULL is undefined behavior.
  static const char kEmptySource[] = "";
  if (!case_text.data) case_text.data = kEmptySource;
  const char* case_end = case_text.data + case_text.size;
  out_case->source_range =
      loom_check_source_range_from_view(source_start, case_text);
  out_case->separator_range = separator_range;

  // Scan through the header region. Directives, blank lines, and
  // non-annotation comments are consumed. The first annotation line
  // or IR line starts the body.
  iree_string_view_t scanner = case_text;
  bool found_run = false;
  bool found_requires = false;
  bool body_started = false;
  const char* body_start = case_end;

  while (!iree_string_view_is_empty(scanner)) {
    const char* line_start = scanner.data;
    iree_string_view_t line = loom_check_consume_line(&scanner);
    iree_string_view_t trimmed = iree_string_view_trim(line);

    if (iree_string_view_is_empty(trimmed)) {
      if (!body_started) {
        body_start = loom_check_scanner_end(scanner, case_end);
      }
      continue;
    }

    if (loom_check_is_run_directive(trimmed)) {
      if (body_started) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stray directive '%.*s' in test case body; "
            "directives must appear before all other content",
            (int)trimmed.size, trimmed.data);
      }
      if (found_run) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "multiple RUN directives in one test case");
      }
      found_run = true;
      out_case->run_directive_range = loom_check_source_range_from_pointers(
          source_start, line_start, line.data + line.size);
      iree_string_view_t run_value = trimmed;
      iree_string_view_consume_prefix(&run_value,
                                      iree_make_cstring_view("// RUN: "));
      IREE_RETURN_IF_ERROR(loom_check_parse_run_directive(
          run_value, &out_case->mode, &out_case->output_flags,
          &out_case->pipeline, &out_case->format_target,
          &out_case->emit_target));
      body_start = loom_check_scanner_end(scanner, case_end);
      continue;
    }

    if (loom_check_is_requires_directive(trimmed)) {
      if (body_started) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stray directive '%.*s' in test case body; "
            "directives must appear before all other content",
            (int)trimmed.size, trimmed.data);
      }
      if (found_requires) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "multiple REQUIRES directives in one test case");
      }
      found_requires = true;
      out_case->has_requires_directive = true;
      out_case->requires_directive_range =
          loom_check_source_range_from_pointers(source_start, line_start,
                                                line.data + line.size);
      iree_string_view_t requires_value = trimmed;
      iree_string_view_consume_prefix(&requires_value,
                                      iree_make_cstring_view("// REQUIRES: "));
      IREE_RETURN_IF_ERROR(loom_check_parse_requirement_list(
          requires_value, arena, &out_case->requirements,
          &out_case->requirement_count));
      body_start = loom_check_scanner_end(scanner, case_end);
      continue;
    }

    if (loom_check_is_xfail_directive(trimmed)) {
      if (body_started) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stray directive '%.*s' in test case body; "
            "directives must appear before all other content",
            (int)trimmed.size, trimmed.data);
      }
      out_case->xfail = true;
      out_case->xfail_directive_range = loom_check_source_range_from_pointers(
          source_start, line_start, line.data + line.size);
      iree_string_view_t xfail_value = trimmed;
      iree_string_view_consume_prefix(&xfail_value,
                                      iree_make_cstring_view("// XFAIL: "));
      out_case->xfail_reason = iree_string_view_trim(xfail_value);
      body_start = loom_check_scanner_end(scanner, case_end);
      continue;
    }

    if (loom_check_is_template_directive(trimmed)) {
      if (!allow_file_directives || body_started) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "TEMPLATE directive is file-level metadata and must appear in the "
            "preamble before the first // ==== separator");
      }
      body_start = loom_check_scanner_end(scanner, case_end);
      continue;
    }

    if (loom_check_looks_like_case(trimmed) ||
        iree_string_view_starts_with(trimmed,
                                     iree_make_cstring_view("//CASE:"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "CASE directives are not supported; use function symbols as case "
          "names");
    }

    // Catch malformed directives. These are lines that look like they
    // are trying to be directives but don't match the exact format.
    // Without this check, they fall through to the generic header
    // comment consumer and silently change the test's behavior.
    //
    // Detected patterns:
    //   "// RUN:verify"   — missing space after colon
    //   "//RUN: verify"   — missing space after //
    //   "// REQUIRES:foo" — missing space after colon
    //   "//REQUIRES: foo" — missing space after //
    //   "// XFAIL:reason" — missing space after colon
    //   "//XFAIL: reason" — missing space after //
    //   "// TEMPLATE:foo" — missing space after colon
    //   "//TEMPLATE: foo" — missing space after //
    if (loom_check_looks_like_run(trimmed) ||
        loom_check_looks_like_requires(trimmed) ||
        loom_check_looks_like_xfail(trimmed) ||
        loom_check_looks_like_template(trimmed) ||
        iree_string_view_starts_with(trimmed,
                                     iree_make_cstring_view("//RUN:")) ||
        iree_string_view_starts_with(trimmed,
                                     iree_make_cstring_view("//REQUIRES:")) ||
        iree_string_view_starts_with(trimmed,
                                     iree_make_cstring_view("//XFAIL:")) ||
        iree_string_view_starts_with(trimmed,
                                     iree_make_cstring_view("//TEMPLATE:"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "malformed directive '%.*s'; expected '// RUN: <mode>', "
          "'// REQUIRES: <name>', '// XFAIL: <reason>', or "
          "'// TEMPLATE: <path>'",
          (int)trimmed.size, trimmed.data);
    }

    // Non-annotation comment lines in the header are consumed. This
    // allows descriptive comments between or before directives:
    //   // Tests edge case for DCE
    //   // RUN: pass dce
    //   func.def @f() {...}
    // Expected separators (// ----) must not be consumed here — they
    // are structural and must survive to the body splitting pass.
    if (!body_started &&
        iree_string_view_starts_with(trimmed, iree_make_cstring_view("//")) &&
        !loom_check_is_annotation_line(trimmed) &&
        !loom_check_is_expected_separator(trimmed)) {
      body_start = loom_check_scanner_end(scanner, case_end);
      continue;
    }

    // Annotation line or IR content — body starts here.
    if (!body_started) {
      body_started = true;
      body_start = line_start;
    }
  }

  // The body contains the IR input and optionally a // ---- separator
  // followed by expected output.
  iree_string_view_t body = {
      .data = body_start,
      .size = (iree_host_size_t)(case_end - body_start),
  };

  // Scan the body for a // ---- separator. Everything before it is
  // input; everything after is expected. Without a separator, input
  // and expected are the same text (round-trip identity).
  iree_string_view_t input = body;
  iree_string_view_t expected = body;
  out_case->has_expected_section = false;

  scanner = body;
  while (!iree_string_view_is_empty(scanner)) {
    const char* line_start = scanner.data;
    iree_string_view_t line = loom_check_consume_line(&scanner);

    if (loom_check_is_expected_separator(line)) {
      out_case->expected_separator_range =
          loom_check_source_range_from_pointers(source_start, line_start,
                                                line.data + line.size);
      iree_host_size_t input_length =
          (iree_host_size_t)(line_start - body.data);
      input = iree_string_view_substr(body, 0, input_length);
      iree_host_size_t expected_offset =
          (iree_host_size_t)(loom_check_scanner_end(scanner,
                                                    body.data + body.size) -
                             body.data);
      expected =
          iree_string_view_substr(body, expected_offset, IREE_HOST_SIZE_MAX);
      out_case->expected_range =
          loom_check_source_range_from_view(source_start, expected);
      out_case->has_expected_section = true;
      break;
    }
  }

  out_case->input = input;
  out_case->expected = expected;
  out_case->input_range =
      loom_check_source_range_from_view(source_start, input);

  // Annotations in the expected section are always a mistake — they
  // belong in the input section where line numbers are meaningful.
  if (out_case->has_expected_section &&
      loom_check_text_contains_annotations(expected)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "annotation in expected output section (below // ----); "
        "annotations must appear in the input section");
  }

  // Record whether this case had its own RUN directive. Cases without
  // one will inherit the file-level default after all cases are parsed.
  out_case->has_run_directive = found_run;

  // Extract diagnostic annotations from comments in the input.
  IREE_RETURN_IF_ERROR(loom_check_extract_annotations(
      source_start, input, arena, &out_case->annotations,
      &out_case->annotation_count));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// File parsing
//===----------------------------------------------------------------------===//

// Boundary of a single test case within the source buffer.
typedef struct loom_check_case_boundary_t {
  const char* start;
  iree_host_size_t length;
  loom_check_source_range_t separator_range;
} loom_check_case_boundary_t;

static iree_status_t loom_check_parse_template_directive_from_preamble(
    const char* source_start, iree_string_view_t preamble_text,
    loom_check_file_t* file) {
  iree_string_view_t scanner = preamble_text;
  while (!iree_string_view_is_empty(scanner)) {
    const char* line_start = scanner.data;
    iree_string_view_t line = loom_check_consume_line(&scanner);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    if (!loom_check_is_template_directive(trimmed)) {
      continue;
    }
    if (file->has_template_directive) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "multiple TEMPLATE directives in one test file");
    }
    file->has_template_directive = true;
    file->template_directive_range = loom_check_source_range_from_pointers(
        source_start, line_start, line.data + line.size);
    iree_string_view_t template_value = trimmed;
    iree_string_view_consume_prefix(&template_value,
                                    iree_make_cstring_view("// TEMPLATE: "));
    IREE_RETURN_IF_ERROR(
        loom_check_parse_template_path(template_value, &file->template_path));
  }
  return iree_ok_status();
}

static bool loom_check_line_is_file_preamble_comment(
    iree_string_view_t trimmed) {
  return iree_string_view_starts_with(trimmed, iree_make_cstring_view("//")) &&
         !loom_check_is_annotation_line(trimmed) &&
         !loom_check_is_expected_separator(trimmed) &&
         !loom_check_is_case_separator(trimmed);
}

// Finds the leading file preamble used by template-synchronized files.
//
// Ordinary .loom-test files treat leading RUN/REQUIRES directives as part of
// the first case. A TEMPLATE directive is different: it is file metadata for
// update tooling, so the leading directive/comment block containing it is a
// real file preamble. The preamble ends at the first blank line after the
// TEMPLATE block, the first case separator, or the first IR/annotation line.
static void loom_check_find_template_preamble_prefix(
    iree_string_view_t source, loom_check_source_range_t* out_range) {
  *out_range = loom_check_source_range_empty();
  if (iree_string_view_is_empty(source)) {
    return;
  }

  const char* source_end = source.data + source.size;
  const char* preamble_end = source.data;
  bool saw_template = false;
  iree_string_view_t scanner = source;
  while (!iree_string_view_is_empty(scanner)) {
    iree_string_view_t line = loom_check_consume_line(&scanner);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    const char* next_line = loom_check_scanner_end(scanner, source_end);

    if (iree_string_view_is_empty(trimmed)) {
      preamble_end = next_line;
      if (saw_template) break;
      continue;
    }

    if (loom_check_is_case_separator(trimmed)) {
      break;
    }

    if (loom_check_is_template_directive(trimmed)) {
      saw_template = true;
      preamble_end = next_line;
      continue;
    }

    if (loom_check_is_run_directive(trimmed) ||
        loom_check_is_requires_directive(trimmed) ||
        loom_check_line_is_file_preamble_comment(trimmed)) {
      preamble_end = next_line;
      continue;
    }

    break;
  }

  if (saw_template) {
    *out_range = (loom_check_source_range_t){
        .start_byte = 0,
        .end_byte = (iree_host_size_t)(preamble_end - source.data),
    };
  }
}

iree_status_t loom_check_parse(iree_string_view_t source,
                               iree_arena_allocator_t* arena,
                               loom_check_file_t* out_file) {
  memset(out_file, 0, sizeof(*out_file));

  // Normalize NULL-backed empty views. iree_string_view_empty() has
  // data == NULL, and pointer arithmetic on NULL is undefined behavior.
  static const char kEmptySource[] = "";
  if (!source.data) source.data = kEmptySource;

  out_file->default_mode = LOOM_CHECK_MODE_ROUNDTRIP;
  out_file->default_output_flags = 0;
  out_file->default_pipeline = iree_string_view_empty();
  out_file->default_format_target = iree_string_view_empty();
  out_file->default_emit_target = iree_string_view_empty();
  out_file->default_requirements = NULL;
  out_file->default_requirement_count = 0;

  loom_check_source_range_t template_preamble_range =
      loom_check_source_range_empty();
  loom_check_find_template_preamble_prefix(source, &template_preamble_range);
  if (!loom_check_source_range_is_empty(template_preamble_range)) {
    iree_string_view_t preamble_text = iree_string_view_substr(
        source, template_preamble_range.start_byte,
        template_preamble_range.end_byte - template_preamble_range.start_byte);
    IREE_RETURN_IF_ERROR(loom_check_parse_template_directive_from_preamble(
        source.data, preamble_text, out_file));

    loom_check_case_t preamble_case;
    IREE_RETURN_IF_ERROR(loom_check_parse_case_sections(
        source.data, preamble_text, loom_check_source_range_empty(),
        &preamble_case, /*allow_file_directives=*/true, arena));
    if (!iree_string_view_is_empty(
            iree_string_view_trim(preamble_case.input))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "TEMPLATE directive must be in a pure file preamble");
    }
    if (preamble_case.has_run_directive) {
      out_file->default_mode = preamble_case.mode;
      out_file->default_output_flags = preamble_case.output_flags;
      out_file->default_pipeline = preamble_case.pipeline;
      out_file->default_format_target = preamble_case.format_target;
      out_file->default_emit_target = preamble_case.emit_target;
    }
    if (preamble_case.has_requires_directive) {
      out_file->default_requirements = preamble_case.requirements;
      out_file->default_requirement_count = preamble_case.requirement_count;
    }
  }

  iree_string_view_t case_source = iree_string_view_substr(
      source, template_preamble_range.end_byte, IREE_HOST_SIZE_MAX);
  if (!loom_check_source_range_is_empty(template_preamble_range) &&
      iree_string_view_is_empty(iree_string_view_trim(case_source))) {
    out_file->cases = NULL;
    out_file->case_count = 0;
    return iree_ok_status();
  }

  // Count case separators to determine how many cases the file contains.
  // The first case starts at byte 0 (no leading separator needed), so
  // N separators means N+1 cases.
  iree_host_size_t case_count = 1;
  iree_string_view_t scanner = case_source;
  while (!iree_string_view_is_empty(scanner)) {
    iree_string_view_t line = loom_check_consume_line(&scanner);
    if (loom_check_is_case_separator(line)) {
      ++case_count;
    }
  }

  // Collect the byte range for each case.
  loom_check_case_boundary_t* boundaries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, case_count, sizeof(loom_check_case_boundary_t),
      (void**)&boundaries));

  boundaries[0].start = case_source.data;
  boundaries[0].separator_range = loom_check_source_range_empty();
  iree_host_size_t case_index = 0;
  const char* source_end = source.data + source.size;

  scanner = case_source;
  while (!iree_string_view_is_empty(scanner)) {
    const char* line_start = scanner.data;
    iree_string_view_t line = loom_check_consume_line(&scanner);

    if (loom_check_is_case_separator(line)) {
      boundaries[case_index].length =
          (iree_host_size_t)(line_start - boundaries[case_index].start);

      ++case_index;
      boundaries[case_index].separator_range =
          loom_check_source_range_from_pointers(source.data, line_start,
                                                line.data + line.size);
      boundaries[case_index].start =
          loom_check_scanner_end(scanner, source_end);
    }
  }
  boundaries[case_index].length =
      (iree_host_size_t)(source_end - boundaries[case_index].start);

  // The first separator must come after the first real case. A leading
  // separator creates an empty case 0 and hides the actual file default.
  // Template-synchronized files use a real file preamble without a separator,
  // handled above by slicing |case_source| past that preamble.
  if (case_count > 1) {
    iree_string_view_t first_boundary_text = {
        .data = boundaries[0].start,
        .size = boundaries[0].length,
    };
    loom_check_case_t first_boundary_case;
    IREE_RETURN_IF_ERROR(loom_check_parse_case_sections(
        source.data, first_boundary_text, boundaries[0].separator_range,
        &first_boundary_case, /*allow_file_directives=*/false, arena));

    if (iree_string_view_is_empty(
            iree_string_view_trim(first_boundary_case.input))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "the first // ==== separator must appear after the first case body");
    }
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, case_count, sizeof(loom_check_case_t), (void**)&out_file->cases));
  out_file->case_count = case_count;

  // Parse each case's directives, input/expected split, and annotations.
  for (iree_host_size_t i = 0; i < case_count; ++i) {
    iree_host_size_t boundary_index = i;
    iree_string_view_t case_text = {
        .data = boundaries[boundary_index].start,
        .size = boundaries[boundary_index].length,
    };
    IREE_RETURN_IF_ERROR(loom_check_parse_case_sections(
        source.data, case_text, boundaries[boundary_index].separator_range,
        &out_file->cases[i], /*allow_file_directives=*/false, arena));
  }

  // The first case's RUN directive becomes the file default for inheritance by
  // later cases.
  if (case_count > 0 && out_file->cases[0].has_run_directive) {
    out_file->default_mode = out_file->cases[0].mode;
    out_file->default_output_flags = out_file->cases[0].output_flags;
    out_file->default_pipeline = out_file->cases[0].pipeline;
    out_file->default_format_target = out_file->cases[0].format_target;
    out_file->default_emit_target = out_file->cases[0].emit_target;
  }
  if (case_count > 0 && out_file->cases[0].has_requires_directive) {
    out_file->default_requirements = out_file->cases[0].requirements;
    out_file->default_requirement_count = out_file->cases[0].requirement_count;
  }

  // Apply RUN inheritance: cases without their own RUN directive inherit the
  // file-level default.
  for (iree_host_size_t i = 0; i < case_count; ++i) {
    if (!out_file->cases[i].has_run_directive) {
      out_file->cases[i].mode = out_file->default_mode;
      out_file->cases[i].output_flags = out_file->default_output_flags;
      out_file->cases[i].pipeline = out_file->default_pipeline;
      out_file->cases[i].format_target = out_file->default_format_target;
      out_file->cases[i].emit_target = out_file->default_emit_target;
    }
  }

  // Apply REQUIRES inheritance: every case receives the file-level default
  // requirements, plus any case-local requirements it declared.
  if (out_file->default_requirement_count > 0) {
    for (iree_host_size_t i = 0; i < case_count; ++i) {
      loom_check_case_t* test_case = &out_file->cases[i];
      if (!test_case->has_requires_directive) {
        test_case->requirements = out_file->default_requirements;
        test_case->requirement_count = out_file->default_requirement_count;
        continue;
      }
      iree_string_view_t* combined_requirements = NULL;
      iree_host_size_t combined_requirement_count = 0;
      IREE_RETURN_IF_ERROR(loom_check_combine_requirement_lists(
          out_file->default_requirements, out_file->default_requirement_count,
          test_case->requirements, test_case->requirement_count, arena,
          &combined_requirements, &combined_requirement_count));
      test_case->requirements = combined_requirements;
      test_case->requirement_count = combined_requirement_count;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Comment stripping
//===----------------------------------------------------------------------===//

iree_status_t loom_check_strip_comments(iree_string_view_t input,
                                        iree_string_builder_t* output) {
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_consume_line(&remaining);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    // Standalone comment lines become blank lines to preserve line count.
    if (iree_string_view_starts_with(trimmed, iree_make_cstring_view("//"))) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
      continue;
    }
    // Non-comment lines (including lines with trailing comments) are
    // preserved as-is. Only standalone comment lines are stripped.
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}
