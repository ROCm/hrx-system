// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "emit.h"

#include <string.h>

#include "context.h"
#include "diagnostic.h"
#include "iree/base/internal/arena.h"
#include "loom/error/json_sink.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/format.h"
#include "loomc/compile_report.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
#include "target.h"
#include "workspace.h"

typedef struct loomc_descriptor_prefix_t {
  // Structure type identifying the descriptor.
  loomc_structure_type_t type;

  // Size of the descriptor in bytes.
  loomc_host_size_t structure_size;

  // Next descriptor in the option extension chain.
  const void* next;
} loomc_descriptor_prefix_t;

static loomc_status_t loomc_emit_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_result_fail_message(
    loomc_result_t* result, loomc_string_view_t code,
    loomc_string_view_t message) {
  loomc_diagnostic_t diagnostic = {
      .severity = LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      .code = code,
      .message = message,
  };
  LOOMC_RETURN_IF_ERROR(loomc_result_add_diagnostic(result, &diagnostic));
  loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_result_fail_cstring(loomc_result_t* result,
                                                     const char* code,
                                                     const char* message) {
  return loomc_emit_result_fail_message(result, loomc_make_cstring_view(code),
                                        loomc_make_cstring_view(message));
}

static loomc_status_t loomc_emit_result_fail_unknown_option(
    loomc_result_t* result, loomc_string_view_t key,
    loomc_allocator_t allocator) {
  iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  loomc_status_t status = loomc_status_from_iree(
      iree_string_builder_append_cstring(&builder, "unknown emit option key"));
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(key)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, " '"));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(key)) {
    status = loomc_status_from_iree(iree_string_builder_append_string(
        &builder, iree_string_view_from_loomc(key)));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(key)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, "'"));
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_result_fail_message(
        result, loomc_make_cstring_view("EMIT/OPTION"),
        loomc_string_view_from_iree(iree_string_builder_view(&builder)));
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}

static loomc_status_t loomc_emit_validate_options(
    const loomc_emit_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "emit options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "emit options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_emit_validate_string_view(options->artifact_format));
  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_string_view(options->identifier));
  const loomc_emit_artifact_flags_t known_artifact_flags =
      LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY |
      LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING;
  if ((options->artifact_flags & ~known_artifact_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "emit options contain unknown artifact flags");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_validate_option_dict(
    const loomc_option_dict_t* dict) {
  if (dict->type != LOOMC_STRUCTURE_TYPE_OPTION_DICT) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "option dictionary has an unknown structure type");
  }
  if (dict->structure_size != 0 && dict->structure_size < sizeof(*dict)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "option dictionary structure_size is too small");
  }
  if (dict->entry_count != 0 && dict->entries == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "option dictionary entry_count is non-zero but entries is NULL");
  }
  for (loomc_host_size_t i = 0; i < dict->entry_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_emit_validate_string_view(dict->entries[i].key));
    LOOMC_RETURN_IF_ERROR(
        loomc_emit_validate_string_view(dict->entries[i].value));
  }
  return loomc_ok_status();
}

static bool loomc_emit_artifact_manifest_mode_is_valid(
    loomc_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOMC_ARTIFACT_MANIFEST_MODE_NONE:
    case LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY:
    case LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS:
    case LOOMC_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return true;
    default:
      return false;
  }
}

static loomc_status_t loomc_emit_validate_artifact_manifest_options(
    const loomc_artifact_manifest_options_t* options) {
  if (options->type != LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "artifact manifest options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "artifact manifest options structure_size is too small");
  }
  if (!loomc_emit_artifact_manifest_mode_is_valid(options->mode)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact manifest mode is invalid");
  }
  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_string_view(options->identifier));
  if (options->mode == LOOMC_ARTIFACT_MANIFEST_MODE_NONE &&
      !loomc_string_view_is_empty(options->identifier)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "artifact manifest identifier requires a non-NONE manifest mode");
  }
  return loomc_ok_status();
}

static bool loomc_emit_compile_report_mode_is_valid(
    loomc_compile_report_mode_t mode) {
  switch (mode) {
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      return true;
    default:
      return false;
  }
}

static bool loomc_emit_compile_report_format_is_valid(
    loomc_compile_report_format_t format) {
  switch (format) {
    case LOOMC_COMPILE_REPORT_FORMAT_JSON:
    case LOOMC_COMPILE_REPORT_FORMAT_TEXT:
      return true;
    default:
      return false;
  }
}

static loomc_status_t loomc_emit_validate_compile_report_options(
    const loomc_compile_report_options_t* options) {
  if (options->type != LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report options structure_size is too small");
  }
  if (!loomc_emit_compile_report_mode_is_valid(options->mode)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile report mode is invalid");
  }
  if (!loomc_emit_compile_report_format_is_valid(options->format)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile report format is invalid");
  }
  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_string_view(options->identifier));
  if (options->mode == LOOMC_COMPILE_REPORT_MODE_NONE &&
      !loomc_string_view_is_empty(options->identifier)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report identifier requires a non-NONE report mode");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_validate_unknown_descriptor(
    const loomc_descriptor_prefix_t* prefix) {
  if (prefix->structure_size != 0 && prefix->structure_size < sizeof(*prefix)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "option extension structure_size is too small");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_apply_option_entry(
    const loomc_option_entry_t* entry, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_emit_resolved_options_t* options) {
  if (loomc_string_view_equal(
          entry->key,
          loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER))) {
    options->identifier = entry->value;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(
          entry->key, loomc_make_cstring_view(
                          LOOMC_EMIT_OPTION_KEY_ARTIFACT_MANIFEST_MODE))) {
    return loomc_artifact_manifest_mode_parse(entry->value,
                                              &options->artifact_manifest_mode);
  }
  if (loomc_string_view_equal(
          entry->key,
          loomc_make_cstring_view(
              LOOMC_EMIT_OPTION_KEY_ARTIFACT_MANIFEST_IDENTIFIER))) {
    options->artifact_manifest_identifier = entry->value;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(
          entry->key,
          loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_COMPILE_REPORT_MODE))) {
    return loomc_compile_report_mode_parse(entry->value,
                                           &options->compile_report_mode);
  }
  if (loomc_string_view_equal(
          entry->key, loomc_make_cstring_view(
                          LOOMC_EMIT_OPTION_KEY_COMPILE_REPORT_IDENTIFIER))) {
    options->compile_report_identifier = entry->value;
    return loomc_ok_status();
  }
  return loomc_emit_result_fail_unknown_option(result, entry->key, allocator);
}

static loomc_status_t loomc_emit_apply_option_dict(
    const loomc_option_dict_t* dict, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_emit_resolved_options_t* options) {
  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_option_dict(dict));
  loomc_status_t status = loomc_ok_status();
  for (loomc_host_size_t i = 0;
       i < dict->entry_count && loomc_status_is_ok(status); ++i) {
    status = loomc_emit_apply_option_entry(&dict->entries[i], result, allocator,
                                           options);
  }
  return status;
}

static loomc_status_t loomc_emit_resolve_options(
    const loomc_emit_options_t* options, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_emit_resolved_options_t* out_options) {
  *out_options = (loomc_emit_resolved_options_t){
      .artifact_format =
          options ? options->artifact_format : loomc_string_view_empty(),
      .identifier = options ? options->identifier : loomc_string_view_empty(),
      .artifact_flags = options ? options->artifact_flags : 0,
      .option_chain = options ? options->next : NULL,
  };

  LOOMC_RETURN_IF_ERROR(loomc_emit_validate_options(options));

  const void* next = options ? options->next : NULL;
  while (next != NULL) {
    const loomc_descriptor_prefix_t* prefix =
        (const loomc_descriptor_prefix_t*)next;
    switch (prefix->type) {
      case LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS:
        return loomc_make_status(
            LOOMC_STATUS_UNIMPLEMENTED,
            "target specialization options are not supported during emission");
      case LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS: {
        const loomc_artifact_manifest_options_t* manifest_options =
            (const loomc_artifact_manifest_options_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_emit_validate_artifact_manifest_options(manifest_options));
        out_options->artifact_manifest_mode = manifest_options->mode;
        out_options->artifact_manifest_identifier =
            manifest_options->identifier;
        next = manifest_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS: {
        const loomc_compile_report_options_t* report_options =
            (const loomc_compile_report_options_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_emit_validate_compile_report_options(report_options));
        out_options->compile_report_mode = report_options->mode;
        out_options->compile_report_format = report_options->format;
        out_options->compile_report_identifier = report_options->identifier;
        next = report_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_OPTION_DICT: {
        const loomc_option_dict_t* dict = (const loomc_option_dict_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_emit_apply_option_dict(dict, result, allocator, out_options));
        next = dict->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_NONE:
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "emit option extension is missing a structure type");
      default:
        LOOMC_RETURN_IF_ERROR(loomc_emit_validate_unknown_descriptor(prefix));
        next = prefix->next;
        break;
    }
  }

  if (out_options->artifact_manifest_mode ==
          LOOMC_ARTIFACT_MANIFEST_MODE_NONE &&
      !loomc_string_view_is_empty(out_options->artifact_manifest_identifier)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "artifact manifest identifier requires a non-NONE manifest mode");
  }
  if (out_options->compile_report_mode == LOOMC_COMPILE_REPORT_MODE_NONE &&
      !loomc_string_view_is_empty(out_options->compile_report_identifier)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report identifier requires a non-NONE report mode");
  }
  return loomc_ok_status();
}

static loom_target_artifact_manifest_mode_t loomc_emit_target_manifest_mode(
    loomc_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
    case LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS;
    case LOOMC_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS;
    case LOOMC_ARTIFACT_MANIFEST_MODE_NONE:
    default:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE;
  }
}

static loom_target_compile_report_format_mode_t
loomc_emit_target_compile_report_mode(loomc_compile_report_mode_t mode) {
  switch (mode) {
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
      return LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      return LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    default:
      return LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
  }
}

static loomc_status_t loomc_emit_result_fail_format_message(
    loomc_result_t* result, const char* message, loomc_string_view_t format,
    loomc_allocator_t allocator) {
  iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  loomc_status_t status = loomc_status_from_iree(
      iree_string_builder_append_cstring(&builder, message));
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(format)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, " '"));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(format)) {
    status = loomc_status_from_iree(iree_string_builder_append_string(
        &builder, iree_string_view_from_loomc(format)));
  }
  if (loomc_status_is_ok(status) && !loomc_string_view_is_empty(format)) {
    status = loomc_status_from_iree(
        iree_string_builder_append_cstring(&builder, "'"));
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_result_fail_message(
        result, loomc_make_cstring_view("EMIT/TARGET"),
        loomc_string_view_from_iree(iree_string_builder_view(&builder)));
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}

static loomc_status_t loomc_emit_select_emitter(
    const loom_target_environment_t* target_environment,
    loomc_string_view_t artifact_format, loomc_result_t* result,
    loomc_allocator_t allocator, const loom_target_emitter_t** out_emitter) {
  *out_emitter = loom_target_environment_lookup_emitter(
      target_environment, iree_string_view_from_loomc(artifact_format));
  if (*out_emitter == NULL && loomc_string_view_is_empty(artifact_format)) {
    return loomc_emit_result_fail_cstring(
        result, "EMIT/TARGET",
        "artifact_format is required unless exactly one emitter is linked");
  }
  if (*out_emitter == NULL) {
    return loomc_emit_result_fail_format_message(
        result, "no linked emitter supports artifact format", artifact_format,
        allocator);
  }
  return loomc_ok_status();
}

static iree_status_t loomc_emit_capture_compile_report_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_emit_transaction_t* transaction = (loomc_emit_transaction_t*)user_data;
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(loom_json_value_list_begin_value(
      &transaction->compile_report_diagnostics, &stream));
  const loomc_diagnostic_type_printer_t type_printer = {
      .module = diagnostic->module,
      .text_print_options = transaction->diagnostic_text_print_options,
  };
  const loom_type_formatter_t type_formatter =
      loomc_diagnostic_type_printer_formatter(diagnostic->module ? &type_printer
                                                                 : NULL);
  return loom_diagnostic_json_write_object(&stream, diagnostic, type_formatter);
}

static loomc_status_t loomc_emit_sidecar_artifact_metadata(
    loom_target_emit_sidecar_artifact_kind_t kind,
    loomc_artifact_kind_t* out_kind, loomc_string_view_t* out_format) {
  *out_kind = LOOMC_ARTIFACT_KIND_REPORT;
  *out_format = loomc_string_view_empty();
  switch (kind) {
    case LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST:
      *out_kind = LOOMC_ARTIFACT_KIND_REPORT;
      *out_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON);
      return loomc_ok_status();
    default:
      return loomc_make_status(LOOMC_STATUS_INTERNAL,
                               "emitter returned an unknown sidecar kind");
  }
}

static loomc_string_view_t loomc_emit_identifier(
    const loomc_emit_resolved_options_t* options,
    const loom_target_emitter_t* emitter) {
  if (!loomc_string_view_is_empty(options->identifier)) {
    return options->identifier;
  }
  return loomc_string_view_from_iree(emitter->default_identifier);
}

static loomc_status_t loomc_emit_make_manifest_identifier(
    const loomc_emit_resolved_options_t* options,
    const loom_target_emitter_t* emitter, loomc_allocator_t allocator,
    loomc_string_view_t* out_identifier) {
  *out_identifier = loomc_string_view_empty();
  if (!loomc_string_view_is_empty(options->artifact_manifest_identifier)) {
    return loomc_string_view_clone(options->artifact_manifest_identifier,
                                   allocator, out_identifier);
  }
  const loomc_string_view_t primary_identifier =
      loomc_emit_identifier(options, emitter);
  const loomc_string_view_t suffix = loomc_make_cstring_view(".manifest.json");
  const loomc_host_size_t identifier_length =
      primary_identifier.size + suffix.size;
  char* identifier = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, identifier_length, (void**)&identifier));
  memcpy(identifier, primary_identifier.data, primary_identifier.size);
  memcpy(identifier + primary_identifier.size, suffix.data, suffix.size);
  *out_identifier = loomc_make_string_view(identifier, identifier_length);
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_make_target_listing_identifier(
    const loomc_emit_resolved_options_t* options,
    const loom_target_emitter_t* emitter, loomc_allocator_t allocator,
    loomc_string_view_t* out_identifier) {
  *out_identifier = loomc_string_view_empty();
  const loomc_string_view_t primary_identifier =
      loomc_emit_identifier(options, emitter);
  const loomc_string_view_t suffix = loomc_make_cstring_view(".listing");
  const loomc_host_size_t identifier_length =
      primary_identifier.size + suffix.size;
  char* identifier = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, identifier_length, (void**)&identifier));
  memcpy(identifier, primary_identifier.data, primary_identifier.size);
  memcpy(identifier + primary_identifier.size, suffix.data, suffix.size);
  *out_identifier = loomc_make_string_view(identifier, identifier_length);
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_make_compile_report_identifier(
    const loomc_emit_resolved_options_t* options,
    const loom_target_emitter_t* emitter, loomc_allocator_t allocator,
    loomc_string_view_t* out_identifier) {
  *out_identifier = loomc_string_view_empty();
  if (!loomc_string_view_is_empty(options->compile_report_identifier)) {
    return loomc_string_view_clone(options->compile_report_identifier,
                                   allocator, out_identifier);
  }
  loomc_string_view_t primary_identifier = options->identifier;
  if (loomc_string_view_is_empty(primary_identifier) && emitter != NULL) {
    primary_identifier =
        loomc_string_view_from_iree(emitter->default_identifier);
  }
  if (loomc_string_view_is_empty(primary_identifier)) {
    return loomc_ok_status();
  }
  const loomc_string_view_t suffix =
      options->compile_report_format == LOOMC_COMPILE_REPORT_FORMAT_TEXT
          ? loomc_make_cstring_view(".compile-report.txt")
          : loomc_make_cstring_view(".compile-report.json");
  const loomc_host_size_t identifier_length =
      primary_identifier.size + suffix.size;
  char* identifier = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, identifier_length, (void**)&identifier));
  memcpy(identifier, primary_identifier.data, primary_identifier.size);
  memcpy(identifier + primary_identifier.size, suffix.data, suffix.size);
  *out_identifier = loomc_make_string_view(identifier, identifier_length);
  return loomc_ok_status();
}

static loomc_status_t loomc_emit_add_compile_report_artifact(
    loomc_result_t* result, const loomc_emit_resolved_options_t* options,
    const loomc_string_view_t identifier,
    const loom_target_compile_report_t* report,
    iree_string_view_t diagnostic_json_objects) {
  loomc_allocator_t allocator = loomc_result_allocator(result);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  const loom_target_compile_report_format_options_t format_options = {
      .mode =
          loomc_emit_target_compile_report_mode(options->compile_report_mode),
      .diagnostics =
          {
              .json_objects = diagnostic_json_objects,
              .count = loomc_result_diagnostic_count(result),
          },
  };
  loomc_status_t status = loomc_ok_status();
  if (options->compile_report_format == LOOMC_COMPILE_REPORT_FORMAT_TEXT) {
    status = loomc_status_from_iree(loom_target_compile_report_format_text(
        report, &format_options, &builder));
  } else {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    status = loomc_status_from_iree(loom_target_compile_report_format_json(
        report, &format_options, &stream));
  }

  char* report_storage = NULL;
  iree_host_size_t report_length = 0;
  if (loomc_status_is_ok(status)) {
    report_length = iree_string_builder_size(&builder);
    report_storage = iree_string_builder_take_storage(&builder);
  }
  if (loomc_status_is_ok(status)) {
    const loomc_string_view_t artifact_format =
        options->compile_report_format == LOOMC_COMPILE_REPORT_FORMAT_TEXT
            ? loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT)
            : loomc_make_cstring_view(
                  LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_REPORT, artifact_format, identifier,
        loomc_make_byte_span(report_storage, report_length));
  }
  if (loomc_status_is_ok(status)) {
    report_storage = NULL;
  }
  loomc_allocator_free(allocator, report_storage);
  iree_string_builder_deinitialize(&builder);
  return status;
}

static loomc_status_t loomc_emit_add_byte_sequence_artifact(
    loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier,
    iree_byte_sequence_t* contents) {
  const loomc_artifact_t artifact = {
      .kind = kind,
      .format = format,
      .identifier = identifier,
      .contents = loomc_byte_sequence_from_iree(contents),
  };
  return loomc_result_add_artifact(result, &artifact);
}

static loomc_status_t loomc_emit_add_artifact(
    loomc_result_t* result, const loomc_emit_resolved_options_t* options,
    const loom_target_emitter_t* emitter,
    loom_target_emit_artifact_t* target_artifact) {
  if (target_artifact->contents == NULL) {
    return loomc_make_status(LOOMC_STATUS_INTERNAL,
                             "emitter returned no artifact contents");
  }
  if (target_artifact->sidecar_count != 0 &&
      target_artifact->sidecars == NULL) {
    return loomc_make_status(LOOMC_STATUS_INTERNAL,
                             "emitter returned sidecar count with no data");
  }
  if (target_artifact->target_listing_contents != NULL &&
      iree_string_view_is_empty(target_artifact->target_listing_format)) {
    return loomc_make_status(
        LOOMC_STATUS_INTERNAL,
        "emitter returned target listing contents without a format");
  }
  if (options->artifact_manifest_mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE &&
      target_artifact->sidecar_count == 0) {
    return loomc_emit_result_fail_cstring(
        result, "EMIT/TARGET",
        "selected emitter did not produce an artifact manifest");
  }
  if (target_artifact->target_artifact_format !=
      LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    if (target_artifact->target_artifact_format !=
        emitter->target_artifact_format) {
      return loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "emitter returned an unexpected target artifact format");
    }
  }

  loomc_status_t status = loomc_emit_add_byte_sequence_artifact(
      result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
      loomc_string_view_from_iree(emitter->public_artifact_format),
      loomc_emit_identifier(options, emitter), target_artifact->contents);
  loomc_string_view_t target_listing_identifier = loomc_string_view_empty();
  if (loomc_status_is_ok(status) &&
      target_artifact->target_listing_contents != NULL) {
    status = loomc_emit_make_target_listing_identifier(
        options, emitter, loomc_result_allocator(result),
        &target_listing_identifier);
  }
  if (loomc_status_is_ok(status) &&
      target_artifact->target_listing_contents != NULL) {
    status = loomc_emit_add_byte_sequence_artifact(
        result, LOOMC_ARTIFACT_KIND_TEXT,
        loomc_string_view_from_iree(target_artifact->target_listing_format),
        target_listing_identifier, target_artifact->target_listing_contents);
  }
  loomc_allocator_free(loomc_result_allocator(result),
                       (void*)target_listing_identifier.data);
  for (iree_host_size_t i = 0;
       i < target_artifact->sidecar_count && loomc_status_is_ok(status); ++i) {
    const loom_target_emit_sidecar_artifact_t* sidecar =
        &target_artifact->sidecars[i];
    if (sidecar->contents == NULL) {
      return loomc_make_status(LOOMC_STATUS_INTERNAL,
                               "emitter returned no sidecar contents");
    }
    loomc_artifact_kind_t kind = LOOMC_ARTIFACT_KIND_REPORT;
    loomc_string_view_t format = loomc_string_view_empty();
    LOOMC_RETURN_IF_ERROR(
        loomc_emit_sidecar_artifact_metadata(sidecar->kind, &kind, &format));
    status = loomc_emit_add_byte_sequence_artifact(
        result, kind, format, loomc_string_view_from_iree(sidecar->identifier),
        sidecar->contents);
  }
  return status;
}

loomc_status_t loomc_emit_transaction_initialize(
    const loomc_emit_options_t* options, loomc_result_t* result,
    loomc_emit_transaction_t* out_transaction) {
  if (result == NULL || out_transaction == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result and out_transaction must not be NULL");
  }
  *out_transaction = (loomc_emit_transaction_t){
      .result = result,
  };
  loomc_status_t status = loomc_emit_resolve_options(
      options, result, loomc_result_allocator(result),
      &out_transaction->options);
  if (loomc_status_is_ok(status) &&
      out_transaction->options.compile_report_mode !=
          LOOMC_COMPILE_REPORT_MODE_NONE) {
    loom_target_compile_report_initialize(
        &out_transaction->compile_report,
        iree_allocator_from_loomc(loomc_result_allocator(result)));
    out_transaction->compile_report.requested_detail_flags =
        loom_target_compile_report_requested_detail_flags(
            loomc_emit_target_compile_report_mode(
                out_transaction->options.compile_report_mode));
    out_transaction->compile_report_initialized = true;
  }
  if (loomc_status_is_ok(status) &&
      out_transaction->options.compile_report_mode ==
          LOOMC_COMPILE_REPORT_MODE_DETAILS &&
      out_transaction->options.compile_report_format ==
          LOOMC_COMPILE_REPORT_FORMAT_JSON) {
    loom_json_value_list_initialize(
        iree_allocator_from_loomc(loomc_result_allocator(result)),
        &out_transaction->compile_report_diagnostics);
    out_transaction->compile_report_diagnostic_sink = (loom_diagnostic_sink_t){
        .fn = loomc_emit_capture_compile_report_diagnostic,
        .user_data = out_transaction,
    };
    loomc_result_set_loom_diagnostic_sink(
        result, &out_transaction->compile_report_diagnostic_sink);
    out_transaction->compile_report_diagnostics_initialized = true;
  }
  return status;
}

loomc_string_view_t loomc_emit_transaction_artifact_format(
    const loomc_emit_transaction_t* transaction) {
  IREE_ASSERT_ARGUMENT(transaction);
  return transaction->options.artifact_format;
}

void loomc_emit_transaction_set_diagnostic_environment(
    loomc_emit_transaction_t* transaction,
    const loomc_target_environment_t* target_environment) {
  IREE_ASSERT_ARGUMENT(transaction);
  IREE_ASSERT_ARGUMENT(target_environment);
  const loomc_target_pass_environment_t* pass_environment =
      loomc_target_environment_pass_environment(target_environment);
  transaction->diagnostic_text_print_options =
      pass_environment ? &pass_environment->diagnostic_type_print_options
                       : NULL;
}

void loomc_emit_transaction_bind_emitter(loomc_emit_transaction_t* transaction,
                                         const loom_target_emitter_t* emitter) {
  IREE_ASSERT_ARGUMENT(transaction);
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT(!transaction->emitter, "emitter can only be selected once");
  transaction->emitter = emitter;
  if (!transaction->compile_report_initialized) {
    return;
  }
  transaction->compile_report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
  transaction->compile_report.backend_name = emitter->name;
  transaction->compile_report.artifact_format =
      loom_target_artifact_format_name(emitter->target_artifact_format);
}

loomc_status_t loomc_emit_transaction_select_emitter(
    loomc_emit_transaction_t* transaction,
    const loom_target_environment_t* target_environment) {
  IREE_ASSERT_ARGUMENT(transaction);
  IREE_ASSERT_ARGUMENT(target_environment);
  const loom_target_emitter_t* emitter = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_emit_select_emitter(
      target_environment, transaction->options.artifact_format,
      transaction->result, loomc_result_allocator(transaction->result),
      &emitter));
  if (loomc_result_succeeded(transaction->result)) {
    loomc_emit_transaction_bind_emitter(transaction, emitter);
  }
  return loomc_ok_status();
}

loom_target_compile_report_t* loomc_emit_transaction_compile_report(
    loomc_emit_transaction_t* transaction) {
  IREE_ASSERT_ARGUMENT(transaction);
  return transaction->compile_report_initialized ? &transaction->compile_report
                                                 : NULL;
}

void loomc_emit_transaction_record_status(loomc_emit_transaction_t* transaction,
                                          iree_status_code_t status_code) {
  IREE_ASSERT_ARGUMENT(transaction);
  if (transaction->compile_report_initialized) {
    loom_target_compile_report_record_status(&transaction->compile_report,
                                             status_code);
  }
}

loomc_status_t loomc_emit_transaction_emit(
    loomc_emit_transaction_t* transaction,
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module) {
  IREE_ASSERT_ARGUMENT(transaction);
  IREE_ASSERT_ARGUMENT(transaction->result);
  IREE_ASSERT_ARGUMENT(transaction->emitter);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(workspace);
  IREE_ASSERT_ARGUMENT(module);

  loomc_result_t* result = transaction->result;
  const loom_target_emitter_t* emitter = transaction->emitter;
  const loomc_emit_resolved_options_t* options = &transaction->options;
  loom_module_t* internal_module = loomc_module_loom_module(module);
  IREE_ASSERT_ARGUMENT(internal_module);
  loomc_emit_transaction_set_diagnostic_environment(transaction,
                                                    target_environment);

  loomc_status_t status =
      loomc_module_admit(module, target_environment, result);
  if (loomc_status_is_ok(status) && !loomc_result_succeeded(result)) {
    loomc_emit_transaction_record_status(transaction,
                                         IREE_STATUS_FAILED_PRECONDITION);
    return status;
  }

  loom_target_emit_artifact_t target_artifact = {0};
  loomc_string_view_t manifest_identifier = loomc_string_view_empty();
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  const loomc_target_pass_environment_t* pass_environment =
      loomc_target_environment_pass_environment(target_environment);
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(
      result, /*source=*/NULL, internal_module,
      loomc_module_source_resolver(module), LOOM_EMITTER_VERIFIER,
      pass_environment ? &pass_environment->diagnostic_type_print_options
                       : NULL,
      &capture);
  if (options->artifact_manifest_mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE) {
    status = loomc_emit_make_manifest_identifier(
        options, emitter, loomc_result_allocator(result), &manifest_identifier);
  }
  if (loomc_status_is_ok(status) && transaction->compile_report_initialized) {
    status =
        loomc_status_from_iree(loom_target_compile_report_record_loop_pipelines(
            &transaction->compile_report, internal_module,
            loomc_module_function_versions(module)));
  }
  if (transaction->compile_report_initialized) {
    for (const loomc_config_binding_record_t* binding =
             loomc_module_config_bindings(module)->head;
         binding != NULL && loomc_status_is_ok(status);
         binding = binding->next) {
      const loom_target_compile_report_config_binding_row_t row = {
          .key = binding->key,
          .value = binding->value,
      };
      status = loomc_status_from_iree(
          loom_target_compile_report_record_config_binding_row(
              &transaction->compile_report, &row));
    }
  }
  loom_target_emit_request_flags_t request_flags =
      LOOM_TARGET_EMIT_REQUEST_FLAG_NONE;
  if (iree_any_bit_set(options->artifact_flags,
                       LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING)) {
    request_flags |= LOOM_TARGET_EMIT_REQUEST_FLAG_TARGET_LISTING;
  }
  const loom_target_emit_request_t request = {
      .target_environment =
          loomc_target_environment_loom_target_environment(target_environment),
      .low_descriptor_registry =
          &pass_environment->low_descriptor_registry.registry,
      .module = internal_module,
      .function_versions = loomc_module_function_versions(module),
      .option_chain = options->option_chain,
      .identifier =
          iree_string_view_from_loomc(loomc_emit_identifier(options, emitter)),
      .artifact_manifest =
          {
              .mode = loomc_emit_target_manifest_mode(
                  options->artifact_manifest_mode),
              .identifier = iree_string_view_from_loomc(manifest_identifier),
          },
      .flags = request_flags,
      .compile_report = loomc_emit_transaction_compile_report(transaction),
      .diagnostic_emitter =
          {
              .fn = loomc_diagnostic_capture_emission,
              .user_data = &capture,
          },
      .max_errors = 20,
      .scratch_arena = &scratch_arena,
      .allocator = iree_allocator_from_loomc(loomc_result_allocator(result)),
  };
  if (loomc_status_is_ok(status)) {
    bool target_emitted = false;
    iree_status_t emit_status =
        emitter->emit(&request, &target_emitted, &target_artifact);
    iree_status_code_t report_status = iree_status_code(emit_status);
    if (report_status == IREE_STATUS_OK && !target_emitted) {
      report_status = IREE_STATUS_FAILED_PRECONDITION;
    }
    loomc_emit_transaction_record_status(transaction, report_status);
    if (transaction->compile_report_initialized &&
        target_artifact.contents != NULL) {
      loom_target_compile_report_record_artifact_size(
          &transaction->compile_report,
          iree_byte_sequence_length(target_artifact.contents));
    }
    status = loomc_status_from_iree(emit_status);
    if (loomc_status_is_ok(status) && !target_emitted) {
      loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
    }
  }
  if (!loomc_status_is_ok(status) &&
      loomc_status_is_result_diagnostic(status)) {
    status = loomc_result_fail_status_diagnostic_consume(
        result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
        loomc_make_cstring_view("EMIT/TARGET"), status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status =
        loomc_emit_add_artifact(result, options, emitter, &target_artifact);
    if (loomc_status_is_ok(status) && !loomc_result_succeeded(result)) {
      loomc_emit_transaction_record_status(transaction,
                                           IREE_STATUS_FAILED_PRECONDITION);
    }
  }

  loomc_allocator_free(loomc_result_allocator(result),
                       (void*)manifest_identifier.data);
  loom_target_emit_artifact_release(&target_artifact);
  iree_arena_deinitialize(&scratch_arena);
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_module_invalidate_verification(module);
  }
  return status;
}

loomc_status_t loomc_emit_transaction_finish(
    loomc_emit_transaction_t* transaction) {
  IREE_ASSERT_ARGUMENT(transaction);
  if (!transaction->compile_report_initialized) {
    return loomc_ok_status();
  }
  loomc_string_view_t identifier = loomc_string_view_empty();
  loomc_allocator_t allocator = loomc_result_allocator(transaction->result);
  loomc_status_t status = loomc_emit_make_compile_report_identifier(
      &transaction->options, transaction->emitter, allocator, &identifier);
  if (loomc_status_is_ok(status)) {
    const iree_string_view_t diagnostic_json_objects =
        transaction->compile_report_diagnostics_initialized
            ? loom_json_value_list_body(
                  &transaction->compile_report_diagnostics)
            : iree_string_view_empty();
    status = loomc_emit_add_compile_report_artifact(
        transaction->result, &transaction->options, identifier,
        &transaction->compile_report, diagnostic_json_objects);
  }
  loomc_allocator_free(allocator, (void*)identifier.data);
  return status;
}

void loomc_emit_transaction_deinitialize(
    loomc_emit_transaction_t* transaction) {
  if (transaction == NULL) {
    return;
  }
  if (transaction->compile_report_diagnostics_initialized) {
    loomc_result_set_loom_diagnostic_sink(transaction->result, NULL);
    loom_json_value_list_deinitialize(&transaction->compile_report_diagnostics);
  }
  if (transaction->compile_report_initialized) {
    loom_target_compile_report_deinitialize(&transaction->compile_report);
  }
  *transaction = (loomc_emit_transaction_t){0};
}

loomc_status_t loomc_emit_module(loomc_target_environment_t* target_environment,
                                 loomc_workspace_t* workspace,
                                 loomc_module_t* module,
                                 const loomc_emit_options_t* options,
                                 loomc_allocator_t allocator,
                                 loomc_result_t** out_result) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_result == NULL) {
    loomc_status_t return_status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT, "out_result must not be NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  *out_result = NULL;
  if (target_environment == NULL || workspace == NULL || module == NULL) {
    loomc_status_t return_status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target_environment, workspace, and module must not be NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  if (loomc_module_loom_module(module) == NULL) {
    loomc_status_t return_status =
        loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                          "module does not contain internal IR");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }

  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_result_create(
      LOOMC_STATUS_OK,
      loomc_context_source_retention(loomc_module_context(module)), allocator,
      &result);
  loomc_emit_transaction_t transaction = {0};
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_transaction_initialize(options, result, &transaction);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_emit_transaction_select_emitter(
        &transaction,
        loomc_target_environment_loom_target_environment(target_environment));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_emit_transaction_emit(&transaction, target_environment,
                                         workspace, module);
  }
  if (loomc_status_is_ok(status) && !loomc_result_succeeded(result)) {
    loomc_emit_transaction_record_status(&transaction,
                                         IREE_STATUS_FAILED_PRECONDITION);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_emit_transaction_finish(&transaction);
  }
  loomc_emit_transaction_deinitialize(&transaction);
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_module_invalidate_verification(module);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
