// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/check/loom_check.h"

#include "iree/io/vec_stream.h"
#include "loom/target/emit/native/x86/function.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

typedef struct loom_x86_loom_check_emit_options_t {
  // Module-local low.func.def symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Candidate selection strategy used by low frame.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed low allocation requests parsed from target options.
  loom_check_low_emit_fixed_value_spec_t allocation_fixed_value_specs
      [LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES];
  // Number of entries in |allocation_fixed_value_specs|.
  iree_host_size_t allocation_fixed_value_spec_count;
} loom_x86_loom_check_emit_options_t;

static bool loom_x86_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("x86-bytes"));
}

static iree_status_t loom_x86_loom_check_parse_key_value_option(
    iree_string_view_t token, loom_x86_loom_check_emit_options_t* options,
    bool* out_matched) {
  *out_matched = false;
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate x86 bytes option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("x86 bytes"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    *out_matched = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_parse_option(
    iree_string_view_t token, loom_x86_loom_check_emit_options_t* options) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_x86_loom_check_parse_key_value_option(token, options, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_check_low_emit_parse_allocation_option(
      token, IREE_SV("x86 bytes"), options->allocation_budgets,
      IREE_ARRAYSIZE(options->allocation_budgets),
      &options->allocation_budget_count, options->allocation_fixed_value_specs,
      IREE_ARRAYSIZE(options->allocation_fixed_value_specs),
      &options->allocation_fixed_value_spec_count);
}

static iree_status_t loom_x86_loom_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_x86_loom_check_emit_options_t* out_options) {
  *out_options = (loom_x86_loom_check_emit_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
  };

  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &symbol_name,
                         &option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  option_text = iree_string_view_trim(option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 bytes requires a low function symbol name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  if (iree_string_view_is_empty(out_options->function_symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 bytes low function symbol name is "
                            "required");
  }

  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_x86_loom_check_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_emit_bytes(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_x86_function_t function;
  IREE_RETURN_IF_ERROR(loom_x86_function_prepare(frame, arena, &function));
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  iree_status_t status = loom_x86_function_write(&function, stream, arena);
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  for (iree_io_stream_pos_t i = 0; i < length && iree_status_is_ok(status);
       ++i) {
    uint8_t byte = 0;
    status = iree_io_stream_read(stream, 1, &byte, NULL);
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_format(builder, "%s%02x",
                                                 i == 0 ? "" : " ", byte);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, "\n");
  }
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_x86_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_x86_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_x86_loom_check_parse_emit_options(request, &options));

  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {0};
  // The fixture supplies incoming value locations. Reserve RSP just as the
  // object provider does before allocation; the native envelope uses it.
  const loom_low_allocation_reserved_range_t stack_pointer = {
      .register_class = IREE_SV("x86.gpr64"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 4,
      .location_count = 1,
  };
  const loom_low_emission_frame_options_t frame_options = {
      .schedule_strategy = options.schedule_strategy,
      .allocation_budgets = options.allocation_budgets,
      .allocation_budget_count = options.allocation_budget_count,
      .allocation_reserved_ranges = &stack_pointer,
      .allocation_reserved_range_count = 1,
  };
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, &frame_options,
      options.allocation_fixed_value_specs,
      options.allocation_fixed_value_spec_count, &spill_free_options, &frame,
      &frame_accepted));
  if (request->diagnostic_collector != NULL &&
      request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }
  if (!frame_accepted) {
    return iree_ok_status();
  }
  return loom_x86_loom_check_emit_bytes(&frame, &request->result->actual_output,
                                        request->case_arena);
}

static iree_status_t loom_x86_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "x86-bytes");
}

const loom_check_emit_provider_t loom_x86_native_loom_check_emit_provider = {
    .name = IREE_SVL("x86-native"),
    .match = loom_x86_loom_check_emit_provider_matches,
    .execute = loom_x86_loom_check_emit_provider_execute,
    .append_names = loom_x86_loom_check_emit_provider_append_names,
};
