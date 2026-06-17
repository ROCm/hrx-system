// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/benchmark.h"

#include <string.h>

#include "iree/hal/utils/profile_file.h"
#include "iree/hal/utils/statistics_sink.h"
#include "iree/io/file_handle.h"

typedef struct loom_run_hal_profile_tee_sink_t {
  // HAL resource header for the sink interface.
  iree_hal_resource_t resource;
  // Host allocator used for sink lifetime.
  iree_allocator_t host_allocator;
  // First retained sink receiving every profiling callback.
  iree_hal_profile_sink_t* first_sink;
  // Second retained sink receiving every profiling callback.
  iree_hal_profile_sink_t* second_sink;
} loom_run_hal_profile_tee_sink_t;

static const iree_hal_profile_sink_vtable_t
    loom_run_hal_profile_tee_sink_vtable;

static loom_run_hal_profile_tee_sink_t* loom_run_hal_profile_tee_sink_cast(
    iree_hal_profile_sink_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &loom_run_hal_profile_tee_sink_vtable);
  return (loom_run_hal_profile_tee_sink_t*)base_value;
}

static void loom_run_hal_profile_tee_sink_destroy(
    iree_hal_profile_sink_t* base_sink) {
  loom_run_hal_profile_tee_sink_t* sink =
      loom_run_hal_profile_tee_sink_cast(base_sink);
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_hal_profile_sink_release(sink->second_sink);
  iree_hal_profile_sink_release(sink->first_sink);
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t loom_run_hal_profile_tee_sink_begin_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  loom_run_hal_profile_tee_sink_t* sink =
      loom_run_hal_profile_tee_sink_cast(base_sink);
  iree_status_t status =
      iree_hal_profile_sink_begin_session(sink->first_sink, metadata);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_sink_begin_session(sink->second_sink, metadata);
    if (!iree_status_is_ok(status)) {
      const iree_status_code_t status_code = iree_status_code(status);
      status = iree_status_join(
          status, iree_hal_profile_sink_end_session(sink->first_sink, metadata,
                                                    status_code));
    }
  }
  return status;
}

static iree_status_t loom_run_hal_profile_tee_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  loom_run_hal_profile_tee_sink_t* sink =
      loom_run_hal_profile_tee_sink_cast(base_sink);
  iree_status_t status = iree_hal_profile_sink_write(sink->first_sink, metadata,
                                                     iovec_count, iovecs);
  return iree_status_join(
      status, iree_hal_profile_sink_write(sink->second_sink, metadata,
                                          iovec_count, iovecs));
}

static iree_status_t loom_run_hal_profile_tee_sink_end_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  loom_run_hal_profile_tee_sink_t* sink =
      loom_run_hal_profile_tee_sink_cast(base_sink);
  iree_status_t status = iree_hal_profile_sink_end_session(
      sink->first_sink, metadata, session_status_code);
  return iree_status_join(
      status, iree_hal_profile_sink_end_session(sink->second_sink, metadata,
                                                session_status_code));
}

static iree_status_t loom_run_hal_profile_tee_sink_create(
    iree_hal_profile_sink_t* first_sink, iree_hal_profile_sink_t* second_sink,
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(first_sink);
  IREE_ASSERT_ARGUMENT(second_sink);
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;

  loom_run_hal_profile_tee_sink_t* sink = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink));
  iree_hal_resource_initialize(&loom_run_hal_profile_tee_sink_vtable,
                               &sink->resource);
  sink->host_allocator = host_allocator;
  sink->first_sink = first_sink;
  iree_hal_profile_sink_retain(sink->first_sink);
  sink->second_sink = second_sink;
  iree_hal_profile_sink_retain(sink->second_sink);
  *out_sink = (iree_hal_profile_sink_t*)sink;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_profile_file_sink_create(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_create(
      IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN |
          IREE_IO_FILE_MODE_SHARE_READ,
      path, /*initial_size=*/0, host_allocator, &file_handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_sink_create(file_handle, host_allocator,
                                               out_sink);
  }
  iree_io_file_handle_release(file_handle);
  return status;
}

void loom_run_hal_benchmark_options_initialize(
    loom_run_hal_benchmark_options_t* out_options) {
  *out_options = (loom_run_hal_benchmark_options_t){0};
  loom_run_benchmark_options_initialize(&out_options->timing);
  loom_run_hal_dispatch_batch_options_initialize(&out_options->dispatch_batch);
  out_options->dispatch_batch.dispatch_count = out_options->timing.batch_size;
  out_options->profile_flags = IREE_HAL_DEVICE_PROFILING_FLAG_NONE;
  out_options->profile_data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  out_options->profile_capture_filter =
      iree_hal_profile_capture_filter_default();
}

void loom_run_hal_benchmark_result_initialize(
    loom_run_hal_benchmark_result_t* out_result) {
  *out_result = (loom_run_hal_benchmark_result_t){0};
}

static iree_status_t loom_run_hal_benchmark_options_validate(
    const loom_run_hal_benchmark_options_t* options) {
  const loom_run_hal_benchmark_flags_t known_flags =
      LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;
  if (iree_any_bit_set(options->flags, ~known_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown HAL benchmark flags 0x%08X",
                            options->flags & ~known_flags);
  }
  if (options->timing.batch_size != options->dispatch_batch.dispatch_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark timing batch_size %" PRIhsz
                            " must match dispatch_count %" PRIhsz,
                            options->timing.batch_size,
                            options->dispatch_batch.dispatch_count);
  }
  if (options->dispatch_batch.dispatch_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark dispatch_count must be positive");
  }
  return iree_ok_status();
}

static bool loom_run_hal_benchmark_options_request_explicit_profile_counters(
    const loom_run_hal_benchmark_options_t* options) {
  if (!options->profile_counter_sets) return false;
  for (iree_host_size_t i = 0; i < options->profile_counter_set_count; ++i) {
    if (options->profile_counter_sets[i].counter_name_count != 0) return true;
  }
  return false;
}

typedef struct loom_run_hal_benchmark_batch_context_t {
  // HAL runtime that owns the device used for dispatch.
  const loom_run_hal_runtime_t* runtime;
  // Prepared reusable dispatch batches.
  loom_run_hal_dispatch_batch_t* batches;
  // Number of entries in |batches|.
  iree_host_size_t batch_count;
  // Index of the next batch to submit.
  iree_host_size_t next_batch_index;
} loom_run_hal_benchmark_batch_context_t;

static iree_status_t loom_run_hal_benchmark_execute_batch(void* user_data) {
  loom_run_hal_benchmark_batch_context_t* context =
      (loom_run_hal_benchmark_batch_context_t*)user_data;
  loom_run_hal_dispatch_batch_t* batch =
      &context->batches[context->next_batch_index];
  context->next_batch_index =
      (context->next_batch_index + 1) % context->batch_count;
  return loom_run_hal_dispatch_batch_execute(context->runtime, batch);
}

static void loom_run_hal_profile_summary_record_error(
    loom_run_hal_profile_summary_t* profile, const iree_status_t status) {
  profile->has_error = true;
  profile->error_code = iree_status_code(status);
  iree_host_size_t required_length = 0;
  if (iree_status_format(status, sizeof(profile->error_message),
                         profile->error_message, &required_length)) {
    profile->error_message_length = required_length;
    if (profile->error_message_length >= sizeof(profile->error_message)) {
      profile->error_message_length = sizeof(profile->error_message) - 1;
    }
  } else {
    const char* error_code_string =
        iree_status_code_string(profile->error_code);
    iree_host_size_t index = 0;
    for (; error_code_string[index] != '\0' &&
           index + 1 < sizeof(profile->error_message);
         ++index) {
      profile->error_message[index] = error_code_string[index];
    }
    profile->error_message[index] = '\0';
    profile->error_message_length = index;
  }
}

static void loom_run_hal_profile_summary_copy_artifact_path(
    loom_run_hal_profile_summary_t* profile, iree_string_view_t artifact_path) {
  if (iree_string_view_is_empty(artifact_path)) {
    return;
  }
  profile->has_artifact_path = true;
  const iree_host_size_t copy_length = iree_min(
      artifact_path.size,
      (iree_host_size_t)LOOM_RUN_HAL_PROFILE_ARTIFACT_PATH_CAPACITY - 1);
  memcpy(profile->artifact_path, artifact_path.data, copy_length);
  profile->artifact_path[copy_length] = '\0';
  profile->artifact_path_length = copy_length;
}

typedef struct loom_run_hal_profile_summary_capture_context_t {
  // Statistics sink used to resolve function names and scale device ticks.
  const iree_hal_profile_statistics_sink_t* sink;
  // Profile summary receiving bounded row copies.
  loom_run_hal_profile_summary_t* profile;
} loom_run_hal_profile_summary_capture_context_t;

static void loom_run_hal_profile_summary_copy_function_name(
    const iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_statistics_row_t* row,
    loom_run_hal_profile_row_summary_t* summary) {
  summary->function_name_length = 0;
  if (row->executable_id == 0 || row->function_ordinal == UINT32_MAX) {
    return;
  }
  iree_string_view_t function_name = iree_string_view_empty();
  if (!iree_hal_profile_statistics_sink_find_function_name(
          sink, row->executable_id, row->function_ordinal, &function_name)) {
    return;
  }
  iree_host_size_t copy_length = iree_min(
      function_name.size,
      (iree_host_size_t)LOOM_RUN_HAL_PROFILE_FUNCTION_NAME_CAPACITY - 1);
  memcpy(summary->function_name, function_name.data, copy_length);
  summary->function_name[copy_length] = '\0';
  summary->function_name_length = copy_length;
}

static void loom_run_hal_profile_summary_copy_scaled_duration(
    const iree_hal_profile_statistics_sink_t* sink,
    const iree_hal_profile_statistics_row_t* row,
    loom_run_hal_profile_row_summary_t* summary) {
  if (!iree_all_bits_set(row->flags,
                         IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING)) {
    return;
  }
  uint64_t total_duration_ns = 0;
  uint64_t minimum_duration_ns = 0;
  uint64_t maximum_duration_ns = 0;
  if (!iree_hal_profile_statistics_sink_scale_duration_to_ns(
          sink, row, row->total_duration, &total_duration_ns) ||
      !iree_hal_profile_statistics_sink_scale_duration_to_ns(
          sink, row, row->minimum_duration, &minimum_duration_ns) ||
      !iree_hal_profile_statistics_sink_scale_duration_to_ns(
          sink, row, row->maximum_duration, &maximum_duration_ns)) {
    return;
  }
  summary->has_scaled_duration_ns = true;
  summary->total_duration_ns = total_duration_ns;
  summary->minimum_duration_ns = minimum_duration_ns;
  summary->maximum_duration_ns = maximum_duration_ns;
}

static iree_status_t loom_run_hal_profile_summary_capture_row(
    void* user_data, const iree_hal_profile_statistics_row_t* row) {
  loom_run_hal_profile_summary_capture_context_t* context =
      (loom_run_hal_profile_summary_capture_context_t*)user_data;
  loom_run_hal_profile_summary_t* profile = context->profile;
  if (profile->captured_row_count >= LOOM_RUN_HAL_PROFILE_SUMMARY_MAX_ROWS) {
    ++profile->truncated_row_count;
    return iree_ok_status();
  }

  loom_run_hal_profile_row_summary_t* summary =
      &profile->rows[profile->captured_row_count++];
  *summary = (loom_run_hal_profile_row_summary_t){
      .row_type = row->row_type,
      .time_domain = row->time_domain,
      .flags = row->flags,
      .physical_device_ordinal = row->physical_device_ordinal,
      .queue_ordinal = row->queue_ordinal,
      .event_type = row->event_type,
      .executable_id = row->executable_id,
      .command_buffer_id = row->command_buffer_id,
      .function_ordinal = row->function_ordinal,
      .command_index = row->command_index,
      .sample_count = row->sample_count,
      .invalid_sample_count = row->invalid_sample_count,
      .operation_count = row->operation_count,
      .payload_bytes = row->payload_bytes,
      .tile_count = row->tile_count,
      .tile_duration_sum_ns = row->tile_duration_sum_ns,
      .first_start_time = row->first_start_time,
      .last_end_time = row->last_end_time,
      .total_duration = row->total_duration,
      .minimum_duration = row->minimum_duration,
      .maximum_duration = row->maximum_duration,
  };
  loom_run_hal_profile_summary_copy_function_name(context->sink, row, summary);
  loom_run_hal_profile_summary_copy_scaled_duration(context->sink, row,
                                                    summary);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_benchmark_run_profiled_batch(
    const loom_run_hal_runtime_t* runtime, loom_run_hal_dispatch_batch_t* batch,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_profile_summary_t* out_profile) {
  *out_profile = (loom_run_hal_profile_summary_t){
      .requested = true,
      .flags = options->profile_flags,
      .data_families = options->profile_data_families,
  };
  loom_run_hal_profile_summary_copy_artifact_path(
      out_profile, options->profile_artifact_path);

  iree_hal_profile_statistics_sink_t* statistics_sink = NULL;
  iree_status_t status =
      iree_hal_profile_statistics_sink_create(allocator, &statistics_sink);
  iree_hal_profile_sink_t* file_sink = NULL;
  iree_hal_profile_sink_t* tee_sink = NULL;
  iree_hal_profile_sink_t* sink = NULL;

  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(options->profile_artifact_path)) {
    status = loom_run_hal_profile_file_sink_create(
        options->profile_artifact_path, allocator, &file_sink);
  }
  if (iree_status_is_ok(status)) {
    sink = iree_hal_profile_statistics_sink_base(statistics_sink);
    if (file_sink != NULL) {
      status = loom_run_hal_profile_tee_sink_create(sink, file_sink, allocator,
                                                    &tee_sink);
      if (iree_status_is_ok(status)) {
        sink = tee_sink;
      }
    } else if (options->profile_artifact_sink != NULL) {
      status = loom_run_hal_profile_tee_sink_create(
          sink, options->profile_artifact_sink, allocator, &tee_sink);
      if (iree_status_is_ok(status)) {
        sink = tee_sink;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_profiling_options_t profiling_options = {
        .flags = options->profile_flags,
        .data_families = options->profile_data_families,
        .sink = sink,
        .capture_filter = options->profile_capture_filter,
        .counter_set_count = options->profile_counter_set_count,
        .counter_sets = options->profile_counter_sets,
    };
    status =
        iree_hal_device_profiling_begin(runtime->device, &profiling_options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch_batch_execute(runtime, batch);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_profiling_flush(runtime->device);
    }
    status = iree_status_join(status,
                              iree_hal_device_profiling_end(runtime->device));
  }
  if (iree_status_is_ok(status)) {
    out_profile->executed = true;
    out_profile->row_count =
        iree_hal_profile_statistics_sink_row_count(statistics_sink);
    out_profile->dropped_record_count =
        iree_hal_profile_statistics_sink_dropped_record_count(statistics_sink);
    loom_run_hal_profile_summary_capture_context_t context = {
        .sink = statistics_sink,
        .profile = out_profile,
    };
    status = iree_hal_profile_statistics_sink_for_each_row(
        statistics_sink, (iree_hal_profile_statistics_row_callback_t){
                             .fn = loom_run_hal_profile_summary_capture_row,
                             .user_data = &context,
                         });
  } else if (!loom_run_hal_benchmark_options_request_explicit_profile_counters(
                 options)) {
    loom_run_hal_profile_summary_record_error(out_profile, status);
    iree_status_free(status);
    status = iree_ok_status();
  }

  iree_hal_profile_statistics_sink_release(statistics_sink);
  iree_hal_profile_sink_release(tee_sink);
  iree_hal_profile_sink_release(file_sink);
  return status;
}

static iree_status_t loom_run_hal_benchmark_profile_final_batch(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    iree_host_size_t binding_list_count,
    const loom_run_hal_binding_list_t* binding_lists,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_profile_summary_t* out_profile) {
  *out_profile = (loom_run_hal_profile_summary_t){
      .requested = true,
      .flags = options->profile_flags,
      .data_families = options->profile_data_families,
  };
  loom_run_hal_profile_summary_copy_artifact_path(
      out_profile, options->profile_artifact_path);

  loom_run_hal_dispatch_batch_options_t profile_dispatch_options =
      options->dispatch_batch;
  // Record a separate metadata-retaining batch so measured submissions keep the
  // fast unretained command-buffer shape.
  profile_dispatch_options.command_buffer_mode |=
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA;
  profile_dispatch_options.command_buffer_mode &=
      ~IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
  profile_dispatch_options.execute_flags &=
      ~IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME;

  loom_run_hal_dispatch_batch_t profile_batch = {0};
  iree_status_t status = loom_run_hal_dispatch_batch_prepare_from_binding_ring(
      runtime, candidate, plan, binding_list_count, binding_lists,
      /*binding_list_offset=*/0, &profile_dispatch_options, allocator,
      &profile_batch);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_benchmark_run_profiled_batch(
        runtime, &profile_batch, options, allocator, out_profile);
  } else {
    loom_run_hal_profile_summary_record_error(out_profile, status);
    iree_status_free(status);
    status = iree_ok_status();
  }
  loom_run_hal_dispatch_batch_deinitialize(&profile_batch);
  return status;
}

static iree_status_t loom_run_hal_benchmark_profile_final_sequence_batch(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_profile_summary_t* out_profile) {
  *out_profile = (loom_run_hal_profile_summary_t){
      .requested = true,
      .flags = options->profile_flags,
      .data_families = options->profile_data_families,
  };
  loom_run_hal_profile_summary_copy_artifact_path(
      out_profile, options->profile_artifact_path);

  loom_run_hal_dispatch_batch_options_t profile_dispatch_options =
      options->dispatch_batch;
  // Record a separate metadata-retaining batch so measured submissions keep the
  // fast unretained command-buffer shape.
  profile_dispatch_options.command_buffer_mode |=
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA;
  profile_dispatch_options.command_buffer_mode &=
      ~IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
  profile_dispatch_options.execute_flags &=
      ~IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME;

  loom_run_hal_dispatch_batch_t profile_batch = {0};
  iree_status_t status =
      loom_run_hal_dispatch_sequence_batch_prepare_from_plan_ring(
          runtime, sequence_count, candidates, plan_ring_count, plans,
          /*plan_ring_offset=*/0, &profile_dispatch_options, allocator,
          &profile_batch);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_benchmark_run_profiled_batch(
        runtime, &profile_batch, options, allocator, out_profile);
  } else {
    loom_run_hal_profile_summary_record_error(out_profile, status);
    iree_status_free(status);
    status = iree_ok_status();
  }
  loom_run_hal_dispatch_batch_deinitialize(&profile_batch);
  return status;
}

static const iree_hal_profile_sink_vtable_t
    loom_run_hal_profile_tee_sink_vtable = {
        .destroy = loom_run_hal_profile_tee_sink_destroy,
        .begin_session = loom_run_hal_profile_tee_sink_begin_session,
        .write = loom_run_hal_profile_tee_sink_write,
        .end_session = loom_run_hal_profile_tee_sink_end_session,
};

iree_status_t loom_run_hal_benchmark_dispatch_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result) {
  return loom_run_hal_benchmark_dispatch_binding_ring(
      runtime, candidate, plan, /*binding_list_count=*/1, &plan->bindings,
      options, allocator, out_result);
}

iree_status_t loom_run_hal_benchmark_dispatch_binding_ring(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    iree_host_size_t binding_list_count,
    const loom_run_hal_binding_list_t* binding_lists,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result) {
  loom_run_hal_benchmark_result_initialize(out_result);
  IREE_RETURN_IF_ERROR(loom_run_hal_benchmark_options_validate(options));
  if (binding_list_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark binding ring must contain at least "
                            "one binding list");
  }

  const iree_host_size_t command_buffer_ring_count = iree_host_size_ceil_div(
      binding_list_count, options->dispatch_batch.dispatch_count);
  loom_run_hal_dispatch_batch_t* batches = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, command_buffer_ring_count, sizeof(*batches), (void**)&batches);
  if (iree_status_is_ok(status)) {
    memset(batches, 0, command_buffer_ring_count * sizeof(*batches));
  }
  const iree_host_size_t batch_binding_list_count =
      iree_min(binding_list_count, options->dispatch_batch.dispatch_count);
  loom_run_hal_binding_list_t* batch_binding_lists = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, batch_binding_list_count,
                                         sizeof(*batch_binding_lists),
                                         (void**)&batch_binding_lists);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < command_buffer_ring_count; ++i) {
    const iree_host_size_t binding_list_offset =
        i * options->dispatch_batch.dispatch_count;
    for (iree_host_size_t j = 0; j < batch_binding_list_count; ++j) {
      batch_binding_lists[j] =
          binding_lists[(binding_list_offset + j) % binding_list_count];
    }
    status = loom_run_hal_dispatch_batch_prepare_from_binding_ring(
        runtime, candidate, plan, batch_binding_list_count, batch_binding_lists,
        /*binding_list_offset=*/0, &options->dispatch_batch, allocator,
        &batches[i]);
  }
  iree_allocator_free(allocator, batch_binding_lists);

  loom_run_hal_benchmark_batch_context_t context = {
      .runtime = runtime,
      .batches = batches,
      .batch_count = command_buffer_ring_count,
  };
  if (iree_status_is_ok(status)) {
    status = loom_run_benchmark_run_batches(
        (loom_run_benchmark_batch_callback_t){
            .fn = loom_run_hal_benchmark_execute_batch,
            .user_data = &context,
        },
        &options->timing, allocator, &out_result->timing);
  }
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(options->flags,
                        LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)) {
    status = loom_run_hal_benchmark_profile_final_batch(
        runtime, candidate, plan, binding_list_count, binding_lists, options,
        allocator, &out_result->profile);
  }
  if (iree_status_is_ok(status)) {
    out_result->binding_ring_count = binding_list_count;
    out_result->command_buffer_ring_count = command_buffer_ring_count;
  }

  if (batches != NULL) {
    for (iree_host_size_t i = 0; i < command_buffer_ring_count; ++i) {
      loom_run_hal_dispatch_batch_deinitialize(&batches[i]);
    }
  }
  iree_allocator_free(allocator, batches);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_benchmark_result_initialize(out_result);
  }
  return status;
}

iree_status_t loom_run_hal_benchmark_dispatch_sequence_plan_ring(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result) {
  loom_run_hal_benchmark_result_initialize(out_result);
  IREE_RETURN_IF_ERROR(loom_run_hal_benchmark_options_validate(options));
  if (sequence_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark dispatch sequence must contain at "
                            "least one dispatch step");
  }
  if (plan_ring_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark sequence plan ring must contain at "
                            "least one ring slot");
  }

  const iree_host_size_t command_buffer_ring_count = iree_host_size_ceil_div(
      plan_ring_count, options->dispatch_batch.dispatch_count);
  loom_run_hal_dispatch_batch_t* batches = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, command_buffer_ring_count, sizeof(*batches), (void**)&batches);
  if (iree_status_is_ok(status)) {
    memset(batches, 0, command_buffer_ring_count * sizeof(*batches));
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < command_buffer_ring_count; ++i) {
    const iree_host_size_t plan_ring_offset =
        i * options->dispatch_batch.dispatch_count;
    status = loom_run_hal_dispatch_sequence_batch_prepare_from_plan_ring(
        runtime, sequence_count, candidates, plan_ring_count, plans,
        plan_ring_offset, &options->dispatch_batch, allocator, &batches[i]);
  }

  loom_run_hal_benchmark_batch_context_t context = {
      .runtime = runtime,
      .batches = batches,
      .batch_count = command_buffer_ring_count,
  };
  if (iree_status_is_ok(status)) {
    status = loom_run_benchmark_run_batches(
        (loom_run_benchmark_batch_callback_t){
            .fn = loom_run_hal_benchmark_execute_batch,
            .user_data = &context,
        },
        &options->timing, allocator, &out_result->timing);
  }
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(options->flags,
                        LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)) {
    status = loom_run_hal_benchmark_profile_final_sequence_batch(
        runtime, sequence_count, candidates, plan_ring_count, plans, options,
        allocator, &out_result->profile);
  }
  if (iree_status_is_ok(status)) {
    out_result->binding_ring_count = plan_ring_count;
    out_result->command_buffer_ring_count = command_buffer_ring_count;
  }

  if (batches != NULL) {
    for (iree_host_size_t i = 0; i < command_buffer_ring_count; ++i) {
      loom_run_hal_dispatch_batch_deinitialize(&batches[i]);
    }
  }
  iree_allocator_free(allocator, batches);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_benchmark_result_initialize(out_result);
  }
  return status;
}
