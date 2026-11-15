// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/device_profiling.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/hal/utils/statistics_sink.h"
#include "iree/io/file_handle.h"

//===----------------------------------------------------------------------===//
// Profiling
//===----------------------------------------------------------------------===//

IREE_FLAG(
    string, device_profiling_mode, "",
    "HAL device profiling data families as a comma-separated list drawn from\n"
    "['queue-events', 'host-execution', 'device-queue-events',\n"
    "'dispatch-events', 'memory-events', 'device-metrics',\n"
    "'command-region-events', 'counters', 'counter-ranges',\n"
    "'executable-metadata',\n"
    "'executable-traces'] or empty to disable profiling. HAL implementations\n"
    "may require additional flags in order to configure profiling support on\n"
    "their devices. Tooling may force command buffers to retain\n"
    "metadata for modes that need command/dispatch attribution; leave this\n"
    "empty for production timing runs.");
IREE_FLAG(
    string, device_profiling_output, "",
    "Path for a raw IREE HAL profiling bundle. Required when\n"
    "--device_profiling_mode is nonempty. The output is written by tooling\n"
    "using a generic profile sink.");
IREE_FLAG(
    string, device_profiling_filter_function, "",
    "Optional glob pattern selecting executable function names that should "
    "emit\n"
    "heavy profiling artifacts such as dispatch timings or hardware counter\n"
    "samples. Cheap session metadata remains available so filtered captures\n"
    "can still be decoded.");
IREE_FLAG(
    int64_t, device_profiling_filter_command_buffer, -1,
    "Optional nonzero command buffer id selecting operations that should emit\n"
    "heavy profiling artifacts. Use iree-profile command/queue output from an\n"
    "earlier capture to find command buffer ids.");
IREE_FLAG(
    int64_t, device_profiling_filter_command_index, -1,
    "Optional zero-based command-buffer operation index selecting operations\n"
    "that should emit heavy profiling artifacts.");
IREE_FLAG(
    int64_t, device_profiling_filter_physical_device, -1,
    "Optional physical device ordinal selecting operations that should emit\n"
    "heavy profiling artifacts.");
IREE_FLAG(int64_t, device_profiling_filter_queue, -1,
          "Optional queue ordinal selecting operations that should emit heavy\n"
          "profiling artifacts.");
IREE_FLAG_LIST(
    string, device_profiling_counter,
    "Optional implementation-specific hardware counter name to capture. May "
    "be\n"
    "specified multiple times; the selected HAL driver decides which counter\n"
    "names and combinations are supported. Use "
    "--device_profiling_mode=counters\n"
    "for dispatch-scoped attribution, which may inject packets around "
    "selected\n"
    "dispatches and perturb queue timing. Use\n"
    "--device_profiling_mode=counter-ranges for low-disturbance range "
    "samples.");
IREE_FLAG(
    int64_t, device_profiling_flush_interval_ms, 0,
    "Optional interval in milliseconds for a tooling-owned background thread\n"
    "to call iree_hal_device_profiling_flush while HAL-native profiling is\n"
    "active. 0 disables periodic flushing. The selected HAL backend must\n"
    "support safe in-flight profiling snapshots for the requested data.\n"
    "Sink failures are reported when profiling ends.");
IREE_FLAG(
    int64_t, device_profiling_trace_target_cu, 0,
    "Compute unit traced on each shader engine by\n"
    "--device_profiling_mode=executable-traces: the workgroup processor index\n"
    "on AMD gfx10 and later (rocprofv3 --att-target-cu). A trace samples one\n"
    "execution lane of this unit, not every wave of a dispatch.");
IREE_FLAG(int64_t, device_profiling_trace_simd_select, 0,
          "Execution lane traced within the target compute unit: a SIMD index "
          "on AMD\n"
          "gfx10 and later (rocprofv3 --att-simd-select).");
IREE_FLAG(
    int64_t, device_profiling_trace_shader_engine_mask, 0,
    "Shader engines that receive an executable trace buffer, one bit per\n"
    "engine. 0 selects every engine (rocprofv3 --att-shader-engine-mask).");
IREE_FLAG(
    int64_t, device_profiling_trace_buffer_size, 0,
    "Byte length of each shader engine's executable trace buffer. 0 selects\n"
    "the driver default. A full buffer truncates the trace (rocprofv3\n"
    "--att-buffer-size).");
IREE_FLAG(
    int64_t, device_profiling_trace_max_traces, 0,
    "Upper bound on operations traced by\n"
    "--device_profiling_mode=executable-traces. 0 selects the driver default.\n"
    "Use --device_profiling_filter_* to choose which operations are traced.");
IREE_FLAG(
    int64_t, device_profiling_counter_max_samples, 0,
    "Upper bound on dispatches sampled by --device_profiling_mode=counters.\n"
    "0 selects the driver default. Sampling brackets each dispatch with a\n"
    "compute join and register copies; dispatches beyond the bound run\n"
    "unsampled. Use --device_profiling_filter_* to choose which are sampled.");
IREE_FLAG(
    bool, device_profiling_allow_partial, false,
    "Lets each device in a multi-device group keep the largest subset of\n"
    "--device_profiling_mode families it implements instead of failing the\n"
    "whole group when one device reports a family as unimplemented. A device\n"
    "that implements none of the requested families is left unprofiled. A\n"
    "device that lacks a counter named by --device_profiling_counter loses\n"
    "its counter families the same way. Each reduction is reported on\n"
    "stderr.");
IREE_FLAG(
    bool, print_device_statistics, false,
    "Enables a lightweight HAL profiling session and prints aggregate device\n"
    "statistics at shutdown. This is a command-line convenience path: it does\n"
    "not write a profiling bundle and cannot be combined with\n"
    "--device_profiling_output. When --device_profiling_mode is empty,\n"
    "tooling requests the producer's lightweight execution-statistics mode.");

struct iree_hal_profiling_from_flags_t {
  // Host allocator used for this session object.
  iree_allocator_t host_allocator;

  // Devices retained while profiling is active.
  iree_hal_device_t** devices;

  // Number of entries in |devices| and |profile_active|.
  iree_host_size_t device_count;

  // Per-device true when a nonempty HAL profiling session is active.
  bool* profile_active;

  // Periodic flush interval in milliseconds, or 0 when disabled.
  iree_duration_t flush_interval_ms;

  // Mutex serializing explicit and periodic flush calls.
  iree_slim_mutex_t flush_mutex;

  // Background flush thread owned by the session when periodic flush is active.
  iree_thread_t* flush_thread;

  // In-memory aggregate sink used by --print_device_statistics.
  iree_hal_profile_statistics_sink_t* statistics_sink;

  // Notification used to wake the flush thread during shutdown.
  iree_notification_t flush_notification;

  // Nonzero when the flush thread should stop.
  iree_atomic_int32_t flush_stop_requested;

  // Mutex protecting |flush_status|.
  iree_slim_mutex_t flush_status_mutex;

  // First background flush failure, joined with any later failures.
  iree_status_t flush_status;
};

static iree_status_t iree_hal_profile_capture_filter_set_u32_from_flag(
    int64_t flag_value, iree_hal_profile_capture_filter_flags_t flag,
    const char* flag_name, iree_hal_profile_capture_filter_t* inout_filter,
    uint32_t* out_value) {
  if (flag_value < 0) {
    return iree_ok_status();
  }
  if (flag_value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%s value exceeds uint32_t", flag_name);
  }
  *out_value = (uint32_t)flag_value;
  inout_filter->flags |= flag;
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_capture_filter_from_flags(
    iree_hal_profile_capture_filter_t* out_filter) {
  *out_filter = iree_hal_profile_capture_filter_default();
  if (strlen(FLAG_device_profiling_filter_function) != 0) {
    out_filter->flags |=
        IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
    out_filter->executable_function_pattern =
        iree_make_cstring_view(FLAG_device_profiling_filter_function);
  }
  if (FLAG_device_profiling_filter_command_buffer == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_filter_command_buffer must be nonzero");
  } else if (FLAG_device_profiling_filter_command_buffer > 0) {
    out_filter->flags |= IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID;
    out_filter->command_buffer_id =
        (uint64_t)FLAG_device_profiling_filter_command_buffer;
  }
  IREE_RETURN_IF_ERROR(iree_hal_profile_capture_filter_set_u32_from_flag(
      FLAG_device_profiling_filter_command_index,
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX,
      "device_profiling_filter_command_index", out_filter,
      &out_filter->command_index));
  IREE_RETURN_IF_ERROR(iree_hal_profile_capture_filter_set_u32_from_flag(
      FLAG_device_profiling_filter_physical_device,
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL,
      "device_profiling_filter_physical_device", out_filter,
      &out_filter->physical_device_ordinal));
  IREE_RETURN_IF_ERROR(iree_hal_profile_capture_filter_set_u32_from_flag(
      FLAG_device_profiling_filter_queue,
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL,
      "device_profiling_filter_queue", out_filter, &out_filter->queue_ordinal));
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_executable_trace_options_from_flags(
    iree_hal_device_profiling_data_families_t data_families,
    iree_hal_profile_executable_trace_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  const int64_t values[] = {
      FLAG_device_profiling_trace_target_cu,
      FLAG_device_profiling_trace_simd_select,
      FLAG_device_profiling_trace_shader_engine_mask,
      FLAG_device_profiling_trace_max_traces,
  };
  bool any = FLAG_device_profiling_trace_buffer_size != 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    if (values[i] < 0 || values[i] > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--device_profiling_trace_* values must fit "
                              "uint32_t");
    }
    any |= values[i] != 0;
  }
  if (FLAG_device_profiling_trace_buffer_size < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_trace_buffer_size must be non-negative");
  }
  if (any &&
      !iree_any_bit_set(data_families,
                        IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device_profiling_trace_* requires "
                            "--device_profiling_mode=executable-traces");
  }
  out_options->target_compute_unit =
      (uint32_t)FLAG_device_profiling_trace_target_cu;
  out_options->simd_select = (uint32_t)FLAG_device_profiling_trace_simd_select;
  out_options->shader_engine_mask =
      (uint32_t)FLAG_device_profiling_trace_shader_engine_mask;
  out_options->max_trace_count =
      (uint32_t)FLAG_device_profiling_trace_max_traces;
  out_options->buffer_byte_length =
      (uint64_t)FLAG_device_profiling_trace_buffer_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_counter_sample_options_from_flags(
    iree_hal_device_profiling_data_families_t data_families,
    iree_hal_profile_counter_sample_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  if (FLAG_device_profiling_counter_max_samples < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_counter_max_samples must be non-negative");
  }
  if (FLAG_device_profiling_counter_max_samples &&
      !iree_any_bit_set(data_families,
                        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device_profiling_counter_max_samples requires "
                            "--device_profiling_mode=counters");
  }
  out_options->max_sample_count =
      (uint64_t)FLAG_device_profiling_counter_max_samples;
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_sink_create_from_flags(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;

  if (strlen(FLAG_device_profiling_output) == 0) {
    return iree_ok_status();
  }

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_create(
      IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN |
          IREE_IO_FILE_MODE_SHARE_READ,
      iree_make_cstring_view(FLAG_device_profiling_output),
      /*initial_size=*/0, host_allocator, &file_handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_sink_create(file_handle, host_allocator,
                                               out_sink);
  }
  iree_io_file_handle_release(file_handle);
  return status;
}

static iree_status_t iree_hal_device_profiling_data_families_from_flags(
    iree_hal_device_profiling_data_families_t* out_data_families) {
  *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  iree_string_view_t remaining =
      iree_string_view_trim(iree_make_cstring_view(FLAG_device_profiling_mode));
  while (remaining.size != 0) {
    iree_string_view_t family_part = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &family_part, &remaining);
    family_part = iree_string_view_trim(family_part);
    if (family_part.size == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--device_profiling_mode contains an empty data "
                              "family");
    } else if (iree_string_view_equal(family_part, IREE_SV("queue-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("host-execution"))) {
      *out_data_families |=
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("device-queue-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("dispatch-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("memory-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("device-metrics"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("command-region-events"))) {
      *out_data_families |=
          IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("counters"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
    } else if (iree_string_view_equal(family_part, IREE_SV("counter-ranges")) ||
               iree_string_view_equal(family_part, IREE_SV("pmc-ranges"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("executable-metadata"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("executable-traces"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported profiling data family '%.*s'",
                              (int)family_part.size, family_part.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

iree_status_t
iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
    bool* out_required) {
  IREE_ASSERT_ARGUMENT(out_required);
  *out_required = false;

  iree_hal_device_profiling_data_families_t data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_profiling_data_families_from_flags(&data_families));

  if (data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    *out_required = FLAG_print_device_statistics;
    return iree_ok_status();
  }

  const iree_hal_device_profiling_data_families_t command_buffer_data =
      IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES |
      IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS;
  *out_required = iree_any_bit_set(data_families, command_buffer_data);
  return iree_ok_status();
}

static bool iree_hal_device_profiling_filter_flags_present(void) {
  return strlen(FLAG_device_profiling_filter_function) != 0 ||
         FLAG_device_profiling_filter_command_buffer >= 0 ||
         FLAG_device_profiling_filter_command_index >= 0 ||
         FLAG_device_profiling_filter_physical_device >= 0 ||
         FLAG_device_profiling_filter_queue >= 0;
}

static bool iree_hal_profiling_from_flags_any_profile_active(
    const iree_hal_profiling_from_flags_t* profiling) {
  for (iree_host_size_t i = 0; i < profiling->device_count; ++i) {
    if (profiling->profile_active[i]) {
      return true;
    }
  }
  return false;
}

static bool iree_hal_profiling_from_flags_flush_should_stop(void* arg) {
  iree_hal_profiling_from_flags_t* profiling =
      (iree_hal_profiling_from_flags_t*)arg;
  return iree_atomic_load(&profiling->flush_stop_requested,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_profiling_from_flags_record_flush_status(
    iree_hal_profiling_from_flags_t* profiling, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_slim_mutex_lock(&profiling->flush_status_mutex);
  profiling->flush_status = iree_status_join(profiling->flush_status, status);
  iree_slim_mutex_unlock(&profiling->flush_status_mutex);
}

static iree_status_t iree_hal_profiling_from_flags_consume_flush_status(
    iree_hal_profiling_from_flags_t* profiling) {
  iree_slim_mutex_lock(&profiling->flush_status_mutex);
  iree_status_t status = profiling->flush_status;
  profiling->flush_status = iree_ok_status();
  iree_slim_mutex_unlock(&profiling->flush_status_mutex);
  return status;
}

iree_status_t iree_hal_flush_profiling_from_flags(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling ||
      !iree_hal_profiling_from_flags_any_profile_active(profiling)) {
    return iree_ok_status();
  }
  iree_slim_mutex_lock(&profiling->flush_mutex);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < profiling->device_count; ++i) {
    if (profiling->profile_active[i]) {
      status = iree_status_join(
          status, iree_hal_device_profiling_flush(profiling->devices[i]));
    }
  }
  iree_slim_mutex_unlock(&profiling->flush_mutex);
  return status;
}

static int iree_hal_profiling_from_flags_flush_thread_main(void* arg) {
  iree_hal_profiling_from_flags_t* profiling =
      (iree_hal_profiling_from_flags_t*)arg;

  while (!iree_hal_profiling_from_flags_flush_should_stop(profiling)) {
    const bool should_stop = iree_notification_await(
        &profiling->flush_notification,
        iree_hal_profiling_from_flags_flush_should_stop, profiling,
        iree_make_timeout_ms(profiling->flush_interval_ms));
    if (should_stop) {
      break;
    }

    iree_status_t status = iree_hal_flush_profiling_from_flags(profiling);
    if (!iree_status_is_ok(status)) {
      iree_hal_profiling_from_flags_record_flush_status(profiling, status);
      break;
    }
  }

  return 0;
}

static iree_status_t iree_hal_profiling_from_flags_start_periodic_flush(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!iree_hal_profiling_from_flags_any_profile_active(profiling) ||
      profiling->flush_interval_ms == 0) {
    return iree_ok_status();
  }

  iree_notification_initialize(&profiling->flush_notification);
  iree_slim_mutex_initialize(&profiling->flush_status_mutex);
  iree_atomic_store(&profiling->flush_stop_requested, 0,
                    iree_memory_order_relaxed);

  const iree_thread_create_params_t thread_params = {
      .name = IREE_SV("iree-hal-profile-flush"),
      .priority_class = IREE_THREAD_PRIORITY_CLASS_LOW,
  };
  iree_status_t status = iree_thread_create(
      iree_hal_profiling_from_flags_flush_thread_main, profiling, thread_params,
      profiling->host_allocator, &profiling->flush_thread);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_deinitialize(&profiling->flush_status_mutex);
    iree_notification_deinitialize(&profiling->flush_notification);
  }
  return status;
}

static iree_status_t iree_hal_profiling_from_flags_stop_periodic_flush(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling || !profiling->flush_thread) {
    return iree_ok_status();
  }

  iree_atomic_store(&profiling->flush_stop_requested, 1,
                    iree_memory_order_release);
  iree_notification_post(&profiling->flush_notification, IREE_ALL_WAITERS);
  // Final release joins the thread. Calling iree_thread_join before release
  // would double-join pthread-backed threads, which is undefined behavior.
  iree_thread_release(profiling->flush_thread);
  profiling->flush_thread = NULL;

  iree_status_t status =
      iree_hal_profiling_from_flags_consume_flush_status(profiling);
  iree_slim_mutex_deinitialize(&profiling->flush_status_mutex);
  iree_notification_deinitialize(&profiling->flush_notification);
  return status;
}

static void iree_hal_profiling_from_flags_release_devices(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling) {
    return;
  }
  for (iree_host_size_t i = 0; i < profiling->device_count; ++i) {
    iree_hal_device_release(profiling->devices[i]);
  }
  iree_allocator_free(profiling->host_allocator, profiling->profile_active);
  iree_allocator_free(profiling->host_allocator, profiling->devices);
}

// True when |status| is a rejection --device_profiling_allow_partial may
// answer by dropping data families: the device does not implement a family,
// or it has none of the counters the caller named. Counter names are
// device-specific while the flag applies to every device of a group.
static bool iree_hal_profiling_from_flags_is_partial_rejection(
    iree_status_t status, const iree_hal_device_profiling_options_t* options) {
  return iree_status_is_unimplemented(status) ||
         (iree_status_is_unavailable(status) && options->counter_set_count &&
          iree_hal_device_profiling_options_requests_counters(options));
}

// Retries |device| with the largest subset of the requested data families it
// implements. Drivers validate families before touching the sink, so rejected
// attempts leave no session behind; the first accepted subset is final.
// Subsets are tried in decreasing size, so the result is a maximum-size subset.
// Consumes |unimplemented_status|. An empty subset leaves the device
// unprofiled and returns OK with |out_options->data_families| cleared.
static iree_status_t iree_hal_profiling_from_flags_begin_partial(
    iree_hal_device_t* device, iree_host_size_t device_index,
    const iree_hal_device_profiling_options_t* options,
    iree_status_t unimplemented_status,
    iree_hal_device_profiling_options_t* out_options) {
  uint64_t family_bits[64];
  uint32_t family_count = 0;
  for (uint32_t bit = 0; bit < 64; ++bit) {
    if (options->data_families & (1ull << bit)) {
      family_bits[family_count++] = 1ull << bit;
    }
  }
  if (family_count == 0 || family_count > 16) {
    return unimplemented_status;
  }
  const uint32_t subset_count = 1u << family_count;
  for (uint32_t kept = family_count - 1; kept > 0; --kept) {
    for (uint32_t subset = subset_count - 1; subset > 0; --subset) {
      uint32_t ones = 0;
      uint64_t families = 0;
      for (uint32_t j = 0; j < family_count; ++j) {
        if (subset & (1u << j)) {
          ++ones;
          families |= family_bits[j];
        }
      }
      if (ones != kept) {
        continue;
      }
      *out_options = *options;
      out_options->data_families = families;
      if (!iree_hal_device_profiling_options_requests_counters(out_options)) {
        // A selection without its family is an invalid request, not a subset.
        out_options->counter_set_count = 0;
        out_options->counter_sets = NULL;
      }
      iree_status_t status = iree_hal_device_profiling_begin(device, out_options);
      if (iree_status_is_ok(status)) {
        iree_status_ignore(unimplemented_status);
        fprintf(stderr,
                "device_profiling: device %" PRIhsz
                " keeps data families 0x%" PRIx64 " of requested 0x%" PRIx64
                " (--device_profiling_allow_partial)\n",
                device_index, (uint64_t)families,
                (uint64_t)options->data_families);
        return iree_ok_status();
      }
      if (!iree_hal_profiling_from_flags_is_partial_rejection(status,
                                                              out_options)) {
        iree_status_ignore(unimplemented_status);
        return status;
      }
      iree_status_ignore(status);
    }
  }
  iree_status_ignore(unimplemented_status);
  *out_options = *options;
  out_options->data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  fprintf(stderr,
          "device_profiling: device %" PRIhsz
          " implements none of the requested data families 0x%" PRIx64
          "; left unprofiled (--device_profiling_allow_partial)\n",
          device_index, (uint64_t)options->data_families);
  return iree_ok_status();
}

static iree_status_t iree_hal_begin_device_list_profiling_from_flags(
    iree_host_size_t device_count, iree_hal_device_t* const* devices,
    iree_allocator_t host_allocator,
    iree_hal_profiling_from_flags_t** out_profiling) {
  IREE_ASSERT_ARGUMENT(out_profiling);
  *out_profiling = NULL;
  if (device_count == 0) {
    return iree_ok_status();
  }

  const iree_flag_string_list_t counter_names =
      FLAG_device_profiling_counter_list();
  iree_hal_device_profiling_options_t options = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_profiling_data_families_from_flags(
      &options.data_families));
  const bool statistics_requested = FLAG_print_device_statistics;
  if (statistics_requested && strlen(FLAG_device_profiling_output) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--print_device_statistics cannot be combined with "
                            "--device_profiling_output");
  }
  if (FLAG_device_profiling_flush_interval_ms < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_flush_interval_ms must be non-negative");
  }
  if (options.data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    if (counter_names.count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--device_profiling_counter requires "
                              "--device_profiling_mode=counters or "
                              "--device_profiling_mode=counter-ranges");
    }
    if (strlen(FLAG_device_profiling_output) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--device_profiling_output requires --device_profiling_mode");
    }
    if (iree_hal_device_profiling_filter_flags_present() &&
        !statistics_requested) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--device_profiling_filter_* requires --device_profiling_mode or "
          "--print_device_statistics");
    }
    if (FLAG_device_profiling_flush_interval_ms != 0 && !statistics_requested) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--device_profiling_flush_interval_ms requires "
                              "--device_profiling_mode or "
                              "--print_device_statistics");
    }
    if (statistics_requested) {
      options.flags |= IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
    } else {
      return iree_ok_status();
    }
  }
  if (counter_names.count != 0 &&
      !iree_any_bit_set(options.data_families,
                        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                            IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_counter requires --device_profiling_mode=counters "
        "or --device_profiling_mode=counter-ranges");
  }
  if (options.data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE &&
      strlen(FLAG_device_profiling_output) == 0 && !statistics_requested) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device_profiling_mode requires "
                            "--device_profiling_output");
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_profile_capture_filter_from_flags(&options.capture_filter));
  IREE_RETURN_IF_ERROR(iree_hal_profile_executable_trace_options_from_flags(
      options.data_families, &options.executable_traces));
  IREE_RETURN_IF_ERROR(iree_hal_profile_counter_sample_options_from_flags(
      options.data_families, &options.counter_samples));
  iree_hal_profile_counter_set_selection_t counter_set = {0};
  if (counter_names.count != 0) {
    counter_set.counter_name_count = counter_names.count;
    counter_set.counter_names = counter_names.values;
    options.counter_set_count = 1;
    options.counter_sets = &counter_set;
  }

  iree_hal_profiling_from_flags_t* profiling = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*profiling),
                                             (void**)&profiling));
  memset(profiling, 0, sizeof(*profiling));
  profiling->host_allocator = host_allocator;
  profiling->flush_interval_ms = FLAG_device_profiling_flush_interval_ms;
  iree_slim_mutex_initialize(&profiling->flush_mutex);

  iree_host_size_t devices_size = 0;
  iree_host_size_t active_size = 0;
  bool allocation_sizes_valid =
      iree_host_size_checked_mul(device_count, sizeof(*profiling->devices),
                                 &devices_size) &&
      iree_host_size_checked_mul(
          device_count, sizeof(*profiling->profile_active), &active_size);
  if (IREE_UNLIKELY(!allocation_sizes_valid)) {
    iree_slim_mutex_deinitialize(&profiling->flush_mutex);
    iree_allocator_free(host_allocator, profiling);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profiling device list is too large");
  }

  iree_status_t status = iree_allocator_malloc(host_allocator, devices_size,
                                               (void**)&profiling->devices);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, active_size,
                                   (void**)&profiling->profile_active);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_profiling_from_flags_release_devices(profiling);
    iree_slim_mutex_deinitialize(&profiling->flush_mutex);
    iree_allocator_free(host_allocator, profiling);
    return status;
  }
  memset(profiling->devices, 0, devices_size);
  memset(profiling->profile_active, 0, active_size);
  profiling->device_count = device_count;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    if (!devices[i]) {
      iree_hal_profiling_from_flags_release_devices(profiling);
      iree_slim_mutex_deinitialize(&profiling->flush_mutex);
      iree_allocator_free(host_allocator, profiling);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "profiling device list contains a null device");
    }
    profiling->devices[i] = devices[i];
    iree_hal_device_retain(profiling->devices[i]);
  }

  iree_hal_profile_sink_t* sink = NULL;
  iree_hal_profile_statistics_sink_t* statistics_sink = NULL;
  if (statistics_requested) {
    status = iree_hal_profile_statistics_sink_create(host_allocator,
                                                     &statistics_sink);
    if (iree_status_is_ok(status)) {
      sink = iree_hal_profile_statistics_sink_base(statistics_sink);
    }
  } else {
    status = iree_hal_profile_sink_create_from_flags(host_allocator, &sink);
  }
  if (iree_status_is_ok(status)) {
    options.sink = sink;
    for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
         ++i) {
      iree_hal_device_profiling_options_t device_options = options;
      status = iree_hal_device_profiling_begin(profiling->devices[i],
                                               &device_options);
      if (FLAG_device_profiling_allow_partial &&
          iree_hal_profiling_from_flags_is_partial_rejection(status,
                                                             &device_options)) {
        status = iree_hal_profiling_from_flags_begin_partial(
            profiling->devices[i], i, &options, status, &device_options);
      }
      profiling->profile_active[i] =
          iree_status_is_ok(status) &&
          (device_options.data_families !=
               IREE_HAL_DEVICE_PROFILING_DATA_NONE ||
           iree_hal_device_profiling_options_requests_lightweight_statistics(
               &device_options));
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_profiling_from_flags_start_periodic_flush(profiling);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_profiling_from_flags_stop_periodic_flush(profiling));
    for (iree_host_size_t i = profiling->device_count; i > 0; --i) {
      if (profiling->profile_active[i - 1]) {
        status = iree_status_join(
            status, iree_hal_device_profiling_end(profiling->devices[i - 1]));
      }
    }
    iree_hal_profiling_from_flags_release_devices(profiling);
    iree_slim_mutex_deinitialize(&profiling->flush_mutex);
    iree_allocator_free(host_allocator, profiling);
  } else {
    profiling->statistics_sink = statistics_sink;
    statistics_sink = NULL;
    *out_profiling = profiling;
  }
  if (statistics_sink) {
    iree_hal_profile_statistics_sink_release(statistics_sink);
  } else if (!statistics_requested) {
    iree_hal_profile_sink_release(sink);
  }
  return status;
}

iree_status_t iree_hal_begin_profiling_from_flags(
    iree_hal_device_t* device, iree_allocator_t host_allocator,
    iree_hal_profiling_from_flags_t** out_profiling) {
  IREE_ASSERT_ARGUMENT(out_profiling);
  *out_profiling = NULL;
  if (!device) {
    return iree_ok_status();
  }
  return iree_hal_begin_device_list_profiling_from_flags(
      1, &device, host_allocator, out_profiling);
}

iree_status_t iree_hal_begin_device_group_profiling_from_flags(
    iree_hal_device_group_t* device_group, iree_allocator_t host_allocator,
    iree_hal_profiling_from_flags_t** out_profiling) {
  IREE_ASSERT_ARGUMENT(out_profiling);
  *out_profiling = NULL;
  if (!device_group) {
    return iree_ok_status();
  }
  iree_host_size_t device_count =
      iree_hal_device_group_device_count(device_group);
  if (device_count == 0) {
    return iree_ok_status();
  }

  iree_hal_device_t** devices = NULL;
  iree_host_size_t devices_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(device_count, sizeof(*devices),
                                                &devices_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profiling device group is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, devices_size, (void**)&devices));
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    devices[i] = iree_hal_device_group_device_at(device_group, i);
  }
  iree_status_t status = iree_hal_begin_device_list_profiling_from_flags(
      device_count, devices, host_allocator, out_profiling);
  iree_allocator_free(host_allocator, devices);
  return status;
}

iree_status_t iree_hal_end_profiling_from_flags(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling) {
    return iree_ok_status();
  }

  iree_status_t status =
      iree_hal_profiling_from_flags_stop_periodic_flush(profiling);
  for (iree_host_size_t i = profiling->device_count; i > 0; --i) {
    if (profiling->profile_active[i - 1]) {
      status = iree_status_join(
          status, iree_hal_device_profiling_end(profiling->devices[i - 1]));
    }
  }
  if (profiling->statistics_sink) {
    status = iree_status_join(status, iree_hal_profile_statistics_sink_fprint(
                                          stderr, profiling->statistics_sink));
    iree_hal_profile_statistics_sink_release(profiling->statistics_sink);
  }
  iree_hal_profiling_from_flags_release_devices(profiling);
  iree_slim_mutex_deinitialize(&profiling->flush_mutex);
  iree_allocator_free(profiling->host_allocator, profiling);
  return status;
}
