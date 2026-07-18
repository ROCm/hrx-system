// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Global runtime state management. Shared infrastructure (VM instance) is
// created on first accelerator init and destroyed when last shuts down.
// Device creation follows the proven pattern from PyTorch's hrx backend:
// driver-based creation via iree_hal_task_driver_create +
// iree_hal_driver_create_default_device.

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hrx_internal.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_handle.h"
#include "iree/modules/hal/types.h"
#include "iree/task/api.h"

#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"
#endif

#ifdef HRX_HAS_IREE_REMOTE_DRIVER
#include "iree/base/internal/path.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/client/registration/driver_module.h"
#endif

//===----------------------------------------------------------------------===//
// Global singletons
//===----------------------------------------------------------------------===//

static hrx_shared_state_t g_shared = {0};
static hrx_gpu_state_t g_gpu = {0};
static hrx_cpu_state_t g_cpu = {0};
static hrx_device_event_sink_t g_device_event_sink = {0};

static_assert(HRX_DEVICE_EVENT_ABI_VERSION_0 ==
                  IREE_HAL_DEVICE_EVENT_ABI_VERSION_0,
              "device event ABI version mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_TYPE_DRIVER_FAILURE ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE,
              "device event type mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_TYPE_ASAN_REPORT ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT,
              "device event type mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_TYPE_UBSAN_REPORT ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT,
              "device event type mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_TYPE_PRINTF ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_TYPE_PRINTF,
              "device event type mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_TYPE_HOST_CALL ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_TYPE_HOST_CALL,
              "device event type mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_TYPE_USER ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_TYPE_USER,
              "device event type mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_SEVERITY_TRACE ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_SEVERITY_TRACE,
              "device event severity mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_SEVERITY_INFO ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_SEVERITY_INFO,
              "device event severity mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_SEVERITY_WARNING ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_SEVERITY_WARNING,
              "device event severity mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_SEVERITY_ERROR ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR,
              "device event severity mismatch");
static_assert((uint32_t)HRX_DEVICE_EVENT_SEVERITY_FATAL ==
                  (uint32_t)IREE_HAL_DEVICE_EVENT_SEVERITY_FATAL,
              "device event severity mismatch");
static_assert(HRX_DEVICE_PRINTF_EVENT_ABI_VERSION_0 ==
                  IREE_HAL_DEVICE_PRINTF_EVENT_ABI_VERSION_0,
              "printf event ABI version mismatch");
static_assert((uint32_t)HRX_DEVICE_PRINTF_STREAM_DEFAULT ==
                  (uint32_t)IREE_HAL_DEVICE_PRINTF_STREAM_DEFAULT,
              "printf stream mismatch");
static_assert((uint32_t)HRX_DEVICE_PRINTF_STREAM_STDOUT ==
                  (uint32_t)IREE_HAL_DEVICE_PRINTF_STREAM_STDOUT,
              "printf stream mismatch");
static_assert((uint32_t)HRX_DEVICE_PRINTF_STREAM_STDERR ==
                  (uint32_t)IREE_HAL_DEVICE_PRINTF_STREAM_STDERR,
              "printf stream mismatch");
static_assert((uint32_t)HRX_DEVICE_PRINTF_FLAG_NONE ==
                  (uint32_t)IREE_HAL_DEVICE_PRINTF_FLAG_NONE,
              "printf flags mismatch");
static_assert(sizeof(hrx_device_printf_event_t) ==
                  sizeof(iree_hal_device_printf_event_t),
              "printf event layout mismatch");
static_assert(offsetof(hrx_device_printf_event_t, text) ==
                  offsetof(iree_hal_device_printf_event_t, text),
              "printf event layout mismatch");
static_assert(offsetof(hrx_device_printf_event_t, arguments) ==
                  offsetof(iree_hal_device_printf_event_t, arguments),
              "printf event layout mismatch");
static_assert(HRX_DEVICE_ASAN_REPORT_ABI_VERSION_0 ==
                  IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0,
              "ASAN report ABI version mismatch");
static_assert((uint32_t)HRX_DEVICE_ASAN_ACCESS_KIND_UNKNOWN ==
                  (uint32_t)IREE_HAL_DEVICE_ASAN_ACCESS_KIND_UNKNOWN,
              "ASAN access kind mismatch");
static_assert((uint32_t)HRX_DEVICE_ASAN_ACCESS_KIND_READ ==
                  (uint32_t)IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ,
              "ASAN access kind mismatch");
static_assert((uint32_t)HRX_DEVICE_ASAN_ACCESS_KIND_WRITE ==
                  (uint32_t)IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE,
              "ASAN access kind mismatch");
static_assert((uint32_t)HRX_DEVICE_ASAN_ACCESS_KIND_ATOMIC ==
                  (uint32_t)IREE_HAL_DEVICE_ASAN_ACCESS_KIND_ATOMIC,
              "ASAN access kind mismatch");
static_assert(sizeof(hrx_device_asan_report_t) ==
                  sizeof(iree_hal_device_asan_report_t),
              "ASAN report layout mismatch");
static_assert(offsetof(hrx_device_asan_report_t, source_dispatch_ptr) ==
                  offsetof(iree_hal_device_asan_report_t, source_dispatch_ptr),
              "ASAN report layout mismatch");

hrx_shared_state_t* hrx_get_shared_state(void) { return &g_shared; }
hrx_gpu_state_t* hrx_get_gpu_state(void) { return &g_gpu; }
hrx_cpu_state_t* hrx_get_cpu_state(void) { return &g_cpu; }

static hrx_string_view_t hrx_string_view_from_iree(iree_string_view_t value) {
  hrx_string_view_t view;
  view.data = value.data;
  view.size = value.size;
  return view;
}

static hrx_const_byte_span_t hrx_const_byte_span_from_iree(
    iree_const_byte_span_t value) {
  hrx_const_byte_span_t span;
  span.data = value.data;
  span.data_length = value.data_length;
  return span;
}

static void hrx_hal_device_event_sink_thunk(
    void* user_data, const iree_hal_device_event_t* hal_event) {
  const hrx_device_event_sink_t* sink =
      (const hrx_device_event_sink_t*)user_data;
  hrx_device_event_t event;
  memset(&event, 0, sizeof(event));
  event.record_length = sizeof(event);
  event.abi_version = HRX_DEVICE_EVENT_ABI_VERSION_0;
  event.type = (hrx_device_event_type_t)hal_event->type;
  event.severity = (hrx_device_event_severity_t)hal_event->severity;
  event.flags = (hrx_device_event_flags_t)hal_event->flags;
  event.sequence = hal_event->sequence;
  event.host_time_ns = hal_event->host_time_ns;
  event.source.device_id =
      hrx_string_view_from_iree(hal_event->source.device_id);
  event.source.driver_id =
      hrx_string_view_from_iree(hal_event->source.driver_id);
  event.source.physical_device_ordinal =
      hal_event->source.physical_device_ordinal;
  event.source.queue_ordinal = hal_event->source.queue_ordinal;
  event.source.executable_id = hal_event->source.executable_id;
  event.source.export_ordinal = hal_event->source.export_ordinal;
  event.payload = hrx_const_byte_span_from_iree(hal_event->payload);
  event.implementation_payload =
      hrx_const_byte_span_from_iree(hal_event->implementation_payload);
  sink->fn(sink->user_data, &event);
}

static iree_hal_device_event_sink_t hrx_hal_device_event_sink(void) {
  if (!g_device_event_sink.fn) {
    return iree_hal_device_event_sink_discard();
  }
  iree_hal_device_event_sink_t sink;
  sink.fn = hrx_hal_device_event_sink_thunk;
  sink.user_data = &g_device_event_sink;
  return sink;
}

bool hrx_runtime_try_get_hal_device_event_sink(
    iree_hal_device_event_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  memset(out_sink, 0, sizeof(*out_sink));
  if (!g_device_event_sink.fn) return false;
  *out_sink = hrx_hal_device_event_sink();
  return true;
}

hrx_status_t hrx_runtime_set_device_event_sink(hrx_device_event_sink_t sink) {
  if (!sink.fn) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device event sink callback is NULL");
  }
  if (g_shared.shared_initialized || g_gpu.initialized || g_cpu.initialized) {
    return hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                           "device event sink must be set before accelerator "
                           "initialization");
  }
  g_device_event_sink = sink;
  return hrx_ok_status();
}

static iree_status_t hrx_create_single_device_group(
    iree_hal_device_t* device, iree_allocator_t host_allocator,
    iree_hal_device_group_t** out_device_group) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_device_group);
  *out_device_group = NULL;

  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), host_allocator,
      &frontier_tracker));

  iree_status_t status = iree_hal_device_group_create_from_device(
      device, frontier_tracker, host_allocator, out_device_group);
  iree_async_frontier_tracker_release(frontier_tracker);
  return status;
}

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

void hrx_runtime_version(int* major, int* minor, int* patch) {
  if (major) *major = HRX_VERSION_MAJOR;
  if (minor) *minor = HRX_VERSION_MINOR;
  if (patch) *patch = HRX_VERSION_PATCH;
}

//===----------------------------------------------------------------------===//
// Shared state init/teardown
//===----------------------------------------------------------------------===//

hrx_status_t hrx_ensure_shared_state(void) {
  if (g_shared.shared_initialized) {
    g_shared.init_count++;
    return hrx_ok_status();
  }
  g_shared.host_allocator = iree_allocator_system();

  iree_status_t status =
      iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                              g_shared.host_allocator, &g_shared.vm_instance);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  status = iree_hal_module_register_all_types(g_shared.vm_instance);
  if (!iree_status_is_ok(status)) {
    iree_vm_instance_release(g_shared.vm_instance);
    g_shared.vm_instance = NULL;
    return hrx_status_from_iree(status);
  }

  // Create proactor pool for async I/O (required by local-task devices).
  uint32_t node_id = 0;
  status = iree_async_proactor_pool_create(
      /*node_count=*/1, &node_id, iree_async_proactor_pool_options_default(),
      g_shared.host_allocator, &g_shared.proactor_pool);
  if (!iree_status_is_ok(status)) {
    iree_vm_instance_release(g_shared.vm_instance);
    g_shared.vm_instance = NULL;
    return hrx_status_from_iree(status);
  }

  g_shared.shared_initialized = true;
  g_shared.init_count = 1;
  return hrx_ok_status();
}

static iree_status_t hrx_set_gpu_architecture_from_hal(
    iree_hal_device_t* hal_device, hrx_device_s* dev) {
  const iree_hal_device_spec_t* device_spec = iree_hal_device_spec(hal_device);
  iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
  };
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec does not report an exact executable target");
  } else if (result.outcome ==
             IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec reports ambiguous exact executable targets");
  }

  const iree_string_view_t target_key = result.target->target_key;
  if (iree_string_view_is_empty(target_key)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec exact target has an empty target key");
  }
  if (target_key.size >= sizeof(dev->architecture)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU target key length %" PRIhsz
                            " exceeds HRX storage capacity",
                            target_key.size);
  }
  memcpy(dev->architecture, target_key.data, target_key.size);
  dev->architecture[target_key.size] = 0;
  return iree_ok_status();
}

static void hrx_release_shared_state(void) {
  if (!g_shared.shared_initialized) return;
  g_shared.init_count--;
  if (g_shared.init_count > 0) return;

  iree_async_proactor_pool_release(g_shared.proactor_pool);
  g_shared.proactor_pool = NULL;
  iree_vm_instance_release(g_shared.vm_instance);
  g_shared.vm_instance = NULL;
  g_shared.shared_initialized = false;
}

//===----------------------------------------------------------------------===//
// Helper: create a local-task device via driver pattern
//===----------------------------------------------------------------------===//

// Creates a local-task HAL device using the driver-based creation pattern.
// This matches the proven path in PyTorch's HrxRuntime::initialize().
// group_count controls the task executor parallelism.
static hrx_status_t hrx_create_local_task_device(
    int group_count, iree_task_executor_t** out_executor,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_hal_device) {
  iree_allocator_t alloc = g_shared.host_allocator;

  // Task topology + executor.
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  iree_task_topology_initialize_from_group_count(group_count, &topology);

  iree_task_executor_options_t exec_options;
  iree_task_executor_options_initialize(&exec_options);
  // GPU runtimes may add TLS that raises the effective minimum pthread stack
  // size from 16KB. Use 256KB which is safe for ASAN builds too.
  exec_options.worker_stack_size = 256 * 1024;

  iree_task_executor_t* executor = NULL;
  iree_status_t status =
      iree_task_executor_create(exec_options, &topology, alloc, &executor);
  iree_task_topology_deinitialize(&topology);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  // Executable loaders.
  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  status = iree_hal_create_all_available_executable_loaders(
      /*plugin_manager=*/NULL, IREE_ARRAYSIZE(loaders), &loader_count, loaders,
      alloc);
  if (!iree_status_is_ok(status)) {
    iree_task_executor_release(executor);
    return hrx_status_from_iree(status);
  }

  // Heap allocator for host-accessible buffers.
  iree_hal_allocator_t* device_allocator = NULL;
  status = iree_hal_allocator_create_heap(iree_make_cstring_view("hrx"), alloc,
                                          alloc, &device_allocator);
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < loader_count; i++)
      iree_hal_executable_loader_release(loaders[i]);
    iree_task_executor_release(executor);
    return hrx_status_from_iree(status);
  }

  // Assemble the local-task driver.
  iree_hal_task_device_params_t task_params;
  iree_hal_task_device_params_initialize(&task_params);

  iree_hal_driver_t* driver = NULL;
  status = iree_hal_task_driver_create(
      iree_make_cstring_view("local-task"), &task_params,
      /*queue_count=*/1, &executor, loader_count, loaders, device_allocator,
      alloc, &driver);

  // Driver takes ownership references; release ours.
  iree_task_executor_release(executor);
  for (iree_host_size_t i = 0; i < loader_count; i++)
    iree_hal_executable_loader_release(loaders[i]);
  iree_hal_allocator_release(device_allocator);

  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  // Create device from driver. Must provide proactor pool.
  iree_hal_device_create_params_t device_params =
      iree_hal_device_create_params_default();
  device_params.proactor_pool = g_shared.proactor_pool;
  device_params.event_sink = hrx_hal_device_event_sink();

  iree_hal_device_t* hal_device = NULL;
  status = iree_hal_driver_create_default_device(driver, &device_params, alloc,
                                                 &hal_device);
  if (!iree_status_is_ok(status)) {
    iree_hal_driver_release(driver);
    return hrx_status_from_iree(status);
  }

  // Re-create executor for caller tracking (driver took ownership of original).
  iree_task_topology_t out_topology;
  iree_task_topology_initialize(&out_topology);
  iree_task_topology_initialize_from_group_count(group_count, &out_topology);
  // Note: we don't need a separate executor for shutdown tracking.
  // The driver owns the executor internally. Set output to NULL.
  iree_task_topology_deinitialize(&out_topology);

  *out_executor = NULL;  // Driver manages executor lifetime.
  *out_driver = driver;
  *out_hal_device = hal_device;
  return hrx_ok_status();
}

static void hrx_copy_string_view_to_cstr(iree_string_view_t source,
                                         char* target,
                                         iree_host_size_t target_capacity) {
  if (target_capacity == 0) return;
  iree_host_size_t length = source.size;
  if (length >= target_capacity) length = target_capacity - 1;
  if (length > 0) memcpy(target, source.data, length);
  target[length] = '\0';
}

static iree_status_t hrx_initialize_device_from_hal(
    hrx_device_s* dev, hrx_accelerator_type_t type, int ordinal,
    iree_hal_device_t* hal_device, iree_string_view_t name,
    iree_string_view_t architecture, iree_allocator_t host_allocator) {
  iree_hal_device_group_t* device_group = NULL;
  IREE_RETURN_IF_ERROR(hrx_create_single_device_group(
      hal_device, host_allocator, &device_group));

  memset(dev, 0, sizeof(*dev));
  iree_atomic_ref_count_init(&dev->ref_count);
  dev->type = type;
  dev->ordinal = ordinal;
  dev->hal_device = hal_device;
  dev->hal_device_group = device_group;
  dev->allocator.hal_allocator = iree_hal_device_allocator(hal_device);
  iree_hal_allocator_retain(dev->allocator.hal_allocator);
  iree_atomic_ref_count_init(&dev->allocator.ref_count);
  dev->allocator.device = dev;
  hrx_buffer_table_initialize(&dev->buffer_table);
  iree_arena_block_pool_initialize(/*block_size=*/32 * 1024,
                                   iree_allocator_system(), &dev->block_pool);
  hrx_copy_string_view_to_cstr(name, dev->name, sizeof(dev->name));
  hrx_copy_string_view_to_cstr(architecture, dev->architecture,
                               sizeof(dev->architecture));
  return iree_ok_status();
}

static void hrx_release_initialized_device(hrx_device_s* dev) {
  if (!dev || !dev->hal_device) return;
  hrx_buffer_table_deinitialize(&dev->buffer_table);
  iree_arena_block_pool_deinitialize(&dev->block_pool);
  hrx_device_release(dev);
}

static const char* hrx_get_gpu_driver_name(void) {
  const char* value = getenv("HRX_GPU_DRIVER");
  return (value && value[0]) ? value : "amdgpu";
}

static bool hrx_getenv_enabled(const char* name) {
  const char* value = getenv(name);
  return value && value[0] && strcmp(value, "0") != 0;
}

static bool hrx_gpu_debug_enabled(void) {
  return hrx_getenv_enabled("HRX_GPU_DEBUG");
}

static const char* hrx_get_profile_file_path(void) {
  const char* value = getenv("HRX_PROFILE_FILE");
  return (value && value[0]) ? value : NULL;
}

static iree_status_t hrx_hal_runtime_features_from_environment(
    iree_hal_device_runtime_feature_flags_t* out_runtime_features) {
  *out_runtime_features = IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  const char* sanitizer = getenv("HRX_HAL_SANITIZER");
  if (!sanitizer || !sanitizer[0]) return iree_ok_status();

  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK;
  if (strcmp(sanitizer, "asan") == 0) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN;
  } else if (strcmp(sanitizer, "tsan") == 0) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN;
  } else if (strcmp(sanitizer, "asan,tsan") == 0 ||
             strcmp(sanitizer, "tsan,asan") == 0) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN |
                        IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported HRX_HAL_SANITIZER '%s'", sanitizer);
  }
  *out_runtime_features = runtime_features;
  return iree_ok_status();
}

static iree_status_t hrx_profile_file_sink_create(
    const char* file_path, iree_allocator_t host_allocator,
    iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;
  if (!file_path || !file_path[0]) {
    return iree_ok_status();
  }

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_create(
      IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN |
          IREE_IO_FILE_MODE_SHARE_READ,
      iree_make_cstring_view(file_path), /*initial_size=*/0, host_allocator,
      &file_handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_sink_create(file_handle, host_allocator,
                                               out_sink);
  }
  iree_io_file_handle_release(file_handle);
  return status;
}

static iree_status_t hrx_get_profile_data_families(
    iree_hal_device_profiling_data_families_t* out_data_families) {
  IREE_ASSERT_ARGUMENT(out_data_families);

  const char* value = getenv("HRX_PROFILE_MODE");
  if (!value || !value[0] || strcmp(value, "queue") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
  } else if (strcmp(value, "dispatch") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                         IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  } else if (strcmp(value, "executable") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
  } else if (strcmp(value, "all") == 0) {
    *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                         IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported HRX_PROFILE_MODE '%s'", value);
  }

  return iree_ok_status();
}

static iree_status_t hrx_device_profile_begin(hrx_device_s* device,
                                              iree_hal_profile_sink_t* sink) {
  if (!device || !sink) {
    return iree_ok_status();
  }

  iree_hal_device_profiling_options_t options = {0};
  IREE_RETURN_IF_ERROR(hrx_get_profile_data_families(&options.data_families));
  options.sink = sink;
  iree_status_t status =
      iree_hal_device_profiling_begin(device->hal_device, &options);
  if (iree_status_is_ok(status)) {
    device->profiling_active = true;
  }
  return status;
}

static iree_status_t hrx_device_profile_end(hrx_device_s* device) {
  if (!device || !device->profiling_active || !device->hal_device) {
    return iree_ok_status();
  }

  iree_hal_semaphore_list_t empty = iree_hal_semaphore_list_empty();
  iree_status_t status = iree_hal_device_wait_semaphores(
      device->hal_device, IREE_ASYNC_WAIT_MODE_ALL, empty,
      iree_infinite_timeout(), /*flags=*/0);
  status = iree_status_join(status,
                            iree_hal_device_profiling_end(device->hal_device));
  device->profiling_active = false;
  return status;
}

static iree_status_t hrx_gpu_end_all_profiling(void) {
  iree_status_t status = iree_ok_status();
  for (int i = 0; i < g_gpu.device_count; ++i) {
    status =
        iree_status_join(status, hrx_device_profile_end(&g_gpu.devices[i]));
  }
  return status;
}

static void hrx_gpu_release_created_devices(int count) {
  for (int i = 0; i < count; ++i) {
    iree_status_ignore(hrx_device_profile_end(&g_gpu.devices[i]));
    hrx_release_initialized_device(&g_gpu.devices[i]);
  }
}

static void hrx_debug_print_iree_status(const char* label,
                                        const iree_status_t status) {
  if (!hrx_gpu_debug_enabled() || iree_status_is_ok(status)) return;
  iree_allocator_t allocator = iree_allocator_system();
  char* message = NULL;
  iree_host_size_t message_length = 0;
  if (iree_status_to_string(status, &allocator, &message, &message_length)) {
    fprintf(stderr, "hrx gpu debug: %s: %s\n", label,
            message ? message : "(no message)");
    iree_allocator_free(allocator, message);
  } else {
    fprintf(stderr, "hrx gpu debug: %s: (could not format status)\n", label);
  }
}

static bool hrx_is_remote_hal_uri(const char* value) {
  if (!value) return false;
  const char* separator = strstr(value, "://");
  return separator && strncmp(value, "remote-", 7) == 0;
}

#ifdef HRX_HAS_IREE_REMOTE_DRIVER
typedef struct hrx_remote_device_connect_state_t {
  // Notification posted after the remote connect callback stores |status|.
  iree_notification_t notification;
  // Connect completion status transferred from the proactor callback.
  iree_status_t status;
  // Non-zero after |status| has been populated.
  iree_atomic_int32_t complete;
} hrx_remote_device_connect_state_t;

typedef struct hrx_remote_device_deactivate_state_t {
  // Notification posted after the remote deactivation callback runs.
  iree_notification_t notification;
  // Non-zero after the remote device has drained all callbacks and endpoints.
  iree_atomic_int32_t complete;
} hrx_remote_device_deactivate_state_t;

static bool hrx_remote_device_connect_complete(void* user_data) {
  const hrx_remote_device_connect_state_t* state =
      (const hrx_remote_device_connect_state_t*)user_data;
  return iree_atomic_load(&state->complete, iree_memory_order_acquire) != 0;
}

static void hrx_remote_device_connected(void* user_data, iree_status_t status) {
  hrx_remote_device_connect_state_t* state =
      (hrx_remote_device_connect_state_t*)user_data;
  state->status = status;
  iree_atomic_store(&state->complete, 1, iree_memory_order_release);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

static bool hrx_remote_device_deactivate_complete(void* user_data) {
  const hrx_remote_device_deactivate_state_t* state =
      (const hrx_remote_device_deactivate_state_t*)user_data;
  return iree_atomic_load(&state->complete, iree_memory_order_acquire) != 0;
}

static void hrx_remote_device_deactivated(void* user_data) {
  hrx_remote_device_deactivate_state_t* state =
      (hrx_remote_device_deactivate_state_t*)user_data;
  iree_atomic_store(&state->complete, 1, iree_memory_order_release);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

static iree_status_t hrx_connect_remote_hal_device(
    iree_string_view_t device_uri, iree_hal_device_t* hal_device) {
  hrx_remote_device_connect_state_t connect_state;
  memset(&connect_state, 0, sizeof(connect_state));
  iree_notification_initialize(&connect_state.notification);

  iree_hal_remote_client_device_connected_callback_t callback = {
      .fn = hrx_remote_device_connected,
      .user_data = &connect_state,
  };
  iree_status_t status =
      iree_hal_remote_client_device_connect(hal_device, callback);
  if (iree_status_is_ok(status)) {
    const bool connected = iree_notification_await(
        &connect_state.notification, hrx_remote_device_connect_complete,
        &connect_state, iree_infinite_timeout());
    if (connected) {
      status = connect_state.status;
    } else {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "remote device connect did not complete");
    }
  }

  iree_notification_deinitialize(&connect_state.notification);
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        status,
        "connecting remote device '%.*s'; verify iree-serve-device is "
        "running and its --bind address is reachable from this client "
        "(the default is tcp://0.0.0.0:5000; use the server's reachable "
        "host address in the client URI)",
        (int)device_uri.size, device_uri.data);
  }
  return status;
}

static iree_status_t hrx_deactivate_remote_hal_device(
    iree_hal_device_t* hal_device) {
  hrx_remote_device_deactivate_state_t deactivate_state;
  memset(&deactivate_state, 0, sizeof(deactivate_state));
  iree_notification_initialize(&deactivate_state.notification);

  iree_hal_remote_client_device_deactivated_callback_t callback = {
      .fn = hrx_remote_device_deactivated,
      .user_data = &deactivate_state,
  };
  iree_status_t status =
      iree_hal_remote_client_device_deactivate(hal_device, callback);
  if (iree_status_is_ok(status)) {
    const bool deactivated = iree_notification_await(
        &deactivate_state.notification, hrx_remote_device_deactivate_complete,
        &deactivate_state, iree_infinite_timeout());
    if (!deactivated) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "remote device deactivation did not complete");
    }
  }

  iree_notification_deinitialize(&deactivate_state.notification);
  return status;
}
#endif  // HRX_HAS_IREE_REMOTE_DRIVER

static hrx_status_t hrx_create_remote_gpu_device(
    const char* device_uri_cstr, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_hal_device) {
  *out_driver = NULL;
  *out_hal_device = NULL;

#ifndef HRX_HAS_IREE_REMOTE_DRIVER
  (void)device_uri_cstr;
  (void)host_allocator;
  return hrx_make_status(
      HRX_STATUS_UNAVAILABLE,
      "remote HAL GPU driver requested but HRX was built without remote HAL "
      "support");
#else
  iree_string_view_t device_uri = iree_make_cstring_view(device_uri_cstr);
  iree_string_view_t driver_name = iree_uri_schema(device_uri);
  if (!iree_string_view_starts_with(driver_name, IREE_SV("remote-"))) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "remote GPU driver must be a remote-* HAL URI");
  }

  iree_hal_driver_registry_t* registry = NULL;
  iree_status_t status =
      iree_hal_driver_registry_allocate(host_allocator, &registry);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_driver_module_register(registry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(registry, driver_name,
                                                 host_allocator, out_driver);
  }
  iree_hal_driver_registry_free(registry);

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = g_shared.proactor_pool;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_device_by_uri(
        *out_driver, device_uri, &create_params, host_allocator,
        out_hal_device);
  }
  if (iree_status_is_ok(status)) {
    status = hrx_connect_remote_hal_device(device_uri, *out_hal_device);
  }

  if (!iree_status_is_ok(status)) {
    if (*out_hal_device) {
      status = iree_status_join(
          status, hrx_deactivate_remote_hal_device(*out_hal_device));
      iree_hal_device_release(*out_hal_device);
    }
    *out_hal_device = NULL;
    iree_hal_driver_release(*out_driver);
    *out_driver = NULL;
    return hrx_status_from_iree(status);
  }
  return hrx_ok_status();
#endif  // HRX_HAS_IREE_REMOTE_DRIVER
}

static hrx_status_t hrx_gpu_initialize_remote(const char* device_uri,
                                              iree_allocator_t host_allocator) {
  iree_hal_driver_t* driver = NULL;
  iree_hal_device_t* hal_device = NULL;
  hrx_status_t create_status = hrx_create_remote_gpu_device(
      device_uri, host_allocator, &driver, &hal_device);
  if (!hrx_status_is_ok(create_status)) return create_status;

  char name[128] = {0};
  snprintf(name, sizeof(name), "Remote GPU (%s)", device_uri);
  iree_status_t status = hrx_initialize_device_from_hal(
      &g_gpu.devices[0], HRX_ACCELERATOR_GPU, 0, hal_device,
      iree_make_cstring_view(name), IREE_SV("unknown"), host_allocator);
  const bool device_initialized = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = hrx_set_gpu_architecture_from_hal(hal_device, &g_gpu.devices[0]);
  }

  iree_hal_profile_sink_t* profile_sink = NULL;
  const char* profile_file_path = hrx_get_profile_file_path();
  if (iree_status_is_ok(status) && profile_file_path) {
    status = hrx_profile_file_sink_create(profile_file_path, host_allocator,
                                          &profile_sink);
  }
  if (iree_status_is_ok(status)) {
    status = hrx_device_profile_begin(&g_gpu.devices[0], profile_sink);
  }
  iree_hal_profile_sink_release(profile_sink);

  if (iree_status_is_ok(status)) {
    g_gpu.driver = driver;
    g_gpu.device_count = 1;
    g_gpu.backend = HRX_GPU_BACKEND_REMOTE;
    g_gpu.initialized = true;
    return hrx_ok_status();
  }

#ifdef HRX_HAS_IREE_REMOTE_DRIVER
  status =
      iree_status_join(status, hrx_deactivate_remote_hal_device(hal_device));
#endif  // HRX_HAS_IREE_REMOTE_DRIVER
  if (device_initialized) {
    hrx_release_initialized_device(&g_gpu.devices[0]);
  } else {
    iree_hal_device_release(hal_device);
  }
  iree_hal_driver_release(driver);
  return hrx_status_from_iree(status);
}

#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
static hrx_status_t hrx_create_iree_amdgpu_driver(
    iree_allocator_t alloc, iree_hal_driver_t** out_driver) {
  *out_driver = NULL;
  iree_hal_driver_registry_t* registry = NULL;
  iree_status_t status = iree_hal_driver_registry_allocate(alloc, &registry);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_driver_module_register(registry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        registry, iree_make_cstring_view("amdgpu"), alloc, out_driver);
  }
  iree_hal_driver_registry_free(registry);
  hrx_debug_print_iree_status("amdgpu driver create", status);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  return hrx_ok_status();
}
#endif

static hrx_status_t hrx_create_gpu_driver(iree_allocator_t alloc,
                                          iree_hal_driver_t** out_driver) {
  const char* driver_name = hrx_get_gpu_driver_name();
#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
  if (strcmp(driver_name, "amdgpu") == 0) {
    return hrx_create_iree_amdgpu_driver(alloc, out_driver);
  }
  char message[128];
  snprintf(message, sizeof(message),
           "unknown HRX_GPU_DRIVER '%s' (expected 'amdgpu')", driver_name);
  return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, message);
#else
  char message[96];
  snprintf(message, sizeof(message),
           "unknown HRX_GPU_DRIVER '%s' (built without AMDGPU support)",
           driver_name);
  return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, message);
#endif
}

//===----------------------------------------------------------------------===//
// CPU accelerator
//===----------------------------------------------------------------------===//

hrx_status_t hrx_cpu_initialize(uint32_t flags) {
  (void)flags;
  if (g_cpu.initialized) {
    return hrx_make_status(HRX_STATUS_ALREADY_EXISTS,
                           "CPU accelerator already initialized");
  }

  hrx_status_t status = hrx_ensure_shared_state();
  if (!hrx_status_is_ok(status)) return status;

  iree_hal_driver_t* driver = NULL;
  iree_hal_device_t* hal_device = NULL;
  iree_task_executor_t* executor = NULL;
  status = hrx_create_local_task_device(4, &executor, &driver, &hal_device);
  if (!hrx_status_is_ok(status)) {
    hrx_release_shared_state();
    return status;
  }

  iree_status_t iree_status = hrx_initialize_device_from_hal(
      &g_cpu.devices[0], HRX_ACCELERATOR_CPU, 0, hal_device,
      IREE_SV("CPU 0 (local-task)"), IREE_SV("host"), g_shared.host_allocator);
  if (!iree_status_is_ok(iree_status)) {
    iree_hal_device_release(hal_device);
    iree_hal_driver_release(driver);
    hrx_release_shared_state();
    return hrx_status_from_iree(iree_status);
  }

  g_cpu.driver = driver;
  g_cpu.device_count = 1;
  g_cpu.initialized = true;
  return hrx_ok_status();
}

hrx_status_t hrx_cpu_shutdown(void) {
  if (!g_cpu.initialized) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "CPU accelerator not initialized");
  }

  for (int i = 0; i < g_cpu.device_count; i++) {
    hrx_release_initialized_device(&g_cpu.devices[i]);
  }
  iree_hal_driver_release(g_cpu.driver);
  g_cpu.driver = NULL;

  g_cpu.device_count = 0;
  g_cpu.initialized = false;
  hrx_release_shared_state();
  return hrx_ok_status();
}

hrx_status_t hrx_cpu_device_count(int* count) {
  if (!count) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "count is NULL");
  }
  if (!g_cpu.initialized) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "CPU accelerator not initialized");
  }
  *count = g_cpu.device_count;
  return hrx_ok_status();
}

hrx_status_t hrx_cpu_device_get(int index, hrx_device_t* device) {
  if (!device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "device is NULL");
  }
  if (!g_cpu.initialized) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "CPU accelerator not initialized");
  }
  if (index < 0 || index >= g_cpu.device_count) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "CPU device index out of range");
  }
  *device = &g_cpu.devices[index];
  return hrx_ok_status();
}

//===----------------------------------------------------------------------===//
// GPU accelerator
//===----------------------------------------------------------------------===//

hrx_status_t hrx_gpu_initialize_with_device_extensions(
    uint32_t flags,
    const iree_hal_device_create_params_extension_t* device_extensions) {
  (void)flags;
  if (g_gpu.initialized) {
    return hrx_make_status(HRX_STATUS_ALREADY_EXISTS,
                           "GPU accelerator already initialized");
  }

  hrx_status_t status = hrx_ensure_shared_state();
  if (!hrx_status_is_ok(status)) return status;

  iree_allocator_t alloc = g_shared.host_allocator;
  const char* driver_name = hrx_get_gpu_driver_name();
  if (hrx_is_remote_hal_uri(driver_name)) {
    status = hrx_gpu_initialize_remote(driver_name, alloc);
    if (!hrx_status_is_ok(status)) {
      hrx_release_shared_state();
    }
    return status;
  }

#ifndef HRX_HAS_IREE_AMDGPU_DRIVER
  hrx_release_shared_state();
  return hrx_make_status(
      HRX_STATUS_UNAVAILABLE,
      "no GPU driver available (built without AMDGPU support)");
#else
  iree_hal_driver_t* driver = NULL;
  status = hrx_create_gpu_driver(alloc, &driver);
  if (!hrx_status_is_ok(status)) {
    hrx_release_shared_state();
    return status;
  }

  // Enumerate available GPU devices.
  iree_status_t iree_status = iree_ok_status();
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  iree_status = iree_hal_driver_query_available_devices(
      driver, alloc, &device_info_count, &device_infos);
  hrx_debug_print_iree_status("query available devices", iree_status);
  if (!iree_status_is_ok(iree_status)) {
    iree_hal_driver_release(driver);
    hrx_release_shared_state();
    return hrx_status_from_iree(iree_status);
  }

  if (device_info_count == 0) {
    iree_allocator_free(alloc, device_infos);
    iree_hal_driver_release(driver);
    hrx_release_shared_state();
    return hrx_make_status(HRX_STATUS_UNAVAILABLE, "no GPU devices found");
  }
  if (hrx_gpu_debug_enabled()) {
    fprintf(stderr, "hrx gpu debug: found %zu available GPU device entries\n",
            (size_t)device_info_count);
  }

  // IREE AMDGPU reports a pseudo-device with an empty path at ordinal 0 that
  // represents all visible GPUs as one logical device, then one entry per
  // physical device. HRX exposes physical devices to callers.
  int physical_count = 0;
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    if (device_infos[i].path.size == 0) continue;
    physical_count++;
  }
  if (physical_count == 0) {
    iree_allocator_free(alloc, device_infos);
    iree_hal_driver_release(driver);
    hrx_release_shared_state();
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "no physical GPU devices found");
  }

  int count =
      physical_count < HRX_MAX_DEVICES ? physical_count : HRX_MAX_DEVICES;

  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  iree_status = hrx_hal_runtime_features_from_environment(&runtime_features);
  if (!iree_status_is_ok(iree_status)) {
    iree_allocator_free(alloc, device_infos);
    iree_hal_driver_release(driver);
    hrx_release_shared_state();
    return hrx_status_from_iree(iree_status);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.next = device_extensions;
  create_params.proactor_pool = g_shared.proactor_pool;
  create_params.event_sink = hrx_hal_device_event_sink();
  create_params.runtime_features = runtime_features;

  iree_hal_profile_sink_t* profile_sink = NULL;
  const char* profile_file_path = hrx_get_profile_file_path();
  if (profile_file_path) {
    iree_status =
        hrx_profile_file_sink_create(profile_file_path, alloc, &profile_sink);
    hrx_debug_print_iree_status("create profile file sink", iree_status);
    if (!iree_status_is_ok(iree_status)) {
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }
  }

  int created_count = 0;
  for (iree_host_size_t info_index = 0;
       info_index < device_info_count && created_count < count; ++info_index) {
    if (device_infos[info_index].path.size == 0) continue;

    iree_hal_device_t* hal_device = NULL;
    iree_status = iree_hal_driver_create_device_by_ordinal(
        driver, info_index, /*param_count=*/0, /*params=*/NULL, &create_params,
        alloc, &hal_device);
    hrx_debug_print_iree_status("create device by ordinal", iree_status);
    if (!iree_status_is_ok(iree_status)) {
      hrx_gpu_release_created_devices(created_count);
      iree_hal_profile_sink_release(profile_sink);
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }

    iree_status = hrx_initialize_device_from_hal(
        &g_gpu.devices[created_count], HRX_ACCELERATOR_GPU, created_count,
        hal_device, device_infos[info_index].name, IREE_SV("unknown"), alloc);
    if (!iree_status_is_ok(iree_status)) {
      iree_hal_device_release(hal_device);
      hrx_gpu_release_created_devices(created_count);
      iree_hal_profile_sink_release(profile_sink);
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }

    hrx_device_s* dev = &g_gpu.devices[created_count];
    iree_status = hrx_set_gpu_architecture_from_hal(hal_device, dev);
    if (!iree_status_is_ok(iree_status)) {
      hrx_release_initialized_device(dev);
      hrx_gpu_release_created_devices(created_count);
      iree_hal_profile_sink_release(profile_sink);
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }

    iree_status = hrx_device_profile_begin(dev, profile_sink);
    hrx_debug_print_iree_status("begin device profiling", iree_status);
    if (!iree_status_is_ok(iree_status)) {
      hrx_release_initialized_device(dev);
      hrx_gpu_release_created_devices(created_count);
      iree_hal_profile_sink_release(profile_sink);
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }

    created_count++;
  }

  iree_hal_profile_sink_release(profile_sink);
  iree_allocator_free(alloc, device_infos);
  g_gpu.driver = driver;
  g_gpu.device_count = created_count;
  g_gpu.backend = HRX_GPU_BACKEND_NATIVE;
  g_gpu.initialized = true;
  return hrx_ok_status();
#endif  // HRX_HAS_IREE_AMDGPU_DRIVER
}

hrx_status_t hrx_gpu_initialize(uint32_t flags) {
  return hrx_gpu_initialize_with_device_extensions(flags,
                                                   /*device_extensions=*/NULL);
}

hrx_status_t hrx_gpu_shutdown(void) {
  if (!g_gpu.initialized) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "GPU accelerator not initialized");
  }

  iree_status_t status = hrx_gpu_end_all_profiling();
#ifdef HRX_HAS_IREE_REMOTE_DRIVER
  if (g_gpu.backend == HRX_GPU_BACKEND_REMOTE) {
    for (int i = 0; i < g_gpu.device_count; ++i) {
      status = iree_status_join(status, hrx_deactivate_remote_hal_device(
                                            g_gpu.devices[i].hal_device));
    }
  }
#endif  // HRX_HAS_IREE_REMOTE_DRIVER
  for (int i = 0; i < g_gpu.device_count; i++) {
    hrx_release_initialized_device(&g_gpu.devices[i]);
  }
  iree_hal_driver_release(g_gpu.driver);
  g_gpu.driver = NULL;

  g_gpu.device_count = 0;
  g_gpu.backend = HRX_GPU_BACKEND_NONE;
  g_gpu.initialized = false;
  hrx_release_shared_state();
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  return hrx_ok_status();
}

hrx_status_t hrx_gpu_device_count(int* count) {
  if (!count) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "count is NULL");
  }
  if (!g_gpu.initialized) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "GPU accelerator not initialized");
  }
  *count = g_gpu.device_count;
  return hrx_ok_status();
}

hrx_status_t hrx_gpu_device_get(int index, hrx_device_t* device) {
  if (!device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "device is NULL");
  }
  if (!g_gpu.initialized) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "GPU accelerator not initialized");
  }
  if (index < 0 || index >= g_gpu.device_count) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "GPU device index out of range");
  }
  *device = &g_gpu.devices[index];
  return hrx_ok_status();
}
