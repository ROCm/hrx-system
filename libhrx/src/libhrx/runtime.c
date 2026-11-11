// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "runtime.h"

#include <stddef.h>
#include <string.h>

#include "device_group.h"
#include "hrx_internal.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/driver/task/driver.h"
#include "iree/task/api.h"

#ifdef HRX_HAS_IREE_AMD_DRIVER
#include "iree/hal/driver/amd/registration/driver_module.h"
#endif
#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
#include "iree/hal/driver/amdgpu/registration/driver_module.h"
#endif

// KFD retains its original render-file identity across device generations.
// Native connections and their code remain process-owned; execution services
// belong to individual runtimes and groups.
static iree_hal_driver_registry_t* g_gpu_driver_registry;
static iree_slim_mutex_t g_driver_registry_mutex;
static iree_once_flag g_driver_registry_once = IREE_ONCE_FLAG_INIT;

static void hrx_initialize_driver_registry_mutex(void) {
  iree_slim_mutex_initialize(&g_driver_registry_mutex);
}

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

void hrx_runtime_version(int* major, int* minor, int* patch) {
  if (major) {
    *major = HRX_VERSION_MAJOR;
  }
  if (minor) {
    *minor = HRX_VERSION_MINOR;
  }
  if (patch) {
    *patch = HRX_VERSION_PATCH;
  }
}

iree_hal_device_event_sink_t hrx_device_event_sink_to_hal(
    const hrx_device_event_sink_t* sink) {
  if (!sink->fn) {
    return iree_hal_device_event_sink_discard();
  }
  return (iree_hal_device_event_sink_t){
      .fn = hrx_hal_device_event_sink_thunk,
      .user_data = (void*)sink,
  };
}

iree_hal_device_event_sink_t hrx_runtime_hal_event_sink(hrx_runtime_t runtime) {
  return hrx_device_event_sink_to_hal(&runtime->options.event_sink);
}

void hrx_runtime_report_failure(hrx_runtime_t runtime,
                                iree_hal_device_t* device,
                                iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = iree_status_code(status),
      .message = iree_status_message(status),
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.device = device;
  if (device) {
    event.source.device_id = iree_hal_device_id(device);
  }
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(runtime
                                         ? hrx_runtime_hal_event_sink(runtime)
                                         : iree_hal_device_event_sink_discard(),
                                     &event);
  iree_status_free(status);
}

iree_hal_device_event_sink_t hrx_device_hal_event_sink(hrx_device_t device) {
  return hrx_runtime_hal_event_sink(device->group->runtime);
}

iree_async_proactor_pool_t* hrx_device_proactor_pool(hrx_device_t device) {
  return device->group->runtime->proactor_pool;
}

hrx_status_t hrx_runtime_create(const hrx_runtime_options_t* options,
                                hrx_host_allocator_t host_allocator,
                                hrx_runtime_t* out_runtime) {
  if (!out_runtime) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_runtime = NULL;
  const hrx_runtime_options_t selected =
      options ? *options : (hrx_runtime_options_t){0};
  if (selected.task.worker_count > IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT ||
      (selected.task.cpu_ids && !selected.task.worker_count)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid task worker configuration");
  }
  iree_allocator_t allocator;
  memcpy(&allocator, &host_allocator, sizeof(allocator));
  if (!allocator.ctl) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "host allocator is NULL");
  }
  const size_t cpu_bytes =
      selected.task.cpu_ids ? selected.task.worker_count * sizeof(uint32_t) : 0;
  hrx_runtime_t runtime = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, sizeof(*runtime) + cpu_bytes, (void**)&runtime);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  iree_atomic_ref_count_init(&runtime->ref_count);
  runtime->host_allocator = allocator;
  runtime->options = selected;
  if (cpu_bytes) {
    memcpy(runtime + 1, selected.task.cpu_ids, cpu_bytes);
    runtime->options.task.cpu_ids = (const uint32_t*)(runtime + 1);
  }
  if (!runtime->options.task.worker_count) {
    runtime->options.task.worker_count = 4;
  }
  if (!runtime->options.task.worker_stack_size) {
    runtime->options.task.worker_stack_size = 256 * 1024;
  }
  iree_slim_mutex_initialize(&runtime->task_mutex);

  const iree_bitmap_t online_nodes = iree_numa_online_nodes();
  uint32_t node_ids[IREE_NUMA_MAX_NODES];
  iree_host_size_t node_count = 0;
  for (uint32_t node = 0; node < online_nodes.bit_count; ++node) {
    if (iree_bitmap_test(online_nodes, node)) {
      node_ids[node_count++] = node;
    }
  }
  status = iree_async_proactor_pool_create(
      node_count, node_ids, iree_async_proactor_pool_options_default(),
      allocator, &runtime->proactor_pool);
  if (iree_status_is_ok(status)) {
    *out_runtime = runtime;
  } else {
    hrx_runtime_release(runtime);
  }
  return hrx_status_from_iree(status);
}

void hrx_runtime_retain(hrx_runtime_t runtime) {
  if (runtime) {
    iree_atomic_ref_count_inc(&runtime->ref_count);
  }
}

void hrx_runtime_release(hrx_runtime_t runtime) {
  if (!runtime) {
    return;
  }
  if (iree_atomic_ref_count_dec(&runtime->ref_count) != 1) {
    return;
  }
  iree_hal_driver_release(runtime->task_driver);
  iree_async_proactor_pool_release(runtime->proactor_pool);
  iree_slim_mutex_deinitialize(&runtime->task_mutex);
  iree_allocator_free(runtime->host_allocator, runtime);
}

static iree_status_t hrx_create_task_driver(hrx_runtime_t runtime,
                                            iree_hal_driver_t** out_driver) {
  const iree_allocator_t allocator = runtime->host_allocator;
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  iree_status_t status = iree_ok_status();
  if (runtime->options.task.cpu_ids) {
    status = iree_task_topology_initialize_from_logical_cpu_set(
        runtime->options.task.worker_count, runtime->options.task.cpu_ids,
        &topology);
  } else {
    iree_task_topology_initialize_from_group_count(
        runtime->options.task.worker_count, &topology);
  }
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_stack_size = runtime->options.task.worker_stack_size;
  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_task_executor_create(options, &topology, allocator, &executor);
  }
  iree_task_topology_deinitialize(&topology);
  iree_hal_executable_loader_t* loaders[8] = {0};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_all_available_executable_loaders(
        IREE_ARRAYSIZE(loaders), allocator, &loader_count, loaders);
  }
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(allocator, &device_allocator);
  }
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_driver_create(
        IREE_SV("task"), &params, 1, &executor, loader_count, loaders,
        device_allocator, allocator, out_driver);
  }
  iree_hal_allocator_release(device_allocator);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  iree_task_executor_release(executor);
  return status;
}

iree_status_t hrx_device_profile_begin(
    hrx_device_t device, const iree_hal_device_profiling_options_t* options) {
  iree_status_t status =
      iree_hal_device_profiling_begin(device->hal_device, options);
  if (iree_status_is_ok(status)) {
    device->profiling_active = true;
  }
  return status;
}

iree_status_t hrx_runtime_profile_flush(hrx_device_t device) {
  if (!device || !device->profiling_active) {
    return iree_ok_status();
  }
  iree_slim_mutex_lock(&device->group->profile_mutex);
  iree_status_t status = iree_hal_device_profiling_flush(device->hal_device);
  iree_slim_mutex_unlock(&device->group->profile_mutex);
  return status;
}

iree_status_t hrx_device_profile_end(hrx_device_s* device) {
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

static iree_status_t hrx_initialize_gpu_driver_registry(void) {
  if (g_gpu_driver_registry) {
    return iree_ok_status();
  }
  const iree_allocator_t allocator = iree_allocator_system();
  iree_hal_driver_registry_t* registry = NULL;
  iree_status_t status =
      iree_hal_driver_registry_allocate(allocator, &registry);
#ifdef HRX_HAS_IREE_AMD_DRIVER
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_driver_module_register(registry);
  }
#endif
#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_driver_module_register(registry);
  }
#endif
#if defined(HRX_BUILDING_SHARED)
  // Keep the owner and its implementation available for the native connection's
  // lifetime. Static embedders provide this lifetime themselves; the HIP
  // binding retains its containing module before initializing its embedded HRX
  // runtime.
  if (iree_status_is_ok(status)) {
    iree_string_builder_t path;
    iree_string_builder_initialize(allocator, &path);
    status = iree_dynamic_library_append_symbol_path_to_builder(
        (void*)&hrx_initialize_gpu_driver_registry, &path);
    iree_dynamic_library_t* library = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_dynamic_library_load_from_file(
          iree_string_builder_buffer(&path), IREE_DYNAMIC_LIBRARY_FLAG_NODELETE,
          allocator, &library);
    }
    iree_dynamic_library_release(library);
    iree_string_builder_deinitialize(&path);
  }
#endif
  if (!iree_status_is_ok(status)) {
    iree_hal_driver_registry_free(registry);
    return status;
  }
  g_gpu_driver_registry = registry;
  return iree_ok_status();
}

iree_status_t hrx_runtime_acquire_driver(hrx_runtime_t runtime,
                                         iree_string_view_t scheme,
                                         iree_hal_driver_t** out_driver) {
  *out_driver = NULL;
  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(scheme, IREE_SV("task"))) {
    iree_slim_mutex_lock(&runtime->task_mutex);
    if (!runtime->task_driver) {
      status = hrx_create_task_driver(runtime, &runtime->task_driver);
    }
    if (iree_status_is_ok(status)) {
      *out_driver = runtime->task_driver;
      iree_hal_driver_retain(*out_driver);
    }
    iree_slim_mutex_unlock(&runtime->task_mutex);
  } else {
    iree_call_once(&g_driver_registry_once,
                   hrx_initialize_driver_registry_mutex);
    iree_slim_mutex_lock(&g_driver_registry_mutex);
    status = hrx_initialize_gpu_driver_registry();
    iree_slim_mutex_unlock(&g_driver_registry_mutex);
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_registry_acquire_driver(g_gpu_driver_registry,
                                                       scheme, out_driver);
    }
  }
  return status;
}

hrx_status_t hrx_runtime_query_available_devices(
    hrx_runtime_t runtime, const char* driver_name,
    hrx_host_allocator_t host_allocator, size_t* out_count,
    hrx_device_info_t** out_infos) {
  if (!out_count || !out_infos) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_count = 0;
  *out_infos = NULL;
  iree_allocator_t allocator;
  memcpy(&allocator, &host_allocator, sizeof(allocator));
  if (!runtime || !driver_name || !driver_name[0] || !allocator.ctl) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "runtime, driver name, and allocator are required");
  }
  const iree_string_view_t scheme = iree_make_cstring_view(driver_name);
  const bool is_task = iree_string_view_equal(scheme, IREE_SV("task"));
  // Task exposes one default device configured by its runtime. Describing it
  // must not construct the executor that task-driver creation would require.
  iree_hal_device_info_t task_info = {
      .name = iree_string_view_literal("default"),
  };
  iree_host_size_t info_count = is_task ? 1 : 0;
  iree_hal_device_info_t* infos = is_task ? &task_info : NULL;
  iree_hal_driver_t* driver = NULL;
  iree_status_t status = iree_ok_status();
  if (!is_task) {
    status = hrx_runtime_acquire_driver(runtime, scheme, &driver);
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_query_available_devices(
          driver, scheme, runtime->host_allocator, &info_count, &infos);
    }
  }

  iree_host_size_t info_bytes = 0;
  iree_host_size_t allocation_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(info_count, sizeof(hrx_device_info_t),
                                  &info_bytes)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "device inventory size overflows");
  }
  allocation_size = info_bytes;
  for (iree_host_size_t i = 0; i < info_count && iree_status_is_ok(status);
       ++i) {
    // Each URI contains the scheme, ://, native path, and NUL; the name has
    // its own NUL. Neither string borrows native discovery storage.
    if (!iree_host_size_checked_add(allocation_size, scheme.size,
                                    &allocation_size) ||
        !iree_host_size_checked_add(allocation_size, infos[i].path.size,
                                    &allocation_size) ||
        !iree_host_size_checked_add(allocation_size, infos[i].name.size,
                                    &allocation_size) ||
        !iree_host_size_checked_add(allocation_size, 5, &allocation_size)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "device inventory strings overflow");
    }
  }
  hrx_device_info_t* result = NULL;
  if (iree_status_is_ok(status) && info_count) {
    status = iree_allocator_malloc(allocator, allocation_size, (void**)&result);
  }
  if (iree_status_is_ok(status) && info_count) {
    char* cursor = (char*)result + info_bytes;
    for (iree_host_size_t i = 0; i < info_count; ++i) {
      result[i].uri = cursor;
      memcpy(cursor, scheme.data, scheme.size);
      cursor += scheme.size;
      memcpy(cursor, "://", 3);
      cursor += 3;
      if (infos[i].path.size) {
        memcpy(cursor, infos[i].path.data, infos[i].path.size);
        cursor += infos[i].path.size;
      }
      *cursor++ = 0;
      result[i].name = cursor;
      if (infos[i].name.size) {
        memcpy(cursor, infos[i].name.data, infos[i].name.size);
        cursor += infos[i].name.size;
      }
      *cursor++ = 0;
    }
  }
  if (!is_task) {
    iree_allocator_free(runtime->host_allocator, infos);
  }
  iree_hal_driver_release(driver);
  if (iree_status_is_ok(status)) {
    *out_count = info_count;
    *out_infos = result;
  }
  return hrx_status_from_iree(status);
}
