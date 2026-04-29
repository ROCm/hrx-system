// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Global runtime state management. Shared infrastructure (VM instance) is
// created on first accelerator init and destroyed when last shuts down.
// Device creation follows the proven pattern from PyTorch's hrx backend:
// driver-based creation via iree_hal_task_driver_create +
// iree_hal_driver_create_default_device.

#include "hrx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_handle.h"
#include "iree/modules/hal/types.h"
#include "iree/task/api.h"

#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"
#endif

//===----------------------------------------------------------------------===//
// Global singletons
//===----------------------------------------------------------------------===//

static hrx_shared_state_t g_shared = {0};
static hrx_gpu_state_t g_gpu = {0};
static hrx_cpu_state_t g_cpu = {0};

hrx_shared_state_t *hrx_get_shared_state(void) { return &g_shared; }
hrx_gpu_state_t *hrx_get_gpu_state(void) { return &g_gpu; }
hrx_cpu_state_t *hrx_get_cpu_state(void) { return &g_cpu; }

static iree_status_t
hrx_create_single_device_group(iree_hal_device_t *device,
                               iree_allocator_t host_allocator,
                               iree_hal_device_group_t **out_device_group) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_device_group);
  *out_device_group = NULL;

  iree_async_frontier_tracker_t *frontier_tracker = NULL;
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

void hrx_runtime_version(int *major, int *minor, int *patch) {
  if (major)
    *major = HRX_VERSION_MAJOR;
  if (minor)
    *minor = HRX_VERSION_MINOR;
  if (patch)
    *patch = HRX_VERSION_PATCH;
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

static void hrx_release_shared_state(void) {
  if (!g_shared.shared_initialized)
    return;
  g_shared.init_count--;
  if (g_shared.init_count > 0)
    return;

  if (g_shared.proactor_pool) {
    iree_async_proactor_pool_release(g_shared.proactor_pool);
    g_shared.proactor_pool = NULL;
  }
  if (g_shared.vm_instance) {
    iree_vm_instance_release(g_shared.vm_instance);
    g_shared.vm_instance = NULL;
  }
  g_shared.shared_initialized = false;
}

//===----------------------------------------------------------------------===//
// Helper: create a local-task device via driver pattern
//===----------------------------------------------------------------------===//

// Creates a local-task HAL device using the driver-based creation pattern.
// This matches the proven path in PyTorch's HrxRuntime::initialize().
// group_count controls the task executor parallelism.
static hrx_status_t hrx_create_local_task_device(
    int group_count, iree_task_executor_t **out_executor,
    iree_hal_driver_t **out_driver, iree_hal_device_t **out_hal_device) {

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

  iree_task_executor_t *executor = NULL;
  iree_status_t status =
      iree_task_executor_create(exec_options, &topology, alloc, &executor);
  iree_task_topology_deinitialize(&topology);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  // Executable loaders.
  iree_hal_executable_loader_t *loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  status = iree_hal_create_all_available_executable_loaders(
      /*plugin_manager=*/NULL, IREE_ARRAYSIZE(loaders), &loader_count, loaders,
      alloc);
  if (!iree_status_is_ok(status)) {
    iree_task_executor_release(executor);
    return hrx_status_from_iree(status);
  }

  // Heap allocator for host-accessible buffers.
  iree_hal_allocator_t *device_allocator = NULL;
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

  iree_hal_driver_t *driver = NULL;
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

  iree_hal_device_t *hal_device = NULL;
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
  iree_task_executor_t *out_exec = NULL;
  iree_task_executor_options_t out_exec_options;
  iree_task_executor_options_initialize(&out_exec_options);
  // Note: we don't need a separate executor for shutdown tracking.
  // The driver owns the executor internally. Set output to NULL.
  iree_task_topology_deinitialize(&out_topology);

  *out_executor = NULL; // Driver manages executor lifetime.
  *out_driver = driver;
  *out_hal_device = hal_device;
  return hrx_ok_status();
}

static const char *hrx_get_gpu_driver_name(void) {
  const char *value = getenv("HRX_GPU_DRIVER");
  return (value && value[0]) ? value : "amdgpu";
}

static bool hrx_gpu_debug_enabled(void) {
  const char *value = getenv("HRX_GPU_DEBUG");
  return value && value[0] && strcmp(value, "0") != 0;
}

static const char *hrx_get_profile_file_path(void) {
  const char *value = getenv("HRX_PROFILE_FILE");
  return (value && value[0]) ? value : NULL;
}

static iree_status_t
hrx_profile_file_sink_create(const char *file_path,
                             iree_allocator_t host_allocator,
                             iree_hal_profile_sink_t **out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;
  if (!file_path || !file_path[0]) {
    return iree_ok_status();
  }

  iree_io_file_handle_t *file_handle = NULL;
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

static iree_status_t
hrx_get_profile_data_families(
    iree_hal_device_profiling_data_families_t *out_data_families) {
  IREE_ASSERT_ARGUMENT(out_data_families);

  const char *value = getenv("HRX_PROFILE_MODE");
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

static iree_status_t hrx_device_profile_begin(hrx_device_s *device,
                                              iree_hal_profile_sink_t *sink) {
  if (!device || !sink) {
    return iree_ok_status();
  }

  iree_hal_device_profiling_options_t options = {0};
  IREE_RETURN_IF_ERROR(
      hrx_get_profile_data_families(&options.data_families));
  options.sink = sink;
  iree_status_t status =
      iree_hal_device_profiling_begin(device->hal_device, &options);
  if (iree_status_is_ok(status)) {
    device->profiling_active = true;
  }
  return status;
}

static iree_status_t hrx_device_profile_end(hrx_device_s *device) {
  if (!device || !device->profiling_active || !device->hal_device) {
    return iree_ok_status();
  }

  iree_hal_semaphore_list_t empty = {0};
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
    hrx_device_release(&g_gpu.devices[i]);
  }
}

static void hrx_debug_print_iree_status(const char *label,
                                        iree_status_t status) {
  if (!hrx_gpu_debug_enabled() || iree_status_is_ok(status))
    return;
  iree_allocator_t allocator = iree_allocator_system();
  char *message = NULL;
  iree_host_size_t message_length = 0;
  if (iree_status_to_string(status, &allocator, &message, &message_length)) {
    fprintf(stderr, "hrx gpu debug: %s: %s\n", label,
            message ? message : "(no message)");
    iree_allocator_free(allocator, message);
  } else {
    fprintf(stderr, "hrx gpu debug: %s: (could not format status)\n", label);
  }
}

#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
static hrx_status_t
hrx_create_iree_amdgpu_driver(iree_allocator_t alloc,
                              iree_hal_driver_t **out_driver) {
  iree_status_t status = iree_hal_amdgpu_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  hrx_debug_print_iree_status("amdgpu driver module register", status);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  iree_hal_driver_t *driver = NULL;
  status = iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(), iree_make_cstring_view("amdgpu"),
      alloc, &driver);
  hrx_debug_print_iree_status("amdgpu driver create", status);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  *out_driver = driver;
  return hrx_ok_status();
}
#endif

static hrx_status_t hrx_create_gpu_driver(iree_allocator_t alloc,
                                          iree_hal_driver_t **out_driver) {
  const char *driver_name = hrx_get_gpu_driver_name();
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
  if (!hrx_status_is_ok(status))
    return status;

  iree_hal_driver_t *driver = NULL;
  iree_hal_device_t *hal_device = NULL;
  iree_task_executor_t *executor = NULL;
  status = hrx_create_local_task_device(4, &executor, &driver, &hal_device);
  if (!hrx_status_is_ok(status)) {
    hrx_release_shared_state();
    return status;
  }

  iree_hal_device_group_t *device_group = NULL;
  iree_status_t iree_status = hrx_create_single_device_group(
      hal_device, g_shared.host_allocator, &device_group);
  if (!iree_status_is_ok(iree_status)) {
    iree_hal_device_release(hal_device);
    iree_hal_driver_release(driver);
    hrx_release_shared_state();
    return hrx_status_from_iree(iree_status);
  }

  hrx_device_s *dev = &g_cpu.devices[0];
  memset(dev, 0, sizeof(*dev));
  iree_atomic_ref_count_init(&dev->ref_count);
  dev->type = HRX_ACCELERATOR_CPU;
  dev->ordinal = 0;
  dev->hal_device = hal_device;
  dev->hal_device_group = device_group;
  dev->allocator.hal_allocator = iree_hal_device_allocator(hal_device);
  iree_hal_allocator_retain(dev->allocator.hal_allocator);
  iree_atomic_ref_count_init(&dev->allocator.ref_count);
  dev->allocator.device = dev;
  snprintf(dev->name, sizeof(dev->name), "CPU 0 (local-task)");
  snprintf(dev->architecture, sizeof(dev->architecture), "host");

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
    hrx_device_release(&g_cpu.devices[i]);
  }
  if (g_cpu.driver) {
    iree_hal_driver_release(g_cpu.driver);
    g_cpu.driver = NULL;
  }

  g_cpu.device_count = 0;
  g_cpu.initialized = false;
  hrx_release_shared_state();
  return hrx_ok_status();
}

hrx_status_t hrx_cpu_device_count(int *count) {
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

hrx_status_t hrx_cpu_device_get(int index, hrx_device_t *device) {
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

hrx_status_t hrx_gpu_initialize(uint32_t flags) {
  (void)flags;
  if (g_gpu.initialized) {
    return hrx_make_status(HRX_STATUS_ALREADY_EXISTS,
                           "GPU accelerator already initialized");
  }

#ifndef HRX_HAS_IREE_AMDGPU_DRIVER
  return hrx_make_status(
      HRX_STATUS_UNAVAILABLE,
      "no GPU driver available (built without AMDGPU support)");
#else
  hrx_status_t status = hrx_ensure_shared_state();
  if (!hrx_status_is_ok(status))
    return status;

  iree_allocator_t alloc = g_shared.host_allocator;

  iree_hal_driver_t *driver = NULL;
  status = hrx_create_gpu_driver(alloc, &driver);
  if (!hrx_status_is_ok(status)) {
    hrx_release_shared_state();
    return status;
  }

  // Enumerate available GPU devices.
  iree_status_t iree_status = iree_ok_status();
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t *device_infos = NULL;
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
    if (device_infos[i].path.size == 0)
      continue;
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

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = g_shared.proactor_pool;

  iree_hal_profile_sink_t *profile_sink = NULL;
  const char *profile_file_path = hrx_get_profile_file_path();
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
    if (device_infos[info_index].path.size == 0)
      continue;

    iree_hal_device_t *hal_device = NULL;
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

    iree_hal_device_group_t *device_group = NULL;
    iree_status =
        hrx_create_single_device_group(hal_device, alloc, &device_group);
    if (!iree_status_is_ok(iree_status)) {
      iree_hal_device_release(hal_device);
      hrx_gpu_release_created_devices(created_count);
      iree_hal_profile_sink_release(profile_sink);
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }

    hrx_device_s *dev = &g_gpu.devices[created_count];
    memset(dev, 0, sizeof(*dev));
    iree_atomic_ref_count_init(&dev->ref_count);
    dev->type = HRX_ACCELERATOR_GPU;
    dev->ordinal = created_count;
    dev->hal_device = hal_device;
    dev->hal_device_group = device_group;
    dev->allocator.hal_allocator = iree_hal_device_allocator(hal_device);
    iree_hal_allocator_retain(dev->allocator.hal_allocator);
    iree_atomic_ref_count_init(&dev->allocator.ref_count);
    dev->allocator.device = dev;

    iree_host_size_t name_len = device_infos[info_index].name.size;
    if (name_len >= sizeof(dev->name))
      name_len = sizeof(dev->name) - 1;
    memcpy(dev->name, device_infos[info_index].name.data, name_len);
    dev->name[name_len] = '\0';

    iree_host_size_t arch_len = device_infos[info_index].name.size;
    if (arch_len >= sizeof(dev->architecture)) {
      arch_len = sizeof(dev->architecture) - 1;
    }
    memcpy(dev->architecture, device_infos[info_index].name.data, arch_len);
    dev->architecture[arch_len] = '\0';

    iree_status = hrx_device_profile_begin(dev, profile_sink);
    hrx_debug_print_iree_status("begin device profiling", iree_status);
    if (!iree_status_is_ok(iree_status)) {
      hrx_device_release(dev);
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
  g_gpu.initialized = true;
  return hrx_ok_status();
#endif // HRX_HAS_IREE_AMDGPU_DRIVER
}

hrx_status_t hrx_gpu_shutdown(void) {
  if (!g_gpu.initialized) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "GPU accelerator not initialized");
  }

  iree_status_t profiling_status = hrx_gpu_end_all_profiling();
  for (int i = 0; i < g_gpu.device_count; i++) {
    hrx_device_release(&g_gpu.devices[i]);
  }
  if (g_gpu.driver) {
    iree_hal_driver_release(g_gpu.driver);
    g_gpu.driver = NULL;
  }

  g_gpu.device_count = 0;
  g_gpu.initialized = false;
  hrx_release_shared_state();
  if (!iree_status_is_ok(profiling_status)) {
    return hrx_status_from_iree(profiling_status);
  }
  return hrx_ok_status();
}

hrx_status_t hrx_gpu_device_count(int *count) {
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

hrx_status_t hrx_gpu_device_get(int index, hrx_device_t *device) {
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
