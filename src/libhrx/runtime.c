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

#include "hsa/hsa.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/modules/hal/types.h"
#include "iree/task/api.h"

#ifdef HRX_HAS_HSA_DRIVER
#include "hsa_driver/api.h"
#include "hsa_driver/registration/driver_module.h"
#endif
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

static iree_status_t hrx_create_single_device_group(
    iree_hal_device_t *device, iree_allocator_t host_allocator,
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

  // Initialize HSA runtime (idempotent — safe to call multiple times).
  hsa_status_t hsa_status = hsa_init();
  if (hsa_status != HSA_STATUS_SUCCESS) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE, "hsa_init() failed");
  }

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
  hsa_shut_down();
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
  // The HSA runtime adds TLS that raises the effective minimum pthread stack
  // size from 16KB to ~48KB. IREE's default 32KB is too small when HSA is
  // linked. Use 256KB which is safe for ASAN builds too.
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

static const char* hrx_get_gpu_driver_name(void) {
  const char* value = getenv("HRX_GPU_DRIVER");
  return (value && value[0]) ? value : "hsa";
}

static bool hrx_gpu_debug_enabled(void) {
  const char* value = getenv("HRX_GPU_DEBUG");
  return value && value[0] && strcmp(value, "0") != 0;
}

static void hrx_debug_print_iree_status(const char* label,
                                        iree_status_t status) {
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

static hrx_status_t hrx_create_hsa_gpu_driver(iree_allocator_t alloc,
                                              iree_hal_driver_t** out_driver) {
  iree_hal_hsa_driver_options_t driver_options;
  iree_hal_hsa_driver_options_initialize(&driver_options);
  iree_hal_hsa_device_params_t device_params;
  iree_hal_hsa_device_params_initialize(&device_params);

  iree_hal_driver_t* driver = NULL;
  iree_status_t status =
      iree_hal_hsa_driver_create(iree_make_cstring_view("hsa"), &driver_options,
                                 &device_params, alloc, &driver);
  hrx_debug_print_iree_status("hsa driver create", status);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  *out_driver = driver;
  return hrx_ok_status();
}

#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
static hrx_status_t hrx_create_iree_amdgpu_driver(
    iree_allocator_t alloc, iree_hal_driver_t** out_driver) {
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

  iree_hal_driver_t* driver = NULL;
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
                                          iree_hal_driver_t** out_driver) {
  const char* driver_name = hrx_get_gpu_driver_name();
  if (strcmp(driver_name, "hsa") == 0) {
    return hrx_create_hsa_gpu_driver(alloc, out_driver);
  }
#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
  if (strcmp(driver_name, "amdgpu") == 0) {
    return hrx_create_iree_amdgpu_driver(alloc, out_driver);
  }
  char message[128];
  snprintf(message, sizeof(message),
           "unknown HRX_GPU_DRIVER '%s' (expected 'hsa' or 'amdgpu')",
           driver_name);
  return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, message);
#else
  char message[96];
  snprintf(message, sizeof(message),
           "unknown HRX_GPU_DRIVER '%s' (expected 'hsa')", driver_name);
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

#ifndef HRX_HAS_HSA_DRIVER
  return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                         "no GPU driver available (built without HSA support)");
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

  // Create a HAL device for each GPU (up to HRX_MAX_DEVICES).
  int count = (int)(device_info_count < HRX_MAX_DEVICES ? device_info_count
                                                        : HRX_MAX_DEVICES);

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = g_shared.proactor_pool;

  for (int i = 0; i < count; i++) {
    iree_hal_device_t *hal_device = NULL;
    iree_status = iree_hal_driver_create_device_by_ordinal(
        driver, (iree_host_size_t)i, /*param_count=*/0, /*params=*/NULL,
        &create_params, alloc, &hal_device);
    hrx_debug_print_iree_status("create device by ordinal", iree_status);
    if (!iree_status_is_ok(iree_status)) {
      for (int j = 0; j < i; j++) {
        hrx_device_release(&g_gpu.devices[j]);
      }
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }

    iree_hal_device_group_t *device_group = NULL;
    iree_status = hrx_create_single_device_group(hal_device, alloc,
                                                 &device_group);
    if (!iree_status_is_ok(iree_status)) {
      iree_hal_device_release(hal_device);
      for (int j = 0; j < i; j++) {
        hrx_device_release(&g_gpu.devices[j]);
      }
      iree_allocator_free(alloc, device_infos);
      iree_hal_driver_release(driver);
      hrx_release_shared_state();
      return hrx_status_from_iree(iree_status);
    }

    hrx_device_s *dev = &g_gpu.devices[i];
    memset(dev, 0, sizeof(*dev));
    iree_atomic_ref_count_init(&dev->ref_count);
    dev->type = HRX_ACCELERATOR_GPU;
    dev->ordinal = i;
    dev->hal_device = hal_device;
    dev->hal_device_group = device_group;
    dev->allocator.hal_allocator = iree_hal_device_allocator(hal_device);
    iree_hal_allocator_retain(dev->allocator.hal_allocator);
    iree_atomic_ref_count_init(&dev->allocator.ref_count);
    dev->allocator.device = dev;

    iree_host_size_t name_len = device_infos[i].name.size;
    if (name_len >= sizeof(dev->name))
      name_len = sizeof(dev->name) - 1;
    memcpy(dev->name, device_infos[i].name.data, name_len);
    dev->name[name_len] = '\0';

    iree_host_size_t arch_len = device_infos[i].name.size;
    if (arch_len >= sizeof(dev->architecture)) {
      arch_len = sizeof(dev->architecture) - 1;
    }
    memcpy(dev->architecture, device_infos[i].name.data, arch_len);
    dev->architecture[arch_len] = '\0';
  }

  iree_allocator_free(alloc, device_infos);
  g_gpu.driver = driver;
  g_gpu.device_count = count;
  g_gpu.initialized = true;
  return hrx_ok_status();
#endif // HRX_HAS_HSA_DRIVER
}

hrx_status_t hrx_gpu_shutdown(void) {
  if (!g_gpu.initialized) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "GPU accelerator not initialized");
  }

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
