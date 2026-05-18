// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/server/bulk_staging_pool.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_CAPACITY 64
#define IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_SLOT_LENGTH (32 * 1024)

typedef struct benchmark_device_t {
  // Local-task driver owning |device|.
  iree_hal_driver_t* driver;

  // Local-task HAL device used for file import and semaphore creation.
  iree_hal_device_t* device;
} benchmark_device_t;

static iree_status_t RegisterLocalTaskDriver(void) {
  iree_status_t status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }
  return status;
}

static iree_status_t CreateBenchmarkDevice(benchmark_device_t* out_device) {
  memset(out_device, 0, sizeof(*out_device));

  iree_status_t status = RegisterLocalTaskDriver();
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(),
        &out_device->driver);
  }

  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(
        out_device->driver, &create_params, iree_allocator_system(),
        &out_device->device);
  }
  iree_async_proactor_pool_release(proactor_pool);

  if (!iree_status_is_ok(status)) {
    iree_hal_device_release(out_device->device);
    iree_hal_driver_release(out_device->driver);
    memset(out_device, 0, sizeof(*out_device));
  }
  return status;
}

static void DestroyBenchmarkDevice(benchmark_device_t* device) {
  iree_hal_device_release(device->device);
  iree_hal_driver_release(device->driver);
  memset(device, 0, sizeof(*device));
}

static iree_status_t AllocateBenchmarkPool(
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_staging_pool_t** out_pool) {
  iree_hal_remote_server_bulk_staging_pool_options_t options =
      iree_hal_remote_server_bulk_staging_pool_options_default();
  options.slot_count = IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_CAPACITY;
  options.slot_length = IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_SLOT_LENGTH;
  return iree_hal_remote_server_bulk_staging_pool_create(
      &options, host_allocator, out_pool);
}

static iree_status_t WarmBindBenchmarkPool(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_device_t* device) {
  iree_hal_remote_server_bulk_staging_slot_t*
      slots[IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_CAPACITY];
  memset(slots, 0, sizeof(slots));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(slots) && iree_status_is_ok(status); ++i) {
    status = iree_hal_remote_server_bulk_staging_pool_acquire(pool, device,
                                                              &slots[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
    iree_hal_remote_server_bulk_staging_slot_release(slots[i],
                                                     /*last_signal_value=*/0);
  }
  return status;
}

static iree_status_t BM_AcquireReleaseSlots(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  benchmark_device_t device;
  iree_status_t status = CreateBenchmarkDevice(&device);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_remote_server_bulk_staging_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    status = AllocateBenchmarkPool(host_allocator, &pool);
  }
  if (iree_status_is_ok(status)) {
    status = WarmBindBenchmarkPool(pool, device.device);
  }

  iree_hal_remote_server_bulk_staging_slot_t*
      slots[IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_CAPACITY];
  memset(slots, 0, sizeof(slots));
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(
             benchmark_state,
             IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_CAPACITY)) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(slots) && iree_status_is_ok(status); ++i) {
      status = iree_hal_remote_server_bulk_staging_pool_acquire(
          pool, device.device, &slots[i]);
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
      iree_hal_remote_server_bulk_staging_slot_release(slots[i],
                                                       /*last_signal_value=*/0);
      slots[i] = NULL;
    }
  }

  iree_hal_remote_server_bulk_staging_pool_release(pool);
  DestroyBenchmarkDevice(&device);
  return status;
}

static iree_status_t BM_StageSaturatedChunks(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  benchmark_device_t device;
  iree_status_t status = CreateBenchmarkDevice(&device);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_remote_server_bulk_staging_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    status = AllocateBenchmarkPool(host_allocator, &pool);
  }
  if (iree_status_is_ok(status)) {
    status = WarmBindBenchmarkPool(pool, device.device);
  }

  iree_hal_remote_server_bulk_staging_slot_t*
      slots[IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_CAPACITY];
  memset(slots, 0, sizeof(slots));
  const iree_host_size_t bytes_per_iteration =
      IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_CAPACITY *
      IREE_HAL_REMOTE_BULK_STAGING_POOL_BENCHMARK_SLOT_LENGTH;
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, bytes_per_iteration)) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(slots) && iree_status_is_ok(status); ++i) {
      status = iree_hal_remote_server_bulk_staging_pool_acquire(
          pool, device.device, &slots[i]);
      if (iree_status_is_ok(status)) {
        iree_byte_span_t contents =
            iree_hal_remote_server_bulk_staging_slot_contents(slots[i]);
        memset(contents.data, (int)i, contents.data_length);
        iree_benchmark_use_ptr((char const volatile*)contents.data);
      }
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
      iree_hal_remote_server_bulk_staging_slot_release(slots[i],
                                                       /*last_signal_value=*/0);
      slots[i] = NULL;
    }
  }

  iree_hal_remote_server_bulk_staging_pool_release(pool);
  DestroyBenchmarkDevice(&device);
  return status;
}

IREE_BENCHMARK_REGISTER(BM_AcquireReleaseSlots);
IREE_BENCHMARK_REGISTER(BM_StageSaturatedChunks);
