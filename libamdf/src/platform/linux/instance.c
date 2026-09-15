// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/platform/linux/instance.h"

#include <fcntl.h>
#include <unistd.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

amdf_status_t amdf_platform_instance_create(
    amdf_allocator_t host_allocator, amdf_platform_instance_t** out_instance) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_platform_instance_t* instance = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*instance),
                  amdf_alignof(amdf_platform_instance_t), (void**)&instance);
  if (!amdf_status_is_ok(status)) return status;
  instance->host_allocator = host_allocator;
  instance->page_size = (size_t)page_size;
  const int mutex_error = pthread_mutex_init(&instance->native_mutex, NULL);
  if (mutex_error != 0) {
    amdf_free(host_allocator, instance);
    return amdf_linux_error(mutex_error);
  }
  instance->sysfs_descriptor = open("/sys", O_PATH | O_DIRECTORY | O_CLOEXEC);
  if (instance->sysfs_descriptor < 0) {
    status = amdf_linux_error(errno);
    const int mutex_error = pthread_mutex_destroy(&instance->native_mutex);
    amdf_assert(mutex_error == 0);
    (void)mutex_error;
    amdf_free(host_allocator, instance);
    return status;
  }
  *out_instance = instance;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_instance_destroy(
    amdf_platform_instance_t* instance) {
  const amdf_status_t status =
      amdf_linux_file_close(&instance->sysfs_descriptor);
  if (amdf_status_is_ok(status)) {
    const int mutex_error = pthread_mutex_destroy(&instance->native_mutex);
    amdf_assert(mutex_error == 0);
    (void)mutex_error;
    const amdf_allocator_t host_allocator = instance->host_allocator;
    amdf_free(host_allocator, instance);
  }
  return status;
}

void amdf_platform_instance_lock_native(amdf_platform_instance_t* instance) {
  const int error = pthread_mutex_lock(&instance->native_mutex);
  amdf_assert(error == 0);
  (void)error;
}

uint64_t amdf_platform_instance_host_allocation_granularity(
    const amdf_platform_instance_t* instance) {
  return instance->page_size;
}

void amdf_platform_instance_unlock_native(amdf_platform_instance_t* instance) {
  const int error = pthread_mutex_unlock(&instance->native_mutex);
  amdf_assert(error == 0);
  (void)error;
}
