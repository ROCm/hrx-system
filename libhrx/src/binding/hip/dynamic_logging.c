// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/dynamic_logging.h"

#include <stdarg.h>
#include <stdio.h>

#include "binding/hip/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"

enum {
  HRX_HIP_LOG_LEVEL_INFO = 3,
  HRX_HIP_LOG_MASK_API = 1,
};

typedef struct hrx_hip_dynamic_logging_state_t {
  // Serializes configuration changes and output writes.
  iree_slim_mutex_t mutex;
  // Logging verbosity used by the next enable operation.
  size_t configured_level;
  // Maximum logging storage requested for the next enable operation.
  size_t configured_size;
  // Logging categories used by the next enable operation.
  size_t configured_mask;
} hrx_hip_dynamic_logging_state_t;

static iree_once_flag hrx_hip_dynamic_logging_once = IREE_ONCE_FLAG_INIT;
static hrx_hip_dynamic_logging_state_t hrx_hip_dynamic_logging_state;
iree_atomic_int32_t hrx_hip_dynamic_logging_api_enabled =
    IREE_ATOMIC_VAR_INIT(0);

static void hrx_hip_dynamic_logging_initialize(void) {
  iree_slim_mutex_initialize(&hrx_hip_dynamic_logging_state.mutex);
}

static void hrx_hip_dynamic_logging_lock(void) {
  iree_call_once(&hrx_hip_dynamic_logging_once,
                 hrx_hip_dynamic_logging_initialize);
  iree_slim_mutex_lock(&hrx_hip_dynamic_logging_state.mutex);
}

void hrx_hip_dynamic_logging_write(const char* format, ...) {
  hrx_hip_dynamic_logging_lock();
  // Disable waits for any active writer and clears this flag while holding the
  // same mutex. Rechecking prevents a caller that observed the old state from
  // writing after hipExtDisableLogging returns.
  if (iree_atomic_load(&hrx_hip_dynamic_logging_api_enabled,
                       iree_memory_order_relaxed) != 0) {
    va_list argument_list;
    va_start(argument_list, format);
    vfprintf(stderr, format, argument_list);
    va_end(argument_list);
    fflush(stderr);
  }
  iree_slim_mutex_unlock(&hrx_hip_dynamic_logging_state.mutex);
}

HIPAPI hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size,
                                         size_t log_mask) {
  HRX_HIP_DYNAMIC_LOG("[HIP_API] %s called\n", __func__);
  hrx_hip_dynamic_logging_lock();
  hrx_hip_dynamic_logging_state.configured_level = log_level;
  hrx_hip_dynamic_logging_state.configured_size = log_size;
  hrx_hip_dynamic_logging_state.configured_mask = log_mask;
  iree_slim_mutex_unlock(&hrx_hip_dynamic_logging_state.mutex);
  return hipSuccess;
}

HIPAPI hipError_t hipExtEnableLogging(void) {
  HRX_HIP_DYNAMIC_LOG("[HIP_API] %s called\n", __func__);
  hrx_hip_dynamic_logging_lock();
  const bool is_api_enabled = hrx_hip_dynamic_logging_state.configured_level >=
                                  HRX_HIP_LOG_LEVEL_INFO &&
                              (hrx_hip_dynamic_logging_state.configured_mask &
                               HRX_HIP_LOG_MASK_API) != 0;
  iree_atomic_store(&hrx_hip_dynamic_logging_api_enabled, is_api_enabled,
                    iree_memory_order_relaxed);
  iree_slim_mutex_unlock(&hrx_hip_dynamic_logging_state.mutex);
  return hipSuccess;
}

HIPAPI hipError_t hipExtDisableLogging(void) {
  HRX_HIP_DYNAMIC_LOG("[HIP_API] %s called\n", __func__);
  hrx_hip_dynamic_logging_lock();
  iree_atomic_store(&hrx_hip_dynamic_logging_api_enabled, 0,
                    iree_memory_order_relaxed);
  iree_slim_mutex_unlock(&hrx_hip_dynamic_logging_state.mutex);
  return hipSuccess;
}
