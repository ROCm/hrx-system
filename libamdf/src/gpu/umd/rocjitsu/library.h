// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_ROCJITSU_LIBRARY_H_
#define AMDF_SRC_GPU_UMD_ROCJITSU_LIBRARY_H_

#include "amdf/amdf.h"
#include "rocjitsu/base/rj_version.h"
#include "rocjitsu/vm/rj_vm_config.h"

// The pinned headers have unversioned structures. This revision is qualified
// as one ABI; loading another revision requires a matching header update.
#define AMDF_ROCJITSU_REVISION "664fdfad1115688d310594cf232e9b9ac10a3c42"

#define AMDF_ROCJITSU_SYMBOLS(_)                                             \
  _(const char*, rj_get_version_string, (void))                              \
  _(rj_status_t, rj_vm_config_create_from_string,                            \
    (const char*, rj_vm_config_t**))                                         \
  _(void, rj_vm_config_destroy, (rj_vm_config_t*))                           \
  _(rj_status_t, rj_vm_config_gpu_count, (const rj_vm_config_t*, uint32_t*)) \
  _(rj_status_t, rj_vm_config_gpu_info_at,                                   \
    (const rj_vm_config_t*, uint32_t, rj_vm_gpu_info_t*))                    \
  _(rj_status_t, rj_vm_create_from_config,                                   \
    (const rj_vm_config_t*, rj_vm_mode_t, rj_vm_t**))                        \
  _(void, rj_vm_destroy, (rj_vm_t*))                                         \
  _(rj_status_t, rj_vm_run, (rj_vm_t*, uint64_t*))                           \
  _(void, rj_vm_request_exit, (rj_vm_t*, const char*))                       \
  _(rj_status_t, rj_vm_device_open, (rj_vm_t*, rj_client_pid_t, uint32_t*))  \
  _(rj_status_t, rj_vm_device_close, (rj_vm_t*, uint32_t))                   \
  _(rj_status_t, rj_vm_execute_as, (rj_vm_t*, uint32_t, rj_vm_cmd_t*))       \
  _(rj_status_t, rj_vm_device_map_as, (rj_vm_t*, uint32_t, rj_vm_map_t*))    \
  _(rj_status_t, rj_vm_device_unmap_as, (rj_vm_t*, uint32_t, rj_vm_unmap_t*))

typedef struct amdf_rocjitsu_library_t {
  // Owned loader reference, live until all simulator objects are destroyed.
  void* handle;
#define AMDF_ROCJITSU_DECLARE(result, name, arguments) \
  /* Immutable C entry point borrowed from handle. */  \
  result(*name) arguments;
  AMDF_ROCJITSU_SYMBOLS(AMDF_ROCJITSU_DECLARE)
#undef AMDF_ROCJITSU_DECLARE
} amdf_rocjitsu_library_t;

// Loads and resolves the qualified ABI transactionally. A nonempty path wins
// over LIBAMDF_ROCJITSU_PATH; absent both, uses the librocjitsu.so loader name.
amdf_status_t amdf_rocjitsu_library_open(const char* path,
                                         amdf_rocjitsu_library_t* out_library);

// Releases the explicit loader reference after all workers and objects retire.
void amdf_rocjitsu_library_close(amdf_rocjitsu_library_t* library);

static inline amdf_status_t amdf_rocjitsu_status(rj_status_t status) {
  return status == ROCJITSU_STATUS_SUCCESS
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_ROCJITSU, (uint32_t)status);
}

#endif  // AMDF_SRC_GPU_UMD_ROCJITSU_LIBRARY_H_
