// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/rocjitsu/library.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

amdf_status_t amdf_rocjitsu_library_open(const char* path,
                                         amdf_rocjitsu_library_t* out_library) {
  if (path == NULL || path[0] == 0) {
    path = getenv("LIBAMDF_ROCJITSU_PATH");
  }
  if (path == NULL || path[0] == 0) {
    path = "librocjitsu.so";
  }
  amdf_rocjitsu_library_t library = {0};
  library.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (library.handle == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  amdf_status_t status = AMDF_STATUS_OK;
#define AMDF_ROCJITSU_LOAD(result, name, arguments)                     \
  if (amdf_status_is_ok(status)) {                                      \
    void* symbol = dlsym(library.handle, #name);                        \
    if (symbol == NULL) {                                               \
      status = amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH); \
    } else {                                                            \
      _Static_assert(sizeof(library.name) == sizeof(symbol),            \
                     "POSIX symbol representation");                    \
      memcpy(&library.name, &symbol, sizeof(symbol));                   \
    }                                                                   \
  }
  AMDF_ROCJITSU_SYMBOLS(AMDF_ROCJITSU_LOAD)
#undef AMDF_ROCJITSU_LOAD
  if (amdf_status_is_ok(status)) {
    const char* identity = library.rj_get_version_string();
    if (identity == NULL ||
        strstr(identity, "\ngit revision: " AMDF_ROCJITSU_REVISION "\n") ==
            NULL) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_library = library;
  } else {
    amdf_rocjitsu_library_close(&library);
  }
  return status;
}

void amdf_rocjitsu_library_close(amdf_rocjitsu_library_t* library) {
  if (library->handle != NULL) {
    // This owner releases exactly the reference acquired by dlopen. A loader
    // rejection indicates invalid handle ownership, not a retryable release.
    const int result = dlclose(library->handle);
    amdf_assert(result == 0);
    (void)result;
  }
  library->handle = NULL;
}
