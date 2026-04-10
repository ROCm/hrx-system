// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Simple HIP Passthrough
//
// This library simply loads the real HIP library with RTLD_GLOBAL and
// RTLD_DEEPBIND so its symbols are used for all lookups. This allows
// interception of init/shutdown while letting all other calls pass through
// transparently.
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *g_real_lib = NULL;
static FILE *g_log_file = NULL;

// Function pointers for the few functions we want to intercept
typedef int (*hipInit_fn)(unsigned int);
typedef int (*hipGetDevice_fn)(int *);
typedef int (*hipGetDeviceCount_fn)(int *);
typedef int (*hipSetDevice_fn)(int);
typedef int (*hipDeviceSynchronize_fn)(void);
typedef int (*hipMalloc_fn)(void **, size_t);
typedef int (*hipFree_fn)(void *);
typedef int (*hipMemcpy_fn)(void *, const void *, size_t, int);

static hipInit_fn real_hipInit = NULL;
static hipGetDevice_fn real_hipGetDevice = NULL;
static hipGetDeviceCount_fn real_hipGetDeviceCount = NULL;
static hipSetDevice_fn real_hipSetDevice = NULL;
static hipDeviceSynchronize_fn real_hipDeviceSynchronize = NULL;
static hipMalloc_fn real_hipMalloc = NULL;
static hipFree_fn real_hipFree = NULL;
static hipMemcpy_fn real_hipMemcpy = NULL;

static void log_call(const char *name) {
  if (g_log_file) {
    fprintf(g_log_file, "%s\n", name);
    fflush(g_log_file);
  }
}

__attribute__((constructor(101))) static void passthrough_init(void) {
  // Open log file if specified
  const char *log_path = getenv("HIP_LOG_FILE");
  if (log_path && *log_path) {
    g_log_file = fopen(log_path, "w");
  }

  // Load real HIP library
  const char *lib_path = getenv("HIP_PASSTHROUGH_BACKEND_LIB");
  if (!lib_path) {
    lib_path = "/opt/rocm/lib/libamdhip64.so.7";
  }

  if (g_log_file) {
    fprintf(g_log_file, "passthrough: loading %s\n", lib_path);
    fflush(g_log_file);
  }

  g_real_lib = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
  if (!g_real_lib) {
    fprintf(stderr, "passthrough: failed to load %s: %s\n", lib_path,
            dlerror());
    return;
  }

  // Load function pointers for functions we want to log
  real_hipInit = (hipInit_fn)dlsym(g_real_lib, "hipInit");
  real_hipGetDevice = (hipGetDevice_fn)dlsym(g_real_lib, "hipGetDevice");
  real_hipGetDeviceCount =
      (hipGetDeviceCount_fn)dlsym(g_real_lib, "hipGetDeviceCount");
  real_hipSetDevice = (hipSetDevice_fn)dlsym(g_real_lib, "hipSetDevice");
  real_hipDeviceSynchronize =
      (hipDeviceSynchronize_fn)dlsym(g_real_lib, "hipDeviceSynchronize");
  real_hipMalloc = (hipMalloc_fn)dlsym(g_real_lib, "hipMalloc");
  real_hipFree = (hipFree_fn)dlsym(g_real_lib, "hipFree");
  real_hipMemcpy = (hipMemcpy_fn)dlsym(g_real_lib, "hipMemcpy");

  if (g_log_file) {
    fprintf(g_log_file, "passthrough: initialized successfully\n");
    fflush(g_log_file);
  }
}

__attribute__((destructor)) static void passthrough_fini(void) {
  if (g_log_file) {
    fprintf(g_log_file, "passthrough: shutting down\n");
    fclose(g_log_file);
    g_log_file = NULL;
  }
}

// Intercepted functions with logging
__attribute__((visibility("default"))) int hipInit(unsigned int flags) {
  if (!real_hipInit)
    return 1;
  int ret = real_hipInit(flags);
  if (g_log_file) {
    fprintf(g_log_file, "hipInit(flags=0x%x) -> %d\n", flags, ret);
    fflush(g_log_file);
  }
  return ret;
}

__attribute__((visibility("default"))) int hipGetDevice(int *deviceId) {
  if (!real_hipGetDevice)
    return 1;
  int ret = real_hipGetDevice(deviceId);
  if (g_log_file) {
    fprintf(g_log_file, "hipGetDevice() -> device=%d, ret=%d\n",
            deviceId ? *deviceId : -1, ret);
    fflush(g_log_file);
  }
  return ret;
}

__attribute__((visibility("default"))) int hipGetDeviceCount(int *count) {
  if (!real_hipGetDeviceCount)
    return 1;
  int ret = real_hipGetDeviceCount(count);
  if (g_log_file) {
    fprintf(g_log_file, "hipGetDeviceCount() -> count=%d, ret=%d\n",
            count ? *count : -1, ret);
    fflush(g_log_file);
  }
  return ret;
}

__attribute__((visibility("default"))) int hipSetDevice(int deviceId) {
  if (!real_hipSetDevice)
    return 1;
  int ret = real_hipSetDevice(deviceId);
  if (g_log_file) {
    fprintf(g_log_file, "hipSetDevice(%d) -> %d\n", deviceId, ret);
    fflush(g_log_file);
  }
  return ret;
}

__attribute__((visibility("default"))) int hipDeviceSynchronize(void) {
  if (!real_hipDeviceSynchronize)
    return 1;
  int ret = real_hipDeviceSynchronize();
  if (g_log_file) {
    fprintf(g_log_file, "hipDeviceSynchronize() -> %d\n", ret);
    fflush(g_log_file);
  }
  return ret;
}

__attribute__((visibility("default"))) int hipMalloc(void **ptr, size_t size) {
  if (!real_hipMalloc)
    return 1;
  int ret = real_hipMalloc(ptr, size);
  if (g_log_file) {
    fprintf(g_log_file, "hipMalloc(size=%zu) -> ptr=%p, ret=%d\n", size,
            ptr ? *ptr : NULL, ret);
    fflush(g_log_file);
  }
  return ret;
}

__attribute__((visibility("default"))) int hipFree(void *ptr) {
  if (!real_hipFree)
    return 1;
  int ret = real_hipFree(ptr);
  if (g_log_file) {
    fprintf(g_log_file, "hipFree(%p) -> %d\n", ptr, ret);
    fflush(g_log_file);
  }
  return ret;
}

__attribute__((visibility("default"))) int hipMemcpy(void *dst, const void *src,
                                                     size_t size, int kind) {
  if (!real_hipMemcpy)
    return 1;
  int ret = real_hipMemcpy(dst, src, size, kind);
  if (g_log_file) {
    const char *kind_str = "unknown";
    switch (kind) {
    case 0:
      kind_str = "H2H";
      break;
    case 1:
      kind_str = "H2D";
      break;
    case 2:
      kind_str = "D2H";
      break;
    case 3:
      kind_str = "D2D";
      break;
    case 4:
      kind_str = "Default";
      break;
    }
    fprintf(g_log_file, "hipMemcpy(dst=%p, src=%p, size=%zu, kind=%s) -> %d\n",
            dst, src, size, kind_str, ret);
    fflush(g_log_file);
  }
  return ret;
}
