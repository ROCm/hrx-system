// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HIP Passthrough Library
//
// This library acts as a transparent proxy for HIP calls. It:
// 1. Loads the real HIP library from HIP_PASSTHROUGH_BACKEND_LIB env var
// 2. Optionally loads an interceptor library from HIP_PASSTHROUGH_INTERCEPTOR env var
// 3. Forwards all HIP calls through the interceptor (if present) or directly
//
// Environment variables:
// - HIP_PASSTHROUGH_BACKEND_LIB: Path to real HIP library (required)
//   Example: /opt/rocm/lib/libamdhip64.so
// - HIP_PASSTHROUGH_INTERCEPTOR: Path to interceptor library (optional)
//   Example: /path/to/my_logging_interceptor.so
//
// The interceptor library must export:
// - hip_function_table_t* hip_interceptor_init(hip_function_table_t* real_funcs)
// - void hip_interceptor_shutdown(void)  (optional)
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "passthrough/hip_function_table.h"

//===----------------------------------------------------------------------===//
// Global State
//===----------------------------------------------------------------------===//

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static void* g_backend_lib = NULL;
static void* g_interceptor_lib = NULL;
static hip_function_table_t g_real_table = {0};
static hip_function_table_t* g_active_table = NULL;
static pfn_hip_interceptor_shutdown g_interceptor_shutdown = NULL;

//===----------------------------------------------------------------------===//
// Symbol Loading Helpers
//===----------------------------------------------------------------------===//

#define LOAD_SYMBOL(lib, table, name)                                    \
  do {                                                                   \
    (table)->name = (pfn_##name)dlsym(lib, #name);                       \
  } while (0)

static bool load_hip_symbols(void* lib, hip_function_table_t* table) {
  table->version = 1;
  table->struct_size = sizeof(hip_function_table_t);

  // Device Management
  LOAD_SYMBOL(lib, table, hipInit);
  LOAD_SYMBOL(lib, table, hipDriverGetVersion);
  LOAD_SYMBOL(lib, table, hipRuntimeGetVersion);
  LOAD_SYMBOL(lib, table, hipGetDevice);
  LOAD_SYMBOL(lib, table, hipGetDeviceCount);
  LOAD_SYMBOL(lib, table, hipSetDevice);
  LOAD_SYMBOL(lib, table, hipDeviceReset);
  LOAD_SYMBOL(lib, table, hipDeviceSynchronize);
  LOAD_SYMBOL(lib, table, hipGetDeviceProperties);
  LOAD_SYMBOL(lib, table, hipDeviceGetAttribute);
  LOAD_SYMBOL(lib, table, hipDeviceGetName);

  // Memory Management
  LOAD_SYMBOL(lib, table, hipMalloc);
  LOAD_SYMBOL(lib, table, hipFree);
  LOAD_SYMBOL(lib, table, hipHostMalloc);
  LOAD_SYMBOL(lib, table, hipHostFree);
  LOAD_SYMBOL(lib, table, hipMemGetInfo);

  // Memory Copy
  LOAD_SYMBOL(lib, table, hipMemcpy);
  LOAD_SYMBOL(lib, table, hipMemcpyAsync);
  LOAD_SYMBOL(lib, table, hipMemset);
  LOAD_SYMBOL(lib, table, hipMemsetAsync);

  // Stream Management
  LOAD_SYMBOL(lib, table, hipStreamCreate);
  LOAD_SYMBOL(lib, table, hipStreamCreateWithFlags);
  LOAD_SYMBOL(lib, table, hipStreamDestroy);
  LOAD_SYMBOL(lib, table, hipStreamSynchronize);
  LOAD_SYMBOL(lib, table, hipStreamQuery);
  LOAD_SYMBOL(lib, table, hipStreamWaitEvent);

  // Event Management
  LOAD_SYMBOL(lib, table, hipEventCreate);
  LOAD_SYMBOL(lib, table, hipEventCreateWithFlags);
  LOAD_SYMBOL(lib, table, hipEventDestroy);
  LOAD_SYMBOL(lib, table, hipEventRecord);
  LOAD_SYMBOL(lib, table, hipEventSynchronize);
  LOAD_SYMBOL(lib, table, hipEventQuery);
  LOAD_SYMBOL(lib, table, hipEventElapsedTime);

  // Module Management
  LOAD_SYMBOL(lib, table, hipModuleLoad);
  LOAD_SYMBOL(lib, table, hipModuleLoadData);
  LOAD_SYMBOL(lib, table, hipModuleUnload);
  LOAD_SYMBOL(lib, table, hipModuleGetFunction);
  LOAD_SYMBOL(lib, table, hipModuleGetGlobal);
  LOAD_SYMBOL(lib, table, hipModuleLaunchKernel);
  LOAD_SYMBOL(lib, table, hipLaunchKernel);
  LOAD_SYMBOL(lib, table, hipExtModuleLaunchKernel);

  // Fat Binary Registration
  LOAD_SYMBOL(lib, table, __hipRegisterFatBinary);
  LOAD_SYMBOL(lib, table, __hipUnregisterFatBinary);
  LOAD_SYMBOL(lib, table, __hipRegisterFunction);
  LOAD_SYMBOL(lib, table, __hipRegisterVar);

  // Error Handling
  LOAD_SYMBOL(lib, table, hipGetErrorString);
  LOAD_SYMBOL(lib, table, hipGetErrorName);
  LOAD_SYMBOL(lib, table, hipGetLastError);
  LOAD_SYMBOL(lib, table, hipPeekAtLastError);

  return table->hipInit != NULL;
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

static void passthrough_init(void) {
  // Load backend library
  const char* backend_path = getenv("HIP_PASSTHROUGH_BACKEND_LIB");
  if (!backend_path) {
    fprintf(stderr, "passthrough: error: HIP_PASSTHROUGH_BACKEND_LIB not set\n");
    fprintf(stderr, "passthrough: set it to the path of the real HIP library\n");
    fprintf(stderr, "passthrough: example: export HIP_PASSTHROUGH_BACKEND_LIB=/opt/rocm/lib/libamdhip64.so\n");
    abort();
  }

  g_backend_lib = dlopen(backend_path, RTLD_NOW | RTLD_LOCAL);
  if (!g_backend_lib) {
    fprintf(stderr, "passthrough: error: failed to load backend library: %s\n",
            backend_path);
    fprintf(stderr, "passthrough: dlerror: %s\n", dlerror());
    abort();
  }

  // Load symbols from backend
  if (!load_hip_symbols(g_backend_lib, &g_real_table)) {
    fprintf(stderr, "passthrough: error: failed to load required symbols\n");
    abort();
  }

  // Default to using real table directly
  g_active_table = &g_real_table;

  // Optionally load interceptor library
  const char* interceptor_path = getenv("HIP_PASSTHROUGH_INTERCEPTOR");
  if (interceptor_path && *interceptor_path) {
    g_interceptor_lib = dlopen(interceptor_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_interceptor_lib) {
      fprintf(stderr, "passthrough: warning: failed to load interceptor: %s\n",
              interceptor_path);
      fprintf(stderr, "passthrough: dlerror: %s\n", dlerror());
    } else {
      // Look for init function
      pfn_hip_interceptor_init init_fn =
          (pfn_hip_interceptor_init)dlsym(g_interceptor_lib, "hip_interceptor_init");
      if (init_fn) {
        hip_function_table_t* interceptor_table = init_fn(&g_real_table);
        if (interceptor_table) {
          g_active_table = interceptor_table;
          fprintf(stderr, "passthrough: interceptor loaded: %s\n", interceptor_path);
        }
      } else {
        fprintf(stderr, "passthrough: warning: interceptor missing hip_interceptor_init\n");
      }

      // Look for shutdown function
      g_interceptor_shutdown =
          (pfn_hip_interceptor_shutdown)dlsym(g_interceptor_lib, "hip_interceptor_shutdown");
    }
  }
}

static void ensure_initialized(void) {
  pthread_once(&g_init_once, passthrough_init);
}

//===----------------------------------------------------------------------===//
// Library Destructor
//===----------------------------------------------------------------------===//

__attribute__((destructor))
static void passthrough_fini(void) {
  if (g_interceptor_shutdown) {
    g_interceptor_shutdown();
  }
  if (g_interceptor_lib) {
    dlclose(g_interceptor_lib);
    g_interceptor_lib = NULL;
  }
  if (g_backend_lib) {
    dlclose(g_backend_lib);
    g_backend_lib = NULL;
  }
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

hip_function_table_t* hip_passthrough_get_real_table(void) {
  ensure_initialized();
  return &g_real_table;
}

hip_function_table_t* hip_passthrough_get_active_table(void) {
  ensure_initialized();
  return g_active_table;
}

//===----------------------------------------------------------------------===//
// HIP API Exports - Forward all calls through active table
//===----------------------------------------------------------------------===//

// Device Management
hipError_t hipInit(unsigned int flags) {
  ensure_initialized();
  if (!g_active_table->hipInit) return 1;
  return g_active_table->hipInit(flags);
}

hipError_t hipGetDevice(int* deviceId) {
  ensure_initialized();
  if (!g_active_table->hipGetDevice) return 1;
  return g_active_table->hipGetDevice(deviceId);
}

hipError_t hipGetDeviceCount(int* count) {
  ensure_initialized();
  if (!g_active_table->hipGetDeviceCount) return 1;
  return g_active_table->hipGetDeviceCount(count);
}

hipError_t hipSetDevice(int deviceId) {
  ensure_initialized();
  if (!g_active_table->hipSetDevice) return 1;
  return g_active_table->hipSetDevice(deviceId);
}

hipError_t hipDeviceSynchronize(void) {
  ensure_initialized();
  if (!g_active_table->hipDeviceSynchronize) return 1;
  return g_active_table->hipDeviceSynchronize();
}

// Memory Management
hipError_t hipMalloc(void** ptr, size_t size) {
  ensure_initialized();
  if (!g_active_table->hipMalloc) return 1;
  return g_active_table->hipMalloc(ptr, size);
}

hipError_t hipFree(void* ptr) {
  ensure_initialized();
  if (!g_active_table->hipFree) return 1;
  return g_active_table->hipFree(ptr);
}

// Memory Copy
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  ensure_initialized();
  if (!g_active_table->hipMemcpy) return 1;
  return g_active_table->hipMemcpy(dst, src, sizeBytes, kind);
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
  ensure_initialized();
  if (!g_active_table->hipMemcpyAsync) return 1;
  return g_active_table->hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

// Stream Management
hipError_t hipStreamCreate(hipStream_t* stream) {
  ensure_initialized();
  if (!g_active_table->hipStreamCreate) return 1;
  return g_active_table->hipStreamCreate(stream);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  ensure_initialized();
  if (!g_active_table->hipStreamDestroy) return 1;
  return g_active_table->hipStreamDestroy(stream);
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  ensure_initialized();
  if (!g_active_table->hipStreamSynchronize) return 1;
  return g_active_table->hipStreamSynchronize(stream);
}

// Event Management
hipError_t hipEventCreate(hipEvent_t* event) {
  ensure_initialized();
  if (!g_active_table->hipEventCreate) return 1;
  return g_active_table->hipEventCreate(event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  ensure_initialized();
  if (!g_active_table->hipEventDestroy) return 1;
  return g_active_table->hipEventDestroy(event);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  ensure_initialized();
  if (!g_active_table->hipEventRecord) return 1;
  return g_active_table->hipEventRecord(event, stream);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  ensure_initialized();
  if (!g_active_table->hipEventSynchronize) return 1;
  return g_active_table->hipEventSynchronize(event);
}

// Module Management
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra) {
  ensure_initialized();
  if (!g_active_table->hipModuleLaunchKernel) return 1;
  return g_active_table->hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
}

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream) {
  ensure_initialized();
  if (!g_active_table->hipLaunchKernel) return 1;
  return g_active_table->hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

// Error Handling
hipError_t hipGetLastError(void) {
  ensure_initialized();
  if (!g_active_table->hipGetLastError) return 0;
  return g_active_table->hipGetLastError();
}

const char* hipGetErrorString(hipError_t hipError) {
  ensure_initialized();
  if (!g_active_table->hipGetErrorString) return "unknown";
  return g_active_table->hipGetErrorString(hipError);
}

const char* hipGetErrorName(hipError_t hipError) {
  ensure_initialized();
  if (!g_active_table->hipGetErrorName) return "unknown";
  return g_active_table->hipGetErrorName(hipError);
}

// Fat Binary Registration
void** __hipRegisterFatBinary(const void* data) {
  ensure_initialized();
  if (!g_active_table->__hipRegisterFatBinary) return NULL;
  return g_active_table->__hipRegisterFatBinary(data);
}

void __hipUnregisterFatBinary(void** fatCubinHandle) {
  ensure_initialized();
  if (g_active_table->__hipUnregisterFatBinary) {
    g_active_table->__hipUnregisterFatBinary(fatCubinHandle);
  }
}

void __hipRegisterFunction(void** fatCubinHandle, const char* hostFun,
                           char* deviceFun, const char* deviceName,
                           int thread_limit, void* tid, void* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize) {
  ensure_initialized();
  if (g_active_table->__hipRegisterFunction) {
    g_active_table->__hipRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                                          deviceName, thread_limit, tid, bid,
                                          blockDim, gridDim, wSize);
  }
}

void __hipRegisterVar(void** fatCubinHandle, char* hostVar, char* deviceAddress,
                      const char* deviceName, int ext, size_t size,
                      int constant, int global) {
  ensure_initialized();
  if (g_active_table->__hipRegisterVar) {
    g_active_table->__hipRegisterVar(fatCubinHandle, hostVar, deviceAddress,
                                     deviceName, ext, size, constant, global);
  }
}
