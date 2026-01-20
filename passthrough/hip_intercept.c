// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HIP Intercept Library
//
// This library is symlinked in place of the real HIP library. It:
// 1. Loads the real HIP library from HIP_PASSTHROUGH_BACKEND_LIB env var
//    (defaults to /opt/rocm/lib/libamdhip64.so)
// 2. Optionally loads an interception library from HIP_INTERCEPTION_LIBRARY
// 3. Each HIP function forwards to the real library (or interceptor if loaded)
//
// Environment variables:
// - HIP_PASSTHROUGH_BACKEND_LIB: Path to real HIP library
//   Default: /opt/rocm/lib/libamdhip64.so
// - HIP_INTERCEPTION_LIBRARY: Path to interception library (optional)
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
// Symbol Loading
//===----------------------------------------------------------------------===//

#define LOAD_SYM(name) \
  g_real_table.name = (pfn_##name)dlsym(g_backend_lib, #name)

static void load_all_symbols(void) {
  // Device Management
  LOAD_SYM(hipInit);
  LOAD_SYM(hipDriverGetVersion);
  LOAD_SYM(hipRuntimeGetVersion);
  LOAD_SYM(hipGetDevice);
  LOAD_SYM(hipGetDeviceCount);
  LOAD_SYM(hipSetDevice);
  LOAD_SYM(hipDeviceReset);
  LOAD_SYM(hipDeviceSynchronize);
  LOAD_SYM(hipGetDeviceProperties);
  LOAD_SYM(hipDeviceGetAttribute);
  LOAD_SYM(hipDeviceGetName);

  // Memory Management
  LOAD_SYM(hipMalloc);
  LOAD_SYM(hipFree);
  LOAD_SYM(hipHostMalloc);
  LOAD_SYM(hipHostFree);
  LOAD_SYM(hipMemGetInfo);

  // Memory Copy
  LOAD_SYM(hipMemcpy);
  LOAD_SYM(hipMemcpyAsync);
  LOAD_SYM(hipMemset);
  LOAD_SYM(hipMemsetAsync);

  // Stream Management
  LOAD_SYM(hipStreamCreate);
  LOAD_SYM(hipStreamCreateWithFlags);
  LOAD_SYM(hipStreamDestroy);
  LOAD_SYM(hipStreamSynchronize);
  LOAD_SYM(hipStreamQuery);
  LOAD_SYM(hipStreamWaitEvent);

  // Event Management
  LOAD_SYM(hipEventCreate);
  LOAD_SYM(hipEventCreateWithFlags);
  LOAD_SYM(hipEventDestroy);
  LOAD_SYM(hipEventRecord);
  LOAD_SYM(hipEventSynchronize);
  LOAD_SYM(hipEventQuery);
  LOAD_SYM(hipEventElapsedTime);

  // Module Management
  LOAD_SYM(hipModuleLoad);
  LOAD_SYM(hipModuleLoadData);
  LOAD_SYM(hipModuleUnload);
  LOAD_SYM(hipModuleGetFunction);
  LOAD_SYM(hipModuleGetGlobal);
  LOAD_SYM(hipModuleLaunchKernel);
  LOAD_SYM(hipLaunchKernel);
  LOAD_SYM(hipExtModuleLaunchKernel);

  // Fat Binary Registration
  LOAD_SYM(__hipRegisterFatBinary);
  LOAD_SYM(__hipUnregisterFatBinary);
  LOAD_SYM(__hipRegisterFunction);
  LOAD_SYM(__hipRegisterVar);

  // Error Handling
  LOAD_SYM(hipGetErrorString);
  LOAD_SYM(hipGetErrorName);
  LOAD_SYM(hipGetLastError);
  LOAD_SYM(hipPeekAtLastError);

  g_real_table.version = 1;
  g_real_table.struct_size = sizeof(hip_function_table_t);
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

static void intercept_init(void) {
  const char* backend_path = getenv("HIP_PASSTHROUGH_BACKEND_LIB");
  if (!backend_path || !*backend_path) {
    backend_path = "/opt/rocm/lib/libamdhip64.so";
  }

  g_backend_lib = dlopen(backend_path, RTLD_NOW | RTLD_GLOBAL);
  if (!g_backend_lib) {
    fprintf(stderr, "hip_intercept: failed to load backend: %s\n", dlerror());
    abort();
  }

  load_all_symbols();
  g_active_table = &g_real_table;

  // Optionally load interception library
  const char* interceptor_path = getenv("HIP_INTERCEPTION_LIBRARY");
  if (interceptor_path && *interceptor_path) {
    g_interceptor_lib = dlopen(interceptor_path, RTLD_NOW | RTLD_LOCAL);
    if (g_interceptor_lib) {
      pfn_hip_interceptor_init init_fn =
          (pfn_hip_interceptor_init)dlsym(g_interceptor_lib, "hip_interceptor_init");
      if (init_fn) {
        hip_function_table_t* interceptor_table = init_fn(&g_real_table);
        if (interceptor_table) {
          g_active_table = interceptor_table;
        }
      }
      g_interceptor_shutdown =
          (pfn_hip_interceptor_shutdown)dlsym(g_interceptor_lib, "hip_interceptor_shutdown");
    }
  }
}

static void ensure_init(void) {
  pthread_once(&g_init_once, intercept_init);
}

// Force early initialization using .init_array with high priority
// This runs before other constructors and hopefully before symbol resolution
__attribute__((constructor(101)))
static void early_init(void) {
  intercept_init();
}

__attribute__((destructor))
static void intercept_fini(void) {
  if (g_interceptor_shutdown) g_interceptor_shutdown();
  if (g_interceptor_lib) dlclose(g_interceptor_lib);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

hip_function_table_t* hip_passthrough_get_real_table(void) {
  ensure_init();
  return &g_real_table;
}

hip_function_table_t* hip_passthrough_get_active_table(void) {
  ensure_init();
  return g_active_table;
}

//===----------------------------------------------------------------------===//
// HIP Function Exports
//===----------------------------------------------------------------------===//

#define FWD0(ret, name) \
  ret name(void) { ensure_init(); return g_active_table->name ? g_active_table->name() : (ret)0; }

#define FWD1(ret, name, t1, a1) \
  ret name(t1 a1) { ensure_init(); return g_active_table->name ? g_active_table->name(a1) : (ret)0; }

#define FWD2(ret, name, t1, a1, t2, a2) \
  ret name(t1 a1, t2 a2) { ensure_init(); return g_active_table->name ? g_active_table->name(a1, a2) : (ret)0; }

#define FWD3(ret, name, t1, a1, t2, a2, t3, a3) \
  ret name(t1 a1, t2 a2, t3 a3) { ensure_init(); return g_active_table->name ? g_active_table->name(a1, a2, a3) : (ret)0; }

#define FWD4(ret, name, t1, a1, t2, a2, t3, a3, t4, a4) \
  ret name(t1 a1, t2 a2, t3 a3, t4 a4) { ensure_init(); return g_active_table->name ? g_active_table->name(a1, a2, a3, a4) : (ret)0; }

#define FWD5(ret, name, t1, a1, t2, a2, t3, a3, t4, a4, t5, a5) \
  ret name(t1 a1, t2 a2, t3 a3, t4 a4, t5 a5) { ensure_init(); return g_active_table->name ? g_active_table->name(a1, a2, a3, a4, a5) : (ret)0; }

// Device Management
FWD1(hipError_t, hipInit, unsigned int, flags)
FWD1(hipError_t, hipDriverGetVersion, int*, driverVersion)
FWD1(hipError_t, hipRuntimeGetVersion, int*, runtimeVersion)
FWD1(hipError_t, hipGetDevice, int*, deviceId)
FWD1(hipError_t, hipGetDeviceCount, int*, count)
FWD1(hipError_t, hipSetDevice, int, deviceId)
FWD0(hipError_t, hipDeviceReset)
FWD0(hipError_t, hipDeviceSynchronize)
FWD2(hipError_t, hipGetDeviceProperties, hipDeviceProp_t*, prop, int, deviceId)
FWD3(hipError_t, hipDeviceGetAttribute, int*, value, hipDeviceAttribute_t, attr, int, deviceId)
FWD3(hipError_t, hipDeviceGetName, char*, name, int, len, int, deviceId)

// ROCm 6.0 variant of hipGetDeviceProperties
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t* prop, int deviceId) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceProp_t*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetDevicePropertiesR0600");
  return fn ? fn(prop, deviceId) : 1;
}

// Memory Management
FWD2(hipError_t, hipMalloc, void**, ptr, size_t, size)
FWD1(hipError_t, hipFree, void*, ptr)
FWD3(hipError_t, hipHostMalloc, void**, ptr, size_t, size, unsigned int, flags)
FWD1(hipError_t, hipHostFree, void*, ptr)
FWD2(hipError_t, hipMemGetInfo, size_t*, free, size_t*, total)

// Memory Copy
FWD4(hipError_t, hipMemcpy, void*, dst, const void*, src, size_t, sizeBytes, hipMemcpyKind, kind)
FWD5(hipError_t, hipMemcpyAsync, void*, dst, const void*, src, size_t, sizeBytes, hipMemcpyKind, kind, hipStream_t, stream)
FWD3(hipError_t, hipMemset, void*, dst, int, value, size_t, sizeBytes)
FWD4(hipError_t, hipMemsetAsync, void*, dst, int, value, size_t, sizeBytes, hipStream_t, stream)

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyWithStream");
  return fn ? fn(dst, src, sizeBytes, kind, stream) : 1;
}

hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMallocAsync");
  return fn ? fn(dev_ptr, size, stream) : 1;
}

hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFreeAsync");
  return fn ? fn(dev_ptr, stream) : 1;
}

hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t*, const hipMemPoolProps*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolCreate");
  return fn ? fn(mem_pool, pool_props) : 1;
}

hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolDestroy");
  return fn ? fn(mem_pool) : 1;
}

hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetDefaultMemPool");
  return fn ? fn(mem_pool, device) : 1;
}

hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
  ensure_init();
  typedef hipError_t (*pfn)(int, hipMemPool_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceSetMemPool");
  return fn ? fn(device, mem_pool) : 1;
}

hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetMemPool");
  return fn ? fn(mem_pool, device) : 1;
}

hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t, hipMemPool_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMallocFromPoolAsync");
  return fn ? fn(dev_ptr, size, mem_pool, stream) : 1;
}

hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t, hipMemPoolAttr, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolSetAttribute");
  return fn ? fn(mem_pool, attr, value) : 1;
}

hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t, hipMemPoolAttr, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolGetAttribute");
  return fn ? fn(mem_pool, attr, value) : 1;
}

hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolTrimTo");
  return fn ? fn(mem_pool, min_bytes_to_hold) : 1;
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpy2D");
  return fn ? fn(dst, dpitch, src, spitch, width, height, kind) : 1;
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpy2DAsync");
  return fn ? fn(dst, dpitch, src, spitch, width, height, kind, stream) : 1;
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, hipDeviceptr_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyDtoD");
  return fn ? fn(dst, src, sizeBytes) : 1;
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyDtoDAsync");
  return fn ? fn(dst, src, sizeBytes, stream) : 1;
}

hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipDeviceptr_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyDtoH");
  return fn ? fn(dst, src, sizeBytes) : 1;
}

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipDeviceptr_t, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyDtoHAsync");
  return fn ? fn(dst, src, sizeBytes, stream) : 1;
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, const void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyHtoD");
  return fn ? fn(dst, src, sizeBytes) : 1;
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void* src, size_t sizeBytes, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, const void*, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyHtoDAsync");
  return fn ? fn(dst, src, sizeBytes, stream) : 1;
}

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, int, size_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemset2D");
  return fn ? fn(dst, pitch, value, width, height) : 1;
}

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, int, size_t, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemset2DAsync");
  return fn ? fn(dst, pitch, value, width, height, stream) : 1;
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, unsigned char, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemsetD8");
  return fn ? fn(dst, value, count) : 1;
}

hipError_t hipMemsetD8Async(hipDeviceptr_t dst, unsigned char value, size_t count, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, unsigned char, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemsetD8Async");
  return fn ? fn(dst, value, count, stream) : 1;
}

hipError_t hipMemsetD16(hipDeviceptr_t dst, unsigned short value, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, unsigned short, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemsetD16");
  return fn ? fn(dst, value, count) : 1;
}

hipError_t hipMemsetD16Async(hipDeviceptr_t dst, unsigned short value, size_t count, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, unsigned short, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemsetD16Async");
  return fn ? fn(dst, value, count, stream) : 1;
}

hipError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, int, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemsetD32");
  return fn ? fn(dst, value, count) : 1;
}

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t, int, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemsetD32Async");
  return fn ? fn(dst, value, count, stream) : 1;
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  ensure_init();
  typedef hipError_t (*pfn)(hipPointerAttribute_t*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipPointerGetAttributes");
  return fn ? fn(attributes, ptr) : 1;
}

hipError_t hipPointerGetAttribute(void* data, int attribute, hipDeviceptr_t ptr) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, hipDeviceptr_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipPointerGetAttribute");
  return fn ? fn(data, attribute, ptr) : 1;
}

hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, int* attributes, void** data, hipDeviceptr_t ptr) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned int, int*, void**, hipDeviceptr_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDrvPointerGetAttributes");
  return fn ? fn(numAttributes, attributes, data, ptr) : 1;
}

// Function/Kernel attributes - use int for enums
hipError_t hipFuncSetAttribute(const void* func, int attr, int value) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFuncSetAttribute");
  return fn ? fn(func, attr, value) : 1;
}

hipError_t hipFuncSetCacheConfig(const void* func, int config) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFuncSetCacheConfig");
  return fn ? fn(func, config) : 1;
}

hipError_t hipFuncSetSharedMemConfig(const void* func, int config) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFuncSetSharedMemConfig");
  return fn ? fn(func, config) : 1;
}

hipError_t hipFuncGetAttributes(void* attr, const void* func) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFuncGetAttributes");
  return fn ? fn(attr, func) : 1;
}

hipError_t hipFuncGetAttribute(int* value, int attrib, hipFunction_t hfunc) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int, hipFunction_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFuncGetAttribute");
  return fn ? fn(value, attrib, hfunc) : 1;
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f, int blockSize, size_t dynamicSMemSize) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, const void*, int, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipOccupancyMaxActiveBlocksPerMultiprocessor");
  return fn ? fn(numBlocks, f, blockSize, dynamicSMemSize) : 1;
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* f, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, const void*, int, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  return fn ? fn(numBlocks, f, blockSize, dynamicSMemSize, flags) : 1;
}

hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, const void* f, size_t dynamicSMemSize, int blockSizeLimit) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, const void*, size_t, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipOccupancyMaxPotentialBlockSize");
  return fn ? fn(gridSize, blockSize, f, dynamicSMemSize, blockSizeLimit) : 1;
}

hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetStreamPriorityRange");
  return fn ? fn(leastPriority, greatestPriority) : 1;
}

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceCanAccessPeer");
  return fn ? fn(canAccessPeer, deviceId, peerDeviceId) : 1;
}

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceEnablePeerAccess");
  return fn ? fn(peerDeviceId, flags) : 1;
}

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  ensure_init();
  typedef hipError_t (*pfn)(int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceDisablePeerAccess");
  return fn ? fn(peerDeviceId) : 1;
}

hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, const char*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetByPCIBusId");
  return fn ? fn(device, pciBusId) : 1;
}

hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  ensure_init();
  typedef hipError_t (*pfn)(char*, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetPCIBusId");
  return fn ? fn(pciBusId, len, device) : 1;
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
  ensure_init();
  typedef hipError_t (*pfn)(hipIpcMemHandle_t*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipIpcGetMemHandle");
  return fn ? fn(handle, devPtr) : 1;
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, hipIpcMemHandle_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipIpcOpenMemHandle");
  return fn ? fn(devPtr, handle, flags) : 1;
}

hipError_t hipIpcCloseMemHandle(void* devPtr) {
  ensure_init();
  typedef hipError_t (*pfn)(void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipIpcCloseMemHandle");
  return fn ? fn(devPtr) : 1;
}

hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  ensure_init();
  typedef hipError_t (*pfn)(hipIpcEventHandle_t*, hipEvent_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipIpcGetEventHandle");
  return fn ? fn(handle, event) : 1;
}

hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  ensure_init();
  typedef hipError_t (*pfn)(hipEvent_t*, hipIpcEventHandle_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipIpcOpenEventHandle");
  return fn ? fn(event, handle) : 1;
}

// Host function and callback
typedef void (*hipHostFn_t)(void* userData);
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn_cb, void* userData) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipHostFn_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipLaunchHostFunc");
  return fn ? fn(stream, fn_cb, userData) : 1;
}

typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipStreamCallback_t, void*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamAddCallback");
  return fn ? fn(stream, callback, userData, flags) : 1;
}

hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus, unsigned long long* pId) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipStreamCaptureStatus*, unsigned long long*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamGetCaptureInfo");
  return fn ? fn(stream, pCaptureStatus, pId) : 1;
}

hipError_t hipStreamBeginCapture(hipStream_t stream, int mode) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamBeginCapture");
  return fn ? fn(stream, mode) : 1;
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipGraph_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamEndCapture");
  return fn ? fn(stream, pGraph) : 1;
}

// Graph functions
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphCreate");
  return fn ? fn(pGraph, flags) : 1;
}

hipError_t hipGraphDestroy(hipGraph_t graph) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphDestroy");
  return fn ? fn(graph) : 1;
}

hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph, hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphExec_t*, hipGraph_t, hipGraphNode_t*, char*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphInstantiate");
  return fn ? fn(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize) : 1;
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphExec_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphLaunch");
  return fn ? fn(graphExec, stream) : 1;
}

hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphExec_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphExecDestroy");
  return fn ? fn(graphExec) : 1;
}

// Host memory
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipHostRegister");
  return fn ? fn(hostPtr, sizeBytes, flags) : 1;
}

hipError_t hipHostUnregister(void* hostPtr) {
  ensure_init();
  typedef hipError_t (*pfn)(void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipHostUnregister");
  return fn ? fn(hostPtr) : 1;
}

hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, void*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipHostGetDevicePointer");
  return fn ? fn(devPtr, hostPtr, flags) : 1;
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned int*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipHostGetFlags");
  return fn ? fn(flagsPtr, hostPtr) : 1;
}

// Module functions
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions, int* options, void** optionValues) {
  ensure_init();
  typedef hipError_t (*pfn)(hipModule_t*, const void*, unsigned int, int*, void**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleLoadDataEx");
  return fn ? fn(module, image, numOptions, options, optionValues) : 1;
}

hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, hipFunction_t, size_t, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleOccupancyMaxPotentialBlockSize");
  return fn ? fn(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit) : 1;
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, hipFunction_t, int, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
  return fn ? fn(numBlocks, f, blockSize, dynSharedMemPerBlk) : 1;
}

// Device memory allocation
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t*, size_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMallocPitch");
  return fn ? fn(ptr, pitch, width, height) : 1;
}

hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMallocManaged");
  return fn ? fn(dev_ptr, size, flags) : 1;
}

hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, size_t, int, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPrefetchAsync");
  return fn ? fn(dev_ptr, count, device, stream) : 1;
}

hipError_t hipMemAdvise(const void* dev_ptr, size_t count, int advice, int device) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, size_t, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemAdvise");
  return fn ? fn(dev_ptr, count, advice, device) : 1;
}

hipError_t hipMemRangeGetAttribute(void* data, size_t dataSize, int attribute, const void* dev_ptr, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, int, const void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemRangeGetAttribute");
  return fn ? fn(data, dataSize, attribute, dev_ptr, count) : 1;
}

// Driver API
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
  ensure_init();
  typedef hipError_t (*pfn)(hipCtx_t*, unsigned int, hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxCreate");
  return fn ? fn(ctx, flags, device) : 1;
}

hipError_t hipCtxDestroy(hipCtx_t ctx) {
  ensure_init();
  typedef hipError_t (*pfn)(hipCtx_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxDestroy");
  return fn ? fn(ctx) : 1;
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  ensure_init();
  typedef hipError_t (*pfn)(hipCtx_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxGetCurrent");
  return fn ? fn(ctx) : 1;
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  ensure_init();
  typedef hipError_t (*pfn)(hipCtx_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxSetCurrent");
  return fn ? fn(ctx) : 1;
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  ensure_init();
  typedef hipError_t (*pfn)(hipCtx_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxPushCurrent");
  return fn ? fn(ctx) : 1;
}

hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  ensure_init();
  typedef hipError_t (*pfn)(hipCtx_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxPopCurrent");
  return fn ? fn(ctx) : 1;
}

hipError_t hipCtxGetDevice(hipDevice_t* device) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDevice_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxGetDevice");
  return fn ? fn(device) : 1;
}

hipError_t hipCtxSynchronize(void) {
  ensure_init();
  typedef hipError_t (*pfn)(void);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCtxSynchronize");
  return fn ? fn() : 1;
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  ensure_init();
  typedef hipError_t (*pfn)(hipCtx_t*, hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDevicePrimaryCtxRetain");
  return fn ? fn(pctx, dev) : 1;
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDevicePrimaryCtxRelease");
  return fn ? fn(dev) : 1;
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDevice_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDevicePrimaryCtxSetFlags");
  return fn ? fn(dev, flags) : 1;
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDevice_t, unsigned int*, int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDevicePrimaryCtxGetState");
  return fn ? fn(dev, flags, active) : 1;
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDevicePrimaryCtxReset");
  return fn ? fn(dev) : 1;
}

hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDevice_t*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGet");
  return fn ? fn(device, ordinal) : 1;
}

hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceComputeCapability");
  return fn ? fn(major, minor, device) : 1;
}

hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
  ensure_init();
  typedef hipError_t (*pfn)(size_t*, hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceTotalMem");
  return fn ? fn(bytes, device) : 1;
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t*, size_t*, hipDeviceptr_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemGetAddressRange");
  return fn ? fn(pbase, psize, dptr) : 1;
}

hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t*, size_t*, size_t, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemAllocPitch");
  return fn ? fn(dptr, pitch, widthInBytes, height, elementSizeBytes) : 1;
}

hipError_t hipMemAlloc(hipDeviceptr_t* dptr, size_t size) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemAlloc");
  return fn ? fn(dptr, size) : 1;
}

hipError_t hipMemFree(hipDeviceptr_t dptr) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemFree");
  return fn ? fn(dptr) : 1;
}

// Symbol memory operations
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, const void*, size_t, size_t, hipMemcpyKind);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyFromSymbol");
  return fn ? fn(dst, symbol, sizeBytes, offset, kind) : 1;
}

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, const void*, size_t, size_t, hipMemcpyKind, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyFromSymbolAsync");
  return fn ? fn(dst, symbol, sizeBytes, offset, kind, stream) : 1;
}

hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset, hipMemcpyKind kind) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, const void*, size_t, size_t, hipMemcpyKind);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyToSymbol");
  return fn ? fn(symbol, src, sizeBytes, offset, kind) : 1;
}

hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, const void*, size_t, size_t, hipMemcpyKind, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyToSymbolAsync");
  return fn ? fn(symbol, src, sizeBytes, offset, kind, stream) : 1;
}

hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetSymbolAddress");
  return fn ? fn(devPtr, symbol) : 1;
}

hipError_t hipGetSymbolSize(size_t* size, const void* symbol) {
  ensure_init();
  typedef hipError_t (*pfn)(size_t*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetSymbolSize");
  return fn ? fn(size, symbol) : 1;
}

// Additional memory functions
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, const void*, int, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyPeer");
  return fn ? fn(dst, dstDeviceId, src, srcDeviceId, sizeBytes) : 1;
}

hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, const void*, int, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyPeerAsync");
  return fn ? fn(dst, dstDeviceId, src, srcDeviceId, sizeBytes, stream) : 1;
}

// Texture functions
hipError_t hipBindTexture(size_t* offset, const void* tex, const void* devPtr, const void* desc, size_t size) {
  ensure_init();
  typedef hipError_t (*pfn)(size_t*, const void*, const void*, const void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipBindTexture");
  return fn ? fn(offset, tex, devPtr, desc, size) : 1;
}

hipError_t hipUnbindTexture(const void* tex) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipUnbindTexture");
  return fn ? fn(tex) : 1;
}

// Extended launch kernel
hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, dim3, dim3, void**, size_t, hipStream_t, hipEvent_t, hipEvent_t, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExtLaunchKernel");
  return fn ? fn(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags) : 1;
}

// Device set/get cache config
hipError_t hipDeviceSetCacheConfig(int cacheConfig) {
  ensure_init();
  typedef hipError_t (*pfn)(int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceSetCacheConfig");
  return fn ? fn(cacheConfig) : 1;
}

hipError_t hipDeviceGetCacheConfig(int* cacheConfig) {
  ensure_init();
  typedef hipError_t (*pfn)(int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetCacheConfig");
  return fn ? fn(cacheConfig) : 1;
}

hipError_t hipDeviceSetSharedMemConfig(int config) {
  ensure_init();
  typedef hipError_t (*pfn)(int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceSetSharedMemConfig");
  return fn ? fn(config) : 1;
}

hipError_t hipDeviceGetSharedMemConfig(int* pConfig) {
  ensure_init();
  typedef hipError_t (*pfn)(int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetSharedMemConfig");
  return fn ? fn(pConfig) : 1;
}

hipError_t hipDeviceGetLimit(size_t* pValue, int limit) {
  ensure_init();
  typedef hipError_t (*pfn)(size_t*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetLimit");
  return fn ? fn(pValue, limit) : 1;
}

hipError_t hipDeviceSetLimit(int limit, size_t value) {
  ensure_init();
  typedef hipError_t (*pfn)(int, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceSetLimit");
  return fn ? fn(limit, value) : 1;
}

hipError_t hipDeviceGetUuid(void* uuid, hipDevice_t device) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetUuid");
  return fn ? fn(uuid, device) : 1;
}

hipError_t hipSetDeviceFlags(unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipSetDeviceFlags");
  return fn ? fn(flags) : 1;
}

hipError_t hipGetDeviceFlags(unsigned int* flags) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetDeviceFlags");
  return fn ? fn(flags) : 1;
}

hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, const hipDeviceProp_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipChooseDevice");
  return fn ? fn(device, prop) : 1;
}

// External memory
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out, const void* memHandleDesc) {
  ensure_init();
  typedef hipError_t (*pfn)(hipExternalMemory_t*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipImportExternalMemory");
  return fn ? fn(extMem_out, memHandleDesc) : 1;
}

hipError_t hipExternalMemoryGetMappedBuffer(void** devPtr, hipExternalMemory_t extMem, const void* bufferDesc) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, hipExternalMemory_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExternalMemoryGetMappedBuffer");
  return fn ? fn(devPtr, extMem, bufferDesc) : 1;
}

hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
  ensure_init();
  typedef hipError_t (*pfn)(hipExternalMemory_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDestroyExternalMemory");
  return fn ? fn(extMem) : 1;
}

hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out, const void* semHandleDesc) {
  ensure_init();
  typedef hipError_t (*pfn)(hipExternalSemaphore_t*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipImportExternalSemaphore");
  return fn ? fn(extSem_out, semHandleDesc) : 1;
}

hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const void* paramsArray, unsigned int numExtSems, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(const hipExternalSemaphore_t*, const void*, unsigned int, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipSignalExternalSemaphoresAsync");
  return fn ? fn(extSemArray, paramsArray, numExtSems, stream) : 1;
}

hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const void* paramsArray, unsigned int numExtSems, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(const hipExternalSemaphore_t*, const void*, unsigned int, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipWaitExternalSemaphoresAsync");
  return fn ? fn(extSemArray, paramsArray, numExtSems, stream) : 1;
}

hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
  ensure_init();
  typedef hipError_t (*pfn)(hipExternalSemaphore_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDestroyExternalSemaphore");
  return fn ? fn(extSem) : 1;
}

// Extended error handling
hipError_t hipExtGetLastError(void) {
  ensure_init();
  typedef hipError_t (*pfn)(void);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExtGetLastError");
  return fn ? fn() : 1;
}

const char* hipDrvGetErrorString(hipError_t hipError, const char** pStr) {
  ensure_init();
  typedef const char* (*pfn)(hipError_t, const char**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDrvGetErrorString");
  return fn ? fn(hipError, pStr) : "unknown";
}

const char* hipDrvGetErrorName(hipError_t hipError, const char** pStr) {
  ensure_init();
  typedef const char* (*pfn)(hipError_t, const char**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDrvGetErrorName");
  return fn ? fn(hipError, pStr) : "unknown";
}

// Thread exchange
hipError_t hipThreadExchangeStreamCaptureMode(int* mode) {
  ensure_init();
  typedef hipError_t (*pfn)(int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipThreadExchangeStreamCaptureMode");
  return fn ? fn(mode) : 1;
}

// Memory pool access
hipError_t hipMemPoolSetAccess(hipMemPool_t memPool, const void* descList, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t, const void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolSetAccess");
  return fn ? fn(memPool, descList, count) : 1;
}

hipError_t hipMemPoolGetAccess(int* flags, hipMemPool_t memPool, void* location) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, hipMemPool_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolGetAccess");
  return fn ? fn(flags, memPool, location) : 1;
}

hipError_t hipMemPoolExportToShareableHandle(void* shareableHandle, hipMemPool_t memPool, int handleType, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipMemPool_t, int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolExportToShareableHandle");
  return fn ? fn(shareableHandle, memPool, handleType, flags) : 1;
}

hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* memPool, void* shareableHandle, int handleType, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t*, void*, int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolImportFromShareableHandle");
  return fn ? fn(memPool, shareableHandle, handleType, flags) : 1;
}

hipError_t hipMemPoolExportPointer(void* exportData, void* dev_ptr) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolExportPointer");
  return fn ? fn(exportData, dev_ptr) : 1;
}

hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t memPool, void* exportData) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, hipMemPool_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPoolImportPointer");
  return fn ? fn(dev_ptr, memPool, exportData) : 1;
}

// Graph extended functions
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph, unsigned long long flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphExec_t*, hipGraph_t, unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphInstantiateWithFlags");
  return fn ? fn(pGraphExec, graph, flags) : 1;
}

hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphExec_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphUpload");
  return fn ? fn(graphExec, stream) : 1;
}

hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const void* pCopyParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddMemcpyNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, pCopyParams) : 1;
}

hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const void* pMemsetParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddMemsetNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams) : 1;
}

hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddKernelNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, pNodeParams) : 1;
}

hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, hipGraphNode_t*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphGetNodes");
  return fn ? fn(graph, nodes, numNodes) : 1;
}

hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes, size_t* pNumRootNodes) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, hipGraphNode_t*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphGetRootNodes");
  return fn ? fn(graph, pRootNodes, pNumRootNodes) : 1;
}

hipError_t hipGraphNodeGetType(hipGraphNode_t node, int* pType) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphNodeGetType");
  return fn ? fn(node, pType) : 1;
}

hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, hipGraph_t* graph_out, const hipGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipStreamCaptureStatus*, unsigned long long*, hipGraph_t*, const hipGraphNode_t**, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamGetCaptureInfo_v2");
  return fn ? fn(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out) : 1;
}

// Cooperative kernel launch
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDim, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, dim3, dim3, void**, unsigned int, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipLaunchCooperativeKernel");
  return fn ? fn(f, gridDim, blockDim, kernelParams, sharedMemBytes, stream) : 1;
}

hipError_t hipOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize, const void* f, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, const void*, size_t, int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipOccupancyMaxPotentialBlockSizeWithFlags");
  return fn ? fn(gridSize, blockSize, f, dynamicSMemSize, blockSizeLimit, flags) : 1;
}

// Array functions
hipError_t hipMallocArray(hipArray_t* array, const void* desc, size_t width, size_t height, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t*, const void*, size_t, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMallocArray");
  return fn ? fn(array, desc, width, height, flags) : 1;
}

hipError_t hipFreeArray(hipArray_t array) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFreeArray");
  return fn ? fn(array) : 1;
}

hipError_t hipMalloc3DArray(hipArray_t* array, const void* desc, void* extent, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t*, const void*, void*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMalloc3DArray");
  return fn ? fn(array, desc, extent, flags) : 1;
}

hipError_t hipArrayGetInfo(void* desc, void* extent, unsigned int* flags, hipArray_t array) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, void*, unsigned int*, hipArray_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipArrayGetInfo");
  return fn ? fn(desc, extent, flags, array) : 1;
}

hipError_t hipMemcpy3D(const void* p) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpy3D");
  return fn ? fn(p) : 1;
}

hipError_t hipMemcpy3DAsync(const void* p, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpy3DAsync");
  return fn ? fn(p, stream) : 1;
}

hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, hipMemcpyKind kind) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t, size_t, size_t, const void*, size_t, hipMemcpyKind);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyToArray");
  return fn ? fn(dst, wOffset, hOffset, src, count, kind) : 1;
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_t src, size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipArray_t, size_t, size_t, size_t, hipMemcpyKind);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyFromArray");
  return fn ? fn(dst, src, wOffset, hOffset, count, kind) : 1;
}

hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t, size_t, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpy2DToArray");
  return fn ? fn(dst, wOffset, hOffset, src, spitch, width, height, kind) : 1;
}

hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, hipArray_t, size_t, size_t, size_t, size_t, hipMemcpyKind);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpy2DFromArray");
  return fn ? fn(dst, dpitch, src, wOffset, hOffset, width, height, kind) : 1;
}

hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray, size_t srcOffset, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipArray_t, size_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyAtoH");
  return fn ? fn(dst, srcArray, srcOffset, count) : 1;
}

hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t, size_t, const void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemcpyHtoA");
  return fn ? fn(dstArray, dstOffset, srcHost, count) : 1;
}

// Mipmapped array
hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray, const void* desc, void* extent, unsigned int numLevels, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMipmappedArray_t*, const void*, void*, unsigned int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMallocMipmappedArray");
  return fn ? fn(mipmappedArray, desc, extent, numLevels, flags) : 1;
}

hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMipmappedArray_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFreeMipmappedArray");
  return fn ? fn(mipmappedArray) : 1;
}

hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray, hipMipmappedArray_t mipmappedArray, unsigned int level) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t*, hipMipmappedArray_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetMipmappedArrayLevel");
  return fn ? fn(levelArray, mipmappedArray, level) : 1;
}

// Extended malloc
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExtMallocWithFlags");
  return fn ? fn(ptr, sizeBytes, flags) : 1;
}

// HIP RT functions used by PyTorch
hipError_t hipRuntimeGetVersion_r(int* runtimeVersion) {
  ensure_init();
  typedef hipError_t (*pfn)(int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRuntimeGetVersion_r");
  return fn ? fn(runtimeVersion) : 1;
}

// Surface reference
hipError_t hipCreateSurfaceObject(void* pSurfObject, const void* pResDesc) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCreateSurfaceObject");
  return fn ? fn(pSurfObject, pResDesc) : 1;
}

hipError_t hipDestroySurfaceObject(unsigned long long surfaceObject) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDestroySurfaceObject");
  return fn ? fn(surfaceObject) : 1;
}

// Texture object
hipError_t hipCreateTextureObject(void* pTexObject, const void* pResDesc, const void* pTexDesc, const void* pResViewDesc) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, const void*, const void*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipCreateTextureObject");
  return fn ? fn(pTexObject, pResDesc, pTexDesc, pResViewDesc) : 1;
}

hipError_t hipDestroyTextureObject(unsigned long long textureObject) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDestroyTextureObject");
  return fn ? fn(textureObject) : 1;
}

hipError_t hipGetTextureObjectResourceDesc(void* pResDesc, unsigned long long textureObject) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetTextureObjectResourceDesc");
  return fn ? fn(pResDesc, textureObject) : 1;
}

hipError_t hipGetTextureObjectTextureDesc(void* pTexDesc, unsigned long long textureObject) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetTextureObjectTextureDesc");
  return fn ? fn(pTexDesc, textureObject) : 1;
}

hipError_t hipGetChannelDesc(void* desc, hipArray_t array) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipArray_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetChannelDesc");
  return fn ? fn(desc, array) : 1;
}

hipError_t hipTexRefSetArray(void* texRef, hipArray_t array, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipArray_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefSetArray");
  return fn ? fn(texRef, array, flags) : 1;
}

hipError_t hipTexRefSetAddress(size_t* offset, void* texRef, hipDeviceptr_t devPtr, size_t size) {
  ensure_init();
  typedef hipError_t (*pfn)(size_t*, void*, hipDeviceptr_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefSetAddress");
  return fn ? fn(offset, texRef, devPtr, size) : 1;
}

hipError_t hipTexRefSetAddress2D(void* texRef, const void* desc, hipDeviceptr_t devPtr, size_t pitch) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, const void*, hipDeviceptr_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefSetAddress2D");
  return fn ? fn(texRef, desc, devPtr, pitch) : 1;
}

hipError_t hipTexRefSetFormat(void* texRef, int format, int numComponents) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefSetFormat");
  return fn ? fn(texRef, format, numComponents) : 1;
}

hipError_t hipTexRefSetFlags(void* texRef, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefSetFlags");
  return fn ? fn(texRef, flags) : 1;
}

hipError_t hipTexRefSetFilterMode(void* texRef, int fm) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefSetFilterMode");
  return fn ? fn(texRef, fm) : 1;
}

hipError_t hipTexRefSetAddressMode(void* texRef, int dim, int am) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefSetAddressMode");
  return fn ? fn(texRef, dim, am) : 1;
}

hipError_t hipTexRefGetAddress(hipDeviceptr_t* dptr, const void* texRef) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipTexRefGetAddress");
  return fn ? fn(dptr, texRef) : 1;
}

// Module symbol operations
hipError_t hipModuleGetTexRef(void** texRef, hipModule_t hmod, const char* name) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, hipModule_t, const char*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleGetTexRef");
  return fn ? fn(texRef, hmod, name) : 1;
}

// Additional register functions
void __hipRegisterSurface(void** fatCubinHandle, char* hostVar, char* deviceAddress, const char* deviceName, int dim, int ext) {
  ensure_init();
  typedef void (*pfn)(void**, char*, char*, const char*, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "__hipRegisterSurface");
  if (fn) fn(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
}

void __hipRegisterTexture(void** fatCubinHandle, char* hostVar, char* deviceAddress, const char* deviceName, int dim, int norm, int ext) {
  ensure_init();
  typedef void (*pfn)(void**, char*, char*, const char*, int, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "__hipRegisterTexture");
  if (fn) fn(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
}

void __hipRegisterManagedVar(void* hipModule, void** pointer, void* init_value, const char* name, size_t size, unsigned align) {
  ensure_init();
  typedef void (*pfn)(void*, void**, void*, const char*, size_t, unsigned);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "__hipRegisterManagedVar");
  if (fn) fn(hipModule, pointer, init_value, name, size, align);
}

// Version 2 stream capture functions
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipGraphNode_t*, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamUpdateCaptureDependencies");
  return fn ? fn(stream, dependencies, numDependencies, flags) : 1;
}

hipError_t hipUserObjectCreate(void** object_out, void* ptr, void (*destroy)(void* ptr), unsigned int initialRefcount, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, void*, void (*)(void*), unsigned int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipUserObjectCreate");
  return fn ? fn(object_out, ptr, destroy, initialRefcount, flags) : 1;
}

hipError_t hipUserObjectRelease(void* object, unsigned int count) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipUserObjectRelease");
  return fn ? fn(object, count) : 1;
}

hipError_t hipUserObjectRetain(void* object, unsigned int count) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipUserObjectRetain");
  return fn ? fn(object, count) : 1;
}

hipError_t hipGraphRetainUserObject(hipGraph_t graph, void* object, unsigned int count, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, void*, unsigned int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphRetainUserObject");
  return fn ? fn(graph, object, count, flags) : 1;
}

hipError_t hipGraphReleaseUserObject(hipGraph_t graph, void* object, unsigned int count) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, void*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphReleaseUserObject");
  return fn ? fn(graph, object, count) : 1;
}

// More graph node functions
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddHostNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, pNodeParams) : 1;
}

hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, hipEvent_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddEventRecordNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, event) : 1;
}

hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, hipEvent_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddEventWaitNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, event) : 1;
}

hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipGraph_t childGraph) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, hipGraph_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddChildGraphNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, childGraph) : 1;
}

hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddEmptyNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies) : 1;
}

hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, const hipGraphNode_t*, const hipGraphNode_t*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddDependencies");
  return fn ? fn(graph, from, to, numDependencies) : 1;
}

hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, const hipGraphNode_t*, const hipGraphNode_t*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphRemoveDependencies");
  return fn ? fn(graph, from, to, numDependencies) : 1;
}

hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphDestroyNode");
  return fn ? fn(node) : 1;
}

hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t*, hipGraph_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphClone");
  return fn ? fn(pGraphClone, originalGraph) : 1;
}

hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraphNode_t, hipGraph_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphNodeFindInClone");
  return fn ? fn(pNode, originalNode, clonedGraph) : 1;
}

hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies, size_t* pNumDependencies) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, hipGraphNode_t*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphNodeGetDependencies");
  return fn ? fn(node, pDependencies, pNumDependencies) : 1;
}

hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, hipGraphNode_t*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphNodeGetDependentNodes");
  return fn ? fn(node, pDependentNodes, pNumDependentNodes) : 1;
}

hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to, size_t* numEdges) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, hipGraphNode_t*, hipGraphNode_t*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphGetEdges");
  return fn ? fn(graph, from, to, numEdges) : 1;
}

hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t* hErrorNode_out, int* updateResult_out) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphExec_t, hipGraph_t, hipGraphNode_t*, int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphExecUpdate");
  return fn ? fn(hGraphExec, hGraph, hErrorNode_out, updateResult_out) : 1;
}

hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphKernelNodeGetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphKernelNodeSetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphExec_t, hipGraphNode_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphExecKernelNodeSetParams");
  return fn ? fn(hGraphExec, node, pNodeParams) : 1;
}

hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphMemcpyNodeGetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphMemcpyNodeSetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphMemsetNodeGetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphMemsetNodeSetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphHostNodeGetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphHostNodeSetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, hipGraph_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphChildGraphNodeGetGraph");
  return fn ? fn(node, pGraph) : 1;
}

hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, hipEvent_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphEventRecordNodeGetEvent");
  return fn ? fn(node, event_out) : 1;
}

hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, hipEvent_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphEventRecordNodeSetEvent");
  return fn ? fn(node, event) : 1;
}

hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, hipEvent_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphEventWaitNodeGetEvent");
  return fn ? fn(node, event_out) : 1;
}

hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, hipEvent_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphEventWaitNodeSetEvent");
  return fn ? fn(node, event) : 1;
}

hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraph_t, const char*, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphDebugDotPrint");
  return fn ? fn(graph, path, flags) : 1;
}

hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* dptr) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddMemFreeNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, dptr) : 1;
}

hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*, size_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphAddMemAllocNode");
  return fn ? fn(pGraphNode, graph, pDependencies, numDependencies, pNodeParams) : 1;
}

hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, void* pNodeParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphMemAllocNodeGetParams");
  return fn ? fn(node, pNodeParams) : 1;
}

hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr) {
  ensure_init();
  typedef hipError_t (*pfn)(hipGraphNode_t, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphMemFreeNodeGetParams");
  return fn ? fn(node, dev_ptr) : 1;
}

hipError_t hipDeviceGraphMemTrim(int device) {
  ensure_init();
  typedef hipError_t (*pfn)(int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGraphMemTrim");
  return fn ? fn(device) : 1;
}

hipError_t hipDeviceGetGraphMemAttribute(int device, int attr, void* value) {
  ensure_init();
  typedef hipError_t (*pfn)(int, int, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetGraphMemAttribute");
  return fn ? fn(device, attr, value) : 1;
}

hipError_t hipDeviceSetGraphMemAttribute(int device, int attr, void* value) {
  ensure_init();
  typedef hipError_t (*pfn)(int, int, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceSetGraphMemAttribute");
  return fn ? fn(device, attr, value) : 1;
}

// Additional stream functions
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipDevice_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamGetDevice");
  return fn ? fn(stream, device) : 1;
}

hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, void*, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamAttachMemAsync");
  return fn ? fn(stream, dev_ptr, length, flags) : 1;
}

// Kernel name reference
const char* hipKernelNameRef(const void* hostFunction) {
  ensure_init();
  typedef const char* (*pfn)(const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipKernelNameRef");
  return fn ? fn(hostFunction) : "unknown";
}

const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream) {
  ensure_init();
  typedef const char* (*pfn)(const void*, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipKernelNameRefByPtr");
  return fn ? fn(hostFunction, stream) : "unknown";
}

// Launch kernel by name
hipError_t hipLaunchKernelGGL(const void* func, dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream, void** kernelParams) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, dim3, dim3, size_t, hipStream_t, void**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipLaunchKernelGGL");
  return fn ? fn(func, gridDim, blockDim, sharedMem, stream, kernelParams) : 1;
}

// Get module occupancy functions
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, hipFunction_t, size_t, int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleOccupancyMaxPotentialBlockSizeWithFlags");
  return fn ? fn(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags) : 1;
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, hipFunction_t, int, size_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  return fn ? fn(numBlocks, f, blockSize, dynSharedMemPerBlk, flags) : 1;
}

// Additional profiling
hipError_t hipProfilerStart(void) {
  ensure_init();
  typedef hipError_t (*pfn)(void);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipProfilerStart");
  return fn ? fn() : 1;
}

hipError_t hipProfilerStop(void) {
  ensure_init();
  typedef hipError_t (*pfn)(void);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipProfilerStop");
  return fn ? fn() : 1;
}

// API version functions
hipError_t hipApiVersion(unsigned int* apiVersion) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipApiVersion");
  return fn ? fn(apiVersion) : 1;
}

// Memory pool memory allocator
hipError_t hipMallocHost(void** ptr, size_t size) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMallocHost");
  return fn ? fn(ptr, size) : 1;
}

hipError_t hipFreeHost(void* ptr) {
  ensure_init();
  typedef hipError_t (*pfn)(void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipFreeHost");
  return fn ? fn(ptr) : 1;
}

// Module launch configurations
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(dim3, dim3, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipConfigureCall");
  return fn ? fn(gridDim, blockDim, sharedMem, stream) : 1;
}

hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, size_t, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipSetupArgument");
  return fn ? fn(arg, size, offset) : 1;
}

hipError_t hipLaunchByPtr(const void* func) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipLaunchByPtr");
  return fn ? fn(func) : 1;
}

// Cooperative kernel launch
hipError_t hipLaunchCooperativeKernelMultiDevice(void* launchParamsList, int numDevices, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipLaunchCooperativeKernelMultiDevice");
  return fn ? fn(launchParamsList, numDevices, flags) : 1;
}

hipError_t hipExtLaunchMultiKernelMultiDevice(void* launchParamsList, int numDevices, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExtLaunchMultiKernelMultiDevice");
  return fn ? fn(launchParamsList, numDevices, flags) : 1;
}

// Memory access info
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const void* prop, int option) {
  ensure_init();
  typedef hipError_t (*pfn)(size_t*, const void*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemGetAllocationGranularity");
  return fn ? fn(granularity, prop, option) : 1;
}

hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t, size_t, void*, unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemAddressReserve");
  return fn ? fn(ptr, size, alignment, addr, flags) : 1;
}

hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemAddressFree");
  return fn ? fn(devPtr, size) : 1;
}

hipError_t hipMemCreate(void* handle, size_t size, const void* prop, unsigned long long flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, const void*, unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemCreate");
  return fn ? fn(handle, size, prop, flags) : 1;
}

hipError_t hipMemRelease(void* handle) {
  ensure_init();
  typedef hipError_t (*pfn)(void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemRelease");
  return fn ? fn(handle) : 1;
}

hipError_t hipMemMap(void* ptr, size_t size, size_t offset, void* handle, unsigned long long flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, size_t, void*, unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemMap");
  return fn ? fn(ptr, size, offset, handle, flags) : 1;
}

hipError_t hipMemUnmap(void* ptr, size_t size) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemUnmap");
  return fn ? fn(ptr, size) : 1;
}

hipError_t hipMemSetAccess(void* ptr, size_t size, const void* desc, size_t count) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t, const void*, size_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemSetAccess");
  return fn ? fn(ptr, size, desc, count) : 1;
}

hipError_t hipMemGetAccess(unsigned long long* flags, const void* location, void* ptr) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned long long*, const void*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemGetAccess");
  return fn ? fn(flags, location, ptr) : 1;
}

hipError_t hipMemExportToShareableHandle(void* shareableHandle, void* handle, int handleType, unsigned long long flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, void*, int, unsigned long long);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemExportToShareableHandle");
  return fn ? fn(shareableHandle, handle, handleType, flags) : 1;
}

hipError_t hipMemImportFromShareableHandle(void* handle, void* osHandle, int shHandleType) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, void*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemImportFromShareableHandle");
  return fn ? fn(handle, osHandle, shHandleType) : 1;
}

hipError_t hipMemGetAllocationPropertiesFromHandle(void* prop, void* handle) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemGetAllocationPropertiesFromHandle");
  return fn ? fn(prop, handle) : 1;
}

hipError_t hipMemRetainAllocationHandle(void* handle, void* addr) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemRetainAllocationHandle");
  return fn ? fn(handle, addr) : 1;
}

// Memory pointer info
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemPtrGetInfo");
  return fn ? fn(ptr, size) : 1;
}

// Extended API
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t*, uint32_t, const uint32_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExtStreamCreateWithCUMask");
  return fn ? fn(stream, cuMaskSize, cuMask) : 1;
}

hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, uint32_t, uint32_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExtStreamGetCUMask");
  return fn ? fn(stream, cuMaskSize, cuMask) : 1;
}

hipError_t hipHccModuleLaunchKernel(hipFunction_t f, unsigned int globalWorkSizeX, unsigned int globalWorkSizeY, unsigned int globalWorkSizeZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, size_t sharedMemBytes, hipStream_t hStream, void** kernelParams, void** extra, hipEvent_t startEvent, hipEvent_t stopEvent) {
  ensure_init();
  typedef hipError_t (*pfn)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, size_t, hipStream_t, void**, void**, hipEvent_t, hipEvent_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipHccModuleLaunchKernel");
  return fn ? fn(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent) : 1;
}

// Device UUID
hipError_t hipDeviceGetUuidString(char* uuid, hipDevice_t device) {
  ensure_init();
  typedef hipError_t (*pfn)(char*, hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetUuidString");
  return fn ? fn(uuid, device) : 1;
}

// Peer memory access
hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, hipStream_t stream);

// Device get arch
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t* prop, int deviceId);

// Additional memory functions used by PyTorch
hipError_t hipMalloc3D(void* pitchedDevPtr, void* extent) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMalloc3D");
  return fn ? fn(pitchedDevPtr, extent) : 1;
}

hipError_t hipMemset3D(void* pitchedDevPtr, int value, void* extent) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemset3D");
  return fn ? fn(pitchedDevPtr, value, extent) : 1;
}

hipError_t hipMemset3DAsync(void* pitchedDevPtr, int value, void* extent, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, void*, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipMemset3DAsync");
  return fn ? fn(pitchedDevPtr, value, extent, stream) : 1;
}

hipError_t hipGetArrayDescriptor(void* pArrayDescriptor, hipArray_t array) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipArray_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetArrayDescriptor");
  return fn ? fn(pArrayDescriptor, array) : 1;
}

hipError_t hipArrayCreate(hipArray_t* pHandle, const void* pAllocateArray) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipArrayCreate");
  return fn ? fn(pHandle, pAllocateArray) : 1;
}

hipError_t hipArray3DCreate(hipArray_t* pHandle, const void* pAllocateArray) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t*, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipArray3DCreate");
  return fn ? fn(pHandle, pAllocateArray) : 1;
}

hipError_t hipArrayGetDescriptor(void* pArrayDescriptor, hipArray_t array) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipArray_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipArrayGetDescriptor");
  return fn ? fn(pArrayDescriptor, array) : 1;
}

hipError_t hipArray3DGetDescriptor(void* pArrayDescriptor, hipArray_t array) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, hipArray_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipArray3DGetDescriptor");
  return fn ? fn(pArrayDescriptor, array) : 1;
}

// External resource interop
hipError_t hipExternalMemoryGetMappedMipmappedArray(hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem, const void* mipmapDesc) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMipmappedArray_t*, hipExternalMemory_t, const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipExternalMemoryGetMappedMipmappedArray");
  return fn ? fn(mipmap, extMem, mipmapDesc) : 1;
}

// Device get architecture
hipError_t hipDeviceGetDefaultMemPoolR0600(hipMemPool_t* memPool, int device) {
  ensure_init();
  typedef hipError_t (*pfn)(hipMemPool_t*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetDefaultMemPoolR0600");
  return fn ? fn(memPool, device) : 1;
}

// Module cooperative kernel launch
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams) {
  ensure_init();
  typedef hipError_t (*pfn)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipStream_t, void**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleLaunchCooperativeKernel");
  return fn ? fn(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams) : 1;
}

hipError_t hipModuleLaunchCooperativeKernelMultiDevice(void* launchParamsList, unsigned int numDevices, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, unsigned int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleLaunchCooperativeKernelMultiDevice");
  return fn ? fn(launchParamsList, numDevices, flags) : 1;
}

hipError_t hipOccupancyMaxPotentialBlockSizeVariableSMem(int* minGridSize, int* blockSize, const void* func, void* blockSizeToDynamicSMemSize, int blockSizeLimit) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, const void*, void*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipOccupancyMaxPotentialBlockSizeVariableSMem");
  return fn ? fn(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit) : 1;
}

hipError_t hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int* minGridSize, int* blockSize, const void* func, void* blockSizeToDynamicSMemSize, int blockSizeLimit, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, const void*, void*, int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags");
  return fn ? fn(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, flags) : 1;
}

// Query functions
hipError_t hipDeviceGetP2PAttribute(int* value, int attr, int srcDevice, int dstDevice) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int, int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetP2PAttribute");
  return fn ? fn(value, attr, srcDevice, dstDevice) : 1;
}

// Query device arch attribute
hipError_t hipDeviceGetArchConfig(int* major, int* minor, hipDevice_t device) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, int*, hipDevice_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetArchConfig");
  return fn ? fn(major, minor, device) : 1;
}

// GetLastError reset
hipError_t hipDrvMemcpy2DUnaligned(const void* pCopy) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDrvMemcpy2DUnaligned");
  return fn ? fn(pCopy) : 1;
}

hipError_t hipDrvMemcpy3D(const void* p) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDrvMemcpy3D");
  return fn ? fn(p) : 1;
}

hipError_t hipDrvMemcpy3DAsync(const void* p, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDrvMemcpy3DAsync");
  return fn ? fn(p, stream) : 1;
}

// Device/Module functions
hipError_t hipModuleLoadGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod, const char* name) {
  ensure_init();
  typedef hipError_t (*pfn)(hipDeviceptr_t*, size_t*, hipModule_t, const char*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipModuleLoadGlobal");
  return fn ? fn(dptr, bytes, hmod, name) : 1;
}

// Device attribute extended
hipError_t hipDeviceGetAttributeByName(int* value, const char* attr_name, int device) {
  ensure_init();
  typedef hipError_t (*pfn)(int*, const char*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDeviceGetAttributeByName");
  return fn ? fn(value, attr_name, device) : 1;
}

// Stream write/wait functions
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, void*, uint32_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamWriteValue32");
  return fn ? fn(stream, ptr, value, flags) : 1;
}

hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, void*, uint64_t, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamWriteValue64");
  return fn ? fn(stream, ptr, value, flags) : 1;
}

hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, void*, uint32_t, unsigned int, uint32_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamWaitValue32");
  return fn ? fn(stream, ptr, value, flags, mask) : 1;
}

hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags, uint64_t mask) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, void*, uint64_t, unsigned int, uint64_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamWaitValue64");
  return fn ? fn(stream, ptr, value, flags, mask) : 1;
}

// Copy params
hipError_t hipDrvMemcpy2D(const void* pCopy) {
  ensure_init();
  typedef hipError_t (*pfn)(const void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipDrvMemcpy2D");
  return fn ? fn(pCopy) : 1;
}

// Kernel compilation
hipError_t hipRtcCompileProgram(void* prog, int numOptions, const char** options) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int, const char**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRtcCompileProgram");
  return fn ? fn(prog, numOptions, options) : 1;
}

hipError_t hipRtcCreateProgram(void** prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, const char*, const char*, int, const char**, const char**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRtcCreateProgram");
  return fn ? fn(prog, src, name, numHeaders, headers, includeNames) : 1;
}

hipError_t hipRtcDestroyProgram(void** prog) {
  ensure_init();
  typedef hipError_t (*pfn)(void**);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRtcDestroyProgram");
  return fn ? fn(prog) : 1;
}

hipError_t hipRtcGetCode(void* prog, char* code) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, char*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRtcGetCode");
  return fn ? fn(prog, code) : 1;
}

hipError_t hipRtcGetCodeSize(void* prog, size_t* codeSizeRet) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRtcGetCodeSize");
  return fn ? fn(prog, codeSizeRet) : 1;
}

hipError_t hipRtcGetProgramLog(void* prog, char* log) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, char*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRtcGetProgramLog");
  return fn ? fn(prog, log) : 1;
}

hipError_t hipRtcGetProgramLogSize(void* prog, size_t* logSizeRet) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, size_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipRtcGetProgramLogSize");
  return fn ? fn(prog, logSizeRet) : 1;
}

// Virtual memory management
hipError_t hipGetDeviceProperties_v2(void* prop, int device) {
  ensure_init();
  typedef hipError_t (*pfn)(void*, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetDeviceProperties_v2");
  return fn ? fn(prop, device) : 1;
}

// OpenGL interop
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices, unsigned int hipDeviceCount, int deviceList) {
  ensure_init();
  typedef hipError_t (*pfn)(unsigned int*, int*, unsigned int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGLGetDevices");
  return fn ? fn(pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList) : 1;
}

hipError_t hipGraphicsGLRegisterBuffer(void** resource, unsigned int buffer, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, unsigned int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphicsGLRegisterBuffer");
  return fn ? fn(resource, buffer, flags) : 1;
}

hipError_t hipGraphicsGLRegisterImage(void** resource, unsigned int image, unsigned int target, unsigned int flags) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, unsigned int, unsigned int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphicsGLRegisterImage");
  return fn ? fn(resource, image, target, flags) : 1;
}

hipError_t hipGraphicsMapResources(int count, void** resources, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(int, void**, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphicsMapResources");
  return fn ? fn(count, resources, stream) : 1;
}

hipError_t hipGraphicsUnmapResources(int count, void** resources, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(int, void**, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphicsUnmapResources");
  return fn ? fn(count, resources, stream) : 1;
}

hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, void* resource) {
  ensure_init();
  typedef hipError_t (*pfn)(void**, size_t*, void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphicsResourceGetMappedPointer");
  return fn ? fn(devPtr, size, resource) : 1;
}

hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, void* resource, unsigned int arrayIndex, unsigned int mipLevel) {
  ensure_init();
  typedef hipError_t (*pfn)(hipArray_t*, void*, unsigned int, unsigned int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphicsSubResourceGetMappedArray");
  return fn ? fn(array, resource, arrayIndex, mipLevel) : 1;
}

hipError_t hipGraphicsUnregisterResource(void* resource) {
  ensure_init();
  typedef hipError_t (*pfn)(void*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGraphicsUnregisterResource");
  return fn ? fn(resource) : 1;
}

// Stream Management
FWD1(hipError_t, hipStreamCreate, hipStream_t*, stream)
FWD2(hipError_t, hipStreamCreateWithFlags, hipStream_t*, stream, unsigned int, flags)
FWD1(hipError_t, hipStreamDestroy, hipStream_t, stream)
FWD1(hipError_t, hipStreamSynchronize, hipStream_t, stream)
FWD1(hipError_t, hipStreamQuery, hipStream_t, stream)
FWD3(hipError_t, hipStreamWaitEvent, hipStream_t, stream, hipEvent_t, event, unsigned int, flags)

hipError_t hipGetStreamDeviceId(hipStream_t stream, int* deviceId) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipGetStreamDeviceId");
  return fn ? fn(stream, deviceId) : 1;
}

hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, hipStreamCaptureStatus*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamIsCapturing");
  return fn ? fn(stream, pCaptureStatus) : 1;
}

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t*, unsigned int, int);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamCreateWithPriority");
  return fn ? fn(stream, flags, priority) : 1;
}

hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamGetPriority");
  return fn ? fn(stream, priority) : 1;
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  ensure_init();
  typedef hipError_t (*pfn)(hipStream_t, unsigned int*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "hipStreamGetFlags");
  return fn ? fn(stream, flags) : 1;
}

// Event Management
FWD1(hipError_t, hipEventCreate, hipEvent_t*, event)
FWD2(hipError_t, hipEventCreateWithFlags, hipEvent_t*, event, unsigned int, flags)
FWD1(hipError_t, hipEventDestroy, hipEvent_t, event)
FWD2(hipError_t, hipEventRecord, hipEvent_t, event, hipStream_t, stream)
FWD1(hipError_t, hipEventSynchronize, hipEvent_t, event)
FWD1(hipError_t, hipEventQuery, hipEvent_t, event)
FWD3(hipError_t, hipEventElapsedTime, float*, ms, hipEvent_t, start, hipEvent_t, stop)

// Module Management
FWD2(hipError_t, hipModuleLoad, hipModule_t*, module, const char*, fname)
FWD2(hipError_t, hipModuleLoadData, hipModule_t*, module, const void*, image)
FWD1(hipError_t, hipModuleUnload, hipModule_t, module)
FWD3(hipError_t, hipModuleGetFunction, hipFunction_t*, function, hipModule_t, module, const char*, kname)
FWD4(hipError_t, hipModuleGetGlobal, hipDeviceptr_t*, dptr, size_t*, bytes, hipModule_t, hmod, const char*, name)

// Error Handling
FWD0(hipError_t, hipGetLastError)
FWD0(hipError_t, hipPeekAtLastError)

const char* hipGetErrorString(hipError_t hipError) {
  ensure_init();
  return g_active_table->hipGetErrorString ? g_active_table->hipGetErrorString(hipError) : "unknown";
}

const char* hipGetErrorName(hipError_t hipError) {
  ensure_init();
  return g_active_table->hipGetErrorName ? g_active_table->hipGetErrorName(hipError) : "unknown";
}

// Module Launch - manual because of many args
hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                  unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                  unsigned int sharedMemBytes, hipStream_t stream,
                                  void** kernelParams, void** extra) {
  ensure_init();
  return g_active_table->hipModuleLaunchKernel ?
    g_active_table->hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                          blockDimX, blockDimY, blockDimZ,
                                          sharedMemBytes, stream, kernelParams, extra) : 1;
}

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                           void** args, size_t sharedMemBytes, hipStream_t stream) {
  ensure_init();
  return g_active_table->hipLaunchKernel ?
    g_active_table->hipLaunchKernel(function_address, numBlocks, dimBlocks,
                                    args, sharedMemBytes, stream) : 1;
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f,
                                     unsigned int globalWorkSizeX, unsigned int globalWorkSizeY, unsigned int globalWorkSizeZ,
                                     unsigned int localWorkSizeX, unsigned int localWorkSizeY, unsigned int localWorkSizeZ,
                                     size_t sharedMemBytes, hipStream_t stream,
                                     void** kernelParams, void** extra,
                                     hipEvent_t startEvent, hipEvent_t stopEvent, unsigned int flags) {
  ensure_init();
  return g_active_table->hipExtModuleLaunchKernel ?
    g_active_table->hipExtModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                                              localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                              sharedMemBytes, stream, kernelParams, extra,
                                              startEvent, stopEvent, flags) : 1;
}

// Fat Binary Registration
void** __hipRegisterFatBinary(const void* data) {
  ensure_init();
  return g_active_table->__hipRegisterFatBinary ? g_active_table->__hipRegisterFatBinary(data) : NULL;
}

void __hipUnregisterFatBinary(void** fatCubinHandle) {
  ensure_init();
  if (g_active_table->__hipUnregisterFatBinary)
    g_active_table->__hipUnregisterFatBinary(fatCubinHandle);
}

void __hipRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                           const char* deviceName, int thread_limit, void* tid, void* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize) {
  ensure_init();
  if (g_active_table->__hipRegisterFunction)
    g_active_table->__hipRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                          thread_limit, tid, bid, blockDim, gridDim, wSize);
}

void __hipRegisterVar(void** fatCubinHandle, char* hostVar, char* deviceAddress,
                      const char* deviceName, int ext, size_t size, int constant, int global) {
  ensure_init();
  if (g_active_table->__hipRegisterVar)
    g_active_table->__hipRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName,
                                     ext, size, constant, global);
}

// Kernel configuration stack
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
  ensure_init();
  typedef hipError_t (*pfn)(dim3, dim3, size_t, hipStream_t);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "__hipPushCallConfiguration");
  return fn ? fn(gridDim, blockDim, sharedMem, stream) : 1;
}

hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, hipStream_t* stream) {
  ensure_init();
  typedef hipError_t (*pfn)(dim3*, dim3*, size_t*, hipStream_t*);
  static pfn fn = NULL;
  if (!fn) fn = (pfn)dlsym(g_backend_lib, "__hipPopCallConfiguration");
  return fn ? fn(gridDim, blockDim, sharedMem, stream) : 1;
}
