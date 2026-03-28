// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Logging Interceptor
//
// This interceptor logs all HIP API calls to stderr.
// Useful for debugging and understanding HIP usage patterns.
//
// Environment variables:
// - HIP_LOG_FILE: Path to log file (default: stderr)
// - HIP_LOG_LEVEL: 0=off, 1=errors, 2=calls, 3=verbose (default: 2)
//===----------------------------------------------------------------------===//

#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "passthrough/hip_function_table.h"

//===----------------------------------------------------------------------===//
// Global State
//===----------------------------------------------------------------------===//

static hip_function_table_t* g_real = NULL;
static hip_function_table_t g_wrapper = {0};
static FILE* g_log_file = NULL;
static int g_log_level = 2;
static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;

//===----------------------------------------------------------------------===//
// Logging Helpers
//===----------------------------------------------------------------------===//

static void log_msg(int level, const char* fmt, ...) {
  if (level > g_log_level) return;
  
  pthread_mutex_lock(&g_log_mutex);
  
  // Timestamp
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  fprintf(g_log_file, "[%ld.%06ld] ", (long)ts.tv_sec, ts.tv_nsec / 1000);
  
  // Message
  va_list args;
  va_start(args, fmt);
  vfprintf(g_log_file, fmt, args);
  va_end(args);
  
  fprintf(g_log_file, "\n");
  fflush(g_log_file);
  
  pthread_mutex_unlock(&g_log_mutex);
}

static const char* memcpy_kind_name(hipMemcpyKind kind) {
  switch (kind) {
    case hipMemcpyHostToHost: return "H2H";
    case hipMemcpyHostToDevice: return "H2D";
    case hipMemcpyDeviceToHost: return "D2H";
    case hipMemcpyDeviceToDevice: return "D2D";
    case hipMemcpyDefault: return "Default";
    default: return "Unknown";
  }
}

//===----------------------------------------------------------------------===//
// Wrapper Functions
//===----------------------------------------------------------------------===//

static hipError_t wrap_hipInit(unsigned int flags) {
  log_msg(2, "hipInit(flags=0x%x)", flags);
  hipError_t err = g_real->hipInit(flags);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipGetDevice(int* deviceId) {
  hipError_t err = g_real->hipGetDevice(deviceId);
  log_msg(2, "hipGetDevice() -> device=%d, err=%d", deviceId ? *deviceId : -1, err);
  return err;
}

static hipError_t wrap_hipGetDeviceCount(int* count) {
  hipError_t err = g_real->hipGetDeviceCount(count);
  log_msg(2, "hipGetDeviceCount() -> count=%d, err=%d", count ? *count : -1, err);
  return err;
}

static hipError_t wrap_hipSetDevice(int deviceId) {
  log_msg(2, "hipSetDevice(device=%d)", deviceId);
  hipError_t err = g_real->hipSetDevice(deviceId);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipDeviceSynchronize(void) {
  log_msg(2, "hipDeviceSynchronize()");
  hipError_t err = g_real->hipDeviceSynchronize();
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipMalloc(void** ptr, size_t size) {
  log_msg(2, "hipMalloc(size=%zu)", size);
  hipError_t err = g_real->hipMalloc(ptr, size);
  log_msg(3, "  -> ptr=%p, err=%d", ptr ? *ptr : NULL, err);
  return err;
}

static hipError_t wrap_hipFree(void* ptr) {
  log_msg(2, "hipFree(ptr=%p)", ptr);
  hipError_t err = g_real->hipFree(ptr);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                                  hipMemcpyKind kind) {
  log_msg(2, "hipMemcpy(dst=%p, src=%p, size=%zu, kind=%s)",
          dst, src, sizeBytes, memcpy_kind_name(kind));
  hipError_t err = g_real->hipMemcpy(dst, src, sizeBytes, kind);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                                       hipMemcpyKind kind, hipStream_t stream) {
  log_msg(2, "hipMemcpyAsync(dst=%p, src=%p, size=%zu, kind=%s, stream=%p)",
          dst, src, sizeBytes, memcpy_kind_name(kind), (void*)stream);
  hipError_t err = g_real->hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipStreamCreate(hipStream_t* stream) {
  log_msg(2, "hipStreamCreate()");
  hipError_t err = g_real->hipStreamCreate(stream);
  log_msg(3, "  -> stream=%p, err=%d", stream ? (void*)*stream : NULL, err);
  return err;
}

static hipError_t wrap_hipStreamDestroy(hipStream_t stream) {
  log_msg(2, "hipStreamDestroy(stream=%p)", (void*)stream);
  hipError_t err = g_real->hipStreamDestroy(stream);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipStreamSynchronize(hipStream_t stream) {
  log_msg(2, "hipStreamSynchronize(stream=%p)", (void*)stream);
  hipError_t err = g_real->hipStreamSynchronize(stream);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra) {
  log_msg(2, "hipModuleLaunchKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), "
          "shared=%u, stream=%p)",
          (void*)f, gridDimX, gridDimY, gridDimZ,
          blockDimX, blockDimY, blockDimZ, sharedMemBytes, (void*)stream);
  hipError_t err = g_real->hipModuleLaunchKernel(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams, extra);
  log_msg(3, "  -> %d", err);
  return err;
}

static hipError_t wrap_hipLaunchKernel(const void* function_address,
                                        dim3 numBlocks, dim3 dimBlocks,
                                        void** args, size_t sharedMemBytes,
                                        hipStream_t stream) {
  log_msg(2, "hipLaunchKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), "
          "shared=%zu, stream=%p)",
          function_address, numBlocks.x, numBlocks.y, numBlocks.z,
          dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes, (void*)stream);
  hipError_t err = g_real->hipLaunchKernel(function_address, numBlocks,
                                            dimBlocks, args, sharedMemBytes,
                                            stream);
  log_msg(3, "  -> %d", err);
  return err;
}

static void** wrap___hipRegisterFatBinary(const void* data) {
  log_msg(2, "__hipRegisterFatBinary(data=%p)", data);
  void** result = g_real->__hipRegisterFatBinary(data);
  log_msg(3, "  -> handle=%p", (void*)result);
  return result;
}

static void wrap___hipUnregisterFatBinary(void** fatCubinHandle) {
  log_msg(2, "__hipUnregisterFatBinary(handle=%p)", (void*)fatCubinHandle);
  g_real->__hipUnregisterFatBinary(fatCubinHandle);
}

static void wrap___hipRegisterFunction(void** fatCubinHandle, const char* hostFun,
                                        char* deviceFun, const char* deviceName,
                                        int thread_limit, void* tid, void* bid,
                                        dim3* blockDim, dim3* gridDim,
                                        int* wSize) {
  log_msg(2, "__hipRegisterFunction(handle=%p, host=%p, device=%s)",
          (void*)fatCubinHandle, hostFun, deviceName ? deviceName : "(null)");
  g_real->__hipRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                 thread_limit, tid, bid, blockDim, gridDim, wSize);
}

//===----------------------------------------------------------------------===//
// Interceptor Interface
//===----------------------------------------------------------------------===//

__attribute__((visibility("default")))
hip_function_table_t* hip_interceptor_init(hip_function_table_t* real_functions) {
  g_real = real_functions;
  
  // Initialize logging
  const char* log_path = getenv("HIP_LOG_FILE");
  if (log_path && *log_path) {
    g_log_file = fopen(log_path, "w");
    if (!g_log_file) {
      g_log_file = stderr;
      fprintf(stderr, "logging_interceptor: failed to open log file: %s\n", log_path);
    }
  } else {
    g_log_file = stderr;
  }
  
  const char* log_level_str = getenv("HIP_LOG_LEVEL");
  if (log_level_str) {
    g_log_level = atoi(log_level_str);
  }
  
  log_msg(1, "HIP Logging Interceptor initialized (level=%d)", g_log_level);
  
  // Copy real table to wrapper
  memcpy(&g_wrapper, g_real, sizeof(g_wrapper));
  
  // Replace functions with wrappers
  g_wrapper.hipInit = wrap_hipInit;
  g_wrapper.hipGetDevice = wrap_hipGetDevice;
  g_wrapper.hipGetDeviceCount = wrap_hipGetDeviceCount;
  g_wrapper.hipSetDevice = wrap_hipSetDevice;
  g_wrapper.hipDeviceSynchronize = wrap_hipDeviceSynchronize;
  g_wrapper.hipMalloc = wrap_hipMalloc;
  g_wrapper.hipFree = wrap_hipFree;
  g_wrapper.hipMemcpy = wrap_hipMemcpy;
  g_wrapper.hipMemcpyAsync = wrap_hipMemcpyAsync;
  g_wrapper.hipStreamCreate = wrap_hipStreamCreate;
  g_wrapper.hipStreamDestroy = wrap_hipStreamDestroy;
  g_wrapper.hipStreamSynchronize = wrap_hipStreamSynchronize;
  g_wrapper.hipModuleLaunchKernel = wrap_hipModuleLaunchKernel;
  g_wrapper.hipLaunchKernel = wrap_hipLaunchKernel;
  g_wrapper.__hipRegisterFatBinary = wrap___hipRegisterFatBinary;
  g_wrapper.__hipUnregisterFatBinary = wrap___hipUnregisterFatBinary;
  g_wrapper.__hipRegisterFunction = wrap___hipRegisterFunction;
  
  return &g_wrapper;
}

__attribute__((visibility("default")))
void hip_interceptor_shutdown(void) {
  log_msg(1, "HIP Logging Interceptor shutting down");
  
  if (g_log_file && g_log_file != stderr) {
    fclose(g_log_file);
    g_log_file = NULL;
  }
}
