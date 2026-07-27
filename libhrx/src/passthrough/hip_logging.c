// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HIP Logging Interception Library
//
// This library is loaded by libhip_intercept to log all HIP API calls.
//
// Environment variables:
// - HIP_LOG_FILE: Path to log file (default: stderr)
// - HIP_LOG_LEVEL: 0=off, 1=errors, 2=calls, 3=verbose (default: 2)
//===----------------------------------------------------------------------===//

#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
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
  if (level > g_log_level || !g_log_file) return;

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
    case hipMemcpyHostToHost:
      return "H2H";
    case hipMemcpyHostToDevice:
      return "H2D";
    case hipMemcpyDeviceToHost:
      return "D2H";
    case hipMemcpyDeviceToDevice:
      return "D2D";
    case hipMemcpyDefault:
      return "Default";
    case hipMemcpyDeviceToDeviceNoCU:
      return "D2DNoCU";
    default:
      return "Unknown";
  }
}

static const char* device_attribute_name(hipDeviceAttribute_t attr) {
  switch (attr) {
    case 0:
      return "MaxThreadsPerBlock";
    case 1:
      return "MaxBlockDimX";
    case 2:
      return "MaxBlockDimY";
    case 3:
      return "MaxBlockDimZ";
    case 4:
      return "MaxGridDimX";
    case 5:
      return "MaxGridDimY";
    case 6:
      return "MaxGridDimZ";
    case 7:
      return "MaxSharedMemoryPerBlock";
    case 8:
      return "TotalConstantMemory";
    case 9:
      return "WarpSize";
    case 10:
      return "MaxRegistersPerBlock";
    case 11:
      return "ClockRate";
    case 15:
      return "MemoryClockRate";
    case 16:
      return "MemoryBusWidth";
    case 17:
      return "MultiProcessorCount";
    case 19:
      return "ComputeCapabilityMajor";
    case 20:
      return "ComputeCapabilityMinor";
    case 21:
      return "L2CacheSize";
    case 22:
      return "MaxThreadsPerMultiProcessor";
    case 24:
      return "ConcurrentKernels";
    case 28:
      return "PCIBusId";
    case 29:
      return "PCIDeviceId";
    case 32:
      return "MaxSharedMemoryPerMultiprocessor";
    case 43:
      return "IsMultiGpuBoard";
    case 70:
      return "CooperativeLaunch";
    case 71:
      return "CooperativeMultiDeviceLaunch";
    case 87:
      return "PageableMemoryAccess";
    case 95:
      return "ManagedMemory";
    case 96:
      return "DirectManagedMemAccessFromHost";
    case 100:
      return "ConcurrentManagedAccess";
    case 101:
      return "PageableMemoryAccessUsesHostPageTables";
    case 1000:
      return "GCN_ARCH";
    case 1003:
      return "GCN_ARCH_NAME";
    default: {
      // Return a static buffer with the numeric value for unknown attributes
      static __thread char buf[32];
      snprintf(buf, sizeof(buf), "attr_%u", (unsigned)attr);
      return buf;
    }
  }
}

//===----------------------------------------------------------------------===//
// Wrapper Functions
//===----------------------------------------------------------------------===//

static hipError_t wrap_hipInit(unsigned int flags) {
  hipError_t err = g_real->hipInit(flags);
  log_msg(2, "hipInit(flags=0x%x) -> %d", flags, err);
  return err;
}

static hipError_t wrap_hipGetDevice(int* deviceId) {
  hipError_t err = g_real->hipGetDevice(deviceId);
  log_msg(2, "hipGetDevice() -> device=%d, ret=%d", deviceId ? *deviceId : -1,
          err);
  return err;
}

static hipError_t wrap_hipGetDeviceCount(int* count) {
  hipError_t err = g_real->hipGetDeviceCount(count);
  log_msg(2, "hipGetDeviceCount() -> count=%d, ret=%d", count ? *count : -1,
          err);
  return err;
}

static hipError_t wrap_hipSetDevice(int deviceId) {
  hipError_t err = g_real->hipSetDevice(deviceId);
  log_msg(2, "hipSetDevice(%d) -> %d", deviceId, err);
  return err;
}

static hipError_t wrap_hipDeviceSynchronize(void) {
  hipError_t err = g_real->hipDeviceSynchronize();
  log_msg(2, "hipDeviceSynchronize() -> %d", err);
  return err;
}

static hipError_t wrap_hipDriverGetVersion(int* driverVersion) {
  hipError_t err = g_real->hipDriverGetVersion(driverVersion);
  log_msg(2, "hipDriverGetVersion() -> version=%d, ret=%d",
          driverVersion ? *driverVersion : -1, err);
  return err;
}

static hipError_t wrap_hipRuntimeGetVersion(int* runtimeVersion) {
  hipError_t err = g_real->hipRuntimeGetVersion(runtimeVersion);
  log_msg(2, "hipRuntimeGetVersion() -> version=%d, ret=%d",
          runtimeVersion ? *runtimeVersion : -1, err);
  return err;
}

static hipError_t wrap_hipGetDeviceProperties(hipDeviceProp_t* prop,
                                              int deviceId) {
  hipError_t err = g_real->hipGetDeviceProperties(prop, deviceId);
  if (err == 0 && prop) {
    // Log key device properties that affect kernel launch decisions
    // hipDeviceProp_t layout: name[256] then numeric fields
    // We cast through char* to access the fields portably
    const char* name = (const char*)prop;  // name is first field, char[256]
    // Access fields after the name using known offsets from HIP headers:
    // These are the fields most relevant to kernel configuration
    log_msg(2, "hipGetDeviceProperties(device=%d) -> ret=%d", deviceId, err);
    log_msg(2, "  DeviceProps: name=%s", name);
    // Use a byte-level dump of the critical numeric fields to avoid
    // depending on the exact hipDeviceProp_t struct layout.
    // Instead, we'll re-query the important attributes individually via
    // hipDeviceGetAttribute which we also log.
    // For now, log what we can safely:
    log_msg(2,
            "  DeviceProps: (use hipDeviceGetAttribute logs for individual "
            "fields)");
  } else {
    log_msg(2, "hipGetDeviceProperties(device=%d) -> ret=%d", deviceId, err);
  }
  return err;
}

static hipError_t wrap_hipDeviceGetAttribute(int* value,
                                             hipDeviceAttribute_t attr,
                                             int deviceId) {
  hipError_t err = g_real->hipDeviceGetAttribute(value, attr, deviceId);
  log_msg(2,
          "hipDeviceGetAttribute(attr=%s(%u), device=%d) -> value=%d, ret=%d",
          device_attribute_name(attr), (unsigned)attr, deviceId,
          value ? *value : -1, err);
  return err;
}

static hipError_t wrap_hipDeviceGetName(char* name, int len, int deviceId) {
  hipError_t err = g_real->hipDeviceGetName(name, len, deviceId);
  log_msg(2, "hipDeviceGetName(device=%d) -> name=%s, ret=%d", deviceId,
          name ? name : "(null)", err);
  return err;
}

static hipError_t wrap_hipMemGetInfo(size_t* free_mem, size_t* total) {
  hipError_t err = g_real->hipMemGetInfo(free_mem, total);
  log_msg(2, "hipMemGetInfo() -> free=%zu, total=%zu, ret=%d",
          free_mem ? *free_mem : 0, total ? *total : 0, err);
  return err;
}

static hipError_t wrap_hipMalloc(void** ptr, size_t size) {
  hipError_t err = g_real->hipMalloc(ptr, size);
  log_msg(2, "hipMalloc(size=%zu) -> ptr=%p, ret=%d", size, ptr ? *ptr : NULL,
          err);
  return err;
}

static hipError_t wrap_hipFree(void* ptr) {
  hipError_t err = g_real->hipFree(ptr);
  log_msg(2, "hipFree(%p) -> %d", ptr, err);
  return err;
}

static hipError_t wrap_hipHostMalloc(void** ptr, size_t size,
                                     unsigned int flags) {
  hipError_t err = g_real->hipHostMalloc(ptr, size, flags);
  log_msg(2, "hipHostMalloc(size=%zu, flags=0x%x) -> ptr=%p, ret=%d", size,
          flags, ptr ? *ptr : NULL, err);
  return err;
}

static hipError_t wrap_hipHostFree(void* ptr) {
  hipError_t err = g_real->hipHostFree(ptr);
  log_msg(2, "hipHostFree(%p) -> %d", ptr, err);
  return err;
}

static hipError_t wrap_hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                                 hipMemcpyKind kind) {
  hipError_t err = g_real->hipMemcpy(dst, src, sizeBytes, kind);
  log_msg(2, "hipMemcpy(dst=%p, src=%p, size=%zu, kind=%s) -> %d", dst, src,
          sizeBytes, memcpy_kind_name(kind), err);
  return err;
}

static hipError_t wrap_hipMemcpyAsync(void* dst, const void* src,
                                      size_t sizeBytes, hipMemcpyKind kind,
                                      hipStream_t stream) {
  hipError_t err = g_real->hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  log_msg(2,
          "hipMemcpyAsync(dst=%p, src=%p, size=%zu, kind=%s, stream=%p) -> %d",
          dst, src, sizeBytes, memcpy_kind_name(kind), (void*)stream, err);
  return err;
}

static hipError_t wrap_hipMemset(void* dst, int value, size_t sizeBytes) {
  hipError_t err = g_real->hipMemset(dst, value, sizeBytes);
  log_msg(2, "hipMemset(dst=%p, value=0x%02x, size=%zu) -> %d", dst, value,
          sizeBytes, err);
  return err;
}

static hipError_t wrap_hipMemsetAsync(void* dst, int value, size_t sizeBytes,
                                      hipStream_t stream) {
  hipError_t err = g_real->hipMemsetAsync(dst, value, sizeBytes, stream);
  log_msg(2, "hipMemsetAsync(dst=%p, value=0x%02x, size=%zu, stream=%p) -> %d",
          dst, value, sizeBytes, (void*)stream, err);
  return err;
}

static hipError_t wrap_hipStreamCreate(hipStream_t* stream) {
  hipError_t err = g_real->hipStreamCreate(stream);
  log_msg(2, "hipStreamCreate() -> stream=%p, ret=%d",
          stream ? (void*)*stream : NULL, err);
  return err;
}

static hipError_t wrap_hipStreamCreateWithFlags(hipStream_t* stream,
                                                unsigned int flags) {
  hipError_t err = g_real->hipStreamCreateWithFlags(stream, flags);
  log_msg(2, "hipStreamCreateWithFlags(flags=0x%x) -> stream=%p, ret=%d", flags,
          stream ? (void*)*stream : NULL, err);
  return err;
}

static hipError_t wrap_hipStreamDestroy(hipStream_t stream) {
  hipError_t err = g_real->hipStreamDestroy(stream);
  log_msg(2, "hipStreamDestroy(stream=%p) -> %d", (void*)stream, err);
  return err;
}

static hipError_t wrap_hipStreamSynchronize(hipStream_t stream) {
  hipError_t err = g_real->hipStreamSynchronize(stream);
  log_msg(2, "hipStreamSynchronize(stream=%p) -> %d", (void*)stream, err);
  return err;
}

static hipError_t wrap_hipEventCreate(hipEvent_t* event) {
  hipError_t err = g_real->hipEventCreate(event);
  log_msg(2, "hipEventCreate() -> event=%p, ret=%d",
          event ? (void*)*event : NULL, err);
  return err;
}

static hipError_t wrap_hipEventCreateWithFlags(hipEvent_t* event,
                                               unsigned int flags) {
  hipError_t err = g_real->hipEventCreateWithFlags(event, flags);
  log_msg(2, "hipEventCreateWithFlags(flags=0x%x) -> event=%p, ret=%d", flags,
          event ? (void*)*event : NULL, err);
  return err;
}

static hipError_t wrap_hipEventDestroy(hipEvent_t event) {
  hipError_t err = g_real->hipEventDestroy(event);
  log_msg(2, "hipEventDestroy(event=%p) -> %d", (void*)event, err);
  return err;
}

static hipError_t wrap_hipEventRecord(hipEvent_t event, hipStream_t stream) {
  hipError_t err = g_real->hipEventRecord(event, stream);
  log_msg(2, "hipEventRecord(event=%p, stream=%p) -> %d", (void*)event,
          (void*)stream, err);
  return err;
}

static hipError_t wrap_hipEventSynchronize(hipEvent_t event) {
  hipError_t err = g_real->hipEventSynchronize(event);
  log_msg(2, "hipEventSynchronize(event=%p) -> %d", (void*)event, err);
  return err;
}

static hipError_t wrap_hipEventQuery(hipEvent_t event) {
  hipError_t err = g_real->hipEventQuery(event);
  log_msg(2, "hipEventQuery(event=%p) -> %d", (void*)event, err);
  return err;
}

static hipError_t wrap_hipEventElapsedTime(float* ms, hipEvent_t start,
                                           hipEvent_t stop) {
  hipError_t err = g_real->hipEventElapsedTime(ms, start, stop);
  log_msg(2, "hipEventElapsedTime(start=%p, stop=%p) -> ms=%f, ret=%d",
          (void*)start, (void*)stop, ms ? *ms : 0.0f, err);
  return err;
}

static hipError_t wrap_hipModuleLoad(hipModule_t* module, const char* fname) {
  hipError_t err = g_real->hipModuleLoad(module, fname);
  log_msg(2, "hipModuleLoad(file=%s) -> module=%p, ret=%d",
          fname ? fname : "(null)", module ? (void*)*module : NULL, err);
  return err;
}

static hipError_t wrap_hipModuleLoadData(hipModule_t* module,
                                         const void* image) {
  hipError_t err = g_real->hipModuleLoadData(module, image);
  log_msg(2, "hipModuleLoadData(image=%p) -> module=%p, ret=%d", image,
          module ? (void*)*module : NULL, err);
  return err;
}

static hipError_t wrap_hipModuleUnload(hipModule_t module) {
  hipError_t err = g_real->hipModuleUnload(module);
  log_msg(2, "hipModuleUnload(module=%p) -> ret=%d", (void*)module, err);
  return err;
}

static hipError_t wrap_hipModuleGetFunction(hipFunction_t* function,
                                            hipModule_t module,
                                            const char* kname) {
  hipError_t err = g_real->hipModuleGetFunction(function, module, kname);
  log_msg(2, "hipModuleGetFunction(module=%p, name=%s) -> function=%p, ret=%d",
          (void*)module, kname ? kname : "(null)",
          function ? (void*)*function : NULL, err);
  return err;
}

static hipError_t wrap_hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                                          hipModule_t module,
                                          const char* name) {
  hipError_t err = g_real->hipModuleGetGlobal(dptr, bytes, module, name);
  log_msg(2,
          "hipModuleGetGlobal(module=%p, name=%s) -> dptr=%p, bytes=%zu, "
          "ret=%d",
          (void*)module, name ? name : "(null)",
          dptr ? (void*)(uintptr_t)*dptr : NULL, bytes ? *bytes : 0, err);
  return err;
}

static hipError_t wrap_hipStreamQuery(hipStream_t stream) {
  hipError_t err = g_real->hipStreamQuery(stream);
  log_msg(2, "hipStreamQuery(stream=%p) -> %d", (void*)stream, err);
  return err;
}

static hipError_t wrap_hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                                          unsigned int flags) {
  hipError_t err = g_real->hipStreamWaitEvent(stream, event, flags);
  log_msg(2, "hipStreamWaitEvent(stream=%p, event=%p, flags=0x%x) -> %d",
          (void*)stream, (void*)event, flags, err);
  return err;
}

static hipError_t wrap_hipGetLastError(void) {
  hipError_t err = g_real->hipGetLastError();
  if (err != 0) {
    log_msg(1, "hipGetLastError() -> %d (ERROR)", err);
  } else {
    log_msg(3, "hipGetLastError() -> 0");
  }
  return err;
}

static hipError_t wrap_hipPeekAtLastError(void) {
  hipError_t err = g_real->hipPeekAtLastError();
  if (err != 0) {
    log_msg(1, "hipPeekAtLastError() -> %d (ERROR)", err);
  } else {
    log_msg(3, "hipPeekAtLastError() -> 0");
  }
  return err;
}

static hipError_t wrap_hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra) {
  hipError_t err = g_real->hipModuleLaunchKernel(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams, extra);
  log_msg(2,
          "hipModuleLaunchKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), "
          "shared=%u, stream=%p, extra=%p) -> %d",
          (void*)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
          blockDimZ, sharedMemBytes, (void*)stream, (void*)extra, err);
  return err;
}

static hipError_t wrap_hipExtModuleLaunchKernel(
    hipFunction_t f, unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ, unsigned int localWorkSizeX,
    unsigned int localWorkSizeY, unsigned int localWorkSizeZ,
    size_t sharedMemBytes, hipStream_t stream, void** kernelParams,
    void** extra, hipEvent_t startEvent, hipEvent_t stopEvent,
    unsigned int flags) {
  hipError_t err = g_real->hipExtModuleLaunchKernel(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX,
      localWorkSizeY, localWorkSizeZ, sharedMemBytes, stream, kernelParams,
      extra, startEvent, stopEvent, flags);
  log_msg(
      2,
      "hipExtModuleLaunchKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), "
      "shared=%zu, stream=%p, flags=0x%x) -> %d",
      (void*)f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
      localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes,
      (void*)stream, flags, err);
  return err;
}

static hipError_t wrap_hipLaunchKernel(const void* function_address,
                                       dim3 numBlocks, dim3 dimBlocks,
                                       void** args, size_t sharedMemBytes,
                                       hipStream_t stream) {
  hipError_t err = g_real->hipLaunchKernel(
      function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
  log_msg(2,
          "hipLaunchKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), "
          "shared=%zu, stream=%p) -> %d",
          function_address, numBlocks.x, numBlocks.y, numBlocks.z, dimBlocks.x,
          dimBlocks.y, dimBlocks.z, sharedMemBytes, (void*)stream, err);
  return err;
}

static void** wrap___hipRegisterFatBinary(const void* data) {
  void** result = g_real->__hipRegisterFatBinary(data);
  log_msg(2, "__hipRegisterFatBinary(data=%p) -> handle=%p", data,
          (void*)result);
  return result;
}

static void wrap___hipUnregisterFatBinary(void** fatCubinHandle) {
  log_msg(2, "__hipUnregisterFatBinary(handle=%p)", (void*)fatCubinHandle);
  g_real->__hipUnregisterFatBinary(fatCubinHandle);
}

static void wrap___hipRegisterFunction(void** fatCubinHandle,
                                       const char* hostFun, char* deviceFun,
                                       const char* deviceName, int thread_limit,
                                       void* tid, void* bid, dim3* blockDim,
                                       dim3* gridDim, int* wSize) {
  log_msg(2, "__hipRegisterFunction(handle=%p, host=%p, device=%s)",
          (void*)fatCubinHandle, hostFun, deviceName ? deviceName : "(null)");
  g_real->__hipRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                thread_limit, tid, bid, blockDim, gridDim,
                                wSize);
}

static void wrap___hipRegisterVar(void** fatCubinHandle, char* hostVar,
                                  char* deviceAddress, const char* deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
  log_msg(2, "__hipRegisterVar(handle=%p, name=%s, size=%zu)",
          (void*)fatCubinHandle, deviceName ? deviceName : "(null)", size);
  g_real->__hipRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName,
                           ext, size, constant, global);
}

//===----------------------------------------------------------------------===//
// Interceptor Interface
//===----------------------------------------------------------------------===//

__attribute__((visibility("default"))) hip_function_table_t*
hip_interceptor_init(hip_function_table_t* real_functions) {
  g_real = real_functions;

  // Initialize logging
  const char* log_path = getenv("HIP_LOG_FILE");
  if (log_path && *log_path) {
    g_log_file = fopen(log_path, "w");
    if (!g_log_file) {
      g_log_file = stderr;
      fprintf(stderr, "hip_logging: failed to open log file: %s\n", log_path);
    }
  } else {
    g_log_file = stderr;
  }

  const char* log_level_str = getenv("HIP_LOG_LEVEL");
  if (log_level_str) {
    g_log_level = atoi(log_level_str);
  }

  log_msg(1, "HIP Logging initialized (level=%d)", g_log_level);

  // Copy real table to wrapper
  memcpy(&g_wrapper, g_real, sizeof(g_wrapper));

  // Replace functions with wrappers
  g_wrapper.hipInit = wrap_hipInit;
  g_wrapper.hipDriverGetVersion = wrap_hipDriverGetVersion;
  g_wrapper.hipRuntimeGetVersion = wrap_hipRuntimeGetVersion;
  g_wrapper.hipGetDevice = wrap_hipGetDevice;
  g_wrapper.hipGetDeviceCount = wrap_hipGetDeviceCount;
  g_wrapper.hipSetDevice = wrap_hipSetDevice;
  g_wrapper.hipDeviceSynchronize = wrap_hipDeviceSynchronize;
  g_wrapper.hipGetDeviceProperties = wrap_hipGetDeviceProperties;
  g_wrapper.hipDeviceGetAttribute = wrap_hipDeviceGetAttribute;
  g_wrapper.hipDeviceGetName = wrap_hipDeviceGetName;
  g_wrapper.hipMalloc = wrap_hipMalloc;
  g_wrapper.hipFree = wrap_hipFree;
  g_wrapper.hipHostMalloc = wrap_hipHostMalloc;
  g_wrapper.hipHostFree = wrap_hipHostFree;
  g_wrapper.hipMemGetInfo = wrap_hipMemGetInfo;
  g_wrapper.hipMemcpy = wrap_hipMemcpy;
  g_wrapper.hipMemcpyAsync = wrap_hipMemcpyAsync;
  g_wrapper.hipMemset = wrap_hipMemset;
  g_wrapper.hipMemsetAsync = wrap_hipMemsetAsync;
  g_wrapper.hipStreamCreate = wrap_hipStreamCreate;
  g_wrapper.hipStreamCreateWithFlags = wrap_hipStreamCreateWithFlags;
  g_wrapper.hipStreamDestroy = wrap_hipStreamDestroy;
  g_wrapper.hipStreamSynchronize = wrap_hipStreamSynchronize;
  g_wrapper.hipStreamQuery = wrap_hipStreamQuery;
  g_wrapper.hipStreamWaitEvent = wrap_hipStreamWaitEvent;
  g_wrapper.hipEventCreate = wrap_hipEventCreate;
  g_wrapper.hipEventCreateWithFlags = wrap_hipEventCreateWithFlags;
  g_wrapper.hipEventDestroy = wrap_hipEventDestroy;
  g_wrapper.hipEventRecord = wrap_hipEventRecord;
  g_wrapper.hipEventSynchronize = wrap_hipEventSynchronize;
  g_wrapper.hipEventQuery = wrap_hipEventQuery;
  g_wrapper.hipEventElapsedTime = wrap_hipEventElapsedTime;
  g_wrapper.hipModuleLoad = wrap_hipModuleLoad;
  g_wrapper.hipModuleLoadData = wrap_hipModuleLoadData;
  g_wrapper.hipModuleUnload = wrap_hipModuleUnload;
  g_wrapper.hipModuleGetFunction = wrap_hipModuleGetFunction;
  g_wrapper.hipModuleGetGlobal = wrap_hipModuleGetGlobal;
  g_wrapper.hipModuleLaunchKernel = wrap_hipModuleLaunchKernel;
  g_wrapper.hipExtModuleLaunchKernel = wrap_hipExtModuleLaunchKernel;
  g_wrapper.hipLaunchKernel = wrap_hipLaunchKernel;
  g_wrapper.__hipRegisterFatBinary = wrap___hipRegisterFatBinary;
  g_wrapper.__hipUnregisterFatBinary = wrap___hipUnregisterFatBinary;
  g_wrapper.__hipRegisterFunction = wrap___hipRegisterFunction;
  g_wrapper.__hipRegisterVar = wrap___hipRegisterVar;
  g_wrapper.hipGetLastError = wrap_hipGetLastError;
  g_wrapper.hipPeekAtLastError = wrap_hipPeekAtLastError;

  return &g_wrapper;
}

__attribute__((visibility("default"))) pfn_hip_log_fn
hip_interceptor_get_log_fn(void) {
  return (pfn_hip_log_fn)log_msg;
}

__attribute__((visibility("default"))) void hip_interceptor_shutdown(void) {
  log_msg(1, "HIP Logging shutting down");

  if (g_log_file && g_log_file != stderr) {
    fclose(g_log_file);
    g_log_file = NULL;
  }
}
