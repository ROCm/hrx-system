// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PASSTHROUGH_HIP_FUNCTION_TABLE_H_
#define IREE_PASSTHROUGH_HIP_FUNCTION_TABLE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// HIP Type Definitions (minimal set needed for function signatures)
//===----------------------------------------------------------------------===//

typedef int hipError_t;
typedef void *hipStream_t;
typedef void *hipEvent_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipDeviceptr_t;
typedef void *hipCtx_t;
typedef void *hipDevice_t;
typedef void *hipArray_t;
typedef void *hipMipmappedArray_t;
typedef void *hipTextureObject_t;
typedef void *hipSurfaceObject_t;
typedef void *hipExternalMemory_t;
typedef void *hipExternalSemaphore_t;
typedef void *hipGraph_t;
typedef void *hipGraphExec_t;
typedef void *hipGraphNode_t;
typedef void *hipMemPool_t;
typedef void *hipUserObject_t;

typedef struct dim3 {
  unsigned int x, y, z;
} dim3;

// Forward declarations for texture types
struct textureReference;

typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
} hipMemcpyKind;

typedef struct hipDeviceProp_t hipDeviceProp_t;
typedef struct hipPointerAttribute_t hipPointerAttribute_t;
typedef struct hipFuncAttributes hipFuncAttributes;
typedef struct hipMemPoolProps hipMemPoolProps;
typedef struct hipMemPoolPtrExportData hipMemPoolPtrExportData;
typedef struct hipMemAllocationProp hipMemAllocationProp;
typedef struct hipMemAccessDesc hipMemAccessDesc;
typedef struct hipArrayDesc hipArrayDesc;
typedef struct hipChannelFormatDesc hipChannelFormatDesc;
typedef struct hipResourceDesc hipResourceDesc;
typedef struct hipTextureDesc hipTextureDesc;
typedef struct hipResourceViewDesc hipResourceViewDesc;
typedef struct hipExternalMemoryHandleDesc hipExternalMemoryHandleDesc;
typedef struct hipExternalMemoryBufferDesc hipExternalMemoryBufferDesc;
typedef struct hipExternalSemaphoreHandleDesc hipExternalSemaphoreHandleDesc;
typedef struct hipExternalSemaphoreSignalParams
    hipExternalSemaphoreSignalParams;
typedef struct hipExternalSemaphoreWaitParams hipExternalSemaphoreWaitParams;
typedef struct hipKernelNodeParams hipKernelNodeParams;
typedef struct hipMemcpy3DParms hipMemcpy3DParms;
typedef struct hipMemsetParams hipMemsetParams;
typedef struct hipHostNodeParams hipHostNodeParams;
typedef struct hipAccessPolicyWindow hipAccessPolicyWindow;
typedef struct HIP_MEMCPY3D HIP_MEMCPY3D;
typedef struct hipPitchedPtr hipPitchedPtr;
typedef struct hipExtent hipExtent;
typedef struct hipPos hipPos;

// IPC handle types need complete definitions for pass-by-value
typedef struct hipIpcMemHandle_t {
  char reserved[64];
} hipIpcMemHandle_t;

typedef struct hipIpcEventHandle_t {
  char reserved[64];
} hipIpcEventHandle_t;

typedef unsigned int hipLimit_t;
typedef unsigned int hipFuncCache_t;
typedef unsigned int hipSharedMemConfig;
typedef unsigned int hipStreamCaptureMode;
typedef unsigned int hipStreamCaptureStatus;
typedef unsigned int hipDeviceAttribute_t;
typedef unsigned int hipComputeMode;
typedef unsigned int hipMemoryType;
typedef unsigned int hipMemPoolAttr;
typedef unsigned int hipMemLocationType;
typedef unsigned int hipMemAccessFlags;
typedef unsigned int hipMemAllocationType;
typedef unsigned int hipMemAllocationHandleType;
typedef unsigned int hipGraphNodeType;
typedef unsigned int hipGraphExecUpdateResult;
typedef unsigned int hipUserObjectFlags;
typedef unsigned int hipUserObjectRetainFlags;
typedef unsigned int hipGraphInstantiateFlags;
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status,
                                    void *userData);
typedef void (*hipHostFn_t)(void *userData);

// Forward declare the function table
typedef struct hip_function_table_t hip_function_table_t;

//===----------------------------------------------------------------------===//
// Function pointer types for all HIP functions
//===----------------------------------------------------------------------===//

// Device Management
typedef hipError_t (*pfn_hipInit)(unsigned int flags);
typedef hipError_t (*pfn_hipDriverGetVersion)(int *driverVersion);
typedef hipError_t (*pfn_hipRuntimeGetVersion)(int *runtimeVersion);
typedef hipError_t (*pfn_hipGetDevice)(int *deviceId);
typedef hipError_t (*pfn_hipGetDeviceCount)(int *count);
typedef hipError_t (*pfn_hipSetDevice)(int deviceId);
typedef hipError_t (*pfn_hipDeviceReset)(void);
typedef hipError_t (*pfn_hipDeviceSynchronize)(void);
typedef hipError_t (*pfn_hipGetDeviceProperties)(hipDeviceProp_t *prop,
                                                 int deviceId);
typedef hipError_t (*pfn_hipDeviceGetAttribute)(int *value,
                                                hipDeviceAttribute_t attr,
                                                int deviceId);
typedef hipError_t (*pfn_hipDeviceGetName)(char *name, int len, int deviceId);
typedef hipError_t (*pfn_hipDeviceGetPCIBusId)(char *pciBusId, int len,
                                               int device);
typedef hipError_t (*pfn_hipDeviceGetByPCIBusId)(int *device,
                                                 const char *pciBusId);
typedef hipError_t (*pfn_hipDeviceTotalMem)(size_t *bytes, int deviceId);
typedef hipError_t (*pfn_hipDeviceGetLimit)(size_t *pValue, hipLimit_t limit);
typedef hipError_t (*pfn_hipDeviceSetLimit)(hipLimit_t limit, size_t value);
typedef hipError_t (*pfn_hipDeviceSetCacheConfig)(hipFuncCache_t cacheConfig);
typedef hipError_t (*pfn_hipDeviceGetCacheConfig)(hipFuncCache_t *cacheConfig);
typedef hipError_t (*pfn_hipDeviceGetSharedMemConfig)(
    hipSharedMemConfig *pConfig);
typedef hipError_t (*pfn_hipDeviceSetSharedMemConfig)(
    hipSharedMemConfig config);
typedef hipError_t (*pfn_hipDeviceGetStreamPriorityRange)(
    int *leastPriority, int *greatestPriority);
typedef hipError_t (*pfn_hipSetDeviceFlags)(unsigned int flags);
typedef hipError_t (*pfn_hipGetDeviceFlags)(unsigned int *flags);
typedef hipError_t (*pfn_hipDeviceCanAccessPeer)(int *canAccessPeer,
                                                 int deviceId,
                                                 int peerDeviceId);
typedef hipError_t (*pfn_hipDeviceEnablePeerAccess)(int peerDeviceId,
                                                    unsigned int flags);
typedef hipError_t (*pfn_hipDeviceDisablePeerAccess)(int peerDeviceId);
typedef hipError_t (*pfn_hipChooseDevice)(int *device,
                                          const hipDeviceProp_t *prop);
typedef hipError_t (*pfn_hipDeviceGetP2PAttribute)(int *value,
                                                   unsigned int attr,
                                                   int srcDevice,
                                                   int dstDevice);

// Memory Management
typedef hipError_t (*pfn_hipMalloc)(void **ptr, size_t size);
typedef hipError_t (*pfn_hipFree)(void *ptr);
typedef hipError_t (*pfn_hipHostMalloc)(void **ptr, size_t size,
                                        unsigned int flags);
typedef hipError_t (*pfn_hipHostFree)(void *ptr);
typedef hipError_t (*pfn_hipMemGetInfo)(size_t *free, size_t *total);

// Memory Copy
typedef hipError_t (*pfn_hipMemcpy)(void *dst, const void *src,
                                    size_t sizeBytes, hipMemcpyKind kind);
typedef hipError_t (*pfn_hipMemcpyAsync)(void *dst, const void *src,
                                         size_t sizeBytes, hipMemcpyKind kind,
                                         hipStream_t stream);
typedef hipError_t (*pfn_hipMemset)(void *dst, int value, size_t sizeBytes);
typedef hipError_t (*pfn_hipMemsetAsync)(void *dst, int value, size_t sizeBytes,
                                         hipStream_t stream);

// Stream Management
typedef hipError_t (*pfn_hipStreamCreate)(hipStream_t *stream);
typedef hipError_t (*pfn_hipStreamCreateWithFlags)(hipStream_t *stream,
                                                   unsigned int flags);
typedef hipError_t (*pfn_hipStreamDestroy)(hipStream_t stream);
typedef hipError_t (*pfn_hipStreamSynchronize)(hipStream_t stream);
typedef hipError_t (*pfn_hipStreamQuery)(hipStream_t stream);
typedef hipError_t (*pfn_hipStreamWaitEvent)(hipStream_t stream,
                                             hipEvent_t event,
                                             unsigned int flags);

// Event Management
typedef hipError_t (*pfn_hipEventCreate)(hipEvent_t *event);
typedef hipError_t (*pfn_hipEventCreateWithFlags)(hipEvent_t *event,
                                                  unsigned int flags);
typedef hipError_t (*pfn_hipEventDestroy)(hipEvent_t event);
typedef hipError_t (*pfn_hipEventRecord)(hipEvent_t event, hipStream_t stream);
typedef hipError_t (*pfn_hipEventSynchronize)(hipEvent_t event);
typedef hipError_t (*pfn_hipEventQuery)(hipEvent_t event);
typedef hipError_t (*pfn_hipEventElapsedTime)(float *ms, hipEvent_t start,
                                              hipEvent_t stop);

// Module Management
typedef hipError_t (*pfn_hipModuleLoad)(hipModule_t *module, const char *fname);
typedef hipError_t (*pfn_hipModuleLoadData)(hipModule_t *module,
                                            const void *image);
typedef hipError_t (*pfn_hipModuleUnload)(hipModule_t module);
typedef hipError_t (*pfn_hipModuleGetFunction)(hipFunction_t *function,
                                               hipModule_t module,
                                               const char *kname);
typedef hipError_t (*pfn_hipModuleGetGlobal)(hipDeviceptr_t *dptr,
                                             size_t *bytes, hipModule_t hmod,
                                             const char *name);
typedef hipError_t (*pfn_hipModuleLaunchKernel)(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void **kernelParams, void **extra);
typedef hipError_t (*pfn_hipLaunchKernel)(const void *function_address,
                                          dim3 numBlocks, dim3 dimBlocks,
                                          void **args, size_t sharedMemBytes,
                                          hipStream_t stream);
typedef hipError_t (*pfn_hipExtModuleLaunchKernel)(
    hipFunction_t f, unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ, unsigned int localWorkSizeX,
    unsigned int localWorkSizeY, unsigned int localWorkSizeZ,
    size_t sharedMemBytes, hipStream_t stream, void **kernelParams,
    void **extra, hipEvent_t startEvent, hipEvent_t stopEvent,
    unsigned int flags);

// Fat Binary Registration
typedef void **(*pfn___hipRegisterFatBinary)(const void *data);
typedef void (*pfn___hipUnregisterFatBinary)(void **fatCubinHandle);
typedef void (*pfn___hipRegisterFunction)(void **fatCubinHandle,
                                          const char *hostFun, char *deviceFun,
                                          const char *deviceName,
                                          int thread_limit, void *tid,
                                          void *bid, dim3 *blockDim,
                                          dim3 *gridDim, int *wSize);
typedef void (*pfn___hipRegisterVar)(void **fatCubinHandle, char *hostVar,
                                     char *deviceAddress,
                                     const char *deviceName, int ext,
                                     size_t size, int constant, int global);

// Error Handling
typedef const char *(*pfn_hipGetErrorString)(hipError_t hipError);
typedef const char *(*pfn_hipGetErrorName)(hipError_t hipError);
typedef hipError_t (*pfn_hipGetLastError)(void);
typedef hipError_t (*pfn_hipPeekAtLastError)(void);

//===----------------------------------------------------------------------===//
// Function Table Structure
//===----------------------------------------------------------------------===//

struct hip_function_table_t {
  // Version for compatibility checking
  uint32_t version;
  uint32_t struct_size;

  // Device Management
  pfn_hipInit hipInit;
  pfn_hipDriverGetVersion hipDriverGetVersion;
  pfn_hipRuntimeGetVersion hipRuntimeGetVersion;
  pfn_hipGetDevice hipGetDevice;
  pfn_hipGetDeviceCount hipGetDeviceCount;
  pfn_hipSetDevice hipSetDevice;
  pfn_hipDeviceReset hipDeviceReset;
  pfn_hipDeviceSynchronize hipDeviceSynchronize;
  pfn_hipGetDeviceProperties hipGetDeviceProperties;
  pfn_hipDeviceGetAttribute hipDeviceGetAttribute;
  pfn_hipDeviceGetName hipDeviceGetName;

  // Memory Management
  pfn_hipMalloc hipMalloc;
  pfn_hipFree hipFree;
  pfn_hipHostMalloc hipHostMalloc;
  pfn_hipHostFree hipHostFree;
  pfn_hipMemGetInfo hipMemGetInfo;

  // Memory Copy
  pfn_hipMemcpy hipMemcpy;
  pfn_hipMemcpyAsync hipMemcpyAsync;
  pfn_hipMemset hipMemset;
  pfn_hipMemsetAsync hipMemsetAsync;

  // Stream Management
  pfn_hipStreamCreate hipStreamCreate;
  pfn_hipStreamCreateWithFlags hipStreamCreateWithFlags;
  pfn_hipStreamDestroy hipStreamDestroy;
  pfn_hipStreamSynchronize hipStreamSynchronize;
  pfn_hipStreamQuery hipStreamQuery;
  pfn_hipStreamWaitEvent hipStreamWaitEvent;

  // Event Management
  pfn_hipEventCreate hipEventCreate;
  pfn_hipEventCreateWithFlags hipEventCreateWithFlags;
  pfn_hipEventDestroy hipEventDestroy;
  pfn_hipEventRecord hipEventRecord;
  pfn_hipEventSynchronize hipEventSynchronize;
  pfn_hipEventQuery hipEventQuery;
  pfn_hipEventElapsedTime hipEventElapsedTime;

  // Module Management
  pfn_hipModuleLoad hipModuleLoad;
  pfn_hipModuleLoadData hipModuleLoadData;
  pfn_hipModuleUnload hipModuleUnload;
  pfn_hipModuleGetFunction hipModuleGetFunction;
  pfn_hipModuleGetGlobal hipModuleGetGlobal;
  pfn_hipModuleLaunchKernel hipModuleLaunchKernel;
  pfn_hipLaunchKernel hipLaunchKernel;
  pfn_hipExtModuleLaunchKernel hipExtModuleLaunchKernel;

  // Fat Binary Registration
  pfn___hipRegisterFatBinary __hipRegisterFatBinary;
  pfn___hipUnregisterFatBinary __hipUnregisterFatBinary;
  pfn___hipRegisterFunction __hipRegisterFunction;
  pfn___hipRegisterVar __hipRegisterVar;

  // Error Handling
  pfn_hipGetErrorString hipGetErrorString;
  pfn_hipGetErrorName hipGetErrorName;
  pfn_hipGetLastError hipGetLastError;
  pfn_hipPeekAtLastError hipPeekAtLastError;
};

//===----------------------------------------------------------------------===//
// Interceptor Interface
//===----------------------------------------------------------------------===//

// Function signature for the interceptor initialization function.
// The interceptor library must export this function.
// Returns a table of wrapper functions, or NULL to use passthrough.
typedef hip_function_table_t *(*pfn_hip_interceptor_init)(
    hip_function_table_t *real_functions);

// Function signature for interceptor cleanup.
typedef void (*pfn_hip_interceptor_shutdown)(void);

// Log callback signature for pass-through function logging.
// The interceptor can optionally export hip_interceptor_get_log_fn()
// to provide a log function for functions that bypass the function table.
typedef void (*pfn_hip_log_fn)(int level, const char *fmt, ...);
typedef pfn_hip_log_fn (*pfn_hip_interceptor_get_log_fn)(void);

//===----------------------------------------------------------------------===//
// Passthrough API
//===----------------------------------------------------------------------===//

// Get the real HIP function table (loaded from the backend library).
hip_function_table_t *hip_passthrough_get_real_table(void);

// Get the active function table (may be interceptor or real).
hip_function_table_t *hip_passthrough_get_active_table(void);

#ifdef __cplusplus
}
#endif

#endif // IREE_PASSTHROUGH_HIP_FUNCTION_TABLE_H_
