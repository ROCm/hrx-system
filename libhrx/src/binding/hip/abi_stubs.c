// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loader compatibility stubs for HIP ABI entry points HRX does not implement.
// These keep binaries linked against upstream libamdhip64 loadable while
// preserving a loud unsupported result if one of these paths is executed.

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "libhrx/src/binding/hip/api.h"

// Local compatibility declarations for unsupported ABI entries. These symbols
// only need type-correct call boundaries here; their implementations below
// always return hipErrorNotSupported.
typedef const struct hipArray_st* hipArray_const_t;
typedef struct hipMipmappedArray_st* hipMipmappedArray_t;
typedef const struct hipMipmappedArray_st* hipMipmappedArray_const_t;
typedef struct __hip_texture* hipTextureObject_t;
typedef struct __hip_surface* hipSurfaceObject_t;
typedef struct textureReference textureReference;
typedef struct HIP_ARRAY_DESCRIPTOR HIP_ARRAY_DESCRIPTOR;
typedef struct HIP_ARRAY3D_DESCRIPTOR HIP_ARRAY3D_DESCRIPTOR;
typedef struct HIP_LAUNCH_CONFIG_st HIP_LAUNCH_CONFIG;
typedef struct HIP_MEMCPY3D HIP_MEMCPY3D;
typedef struct HIP_RESOURCE_DESC HIP_RESOURCE_DESC;
typedef struct HIP_RESOURCE_VIEW_DESC HIP_RESOURCE_VIEW_DESC;
typedef struct HIP_TEXTURE_DESC HIP_TEXTURE_DESC;
typedef struct hipArrayMemoryRequirements hipArrayMemoryRequirements;
typedef struct hipDeviceProp_tR0000 hipDeviceProp_tR0000;
typedef hipDeviceProp_t hipDeviceProp_tR0600;
typedef struct ihipDevResourceDesc_t* hipDevResourceDesc_t;
typedef struct ihipExecutionCtx_t* hipExecutionCtx_t;
typedef void* hipExternalMemory_t;
typedef struct hipExternalMemoryBufferDesc_st hipExternalMemoryBufferDesc;
typedef struct hipExternalMemoryHandleDesc_st hipExternalMemoryHandleDesc;
typedef struct hipExternalMemoryMipmappedArrayDesc_st
    hipExternalMemoryMipmappedArrayDesc;
typedef void* hipExternalSemaphore_t;
typedef struct hipExternalSemaphoreHandleDesc_st hipExternalSemaphoreHandleDesc;
typedef struct hipExternalSemaphoreSignalParams_st
    hipExternalSemaphoreSignalParams;
typedef struct hipExternalSemaphoreWaitParams_st hipExternalSemaphoreWaitParams;
typedef struct hipExternalSemaphoreSignalNodeParams
    hipExternalSemaphoreSignalNodeParams;
typedef struct hipExternalSemaphoreWaitNodeParams
    hipExternalSemaphoreWaitNodeParams;
typedef struct hipFunctionLaunchParams_t hipFunctionLaunchParams;
typedef struct hipGraphicsResource hipGraphicsResource;
typedef hipGraphicsResource* hipGraphicsResource_t;
typedef struct hipKernel_st* hipKernel_t;
typedef struct hipLaunchConfig_st hipLaunchConfig_t;
typedef struct hipLaunchParams_t hipLaunchParams;
typedef struct hipLibrary_st* hipLibrary_t;
typedef struct ihipLinkState_t* hipLinkState_t;
typedef struct hipMemAllocNodeParams hipMemAllocNodeParams;
typedef struct hipMemcpy3DBatchOp hipMemcpy3DBatchOp;
typedef struct hipMemcpy3DPeerParms hipMemcpy3DPeerParms;
typedef struct hipMemcpyAttributes hipMemcpyAttributes;
typedef struct hipResourceDesc hipResourceDesc;
typedef struct hipResourceViewDesc hipResourceViewDesc;
typedef struct hipStreamBatchMemOpParams hipStreamBatchMemOpParams;
typedef struct hipTextureDesc hipTextureDesc;
typedef struct hip_Memcpy2D hip_Memcpy2D;
typedef struct hipDevResource_st hipDevResource;
typedef struct hipDevSmResourceGroupParams_st hipDevSmResourceGroupParams;
typedef int hipArray_Format;
typedef int hipDevResourceType;
typedef int hipDriverEntryPointQueryResult;
typedef int hipFunction_attribute;
typedef int hipGraphMemAttributeType;
typedef int hipJitInputType;
typedef int hipKernelNodeAttrID;
typedef int hipKernelNodeAttrValue;
typedef int hipLibraryOption;
typedef int hipMemRangeHandleType;
typedef int hipMemoryAdvise;
typedef int hipStreamAttrID;
typedef int hipStreamAttrValue;
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status,
                                    void* user_data);
enum hipTextureAddressMode {
  hipAddressModeWrap = 0,
  hipAddressModeClamp = 1,
  hipAddressModeMirror = 2,
  hipAddressModeBorder = 3,
};
enum hipTextureFilterMode {
  hipFilterModePoint = 0,
  hipFilterModeLinear = 1,
};

static _Thread_local hipStream_t hrx_hip_spt_stream = NULL;

static hipError_t hrx_hip_spt_default_stream(hipStream_t* stream) {
  if (!stream) return hipErrorInvalidValue;
  if (!hrx_hip_spt_stream) {
    hipError_t result =
        hipStreamCreateWithFlags(&hrx_hip_spt_stream, hipStreamNonBlocking);
    if (result != hipSuccess) return result;
  }
  *stream = hrx_hip_spt_stream;
  return hipSuccess;
}

static hipError_t hrx_hip_spt_stream_or_explicit(hipStream_t stream,
                                                 hipStream_t* resolved_stream) {
  if (!resolved_stream) return hipErrorInvalidValue;
  if (stream && stream != hipStreamPerThread) {
    *resolved_stream = stream;
    return hipSuccess;
  }
  return hrx_hip_spt_default_stream(resolved_stream);
}

static hipError_t hrx_hip_spt_lookup(const char* symbol, void** function,
                                     void* symbol_status) {
  if (!symbol || !function) return hipErrorInvalidValue;

  void* process = dlopen(NULL, RTLD_LAZY);
  if (!process) {
    *function = NULL;
    if (symbol_status) *(int*)symbol_status = 1;
    return hipErrorSharedObjectInitFailed;
  }

  char stream_per_thread_symbol[256];
  void* found = NULL;
  size_t symbol_length = strlen(symbol);
  if (symbol_length < sizeof(stream_per_thread_symbol) - 4 &&
      (symbol_length < 4 || strcmp(symbol + symbol_length - 4, "_spt") != 0)) {
    snprintf(stream_per_thread_symbol, sizeof(stream_per_thread_symbol),
             "%s_spt", symbol);
    found = dlsym(process, stream_per_thread_symbol);
  }
  if (!found) found = dlsym(process, symbol);

  *function = found;
  if (symbol_status) *(int*)symbol_status = found ? 0 : 1;
  return found ? hipSuccess : hipErrorNotFound;
}

HIPAPI const char* hipApiName(uint32_t id) {
  (void)id;
  return NULL;
}

HIPAPI hipError_t hipArray3DCreate(
    hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  (void)array;
  (void)pAllocateArray;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipArray3DGetDescriptor(
    HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array) {
  (void)pArrayDescriptor;
  (void)array;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipArrayCreate(hipArray_t* pHandle,
                                 const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  (void)pHandle;
  (void)pAllocateArray;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipArrayDestroy(hipArray_t array) {
  (void)array;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                        hipArray_t array) {
  (void)pArrayDescriptor;
  (void)array;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent,
                                  unsigned int* flags, hipArray_t array) {
  (void)desc;
  (void)extent;
  (void)flags;
  (void)array;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipBindTexture(size_t* offset, const textureReference* tex,
                                 const void* devPtr,
                                 const hipChannelFormatDesc* desc,
                                 size_t size) {
  (void)offset;
  (void)tex;
  (void)devPtr;
  (void)desc;
  (void)size;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipBindTexture2D(size_t* offset, const textureReference* tex,
                                   const void* devPtr,
                                   const hipChannelFormatDesc* desc,
                                   size_t width, size_t height, size_t pitch) {
  (void)offset;
  (void)tex;
  (void)devPtr;
  (void)desc;
  (void)width;
  (void)height;
  (void)pitch;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipBindTextureToArray(const textureReference* tex,
                                        hipArray_const_t array,
                                        const hipChannelFormatDesc* desc) {
  (void)tex;
  (void)array;
  (void)desc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipBindTextureToMipmappedArray(
    const textureReference* tex, hipMipmappedArray_const_t mipmappedArray,
    const hipChannelFormatDesc* desc) {
  (void)tex;
  (void)mipmappedArray;
  (void)desc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipChooseDeviceR0000(int* device,
                                       const hipDeviceProp_tR0000* properties) {
  (void)device;
  (void)properties;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipChooseDeviceR0600(int* device,
                                       const hipDeviceProp_tR0600* properties) {
  (void)device;
  (void)properties;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim,
                                   size_t sharedMem, hipStream_t stream) {
  (void)gridDim;
  (void)blockDim;
  (void)sharedMem;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                         const hipResourceDesc* pResDesc) {
  (void)pSurfObject;
  (void)pResDesc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCreateTextureObject(
    hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
    const hipTextureDesc* pTexDesc,
    const struct hipResourceViewDesc* pResViewDesc) {
  (void)pTexObject;
  (void)pResDesc;
  (void)pTexDesc;
  (void)pResViewDesc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int* apiVersion) {
  (void)ctx;
  (void)apiVersion;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
  (void)cacheConfig;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCtxGetFlags(unsigned int* flags) {
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  (void)pConfig;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  (void)cacheConfig;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  (void)config;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
  (void)surfaceObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  (void)textureObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem) {
  (void)extMem;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem) {
  (void)extSem;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDevResourceGenerateDesc(hipDevResourceDesc_t* phDesc,
                                             hipDevResource* resources,
                                             unsigned int nbResources) {
  (void)phDesc;
  (void)resources;
  (void)nbResources;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDevSmResourceSplit(
    hipDevResource* result, unsigned int nbGroups, const hipDevResource* input,
    hipDevResource* remainder, unsigned int flags,
    hipDevSmResourceGroupParams* groupParams) {
  (void)result;
  (void)nbGroups;
  (void)input;
  (void)remainder;
  (void)flags;
  (void)groupParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDevSmResourceSplitByCount(
    hipDevResource* result, unsigned int* nbGroups, const hipDevResource* input,
    hipDevResource* remainder, unsigned int flags, unsigned int minCount) {
  (void)result;
  (void)nbGroups;
  (void)input;
  (void)remainder;
  (void)flags;
  (void)minCount;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDeviceComputeCapability(int* major, int* minor,
                                             hipDevice_t device) {
  (void)major;
  (void)minor;
  (void)device;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDeviceGetDevResource(hipDevice_t device,
                                          hipDevResource* resource,
                                          hipDevResourceType type) {
  (void)device;
  (void)resource;
  (void)type;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDeviceGetExecutionCtx(hipExecutionCtx_t* ctx,
                                           hipDevice_t device) {
  (void)ctx;
  (void)device;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDeviceGetTexture1DLinearMaxWidth(
    size_t* maxWidthInElements, const hipChannelFormatDesc* fmtDesc,
    int device) {
  (void)maxWidthInElements;
  (void)fmtDesc;
  (void)device;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDeviceSetGraphMemAttribute(int device,
                                                hipGraphMemAttributeType attr,
                                                void* value) {
  (void)device;
  (void)attr;
  (void)value;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode,
                                           hipGraph_t hGraph,
                                           const hipGraphNode_t* dependencies,
                                           size_t numDependencies,
                                           const hipMemsetParams* memsetParams,
                                           hipCtx_t ctx) {
  (void)phGraphNode;
  (void)hGraph;
  (void)dependencies;
  (void)numDependencies;
  (void)memsetParams;
  (void)ctx;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvGraphExecMemsetNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  (void)hGraphExec;
  (void)hNode;
  (void)memsetParams;
  (void)ctx;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode,
                                                 HIP_MEMCPY3D* nodeParams) {
  (void)hNode;
  (void)nodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvGraphMemcpyNodeSetParams(
    hipGraphNode_t hNode, const HIP_MEMCPY3D* nodeParams) {
  (void)hNode;
  (void)nodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config,
                                       hipFunction_t f, void** params,
                                       void** extra) {
  (void)config;
  (void)f;
  (void)params;
  (void)extra;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
  (void)pCopy;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
  (void)pCopy;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy,
                                      hipStream_t stream) {
  (void)pCopy;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream,
                                          unsigned int flags) {
  (void)event;
  (void)stream;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxDestroy(hipExecutionCtx_t ctx) {
  (void)ctx;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxGetDevResource(hipExecutionCtx_t ctx,
                                                hipDevResource* resource,
                                                hipDevResourceType type) {
  (void)ctx;
  (void)resource;
  (void)type;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxGetDevice(hipDevice_t* device,
                                           hipExecutionCtx_t ctx) {
  (void)device;
  (void)ctx;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxGetId(hipExecutionCtx_t ctx,
                                       unsigned long long* ctxId) {
  (void)ctx;
  (void)ctxId;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxRecordEvent(hipExecutionCtx_t ctx,
                                             hipEvent_t event) {
  (void)ctx;
  (void)event;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxStreamCreate(hipStream_t* stream,
                                              hipExecutionCtx_t greenctx,
                                              unsigned int flags,
                                              int priority) {
  (void)stream;
  (void)greenctx;
  (void)flags;
  (void)priority;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxWaitEvent(hipExecutionCtx_t ctx,
                                           hipEvent_t event) {
  (void)ctx;
  (void)event;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExecutionCtxSynchronize(hipExecutionCtx_t ctx) {
  (void)ctx;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExternalMemoryGetMappedBuffer(
    void** devPtr, hipExternalMemory_t extMem,
    const hipExternalMemoryBufferDesc* bufferDesc) {
  (void)devPtr;
  (void)extMem;
  (void)bufferDesc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExternalMemoryGetMappedMipmappedArray(
    hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem,
    const hipExternalMemoryMipmappedArrayDesc* mipmapDesc) {
  (void)mipmap;
  (void)extMem;
  (void)mipmapDesc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExtDisableLogging(void) { return hipErrorNotSupported; }

HIPAPI hipError_t hipExtEnableLogging(void) { return hipErrorNotSupported; }

HIPAPI hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2,
                                               uint32_t* linktype,
                                               uint32_t* hopcount) {
  (void)device1;
  (void)device2;
  (void)linktype;
  (void)hopcount;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExtLaunchMultiKernelMultiDevice(
    hipLaunchParams* launchParamsList, int numDevices, unsigned int flags) {
  (void)launchParamsList;
  (void)numDevices;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size,
                                         size_t log_mask) {
  (void)log_level;
  (void)log_size;
  (void)log_mask;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  (void)mipmappedArray;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc,
                                    hipArray_const_t array) {
  (void)desc;
  (void)array;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop,
                                              int device) {
  (void)prop;
  (void)device;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetDriverEntryPoint(
    const char* symbol, void** funcPtr, unsigned long long flags,
    hipDriverEntryPointQueryResult* status) {
  (void)symbol;
  (void)funcPtr;
  (void)flags;
  (void)status;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetFuncBySymbol(hipFunction_t* functionPtr,
                                     const void* symbolPtr) {
  (void)functionPtr;
  (void)symbolPtr;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetMipmappedArrayLevel(
    hipArray_t* levelArray, hipMipmappedArray_const_t mipmappedArray,
    unsigned int level) {
  (void)levelArray;
  (void)mipmappedArray;
  (void)level;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetTextureAlignmentOffset(size_t* offset,
                                               const textureReference* texref) {
  (void)offset;
  (void)texref;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetTextureObjectResourceDesc(
    hipResourceDesc* pResDesc, hipTextureObject_t textureObject) {
  (void)pResDesc;
  (void)textureObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t
hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
                                    hipTextureObject_t textureObject) {
  (void)pResViewDesc;
  (void)textureObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetTextureObjectTextureDesc(
    hipTextureDesc* pTexDesc, hipTextureObject_t textureObject) {
  (void)pTexDesc;
  (void)textureObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGetTextureReference(const textureReference** texref,
                                         const void* symbol) {
  (void)texref;
  (void)symbol;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,
                                                      hipGraphNode_t node,
                                                      hipGraph_t childGraph) {
  (void)hGraphExec;
  (void)node;
  (void)childGraph;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,
                                                    hipGraphNode_t hNode,
                                                    hipEvent_t event) {
  (void)hGraphExec;
  (void)hNode;
  (void)event;
  return hipErrorNotSupported;
}

HIPAPI hipError_t
hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                const hipKernelNodeParams* pNodeParams) {
  (void)hGraphExec;
  (void)node;
  (void)pNodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node,
                                            hipHostNodeParams* pNodeParams) {
  (void)node;
  (void)pNodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc,
                                                   hipGraphNode_t hDst) {
  (void)hSrc;
  (void)hDst;
  return hipErrorNotSupported;
}

HIPAPI hipError_t
hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                               hipKernelNodeAttrValue* value) {
  (void)hNode;
  (void)attr;
  (void)value;
  return hipErrorNotSupported;
}

HIPAPI hipError_t
hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                               const hipKernelNodeAttrValue* value) {
  (void)hNode;
  (void)attr;
  (void)value;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphMemAllocNodeGetParams(
    hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams) {
  (void)node;
  (void)pNodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphMemcpyNodeSetParams(
    hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
  (void)node;
  (void)pNodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out) {
  (void)hNode;
  (void)params_out;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
    hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  (void)hNode;
  (void)nodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out) {
  (void)hNode;
  (void)params_out;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
    hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  (void)hNode;
  (void)nodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  (void)hGraphExec;
  (void)hNode;
  (void)nodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  (void)hGraphExec;
  (void)hNode;
  (void)nodeParams;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec,
                                         hipGraphNode_t hNode,
                                         unsigned int* isEnabled) {
  (void)hGraphExec;
  (void)hNode;
  (void)isEnabled;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec,
                                         hipGraphNode_t hNode,
                                         unsigned int isEnabled) {
  (void)hGraphExec;
  (void)hNode;
  (void)isEnabled;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  (void)graphExec;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGreenCtxCreate(hipExecutionCtx_t* ctx,
                                    hipDevResourceDesc_t desc, int device,
                                    unsigned int flags) {
  (void)ctx;
  (void)desc;
  (void)device;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipHccModuleLaunchKernel(
    hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY,
    uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY,
    uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream,
    void** kernelParams, void** extra, hipEvent_t startEvent,
    hipEvent_t stopEvent) {
  (void)f;
  (void)globalWorkSizeX;
  (void)globalWorkSizeY;
  (void)globalWorkSizeZ;
  (void)localWorkSizeX;
  (void)localWorkSizeY;
  (void)localWorkSizeZ;
  (void)sharedMemBytes;
  (void)hStream;
  (void)kernelParams;
  (void)extra;
  (void)startEvent;
  (void)stopEvent;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphicsMapResources(int count,
                                          hipGraphicsResource_t* resources,
                                          hipStream_t stream) {
  (void)count;
  (void)resources;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphicsResourceGetMappedPointer(
    void** devPtr, size_t* size, hipGraphicsResource_t resource) {
  (void)devPtr;
  (void)size;
  (void)resource;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphicsSubResourceGetMappedArray(
    hipArray_t* array, hipGraphicsResource_t resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  (void)array;
  (void)resource;
  (void)arrayIndex;
  (void)mipLevel;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipGraphicsUnmapResources(int count,
                                            hipGraphicsResource_t* resources,
                                            hipStream_t stream) {
  (void)count;
  (void)resources;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t
hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
  (void)resource;
  return hipErrorNotSupported;
}

HIPAPI hipError_t
hipImportExternalMemory(hipExternalMemory_t* extMem_out,
                        const hipExternalMemoryHandleDesc* memHandleDesc) {
  (void)extMem_out;
  (void)memHandleDesc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipImportExternalSemaphore(
    hipExternalSemaphore_t* extSem_out,
    const hipExternalSemaphoreHandleDesc* semHandleDesc) {
  (void)extSem_out;
  (void)semHandleDesc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipKernelGetAttribute(int* pi, hipFunction_attribute attrib,
                                        hipKernel_t kernel, hipDevice_t dev) {
  (void)pi;
  (void)attrib;
  (void)kernel;
  (void)dev;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipKernelGetFunction(hipFunction_t* pFunc,
                                       hipKernel_t kernel) {
  (void)pFunc;
  (void)kernel;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipKernelGetLibrary(hipLibrary_t* library,
                                      hipKernel_t kernel) {
  (void)library;
  (void)kernel;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipKernelGetName(const char** name, hipKernel_t kernel) {
  (void)name;
  (void)kernel;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipKernelGetParamInfo(hipKernel_t kernel, size_t paramIndex,
                                        size_t* paramOffset,
                                        size_t* paramSize) {
  (void)kernel;
  (void)paramIndex;
  (void)paramOffset;
  (void)paramSize;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipKernelSetAttribute(hipFunction_attribute attrib, int value,
                                        hipKernel_t kernel, hipDevice_t dev) {
  (void)attrib;
  (void)value;
  (void)kernel;
  (void)dev;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLaunchByPtr(const void* func) {
  (void)func;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim,
                                             dim3 blockDimX,
                                             void** kernelParams,
                                             unsigned int sharedMemBytes,
                                             hipStream_t stream) {
  (void)f;
  (void)gridDim;
  (void)blockDimX;
  (void)kernelParams;
  (void)sharedMemBytes;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLaunchCooperativeKernelMultiDevice(
    hipLaunchParams* launchParamsList, int numDevices, unsigned int flags) {
  (void)launchParamsList;
  (void)numDevices;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config,
                                     const void* fPtr, void** args) {
  (void)config;
  (void)fPtr;
  (void)args;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryEnumerateKernels(hipKernel_t* kernels,
                                             unsigned int numKernels,
                                             hipLibrary_t library) {
  (void)kernels;
  (void)numKernels;
  (void)library;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryGetGlobal(void** dptr, size_t* bytes,
                                      hipLibrary_t library, const char* name) {
  (void)dptr;
  (void)bytes;
  (void)library;
  (void)name;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryGetKernel(hipKernel_t* pKernel,
                                      hipLibrary_t library, const char* name) {
  (void)pKernel;
  (void)library;
  (void)name;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryGetKernelCount(unsigned int* count,
                                           hipLibrary_t library) {
  (void)count;
  (void)library;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryGetManaged(void** dptr, size_t* bytes,
                                       hipLibrary_t library, const char* name) {
  (void)dptr;
  (void)bytes;
  (void)library;
  (void)name;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryLoadData(hipLibrary_t* library, const void* code,
                                     hipJitOption* jitOptions,
                                     void** jitOptionsValues,
                                     unsigned int numJitOptions,
                                     hipLibraryOption* libraryOptions,
                                     void** libraryOptionValues,
                                     unsigned int numLibraryOptions) {
  (void)library;
  (void)code;
  (void)jitOptions;
  (void)jitOptionsValues;
  (void)numJitOptions;
  (void)libraryOptions;
  (void)libraryOptionValues;
  (void)numLibraryOptions;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryLoadFromFile(
    hipLibrary_t* library, const char* fileName, hipJitOption* jitOptions,
    void** jitOptionsValues, unsigned int numJitOptions,
    hipLibraryOption* libraryOptions, void** libraryOptionValues,
    unsigned int numLibraryOptions) {
  (void)library;
  (void)fileName;
  (void)jitOptions;
  (void)jitOptionsValues;
  (void)numJitOptions;
  (void)libraryOptions;
  (void)libraryOptionValues;
  (void)numLibraryOptions;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLibraryUnload(hipLibrary_t library) {
  (void)library;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLinkAddData(hipLinkState_t state, hipJitInputType type,
                                 void* data, size_t size, const char* name,
                                 unsigned int numOptions, hipJitOption* options,
                                 void** optionValues) {
  (void)state;
  (void)type;
  (void)data;
  (void)size;
  (void)name;
  (void)numOptions;
  (void)options;
  (void)optionValues;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLinkAddFile(hipLinkState_t state, hipJitInputType type,
                                 const char* path, unsigned int numOptions,
                                 hipJitOption* options, void** optionValues) {
  (void)state;
  (void)type;
  (void)path;
  (void)numOptions;
  (void)options;
  (void)optionValues;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLinkComplete(hipLinkState_t state, void** hipBinOut,
                                  size_t* sizeOut) {
  (void)state;
  (void)hipBinOut;
  (void)sizeOut;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLinkCreate(unsigned int numOptions, hipJitOption* options,
                                void** optionValues, hipLinkState_t* stateOut) {
  (void)numOptions;
  (void)options;
  (void)optionValues;
  (void)stateOut;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipLinkDestroy(hipLinkState_t state) {
  (void)state;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMallocArray(hipArray_t* array,
                                 const hipChannelFormatDesc* desc, size_t width,
                                 size_t height, unsigned int flags) {
  (void)array;
  (void)desc;
  (void)width;
  (void)height;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMallocMipmappedArray(
    hipMipmappedArray_t* mipmappedArray,
    const struct hipChannelFormatDesc* desc, struct hipExtent extent,
    unsigned int numLevels, unsigned int flags) {
  (void)mipmappedArray;
  (void)desc;
  (void)extent;
  (void)numLevels;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemAdvise_v2(const void* dev_ptr, size_t count,
                                  hipMemoryAdvise advice,
                                  hipMemLocation device) {
  (void)dev_ptr;
  (void)count;
  (void)advice;
  (void)device;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemAllocHost(void** ptr, size_t size) {
  (void)ptr;
  (void)size;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch,
                                   size_t widthInBytes, size_t height,
                                   unsigned int elementSizeBytes) {
  (void)dptr;
  (void)pitch;
  (void)widthInBytes;
  (void)height;
  (void)elementSizeBytes;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemGetHandleForAddressRange(
    void* handle, hipDeviceptr_t dptr, size_t size,
    hipMemRangeHandleType handleType, unsigned long long flags) {
  (void)handle;
  (void)dptr;
  (void)size;
  (void)handleType;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemGetMemPool(hipMemPool_t* pool, hipMemLocation* location,
                                   hipMemAllocationType type) {
  (void)pool;
  (void)location;
  (void)type;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count,
                                         hipMemLocation location,
                                         unsigned int flags,
                                         hipStream_t stream) {
  (void)dev_ptr;
  (void)count;
  (void)location;
  (void)flags;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemPrefetchBatchAsync(
    void** dev_ptrs, size_t* sizes, size_t count, hipMemLocation* prefetch_locs,
    size_t* prefetch_loc_idxs, size_t num_prefetch_locs,
    unsigned long long flags, hipStream_t stream) {
  (void)dev_ptrs;
  (void)sizes;
  (void)count;
  (void)prefetch_locs;
  (void)prefetch_loc_idxs;
  (void)num_prefetch_locs;
  (void)flags;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemSetMemPool(hipMemLocation* location,
                                   hipMemAllocationType type,
                                   hipMemPool_t pool) {
  (void)location;
  (void)type;
  (void)pool;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst,
                                          size_t hOffsetDst,
                                          hipArray_const_t src,
                                          size_t wOffsetSrc, size_t hOffsetSrc,
                                          size_t width, size_t height,
                                          hipMemcpyKind kind) {
  (void)dst;
  (void)wOffsetDst;
  (void)hOffsetDst;
  (void)src;
  (void)wOffsetSrc;
  (void)hOffsetSrc;
  (void)width;
  (void)height;
  (void)kind;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch,
                                       hipArray_const_t src, size_t wOffset,
                                       size_t hOffset, size_t width,
                                       size_t height, hipMemcpyKind kind) {
  (void)dst;
  (void)dpitch;
  (void)src;
  (void)wOffset;
  (void)hOffset;
  (void)width;
  (void)height;
  (void)kind;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch,
                                            hipArray_const_t src,
                                            size_t wOffset, size_t hOffset,
                                            size_t width, size_t height,
                                            hipMemcpyKind kind,
                                            hipStream_t stream) {
  (void)dst;
  (void)dpitch;
  (void)src;
  (void)wOffset;
  (void)hOffset;
  (void)width;
  (void)height;
  (void)kind;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset,
                                     size_t hOffset, const void* src,
                                     size_t spitch, size_t width, size_t height,
                                     hipMemcpyKind kind) {
  (void)dst;
  (void)wOffset;
  (void)hOffset;
  (void)src;
  (void)spitch;
  (void)width;
  (void)height;
  (void)kind;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset,
                                          size_t hOffset, const void* src,
                                          size_t spitch, size_t width,
                                          size_t height, hipMemcpyKind kind,
                                          hipStream_t stream) {
  (void)dst;
  (void)wOffset;
  (void)hOffset;
  (void)src;
  (void)spitch;
  (void)width;
  (void)height;
  (void)kind;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p,
                                   hipStream_t stream) {
  (void)p;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy3DBatchAsync(size_t numOps,
                                        struct hipMemcpy3DBatchOp* opList,
                                        size_t* failIdx,
                                        unsigned long long flags,
                                        hipStream_t stream) {
  (void)numOps;
  (void)opList;
  (void)failIdx;
  (void)flags;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy3DPeer(hipMemcpy3DPeerParms* p) {
  (void)p;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpy3DPeerAsync(hipMemcpy3DPeerParms* p,
                                       hipStream_t stream) {
  (void)p;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset,
                                hipArray_t srcArray, size_t srcOffset,
                                size_t ByteCount) {
  (void)dstArray;
  (void)dstOffset;
  (void)srcArray;
  (void)srcOffset;
  (void)ByteCount;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray,
                                size_t srcOffset, size_t ByteCount) {
  (void)dstDevice;
  (void)srcArray;
  (void)srcOffset;
  (void)ByteCount;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray,
                                size_t srcOffset, size_t count) {
  (void)dst;
  (void)srcArray;
  (void)srcOffset;
  (void)count;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray,
                                     size_t srcOffset, size_t ByteCount,
                                     hipStream_t stream) {
  (void)dstHost;
  (void)srcArray;
  (void)srcOffset;
  (void)ByteCount;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes,
                                      size_t count, hipMemcpyAttributes* attrs,
                                      size_t* attrsIdxs, size_t numAttrs,
                                      size_t* failIdx, hipStream_t stream) {
  (void)dsts;
  (void)srcs;
  (void)sizes;
  (void)count;
  (void)attrs;
  (void)attrsIdxs;
  (void)numAttrs;
  (void)failIdx;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset,
                                hipDeviceptr_t srcDevice, size_t ByteCount) {
  (void)dstArray;
  (void)dstOffset;
  (void)srcDevice;
  (void)ByteCount;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray,
                                     size_t wOffset, size_t hOffset,
                                     size_t count, hipMemcpyKind kind) {
  (void)dst;
  (void)srcArray;
  (void)wOffset;
  (void)hOffset;
  (void)count;
  (void)kind;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset,
                                const void* srcHost, size_t count) {
  (void)dstArray;
  (void)dstOffset;
  (void)srcHost;
  (void)count;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset,
                                     const void* srcHost, size_t ByteCount,
                                     hipStream_t stream) {
  (void)dstArray;
  (void)dstOffset;
  (void)srcHost;
  (void)ByteCount;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  (void)pCopy;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy,
                                        hipStream_t stream) {
  (void)pCopy;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset,
                                   size_t hOffset, const void* src,
                                   size_t count, hipMemcpyKind kind) {
  (void)dst;
  (void)wOffset;
  (void)hOffset;
  (void)src;
  (void)count;
  (void)kind;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value,
                              hipExtent extent) {
  (void)pitchedDevPtr;
  (void)value;
  (void)extent;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value,
                                   hipExtent extent, hipStream_t stream) {
  (void)pitchedDevPtr;
  (void)value;
  (void)extent;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch,
                                 unsigned short value, size_t width,
                                 size_t height) {
  (void)dst;
  (void)dstPitch;
  (void)value;
  (void)width;
  (void)height;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch,
                                      unsigned short value, size_t width,
                                      size_t height, hipStream_t stream) {
  (void)dst;
  (void)dstPitch;
  (void)value;
  (void)width;
  (void)height;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch,
                                 unsigned int value, size_t width,
                                 size_t height) {
  (void)dst;
  (void)dstPitch;
  (void)value;
  (void)width;
  (void)height;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch,
                                      unsigned int value, size_t width,
                                      size_t height, hipStream_t stream) {
  (void)dst;
  (void)dstPitch;
  (void)value;
  (void)width;
  (void)height;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch,
                                unsigned char value, size_t width,
                                size_t height) {
  (void)dst;
  (void)dstPitch;
  (void)value;
  (void)width;
  (void)height;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch,
                                     unsigned char value, size_t width,
                                     size_t height, hipStream_t stream) {
  (void)dst;
  (void)dstPitch;
  (void)value;
  (void)width;
  (void)height;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMipmappedArrayCreate(
    hipMipmappedArray_t* pHandle, HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
  (void)pHandle;
  (void)pMipmappedArrayDesc;
  (void)numMipmapLevels;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipMipmappedArrayGetMemoryRequirements(
    hipArrayMemoryRequirements* memoryRequirements, hipMipmappedArray_t mipmap,
    hipDevice_t device) {
  (void)memoryRequirements;
  (void)mipmap;
  (void)device;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipModuleGetFunctionCount(unsigned int* count,
                                            hipModule_t module) {
  (void)count;
  (void)module;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipModuleGetTexRef(textureReference** texRef,
                                     hipModule_t hmod, const char* name) {
  (void)texRef;
  (void)hmod;
  (void)name;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipModuleLaunchCooperativeKernelMultiDevice(
    hipFunctionLaunchParams* launchParamsList, unsigned int numDevices,
    unsigned int flags) {
  (void)launchParamsList;
  (void)numDevices;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipModuleLoadFatBinary(hipModule_t* module,
                                         const void* fatbin) {
  (void)module;
  (void)fatbin;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipOccupancyAvailableDynamicSMemPerBlock(
    size_t* dynamicSmemSize, const void* f, int numBlocks, int blockSize) {
  (void)dynamicSmemSize;
  (void)f;
  (void)numBlocks;
  (void)blockSize;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipProfilerStart(void) { return hipErrorNotSupported; }

HIPAPI hipError_t hipProfilerStop(void) { return hipErrorNotSupported; }

HIPAPI hipError_t hipSetValidDevices(int* device_arr, int len) {
  (void)device_arr;
  (void)len;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipSetupArgument(const void* arg, size_t size,
                                   size_t offset) {
  (void)arg;
  (void)size;
  (void)offset;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipSignalExternalSemaphoresAsync(
    const hipExternalSemaphore_t* extSemArray,
    const hipExternalSemaphoreSignalParams* paramsArray,
    unsigned int numExtSems, hipStream_t stream) {
  (void)extSemArray;
  (void)paramsArray;
  (void)numExtSems;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipStreamAddCallback(hipStream_t stream,
                                       hipStreamCallback_t callback,
                                       void* userData, unsigned int flags) {
  (void)stream;
  (void)callback;
  (void)userData;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr,
                                          size_t length, unsigned int flags) {
  (void)stream;
  (void)dev_ptr;
  (void)length;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                                      hipStreamBatchMemOpParams* paramArray,
                                      unsigned int flags) {
  (void)stream;
  (void)count;
  (void)paramArray;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src) {
  (void)dst;
  (void)src;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipStreamGetAttribute(hipStream_t stream,
                                        hipStreamAttrID attr,
                                        hipStreamAttrValue* value_out) {
  (void)stream;
  (void)attr;
  (void)value_out;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipStreamGetDevResource(hipStream_t stream,
                                          hipDevResource* resource,
                                          hipDevResourceType type) {
  (void)stream;
  (void)resource;
  (void)type;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipStreamSetAttribute(hipStream_t stream,
                                        hipStreamAttrID attr,
                                        const hipStreamAttrValue* value) {
  (void)stream;
  (void)attr;
  (void)value;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexObjectCreate(
    hipTextureObject_t* pTexObject, const HIP_RESOURCE_DESC* pResDesc,
    const HIP_TEXTURE_DESC* pTexDesc,
    const HIP_RESOURCE_VIEW_DESC* pResViewDesc) {
  (void)pTexObject;
  (void)pResDesc;
  (void)pTexDesc;
  (void)pResViewDesc;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
  (void)texObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc,
                                              hipTextureObject_t texObject) {
  (void)pResDesc;
  (void)texObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexObjectGetResourceViewDesc(
    HIP_RESOURCE_VIEW_DESC* pResViewDesc, hipTextureObject_t texObject) {
  (void)pResViewDesc;
  (void)texObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc,
                                             hipTextureObject_t texObject) {
  (void)pTexDesc;
  (void)texObject;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr,
                                      const textureReference* texRef) {
  (void)dev_ptr;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode* pam,
                                          const textureReference* texRef,
                                          int dim) {
  (void)pam;
  (void)texRef;
  (void)dim;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetArray(hipArray_t* pArray,
                                    const textureReference* texRef) {
  (void)pArray;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetBorderColor(float* pBorderColor,
                                          const textureReference* texRef) {
  (void)pBorderColor;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode* pfm,
                                         const textureReference* texRef) {
  (void)pfm;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetFlags(unsigned int* pFlags,
                                    const textureReference* texRef) {
  (void)pFlags;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetFormat(hipArray_Format* pFormat,
                                     int* pNumChannels,
                                     const textureReference* texRef) {
  (void)pFormat;
  (void)pNumChannels;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio,
                                            const textureReference* texRef) {
  (void)pmaxAnsio;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,
                                             const textureReference* texRef) {
  (void)pArray;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode* pfm,
                                               const textureReference* texRef) {
  (void)pfm;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetMipmapLevelBias(float* pbias,
                                              const textureReference* texRef) {
  (void)pbias;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp,
                                               float* pmaxMipmapLevelClamp,
                                               const textureReference* texRef) {
  (void)pminMipmapLevelClamp;
  (void)pmaxMipmapLevelClamp;
  (void)texRef;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetAddress(size_t* ByteOffset,
                                      textureReference* texRef,
                                      hipDeviceptr_t dptr, size_t bytes) {
  (void)ByteOffset;
  (void)texRef;
  (void)dptr;
  (void)bytes;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetAddress2D(textureReference* texRef,
                                        const HIP_ARRAY_DESCRIPTOR* desc,
                                        hipDeviceptr_t dptr, size_t Pitch) {
  (void)texRef;
  (void)desc;
  (void)dptr;
  (void)Pitch;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetAddressMode(textureReference* texRef, int dim,
                                          enum hipTextureAddressMode am) {
  (void)texRef;
  (void)dim;
  (void)am;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetArray(textureReference* tex,
                                    hipArray_const_t array,
                                    unsigned int flags) {
  (void)tex;
  (void)array;
  (void)flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetBorderColor(textureReference* texRef,
                                          float* pBorderColor) {
  (void)texRef;
  (void)pBorderColor;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetFilterMode(textureReference* texRef,
                                         enum hipTextureFilterMode fm) {
  (void)texRef;
  (void)fm;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetFlags(textureReference* texRef,
                                    unsigned int Flags) {
  (void)texRef;
  (void)Flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetFormat(textureReference* texRef,
                                     hipArray_Format fmt,
                                     int NumPackedComponents) {
  (void)texRef;
  (void)fmt;
  (void)NumPackedComponents;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef,
                                            unsigned int maxAniso) {
  (void)texRef;
  (void)maxAniso;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef,
                                               enum hipTextureFilterMode fm) {
  (void)texRef;
  (void)fm;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef,
                                              float bias) {
  (void)texRef;
  (void)bias;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef,
                                               float minMipMapLevelClamp,
                                               float maxMipMapLevelClamp) {
  (void)texRef;
  (void)minMipMapLevelClamp;
  (void)maxMipMapLevelClamp;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipTexRefSetMipmappedArray(
    textureReference* texRef, struct hipMipmappedArray_st* mipmappedArray,
    unsigned int Flags) {
  (void)texRef;
  (void)mipmappedArray;
  (void)Flags;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipUnbindTexture(const textureReference* tex) {
  (void)tex;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipWaitExternalSemaphoresAsync(
    const hipExternalSemaphore_t* extSemArray,
    const hipExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems,
    hipStream_t stream) {
  (void)extSemArray;
  (void)paramsArray;
  (void)numExtSems;
  (void)stream;
  return hipErrorNotSupported;
}

HIPAPI hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipEventRecord(event, resolved_stream);
}

HIPAPI hipError_t hipGetDriverEntryPoint_spt(const char* symbol,
                                             void** function,
                                             unsigned long long flags,
                                             void* status) {
  (void)flags;
  return hrx_hip_spt_lookup(symbol, function, status);
}

HIPAPI hipError_t hipGetProcAddress_spt(const char* symbol, void** function,
                                        int hip_version, uint64_t flags,
                                        void* symbol_status) {
  (void)hip_version;
  (void)flags;
  return hrx_hip_spt_lookup(symbol, function, symbol_status);
}

HIPAPI hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim,
                                                 dim3 blockDim,
                                                 void** kernelParams,
                                                 uint32_t sharedMemBytes,
                                                 hipStream_t hStream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(hStream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipLaunchCooperativeKernel(f, gridDim, blockDim, kernelParams,
                                    sharedMemBytes, resolved_stream);
}

HIPAPI hipError_t hipLaunchKernel_spt(const void* function_address,
                                      dim3 num_blocks, dim3 dim_blocks,
                                      void** args, size_t shared_mem_bytes,
                                      hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipLaunchKernel(function_address, num_blocks, dim_blocks, args,
                         shared_mem_bytes, resolved_stream);
}

HIPAPI hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec,
                                     hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipGraphLaunch(graphExec, resolved_stream);
}

HIPAPI hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn,
                                        void* userData) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipLaunchHostFunc(resolved_stream, fn, userData);
}

HIPAPI hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch,
                                       const void* src, size_t spitch,
                                       size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
                          resolved_stream);
}

HIPAPI hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch,
                                           hipArray_const_t src, size_t wOffset,
                                           size_t hOffset, size_t width,
                                           size_t height, hipMemcpyKind kind) {
  return hipMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height,
                              kind);
}

HIPAPI hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset,
                                         size_t hOffset, const void* src,
                                         size_t spitch, size_t width,
                                         size_t height, hipMemcpyKind kind) {
  return hipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height,
                            kind);
}

HIPAPI hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src,
                                  size_t spitch, size_t width, size_t height,
                                  hipMemcpyKind kind) {
  hipStream_t stream = NULL;
  hipError_t result = hrx_hip_spt_default_stream(&stream);
  if (result != hipSuccess) return result;
  result =
      hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
  return result == hipSuccess ? hipStreamSynchronize(stream) : result;
}

HIPAPI hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p) {
  return hipMemcpy3D(p);
}

HIPAPI hipError_t hipMemcpy3DAsync_spt(const struct hipMemcpy3DParms* p,
                                       hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemcpy3DAsync(p, resolved_stream);
}

HIPAPI hipError_t hipMemcpyAsync_spt(void* dst, const void* src,
                                     size_t size_bytes, hipMemcpyKind kind,
                                     hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemcpyAsync(dst, src, size_bytes, kind, resolved_stream);
}

HIPAPI hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol,
                                          size_t size_bytes, size_t offset,
                                          hipMemcpyKind kind) {
  hipStream_t stream = NULL;
  hipError_t result = hrx_hip_spt_default_stream(&stream);
  if (result != hipSuccess) return result;
  result =
      hipMemcpyFromSymbolAsync(dst, symbol, size_bytes, offset, kind, stream);
  return result == hipSuccess ? hipStreamSynchronize(stream) : result;
}

HIPAPI hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src,
                                        size_t size_bytes, size_t offset,
                                        hipMemcpyKind kind) {
  hipStream_t stream = NULL;
  hipError_t result = hrx_hip_spt_default_stream(&stream);
  if (result != hipSuccess) return result;
  result =
      hipMemcpyToSymbolAsync(symbol, src, size_bytes, offset, kind, stream);
  return result == hipSuccess ? hipStreamSynchronize(stream) : result;
}

HIPAPI hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol,
                                             const void* src, size_t size_bytes,
                                             size_t offset, hipMemcpyKind kind,
                                             hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemcpyToSymbolAsync(symbol, src, size_bytes, offset, kind,
                                resolved_stream);
}

HIPAPI hipError_t hipMemcpy_spt(void* dst, const void* src, size_t size_bytes,
                                hipMemcpyKind kind) {
  hipStream_t stream = NULL;
  hipError_t result = hrx_hip_spt_default_stream(&stream);
  if (result != hipSuccess) return result;
  result = hipMemcpyAsync(dst, src, size_bytes, kind, stream);
  return result == hipSuccess ? hipStreamSynchronize(stream) : result;
}

HIPAPI hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value,
                                       size_t width, size_t height,
                                       hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemset2DAsync(dst, pitch, value, width, height, resolved_stream);
}

HIPAPI hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value,
                                  size_t width, size_t height) {
  hipStream_t stream = NULL;
  hipError_t result = hrx_hip_spt_default_stream(&stream);
  if (result != hipSuccess) return result;
  result = hipMemset2DAsync(dst, pitch, value, width, height, stream);
  return result == hipSuccess ? hipStreamSynchronize(stream) : result;
}

HIPAPI hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value,
                                       hipExtent extent, hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemset3DAsync(pitchedDevPtr, value, extent, resolved_stream);
}

HIPAPI hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value,
                                  hipExtent extent) {
  return hipMemset3D(pitchedDevPtr, value, extent);
}

HIPAPI hipError_t hipMemsetAsync_spt(void* dst, int value, size_t size_bytes,
                                     hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemsetAsync(dst, value, size_bytes, resolved_stream);
}

HIPAPI hipError_t hipMemset_spt(void* dst, int value, size_t size_bytes) {
  hipStream_t stream = NULL;
  hipError_t result = hrx_hip_spt_default_stream(&stream);
  if (result != hipSuccess) return result;
  result = hipMemsetAsync(dst, value, size_bytes, stream);
  return result == hipSuccess ? hipStreamSynchronize(stream) : result;
}

HIPAPI hipError_t hipStreamAddCallback_spt(hipStream_t stream,
                                           hipStreamCallback_t callback,
                                           void* userData, unsigned int flags) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamAddCallback(resolved_stream, callback, userData, flags);
}

HIPAPI hipError_t hipStreamBeginCapture_spt(hipStream_t stream,
                                            hipStreamCaptureMode mode) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamBeginCapture(resolved_stream, mode);
}

HIPAPI hipError_t hipStreamEndCapture_spt(hipStream_t stream,
                                          hipGraph_t* graph) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamEndCapture(resolved_stream, graph);
}

HIPAPI hipError_t hipStreamGetCaptureInfo_spt(
    hipStream_t stream, hipStreamCaptureStatus* capture_status,
    unsigned long long* id) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamGetCaptureInfo(resolved_stream, capture_status, id);
}

HIPAPI hipError_t hipStreamGetCaptureInfo_v2_spt(
    hipStream_t stream, hipStreamCaptureStatus* capture_status,
    unsigned long long* id, hipGraph_t* graph,
    const hipGraphNode_t** dependencies, size_t* dependency_count) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamGetCaptureInfo_v2(resolved_stream, capture_status, id, graph,
                                    dependencies, dependency_count);
}

HIPAPI hipError_t hipStreamGetFlags_spt(hipStream_t stream,
                                        unsigned int* flags) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamGetFlags(resolved_stream, flags);
}

HIPAPI hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamGetPriority(resolved_stream, priority);
}

HIPAPI hipError_t hipStreamIsCapturing_spt(
    hipStream_t stream, hipStreamCaptureStatus* capture_status) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamIsCapturing(resolved_stream, capture_status);
}

HIPAPI hipError_t hipStreamQuery_spt(hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamQuery(resolved_stream);
}

HIPAPI hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamSynchronize(resolved_stream);
}

HIPAPI hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event,
                                         unsigned int flags) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipStreamWaitEvent(resolved_stream, event, flags);
}
