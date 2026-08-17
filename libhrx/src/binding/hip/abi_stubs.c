// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loader compatibility stubs for HIP ABI entry points HRX does not implement.
// These keep binaries linked against upstream libamdhip64 loadable while
// preserving a loud unsupported result if one of these paths is executed.

#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
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
typedef struct HIP_RESOURCE_DESC HIP_RESOURCE_DESC;
typedef struct HIP_RESOURCE_VIEW_DESC HIP_RESOURCE_VIEW_DESC;
typedef struct HIP_TEXTURE_DESC HIP_TEXTURE_DESC;
typedef struct hipArrayMemoryRequirements {
  size_t alignment;
  size_t size;
} hipArrayMemoryRequirements;
typedef struct hipDeviceProp_tR0000 {
  // Device name.
  char name[256];
  // Global memory size in bytes.
  size_t totalGlobalMem;
  // Workgroup-local memory size in bytes.
  size_t sharedMemPerBlock;
  // Registers available per workgroup.
  int regsPerBlock;
  // Hardware wavefront size.
  int warpSize;
  // Maximum invocations per workgroup.
  int maxThreadsPerBlock;
  // Maximum workgroup dimensions.
  int maxThreadsDim[3];
  // Maximum grid dimensions.
  int maxGridSize[3];
  // Maximum core clock rate in kHz.
  int clockRate;
  // Maximum memory clock rate in kHz.
  int memoryClockRate;
  // Global memory bus width in bits.
  int memoryBusWidth;
  // Constant memory size in bytes.
  size_t totalConstMem;
  // Major compute capability.
  int major;
  // Minor compute capability.
  int minor;
  // Number of execution units.
  int multiProcessorCount;
  // L2 cache size in bytes.
  int l2CacheSize;
  // Maximum resident invocations per execution unit.
  int maxThreadsPerMultiProcessor;
  // Device compute mode.
  int computeMode;
  // Device clock-instruction rate in kHz.
  int clockInstructionRate;
  // Architectural feature flags.
  hipDeviceArch_t arch;
  // Whether concurrent kernel execution is supported.
  int concurrentKernels;
  // PCI domain identifier.
  int pciDomainID;
  // PCI bus identifier.
  int pciBusID;
  // PCI device identifier.
  int pciDeviceID;
  // Maximum shared memory per execution unit in bytes.
  size_t maxSharedMemoryPerMultiProcessor;
  // Whether this device belongs to a multi-GPU board.
  int isMultiGpuBoard;
  // Whether host memory can be mapped.
  int canMapHostMemory;
  // Deprecated numeric architecture identifier.
  int gcnArch;
  // Architecture target identifier.
  char gcnArchName[256];
  // Whether the device shares memory with the host.
  int integrated;
  // Whether cooperative launch is supported.
  int cooperativeLaunch;
  // Whether multi-device cooperative launch is supported.
  int cooperativeMultiDeviceLaunch;
  // Maximum linear one-dimensional texture size.
  int maxTexture1DLinear;
  // Maximum one-dimensional texture size.
  int maxTexture1D;
  // Maximum two-dimensional texture dimensions.
  int maxTexture2D[2];
  // Maximum three-dimensional texture dimensions.
  int maxTexture3D[3];
  // HDP memory-flush register address, when available.
  unsigned int* hdpMemFlushCntl;
  // HDP register-flush address, when available.
  unsigned int* hdpRegFlushCntl;
  // Maximum memory-copy pitch in bytes.
  size_t memPitch;
  // Texture base-address alignment in bytes.
  size_t textureAlignment;
  // Texture pitch alignment in bytes.
  size_t texturePitchAlignment;
  // Whether kernels have an execution timeout.
  int kernelExecTimeoutEnabled;
  // Whether error-correcting memory is enabled.
  int ECCEnabled;
  // Whether the device uses a compute-only driver mode.
  int tccDriver;
  // Whether cooperative devices may use different functions.
  int cooperativeMultiDeviceUnmatchedFunc;
  // Whether cooperative devices may use different grid dimensions.
  int cooperativeMultiDeviceUnmatchedGridDim;
  // Whether cooperative devices may use different block dimensions.
  int cooperativeMultiDeviceUnmatchedBlockDim;
  // Whether cooperative devices may use different shared-memory sizes.
  int cooperativeMultiDeviceUnmatchedSharedMem;
  // Whether the entire device allocation is host-addressable.
  int isLargeBar;
  // Hardware revision identifier.
  int asicRevision;
  // Whether managed allocation is supported.
  int managedMemory;
  // Whether the host can directly access managed allocations.
  int directManagedMemAccessFromHost;
  // Whether managed memory supports concurrent host and device access.
  int concurrentManagedAccess;
  // Whether pageable host memory is directly accessible.
  int pageableMemoryAccess;
  // Whether pageable access uses host page tables.
  int pageableMemoryAccessUsesHostPageTables;
} hipDeviceProp_tR0000;
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
typedef struct hipLaunchConfig_st hipLaunchConfig_t;
typedef struct hipLaunchParams_t hipLaunchParams;
typedef struct hipLibrary_st* hipLibrary_t;
typedef struct ihipLinkState_t* hipLinkState_t;
typedef struct hipMemcpy3DPeerParms hipMemcpy3DPeerParms;
typedef enum hipMemcpyFlags {
  hipMemcpyFlagDefault = 0x0,
  hipMemcpyFlagPreferOverlapWithCompute = 0x1,
  hipMemcpyFlagExtPreferCE = 0x100,
  hipMemcpyFlagExtOpSwap = 0x200,
  hipMemcpyFlagExtOpIndirectSrc = 0x400,
  hipMemcpyFlagExtOpIndirectDst = 0x800,
} hipMemcpyFlags;
typedef enum hipMemcpySrcAccessOrder {
  hipMemcpySrcAccessOrderInvalid = 0x0,
  hipMemcpySrcAccessOrderStream = 0x1,
  hipMemcpySrcAccessOrderDuringApiCall = 0x2,
  hipMemcpySrcAccessOrderAny = 0x3,
  hipMemcpySrcAccessOrderMax = 0x7fffffff,
} hipMemcpySrcAccessOrder;
typedef struct hipMemcpyAttributes {
  hipMemcpySrcAccessOrder srcAccessOrder;
  hipMemLocation srcLocHint;
  hipMemLocation dstLocHint;
  unsigned int flags;
} hipMemcpyAttributes;
typedef enum hipMemcpy3DOperandType {
  hipMemcpyOperandTypePointer = 0x1,
  hipMemcpyOperandTypeArray = 0x2,
  hipMemcpyOperandTypeMax = 0x7fffffff,
} hipMemcpy3DOperandType;
typedef struct hipOffset3D {
  size_t x;
  size_t y;
  size_t z;
} hipOffset3D;
typedef struct hipMemcpy3DOperand {
  hipMemcpy3DOperandType type;
  union {
    struct {
      void* ptr;
      size_t rowLength;
      size_t layerHeight;
      hipMemLocation locHint;
    } ptr;
    struct {
      hipArray_t array;
      hipOffset3D offset;
    } array;
  } op;
} hipMemcpy3DOperand;
typedef struct hipMemcpy3DBatchOp {
  hipMemcpy3DOperand src;
  hipMemcpy3DOperand dst;
  hipExtent extent;
  hipMemcpySrcAccessOrder srcAccessOrder;
  unsigned int flags;
} hipMemcpy3DBatchOp;
typedef struct hipResourceDesc hipResourceDesc;
typedef struct hipResourceViewDesc hipResourceViewDesc;
typedef struct hipTextureDesc hipTextureDesc;
typedef struct hipDevResource_st hipDevResource;
typedef struct hipDevSmResourceGroupParams_st hipDevSmResourceGroupParams;
typedef int hipDevResourceType;
typedef int hipMemRangeHandleType;
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

#define HRX_HIP_MIPMAPPED_ARRAY_MAGIC 0x6872786869706d70ull
#define HRX_HIP_MIPMAPPED_ARRAY_ALIGNMENT 512u

struct hipMipmappedArray_st {
  // References held by the registry and active API callers.
  iree_atomic_ref_count_t ref_count;
  // Next live mipmapped-array handle in the process registry.
  struct hipMipmappedArray_st* next_live_mipmapped_array;
  // Magic value used to reject invalid or freed handles.
  uint64_t magic;
  // Number of array levels owned by this handle.
  unsigned int level_count;
  // Owned array handles, one per mip level.
  hipArray_t* level_arrays;
  // Sum of level allocation sizes reported by memory-requirements APIs.
  size_t memory_size;
};

static iree_once_flag hrx_hip_mipmapped_array_registry_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t hrx_hip_mipmapped_array_registry_mutex;
static hipMipmappedArray_t hrx_hip_mipmapped_array_registry_head = NULL;

static void hrx_hip_mipmapped_array_registry_initialize(void) {
  iree_slim_mutex_initialize(&hrx_hip_mipmapped_array_registry_mutex);
}

static void hrx_hip_mipmapped_array_registry_lock(void) {
  iree_call_once(&hrx_hip_mipmapped_array_registry_once,
                 hrx_hip_mipmapped_array_registry_initialize);
  iree_slim_mutex_lock(&hrx_hip_mipmapped_array_registry_mutex);
}

static void hrx_hip_mipmapped_array_registry_insert(
    hipMipmappedArray_t mipmapped_array) {
  hrx_hip_mipmapped_array_registry_lock();
  mipmapped_array->next_live_mipmapped_array =
      hrx_hip_mipmapped_array_registry_head;
  hrx_hip_mipmapped_array_registry_head = mipmapped_array;
  iree_slim_mutex_unlock(&hrx_hip_mipmapped_array_registry_mutex);
}

static bool hrx_hip_mipmapped_array_registry_lookup(
    hipMipmappedArray_const_t mipmapped_array,
    hipMipmappedArray_t* out_mipmapped_array) {
  if (out_mipmapped_array) *out_mipmapped_array = NULL;
  if (!mipmapped_array) return false;
  bool found = false;
  hrx_hip_mipmapped_array_registry_lock();
  for (hipMipmappedArray_t current = hrx_hip_mipmapped_array_registry_head;
       current; current = current->next_live_mipmapped_array) {
    if ((hipMipmappedArray_const_t)current == mipmapped_array &&
        current->magic == HRX_HIP_MIPMAPPED_ARRAY_MAGIC) {
      iree_atomic_ref_count_inc(&current->ref_count);
      if (out_mipmapped_array) *out_mipmapped_array = current;
      found = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&hrx_hip_mipmapped_array_registry_mutex);
  return found;
}

static void hrx_hip_mipmapped_array_release(
    hipMipmappedArray_t mipmapped_array);

static hipError_t hrx_hip_mipmapped_array_level(
    hipArray_t* out_level_array, hipMipmappedArray_const_t mipmapped_array,
    unsigned int level) {
  if (out_level_array) *out_level_array = NULL;
  if (!out_level_array || !mipmapped_array) return hipErrorInvalidValue;
  hipMipmappedArray_t current = NULL;
  if (!hrx_hip_mipmapped_array_registry_lookup(mipmapped_array, &current)) {
    return hipErrorInvalidHandle;
  }
  hipError_t result = hipSuccess;
  if (level >= current->level_count) {
    result = hipErrorInvalidValue;
  } else {
    *out_level_array = current->level_arrays[level];
  }
  hrx_hip_mipmapped_array_release(current);
  return result;
}

static hipError_t hrx_hip_mipmapped_array_memory_size(
    hipMipmappedArray_const_t mipmapped_array, size_t* out_memory_size) {
  if (out_memory_size) *out_memory_size = 0;
  if (!mipmapped_array || !out_memory_size) return hipErrorInvalidValue;
  hipMipmappedArray_t current = NULL;
  if (!hrx_hip_mipmapped_array_registry_lookup(mipmapped_array, &current)) {
    return hipErrorInvalidHandle;
  }
  *out_memory_size = current->memory_size;
  hrx_hip_mipmapped_array_release(current);
  return hipSuccess;
}

static bool hrx_hip_mipmapped_array_registry_remove(
    hipMipmappedArray_t mipmapped_array,
    hipMipmappedArray_t* out_mipmapped_array) {
  if (out_mipmapped_array) *out_mipmapped_array = NULL;
  if (!mipmapped_array) return false;
  bool removed = false;
  hrx_hip_mipmapped_array_registry_lock();
  hipMipmappedArray_t* current = &hrx_hip_mipmapped_array_registry_head;
  while (*current) {
    if (*current == mipmapped_array &&
        (*current)->magic == HRX_HIP_MIPMAPPED_ARRAY_MAGIC) {
      if (out_mipmapped_array) *out_mipmapped_array = *current;
      *current = mipmapped_array->next_live_mipmapped_array;
      mipmapped_array->next_live_mipmapped_array = NULL;
      removed = true;
      break;
    }
    current = &(*current)->next_live_mipmapped_array;
  }
  iree_slim_mutex_unlock(&hrx_hip_mipmapped_array_registry_mutex);
  return removed;
}

static void hrx_hip_mipmapped_array_destroy(
    hipMipmappedArray_t mipmapped_array) {
  mipmapped_array->magic = 0;
  for (unsigned int i = 0; i < mipmapped_array->level_count; ++i) {
    if (mipmapped_array->level_arrays[i]) {
      (void)hipFreeArray(mipmapped_array->level_arrays[i]);
    }
  }
  free(mipmapped_array->level_arrays);
  free(mipmapped_array);
}

static void hrx_hip_mipmapped_array_release(
    hipMipmappedArray_t mipmapped_array) {
  if (mipmapped_array &&
      iree_atomic_ref_count_dec(&mipmapped_array->ref_count) == 1) {
    hrx_hip_mipmapped_array_destroy(mipmapped_array);
  }
}

static hipError_t hrx_hip_destroy_mipmapped_array(
    hipMipmappedArray_t mipmapped_array) {
  if (!mipmapped_array) return hipErrorInvalidValue;
  hipMipmappedArray_t removed_array = NULL;
  if (!hrx_hip_mipmapped_array_registry_remove(mipmapped_array,
                                               &removed_array)) {
    return hipErrorInvalidHandle;
  }
  hrx_hip_mipmapped_array_release(removed_array);
  return hipSuccess;
}

static hipError_t hrx_hip_valid_device(hipDevice_t device) {
  int count = 0;
  hipError_t result = hipGetDeviceCount(&count);
  if (result != hipSuccess) return result;
  return device >= 0 && device < count ? hipSuccess : hipErrorInvalidDevice;
}

static bool hrx_hip_no_visible_devices_requested(void) {
  int count = 0;
  hipError_t result = hipGetDeviceCount(&count);
  return result == hipErrorNoDevice || (result == hipSuccess && count == 0);
}

static size_t hrx_hip_mipmapped_level_dimension(size_t dimension,
                                                unsigned int level) {
  if (dimension <= 1 || level >= sizeof(size_t) * CHAR_BIT) return 1;
  const size_t shifted_dimension = dimension >> level;
  return shifted_dimension ? shifted_dimension : 1;
}

static hipError_t hrx_hip_array3d_descriptor_element_size(
    const HIP_ARRAY3D_DESCRIPTOR* descriptor, size_t* out_element_size) {
  if (!descriptor || !out_element_size) return hipErrorInvalidValue;
  *out_element_size = 0;
  if (descriptor->NumChannels != 1 && descriptor->NumChannels != 2 &&
      descriptor->NumChannels != 4) {
    return hipErrorInvalidValue;
  }
  size_t channel_bits = 0;
  switch (descriptor->Format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
    case HIP_AD_FORMAT_SIGNED_INT8:
      channel_bits = 8;
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
    case HIP_AD_FORMAT_SIGNED_INT16:
    case HIP_AD_FORMAT_HALF:
      channel_bits = 16;
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
    case HIP_AD_FORMAT_SIGNED_INT32:
    case HIP_AD_FORMAT_FLOAT:
      channel_bits = 32;
      break;
    default:
      return hipErrorInvalidValue;
  }
  if (channel_bits > SIZE_MAX / descriptor->NumChannels) {
    return hipErrorInvalidValue;
  }
  const size_t total_bits = channel_bits * descriptor->NumChannels;
  if (total_bits == 0 || total_bits % 8 != 0) return hipErrorInvalidValue;
  *out_element_size = total_bits / 8;
  return hipSuccess;
}

static hipError_t hrx_hip_mipmapped_array_level_size(
    const HIP_ARRAY3D_DESCRIPTOR* descriptor, unsigned int level,
    size_t* out_size) {
  if (!descriptor || !out_size) return hipErrorInvalidValue;
  *out_size = 0;
  size_t element_size = 0;
  hipError_t result =
      hrx_hip_array3d_descriptor_element_size(descriptor, &element_size);
  if (result != hipSuccess) return result;
  const size_t width =
      hrx_hip_mipmapped_level_dimension(descriptor->Width, level);
  const size_t height = hrx_hip_mipmapped_level_dimension(
      descriptor->Height ? descriptor->Height : 1, level);
  const size_t depth = hrx_hip_mipmapped_level_dimension(
      descriptor->Depth ? descriptor->Depth : 1, level);
  size_t row_size = 0;
  size_t slice_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(width, element_size, &row_size) ||
          !iree_host_size_checked_mul(row_size, height, &slice_size) ||
          !iree_host_size_checked_mul(slice_size, depth, out_size))) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t hrx_hip_validate_mipmapped_array_descriptor(
    const HIP_ARRAY3D_DESCRIPTOR* descriptor, unsigned int level_count) {
  if (!descriptor || level_count == 0 || descriptor->Width == 0) {
    return hipErrorInvalidValue;
  }
  if (descriptor->Height == 0 && descriptor->Depth != 0) {
    return hipErrorInvalidValue;
  }
  size_t ignored_size = 0;
  return hrx_hip_mipmapped_array_level_size(descriptor, 0, &ignored_size);
}

static hipError_t hrx_hip_spt_default_stream(hipStream_t* stream) {
  if (!stream) return hipErrorInvalidValue;
  // The public sentinel is resolved by the regular entry points through the
  // shared per-thread stream state. Keeping that state in one place gives
  // reset and context changes the same lifetime behavior for every API form.
  *stream = hipStreamPerThread;
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

typedef struct hrx_hip_stream_callback_thunk_t {
  hipStreamCallback_t callback;
  hipStream_t stream;
  void* user_data;
} hrx_hip_stream_callback_thunk_t;

static void hrx_hip_stream_callback_host_fn(void* user_data) {
  hrx_hip_stream_callback_thunk_t* thunk =
      (hrx_hip_stream_callback_thunk_t*)user_data;
  hipStreamCallback_t callback = thunk->callback;
  hipStream_t stream = thunk->stream;
  void* callback_user_data = thunk->user_data;
  free(thunk);
  callback(stream, hipSuccess, callback_user_data);
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

static void iree_hip_convert_device_properties_r0600_to_r0000(
    const hipDeviceProp_tR0600* source, hipDeviceProp_tR0000* target) {
  memset(target, 0, sizeof(*target));
  memcpy(target->name, source->name, sizeof(target->name));
  target->totalGlobalMem = source->totalGlobalMem;
  target->sharedMemPerBlock = source->sharedMemPerBlock;
  target->regsPerBlock = source->regsPerBlock;
  target->warpSize = source->warpSize;
  target->maxThreadsPerBlock = source->maxThreadsPerBlock;
  memcpy(target->maxThreadsDim, source->maxThreadsDim,
         sizeof(target->maxThreadsDim));
  memcpy(target->maxGridSize, source->maxGridSize, sizeof(target->maxGridSize));
  target->clockRate = source->clockRate;
  target->memoryClockRate = source->memoryClockRate;
  target->memoryBusWidth = source->memoryBusWidth;
  target->totalConstMem = source->totalConstMem;
  target->major = source->major;
  target->minor = source->minor;
  target->multiProcessorCount = source->multiProcessorCount;
  target->l2CacheSize = source->l2CacheSize;
  target->maxThreadsPerMultiProcessor = source->maxThreadsPerMultiProcessor;
  target->computeMode = source->computeMode;
  target->clockInstructionRate = source->clockInstructionRate;
  target->arch = source->arch;
  target->concurrentKernels = source->concurrentKernels;
  target->pciDomainID = source->pciDomainID;
  target->pciBusID = source->pciBusID;
  target->pciDeviceID = source->pciDeviceID;
  target->maxSharedMemoryPerMultiProcessor =
      source->maxSharedMemoryPerMultiProcessor;
  target->isMultiGpuBoard = source->isMultiGpuBoard;
  target->canMapHostMemory = source->canMapHostMemory;
  memcpy(target->gcnArchName, source->gcnArchName, sizeof(target->gcnArchName));
  target->integrated = source->integrated;
  target->cooperativeLaunch = source->cooperativeLaunch;
  target->cooperativeMultiDeviceLaunch = source->cooperativeMultiDeviceLaunch;
  target->maxTexture1DLinear = source->maxTexture1DLinear;
  target->maxTexture1D = source->maxTexture1D;
  memcpy(target->maxTexture2D, source->maxTexture2D,
         sizeof(target->maxTexture2D));
  memcpy(target->maxTexture3D, source->maxTexture3D,
         sizeof(target->maxTexture3D));
  target->hdpMemFlushCntl = source->hdpMemFlushCntl;
  target->hdpRegFlushCntl = source->hdpRegFlushCntl;
  target->memPitch = source->memPitch;
  target->textureAlignment = source->textureAlignment;
  target->texturePitchAlignment = source->texturePitchAlignment;
  target->kernelExecTimeoutEnabled = source->kernelExecTimeoutEnabled;
  target->ECCEnabled = source->ECCEnabled;
  target->tccDriver = source->tccDriver;
  target->cooperativeMultiDeviceUnmatchedFunc =
      source->cooperativeMultiDeviceUnmatchedFunc;
  target->cooperativeMultiDeviceUnmatchedGridDim =
      source->cooperativeMultiDeviceUnmatchedGridDim;
  target->cooperativeMultiDeviceUnmatchedBlockDim =
      source->cooperativeMultiDeviceUnmatchedBlockDim;
  target->cooperativeMultiDeviceUnmatchedSharedMem =
      source->cooperativeMultiDeviceUnmatchedSharedMem;
  target->isLargeBar = source->isLargeBar;
  target->asicRevision = source->asicRevision;
  target->managedMemory = source->managedMemory;
  target->directManagedMemAccessFromHost =
      source->directManagedMemAccessFromHost;
  target->concurrentManagedAccess = source->concurrentManagedAccess;
  target->pageableMemoryAccess = source->pageableMemoryAccess;
  target->pageableMemoryAccessUsesHostPageTables =
      source->pageableMemoryAccessUsesHostPageTables;
}

static hipError_t iree_hip_choose_device_r0600(
    int* device, const hipDeviceProp_tR0600* properties) {
  if (!device || !properties) return hipErrorInvalidValue;

  int device_count = 0;
  hipError_t result = hipGetDeviceCount(&device_count);
  if (result != hipSuccess) return result;

  *device = 0;
  unsigned int best_match_count = 0;
  for (int i = 0; i < device_count; ++i) {
    hipDeviceProp_t current = {0};
    result = hipGetDeviceProperties(&current, i);
    if (result != hipSuccess) return result;

    unsigned int requested_count = 0;
    unsigned int match_count = 0;
#define HRX_HIP_MATCH_MINIMUM(field)                         \
  do {                                                       \
    if (properties->field != 0) {                            \
      ++requested_count;                                     \
      if (current.field >= properties->field) ++match_count; \
    }                                                        \
  } while (0)
    HRX_HIP_MATCH_MINIMUM(major);
    HRX_HIP_MATCH_MINIMUM(minor);
    HRX_HIP_MATCH_MINIMUM(totalGlobalMem);
    HRX_HIP_MATCH_MINIMUM(sharedMemPerBlock);
    HRX_HIP_MATCH_MINIMUM(maxThreadsPerBlock);
    HRX_HIP_MATCH_MINIMUM(totalConstMem);
    HRX_HIP_MATCH_MINIMUM(multiProcessorCount);
    HRX_HIP_MATCH_MINIMUM(maxThreadsPerMultiProcessor);
    HRX_HIP_MATCH_MINIMUM(memoryClockRate);
    HRX_HIP_MATCH_MINIMUM(memoryBusWidth);
    HRX_HIP_MATCH_MINIMUM(l2CacheSize);
    HRX_HIP_MATCH_MINIMUM(regsPerBlock);
    HRX_HIP_MATCH_MINIMUM(maxSharedMemoryPerMultiProcessor);
    HRX_HIP_MATCH_MINIMUM(warpSize);
#undef HRX_HIP_MATCH_MINIMUM

    if (requested_count == match_count && match_count > best_match_count) {
      *device = i;
      best_match_count = match_count;
    }
  }
  return hipSuccess;
}

HIPAPI hipError_t hipChooseDeviceR0000(int* device,
                                       const hipDeviceProp_tR0000* properties) {
  if (!device || !properties) return hipErrorInvalidValue;
  hipDeviceProp_tR0600 current_properties = {0};
  current_properties.major = properties->major;
  current_properties.minor = properties->minor;
  current_properties.totalGlobalMem = properties->totalGlobalMem;
  current_properties.sharedMemPerBlock = properties->sharedMemPerBlock;
  current_properties.maxThreadsPerBlock = properties->maxThreadsPerBlock;
  current_properties.totalConstMem = properties->totalConstMem;
  current_properties.multiProcessorCount = properties->multiProcessorCount;
  current_properties.maxThreadsPerMultiProcessor =
      properties->maxThreadsPerMultiProcessor;
  current_properties.memoryClockRate = properties->memoryClockRate;
  current_properties.memoryBusWidth = properties->memoryBusWidth;
  current_properties.l2CacheSize = properties->l2CacheSize;
  current_properties.regsPerBlock = properties->regsPerBlock;
  current_properties.maxSharedMemoryPerMultiProcessor =
      properties->maxSharedMemoryPerMultiProcessor;
  current_properties.warpSize = properties->warpSize;
  return iree_hip_choose_device_r0600(device, &current_properties);
}

HIPAPI hipError_t hipChooseDeviceR0600(int* device,
                                       const hipDeviceProp_tR0600* properties) {
  return iree_hip_choose_device_r0600(device, properties);
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
  if (!major || !minor) return hipErrorInvalidValue;
  hipError_t result = hipDeviceGetAttribute(
      major, hipDeviceAttributeComputeCapabilityMajor, device);
  if (result != hipSuccess) return result;
  return hipDeviceGetAttribute(minor, hipDeviceAttributeComputeCapabilityMinor,
                               device);
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
  if (!pCopy) return hipErrorInvalidValue;
  if (pCopy->srcMemoryType == hipMemoryTypeArray) {
    return pCopy->srcArray ? hipErrorNotSupported : hipErrorInvalidValue;
  }
  if (pCopy->dstMemoryType == hipMemoryTypeArray) {
    return pCopy->dstArray ? hipErrorNotSupported : hipErrorInvalidValue;
  }

  const void* src = NULL;
  switch (pCopy->srcMemoryType) {
    case hipMemoryTypeHost:
      src = pCopy->srcHost;
      break;
    case hipMemoryTypeDevice:
    case hipMemoryTypeUnified:
      src = pCopy->srcDevice;
      break;
    default:
      return hipErrorInvalidValue;
  }
  void* dst = NULL;
  switch (pCopy->dstMemoryType) {
    case hipMemoryTypeHost:
      dst = pCopy->dstHost;
      break;
    case hipMemoryTypeDevice:
    case hipMemoryTypeUnified:
      dst = pCopy->dstDevice;
      break;
    default:
      return hipErrorInvalidValue;
  }
  if (!src || !dst) return hipErrorInvalidValue;
  if ((pCopy->WidthInBytes != 0 &&
       (pCopy->srcXInBytes > pCopy->srcPitch ||
        pCopy->dstXInBytes > pCopy->dstPitch ||
        pCopy->WidthInBytes > pCopy->srcPitch - pCopy->srcXInBytes ||
        pCopy->WidthInBytes > pCopy->dstPitch - pCopy->dstXInBytes))) {
    return hipErrorInvalidValue;
  }
  const size_t max_size = (size_t)-1;
  if ((pCopy->srcY != 0 &&
       pCopy->srcPitch > (max_size - pCopy->srcXInBytes) / pCopy->srcY) ||
      (pCopy->dstY != 0 &&
       pCopy->dstPitch > (max_size - pCopy->dstXInBytes) / pCopy->dstY)) {
    return hipErrorInvalidValue;
  }

  hipMemcpyKind kind = hipMemcpyDefault;
  if (pCopy->srcMemoryType == hipMemoryTypeHost &&
      pCopy->dstMemoryType == hipMemoryTypeHost) {
    kind = hipMemcpyHostToHost;
  } else if (pCopy->srcMemoryType == hipMemoryTypeHost) {
    kind = hipMemcpyHostToDevice;
  } else if (pCopy->dstMemoryType == hipMemoryTypeHost) {
    kind = hipMemcpyDeviceToHost;
  } else if (pCopy->srcMemoryType == hipMemoryTypeDevice &&
             pCopy->dstMemoryType == hipMemoryTypeDevice) {
    kind = hipMemcpyDeviceToDevice;
  }

  const char* src_base =
      (const char*)src + pCopy->srcY * pCopy->srcPitch + pCopy->srcXInBytes;
  char* dst_base =
      (char*)dst + pCopy->dstY * pCopy->dstPitch + pCopy->dstXInBytes;
  hipError_t result =
      hipMemcpy2D(dst_base, pCopy->dstPitch, src_base, pCopy->srcPitch,
                  pCopy->WidthInBytes, pCopy->Height, kind);
  return result == hipErrorNotFound ? hipErrorInvalidValue : result;
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

HIPAPI hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  return hrx_hip_destroy_mipmapped_array(mipmappedArray);
}

HIPAPI hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop,
                                              int device) {
  if (!prop) return hipErrorInvalidValue;
  hipDeviceProp_tR0600 current_properties = {0};
  hipError_t result = hipGetDeviceProperties(&current_properties, device);
  if (result != hipSuccess) return result;
  iree_hip_convert_device_properties_r0600_to_r0000(&current_properties, prop);
  return hipSuccess;
}

HIPAPI hipError_t hipGetMipmappedArrayLevel(
    hipArray_t* levelArray, hipMipmappedArray_const_t mipmappedArray,
    unsigned int level) {
  return hrx_hip_mipmapped_array_level(levelArray, mipmappedArray, level);
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

HIPAPI hipError_t hipGreenCtxCreate(hipExecutionCtx_t* ctx,
                                    hipDevResourceDesc_t desc, int device,
                                    unsigned int flags) {
  (void)ctx;
  (void)desc;
  (void)device;
  (void)flags;
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

HIPAPI hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim,
                                             dim3 blockDimX,
                                             void** kernelParams,
                                             unsigned int sharedMemBytes,
                                             hipStream_t stream) {
  int device_count = 0;
  hipError_t init_result = hipGetDeviceCount(&device_count);
  if (init_result != hipSuccess) return init_result;
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
  int device_count = 0;
  hipError_t init_result = hipGetDeviceCount(&device_count);
  if (init_result != hipSuccess) return init_result;
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

HIPAPI hipError_t hipMemAllocHost(void** ptr, size_t size) {
  return hipMallocHost(ptr, size);
}

HIPAPI hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch,
                                   size_t widthInBytes, size_t height,
                                   unsigned int elementSizeBytes) {
  (void)elementSizeBytes;
  return hipMallocPitch((void**)dptr, pitch, widthInBytes, height);
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

static bool hrx_hip_batch_access_order_valid(hipMemcpySrcAccessOrder order) {
  return order == hipMemcpySrcAccessOrderInvalid ||
         order == hipMemcpySrcAccessOrderStream ||
         order == hipMemcpySrcAccessOrderDuringApiCall ||
         order == hipMemcpySrcAccessOrderAny;
}

static hipError_t hrx_hip_channel_desc_element_size(
    const hipChannelFormatDesc* desc, size_t* out_element_size) {
  if (!desc || !out_element_size) return hipErrorInvalidValue;
  *out_element_size = 0;
  const int components[4] = {desc->x, desc->y, desc->z, desc->w};
  size_t channel_count = 0;
  size_t channel_bits = 0;
  for (size_t i = 0; i < 4; ++i) {
    if (components[i] == 0) continue;
    if (components[i] < 0) return hipErrorInvalidValue;
    const size_t component_bits = (size_t)components[i];
    if (channel_bits == 0) {
      channel_bits = component_bits;
    } else if (channel_bits != component_bits) {
      return hipErrorInvalidValue;
    }
    ++channel_count;
  }
  if (channel_count == 0 || channel_bits == 0 ||
      channel_bits > SIZE_MAX / channel_count ||
      (channel_bits * channel_count) % 8 != 0) {
    return hipErrorInvalidValue;
  }
  *out_element_size = channel_bits * channel_count / 8;
  return hipSuccess;
}

static hipError_t hrx_hip_batch_array_element_size(const hipMemcpy3DBatchOp* op,
                                                   size_t* out_element_size) {
  if (!op || !out_element_size) return hipErrorInvalidValue;
  *out_element_size = 1;
  hipArray_const_t array = NULL;
  if (op->src.type == hipMemcpyOperandTypeArray) {
    array = (hipArray_const_t)op->src.op.array.array;
  } else if (op->dst.type == hipMemcpyOperandTypeArray) {
    array = (hipArray_const_t)op->dst.op.array.array;
  }
  if (!array) return hipSuccess;
  hipChannelFormatDesc desc = {0};
  hipError_t result = hipGetChannelDesc(&desc, array);
  if (result != hipSuccess) return result;
  return hrx_hip_channel_desc_element_size(&desc, out_element_size);
}

static hipError_t hrx_hip_batch_set_operand(const hipMemcpy3DOperand* operand,
                                            bool source,
                                            const hipExtent* extent,
                                            size_t element_size,
                                            hipMemcpy3DParms* params) {
  if (!operand || !extent || !params) return hipErrorInvalidValue;
  switch (operand->type) {
    case hipMemcpyOperandTypePointer: {
      if (!operand->op.ptr.ptr && extent->width != 0 && extent->height != 0 &&
          extent->depth != 0) {
        return hipErrorInvalidValue;
      }
      const size_t row_length =
          operand->op.ptr.rowLength ? operand->op.ptr.rowLength : extent->width;
      const size_t layer_height = operand->op.ptr.layerHeight
                                      ? operand->op.ptr.layerHeight
                                      : extent->height;
      if (element_size == 0 || row_length > SIZE_MAX / element_size) {
        return hipErrorInvalidValue;
      }
      hipPitchedPtr pointer = {
          .ptr = operand->op.ptr.ptr,
          .pitch = row_length * element_size,
          .xsize = row_length,
          .ysize = layer_height,
      };
      if (source) {
        params->srcPtr = pointer;
      } else {
        params->dstPtr = pointer;
      }
      return hipSuccess;
    }
    case hipMemcpyOperandTypeArray:
      if (!operand->op.array.array && extent->width != 0 &&
          extent->height != 0 && extent->depth != 0) {
        return hipErrorInvalidValue;
      }
      if (source) {
        params->srcArray = operand->op.array.array;
        params->srcPos.x = operand->op.array.offset.x;
        params->srcPos.y = operand->op.array.offset.y;
        params->srcPos.z = operand->op.array.offset.z;
      } else {
        params->dstArray = operand->op.array.array;
        params->dstPos.x = operand->op.array.offset.x;
        params->dstPos.y = operand->op.array.offset.y;
        params->dstPos.z = operand->op.array.offset.z;
      }
      return hipSuccess;
    default:
      return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t hrx_hip_batch_make_3d_params(const hipMemcpy3DBatchOp* op,
                                               hipMemcpy3DParms* params) {
  if (!hrx_hip_batch_access_order_valid(op->srcAccessOrder) ||
      op->flags != hipMemcpyFlagDefault) {
    return hipErrorInvalidValue;
  }

  memset(params, 0, sizeof(*params));
  params->extent.width = op->extent.width;
  params->extent.height = op->extent.height;
  params->extent.depth = op->extent.depth;
  params->kind = hipMemcpyDefault;

  size_t element_size = 1;
  hipError_t result = hrx_hip_batch_array_element_size(op, &element_size);
  if (result != hipSuccess) return result;
  result = hrx_hip_batch_set_operand(&op->src, true, &op->extent, element_size,
                                     params);
  if (result != hipSuccess) return result;
  return hrx_hip_batch_set_operand(&op->dst, false, &op->extent, element_size,
                                   params);
}

HIPAPI hipError_t hipMemcpy3DBatchAsync(size_t numOps,
                                        struct hipMemcpy3DBatchOp* opList,
                                        size_t* failIdx,
                                        unsigned long long flags,
                                        hipStream_t stream) {
  if (numOps == 0 || flags != 0 || !opList) {
    return hipErrorInvalidValue;
  }
  for (size_t i = 0; i < numOps; ++i) {
    hipMemcpy3DParms params;
    hipError_t result = hrx_hip_batch_make_3d_params(&opList[i], &params);
    if (result == hipSuccess) result = hipMemcpy3DAsync(&params, stream);
    if (result != hipSuccess) {
      if (failIdx) *failIdx = i;
      return result;
    }
  }
  return hipSuccess;
}

HIPAPI hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes,
                                      size_t count, hipMemcpyAttributes* attrs,
                                      size_t* attrsIdxs, size_t numAttrs,
                                      size_t* failIdx, hipStream_t stream) {
  (void)attrsIdxs;
  if (!dsts || !srcs || !sizes || count == 0) return hipErrorInvalidValue;
  if ((attrs && numAttrs == 0) || (!attrs && numAttrs != 0)) {
    return hipErrorInvalidValue;
  }
  if (numAttrs != 0) return hipErrorNotSupported;

  for (size_t i = 0; i < count; ++i) {
    if (sizes[i] == 0) continue;
    if (!dsts[i] || !srcs[i]) {
      if (failIdx) *failIdx = i;
      return hipErrorInvalidValue;
    }
    hipError_t result =
        hipMemcpyAsync(dsts[i], srcs[i], sizes[i], hipMemcpyDefault, stream);
    if (result != hipSuccess) {
      if (failIdx) *failIdx = i;
      return result;
    }
  }
  return hipSuccess;
}

static hipError_t iree_hip_memset_d2d_async_rows(
    hipDeviceptr_t dst, size_t dstPitch, const void* value, size_t element_size,
    size_t width, size_t height, hipStream_t stream) {
  if (width == 0 || height == 0) return hipSuccess;
  if (!dst || !value || element_size == 0 || width > dstPitch ||
      width % element_size != 0) {
    return hipErrorInvalidValue;
  }
  const size_t max_size = (size_t)-1;
  if (height > 1 && dstPitch > (max_size - width) / (height - 1)) {
    return hipErrorInvalidValue;
  }
  const size_t row_elements = width / element_size;
  for (size_t row = 0; row < height; ++row) {
    hipError_t result = hipSuccess;
    hipDeviceptr_t row_dst = (hipDeviceptr_t)((uintptr_t)dst + row * dstPitch);
    switch (element_size) {
      case 1:
        result = hipMemsetD8Async(row_dst, *(const unsigned char*)value,
                                  row_elements, stream);
        break;
      case 2:
        result = hipMemsetD16Async(row_dst, *(const unsigned short*)value,
                                   row_elements, stream);
        break;
      case 4:
        result = hipMemsetD32Async(row_dst, *(const int*)value, row_elements,
                                   stream);
        break;
      default:
        return hipErrorInvalidValue;
    }
    if (result == hipErrorNotFound) return hipErrorInvalidValue;
    if (result != hipSuccess) return result;
  }
  return hipSuccess;
}

static hipError_t iree_hip_memset_d2d_rows(hipDeviceptr_t dst, size_t dstPitch,
                                           const void* value,
                                           size_t element_size, size_t width,
                                           size_t height) {
  hipError_t result = iree_hip_memset_d2d_async_rows(
      dst, dstPitch, value, element_size, width, height, NULL);
  if (result == hipSuccess) result = hipDeviceSynchronize();
  return result;
}

HIPAPI hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch,
                                 unsigned short value, size_t width,
                                 size_t height) {
  return iree_hip_memset_d2d_rows(dst, dstPitch, &value, sizeof(value), width,
                                  height);
}

HIPAPI hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch,
                                      unsigned short value, size_t width,
                                      size_t height, hipStream_t stream) {
  return iree_hip_memset_d2d_async_rows(dst, dstPitch, &value, sizeof(value),
                                        width, height, stream);
}

HIPAPI hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch,
                                 unsigned int value, size_t width,
                                 size_t height) {
  return iree_hip_memset_d2d_rows(dst, dstPitch, &value, sizeof(value), width,
                                  height);
}

HIPAPI hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch,
                                      unsigned int value, size_t width,
                                      size_t height, hipStream_t stream) {
  return iree_hip_memset_d2d_async_rows(dst, dstPitch, &value, sizeof(value),
                                        width, height, stream);
}

HIPAPI hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch,
                                unsigned char value, size_t width,
                                size_t height) {
  return iree_hip_memset_d2d_rows(dst, dstPitch, &value, sizeof(value), width,
                                  height);
}

HIPAPI hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch,
                                     unsigned char value, size_t width,
                                     size_t height, hipStream_t stream) {
  return iree_hip_memset_d2d_async_rows(dst, dstPitch, &value, sizeof(value),
                                        width, height, stream);
}

HIPAPI hipError_t hipMipmappedArrayCreate(
    hipMipmappedArray_t* pHandle, HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
  if (!pHandle || !pMipmappedArrayDesc) return hipErrorInvalidValue;
  *pHandle = NULL;
  if (hrx_hip_no_visible_devices_requested()) return hipErrorNoDevice;
  hipError_t result = hrx_hip_validate_mipmapped_array_descriptor(
      pMipmappedArrayDesc, numMipmapLevels);
  if (result != hipSuccess) return result;

  hipArray_t* level_arrays =
      (hipArray_t*)calloc(numMipmapLevels, sizeof(*level_arrays));
  if (!level_arrays) return hipErrorOutOfMemory;

  size_t memory_size = 0;
  for (unsigned int level = 0; level < numMipmapLevels; ++level) {
    HIP_ARRAY3D_DESCRIPTOR level_descriptor = *pMipmappedArrayDesc;
    level_descriptor.Width =
        hrx_hip_mipmapped_level_dimension(pMipmappedArrayDesc->Width, level);
    level_descriptor.Height =
        hrx_hip_mipmapped_level_dimension(pMipmappedArrayDesc->Height, level);
    level_descriptor.Depth =
        hrx_hip_mipmapped_level_dimension(pMipmappedArrayDesc->Depth, level);

    size_t level_size = 0;
    result =
        hrx_hip_mipmapped_array_level_size(&level_descriptor, 0, &level_size);
    if (result == hipSuccess) {
      result = hipArray3DCreate(&level_arrays[level], &level_descriptor);
    }
    if (result != hipSuccess) {
      for (unsigned int i = 0; i < level; ++i) {
        if (level_arrays[i]) {
          (void)hipFreeArray(level_arrays[i]);
        }
      }
      free(level_arrays);
      return result;
    }
    if (IREE_UNLIKELY(!iree_host_size_checked_add(memory_size, level_size,
                                                  &memory_size))) {
      for (unsigned int i = 0; i <= level; ++i) {
        if (level_arrays[i]) {
          (void)hipFreeArray(level_arrays[i]);
        }
      }
      free(level_arrays);
      return hipErrorInvalidValue;
    }
  }

  hipMipmappedArray_t mipmapped_array =
      (hipMipmappedArray_t)calloc(1, sizeof(*mipmapped_array));
  if (!mipmapped_array) {
    for (unsigned int i = 0; i < numMipmapLevels; ++i) {
      if (level_arrays[i]) {
        (void)hipFreeArray(level_arrays[i]);
      }
    }
    free(level_arrays);
    return hipErrorOutOfMemory;
  }

  mipmapped_array->magic = HRX_HIP_MIPMAPPED_ARRAY_MAGIC;
  iree_atomic_ref_count_init(&mipmapped_array->ref_count);
  mipmapped_array->level_count = numMipmapLevels;
  mipmapped_array->level_arrays = level_arrays;
  mipmapped_array->memory_size = memory_size;
  hrx_hip_mipmapped_array_registry_insert(mipmapped_array);
  *pHandle = mipmapped_array;
  return hipSuccess;
}

HIPAPI hipError_t hipMipmappedArrayGetMemoryRequirements(
    hipArrayMemoryRequirements* memoryRequirements, hipMipmappedArray_t mipmap,
    hipDevice_t device) {
  if (!memoryRequirements) return hipErrorInvalidValue;
  hipError_t result = hrx_hip_valid_device(device);
  if (result != hipSuccess) return result;
  size_t memory_size = 0;
  result = hrx_hip_mipmapped_array_memory_size(mipmap, &memory_size);
  if (result != hipSuccess) return result;
  memoryRequirements->alignment = HRX_HIP_MIPMAPPED_ARRAY_ALIGNMENT;
  memoryRequirements->size = memory_size;
  return hipSuccess;
}

HIPAPI hipError_t
hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
  return hrx_hip_destroy_mipmapped_array(hMipmappedArray);
}

HIPAPI hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray,
                                            hipMipmappedArray_t hMipMappedArray,
                                            unsigned int level) {
  return hipGetMipmappedArrayLevel(
      pLevelArray, (hipMipmappedArray_const_t)hMipMappedArray, level);
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
  if (!callback || flags != 0) return hipErrorInvalidValue;

  hrx_hip_stream_callback_thunk_t* thunk =
      (hrx_hip_stream_callback_thunk_t*)malloc(sizeof(*thunk));
  if (!thunk) return hipErrorOutOfMemory;
  thunk->callback = callback;
  thunk->stream = stream;
  thunk->user_data = userData;

  hipError_t result =
      hipLaunchHostFunc(stream, hrx_hip_stream_callback_host_fn, thunk);
  if (result != hipSuccess) {
    free(thunk);
  }
  return result;
}

HIPAPI hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr,
                                          size_t length, unsigned int flags) {
  if (!dev_ptr) return hipErrorInvalidValue;
  if (flags != hipMemAttachGlobal && flags != hipMemAttachHost &&
      flags != hipMemAttachSingle) {
    return hipErrorInvalidValue;
  }
  if (!stream && flags == hipMemAttachSingle) {
    return hipErrorInvalidValue;
  }

  hipMemoryType memory_type = hipMemoryTypeUnregistered;
  hipError_t result = hipPointerGetAttribute(
      &memory_type, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, dev_ptr);
  if (result != hipSuccess || memory_type != hipMemoryTypeManaged) {
    return hipErrorInvalidValue;
  }

  if (length != 0) {
    size_t allocation_size = 0;
    result = hipPointerGetAttribute(&allocation_size,
                                    HIP_POINTER_ATTRIBUTE_RANGE_SIZE, dev_ptr);
    if (result != hipSuccess || length != allocation_size) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

HIPAPI hipError_t hipStreamGetDevResource(hipStream_t stream,
                                          hipDevResource* resource,
                                          hipDevResourceType type) {
  (void)stream;
  (void)resource;
  (void)type;
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

HIPAPI hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch,
                                                hipArray_const_t src,
                                                size_t wOffset, size_t hOffset,
                                                size_t width, size_t height,
                                                hipMemcpyKind kind,
                                                hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width,
                                   height, kind, resolved_stream);
}

HIPAPI hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch,
                                           hipArray_const_t src, size_t wOffset,
                                           size_t hOffset, size_t width,
                                           size_t height, hipMemcpyKind kind) {
  return hipMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height,
                              kind);
}

HIPAPI hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset,
                                              size_t hOffset, const void* src,
                                              size_t spitch, size_t width,
                                              size_t height, hipMemcpyKind kind,
                                              hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width,
                                 height, kind, resolved_stream);
}

HIPAPI hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset,
                                         size_t hOffset, const void* src,
                                         size_t spitch, size_t width,
                                         size_t height, hipMemcpyKind kind) {
  return hipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height,
                            kind);
}

HIPAPI hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src,
                                         size_t wOffset, size_t hOffset,
                                         size_t count, hipMemcpyKind kind) {
  return hipMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
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

HIPAPI hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol,
                                               size_t size_bytes, size_t offset,
                                               hipMemcpyKind kind,
                                               hipStream_t stream) {
  hipStream_t resolved_stream = NULL;
  hipError_t result = hrx_hip_spt_stream_or_explicit(stream, &resolved_stream);
  if (result != hipSuccess) return result;
  return hipMemcpyFromSymbolAsync(dst, symbol, size_bytes, offset, kind,
                                  resolved_stream);
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
