// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_PROVIDER_API_H_
#define AMDF_SRC_GPU_UMD_DXG_PROVIDER_API_H_

#include <stddef.h>
#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Result of one statically selected provider operation.
typedef uint32_t amdf_dxg_provider_result_t;
enum amdf_dxg_provider_result_e {
  // The operation completed successfully.
  AMDF_DXG_PROVIDER_RESULT_SUCCESS = 0,
  // The adapter or requested operation is not represented by the provider.
  AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED = 1,
  // A required argument or output structure was invalid.
  AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT = 2,
  // The private provider or kernel-mode driver returned a native failure.
  AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE = 3,
  // A provider-owned allocation failed.
  AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED = 4,
  // The provider failed without a representable native status.
  AMDF_DXG_PROVIDER_RESULT_INTERNAL = 5,
  // Caller storage cannot hold the complete result.
  AMDF_DXG_PROVIDER_RESULT_BUFFER_TOO_SMALL = 6,
  // A scalar input cannot be represented by the native interface.
  AMDF_DXG_PROVIDER_RESULT_OUT_OF_RANGE = 7,
  // Live child state prevents the requested lifecycle transition.
  AMDF_DXG_PROVIDER_RESULT_BUSY = 8,
};

// Opaque parsed adapter state retained entirely inside the provider.
typedef struct amdf_dxg_provider_gpu_adapter_t amdf_dxg_provider_gpu_adapter_t;

// Synchronously issues one native operation through its explicit owner.
// |request| carries the complete Linux ioctl request on a direct DXG provider.
typedef uint32_t (*amdf_dxg_provider_native_invoke_fn_t)(void* context,
                                                         uint64_t request,
                                                         void* arguments);

// Borrowed native operation boundary retained through adapter close.
typedef struct amdf_dxg_provider_native_transport_t {
  // Owner-defined context retained or forwarded according to provider policy.
  void* context;
  // Synchronous native operation callback, or NULL when the provider consumes
  // its typed context directly.
  amdf_dxg_provider_native_invoke_fn_t invoke;
} amdf_dxg_provider_native_transport_t;

// Native identity used to open one physical GPU adapter.
typedef struct amdf_dxg_provider_gpu_adapter_open_info_t {
  // Live native adapter handle.
  uint32_t adapter_handle;
  // Physical adapter represented by the native handles.
  uint32_t physical_adapter_index;
  // Exact DXGI adapter LUID, low word first.
  uint64_t adapter_luid;
  // Explicit host operation boundary retained by the provider adapter.
  amdf_dxg_provider_native_transport_t native_transport;
} amdf_dxg_provider_gpu_adapter_open_info_t;

// Complete engine clock tuple returned by KMT calibration.
typedef struct amdf_dxg_provider_gpu_clock_sample_t {
  // GPU reference clock ticks per second.
  uint64_t frequency_hz;
  // GPU counter sampled by the native driver.
  uint64_t timestamp;
  // Paired host observation in the provider platform's common clock units.
  // This is QPC on Windows and CLOCK_MONOTONIC_RAW nanoseconds on Linux.
  uint64_t host_timestamp;
} amdf_dxg_provider_gpu_clock_sample_t;

// Opaque native GPU kernel queue retained entirely inside the provider.
typedef struct amdf_dxg_provider_gpu_kernel_queue_t
    amdf_dxg_provider_gpu_kernel_queue_t;

// Provider properties returned for one qualified physical GPU adapter.
typedef struct amdf_dxg_provider_gpu_properties_t {
  // Graphics IP major version, or a negative value when unavailable.
  int32_t gfx_ip_major;
  // Graphics IP minor version, or a negative value when unavailable.
  int32_t gfx_ip_minor;
  // Graphics IP stepping, or a negative value when unavailable.
  int32_t gfx_ip_stepping;
  // Raw HSA/KFD ASIC revision published by the private provider.
  uint32_t asic_revision;
  // Number of lanes in one hardware wavefront.
  uint32_t wavefront_size;
  // Total active compute units across every XCC.
  uint32_t compute_unit_count;
  // Maximum resident hardware waves per compute unit.
  uint32_t maximum_wave_count_per_compute_unit;
  // Maximum scratch-backed waves per compute unit.
  uint32_t maximum_scratch_wave_count_per_compute_unit;
  // Local data share capacity per compute unit in bytes.
  uint64_t local_data_share_byte_length;
  // Number of active XCCs represented by the physical adapter.
  uint32_t xcc_count;
  // Total number of active shader engines across every XCC.
  uint32_t shader_engine_count;
  // Nonzero when a native PM4 hardware queue can be constructed.
  uint32_t supports_pm4_kernel_queue;
  // Nonzero when a native SDMA command queue can be constructed.
  uint32_t supports_sdma_kernel_queue;
  // Nonzero for a discrete adapter, zero for an integrated processor.
  uint32_t is_discrete;
  // Nonzero when the selected SDMA scheduler exposes a hardware-queue handle.
  uint32_t supports_sdma_hardware_queue;
  // Nonzero when the discrete adapter's CPU-facing route supports AtomicOps.
  // Integrated adapters use their on-package fabric independently of this bit.
  uint32_t platform_atomic_support;
} amdf_dxg_provider_gpu_properties_t;

// Physical allocation domain consumed by a private DXG provider.
typedef uint32_t amdf_dxg_provider_gpu_allocation_domain_t;
enum amdf_dxg_provider_gpu_allocation_domain_e {
  // Driver-owned system memory backed by the supplied host storage.
  AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_SYSTEM = 1,
  // Device-local memory placed at the supplied GPU virtual address.
  AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL = 2,
  // Caller-owned host pages registered with the GPU.
  AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST = 3,
};

// Private DXG allocation behavior bits.
typedef uint32_t amdf_dxg_provider_gpu_allocation_flags_t;
enum amdf_dxg_provider_gpu_allocation_flag_bits_e {
  // Requests fine-grained host/device coherence.
  AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_FINE_GRAIN = 1u << 0,
  // Marks directly published user-mode queue storage.
  AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_QUEUE_STORAGE = 1u << 1,
  // Requests an uncached GPU mapping independently of coherence.
  AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED = 1u << 2,
};

// Parameters for one grouped native allocation creation.
typedef struct amdf_dxg_provider_gpu_allocation_create_info_t {
  // Live logical D3DKMT device receiving the allocations.
  uint32_t device_handle;
  // One `AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_*` value.
  amdf_dxg_provider_gpu_allocation_domain_t domain;
  // `AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_*` bits.
  amdf_dxg_provider_gpu_allocation_flags_t flags;
  // Page-aligned aggregate physical byte length.
  uint64_t byte_length;
  // First GPU address for local placement, otherwise zero.
  uint64_t placement_device_address;
  // First host byte for system or registered memory, otherwise `NULL`.
  void* host_pointer;
} amdf_dxg_provider_gpu_allocation_create_info_t;

// Native GPU command representation accepted by one DXG kernel queue.
typedef uint32_t amdf_dxg_provider_gpu_queue_command_type_t;
enum amdf_dxg_provider_gpu_queue_command_type_e {
  // Native PM4 command streams submitted to a compute scheduler.
  AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4 = 1,
  // Native SDMA command streams submitted to an SDMA scheduler.
  AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA = 2,
};

// Parameters used to construct one native GPU kernel queue.
typedef struct amdf_dxg_provider_gpu_kernel_queue_create_info_t {
  // Live logical D3DKMT device receiving the execution context.
  uint32_t device_handle;
  // Native command representation accepted by the queue.
  amdf_dxg_provider_gpu_queue_command_type_t command_type;
} amdf_dxg_provider_gpu_kernel_queue_create_info_t;

// Native GPU kernel-queue properties established during creation.
typedef struct amdf_dxg_provider_gpu_kernel_queue_info_t {
  // Required command-buffer base alignment in bytes.
  uint32_t command_buffer_alignment;
  // Maximum command-buffer byte length accepted by one submission.
  uint64_t maximum_command_buffer_byte_length;
  // Monitored progress-fence object owned by the native queue state.
  uint32_t progress_fence_handle;
  // Read-only CPU mapping of the monotonic progress fence.
  const volatile uint64_t* progress_fence_pointer;
  // GPU address of the monotonic progress fence.
  uint64_t progress_fence_device_address;
  // Execution context borrowed through queue destruction.
  uint32_t context_handle;
  // Hardware queue for native dependencies, or zero for a SWS context.
  uint32_t queue_handle;
} amdf_dxg_provider_gpu_kernel_queue_info_t;

// Native input for a same-adapter committed D3D12 buffer import.
typedef struct amdf_dxg_provider_gpu_buffer_import_info_t {
  // Live logical KMT device that receives the imported allocation.
  uint32_t device_handle;
  // Exact DXGI adapter LUID, low word first.
  uint64_t adapter_luid;
  // Borrowed owning NT handle for the complete committed resource.
  void* shared_handle;
} amdf_dxg_provider_gpu_buffer_import_info_t;

// Immutable entry-point table for one statically selected private provider.
typedef struct amdf_dxg_provider_api_t {
  // Parses and retains one physical GPU adapter.
  //
  // |open_info| carries the native adapter identity required by the selected
  // private provider.
  // |host_allocator| is copied into the adapter and must remain callable until
  // adapter close succeeds.
  // Success publishes the adapter and properties. Every other result leaves
  // both outputs unchanged and retains no native adapter ownership.
  // |out_native_status| receives the NTSTATUS only for
  // AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE and is zero otherwise.
  amdf_dxg_provider_result_t (*gpu_adapter_open)(
      const amdf_dxg_provider_gpu_adapter_open_info_t* open_info,
      const amdf_allocator_t* host_allocator,
      amdf_dxg_provider_gpu_adapter_t** out_adapter,
      amdf_dxg_provider_gpu_properties_t* out_properties,
      uint32_t* out_native_status);

  // Releases parsed adapter state after every dependent provider call returns.
  amdf_dxg_provider_result_t (*gpu_adapter_close)(
      amdf_dxg_provider_gpu_adapter_t* adapter, uint32_t* out_native_status);

  // Queries the native allocation layout for an aggregate byte length.
  // Success publishes both layout outputs. Every other result leaves them
  // unchanged.
  amdf_dxg_provider_result_t (*gpu_allocation_query_layout)(
      amdf_dxg_provider_gpu_adapter_t* adapter, uint64_t byte_length,
      uint32_t* out_allocation_count,
      uint64_t* out_maximum_allocation_byte_length);

  // Creates one grouped set of native allocations through the provider.
  //
  // No allocation is created unless `allocation_handle_capacity` is
  // sufficient. BUFFER_TOO_SMALL publishes only the required allocation count.
  // Zero capacity permits a NULL handle array for the count query.
  // Success publishes the handles, resource, and count. Every other result
  // leaves those outputs unchanged and retains no caller-owned allocation.
  // Native success forwards the driver's handle values without validation;
  // the backing owner validates them after capturing all cleanup identities.
  // The provider retains no native allocation or deferred cleanup state.
  // `out_native_status` receives an NTSTATUS only for
  // AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE and is zero otherwise.
  amdf_dxg_provider_result_t (*gpu_allocation_create)(
      amdf_dxg_provider_gpu_adapter_t* adapter,
      const amdf_dxg_provider_gpu_allocation_create_info_t* create_info,
      uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
      uint32_t* out_resource_handle, uint32_t* out_allocation_count,
      uint32_t* out_native_status);

  // Creates one kernel-mediated native GPU queue through the provider.
  // Success publishes the queue and its complete information. Every other
  // result leaves both outputs unchanged and retains no caller-owned queue.
  amdf_dxg_provider_result_t (*gpu_kernel_queue_create)(
      amdf_dxg_provider_gpu_adapter_t* adapter,
      const amdf_dxg_provider_gpu_kernel_queue_create_info_t* create_info,
      amdf_dxg_provider_gpu_kernel_queue_t** out_queue,
      amdf_dxg_provider_gpu_kernel_queue_info_t* out_info,
      uint32_t* out_native_status);

  // Publishes one native GPU command without reading its bytes. Sets accepted
  // after native submission even if a later completion-signal call fails.
  // Rejection sets accepted to zero; invalid arguments leave outputs unchanged.
  amdf_dxg_provider_result_t (*gpu_kernel_queue_submit)(
      amdf_dxg_provider_gpu_kernel_queue_t* queue,
      uint64_t command_buffer_address, uint64_t command_buffer_byte_length,
      uint64_t progress_value, uint32_t* out_accepted,
      uint32_t* out_native_status);

  // Consumes an idle queue even when native queue/context cleanup fails.
  // Returns the first native error and leaks unreleased native handles, without
  // retaining host metadata or an adapter child. The caller proves retirement
  // and retains application backing independently; release does not cancel
  // work.
  amdf_dxg_provider_result_t (*gpu_kernel_queue_destroy)(
      amdf_dxg_provider_gpu_kernel_queue_t* queue, uint32_t* out_native_status);

  // Prepares one typed native resource in initially zero slots of a live owner.
  // Native acquisition updates both slots before any later fallible work; the
  // owner destroys retained progress even on failure. Geometry outputs are
  // published only on success. The input handle is borrowed and never closed.
  // This operation reports portable libamdf statuses, preserving HRESULT,
  // Win32 and NTSTATUS failures without conflating their numeric codes.
  amdf_status_t (*gpu_buffer_prepare_import)(
      amdf_dxg_provider_gpu_adapter_t* adapter,
      const amdf_dxg_provider_gpu_buffer_import_info_t* import_info,
      uint32_t* resource_handle, uint32_t* allocation_handle,
      uint64_t* out_native_byte_length, uint64_t* out_buffer_byte_length);

  // Queries the native calibration tuple for the admitted queue scheduler.
  // Other results preserve the output; out_native_status receives NTSTATUS
  // only for NATIVE_FAILURE. No cross-source epoch identity is inferred.
  amdf_dxg_provider_result_t (*gpu_clock_sample)(
      amdf_dxg_provider_gpu_adapter_t* adapter, uint32_t adapter_handle,
      uint32_t physical_adapter_index,
      amdf_dxg_provider_gpu_queue_command_type_t command_type,
      amdf_dxg_provider_gpu_clock_sample_t* out_sample,
      uint32_t* out_native_status);

} amdf_dxg_provider_api_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_PROVIDER_API_H_
