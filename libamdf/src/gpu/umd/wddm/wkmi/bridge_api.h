// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_BRIDGE_API_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_BRIDGE_API_H_

#include <stdint.h>

#include "amdf/amdf.h"

#if defined(_WIN32)
#define AMDF_WKMI_BRIDGE_CALL __cdecl
#else
#define AMDF_WKMI_BRIDGE_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// First supported private bridge ABI version.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_1 1u

// First version carrying the instance allocator across the DLL boundary.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_2 2u

// First version providing typed shared-buffer preparation.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_3 3u

// First version consuming GPU queues even when native destruction fails.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_4 4u

// First version carrying native CPU atomic route properties.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_5 5u

// First version exposing borrowed native queue handles for mapping ordering.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_6 6u

// First version querying the selected queue engine's timestamp frequency.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_7 7u

// First version preserving native GPU and host calibration timestamps.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_8 8u

// First version distinguishing accepted commands from signal publication and
// identifying whether the SDMA scheduler exposes a hardware-queue handle.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_9 9u

// Most recent private bridge ABI version described by this header.
#define AMDF_WKMI_BRIDGE_ABI_VERSION_LATEST AMDF_WKMI_BRIDGE_ABI_VERSION_9

// Result of one bridge operation.
typedef uint32_t amdf_wkmi_bridge_result_t;
enum amdf_wkmi_bridge_result_e {
  // The operation completed successfully.
  AMDF_WKMI_BRIDGE_RESULT_SUCCESS = 0,
  // The adapter or requested operation is not represented by pinned WKMI.
  AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED = 1,
  // A required argument or output structure was invalid.
  AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT = 2,
  // The caller and bridge have no mutually supported ABI version.
  AMDF_WKMI_BRIDGE_RESULT_VERSION_MISMATCH = 3,
  // WKMI or the Windows kernel-mode driver returned a native failure.
  AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE = 4,
  // A bridge-owned allocation failed.
  AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED = 5,
  // The binary dependency failed without a representable native status.
  AMDF_WKMI_BRIDGE_RESULT_INTERNAL = 6,
  // Caller storage cannot hold the complete result.
  AMDF_WKMI_BRIDGE_RESULT_BUFFER_TOO_SMALL = 7,
  // A scalar input cannot be represented by the native interface.
  AMDF_WKMI_BRIDGE_RESULT_OUT_OF_RANGE = 8,
  // Live child state prevents the requested lifecycle transition.
  AMDF_WKMI_BRIDGE_RESULT_BUSY = 9,
};

// Opaque parsed adapter state retained entirely inside the bridge.
typedef struct amdf_wkmi_bridge_gpu_adapter_t amdf_wkmi_bridge_gpu_adapter_t;

// Complete engine clock tuple returned by KMT calibration.
typedef struct amdf_wkmi_bridge_gpu_clock_sample_t {
  // GPU reference clock ticks per second.
  uint64_t frequency_hz;
  // GPU counter sampled by the native driver.
  uint64_t timestamp;
  // Paired CPU counter in QueryPerformanceCounter units.
  uint64_t host_timestamp;
} amdf_wkmi_bridge_gpu_clock_sample_t;

// Opaque native GPU kernel queue retained entirely inside the bridge.
typedef struct amdf_wkmi_bridge_gpu_kernel_queue_t
    amdf_wkmi_bridge_gpu_kernel_queue_t;

// Provider properties returned for one qualified physical GPU adapter.
typedef struct amdf_wkmi_bridge_gpu_properties_t {
  // Graphics IP major version, or a negative value when unavailable.
  int32_t gfx_ip_major;
  // Graphics IP minor version, or a negative value when unavailable.
  int32_t gfx_ip_minor;
  // Graphics IP stepping, or a negative value when unavailable.
  int32_t gfx_ip_stepping;
  // Raw HSA/KFD ASIC revision published by WKMI.
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
  // Nonzero when the discrete adapter's platform route supports CPU atomics.
  uint32_t supports_platform_atomics;
  // Nonzero when the selected SDMA scheduler exposes a hardware-queue handle.
  uint32_t supports_sdma_hardware_queue;
} amdf_wkmi_bridge_gpu_properties_t;

#ifdef __cplusplus
static_assert(sizeof(amdf_wkmi_bridge_gpu_properties_t) == 72,
              "WKMI GPU property ABI must remain stable");
#else
_Static_assert(sizeof(amdf_wkmi_bridge_gpu_properties_t) == 72,
               "WKMI GPU property ABI must remain stable");
#endif

// Physical allocation domain consumed by pinned WKMI.
typedef uint32_t amdf_wkmi_bridge_gpu_allocation_domain_t;
enum amdf_wkmi_bridge_gpu_allocation_domain_e {
  // Driver-owned system memory backed by the supplied host storage.
  AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM = 1,
  // Device-local memory placed at the supplied GPU virtual address.
  AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL = 2,
  // Caller-owned host pages registered with the GPU.
  AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST = 3,
};

// Private WKMI allocation behavior bits.
typedef uint32_t amdf_wkmi_bridge_gpu_allocation_flags_t;
enum amdf_wkmi_bridge_gpu_allocation_flag_bits_e {
  // Requests fine-grained host/device coherence.
  AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN = 1u << 0,
  // Marks directly published user-mode queue storage.
  AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE = 1u << 1,
};

// Parameters for one grouped native allocation creation.
typedef struct amdf_wkmi_bridge_gpu_allocation_create_info_t {
  // Must be at least the size of this structure.
  uint32_t structure_size;
  // Live logical D3DKMT device receiving the allocations.
  uint32_t device_handle;
  // One `AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_*` value.
  amdf_wkmi_bridge_gpu_allocation_domain_t domain;
  // `AMDF_WKMI_BRIDGE_GPU_ALLOCATION_FLAG_*` bits.
  amdf_wkmi_bridge_gpu_allocation_flags_t flags;
  // Page-aligned aggregate physical byte length.
  uint64_t byte_length;
  // First GPU address for local placement, otherwise zero.
  uint64_t placement_device_address;
  // First host byte for system or registered memory, otherwise `NULL`.
  void* host_pointer;
} amdf_wkmi_bridge_gpu_allocation_create_info_t;

#ifdef __cplusplus
static_assert(sizeof(amdf_wkmi_bridge_gpu_allocation_create_info_t) == 40,
              "WKMI allocation create ABI must remain stable");
#else
_Static_assert(sizeof(amdf_wkmi_bridge_gpu_allocation_create_info_t) == 40,
               "WKMI allocation create ABI must remain stable");
#endif

// Native GPU command representation accepted by one WKMI kernel queue.
typedef uint32_t amdf_wkmi_bridge_gpu_queue_command_type_t;
enum amdf_wkmi_bridge_gpu_queue_command_type_e {
  // Native PM4 command streams submitted to a compute scheduler.
  AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4 = 1,
  // Native SDMA command streams submitted to an SDMA scheduler.
  AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA = 2,
};

// Parameters used to construct one native GPU kernel queue.
typedef struct amdf_wkmi_bridge_gpu_kernel_queue_create_info_t {
  // Must be at least the size of this structure.
  uint32_t structure_size;
  // Live logical D3DKMT device receiving the execution context.
  uint32_t device_handle;
  // Native command representation accepted by the queue.
  amdf_wkmi_bridge_gpu_queue_command_type_t command_type;
  // Reserved for compatible growth and must be zero.
  uint32_t reserved;
} amdf_wkmi_bridge_gpu_kernel_queue_create_info_t;

#ifdef __cplusplus
static_assert(sizeof(amdf_wkmi_bridge_gpu_kernel_queue_create_info_t) == 16,
              "WKMI queue create ABI must remain stable");
#else
_Static_assert(sizeof(amdf_wkmi_bridge_gpu_kernel_queue_create_info_t) == 16,
               "WKMI queue create ABI must remain stable");
#endif

// Native GPU kernel-queue properties established during creation.
typedef struct amdf_wkmi_bridge_gpu_kernel_queue_info_t {
  // Must be at least the size of this structure.
  uint32_t structure_size;
  // Required command-buffer base alignment in bytes.
  uint32_t command_buffer_alignment;
  // Maximum command-buffer byte length accepted by one submission.
  uint64_t maximum_command_buffer_byte_length;
  // Monitored progress-fence object owned by the native queue state.
  uint32_t progress_fence_handle;
  // Reserved for compatible growth and always zero.
  uint32_t reserved;
  // Read-only CPU mapping of the monotonic progress fence.
  const volatile uint64_t* progress_fence_pointer;
  // GPU address of the monotonic progress fence.
  uint64_t progress_fence_device_address;
  // Execution context borrowed through queue destruction.
  uint32_t context_handle;
  // Hardware queue for native dependencies, or zero for a SWS context.
  uint32_t queue_handle;
} amdf_wkmi_bridge_gpu_kernel_queue_info_t;

#ifdef __cplusplus
static_assert(sizeof(amdf_wkmi_bridge_gpu_kernel_queue_info_t) == 48,
              "WKMI queue info ABI must remain stable");
#else
_Static_assert(sizeof(amdf_wkmi_bridge_gpu_kernel_queue_info_t) == 48,
               "WKMI queue info ABI must remain stable");
#endif

// Native input for a same-adapter committed D3D12 buffer import.
typedef struct amdf_wkmi_bridge_gpu_buffer_import_info_t {
  // Live logical KMT device that receives the imported allocation.
  uint32_t device_handle;
  // Reserved for compatible growth and always zero.
  uint32_t reserved;
  // Exact DXGI adapter LUID, low word first.
  uint64_t adapter_luid;
  // Borrowed owning NT handle for the complete committed resource.
  void* shared_handle;
} amdf_wkmi_bridge_gpu_buffer_import_info_t;

// Immutable entry-point table for the negotiated private bridge ABI.
typedef struct amdf_wkmi_bridge_api_t {
  // Size in bytes of this table version.
  uint32_t structure_size;
  // Bridge ABI version implemented by this table.
  uint32_t abi_version;

  // Parses and retains one physical GPU adapter.
  //
  // |adapter_handle| is a live D3DKMT adapter handle and
  // |physical_adapter_index| selects one physical adapter represented by it.
  // |host_allocator| is copied into the adapter and must remain callable until
  // adapter close succeeds.
  // Success publishes the adapter and properties. Every other result leaves
  // both outputs unchanged and retains no native adapter ownership.
  // |out_native_status| receives the NTSTATUS only for
  // AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE and is zero otherwise.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_adapter_open)(
      uint32_t adapter_handle, uint32_t physical_adapter_index,
      const amdf_allocator_t* host_allocator,
      amdf_wkmi_bridge_gpu_adapter_t** out_adapter,
      amdf_wkmi_bridge_gpu_properties_t* out_properties,
      uint32_t* out_native_status);

  // Releases parsed adapter state after every dependent bridge call returns.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_adapter_close)(
      amdf_wkmi_bridge_gpu_adapter_t* adapter, uint32_t* out_native_status);

  // Queries the native allocation layout for an aggregate byte length.
  // Success publishes both layout outputs. Every other result leaves them
  // unchanged.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_allocation_query_layout)(
      amdf_wkmi_bridge_gpu_adapter_t* adapter, uint64_t byte_length,
      uint32_t* out_allocation_count,
      uint64_t* out_maximum_allocation_byte_length);

  // Creates one grouped set of native allocations through pinned WKMI.
  //
  // No allocation is created unless `allocation_handle_capacity` is
  // sufficient. BUFFER_TOO_SMALL publishes only the required allocation count.
  // Zero capacity permits a NULL handle array for the count query.
  // Success publishes the handles, resource, and count. Every other result
  // leaves those outputs unchanged and retains no caller-owned allocation.
  // Native success forwards the driver's handle values without validation;
  // the backing owner validates them after capturing all cleanup identities.
  // The bridge retains no native allocation or deferred cleanup state.
  // `out_native_status` receives an NTSTATUS only for
  // AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE and is zero otherwise.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_allocation_create)(
      amdf_wkmi_bridge_gpu_adapter_t* adapter,
      const amdf_wkmi_bridge_gpu_allocation_create_info_t* create_info,
      uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
      uint32_t* out_resource_handle, uint32_t* out_allocation_count,
      uint32_t* out_native_status);

  // Creates one kernel-mediated native GPU queue through pinned WKMI.
  // Success publishes the queue and its complete information. Every other
  // result leaves both outputs unchanged and retains no caller-owned queue.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_kernel_queue_create)(
      amdf_wkmi_bridge_gpu_adapter_t* adapter,
      const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t* create_info,
      amdf_wkmi_bridge_gpu_kernel_queue_t** out_queue,
      amdf_wkmi_bridge_gpu_kernel_queue_info_t* out_info,
      uint32_t* out_native_status);

  // Publishes one native GPU command without reading its bytes. Sets accepted
  // after native submission even if a later completion-signal call fails.
  // Rejection sets accepted to zero; invalid arguments leave outputs unchanged.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_kernel_queue_submit)(
      amdf_wkmi_bridge_gpu_kernel_queue_t* queue,
      uint64_t command_buffer_address, uint64_t command_buffer_byte_length,
      uint64_t progress_value, uint32_t* out_accepted,
      uint32_t* out_native_status);

  // Consumes an idle queue even when native queue/context cleanup fails.
  // Returns the first native error and leaks unreleased native handles, without
  // retaining host metadata or an adapter child. The caller proves retirement
  // and retains application backing independently; release does not cancel
  // work.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_kernel_queue_destroy)(
      amdf_wkmi_bridge_gpu_kernel_queue_t* queue, uint32_t* out_native_status);

  // Prepares one typed native resource in initially zero slots of a live owner.
  // Native acquisition updates both slots before any later fallible work; the
  // owner destroys retained progress even on failure. Geometry outputs are
  // published only on success. The input handle is borrowed and never closed.
  // This operation reports portable libamdf statuses, preserving HRESULT,
  // Win32 and NTSTATUS failures without conflating their numeric codes.
  amdf_status_t(AMDF_WKMI_BRIDGE_CALL* gpu_buffer_prepare_import)(
      amdf_wkmi_bridge_gpu_adapter_t* adapter,
      const amdf_wkmi_bridge_gpu_buffer_import_info_t* import_info,
      uint32_t* resource_handle, uint32_t* allocation_handle,
      uint64_t* out_native_byte_length, uint64_t* out_buffer_byte_length);

  // Queries the native calibration tuple for the admitted queue scheduler.
  // Other results preserve the output; out_native_status receives NTSTATUS
  // only for NATIVE_FAILURE. No cross-source epoch identity is inferred.
  amdf_wkmi_bridge_result_t(AMDF_WKMI_BRIDGE_CALL* gpu_clock_sample)(
      amdf_wkmi_bridge_gpu_adapter_t* adapter, uint32_t adapter_handle,
      uint32_t physical_adapter_index,
      amdf_wkmi_bridge_gpu_queue_command_type_t command_type,
      amdf_wkmi_bridge_gpu_clock_sample_t* out_sample,
      uint32_t* out_native_status);

} amdf_wkmi_bridge_api_t;

#ifdef __cplusplus
static_assert(sizeof(amdf_wkmi_bridge_api_t) == 80,
              "WKMI entry-point table ABI must remain stable");
#else
_Static_assert(sizeof(amdf_wkmi_bridge_api_t) == 80,
               "WKMI entry-point table ABI must remain stable");
#endif

// Negotiates one immutable bridge API table. Failure leaves |out_api|
// unchanged.
typedef amdf_wkmi_bridge_result_t(
    AMDF_WKMI_BRIDGE_CALL* amdf_wkmi_bridge_query_api_fn_t)(
    uint32_t minimum_version, uint32_t maximum_version,
    const amdf_wkmi_bridge_api_t** out_api);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_BRIDGE_API_H_
