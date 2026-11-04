// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_PRIVATE_RECORDS_H_
#define AMDF_SRC_GPU_UMD_DXG_PRIVATE_RECORDS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum number of physical adapters represented by one private chain.
#define AMDF_DXG_PRIVATE_MAX_PHYSICAL_ADAPTER_COUNT 4u

// Number of scheduler identifiers published by one extended adapter record.
#define AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT 32u

// Caller-owned transient storage required by a linked-adapter query.
#define AMDF_DXG_PRIVATE_LINKED_QUERY_SCRATCH_BYTE_LENGTH 0xF160u

// Caller-owned transient storage required by a single-adapter query.
#define AMDF_DXG_PRIVATE_SINGLE_QUERY_SCRATCH_BYTE_LENGTH 0x4360u

// Fixed private record sizes accepted by the WDDM compute interface.
#define AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH 0x40u
#define AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH 0x218u
#define AMDF_DXG_PRIVATE_PROXY_RESOURCE_RECORD_BYTE_LENGTH 0x40u
#define AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH 0x1A8u
#define AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH 0x40u
#define AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH 0x48u

// Result of one private-record operation.
typedef uint32_t amdf_dxg_private_result_t;
enum amdf_dxg_private_result_e {
  // The operation completed successfully.
  AMDF_DXG_PRIVATE_RESULT_SUCCESS = 0,
  // The selected adapter or its decoded hardware identity is unsupported.
  AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED = 1,
  // The synchronous native query returned a nonzero status.
  AMDF_DXG_PRIVATE_RESULT_NATIVE_FAILURE = 2,
  // The native response violates the private record contract.
  AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD = 3,
};

// Synchronously fills one zero-initialized UMD private query buffer.
//
// The callback owns the D3DKMT transport and returns its native status. Buffer
// storage remains owned by the caller and is borrowed only for this call.
typedef uint32_t (*amdf_dxg_private_query_fn_t)(void* context,
                                                uint32_t adapter_handle,
                                                void* buffer,
                                                uint32_t buffer_byte_length);

// Caller-owned transport and scratch for one private adapter query.
typedef struct amdf_dxg_private_adapter_query_t {
  // Live D3DKMT adapter handle.
  uint32_t adapter_handle;
  // Physical adapter selected from the private chain.
  uint32_t physical_adapter_index;
  // Owner-defined context forwarded unchanged to |query|.
  void* query_context;
  // Synchronous UMD private query callback.
  amdf_dxg_private_query_fn_t query;
  // Caller-owned transient query storage.
  void* scratch;
} amdf_dxg_private_adapter_query_t;

// Immutable facts decoded for one physical adapter.
typedef struct amdf_dxg_private_adapter_info_t {
  // ASIC revision identifier.
  uint32_t asic_revision;
  // GFXIP major version decoded from the extended adapter record.
  int32_t gfx_ip_major;
  // GFXIP minor version decoded from the extended adapter record.
  int32_t gfx_ip_minor;
  // GFXIP stepping decoded from the extended adapter record.
  int32_t gfx_ip_stepping;
  // Nonzero for a discrete adapter and zero for an integrated adapter.
  uint32_t is_discrete;
  // Number of lanes in one hardware wavefront.
  uint32_t wavefront_size;
  // Total active compute units represented by the record.
  uint32_t compute_unit_count;
  // Maximum resident hardware waves per compute unit.
  uint32_t maximum_wave_count_per_compute_unit;
  // Maximum scratch-backed waves per compute unit.
  uint32_t maximum_scratch_wave_count_per_compute_unit;
  // Local data share capacity per compute unit in bytes.
  uint64_t local_data_share_byte_length;
  // Total number of active shader engines.
  uint32_t shader_engine_count;
  // Number of active XCCs represented by the physical adapter.
  uint32_t xcc_count;
  // Nonzero when firmware-managed graphics state is available.
  uint32_t supports_firmware_state_shadowing;
  // Nonzero when the CPU-facing route provides native platform atomics.
  uint32_t supports_platform_atomics;
  // Native persistent user-queue byte length, or zero when absent.
  uint32_t user_queue_byte_length;
  // Compute scheduler selected for immutable PM4 command streams.
  uint8_t pm4_scheduler_id;
  // AMDF_DXG_PRIVATE_NATIVE_QUEUE_* capabilities published by the driver.
  uint8_t native_queue_flags;
  // Device-local heap bytes that are not CPU visible.
  uint64_t invisible_local_heap_byte_length;
  // Scheduler identifier for each D3DKMT node ordinal.
  uint8_t scheduler_ids[AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT];
  // Bit i is set when scheduler_ids[i] uses hardware scheduling.
  uint64_t hardware_scheduling_node_mask;
} amdf_dxg_private_adapter_info_t;

// Native directly published user-queue capabilities.
enum amdf_dxg_private_native_queue_flag_bits_e {
  // Native AQL compute queues are accepted by the kernel driver.
  AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL = 1u << 0,
  // Native HSA SDMA queues are accepted by the kernel driver.
  AMDF_DXG_PRIVATE_NATIVE_QUEUE_SDMA = 1u << 1,
};

// Immutable scheduler facts for one native command family.
typedef struct amdf_dxg_private_queue_profile_t {
  // Scheduler identifier encoded into the private context record.
  uint32_t scheduler_id;
  // D3DKMT node ordinal used to create the execution context.
  uint32_t node_ordinal;
  // Nonzero when the driver exposes a usable scheduler.
  uint32_t is_supported;
  // Nonzero when the scheduler requires a hardware-queue object.
  uint32_t uses_hardware_scheduling;
} amdf_dxg_private_queue_profile_t;

// Derives the immutable PM4 and SDMA command-stream scheduler profiles.
void amdf_dxg_private_initialize_queue_profiles(
    const amdf_dxg_private_adapter_info_t* info,
    amdf_dxg_private_queue_profile_t* out_pm4_profile,
    amdf_dxg_private_queue_profile_t* out_sdma_profile);

// Queries and decodes one physical adapter from a linked-adapter chain.
//
// |query->scratch| must provide
// AMDF_DXG_PRIVATE_LINKED_QUERY_SCRATCH_BYTE_LENGTH writable bytes.
// |out_info| is written only on success. |out_native_status| receives zero for
// every locally resolved result and the exact callback status on native
// failure.
amdf_dxg_private_result_t amdf_dxg_private_query_linked_adapter(
    const amdf_dxg_private_adapter_query_t* query,
    amdf_dxg_private_adapter_info_t* out_info, uint32_t* out_native_status);

// Queries and decodes the single physical adapter represented by a native
// Windows adapter handle without retaining state.
//
// physical_adapter_index must be zero. |query->scratch| must provide
// AMDF_DXG_PRIVATE_SINGLE_QUERY_SCRATCH_BYTE_LENGTH writable bytes.
// |out_info| is written only on success. |out_native_status| follows the same
// contract as amdf_dxg_private_query_linked_adapter.
amdf_dxg_private_result_t amdf_dxg_private_query_single_adapter(
    const amdf_dxg_private_adapter_query_t* query,
    amdf_dxg_private_adapter_info_t* out_info, uint32_t* out_native_status);

// Physical allocation domain encoded into one private allocation record.
typedef uint32_t amdf_dxg_private_allocation_domain_t;
enum amdf_dxg_private_allocation_domain_e {
  // Driver-owned system memory backed by caller-provided host storage.
  AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM = 0,
  // Device-local memory placed by the D3DKMT allocation request.
  AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL = 1,
  // Caller-owned host pages registered with the GPU.
  AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_REGISTERED_HOST = 2,
};

// Private allocation behavior bits used by system and registered memory.
typedef uint32_t amdf_dxg_private_allocation_flags_t;
enum amdf_dxg_private_allocation_flag_bits_e {
  // Requests fine-grained host/device coherence.
  AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN = 1u << 0,
  // Requests the uncached device mapping used by kernel arguments.
  AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED = 1u << 1,
  // Marks storage directly published as a user-mode queue.
  AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE = 1u << 2,
};

// Inputs for one fixed per-allocation record.
typedef struct amdf_dxg_private_allocation_info_t {
  // Physical allocation length in bytes.
  uint64_t byte_length;
  // One AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_* value.
  amdf_dxg_private_allocation_domain_t domain;
  // AMDF_DXG_PRIVATE_ALLOCATION_FLAG_* bits.
  amdf_dxg_private_allocation_flags_t flags;
} amdf_dxg_private_allocation_info_t;

// Initializes the fixed driver-level allocation record.
void amdf_dxg_private_initialize_allocation_driver_record(void* record);

// Initializes one fixed WSL per-allocation record.
//
// |info| must contain a defined domain and flags. |adapter_info| is the
// immutable result of the adapter query used for the allocation. |record|
// must provide AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH writable bytes.
void amdf_dxg_private_initialize_wsl_allocation_record(
    const amdf_dxg_private_allocation_info_t* info,
    const amdf_dxg_private_adapter_info_t* adapter_info, void* record);

// Initializes one fixed native-Windows per-allocation record.
//
// The Windows and WSL UMD protocols share a record envelope but use distinct
// segment encodings. Inputs follow the WSL initializer contract above.
void amdf_dxg_private_initialize_wddm_allocation_record(
    const amdf_dxg_private_allocation_info_t* info,
    const amdf_dxg_private_adapter_info_t* adapter_info, void* record);

// Allocation metadata returned by an opened native-Windows resource.
typedef struct amdf_dxg_private_allocation_metadata_t {
  // Physical allocation length in bytes.
  uint64_t byte_length;
  // Addrlib surface swizzle mode.
  uint32_t swizzle_mode;
  // Plane-zero pipe-bank XOR value.
  uint32_t tile_swizzle;
  // Distributed compression mode, where zero means uncompressed.
  uint32_t compression_mode;
  // Driver encoding of the maximum compressed block size.
  uint32_t maximum_compressed_block_size;
  // Driver encoding of the maximum uncompressed block size.
  uint32_t maximum_uncompressed_block_size;
} amdf_dxg_private_allocation_metadata_t;

// Decodes returned private allocation metadata.
//
// Returns false without publishing |out_metadata| when the supplied response
// is too short or does not carry a complete surface description.
bool amdf_dxg_private_decode_allocation_metadata(
    const void* record, size_t record_byte_length,
    amdf_dxg_private_allocation_metadata_t* out_metadata);

// Initializes one compute context record.
void amdf_dxg_private_initialize_context_record(
    uint32_t supports_firmware_state_shadowing, uint32_t scheduler_id,
    uint64_t debugger_data, void* record);

// Hardware queue scheduling priority encoded into one PM4 queue record.
typedef uint32_t amdf_dxg_private_queue_priority_t;
enum amdf_dxg_private_queue_priority_e {
  // Low-priority queue scheduling.
  AMDF_DXG_PRIVATE_QUEUE_PRIORITY_LOW = 0,
  // Normal-priority queue scheduling.
  AMDF_DXG_PRIVATE_QUEUE_PRIORITY_NORMAL = 1,
  // High-priority queue scheduling.
  AMDF_DXG_PRIVATE_QUEUE_PRIORITY_HIGH = 2,
};

// Initializes one PM4 hardware-queue record.
//
// |priority| must be one AMDF_DXG_PRIVATE_QUEUE_PRIORITY_* value.
void amdf_dxg_private_initialize_hardware_queue_record(
    uint32_t supports_firmware_state_shadowing,
    amdf_dxg_private_queue_priority_t priority,
    uint32_t context_save_restore_allocation_handle, void* record);

// Initializes one immutable command submission record.
void amdf_dxg_private_initialize_submission_record(
    uint32_t queue_handle, uint64_t command_buffer_address,
    uint32_t command_buffer_byte_length, uint32_t uses_hardware_queue,
    void* record);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_PRIVATE_RECORDS_H_
