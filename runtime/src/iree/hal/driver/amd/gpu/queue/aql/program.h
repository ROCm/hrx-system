// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_PROGRAM_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_PROGRAM_H_

#include "iree/hal/driver/amd/gpu/code.h"
#include "iree/hal/driver/amd/gpu/queue/aql/counts.h"
#include "iree/hal/driver/amd/gpu/queue/aql/indirect.h"
#include "iree/hal/driver/amd/gpu/queue/aql/materialize.h"
#include "iree/hal/driver/amd/gpu/queue/aql/ring.h"
#include "iree/hal/driver/amd/gpu/queue/blit.h"
#include "iree/hal/driver/amd/gpu/queue/control_program.h"
#include "iree/hal/driver/amd/gpu/queue/program.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Concrete AQL executor. Resident ownership is shared with other protocols;
// native descriptor launches and ring materialization belong to AQL.
typedef struct iree_hal_amd_gpu_aql_program_pool_t {
  // Shared placement, causal instance ownership, and explicit trim registry.
  iree_hal_amd_gpu_program_pool_t storage;
  // One immutable descriptor/instruction allocation for all builtin kernels.
  iree_hal_amd_fabric_memory_t code;
  // Temporary initialization bytes, published before the device is exposed.
  iree_hal_amd_gpu_code_upload_t* code_upload;
  // Native descriptor for resident instance initialization and copy/fill.
  uint64_t blit;
  // Native descriptor for independent complete-DWORD replacements.
  uint64_t fixup;
  // Native descriptor combining ordinary replacements and static-count reads.
  uint64_t counts;
  // Native descriptor for write-only packet materialization.
  uint64_t materialize;
  // Native descriptor for tagged completion operands and private samples.
  uint64_t materialize_profile;
  // Native descriptor for execution-time grid/hidden-count publication.
  uint64_t indirect;
  // Scalar atomic descriptors indexed by instruction scope and integer width.
  uint64_t control[IREE_HAL_AMD_GPU_CONTROL_PROGRAM_COUNT];
  // Native one-workitem GWS initializer, zero for memory-barrier targets.
  uint64_t grid_sync;
} iree_hal_amd_gpu_aql_program_pool_t;

// Cold construction before queue creation. A failure leaves a deinitializable
// partial owner. Every queue and resident recording is joined before teardown.
iree_status_t iree_hal_amd_gpu_aql_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_aql_program_pool_t* out_pool);
void iree_hal_amd_gpu_aql_program_pool_deinitialize(
    iree_hal_amd_gpu_aql_program_pool_t* pool);

// One-workitem atomic dispatch. The caller owns ordering, argument publication,
// and completion exactly as for an application dispatch.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_control_program_packet(
    uint64_t kernel, uint64_t arguments, uint32_t header, uint64_t completion);

// Initial invocation upload. Static counts extend fixup to count_arguments_t;
// materialize follows the complete fixup record. Instrumented recordings use
// materialize_profile_arguments_t. The dense uint64_t value table follows
// the complete argument records. Later chunks upload only their materializer
// arguments. The first native chunk releases the table independently of later
// work.
typedef struct iree_hal_amd_gpu_aql_program_arguments_t {
  // Optional initial resident copy, including the first instance's fixups.
  iree_hal_amd_gpu_blit_arguments_t copy;
  // Invocation-wide disjoint replacements, applied once before any payload.
  iree_hal_amd_gpu_fixup_arguments_t fixup;
  // First native packet block and its exact reserved ring slots.
  iree_hal_amd_gpu_aql_materialize_arguments_t materialize;
} iree_hal_amd_gpu_aql_program_arguments_t;

// Complete initial argument extent before the dense uint64_t value table.
// Static counts and profiling each extend their owning argument record.
uint32_t iree_hal_amd_gpu_aql_program_argument_length(
    iree_hal_amd_gpu_aql_count_table_t counts, bool profiling);

// Instrumented materialization extends the last argument record by 16 bytes
// before the value table. Ordinary invocation uploads retain their compact ABI.
enum iree_hal_amd_gpu_aql_program_flag_bits_e {
  IREE_HAL_AMD_GPU_AQL_PROGRAM_FIRST = 1u << 0,
  IREE_HAL_AMD_GPU_AQL_PROGRAM_INITIALIZE = 1u << 1,
  IREE_HAL_AMD_GPU_AQL_PROGRAM_PROFILE = 1u << 2,
  // GPU-patched private command bytes are consumed by the command processor.
  IREE_HAL_AMD_GPU_AQL_PROGRAM_CONTROL = 1u << 3,
  // This invocation must replace resident operands before materialization.
  IREE_HAL_AMD_GPU_AQL_PROGRAM_FIXUP = 1u << 4,
};
typedef uint32_t iree_hal_amd_gpu_aql_program_flags_t;

// Captures preparation before publication changes the reserved instance's
// writer coordinate. A value_count of one contains only the instance base:
// those relocations remain valid after that instance's first invocation.
// Additional inputs require replacement on every invocation. Static counts
// are gathered once per invocation from their current source values.
iree_hal_amd_gpu_aql_program_flags_t iree_hal_amd_gpu_aql_program_prepare(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    iree_hal_amd_gpu_aql_count_table_t counts, uint32_t value_count,
    bool first);

// Counts host-owned prelude packets; block packets and the terminal barrier
// are separate. Emission consumes the same captured preparation flags.
uint32_t iree_hal_amd_gpu_aql_program_prefix_count(
    iree_hal_amd_gpu_aql_program_flags_t flags);

// Publishes a bounded immutable packet sequence from invocation-private upload
// storage. Dense inputs and fixups are already resolved. Completion operands
// contain producer-assigned profile ordinals, or zero for unsampled packets.
// samples is invocation-private storage at sample_address, or NULL/zero when
// capture is disabled. The caller retains both uploads through completion.
amdf_status_t iree_hal_amd_gpu_aql_program_emit_host(
    uint32_t packet_count, const void* upload,
    iree_hal_amd_gpu_aql_ring_t* ring,
    const iree_hal_amd_gpu_user_queue_t* storage, uint64_t position,
    uint64_t sample_address, iree_hal_amd_gpu_aql_signal_t* samples);

// Emits one fully reserved block's host prelude. Resident templates are copied
// exclusively by the GPU. upload is coherent native argument storage and its
// value table is already written. The caller establishes ownership before
// emission and publishes a terminal completion after the block's slots.
// A native release failure is terminal; accepted ownership stays with the
// caller, and the remaining headers are not published.
amdf_status_t iree_hal_amd_gpu_aql_program_emit(
    const iree_hal_amd_gpu_aql_program_pool_t* pool,
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    iree_hal_amd_gpu_aql_count_table_t counts, uint32_t block_offset,
    uint32_t packet_count, iree_hal_amd_gpu_aql_ring_t* ring,
    const iree_hal_amd_gpu_user_queue_t* storage, uint64_t position,
    uint64_t argument_address, uint64_t signal_address, void* upload,
    iree_hal_amd_gpu_aql_program_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_PROGRAM_H_
