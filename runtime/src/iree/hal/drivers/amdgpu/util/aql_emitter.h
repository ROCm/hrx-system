// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AQL packet emission helpers. Pure functions that populate packet fields and
// return the header bits. They do NOT write the header — the caller commits
// it separately via iree_hal_amdgpu_aql_ring_commit(), which performs the
// atomic store-release that publishes the packet to the CP.
//
// This separation allows the caller to:
//   - Batch multiple packet commits before a single doorbell ring
//   - Control completion_signal assignment (epoch signal on last packet only)
//   - Populate kernarg memory between emission and commit
//
// All emitters zero reserved fields to prevent undefined behavior from stale
// ring data.
//
// Direct host-queue submissions currently set the BARRIER bit on every packet
// so one AQL queue behaves as a single in-order dependency chain.
// Command-buffer replay is more precise: it sets the BARRIER bit only at
// wait-prefix, execution-barrier, and final-completion boundaries. HAL queue
// submission order alone is not user-visible ordering; semaphore signal->wait
// edges define the visible DAG.
//
// Submission ordering contract (from kernarg_ring.h):
//   1. Reserve AQL ring slots (backpressure gate)
//   2. Allocate kernarg blocks (sizing invariant guarantees space)
//   3. Populate kernarg + packet fields (emit helpers)
//   4. Commit packet headers (atomic store-release)
//   5. Ring doorbell (once per batch)

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_EMITTER_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_EMITTER_H_

#include <stdbool.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Packet header controls shared by all AQL emitters.
//
// Direct host-queue submissions still use barrier+system-fence packets so one
// queue behaves as an in-order chain. Command-buffer replay can opt into
// non-barrier dispatch packets and only set the barrier bit at logical ordering
// boundaries.
typedef struct iree_hal_amdgpu_aql_packet_control_t {
  // True when the packet participates in AQL queue-order dependency chaining.
  bool has_barrier;
  // Acquire fence scope encoded in the packet header.
  iree_hsa_fence_scope_t acquire_fence_scope;
  // Release fence scope encoded in the packet header.
  iree_hsa_fence_scope_t release_fence_scope;
} iree_hal_amdgpu_aql_packet_control_t;

// Returns packet control with caller-selected barrier policy and scopes.
static inline iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_aql_packet_control(bool has_barrier,
                                   iree_hsa_fence_scope_t acquire_fence_scope,
                                   iree_hsa_fence_scope_t release_fence_scope) {
  iree_hal_amdgpu_aql_packet_control_t packet_control;
  packet_control.has_barrier = has_barrier;
  packet_control.acquire_fence_scope = acquire_fence_scope;
  packet_control.release_fence_scope = release_fence_scope;
  return packet_control;
}

// Returns packet control for a barrier packet with caller-selected scopes.
static inline iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_aql_packet_control_barrier(
    iree_hsa_fence_scope_t acquire_fence_scope,
    iree_hsa_fence_scope_t release_fence_scope) {
  return iree_hal_amdgpu_aql_packet_control(
      /*has_barrier=*/true, acquire_fence_scope, release_fence_scope);
}

// Returns the current host-queue packet policy: barrier + system-scope fences.
static inline iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_aql_packet_control_barrier_system(void) {
  return iree_hal_amdgpu_aql_packet_control_barrier(
      IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_SYSTEM);
}

// Builds the 16-bit packet header from |packet_type| and |packet_control|.
static inline uint16_t iree_hal_amdgpu_aql_make_header(
    iree_hsa_packet_type_t packet_type,
    iree_hal_amdgpu_aql_packet_control_t packet_control) {
  return (uint16_t)iree_hsa_make_packet_header(
      packet_type, packet_control.has_barrier,
      packet_control.acquire_fence_scope, packet_control.release_fence_scope);
}

// Populates a kernel dispatch packet and returns the 16-bit AQL header.
// The grid dimensions (setup field) are returned via |out_setup| for the
// caller to pass to iree_hal_amdgpu_aql_ring_commit().
//
// Does NOT write the header word — the caller commits it after all packet
// fields and kernarg memory are fully populated.
static inline uint16_t iree_hal_amdgpu_aql_emit_dispatch(
    iree_hsa_kernel_dispatch_packet_t* packet, uint64_t kernel_object,
    const void* kernarg_address, const uint16_t workgroup_size[3],
    const uint32_t grid_size[3], uint32_t private_segment_size,
    uint32_t group_segment_size,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  // Setup encodes the number of grid dimensions (always 3 for IREE).
  *out_setup = 3;

  packet->workgroup_size[0] = workgroup_size[0];
  packet->workgroup_size[1] = workgroup_size[1];
  packet->workgroup_size[2] = workgroup_size[2];
  packet->reserved0 = 0;
  packet->grid_size[0] = grid_size[0];
  packet->grid_size[1] = grid_size[1];
  packet->grid_size[2] = grid_size[2];
  packet->private_segment_size = private_segment_size;
  packet->group_segment_size = group_segment_size;
  packet->kernel_object = kernel_object;
  packet->kernarg_address = (void*)kernarg_address;
  packet->reserved2 = 0;
  packet->completion_signal = completion_signal;

  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
                                         packet_control);
}

// Inputs for one direct kernel dispatch packet.
//
// An all-zero |workgroup_cluster_size| selects an ordinary HSA kernel dispatch
// packet. A nontrivial positive shape selects the AMD extended kernel dispatch
// packet. Partial-zero and trivial 1x1x1 shapes are invalid.
typedef struct iree_hal_amdgpu_aql_dispatch_params_t {
  // Kernel object handle returned by the HSA executable loader.
  uint64_t kernel_object;
  // Kernel argument storage, which must remain live through completion.
  const void* kernarg_address;
  // Workgroup size in work-items. Every dimension must be nonzero.
  uint16_t workgroup_size[3];
  // Direct dispatch size in workgroups. Clustered dimensions must be nonzero;
  // ordinary dispatch permits the HAL no-op zero shape.
  uint32_t workgroup_count[3];
  // Workgroup cluster shape, or all zeroes for ordinary dispatch.
  uint32_t workgroup_cluster_size[3];
  // Private segment byte size per work-item.
  uint32_t private_segment_size;
  // Group segment byte size per workgroup.
  uint32_t group_segment_size;
  // Header barrier and fence policy.
  iree_hal_amdgpu_aql_packet_control_t packet_control;
  // Optional signal decremented when the dispatch completes.
  iree_hsa_signal_t completion_signal;
} iree_hal_amdgpu_aql_dispatch_params_t;

// Returns true when |params| requests an AMD extended clustered dispatch.
// Malformed partial-zero shapes also return true so validation rejects them
// instead of silently selecting an ordinary packet.
static inline bool iree_hal_amdgpu_aql_dispatch_uses_workgroup_clusters(
    const iree_hal_amdgpu_aql_dispatch_params_t* params) {
  return params->workgroup_cluster_size[0] != 0 ||
         params->workgroup_cluster_size[1] != 0 ||
         params->workgroup_cluster_size[2] != 0;
}

// Validates the geometry shared by ordinary and extended direct dispatch.
//
// Every workgroup size must be nonzero and each complete work-item grid must
// fit in the ordinary packet's u32 contract. Clustered dispatches additionally
// require nonzero workgroup counts, positive u8-compatible cluster dimensions,
// exact workgroup-count divisibility, and packet-compatible cluster counts.
// When non-NULL, |out_cluster_count| receives the validated extended packet
// cluster counts or all zeroes for an ordinary dispatch.
static inline iree_status_t iree_hal_amdgpu_aql_validate_dispatch_params(
    const iree_hal_amdgpu_aql_dispatch_params_t* params,
    uint32_t out_cluster_count[3]) {
  uint32_t cluster_count[3] = {0};
  const bool uses_workgroup_clusters =
      iree_hal_amdgpu_aql_dispatch_uses_workgroup_clusters(params);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(cluster_count); ++i) {
    if (IREE_UNLIKELY(params->workgroup_size[i] == 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch workgroup size dimension %u must be nonzero", (unsigned)i);
    }
    if (IREE_UNLIKELY(params->workgroup_count[i] >
                      UINT32_MAX / params->workgroup_size[i])) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch work-item grid dimension %u exceeds u32", (unsigned)i);
    }
    if (!uses_workgroup_clusters) continue;
    if (IREE_UNLIKELY(params->workgroup_count[i] == 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "extended dispatch workgroup count dimension %u must be nonzero",
          (unsigned)i);
    }
    if (IREE_UNLIKELY(params->workgroup_cluster_size[i] == 0 ||
                      params->workgroup_cluster_size[i] > UINT8_MAX)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "extended dispatch cluster size dimension %u must be a positive u8 "
          "value; got %u",
          (unsigned)i, (unsigned)params->workgroup_cluster_size[i]);
    }
    if (IREE_UNLIKELY(params->workgroup_count[i] %
                          params->workgroup_cluster_size[i] !=
                      0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "extended dispatch workgroup count dimension %u is not divisible "
          "by its cluster size",
          (unsigned)i);
    }
    cluster_count[i] =
        params->workgroup_count[i] / params->workgroup_cluster_size[i];
    if (IREE_UNLIKELY(i != 0 && cluster_count[i] > UINT16_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "extended dispatch cluster count dimension %u exceeds u16",
          (unsigned)i);
    }
  }
  if (uses_workgroup_clusters &&
      IREE_UNLIKELY(params->workgroup_cluster_size[0] == 1 &&
                    params->workgroup_cluster_size[1] == 1 &&
                    params->workgroup_cluster_size[2] == 1)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "extended dispatch requires a nontrivial workgroup cluster shape");
  }
  if (out_cluster_count) {
    memcpy(out_cluster_count, cluster_count, sizeof(cluster_count));
  }
  return iree_ok_status();
}

// Populates an already-validated direct dispatch packet body and returns its
// header/setup commit halfwords. |ordinary_packet| and |extended_packet| must
// refer to the same 64-byte AQL packet slot. The first packet dword remains
// untouched until the caller commits the returned halfwords.
static inline void iree_hal_amdgpu_aql_emplace_dispatch_packet(
    iree_hsa_kernel_dispatch_packet_t* ordinary_packet,
    iree_hsa_amd_ext_kernel_dispatch_packet_t* extended_packet,
    const iree_hal_amdgpu_aql_dispatch_params_t* params,
    const uint32_t cluster_count[3], uint16_t* out_header,
    uint16_t* out_setup) {
  if (!iree_hal_amdgpu_aql_dispatch_uses_workgroup_clusters(params)) {
    const uint32_t grid_size[3] = {
        params->workgroup_count[0] * params->workgroup_size[0],
        params->workgroup_count[1] * params->workgroup_size[1],
        params->workgroup_count[2] * params->workgroup_size[2],
    };
    *out_header = iree_hal_amdgpu_aql_emit_dispatch(
        ordinary_packet, params->kernel_object, params->kernarg_address,
        params->workgroup_size, grid_size, params->private_segment_size,
        params->group_segment_size, params->packet_control,
        params->completion_signal, out_setup);
    return;
  }

  extended_packet->workgroup_size[0] = params->workgroup_size[0];
  extended_packet->workgroup_size[1] = params->workgroup_size[1];
  extended_packet->workgroup_size[2] = params->workgroup_size[2];
  extended_packet->reserved0 = 0;
  extended_packet->cluster_count_x = cluster_count[0];
  extended_packet->cluster_count_y = (uint16_t)cluster_count[1];
  extended_packet->cluster_count_z = (uint16_t)cluster_count[2];
  extended_packet->cluster_size[0] = (uint8_t)params->workgroup_cluster_size[0];
  extended_packet->cluster_size[1] = (uint8_t)params->workgroup_cluster_size[1];
  extended_packet->cluster_size[2] = (uint8_t)params->workgroup_cluster_size[2];
  extended_packet->perf_hint = 0;
  extended_packet->private_segment_size = params->private_segment_size;
  extended_packet->group_segment_size = params->group_segment_size;
  extended_packet->kernel_object = params->kernel_object;
  extended_packet->kernarg_address = (void*)params->kernarg_address;
  extended_packet->dep_signal = iree_hsa_signal_null();
  extended_packet->completion_signal = params->completion_signal;

  *out_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, params->packet_control);
  *out_setup =
      (uint16_t)IREE_HSA_AMD_AQL_FORMAT_EXT_KERNEL_DISPATCH | (3u << 8);
}

// Validates and populates either an ordinary or AMD extended direct dispatch
// packet. On failure both packet views remain unchanged and commit outputs are
// zero.
static inline iree_status_t iree_hal_amdgpu_aql_emit_dispatch_packet(
    iree_hsa_kernel_dispatch_packet_t* ordinary_packet,
    iree_hsa_amd_ext_kernel_dispatch_packet_t* extended_packet,
    const iree_hal_amdgpu_aql_dispatch_params_t* params, uint16_t* out_header,
    uint16_t* out_setup) {
  *out_header = 0;
  *out_setup = 0;
  uint32_t cluster_count[3] = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_validate_dispatch_params(params, cluster_count));
  iree_hal_amdgpu_aql_emplace_dispatch_packet(ordinary_packet, extended_packet,
                                              params, cluster_count, out_header,
                                              out_setup);
  return iree_ok_status();
}

// Validates and populates an AMD extended kernel dispatch packet body.
//
// Workgroup counts must divide exactly by the cluster shape. The resulting Y
// and Z cluster counts must fit their 16-bit packet fields; X has a 32-bit
// field. The full work-item grid in every dimension must remain u32-compatible
// with the ordinary HAL dispatch contract.
//
// The explicit dependency signal and performance hints are zero because HAL
// queue sequencing supplies dependencies with preceding AQL packets. The first
// packet dword remains untouched until the caller passes |out_header| and
// |out_setup| to iree_hal_amdgpu_aql_ring_commit(). The upper commit halfword
// in |out_setup| contains both the AMD format byte and the dispatch setup byte.
// On failure the packet body is unchanged and both commit outputs are zero.
static inline iree_status_t iree_hal_amdgpu_aql_emit_extended_dispatch(
    iree_hsa_amd_ext_kernel_dispatch_packet_t* packet,
    const iree_hal_amdgpu_aql_dispatch_params_t* params, uint16_t* out_header,
    uint16_t* out_setup) {
  *out_header = 0;
  *out_setup = 0;
  uint32_t cluster_count[3] = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_aql_validate_dispatch_params(params, cluster_count));
  if (IREE_UNLIKELY(
          !iree_hal_amdgpu_aql_dispatch_uses_workgroup_clusters(params))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "extended dispatch requires a workgroup cluster shape");
  }
  iree_hal_amdgpu_aql_emplace_dispatch_packet(
      (iree_hsa_kernel_dispatch_packet_t*)packet, packet, params, cluster_count,
      out_header, out_setup);
  return iree_ok_status();
}

// Populates an AMD barrier-value packet and returns the 16-bit AQL header.
// The vendor packet's upper 16 commit bits carry AmdFormat/reserved instead of
// the normal dispatch setup field and are returned in |out_setup|.
//
// The barrier halts the CP until:
//   (signal_load(dep_signal) & mask) CONDITION compare_value
//
// For cross-queue epoch waits the typical usage is:
//   dep_signal = source_queue->epoch.signal
//   condition  = IREE_HSA_SIGNAL_CONDITION_LT
//   compare_value = EPOCH_INITIAL_VALUE - target_epoch + 1
//   mask = INT64_MAX (all non-sign bits)
static inline uint16_t iree_hal_amdgpu_aql_emit_barrier_value(
    iree_hsa_amd_barrier_value_packet_t* packet, iree_hsa_signal_t dep_signal,
    iree_hsa_signal_condition_t condition,
    iree_hsa_signal_value_t compare_value, iree_hsa_signal_value_t mask,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  // Keep the entire first dword (primary header + AmdFormat/reserved) untouched
  // until aql_ring_commit publishes it with release semantics.
  packet->reserved0 = 0;
  packet->signal = dep_signal;
  packet->value = compare_value;
  packet->mask = mask;
  packet->cond = (iree_hsa_signal_condition32_t)condition;
  packet->reserved1 = 0;
  packet->reserved2 = 0;
  packet->reserved3 = 0;
  packet->completion_signal = completion_signal;
  *out_setup = IREE_HSA_AMD_AQL_FORMAT_BARRIER_VALUE;

  // The primary header uses VENDOR_SPECIFIC packet type for AMD extensions.
  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC,
                                         packet_control);
}

// Populates a barrier-AND packet and returns the 16-bit AQL header.
// The barrier halts the CP until all non-null dependency signals reach 0.
// Up to 5 dependency signals are supported per packet.
static inline uint16_t iree_hal_amdgpu_aql_emit_barrier_and(
    iree_hsa_barrier_and_packet_t* packet, const iree_hsa_signal_t* dep_signals,
    uint32_t dep_count, iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal) {
  packet->reserved0 = 0;
  packet->reserved1 = 0;
  // Fill dependency signals, nulling any unused slots.
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(packet->dep_signal); ++i) {
    packet->dep_signal[i] =
        i < dep_count ? dep_signals[i] : iree_hsa_signal_null();
  }
  packet->reserved2 = 0;
  packet->completion_signal = completion_signal;

  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_BARRIER_AND,
                                         packet_control);
}

// Populates a no-op packet in |packet| and returns its 16-bit AQL header.
// Zero-dependency BARRIER_AND is the canonical "consume one slot, do no work"
// packet. Callers choose whether that no-op packet carries a barrier edge and
// what fence scopes it should use.
static inline uint16_t iree_hal_amdgpu_aql_emit_nop(
    iree_hsa_barrier_and_packet_t* packet,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal) {
  return iree_hal_amdgpu_aql_emit_barrier_and(packet, /*dep_signals=*/NULL,
                                              /*dep_count=*/0, packet_control,
                                              completion_signal);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_EMITTER_H_
