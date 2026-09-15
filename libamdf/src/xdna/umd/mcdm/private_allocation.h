// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_PRIVATE_ALLOCATION_H_
#define AMDF_SRC_XDNA_UMD_MCDM_PRIVATE_ALLOCATION_H_

#include <stdint.h>

#include "libamdf/src/xdna/umd/mcdm/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native operations performed while realizing one private allocation.
typedef uint32_t amdf_windows_xdna_private_allocation_flags_t;
enum amdf_windows_xdna_private_allocation_flag_bits_e {
  // Creates a resource handle around the allocation.
  AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_SHARED_RESOURCE = 1u << 0,
  // Establishes a stable XDNA device address and ordinary residency.
  AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS = 1u << 1,
};

// Exact private allocation parameters consumed by the installed XDNA UMD.
typedef struct amdf_windows_xdna_private_allocation_descriptor_t {
  // Logical byte length reported to the private allocation contract.
  uint64_t requested_byte_length;
  // Page-aligned physical allocation length.
  uint64_t allocation_byte_length;
  // Private allocation role discriminator.
  uint32_t type;
  // Driver placement policy.
  uint32_t policy;
  // Driver command-aperture and allocation flags.
  uint32_t xcl_flags;
  // Additional private allocation selector.
  uint32_t selector;
  // Native realization operations to perform.
  amdf_windows_xdna_private_allocation_flags_t flags;
} amdf_windows_xdna_private_allocation_descriptor_t;

// Retryable ownership state of one native private allocation.
typedef struct amdf_windows_xdna_private_allocation_t {
  // Device borrowed until destruction succeeds.
  amdf_xdna_umd_device_t* device;
  // KMT resource owning the allocation, or zero for an ungrouped allocation.
  D3DKMT_HANDLE resource;
  // KMT allocation handle.
  D3DKMT_HANDLE allocation;
  // Explicit host lock, or NULL while unlocked.
  void* host_pointer;
  // Stable XDNA virtual address, or zero when not requested.
  uint64_t device_address;
  // Context-qualified firmware base returned for an instruction aperture;
  // zero for allocation roles without a firmware address. Not a KMT GPU VA.
  uint64_t firmware_address;
  // Immutable descriptor captured during initialization.
  amdf_windows_xdna_private_allocation_descriptor_t descriptor;
  // Fence value of an accepted paging operation awaiting observation.
  uint64_t pending_paging_fence;
  // Device address returned by an accepted mapping operation.
  uint64_t pending_device_address;
  // Completed native realization phase.
  uint32_t realization_phase;
  // One while an accepted paging operation remains to be observed.
  uint32_t paging_operation_pending;
  // One when the accepted paging operation returned a valid result.
  uint32_t paging_operation_valid;
  // One while an explicit KMT host lock is owned.
  uint32_t host_lock_owned;
} amdf_windows_xdna_private_allocation_t;

// Initializes inert private-allocation state without performing native work.
// All arguments must be valid and `allocation` must be uninitialized storage;
// the descriptor is retained by value and the operation is infallible.
void amdf_windows_xdna_private_allocation_initialize(
    amdf_xdna_umd_device_t* device,
    const amdf_windows_xdna_private_allocation_descriptor_t* descriptor,
    amdf_windows_xdna_private_allocation_t* allocation);

// Realizes one initialized private allocation. A failed call may retain an
// accepted native allocation or paging operation; retry resumes that work and
// destruction releases every retained phase.
amdf_status_t amdf_windows_xdna_private_allocation_realize(
    amdf_windows_xdna_private_allocation_t* allocation);

// Establishes an explicit host lock on one private allocation.
amdf_status_t amdf_windows_xdna_private_allocation_lock(
    amdf_windows_xdna_private_allocation_t* allocation);

// Publishes host writes over one allocation-relative range.
amdf_status_t amdf_windows_xdna_private_allocation_publish(
    amdf_windows_xdna_private_allocation_t* allocation, uint64_t byte_offset,
    uint64_t byte_length);

// Releases one private allocation in retryable reverse acquisition order.
amdf_status_t amdf_windows_xdna_private_allocation_destroy(
    amdf_windows_xdna_private_allocation_t* allocation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MCDM_PRIVATE_ALLOCATION_H_
