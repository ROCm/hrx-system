// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_ATOMIC_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_ATOMIC_H_

#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/device/kernels.h"

#if !defined(IREE_AMDGPU_TARGET_DEVICE)
#include "iree/hal/atomic.h"
#endif  // !IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Atomic kernel ABI
//===----------------------------------------------------------------------===//

// Ordering and scope mode passed to the builtin atomic kernels. Values match
// the public HAL flags so host packet emplacement only needs to discard flags
// that are inapplicable to an operation.
typedef uint32_t iree_hal_amdgpu_device_atomic_mode_t;
typedef enum iree_hal_amdgpu_device_atomic_mode_bits_e {
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_NONE = 0u,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_ACQUIRE = 1u << 0,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_RELEASE = 1u << 1,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE = 1u << 2,
} iree_hal_amdgpu_device_atomic_mode_bits_e;

// Wait condition passed to the builtin atomic wait kernels.
typedef uint32_t iree_hal_amdgpu_device_atomic_wait_condition_t;
typedef enum iree_hal_amdgpu_device_atomic_wait_condition_e {
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_EQUAL = 0,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_NOT_EQUAL = 1,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL = 2,
} iree_hal_amdgpu_device_atomic_wait_condition_e;

// Operation passed to the builtin atomic RMW kernels.
typedef uint32_t iree_hal_amdgpu_device_atomic_rmw_operation_t;
typedef enum iree_hal_amdgpu_device_atomic_rmw_operation_e {
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_ADD = 0,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_SUBTRACT = 1,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_AND = 2,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_OR = 3,
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_XOR = 4,
} iree_hal_amdgpu_device_atomic_rmw_operation_e;

// Kernel arguments shared by the 32- and 64-bit atomic wait kernels.
typedef struct iree_hal_amdgpu_device_atomic_wait_kernargs_t {
  // Naturally aligned device pointer to the atomic word.
  const void* target_ptr;
  // Unsigned value compared against the masked observed word.
  uint64_t value;
  // Mask applied to each observed word before comparison.
  uint64_t mask;
  // Comparison that terminates the wait.
  iree_hal_amdgpu_device_atomic_wait_condition_t condition;
  // Acquire and scope mode; release is always absent.
  iree_hal_amdgpu_device_atomic_mode_t mode;
} iree_hal_amdgpu_device_atomic_wait_kernargs_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_atomic_wait_kernargs_t) == 32,
    "atomic wait kernargs must match the kernel ABI");
#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_SIZE \
  sizeof(iree_hal_amdgpu_device_atomic_wait_kernargs_t)
#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_ALIGNMENT \
  IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_device_atomic_wait_kernargs_t)

// Kernel arguments shared by the 32- and 64-bit atomic store kernels.
typedef struct iree_hal_amdgpu_device_atomic_store_kernargs_t {
  // Naturally aligned device pointer to the atomic word.
  void* target_ptr;
  // Unsigned value stored to the word.
  uint64_t value;
  // Release and scope mode; acquire is always absent.
  iree_hal_amdgpu_device_atomic_mode_t mode;
  // Reserved padding that must be zero.
  uint32_t reserved;
} iree_hal_amdgpu_device_atomic_store_kernargs_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_atomic_store_kernargs_t) == 24,
    "atomic store kernargs must match the kernel ABI");
#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_SIZE \
  sizeof(iree_hal_amdgpu_device_atomic_store_kernargs_t)
#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_ALIGNMENT \
  IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_device_atomic_store_kernargs_t)

// Kernel arguments shared by the 32- and 64-bit atomic RMW kernels.
typedef struct iree_hal_amdgpu_device_atomic_rmw_kernargs_t {
  // Naturally aligned device pointer to the atomic word.
  void* target_ptr;
  // Unsigned right-hand operand applied to the word.
  uint64_t operand;
  // Acquire, release, and scope mode.
  iree_hal_amdgpu_device_atomic_mode_t mode;
  // Read-modify-write operation to perform.
  iree_hal_amdgpu_device_atomic_rmw_operation_t operation;
} iree_hal_amdgpu_device_atomic_rmw_kernargs_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_atomic_rmw_kernargs_t) == 24,
    "atomic RMW kernargs must match the kernel ABI");
#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_SIZE \
  sizeof(iree_hal_amdgpu_device_atomic_rmw_kernargs_t)
#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_ALIGNMENT \
  IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_device_atomic_rmw_kernargs_t)

#if !defined(IREE_AMDGPU_TARGET_DEVICE)

// Initializes atomic wait kernargs for a validated operation.
void iree_hal_amdgpu_device_atomic_wait_initialize_kernargs(
    const void* target_ptr, iree_hal_atomic_wait_params_t params,
    iree_hal_amdgpu_device_atomic_wait_kernargs_t* out_kernargs);

// Initializes atomic store kernargs for a validated operation.
void iree_hal_amdgpu_device_atomic_store_initialize_kernargs(
    void* target_ptr, iree_hal_atomic_store_params_t params,
    iree_hal_amdgpu_device_atomic_store_kernargs_t* out_kernargs);

// Initializes atomic RMW kernargs for a validated operation.
void iree_hal_amdgpu_device_atomic_rmw_initialize_kernargs(
    void* target_ptr, iree_hal_atomic_rmw_params_t params,
    iree_hal_amdgpu_device_atomic_rmw_kernargs_t* out_kernargs);

// Populates a one-workitem atomic wait dispatch and its kernargs in
// already-reserved storage. The caller owns packet header commit,
// completion-signal assignment, and doorbell signaling.
//
// |params| must have passed iree_hal_atomic_wait_params_validate() and
// |target_ptr| must satisfy its natural width alignment.
void iree_hal_amdgpu_device_atomic_wait_emplace(
    const iree_hal_amdgpu_device_kernels_t* IREE_AMDGPU_RESTRICT kernels,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const void* target_ptr, iree_hal_atomic_wait_params_t params,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates a one-workitem atomic store dispatch and its kernargs in
// already-reserved storage. The caller owns packet header commit,
// completion-signal assignment, and doorbell signaling.
//
// |params| must have passed iree_hal_atomic_store_params_validate() and
// |target_ptr| must satisfy its natural width alignment.
void iree_hal_amdgpu_device_atomic_store_emplace(
    const iree_hal_amdgpu_device_kernels_t* IREE_AMDGPU_RESTRICT kernels,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, iree_hal_atomic_store_params_t params,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates a one-workitem atomic RMW dispatch and its kernargs in
// already-reserved storage. The caller owns packet header commit,
// completion-signal assignment, and doorbell signaling.
//
// |params| must have passed iree_hal_atomic_rmw_params_validate() and
// |target_ptr| must satisfy its natural width alignment.
void iree_hal_amdgpu_device_atomic_rmw_emplace(
    const iree_hal_amdgpu_device_kernels_t* IREE_AMDGPU_RESTRICT kernels,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, iree_hal_atomic_rmw_params_t params,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#endif  // !IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_ATOMIC_H_
