// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/atomic.h"

#include "iree/hal/drivers/amdgpu/device/support/common.h"

#if !defined(IREE_AMDGPU_TARGET_DEVICE)

#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(lhs, rhs, message) \
  IREE_AMDGPU_STATIC_ASSERT((uint32_t)(lhs) == (uint32_t)(rhs), message)

IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_ACQUIRE, IREE_HAL_ATOMIC_FLAG_ACQUIRE,
    "device ABI must match HAL acquire flag");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_RELEASE, IREE_HAL_ATOMIC_FLAG_RELEASE,
    "device ABI must match HAL release flag");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE,
    IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
    "device ABI must match HAL system-scope flag");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_EQUAL,
    IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
    "device ABI must match HAL wait conditions");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
    IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
    "device ABI must match HAL wait conditions");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
    IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
    "device ABI must match HAL wait conditions");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_ADD,
    IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
    "device ABI must match HAL RMW operations");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_SUBTRACT,
    IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
    "device ABI must match HAL RMW operations");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_AND,
    IREE_HAL_ATOMIC_RMW_OPERATION_AND,
    "device ABI must match HAL RMW operations");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_OR,
    IREE_HAL_ATOMIC_RMW_OPERATION_OR,
    "device ABI must match HAL RMW operations");
IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE(
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_XOR,
    IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
    "device ABI must match HAL RMW operations");

#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_ASSERT_ABI_VALUE

static const iree_hal_amdgpu_device_kernel_args_t*
iree_hal_amdgpu_device_atomic_wait_kernel(
    const iree_hal_amdgpu_device_kernels_t* kernels,
    iree_hal_atomic_width_t width) {
  return width == IREE_HAL_ATOMIC_WIDTH_32
             ? &kernels->iree_hal_amdgpu_device_atomic_wait_x32
             : &kernels->iree_hal_amdgpu_device_atomic_wait_x64;
}

static const iree_hal_amdgpu_device_kernel_args_t*
iree_hal_amdgpu_device_atomic_store_kernel(
    const iree_hal_amdgpu_device_kernels_t* kernels,
    iree_hal_atomic_width_t width) {
  return width == IREE_HAL_ATOMIC_WIDTH_32
             ? &kernels->iree_hal_amdgpu_device_atomic_store_x32
             : &kernels->iree_hal_amdgpu_device_atomic_store_x64;
}

static const iree_hal_amdgpu_device_kernel_args_t*
iree_hal_amdgpu_device_atomic_rmw_kernel(
    const iree_hal_amdgpu_device_kernels_t* kernels,
    iree_hal_atomic_width_t width) {
  return width == IREE_HAL_ATOMIC_WIDTH_32
             ? &kernels->iree_hal_amdgpu_device_atomic_rmw_x32
             : &kernels->iree_hal_amdgpu_device_atomic_rmw_x64;
}

static void iree_hal_amdgpu_device_atomic_emplace_dispatch(
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    iree_hsa_kernel_dispatch_packet_t* dispatch_packet, void* kernarg_ptr) {
  const uint32_t workgroup_count[3] = {1, 1, 1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      kernel_args, workgroup_count, /*dynamic_workgroup_local_memory=*/0,
      dispatch_packet, kernarg_ptr);
}

void iree_hal_amdgpu_device_atomic_wait_emplace(
    const iree_hal_amdgpu_device_kernels_t* IREE_AMDGPU_RESTRICT kernels,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const void* target_ptr, iree_hal_atomic_wait_params_t params,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_device_atomic_wait_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_atomic_wait_kernargs_t*)kernarg_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->value = params.value;
  kernargs->mask = params.mask;
  kernargs->condition = params.condition;
  kernargs->mode = params.flags & (IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                                   IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
  iree_hal_amdgpu_device_atomic_emplace_dispatch(
      iree_hal_amdgpu_device_atomic_wait_kernel(kernels, params.width),
      dispatch_packet, kernarg_ptr);
}

void iree_hal_amdgpu_device_atomic_store_emplace(
    const iree_hal_amdgpu_device_kernels_t* IREE_AMDGPU_RESTRICT kernels,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, iree_hal_atomic_store_params_t params,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_device_atomic_store_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_atomic_store_kernargs_t*)kernarg_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->value = params.value;
  kernargs->mode = params.flags & (IREE_HAL_ATOMIC_FLAG_RELEASE |
                                   IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
  kernargs->reserved = 0;
  iree_hal_amdgpu_device_atomic_emplace_dispatch(
      iree_hal_amdgpu_device_atomic_store_kernel(kernels, params.width),
      dispatch_packet, kernarg_ptr);
}

void iree_hal_amdgpu_device_atomic_rmw_emplace(
    const iree_hal_amdgpu_device_kernels_t* IREE_AMDGPU_RESTRICT kernels,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, iree_hal_atomic_rmw_params_t params,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_device_atomic_rmw_kernargs_t* kernargs =
      (iree_hal_amdgpu_device_atomic_rmw_kernargs_t*)kernarg_ptr;
  kernargs->target_ptr = target_ptr;
  kernargs->operand = params.operand;
  kernargs->mode = params.flags;
  kernargs->operation = params.operation;
  iree_hal_amdgpu_device_atomic_emplace_dispatch(
      iree_hal_amdgpu_device_atomic_rmw_kernel(kernels, params.width),
      dispatch_packet, kernarg_ptr);
}

#endif  // !IREE_AMDGPU_TARGET_DEVICE

#if defined(IREE_AMDGPU_TARGET_DEVICE)

#if !__has_extension(clang_atomic_attributes)
#error "AMDGPU atomic fallbacks require Clang atomic memory attributes"
#endif

// Allows atomic operations in the attributed statement to target any memory
// reachable through a HAL buffer, including fine-grained and peer memory.
#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_GENERIC_MEMORY \
  [[clang::atomic(fine_grained_memory, remote_memory)]]

#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_LOOP(value_type, atomic_suffix,  \
                                                condition, scope)           \
  do {                                                                      \
    for (;;) {                                                              \
      const value_type observed = iree_amdgpu_scoped_atomic_load(           \
          (const iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, \
          iree_amdgpu_memory_order_relaxed, scope);                         \
      if (condition) break;                                                 \
      iree_amdgpu_yield();                                                  \
    }                                                                       \
  } while (0)

#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CASES(value_type, atomic_suffix,   \
                                                 scope_bit, scope)            \
  case (IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_EQUAL << 1) | scope_bit: \
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_LOOP(                                  \
        value_type, atomic_suffix, (observed & mask) == value, scope);        \
    break;                                                                    \
  case (IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_NOT_EQUAL << 1) |        \
      scope_bit:                                                              \
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_LOOP(                                  \
        value_type, atomic_suffix, (observed & mask) != value, scope);        \
    break;                                                                    \
  case (IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL   \
        << 1) |                                                               \
      scope_bit:                                                              \
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_LOOP(                                  \
        value_type, atomic_suffix, (observed & mask) >= value, scope);        \
    break

#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_WAIT(name, value_type,         \
                                                  atomic_suffix)            \
  IREE_AMDGPU_ATTRIBUTE_KERNEL void name(                                   \
      const value_type* target_ptr, uint64_t value_u64, uint64_t mask_u64,  \
      iree_hal_amdgpu_device_atomic_wait_condition_t condition,             \
      iree_hal_amdgpu_device_atomic_mode_t mode) {                          \
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_GENERIC_MEMORY {                          \
      const value_type value = (value_type)value_u64;                       \
      const value_type mask = (value_type)mask_u64;                         \
      const uint32_t selector =                                             \
          (condition << 1) |                                                \
          ((mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE) != 0);  \
      switch (selector) {                                                   \
        IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CASES(                           \
            value_type, atomic_suffix, 0, iree_amdgpu_memory_scope_device); \
        IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CASES(                           \
            value_type, atomic_suffix, 1, iree_amdgpu_memory_scope_system); \
        default:                                                            \
          __builtin_unreachable();                                          \
      }                                                                     \
      if (mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_ACQUIRE) {              \
        if (mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE) {       \
          iree_amdgpu_scoped_atomic_thread_fence(                           \
              iree_amdgpu_memory_order_acquire,                             \
              iree_amdgpu_memory_scope_system);                             \
        } else {                                                            \
          iree_amdgpu_scoped_atomic_thread_fence(                           \
              iree_amdgpu_memory_order_acquire,                             \
              iree_amdgpu_memory_scope_device);                             \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  }

IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_WAIT(
    iree_hal_amdgpu_device_atomic_wait_x32, uint32_t, uint32)
IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_WAIT(
    iree_hal_amdgpu_device_atomic_wait_x64, uint64_t, uint64)

#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_WAIT
#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CASES
#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_LOOP

#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_STORE(name, value_type,           \
                                                   atomic_suffix)              \
  IREE_AMDGPU_ATTRIBUTE_KERNEL void name(                                      \
      value_type* target_ptr, uint64_t value_u64,                              \
      iree_hal_amdgpu_device_atomic_mode_t mode, uint32_t reserved) {          \
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_GENERIC_MEMORY {                             \
      (void)reserved;                                                          \
      const value_type value = (value_type)value_u64;                          \
      const bool is_system_scope =                                             \
          mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE;              \
      if (mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_RELEASE) {                 \
        if (is_system_scope) {                                                 \
          iree_amdgpu_scoped_atomic_thread_fence(                              \
              iree_amdgpu_memory_order_release,                                \
              iree_amdgpu_memory_scope_system);                                \
        } else {                                                               \
          iree_amdgpu_scoped_atomic_thread_fence(                              \
              iree_amdgpu_memory_order_release,                                \
              iree_amdgpu_memory_scope_device);                                \
        }                                                                      \
      }                                                                        \
      if (is_system_scope) {                                                   \
        iree_amdgpu_scoped_atomic_store(                                       \
            (iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, value, \
            iree_amdgpu_memory_order_relaxed,                                  \
            iree_amdgpu_memory_scope_system);                                  \
      } else {                                                                 \
        iree_amdgpu_scoped_atomic_store(                                       \
            (iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, value, \
            iree_amdgpu_memory_order_relaxed,                                  \
            iree_amdgpu_memory_scope_device);                                  \
      }                                                                        \
    }                                                                          \
  }

IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_STORE(
    iree_hal_amdgpu_device_atomic_store_x32, uint32_t, uint32)
IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_STORE(
    iree_hal_amdgpu_device_atomic_store_x64, uint64_t, uint64)

#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_STORE

#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_CASES(value_type, atomic_suffix,     \
                                                scope)                         \
  switch (operation) {                                                         \
    case IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_ADD:                      \
      (void)iree_amdgpu_scoped_atomic_fetch_add(                               \
          (iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, operand, \
          iree_amdgpu_memory_order_relaxed, scope);                            \
      break;                                                                   \
    case IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_SUBTRACT:                 \
      (void)iree_amdgpu_scoped_atomic_fetch_sub(                               \
          (iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, operand, \
          iree_amdgpu_memory_order_relaxed, scope);                            \
      break;                                                                   \
    case IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_AND:                      \
      (void)iree_amdgpu_scoped_atomic_fetch_and(                               \
          (iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, operand, \
          iree_amdgpu_memory_order_relaxed, scope);                            \
      break;                                                                   \
    case IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_OR:                       \
      (void)iree_amdgpu_scoped_atomic_fetch_or(                                \
          (iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, operand, \
          iree_amdgpu_memory_order_relaxed, scope);                            \
      break;                                                                   \
    case IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_XOR:                      \
      (void)iree_amdgpu_scoped_atomic_fetch_xor(                               \
          (iree_amdgpu_scoped_atomic_##atomic_suffix##_t*)target_ptr, operand, \
          iree_amdgpu_memory_order_relaxed, scope);                            \
      break;                                                                   \
    default:                                                                   \
      __builtin_unreachable();                                                 \
  }

#define IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_RMW(name, value_type,       \
                                                 atomic_suffix)          \
  IREE_AMDGPU_ATTRIBUTE_KERNEL void name(                                \
      value_type* target_ptr, uint64_t operand_u64,                      \
      iree_hal_amdgpu_device_atomic_mode_t mode,                         \
      iree_hal_amdgpu_device_atomic_rmw_operation_t operation) {         \
    IREE_HAL_AMDGPU_DEVICE_ATOMIC_GENERIC_MEMORY {                       \
      const value_type operand = (value_type)operand_u64;                \
      const bool is_system_scope =                                       \
          mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE;        \
      if (mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_RELEASE) {           \
        if (is_system_scope) {                                           \
          iree_amdgpu_scoped_atomic_thread_fence(                        \
              iree_amdgpu_memory_order_release,                          \
              iree_amdgpu_memory_scope_system);                          \
        } else {                                                         \
          iree_amdgpu_scoped_atomic_thread_fence(                        \
              iree_amdgpu_memory_order_release,                          \
              iree_amdgpu_memory_scope_device);                          \
        }                                                                \
      }                                                                  \
      if (is_system_scope) {                                             \
        IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_CASES(                         \
            value_type, atomic_suffix, iree_amdgpu_memory_scope_system); \
      } else {                                                           \
        IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_CASES(                         \
            value_type, atomic_suffix, iree_amdgpu_memory_scope_device); \
      }                                                                  \
      if (mode & IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_ACQUIRE) {           \
        if (is_system_scope) {                                           \
          iree_amdgpu_scoped_atomic_thread_fence(                        \
              iree_amdgpu_memory_order_acquire,                          \
              iree_amdgpu_memory_scope_system);                          \
        } else {                                                         \
          iree_amdgpu_scoped_atomic_thread_fence(                        \
              iree_amdgpu_memory_order_acquire,                          \
              iree_amdgpu_memory_scope_device);                          \
        }                                                                \
      }                                                                  \
    }                                                                    \
  }

IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_RMW(iree_hal_amdgpu_device_atomic_rmw_x32,
                                         uint32_t, uint32)
IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_RMW(iree_hal_amdgpu_device_atomic_rmw_x64,
                                         uint64_t, uint64)

#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_DEFINE_RMW
#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_CASES
#undef IREE_HAL_AMDGPU_DEVICE_ATOMIC_GENERIC_MEMORY

#endif  // IREE_AMDGPU_TARGET_DEVICE
