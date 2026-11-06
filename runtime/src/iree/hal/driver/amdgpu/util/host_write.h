// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_UTIL_HOST_WRITE_H_
#define IREE_HAL_DRIVER_AMDGPU_UTIL_HOST_WRITE_H_

#include "iree/base/api.h"

#if defined(IREE_ARCH_X86_64)
#if defined(IREE_COMPILER_MSVC_COMPAT)
#include <intrin.h>
#else
#include <emmintrin.h>
#endif  // IREE_COMPILER_MSVC_COMPAT
#endif  // IREE_ARCH_X86_64

#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true if the host has a store fence suitable for ordering normal or
// write-combined host writes before a following device-visible publication.
static inline bool iree_hal_amdgpu_host_write_fence_is_supported(void) {
#if defined(IREE_ARCH_X86_64) || \
    (defined(IREE_ARCH_ARM_64) && defined(IREE_COMPILER_GCC_COMPAT))
  return true;
#else
  return false;
#endif  // IREE_ARCH_*
}

// Completes preceding host writes before a following device-visible
// publication such as an MMIO doorbell or HDP flush. C atomic ordering only
// orders CPU observers and does not establish this host-to-device boundary.
static inline void iree_hal_amdgpu_host_write_fence(void) {
#if defined(IREE_ARCH_X86_64)
  _mm_sfence();
#elif defined(IREE_ARCH_ARM_64) && defined(IREE_COMPILER_GCC_COMPAT)
  __asm__ __volatile__("dmb oshst" ::: "memory");
#else
  // Callers must not use this fallback for device-visible or write-combined
  // memory when host_write_fence_is_supported() is false.
  iree_atomic_thread_fence(iree_memory_order_seq_cst);
#endif  // IREE_ARCH_*
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_UTIL_HOST_WRITE_H_
