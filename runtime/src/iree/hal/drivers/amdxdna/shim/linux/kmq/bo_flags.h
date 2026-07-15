// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_SHIM_LINUX_KMQ_BO_FLAGS_H_
#define IREE_HAL_DRIVERS_AMDXDNA_SHIM_LINUX_KMQ_BO_FLAGS_H_

#include <cstdint>

namespace shim_xdna {

// Encoding of flags passed to the amdxdna BO allocation ioctl path. The bit
// layout matches the kernel/firmware ABI expected by the amdxdna KMQ driver.
struct shim_amdxdna_bo_flags {
  union {
    uint64_t all;  // [63-0]

    struct {
      uint32_t flags;      // [31-0]
      uint32_t extension;  // [63-32]
    };

    struct {
      uint16_t bank;    // [15-0]
      uint8_t slot;     // [23-16]
      uint8_t boflags;  // [31-24]

      uint32_t access : 2;   // [33-32]
      uint32_t dir : 2;      // [35-34]
      uint32_t use : 1;      // [36]
      uint32_t unused : 27;  // [63-35]
    };
  };
};

// BO flag bits: [15:0] bank index, [31:24] BO kind.
#define AMDXDNA_BO_FLAGS_MEMIDX_MASK (0xFFFFFFUL)
#define AMDXDNA_BO_FLAGS_NONE (0)
#define AMDXDNA_BO_FLAGS_CACHEABLE (1U << 24)
#define AMDXDNA_BO_FLAGS_KERNBUF (1U << 25)
#define AMDXDNA_BO_FLAGS_SGL (1U << 26)
#define AMDXDNA_BO_FLAGS_SVM (1U << 27)
#define AMDXDNA_BO_FLAGS_DEV_ONLY (1U << 28)
#define AMDXDNA_BO_FLAGS_HOST_ONLY (1U << 29)
#define AMDXDNA_BO_FLAGS_P2P (1U << 30)
#define AMDXDNA_BO_FLAGS_EXECBUF (1U << 31)

#define AMDXDNA_BO_ACCESS_LOCAL 0
#define AMDXDNA_BO_ACCESS_SHARED 1
#define AMDXDNA_BO_ACCESS_PROCESS 2
#define AMDXDNA_BO_ACCESS_HYBRID 3

#define AMDXDNA_BO_TRANSFER_READ (1U << 0)
#define AMDXDNA_BO_TRANSFER_WRITE (1U << 1)
#define AMDXDNA_BO_TRANSFER_READ_WRITE \
  (AMDXDNA_BO_TRANSFER_READ | AMDXDNA_BO_TRANSFER_WRITE)

// Distinguishes normal BOs from firmware/driver debug BOs.
#define AMDXDNA_BO_USE_NORMAL 0
#define AMDXDNA_BO_USE_DEBUG 1

enum amdxdna_bo_sync_direction {
  AMDXDNA_BO_SYNC_TO_DEVICE = 0,
  AMDXDNA_BO_SYNC_FROM_DEVICE,
};

}  // namespace shim_xdna

#endif  // IREE_HAL_DRIVERS_AMDXDNA_SHIM_LINUX_KMQ_BO_FLAGS_H_
