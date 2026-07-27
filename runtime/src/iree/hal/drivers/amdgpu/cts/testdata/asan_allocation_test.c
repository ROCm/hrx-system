// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

enum asan_allocation_hook_selector_e {
  ASAN_ALLOCATION_HOOK_LOAD1 = 1,
  ASAN_ALLOCATION_HOOK_LOAD2 = 2,
  ASAN_ALLOCATION_HOOK_REPORT_LOAD_N = 3,
  ASAN_ALLOCATION_HOOK_POISON_REGION = 4,
  ASAN_ALLOCATION_HOOK_UNPOISON_REGION = 5,
};

extern void __asan_load1(uint64_t address);
extern void __asan_load2(uint64_t address);
extern void __asan_report_load_n(uint64_t address, uint64_t size);
extern void __asan_poison_region(uint64_t address, uint64_t size);
extern void __asan_unpoison_memory_region(const void* address, uint64_t size);

IREE_AMDGPU_ATTRIBUTE_KERNEL void export0(uint32_t* output, uint32_t selector,
                                          uint32_t dynamic_size,
                                          int32_t address_adjustment) {
  uint64_t address = (uint64_t)((uint8_t*)output + address_adjustment);
  switch (selector) {
    case ASAN_ALLOCATION_HOOK_LOAD1:
      __asan_load1(address);
      break;
    case ASAN_ALLOCATION_HOOK_LOAD2:
      __asan_load2(address);
      break;
    case ASAN_ALLOCATION_HOOK_REPORT_LOAD_N:
      __asan_report_load_n(address, dynamic_size);
      break;
    case ASAN_ALLOCATION_HOOK_POISON_REGION:
      __asan_poison_region(address, dynamic_size);
      break;
    case ASAN_ALLOCATION_HOOK_UNPOISON_REGION:
      __asan_unpoison_memory_region((const void*)address, dynamic_size);
      break;
    default:
      break;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void export1(uint32_t address_lo,
                                          uint32_t address_hi,
                                          uint32_t selector,
                                          uint32_t dynamic_size) {
  uint64_t address = ((uint64_t)address_hi << 32) | (uint64_t)address_lo;
  switch (selector) {
    case ASAN_ALLOCATION_HOOK_LOAD1:
      __asan_load1(address);
      break;
    case ASAN_ALLOCATION_HOOK_LOAD2:
      __asan_load2(address);
      break;
    case ASAN_ALLOCATION_HOOK_REPORT_LOAD_N:
      __asan_report_load_n(address, dynamic_size);
      break;
    case ASAN_ALLOCATION_HOOK_POISON_REGION:
      __asan_poison_region(address, dynamic_size);
      break;
    case ASAN_ALLOCATION_HOOK_UNPOISON_REGION:
      __asan_unpoison_memory_region((const void*)address, dynamic_size);
      break;
    default:
      break;
  }
}
