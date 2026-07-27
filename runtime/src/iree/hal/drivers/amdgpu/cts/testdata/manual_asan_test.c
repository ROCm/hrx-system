// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

enum manual_asan_hook_selector_e {
  MANUAL_ASAN_HOOK_LOAD1 = 1,
  MANUAL_ASAN_HOOK_LOAD2 = 2,
  MANUAL_ASAN_HOOK_LOAD4 = 3,
  MANUAL_ASAN_HOOK_LOAD8 = 4,
  MANUAL_ASAN_HOOK_LOAD16 = 5,
  MANUAL_ASAN_HOOK_LOADN = 6,
  MANUAL_ASAN_HOOK_STORE1 = 7,
  MANUAL_ASAN_HOOK_STORE2 = 8,
  MANUAL_ASAN_HOOK_STORE4 = 9,
  MANUAL_ASAN_HOOK_STORE8 = 10,
  MANUAL_ASAN_HOOK_STORE16 = 11,
  MANUAL_ASAN_HOOK_STOREN = 12,
  MANUAL_ASAN_HOOK_LOAD1_NOABORT = 13,
  MANUAL_ASAN_HOOK_LOAD2_NOABORT = 14,
  MANUAL_ASAN_HOOK_LOAD4_NOABORT = 15,
  MANUAL_ASAN_HOOK_LOAD8_NOABORT = 16,
  MANUAL_ASAN_HOOK_LOAD16_NOABORT = 17,
  MANUAL_ASAN_HOOK_LOADN_NOABORT = 18,
  MANUAL_ASAN_HOOK_STORE1_NOABORT = 19,
  MANUAL_ASAN_HOOK_STORE2_NOABORT = 20,
  MANUAL_ASAN_HOOK_STORE4_NOABORT = 21,
  MANUAL_ASAN_HOOK_STORE8_NOABORT = 22,
  MANUAL_ASAN_HOOK_STORE16_NOABORT = 23,
  MANUAL_ASAN_HOOK_STOREN_NOABORT = 24,
  MANUAL_ASAN_HOOK_REPORT_LOAD1 = 25,
  MANUAL_ASAN_HOOK_REPORT_LOAD2 = 26,
  MANUAL_ASAN_HOOK_REPORT_LOAD4 = 27,
  MANUAL_ASAN_HOOK_REPORT_LOAD8 = 28,
  MANUAL_ASAN_HOOK_REPORT_LOAD16 = 29,
  MANUAL_ASAN_HOOK_REPORT_LOAD_N = 30,
  MANUAL_ASAN_HOOK_REPORT_STORE1 = 31,
  MANUAL_ASAN_HOOK_REPORT_STORE2 = 32,
  MANUAL_ASAN_HOOK_REPORT_STORE4 = 33,
  MANUAL_ASAN_HOOK_REPORT_STORE8 = 34,
  MANUAL_ASAN_HOOK_REPORT_STORE16 = 35,
  MANUAL_ASAN_HOOK_REPORT_STORE_N = 36,
  MANUAL_ASAN_HOOK_REPORT_LOAD1_NOABORT = 37,
  MANUAL_ASAN_HOOK_REPORT_LOAD2_NOABORT = 38,
  MANUAL_ASAN_HOOK_REPORT_LOAD4_NOABORT = 39,
  MANUAL_ASAN_HOOK_REPORT_LOAD8_NOABORT = 40,
  MANUAL_ASAN_HOOK_REPORT_LOAD16_NOABORT = 41,
  MANUAL_ASAN_HOOK_REPORT_LOAD_N_NOABORT = 42,
  MANUAL_ASAN_HOOK_REPORT_STORE1_NOABORT = 43,
  MANUAL_ASAN_HOOK_REPORT_STORE2_NOABORT = 44,
  MANUAL_ASAN_HOOK_REPORT_STORE4_NOABORT = 45,
  MANUAL_ASAN_HOOK_REPORT_STORE8_NOABORT = 46,
  MANUAL_ASAN_HOOK_REPORT_STORE16_NOABORT = 47,
  MANUAL_ASAN_HOOK_REPORT_STORE_N_NOABORT = 48,
  MANUAL_ASAN_HOOK_POISON_REGION = 49,
  MANUAL_ASAN_HOOK_UNPOISON_REGION = 50,
};

extern void __asan_load1(uint64_t address);
extern void __asan_load2(uint64_t address);
extern void __asan_load4(uint64_t address);
extern void __asan_load8(uint64_t address);
extern void __asan_load16(uint64_t address);
extern void __asan_loadN(uint64_t address, uint64_t size);
extern void __asan_store1(uint64_t address);
extern void __asan_store2(uint64_t address);
extern void __asan_store4(uint64_t address);
extern void __asan_store8(uint64_t address);
extern void __asan_store16(uint64_t address);
extern void __asan_storeN(uint64_t address, uint64_t size);
extern void __asan_load1_noabort(uint64_t address);
extern void __asan_load2_noabort(uint64_t address);
extern void __asan_load4_noabort(uint64_t address);
extern void __asan_load8_noabort(uint64_t address);
extern void __asan_load16_noabort(uint64_t address);
extern void __asan_loadN_noabort(uint64_t address, uint64_t size);
extern void __asan_store1_noabort(uint64_t address);
extern void __asan_store2_noabort(uint64_t address);
extern void __asan_store4_noabort(uint64_t address);
extern void __asan_store8_noabort(uint64_t address);
extern void __asan_store16_noabort(uint64_t address);
extern void __asan_storeN_noabort(uint64_t address, uint64_t size);
extern void __asan_report_load1(uint64_t address);
extern void __asan_report_load2(uint64_t address);
extern void __asan_report_load4(uint64_t address);
extern void __asan_report_load8(uint64_t address);
extern void __asan_report_load16(uint64_t address);
extern void __asan_report_load_n(uint64_t address, uint64_t size);
extern void __asan_report_store1(uint64_t address);
extern void __asan_report_store2(uint64_t address);
extern void __asan_report_store4(uint64_t address);
extern void __asan_report_store8(uint64_t address);
extern void __asan_report_store16(uint64_t address);
extern void __asan_report_store_n(uint64_t address, uint64_t size);
extern void __asan_report_load1_noabort(uint64_t address);
extern void __asan_report_load2_noabort(uint64_t address);
extern void __asan_report_load4_noabort(uint64_t address);
extern void __asan_report_load8_noabort(uint64_t address);
extern void __asan_report_load16_noabort(uint64_t address);
extern void __asan_report_load_n_noabort(uint64_t address, uint64_t size);
extern void __asan_report_store1_noabort(uint64_t address);
extern void __asan_report_store2_noabort(uint64_t address);
extern void __asan_report_store4_noabort(uint64_t address);
extern void __asan_report_store8_noabort(uint64_t address);
extern void __asan_report_store16_noabort(uint64_t address);
extern void __asan_report_store_n_noabort(uint64_t address, uint64_t size);
extern void __asan_poison_region(uint64_t address, uint64_t size);
extern void __asan_unpoison_memory_region(const void* address, uint64_t size);

IREE_AMDGPU_ATTRIBUTE_KERNEL void export0(uint64_t* output, uint32_t selector,
                                          uint32_t dynamic_size,
                                          int32_t address_adjustment) {
  uint64_t address = (uint64_t)((uint8_t*)output + address_adjustment);
  switch (selector) {
    case MANUAL_ASAN_HOOK_LOAD1:
      __asan_load1(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD2:
      __asan_load2(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD4:
      __asan_load4(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD8:
      __asan_load8(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD16:
      __asan_load16(address);
      break;
    case MANUAL_ASAN_HOOK_LOADN:
      __asan_loadN(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_STORE1:
      __asan_store1(address);
      break;
    case MANUAL_ASAN_HOOK_STORE2:
      __asan_store2(address);
      break;
    case MANUAL_ASAN_HOOK_STORE4:
      __asan_store4(address);
      break;
    case MANUAL_ASAN_HOOK_STORE8:
      __asan_store8(address);
      break;
    case MANUAL_ASAN_HOOK_STORE16:
      __asan_store16(address);
      break;
    case MANUAL_ASAN_HOOK_STOREN:
      __asan_storeN(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_LOAD1_NOABORT:
      __asan_load1_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD2_NOABORT:
      __asan_load2_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD4_NOABORT:
      __asan_load4_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD8_NOABORT:
      __asan_load8_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_LOAD16_NOABORT:
      __asan_load16_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_LOADN_NOABORT:
      __asan_loadN_noabort(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_STORE1_NOABORT:
      __asan_store1_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_STORE2_NOABORT:
      __asan_store2_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_STORE4_NOABORT:
      __asan_store4_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_STORE8_NOABORT:
      __asan_store8_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_STORE16_NOABORT:
      __asan_store16_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_STOREN_NOABORT:
      __asan_storeN_noabort(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD1:
      __asan_report_load1(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD2:
      __asan_report_load2(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD4:
      __asan_report_load4(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD8:
      __asan_report_load8(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD16:
      __asan_report_load16(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD_N:
      __asan_report_load_n(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE1:
      __asan_report_store1(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE2:
      __asan_report_store2(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE4:
      __asan_report_store4(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE8:
      __asan_report_store8(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE16:
      __asan_report_store16(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE_N:
      __asan_report_store_n(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD1_NOABORT:
      __asan_report_load1_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD2_NOABORT:
      __asan_report_load2_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD4_NOABORT:
      __asan_report_load4_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD8_NOABORT:
      __asan_report_load8_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD16_NOABORT:
      __asan_report_load16_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_LOAD_N_NOABORT:
      __asan_report_load_n_noabort(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE1_NOABORT:
      __asan_report_store1_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE2_NOABORT:
      __asan_report_store2_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE4_NOABORT:
      __asan_report_store4_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE8_NOABORT:
      __asan_report_store8_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE16_NOABORT:
      __asan_report_store16_noabort(address);
      break;
    case MANUAL_ASAN_HOOK_REPORT_STORE_N_NOABORT:
      __asan_report_store_n_noabort(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_POISON_REGION:
      __asan_poison_region(address, dynamic_size);
      break;
    case MANUAL_ASAN_HOOK_UNPOISON_REGION:
      __asan_unpoison_memory_region((void*)address, dynamic_size);
      break;
    default:
      break;
  }
  output[0] = 0x4153414E53544F34ull;
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void export1(uint32_t address_lo,
                                          uint32_t address_hi,
                                          uint32_t selector,
                                          uint32_t dynamic_size) {
  uint64_t address = ((uint64_t)address_hi << 32) | address_lo;
  switch (selector) {
    case MANUAL_ASAN_HOOK_LOAD1:
      __asan_load1(address);
      break;
    case MANUAL_ASAN_HOOK_LOADN:
      __asan_loadN(address, dynamic_size);
      break;
    default:
      break;
  }
  volatile uint8_t value = *(volatile uint8_t*)(uintptr_t)address;
  (void)value;
}
