// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/device/support/asan.h"

// Replacement AMDGPU ASAN runtime bitcode linked into checked device programs.
// The HAL owns shadow memory, executable-global publication, and feedback
// decoding; this artifact adapts clang/ROCm's `__asan_*` hook ABI to that HAL
// contract.

#if defined(__clang__)
// The replacement ASAN runtime is linked as bitcode into instrumented programs.
// Its ABI globals and helper functions must keep their exact layout and must
// not recursively call back into the sanitizer hooks they implement.
#pragma clang attribute push(                           \
    __attribute__((disable_sanitizer_instrumentation)), \
    apply_to = any(function, variable(is_global)))
#endif  // defined(__clang__)

__attribute__((no_sanitize("address")))
[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_asan_config_t iree_asan_config = {0};

__attribute__((no_sanitize("address")))
[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_feedback_config_t iree_feedback_config = {0};

#define IREE_ASAN_SHADOW_POISONED 0xF7u
#define IREE_ASAN_SHADOW_HEAP_LEFT_REDZONE 0xFAu

static bool iree_asan_is_enabled(void) {
  return (iree_asan_config.flags & IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED) &&
         iree_asan_config.shadow_scale_shift < 63 &&
         iree_asan_config.shadow_size != 0;
}

static bool iree_asan_try_shadow_address(uint64_t address,
                                         uint64_t* out_shadow_address) {
  *out_shadow_address = 0;
  if (!iree_asan_is_enabled()) return false;
  if (address < iree_asan_config.application_window_base) return false;
  uint64_t application_offset =
      address - iree_asan_config.application_window_base;
  if (application_offset >= iree_asan_config.application_window_size) {
    return false;
  }
  uint64_t reservation_offset =
      application_offset >> iree_asan_config.shadow_scale_shift;
  if (reservation_offset >= iree_asan_config.shadow_size) return false;
  *out_shadow_address = iree_asan_config.shadow_base +
                        (address >> iree_asan_config.shadow_scale_shift);
  return true;
}

static void iree_asan_report(iree_hal_amdgpu_asan_access_kind_t access_kind,
                             uint64_t address, uint64_t access_size,
                             uint64_t shadow_address, uint64_t shadow_value) {
  (void)iree_hal_amdgpu_asan_report_access(&iree_feedback_config, access_kind,
                                           address, access_size, 0,
                                           shadow_address, shadow_value);
}

static void iree_asan_report_current_shadow(
    iree_hal_amdgpu_asan_access_kind_t access_kind, uint64_t address,
    uint64_t access_size) {
  uint64_t shadow_address = 0;
  uint64_t shadow_value = 0;
  if (iree_asan_try_shadow_address(address, &shadow_address)) {
    shadow_value = *(volatile uint8_t*)shadow_address;
  }
  iree_asan_report(access_kind, address, access_size, shadow_address,
                   shadow_value);
}

static uint64_t iree_asan_shadow_byte_application_base(
    uint64_t shadow_address) {
  return (shadow_address - iree_asan_config.shadow_base)
         << iree_asan_config.shadow_scale_shift;
}

static uint64_t iree_asan_shadow_granule_size(void) {
  return 1ull << iree_asan_config.shadow_scale_shift;
}

static uint64_t iree_asan_shadow_granule_mask(void) {
  return iree_asan_shadow_granule_size() - 1;
}

static uint64_t iree_asan_round_down_to_shadow_granule(uint64_t address) {
  return address & ~iree_asan_shadow_granule_mask();
}

static bool iree_asan_try_round_up_to_shadow_granule(uint64_t address,
                                                     uint64_t* out_address) {
  const uint64_t granule_mask = iree_asan_shadow_granule_mask();
  const uint64_t offset = address & granule_mask;
  if (offset == 0) {
    *out_address = address;
    return true;
  }
  const uint64_t adjustment = iree_asan_shadow_granule_size() - offset;
  if (adjustment > UINT64_MAX - address) return false;
  *out_address = address + adjustment;
  return true;
}

static bool iree_asan_shadow_value_is_magic(uint8_t shadow_value) {
  return shadow_value >= iree_asan_shadow_granule_size();
}

static uint8_t iree_asan_min_shadow_value(uint8_t lhs, uint8_t rhs) {
  return iree_asan_shadow_value_is_magic(lhs) || lhs < rhs ? lhs : rhs;
}

static uint8_t iree_asan_max_shadow_value(uint8_t lhs, uint8_t rhs) {
  return iree_asan_shadow_value_is_magic(lhs) || lhs < rhs ? rhs : lhs;
}

static bool iree_asan_is_shadow_value_poisoned(uint64_t access_end,
                                               uint64_t shadow_address,
                                               uint8_t shadow_value) {
  if (shadow_value == 0) return false;
  const uint64_t granule_size = iree_asan_shadow_granule_size();
  if (shadow_value >= granule_size) return true;

  const uint64_t granule_base =
      iree_asan_shadow_byte_application_base(shadow_address);
  const uint64_t granule_end = granule_base + granule_size - 1;
  const uint64_t checked_end =
      access_end < granule_end ? access_end : granule_end;
  if (checked_end < granule_base) return false;
  return checked_end - granule_base >= shadow_value;
}

static bool iree_asan_region_is_poisoned(uint64_t address, uint64_t size,
                                         uint64_t* out_fault_address,
                                         uint64_t* out_shadow_address,
                                         uint64_t* out_shadow_value) {
  *out_fault_address = address;
  *out_shadow_address = 0;
  *out_shadow_value = 0;
  if (size == 0 || !iree_asan_is_enabled()) return false;
  if (size > UINT64_MAX - address) {
    *out_shadow_value = IREE_ASAN_SHADOW_POISONED;
    return true;
  }

  const uint64_t access_end = address + size - 1;
  uint64_t first_shadow_address = 0;
  uint64_t last_shadow_address = 0;
  if (!iree_asan_try_shadow_address(address, &first_shadow_address) ||
      !iree_asan_try_shadow_address(access_end, &last_shadow_address)) {
    *out_shadow_value = IREE_ASAN_SHADOW_POISONED;
    return true;
  }

  for (uint64_t shadow_address = first_shadow_address;
       shadow_address <= last_shadow_address; ++shadow_address) {
    uint8_t shadow_value = *(volatile uint8_t*)shadow_address;
    if (!iree_asan_is_shadow_value_poisoned(access_end, shadow_address,
                                            shadow_value)) {
      continue;
    }

    *out_shadow_address = shadow_address;
    *out_shadow_value = shadow_value;
    uint64_t fault_address =
        iree_asan_shadow_byte_application_base(shadow_address);
    if (shadow_value != 0 && shadow_value < iree_asan_shadow_granule_size()) {
      fault_address += shadow_value;
    }
    if (fault_address < address) fault_address = address;
    *out_fault_address = fault_address;
    return true;
  }
  return false;
}

static void iree_asan_load(uint64_t address, uint64_t access_size) {
  uint64_t fault_address = 0;
  uint64_t shadow_address = 0;
  uint64_t shadow_value = 0;
  if (iree_asan_region_is_poisoned(address, access_size, &fault_address,
                                   &shadow_address, &shadow_value)) {
    iree_asan_report(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ, fault_address,
                     access_size, shadow_address, shadow_value);
  }
}

static void iree_asan_store(uint64_t address, uint64_t access_size) {
  uint64_t fault_address = 0;
  uint64_t shadow_address = 0;
  uint64_t shadow_value = 0;
  if (iree_asan_region_is_poisoned(address, access_size, &fault_address,
                                   &shadow_address, &shadow_value)) {
    iree_asan_report(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE, fault_address,
                     access_size, shadow_address, shadow_value);
  }
}

void __asan_load1(uint64_t address) { iree_asan_load(address, 1); }
void __asan_load2(uint64_t address) { iree_asan_load(address, 2); }
void __asan_load4(uint64_t address) { iree_asan_load(address, 4); }
void __asan_load8(uint64_t address) { iree_asan_load(address, 8); }
void __asan_load16(uint64_t address) { iree_asan_load(address, 16); }
void __asan_loadN(uint64_t address, uint64_t size) {
  iree_asan_load(address, size);
}

void __asan_load1_noabort(uint64_t address) { __asan_load1(address); }
void __asan_load2_noabort(uint64_t address) { __asan_load2(address); }
void __asan_load4_noabort(uint64_t address) { __asan_load4(address); }
void __asan_load8_noabort(uint64_t address) { __asan_load8(address); }
void __asan_load16_noabort(uint64_t address) { __asan_load16(address); }
void __asan_loadN_noabort(uint64_t address, uint64_t size) {
  __asan_loadN(address, size);
}

void __asan_store1(uint64_t address) { iree_asan_store(address, 1); }
void __asan_store2(uint64_t address) { iree_asan_store(address, 2); }
void __asan_store4(uint64_t address) { iree_asan_store(address, 4); }
void __asan_store8(uint64_t address) { iree_asan_store(address, 8); }
void __asan_store16(uint64_t address) { iree_asan_store(address, 16); }
void __asan_storeN(uint64_t address, uint64_t size) {
  iree_asan_store(address, size);
}

void __asan_store1_noabort(uint64_t address) { __asan_store1(address); }
void __asan_store2_noabort(uint64_t address) { __asan_store2(address); }
void __asan_store4_noabort(uint64_t address) { __asan_store4(address); }
void __asan_store8_noabort(uint64_t address) { __asan_store8(address); }
void __asan_store16_noabort(uint64_t address) { __asan_store16(address); }
void __asan_storeN_noabort(uint64_t address, uint64_t size) {
  __asan_storeN(address, size);
}

void __asan_report_load1(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ,
                                  address, 1);
}
void __asan_report_load2(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ,
                                  address, 2);
}
void __asan_report_load4(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ,
                                  address, 4);
}
void __asan_report_load8(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ,
                                  address, 8);
}
void __asan_report_load16(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ,
                                  address, 16);
}
void __asan_report_load_n(uint64_t address, uint64_t size) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_READ,
                                  address, size);
}

void __asan_report_load1_noabort(uint64_t address) {
  __asan_report_load1(address);
}
void __asan_report_load2_noabort(uint64_t address) {
  __asan_report_load2(address);
}
void __asan_report_load4_noabort(uint64_t address) {
  __asan_report_load4(address);
}
void __asan_report_load8_noabort(uint64_t address) {
  __asan_report_load8(address);
}
void __asan_report_load16_noabort(uint64_t address) {
  __asan_report_load16(address);
}
void __asan_report_load_n_noabort(uint64_t address, uint64_t size) {
  __asan_report_load_n(address, size);
}

void __asan_report_store1(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE,
                                  address, 1);
}
void __asan_report_store2(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE,
                                  address, 2);
}
void __asan_report_store4(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE,
                                  address, 4);
}
void __asan_report_store8(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE,
                                  address, 8);
}
void __asan_report_store16(uint64_t address) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE,
                                  address, 16);
}
void __asan_report_store_n(uint64_t address, uint64_t size) {
  iree_asan_report_current_shadow(IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE,
                                  address, size);
}

void __asan_report_store1_noabort(uint64_t address) {
  __asan_report_store1(address);
}
void __asan_report_store2_noabort(uint64_t address) {
  __asan_report_store2(address);
}
void __asan_report_store4_noabort(uint64_t address) {
  __asan_report_store4(address);
}
void __asan_report_store8_noabort(uint64_t address) {
  __asan_report_store8(address);
}
void __asan_report_store16_noabort(uint64_t address) {
  __asan_report_store16(address);
}
void __asan_report_store_n_noabort(uint64_t address, uint64_t size) {
  __asan_report_store_n(address, size);
}

uint64_t __asan_region_is_poisoned(uint64_t address, uint64_t size) {
  uint64_t fault_address = 0;
  uint64_t shadow_address = 0;
  uint64_t shadow_value = 0;
  return iree_asan_region_is_poisoned(address, size, &fault_address,
                                      &shadow_address, &shadow_value)
             ? fault_address
             : 0;
}

static bool iree_asan_try_memory_region_shadow_bounds(
    uint64_t address, uint64_t size, uint64_t* out_begin_shadow_address,
    uint64_t* out_end_shadow_address, uint8_t* out_begin_offset,
    uint8_t* out_end_offset) {
  *out_begin_shadow_address = 0;
  *out_end_shadow_address = 0;
  *out_begin_offset = 0;
  *out_end_offset = 0;
  if (size == 0 || !iree_asan_is_enabled()) return false;
  if (size > UINT64_MAX - address) return false;

  const uint64_t end_address = address + size;
  const uint64_t granule_mask = iree_asan_shadow_granule_mask();
  uint64_t last_shadow_address = 0;
  if (!iree_asan_try_shadow_address(address, out_begin_shadow_address) ||
      !iree_asan_try_shadow_address(end_address - 1, &last_shadow_address)) {
    return false;
  }

  *out_begin_offset = (uint8_t)(address & granule_mask);
  *out_end_offset = (uint8_t)(end_address & granule_mask);
  *out_end_shadow_address =
      *out_end_offset == 0 ? last_shadow_address + 1 : last_shadow_address;
  return true;
}

static void iree_asan_poison_memory_region(uint64_t address, uint64_t size) {
  uint64_t begin_shadow_address = 0;
  uint64_t end_shadow_address = 0;
  uint8_t begin_offset = 0;
  uint8_t end_offset = 0;
  if (!iree_asan_try_memory_region_shadow_bounds(
          address, size, &begin_shadow_address, &end_shadow_address,
          &begin_offset, &end_offset)) {
    return;
  }

  volatile uint8_t* begin_shadow = (volatile uint8_t*)begin_shadow_address;
  volatile uint8_t* end_shadow = (volatile uint8_t*)end_shadow_address;
  uint8_t begin_value = *begin_shadow;
  uint8_t end_value = end_offset != 0 ? *end_shadow : 0;

  if (begin_shadow == end_shadow) {
    if (begin_value != 0 && !iree_asan_shadow_value_is_magic(begin_value) &&
        begin_value <= end_offset) {
      *begin_shadow = begin_offset > 0 ? iree_asan_min_shadow_value(
                                             begin_value, begin_offset)
                                       : IREE_ASAN_SHADOW_POISONED;
    }
    return;
  }

  volatile uint8_t* full_shadow_begin = begin_shadow;
  if (begin_offset > 0) {
    *begin_shadow = begin_value == 0
                        ? begin_offset
                        : iree_asan_min_shadow_value(begin_value, begin_offset);
    ++full_shadow_begin;
  }
  for (volatile uint8_t* shadow = full_shadow_begin; shadow < end_shadow;
       ++shadow) {
    *shadow = IREE_ASAN_SHADOW_POISONED;
  }
  if (end_offset != 0 && end_value != 0 &&
      !iree_asan_shadow_value_is_magic(end_value) && end_value <= end_offset) {
    *end_shadow = IREE_ASAN_SHADOW_POISONED;
  }
}

static void iree_asan_unpoison_memory_region(uint64_t address, uint64_t size) {
  uint64_t begin_shadow_address = 0;
  uint64_t end_shadow_address = 0;
  uint8_t begin_offset = 0;
  uint8_t end_offset = 0;
  if (!iree_asan_try_memory_region_shadow_bounds(
          address, size, &begin_shadow_address, &end_shadow_address,
          &begin_offset, &end_offset)) {
    return;
  }
  (void)begin_offset;

  volatile uint8_t* begin_shadow = (volatile uint8_t*)begin_shadow_address;
  volatile uint8_t* end_shadow = (volatile uint8_t*)end_shadow_address;
  uint8_t begin_value = *begin_shadow;
  uint8_t end_value = end_offset != 0 ? *end_shadow : 0;

  if (begin_shadow == end_shadow) {
    if (begin_value != 0) {
      *begin_shadow = iree_asan_max_shadow_value(begin_value, end_offset);
    }
    return;
  }

  for (volatile uint8_t* shadow = begin_shadow; shadow < end_shadow; ++shadow) {
    *shadow = 0;
  }
  if (end_offset != 0 && end_value != 0) {
    *end_shadow = iree_asan_max_shadow_value(end_value, end_offset);
  }
}

void __asan_poison_region(uint64_t address, uint64_t size) {
  if (size == 0 || !iree_asan_is_enabled()) return;
  if (size > UINT64_MAX - address) return;

  const uint64_t granule_mask = iree_asan_shadow_granule_mask();
  if ((address & granule_mask) != 0) {
    const uint64_t aligned_address =
        iree_asan_round_down_to_shadow_granule(address);
    uint64_t shadow_address = 0;
    if (!iree_asan_try_shadow_address(aligned_address, &shadow_address)) {
      return;
    }
    *(volatile uint8_t*)shadow_address = (uint8_t)(address - aligned_address);
  }

  const uint64_t end_address =
      iree_asan_round_down_to_shadow_granule(address + size);
  uint64_t aligned_address = 0;
  if (!iree_asan_try_round_up_to_shadow_granule(address, &aligned_address)) {
    return;
  }
  if (end_address <= aligned_address) return;

  uint64_t first_shadow_address = 0;
  uint64_t last_shadow_address = 0;
  if (!iree_asan_try_shadow_address(aligned_address, &first_shadow_address) ||
      !iree_asan_try_shadow_address(end_address - 1, &last_shadow_address)) {
    return;
  }
  for (uint64_t shadow_address = first_shadow_address;
       shadow_address <= last_shadow_address; ++shadow_address) {
    *(volatile uint8_t*)shadow_address = IREE_ASAN_SHADOW_HEAP_LEFT_REDZONE;
  }
}

void __asan_poison_memory_region(const void* address, uint64_t size) {
  iree_asan_poison_memory_region((uint64_t)(uintptr_t)address, size);
}

void __asan_unpoison_memory_region(const void* address, uint64_t size) {
  iree_asan_unpoison_memory_region((uint64_t)(uintptr_t)address, size);
}

int __asan_address_is_poisoned(const void* address) {
  uint64_t fault_address = 0;
  uint64_t shadow_address = 0;
  uint64_t shadow_value = 0;
  return iree_asan_region_is_poisoned((uint64_t)(uintptr_t)address, 1,
                                      &fault_address, &shadow_address,
                                      &shadow_value);
}

void* __asan_memcpy(void* target, const void* source, uint64_t size) {
  __asan_loadN((uint64_t)source, size);
  __asan_storeN((uint64_t)target, size);
  uint8_t* target_bytes = (uint8_t*)target;
  const uint8_t* source_bytes = (const uint8_t*)source;
  for (uint64_t i = 0; i < size; ++i) {
    target_bytes[i] = source_bytes[i];
  }
  return target;
}

void* __asan_memmove(void* target, const void* source, uint64_t size) {
  __asan_loadN((uint64_t)source, size);
  __asan_storeN((uint64_t)target, size);
  uint8_t* target_bytes = (uint8_t*)target;
  const uint8_t* source_bytes = (const uint8_t*)source;
  if (target_bytes <= source_bytes) {
    for (uint64_t i = 0; i < size; ++i) {
      target_bytes[i] = source_bytes[i];
    }
  } else {
    for (uint64_t i = size; i > 0; --i) {
      target_bytes[i - 1] = source_bytes[i - 1];
    }
  }
  return target;
}

void* __asan_memset(void* target, int value, uint64_t size) {
  __asan_storeN((uint64_t)target, size);
  uint8_t* target_bytes = (uint8_t*)target;
  for (uint64_t i = 0; i < size; ++i) {
    target_bytes[i] = (uint8_t)value;
  }
  return target;
}

uint64_t __asan_load_cxx_array_cookie(uint64_t address) {
  __asan_load8(address);
  return *(uint64_t*)address;
}

void __asan_poison_cxx_array_cookie(uint64_t address) {
  __asan_store8(address);
}

uint64_t __asan_malloc_impl(uint64_t size, uint64_t alignment) {
  (void)size;
  (void)alignment;
  return 0;
}

void __asan_free_impl(uint64_t address, uint64_t size) {
  (void)address;
  (void)size;
}

void __asan_register_globals(uint64_t globals, uint64_t size) {
  (void)globals;
  (void)size;
}

void __asan_unregister_globals(uint64_t globals, uint64_t size) {
  (void)globals;
  (void)size;
}

void __asan_register_elf_globals(uint64_t flag, uint64_t start, uint64_t stop) {
  (void)flag;
  (void)start;
  (void)stop;
}

void __asan_unregister_elf_globals(uint64_t flag, uint64_t start,
                                   uint64_t stop) {
  (void)flag;
  (void)start;
  (void)stop;
}

void __asan_register_image_globals(uint64_t image_id) { (void)image_id; }
void __asan_unregister_image_globals(uint64_t image_id) { (void)image_id; }
void __asan_after_dynamic_init(void) {}
void __asan_before_dynamic_init(uint64_t module_name) { (void)module_name; }
void __asan_handle_no_return(void) {}
void __asan_init(void) {}
void __asan_version_mismatch_check_v8(void) {}
void __sanitizer_ptr_cmp(uint64_t lhs, uint64_t rhs) {
  (void)lhs;
  (void)rhs;
}
void __sanitizer_ptr_sub(uint64_t lhs, uint64_t rhs) {
  (void)lhs;
  (void)rhs;
}

void __ockl_dm_init_v1(uint64_t heap_address, uint64_t stack_address,
                       uint32_t heap_base, uint32_t initial_slab_count) {
  (void)heap_address;
  (void)stack_address;
  (void)heap_base;
  (void)initial_slab_count;
}
void __ockl_dm_trim(int* memory) { (void)memory; }
uint64_t __ockl_dm_nna(void) { return 0; }

#if defined(__clang__)
#pragma clang attribute pop
#endif  // defined(__clang__)
