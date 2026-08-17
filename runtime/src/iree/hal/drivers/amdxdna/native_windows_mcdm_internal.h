// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_NATIVE_WINDOWS_MCDM_INTERNAL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_NATIVE_WINDOWS_MCDM_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "iree/hal/drivers/amdxdna/native.h"

// Internal transaction parser seam shared with focused unit tests.
bool iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
    const uint8_t* bytes, size_t total, uint32_t op_count,
    size_t queue_offset, uint32_t key, const uint8_t** out_dma,
    const uint8_t** out_ddr);

// Computes the touched word span for a transaction DMA descriptor. Arithmetic
// saturates at UINT64_MAX so malformed dimensions cannot wrap to a small range.
uint64_t iree_hal_amdxdna_native_windows_partial_elf_dma_span_words(
    const uint8_t* dma);

typedef struct iree_hal_amdxdna_native_windows_buffer_range_t {
  void* buffer;
  uint64_t offset;
  uint64_t length;
} iree_hal_amdxdna_native_windows_buffer_range_t;

// Sorts ranges by buffer and offset, drops empty ranges, and merges overlapping
// or adjacent ranges in place. Returns the number of ranges retained.
size_t iree_hal_amdxdna_native_windows_coalesce_buffer_ranges(
    iree_hal_amdxdna_native_windows_buffer_range_t* ranges,
    size_t range_count);

// Returns the initialized ERT packet size, including its header dword, if it
// fits in the backing allocation.
bool iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
    uint32_t payload_dword_count, size_t allocation_size,
    size_t* out_packet_bytes);

// Returns true when a buffer belongs to context-owned command staging and
// therefore cannot be materialized until the context exists.
bool iree_hal_amdxdna_native_windows_buffer_requires_context(
    iree_hal_amdxdna_native_buffer_c_type_t type);

// Reserves completion records 1..slot_capacity and returns their byte offsets
// in the 8-byte status ring. Reservation is all-or-nothing.
bool iree_hal_amdxdna_native_windows_reserve_completion_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t requested_count,
    size_t start_slot, uint32_t* out_slot_offsets, size_t* out_next_slot);

// Releases a complete reservation. Invalid, duplicate, or already-free
// offsets leave the reservation table unchanged.
bool iree_hal_amdxdna_native_windows_release_completion_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t slot_count,
    const uint32_t* slot_offsets);

// Returns true when no completion record is owned by an active submission.
bool iree_hal_amdxdna_native_windows_completion_slots_are_free(
    const uint8_t* slots_in_use, size_t slot_capacity);

// Reserves the first contiguous run of persistent command-code slots.
// Reservation is all-or-nothing.
bool iree_hal_amdxdna_native_windows_reserve_code_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t requested_count,
    size_t* out_first_slot);

// Releases one complete contiguous run. Invalid or already-free ranges leave
// the allocation table unchanged.
bool iree_hal_amdxdna_native_windows_release_code_slots(
    uint8_t* slots_in_use, size_t slot_capacity, size_t first_slot,
    size_t slot_count);

// Returns one past the highest live slot, or zero when no slots are live.
size_t iree_hal_amdxdna_native_windows_code_slot_high_watermark(
    const uint8_t* slots_in_use, size_t slot_capacity);

#endif  // IREE_HAL_DRIVERS_AMDXDNA_NATIVE_WINDOWS_MCDM_INTERNAL_H_
