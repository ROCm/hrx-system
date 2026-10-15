// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMANDS_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMANDS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native MEC packet writers for admitted PM4 programs. Each writer returns the
// next DWORD in caller-owned command storage. The caller has already
// established capacity, address alignment/accessibility, register and
// cache-control fields, and resource lifetime. Writers perform no validation,
// allocation, publication, or ownership transfer. Direct submission and
// recording use the same encodings.
//
// Target-specific launch and transition preparation owns register addresses and
// control values. In particular, ACQUIRE_MEM and RELEASE_MEM use different GCR
// layouts, and those layouts vary by target. These writers do not select cache
// policy, infer a barrier from an address, or select a queue transport.

// Emits 2 + value_count DWORDs to consecutive compute shader registers. Values
// are copied now. The nonempty interval is in the admitted SH register bank,
// whose base is 0x2c00. Uses ordinary SET_SH_REG without an indexed mode.
uint32_t* iree_hal_amd_gpu_pm4_emit_registers(uint32_t* words,
                                              uint32_t first_register,
                                              uint32_t value_count,
                                              const uint32_t* values);

// Emits five DWORDs, capturing three dimensions and a prepared initiator. The
// initiator selects workgroup or workitem units, start handling, partial
// groups, and the compiled wave size. Shader-visible geometry is prepared
// separately.
uint32_t* iree_hal_amd_gpu_pm4_emit_dispatch_direct(
    uint32_t* words, const uint32_t dimensions[3], uint32_t initiator);

// Emits four DWORDs using the MEC absolute-address form. The device reads three
// DWORD workgroup counts at the aligned address after preceding dependencies
// and transitions. The prepared initiator uses workgroup units. Argument
// storage and any shader-visible geometry remain owned through execution
// completion.
uint32_t* iree_hal_amd_gpu_pm4_emit_dispatch_indirect(
    uint32_t* words, uint64_t dimensions_address, uint32_t initiator);

// Emits two DWORDs waiting for preceding compute shader work to complete.
// This CS_PARTIAL_FLUSH establishes execution ordering, not cache visibility.
uint32_t* iree_hal_amd_gpu_pm4_emit_compute_flush(uint32_t* words);

// Emits seven DWORDs joining prior CP-DMA work. The zero-byte MEC DMA_DATA
// accesses no source or destination memory and leaves PFP-only fields clear.
// Shader completion and cache visibility require their own operations.
uint32_t* iree_hal_amd_gpu_pm4_emit_dma_flush(uint32_t* words);

// Emits eight DWORDs of whole-address-space ACQUIRE_MEM with the prepared
// target-specific GCR word. This performs cache work and does not by itself
// wait for prior shader execution or establish an external producer dependency.
uint32_t* iree_hal_amd_gpu_pm4_emit_acquire(uint32_t* words,
                                            uint32_t cache_control);

// Native unsigned comparisons on (memory & mask), with the reference value
// already restricted to that mask. These are packet function encodings, not
// HAL atomic comparison values.
typedef enum iree_hal_amd_gpu_pm4_comparison_e {
  IREE_HAL_AMD_GPU_PM4_COMPARISON_ALWAYS = 0,
  IREE_HAL_AMD_GPU_PM4_COMPARISON_LESS = 1,
  IREE_HAL_AMD_GPU_PM4_COMPARISON_LESS_OR_EQUAL = 2,
  IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL = 3,
  IREE_HAL_AMD_GPU_PM4_COMPARISON_NOT_EQUAL = 4,
  IREE_HAL_AMD_GPU_PM4_COMPARISON_GREATER_OR_EQUAL = 5,
  IREE_HAL_AMD_GPU_PM4_COMPARISON_GREATER = 6,
} iree_hal_amd_gpu_pm4_comparison_t;

// Emits seven DWORDs of MEC WAIT_REG_MEM against a DWORD-aligned address.
// The target's visibility and atomicity are already qualified for this queue.
// The backing and its expected generation remain valid through the last native
// reader, independently of producer completion. A wait establishes control
// ordering only; payload visibility requires its separate release/acquire.
uint32_t* iree_hal_amd_gpu_pm4_emit_wait32(
    uint32_t* words, uint64_t address, uint32_t reference, uint32_t mask,
    iree_hal_amd_gpu_pm4_comparison_t comparison);

// Emits nine DWORDs of MEC WAIT_REG_MEM64. The eight-byte-aligned target also
// has qualified native 64-bit read/write atomicity. Other ownership and
// visibility requirements are the same as the DWORD form.
uint32_t* iree_hal_amd_gpu_pm4_emit_wait64(
    uint32_t* words, uint64_t address, uint64_t reference, uint64_t mask,
    iree_hal_amd_gpu_pm4_comparison_t comparison);

// Emits eight DWORDs of RELEASE_MEM after preceding compute execution. Applies
// the prepared, unshifted release GCR field and writes a confirmed DWORD to the
// aligned destination through TC/L2. Zero cache_control requests no cache
// action. Readiness of this value releases only the resources covered by this
// tail; it does not prove consumption of later packets such as event
// notification.
uint32_t* iree_hal_amd_gpu_pm4_emit_release32(uint32_t* words,
                                              uint32_t cache_control,
                                              uint64_t target_address,
                                              uint32_t value);

// Emits the eight-DWORD immediate-64 form of the same release. The destination
// is eight-byte aligned and covers eight accessible bytes.
uint32_t* iree_hal_amd_gpu_pm4_emit_release64(uint32_t* words,
                                              uint32_t cache_control,
                                              uint64_t target_address,
                                              uint64_t value);

// Emits eight DWORDs of RELEASE_MEM with an interrupt after write confirmation.
// The destination, value and interrupt context are a previously admitted native
// event recipe. Its owner outlives all notification tails. The event is a wake
// hint; independent completion records remain authoritative.
uint32_t* iree_hal_amd_gpu_pm4_emit_notify32(uint32_t* words,
                                             uint32_t cache_control,
                                             uint64_t target_address,
                                             uint32_t value,
                                             uint32_t interrupt_context);

// Emits six DWORDs of confirmed 64-bit COPY_DATA through TC/L2. Both addresses
// are eight-byte aligned and cover eight accessible bytes. This is a command
// processor copy, not an atomic operation or a shader/cache completion barrier.
uint32_t* iree_hal_amd_gpu_pm4_emit_copy64(uint32_t* words,
                                           uint64_t source_address,
                                           uint64_t target_address);

// Emits the six-DWORD 32-bit copy form. Both addresses are DWORD aligned. The
// source includes target-qualified trailing readable backing because native
// fetches may read ahead; the result width does not bound the read footprint.
// Only four destination bytes are written. An arbitrary four-byte public
// buffer range does not establish the source-backing requirement.
uint32_t* iree_hal_amd_gpu_pm4_emit_copy32(uint32_t* words,
                                           uint64_t source_address,
                                           uint64_t target_address);

// Emits 4 + value_count DWORDs of confirmed, incrementing WRITE_DATA through
// TC/L2. Captures 1..16381 payload DWORDs into the command stream. The
// destination is DWORD aligned; values do not overlap the output command
// extent. This does not invalidate shader caches or complete earlier shader
// execution.
uint32_t* iree_hal_amd_gpu_pm4_emit_write(uint32_t* words,
                                          uint64_t target_address,
                                          uint32_t value_count,
                                          const uint32_t* values);

// Emits a NOP occupying 2..16385 DWORDs, initializing the entire extent.
// Recording and ring owners choose padding and account for it before calling
// the writer.
uint32_t* iree_hal_amd_gpu_pm4_emit_nop(uint32_t* words, uint32_t word_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMANDS_H_
