// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low helpers for AMDGPU global memory operations.
//
// Runtime producer paths such as feedback packets, host signals, sanitizer
// state, host calls, and future device-side queue producers all need the same
// target-specific memory policy. This file owns those cache attrs and explicit
// ordering packets.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SYSTEM_MEMORY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SYSTEM_MEMORY_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"
#include "loom/ops/cache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Returns true when the descriptor set provides every packet required to emit
// release ordering for prior global-memory accesses.
bool loom_amdgpu_system_memory_release_ordering_available(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when the descriptor set provides every packet required to emit
// acquire ordering for later global-memory accesses.
bool loom_amdgpu_system_memory_acquire_ordering_available(
    const loom_low_descriptor_set_t* descriptor_set);

// Flags controlling system-memory loads.
typedef uint32_t loom_amdgpu_system_memory_load_flags_t;

enum loom_amdgpu_system_memory_load_flag_bits_e {
  // No additional ordering is emitted after the load.
  LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE = 0u,
  // Emits acquire ordering after the vector-memory load.
  LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE = 1u << 0,
};

// Builds a 32-bit integer descriptor attr.
iree_status_t loom_amdgpu_system_memory_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr);

// Builds a byte offset descriptor attr.
iree_status_t loom_amdgpu_system_memory_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset, loom_named_attr_t* out_attr);

// Materializes |base_address| plus |byte_offset| as an SGPRx2 address.
//
// Returns |base_address| unchanged when |byte_offset| is zero.
iree_status_t loom_amdgpu_system_memory_build_saddr_byte_offset(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_location_id_t location, loom_value_id_t* out_address);

// Appends attrs for a system-scope load cursor read.
iree_status_t loom_amdgpu_system_memory_append_load_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Appends attrs for a load at |scope|.
iree_status_t loom_amdgpu_system_memory_append_load_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count);

// Appends attrs for a system-release store that publishes host-visible data.
iree_status_t loom_amdgpu_system_memory_append_release_store_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Appends attrs for a release store at |scope|.
iree_status_t loom_amdgpu_system_memory_append_release_store_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count);

// Appends attrs for a system-scope no-return atomic update.
iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Appends attrs for a no-return atomic update at |scope|.
iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count);

// Appends attrs for a system-scope returning atomic update.
iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count);

// Appends attrs for a returning atomic update at |scope|.
iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count);

// Emits explicit target-low packets that release prior global writes before a
// following system-scope publication operation.
iree_status_t loom_amdgpu_system_memory_build_release_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location);

// Emits explicit target-low packets that release prior global writes at
// |scope|.
iree_status_t loom_amdgpu_system_memory_build_release_ordering_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_location_id_t location);

// Emits explicit target-low packets that wait for prior vector-memory loads to
// complete without performing an acquire cache operation.
iree_status_t loom_amdgpu_system_memory_build_load_wait(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location);

// Emits explicit target-low packets that make later global reads observe data
// published by a preceding system-scope acquire load or returning atomic.
iree_status_t loom_amdgpu_system_memory_build_acquire_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location);

// Emits explicit target-low packets that make later global reads observe data
// published by a preceding acquire load or returning atomic at |scope|.
iree_status_t loom_amdgpu_system_memory_build_acquire_ordering_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_location_id_t location);

// Emits target-low IR that loads a uniform 32-bit value from host-visible
// system memory.
//
// |base_address| must be an SGPRx2 pointer. The helper emits a GLOBAL_LOAD
// vector-memory packet with the target-specific system-memory cache policy and
// then moves the uniform lane value back to an SGPR with V_READFIRSTLANE.
iree_status_t loom_amdgpu_system_memory_build_uniform_load_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value);

// Emits target-low IR that loads a uniform 64-bit value from host-visible
// system memory.
//
// |base_address| must be an SGPRx2 pointer. The helper emits a GLOBAL_LOAD
// vector-memory packet with the target-specific system-memory cache policy,
// extracts both 32-bit lanes, moves each lane to an SGPR with V_READFIRSTLANE,
// and concatenates the pair into an SGPRx2 result.
iree_status_t loom_amdgpu_system_memory_build_uniform_load_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SYSTEM_MEMORY_H_
