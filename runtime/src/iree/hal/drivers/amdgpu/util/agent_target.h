// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_AGENT_TARGET_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_AGENT_TARGET_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/target/identity.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_agent_target_t
//===----------------------------------------------------------------------===//

// HSA ISA value used to initialize an agent target record.
typedef struct iree_hal_amdgpu_agent_isa_value_t {
  // HSA ISA handle.
  hsa_isa_t isa;
  // HSA ISA name borrowed for the duration of initialization.
  iree_string_view_t name;
} iree_hal_amdgpu_agent_isa_value_t;

// Immutable target identity for one HSA ISA.
//
// The record owns all string storage referenced by |identity| and must remain
// at a stable address after initialization.
typedef struct iree_hal_amdgpu_agent_isa_target_t {
  // HSA ISA handle.
  hsa_isa_t isa;
  // NUL-terminated storage for the HSA ISA name.
  char isa_name_storage[128];
  // HSA ISA name borrowing from |isa_name_storage|.
  iree_string_view_t isa_name;
  // Parsed canonical target identity borrowing from
  // |isa_name_storage|.
  iree_hal_amdgpu_target_identity_t identity;
} iree_hal_amdgpu_agent_isa_target_t;

// Immutable target identities queried for one physical HSA GPU agent.
//
// HSA orders reported ISAs by priority. |primary_isa| is the first reported
// ISA and additional entries preserve their reported order. The record owns
// all target storage and must remain at a stable address after initialization.
typedef struct iree_hal_amdgpu_agent_target_t {
  // HSA GPU agent these identities describe.
  hsa_agent_t agent;
  // Total number of reported ISAs, including |primary_isa|.
  iree_host_size_t isa_count;
  // First and highest-priority HSA ISA reported for |agent|.
  iree_hal_amdgpu_agent_isa_target_t primary_isa;
  // Remaining |isa_count - 1| target identities, or NULL when there are none.
  iree_hal_amdgpu_agent_isa_target_t* additional_isas;
  // Host allocator owning |additional_isas|.
  iree_allocator_t host_allocator;
} iree_hal_amdgpu_agent_target_t;

// Initializes |out_target| from already queried HSA identity values.
//
// This is the pure representation boundary used by tests and the HSA query
// path. ISA names are copied into the record. |asic_revision| resolves
// processors with physical target mappings to their canonical targets.
iree_status_t iree_hal_amdgpu_agent_target_initialize(
    hsa_agent_t agent, iree_host_size_t isa_count,
    const iree_hal_amdgpu_agent_isa_value_t* isa_values, uint32_t asic_revision,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_agent_target_t* out_target);

// Queries and initializes the immutable target identity for |agent|.
//
// HSA ISA and ASIC revision discovery is intentionally centralized here so
// device-library selection, executable loading, and physical-device
// construction consume one consistent target fact.
iree_status_t iree_hal_amdgpu_agent_target_query(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_agent_target_t* out_target);

// Deinitializes |target| and releases its owned storage.
void iree_hal_amdgpu_agent_target_deinitialize(
    iree_hal_amdgpu_agent_target_t* target);

// Returns the highest-priority agent ISA compatible with |required_identity|.
//
// Returns NULL when no reported ISA is compatible.
const iree_hal_amdgpu_agent_isa_target_t*
iree_hal_amdgpu_agent_target_find_compatible_isa(
    const iree_hal_amdgpu_agent_target_t* target,
    const iree_hal_amdgpu_target_identity_t* required_identity);

// Returns the HSA ISA target at |ordinal|, or NULL if out of range.
static inline const iree_hal_amdgpu_agent_isa_target_t*
iree_hal_amdgpu_agent_target_isa_at(
    const iree_hal_amdgpu_agent_target_t* target, iree_host_size_t ordinal) {
  if (ordinal >= target->isa_count) return NULL;
  return ordinal == 0 ? &target->primary_isa
                      : &target->additional_isas[ordinal - 1];
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_AGENT_TARGET_H_
