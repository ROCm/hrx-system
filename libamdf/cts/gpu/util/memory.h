// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_UTIL_MEMORY_H_
#define AMDF_CTS_GPU_UTIL_MEMORY_H_

#include "amdf/amdf.h"

// Case-owned backing with one exact GPU access. HOST_VISIBLE requests also
// obtain the profile's explicit host view; other backing remains unmapped.
// Initialization and release are explicit: failed native cleanup must not
// turn into an implicit destructor retry or premature backing reuse.
struct GpuMemory {
  GpuMemory() = default;
  GpuMemory(const GpuMemory&) = delete;
  GpuMemory& operator=(const GpuMemory&) = delete;
  // Constructs the exact selected single-consumer contract in the given scope.
  void Initialize(const amdf_api_t* api, amdf_memory_scope_t* scope,
                  const amdf_memory_create_info_t& create_info);
  bool Release(const amdf_api_t* api);

  amdf_memory_site_t HostSite() const;
  amdf_memory_site_t DeviceSite(uint32_t queue_family_ordinal) const;

  // Applies the queried CPU side of a whole-backing host/device transition.
  // The witness owns ordering and the corresponding device cache operation.
  void TransferHostCache(const amdf_api_t* api, uint32_t queue_family_ordinal,
                         amdf_host_cache_operation_t operation) const;

  // Allocation retained until every accessing queue has been retired.
  amdf_memory_t* memory = nullptr;
  // Optional host view released before the allocation.
  amdf_host_mapping_t* mapping = nullptr;
  // Stable device address, independent of the CPU virtual address.
  uint64_t device_address = 0;
  // Construction contract retained for prospective visibility queries.
  amdf_memory_create_info_t creation = {};
  // Exact device attachment borrowed by creation.accesses.
  amdf_memory_device_access_t attachment = {};
  // Achieved backing properties.
  amdf_memory_info_t info = {};
  // Achieved device access properties.
  amdf_memory_access_info_t access_info = {};
  // Host pointer, range and cache behavior.
  amdf_host_mapping_info_t host = {};
};

#endif  // AMDF_CTS_GPU_UTIL_MEMORY_H_
