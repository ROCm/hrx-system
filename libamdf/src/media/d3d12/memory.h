// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_MEMORY_H_
#define AMDF_SRC_MEDIA_D3D12_MEMORY_H_

#include "libamdf/src/media/d3d12/device.h"
#include "libamdf/src/memory_resource.h"

namespace amdf::media::d3d12 {

enum MemoryProfile : uint32_t {
  kSharedHeap = 0,
  kFeedbackBuffer = 1,
};

struct Memory {
  // Media device borrowed through the core memory's lifetime.
  Device* device;
  // Exact native construction contract used for this backing.
  MemoryProfile profile;
  // Explicit shared image heap or host-page heap attachment.
  ComPtr<ID3D12Heap> heap;
  // Cold linear interpretation of the heap, usable for compressed bytes or
  // feedback. It allocates no payload; image identities alias explicit ranges.
  ComPtr<ID3D12Resource> resource;
  // Independent export identity for shared default-memory backing.
  HANDLE shared_handle;
  // Cold KMT import owning the allocation metadata reference until release.
  D3DKMT_HANDLE kmt_resource;
  // Allocation reference returned with the cold KMT resource import.
  D3DKMT_HANDLE kmt_allocation;
  // Explicit feedback payload; native heap borrows it until detach.
  void* host_pointer;
  // Complete owned payload extent, distinct from the logical range.
  uint64_t native_byte_length;
};

// Resolves an already-validated exact media consumer without a native query.
inline Memory* NativeMemory(amdf_memory_t* memory, uint32_t access_ordinal) {
  return static_cast<Memory*>(
      memory->accesses[memory->accesses[access_ordinal].native_owner_ordinal]
          .native);
}

amdf_status_t QueryMemoryProfile(amdf_device_t* device, uint32_t ordinal,
                                 amdf_memory_native_profile_t* out_profile);

amdf_status_t PrepareMemory(amdf_memory_t* memory,
                            const amdf_memory_native_group_t* group,
                            const amdf_memory_native_create_info_t* create_info,
                            amdf_memory_info_t* out_info);

amdf_status_t PrepareMemoryImport(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_MEMORY_H_
