// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_OBJECT_H_
#define AMDF_SRC_MEDIA_D3D12_OBJECT_H_

#include <new>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/windows/kmt_api.h"

namespace amdf::media::d3d12 {

inline amdf_status_t Hresult(HRESULT result) {
  return SUCCEEDED(result) ? AMDF_STATUS_OK
                           : amdf_make_status(AMDF_STATUS_DOMAIN_HRESULT,
                                              static_cast<uint32_t>(result));
}

// Cold native owners use the caller's allocator, including C++ object storage.
// Publication transfers the complete object; native methods own partial state
// during construction so its destructor releases every acquired COM reference.
template <typename T>
amdf_status_t Allocate(amdf_allocator_t allocator, T** out_object) {
  void* storage = nullptr;
  const amdf_status_t status =
      amdf_malloc(allocator, sizeof(T), alignof(T), &storage);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *out_object = new (storage) T{};
  return AMDF_STATUS_OK;
}

template <typename T>
void Delete(amdf_allocator_t allocator, T* object) {
  if (object == nullptr) {
    return;
  }
  object->~T();
  amdf_free(allocator, object);
}

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_OBJECT_H_
