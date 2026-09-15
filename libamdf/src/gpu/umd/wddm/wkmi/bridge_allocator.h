// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_BRIDGE_ALLOCATOR_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_BRIDGE_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

namespace amdf::wkmi_bridge {

// Allocates and zeroes bridge-controlled storage through the parent instance.
inline void* AllocateHostBytes(amdf_allocator_t allocator, size_t byte_length,
                               size_t minimum_alignment) noexcept {
  const size_t alignment = minimum_alignment < amdf_max_align_t
                               ? amdf_max_align_t
                               : minimum_alignment;
  void* pointer =
      allocator.allocate(allocator.user_data, byte_length, alignment);
  if (pointer == nullptr) return nullptr;
  if ((reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) != 0) {
    allocator.free(allocator.user_data, pointer);
    return nullptr;
  }
  std::memset(pointer, 0, byte_length);
  return pointer;
}

// Frees bridge-controlled storage through the allocator that created it.
inline void FreeHostBytes(amdf_allocator_t allocator, void* pointer) noexcept {
  if (pointer != nullptr) allocator.free(allocator.user_data, pointer);
}

// Owns one zeroed bridge scratch buffer without using the sidecar CRT heap.
class HostBuffer {
 public:
  HostBuffer() = default;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  ~HostBuffer() { Reset(); }

  bool Allocate(amdf_allocator_t allocator, size_t byte_length,
                size_t minimum_alignment = amdf_max_align_t) noexcept {
    if (pointer_ != nullptr || byte_length == 0 || minimum_alignment == 0 ||
        (minimum_alignment & (minimum_alignment - 1)) != 0) {
      return false;
    }
    void* pointer =
        AllocateHostBytes(allocator, byte_length, minimum_alignment);
    if (pointer == nullptr) return false;
    allocator_ = allocator;
    pointer_ = pointer;
    byte_length_ = byte_length;
    return true;
  }

  void Reset() noexcept {
    FreeHostBytes(allocator_, pointer_);
    pointer_ = nullptr;
    byte_length_ = 0;
  }

  void* data() const noexcept { return pointer_; }
  size_t byte_length() const noexcept { return byte_length_; }

 private:
  // Allocator copied when storage is acquired.
  amdf_allocator_t allocator_ = {};
  // Owned storage, or `nullptr` while empty.
  void* pointer_ = nullptr;
  // Owned storage length in bytes.
  size_t byte_length_ = 0;
};

// Owns one placement-constructed bridge object through a supplied allocator.
template <typename T>
class HostObject {
 public:
  explicit HostObject(amdf_allocator_t allocator) : allocator_(allocator) {}
  HostObject(const HostObject&) = delete;
  HostObject& operator=(const HostObject&) = delete;

  ~HostObject() { Reset(); }

  bool Allocate() {
    void* storage = AllocateHostBytes(allocator_, sizeof(T), amdf_alignof(T));
    if (storage == nullptr) return false;
    try {
      object_ = ::new (storage) T();
    } catch (...) {
      FreeHostBytes(allocator_, storage);
      throw;
    }
    return true;
  }

  void Reset() noexcept {
    if (object_ == nullptr) return;
    object_->~T();
    FreeHostBytes(allocator_, object_);
    object_ = nullptr;
  }

  T* get() const noexcept { return object_; }
  T* operator->() const noexcept { return object_; }

  T* release() noexcept {
    T* object = object_;
    object_ = nullptr;
    return object;
  }

 private:
  // Allocator copied before object construction.
  amdf_allocator_t allocator_;
  // Owned object, or `nullptr` while empty.
  T* object_ = nullptr;
};

// Destroys one placement-constructed object through its creating allocator.
template <typename T>
void DestroyHostObject(amdf_allocator_t allocator, T* object) noexcept {
  if (object == nullptr) return;
  object->~T();
  FreeHostBytes(allocator, object);
}

}  // namespace amdf::wkmi_bridge

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_BRIDGE_ALLOCATOR_H_
