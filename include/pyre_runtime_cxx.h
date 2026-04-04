// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_RUNTIME_CXX_H_
#define PYRE_RUNTIME_CXX_H_

#include "pyre_runtime.h"

#include <string>

namespace pyre::runtime {

inline std::string format_status(pyre_status_t status) {
  if (pyre_status_is_ok(status)) return "OK";

  char* message = nullptr;
  size_t length = 0;
  pyre_status_t format_status =
      pyre_status_to_string(status, &message, &length);
  if (pyre_status_is_ok(format_status) && message) {
    std::string result(message, length);
    pyre_status_free_message(message);
    return result;
  }

  if (message) {
    pyre_status_free_message(message);
  }
  pyre_status_ignore(format_status);
  return "<<could not format pyre_status_t>>";
}

template <typename T, void (*RetainFn)(T), void (*ReleaseFn)(T)>
class pyre_ptr {
 public:
  pyre_ptr() = default;
  explicit pyre_ptr(T owned) : ptr_(owned) {}

  pyre_ptr(const pyre_ptr& other) : ptr_(other.ptr_) {
    if (ptr_) {
      RetainFn(ptr_);
    }
  }

  pyre_ptr(pyre_ptr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  pyre_ptr& operator=(const pyre_ptr& other) {
    if (this == &other) {
      return *this;
    }
    if (other.ptr_) {
      RetainFn(other.ptr_);
    }
    reset();
    ptr_ = other.ptr_;
    return *this;
  }

  pyre_ptr& operator=(pyre_ptr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  ~pyre_ptr() { reset(); }

  static pyre_ptr steal(T ptr) { return pyre_ptr(ptr); }

  static pyre_ptr borrow(T ptr) {
    if (ptr) {
      RetainFn(ptr);
    }
    return pyre_ptr(ptr);
  }

  T* for_output() {
    reset();
    return &ptr_;
  }

  void reset(T ptr = nullptr) {
    if (ptr == ptr_) {
      return;
    }
    if (ptr_) {
      ReleaseFn(ptr_);
    }
    ptr_ = ptr;
  }

  T get() const { return ptr_; }

  T release() {
    T ptr = ptr_;
    ptr_ = nullptr;
    return ptr;
  }

  operator T() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T ptr_ = nullptr;
};

using device_ptr =
    pyre_ptr<pyre_device_t, pyre_device_retain, pyre_device_release>;
using allocator_ptr =
    pyre_ptr<pyre_allocator_t, pyre_allocator_retain, pyre_allocator_release>;
using semaphore_ptr =
    pyre_ptr<pyre_semaphore_t, pyre_semaphore_retain, pyre_semaphore_release>;
using stream_ptr =
    pyre_ptr<pyre_stream_t, pyre_stream_retain, pyre_stream_release>;
using buffer_ptr =
    pyre_ptr<pyre_buffer_t, pyre_buffer_retain, pyre_buffer_release>;
using module_ptr =
    pyre_ptr<pyre_module_t, pyre_module_retain, pyre_module_release>;
using function_ptr =
    pyre_ptr<pyre_function_t, pyre_function_retain, pyre_function_release>;
using value_list_ptr =
    pyre_ptr<pyre_value_list_t, pyre_value_list_retain,
             pyre_value_list_release>;
using fence_ptr =
    pyre_ptr<pyre_fence_t, pyre_fence_retain, pyre_fence_release>;
using buffer_view_ptr =
    pyre_ptr<pyre_buffer_view_t, pyre_buffer_view_retain,
             pyre_buffer_view_release>;

}  // namespace pyre::runtime

#endif  // PYRE_RUNTIME_CXX_H_
