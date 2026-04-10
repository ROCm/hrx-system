// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_RUNTIME_CXX_H_
#define HRX_RUNTIME_CXX_H_

#include "hrx_runtime.h"

#include <string>

namespace hrx::runtime {

inline std::string format_status(hrx_status_t status) {
  if (hrx_status_is_ok(status)) return "OK";

  char* message = nullptr;
  size_t length = 0;
  hrx_status_t format_status =
      hrx_status_to_string(status, &message, &length);
  if (hrx_status_is_ok(format_status) && message) {
    std::string result(message, length);
    hrx_status_free_message(message);
    return result;
  }

  if (message) {
    hrx_status_free_message(message);
  }
  hrx_status_ignore(format_status);
  return "<<could not format hrx_status_t>>";
}

template <typename T, void (*RetainFn)(T), void (*ReleaseFn)(T)>
class hrx_ptr {
 public:
  hrx_ptr() = default;
  explicit hrx_ptr(T owned) : ptr_(owned) {}

  hrx_ptr(const hrx_ptr& other) : ptr_(other.ptr_) {
    if (ptr_) {
      RetainFn(ptr_);
    }
  }

  hrx_ptr(hrx_ptr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  hrx_ptr& operator=(const hrx_ptr& other) {
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

  hrx_ptr& operator=(hrx_ptr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  ~hrx_ptr() { reset(); }

  static hrx_ptr steal(T ptr) { return hrx_ptr(ptr); }

  static hrx_ptr borrow(T ptr) {
    if (ptr) {
      RetainFn(ptr);
    }
    return hrx_ptr(ptr);
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
    hrx_ptr<hrx_device_t, hrx_device_retain, hrx_device_release>;
using allocator_ptr =
    hrx_ptr<hrx_allocator_t, hrx_allocator_retain, hrx_allocator_release>;
using semaphore_ptr =
    hrx_ptr<hrx_semaphore_t, hrx_semaphore_retain, hrx_semaphore_release>;
using stream_ptr =
    hrx_ptr<hrx_stream_t, hrx_stream_retain, hrx_stream_release>;
using buffer_ptr =
    hrx_ptr<hrx_buffer_t, hrx_buffer_retain, hrx_buffer_release>;
using module_ptr =
    hrx_ptr<hrx_module_t, hrx_module_retain, hrx_module_release>;
using function_ptr =
    hrx_ptr<hrx_function_t, hrx_function_retain, hrx_function_release>;
using value_list_ptr =
    hrx_ptr<hrx_value_list_t, hrx_value_list_retain,
             hrx_value_list_release>;
using fence_ptr =
    hrx_ptr<hrx_fence_t, hrx_fence_retain, hrx_fence_release>;
using buffer_view_ptr =
    hrx_ptr<hrx_buffer_view_t, hrx_buffer_view_retain,
             hrx_buffer_view_release>;

}  // namespace hrx::runtime

#endif  // HRX_RUNTIME_CXX_H_
