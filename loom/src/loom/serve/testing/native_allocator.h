// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_NATIVE_ALLOCATOR_H_
#define LOOM_SERVE_TESTING_NATIVE_ALLOCATOR_H_

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "loom/serve/testing/device.h"

namespace loom_serve::testing {

// Holds an allocator dependency on the native allocation worker. A cold source
// acquisition identifies that worker before the test is exposed to consumers.
// No HAL queue, pool, mapping, or completion callback is replaced.
class NativeAllocator {
 public:
  explicit NativeAllocator(iree_allocator_t allocator = iree_allocator_system())
      : allocator_(allocator) {}
  ~NativeAllocator() { AwaitReleased(); }
  iree_allocator_t allocator() { return {this, Control}; }
  // Public owners must be released before this join. XDNA can retire its
  // observer and final device storage asynchronously after the last release.
  void AwaitReleased() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(
        lock, [&] { return live_allocations_ == 0 && active_calls_ == 0; });
  }
  iree_status_t Observe(loom_serve_device_t* device) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      observing_ = true;
    }
    iree_hal_buffer_t* probe = nullptr;
    iree_status_t status = iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(device),
        iree_hal_make_pool_reservation_request(4096, 0),
        iree_infinite_timeout(), &probe);
    iree_hal_buffer_release(probe);
    if (iree_status_is_ok(status)) {
      status = loom_serve_test_device_trim_and_drain(device, 0);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (iree_status_is_ok(status) && observing_) {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "test requires a native allocation worker");
      }
      observing_ = false;
    }
    return status;
  }
  static NativeAllocator& ForDevice(loom_serve_device_t* device) {
    auto allocator =
        iree_hal_device_host_allocator(loom_serve_device_handle(device));
    // The fixture installs this allocator before creating the device.
    IREE_ASSERT(allocator.ctl == Control);
    return *static_cast<NativeAllocator*>(allocator.self);
  }
  void Arm(iree_allocator_command_t command) {
    std::lock_guard<std::mutex> lock(mutex_);
    command_ = command;
    armed_ = true;
    entered_ = released_ = retired_ = false;
  }
  void WaitEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  void FailNextAllocation(iree_allocator_command_t command,
                          bool native_only = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_.armed = true;
    failure_.command = command;
    failure_.native_only = native_only;
  }
  void Release() {
    std::unique_lock<std::mutex> lock(mutex_);
    armed_ = false;
    released_ = true;
    condition_.notify_all();
    condition_.wait(lock, [&] { return !entered_ || retired_; });
  }

 private:
  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* self = static_cast<NativeAllocator*>(user_data);
    const bool had_pointer = (command == IREE_ALLOCATOR_COMMAND_REALLOC ||
                              command == IREE_ALLOCATOR_COMMAND_FREE) &&
                             *inout_pointer != nullptr;
    bool fail_allocation = false;
    {
      std::unique_lock<std::mutex> lock(self->mutex_);
      ++self->active_calls_;
      const auto thread = std::this_thread::get_id();
      if (self->observing_ && thread != self->caller_ &&
          command == IREE_ALLOCATOR_COMMAND_CALLOC) {
        if (std::find(self->native_.begin(), self->native_.end(), thread) ==
            self->native_.end()) {
          self->native_.push_back(thread);
        }
        self->observing_ = false;
      }
      const bool native = std::find(self->native_.begin(), self->native_.end(),
                                    thread) != self->native_.end();
      if (self->failure_.armed && command == self->failure_.command &&
          (!self->failure_.native_only || native)) {
        self->failure_.armed = false;
        fail_allocation = true;
      }
      if (self->armed_ && command == self->command_ && native) {
        self->armed_ = false;
        self->entered_ = true;
        self->condition_.notify_all();
        self->condition_.wait(lock, [&] { return self->released_; });
        self->retired_ = true;
        self->condition_.notify_all();
      }
    }
    iree_status_t status =
        fail_allocation ? iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED)
                        : self->allocator_.ctl(self->allocator_.self, command,
                                               params, inout_pointer);
    {
      std::lock_guard<std::mutex> lock(self->mutex_);
      if (iree_status_is_ok(status)) {
        if (command == IREE_ALLOCATOR_COMMAND_FREE) {
          self->live_allocations_ -= had_pointer ? 1 : 0;
        } else if (command != IREE_ALLOCATOR_COMMAND_REALLOC || !had_pointer) {
          self->live_allocations_ += *inout_pointer ? 1 : 0;
        }
      }
      --self->active_calls_;
      self->condition_.notify_all();
    }
    return status;
  }
  // Borrowed underlying allocator outliving this fixture and its device.
  iree_allocator_t allocator_;
  // Serializes the selected allocator dependency and its lifetime.
  std::mutex mutex_;
  // Joins entry and release without timing assumptions.
  std::condition_variable condition_;
  // Successful allocations whose final FREE has not returned.
  size_t live_allocations_ = 0;
  // Allocator calls still using this fixture, including the final FREE.
  size_t active_calls_ = 0;
  // Caller capture allocations remain available while native work is held.
  std::thread::id caller_ = std::this_thread::get_id();
  // Worker observed through a real cold source allocation after device
  // creation.
  std::vector<std::thread::id> native_;
  // Command selected for the next native allocator call.
  iree_allocator_command_t command_ = IREE_ALLOCATOR_COMMAND_CALLOC;
  // The initial source allocation has not yet identified its native worker.
  bool observing_ = false;
  // A matching native dependency is still being selected.
  bool armed_ = false;
  // One selected allocator dependency reports allocation failure.
  struct {
    // A matching allocator call has not consumed the failure yet.
    bool armed = false;
    // Allocation kind to fail without invoking the underlying allocator.
    iree_allocator_command_t command = IREE_ALLOCATOR_COMMAND_CALLOC;
    // Restricts matching to previously observed native allocation workers.
    bool native_only = false;
  } failure_;
  // The selected dependency is currently held.
  bool entered_ = false;
  // The outer test owner permits the selected dependency to continue.
  bool released_ = false;
  // The held dependency observed release before another hold can be armed.
  bool retired_ = false;
};

class ScopedNativeHold {
 public:
  ScopedNativeHold(NativeAllocator& allocator, iree_allocator_command_t command)
      : allocator_(allocator) {
    allocator_.Arm(command);
  }
  ScopedNativeHold(
      loom_serve_device_t* device,
      iree_allocator_command_t command = IREE_ALLOCATOR_COMMAND_CALLOC)
      : ScopedNativeHold(NativeAllocator::ForDevice(device), command) {}
  ~ScopedNativeHold() { allocator_.Release(); }
  void Release() { allocator_.Release(); }
  void WaitEntered() { allocator_.WaitEntered(); }

 private:
  // Borrowed allocator outliving both the hold and the device.
  NativeAllocator& allocator_;
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_NATIVE_ALLOCATOR_H_
