// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_CTS_TEST_DEVICE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_CTS_TEST_DEVICE_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <semaphore>
#include <thread>
#include <unordered_map>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/gpu/code.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {

// Matches the runner's portable identity using passive discovery. NULL admits
// every endpoint; opening the query handle activates no device.
iree_status_t MatchesNativeIdentity(iree_hal_amd_fabric_t* fabric,
                                    const amdf_endpoint_id_t* id,
                                    const char* selected, bool* out_matches);

// Observe complete resource reclamation at synchronous final device release.
struct DeviceAllocator {
  // Live allocations made by the concrete device and its children.
  std::atomic<size_t> live_count = 0;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self_pointer,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer);
};

// Holds the real allocation dependency while forwarding all native calls.
// Fixtures run serially and join fabric/device borrowers before destroying
// this table. The gate qualifies HAL ownership, not kernel lock isolation.
class NativeMemoryGate {
 public:
  ~NativeMemoryGate();

  // Optional allocation limit is both advertised and enforced by the native
  // dependency; accepted backing still comes from the real driver.
  iree_status_t CreateFabric(
      iree_hal_amd_fabric_t** out_fabric,
      uint64_t maximum_byte_length = UINT64_MAX,
      amdf_native_lifetime_t lifetime = AMDF_NATIVE_LIFETIME_INSTANCE);

  // PROCESS native bindings share one fabric across fixtures, as the registered
  // driver does. Reopening a different render file cannot replace that VM.
  static iree_status_t AcquireProcessFabric(iree_hal_amd_fabric_t** out_fabric);

  struct MemoryUsage {
    // Native payload bytes whose real destroy call has not yet returned.
    uint64_t live_bytes = 0;
    // Maximum simultaneously live payload since ResetPeak.
    uint64_t peak_bytes = 0;
    // Successful calls to the real native allocation dependency.
    uint64_t allocation_count = 0;
    // Native destroy failures, which cannot prove payload reclamation.
    uint64_t destroy_failures = 0;
    // Native extent geometry of the most recently created real allocation.
    amdf_memory_info_t last_allocation = {};
  };

  // Removes host publication for local code without changing native backing.
  void RequireCodeUpload() { require_code_upload_ = true; }

  MemoryUsage ResetPeak();
  MemoryUsage QueryUsage();
  MemoryUsage WaitForBytes(uint64_t byte_count);

  // Read only after Hold::WaitEntered has published the native caller.
  std::thread::id native_thread() const { return native_thread_; }

  class Hold {
   public:
    explicit Hold(NativeMemoryGate* owner);
    ~Hold();

    void WaitEntered();
    void ReleaseAndHoldNext();

   private:
    // Borrowed gate outliving the native call and its explicit hold.
    NativeMemoryGate* owner_;
  };

 private:
  static amdf_status_t AMDF_CALL QueryDeviceProfile(
      amdf_memory_scope_t* scope, uint32_t profile_ordinal,
      uint32_t access_count, const amdf_memory_device_access_t* accesses,
      amdf_memory_profile_t* out_profile,
      amdf_memory_access_capabilities_t* out_access_capabilities);

  static amdf_status_t AMDF_CALL QueryProfilePair(
      amdf_memory_scope_t* scope, const amdf_memory_profile_pair_query_t* query,
      amdf_memory_pair_info_t* out_info);

  static amdf_status_t AMDF_CALL CreateMemory(
      amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
      amdf_memory_t** out_memory, amdf_memory_release_info_t* out_release_info);

  static amdf_status_t AMDF_CALL DestroyMemory(
      amdf_memory_t* memory, amdf_memory_release_info_t* out_release_info);

  // Exact native API table, borrowed for all forwarded calls.
  inline static const amdf_api_t* native_api_ = nullptr;
  // Same native API with an allocation-entry gate, never a substitute
  // allocator.
  amdf_api_t forwarding_api_ = {};
  // Maximum backing size advertised and accepted by this native dependency.
  uint64_t maximum_byte_length_ = UINT64_MAX;
  // Exercise staged loading even on a device with a usable CPU aperture.
  bool require_code_upload_ = false;
  // Serial fixture identity required by the native ABI's context-free table.
  inline static NativeMemoryGate* active_ = nullptr;
  // One-shot hold consumed by the next native backing acquisition.
  std::atomic<bool> hold_{false};
  // Publishes entry and thread identity of the held native call.
  std::binary_semaphore entered_{0};
  // Permission to continue the native call after independent work completes.
  std::binary_semaphore resume_{0};
  // Joins the forwarding function's final fixture access.
  std::binary_semaphore returned_{0};
  // Observed allocation owner, read only after entered_.
  std::thread::id native_thread_;
  // Serializes observation only, never a real native allocation/free call.
  std::mutex usage_mutex_;
  // Publishes actual native destruction to tests awaiting quiescent trim.
  std::condition_variable usage_changed_;
  // Exact native extents of handles created through this forwarding table.
  std::unordered_map<amdf_memory_t*, uint64_t> allocations_;
  // Observed live/peak payload and native operation counts.
  MemoryUsage usage_;
};

// Holds the real completion callback boundary while native work progresses.
struct CompletionPause {
  // Borrowed timeline owning timepoint enrollment.
  iree_async_semaphore_t* semaphore;
  // Stack callback storage, joined before leaving this scope.
  iree_async_semaphore_timepoint_t timepoint = {};
  // Callback entry, observed before submitting the native dependency chain.
  std::binary_semaphore entered{0};
  // Releases the callback after the test establishes its pending-work state.
  std::binary_semaphore proceed{0};
  // Joins the callback's final stack access.
  std::binary_semaphore finished{0};
  // Successful enrollment still requiring cancellation or callback joining.
  bool enrolled = false;
  iree_status_t Arm(uint64_t value) {
    timepoint = {
        .callback =
            [](void* user_data, iree_async_semaphore_timepoint_t*,
               iree_status_t status) {
              auto* pause = static_cast<CompletionPause*>(user_data);
              IREE_EXPECT_OK(status);
              pause->entered.release();
              pause->proceed.acquire();
              pause->finished.release();
            },
        .user_data = this,
    };
    IREE_RETURN_IF_ERROR(
        iree_async_semaphore_acquire_timepoint(semaphore, value, &timepoint));
    enrolled = true;
    return iree_ok_status();
  }
  void Join() {
    if (enrolled &&
        !iree_async_semaphore_cancel_timepoint(semaphore, &timepoint)) {
      proceed.release();
      finished.acquire();
    }
    enrolled = false;
  }
  ~CompletionPause() { Join(); }
};

class GpuNativeTest : public ::testing::Test {
 protected:
  // Native packet protocol exercised by this fixture, independent of ISA.
  virtual amdf_queue_command_type_t command_type() const {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  }

  void SetUp() override;

  void TearDown() override;

  void Create(iree_hal_device_runtime_feature_flags_t runtime_features = 0);

  void CreateSemaphore(iree_hal_semaphore_t** out_semaphore);

  // Native ring/control backing is separate from device-local program storage.
  void SelectControlMemory(amdf_memory_device_access_t* access,
                           iree_hal_amd_fabric_memory_options_t* policy);

  // Raw packet fixtures publish through the production transfer owner before
  // entering programs; HAL device tests use their device-owned transfer queue.
  void PublishCode(uint32_t compute_family_ordinal,
                   iree_hal_amd_fabric_memory_t* memory,
                   iree_hal_amd_gpu_code_upload_t** upload);

  // Independent native discovery ownership, released after device cleanup.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Constructing native owner, transferred to device on success.
  iree_hal_amd_gpu_context_t* context_ = nullptr;
  // Explicit provisioning used by each test.
  iree_hal_amd_gpu_device_options_t options_ = {};
  // Process-shared progress owners supplied through the public creation API.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Public I/O provisioning applied before any device acquires a poll owner.
  iree_async_proactor_pool_options_t proactor_options_ =
      iree_async_proactor_pool_options_default();
  // External tracker reference used to observe actual queue progress.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Allocator lifetime extends through the final worker cleanup.
  DeviceAllocator allocator_;
  // Owning public device reference, independently of its group.
  iree_hal_device_t* device_ = nullptr;
  // Immutable group assigning real memory scopes and queue axes.
  iree_hal_device_group_t* group_ = nullptr;
  // Borrowed provisioned queues, never separately retained by this fixture.
  std::array<iree_hal_queue_t*, 2> queues_ = {};
};

}  // namespace iree::hal::amd::gpu::cts

#endif  // IREE_HAL_DRIVERS_AMD_GPU_CTS_TEST_DEVICE_H_
