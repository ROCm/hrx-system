// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/program.h"

#include <atomic>
#include <cstring>
#include <future>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum class LocalHostView {
  kUnavailable,
  kPublishable,
  kUnpublishable,
};

// The native memory dependency supplies mapped system storage and distinct
// local mapping/publication capabilities. Program placement, publication,
// leases, trim, and cleanup all use the production implementation.
class ProgramLifetimeTest : public ::testing::TestWithParam<LocalHostView> {
 protected:
  bool PublishesLocalWrites() const {
    return GetParam() == LocalHostView::kPublishable;
  }

  struct Scope {
    // Fixture owning this native scope.
    ProgramLifetimeTest* owner;
    // Whether this is physical-local storage instead of the upload scope.
    bool local;
  };
  struct Memory {
    // Native scope under which this backing was allocated.
    Scope* scope;
    // Bytes used for the persistent host view when one was requested.
    std::vector<uint8_t> bytes;
    // Native allocation flags after admission.
    amdf_memory_flags_t flags;
  };

  void SetUp() override {
    api_.memory_scope_query_info = [](amdf_memory_scope_t* handle,
                                      amdf_memory_scope_info_t* info) {
      auto* scope = reinterpret_cast<Scope*>(handle);
      info->kind = scope->local ? AMDF_MEMORY_SCOPE_KIND_LOCAL
                                : AMDF_MEMORY_SCOPE_KIND_SYSTEM;
      info->memory_profile_count = 1;
      return AMDF_STATUS_OK;
    };
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t* handle, uint32_t, uint32_t,
           const amdf_memory_device_access_t*, amdf_memory_profile_t* profile,
           amdf_memory_access_capabilities_t*) {
          auto* scope = reinterpret_cast<Scope*>(handle);
          const bool mapped = !scope->local || scope->owner->GetParam() !=
                                                   LocalHostView::kUnavailable;
          profile->roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                           (mapped ? AMDF_MEMORY_PROFILE_ROLE_HOST_MAP : 0);
          profile->supported_flags =
              (scope->local || scope->owner->unified_
                   ? AMDF_MEMORY_FLAG_DEVICE_LOCAL
                   : 0) |
              (mapped ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0);
          profile->host_mapping.supported_access =
              mapped ? AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE
                     : 0;
          profile->allocation.minimum_alignment = 256;
          profile->allocation.byte_length_granularity = 256;
          profile->allocation.maximum_byte_length = UINT32_MAX;
          return AMDF_STATUS_OK;
        };
    api_.memory_scope_query_pair_info =
        [](amdf_memory_scope_t* handle,
           const amdf_memory_profile_pair_query_t* query,
           amdf_memory_pair_info_t* pair) {
          auto* scope = reinterpret_cast<Scope*>(handle);
          EXPECT_EQ(query->producer.kind, AMDF_MEMORY_SITE_KIND_HOST);
          EXPECT_EQ(query->producer.value.host_access,
                    AMDF_MEMORY_MAP_FLAG_WRITE);
          EXPECT_EQ(query->consumer.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
          EXPECT_EQ(query->consumer.value.device.queue_family_ordinal, 0u);
          EXPECT_NE(query->required_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
          if (scope->local && !scope->owner->PublishesLocalWrites()) {
            return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
          }
          pair->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
          pair->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
          return AMDF_STATUS_OK;
        };
    api_.memory_create = [](amdf_memory_scope_t* handle,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out_memory,
                            amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
      }
      auto* scope = reinterpret_cast<Scope*>(handle);
      auto* self = scope->owner;
      if (self->allocate_entered_) {
        self->allocate_entered_->release();
        self->allocate_resume_->acquire();
        self->allocate_entered_ = nullptr;
      }
      if (++self->allocate_count_ == self->fail_allocate_) {
        return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      }
      auto* memory = new Memory{scope, std::vector<uint8_t>(info->byte_length),
                                info->required_flags};
      ++self->live_count_;
      self->allocation_thread_ = std::this_thread::get_id();
      *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
      return AMDF_STATUS_OK;
    };
    api_.memory_query_info = [](amdf_memory_t* handle,
                                amdf_memory_info_t* info) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      info->byte_length = memory->bytes.size();
      info->alignment = 256;
      info->native_allocation_byte_length = memory->bytes.size();
      info->native_allocation_granularity = 256;
      info->flags = memory->flags;
      return AMDF_STATUS_OK;
    };
    api_.memory_query_address = [](amdf_memory_t* handle, uint32_t,
                                   amdf_memory_address_kind_t,
                                   uint64_t* out_address) {
      *out_address = reinterpret_cast<uintptr_t>(handle) & ~UINT64_C(255);
      return AMDF_STATUS_OK;
    };
    api_.memory_map = [](amdf_memory_t* handle, const amdf_memory_map_info_t*,
                         amdf_host_mapping_t** out_mapping) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      EXPECT_TRUE(memory->flags & AMDF_MEMORY_FLAG_HOST_VISIBLE);
      *out_mapping = reinterpret_cast<amdf_host_mapping_t*>(memory);
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_query_info = [](amdf_host_mapping_t* handle,
                                      amdf_host_mapping_info_t* info) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      info->pointer = memory->bytes.data();
      info->byte_length = memory->bytes.size();
      info->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_cache_control = [](amdf_host_mapping_t* handle,
                                         amdf_host_cache_operation_t operation,
                                         uint64_t offset, uint64_t length) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      EXPECT_LE(offset + length, memory->bytes.size());
      EXPECT_EQ(operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
      EXPECT_TRUE(!memory->scope->local ||
                  memory->scope->owner->PublishesLocalWrites());
      memory->scope->owner->publication_thread_ = std::this_thread::get_id();
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_destroy = [](amdf_host_mapping_t*) {
      return AMDF_STATUS_OK;
    };
    api_.memory_destroy = [](amdf_memory_t* handle,
                             amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
      }
      auto* memory = reinterpret_cast<Memory*>(handle);
      auto* self = memory->scope->owner;
      if (self->free_entered_) {
        self->free_entered_->release();
        self->free_resume_->acquire();
        self->free_entered_ = nullptr;
      }
      --self->live_count_;
      self->release_thread_ = std::this_thread::get_id();
      delete memory;
      return AMDF_STATUS_OK;
    };
    api_.instance_destroy = [](amdf_instance_t*) { return AMDF_STATUS_OK; };
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = iree_allocator_system();
    fabric_->api = &api_;
    fabric_->system_scope = reinterpret_cast<amdf_memory_scope_t*>(&system_);
    local_handle_ = reinterpret_cast<amdf_memory_scope_t*>(&local_);
    context_.fabric = fabric_;
    context_.api = &api_;
    context_.memory_scope_count = 1;
    context_.memory_scopes = &local_handle_;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    IREE_ASSERT_OK(iree_hal_amd_gpu_program_pool_initialize(
        &context_, 0, 3, iree_allocator_system(), &pool_));
  }

  void TearDown() override {
    iree_hal_amd_gpu_program_pool_deinitialize(&pool_);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    EXPECT_EQ(live_count_, 0u);
    iree_hal_amd_fabric_release(fabric_);
  }

  void Record(iree_hal_amd_gpu_program_t* program) {
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_program_initialize(&pool_, 256, 4096, 16, program));
    memset(program->contents.data, 0x5a, program->contents.data_length);
    IREE_ASSERT_OK(iree_hal_amd_gpu_program_seal(program));
  }

  static iree_hal_amd_fabric_memory_callback_t Completion(
      std::promise<iree::Status>* result) {
    return {
        .fn =
            [](void* user_data, iree_status_t status) {
              static_cast<std::promise<iree::Status>*>(user_data)->set_value(
                  iree::Status(std::move(status)));
            },
        .user_data = result,
    };
  }

  void Grow(iree_hal_amd_gpu_program_t* program, uint32_t queue_ordinal) {
    iree_hal_amd_gpu_program_growth_t growth = {};
    std::promise<iree::Status> completed;
    IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_begin(
        program, queue_ordinal, Completion(&completed), &growth));
    auto status = completed.get_future().get();
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_program_grow_end(&growth, status.release()));
    iree_hal_amd_gpu_context_join_memory(&context_);
  }

  void ReuseWhileGrowing(bool cancel_borrow, bool before_enrollment) {
    iree_hal_amd_gpu_program_t program = {};
    ASSERT_NO_FATAL_FAILURE(Record(&program));
    uint64_t frontier[3] = {};
    auto* initial = iree_hal_amd_gpu_program_try_acquire(&program, 0, frontier);
    ASSERT_NE(initial, nullptr);
    iree_hal_amd_gpu_program_publish(&program, initial, 0, 1);
    if (cancel_borrow) {
      // An unpublished Q2 borrow blocks Q1's otherwise valid causal reuse.
      frontier[0] = 1;
      ASSERT_EQ(iree_hal_amd_gpu_program_try_acquire(&program, 2, frontier),
                initial);
    }
    ASSERT_EQ(iree_hal_amd_gpu_program_try_acquire(&program, 1, frontier),
              nullptr);
    std::binary_semaphore entered{0}, resume{0};
    allocate_entered_ = &entered;
    allocate_resume_ = &resume;
    iree_hal_amd_gpu_program_growth_t growth = {};
    std::promise<iree::Status> completed;
    IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_begin(
        &program, 1, Completion(&completed), &growth));
    entered.acquire();
    auto return_instance = [&] {
      if (cancel_borrow) {
        iree_hal_amd_gpu_program_cancel(&program, initial);
      } else {
        iree_hal_amd_gpu_program_release(&program, initial);
      }
    };
    uint32_t notifications = 0;
    if (before_enrollment) {
      return_instance();
    }
    iree_hal_amd_gpu_program_grow_await_reuse(
        &growth, [](void* data) { ++*static_cast<uint32_t*>(data); },
        &notifications);
    if (!before_enrollment) {
      EXPECT_EQ(notifications, 0u);
      return_instance();
    }
    EXPECT_EQ(notifications, 1u);
    EXPECT_EQ(program.waiters, nullptr);
    auto* reused = iree_hal_amd_gpu_program_try_acquire(&program, 1, frontier);
    EXPECT_EQ(reused, initial);
    if (reused) {
      iree_hal_amd_gpu_program_publish(&program, reused, 1, 1);
      iree_hal_amd_gpu_program_release(&program, reused);
    }
    EXPECT_EQ(notifications, 1u);
    EXPECT_EQ(program.queues[1].growth, &growth);
    // The existing instance was reused before native acquisition could return.
    resume.release();
    auto status = completed.get_future().get();
    IREE_EXPECT_OK(
        iree_hal_amd_gpu_program_grow_end(&growth, status.release()));
    iree_hal_amd_gpu_context_join_memory(&context_);
    if (cancel_borrow) {
      iree_hal_amd_gpu_program_release(&program, initial);
    }
    EXPECT_EQ(notifications, 1u);
    iree_hal_amd_gpu_program_deinitialize(&program);
  }

  // Injected native memory API; queue ownership is implemented by the fabric.
  amdf_api_t api_ = {};
  // Retained domain registry used by the real allocation queue.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Context borrowing the fixture's native identities.
  iree_hal_amd_gpu_context_t context_ = {};
  // System upload scope.
  Scope system_ = {this, false};
  // SYSTEM storage is also device-local on a unified-memory device.
  bool unified_ = false;
  // Physical-local program scope.
  Scope local_ = {this, true};
  // Context-owned native scope handle array.
  amdf_memory_scope_t* local_handle_ = nullptr;
  // Production resident program registry and policies.
  iree_hal_amd_gpu_program_pool_t pool_ = {};
  // Number of attempted native acquisitions, including a selected failure.
  uint32_t allocate_count_ = 0;
  // Native acquisition ordinal to fail, or zero for success.
  uint32_t fail_allocate_ = 0;
  // Native owners remaining alive across asynchronous release.
  std::atomic<uint32_t> live_count_ = 0;
  // Thread on which the most recent native acquisition ran.
  std::thread::id allocation_thread_;
  // Thread on which program bytes were published after recording.
  std::thread::id publication_thread_;
  // Thread on which native backing was destroyed.
  std::thread::id release_thread_;
  // Explicitly observable entry into a held native acquisition.
  std::binary_semaphore* allocate_entered_ = nullptr;
  // Permission for that acquisition to complete.
  std::binary_semaphore* allocate_resume_ = nullptr;
  // Explicitly observable entry into the held native free.
  std::binary_semaphore* free_entered_ = nullptr;
  // Permission for that native free to complete.
  std::binary_semaphore* free_resume_ = nullptr;
};

TEST_P(ProgramLifetimeTest, QualifiedUnifiedStorageIsResident) {
  context_.memory_scope_count = 0;
  unified_ = true;
  iree_hal_amd_fabric_memory_options_t upload = {}, resident = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_context_query_program_memory(
      &context_, 0, &pool_.access, &upload, &resident));
  EXPECT_EQ(resident.scope, fabric_->system_scope);
  EXPECT_NE(resident.required_flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
  EXPECT_NE(resident.host_access, 0u);
}

TEST_P(ProgramLifetimeTest, UploadAvailabilityDoesNotPromiseLocalPlacement) {
  context_.memory_scope_count = 0;
  iree_hal_amd_fabric_memory_options_t upload = {}, resident = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_amd_gpu_context_query_program_memory(
                            &context_, 0, &pool_.access, &upload, &resident));
}

TEST_P(ProgramLifetimeTest, LastReleaseReturnsBeforeHeldNativeFree) {
  auto* program = new iree_hal_amd_gpu_program_t{};
  ASSERT_NO_FATAL_FAILURE(Record(program));
  EXPECT_EQ(allocation_thread_, std::this_thread::get_id());
  EXPECT_EQ(publication_thread_, std::this_thread::get_id());
  EXPECT_EQ(program->upload != nullptr, !PublishesLocalWrites());
  EXPECT_EQ(pool_.resident.scope, local_handle_);
  EXPECT_EQ(pool_.resident.host_access != 0, PublishesLocalWrites());
  EXPECT_EQ(pool_.head, program);
  std::binary_semaphore entered{0}, resume{0};
  free_entered_ = &entered;
  free_resume_ = &resume;
  iree_hal_amd_gpu_program_deinitialize(program);
  delete program;
  EXPECT_EQ(pool_.head, nullptr);
  entered.acquire();
  EXPECT_EQ(live_count_, PublishesLocalWrites() ? 1u : 2u);
  resume.release();
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(live_count_, 0u);
  EXPECT_NE(release_thread_, allocation_thread_);
}

TEST_P(ProgramLifetimeTest, RetiredInstanceNotifiesWhileCloneAllocationIsHeld) {
  ReuseWhileGrowing(false, false);
}

TEST_P(ProgramLifetimeTest, ReturnBeforeEnrollmentNotifiesWithoutAnotherScan) {
  ReuseWhileGrowing(false, true);
}

TEST_P(ProgramLifetimeTest, CancelledBorrowRestoresCausalReuseBeforeGrowth) {
  ReuseWhileGrowing(true, false);
}

TEST_P(ProgramLifetimeTest, CancellationBeforeEnrollmentPreservesReuseHint) {
  ReuseWhileGrowing(true, true);
}

TEST_P(ProgramLifetimeTest, OtherQueuePublicationDoesNotRetryCloneWaiters) {
  iree_hal_amd_gpu_program_t program = {};
  ASSERT_NO_FATAL_FAILURE(Record(&program));
  uint64_t frontier[3] = {};
  auto* initial = iree_hal_amd_gpu_program_try_acquire(&program, 2, frontier);
  ASSERT_NE(initial, nullptr);
  for (uint32_t queue = 0; queue < 2; ++queue) {
    ASSERT_EQ(iree_hal_amd_gpu_program_try_acquire(&program, queue, frontier),
              nullptr);
  }
  std::binary_semaphore entered{0}, resume{0};
  allocate_entered_ = &entered;
  allocate_resume_ = &resume;
  iree_hal_amd_gpu_program_growth_t growth[2] = {};
  std::promise<iree::Status> completed[2];
  uint32_t notifications[2] = {};
  for (uint32_t queue = 0; queue < 2; ++queue) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_begin(
        &program, queue, Completion(&completed[queue]), &growth[queue]));
    iree_hal_amd_gpu_program_grow_await_reuse(
        &growth[queue], [](void* data) { ++*static_cast<uint32_t*>(data); },
        &notifications[queue]);
  }
  entered.acquire();
  iree_hal_amd_gpu_program_publish(&program, initial, 2, 1);
  for (uint64_t epoch = 2; epoch <= 32; ++epoch) {
    frontier[2] = epoch - 1;
    auto* next = iree_hal_amd_gpu_program_try_acquire(&program, 2, frontier);
    EXPECT_EQ(next, initial);
    if (next) {
      iree_hal_amd_gpu_program_publish(&program, next, 2, epoch);
      iree_hal_amd_gpu_program_release(&program, next);
    }
  }
  // Q0/Q1 have no dependency on those new Q2 coordinates. Only Q2 can reuse
  // its ordered instance until its last outstanding use actually returns.
  EXPECT_EQ(notifications[0], 0u);
  EXPECT_EQ(notifications[1], 0u);
  iree_hal_amd_gpu_program_release(&program, initial);
  EXPECT_EQ(notifications[0], 1u);
  EXPECT_EQ(notifications[1], 1u);
  resume.release();
  for (uint32_t queue = 0; queue < 2; ++queue) {
    auto status = completed[queue].get_future().get();
    IREE_EXPECT_OK(
        iree_hal_amd_gpu_program_grow_end(&growth[queue], status.release()));
  }
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(notifications[0], 1u);
  EXPECT_EQ(notifications[1], 1u);
  iree_hal_amd_gpu_program_deinitialize(&program);
}

TEST_P(ProgramLifetimeTest, TrimReclaimsOnlyIdleClonesAndCompletedUpload) {
  iree_hal_amd_gpu_program_t program = {};
  ASSERT_NO_FATAL_FAILURE(Record(&program));
  uint64_t frontier[3] = {};
  auto* initial = iree_hal_amd_gpu_program_try_acquire(&program, 0, frontier);
  ASSERT_NE(initial, nullptr);
  iree_hal_amd_gpu_program_publish(&program, initial, 0, 1);
  ASSERT_NO_FATAL_FAILURE(Grow(&program, 1));
  auto* clone = iree_hal_amd_gpu_program_try_acquire(&program, 1, frontier);
  ASSERT_NE(clone, nullptr);
  ASSERT_NE(clone, initial);
  iree_hal_amd_gpu_program_publish(&program, clone, 1, 1);
  iree_hal_amd_gpu_program_pool_trim(&pool_);
  EXPECT_EQ(program.instances[clone->ordinal], clone);
  EXPECT_EQ(program.upload != nullptr, !PublishesLocalWrites());
  iree_hal_amd_gpu_program_release(&program, initial);
  iree_hal_amd_gpu_program_release(&program, clone);
  const auto clone_ordinal = clone->ordinal;
  iree_hal_amd_gpu_program_pool_trim(&pool_);
  EXPECT_EQ(program.instances[clone_ordinal], nullptr);
  EXPECT_EQ(program.queues[1].preferred, initial);
  EXPECT_EQ(program.upload, nullptr);
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(live_count_, 1u);
  iree_hal_amd_gpu_program_deinitialize(&program);
}

TEST_P(ProgramLifetimeTest, PartialConstructionReleasesItsAcquiredOwners) {
  const uint32_t acquisition_count = PublishesLocalWrites() ? 1 : 2;
  for (uint32_t failure = 1; failure <= acquisition_count; ++failure) {
    SCOPED_TRACE(failure);
    fail_allocate_ = allocate_count_ + failure;
    iree_hal_amd_gpu_program_t program = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_hal_amd_gpu_program_initialize(&pool_, 256, 4096, 16, &program));
    iree_hal_amd_gpu_program_deinitialize(&program);
    iree_hal_amd_gpu_context_join_memory(&context_);
    EXPECT_EQ(live_count_, 0u);
    EXPECT_EQ(pool_.head, nullptr);
  }
}

TEST_P(ProgramLifetimeTest, PendingClonesHaveDistinctTrimProtectedSlots) {
  iree_hal_amd_gpu_program_t program = {};
  ASSERT_NO_FATAL_FAILURE(Record(&program));
  uint64_t frontier[3] = {};
  auto* initial = iree_hal_amd_gpu_program_try_acquire(&program, 0, frontier);
  ASSERT_NE(initial, nullptr);
  iree_hal_amd_gpu_program_publish(&program, initial, 0, 1);

  std::binary_semaphore entered{0}, resume{0};
  allocate_entered_ = &entered;
  allocate_resume_ = &resume;
  iree_hal_amd_gpu_program_growth_t first = {}, second = {};
  std::promise<iree::Status> first_completed, second_completed;
  IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_begin(
      &program, 1, Completion(&first_completed), &first));
  entered.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_begin(
      &program, 2, Completion(&second_completed), &second));
  EXPECT_EQ(program.queues[1].growth, &first);
  EXPECT_EQ(program.queues[2].growth, &second);
  EXPECT_NE(first.instance, second.instance);
  EXPECT_NE(first.instance->ordinal, second.instance->ordinal);
  iree_hal_amd_gpu_program_pool_trim(&pool_);
  EXPECT_EQ(program.instances[first.instance->ordinal], first.instance);
  EXPECT_EQ(program.instances[second.instance->ordinal], second.instance);
  EXPECT_EQ(iree_hal_amd_gpu_program_try_acquire(&program, 1, frontier),
            nullptr);
  resume.release();
  auto first_status = first_completed.get_future().get();
  auto second_status = second_completed.get_future().get();
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_program_grow_end(&first, first_status.release()));
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_program_grow_end(&second, second_status.release()));
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(program.queues[1].growth, nullptr);
  EXPECT_EQ(program.queues[2].growth, nullptr);
  EXPECT_FALSE(first.instance->reserved);
  EXPECT_FALSE(second.instance->reserved);
  iree_hal_amd_gpu_program_release(&program, initial);
  iree_hal_amd_gpu_program_deinitialize(&program);
}

TEST_P(ProgramLifetimeTest, FailedCloneReturnsItsSlotForRetry) {
  iree_hal_amd_gpu_program_t program = {};
  ASSERT_NO_FATAL_FAILURE(Record(&program));
  uint64_t frontier[3] = {};
  auto* initial = iree_hal_amd_gpu_program_try_acquire(&program, 0, frontier);
  ASSERT_NE(initial, nullptr);
  iree_hal_amd_gpu_program_publish(&program, initial, 0, 1);
  fail_allocate_ = allocate_count_ + 1;
  iree_hal_amd_gpu_program_growth_t growth = {};
  std::promise<iree::Status> completed;
  IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_begin(
      &program, 1, Completion(&completed), &growth));
  const auto ordinal = growth.instance->ordinal;
  auto status = completed.get_future().get();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amd_gpu_program_grow_end(&growth, status.release()));
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(program.instances[ordinal], nullptr);
  EXPECT_EQ(program.queues[1].growth, nullptr);
  ASSERT_NO_FATAL_FAILURE(Grow(&program, 1));
  EXPECT_NE(program.instances[ordinal], nullptr);
  iree_hal_amd_gpu_program_release(&program, initial);
  iree_hal_amd_gpu_program_deinitialize(&program);
}

TEST_P(ProgramLifetimeTest, CancelledBorrowCanRequireMoreInstancesThanQueues) {
  iree_hal_amd_gpu_program_t program = {};
  ASSERT_NO_FATAL_FAILURE(Record(&program));
  uint64_t frontier[3] = {};
  auto* first = iree_hal_amd_gpu_program_try_acquire(&program, 0, frontier);
  ASSERT_NE(first, nullptr);
  iree_hal_amd_gpu_program_publish(&program, first, 0, 1);
  ASSERT_NO_FATAL_FAILURE(Grow(&program, 1));
  auto* second = iree_hal_amd_gpu_program_try_acquire(&program, 1, frontier);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(second, first);
  iree_hal_amd_gpu_program_publish(&program, second, 1, 1);

  // Q2 borrows Q1's live instance using a native dependency, but has not
  // published. Q1 can independently reuse Q0's instance after waiting on Q0.
  frontier[1] = 1;
  auto* borrowed = iree_hal_amd_gpu_program_try_acquire(&program, 2, frontier);
  ASSERT_EQ(borrowed, second);
  frontier[0] = 1;
  auto* replaced = iree_hal_amd_gpu_program_try_acquire(&program, 1, frontier);
  ASSERT_EQ(replaced, first);
  iree_hal_amd_gpu_program_publish(&program, replaced, 1, 2);
  iree_hal_amd_gpu_program_cancel(&program, borrowed);

  // Canceling Q2 does not make either live Q1 writer reusable by independent
  // Q0/Q2 work. Q0 consumes the final initially provisioned physical slot.
  frontier[1] = 0;
  EXPECT_EQ(iree_hal_amd_gpu_program_try_acquire(&program, 0, frontier),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Grow(&program, 0));
  auto* third = iree_hal_amd_gpu_program_try_acquire(&program, 0, frontier);
  ASSERT_NE(third, nullptr);
  iree_hal_amd_gpu_program_publish(&program, third, 0, 2);
  EXPECT_EQ(program.capacity, pool_.queue_count);

  const auto* queues = program.queues;
  const auto source = iree_hal_amd_gpu_program_source(&program);
  frontier[0] = 0;
  EXPECT_EQ(iree_hal_amd_gpu_program_try_acquire(&program, 2, frontier),
            nullptr);
  ASSERT_NO_FATAL_FAILURE(Grow(&program, 2));
  auto* fourth = iree_hal_amd_gpu_program_try_acquire(&program, 2, frontier);
  ASSERT_NE(fourth, nullptr);
  EXPECT_GE(fourth->ordinal, pool_.queue_count);
  EXPECT_GT(program.capacity, pool_.queue_count);
  EXPECT_EQ(program.queues, queues);
  EXPECT_EQ(program.initial, first);
  EXPECT_EQ(iree_hal_amd_gpu_program_source(&program).address, source.address);
  iree_hal_amd_gpu_program_publish(&program, fourth, 2, 1);

  // Every actual publication still owns a lease; cancellation consumed only
  // the unpublished borrow. Trim then returns all independent spare backing.
  for (auto* instance : {first, second, replaced, third, fourth}) {
    iree_hal_amd_gpu_program_release(&program, instance);
  }
  iree_hal_amd_gpu_program_pool_trim(&pool_);
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(live_count_, 1u);
  EXPECT_EQ(iree_hal_amd_gpu_program_source(&program).address, first->address);
  for (uint32_t i = 1; i < program.capacity; ++i) {
    EXPECT_EQ(program.instances[i], nullptr);
  }
  iree_hal_amd_gpu_program_deinitialize(&program);
}

INSTANTIATE_TEST_SUITE_P(LocalHostView, ProgramLifetimeTest,
                         ::testing::Values(LocalHostView::kUnavailable,
                                           LocalHostView::kPublishable,
                                           LocalHostView::kUnpublishable));

}  // namespace
