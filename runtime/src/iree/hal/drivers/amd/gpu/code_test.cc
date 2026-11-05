// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/code.h"

#include <array>
#include <atomic>
#include <cstring>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/hal/drivers/amd/gpu/device_failure.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/code.h"
#include "iree/hal/drivers/amd/gpu/queue/sdma/queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Controls native allocation and submission dependencies. Code loading, SDMA
// encoding, publication, retirement, and allocation-domain release are real.
class CodeTest : public ::testing::Test {
 protected:
  struct Memory {
    // Native bytes, distinct for queue control, final code, and staging.
    std::vector<uint8_t> bytes;
    // Placement requirements received from the production allocation request.
    amdf_memory_flags_t flags = 0;
    // Native allocation remains owned until explicit destruction.
    bool allocated = false;
    // Native host view remains owned until explicit destruction.
    bool mapped = false;
    // Calling thread that acquired this native backing.
    std::thread::id allocation_thread;
  };

  static Memory& GetMemory(amdf_memory_t* memory) {
    return *reinterpret_cast<Memory*>(memory);
  }

  void SetUp() override {
    api_.memory_create =
        [](amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
           amdf_memory_t** out_memory, amdf_memory_release_info_t* release) {
          if (release) {
            *release = {};
          }
          auto* self = reinterpret_cast<CodeTest*>(scope);
          if (++self->acquisitions_ == self->failed_acquisition_) {
            return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
          }
          auto& memory = self->memories_[self->acquisitions_ - 1];
          memory.bytes.assign(info->byte_length, 0xa5);
          memory.flags = info->required_flags;
          memory.allocated = true;
          memory.allocation_thread = std::this_thread::get_id();
          *out_memory = reinterpret_cast<amdf_memory_t*>(&memory);
          return AMDF_STATUS_OK;
        };
    api_.memory_query_info = [](amdf_memory_t* memory,
                                amdf_memory_info_t* info) {
      info->byte_length = GetMemory(memory).bytes.size();
      info->alignment = 64;
      info->flags = GetMemory(memory).flags;
      return AMDF_STATUS_OK;
    };
    api_.memory_query_address = [](amdf_memory_t* memory, uint32_t,
                                   amdf_memory_address_kind_t,
                                   uint64_t* address) {
      *address = reinterpret_cast<uintptr_t>(GetMemory(memory).bytes.data());
      return AMDF_STATUS_OK;
    };
    api_.memory_map = [](amdf_memory_t* memory, const amdf_memory_map_info_t*,
                         amdf_host_mapping_t** mapping) {
      GetMemory(memory).mapped = true;
      *mapping = reinterpret_cast<amdf_host_mapping_t*>(memory);
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_query_info = [](amdf_host_mapping_t* mapping,
                                      amdf_host_mapping_info_t* info) {
      auto& memory = *reinterpret_cast<Memory*>(mapping);
      info->pointer = memory.bytes.data();
      info->byte_length = memory.bytes.size();
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_cache_control = [](amdf_host_mapping_t*,
                                         amdf_host_cache_operation_t operation,
                                         uint64_t, uint64_t) {
      EXPECT_EQ(operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
      return AMDF_STATUS_OK;
    };
    api_.memory_query_pair_info = [](const amdf_memory_site_t*,
                                     const amdf_memory_site_t*,
                                     amdf_memory_pair_info_t* info) {
      info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
      info->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
      info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_destroy = [](amdf_host_mapping_t* mapping) {
      auto& memory = *reinterpret_cast<Memory*>(mapping);
      EXPECT_TRUE(memory.mapped);
      memory.mapped = false;
      return AMDF_STATUS_OK;
    };
    api_.memory_destroy = [](amdf_memory_t* handle,
                             amdf_memory_release_info_t* release) {
      if (release) {
        *release = {};
      }
      auto& memory = GetMemory(handle);
      EXPECT_TRUE(memory.allocated);
      EXPECT_FALSE(memory.mapped);
      memory.allocated = false;
      memory.bytes.clear();
      return AMDF_STATUS_OK;
    };
    api_.instance_destroy = [](amdf_instance_t*) { return AMDF_STATUS_OK; };
    gpu_.kernel_queue_create = [](amdf_device_t* device,
                                  const amdf_gpu_kernel_queue_create_info_t*,
                                  amdf_kernel_queue_t** queue) {
      *queue = reinterpret_cast<amdf_kernel_queue_t*>(device);
      return AMDF_STATUS_OK;
    };
    api_.kernel_queue_query_info = [](amdf_kernel_queue_t*,
                                      amdf_kernel_queue_info_t* info) {
      info->maximum_pending_submission_count = 64;
      info->notification_types = UINT64_C(1) << AMDF_NATIVE_EVENT_TYPE_EVENTFD;
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_submit =
        [](amdf_kernel_queue_t* queue,
           const amdf_gpu_kernel_queue_submission_info_t* info,
           uint64_t* point) {
          auto* self = reinterpret_cast<CodeTest*>(queue);
          if (!amdf_status_is_ok(self->submit_status_)) {
            return self->submit_status_;
          }
          const auto& command = info->commands[0];
          const auto* words = reinterpret_cast<const uint32_t*>(
              GetMemory(command.memory).bytes.data() + command.byte_offset);
          EXPECT_EQ(words[0], 1u);  // Native SDMA COPY_LINEAR.
          const uint64_t source = words[3] | (uint64_t{words[4]} << 32);
          const uint64_t target = words[5] | (uint64_t{words[6]} << 32);
          self->copied_bytes_ += words[1] + 1;
          std::memcpy(reinterpret_cast<void*>(target),
                      reinterpret_cast<const void*>(source), words[1] + 1);
          *point = ++self->submitted_;
          return AMDF_STATUS_OK;
        };
    api_.kernel_queue_request_notification =
        [](amdf_kernel_queue_t* queue, uint64_t, const amdf_native_event_t*) {
          auto* self = reinterpret_cast<CodeTest*>(queue);
          self->submitted_notification_.release();
          return self->notification_status_;
        };
    api_.kernel_queue_refresh_status = [](amdf_kernel_queue_t* queue,
                                          amdf_kernel_queue_status_t* info) {
      info->retired_submission =
          reinterpret_cast<CodeTest*>(queue)->retired_.load();
      return AMDF_STATUS_OK;
    };
    api_.kernel_queue_wait = [](amdf_kernel_queue_t* queue, uint64_t point,
                                uint64_t, uint64_t) {
      return point <= reinterpret_cast<CodeTest*>(queue)->retired_.load()
                 ? AMDF_STATUS_OK
                 : amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    };
    api_.kernel_queue_destroy = [](amdf_kernel_queue_t*) {
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_inactivate = [](amdf_kernel_queue_t* queue) {
      auto* self = reinterpret_cast<CodeTest*>(queue);
      EXPECT_TRUE(self->memories_[1].allocated);
      EXPECT_TRUE(self->memories_[2].allocated);
      return AMDF_STATUS_OK;
    };
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = iree_allocator_system();
    fabric_->api = &api_;
    context_.fabric = fabric_;
    context_.api = &api_;
    context_.gpu = &gpu_;
    context_.device = reinterpret_cast<amdf_device_t*>(this);
    context_.families = &family_;
    context_.kernel_notification.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    context_.event_sink = {[](void*, const iree_hal_device_event_t*) {},
                           nullptr};
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    staging_.scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    staging_.profile.allocation.maximum_byte_length = UINT32_MAX;
    staging_.profile.allocation.byte_length_granularity = 64;
    staging_.profile.allocation.minimum_alignment = 64;
    staging_.profile.allocation.maximum_alignment = 4096;
    staging_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    resident_ = staging_;
    resident_.host_access = 0;
    resident_.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_queue_allocate(
        &context_, 0, AMDF_QUEUE_PUBLICATION_MODE_KERNEL, &staging_,
        &publication_, 0, iree_allocator_system(), &queue_));
    source_ = iree_hal_amd_gpu_sdma_queue_publication_source(queue_);
    membership_.context = &context_;
    membership_.host_allocator = iree_allocator_system();
    iree_hal_amd_gpu_queue_membership_initialize(&membership_);
    iree_hal_amd_gpu_device_failure_initialize(
        &context_, nullptr, 30000, &publication_, &completion_, &membership_,
        queue_, nullptr, &failure_);
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        1, &source_, {}, iree_allocator_system(), &publication_));
  }

  void TearDown() override {
    iree_hal_amd_gpu_code_upload_free(&context_, upload_);
    iree_hal_amd_fabric_memory_deinitialize(context_.api, context_.event_sink,
                                            &code_);
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    // Only the controlled native dependency can establish quiescence after an
    // accepted failure. The loader must leave both operands untouched until it.
    retired_ = submitted_;
    EXPECT_EQ(iree_hal_amd_gpu_sdma_queue_free(queue_), AMDF_STATUS_OK);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    iree_hal_amd_gpu_device_failure_deinitialize(&failure_);
    EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_deinitialize(&membership_));
    iree_hal_amd_fabric_release(fabric_);
    for (const auto& memory : memories_) {
      EXPECT_FALSE(memory.allocated);
    }
  }

  iree_status_t Allocate() {
    return iree_hal_amd_gpu_code_allocate(&context_, &resident_, &staging_,
                                          12345, 64, iree_allocator_system(),
                                          &code_, &upload_, &contents_);
  }

  iree_status_t Publish(iree_hal_amd_gpu_sdma_queue_t* queue) {
    return iree_hal_amd_gpu_code_publish(&context_, queue, 0, 1, &code_,
                                         &upload_);
  }

  void JoinFailure() {
    // This fixture has no healthy observer. Deliver the same final source
    // failure and terminal callback used by the real completion worker.
    auto source = iree_hal_amd_gpu_sdma_queue_completion_source(queue_);
    iree_hal_amd_gpu_completion_wait_t wait = {};
    iree_hal_amd_gpu_completion_t completion = {};
    iree_slim_mutex_initialize(&completion.drain_mutex);
    auto poll = [&](amdf_status_t status) {
      iree_slim_mutex_lock(&completion.drain_mutex);
      iree_hal_amd_gpu_completion_scope_t scope = {
          .completion = &completion,
          .source = &source,
          .fn = source.fn,
      };
      const bool ready =
          source.fn(source.user_data, status, &wait, nullptr, &scope);
      EXPECT_FALSE(iree_hal_amd_gpu_completion_scope_finish(&scope));
      iree_slim_mutex_unlock(&completion.drain_mutex);
      return ready;
    };
    const amdf_status_t error =
        amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
    iree_hal_amd_gpu_context_fail_execution(&context_, error, "native wait");
    poll(error);
    iree_slim_mutex_deinitialize(&completion.drain_mutex);
    context_.execution_failure.join(context_.execution_failure.user_data,
                                    error);
    EXPECT_EQ(membership_.terminal.status, AMDF_STATUS_OK);
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
  }

  // Native API dependencies used by both the loader and real SDMA transport.
  amdf_api_t api_ = {};
  // Native GPU submission dependency.
  amdf_gpu_api_t gpu_ = {};
  // Real terminal ownership registry; this fixture has only its SDMA/code
  // owners.
  iree_hal_amd_gpu_queue_membership_t membership_ = {};
  // Native observation is explicitly driven by the fixture, without a worker.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Installed before publication, as required by the production contract.
  iree_hal_amd_gpu_device_failure_t failure_ = {};
  // Queue control, final code, and temporary staging, in acquisition order.
  std::array<Memory, 3> memories_;
  // Number of native acquisition attempts, including rejected requests.
  uint32_t acquisitions_ = 0;
  // One-based acquisition to reject, or zero for success.
  uint32_t failed_acquisition_ = 0;
  // Submission rejection, before any native consumer exists.
  amdf_status_t submit_status_ = AMDF_STATUS_OK;
  // Wake registration failure after native submission acceptance.
  amdf_status_t notification_status_ = AMDF_STATUS_OK;
  // Number of accepted native submissions.
  uint64_t submitted_ = 0;
  // Total bytes copied by the native command dependency.
  uint64_t copied_bytes_ = 0;
  // Explicit native retirement proof, independent of logical completion.
  std::atomic<uint64_t> retired_ = 0;
  // Native notification enrollment has observed the published queue record.
  std::binary_semaphore submitted_notification_{0};
  // Real allocation-domain registry and its native allocation owner.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Native ownership borrowed by the production loader and transfer queue.
  iree_hal_amd_gpu_context_t context_ = {};
  // Captured native transfer-family encoding capability.
  amdf_queue_family_info_t family_ = {};
  // Host-mapped staging and control allocation policy.
  iree_hal_amd_fabric_memory_options_t staging_ = {};
  // Device-local final allocation policy, independently host-mappable.
  iree_hal_amd_fabric_memory_options_t resident_ = {};
  // Final code allocation owned by the loading caller.
  iree_hal_amd_fabric_memory_t code_ = {};
  // Temporary upload state consumed by publication or cancellation.
  iree_hal_amd_gpu_code_upload_t* upload_ = nullptr;
  // Logical initialization span returned by the loader.
  iree_byte_span_t contents_ = {};
  // Production native transfer queue.
  iree_hal_amd_gpu_sdma_queue_t* queue_ = nullptr;
  // Fixed publication source borrowed through worker shutdown.
  iree_hal_amd_gpu_publication_source_t source_ = {};
  // Production deferred publication owner, including terminal failure delivery.
  iree_hal_amd_gpu_publication_t publication_ = {};
};

TEST_F(CodeTest, MappedLocalCodeNeedsNoUpload) {
  resident_.host_access = staging_.host_access;
  IREE_ASSERT_OK(Allocate());
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(contents_.data, code_.contents.data);
  EXPECT_EQ(contents_.data_length, 12345u);
  EXPECT_EQ(acquisitions_, 2u);
  EXPECT_EQ(memories_[1].flags, AMDF_MEMORY_FLAG_DEVICE_LOCAL);
  EXPECT_EQ(memories_[1].allocation_thread, std::this_thread::get_id());
  IREE_ASSERT_OK(Publish(nullptr));
  EXPECT_EQ(submitted_, 0u);
}

TEST_F(CodeTest, StagingIsReleasedOnlyAfterCopyRetires) {
  IREE_ASSERT_OK(Allocate());
  ASSERT_NE(upload_, nullptr);
  EXPECT_EQ(memories_[1].allocation_thread, std::this_thread::get_id());
  EXPECT_EQ(memories_[2].allocation_thread, std::this_thread::get_id());
  EXPECT_EQ(code_.mapping, nullptr);
  EXPECT_EQ(contents_.data_length, 12345u);
  std::memset(contents_.data, 0x3b, contents_.data_length);
  std::thread observer([&] {
    submitted_notification_.acquire();
    EXPECT_TRUE(memories_[1].allocated);
    EXPECT_TRUE(memories_[2].allocated);
    EXPECT_TRUE(memories_[2].mapped);
    retired_ = submitted_;
    auto source = iree_hal_amd_gpu_sdma_queue_completion_source(queue_);
    iree_hal_amd_gpu_completion_wait_t wait = {};
    iree_hal_amd_gpu_completion_t completion = {};
    iree_slim_mutex_initialize(&completion.drain_mutex);
    auto poll = [&](amdf_status_t status) {
      iree_slim_mutex_lock(&completion.drain_mutex);
      iree_hal_amd_gpu_completion_scope_t scope = {
          .completion = &completion,
          .source = &source,
          .fn = source.fn,
      };
      const bool ready =
          source.fn(source.user_data, status, &wait, nullptr, &scope);
      EXPECT_FALSE(iree_hal_amd_gpu_completion_scope_finish(&scope));
      iree_slim_mutex_unlock(&completion.drain_mutex);
      return ready;
    };
    while (poll(AMDF_STATUS_OK)) {
    }
    iree_slim_mutex_deinitialize(&completion.drain_mutex);
  });
  iree_status_t status = Publish(queue_);
  observer.join();
  IREE_ASSERT_OK(status);
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(copied_bytes_, 12345u);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_FALSE(memories_[2].allocated);
  for (size_t i = 0; i < memories_[1].bytes.size(); ++i) {
    EXPECT_EQ(memories_[1].bytes[i], i < 12345 ? 0x3b : 0xa5);
  }
}

TEST_F(CodeTest, StagingAcquisitionFailureRollsBackFinalCode) {
  failed_acquisition_ = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Allocate());
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(contents_.data, nullptr);
  EXPECT_FALSE(memories_[1].allocated);
}

TEST_F(CodeTest, MissingTransferQueueCancelsStaging) {
  IREE_ASSERT_OK(Allocate());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Publish(nullptr));
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(upload_, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_FALSE(memories_[2].allocated);
}

TEST_F(CodeTest, RejectedCopyReturnsStagingAndFinalOwnership) {
  IREE_ASSERT_OK(Allocate());
  submit_status_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Publish(queue_));
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(upload_, nullptr);
  EXPECT_NE(code_.handle, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_FALSE(memories_[2].allocated);
  EXPECT_EQ(submitted_, 0u);
}

TEST_F(CodeTest, AcceptedCopyFailurePreservesBothNativeOperands) {
  IREE_ASSERT_OK(Allocate());
  notification_status_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Publish(queue_));
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_EQ(code_.mapping, nullptr);
  EXPECT_EQ(code_.contents.data, nullptr);
  EXPECT_EQ(code_.info.byte_length, 12352u);
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(retired_.load(), 0u);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_TRUE(memories_[2].allocated);
  EXPECT_TRUE(memories_[2].mapped);
  EXPECT_FALSE(iree_hal_amd_gpu_sdma_queue_is_idle(queue_));
  JoinFailure();
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(retired_.load(), 0u);
  EXPECT_FALSE(memories_[1].allocated);
  EXPECT_FALSE(memories_[2].allocated);
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_queue_is_idle(queue_));
}

TEST_F(CodeTest, FailedCachePublicationKeepsItsTargetUntilNativeStop) {
  // The cold cache queue uses the same allocation/context owner as SDMA.
  resident_.host_access = staging_.host_access;
  context_.target_info.gfx_ip.major = 9;
  // Cache publication consumes opaque aligned device addresses. This fixture
  // does not execute SDMA copies through these independent native addresses.
  api_.memory_query_address = [](amdf_memory_t* memory, uint32_t,
                                 amdf_memory_address_kind_t,
                                 uint64_t* address) {
    *address = iree_align_uint64(
        reinterpret_cast<uintptr_t>(GetMemory(memory).bytes.data()), 64);
    return AMDF_STATUS_OK;
  };
  IREE_ASSERT_OK(Allocate());
  struct NativeCache {
    // Host-only native dependency storage; no packet is executed.
    alignas(64) std::array<uint8_t, 4096> ring = {};
    // Native queue read frontier, left unconsumed across failure.
    uint64_t read_index = 0;
    // Host producer write frontier.
    uint64_t write_index = 0;
    // Host-only doorbell operand.
    uint64_t doorbell = 0;
    // Number of native access-cessation operations.
    uint32_t stops = 0;
  } native;
  context_.device = reinterpret_cast<amdf_device_t*>(&native);
  gpu_.event_create = [](amdf_device_t* device,
                         const amdf_gpu_event_create_info_t*,
                         amdf_gpu_event_t** out_event) {
    *out_event = reinterpret_cast<amdf_gpu_event_t*>(device);
    return AMDF_STATUS_OK;
  };
  gpu_.event_query_info = [](amdf_gpu_event_t*, amdf_gpu_event_info_t* info) {
    info->format = AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL;
    info->format_version = 1;
    return AMDF_STATUS_OK;
  };
  gpu_.event_wait = [](amdf_gpu_event_t*, uint64_t) {
    return amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  };
  gpu_.event_signal = [](amdf_gpu_event_t*) { return AMDF_STATUS_OK; };
  gpu_.event_destroy = [](amdf_gpu_event_t*) { return AMDF_STATUS_OK; };
  gpu_.user_queue_create = [](amdf_device_t* device,
                              const amdf_gpu_user_queue_create_info_t*,
                              amdf_user_queue_t** out_queue) {
    *out_queue = reinterpret_cast<amdf_user_queue_t*>(device);
    return AMDF_STATUS_OK;
  };
  api_.user_queue_query_info = [](amdf_user_queue_t*, amdf_user_queue_info_t*) {
    return AMDF_STATUS_OK;
  };
  api_.user_queue_map = [](amdf_user_queue_t* queue, amdf_device_t*,
                           amdf_user_queue_mapping_t** out_mapping) {
    *out_mapping = reinterpret_cast<amdf_user_queue_mapping_t*>(queue);
    return AMDF_STATUS_OK;
  };
  api_.user_queue_mapping_query_info =
      [](amdf_user_queue_mapping_t* mapping,
         amdf_user_queue_mapping_info_t* info) {
        auto* native = reinterpret_cast<NativeCache*>(mapping);
        info->ring_address = reinterpret_cast<uintptr_t>(native->ring.data());
        info->ring_byte_length = native->ring.size();
        info->read_index_address =
            reinterpret_cast<uintptr_t>(&native->read_index);
        info->write_index_address =
            reinterpret_cast<uintptr_t>(&native->write_index);
        info->doorbell_address = reinterpret_cast<uintptr_t>(&native->doorbell);
        info->index_bits = info->doorbell_bits = 64;
        return AMDF_STATUS_OK;
      };
  api_.user_queue_wait_consumed = [](amdf_user_queue_t*, uint64_t, uint64_t,
                                     uint64_t) {
    ADD_FAILURE() << "inactivated cache queue must not wait for firmware";
    return amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  };
  api_.user_queue_mapping_destroy = [](amdf_user_queue_mapping_t*) {
    return AMDF_STATUS_OK;
  };
  api_.user_queue_destroy = [](amdf_user_queue_t*) { return AMDF_STATUS_OK; };
  gpu_.user_queue_inactivate = [](amdf_user_queue_t* queue) {
    ++reinterpret_cast<NativeCache*>(queue)->stops;
    return AMDF_STATUS_OK;
  };
  iree_hal_amd_gpu_aql_code_t* cache = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_code_allocate(
      &context_, 0, &staging_, iree_allocator_system(), &cache));
  failure_.code = cache;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_aql_code_publish(cache, &code_));
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  JoinFailure();
  EXPECT_EQ(native.stops, 1u);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_EQ(iree_hal_amd_gpu_aql_code_free(cache), AMDF_STATUS_OK);
  EXPECT_FALSE(memories_[1].allocated);
  EXPECT_FALSE(memories_[2].allocated);
}

}  // namespace
