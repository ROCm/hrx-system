// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/code.h"

#include <array>
#include <cstring>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/hal/driver/amd/fabric/event.h"
#include "iree/hal/driver/amd/gpu/code_queue.h"
#include "iree/hal/driver/amd/gpu/queue/aql/packet.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Controls native allocation and queue dependencies. Code loading, command
// encoding, publication, retirement, and allocation-domain release are real.
class CodeTest : public ::testing::Test {
 protected:
  struct Memory {
    // Native bytes, distinct for queue control and final code.
    std::vector<uint8_t> bytes;
    // Placement requirements received from the production allocation request.
    amdf_memory_flags_t flags = 0;
    // Native allocation remains owned until explicit destruction.
    bool allocated = false;
    // Native host view remains owned until explicit destruction.
    bool mapped = false;
    // Completed host flushes before native code publication.
    uint32_t flush_count = 0;
    // Calling thread that acquired this native backing.
    std::thread::id allocation_thread;
  };

  struct NativeCache {
    // Host-only native dependency storage; no packet is executed.
    alignas(64) std::array<uint8_t, 4096> ring = {};
    // Native queue read frontier, left unconsumed until inactivation.
    uint64_t read_index = 0;
    // Host producer write frontier.
    uint64_t write_index = 0;
    // Host-only doorbell operand.
    uint64_t doorbell = 0;
    // USER completion predicate written while simulating a lost notification.
    iree_hal_amd_gpu_aql_signal_t* signal = nullptr;
    // Native event result returned after each simulated wait.
    amdf_status_t event_wait_status = AMDF_STATUS_OK;
    // Whether the first wait observes completion without a notification.
    bool complete_during_wait = false;
    // Relative native timeout received by the most recent event wait.
    uint64_t wait_timeout_ns = 0;
    // Number of native event waits.
    uint32_t wait_count = 0;
    // Number of native access-cessation operations.
    uint32_t stops = 0;
  };

  static Memory& GetMemory(amdf_memory_t* memory) {
    return *reinterpret_cast<Memory*>(memory);
  }

  void SetUp() override {
    api_.memory_scope_query_info = [](amdf_memory_scope_t*,
                                      amdf_memory_scope_info_t* info) {
      info->kind = AMDF_MEMORY_SCOPE_KIND_LOCAL;
      info->memory_profile_count = 1;
      info->capacity_byte_length = UINT32_MAX;
      return AMDF_STATUS_OK;
    };
    api_.memory_create = [](amdf_memory_scope_t* scope,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out_memory,
                            amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
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
      info->properties = AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_cache_control = [](amdf_host_mapping_t* mapping,
                                         amdf_host_cache_operation_t operation,
                                         uint64_t, uint64_t) {
      EXPECT_EQ(operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
      ++reinterpret_cast<Memory*>(mapping)->flush_count;
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
    gpu_.kernel_queue_create =
        [](amdf_device_t* device,
           const amdf_gpu_kernel_queue_create_info_t* info,
           amdf_kernel_queue_t** queue) {
          EXPECT_EQ(info->queue_family_ordinal, 0u);
          *queue = reinterpret_cast<amdf_kernel_queue_t*>(device);
          return AMDF_STATUS_OK;
        };
    api_.kernel_queue_query_info = [](amdf_kernel_queue_t* queue,
                                      amdf_kernel_queue_info_t* info) {
      auto* self = reinterpret_cast<CodeTest*>(queue);
      info->maximum_pending_submission_count = 1;
      info->notification_types = UINT64_C(1)
                                 << self->context_.kernel_notification.type;
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
          EXPECT_EQ(info->command_count, 1u);
          const auto& command = info->commands[0];
          EXPECT_EQ(command.byte_length % 32, 0u);
          const auto* words = reinterpret_cast<const uint32_t*>(
              GetMemory(command.memory).bytes.data() + command.byte_offset);
          const size_t word_count = command.byte_length / 4;
          for (size_t i = 0; i < word_count;) {
            const uint32_t header = words[i];
            EXPECT_EQ(header >> 30, 3u);
            const size_t count = ((header >> 16) & 0x3fff) + 2;
            if (i + count > word_count) {
              ADD_FAILURE() << "native packet exceeds command extent";
              break;
            }
            switch ((header >> 8) & 0xff) {
              case 0x37: {  // WRITE_DATA, confirmed TC/L2 memory destination.
                EXPECT_EQ(words[i + 1], 2u << 8 | 1u << 20);
                const uint64_t target =
                    words[i + 2] | (uint64_t{words[i + 3]} << 32);
                std::memcpy(reinterpret_cast<void*>(target), words + i + 4,
                            (count - 4) * 4);
                self->copied_bytes_ += (count - 4) * 4;
                break;
              }
              case 0x58:  // ACQUIRE_MEM: invalidate code after its last write.
                EXPECT_EQ(count, 8u);
                EXPECT_NE(words[i + 7] & 1u, 0u);
                ++self->cache_actions_;
                EXPECT_EQ(self->copied_bytes_, self->expected_copy_length_);
                if (self->resident_.host_access) {
                  EXPECT_EQ(self->memories_[1].flush_count, 1u);
                }
                break;
              case 0x10:  // NOP aligns KERNEL commands.
                break;
              default:
                ADD_FAILURE() << "unexpected bootstrap opcode " << header;
                break;
            }
            i += count;
          }
          *point = ++self->submitted_;
          if (self->auto_retire_) {
            self->retired_ = *point;
          }
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
      auto* self = reinterpret_cast<CodeTest*>(queue);
      info->retired_submission = self->retired_;
      info->terminal_status = self->retirement_status_;
      return AMDF_STATUS_OK;
    };
    api_.kernel_queue_wait = [](amdf_kernel_queue_t* queue, uint64_t point,
                                uint64_t, uint64_t) {
      return point <= reinterpret_cast<CodeTest*>(queue)->retired_
                 ? AMDF_STATUS_OK
                 : amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    };
    api_.kernel_queue_destroy = [](amdf_kernel_queue_t*) {
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_inactivate = [](amdf_kernel_queue_t* queue) {
      auto* self = reinterpret_cast<CodeTest*>(queue);
      EXPECT_TRUE(self->memories_[0].allocated);
      ++self->stops_;
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
    context_.target_info.gfx_ip.major = 11;
    family_.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
    IREE_ASSERT_OK(iree_async_event_native_initialize(&context_.kernel_event));
    context_.kernel_notification =
        iree_hal_amd_fabric_event_native_handle(&context_.kernel_event);
    context_.event_sink = {[](void*, const iree_hal_device_event_t*) {},
                           nullptr};
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    control_.scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    control_.profile.memory_class = AMDF_MEMORY_CLASS_LOCAL;
    control_.profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    control_.profile.allocation.maximum_byte_length = UINT32_MAX;
    control_.profile.allocation.byte_length_granularity = 64;
    control_.profile.allocation.minimum_alignment = 64;
    control_.profile.allocation.maximum_alignment = 4096;
    control_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    resident_ = control_;
    resident_.host_access = 0;
    resident_.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    IREE_ASSERT_OK(iree_hal_amd_gpu_code_queue_allocate(
        &context_, 0, AMDF_QUEUE_PUBLICATION_MODE_KERNEL, &control_,
        iree_allocator_system(), &queue_));
  }

  void TearDown() override {
    iree_hal_amd_gpu_code_upload_free(upload_);
    iree_hal_amd_fabric_memory_deinitialize(context_.api, context_.event_sink,
                                            &code_);
    EXPECT_EQ(iree_hal_amd_gpu_code_queue_free(queue_), AMDF_STATUS_OK);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    iree_async_event_native_deinitialize(&context_.kernel_event);
    iree_hal_amd_fabric_release(fabric_);
    for (const auto& memory : memories_) {
      EXPECT_FALSE(memory.allocated);
    }
  }

  iree_status_t Allocate(uint64_t byte_length = 12345) {
    expected_copy_length_ =
        resident_.host_access ? 0 : iree_align_uint64(byte_length, 4);
    iree_status_t status = iree_hal_amd_gpu_code_allocate(
        &context_, &resident_, byte_length, 64, iree_allocator_system(), &code_,
        &upload_, &contents_);
    if (iree_status_is_ok(status)) {
      std::memset(contents_.data, 0x71, contents_.data_length);
    }
    return status;
  }

  iree_status_t AllocateUserCache(amdf_status_t event_wait_status,
                                  bool complete_during_wait,
                                  iree_hal_amd_gpu_code_queue_t** out_cache) {
    *out_cache = nullptr;
    resident_.host_access = control_.host_access;
    context_.target_info.gfx_ip.major = 9;
    family_.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
    // Cache publication consumes opaque aligned device addresses. This fixture
    // does not execute uploads through these independent native addresses.
    api_.memory_query_address = [](amdf_memory_t* memory, uint32_t,
                                   amdf_memory_address_kind_t,
                                   uint64_t* address) {
      *address = iree_align_uint64(
          reinterpret_cast<uintptr_t>(GetMemory(memory).bytes.data()), 64);
      return AMDF_STATUS_OK;
    };
    IREE_RETURN_IF_ERROR(Allocate());

    native_cache_.event_wait_status = event_wait_status;
    native_cache_.complete_during_wait = complete_during_wait;
    context_.device = reinterpret_cast<amdf_device_t*>(&native_cache_);
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
    gpu_.event_wait = [](amdf_gpu_event_t* event, uint64_t timeout_ns) {
      auto* native = reinterpret_cast<NativeCache*>(event);
      native->wait_timeout_ns = timeout_ns;
      ++native->wait_count;
      if (native->complete_during_wait) {
        iree_atomic_store(
            reinterpret_cast<iree_atomic_int64_t*>(&native->signal->value), 0,
            iree_memory_order_release);
        native->complete_during_wait = false;
      }
      return native->event_wait_status;
    };
    gpu_.event_signal = [](amdf_gpu_event_t*) { return AMDF_STATUS_OK; };
    gpu_.event_destroy = [](amdf_gpu_event_t*) { return AMDF_STATUS_OK; };
    gpu_.user_queue_create = [](amdf_device_t* device,
                                const amdf_gpu_user_queue_create_info_t*,
                                amdf_user_queue_t** out_queue) {
      *out_queue = reinterpret_cast<amdf_user_queue_t*>(device);
      return AMDF_STATUS_OK;
    };
    api_.user_queue_query_info = [](amdf_user_queue_t*,
                                    amdf_user_queue_info_t*) {
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
          info->doorbell_address =
              reinterpret_cast<uintptr_t>(&native->doorbell);
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
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_code_queue_allocate(
        &context_, 0, AMDF_QUEUE_PUBLICATION_MODE_USER, &control_,
        iree_allocator_system(), out_cache));
    native_cache_.signal = reinterpret_cast<iree_hal_amd_gpu_aql_signal_t*>(
        memories_[2].bytes.data());
    return iree_ok_status();
  }

  iree_status_t Publish(iree_timeout_t timeout = iree_infinite_timeout()) {
    return iree_hal_amd_gpu_code_publish(&context_, queue_, timeout, &code_,
                                         &upload_);
  }

  void Stop() {
    amdf_status_t status = AMDF_STATUS_OK;
    EXPECT_TRUE(iree_hal_amd_gpu_code_queue_try_inactivate(queue_, &status));
    EXPECT_EQ(status, AMDF_STATUS_OK);
  }

  // Native API dependencies used by the loader and compute publisher.
  amdf_api_t api_ = {};
  // Native compute submission dependency.
  amdf_gpu_api_t gpu_ = {};
  // Queue control, final code, and optional USER publisher control.
  std::array<Memory, 3> memories_;
  // Number of native acquisition attempts, including rejected requests.
  uint32_t acquisitions_ = 0;
  // One-based acquisition to reject, or zero for success.
  uint32_t failed_acquisition_ = 0;
  // Submission rejection, before any native consumer exists.
  amdf_status_t submit_status_ = AMDF_STATUS_OK;
  // Wake registration failure after native submission acceptance.
  amdf_status_t notification_status_ = AMDF_STATUS_OK;
  // Checked native terminal failure, independent of retirement position.
  amdf_status_t retirement_status_ = AMDF_STATUS_OK;
  // Number of accepted native submissions.
  uint64_t submitted_ = 0;
  // Total bytes copied by the native command dependency.
  uint64_t copied_bytes_ = 0;
  // Required byte count before instruction-cache publication.
  uint64_t expected_copy_length_ = 0;
  // Number of final instruction-cache actions.
  uint32_t cache_actions_ = 0;
  // Number of successful native access-cessation operations.
  uint32_t stops_ = 0;
  // Explicit native retirement proof, independent of command-memory contents.
  uint64_t retired_ = 0;
  // False leaves accepted work pending until terminal closure.
  bool auto_retire_ = true;
  // Native notification enrollment has observed the accepted submission.
  std::counting_semaphore<> submitted_notification_{0};
  // Real allocation-domain registry and its native allocation owner.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Native ownership borrowed by the production loader and compute queue.
  iree_hal_amd_gpu_context_t context_ = {};
  // Captured native compute-family capability.
  amdf_queue_family_info_t family_ = {};
  // Host-mapped command allocation policy.
  iree_hal_amd_fabric_memory_options_t control_ = {};
  // Device-local final allocation policy, independently host-mappable.
  iree_hal_amd_fabric_memory_options_t resident_ = {};
  // Final code allocation owned by the loading caller.
  iree_hal_amd_fabric_memory_t code_ = {};
  // Temporary upload state consumed by publication or cancellation.
  iree_hal_amd_gpu_code_upload_t* upload_ = nullptr;
  // Logical initialization span returned by the loader.
  iree_byte_span_t contents_ = {};
  // Production compute bootstrap and instruction-cache publisher.
  iree_hal_amd_gpu_code_queue_t* queue_ = nullptr;
  // USER publication dependencies used to model native event outcomes.
  NativeCache native_cache_;
};

TEST_F(CodeTest, MappedCodePublishesInstructionsWithoutUpload) {
  resident_.host_access = control_.host_access;
  IREE_ASSERT_OK(Allocate());
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(contents_.data, code_.contents.data);
  EXPECT_EQ(contents_.data_length, 12345u);
  EXPECT_EQ(acquisitions_, 2u);
  EXPECT_EQ(memories_[1].flags, AMDF_MEMORY_FLAG_DEVICE_LOCAL);
  EXPECT_EQ(memories_[1].allocation_thread, std::this_thread::get_id());
  IREE_ASSERT_OK(Publish());
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(retired_, submitted_);
  EXPECT_EQ(cache_actions_, 1u);
  EXPECT_EQ(copied_bytes_, 0u);
}

TEST_F(CodeTest, UnmappedCodeUsesOneNativeAllocationAndBoundedChunks) {
  IREE_ASSERT_OK(Allocate(3 * 65536 + 1));
  ASSERT_NE(upload_, nullptr);
  EXPECT_EQ(acquisitions_, 2u);
  EXPECT_EQ(memories_[1].allocation_thread, std::this_thread::get_id());
  EXPECT_EQ(code_.mapping, nullptr);
  EXPECT_NE(contents_.data, memories_[1].bytes.data());
  for (size_t i = 0; i < contents_.data_length; ++i) {
    contents_.data[i] = static_cast<uint8_t>(i * 57 + 13);
  }
  IREE_ASSERT_OK(Publish());
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(copied_bytes_, expected_copy_length_);
  EXPECT_GT(submitted_, 3u);
  EXPECT_EQ(retired_, submitted_);
  EXPECT_EQ(cache_actions_, 1u);
  EXPECT_TRUE(memories_[1].allocated);
  for (size_t i = 0; i < memories_[1].bytes.size(); ++i) {
    const uint8_t expected = i < 3 * 65536 + 1
                                 ? static_cast<uint8_t>(i * 57 + 13)
                             : i < expected_copy_length_ ? 0
                                                         : 0xa5;
    ASSERT_EQ(memories_[1].bytes[i], expected) << "byte " << i;
  }
}

TEST_F(CodeTest, NativeAcquisitionFailureRollsBackHostImage) {
  failed_acquisition_ = 2;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Allocate());
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(contents_.data, nullptr);
  EXPECT_FALSE(memories_[1].allocated);
}

TEST_F(CodeTest, RejectedUploadRetainsCallerOwnership) {
  IREE_ASSERT_OK(Allocate());
  submit_status_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Publish());
  EXPECT_EQ(upload_, nullptr);
  EXPECT_NE(code_.handle, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_EQ(submitted_, 0u);
  Stop();
}

TEST_F(CodeTest, AcceptedFailureKeepsTargetUntilNativeStop) {
  IREE_ASSERT_OK(Allocate());
  auto_retire_ = false;
  notification_status_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Publish());
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_EQ(code_.mapping, nullptr);
  EXPECT_EQ(code_.contents.data, nullptr);
  EXPECT_EQ(code_.info.byte_length, 12352u);
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(retired_, 0u);
  EXPECT_TRUE(memories_[0].allocated);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_EQ(iree_hal_amd_gpu_code_queue_free(queue_), notification_status_);
  Stop();
  EXPECT_EQ(stops_, 1u);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_EQ(iree_hal_amd_gpu_code_queue_free(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_FALSE(memories_[1].allocated);
}

TEST_F(CodeTest, PublicationDeadlineKeepsTargetUntilNativeStop) {
  IREE_ASSERT_OK(Allocate());
  auto_retire_ = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        Publish(iree_immediate_timeout()));
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_EQ(code_.mapping, nullptr);
  EXPECT_EQ(code_.contents.data, nullptr);
  EXPECT_EQ(code_.info.byte_length, 12352u);
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(retired_, 0u);
  EXPECT_TRUE(memories_[1].allocated);
  Stop();
}

TEST_F(CodeTest, CheckedNativeFailureOverridesRetirementPosition) {
  IREE_ASSERT_OK(Allocate());
  retirement_status_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Publish());
  EXPECT_EQ(retired_, submitted_);
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  Stop();
}

TEST_F(CodeTest, ClosingWakesPendingLoaderBeforeNativeStop) {
  IREE_ASSERT_OK(Allocate());
  auto_retire_ = false;
  std::thread loader(
      [&] { IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Publish()); });
  submitted_notification_.acquire();
  EXPECT_FALSE(iree_hal_amd_gpu_code_queue_is_host_idle(queue_));
  amdf_status_t native_status = AMDF_STATUS_OK;
  EXPECT_FALSE(
      iree_hal_amd_gpu_code_queue_try_inactivate(queue_, &native_status));
  EXPECT_EQ(stops_, 0u);
  const amdf_status_t error =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(iree_hal_amd_gpu_code_queue_close(queue_, error), AMDF_STATUS_OK);
  loader.join();
  EXPECT_EQ(upload_, nullptr);
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_TRUE(iree_hal_amd_gpu_code_queue_is_host_idle(queue_));
  Stop();
  EXPECT_EQ(stops_, 1u);
}

TEST_F(CodeTest, FailedCachePublicationKeepsItsTargetUntilNativeStop) {
  iree_hal_amd_gpu_code_queue_t* cache = nullptr;
  IREE_ASSERT_OK(AllocateUserCache(
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST), false, &cache));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_code_queue_publish(
                            cache, iree_infinite_timeout(), &code_,
                            iree_const_byte_span_empty()));
  EXPECT_EQ(code_.handle, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  amdf_status_t native_status = AMDF_STATUS_OK;
  EXPECT_TRUE(
      iree_hal_amd_gpu_code_queue_try_inactivate(cache, &native_status));
  EXPECT_EQ(native_status, AMDF_STATUS_OK);
  EXPECT_EQ(native_cache_.stops, 1u);
  EXPECT_TRUE(memories_[1].allocated);
  EXPECT_EQ(iree_hal_amd_gpu_code_queue_free(cache), AMDF_STATUS_OK);
  EXPECT_FALSE(memories_[1].allocated);
  EXPECT_FALSE(memories_[2].allocated);
}

TEST_F(CodeTest, UserPublicationRechecksAfterNativeEventExpiry) {
  iree_hal_amd_gpu_code_queue_t* cache = nullptr;
  IREE_ASSERT_OK(AllocateUserCache(
      amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED), true, &cache));
  IREE_ASSERT_OK(iree_hal_amd_gpu_code_queue_publish(
      cache, iree_infinite_timeout(), &code_, iree_const_byte_span_empty()));
  EXPECT_EQ(native_cache_.wait_count, 1u);
  EXPECT_EQ(native_cache_.write_index, 1u);
  EXPECT_GT(native_cache_.wait_timeout_ns, 0u);
  EXPECT_LE(native_cache_.wait_timeout_ns, 5000000u);
  EXPECT_NE(code_.handle, nullptr);
  EXPECT_TRUE(memories_[1].allocated);
  amdf_status_t native_status = AMDF_STATUS_OK;
  EXPECT_TRUE(
      iree_hal_amd_gpu_code_queue_try_inactivate(cache, &native_status));
  EXPECT_EQ(native_status, AMDF_STATUS_OK);
  EXPECT_EQ(native_cache_.stops, 1u);
  EXPECT_EQ(iree_hal_amd_gpu_code_queue_free(cache), AMDF_STATUS_OK);
  EXPECT_FALSE(memories_[2].allocated);
}

}  // namespace
