// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/queue.h"

#include <linux/sync_file.h>
#include <sys/eventfd.h>

#include <array>
#include <cstdarg>
#include <map>
#include <set>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/media/drm/command.h"
#include "libamdf/src/platform/linux/file.h"

namespace {
struct NativeQueue {
  // Live native sync objects, owned by queue creation/destruction.
  std::set<uint32_t> handles;
  // Next native sync object identity.
  uint32_t next_handle = 1;
  // Original job point retained by each reusable binary sync object.
  std::map<uint32_t, uint64_t> originals;
  // Real host descriptors standing in for kernel-exported sync files.
  std::map<int, uint32_t> files;
  // Greatest signaled timeline point; does not imply successful job execution.
  uint64_t completed = 0;
  // Original job reporting a terminal execution error, or zero.
  uint64_t failed_point = 0;
  // Accepted CS count, independent of original-fence observation.
  uint32_t submissions = 0;
  // Injected CS rejection before native acceptance.
  int submit_error = 0;
  // Injected temporary original-fence export failure.
  int export_error = 0;
  // Number of native observation calls.
  uint32_t observations = 0;
  // Library allocation count, used to qualify publication's allocation
  // contract.
  uint32_t allocations = 0;
};
thread_local NativeQueue* native_queue = nullptr;
}  // namespace

extern "C" int __real_ioctl(int descriptor, unsigned long request, ...);
extern "C" int __real_close(int descriptor);
extern "C" int __wrap_close(int descriptor) {
  if (native_queue != nullptr) {
    native_queue->files.erase(descriptor);
  }
  return __real_close(descriptor);
}
extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_queue == nullptr) {
    return __real_ioctl(descriptor, request, argument);
  }
  auto& native = *native_queue;
  if (request == SYNC_IOC_FILE_INFO) {
    const auto found = native.files.find(descriptor);
    EXPECT_NE(found, native.files.end());
    if (found == native.files.end()) {
      errno = EBADF;
      return -1;
    }
    const uint64_t point = native.originals.at(found->second);
    auto* info = static_cast<sync_file_info*>(argument);
    info->status = point > native.completed       ? 0
                   : point == native.failed_point ? -EIO
                                                  : 1;
    return 0;
  }
  EXPECT_EQ(descriptor, 17);
  switch (request) {
    case DRM_IOCTL_AMDGPU_CTX: {
      auto* context = static_cast<drm_amdgpu_ctx*>(argument);
      if (context->in.op == AMDGPU_CTX_OP_ALLOC_CTX) {
        context->out.alloc.ctx_id = 1;
      } else {
        EXPECT_EQ(context->in.op, AMDGPU_CTX_OP_FREE_CTX);
      }
      return 0;
    }
    case DRM_IOCTL_SYNCOBJ_CREATE: {
      auto* create = static_cast<drm_syncobj_create*>(argument);
      create->handle = native.next_handle++;
      native.handles.insert(create->handle);
      return 0;
    }
    case DRM_IOCTL_SYNCOBJ_DESTROY: {
      const auto* destroy = static_cast<drm_syncobj_destroy*>(argument);
      EXPECT_EQ(native.handles.erase(destroy->handle), 1u);
      return 0;
    }
    case DRM_IOCTL_SYNCOBJ_EVENTFD:
      errno = ENOTTY;
      return -1;
    case DRM_IOCTL_AMDGPU_CS: {
      if (native.submit_error != 0) {
        errno = native.submit_error;
        return -1;
      }
      const auto* cs = static_cast<drm_amdgpu_cs*>(argument);
      EXPECT_EQ(cs->in.num_chunks, 3u);
      const auto* pointers = reinterpret_cast<const uint64_t*>(cs->in.chunks);
      const auto* chunk =
          reinterpret_cast<const drm_amdgpu_cs_chunk*>(pointers[2]);
      EXPECT_EQ(chunk->chunk_id, AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL);
      EXPECT_EQ(chunk->length_dw, 2 * sizeof(drm_amdgpu_cs_chunk_syncobj) / 4);
      const auto* signals =
          reinterpret_cast<const drm_amdgpu_cs_chunk_syncobj*>(
              chunk->chunk_data);
      EXPECT_EQ(signals[0].point, 0u);
      EXPECT_EQ(signals[1].point, native.submissions + 1);
      native.originals[signals[0].handle] = signals[1].point;
      ++native.submissions;
      return 0;
    }
    case DRM_IOCTL_SYNCOBJ_QUERY: {
      ++native.observations;
      auto* query = static_cast<drm_syncobj_timeline_array*>(argument);
      *reinterpret_cast<uint64_t*>(query->points) = native.completed;
      return 0;
    }
    case DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD: {
      ++native.observations;
      if (native.export_error != 0) {
        errno = native.export_error;
        return -1;
      }
      auto* export_info = static_cast<drm_syncobj_handle*>(argument);
      EXPECT_EQ(export_info->flags,
                DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE);
      export_info->fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
      if (export_info->fd < 0) {
        return -1;
      }
      native.files[export_info->fd] = export_info->handle;
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected queue ioctl " << request;
      errno = ENOTTY;
      return -1;
  }
}

namespace {
void QueryDevice(const amdf_device_t*, amdf_device_info_t* info) {
  info->reset_epoch = 1;
}
constexpr amdf_device_vtable_t kDevice = {.query_info = QueryDevice};

class MediaQueueTest : public ::testing::Test {
 protected:
  virtual uint32_t EngineType() { return AMDGPU_HW_IP_VCN_JPEG; }
  void SetUp() override {
    native_queue = &native_;
    device_.base.base.vtable = &kDevice;
    device_.base.base.host_allocator = amdf_allocator_system();
    device_.base.base.host_allocator.user_data = &native_;
    device_.base.base.host_allocator.allocate = [](void* data, uint64_t length,
                                                   uint64_t alignment) {
      ++static_cast<NativeQueue*>(data)->allocations;
      const auto allocator = amdf_allocator_system();
      return allocator.allocate(allocator.user_data, length, alignment);
    };
    amdf_child_tracker_initialize(&device_.base.base.children);
    device_.descriptor = 17;
    device_.engines[0].ip_type = EngineType();
    device_.engines[0].information.available_rings = 1;
    const amdf_media_kernel_queue_create_info_t create = {
        .maximum_pending_submission_count = 4,
    };
    ASSERT_EQ(amdf_media_drm_queue_create(&device_.base.base, &create, &queue_),
              AMDF_STATUS_OK);
    storage_.address = 4096;
    for (auto& command : commands_) {
      command.base.queue = queue_;
      command.storage = &storage_;
      command.byte_length = 64;
    }
  }
  void TearDown() override {
    if (queue_ != nullptr) {
      if (native_.failed_point == 0) {
        native_.completed = native_.submissions;
        auto status = Snapshot();
        ASSERT_EQ(amdf_kernel_queue_refresh_status(queue_, &status),
                  AMDF_STATUS_OK);
        ASSERT_EQ(amdf_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
        EXPECT_TRUE(native_.handles.empty());
      } else {
        // The model has no native work or physical resources. Dispose its
        // inert ownership domain after proving production destruction refuses;
        // no failed point is relabeled as retired to make cleanup pass.
        EXPECT_EQ(amdf_kernel_queue_destroy(queue_),
                  amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
        amdf_kernel_queue_deinitialize(queue_);
        amdf_free(device_.base.base.host_allocator, queue_);
      }
    }
    EXPECT_TRUE(native_.files.empty());
    native_queue = nullptr;
  }
  amdf_kernel_queue_status_t Snapshot() {
    return {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(amdf_kernel_queue_status_t),
    };
  }
  amdf_status_t Submit(uint32_t slot, uint64_t* accepted) {
    const amdf_media_kernel_queue_submission_info_t submit = {
        .command = &commands_[slot].base,
    };
    return amdf_media_drm_queue_submit(queue_, &submit, accepted);
  }
  // Actual native queue implementation with a modeled DRM syscall boundary.
  amdf_kernel_queue_t* queue_ = nullptr;
  // Device lifetime and immutable native ring selection.
  amdf_media_drm_device_t device_ = {};
  // Inert native recording backing consumed only by the modeled submission.
  amdf_media_drm_buffer_t storage_ = {};
  // Explicit application recording slots, without native command allocation.
  std::array<amdf_media_drm_command_t, 4> commands_ = {};
  // Native dependency state and counters, separate from production queue state.
  NativeQueue native_;
};

TEST_F(MediaQueueTest, RejectedPublicationPreservesPointsAndAllocatesNothing) {
  const uint32_t allocations = native_.allocations;
  uint64_t accepted = 0;
  native_.submit_error = EBUSY;
  EXPECT_EQ(Submit(0, &accepted), amdf_linux_error(EBUSY));
  EXPECT_EQ(accepted, 0u);
  native_.submit_error = 0;
  EXPECT_EQ(Submit(0, &accepted), AMDF_STATUS_OK);
  EXPECT_EQ(accepted, 1u);
  auto status = Snapshot();
  EXPECT_EQ(amdf_kernel_queue_query_status(queue_, &status), AMDF_STATUS_OK);
  EXPECT_EQ(status.retired_submission, 0u);
  EXPECT_EQ(native_.observations, 0u);
  EXPECT_EQ(native_.allocations, allocations);
}

TEST_F(MediaQueueTest, ObservationFailureDoesNotRetireOrPoisonAcceptedWork) {
  uint64_t accepted = 0;
  ASSERT_EQ(Submit(0, &accepted), AMDF_STATUS_OK);
  native_.completed = 1;
  native_.export_error = EMFILE;
  auto status = Snapshot();
  status.retired_submission = 99;
  EXPECT_EQ(amdf_kernel_queue_refresh_status(queue_, &status),
            amdf_linux_error(EMFILE));
  EXPECT_EQ(status.retired_submission, 99u);
  EXPECT_EQ(amdf_kernel_queue_query_status(queue_, &status), AMDF_STATUS_OK);
  EXPECT_EQ(status.retired_submission, 0u);
  EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
  native_.export_error = 0;
  EXPECT_EQ(amdf_kernel_queue_refresh_status(queue_, &status), AMDF_STATUS_OK);
  EXPECT_EQ(status.retired_submission, 1u);
}

TEST_F(MediaQueueTest, SignaledTimelineCannotHideAnOriginalJobFailure) {
  for (uint32_t i = 0; i < 3; ++i) {
    uint64_t accepted = 0;
    ASSERT_EQ(Submit(i, &accepted), AMDF_STATUS_OK);
  }
  native_.completed = 3;
  native_.failed_point = 2;
  auto status = Snapshot();
  EXPECT_EQ(amdf_kernel_queue_refresh_status(queue_, &status), AMDF_STATUS_OK);
  EXPECT_EQ(status.retired_submission, 1u);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
  EXPECT_EQ(status.terminal_status, amdf_linux_error(EIO));
  uint64_t accepted = 0;
  EXPECT_EQ(Submit(3, &accepted), amdf_linux_error(EIO));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(native_.submissions, 3u);
}

class MediaVideoQueueTest : public MediaQueueTest {
 protected:
  uint32_t EngineType() override { return AMDGPU_HW_IP_VCN_ENC; }
};

TEST_F(MediaVideoQueueTest,
       RejectedControlPublicationPreservesSessionLifetime) {
  amdf_media_drm_session_t session = {};
  session.base.queue = queue_;
  commands_[0].session = &session;
  commands_[0].operation = AMDF_MEDIA_COMMAND_OPERATION_BEGIN;
  commands_[1].session = &session;
  commands_[1].operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE;
  commands_[2].session = &session;
  commands_[2].operation = AMDF_MEDIA_COMMAND_OPERATION_END;
  uint64_t accepted = 0;
  native_.submit_error = EBUSY;
  EXPECT_EQ(Submit(0, &accepted), amdf_linux_error(EBUSY));
  EXPECT_EQ(accepted, 0u);
  native_.submit_error = 0;
  EXPECT_EQ(Submit(1, &accepted),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  EXPECT_EQ(native_.submissions, 0u);
  ASSERT_EQ(Submit(0, &accepted), AMDF_STATUS_OK);
  EXPECT_EQ(accepted, 1u);
  accepted = 0;
  native_.submit_error = EBUSY;
  EXPECT_EQ(Submit(2, &accepted), amdf_linux_error(EBUSY));
  EXPECT_EQ(accepted, 0u);
  native_.submit_error = 0;
  ASSERT_EQ(Submit(1, &accepted), AMDF_STATUS_OK);
  EXPECT_EQ(accepted, 2u);
  ASSERT_EQ(Submit(2, &accepted), AMDF_STATUS_OK);
  EXPECT_EQ(accepted, 3u);
  native_.completed = 3;
  auto status = Snapshot();
  ASSERT_EQ(amdf_kernel_queue_refresh_status(queue_, &status), AMDF_STATUS_OK);
  accepted = 0;
  EXPECT_EQ(Submit(1, &accepted),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(native_.submissions, 3u);
}
}  // namespace
